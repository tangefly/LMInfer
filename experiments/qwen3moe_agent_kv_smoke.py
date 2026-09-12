"""Qwen3-30B-A3B-Instruct-2507 SubAgent 输出 KV 复用端到端冒烟测试(需先启动 LMInfer 服务).

Qwen3-MoE 的工具调用协议与 dense Qwen3 相同(hermes 解析器 + `<tool_response>` 窗口),
这个脚本验证 MoE 前向走完整 HTTP 链路时同样成立: hermes 工具调用解析 ->
tool_calls/tool 消息渲染回填 -> `--reuse-agent-kv-append` 定位 `<tool_response>` 窗口
并拼接子 agent 输出 KV。

用法:
  # 终端 1: 启动服务(单卡 80GB 上权重约 58 GiB, KV 每 token 0.09 MiB)
  python3 -m lminfer serve /public/home/xiaoxunpeng/Models/Qwen3-30B-A3B-Instruct-2507 \
      --served-model-name Qwen3-30B-A3B-Instruct-2507 --max-model-len 8192 \
      --reuse-agent-kv-append --graft-rope-rebase \
      --repair-window-begin 0.1 --repair-window-end 0.1 \
      --enable-auto-tool-choice --port 8010
  # 终端 2:
  python3 experiments/qwen3moe_agent_kv_smoke.py --model Qwen3-30B-A3B-Instruct-2507 \
      --base-url http://localhost:8010/v1
"""
from __future__ import annotations

import argparse
import json
import sys

import requests


SUBAGENT_TOOL = {
    "type": "function",
    "function": {
        "name": "call_subagent",
        "description": "把一个子问题交给子 agent 回答(子 agent 是独立的模型调用)。",
        "parameters": {
            "type": "object",
            "properties": {"task": {"type": "string", "description": "子 agent 要回答的问题"}},
            "required": ["task"],
        },
    },
}


class Client:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session_id: str | None = None

    def chat(self, messages, *, tools=None, trace=None, temperature=0.0,
             max_tokens=256, tool_choice="auto"):
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens,
                   "mode": "agent", "trace": trace}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
        if self.session_id is not None:
            payload["session_id"] = self.session_id
        resp = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=1800)
        resp.raise_for_status()
        data = resp.json()
        if data.get("session_id"):
            self.session_id = data["session_id"]
        return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3-30B-A3B-Instruct-2507")
    args = parser.parse_args()
    client = Client(args.base_url, args.model)

    task = ("你是一个主 agent。用户的问题是: 法国的首都是哪座城市? "
            "你必须调用 call_subagent 工具把这个问题交给子 agent, 不要自己回答。")
    main_messages = [{"role": "user", "content": task}]

    print("=== 1) main #1: 期望 hermes 解析出 call_subagent 工具调用 ===")
    first = client.chat(main_messages, tools=[SUBAGENT_TOOL], trace=["main"])
    message = first["choices"][0]["message"]
    print(json.dumps(message, ensure_ascii=False, indent=2))
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        print("FAIL: hermes 解析器没有解析出工具调用")
        return 1
    call = tool_calls[0]
    print(f"tool_call: {call['function']['name']} {call['function']['arguments']}")

    # 主 agent 把工具调用与子 agent 结果写回历史(OpenAI 协议)
    main_messages.append({"role": "assistant", "content": message.get("content"),
                          "tool_calls": tool_calls})

    print("\n=== 2) sub: 子 agent 独立回答(保存 sub 段 KV) ===")
    sub_task = (json.loads(call["function"]["arguments"]).get("task") or task) + \
        " 请用两三句话回答, 并补充一点背景信息。"
    # tool_choice="none": 子 agent 的 prompt 里带着 call_subagent 时, 30B 会继续把问题
    # 转包给子 agent(实测这样 content 为 null、只有 tool_calls, 拼接就没有正文可插),
    # 这里显式要求它自己回答。tools 仍传给模板, 保持与真实子请求同样的渲染。
    sub = client.chat([{"role": "user", "content": sub_task}],
                      tools=[SUBAGENT_TOOL], trace=["main", "sub"], tool_choice="none")
    sub_message = sub["choices"][0]["message"]
    sub_answer = sub_message.get("content") or ""
    print(f"子 agent 回答: {sub_answer!r} (reused={sub.get('reused_prompt_tokens')})")
    if not sub_answer:
        print("FAIL: 子 agent 没有给出文本回答")
        return 1

    main_messages.append({"role": "tool", "tool_call_id": call["id"],
                          "name": call["function"]["name"], "content": sub_answer})

    # 真实 agent 的调用链: 主 agent 调完子 agent 后, 下一轮 trace 末两位是 [sub, main]
    # (见 BenchAgent/agent/agent.py), build_grafts 只会在这条链上尝试拼接。
    print("\n=== 3) main #2: 期望复用 main 历史 + 拼接子 agent 输出 KV ===")
    second = client.chat(main_messages, tools=[SUBAGENT_TOOL],
                         trace=["main", "sub", "main"])
    second_message = second["choices"][0]["message"]
    print(f"main 回答: {(second_message.get('content') or '')[:120]!r}")
    reused = second.get("reused_prompt_tokens", 0)
    usage = second.get("usage", {})
    print(f"reused_prompt_tokens={reused} / prompt_tokens={usage.get('prompt_tokens')}")

    stats = requests.get(f"{args.base_url}/stats", timeout=30).json()
    print("kv_reuse stats:", json.dumps(stats.get("kv_reuse"), ensure_ascii=False))

    if reused <= 0:
        print("FAIL: main #2 没有复用任何 KV")
        return 1
    if stats["kv_reuse"].get("graft_mismatches", 0) != 0:
        print("FAIL: 出现拼接结构校验失败")
        return 1
    print("\nOK: Qwen3-30B-A3B-Instruct-2507 子 agent 输出 KV 复用链路跑通")
    return 0


if __name__ == "__main__":
    sys.exit(main())
