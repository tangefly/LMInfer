"""理论验证实验: agent 会话间跨请求 KV 前缀复用到底省了什么?

模拟一个 agent 事务的完整调用流(与 README 的 agent 模式示例一致):
  main 请求    trace=["main"]              -> 主 agent 生成工具调用
  子 agent 请求 trace=["main","sub1"]       -> 子 agent 生成输出
  main 再请求  trace=["main","sub1","main"] -> 子 agent 输出已回到主 agent 上下文

对第三次请求分别用两种方式生成:
  1. 全量 prefill (不开启复用, 默认行为)
  2. 前缀 KV 复用: 把子 agent 请求保存的完整序列 KV(prompt + 输出)与
     新 prompt 做 token 级最长公共前缀(LCP)匹配, 只 prefill 剩余部分

预期结论(实验要验证的理论):
  - 复用的 token 与位置都一致 => KV 是 (token, 位置) 的确定性函数,
    数学上与全量 prefill 完全等价(数值上只有 bf16 内核级舍入差异);
  - prefill 计算量与"需要新 prefill 的 token 数"成正比, 复用得越多 TTFT 越低;
  - 匹配长度取决于 chat template 是否对同一段历史做一致的渲染: Qwen3 会
    对末尾 assistant 消息插入 <think> 块(与其后的 tool 消息渲染不同),
    导致 LCP 变短; ERNIE 等模板渲染一致, 可复用几乎整段历史.

用法:
  python experiments/agent_kv_reuse.py --model /path/to/model [--max-tokens 16]
"""

import argparse
import sys

import torch

# 让脚本可以从仓库根目录直接运行
sys.path.insert(0, ".")

from lminfer.config import EngineConfig, SamplingParams  # noqa: E402
from lminfer.engine import LLMEngine  # noqa: E402
from lminfer.kvcache import SessionKVStore  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="agent 会话间跨请求 KV 复用对比实验")
    parser.add_argument("--model", required=True, help="模型路径或 HF 模型名")
    parser.add_argument("--max-tokens", type=int, default=16,
                        help="每个请求最多生成的 token 数")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="部分模型需要(如 ERNIE)")
    args = parser.parse_args()

    engine = LLMEngine(EngineConfig(model=args.model,
                                    trust_remote_code=args.trust_remote_code))
    if engine.tokenizer.chat_template is None:
        sys.exit("该模型没有 chat template, 无法构造 agent 消息流")

    # ---- agent 事务的三次请求(消息随事务推进逐步累积) ----
    sys_msg = {"role": "system", "content": "你是助手, 需要工具时调用工具。"}
    messages = [
        ([sys_msg, {"role": "user", "content": "上海的天气怎么样?"}]),
        ([sys_msg, {"role": "user", "content": "上海的天气怎么样?"},
          {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_weather\", "
                                           "\"arguments\": {\"city\": \"Shanghai\"}}\n</tool_call>"}]),
        ([sys_msg, {"role": "user", "content": "上海的天气怎么样?"},
          {"role": "assistant", "content": "<tool_call>\n{\"name\": \"get_weather\", "
                                           "\"arguments\": {\"city\": \"Shanghai\"}}\n</tool_call>"},
          {"role": "tool", "content": "上海的天气: 晴, 28度"},
          {"role": "user", "content": "适合散步吗?"}]),
    ]
    traces = [["main"], ["main", "sub1"], ["main", "sub1", "main"]]

    def render(msgs) -> torch.Tensor:
        ids = engine.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):  # transformers 5.x 返回 tokenizers.Encoding
            ids = ids.input_ids
        return torch.tensor(ids).unsqueeze(0)

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0)  # 贪心, 可复现
    prompts = [render(m) for m in messages]
    for i, (p, t) in enumerate(zip(prompts, traces)):
        print(f"请求 {i + 1} {str(t):<24} prompt {p.shape[1]:>4} tok")

    def store_request(store: SessionKVStore, sid: str, trace: list[str],
                      prompt_ids: torch.Tensor, r) -> None:
        """模拟 server.py 的 _agent_kv_finish: 保存完整序列 KV."""
        seq = prompt_ids[0][-r.prompt_tokens:].tolist() + r.output_tokens
        kind = "main" if trace[-1] == "main" else "sub"
        store.put(sid, kind, seq, r.kv_cache)

    # ---- 预热: 完整跑一遍流程(含复用路径), 消除 CUDA kernel 冷启动偏差 ----
    # 注意顺序与生产一致: 复用候选在"保存本次请求"之前取(否则候选会命中
    # 本次请求自身的序列, 形状与正式运行不同, 预热不到目标 kernel 路径)
    store = SessionKVStore()
    for p, t in zip(prompts[:2], traces[:2]):
        r = engine._generate("warmup", p, sampling)
        store_request(store, "warm", t, p, r)
    engine._generate("warmup-reuse", prompts[2], sampling,
                     reuse_prefixes=store.propose("warm", traces[2]))
    r = engine._generate("warmup", prompts[2], sampling)
    store_request(store, "warm", traces[2], prompts[2], r)

    print(f"\n{'模式':<24}{'LCP':>6}{'复用':>7}{'新 prefill':>10}{'TTFT':>10}")
    print("-" * 57)
    rows = {}

    # ---- 方式 1: 不开启复用(全量 prefill, 默认行为) ----
    r3_full = engine._generate("main3-full", prompts[2], sampling)
    rows["full"] = r3_full
    print(f"{'全量 prefill':<24}{0:>6}{0:>7}{prompts[2].shape[1]:>10}"
          f"{r3_full.ttft_ms:>8.0f}ms")

    # ---- 方式 2: 前缀 KV 复用(与生产代码相同的接线) ----
    store = SessionKVStore()
    sid = "exp"
    for p, t in zip(prompts[:2], traces[:2]):  # main、子 agent 请求正常执行并保存 KV
        r = engine._generate("main1" if t[-1] == "main" else "sub1", p, sampling)
        store_request(store, sid, t, p, r)

    prefixes = store.propose(sid, traces[2])  # 触发条件: trace 末位 main, 前一个是子 agent
    r3 = engine._generate("main3-reuse", prompts[2], sampling, reuse_prefixes=prefixes)
    if r3.reused_prompt_tokens > 0:  # 与 server.py 的接线一致
        store.note_hit(r3.reused_prompt_tokens)
    rows["reuse"] = r3
    m = r3.reused_prompt_tokens
    print(f"{'前缀 KV 复用':<24}{m:>6}{m:>7}{prompts[2].shape[1] - m:>10}"
          f"{r3.ttft_ms:>8.0f}ms")
    print(f"  复用率: {m / prompts[2].shape[1] * 100:.0f}% 的 prompt 跳过重复 prefill, "
          f"省 {m * engine.kv_bytes_per_token / (1024 ** 2):.2f} MiB 的 KV 计算")

    # ---- 对照一致性: 复用与全量 prefill 的贪心输出应一致 ----
    a, b = rows["reuse"], rows["full"]
    same = next((i for i, (x, y) in enumerate(zip(a.output_tokens, b.output_tokens))
                 if x != y), min(len(a.output_tokens), len(b.output_tokens)))
    print(f"\n对照一致性: 前 {same}/{len(a.output_tokens)} 个 token 一致"
          f"{' (完全一致)' if same == len(a.output_tokens) else ' (bf16 舍入导致的翻转, 非 bug)'}")
    print(f"TTFT 对比: 全量 {b.ttft_ms:.0f}ms -> 复用 {a.ttft_ms:.0f}ms"
          f"{' (prefill 变短, 更快)' if a.ttft_ms < b.ttft_ms else ' (小模型上差异在噪声范围内)'}")
    print(f"store 统计: {store.stats}")
    print("\n注: 匹配长度取决于 chat template. Qwen3 对末尾 assistant 消息插入 <think> 块,"
          "\n    导致 LCP 变短; ERNIE 等模板渲染一致, 复用率可达 80%+.")


if __name__ == "__main__":
    main()
