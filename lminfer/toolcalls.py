"""Qwen / Hermes 风格工具调用解析(对应 vLLM 的 --tool-call-parser qwen / hermes).

Qwen2.5/Qwen3 模型在回复中以如下格式输出工具调用(与 Hermes 格式相同,
vLLM 跑 Qwen3 用的 hermes parser 解析的就是这个格式):

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Shanghai"}}
    </tool_call>

<tool_call> 等是模型的特殊 token, 生成时需要用 skip_special_tokens=False 解码,
解析成功后转成 OpenAI 的 tool_calls 字段(与 /v1/chat/completions 响应格式一致).
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Tuple

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
# 两个标记都是特殊 token, 解码后原样出现; 但保险起见仍按子串匹配
TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
THINK_START, THINK_END = "<think>", "</think>"


def _match_json_value(text: str, start: int) -> str | None:
    """从 start 处取一个完整 JSON 值的原始子串(对象/数组/字符串/标量)."""
    if start >= len(text):
        return None
    c = text[start]
    if c in "{[":
        close = "}" if c == "{" else "]"
        depth, in_str = 0, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if ch == "\\":
                    continue
                if ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == c:
                depth += 1
            elif ch == close:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        return None
    if c == '"':
        i = start + 1
        while i < len(text):
            if text[i] == "\\":
                i += 2
                continue
            if text[i] == '"':
                return text[start:i + 1]
            i += 1
        return None
    # 标量(数字/true/false/null): 到逗号或右括号为止
    i = start
    while i < len(text) and text[i] not in ",}":
        i += 1
    return text[start:i].strip() or None


def _extract_raw_arguments(block: str) -> str | None:
    """提取块内顶层 "arguments" 键的原始 JSON 值子串(round-trip 保真用).

    模型原始生成的 arguments(如 {"city":"Shanghai"} 紧凑格式)经 json.loads +
    json.dumps 会被归一化补空格; 客户端把 assistant 消息原样回传时, 模板按
    is string 分支原样渲染原始子串, 才能与生成流逐位一致, KV 前缀复用才
    不会断在 <tool_call> 块。提取失败返回 None(调用方回退重序列化).
    """
    depth = 0      # 0 = 根对象外, 1 = 根对象内
    in_str = False
    i, n = 0, len(block)
    while i < n:
        c = block[i]
        if in_str:
            i += 2 if (c == "\\" and i + 1 < n) else 1
            if c == '"':
                in_str = False
            continue
        if c == '"':
            if depth == 1:
                # 根对象内的字符串: 先判断是键还是值(键后紧跟冒号)
                j = i + 1
                while j < n and block[j] != '"':
                    j += 2 if block[j] == "\\" else 1
                if j < n:
                    k = j + 1
                    while k < n and block[k] in " \t\r\n":
                        k += 1
                    if k < n and block[k] == ":" and block[i + 1:j] == "arguments":
                        k += 1
                        while k < n and block[k] in " \t\r\n":
                            k += 1
                        return _match_json_value(block, k) if k < n else None
                i = j + 1
            else:
                in_str = True
                i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    return None


def _parse_call_block(block: str) -> Dict[str, Any] | None:
    """把一个 <tool_call> 块内的 JSON 解析成 OpenAI 工具调用, 失败返回 None."""
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return None
    name = data.get("name") if isinstance(data, dict) else None
    if not isinstance(name, str) or not name:
        return None
    args = data.get("arguments") or {}
    if isinstance(args, str):  # 容忍 arguments 是 JSON 字符串
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    # 优先返回原始 JSON 子串(保真 round-trip), 提取失败才回退重序列化
    raw_args = _extract_raw_arguments(block)
    if raw_args is not None:
        try:
            json.loads(raw_args)
        except json.JSONDecodeError:
            raw_args = None
    if raw_args is None:
        raw_args = json.dumps(args, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name, "arguments": raw_args},
    }


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """扫描整段生成文本, 返回 OpenAI 格式的 tool_calls 列表(解析失败的块跳过)."""
    calls: List[Dict[str, Any]] = []
    for block in TOOL_CALL_BLOCK.findall(text):
        call = _parse_call_block(block)
        if call is not None:
            calls.append(call)
    return calls


def clean_content(text: str) -> str:
    """去掉 <tool_call> 块与 <think>/</think> 标记, 得到应返回的 content 文本."""
    text = TOOL_CALL_BLOCK.sub("", text)
    text = text.replace(THINK_START, "").replace(THINK_END, "")
    return text.strip()


class ToolCallStreamSplitter:
    """把逐 token 文本流切成 content / tool_call 两类事件(流式响应用).

    行为:
      - <tool_call>...</tool_call> 整块不进入 content, 块内 JSON 解析成工具调用;
      - <think>/</think> 标记直接丢弃, 思考内容仍作为 content 透传(与非流式一致);
      - 流结束时未闭合的块按普通文本返回(模型没写完就当文本);
      - 标记即使被拆成多个 chunk 到达也能正确识别(每个状态只保留可能
        是不完整标记的尾部, 其余内容立即输出).

    用法: events = splitter.push(chunk) 返回事件列表, 事件为
      ("content", str) 或 ("tool_call", dict); 流结束后取 splitter.flush().
    """

    def __init__(self) -> None:
        self._buf = ""
        self._state = "normal"  # normal / in_think / in_tool_call

    def _emit_content(self, events: List[Tuple[str, Any]], text: str) -> None:
        # 正常状态下不会出现 <think> 开标记(它已进入 in_think 状态);
        # 这里只清理游离的 </think> 闭标记(模型输出不配对时的兜底)
        text = text.replace(THINK_END, "")
        if text:
            events.append(("content", text))

    def _switch(self, events: List[Tuple[str, Any]], marker: str,
                state: str) -> None:
        start = self._buf.find(marker)
        self._emit_content(events, self._buf[:start])
        self._buf = self._buf[start + len(marker):]
        self._state = state

    def push(self, chunk: str) -> List[Tuple[str, Any]]:
        events: List[Tuple[str, Any]] = []
        self._buf += chunk
        while True:
            if self._state == "in_tool_call":
                end = self._buf.find(TOOL_CALL_END)
                if end == -1:
                    break  # 未闭合: 继续攒缓冲, 不提前输出
                call = _parse_call_block(self._buf[:end])
                if call is not None:
                    events.append(("tool_call", call))
                self._buf = self._buf[end + len(TOOL_CALL_END):]
                self._state = "normal"
                continue
            if self._state == "in_think":
                end = self._buf.find(THINK_END)
                if end == -1:
                    # 把可能是 </think> 前缀的尾部留下, 其余思考内容输出
                    emit_len = max(len(self._buf) - (len(THINK_END) - 1), 0)
                    if emit_len:
                        self._emit_content(events, self._buf[:emit_len])
                        self._buf = self._buf[emit_len:]
                    break
                # 思考内容照常输出, 只丢弃 </think> 标记本身
                self._emit_content(events, self._buf[:end])
                self._buf = self._buf[end + len(THINK_END):]
                self._state = "normal"
                continue
            # normal: 找下一个开标记(取先出现的)
            call_pos = self._buf.find(TOOL_CALL_START)
            think_pos = self._buf.find(THINK_START)
            if call_pos == -1 and think_pos == -1:
                hold = max(len(TOOL_CALL_START), len(THINK_START)) - 1
                emit_len = max(len(self._buf) - hold, 0)
                if emit_len:
                    self._emit_content(events, self._buf[:emit_len])
                    self._buf = self._buf[emit_len:]
                break
            if think_pos != -1 and (call_pos == -1 or think_pos < call_pos):
                self._switch(events, THINK_START, "in_think")
            else:
                self._switch(events, TOOL_CALL_START, "in_tool_call")
        return events

    def flush(self) -> List[Tuple[str, Any]]:
        """流结束收尾: 未闭合的 <think>/<tool_call> 块按普通文本返回."""
        events: List[Tuple[str, Any]] = []
        self._emit_content(events, self._buf)
        self._buf = ""
        self._state = "normal"
        return events
