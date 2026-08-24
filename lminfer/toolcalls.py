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
THINK_START, THINK_END = "<think>", "</think>"  # engine.py 用于 think_len 统计

# ---- Llama 3.x JSON 工具调用(vLLM 的 --tool-call-parser llama3_json) ----
# Llama 3.1/3.2/3.3 等模型以 JSON 形式输出工具调用(可能带 <|python_tag|> 前缀):
#   <|python_tag|>{"name": "get_weather", "parameters": {"city": "上海"}}
# 多个调用之间以 ; 分隔, 周围允许普通文本(vLLM 的 Llama3JsonToolParser 语义)
LLAMA_PYTHON_TAG = "<|python_tag|>"
# 整个 <think>...</think> 块: 返回文本中保留(客户端可自行剥离);
# 但历史 assistant 消息回传渲染时需剔除(见 server._message_dicts)——
# 思考内容重新进 prompt 会让 Qwen3 后续生成退化(不闭合 think 就调工具/答非所问)
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


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
    """去掉 <tool_call> 块(已解析为 tool_calls), 其余内容原样返回.

    <think>...</think> 思考块(含标签)保留在 content 中, 供需要观测思考过程的
    客户端直接使用; 需要剥离的客户端(如 ToyAgent 的 _strip_think)按标签自行处理.
    """
    text = TOOL_CALL_BLOCK.sub("", text)
    return text.strip()


class ToolCallStreamSplitter:
    """把逐 token 文本流切成 content / tool_call 两类事件(流式响应用).

    行为:
      - <tool_call>...</tool_call> 整块不进入 content, 块内 JSON 解析成工具调用;
      - <think>...</think> 思考块视为普通文本, 连同标签原样透传进 content
        (开启 think 时思考内容不丢弃, 由客户端决定是否剥离);
      - 流结束时未闭合的块按普通文本返回(模型没写完就当文本);
      - 标记即使被拆成多个 chunk 到达也能正确识别(每个状态只保留可能
        是不完整标记的尾部, 其余内容立即输出).

    用法: events = splitter.push(chunk) 返回事件列表, 事件为
      ("content", str) 或 ("tool_call", dict); 流结束后取 splitter.flush().
    """

    def __init__(self) -> None:
        self._buf = ""
        self._state = "normal"  # normal / in_tool_call

    def _emit_content(self, events: List[Tuple[str, Any]], text: str) -> None:
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
            # normal: 只拦截 <tool_call> 开标记, <think> 等其余文本原样输出
            call_pos = self._buf.find(TOOL_CALL_START)
            if call_pos == -1:
                hold = len(TOOL_CALL_START) - 1
                emit_len = max(len(self._buf) - hold, 0)
                if emit_len:
                    self._emit_content(events, self._buf[:emit_len])
                    self._buf = self._buf[emit_len:]
                break
            self._switch(events, TOOL_CALL_START, "in_tool_call")
        return events

    def flush(self) -> List[Tuple[str, Any]]:
        """流结束收尾: 未闭合的块按普通文本返回."""
        events: List[Tuple[str, Any]] = []
        self._emit_content(events, self._buf)
        self._buf = ""
        self._state = "normal"
        return events


# ---------------------------------------------------------------------------
# Llama 3.x JSON 工具调用
# ---------------------------------------------------------------------------

def _extract_llama_raw_args(block: str) -> str | None:
    """提取根对象内顶层 "parameters" 或 "arguments" 键的原始 JSON 值子串.

    与 _extract_raw_arguments 同理: 模型原始生成的参数(如 {"city":"上海"} 紧凑格式)
    原样保留, 客户端回传后模板再渲染才能与生成流一致. 两个键名都接受
    (Llama 3.x 模型两种写法都出现过), 提取失败返回 None(调用方回退重序列化).
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
                    if k < n and block[k] == ":" and block[i + 1:j] in ("parameters", "arguments"):
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


def _llama_call_from_obj(obj: Dict[str, Any], block: str) -> Dict[str, Any] | None:
    """把一个已解析的 Llama JSON 工具调用对象转成 OpenAI 格式, 失败返回 None.

    block 是对象对应的原始 JSON 子串(用于提取参数原始子串做 round-trip 保真).
    与 vLLM 语义一致: 必须有 "name" 键, 参数取 "parameters"(优先)或 "arguments".
    """
    name = obj.get("name") if isinstance(obj, dict) else None
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("parameters", obj.get("arguments"))
    if isinstance(args, str):  # 容忍模型把 parameters 写成 JSON 字符串
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    raw_args = _extract_llama_raw_args(block)
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


def parse_llama3_json_tool_calls(text: str) -> List[Dict[str, Any]]:
    """扫描整段生成文本, 提取 Llama 3.x JSON 工具调用.

    与 vLLM 的 llama3_json 解析一致: 用 JSONDecoder.raw_decode 从每个 { 处解析
    完整 JSON 对象(正确处理任意嵌套深度与字符串内的括号), 跳过已解析对象内部的
    {, 支持多个对象以 ; 分隔及周围任意文本. 解析失败/缺 name 键的对象跳过.
    """
    calls: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    end = -1  # 已解析对象覆盖到的下标: 跳过其内部的 {, 避免把嵌套对象当新调用
    for m in re.finditer(r"\{", text):
        start = m.start()
        if start <= end:
            continue
        try:
            obj, n = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        end = start + n
        call = _llama_call_from_obj(obj, text[start:end])
        if call is not None:
            calls.append(call)
    return calls


def clean_llama3_json_content(text: str) -> str:
    """去掉开头的 <|python_tag|> 标记, 其余内容原样返回.

    JSON 工具调用对象已被解析为 tool_calls(与 vLLM 一致, 有 tool_calls 时
    content 为 null), 无需在这里剔除; 这里只清理模型在普通文本回复时
    误带的 <|python_tag|> 前缀(特殊 token, skip_special_tokens=False 解码
    后原样出现).
    """
    return text.removeprefix(LLAMA_PYTHON_TAG).strip()


class LlamaJsonStreamSplitter:
    """把逐 token 文本流切成 content / tool_call 事件(Llama 3.x JSON 工具调用).

    与 vLLM 的 llama3_json 流式语义一致: 输出以 <|python_tag|> 或 { 开头时
    进入 JSON 工具调用模式(整段解析成 tool_calls), 否则按普通文本透传 content.
    JSON 模式内部:
      - 解析完整的 {"name":..., "parameters":...} 对象(支持多个, 以 ; 分隔),
        每个对象解析成一个 tool_call 事件;
      - 对象之间的说明文字/尾部收尾文字按 content 输出;
      - 未闭合的 JSON 对象继续攒缓冲, 流结束时按普通文本返回(模型没写完就当文本).

    用法与 ToolCallStreamSplitter 相同: events = splitter.push(chunk),
    流结束后取 splitter.flush().
    """

    def __init__(self) -> None:
        self._buf = ""
        self._state = "undecided"  # undecided(未定)/ json / text

    def _emit_content(self, events: List[Tuple[str, Any]], text: str) -> None:
        if text:
            events.append(("content", text))

    def _decide(self, events: List[Tuple[str, Any]]) -> None:
        """根据已积累的缓冲决定走 JSON 工具模式还是普通文本模式."""
        buf = self._buf
        stripped = buf.lstrip()
        lead_ws = len(buf) - len(stripped)
        # 前导 <|python_tag|>(可能跨 chunk 到达): 完整出现才剥离, 否则继续等
        if stripped.startswith(LLAMA_PYTHON_TAG):
            if len(stripped) == len(LLAMA_PYTHON_TAG):
                return  # 只有标记本身, 等更多内容
            self._buf = buf[lead_ws + len(LLAMA_PYTHON_TAG):]
            stripped = self._buf.lstrip()
            lead_ws = len(self._buf) - len(stripped)
        elif stripped.startswith(LLAMA_PYTHON_TAG[:max(len(stripped), 1)]):
            # 当前缓冲是标记的前缀(如 "<|py"): 可能是标记被拆成多 chunk, 继续等
            if len(stripped) < len(LLAMA_PYTHON_TAG):
                return
        if stripped.startswith("{"):
            self._state = "json"
            return
        # 其他字符开头: 普通文本, 已积累的内容全部按 content 输出
        self._state = "text"
        self._emit_content(events, buf)
        self._buf = ""

    def _parse_json(self, events: List[Tuple[str, Any]]) -> None:
        """JSON 工具模式: 逐个解析完整对象; 其余文本按 content 输出."""
        while True:
            stripped = self._buf.lstrip()
            ws = len(self._buf) - len(stripped)
            if not stripped:
                return  # 全空白: 等更多内容
            if stripped[0] in ";,":
                self._buf = stripped[1:]  # 对象分隔符: 跳过(空白在下一轮 lstrip 处理)
                continue
            if stripped[0] != "{":
                # 对象之间的说明文字 / 尾部收尾文字: 按 content 输出
                self._emit_content(events, self._buf)
                self._buf = ""
                return
            try:
                obj, n = json.JSONDecoder().raw_decode(stripped)
            except json.JSONDecodeError:
                return  # 未完成的 JSON 对象: 攒缓冲等闭合, 流结束时 flush 兜底
            raw = stripped[:n]
            self._buf = stripped[n:]
            call = _llama_call_from_obj(obj, raw)
            if call is not None:
                events.append(("tool_call", call))
            else:
                # 是 JSON 对象但不是工具调用(缺 name 键): 当普通文本输出
                self._emit_content(events, raw)

    def push(self, chunk: str) -> List[Tuple[str, Any]]:
        events: List[Tuple[str, Any]] = []
        self._buf += chunk
        if self._state == "undecided":
            self._decide(events)
            if self._state == "undecided":
                return events
        if self._state == "text":
            self._emit_content(events, self._buf)
            self._buf = ""
            return events
        self._parse_json(events)
        return events

    def flush(self) -> List[Tuple[str, Any]]:
        """流结束收尾: 未闭合的 JSON/残留分隔符按普通文本返回."""
        events: List[Tuple[str, Any]] = []
        tail = self._buf
        if tail.strip(" \t\r\n;,"):
            self._emit_content(events, tail)
        self._buf = ""
        self._state = "undecided"
        return events
