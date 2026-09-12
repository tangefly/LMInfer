"""Qwen / Hermes / Llama / Mistral 风格工具调用解析(对应 vLLM 的 --tool-call-parser).

Qwen2.5/Qwen3 模型在回复中以如下格式输出工具调用(与 Hermes 格式相同,
vLLM 跑 Qwen3 用的 hermes parser 解析的就是这个格式):

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Shanghai"}}
    </tool_call>

Llama 3.x 输出 `{"name": ..., "parameters": {...}}` JSON(可能带 `<|python_tag|>`);

Mistral v11+ 系(Ministral 3 等)输出(见 vLLM 的 mistral parser):

    [TOOL_CALLS]get_weather[ARGS]{"city": "Shanghai"}

其中 `[TOOL_CALLS]`/`[ARGS]`/`[CALL_ID]` 都是特殊 token; 多个调用直接首尾相接
(每个都以 `[TOOL_CALLS]` 开头)。v11 tokenizer 会在 name 与 `[ARGS]` 之间插入
`[CALL_ID]<id>`, v13 不再插入(本机 Ministral-3 是 v13)。

GLM-4.5/4.6/4.7 系 MoE(glm4_moe / glm4_moe_lite, 见 vLLM 的 glm45/glm47 parser)
输出 XML 形态的工具调用:

    <tool_call>get_weather<arg_key>city</arg_key><arg_value>上海</arg_value></tool_call>

函数名直接跟在 `<tool_call>` 后, 参数按 `<arg_key>`/`<arg_value>` 成对出现,
值是标签之间的原始文本(按字符串返回, 不做 JSON 反序列化)。

<tool_call> 等是模型的特殊 token, 生成时需要用 skip_special_tokens=False 解码,
解析成功后转成 OpenAI 的 tool_calls 字段(与 /v1/chat/completions 响应格式一致).
"""
from __future__ import annotations

import ast
import json
import logging
import random
import re
import string
import uuid
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("lminfer")

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
# 两个标记都是特殊 token, 解码后原样出现; 但保险起见仍按子串匹配
TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
THINK_START, THINK_END = "<think>", "</think>"  # engine.py 用于 think_len 统计

# ---- Mistral v11+ 工具调用(vLLM 的 --tool-call-parser mistral) ----
MISTRAL_TOOL_CALLS = "[TOOL_CALLS]"
MISTRAL_TOOL_ARGS = "[ARGS]"
MISTRAL_CALL_ID = "[CALL_ID]"
# mistral-common 校验 tool_call id: a-z/A-Z/0-9, 长度固定 9(见 vLLM MistralToolCall)
MISTRAL_CALL_ID_ALPHABET = string.ascii_letters + string.digits
MISTRAL_CALL_ID_LEN = 9
# 旧版(<v11)Mistral 用 [TOOL_CALLS] [{...}, {...}] 的 JSON 数组(仍兼容解析)
MISTRAL_JSON_ARRAY = re.compile(r"\[{.*}\]", re.DOTALL)

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


def _repair_array_args(args: Dict[str, Any], schema: dict | None) -> bool:
    """按 schema 把"字符串形式的列表"参数还原成真正的 JSON 数组, 返回是否有修改.

    Llama 3.x 模型经常把 array 参数输出成字符串化的 Python 列表(如
    "source_ids": "['S1']"), 客户端按 schema 校验会拒绝执行该调用, 多轮
    agent 场景下模型又往往读不懂"必须给数组"的报错, 直接退化成死循环.
    schema 声明 type=array 且值是字符串时, 尝试按 Python 字面量解析
    (ast.literal_eval, 只接受字面量, 不会执行任意代码), 成功且结果是
    列表则替换; 解析失败保持原样(原值仍是合法 JSON 字符串).
    """
    if not isinstance(schema, dict):
        return False
    props = schema.get("properties")
    if not isinstance(props, dict):
        return False
    changed = False
    for key, spec in props.items():
        if (not isinstance(spec, dict) or spec.get("type") != "array"
                or not isinstance(args.get(key), str)):
            continue
        try:
            value = ast.literal_eval(args[key])
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, list):
            args[key] = value
            changed = True
    return changed


def _parse_call_block(block: str, schema: dict | None = None) -> Dict[str, Any] | None:
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
    # 优先返回原始 JSON 子串(保真 round-trip); schema 修复改变了参数时,
    # 原始子串不再有效, 改用重序列化结果
    if not _repair_array_args(args, schema):
        raw_args = _extract_raw_arguments(block)
        if raw_args is not None:
            try:
                json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = None
        if raw_args is None:
            raw_args = json.dumps(args, ensure_ascii=False)
    else:
        raw_args = json.dumps(args, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name, "arguments": raw_args},
    }


def parse_tool_calls(text: str, schemas: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """扫描可见输出, 返回 OpenAI 格式的 tool_calls 列表(解析失败的块跳过).

    schemas: 函数名 -> parameters schema, 供 _repair_array_args 修复
    "字符串形式的列表"参数; 为 None 时不做修复(原样返回模型输出).
    """
    text = THINK_BLOCK.sub("", text)
    calls: List[Dict[str, Any]] = []
    for block in TOOL_CALL_BLOCK.findall(text):
        call = _parse_call_block(block, (schemas or {}).get(_block_name(block)))
        if call is not None:
            calls.append(call)
    return calls


def _block_name(block: str) -> str:
    """从 <tool_call> 块里取函数名(供 schema 查找; 解析失败返回空串)."""
    try:
        data = json.loads(block)
    except json.JSONDecodeError:
        return ""
    name = data.get("name") if isinstance(data, dict) else None
    return name if isinstance(name, str) else ""


def clean_content(text: str) -> str:
    """去掉 <tool_call> 与 <think> 块, 只返回可进入对话历史的内容."""
    text = TOOL_CALL_BLOCK.sub("", text)
    text = THINK_BLOCK.sub("", text)
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

    def _parse_block(self, block: str) -> Dict[str, Any] | None:
        """解析一个完整的 <tool_call> 块体(子类按自己的协议覆写, 见
        Glm4MoeStreamSplitter: 同样的标记, 块体是 XML 参数而不是 JSON)。
        """
        return _parse_call_block(block)

    def push(self, chunk: str) -> List[Tuple[str, Any]]:
        events: List[Tuple[str, Any]] = []
        self._buf += chunk
        while True:
            if self._state == "in_tool_call":
                end = self._buf.find(TOOL_CALL_END)
                if end == -1:
                    break  # 未闭合: 继续攒缓冲, 不提前输出
                call = self._parse_block(self._buf[:end])
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


def _llama_call_from_obj(obj: Dict[str, Any], block: str,
                         schema: dict | None = None) -> Dict[str, Any] | None:
    """把一个已解析的 Llama JSON 工具调用对象转成 OpenAI 格式, 失败返回 None.

    block 是对象对应的原始 JSON 子串(用于提取参数原始子串做 round-trip 保真).
    与 vLLM 语义一致: 必须有 "name" 键, 参数取 "parameters"(优先)或 "arguments".
    schema: 该函数的 parameters schema, 供 _repair_array_args 修复
    "字符串形式的列表"参数; 修复改变参数时原始子串失效, 改用重序列化结果.
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
    if not _repair_array_args(args, schema):
        raw_args = _extract_llama_raw_args(block)
        if raw_args is not None:
            try:
                json.loads(raw_args)
            except json.JSONDecodeError:
                raw_args = None
        if raw_args is None:
            raw_args = json.dumps(args, ensure_ascii=False)
    else:
        raw_args = json.dumps(args, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name, "arguments": raw_args},
    }


def parse_llama3_json_tool_calls(text: str,
                                 schemas: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """扫描可见输出, 提取 Llama 3.x JSON 工具调用.

    与 vLLM 的 llama3_json 解析一致: 用 JSONDecoder.raw_decode 从每个 { 处解析
    完整 JSON 对象(正确处理任意嵌套深度与字符串内的括号), 跳过已解析对象内部的
    {, 支持多个对象以 ; 分隔及周围任意文本. 解析失败/缺 name 键的对象跳过.
    schemas: 函数名 -> parameters schema, 供 _repair_array_args 修复
    "字符串形式的列表"参数; 为 None 时不做修复(原样返回模型输出).
    """
    text = THINK_BLOCK.sub("", text)
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
        call = _llama_call_from_obj(obj, text[start:end], (schemas or {}).get(
            obj.get("name") if isinstance(obj, dict) else None))
        if call is not None:
            calls.append(call)
    return calls


def clean_llama3_json_content(text: str) -> str:
    """去掉工具调用前缀与 <think> 块, 只返回可进入对话历史的内容."""
    text = text.removeprefix(LLAMA_PYTHON_TAG)
    text = THINK_BLOCK.sub("", text)
    return text.strip()


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


# ---------------------------------------------------------------------------
# Mistral v11+ 工具调用
# ---------------------------------------------------------------------------

def _mistral_new_call_id() -> str:
    """生成符合 mistral-common 校验的 tool_call id(9 位字母数字).

    Mistral 的 chat template 由 mistral-common 渲染, 它会校验 id 格式: 非
    `[a-zA-Z0-9]{9}` 直接抛 InvalidFunctionCallException —— 也就是说 id 不合规时,
    工具结果根本回填不进下一轮 prompt, 多轮 agent 会在第二次请求上直接 400。
    """
    return "".join(random.choices(MISTRAL_CALL_ID_ALPHABET, k=MISTRAL_CALL_ID_LEN))


def _mistral_arguments(raw: str | None, schema: dict | None) -> str:
    """把 Mistral 输出里的参数原始 JSON 子串转成 OpenAI 的 arguments 字符串.

    与 hermes/llama 解析一致: 能逐位保留模型原始输出就保留(模板二次渲染才能与
    生成流对齐 —— Ministral 的 mistral-common 会把参数 json.dumps 归一化, 模型
    自身输出的 `": "` 空格风格与之一致, 因此 LCP 能整段命中); schema 修复改变了
    参数时改用重序列化结果。JSON 不合法时原样返回, 不静默丢成 `{}`(与 vLLM 的
    mistral parser 一致, 让模型知道自己尝试过调用)。
    """
    raw = (raw or "").strip() or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if isinstance(data, dict) and _repair_array_args(data, schema):
        return json.dumps(data, ensure_ascii=False)
    return raw


def _mistral_call(name: str, raw_args: str | None,
                  schema: dict | None = None) -> Dict[str, Any] | None:
    """(name, 参数原始 JSON 子串) -> OpenAI 格式工具调用; name 为空返回 None."""
    name = (name or "").strip()
    if not name:
        return None
    return {
        "id": _mistral_new_call_id(),
        "type": "function",
        "function": {"name": name, "arguments": _mistral_arguments(raw_args, schema)},
    }


def _mistral_head_name(head: str) -> str:
    """从 `name[CALL_ID]<id>[ARGS]` 头部里取出函数名."""
    for marker in (MISTRAL_CALL_ID, MISTRAL_TOOL_ARGS):
        head = head.split(marker)[0]
    return head.strip()


def parse_mistral_tool_calls(text: str,
                             schemas: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """解析 Mistral v11+ 的 `[TOOL_CALLS]name[ARGS]{json}`(支持多个连续调用).

    兼容旧版 `<v11` 的 `[TOOL_CALLS] [{"name": ..., "arguments": {...}}]` 数组形式
    (vLLM 的 mistral parser 同样两代都支持)。schemas: 函数名 -> parameters schema,
    供 _repair_array_args 修复"字符串形式的列表"参数。
    """
    text = THINK_BLOCK.sub("", text)
    if MISTRAL_TOOL_CALLS not in text:
        return []
    calls: List[Dict[str, Any]] = []
    for part in text.split(MISTRAL_TOOL_CALLS)[1:]:
        if part.lstrip().startswith("[{"):
            # 旧版(<v11)的数组形式: [TOOL_CALLS] [{"name": ..., "arguments": {...}}]
            calls.extend(_parse_mistral_json_array(part, schemas))
            continue
        brace = part.find("{")
        head, rest = part[:brace], part[brace:]
        raw = _match_json_value(rest, 0)
        if raw is None:
            raw = rest  # JSON 没闭合(或畸形): 原样收下, 由客户端/模型自行纠正
        name = _mistral_head_name(head)
        call = _mistral_call(name, raw, (schemas or {}).get(name))
        if call is not None:
            calls.append(call)
    return calls


def _parse_mistral_json_array(part: str,
                              schemas: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    """旧版 Mistral: `[TOOL_CALLS] [{...}, {...}]`(参数在 JSON 对象里)."""
    match = MISTRAL_JSON_ARRAY.search(part)
    if match is None:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    calls: List[Dict[str, Any]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        args = item.get("arguments", item.get("parameters"))
        if isinstance(args, str):
            raw = args
        else:
            raw = json.dumps(args if args is not None else {}, ensure_ascii=False)
        calls.append(_mistral_call(name, raw, (schemas or {}).get(name)))
    return [c for c in calls if c is not None]


def clean_mistral_content(text: str) -> str:
    """只保留第一个 [TOOL_CALLS] 之前的文本(与 vLLM mistral parser 语义一致)."""
    text = THINK_BLOCK.sub("", text)
    return text.split(MISTRAL_TOOL_CALLS)[0].strip()


class MistralStreamSplitter:
    """把逐 token 文本流切成 content / tool_call 事件(Mistral [TOOL_CALLS] 协议).

    `[TOOL_CALLS]` 之前的内容按 content 透传; 之后的 `name[CALL_ID]<id>[ARGS]`
    头部攒起来, 等参数 JSON 闭合再产出一个 tool_call 事件, 然后回到普通模式
    (多个调用首尾相接时会依次产出多个事件)。流结束时没闭合的调用按普通文本
    返回(与 hermes/llama 的流式实现一致: 模型没写完就当文本, 不静默丢弃)。

    用法与 ToolCallStreamSplitter 相同: events = splitter.push(chunk),
    流结束后取 splitter.flush()。
    """

    def __init__(self) -> None:
        self._buf = ""
        self._head = ""            # 当前调用的 `name[CALL_ID]<id>[ARGS]` 头部
        self._state = "normal"     # normal / in_call

    def _emit_content(self, events: List[Tuple[str, Any]], text: str) -> None:
        if text:
            events.append(("content", text))

    def push(self, chunk: str) -> List[Tuple[str, Any]]:
        events: List[Tuple[str, Any]] = []
        self._buf += chunk
        while True:
            if self._state == "normal":
                pos = self._buf.find(MISTRAL_TOOL_CALLS)
                if pos == -1:
                    # 只保留可能是"半个标记"的尾部, 其余立即输出
                    hold = len(MISTRAL_TOOL_CALLS) - 1
                    emit = max(len(self._buf) - hold, 0)
                    if emit:
                        self._emit_content(events, self._buf[:emit])
                        self._buf = self._buf[emit:]
                    break
                if pos:
                    self._emit_content(events, self._buf[:pos])
                self._buf = self._buf[pos + len(MISTRAL_TOOL_CALLS):]
                self._head, self._state = "", "in_call"
                continue
            brace = self._buf.find("{")
            if brace < 0:
                self._head += self._buf
                self._buf = ""
                break
            self._head += self._buf[:brace]
            self._buf = self._buf[brace:]
            raw = _match_json_value(self._buf, 0)
            if raw is None:
                break  # 参数 JSON 未闭合: 继续攒缓冲
            call = _mistral_call(_mistral_head_name(self._head), raw)
            if call is not None:
                events.append(("tool_call", call))
            self._buf = self._buf[len(raw):]
            self._head, self._state = "", "normal"
        return events

    def flush(self) -> List[Tuple[str, Any]]:
        """流结束收尾: 未闭合的调用按普通文本返回."""
        events: List[Tuple[str, Any]] = []
        self._emit_content(events, self._head + self._buf)
        self._head, self._buf, self._state = "", "", "normal"
        return events


# ---------------------------------------------------------------------------
# GLM-4 工具调用(name\n{json})
# ---------------------------------------------------------------------------

# GLM-4-0414 的函数调用有两种实测形态:
#   1) 函数名独占一行, 紧跟一个 JSON 参数对象(输出开头即调用):
#        'get_weather\n{"city": "北京"}'
#   2) 函数名前面带一个**字面量角色标记** `<|assistant|>`(模型卡参考实现按
#      `<|assistant|>` split 以后逐段解析, 就是为这种形态):
#        '...decision note...<|assistant|>research\n{"query": "...", "source_ids": ["S1"]}'
# 之后以 <|observation|>(eos)或 <|user|> 结束。没有专用的起始特殊 token, 因此
# 自动识别看 config.model_type(见 model_adapters._is_glm4_family)。
GLM4_NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_.\-]*"
GLM4_ROLE_MARKER = "<|assistant|>"
# 两个候选锚点(用命名组区分): 角色标记锚点, 或行首锚点。用 lookahead 让 match.end()
# 正好落在 '{' 上, 便于 raw_decode 取完整 JSON 对象。
GLM4_CALL = re.compile(
    rf"(?m)(?:(?P<role>{re.escape(GLM4_ROLE_MARKER)})[ \t]*|^[ \t]*)"
    rf"(?P<name>{GLM4_NAME_PATTERN})[ \t]*\r?\n[ \t]*(?=\{{)")
# 尚未成形的调用尾巴(流式扣留用): 行尾的裸函数名, 可选一个换行, 可选 '{'。
_GLM4_PARTIAL_TAIL = re.compile(
    rf"(?m)^[ \t]*(?P<name>{GLM4_NAME_PATTERN})[ \t]*\r?\n?[ \t]*\{{?\Z")


def _glm4_candidate_ok(match: re.Match, schemas: Dict[str, Any] | None,
                       at_output_start: bool) -> bool:
    """这个 `名字\\n{` 候选是否可信为工具调用(而不是正文里的同形排版).

    三个接受条件, 满足其一即可:
    - **角色标记锚点**: 名字前面紧跟 `<|assistant|>`(GLM-4 真实输出的形态 2);
    - **输出开头**: 正文之前没有任何非空白内容(形态 1; 模型直接以调用开头);
    - **白名单**: 名字在请求声明的工具里(正文中间出现的合法工具调用也不漏)。

    只看"行首标识符 + JSON 对象"会在正文里误判(模型写计划时常见的
    `Plan\\n{"requirements": [...]}`); 但也不能反过来按白名单硬过滤 —— 那样模型
    一旦输出未声明的函数名(实测 `research`), 调用会被静默丢掉, 客户端拿不到任何
    反馈, 主 agent 会一直空转到 turn budget 耗尽。所以: 未声明的名字只有在
    角色标记/输出开头这两处**协议锚点**上才认。
    """
    if schemas and match.group("name") in schemas:
        return True
    if match.group("role") is not None:
        return True
    return at_output_start



def _glm4_call(name: str, raw_args: str | None,
               schema: dict | None = None) -> Dict[str, Any] | None:
    """(函数名, 参数原始 JSON 子串) -> OpenAI 格式工具调用; 不合法返回 None.

    参数必须是合法 JSON **对象**: GLM 的协议里函数名下一行就是参数对象, 若 JSON
    畸形(模型没写完)则不当作工具调用 —— 宁可让这段文本留在 content 里, 也不要
    抛出一个客户端无法执行、且会污染下一轮 prompt 的假调用。arguments 保留模型
    原始 JSON 子串(round-trip 保真, 模板二次渲染才能与生成流逐位一致)。
    """
    name = (name or "").strip()
    if not name:
        return None
    raw = (raw_args or "").strip() or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if _repair_array_args(data, schema):
        raw = json.dumps(data, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name, "arguments": raw},
    }


def parse_glm4_tool_calls(text: str,
                          schemas: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """扫描可见输出, 提取 GLM-4 的 `name\\n{json}` 工具调用.

    schemas: 函数名 -> parameters schema。既用于 schema 感知的 array 参数修复, 也
    作为**候选接受条件之一**(见 _glm4_candidate_ok): 名字在请求声明的工具里, 或
    名字前面紧跟 `<|assistant|>` 角色标记, 或调用出现在输出开头。这样既能容纳
    GLM-4 真实输出的两种形态, 又不会把正文里"行首单词 + JSON 对象"的排版当成调用。
    """
    text = THINK_BLOCK.sub("", text)
    calls: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    end = -1  # 已解析对象覆盖到的下标: 跳过其内部的 '{', 避免把嵌套对象当新调用
    for m in GLM4_CALL.finditer(text):
        if m.start() <= end:
            continue
        if not _glm4_candidate_ok(m, schemas, text[:m.start()].strip() == ""):
            continue
        name = m.group("name")
        brace = m.end()
        try:
            obj, n = decoder.raw_decode(text[brace:])
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        call = _glm4_call(name, text[brace:brace + n], (schemas or {}).get(name))
        if call is not None:
            calls.append(call)
            end = brace + n
    return calls


def clean_glm4_content(text: str) -> str:
    """去掉 think 块, 返回可进入对话历史的正文.

    GLM-4 没有工具调用专用起始标记, 无法像 hermes 那样"剥掉标记块"; 只有在**真正
    解析出**工具调用时(parse_model_output 的 content 分支)才切掉调用本体。这里
    没有解析结果可依据, 因此刻意不删任何 `name\\n{json}` 形态的文本 —— 否则
    "一行单词 + JSON 对象"的正常回答会被误删成空串(与 llama3_json 的 cleaner 一致,
    它也只删 <|python_tag|> 前缀与 think 块)。
    """
    return THINK_BLOCK.sub("", text).strip()


def clean_glm4_prefix(text: str, schemas: Dict[str, Any] | None = None) -> str:
    """取第一个工具调用之前的正文(**不 strip**, round-trip 保真用).

    必须用与 parse_glm4_tool_calls 相同的接受条件定位起点: 否则正文里被解析器
    忽略的"行首单词 + JSON"排版会把切口提前。`
    """
    text = THINK_BLOCK.sub("", text)
    for match in GLM4_CALL.finditer(text):
        if _glm4_candidate_ok(match, schemas, text[:match.start()].strip() == ""):
            return text[:match.start()]
    return text


class Glm4StreamSplitter:
    """把逐 token 文本流切成 content / tool_call 事件(GLM-4 name\\n{json} 协议).

    与 hermes/llama/mistral 的流式切分器同一组 push/flush 接口。GLM 的工具调用没有
    起始特殊 token, 只有两种协议锚点(见 _glm4_candidate_ok): 输出开头的
    `name\\n{json}`, 或任意位置 `<|assistant|>name\\n{json}`。切分器在缓冲里持续扫描:
    - 命中完整调用 -> 把锚点之前的文本作为 content 发出, 再发 tool_call;
    - 命中尚未闭合的调用(锚点已出现但 JSON 没写完)-> 从锚点起扣留, 等后续 chunk;
    - 都没有 -> 只扣留"可能长成锚点"的尾巴(角色标记前缀/裸标识符), 其余立即发 content。

    流结束时残留缓冲按普通文本返回(与既有各协议一致: 模型没写完就当文本)。
    """

    def __init__(self, tool_names=None) -> None:
        self._buf = ""
        # 已声明的工具名(server 传入): 只用于放宽"正文中间的合法工具调用"这一
        # 接受条件(见 _glm4_candidate_ok); 未声明的名字靠角色标记/输出开头锚点识别。
        self._schemas = ({name: None for name in tool_names} if tool_names else None)
        self._emitted = False     # 是否已吐出过 content(判断"输出开头"用)
        self._after_call = False  # 上一个事件是 tool_call(连续多个调用时后一个也在锚点上)

    def _emit_content(self, events: List[Tuple[str, Any]], text: str) -> None:
        if text:
            self._emitted = True
            self._after_call = False
            events.append(("content", text))

    def _scan(self, buf: str):
        """返回 ('call', start, name, raw, end) / ('pending', start) / None."""
        decoder = json.JSONDecoder()
        for match in GLM4_CALL.finditer(buf):
            at_start = ((not self._emitted and match.start() == 0)
                        or (self._after_call and buf[:match.start()].strip() == ""))
            if not _glm4_candidate_ok(match, self._schemas, at_start):
                continue
            brace = match.end()
            try:
                obj, n = decoder.raw_decode(buf[brace:])
            except json.JSONDecodeError:
                return ("pending", match.start())
            if not isinstance(obj, dict):
                continue
            return ("call", match.start(), match.group("name"),
                    buf[brace:brace + n], brace + n)
        return None

    def _role_tail_is_call_prefix(self, tail: str) -> bool:
        """`<|assistant|>` 之后的内容是否还像"正在写的调用"(而不是正文)."""
        rest = tail.lstrip(" \t")
        if rest == "":
            return True
        match = re.match(GLM4_NAME_PATTERN, rest)
        if match is None:
            return False
        rest = rest[match.end():]
        if rest.strip(" \t") == "":
            return True
        if rest.startswith("\n") or rest.startswith("\r\n"):
            after = rest.lstrip("\r\n \t")
            return after == "" or after.startswith("{")
        return False

    def _holdback(self, buf: str) -> int:
        """从哪个下标起扣留(尾部可能长成调用锚点); 没有则 len(buf)."""
        best = len(buf)
        index = buf.rfind(GLM4_ROLE_MARKER)
        if index >= 0 and self._role_tail_is_call_prefix(
                buf[index + len(GLM4_ROLE_MARKER):]):
            best = min(best, index)
        # 半截角色标记(如 "<|assist")
        for k in range(1, len(GLM4_ROLE_MARKER)):
            if buf.endswith(GLM4_ROLE_MARKER[:k]):
                best = min(best, len(buf) - k)
                break
        # 行尾未成形的 "name" / "name\n" / "name\n{": 输出开头, 白名单工具名,
        # 或紧跟上一个 tool_call(连续多个调用)时扣留
        match = _GLM4_PARTIAL_TAIL.search(buf)
        if match is not None:
            at_start = ((not self._emitted and match.start() == 0)
                        or (self._after_call and buf[:match.start()].strip() == ""))
            known = self._schemas is not None and match.group("name") in self._schemas
            if at_start or known:
                best = min(best, match.start())
        return best

    def push(self, chunk: str) -> List[Tuple[str, Any]]:
        events: List[Tuple[str, Any]] = []
        buf = self._buf + chunk
        while True:
            found = self._scan(buf)
            if found is None:
                hold = self._holdback(buf)
                if hold > 0:
                    self._emit_content(events, buf[:hold])
                    buf = buf[hold:]
                break
            if found[0] == "pending":
                start = found[1]
                if start > 0:
                    self._emit_content(events, buf[:start])
                    buf = buf[start:]
                break
            _, start, name, raw, end = found
            if start > 0:
                self._emit_content(events, buf[:start])
            call = _glm4_call(name, raw)
            if call is not None:
                events.append(("tool_call", call))
                self._after_call = True
            else:
                # 形态像调用但参数不是合法 JSON 对象: 当正文输出
                self._emit_content(events, buf[start:end])
            buf = buf[end:]
        self._buf = buf
        return events

    def flush(self) -> List[Tuple[str, Any]]:
        """流结束收尾: 未闭合的调用/尾部文字按普通文本返回."""
        events: List[Tuple[str, Any]] = []
        if self._buf:
            self._emit_content(events, self._buf)
        self._buf = ""
        return events


# ---------------------------------------------------------------------------
# GLM-4.5/4.6/4.7 MoE 工具调用(<tool_call>name<arg_key>k</arg_key><arg_value>v)
# ---------------------------------------------------------------------------

# GLM-4.5/4.6/4.7 系 MoE(config.model_type = glm4_moe / glm4_moe_lite)的工具调用
# 是 XML 形态, 与 GLM-4-0414 的 `name\n{json}` 完全不同, 因此单独一套解析器
# (vLLM 也是分开注册的: glm45/glm47 → glm47_moe parser):
#
#   <tool_call>get_weather<arg_key>city</arg_key><arg_value>北京</arg_value></tool_call>
#
# - 函数名直接跟在 <tool_call> 后(可无参数: 名字后面就是 </tool_call>);
# - 参数按 <arg_key>/<arg_value> 成对出现, 值是标签之间的**原始文本**;
# - 参数值不做 JSON 反序列化, 一律按字符串返回(与 vLLM 的 glm47 一致):
#   GLM-4.7 的模板对字符串参数原样渲染、只对非字符串才 `| tojson`, 保留原始文本
#   才能让客户端回填后的 prompt 与模型生成流逐位一致 —— 跨请求 KV 前缀要覆盖
#   工具调用回合, 靠的就是这一点(实测 round-trip 逐位全等)。
GLM4_MOE_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
GLM4_MOE_ARG = re.compile(
    r"<arg_key>(?P<key>.*?)</arg_key>\s*"
    r"<arg_value>(?P<value>.*?)</arg_value>",
    re.DOTALL,
)


def _glm4_moe_block_name(block: str) -> str:
    """从一个 <tool_call> 块体里取函数名(用于按 schema 查参数; 取不到返回空串)."""
    first = GLM4_MOE_ARG.search(block)
    return (block[:first.start()] if first is not None else block).strip()


def _glm4_moe_call(block: str, schema: dict | None = None) -> Dict[str, Any] | None:
    """一个 <tool_call> 块体 -> OpenAI 格式工具调用; 函数名为空返回 None.

    schema 只用于把"字符串形式的列表"参数还原成真正的数组(见 _repair_array_args,
    与 hermes/mistral/glm4 一致的修复): 修复动了参数才会改变 arguments 的形态,
    不动时逐位保留原始值。
    """
    name = _glm4_moe_block_name(block)
    if not name:
        return None
    args: Dict[str, Any] = {}
    for match in GLM4_MOE_ARG.finditer(block):
        key = match.group("key").strip()
        if key:
            args[key] = match.group("value")
    _repair_array_args(args, schema)
    return {
        "id": f"call_{uuid.uuid4().hex[:16]}",
        "type": "function",
        "function": {"name": name,
                     "arguments": json.dumps(args, ensure_ascii=False)},
    }


def parse_glm4_moe_tool_calls(text: str,
                              schemas: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """扫描可见输出, 提取 GLM-4.5/4.6/4.7 的 <tool_call>name<arg_key>... 调用.

    schemas: 函数名 -> parameters schema, 供 _repair_array_args 修复"字符串形式的
    列表"参数; 为 None 时不做修复(原样返回模型输出的值)。
    """
    text = THINK_BLOCK.sub("", text)
    calls: List[Dict[str, Any]] = []
    for block in GLM4_MOE_CALL_BLOCK.findall(text):
        call = _glm4_moe_call(block, (schemas or {}).get(_glm4_moe_block_name(block)))
        if call is not None:
            calls.append(call)
    return calls


def clean_glm4_moe_content(text: str) -> str:
    """去掉 <tool_call> 块与 think 块, 只返回可进入对话历史的内容."""
    text = GLM4_MOE_CALL_BLOCK.sub("", text)
    text = THINK_BLOCK.sub("", text)
    return text.strip()


class Glm4MoeStreamSplitter(ToolCallStreamSplitter):
    """GLM-4.5/4.6/4.7 的流式切分器: 切分逻辑与 hermes 完全一致(<tool_call> 与
    </tool_call> 都是单特殊 token), 只是块体是 XML 参数而不是 JSON —— 覆写块解析
    即可(见 ToolCallStreamSplitter._parse_block)。
    """

    def _parse_block(self, block: str) -> Dict[str, Any] | None:
        return _glm4_moe_call(block)


# ---------------------------------------------------------------------------
# 解析器分派与协议冲突回退(server 用)
# ---------------------------------------------------------------------------

def parse_output_tool_calls(text: str, parser: str,
                            schemas: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    """按解析器名解析输出文本里的工具调用(qwen 与 hermes 是同一协议).

    schemas: 函数名 -> parameters schema, 供各解析器修复模型把 array
    参数写成字符串列表的格式滑移; 为 None 时不做修复.
    """
    if parser == "llama3_json":
        return parse_llama3_json_tool_calls(text, schemas)
    if parser == "mistral":
        return parse_mistral_tool_calls(text, schemas)
    if parser == "glm4":
        return parse_glm4_tool_calls(text, schemas)
    if parser == "glm4_moe":
        return parse_glm4_moe_tool_calls(text, schemas)
    return parse_tool_calls(text, schemas)


def clean_output_content(text: str, parser: str) -> str:
    """按解析器名剥掉输出里的工具调用标记/think 块, 返回对话内容."""
    if parser == "llama3_json":
        return clean_llama3_json_content(text)
    if parser == "mistral":
        return clean_mistral_content(text)
    if parser == "glm4":
        return clean_glm4_content(text)
    if parser == "glm4_moe":
        return clean_glm4_moe_content(text)
    return clean_content(text)


def make_stream_splitter(parser: str, tool_names=None):
    """按解析器名构造流式切分器(server 用).

    tool_names: 可用工具名集合, 目前只有 glm4 切分器用(它的协议没有起始标记,
    需要白名单排除正文里同形的 "一行单词 + JSON 对象"); 其他协议忽略。
    """
    if parser == "llama3_json":
        return LlamaJsonStreamSplitter()
    if parser == "mistral":
        return MistralStreamSplitter()
    if parser == "glm4":
        return Glm4StreamSplitter(tool_names)
    if parser == "glm4_moe":
        return Glm4MoeStreamSplitter()
    return ToolCallStreamSplitter()


# 各流式切分器都实现同一组 push/flush 接口, 供 server 的类型标注使用
StreamSplitter = (ToolCallStreamSplitter | LlamaJsonStreamSplitter
                  | MistralStreamSplitter | Glm4StreamSplitter
                  | Glm4MoeStreamSplitter)


def clean_tool_call_prefix(text: str, parser: str,
                           schemas: Dict[str, Any] | None = None) -> str:
    """取第一个工具调用标记之前的正文, 供"有 tool_calls 时也返回 content"使用.

    刻意**不 strip**: 客户端把这段正文随 tool_calls 一起回填时模板会原样渲染,
    只有与模型实际生成的 token 逐位一致, 跨请求 KV 前缀才能覆盖整段输出
    (见 parse_model_output)。schemas 只被 glm4 用到: 它没有起始标记, 定位切口
    必须复用解析器的候选接受条件(见 clean_glm4_prefix)。
    """
    if parser == "glm4":
        return clean_glm4_prefix(text, schemas)
    marker = MISTRAL_TOOL_CALLS if parser == "mistral" else TOOL_CALL_START
    pos = text.find(marker)
    return THINK_BLOCK.sub("", text if pos < 0 else text[:pos])


# 有 tool_calls 时仍返回"调用前正文"的协议(与 vLLM 各 parser 一致):
#   hermes/qwen: content = 第一个 <tool_call> 之前的正文;
#   mistral    : content = 第一个 [TOOL_CALLS] 之前的正文;
#   glm4       : content = 第一个 "name\n{json}" 之前的正文;
#   glm4_moe   : content = 第一个 <tool_call> 之前的正文(XML 块有起止标记);
#   llama3_json: 返回 null(JSON 调用本身就是整条消息)。
_KEEP_PREFIX_CONTENT = ("hermes", "qwen", "mistral", "glm4", "glm4_moe")


def parse_model_output(text: str, parser: str, fallback_parser: str | None,
                       request_id: str | None = None,
                       schemas: Dict[str, Any] | None = None) -> Tuple[List[Dict[str, Any]], str | None]:
    """把模型可见输出解析成 (tool_calls, content), 含协议冲突回退.

    - 先按显式配置解析器解析; 解析不到且 fallback_parser 给出时, 按模型
      原生协议再解析一次 —— 显式配置了与模型家族不符的解析器(如 Llama 3.x
      配 hermes)时, 该解析器对模型输出永远解析不出结果, 不兜底的话工具调用
      会整段漏进 content, 客户端永远拿不到 tool_calls(只会看到原始文本);
    - **有 tool_calls 时是否保留正文跟随模型协议**(与 vLLM 相同): Mistral/Qwen
      系模型经常先输出一段推理正文再调工具, vLLM 的 mistral/hermes 解析器把
      这段正文放进 content(与本文件流式切分器的行为一致), 只有 llama3_json
      返回 null。LMInfer 早期对所有解析器都返回 null —— 代价不只是少返回一段
      文本: 客户端回填 assistant 消息时 content 是 null, 模板渲染出的消息从
      `[TOOL_CALLS]` 开始, 而模型实际生成的 token 从正文开始, 于是 token 级
      LCP 在正文处就断开, 每个含工具调用的回合都白白丢掉"正文 + 工具调用"的
      整段 KV 复用(实测 Ministral-3 的 research agent: 主 agent 复用率被压到
      55%~82%, 其中约 347 tok 的正文本来可以整段复用)。
    - schemas 供解析时做 schema 感知的参数修复(见 _repair_array_args),
      流式路径不做修复(切流分片无法整段重写 arguments).
    """
    calls = parse_output_tool_calls(text, parser, schemas)
    cleaner = parser
    if not calls and fallback_parser is not None:
        fb_calls = parse_output_tool_calls(text, fallback_parser, schemas)
        if fb_calls:
            logger.warning(
                "tool-call-parser=%s 未识别到工具调用%s, 按模型原生协议 %s "
                "解析出 %d 个(建议改用 --tool-call-parser auto 或 %s)",
                parser, f"(请求 {request_id})" if request_id else "",
                fallback_parser, len(fb_calls), fallback_parser)
            calls = fb_calls
        cleaner = fallback_parser
    if calls:
        content = (clean_tool_call_prefix(text, cleaner, schemas)
                   if cleaner in _KEEP_PREFIX_CONTENT else "")
    else:
        content = clean_output_content(text, cleaner)
    return calls, content or None
