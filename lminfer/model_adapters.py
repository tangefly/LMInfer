"""模型适配层: 识别模型家族, 统一"工具调用解析"与"模板渲染"的模型差异.

LMInfer 只依赖 transformers 的高层 API, 但不同模型家族在工具调用上有
两套完全不同的协议, 需要在这里显式适配:

- Qwen / Hermes 系: 模型输出 `<tool_call>{json}</tool_call>` 块(特殊 token),
  解析器是 toolcalls.py 的 hermes 解析器; 模板用 `<tool_response>` 包裹工具结果;
- Llama 3.x 系: 模型输出 `{"name": ..., "parameters": {...}}` 形式的 JSON
  (可能带 `<|python_tag|>` 前缀), 解析器是 llama3_json; 模板把 OpenAI 格式的
  arguments(JSON 字符串)渲染成对象、把工具结果渲染成 `{"output": ...}` 对象.

`--tool-call-parser` 的默认值 auto 在这里解析成具体解析器(依据 tokenizer 的
特殊 token 自动识别), 显式指定(hermes/qwen/llama3_json/none)则原样使用.
"""

from dataclasses import dataclass

TOOL_CALL_START = "<tool_call>"
LLAMA_PYTHON_TAG = "<|python_tag|>"


@dataclass
class ModelProfile:
    """一次服务启动解析出的模型适配参数."""

    tool_parser: str  # "hermes" | "llama3_json" | "none": 实际生效的工具调用解析器
    arguments_as_dict: bool  # True: 模板把 OpenAI arguments(JSON 字符串)当对象渲染。
                             # Llama 3.x 模板写 `tool_call.arguments | tojson`,
                             # 传 JSON 字符串会被加引号变成 "parameters": "{\"city\": ...}",
                             # 必须在渲染前把字符串还原成 dict, 才能渲染成合法对象;
    wrap_tool_output: bool   # True: 工具结果(content 字符串)渲染成 {"output": ...}。
                             # Llama 3.x 模板的 ipython 块对字符串直接 | tojson 会加引号,
                             # 包成对象后与模型训练时的工具结果格式一致;


def _has_special_token(tokenizer, token: str) -> bool:
    """token 是 tokenizer 的单个特殊 token(而不是被切分成多个普通 token)."""
    tid = tokenizer.convert_tokens_to_ids(token)
    return (isinstance(tid, int) and tid >= 0
            and tokenizer.convert_ids_to_tokens(tid) == token)


def resolve_tool_parser(configured: str, tokenizer) -> str:
    """把配置值解析成具体解析器: auto 依据 tokenizer 特殊 token 自动识别.

    - 有 `<tool_call>` 特殊 token(Qwen/Hermes 系) -> hermes 风格块解析;
    - 有 `<|python_tag|>` 特殊 token(Llama 3.x 系) -> llama3_json 风格 JSON 解析;
    - 都没有 -> none(关闭工具解析, 按普通文本返回).
    显式指定的 hermes/qwen/llama3_json/none 原样使用.
    """
    if configured in ("hermes", "qwen", "llama3_json", "none"):
        return configured
    if _has_special_token(tokenizer, TOOL_CALL_START):
        return "hermes"
    if _has_special_token(tokenizer, LLAMA_PYTHON_TAG):
        return "llama3_json"
    return "none"


def resolve_model_profile(configured_parser: str, tokenizer,
                          model_config=None) -> ModelProfile:
    """解析出本次服务实际使用的模型适配参数(见 ModelProfile)."""
    parser = resolve_tool_parser(configured_parser, tokenizer)
    llama3 = _has_special_token(tokenizer, LLAMA_PYTHON_TAG)
    return ModelProfile(
        tool_parser=parser,
        arguments_as_dict=llama3,
        wrap_tool_output=llama3,
    )
