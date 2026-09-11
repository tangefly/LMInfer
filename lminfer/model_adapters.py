"""模型适配层: 识别模型家族, 统一"工具调用解析""模板渲染"与"模型加载"的差异.

LMInfer 只依赖 transformers 的高层 API, 但不同模型家族在工具调用上有
三套完全不同的协议, 需要在这里显式适配:

- Qwen / Hermes 系: 模型输出 `<tool_call>{json}</tool_call>` 块(特殊 token),
  解析器是 toolcalls.py 的 hermes 解析器; 模板用 `<tool_response>` 包裹工具结果;
- Llama 3.x 系: 模型输出 `{"name": ..., "parameters": {...}}` 形式的 JSON
  (可能带 `<|python_tag|>` 前缀), 解析器是 llama3_json; 模板把 OpenAI 格式的
  arguments(JSON 字符串)渲染成对象、把工具结果渲染成 `{"output": ...}` 对象;
- Mistral 系(Ministral 3 / Mistral-Small 等 v11+ tokenizer): 模型输出
  `[TOOL_CALLS]name[ARGS]{json}`(可选 `[CALL_ID]<id>`), 解析器是 mistral;
  模板用 `[TOOL_RESULTS]...[/TOOL_RESULTS]` 包裹工具结果, 且由 mistral-common
  而不是 jinja 渲染(见 supports_chat_template).

`--tool-call-parser` 的默认值 auto 在这里解析成具体解析器(依据 tokenizer 的
特殊 token 自动识别), 显式指定(hermes/qwen/llama3_json/mistral/none)则原样使用.
显式指定与模型家族协议冲突时(如 Llama 3.x 模型配 hermes), 启动时告警,
请求时先按显式配置解析、失败再按模型原生协议回退解析(见 ModelProfile).

本模块同时承载"模型怎么加载"的差异(load_text_model): 多模态包装(如
Mistral3ForConditionalGeneration)只取文本解码器, fine-grained FP8 权重在
transformers 后端反量化成普通线性层。这两件事都是模型家族知识, 不应散落在引擎里。
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("lminfer")

TOOL_CALL_START = "<tool_call>"
LLAMA_PYTHON_TAG = "<|python_tag|>"
TOOL_RESPONSE_OPEN = "<tool_response>"   # Qwen/Hermes 模板包裹工具结果的标记
TOOL_RESPONSE_CLOSE = "</tool_response>"
MISTRAL_TOOL_CALLS = "[TOOL_CALLS]"  # Mistral v11+ 工具调用起始特殊 token
MISTRAL_TOOL_RESULTS = "[TOOL_RESULTS]"
MISTRAL_TOOL_RESULTS_END = "[/TOOL_RESULTS]"

# 工具结果包裹标记候选: 拼接模式(--reuse-agent-kv-append)据此在渲染后的 prompt 中
# 定位子 agent 输出正文的窗口。按模型家族协议探测, 取 tokenizer 里真实存在的
# 单 token 特殊标记(见 resolve_tool_result_wrapper)。
TOOL_RESULT_WRAPPERS = (
    (TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE),        # Qwen/Hermes
    (MISTRAL_TOOL_RESULTS, MISTRAL_TOOL_RESULTS_END),  # Mistral
)

# Mistral 的 tool_call id 必须满足 mistral-common 的校验(a-z/A-Z/0-9, 长度 9),
# 否则模板渲染抛 InvalidFunctionCallException、工具结果回填不进 prompt;
# id 的具体生成在 toolcalls.py(该协议解析器里)。
FP8_KERNEL_NAME = "kernels"  # fine-grained FP8 前向所需的 Triton 内核包


@dataclass
class ModelProfile:
    """一次服务启动解析出的模型适配参数."""

    tool_parser: str  # "hermes" | "llama3_json" | "none": 实际生效的工具调用解析器
    native_parser: str  # 模型家族原生协议解析器(auto 的识别结果), 冲突回退用
    arguments_as_dict: bool  # True: 模板把 OpenAI arguments(JSON 字符串)当对象渲染。
                             # Llama 3.x 模板写 `tool_call.arguments | tojson`,
                             # 传 JSON 字符串会被加引号变成 "parameters": "{\"city\": ...}",
                             # 必须在渲染前把字符串还原成 dict, 才能渲染成合法对象;
    wrap_tool_output: bool   # True: 工具结果(content 字符串)渲染成 {"output": ...}。
                             # Llama 3.x 模板的 ipython 块对字符串直接 | tojson 会加引号,
                             # 包成对象后与模型训练时的工具结果格式一致;

    @property
    def fallback_parser(self) -> str | None:
        """显式配置与模型家族协议冲突时的回退解析器; 无冲突返回 None.

        配置了与模型家族不符的解析器(如 Llama 3.x 模型配 hermes)时, 该解析器
        对模型输出永远解析不出结果, 工具调用会静默丢失 —— 请求时先按显式
        配置解析, 解析不到再按原生协议解析一次(见 toolcalls.parse_model_output),
        显式配置不被丢弃, 工具调用也不丢.
        """
        if (self.tool_parser in ("none", self.native_parser)
                or self.native_parser == "none"):
            return None
        return self.native_parser


def _has_special_token(tokenizer, token: str) -> bool:
    """token 是 tokenizer 的单个特殊 token(而不是被切分成多个普通 token)."""
    try:
        tid = tokenizer.convert_tokens_to_ids(token)
    except Exception:  # 桩 tokenizer / 词表里没有该 token 时的各种异常都算"没有"
        return False
    return (isinstance(tid, int) and tid >= 0
            and tokenizer.convert_ids_to_tokens(tid) == token)


def resolve_tool_parser(configured: str, tokenizer) -> str:
    """把配置值解析成具体解析器: auto 依据 tokenizer 特殊 token 自动识别.

    - 有 `<tool_call>` 特殊 token(Qwen/Hermes 系) -> hermes 风格块解析;
    - 有 `<|python_tag|>` 特殊 token(Llama 3.x 系) -> llama3_json 风格 JSON 解析;
    - 有 `[TOOL_CALLS]` 特殊 token(Mistral v11+ 系) -> mistral 风格解析;
    - 都没有 -> none(关闭工具解析, 按普通文本返回).
    显式指定的 hermes/qwen/llama3_json/mistral/none 原样使用.
    """
    if configured in ("hermes", "qwen", "llama3_json", "mistral", "none"):
        return configured
    if _has_special_token(tokenizer, TOOL_CALL_START):
        return "hermes"
    if _has_special_token(tokenizer, LLAMA_PYTHON_TAG):
        return "llama3_json"
    if _has_special_token(tokenizer, MISTRAL_TOOL_CALLS):
        return "mistral"
    return "none"


def resolve_tool_result_wrapper(tokenizer) -> tuple[str, str] | None:
    """探测该 tokenizer 用哪一对标记包裹工具结果(拼接模式定位正文用).

    两端的标记都必须是**单个特殊 token**(id 与上下文无关), 拼接模式才能在渲染后的
    prompt 里直接按 id 定位窗口。都探测不到时返回 None, 拼接模式自动不可用, 由
    SessionKVStore 回退 LCP 复用(见 kvcache.py)。
    """
    for open_marker, close_marker in TOOL_RESULT_WRAPPERS:
        if (_has_special_token(tokenizer, open_marker)
                and _has_special_token(tokenizer, close_marker)):
            return open_marker, close_marker
    return None


def supports_chat_template(tokenizer) -> bool:
    """该 tokenizer 能否把消息列表渲染成 prompt.

    Mistral 系由 transformers 的 MistralCommonBackend 走 mistral-common 渲染,
    它没有 `chat_template` 属性(模型自带的 chat_template.jinja 不会被加载),
    但 apply_chat_template 完全可用 —— 不能按 `chat_template is None` 判定为
    "模型没有模板"。
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        return True
    return type(tokenizer).__name__ == "MistralCommonBackend"


def resolve_model_profile(configured_parser: str, tokenizer,
                          model_config=None) -> ModelProfile:
    """解析出本次服务实际使用的模型适配参数(见 ModelProfile)."""
    parser = resolve_tool_parser(configured_parser, tokenizer)
    native = resolve_tool_parser("auto", tokenizer)
    if parser != native and native != "none" and parser != "none":
        # 显式解析器与模型家族协议冲突: 解析器对模型输出永远解析不出结果,
        # 工具调用会整段漏进 content, 客户端拿不到 tool_calls(只会看到原始
        # 文本, 多轮 agent 场景直接退化成死循环). 请求时按原生协议兜底解析.
        logger.warning(
            "--tool-call-parser %s 与模型家族协议(%s)不匹配: 该解析器识别不了 "
            "这个模型的工具调用输出(如 Llama 3.x 的 <|python_tag|> JSON 不会被 "
            "hermes 块解析器匹配). 请求时会按模型原生协议回退解析; 建议改用 "
            "--tool-call-parser auto 或 %s",
            configured_parser, native, native)
    llama3 = _has_special_token(tokenizer, LLAMA_PYTHON_TAG)
    return ModelProfile(
        tool_parser=parser,
        native_parser=native,
        arguments_as_dict=llama3,
        wrap_tool_output=llama3,
    )


# ---------------------------------------------------------------------------
# 模型加载适配(transformers 后端)
# ---------------------------------------------------------------------------

# 文本解码器 RoPE 模块的典型位置, 按顺序探测(拼接模式的 RoPE rebase 要用模型
# 自己的逆频率, 不能假设 theta 直接开方 —— 见 kvcache.rebase_rope_cache)
_ROTARY_PATHS = (
    ("model", "rotary_emb"),                   # 纯文本 CausalLM(Qwen3/Llama/Ministral3)
    ("model", "language_model", "rotary_emb"),  # 多模态包装(Mistral3 等)的文本塔
    ("language_model", "rotary_emb"),
)


def resolve_rotary_emb(model):
    """定位文本解码器的 RoPE 模块; 找不到返回 None(回退默认 RoPE 公式).

    必须取**文本塔**的 RoPE: 多模态模型的视觉塔可能自带一套 RoPE(如 Pixtral),
    按名字全局扫描会拿错。所以先按确定路径找, 最后才退化到扫描。
    """
    for path in _ROTARY_PATHS:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    for name, module in model.named_modules():
        if name.endswith("rotary_emb"):
            return module
    return None


def _is_finegrained_fp8(config) -> bool:
    """checkpoint 是否声明了 fine-grained FP8 量化(需要 kernels 包才能前向)."""
    quant = getattr(config, "quantization_config", None)
    if quant is None:
        text = getattr(config, "text_config", None)
        quant = getattr(text, "quantization_config", None) if text is not None else None
    if isinstance(quant, dict):
        return str(quant.get("quant_method", "")).lower() == "fp8"
    method = getattr(quant, "quant_method", None)
    return method is not None and str(getattr(method, "value", method)).lower() == "fp8"


def _fp8_kernel_available() -> bool:
    import importlib.util
    return importlib.util.find_spec(FP8_KERNEL_NAME) is not None


def _supports_causal_lm(config) -> bool:
    """AutoModelForCausalLM 是否认识这个 config(多模态包装的 config 不认识)."""
    try:
        from transformers import AutoModelForCausalLM
        return type(config) in AutoModelForCausalLM._model_mapping
    except Exception:
        arch = ((getattr(config, "architectures", None) or [""])[0]) or ""
        return "ConditionalGeneration" not in arch


def load_text_model(model_path: str, *, dtype, device_map="auto",
                    trust_remote_code: bool = False, attn_implementation=None,
                    dequantize_fp8: bool | None = None):
    """加载 LMInfer 生成循环用的文本因果 LM.

    返回 (model, text_config, notes)。notes 是给启动日志用的说明文本列表。

    需要适配的两类模型:
    - **多模态包装**: checkpoint 的架构是 `*ForConditionalGeneration`(如
      Ministral-3 的 Mistral3ForConditionalGeneration), AutoModelForCausalLM
      不认识它。这里用 AutoModelForImageTextToText 加载, 只跑文本路径(不传
      pixel_values), 并返回它的文本 config —— 引擎的 KV 形状/位置/RoPE 计算
      全部基于文本塔, 必须用 text_config 而不是顶层 config(顶层 config 连
      num_hidden_layers/rope_parameters 都没有)。
    - **fine-grained FP8 权重**: 前向依赖 `kernels` 包提供的 Triton 内核; 内核
      不可用时在加载期反量化成普通线性层(纯 transformers 路径, 与 Qwen3 等
      模型一致), 保证 KV 切片/RoPE rebase 都在同一种 dtype 上做。
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    notes: list[str] = []
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    kwargs = dict(torch_dtype=dtype, device_map=device_map,
                  trust_remote_code=trust_remote_code,
                  attn_implementation=attn_implementation)
    if _is_finegrained_fp8(config):
        use_kernel = None if dequantize_fp8 is None else not dequantize_fp8
        if use_kernel is None:
            use_kernel = _fp8_kernel_available()
        if use_kernel:
            notes.append("fine-grained FP8 权重: 使用 kernels 内核原生前向")
        else:
            from transformers import FineGrainedFP8Config
            # 保留 checkpoint 自己的量化参数(per-tensor 时 weight_block_size 为
            # None), 只打开加载期反量化
            source = getattr(config, "quantization_config", None) or {}
            kwargs["quantization_config"] = FineGrainedFP8Config(
                activation_scheme=(source.get("activation_scheme", "dynamic")
                                   if isinstance(source, dict) else "dynamic"),
                weight_block_size=(source.get("weight_block_size")
                                   if isinstance(source, dict) else None),
                modules_to_not_convert=(source.get("modules_to_not_convert")
                                        if isinstance(source, dict) else None),
                dequantize=True,
            )
            reason = ("--no-dequantize-fp8" if dequantize_fp8 is False
                      else f"未安装 {FP8_KERNEL_NAME} 包")
            notes.append(f"fine-grained FP8 权重: {reason}, 加载期反量化成普通线性层")

    if _supports_causal_lm(config):
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        text_config = model.config.get_text_config(decoder=True)
        return model, text_config, notes

    from transformers import AutoModelForImageTextToText
    notes.append("多模态 checkpoint: 用 AutoModelForImageTextToText 加载, "
                 "只使用文本塔(KV 复用/拼接都在文本塔上)")
    model = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    text_config = model.config.get_text_config(decoder=True)
    return model, text_config, notes
