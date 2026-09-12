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
  而不是 jinja 渲染(见 supports_chat_template);
- GLM-4 系(GLM-4-9B-0414 等): 模型输出 `name\n{json}`(函数名独占一行, 紧跟一个
  JSON 参数对象; 函数名前**可能带字面量 `<|assistant|>`**, 以 `<|observation|>`/
  `<|user|>` 结束), 解析器是 glm4; 消息协议用 assistant 的 `metadata` 字段承载
  函数名、content 承载参数 JSON, 工具结果用 `observation` 角色 —— 该 jinja 模板
  **不认** OpenAI 的 `tool` 角色与 `tool_calls` 字段, 直接透传会把工具调用整段
  丢掉(见 ModelProfile.tool_protocol);
- GLM-4.5/4.6/4.7 系 MoE(glm4_moe / glm4_moe_lite): 工具调用是 **XML 协议**
  `<tool_call>函数名<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
  (vLLM 的 glm45/glm47 同一个 parser), 与上面 glm4 的 `name\n{json}` 完全不同;
  模板**原生认 OpenAI 的 `tool_calls` / `tool` 角色**, 只是 arguments 必须是对象
  (见 ModelProfile.arguments_as_dict)。这一系的 tokenizer 里 `<tool_call>` 是单
  特殊 token, 按 token 探测会先命中 hermes(块体却是 XML 而非 JSON), 必须按
  config.model_type 抢先判定。GLM-4.7-Flash(glm4_moe_lite)还是 **MLA** 注意力:
  缓存里带位置信息的是 k_rot(value 槽), RoPE 布局见 RopeLayout.rotated_slot。

`--tool-call-parser` 的默认值 auto 在这里解析成具体解析器(依据 config 的
model_type 或 tokenizer 的特殊 token 自动识别), 显式指定
(hermes/qwen/llama3_json/mistral/glm4/none)则原样使用.
显式指定与模型家族协议冲突时(如 Llama 3.x 模型配 hermes), 启动时告警,
请求时先按显式配置解析、失败再按模型原生协议回退解析(见 ModelProfile).

本模块同时承载"模型怎么加载"的差异(load_text_model): 多模态包装(如
Mistral3ForConditionalGeneration)只取文本解码器, fine-grained FP8 权重在
transformers 后端反量化成普通线性层。这两件事都是模型家族知识, 不应散落在引擎里。
"前向要传哪些参数"同理(resolve_logits_kwargs): 只算末尾位置的 logits, 是 30B 级
权重 + 长 prompt 下不被临时张量顶爆显存的前提。
"""

import inspect
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
# GLM-4 系的工具结果角色标记: 模板把工具结果渲染成 `<|observation|>\n正文`,
# 正文后面紧跟下一个角色标记(通常是 add_generation_prompt 的 `<|assistant|>`)。
GLM_OBSERVATION = "<|observation|>"
GLM_ROLE_MARKERS = ("<|system|>", "<|user|>", "<|assistant|>", GLM_OBSERVATION)
# GLM-4 系(config.model_type): GLM-4-0414 这类 dense 模型的工具调用走 name\n{json}
# 文本协议, RoPE 是部分旋转 + 奇偶交错。刻意只列 "glm4": GLM-4.5/MoE(glm4_moe)
# 用的是 <tool_call><arg_key>... 的 XML 协议, 不能共用这套识别。
GLM4_MODEL_TYPES = ("glm4",)
# GLM-4.5/4.6/4.7 系 MoE: 工具调用是 <tool_call>函数名<arg_key>k</arg_key>
# <arg_value>v</arg_value></tool_call> 的 XML 协议(解析器 glm4_moe)。
GLM4_MOE_MODEL_TYPES = ("glm4_moe", "glm4_moe_lite")
# MLA(Multi-head Latent Attention)的家族: 缓存里存的是压缩潜向量(kv_lora_rank)
# 与共享的旋转键 k_rot(qk_rope_head_dim), 带位置信息的只有后者且落在 value 槽。
# 只有 glm4_moe_lite(GLM-4.7-Flash)在 transformers 里是 MLA 实现: glm4_moe
# (GLM-4.5)走的是普通 q/k/v 投影 + 经典 (key, value) 缓存, 不能按 MLA 处理。
MLA_MODEL_TYPES = ("glm4_moe_lite",)
# --tool-call-parser 的 vLLM 兼容别名: vLLM 把 glm45/glm47 都注册到同一个
# glm47_moe parser(即本仓库的 glm4_moe), 从 vLLM 命令直接抄过来要能用。
TOOL_PARSER_ALIASES = {"glm45": "glm4_moe", "glm47": "glm4_moe"}


@dataclass(frozen=True)
class ToolResultWrapper:
    """chat template 渲染工具结果时用的结构锚点(拼接模式定位子 agent 输出正文).

    两种形态:
    - **显式闭合标记**(Qwen 的 `<tool_response>`/`</tool_response>`、Mistral 的
      `[TOOL_RESULTS]`/`[/TOOL_RESULTS]`): 正文位于一对特殊 token 之间;
    - **终止标记集合**(GLM-4): 模板只写 `<|observation|>\n{{ content }}`, 没有
      闭合标记, 正文一直延伸到下一个角色标记(`<|user|>` / `<|assistant|>` /
      `<|system|>` / 下一个 `<|observation|>`)。

    两端的标记都必须是**单个特殊 token**(id 与上下文无关), 拼接模式才能直接按
    id 定位窗口。
    """

    open_marker: str
    close_marker: str | None = None
    terminators: tuple[str, ...] = ()


# 工具结果包裹标记候选: 拼接模式(--reuse-agent-kv-append)据此在渲染后的 prompt 中
# 定位子 agent 输出正文的窗口。按模型家族协议探测, 取 tokenizer 里真实存在的
# 单 token 特殊标记(见 resolve_tool_result_wrapper)。
TOOL_RESULT_WRAPPERS = (
    ToolResultWrapper(TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE),        # Qwen/Hermes
    ToolResultWrapper(MISTRAL_TOOL_RESULTS, MISTRAL_TOOL_RESULTS_END),  # Mistral
    ToolResultWrapper(GLM_OBSERVATION, terminators=GLM_ROLE_MARKERS),   # GLM-4
)

# Mistral 的 tool_call id 必须满足 mistral-common 的校验(a-z/A-Z/0-9, 长度 9),
# 否则模板渲染抛 InvalidFunctionCallException、工具结果回填不进 prompt;
# id 的具体生成在 toolcalls.py(该协议解析器里)。
FP8_KERNEL_NAME = "kernels"  # fine-grained FP8 前向所需的 Triton 内核包


@dataclass
class ModelProfile:
    """一次服务启动解析出的模型适配参数."""

    tool_parser: str  # "hermes" | "llama3_json" | "mistral" | "glm4" | "none":
                      # 实际生效的工具调用解析器
    native_parser: str  # 模型家族原生协议解析器(auto 的识别结果), 冲突回退用
    arguments_as_dict: bool  # True: 模板把 OpenAI arguments(JSON 字符串)当对象渲染。
                             # Llama 3.x 模板写 `tool_call.arguments | tojson`,
                             # 传 JSON 字符串会被加引号变成 "parameters": "{\"city\": ...}",
                             # 必须在渲染前把字符串还原成 dict, 才能渲染成合法对象;
    wrap_tool_output: bool   # True: 工具结果(content 字符串)渲染成 {"output": ...}。
                             # Llama 3.x 模板的 ipython 块对字符串直接 | tojson 会加引号,
                             # 包成对象后与模型训练时的工具结果格式一致;
    tool_protocol: str = "openai"  # "openai": OpenAI 的 tool_calls/tool 消息原样渲染
                                   # (模板自己认识这两个字段); "glm4": GLM-4 模板只认
                                   # assistant.metadata 与 observation 角色, 渲染前必须
                                   # 把 OpenAI 消息翻译成原生形态(见 server._message_dicts)

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


def _is_glm4_family(model_config, tokenizer) -> bool:
    """是否 GLM-4 系(工具调用走 `name\\n{json}` 文本协议).

    优先看 config.model_type(权威): glm4 是 GLM-4-0414 这类 dense 模型的
    model_type。只有在拿不到 config 时(单元测试的桩 / 未传 model_config)才退化到
    tokenizer 的特殊 token 探测 —— `<|observation|>` 是 GLM chat 系独有的角色标记。
    """
    model_type = (str(getattr(model_config, "model_type", "") or "")
                  if model_config is not None else "")
    if model_type:
        return model_type.lower() in GLM4_MODEL_TYPES
    return _has_special_token(tokenizer, GLM_OBSERVATION)


def _is_glm4_moe_family(model_config) -> bool:
    """是否 GLM-4.5/4.6/4.7 系 MoE(工具调用是 `<tool_call>...<arg_key>` XML 协议).

    只看 config.model_type: 这一系的 tokenizer 里 `<tool_call>` 是**单特殊 token**,
    按 token 探测会先命中 hermes —— 而 hermes 的块体是 JSON, 对 XML 块 `json.loads`
    必然失败, 结果是工具调用被 cleaner 整段删掉、客户端拿到的 content 里连原文都
    没有(实测)。所以家族判定必须在 token 探测之前。
    """
    model_type = (str(getattr(model_config, "model_type", "") or "")
                  if model_config is not None else "")
    return model_type.lower() in GLM4_MOE_MODEL_TYPES


def resolve_tool_parser(configured: str, tokenizer, model_config=None) -> str:
    """把配置值解析成具体解析器: auto 依据 config/tokenizer 自动识别.

    - config.model_type 是 glm4_moe/glm4_moe_lite(GLM-4.5/4.6/4.7 系) -> glm4_moe
      的 XML 解析(必须最先判: 这些 tokenizer 里 `<tool_call>` 也是单 token);
    - 有 `<tool_call>` 特殊 token(Qwen/Hermes 系) -> hermes 风格块解析;
    - 有 `<|python_tag|>` 特殊 token(Llama 3.x 系) -> llama3_json 风格 JSON 解析;
    - 有 `[TOOL_CALLS]` 特殊 token(Mistral v11+ 系) -> mistral 风格解析;
    - config.model_type 是 glm4(GLM-4 系) -> glm4 的 `name\\n{json}` 解析;
    - 都没有 -> none(关闭工具解析, 按普通文本返回).
    显式指定的 hermes/qwen/llama3_json/mistral/glm4/glm4_moe/none 原样使用;
    vLLM 的 glm45/glm47 当作 glm4_moe 的别名(同一套 XML 协议)。
    """
    configured = TOOL_PARSER_ALIASES.get(configured, configured)
    if configured in ("hermes", "qwen", "llama3_json", "mistral", "glm4", "glm4_moe",
                      "none"):
        return configured
    if _is_glm4_moe_family(model_config):
        return "glm4_moe"
    if _has_special_token(tokenizer, TOOL_CALL_START):
        return "hermes"
    if _has_special_token(tokenizer, LLAMA_PYTHON_TAG):
        return "llama3_json"
    if _has_special_token(tokenizer, MISTRAL_TOOL_CALLS):
        return "mistral"
    if _is_glm4_family(model_config, tokenizer):
        return "glm4"
    return "none"


def _wrapper_is_usable(tokenizer, wrapper: ToolResultWrapper) -> bool:
    """包裹标记(以及终止标记集合)是否都由单个特殊 token 组成."""
    markers = [wrapper.open_marker]
    if wrapper.close_marker is not None:
        markers.append(wrapper.close_marker)
    markers.extend(wrapper.terminators)
    return all(_has_special_token(tokenizer, m) for m in markers)


def resolve_tool_result_wrapper(tokenizer) -> ToolResultWrapper | None:
    """探测该 tokenizer 用哪组标记包裹工具结果(拼接模式定位正文用).

    标记都必须是**单个特殊 token**(id 与上下文无关), 拼接模式才能在渲染后的
    prompt 里直接按 id 定位窗口。都探测不到时返回 None, 拼接模式自动不可用, 由
    SessionKVStore 回退 LCP 复用(见 kvcache.py)。
    """
    for wrapper in TOOL_RESULT_WRAPPERS:
        if _wrapper_is_usable(tokenizer, wrapper):
            return wrapper
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
    parser = resolve_tool_parser(configured_parser, tokenizer, model_config)
    native = resolve_tool_parser("auto", tokenizer, model_config)
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
    # GLM-4.5/4.7 系的模板用 `{% for k, v in tc.arguments.items() %}` 渲染工具调用,
    # arguments 是 JSON 字符串时直接抛 UndefinedError(→ 400), 与 Llama 3.x 同样
    # 需要在渲染前把字符串还原成 dict(见 server._adapt_tool_calls)。
    glm4_moe = native == "glm4_moe"
    return ModelProfile(
        tool_parser=parser,
        native_parser=native,
        arguments_as_dict=llama3 or glm4_moe,
        wrap_tool_output=llama3,
        # 工具消息的**渲染**协议由模型家族决定(与配置的解析器无关): GLM-4 的
        # jinja 模板只认 assistant.metadata / observation, 不认 OpenAI 的
        # tool_calls / tool 角色 —— 必须按原生形态翻译, 否则工具调用整段丢失.
        tool_protocol="glm4" if native == "glm4" else "openai",
    )


# ---------------------------------------------------------------------------
# 模型家族知识: RoPE 布局(拼接模式的 K 位置重映射用)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RopeLayout:
    """文本解码器 RoPE 的旋转布局(位置差 rotation 必须与模型前向逐位一致).

    - rotary_dim : 实际参与旋转的维度数(<= head_dim)。GLM-4 声明
      `partial_rotary_factor: 0.5`, 即 128 维 head 只有前 64 维旋转, 后 64 维
      原样透传(`apply_rotary_pos_emb` 里 q_pass/k_pass);
    - interleaved: rotate_half 的配对方式。False = Qwen/Llama/Mistral 的
      "前后对半"(`cat((-x2, x1))`, 配对 (i, i+d/2)); True = GLM/GPT-NeoX 的
      "奇偶交错"(`stack((-x2, x1), -1).flatten()`, 配对 (2i, 2i+1)), 且 cos/sin
      要用 `repeat_interleave(2)` 展开;
    - rotated_slot: **KV cache 里带位置信息的那个槽**。普通 GQA/MHA 是 key 槽
      (K 旋转、V 不转); MLA(GLM-4.7-Flash)把旋转后的共享键 k_rot 存在 value 槽
      (实测 `layer.values = [b, 1, s, qk_rope_head_dim]`, `layer.keys` 是位置
      无关的压缩潜向量 kv_lora_rank), 位置重映射必须按这个槽来选张量。
    """

    rotary_dim: int
    interleaved: bool = False
    rotated_slot: str = "keys"   # "keys" | "values"


def _interleaved_rope(config, model_type: str) -> bool:
    """**KV cache 里的** RoPE 张量是否用奇偶交错配对(GLM/GPT-NeoX 式).

    按模型家族判定: GLM-4-0414 的 attention 用交错式 rotate_half(cos/sin 也要
    `repeat_interleave(2)` 展开); Qwen/Llama/Mistral 与 GLM-4.5 的普通 q/k/v 实现
    都是前后对半。

    注意**不能**拿 GLM-4.7 的 `config.rope_interleave=True` 判定这里: 那个标志
    描述的是"模型读入的原始 k_rot 里复数是 (2i, 2i+1) 排列", 而它的
    `apply_rotary_pos_emb_interleave` 在写缓存**之前**把结果重排成了前后对半配对
    (实测: 用模型自己的 apply 在目标位置旋转, 与"对缓存张量做前后对半 delta 旋转"
    相差 3.6e-07, 用交错式则偏差 4.9 —— 而 k_rot 自身的量级是 3.0)。拼接模式
    旋转的是缓存里的张量, 必须按缓存的布局来。
    """
    return model_type.lower() in GLM4_MODEL_TYPES


def resolve_rope_layout(config, head_dim: int | None = None) -> RopeLayout:
    """从文本 config 解析 RoPE 旋转布局.

    `partial_rotary_factor` 出现在 config.rope_parameters(GLM-4-0414 实测)或
    顶层字段(旧版 Llama 实现), 两处都探测。交错式 rotate_half 由模型家族决定,
    无法从 rope 模块的返回值区分(它们都返回 `cat((freqs, freqs))`, 差别只在
    attention 里的 apply), 因此按 model_type 判定(见 _interleaved_rope)。
    MLA(glm4_moe_lite)另走一支: 参与旋转的只有 k_rot, 宽度是 `qk_rope_head_dim`
    而不是传进来的 head_dim(key 槽宽度 kv_lora_rank, 对 MLA 没有意义)。
    """
    model_type = (str(getattr(config, "model_type", "") or "") if config is not None else "")
    params = getattr(config, "rope_parameters", None) or {}
    factor = params.get("partial_rotary_factor")
    if factor is None:
        factor = getattr(config, "partial_rotary_factor", 1.0)
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        factor = 1.0
    if model_type.lower() in MLA_MODEL_TYPES:
        rope_dim = int(getattr(config, "qk_rope_head_dim", 0) or 0) or (head_dim or 0)
        rotary_dim = max(1, min(int(rope_dim * factor), rope_dim)) if rope_dim else 1
        # 配对是前后对半(缓存的 k_rot 经过 apply_rotary_pos_emb_interleave 的重排),
        # 只是张量落在 value 槽 —— 见 _interleaved_rope 的说明
        return RopeLayout(rotary_dim=rotary_dim, interleaved=False,
                          rotated_slot="values")
    if head_dim is None:
        head_dim = (getattr(config, "head_dim", None)
                    or config.hidden_size // config.num_attention_heads)
    return RopeLayout(rotary_dim=max(1, min(int(head_dim * factor), head_dim)),
                      interleaved=_interleaved_rope(config, model_type))


def kv_bytes_per_token(config, dtype_size: int) -> int:
    """每生成 1 个 token、全部层新增的 KV 显存字节数(引擎日志与 /v1/stats 用).

    普通 GQA/MHA: `2(K+V) × 层数 × KV头数 × head_dim × 每元素字节数`;
    MLA(glm4_moe_lite): 缓存的是压缩潜向量 —— 每层每 token 只有
    `kv_lora_rank + qk_rope_head_dim` 个数(实测 `layer.keys` 512 + `layer.values`
    64), 与注意力头数无关。按通用公式算会高估 4 倍以上。
    """
    num_layers = config.num_hidden_layers
    model_type = (str(getattr(config, "model_type", "") or "") if config is not None else "")
    if model_type.lower() in MLA_MODEL_TYPES:
        per_token = (int(getattr(config, "kv_lora_rank", 0) or 0)
                     + int(getattr(config, "qk_rope_head_dim", 0) or 0))
        return num_layers * per_token * dtype_size
    num_kv_heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
    return 2 * num_layers * num_kv_heads * head_dim * dtype_size


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


def resolve_logits_kwargs(model) -> dict:
    """前向时只计算末尾位置的 logits(transformers 5.x 的 `logits_to_keep=1`).

    引擎的 prefill/decode 与精确修复(`context_repair.exact_prefill`)都只取
    `out.logits[:, -1]`, 但默认实现会为**所有**位置算 logits —— 30B 级模型
    (Qwen3-30B-A3B 词表 151936)在 40K prompt 下那是
    `40960 x 151936 x 2B ≈ 11.6 GiB` 的临时张量, 与 58 GiB 权重叠加直接 OOM。
    只在 forward 签名真的支持该参数时传(多模态包装/自定义模型不支持就保持原行为)。
    """
    try:
        if "logits_to_keep" in inspect.signature(model.forward).parameters:
            return {"logits_to_keep": 1}
    except (TypeError, ValueError):  # 拿不到签名的包装模型: 保持默认行为
        pass
    return {}


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
