# LMInfer

现已支持实验性 `--backend vllm`：LMInfer 保留 agent KV 复用层，
vLLM 执行单卡 dense Qwen3 的分段 prefill、KV 拼接、首尾重算和 decode。
启动方式、实现边界及真实 GPU 验收见 [vLLM 后端说明](docs/vllm_backend.md)。
默认仍使用下文介绍的 Transformers 后端。

一个刻意保持**朴素**（naive）的 LLM 推理服务，用于学习与验证 **KV Cache** 相关理论。
常规生成使用 transformers 的高层 API（模型加载、`DynamicCache`、采样 warper、chat template），
实验性上下文 KV 修复增加了 Qwen3 按层执行路径，
提供 **vLLM 风格的启动命令** 与 **OpenAI 兼容的 HTTP 接口**。

> vLLM 太复杂？先跑通这个精简版本，把 prefill / decode / KV cache / 并发这几件事
> 看明白，再回去读 vLLM 的 scheduler 和 paged attention 会轻松很多。

## 设计理念（为什么这样写）

| 关注点 | 朴素做法 | 对比 vLLM |
|---|---|---|
| 批处理 | 每个请求独占 batch=1，**不拼接不 padding**，mask 恒为全 1 | 动态 batch、padding、变长处理 |
| KV cache | 完全交给 `transformers.DynamicCache`，只传递对象 | 自研 PagedAttention + BlockManager |
| 并发 | `ThreadPoolExecutor(max_num_seqs)`，每个请求一个线程跑生成循环 | 事件循环 + scheduler + worker 分层 |
| 显存管理 | 动态分配，不做预分配 | `--gpu-memory-utilization` 预分 KV 池 |
| 采样 | 复用 transformers 的 LogitsProcessor | 自研 sampler |

三个关键决策：

1. **batch=1**：这是 5.x 里语义最安全的用法——attention mask 全 1、position_ids 用默认值，
   不需要处理任何 padding 边界。代价是并发请求之间没有共享 decode 步，这就是
   "连续批处理" 被省掉的部分（理解之后可以自己加）。
2. **KV cache 交给 transformers**：`model.forward(past_key_values=cache)` 之后，
   新 token 的 K/V 由模型内部追加进 cache。我们只做两件事——prefill 时传入空 cache，
   decode 时把 cache 传回去。这正是研究 KV cache 最关心的部分，全部显式可见。
3. **并发 = 线程池**：多个请求在不同线程里各自跑生成循环，GPU 计算天然串行化。
   这就是"朴素版的连续批处理"，也是理解 vLLM 事件循环的好起点。

## 目录结构

```
LMInfer/
├── lminfer/                  # 核心包
│   ├── cli.py                # lminfer serve / chat 命令行入口
│   ├── config.py             # EngineConfig / SamplingParams
│   ├── engine.py             # ★ 核心: prefill + decode 生成循环, KV cache 显式可见
│   ├── kvcache.py            # ★ agent 会话间跨请求 KV 前缀复用(简单 KV Cache 系统)
│   ├── schemas.py            # OpenAI 兼容请求模型
│   ├── sessions.py           # agent 会话注册表(trace 追踪 / token 累计)
│   ├── toolcalls.py          # Qwen 风格 <tool_call> 解析
│   └── server.py             # FastAPI 路由 + SSE 流式
├── examples/
│   ├── client.py             # 客户端示例(chat/completions, 流式/非流式)
│   └── bench.py              # 并发压测(TTFT / TPOT / 吞吐)
├── experiments/
│   ├── kv_cache_compare.py   # ★ 理论验证实验: KV cache 开/关对比
│   └── agent_kv_reuse.py     # ★ 理论验证实验: agent 会话间 KV 前缀复用开/关对比
└── README.md
```

## 快速开始

```bash
pip install -e .

# 启动服务(两种写法都与 vLLM 一致: 位置参数 或 --model)
lminfer serve /home/tanger/workspace/models/Qwen2.5-7B-Instruct --port 8000
lminfer serve --model /home/tanger/workspace/models/Qwen3-0.6B --max-num-seqs 4

# Llama 3.1 系(工具调用 JSON 格式自动识别, 无需额外参数)
lminfer serve /home/tanger/workspace/models/Meta-Llama-3.1-8B-Instruct --port 8000

# Mistral 系 / Ministral-3(FP8 多模态 checkpoint; 工具调用 [TOOL_CALLS]name[ARGS]{json})
lminfer serve /home/tanger/workspace/models/Ministral-3-8B-Instruct-2512 \
    --served-model-name Ministral-3-8B --max-model-len 40960 \
    --reuse-agent-kv-append --graft-rope-rebase \
    --repair-window-begin 0.1 --repair-window-end 0.1 \
    --enable-auto-tool-choice --port 8000

# GLM-4 系 / GLM-4-9B-0414(工具调用 name\n{json}, 工具结果角色 observation)
lminfer serve /home/tanger/workspace/models/GLM-4-9B-0414 \
    --served-model-name GLM-4-9B-0414 --max-model-len 32768 \
    --reuse-agent-kv-append --graft-rope-rebase \
    --repair-window-begin 0.1 --repair-window-end 0.1 \
    --enable-auto-tool-choice --port 8000

# 也可以不带参数安装直接运行
python -m lminfer serve /home/tanger/workspace/models/Qwen3-0.6B
```

`--served-model-name` 与 vLLM 语义一致（覆盖对外暴露的模型名）；
`--gpu-memory-utilization`、`--tensor-parallel-size`、`--kv-transfer-config`
等参数会被接受，但朴素实现中不生效（启动时会打印 WARNING 说明原因）。
`--tool-call-parser` 与 `--enable-auto-tool-choice` 真实生效，
见下文 [工具调用](#工具调用原生-tool-call)。
模型家族差异（多模态包装 / FP8 权重）会自动适配，见
[模型适配](#模型适配qwen--llama--mistral--多模态--fp8)。

`--reuse-agent-kv` 是 agent 模式下的跨请求 KV 前缀复用开关（默认关闭），
详见下文 [跨请求 KV 复用](#跨请求-kv-复用--reuse-agent-kv)。

验证服务：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

# 对话(流式)
curl -N http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "用一句话介绍大语言模型"}], "stream": true}'

# 文本补全
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32, "temperature": 0}'

# 客户端示例 / 交互式对话 / 压测
python examples/client.py --stream "讲个冷笑话"
lminfer chat --model /home/tanger/workspace/models/Qwen3-0.6B
python examples/bench.py --prompts 8 --max-tokens 64
```

支持的采样参数：`temperature`（0=贪心）、`top_p`、`top_k`、`repetition_penalty`、
`max_tokens`、`stop`（字符串或列表）、`stream`。`n`、`suffix`、`seed` 等暂不支持。

## 支持的接口

| 接口 | 说明 |
|---|---|
| `GET /health` | 健康检查 |
| `GET /v1/models` | 模型列表 |
| `POST /v1/completions` | 文本补全（OpenAI 格式，支持 stream） |
| `POST /v1/chat/completions` | 对话补全（支持 stream，支持 agent 模式） |
| `GET /v1/agent/sessions` | agent 模式会话列表（会话 id / trace / token 累计，朴素实现额外提供） |
| `GET /v1/stats` | 运行时统计（KV 占用/请求数/吞吐，朴素实现额外提供） |

## 工具调用（原生 tool call）

与 vLLM 的命令完全兼容，可直接照搬 `commands.txt` 里的启动方式：

```bash
lminfer serve /path/to/model --enable-auto-tool-choice --tool-call-parser hermes   # Qwen/Hermes 系
lminfer serve /path/to/model --enable-auto-tool-choice                             # Llama 3.x 系(或任意模型: auto)
```

两个参数的语义与 vLLM 一致：

- **`--tool-call-parser`**：`auto`（默认）按模型 tokenizer 自动识别解析器——
  - Qwen/Hermes 系（有 `<tool_call>` 特殊 token）走 `hermes`：解析输出里的
    `<tool_call>{"name": ..., "arguments": ...}</tool_call>` 块并转成 OpenAI 格式的
    `tool_calls` 字段。Qwen 与 Hermes 的格式相同（vLLM 跑 Qwen3 用的就是 hermes
    parser），三个名字是同一个解析器；
  - Llama 3.x 系（有 `<|python_tag|>` 特殊 token）走 `llama3_json`：解析输出里的
    `{"name": ..., "parameters": ...}` JSON 工具调用（可多个、以 `;` 分隔、周围
    允许普通文本），对应 vLLM 的 `--tool-call-parser llama3_json`；
  - Mistral v11+ 系（有 `[TOOL_CALLS]` 特殊 token，如 Ministral-3）走 `mistral`：
    解析输出里的 `[TOOL_CALLS]name[ARGS]{json}`（可多个首尾相接，name 与 `[ARGS]`
    之间可能有 `[CALL_ID]<id>`），对应 vLLM 的 `--tool-call-parser mistral`；
  - GLM-4 系（`config.model_type == "glm4"`，如 GLM-4-9B-0414）走 `glm4`：解析输出里的
    `函数名\n{json}`（函数名独占一行、紧跟一个 JSON 参数对象，前面可能带字面量
    `<|assistant|>`，以 `<|observation|>` / `<|user|>` 结束），对应 vLLM 的
    `--tool-call-parser glm4`；
  - 都没有则 `none` 关闭解析，输出按普通文本返回。
  也可以显式指定 `hermes` / `qwen` / `llama3_json` / `mistral` / `glm4` / `none` 强制使用某
  解析器，但注意 **每个解析器只认自己那套标记**：
  给 Llama 3.x 模型配 `hermes`（或反之）会解析不到任何工具调用，工具调用文本会
  整段漏进 `content`、客户端拿不到 `tool_calls`。为避免这类静默失效，显式配置与
  模型家族冲突时服务会：启动时告警，请求时先按显式配置解析、解析不到再按模型
  原生协议自动兜底（流式直接按原生协议切流）。建议直接用 `auto`。
- **`--enable-auto-tool-choice`**：请求带 `tools` 但**未显式给 `tool_choice`** 时，
  默认按 `auto` 处理；不加该参数时默认 `none`（忽略 tools，模型按普通对话回复）。
  请求里显式的 `tool_choice` 始终优先，支持 `"auto"` / `"none"` / `"required"` /
  `{"type": "function", "function": {"name": ...}}`（指定单个函数时只渲染该工具）。
  注意 `"auto"` 在请求没有 `tools` 时退化为普通对话（与 vLLM 一致），只有
  `"required"` 才强制要求 tools。

### Llama 3.x 的适配点（`lminfer/model_adapters.py`）

Llama 3.1/3.2/3.3 系的工具调用协议与 Qwen 完全不同，适配层做了三件事：

1. **JSON 工具调用解析**（`llama3_json`）：模型输出形如
   `{"name": "get_weather", "parameters": {"city": "上海"}}`（可能带 `<|python_tag|>`
   前缀，多个调用以 `;` 分隔）。非流式按 `JSONDecoder.raw_decode` 从每个 `{` 解析
   完整对象（正确处理嵌套与字符串内括号），流式在输出以 `<|python_tag|>` 或 `{`
   开头时进入 JSON 模式（否则按普通文本透传）；`arguments` 保留模型输出的原始
   JSON 子串（round-trip 保真）。
2. **模板渲染适配**：Llama 官方 chat template 把 OpenAI 格式的
   `tool_calls.arguments`（JSON 字符串）直接 `| tojson`，会渲染成
   `"parameters": "{\"city\": ...}"`（字符串被加引号）；工具结果字符串也会被加
   引号。渲染前把 `arguments` 还原成 dict、工具结果包成 `{"output": ...}` 对象
   （与 vLLM 的 `tool_chat_template_llama3.1_json.jinja` 一致），模型才能读到
   合法的 JSON。
3. **schema 感知的参数修复**（非流式）：Llama 3.x 模型经常把 array 参数输出成
   字符串形式的 Python 列表（如 `"source_ids": "['S1']"`），客户端按 schema 校验
   会拒绝执行，多轮 agent 场景下模型读不懂报错、直接退化死循环。解析时按请求
   工具 schema 检查：schema 声明 `type: array` 且值是字符串时，尝试按 Python
   字面量解析（`ast.literal_eval`，安全），成功且结果是列表则还原成真正的 JSON
   数组再返回；修复改变了参数时 `arguments` 改用重序列化结果（不再逐位保留）。
   hermes 块解析同样支持该修复；流式路径不做（分片无法整段重写）。

### 模型适配（Qwen / Llama / Mistral / GLM-4 / 多模态 / FP8）

除工具调用协议外，"模型怎么加载、KV 怎么拼"也有家族差异，全部集中在
`lminfer/model_adapters.py`（加载/协议识别）与 `lminfer/kvcache.py`（KV 拼接）。

**Mistral 系（Ministral-3-8B-Instruct-2512 实测）**

1. **工具调用**（`mistral` 解析器）：输出形如
   `[TOOL_CALLS]call_subagent[ARGS]{"task": "..."}`，多个调用首尾相接。
   `[TOOL_CALLS]` / `[ARGS]` / `[CALL_ID]` / `[TOOL_RESULTS]` 都是特殊 token；
   兼容旧版（`<v11`）的 `[TOOL_CALLS] [{...}]` 数组形式。`arguments` 保留模型
   原始 JSON 子串（round-trip 保真）。
2. **tool_call id 必须是 9 位字母数字**：Mistral 的 prompt 由 transformers 的
   `MistralCommonBackend` 交给 **mistral-common** 渲染（而不是 jinja），它会校验
   `[a-zA-Z0-9]{9}`，不合规直接抛 `InvalidFunctionCallException`。若沿用其他协议的
   `call_<hex>` id，第二轮请求会 400 —— 工具结果根本回填不进 prompt。因此该解析器
   生成 9 位字母数字 id（与 vLLM 的 `MistralToolCall` 一致）。
3. **`chat_template is None` 不代表没有模板**：Mistral 的 tokenizer 不加载模型自带的
   `chat_template.jinja`，`tokenizer.chat_template` 恒为 None，但
   `apply_chat_template` 完全可用。服务端改用 `supports_chat_template()` 判定，
   否则 `/v1/chat/completions` 会对 Ministral 直接返回 400。
4. **拼接锚点是 `[TOOL_RESULTS]` / `[/TOOL_RESULTS]`**：拼接模式不再硬编码 Qwen3 的
   `<tool_response>`，而是按 tokenizer 探测这一对包裹标记（`TOOL_RESULT_WRAPPERS`），
   探测不到就自动回退 LCP 复用。Ministral 的正文窗口实测**逐位全中**
   （`定位子 agent 输出 KV 1 段/301 tok（窗口 301 tok）`，无边界漂移）。
5. **多模态包装只取文本塔**：checkpoint 架构是
   `Mistral3ForConditionalGeneration`（8.4B 语言模型 + 0.4B 视觉塔），
   `AutoModelForCausalLM` 不认识它。加载时改用 `AutoModelForImageTextToText`，
   只跑文本路径，并把 `config.get_text_config(decoder=True)` 作为引擎的
   `model_config` —— 顶层 config 连 `num_hidden_layers` / `rope_parameters` 都没有，
   直接拿来算 KV 形状或 RoPE 会错。
6. **FP8 权重默认反量化**：checkpoint 是 per-tensor fine-grained FP8（`weight_scale_inv`
   是标量），前向需要 `kernels` 包提供的 Triton 内核；没装时 transformers 会报
   `ImportError`。适配层在加载期用 `FineGrainedFP8Config(dequantize=True)` 反量化成
   模型 dtype（纯 transformers 路径，KV 切片与 RoPE rebase 都保持同一种 dtype），
   启动日志会打印实际路径；装了 kernels 包则保留原生 FP8，可用
   `--dequantize-fp8/--no-dequantize-fp8` 强制。8.4B 权重反量化后约占 16.6 GiB 显存。
7. **YaRN RoPE 的位置重映射**：Ministral-3 声明 `rope_type=yarn`（factor 16），
   低频频段的逆频率与默认 `theta^(-2i/d)` 公式相差 16 倍。RoPE 只有"每频率旋转
   可复合"这一条性质被 rebase 依赖（与逆频率取值无关），所以只要用**模型自己的**
   逆频率算位置差即可；`rebase_rope_cache` 因此优先调用模型文本塔的 `rotary_emb`
   模块（`--graft-rope-rebase`）。实测：用模型 RoPE 时与"直接在目标位置旋转"
   相差 4.8e-7（float32 舍入），用旧的默认公式最大偏差 **6.6**（K 的量级是 1）——
   即旧实现对 YaRN 模型会转到完全错误的相位。
   位置缩放类 RoPE（YaRN / Llama-3 / dynamic）都适用；`llama_4_scaling` 只作用在
   query 侧（`get_llama_4_attn_scale`），不影响 K 侧 rebase。

完整的适配清单、启动命令与实测日志见
[Ministral-3 适配记录](docs/ministral3.md)。

**GLM-4 系（GLM-4-9B-0414 实测）**

1. **工具调用**（`glm4` 解析器）：模型输出 `函数名\n{json}` —— 函数名独占一行，紧跟
   一个 JSON 参数对象，以 `<|observation|>`（本身就是 eos）或 `<|user|>` 结束。
   函数名前面**可能带字面量 `<|assistant|>`**（实测 research agent 会输出
   `...decision note...<|assistant|>research\n{"query": ...}`，模型卡参考实现也按
   `<|assistant|>` split 后逐段解析），解析器对两种形态都识别。没有专用的起始特殊
   token，所以 `auto` 按 `config.model_type == "glm4"` 识别（不是按 tokenizer 的标记）。
   `arguments` 保留模型原始 JSON 子串；
2. **消息协议是 `metadata` + `observation`**：模板写
   `<|assistant|>{{ metadata }}\n{{ content }}` 与 `<|observation|>\n{{ content }}`，
   **不认** OpenAI 的 `tool_calls` / `tool` 角色。直接透传会把工具调用渲染成字面量
   `None`、把工具结果整段丢掉（实测）。服务端 `adapt_glm4_messages` 在渲染前把
   `tool_calls` 翻译成 `metadata=函数名, content=参数字符串`、把 `tool` 角色翻译成
   `observation`。这样渲染出的 token 与模型生成流逐位一致，跨请求 KV 前缀能整段命中；
3. **拼接锚点是 `<|observation|>` 起、到下一个角色标记止**：GLM 的工具结果没有闭合
   标记，`ToolResultWrapper` 因此支持「终止标记集合」（`<|system|>`/`<|user|>`/
   `<|assistant|>`/`<|observation|>`）。实测窗口与子输出逐位全中；
4. **RoPE 是部分旋转 + 奇偶交错**：`partial_rotary_factor: 0.5`（128 维 head 只转前
   64 维），且 `rotate_half` 是 GPT-NeoX 式的 `(2i, 2i+1)` 配对（cos/sin 用
   `repeat_interleave(2)` 展开）。`--graft-rope-rebase` 用 `RopeLayout` 按模型布局
   旋转：实测与「直接在目标位置旋转」相差 4.8e-7；旧的「全 128 维 + 前后对半」写法
   偏差 > 1.0（K 的量级是 1）；
5. **一次回复多个工具调用**：模板一条 assistant 消息只带一个 `metadata`，适配层把
   多个调用拆成多条消息（与模型卡 README 的处理一致）；agent 场景本来就要求一次一个。

完整的适配清单、启动命令与实测日志见 [GLM-4 适配记录](docs/glm4.md)。

```bash
# 带 tools 的请求: 模型会输出 <tool_call> 块, 服务端解析为 tool_calls 返回
# (启动时加了 --enable-auto-tool-choice, tool_choice 可以不写; 不加则必须显式给 "auto")
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "上海天气怎么样?"}],
       "tools": [{"type": "function", "function": {
           "name": "get_weather", "description": "查询城市天气",
           "parameters": {"type": "object",
                          "properties": {"city": {"type": "string"}},
                          "required": ["city"]}}}]}'

# 工具执行结果以 tool 消息回传(chat template 渲染成 Qwen3 的 <tool_response> 块)
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [
        {"role": "user", "content": "上海天气怎么样?"},
        {"role": "assistant", "content": null,
         "tool_calls": [{"id": "call_xxx", "type": "function",
                         "function": {"name": "get_weather",
                                      "arguments": "{\"city\": \"上海\"}"}}]},
        {"role": "tool", "tool_call_id": "call_xxx", "content": "晴, 28 度"}],
       "tools": [...]}'
```

实现细节与边界（朴素实现刻意保留的部分）：

- 带 `tools` 时生成用 `skip_special_tokens=False` 解码，`<tool_call>` 等特殊 token
  原样保留供解析；流式响应同样按块拆分，`</tool_call>` 闭合时以 `tool_calls` delta
  发出，`finish_reason` 为 `"tool_calls"`（与 OpenAI 协议一致）；
- assistant 的 `tool_calls` 回传时，`arguments` 子串**逐位保留**模型原始输出
  （不做 JSON 归一化），保证模板二次渲染与生成流一致——这是 agent 模式跨请求
  KV 前缀复用不断在 `<tool_call>` 块上的前提（见 [跨请求 KV 复用](#跨请求-kv-复用--reuse-agent-kv)）；
- `tool_choice="required"` 与 `auto` 一样把全部 tools 渲染进模板（本地 Qwen
  模板没有 required 分支，是否调用由模型自行决定）；
- **不做 guided decoding**：vLLM 的 tool parser 会按 JSON schema 约束解码保证
  arguments 合法，朴素实现只解析不约束，JSON 的合法性依赖模型自身（对
  Qwen3 系模型通常没问题）。

## Agent 模式（会话追踪）

模型调用程序可以按两种模式调用 `/v1/chat/completions`：

- `mode: "chat"`（默认，OpenAI 兼容）：每个请求相互独立，与普通 vLLM 用法一致；
- `mode: "agent"`：把一个任务由主/子 agent 发起的多次模型调用关联到同一个会话。

agent 模式下请求需携带 `trace`（agent 调用路径，最后一个元素是当前 agent，
例如 `["main", "sub1", "main"]`）：

- 首次请求不带 `session_id`，LMInfer 会生成一个 UUID 字符串并随响应返回
  （流式时每个 SSE chunk 顶层都带），应用保存它供后续请求回传；
- 后续请求回传 `session_id`，同一会话的请求数、累计 token、出现过的 agent
  名单都会被记录，可通过 `GET /v1/agent/sessions` 观测整个大任务的资源消耗；
- 传了不存在的 `session_id` 返回 404（不静默新建）；`mode: "agent"` 不带
  `trace` 返回 400。

```bash
# 首次 agent 请求: 响应里会多出 session_id / trace 字段
curl -N http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"mode": "agent", "trace": ["main"], "messages": [{"role": "user", "content": "..."}], "stream": true}'

# 后续请求回传 session_id(示例: 主 agent 调起子 agent)
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"mode": "agent", "session_id": "<上一步返回的 id>", "trace": ["main", "sub1"], "messages": [...]}'

# 观测: 会话 id 列表 + 每个会话的请求数与 token 累计
curl http://localhost:8000/v1/agent/sessions
```

## 跨请求 KV 复用（`--reuse-agent-kv`）

### 解决的问题

主 agent 调起子 agent 后，子 agent 的输出会回到主 agent 的上下文，
主 agent 的下一轮请求在 prompt 里**重新包含了这段历史**。默认实现会对整个
prompt 重新 prefill——历史 token 的 KV 明明在子 agent 请求里已经算过一遍，
却被重复计算。

本特性（`lminfer serve ... --reuse-agent-kv`，默认关闭）让 agent 会话
**保存最近的 KV cache 并在下一次请求中复用**，这正是 vLLM 自动前缀缓存的
朴素版。

### 工作原理

1. **保存**：每个 agent 会话保存最新 `main` 完整序列；`sub` 则保留最新
   `main` 之后的全部请求（prompt + 生成输出）及其 KV cache，位置从 0 开始、
   与 token 序列逐位对齐（`lminfer/kvcache.py` 的 `SessionKVStore`）；
2. **触发**：同一会话后续请求都会尝试复用候选 KV。`main` 请求会拿到最新
   `main` 段和最新 `main` 之后的所有 `sub` 段；`sub` 请求拿到最近 `sub` 段
   用于多轮续接；
3. **正确性守卫**：引擎对新 prompt 与保存的序列做 **token 级最长公共前缀
   （LCP）匹配**，只复用真正相同的部分，其余继续 prefill。KV 是 (token, 位置)
   的确定性函数——token 与位置都一致，注意力结果在数学上就与全量 prefill
   完全相同。任何不匹配都安全回退，绝不产生错误结果；
4. **深拷贝**：transformers 的 `DynamicCache.update` 是原地拼接，复用前必须
   拷贝切片，否则后续请求会污染已保存的缓存（并发安全：store 读写都在事件
   循环上，生成线程只持有拷贝）。

### 观测方式

```bash
lminfer serve /path/to/model --reuse-agent-kv

# 服务日志: 复用发生时打印
#   请求 xxx: 复用前缀 KV 31 tok, 剩余 55 tok prefill
#   请求 xxx: prompt 86 tok, 生成 8 tok, ..., 复用前缀 31 tok

# 全局统计与每会话统计
curl http://localhost:8000/v1/stats             # kv_reuse: attempts/hits/tokens
curl http://localhost:8000/v1/agent/sessions    # 每个会话的 kv_reuse_count/tokens
```

### 实验与注意事项

```bash
python experiments/agent_kv_reuse.py --model /path/to/model
```

- **复用率取决于 chat template**：模板对同一段历史是否做一致的渲染决定 LCP
  长度。Qwen3 会对**末尾 assistant 消息**插入 `<think>` 块（与后续有 tool
  消息时的渲染不同），导致 LCP 变短（约 30%）；ERNIE 等模板渲染一致，
  复用率可达 80%+。Llama 3.1 的模板对历史逐字一致渲染（无 think 块插入），
  实测复用率 82%–92%（`Meta-Llama-3.1-8B-Instruct`，agent 多轮续接）。
  Ministral-3（mistral-common 渲染）实测同样逐字一致：主 agent 第一轮生成的
  `[TOOL_CALLS]name[ARGS]{json}` 与下一轮重新渲染的 token 完全相同（模型输出的
  `": "` 空格风格与 mistral-common 的 `json.dumps` 归一化结果一致），最后一步
  **复用 850/993 tok（86% 的 prefill 被跳过）**；若模型退化输出紧凑 JSON，
  LCP 会在参数处提前停止，属安全回退；
- **数值等价性**：复用与全量 prefill 数学上等价，但 bf16 精度下存在
  内核级舍入差异（与切换 attention 实现同级，~1e-2 相对误差），贪心输出
  通常逐 token 一致，个别低置信位置可能翻转，属正常现象；
- 显存代价：每个会话保留最近一次 `main` 请求，以及最新 `main` 之后所有
  `sub` 请求的完整 KV cache；下一次 `main` 保存成功后会清空这批 `sub` 段。

### 位置感知拼接模式（`--reuse-agent-kv-append`，实验）

LCP 安全模式只能复用"前缀完全一致"的 KV。子 agent 的输出作为 tool 结果
回填进 main 的下一轮 prompt 时，其 token 由 chat template 重新渲染（正文
前后带 role 标记，如 Qwen3 的 `<tool_response>`、Mistral 的 `[TOOL_RESULTS]`
包裹），不构成任何已保存
段的公共前缀，LCP 匹配不到 —— 这段 KV 明明在子 agent 请求里已经算过，
却只能重新 prefill。若想**直接复用子 agent 输出的 KV**，用拼接模式：

```bash
lminfer serve /path/to/model --reuse-agent-kv-append
```

机制：main 在一个或多个子 agent 返回后继续请求时，服务端
（`SessionKVStore.build_grafts`）在渲染后的 prompt token 序列中**定位每个
子 agent 输出正文的位置**，把子 agent 请求时算好的输出 KV 按 prompt 顺序
**直接插进 main 的 KV cache 对应位置**（引擎的 `KVGraft` 流程）：

```text
新 prompt = [main 历史] [标记] [sub1 正文] [标记] [sub2 正文] [标记] [新内容]
              LCP 复用  prefill   ↑ graft    prefill   ↑ graft    prefill
```

- **锚点**：包裹标记由 tokenizer 探测得到（Qwen3 是 `<tool_response>`，Mistral 是
  `[TOOL_RESULTS]`，GLM-4 是 `<|observation|>` + 下一个角色标记，见
  `TOOL_RESULT_WRAPPERS`），标记都必须是单个特殊 token；
  探测不到时拼接模式自动不可用（启动日志会说明），退化为 LCP 复用。子输出是
  "本轮新内容"，搜索起点取最新 main 段长度，历史里与子输出相似的文本（如任务
  原文）直接被排除；
- **边界漂移**：客户端把输出解码成文本再回填，模板对同一文本重新分词 ——
  BPE 对同一字符串的切分是确定的，但正文首尾可能与其相邻的换行/标记合并
  （如结尾 `。` 与模板追加的 `\n` 合成一个 token），导致渲染出的 token 与
  保存的正文在边界处不一致。`build_graft` 搜索**最长逐位一致前缀**（允许
  开头 1-2 个 token 并入前一标记），只拼接一致的部分，其余（通常是尾部
  1-2 个边界 token）正常 prefill —— 拼接的 KV 与 token 严格对齐。
  Mistral 的 `[TOOL_RESULTS]` 本身是特殊 token，正文不会被前一个标记吞并，
  实测窗口与子输出逐位全中（Ministral-3: `定位子 agent 输出 KV 1 段/301 tok
  (窗口 301 tok, 输出 301 tok)`）；
- **剔除 thinking**：`<think>...</think>` 块通常不作为下一轮对话的 prompt
  （客户端回填时剥离），子段输出里开头 think 块的 KV 被挖掉，只拼接正文
  （Ministral-3 没有 think token，该步骤自动禁用）；
- **回退**：定位失败或匹配太短（< 4 token）时安全回退到 LCP 模式（复用
  main 历史），绝不错误拼接。校验失败记入 `/v1/stats` 的 `graft_mismatches`。

⚠️ 注意：拼接的 KV 是在**子 agent 自己的上下文**里计算的，插入 main 上下文
后，注意力结果与全量 prefill 存在**近似差异**（实验用途）。但位置与 token
是对齐的，且 main 历史部分仍走 LCP 精确复用 —— 这是"跨请求 KV 直通"的
朴素实现，正确性边界见下方日志与一致性实验。

可使用 `--repair-window-begin 0.15 --repair-window-end 0.20` 独立设置每段复用
KV 首尾的重计算比例。例如匹配到 100 token 时，首部重算 15 个、尾部重算 20 个，
仅拼接中间 65 个 token 的 KV。比例范围为 `[0, 1]`，按每段匹配长度计算并向下
取整；首尾重叠时整段重算，不重复计算。服务端两项默认均为 `0`（关闭重算）。
这些参数替代原服务端 `--graft-recompute-window`；attention 示例中的
`--repair-window` 也替换为这两项，示例默认也均为 `0`。只配置一项时，另一端不重算；
两项都不配置时，不进行首尾 KV 重算。

```bash
# 服务日志会显示定位、拼接与复用
#   会话 xxx: 共定位 3/3 段子 agent 输出 KV, 准备多段拼接
#   请求 xxx: 拼接子 agent 输出 KV 3 段/156 tok(位置 582..811) + 复用 main 历史 KV 575 tok, 剩余 18 tok prefill
#   请求 xxx: prompt 749 tok, 生成 60 tok, ..., 复用前缀 731 tok
#   请求 xxx: 会话 xxx trace ['main', 'researcher', 'main'] KV 前缀复用 627 tok(prompt 645 tok, 跳过 97% prefill; 来源见引擎日志)
```

与 `--reuse-agent-kv`（LCP 模式）的关系：拼接模式是 LCP 模式的超集 ——
main 历史仍按 LCP 精确复用，定位失败时自动回退到 LCP 行为，因此单独开
`--reuse-agent-kv-append` 即可同时获得两者收益。

`--graft-rope-rebase` 在插入前把子输出 K 从子上下文位置重映射到 main prompt
的插入位置。它使用**模型文本塔自己的 RoPE 逆频率**（见
[模型适配](#模型适配qwen--llama--mistral--多模态--fp8) 第 7 条），因此对
YaRN / Llama-3 / dynamic 等缩放型 RoPE 同样正确；只修正位置差，不修正上下文差。
模型没有可用的 RoPE 模块时按默认公式回退并打印 WARNING。

Ministral-3-8B-Instruct-2512 实测（两个问题、main → sub → main → sub → main，
`--reuse-agent-kv-append --graft-rope-rebase --repair-window-begin 0.1
--repair-window-end 0.1`）：7 次请求 4 次命中复用、共跳过 2312 tok，
`graft_mismatches` 为 0，最后一步 850/993 tok（86%）；拼接日志形如
`拼接子 agent 输出 KV 1/1 段, 241/301 tok(位置 691..991) + RoPE rebase,
每段首部重算 10.0%, 尾部重算 10.0% + 复用 main 历史 KV 609 tok, 剩余 143 tok prefill`。

### 上下文误差驱动的分层修复

新增 `--repair-mode context`：完整计算浅层，用目标上下文 KV 差异和注意力读取权重选点，
深层按原位置稀疏重算，并支持检查阈值触发精确回退。
`--repair-mode exact` 直接使用精确前缀加完整后缀重算。
所有模式都会记录缓存精确前缀，防止跨轮 LCP 将近似缓存当成精确结果。

参见 [实现、启动参数和正确性边界](docs/context_repair.md) 与
[Qwen3-8B 实测结果](artifacts/context_repair.md)。30% 选择性修复在本次测试中更快，
但仍存在首尾策略答对而新策略答错的案例，不能宣称评分已经优于首尾重算。

## KV Cache 理论速览（本项目要验证的东西）

### 为什么需要 KV cache

一次 transformer 前向里，attention 层要做：

```
Q = x @ Wq        K = x @ Wk        V = x @ Wv
Attn = softmax(Q @ K^T / √d) @ V
```

生成第 n 个 token 时，**K/V 只依赖已经生成的前缀**，与"当前正在预测哪个 token"无关。
所以可以把历史 token 的 K/V 存起来（这就是 KV cache），decode 时只计算新 token 自己的
Q/K/V 再与缓存的 K/V 做 attention：

- **prefill**：一次性前向整个 prompt，计算并缓存所有层的 K/V。计算量 ∝ prompt 长度。
- **decode**：每步只输入 1 个 token，计算量**与序列长度无关**（O(1) 注意力）。

如果不缓存（本项目 `experiments/kv_cache_compare.py` 的对照模式），每步都要重算整个前缀，
总成本约 O(n²)，并且越到后面越慢。

### KV cache 的显存开销

```
每 token 新增 KV 字节数 = 2 × 层数 × KV头数 × head_dim × 每元素字节数
```

例如 Qwen2.5-7B-Instruct（28 层，4 个 KV 头，head_dim 128，bf16）：
`2 × 28 × 4 × 128 × 2 = 57344 B ≈ 56 KiB/token`，32K 上下文就需要约 1.75 GiB。
这就是 vLLM 用 PagedAttention 管理显存、以及 KV cache 量化/压缩成为研究方向的原因。

### 动手实验

```bash
python experiments/kv_cache_compare.py --model /home/tanger/workspace/models/Qwen3-0.6B --tokens 32
```

会输出两种模式的 prefill 耗时、decode 吞吐、KV 显存占用，并给出速度提升倍数。
每次服务启动时也会打印当前模型"每 token KV 占用"的理论值（`/v1/stats` 里也有）。

## 与 vLLM 的差异（学习路线图）

1. **连续批处理**：本项目线程池并发，每个请求独立 decode；vLLM 把多个序列拼进同一个
   decode 步共享矩阵乘法。下一步改造：在引擎里维护序列列表，把长度相同的序列拼批。
2. **PagedAttention**：本项目 KV 是一个随长度增长的连续张量；vLLM 按 16-token 块分配，
   解决碎片化与内存浪费。理解 `DynamicCache` 之后再对比 `StaticCache`。
3. **调度与抢占**：vLLM 有 preemption（长序列被抢占时交换或重算）；本项目直接排队。
4. **前缀缓存**：vLLM 有 automatic prefix caching；本项目在 agent 模式下用
   `--reuse-agent-kv` 实现了朴素版（trace 触发 + token 级 LCP 匹配，见
   [跨请求 KV 复用](#跨请求-kv-复用--reuse-agent-kv)），全局自动前缀缓存仍未实现。
