# Ministral-3-8B-Instruct-2512 适配记录

本文记录把 `mistralai/Ministral-3-8B-Instruct-2512`（本机路径
`/home/tanger/workspace/models/Ministral-3-8B-Instruct-2512`）接到 LMInfer 的
Transformers 后端、并跑通 **SubAgent Output KV Reuse** 所做的改动与实测证据。
模型本身的差异（工具调用协议、渲染后端、RoPE、权重格式、多模态包装）较多，
适配代码集中在 `lminfer/model_adapters.py`、`lminfer/toolcalls.py`、
`lminfer/kvcache.py`，引擎只增加了"文本塔 config / 模型自带 RoPE / 只算末尾 logits"。

## 启动

```bash
.venv/bin/python -m lminfer serve /home/tanger/workspace/models/Ministral-3-8B-Instruct-2512 \
  --served-model-name Ministral-3-8B \
  --max-model-len 40960 \
  --reuse-agent-kv-append \
  --graft-rope-rebase \
  --repair-window-begin 0.1 \
  --repair-window-end 0.1 \
  --enable-auto-tool-choice \
  --port 8000
```

`--tool-call-parser` 不用写：`auto` 会识别成 `mistral`。写死成别的协议（例如从
Qwen3 命令里直接抄过来的 `--tool-call-parser hermes`）也不会静默失效 —— 启动时
告警，请求时按原生 `mistral` 协议兜底解析（已实测，见下文）。

启动日志中的适配信息：

```text
模型适配: fine-grained FP8 权重: 未安装 kernels 包, 加载期反量化成普通线性层
模型适配: 多模态 checkpoint: 用 AutoModelForImageTextToText 加载, 只使用文本塔(KV 复用/拼接都在文本塔上)
模型加载完成: Ministral-3-8B (1.73s), attention 实现: sdpa
KV cache 每 token 占用(理论): 0.13 MiB
tool-call-parser=auto 自动识别为: mistral
拼接模式使用工具结果包裹标记: [TOOL_RESULTS]/[/TOOL_RESULTS] (id (7, 8))
```

## 与 Qwen3 的差异与对应改动

| 差异 | 现象（不做适配会怎样） | 改动 |
|---|---|---|
| 工具调用是 `[TOOL_CALLS]name[ARGS]{json}` | hermes 解析器一个都匹配不到，`tool_calls` 为空、整段原始文本漏进 `content` | 新增 `mistral` 解析器（非流式 + 流式）、`auto` 识别 `[TOOL_CALLS]` 特殊 token；显式配置冲突时请求级兜底 |
| mistral-common 校验 tool_call id 为 `[a-zA-Z0-9]{9}` | 第二轮请求抛 `InvalidFunctionCallException`（400），工具结果永远回填不进 prompt | Mistral 解析器生成 9 位字母数字 id（与 vLLM `MistralToolCall` 一致） |
| 渲染由 **mistral-common** 负责，不是 jinja | `tokenizer.chat_template is None`，`/v1/chat/completions` 直接 400 "当前模型没有 chat template" | 新增 `supports_chat_template()`；服务端不再用 `chat_template is None` 判定 |
| 工具结果包裹标记是 `[TOOL_RESULTS]` | 拼接模式写死 `<tool_response>`，永远定位不到窗口，`--reuse-agent-kv-append` 静默退化为纯 LCP | `TOOL_RESULT_WRAPPERS` 按 tokenizer 探测，探测不到才回退 |
| 多模态包装 `Mistral3ForConditionalGeneration` | `AutoModelForCausalLM` 不认识该架构；顶层 config 没有 `num_hidden_layers`/`rope_parameters`，KV 形状与 RoPE 计算全错 | `load_text_model()` 用 `AutoModelForImageTextToText` 加载并返回 `config.get_text_config(decoder=True)` 作为 `engine.model_config` |
| per-tensor FP8 权重（`weight_scale_inv` 是标量） | 前向 `ImportError: finegrained-fp8 kernel requires the kernels package` | 加载期 `FineGrainedFP8Config(dequantize=True)` 反量化成 bf16；装了 kernels 包则保留原生 FP8，`--dequantize-fp8/--no-dequantize-fp8` 可强制 |
| `rope_type: yarn`（factor 16）+ `llama_4_scaling` | `--graft-rope-rebase` 用默认 `theta^(-2i/d)` 公式，K 被转到完全错误的相位 | `rebase_rope_cache` 改用模型文本塔自己的 `rotary_emb` 求位置差 cos/sin；`llama_4_scaling` 只作用在 query 侧，不影响 K 侧 |
| 没有 `<think>` / `</think>` token | `convert_tokens_to_ids` 返回 unk id，旧判断会把 `<unk>` 当成 think 起点 | 用"id 能反查回同名 token"严格校验；Ministral 上 think 检测自动禁用 |
| 13 万词表 + 40K 上下文 | 每次前向为全部位置算 logits：`40960*131072*2B ≈ 10.7 GiB`，与 16.6 GiB 权重叠加会 OOM | 引擎在 forward 签名支持时传 `logits_to_keep=1`（只算末尾 logits；本引擎只取 `out.logits[:, -1, :]`） |

## 实测证据

环境：RTX 4080 SUPER 32GB、transformers 5.12.1、torch 2.11.0+cu130。
本次验证走 **Transformers 后端**，未使用 vLLM 后端。
FP8 反量化后权重约占 16.6 GiB，运行中进程显存约 19.1 GiB。

### 1. LCP 复用：工具调用回合逐位对齐

Ministral 生成的工具调用与 mistral-common 重新渲染的结果 **token 完全相同**
（模型输出的 `": "` 空格风格与 `json.dumps` 归一化结果一致）：

```text
会话 xxx: 保存 main 段 KV 605 tok(输出 52 tok, think 0 tok) 供跨请求复用
会话 xxx: 保存 sub 段 KV 416 tok(输出 23 tok, think 0 tok) 供跨请求复用
请求 xxx: 复用子 agent 历史 KV 393 tok + 输出 KV 23 tok(prompt 1424 tok 的 29%), 剩余 1008 tok prefill
```

### 2. 拼接模式：`[TOOL_RESULTS]` 窗口逐位全中

`[TOOL_RESULTS]` 是特殊 token，正文不会被前一个标记吞并，因此窗口与子输出
逐位一致（Qwen3/Llama 常见的首尾边界漂移在这里没有出现）：

```text
会话 xxx: build_grafts trace ['main','sub','main'], prompt 633 tok, main_lcp 605 tok, tool_response 窗口 1 个, 候选 sub 段 1 个
会话 xxx: 定位子 agent 输出 KV 1 段/25 tok(窗口 25 tok, 输出 25 tok, think_len 0), 准备插入
请求 xxx: 拼接子 agent 输出 KV 1/1 段, 21/25 tok(位置 607..631) + RoPE rebase, 每段首部重算 10.0%, 尾部重算 10.0% + 复用 main 历史 KV 605 tok, 剩余 7 tok prefill
请求 xxx: 会话 xxx trace ['main','sub','main'] KV 前缀复用 626 tok(prompt 633 tok, 跳过 99% prefill)
```

### 3. 端到端（MainAgent → SubAgent 串行链）

`BenchAgent/scripts/example/agent_infer.py --model Ministral-3-8B`，doc1.txt 的
5 个问题全部按 `main → sub → main → ... ` 顺序跑完，最终 JSON 五个答案均正确：

```json
{"answer1": "Ken Maynard",
 "answer2": "Ken Neville pretended to be a suspect in his father's murder to manipulate Rance Collins into trusting him by orchestrating a rescue scenario.",
 "answer3": "Ken Neville hid his identity to avoid suspicion for his father's murder and to investigate rustlers without being targeted.",
 "answer4": "One film (*Beauty and the Bad Man*) is lost, while the other (*Alias – the Bad Man*) survives; ...",
 "answer5": "The female lead starts distrustful or hostile toward the hero but gradually realizes his true nature, leading to emotional or romantic reconciliation."}
```

`GET /v1/stats`：

```json
{"model": "Ministral-3-8B", "backend": "transformers",
 "completed": 31, "generated_tokens": 2620, "prefill_tokens": 30705,
 "kv_reuse": {"reuse_attempts": 36, "reuse_hits": 18, "reuse_tokens": 14313, "graft_mismatches": 0}}
```

最后一次 main 请求 `reused_prompt_tokens = 1877 / 2272 tok`（跳过 82.6% prefill）。
`graft_mismatches` 为 0：所有插入都通过了位置/长度/token 逐位校验。

### 4. RoPE rebase 数值校验（YaRN）

用模型真实的 `Ministral3RotaryEmbedding`（不需要权重）：把 K 在位置 1200 旋转后
rebase 到 2400，与"直接在 2400 旋转"比较：

| rebase 用的逆频率 | 与直接旋转的最大偏差 |
|---|---|
| 模型自己的 YaRN 逆频率 | **4.8e-07**（float32 舍入） |
| 旧的默认 `theta` 公式 | **6.62**（K 的量级是 1，即相位完全错误） |

对应单元测试：`tests/test_kvcache.py::Ministral3YarnRebaseTest`。

### 5. 显式 `--tool-call-parser hermes`（照抄 Qwen3 命令）

```text
WARNING --tool-call-parser hermes 与模型家族协议(mistral)不匹配: ... 请求时会按模型原生协议回退解析; 建议改用 --tool-call-parser auto 或 mistral
（请求日志）tool-call-parser=hermes 未识别到工具调用(请求 xxx), 按模型原生协议 mistral 解析出 1 个
```

工具调用没有丢失：2 个问题跑完，答案正确，`reused_prompt_tokens = 1085 / 1196`。

## 已知边界

- **`--repair-mode context` 不适用于本模型**：分层修复的实现是 Qwen3 专用的
  （`context_repair.support_reason`），Ministral 上会打印原因并**自动回退
  `exact_prefill`**（实测：`{'fallback_reason': 'context repair requires a dense
  Qwen3 model with >= 2 layers', 'exact': True}`），不会给出错误结果。
  `--repair-mode window` / `exact` 可用。
- **`--backend vllm` 尚未支持本模型**：`vllm_engine.validate_model` 目前只接受
  非量化、默认 RoPE 的 dense Qwen3，Ministral 会在启动时报错退出（不会静默降级）。
  要走 vLLM 分段拼接需要单独适配 FP8 与 YaRN。
- **思考模式**：该 Instruct 模型没有 think token，`think_len` 恒为 0，拼接时不做
  挖空；`--enable-thinking` 传给它会被 mistral-common 忽略（不报错）。
- **流式工具调用粒度**：与既有 hermes/llama 实现一致，参数 JSON 闭合后才一次性
  以 `tool_calls` delta 发出，不做逐字符增量。
- **拼接本身仍是近似**：位置与 token 对齐，但子输出 KV 是在子 agent 自己的上下文
  里算出来的，插入 main 上下文后与全量 prefill 存在上下文差；这是
  `--reuse-agent-kv-append` 的固有边界，不是本次适配引入的。
- 反量化后权重 16.6 GiB，`--max-model-len 40960` 时单序列 KV 上限约 5.4 GiB
  （0.13 MiB/token）；多会话同时保留段时需自行评估显存。
