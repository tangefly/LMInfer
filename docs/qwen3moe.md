# Qwen3-30B-A3B-Instruct-2507 适配记录

本文记录把 `Qwen/Qwen3-30B-A3B-Instruct-2507`（本机路径
`/public/home/xiaoxunpeng/Models/Qwen3-30B-A3B-Instruct-2507`）接到 LMInfer 的
Transformers 后端、并跑通 **SubAgent Output KV Reuse** 所做的改动与实测证据。
（`--backend vllm` 不在本次范围内，原因见文末「已知边界」。）

这个模型是 LMInfer 接的第一个 **MoE** 模型：48 层、每层 128 个专家、每个 token 激活
8 个（总参数 30B，激活约 3B）。好消息是它的**工具调用协议、KV 形状、RoPE 布局与
dense Qwen3 完全一致** —— 家族判定、`hermes` 解析器、`<tool_response>` 窗口、
拼接模式的 RoPE rebase 都不用为它加分叉；差异全在**权重结构（专家路由）与显存预算**，
以及由此暴露出来的一处**临时张量**问题（见下文 `logits_to_keep`）。

适配代码集中在 `lminfer/model_adapters.py`（`resolve_logits_kwargs`：只取末尾
logits 的探测）、`lminfer/context_repair.py`（exact 回退路径共用同一探测）；
引擎与 KV 复用链路本身**没有改动**。

## 启动

```bash
python3 -m lminfer serve /public/home/xiaoxunpeng/Models/Qwen3-30B-A3B-Instruct-2507 \
  --served-model-name Qwen3-30B-A3B-Instruct-2507 \
  --max-model-len 40960 \
  --reuse-agent-kv-append \
  --graft-rope-rebase \
  --repair-window-begin 0.1 \
  --repair-window-end 0.1 \
  --enable-auto-tool-choice \
  --port 8000
```

`--tool-call-parser` 不用写：`auto` 按 tokenizer 的 `<tool_call>` / `<tool_response>`
识别成 `hermes`，窗口探测落在 `<tool_response>`/`</tool_response>` 上 —— 与 dense
Qwen3 走的是同一条判定路径。

启动日志中的适配信息：

```text
模型加载完成: Qwen3-30B-A3B-Instruct-2507 (34.47s), attention 实现: sdpa
KV cache 每 token 占用(理论): 0.09 MiB
tool-call-parser=auto 自动识别为: hermes
拼接模式使用工具结果包裹标记: <tool_response>/</tool_response> (id (151665, 151666))
```

## 与 dense Qwen3 的差异与对应改动

| 差异 | 现象（不做适配会怎样） | 改动 |
|---|---|---|
| 层内是 **MoE**（128 专家 / top-8），不是单个 MLP | 无：KV cache 只由注意力决定（`num_hidden_layers × num_kv_heads × head_dim`），专家路由不参与 KV。本模型 48 × 4 × 128 × 2 × 2B = 0.09 MiB/token | 无需改动（`Qwen3MoeForCausalLM` 在 `AutoModelForCausalLM` 里，走纯文本加载分支） |
| RoPE 模块是 `Qwen3MoeRotaryEmbedding` | 无：布局与 dense Qwen3 相同（**全 128 维旋转 + 前后对半配对**），theta 是 1e7 | 无需改动：`resolve_rope_layout` 按 `model_type` 判交错（只有 GLM 系是交错），rebase 用模型自己的逆频率。实测偏差 4.8e-07（见下） |
| 词表 151936，且是 30B 级权重 | `--repair-mode context` 的回退路径（`exact_prefill`）默认会为**所有位置**算 logits：40K prompt 下 `40960 × 151936 × 2B ≈ 11.6 GiB`，与 58 GiB 权重叠加直接 OOM | `exact_prefill` 改用 `resolve_logits_kwargs(model)` 传 `logits_to_keep=1`（与引擎 prefill 同一探测），只算末尾位置的 logits |
| 分层 `context` 修复直接调用 `layer.self_attn` / `layer.mlp` | MoE 层的 MLP 是稀疏专家路由，没有 dense 层的语义 | **明确回退**：`support_reason` 对 `qwen3_moe` 返回原因，`context_prefill` 转 `exact_prefill`（结果正确、只是不省 prefill），日志带 `fallback_reason` |
| 模板**没有** `enable_thinking` 分支（Instruct-2507 是非 thinking 模型） | 无：请求/启动参数里的 `enable_thinking` 被 jinja 当未使用变量忽略，不报错，也不产生 think 块 | 无需改动；`<think>`/`</think>` 虽在词表里但模型不输出，think 检测自动失效（`think_len` 恒为 0） |
| `config_1m.json` 声明 dual-chunk + 稀疏注意力（1M 上下文配置） | 该配置需要 Qwen3 自带的 dual-chunk 实现，LMInfer 的朴素前向不支持 | 不使用该配置（默认 `config.json` 是常规全注意力） |

## 实测证据

环境：单张 H100 80GB（PCIe）、bf16、`sdpa`、batch=1、Transformers 后端、
transformers 5.15.0 / torch 2.13.0+cu130。权重约占 **57.7 GiB** 显存。

### 1. 工具调用 round-trip 逐位对齐（LCP 复用整段命中）

模型对「北京现在天气如何？请用工具查询。」的真实原始输出（`skip_special_tokens=False`）：

```text
'<tool_call>\n{"name": "get_weather", "arguments": {"city": "北京"}}\n</tool_call>'
```

`hermes` 解析器把它转成 OpenAI 工具调用，`arguments` 逐位保留模型原始 JSON 子串：

```json
{"id": "call_6250016ade7e481d", "type": "function",
 "function": {"name": "get_weather", "arguments": "{\"city\": \"北京\"}"}}
```

客户端回填后模板重新渲染出的 token 与生成流完全相同，所以 main 的下一轮请求能在
工具调用段上直接 LCP 复用（`216` prompt + `27` 输出 = `243`，一位不多一位不少）：

```text
请求 51a12ddb7e0f: prompt 216 tok, 生成 27 tok, TTFT 1501ms, 9.6 tok/s, KV cache 22.78 MiB, stop
会话 ...: 保存 main 段 KV 243 tok(输出 27 tok, think 0 tok) 供跨请求复用
请求 a970efdc4fa8: 拼接子 agent 输出 KV 1/1 段, ... + 复用 main 历史 KV 243 tok, 剩余 20 tok prefill
```

### 2. 拼接模式：`<tool_response>` 窗口定位 + 子输出 KV 插入

子 agent 回答 36 tok；main 第 2 轮 prompt 292 tok，其中子回答正文被完整定位
（窗口 37 tok，含首尾换行），首尾各重算 10% 后插入 29 tok 的 KV：

```text
会话 ...: build_grafts trace ['main', 'sub', 'main'], prompt 292 tok, main_lcp 243 tok,
          tool_response 窗口 1 个, 候选 sub 段 1 个
会话 ...: 定位子 agent 输出 KV 1 段/35 tok(窗口 37 tok, 输出 36 tok, think_len 0), 准备插入
请求 a970efdc4fa8: 拼接子 agent 输出 KV 1/1 段, 29/35 tok(位置 250..284) + RoPE rebase,
          每段首部重算 10.0%, 尾部重算 10.0% + 复用 main 历史 KV 243 tok, 剩余 20 tok prefill
请求 a970efdc4fa8: prompt 292 tok, 生成 35 tok, ..., 复用前缀 272 tok
请求 a970efdc4fa8: 会话 ... trace ['main', 'sub', 'main'] KV 前缀复用 272 tok
          (prompt 292 tok, 跳过 93% prefill; 来源见引擎日志)
```

`GET /v1/stats`：

```json
{"kv_reuse": {"reuse_attempts": 4, "reuse_hits": 1, "reuse_tokens": 272,
              "graft_mismatches": 0}}
```

`graft_mismatches` 为 0：插入位置/长度/token 逐位校验全部通过。

### 3. 端到端冒烟脚本

`experiments/qwen3moe_agent_kv_smoke.py` 走完整 HTTP 链路（hermes 解析 →
tool_calls/tool 消息渲染回填 → 窗口拼接），断言 main #2 复用 > 0 且无拼接失败：

```text
=== 1) main #1: 期望 hermes 解析出 call_subagent 工具调用 ===
  "tool_calls": [{"function": {"name": "call_subagent",
                               "arguments": "{\"task\": \"法国的首都是哪座城市?\"}"}}]
=== 2) sub: 子 agent 独立回答(保存 sub 段 KV) ===
  子 agent 回答: '法国的首都是巴黎。  \n巴黎不仅是法国的政治、经济和文化中心，...'
=== 3) main #2: 期望复用 main 历史 + 拼接子 agent 输出 KV ===
  reused_prompt_tokens=272 / prompt_tokens=292
OK: Qwen3-30B-A3B-Instruct-2507 子 agent 输出 KV 复用链路跑通
```

脚本里子 agent 请求显式 `tool_choice="none"`：30B 在 prompt 里带着 `call_subagent`
时会**继续把问题转包给子 agent**（实测这样 `content` 为 null、只有 `tool_calls`），
没有正文可拼接。

### 4. `--repair-mode context` 在 MoE 上自动回退（实测日志）

```text
请求 65f24a337c5f: context repair {'fallback_reason': 'context repair requires
          a dense Qwen3 model with >= 2 layers', 'exact': True}
```

同一份冒烟脚本在该模式下照常通过（复用 243 tok —— 只有 main 历史，没有拼子输出 KV）。

### 5. RoPE rebase 数值校验（`Qwen3MoeRotaryEmbedding`，不需要权重）

把 K 在位置 1200 旋转后 rebase 到 2400，与「直接在 2400 旋转」比较：

| rebase 用的 RoPE | 与直接旋转的最大偏差 |
|---|---|
| 模型自己的 `Qwen3MoeRotaryEmbedding`（当前实现） | **4.8e-07**（float32 舍入） |
| 无可调用 RoPE 模块时的默认公式（full-dim + half-split, theta=1e7） | **0.0**（默认 RoPE 与模型一致） |

对应单元测试：`tests/test_kvcache.py::Qwen3MoeRopeRebaseTest`。

### 6. 长上下文与吞吐（单卡 80GB）

`--max-model-len 40960` + `--reuse-agent-kv`，36K token prompt 的一次请求：

```text
36K tok prompt: TTFT 2901ms (≈12.4K tok/s prefill), decode ≈12 tok/s
176 tok prompt: TTFT 171ms, decode ≈9–13 tok/s
峰值显存 68.8 GiB / 81.6 GiB（权重 57.7 GiB + KV 3.4 GiB + 激活）
```

batch=1、`sdpa`、无连续批处理、无 CUDA Graph —— 这是朴素实现的预期量级；
MoE 的收益在**权重显存**（30B 参数只激活 3B，但朴素实现仍是全权重常驻）。

## 已知边界

- **`--repair-mode context` 不适用于本模型**：分层修复是 dense Qwen3 专用的
  （`context_repair.support_reason`），MoE 上打印原因并**自动回退 `exact_prefill`**
  （结果精确，但不省这段 prefill）。`--repair-mode window` / `exact` 可用；
  `exact` 路径现在只算末尾 logits（见上表），40K 上下文下不再有 11.6 GiB 的
  logits 临时张量。
- **`--backend vllm` 不支持本模型**：`vllm_engine.validate_model` 目前只接受
  非量化、默认 RoPE 的 dense Qwen3，`qwen3_moe` 会在启动时报错退出
  （`vllm agent KV backend currently supports dense Qwen3 only`），不会静默降级。
  本次适配不含 vLLM 后端。
- **显存是硬约束**：权重 57.7 GiB + KV 0.09 MiB/token。单张 80GB 卡上
  36K 上下文实测峰值 68.8 GiB；把 `--max-model-len` 开到 40K 以上时请留足余量
  （KV 40K ≈ 3.75 GiB/序列，`--max-num-seqs` 会成倍放大）。
- **`<tool_call>` / `<tool_response>` 不是 `special` token**（dense Qwen3 同样如此）：
  它们不在 `all_special_ids` 里，`skip_special_tokens=True` 不会剥掉。工具路径本就
  用 `skip_special_tokens=False` 解码、再由解析器剥离标记；非工具路径若模型自发
  输出调用标记，会原样出现在 `content` 里（与 vLLM 的朴素行为一致）。
- **`enable_thinking` 对本模型无效**：Instruct-2507 的模板没有 thinking 分支，
  请求里带该字段不会报错，也不会产生 think 块；LMInfer 的历史 think 剥离逻辑
  在这里是空操作。
- **拼接本身仍是近似**：位置与 token 对齐且 RoPE 相位已修正，但子输出 KV 是在子
  agent 自己的上下文里算出来的，插入 main 上下文后与全量 prefill 存在上下文差；
  这是 `--reuse-agent-kv-append` 的固有边界，不是本次适配引入的。
