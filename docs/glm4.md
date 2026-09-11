# GLM-4-9B-0414 适配记录

本文记录把 `THUDM/GLM-4-9B-0414`（本机路径
`/home/tanger/workspace/models/GLM-4-9B-0414`）接到 LMInfer 的 Transformers 后端、
并跑通 **SubAgent Output KV Reuse** 所做的改动与实测证据。

这个模型的差异集中在**工具调用协议与消息渲染**：GLM-4-0414 不输出 `<tool_call>`
之类的特殊 token，而是直接生成「函数名一行 + 参数 JSON」；工具结果角色是
`observation`，函数名挂在 assistant 消息的 `metadata` 字段上 —— OpenAI 的
`tool_calls` / `tool` 消息直接透传会被模板丢掉。另外它的 RoPE 是**部分旋转 +
奇偶交错**（`partial_rotary_factor: 0.5`），拼接模式的 K 位置重映射必须按这个布局来。

适配代码集中在 `lminfer/model_adapters.py`（识别/协议/窗口/RoPE 布局）、
`lminfer/toolcalls.py`（glm4 解析器）、`lminfer/server.py`（消息渲染翻译）与
`lminfer/kvcache.py`（窗口定位 + RoPE rebase），引擎只加了「max_tokens 与
max_model_len 冲突时给出可读报错」。

## 启动

```bash
.venv/bin/python -m lminfer serve /home/tanger/workspace/models/GLM-4-9B-0414 \
  --served-model-name GLM-4-9B-0414 \
  --max-model-len 32768 \
  --reuse-agent-kv-append \
  --graft-rope-rebase \
  --repair-window-begin 0.1 \
  --repair-window-end 0.1 \
  --enable-auto-tool-choice \
  --port 8000
```

`--tool-call-parser` 不用写：`auto` 会按 `config.model_type == "glm4"` 识别成
`glm4`。写死成别的协议（例如从 Qwen3 命令里抄过来的 `--tool-call-parser hermes`）
也不会静默失效 —— 启动时告警，请求时按原生 `glm4` 协议兜底解析（已实测，见下文）。

`--max-model-len` 至少要大于客户端请求的 `max_tokens`（本仓库 agent demo 用
10240），否则没有空间留给 prompt；现在会直接报
`max_tokens(...) 不小于 max_model_len(...)`，而不是在模型 forward 里抛
`cannot reshape tensor of 0 elements`。

启动日志中的适配信息：

```text
模型加载完成: GLM-4-9B-0414 (2.59s), attention 实现: sdpa
KV cache 每 token 占用(理论): 0.04 MiB
tool-call-parser=auto 自动识别为: glm4
拼接模式使用工具结果窗口: <|observation|> 起, 至下一个角色标记
  ['<|system|>', '<|user|>', '<|assistant|>', '<|observation|>']
  (open id 151338, terminator ids [151335, 151336, 151337, 151338])
```

## 与 Qwen3 的差异与对应改动

| 差异 | 现象（不做适配会怎样） | 改动 |
|---|---|---|
| 工具调用是 `函数名\n{json}`，没有专用特殊 token | hermes/llama3_json/mistral 三个解析器一个都匹配不到，`tool_calls` 为空、原始文本漏进 `content`，agent 死循环 | 新增 `glm4` 解析器（非流式 + 流式）；`auto` 按 `config.model_type` 识别 |
| 工具调用可能带**字面量 `<\|assistant\|>` 前缀** | 正则按"行首就是函数名"匹配，`<\|assistant\|>research\n{...}` 整段匹配不到 → `tool_calls` 为空，主 agent 每轮都空转，turn budget 耗尽也调不起子 agent | 解析器增加"角色标记锚点"：名字前可跟 `<\|assistant\|>`（模型卡参考实现正是按 `<\|assistant\|>` split 后逐段解析）；流式切分器同样识别 |
| 模板只认 `assistant.metadata` 与 `observation` 角色 | OpenAI 的 `tool_calls` 被渲染成字面量 `None`，`tool` 消息整段丢弃（实测 `...<\|assistant\|>\nNone<\|assistant\|>`），多轮 agent 直接失效 | `server.adapt_glm4_messages` 在渲染前翻译：`tool_calls` → `metadata=函数名, content=参数字符串`；`tool` → `observation` |
| 工具结果窗口**没有闭合标记** | 拼接模式原先要求一对 `(open, close)` 特殊 token；GLM 只有 `<\|observation\|>` 起始，写死闭合标记永远定位不到窗口，`--reuse-agent-kv-append` 静默退化为纯 LCP | `ToolResultWrapper` 支持「终止标记集合」：窗口从 `<\|observation\|>` 起，到下一个角色标记（`<\|system\|>`/`<\|user\|>`/`<\|assistant\|>`/`<\|observation\|>`）为止 |
| `partial_rotary_factor: 0.5` + 奇偶交错 `rotate_half` | `--graft-rope-rebase` 用「全 head_dim + 前后对半」旋转，K 被转到错误相位（偏差是 K 自己的量级） | `RopeLayout` + 布局感知 rebase：只旋转前 `rotary_dim` 维、用模型自己的交错 `rotate_half`，非旋转维原样保留 |
| 没有 `<think>` / `</think>` token | 无（复用 Ministral 适配时加的「id 能反查回同名 token」严格校验，think 检测自动禁用） | `think_len` 恒为 0，拼接时不做挖空 |
| 一条 assistant 消息只带一个 `metadata` | 一次回复里多个函数调用无法用一条消息表达 | 适配层把多个调用拆成多条 assistant 消息（与模型卡 README 的 `split("<\|assistant\|>")` 处理一致）；agent 场景本就要求一次一个 |

## 实测证据

环境：RTX 4080 SUPER 32GB、transformers 5.12.1、torch 2.11.0+cu130、bf16、
Transformers 后端。模型权重约 18 GiB。

### 1. 工具调用 round-trip 逐位对齐（LCP 复用整段命中）

模型对「北京今天天气怎么样?」的真实原始输出（`skip_special_tokens=False`）：

```text
'get_weather\n{"city": "北京"}<|observation|>'
```

`<|observation|>`（id 151338）本身就是 eos：生成在工具调用结束时自然停止，工具调用
正文不含任何标记 token。`glm4` 解析器把它转成 OpenAI 工具调用，`arguments` 逐位保留
模型原始 JSON 子串；客户端回填后模板用 `metadata` 重新渲染出的 token 与生成流完全
相同 —— 所以 main 的下一轮请求能在工具调用段上直接 LCP 复用：

```text
请求 xxx: prompt 159 tok, 生成 16 tok, ...
会话 xxx: 保存 main 段 KV 175 tok(输出 16 tok, think 0 tok) 供跨请求复用
请求 xxx: 复用main 历史 KV 159 tok + 输出 KV 16 tok(prompt 183 tok 的 96%)
```

`159 + 16 = 175` 正是工具调用回合的全部 token：渲染不一致的话 LCP 会在
`<|assistant|>` 之后立刻断开。

### 1.5 工具调用的两种形态（`<|assistant|>` 前缀）

GLM-4-0414 实测会输出两种调用形态，解析器都要认：

```text
# 形态 1: 输出开头直接是调用（short prompt / 简单 agent）
get_weather\n{"city": "北京"}

# 形态 2: 一段决策正文之后跟 `<|assistant|>函数名\n{json}`（research agent 实测）
After inspecting S1, ...to make a prediction.<|assistant|>research\n{"query": "...", "source_ids": ["S1"]}
```

模型卡的参考实现写的是 `for m in generate_resp.split("<|assistant|>"):`，正是为了
处理形态 2 —— 角色标记是字面量 token，会原样出现在可见输出里。修复前的正则要求
函数名在行首，形态 2 永远匹配不到：`tool_calls` 为空、正文（含 `<|assistant|>research`
和参数 JSON）整段漏进 `content`，客户端 `_loop` 看到"没有工具调用"就追加一句
"Continue via tools"，主 agent 连续空转 24 轮直到 `max_main_turns` 耗尽 —— 表现为
"没有看到调用 sub agent，最终 answer 为空/insufficient"。

解析器的接受条件（`_glm4_candidate_ok`）现在是三者之一：名字前紧跟 `<|assistant|>`、
调用在输出开头、或名字在本次声明的工具列表里。这样正文里"行首单词 + JSON 对象"的
计划排版不会被误判（`Plan\n{"requirements": [...]}`），而未声明/未加标记的调用
也不会被静默丢掉。

用修复前后同一份真实输出回归：3 个模型回复全部从"解析不到"变为解析出 `research`
调用；重跑 `run_research.py --index 4`，主 agent 依次 `research` 了 S1/S2/S3 三篇
文档，子 agent 正常产出 findings 并给出最终 JSON。

### 2. 拼接模式：`<|observation|>` 窗口逐位全中

子 agent 回答 53 tok；main 第 2 轮 prompt 230 tok，其中子回答正文被完整定位并拼进
main 的 KV cache（首尾各重算 10%）：

```text
会话 xxx: build_grafts trace ['main', 'sub', 'main'], prompt 230 tok, main_lcp 175 tok,
          tool_response 窗口 1 个, 候选 sub 段 1 个
会话 xxx: 定位子 agent 输出 KV 1 段/53 tok(窗口 53 tok, 输出 53 tok, think_len 0), 准备插入
请求 xxx: 拼接子 agent 输出 KV 1/1 段, 43/53 tok(位置 176..228) + RoPE rebase,
          每段首部重算 10.0%, 尾部重算 10.0% + 复用 main 历史 KV 175 tok, 剩余 12 tok prefill
请求 xxx: prompt 230 tok, 生成 61 tok, ..., 复用前缀 218 tok
请求 xxx: 会话 xxx trace ['main', 'sub', 'main'] KV 前缀复用 218 tok(prompt 230 tok, 跳过 95% prefill)
```

`GET /v1/stats`：

```json
{"kv_reuse": {"reuse_attempts": 5, "reuse_hits": 2, "reuse_tokens": 393,
              "graft_mismatches": 0}}
```

`graft_mismatches` 为 0：插入位置/长度/token 逐位校验全部通过。

### 3. 端到端冒烟脚本

`experiments/glm4_agent_kv_smoke.py` 走完整 HTTP 链路（工具调用解析 →
metadata/observation 回填 → 窗口拼接），断言 main #2 复用 > 0 且无拼接失败：

```text
=== 1) main #1: 期望 glm4 解析出 call_subagent 工具调用 ===
  "tool_calls": [{"function": {"name": "call_subagent",
                               "arguments": "{\"task\": \"法国的首都是哪座城市?\"}"}}]
=== 2) sub: 子 agent 独立回答(保存 sub 段 KV) ===
  子 agent 回答: '法国的首都是巴黎。巴黎是法国最大的城市，...' (reused=0)
=== 3) main #2: 期望复用 main 历史 + 拼接子 agent 输出 KV ===
  reused_prompt_tokens=218 / prompt_tokens=230
OK: GLM-4-9B-0414 子 agent 输出 KV 复用链路跑通
```

### 4. 流式工具调用

`Glm4StreamSplitter` 在 undecided 状态攒到「标识符行 + 换行 + `{`」才进入工具模式，
正常文本（含中文/空格/换行的回答）原样透传：

```text
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_...","type":"function",
  "function":{"name":"call_subagent","arguments":"{\"task\": \"法国的首都是哪座城市?\"}"}}]},
  "finish_reason":null}]}
data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
```

### 5. RoPE rebase 数值校验（部分旋转 + 奇偶交错）

用模型真实的 `Glm4RotaryEmbedding` + `apply_rotary_pos_emb`（不需要权重）：把 K 在
位置 1200 旋转后 rebase 到 2400，与「直接在 2400 旋转」比较：

| rebase 用的布局 | 与直接旋转的最大偏差 |
|---|---|
| 部分旋转（前 64 维）+ 奇偶交错（当前实现） | **4.8e-07**（float32 舍入） |
| 全 head_dim（128）+ 前后对半（旧写法） | **> 1.0**（K 的量级是 1，即相位完全错误） |

对应单元测试：`tests/test_kvcache.py::Glm4RopeRebaseTest`。后 64 维不参与旋转，
rebase 后逐位不变（`test_pass_through_dims_are_untouched`）。

## 已知边界

- **`--repair-mode context` 不适用于本模型**：分层修复是 Qwen3 专用的
  （`context_repair.support_reason`），GLM-4 上会打印原因并**自动回退
  `exact_prefill`**。`--repair-mode window` / `exact` 可用。
- **`--backend vllm` 尚未支持本模型**：`vllm_engine.validate_model` 目前只接受
  非量化、默认 RoPE 的 dense Qwen3，GLM-4 会在启动时报错退出（不会静默降级）。
- **一次回复多个工具调用**：GLM 协议一条 assistant 消息只能带一个 `metadata`，
  适配层会把多个调用拆成多条消息；agent 场景要求一次一个，正常不会触发。
- **模型可能给出 schema 之外的参数**：LMInfer 不做 guided decoding，`glm4` 解析器
  逐位保留模型输出的 `arguments`（这是 KV 前缀能逐位对齐的前提），不会按 schema
  过滤。实测 GLM-4-9B-0414 在 `BenchAgent` 的任务提示下会把 `document_path` 作为
  **独立参数**传进 `call_subagent`，而该工具的 schema 只声明了 `task` —— 客户端
  `Tool.call` 直接 `func(**arguments)` 会抛 `TypeError`，子 agent 根本没被调起
  （服务端日志里没有 trace 以 `sub` 结尾的请求）。这不是 LMInfer 的错误：要跑通
  BenchAgent，请在 `agent/tools.py` 的 `call_subagent` schema 里声明
  `document_path`（或在 `Tool.call` 里忽略未声明的键），只依赖协议本身的
  round-trip 保真。
- **拼接本身仍是近似**：位置与 token 对齐且 RoPE 相位已修正，但子输出 KV 是在子
  agent 自己的上下文里算出来的，插入 main 上下文后与全量 prefill 存在上下文差；
  这是 `--reuse-agent-kv-append` 的固有边界，不是本次适配引入的。
- 显存：每 token KV 约 0.04 MiB（40 层 × 2 KV 头 × 128 head_dim × 2 × 2B），
  32K 上下文单序列约 1.25 GiB。
