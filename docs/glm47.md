# GLM-4.7-Flash 适配记录

本文记录把 `zai-org/GLM-4.7-Flash`（本机路径
`/public/home/xiaoxunpeng/Models/GLM-4.7-Flash`）接到 LMInfer 的 Transformers 后端、
并跑通 **SubAgent Output KV Reuse** 所做的改动与实测证据。

这个模型与前面几个家族的差异在**两个互不相干的轴上**：

1. **工具调用是 XML 协议**，且 `<tool_call>`/`<arg_key>`/`<arg_value>` 在 tokenizer 里
   都是**单特殊 token**（GLM-4-0414 没有这些 token，靠 `config.model_type` 识别）：
   `<tool_call>函数名<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`，
   即 vLLM 的 `glm45`/`glm47` parser（`glm47_moe_tool_parser`）。模板**原生认 OpenAI 的
   `tool_calls`/`tool` 角色**，但 arguments 必须是对象。
2. **注意力是 MLA**（DeepSeek 式压缩潜向量）：KV cache 里 `layer.keys` 是位置无关的
   压缩潜向量 `kv_lora_rank=512`，`layer.values` 才是唯一带位置信息的 `k_rot`
   （`qk_rope_head_dim=64`）。拼接模式的 RoPE 位置重映射必须按这个布局来。

适配代码集中在 `lminfer/toolcalls.py`（glm4_moe 解析器）、
`lminfer/model_adapters.py`（家族识别 / arguments 渲染 / MLA 的 RoPE 布局与 KV 公式）、
`lminfer/kvcache.py`（按槽选张量的 RoPE rebase）、`lminfer/server.py`
（消息渲染拆成可测的模块级函数 + content=null 归一）。引擎只改了
`kv_bytes_per_token`（走家族公式）。

## 启动

```bash
python3 -m lminfer serve /public/home/xiaoxunpeng/Models/GLM-4.7-Flash \
  --served-model-name GLM-4.7-Flash \
  --max-model-len 40960 \
  --reuse-agent-kv-append \
  --graft-rope-rebase \
  --repair-window-begin 0.1 \
  --repair-window-end 0.1 \
  --no-enable-thinking \
  --enable-auto-tool-choice \
  --port 8010
```

`--tool-call-parser` 不用写：`auto` 按 `config.model_type`（`glm4_moe_lite`）识别成
`glm4_moe`。vLLM 的 `glm45`/`glm47` 是本解析器的别名，从 vLLM 命令直接抄过来也能用。

`--no-enable-thinking` 是**推荐**项（不是必需）：这个模板的 `enable_thinking` 缺省是
**开**（生成提示以 `<think>` 结尾，与 Qwen3 的约定相反），开启时模型输出以裸推理正文
开头，客户端回填历史时会剥掉 think 段，可复用前缀随之变短。关掉后实测 round-trip
逐位全等（见下文）。

启动日志中的适配信息：

```text
模型加载完成: GLM-4.7-Flash (31.23s), attention 实现: sdpa
KV cache 每 token 占用(理论): 0.05 MiB
tool-call-parser=auto 自动识别为: glm4_moe
拼接模式使用工具结果包裹标记: <tool_response>/</tool_response> (id (154845, 154846))
```

## 与 Qwen3 的差异与对应改动

| 差异 | 现象（不做适配会怎样） | 改动 |
|---|---|---|
| 工具调用是 `<tool_call>函数名<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`，且 `<tool_call>` 是单特殊 token | `auto` 先命中 `<tool_call>` token 探测 → **hermes**；hermes 对 XML 块 `json.loads` 必然失败，块被 cleaner **整段删掉**（content 里连原文都不剩），而 `parser == native` 连兜底回退都不触发 —— 客户端只看到空回复，agent 死循环 | 新增 `glm4_moe` 解析器（非流式 + 流式）；`resolve_tool_parser` 把 model_type 判定**提到 token 探测之前**；`--tool-call-parser` 接受 vLLM 别名 `glm45`/`glm47` |
| 模板用 `{% for k, v in tc.arguments.items() %}` 渲染工具调用 | arguments 是 JSON 字符串（OpenAI 线格式）时抛 `UndefinedError: 'str object' has no attribute 'items'` → **每个回填了工具调用的回合都 400** | `ModelProfile.arguments_as_dict` 对本家族打开，渲染前把字符串还原成 dict（与 Llama 3.x 同一处适配） |
| 模板对 `assistant.content = None` 渲染出字面量 `None` | OpenAI 协议里工具调用消息的 content 就是 null：渲染出的 prompt 里多一段 `None`，与生成流对不上 | `server.adapt_openai_messages` 只在“带 tool_calls 的 assistant 消息”这一条件下把 null 归一成空串 |
| **MLA**：带位置信息的只有 `k_rot`，且缓存在 **value 槽** | `--graft-rope-rebase` 只旋转 `layer.keys` —— 对本模型转的是**位置无关的潜向量**，真正该转的 `k_rot` 纹丝不动（静默算错，不抛任何异常） | `RopeLayout.rotated_slot` 描述“哪个槽带 RoPE”，`rebase_rope_cache` 按槽选张量，另一槽逐位 clone |
| `config.rope_interleave=True` 描述的是**输入侧**布局 | 照它选“奇偶交错”rotate_half 会把缓存里的 `k_rot` 转到错误相位（偏差是 `k_rot` 自身的量级）；而且传进 rebase 的 `head_dim` 是 key 槽宽度 512，与 RoPE 无关 | 缓存的配对是**前后对半**：模型的 `apply_rotary_pos_emb_interleave` 在写缓存**之前**已把配对重排；`RopeLayout` 的 `rotary_dim` 取 `qk_rope_head_dim` 而不是 key 槽宽度 |
| KV cache 存的是压缩潜向量 | 通用公式 `2 × 层数 × KV头数 × head_dim` 报 0.23 MiB/token，高估 4.4 倍 | `model_adapters.kv_bytes_per_token` 按家族算：`(kv_lora_rank + qk_rope_head_dim) × 层数 × dtype` |
| thinking 默认**开**（模板缺省走 `<think>`） | 开启时输出以裸推理正文开头，回填剥 think 后 LCP 提前断开 | 文档与冒烟脚本推荐 `--no-enable-thinking`；think 检测本身有效（`think_len` 正常统计，实测为 0） |
| MoE（64 路由专家 + 1 共享，top-4） | 无（与 Qwen3-MoE 一样：专家路由不参与 KV 形状与 RoPE，不需要额外适配） | 无 |

## 实测证据

环境：8×H100 80GB（本次跑单卡）、transformers 5.15.0、torch 2.13.0+cu130、bf16、
Transformers 后端、`attention 实现: sdpa`。权重（48 分片、约 59 GiB）加载后
**常驻显存 60,108 MiB**（含 KV）。

### 1. 工具调用 round-trip 逐位对齐（LCP 整段命中）

模型对「法国的首都是哪座城市?」的真实回复（`temperature=0`）：

```text
content: "我来帮你查询法国的首都。"
tool_calls: [{"function": {"name": "call_subagent",
                           "arguments": "{\"task\": \"法国的首都是哪座城市?\"}"}}]
```

`arguments` 里的参数值是标签之间的原始文本（**一律按字符串**返回，与 vLLM 的 glm47
一致），客户端回填后模板重新渲染出的 token 与生成流完全一致 —— main 的下一轮请求
能在整个工具调用回合上直接复用：

```text
请求 edb280368e44: prompt 215 tok, 生成 24 tok, ... , stop
会话 68fe497467464807: 保存 main 段 KV 239 tok(输出 24 tok, think 0 tok) 供跨请求复用
...
请求 8728d1a61dc8: 复用main 历史 KV 239 tok ... 复用前缀 276 tok
```

`215 + 24 = 239` 正是这轮的全部 token。**不需要 GPU 的回归测试**也在同一断言上：
`tests/test_toolcalls.py::Glm4MoeMessageRenderingTest::test_renders_with_real_tokenizer_and_round_trips_token_exact`
用真实 tokenizer 渲染“系统+用户”与“回填后的第二轮”两段 prompt，断言第一段是第二段的
前缀、且回填段与模型原始输出的 token **逐位相等**。

### 2. 拼接模式：`<tool_response>` 窗口逐位全中 + MLA 的 RoPE rebase

子 agent 回答 45 tok；main 第 2 轮 prompt 289 tok，其中子回答正文被完整定位并拼进
main 的 KV cache（首尾各重算 10%）：

```text
会话 68fe497467464807: build_grafts trace ['main', 'sub', 'main'], prompt 289 tok,
          main_lcp 239 tok, tool_response 窗口 1 个, 候选 sub 段 1 个
会话 68fe497467464807: 定位子 agent 输出 KV 1 段/45 tok(窗口 45 tok, 输出 45 tok, think_len 0), 准备插入
请求 8728d1a61dc8: 拼接子 agent 输出 KV 1/1 段, 37/45 tok(位置 241..285) + RoPE rebase,
          每段首部重算 10.0%, 尾部重算 10.0% + 复用 main 历史 KV 239 tok, 剩余 13 tok prefill
请求 8728d1a61dc8: prompt 289 tok, 生成 13 tok, ... , 复用前缀 276 tok(prompt 289 tok, 跳过 96% prefill)
```

`GET /v1/stats`：

```json
{"kv_reuse": {"reuse_attempts": 3, "reuse_hits": 1, "reuse_tokens": 276,
              "graft_mismatches": 0}}
```

工具结果窗口就是 Qwen 系那一对 `<tool_response>`/`</tool_response>`（本 tokenizer 里
两者都是单特殊 token，id 154845/154846），模板渲染 tool 消息时写作
`<|observation|><tool_response>正文</tool_response>`。

### 3. 端到端冒烟脚本

`experiments/glm47_agent_kv_smoke.py` 走完整 HTTP 链路（XML 工具调用解析 →
tool_calls/tool 回填 → 窗口拼接 → MLA 的 value 槽 RoPE rebase），断言 main #2
复用 > 0 且无拼接失败：

```text
=== 1) main #1: 期望 glm4_moe 解析出 call_subagent 工具调用 ===
  "tool_calls": [{"function": {"name": "call_subagent",
                               "arguments": "{\"task\": \"法国的首都是哪座城市?\"}"}}]
=== 2) sub: 子 agent 独立回答(保存 sub 段 KV) ===
  子 agent 回答: '法国的首都是**巴黎**。...' (reused=0)
=== 3) main #2: 期望复用 main 历史 + 拼接子 agent 输出 KV ===
  reused_prompt_tokens=276 / prompt_tokens=289
OK: GLM-4.7-Flash 子 agent 输出 KV 复用链路跑通
```

### 4. 流式工具调用

`Glm4MoeStreamSplitter` 复用 hermes 的切分逻辑（`<tool_call>`/`</tool_call>` 都是单
token），只把块体解析换成 XML 参数：

```text
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_...","type":"function",
  "function":{"name":"get_weather","arguments":"{\"city\": \"北京\"}"}}]},"finish_reason":null}]}
data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
```

### 5. 拼错协议时的兜底（不静默丢调用）

照抄别的模型的启动参数（`--tool-call-parser hermes`）：

```text
WARNING --tool-call-parser hermes 与模型家族协议(glm4_moe)不匹配: 该解析器识别不了
        这个模型的工具调用输出 ... 建议改用 --tool-call-parser auto 或 glm4_moe
WARNING tool-call-parser=hermes 未识别到工具调用(请求 4da04974b378), 按模型原生协议
        glm4_moe 解析出 1 个(建议改用 --tool-call-parser auto 或 glm4_moe)
```

请求仍拿到 `tool_calls`（而不是被 hermes 删空的 content）。

### 6. RoPE rebase 数值校验（MLA：value 槽 + 前后对半）

用模型真实的 `Glm4MoeLiteRotaryEmbedding` + `apply_rotary_pos_emb_interleave`
（不需要权重）把 `k_rot` 从位置 1200 重映射到 2400，与“直接在 2400 旋转”比较
（真实维度：`kv_lora_rank=512`、`qk_rope_head_dim=64`）：

| rebase 用的布局 | 与直接旋转的最大偏差 |
|---|---|
| **value 槽 + 前后对半**（当前实现） | **2.38e-07**（float32 舍入） |
| 潜向量（key 槽）逐位校验 | **0.00e+00**（完全没碰） |
| 旧写法：key 槽 + 奇偶交错 | **4.49**（k_rot 的量级是 3.10，即相位完全错误） |

对应单元测试：`tests/test_kvcache.py::Glm4MoeLiteRopeRebaseTest`；布局判定见
`tests/test_model_adapters.py::ResolveRopeLayoutTest::test_glm4_moe_lite_is_mla_with_value_slot_rope`。

## 已知边界

- **`--repair-mode context` 不适用于本模型**：分层修复是 Qwen3 专用的
  （`context_repair.support_reason` 要求 dense Qwen3），GLM-4.7-Flash 上会打印原因并
  **自动回退 `exact_prefill`**（此时只复用 LCP 前缀，拼接段整段重算）：

  ```text
  请求 83cf1b16d1c1: context repair {'fallback_reason': 'context repair requires a dense
            Qwen3 model with >= 2 layers', 'exact': True}
  请求 83cf1b16d1c1: 会话 ... KV 前缀复用 239 tok(prompt 289 tok, 跳过 83% prefill)
  ```

  `--repair-mode window`（默认，跳过 96%）/ `exact` 可用。
- **`--backend vllm` 尚未支持本模型**：`vllm_engine.validate_model` 目前只接受非量化、
  默认 RoPE 的 dense Qwen3；另外 vLLM 后端按 `(1, num_key_value_heads, len, head_dim)`
  写分页 KV，对 MLA 的压缩潜向量不成立。会启动即报错退出，不会静默降级。
- **参数值一律是字符串**：与 vLLM 的 glm47 一致（那套 parser 也按字符串收集
  `<arg_value>`），客户端要数字/对象需自行转换。例外：schema 声明 `type: array` 且值
  是可解析的 Python 字面量时，会按 schema 修成真正的数组（修复改变参数时 arguments
  改为重序列化，不再逐位保真——与 hermes/mistral 的既有取舍一致）。
- **开启 thinking 时复用前缀会变短**：模型输出以裸推理正文开头（模板已预开
  `<think>`），而历史 assistant 的 think 段在渲染期被剥掉（`strip_assistant_think`），
  LCP 会在 think 段处断开。工具调用与正文仍在 think 之后，`think_len` 统计正常
  （本模型 `<think>`/`</think>` 是单特殊 token，id 154841/154842）。agent 场景推荐
  `--no-enable-thinking`。
- **拼接本身仍是近似**：位置与 token 对齐、MLA 的 RoPE 相位已按 value 槽修正，但子输出
  KV 是在子 agent 自己的上下文里算出来的，插入 main 上下文后与全量 prefill 存在上下文
  差；这是 `--reuse-agent-kv-append` 的固有边界，不是本次适配引入的。
- 显存：每 token KV 约 **0.05 MiB**（`(512 + 64) × 47 层 × 2B`，与 KV 头数无关），
  40K 上下文单序列约 2.2 GiB；权重 59 GiB + KV 实测常驻 60,108 MiB，单张 80GB 卡可跑。
