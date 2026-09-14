# LMInfer 的 vLLM 后端

第一版已经实现：LMInfer 保留 agent 会话、sub 正文定位、拼接计划和首尾重算策略，vLLM 负责模型加载、prefill、attention 和 decode。代码全部位于 LMInfer，不需要修改 vLLM 源码。

当前范围为单卡、默认 RoPE、FP16/BF16、全注意力，接受两类模型：dense Qwen3（GQA）与
MLA 家族（GLM-4.7-Flash / `glm4_moe_lite`）。已在 vLLM 0.27.1 上用 Qwen3-8B 与
GLM-4.7-Flash 分别跑通完整验收（见下文）。请求串行执行，使用 eager 注意力和 vLLM 的
V1 KVConnector 接口。按层稀疏 `context` 修复尚未迁移，显式选择时启动报错。

vLLM 的分页 KV 布局换过版本，本后端三种都认（`lminfer/vllm_bridge.detect_layout`）：

| 布局 | 张量形状 | K / V 在哪 |
|---|---|---|
| 0.23 时代 FLASH_ATTN | `[blocks, 2, block_size, heads, dim]` | 第 1 维的 0/1 |
| 0.27+ FLASH_ATTN（Qwen3） | `(blocks, heads, block_size, 2*head_size)` | 尾维前后各一半 |
| MLA（GLM-4.7-Flash） | `(blocks, block_size, kv_lora_rank + qk_rope_head_dim)` | 尾维 `[潜向量 \| k_rot]` |

MLA 的快照沿用 transformers 后端的约定：位置无关的压缩潜向量进 **keys 槽**，
带位置的 `k_rot` 进 **values 槽**（两槽不等宽）—— 于是 `kvcache.py` 的切片/拼接/
RoPE rebase 一行不改地复用，`--graft-rope-rebase` 天然只旋转该转的那个槽。
MLA 模型的 attention 后端必须显式给 `FLASH_ATTN_MLA`：给 `FLASH_ATTN` 会被 vLLM
判为 "MLA not supported" 并抛错（不会自动映射），因此在 `vllm_engine` 里按
`config.model_type` 选（本机 sm90 + vLLM 自带 FA3 实测走 `FLASH_ATTN_MLA`）。

## 启动

本机已建立 `.venv`，复用 `/home/tanger/miniconda3/envs/lmcache` 的现有依赖。直接在 LMInfer 根目录运行：

```bash
.venv/bin/python -m lminfer serve /home/tanger/workspace/models/Qwen3-0.6B \
  --backend vllm \
  --gpu-memory-utilization 0.2 \
  --max-model-len 4096 \
  --reuse-agent-kv-append \
  --graft-rope-rebase \
  --repair-window-begin 0.15 \
  --repair-window-end 0.20 \
  --no-enable-thinking \
  --enable-auto-tool-choice \
  --port 8000
```

MLA 模型（GLM-4.7-Flash：权重 58.2 GiB，必须给足 `--gpu-memory-utilization`）：

```bash
CUDA_VISIBLE_DEVICES=5 python3 -m lminfer serve \
  /public/home/xiaoxunpeng/Models/GLM-4.7-Flash \
  --backend vllm \
  --gpu-memory-utilization 0.9 \
  --max-model-len 40960 \
  --reuse-agent-kv-append \
  --graft-rope-rebase \
  --repair-window-begin 0.1 \
  --repair-window-end 0.1 \
  --no-enable-thinking \
  --enable-auto-tool-choice \
  --port 8010
```

启动日志里的关键行（实测）：

```text
vLLM backend: glm4_moe_lite, serialized requests, eager FLASH_ATTN_MLA; global prefix caching disabled to isolate approximate KV
Model loading took 55.87 GiB memory and 49.185168 seconds
Available KV cache memory: 15.03 GiB / GPU KV cache size: 298,096 tokens
```
```

在其他环境中，需要先准备与本地 vLLM 源码兼容的 PyTorch/CUDA/vLLM 安装，再安装 LMInfer。不要为这个后端额外加载一份 Transformers 模型。`DynamicCache` 仅作为会话 KV 快照的数据容器。

默认后端仍为 `transformers`。`--backend vllm` 时：

- `--gpu-memory-utilization` 实际控制 vLLM 模型及 paged KV 预算，默认 0.5。会话快照和拼接临时张量另外占用显存，需留出余量；上面的 0.2 用于本机已有其他 GPU 进程的情况。
- `--tensor-parallel-size` 仅接受 1。
- `--max-num-seqs` 不会开启并行推理，实际执行并发数固定为 1，HTTP 请求进入单线程队列。`/v1/stats.execution_max_num_seqs` 报告实际值。
- 后端自动设置 `VLLM_ENABLE_V1_MULTIPROCESSING=0`。若外部显式设置为 1，会报错。一个进程只支持一个存活的 LMInfer vLLM 引擎。
- 不接受自定义 `--kv-transfer-config`，因为后端需要自己的 connector。
- `--repair-mode window` 执行拼接与首尾重算；`--repair-mode exact` 只复用精确前缀，完整计算剩余 prompt。
- 全局 automatic prefix caching 被关闭，避免近似拼接缓存进入普通精确缓存。main 历史由 LMInfer 的精确 LCP 快照复用。
- 暂不支持抢占恢复、多卡、量化 KV、滑动窗口、CUDA Graph 和按层稀疏修复。遇到不支持的模型/配置会报错（`validate_model` 只放行 dense Qwen3 与 MLA 家族；模型内部的 MoE 路由不参与 KV，GLM-4.7-Flash 的 64 专家即由 vLLM 正常执行）。

服务接口及 `mode/session_id/trace` 与原 LMInfer 相同。main 使用 `trace[-1] == "main"`；sub 使用其他名称。sub 输出作为 main 的 `tool` 消息返回后，服务会在渲染后的 token 序列中匹配正文。已有的多 sub、thinking 剔除和 token 边界匹配逻辑继续使用。

## 执行机制

```text
main prompt:
[精确历史][标记 + sub1 首部][sub1 中间][尾部 + 标记 + sub2 首部][sub2 中间][尾部 + 问题]
    LCP          prefill      graft              prefill      graft        prefill/decode
```

实现文件：

| 文件 | 职责 |
|---|---|
| `lminfer/vllm_plan.py` | 校验 token、源长度、区间重叠；生成复用区间；保留最后一个 query |
| `lminfer/vllm_bridge.py` | 非连续 paged blocks 与连续 GPU KV 快照之间的读写，支持未对齐的 token 边界；识别三种分页布局并处理 MLA 的不等宽 KV 槽 |
| `lminfer/vllm_connector.py` | 告知 scheduler 连续可用前缀长度；在 forward 前装入 KV，在各层执行后保存新 KV |
| `lminfer/vllm_engine.py` | 分段执行、RoPE rebase、vLLM 采样和流式输出 |
| `lminfer/server.py` | agent 会话与快照保存、HTTP/SSE、释放接口 |

每个需要计算的连续区间，作为一个 vLLM 内部请求执行。中间请求只采样一个 token 来结束请求；这个采样结果被丢弃，**不会加入 main prompt，也没有作为新 token 进行 forward**。然后将 sub 中段 KV 拼入已算出的快照，作为下一个内部请求的连续外部前缀。最后一个内部请求执行正常生成。

这种实现无需把“有洞的 prompt”伪装成连续命中，也无需更改 attention mask：每次真正执行的 query 都是连续后缀。首尾重算 token 看见的是其真实位置之前的 KV。未来若要减少分段调度和重复拷贝，可将这些区间进一步接入 scheduler 的同一请求生命周期。

## 缓存语义和统计

- 源 KV 快照拥有独立存储，vLLM 请求结束释放 paged blocks 后不会悬空。
- 拼接写入新的目标缓存，不原地修改其他 agent 的源 K/V。默认 RoPE rebase 只旋转 K，V 保持不变。
- 源 KV 来自 sub 上下文，rebase 和首尾重算不保证与 main 全量 prefill 等价。
- `exact_prefix_len` 标记第一个近似位置之前的有效精确前缀。后续 main 的 LCP 命中不得越过该边界。
- 最后采样出的 token 通常还没有经过 forward。保存时只保留实际有效的 `prompt + output[:有效长度]`，不虚构最后一个 token 的 KV，也不增加一次补算请求。
- 校验不通过的 token/区间计划回退到精确 LCP。截断 prompt 时清除旧拼接计划，且服务不保存截断后的会话段。
- vLLM 的 main 请求准备复用时，会接管本轮 sub 缓存。选好并复制精确前缀后，立即释放原始 sub KV；每段 graft 插入目标缓存后，立即删除该段快照及临时副本的引用，**不等 main decode 结束**。main 自己已经拼接好的 KV 继续用于生成。
- main 开始后新产生的 sub 缓存属于后续批次，不会被这个 main 的完成回调清除。其他在途请求如果仍持有某段缓存，会在其最后一个引用释放后回收。
- 若 main 失败，已消费的 sub KV 不恢复，重试可从 prompt 文本重新计算。没有匹配、完整重算和 prompt 截断路径也会清理本请求接管的缓存。
- 释放后显存可由 PyTorch 分配器再次使用，`nvidia-smi` 的进程显存数不一定立即下降。原有会话 TTL 和 `POST /v1/release` 继续生效。无需增加启动参数，重启服务后自动启用。
- 直接调用 Python 引擎默认保留调用者的缓存；需要转移所有权时可传 `consume_reuse=True`，此时 `reuse_prefixes` 和 `graft` 必须是请求独占的列表，调用过程中会被清空。HTTP main 请求自动使用此行为。

agent 响应中的 `repair_stats` 包含：

| 字段 | 含义 |
|---|---|
| `stages` | 实际 vLLM 内部请求数 |
| `graft_tokens` | 扣除首尾窗口和末尾 query 后，真正跳过的 sub token 数 |
| `computed_prompt_tokens` | prompt 长度减去精确 LCP 与 graft 的复用量 |
| `executed_tokens` | connector 观测到的实际 forward token 数，包含 decode |
| `loaded_tokens` | 各 stage 导入前缀 token 数之和，可能大于 prompt 长度 |
| `cached_sequence_tokens` | 最终快照的实际有效长度 |
| `exact` | 是否完全使用精确前缀和正常计算，无近似拼接 |

引擎会检查实际执行 token 数与计划是否相符。TTFT 包含分段执行和拼接开销，不包含 HTTP 排队时间。

## 验证

```bash
.venv/bin/python -m unittest discover -s tests -v

.venv/bin/python experiments/vllm_agent_kv_smoke.py \
  --model /home/tanger/workspace/models/Qwen3-0.6B \
  --gpu-memory-utilization 0.2 \
  --output artifacts/vllm_agent_kv_smoke.json

# MLA（GLM-4.7-Flash）
python3 experiments/vllm_glm47_agent_kv_smoke.py \
  --model /public/home/xiaoxunpeng/Models/GLM-4.7-Flash \
  --gpu-memory-utilization 0.9 \
  --output artifacts/vllm_glm47_agent_kv_smoke.json
```

CPU 测试覆盖非连续 blocks、非连续张量 strides、首尾未对齐位置、源缓存独立存储、chunked capture、重叠/失配计划回退、完整重算窗口和精确边界，以及三种分页布局（含 MLA 的不等宽 KV 槽与 `latent_dim` 缺失/越界的 fail-closed）。

GPU 验收覆盖精确 LCP 输出一致性、同上下文两段拼接、真实 sub 生成后的跨位置拼接、被复用区间 K/V 的逐元素校验、exact 模式、chunked prefill、两个 sub 的 HTTP 回填、多轮近似边界、中文流式输出、错误收尾及会话释放。额外通过张量弱引用检查，确认原始 sub KV 和 graft 快照在 main 最终阶段之前已销毁。

MLA 额外断言：拼接段里位置无关的潜向量（keys 槽）逐位不变、带位置的 `k_rot`（values 槽）等于按目标位置重映射后的结果，并显式检查 rebase 不是空操作。

结果见 [Qwen3-0.6B 验收记录](../artifacts/vllm_agent_kv_smoke.json)（vLLM `0.23.1rc1.dev723+ga2f713002.d20260709`、Transformers 5.12.1、PyTorch 2.11.0+cu130）与 [GLM-4.7-Flash 验收记录](../artifacts/vllm_glm47_agent_kv_smoke.json)（vLLM 0.27.1、Transformers 5.15.0、PyTorch 2.13.0+cu130、H100 PCIe）。**0.27.1 上两种模型都实测通过**：Qwen3-8B 走 4 维 packed 布局，GLM-4.7-Flash 走 MLA 3 维布局。

这是一版功能实现，验收脚本是正确性冒烟测试，**不是公平的吞吐或加速比基准**。短输入上，分段请求、快照和重复前缀拷贝可能超过省下的 prefill 成本。应在目标工作负载上进一步测量 TTFT、显存和答案质量。
