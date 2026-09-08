# SubAgent 上下文误差驱动的 KV 修复

实现入口为 `lminfer/context_repair.py`，由 `LLMEngine._generate()` 在 graft token/位置校验后调用。
本实现是可校准的近似算法，并提供精确模式。实测尚不能证明选择性评分优于首尾重算。

## 启动与参数

```bash
python3 -m lminfer serve /home/tanger/workspace/models/Qwen3-8B \
  --dtype bfloat16 --attn-implementation sdpa \
  --reuse-agent-kv-append \
  --repair-mode context --repair-shallow-layers 1 \
  --repair-budget 0.3 --repair-coverage 0.95 --repair-min-ratio 0.05 \
  --repair-probe-tokens 8 --repair-probe-threshold 1.0
```

此例启用检查回退，阈值 `1.0` 只是演示配置，未经过真实任务集校准。在本次四个合成案例中均触发了回退。
要单独测量近似修复，设 `--repair-probe-threshold 0`；要求恢复 full prefill 语义时使用 `--repair-mode exact`。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--repair-mode` | `window` | `window` 使用原首尾策略；`context` 使用分层修复；`exact` 从精确 LCP 之后完整 prefill |
| `--repair-shallow-layers` | 1 | 完整计算 attention/MLP 的浅层数 d；用第 d+1 层 KV 差异选点；d 覆盖模型时走精确路径 |
| `--repair-budget` | 0.3 | 每段评分选点比例上限，向上取整；额外检查点和末 token 不受此上限约束 |
| `--repair-coverage` | 0.95 | 每段累计评分覆盖目标，受 budget 限制；不代表真实误差覆盖率 |
| `--repair-min-ratio` | 0.05 | 每段最低选点比例，向上取整；不能大于 budget |
| `--repair-probe-tokens` | 8 | 从未选位置按位置均匀抽取的检查点，全层持续计算；全请求共计 |
| `--repair-probe-threshold` | 0 | 大于零时，深层检查点相对 KV 误差超过阈值即精确回退；0 禁用阈值回退 |

`context` 自动重定位 RoPE，不需要 `--graft-rope-rebase`；首尾窗口参数只用于 `window`。
`--repair-budget 1 --repair-min-ratio 1` 强制全选，用于检验分层路径的正确性。
`--repair-budget 1` 单独使用仍受评分覆盖率影响，不保证全选。

## 计算路径

1. LCP 只能命中可信的精确前缀。其余位置从 token embedding 开始完整计算前 d 层。
2. 计算所有后缀 token 的第 d+1 层 Q/K/V，与重新定位的 SubAgent 缓存比较。误差为相对 K、V 范数误差之和，再取 KV 头最大值。
3. 在第 d 层目标注意力中均匀采样最多 32 个 query（包含末 query），取各头、各 query 对每个 key 的最大读取权重 I。评分为 `error * (0.05 + I)`，每段独立选择。
4. 第 d+1 层全部新 K/V 写回，随后才缩减 query 和 hidden state。新文本、模板间隙、MainAgent 后缀和末 token 始终保留。更深层未选 graft 位置复用旧 KV。
5. 每层先覆盖选中位置的 K/V，再进行 attention。稀疏 query 使用原位置的 RoPE，mask 判断 `key_position <= query_position`；不把稀疏位置重新编号。
6. 检查点比较更深层中新算和旧缓存的 KV；触发阈值时从原始精确前缀重新 full prefill。没有基于置信度或近似结果一致性的“正确性认证”。

完整序列使用标准因果 SDPA 调用，避免显式 mask 改变内核而放大 BF16 数值差异。
稀疏查询按最多 128 个 query 分块，并采用 Transformers 的 masked SDPA GQA 展开方式。
缓存更新和中间张量均为请求局部数据，不安装全局模型 hook，也不修改保存的源缓存。
第一层 K/V 当前也重新投影，以保持路径简单；节省来自后续层减少的投影、attention query 和 MLP。

## 精确性来源与观测

`GenerationResult.exact_prefix_len` 传入 `SessionKVStore.put()`，保存在 `KVPrefix` 中。
近似缓存的可信范围止于最早未修复的深层 graft 位置，其后的生成 KV 同样属于近似后缀。
下一轮即使 token 完全匹配，也只复用这个可信范围；其后重新计算。
`window` 模式同样记录近似边界。全选或精确回退成功后可将完整序列标记为精确。
这里的精确指相同因果计算语义，通常仍允许不同内核、切块造成的浮点误差。

非流式 agent HTTP 响应附带 `exact_prefix_len` 和 `repair_stats`，流式请求的最终结果也会保存元数据。
日志包含实际选点数、检查点数量、最大误差和回退原因。
`reused_prompt_tokens` 对分层模式只计完全跳过的精确前缀；分层节省请查看
`repair_stats.projected_token_layers / full_token_layers`。这是投影位置×层数的代理量，**不是 FLOPs 或加速比**。

引擎 TTFT 从缓存匹配、复制之前计时，直到首次采样得到 token；包含重定位、评分和回退成本。
不包含 HTTP 排队、tokenization 和 server 端 graft 定位/构造，不能直接当作客户端端到端延迟。

## 支持范围与验证

当前专门适配单设备、非量化、稠密 Qwen3、默认 RoPE、batch=1、无 padding。
其他模型、量化模型、非默认 RoPE、滑窗 attention、多设备或磁盘 offload 会走精确路径。
缓存必须来自同一个服务加载的模型；目前没有跨进程模型指纹交换协议。
已在 PyTorch 2.11.0+cu130、Transformers 5.12.1 和本地 Qwen3-8B/BF16 上验证。

```bash
python3 -m unittest discover -s tests -v
python3 examples/bench_context_repair.py \
  --model /home/tanger/workspace/models/Qwen3-8B \
  --repeats 3 --output artifacts/context_repair.json
```

25 项测试涵盖逐层全选等价、浅层写回、稀疏因果位置、末尾 graft、多段复用、源缓存不变、
检查回退、非默认 RoPE 回退和多轮 LCP 精确性传播。

评测固定 SubAgent 输出 token 和 MainAgent prompt，以 teacher forcing 构造来源 KV；
来源上下文包含不会回填的私有 thinking。各模式使用同一 continuation（full 模式生成文本加 EOS）
计算 `KL(full || mode)`，并记录每层 K/V 相对误差。
随机对照与 context 每段保持相同深层位置数量，包含等量检查点预算，但不计算深层检查误差，
因此其总运行成本并非严格相等。首尾对照为 15%+15%，与分层计算成本也不等价。

完整结果见 [实测记录](../artifacts/context_repair.md) 与 [原始 JSON](../artifacts/context_repair.json)。
这些短合成案例用于发现反例和验证实现，不是通用准确率评测；增加预算或浅层深度也不保证误差单调降低。
