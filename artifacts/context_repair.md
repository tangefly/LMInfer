# Qwen3-8B KV 修复实测（2026-09-08）

环境：本地 Qwen3-8B，BF16，SDPA，PyTorch 2.11.0+cu130，Transformers 5.12.1，RTX 4080 SUPER（32 GiB）。

4 个固定合成案例：中间数值、否定关系、thinking 剥离、多 Agent 比较。每模式预热一次，测量 3 次；TTFT 为各案例中位数的算术平均，KL 为各案例均值。

| 模式 | 答对数 | 平均 TTFT (ms) | 平均 continuation KL |
|---|---:|---:|---:|
| full | 4/4 | 100.97 | 0 |
| rebase | 3/4 | 92.66 | 1.89275 |
| window | 4/4 | 101.03 | 0.0129406 |
| context | 3/4 | 63.65 | 1.47958 |
| context_60 | 3/4 | 95.89 | 2.26252 |
| context_depth4 | 3/4 | 66.99 | 1.93016 |
| context_guarded | 4/4 | 120.46 | 0 |
| random | 3/4 | 59.67 | 1.44819 |
| context_all | 4/4 | 103.61 | 0 |
| exact | 4/4 | 100.97 | 0 |

`context` 为 1 层完整计算、30% 评分预算、8 个额外检查点。`context_60` 将预算增到 60%；`context_depth4` 完整计算 4 层；`context_guarded` 设置检查阈值 1.0；`context_all` 强制全选。

**选择性修复没有通过本次准确性对照。** 多 Agent 比较中，30%、60%、4 层浅层和随机模式均选错了最快团队；首尾 15%+15% 模式答对。不能据此声称浅层误差评分更优。

阈值 1.0 的 guarded 模式在全部四例触发精确回退，恢复了参考输出，但耗时高于直接 full prefill。这个阈值没有经过真实任务校准。

全选分层路径在本次四例中的全部 36 层 K/V 相对误差均为 0，固定 continuation 的 logits KL 也为 0。精确模式和回退模式同样恢复参考结果。这是当前环境的观测，不是所有硬件和切块方式逐位一致的承诺。

TTFT 包含引擎内评分、复制、重定位和回退；不含 HTTP 排队、tokenization、server 端 graft 构造。随机对照只有深层位置数量相等，不包含检查误差计算。短合成任务与少量重复不支持通用准确率或生产吞吐结论。

复现方式与参数见 [实现说明](../docs/context_repair.md)，逐案例、逐层和每次计时见 [原始 JSON](context_repair.json)。
