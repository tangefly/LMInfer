"""理论验证实验: KV cache 到底省了什么?

对同一个模型、同一个 prompt, 分别用两种方式生成相同数量的 token:
  1. 启用 KV cache (正常推理): prefill 一次前向, 之后每步只算 1 个 token
  2. 禁用 KV cache (对照):   每步都要把整个前缀重新前向一遍

预期结论(实验要验证的理论):
  - 启用 cache 时, 每步注意力计算量与序列长度无关(O(1)/token);
  - 禁用 cache 时, 每步计算量随序列长度线性增长, 总成本近似 O(n^2);
  - 代价: cache 要占显存(每 token = 2 x 层数 x KV头数 x head_dim x 字节数).

用法:
  python experiments/kv_cache_compare.py --model /path/to/model [--tokens 32] [--prompt "..."]
"""

import argparse
import sys
import time

import torch

# 让脚本可以从仓库根目录直接运行
sys.path.insert(0, ".")

from lminfer.config import EngineConfig, SamplingParams  # noqa: E402
from lminfer.engine import LLMEngine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="KV cache 开/关对比实验")
    parser.add_argument("--model", required=True, help="模型路径或 HF 模型名")
    parser.add_argument("--tokens", type=int, default=32, help="生成的 token 数")
    parser.add_argument("--prompt", default="The history of artificial intelligence is",
                        help="实验用 prompt")
    args = parser.parse_args()

    engine = LLMEngine(EngineConfig(model=args.model))
    prompt_ids = torch.tensor(engine.tokenizer(args.prompt)["input_ids"]).unsqueeze(0)
    print(f"\nprompt: {args.prompt!r} ({prompt_ids.shape[1]} tokens), "
          f"生成 {args.tokens} 个 token\n")

    # 理论值: 每 token 新增 KV 显存
    mib_per_token = engine.kv_bytes_per_token / (1024 ** 2)
    print(f"理论 KV 占用: {mib_per_token:.3f} MiB / token, "
          f"{args.tokens} token 序列 ≈ {mib_per_token * args.tokens:.1f} MiB\n")

    sampling = SamplingParams(max_tokens=args.tokens, temperature=0)  # 贪心, 结果可复现

    # 预热: 每种模式先跑一遍丢弃(消除 CUDA kernel 冷启动的偏差)
    engine._generate("warmup", prompt_ids, sampling, use_kv_cache=True)
    engine._generate("warmup", prompt_ids, sampling, use_kv_cache=False)

    print(f"{'模式':<22}{'prefill':>10}{'decode 总耗时':>14}{'decode 吞吐':>14}{'KV 显存':>12}")
    print("-" * 72)
    rows = {}
    for use_kv in (True, False):
        r = engine._generate("exp", prompt_ids, sampling, use_kv_cache=use_kv)
        mode = "启用 KV cache" if use_kv else "禁用 KV cache(每步重算)"
        rows[use_kv] = r
        print(f"{mode:<22}{r.ttft_ms:>8.0f}ms{r.decode_ms:>12.0f}ms"
              f"{r.decode_tokens_per_sec:>12.1f} tok/s{r.kv_cache_bytes / (1024**2):>10.2f} MiB")

    # 对照一致性: 两种模式应先生成相同前缀, 只在个别位置可能因
    # bf16 舍入差异导致 argmax 翻转, 之后指数发散(正常现象, 非 bug)
    a, b = rows[True], rows[False]
    same = next((i for i, (x, y) in enumerate(zip(a.output_tokens, b.output_tokens))
                 if x != y), min(len(a.output_tokens), len(b.output_tokens)))
    print(f"对照一致性: 前 {same}/{len(a.output_tokens)} 个 token 一致"
          f"{' (完全一致)' if same == len(a.output_tokens) else ' (数值舍入导致的翻转)'}")
    print(f"速度提升: {b.decode_ms / a.decode_ms:.1f}x (decode 阶段)")
    print(f"显存代价: 启用 cache 多占 {a.kv_cache_bytes / (1024**2):.2f} MiB")
    print("\n注: 序列很短或模型很小时, GPU 处于 kernel 启动延迟受限状态, "
          "KV cache 的 FLOPs 优势看不出来; 换更大模型或更长序列即可见.")


if __name__ == "__main__":
    main()
