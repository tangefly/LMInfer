"""朴素并发压测: 并发请求数可调, 统计 TTFT / TPOT / 吞吐.

用法(先启动服务):
  lminfer serve --model /path/to/model --port 8000 --max-num-seqs 4

  python examples/bench.py --prompts 8 --max-tokens 64
"""

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def one_request(url: str, prompt: str, max_tokens: int, results: list, idx: int):
    """发起一次流式请求, 记录 TTFT / 生成耗时 / token 数."""
    t_start = time.perf_counter()
    t_first = None
    text = ""
    resp = requests.post(
        f"{url}/v1/completions",
        json={"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.7, "stream": True},
        stream=True, timeout=600,
    )
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        piece = chunk["choices"][0].get("text", "")
        if piece:
            if t_first is None:
                t_first = time.perf_counter()
            text += piece
    t_end = time.perf_counter()
    n_tokens = chunk["usage"]["completion_tokens"] if "usage" in chunk else len(text)
    results[idx] = {
        "ttft_ms": (t_first - t_start) * 1000,
        "total_ms": (t_end - t_start) * 1000,
        "tokens": n_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="lminfer 并发压测")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--prompts", type=int, default=8, help="并发请求数")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prefix", default="请写一段关于人工智能的短文, 主题是",
                        help="每个请求的 prompt 前缀")
    args = parser.parse_args()

    prompts = [f"{args.prefix}{i + 1}" for i in range(args.prompts)]
    results = [None] * args.prompts
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.prompts) as pool:
        futures = [pool.submit(one_request, args.url, p, args.max_tokens, results, i)
                   for i, p in enumerate(prompts)]
        for f in futures:
            f.result()
    wall = time.perf_counter() - t0

    total_tokens = sum(r["tokens"] for r in results)
    ttfts = sorted(r["ttft_ms"] for r in results)
    print(f"\n并发 {args.prompts} 个请求, 墙钟耗时 {wall:.1f}s")
    print(f"总生成 token 数: {total_tokens}, 总吞吐: {total_tokens / wall:.1f} tok/s")
    print(f"TTFT: 平均 {statistics.mean(ttfts):.0f}ms, 中位 {statistics.median(ttfts):.0f}ms, "
          f"P90 {ttfts[int(len(ttfts) * 0.9) - 1]:.0f}ms")
    tpots = [r["total_ms"] / r["tokens"] for r in results if r["tokens"] > 0]
    print(f"TPOT: 平均 {statistics.mean(tpots):.1f}ms/token")


if __name__ == "__main__":
    sys.exit(main())
