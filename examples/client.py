"""OpenAI 兼容客户端示例: 调用本地 lminfer 服务的 /v1 接口.

用法(先启动服务):
  lminfer serve --model /path/to/model --port 8000

  python examples/client.py "用一句话介绍大语言模型"
  python examples/client.py --stream "用一句话介绍大语言模型"
  python examples/client.py --completions "The capital of France is"
"""

import argparse
import json
import sys

import requests

DEFAULT_URL = "http://localhost:8000"


def chat(url: str, prompt: str, stream: bool, **sampling):
    """POST /v1/chat/completions, 支持流式与非流式."""
    body = {"messages": [{"role": "user", "content": prompt}], "stream": stream, **sampling}
    resp = requests.post(f"{url}/v1/chat/completions", json=body, stream=stream, timeout=300)
    resp.raise_for_status()

    if not stream:
        data = resp.json()
        print("回复:", data["choices"][0]["message"]["content"])
        print("finish_reason:", data["choices"][0]["finish_reason"])
        print("usage:", data["usage"])
        return

    # 流式: SSE 按行解析, 累加 delta 文本
    print("回复:", end="", flush=True)
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            print("\n(流式结束)")
            return
        delta = json.loads(payload)["choices"][0]["delta"].get("content", "")
        if delta:
            print(delta, end="", flush=True)
    print()


def completion(url: str, prompt: str, stream: bool, **sampling):
    """POST /v1/completions."""
    body = {"prompt": prompt, "stream": stream, **sampling}
    resp = requests.post(f"{url}/v1/completions", json=body, stream=stream, timeout=300)
    resp.raise_for_status()
    if not stream:
        data = resp.json()
        print("补全:", data["choices"][0]["text"])
        print("finish_reason:", data["choices"][0]["finish_reason"])
        return
    print("补全:", end="", flush=True)
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            print("\n(流式结束)")
            return
        text = json.loads(payload)["choices"][0].get("text", "")
        if text:
            print(text, end="", flush=True)
    print()


def main():
    parser = argparse.ArgumentParser(description="lminfer 客户端示例")
    parser.add_argument("prompt", nargs="?", default="用一句话介绍大语言模型")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--stream", action="store_true", help="流式输出")
    parser.add_argument("--completions", action="store_true", help="走 /v1/completions")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.completions:
        completion(args.url, args.prompt, args.stream, max_tokens=args.max_tokens,
                   temperature=args.temperature)
    else:
        chat(args.url, args.prompt, args.stream, max_tokens=args.max_tokens,
             temperature=args.temperature)


if __name__ == "__main__":
    sys.exit(main())
