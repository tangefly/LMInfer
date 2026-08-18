"""命令行入口: 尽量模仿 vLLM 的启动方式.

用法:
  lminfer serve --model /path/to/model [--host 0.0.0.0] [--port 8000] ...
  lminfer chat  --model /path/to/model [--max-tokens 1024] ...
  python -m lminfer serve --model ...
"""

import argparse
import logging
import sys

logger = logging.getLogger("lminfer")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lminfer",
        description="基于 transformers 的朴素 LLM 推理服务(用于 KV Cache 理论学习)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- lminfer serve: 启动 HTTP 服务(类似 vllm serve) ----
    p_serve = sub.add_parser("serve", help="启动 OpenAI 兼容的 HTTP 服务")
    p_serve.add_argument("--model", required=True, help="模型路径或 HF 模型名")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--dtype", default="auto",
                         choices=["auto", "bfloat16", "float16", "float32"])
    p_serve.add_argument("--max-model-len", type=int, default=4096,
                         help="单序列最大总长度(prompt + 生成)")
    p_serve.add_argument("--max-num-seqs", type=int, default=4,
                         help="最大并发请求数(朴素并发 = 线程数)")
    p_serve.add_argument("--trust-remote-code", action="store_true")
    p_serve.add_argument("--enable-thinking", action="store_true",
                         help="给 apply_chat_template 传 enable_thinking=True(Qwen3 等)")
    p_serve.add_argument("--disable-log-stats", action="store_true")
    p_serve.add_argument("--gpu-memory-utilization", type=float, default=None,
                         help="仅为了与 vLLM 命令兼容而接受; 朴素实现中 KV cache "
                              "由 transformers 动态管理, 此参数不生效")

    # ---- lminfer chat: 交互式对话(便于不启动服务快速验证) ----
    p_chat = sub.add_parser("chat", help="交互式命令行对话")
    p_chat.add_argument("--model", required=True)
    p_chat.add_argument("--dtype", default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"])
    p_chat.add_argument("--max-model-len", type=int, default=4096)
    p_chat.add_argument("--max-tokens", type=int, default=1024)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--trust-remote-code", action="store_true")

    return parser


def cmd_serve(args: argparse.Namespace) -> None:
    if args.gpu_memory_utilization is not None:
        logger.warning("--gpu-memory-utilization 在朴素实现中不生效: "
                       "KV cache 由 transformers 动态分配, 无需预分显存")

    from .config import EngineConfig
    from .server import run_server

    config = EngineConfig(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=args.disable_log_stats,
        enable_thinking=True if args.enable_thinking else None,
    )
    logger.info("启动服务: %s:%d (模型 %s)", args.host, args.port, args.model)
    run_server(config, host=args.host, port=args.port)


def cmd_chat(args: argparse.Namespace) -> None:
    import torch

    from .config import EngineConfig, SamplingParams
    from .engine import LLMEngine

    config = EngineConfig(model=args.model, dtype=args.dtype,
                          max_model_len=args.max_model_len,
                          trust_remote_code=args.trust_remote_code)
    engine = LLMEngine(config)
    if engine.tokenizer.chat_template is None:
        sys.exit("该模型没有 chat template, 无法使用对话模式")

    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    messages: list[dict] = []
    print("进入对话模式(输入 exit / quit 退出):\n")
    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        ids = engine.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):  # transformers 5.x 返回 tokenizers.Encoding
            ids = ids.input_ids
        result = engine._generate("chat", torch.tensor(ids).unsqueeze(0), sampling)
        messages.append({"role": "assistant", "content": result.output_text})
        print(f"Assistant: {result.output_text}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args()
    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()
