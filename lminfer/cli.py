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
    # vLLM 两种写法都支持: `vllm serve /path/to/model` 与 `vllm serve --model /path`
    p_serve.add_argument("model", nargs="?", default=None,
                         help="模型路径或 HF 模型名(vLLM 风格的位置参数)")
    p_serve.add_argument("--model", dest="model_flag", default=None,
                         help="模型路径或 HF 模型名(与位置参数二选一)")
    p_serve.add_argument("--served-model-name", default=None,
                         help="对外暴露的模型名(覆盖 --model 的最后一段路径名)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--dtype", default="auto",
                         choices=["auto", "bfloat16", "float16", "float32"])
    p_serve.add_argument("--attn-implementation", default="auto",
                         choices=["auto", "eager", "sdpa", "flash_attention_2"],
                         help="attention 实现: auto 用 transformers 默认(sdpa); "
                              "flash_attention_2 需安装 flash-attn(未装时回退 kernels 或报错)")
    p_serve.add_argument("--max-model-len", type=int, default=4096,
                         help="单序列最大总长度(prompt + 生成)")
    p_serve.add_argument("--max-num-seqs", type=int, default=4,
                         help="最大并发请求数(朴素并发 = 线程数)")
    p_serve.add_argument("--trust-remote-code", action="store_true")
    p_serve.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None,
                         help="给 apply_chat_template 传 enable_thinking 开关(Qwen3 等): "
                              "--enable-thinking 传 True, --no-enable-thinking 传 False; "
                              "都不传则走模板默认(不传该参数)")
    p_serve.add_argument("--disable-log-stats", action="store_true")

    # ---- 为兼容 vLLM 命令而接受、但朴素实现不生效的参数 ----
    for name, desc in [
        ("--gpu-memory-utilization", "KV cache 由 transformers 动态管理, 无需预分显存"),
        ("--tensor-parallel-size", "朴素实现只支持单卡"),
        ("--kv-transfer-config", "朴素实现不接入 KV 传输(如 LMCache)"),
    ]:
        p_serve.add_argument(name, default=None, help=f"兼容 vLLM 参数; {desc}.")
    p_serve.add_argument("--tool-call-parser", choices=["auto", "qwen", "hermes", "llama3_json", "none"], default=None,
                         help="工具调用解析: auto 自动识别模型家族(Qwen/Hermes 系解析 "
                              "<tool_call> 块, Llama 3.x 系解析 {\"name\":...,\"parameters\":...} "
                              "JSON); qwen/hermes 强制块解析; llama3_json 强制 JSON 解析; "
                              "none 关闭. 默认 auto")
    p_serve.add_argument("--enable-auto-tool-choice", action="store_true",
                         help="请求带 tools 且未显式给 tool_choice 时默认按 auto 处理"
                              "(不开启时默认 none, 忽略 tools; 与 vLLM 语义一致)")
    p_serve.add_argument("--reuse-agent-kv", action="store_true",
                         help="agent 模式跨请求前缀 KV 复用(默认关闭): main 在子 agent 返回后"
                              "继续请求时, 复用已保存的子 agent 输出 KV 与前缀历史 KV, 跳过重复"
                              " prefill. 通过 token 级最长公共前缀匹配保证正确性, 不匹配时自动"
                              " 回退全量 prefill")
    p_serve.add_argument("--reuse-agent-kv-append", action="store_true",
                         help="位置感知拼接模式(默认关闭, 实验用途): main 在子 agent 返回后"
                              "继续请求时, 在渲染后的 prompt 中定位子 agent 输出正文, 把其 KV"
                              "直接插入 main 的 KV cache 对应位置, 跳过这段的重复 prefill —— "
                              "main 历史仍按 LCP 复用, 定位失败自动回退. 注意: 子输出 KV 在"
                              "子 agent 自己的上下文里计算, 插入后与全量 prefill 存在近似差异")
    p_serve.add_argument("--kv-segment-idle-ttl", type=float, default=None,
                         help="已保存 KV 段的会话闲置超时秒数(默认 3600): 会话闲置超过该时长, "
                              "其全部 KV 段被清理释放显存. 0 表示不清理")

    # ---- lminfer chat: 交互式对话(便于不启动服务快速验证) ----
    p_chat = sub.add_parser("chat", help="交互式命令行对话")
    p_chat.add_argument("--model", required=True)
    p_chat.add_argument("--dtype", default="auto",
                        choices=["auto", "bfloat16", "float16", "float32"])
    p_chat.add_argument("--attn-implementation", default="auto",
                        choices=["auto", "eager", "sdpa", "flash_attention_2"],
                        help="attention 实现, 同 serve; flash_attention_2 需安装 flash-attn")
    p_chat.add_argument("--max-model-len", type=int, default=4096)
    p_chat.add_argument("--max-tokens", type=int, default=1024)
    p_chat.add_argument("--temperature", type=float, default=0.7)
    p_chat.add_argument("--trust-remote-code", action="store_true")

    return parser


def cmd_serve(args: argparse.Namespace) -> None:
    # 接受但忽略的 vLLM 兼容参数(显式传了才提示, 便于对照 vLLM 命令)
    for name, desc in [
        ("gpu_memory_utilization", "KV cache 由 transformers 动态分配, 无需预分显存"),
        ("tensor_parallel_size", "朴素实现只支持单卡"),
        ("kv_transfer_config", "朴素实现不接入 KV 传输(如 LMCache)"),
    ]:
        if getattr(args, name, None):
            logger.warning("--%s 在朴素实现中不生效: %s", name.replace("_", "-"), desc)

    from .config import EngineConfig
    from .server import run_server

    model = args.model_flag or args.model
    if model is None:
        raise SystemExit("错误: 需要提供模型, 用法: lminfer serve /path/to/model 或 lminfer serve --model /path/to/model")

    config = EngineConfig(
        model=model,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        served_model_name=args.served_model_name,
        trust_remote_code=args.trust_remote_code,
        disable_log_stats=args.disable_log_stats,
        enable_thinking=args.enable_thinking,  # --enable-thinking=True / --no-enable-thinking=False / 缺省 None
        tool_call_parser=args.tool_call_parser or "auto",
        enable_auto_tool_choice=args.enable_auto_tool_choice,
        reuse_agent_kv=args.reuse_agent_kv,
        reuse_agent_kv_append=args.reuse_agent_kv_append,
        kv_segment_idle_ttl=args.kv_segment_idle_ttl if args.kv_segment_idle_ttl is not None else 3600.0,
    )
    logger.info("启动服务: %s:%d (模型 %s)", args.host, args.port, args.model)
    run_server(config, host=args.host, port=args.port)


def cmd_chat(args: argparse.Namespace) -> None:
    import torch

    from .config import EngineConfig, SamplingParams
    from .engine import LLMEngine

    config = EngineConfig(model=args.model, dtype=args.dtype,
                          attn_implementation=args.attn_implementation,
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
