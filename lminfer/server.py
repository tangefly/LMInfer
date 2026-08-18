"""FastAPI 服务: 提供 vLLM 风格的 OpenAI 兼容 HTTP 接口.

接口列表:
  GET  /health               健康检查(200 = 可用)
  GET  /v1/models            模型列表
  POST /v1/completions       文本补全(支持 stream)
  POST /v1/chat/completions  对话补全(支持 stream)
  GET  /v1/stats             运行时统计(朴素实现额外提供的接口)
"""

import asyncio
import json
import time
import uuid

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config import EngineConfig, SamplingParams
from .engine import GenerationResult, LLMEngine
from .schemas import ChatCompletionRequest, CompletionRequest

STREAM_HEADERS = {
    "Cache-Control": "no-cache",       # SSE 要求不缓存
    "X-Accel-Buffering": "no",         # 避免反代缓冲导致流式延迟
}


def create_app(engine: LLMEngine) -> FastAPI:
    """把引擎包装成 FastAPI 应用."""

    app = FastAPI(
        title="LMInfer",
        description="基于 transformers 的朴素推理服务(学习 KV Cache 用), OpenAI 兼容接口",
        version="0.1.0",
    )

    # ------------------------------------------------------------------
    # 辅助函数: 构造 OpenAI 格式响应
    # ------------------------------------------------------------------
    def _base(req_id: str) -> dict:
        return {"id": req_id, "created": int(time.time()), "model": engine.model_name}

    def _usage(r: GenerationResult) -> dict:
        return {
            "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "total_tokens": r.prompt_tokens + r.completion_tokens,
        }

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def _chat_streamer(req_id: str, queue: asyncio.Queue):
        """chat 流式响应生成器: role 分片 -> 文本分片 -> 结束分片 -> [DONE]."""
        base = _base(f"chatcmpl-{req_id}")
        yield _sse({**base, "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        while True:
            kind, payload = await queue.get()
            if kind == "done":
                r: GenerationResult = payload
                yield _sse({**base, "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {}, "finish_reason": r.finish_reason}],
                            "usage": _usage(r)})
                yield "data: [DONE]\n\n"
                return
            yield _sse({**base, "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}]})

    async def _completion_streamer(req_id: str, queue: asyncio.Queue):
        """completions 流式响应生成器."""
        base = _base(f"cmpl-{req_id}")
        while True:
            kind, payload = await queue.get()
            if kind == "done":
                r: GenerationResult = payload
                yield _sse({**base, "object": "text_completion",
                            "choices": [{"index": 0, "text": "", "finish_reason": r.finish_reason}],
                            "usage": _usage(r)})
                yield "data: [DONE]\n\n"
                return
            yield _sse({**base, "object": "text_completion",
                        "choices": [{"index": 0, "text": payload, "finish_reason": None}]})

    def _check_model(model_field: str | None) -> None:
        """请求里的 model 若与当前加载模型不一致则报错(与 vLLM 行为一致)."""
        if model_field is not None and model_field != engine.model_name:
            raise HTTPException(400, f"模型 {model_field} 未加载, 当前模型: {engine.model_name}")

    def _prompt_ids(text: str, sampling: SamplingParams) -> torch.Tensor:
        """文本 -> [1, n] token id(放到 CPU, 由引擎线程搬到模型设备)."""
        return torch.tensor(engine.tokenizer(text)["input_ids"]).unsqueeze(0)

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [{
                "id": engine.model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "lminfer",
            }],
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        _check_model(req.model)
        if req.n != 1:
            raise HTTPException(400, "朴素实现只支持 n=1")
        if req.suffix is not None:
            raise HTTPException(400, "朴素实现不支持 suffix 参数")
        if req.stream and req.n != 1:
            raise HTTPException(400, "朴素实现只支持 n=1")

        sampling = req.to_sampling()
        prompt_ids = _prompt_ids(req.prompt, sampling)
        req_id = uuid.uuid4().hex[:12]

        if req.stream:
            queue = await engine.generate(req_id, prompt_ids, sampling, stream=True)
            return StreamingResponse(_completion_streamer(req_id, queue),
                                     media_type="text/event-stream", headers=STREAM_HEADERS)

        r = await engine.generate(req_id, prompt_ids, sampling)
        return {**_base(f"cmpl-{req_id}"), "object": "text_completion",
                "choices": [{"index": 0, "text": r.output_text, "finish_reason": r.finish_reason}],
                "usage": _usage(r)}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        _check_model(req.model)
        if engine.tokenizer.chat_template is None:
            raise HTTPException(400, "当前模型没有 chat template, 请改用 /v1/completions")

        # 用 transformers 的 chat template 把消息列表渲染成 prompt(与参考脚本一致)
        kwargs = {"add_generation_prompt": True, "tokenize": True}
        if engine.config.enable_thinking is not None:
            kwargs["enable_thinking"] = engine.config.enable_thinking
        try:
            ids = engine.tokenizer.apply_chat_template(req.messages, **kwargs)
        except Exception as e:  # 模板缺参数等
            raise HTTPException(400, f"chat template 渲染失败: {e}")
        if hasattr(ids, "input_ids"):  # transformers 5.x 返回 tokenizers.Encoding
            ids = ids.input_ids
        prompt_ids = torch.tensor(ids).unsqueeze(0)

        sampling = req.to_sampling()
        req_id = uuid.uuid4().hex[:12]

        if req.stream:
            queue = await engine.generate(req_id, prompt_ids, sampling, stream=True)
            return StreamingResponse(_chat_streamer(req_id, queue),
                                     media_type="text/event-stream", headers=STREAM_HEADERS)

        r = await engine.generate(req_id, prompt_ids, sampling)
        return {**_base(f"chatcmpl-{req_id}"), "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": r.output_text},
                             "finish_reason": r.finish_reason}],
                "usage": _usage(r)}

    @app.get("/v1/stats")
    async def stats():
        return {
            "model": engine.model_name,
            "kv_bytes_per_token": engine.kv_bytes_per_token,
            "kv_mib_per_token": engine.kv_bytes_per_token / (1024 ** 2),
            "max_num_seqs": engine.config.max_num_seqs,
            "max_model_len": engine.config.max_model_len,
            **engine._stats,
        }

    return app


def run_server(config: EngineConfig, host: str = "0.0.0.0", port: int = 8000):
    """加载引擎并启动 uvicorn(供 cli 调用)."""
    import uvicorn

    engine = LLMEngine(config)
    app = create_app(engine)
    uvicorn.run(app, host=host, port=port, log_level="info")
