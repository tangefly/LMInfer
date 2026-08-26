"""FastAPI 服务: 提供 vLLM 风格的 OpenAI 兼容 HTTP 接口.

接口列表:
  GET  /health               健康检查(200 = 可用)
  GET  /v1/models            模型列表
  POST /v1/completions       文本补全(支持 stream)
  POST /v1/chat/completions  对话补全(支持 stream, 支持 agent 模式)
  GET  /v1/agent/sessions    agent 模式会话列表(朴素实现额外提供的接口)
  GET  /v1/stats             运行时统计(朴素实现额外提供的接口)
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Callable

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config import EngineConfig, SamplingParams
from .engine import GenerationResult, LLMEngine
from .kvcache import KIND_MAIN, KIND_SUB, SessionKVStore
from .model_adapters import resolve_model_profile
from .schemas import ChatCompletionRequest, ChatMessage, CompletionRequest
from .sessions import AgentSessionRegistry
from .toolcalls import (
    THINK_BLOCK,
    LlamaJsonStreamSplitter,
    ToolCallStreamSplitter,
    clean_content,
    clean_llama3_json_content,
    parse_llama3_json_tool_calls,
    parse_tool_calls,
)

logger = logging.getLogger("lminfer")

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

    # agent 模式会话注册表(服务层状态; handler 都在事件循环上执行, 无需加锁)
    agent_sessions = AgentSessionRegistry()
    # 模型适配参数: --tool-call-parser auto 在这里解析成具体解析器, 并给出
    # 模板渲染所需的参数(arguments 还原成 dict / 工具结果包成 {"output": ...})
    profile = resolve_model_profile(engine.config.tool_call_parser,
                                    engine.tokenizer, engine.model.config)
    if engine.config.tool_call_parser == "auto":
        logger.info("tool-call-parser=auto 自动识别为: %s", profile.tool_parser)
    # 跨请求前缀 KV 复用存储(仅 --reuse-agent-kv / --reuse-agent-kv-append 时使用, 见 kvcache.py)
    # tokenizer 供拼接模式取 <tool_response> 包裹标记的 token id(见 SessionKVStore.build_graft)
    kv_store = SessionKVStore(config=engine.model.config,
                              tokenizer=engine.tokenizer,
                              idle_ttl=engine.config.kv_segment_idle_ttl)

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

    def _tool_call_delta(call: dict, index: int) -> dict:
        """把一个工具调用变成流式 delta 分片(OpenAI 流式格式)."""
        return {
            "index": index,
            "id": call["id"],
            "type": "function",
            "function": {"name": call["function"]["name"],
                         "arguments": call["function"]["arguments"]},
        }

    async def _chat_streamer(req_id: str, queue: asyncio.Queue,
                             splitter: ToolCallStreamSplitter | None = None,
                             extra: dict | None = None,
                             on_done: Callable[[GenerationResult], None] | None = None):
        """chat 流式响应生成器: role 分片 -> 文本分片 -> 结束分片 -> [DONE].

        带 splitter 时(请求带 tools): <tool_call> 块不进入 content, 解析后
        以 tool_calls delta 发出, finish_reason 为 "tool_calls".
        extra: agent 模式时追加到每个 chunk 顶层的字段(session_id/trace),
        客户端从首个 chunk 就能拿到; on_done: 生成结束时回调(累计会话用量).
        """
        base = {**_base(f"chatcmpl-{req_id}"), **(extra or {})}

        def _chunk(delta: dict, finish_reason: str | None) -> str:
            return _sse({**base, "object": "chat.completion.chunk",
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]})

        def _emit(events):
            """把 splitter 事件变成响应分片, 返回发出的工具调用个数."""
            nonlocal call_index
            for ev_kind, ev_payload in events:
                if ev_kind == "content":
                    yield _chunk({"content": ev_payload}, None)
                else:
                    yield _chunk({"tool_calls": [_tool_call_delta(ev_payload, call_index)]}, None)
                    call_index += 1

        call_index = 0
        yield _chunk({"role": "assistant"}, None)
        while True:
            kind, payload = await queue.get()
            if kind == "done":
                r: GenerationResult = payload
                if on_done is not None:
                    on_done(r)
                if splitter is not None:
                    for chunk in _emit(splitter.flush()):
                        yield chunk
                finish = "tool_calls" if call_index else r.finish_reason
                yield _chunk({}, finish)
                yield "data: [DONE]\n\n"
                return
            if splitter is None:
                yield _chunk({"content": payload}, None)
            else:
                for chunk in _emit(splitter.push(payload)):
                    yield chunk

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

    def _message_dicts(messages: list[ChatMessage]) -> list[dict]:
        """pydantic 消息转纯 dict, 交给 chat template 渲染.

        模板里既可能写 message['role'] 也可能写 message.role(jinja 的 `.` 对 dict
        会回退到下标访问, 对 pydantic 对象则只支持属性访问), 统一转 dict 才能
        适配任意模板; tool_calls / tool_call_id / content(null) 原样透传,
        不遗漏客户端 post 过来的工具调用字段.

        Llama 3.x 适配(profile.arguments_as_dict / wrap_tool_output): 官方模板把
        OpenAI 格式的 tool_calls.arguments(JSON 字符串)直接 | tojson 会渲染成
        "parameters": "{\"city\": ...}"(字符串被加引号), 工具结果 content 字符串
        也会被加引号 —— 渲染前把 arguments 还原成 dict、工具结果包成 {"output": ...}
        对象, 才能得到模型训练时见到的合法 JSON(与 vLLM 的 llama3.1_json 模板一致).
        """
        out = []
        for m in messages:
            content = m.content
            # 历史 assistant 消息里的 think 块在渲染前剔除: 返回文本保留 think(见
            # clean_content), 但思考内容重新进入 prompt 会让 Qwen3 后续生成退化
            # (不闭合 think 就调工具/空思考+答非所问), 渲染期剥掉与 vLLM 行为一致
            if m.role == "assistant" and isinstance(content, str):
                content = THINK_BLOCK.sub("", content)
            d = {"role": m.role, "content": content}
            if m.tool_calls is not None:
                d["tool_calls"] = _adapt_tool_calls(m.tool_calls)
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
                if profile.wrap_tool_output and isinstance(d["content"], str):
                    # Llama 3.x: 工具结果渲染成 {"output": ...} 对象(模板对
                    # 非字符串内容 | tojson 才不加引号)
                    d["content"] = {"output": d["content"]}
            out.append(d)
        return out

    def _adapt_tool_calls(tool_calls: list[dict]) -> list[dict]:
        """按模型适配参数调整 assistant 消息里的 tool_calls, 供模板渲染.

        Llama 3.x: OpenAI 协议里 arguments 是 JSON 字符串, 模板却按对象渲染
        (tool_call.arguments | tojson), 需要把字符串解析回 dict; 解析失败则
        原样保留(模板会渲染成带引号的字符串, 与协议一致, 不报错).
        """
        if not profile.arguments_as_dict:
            return tool_calls
        out = []
        for call in tool_calls:
            fn = call.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try:
                    fn = {**fn, "arguments": json.loads(fn["arguments"])}
                except json.JSONDecodeError:
                    pass
            out.append({**call, "function": fn})
        return out

    def _check_model(model_field: str | None) -> None:
        """请求里的 model 若与当前加载模型不一致则报错(与 vLLM 行为一致)."""
        if model_field is not None and model_field != engine.model_name:
            raise HTTPException(400, f"模型 {model_field} 未加载, 当前模型: {engine.model_name}")

    def _prompt_ids(text: str, sampling: SamplingParams) -> torch.Tensor:
        """文本 -> [1, n] token id(放到 CPU, 由引擎线程搬到模型设备)."""
        return torch.tensor(engine.tokenizer(text)["input_ids"]).unsqueeze(0)

    def _handle_agent_request(req: ChatCompletionRequest) -> str | None:
        """agent 模式: 校验 trace、取回/新建会话; chat 模式返回 None.

        stream 与非 stream 都先走这一步, 流式首 chunk 就能回传 session_id.
        返回的 session_id 同时用于: 响应回传、生成结束后累计会话用量.
        """
        if req.mode != "agent":
            return None  # chat 模式: session_id/trace 字段忽略, 保持 OpenAI 兼容
        if not req.trace:
            raise HTTPException(400, "agent 模式必须提供 trace 调用路径")
        try:
            session_id, _ = agent_sessions.get_or_create(req.session_id, req.trace)
        except KeyError:
            raise HTTPException(404, f"会话 {req.session_id} 不存在, 请先发起不带 session_id 的 agent 请求")
        return session_id

    def _agent_kv_finish(session_id: str, trace: list[str], prompt_ids: torch.Tensor,
                         r: GenerationResult, attempted: bool) -> None:
        """agent 请求生成结束后: 累计复用统计, 并保存本次完整序列 KV.

        保存的序列 = 实际 prefill 的 prompt + 生成输出, 与 KV cache 逐位对齐、
        位置从 0 开始 —— 下一次请求即可作为前缀复用候选. 被截断的请求不保存
        (尾锚定段与下一轮完整历史无法对齐, 且会覆盖之前保存的好段).
        """
        if r.reused_prompt_tokens > 0:
            kv_store.note_hit(r.reused_prompt_tokens)
            agent_sessions.record_kv_reuse(session_id, r.reused_prompt_tokens)
            logger.info("请求 %s: 会话 %s trace %s KV 前缀复用 %d tok"
                        "(prompt %d tok, 跳过 %.0f%% prefill; 来源见引擎日志)",
                        r.request_id, session_id, trace, r.reused_prompt_tokens,
                        r.prompt_tokens,
                        100.0 * r.reused_prompt_tokens / max(r.prompt_tokens, 1))
        elif attempted:
            logger.info("请求 %s: trace %s 触发 KV 复用但前缀不匹配, 回退全量 prefill "
                        "(常见原因: chat template 对末尾 assistant 消息重新渲染, "
                        "如 Qwen3 的 <think> 块)", r.request_id, trace)
        if r.kv_cache is None:
            return
        if r.kv_graft_mismatch:
            kv_store.note_graft_mismatch()
        # 被截断的请求不保存: 保存段尾锚定(从截断点起), 与下一轮完整历史的头部
        # 必然错位, 永远无法作为前缀复用 —— 单槽位 store 下还会覆盖之前的好段
        if prompt_ids.shape[1] > r.prompt_tokens:
            logger.info("请求 %s: prompt 被截断(%d -> %d tok), 跳过 KV 保存",
                        r.request_id, prompt_ids.shape[1], r.prompt_tokens)
            return
        seq_tokens = prompt_ids[0][-r.prompt_tokens:].tolist() + r.output_tokens
        kind = KIND_MAIN if trace[-1] == KIND_MAIN else KIND_SUB
        # prompt_len/think_len 用于拼接模式: 记录输出 KV 起始位置与开头
        # <think> 块的 token 数, 拼接时切出/剔除对应 KV
        kv_store.put(session_id, kind, seq_tokens, r.kv_cache,
                     prompt_len=r.prompt_tokens, think_len=r.output_think_tokens)

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

        # agent 模式: 建/取会话(见 _handle_agent_request)
        session_id = _handle_agent_request(req)

        # ---- tool_choice 解析(vLLM 语义) ----
        # 请求显式给出的 tool_choice 优先; 未给出时由 --enable-auto-tool-choice
        # 决定默认值: 开启 = "auto"(带 tools 即渲染), 关闭 = "none"(忽略 tools).
        # 指定单个函数({"type":"function","function":{"name":...}}) 时只渲染该
        # 工具(与 vLLM 一致). 本地 Qwen 模板只认 tools 参数, 传入 tool_choice
        # 会被忽略, 语义完全由这里的分支决定.
        choice = req.tool_choice
        if choice is None:
            choice = "auto" if engine.config.enable_auto_tool_choice else "none"
        choice_name: str | None = None  # 传给模板的 tool_choice 值(auto/required/函数名)
        if choice == "none":
            tools = None  # 不渲染工具(显式 none 与默认 none 行为一致)
        elif isinstance(choice, str):
            if choice not in ("auto", "required"):
                raise HTTPException(400, f"不支持的 tool_choice: {choice}")
            if not req.tools:
                # vLLM 语义: auto 在无 tools 时退化为普通对话(工具开关不改变
                # 纯聊天请求的行为); 只有 required 才强制要求 tools
                if choice == "required":
                    raise HTTPException(400, f"tool_choice={choice} 但请求未提供 tools")
                tools = None
            else:
                tools, choice_name = req.tools, choice
        else:
            # dict 形式: {"type": "function", "function": {"name": ...}}, 只保留该工具
            name = (choice.get("function") or {}).get("name") if isinstance(choice, dict) else None
            if not isinstance(name, str) or not name:
                raise HTTPException(400, "tool_choice 格式错误: 需要 "
                                         '{"type": "function", "function": {"name": ...}}')
            if not req.tools:
                raise HTTPException(400, f"tool_choice 指定了函数 {name} 但请求未提供 tools")
            tools = [t for t in req.tools
                     if isinstance(t.get("function"), dict)
                     and t["function"].get("name") == name]
            if not tools:
                raise HTTPException(400, f"tool_choice 指定的函数 {name} 不在 tools 中")
            choice_name = name
        # 只有渲染了工具才需要解析工具调用输出(此时保留特殊 token 供解析);
        # 具体解析器由模型适配层决定(--tool-call-parser auto 自动识别模型家族):
        #   hermes     : Qwen/Hermes 系, 解析 <tool_call> 块;
        #   llama3_json: Llama 3.x 系, 解析 {"name":..., "parameters":...} JSON.
        parse_tools = bool(tools) and profile.tool_parser != "none"

        # 用 transformers 的 chat template 把消息列表渲染成 prompt(与参考脚本一致)
        kwargs = {"add_generation_prompt": True, "tokenize": True}
        if tools is not None:
            kwargs["tools"] = tools
        if choice_name is not None:
            # 本地模板无 tool_choice 参数会自动忽略; 支持它的模板按 vLLM 语义渲染
            kwargs["tool_choice"] = choice_name
        # 请求级 enable_thinking 优先, 其次服务端配置(--enable-thinking/--no-enable-thinking)
        if req.enable_thinking is not None:
            kwargs["enable_thinking"] = req.enable_thinking
        elif engine.config.enable_thinking is not None:
            kwargs["enable_thinking"] = engine.config.enable_thinking
        try:
            ids = engine.tokenizer.apply_chat_template(_message_dicts(req.messages), **kwargs)
            new_kwargs = kwargs.copy()
            new_kwargs["tokenize"] = False
            text_prompt = engine.tokenizer.apply_chat_template(_message_dicts(req.messages), **new_kwargs)
        except Exception as e:  # 模板缺参数、模板不支持 tools 等
            raise HTTPException(400, f"chat template 渲染失败: {e}")
        if hasattr(ids, "input_ids"):  # transformers 5.x 返回 tokenizers.Encoding
            ids = ids.input_ids
        prompt_ids = torch.tensor(ids).unsqueeze(0)

        # 跨请求前缀 KV 复用(agent 模式):
        # - --reuse-agent-kv       : LCP 安全模式, 引擎做 token 级匹配, 不匹配回退
        # - --reuse-agent-kv-append: 位置感知拼接模式(实验), 在 prompt 中定位子 agent
        #                            输出正文, 把其 KV 直接插进 main 的 KV cache;
        #                            main 段同时作为拼接基础与回退候选(定位失败回退 LCP)
        reuse_prefixes, graft = None, None
        if session_id is not None and engine.config.reuse_agent_kv_append:
            prompt_tokens = prompt_ids[0].tolist()
            graft = kv_store.build_graft(session_id, req.trace, prompt_tokens)
            reuse_prefixes = kv_store.propose(session_id, req.trace)
        elif session_id is not None and engine.config.reuse_agent_kv:
            reuse_prefixes = kv_store.propose(session_id, req.trace)

        sampling = req.to_sampling()
        req_id = uuid.uuid4().hex[:12]

        if req.stream:
            splitter = (LlamaJsonStreamSplitter() if profile.tool_parser == "llama3_json"
                        else ToolCallStreamSplitter()) if parse_tools else None
            queue = await engine.generate(req_id, prompt_ids, sampling, stream=True,
                                          skip_special_tokens=not parse_tools,
                                          reuse_prefixes=reuse_prefixes,
                                          graft=graft)
            extra, on_done = None, None
            if session_id is not None:
                # 每个 SSE chunk 顶层都带会话字段; 生成结束(done)时累计用量并保存 KV
                extra = {"session_id": session_id, "trace": req.trace}
                attempted = bool(reuse_prefixes) or graft is not None
                kv_reuse_on = engine.config.reuse_agent_kv or engine.config.reuse_agent_kv_append

                def on_done(r):
                    agent_sessions.record_usage(
                        session_id, r.prompt_tokens, r.completion_tokens)
                    if kv_reuse_on:
                        _agent_kv_finish(session_id, req.trace, prompt_ids, r, attempted)

            return StreamingResponse(_chat_streamer(req_id, queue, splitter, extra, on_done),
                                     media_type="text/event-stream", headers=STREAM_HEADERS)

        r = await engine.generate(req_id, prompt_ids, sampling,
                                  skip_special_tokens=not parse_tools,
                                  reuse_prefixes=reuse_prefixes,
                                  graft=graft)
        if parse_tools:
            if profile.tool_parser == "llama3_json":
                # Llama 3.x: 模型输出 JSON 工具调用; 有 tool_calls 时 content 为
                # null(vLLM 语义), 无 tool_calls 时返回文本(剥掉 <|python_tag|> 前缀)
                tool_calls = parse_llama3_json_tool_calls(r.output_text)
                content = None if tool_calls else (
                    clean_llama3_json_content(r.output_text) or None)
            else:
                tool_calls = parse_tool_calls(r.output_text)
                content = clean_content(r.output_text) or None
        else:
            tool_calls, content = [], r.output_text
        message: dict = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        resp = {**_base(f"chatcmpl-{req_id}"), "object": "chat.completion",
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "tool_calls" if tool_calls else r.finish_reason}],
                "usage": _usage(r)}
        if session_id is not None:
            agent_sessions.record_usage(session_id, r.prompt_tokens, r.completion_tokens)
            if engine.config.reuse_agent_kv or engine.config.reuse_agent_kv_append:
                attempted = bool(reuse_prefixes) or graft is not None
                _agent_kv_finish(session_id, req.trace, prompt_ids, r, attempted)
            resp["session_id"] = session_id
            resp["trace"] = req.trace
            # 实验观测字段: 本次请求跳过 prefill 的 token 数(含拼接的子 agent 输出 KV),
            # 供客户端验证 --reuse-agent-kv / --reuse-agent-kv-append 的效果
            resp["reused_prompt_tokens"] = r.reused_prompt_tokens
        return resp

    @app.get("/v1/agent/sessions")
    async def agent_sessions_list():
        """agent 模式会话列表: 会话 id / 最新 trace / agent 名单 / 请求数与 token 累计."""
        return {"object": "list", "data": agent_sessions.list_sessions()}

    @app.get("/v1/stats")
    async def stats():
        return {
            "model": engine.model_name,
            "kv_bytes_per_token": engine.kv_bytes_per_token,
            "kv_mib_per_token": engine.kv_bytes_per_token / (1024 ** 2),
            "max_num_seqs": engine.config.max_num_seqs,
            "max_model_len": engine.config.max_model_len,
            **engine._stats,
            "kv_reuse": kv_store.stats,
        }
        
    @app.post("/v1/release")
    async def release(req: dict):
        if "session_id" not in req:
            logger.info("release 请求不含 session_id 字段")
            return {"state": False}
        if req["session_id"] is None:
            return {"state": False}
        kv_store.release(req["session_id"])
        return {"state": True}

    return app

def run_server(config: EngineConfig, host: str = "0.0.0.0", port: int = 8000):
    """加载引擎并启动 uvicorn(供 cli 调用)."""
    import uvicorn

    engine = LLMEngine(config)
    app = create_app(engine)
    uvicorn.run(app, host=host, port=port, log_level="info")
