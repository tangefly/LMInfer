"""vLLM-backed inference with request-local segmented KV reuse.

The first backend uses serialized V1 requests and an in-process connector.
Each gap runs through normal vLLM prefill; intermediate sampled tokens are
discarded. The final stage uses normal vLLM decode and streaming detokenization.
"""
import asyncio
import functools
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoTokenizer

from .engine import GenerationResult, LLMEngine
from .kvcache import KVGraft, concat_cache, rebase_rope_cache, slice_cache, tail_cache
from .toolcalls import THINK_END, THINK_START
from .vllm_bridge import BRIDGES, Bridge, Stage
from .vllm_plan import plan_prefill

logger = logging.getLogger("lminfer")


def validate_model(config):
    if config.model_type != "qwen3":
        raise ValueError("vllm agent KV backend currently supports dense Qwen3 only")
    if getattr(config, "quantization_config", None):
        raise ValueError("Quantized models are not supported by this backend")
    if getattr(config, "use_sliding_window", False) or any(
        t != "full_attention" for t in getattr(config, "layer_types", [])
    ):
        raise ValueError("Only full-attention models are supported")
    rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        raise ValueError("Only default RoPE is supported")


class VLLMEngine(LLMEngine):
    def __init__(self, config):
        self.config = config
        self._lock = threading.Lock()
        self._closed = False
        if BRIDGES:
            raise RuntimeError("Only one live LMInfer vLLM engine is supported per process")
        if config.tensor_parallel_size != 1 or config.repair_mode == "context":
            raise ValueError("vllm backend requires TP=1 and window/exact repair")
        model_config = AutoConfig.from_pretrained(
            config.model, trust_remote_code=config.trust_remote_code)
        validate_model(model_config)
        # Connector snapshots are shared Python objects; fail instead of silently
        # launching another process with a separate registry.
        if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "0") != "0":
            raise ValueError("Set VLLM_ENABLE_V1_MULTIPROCESSING=0 for the LMInfer vllm backend")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        from vllm import LLM
        from vllm.config import KVTransferConfig

        self._bridge_id = uuid.uuid4().hex
        self._bridge = Bridge()
        BRIDGES[self._bridge_id] = self._bridge
        try:
            self.llm = LLM(
                model=config.model, dtype=config.dtype,
                trust_remote_code=config.trust_remote_code,
                max_model_len=config.max_model_len,
                max_num_seqs=1,
                max_num_batched_tokens=min(512, config.max_model_len),
                gpu_memory_utilization=config.gpu_memory_utilization,
                tensor_parallel_size=1,
                distributed_executor_backend="uni",
                enforce_eager=True,
                enable_prefix_caching=False,
                enable_chunked_prefill=True,
                async_scheduling=False,
                attention_config={"backend": "FLASH_ATTN"},
                kv_transfer_config=KVTransferConfig(
                    kv_connector="AgentKVConnector", kv_role="kv_both",
                    kv_connector_module_path="lminfer.vllm_connector",
                    kv_connector_extra_config={"bridge_id": self._bridge_id},
                ),
                disable_log_stats=config.disable_log_stats,
            )
        except BaseException:
            BRIDGES.pop(self._bridge_id, None)
            raise
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=config.trust_remote_code)
        dtype = self.llm.llm_engine.vllm_config.model_config.dtype
        # Server and KV utilities use this configuration, never a HF model.
        self.model = SimpleNamespace(config=model_config, dtype=dtype)
        self.model_name = config.served_model_name or config.model.rstrip("/").split("/")[-1]
        self._stats = {"completed": 0, "generated_tokens": 0, "prefill_tokens": 0}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lminfer-vllm")
        logger.info("vLLM backend: dense Qwen3, serialized requests, eager FLASH_ATTN; "
                    "global prefix caching disabled to isolate approximate KV")

    def close(self):
        self.executor.shutdown(wait=True)
        with self._lock:
            if not self._closed:
                self.llm.llm_engine.engine_core.shutdown()
                self._bridge.stage = None
                BRIDGES.pop(self._bridge_id, None)
                self._closed = True

    def _validate_cache(self, cache):
        c = self.model.config
        length = cache.get_seq_length()
        expected = (1, c.num_key_value_heads, length, c.head_dim)
        if len(cache.layers) != c.num_hidden_layers:
            raise ValueError("KV layer count does not match the model")
        for layer in cache.layers:
            if (not layer.is_initialized or tuple(layer.keys.shape) != expected
                    or tuple(layer.values.shape) != expected
                    or layer.keys.dtype != self.model.dtype
                    or layer.values.dtype != self.model.dtype):
                raise ValueError("KV snapshot shape/dtype does not match the model")

    def _stage(self, tokens, prefix, sampling, *, on_token=None, on_text=None):
        """Run one contiguous suffix with an imported prefix through vLLM."""
        if prefix is not None:
            self._validate_cache(prefix)
        stage = Stage(list(tokens), prefix, len(tokens) + sampling.max_tokens,
                      self.model.config.num_hidden_layers)
        self._bridge.stage = stage
        engine = self.llm.llm_engine
        req_id = "lminfer-" + uuid.uuid4().hex
        added = False
        started = time.perf_counter()
        first_ms = None
        emitted_tokens = 0
        emitted_text = ""
        output = None
        try:
            engine.add_request(req_id, {"prompt_token_ids": list(tokens)}, sampling)
            added = True
            while engine.has_unfinished_requests():
                for result in engine.step():
                    if result.request_id != req_id:
                        raise RuntimeError("Unexpected concurrent vLLM request")
                    if not result.outputs:
                        continue
                    output = result.outputs[0]
                    ids = list(output.token_ids)
                    if first_ms is None and ids:
                        first_ms = (time.perf_counter() - started) * 1000
                    if on_token:
                        for token in ids[emitted_tokens:]:
                            on_token(token)
                    emitted_tokens = len(ids)
                    if on_text and len(output.text) > len(emitted_text):
                        if not output.text.startswith(emitted_text):
                            raise RuntimeError("vLLM cumulative text changed after streaming")
                        on_text(output.text[len(emitted_text):])
                        emitted_text = output.text
            if output is None:
                raise RuntimeError("vLLM returned no completion")
            torch.cuda.synchronize()
            cache = stage.finish(self.model.config)
            return output, cache, {
                "computed_tokens": stage.computed_tokens,
                "loaded_tokens": stage.loaded_tokens,
                "first_ms": first_ms or (time.perf_counter() - started) * 1000,
            }
        finally:
            try:
                if added and engine.has_unfinished_requests():
                    engine.abort_request([req_id])
            finally:
                self._bridge.stage = None

    def _generate(self, request_id, prompt_ids, sampling, use_kv_cache=True,
                  on_token=None, skip_special_tokens=True, reuse_prefixes=None,
                  graft=None, *, on_text=None, consume_reuse=False):
        # Only request-owned lists may be consumed. Direct callers retain the
        # previous non-destructive behavior unless they explicitly opt in.
        if consume_reuse and any(x is not None and not isinstance(x, list)
                                 for x in (reuse_prefixes, graft)):
            raise TypeError("consume_reuse requires request-owned lists")
        with self._lock, torch.inference_mode():
            try:
                if self._closed:
                    raise RuntimeError("vLLM engine has been closed")
                return self._generate_locked(request_id, prompt_ids, sampling,
                                             use_kv_cache, on_token, skip_special_tokens,
                                             reuse_prefixes, graft, on_text, consume_reuse)
            finally:
                if consume_reuse:
                    for owned in (reuse_prefixes, graft):
                        if owned is not None:
                            owned.clear()

    def _prefill_reuse(self, tokens, reuse_prefixes, graft, consume_reuse):
        """Copy the selected prefix, then release each graft after insertion.

        All tensor-bearing plan references are dropped before the final vLLM
        stage starts. The returned metadata contains only scalar statistics.
        """
        from vllm import SamplingParams as VSamplingParams
        plan = span = source = None
        grafts = [graft] if isinstance(graft, KVGraft) else graft or []
        try:
            plan = plan_prefill(
                tokens, reuse_prefixes, grafts,
                begin=self.config.repair_window_begin, end=self.config.repair_window_end,
                exact=self.config.repair_mode == "exact")
            info = {"reused_tokens": plan.reused_tokens,
                    "graft_tokens": sum(s.end - s.start for s in plan.spans),
                    "exact_prefix_length": plan.exact_prefix_length,
                    "mismatch": plan.mismatch, "exact": not plan.spans,
                    "stages": 0, "computed_tokens": 0, "loaded_tokens": 0}
            if plan.prefix is not None:
                self._validate_cache(plan.prefix.cache)
            for span in plan.spans:
                self._validate_cache(span.graft.cache)
            span = None
            cache = (slice_cache(plan.prefix.cache, plan.prefix_length, self.model.config)
                     if plan.prefix_length else None)
            cur = plan.prefix_length
            plan.prefix = None
            if consume_reuse:
                if reuse_prefixes is not None:
                    reuse_prefixes.clear()
                if graft is not None:
                    graft.clear()
            grafts = None
            probe_sampling = VSamplingParams(temperature=0, max_tokens=1, detokenize=False)
            while plan.spans:
                span = plan.spans.pop(0)
                if cur < span.start:
                    _, cache, stats = self._stage(tokens[:span.start], cache, probe_sampling)
                    info["stages"] += 1
                    info["computed_tokens"] += stats["computed_tokens"]
                    info["loaded_tokens"] += stats["loaded_tokens"]
                source = tail_cache(span.graft.cache, span.source_offset, self.model.config)
                source = slice_cache(source, span.end - span.start, self.model.config)
                if self.config.graft_rope_rebase:
                    source = rebase_rope_cache(
                        source, span.graft.source_position + span.source_offset,
                        span.start, self.model.config)
                cache = source if cache is None else concat_cache(cache, source, self.model.config)
                cur = span.end
                # Do not leave the previous sub or its rebased copy live during
                # the next gap, or the final main prefill/decode stage.
                span = source = None
            return cache, info
        finally:
            if plan is not None:
                plan.prefix = None
                plan.spans.clear()
            span = source = grafts = None
            if consume_reuse:
                for owned in (reuse_prefixes, graft):
                    if owned is not None:
                        owned.clear()

    def _generate_locked(self, request_id, prompt_ids, sampling, use_kv_cache,
                         on_token, skip_special_tokens, reuse_prefixes, graft, on_text,
                         consume_reuse=False):
        started = time.perf_counter()
        from vllm import SamplingParams as VSamplingParams
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("Expected a single [1, tokens] prompt")
        if sampling.max_tokens <= 0 or sampling.max_tokens >= self.config.max_model_len:
            raise ValueError("max_tokens must be positive and smaller than max_model_len")
        tokens = prompt_ids[0].tolist()
        max_prompt = self.config.max_model_len - sampling.max_tokens
        truncated = len(tokens) > max_prompt
        if truncated:
            tokens = tokens[-max_prompt:]
            if consume_reuse:
                for owned in (reuse_prefixes, graft):
                    if owned is not None:
                        owned.clear()
            reuse_prefixes, graft = None, None
        if not use_kv_cache:
            raise ValueError("vLLM always uses decode KV; disable reuse via empty prefixes/grafts")
        cache, info = self._prefill_reuse(tokens, reuse_prefixes, graft, consume_reuse)
        stages = info["stages"]
        computed = info["computed_tokens"]
        loaded = info["loaded_tokens"]
        before_final_ms = (time.perf_counter() - started) * 1000
        vparams = VSamplingParams(
            temperature=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k,
            repetition_penalty=sampling.repetition_penalty, max_tokens=sampling.max_tokens,
            stop=sampling.stop or None, skip_special_tokens=skip_special_tokens)
        output, cache, stats = self._stage(tokens, cache, vparams,
                                          on_token=on_token, on_text=on_text)
        stages += 1
        computed += stats["computed_tokens"]
        loaded += stats["loaded_tokens"]
        expected_computed = cache.get_seq_length() - info["reused_tokens"]
        if computed != expected_computed:
            raise RuntimeError("Actual vLLM execution differs from the prefill reuse plan")
        generated = list(output.token_ids)
        eos = self.model.config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        while generated and generated[-1] in eos_ids:
            generated.pop()
        target = len(tokens) + len(generated)
        if cache.get_seq_length() > target:
            cache = slice_cache(cache, target, self.model.config)
        # The last sampled token may not have KV. Preserve the actual valid
        # range instead of adding an extra inference request just to fill it.
        valid = cache.get_seq_length()
        if not len(tokens) <= valid <= target:
            raise RuntimeError("Captured KV is outside the returned token sequence")
        think_len = 0
        think_open, think_close = self.tokenizer.convert_tokens_to_ids([THINK_START, THINK_END])
        if generated and generated[0] == think_open and think_close in generated:
            think_len = min(generated.index(think_close) + 1, valid - len(tokens))
        exact_len = info["exact_prefix_length"]
        exact_len = valid if exact_len is None else exact_len
        ttft_ms = before_final_ms + stats["first_ms"]
        total_ms = (time.perf_counter() - started) * 1000
        result = GenerationResult(
            request_id=request_id, prompt_tokens=len(tokens), output_tokens=generated,
            output_text=output.text, finish_reason=output.finish_reason or "length",
            ttft_ms=ttft_ms, decode_ms=max(0, total_ms - ttft_ms),
            kv_cache_bytes=valid * self.kv_bytes_per_token,
            reused_prompt_tokens=info["reused_tokens"],
            kv_cache=cache if use_kv_cache else None, output_think_tokens=think_len,
            kv_graft_mismatch=info["mismatch"], exact_prefix_len=exact_len,
            repair_stats={
                "backend": "vllm", "mode": self.config.repair_mode,
                "stages": stages, "graft_tokens": info["graft_tokens"],
                "computed_prompt_tokens": len(tokens) - info["reused_tokens"],
                "executed_tokens": computed, "loaded_tokens": loaded,
                "cached_sequence_tokens": valid, "truncated": truncated,
                "exact": info["exact"],
            })
        self._stats["completed"] += 1
        self._stats["generated_tokens"] += len(generated)
        self._stats["prefill_tokens"] += len(tokens) - info["reused_tokens"]
        logger.info("vLLM request %s: prompt=%d reused=%d stages=%d TTFT=%.1fms",
                    request_id, len(tokens), info["reused_tokens"], stages, ttft_ms)
        return result

    async def generate(self, request_id, prompt_ids, sampling, stream=False,
                       skip_special_tokens=True, reuse_prefixes=None, graft=None,
                       consume_reuse=False):
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue() if stream else None

        def on_text(text):
            loop.call_soon_threadsafe(queue.put_nowait, ("text", text))

        task = functools.partial(
            self._generate, request_id or uuid.uuid4().hex, prompt_ids, sampling,
            skip_special_tokens=skip_special_tokens, reuse_prefixes=reuse_prefixes,
            graft=graft, on_text=on_text if stream else None, consume_reuse=consume_reuse)
        future = loop.run_in_executor(self.executor, task)
        if not stream:
            return await future

        async def finish():
            try:
                await queue.put(("done", await future))
            except Exception as exc:
                logger.exception("vLLM streaming request failed")
                await queue.put(("error", str(exc)))
        loop.create_task(finish())
        return queue
