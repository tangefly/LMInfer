"""LLMEngine: 模型的加载、KV cache 生成循环与朴素并发管理.

设计说明(刻意保持朴素, 便于学习 KV Cache):
1. 每个请求独占一个 batch 位(batch=1), 不做拼接、不做 padding.
   因此 attention mask 恒为全 1, position_ids 用 transformers 默认值即可,
   语义最简单、最不容易出错.
2. KV cache 完全交给 transformers 的 DynamicCache 管理:
   - prefill: 一次性前向整个 prompt, 模型为每一层生成并缓存 K/V;
   - decode: 每步只输入 1 个 token, DynamicCache 自动把新 K/V 追加到末尾,
     我们不手工做任何 KV 操作, 只需把 cache 对象在步骤之间传下去.
3. 并发 = ThreadPoolExecutor(max_workers=max_num_seqs):
   多个请求在不同线程里各自跑生成循环. GPU 计算天然串行化,
   这就是"朴素版的连续批处理"——理解了这个, 再看 vLLM 的
   continuous batching 就明白它优化了什么.
"""

import asyncio
import functools
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DynamicCache,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .config import EngineConfig, SamplingParams

logger = logging.getLogger("lminfer")


@dataclass
class GenerationResult:
    """一次完整生成的结果与统计信息."""

    request_id: str
    prompt_tokens: int              # prompt 实际 token 数(截断后)
    output_tokens: list[int]        # 生成的 token id 列表
    output_text: str                # 解码后的文本(不含 stop 字符串)
    finish_reason: str              # "stop"(命中 eos/stop) | "length"(达上限)
    ttft_ms: float                  # 首 token 延迟(time to first token)
    decode_ms: float                # 首 token 之后的生成耗时
    kv_cache_bytes: int             # 结束时 KV cache 显存占用(字节)

    @property
    def completion_tokens(self) -> int:
        return len(self.output_tokens)

    @property
    def decode_tokens_per_sec(self) -> float:
        """decode 阶段吞吐(不含 prefill 与首 token 时间)."""
        return self.completion_tokens / (self.decode_ms / 1000) if self.decode_ms > 0 else float("inf")


class LLMEngine:
    """加载模型, 提供同步生成循环与异步(HTTP)入口."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self._load_model()

        # 朴素并发: 一个线程跑一个请求的完整生成循环.
        # 线程数 = max_num_seqs, 多出的请求在 executor 队列里等待(相当于排队).
        self.executor = ThreadPoolExecutor(
            max_workers=config.max_num_seqs, thread_name_prefix="lminfer-gen"
        )

        # 全局统计(供 /v1/stats 与日志展示)
        self._stats = {"completed": 0, "generated_tokens": 0, "prefill_tokens": 0}

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _load_model(self):
        t0 = time.time()
        dtype_map = {"auto": "auto", "bfloat16": torch.bfloat16,
                     "float16": torch.float16, "float32": torch.float32}
        if self.config.dtype not in dtype_map:
            raise ValueError(f"不支持的 dtype: {self.config.dtype}, 可选 {list(dtype_map)}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model, trust_remote_code=self.config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            # 与 vLLM 相同: 无 pad token 的模型用 eos token 兜底
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # attn_implementation: "auto" 传 None, 让 transformers 用默认(sdpa);
        # 显式指定(如 flash_attention_2)则透传, 未安装 flash-attn 时由 transformers 回退 kernels 或报错
        attn_impl = None if self.config.attn_implementation == "auto" else self.config.attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=dtype_map[self.config.dtype],
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            attn_implementation=attn_impl,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        # 对外模型名: 优先用 --served-model-name(与 vLLM 语义一致), 否则取路径最后一段
        self.model_name = (self.config.served_model_name
                           or self.config.model.rstrip("/").split("/")[-1]
                           or self.config.model)

        # 打印实际解析出的 attention 实现(transformers 会把 auto/回退解析为具体值)
        resolved_impl = (getattr(self.model.config, "_attn_implementation_internal", None)
                         or self.model.config._attn_implementation or "auto")
        logger.info("模型加载完成: %s (%.2fs), attention 实现: %s",
                    self.model_name, time.time() - t0, resolved_impl)
        logger.info("KV cache 每 token 占用(理论): %.2f MiB", self.kv_bytes_per_token / (1024 ** 2))

    @property
    def kv_bytes_per_token(self) -> int:
        """每生成 1 个 token, 全部层新增 K+V 的显存字节数.

        公式: 2(K 和 V) x 层数 x KV头数 x head_dim x 每元素字节数
        这是理解 KV cache 内存开销的关键数字.
        """
        c = self.model.config
        num_layers = c.num_hidden_layers
        num_kv_heads = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
        head_dim = getattr(c, "head_dim", None) or (c.hidden_size // c.num_attention_heads)
        dtype_size = torch.tensor([], dtype=self.model.dtype).element_size()
        return 2 * num_layers * num_kv_heads * head_dim * dtype_size

    def _eos_ids(self) -> set[int]:
        """收集所有需要触发停止的 eos token id."""
        ids: set[int] = set()
        for v in (self.model.config.eos_token_id, self.model.generation_config.eos_token_id):
            if isinstance(v, (list, tuple)):
                ids.update(v)
            elif v is not None:
                ids.add(v)
        return ids

    # ------------------------------------------------------------------
    # 采样
    # ------------------------------------------------------------------
    def _sample(self, logits: torch.Tensor, input_ids: torch.Tensor,
                params: SamplingParams) -> int:
        """对 [vocab] 形状的 logits 采样, 返回下一个 token id.

        temperature == 0 时贪心; 否则用 transformers 的 warper 处理
        (temperature / top_p / top_k / repetition_penalty, 与 generate 行为一致).
        """
        if params.greedy:
            return int(logits.argmax(-1).item())

        processors = LogitsProcessorList()
        if params.temperature != 1.0:
            processors.append(TemperatureLogitsWarper(params.temperature))
        if params.top_p < 1.0:
            processors.append(TopPLogitsWarper(params.top_p))
        if params.top_k > 0:
            processors.append(TopKLogitsWarper(params.top_k))
        if params.repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(params.repetition_penalty))

        # LogitsProcessorList 的调用约定是 (input_ids, logits), 需要 [1, vocab] 形状
        processed = processors(input_ids, logits.unsqueeze(0))
        probs = torch.softmax(processed[0], dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ------------------------------------------------------------------
    # 核心: 朴素生成循环
    # ------------------------------------------------------------------
    def _generate(
        self,
        request_id: str,
        prompt_ids: torch.Tensor,          # [1, n] 的 prompt token id
        sampling: SamplingParams,
        use_kv_cache: bool = True,         # False 仅用于理论对比实验
        on_token: Callable[[int], None] | None = None,  # 每生成一个 token 回调(流式输出用)
        skip_special_tokens: bool = True,  # False 用于工具调用: 保留 <tool_call> 等标记供解析
    ) -> GenerationResult:
        """prefill + decode 的生成循环, 返回完整结果.

        两种模式的对照(这正是"KV cache 省了什么"的实验基础):
        - use_kv_cache=True : decode 每次只喂 1 个 token, 其余全靠 cache
        - use_kv_cache=False: decode 每次把整个前缀重新前向(无缓存)
        """
        # 截断 prompt, 保证总长度不超过 max_model_len
        max_prompt = self.config.max_model_len - sampling.max_tokens
        if prompt_ids.shape[1] > max_prompt:
            logger.warning("请求 %s: prompt %d tokens 超过上限 %d, 截断头部保留末尾",
                           request_id, prompt_ids.shape[1], max_prompt)
            prompt_ids = prompt_ids[:, -max_prompt:]
        prompt_ids = prompt_ids.to(self.model.device)
        n_prompt = prompt_ids.shape[1]

        eos_ids = self._eos_ids()
        generated: list[int] = []                    # 已生成的 token id
        all_ids = prompt_ids                         # 完整序列 = prompt + 已生成(采样器用)
        cache = DynamicCache(config=self.model.config)  # KV cache 容器
        t0 = time.perf_counter()
        ttft_ms = 0.0

        def _on_token(token: int) -> None:
            generated.append(token)
            if on_token is not None:
                on_token(token)

        with torch.inference_mode():
            # ---- prefill: 一次性前向整个 prompt ----
            # 有缓存模式: 模型把每一层的 K/V 存入 cache, 后续 decode 全部复用;
            # 无缓存模式: 同样只做一次前向得到首 token, 但 cache 保持为空.
            out = self.model(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                past_key_values=cache if use_kv_cache else None,
                use_cache=use_kv_cache,
            )
            ttft_ms = (time.perf_counter() - t0) * 1000
            token = self._sample(out.logits[:, -1, :], all_ids, sampling)
            _on_token(token)
            all_ids = torch.cat([all_ids, torch.tensor([[token]], device=self.model.device)], dim=-1)

            # ---- decode: 逐 token 生成, 直到 eos / stop 子串 / 达上限 ----
            output_text = None
            while len(generated) < sampling.max_tokens:
                if use_kv_cache:
                    # 只输入最后一个 token; mask 覆盖 [历史 + 当前];
                    # 新 K/V 由模型在 forward 内部追加进 cache, 我们只传递对象
                    cur = torch.tensor([[token]], device=self.model.device)
                    mask = torch.ones(1, cache.get_seq_length() + 1, device=self.model.device)
                    out = self.model(
                        input_ids=cur, attention_mask=mask,
                        past_key_values=cache, use_cache=True,
                    )
                else:
                    # 无缓存对照模式: 把整个前缀重新前向(注意力计算量随长度线性增长)
                    out = self.model(
                        input_ids=all_ids, attention_mask=torch.ones_like(all_ids),
                        use_cache=False,
                    )

                token = self._sample(out.logits[:, -1, :], all_ids, sampling)
                if token in eos_ids:
                    # eos 不入输出(与 vLLM 一致); 先检查再回调, 流式也不会漏出
                    finish_reason = "stop"
                    break
                _on_token(token)
                all_ids = torch.cat([all_ids, torch.tensor([[token]], device=self.model.device)], dim=-1)

                # 朴素 stop 检查: 每步重解出全部文本并找子串(简单但 O(n), 足够教学)
                text = self.tokenizer.decode(generated, skip_special_tokens=skip_special_tokens)
                stopped = next((s for s in sampling.stop if s in text), None)
                if stopped is not None:
                    finish_reason = "stop"
                    output_text = text.split(stopped)[0]  # 去掉 stop 字符串及其后内容
                    break
            else:
                # 循环条件不满足退出 = 达到 max_tokens
                finish_reason = "length"

            if output_text is None:
                # 兜底: prefill 首 token 即 eos 的极端情况, 同样不入输出
                while generated and generated[-1] in eos_ids:
                    generated.pop()
                output_text = self.tokenizer.decode(generated, skip_special_tokens=skip_special_tokens)

        # ---- 统计 ----
        total_ms = (time.perf_counter() - t0) * 1000
        kv_bytes = sum(
            int(layer.keys.numel() + layer.values.numel()) * layer.keys.element_size()
            for layer in cache.layers if layer.is_initialized
        )
        result = GenerationResult(
            request_id=request_id,
            prompt_tokens=n_prompt,
            output_tokens=generated,
            output_text=output_text,
            finish_reason=finish_reason,
            ttft_ms=ttft_ms,
            decode_ms=max(total_ms - ttft_ms, 0.0),
            kv_cache_bytes=kv_bytes,
        )

        # 全局统计 + 每请求日志(风格类似 vLLM 的日志行)
        self._stats["completed"] += 1
        self._stats["generated_tokens"] += result.completion_tokens
        self._stats["prefill_tokens"] += n_prompt
        if not self.config.disable_log_stats:
            logger.info(
                "请求 %s: prompt %d tok, 生成 %d tok, TTFT %.0fms, %.1f tok/s, KV cache %.2f MiB, %s",
                request_id, n_prompt, result.completion_tokens, ttft_ms,
                result.decode_tokens_per_sec, kv_bytes / (1024 ** 2), finish_reason,
            )
        return result

    # ------------------------------------------------------------------
    # 异步(HTTP)入口
    # ------------------------------------------------------------------
    async def generate(self, request_id: str | None, prompt_ids: torch.Tensor,
                       sampling: SamplingParams, stream: bool = False,
                       skip_special_tokens: bool = True):
        """异步生成.

        返回:
          - stream=False: await 得到 GenerationResult
          - stream=True : 返回 asyncio.Queue, 队列元素为 ("text", 文本)
                          或 ("done", GenerationResult) 哨兵
        """
        request_id = request_id or uuid.uuid4().hex[:8]
        loop = asyncio.get_running_loop()

        if stream:
            queue: asyncio.Queue = asyncio.Queue()

            def on_token(tok: int):
                # 在 worker 线程里把 token 解码成文本, 再安全塞回事件循环
                text = self.tokenizer.decode(tok, skip_special_tokens=skip_special_tokens)
                loop.call_soon_threadsafe(queue.put_nowait, ("text", text))

            fut = loop.run_in_executor(
                self.executor,
                functools.partial(
                    self._generate, request_id, prompt_ids, sampling, True, on_token,
                    skip_special_tokens,
                ),
            )

            async def _finish():
                # 生成结束后投递哨兵, 通知响应生成器收尾
                result = await asyncio.wrap_future(fut)
                await queue.put(("done", result))

            loop.create_task(_finish())
            return queue

        result = await asyncio.wrap_future(
            loop.run_in_executor(
                self.executor,
                functools.partial(
                    self._generate, request_id, prompt_ids, sampling, True, None,
                    skip_special_tokens,
                ),
            )
        )
        return result
