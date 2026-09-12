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
from dataclasses import dataclass, field
from typing import Callable

import torch
from transformers import (
    AutoTokenizer,
    DynamicCache,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .config import EngineConfig, SamplingParams
from .model_adapters import (kv_bytes_per_token, load_text_model,
                             resolve_logits_kwargs, resolve_rotary_emb)
from .repair import repair_token_counts
from .context_repair import context_prefill, exact_prefill
from .kvcache import (
    KIND_MAIN,
    KIND_SUB,
    KVGraft,
    KVPrefix,
    concat_cache,
    longest_common_prefix,
    rebase_rope_cache,
    slice_cache,
    tail_cache,
)
from .toolcalls import THINK_END, THINK_START

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
    reused_prompt_tokens: int = 0   # 本次请求复用前缀 KV 的 token 数(0 = 全量 prefill)
    kv_cache: DynamicCache | None = None  # 完整 KV cache(prompt + 输出, 与序列严格对齐),
                                          # 供 agent 会话间复用保存; 无缓存模式为 None
    output_think_tokens: int = 0    # 输出开头 <think> 块的 token 数(拼接模式剔除用:
                                    # think 不作为下一轮对话的 prompt, 拼接 KV 时挖掉)
    kv_graft_mismatch: bool = False  # 位置感知拼接模式: 插入位置/长度/token 校验失败(已回退)

    exact_prefix_len: int | None = None
    repair_stats: dict = field(default_factory=dict)

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

        # <think>/</think> 特殊 token id(拼接模式剔除 think KV 用);
        # 模型没有这两个 token 时置 None, think 检测自动禁用。
        # 必须校验 id 能反查回同名 token: 词表里没有时 convert_tokens_to_ids 会
        # 返回 unk id, 只判 >=0 会把 <unk> 当成 think 标记(Ministral 等无 think
        # 模型的输出首个 token 恰好是 <unk> 时会误判)。
        ids = self.tokenizer.convert_tokens_to_ids([THINK_START, THINK_END])
        self._think_ids = (ids[0], ids[1]) if all(
            isinstance(i, int) and i >= 0
            and self.tokenizer.convert_ids_to_tokens(i) == t
            for i, t in zip(ids, (THINK_START, THINK_END))) else None

        # attn_implementation: "auto" 传 None, 让 transformers 用默认(sdpa);
        # 显式指定(如 flash_attention_2)则透传, 未安装 flash-attn 时由 transformers 回退 kernels 或报错
        attn_impl = None if self.config.attn_implementation == "auto" else self.config.attn_implementation
        # 模型加载的家族差异(多模态包装 / fine-grained FP8)交给适配层(见 model_adapters.py)
        self.model, text_config, notes = load_text_model(
            self.config.model,
            dtype=dtype_map[self.config.dtype],
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
            attn_implementation=attn_impl,
            dequantize_fp8=self.config.dequantize_fp8,
        )
        for note in notes:
            logger.info("模型适配: %s", note)
        self.model.eval()
        self.model.requires_grad_(False)
        # KV 形状/位置/RoPE 计算一律基于**文本塔** config(见 model_config 属性)
        self._model_config = text_config
        # 文本解码器的 RoPE 模块: 拼接模式的 K 位置重映射必须用模型自己的逆频率
        # (Ministral-3 是 YaRN, 与默认 theta 公式不同; 见 kvcache._rope_delta_cos_sin)
        self._rotary_emb = resolve_rotary_emb(self.model)
        # transformers 5.x 的 CausalLM 可以只要最后一个位置的 logits(logits_to_keep=1):
        # 本引擎每次前向后只取 out.logits[:, -1, :], 但默认会为全部位置算 logits ——
        # 40K prompt、13 万词表时那是 40960*131072*2B ≈ 10.7 GiB, 与 8B 权重叠加直接 OOM。
        # 只在 forward 签名真的支持该参数时传(包装/自定义模型不支持就保持原行为)。
        # context repair 的 exact_prefill 共用同一探测(见 model_adapters)。
        self._logits_kwargs = resolve_logits_kwargs(self.model)
        # 对外模型名: 优先用 --served-model-name(与 vLLM 语义一致), 否则取路径最后一段
        self.model_name = (self.config.served_model_name
                           or self.config.model.rstrip("/").split("/")[-1]
                           or self.config.model)

        # 打印实际解析出的 attention 实现(transformers 会把 auto/回退解析为具体值)
        resolved_impl = (getattr(self.model.config, "_attn_implementation_internal", None)
                         or self.model.config._attn_implementation or "auto")
        logger.info("模型加载完成: %s (%.2fs), attention 实现: %s",
                    self.model_name, time.time() - t0, resolved_impl)
        if self.rotary_emb is None:
            logger.warning("未找到文本解码器的 RoPE 模块, --graft-rope-rebase 将按"
                           "默认 RoPE 公式重映射(缩放型 RoPE 如 YaRN 会不准确)")
        logger.info("KV cache 每 token 占用(理论): %.2f MiB", self.kv_bytes_per_token / (1024 ** 2))

    @property
    def model_config(self):
        """KV 形状 / 位置 / RoPE 计算使用的 config, 一律是**文本塔** config.

        多模态包装(如 Ministral-3 的 Mistral3ForConditionalGeneration)的顶层
        config 没有 num_hidden_layers / rope_parameters / head_dim, 必须换成
        `config.get_text_config(decoder=True)`; _load_model 会把结果存进
        `_model_config`。只构造引擎对象不加载模型的桩(测试)回退 model.config。
        """
        config = getattr(self, "_model_config", None)
        return config if config is not None else self.model.config

    @property
    def rotary_emb(self):
        """文本解码器的 RoPE 模块(--graft-rope-rebase 用); 桩对象上按需探测一次."""
        if hasattr(self, "_rotary_emb"):        # _load_model 已探测(结果可能是 None)
            return self._rotary_emb
        return resolve_rotary_emb(self.model)   # 只构造引擎对象不加载模型的桩(测试)

    @property
    def logits_kwargs(self) -> dict:
        """前向时只算末尾 logits 的参数(见 _load_model); 桩对象上为空."""
        return getattr(self, "_logits_kwargs", {})

    @property
    def kv_bytes_per_token(self) -> int:
        """每生成 1 个 token, 全部层新增 K+V 的显存字节数.

        公式按模型家族取(见 model_adapters.kv_bytes_per_token): 普通 GQA/MHA 是
        2(K+V) x 层数 x KV头数 x head_dim x 每元素字节数; MLA(如 GLM-4.7-Flash)
        缓存的是压缩潜向量 kv_lora_rank + qk_rope_head_dim, 与 KV 头数无关。
        """
        dtype_size = torch.tensor([], dtype=self.model.dtype).element_size()
        return kv_bytes_per_token(self.model_config, dtype_size)

    def _eos_ids(self) -> set[int]:
        """收集所有需要触发停止的 eos token id.

        顶层 config 不一定有 eos_token_id(多模态包装的 config 就没有, 如
        Mistral3Config), 文本 config / generation config / tokenizer 依次兜底.
        """
        ids: set[int] = set()
        for source in (getattr(self.model_config, "eos_token_id", None),
                       getattr(self.model.config, "eos_token_id", None),
                       getattr(self.model.generation_config, "eos_token_id", None),
                       getattr(self.tokenizer, "eos_token_id", None)):
            if isinstance(source, (list, tuple)):
                ids.update(int(i) for i in source)
            elif source is not None:
                ids.add(int(source))
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
        reuse_prefixes: list[KVPrefix] | None = None,  # 跨请求前缀 KV 复用候选(agent 模式, LCP 安全)
        graft: KVGraft | list[KVGraft] | None = None,  # 位置感知拼接: 插入一个或多个 sub 输出 KV
    ) -> GenerationResult:
        """prefill + decode 的生成循环, 返回完整结果.

        两种模式的对照(这正是"KV cache 省了什么"的实验基础):
        - use_kv_cache=True : decode 每次只喂 1 个 token, 其余全靠 cache
        - use_kv_cache=False: decode 每次把整个前缀重新前向(无缓存)

        跨请求复用有两种模式(见 kvcache.py, 可叠加):
        - reuse_prefixes: LCP 安全模式。对每个候选段与本次 prompt 做 token 级
          最长公共前缀匹配, 只复用真正相同的部分, 不匹配时安全回退全量 prefill;
        - graft         : 位置感知拼接模式(实验)。子 agent 输出的 KV 由 server 端
          定位到其在 prompt 中的位置(KVGraft.position), 引擎先 LCP 复用 main
          历史 KV, 再按位置顺序 prefill 间隙并插入一个或多个子输出 KV。
          插入位置/长度/token 逐位校验, 失败自动回退 LCP 模式(绝不错误拼接).
          注意: 子输出 KV 是在子 agent 自己的上下文里计算的, 插入 main 上下文
          后注意力结果与全量 prefill 存在近似差异(实验用途).
        """
        # 截断 prompt, 保证总长度不超过 max_model_len
        max_prompt = self.config.max_model_len - sampling.max_tokens
        if max_prompt < 1:
            # max_tokens >= max_model_len 时没有空间留给 prompt: 原先会截断成空
            # 序列, 在模型 forward 里报 "cannot reshape tensor of 0 elements" 这种
            # 看不懂的错误。这里直接给出可操作的提示(提高 --max-model-len 或降低
            # 请求的 max_tokens)。
            raise ValueError(
                f"max_tokens({sampling.max_tokens}) 不小于 max_model_len"
                f"({self.config.max_model_len}), 没有空间留给 prompt: "
                f"请提高 --max-model-len 或降低请求的 max_tokens")
        if prompt_ids.shape[1] > max_prompt:
            logger.warning("请求 %s: prompt %d tokens 超过上限 %d, 截断头部保留末尾",
                           request_id, prompt_ids.shape[1], max_prompt)
            prompt_ids = prompt_ids[:, -max_prompt:]
        prompt_ids = prompt_ids.to(self.model.device)
        n_prompt = prompt_ids.shape[1]
        if prompt_ids.is_cuda:
            torch.cuda.synchronize(prompt_ids.device)
        t0 = time.perf_counter()
        exact_len = n_prompt
        repair_stats = {}

        eos_ids = self._eos_ids()
        generated: list[int] = []                    # 已生成的 token id
        all_ids = prompt_ids                         # 完整序列 = prompt + 已生成(采样器用)
        cache = DynamicCache(config=self.model_config)  # KV cache 容器

        # ---- 跨请求前缀 KV 复用 ----
        # KV 依赖完整因果上下文; LCP 只能复用已知精确的前缀.
        # 整段命中(LCP == n_prompt, 即客户端原样重发同一 prompt)时只需 prefill
        # 最后 1 个 token 就能拿到它的 logits, 同样成立.
        reuse_len = 0
        graft_mismatch = False
        graft_plan: tuple[int, list[KVGraft]] | None = None
        prompt_list = prompt_ids[0].tolist()

        def _best_prefix(cap: int = n_prompt) -> tuple[int, DynamicCache | None, KVPrefix | None]:
            best_len, best_cache, best_prefix = 0, None, None
            for prefix in sorted(reuse_prefixes or [],
                                 key=lambda pr: len(pr.tokens), reverse=True):
                m = min(longest_common_prefix(prompt_list, prefix.tokens), cap, prefix.exact_length)
                if m > best_len:
                    best_len, best_cache, best_prefix = m, prefix.cache, prefix
                if best_len == cap:
                    break
            return best_len, best_cache, best_prefix

        grafts: list[KVGraft] = []
        if graft is not None:
            grafts = graft if isinstance(graft, list) else [graft]
            grafts = sorted(grafts, key=lambda g: g.position)

        if use_kv_cache and grafts:
            print("\n[复用子 Agent 的输出]\n")
            n = n_prompt
            first_p = grafts[0].position
            reuse_cap = first_p
            base_len, base_cache, _base_prefix = _best_prefix(reuse_cap)

            valid = base_len <= reuse_cap
            prev_end = -1
            for item in grafts:
                p, L = item.position, len(item.tokens)
                if not (0 < p < n and L > 0 and p + L <= n and prev_end <= p
                        and prompt_list[p:p + L] == item.tokens):
                    valid = False
                    break
                prev_end = p + L
            if not valid:
                graft_mismatch = True
                best_len, best_cache, _best_prefix_obj = _best_prefix(n_prompt)
                logger.warning(
                    "请求 %s: 多段拼接结构校验失败(%d 段), 回退 LCP 复用 %d tok",
                    request_id, len(grafts), best_len)
                reuse_len = min(best_len, n_prompt - 1)
                if reuse_len > 0:
                    cache = slice_cache(best_cache, reuse_len, self.model_config)
            else:
                graft_plan = (base_len, grafts)
                if base_len > 0:
                    cache = slice_cache(base_cache, base_len, self.model_config)
        elif use_kv_cache and reuse_prefixes:
            best_len, best_cache, best_prefix = _best_prefix(n_prompt)
            if best_len > 0:
                # 必须深拷贝切片: transformers 的 DynamicCache.update 原地拼接,
                # 直接把已保存的缓存传给生成循环会污染 store 里的对象
                reuse_len = min(best_len, n_prompt - 1)
                cache = slice_cache(best_cache, reuse_len, self.model_config)
                # 复用的前缀来自哪个段: 子 agent 输出(核心收益)还是 main 历史
                src = "子 agent" if best_prefix.kind == KIND_SUB else \
                    ("main" if best_prefix.kind == KIND_MAIN else "未知来源")
                out_reused = max(0, best_len - best_prefix.output_start)
                if out_reused > 0:
                    # 复用穿透到了上一轮的输出段(prompt KV + 输出 KV 双复用)
                    logger.info(
                        "请求 %s: 复用%s 历史 KV %d tok + 输出 KV %d tok"
                        "(prompt %d tok 的 %.0f%%), 剩余 %d tok prefill",
                        request_id, src, best_prefix.output_start, out_reused,
                        n_prompt, 100.0 * best_len / n_prompt, n_prompt - best_len)
                else:
                    logger.info("请求 %s: 复用%s 历史 KV %d tok(prompt %d tok 的 %.0f%%), "
                                "剩余 %d tok prefill", request_id, src, best_len,
                                n_prompt, 100.0 * best_len / n_prompt,
                                n_prompt - best_len)

        ttft_ms = 0.0

        # 输出开头 <think> 块的 token 数跟踪(状态机):
        #   not_started: 第一个 token 是 <think> 才进入 in_think, 否则直接 done(无 think 块)
        #   in_think   : 累计 token 数, 遇到 </think> 结束
        # 只处理"输出开头"的 think 块(Qwen 系标准行为), 中间/结尾的 think 不处理
        in_think = False
        think_done = self._think_ids is None
        output_think = 0

        def _on_token(token: int) -> None:
            nonlocal in_think, think_done, output_think
            generated.append(token)
            if not think_done:
                if not in_think:
                    if token == self._think_ids[0]:
                        in_think = True
                        output_think = 1
                    else:
                        think_done = True  # 输出不以 <think> 开头
                else:
                    output_think += 1
                    if token == self._think_ids[1]:
                        in_think = False
                        think_done = True
            if on_token is not None:
                on_token(token)

        with torch.inference_mode():
            # ---- prefill: 一次性前向整个 prompt ----
            # 有缓存模式: 模型把每一层的 K/V 存入 cache, 后续 decode 全部复用;
            #   复用前缀时只前向 [reuse_len:], 前缀 K/V 由切片缓存提供(position
            #   id 自动从 cache 长度继续, mask 仍覆盖整个 prompt, 恒为全 1);
            # 无缓存模式: 同样只做一次前向得到首 token, 但 cache 保持为空.
            # 位置感知拼接模式: prefill 分两段 —— 先前向 [base_len, p)(补 role
            # 标记等插入点前的 token), 把子输出 KV 插进 cache, 再前向 [p+L, n);
            # 若子输出正好到 prompt 末尾, 用最后 1 个 token 的前向拿 logits
            # (与 LCP 整段命中的处理一致), 此时 cache 长度 n-1, 前向后补到 n.
            if graft_plan is not None and self.config.repair_mode in ("context", "exact"):
                base_len, plan_grafts = graft_plan
                if self.config.repair_mode == "exact":
                    repaired = exact_prefill(self.model, prompt_ids, cache, base_len)
                else:
                    repaired = context_prefill(self.model, prompt_ids, cache, base_len,
                                               plan_grafts, self.config)
                out, cache = repaired, repaired.cache
                exact_len, repair_stats = repaired.exact_prefix_len, repaired.stats
                reuse_len = base_len
                logger.info("请求 %s: %s repair %s", request_id, self.config.repair_mode, repair_stats)
            elif graft_plan is not None:
                base_len, plan_grafts = graft_plan
                device = self.model.device
                total_graft_tokens = sum(len(g.tokens) for g in plan_grafts)
                first_p = plan_grafts[0].position
                last_end = max(g.position + len(g.tokens) for g in plan_grafts)
                cur = base_len
                skipped = base_len
                grafted_segments = 0
                grafted_tokens = 0
                for item in plan_grafts:
                    p, L = item.position, len(item.tokens)
                    left_trim, right_trim = repair_token_counts(
                        L, self.config.repair_window_begin, self.config.repair_window_end)
                    graft_start = p + left_trim
                    graft_len = L - left_trim - right_trim
                    if graft_len <= 0:
                        continue
                    if graft_start > cur:
                        out = self.model(
                            input_ids=prompt_ids[:, cur:graft_start],
                            attention_mask=torch.ones(1, graft_start, device=device),
                            past_key_values=cache, use_cache=True,
                            **self.logits_kwargs,
                        )
                    graft_cache = slice_cache(
                        item.cache, graft_len + left_trim, self.model_config)
                    graft_cache = slice_cache(
                        tail_cache(graft_cache, left_trim, self.model_config),
                        graft_len, self.model_config)
                    if self.config.graft_rope_rebase:
                        graft_cache = rebase_rope_cache(
                            graft_cache, item.source_position + left_trim,
                            graft_start, self.model_config, rope=self.rotary_emb)
                    exact_len = min(exact_len, graft_start)
                    # 插入子输出 KV: concat 产生新对象, 后续原地拼接不会污染 store
                    cache = concat_cache(cache, graft_cache, self.model_config)
                    cur = graft_start + graft_len
                    skipped += graft_len
                    grafted_segments += 1
                    grafted_tokens += graft_len
                if cur < n_prompt:
                    out = self.model(
                        input_ids=prompt_ids[:, cur:],
                        attention_mask=torch.ones(1, n_prompt, device=device),
                        past_key_values=cache, use_cache=True,
                        **self.logits_kwargs,
                    )
                    reuse_len = skipped
                else:
                    cache = slice_cache(cache, n_prompt - 1, self.model_config)
                    out = self.model(
                        input_ids=prompt_ids[:, n_prompt - 1:],
                        attention_mask=torch.ones(1, n_prompt, device=device),
                        past_key_values=cache, use_cache=True,
                        **self.logits_kwargs,
                    )
                    reuse_len = max(0, skipped - 1)  # 最后 1 个 token 需要前向拿 logits
                rebase_note = " + RoPE rebase" if self.config.graft_rope_rebase else ""
                repair_note = (f", 每段首部重算 {self.config.repair_window_begin:.1%}, "
                               f"尾部重算 {self.config.repair_window_end:.1%}")
                logger.info(
                    "请求 %s: 拼接子 agent 输出 KV %d/%d 段, %d/%d tok(位置 %d..%d)%s%s + "
                    "复用 main 历史 KV %d tok, 剩余 %d tok prefill",
                    request_id, grafted_segments, len(plan_grafts), grafted_tokens,
                    total_graft_tokens, first_p, last_end - 1, rebase_note, repair_note,
                    base_len, n_prompt - reuse_len)
            else:
                out = self.model(
                    input_ids=prompt_ids[:, reuse_len:],
                    attention_mask=torch.ones_like(prompt_ids),
                    past_key_values=cache if use_kv_cache else None,
                    use_cache=use_kv_cache,
                    **self.logits_kwargs,
                )
            token = self._sample(out.logits[:, -1, :], all_ids, sampling)
            ttft_ms = (time.perf_counter() - t0) * 1000
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
                        **self.logits_kwargs,
                    )
                else:
                    # 无缓存对照模式: 把整个前缀重新前向(注意力计算量随长度线性增长)
                    out = self.model(
                        input_ids=all_ids, attention_mask=torch.ones_like(all_ids),
                        use_cache=False,
                        **self.logits_kwargs,
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
        # ---- 缓存与序列对齐 ----
        # 对齐目标: cache 长度 == n_prompt + len(generated), 否则下一次跨请求
        # 复用的位置会错位(见 kvcache.py 的 put 长度校验). 两种偏差来源:
        #   - 短 1: max_tokens 提前结束, 循环是"先前向再采样", 最后一个生成
        #     token 的 KV 从未前向 -> 补一次 decode 前向;
        #   - 长 1: 首 token 即 eos 被清理时, 它的 KV 悬垂在尾部 -> 裁剪.
        if use_kv_cache:
            target = n_prompt + len(generated)
            cur_len = cache.get_seq_length()
            if cur_len < target:
                with torch.inference_mode():
                    self.model(
                        input_ids=torch.tensor([[generated[-1]]], device=self.model.device),
                        attention_mask=torch.ones(1, cur_len + 1, device=self.model.device),
                        past_key_values=cache, use_cache=True,
                        **self.logits_kwargs,
                    )
            elif cur_len > target:
                cache = slice_cache(cache, target, self.model_config)

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
            reused_prompt_tokens=reuse_len,
            kv_cache=cache if use_kv_cache else None,
            output_think_tokens=output_think,
            kv_graft_mismatch=graft_mismatch,
            exact_prefix_len=(n_prompt + len(generated) if exact_len == n_prompt else exact_len),
            repair_stats=repair_stats,
        )

        # 全局统计 + 每请求日志(风格类似 vLLM 的日志行)
        self._stats["completed"] += 1
        self._stats["generated_tokens"] += result.completion_tokens
        self._stats["prefill_tokens"] += n_prompt
        if not self.config.disable_log_stats:
            reuse_note = f", 复用前缀 {reuse_len} tok" if reuse_len else ""
            logger.info(
                "请求 %s: prompt %d tok, 生成 %d tok, TTFT %.0fms, %.1f tok/s, KV cache %.2f MiB, %s%s",
                request_id, n_prompt, result.completion_tokens, ttft_ms,
                result.decode_tokens_per_sec, kv_bytes / (1024 ** 2), finish_reason, reuse_note,
            )
        return result

    # ------------------------------------------------------------------
    # 异步(HTTP)入口
    # ------------------------------------------------------------------
    async def generate(self, request_id: str | None, prompt_ids: torch.Tensor,
                       sampling: SamplingParams, stream: bool = False,
                       skip_special_tokens: bool = True,
                       reuse_prefixes: list[KVPrefix] | None = None,
                       graft: KVGraft | list[KVGraft] | None = None):
        """异步生成.

        返回:
          - stream=False: await 得到 GenerationResult
          - stream=True : 返回 asyncio.Queue, 队列元素为 ("text", 文本)
                          或 ("done", GenerationResult) 哨兵
        reuse_prefixes: LCP 安全模式的跨请求前缀 KV 复用候选(见 kvcache.py);
        graft:          位置感知拼接的子输出 KV(见 kvcache.py, 与 reuse_prefixes 可叠加).
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
                    skip_special_tokens, reuse_prefixes, graft,
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
                    skip_special_tokens, reuse_prefixes, graft,
                ),
            )
        )
        return result
