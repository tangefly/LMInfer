"""Agent 会话间的跨请求 KV 前缀复用(本项目自研的"简单 KV Cache 系统").

背景
----
agent 模式下, 主 agent 调起子 agent, 子 agent 的输出会作为新消息回到主 agent,
主 agent 的下一轮请求在 prompt 里重新包含了这段历史 —— 这些 token 的 KV 在
上一次请求里已经算过, 但默认实现会整体重新 prefill, 白白重复计算.

做法
----
1. 每个 agent 会话按请求来源保存**完整序列**(prompt + 生成输出)及其 KV cache,
   要求 KV 与 token 序列逐位对齐、位置从 0 开始。main 段只保留最新一条;
   sub 段按 trace_key 保留“最新 main 之后”的每次 sub invocation。一次 sub
   invocation 内部可能有多轮模型调用, 但只覆盖保存为最新一整段 KV；多个并行/
   连续 sub invocation 则各保留一段, 直到下一次 main 完成后清空。这样
   `main -> {sub, sub, sub} -> main` 的最终 main 能一次性定位并复用多个子
   agent 输出 KV;
2. 新请求到达时由 server 调用 SessionKVStore.propose() 给出候选段(触发条件已放宽:
   会话内任意请求, main 请求给最新 main 段 + 最新 main 后的 sub 段, sub 请求给
   最近 sub 段);
3. 引擎对候选段与新 prompt 做 **token 级最长公共前缀(LCP)匹配**, 只复用
   真正相同的部分, 其余继续 prefill —— 这是正确性的根本保证:

   KV 是 (token, 位置) 的确定性函数。只要复用的 token 与位置和全量 prefill
   逐位一致, 注意力结果在数学上就完全相同(数值上存在 bf16 内核级舍入差异,
   与切换 attention 实现同级)。反之, 任何渲染不一致(如 Qwen3 模板对末尾
   assistant 消息插入 <think> 块)都会让 LCP 提前停止, 安全回退到全量 prefill,
   绝不会产生错误结果。

4. 位置感知拼接(build_grafts, 实验): 子 agent 的输出作为 tool 结果回填进 main
   的下一轮 prompt 时, 其 token 由 chat template 重新渲染(正文前后带 role 标记,
   如 Qwen3 的 <tool_response> 包裹), 不构成任何已保存段的公共前缀, LCP 模式
   无法复用。拼接模式利用 <tool_response> 包裹标记(特殊 token, id 与上下文
   无关)在渲染后的 prompt 中**锚定正文所在窗口**, 再把窗口与子请求保存的输出
   序列逐位对齐(正文可能写在 think 块内/外, 见 SessionKVStore.build_grafts),
   把子 agent 请求时算好的输出 KV 直接插入 main 的 KV cache 对应位置 ——
   窗口外的包裹标记由引擎照常 prefill, 跳过的是正文这段的重复 prefill。
   注意: 该 KV 在子 agent 自己的上下文里计算, 插入 main 上下文后注意力结果
   与全量 prefill 存在**近似差异**(实验用途, 见 KVGraft/build_grafts);
   定位失败时安全回退 LCP 模式。

5. 复用段必须**深拷贝**(slice_cache/tail_cache)后交给生成循环: transformers 的
   DynamicCache.update 是原地拼接, 直接传会让后续请求污染已保存的缓存.

并发说明: propose / build_grafts / put 都发生在 asyncio 事件循环上(与
AgentSessionRegistry 相同), 无需加锁; 生成循环线程只持有深拷贝的 cache,
不会触碰 store 里的对象.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
from transformers import DynamicCache

logger = logging.getLogger("lminfer")

# 每个会话按请求来源保存的段类别
KIND_MAIN = "main"   # 主 agent 请求的完整序列
KIND_SUB = "sub"     # 子 agent 请求的完整序列(含其输出, 即"子 agent 的输出 KV")

# Qwen3 chat template 渲染 tool 消息用的包裹标记(特殊 token, id 与上下文无关,
# 拼接模式据此在渲染后的 prompt token 序列中直接定位正文窗口)
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"

# 拼接模式定位参数(见 SessionKVStore.build_grafts):
GRAFT_HEAD_SLACK = 2    # 窗口内对齐时允许的首部总漂移 token 数(边界 token 与
                        # 正文首个 token 合并 / 客户端剥离首部空白)
GRAFT_WINDOW_SLACK = 8  # 包裹标记间窗口长度超出正文长度的容忍上限; 超出说明
                        # 窗口里混入了其他消息(如多段 tool 消息), 锚点不可靠
GRAFT_MIN_MATCH = 4     # 拼接的最短匹配 token 数(更短可能是误命中, 保守放弃)


@dataclass
class KVPrefix:
    """一段与 token 序列逐位对齐的 KV cache(默认覆盖位置 0 .. len(tokens)-1).

    对齐是复用的前提: cache 的第 i 个位置必须是 tokens[i] 的前向结果,
    且位置 id 从 0 开始. 由 put() 的长度校验与引擎的 LCP 匹配共同保证.

    拼接模式(位置感知插入子 agent 输出 KV)下, 保存的是完整序列(prompt + 输出):
    - output_start: 输出 KV 的起始位置(= prompt 长度), 拼接时切出输出部分;
    - think_len    : 输出开头 <think> 块的 token 数 —— thinking 通常不作为
      下一轮对话的 prompt(客户端回填时会剥离), 拼接 KV 时也要剔除,
      否则拼接长度与正常 prompt 结构不一致.
    """

    tokens: list[int]        # 该段 KV 覆盖的 token id 序列(prompt + 输出)
    cache: DynamicCache      # 与 tokens 严格对齐的 KV cache
    output_start: int = 0    # 输出 KV 的起始位置(= prompt 长度)
    think_len: int = 0       # 输出开头 <think> 块的 token 数(拼接时剔除)
    kind: str = ""           # 段来源(KIND_MAIN / KIND_SUB), 供引擎日志区分复用的
                             # 是子 agent 输出 KV 还是 main 历史 KV
    trace_key: tuple[str, ...] | None = None  # sub invocation 身份; 同一次 sub 多轮推理覆盖保存


@dataclass
class KVGraft:
    """位置感知拼接(实验): 一段子 agent 输出 KV, 插入到主 agent prompt 的指定位置.

    chat template 渲染 tool 消息时, 子 agent 的输出正文前后会带 role 标记/
    特殊标签(如 Qwen3 的 `<tool_response>` 包裹), 因此它在新 prompt 里出现在
    [main 历史 + 标记] 之后 —— build_graft 在渲染后的 prompt token 序列中
    定位正文的起始位置, 引擎据此把这段 KV 精确插到该位置.

    - position: 子输出正文 tokens 在 prompt 中的起始位置(0-based);
    - tokens  : 与 cache 逐位对齐的子输出连续段(取自子请求输出序列, 正文写在
      think 块内时该段也落在 think 块内; 首尾边界漂移的 token 已截断,
      可能短于子输出全文);
    - cache   : 与 tokens 对齐的 KV(子 agent 请求输出段的深拷贝).
    """

    position: int            # 子输出正文 tokens 在 main prompt 中的起始位置
    tokens: list[int]        # 与 cache 逐位对齐的子输出连续段
    cache: DynamicCache      # 与 tokens 对齐的子输出 KV cache
    source_position: int = 0  # 该 KV 在子 agent 序列中的原始起始位置(RoPE rebase 用)


def slice_cache(cache: DynamicCache, length: int,
                config) -> DynamicCache:
    """深拷贝 cache 的前 `length` 个位置, 返回新的 DynamicCache.

    用于两个场景:
    - 复用: 把已保存的整段 KV 裁到 LCP 匹配长度, 拷贝后交给生成循环
      (原地拼接的 update 会污染原对象, 必须拷贝);
    - 收尾: 裁掉 eos 等提前结束时悬垂在缓存尾部的多余 KV, 保证
      cache 长度 == 序列长度(否则下一次复用的位置会错位).
    """
    for layer in cache.layers:
        if not layer.is_initialized:
            # 生成结束后所有层都应已初始化(batch=1, 无 padding);
            # 若出现未初始化层, 直接报错而非静默错位
            raise ValueError("slice_cache: 存在未初始化的缓存层, 无法切片")
    layers = [
        (layer.keys[:, :, :length].clone(), layer.values[:, :, :length].clone())
        for layer in cache.layers
    ]
    return DynamicCache(ddp_cache_data=layers, config=config)


def tail_cache(cache: DynamicCache, start: int, config) -> DynamicCache:
    """深拷贝 cache 从位置 `start` 起的 KV(即 [start:] 部分), 返回新 DynamicCache.

    用于拼接模式: 子 agent 段的输出 KV 在完整 cache 的尾部, 从 prompt 长度起.
    """
    for layer in cache.layers:
        if not layer.is_initialized:
            raise ValueError("tail_cache: 存在未初始化的缓存层, 无法切片")
    layers = [
        (layer.keys[:, :, start:].clone(), layer.values[:, :, start:].clone())
        for layer in cache.layers
    ]
    return DynamicCache(ddp_cache_data=layers, config=config)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen/Llama-style RoPE half rotation."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_delta_cos_sin(length: int, delta: int, head_dim: int, config,
                        device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cos/sin for RoPE position delta.

    Existing K cache has already been rotated at source positions. Because RoPE
    rotations compose, rotating by (target_position - source_position) maps it
    to the target position for default Qwen/Llama RoPE.
    """
    rope_params = getattr(config, "rope_parameters", None) or {}
    theta = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.full((length,), float(delta), device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def rebase_rope_cache(cache: DynamicCache, source_start: int, target_start: int,
                      config) -> DynamicCache:
    """Deep-copy `cache` and rebase K RoPE positions from source to target.

    Only K carries RoPE; V is cloned unchanged. This corrects position mismatch
    for grafted sub-agent output KV, but it does not fix the different-context
    hidden-state gap. The implementation targets default Qwen/Llama-style RoPE.
    """
    delta = target_start - source_start
    if delta == 0:
        return slice_cache(cache, cache.get_seq_length(), config)
    for layer in cache.layers:
        if not layer.is_initialized:
            raise ValueError("rebase_rope_cache: 存在未初始化的缓存层, 无法重映射")
    layers = []
    for layer in cache.layers:
        keys = layer.keys.clone()
        values = layer.values.clone()
        length = keys.shape[-2]
        head_dim = keys.shape[-1]
        cos, sin = _rope_delta_cos_sin(length, delta, head_dim, config,
                                       keys.device, keys.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        keys = (keys * cos) + (_rotate_half(keys) * sin)
        layers.append((keys, values))
    return DynamicCache(ddp_cache_data=layers, config=config)


def concat_cache(a: DynamicCache, b: DynamicCache, config) -> DynamicCache:
    """把两个 KV cache 沿序列维拼接(先 a 后 b), 返回新的 DynamicCache.

    拼接模式(强制 KV 复用)核心: 子 agent 输出的 KV 直接接在主 agent
    历史 KV 之后, 组成下一轮请求的前缀 KV —— 不校验 token 内容,
    位置语义由调用方负责(实验用途, 结果可能不正确).
    """
    if len(a.layers) != len(b.layers):
        raise ValueError(f"concat_cache: 层数不一致 {len(a.layers)} vs {len(b.layers)}")
    layers = []
    for la, lb in zip(a.layers, b.layers):
        if not la.is_initialized or not lb.is_initialized:
            raise ValueError("concat_cache: 存在未初始化的缓存层, 无法拼接")
        layers.append((
            torch.cat([la.keys, lb.keys], dim=-2),
            torch.cat([la.values, lb.values], dim=-2),
        ))
    return DynamicCache(ddp_cache_data=layers, config=config)


def longest_common_prefix(a: list[int], b: list[int]) -> int:
    """两个 token 序列的最长公共前缀长度(逐元素比较)."""
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return i


class SessionKVStore:
    """按 agent 会话保存请求的完整序列 KV，供跨请求前缀复用.

    保留策略:
    - **main 段保留最新一条**: main 请求每轮覆盖, 用于后续 main 的 LCP 基础复用;
    - **sub 段按 trace_key 保留最新 main 之后的每次 sub invocation**: 同一个
      sub 内部多轮模型调用覆盖为最新一整段 KV, 不拆成多个候选; 多个 sub
      invocation 各保留一段。新的 main 保存成功后清空 sub 列表, 因为这些历史
      已被 main 段覆盖.
    """

    def __init__(self, config=None, tokenizer=None, idle_ttl: float = 3600.0) -> None:
        # session_id -> {"main": KVPrefix | None, "subs": list[KVPrefix]}
        self._segments: dict[str, dict[str, Any]] = {}
        self._last_seen: dict[str, float] = {}  # session_id -> 最近一次 put 时间(闲置清理用)
        self._idle_ttl = idle_ttl
        self._stats = {"reuse_attempts": 0, "reuse_hits": 0, "reuse_tokens": 0,
                       "graft_mismatches": 0}
        # 拼接 cache 时需要模型 config 构造 DynamicCache(server 传入 engine.model.config)
        self._config = config
        # 拼接模式的结构锚点: Qwen3 模板渲染 tool 消息的包裹标记是特殊 token,
        # id 与上下文无关, build_graft 据此在 prompt 中直接定位正文窗口(见
        # TOOL_RESPONSE_OPEN/CLOSE)。tokenizer 缺失或标记不存在(含被当成未知
        # token 的情况)时置 None, 拼接模式自动不可用, 回退 LCP 复用。
        self._resp_marker_ids: tuple[int | None, int | None] = (None, None)
        if tokenizer is not None:
            open_id = tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_OPEN)
            close_id = tokenizer.convert_tokens_to_ids(TOOL_RESPONSE_CLOSE)
            if (isinstance(open_id, int) and isinstance(close_id, int)
                    and tokenizer.convert_ids_to_tokens(open_id) == TOOL_RESPONSE_OPEN
                    and tokenizer.convert_ids_to_tokens(close_id) == TOOL_RESPONSE_CLOSE):
                self._resp_marker_ids = (open_id, close_id)
            else:
                logger.warning("tokenizer 没有 %s/%s 特殊 token, 拼接模式"
                               "(--reuse-agent-kv-append)不可用, 回退 LCP 复用",
                               TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE)

    def _prune_idle(self, now: float) -> None:
        """清理闲置超过 idle_ttl 的会话段(会话注册表本身无 TTL, 这里兜底防显存泄漏).

        idle_ttl <= 0 表示不清理(段保留到进程结束).
        """
        if self._idle_ttl <= 0 or not self._last_seen:
            return
        stale = [sid for sid, t in self._last_seen.items() if now - t > self._idle_ttl]
        for sid in stale:
            del self._segments[sid]
            del self._last_seen[sid]
            
    def release(self, session_id: str) -> None:
        """根据指定的 session_id 清理显存"""
        if session_id in self._segments:
            del self._segments[session_id]
        if session_id in self._last_seen:
            del self._last_seen[session_id]
        
        logger.info(f"释放 {session_id} 的显存")

    def clear_subs(self, session_id: str) -> None:
        """释放指定会话中最新 main 之后累计的 sub KV 段。

        final main 已经把这些 tool response 纳入自己的 prompt/cache 后, sub 段
        不再需要单独保留; 清掉引用即可让 PyTorch 回收/复用显存缓存。
        """
        segs = self._segments.get(session_id)
        if not segs:
            return
        subs = segs.get("subs")
        if not isinstance(subs, list) or not subs:
            return
        count = len(subs)
        tokens = sum(len(seg.tokens) for seg in subs)
        segs["subs"] = []
        logger.info("会话 %s: 释放已被 main 消费的 sub KV %d 段/%d tok",
                    session_id, count, tokens)

    def put(self, session_id: str, kind: str, seq_tokens: list[int],
            cache: DynamicCache, prompt_len: int = 0,
            think_len: int = 0,
            trace: list[str] | None = None) -> bool:
        """保存一次请求的完整序列 KV; 返回是否保存成功.

        seq_tokens 必须与 cache 长度一致(prompt + 输出, 位置从 0 开始),
        否则说明缓存与序列未对齐(理论上不会发生, 防御性检查), 丢弃并告警.

        保留策略: main 段替换为最新一条; sub 段按 trace_key 代表的一次
        sub agent invocation 覆盖保存。也就是说同一个 sub 内部多次模型调用
        只保留最终/最新一整段 KV, 不拆成多个候选段。
        每次 put 顺带按 idle_ttl 清扫闲置会话段.

        prompt_len / think_len: 拼接模式用 —— 记录输出 KV 的起始位置与
        输出开头 <think> 块的 token 数, 拼接时据此切出/剔除对应 KV.
        """
        if len(seq_tokens) != cache.get_seq_length():
            logger.warning(
                "会话 %s: KV cache 长度 %d 与序列长度 %d 不一致, 放弃保存"
                "(该段无法用于后续复用)",
                session_id, cache.get_seq_length(), len(seq_tokens),
            )
            return False
        now = time.time()
        self._prune_idle(now)
        self._last_seen[session_id] = now
        segs = self._segments.setdefault(session_id, {"main": None, "subs": []})
        trace_key = tuple(trace) if trace else None
        prefix = KVPrefix(seq_tokens, cache, output_start=prompt_len,
                          think_len=think_len, kind=kind, trace_key=trace_key)
        if kind == KIND_MAIN:
            segs["main"] = prefix
            self.clear_subs(session_id)
        else:
            subs = segs.setdefault("subs", [])
            assert isinstance(subs, list)
            replaced = False
            if trace_key is not None:
                for i, seg in enumerate(subs):
                    if isinstance(seg, KVPrefix) and seg.trace_key == trace_key:
                        subs[i] = prefix
                        replaced = True
                        logger.info("会话 %s: 更新同一 sub 调用 trace %s 的 KV 段 "
                                    "%d -> %d tok", session_id, list(trace_key),
                                    len(seg.tokens), len(seq_tokens))
                        break
            if not replaced:
                subs.append(prefix)
        logger.info("会话 %s: 保存 %s 段 KV %d tok(输出 %d tok, think %d tok) "
                    "供跨请求复用", session_id, kind, len(seq_tokens),
                    len(seq_tokens) - prompt_len, think_len)
        return True

    def _match_graft_window(
        self,
        session_id: str,
        sub_seg: KVPrefix,
        window: list[int],
        body_start: int,
    ) -> KVGraft | None:
        """把一个 tool_response 窗口与一个 sub 输出对齐。

        一个 sub agent 的最终输出 KV 作为一个整体候选处理: 每个窗口最多
        产出一个连续 graft 片段。边界上不能匹配的 token 由引擎正常 prefill;
        不把同一个 sub 输出拆成多个 KV 片段。
        """
        sub_out = sub_seg.tokens[sub_seg.output_start:]
        if not sub_out:
            logger.info("会话 %s: 子 agent 段没有输出, 无法拼接", session_id)
            return None
        if not window:
            logger.info("会话 %s: tool response 窗口为空, 放弃该窗口拼接", session_id)
            return None

        # 在 window/sub_out 中找一个最长公共连续片段。未匹配 token 通常是
        # tool_response 包裹、换行、边界 BPE 合并、thinking 剥离或 eos 差异,
        # 留给 main 上下文正常 prefill。
        index: dict[int, list[int]] = {}
        for j, tok in enumerate(sub_out):
            index.setdefault(tok, []).append(j)
        best_k, best_i, best_j = 0, -1, -1
        for i, tok in enumerate(window):
            for j in index.get(tok, ()):
                if len(window) - i <= best_k or len(sub_out) - j <= best_k:
                    continue
                k = 1
                while (i + k < len(window) and j + k < len(sub_out)
                       and window[i + k] == sub_out[j + k]):
                    k += 1
                if k > best_k:
                    best_k, best_i, best_j = k, i, j
        if best_k < GRAFT_MIN_MATCH:
            logger.info("会话 %s: 未在窗口中定位到子 agent 输出正文, 放弃该窗口拼接",
                        session_id)
            return None

        pos = body_start + best_i
        tokens = sub_out[best_j:best_j + best_k]
        cache = slice_cache(
            tail_cache(sub_seg.cache, sub_seg.output_start + best_j, self._config),
            best_k, self._config)
        logger.info("会话 %s: 定位子 agent 输出 KV 1 段/%d tok(窗口 %d tok, "
                    "输出 %d tok, think_len %d), 准备插入",
                    session_id, best_k, len(window), len(sub_out), sub_seg.think_len)
        return KVGraft(position=pos, tokens=tokens, cache=cache,
                       source_position=sub_seg.output_start + best_j)

    def build_grafts(self, session_id: str, trace: list[str],
                     prompt_tokens: list[int]) -> list[KVGraft]:
        """位置感知拼接(实验): 定位多个子 agent 输出正文, 供引擎插入其 KV.

        chat template 渲染 tool 消息时, 客户端回填的子输出正文前后带包裹标记
        (如 Qwen3 的 <tool_response>), 这层标记是 main prompt 相对子请求多出来
        的几个 token。窗口内不能与 sub 最终输出逐位匹配的边界 token 由引擎
        正常 prefill; 匹配到的最长连续正文片段则插入子请求时算好的 KV。每个
        tool response 会向后寻找能匹配的 sub 输出; 同一次 sub invocation 的中间
        KV 在保存时已被最新段覆盖。一个 sub 输出最多生成一个 graft 片段。

        返回按 prompt 位置升序排列的 KVGraft 列表; 无 sub 段或定位失败时返回空列表.
        """
        if len(trace) < 2 or trace[-1] != KIND_MAIN or trace[-2] == KIND_MAIN:
            return []
        open_id, close_id = self._resp_marker_ids
        if open_id is None or close_id is None:
            return []  # tokenizer 无包裹标记, 拼接模式不可用(见 __init__)
        segs = self._segments.get(session_id)
        if not segs:
            return []
        subs = segs.get("subs")
        if not isinstance(subs, list) or not subs:
            return []
        self._stats["reuse_attempts"] += 1

        # 从最新 main 与当前 prompt 的 LCP 之后扫描。不能直接用 len(main_seg.tokens):
        # 第一次 main 原始生成的 tool-call 文本与 final prompt 中结构化 tool_calls
        # 的模板渲染可能不等长, 直接按长度跳会越过前面的 tool_response。
        main_seg = segs.get(KIND_MAIN)
        search_start = (longest_common_prefix(prompt_tokens, main_seg.tokens)
                        if isinstance(main_seg, KVPrefix) else 0)
        windows: list[tuple[int, int, list[int]]] = []
        n = len(prompt_tokens)
        i = search_start
        while i < n:
            if prompt_tokens[i] != open_id:
                i += 1
                continue
            close_pos = -1
            for j in range(i + 1, n):
                if prompt_tokens[j] == close_id:
                    close_pos = j
                    break
            if close_pos < 0:
                break
            windows.append((i + 1, close_pos, prompt_tokens[i + 1:close_pos]))
            i = close_pos + 1
        logger.info("会话 %s: build_grafts trace %s, prompt %d tok, main_lcp %d tok, "
                    "tool_response 窗口 %d 个, 候选 sub 段 %d 个",
                    session_id, trace, n, search_start, len(windows), len(subs))
        if not windows:
            logger.info("会话 %s: prompt 中未找到 %s/%s 包裹标记, "
                        "放弃拼接(回退 LCP)", session_id,
                        TOOL_RESPONSE_OPEN, TOOL_RESPONSE_CLOSE)
            return []

        grafts: list[KVGraft] = []
        matched_windows = 0
        sub_i = 0
        for win_idx, (body_start, _close_pos, window) in enumerate(windows, start=1):
            best_match: KVGraft | None = None
            best_i = -1
            best_tokens = 0
            for match_i in range(sub_i, len(subs)):
                candidate = self._match_graft_window(
                    session_id, subs[match_i], window, body_start)
                candidate_tokens = len(candidate.tokens) if candidate is not None else 0
                if candidate_tokens > best_tokens:
                    best_match = candidate
                    best_i = match_i
                    best_tokens = candidate_tokens
            if best_match is not None:
                sub_i = best_i + 1
                matched_windows += 1
                grafts.append(best_match)
                logger.info("会话 %s: 第 %d/%d 个 tool response 选择第 %d/%d 个 sub "
                            "候选, 匹配 %d tok", session_id, win_idx, len(windows),
                            best_i + 1, len(subs), best_tokens)
            else:
                logger.info("会话 %s: 第 %d/%d 个 tool response 未匹配到后续 sub "
                            "最终正文 KV, 跳过该窗口", session_id, win_idx, len(windows))
        if grafts:
            logger.info("会话 %s: 共定位 %d/%d 个 tool response, 生成 %d 个 KV 片段"
                        "(候选 sub 段 %d 个), 准备多段拼接",
                        session_id, matched_windows, len(windows), len(grafts), len(subs))
        return grafts

    def build_graft(self, session_id: str, trace: list[str],
                    prompt_tokens: list[int]) -> KVGraft | None:
        """兼容旧调用: 返回最后一段可拼接的子 agent 输出 KV."""
        grafts = self.build_grafts(session_id, trace, prompt_tokens)
        return grafts[-1] if grafts else None

    def propose(self, session_id: str, trace: list[str]) -> list[KVPrefix]:
        """给出可尝试复用的候选段(空列表 = 本次不尝试).

        触发条件已放宽(不再限定"main 在子 agent 返回后继续"): 只要会话里有
        已保存的段就给出候选 —— main 请求给最新 main 段 + 最新 main 后的 sub 段,
        sub 请求给最近 sub 段(同一 sub 的多轮续接)。具体能否复用、复用多少
        由引擎的 LCP 匹配决定(前缀不一致自动回退全量 prefill, 零风险).
        """
        segs = self._segments.get(session_id)
        if not segs:
            return []
        self._stats["reuse_attempts"] += 1
        if trace and trace[-1] == KIND_MAIN:
            candidates: list[KVPrefix] = []
            if isinstance(segs.get("main"), KVPrefix):
                candidates.append(segs["main"])  # 最新 main 段(含其输出 KV)
            subs = segs.get("subs")
            if isinstance(subs, list):
                candidates.extend(subs)          # 最新 main 后的所有 sub 段
            return candidates
        subs = segs.get("subs")
        if isinstance(subs, list) and subs:
            return [subs[-1]]  # sub 请求: 最近 sub 段(续接时可复用)
        return []

    def note_hit(self, reused_tokens: int) -> None:
        """记录一次成功复用(由 server 在 reused_prompt_tokens > 0 时调用)."""
        self._stats["reuse_hits"] += 1
        self._stats["reuse_tokens"] += reused_tokens

    def note_graft_mismatch(self) -> None:
        """记录一次拼接结构校验失败(插入位置/长度/token 与 prompt 不一致)."""
        self._stats["graft_mismatches"] += 1

    @property
    def stats(self) -> dict:
        return dict(self._stats)
