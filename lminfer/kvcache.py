"""Agent 会话间的跨请求 KV 前缀复用(本项目自研的"简单 KV Cache 系统").

背景
----
agent 模式下, 主 agent 调起子 agent, 子 agent 的输出会作为新消息回到主 agent,
主 agent 的下一轮请求在 prompt 里重新包含了这段历史 —— 这些 token 的 KV 在
上一次请求里已经算过, 但默认实现会整体重新 prefill, 白白重复计算.

做法
----
1. 每个 agent 会话按请求来源保存**完整序列**(prompt + 生成输出)及其 KV cache,
   要求 KV 与 token 序列逐位对齐、位置从 0 开始 —— **main/sub 段都只保留最新
   一条**(main 每轮覆盖): main 下一轮 prompt 是上一轮完整序列的逐字延续(think
   由 enable_thinking + chat template 处理, 仅 sub 回填时剥离), 保存段构成嵌套
   前缀链, 最新段与任意新 prompt 的 LCP 恒 ≥ 旧段, 引擎取最长 LCP 时旧段永不
   胜出, 无需多段(截断产生的尾锚定段在 server 保存时丢弃);
2. 新请求到达时由 server 调用 SessionKVStore.propose() 给出候选段(触发条件已放宽:
   会话内任意请求, main 请求给最新 main 段 + 最新 sub 段, sub 请求给最新 sub 段);
3. 引擎对候选段与新 prompt 做 **token 级最长公共前缀(LCP)匹配**, 只复用
   真正相同的部分, 其余继续 prefill —— 这是正确性的根本保证:

   KV 是 (token, 位置) 的确定性函数。只要复用的 token 与位置和全量 prefill
   逐位一致, 注意力结果在数学上就完全相同(数值上存在 bf16 内核级舍入差异,
   与切换 attention 实现同级)。反之, 任何渲染不一致(如 Qwen3 模板对末尾
   assistant 消息插入 <think> 块)都会让 LCP 提前停止, 安全回退到全量 prefill,
   绝不会产生错误结果。

4. 位置感知拼接(build_graft, 实验): 子 agent 的输出作为 tool 结果回填进 main
   的下一轮 prompt 时, 其 token 由 chat template 重新渲染(正文前后带 role 标记,
   如 Qwen3 的 <tool_response> 包裹), 不构成任何已保存段的公共前缀, LCP 模式
   无法复用。拼接模式在渲染后的 prompt 中**定位子输出正文的位置**(build_graft),
   把子 agent 请求时算好的输出 KV 直接插入 main 的 KV cache 对应位置, 跳过这段
   的重复 prefill。注意: 该 KV 在子 agent 自己的上下文里计算, 插入 main 上下文
   后注意力结果与全量 prefill 存在**近似差异**(实验用途, 见 KVGraft/build_graft);
   定位失败时安全回退 LCP 模式。

5. 复用段必须**深拷贝**(slice_cache/tail_cache)后交给生成循环: transformers 的
   DynamicCache.update 是原地拼接, 直接传会让后续请求污染已保存的缓存.

并发说明: propose / build_graft / put 都发生在 asyncio 事件循环上(与
AgentSessionRegistry 相同), 无需加锁; 生成循环线程只持有深拷贝的 cache,
不会触碰 store 里的对象.
"""

import logging
import time
from dataclasses import dataclass

import torch
from transformers import DynamicCache

logger = logging.getLogger("lminfer")

# 每个会话按请求来源保存的段类别
KIND_MAIN = "main"   # 主 agent 请求的完整序列
KIND_SUB = "sub"     # 子 agent 请求的完整序列(含其输出, 即"子 agent 的输出 KV")


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


@dataclass
class KVGraft:
    """位置感知拼接(实验): 一段子 agent 输出 KV, 插入到主 agent prompt 的指定位置.

    chat template 渲染 tool 消息时, 子 agent 的输出正文前后会带 role 标记/
    特殊标签(如 Qwen3 的 `<tool_response>` 包裹), 因此它在新 prompt 里出现在
    [main 历史 + 标记] 之后 —— build_graft 在渲染后的 prompt token 序列中
    定位正文的起始位置, 引擎据此把这段 KV 精确插到该位置.

    - position: 子输出正文 tokens 在 prompt 中的起始位置(0-based);
    - tokens  : 与 cache 逐位对齐的子输出正文 token 前缀(think 与首尾边界
      漂移的 token 已剔除/截断, 可能短于子输出全文);
    - cache   : 与 tokens 对齐的 KV(子 agent 请求输出段的深拷贝).
    """

    position: int
    tokens: list[int]
    cache: DynamicCache

    tokens: list[int]        # 该段 KV 覆盖的 token id 序列(prompt + 输出)
    cache: DynamicCache      # 与 tokens 严格对齐的 KV cache
    output_start: int = 0    # 输出 KV 的起始位置(= prompt 长度)
    think_len: int = 0       # 输出开头 <think> 块的 token 数(拼接时剔除)
    kind: str = ""           # 段来源(KIND_MAIN / KIND_SUB), 供引擎日志区分复用的
                             # 是子 agent 输出 KV 还是 main 历史 KV


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
    - **main/sub 段都只保留最新一条**(main 请求每轮覆盖): main 下一轮请求的
      prompt 是上一轮完整序列(模板渲染逐字一致)的延续, 保存段构成嵌套前缀链 ——
      最新段与任意新 prompt 的 LCP 恒 ≥ 旧段, 引擎对候选取最长 LCP 时旧段
      永远不可能胜出, 只留最新一条即可; 被截断的请求不保存(尾锚定段无法对齐,
      见 server._agent_kv_finish);
    - **sub 段保留最新一条**: 只留最近一次 sub 请求的完整序列(含其输出),
      即"保存好最新的子 Agent 的输出 KV", 供子 agent 多轮续接与拼接模式使用.
    """

    def __init__(self, config=None, idle_ttl: float = 3600.0) -> None:
        # session_id -> {"main": KVPrefix | None, "sub": KVPrefix | None}
        self._segments: dict[str, dict[str, KVPrefix | None]] = {}
        self._last_seen: dict[str, float] = {}  # session_id -> 最近一次 put 时间(闲置清理用)
        self._idle_ttl = idle_ttl
        self._stats = {"reuse_attempts": 0, "reuse_hits": 0, "reuse_tokens": 0,
                       "graft_mismatches": 0}
        # 拼接 cache 时需要模型 config 构造 DynamicCache(server 传入 engine.model.config)
        self._config = config

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

    def put(self, session_id: str, kind: str, seq_tokens: list[int],
            cache: DynamicCache, prompt_len: int = 0,
            think_len: int = 0) -> bool:
        """保存一次请求的完整序列 KV; 返回是否保存成功.

        seq_tokens 必须与 cache 长度一致(prompt + 输出, 位置从 0 开始),
        否则说明缓存与序列未对齐(理论上不会发生, 防御性检查), 丢弃并告警.

        保留策略: main 段与 sub 段都替换为最新一条(每轮覆盖).
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
        segs = self._segments.setdefault(session_id, {"main": None, "sub": None})
        prefix = KVPrefix(seq_tokens, cache, output_start=prompt_len,
                          think_len=think_len, kind=kind)
        if kind == KIND_MAIN:
            segs["main"] = prefix  # main 段保留最新一条(嵌套前缀链下最新段恒为最长候选)
        else:
            segs["sub"] = prefix   # sub 段只留最新一条
        logger.info("会话 %s: 保存 %s 段 KV %d tok(输出 %d tok, think %d tok) "
                    "供跨请求复用", session_id, kind, len(seq_tokens),
                    len(seq_tokens) - prompt_len, think_len)
        return True

    def build_graft(self, session_id: str, trace: list[str],
                    prompt_tokens: list[int]) -> KVGraft | None:
        """位置感知拼接(实验): 定位子 agent 输出正文在 prompt 中的位置, 供引擎插入其 KV.

        与旧版"裸拼接"(已移除)的区别: 不再假设新请求的 prompt 是
        [main 历史][子输出正文][新内容] 的逐段拼接 —— chat template 渲染 tool
        消息时子输出前后会带 role 标记/特殊标签(如 Qwen3 的 <tool_response>
        包裹), 真实 prompt 里正文出现在 [main 历史 + 标记] 之后. 这里直接在
        渲染后的 prompt token 序列中搜索子输出正文, 返回其起始位置与对应 KV,
        引擎据此把子输出 KV 精确插到该位置, 其余部分照常 prefill.

        定位策略:
        - 锚点: 子输出是"本轮新内容", 必然出现在最新 main 段之后 —— 搜索起点
          取 len(最新 main 段), 历史里与子输出相似的文本(如任务原文)直接被排除;
        - 最长匹配前缀: 客户端把输出解码成文本再回填, 模板对同一文本重新分词
          —— BPE 分词对同一字符串是确定的, 但正文**首尾边界**可能与相邻换行/
          标记合并(如结尾 '。' 与模板追加的 '\n' 合成一个 token '。\n'), 导致
          渲染出的 token 序列与保存的 body 在边界处不一致. 这里搜索 body 在
          prompt 中**逐位一致的最长前缀**(允许开头 1-2 个 token 并入前一标记),
          只拼接一致的部分, 其余(通常是尾部 1-2 个边界 token)继续 prefill;
        - 回退: 匹配太短(不足 4 token, 可能误命中)或未找到时返回 None,
          引擎安全回退到 LCP 模式, 绝不错误拼接.

        返回 KVGraft(position/tokens/cache), 无 sub 段或定位失败时返回 None.
        """
        if len(trace) < 2 or trace[-1] != KIND_MAIN or trace[-2] == KIND_MAIN:
            return None
        segs = self._segments.get(session_id)
        if not segs:
            return None
        sub_seg = segs.get(KIND_SUB)
        if sub_seg is None:
            return None
        self._stats["reuse_attempts"] += 1

        # 子输出正文 = 输出中剔除开头 think 块(客户端回填时会剥离 thinking)
        body_start = sub_seg.output_start + sub_seg.think_len
        body = sub_seg.tokens[body_start:]
        if not body:
            logger.info("会话 %s: 子 agent 段没有输出正文, 无法拼接", session_id)
            return None

        # 锚点: 子输出是"新内容", 必然出现在最新 main 段之后
        main_seg = segs["main"]
        min_pos = len(main_seg.tokens) if main_seg else 0
        # 最长逐位匹配前缀: 允许开头 1-2 个 token 并入前一标记(边界重分词),
        # 在 [min_pos, len) 内找 (位置, 匹配长度, 跳过前缀长度) 的最优组合
        n, m = len(prompt_tokens), len(body)
        max_off = min(3, m)  # off = 跳过 body 前 off 个 token(它们并入了插入点前的标记)
        best_pos, best_k, best_off = -1, 0, 0
        for i in range(min_pos, n):
            for off in range(max_off):
                if i - off < min_pos or i + (m - off) > n:
                    continue
                if prompt_tokens[i] != body[off]:
                    continue
                k = 0
                while k < m - off and prompt_tokens[i + k] == body[off + k]:
                    k += 1
                if k > best_k:  # 同长取更早位置(i 升序遍历天然满足)
                    best_pos, best_k, best_off = i, k, off
        # 匹配太短不可靠(可能误命中), 保守放弃拼接
        if best_pos < 0 or best_k < min(4, m):
            logger.info("会话 %s: 未在 prompt 中定位到子 agent 输出正文(%d tok), "
                        "放弃拼接(回退 LCP)", session_id, m)
            return None
        pos, k, off = best_pos, best_k, best_off
        body = body[off:off + k]  # 只拼接逐位一致的部分
        cache = slice_cache(
            tail_cache(sub_seg.cache, body_start + off, self._config),
            k, self._config)
        logger.info("会话 %s: 定位子 agent 输出正文 KV %d tok(prompt 位置 %d..%d, "
                    "剔除 think %d tok / 边界漂移 %d tok), 准备插入",
                    session_id, k, pos, pos + k - 1, sub_seg.think_len, m - k)
        return KVGraft(position=pos, tokens=body, cache=cache)

    def propose(self, session_id: str, trace: list[str]) -> list[KVPrefix]:
        """给出可尝试复用的候选段(空列表 = 本次不尝试).

        触发条件已放宽(不再限定"main 在子 agent 返回后继续"): 只要会话里有
        已保存的段就给出候选 —— main 请求给最新 main 段 + 最新 sub 段,
        sub 请求给最新 sub 段(同一 sub 的多轮续接)。具体能否复用、复用多少
        由引擎的 LCP 匹配决定(前缀不一致自动回退全量 prefill, 零风险).
        """
        segs = self._segments.get(session_id)
        if not segs:
            return []
        self._stats["reuse_attempts"] += 1
        if trace and trace[-1] == KIND_MAIN:
            candidates: list[KVPrefix] = []
            if segs["main"] is not None:
                candidates.append(segs["main"])  # 最新 main 段(含其输出 KV)
            if segs["sub"] is not None:
                candidates.append(segs["sub"])   # 最新 sub 段(含其输出 KV)
            return candidates
        if segs["sub"] is not None:
            return [segs["sub"]]  # sub 请求: 该 sub 最近一轮的段(续接时可复用)
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
