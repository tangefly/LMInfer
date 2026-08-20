"""Agent 会话注册表: 把一个 agent 任务(主/子 agent)的多次模型调用关联起来.

设计:
- 会话由 LMInfer 在 agent 模式首次请求时创建(uuid 字符串), session_id 随响应
  回传给应用; 应用在后续请求中回传, 从而标识同一个大任务的请求;
- 每次请求还携带调用路径 trace(如 ["main","sub1","main"]), 末位是当前 agent,
  注册表记录最新 trace 与出现过的 agent 名单, 并累计每次请求的 token 消耗;
- 所有读写都发生在 asyncio 事件循环上的请求 handler 中(单线程), 因此无需加锁.
"""

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class AgentSession:
    """一个 agent 事务会话的运行时信息."""

    session_id: str
    created_at: float                   # 首次请求时间(unix 秒)
    last_seen_at: float = field(default=0.0)     # 最近一次请求时间
    trace: list[str] = field(default_factory=list)    # 最新一次请求的调用路径
    agents: list[str] = field(default_factory=list)   # 出现过的 agent 名单(按首次出现顺序)
    request_count: int = 0              # 已完成的请求数
    prompt_tokens: int = 0              # 累计 prompt token
    completion_tokens: int = 0          # 累计生成 token
    kv_reuse_count: int = 0             # 成功复用前缀 KV 的请求数(跨请求 KV 复用)
    kv_reuse_tokens: int = 0            # 累计复用的前缀 KV token 数

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
            "trace": self.trace,
            "agents": self.agents,
            "request_count": self.request_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "kv_reuse_count": self.kv_reuse_count,
            "kv_reuse_tokens": self.kv_reuse_tokens,
        }


class AgentSessionRegistry:
    """agent session id 列表: 取回/新建会话、校验存在性、累计用量."""

    def __init__(self) -> None:
        self._sessions: dict[str, AgentSession] = {}

    def get_or_create(self, session_id: str | None, trace: list[str]) -> tuple[str, bool]:
        """取回或新建会话, 返回 (session_id, 是否新创建).

        - session_id 缺省: 生成 uuid 创建新会话(首次 agent 请求);
        - session_id 未知: 抛 KeyError(由 server 转成 404, 不静默新建,
          避免客户端传错 id 时悄悄开新会话污染统计).
        """
        now = time.time()
        if session_id is None:
            session_id = uuid.uuid4().hex[:16]
            self._sessions[session_id] = AgentSession(
                session_id=session_id, created_at=now, last_seen_at=now, trace=list(trace),
                agents=[trace[-1]],  # 首个 agent 即当前 agent
            )
            return session_id, True
        s = self._sessions.get(session_id)
        if s is None:
            raise KeyError(session_id)
        s.last_seen_at = now
        s.trace = list(trace)
        if trace[-1] not in s.agents:
            s.agents.append(trace[-1])
        return session_id, False

    def record_usage(self, session_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        """生成结束后累计一次请求的 token 消耗(流式在 done 事件处调用)."""
        s = self._sessions.get(session_id)
        if s is None:
            return  # 会话不存在时忽略, 统计不影响生成
        s.request_count += 1
        s.prompt_tokens += prompt_tokens
        s.completion_tokens += completion_tokens

    def record_kv_reuse(self, session_id: str, reused_tokens: int) -> None:
        """累计一次成功的前缀 KV 复用(配合 --reuse-agent-kv)."""
        s = self._sessions.get(session_id)
        if s is None:
            return
        s.kv_reuse_count += 1
        s.kv_reuse_tokens += reused_tokens

    def list_sessions(self) -> list[dict]:
        """按创建时间排序的全部会话(供 GET /v1/agent/sessions)."""
        return [s.to_dict() for s in sorted(self._sessions.values(), key=lambda s: s.created_at)]

    def __len__(self) -> int:
        return len(self._sessions)
