"""OpenAI 兼容的请求模型(与 vLLM 的 /v1 接口保持一致的字段)."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .config import SamplingParams


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None            # assistant 工具调用消息为 null, 与 OpenAI 一致
    tool_calls: list[dict] | None = None  # assistant 消息: OpenAI 格式的工具调用列表
    tool_call_id: str | None = None       # tool 消息: 关联的 assistant 工具调用 id


class _SamplingFields(BaseModel):
    """采样参数公共字段(completions 与 chat/completions 共用)."""

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int | None = None            # -1 表示不启用(OpenAI 无此字段, vLLM 有)
    repetition_penalty: float | None = Field(default=None, gt=0.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    seed: int | None = None             # 朴素实现忽略 seed, 仅接受字段

    @field_validator("stop")
    @classmethod
    def _normalize_stop(cls, v):
        """把单个字符串统一成列表, 方便后续处理."""
        if isinstance(v, str):
            return [v]
        return v

    def to_sampling(self, default_max_tokens: int = 256) -> SamplingParams:
        """按 vLLM 的默认值填充未显式给出的字段."""
        return SamplingParams(
            temperature=1.0 if self.temperature is None else self.temperature,
            top_p=1.0 if self.top_p is None else self.top_p,
            top_k=-1 if self.top_k is None else self.top_k,
            repetition_penalty=1.0 if self.repetition_penalty is None else self.repetition_penalty,
            max_tokens=default_max_tokens if self.max_tokens is None else self.max_tokens,
            stop=[] if self.stop is None else self.stop,
        )


class CompletionRequest(_SamplingFields):
    """POST /v1/completions 请求体."""

    model: str | None = None
    prompt: str                     # 朴素实现只支持单个字符串 prompt
    n: int = Field(default=1, ge=1)  # 朴素实现只支持 n=1, 其他值返回 400
    suffix: str | None = None       # 不支持, 仅接受字段


class ChatCompletionRequest(_SamplingFields):
    """POST /v1/chat/completions 请求体."""

    model: str | None = None
    messages: list[ChatMessage]
    tools: list[dict] | None = None      # OpenAI 函数 schema 列表, 渲染进 chat template
    tool_choice: str | dict | None = None  # "auto"/"none"/"required" 或 {"type":"function",...}

