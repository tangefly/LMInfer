"""配置定义: 引擎参数与采样参数.

刻意保持朴素: 参数数量只覆盖日常实验需要的那一小部分,
但命名与 vLLM 保持一致, 方便对照记忆.
"""

from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    """服务端引擎配置(对应 vLLM 的 EngineArgs)."""

    model: str                      # 模型路径或 HF 上的模型名
    dtype: str = "auto"             # auto / bfloat16 / float16 / float32
    device_map: str = "auto"        # 模型放置方式, 与参考脚本一致
    max_model_len: int = 4096       # 单序列最大总长度(prompt + 生成部分)
    max_num_seqs: int = 4           # 最大并发请求数(朴素并发 = 线程池大小)
    trust_remote_code: bool = False  # 允许执行远程模型代码(个别模型需要)
    disable_log_stats: bool = False  # 关闭每请求统计日志
    enable_thinking: bool | None = None  # Qwen3 等模型的 thinking 开关, None 表示不传


@dataclass
class SamplingParams:
    """采样参数(对应 vLLM 的 SamplingParams)."""

    temperature: float = 1.0        # 0 表示贪心(与 OpenAI 约定一致)
    top_p: float = 1.0              # 1.0 表示不启用
    top_k: int = -1                 # -1 表示不启用
    repetition_penalty: float = 1.0  # 1.0 表示不启用
    max_tokens: int = 256           # 最多生成 token 数
    stop: list[str] = field(default_factory=list)  # 遇到这些子串即停止

    @property
    def greedy(self) -> bool:
        """temperature == 0 时走贪心解码(取 argmax)."""
        return self.temperature == 0
