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
    attn_implementation: str = "auto"  # auto / eager / sdpa / flash_attention_2(需安装 flash-attn)
    max_model_len: int = 4096       # 单序列最大总长度(prompt + 生成部分)
    max_num_seqs: int = 4           # 最大并发请求数(朴素并发 = 线程池大小)
    served_model_name: str | None = None  # 对外暴露的模型名(vLLM 的 --served-model-name)
    trust_remote_code: bool = False  # 允许执行远程模型代码(个别模型需要)
    disable_log_stats: bool = False  # 关闭每请求统计日志
    enable_thinking: bool | None = None  # Qwen3 等模型的 thinking 开关, None 表示不传
    tool_call_parser: str = "auto"  # auto/qwen: 解析 <tool_call> 块; none: 关闭(对应 vLLM 的 --tool-call-parser)
    reuse_agent_kv: bool = False    # agent 模式跨请求前缀 KV 复用(见 kvcache.py): 同一会话内
                                    # 后续请求复用已保存的前缀 KV(prompt + 输出), 跳过重复 prefill
    reuse_agent_kv_append: bool = False  # 位置感知拼接模式(实验): 在渲染后的 prompt 中定位
                                         # 子 agent 输出正文, 把其 KV 直接插入 main 的 KV cache
                                         # 对应位置(见 kvcache.build_graft), main 历史仍按 LCP
                                         # 复用, 定位失败自动回退; 子输出 KV 在子 agent 自己的
                                         # 上下文里计算, 与全量 prefill 存在近似差异
    kv_segment_idle_ttl: float = 3600.0  # 已保存 KV 段的会话闲置超时(秒): 超过后整段清理,
                                         # 防止会话注册表无 TTL 导致 KV 显存随会话永久增长


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
