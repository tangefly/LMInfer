"""配置定义: 引擎参数与采样参数.

刻意保持朴素: 参数数量只覆盖日常实验需要的那一小部分,
但命名与 vLLM 保持一致, 方便对照记忆.
"""

from dataclasses import dataclass, field
from .repair import repair_ratio


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
    dequantize_fp8: bool | None = None  # fine-grained FP8 权重(如 Ministral-3)是否在加载期
                                        # 反量化成模型 dtype: None=自动(没装 kernels 包就反量化),
                                        # True/False 由 --dequantize-fp8/--no-dequantize-fp8 强制
    tool_call_parser: str = "auto"  # auto/qwen/hermes: 解析 <tool_call> 块; llama3_json:
                                    # 解析 Llama 3.x 的 {"name":...,"parameters":...} JSON;
                                    # mistral: 解析 [TOOL_CALLS]name[ARGS]{json};
                                    # glm4: 解析 GLM-4 的 name\n{json};
                                    # glm4_moe: 解析 GLM-4.5/4.6/4.7 的
                                    #   <tool_call>name<arg_key>k</arg_key><arg_value>v
                                    #   </arg_value></tool_call>(别名 glm45/glm47 同义,
                                    #   与 vLLM 的 --tool-call-parser 名字对齐);
                                    # none: 关闭解析. auto 按 config.model_type/tokenizer
                                    # 自动识别(见 model_adapters.py, 对应 vLLM 的
                                    # --tool-call-parser)
    enable_auto_tool_choice: bool = False  # 请求带 tools 且未显式给 tool_choice 时默认按 auto
                                           # 处理; 关闭时默认 none(忽略 tools). 显式 tool_choice
                                           # 始终优先(对应 vLLM 的 --enable-auto-tool-choice)
    reuse_agent_kv: bool = False    # agent 模式跨请求前缀 KV 复用(见 kvcache.py): 同一会话内
                                    # 后续请求复用已保存的前缀 KV(prompt + 输出), 跳过重复 prefill
    reuse_agent_kv_append: bool = False  # 位置感知拼接模式(实验): 在渲染后的 prompt 中定位
                                         # 子 agent 输出正文, 把其 KV 直接插入 main 的 KV cache
                                         # 对应位置(见 kvcache.build_graft), main 历史仍按 LCP
                                         # 复用, 定位失败自动回退; 子输出 KV 在子 agent 自己的
                                         # 上下文里计算, 与全量 prefill 存在近似差异
    graft_rope_rebase: bool = False      # 拼接前把子输出 K 从子上下文 RoPE 位置重映射到
                                         # main prompt 位置; 只修正位置差, 不修正上下文差
    repair_window_begin: float = 0.0    # 每段复用 KV 首部重算比例 [0, 1]
    repair_window_end: float = 0.0      # 每段复用 KV 尾部重算比例 [0, 1]
    repair_mode: str = "window"       # window / context / exact
    repair_shallow_layers: int = 1    # 完整计算 attention/MLP 的浅层数
    repair_budget: float = 0.3        # 每段深层修复比例上限(不含 probes)
    repair_coverage: float = 0.95     # 累计评分覆盖目标, 不是误差保证
    repair_min_ratio: float = 0.05    # 每段最低修复比例
    repair_probe_tokens: int = 8      # 未选中位置中的确定性检查点
    repair_probe_threshold: float = 0.0  # 深层相对 KV 偏差阈值; 0 禁用精确回退
    kv_segment_idle_ttl: float = 3600.0  # 已保存 KV 段的会话闲置超时(秒): 超过后整段清理,
                                         # 防止会话注册表无 TTL 导致 KV 显存随会话永久增长


    backend: str = "transformers"
    gpu_memory_utilization: float = 0.5
    tensor_parallel_size: int = 1

    def __post_init__(self):
        if self.backend not in ("transformers", "vllm"):
            raise ValueError("backend must be transformers or vllm")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1)")
        if self.backend == "vllm":
            if self.max_model_len < 2:
                raise ValueError("vllm backend requires max_model_len >= 2")
            if self.tensor_parallel_size != 1:
                raise ValueError("vllm agent KV backend currently requires tensor_parallel_size=1")
            if self.repair_mode == "context":
                raise ValueError("vllm backend supports window/exact repair; context repair is not implemented")
            if self.dtype == "float32":
                raise ValueError("vllm agent KV backend requires float16 or bfloat16")
        self.repair_window_begin = repair_ratio(self.repair_window_begin)
        self.repair_window_end = repair_ratio(self.repair_window_end)
        if self.repair_mode not in ("window", "context", "exact"):
            raise ValueError("repair_mode must be window, context or exact")
        for name in ("repair_budget", "repair_coverage", "repair_min_ratio"):
            setattr(self, name, repair_ratio(getattr(self, name)))
        if self.repair_min_ratio > self.repair_budget:
            raise ValueError("repair_min_ratio must not exceed repair_budget")
        if not isinstance(self.repair_shallow_layers, int) or self.repair_shallow_layers < 1:
            raise ValueError("repair_shallow_layers must be a positive integer")
        if self.repair_probe_threshold > 0 and self.repair_probe_tokens == 0:
            raise ValueError("probe threshold requires probe tokens")
        import math
        if (not isinstance(self.repair_probe_tokens, int) or self.repair_probe_tokens < 0
                or not math.isfinite(self.repair_probe_threshold)
                or self.repair_probe_threshold < 0):
            raise ValueError("invalid repair probe settings")


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
