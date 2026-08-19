# LMInfer

一个刻意保持**朴素**（naive）的 LLM 推理服务，用于学习与验证 **KV Cache** 相关理论。
只依赖 transformers 的高层 API（模型加载、`DynamicCache`、采样 warper、chat template），
提供 **vLLM 风格的启动命令** 与 **OpenAI 兼容的 HTTP 接口**。

> vLLM 太复杂？先跑通这个精简版本，把 prefill / decode / KV cache / 并发这几件事
> 看明白，再回去读 vLLM 的 scheduler 和 paged attention 会轻松很多。

## 设计理念（为什么这样写）

| 关注点 | 朴素做法 | 对比 vLLM |
|---|---|---|
| 批处理 | 每个请求独占 batch=1，**不拼接不 padding**，mask 恒为全 1 | 动态 batch、padding、变长处理 |
| KV cache | 完全交给 `transformers.DynamicCache`，只传递对象 | 自研 PagedAttention + BlockManager |
| 并发 | `ThreadPoolExecutor(max_num_seqs)`，每个请求一个线程跑生成循环 | 事件循环 + scheduler + worker 分层 |
| 显存管理 | 动态分配，不做预分配 | `--gpu-memory-utilization` 预分 KV 池 |
| 采样 | 复用 transformers 的 LogitsProcessor | 自研 sampler |

三个关键决策：

1. **batch=1**：这是 5.x 里语义最安全的用法——attention mask 全 1、position_ids 用默认值，
   不需要处理任何 padding 边界。代价是并发请求之间没有共享 decode 步，这就是
   "连续批处理" 被省掉的部分（理解之后可以自己加）。
2. **KV cache 交给 transformers**：`model.forward(past_key_values=cache)` 之后，
   新 token 的 K/V 由模型内部追加进 cache。我们只做两件事——prefill 时传入空 cache，
   decode 时把 cache 传回去。这正是研究 KV cache 最关心的部分，全部显式可见。
3. **并发 = 线程池**：多个请求在不同线程里各自跑生成循环，GPU 计算天然串行化。
   这就是"朴素版的连续批处理"，也是理解 vLLM 事件循环的好起点。

## 目录结构

```
LMInfer/
├── lminfer/                  # 核心包
│   ├── cli.py                # lminfer serve / chat 命令行入口
│   ├── config.py             # EngineConfig / SamplingParams
│   ├── engine.py             # ★ 核心: prefill + decode 生成循环, KV cache 显式可见
│   ├── schemas.py            # OpenAI 兼容请求模型
│   └── server.py             # FastAPI 路由 + SSE 流式
├── examples/
│   ├── client.py             # 客户端示例(chat/completions, 流式/非流式)
│   └── bench.py              # 并发压测(TTFT / TPOT / 吞吐)
├── experiments/
│   └── kv_cache_compare.py   # ★ 理论验证实验: KV cache 开/关对比
└── README.md
```

## 快速开始

```bash
pip install -e .

# 启动服务(两种写法都与 vLLM 一致: 位置参数 或 --model)
lminfer serve /home/tanger/workspace/models/Qwen2.5-7B-Instruct --port 8000
lminfer serve --model /home/tanger/workspace/models/Qwen3-0.6B --max-num-seqs 4

# 也可以不带参数安装直接运行
python -m lminfer serve /home/tanger/workspace/models/Qwen3-0.6B
```

`--served-model-name` 与 vLLM 语义一致（覆盖对外暴露的模型名）；
`--gpu-memory-utilization`、`--tensor-parallel-size`、`--kv-transfer-config`、
`--enable-auto-tool-choice`、`--tool-call-parser` 等参数会被接受，
但朴素实现中不生效（启动时会打印 WARNING 说明原因）。

验证服务：

```bash
curl http://localhost:8000/health
curl http://localhost:8000/v1/models

# 对话(流式)
curl -N http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "用一句话介绍大语言模型"}], "stream": true}'

# 文本补全
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 32, "temperature": 0}'

# 客户端示例 / 交互式对话 / 压测
python examples/client.py --stream "讲个冷笑话"
lminfer chat --model /home/tanger/workspace/models/Qwen3-0.6B
python examples/bench.py --prompts 8 --max-tokens 64
```

支持的采样参数：`temperature`（0=贪心）、`top_p`、`top_k`、`repetition_penalty`、
`max_tokens`、`stop`（字符串或列表）、`stream`。`n`、`suffix`、`seed` 等暂不支持。

## 支持的接口

| 接口 | 说明 |
|---|---|
| `GET /health` | 健康检查 |
| `GET /v1/models` | 模型列表 |
| `POST /v1/completions` | 文本补全（OpenAI 格式，支持 stream） |
| `POST /v1/chat/completions` | 对话补全（支持 stream） |
| `GET /v1/stats` | 运行时统计（KV 占用/请求数/吞吐，朴素实现额外提供） |

## KV Cache 理论速览（本项目要验证的东西）

### 为什么需要 KV cache

一次 transformer 前向里，attention 层要做：

```
Q = x @ Wq        K = x @ Wk        V = x @ Wv
Attn = softmax(Q @ K^T / √d) @ V
```

生成第 n 个 token 时，**K/V 只依赖已经生成的前缀**，与"当前正在预测哪个 token"无关。
所以可以把历史 token 的 K/V 存起来（这就是 KV cache），decode 时只计算新 token 自己的
Q/K/V 再与缓存的 K/V 做 attention：

- **prefill**：一次性前向整个 prompt，计算并缓存所有层的 K/V。计算量 ∝ prompt 长度。
- **decode**：每步只输入 1 个 token，计算量**与序列长度无关**（O(1) 注意力）。

如果不缓存（本项目 `experiments/kv_cache_compare.py` 的对照模式），每步都要重算整个前缀，
总成本约 O(n²)，并且越到后面越慢。

### KV cache 的显存开销

```
每 token 新增 KV 字节数 = 2 × 层数 × KV头数 × head_dim × 每元素字节数
```

例如 Qwen2.5-7B-Instruct（28 层，4 个 KV 头，head_dim 128，bf16）：
`2 × 28 × 4 × 128 × 2 = 57344 B ≈ 56 KiB/token`，32K 上下文就需要约 1.75 GiB。
这就是 vLLM 用 PagedAttention 管理显存、以及 KV cache 量化/压缩成为研究方向的原因。

### 动手实验

```bash
python experiments/kv_cache_compare.py --model /home/tanger/workspace/models/Qwen3-0.6B --tokens 32
```

会输出两种模式的 prefill 耗时、decode 吞吐、KV 显存占用，并给出速度提升倍数。
每次服务启动时也会打印当前模型"每 token KV 占用"的理论值（`/v1/stats` 里也有）。

## 与 vLLM 的差异（学习路线图）

1. **连续批处理**：本项目线程池并发，每个请求独立 decode；vLLM 把多个序列拼进同一个
   decode 步共享矩阵乘法。下一步改造：在引擎里维护序列列表，把长度相同的序列拼批。
2. **PagedAttention**：本项目 KV 是一个随长度增长的连续张量；vLLM 按 16-token 块分配，
   解决碎片化与内存浪费。理解 `DynamicCache` 之后再对比 `StaticCache`。
3. **调度与抢占**：vLLM 有 preemption（长序列被抢占时交换或重算）；本项目直接排队。
4. **前缀缓存**：vLLM 有 automatic prefix caching；本项目完全没有。
