"""Request-local GPU snapshots for an in-process vLLM V1 connector.

Snapshots own storage, allowing vLLM to free its paged blocks after requests.
No vLLM imports here: paged operations are also testable on CPU.

vLLM 每一层分页 KV 的布局换过几次, 三种都要认(见 `PagedLayout`):

| kind | 来源 | 形状 | 内容维 |
|---|---|---|---|
| `pair` | 0.23 时代 FLASH_ATTN | `[blocks, 2, block_size, heads, dim]` | K/V 各占一个轴 |
| `packed` | 0.27+ FLASH_ATTN | `(blocks, heads, block_size, 2*head_size)` | K=`[:d]`, V=`[d:]` |
| `mla` | MLA 家族(GLM-4.7-Flash) | `(blocks, block_size, kv_lora_rank + qk_rope_head_dim)` | 潜向量=`[:rank]`, k_rot=`[rank:]` |

MLA 的尾维是**紧邻拼接**的压缩潜向量与 `k_rot`(vLLM 的 MLA kernel 就按
`[..., :kv_lora_rank]` 取 V、`[..., kv_lora_rank:]` 取 K)。落到仓库的
`DynamicCache` 约定时, 位置无关的潜向量进 **keys 槽**, 带位置的 `k_rot` 进
**values 槽** —— 与 transformers 后端、`RopeLayout.rotated_slot == "values"` 一致,
所以 `kvcache.py` 的切片/拼接/RoPE rebase 全部原样复用。
"""
from dataclasses import dataclass, field
import re
import torch
from transformers import DynamicCache


def layer_index(name):
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"Unsupported attention layer name: {name}")
    return int(match[1])


def slots_for(block_ids, start, end, block_size, device):
    if start < 0 or end < start or end > len(block_ids) * block_size:
        raise ValueError("KV token range exceeds allocated blocks")
    positions = torch.arange(start, end, device=device)
    blocks = torch.tensor(block_ids, dtype=torch.long, device=device)
    return blocks[positions // block_size], positions % block_size


@dataclass(frozen=True)
class PagedLayout:
    """一层分页 KV 的布局. `latent_dim` 仅 MLA 有意义(潜向量宽度)."""

    kind: str
    block_size: int
    num_heads: int
    content_dim: int
    latent_dim: int | None = None

    @property
    def is_mla(self):
        return self.kind == "mla"


def detect_layout(cache, block_size, latent_dim=None) -> PagedLayout:
    """按张量维度识别布局; 形状不符合任何已知布局时 fail-closed."""
    shape = tuple(cache.shape)
    if cache.ndim == 5:
        if shape[1] != 2 or shape[2] != block_size:
            raise ValueError(
                f"Expected FLASH_ATTN [blocks,2,{block_size},heads,dim], got {shape}")
        return PagedLayout("pair", block_size, shape[3], shape[4])
    if cache.ndim == 4:
        if shape[2] != block_size or shape[3] % 2:
            raise ValueError(
                f"Expected FLASH_ATTN [blocks,heads,{block_size},2*dim], got {shape}")
        return PagedLayout("packed", block_size, shape[1], shape[3] // 2)
    if cache.ndim == 3:
        if shape[1] != block_size:
            raise ValueError(
                f"Expected MLA [blocks,{block_size},latent+rope], got {shape}")
        if not isinstance(latent_dim, int) or not 0 < latent_dim < shape[2]:
            raise ValueError(
                f"MLA paged cache needs 0 < latent_dim < {shape[2]} (kv_lora_rank), "
                f"got {latent_dim!r}")
        return PagedLayout("mla", block_size, 1, shape[2], latent_dim)
    raise ValueError(f"Unsupported vLLM KV cache layout: {shape}")


def read_paged(cache, block_ids, start, end, block_size, latent_dim=None):
    """把分页 cache 里的 [start, end) 读成 (keys, values) 两块连续张量.

    两者都是 `[1, heads, tokens, dim]`(MLA 的 heads 恒为 1), 即仓库
    `DynamicCache` 每一层的 keys/values 约定.
    """
    layout = detect_layout(cache, block_size, latent_dim)
    blocks, offsets = slots_for(block_ids, start, end, block_size, cache.device)
    if layout.kind == "pair":
        values = cache[blocks, :, offsets]
        return (values[:, 0].permute(1, 0, 2)[None],
                values[:, 1].permute(1, 0, 2)[None])
    if layout.kind == "packed":
        rows = cache[blocks, :, offsets]          # [tokens, heads, 2*dim]
        dim = layout.content_dim
        return (rows[..., :dim].permute(1, 0, 2)[None],
                rows[..., dim:].permute(1, 0, 2)[None])
    rows = cache[blocks, offsets]                 # [tokens, latent + rope]
    rank = layout.latent_dim
    return rows[:, :rank][None, None], rows[:, rank:][None, None]


def write_paged(cache, block_ids, keys, values, block_size):
    """把 (keys, values) 写回分页 cache 的 [0, tokens) 位置. 形状必须逐位相符."""
    if keys.ndim != 4 or values.ndim != 4 or keys.shape[0] != 1 or values.shape[0] != 1:
        raise ValueError(f"Expected [1, heads, tokens, dim] KV slots, got "
                         f"{tuple(keys.shape)} / {tuple(values.shape)}")
    layout = detect_layout(cache, block_size, keys.shape[-1])
    tokens = keys.shape[-2]
    if values.shape[-2] != tokens:
        raise ValueError("keys and values cover different token counts")
    if keys.dtype != cache.dtype or values.dtype != cache.dtype:
        raise ValueError("KV dtype does not match vLLM cache")
    blocks, offsets = slots_for(block_ids, 0, tokens, block_size, cache.device)
    if layout.kind == "pair":
        expected = (1, layout.num_heads, tokens, layout.content_dim)
        if tuple(keys.shape) != expected or tuple(values.shape) != expected:
            raise ValueError("KV dimensions do not match vLLM cache")
        cache[blocks, 0, offsets] = keys[0].permute(1, 0, 2).to(cache.device)
        cache[blocks, 1, offsets] = values[0].permute(1, 0, 2).to(cache.device)
    elif layout.kind == "packed":
        expected = (1, layout.num_heads, tokens, layout.content_dim)
        if tuple(keys.shape) != expected or tuple(values.shape) != expected:
            raise ValueError("KV dimensions do not match vLLM cache")
        dim = layout.content_dim
        cache[blocks, :, offsets, :dim] = keys[0].permute(1, 0, 2).to(cache.device)
        cache[blocks, :, offsets, dim:] = values[0].permute(1, 0, 2).to(cache.device)
    else:
        if keys.shape[1] != 1 or values.shape[1] != 1:
            raise ValueError("MLA KV slots have a single head")
        rank = keys.shape[-1]
        if rank + values.shape[-1] != layout.content_dim:
            raise ValueError("KV dimensions do not match vLLM cache")
        cache[blocks, offsets, :rank] = keys[0, 0].to(cache.device)
        cache[blocks, offsets, rank:] = values[0, 0].to(cache.device)


@dataclass
class Stage:
    tokens: list[int]
    prefix: DynamicCache | None
    capacity: int
    num_layers: int
    layers: dict = field(default_factory=dict)
    valid: dict = field(default_factory=dict)
    computed_tokens: int = 0
    loaded_tokens: int = 0

    @property
    def prefix_length(self):
        return self.prefix.get_seq_length() if self.prefix is not None else 0

    def save(self, index, keys, values, start, end):
        if not 0 <= index < self.num_layers or not 0 <= start < end <= self.capacity:
            raise ValueError("Invalid snapshot layer or token interval")
        if keys.shape[:2] != values.shape[:2] or keys.shape[-2] != values.shape[-2]:
            raise ValueError("keys and values disagree on batch/head/token dimensions")
        if index not in self.layers:
            # 两个槽独立分配: MLA 的潜向量(512)与 k_rot(64)宽度不同, 不能互相取形状
            self.layers[index] = (
                keys.new_empty((*keys.shape[:2], self.capacity, keys.shape[-1])),
                values.new_empty((*values.shape[:2], self.capacity, values.shape[-1])),
            )
            if self.prefix is not None:
                old = self.prefix.layers[index]
                self.layers[index][0][:, :, :self.prefix_length].copy_(old.keys)
                self.layers[index][1][:, :, :self.prefix_length].copy_(old.values)
            self.valid[index] = self.prefix_length
        if start != self.valid[index]:
            raise RuntimeError("Non-contiguous KV capture (preemption is unsupported)")
        k, v = self.layers[index]
        k[:, :, start:end].copy_(keys)
        v[:, :, start:end].copy_(values)
        self.valid[index] = end

    def finish(self, config):
        if set(self.valid) != set(range(self.num_layers)):
            raise RuntimeError("vLLM did not capture all KV layers")
        lengths = set(self.valid.values())
        if len(lengths) != 1:
            raise RuntimeError("Captured KV layers have different lengths")
        length = lengths.pop()
        # 这里**不要** clone: DynamicCache.__init__ 会把每个槽 cat 进自己的空张量
        # (DynamicLayer.update -> torch.cat), 那一步已经复制了一份。再 clone 一份
        # 会让每层快照的瞬时占用翻倍 —— 47 层 MLA 下这正是 agent 长会话 OOM 的来源
        # (实测: main 第 19 个请求, 分配 38 MiB 时整卡只剩 28 MiB)。
        # 独立性由 test_snapshot_prefix_chunked_capture_and_ownership 守住:
        # 若将来 DynamicCache 改成持有引用而不复制, 那条用例会失败。
        layers = [(self.layers[i][0][:, :, :length],
                   self.layers[i][1][:, :, :length])
                  for i in range(self.num_layers)]
        return DynamicCache(ddp_cache_data=layers, config=config)


@dataclass
class Bridge:
    stage: Stage | None = None


# Only the engine's single inference thread mutates its stage.
BRIDGES: dict[str, Bridge] = {}
