"""Request-local GPU snapshots for an in-process vLLM V1 connector.

Snapshots own storage, allowing vLLM to free its paged blocks after requests.
No vLLM imports here: paged operations are also testable on CPU.
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


def check_layout(cache, block_size):
    if cache.ndim != 5 or cache.shape[1] != 2 or cache.shape[2] != block_size:
        raise ValueError(f"Expected FLASH_ATTN [blocks,2,{block_size},heads,dim], got {cache.shape}")


def read_paged(cache, block_ids, start, end, block_size):
    check_layout(cache, block_size)
    blocks, offsets = slots_for(block_ids, start, end, block_size, cache.device)
    values = cache[blocks, :, offsets]
    return values[:, 0].permute(1, 0, 2)[None], values[:, 1].permute(1, 0, 2)[None]


def write_paged(cache, block_ids, keys, values, block_size):
    check_layout(cache, block_size)
    expected = (1, cache.shape[3], keys.shape[-2], cache.shape[4])
    if tuple(keys.shape) != expected or tuple(values.shape) != expected:
        raise ValueError("KV dimensions do not match vLLM cache")
    if keys.dtype != cache.dtype or values.dtype != cache.dtype:
        raise ValueError("KV dtype does not match vLLM cache")
    blocks, offsets = slots_for(block_ids, 0, keys.shape[-2], block_size, cache.device)
    cache[blocks, 0, offsets] = keys[0].permute(1, 0, 2).to(cache.device)
    cache[blocks, 1, offsets] = values[0].permute(1, 0, 2).to(cache.device)


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
        if index not in self.layers:
            shape = (*keys.shape[:2], self.capacity, keys.shape[-1])
            self.layers[index] = (keys.new_empty(shape), values.new_empty(shape))
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
        layers = [(self.layers[i][0][:, :, :length].clone(),
                   self.layers[i][1][:, :, :length].clone())
                  for i in range(self.num_layers)]
        return DynamicCache(ddp_cache_data=layers, config=config)


@dataclass
class Bridge:
    stage: Stage | None = None


# Only the engine's single inference thread mutates its stage.
BRIDGES: dict[str, Bridge] = {}
