"""vLLM V1 connector for single-GPU segmented prefill."""
from dataclasses import dataclass, field
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata,
)
from .vllm_bridge import BRIDGES, layer_index, read_paged, write_paged


@dataclass
class SegmentMetadata(KVConnectorMetadata):
    # (block IDs, computed start, computed end, import prefix)
    operations: list = field(default_factory=list)


class AgentKVConnector(KVConnectorBase_V1):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        self.block_size = vllm_config.cache_config.block_size
        bridge_id = self._kv_transfer_config.get_from_extra_config("bridge_id", None)
        if bridge_id not in BRIDGES:
            raise RuntimeError("LMInfer connector requires VLLM_ENABLE_V1_MULTIPROCESSING=0")
        self.bridge = BRIDGES[bridge_id]
        self.blocks = {}

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config):
        return "NHD"

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        stage = self.bridge.stage
        if stage is None or list(request.prompt_token_ids) != stage.tokens:
            raise RuntimeError("Request does not match the active LMInfer KV stage")
        if stage.prefix_length >= len(stage.tokens):
            raise ValueError("The last prompt token must be computed for logits")
        return max(0, stage.prefix_length - num_computed_tokens), False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        meta = SegmentMetadata()
        for req in scheduler_output.scheduled_new_reqs:
            if len(req.block_ids) != 1:
                raise RuntimeError("Only one full-attention KV group is supported")
            self.blocks[req.req_id] = list(req.block_ids[0])
            start = req.num_computed_tokens
            end = start + scheduler_output.num_scheduled_tokens[req.req_id]
            meta.operations.append((list(self.blocks[req.req_id]), start, end, True))
        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            if req_id in cached.resumed_req_ids:
                raise RuntimeError("LMInfer vLLM backend does not support preemption")
            new_blocks = cached.new_block_ids[i]
            if new_blocks is not None:
                self.blocks[req_id].extend(new_blocks[0])
            start = cached.num_computed_tokens[i]
            end = start + scheduler_output.num_scheduled_tokens[req_id]
            meta.operations.append((list(self.blocks[req_id]), start, end, False))
        return meta

    def request_finished(self, request, block_ids):
        self.blocks.pop(request.request_id, None)
        return False, None

    def start_load_kv(self, forward_context, **kwargs):
        stage = self.bridge.stage
        for blocks, start, end, load in self._get_connector_metadata().operations:
            stage.computed_tokens += end - start
            if not load or stage.prefix is None:
                continue
            if start != stage.prefix_length:
                raise RuntimeError("vLLM external prefix length does not match snapshot")
            loaded = set()
            for name, layer in forward_context.no_compile_layers.items():
                cache = getattr(layer, "kv_cache", None)
                if cache is None:
                    continue
                index = layer_index(name)
                old = stage.prefix.layers[index]
                write_paged(cache, blocks, old.keys, old.values, self.block_size)
                loaded.add(index)
            if loaded != set(range(stage.num_layers)):
                raise RuntimeError("Could not load all vLLM attention layers")
            stage.loaded_tokens += start

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        for blocks, start, end, _ in self._get_connector_metadata().operations:
            keys, values = read_paged(kv_layer, blocks, start, end, self.block_size)
            self.bridge.stage.save(layer_index(layer_name), keys, values, start, end)

    def wait_for_save(self):
        # Copies use the model's CUDA stream.
        pass
