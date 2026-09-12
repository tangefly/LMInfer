"""Qwen3 dense attention: exact shallow KV, then fixed-position sparse queries.

No model hooks or global patches: request-local tensors are safe across threads.
Unselected deep KV remains approximate. Probe errors are heuristics, not bounds.
"""
from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F
from transformers import DynamicCache

from .kvcache import rebase_rope_cache, slice_cache
from .model_adapters import resolve_logits_kwargs


@dataclass
class RepairOutput:
    logits: torch.Tensor
    cache: DynamicCache
    exact_prefix_len: int
    stats: dict = field(default_factory=dict)


def support_reason(model) -> str | None:
    c = model.config
    if getattr(model, "is_quantized", False):
        return "quantized models are not supported"
    if c.model_type != "qwen3" or c.num_hidden_layers < 2:
        return "context repair requires a dense Qwen3 model with >= 2 layers"
    if any(t != "full_attention" for t in getattr(c, "layer_types", [])):
        return "sliding attention is not supported"
    rope = getattr(c, "rope_parameters", None) or getattr(c, "rope_scaling", None) or {}
    if rope.get("rope_type", rope.get("type", "default")) != "default":
        return "non-default RoPE is not supported"
    if len({p.device for p in model.parameters()}) != 1 or any(
        str(d) in ("disk", "meta") for d in getattr(model, "hf_device_map", {}).values()
    ):
        return "context repair requires a model resident on one device"
    return None


def _relative_error(k, v, old_k, old_v):
    # Max across KV heads retains a change concentrated in a single head.
    def rel(a, b):
        return ((a.float() - b.float()).norm(dim=-1)
                / a.float().norm(dim=-1).clamp_min(1e-6))
    return (rel(k, old_k) + rel(v, old_v)).amax(dim=1)[0]


def _project(layer, h, cos, sin):
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    attn = layer.self_attn
    u = layer.input_layernorm(h)
    shape = (*u.shape[:-1], -1, attn.head_dim)
    q = attn.q_norm(attn.q_proj(u).view(shape)).transpose(1, 2)
    k = attn.k_norm(attn.k_proj(u).view(shape)).transpose(1, 2)
    v = attn.v_proj(u).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    return q, k, v


def _attention(q, k, v, positions):
    # Explicit original-position mask: sparse queries are never renumbered.
    if q.shape[-2] == k.shape[-2]:
        # Same causal SDPA kernel as standard full prefill (also limits BF16 drift).
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                             is_causal=True, enable_gqa=True)
        return out.transpose(1, 2).contiguous().flatten(2)
    # Match Transformers' masked SDPA path: native GQA + mask can force math kernels.
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    groups = q.shape[1] // k.shape[1]
    k, v = repeat_kv(k, groups), repeat_kv(v, groups)
    keys = torch.arange(k.shape[-2], device=q.device)
    pieces = []
    for start in range(0, q.shape[-2], 128):
        mask = keys[None, :] <= positions[start:start + 128, None]
        pieces.append(F.scaled_dot_product_attention(
            q[:, :, start:start + 128], k, v,
            attn_mask=mask[None, None], dropout_p=0.0))
    return torch.cat(pieces, dim=-2).transpose(1, 2).contiguous().flatten(2)


def _importance(q, k, positions):
    # Bounded sampling of target queries, including the final MainAgent query.
    # Max over heads/queries avoids averaging away an important minority head.
    count = min(32, positions.numel())
    idx = torch.linspace(0, positions.numel() - 1, count, device=q.device).long().unique()
    key_pos = torch.arange(k.shape[-2], device=q.device)
    groups = q.shape[1] // k.shape[1]
    scores = torch.zeros(k.shape[-2], device=q.device)
    for head in range(q.shape[1]):
        logits = q[0, head, idx].float() @ k[0, head // groups].float().T
        logits *= q.shape[-1] ** -0.5
        logits.masked_fill_(key_pos[None] > positions[idx, None], -torch.inf)
        scores = torch.maximum(scores, logits.softmax(-1).amax(0))
    return scores


def _select(scores, grafts, options, base_len, n):
    selected = torch.ones(n, dtype=torch.bool, device=scores.device)
    selected[:base_len] = False
    for graft in grafts:
        p, length = graft.position, len(graft.tokens)
        selected[p:p + length] = False
        local = scores[p:p + length]
        order = torch.argsort(local, descending=True, stable=True)
        cap = math.ceil(length * options.repair_budget)
        minimum = math.ceil(length * options.repair_min_ratio)
        total = local.sum()
        count = minimum
        if total > 0:
            count = max(minimum, int(torch.searchsorted(
                local[order].cumsum(0), total * options.repair_coverage).item()) + 1)
        count = min(length, cap, count)
        selected[p + order[:count]] = True
    selected[n - 1] = True  # final logits always computed
    candidates = (~selected).nonzero().flatten()
    candidates = candidates[candidates >= base_len]
    count = min(options.repair_probe_tokens, candidates.numel())
    probes = candidates[torch.linspace(0, candidates.numel() - 1, count,
                                      device=scores.device).long()] if count else candidates[:0]
    selected[probes] = True
    return selected, probes


@torch.inference_mode()
def exact_prefill(model, input_ids, prefix, base_len, reason=None):
    cache = slice_cache(prefix, base_len, model.config) if base_len else DynamicCache(config=model.config)
    # 只要末尾 logits: 整段重算(以及所有 MoE/非 dense Qwen3 的回退路径)都走这里,
    # 大词表模型长 prompt 下全位置 logits 是 GiB 级的临时张量(见 resolve_logits_kwargs)
    out = model(input_ids=input_ids[:, base_len:], past_key_values=cache,
                attention_mask=torch.ones_like(input_ids), use_cache=True,
                **resolve_logits_kwargs(model))
    return RepairOutput(out.logits[:, -1:], cache, input_ids.shape[1],
                        {"fallback_reason": reason, "exact": True})


@torch.inference_mode()
def context_prefill(model, input_ids, prefix, base_len, grafts, options):
    reason = support_reason(model)
    if reason:
        return exact_prefill(model, input_ids, prefix, base_len, reason)
    c, n = model.config, input_ids.shape[1]
    selection_layer = options.repair_shallow_layers
    if selection_layer >= c.num_hidden_layers:
        return exact_prefill(model, input_ids, prefix, base_len, "shallow depth covers full model")
    device = input_ids.device
    # Validate caches before using any tensor; the caller validates token/position matching.
    for g in grafts:
        if len(g.cache.layers) != c.num_hidden_layers:
            return exact_prefill(model, input_ids, prefix, base_len, "graft layer count mismatch")
        for layer in g.cache.layers:
            expected = (1, c.num_key_value_heads, len(g.tokens), c.head_dim)
            if (not layer.is_initialized or tuple(layer.keys.shape) != expected
                    or tuple(layer.values.shape) != expected):
                return exact_prefill(model, input_ids, prefix, base_len, "graft KV shape mismatch")
    rebased = [rebase_rope_cache(g.cache, g.source_position, g.position, c) for g in grafts]
    positions = torch.arange(base_len, n, device=device)
    h = model.model.embed_tokens(input_ids[:, base_len:])
    full_positions = positions.clone()
    cos, sin = model.model.rotary_emb(h, positions[None])
    cache = DynamicCache(config=c)
    selected = None
    probes = positions[:0]
    probe_max = 0.0
    stats = {}
    for index, layer in enumerate(model.model.layers):
        q, fresh_k, fresh_v = _project(layer, h, cos, sin)
        shape = (1, fresh_k.shape[1], n, fresh_k.shape[-1])
        k, v = fresh_k.new_zeros(shape), fresh_v.new_zeros(shape)
        if base_len:
            k[:, :, :base_len] = prefix.layers[index].keys[:, :, :base_len]
            v[:, :, :base_len] = prefix.layers[index].values[:, :, :base_len]
        for g, old in zip(grafts, rebased):
            k[:, :, g.position:g.position + len(g.tokens)] = old.layers[index].keys
            v[:, :, g.position:g.position + len(g.tokens)] = old.layers[index].values
        if index == selection_layer:
            errors = torch.zeros(n, device=device)
            for g in grafts:
                sl = slice(g.position, g.position + len(g.tokens))
                fresh = slice(g.position - base_len, g.position + len(g.tokens) - base_len)
                errors[sl] = _relative_error(fresh_k[:, :, fresh], fresh_v[:, :, fresh],
                                              k[:, :, sl], v[:, :, sl])
            scores = errors * (0.05 + importance)
            if not torch.isfinite(scores).all():
                return exact_prefill(model, input_ids, prefix, base_len, "nonfinite shallow score")
            selected, probes = _select(scores, grafts, options, base_len, n)
            stats["shallow_error_max"] = float(errors.max())
        elif index > selection_layer and probes.numel():
            probe_idx = torch.searchsorted(positions, probes)
            error = _relative_error(fresh_k[:, :, probe_idx], fresh_v[:, :, probe_idx],
                                    k[:, :, probes], v[:, :, probes])
            probe_max = max(probe_max, float(error.max()))
            if not torch.isfinite(error).all() or (
                options.repair_probe_threshold > 0 and probe_max > options.repair_probe_threshold
            ):
                return exact_prefill(model, input_ids, prefix, base_len, "deep probe threshold exceeded")
        # Write ALL fresh selection-layer KV before narrowing queries/hidden states.
        k[:, :, positions] = fresh_k
        v[:, :, positions] = fresh_v
        cache.update(k, v, index)
        if index == selection_layer - 1:
            importance = _importance(q, k, positions)
        if index == selection_layer:
            keep = selected[full_positions]
            h, q = h[:, keep], q[:, :, keep]
            positions = full_positions[keep]
            cos, sin = cos[:, keep], sin[:, keep]
        h = h + layer.self_attn.o_proj(_attention(q, k, v, positions))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    logits = model.lm_head(model.model.norm(h[:, -1:]))
    omitted = ((~selected) & (torch.arange(n, device=device) >= base_len)).nonzero().flatten()
    # If selection is at the final layer, every cached KV and the final query are exact.
    exact_len = int(omitted[0]) if omitted.numel() and c.num_hidden_layers > selection_layer + 1 else n
    graft_count = sum(len(g.tokens) for g in grafts)
    repaired = sum(int(selected[g.position:g.position + len(g.tokens)].sum()) for g in grafts)
    stats.update(shallow_layers=selection_layer, graft_tokens=graft_count, repaired_tokens=repaired,
                 probe_tokens=probes.numel(), probe_error_max=probe_max,
                 active_deep_tokens=positions.numel(), exact=exact_len == n,
                 projected_token_layers=(selection_layer + 1) * (n - base_len) + (c.num_hidden_layers - selection_layer - 1) * positions.numel(),
                 full_token_layers=c.num_hidden_layers * (n - base_len))
    return RepairOutput(logits, cache, exact_len, stats)
