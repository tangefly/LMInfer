"""Visualize attention maps for standard prefill vs sub-agent KV grafting.

This script builds a small offline MainAgent -> SubAgent -> MainAgent trace with
transformers only.  It compares three final-MainAgent prefill/cache modes:

1. standard_prefill: full prompt prefill in the MainAgent context.
2. direct_concat: prefill before the tool response, directly splice SubAgent
   output KV into the MainAgent cache, then prefill the suffix.
3. rebase_recompute: recompute edge tokens around the graft in MainAgent
   context, RoPE-rebase and splice the middle of the SubAgent output KV, then
   prefill the suffix.

Each selected layer saves one heatmap per attention head.  Rows skipped by a KV
splice are left blank in the image because those query attentions were never
computed in the MainAgent context.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from lminfer.kvcache import concat_cache, rebase_rope_cache, slice_cache, tail_cache


@dataclass
class RecordedAttention:
    row_start: int
    attn: torch.Tensor  # [heads, q_len, k_len], CPU float32


@dataclass
class RunResult:
    mode: str
    output_text: str
    prompt_len: int
    generated_ids: list[int]
    graft_start: int
    graft_end: int
    records: dict[int, list[RecordedAttention]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw attention heatmaps for MainAgent -> SubAgent -> MainAgent KV grafting."
    )
    parser.add_argument("--model", required=True, help="HF model id or local model path.")
    parser.add_argument("--output-dir", default="attention_maps")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sub-max-new-tokens", type=int, default=96)
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--layers", default=None, help="Comma-separated layer ids. Overrides --layer-stride.")
    parser.add_argument("--repair-window", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--document",
        default=(
            "Alias - the Bad Man stars Ken Maynard as Ken Neville. "
            "Ken Neville gains Rance Collins's trust by posing as an outlaw and helping the gang. "
            "He hides his identity because he is secretly working to expose the criminals. "
            "The second film has a darker ending for the villain, while Alias - the Bad Man resolves "
            "with the hero restoring order. In both stories, the female lead first distrusts the hero "
            "and later recognizes his integrity."
        ),
    )
    parser.add_argument(
        "--question",
        default="Who played Ken Neville, and why did he hide his identity?",
    )
    return parser.parse_args()


def dtype_from_name(name: str):
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def choose_layers(num_layers: int, stride: int, explicit: str | None) -> list[int]:
    if explicit:
        layers = sorted({int(x) for x in explicit.split(",") if x.strip()})
        bad = [x for x in layers if x < 0 or x >= num_layers]
        if bad:
            raise ValueError(f"Layer ids out of range 0..{num_layers - 1}: {bad}")
        return layers
    stride = max(1, stride)
    layers = list(range(0, num_layers, stride))
    if (num_layers - 1) not in layers:
        layers.append(num_layers - 1)
    return layers


def apply_chat_template(tokenizer, messages: list[dict], *, tools=None, enable_thinking=False) -> list[int]:
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
    }
    if tools is not None:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    # Qwen3 accepts enable_thinking. Other templates may reject it.
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return list(ids)


def simple_tools() -> list[dict]:
    return [{
        "type": "function",
        "function": {
            "name": "SubAgent",
            "description": "Ask a SubAgent to answer one question from a document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "document": {"type": "string"},
                },
                "required": ["question", "document"],
            },
        },
    }]


def sample_next(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    logits = logits.float()
    if temperature == 0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        keep = torch.cumsum(sorted_probs, dim=-1) <= top_p
        keep[..., 0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, sorted_idx[keep], sorted_probs[keep])
        probs = filtered / filtered.sum()
    return int(torch.multinomial(probs, 1).item())


def eos_ids(model, tokenizer) -> set[int]:
    ids = set()
    for value in (model.config.eos_token_id, model.generation_config.eos_token_id, tokenizer.eos_token_id):
        if isinstance(value, (list, tuple)):
            ids.update(x for x in value if x is not None)
        elif value is not None:
            ids.add(value)
    return ids


def cache_len(cache: DynamicCache) -> int:
    return int(cache.get_seq_length())


def forward_record(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: DynamicCache | None,
    selected_layers: Iterable[int],
    records: dict[int, list[RecordedAttention]],
    row_start: int,
    *,
    use_cache: bool = True,
):
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=use_cache,
        output_attentions=True,
        return_dict=True,
    )
    for layer_id in selected_layers:
        attn = out.attentions[layer_id]
        # [batch=1, heads, q_len, k_len] -> [heads, q_len, k_len]
        records.setdefault(layer_id, []).append(
            RecordedAttention(row_start=row_start, attn=attn[0].detach().float().cpu())
        )
    return out


def generate_with_records(
    model,
    tokenizer,
    prompt_ids: list[int],
    selected_layers: list[int],
    *,
    mode: str,
    graft_tokens: list[int] | None = None,
    graft_cache: DynamicCache | None = None,
    graft_source_start: int = 0,
    graft_start: int = -1,
    repair_window: int = 0,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> RunResult:
    device = model.device
    prompt = torch.tensor([prompt_ids], device=device)
    prompt_len = prompt.shape[1]
    records: dict[int, list[RecordedAttention]] = {}
    generated: list[int] = []
    cache = DynamicCache(config=model.config)

    def prefill_span(start: int, end: int):
        nonlocal cache
        if end <= start:
            return None
        ids = prompt[:, start:end]
        mask = torch.ones(1, end, device=device)
        row_start = start
        return forward_record(model, ids, mask, cache, selected_layers, records, row_start)

    with torch.inference_mode():
        if mode == "standard_prefill":
            out = forward_record(
                model,
                prompt,
                torch.ones_like(prompt, device=device),
                cache,
                selected_layers,
                records,
                0,
            )
        else:
            if graft_tokens is None or graft_cache is None or graft_start < 0:
                raise ValueError(f"{mode} requires graft_tokens/graft_cache/graft_start")
            graft_len = len(graft_tokens)
            if prompt_ids[graft_start:graft_start + graft_len] != graft_tokens:
                raise ValueError("Graft tokens do not match the final MainAgent prompt.")

            if mode == "direct_concat":
                out = prefill_span(0, graft_start)
                cache = concat_cache(cache, graft_cache, model.config)
                out = prefill_span(graft_start + graft_len, prompt_len) or out
            elif mode == "rebase_recompute":
                rw = max(0, repair_window)
                left = min(rw, graft_len)
                right = min(rw, max(0, graft_len - left))
                middle_start = graft_start + left
                middle_end = graft_start + graft_len - right
                out = prefill_span(0, middle_start)
                if middle_end > middle_start:
                    source_start = graft_source_start + left
                    middle_cache = slice_cache(
                        tail_cache(graft_cache, left, model.config),
                        middle_end - middle_start,
                        model.config,
                    )
                    middle_cache = rebase_rope_cache(
                        middle_cache, source_start, middle_start, model.config
                    )
                    cache = concat_cache(cache, middle_cache, model.config)
                out = prefill_span(middle_end, prompt_len) or out
            else:
                raise ValueError(f"Unknown mode: {mode}")

        token = sample_next(out.logits[0, -1], temperature, top_p)
        stop_ids = eos_ids(model, tokenizer)
        if token not in stop_ids:
            generated.append(token)

        while generated and len(generated) < max_new_tokens:
            cur = torch.tensor([[generated[-1]]], device=device)
            row_start = prompt_len + len(generated) - 1
            out = forward_record(
                model,
                cur,
                torch.ones(1, cache_len(cache) + 1, device=device),
                cache,
                selected_layers,
                records,
                row_start,
            )
            token = sample_next(out.logits[0, -1], temperature, top_p)
            if token in stop_ids:
                break
            generated.append(token)

        # Record the attention row for the final generated token. The generated
        # text itself is unchanged; this only completes the visible map rows.
        target_len = prompt_len + len(generated)
        if generated and cache_len(cache) < target_len:
            cur = torch.tensor([[generated[-1]]], device=device)
            forward_record(
                model,
                cur,
                torch.ones(1, cache_len(cache) + 1, device=device),
                cache,
                selected_layers,
                records,
                prompt_len + len(generated) - 1,
            )

    return RunResult(
        mode=mode,
        output_text=tokenizer.decode(generated, skip_special_tokens=True),
        prompt_len=prompt_len,
        generated_ids=generated,
        graft_start=graft_start,
        graft_end=graft_start + (len(graft_tokens) if graft_tokens is not None else 0),
        records=records,
    )


def generate_plain(model, tokenizer, prompt_ids: list[int], max_new_tokens: int, temperature: float, top_p: float):
    device = model.device
    prompt = torch.tensor([prompt_ids], device=device)
    cache = DynamicCache(config=model.config)
    generated: list[int] = []
    with torch.inference_mode():
        out = model(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt, device=device),
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        token = sample_next(out.logits[0, -1], temperature, top_p)
        stop_ids = eos_ids(model, tokenizer)
        if token not in stop_ids:
            generated.append(token)
        while generated and len(generated) < max_new_tokens:
            cur = torch.tensor([[generated[-1]]], device=device)
            out = model(
                input_ids=cur,
                attention_mask=torch.ones(1, cache_len(cache) + 1, device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            token = sample_next(out.logits[0, -1], temperature, top_p)
            if token in stop_ids:
                break
            generated.append(token)
        if generated and cache_len(cache) < prompt.shape[1] + len(generated):
            model(
                input_ids=torch.tensor([[generated[-1]]], device=device),
                attention_mask=torch.ones(1, cache_len(cache) + 1, device=device),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
    return tokenizer.decode(generated, skip_special_tokens=True), prompt_ids + generated, cache, len(prompt_ids)


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    first = needle[0]
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i] == first and haystack[i:i + len(needle)] == needle:
            return i
    return -1


def clipped_token(tokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text if len(text) <= 14 else text[:13] + "…"


def build_matrix(records: list[RecordedAttention], head: int, total_len: int) -> np.ndarray:
    mat = np.full((total_len, total_len), np.nan, dtype=np.float32)
    for rec in records:
        data = rec.attn[head].numpy()
        q_len, k_len = data.shape
        row_end = min(total_len, rec.row_start + q_len)
        mat[rec.row_start:row_end, :min(total_len, k_len)] = data[:row_end - rec.row_start, :min(total_len, k_len)]
    return mat


def plot_head(
    matrix: np.ndarray,
    tokens: list[str],
    path: Path,
    *,
    mode: str,
    layer: int,
    head: int,
    prompt_len: int,
    graft_start: int,
    graft_end: int,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(7.0, min(18.0, matrix.shape[0] / 10.0))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#eeeeee")
    im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_title(f"{mode} | layer {layer} head {head}")
    ax.set_xlabel("Key tokens")
    ax.set_ylabel("Query tokens")
    for pos, color, label in [
        (graft_start, "white", "graft start"),
        (graft_end, "white", "graft end"),
        (prompt_len, "red", "generation start"),
    ]:
        if 0 <= pos < matrix.shape[0]:
            ax.axvline(pos - 0.5, color=color, linewidth=0.8)
            ax.axhline(pos - 0.5, color=color, linewidth=0.8)
            ax.text(pos, 0, label, color=color, fontsize=7, rotation=90, va="top")
    if matrix.shape[0] <= 80:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=5)
        ax.set_yticklabels(tokens, fontsize=5)
    else:
        step = max(1, math.ceil(matrix.shape[0] / 32))
        ticks = list(range(0, matrix.shape[0], step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_maps(tokenizer, result: RunResult, out_dir: Path, selected_layers: list[int], final_ids: list[int]):
    total_len = result.prompt_len + len(result.generated_ids)
    tokens = [clipped_token(tokenizer, tid) for tid in final_ids[:total_len]]
    for layer in selected_layers:
        records = result.records[layer]
        num_heads = records[0].attn.shape[0]
        for head in range(num_heads):
            matrix = build_matrix(records, head, total_len)
            path = out_dir / result.mode / f"layer_{layer:02d}" / f"head_{head:02d}.png"
            plot_head(
                matrix,
                tokens,
                path,
                mode=result.mode,
                layer=layer,
                head=head,
                prompt_len=result.prompt_len,
                graft_start=result.graft_start,
                graft_end=result.graft_end,
            )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_name(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    )
    model.eval()
    model.requires_grad_(False)

    selected_layers = choose_layers(model.config.num_hidden_layers, args.layer_stride, args.layers)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_system = (
        "You are the MainAgent. Delegate document lookup to SubAgent, then use the returned "
        "tool result to answer concisely."
    )
    sub_system = "You are the SubAgent. Answer only from the provided document, concisely."
    tool_call = {
        "id": "call_subagent_1",
        "type": "function",
        "function": {
            "name": "SubAgent",
            "arguments": json.dumps(
                {"question": args.question, "document": args.document},
                ensure_ascii=False,
            ),
        },
    }
    main_first_messages = [
        {"role": "system", "content": main_system},
        {"role": "user", "content": args.question},
    ]
    sub_messages = [
        {"role": "system", "content": sub_system},
        {"role": "user", "content": f"Document:\n{args.document}\n\nQuestion:\n{args.question}"},
    ]
    sub_prompt_ids = apply_chat_template(tokenizer, sub_messages, enable_thinking=False)
    sub_text, sub_full_ids, sub_cache, sub_prompt_len = generate_plain(
        model,
        tokenizer,
        sub_prompt_ids,
        args.sub_max_new_tokens,
        args.temperature,
        args.top_p,
    )
    sub_output_ids = sub_full_ids[sub_prompt_len:]

    final_main_messages = [
        *main_first_messages,
        {"role": "assistant", "content": None, "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": tool_call["id"], "content": sub_text},
    ]
    final_prompt_ids = apply_chat_template(
        tokenizer,
        final_main_messages,
        tools=simple_tools(),
        enable_thinking=False,
    )

    graft_start = find_subsequence(final_prompt_ids, sub_output_ids)
    graft_tokens = sub_output_ids
    graft_source_start = sub_prompt_len
    graft_cache = tail_cache(sub_cache, sub_prompt_len, model.config)
    if graft_start < 0:
        # Boundary tokenization around <tool_response> may merge a token at the
        # start/end. Use the longest inner slice that appears in the final prompt.
        best = (-1, 0, 0)
        for left in range(0, min(4, len(sub_output_ids)) + 1):
            for right in range(0, min(4, len(sub_output_ids) - left) + 1):
                candidate = sub_output_ids[left:len(sub_output_ids) - right if right else len(sub_output_ids)]
                pos = find_subsequence(final_prompt_ids, candidate)
                if pos >= 0 and len(candidate) > best[2]:
                    best = (pos, left, len(candidate))
        if best[0] < 0:
            raise RuntimeError("Could not locate the SubAgent output tokens inside the final MainAgent prompt.")
        graft_start, left, length = best
        graft_tokens = sub_output_ids[left:left + length]
        graft_source_start = sub_prompt_len + left
        graft_cache = slice_cache(tail_cache(sub_cache, graft_source_start, model.config), length, model.config)

    results = []
    for mode in ("standard_prefill", "direct_concat", "rebase_recompute"):
        result = generate_with_records(
            model,
            tokenizer,
            final_prompt_ids,
            selected_layers,
            mode=mode,
            graft_tokens=graft_tokens,
            graft_cache=graft_cache,
            graft_source_start=graft_source_start,
            graft_start=graft_start,
            repair_window=args.repair_window,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        results.append(result)
        final_ids = final_prompt_ids + result.generated_ids
        save_maps(tokenizer, result, out_dir, selected_layers, final_ids)

    metadata = {
        "model": args.model,
        "selected_layers": selected_layers,
        "repair_window": args.repair_window,
        "question": args.question,
        "subagent_output": sub_text,
        "prompt_tokens": len(final_prompt_ids),
        "graft_start": graft_start,
        "graft_end": graft_start + len(graft_tokens),
        "graft_tokens": len(graft_tokens),
        "modes": {
            r.mode: {
                "generated_tokens": len(r.generated_ids),
                "output_text": r.output_text,
            }
            for r in results
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
