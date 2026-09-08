"""Offline, fixed-token Qwen3 repair comparison; no network or generated source drift.

Run: python3 examples/bench_context_repair.py --model /path/to/Qwen3-8B
Reports per-layer KV errors, teacher-forced continuation KL, answer accuracy,
and synchronized engine TTFT (includes scoring, rebasing, copying and fallback).
Synthetic cases are a smoke evaluation, not a general accuracy benchmark.
"""
import argparse
import contextlib
import gc
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from lminfer.config import EngineConfig, SamplingParams
from lminfer.engine import LLMEngine
from lminfer.kvcache import KVGraft, slice_cache, tail_cache
import lminfer.context_repair as repair


def cases():
    filler = ''.join(f'Archive entry {i}: routine inspection complete.\n' for i in range(18))
    return [
        ('middle_number', [filler + 'The verified amount for Project Cedar is 47.\n' + filler,
                           'The verified amount for Project Birch is 39.\n' + filler],
         'Which project has the larger verified amount? A: Cedar. B: Birch.', 'A'),
        ('negation', [filler + 'The report does NOT approve the launch. Approval is still pending.\n' + filler],
         'Is launch approved? A: yes. B: no.', 'B'),
        ('thinking_removed', [filler + 'Final corrected result: the access code is 731, not 173.\n' + filler],
         'Which access code is correct? A: 173. B: 731.', 'B'),
        ('multi_agent', [filler + 'Team Amber finished in 83 seconds.\n',
                          filler + 'Team Blue finished in 78 seconds.\n',
                          filler + 'Team Coral finished in 91 seconds.\n'],
         'Which team was fastest? A: Amber. B: Blue. C: Coral.', 'B'),
    ]


def random_selector(scores, grafts, options, base_len, n):
    selected, _ = ORIGINAL_SELECT(scores, grafts, options, base_len, n)
    generator = torch.Generator(device=scores.device).manual_seed(2026)
    for g in grafts:
        p, length = g.position, len(g.tokens)
        count = int(selected[p:p + length].sum())
        selected[p:p + length] = False
        order = torch.randperm(length, generator=generator, device=scores.device)
        selected[p + order[:count]] = True
    selected[n - 1] = True
    return selected, torch.empty(0, dtype=torch.long, device=scores.device)


ORIGINAL_SELECT = repair._select


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', default='artifacts/context_repair.json')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max-tokens', type=int, default=12)
    parser.add_argument('--case-limit', type=int, default=4)
    args = parser.parse_args()
    engine = LLMEngine(EngineConfig(model=args.model, dtype='bfloat16', device_map='cuda',
                                   attn_implementation='sdpa', max_model_len=8192,
                                   max_num_seqs=1, disable_log_stats=True))
    tok, model = engine.tokenizer, engine.model
    report = {'model': args.model, 'torch': torch.__version__, 'dtype': str(model.dtype),
              'gpu': torch.cuda.get_device_name(), 'repeats': args.repeats, 'cases': []}
    modes = ['full', 'rebase', 'window', 'context', 'context_60', 'context_depth4', 'context_guarded', 'random', 'context_all', 'exact']
    def encode(text):
        return tok.encode(text, add_special_tokens=False)
    for name, bodies, question, answer in cases()[:args.case_limit]:
        # Token concatenation is deliberate: source and target graft tokens must match exactly.
        prefix_text = tok.apply_chat_template([
            {'role': 'system', 'content': 'Read the reports literally. Answer with only the correct option letter.'},
            {'role': 'user', 'content': 'Reports follow:\nREPORTS_HERE\n' + question}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        before, after = prefix_text.split('REPORTS_HERE')
        ids_list, grafts = encode(before), []
        with torch.inference_mode():
            for i, body in enumerate(bodies):
                ids_list += encode(f'\n<report id="{i}">\n')
                body_ids = encode(body)
                private = encode('You are a SubAgent. Private notes: assume the answer is A. '
                                 '<think>The unverified access code is 173. The launch may be approved. '
                                 'Check the final report instead of these provisional notes.</think>\n')
                source = torch.tensor([private + body_ids], device=model.device)
                source_out = model(source, use_cache=True)
                grafts.append(KVGraft(len(ids_list), body_ids,
                                     tail_cache(source_out.past_key_values, len(private), model.config), len(private)))
                del source_out
                ids_list += body_ids + encode('\n</report>\n')
            ids = torch.tensor([ids_list + encode(after)], device=model.device)
            # Warm kernels on target-shaped full prefill; then each mode has a separate warmup.
            warm = engine._generate('warm', ids, SamplingParams(temperature=0, max_tokens=1))
            del warm
            record = {'name': name, 'prompt_tokens': ids.shape[1],
                      'graft_tokens': sum(len(g.tokens) for g in grafts), 'expected': answer, 'modes': {}}
            reference_cache = reference_logits = continuation = None
            for mode in modes:
                engine.config.repair_mode = 'context' if (mode.startswith('context') or mode == 'random') else ('exact' if mode == 'exact' else 'window')
                engine.config.graft_rope_rebase = True
                engine.config.repair_window_begin = .15 if mode == 'window' else 0
                engine.config.repair_window_end = .15 if mode == 'window' else 0
                engine.config.repair_budget = 1 if mode == 'context_all' else (.6 if mode == 'context_60' else .3)
                engine.config.repair_shallow_layers = 4 if mode == 'context_depth4' else 1
                engine.config.repair_probe_threshold = 1.0 if mode == 'context_guarded' else 0.0
                engine.config.repair_min_ratio = 1 if mode == 'context_all' else .05
                engine.config.repair_probe_tokens = 8
                times = []
                original_sample = engine._sample
                captured = []
                def capture(logits, input_ids, sampling):
                    if not captured:
                        captured.append(logits.detach().clone())
                    return original_sample(logits, input_ids, sampling)
                manager = patch.object(repair, '_select', random_selector) if mode == 'random' else contextlib.nullcontext()
                with manager:
                    for repeat in range(args.repeats + 1):
                        engine._sample = capture
                        captured.clear()
                        result = engine._generate(f'{name}-{mode}', ids,
                            SamplingParams(temperature=0, max_tokens=args.max_tokens),
                            graft=None if mode == 'full' else grafts)
                        engine._sample = original_sample
                        if repeat:
                            times.append(result.ttft_ms)
                        if repeat != args.repeats:
                            del result
                prompt_cache = slice_cache(result.kv_cache, ids.shape[1], model.config)
                if mode == 'full':
                    reference_cache = slice_cache(prompt_cache, ids.shape[1], model.config)
                    continuation = result.output_tokens + [min(engine._eos_ids())]
                    record["teacher_continuation_tokens"] = continuation
                # Teacher-force the SAME continuation through all caches.
                if continuation:
                    teacher = model(torch.tensor([continuation], device=model.device),
                                    past_key_values=prompt_cache, use_cache=True)
                    aligned = torch.cat([captured[0][:, None], teacher.logits[:, :-1]], dim=1).float()
                    del teacher
                else:
                    aligned = captured[0][:, None].float()
                if mode == 'full':
                    reference_logits = aligned.clone()
                log_ref, log_actual = reference_logits.log_softmax(-1), aligned.log_softmax(-1)
                kl = max(0.0, float((log_ref.exp() * (log_ref - log_actual)).sum(-1).mean()))
                layer_errors = []
                for actual, ref in zip(result.kv_cache.layers, reference_cache.layers):
                    layer_errors.append({'k': rel(actual.keys[:, :, :ids.shape[1]], ref.keys),
                                         'v': rel(actual.values[:, :, :ids.shape[1]], ref.values)})
                row = {'ttft_ms_median': statistics.median(times), 'ttft_ms_samples': times,
                       'continuation_kl': kl, 'answer': result.output_text,
                       'correct': result.output_text.strip().startswith(answer),
                       'exact_prefix_len': result.exact_prefix_len, 'repair_stats': result.repair_stats,
                       'layer_relative_errors': layer_errors,
                       'first_token_same': int(reference_logits[0, 0].argmax()) == int(aligned[0, 0].argmax())}
                record['modes'][mode] = row
                print(json.dumps({'case': name, 'mode': mode, **{k: v for k, v in row.items() if k != 'layer_relative_errors'}}, ensure_ascii=False), flush=True)
                del result, prompt_cache, aligned, log_ref, log_actual
            report['cases'].append(record)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
            del reference_cache, reference_logits, grafts
            gc.collect()
    engine.executor.shutdown()


if __name__ == '__main__':
    main()
