import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, DynamicCache

from lminfer.config import EngineConfig, SamplingParams
from lminfer.context_repair import context_prefill
from lminfer.engine import LLMEngine
from lminfer.kvcache import KVGraft, KVPrefix, SessionKVStore, slice_cache, tail_cache, rebase_rope_cache


class ContextRepairTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        c = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, eos_token_id=96)
        c._attn_implementation = 'sdpa'
        cls.model = Qwen3ForCausalLM(c).eval()
        cls.ids = torch.randint(0, 90, (1, 30))
        with torch.inference_mode():
            cls.full = cls.model(cls.ids, use_cache=True)
            cls.prefix = slice_cache(cls.full.past_key_values, 3, c)
            cls.grafts = []
            for position, length, source_start in [(6, 9, 4), (20, 5, 2)]:
                source = torch.cat([torch.randint(0, 90, (1, source_start)),
                                    cls.ids[:, position:position + length]], dim=1)
                cache = cls.model(source, use_cache=True).past_key_values
                cls.grafts.append(KVGraft(position, cls.ids[0, position:position + length].tolist(),
                                         tail_cache(cache, source_start, c), source_start))

    def options(self, **kw):
        return EngineConfig(model='tiny', repair_mode='context', **kw)

    def test_full_selection_matches_every_layer_and_logits(self):
        snapshots = [(g.cache.layers[2].keys.clone(), g.cache.layers[2].values.clone()) for g in self.grafts]
        out = context_prefill(self.model, self.ids, self.prefix, 3, self.grafts,
                              self.options(repair_budget=1, repair_min_ratio=1))
        self.assertEqual(out.exact_prefix_len, 30)
        for actual, expected in zip(out.cache.layers, self.full.past_key_values.layers):
            torch.testing.assert_close(actual.keys, expected.keys, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(actual.values, expected.values, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(out.logits, self.full.logits[:, -1:], atol=1e-6, rtol=1e-5)
        for g, (k, v) in zip(self.grafts, snapshots):
            self.assertTrue(torch.equal(g.cache.layers[2].keys, k))
            self.assertTrue(torch.equal(g.cache.layers[2].values, v))

    def test_partial_selection_writes_all_shallow_kv_and_preserves_omitted_deep(self):
        out = context_prefill(self.model, self.ids, self.prefix, 3, self.grafts,
                              self.options(repair_budget=0, repair_min_ratio=0, repair_probe_tokens=0))
        self.assertEqual(out.exact_prefix_len, 6)
        for i in (0, 1):
            torch.testing.assert_close(out.cache.layers[i].keys, self.full.past_key_values.layers[i].keys,
                                       atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(out.cache.layers[i].values, self.full.past_key_values.layers[i].values,
                                       atol=1e-6, rtol=1e-5)
        for g in self.grafts:
            old = rebase_rope_cache(g.cache, g.source_position, g.position, self.model.config)
            for i in (2, 3):
                self.assertTrue(torch.equal(out.cache.layers[i].keys[:, :, g.position:g.position + len(g.tokens)],
                                            old.layers[i].keys))
        # Tokens before the first graft cannot be affected by any later graft.
        for i in (2, 3):
            torch.testing.assert_close(out.cache.layers[i].keys[:, :, :6],
                                       self.full.past_key_values.layers[i].keys[:, :, :6], atol=1e-6, rtol=1e-5)

    def test_probe_threshold_restarts_from_exact_prefix(self):
        out = context_prefill(self.model, self.ids, self.prefix, 3, self.grafts,
                              self.options(repair_probe_threshold=1e-10, repair_probe_tokens=2))
        self.assertEqual(out.exact_prefix_len, 30)
        self.assertEqual(out.stats['fallback_reason'], 'deep probe threshold exceeded')
        torch.testing.assert_close(out.logits, self.full.logits[:, -1:], atol=1e-6, rtol=1e-5)

    def test_graft_at_prompt_end_still_produces_final_logits(self):
        ids = self.ids[:, :25]
        out = context_prefill(self.model, ids, self.prefix, 3, self.grafts,
                              self.options(repair_budget=0, repair_min_ratio=0, repair_probe_tokens=0))
        self.assertEqual(out.logits.shape, (1, 1, 97))
        self.assertEqual(out.cache.get_seq_length(), 25)
        self.assertEqual(out.stats['repaired_tokens'], 1)

    def engine(self, options):
        e = LLMEngine.__new__(LLMEngine)
        e.config, e.model = options, self.model
        e._think_ids = None
        e._stats = dict(completed=0, generated_tokens=0, prefill_tokens=0)
        e.tokenizer = SimpleNamespace(decode=lambda ids, **kw: ' '.join(map(str, ids)))
        return e

    def test_engine_and_multiround_lcp_do_not_launder_approximation(self):
        engine = self.engine(self.options(repair_budget=0, repair_min_ratio=0, repair_probe_tokens=0))
        result = engine._generate('first', self.ids, SamplingParams(temperature=0, max_tokens=1), graft=self.grafts)
        self.assertEqual(result.exact_prefix_len, 6)
        tokens = self.ids[0].tolist() + result.output_tokens
        store = SessionKVStore(config=self.model.config)
        store.put('s', 'main', tokens, result.kv_cache, exact_prefix_len=result.exact_prefix_len)
        prefix = store._segments['s']['main']
        extended = torch.tensor([tokens + [11, 12]])
        next_result = engine._generate('next', extended, SamplingParams(temperature=0, max_tokens=1),
                                       reuse_prefixes=[prefix])
        self.assertEqual(next_result.reused_prompt_tokens, 6)
        self.assertEqual(next_result.exact_prefix_len, extended.shape[1] + len(next_result.output_tokens))
        with torch.inference_mode():
            reference = self.model(extended, use_cache=True)
        for layer, expected in zip(next_result.kv_cache.layers, reference.past_key_values.layers):
            torch.testing.assert_close(layer.keys[:, :, :extended.shape[1]], expected.keys, atol=1e-6, rtol=1e-5)

    def test_window_marks_approximate_and_exact_mode_repairs(self):
        for mode, expected in [('window', 6), ('exact', 31)]:
            engine = self.engine(EngineConfig(model='tiny', repair_mode=mode, graft_rope_rebase=True))
            result = engine._generate(mode, self.ids, SamplingParams(temperature=0, max_tokens=1), graft=self.grafts)
            self.assertEqual(result.exact_prefix_len, expected)

    def test_best_prefix_checks_exact_length_before_early_exit(self):
        engine = self.engine(self.options())
        prefixes = [KVPrefix(self.ids[0].tolist(), self.full.past_key_values, exact_prefix_len=2),
                    KVPrefix(self.ids[0, :10].tolist(), slice_cache(self.full.past_key_values, 10, self.model.config))]
        result = engine._generate('lcp', self.ids, SamplingParams(temperature=0, max_tokens=1), reuse_prefixes=prefixes)
        self.assertEqual(result.reused_prompt_tokens, 10)

    def test_deeper_shallow_pass_writes_exact_kv(self):
        out = context_prefill(self.model, self.ids, self.prefix, 3, self.grafts,
                              self.options(repair_shallow_layers=2, repair_probe_tokens=0))
        for i in range(3):
            torch.testing.assert_close(out.cache.layers[i].keys, self.full.past_key_values.layers[i].keys,
                                       atol=1e-6, rtol=1e-5)
        final = context_prefill(self.model, self.ids, self.prefix, 3, self.grafts,
                                self.options(repair_shallow_layers=3, repair_probe_tokens=0))
        self.assertEqual(final.exact_prefix_len, 30)
        torch.testing.assert_close(final.logits, self.full.logits[:, -1:], atol=1e-6, rtol=1e-5)

    def test_sparse_queries_cannot_read_future(self):
        options = self.options(repair_budget=0, repair_min_ratio=0, repair_probe_tokens=0)
        out = context_prefill(self.model, self.ids, self.prefix, 3, self.grafts, options)
        changed = self.ids.clone()
        changed[0, -1] = (changed[0, -1] + 1) % 90
        other = context_prefill(self.model, changed, self.prefix, 3, self.grafts, options)
        for a, b in zip(out.cache.layers, other.cache.layers):
            torch.testing.assert_close(a.keys[:, :, :-1], b.keys[:, :, :-1], atol=1e-6, rtol=1e-5)

    def test_unsupported_rope_uses_exact_fallback(self):
        from unittest.mock import patch
        with patch.dict(self.model.config.rope_parameters, {"rope_type": "linear"}):
            out = context_prefill(self.model, self.ids, self.prefix, 3, self.grafts, self.options())
        self.assertEqual(out.exact_prefix_len, 30)
        self.assertEqual(out.stats['fallback_reason'], 'non-default RoPE is not supported')

    def test_quantized_model_uses_exact_fallback(self):
        from unittest.mock import patch
        with patch.object(self.model, 'is_quantized', True, create=True):
            out = context_prefill(self.model, self.ids, self.prefix, 3, self.grafts, self.options())
        self.assertEqual(out.exact_prefix_len, 30)
        self.assertEqual(out.stats['fallback_reason'], 'quantized models are not supported')

    def test_invalid_config(self):
        for kw in [dict(repair_shallow_layers=0), dict(repair_budget=float('nan')),
                   dict(repair_budget=.01, repair_min_ratio=.1), dict(repair_probe_tokens=-1),
                   dict(repair_probe_threshold=1, repair_probe_tokens=0)]:
            with self.assertRaises(ValueError):
                self.options(**kw)

    def test_invalid_cache_falls_back(self):
        g = self.grafts[0]
        bad = KVGraft(g.position, g.tokens, DynamicCache(config=self.model.config), g.source_position)
        out = context_prefill(self.model, self.ids, self.prefix, 3, [bad], self.options())
        self.assertEqual(out.exact_prefix_len, 30)
        self.assertIn('mismatch', out.stats['fallback_reason'])


if __name__ == '__main__':
    unittest.main()
