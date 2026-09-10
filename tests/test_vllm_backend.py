import unittest
import torch
from transformers import DynamicCache, Qwen3Config

from lminfer.config import EngineConfig
from lminfer.kvcache import KVGraft, KVPrefix
from lminfer.vllm_bridge import Stage, layer_index, read_paged, write_paged
from lminfer.vllm_plan import plan_prefill


def config():
    return Qwen3Config(num_hidden_layers=2, num_attention_heads=2,
                       num_key_value_heads=2, head_dim=4, hidden_size=8)


def cache(length):
    c = config()
    return DynamicCache(ddp_cache_data=[
        (torch.randn(1, 2, length, 4), torch.randn(1, 2, length, 4))
        for _ in range(2)], config=c)


class ConsumedKVTests(unittest.TestCase):
    def engine(self):
        from types import SimpleNamespace
        from lminfer.vllm_engine import VLLMEngine
        engine = VLLMEngine.__new__(VLLMEngine)
        engine.config = EngineConfig(model="test", graft_rope_rebase=True)
        engine.model = SimpleNamespace(config=config(), dtype=torch.float32)
        return engine

    def inputs(self):
        import weakref
        tokens = list(range(30))
        prefixes = [KVPrefix(tokens[:3], cache(3)),
                    KVPrefix([100] * 7, cache(7), kind="sub")]
        grafts = [KVGraft(5, tokens[5:12], cache(7), 20),
                  KVGraft(17, tokens[17:24], cache(7), 40)]
        refs = [weakref.ref(p.cache.layers[0].keys) for p in prefixes]
        graft_refs = [weakref.ref(g.cache.layers[0].keys) for g in grafts]
        return tokens, prefixes, grafts, refs, graft_refs

    def fake_vllm(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        return patch.dict("sys.modules", {"vllm": SimpleNamespace(SamplingParams=SimpleNamespace)})

    def test_sources_die_before_next_stage_and_final_decode(self):
        engine = self.engine()
        tokens, prefixes, grafts, refs, graft_refs = self.inputs()
        seen = []

        def stage(prompt, prefix, sampling):
            self.assertTrue(all(ref() is None for ref in refs))
            self.assertEqual(prefixes, [])
            self.assertEqual(grafts, [])
            if len(prompt) == 5:
                self.assertTrue(all(ref() is not None for ref in graft_refs))
            else:
                self.assertEqual(len(prompt), 17)
                self.assertIsNone(graft_refs[0]())
                self.assertIsNotNone(graft_refs[1]())
            seen.append(len(prompt))
            return None, cache(len(prompt)), {"computed_tokens": len(prompt)-prefix.get_seq_length(),
                                             "loaded_tokens": prefix.get_seq_length()}
        engine._stage = stage
        with self.fake_vllm():
            target, info = engine._prefill_reuse(tokens, prefixes, grafts, True)
        self.assertEqual(seen, [5, 17])
        self.assertTrue(all(ref() is None for ref in graft_refs))
        self.assertEqual(target.get_seq_length(), 24)
        self.assertEqual(info["reused_tokens"], 17)
        self.assertEqual(info["graft_tokens"], 14)
        self.assertEqual(info["exact_prefix_length"], 5)

    def test_error_releases_pending_sources(self):
        engine = self.engine()
        tokens, prefixes, grafts, refs, graft_refs = self.inputs()

        def fail(*args):
            raise RuntimeError("simulated inference failure")
        engine._stage = fail
        with self.fake_vllm(), self.assertRaisesRegex(RuntimeError, "simulated"):
            engine._prefill_reuse(tokens, prefixes, grafts, True)
        self.assertEqual(prefixes, [])
        self.assertEqual(grafts, [])
        self.assertTrue(all(ref() is None for ref in refs + graft_refs))

    def test_exact_mode_releases_unused_grafts(self):
        engine = self.engine()
        engine.config.repair_mode = "exact"
        tokens, prefixes, grafts, refs, graft_refs = self.inputs()
        with self.fake_vllm():
            target, info = engine._prefill_reuse(tokens, prefixes, grafts, True)
        self.assertEqual(target.get_seq_length(), 3)
        self.assertEqual(info["graft_tokens"], 0)
        self.assertTrue(all(ref() is None for ref in refs + graft_refs))

    def test_direct_call_keeps_caller_caches(self):
        engine = self.engine()
        tokens, prefixes, grafts, refs, graft_refs = self.inputs()
        engine._stage = lambda prompt, prefix, sampling: (
            None, cache(len(prompt)), {"computed_tokens": len(prompt)-prefix.get_seq_length(),
                                      "loaded_tokens": prefix.get_seq_length()})
        with self.fake_vllm():
            engine._prefill_reuse(tokens, prefixes, grafts, False)
        self.assertEqual(len(prefixes), 2)
        self.assertEqual(len(grafts), 2)
        self.assertTrue(all(ref() is not None for ref in refs + graft_refs))


class PlanTests(unittest.TestCase):
    def test_multiple_spans_and_partial_windows(self):
        tokens = list(range(60))
        grafts = [KVGraft(10, tokens[10:30], cache(20)),
                  KVGraft(35, tokens[35:55], cache(20))]
        prefix = KVPrefix(tokens[:8], cache(8))
        plan = plan_prefill(tokens, [prefix], grafts, begin=.15, end=.2)
        self.assertEqual([(s.start, s.end) for s in plan.spans], [(13, 26), (38, 51)])
        self.assertEqual(plan.reused_tokens, 34)
        self.assertEqual(plan.exact_prefix_length, 13)

    def test_bad_graft_invalidates_whole_plan_and_falls_back(self):
        tokens = list(range(30))
        prefix = KVPrefix(tokens[:10], cache(10), exact_prefix_len=7)
        bad_cases = [
            [KVGraft(5, [99] * 4, cache(4))],
            [KVGraft(5, tokens[5:15], cache(10)), KVGraft(10, tokens[10:20], cache(10))],
            [KVGraft(5, tokens[5:10], cache(4))],
            [KVGraft(0, tokens[:5], cache(5))],
            [KVGraft(25, tokens[25:] + [30], cache(6))],
        ]
        for grafts in bad_cases:
            plan = plan_prefill(tokens, [prefix], grafts)
            self.assertTrue(plan.mismatch)
            self.assertEqual(plan.spans, [])
            self.assertEqual(plan.prefix_length, 7)

    def test_final_query_never_skipped(self):
        tokens = list(range(20))
        plan = plan_prefill(tokens, [], [KVGraft(5, tokens[5:], cache(15))])
        self.assertEqual(plan.spans[0].end, 19)
        plan = plan_prefill(tokens, [KVPrefix(tokens, cache(20))])
        self.assertEqual(plan.prefix_length, 19)

    def test_full_window_and_exact_mode(self):
        tokens = list(range(20))
        graft = KVGraft(5, tokens[5:15], cache(10))
        for kwargs in ({"begin": .8, "end": .7}, {"exact": True}):
            plan = plan_prefill(tokens, [], [graft], **kwargs)
            self.assertEqual(plan.reused_tokens, 0)
            self.assertIsNone(plan.exact_prefix_length)

    def test_approximate_prefix_is_capped(self):
        tokens = list(range(20))
        prefix = KVPrefix(tokens, cache(20), exact_prefix_len=4)
        self.assertEqual(plan_prefill(tokens, [prefix]).prefix_length, 4)

    def test_backend_constraints(self):
        for kwargs in ({"tensor_parallel_size": 2}, {"repair_mode": "context"},
                       {"dtype": "float32"}, {"gpu_memory_utilization": float("nan")}):
            with self.assertRaises(ValueError):
                EngineConfig(model="test", backend="vllm", **kwargs)


class PagedCacheTests(unittest.TestCase):
    def test_noncontiguous_blocks_unaligned_span_and_no_alias(self):
        for noncontiguous in (False, True):
            paged = torch.full((5, 2, 16, 2, 4), -1.)
            if noncontiguous:
                paged = paged.transpose(2, 3).contiguous().transpose(2, 3)
            keys, values = torch.randn(1, 2, 37, 4), torch.randn(1, 2, 37, 4)
            blocks = [3, 0, 4]
            write_paged(paged, blocks, keys, values, 16)
            k, v = read_paged(paged, blocks, 7, 35, 16)
            torch.testing.assert_close(k, keys[:, :, 7:35])
            torch.testing.assert_close(v, values[:, :, 7:35])
            self.assertTrue(torch.all(paged[1] == -1))
            paged.zero_()
            torch.testing.assert_close(k, keys[:, :, 7:35])

    def test_layout_and_capacity_fail_closed(self):
        with self.assertRaises(ValueError):
            write_paged(torch.zeros(2, 5, 16, 2, 4), [0], torch.zeros(1, 2, 3, 4),
                        torch.zeros(1, 2, 3, 4), 16)
        with self.assertRaises(ValueError):
            read_paged(torch.zeros(5, 2, 16, 2, 4), [0], 0, 17, 16)
        self.assertEqual(layer_index("model.layers.12.self_attn.attn"), 12)

    def test_snapshot_prefix_chunked_capture_and_ownership(self):
        prefix = cache(7)
        stage = Stage(list(range(23)), prefix, 25, 2)
        for start, end in [(7, 16), (16, 23)]:
            for i in range(2):
                stage.save(i, torch.full((1, 2, end-start, 4), float(start)),
                           torch.zeros(1, 2, end-start, 4), start, end)
        result = stage.finish(config())
        self.assertEqual(result.get_seq_length(), 23)
        torch.testing.assert_close(result.layers[0].keys[:, :, :7], prefix.layers[0].keys)
        prefix.layers[0].keys.zero_()
        stage.layers[0][0].zero_()
        self.assertTrue(torch.all(result.layers[0].keys[:, :, 7:16] == 7))
        with self.assertRaises(RuntimeError):
            stage.save(0, torch.zeros(1, 2, 1, 4), torch.zeros(1, 2, 1, 4), 22, 23)

    def test_missing_layer_cannot_be_saved(self):
        stage = Stage([1], None, 2, 2)
        stage.save(0, torch.ones(1, 2, 1, 4), torch.ones(1, 2, 1, 4), 0, 1)
        with self.assertRaises(RuntimeError):
            stage.finish(config())
