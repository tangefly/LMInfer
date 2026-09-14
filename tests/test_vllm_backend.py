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


def glm_config(num_hidden_layers=2):
    from transformers import Glm4MoeLiteConfig
    return Glm4MoeLiteConfig(num_hidden_layers=num_hidden_layers, hidden_size=16,
                             num_attention_heads=2, num_key_value_heads=2,
                             kv_lora_rank=8, qk_rope_head_dim=4)


def mla_cache(length, layers=2, latent=8, rope=4):
    """MLA 快照: 位置无关的潜向量在 keys 槽, 带位置的 k_rot 在 values 槽(不等宽)."""
    return DynamicCache(ddp_cache_data=[
        (torch.randn(1, 1, length, latent), torch.randn(1, 1, length, rope))
        for _ in range(layers)], config=glm_config(layers))


class PackedPagedCacheTests(unittest.TestCase):
    """0.27+ 的 FLASH_ATTN 布局: (blocks, heads, block_size, 2*head_dim), K/V 在尾维.

    与 `PagedCacheTests` 的 5 维 legacy 布局并存 —— 两种 vLLM 都要能跑。
    """

    def test_kv_split_on_the_content_dim(self):
        paged = torch.full((5, 2, 16, 8), -1.)
        keys, values = torch.randn(1, 2, 37, 4), torch.randn(1, 2, 37, 4)
        blocks = [3, 0, 4]
        write_paged(paged, blocks, keys, values, 16)
        k, v = read_paged(paged, blocks, 7, 35, 16)
        torch.testing.assert_close(k, keys[:, :, 7:35])
        torch.testing.assert_close(v, values[:, :, 7:35])
        # token 0 落在 block 3 的第 0 槽(blocks 顺序是 [3, 0, 4])
        torch.testing.assert_close(paged[3, :, 0, :4], keys[0, :, 0])
        torch.testing.assert_close(paged[3, :, 0, 4:], values[0, :, 0])
        self.assertTrue(torch.all(paged[1] == -1))

    def test_shape_mismatch_fail_closed(self):
        paged = torch.zeros(5, 2, 16, 8)
        with self.assertRaises(ValueError):  # 奇数内容维不是 K/V 对半
            read_paged(torch.zeros(5, 2, 16, 7), [0], 0, 1, 16)
        with self.assertRaises(ValueError):  # 宽度不符
            write_paged(paged, [0], torch.zeros(1, 2, 3, 5), torch.zeros(1, 2, 3, 5), 16)
        with self.assertRaises(ValueError):  # block_size 不符
            write_paged(paged, [0], torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4), 8)


class MlaPagedCacheTests(unittest.TestCase):
    """MLA 的 3 维分页 cache: 尾维是 [压缩潜向量 | k_rot] 紧邻拼接."""

    LATENT, ROPE, BLOCK = 8, 4, 16
    CONTENT = LATENT + ROPE

    def paged(self, blocks=5):
        return torch.full((blocks, self.BLOCK, self.CONTENT), -1.)

    def test_round_trip_splits_latent_and_rope(self):
        paged = self.paged()
        keys = torch.randn(1, 1, 37, self.LATENT)
        values = torch.randn(1, 1, 37, self.ROPE)
        blocks = [3, 0, 4]
        write_paged(paged, blocks, keys, values, self.BLOCK)
        k, v = read_paged(paged, blocks, 7, 35, self.BLOCK, self.LATENT)
        torch.testing.assert_close(k, keys[:, :, 7:35])
        torch.testing.assert_close(v, values[:, :, 7:35])
        # token 0 落在 block 3 的第 0 槽, 潜向量在前半、k_rot 在后半
        torch.testing.assert_close(paged[3, 0, :self.LATENT], keys[0, 0, 0])
        torch.testing.assert_close(paged[3, 0, self.LATENT:], values[0, 0, 0])
        self.assertTrue(torch.all(paged[1] == -1))

    def test_shape_gates_fail_closed(self):
        with self.assertRaises(ValueError):  # 读回来必须先知道潜向量宽度
            read_paged(self.paged(), [0], 0, 1, self.BLOCK)
        with self.assertRaises(ValueError):  # block_size 不符
            read_paged(self.paged(), [0], 0, 1, 8, self.LATENT)
        with self.assertRaises(ValueError):  # 两槽宽度之和必须等于内容维
            write_paged(self.paged(), [0], torch.zeros(1, 1, 3, self.LATENT),
                        torch.zeros(1, 1, 3, self.ROPE + 1), self.BLOCK)
        with self.assertRaises(ValueError):  # 越界
            read_paged(self.paged(), [0], 0, 17, self.BLOCK, self.LATENT)

    def test_snapshot_keeps_unequal_slot_widths(self):
        prefix = mla_cache(5)
        stage = Stage(list(range(9)), prefix, 11, 2)
        for start, end in [(5, 8), (8, 9)]:
            for i in range(2):
                stage.save(i, torch.full((1, 1, end - start, self.LATENT), float(start)),
                           torch.full((1, 1, end - start, self.ROPE), -float(start)),
                           start, end)
        result = stage.finish(glm_config())
        self.assertEqual(result.get_seq_length(), 9)
        self.assertEqual(tuple(result.layers[0].keys.shape), (1, 1, 9, self.LATENT))
        self.assertEqual(tuple(result.layers[0].values.shape), (1, 1, 9, self.ROPE))
        torch.testing.assert_close(result.layers[0].keys[:, :, :5], prefix.layers[0].keys)
        torch.testing.assert_close(result.layers[0].values[:, :, :5], prefix.layers[0].values)
        self.assertTrue(torch.all(result.layers[0].keys[:, :, 5:8] == 5))
        self.assertTrue(torch.all(result.layers[0].values[:, :, 5:8] == -5))

    def test_capture_rejects_slots_that_disagree_on_length(self):
        stage = Stage([1], None, 2, 1)
        stage.save(0, torch.ones(1, 1, 1, self.LATENT), torch.ones(1, 1, 1, self.ROPE), 0, 1)
        with self.assertRaises(ValueError):  # 两个槽的 token 数不同
            stage.save(0, torch.ones(1, 1, 1, self.LATENT), torch.ones(1, 1, 2, self.ROPE), 1, 2)


class MlaEngineCacheTests(unittest.TestCase):
    """vLLM 引擎对 MLA 快照的形状校验(不加载 vLLM, 只借 `_validate_cache`)."""

    def engine(self, layers=2):
        from types import SimpleNamespace
        from lminfer.vllm_engine import VLLMEngine
        engine = VLLMEngine.__new__(VLLMEngine)
        engine.model = SimpleNamespace(config=glm_config(layers), dtype=torch.float32)
        return engine

    def test_accepts_mla_snapshot_and_rejects_other_shapes(self):
        engine = self.engine()
        engine._validate_cache(mla_cache(6))                       # 正确: 不等宽两个槽
        with self.assertRaises(ValueError):
            engine._validate_cache(cache(6))                       # 对称槽(GQA 形状)
        with self.assertRaises(ValueError):
            self.engine(layers=3)._validate_cache(mla_cache(6))    # 层数不符


class ModelFamilyGateTests(unittest.TestCase):
    def test_mla_family_is_accepted_and_selects_the_mla_backend(self):
        from lminfer.vllm_engine import attention_backend_name, validate_model
        mla = glm_config()
        validate_model(mla)
        self.assertEqual(attention_backend_name(mla), "FLASH_ATTN_MLA")
        self.assertEqual(attention_backend_name(config()), "FLASH_ATTN")

    def test_unsupported_family_is_rejected(self):
        from lminfer.vllm_engine import validate_model
        from transformers import LlamaConfig
        with self.assertRaises(ValueError):
            validate_model(LlamaConfig(num_hidden_layers=2, hidden_size=8,
                                       num_attention_heads=2, num_key_value_heads=2))
