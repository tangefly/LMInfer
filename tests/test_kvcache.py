import unittest

import torch
from transformers import DynamicCache

from lminfer.kvcache import (
    KIND_MAIN,
    KIND_SUB,
    TOOL_RESPONSE_CLOSE,
    TOOL_RESPONSE_OPEN,
    SessionKVStore,
    rebase_rope_cache,
)


class FakeTokenizer:
    ids = {TOOL_RESPONSE_OPEN: 900, TOOL_RESPONSE_CLOSE: 901}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def convert_ids_to_tokens(self, token_id):
        for token, tid in self.ids.items():
            if tid == token_id:
                return token
        return f"tok_{token_id}"


def make_cache(length: int) -> DynamicCache:
    keys = torch.arange(length * 2, dtype=torch.float32).reshape(1, 1, length, 2)
    values = keys + 1000
    return DynamicCache(ddp_cache_data=[(keys, values)], config=None)


class SessionKVStoreTest(unittest.TestCase):

    def test_take_main_reuse_transfers_old_batch_and_preserves_new_subs(self):
        import weakref
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        store.put("s", KIND_MAIN, [1, 2], make_cache(2))
        store.put("s", KIND_SUB, [8, 101, 102, 103, 104], make_cache(5), prompt_len=1)
        original = weakref.ref(store._segments["s"]["subs"][0].cache.layers[0].keys)
        prefixes, grafts = store.take_main_reuse(
            "s", ["main", "sub", "main"], [1, 2, 900, 101, 102, 103, 104, 901],
            append=True)
        self.assertEqual(len(prefixes), 2)
        self.assertEqual(len(grafts), 1)
        self.assertEqual(len(store.propose("s", ["main"])), 1)
        self.assertIsNotNone(original())  # owned by the main request until selection
        prefixes.clear()
        self.assertIsNone(original())
        self.assertEqual(grafts[0].cache.get_seq_length(), 4)  # independent copy
        store.put("s", KIND_SUB, [9, 201, 202, 203, 204], make_cache(5), prompt_len=1)
        store.put("s", KIND_MAIN, [1, 2, 3], make_cache(3), clear_subs_on_main=False)
        self.assertEqual([p.tokens for p in store.propose("s", ["main"])],
                         [[1, 2, 3], [9, 201, 202, 203, 204]])
    def test_build_grafts_reuses_all_subs_since_latest_main(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3, 4]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(4), prompt_len=4))

        sub_outputs = [
            [101, 102, 103, 104],
            [201, 202, 203, 204],
            [301, 302, 303, 304],
        ]
        for i, output in enumerate(sub_outputs, start=1):
            seq = [10 * i, 10 * i + 1] + output
            self.assertTrue(
                store.put(session_id, KIND_SUB, seq, make_cache(len(seq)), prompt_len=2)
            )

        prompt = (
            main_tokens
            + [900] + sub_outputs[0] + [901, 88]
            + [900] + sub_outputs[1] + [901, 89]
            + [900] + sub_outputs[2] + [901, 77]
        )
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub3", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], sub_outputs)
        self.assertEqual([g.position for g in grafts], [5, 12, 19])
        self.assertEqual([g.cache.get_seq_length() for g in grafts], [4, 4, 4])
        self.assertEqual(len(store.propose(session_id, [KIND_MAIN])), 4)



    def test_build_grafts_partially_matches_one_tool_response_window(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(3), prompt_len=3))

        sub_out = [101, 102, 103, 104, 201, 202, 203, 204]
        sub_seq = [9] + sub_out
        self.assertTrue(store.put(session_id, KIND_SUB, sub_seq, make_cache(len(sub_seq)), prompt_len=1))

        # 900/901 are tool_response markers. The window has unmatched boundary tokens
        # and an unmatched token in the middle. A sub invocation is treated as one
        # KV segment, so only one longest contiguous span is grafted.
        prompt = main_tokens + [900, 77, 101, 102, 103, 104, 88, 201, 202, 203, 204, 99, 901]
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], [[101, 102, 103, 104]])
        self.assertEqual([g.position for g in grafts], [5])
        self.assertEqual([g.source_position for g in grafts], [1])


    def test_sub_put_replaces_same_trace_invocation(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        self.assertTrue(store.put(session_id, KIND_MAIN, [1, 2], make_cache(2), prompt_len=2,
                                  trace=[KIND_MAIN]))

        self.assertTrue(store.put(session_id, KIND_SUB, [7, 8, 9, 10], make_cache(4),
                                  prompt_len=1, trace=[KIND_MAIN, KIND_SUB]))
        self.assertTrue(store.put(session_id, KIND_SUB, [11, 12, 13, 14, 15], make_cache(5),
                                  prompt_len=1, trace=[KIND_MAIN, KIND_SUB]))
        candidates = store.propose(session_id, [KIND_MAIN])

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[1].tokens, [11, 12, 13, 14, 15])

        self.assertTrue(store.put(session_id, KIND_SUB, [21, 22, 23, 24], make_cache(4),
                                  prompt_len=1, trace=[KIND_MAIN, KIND_SUB, KIND_SUB]))
        candidates = store.propose(session_id, [KIND_MAIN])

        self.assertEqual(len(candidates), 3)
        self.assertEqual([c.tokens for c in candidates[1:]],
                         [[11, 12, 13, 14, 15], [21, 22, 23, 24]])

    def test_build_grafts_skips_sub_internal_intermediate_outputs(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3, 4]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(4), prompt_len=4))

        final_outputs = [
            [101, 102, 103, 104],
            [201, 202, 203, 204],
            [301, 302, 303, 304],
        ]
        internal_outputs = [
            [11, 12, 13, 14, 15, 16],
            [21, 22, 23, 24, 25, 26],
            [31, 32, 33, 34, 35, 36],
        ]
        for i, output in enumerate(final_outputs):
            internal_seq = [40 + i] + internal_outputs[i]
            final_seq = [50 + i, 60 + i] + output
            self.assertTrue(
                store.put(session_id, KIND_SUB, internal_seq, make_cache(len(internal_seq)), prompt_len=1)
            )
            self.assertTrue(
                store.put(session_id, KIND_SUB, final_seq, make_cache(len(final_seq)), prompt_len=2)
            )

        prompt = (
            main_tokens
            + [900] + final_outputs[0] + [901, 88]
            + [900] + final_outputs[1] + [901, 89]
            + [900] + final_outputs[2] + [901, 77]
        )
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", "sub", "sub", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], final_outputs)
        self.assertEqual([g.position for g in grafts], [5, 12, 19])


    def test_build_grafts_uses_main_lcp_and_skips_unmatched_windows(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        # Saved main output is longer than the structured assistant tool-call rendering
        # in the final prompt, so len(main_seg.tokens) would skip the first real window.
        main_saved = [1, 2, 3, 70, 71, 72, 73, 74, 75, 76]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_saved, make_cache(len(main_saved)), prompt_len=3))

        outputs = [[101, 102, 103, 104], [201, 202, 203, 204]]
        for i, output in enumerate(outputs):
            # Add one intermediate sub output before each final answer. These should be skipped.
            self.assertTrue(store.put(session_id, KIND_SUB, [10 + i, 41, 42, 43, 44], make_cache(5), prompt_len=1))
            self.assertTrue(store.put(session_id, KIND_SUB, [20 + i] + output, make_cache(5), prompt_len=1))

        prompt = (
            [1, 2, 3, 80, 81]
            + [900, 51, 52, 53, 54, 901]  # stale/unmatched window before real tool results
            + [900] + outputs[0] + [901, 88]
            + [900] + outputs[1] + [901, 89]
        )
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", "sub", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], outputs)
        self.assertEqual([g.position for g in grafts], [12, 19])


    def test_build_grafts_prefers_final_answer_over_short_spurious_match(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        main_tokens = [1, 2, 3]
        self.assertTrue(store.put(session_id, KIND_MAIN, main_tokens, make_cache(3), prompt_len=3))

        # The intermediate sub turn shares a short phrase with the returned tool response,
        # but the later final sub answer covers much more and must be selected.
        intermediate = [10, 101, 102, 103, 104, 99]
        final = [201, 202, 101, 102, 103, 104, 105, 106, 107, 108, 203]
        self.assertTrue(store.put(session_id, KIND_SUB, [7] + intermediate, make_cache(7), prompt_len=1))
        self.assertTrue(store.put(session_id, KIND_SUB, [8] + final, make_cache(12), prompt_len=1))

        prompt = main_tokens + [900, 77, 101, 102, 103, 104, 105, 106, 107, 108, 88, 901]
        grafts = store.build_grafts(session_id, [KIND_MAIN, "sub", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], [[101, 102, 103, 104, 105, 106, 107, 108]])
        self.assertEqual([g.position for g in grafts], [5])
        self.assertEqual([g.source_position for g in grafts], [3])

    def test_new_main_clears_previous_sub_batch(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        self.assertTrue(store.put(session_id, KIND_MAIN, [1, 2], make_cache(2), prompt_len=2))
        self.assertTrue(store.put(session_id, KIND_SUB, [7, 8, 9, 10, 11, 12], make_cache(6), prompt_len=2))
        self.assertEqual(len(store.propose(session_id, [KIND_MAIN])), 2)

        self.assertTrue(store.put(session_id, KIND_MAIN, [1, 2, 3], make_cache(3), prompt_len=3))
        candidates = store.propose(session_id, [KIND_MAIN])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tokens, [1, 2, 3])
        self.assertEqual(store.build_grafts(session_id, [KIND_MAIN, "sub", KIND_MAIN], [1, 2, 3]), [])

    def test_clear_subs_releases_only_sub_batch(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        self.assertTrue(store.put(session_id, KIND_MAIN, [1, 2], make_cache(2), prompt_len=2))
        self.assertTrue(store.put(session_id, KIND_SUB, [3, 4, 5, 6], make_cache(4), prompt_len=2))
        self.assertTrue(store.put(session_id, KIND_SUB, [7, 8, 9, 10], make_cache(4), prompt_len=2))

        store.clear_subs(session_id)
        candidates = store.propose(session_id, [KIND_MAIN])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].tokens, [1, 2])

    def test_build_graft_compat_returns_last_matched_sub(self):
        store = SessionKVStore(config=None, tokenizer=FakeTokenizer())
        session_id = "s1"
        self.assertTrue(store.put(session_id, KIND_MAIN, [1], make_cache(1), prompt_len=1))
        self.assertTrue(store.put(session_id, KIND_SUB, [5, 6, 21, 22, 23, 24], make_cache(6), prompt_len=2))
        self.assertTrue(store.put(session_id, KIND_SUB, [7, 8, 31, 32, 33, 34], make_cache(6), prompt_len=2))

        prompt = [1, 900, 21, 22, 23, 24, 901, 900, 31, 32, 33, 34, 901]
        graft = store.build_graft(session_id, [KIND_MAIN, "sub", KIND_MAIN], prompt)

        self.assertIsNotNone(graft)
        self.assertEqual(graft.tokens, [31, 32, 33, 34])
        self.assertEqual(graft.position, 8)


class MistralFakeTokenizer:
    """Ministral 式的 tokenizer: 工具结果用 [TOOL_RESULTS] 包裹."""

    ids = {"[TOOL_RESULTS]": 7, "[/TOOL_RESULTS]": 8}

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, -1)

    def convert_ids_to_tokens(self, token_id):
        for token, tid in self.ids.items():
            if tid == token_id:
                return token
        return f"tok_{token_id}"


class NoWrapperFakeTokenizer:
    """既没有 <tool_response> 也没有 [TOOL_RESULTS]: 拼接模式应自动禁用."""

    def convert_tokens_to_ids(self, token):
        return -1

    def convert_ids_to_tokens(self, token_id):
        return f"tok_{token_id}"


class GraftWrapperTest(unittest.TestCase):
    def test_mistral_tool_results_window_is_grafted(self):
        store = SessionKVStore(config=None, tokenizer=MistralFakeTokenizer())
        main_tokens = [1, 2, 3]
        self.assertTrue(store.put("s", KIND_MAIN, main_tokens, make_cache(3), prompt_len=3))
        sub_out = [101, 102, 103, 104]
        self.assertTrue(store.put("s", KIND_SUB, [9] + sub_out, make_cache(5), prompt_len=1))

        prompt = main_tokens + [7] + sub_out + [8]
        grafts = store.build_grafts("s", [KIND_MAIN, "sub", KIND_MAIN], prompt)

        self.assertEqual([g.tokens for g in grafts], [sub_out])
        self.assertEqual([g.position for g in grafts], [4])
        self.assertEqual([g.source_position for g in grafts], [1])

    def test_unknown_wrapper_disables_append_mode(self):
        store = SessionKVStore(config=None, tokenizer=NoWrapperFakeTokenizer())
        self.assertTrue(store.put("s", KIND_MAIN, [1], make_cache(1), prompt_len=1))
        self.assertTrue(store.put("s", KIND_SUB, [9, 101, 102, 103, 104], make_cache(5), prompt_len=1))
        prompt = [1, 7, 101, 102, 103, 104, 8]
        self.assertEqual(store.build_grafts("s", [KIND_MAIN, "sub", KIND_MAIN], prompt), [])
        # LCP 复用不受影响, 仍然给出候选段
        self.assertEqual(len(store.propose("s", [KIND_MAIN])), 2)


class _SinusoidalRope(torch.nn.Module):
    """最小 RoPE 模块: 形状/约定与 transformers 的 RotaryEmbedding 一致."""

    def __init__(self, inv_freq):
        super().__init__()
        self.register_buffer("inv_freq", torch.as_tensor(inv_freq, dtype=torch.float32))

    def forward(self, x, position_ids):
        positions = position_ids.reshape(-1).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype)[None], emb.sin().to(x.dtype)[None]


def _rotate(x, cos, sin):
    """half-split RoPE 旋转(与 Qwen/Llama/Mistral 的 apply_rotary_pos_emb 一致)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin


class RopeRebaseTest(unittest.TestCase):
    def setUp(self):
        self.head_dim = 8
        self.length = 4
        self.source, self.target = 20, 35
        torch.manual_seed(0)
        self.keys = torch.randn(1, 1, self.length, self.head_dim)
        self.values = torch.randn(1, 1, self.length, self.head_dim)
        # 默认 RoPE 的逆频率(测试里显式写出来, 不依赖被测代码)
        self.inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))

    def _cache_with_keys(self, keys):
        return DynamicCache(ddp_cache_data=[(keys.clone(), self.values.clone())], config=None)

    def test_rebase_composes_rope_positions(self):
        # 子段 KV 是在 source 位置旋转过的; rebase 到 target 后应与"直接在 target
        # 位置旋转"逐元素一致(位置差旋转与位置旋转复合)
        rope = _SinusoidalRope(self.inv_freq)
        source_cos, source_sin = rope(self.keys, torch.full((1, self.length), self.source))
        at_source = _rotate(self.keys, source_cos, source_sin)
        at_target = _rotate(self.keys, *rope(self.keys, torch.full((1, self.length), self.target)))

        rebased = rebase_rope_cache(self._cache_with_keys(at_source), self.source,
                                    self.target, None, rope=rope)

        torch.testing.assert_close(rebased.layers[0].keys, at_target,
                                   atol=1e-6, rtol=1e-6)
        # V 不参与 RoPE, 原样深拷贝
        torch.testing.assert_close(rebased.layers[0].values, self.values)

    def test_rebase_uses_model_rope_not_plain_theta(self):
        # 缩放型 RoPE(YaRN/Llama-3)的逆频率与默认 theta 公式不同: 传了模型 RoPE
        # 就必须用它, 否则位置重映射会转到错误的相位(Ministral-3 是 YaRN factor 16)
        scaled = self.inv_freq / 16.0
        rope = _SinusoidalRope(scaled)
        source_cos, source_sin = rope(self.keys, torch.full((1, self.length), self.source))
        cache = self._cache_with_keys(_rotate(self.keys, source_cos, source_sin))

        with_rope = rebase_rope_cache(cache, self.source, self.target, None, rope=rope)
        without = rebase_rope_cache(cache, self.source, self.target, None)

        expected = _rotate(self.keys, *rope(self.keys, torch.full((1, self.length), self.target)))
        torch.testing.assert_close(with_rope.layers[0].keys, expected, atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(with_rope.layers[0].keys, without.layers[0].keys))

    def test_rebase_falls_back_when_rope_shape_is_unusable(self):
        class WrongShape(torch.nn.Module):
            def forward(self, x, position_ids):
                return torch.ones(1, 3, 4), torch.zeros(1, 3, 4)

        cache = self._cache_with_keys(self.keys)
        rebased = rebase_rope_cache(cache, self.source, self.target, None,
                                    rope=WrongShape())
        expected = rebase_rope_cache(cache, self.source, self.target, None)
        torch.testing.assert_close(rebased.layers[0].keys, expected.layers[0].keys)


class Ministral3YarnRebaseTest(unittest.TestCase):
    """用模型真实的 YaRN RoPE 模块验证 rebase 相位(Ministral-3, 不需要权重).

    Ministral-3 的 text_config 声明 rope_type=yarn(factor 16): 低频频段的逆频率
    与默认 theta 公式相差 16 倍。默认公式会把拼接进来的 K 转到错误相位 —— 这个
    用例就是那次适配的回归测试。
    """

    SOURCE, TARGET, LENGTH, HEAD_DIM = 1200, 2400, 6, 128

    @staticmethod
    def _rope():
        from transformers import Ministral3Config
        from transformers.models.ministral3.modeling_ministral3 import (
            Ministral3RotaryEmbedding,
        )
        config = Ministral3Config(hidden_size=4096, num_attention_heads=32,
                                  num_key_value_heads=8, num_hidden_layers=2,
                                  head_dim=128)
        config.rope_parameters = {
            "beta_fast": 32.0, "beta_slow": 1.0, "factor": 16.0,
            "llama_4_scaling_beta": 0.1, "mscale": 1.0, "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 16384, "rope_theta": 1000000.0,
            "rope_type": "yarn", "type": "yarn",
        }
        return Ministral3RotaryEmbedding(config)

    def _keys_at(self, rope, keys, position):
        positions = torch.full((1, keys.shape[-2]), position)
        return _rotate(keys, *rope(keys, positions))

    def test_rebase_matches_model_rope_at_target_position(self):
        rope = self._rope()
        torch.manual_seed(0)
        keys = torch.randn(1, 1, self.LENGTH, self.HEAD_DIM)
        at_source = self._keys_at(rope, keys, self.SOURCE)
        at_target = self._keys_at(rope, keys, self.TARGET)
        cache = DynamicCache(ddp_cache_data=[(at_source, at_source.clone())], config=None)

        rebased = rebase_rope_cache(cache, self.SOURCE, self.TARGET, None, rope=rope)

        torch.testing.assert_close(rebased.layers[0].keys, at_target,
                                   atol=1e-5, rtol=1e-5)

    def test_plain_theta_formula_rotates_to_the_wrong_phase(self):
        rope = self._rope()
        torch.manual_seed(0)
        keys = torch.randn(1, 1, self.LENGTH, self.HEAD_DIM)
        at_source = self._keys_at(rope, keys, self.SOURCE)
        at_target = self._keys_at(rope, keys, self.TARGET)
        cache = DynamicCache(ddp_cache_data=[(at_source, at_source.clone())], config=None)

        without_rope = rebase_rope_cache(cache, self.SOURCE, self.TARGET, None)

        # 相位错误不是舍入误差量级: 与正确答案相差 K 自身的量级
        error = (without_rope.layers[0].keys - at_target).abs().max()
        self.assertGreater(float(error), 1.0)


if __name__ == "__main__":
    unittest.main()
