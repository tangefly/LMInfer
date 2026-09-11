"""模型适配层的单元测试: 模型加载分支与 RoPE 模块定位.

不加载真实权重(那是 GPU 集成测试的事): 这里用真实的 transformers config 类
+ mock 掉 from_pretrained, 验证"多模态包装 / FP8 反量化"的分支选择。
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from transformers import Glm4Config, Mistral3Config, Qwen3Config

from lminfer import model_adapters
from lminfer.model_adapters import (
    _is_finegrained_fp8,
    _supports_causal_lm,
    load_text_model,
    resolve_rope_layout,
    resolve_rotary_emb,
)


class CausalLmSupportTest(unittest.TestCase):
    def test_text_models_map_to_auto_model_for_causal_lm(self):
        self.assertTrue(_supports_causal_lm(Qwen3Config()))

    def test_multimodal_wrapper_is_not_a_causal_lm(self):
        # Mistral3ForConditionalGeneration 只注册在 AutoModelForImageTextToText 下
        self.assertFalse(_supports_causal_lm(Mistral3Config()))


class FineGrainedFp8Test(unittest.TestCase):
    def test_detects_dict_and_config_forms(self):
        as_dict = SimpleNamespace(quantization_config={"quant_method": "fp8"})
        self.assertTrue(_is_finegrained_fp8(as_dict))
        inner = SimpleNamespace(quantization_config={"quant_method": "fp8"})
        self.assertTrue(_is_finegrained_fp8(SimpleNamespace(text_config=inner)))
        self.assertFalse(_is_finegrained_fp8(Qwen3Config()))
        # 其他量化方法(如 awq)不走 FP8 反量化分支
        self.assertFalse(_is_finegrained_fp8(
            SimpleNamespace(quantization_config={"quant_method": "awq"})))


class LoadTextModelTest(unittest.TestCase):
    """验证加载分支: 纯文本 / 多模态 / FP8 反量化."""

    def _load(self, config, **kwargs):
        fake = SimpleNamespace(config=config)
        with mock.patch("transformers.AutoConfig.from_pretrained", return_value=config), \
             mock.patch("transformers.AutoModelForCausalLM.from_pretrained",
                        return_value=fake) as causal_call, \
             mock.patch("transformers.AutoModelForImageTextToText.from_pretrained",
                        return_value=fake) as multimodal_call:
            model, text_config, notes = load_text_model("/tmp/model", dtype=None, **kwargs)
        return model, text_config, notes, causal_call, multimodal_call

    def test_text_model_uses_causal_lm_path(self):
        config = Qwen3Config()
        _, text_config, notes, causal_call, multimodal_call = self._load(config)
        self.assertEqual(causal_call.call_count, 1)
        self.assertEqual(multimodal_call.call_count, 0)
        self.assertIs(text_config, config)
        self.assertEqual(notes, [])

    def test_multimodal_wrapper_returns_text_config(self):
        config = Mistral3Config()
        _, text_config, notes, causal_call, multimodal_call = self._load(config)
        self.assertEqual(causal_call.call_count, 0)
        self.assertEqual(multimodal_call.call_count, 1)
        # 顶层 config 没有 num_hidden_layers/rope_parameters, 必须拿到文本塔 config
        self.assertIs(text_config, config.get_text_config(decoder=True))
        self.assertTrue(hasattr(text_config, "num_hidden_layers"))
        self.assertTrue(any("多模态" in note for note in notes))

    def test_fp8_dequantizes_when_kernel_missing(self):
        config = Mistral3Config()
        config.quantization_config = {
            "quant_method": "fp8", "activation_scheme": "static",
            "weight_block_size": None, "dequantize": False,
            "modules_to_not_convert": ["lm_head"],
        }
        with mock.patch.object(model_adapters, "_fp8_kernel_available", return_value=False):
            _, _, notes, _, multimodal_call = self._load(config)
        quant = multimodal_call.call_args.kwargs["quantization_config"]
        self.assertTrue(quant.dequantize)
        # per-tensor 量化的 weight_block_size 是 None, 不能被默认值 (128, 128) 覆盖
        self.assertIsNone(quant.weight_block_size)
        self.assertEqual(quant.activation_scheme, "static")
        self.assertIn("反量化", " ".join(notes))

    def test_fp8_keeps_native_weights_when_kernel_available(self):
        config = Mistral3Config()
        config.quantization_config = {"quant_method": "fp8", "weight_block_size": None}
        with mock.patch.object(model_adapters, "_fp8_kernel_available", return_value=True):
            _, _, notes, _, multimodal_call = self._load(config)
        self.assertNotIn("quantization_config", multimodal_call.call_args.kwargs)
        self.assertIn("kernels", " ".join(notes))

    def test_fp8_dequantize_can_be_forced(self):
        config = Qwen3Config()
        config.quantization_config = {"quant_method": "fp8", "weight_block_size": None}
        with mock.patch.object(model_adapters, "_fp8_kernel_available", return_value=True):
            _, _, notes, causal_call, _ = self._load(config, dequantize_fp8=True)
        self.assertTrue(causal_call.call_args.kwargs["quantization_config"].dequantize)
        self.assertIn("反量化", " ".join(notes))


class ResolveRotaryEmbTest(unittest.TestCase):
    def test_prefers_text_tower_over_vision_tower(self):
        # 视觉塔(Pixtral)自带一套 RoPE, 且常排在前面; 必须取文本塔的
        text_rope, vision_rope = torch.nn.Identity(), torch.nn.Linear(2, 2)
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.vision_tower = torch.nn.Module()
        model.model.vision_tower.rotary_emb = vision_rope   # 先注册, named_modules 先命中
        model.model.language_model = torch.nn.Module()
        model.model.language_model.rotary_emb = text_rope

        self.assertIs(resolve_rotary_emb(model), text_rope)

    def test_plain_text_model_uses_model_rotary_emb(self):
        rope = torch.nn.Identity()
        model = torch.nn.Module()
        model.model = torch.nn.Module()
        model.model.rotary_emb = rope
        self.assertIs(resolve_rotary_emb(model), rope)

    def test_missing_rope_returns_none(self):
        self.assertIsNone(resolve_rotary_emb(torch.nn.Linear(2, 2)))


class ResolveRopeLayoutTest(unittest.TestCase):
    def test_glm4_is_partial_and_interleaved(self):
        # GLM-4-9B-0414: head_dim 128, partial_rotary_factor 0.5 -> 只转前 64 维,
        # 且 rotate_half 是奇偶交错(GPT-NeoX 式)
        layout = resolve_rope_layout(Glm4Config())
        self.assertEqual(layout.rotary_dim, 64)
        self.assertTrue(layout.interleaved)

    def test_default_rope_is_full_and_half_split(self):
        config = Qwen3Config()
        layout = resolve_rope_layout(config)
        head_dim = config.head_dim
        self.assertEqual(layout.rotary_dim, head_dim)
        self.assertFalse(layout.interleaved)

    def test_partial_factor_from_top_level_attribute(self):
        # 旧版实现把 partial_rotary_factor 放在顶层字段: 两处都要探测
        config = Qwen3Config()
        config.rope_parameters.pop("partial_rotary_factor", None)
        config.partial_rotary_factor = 0.5
        layout = resolve_rope_layout(config)
        self.assertEqual(layout.rotary_dim, config.head_dim // 2)


if __name__ == "__main__":
    unittest.main()
