import json
import unittest

from lminfer.model_adapters import resolve_model_profile, resolve_tool_parser
from lminfer.toolcalls import (
    clean_output_content,
    parse_model_output,
    parse_output_tool_calls,
)


class _Tokenizer:
    """只实现自动识别需要的两个方法的 tokenizer 桩."""

    def __init__(self, specials):
        self._specials = {tok: i for i, tok in enumerate(specials)}

    def convert_tokens_to_ids(self, token):
        return self._specials.get(token, -1)

    def convert_ids_to_tokens(self, tid):
        for tok, i in self._specials.items():
            if i == tid:
                return tok
        return None


LLAMA_TOK = _Tokenizer(["<|python_tag|>"])
HERMES_TOK = _Tokenizer(["<tool_call>", "</tool_call>"])
PLAIN_TOK = _Tokenizer([])

LLAMA_CALL = ('<|python_tag|>{"name": "research", "parameters": '
              '{"query": "What event did the institution hold in 2002?", '
              '"source_ids": "[\'S1\']"}}')
HERMES_CALL = ('<tool_call>\n{"name": "get_weather", "arguments": '
               '{"city": "Shanghai"}}\n</tool_call>')

RESEARCH_SCHEMA = {"type": "object", "properties": {
    "query": {"type": "string"},
    "source_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["query"]}
SCHEMAS = {"research": RESEARCH_SCHEMA}


class ToolParserResolutionTest(unittest.TestCase):
    def test_auto_detection_by_special_tokens(self):
        self.assertEqual(resolve_tool_parser("auto", LLAMA_TOK), "llama3_json")
        self.assertEqual(resolve_tool_parser("auto", HERMES_TOK), "hermes")
        self.assertEqual(resolve_tool_parser("auto", PLAIN_TOK), "none")

    def test_explicit_parser_kept(self):
        self.assertEqual(resolve_tool_parser("hermes", LLAMA_TOK), "hermes")
        self.assertEqual(resolve_tool_parser("qwen", HERMES_TOK), "qwen")
        self.assertEqual(resolve_tool_parser("none", LLAMA_TOK), "none")

    def test_profile_fallback_only_on_mismatch(self):
        # Llama 模型配 hermes: 冲突, 回退原生 llama3_json
        profile = resolve_model_profile("hermes", LLAMA_TOK)
        self.assertEqual(profile.tool_parser, "hermes")
        self.assertEqual(profile.native_parser, "llama3_json")
        self.assertEqual(profile.fallback_parser, "llama3_json")
        # 配置与家族一致: 不回退
        self.assertIsNone(resolve_model_profile("auto", LLAMA_TOK).fallback_parser)
        self.assertIsNone(resolve_model_profile("hermes", HERMES_TOK).fallback_parser)
        self.assertIsNone(resolve_model_profile("llama3_json", LLAMA_TOK).fallback_parser)
        # 模型没有工具调用协议: 无原生解析器可回退
        self.assertIsNone(resolve_model_profile("hermes", PLAIN_TOK).fallback_parser)
        # 显式 none: 解析整体关闭, 回退不参与
        self.assertIsNone(resolve_model_profile("none", LLAMA_TOK).fallback_parser)


class ParseModelOutputTest(unittest.TestCase):
    def test_dispatch(self):
        self.assertEqual(parse_output_tool_calls(HERMES_CALL, "hermes")[0]
                         ["function"]["name"], "get_weather")
        self.assertEqual(parse_output_tool_calls(LLAMA_CALL, "llama3_json")[0]
                         ["function"]["name"], "research")
        self.assertEqual(parse_output_tool_calls(LLAMA_CALL, "hermes"), [])
        self.assertEqual(clean_output_content("<|python_tag|>hello", "llama3_json"),
                         "hello")

    def test_mismatch_falls_back_to_native_parser(self):
        # Llama 3.x 输出 + 显式 hermes 配置: hermes 解析不到, 回退 llama3_json
        calls, content = parse_model_output(LLAMA_CALL, "hermes", "llama3_json",
                                            request_id="test")
        self.assertEqual(len(calls), 1)
        fn = calls[0]["function"]
        self.assertEqual(fn["name"], "research")
        # arguments 保留模型原始 JSON 子串(round-trip 保真)
        args = json.loads(fn["arguments"])
        self.assertEqual(args["source_ids"], "['S1']")
        # 有 tool_calls 时 content 为 None(vLLM 语义)
        self.assertIsNone(content)

    def test_no_fallback_when_configured_parser_matches(self):
        calls, content = parse_model_output(HERMES_CALL, "hermes", "llama3_json")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertIsNone(content)

    def test_fallback_used_for_content_cleanup_without_calls(self):
        # 冲突且模型没有输出工具调用: 清理语义跟随原生协议
        calls, content = parse_model_output("<|python_tag|>plain answer",
                                            "hermes", "llama3_json")
        self.assertEqual(calls, [])
        self.assertEqual(content, "plain answer")

    def test_without_fallback_legacy_behavior(self):
        # 无回退配置时(如模型无工具协议), 输出按配置解析器原样处理
        calls, content = parse_model_output(LLAMA_CALL, "hermes", None)
        self.assertEqual(calls, [])
        self.assertEqual(content, LLAMA_CALL)

    def test_untagged_json_call_parsed_by_native_parser(self):
        # 模型退化掉 <|python_tag|> 前缀时, llama3_json 仍能从 { 解析
        text = LLAMA_CALL.removeprefix("<|python_tag|>")
        calls, _ = parse_model_output(text, "hermes", "llama3_json")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "research")

    def test_schema_repairs_stringified_array(self):
        # schema 声明 array 但模型写成字符串列表 "['S1']" -> 还原成 ["S1"]
        calls, content = parse_model_output(LLAMA_CALL, "hermes", "llama3_json",
                                            schemas=SCHEMAS)
        self.assertEqual(len(calls), 1)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["source_ids"], ["S1"])
        self.assertEqual(args["query"], "What event did the institution hold in 2002?")
        self.assertIsNone(content)

    def test_schema_repair_also_works_for_hermes_blocks(self):
        text = ('<tool_call>\n{"name": "research", "arguments": '
                '{"query": "q", "source_ids": "[\'S1\', \'S2\']"}}\n</tool_call>')
        calls, _ = parse_model_output(text, "hermes", None, schemas=SCHEMAS)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["source_ids"], ["S1", "S2"])

    def test_schema_repair_leaves_non_array_and_broken_values_alone(self):
        # 字符串字段不被改写; 不是合法列表字面量的 array 值保持原样
        text = ('<|python_tag|>{"name": "research", "parameters": '
                '{"query": "[\'S1\']", "source_ids": "[S1"}}')
        calls, _ = parse_model_output(text, "llama3_json", None, schemas=SCHEMAS)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["query"], "['S1']")   # string 类型, 不修复
        self.assertEqual(args["source_ids"], "[S1")  # 非法字面量, 保持原样

    def test_no_schema_keeps_model_output_verbatim(self):
        # 不传 schema 时不做修复: 保持与模型原始输出逐位一致(round-trip 保真)
        calls, _ = parse_model_output(LLAMA_CALL, "hermes", "llama3_json")
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["source_ids"], "['S1']")

    def test_schema_repair_only_touches_matching_function(self):
        # 其他函数的 schema 不影响当前调用
        calls, _ = parse_model_output(LLAMA_CALL, "hermes", "llama3_json",
                                      schemas={"other": RESEARCH_SCHEMA})
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["source_ids"], "['S1']")


if __name__ == "__main__":
    unittest.main()
