import json
import unittest

from lminfer.model_adapters import (
    resolve_model_profile,
    resolve_tool_parser,
    resolve_tool_result_wrapper,
    supports_chat_template,
)
from lminfer.toolcalls import (
    MISTRAL_CALL_ID_LEN,
    MistralStreamSplitter,
    clean_output_content,
    make_stream_splitter,
    parse_mistral_tool_calls,
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
HERMES_TOK = _Tokenizer(["<tool_call>", "</tool_call>",
                         "<tool_response>", "</tool_response>"])
MISTRAL_TOK = _Tokenizer(["[TOOL_CALLS]", "[ARGS]",
                          "[TOOL_RESULTS]", "[/TOOL_RESULTS]"])
PLAIN_TOK = _Tokenizer([])

LLAMA_CALL = ('<|python_tag|>{"name": "research", "parameters": '
              '{"query": "What event did the institution hold in 2002?", '
              '"source_ids": "[\'S1\']"}}')
HERMES_CALL = ('<tool_call>\n{"name": "get_weather", "arguments": '
               '{"city": "Shanghai"}}\n</tool_call>')
# Ministral-3(v13 tokenizer)的真实输出形态: [TOOL_CALLS]name[ARGS]{json}
MISTRAL_CALL = '[TOOL_CALLS]get_weather[ARGS]{"city": "Shanghai"}'
MISTRAL_MULTI = (MISTRAL_CALL + '[TOOL_CALLS]research[ARGS]'
                 '{"query": "q", "source_ids": "[\'S1\']"}')

RESEARCH_SCHEMA = {"type": "object", "properties": {
    "query": {"type": "string"},
    "source_ids": {"type": "array", "items": {"type": "string"}}},
    "required": ["query"]}
SCHEMAS = {"research": RESEARCH_SCHEMA}


class ToolParserResolutionTest(unittest.TestCase):
    def test_auto_detection_by_special_tokens(self):
        self.assertEqual(resolve_tool_parser("auto", LLAMA_TOK), "llama3_json")
        self.assertEqual(resolve_tool_parser("auto", HERMES_TOK), "hermes")
        self.assertEqual(resolve_tool_parser("auto", MISTRAL_TOK), "mistral")
        self.assertEqual(resolve_tool_parser("auto", PLAIN_TOK), "none")

    def test_tool_result_wrapper_detection(self):
        self.assertEqual(resolve_tool_result_wrapper(HERMES_TOK),
                         ("<tool_response>", "</tool_response>"))
        self.assertEqual(resolve_tool_result_wrapper(MISTRAL_TOK),
                         ("[TOOL_RESULTS]", "[/TOOL_RESULTS]"))
        self.assertIsNone(resolve_tool_result_wrapper(PLAIN_TOK))

    def test_mistral_common_backend_counts_as_having_a_template(self):
        # mistral-common 后端没有 chat_template 属性, 但 apply_chat_template 可用
        tokenizer = type("MistralCommonBackend", (), {})()
        self.assertTrue(supports_chat_template(tokenizer))
        self.assertFalse(supports_chat_template(type("OtherTokenizer", (), {})()))

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


class MistralToolCallTest(unittest.TestCase):
    def test_single_call_round_trips_arguments_verbatim(self):
        calls = parse_mistral_tool_calls(MISTRAL_CALL)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        # 逐位保留模型原始 JSON 子串: mistral-common 二次渲染会 json.dumps 归一化,
        # 模型自己的 `": "` 风格与之一致, 保留原串 LCP 才能整段命中
        self.assertEqual(calls[0]["function"]["arguments"], '{"city": "Shanghai"}')
        self.assertEqual(calls[0]["type"], "function")

    def test_generated_call_id_satisfies_mistral_common(self):
        # mistral-common 校验 id 必须是 9 位字母数字, 否则下一轮模板渲染直接抛异常
        call = parse_mistral_tool_calls(MISTRAL_CALL)[0]
        self.assertEqual(len(call["id"]), MISTRAL_CALL_ID_LEN)
        self.assertTrue(call["id"].isalnum())

    def test_multiple_concatenated_calls(self):
        # main agent 一次回复里发多个子 agent 调用(multi_subagent_kv_reuse 场景)
        calls = parse_mistral_tool_calls(MISTRAL_MULTI)
        self.assertEqual([c["function"]["name"] for c in calls],
                         ["get_weather", "research"])
        self.assertNotEqual(calls[0]["id"], calls[1]["id"])

    def test_call_id_marker_removed_from_name(self):
        # v11 tokenizer 会在 name 与 [ARGS] 之间插 [CALL_ID]<id>
        calls = parse_mistral_tool_calls(
            '[TOOL_CALLS]get_weather[CALL_ID]Abc123xyz[ARGS]{"city": "S"}')
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(calls[0]["function"]["arguments"], '{"city": "S"}')

    def test_legacy_json_array_format(self):
        calls = parse_mistral_tool_calls(
            '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "S"}}]')
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "S"})

    def test_content_before_call_is_kept_for_kv_alignment(self):
        # 与 vLLM 的 mistral/hermes 解析器一致: 有 tool_calls 时也返回调用前的正文。
        # 客户端把它回填后, 模板渲染的 assistant 消息才能与模型实际生成的 token
        # 逐位对齐, 跨请求 KV 前缀复用才能覆盖"正文 + 工具调用"整段。
        calls, content = parse_model_output(
            "let me check[TOOL_CALLS]get_weather[ARGS]{}", "mistral", None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, "let me check")
        # 只保留第一个调用标记之前的正文(不 strip: 空正文返回 None)
        calls, content = parse_model_output(
            MISTRAL_CALL + " trailing text", "mistral", None)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(content)
        self.assertEqual(clean_output_content(
            "let me check[TOOL_CALLS]get_weather[ARGS]{}", "mistral"), "let me check")

    def test_prefix_content_policy_matches_vllm(self):
        # hermes 同样保留调用前正文(vLLM hermes_tool_parser: content if content else None)
        calls, content = parse_model_output(
            "thinking...<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}"
            "</tool_call>", "hermes", None)
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, "thinking...")
        # llama3_json 保持 null(vLLM llama_tool_parser: content=None)
        calls, content = parse_model_output(
            'ok<|python_tag|>{"name": "research", "parameters": {"query": "q"}}',
            "llama3_json", None)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(content)
        # 冲突回退时按最终生效的原生协议决定
        calls, content = parse_model_output(
            "preamble" + MISTRAL_CALL, "hermes", "mistral")
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, "preamble")

    def test_schema_repairs_stringified_array(self):
        calls = parse_mistral_tool_calls(MISTRAL_MULTI, SCHEMAS)
        args = json.loads(calls[1]["function"]["arguments"])
        self.assertEqual(args["source_ids"], ["S1"])

    def test_no_marker_returns_nothing(self):
        self.assertEqual(parse_mistral_tool_calls("just an answer"), [])
        self.assertEqual(clean_output_content("just an answer", "mistral"),
                         "just an answer")

    def test_dispatch_and_fallback(self):
        self.assertEqual(parse_output_tool_calls(MISTRAL_CALL, "mistral")[0]
                         ["function"]["name"], "get_weather")
        self.assertEqual(parse_output_tool_calls(MISTRAL_CALL, "hermes"), [])
        # 显式配置与模型协议冲突: 按原生 mistral 协议回退
        calls, content = parse_model_output(MISTRAL_CALL, "hermes", "mistral",
                                            request_id="test")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertIsNone(content)


class MistralStreamSplitterTest(unittest.TestCase):
    def split(self, chunks):
        splitter = make_stream_splitter("mistral")
        events = []
        for chunk in chunks:
            events.extend(splitter.push(chunk))
        events.extend(splitter.flush())
        return events

    def test_stream_splits_content_and_call(self):
        events = self.split(["sure, ", "[TOOL_CALLS]get_", "weather[ARGS]",
                             '{"city": "Sha', 'nghai"}'])
        self.assertEqual(events[0], ("content", "sure, "))
        self.assertEqual([k for k, _ in events], ["content", "tool_call"])
        call = events[1][1]
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(call["function"]["arguments"], '{"city": "Shanghai"}')
        self.assertTrue(call["id"].isalnum() and len(call["id"]) == MISTRAL_CALL_ID_LEN)

    def test_stream_multiple_calls_then_trailing_text(self):
        events = self.split([MISTRAL_MULTI + "done"])
        calls = [p for k, p in events if k == "tool_call"]
        self.assertEqual([c["function"]["name"] for c in calls],
                         ["get_weather", "research"])
        self.assertEqual([p for k, p in events if k == "content"], ["done"])

    def test_incomplete_call_flushed_as_text(self):
        # 与 hermes/llama 的流式实现一致: 标记本身被消费(不进入正文),
        # 没写完的内容按普通文本返回, 不静默丢弃
        events = self.split(["[TOOL_CALLS]get_weather[ARGS]{\"city\": "])
        self.assertEqual([k for k, _ in events], ["content"])
        self.assertEqual(events[0][1], 'get_weather[ARGS]{"city": ')

    def test_marker_never_leaks_into_content(self):
        # 标记被拆成多 chunk 时不能提前当普通文本输出
        events = self.split(["[TOOL", "_CALLS]f[ARGS]{}"])
        self.assertEqual([k for k, _ in events], ["tool_call"])


if __name__ == "__main__":
    unittest.main()
