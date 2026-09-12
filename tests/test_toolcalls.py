import json
import unittest

from lminfer.model_adapters import (
    ToolResultWrapper,
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
    parse_glm4_moe_tool_calls,
    parse_glm4_tool_calls,
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
# GLM-4 tokenizer: 没有专用工具调用 token, 靠角色标记 + config.model_type 识别
GLM4_ROLE_TOKENS = ["[gMASK]", "<sop>", "<|system|>", "<|user|>",
                    "<|assistant|>", "<|observation|>"]
GLM4_TOK = _Tokenizer(GLM4_ROLE_TOKENS)
# GLM-4.7-Flash tokenizer: <tool_call>/<arg_key>/<arg_value> 都是**单特殊 token**
# (按 token 探测会先命中 hermes, 必须靠 config.model_type 抢在它前面判定)
GLM4_MOE_TOKENS = GLM4_ROLE_TOKENS + ["<tool_call>", "</tool_call>",
                                      "<arg_key>", "</arg_key>",
                                      "<arg_value>", "</arg_value>",
                                      "<tool_response>", "</tool_response>",
                                      "<think>", "</think>"]
GLM4_MOE_TOK = _Tokenizer(GLM4_MOE_TOKENS)
PLAIN_TOK = _Tokenizer([])


class _Config:
    def __init__(self, model_type):
        self.model_type = model_type

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

    def test_auto_detection_glm4_by_config_and_tokenizer(self):
        # config.model_type 是权威信号(GLM-4-9B-0414 的 config)
        self.assertEqual(resolve_tool_parser("auto", GLM4_TOK, _Config("glm4")),
                         "glm4")
        # 拿不到 config 时退化到 <|observation|> 特殊 token 探测
        self.assertEqual(resolve_tool_parser("auto", GLM4_TOK), "glm4")
        # 别的模型家族不会被误判
        self.assertEqual(resolve_tool_parser("auto", HERMES_TOK, _Config("qwen3")),
                         "hermes")
        self.assertEqual(resolve_tool_parser("auto", PLAIN_TOK, _Config("llama")),
                         "none")

    def test_explicit_glm4_parser_kept(self):
        self.assertEqual(resolve_tool_parser("glm4", PLAIN_TOK), "glm4")

    def test_auto_detection_glm4_moe_beats_tool_call_token(self):
        # GLM-4.7-Flash: <tool_call> 是单特殊 token, 但块体是 XML 而不是 JSON ——
        # 若被 hermes 抢先识别, 解析必然失败且工具调用会被 cleaner 整段删掉。
        for model_type in ("glm4_moe_lite", "glm4_moe"):
            self.assertEqual(
                resolve_tool_parser("auto", GLM4_MOE_TOK, _Config(model_type)),
                "glm4_moe")
        # 同样的 tokenizer 上没有 config 时仍退回 hermes(其他家族的既有行为不变)
        self.assertEqual(resolve_tool_parser("auto", GLM4_MOE_TOK), "hermes")

    def test_vllm_parser_name_aliases(self):
        # vLLM 把 glm45/glm47 注册到同一个 parser, 从 vLLM 命令抄过来要能用
        self.assertEqual(resolve_tool_parser("glm47", PLAIN_TOK), "glm4_moe")
        self.assertEqual(resolve_tool_parser("glm45", PLAIN_TOK), "glm4_moe")
        profile = resolve_model_profile("glm47", GLM4_MOE_TOK,
                                        _Config("glm4_moe_lite"))
        self.assertEqual(profile.tool_parser, "glm4_moe")
        # 别名归一化后与原生协议一致: 不该报"配置与模型家族冲突"
        self.assertIsNone(profile.fallback_parser)

    def test_profile_for_glm4_moe(self):
        profile = resolve_model_profile("auto", GLM4_MOE_TOK,
                                        _Config("glm4_moe_lite"))
        self.assertEqual(profile.native_parser, "glm4_moe")
        # 模板用 `tc.arguments.items()` 渲染: arguments 必须是对象
        self.assertTrue(profile.arguments_as_dict)
        # 工具结果窗口与 Qwen 系是同一对 <tool_response>, 且走通用渲染分支
        self.assertEqual(resolve_tool_result_wrapper(GLM4_MOE_TOK),
                         ToolResultWrapper("<tool_response>", "</tool_response>"))
        self.assertEqual(profile.tool_protocol, "openai")

    def test_tool_result_wrapper_detection(self):
        self.assertEqual(resolve_tool_result_wrapper(HERMES_TOK),
                         ToolResultWrapper("<tool_response>", "</tool_response>"))
        self.assertEqual(resolve_tool_result_wrapper(MISTRAL_TOK),
                         ToolResultWrapper("[TOOL_RESULTS]", "[/TOOL_RESULTS]"))
        # GLM-4 没有闭合标记: 窗口到下一个角色标记为止
        glm4_wrapper = resolve_tool_result_wrapper(GLM4_TOK)
        self.assertEqual(glm4_wrapper.open_marker, "<|observation|>")
        self.assertIsNone(glm4_wrapper.close_marker)
        self.assertIn("<|assistant|>", glm4_wrapper.terminators)
        self.assertIsNone(resolve_tool_result_wrapper(PLAIN_TOK))

    def test_profile_sets_glm4_tool_protocol(self):
        profile = resolve_model_profile("auto", GLM4_TOK, _Config("glm4"))
        self.assertEqual(profile.native_parser, "glm4")
        self.assertEqual(profile.tool_protocol, "glm4")
        # 显式配置别的解析器时, 输出解析可以冲突, 但渲染协议必须仍是 GLM 原生
        profile = resolve_model_profile("hermes", GLM4_TOK, _Config("glm4"))
        self.assertEqual(profile.tool_parser, "hermes")
        self.assertEqual(profile.fallback_parser, "glm4")
        self.assertEqual(profile.tool_protocol, "glm4")
        self.assertEqual(resolve_model_profile("auto", HERMES_TOK).tool_protocol,
                         "openai")

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


class Glm4ToolCallTest(unittest.TestCase):
    """GLM-4-0414 的 `name\\n{json}` 工具调用协议(实测输出形态)."""

    # 真实模型输出(以 <|observation|> eos 结束; eos 不入 output_text)
    GLM4_CALL = 'get_weather\n{"city": "北京"}'
    SCHEMAS_GLM4 = {"get_weather": {"type": "object", "properties": {
        "city": {"type": "string"}}}}

    def test_real_output_parses_with_verbatim_arguments(self):
        calls = parse_glm4_tool_calls(self.GLM4_CALL, self.SCHEMAS_GLM4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        # arguments 逐位保留模型原始 JSON 子串: 回填后模板渲染的
        # `get_weather\n{"city": "北京"}` 才能与生成流逐位一致(LCP 整段命中)
        self.assertEqual(calls[0]["function"]["arguments"], '{"city": "北京"}')
        self.assertEqual(calls[0]["type"], "function")

    def test_parse_output_and_content_policy(self):
        calls, content = parse_model_output(self.GLM4_CALL, "glm4", None,
                                            schemas=self.SCHEMAS_GLM4)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(content)  # 纯工具调用没有前导正文
        self.assertEqual(parse_output_tool_calls(self.GLM4_CALL, "glm4",
                                                 self.SCHEMAS_GLM4)[0]
                         ["function"]["name"], "get_weather")

    def test_prefix_text_is_kept_for_kv_alignment(self):
        calls, content = parse_model_output(
            "让我查一下天气\n" + self.GLM4_CALL, "glm4", None,
            schemas=self.SCHEMAS_GLM4)
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, "让我查一下天气\n")

    def test_normal_answer_is_not_a_tool_call(self):
        # 实测第二轮回答(以 <|user|> eos 结束)
        answer = "\n根据您的查询，北京今天的天气情况是晴，温度为28度。"
        calls, content = parse_model_output(answer, "glm4", None,
                                            schemas=self.SCHEMAS_GLM4)
        self.assertEqual(calls, [])
        self.assertEqual(content, answer.strip())

    def test_call_acceptance_rules(self):
        # 形态 1(输出开头): 模型直接以 `name\n{json}` 开头, 接受
        self.assertEqual(len(parse_glm4_tool_calls(
            'answer\n{"value": 42}', self.SCHEMAS_GLM4)), 1)
        # 正文中间的同形排版(非白名单名): 不接受, 避免把计划 JSON 误判成调用
        self.assertEqual(parse_glm4_tool_calls(
            'The result is\nanswer\n{"value": 42}', self.SCHEMAS_GLM4), [])
        # 正文中间但名字确实是本次声明的工具: 接受
        self.assertEqual(len(parse_glm4_tool_calls(
            'I will check.\nget_weather\n{"city": "北京"}',
            self.SCHEMAS_GLM4)), 1)

    def test_role_marker_form_from_real_research_output(self):
        # GLM-4 真实输出: 决策正文之后跟 `<|assistant|>函数名\n{json}`(模型卡的
        # 参考实现正是按 `<|assistant|>` split 后逐段解析)。未声明的函数名
        # (research)也要在角色标记锚点上被认出来 —— 否则客户端拿不到 tool_calls,
        # 主 agent 会一直空转到 turn budget 耗尽。
        text = ('After inspecting S1, I found nothing.\n'
                '<|assistant|>research\n'
                '{"query": "q", "source_ids": ["S1"]}')
        calls, content = parse_model_output(text, "glm4", None,
                                            schemas={"search": {}, "read": {}})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "research")
        self.assertEqual(calls[0]["function"]["arguments"],
                         '{"query": "q", "source_ids": ["S1"]}')
        self.assertEqual(content, "After inspecting S1, I found nothing.\n")

    def test_malformed_arguments_are_not_a_call(self):
        # JSON 没写完: 宁可留在 content, 也不抛出客户端无法执行的假调用
        calls, content = parse_model_output('get_weather\n{"city": ',
                                            "glm4", None,
                                            schemas=self.SCHEMAS_GLM4)
        self.assertEqual(calls, [])
        self.assertEqual(content, 'get_weather\n{"city":')

    def test_multiple_calls(self):
        text = ('get_weather\n{"city": "北京"}\n'
                'research\n{"query": "q"}')
        calls = parse_glm4_tool_calls(
            text, {"get_weather": {}, "research": {}})
        self.assertEqual([c["function"]["name"] for c in calls],
                         ["get_weather", "research"])
        self.assertEqual(calls[1]["function"]["arguments"], '{"query": "q"}')

    def test_schema_repairs_stringified_array(self):
        calls = parse_glm4_tool_calls(
            'research\n{"query": "q", "source_ids": "[\'S1\']"}', SCHEMAS)
        args = json.loads(calls[0]["function"]["arguments"])
        self.assertEqual(args["source_ids"], ["S1"])

    def test_conflicting_parser_falls_back_to_glm4(self):
        # 照抄别的模型的启动参数(hermes)时输出解析不丢: 按原生 glm4 兜底
        calls, content = parse_model_output(self.GLM4_CALL, "hermes", "glm4",
                                            schemas=self.SCHEMAS_GLM4)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertIsNone(content)
        self.assertEqual(parse_output_tool_calls(self.GLM4_CALL, "hermes"),
                         [])


class Glm4StreamSplitterTest(unittest.TestCase):
    def split(self, chunks):
        splitter = make_stream_splitter("glm4")
        events = []
        for chunk in chunks:
            events.extend(splitter.push(chunk))
        events.extend(splitter.flush())
        return events

    def test_stream_splits_call(self):
        events = self.split(["get_", "weather", "\n", '{"city": ', '"北京"}'])
        self.assertEqual([k for k, _ in events], ["tool_call"])
        call = events[0][1]
        self.assertEqual(call["function"]["name"], "get_weather")
        self.assertEqual(call["function"]["arguments"], '{"city": "北京"}')

    def test_stream_normal_answer_passes_through(self):
        events = self.split(["\n根据您的查询，", "北京今天是晴天。"])
        self.assertEqual([k for k, _ in events], ["content", "content"])
        self.assertEqual("".join(p for _, p in events),
                         "\n根据您的查询，北京今天是晴天。")

    def test_stream_word_then_newline_is_not_a_call(self):
        events = self.split(["Sure", "\nHere is the answer"])
        self.assertEqual([k for k, _ in events], ["content"])
        self.assertEqual(events[0][1], "Sure\nHere is the answer")

    def test_incomplete_call_flushed_as_text(self):
        events = self.split(['get_weather\n{"city": '])
        self.assertEqual([k for k, _ in events], ["content"])
        self.assertEqual(events[0][1], 'get_weather\n{"city": ')

    def test_stream_multiple_calls(self):
        events = self.split(['get_weather\n{"city": "S"}\nresearch\n{"query": "q"}'])
        calls = [p for k, p in events if k == "tool_call"]
        self.assertEqual([c["function"]["name"] for c in calls],
                         ["get_weather", "research"])

    def test_tool_name_whitelist_suppresses_false_positive(self):
        # 正文里"一行单词 + JSON 对象"的同形排版(非白名单名, 无角色标记): 按正文输出
        splitter = make_stream_splitter("glm4", {"call_subagent"})
        text = 'The result is\nanswer\n{"value": 42}'
        events = list(splitter.push(text)) + list(splitter.flush())
        self.assertEqual([k for k, _ in events], ["content"])
        self.assertEqual("".join(p for _, p in events), text)

    def test_stream_role_marker_call_after_prose(self):
        # 真实 research 形态: 先正文, 再 `<|assistant|>name\n{json}`(名字未声明)
        splitter = make_stream_splitter("glm4", {"delegate"})
        chunks = ["After S1, I will delegate.", "<|assistant|>",
                  "research\n", '{"query": "q"}']
        events = []
        for chunk in chunks:
            events.extend(splitter.push(chunk))
        events.extend(splitter.flush())
        self.assertEqual([k for k, _ in events], ["content", "tool_call"])
        self.assertEqual(events[0][1], "After S1, I will delegate.")
        self.assertEqual(events[1][1]["function"]["name"], "research")

    def test_stream_role_marker_before_plain_text_passes_through(self):
        # `<|assistant|>` 后面不是调用形态(普通文本)时不能一直扣留到 flush
        splitter = make_stream_splitter("glm4", {"delegate"})
        events = list(splitter.push("<|assistant|>Bon")) + \
            list(splitter.push("jour, voici la réponse.")) + list(splitter.flush())
        self.assertEqual([k for k, _ in events], ["content"])
        self.assertEqual(events[0][1], "<|assistant|>Bonjour, voici la réponse.")

    def test_tool_name_whitelist_allows_real_call(self):
        splitter = make_stream_splitter("glm4", {"call_subagent"})
        events = list(splitter.push('call_subagent\n{"task": "q"}')) + \
            list(splitter.flush())
        self.assertEqual([k for k, _ in events], ["tool_call"])
        self.assertEqual(events[0][1]["function"]["name"], "call_subagent")


class Glm4MoeToolCallTest(unittest.TestCase):
    """GLM-4.7-Flash 的 XML 工具调用协议(实测输出形态, vLLM 的 glm45/glm47)."""

    # 本机模型真实输出(enable_thinking=False; 结尾 <|observation|> 是 eos)
    GLM4_MOE_CALL = ("<tool_call>call_subagent<arg_key>task</arg_key>"
                     "<arg_value>Find the current population of Reykjavik."
                     "</arg_value></tool_call>")
    SCHEMAS_GLM4_MOE = {"call_subagent": {"type": "object", "properties": {
        "task": {"type": "string"}}}}

    def test_real_output_parses_with_verbatim_argument_values(self):
        text = ("I'll delegate a research task.<tool_call>call_subagent"
                "<arg_key>task</arg_key><arg_value>Find the current population "
                "of Reykjavik, Iceland.</arg_value></tool_call><|observation|>")
        calls, content = parse_model_output(text, "glm4_moe", None,
                                            schemas=self.SCHEMAS_GLM4_MOE)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "call_subagent")
        self.assertEqual(calls[0]["type"], "function")
        # 值按原始文本保留字符串: 模板对字符串参数原样渲染, 回填后的 prompt 才能
        # 与生成流逐位一致(实测 round-trip LCP 整段命中)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"task": "Find the current population of Reykjavik, Iceland."})
        # 调用前的正文原样返回(不 strip), 供客户端回填、KV 前缀覆盖整段输出
        self.assertEqual(content, "I'll delegate a research task.")

    def test_call_without_arguments(self):
        calls = parse_glm4_moe_tool_calls("<tool_call>get_time</tool_call>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_time")
        self.assertEqual(calls[0]["function"]["arguments"], "{}")

    def test_multiple_calls_and_surrounding_prose(self):
        text = (self.GLM4_MOE_CALL
                + "\nmore prose\n"
                + '<tool_call>search<arg_key>q</arg_key><arg_value>r</arg_value>'
                  '</arg_key></tool_call>')
        calls = parse_glm4_moe_tool_calls(text, {"call_subagent": {}, "search": {}})
        self.assertEqual([c["function"]["name"] for c in calls],
                         ["call_subagent", "search"])
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"q": "r"})

    def test_think_block_is_not_part_of_arguments(self):
        # thinking 开启时输出以推理正文开头(模板已预开 <think>), 解析器只看工具块
        text = "<think>the user wants ...</think>Let me delegate." + self.GLM4_MOE_CALL
        calls, content = parse_model_output(text, "glm4_moe", None,
                                            schemas=self.SCHEMAS_GLM4_MOE)
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, "Let me delegate.")

    def test_content_cleaner_drops_call_blocks(self):
        # 输入是引擎给解析器的可见输出: 结尾的 <|observation|> 是 eos, 不入 output_text
        text = "答案在下面\n" + self.GLM4_MOE_CALL
        self.assertEqual(clean_output_content(text, "glm4_moe"), "答案在下面")
        # 普通回答不受影响(没有 <tool_call> 标记)
        answer = "雷克雅未克人口约 14 万。"
        self.assertEqual(clean_output_content(answer, "glm4_moe"), answer)

    def test_schema_repairs_stringified_array(self):
        calls = parse_glm4_moe_tool_calls(
            '<tool_call>research<arg_key>query</arg_key><arg_value>q</arg_value>'
            '<arg_key>source_ids</arg_key>'
            "<arg_value>['S1']</arg_value></tool_call>", SCHEMAS)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"])["source_ids"],
                         ["S1"])

    def test_conflicting_parser_falls_back_to_glm4_moe(self):
        # 照抄别的模型的启动参数(hermes)时不能静默丢调用: hermes 会把 XML 块
        # 整段删掉, 必须靠原生协议兜底解析出来
        self.assertEqual(parse_output_tool_calls(self.GLM4_MOE_CALL, "hermes"), [])
        self.assertEqual(clean_output_content(self.GLM4_MOE_CALL, "hermes"), "")
        calls, content = parse_model_output(self.GLM4_MOE_CALL, "hermes", "glm4_moe",
                                            schemas=self.SCHEMAS_GLM4_MOE)
        self.assertEqual(calls[0]["function"]["name"], "call_subagent")
        self.assertIsNone(content)


class Glm4MoeStreamSplitterTest(unittest.TestCase):
    def split(self, chunks):
        splitter = make_stream_splitter("glm4_moe")
        events = []
        for chunk in chunks:
            events.extend(splitter.push(chunk))
        events.extend(splitter.flush())
        return events

    def test_stream_splits_call(self):
        events = self.split(["I'll delegate.", "<tool_call>call_subagent<arg_key>",
                             "task</arg_key><arg_value>Find it",
                             "</arg_value></tool_call>"])
        self.assertEqual([k for k, _ in events], ["content", "content", "tool_call"])
        # 正文可能被"半截标记"扣留逻辑切成多个 content 事件, 拼起来必须原样
        self.assertEqual("".join(p for k, p in events if k == "content"),
                         "I'll delegate.")
        call = events[-1][1]
        self.assertEqual(call["function"]["name"], "call_subagent")
        self.assertEqual(json.loads(call["function"]["arguments"]),
                         {"task": "Find it"})

    def test_marker_never_leaks_into_content(self):
        # 标记被拆成多 chunk 到达时不能提前当普通文本输出
        events = self.split(["<tool", "_call>get_time</tool", "_call>"])
        self.assertEqual([k for k, _ in events], ["tool_call"])

    def test_normal_answer_passes_through(self):
        events = self.split(["雷克雅未克人口", "约 14 万。"])
        self.assertEqual([k for k, _ in events], ["content", "content"])
        self.assertEqual("".join(p for _, p in events), "雷克雅未克人口约 14 万。")

    def test_incomplete_call_flushed_as_text(self):
        events = self.split(["<tool_call>call_subagent<arg_key>task</arg_key>"])
        self.assertEqual([k for k, _ in events], ["content"])
        self.assertIn("call_subagent", events[0][1])

    def test_stream_multiple_calls(self):
        events = self.split([Glm4MoeToolCallTest.GLM4_MOE_CALL,
                             "<tool_call>search<arg_key>q</arg_key>"
                             "<arg_value>r</arg_value></tool_call>"])
        calls = [p for k, p in events if k == "tool_call"]
        self.assertEqual([c["function"]["name"] for c in calls],
                         ["call_subagent", "search"])


class Glm4MoeMessageRenderingTest(unittest.TestCase):
    """GLM-4.7-Flash 的 OpenAI 消息渲染: 模板原生认 tool_calls/tool, 但 arguments
    必须是对象、content=null 必须归一(否则渲染出字面量 None)。"""

    SYSTEM = "You are a research agent. Delegate work with the tool."
    USER = "Delegate a sub agent to find the population of Reykjavik."
    # 本机模型实测原始输出(enable_thinking=False) = 调用前正文 + 工具调用 + eos
    PROSE = "I'll delegate a research task to find the population of Reykjavik."
    GEN = (PROSE + "<tool_call>call_subagent<arg_key>task</arg_key>"
           "<arg_value>Find the current population of Reykjavik, Iceland."
           "</arg_value></tool_call><|observation|>")
    SUB_ANSWER = "Reykjavik has about 140000 residents."

    def _profile(self):
        from lminfer.model_adapters import resolve_model_profile
        return resolve_model_profile("auto", GLM4_MOE_TOK, _Config("glm4_moe_lite"))

    def _messages(self, content=None):
        from lminfer.schemas import ChatMessage
        return [
            ChatMessage(role="system", content=self.SYSTEM),
            ChatMessage(role="user", content=self.USER),
            ChatMessage(role="assistant", content=content, tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "call_subagent",
                             "arguments": json.dumps(
                                 {"task": "Find the current population of "
                                          "Reykjavik, Iceland."})}}]),
            ChatMessage(role="tool", tool_call_id="call_1", content=self.SUB_ANSWER),
        ]

    def test_arguments_string_becomes_object_and_null_content_is_empty(self):
        from lminfer.server import adapt_openai_messages
        out = adapt_openai_messages(self._messages(), self._profile())
        # arguments(JSON 字符串)还原成对象: 模板用 `tc.arguments.items()` 渲染,
        # 字符串会直接抛 UndefinedError(→ 400)
        self.assertEqual(out[2]["tool_calls"][0]["function"]["arguments"],
                         {"task": "Find the current population of Reykjavik, Iceland."})
        # OpenAI 协议里工具调用消息的 content 是 null: 归一成空串, 否则模板渲染出
        # 字面量 "None"
        self.assertEqual(out[2]["content"], "")
        self.assertEqual(out[3]["role"], "tool")
        self.assertEqual(out[3]["content"], self.SUB_ANSWER)

    def test_renders_with_real_tokenizer_and_round_trips_token_exact(self):
        import os
        from lminfer.server import adapt_openai_messages
        model = "/public/home/xiaoxunpeng/Models/GLM-4.7-Flash"
        if not os.path.isdir(model):
            self.skipTest("GLM-4.7-Flash not present locally")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)

        def ids(messages):
            text = tok.apply_chat_template(adapt_openai_messages(messages, profile),
                                           add_generation_prompt=True,
                                           enable_thinking=False, tokenize=False)
            return text, tok(text)["input_ids"]

        profile = self._profile()
        _, turn1 = ids(self._messages()[:2])
        # 客户端把模型的回复(含正文与 tool_calls)回填, 服务器渲染出的第二轮 prompt
        text2, turn2 = ids(self._messages(content=self.PROSE))
        # 工具调用渲染成 XML 块, 工具结果是 <|observation|><tool_response>...</...>
        self.assertIn("<tool_call>call_subagent<arg_key>task</arg_key>", text2)
        self.assertIn(f"<|observation|><tool_response>{self.SUB_ANSWER}"
                      "</tool_response>", text2)
        self.assertNotIn("None", text2)
        # 第一轮 prompt 是第二轮 prompt 的前缀(带 tool_calls 的回合能整段复用 KV)
        self.assertEqual(turn2[:len(turn1)], turn1)
        # 回填出的"正文 + 工具调用 + <|observation|>"与模型自己生成的 token 逐位
        # 一致 —— 跨请求 KV 前缀复用(LCP)能覆盖整个工具调用回合
        gen_ids = tok(self.GEN)["input_ids"]
        self.assertEqual(turn2[len(turn1):len(turn1) + len(gen_ids)], gen_ids)


class Glm4MessageRenderingTest(unittest.TestCase):
    """OpenAI 消息 -> GLM 原生消息的渲染翻译(tool_calls/metadata + observation)."""

    def _messages(self):
        from lminfer.schemas import ChatMessage
        return [
            ChatMessage(role="user", content="北京天气?"),
            ChatMessage(role="assistant", content=None, tool_calls=[{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"city": "北京"}'}}]),
            ChatMessage(role="tool", tool_call_id="call_1", content="晴, 28 度"),
        ]

    def test_tool_calls_render_as_metadata_and_observation(self):
        from lminfer.server import adapt_glm4_messages
        out = adapt_glm4_messages(self._messages())
        self.assertEqual(out, [
            {"role": "user", "content": "北京天气?"},
            {"role": "assistant", "metadata": "get_weather",
             "content": '{"city": "北京"}'},
            {"role": "observation", "content": "晴, 28 度"},
        ])

    def test_arguments_are_rendered_verbatim(self):
        # content 必须原样是 arguments 字符串(模板直接 `{{ content }}` 渲染),
        # 任何 json.dumps 归一化都会让下一轮渲染多/少空格、LCP 提前断开
        from lminfer.server import adapt_glm4_messages
        out = adapt_glm4_messages(self._messages())
        self.assertEqual(out[1]["content"], '{"city": "北京"}')

    def test_renders_with_real_glm4_tokenizer_if_available(self):
        import os
        model = "/home/tanger/workspace/models/GLM-4-9B-0414"
        if not os.path.isdir(model):
            self.skipTest("GLM-4-9B-0414 not present locally")
        from transformers import AutoTokenizer
        from lminfer.server import adapt_glm4_messages
        tok = AutoTokenizer.from_pretrained(model)
        text = tok.apply_chat_template(adapt_glm4_messages(self._messages()),
                                       add_generation_prompt=True, tokenize=False)
        # 工具调用与工具结果都真实进入 prompt(直接透传 OpenAI 字段会渲染成 None)
        self.assertIn('<|assistant|>get_weather\n{"city": "北京"}', text)
        self.assertIn("<|observation|>\n晴, 28 度", text)
        self.assertNotIn("None", text)


if __name__ == "__main__":
    unittest.main()

