"""Real-GPU acceptance checks for the LMInfer vLLM backend.

Run with the vLLM-compatible environment:
  .venv/bin/python experiments/vllm_agent_kv_smoke.py --model /path/to/Qwen3-0.6B
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lminfer.config import EngineConfig, SamplingParams
from lminfer.kvcache import KVGraft, KVPrefix, slice_cache, tail_cache, rebase_rope_cache
from lminfer.vllm_engine import VLLMEngine
from lminfer.repair import repair_token_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    engine = VLLMEngine(EngineConfig(
        model=args.model, backend="vllm", max_model_len=1024,
        gpu_memory_utilization=args.gpu_memory_utilization,
        reuse_agent_kv_append=True, graft_rope_rebase=True,
        repair_window_begin=.15, repair_window_end=.2,
        enable_thinking=False, disable_log_stats=True))
    records = []
    try:
        sampling = SamplingParams(temperature=0, max_tokens=16)
        tokens = engine.tokenizer.encode(
            "The following is a list of facts. " * 20 + "The capital of France is")
        prompt = torch.tensor([tokens])
        full = engine._generate("full", prompt, sampling)
        assert full.kv_cache.get_seq_length() == full.repair_stats["cached_sequence_tokens"]
        records.append({"case": "full", **full.repair_stats, "ttft_ms": full.ttft_ms})
        prefix = KVPrefix(tokens + full.output_tokens[:full.kv_cache.get_seq_length()-len(tokens)],
                          full.kv_cache, exact_prefix_len=full.exact_prefix_len)
        reused = engine._generate("prefix", prompt, sampling, reuse_prefixes=[prefix])
        assert reused.output_tokens == full.output_tokens, "Exact prefix changed greedy output"
        assert reused.reused_prompt_tokens == len(tokens)-1
        records.append({"case": "prefix", **reused.repair_stats, "ttft_ms": reused.ttft_ms})

        # Same-context grafts: numerical equivalence must hold before testing
        # the intentionally approximate cross-context case.
        spans = [(19, 79), (92, 142)]
        grafts = [KVGraft(p, tokens[p:e],
                         slice_cache(tail_cache(full.kv_cache, p, engine.model.config),
                                     e-p, engine.model.config), p) for p, e in spans]
        grafted = engine._generate("same-context-grafts", prompt, sampling, graft=grafts)
        assert grafted.output_tokens == full.output_tokens
        assert grafted.reused_prompt_tokens > 0
        assert grafted.repair_stats["stages"] == 3
        records.append({"case": "same_context_grafts", **grafted.repair_stats,
                        "ttft_ms": grafted.ttft_ms})

        # Real sub generation followed by insertion at a different position.
        sub_prompt = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a long list of facts about Paris and France."}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(sub_prompt, "input_ids"):
            sub_prompt = sub_prompt.input_ids
        sub = engine._generate("sub", torch.tensor([sub_prompt]),
                               SamplingParams(temperature=0, max_tokens=40))
        length = sub.kv_cache.get_seq_length() - len(sub_prompt)
        assert length > 4
        body = sub.output_tokens[:length]
        intro = engine.tokenizer.encode("Use this researcher report to answer the question. Report: ")
        suffix = engine.tokenizer.encode("\nQuestion: What is the capital of France? Answer:")
        target = intro + body + suffix
        source = tail_cache(sub.kv_cache, len(sub_prompt), engine.model.config)
        graft = KVGraft(len(intro), body, source, len(sub_prompt))
        result = engine._generate("cross-context-graft", torch.tensor([target]), sampling,
                                  graft=[graft])
        assert result.reused_prompt_tokens > 0
        assert result.exact_prefix_len < len(target)
        left, right = repair_token_counts(length, .15, .2)
        rebased = rebase_rope_cache(source, len(sub_prompt), len(intro), engine.model.config)
        for actual_layer, source_layer in zip(result.kv_cache.layers, rebased.layers):
            torch.testing.assert_close(
                actual_layer.keys[:, :, len(intro)+left:len(intro)+length-right],
                source_layer.keys[:, :, left:length-right], rtol=0, atol=0)
            torch.testing.assert_close(
                actual_layer.values[:, :, len(intro)+left:len(intro)+length-right],
                source_layer.values[:, :, left:length-right], rtol=0, atol=0)
        records.append({"case": "cross_context_graft", **result.repair_stats,
                        "ttft_ms": result.ttft_ms, "text": result.output_text})
        engine.config.repair_mode = "exact"
        exact = engine._generate("exact", torch.tensor([target]), sampling, graft=[graft])
        assert exact.reused_prompt_tokens == 0
        assert exact.repair_stats["exact"]
        records.append({"case": "exact_fallback", **exact.repair_stats, "ttft_ms": exact.ttft_ms})
        engine.config.repair_mode = "window"

        # Exercise chunked prefill, including imported prefixes beyond a chunk.
        long_tokens = engine.tokenizer.encode("A simple factual statement. " * 120)
        long_full = engine._generate("chunked-full", torch.tensor([long_tokens]), sampling)
        long_prefix = KVPrefix(long_tokens[:531],
                               slice_cache(long_full.kv_cache, 531, engine.model.config))
        long_reused = engine._generate("chunked-prefix", torch.tensor([long_tokens]), sampling,
                                      reuse_prefixes=[long_prefix])
        assert long_full.output_tokens == long_reused.output_tokens
        assert long_reused.reused_prompt_tokens == 531
        records.append({"case": "chunked_prefix", **long_reused.repair_stats,
                        "ttft_ms": long_reused.ttft_ms})

        # The public agent API must discover the sub body from its tool response,
        # persist only valid KV, and work on the engine's inference thread.
        from fastapi.testclient import TestClient
        from lminfer.server import create_app
        with TestClient(create_app(engine)) as client:
            def chat(messages, trace, session=None, max_tokens=40):
                response = client.post("/v1/chat/completions", json={
                    "messages": messages, "mode": "agent", "trace": trace,
                    "session_id": session, "max_tokens": max_tokens,
                    "temperature": 0, "enable_thinking": False})
                assert response.status_code == 200, response.text
                return response.json()
            messages = [{"role": "user", "content": "We are researching Paris. Say ready."}]
            first = chat(messages, ["main"], max_tokens=8)
            session = first["session_id"]
            sub_response = chat(
                [{"role": "user", "content": "Write a long list of facts about Paris and France."}],
                ["main", "researcher"], session)
            body_text = sub_response["choices"][0]["message"]["content"]
            second_sub = chat(
                [{"role": "user", "content": "Write a long list of facts about Berlin and Germany."}],
                ["main", "researcher2"], session)
            resumed_messages = messages + [
                first["choices"][0]["message"],
                {"role": "tool", "tool_call_id": "call_report", "content": body_text},
                {"role": "tool", "tool_call_id": "call_report2",
                 "content": second_sub["choices"][0]["message"]["content"]},
                {"role": "user", "content": "What is the capital of France?"}]
            # Observe actual GPU tensor lifetime while the HTTP request is
            # still running, rather than checking only the store after decode.
            import weakref
            from unittest.mock import patch
            from lminfer.kvcache import SessionKVStore
            watched = {}
            original_take = SessionKVStore.take_main_reuse
            original_stage = engine._stage

            def watch_take(store, sid, trace, prompt_tokens, **kwargs):
                prefixes, grafts = original_take(store, sid, trace, prompt_tokens, **kwargs)
                source_tensors = [tensor for p in prefixes if p.kind == "sub"
                                  for layer in p.cache.layers for tensor in (layer.keys, layer.values)]
                graft_tensors = [tensor for g in grafts for layer in g.cache.layers
                                 for tensor in (layer.keys, layer.values)]
                watched.update(
                    sources=[weakref.ref(t) for t in source_tensors],
                    grafts=[weakref.ref(t) for t in graft_tensors],
                    bytes=sum(t.numel() * t.element_size() for t in source_tensors + graft_tensors),
                    prompt_len=len(prompt_tokens), final_checked=False)
                return prefixes, grafts

            def watch_stage(stage_tokens, prefix, sampling, **kwargs):
                assert all(ref() is None for ref in watched["sources"]), "Original sub KV still live"
                if len(stage_tokens) == watched["prompt_len"]:
                    assert all(ref() is None for ref in watched["grafts"]), "Consumed graft KV still live"
                    watched["final_checked"] = True
                return original_stage(stage_tokens, prefix, sampling, **kwargs)

            with patch.object(SessionKVStore, "take_main_reuse", watch_take), \
                    patch.object(engine, "_stage", watch_stage):
                resumed = chat(resumed_messages, ["main", "researcher", "main"], session, 16)
            assert watched["final_checked"] and watched["bytes"] > 0
            records.append({"case": "sub_kv_freed_before_main_decode", "checks": "passed",
                            "released_snapshot_bytes": watched["bytes"]})
            assert resumed["repair_stats"]["graft_tokens"] > 0, resumed
            assert resumed["repair_stats"]["stages"] == 3, resumed
            assert resumed["exact_prefix_len"] < resumed["usage"]["prompt_tokens"]
            records.append({"case": "http_two_subagent_roundtrip", **resumed["repair_stats"]})
            next_round = chat(
                resumed_messages + [resumed["choices"][0]["message"],
                                    {"role": "user", "content": "Repeat the capital in one word."}],
                ["main"], session, 8)
            assert next_round["reused_prompt_tokens"] <= resumed["exact_prefix_len"]
            assert next_round["repair_stats"]["graft_tokens"] == 0
            records.append({"case": "http_multiround_exact_boundary", **next_round["repair_stats"]})
            # Streaming text must agree with nonstreaming text, including UTF-8.
            stream_request = {"prompt": "请用中文回答：法国的首都是", "max_tokens": 12,
                              "temperature": 0}
            plain = client.post("/v1/completions", json=stream_request)
            assert plain.status_code == 200, plain.text
            streamed = client.post("/v1/completions", json={**stream_request, "stream": True})
            chunks = [json.loads(line[6:]) for line in streamed.text.splitlines()
                      if line.startswith("data: ") and line != "data: [DONE]"]
            assert all("error" not in chunk for chunk in chunks), chunks
            assert "".join(c["choices"][0]["text"] for c in chunks) == plain.json()["choices"][0]["text"]
            failed = client.post("/v1/completions", json={
                **stream_request, "stream": True, "max_tokens": 1024})
            assert '"inference_error"' in failed.text and "[DONE]" in failed.text
            released = client.post("/v1/release", json={"session_id": session})
            assert released.json()["state"]
            stats = client.get("/v1/stats").json()
            assert stats["backend"] == "vllm" and stats["execution_max_num_seqs"] == 1
            records.append({"case": "http_stream_release_errors", "checks": "passed"})
        import vllm
        import transformers
        report = {"model": args.model, "vllm_version": vllm.__version__,
                  "transformers_version": transformers.__version__,
                  "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(),
                  "checks": "passed", "records": records}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
