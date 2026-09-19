from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from deep_research import cli, experiment, paper_agent_runtime as runtime
from deep_research.llm import LLMClient


@pytest.mark.parametrize("layout", ["standard", "cache-friendly"])
@pytest.mark.parametrize("images", [[], ["image-one", "image-two"]])
def test_layout_preserves_payload_images_and_records_actual_request(monkeypatch, layout, images):
    sent = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(
        completion=lambda **kwargs: (sent.append(kwargs), response)[1],
    ))
    recorder = runtime.ModelCallRecorder(prompt_layout=layout)
    client = LLMClient("test/model")
    for question in ["first question", "second question"]:
        prompt = json.dumps({
            "question": question,
            "state": {"findings": [{"id": "r1-t1-f1", "evidence": "K=1"}]},
            "instructions": ["Use only supplied evidence."],
            "required_json_shape": {"finding": "text"},
            "compact_page_index": [{"page_number": 1, "preview": "method"}],
        }, ensure_ascii=False, indent=2)
        runtime._invoke_model(
            llm=client, prompt=prompt, system_prompt="same role",
            recorder=recorder, role="evidence", round_number=1,
            task_question=question, image_inputs=(), image_urls=images,
        )
        actual = sent[-1]["messages"][1]["content"]
        parts = actual if isinstance(actual, list) else [{"type": "text", "text": actual}]
        text = "".join(item["text"] for item in parts if item["type"] == "text")
        assert json.loads(text) == json.loads(prompt)
        assert [item["image_url"]["url"] for item in parts if item["type"] == "image_url"] == images
        assert recorder.records[-1].user_prompt == text
        assert recorder.records[-1].prompt_layout == layout
        if layout == "standard":
            assert text == prompt
            assert "extra_body" not in sent[-1]
            assert sent[-1]["messages"][0]["content"] == "same role"
            assert all("prompt_cache_breakpoint" not in item for item in parts)
        else:
            assert sent[-1]["extra_body"]["prompt_cache_options"] == {"mode": "explicit"}
            assert sent[-1]["messages"][0]["content"] == [{
                "type": "text", "text": "same role", "prompt_cache_breakpoint": {"mode": "explicit"},
            }]
            assert parts[0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
            assert '"compact_page_index"' in parts[0]["text"]
            assert '"question"' not in parts[0]["text"]
            assert '"question"' in parts[1]["text"]
            assert all("prompt_cache_breakpoint" not in item for item in parts[1:])
            assert recorder.records[-1].prompt_cache_prefix_chars == len(parts[0]["text"])
            assert recorder.records[-1].prompt_layout_version == "explicit-breakpoints-v3"
    if layout == "cache-friendly":
        assert recorder.records[0].user_prompt.split('"question"')[0] == recorder.records[1].user_prompt.split('"question"')[0]
    assert len(sent) == 2  # No response replay, even with a common prefix.


@pytest.mark.parametrize("details,expected", [
    ({"cached_tokens": 8, "cache_write_tokens": 2}, (8, 2)),
    (SimpleNamespace(cached_tokens=0, cache_write_tokens=0), (0, 0)),
    (None, (None, None)),
    ({"cached_tokens": True, "cache_write_tokens": -1}, (None, None)),
])
@pytest.mark.parametrize("as_dict", [False, True])
def test_provider_cache_usage_is_recorded_without_inventing_zero(monkeypatch, details, expected, as_dict):
    usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
             "prompt_tokens_details": details}
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))],
        usage=usage if as_dict else SimpleNamespace(**usage),
    )
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=lambda **kwargs: response))
    recorder = runtime.ModelCallRecorder()
    runtime._invoke_model(llm=LLMClient("test/model"), prompt="{}", system_prompt="role",
                          recorder=recorder, role="master", round_number=1,
                          task_question=None, image_inputs=())
    call = recorder.records[0]
    assert (call.cached_prompt_tokens, call.cache_write_prompt_tokens) == expected
    assert (call.prompt_tokens, call.completion_tokens, call.total_tokens) == (10, 4, 14)


@pytest.mark.parametrize("layout", ["standard", "cache-friendly"])
@pytest.mark.parametrize("context", ["rubric-union", "full-history"])
def test_cli_layout_reaches_runtime_and_saved_manifest(monkeypatch, tmp_path, layout, context):
    captured = {}
    class Trace:
        outcome = "DECIDE"
        assessment = "bounded finding"
        error = None
        def to_json(self):
            return '{"model_calls": []}'
    def run(**kwargs):
        captured.update(kwargs)
        return Trace()
    monkeypatch.setattr(experiment, "_role_clients", lambda _: ({role: object() for role in experiment.ROLE_MODELS}, ()))
    monkeypatch.setattr(experiment, "run_local_paper_agent", run)
    ticks = iter([10.0, 12.5])
    monkeypatch.setattr(experiment.time, "perf_counter", lambda: next(ticks))
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4")
    output = tmp_path / "run"
    assert cli.main(["audit", str(paper), "--output", str(output), "--prompt-layout", layout, "--reflection-context", context]) == 0
    assert captured["prompt_layout"] == layout
    assert captured["reflection_context_mode"] == context
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["prompt_layout"] == layout
    assert manifest["prompt_layout_version"] == ("explicit-breakpoints-v3" if layout == "cache-friendly" else "field-order-v1")
    assert json.loads((output / "result.json").read_text())["wall_seconds"] == 2.5


def test_invalid_layout_is_rejected_before_loading_paper_or_clients(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "extract_pdf_pages", lambda _: pytest.fail("must not read PDF"))
    with pytest.raises(ValueError, match="prompt_layout"):
        runtime.run_local_paper_agent(pdf_path=tmp_path / "paper.pdf", llm=object(), max_rounds=1, prompt_layout="off")
    monkeypatch.setattr(experiment, "_role_clients", lambda _: pytest.fail("must not construct clients"))
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4")
    assert experiment.run_audit(paper, tmp_path / "out", config=experiment.AuditConfig(prompt_layout="off")) == 1
    assert not (tmp_path / "out").exists()


def test_default_layout_is_standard():
    assert experiment.AuditConfig().prompt_layout == "standard"
    assert runtime.ModelCallRecorder().prompt_layout == "standard"
    assert cli.build_parser().parse_args(["audit", "paper.pdf", "--output", "out"]).prompt_layout == "standard"


@pytest.mark.parametrize("payload,has_prefix", [
    ({"state": {"instructions": "nested dynamic text"}}, False),
    ({"instructions": ["中文 \\\" { }\n"], "paper": {"title": "t"}}, True),
    ({}, False),
    ([], False),
])
def test_explicit_cache_handles_empty_static_or_dynamic_fields(monkeypatch, payload, has_prefix):
    sent = []
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=lambda **kw: (
        sent.append(kw), SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))]),
    )[1]))
    recorder = runtime.ModelCallRecorder(prompt_layout="cache-friendly")
    runtime._invoke_model(llm=LLMClient("openai/gpt-5.6-luna"), prompt=json.dumps(payload),
                          system_prompt="", recorder=recorder, role="master", round_number=1,
                          task_question=None, image_inputs=())
    parts = sent[0]["messages"][0]["content"]
    assert isinstance(parts, list)
    assert json.loads("".join(part["text"] for part in parts)) == payload
    assert sum("prompt_cache_breakpoint" in part for part in parts) == int(has_prefix)
    assert all(part["text"] for part in parts)


def test_cache_parameter_rejection_never_falls_back_to_ordinary_request(monkeypatch):
    sent = []
    def reject(**kwargs):
        sent.append(kwargs)
        raise ValueError("unsupported prompt_cache_options")
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=reject))
    recorder = runtime.ModelCallRecorder(prompt_layout="cache-friendly")
    with pytest.raises(RuntimeError, match="unsupported prompt_cache_options"):
        runtime._invoke_model(llm=LLMClient("openai/gpt-5.6-luna", max_retries=1),
                              prompt='{"instructions": ["Return JSON"], "state": {}}',
                              system_prompt="role", recorder=recorder, role="master", round_number=1,
                              task_question=None, image_inputs=())
    assert len(sent) == 2
    assert all(call["extra_body"]["prompt_cache_options"] == {"mode": "explicit"} for call in sent)
    assert recorder.records[0].attempt_count == 2
    assert "unsupported prompt_cache_options" in recorder.records[0].error


@pytest.mark.parametrize("layout", ["standard", "cache-friendly"])
def test_installed_sdk_preserves_cache_fields_in_http_body(monkeypatch, layout):
    # Exercise both installed adapters without a real credential or network call.
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import httpx
    import litellm
    from openai import OpenAI

    bodies = []
    def respond(request):
        assert str(request.url) == "https://cache-test.invalid/v1/chat/completions"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "offline-cache-test", "object": "chat.completion", "created": 0,
            "model": "gpt-5.6-luna",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 2, "total_tokens": 102,
                      "prompt_tokens_details": {"cached_tokens": 50, "cache_write_tokens": 10}},
        })

    completion = litellm.completion
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        with OpenAI(api_key="offline-placeholder", base_url="https://cache-test.invalid/v1", http_client=transport) as client:
            monkeypatch.setattr(litellm, "completion", lambda **kwargs: completion(**kwargs, client=client))
            recorder = runtime.ModelCallRecorder(prompt_layout=layout)
            runtime._invoke_model(
                llm=LLMClient("openai/gpt-5.6-luna", api_key="offline-placeholder", max_retries=0, temperature=1.0),
                prompt='{"state": {"round": 2}, "instructions": ["Return JSON"]}',
                system_prompt="Return JSON", recorder=recorder, role="master", round_number=2,
                task_question=None, image_inputs=(),
                image_urls=["data:image/png;base64,b2ZmbGluZQ=="],
            )
    assert len(bodies) == 1
    body = bodies[0]
    parts = body["messages"][1]["content"]
    if layout == "cache-friendly":
        assert body["prompt_cache_options"] == {"mode": "explicit"}
        assert body["messages"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
        assert parts[0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
        assert '"state"' not in parts[0]["text"]
        assert '"state"' in parts[1]["text"]
    else:
        assert "prompt_cache_options" not in body
        assert "prompt_cache_breakpoint" not in json.dumps(body)
    assert parts[-1]["image_url"]["url"] == "data:image/png;base64,b2ZmbGluZQ=="
    assert "prompt_cache_breakpoint" not in parts[-1]
    assert recorder.records[0].cached_prompt_tokens == 50
    assert recorder.records[0].cache_write_prompt_tokens == 10
