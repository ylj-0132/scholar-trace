from __future__ import annotations

import sys
import types
import logging

import pytest

import deep_research.llm as llm_module
from deep_research.llm import LLMClient, parse_json_lenient
from deep_research.paper_agent_runtime import ModelCallRecorder


def test_parse_json_lenient_handles_fenced_json() -> None:
    parsed = parse_json_lenient(
        """```json
{"items": [{"status": "maybe"}]}
```"""
    )

    assert parsed == {"items": [{"status": "maybe"}]}


def test_gemini_3_uses_model_default_temperature() -> None:
    client = LLMClient("gemini/gemini-3.1-pro-preview")

    kwargs = client._build_kwargs("prompt", system=None, temperature=None)

    assert "temperature" not in kwargs


def test_non_gemini_3_uses_low_temperature() -> None:
    client = LLMClient("deepseek/deepseek-chat")

    kwargs = client._build_kwargs("prompt", system=None, temperature=None)

    assert kwargs["temperature"] == 0.1


def test_multimodal_kwargs_contain_prompt_and_all_images() -> None:
    client = LLMClient("test/model")

    kwargs = client._build_kwargs(
        "inspect these pages",
        system="paper evidence",
        temperature=None,
        image_urls=[
            "data:image/png;base64,one",
            "data:image/png;base64,two",
        ],
    )

    content = kwargs["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "inspect these pages"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"] == {
        "url": "data:image/png;base64,one",
        "detail": "high",
    }
    assert content[2]["image_url"]["url"] == "data:image/png;base64,two"


def test_malformed_json_does_not_retry_the_model_call(monkeypatch: object) -> None:
    calls = 0

    class FakeMessage:
        content = "malformed"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def completion(**kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(completion=completion),
    )

    def fail_parse(raw: str) -> dict[str, object]:
        raise ValueError("malformed JSON")

    monkeypatch.setattr(llm_module, "parse_json_lenient_with_metadata", lambda raw: fail_parse(raw))

    with pytest.raises(ValueError, match="malformed JSON"):
        LLMClient("test/model", max_retries=1).complete_json("prompt")

    assert calls == 1


def test_recorded_completion_captures_raw_parsed_usage_and_attempts(
    monkeypatch: object,
) -> None:
    class FakeMessage:
        content = '{"kind": "DECIDE"}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = types.SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
        )

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(completion=lambda **kwargs: FakeResponse()),
    )
    recorder = ModelCallRecorder(model="test/model", temperature=0.1)
    call_id = recorder.start_call(
        role="master",
        round_number=1,
        task_question=None,
        system_prompt="system",
        user_prompt="prompt",
        image_inputs=(),
    )

    result = LLMClient("test/model", max_retries=1).complete_json(
        "prompt",
        system="system",
        recorder=recorder,
        call_id=call_id,
    )

    record = recorder.records[0]
    assert result == {"kind": "DECIDE"}
    assert record.raw_response == '{"kind": "DECIDE"}'
    assert record.parsed_response == {"kind": "DECIDE"}
    assert record.json_repaired is False
    assert record.attempt_count == 1
    assert record.prompt_tokens == 11
    assert record.completion_tokens == 7
    assert record.total_tokens == 18
    assert record.latency_seconds is not None


def test_recorded_response_redacts_api_key_before_parsing(
    monkeypatch: object,
) -> None:
    api_key = "sk-test-secret-value"
    calls = 0

    class FakeMessage:
        content = '{"echo": "sk-test-secret-value"}'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def completion(**kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(completion=completion),
    )
    recorder = ModelCallRecorder(model="test/model", temperature=0.1)
    call_id = recorder.start_call(
        role="master",
        round_number=1,
        task_question=None,
        system_prompt="system",
        user_prompt="prompt",
        image_inputs=(),
    )

    LLMClient("test/model", api_key=api_key).complete_json(
        "prompt",
        recorder=recorder,
        call_id=call_id,
    )

    record = recorder.records[0]
    assert calls == 1
    assert record.raw_response == '{"echo": "[REDACTED]"}'
    assert record.parsed_response == {"echo": "[REDACTED]"}
    assert api_key not in record.raw_response
    assert api_key not in str(record.parsed_response)
    assert api_key not in str(record)
    assert "[REDACTED]" in record.raw_response
    assert "[REDACTED]" in str(record.parsed_response)


def test_recorded_repair_is_true_without_a_second_provider_call(
    monkeypatch: object,
) -> None:
    calls = 0

    class FakeMessage:
        content = "malformed"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def completion(**kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(completion=completion),
    )
    monkeypatch.setitem(
        sys.modules,
        "json_repair",
        types.SimpleNamespace(repair_json=lambda raw: '{"fixed": true}'),
    )
    recorder = ModelCallRecorder(model="test/model", temperature=0.1)
    call_id = recorder.start_call(
        role="locator",
        round_number=1,
        task_question="Find evidence.",
        system_prompt="system",
        user_prompt="prompt",
        image_inputs=(),
    )

    result = LLMClient("test/model", max_retries=1).complete_json(
        "prompt",
        recorder=recorder,
        call_id=call_id,
    )

    assert result == {"fixed": True}
    assert calls == 1
    assert recorder.records[0].json_repaired is True
    assert recorder.records[0].attempt_count == 1


def test_strict_json_rejects_repair_and_records_raw_response_once(
    monkeypatch: object,
) -> None:
    calls = 0

    class FakeMessage:
        content = '{"broken":'

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def completion(**kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse()

    monkeypatch.setitem(sys.modules, "litellm", types.SimpleNamespace(completion=completion))
    monkeypatch.setitem(
        sys.modules,
        "json_repair",
        types.SimpleNamespace(repair_json=lambda raw: '{"fixed": true}'),
    )
    recorder = ModelCallRecorder(model="test/model", temperature=0.1)
    call_id = recorder.start_call(
        role="master",
        round_number=1,
        task_question=None,
        system_prompt="system",
        user_prompt="prompt",
        image_inputs=(),
    )

    with pytest.raises(ValueError, match="invalid_json"):
        LLMClient("test/model", allow_json_repair=False).complete_json(
            "prompt", recorder=recorder, call_id=call_id
        )

    record = recorder.records[0]
    assert calls == 1
    assert record.raw_response == '{"broken":'
    assert record.parsed_response is None
    assert record.json_repaired is False
    assert (record.error or "").startswith("invalid_json:")


def test_recorded_provider_failure_keeps_error_and_transport_attempts(
    monkeypatch: object,
) -> None:
    calls = 0

    def completion(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("provider down")

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(completion=completion),
    )
    recorder = ModelCallRecorder(model="test/model", temperature=0.1)
    call_id = recorder.start_call(
        role="evidence",
        round_number=1,
        task_question="Find evidence.",
        system_prompt="system",
        user_prompt="prompt",
        image_inputs=(),
    )

    with pytest.raises(RuntimeError, match="LLM call failed after retries"):
        LLMClient("test/model", max_retries=1).complete_json(
            "prompt",
            recorder=recorder,
            call_id=call_id,
        )

    assert calls == 2
    assert recorder.records[0].attempt_count == 2
    assert "provider down" in (recorder.records[0].error or "")


def test_api_key_is_redacted_from_error_log_and_record(
    monkeypatch: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    api_key = "sk-test-secret-value"
    calls = 0

    def completion(**kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"provider echoed {api_key}")

    monkeypatch.setitem(
        sys.modules,
        "litellm",
        types.SimpleNamespace(completion=completion),
    )
    recorder = ModelCallRecorder(model="test/model", temperature=0.1)
    call_id = recorder.start_call(
        role="master",
        round_number=1,
        task_question=None,
        system_prompt="system",
        user_prompt="prompt",
        image_inputs=(),
    )

    with caplog.at_level(logging.WARNING, logger=llm_module.log.name):
        with pytest.raises(RuntimeError) as raised:
            LLMClient(
                "test/model",
                api_key=api_key,
                max_retries=1,
            ).complete_json("prompt", recorder=recorder, call_id=call_id)

    record_error = recorder.records[0].error or ""
    assert calls == 2
    assert api_key not in record_error
    assert api_key not in caplog.text
    assert api_key not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "provider echoed [REDACTED]" in record_error
    assert "provider echoed [REDACTED]" in caplog.text
    assert "provider echoed [REDACTED]" in str(raised.value)
