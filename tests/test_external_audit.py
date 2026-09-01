from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from threading import Barrier
from urllib.error import HTTPError

import pytest

from deep_research import external_audit


def test_public_runner_exposes_only_the_supported_formal_workflow() -> None:
    assert tuple(inspect.signature(external_audit.run_external_audit).parameters) == (
        "source_result",
        "output_dir",
        "client",
        "transport",
    )


def test_prompts_focus_on_consequential_experimental_evidence() -> None:
    planner = external_audit.EXTERNAL_PLANNER_SYSTEM_PROMPT
    auditor = external_audit.EXTERNAL_AUDITOR_SYSTEM_PROMPT

    assert "official implementation, default configuration, or metric computation" in planner
    assert "official benchmark split or test-use rule" in planner
    assert "official repository issue" in planner
    assert "site:github.com" in planner
    assert "A narrow corroborated fact does not by itself strengthen" in auditor
    assert "Official-repository issues are reproduction-risk signals" in auditor
    assert "model-generated retrieval summaries" in auditor


def test_parse_planner_questions_requires_unique_supported_questions() -> None:
    question = {
        "claim": "A claim",
        "decision_impact": "It could change the judgment.",
        "query": "method official implementation",
        "source_mode": "obsolete-field",
    }
    parsed = external_audit._parse_planner_questions({"questions": [question]})
    assert "source_mode" not in parsed[0]

    with pytest.raises(ValueError, match="duplicate_query"):
        external_audit._parse_planner_questions({"questions": [question, question]})


def test_external_result_requires_known_sources_for_material_findings() -> None:
    payload = {
        "assessment_delta": "narrowed",
        "revised_assessment": "The claim is narrower.",
        "findings": [{
            "claim": "A claim",
            "verdict": "qualified",
            "analysis": "The source limits it.",
            "source_ids": ["S2"],
        }],
        "unresolved_questions": [],
    }
    with pytest.raises(ValueError, match="unknown_source_id"):
        external_audit._parse_external_result(payload, valid_source_ids={"S1"})

    payload["findings"][0]["source_ids"] = []
    with pytest.raises(ValueError, match="missing_source_ids"):
        external_audit._parse_external_result(payload, valid_source_ids={"S1"})

    payload["findings"] = []
    with pytest.raises(ValueError, match="unsupported_assessment_delta"):
        external_audit._parse_external_result(payload, valid_source_ids={"S1"})


def test_overlong_citation_url_is_rejected_with_a_diagnostic() -> None:
    warnings: list[str] = []

    sources = external_audit._sanitize_sources(
        [{"url": "https://example.test/" + "x" * 2_100}], warnings
    )

    assert sources == []
    assert warnings == ["citation_url_rejected"]


class _QueuedClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete_json(self, prompt: str, **kwargs: object) -> dict[str, object]:
        self.calls.append({"prompt": prompt, **kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _completed_source(path: Path, *, paper: object = "reme.pdf") -> bytes:
    raw = json.dumps({
        "status": "completed",
        "judgment_status": "decided",
        "paper": paper,
        "trace": {
            "source_document": "fallback.pdf",
            "final_judgment": {
                "assessment": "Paper-only assessment.",
                "summary": "A completed judgment.",
            },
        },
    }).encode()
    path.write_bytes(raw)
    return raw


def _questions(count: int = 2) -> list[dict[str, str]]:
    return [
        {
            "claim": f"Claim {index}",
            "decision_impact": "Could change the experimental interpretation.",
            "query": f"ReMe official evidence query {index}",
        }
        for index in range(count)
    ]


def _auditor_result(*, source_ids: list[str] | None = None) -> dict[str, object]:
    ids = ["S1"] if source_ids is None else source_ids
    return {
        "assessment_delta": "narrowed" if ids else "unchanged",
        "revised_assessment": "External evidence qualifies the paper-only assessment.",
        "findings": [{
            "claim": "The implementation detail matters.",
            "verdict": "qualified" if ids else "unresolved",
            "analysis": "The source narrows the interpretation." if ids else "No source resolved it.",
            "source_ids": ids,
        }],
        "unresolved_questions": [],
    }


def _cited_response(index: int, *, content: str | None = None) -> dict[str, object]:
    url = f"https://example.test/source-{index}"
    return {
        "model": "x-ai/grok-4.5",
        "choices": [{
            "finish_reason": "stop",
            "message": {
                "content": f"Retrieved evidence for query {index}.",
                "annotations": [{
                    "type": "url_citation",
                    "url_citation": {
                        "url": url,
                        "title": f"Source {index}",
                        "content": content or f"Evidence {index}.",
                    },
                }],
            },
        }],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 13,
            "total_tokens": 24,
            "server_tool_use": {"web_search_requests": 1},
        },
        "openrouter_metadata": {
            "pipeline": [{
                "type": "server_tools",
                "name": "server-tools",
                "data": {"mode": "native", "tools": ["openrouter:web_search"]},
            }],
        },
    }


def test_formal_audit_plans_searches_and_audits_without_mutating_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "adaptive.json"
    source_bytes = _completed_source(source)
    output = tmp_path / "external"
    questions = _questions()
    client = _QueuedClient([{"questions": questions}, _auditor_result()])
    monkeypatch.setattr(external_audit, "openrouter_api_key", lambda: "router-secret")
    requests: list[dict[str, object]] = []

    def transport(native_request, *, timeout):
        payload = json.loads(native_request.data)
        requests.append(payload)
        query = payload["messages"][1]["content"]
        index = next(i for i, question in enumerate(questions) if question["query"] == query)
        return _FakeResponse(
            _cited_response(index, content="x" * 5_000 if index == 0 else None)
        )

    assert external_audit.run_external_audit(
        source, output, client=client, transport=transport
    ) == 0

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["backend"] == "grok-terra"
    assert manifest["questions_source"] == "planner"
    assert manifest["retrieval_request_mode"] == "one-query-per-request"
    assert manifest["retrieval_parallelism"] == 3
    assert set(manifest["prompt_versions"]) == {"planner", "grok_retrieval", "auditor"}
    assert result["status"] == "completed"
    assert result["verification_status"] == "verified"
    assert result["selected_questions"] == questions
    assert [call["role"] for call in result["model_calls"]] == [
        "planner", "grok-retrieval", "grok-retrieval", "auditor"
    ]
    assert [source["source_id"] for source in result["source_registry"]] == ["S1", "S2"]
    assert [answer["source_ids"] for answer in result["retrieval_answers"]] == [["S1"], ["S2"]]
    assert result["external_result"]["assessment_delta"] == "narrowed"
    assert "citation_field_truncated" in result["diagnostic_warnings"]
    assert len(result["source_registry"][0]["content"]) == 4_000
    assert source.read_bytes() == source_bytes
    assert result["source_result_sha256"] == hashlib.sha256(source_bytes).hexdigest()

    assert len(requests) == 2
    for request_payload in requests:
        assert request_payload["model"] == "x-ai/grok-4.5"
        assert request_payload["tool_choice"] == "required"
        assert request_payload["max_tool_calls"] == 1
        assert request_payload["tools"][0]["type"] == "openrouter:web_search"
        assert "response_format" not in request_payload
        assert "final_judgment" not in request_payload["messages"][1]["content"]

    planner_payload = json.loads(client.calls[0]["prompt"])
    auditor_payload = json.loads(client.calls[1]["prompt"])
    assert planner_payload["final_judgment"]["assessment"] == "Paper-only assessment."
    assert auditor_payload["retrieval_answers"] == result["retrieval_answers"]


def test_queries_run_in_parallel_but_results_keep_question_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "adaptive.json"
    _completed_source(source)
    output = tmp_path / "parallel"
    questions = _questions(3)
    client = _QueuedClient([{"questions": questions}, _auditor_result()])
    monkeypatch.setattr(external_audit, "openrouter_api_key", lambda: "router-secret")
    barrier = Barrier(3)

    def transport(native_request, *, timeout):
        query = json.loads(native_request.data)["messages"][1]["content"]
        index = next(i for i, question in enumerate(questions) if question["query"] == query)
        barrier.wait(timeout=2)
        return _FakeResponse(_cited_response(index))

    assert external_audit.run_external_audit(
        source, output, client=client, transport=transport
    ) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert [item["question_index"] for item in result["searches_performed"]] == [0, 1, 2]
    assert [item["answer"] for item in result["retrieval_answers"]] == [
        "Retrieved evidence for query 0.",
        "Retrieved evidence for query 1.",
        "Retrieved evidence for query 2.",
    ]


def test_missing_citation_is_diagnostic_and_does_not_discard_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "adaptive.json"
    _completed_source(source)
    output = tmp_path / "partial"
    questions = _questions(1)
    client = _QueuedClient([{"questions": questions}, _auditor_result(source_ids=[])])
    monkeypatch.setattr(external_audit, "openrouter_api_key", lambda: "router-secret")

    def transport(_request, *, timeout):
        response = _cited_response(0)
        response["choices"][0]["message"]["annotations"] = []
        return _FakeResponse(response)

    assert external_audit.run_external_audit(
        source, output, client=client, transport=transport
    ) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["verification_status"] == "unverified"
    assert "query_returned_no_sources" in result["diagnostic_warnings"]
    assert result["external_result"]["findings"][0]["verdict"] == "unresolved"


def test_query_transport_failure_is_recorded_and_secret_is_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "adaptive.json"
    _completed_source(source)
    output = tmp_path / "transport-failure"
    questions = _questions(1)
    client = _QueuedClient([{"questions": questions}, _auditor_result(source_ids=[])])
    monkeypatch.setattr(external_audit, "openrouter_api_key", lambda: "router-secret")

    def transport(_request, *, timeout):
        raise HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            503,
            "router-secret",
            {},
            None,
        )

    assert external_audit.run_external_audit(
        source, output, client=client, transport=transport
    ) == 0
    result_text = (output / "result.json").read_text(encoding="utf-8")
    result = json.loads(result_text)
    assert result["model_calls"][1]["attempt_count"] == 2
    assert result["model_calls"][1]["error"] == "OpenRouter retrieval request failed"
    assert "query_transport_failed" in result["diagnostic_warnings"]
    assert "router-secret" not in result_text


def test_invalid_auditor_citations_are_preserved_as_unverified_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "adaptive.json"
    _completed_source(source)
    output = tmp_path / "invalid-auditor"
    questions = _questions(1)
    client = _QueuedClient([
        {"questions": questions},
        _auditor_result(source_ids=["S99"]),
    ])
    monkeypatch.setattr(external_audit, "openrouter_api_key", lambda: "router-secret")

    assert external_audit.run_external_audit(
        source,
        output,
        client=client,
        transport=lambda _request, *, timeout: _FakeResponse(_cited_response(0)),
    ) == 0
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["verification_status"] == "unverified"
    assert "external_result_validation_failed" in result["diagnostic_warnings"]
    assert "external_result" not in result
    assert result["unverified_model_output"]["findings"][0]["source_ids"] == ["S99"]


def test_existing_output_and_invalid_source_fail_before_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "adaptive.json"
    _completed_source(source)
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(
        external_audit,
        "openrouter_api_key",
        lambda: (_ for _ in ()).throw(AssertionError("must not configure")),
    )

    assert external_audit.run_external_audit(source, existing) == 1
    assert external_audit.run_external_audit(tmp_path / "missing.json", tmp_path / "new") == 1


def test_planner_failure_writes_a_compact_safe_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "adaptive.json"
    _completed_source(source, paper="C:/Users/example/private/reme.pdf")
    output = tmp_path / "planner-failure"
    client = _QueuedClient([{"questions": []}])
    monkeypatch.setattr(external_audit, "openrouter_api_key", lambda: "router-secret")

    assert external_audit.run_external_audit(source, output, client=client) == 1
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    result_text = (output / "result.json").read_text(encoding="utf-8")
    result = json.loads(result_text)
    assert manifest["paper"] == "reme.pdf"
    assert manifest["status"] == "failed"
    assert result["failure_stage"] == "planning"
    assert result["selected_questions"] == []
    assert "router-secret" not in result_text
    assert "C:/Users/example/private" not in result_text


def test_serialization_redacts_only_project_paths_and_configured_keys() -> None:
    root = str(external_audit._project_root().resolve())
    payload, warnings = external_audit._sanitize_external_payload(
        {
            "project": f"{root}/private.json",
            "public": "/workspace/ReMe/config.json",
            "error": "failed with router-secret",
        },
        ("router-secret",),
    )
    assert payload["project"] == "<PROJECT_ROOT>/private.json"
    assert payload["public"] == "/workspace/ReMe/config.json"
    assert payload["error"] == "failed with [REDACTED]"
    assert warnings == ["api_key_redacted", "project_root_redacted"]
