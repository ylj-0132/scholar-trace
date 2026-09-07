from __future__ import annotations

import copy
import json

import pytest

import deep_research.paper_agent_runtime as runtime
from deep_research.paper_agent import AgentTrace


def _payload() -> dict:
    return {
        "assessment": "The comparison does not isolate the mechanism.",
        "key_findings": [{
            "finding": "The proposer models differ.",
            "evidence": [{"content": "Model A versus Model B", "evidence_type": "text", "locator": "p. 2"}],
            "caveat": "The effect of the difference is not measured.",
            "source_finding_ids": ["F1", "F2"],
        }],
        "unresolved_questions": ["Is the resource budget matched?"],
        "finding_dispositions": [
            {"finding_id": "F1", "status": "merged", "reason": "Compared with F2."},
            {"finding_id": "F2", "status": "merged", "reason": "Compared with F1."},
            {"finding_id": "F3", "status": "unresolved", "reason": "No matched budget reported."},
        ],
    }


def _attach(payload: dict):
    judgment = runtime._parse_single_pass_result(payload, [])[2]
    return runtime._attach_finding_provenance(payload, judgment, ("F1", "F2", "F3"))


def test_provenance_links_merged_findings_and_preserves_unresolved_disposition() -> None:
    payload = _payload()
    original = copy.deepcopy(payload)
    judgment = _attach(payload)
    assert judgment.key_findings[0].source_finding_ids == ("F1", "F2")
    assert judgment.finding_dispositions[0].final_finding_indexes == (1,)
    assert judgment.finding_dispositions[2].final_finding_indexes == ()
    assert judgment.provenance_status == "complete"
    assert judgment.provenance_warnings == ()
    assert payload == original
    trace = AgentTrace((), "DECIDE", judgment.assessment, "done", final_judgment=judgment)
    assert json.loads(trace.to_json())["final_judgment"]["key_findings"][0]["source_finding_ids"] == ["F1", "F2"]


@pytest.mark.parametrize(("field", "value", "warning"), [
    ("finding_dispositions", [], "unaccounted_finding:F1"),
    ("finding_dispositions", None, "invalid_finding_dispositions"),
    ("finding_dispositions", [{"finding_id": "F1", "status": "discarded", "reason": ""}], "invalid_disposition:F1"),
    ("finding_dispositions", [{"finding_id": "F99", "status": "retained", "reason": "Unsupported ID"}], "unknown_disposition_finding:F99"),
])
def test_provenance_errors_are_diagnostic_not_loss_of_judgment(field, value, warning) -> None:
    payload = _payload()
    payload[field] = value
    judgment = _attach(payload)
    assert judgment.assessment == payload["assessment"]
    assert judgment.provenance_status == "incomplete"
    assert warning in judgment.provenance_warnings


def test_provenance_detects_unknown_links_duplicate_and_unlinked_dispositions() -> None:
    payload = _payload()
    payload["key_findings"][0]["source_finding_ids"] = ["F99"]
    payload["finding_dispositions"].append(payload["finding_dispositions"][0])
    judgment = _attach(payload)
    assert judgment.key_findings[0].source_finding_ids == ()
    assert "unknown_source_finding:1:F99" in judgment.provenance_warnings
    assert "unlinked_final_finding:1" in judgment.provenance_warnings
    assert "unlinked_disposition:F1" in judgment.provenance_warnings
    assert "duplicate_disposition:F1" in judgment.provenance_warnings


@pytest.mark.parametrize("status", ["retained", "corrected", "discarded"])
def test_disposition_statuses_require_consistent_links(status: str) -> None:
    payload = _payload()
    payload["finding_dispositions"][2] = {"finding_id": "F3", "status": status, "reason": "Explicit explanation."}
    if status != "discarded":
        payload["key_findings"][0]["source_finding_ids"].append("F3")
    assert _attach(payload).provenance_status == "complete"


def test_history_synthesis_requests_links_and_dispositions_but_single_pass_does_not() -> None:
    prompt = json.loads(runtime._build_synthesis_prompt_from_history(
        "paper.pdf", [], "[]", {"findings": [{"finding_id": "F1"}]},
        paper_context_mode="master-overview-history-only",
    ))
    assert "source_finding_ids" in prompt["required_json_shape"]["key_findings"][0]
    assert "finding_dispositions" in prompt["required_json_shape"]
    assert "finding_dispositions" not in runtime._final_judgment_shape()
