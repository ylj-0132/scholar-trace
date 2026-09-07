"""Offline contracts for rubric routing, not claims of model judgment quality."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_research import paper_agent_runtime as runtime
from deep_research.paper_agent import (
    AgentState, AgentTrace, EvidenceTask, FindingEvidence, MasterAction,
    TraceStep, WorkerFinding, WorkerResult, completed_finding_records,
    _selected_research_context,
)
from deep_research.paper_reading import PaperPage


class ResponseLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.prompts = []

    def complete_json(self, prompt, *, system=None, image_urls=None):
        self.prompts.append(json.loads(prompt))
        return next(self.responses)


def read_payload(**task_fields):
    return {
        "kind": "READ_PAPER",
        "tasks": [{"question": "Check memory input", "source_scope": "paper", **task_fields}],
        "conclusion_at_risk": "Injection is unspecified",
        "missing_evidence": "Appendix template",
        "expected_judgment_delta": "Qualify the missing-detail claim",
    }


def rubric_state():
    results = tuple(
        WorkerResult(
            EvidenceTask(label, rubric_ids=keys),
            structured_findings=(WorkerFinding(
                label, (FindingEvidence("QUOTE_" + label, "text", f"p. {i}"),),
                "CAVEAT_" + label,
            ),), pages_read=(i,),
        )
        for i, (label, keys) in enumerate([
            ("MISSING_INPUT", ("core_contribution",)),
            ("TEMPLATE_PRESENT", ("core_contribution", "claim_boundary")),
            ("UNRELATED_COST", ("matched_resource_efficiency",)),
            ("LEGACY_UNASSIGNED", ()),
        ], 1)
    )
    return AgentState(findings=results, steps=(TraceStep(
        1, MasterAction("READ_PAPER", tuple(r.task for r in results)), results,
    ),))


def test_task_rubric_labels_roundtrip_and_legacy_default():
    action = runtime.parse_master_action(read_payload(rubric_ids=["core_contribution", "claim_boundary"]))
    assert action.tasks[0].rubric_ids == ("core_contribution", "claim_boundary")
    assert runtime.parse_master_action(read_payload()).tasks[0].rubric_ids == ()
    trace = AgentTrace((TraceStep(1, action),), "DECIDE", "bounded", "done")
    assert json.loads(trace.to_json())["steps"][0]["action"]["tasks"][0]["rubric_ids"] == ["core_contribution", "claim_boundary"]


@pytest.mark.parametrize("value", ["core_contribution", None, [2], ["unknown"],
    ["core_contribution", "core_contribution"], [""],
    ["core_contribution", "claim_boundary", "main_evidence"]])
def test_invalid_task_rubric_labels_are_rejected(value):
    with pytest.raises(ValueError, match="rubric_ids"):
        runtime.parse_master_action(read_payload(rubric_ids=value))


def test_rubric_index_routes_ids_without_copying_evidence_or_hiding_legacy_findings():
    state = rubric_state()
    expected = {
        "by_rubric": {
            "core_contribution": ["r1-t1-f1", "r1-t2-f1"],
            "claim_boundary": ["r1-t2-f1"],
            "matched_resource_efficiency": ["r1-t3-f1"],
        },
        "unassigned_finding_ids": ["r1-t4-f1"],
    }
    for build in (runtime._state_payload, runtime._master_state_payload):
        payload = build(state)
        assert payload["rubric_context_index"] == expected
        assert len(payload["findings"]) == 4
        assert payload["findings"][1]["task_rubric_ids"] == ["core_contribution", "claim_boundary"]
        assert "QUOTE_" not in json.dumps(payload["rubric_context_index"])
    compact = json.dumps(runtime._master_state_payload(state))
    assert "QUOTE_" not in compact and "CAVEAT_TEMPLATE_PRESENT" in compact
    records = completed_finding_records(state)
    assert records[1].task_rubric_ids == ("core_contribution", "claim_boundary")
    selected, diagnostics = _selected_research_context(state, EvidenceTask(
        "Check a cross-dimension relationship", related_finding_ids=("r1-t2-f1",),
        rubric_ids=("matched_resource_efficiency",),
    ))
    assert diagnostics == () and selected[0].task_rubric_ids == records[1].task_rubric_ids
    assert runtime._research_context_payload(selected)[0]["task_rubric_ids"] == list(records[1].task_rubric_ids)


def test_actual_worker_calls_receive_only_assigned_rubric_guidance():
    llm = ResponseLLM(
        {"page_ranges": [{"start": 1, "end": 1}], "rationale": "Template"},
        {"findings": [{"finding": "Input exists", "evidence": [
            {"content": "Previous results", "evidence_type": "text", "locator": "p. 1"}], "caveat": "Selected page only"}]},
    )
    worker = runtime.PaperEvidenceWorker(
        pdf_path=Path("not-opened.pdf"), pages=[PaperPage(1, "Previous results", 16, False, False)],
        page_index="[]", llm=llm, render_pages=lambda _p, _ns: {1: "data:image/png;base64,x"},
    )
    result = worker(EvidenceTask("Where is prior state injected?", rubric_ids=("core_contribution",)))
    assert result.error is None
    for payload in llm.prompts:
        assert payload["rubric_focus"] == {"core_contribution": runtime.REFLECTION_RUBRIC_GUIDANCE["core_contribution"]}
        assert "research_context" not in payload
    assert "not evidence of absence" in " ".join(llm.prompts[1]["instructions"])


@pytest.mark.parametrize("keys", [("unknown",), ("core_contribution", "core_contribution"),
    ("core_contribution", "claim_boundary", "main_evidence"), "core_contribution"])
def test_programmatic_worker_rejects_bad_rubric_before_any_model_call(keys):
    llm = ResponseLLM()
    worker = runtime.PaperEvidenceWorker(pdf_path=Path("not-opened.pdf"), pages=[], page_index="[]", llm=llm)
    result = worker(EvidenceTask("Question", rubric_ids=keys))
    assert result.error == "invalid_task_rubric_ids"
    assert llm.prompts == []


def test_focused_reflection_indexes_only_visible_slice_and_preserves_old_mechanism_prompt():
    llm = ResponseLLM({"reflection_memo": "Compare the two reports before concluding absence."})
    reflector = runtime.PaperReflector(llm=llm, paper_name="paper", page_index="[]", overview_text="NO_FULL_TEXT",
        reflection_context_mode="rubric-union")
    decision = MasterAction("DECIDE", reflection_focus="Missing input versus appendix template",
        reflection_rubric_ids=("core_contribution",),
        reflection_finding_ids=("r1-t1-f1", "r1-t2-f1"))
    report = reflector(rubric_state(), trigger="master_requested", finding_ids=("r1-t2-f1",), proposed_decision=decision)
    prompt = llm.prompts[0]
    assert report.context_finding_ids == ("r1-t1-f1", "r1-t2-f1")
    assert prompt["state"]["rubric_context_index"]["by_rubric"] == {
        "core_contribution": ["r1-t1-f1", "r1-t2-f1"], "claim_boundary": ["r1-t2-f1"],
    }
    assert "UNRELATED_COST" not in json.dumps(prompt) and "LEGACY_UNASSIGNED" not in json.dumps(prompt)
    instructions = " ".join(prompt["instructions"])
    assert "cross-Worker" in instructions and "not evidence of absence" in instructions
    assert "Analyze only proposed_decision.reflection_focus" in instructions
    assert "expected judgment delta" in instructions


def test_master_and_synthesis_use_rubric_index_for_relationships_not_scoring():
    llm = ResponseLLM({"kind": "NEEDS_HUMAN"})
    master = runtime.PaperAgentMaster(llm=llm, paper_name="paper", page_index="[]", overview_text="OVERVIEW",
        overview_images=(), master_context_mode="incremental-no-raw-evidence")
    master(rubric_state())
    prompt = llm.prompts[0]
    assert "rubric_context_index" in prompt["state"]
    assert "rubric_ids" in prompt["output_contract"]["READ_PAPER"]["tasks"][0]
    instructions = " ".join(prompt["instructions"])
    assert "cross-Worker" in instructions and "routing hints" in instructions
    assert "one named unresolved mechanism conflict" in instructions
    assert "reading value and verification value separately" in instructions
    assert "No new findings or already-identified contradiction is required" in instructions
    assert "When skipping Reflection, explain" in instructions
    assert "optional anchors, not a whitelist" in instructions
    trace = AgentTrace(rubric_state().steps, "DECIDE", "bounded", "done")
    synthesis = json.loads(runtime._build_synthesis_prompt("paper", [], "[]", trace,
        paper_context_mode="master-overview-history-only"))
    assert synthesis["history"]["rubric_context_index"] == prompt["state"]["rubric_context_index"]
    assert "rubric_context_index" in " ".join(synthesis["instructions"])
    assert "full_paper_pages" not in synthesis
