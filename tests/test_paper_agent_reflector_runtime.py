from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import deep_research.paper_agent_runtime as runtime

from deep_research.paper_agent import (
    AgentState,
    EvidenceTask,
    FindingEvidence,
    MasterAction,
    ReflectionReport,
    TraceStep,
    WorkerFinding,
    WorkerResult,
)
from deep_research.paper_agent_runtime import ModelCallRecorder, PaperReflector
from deep_research.paper_reading import PaperPage


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def complete_json(self, prompt: str, *, system=None, image_urls=None):
        self.calls.append({"prompt": prompt, "system": system, "image_urls": image_urls})
        return {
            "reflection_memo": (
                "The current evidence leaves open whether a coupled configuration, "
                "rather than the claimed mechanism, explains both results. The Master "
                "should ask whether the reported comparison isolates that setting."
            )
        }


def _focused_state() -> AgentState:
    results = tuple(
        WorkerResult(
            task=EvidenceTask(label),
            structured_findings=(WorkerFinding(
                label, (FindingEvidence(f"QUOTE_{label}", "text", f"p. {index}"),),
                f"CAVEAT_{label}",
            ),),
        )
        for index, label in enumerate(("SUPPORT", "COUNTER", "UNRELATED"), start=1)
    )
    return AgentState(
        findings=results,
        steps=(TraceStep(1, MasterAction("READ_PAPER", tuple(r.task for r in results)), results),),
        provisional_assessment="UNRELATED_ASSESSMENT",
        unresolved_questions=("UNRELATED_GAP",),
        reflection_reports=(ReflectionReport("post_method_model", (), "UNRELATED_OLD_MEMO"),),
    )


def _focused_decision() -> MasterAction:
    return runtime.parse_master_action({
        "kind": "REFLECT", "assessment": "GLOBAL_ASSESSMENT_NOT_FOR_FOCUSED_REVIEW",
        "stop_reason_code": "evidence_sufficient",
        "checklist_coverage": {"matched_resource_efficiency": "covered"},
        "reflection_focus": "Mechanism benefit or resource difference?",
        "reflection_rubric_ids": ["ablation_or_counterevidence", "matched_resource_efficiency"],
        "reflection_finding_ids": ["r1-t1-f1", "r1-t2-f1"],
    })


def _rubric_routed_state() -> AgentState:
    state = _focused_state()
    results = tuple(
        replace(result, task=replace(result.task, rubric_ids=(key,)))
        for result, key in zip(state.findings, (
            "ablation_or_counterevidence", "ablation_or_counterevidence", "matched_resource_efficiency",
        ))
    )
    return replace(state, findings=results, steps=(TraceStep(
        1, MasterAction("READ_PAPER", tuple(r.task for r in results)), results,
    ),))


@pytest.mark.parametrize("anchors", [(), ("r1-t1-f1",)])
def test_default_rubric_selection_includes_other_workers_without_master_picking_counterevidence(anchors) -> None:
    llm = _FakeLLM()
    reflector = PaperReflector(llm=llm, paper_name="paper", page_index="[]", overview_text="OVERVIEW")
    decision = replace(_focused_decision(),
        reflection_rubric_ids=("ablation_or_counterevidence",),
        reflection_finding_ids=anchors)
    report = reflector(_rubric_routed_state(), trigger="master_requested", finding_ids=("r1-t1-f1",), proposed_decision=decision)
    prompt = json.loads(llm.calls[0]["prompt"])
    assert report.context_mode == "rubric-union"
    assert report.context_finding_ids == ("r1-t1-f1", "r1-t2-f1")
    assert prompt["findings_to_reflect"] == list(report.context_finding_ids)
    assert "QUOTE_COUNTER" in json.dumps(prompt) and "CAVEAT_COUNTER" in json.dumps(prompt)
    assert "UNRELATED" not in json.dumps(prompt)
    assert prompt["context_selection"]["strategy"] == "rubric-union-plus-anchors"
    assert prompt["context_selection"]["anchor_finding_ids"] == list(anchors)


def test_rubric_selection_preserves_explicit_cross_dimension_anchor() -> None:
    llm = _FakeLLM()
    reflector = PaperReflector(llm=llm, paper_name="paper", page_index="[]", overview_text="OVERVIEW",
        reflection_context_mode="rubric-union")
    decision = replace(_focused_decision(), reflection_rubric_ids=("ablation_or_counterevidence",),
        reflection_finding_ids=("r1-t3-f1",))
    report = reflector(_rubric_routed_state(), trigger="master_requested", finding_ids=(), proposed_decision=decision)
    assert report.context_finding_ids == ("r1-t1-f1", "r1-t2-f1", "r1-t3-f1")


def test_rubric_with_no_matching_evidence_records_fallback() -> None:
    llm = _FakeLLM()
    reflector = PaperReflector(llm=llm, paper_name="paper", page_index="[]", overview_text="OVERVIEW",
        reflection_context_mode="rubric-union")
    decision = replace(_focused_decision(), reflection_rubric_ids=("claim_boundary",),
        reflection_finding_ids=())
    report = reflector(_rubric_routed_state(), trigger="master_requested", finding_ids=(), proposed_decision=decision)
    assert report.context_mode == "full-history"
    assert report.context_diagnostics == ("rubric_union_context_fallback:no_matching_findings",)


def test_rubric_assembly_does_not_silently_cap_matching_findings_at_six() -> None:
    state = _rubric_routed_state()
    result = replace(state.findings[0], structured_findings=state.findings[0].structured_findings * 7)
    state = replace(state, findings=(result,), steps=(TraceStep(1, MasterAction("READ_PAPER", (result.task,)), (result,)),))
    llm = _FakeLLM()
    reflector = PaperReflector(llm=llm, paper_name="paper", page_index="[]", overview_text="OVERVIEW",
        reflection_context_mode="rubric-union")
    decision = replace(_focused_decision(), reflection_rubric_ids=("ablation_or_counterevidence",),
        reflection_finding_ids=())
    report = reflector(state, trigger="master_requested", finding_ids=(), proposed_decision=decision)
    assert report.context_mode == "rubric-union"
    assert report.context_finding_ids == tuple(f"r1-t1-f{i}" for i in range(1, 8))


def test_focused_second_reflection_isolates_actual_payload_and_records_selection() -> None:
    llm = _FakeLLM()
    reflector = PaperReflector(
        llm=llm, paper_name="paper.pdf", page_index="[]", overview_text="OVERVIEW",
        reflection_context_mode="rubric-union",
    )
    report = reflector(_focused_state(), trigger="master_requested", finding_ids=("r1-t2-f1",), proposed_decision=_focused_decision())
    prompt = json.loads(llm.calls[0]["prompt"])
    serialized = json.dumps(prompt)
    assert "QUOTE_SUPPORT" in serialized and "QUOTE_COUNTER" in serialized
    assert "CAVEAT_COUNTER" in serialized
    assert "UNRELATED" not in serialized and "GLOBAL_ASSESSMENT" not in serialized
    assert "overview_pages" not in prompt and "compact_page_index" not in prompt
    assert prompt["findings_to_reflect"] == ["r1-t1-f1", "r1-t2-f1"]
    assert set(prompt["state"]["checklist_coverage"]) == {"ablation_or_counterevidence", "matched_resource_efficiency"}
    assert prompt["state"]["checklist_coverage"]["matched_resource_efficiency"] == "covered"
    assert len(prompt["mechanism_audit_principles"]) == 2
    assert report.context_mode == "rubric-union"
    assert report.context_finding_ids == ("r1-t1-f1", "r1-t2-f1")
    assert report.reflected_finding_ids == ("r1-t2-f1",)
    assert report.context_diagnostics == ()
    assert len(llm.calls) == 1


def test_focused_mode_does_not_narrow_first_reflection() -> None:
    llm = _FakeLLM()
    reflector = PaperReflector(
        llm=llm, paper_name="paper.pdf", page_index="[]", overview_text="Overview",
        paper_context_mode="master-overview-history-only", reflection_context_mode="rubric-union",
    )
    report = reflector(_focused_state(), trigger="post_method_model", finding_ids=("r1-t1-f1",))
    prompt = json.loads(llm.calls[0]["prompt"])
    assert "QUOTE_UNRELATED" in json.dumps(prompt)
    assert prompt["mechanism_audit_principles"] == list(runtime.MECHANISM_AUDIT_PRINCIPLES)
    assert report.context_mode == "full-history"


def test_missing_focused_selection_uses_explicit_recorded_fallback() -> None:
    llm = _FakeLLM()
    reflector = PaperReflector(
        llm=llm, paper_name="paper.pdf", page_index="[]", overview_text="Overview",
        reflection_context_mode="rubric-union",
    )
    report = reflector(_focused_state(), trigger="master_requested", finding_ids=("r1-t1-f1",),
        proposed_decision=MasterAction("DECIDE", reflection_focus="Named conflict"))
    assert report.error is None
    assert report.context_mode == "full-history"
    assert "rubric_union_context_fallback:missing_selection" in report.context_diagnostics


def test_unknown_selected_finding_records_fallback_in_trace_and_master_state() -> None:
    llm = _FakeLLM()
    reflector = PaperReflector(
        llm=llm, paper_name="paper.pdf", page_index="[]", overview_text="Overview",
        reflection_context_mode="rubric-union",
    )
    state = _focused_state()
    decision = replace(_focused_decision(), reflection_finding_ids=("r99-t1-f1",))
    report = reflector(state, trigger="master_requested", finding_ids=("r1-t1-f1",), proposed_decision=decision)
    assert report.context_mode == "full-history"
    assert report.context_diagnostics == ("rubric_union_context_fallback:invalid_selection",)
    assert "r99-t1-f1" not in report.context_finding_ids
    updated = replace(state, reflection_reports=(report,))
    assert runtime._master_state_payload(updated)["latest_reflection_report"]["context_diagnostics"] == list(report.context_diagnostics)
    assert runtime._state_payload(updated)["reflection_reports"][0]["context_mode"] == "full-history"


@pytest.mark.parametrize("field,value", [
    ("reflection_rubric_ids", ["unknown"]),
    ("reflection_rubric_ids", ["core_contribution"] * 2),
    ("reflection_finding_ids", "F1"),
    ("reflection_finding_ids", [""]),
])
def test_master_rejects_malformed_reflection_selection(field, value) -> None:
    with pytest.raises(ValueError, match=field):
        runtime.parse_master_action({"kind": "NEEDS_HUMAN", field: value})


def test_reflector_uses_one_no_image_call_and_keeps_report_separate_from_evidence() -> None:
    task = EvidenceTask("Inspect the first result")
    result = WorkerResult(
        task=task,
        structured_findings=(WorkerFinding(
            "A local finding.",
            (FindingEvidence("quoted text", "text", "p. 1"),),
            "A caveat.",
        ),),
    )
    state = AgentState(steps=(TraceStep(1, MasterAction("READ_PAPER", (task,)), (result,)),))
    llm = _FakeLLM()
    recorder = ModelCallRecorder()
    reflector = PaperReflector(
        llm=llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview text.",
        recorder=recorder,
    )

    report = reflector(
        state,
        trigger="post_method_model",
        finding_ids=("r1-t1-f1",),
    )

    prompt = json.loads(llm.calls[0]["prompt"])
    assert llm.calls[0]["image_urls"] is None
    assert prompt["findings_to_reflect"] == ["r1-t1-f1"]
    instructions = " ".join(prompt["instructions"])
    assert "induced optimization target" in instructions
    assert "cheapest winning strategy" in instructions
    assert "actually observes" in instructions
    assert "single most consequential issue" in instructions
    assert "declared search space" in instructions
    assert "observed accepted artifacts" in instructions
    assert "independently credited mechanisms" in instructions
    assert "Do not summarize Worker findings" in instructions
    assert "coherent prose memo" in instructions
    assert "bullet" in instructions
    assert prompt["required_json_shape"] == {"reflection_memo": "coherent analysis for the Master"}
    assert prompt["state"]["reflection_reports"] == []
    assert "coupled configuration" in report.reflection_memo
    assert recorder.records[0].role == "reflection"


def test_full_history_reflector_prompt_includes_the_requested_conclusion() -> None:
    llm = _FakeLLM()
    reflector = PaperReflector(
        llm=llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview text.",
        reflection_context_mode="full-history",
    )
    decision = MasterAction(
        "DECIDE",
        assessment="A proposed conclusion.",
        rationale="Evidence is sufficient.",
        unresolved_questions=("Open boundary.",),
        checklist_coverage={"core_contribution": "covered"},
        reflection_focus="Whether a coupled setting changes the conclusion.",
    )

    reflector(
        AgentState(),
        trigger="master_requested",
        finding_ids=("r1-t1-f1",),
        proposed_decision=decision,
    )

    prompt = json.loads(llm.calls[0]["prompt"])
    assert prompt["proposed_decision"] == {
        "assessment": "A proposed conclusion.",
        "rationale": "Evidence is sufficient.",
        "unresolved_questions": ["Open boundary."],
        "checklist_coverage": {"core_contribution": "covered"},
        "reflection_focus": "Whether a coupled setting changes the conclusion.",
    }
    instructions = " ".join(prompt["instructions"])
    assert "Analyze only proposed_decision.reflection_focus" in instructions
    assert "Use only accumulated findings relevant to that named audit question" in instructions
    assert "does not assert that a contradiction exists" in instructions
    assert "Evidence availability is not evidence of prior verification" in instructions
    assert "Think across the accumulated findings" not in instructions
    assert "second global omission" in instructions
    assert "proposed decision stands" in instructions
    assert "no material unchecked relationship remains within the focus" in instructions
    assert "Insufficient evidence to support a correction does not itself validate the conclusion" in instructions
    assert "do not recommend more reading" in instructions
    assert "conclusion at risk" in instructions
    assert "expected judgment delta" in instructions


def test_context_ownership_reflection_keeps_full_state_without_overview_or_index() -> None:
    task = EvidenceTask("Inspect the method")
    result = WorkerResult(
        task=task,
        structured_findings=(WorkerFinding(
            "FULL_FINDING_MARKER",
            (FindingEvidence("FULL_EVIDENCE_ITEM_MARKER", "text", "p. 2"),),
            "FULL_CAVEAT_MARKER",
        ),),
    )
    state = AgentState(steps=(TraceStep(1, MasterAction("READ_PAPER", (task,)), (result,)),))
    llm = _FakeLLM()
    reflector = PaperReflector(
        llm=llm,
        paper_name="paper.pdf",
        page_index='[{"page_number": 2}]',
        overview_text="OVERVIEW_NOT_SENT",
        paper_context_mode="master-main-text-history-only",
    )
    decision = MasterAction(
        "DECIDE",
        assessment="Proposed assessment.",
        reflection_focus="One bounded conflict.",
    )

    reflector(state, trigger="post_method_model", finding_ids=("r1-t1-f1",))
    reflector(state, trigger="master_requested", finding_ids=("r1-t1-f1",), proposed_decision=decision)

    for call in llm.calls:
        prompt = json.loads(call["prompt"])
        assert "FULL_FINDING_MARKER" in json.dumps(prompt["state"])
        assert "FULL_CAVEAT_MARKER" in json.dumps(prompt["state"])
        assert "FULL_EVIDENCE_ITEM_MARKER" in json.dumps(prompt["state"])
        assert "overview_pages" not in prompt
        assert "compact_page_index" not in prompt


def test_invalid_required_reflection_fails_closed_without_a_synthesis_call(
    tmp_path: Path, monkeypatch
) -> None:
    pages = [
        PaperPage(1, "Overview evidence.", 18, False, False),
        PaperPage(2, "Selected evidence.", 18, False, False),
    ]

    class SequenceLLM:
        def __init__(self) -> None:
            self.responses = [
                    {
                        "kind": "READ_PAPER",
                        "tasks": [{"question": "Inspect evidence", "source_scope": "paper"}],
                        "conclusion_at_risk": "The finding may not support the conclusion.",
                        "missing_evidence": "The selected evidence page.",
                        "expected_judgment_delta": "Bound the conclusion.",
                    },
                {"page_ranges": [{"start": 2, "end": 2}], "rationale": "target page"},
                {"finding": "A finding.", "evidence": "Selected evidence.", "caveat": "", "evidence_type": "text", "evidence_locator": "p. 2"},
                {"reflection_memo": ""},
            ]

        def complete_json(self, prompt: str, *, system=None, image_urls=None):
            del prompt, system, image_urls
            return self.responses.pop(0)

    monkeypatch.setattr(runtime, "extract_pdf_pages", lambda _path: pages)
    monkeypatch.setattr(
        runtime,
        "render_pdf_pages",
        lambda _path, numbers: {number: "data:image/png;base64,x" for number in numbers},
    )
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF fake")

    trace = runtime.run_local_paper_agent(
        pdf_path=path,
        llm=SequenceLLM(),
        max_rounds=2,
        max_reflections=2,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.final_judgment is None
    assert [record.role for record in trace.model_calls] == [
        "master", "locator", "evidence", "reflection"
    ]
    assert trace.model_calls[-1].validation_error is not None
