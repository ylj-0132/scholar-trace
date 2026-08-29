from __future__ import annotations

import json
from pathlib import Path

import deep_research.paper_agent_runtime as runtime

from deep_research.paper_agent import (
    AgentState,
    EvidenceTask,
    FindingEvidence,
    MasterAction,
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


def test_pre_decide_reflector_prompt_includes_only_the_proposed_decision_fields() -> None:
    llm = _FakeLLM()
    reflector = PaperReflector(
        llm=llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview text.",
    )
    decision = MasterAction(
        "DECIDE",
        assessment="A proposed conclusion.",
        rationale="Evidence is sufficient.",
        unresolved_questions=("Open boundary.",),
        checklist_coverage={"core_contribution": "covered"},
        pre_decide_reflection_focus="Whether a coupled setting changes the conclusion.",
    )

    reflector(
        AgentState(),
        trigger="pre_decide",
        finding_ids=("r1-t1-f1",),
        proposed_decision=decision,
    )

    prompt = json.loads(llm.calls[0]["prompt"])
    assert prompt["proposed_decision"] == {
        "assessment": "A proposed conclusion.",
        "rationale": "Evidence is sufficient.",
        "unresolved_questions": ["Open boundary."],
        "checklist_coverage": {"core_contribution": "covered"},
        "pre_decide_reflection_focus": "Whether a coupled setting changes the conclusion.",
    }
    instructions = " ".join(prompt["instructions"])
    assert "Analyze only proposed_decision.pre_decide_reflection_focus" in instructions
    assert "Use only accumulated findings relevant to that named conflict" in instructions
    assert "Think across the accumulated findings" not in instructions
    assert "second global omission" in instructions
    assert "proposed decision stands" in instructions
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
        pre_decide_reflection_focus="One bounded conflict.",
    )

    reflector(state, trigger="post_method_model", finding_ids=("r1-t1-f1",))
    reflector(state, trigger="pre_decide", finding_ids=("r1-t1-f1",), proposed_decision=decision)

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
