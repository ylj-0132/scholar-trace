import json
from dataclasses import replace

import pytest

from deep_research import paper_agent_runtime as runtime
from deep_research.paper_agent import (
    AgentState, EvidenceTask, FindingEvidence, MasterAction, ReflectionReport,
    TraceStep, WorkerFinding, WorkerResult,
)


def state_with_findings():
    task = EvidenceTask("What is the flow?", rubric_ids=("core_contribution",))
    result = WorkerResult(task, structured_findings=(
        WorkerFinding("Stage A feeds B", (FindingEvidence("A to B", "text", "p. 2"),), "",
                      content_rubric_ids=("method_workflow",)),
        WorkerFinding("Unassigned detail", (FindingEvidence("detail", "text", "p. 2"),), ""),
    ))
    return AgentState(findings=(result,), steps=(TraceStep(1, MasterAction("READ_PAPER", (task,)), (result,)),))


def content(state):
    return runtime._state_payload(state)["rubric_content"]


def update(text, **extras):
    return {"rubric_id": "method_workflow", "kind": "understanding", "text": text,
            "source_finding_ids": ["r1-t1-f1"], "reason": "Based on the supplied transition.", **extras}


def action(*updates, links=None):
    return runtime.parse_master_action({"kind": "NEEDS_HUMAN", "rubric_updates": list(updates),
                                        "rubric_links": links or []})


def test_content_associations_do_not_inherit_task_labels_or_copy_evidence():
    state = state_with_findings()
    layer = content(state)
    assert layer["dimensions"]["method_workflow"]["finding_ids"] == ["r1-t1-f1"]
    assert layer["dimensions"]["core_contribution"]["finding_ids"] == []
    assert layer["unassigned_finding_ids"] == ["r1-t1-f2"]
    assert "A to B" not in json.dumps(layer)
    assert runtime._state_payload(state)["rubric_context_index"]["by_rubric"] == {
        "core_contribution": ["r1-t1-f1", "r1-t1-f2"]}


def test_master_supersedes_interpretation_with_reason_without_erasing_history():
    state = state_with_findings()
    state = replace(state, steps=state.steps + (TraceStep(2, action(update("Initial explanation"))),))
    state = replace(state, steps=state.steps + (TraceStep(3, action(update("Corrected explanation",
        supersedes=["r2-master-1"]))),))
    layer = content(state)
    assert [item["text"] for item in layer["entries"]] == ["Initial explanation", "Corrected explanation"]
    assert layer["dimensions"]["method_workflow"]["current_entry_ids"] == ["r3-master-1"]
    assert layer["entries"][1]["supersedes"] == ["r2-master-1"]
    assert layer["entries"][1]["reason"]
    assert runtime._master_state_payload(state)["rubric_content"] == layer


def test_association_revision_retains_original_and_unassigned_material():
    state = state_with_findings()
    state = replace(state, steps=state.steps + (TraceStep(2, action(links=[{
        "finding_id": "r1-t1-f1", "rubric_ids": [], "reason": "This classification is not appropriate."}])),))
    layer = content(state)
    assert layer["dimensions"]["method_workflow"]["finding_ids"] == []
    assert layer["unassigned_finding_ids"] == ["r1-t1-f1", "r1-t1-f2"]
    assert layer["association_history"][0]["rubric_ids"] == ["method_workflow"]
    assert layer["association_history"][-1]["rubric_ids"] == []
    assert len(runtime._state_payload(state)["findings"]) == 2


def test_future_sources_and_invalid_supersession_are_diagnostic_not_state():
    state = state_with_findings()
    # The first Master cannot cite results produced by its own batch.
    state = replace(state, steps=(replace(state.steps[0], action=action(update("Future assertion"))),
        TraceStep(2, action(update("Bad correction", supersedes=["r99-master-1"])))))
    layer = content(state)
    assert layer["entries"] == []
    assert any("unknown_finding" in warning for warning in layer["warnings"])
    assert any("invalid_supersedes" in warning for warning in layer["warnings"])
    assert len(runtime._state_payload(state)["findings"]) == 2


def test_reflection_notes_are_candidates_and_cannot_cite_outside_actual_input():
    state = state_with_findings()
    report = runtime.parse_reflection_report({"reflection_memo": "Check the transition.",
        "rubric_notes": [update("Candidate explanation"), update("Unseen source", source_finding_ids=["r1-t1-f2"])]},
        trigger="post_method_model", reflected_finding_ids=("r1-t1-f1", "r1-t1-f2"),
        context_finding_ids=("r1-t1-f1",))
    report = replace(report, available_after_round=1)
    state = replace(state, reflection_reports=(report,))
    layer = content(state)
    assert len(layer["entries"]) == 1
    assert layer["entries"][0]["role"] == "reflection"
    assert layer["dimensions"]["method_workflow"]["candidate_entry_ids"] == ["reflection-1-note-1"]
    assert layer["dimensions"]["method_workflow"]["current_entry_ids"] == []
    assert report.content_warnings


def test_bad_optional_metadata_preserves_master_action():
    parsed = runtime.parse_master_action({"kind": "NEEDS_HUMAN", "rubric_updates": "bad", "rubric_links": [None]})
    assert parsed.kind == "NEEDS_HUMAN"
    assert parsed.rubric_updates == () and parsed.rubric_links == ()
    assert parsed.content_warnings


def test_master_cannot_reference_entries_it_is_creating_in_the_same_action():
    state = state_with_findings()
    state = replace(state, steps=state.steps + (TraceStep(2, action(
        update("First"), update("Second", source_entry_ids=["r2-master-1"]))),))
    layer = content(state)
    assert [entry["text"] for entry in layer["entries"]] == ["First"]
    assert "r2-master-2:unknown_source_entry" in layer["warnings"]


def test_closing_question_preserves_history_and_explicit_resolution():
    state = state_with_findings()
    state = replace(state, steps=state.steps + (TraceStep(2, action(update("Which transition?", kind="open_question"))),))
    state = replace(state, steps=state.steps + (TraceStep(3, action(update("The source describes A to B.",
        kind="open_question", status="resolved", supersedes=["r2-master-1"]))),))
    layer = content(state)
    assert layer["dimensions"]["method_workflow"]["current_entry_ids"] == []
    assert layer["entries"][-1]["status"] == "resolved"
    assert len(layer["entries"]) == 2


def test_combined_action_cannot_adopt_its_own_pending_reflection():
    state = state_with_findings()
    report = runtime.parse_reflection_report({"reflection_memo": "Candidate", "rubric_notes": [update("Analysis")]},
        trigger="master_requested", reflected_finding_ids=("r1-t1-f1",))
    state = replace(state, reflection_reports=(replace(report, available_after_round=2),),
        steps=state.steps + (TraceStep(2, action(update("Too early", source_entry_ids=["reflection-1-note-1"]))),))
    assert "r2-master-1:unknown_source_entry" in content(state)["warnings"]
    state = replace(state, steps=state.steps + (TraceStep(3, action(update("Now available",
        source_entry_ids=["reflection-1-note-1"]))),))
    assert content(state)["dimensions"]["method_workflow"]["current_entry_ids"] == ["r3-master-1"]


def test_worker_content_metadata_is_parsed_without_losing_a_valid_finding():
    task = EvidenceTask("Flow", rubric_ids=("core_contribution",))
    raw = {"findings": [{"finding": "Flow fact", "evidence": [
        {"content": "A to B", "evidence_type": "table", "locator": "p. 1"}],
        "caveat": "", "content_rubric_ids": ["method_workflow", "key_details_and_assumptions"]}]}
    result = runtime._parse_evidence_result(payload=raw, task=task, selected_pages=(1,), location_rationale="Method", selected_text={1: "A to B"})
    assert result.error is None
    assert result.structured_findings[0].content_rubric_ids == ("method_workflow", "key_details_and_assumptions")
    raw["findings"][0]["content_rubric_ids"] = ["unknown"]
    result = runtime._parse_evidence_result(payload=raw, task=task, selected_pages=(1,), location_rationale="Method", selected_text={1: "A to B"})
    assert result.error is None and result.structured_findings[0].content_rubric_ids == ()
    assert result.structured_findings[0].content_warnings


@pytest.mark.parametrize("mode", ["full-history", "rubric-union"])
@pytest.mark.parametrize("layout", ["standard", "cache-friendly"])
def test_mocked_run_manages_content_and_writes_coherent_report_without_changing_reflection_selection(
    tmp_path, monkeypatch, mode, layout,
):
    from deep_research.paper_reading import PaperPage

    class Fake:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = []

        def complete_json(self, prompt, *, system=None, image_urls=None):
            self.calls.append((json.loads(prompt), system))
            return next(self.responses)

    pages = [PaperPage(1, "A to B", 6, False, False)]
    monkeypatch.setattr(runtime, "extract_pdf_pages", lambda _path: pages)
    monkeypatch.setattr(runtime, "render_pdf_pages", lambda _path, numbers: {n: "data:image/png;base64,x" for n in numbers})
    master = Fake([
        {"kind": "READ_PAPER", "tasks": [{"question": "Flow?", "source_scope": "paper", "rubric_ids": ["core_contribution"]}],
         "conclusion_at_risk": "Flow unclear", "missing_evidence": "Transition", "expected_judgment_delta": "Explain flow"},
        {"kind": "READ_PAPER_AND_REFLECT", "tasks": [{"question": "Rule?", "source_scope": "paper", "rubric_ids": ["method_workflow"]}],
         "conclusion_at_risk": "Rule unclear", "missing_evidence": "Rule", "expected_judgment_delta": "Clarify rule",
         "reflection_focus": "Check prior transition", "reflection_rubric_ids": ["core_contribution"],
         "independence_rationale": "New rule check is independent of prior flow analysis",
         "rubric_updates": [update("Current flow", source_entry_ids=["reflection-1-note-1"])],
         "rubric_links": [{"finding_id": "r1-t1-f1", "rubric_ids": ["key_details_and_assumptions"], "reason": "Actually an operating rule"}]},
        {"kind": "DECIDE", "assessment": "Bounded report", "stop_reason_code": "evidence_sufficient",
         "rubric_updates": [update("Revised flow", source_finding_ids=["r1-t1-f1", "r2-t1-f1"],
                                   source_entry_ids=["reflection-2-note-1"], supersedes=["r2-master-1"])]},
    ])
    locator = Fake([{"page_ranges": [{"start": 1, "end": 1}], "rationale": "Method"}] * 2)
    evidence = Fake([{"findings": [{"finding": "Flow", "evidence": [{"content": "A to B", "evidence_type": "text", "locator": "p. 1"}],
                                    "caveat": "Bounded", "content_rubric_ids": ["method_workflow"]}]}] * 2)
    reflection = Fake([
        {"reflection_memo": "Candidate 1", "rubric_notes": [update("Candidate 1")]},
        {"reflection_memo": "Candidate 2", "rubric_notes": [update("Candidate 2"), update("Future", source_finding_ids=["r2-t1-f1"])]},
    ])
    synthesis = Fake([{"assessment": "Connected evaluation", "key_findings": [{
        "finding": "Bounded outcome", "evidence": [{"content": "A to B", "evidence_type": "text", "locator": "p. 1"}],
        "caveat": "", "source_finding_ids": ["r1-t1-f1", "r2-t1-f1"]}],
        "finding_dispositions": [{"finding_id": key, "status": "merged", "reason": "Combined"} for key in ("r1-t1-f1", "r2-t1-f1")],
        "method_understanding": {"sections": [{"section_id": key, "explanation": "Partial connected explanation",
            "basis": "unresolved", "source_finding_ids": ["r1-t1-f1"], "caveat": "Missing details"}
            for key in runtime.METHOD_SECTION_GUIDANCE], "unresolved_questions": ["Details?"]}}])
    trace = runtime.run_local_paper_agent(pdf_path=tmp_path / "fake.pdf", llm=master,
        role_llms={"master": master, "locator": locator, "evidence": evidence, "reflection": reflection, "synthesis": synthesis},
        max_rounds=3, max_reflections=2, reflection_context_mode=mode, prompt_layout=layout,
        master_context_mode="incremental-with-evidence", paper_context_mode="master-overview-history-only")
    assert trace.outcome == "DECIDE"
    assert trace.reflection_reports[1].context_mode == mode
    assert trace.reflection_reports[1].context_finding_ids == ("r1-t1-f1",)
    assert trace.reflection_reports[1].content_warnings  # Same-batch future evidence rejected.
    assert [r.available_after_round for r in trace.reflection_reports] == [1, 2]
    assert all("rubric_content" not in call[0]["state"] for call in reflection.calls)
    assert "rubric_content" in master.calls[1][0]["state"]
    assert trace.rubric_content["dimensions"]["method_workflow"]["current_entry_ids"] == ["r3-master-1"]
    assert trace.rubric_content["dimensions"]["key_details_and_assumptions"]["finding_ids"] == ["r1-t1-f1"]
    assert trace.rubric_content["entries"][-1]["supersedes"] == ["r2-master-1"]
    final_input = synthesis.calls[0][0]
    assert "human_review_warnings" not in final_input["history"]["rubric_content"]
    assert final_input["history"]["rubric_content"] == {
        key: value for key, value in trace.rubric_content.items() if key != "human_review_warnings"
    }
    assert len(final_input["history"]["findings"]) == 2
    assert "full_paper_pages" not in final_input and "compact_page_index" not in final_input
    assert "not twelve rubric-by-rubric reports" in " ".join(final_input["instructions"])
    assert trace.final_judgment.assessment == "Connected evaluation"
    assert trace.final_judgment.provenance_status == "complete"
    assert json.loads(trace.to_json())["rubric_content"] == trace.rubric_content
