import json
from dataclasses import replace

import pytest

from deep_research import paper_agent_runtime as runtime
from deep_research.paper_agent import AgentState, EvidenceTask, MasterAction, TraceStep


TEXT = "Histories shorter than 7 days activate day and week levels."


def worker(details=None):
    return runtime._parse_evidence_result(
        payload={"findings": [{"finding": TEXT, "evidence": [{"content": TEXT, "evidence_type": "text", "locator": "p. 13"}],
          "caveat": "Reported rule, not reproduced.", "key_details": details if details is not None else [{
              "text": TEXT, "rubric_id": "key_details_and_assumptions", "destination": "method"}]}]},
        task=EvidenceTask("Read activation rules"), selected_pages=(13,), location_rationale="Rules", selected_text={13: TEXT})


def state():
    result = worker()
    return AgentState(steps=(TraceStep(1, MasterAction("READ_PAPER", (result.task,)), (result,)),), findings=(result,))


def layer(s):
    return runtime._state_payload(s)["rubric_content"]


def update(**changes):
    return {"detail_id": "r1-t1-f1-d1", "status": "active", "text": TEXT,
            "rubric_id": "key_details_and_assumptions", "destination": "method",
            "source_finding_ids": ["r1-t1-f1"], "reason": "Preserve the exact operating condition.", **changes}


def action(*updates):
    return runtime.parse_master_action({"kind": "NEEDS_HUMAN", "detail_updates": list(updates)})


def test_worker_records_concrete_detail_with_automatic_source_and_stable_id():
    detail = layer(state())["key_details"][0]
    assert detail["id"] == "r1-t1-f1-d1"
    assert detail["text"] == TEXT
    assert detail["source_finding_ids"] == ["r1-t1-f1"]
    assert layer(state())["dimensions"]["key_details_and_assumptions"]["detail_ids"] == [detail["id"]]


def test_silent_omission_in_later_master_turn_cannot_remove_detail():
    s = state()
    later = replace(s, steps=s.steps + (TraceStep(2, action()), TraceStep(3, action())))
    assert layer(later)["key_details"] == layer(s)["key_details"]
    assert runtime._master_state_payload(later)["rubric_content"]["key_details"] == layer(s)["key_details"]


def test_withdrawal_keeps_original_and_reason_in_history():
    s = state()
    later = replace(s, steps=s.steps + (TraceStep(2, action(update(status="withdrawn", reason="Redundant with another retained condition."))),))
    c = layer(later)
    assert c["key_details"][0]["status"] == "withdrawn"
    assert len(c["detail_history"]) == 2
    assert c["detail_history"][0]["text"] == TEXT
    assert c["detail_history"][1]["reason"]


@pytest.mark.parametrize("change", [{"reason": ""}, {"source_finding_ids": ["r9-t1-f1"]}])
def test_invalid_changes_cannot_mutate_current_fact(change):
    s = state()
    later = replace(s, steps=s.steps + (TraceStep(2, action(update(**change))),))
    assert layer(later)["key_details"] == layer(s)["key_details"]
    assert layer(later)["warnings"]


def test_master_cannot_change_its_own_future_batch_detail():
    s = state()
    s = replace(s, steps=(replace(s.steps[0], action=action(update(status="withdrawn"))),))
    assert layer(s)["key_details"][0]["status"] == "active"
    assert any("unknown_detail" in w for w in layer(s)["warnings"])


def test_nonverbatim_detail_is_preserved_with_human_advisory():
    result = worker([{"text": "Invented 70-day threshold.", "rubric_id": "key_details_and_assumptions", "destination": "method"}])
    assert result.error is None and result.structured_findings[0].finding == TEXT
    assert result.structured_findings[0].key_details[0].text == "Invented 70-day threshold."
    assert result.structured_findings[0].content_warnings


def judgment(text=TEXT, destination="method"):
    from deep_research.paper_agent import FinalJudgment, FinalJudgmentFinding
    from deep_research.paper_understanding import MethodSection, MethodUnderstanding
    if destination == "method":
        return FinalJudgment("Audit", (), method_understanding=MethodUnderstanding((MethodSection(
            "key_details_and_assumptions", text, "paper", ("r1-t1-f1",)),)))
    return FinalJudgment("Audit", (FinalJudgmentFinding(text, (), "", ("r1-t1-f1",)),))


def disposition(**changes):
    return {"detail_id": "r1-t1-f1-d1", "status": "retained", "report_quote": TEXT,
            "method_section_id": "key_details_and_assumptions", "final_finding_index": None,
            "source_finding_ids": ["r1-t1-f1"], "reason": "Exact condition preserved.", **changes}


def check(j, rows, s=None):
    from deep_research.rubric_details import check_detail_retention
    s = s or state()
    return check_detail_retention(j, rows, layer(s), s)


def test_paraphrase_is_human_advisory_not_retention_failure():
    result = check(judgment("Levels depend on memory age."), [disposition(report_quote="Levels depend on memory age.")])
    assert result.detail_retention_status == "complete"
    assert not result.detail_retention_warnings
    assert result.human_review_warnings


def test_exact_fact_in_connected_method_report_passes():
    result = check(judgment("Activation rules follow. " + TEXT), [disposition()])
    assert result.detail_retention_status == "complete"
    assert not result.detail_retention_warnings


def test_evaluation_detail_does_not_have_to_appear_in_method_section():
    s = state()
    s = replace(s, steps=s.steps + (TraceStep(2, action(update(destination="evaluation", rubric_id="main_evidence"))),))
    result = check(judgment(destination="evaluation"), [disposition(method_section_id=None, final_finding_index=1)], s)
    assert result.detail_retention_status == "complete"


def test_missing_or_explicitly_omitted_detail_is_visible():
    assert check(judgment(), []).detail_retention_status == "incomplete"
    result = check(judgment(), [disposition(status="omitted", reason="Too long.")])
    assert result.detail_retention_status == "incomplete"
    assert result.detail_dispositions[0]["reason"] == "Too long."


def test_master_can_promote_preexisting_concrete_evidence_without_worker_metadata():
    result = worker([])
    s = AgentState(steps=(TraceStep(1, MasterAction("READ_PAPER"), (result,)),))
    s = replace(s, steps=s.steps + (TraceStep(2, action(update(detail_id=None))),))
    assert layer(s)["key_details"][0]["id"] == "r2-master-detail-1"
    assert layer(s)["key_details"][0]["text"] == TEXT


def test_sourced_correction_preserves_original_history():
    s = state()
    corrected = "Histories shorter than 8 days activate day and week levels."
    f = replace(s.findings[0].structured_findings[0], finding=corrected, key_details=())
    result = replace(s.findings[0], structured_findings=(f,))
    s = replace(s, steps=s.steps + (TraceStep(2, MasterAction("READ_PAPER"), (result,)),
        TraceStep(3, action(update(text=corrected, source_finding_ids=["r2-t1-f1"], reason="New passage corrects the threshold.")))))
    assert layer(s)["key_details"][0]["text"] == corrected
    assert layer(s)["detail_history"][0]["text"] == TEXT


def test_nonverbatim_correction_is_human_advisory_only():
    j = judgment("The actual threshold is 70 days.")
    result = check(j, [disposition(status="corrected", report_quote="The actual threshold is 70 days.")])
    assert result.human_review_warnings
    assert not result.detail_retention_warnings


def test_empty_legacy_detail_layer_is_not_a_completeness_claim():
    result = worker([])
    s = AgentState(steps=(TraceStep(1, MasterAction("READ_PAPER"), (result,)),))
    assert check(judgment(), [], s).detail_retention_status == "not_checked"


def test_duplicate_disposition_does_not_mask_error():
    assert check(judgment(), [disposition(), disposition()]).detail_retention_status == "incomplete"


@pytest.mark.parametrize("keep", [True, False])
def test_history_only_synthesis_receives_ledger_and_checks_actual_report(monkeypatch, tmp_path, keep):
    import json
    from deep_research.paper_agent import AgentTrace
    from deep_research.paper_reading import PaperPage
    from deep_research.paper_understanding import METHOD_SECTION_IDS

    class Fake:
        prompts = []

        def complete_json(self, prompt, **kwargs):
            self.prompts.append(json.loads(prompt))
            return {"assessment": "Bounded report", "key_findings": [{"finding": "Reported rule", "caveat": "",
                "evidence": [{"content": TEXT, "evidence_type": "text", "locator": "p. 13"}],
                "source_finding_ids": ["r1-t1-f1"]}],
                "method_understanding": {"sections": [{"section_id": section, "basis": "paper",
                    "explanation": TEXT if keep else "Levels depend on memory age.", "source_finding_ids": ["r1-t1-f1"], "caveat": ""}
                    for section in METHOD_SECTION_IDS]},
                "finding_dispositions": [{"finding_id": "r1-t1-f1", "status": "retained", "reason": "Used"}],
                "detail_dispositions": [disposition(report_quote=TEXT if keep else "Levels depend on memory age.")]}

    monkeypatch.setattr(runtime, "extract_pdf_pages", lambda _: [PaperPage(1, "Overview", 8, False, False)])
    monkeypatch.setattr(runtime, "render_pdf_pages", lambda *_: {1: "data:image/png;base64,x"})
    monkeypatch.setattr(runtime, "run_paper_agent", lambda **_: AgentTrace(state().steps, "DECIDE", "Bounded", "Done"))
    llm = Fake()
    trace = runtime.run_local_paper_agent(pdf_path=tmp_path / "paper.pdf", llm=llm, max_rounds=5,
                                         paper_context_mode="master-overview-history-only")
    assert trace.final_judgment is not None
    assert trace.final_judgment.detail_retention_status == "complete"
    assert bool(trace.final_judgment.human_review_warnings) is (not keep)
    assert len(llm.prompts) == 1
    assert llm.prompts[0]["history"]["rubric_content"]["key_details"][0]["text"] == TEXT
    assert "full_paper_pages" not in llm.prompts[0]



def test_master_paraphrase_updates_state_but_advisory_is_not_sent_to_models():
    s = state()
    text = "For histories under seven days, use the day and week layers."
    s = replace(s, steps=s.steps + (TraceStep(2, action(update(text=text))),))
    from deep_research.rubric_content import build_rubric_content
    full = build_rubric_content(s, runtime.DECISION_CHECKLIST)
    assert full["key_details"][0]["text"] == text
    assert full["human_review_warnings"]
    assert not full["warnings"]
    for make in (runtime._state_payload, runtime._master_state_payload):
        payload = make(s)["rubric_content"]
        assert payload["key_details"][0]["text"] == text
        assert "human_review_warnings" not in payload
        assert "human_review:" not in json.dumps(make(s))


def test_nonverbatim_worker_detail_remains_available_to_master_and_synthesis():
    result = worker([{"text": "Under seven days, activate the day/week levels.",
                     "rubric_id": "key_details_and_assumptions", "destination": "method"}])
    s = AgentState(steps=(TraceStep(1, MasterAction("READ_PAPER"), (result,)),))
    for make in (runtime._state_payload, runtime._master_state_payload):
        assert make(s)["rubric_content"]["key_details"][0]["text"].startswith("Under seven")
        assert not make(s)["rubric_content"]["warnings"]
