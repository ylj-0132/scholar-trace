"""Offline output contracts; these do not measure explanation quality."""
import copy
import json

import pytest

from deep_research import paper_agent_runtime as runtime


METHOD_IDS = (
    "problem_definition", "core_contribution", "representations_and_components",
    "method_workflow", "key_details_and_assumptions", "worked_example",
)


def payload():
    return {
        "assessment": "The reported outcome is promising but narrowly evaluated.",
        "key_findings": [{"finding": "One dataset was evaluated.", "evidence": [
            {"content": "one dataset", "evidence_type": "text", "locator": "p. 4"}],
            "caveat": "Generalization is untested.", "source_finding_ids": ["F2"]}],
        "method_understanding": {"sections": [
            {"section_id": key, "explanation": "The supplied method detail.",
             "basis": "paper" if key != "worked_example" else "illustrative",
             "source_finding_ids": ["F1"], "caveat": "Illustration is not an experiment."}
            for key in METHOD_IDS], "unresolved_questions": []},
        "finding_dispositions": [
            {"finding_id": "F1", "status": "retained", "reason": "Explained in the method."},
            {"finding_id": "F2", "status": "retained", "reason": "Retained in the judgment."}],
    }


def attach(value, required=True):
    judgment = runtime._parse_single_pass_result(value, [])[2]
    return runtime._attach_finding_provenance(
        value, judgment, ("F1", "F2"), require_method_understanding=required)


def test_twelve_rubric_directions_include_reading_and_conditional_audits():
    assert runtime.DECISION_CHECKLIST == METHOD_IDS[:5] + (
        "main_evidence", "evaluation_validity", "ablation_or_counterevidence",
        "matched_resource_efficiency", "stability_and_scope",
        "auxiliary_model_reliability", "generative_transformation_fidelity")
    assert set(runtime.REFLECTION_RUBRIC_GUIDANCE) == set(runtime.DECISION_CHECKLIST)


def test_method_only_sources_are_retained_and_serializable_without_mutating_input():
    value = payload()
    original = copy.deepcopy(value)
    judgment = attach(value)
    assert judgment.method_understanding.sections[-1].basis == "illustrative"
    assert judgment.finding_dispositions[0].final_finding_indexes == ()
    assert judgment.finding_dispositions[0].method_section_ids == METHOD_IDS
    assert judgment.provenance_status == "complete"
    assert judgment.method_understanding_warnings == ()
    assert value == original
    from dataclasses import asdict
    assert json.loads(json.dumps(asdict(judgment)))["method_understanding"]["sections"][0]["source_finding_ids"] == ["F1"]


def test_missing_method_is_visible_but_does_not_destroy_existing_judgment():
    value = payload()
    del value["method_understanding"]
    judgment = attach(value)
    assert judgment.assessment == value["assessment"]
    assert judgment.method_understanding is None
    assert "missing_method_understanding" in judgment.method_understanding_warnings
    assert attach(value, required=False).method_understanding_warnings == ()


@pytest.mark.parametrize("bad", [None, [], "invented", {"sections": "bad"}])
def test_malformed_report_is_diagnostic(bad):
    value = payload()
    value["method_understanding"] = bad
    judgment = attach(value)
    assert judgment.assessment == value["assessment"]
    assert judgment.method_understanding_warnings


def test_unknown_sources_and_duplicate_sections_are_not_silently_complete():
    value = payload()
    value["method_understanding"]["sections"][0]["source_finding_ids"] = ["UNKNOWN"]
    value["method_understanding"]["sections"].append(value["method_understanding"]["sections"][1])
    judgment = attach(value)
    assert judgment.provenance_status == "incomplete"
    assert judgment.method_understanding.sections[0].source_finding_ids == ()
    assert "unknown_method_source:problem_definition:UNKNOWN" in judgment.method_understanding_warnings
    assert "duplicate_method_section:core_contribution" in judgment.method_understanding_warnings


def test_explicit_unresolved_example_needs_no_fabricated_source():
    value = payload()
    value["method_understanding"]["sections"][-1].update(
        basis="unresolved", source_finding_ids=[], explanation="The supplied evidence cannot support a worked example.")
    assert attach(value).method_understanding_warnings == ()


def test_discarded_method_source_and_false_merge_are_detected():
    value = payload()
    value["finding_dispositions"][0]["status"] = "discarded"
    assert "discarded_finding_still_linked:F1" in attach(value).provenance_warnings
    value["finding_dispositions"][0]["status"] = "merged"
    assert "unmerged_disposition:F1" in attach(value).provenance_warnings


def test_synthesis_contract_requests_explanation_without_another_reading_pass():
    prompt = json.loads(runtime._build_synthesis_prompt_from_history(
        "paper.pdf", [], "[]", {"findings": []}, paper_context_mode="master-overview-history-only"))
    sections = prompt["required_json_shape"]["method_understanding"]["sections"]
    assert tuple(item["section_id"] for item in sections) == METHOD_IDS
    assert "full_paper_pages" not in prompt and "compact_page_index" not in prompt
    assert "method_understanding" not in runtime._final_judgment_shape()


@pytest.mark.parametrize(("field", "value", "warning"), [
    ("section_id", [], "invalid_method_section"),
    ("explanation", None, "invalid_method_section:problem_definition"),
    ("basis", [], "invalid_method_section:problem_definition"),
    ("basis", "not_applicable", "inapplicable_core_method_section:problem_definition"),
    ("basis", "illustrative", "illustrative_method_section:problem_definition"),
    ("source_finding_ids", "F1", "invalid_method_sources:problem_definition"),
    ("source_finding_ids", [None], "invalid_method_sources:problem_definition"),
    ("caveat", {}, "invalid_method_section:problem_definition"),
])
def test_invalid_section_fields_preserve_other_sections(field, value, warning):
    raw = payload()
    raw["method_understanding"]["sections"][0][field] = value
    judgment = attach(raw)
    assert warning in judgment.method_understanding_warnings
    assert judgment.method_understanding.sections[-1].section_id == "worked_example"
    assert judgment.provenance_status == "incomplete"


def test_method_merge_can_account_for_multiple_sources_without_a_judgment_merge():
    raw = payload()
    raw["method_understanding"]["sections"][0]["source_finding_ids"] = ["F1", "F2"]
    raw["finding_dispositions"][0]["status"] = "merged"
    assert attach(raw).provenance_status == "complete"


def test_new_method_dimension_reaches_worker_and_rubric_union_with_bounded_context():
    from pathlib import Path
    from deep_research.paper_agent import AgentState, EvidenceTask, MasterAction, TraceStep, WorkerResult
    from deep_research.paper_reading import PaperPage

    class FakeLLM:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.calls = []

        def complete_json(self, prompt, *, system=None, image_urls=None):
            self.calls.append((json.loads(prompt), system))
            return next(self.responses)

    read = {
        "kind": "READ_PAPER", "tasks": [{"question": "What connects the two stages?", "source_scope": "paper",
            "rubric_ids": ["method_workflow"], "decision_relevance": "Explain the stage transition."}],
        "conclusion_at_risk": "The method explanation is missing its stage transition.",
        "missing_evidence": "The next stage input.",
        "expected_judgment_delta": "Explain information flow without changing the evaluation.",
    }
    master_llm = FakeLLM([read])
    master = runtime.PaperAgentMaster(llm=master_llm, paper_name="paper.pdf", page_index="[]",
                                     overview_text="Overview", overview_images=())
    action = master(AgentState(remaining_rounds=5))
    task = action.tasks[0]
    assert task.rubric_ids == ("method_workflow",)
    assert "method explanation" in " ".join(master_llm.calls[0][0]["instructions"])
    worker_llm = FakeLLM([
        {"page_ranges": [{"start": 1, "end": 1}], "rationale": "Method"},
        {"findings": [{"finding": "The stages are connected by a vector.", "evidence": [
            {"content": "Stage two consumes the vector.", "evidence_type": "text", "locator": "p. 1"}],
            "caveat": "Dimension is not specified on this page."}]},
    ])
    worker = runtime.PaperEvidenceWorker(pdf_path=Path("not-opened.pdf"),
        pages=[PaperPage(1, "Stage two consumes the vector.", 30, False, False)],
        page_index="[]", llm=worker_llm, render_pages=lambda _p, _ns: {1: "data:image/png;base64,x"})
    result = worker(task)
    assert result.error is None
    assert all(call[0]["rubric_focus"] == {
        "method_workflow": runtime.REFLECTION_RUBRIC_GUIDANCE["method_workflow"]} for call in worker_llm.calls)
    unrelated = WorkerResult(EvidenceTask("Unrelated cost", rubric_ids=("matched_resource_efficiency",)),
        finding="UNRELATED_COST", evidence="UNRELATED_QUOTE", caveat="UNRELATED_CAVEAT")
    state = AgentState(findings=(result, unrelated), steps=(TraceStep(
        1, MasterAction("READ_PAPER", (task, unrelated.task)), (result, unrelated)),))
    reflection_llm = FakeLLM([{"reflection_memo": "The link is described but its dimension remains unknown."}])
    reflector = runtime.PaperReflector(llm=reflection_llm, paper_name="paper.pdf", page_index="[]",
        overview_text="OVERVIEW_NOT_IN_SLICE", paper_context_mode="master-overview-history-only")
    report = reflector(state, trigger="master_requested", finding_ids=(), proposed_decision=MasterAction(
        "REFLECT", reflection_focus="Check stage coherence", reflection_rubric_ids=("method_workflow",)))
    assert report.context_mode == "rubric-union"
    assert report.context_finding_ids == ("r1-t1-f1",)
    sent = json.dumps(reflection_llm.calls[0][0])
    assert "Stage two consumes the vector." in sent
    assert "UNRELATED" not in sent and "OVERVIEW_NOT_IN_SLICE" not in sent
    assert "Review explanatory coherence" in sent
