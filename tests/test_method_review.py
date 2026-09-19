import json
from pathlib import Path

import pytest

from deep_research import paper_agent_runtime as runtime
from deep_research import paper_understanding as understanding
from deep_research.paper_agent import AgentState, EvidenceTask, MasterAction, TraceStep, WorkerFinding, WorkerResult, FindingEvidence


def review(**changes):
    return {"summary": "Input passes through A to B; the fallback remains unknown.",
            "source_finding_ids": ["r1-t1-f1"], "essential_finding_ids": ["r1-t1-f1"],
            "checked_aspects": ["problem", "modules", "workflow_and_branches", "details", "example", "evaluation_support"],
            "gaps": [], "stop_reason": "The method is connected and remaining limits are explicit.", **changes}


def state():
    task = EvidenceTask("Describe flow")
    result = WorkerResult(task, structured_findings=(WorkerFinding("A to B", (FindingEvidence("A to B", "text", "p. 1"),), ""),))
    return AgentState(steps=(TraceStep(1, MasterAction("READ_PAPER", (task,)), (result,)),), findings=(result,), remaining_rounds=3)


class Fake:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.prompts = []

    def complete_json(self, prompt, **kwargs):
        self.prompts.append(json.loads(prompt))
        return next(self.outputs)


def master(output):
    return runtime.PaperAgentMaster(llm=Fake([output]), paper_name="p.pdf", page_index="[]",
        overview_text="Overview", overview_images=(), require_method_review=True)


def decision(**fields):
    return {"kind": "DECIDE", "assessment": "Bounded", "stop_reason_code": "paper_saturated", **fields}


def test_supported_stop_requires_review_but_allows_explicitly_bounded_gaps():
    with pytest.raises(ValueError, match="method_review"):
        master(decision())(state())
    action = master(decision(method_review=review(gaps=[{
        "question": "Exact fallback policy?", "disposition": "bounded", "reason": "Targeted appendix reading did not establish it."}])))(state())
    assert action.method_review.summary.startswith("Input")


@pytest.mark.parametrize("change", [
    {"checked_aspects": ["problem"]}, {"stop_reason": ""},
    {"source_finding_ids": ["r2-t1-f1"]},
    {"gaps": [{"question": "Missing branch", "disposition": "paper_check", "reason": "Appendix is unread"}]},
])
def test_stop_cannot_hide_unchecked_aspects_future_sources_or_actionable_gaps(change):
    with pytest.raises(ValueError, match="method_review"):
        master(decision(method_review=review(**change)))(state())


def test_current_method_review_reaches_master_and_synthesis_history():
    parsed = runtime.parse_master_action(decision(method_review=review()))
    previous = state()
    current = AgentState(steps=previous.steps + (TraceStep(2, parsed),), findings=previous.findings)
    for build in (runtime._state_payload, runtime._master_state_payload):
        assert build(current)["method_review"]["essential_finding_ids"] == ["r1-t1-f1"]


def test_independent_read_omits_prior_context_and_decision_from_both_requests():
    from deep_research.paper_agent import ResearchContext
    from deep_research.paper_reading import PaperPage
    llm = Fake([{"page_ranges": [{"start": 1, "end": 1}], "rationale": "Figure"},
        {"findings": [{"finding": "F1 near 55.6", "evidence": [{"content": "Point near 55.6", "evidence_type": "figure", "locator": "p. 1"}], "caveat": "Approximate"}]}])
    worker = runtime.PaperEvidenceWorker(pdf_path=Path("unused.pdf"), pages=[PaperPage(1, "Source", 6, False, False)],
        page_index="[]", llm=llm, render_pages=lambda *_: {1: "data:image/png;base64,x"})
    worker.set_research_context((ResearchContext("old", "q", "OLD_VALUE_57.8", "old", "", "figure", "p1", ""),), ())
    result = worker(EvidenceTask("Read both axes and the point at k=30", decision_relevance="OLD_JUDGMENT", independent_read=True))
    assert result.error is None
    for prompt in llm.prompts:
        assert "research_context" not in prompt and "decision_context" not in prompt
        assert "OLD_VALUE" not in json.dumps(prompt) and "OLD_JUDGMENT" not in json.dumps(prompt)


def test_independent_task_cannot_select_prior_findings():
    with pytest.raises(ValueError, match="independent_read"):
        runtime.parse_master_action({"kind": "READ_PAPER", "tasks": [{"question": "Read figure", "source_scope": "paper",
            "independent_read": True, "related_finding_ids": ["r1-t1-f1"]}],
            "conclusion_at_risk": "Readout", "missing_evidence": "Raw values", "expected_judgment_delta": "Verify"})


def test_essential_method_sources_omitted_from_explanation_are_flagged():
    from deep_research.paper_agent import FinalJudgment
    parsed = understanding.parse_method_review(review())
    result = understanding.check_method_retention(FinalJudgment("Judgment", ()), parsed)
    assert "essential_method_finding_unrepresented:r1-t1-f1" in result.method_understanding_warnings


def test_retention_links_do_not_claim_semantic_verification():
    from deep_research.paper_agent import FinalJudgment
    method = understanding.MethodUnderstanding((understanding.MethodSection(
        "method_workflow", "An incomplete paraphrase", "paper", ("r1-t1-f1",)),))
    judgment = FinalJudgment("Judgment", (), method_understanding=method)
    assert understanding.check_method_retention(judgment, understanding.parse_method_review(review())) is judgment


def test_independent_context_selection_is_empty_even_for_direct_task_construction():
    from deep_research.paper_agent import _selected_research_context
    assert _selected_research_context(state(), EvidenceTask(
        "Read figure", independent_read=True, related_finding_ids=("r1-t1-f1",))) == ((), ())
    context, diagnostics = _selected_research_context(state(), EvidenceTask(
        "Compare", related_finding_ids=("r1-t1-f1",)))
    assert len(context) == 1 and not diagnostics


def test_invalid_stop_uses_existing_error_path_without_extra_calls():
    from deep_research.paper_agent import run_paper_agent
    model = master(decision())
    trace = run_paper_agent(master=model, worker=lambda _: pytest.fail("must not read"), max_rounds=5)
    assert trace.outcome == "NEEDS_HUMAN"
    assert "method_review" in trace.stop_reason
    assert len(model.llm.prompts) == 1


def test_latest_review_replaces_current_model_without_losing_trace():
    old = runtime.parse_master_action(decision(method_review=review()))
    new = runtime.parse_master_action(decision(method_review=review(summary="Corrected flow", gaps=[{
        "question": "Deployment behavior?", "disposition": "external", "reason": "Requires running code"}])))
    current = AgentState(steps=(TraceStep(2, old), TraceStep(3, new)))
    assert runtime._master_state_payload(current)["method_review"]["summary"] == "Corrected flow"
    assert current.steps[0].action.method_review.summary.startswith("Input")


def test_nonstopping_review_may_keep_actionable_gaps_but_never_future_sources():
    parsed = understanding.parse_method_review(review(checked_aspects=["problem"], gaps=[{
        "question": "Branch?", "disposition": "paper_check", "reason": "Read appendix"}]))
    understanding.validate_method_review(parsed, ["r1-t1-f1"])
    with pytest.raises(ValueError, match="future source"):
        understanding.validate_method_review(parsed, [])
