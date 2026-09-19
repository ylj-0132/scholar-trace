"""Offline payload contracts, not a claim that an LLM will follow them."""
import json
from dataclasses import replace

import pytest

from deep_research import paper_agent_runtime as runtime
from deep_research.llm import cache_friendly_prompt_parts
from deep_research.paper_agent import (
    AgentState, EvidenceTask, FindingEvidence, MasterAction, ReflectionReport,
    TraceStep, WorkerFinding, WorkerResult,
    completed_finding_records,
)


def state_with_findings():
    results = tuple(WorkerResult(EvidenceTask(f"question {i}"), pages_read=(i,),
        structured_findings=(WorkerFinding(f"finding {i}",
            (FindingEvidence(f"quote {i}", "text", f"p. {i}"),), "selected pages only"),))
        for i in (1, 2))
    steps = tuple(TraceStep(i, MasterAction("READ_PAPER", (result.task,)), (result,))
                  for i, result in enumerate(results, 1))
    return AgentState(findings=results, steps=steps)


def master_payload(state, mode="incremental-with-evidence"):
    class LLM:
        def complete_json(self, prompt, **kwargs):
            self.payload = json.loads(prompt)
            return {"kind": "DECIDE", "assessment": "bounded", "stop_reason_code": "evidence_sufficient"}
    llm = LLM()
    runtime.PaperAgentMaster(llm=llm, paper_name="paper.pdf", page_index="[]",
        overview_text="overview", overview_images=(), reflection_enabled=True,
        master_context_mode=mode)(state)
    return llm.payload


@pytest.mark.parametrize("mode", ["full-history", "incremental-with-evidence"])
def test_review_status_tracks_snapshot_separately_from_actual_union_input(mode):
    state = state_with_findings()
    # The second Worker result is from the branch parallel to this Reflection.
    report = ReflectionReport("master_requested", ("r1-t1-f1",), "memo",
        context_mode="rubric-union", context_finding_ids=())
    payload = master_payload(replace(state, reflection_reports=(report,)), mode)
    status = payload["state"]["finding_review_status"]
    assert status["new_since_last_successful_reflection"] == ["r2-t1-f1"]
    assert status["never_supplied_to_successful_reflection"] == ["r1-t1-f1", "r2-t1-f1"]
    assert "quote 1" in json.dumps(payload["state"])
    assert "quote 2" in json.dumps(payload["state"])
    assert list(payload["state"])[0] == "finding_review_status"


def test_review_status_accumulates_successes_and_ignores_failed_or_empty_memos():
    first = ReflectionReport("after_method", ("r1-t1-f1",), "first")
    union = ReflectionReport("master_requested", ("r1-t1-f1", "r2-t1-f1"), "second",
        context_mode="rubric-union", context_finding_ids=("r2-t1-f1",))
    failed = replace(union, error="timeout")
    empty = replace(union, reflection_memo=" ")
    state = state_with_findings()
    status = master_payload(replace(state, reflection_reports=(first, failed, empty)))["state"]["finding_review_status"]
    assert status["new_since_last_successful_reflection"] == ["r2-t1-f1"]
    assert status["never_supplied_to_successful_reflection"] == ["r2-t1-f1"]
    status = master_payload(replace(state, reflection_reports=(first, union, failed)))["state"]["finding_review_status"]
    assert status["new_since_last_successful_reflection"] == []
    assert status["never_supplied_to_successful_reflection"] == []


def test_no_successful_reflection_marks_all_findings_pending_without_forcing_action():
    payload = master_payload(state_with_findings())
    assert payload["state"]["finding_review_status"] == {
        "new_since_last_successful_reflection": ["r1-t1-f1", "r2-t1-f1"],
        "never_supplied_to_successful_reflection": ["r1-t1-f1", "r2-t1-f1"],
    }


def test_older_rubric_omission_is_not_mislabeled_as_new():
    report = ReflectionReport("master_requested", ("r1-t1-f1", "r2-t1-f1"), "memo",
        context_mode="rubric-union", context_finding_ids=("r1-t1-f1",))
    payload = master_payload(replace(state_with_findings(), reflection_reports=(report,)))
    assert payload["state"]["finding_review_status"] == {
        "new_since_last_successful_reflection": [],
        "never_supplied_to_successful_reflection": ["r2-t1-f1"],
    }


def test_master_prefix_is_stable_across_research_phases():
    first = master_payload(AgentState())
    later = master_payload(replace(state_with_findings(), reflection_reports=(
        ReflectionReport("after_method", ("r1-t1-f1",), "memo"),)))
    assert first["phase_instructions"] != later["phase_instructions"]
    first_parts = cache_friendly_prompt_parts(json.dumps(first))
    later_parts = cache_friendly_prompt_parts(json.dumps(later))
    assert first_parts[0] == later_parts[0]
    assert '"phase_instructions"' not in first_parts[0]
    assert json.loads("".join(later_parts)) == later


def test_worker_and_locator_prefixes_stay_stable_when_task_context_changes():
    for build, fields in [
        (runtime._build_locator_prompt, {"page_index": "[]", "page_count": 2}),
        (runtime._build_evidence_prompt, {"paper_name": "paper.pdf", "location_rationale": "why",
            "selected_text": {1: "quote"}, "image_inputs": ()}),
    ]:
        first = build(question="question", **fields)
        extra = {"role_instructions": ("Independently cross-check the source.",)} if build is runtime._build_evidence_prompt else {
            "navigation_snippets": [{"page_number": 2, "text": "dynamic hint"}],
        }
        later = build(question="follow up", decision_context="unverified lead",
            rubric_ids=("core_contribution",),
            research_context=completed_finding_records(state_with_findings())[:1], **fields, **extra)
        assert cache_friendly_prompt_parts(first)[0] == cache_friendly_prompt_parts(later)[0]
        assert "unverified lead" in cache_friendly_prompt_parts(later)[1]
        assert json.loads("".join(cache_friendly_prompt_parts(later))) == json.loads(later)


def test_worker_preserves_qualifications_and_explicit_role_reference_contract():
    payload = json.loads(runtime._build_evidence_prompt(paper_name="paper.pdf", question="How?",
        location_rationale="why", selected_text={1: "source"}, image_inputs=()))
    instructions = " ".join(payload["instructions"])
    assert "author-reported qualifications" in instructions
    assert "sensitivity analyses" in instructions
    assert "explicit cross-reference" in instructions
    assert "caveat" in instructions and "selected pages" in instructions
