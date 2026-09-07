from threading import Event

import pytest

from deep_research import experiment
from deep_research.paper_agent import AgentState, EvidenceTask, MasterAction, ReflectionReport, TraceStep, WorkerResult, run_paper_agent
from deep_research.paper_agent_runtime import PaperAgentMaster, parse_master_action, _state_payload, _master_state_payload, _role_mixed_master_instructions


def result(task):
    return WorkerResult(task, finding=task.question, evidence="quoted source", evidence_type="text", evidence_locator="p. 1")


def report(state, *, trigger, finding_ids, proposed_decision=None):
    return ReflectionReport(trigger, finding_ids, "A bounded relationship check.")


def request(kind="REFLECT", **kwargs):
    return parse_master_action({
        "kind": kind, "reflection_focus": "Does the template qualify the absence claim?",
        "reflection_rubric_ids": ["core_contribution"], **kwargs,
    })


def test_standalone_reflection_uses_existing_evidence_and_returns_to_master():
    states = []
    actions = iter([MasterAction("READ_PAPER", (EvidenceTask("initial"),)), request(), MasterAction("DECIDE", assessment="bounded")])
    def master(state):
        states.append(state)
        return next(actions)
    trace = run_paper_agent(master=master, worker=result, reflector=report, max_reflections=2, max_rounds=3)
    assert trace.outcome == "DECIDE"
    assert [step.action.kind for step in trace.steps] == ["READ_PAPER", "REFLECT", "DECIDE"]
    assert [r.trigger for r in trace.reflection_reports] == ["post_method_model", "master_requested"]
    assert states[1].findings == states[2].findings
    assert len(states[2].reflection_reports) == 2
    assert trace.deferred_decisions == ()


def test_combined_actions_overlap_and_use_only_pre_round_snapshot():
    worker_started, reflection_started = Event(), Event()
    seen = []
    combined = request("READ_PAPER_AND_REFLECT", tasks=[{"question": "new evidence", "source_scope": "paper"}],
        conclusion_at_risk="baseline comparability", missing_evidence="judge configuration", expected_judgment_delta="qualify comparison",
        independence_rationale="The template check uses prior findings and does not depend on the new judge evidence.")
    actions = iter([MasterAction("READ_PAPER", (EvidenceTask("initial"),)), combined, MasterAction("DECIDE", assessment="bounded")])
    def worker(task):
        if task.question == "new evidence":
            worker_started.set()
            assert reflection_started.wait(3), "Reflection must overlap Worker execution"
        return result(task)
    def reflector(state, *, trigger, finding_ids, proposed_decision=None):
        if trigger == "master_requested":
            reflection_started.set()
            assert worker_started.wait(3), "Worker must overlap Reflection"
            assert finding_ids == ("r1-t1-f1",)
            assert len(state.findings) == 1
            assert state.findings[0].task.question == "initial"
        return report(state, trigger=trigger, finding_ids=finding_ids)
    def master(state):
        seen.append(state)
        return next(actions)
    trace = run_paper_agent(master=master, worker=worker, reflector=reflector, max_reflections=2, max_rounds=3)
    assert trace.outcome == "DECIDE"
    assert worker_started.is_set() and reflection_started.is_set()
    assert len(seen[-1].findings) == 2 and len(seen[-1].reflection_reports) == 2
    assert not any(r.error for r in trace.reflection_reports)
    assert not any(r.error for step in trace.steps for r in step.results)


@pytest.mark.parametrize("kind", ["DECIDE", "NEEDS_HUMAN", "READ_PAPER"])
def test_reflection_cannot_be_hidden_in_another_action(kind):
    with pytest.raises(ValueError, match="reflection"):
        request(kind, stop_reason_code="evidence_sufficient", conclusion_at_risk="x", missing_evidence="x", expected_judgment_delta="x")


@pytest.mark.parametrize("payload", [
    {"kind": "DECIDE", "tasks": [{"question": "q", "source_scope": "paper"}], "stop_reason_code": "evidence_sufficient"},
    {"kind": "REFLECT"},
    {"kind": "READ_PAPER_AND_REFLECT", "reflection_focus": "q", "reflection_rubric_ids": ["core_contribution"],
     "tasks": [{"question": "q", "source_scope": "paper"}], "conclusion_at_risk": "x", "missing_evidence": "x", "expected_judgment_delta": "x"},
])
def test_parser_rejects_incomplete_or_mixed_actions(payload):
    with pytest.raises(ValueError):
        parse_master_action(payload)


def test_reflection_requires_evidence_and_first_mechanism_review():
    trace = run_paper_agent(master=lambda state: request(), worker=lambda task: pytest.fail("must not read"),
        reflector=lambda *a, **kw: pytest.fail("must not reflect"), max_reflections=2, max_rounds=3)
    assert trace.outcome == "NEEDS_HUMAN"
    assert "reflection_requires_existing_evidence" in trace.stop_reason


def test_reflection_budget_prevents_third_review():
    actions = iter([MasterAction("READ_PAPER", (EvidenceTask("initial"),)), request(), request()])
    trace = run_paper_agent(master=lambda state: next(actions), worker=result, reflector=report, max_reflections=2, max_rounds=4)
    assert trace.outcome == "NEEDS_HUMAN"
    assert "reflection_budget_exhausted" in trace.stop_reason
    assert len(trace.reflection_reports) == 2


def test_real_master_payload_explicitly_permits_both_and_exposes_budget():
    class Client:
        def complete_json(self, prompt, **kwargs):
            import json
            self.payload = json.loads(prompt)
            return {"kind": "NEEDS_HUMAN"}
    client = Client()
    PaperAgentMaster(llm=client, paper_name="paper", page_index="[]", overview_text="overview", overview_images=(), reflection_enabled=True)(AgentState(remaining_rounds=5))
    prompt = client.payload
    assert {"REFLECT", "READ_PAPER_AND_REFLECT"} <= set(prompt["output_contract"])
    instructions = " ".join(prompt["instructions"])
    assert "You may choose READ_PAPER and REFLECT together" in instructions
    assert "same pre-round evidence snapshot" in instructions
    assert "Missing source evidence belongs to READ_PAPER" in instructions
    assert prompt["reflection_budget"]["remaining"] == 2
    assert prompt["reflection_budget"]["request_available"] is False


def test_default_target_follows_paper_claims_not_presumed_evolution():
    assert "paper's own" in experiment.INVESTIGATION_TARGET
    assert "state or artifact that evolves continuously" not in experiment.INVESTIGATION_TARGET
    assert "only when the paper explicitly claims" in experiment.INVESTIGATION_TARGET


def test_reflection_context_preserves_request_and_last_worker_suggestions():
    from dataclasses import replace
    task = EvidenceTask("initial")
    worker_result = replace(result(task), suggested_questions=("Missing source passage?",))
    action = request()
    state = AgentState(steps=(
        TraceStep(1, MasterAction("READ_PAPER", (task,)), (worker_result,)), TraceStep(2, action),
    ))
    assert _state_payload(state)["history"][-1]["reflection_focus"] == action.reflection_focus
    latest = _master_state_payload(state)
    assert latest["latest_action"]["reflection_rubric_ids"] == ["core_contribution"]
    assert latest["latest_worker_suggested_questions"][0]["questions"] == ["Missing source passage?"]


def test_optional_role_mode_does_not_route_existing_relationships_to_worker():
    instructions = " ".join(_role_mixed_master_instructions("discovery-cross-check"))
    assert "only when missing source evidence" in instructions
    assert "existing-evidence integration belongs to REFLECT" in instructions


def test_combined_reflection_error_retains_completed_worker_evidence():
    combined = request("READ_PAPER_AND_REFLECT", tasks=[{"question": "new", "source_scope": "paper"}],
        conclusion_at_risk="x", missing_evidence="x", expected_judgment_delta="x", independence_rationale="independent")
    actions = iter([MasterAction("READ_PAPER", (EvidenceTask("initial"),)), combined])
    def reflector(state, *, trigger, finding_ids, proposed_decision=None):
        if trigger == "master_requested":
            raise RuntimeError("review unavailable")
        return report(state, trigger=trigger, finding_ids=finding_ids)
    trace = run_paper_agent(master=lambda state: next(actions), worker=result, reflector=reflector, max_reflections=2, max_rounds=3)
    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.steps[-1].results[0].finding == "new"
    assert "review unavailable" in trace.reflection_reports[-1].error
