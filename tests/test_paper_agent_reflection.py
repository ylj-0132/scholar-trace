from __future__ import annotations

import pytest

from deep_research.paper_agent import (
    AgentState,
    EvidenceTask,
    MasterAction,
    ReflectionReport,
    WorkerResult,
    run_paper_agent,
)
from deep_research.paper_agent_runtime import parse_reflection_report
from deep_research.paper_agent_runtime import _state_payload


def _result(task: EvidenceTask) -> WorkerResult:
    return WorkerResult(
        task=task,
        finding=f"finding for {task.question}",
        evidence=f"evidence for {task.question}",
        caveat="bounded",
        evidence_type="text",
        evidence_locator="p. 1",
    )


def _report(trigger: str, finding_ids: tuple[str, ...]) -> ReflectionReport:
    return ReflectionReport(
        trigger=trigger,
        reflected_finding_ids=finding_ids,
        reflection_memo=(
            "The findings suggest a relationship that may be explained by a coupled "
            "setting. The Master should check whether the reported comparison isolates "
            "that setting before relying on the mechanism claim."
        ),
    )


def test_post_method_model_reflection_is_visible_to_the_next_master() -> None:
    seen_reflection_counts: list[int] = []
    reflection_calls: list[tuple[str, tuple[str, ...]]] = []

    def master(state: AgentState) -> MasterAction:
        seen_reflection_counts.append(len(state.reflection_reports))
        if not state.steps:
            return MasterAction("READ_PAPER", (EvidenceTask("first question"),))
        return MasterAction("DECIDE", assessment="A bounded conclusion.")

    def reflector(state: AgentState, *, trigger: str, finding_ids: tuple[str, ...], proposed_decision: MasterAction | None = None) -> ReflectionReport:
        del proposed_decision
        reflection_calls.append((trigger, finding_ids))
        return _report(trigger, finding_ids)

    trace = run_paper_agent(
        master=master,
        worker=_result,
        reflector=reflector,
        max_reflections=2,
        max_rounds=3,
    )

    assert trace.outcome == "DECIDE"
    assert seen_reflection_counts == [0, 1]
    assert reflection_calls == [("post_method_model", ("r1-t1-f1",))]
    assert "coupled setting" in trace.reflection_reports[0].reflection_memo


def test_pre_decide_reflection_is_skipped_without_a_named_focus() -> None:
    master_states: list[AgentState] = []
    reflection_calls: list[tuple[str, tuple[str, ...]]] = []

    def master(state: AgentState) -> MasterAction:
        master_states.append(state)
        if not state.steps:
            return MasterAction("READ_PAPER", (EvidenceTask("first"),))
        if len(state.steps) == 1:
            return MasterAction("READ_PAPER", (EvidenceTask("second"),))
        return MasterAction(
            "DECIDE",
            assessment="Enough for a bounded judgment.",
            stop_reason_code="paper_saturated",
        )

    def reflector(state: AgentState, *, trigger: str, finding_ids: tuple[str, ...], proposed_decision: MasterAction | None = None) -> ReflectionReport:
        del proposed_decision
        reflection_calls.append((trigger, finding_ids))
        return _report(trigger, finding_ids)

    trace = run_paper_agent(
        master=master,
        worker=_result,
        reflector=reflector,
        max_reflections=2,
        max_rounds=3,
    )

    assert trace.outcome == "DECIDE"
    assert reflection_calls == [("post_method_model", ("r1-t1-f1",))]
    assert len(master_states) == 3
    assert trace.deferred_decisions == ()
    assert trace.steps[-1].action.kind == "DECIDE"


def test_pre_decide_reflection_runs_once_for_a_named_mechanism_conflict() -> None:
    def master(state: AgentState) -> MasterAction:
        if not state.steps:
            return MasterAction("READ_PAPER", (EvidenceTask("first"),))
        if len(state.steps) == 1:
            return MasterAction("READ_PAPER", (EvidenceTask("second"),))
        return MasterAction(
            "DECIDE",
            assessment="Enough for a bounded judgment.",
            stop_reason_code="evidence_sufficient",
            pre_decide_reflection_focus="Whether joint rewriting changes utility credit.",
        )

    trace = run_paper_agent(
        master=master,
        worker=_result,
        reflector=lambda state, trigger, finding_ids, proposed_decision=None: _report(
            trigger, finding_ids
        ),
        max_reflections=2,
        max_rounds=3,
    )

    assert [report.trigger for report in trace.reflection_reports] == [
        "post_method_model",
        "pre_decide",
    ]
    assert len(trace.deferred_decisions) == 1


def test_pre_decide_reflection_receives_the_deferred_decision() -> None:
    proposed: list[MasterAction | None] = []

    def master(state: AgentState) -> MasterAction:
        if not state.steps:
            return MasterAction("READ_PAPER", (EvidenceTask("first"),))
        if len(state.steps) == 1:
            return MasterAction("READ_PAPER", (EvidenceTask("second"),))
        return MasterAction(
            "DECIDE",
            assessment="A proposed conclusion.",
            rationale="Evidence is sufficient.",
            unresolved_questions=("A bounded open question.",),
            checklist_coverage={"core_contribution": "covered"},
            stop_reason_code="evidence_sufficient",
            pre_decide_reflection_focus="Whether the remaining caveat changes the conclusion.",
        )

    def reflector(
        state: AgentState,
        *,
        trigger: str,
        finding_ids: tuple[str, ...],
        proposed_decision: MasterAction | None,
    ) -> ReflectionReport:
        del state
        proposed.append(proposed_decision)
        return _report(trigger, finding_ids)

    trace = run_paper_agent(
        master=master,
        worker=_result,
        reflector=reflector,
        max_reflections=2,
        max_rounds=3,
    )

    assert trace.outcome == "DECIDE"
    assert proposed[0] is None
    assert proposed[1] is not None
    assert proposed[1].assessment == "A proposed conclusion."
    assert proposed[1].rationale == "Evidence is sufficient."
    assert proposed[1].unresolved_questions == ("A bounded open question.",)
    assert proposed[1].checklist_coverage == {"core_contribution": "covered"}


def test_required_reflection_failure_fails_closed_without_accepting_decide() -> None:
    def master(state: AgentState) -> MasterAction:
        if not state.steps:
            return MasterAction("READ_PAPER", (EvidenceTask("first"),))
        return MasterAction("DECIDE", assessment="must not be accepted")

    def reflector(_state: AgentState, **_kwargs: object) -> ReflectionReport:
        raise RuntimeError("reflector unavailable")

    trace = run_paper_agent(
        master=master,
        worker=_result,
        reflector=reflector,
        max_reflections=2,
        max_rounds=2,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.assessment is None
    assert trace.stop_reason.startswith("reflection_error@round_1:RuntimeError")
    assert trace.reflection_reports[0].error == "RuntimeError: reflector unavailable"


def test_required_method_reflection_rejects_decide_before_any_evidence() -> None:
    trace = run_paper_agent(
        master=lambda _state: MasterAction("DECIDE", assessment="Too early."),
        worker=lambda _task: pytest.fail("Worker must not run"),
        reflector=lambda *_args, **_kwargs: pytest.fail("Reflection needs evidence first"),
        max_reflections=2,
        max_rounds=2,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.assessment is None
    assert trace.stop_reason == "decide_before_required_method_reflection"
    assert trace.reflection_reports == ()


def test_reflection_parser_requires_one_nonempty_prose_memo() -> None:
    report = parse_reflection_report(
        {"reflection_memo": "No material design gap is supported by the current evidence."},
        trigger="post_method_model",
        reflected_finding_ids=("r1-t1-f1",),
    )

    assert report.reflection_memo.startswith("No material design gap")
    for payload in ({}, {"reflection_memo": ""}, {"reflection_memo": []}):
        with pytest.raises(ValueError, match="reflection_memo"):
            parse_reflection_report(
                payload,
                trigger="post_method_model",
                reflected_finding_ids=(),
            )


def test_reflection_parser_does_not_accept_the_old_hypothesis_card_shape() -> None:
    with pytest.raises(ValueError, match="reflection_memo"):
        parse_reflection_report(
            {"hypotheses": []},
            trigger="post_method_model",
            reflected_finding_ids=(),
        )


def test_pre_decide_reflection_can_return_master_to_reading_without_becoming_evidence() -> None:
    calls = 0

    def master(state: AgentState) -> MasterAction:
        nonlocal calls
        calls += 1
        if not state.steps:
            return MasterAction("READ_PAPER", (EvidenceTask("first"),))
        if len(state.steps) == 1:
            return MasterAction("READ_PAPER", (EvidenceTask("second"),))
        if len(state.reflection_reports) == 2 and len(state.steps) == 2:
            return MasterAction("READ_PAPER", (EvidenceTask("verification"),))
        return MasterAction(
            "DECIDE",
            assessment="final bounded conclusion",
            pre_decide_reflection_focus="Whether the verification changes the conclusion.",
        )

    trace = run_paper_agent(
        master=master,
        worker=_result,
        reflector=lambda state, trigger, finding_ids, proposed_decision=None: _report(trigger, finding_ids),
        max_reflections=2,
        max_rounds=4,
    )

    assert trace.outcome == "DECIDE"
    assert [step.action.tasks[0].question for step in trace.steps[:-1]] == [
        "first", "second", "verification"
    ]
    assert len(trace.reflection_reports) == 2
    state = AgentState(
        findings=tuple(result for step in trace.steps for result in step.results),
        steps=trace.steps,
        reflection_reports=trace.reflection_reports,
    )
    payload = _state_payload(state)
    assert "coupled setting" in payload["reflection_reports"][0]["reflection_memo"]
    assert all("reflection" not in finding for finding in payload["findings"])
    assert calls == 5
