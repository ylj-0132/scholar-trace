from __future__ import annotations

import json
import threading

import pytest

from deep_research.paper_agent import (
    AgentState,
    EvidenceTask,
    FindingEvidence,
    MasterAction,
    TraceStep,
    WorkerFinding,
    WorkerResult,
    _selected_research_context,
    completed_finding_records,
    run_paper_agent,
)


def test_decide_records_structured_stop_reason() -> None:
    trace = run_paper_agent(
        master=lambda _state: MasterAction(
            "DECIDE",
            assessment="Enough evidence.",
            stop_reason_code="evidence_sufficient",
        ),
        worker=lambda _task: pytest.fail("worker must not run"),
        max_rounds=1,
    )

    assert trace.stop_reason_code == "evidence_sufficient"
    assert json.loads(trace.to_json())["stop_reason_code"] == "evidence_sufficient"


def test_next_round_depends_on_returned_evidence() -> None:
    seen_questions: list[str] = []

    def master(state: AgentState) -> MasterAction:
        if not state.findings:
            return MasterAction(
                kind="READ_PAPER",
                tasks=(EvidenceTask("Does the planner improve accuracy?"),),
            )
        if len(state.findings) == 1:
            assert "no accuracy gain" in state.findings[0].finding
            return MasterAction(
                kind="READ_PAPER",
                tasks=(EvidenceTask("Does the planner reduce tokens?"),),
            )
        return MasterAction(
            kind="DECIDE",
            assessment="Planner value is efficiency, not demonstrated accuracy.",
        )

    def worker(task: EvidenceTask) -> WorkerResult:
        seen_questions.append(task.question)
        if "accuracy" in task.question:
            return WorkerResult(
                task=task,
                finding="The adaptive planner shows no accuracy gain over fixed MIXED.",
                evidence="Table 3: MIXED 92.21, adaptive 92.01.",
                caveat="The comparison is benchmark-specific.",
            )
        return WorkerResult(
            task=task,
            finding="The planner reduces retrieval tokens by about 1.8x.",
            evidence="Efficiency analysis reports the token reduction.",
        )

    trace = run_paper_agent(master=master, worker=worker, max_rounds=3)

    assert trace.outcome == "DECIDE"
    assert trace.assessment == "Planner value is efficiency, not demonstrated accuracy."
    assert seen_questions == [
        "Does the planner improve accuracy?",
        "Does the planner reduce tokens?",
    ]
    assert [step.action.kind for step in trace.steps] == [
        "READ_PAPER",
        "READ_PAPER",
        "DECIDE",
    ]


def test_master_waits_for_all_workers_in_the_current_round() -> None:
    events: list[str] = []

    def master(state: AgentState) -> MasterAction:
        events.append("master")
        if not state.findings:
            return MasterAction(
                kind="READ_PAPER",
                tasks=(EvidenceTask("one"), EvidenceTask("two")),
            )
        return MasterAction(kind="NEEDS_HUMAN", rationale="batch complete")

    def worker(task: EvidenceTask) -> WorkerResult:
        events.append(f"worker:{task.question}")
        return WorkerResult(task=task, finding=task.question)

    trace = run_paper_agent(master=master, worker=worker, max_rounds=2)

    assert trace.outcome == "NEEDS_HUMAN"
    assert events == ["master", "worker:one", "worker:two", "master"]


def test_three_tasks_execute_before_next_master_call() -> None:
    events: list[str] = []

    def master(state: AgentState) -> MasterAction:
        events.append("master")
        if not state.findings:
            return MasterAction(
                kind="READ_PAPER",
                tasks=(
                    EvidenceTask("one"),
                    EvidenceTask("two"),
                    EvidenceTask("three"),
                ),
            )
        return MasterAction(kind="NEEDS_HUMAN", rationale="batch complete")

    def worker(task: EvidenceTask) -> WorkerResult:
        events.append(f"worker:{task.question}")
        return WorkerResult(task=task, finding=task.question)

    trace = run_paper_agent(master=master, worker=worker, max_rounds=2)

    assert trace.outcome == "NEEDS_HUMAN"
    assert events == ["master", "worker:one", "worker:two", "worker:three", "master"]
    assert len(trace.steps[0].results) == 3


def test_worker_failure_is_visible_to_master_and_trace() -> None:
    def master(state: AgentState) -> MasterAction:
        if not state.findings:
            return MasterAction(
                kind="READ_PAPER",
                tasks=(EvidenceTask("Find the ablation."),),
            )
        assert state.findings[0].error == "RuntimeError: unreadable page"
        return MasterAction(
            kind="NEEDS_HUMAN",
            rationale="The decisive evidence could not be read.",
        )

    def worker(task: EvidenceTask) -> WorkerResult:
        del task
        raise RuntimeError("unreadable page")

    trace = run_paper_agent(master=master, worker=worker, max_rounds=2)

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.steps[0].results[0].error == "RuntimeError: unreadable page"
    assert trace.stop_reason == "The decisive evidence could not be read."


def test_round_budget_exhaustion_escalates_instead_of_guessing() -> None:
    def master(state: AgentState) -> MasterAction:
        del state
        return MasterAction(kind="READ_PAPER", tasks=(EvidenceTask("Keep reading."),))

    def worker(task: EvidenceTask) -> WorkerResult:
        return WorkerResult(task=task, finding="partial", evidence="p. 1")

    trace = run_paper_agent(master=master, worker=worker, max_rounds=1)

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.assessment is None
    assert trace.stop_reason == "round_budget_exhausted"


def test_trace_is_json_serializable() -> None:
    def master(state: AgentState) -> MasterAction:
        if not state.findings:
            return MasterAction(kind="READ_PAPER", tasks=(EvidenceTask("question"),))
        return MasterAction(
            kind="NEEDS_HUMAN",
            rationale="Worker result is inspectable.",
        )

    trace = run_paper_agent(
        master=master,
        worker=lambda task: WorkerResult(
            task=task,
            finding="found",
            evidence="page 1",
            caveat="limited",
        ),
        max_rounds=2,
    )

    payload = json.loads(trace.to_json())
    assert payload["outcome"] == "NEEDS_HUMAN"
    assert payload["stop_reason"] == "Worker result is inspectable."
    assert payload["steps"][0]["results"][0]["finding"] == "found"
    assert payload["steps"][0]["results"][0]["evidence"] == "page 1"
    assert payload["steps"][0]["results"][0]["caveat"] == "limited"
    assert payload["source_document"] is None


def test_worker_suggested_questions_default_and_trace_are_compatible() -> None:
    task = EvidenceTask("Inspect the ablation.")
    default_result = WorkerResult(task=task)
    suggested_result = WorkerResult(
        task=task,
        finding="A local result raises a follow-up.",
        suggested_questions=(
            "Does the ablation also change the available retrieval budget?",
        ),
    )

    trace = run_paper_agent(
        master=lambda state: MasterAction(
            "READ_PAPER", (task,)
        ) if not state.findings else MasterAction("NEEDS_HUMAN"),
        worker=lambda current: suggested_result,
        max_rounds=2,
    )

    assert default_result.suggested_questions == ()
    assert trace.steps[0].results[0].suggested_questions == (
        "Does the ablation also change the available retrieval budget?",
    )
    assert json.loads(trace.to_json())["steps"][0]["results"][0][
        "suggested_questions"
    ] == ["Does the ablation also change the available retrieval budget?"]


def test_worker_suggestions_do_not_create_tasks_automatically() -> None:
    worker_calls: list[str] = []
    first = EvidenceTask("Inspect the reported comparison.")

    def master(state: AgentState) -> MasterAction:
        if not state.findings:
            return MasterAction("READ_PAPER", (first,))
        return MasterAction("NEEDS_HUMAN")

    def worker(task: EvidenceTask) -> WorkerResult:
        worker_calls.append(task.question)
        return WorkerResult(
            task=task,
            suggested_questions=("Check the corresponding budget control.",),
        )

    run_paper_agent(master=master, worker=worker, max_rounds=2)

    assert worker_calls == ["Inspect the reported comparison."]


def test_phase1_trace_has_no_source_document_by_default() -> None:
    trace = run_paper_agent(
        master=lambda state: MasterAction(kind="NEEDS_HUMAN"),
        worker=lambda task: WorkerResult(task=task),
        max_rounds=1,
    )

    assert trace.source_document is None
    assert json.loads(trace.to_json())["source_document"] is None


@pytest.mark.parametrize(
    ("kind", "rationale", "expected_reason"),
    [
        ("DECIDE", "   ", "master_decided"),
        ("NEEDS_HUMAN", " \t ", "master_requested_human"),
        ("DECIDE", "  evidence complete  ", "evidence complete"),
    ],
)
def test_terminal_rationale_is_normalized(
    kind: str,
    rationale: str,
    expected_reason: str,
) -> None:
    action = MasterAction(
        kind=kind,
        assessment="assessment" if kind == "DECIDE" else None,
        rationale=rationale,
    )

    trace = run_paper_agent(
        master=lambda state: action,
        worker=lambda task: WorkerResult(task=task),
        max_rounds=1,
    )

    assert trace.stop_reason == expected_reason


def test_state_tracks_remaining_rounds_and_replaces_unresolved_questions() -> None:
    observations: list[tuple[tuple[str, ...], int]] = []

    def master(state: AgentState) -> MasterAction:
        observations.append((state.unresolved_questions, state.remaining_rounds))
        if state.remaining_rounds == 3:
            return MasterAction(
                kind="READ_PAPER",
                tasks=(EvidenceTask("first"),),
                unresolved_questions=("novelty", "evidence"),
            )
        if state.remaining_rounds == 2:
            return MasterAction(
                kind="READ_PAPER",
                tasks=(EvidenceTask("second"),),
                unresolved_questions=("evidence",),
            )
        return MasterAction(kind="NEEDS_HUMAN")

    trace = run_paper_agent(
        master=master,
        worker=lambda task: WorkerResult(task=task, finding="done"),
        max_rounds=3,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    assert observations == [
        ((), 3),
        (("novelty", "evidence"), 2),
        (("evidence",), 1),
    ]


@pytest.mark.parametrize("bad_output", [None, {}, "bad"])
def test_invalid_master_output_escalates_without_workers(bad_output: object) -> None:
    worker_calls = 0

    def master(state: AgentState) -> object:
        del state
        return bad_output

    def worker(task: EvidenceTask) -> WorkerResult:
        nonlocal worker_calls
        worker_calls += 1
        return WorkerResult(task=task)

    trace = run_paper_agent(master=master, worker=worker, max_rounds=2)

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.assessment is None
    assert trace.stop_reason == f"invalid_master_output@round_1:{type(bad_output).__name__}"
    assert worker_calls == 0
    assert trace.steps == ()


def test_master_exception_preserves_completed_steps() -> None:
    calls = 0

    def master(state: AgentState) -> MasterAction:
        nonlocal calls
        calls += 1
        if calls == 1:
            return MasterAction(kind="READ_PAPER", tasks=(EvidenceTask("first"),))
        raise RuntimeError("master unavailable")

    def worker(task: EvidenceTask) -> WorkerResult:
        return WorkerResult(task=task, finding="first result")

    trace = run_paper_agent(master=master, worker=worker, max_rounds=2)

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.stop_reason == "master_error@round_2:RuntimeError: master unavailable"
    assert len(trace.steps) == 1
    assert trace.steps[0].results[0].finding == "first result"


@pytest.mark.parametrize(
    ("action", "expected_reason"),
    [
        (MasterAction(kind="READ_PAPER"), "read_paper_without_tasks"),
        (MasterAction(kind="DECIDE"), "decide_without_assessment"),
        (MasterAction(kind="UNSUPPORTED"), "unsupported_action:UNSUPPORTED"),
        (
            MasterAction(kind="READ_PAPER", tasks=(EvidenceTask("   "),)),
            "read_paper_invalid_task_1:empty_question",
        ),
        (
            MasterAction(
                kind="READ_PAPER",
                tasks=(EvidenceTask("question", source_scope="web"),),
            ),
            "read_paper_invalid_task_1:unsupported_source_scope:web",
        ),
    ],
)
def test_invalid_master_actions_stop_without_workers(
    action: MasterAction,
    expected_reason: str,
) -> None:
    worker_calls = 0

    def master(state: AgentState) -> MasterAction:
        del state
        return action

    def worker(task: EvidenceTask) -> WorkerResult:
        nonlocal worker_calls
        worker_calls += 1
        return WorkerResult(task=task)

    trace = run_paper_agent(master=master, worker=worker, max_rounds=1)

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.assessment is None
    assert trace.stop_reason == expected_reason
    assert worker_calls == 0
    assert trace.steps[0].action == action


def test_empty_question_is_invalid_even_when_task_is_not_first() -> None:
    action = MasterAction(
        kind="READ_PAPER",
        tasks=(EvidenceTask("valid"), EvidenceTask("  ")),
    )

    trace = run_paper_agent(
        master=lambda state: action,
        worker=lambda task: WorkerResult(task=task),
        max_rounds=1,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.stop_reason == "read_paper_invalid_task_2:empty_question"


def test_read_paper_provisional_assessment_reaches_the_next_master_round() -> None:
    observations: list[str | None] = []

    def master(state: AgentState) -> MasterAction:
        observations.append(state.provisional_assessment)
        if not state.findings:
            return MasterAction(
                kind="READ_PAPER",
                tasks=(EvidenceTask("find evidence"),),
                assessment="provisional contribution",
            )
        return MasterAction(kind="NEEDS_HUMAN")

    trace = run_paper_agent(
        master=master,
        worker=lambda task: WorkerResult(task=task, finding="evidence"),
        max_rounds=2,
    )

    assert observations == [None, "provisional contribution"]
    assert trace.steps[0].action.assessment == "provisional contribution"
    assert trace.steps[0].results[0].finding == "evidence"


def test_provisional_assessment_survives_a_later_null_action() -> None:
    observations: list[str | None] = []
    def master(state: AgentState) -> MasterAction:
        observations.append(state.provisional_assessment)
        if len(observations) == 1:
            return MasterAction("READ_PAPER", (EvidenceTask("first"),), assessment="kept")
        if len(observations) == 2:
            return MasterAction("READ_PAPER", (EvidenceTask("second"),))
        return MasterAction("NEEDS_HUMAN")
    run_paper_agent(master=master, worker=lambda task: WorkerResult(task=task), max_rounds=3)
    assert observations == [None, "kept", "kept"]


def test_needs_human_preserves_provisional_assessment() -> None:
    trace = run_paper_agent(
        master=lambda state: MasterAction(
            kind="NEEDS_HUMAN",
            assessment="evidence remains inconclusive",
        ),
        worker=lambda task: WorkerResult(task=task),
        max_rounds=1,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.assessment == "evidence remains inconclusive"


def test_decide_with_tasks_is_rejected_without_running_tasks() -> None:
    worker_calls = 0

    def worker(task: EvidenceTask) -> WorkerResult:
        nonlocal worker_calls
        worker_calls += 1
        return WorkerResult(task=task)

    action = MasterAction(
        kind="DECIDE",
        assessment="decided assessment",
        tasks=(EvidenceTask("do not run"),),
    )
    trace = run_paper_agent(master=lambda state: action, worker=worker, max_rounds=1)

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.assessment is None
    assert trace.stop_reason == "exclusive_action_with_reading_tasks"
    assert trace.steps[0].action == action
    assert worker_calls == 0


@pytest.mark.parametrize(
    ("worker_output", "expected_error"),
    [
        (None, "invalid_worker_output:NoneType"),
        ({}, "invalid_worker_output:dict"),
    ],
)
def test_invalid_worker_output_is_returned_to_master(
    worker_output: object,
    expected_error: str,
) -> None:
    def master(state: AgentState) -> MasterAction:
        if not state.findings:
            return MasterAction(kind="READ_PAPER", tasks=(EvidenceTask("question"),))
        assert state.findings[0].error == expected_error
        return MasterAction(kind="NEEDS_HUMAN", rationale="invalid result visible")

    trace = run_paper_agent(
        master=master,
        worker=lambda task: worker_output,
        max_rounds=2,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.steps[0].results[0].error == expected_error
    payload = json.loads(trace.to_json())
    assert payload["steps"][0]["results"][0]["error"] == expected_error


def test_worker_task_mismatch_is_replaced_with_current_task_error() -> None:
    current_task = EvidenceTask("current")
    wrong_task = EvidenceTask("wrong")

    def master(state: AgentState) -> MasterAction:
        if not state.findings:
            return MasterAction(kind="READ_PAPER", tasks=(current_task,))
        assert state.findings[0].task == current_task
        assert state.findings[0].error == "worker_task_mismatch"
        return MasterAction(kind="NEEDS_HUMAN")

    trace = run_paper_agent(
        master=master,
        worker=lambda task: WorkerResult(task=wrong_task, finding="wrong"),
        max_rounds=2,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    result = trace.steps[0].results[0]
    assert result.task == current_task
    assert result.error == "worker_task_mismatch"


def test_max_rounds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_rounds must be at least 1"):
        run_paper_agent(
            master=lambda state: MasterAction(kind="NEEDS_HUMAN"),
            worker=lambda task: WorkerResult(task=task),
            max_rounds=0,
        )


def test_duplicate_questions_run_until_the_round_budget() -> None:
    worker_calls = 0

    def master(state: AgentState) -> MasterAction:
        if not state.findings:
            return MasterAction(
                kind="READ_PAPER",
                tasks=(EvidenceTask("Does it work?"),),
            )
        return MasterAction(
            kind="READ_PAPER",
            tasks=(EvidenceTask("  does   IT work?  "),),
        )

    def worker(task: EvidenceTask) -> WorkerResult:
        nonlocal worker_calls
        worker_calls += 1
        return WorkerResult(task=task, finding="first answer")

    trace = run_paper_agent(master=master, worker=worker, max_rounds=2)

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.stop_reason == "round_budget_exhausted"
    assert worker_calls == 2
    assert trace.steps[-1].action.tasks[0].question == "  does   IT work?  "


def test_duplicate_questions_in_one_round_are_all_executed() -> None:
    worker_calls = 0

    def worker(task: EvidenceTask) -> WorkerResult:
        nonlocal worker_calls
        worker_calls += 1
        return WorkerResult(task=task)

    trace = run_paper_agent(
        master=lambda state: MasterAction(
            kind="READ_PAPER",
            tasks=(EvidenceTask("same"), EvidenceTask(" SAME ")),
        ),
        worker=worker,
        max_rounds=1,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.stop_reason == "round_budget_exhausted"
    assert worker_calls == 2


def test_selected_context_uses_only_prior_round_findings_and_keeps_invalid_ids_nonfatal() -> None:
    contexts: list[tuple[tuple[object, ...], tuple[str, ...]]] = []

    class ContextWorker:
        worker_context_mode = "selected-context"

        def set_research_context(
            self, records: tuple[object, ...], diagnostics: tuple[str, ...]
        ) -> None:
            contexts.append((records, diagnostics))

        def __call__(self, task: EvidenceTask) -> WorkerResult:
            return WorkerResult(
                task=task,
                finding=f"finding for {task.question}",
                evidence=f"evidence for {task.question}",
                caveat=f"caveat for {task.question}",
                evidence_type="text",
                evidence_locator="p. 1",
            )

    def master(state: AgentState) -> MasterAction:
        if not state.steps:
            return MasterAction("READ_PAPER", (EvidenceTask("first"),))
        if len(state.steps) == 1:
            return MasterAction(
                "READ_PAPER",
                (
                    EvidenceTask(
                        "cross-check",
                        related_finding_ids=("r1-t1-f1", "unknown", "r1-t1-f1"),
                        decision_relevance="This relation could change the assessment.",
                    ),
                ),
            )
        return MasterAction("NEEDS_HUMAN")

    trace = run_paper_agent(master=master, worker=ContextWorker(), max_rounds=3)

    assert contexts[0] == ((), ())
    selected, diagnostics = contexts[1]
    assert len(selected) == 1
    assert selected[0].finding_id == "r1-t1-f1"
    assert selected[0].question == "first"
    assert selected[0].finding == "finding for first"
    assert selected[0].evidence == "evidence for first"
    assert selected[0].decision_relevance == "This relation could change the assessment."
    assert diagnostics == ("unknown_related_finding_id", "duplicate_related_finding_id")
    assert trace.steps[1].results[0].research_context == selected


def test_context_task_fields_default_to_legacy_empty_values() -> None:
    task = EvidenceTask("inspect a bounded result")

    assert task.related_finding_ids == ()
    assert task.decision_relevance == ""


def test_selected_context_keeps_at_most_three_prior_findings() -> None:
    seen_contexts: list[tuple[object, ...]] = []

    class ContextWorker:
        worker_context_mode = "selected-context"

        def set_research_context(
            self, records: tuple[object, ...], diagnostics: tuple[str, ...]
        ) -> None:
            del diagnostics
            seen_contexts.append(records)

        def __call__(self, task: EvidenceTask) -> WorkerResult:
            return WorkerResult(task=task, finding=task.question)

    def master(state: AgentState) -> MasterAction:
        if not state.steps:
            return MasterAction(
                "READ_PAPER",
                tuple(EvidenceTask(f"first-{index}") for index in range(1, 5)),
            )
        if len(state.steps) == 1:
            return MasterAction(
                "READ_PAPER",
                (
                    EvidenceTask(
                        "bounded cross-check",
                        related_finding_ids=tuple(f"r1-t{index}-f1" for index in range(1, 5)),
                        decision_relevance="The combined evidence could change the assessment.",
                    ),
                ),
            )
        return MasterAction("NEEDS_HUMAN")

    run_paper_agent(master=master, worker=ContextWorker(), max_rounds=3)

    assert seen_contexts[0] == ()
    assert seen_contexts[1] == ()
    assert seen_contexts[2] == ()
    assert seen_contexts[3] == ()
    assert [item.finding_id for item in seen_contexts[4]] == [
        "r1-t1-f1",
        "r1-t2-f1",
        "r1-t3-f1",
    ]


def test_selected_context_diagnostics_remain_in_trace_when_worker_fails() -> None:
    class FailingContextWorker:
        worker_context_mode = "selected-context"

        def set_research_context(
            self, records: tuple[object, ...], diagnostics: tuple[str, ...]
        ) -> None:
            del records, diagnostics

        def __call__(self, task: EvidenceTask) -> WorkerResult:
            if task.question == "first":
                return WorkerResult(task=task, finding="first finding")
            raise RuntimeError("worker unavailable")

    def master(state: AgentState) -> MasterAction:
        if not state.steps:
            return MasterAction("READ_PAPER", (EvidenceTask("first"),))
        if len(state.steps) == 1:
            return MasterAction(
                "READ_PAPER",
                (
                    EvidenceTask(
                        "second",
                        related_finding_ids=("unknown",),
                        decision_relevance="A check is useful.",
                    ),
                ),
            )
        return MasterAction("NEEDS_HUMAN")

    trace = run_paper_agent(master=master, worker=FailingContextWorker(), max_rounds=3)

    failed_result = trace.steps[1].results[0]
    assert failed_result.error == "RuntimeError: worker unavailable"
    assert failed_result.context_diagnostics == ("unknown_related_finding_id",)


def test_completed_findings_receive_one_stable_id_per_structured_finding() -> None:
    task = EvidenceTask("inspect local evidence")
    result = WorkerResult(
        task=task,
        structured_findings=(
            WorkerFinding("first", (FindingEvidence("evidence one", "text", "p. 1"),), "caveat one"),
            WorkerFinding("second", (FindingEvidence("evidence two", "table", "Table 2, p. 2"),), "caveat two"),
            WorkerFinding("third", (FindingEvidence("evidence three", "figure", "Figure 3, p. 3"),), "caveat three"),
        ),
    )
    state = AgentState(
        findings=(result,),
        steps=(TraceStep(1, MasterAction("READ_PAPER", (task,)), (result,)),),
    )

    records = completed_finding_records(state)

    assert [record.finding_id for record in records] == ["r1-t1-f1", "r1-t1-f2", "r1-t1-f3"]
    assert [(record.finding, record.evidence, record.caveat, record.evidence_locator) for record in records] == [
        ("first", "evidence one", "caveat one", "p. 1"),
        ("second", "evidence two", "caveat two", "Table 2, p. 2"),
        ("third", "evidence three", "caveat three", "Figure 3, p. 3"),
    ]


def test_selecting_one_structured_finding_does_not_leak_its_siblings() -> None:
    task = EvidenceTask("initial")
    result = WorkerResult(
        task=task,
        structured_findings=(
            WorkerFinding("first", (FindingEvidence("one", "text", "p. 1"),), "c1"),
            WorkerFinding("second", (FindingEvidence("two", "table", "Table 2, p. 2"),), "c2"),
            WorkerFinding("third", (FindingEvidence("three", "figure", "Figure 3, p. 3"),), "c3"),
        ),
    )
    state = AgentState(
        steps=(TraceStep(1, MasterAction("READ_PAPER", (task,)), (result,)),),
    )

    context, diagnostics = _selected_research_context(
        state,
        EvidenceTask(
            "cross-check",
            related_finding_ids=("r1-t1-f2",),
            decision_relevance="Check the relationship.",
        ),
    )

    assert diagnostics == ()
    assert [(item.finding, item.evidence, item.caveat, item.evidence_locator) for item in context] == [
        ("second", "two", "c2", "Table 2, p. 2")
    ]


def test_context_limit_applies_to_independent_findings_not_worker_results() -> None:
    task = EvidenceTask("initial")
    result = WorkerResult(
        task=task,
        structured_findings=tuple(
            WorkerFinding(str(index), (FindingEvidence(str(index), "text", f"p. {index}"),), "")
            for index in range(1, 5)
        ),
    )
    state = AgentState(steps=(TraceStep(1, MasterAction("READ_PAPER", (task,)), (result,)),))

    context, diagnostics = _selected_research_context(
        state,
        EvidenceTask(
            "cross-check",
            related_finding_ids=tuple(f"r1-t1-f{index}" for index in range(1, 5)),
            decision_relevance="bounded",
        ),
    )

    assert [item.finding_id for item in context] == ["r1-t1-f1", "r1-t1-f2", "r1-t1-f3"]
    assert diagnostics == ("related_finding_id_limit",)


def test_legacy_worker_result_remains_a_single_f1_context_record() -> None:
    task = EvidenceTask("legacy")
    result = WorkerResult(
        task=task,
        finding="legacy finding",
        evidence="legacy evidence",
        caveat="legacy caveat",
        evidence_type="text",
        evidence_locator="p. 1",
    )
    state = AgentState(steps=(TraceStep(1, MasterAction("READ_PAPER", (task,)), (result,)),))

    assert [(item.finding_id, item.finding, item.evidence, item.caveat) for item in completed_finding_records(state)] == [
        ("r1-t1-f1", "legacy finding", "legacy evidence", "legacy caveat")
    ]
def test_parallel_worker_batch_is_isolated_ordered_and_waits_for_all_tasks() -> None:
    entered = threading.Event()
    release = threading.Event()
    observations: list[tuple[str, tuple[str, ...]]] = []

    class Worker:
        def fork_for_task(self, *, round_number, research_context, context_diagnostics):
            return self

        def __call__(self, task):
            observations.append((task.question, tuple(item.finding_id for item in getattr(self, "context", ()))))
            entered.set()
            release.wait(timeout=1)
            return WorkerResult(task=task, finding=task.question, evidence="e", caveat="")

    tasks = (EvidenceTask("first"), EvidenceTask("second"))
    actions = iter((MasterAction("READ_PAPER", tasks), MasterAction("DECIDE", assessment="done")))
    worker = Worker()
    trace_box: list[object] = []
    thread = threading.Thread(
        target=lambda: trace_box.append(run_paper_agent(
            master=lambda _state: next(actions), worker=worker, max_rounds=2, worker_parallelism=2
        ))
    )
    thread.start()
    assert entered.wait(timeout=1)
    release.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    trace = trace_box[0]
    assert [result.task.question for result in trace.steps[0].results] == ["first", "second"]
    assert [question for question, _context in observations] == ["first", "second"]
