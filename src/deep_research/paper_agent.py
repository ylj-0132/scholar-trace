"""Deterministic control loop for adaptive paper-evidence collection."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import Callable, Literal, Protocol

ActionKind = Literal["READ_PAPER", "DECIDE", "NEEDS_HUMAN"]
@dataclass(frozen=True)
class EvidenceTask:
    question: str
    source_scope: str = "paper"
    related_finding_ids: tuple[str, ...] = ()
    decision_relevance: str = ""


@dataclass(frozen=True)
class FindingEvidence:
    content: str
    evidence_type: str
    locator: str


@dataclass(frozen=True)
class WorkerFinding:
    finding: str
    evidence: tuple[FindingEvidence, ...]
    caveat: str


@dataclass(frozen=True)
class ResearchContext:
    finding_id: str
    question: str
    finding: str
    evidence: str
    caveat: str
    evidence_type: str
    evidence_locator: str
    decision_relevance: str
    evidence_items: tuple[FindingEvidence, ...] = ()


@dataclass(frozen=True)
class WorkerResult:
    task: EvidenceTask
    finding: str = ""
    evidence: str = ""
    caveat: str = ""
    error: str | None = None
    pages_read: tuple[int, ...] = ()
    location_rationale: str = ""
    evidence_type: str = ""
    evidence_locator: str = ""
    noncanonical_output_shape: str | None = None
    suggested_questions: tuple[str, ...] = ()
    research_context: tuple[ResearchContext, ...] = ()
    context_diagnostics: tuple[str, ...] = ()
    structured_findings: tuple[WorkerFinding, ...] = ()


@dataclass(frozen=True)
class MasterAction:
    kind: ActionKind | str
    tasks: tuple[EvidenceTask, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    assessment: str | None = None
    rationale: str = ""
    checklist_coverage: dict[str, str] | None = None
    conclusion_at_risk: str = ""
    missing_evidence: str = ""
    expected_judgment_delta: str = ""
    stop_reason_code: str | None = None
    pre_decide_reflection_focus: str | None = None


@dataclass(frozen=True)
class TraceStep:
    round_number: int
    action: MasterAction
    results: tuple[WorkerResult, ...] = ()


@dataclass(frozen=True)
class ReflectionReport:
    trigger: str
    reflected_finding_ids: tuple[str, ...]
    reflection_memo: str = ""
    error: str | None = None


@dataclass(frozen=True)
class DeferredDecision:
    round_number: int
    trigger: str
    finding_ids: tuple[str, ...]
    action: MasterAction


@dataclass(frozen=True)
class AgentState:
    findings: tuple[WorkerResult, ...] = ()
    provisional_assessment: str | None = None
    unresolved_questions: tuple[str, ...] = ()
    reflection_reports: tuple[ReflectionReport, ...] = ()
    steps: tuple[TraceStep, ...] = ()
    remaining_rounds: int = 0


@dataclass(frozen=True)
class ImageInputRef:
    image_index: int
    source_document: str
    page_number: int
    dpi: int


@dataclass(frozen=True)
class ModelCallRecord:
    call_id: str
    role: Literal["master", "locator", "evidence", "single_pass", "synthesis", "reflection"]
    round_number: int | None
    task_question: str | None
    model: str | None
    temperature: float | None
    system_prompt: str
    user_prompt: str
    task_number: int | None = None
    task_count: int | None = None
    image_inputs: tuple[ImageInputRef, ...] = ()
    raw_response: str | None = None
    parsed_response: dict[str, object] | None = None
    json_repaired: bool = False
    validation_error: str | None = None
    error: str | None = None
    latency_seconds: float | None = None
    attempt_count: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class SinglePassFinding:
    finding: str
    evidence: str
    caveat: str
    evidence_type: str
    evidence_locator: str


@dataclass(frozen=True)
class JudgmentEvidence:
    content: str
    evidence_type: str
    locator: str


@dataclass(frozen=True)
class FinalJudgmentFinding:
    finding: str
    evidence: tuple[JudgmentEvidence, ...]
    caveat: str


@dataclass(frozen=True)
class FinalJudgment:
    assessment: str
    key_findings: tuple[FinalJudgmentFinding, ...]
    unresolved_questions: tuple[str, ...] = ()
    checklist_coverage: dict[str, str] | None = None


@dataclass(frozen=True)
class SinglePassTrace:
    source_document: str
    assessment: str | None
    findings: tuple[SinglePassFinding, ...]
    error: str | None
    model_calls: tuple[ModelCallRecord, ...]
    final_judgment: FinalJudgment | None = None
    images_sent_to_model: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class AgentTrace:
    steps: tuple[TraceStep, ...]
    outcome: Literal["DECIDE", "NEEDS_HUMAN"]
    assessment: str | None
    stop_reason: str
    source_document: str | None = None
    model_calls: tuple[ModelCallRecord, ...] = ()
    final_judgment: FinalJudgment | None = None
    reflection_reports: tuple[ReflectionReport, ...] = ()
    deferred_decisions: tuple[DeferredDecision, ...] = ()
    stop_reason_code: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


class Master(Protocol):
    def __call__(self, state: AgentState) -> MasterAction: ...


class EvidenceWorker(Protocol):
    def __call__(self, task: EvidenceTask) -> WorkerResult: ...


class Reflector(Protocol):
    def __call__(
        self,
        state: AgentState,
        *,
        trigger: str,
        finding_ids: tuple[str, ...],
        proposed_decision: MasterAction | None = None,
    ) -> ReflectionReport: ...


def run_paper_agent(
    *,
    master: Master,
    worker: EvidenceWorker,
    max_rounds: int,
    worker_parallelism: int = 1,
    reflector: Reflector | None = None,
    max_reflections: int = 0,
    on_progress: Callable[[str, dict[str, object]], None] | None = None,
) -> AgentTrace:
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    if not 1 <= worker_parallelism <= 4:
        raise ValueError("worker_parallelism must be between 1 and 4")
    if not 0 <= max_reflections <= 2:
        raise ValueError("max_reflections must be between 0 and 2")
    if max_reflections and reflector is None:
        raise ValueError("reflector is required when max_reflections is positive")

    state = AgentState(remaining_rounds=max_rounds)
    reflection_reports: list[ReflectionReport] = []
    deferred_decisions: list[DeferredDecision] = []

    def finish(trace: AgentTrace) -> AgentTrace:
        return replace(
            trace,
            reflection_reports=tuple(reflection_reports),
            deferred_decisions=tuple(deferred_decisions),
        )

    for round_number in range(1, max_rounds + 1):
        while True:
            try:
                action = master(state)
            except Exception as exc:
                return finish(_invalid_action_trace(
                    state.steps,
                    f"master_error@round_{round_number}:{type(exc).__name__}: {exc}",
                ))

            if not isinstance(action, MasterAction):
                return finish(_invalid_action_trace(
                    state.steps,
                    f"invalid_master_output@round_{round_number}:{type(action).__name__}",
                ))

            if action.kind == "DECIDE":
                if (
                    reflector is not None
                    and max_reflections
                    and not state.steps
                    and not reflection_reports
                ):
                    step = TraceStep(round_number, action)
                    return finish(_invalid_action_trace(
                        state.steps + (step,),
                        _record_master_error(
                            master, "decide_before_required_method_reflection"
                        ),
                    ))
                finding_ids = _successful_finding_ids(state)
                last_reflected = (
                    reflection_reports[-1].reflected_finding_ids
                    if reflection_reports else ()
                )
                new_finding_ids = tuple(
                    finding_id for finding_id in finding_ids
                    if finding_id not in last_reflected
                )
                if (
                    reflector is not None
                    and len(reflection_reports) < max_reflections
                    and new_finding_ids
                    and action.pre_decide_reflection_focus is not None
                ):
                    deferred_decisions.append(
                        DeferredDecision(round_number, "pre_decide", new_finding_ids, action)
                    )
                    report = _run_reflection(
                        reflector, state, "pre_decide", new_finding_ids,
                        round_number, on_progress, action,
                    )
                    reflection_reports.append(report)
                    if report.error is not None:
                        return finish(_invalid_action_trace(
                            state.steps,
                            f"reflection_error@round_{round_number}:{report.error}",
                        ))
                    state = replace(state, reflection_reports=tuple(reflection_reports))
                    continue
                step = TraceStep(round_number, action)
                reason = _reason(action.rationale, "master_decided")
                if not _is_empty_string(action.assessment):
                    return finish(AgentTrace(
                        steps=state.steps + (step,),
                        outcome="DECIDE",
                        assessment=action.assessment,
                        stop_reason=reason,
                        stop_reason_code=action.stop_reason_code,
                    ))
                return finish(_invalid_action_trace(
                    state.steps + (step,),
                    _record_master_error(master, "decide_without_assessment"),
                ))

            if action.kind == "NEEDS_HUMAN":
                step = TraceStep(round_number, action)
                return finish(AgentTrace(
                    steps=state.steps + (step,),
                    outcome="NEEDS_HUMAN",
                    assessment=action.assessment,
                    stop_reason=_reason(action.rationale, "master_requested_human"),
                ))

            if action.kind != "READ_PAPER":
                return finish(_invalid_action_trace(
                    state.steps + (TraceStep(round_number, action),),
                    _record_master_error(master, f"unsupported_action:{action.kind}"),
                ))

            action_error = _read_action_error(action)
            if action_error is not None:
                return finish(_invalid_action_trace(
                    state.steps + (TraceStep(round_number, action),),
                    _record_master_error(master, action_error),
                ))

            _emit_progress(
                on_progress,
                "round_started",
                {
                    "round_number": round_number,
                    "action_kind": action.kind,
                    "task_count": len(action.tasks),
                    "discovery_tasks": sum(not task.related_finding_ids for task in action.tasks),
                    "cross_check_tasks": sum(bool(task.related_finding_ids) for task in action.tasks),
                },
            )
            results = _run_worker_batch(
                worker, action.tasks, state, round_number, worker_parallelism
            )
            step = TraceStep(round_number, action, results)
            state = AgentState(
                findings=state.findings + results,
                provisional_assessment=(
                    action.assessment
                    if isinstance(action.assessment, str) and action.assessment.strip()
                    else state.provisional_assessment
                ),
                unresolved_questions=action.unresolved_questions,
                reflection_reports=tuple(reflection_reports),
                steps=state.steps + (step,),
                remaining_rounds=max_rounds - round_number,
            )
            _emit_progress(
                on_progress,
                "round_completed",
                {
                    "round_number": round_number,
                    "action_kind": action.kind,
                    "task_count": len(action.tasks),
                    "discovery_tasks": sum(not task.related_finding_ids for task in action.tasks),
                    "cross_check_tasks": sum(bool(task.related_finding_ids) for task in action.tasks),
                    "successful_workers": sum(result.error is None for result in results),
                    "failed_workers": sum(result.error is not None for result in results),
                    "finding_count": sum(len(result.structured_findings) for result in results),
                    "evidence_item_count": sum(
                        len(finding.evidence)
                        for result in results
                        for finding in result.structured_findings
                    ),
                    "suggested_question_count": sum(len(result.suggested_questions) for result in results),
                    "checklist_coverage": action.checklist_coverage or {},
                    "provisional_assessment_present": state.provisional_assessment is not None,
                    "unresolved_question_count": len(state.unresolved_questions),
                },
            )
            if reflector is not None and not reflection_reports and max_reflections:
                report = _run_reflection(
                    reflector, state, "post_method_model", _successful_finding_ids(state),
                    round_number, on_progress, None,
                )
                reflection_reports.append(report)
                if report.error is not None:
                    return finish(_invalid_action_trace(
                        state.steps,
                        f"reflection_error@round_{round_number}:{report.error}",
                    ))
                state = replace(state, reflection_reports=tuple(reflection_reports))
            break

    return finish(_invalid_action_trace(state.steps, "round_budget_exhausted"))


def _successful_finding_ids(state: AgentState) -> tuple[str, ...]:
    return tuple(record.finding_id for record in completed_finding_records(state))


def _run_reflection(
    reflector: Reflector,
    state: AgentState,
    trigger: str,
    finding_ids: tuple[str, ...],
    round_number: int,
    on_progress: Callable[[str, dict[str, object]], None] | None,
    proposed_decision: MasterAction | None,
) -> ReflectionReport:
    _emit_progress(on_progress, "reflection_started", {
        "round_number": round_number,
        "trigger": trigger,
        "finding_count": len(finding_ids),
    })
    try:
        report = reflector(
            state,
            trigger=trigger,
            finding_ids=finding_ids,
            proposed_decision=proposed_decision,
        )
        if not isinstance(report, ReflectionReport):
            raise ValueError(f"invalid_reflection_output:{type(report).__name__}")
        if report.trigger != trigger:
            raise ValueError("invalid_reflection_output:trigger")
    except Exception as exc:
        report = ReflectionReport(
            trigger=trigger,
            reflected_finding_ids=finding_ids,
            error=f"{type(exc).__name__}: {exc}",
        )
    _emit_progress(on_progress, "reflection_completed", {
        "round_number": round_number,
        "trigger": trigger,
        "finding_count": len(finding_ids),
    })
    return report


def _run_worker(
    worker: EvidenceWorker, task: EvidenceTask, state: AgentState
) -> WorkerResult:
    context, diagnostics = _selected_research_context(state, task)
    return _run_worker_with_context(worker, task, context, diagnostics)


def _run_worker_with_context(
    worker: EvidenceWorker,
    task: EvidenceTask,
    context: tuple[ResearchContext, ...],
    diagnostics: tuple[str, ...],
) -> WorkerResult:
    _set_worker_research_context(worker, context, diagnostics)
    try:
        result = worker(task)
    except Exception as exc:
        result = WorkerResult(task=task, error=f"{type(exc).__name__}: {exc}")
    if not isinstance(result, WorkerResult):
        result = WorkerResult(task=task, error=f"invalid_worker_output:{type(result).__name__}")
    elif result.task != task:
        result = WorkerResult(task=task, error="worker_task_mismatch")
    return replace(
        result,
        research_context=context,
        context_diagnostics=diagnostics,
    )


def _run_worker_batch(
    worker: EvidenceWorker,
    tasks: tuple[EvidenceTask, ...],
    state: AgentState,
    round_number: int,
    worker_parallelism: int,
) -> tuple[WorkerResult, ...]:
    prepared = [
        (task, *_selected_research_context(state, task), task_number, len(tasks))
        for task_number, task in enumerate(tasks, start=1)
    ]
    if worker_parallelism == 1:
        _set_worker_round(worker, round_number)
        return tuple(
            _run_worker_with_context(
                _set_worker_task_position(worker, task_number, task_count),
                task,
                context,
                diagnostics,
            )
            for task, context, diagnostics, task_number, task_count in prepared
        )
    fork = getattr(worker, "fork_for_task", None)
    if not callable(fork):
        raise ValueError("parallel worker must provide fork_for_task")
    workers = [
        fork(
            round_number=round_number,
            research_context=context,
            context_diagnostics=diagnostics,
        )
        for _task, context, diagnostics, _task_number, _task_count in prepared
    ]
    for task_worker, (_task, _context, _diagnostics, task_number, task_count) in zip(
        workers, prepared
    ):
        _set_worker_task_position(task_worker, task_number, task_count)
    with ThreadPoolExecutor(max_workers=worker_parallelism) as executor:
        futures = [
            executor.submit(_run_worker_with_context, task_worker, task, context, diagnostics)
            for task_worker, (task, context, diagnostics, _task_number, _task_count) in zip(
                workers, prepared
            )
        ]
        return tuple(future.result() for future in futures)


def _emit_progress(
    callback: Callable[[str, dict[str, object]], None] | None,
    event: str,
    payload: dict[str, object],
) -> None:
    if callback is not None:
        callback(event, payload)


def _read_action_error(action: MasterAction) -> str | None:
    if not isinstance(action.tasks, tuple):
        return f"read_paper_invalid_tasks_type:{type(action.tasks).__name__}"
    if not action.tasks:
        return "read_paper_without_tasks"
    for index, task in enumerate(action.tasks, start=1):
        if not isinstance(task, EvidenceTask):
            return f"read_paper_invalid_task_{index}:invalid_task_type:{type(task).__name__}"
        if not isinstance(task.question, str) or not task.question.strip():
            return f"read_paper_invalid_task_{index}:empty_question"
        if task.source_scope != "paper":
            return (
                f"read_paper_invalid_task_{index}:"
                f"unsupported_source_scope:{task.source_scope}"
            )
        if not isinstance(task.related_finding_ids, tuple) or any(
            not isinstance(finding_id, str) for finding_id in task.related_finding_ids
        ):
            return f"read_paper_invalid_task_{index}:related_finding_ids"
        if not isinstance(task.decision_relevance, str):
            return f"read_paper_invalid_task_{index}:decision_relevance"
    return None


def _set_worker_round(worker: EvidenceWorker, round_number: int) -> None:
    setter = getattr(worker, "set_round_number", None)
    if callable(setter):
        setter(round_number)


def _set_worker_task_position(
    worker: EvidenceWorker, task_number: int, task_count: int
) -> EvidenceWorker:
    setter = getattr(worker, "set_task_position", None)
    if callable(setter):
        setter(task_number, task_count)
    return worker


def _set_worker_research_context(
    worker: EvidenceWorker,
    context: tuple[ResearchContext, ...],
    diagnostics: tuple[str, ...],
) -> None:
    if getattr(worker, "worker_context_mode", "no-context") != "selected-context":
        return
    setter = getattr(worker, "set_research_context", None)
    if callable(setter):
        setter(context, diagnostics)


def _selected_research_context(
    state: AgentState, task: EvidenceTask
) -> tuple[tuple[ResearchContext, ...], tuple[str, ...]]:
    if not task.related_finding_ids:
        return (), ()
    prior = {
        record.finding_id: record
        for record in completed_finding_records(state)
    }
    selected: list[ResearchContext] = []
    diagnostics: list[str] = []
    seen: set[str] = set()
    for finding_id in task.related_finding_ids:
        if finding_id in seen:
            diagnostics.append("duplicate_related_finding_id")
            continue
        seen.add(finding_id)
        record = prior.get(finding_id)
        if record is None:
            diagnostics.append("unknown_related_finding_id")
            continue
        if len(selected) == 3:
            diagnostics.append("related_finding_id_limit")
            continue
        selected.append(replace(record, decision_relevance=task.decision_relevance))
    return tuple(selected), tuple(diagnostics)


def completed_finding_records(state: AgentState) -> tuple[ResearchContext, ...]:
    records: list[ResearchContext] = []
    for step in state.steps:
        for task_number, result in enumerate(step.results, start=1):
            if result.error is not None:
                continue
            findings = result.structured_findings or _legacy_worker_findings(result)
            for finding_number, finding in enumerate(findings, start=1):
                if not finding.finding.strip() or not finding.evidence:
                    continue
                primary = finding.evidence[0]
                records.append(
                    ResearchContext(
                        finding_id=(
                            f"r{step.round_number}-t{task_number}-f{finding_number}"
                        ),
                        question=result.task.question,
                        finding=finding.finding,
                        evidence=primary.content,
                        caveat=finding.caveat,
                        evidence_type=primary.evidence_type,
                        evidence_locator=primary.locator,
                        decision_relevance="",
                        evidence_items=finding.evidence,
                    )
                )
    return tuple(records)


def _legacy_worker_findings(result: WorkerResult) -> tuple[WorkerFinding, ...]:
    if not result.finding.strip():
        return ()
    return (
        WorkerFinding(
            finding=result.finding,
            evidence=(
                FindingEvidence(
                    content=result.evidence,
                    evidence_type=result.evidence_type,
                    locator=result.evidence_locator,
                ),
            ),
            caveat=result.caveat,
        ),
    )


def _record_master_error(master: Master, error: str) -> str:
    recorder = getattr(master, "record_validation_error", None)
    if callable(recorder):
        recorder(error)
    return error


def _invalid_action_trace(steps: tuple[TraceStep, ...], reason: str) -> AgentTrace:
    return AgentTrace(
        steps=steps,
        outcome="NEEDS_HUMAN",
        assessment=None,
        stop_reason=reason,
    )


def _is_empty_string(value: object) -> bool:
    return not isinstance(value, str) or not value.strip()


def _reason(value: object, default: str) -> str:
    if not isinstance(value, str):
        return default
    normalized = value.strip()
    return normalized or default
