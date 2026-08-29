"""Paper-only LLM adapters built on top of the deterministic Phase 1 loop."""

from __future__ import annotations

import json
import inspect
import re
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from deep_research.paper_agent import (
    AgentState,
    AgentTrace,
    ResearchContext,
    EvidenceTask,
    FindingEvidence,
    FinalJudgment,
    FinalJudgmentFinding,
    ImageInputRef,
    JudgmentEvidence,
    MasterAction,
    ModelCallRecord,
    ReflectionReport,
    SinglePassFinding,
    SinglePassTrace,
    WorkerResult,
    WorkerFinding,
    completed_finding_records,
    run_paper_agent,
)
from deep_research.paper_reading import (
    PaperPage,
    extract_main_text,
    extract_pdf_pages,
    normalize_text,
    render_page_text,
    render_pdf_pages,
    section_kind_for_heading,
)

ALLOWED_EVIDENCE_TYPES = {"text", "table", "figure"}
DECISION_CHECKLIST = (
    "core_contribution", "main_evidence", "ablation_or_counterevidence",
    "direct_vs_proxy_metric", "evaluation_validity", "cost_stability_robustness",
    "claim_boundary", "auxiliary_model_reliability",
    "generative_transformation_fidelity", "matched_resource_efficiency",
    "claim_result_consistency",
)
MECHANISM_AUDIT_PRINCIPLES = (
    "Distinguish full-system performance from causal evidence for one mechanism.",
    "Check whether ablations also change structures, access paths, retrieval scope, or budget.",
    "Consider alternative factors that could explain the reported result.",
    "Check whether claimed data-structure advantages are used by the runtime access path.",
    "Check tables, figures, body text, and appendices for negative results, metric disagreement, or configuration inconsistency.",
    "Check whether preprocessing, query decomposition, reranking, candidate-set size, context size, or another coupled stage could explain part of a reported gain.",
    "For temporal or continual-operation claims, check chronological evaluation, boundary effects, sustained updates, storage growth, and update cost when the paper supplies relevant evidence.",
    "When a validator or judge is an auxiliary model, check its reliability, calibration or human validation, model dependence, and self-confirmation risk when the paper supplies relevant evidence.",
    "Do not infer one auxiliary role's model identity or configuration from another role; when role ownership or model assignment is not explicitly reported, state that it is unreported.",
    "When summarization, rewrite, compression, or another generative transformation changes information consumed downstream, check fidelity, omitted conditions, unsupported additions, and output stability.",
    "Treat a smaller-model result as an outcome comparison, not an efficiency result, unless total model calls, tokens, latency, or cost are matched or reported.",
    "Reconcile headline prose and claimed deltas with table values, metric definitions, labels, and simple arithmetic; preserve material inconsistencies even when the overall conclusion is unchanged.",
)
DEFAULT_IMAGE_DPI = 144
CAPTION_RE = re.compile(r"^(?:figure|fig\.?|table)\s+\d+\b", re.IGNORECASE)

LOCATOR_SYSTEM_PROMPT = (
    "You locate bounded paper pages for one evidence question. "
    "Return only JSON and never answer from pages outside the supplied index."
)
EVIDENCE_SYSTEM_PROMPT = (
    "You are a local evidence analyst. Answer one bounded question and analyze "
    "all distinct issues supported by the supplied paper pages and images. "
    "Return only JSON and distinguish text, table, and figure evidence. "
    "For text evidence, copy a short verbatim quote from one selected page; "
    "do not paraphrase, summarize, or join non-contiguous passages. You may "
    "suggest up to two bounded follow-up questions, but do not schedule them."
)
MASTER_SYSTEM_PROMPT = (
    "You are the Master for a comprehensive paper research-contribution assessment. "
    "Return only JSON. Use only paper text supplied in the current call and accumulated "
    "Worker findings as evidence; "
    "do not claim unseen pages were inspected or use personal-value labels."
)
SYNTHESIS_SYSTEM_PROMPT = (
    "Produce one comprehensive, evidence-grounded paper judgment as JSON. "
    "Preserve distinct supported findings, caveats, and honest unresolved questions."
)
REFLECTION_SYSTEM_PROMPT = (
    "You are a paper-research reflector. Return only JSON. Analyze only the "
    "supplied context and accumulated Worker state; reflection is a candidate "
    "research lead, not new paper evidence or a final assessment."
)
SINGLE_PASS_SYSTEM_PROMPT = (
    "You are a single-pass full-paper judge. Evaluate the paper's research "
    "contribution in its field, distinguish author claims from supported "
    "conclusions, assess evidence strength and boundaries, and identify key "
    "caveats. Use only the supplied paper content. Do not evaluate personal "
    "usefulness, reading priority, or whether a reader should adopt the method; "
    "do evaluate the method's computational, storage, latency, and deployment "
    "costs when supported. Return only JSON and do "
    "not claim to have checked evidence that was not supplied. For text "
    "evidence, copy a short verbatim quote from the supplied paper text; do "
    "not paraphrase, summarize, or join non-contiguous passages. Audit whether "
    "reported full-system results isolate the claimed mechanism."
)

MASTER_OUTPUT_CONTRACT = {
    "READ_PAPER": {
        "kind": "READ_PAPER",
        "tasks": [
            {
                "question": "one bounded evidence question",
                "source_scope": "paper",
                "related_finding_ids": [],
                "decision_relevance": "this task's unverified purpose and conclusion at risk, competing explanation or boundary, and expected judgment delta",
            }
        ],
        "unresolved_questions": ["remaining question"],
        "assessment": (
            "current provisional assessment and how existing evidence supports or bounds it"
        ),
        "rationale": "why this evidence is needed next",
        "conclusion_at_risk": "the current paper-value or mechanism conclusion that could change",
        "missing_evidence": "the bounded paper evidence not already checked",
        "expected_judgment_delta": "how plausible answers could add a distinct finding, caveat, experimental boundary, counterexample, or reporting inconsistency; this need not change the overall positive or negative assessment",
        "checklist_coverage": {key: "covered|unresolved|not_applicable" for key in DECISION_CHECKLIST},
    },
    "DECIDE": {
        "kind": "DECIDE",
        "tasks": [],
        "unresolved_questions": [
            "material question that remains open but does not prevent a supported judgment"
        ],
        "assessment": (
            "evidence-grounded assessment of contribution, evidence strength, and boundaries"
        ),
        "rationale": (
            "why the judgment is complete enough to finalize and how remaining questions are bounded"
        ),
        "checklist_coverage": {key: "covered|unresolved|not_applicable" for key in DECISION_CHECKLIST},
        "stop_reason_code": "evidence_sufficient|paper_saturated|remaining_gaps_unreported|remaining_gaps_external",
        "pre_decide_reflection_focus": None,
    },
    "NEEDS_HUMAN": {
        "kind": "NEEDS_HUMAN",
        "tasks": [],
        "unresolved_questions": ["what remains unresolved"],
        "assessment": None,
        "rationale": "why the Agent cannot safely decide",
        "checklist_coverage": {key: "covered|unresolved|not_applicable" for key in DECISION_CHECKLIST},
    },
}

STOP_REASON_CODES = {
    "evidence_sufficient",
    "paper_saturated",
    "remaining_gaps_unreported",
    "remaining_gaps_external",
}
MASTER_CONTEXT_MODES = {"full-history", "incremental-no-raw-evidence"}
PAPER_CONTEXT_MODES = {
    "legacy",
    "master-main-text-history-only",
    "master-overview-history-only",
}

COMPREHENSIVE_JUDGMENT_INSTRUCTIONS = (
    "Produce a comprehensive paper judgment, not a short report or abstract. Use as many key_findings as needed to preserve the complete supported assessment.",
    "The single key_finding in required_json_shape is an illustrative item, not a length limit.",
    "Preserve material caveats and unresolved questions in the final judgment.",
    "Do not silently omit any supported negative result, metric disagreement, configuration inconsistency, ablation confound, alternative explanation, missing control, or material cost, scaling, or stability limitation.",
    "Retain a concrete issue even when it does not change the overall assessment.",
    "Merge overlapping points only when the final finding or caveat can explicitly retain every distinct issue and its evidence.",
    "The fact that a checklist item is covered does not permit dropping the concrete findings that support or qualify it.",
    "Distinguish whether the full system works from whether a specific mechanism is isolated by the experiments.",
    "Before calling a comparison aligned or attributing an outcome to a method, jointly compare the reported task-model, proposer/evolver, verifier/judge, data-split, and resource-budget assignments across methods; treat any unresolved consequential difference as a confound.",
)


def _final_judgment_shape() -> dict[str, object]:
    return {
        "assessment": "overall judgment of contribution, evidence strength, and boundaries",
        "key_findings": [
            {
                "finding": "one important judgment",
                "evidence": [
                    {
                        "content": "short verbatim quote from the paper text or visual result",
                        "evidence_type": "text|table|figure",
                        "locator": "paper.pdf, p. 6 or Table 3, p. 6",
                    }
                ],
                "caveat": "limitation or uncertainty, possibly empty",
            }
        ],
        "unresolved_questions": [],
        "checklist_coverage": {
            key: "covered|unresolved|not_applicable" for key in DECISION_CHECKLIST
        },
    }


class ModelCallRecorder:
    """Small per-run recorder for auditable model calls."""

    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        on_event: Callable[[str, ModelCallRecord], None] | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._on_event = on_event
        self._lock = threading.Lock()
        self._records: list[ModelCallRecord] = []
        self._next_call_number = 1

    @property
    def records(self) -> list[ModelCallRecord]:
        with self._lock:
            return list(self._records)

    def start_call(
        self,
        *,
        role: str,
        round_number: int | None,
        task_question: str | None,
        system_prompt: str,
        user_prompt: str,
        image_inputs: Sequence[ImageInputRef],
        task_number: int | None = None,
        task_count: int | None = None,
        model: str | None = None,
    ) -> str:
        if role not in {"master", "locator", "evidence", "single_pass", "synthesis", "reflection"}:
            raise ValueError(f"unsupported model call role: {role}")
        with self._lock:
            call_id = f"call-{self._next_call_number:03d}"
            self._next_call_number += 1
            record = ModelCallRecord(
                call_id=call_id,
                role=role,  # type: ignore[arg-type]
                round_number=round_number,
                task_question=task_question,
                model=model if model is not None else self.model,
                temperature=self.temperature,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                task_number=task_number,
                task_count=task_count,
                image_inputs=tuple(image_inputs),
            )
            self._records.append(record)
        self._emit("started", record)
        return call_id

    def complete_call(self, call_id: str, **updates: object) -> None:
        self._replace_call(call_id, **updates)

    def error_call(self, call_id: str, **updates: object) -> None:
        with self._lock:
            index = self._index_locked(call_id)
            current = self._records[index]
            if current.error is not None:
                return
            record = replace(current, **updates)
            self._records[index] = record
        self._emit("error", record)

    def validation_error(self, call_id: str, error: str) -> None:
        self._replace_call(call_id, validation_error=error)

    def _replace_call(self, call_id: str, **updates: object) -> None:
        with self._lock:
            index = self._index_locked(call_id)
            record = replace(self._records[index], **updates)
            self._records[index] = record
        self._emit("updated", record)

    def _index_locked(self, call_id: str) -> int:
        for index, record in enumerate(self._records):
            if record.call_id == call_id:
                return index
        raise KeyError(call_id)

    def _emit(self, event: str, record: ModelCallRecord) -> None:
        if self._on_event is not None:
            self._on_event(event, record)


def build_compact_page_index(
    pages: Sequence[PaperPage],
    *,
    preview_chars: int = 400,
) -> str:
    """Build a deterministic, bounded JSON page index without full page text."""

    if preview_chars < 1:
        raise ValueError("preview_chars must be at least 1")
    entries: list[dict[str, object]] = []
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if line.strip()]
        headings = [line for line in lines if section_kind_for_heading(line)]
        captions = [line for line in lines if CAPTION_RE.match(line)]
        entries.append(
            {
                "page_number": page.page_number,
                "preview": normalize_text(page.text)[:preview_chars],
                "headings": headings,
                "captions": captions,
                "low_text": page.low_text,
                "error": page.error,
            }
        )
    return json.dumps(entries, ensure_ascii=False, indent=2)


def validate_page_selection(
    payload: object,
    *,
    page_count: int,
) -> tuple[int, ...]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    raw_ranges = payload.get("page_ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError("page_ranges must be a non-empty list")
    selected: set[int] = set()
    for raw_range in raw_ranges:
        if not isinstance(raw_range, dict):
            raise ValueError("page range must be an object")
        start = raw_range.get("start")
        end = raw_range.get("end")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError("page range start and end must be integers")
        if start > end:
            raise ValueError("page range start must not exceed end")
        if start < 1 or end > page_count:
            raise ValueError(f"page range {start}-{end} is outside 1..{page_count}")
        selected.update(range(start, end + 1))

    if not selected:
        raise ValueError("page selection is empty")
    return tuple(sorted(selected))


class PaperEvidenceWorker:
    """One bounded question implemented as exactly one locator and one extractor call."""

    def __init__(
        self,
        *,
        pdf_path: Path,
        pages: Sequence[PaperPage],
        page_index: str,
        llm: Any,
        locator_llm: Any | None = None,
        evidence_llm: Any | None = None,
        render_pages: Callable[[Path, Sequence[int]], dict[int, str]] = render_pdf_pages,
        recorder: ModelCallRecorder | None = None,
        worker_context_mode: str = "no-context",
        worker_role_mode: str = "legacy",
        _image_cache: dict[int, str] | None = None,
        _image_cache_lock: threading.Lock | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        self.pages = tuple(pages)
        self.page_index = page_index
        self.llm = llm
        self.locator_llm = llm if locator_llm is None else locator_llm
        self.evidence_llm = llm if evidence_llm is None else evidence_llm
        self.render_pages = render_pages
        self.recorder = recorder
        self.worker_context_mode = worker_context_mode
        self.worker_role_mode = worker_role_mode
        self._last_call_id: str | None = None
        self.round_number: int | None = None
        self._page_text = {page.page_number: page.text for page in self.pages}
        self._research_context: tuple[ResearchContext, ...] = ()
        self._context_diagnostics: tuple[str, ...] = ()
        self._task_number: int | None = None
        self._task_count: int | None = None
        self._image_cache = {} if _image_cache is None else _image_cache
        self._image_cache_lock = (
            threading.Lock() if _image_cache_lock is None else _image_cache_lock
        )

    def set_round_number(self, round_number: int) -> None:
        self.round_number = round_number

    def set_research_context(
        self,
        context: tuple[ResearchContext, ...],
        diagnostics: tuple[str, ...],
    ) -> None:
        self._research_context = context
        self._context_diagnostics = diagnostics

    def fork_for_task(
        self,
        *,
        round_number: int,
        research_context: tuple[ResearchContext, ...],
        context_diagnostics: tuple[str, ...],
    ) -> PaperEvidenceWorker:
        clone = PaperEvidenceWorker(
            pdf_path=self.pdf_path,
            pages=self.pages,
            page_index=self.page_index,
            llm=self.llm,
            locator_llm=self.locator_llm,
            evidence_llm=self.evidence_llm,
            render_pages=self.render_pages,
            recorder=self.recorder,
            worker_context_mode=self.worker_context_mode,
            worker_role_mode=self.worker_role_mode,
            _image_cache=self._image_cache,
            _image_cache_lock=self._image_cache_lock,
        )
        clone.set_round_number(round_number)
        clone.set_research_context(research_context, context_diagnostics)
        return clone

    def set_task_position(self, task_number: int | None, task_count: int | None) -> None:
        self._task_number = task_number
        self._task_count = task_count

    def _page_images(self, selected_pages: Sequence[int]) -> dict[int, str]:
        with self._image_cache_lock:
            missing = [page for page in selected_pages if page not in self._image_cache]
            if missing:
                self._image_cache.update(self.render_pages(self.pdf_path, missing))
            return {page: self._image_cache[page] for page in selected_pages if page in self._image_cache}

    def __call__(self, task: EvidenceTask) -> WorkerResult:
        if task.source_scope != "paper":
            return WorkerResult(task=task, error="unsupported_source_scope:paper_only")

        role_instructions = _worker_role_instructions(task, self.worker_role_mode)
        locator_prompt = _build_locator_prompt(
            question=task.question,
            decision_context=task.decision_relevance,
            page_index=self.page_index,
            page_count=len(self.pages),
            research_context=self._research_context,
        )
        try:
            locator_payload, locator_call_id = _invoke_model(
                llm=self.locator_llm,
                prompt=locator_prompt,
                system_prompt=LOCATOR_SYSTEM_PROMPT,
                recorder=self.recorder,
                role="locator",
                round_number=self.round_number,
                task_question=task.question,
                image_inputs=(),
                task_number=self._task_number,
                task_count=self._task_count,
            )
        except Exception as exc:
            return WorkerResult(
                task=task,
                error=f"locator_call_error:{type(exc).__name__}: {exc}",
            )

        if not isinstance(locator_payload, dict):
            _mark_validation(self.recorder, locator_call_id, "payload must be an object")
            return WorkerResult(task=task, error="invalid_locator:payload must be an object")
        rationale = locator_payload.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            _mark_validation(self.recorder, locator_call_id, "rationale")
            return WorkerResult(task=task, error="invalid_locator:rationale")
        rationale = rationale.strip()
        try:
            selected_pages = validate_page_selection(
                locator_payload,
                page_count=len(self.pages),
            )
        except ValueError as exc:
            _mark_validation(self.recorder, locator_call_id, str(exc))
            return _worker_error(
                task,
                (),
                rationale,
                f"invalid_locator:{exc}",
            )

        try:
            page_images = self._page_images(selected_pages)
        except Exception as exc:
            return _worker_error(
                task,
                selected_pages,
                rationale,
                f"page_render_error:{type(exc).__name__}: {exc}",
            )
        missing_images = [page for page in selected_pages if page not in page_images]
        if missing_images:
            return _worker_error(
                task,
                selected_pages,
                rationale,
                f"page_render_error:missing pages {missing_images}",
            )

        selected_text = {
            page: self._page_text.get(page, "") for page in selected_pages
        }
        if any(not text for text in selected_text.values()):
            return _worker_error(
                task,
                selected_pages,
                rationale,
                "page_text_error:missing selected page text",
            )
        evidence_prompt = _build_evidence_prompt(
            paper_name=self.pdf_path.name,
            question=task.question,
            decision_context=task.decision_relevance,
            location_rationale=rationale,
            selected_text=selected_text,
            image_inputs=_image_input_refs(
                source_document=self.pdf_path.name,
                page_numbers=selected_pages,
            ),
            research_context=self._research_context,
            role_instructions=role_instructions,
        )
        try:
            evidence_payload, evidence_call_id = _invoke_model(
                llm=self.evidence_llm,
                prompt=evidence_prompt,
                system_prompt=EVIDENCE_SYSTEM_PROMPT,
                recorder=self.recorder,
                role="evidence",
                round_number=self.round_number,
                task_question=task.question,
                image_inputs=_image_input_refs(
                    source_document=self.pdf_path.name,
                    page_numbers=selected_pages,
                ),
                image_urls=[page_images[page] for page in selected_pages],
                task_number=self._task_number,
                task_count=self._task_count,
            )
        except Exception as exc:
            return _worker_error(
                task,
                selected_pages,
                rationale,
                f"evidence_call_error:{type(exc).__name__}: {exc}",
            )

        result = _parse_evidence_result(
            task=task,
            payload=evidence_payload,
            selected_pages=selected_pages,
            location_rationale=rationale,
            selected_text=selected_text,
        )
        if result.error:
            _mark_validation(self.recorder, evidence_call_id, result.error)
        return result


class PaperAgentMaster:
    """Minimal LLM Master adapter that maps JSON into the Phase 1 action contract."""

    def __init__(
        self,
        *,
        llm: Any,
        paper_name: str,
        page_index: str,
        overview_text: str,
        overview_images: Sequence[str],
        overview_page_numbers: Sequence[int] | None = None,
        recorder: ModelCallRecorder | None = None,
        worker_role_mode: str = "legacy",
        reflection_enabled: bool = False,
        investigation_target: str | None = None,
        master_context_mode: str = "full-history",
        paper_context_mode: str = "legacy",
        main_paper_text: str = "",
    ) -> None:
        if master_context_mode not in MASTER_CONTEXT_MODES:
            raise ValueError("unsupported master_context_mode")
        if paper_context_mode not in PAPER_CONTEXT_MODES:
            raise ValueError("unsupported paper_context_mode")
        self.llm = llm
        self.paper_name = paper_name
        self.page_index = page_index
        self.overview_text = overview_text
        self.overview_images = tuple(overview_images)
        self.overview_page_numbers = tuple(
            overview_page_numbers
            if overview_page_numbers is not None
            else range(1, len(self.overview_images) + 1)
        )
        self.recorder = recorder
        self.worker_role_mode = worker_role_mode
        self.reflection_enabled = reflection_enabled
        self.investigation_target = investigation_target
        self.master_context_mode = master_context_mode
        self.paper_context_mode = paper_context_mode
        self.main_paper_text = main_paper_text
        self.overview_image_inputs = _image_input_refs(
            source_document=self.paper_name,
            page_numbers=self.overview_page_numbers,
        )

    def __call__(self, state: AgentState) -> MasterAction:
        research_phase = (
            "understand_design"
            if self.reflection_enabled and not state.reflection_reports
            else "investigate_implications"
            if self.reflection_enabled
            else "general_evidence_research"
        )
        phase_instructions: list[str] = []
        if research_phase == "understand_design":
            phase_instructions = [
                "Treat design understanding as the current attention priority, not a hard reading stage: all relevant paper sections remain available when they help explain the claimed problem, core design choices, intended functions, evaluator or task target, and what the paper actually observes.",
                "Trace the information and representation flow, transformations, runtime access path, and any components that may provide overlapping functions.",
                "Reported outcomes, tables, and ablations may help explain the design, but headline results alone do not establish why a mechanism works or whether the evaluation isolates it.",
                "Do not make this phase a fixed page-order exercise; locate the smallest sufficient method, task-definition, or evaluator-design evidence wherever the paper places it.",
            ]
        elif research_phase == "investigate_implications":
            phase_instructions = [
                "Use the Reflection memo as candidate reasoning to investigate consequences, alternatives, and boundaries; carry the relevant reasoning faithfully into decision_relevance when assigning a bounded paper-internal question.",
                "Choose evidence that can support, reject, or discriminate between intended and competing explanations, while treating the memo itself as unverified rather than evidence.",
                "You may return to methods, experiments, appendices, or any other relevant section, and may still open an independent direction not suggested by the memo.",
                "If every proposed task uses selected context, account in the rationale for whether one independent no-context blind-spot task remains useful; do not add a task merely to satisfy a count.",
            ]
        prompt_payload: dict[str, object] = {
                "paper": self.paper_name,
                "research_phase": research_phase,
                "overview_pages": self.overview_text,
                "compact_page_index": json.loads(self.page_index),
                "state": _state_payload(state),
                "decision_checklist": DECISION_CHECKLIST,
                "mechanism_audit_principles": MECHANISM_AUDIT_PRINCIPLES,
                "output_contract": MASTER_OUTPUT_CONTRACT,
                "image_inputs": _image_input_payload(self.overview_image_inputs),
                "instructions": [
                    "Choose concrete paper-internal evidence questions necessary for a complete, evidence-supported paper judgment, not merely enough evidence to choose an overall positive or negative direction.",
                    "Each task must be one bounded evidence question about the paper and use source_scope=paper.",
                    "Ask one decisive question per task; never ask to transcribe all tables, benchmarks, or ablations.",
                    "A READ_PAPER action may contain multiple independent tasks; do not default to one task when several distinct, evidence-bearing questions can be investigated in the same round.",
                    "When main_paper_text is supplied, identify one or two load-bearing headline claims whose validity can be checked inside the paper. If a claim depends on arithmetic, metric labels, model or configuration matching, or a table or figure comparison, convert it into a bounded Worker verification task; do not treat seeing it in main_paper_text as verified downstream evidence.",
                    "Select only headline claims that materially support the paper's contribution, superiority, or efficiency narrative; do not audit every number.",
                    "Use prior findings and caveats to change the next question when needed.",
                    "Reflection reports are coherent candidate reasoning, not paper evidence. Adopt, rewrite, defer, or reject their analysis; when verification is needed, preserve the relevant purpose in decision_relevance and assign one coherent bounded EvidenceTask rather than mechanically splitting every thought into separate tasks.",
                    "Use empty related_finding_ids for independent Discovery. Select up to three prior IDs only when testing an explicit relationship, contradiction, or alternative explanation.",
                    "Treat Worker suggested questions as candidate questions, not evidence; adopt, rewrite, prioritize, or ignore them based on the current state.",
                    "Before choosing the next action, inspect relationships among prior findings, caveats, and unresolved questions.",
                    "Investigate evidence that can add, qualify, bound, contradict, or change the assessment; a question may be valuable even when it will not reverse the overall judgment.",
                    "When a relationship could change, qualify or bound the assessment without reversing it, use a bounded Cross-check task that tests it; do not let Cross-check automatically replace independent Discovery of unreviewed directions.",
                    "Do not repeat a near-duplicate search for an absent detail after a targeted confirmation unless new locator evidence points to a different source.",
                    "For every task, use that task's decision_relevance to preserve the unverified purpose, competing explanation, or judgment boundary that makes the question useful. For Cross-check, also name the relationship being tested. Do not present decision_relevance as paper evidence.",
                    "For a Cross-check task, optionally provide up to three related_finding_ids from state.findings; they are reports from prior Workers that the new Worker must independently test, not facts to accept automatically.",
                    "For every READ_PAPER action, provide a provisional assessment that states the current judgment and how the accumulated evidence supports, changes, or bounds it.",
                    "Before DECIDE, inspect Worker suggested questions and unreviewed paper directions for distinct evidence-bearing directions: investigate the strongest one or two in one bounded batch, explain why they add no important finding or caveat, or preserve them as unresolved.",
                    "Before READ_PAPER, state the conclusion at risk, the missing paper evidence not already checked, and the expected judgment delta. The delta may be a distinct supported finding, material caveat, experimental boundary, counterexample, or reporting inconsistency and need not change the overall positive or negative assessment.",
                    "READ_PAPER when the paper could resolve it, no prior targeted check established it as unreported, and the answer could add distinct evidence-bearing information to the complete assessment.",
                    "Every task in a multi-task READ_PAPER action must add a distinct, non-duplicate evidence direction. Use each task's question to identify its missing evidence and decision_relevance to identify its conclusion at risk and expected judgment delta; use the action-level fields to summarize all dispatched tasks.",
                    "A stable overall assessment is not sufficient reason to stop when one bounded paper-internal question could still add a load-bearing finding or caveat.",
                    "Before DECIDE, check whether one independent direction not originating in Reflection remains unreviewed; investigate it only if it can add distinct evidence-bearing information, and do not add a task merely to satisfy this check.",
                    "Budget remaining is not evidence value. Do not continue because rounds remain; DECIDE once the accumulated evidence supports the paper judgment and every remaining gap is non-blocking, already paper-unreported, answerable only by code or external evidence, or unlikely to change the assessment.",
                    "Once a targeted check establishes that a detail is paper-unreported, preserve it as unresolved and do not reopen it through another section, paraphrase, locator query, or Worker task unless new locator evidence identifies a specific unexamined source.",
                    "Set pre_decide_reflection_focus only for one named unresolved mechanism conflict or unverified direction visible in accumulated evidence that could add or correct a material caveat to the proposed assessment; otherwise set it to null. Do not request Reflection as a general completeness check.",
                    "For DECIDE, use evidence_sufficient when no material paper-internal gap remains; remaining_gaps_unreported when targeted checks found no report; remaining_gaps_external when only code or external evidence can resolve the gaps; otherwise use paper_saturated when further paper reading has no distinct evidence-bearing paper-internal direction.",
                    "Report checklist_coverage for the accumulated evidence. A covered item must correspond to a supported finding or bounded limitation. Checklist coverage records what has been examined; it is not by itself evidence that the paper judgment is complete.",
                    "DECIDE may retain honest unresolved_questions when they bound the conclusions but do not prevent a supported final judgment.",
                    "Only READ_PAPER, DECIDE, and NEEDS_HUMAN are available.",
                    "Do not assign personal usefulness or reading-priority labels.",
                    "Do not treat the compact page index as verified paper evidence.",
                    "Use only paper text supplied to this Master call and Worker findings for final assessment.",
                    "Do not claim evidence from pages not supplied to this Master call or Worker findings.",
                ] + phase_instructions + _role_mixed_master_instructions(self.worker_role_mode),
            }
        use_incremental_context = (
            bool(state.steps)
            and (
                self.master_context_mode == "incremental-no-raw-evidence"
                or self.paper_context_mode == "master-main-text-history-only"
            )
        )
        if self.paper_context_mode == "master-main-text-history-only" and not state.steps:
            del prompt_payload["overview_pages"]
            prompt_payload["main_paper_text"] = self.main_paper_text
        elif use_incremental_context:
            prompt_payload["state"] = _master_state_payload(state)
            del prompt_payload["overview_pages"]
            del prompt_payload["image_inputs"]
        prompt = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
        if self.investigation_target is not None:
            prompt_payload = json.loads(prompt)
            prompt_payload["investigation_target"] = self.investigation_target
            prompt = json.dumps(prompt_payload, ensure_ascii=False, indent=2)
        call_id: str | None = None
        payload, call_id = _invoke_model(
            llm=self.llm,
            prompt=prompt,
            system_prompt=MASTER_SYSTEM_PROMPT,
            recorder=self.recorder,
            role="master",
            round_number=len(state.steps) + 1,
            task_question=None,
            image_inputs=() if use_incremental_context else self.overview_image_inputs,
            image_urls=None if use_incremental_context else list(self.overview_images) or None,
        )
        self._last_call_id = call_id
        try:
            return parse_master_action(payload)
        except ValueError as exc:
            _mark_validation(self.recorder, call_id, str(exc))
            raise

    def record_validation_error(self, error: str) -> None:
        _mark_validation(self.recorder, self._last_call_id, error)


class PaperReflector:
    """One bounded reflection over accumulated state; it never reads new pages."""

    def __init__(
        self,
        *,
        llm: Any,
        paper_name: str,
        page_index: str,
        overview_text: str,
        recorder: ModelCallRecorder | None = None,
        paper_context_mode: str = "legacy",
    ) -> None:
        if paper_context_mode not in PAPER_CONTEXT_MODES:
            raise ValueError("unsupported paper_context_mode")
        self.llm = llm
        self.paper_name = paper_name
        self.page_index = page_index
        self.overview_text = overview_text
        self.recorder = recorder
        self.paper_context_mode = paper_context_mode
        self._last_call_id: str | None = None

    def __call__(
        self,
        state: AgentState,
        *,
        trigger: str,
        finding_ids: tuple[str, ...],
        proposed_decision: MasterAction | None = None,
    ) -> ReflectionReport:
        prompt_payload: dict[str, object] = {
                "paper": self.paper_name,
                "trigger": trigger,
                "findings_to_reflect": list(finding_ids),
                "overview_pages": self.overview_text,
                "compact_page_index": json.loads(self.page_index),
                "state": _state_payload(state),
                "mechanism_audit_principles": MECHANISM_AUDIT_PRINCIPLES,
                "required_json_shape": {
                    "reflection_memo": "coherent analysis for the Master"
                },
                "instructions": [
                    "Return one coherent prose memo for the Master, not a bullet list, checklist, hypothesis array, or final paper judgment.",
                    "Use the accumulated findings relevant to the requested reflection trigger rather than filling a fixed number of issue slots.",
                    "Explain the reasoning far enough for the Master to understand what may matter, why it may matter, and what paper-internal evidence would discriminate the plausible interpretations.",
                    "Do not claim to have read unprovided pages, provide new paper facts, make a final assessment, or automatically create a Worker task.",
                    "The memo is candidate reasoning, not paper evidence. Clearly distinguish observed Worker reports from inference, and leave any proposed explanation unverified until a Worker checks paper evidence.",
                ],
            }
        if self.paper_context_mode in {
            "master-main-text-history-only",
            "master-overview-history-only",
        }:
            del prompt_payload["overview_pages"]
            del prompt_payload["compact_page_index"]
        if trigger == "post_method_model":
            prompt_payload["instructions"].extend([
                "Reflect on the design before allowing reported experiment outcomes to settle its merits. In the memo, connect material design choices to their intended benefit and induced optimization target.",
                "Consider whether a cheapest winning strategy or shortcut, an excluded valid answer, a simpler functional substitute, component compensation, or a representation-consistency failure could also explain the observations; select the single most consequential issue rather than cataloguing every plausible criticism.",
                "Distinguish the capability the paper claims to evaluate or provide from what its final observable, score, or output actually observes. When accumulated findings name multiple editable components, stages, or mechanism classes, distinguish the declared search space, observed accepted artifacts, and independently credited mechanisms; do not treat those levels as equivalent.",
                "Describe only the single smallest useful paper-internal investigation or matched control that would discriminate the intended mechanism from the selected competing explanation; the Master will decide whether and how to turn it into a task.",
                "Do not summarize Worker findings or repeat ordinary missing metrics, costs, baseline gaps, or completeness issues unless they expose that selected hidden assumption or evidence-scope mismatch.",
                "Do not force a criticism. If the supplied evidence supports no material design consequence, say that plainly in the memo and explain the boundary of that conclusion.",
            ])
        if trigger == "pre_decide" and proposed_decision is not None:
            prompt_payload["proposed_decision"] = {
                "assessment": proposed_decision.assessment,
                "rationale": proposed_decision.rationale,
                "unresolved_questions": list(proposed_decision.unresolved_questions),
                "checklist_coverage": proposed_decision.checklist_coverage,
                "pre_decide_reflection_focus": proposed_decision.pre_decide_reflection_focus,
            }
            prompt_payload["instructions"].extend([
                "Analyze only proposed_decision.pre_decide_reflection_focus; do not perform a second global omission, completeness, or brainstorming review.",
                "Use only accumulated findings relevant to that named conflict; do not introduce unrelated mechanism, metric, cost, or completeness concerns.",
                "Using only accumulated evidence, determine whether that named conflict exposes a material correction or qualification to the proposed assessment and whether one bounded paper-internal check could resolve it.",
                "If it does not, say plainly that the proposed decision stands and do not recommend more reading. Do not reopen a paper-unreported, external, or merely interesting gap.",
                "If it does, explain the exact conclusion at risk, missing paper evidence, and expected judgment delta; do not create a Worker task or claim new paper evidence.",
            ])
        prompt = json.dumps(
            prompt_payload,
            ensure_ascii=False,
            indent=2,
        )
        call_id: str | None = None
        payload, call_id = _invoke_model(
            llm=self.llm,
            prompt=prompt,
            system_prompt=REFLECTION_SYSTEM_PROMPT,
            recorder=self.recorder,
            role="reflection",
            round_number=len(state.steps) + 1,
            task_question=None,
            image_inputs=(),
            image_urls=None,
        )
        self._last_call_id = call_id
        try:
            return parse_reflection_report(
                payload,
                trigger=trigger,
                reflected_finding_ids=finding_ids,
            )
        except ValueError as exc:
            _mark_validation(self.recorder, call_id, str(exc))
            raise


def parse_reflection_report(
    payload: object,
    *,
    trigger: str,
    reflected_finding_ids: tuple[str, ...],
) -> ReflectionReport:
    if not isinstance(payload, dict):
        raise ValueError("reflection payload must be an object")
    reflection_memo = payload.get("reflection_memo")
    if not isinstance(reflection_memo, str) or not reflection_memo.strip():
        raise ValueError("reflection_memo must be a non-empty string")
    return ReflectionReport(
        trigger=trigger,
        reflected_finding_ids=reflected_finding_ids,
        reflection_memo=reflection_memo.strip(),
    )


def parse_master_action(payload: object) -> MasterAction:
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    kind = payload.get("kind")
    if not isinstance(kind, str):
        raise ValueError("kind must be a string")

    raw_tasks = payload.get("tasks", [])
    if not isinstance(raw_tasks, list):
        raise ValueError("tasks must be a list")
    tasks: list[EvidenceTask] = []
    for index, raw_task in enumerate(raw_tasks, start=1):
        if not isinstance(raw_task, dict):
            raise ValueError(f"task {index} must be an object")
        question = raw_task.get("question")
        if not isinstance(question, str):
            raise ValueError(f"task {index} question must be a string")
        source_scope = raw_task.get("source_scope")
        if not isinstance(source_scope, str):
            raise ValueError(f"task {index} source_scope must be a string")
        raw_related = raw_task.get("related_finding_ids", [])
        if not isinstance(raw_related, list) or any(
            not isinstance(finding_id, str) for finding_id in raw_related
        ):
            raise ValueError(f"task {index} related_finding_ids must be a list of strings")
        decision_relevance = raw_task.get("decision_relevance", "")
        if not isinstance(decision_relevance, str):
            raise ValueError(f"task {index} decision_relevance must be a string")
        tasks.append(
            EvidenceTask(
                question=question,
                source_scope=source_scope,
                related_finding_ids=tuple(raw_related),
                decision_relevance=decision_relevance,
            )
        )

    raw_unresolved = payload.get("unresolved_questions", [])
    if not isinstance(raw_unresolved, list):
        raise ValueError("unresolved_questions must be a list")
    unresolved_items: list[str] = []
    for index, item in enumerate(raw_unresolved):
        if not isinstance(item, str):
            raise ValueError(f"unresolved_questions[{index}] must be a string")
        unresolved_items.append(item)

    assessment = payload.get("assessment")
    if assessment is not None and not isinstance(assessment, str):
        raise ValueError("assessment must be a string or null")
    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        raise ValueError("rationale must be a string")
    marginal_value = {
        field: payload.get(field, "")
        for field in (
            "conclusion_at_risk",
            "missing_evidence",
            "expected_judgment_delta",
        )
    }
    if kind == "READ_PAPER":
        for field, value in marginal_value.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(field)
    focus = payload.get("pre_decide_reflection_focus")
    if focus is not None and (not isinstance(focus, str) or not focus.strip()):
        raise ValueError("pre_decide_reflection_focus")
    stop_reason_code = payload.get("stop_reason_code")
    if kind == "DECIDE" and stop_reason_code not in STOP_REASON_CODES:
        raise ValueError("stop_reason_code")
    return MasterAction(
        kind=kind,
        tasks=tuple(tasks),
        unresolved_questions=tuple(unresolved_items),
        assessment=assessment,
        rationale=rationale,
        checklist_coverage=_parse_checklist_coverage(
            payload.get("checklist_coverage", {}),
            "invalid_master_output:checklist_coverage",
            fill_defaults=False,
        ),
        **{
            field: value.strip() if isinstance(value, str) else ""
            for field, value in marginal_value.items()
        },
        stop_reason_code=stop_reason_code if isinstance(stop_reason_code, str) else None,
        pre_decide_reflection_focus=focus.strip() if isinstance(focus, str) else None,
    )


def run_local_paper_agent(
    *,
    pdf_path: Path,
    llm: Any,
    role_llms: Mapping[str, Any] | None = None,
    max_rounds: int,
    worker_context_mode: str = "no-context",
    worker_role_mode: str = "legacy",
    master_context_mode: str = "full-history",
    paper_context_mode: str = "legacy",
    worker_parallelism: int = 1,
    max_reflections: int = 0,
    investigation_target: str | None = None,
    on_progress: Callable[[str, dict[str, object]], None] | None = None,
    on_model_event: Callable[[str, ModelCallRecord], None] | None = None,
) -> AgentTrace:
    if paper_context_mode not in PAPER_CONTEXT_MODES:
        raise ValueError("unsupported paper_context_mode")
    path = Path(pdf_path)
    pages = extract_pdf_pages(path)
    if not pages:
        raise ValueError("paper has no extractable pages")
    page_index = build_compact_page_index(pages)
    overview_pages = pages[:2]
    overview_numbers = tuple(page.page_number for page in overview_pages)
    overview_images = render_pdf_pages(path, overview_numbers)
    main_paper_text = (
        extract_main_text(pages)
        if paper_context_mode == "master-main-text-history-only"
        else ""
    )
    recorder = ModelCallRecorder(
        model=getattr(llm, "model", None),
        temperature=getattr(llm, "temperature", None),
        on_event=on_model_event,
    )
    role_llms = {} if role_llms is None else role_llms
    master_llm = role_llms.get("master", llm)
    reflection_llm = role_llms.get("reflection", llm)
    locator_llm = role_llms.get("locator", llm)
    evidence_llm = role_llms.get("evidence", llm)
    synthesis_llm = role_llms.get("synthesis", llm)
    master = PaperAgentMaster(
        llm=master_llm,
        paper_name=path.name,
        page_index=page_index,
        overview_text=render_page_text(overview_pages),
        overview_images=[overview_images[page] for page in overview_numbers],
        overview_page_numbers=overview_numbers,
        recorder=recorder,
        worker_role_mode=worker_role_mode,
        reflection_enabled=bool(max_reflections),
        investigation_target=investigation_target,
        master_context_mode=master_context_mode,
        paper_context_mode=paper_context_mode,
        main_paper_text=main_paper_text,
    )
    worker = PaperEvidenceWorker(
        pdf_path=path,
        pages=pages,
        page_index=page_index,
        llm=llm,
        locator_llm=locator_llm,
        evidence_llm=evidence_llm,
        render_pages=render_pdf_pages,
        recorder=recorder,
        worker_context_mode=worker_context_mode,
        worker_role_mode=worker_role_mode,
    )
    reflector = (
        PaperReflector(
            llm=reflection_llm,
            paper_name=path.name,
            page_index=page_index,
            overview_text=render_page_text(overview_pages),
            recorder=recorder,
            paper_context_mode=paper_context_mode,
        )
        if max_reflections
        else None
    )
    trace = run_paper_agent(
        master=master,
        worker=worker,
        max_rounds=max_rounds,
        worker_parallelism=worker_parallelism,
        reflector=reflector,
        max_reflections=max_reflections,
        on_progress=on_progress,
    )
    final_judgment = None
    reflection_failed = any(report.error is not None for report in trace.reflection_reports)
    try:
        if reflection_failed:
            raise RuntimeError("required reflection failed")
        payload, call_id = _invoke_model(
            llm=synthesis_llm,
            prompt=_build_synthesis_prompt(
                path.name,
                pages,
                page_index,
                trace,
                paper_context_mode=paper_context_mode,
            ),
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            recorder=recorder,
            role="synthesis",
            round_number=None,
            task_question=None,
            image_inputs=(),
            image_urls=None,
        )
        _, _, final_judgment = _parse_single_pass_result(payload, pages)
    except ValueError as exc:
        _mark_validation(recorder, locals().get("call_id"), str(exc))
    except Exception:
        pass
    return replace(
        trace,
        assessment=final_judgment.assessment if final_judgment is not None else None,
        source_document=path.name,
        model_calls=tuple(recorder.records),
        final_judgment=final_judgment,
    )


def _latest_provisional_assessment(trace: AgentTrace) -> str | None:
    for step in reversed(trace.steps):
        assessment = step.action.assessment
        if isinstance(assessment, str) and assessment.strip():
            return assessment
    return None


def _build_synthesis_prompt(
    paper_name: str,
    pages: Sequence[PaperPage],
    page_index: str,
    trace: AgentTrace,
    *,
    paper_context_mode: str = "legacy",
) -> str:
    latest_unresolved = trace.steps[-1].action.unresolved_questions if trace.steps else ()
    history = _state_payload(
        AgentState(
            findings=tuple(result for step in trace.steps for result in step.results),
            unresolved_questions=latest_unresolved,
            reflection_reports=trace.reflection_reports,
            steps=trace.steps,
            provisional_assessment=_latest_provisional_assessment(trace),
            remaining_rounds=0,
        )
    )
    return _build_synthesis_prompt_from_history(
        paper_name,
        pages,
        page_index,
        history,
        paper_context_mode=paper_context_mode,
    )


def _build_synthesis_prompt_from_history(
    paper_name: str,
    pages: Sequence[PaperPage],
    page_index: str,
    history: object,
    *,
    paper_context_mode: str = "legacy",
) -> str:
    if paper_context_mode not in PAPER_CONTEXT_MODES:
        raise ValueError("unsupported paper_context_mode")
    payload: dict[str, object] = {
        "paper": paper_name,
        "history": history,
        "decision_checklist": DECISION_CHECKLIST,
        "mechanism_audit_principles": MECHANISM_AUDIT_PRINCIPLES,
        "required_json_shape": _final_judgment_shape(),
        "instructions": (
            _history_only_synthesis_instructions()
            if paper_context_mode in {
                "master-main-text-history-only",
                "master-overview-history-only",
            }
            else _synthesis_instructions()
        ),
        "image_inputs": [],
    }
    if paper_context_mode == "legacy":
        payload["full_paper_pages"] = [
            {"page_number": page.page_number, "text": page.text}
            for page in pages
        ]
        payload["compact_page_index"] = json.loads(page_index)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _synthesis_instructions() -> list[str]:
    return [
        "Produce one FinalJudgment matching required_json_shape.",
        "Use only supplied paper text and structured history.",
    ] + list(COMPREHENSIVE_JUDGMENT_INSTRUCTIONS) + [
        "Perform an independent full-paper gap check after reading the structured history: look for supported issues in the supplied full text that Workers missed, and add them with paper evidence rather than merely summarizing history.",
        "Before finalizing, account for every distinct supported issue in structured history: retained, explicitly merged, corrected, or left unresolved; do not silently discard any issue.",
        "Every applicable checklist item marked covered must appear in the assessment or an explicit key finding with grounded evidence; if the supplied paper cannot support either a conclusion or a bounded limitation, mark it unresolved instead of inventing one.",
        "Checklist coverage records whether the audit was completed, not whether the paper passed it: a grounded favorable conclusion, confirmed limitation or inconsistency counts as covered; use unresolved only when the supplied evidence cannot bound the audit question.",
        "Use the full paper to check Worker claims and reconcile contradictions across findings, body text, tables, and appendices; correct overclaims rather than preserving them mechanically.",
        "Treat Worker suggested questions as candidates, not findings; include them only when verified or still materially unresolved.",
        "Treat reflection reports as candidate reasoning, not paper evidence. Use them only to preserve or articulate an unresolved verification gap; do not cite a reflection as evidence.",
        "No page images are supplied to synthesis. Do not claim independent visual verification of a table or figure; preserve a grounded Worker visual report with its locator, verify it from supplied text when possible, or leave the visual-only point unresolved.",
    ]


def _history_only_synthesis_instructions() -> list[str]:
    return [
        "Produce one FinalJudgment matching required_json_shape.",
        "Use only structured history and the paper evidence quoted in Worker findings.",
        "Do not perform an independent paper-reading pass; no full paper text or compact page index is supplied.",
        "Treat Master assessments and Reflection memos as candidate reasoning, not paper evidence.",
        "Account for every distinct supported issue in structured history: retain it, explicitly merge it, correct it from stronger quoted evidence, or leave it honestly unresolved.",
        "When findings conflict, reconcile them only from supplied evidence snippets and caveats; if those are insufficient, preserve the conflict as unresolved rather than inventing a correction.",
    ] + list(COMPREHENSIVE_JUDGMENT_INSTRUCTIONS) + [
        "Every applicable checklist item marked covered must appear in the assessment or an explicit key finding with grounded evidence; if the supplied paper cannot support either a conclusion or a bounded limitation, mark it unresolved instead of inventing one.",
        "Checklist coverage records whether the audit was completed, not whether the paper passed it: a grounded favorable conclusion, confirmed limitation or inconsistency counts as covered; use unresolved only when the supplied evidence cannot bound the audit question.",
        "Treat Worker suggested questions as candidates, not findings; include them only when verified or still materially unresolved.",
        "Treat reflection reports as candidate reasoning, not paper evidence. Use them only to preserve or articulate an unresolved verification gap; do not cite a reflection as evidence.",
        "No page images are supplied to synthesis. Do not claim independent visual verification of a table or figure; preserve a grounded Worker visual report with its locator, verify it from supplied text when possible, or leave the visual-only point unresolved.",
    ]


def run_single_pass_paper_judge(
    *,
    pdf_path: Path,
    llm: Any,
    text_only: bool = False,
) -> SinglePassTrace:
    path = Path(pdf_path)
    pages = extract_pdf_pages(path)
    if not pages:
        raise ValueError("paper has no extractable pages")
    page_numbers = tuple(page.page_number for page in pages)
    page_images = {} if text_only else render_pdf_pages(path, page_numbers)
    image_inputs = () if text_only else _image_input_refs(
        source_document=path.name,
        page_numbers=page_numbers,
    )
    prompt = _build_single_pass_prompt(
        paper_name=path.name,
        pages=pages,
        image_inputs=image_inputs,
        page_index=build_compact_page_index(pages),
    )
    recorder = ModelCallRecorder(
        model=getattr(llm, "model", None),
        temperature=getattr(llm, "temperature", None),
    )
    try:
        payload, call_id = _invoke_model(
            llm=llm,
            prompt=prompt,
            system_prompt=SINGLE_PASS_SYSTEM_PROMPT,
            recorder=recorder,
            role="single_pass",
            round_number=None,
            task_question=None,
            image_inputs=image_inputs,
            image_urls=None if text_only else [page_images[page] for page in page_numbers],
        )
    except Exception as exc:
        return SinglePassTrace(
            source_document=path.name,
            assessment=None,
            findings=(),
            error=f"single_pass_call_error:{type(exc).__name__}: {exc}",
            model_calls=tuple(recorder.records),
            images_sent_to_model=0 if text_only else len(page_numbers),
        )

    try:
        assessment, findings, final_judgment = _parse_single_pass_result(payload, pages)
    except ValueError as exc:
        _mark_validation(recorder, call_id, str(exc))
        return SinglePassTrace(
            source_document=path.name,
            assessment=None,
            findings=(),
            error=str(exc),
            model_calls=tuple(recorder.records),
            images_sent_to_model=0 if text_only else len(page_numbers),
        )
    return SinglePassTrace(
        source_document=path.name,
        assessment=assessment,
        findings=findings,
        error=None,
        model_calls=tuple(recorder.records),
        final_judgment=final_judgment,
        images_sent_to_model=0 if text_only else len(page_numbers),
    )


def _build_single_pass_prompt(
    *,
    paper_name: str,
    pages: Sequence[PaperPage],
    image_inputs: Sequence[ImageInputRef],
    page_index: str | None = None,
) -> str:
    return json.dumps(
        {
            "paper": paper_name,
            "full_paper_pages": [
                {"page_number": page.page_number, "text": page.text}
                for page in pages
            ],
            "compact_page_index": json.loads(page_index or build_compact_page_index(pages)),
            "image_inputs": _image_input_payload(image_inputs),
            "decision_checklist": DECISION_CHECKLIST,
            "mechanism_audit_principles": MECHANISM_AUDIT_PRINCIPLES,
            "required_json_shape": _final_judgment_shape(),
            "instructions": list(COMPREHENSIVE_JUDGMENT_INSTRUCTIONS) + [
                "Use the mechanism audit principles to distinguish system-level success from isolated mechanism evidence.",
                "Perform an independent full-paper gap check across the body, tables, figures, and appendices represented in the supplied content before finalizing.",
                "Reconcile related results and configurations rather than reporting each section in isolation.",
                "Do not evaluate personal usefulness or reading priority. Do evaluate the method's computational, storage, latency, and deployment costs when supported by the paper.",
                "When image_inputs is empty, this is text-only input: do not claim independent verification of visual-only content, and keep any material visual uncertainty in unresolved_questions.",
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


def _parse_single_pass_result(
    payload: object,
    pages: Sequence[PaperPage],
) -> tuple[str, tuple[SinglePassFinding, ...], FinalJudgment]:
    if not isinstance(payload, dict):
        raise ValueError("invalid_single_pass_output:payload")
    assessment = payload.get("assessment")
    if not isinstance(assessment, str) or not assessment.strip():
        raise ValueError("invalid_single_pass_output:assessment")
    raw_findings = payload.get("key_findings")
    if not isinstance(raw_findings, list) or not raw_findings:
        raise ValueError("invalid_single_pass_output:key_findings")

    findings: list[SinglePassFinding] = []
    final_evidence: list[tuple[JudgmentEvidence, ...]] = []
    for index, raw_finding in enumerate(raw_findings):
        prefix = f"key_findings[{index}]"
        if not isinstance(raw_finding, dict):
            raise ValueError(f"invalid_single_pass_output:{prefix}")
        values = {
            name: raw_finding.get(name)
            for name in ("finding", "evidence", "evidence_type", "evidence_locator")
        }
        evidence_items: tuple[JudgmentEvidence, ...] = ()
        if isinstance(values["evidence"], list):
            evidence_items = values["evidence"]
            if not evidence_items or not all(isinstance(item, dict) for item in evidence_items):
                raise ValueError(f"invalid_single_pass_output:{prefix}.evidence")
            evidence_items = tuple(
                JudgmentEvidence(
                    str(item.get("content", "")).strip(),
                    str(item.get("evidence_type", "")).strip().lower(),
                    str(item.get("locator", "")).strip(),
                )
                for item in evidence_items
            )
            if not all(item.content and item.evidence_type in ALLOWED_EVIDENCE_TYPES and item.locator for item in evidence_items):
                raise ValueError(f"invalid_single_pass_output:{prefix}.evidence")
            first = evidence_items[0]
            values["evidence"] = first.content
            values["evidence_type"] = first.evidence_type
            values["evidence_locator"] = first.locator
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid_single_pass_output:{prefix}.{name}")
        caveat = raw_finding.get("caveat")
        if not isinstance(caveat, str):
            raise ValueError(f"invalid_single_pass_output:{prefix}.caveat")
        evidence_type = values["evidence_type"].strip().lower()
        if evidence_type not in ALLOWED_EVIDENCE_TYPES:
            raise ValueError(f"invalid_single_pass_output:{prefix}.evidence_type")
        evidence = values["evidence"].strip()
        findings.append(
            SinglePassFinding(
                finding=values["finding"].strip(),
                evidence=evidence,
                caveat=caveat.strip(),
                evidence_type=evidence_type,
                evidence_locator=values["evidence_locator"].strip(),
            )
        )
        final_evidence.append(evidence_items or (JudgmentEvidence(evidence, evidence_type, values["evidence_locator"].strip()),))
    final_findings = tuple(
        FinalJudgmentFinding(
            finding=finding.finding,
            evidence=final_evidence[index],
            caveat=finding.caveat,
        )
        for index, finding in enumerate(findings)
    )
    raw_unresolved = payload.get("unresolved_questions", [])
    if not isinstance(raw_unresolved, list) or not all(isinstance(item, str) for item in raw_unresolved):
        raise ValueError("invalid_single_pass_output:unresolved_questions")
    coverage = _parse_checklist_coverage(
        payload.get("checklist_coverage", {}), "invalid_single_pass_output:checklist_coverage"
    )
    return assessment.strip(), tuple(findings), FinalJudgment(
        assessment=assessment.strip(),
        key_findings=final_findings,
        unresolved_questions=tuple(raw_unresolved),
        checklist_coverage=coverage,
    )


def _parse_checklist_coverage(
    value: object, error: str, *, fill_defaults: bool = True
) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        key not in DECISION_CHECKLIST
        or not isinstance(item, str)
        or item not in {"covered", "unresolved", "not_applicable"}
        for key, item in value.items()
    ):
        raise ValueError(error)
    return (
        {key: value.get(key, "unresolved") for key in DECISION_CHECKLIST}
        if fill_defaults
        else dict(value)
    )


def _build_locator_prompt(
    *,
    question: str,
    decision_context: str = "",
    page_index: str,
    page_count: int,
    research_context: Sequence[ResearchContext] = (),
) -> str:
    payload: dict[str, object] = {
            "question": question,
            "compact_page_index": json.loads(page_index),
            "available_pages": {"start": 1, "end": page_count},
            "image_inputs": [],
            "required_json_shape": {
                "page_ranges": [{"start": 1, "end": 2}],
                "rationale": "why these pages answer the question",
            },
    }
    instructions = [
        "The compact_page_index is an incomplete navigation aid, not paper evidence; absence from the preview is not evidence of absence from the paper.",
        "Return the smallest sufficient page set for the bounded question, normally one to four pages; select more only when a comparison genuinely spans them.",
        "When the question names a specific Table, Figure, Appendix, or numbered section, include the page where that label or heading itself appears; do not substitute nearby pages that merely discuss the same topic.",
        "If the named source begins or continues across a page boundary, include the immediately adjacent page when the normal one-to-four-page range allows it.",
        "Use headings, captions, and previews only to locate pages. Do not answer the evidence question or perform the Evidence Worker's analysis.",
    ]
    if decision_context.strip():
        payload["decision_context"] = decision_context.strip()
        instructions.append(
            "decision_context is the Master's unverified reason for asking, not paper evidence; use it only to locate evidence that could support, qualify, contradict, or distinguish the stated possibilities."
        )
    if research_context:
        payload["research_context"] = _research_context_payload(research_context)
        instructions.extend([
            "research_context contains findings reported by another Worker; do not accept them as established facts.",
            "Use it only to locate pages that can confirm, qualify, contradict, or independently test the stated relationship.",
        ])
    if instructions:
        payload["instructions"] = instructions
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _build_evidence_prompt(
    *,
    paper_name: str,
    question: str,
    decision_context: str = "",
    location_rationale: str,
    selected_text: dict[int, str],
    image_inputs: Sequence[ImageInputRef],
    research_context: Sequence[ResearchContext] = (),
    role_instructions: Sequence[str] = (),
) -> str:
    payload: dict[str, object] = {
            "paper": paper_name,
            "question": question,
            "location_rationale": location_rationale,
            "image_inputs": _image_input_payload(image_inputs),
            "selected_pages": [
                {"page_number": page, "text": text}
                for page, text in selected_text.items()
            ],
            "mechanism_audit_principles": MECHANISM_AUDIT_PRINCIPLES,
            "required_json_shape": {
                "findings": [
                    {
                        "finding": "direct answer",
                        "evidence": [
                            {
                                "content": "short verbatim quote or visual result",
                                "evidence_type": "text|table|figure",
                                "locator": f"{paper_name}, p. 6 or Table 3, p. 6",
                            }
                        ],
                        "caveat": "limitation or uncertainty",
                    }
                ],
                "suggested_questions": [
                    "one bounded follow-up question, if needed"
                ],
            },
            "instructions": [
                "When evidence_type is text, copy a short verbatim quote from one selected page.",
                "Allow only whitespace and deterministic PDF line-break normalization; do not paraphrase.",
                "Do not summarize or concatenate non-contiguous passages.",
                "For table or figure evidence, report the relevant row, column, metric, comparison, and configuration when available, provide a precise locator, and distinguish what is directly observed from your inference.",
                "Answer the assigned question first, then inspect every applicable local audit direction supported by the selected pages: whether a discrete boundary or threshold creates artifacts; one parameter changes multiple components; activation frequency dilutes an aggregate result; preprocessing, decomposition, or reranking provides an alternative explanation; metrics or reported configurations disagree; or a control, comparison, or relevant negative result is missing.",
                "Report each supported implication as a separate finding and report every distinct supported issue. Do not stop after the first additional issue, and do not merge issues that require different evidence or impose different boundaries on the paper's claims.",
                "An additional local issue is material when it adds, qualifies, bounds, contradicts, or otherwise contributes to the complete paper judgment; it need not reverse the overall assessment.",
                "If none is supported by the selected pages, return no additional local issue rather than inventing criticism; suggest a bounded follow-up question only when the paper can resolve it.",
                "Do not invent criticism or speculate beyond the supplied pages.",
                "After answering the assigned question, suggest zero to two bounded paper-internal follow-up questions only when selected pages expose a material contradiction, alternative explanation, missing control, access-path mismatch, or cross-result inconsistency.",
                "Do not schedule follow-up questions or request broad reading or transcription.",
            ],
    }
    if decision_context.strip():
        payload["decision_context"] = decision_context.strip()
        payload["instructions"].insert(
            0,
            "decision_context is the Master's unverified reason for asking, not paper evidence; independently answer the bounded question from selected pages and use the context only to test, qualify, contradict, or distinguish the stated possibilities.",
        )
    if research_context:
        payload["research_context"] = _research_context_payload(research_context)
        payload["instructions"].insert(
            0,
            "research_context contains evidence and interpretations reported by another Worker, not facts to accept automatically; independently verify, qualify, or contradict the relevant relationship using the selected pages, and support every new finding with those pages.",
        )
    payload["instructions"].extend(role_instructions)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _worker_role_instructions(task: EvidenceTask, mode: str) -> tuple[str, ...]:
    if mode != "discovery-cross-check":
        return ()
    if task.related_finding_ids:
        return (
            "Act as a cross-finding reviewer.",
            "Check whether the selected pages support, conflict with, qualify, or offer an alternative explanation for the research leads.",
            "Check whether multiple components, candidate sources, access paths, or budgets change together.",
            "When an inference depends on research_context plus current-page evidence, name the related finding IDs in the finding and separate current-page observations from the cross-finding inference.",
            "do not delete, overwrite, or rewrite historical findings; return only this task's new analysis.",
        )
    return (
        "Act as an independent local evidence analyst.",
        "Answer the assigned question, then inspect the selected pages for negative results, metric or configuration disagreement, alternative explanations, parameter coupling, boundary effects, or missing controls.",
        "Report distinct supported local findings even when they only qualify or bound the eventual assessment.",
    )


def _role_mixed_master_instructions(mode: str) -> list[str]:
    if mode != "discovery-cross-check":
        return []
    return [
        "First round normally uses Discovery because no prior finding is available.",
        "Later READ_PAPER actions may mix Discovery and Cross-check tasks when both are independently useful.",
        "Cross-check supplements Discovery; it does not replace unreviewed mechanisms, negative results, experimental dimensions, or paper directions.",
        "Use Cross-check when relationships among findings could change, qualify, or bound the assessment, with at most three related_finding_ids and decision_relevance.",
        "do not force one task of each kind or create tasks merely for count.",
        "Do not choose a single Worker role merely because it was used in the previous round; choose each task by the evidence gap it addresses.",
        "Do not stop Discovery merely because every checklist label is covered when a distinct unreviewed evidence-bearing direction remains; checklist coverage alone is neither a stop rule nor a reason to continue.",
        "Continue to avoid near-duplicate searches for an already targeted missing detail.",
    ]


def _research_context_payload(
    context: Sequence[ResearchContext],
) -> list[dict[str, str]]:
    payload: list[dict[str, object]] = []
    for item in context:
        record: dict[str, object] = {
            "finding_id": item.finding_id,
            "question": item.question,
            "finding": item.finding,
            "evidence": item.evidence,
            "caveat": item.caveat,
            "evidence_type": item.evidence_type,
            "evidence_locator": item.evidence_locator,
            "decision_relevance": item.decision_relevance,
        }
        if len(item.evidence_items) > 1:
            record["evidence_items"] = [
                {
                    "content": evidence.content,
                    "evidence_type": evidence.evidence_type,
                    "locator": evidence.locator,
                }
                for evidence in item.evidence_items
            ]
        payload.append(record)
    return payload


def _parse_evidence_result(
    *,
    task: EvidenceTask,
    payload: object,
    selected_pages: tuple[int, ...],
    location_rationale: str,
    selected_text: dict[int, str],
) -> WorkerResult:
    if not isinstance(payload, dict):
        return _worker_error(
            task,
            selected_pages,
            location_rationale,
            "invalid_evidence_output:payload",
        )
    suggested_questions = _parse_suggested_questions(payload)
    if suggested_questions is None:
        return _worker_error(
            task,
            selected_pages,
            location_rationale,
            "invalid_evidence_output:suggested_questions",
        )
    findings_list = payload.get("findings")
    if "findings" in payload:
        if not isinstance(findings_list, list) or not findings_list:
            return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:findings")
        parsed_findings: list[dict[str, object]] = []
        structured_findings: list[WorkerFinding] = []
        all_evidence: list[dict[str, str]] = []
        noncanonical: list[str] = []
        for item in findings_list:
            if not isinstance(item, dict):
                return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:findings")
            finding_value = item.get("finding")
            caveat_value = item.get("caveat", "")
            evidence_values = item.get("evidence")
            if isinstance(finding_value, (list, dict)):
                finding_value = json.dumps(finding_value, ensure_ascii=False, sort_keys=True)
                noncanonical.append("finding")
            if not isinstance(finding_value, str) or not finding_value.strip():
                return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:finding")
            if not isinstance(caveat_value, str):
                return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:caveat")
            if isinstance(evidence_values, dict):
                evidence_values = [evidence_values]
                noncanonical.append("evidence")
            if not isinstance(evidence_values, list) or not evidence_values:
                return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:evidence")
            normalized_evidence: list[dict[str, str]] = []
            for evidence_value in evidence_values:
                if not isinstance(evidence_value, dict):
                    return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:evidence")
                content = evidence_value.get("content")
                evidence_type = evidence_value.get("evidence_type")
                locator = evidence_value.get("locator")
                if not isinstance(content, str) or not content.strip():
                    return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:evidence")
                if not isinstance(evidence_type, str) or evidence_type.strip().lower() not in ALLOWED_EVIDENCE_TYPES:
                    return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:evidence_type")
                if not isinstance(locator, str) or not locator.strip():
                    return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:evidence_locator")
                normalized = {"content": content.strip(), "evidence_type": evidence_type.strip().lower(), "locator": locator.strip()}
                normalized_evidence.append(normalized)
                all_evidence.append(normalized)
            parsed_findings.append({"finding": finding_value.strip(), "caveat": caveat_value.strip(), "evidence": normalized_evidence})
            structured_findings.append(
                WorkerFinding(
                    finding=finding_value.strip(),
                    evidence=tuple(
                        FindingEvidence(
                            content=evidence["content"],
                            evidence_type=evidence["evidence_type"],
                            locator=evidence["locator"],
                        )
                        for evidence in normalized_evidence
                    ),
                    caveat=caveat_value.strip(),
                )
            )
        first_evidence = all_evidence[0]
        return WorkerResult(
            task=task,
            finding=json.dumps([item["finding"] for item in parsed_findings], ensure_ascii=False),
            evidence=json.dumps(all_evidence, ensure_ascii=False, sort_keys=True),
            caveat=json.dumps([item["caveat"] for item in parsed_findings], ensure_ascii=False),
            pages_read=selected_pages,
            location_rationale=location_rationale,
            evidence_type=first_evidence["evidence_type"],
            evidence_locator=first_evidence["locator"],
            noncanonical_output_shape=",".join(dict.fromkeys(noncanonical)) or None,
            suggested_questions=suggested_questions,
            structured_findings=tuple(structured_findings),
        )
    finding = payload.get("finding")
    evidence = payload.get("evidence")
    evidence_type = payload.get("evidence_type")
    locator = payload.get("evidence_locator")
    caveat = payload.get("caveat", "")
    noncanonical: list[str] = []
    if isinstance(finding, (list, dict)):
        finding = json.dumps(finding, ensure_ascii=False, sort_keys=True)
        noncanonical.append("finding")
    if isinstance(evidence, (list, dict)):
        evidence = json.dumps(evidence, ensure_ascii=False, sort_keys=True)
        noncanonical.append("evidence")
    if not isinstance(finding, str) or not finding.strip():
        return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:finding")
    if not isinstance(evidence, str) or not evidence.strip():
        return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:evidence")
    if not isinstance(evidence_type, str) or evidence_type.strip().lower() not in ALLOWED_EVIDENCE_TYPES:
        return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:evidence_type")
    if not isinstance(locator, str):
        return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:evidence_locator")
    normalized_type = evidence_type.strip().lower()
    normalized_locator = locator.strip()
    if not normalized_locator:
        return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:evidence_locator")
    if caveat is None:
        caveat = ""
    if not isinstance(caveat, str):
        return _worker_error(task, selected_pages, location_rationale, "invalid_evidence_output:caveat")
    return WorkerResult(
        task=task,
        finding=finding.strip(),
        evidence=evidence.strip(),
        caveat=caveat.strip(),
        pages_read=selected_pages,
        location_rationale=location_rationale,
        evidence_type=normalized_type,
        evidence_locator=normalized_locator,
        noncanonical_output_shape=",".join(noncanonical) or None,
        suggested_questions=suggested_questions,
        structured_findings=(
            WorkerFinding(
                finding=finding.strip(),
                evidence=(
                    FindingEvidence(
                        content=evidence.strip(),
                        evidence_type=normalized_type,
                        locator=normalized_locator,
                    ),
                ),
                caveat=caveat.strip(),
            ),
        ),
    )


def _parse_suggested_questions(payload: dict[str, object]) -> tuple[str, ...] | None:
    value = payload.get("suggested_questions", [])
    if (
        not isinstance(value, list)
        or len(value) > 2
        or any(not isinstance(question, str) or not question.strip() for question in value)
    ):
        return None
    return tuple(question.strip() for question in value)


def _worker_error(
    task: EvidenceTask,
    pages_read: tuple[int, ...],
    location_rationale: str,
    error: str,
) -> WorkerResult:
    return WorkerResult(
        task=task,
        error=error,
        pages_read=pages_read,
        location_rationale=location_rationale,
    )


def _image_input_refs(
    *,
    source_document: str,
    page_numbers: Sequence[int],
) -> tuple[ImageInputRef, ...]:
    return tuple(
        ImageInputRef(
            image_index=index,
            source_document=source_document,
            page_number=page_number,
            dpi=DEFAULT_IMAGE_DPI,
        )
        for index, page_number in enumerate(page_numbers, start=1)
    )


def _image_input_payload(
    image_inputs: Sequence[ImageInputRef],
) -> list[dict[str, object]]:
    return [
        {
            "image_index": ref.image_index,
            "source_document": ref.source_document,
            "page_number": ref.page_number,
            "dpi": ref.dpi,
        }
        for ref in image_inputs
    ]


def _invoke_model(
    *,
    llm: Any,
    prompt: str,
    system_prompt: str,
    recorder: ModelCallRecorder | None,
    role: str,
    round_number: int | None,
    task_question: str | None,
    image_inputs: Sequence[ImageInputRef],
    image_urls: Sequence[str] | None = None,
    task_number: int | None = None,
    task_count: int | None = None,
) -> tuple[object, str | None]:
    call_id = (
        recorder.start_call(
            role=role,
            round_number=round_number,
            task_question=task_question,
            system_prompt=system_prompt,
            user_prompt=prompt,
            image_inputs=image_inputs,
            task_number=task_number,
            task_count=task_count,
            model=getattr(llm, "model", None),
        )
        if recorder is not None
        else None
    )
    started = time.perf_counter()
    supports_recording = _supports_recording(llm)
    try:
        if supports_recording:
            payload = llm.complete_json(
                prompt,
                system=system_prompt,
                image_urls=image_urls,
                recorder=recorder,
                call_id=call_id,
            )
        else:
            payload = llm.complete_json(
                prompt,
                system=system_prompt,
                image_urls=image_urls,
            )
    except Exception as exc:
        if recorder is not None and not supports_recording and call_id is not None:
            recorder.error_call(
                call_id,
                error=f"{type(exc).__name__}: {exc}",
                attempt_count=1,
                latency_seconds=time.perf_counter() - started,
                model=getattr(llm, "model", None),
            )
        raise
    if recorder is not None and not supports_recording and call_id is not None:
        recorder.complete_call(
            call_id,
            raw_response=None,
            parsed_response=payload if isinstance(payload, dict) else None,
            json_repaired=False,
            error=None,
            attempt_count=1,
            latency_seconds=time.perf_counter() - started,
            model=getattr(llm, "model", None),
        )
    return payload, call_id


def _supports_recording(llm: Any) -> bool:
    try:
        parameters = inspect.signature(llm.complete_json).parameters.values()
    except (TypeError, ValueError):
        return False
    parameters = list(parameters)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return True
    names = {parameter.name for parameter in parameters}
    return {"recorder", "call_id"}.issubset(names)


def _mark_validation(
    recorder: ModelCallRecorder | None,
    call_id: str | None,
    error: str,
) -> None:
    if recorder is not None and call_id is not None:
        recorder.validation_error(call_id, error)


def _state_payload(state: AgentState) -> dict[str, object]:
    cumulative_unresolved = tuple(
        dict.fromkeys(
            question
            for step in state.steps
            for question in step.action.unresolved_questions
        )
    )
    coverage = {key: "unresolved" for key in DECISION_CHECKLIST}
    for step in state.steps:
        if step.action.checklist_coverage:
            coverage.update(step.action.checklist_coverage)
    return {
        "history": [
            {
                "round_number": step.round_number,
                "kind": step.action.kind,
                "questions": [task.question for task in step.action.tasks],
                "rationale": step.action.rationale,
                "assessment": step.action.assessment,
                "conclusion_at_risk": step.action.conclusion_at_risk,
                "missing_evidence": step.action.missing_evidence,
                "expected_judgment_delta": step.action.expected_judgment_delta,
                "unresolved_questions": step.action.unresolved_questions,
                "checklist_coverage": step.action.checklist_coverage,
            }
            for step in state.steps
        ],
        "findings": _state_findings(state),
        "worker_failures": _worker_failure_diagnostics(state),
        "worker_suggested_questions": [
            {
                "origin_question": result.task.question,
                "questions": list(result.suggested_questions),
            }
            for result in state.findings
            if result.suggested_questions
        ],
        "reflection_reports": [
            {
                "trigger": report.trigger,
                "reflected_finding_ids": list(report.reflected_finding_ids),
                "reflection_memo": report.reflection_memo,
                "error": report.error,
            }
            for report in state.reflection_reports
        ],
        "provisional_assessment": state.provisional_assessment,
        "unresolved_questions": state.unresolved_questions,
        "cumulative_unresolved_questions": cumulative_unresolved,
        "checklist_coverage": coverage,
        "remaining_rounds": state.remaining_rounds,
    }


def _master_state_payload(state: AgentState) -> dict[str, object]:
    records = completed_finding_records(state)
    by_task: dict[tuple[int, int], list[str]] = {}
    by_id = {record.finding_id: record for record in records}
    for step in state.steps:
        for task_number, result in enumerate(step.results, start=1):
            if result.error is not None:
                continue
            for finding_number, _finding in enumerate(
                result.structured_findings or _legacy_structured_findings(result), start=1
            ):
                finding_id = f"r{step.round_number}-t{task_number}-f{finding_number}"
                if finding_id in by_id:
                    by_task.setdefault((step.round_number, task_number), []).append(finding_id)
    completed_tasks = [
        {
            "round_number": step.round_number,
            "task_number": task_number,
            "question": result.task.question,
            "pages_read": list(result.pages_read),
            "status": "failed" if result.error is not None else "completed",
            "finding_ids": by_task.get((step.round_number, task_number), []),
        }
        for step in state.steps
        for task_number, result in enumerate(step.results, start=1)
    ]
    findings = [
        {
            "finding_id": record.finding_id,
            "question": record.question,
            "finding": record.finding,
            "caveat": record.caveat,
            "evidence_type": record.evidence_type,
            "evidence_locator": record.evidence_locator,
            "pages_read": list(result.pages_read),
        }
        for step in state.steps
        for task_number, result in enumerate(step.results, start=1)
        if result.error is None
        for finding_number, _finding in enumerate(
            result.structured_findings or _legacy_structured_findings(result), start=1
        )
        if (record := by_id.get(f"r{step.round_number}-t{task_number}-f{finding_number}")) is not None
    ]
    coverage = {key: "unresolved" for key in DECISION_CHECKLIST}
    for step in state.steps:
        if step.action.checklist_coverage:
            coverage.update(step.action.checklist_coverage)
    latest_step = state.steps[-1] if state.steps else None
    latest_action = None if latest_step is None else {
        "round_number": latest_step.round_number,
        "kind": latest_step.action.kind,
        "conclusion_at_risk": latest_step.action.conclusion_at_risk,
        "missing_evidence": latest_step.action.missing_evidence,
        "expected_judgment_delta": latest_step.action.expected_judgment_delta,
    }
    latest_reflection = state.reflection_reports[-1] if state.reflection_reports else None
    latest_suggestions = next(
        (result for result in reversed(state.findings) if result.suggested_questions), None
    )
    return {
        "completed_tasks": completed_tasks,
        "findings": findings,
        "worker_failures": _worker_failure_diagnostics(state),
        "latest_action": latest_action,
        "latest_reflection_report": None if latest_reflection is None else {
            "trigger": latest_reflection.trigger,
            "reflected_finding_ids": list(latest_reflection.reflected_finding_ids),
            "reflection_memo": latest_reflection.reflection_memo,
            "error": latest_reflection.error,
        },
        "provisional_assessment": state.provisional_assessment,
        "unresolved_questions": state.unresolved_questions,
        "checklist_coverage": coverage,
        "remaining_rounds": state.remaining_rounds,
        "latest_worker_suggested_questions": [] if latest_suggestions is None else [
            {
                "origin_question": latest_suggestions.task.question,
                "questions": list(latest_suggestions.suggested_questions),
            }
        ],
    }


def _finding_id_for_result(state: AgentState, result: WorkerResult) -> str | None:
    for record in completed_finding_records(state):
        if (
            record.question == result.task.question
            and record.finding == result.finding
            and record.evidence == result.evidence
            and record.caveat == result.caveat
            and record.evidence_type == result.evidence_type
            and record.evidence_locator == result.evidence_locator
        ):
            return record.finding_id
    return None


def _state_findings(state: AgentState) -> list[dict[str, object]]:
    records = completed_finding_records(state)
    if records:
        by_id = {record.finding_id: record for record in records}
        payload: list[dict[str, object]] = []
        for step in state.steps:
            for task_number, result in enumerate(step.results, start=1):
                if result.error is not None:
                    continue
                findings = result.structured_findings or _legacy_structured_findings(result)
                for finding_number, _finding in enumerate(findings, start=1):
                    record = by_id.get(
                        f"r{step.round_number}-t{task_number}-f{finding_number}"
                    )
                    if record is None:
                        continue
                    payload.append(
                        {
                            "finding_id": record.finding_id,
                            "question": record.question,
                            "finding": record.finding,
                            "evidence": record.evidence,
                            "caveat": record.caveat,
                            "evidence_type": record.evidence_type,
                            "evidence_locator": record.evidence_locator,
                            "evidence_items": [
                                {
                                    "content": evidence.content,
                                    "evidence_type": evidence.evidence_type,
                                    "locator": evidence.locator,
                                }
                                for evidence in record.evidence_items
                            ],
                            "pages_read": list(result.pages_read),
                            "location_rationale": result.location_rationale,
                        }
                    )
        return payload
    return [
        {
            "finding_id": _finding_id_for_result(state, result),
            "question": result.task.question,
            "finding": result.finding,
            "evidence": result.evidence,
            "caveat": result.caveat,
            "error": result.error,
            "pages_read": result.pages_read,
            "location_rationale": result.location_rationale,
            "evidence_type": result.evidence_type,
            "evidence_locator": result.evidence_locator,
        }
        for result in state.findings
    ]


def _worker_failure_diagnostics(state: AgentState) -> list[dict[str, object]]:
    return [
        {
            "question": result.task.question,
            "error": result.error,
            "pages_read": list(result.pages_read),
            "location_rationale": result.location_rationale,
        }
        for result in state.findings
        if result.error is not None
    ]


def _legacy_structured_findings(result: WorkerResult) -> tuple[WorkerFinding, ...]:
    if not result.finding.strip() or not result.evidence.strip():
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
