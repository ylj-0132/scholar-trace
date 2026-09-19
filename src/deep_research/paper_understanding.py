"""Sourced method explanations; validation checks structure, not scientific truth."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Collection, TYPE_CHECKING

if TYPE_CHECKING:
    from deep_research.paper_agent import AgentState, FinalJudgment


METHOD_SECTION_GUIDANCE = {
    "problem_definition": "Explain the research problem, inputs, outputs, objective and scope.",
    "core_contribution": "Explain the core idea and design motivation; distinguish author rationale from a verified explanation of gains.",
    "representations_and_components": "Explain important representations, variables, states and component responsibilities, adapting to the paper rather than assuming a software agent.",
    "method_workflow": "Walk through the complete method from input to output: ordered operations, information passed, state changes, branches and training/inference distinctions when applicable.",
    "key_details_and_assumptions": "Explain essential rules, equations, parameters, assumptions and operating conditions; define notation and distinguish unreported details from details not yet read.",
    "worked_example": "Walk through a concrete example using the established method. Prefer a supplied paper example; otherwise label a constructed illustration and its assumptions explicitly. Do not invent reported results or unreported algorithm rules.",
}
METHOD_SECTION_IDS = tuple(METHOD_SECTION_GUIDANCE)

METHOD_REVIEW_ASPECTS = ("problem", "modules", "workflow_and_branches", "details", "example", "evaluation_support")


@dataclass(frozen=True)
class MethodGap:
    question: str
    disposition: str
    reason: str


@dataclass(frozen=True)
class MethodReview:
    summary: str
    source_finding_ids: tuple[str, ...]
    essential_finding_ids: tuple[str, ...]
    checked_aspects: tuple[str, ...]
    gaps: tuple[MethodGap, ...]
    stop_reason: str = ""


def method_review_shape() -> dict[str, object]:
    return {
        "summary": "Concise current input-to-output understanding; distinguish evidence from provisional overview interpretation",
        "source_finding_ids": [], "essential_finding_ids": [],
        "checked_aspects": list(METHOD_REVIEW_ASPECTS),
        "gaps": [{"question": "Specific missing understanding", "disposition": "paper_check|bounded|external", "reason": "Next source to check, or honest reason to leave this gap bounded"}],
        "stop_reason": "Before DECIDE explain why method understanding is adequate despite explicitly bounded gaps",
    }


def parse_method_review(raw: object) -> MethodReview:
    if not isinstance(raw, dict):
        raise ValueError("method_review must be an object")
    def strings(key: str) -> tuple[str, ...]:
        values = raw.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
            raise ValueError(f"method_review:{key}")
        values = tuple(v.strip() for v in values)
        if len(values) != len(set(values)):
            raise ValueError(f"method_review:{key}:duplicates")
        return values
    summary, stop = raw.get("summary"), raw.get("stop_reason", "")
    if not isinstance(summary, str) or not summary.strip() or not isinstance(stop, str):
        raise ValueError("method_review:summary or stop_reason")
    sources, essential, checked = strings("source_finding_ids"), strings("essential_finding_ids"), strings("checked_aspects")
    if not set(essential) <= set(sources) or not set(checked) <= set(METHOD_REVIEW_ASPECTS):
        raise ValueError("method_review:invalid essential sources or aspects")
    gaps = raw.get("gaps", [])
    if not isinstance(gaps, list):
        raise ValueError("method_review:gaps")
    parsed = []
    for gap in gaps:
        if (not isinstance(gap, dict) or gap.get("disposition") not in ("paper_check", "bounded", "external")
                or any(not isinstance(gap.get(k), str) or not gap[k].strip() for k in ("question", "reason"))):
            raise ValueError("method_review:gap")
        parsed.append(MethodGap(gap["question"].strip(), gap["disposition"], gap["reason"].strip()))
    return MethodReview(summary.strip(), sources, essential, checked, tuple(parsed), stop.strip())


def validate_method_review(review: MethodReview | None, known_ids: Collection[str], *, stopping: bool = False) -> None:
    if review is None:
        if stopping:
            raise ValueError("method_review required before DECIDE")
        return
    if not set(review.source_finding_ids) <= set(known_ids):
        raise ValueError("method_review:unknown or future source")
    if stopping and (set(review.checked_aspects) != set(METHOD_REVIEW_ASPECTS)
                     or not review.stop_reason or any(g.disposition == "paper_check" for g in review.gaps)):
        raise ValueError("method_review:incomplete stop review")


def latest_method_review(state: AgentState) -> MethodReview | None:
    return next((step.action.method_review for step in reversed(state.steps) if step.action.method_review is not None), None)


def method_review_payload(state: AgentState) -> dict[str, object] | None:
    review = latest_method_review(state)
    if review is None:
        return None
    payload = asdict(review)
    for key in ("source_finding_ids", "essential_finding_ids", "checked_aspects", "gaps"):
        payload[key] = list(payload[key])
    return payload


def check_method_retention(judgment: FinalJudgment, review: MethodReview | None) -> FinalJudgment:
    if review is None:
        return judgment
    represented = {source for section in (judgment.method_understanding.sections if judgment.method_understanding else ())
                   for source in section.source_finding_ids}
    warnings = tuple(f"essential_method_finding_unrepresented:{source}" for source in review.essential_finding_ids if source not in represented)
    if not warnings:
        return judgment
    return replace(judgment, method_understanding_warnings=judgment.method_understanding_warnings + warnings,
                   provenance_warnings=judgment.provenance_warnings + warnings, provenance_status="incomplete")


@dataclass(frozen=True)
class MethodSection:
    section_id: str
    explanation: str
    basis: str
    source_finding_ids: tuple[str, ...] = ()
    caveat: str = ""


@dataclass(frozen=True)
class MethodUnderstanding:
    sections: tuple[MethodSection, ...]
    unresolved_questions: tuple[str, ...] = ()


def method_understanding_shape() -> dict[str, object]:
    return {
        "sections": [{
            "section_id": key, "explanation": guidance,
            "basis": "paper|inference|illustrative|unresolved|not_applicable",
            "source_finding_ids": ["r1-t1-f1"],
            "caveat": "Scope, uncertainty and any explanatory assumptions; possibly empty",
        } for key, guidance in METHOD_SECTION_GUIDANCE.items()],
        "unresolved_questions": ["A specific remaining gap in understanding, if any"],
    }


def parse_method_understanding(
    raw: object, known_finding_ids: Collection[str],
) -> tuple[MethodUnderstanding | None, tuple[str, ...]]:
    if not isinstance(raw, dict) or not isinstance(raw.get("sections"), list):
        return None, ("invalid_method_understanding",)
    warnings: set[str] = set()
    sections: dict[str, MethodSection] = {}
    for item in raw["sections"]:
        if not isinstance(item, dict) or item.get("section_id") not in METHOD_SECTION_IDS:
            warnings.add("invalid_method_section")
            continue
        key = item["section_id"]
        if key in sections:
            warnings.add(f"duplicate_method_section:{key}")
            continue
        explanation, basis, caveat = item.get("explanation"), item.get("basis"), item.get("caveat", "")
        if (not isinstance(explanation, str) or not explanation.strip()
                or not isinstance(basis, str)
                or basis not in {"paper", "inference", "illustrative", "unresolved", "not_applicable"}
                or not isinstance(caveat, str)):
            warnings.add(f"invalid_method_section:{key}")
            continue
        if basis == "illustrative" and key != "worked_example":
            warnings.add(f"illustrative_method_section:{key}")
        if basis == "not_applicable" and key != "worked_example":
            warnings.add(f"inapplicable_core_method_section:{key}")
        raw_ids = item.get("source_finding_ids", [])
        if not isinstance(raw_ids, list) or any(not isinstance(v, str) or not v.strip() for v in raw_ids):
            warnings.add(f"invalid_method_sources:{key}")
            raw_ids = []
        source_ids: list[str] = []
        for finding_id in raw_ids:
            finding_id = finding_id.strip()
            if finding_id not in known_finding_ids:
                warnings.add(f"unknown_method_source:{key}:{finding_id}")
            elif finding_id not in source_ids:
                source_ids.append(finding_id)
        if basis in {"paper", "inference", "illustrative"} and not source_ids:
            warnings.add(f"unlinked_method_section:{key}")
        sections[key] = MethodSection(key, explanation.strip(), basis, tuple(source_ids), caveat.strip())
    warnings.update(f"missing_method_section:{key}" for key in METHOD_SECTION_IDS if key not in sections)
    questions = raw.get("unresolved_questions", [])
    if not isinstance(questions, list) or any(not isinstance(q, str) or not q.strip() for q in questions):
        warnings.add("invalid_method_unresolved_questions")
        questions = []
    return MethodUnderstanding(
        tuple(sections[key] for key in METHOD_SECTION_IDS if key in sections),
        tuple(q.strip() for q in questions),
    ), tuple(sorted(warnings))
