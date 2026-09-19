"""Persistent, sourced details inside the Rubric layer; text comparisons are human-only advisories, never semantic gates."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Collection

if TYPE_CHECKING:
    from .paper_agent import AgentState, FinalJudgment


@dataclass(frozen=True)
class KeyDetail:
    text: str
    rubric_id: str
    destination: str


@dataclass(frozen=True)
class DetailUpdate:
    detail_id: str | None
    status: str
    text: str
    rubric_id: str
    destination: str
    source_finding_ids: tuple[str, ...]
    reason: str


def _normalized(text: str) -> str:
    return " ".join(text.split())


def _contained(text: str, passages: Collection[str]) -> bool:
    return bool(text.strip()) and any(_normalized(text) in _normalized(p) for p in passages)


def detail_shape() -> dict[str, object]:
    return {"text": "One self-contained concrete fact grounded in this finding or its evidence; paraphrase is allowed while preserving values, units, conditions and caveats",
            "rubric_id": "one content dimension ID", "destination": "method|evaluation"}


def detail_update_shape() -> dict[str, object]:
    return {**detail_shape(), "detail_id": "existing detail ID, or null to promote a fact from existing Worker evidence", "status": "active|withdrawn",
            "source_finding_ids": ["r1-t1-f1"], "reason": "Why correct, reclassify, restore or withdraw this existing detail; never omit it just for brevity"}


def disposition_shape() -> dict[str, object]:
    return {"detail_id": "r1-t1-f1-d1", "status": "retained|corrected|unresolved|omitted",
            "method_section_id": "section ID or null", "final_finding_index": "1-based index or null",
            "report_quote": "Exact sentence occurring in the designated report section",
            "source_finding_ids": ["r1-t1-f1"], "reason": "Explanation of this disposition"}


def parse_key_details(raw: object, allowed: Collection[str], passages: Collection[str]):
    if not isinstance(raw, list):
        return (), ("invalid_key_details",)
    parsed, warnings = [], []
    for i, item in enumerate(raw, 1):
        if (not isinstance(item, dict) or not isinstance(item.get("text"), str) or not item["text"].strip()
                or not isinstance(item.get("rubric_id"), str) or item["rubric_id"] not in allowed
                or item.get("destination") not in ("method", "evaluation")):
            warnings.append(f"key_detail:{i}:invalid_fields")
            continue
        if not _contained(item["text"], passages):
            warnings.append(f"human_review:key_detail:{i}:text_not_verbatim")
        parsed.append(KeyDetail(item["text"].strip(), item["rubric_id"], item["destination"]))
    return tuple(parsed), tuple(warnings)


def parse_detail_updates(raw: object, allowed: Collection[str]):
    from .rubric_content import string_ids
    if not isinstance(raw, list):
        return (), ("invalid_detail_updates",)
    parsed, warnings = [], []
    for i, item in enumerate(raw, 1):
        ids = string_ids(item.get("source_finding_ids")) if isinstance(item, dict) else None
        if (not isinstance(item, dict) or not ids
                or any(not isinstance(item.get(k), str) or not item[k].strip() for k in ("text", "reason", "rubric_id"))
                or (item.get("detail_id") is not None and (not isinstance(item["detail_id"], str) or not item["detail_id"].strip()))
                or (item.get("detail_id") is None and item.get("status") != "active")
                or item["rubric_id"] not in allowed or item.get("destination") not in ("method", "evaluation")
                or item.get("status") not in ("active", "withdrawn")):
            warnings.append(f"detail_update:{i}:invalid_fields")
            continue
        parsed.append(DetailUpdate(item["detail_id"].strip() if item.get("detail_id") is not None else None, item["status"], item["text"].strip(),
                                   item["rubric_id"], item["destination"], ids, item["reason"].strip()))
    return tuple(parsed), tuple(warnings)


def _passages(records, ids):
    return [text for key in ids for text in [records[key].finding, records[key].evidence,
                                            *(e.content for e in records[key].evidence_items)]]


def build_detail_ledger(state: AgentState, allowed: Collection[str]):
    from .paper_agent import AgentState, completed_finding_records
    records = {r.finding_id: r for r in completed_finding_records(state)}
    current, history, warnings = {}, [], []
    visible = set()
    for step in state.steps:
        changed = set()
        for i, update in enumerate(step.action.detail_updates, 1):
            prefix = f"r{step.round_number}-detail-update-{i}"
            ident = update.detail_id or f"r{step.round_number}-master-detail-{i}"
            if update.detail_id is not None and update.detail_id not in current:
                warnings.append(f"{prefix}:unknown_detail")
                continue
            if ident in changed:
                warnings.append(f"{prefix}:duplicate_detail_update")
                continue
            if not set(update.source_finding_ids) <= visible:
                warnings.append(f"{prefix}:unknown_or_future_finding")
                continue
            if not _contained(update.text, _passages(records, update.source_finding_ids)):
                warnings.append(f"human_review:{prefix}:replacement_not_verbatim")
            value = {**asdict(update), "id": ident, "source_finding_ids": list(update.source_finding_ids),
                     "round_number": step.round_number, "role": "master"}
            value.pop("detail_id")
            current[ident] = value
            history.append(dict(value))
            changed.add(ident)
        for ti, result in enumerate(step.results, 1):
            if result.error is not None:
                continue
            for fi, finding in enumerate(result.structured_findings, 1):
                fid = f"r{step.round_number}-t{ti}-f{fi}"
                if fid not in records:
                    continue
                for di, detail in enumerate(finding.key_details, 1):
                    ident = f"{fid}-d{di}"
                    if (detail.rubric_id not in allowed or detail.destination not in ("method", "evaluation")
                            or not isinstance(detail.text, str) or not detail.text.strip()):
                        warnings.append(f"{ident}:invalid_key_detail")
                        continue
                    value = {**asdict(detail), "id": ident, "source_finding_ids": [fid], "status": "active",
                             "round_number": step.round_number, "role": "evidence", "reason": "Worker extracted detail"}
                    current[ident] = value
                    history.append(dict(value))
            visible.update(r.finding_id for r in completed_finding_records(AgentState(steps=(step,))))
    return list(current.values()), history, warnings


def check_detail_retention(judgment: FinalJudgment, raw: object, content: dict, state: AgentState) -> FinalJudgment:
    from .paper_agent import completed_finding_records
    from .rubric_content import string_ids
    details = {d["id"]: d for d in content["key_details"] if d["status"] == "active"}
    records = {r.finding_id: r for r in completed_finding_records(state)}
    warnings = [w for w in content["warnings"] if "detail" in w]
    review = list(content.get("human_review_warnings", []))
    rows, seen = [], set()
    if not isinstance(raw, list):
        warnings.append("invalid_detail_dispositions")
        raw = []
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("detail_id"), str):
            warnings.append("invalid_detail_disposition")
            continue
        ident = item["detail_id"]
        if ident not in details or ident in seen:
            warnings.append(f"{ident}:unknown_or_duplicate_detail_disposition")
            continue
        seen.add(ident)
        rows.append(dict(item))
        detail = details[ident]
        status, reason, quote = item.get("status"), item.get("reason"), item.get("report_quote")
        sources = string_ids(item.get("source_finding_ids"))
        if (status not in ("retained", "corrected", "unresolved", "omitted") or not isinstance(reason, str)
                or not reason.strip() or not sources or not set(sources) <= set(records)):
            warnings.append(f"{ident}:invalid_detail_disposition")
            continue
        if status in ("unresolved", "omitted"):
            warnings.append(f"{ident}:{status}")
            continue
        section, index = item.get("method_section_id"), item.get("final_finding_index")
        target = None
        if detail["destination"] == "method" and index is None and judgment.method_understanding:
            target = next((s for s in judgment.method_understanding.sections if s.section_id == section), None)
        elif detail["destination"] == "evaluation" and section is None and type(index) is int and 1 <= index <= len(judgment.key_findings):
            target = judgment.key_findings[index - 1]
        if target is None:
            warnings.append(f"{ident}:invalid_report_destination")
            continue
        text = target.explanation if detail["destination"] == "method" else target.finding
        if (not isinstance(quote, str) or not quote.strip()
                or not set(sources) <= set(target.source_finding_ids)
                or not set(detail["source_finding_ids"]) <= set(target.source_finding_ids)):
            warnings.append(f"{ident}:unlinked_report_text")
            continue
        if not _contained(quote, [text, target.caveat]):
            review.append(f"human_review:{ident}:report_excerpt_not_verbatim")
        if status == "retained" and _normalized(quote) != _normalized(detail["text"]):
            review.append(f"human_review:{ident}:detail_text_differs")
        elif status == "corrected" and (not _contained(quote, _passages(records, sources))
                                        or _normalized(quote) == _normalized(detail["text"])):
            review.append(f"human_review:{ident}:correction_text_needs_review")
    warnings.extend(f"{ident}:missing_detail_disposition" for ident in details if ident not in seen)
    status = "incomplete" if warnings else "complete" if content["key_details"] else "not_checked"
    return replace(judgment, detail_dispositions=tuple(rows), detail_retention_status=status,
                   detail_retention_warnings=tuple(warnings),
                   human_review_warnings=tuple(dict.fromkeys(review)),
                   provenance_status="incomplete" if warnings else judgment.provenance_status,
                   provenance_warnings=judgment.provenance_warnings + tuple(warnings))
