"""Trace-derived content management, separate from task routing and paper evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Collection, Sequence

if TYPE_CHECKING:
    from .paper_agent import AgentState


@dataclass(frozen=True)
class RubricEntry:
    rubric_id: str
    kind: str
    text: str
    source_finding_ids: tuple[str, ...] = ()
    source_entry_ids: tuple[str, ...] = ()
    supersedes: tuple[str, ...] = ()
    reason: str = ""
    status: str = "active"


@dataclass(frozen=True)
class RubricLink:
    finding_id: str
    rubric_ids: tuple[str, ...]
    reason: str


def string_ids(raw: object) -> tuple[str, ...] | None:
    if not isinstance(raw, (list, tuple)) or any(not isinstance(v, str) or not v.strip() for v in raw):
        return None
    ids = tuple(v.strip() for v in raw)
    return ids if len(ids) == len(set(ids)) else None


def parse_content_rubrics(raw: object, allowed: Collection[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ids = string_ids(raw)
    if ids is None or any(key not in allowed for key in ids):
        return (), ("invalid_content_rubric_ids",)
    return ids, ()


def parse_entries(
    raw: object, allowed: Collection[str], *, reflection: bool = False,
    known_findings: Collection[str] | None = None,
) -> tuple[tuple[RubricEntry, ...], tuple[str, ...]]:
    if not isinstance(raw, list):
        return (), ("invalid_rubric_entries",)
    entries, warnings = [], []
    for index, item in enumerate(raw, 1):
        prefix = f"rubric_entry:{index}"
        if not isinstance(item, dict):
            warnings.append(f"{prefix}:invalid_object")
            continue
        rubric = item.get("rubric_id")
        kind, text, reason = item.get("kind"), item.get("text"), item.get("reason", "")
        status = item.get("status", "active")
        sources = string_ids(item.get("source_finding_ids", []))
        ancestors = string_ids(item.get("source_entry_ids", []))
        supersedes = string_ids(item.get("supersedes", []))
        if (not isinstance(rubric, str) or rubric not in allowed
                or not isinstance(kind, str) or kind not in {"understanding", "assessment", "qualification", "open_question"}
                or not isinstance(text, str) or not text.strip()
                or not isinstance(reason, str) or (not reflection and not reason.strip())
                or not isinstance(status, str) or status not in {"active", "resolved", "withdrawn"}
                or sources is None or ancestors is None or supersedes is None):
            warnings.append(f"{prefix}:invalid_fields")
            continue
        if ((reflection or kind != "open_question") and not sources
                or status != "active" and not supersedes
                or reflection and (ancestors or supersedes or status != "active")):
            warnings.append(f"{prefix}:invalid_references")
            continue
        if known_findings is not None and any(key not in known_findings for key in sources):
            warnings.append(f"{prefix}:unknown_finding")
            continue
        entries.append(RubricEntry(rubric, kind, text.strip(), sources, ancestors, supersedes, reason.strip(), status))
    return tuple(entries), tuple(warnings)


def parse_links(raw: object, allowed: Collection[str]) -> tuple[tuple[RubricLink, ...], tuple[str, ...]]:
    if not isinstance(raw, list):
        return (), ("invalid_rubric_links",)
    links, warnings = [], []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            warnings.append(f"rubric_link:{index}:invalid_object")
            continue
        finding_id, reason = item.get("finding_id"), item.get("reason")
        ids, errors = parse_content_rubrics(item.get("rubric_ids"), allowed)
        if (errors or not isinstance(finding_id, str) or not finding_id.strip()
                or not isinstance(reason, str) or not reason.strip()):
            warnings.append(f"rubric_link:{index}:invalid_fields")
            continue
        links.append(RubricLink(finding_id.strip(), ids, reason.strip()))
    return tuple(links), tuple(warnings)


def entry_shape(allowed: Sequence[str], *, reflection: bool = False) -> dict[str, object]:
    shape: dict[str, object] = {
        "rubric_id": "|".join(allowed),
        "kind": "understanding|assessment|qualification|open_question",
        "text": "one concise, specific interpretation, qualification or open question",
        "source_finding_ids": ["r1-t1-f1"],
    }
    if not reflection:
        shape.update(source_entry_ids=[], supersedes=[], status="active|resolved|withdrawn",
                     reason="why this entry is added, corrected, resolved or withdrawn")
    return shape


def build_rubric_content(state: AgentState, rubric_ids: Sequence[str]) -> dict[str, object]:
    # Local import avoids a dependency cycle with the immutable trace types.
    from .paper_agent import completed_finding_records

    records = {record.finding_id: record for record in completed_finding_records(state)}
    associations: dict[str, tuple[str, ...]] = {}
    history: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []
    by_id: dict[str, dict[str, object]] = {}
    active: set[str] = set()
    warnings: list[str] = []

    def add(entry: RubricEntry, entry_id: str, role: str, round_number: int, visible: Collection[str],
            visible_entries: Collection[str]) -> None:
        if any(key not in visible for key in entry.source_finding_ids):
            warnings.append(f"{entry_id}:unknown_finding")
            return
        if any(key not in visible_entries for key in entry.source_entry_ids):
            warnings.append(f"{entry_id}:unknown_source_entry")
            return
        if any(key not in visible_entries or key not in active or by_id[key]["role"] != "master"
               or by_id[key]["rubric_id"] != entry.rubric_id for key in entry.supersedes):
            warnings.append(f"{entry_id}:invalid_supersedes")
            return
        value = asdict(entry)
        for name in ("source_finding_ids", "source_entry_ids", "supersedes"):
            value[name] = list(value[name])
        value.update(id=entry_id, role=role, round_number=round_number)
        entries.append(value)
        by_id[entry_id] = value
        active.difference_update(entry.supersedes)
        if entry.status == "active":
            active.add(entry_id)

    for step in state.steps:
        number = step.round_number
        # Master updates precede this batch's new evidence and Reflection result.
        warnings.extend(f"r{number}-master:{w}" for w in step.action.content_warnings)
        prior_entry_ids = set(by_id)
        for index, entry in enumerate(step.action.rubric_updates, 1):
            add(entry, f"r{number}-master-{index}", "master", number, associations, prior_entry_ids)
        for index, link in enumerate(step.action.rubric_links, 1):
            if link.finding_id not in associations:
                warnings.append(f"r{number}-link-{index}:unknown_finding")
                continue
            associations[link.finding_id] = link.rubric_ids
            history.append({"role": "master", "round_number": number, "finding_id": link.finding_id,
                            "rubric_ids": list(link.rubric_ids), "reason": link.reason})
        for task_number, result in enumerate(step.results, 1):
            if result.error is not None:
                continue
            for finding_number, finding in enumerate(result.structured_findings, 1):
                finding_id = f"r{number}-t{task_number}-f{finding_number}"
                if finding_id not in records:
                    continue
                labels, errors = parse_content_rubrics(finding.content_rubric_ids, rubric_ids)
                warnings.extend(f"{finding_id}:{w}" for w in (*finding.content_warnings, *errors))
                associations[finding_id] = labels
                history.append({"role": "evidence", "round_number": number, "finding_id": finding_id,
                                "rubric_ids": list(labels), "reason": "Worker content association; not verified classification"})
            if not result.structured_findings:
                finding_id = f"r{number}-t{task_number}-f1"
                if finding_id in records:
                    associations[finding_id] = ()
        for report_number, report in enumerate(state.reflection_reports, 1):
            if report.available_after_round != number:
                continue
            warnings.extend(f"reflection-{report_number}:{w}" for w in report.content_warnings)
            if report.error is not None:
                continue
            visible = report.context_finding_ids or (
                report.reflected_finding_ids if report.context_mode == "full-history" else ())
            visible = set(visible).intersection(associations)
            for index, entry in enumerate(report.rubric_notes, 1):
                add(entry, f"reflection-{report_number}-note-{index}", "reflection", number, visible, ())
    for report_number, report in enumerate(state.reflection_reports, 1):
        if report.rubric_notes and report.available_after_round not in {step.round_number for step in state.steps}:
            warnings.append(f"reflection-{report_number}:missing_round")
    from .rubric_details import build_detail_ledger
    details, detail_history, detail_warnings = build_detail_ledger(state, rubric_ids)
    warnings.extend(detail_warnings)
    dimensions = {
        rubric: {
            "finding_ids": [key for key, labels in associations.items() if rubric in labels],
            "detail_ids": [d["id"] for d in details if d["rubric_id"] == rubric and d["status"] == "active"],
            "current_entry_ids": [item["id"] for item in entries if item["rubric_id"] == rubric
                                  and item["role"] == "master" and item["id"] in active],
            "candidate_entry_ids": [item["id"] for item in entries if item["rubric_id"] == rubric
                                    and item["role"] == "reflection"],
        } for rubric in rubric_ids
    }
    human_review = [w for w in warnings if "human_review:" in w]
    warnings = [w for w in warnings if "human_review:" not in w]
    return {"version": "rubric-content-v3-advisory", "human_review_warnings": human_review, "key_details": details, "detail_history": detail_history, "dimensions": dimensions, "entries": entries,
            "association_history": history,
            "unassigned_finding_ids": [key for key in records if not associations.get(key)],
            "warnings": warnings}
