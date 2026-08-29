from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

import deep_research.paper_agent_runtime as runtime
from deep_research.paper_agent import (
    AgentState,
    AgentTrace,
    EvidenceTask,
    FindingEvidence,
    MasterAction,
    ReflectionReport,
    ResearchContext,
    TraceStep,
    WorkerFinding,
    WorkerResult,
    run_paper_agent,
)
from deep_research.paper_reading import PaperPage


class FakeLLM:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        image_urls: list[str] | None = None,
    ) -> object:
        self.calls.append(
            {"prompt": prompt, "system": system, "image_urls": image_urls}
        )
        return self.responses.pop(0)


def test_master_parser_preserves_phase5_value_contract() -> None:
    read = runtime.parse_master_action({
        "kind": "READ_PAPER",
        "tasks": [{
            "question": "Check the evaluation reset policy.",
            "source_scope": "paper",
            "related_finding_ids": [],
            "decision_relevance": "This bounds the continual-learning claim.",
        }],
        "conclusion_at_risk": "The dynamic score measures chronology-safe continual learning.",
        "missing_evidence": "A paper statement of reset, ordering, and update timing.",
        "expected_judgment_delta": "Narrow the claim to sequential adaptation if updates persist.",
    })
    stop = runtime.parse_master_action({
        "kind": "DECIDE",
        "tasks": [],
        "assessment": "The pipeline works, with bounded continual-learning evidence.",
        "stop_reason_code": "paper_saturated",
    })

    assert read.conclusion_at_risk.startswith("The dynamic score")
    assert read.missing_evidence.startswith("A paper statement")
    assert read.expected_judgment_delta.startswith("Narrow the claim")
    assert stop.stop_reason_code == "paper_saturated"


@pytest.mark.parametrize("missing", [
    "conclusion_at_risk",
    "missing_evidence",
    "expected_judgment_delta",
])
def test_read_paper_requires_each_marginal_value_field(missing: str) -> None:
    payload = {
        "kind": "READ_PAPER",
        "tasks": [{"question": "q", "source_scope": "paper"}],
        "conclusion_at_risk": "claim",
        "missing_evidence": "evidence",
        "expected_judgment_delta": "delta",
    }
    del payload[missing]

    with pytest.raises(ValueError, match=missing):
        runtime.parse_master_action(payload)


def test_decide_requires_known_stop_reason_code() -> None:
    with pytest.raises(ValueError, match="stop_reason_code"):
        runtime.parse_master_action({
            "kind": "DECIDE",
            "tasks": [],
            "assessment": "Enough evidence.",
            "stop_reason_code": "more_rounds_exist",
        })


class RoutedFakeLLM:
    def __init__(self, model: str, *, master: bool = False, reflection: bool = False) -> None:
        self.model = model
        self.master = master
        self.reflection = reflection
        self.calls: list[str | None] = []
        self._lock = threading.Lock()

    def complete_json(
        self,
        prompt: str,
        *,
        system: str | None = None,
        image_urls: list[str] | None = None,
    ) -> object:
        with self._lock:
            self.calls.append(system)
        if self.master:
            return {
                "kind": "READ_PAPER",
                "tasks": [
                    {"question": "What does the method change?", "source_scope": "paper"},
                    {"question": "What does the experiment compare?", "source_scope": "paper"},
                ],
                "assessment": "A provisional assessment.",
                "conclusion_at_risk": "The claimed mechanism may be overstated.",
                "missing_evidence": "A bounded method comparison.",
                "expected_judgment_delta": "Bound the mechanism claim.",
            }
        if self.reflection:
            return {"reflection_memo": "A candidate mechanism concern."}
        if system == runtime.LOCATOR_SYSTEM_PROMPT:
            return {"page_ranges": [{"start": 2, "end": 2}], "rationale": "method page"}
        if system == runtime.EVIDENCE_SYSTEM_PROMPT:
            return {
                "finding": "The ablation removes the planner.",
                "evidence": "Selected evidence: the ablation removes the planner.",
                "caveat": "Reported benchmark only.",
                "evidence_type": "text",
                "evidence_locator": "paper.pdf, p. 2",
            }
        if system == runtime.SYNTHESIS_SYSTEM_PROMPT:
            return _single_pass_payload()
        raise AssertionError(f"unexpected system prompt: {system}")


def _single_pass_payload() -> dict[str, object]:
    return {
        "assessment": "The paper makes a bounded planner contribution.",
        "key_findings": [
            {
                "finding": "The ablation removes the planner.",
                "evidence": "Selected evidence: the ablation removes the planner.",
                "caveat": "The evidence is limited to the reported benchmark.",
                "evidence_type": "text",
                "evidence_locator": "paper.pdf, p. 2",
            }
        ],
    }


def _patch_single_pass_io(
    monkeypatch: object,
    pages: list[PaperPage],
    rendered: list[tuple[int, ...]] | None = None,
) -> None:
    def fake_extract(path: Path) -> list[PaperPage]:
        return pages

    def fake_render(path: Path, page_numbers: list[int]) -> dict[int, str]:
        if rendered is not None:
            rendered.append(tuple(page_numbers))
        return {page: f"data:image/png;base64,page-{page}" for page in page_numbers}

    monkeypatch.setattr(runtime, "extract_pdf_pages", fake_extract)
    monkeypatch.setattr(runtime, "render_pdf_pages", fake_render)


def make_pages() -> list[PaperPage]:
    return [
        PaperPage(
            page_number=1,
            text="Abstract\nThe paper proposes a bounded planner.\nFigure 1: overview.",
            char_count=70,
            low_text=False,
            used_fallback=False,
        ),
        PaperPage(
            page_number=2,
            text=(
                "2 Method\n"
                + ("method prefix " * 80)
                + "\nSelected evidence: the ablation removes the planner.\n"
                + "TARGET_PAGE_TEXT_NOT_IN_INDEX"
            ),
            char_count=900,
            low_text=False,
            used_fallback=False,
        ),
        PaperPage(
            page_number=3,
            text="3 Experiments\nTable 2 reports the planner comparison.",
            char_count=60,
            low_text=False,
            used_fallback=False,
        ),
    ]


def test_role_llms_route_parallel_workers_without_cross_role_model_leakage(
    tmp_path: Path, monkeypatch: object
) -> None:
    pages = make_pages()
    _patch_single_pass_io(monkeypatch, pages)
    master = RoutedFakeLLM("openai/gpt-5.6-terra", master=True)
    reflection = RoutedFakeLLM("openai/gpt-5.6-terra", reflection=True)
    luna = RoutedFakeLLM("openai/gpt-5.6-luna")
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    trace = runtime.run_local_paper_agent(
        pdf_path=pdf_path,
        llm=master,
        role_llms={
            "master": master,
            "reflection": reflection,
            "locator": luna,
            "evidence": luna,
            "synthesis": luna,
        },
        max_rounds=1,
        max_reflections=1,
        worker_parallelism=2,
    )

    by_role = {record.role: [] for record in trace.model_calls}
    for record in trace.model_calls:
        by_role[record.role].append(record.model)
    assert by_role["master"] == ["openai/gpt-5.6-terra"]
    assert by_role["reflection"] == ["openai/gpt-5.6-terra"]
    assert by_role["locator"] == ["openai/gpt-5.6-luna"] * 2
    assert by_role["evidence"] == ["openai/gpt-5.6-luna"] * 2
    assert by_role["synthesis"] == ["openai/gpt-5.6-luna"]
    assert len(master.calls) == 1
    assert len(reflection.calls) == 1
    assert len(luna.calls) == 5


def test_compact_page_index_is_deterministic_and_bounded() -> None:
    index = json.loads(runtime.build_compact_page_index(make_pages(), preview_chars=40))

    assert [entry["page_number"] for entry in index] == [1, 2, 3]
    assert "Figure 1: overview." in index[0]["captions"]
    assert "2 Method" in index[1]["headings"]
    assert "TARGET_PAGE_TEXT_NOT_IN_INDEX" not in runtime.build_compact_page_index(
        make_pages(), preview_chars=40
    )


def test_master_keeps_two_page_overview_and_default_400_character_index() -> None:
    pages = make_pages()
    master = runtime.PaperAgentMaster(
        llm=FakeLLM([{"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None}]),
        paper_name="paper.pdf",
        page_index=runtime.build_compact_page_index(pages),
        overview_text=runtime.render_page_text(pages[:2]),
        overview_images=("page-1", "page-2"),
        overview_page_numbers=(1, 2),
    )

    master(AgentState(remaining_rounds=5))

    prompt = json.loads(master.llm.calls[0]["prompt"])
    assert prompt["overview_pages"] == runtime.render_page_text(pages[:2])
    assert [item["page_number"] for item in prompt["image_inputs"]] == [1, 2]
    assert all(len(item["preview"]) <= 400 for item in prompt["compact_page_index"])


def test_evidence_worker_uses_one_locator_and_one_multimodal_evidence_call(
    tmp_path: Path,
) -> None:
    llm = FakeLLM(
        [
            {
                "page_ranges": [{"start": 2, "end": 3}],
                "rationale": "The method and comparison are on these pages.",
            },
            {
                "finding": "The ablation removes the planner.",
                "evidence": "Selected evidence: the ablation removes the planner.",
                "caveat": "This is one ablation.",
                "evidence_type": "text",
                "evidence_locator": "paper.pdf, p. 2",
            },
        ]
    )
    rendered_pages: list[tuple[int, ...]] = []

    def render_pages(path: Path, page_numbers: list[int]) -> dict[int, str]:
        assert path == pdf_path
        rendered_pages.append(tuple(page_numbers))
        return {page: f"data:image/png;base64,page-{page}" for page in page_numbers}

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=render_pages,
    )

    result = worker(EvidenceTask("Does the planner have a demonstrated contribution?"))

    assert len(llm.calls) == 2
    assert llm.calls[0]["image_urls"] is None
    assert json.loads(llm.calls[0]["prompt"])["image_inputs"] == []
    assert "TARGET_PAGE_TEXT_NOT_IN_INDEX" not in llm.calls[0]["prompt"]
    assert "Selected evidence: the ablation removes the planner." in llm.calls[1]["prompt"]
    assert llm.calls[1]["image_urls"] == [
        "data:image/png;base64,page-2",
        "data:image/png;base64,page-3",
    ]
    assert rendered_pages == [(2, 3)]
    assert result.error is None
    assert result.pages_read == (2, 3)
    assert result.location_rationale == "The method and comparison are on these pages."
    assert result.evidence_type == "text"
    assert result.evidence_locator == "paper.pdf, p. 2"
    evidence_prompt = json.loads(llm.calls[1]["prompt"])
    assert evidence_prompt["image_inputs"] == [
        {
            "image_index": 1,
            "source_document": "paper.pdf",
            "page_number": 2,
            "dpi": 144,
        },
        {
            "image_index": 2,
            "source_document": "paper.pdf",
            "page_number": 3,
            "dpi": 144,
        },
    ]


def test_master_prompt_declares_all_action_json_contracts() -> None:
    llm = FakeLLM([{"kind": "NEEDS_HUMAN"}])
    pages = make_pages()
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="actual-paper.pdf",
        page_index=runtime.build_compact_page_index(pages),
        overview_text=pages[0].text,
        overview_images=[],
    )

    master(AgentState(remaining_rounds=1))

    prompt = json.loads(llm.calls[0]["prompt"])
    contract = prompt["output_contract"]
    assert set(contract) == {"READ_PAPER", "DECIDE", "NEEDS_HUMAN"}
    assert contract["READ_PAPER"]["kind"] == "READ_PAPER"
    assert contract["READ_PAPER"]["tasks"][0]["question"]
    assert contract["READ_PAPER"]["tasks"][0]["source_scope"] == "paper"
    assert "conclusion at risk" in contract["READ_PAPER"]["tasks"][0]["decision_relevance"]
    assert "expected judgment delta" in contract["READ_PAPER"]["tasks"][0]["decision_relevance"]
    assert contract["READ_PAPER"]["assessment"] == (
        "current provisional assessment and how existing evidence supports or bounds it"
    )
    assert contract["READ_PAPER"]["conclusion_at_risk"]
    assert contract["READ_PAPER"]["missing_evidence"]
    assert "distinct finding, caveat, experimental boundary, counterexample, or reporting inconsistency" in contract["READ_PAPER"]["expected_judgment_delta"]
    assert "at least one plausible answer" not in contract["READ_PAPER"]["expected_judgment_delta"]
    assert contract["DECIDE"] == {
        "kind": "DECIDE",
        "tasks": [],
        "unresolved_questions": [
            "material question that remains open but does not prevent a supported judgment"
        ],
        "assessment": "evidence-grounded assessment of contribution, evidence strength, and boundaries",
        "rationale": "why the judgment is complete enough to finalize and how remaining questions are bounded",
        "checklist_coverage": {
            key: "covered|unresolved|not_applicable"
            for key in runtime.DECISION_CHECKLIST
        },
        "stop_reason_code": "evidence_sufficient|paper_saturated|remaining_gaps_unreported|remaining_gaps_external",
        "pre_decide_reflection_focus": None,
    }
    assert contract["NEEDS_HUMAN"]["assessment"] is None
    instructions = " ".join(prompt["instructions"])
    assert "at most two" not in instructions
    assert "necessary for a complete, evidence-supported paper judgment" in instructions
    assert "conclusion at risk" in instructions
    assert "missing paper evidence" in instructions
    assert "expected judgment delta" in instructions
    assert "Budget remaining is not evidence value" in instructions
    assert "paper-unreported" in instructions
    assert "do not reopen it" in instructions
    assert "general completeness check" in instructions
    assert "Every task in a multi-task READ_PAPER action must add a distinct, non-duplicate evidence direction" in instructions
    assert "the paper could resolve it" in instructions
    assert "A stable overall assessment is not sufficient reason to stop" in instructions
    assert "need not change the overall positive or negative assessment" in instructions
    assert "one independent direction not originating in Reflection" in instructions
    assert "add or correct a material caveat" in instructions
    assert "no distinct evidence-bearing paper-internal direction" in instructions
    assert "supported finding or bounded limitation" in instructions
    assert "At least one plausible answer must materially" not in instructions
    assert "a merely interesting detail, implementation completeness" not in instructions
    assert "Every task in a multi-task READ_PAPER action must separately pass" not in instructions
    assert "account only for material Worker suggested questions" not in instructions
    assert "could still change the proposed assessment" not in instructions
    assert "no distinct decision-changing direction" not in instructions
    assert "every material Worker suggested question and every high-value paper direction" not in instructions
    assert "missing evidence is available inside the paper" not in instructions
    assert "Use remaining rounds when another distinct bounded question" not in instructions
    assert "not treat the compact page index as verified paper evidence" in " ".join(
        prompt["instructions"]
    )


def test_master_prompt_explicitly_allows_multiple_independent_tasks_per_round() -> None:
    llm = FakeLLM([{"kind": "NEEDS_HUMAN"}])
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview.",
        overview_images=(),
    )

    master(AgentState(remaining_rounds=1))

    instructions = " ".join(json.loads(llm.calls[0]["prompt"])["instructions"])
    assert "A READ_PAPER action may contain multiple independent tasks" in instructions
    assert "do not default to one task" in instructions


def test_master_prompt_routes_load_bearing_headline_claims_to_worker_verification() -> None:
    llm = FakeLLM([{"kind": "NEEDS_HUMAN"}])
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview.",
        overview_images=(),
        paper_context_mode="master-main-text-history-only",
        main_paper_text="The headline claim depends on Table 1 arithmetic.",
    )

    master(AgentState(remaining_rounds=1))

    instructions = " ".join(json.loads(llm.calls[0]["prompt"])["instructions"])
    assert "one or two load-bearing headline claims" in instructions
    assert "arithmetic, metric labels, model or configuration matching" in instructions
    assert "bounded Worker verification task" in instructions
    assert "do not audit every number" in instructions
    assert "verified downstream evidence" in instructions


def test_master_prompt_balances_cross_finding_questions_with_discovery() -> None:
    llm = FakeLLM([{"kind": "NEEDS_HUMAN"}])
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview.",
        overview_images=(),
    )

    master(AgentState(remaining_rounds=2))

    instructions = " ".join(json.loads(llm.calls[0]["prompt"])["instructions"])
    assert "inspect relationships among prior findings" in instructions
    assert "bounded Cross-check task" in instructions
    assert "change, qualify or bound the assessment" in instructions
    assert "do not let Cross-check automatically replace independent Discovery" in instructions
    assert "near-duplicate search for an absent detail" in instructions
    assert "new locator evidence" in instructions
    assert "task's decision_relevance" in instructions
    assert "Do not create extra tasks" not in instructions


def test_reflection_enabled_master_moves_from_design_understanding_to_hypothesis_testing() -> None:
    llm = FakeLLM([
        {"kind": "NEEDS_HUMAN"},
        {"kind": "NEEDS_HUMAN"},
    ])
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview.",
        overview_images=(),
        reflection_enabled=True,
    )

    master(AgentState(remaining_rounds=5))
    master(AgentState(
        reflection_reports=(ReflectionReport(
            trigger="post_method_model",
            reflected_finding_ids=("r1-t1-f1",),
            reflection_memo="A mechanism-level relationship remains to be tested.",
        ),),
        remaining_rounds=4,
    ))

    first_prompt = json.loads(llm.calls[0]["prompt"])
    second_prompt = json.loads(llm.calls[1]["prompt"])
    first_instructions = " ".join(first_prompt["instructions"])
    second_instructions = " ".join(second_prompt["instructions"])
    assert first_prompt["research_phase"] == "understand_design"
    assert "attention priority" in first_instructions.lower()
    assert "all relevant paper sections remain available" in first_instructions.lower()
    assert "information and representation flow" in first_instructions
    assert second_prompt["research_phase"] == "investigate_implications"
    assert "discriminate" in second_instructions
    assert "reflection memo" in second_instructions.lower()
    assert "independent no-context" in second_instructions


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"kind": "DECIDE", "tasks": "not-a-list", "assessment": "bad"}, "tasks must be a list"),
        ({"kind": "NEEDS_HUMAN", "tasks": {}}, "tasks must be a list"),
        (
            {"kind": "READ_PAPER", "tasks": [{"question": 1, "source_scope": "paper"}]},
            "task 1 question must be a string",
        ),
        (
            {"kind": "READ_PAPER", "tasks": [{"question": "q", "source_scope": 1}]},
            "task 1 source_scope must be a string",
        ),
        (
            {"kind": "READ_PAPER", "tasks": [{"question": "q"}]},
            "task 1 source_scope must be a string",
        ),
        (
            {"kind": "READ_PAPER", "unresolved_questions": ["ok", 1]},
            "unresolved_questions[1] must be a string",
        ),
        ({"kind": "DECIDE", "assessment": 1}, "assessment must be a string or null"),
        ({"kind": "DECIDE", "rationale": 1}, "rationale must be a string"),
    ],
)
def test_malformed_master_payload_raises_clear_value_error(
    payload: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        runtime.parse_master_action(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "kind": "READ_PAPER",
            "tasks": [{"question": "q", "source_scope": "paper"}],
            "conclusion_at_risk": "claim",
            "missing_evidence": "evidence",
            "expected_judgment_delta": "delta",
        },
        {
            "kind": "DECIDE",
            "tasks": [],
            "assessment": "assessment",
            "stop_reason_code": "evidence_sufficient",
        },
        {"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None},
    ],
)
def test_normal_master_payloads_still_parse(payload: dict[str, object]) -> None:
    action = runtime.parse_master_action(payload)
    assert action.kind in {"READ_PAPER", "DECIDE", "NEEDS_HUMAN"}


def test_malformed_master_payload_stops_without_running_worker() -> None:
    llm = FakeLLM([{"kind": "DECIDE", "tasks": "bad", "assessment": "x"}])
    pages = make_pages()
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index=runtime.build_compact_page_index(pages),
        overview_text=pages[0].text,
        overview_images=[],
    )
    worker_calls = 0

    def worker(task: EvidenceTask) -> WorkerResult:
        nonlocal worker_calls
        worker_calls += 1
        return WorkerResult(task=task)

    trace = run_paper_agent(master=master, worker=worker, max_rounds=1)

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.stop_reason == "master_error@round_1:ValueError: tasks must be a list"
    assert worker_calls == 0


@pytest.mark.parametrize(
    "locator_payload",
    [
        {"page_ranges": [], "rationale": "empty"},
        {"page_ranges": [{"start": 0, "end": 1}], "rationale": "out of range"},
        {"page_ranges": [{"start": 2, "end": 1}], "rationale": "reversed"},
        {"page_ranges": [{"start": True, "end": 1}], "rationale": "bool"},
        {"page_ranges": [{"start": "1", "end": 1}], "rationale": "string"},
    ],
)
def test_invalid_locator_never_triggers_evidence_call(
    tmp_path: Path,
    locator_payload: dict[str, object],
) -> None:
    llm = FakeLLM([locator_payload])
    render_calls = 0

    def render_pages(path: Path, page_numbers: list[int]) -> dict[int, str]:
        nonlocal render_calls
        render_calls += 1
        return {}

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=render_pages,
    )

    result = worker(EvidenceTask("Find the decisive evidence."))

    assert len(llm.calls) == 1
    assert render_calls == 0
    assert result.error is not None
    assert result.error.startswith("invalid_locator:")


def test_page_selection_allows_many_ranges_pages_and_deduplicates() -> None:
    selected = runtime.validate_page_selection(
        {
            "page_ranges": [
                {"start": 1, "end": 3},
                {"start": 3, "end": 5},
                {"start": 6, "end": 8},
            ]
        },
        page_count=8,
    )

    assert selected == (1, 2, 3, 4, 5, 6, 7, 8)


def test_locator_prompt_has_no_experimental_page_budget() -> None:
    prompt = json.loads(
        runtime._build_locator_prompt(
            question="Find the result.",
            page_index=runtime.build_compact_page_index(make_pages()),
            page_count=3,
        )
    )

    assert "budget" not in prompt
    assert "max_page_ranges" not in json.dumps(prompt)
    assert "max_pages" not in json.dumps(prompt)


def test_locator_prompt_prioritizes_the_exact_page_for_a_named_source() -> None:
    prompt = json.loads(
        runtime._build_locator_prompt(
            question="What exact controls are reported in Appendix B.2 and Table 13?",
            page_index=runtime.build_compact_page_index(make_pages()),
            page_count=3,
        )
    )

    instructions = " ".join(prompt["instructions"])
    assert "page where that label or heading itself appears" in instructions
    assert "nearby pages that merely discuss the same topic" in instructions
    assert "immediately adjacent page" in instructions
    assert "normal one-to-four-page range" in instructions


def test_invalid_locator_preserves_nonempty_rationale(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            {
                "page_ranges": [{"start": 0, "end": 1}],
                "rationale": "  these pages were selected for the ablation  ",
            }
        ]
    )
    rendered = False

    def render_pages(path: Path, page_numbers: list[int]) -> dict[int, str]:
        nonlocal rendered
        rendered = True
        return {}

    pdf_path = tmp_path / "actual-paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    result = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=render_pages,
    )(EvidenceTask("Find the ablation."))

    assert result.pages_read == ()
    assert result.location_rationale == "these pages were selected for the ablation"
    assert result.error == "invalid_locator:page range 0-1 is outside 1..3"
    assert rendered is False


def test_malformed_evidence_result_does_not_trigger_a_third_call(
    tmp_path: Path,
) -> None:
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "method"},
            {},
        ]
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda path, page_numbers: {
            page: f"data:image/png;base64,page-{page}" for page in page_numbers
        },
    )

    result = worker(EvidenceTask("What does the ablation show?"))

    assert len(llm.calls) == 2
    assert result.error == "invalid_evidence_output:finding"


def test_unmatched_text_evidence_is_preserved_without_automatic_matching(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "method"},
            {
                "finding": "A finding",
                "evidence": "This sentence is not on page two.",
                "caveat": "none",
                "evidence_type": "text",
                "evidence_locator": "p. 2",
            },
        ]
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda path, page_numbers: {
            page: f"data:image/png;base64,page-{page}" for page in page_numbers
        },
    )

    result = worker(EvidenceTask("Check the evidence."))

    assert result.error is None
    assert not hasattr(result, "audit_warning")
    assert result.finding == "A finding"
    assert result.evidence == "This sentence is not on page two."
    assert result.caveat == "none"
    assert result.pages_read == (2,)


def test_master_state_keeps_worker_evidence() -> None:
    result = WorkerResult(
        task=EvidenceTask("Check the evidence."),
        finding="A finding",
        evidence="A lightly paraphrased quote.",
        caveat="A caveat.",
    )

    payload = runtime._state_payload(
        AgentState(
            findings=(result,),
            provisional_assessment="provisional assessment",
            remaining_rounds=1,
        )
    )

    finding = payload["findings"][0]
    assert finding["finding"] == "A finding"
    assert finding["evidence"] == "A lightly paraphrased quote."
    assert finding["caveat"] == "A caveat."
    assert finding["error"] is None
    assert payload["provisional_assessment"] == "provisional assessment"


def test_evidence_prompt_requires_verbatim_text_quotes(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "method"},
            {
                "finding": "The ablation removes the planner.",
                "evidence": "Selected evidence: the ablation removes the planner.",
                "caveat": "",
                "evidence_type": "text",
                "evidence_locator": "paper.pdf, p. 2",
            },
        ]
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda path, page_numbers: {
            page: f"data:image/png;base64,page-{page}" for page in page_numbers
        },
    )(EvidenceTask("What does the ablation show?"))

    prompt = json.loads(llm.calls[1]["prompt"])
    instructions = " ".join(prompt["instructions"]).lower()
    assert "verbatim" in instructions
    assert "do not paraphrase" in instructions


@pytest.mark.parametrize("evidence_type", ["table", "figure"])
def test_visual_evidence_requires_locator_and_preserves_it(
    tmp_path: Path,
    evidence_type: str,
) -> None:
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 3, "end": 3}], "rationale": "results"},
            {
                "finding": "The curve improves.",
                "evidence": "The plotted curve is higher.",
                "caveat": "Visual comparison only.",
                "evidence_type": evidence_type,
                "evidence_locator": "Table 2, p. 3",
            },
        ]
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda path, page_numbers: {
            page: f"data:image/png;base64,page-{page}" for page in page_numbers
        },
    )

    result = worker(EvidenceTask("What does the result show?"))

    assert result.error is None
    assert result.evidence_type == evidence_type
    assert result.evidence_locator == "Table 2, p. 3"


def test_missing_visual_locator_is_an_error(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 3, "end": 3}], "rationale": "results"},
            {
                "finding": "The curve improves.",
                "evidence": "The plotted curve is higher.",
                "caveat": "Visual comparison only.",
                "evidence_type": "figure",
                "evidence_locator": "",
            },
        ]
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda path, page_numbers: {
            page: f"data:image/png;base64,page-{page}" for page in page_numbers
        },
    )

    result = worker(EvidenceTask("What does the figure show?"))

    assert result.error == "invalid_evidence_output:evidence_locator"


def test_text_evidence_also_requires_locator(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "method"},
            {
                "finding": "The ablation removes the planner.",
                "evidence": "Selected evidence: the ablation removes the planner.",
                "caveat": "Only one ablation is shown.",
                "evidence_type": "text",
                "evidence_locator": "",
            },
        ]
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda path, page_numbers: {
            page: f"data:image/png;base64,page-{page}" for page in page_numbers
        },
    )

    result = worker(EvidenceTask("What does the ablation show?"))

    assert result.error == "invalid_evidence_output:evidence_locator"


def test_evidence_prompt_contains_actual_paper_name(tmp_path: Path) -> None:
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "method"},
            {
                "finding": "The ablation removes the planner.",
                "evidence": "Selected evidence: the ablation removes the planner.",
                "caveat": "One ablation.",
                "evidence_type": "text",
                "evidence_locator": "actual-paper.pdf, p. 2",
            },
        ]
    )
    pdf_path = tmp_path / "actual-paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda path, page_numbers: {
            page: f"data:image/png;base64,page-{page}" for page in page_numbers
        },
    )(EvidenceTask("What does the ablation show?"))

    evidence_prompt = llm.calls[1]["prompt"]
    assert "actual-paper.pdf" in evidence_prompt
    assert "actual-paper.pdf, p. 6 or Table 3, p. 6" in evidence_prompt


def test_master_second_round_prompt_contains_prior_finding() -> None:
    llm = FakeLLM(
        [
            {
                "kind": "READ_PAPER",
                "tasks": [
                    {
                        "question": "Does the planner improve accuracy?",
                        "source_scope": "paper",
                    }
                ],
                "unresolved_questions": ["What benefit remains?"],
                "conclusion_at_risk": "The planner may not improve accuracy.",
                "missing_evidence": "A direct accuracy comparison.",
                "expected_judgment_delta": "Limit the planner to efficiency.",
            },
            {
                "kind": "READ_PAPER",
                "tasks": [
                    {
                        "question": "Does the planner reduce tokens?",
                        "source_scope": "paper",
                    }
                ],
                "conclusion_at_risk": "The planner value may be efficiency only.",
                "missing_evidence": "The retrieval token comparison.",
                "expected_judgment_delta": "Bound the reported benefit.",
            },
        ]
    )
    pages = make_pages()
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index=runtime.build_compact_page_index(pages),
        overview_text="\n\n".join(page.text for page in pages[:2]),
        overview_images=["data:image/png;base64,overview"],
    )

    first = master(AgentState(remaining_rounds=2))
    second = master(
        AgentState(
            findings=(
                WorkerResult(
                    task=first.tasks[0],
                    finding="The adaptive planner shows no accuracy gain.",
                    evidence="Table 3.",
                    caveat="Benchmark-specific.",
                    pages_read=(2,),
                ),
            ),
            remaining_rounds=1,
        )
    )

    assert first.tasks[0].question == "Does the planner improve accuracy?"
    assert second.tasks[0].question == "Does the planner reduce tokens?"
    assert "no accuracy gain" in llm.calls[1]["prompt"]
    assert llm.calls[1]["image_urls"] == ["data:image/png;base64,overview"]


def test_state_history_preserves_prior_marginal_value_contract() -> None:
    action = MasterAction(
        "READ_PAPER",
        (EvidenceTask("Check the reset policy"),),
        conclusion_at_risk="The score may measure sequential adaptation.",
        missing_evidence="The evaluation reset protocol.",
        expected_judgment_delta="Narrow the claim if memory persists.",
    )

    history = runtime._state_payload(
        AgentState(steps=(TraceStep(1, action),))
    )["history"][0]

    assert history["conclusion_at_risk"] == action.conclusion_at_risk
    assert history["missing_evidence"] == action.missing_evidence
    assert history["expected_judgment_delta"] == action.expected_judgment_delta


def test_run_local_paper_agent_is_end_to_end_with_fake_llm_and_renderer(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    llm = FakeLLM(
        [
            {
                "kind": "READ_PAPER",
                    "tasks": [
                        {
                            "question": "What does the ablation remove?",
                            "source_scope": "paper",
                        }
                    ],
                    "conclusion_at_risk": "The ablation may not isolate the planner.",
                    "missing_evidence": "The ablation result.",
                    "expected_judgment_delta": "Bound the planner contribution.",
            },
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "ablation"},
            {
                "finding": "The ablation removes the planner.",
                "evidence": "Selected evidence: the ablation removes the planner.",
                "caveat": "Only one ablation is shown.",
                "evidence_type": "text",
                "evidence_locator": "paper.pdf, p. 2",
            },
            {
                "kind": "DECIDE",
                "assessment": "The paper demonstrates a bounded planner contribution.",
                "rationale": "The ablation evidence is sufficient.",
                "stop_reason_code": "evidence_sufficient",
            },
            {
                "assessment": "The final synthesis confirms a bounded planner contribution.",
                "key_findings": [
                    {
                        "finding": "The ablation removes the planner.",
                        "evidence": [
                            {
                                "content": "Selected evidence: the ablation removes the planner.",
                                "evidence_type": "text",
                                "locator": "paper.pdf, p. 2",
                            }
                        ],
                        "caveat": "Only one ablation is shown.",
                    }
                ],
            },
        ]
    )
    pages = make_pages()
    rendered: list[tuple[int, ...]] = []

    def fake_extract(path: Path) -> list[PaperPage]:
        assert path == pdf_path
        return pages

    def fake_render(path: Path, page_numbers: list[int]) -> dict[int, str]:
        rendered.append(tuple(page_numbers))
        return {page: f"data:image/png;base64,page-{page}" for page in page_numbers}

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    monkeypatch.setattr(runtime, "extract_pdf_pages", fake_extract)
    monkeypatch.setattr(runtime, "render_pdf_pages", fake_render)

    trace = runtime.run_local_paper_agent(
        pdf_path=pdf_path,
        llm=llm,
        max_rounds=2,
    )

    payload = json.loads(trace.to_json())
    assert trace.outcome == "DECIDE"
    assert trace.source_document == "paper.pdf"
    assert trace.assessment == "The final synthesis confirms a bounded planner contribution."
    assert trace.final_judgment is not None
    assert payload["steps"][0]["results"]
    assert payload["steps"][0]["results"][0]["pages_read"] == [2]
    assert payload["source_document"] == "paper.pdf"
    assert len(llm.calls) == 5
    assert llm.calls[-1]["image_urls"] is None
    assert rendered == [(1, 2), (2,)]

    assert [record.role for record in trace.model_calls] == [
        "master",
        "locator",
        "evidence",
        "master",
        "synthesis",
    ]
    assert [record.call_id for record in trace.model_calls] == [
        "call-001",
        "call-002",
        "call-003",
        "call-004",
        "call-005",
    ]
    assert trace.model_calls[0].round_number == 1
    assert trace.model_calls[1].task_question == "What does the ablation remove?"
    assert trace.model_calls[2].task_question == "What does the ablation remove?"
    assert trace.model_calls[1].image_inputs == ()
    assert [ref.page_number for ref in trace.model_calls[2].image_inputs] == [2]
    assert trace.model_calls[2].image_inputs[0].source_document == "paper.pdf"
    assert trace.model_calls[2].image_inputs[0].dpi == 144
    assert trace.model_calls[0].parsed_response["kind"] == "READ_PAPER"
    assert trace.model_calls[0].json_repaired is False
    trace_json = trace.to_json()
    assert "data:image/png;base64" not in trace_json
    assert "paper.pdf" in trace_json
    master_prompt = json.loads(llm.calls[0]["prompt"])
    assert master_prompt["image_inputs"] == [
        {
            "image_index": 1,
            "source_document": "paper.pdf",
            "page_number": 1,
            "dpi": 144,
        },
        {
            "image_index": 2,
            "source_document": "paper.pdf",
            "page_number": 2,
            "dpi": 144,
        },
    ]


def test_malformed_master_output_is_saved_as_validation_error() -> None:
    pages = make_pages()
    llm = FakeLLM([{"kind": "DECIDE", "tasks": "bad", "assessment": "x"}])
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index=runtime.build_compact_page_index(pages),
        overview_text=pages[0].text,
        overview_images=[],
        recorder=runtime.ModelCallRecorder(model="fake/model"),
    )
    recorder = master.recorder

    trace = run_paper_agent(
        master=master,
        worker=lambda task: WorkerResult(task=task),
        max_rounds=1,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.stop_reason == "master_error@round_1:ValueError: tasks must be a list"
    assert recorder.records[0].validation_error == "tasks must be a list"


def test_kernel_master_validation_is_saved_as_validation_error() -> None:
    pages = make_pages()
    recorder = runtime.ModelCallRecorder(model="fake/model")
    llm = FakeLLM(
        [
            {
                "kind": "READ_PAPER",
                "tasks": [],
                "assessment": None,
                "conclusion_at_risk": "claim",
                "missing_evidence": "evidence",
                "expected_judgment_delta": "delta",
            }
        ]
    )
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index=runtime.build_compact_page_index(pages),
        overview_text=pages[0].text,
        overview_images=[],
        recorder=recorder,
    )

    trace = run_paper_agent(
        master=master,
        worker=lambda task: WorkerResult(task=task),
        max_rounds=1,
    )

    assert trace.outcome == "NEEDS_HUMAN"
    assert trace.stop_reason == "read_paper_without_tasks"
    assert recorder.records[0].validation_error == "read_paper_without_tasks"


def test_invalid_locator_is_saved_as_validation_error_without_evidence_call(
    tmp_path: Path,
) -> None:
    recorder = runtime.ModelCallRecorder(model="fake/model")
    llm = FakeLLM(
        [{"page_ranges": [{"start": 0, "end": 1}], "rationale": "bad range"}]
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    result = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        recorder=recorder,
        render_pages=lambda path, page_numbers: {},
    )(EvidenceTask("Find evidence."))

    assert result.error == "invalid_locator:page range 0-1 is outside 1..3"
    assert len(recorder.records) == 1
    assert recorder.records[0].role == "locator"
    assert recorder.records[0].validation_error == "page range 0-1 is outside 1..3"


def test_invalid_evidence_is_saved_as_validation_error_without_third_call(
    tmp_path: Path,
) -> None:
    recorder = runtime.ModelCallRecorder(model="fake/model")
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "method"},
            {},
        ]
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    pages = make_pages()
    result = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        recorder=recorder,
        render_pages=lambda path, page_numbers: {
            page: f"data:image/png;base64,page-{page}" for page in page_numbers
        },
    )(EvidenceTask("Find evidence."))

    assert result.error == "invalid_evidence_output:finding"
    assert len(recorder.records) == 2
    assert recorder.records[1].role == "evidence"
    assert recorder.records[1].validation_error == "invalid_evidence_output:finding"


def test_two_local_runs_do_not_share_model_call_records(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    pages = make_pages()

    def fake_extract(path: Path) -> list[PaperPage]:
        return pages

    def fake_render(path: Path, page_numbers: list[int]) -> dict[int, str]:
        return {page: f"data:image/png;base64,page-{page}" for page in page_numbers}

    monkeypatch.setattr(runtime, "extract_pdf_pages", fake_extract)
    monkeypatch.setattr(runtime, "render_pdf_pages", fake_render)

    llm = FakeLLM(
        [
            {"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None},
            {"assessment": "final", "key_findings": [{"finding": "f", "evidence": [{"content": pages[0].text, "evidence_type": "text", "locator": "p. 1"}], "caveat": ""}]},
            {"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None},
            {"assessment": "final", "key_findings": [{"finding": "f", "evidence": [{"content": pages[0].text, "evidence_type": "text", "locator": "p. 1"}], "caveat": ""}]},
        ]
    )

    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"
    first_path.write_bytes(b"%PDF fake")
    second_path.write_bytes(b"%PDF fake")

    first = runtime.run_local_paper_agent(pdf_path=first_path, llm=llm, max_rounds=1)
    second = runtime.run_local_paper_agent(pdf_path=second_path, llm=llm, max_rounds=1)

    assert len(first.model_calls) == 2
    assert len(second.model_calls) == 2
    assert first.model_calls[0].call_id == "call-001"
    assert second.model_calls[0].call_id == "call-001"
    assert first.source_document == "first.pdf"
    assert second.source_document == "second.pdf"
    assert len(llm.calls) == 4


def test_single_pass_reads_full_paper_once_and_records_safe_trace(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    pages = make_pages()
    rendered: list[tuple[int, ...]] = []
    _patch_single_pass_io(monkeypatch, pages, rendered)
    llm = FakeLLM([_single_pass_payload()])
    pdf_path = tmp_path / "full-paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    trace = runtime.run_single_pass_paper_judge(pdf_path=pdf_path, llm=llm)

    assert rendered == [(1, 2, 3)]
    assert len(llm.calls) == 1
    prompt = json.loads(llm.calls[0]["prompt"])
    assert [page["page_number"] for page in prompt["full_paper_pages"]] == [1, 2, 3]
    assert [page["text"] for page in prompt["full_paper_pages"]] == [
        page.text for page in pages
    ]
    assert llm.calls[0]["image_urls"] == [
        "data:image/png;base64,page-1",
        "data:image/png;base64,page-2",
        "data:image/png;base64,page-3",
    ]
    assert trace.source_document == "full-paper.pdf"
    assert trace.assessment == "The paper makes a bounded planner contribution."
    assert len(trace.findings) == 1
    assert trace.model_calls[0].role == "single_pass"
    assert trace.model_calls[0].call_id == "call-001"
    assert [ref.page_number for ref in trace.model_calls[0].image_inputs] == [1, 2, 3]
    assert [ref.image_index for ref in trace.model_calls[0].image_inputs] == [1, 2, 3]
    trace_json = trace.to_json()
    payload = json.loads(trace_json)
    assert payload["model_calls"][0]["image_inputs"][2]["page_number"] == 3
    assert "data:image" not in trace_json
    assert str(pdf_path) not in trace_json
    assert "full-paper.pdf" in trace_json


def test_single_pass_prompt_requires_verbatim_text_quotes() -> None:
    prompt = json.loads(
        runtime._build_single_pass_prompt(
            paper_name="paper.pdf",
            pages=make_pages(),
            image_inputs=(),
        )
    )

    assert "verbatim quote" in runtime.SINGLE_PASS_SYSTEM_PROMPT
    assert "do not paraphrase" in runtime.SINGLE_PASS_SYSTEM_PROMPT
    assert "verbatim quote" in prompt["required_json_shape"]["key_findings"][0][
        "evidence"
    ][0]["content"]


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"assessment": "", "key_findings": []},
        {"assessment": "ok"},
        {"assessment": "ok", "key_findings": "bad"},
        {"assessment": "ok", "key_findings": []},
        {
            "assessment": "ok",
            "key_findings": [{"evidence": "x", "caveat": "", "evidence_type": "text", "evidence_locator": "p. 1"}],
        },
        {
            "assessment": "ok",
            "key_findings": [{"finding": "x", "caveat": "", "evidence_type": "text", "evidence_locator": "p. 1"}],
        },
        {
            "assessment": "ok",
            "key_findings": [{"finding": "x", "evidence": "x", "caveat": "", "evidence_type": "audio", "evidence_locator": "p. 1"}],
        },
        {
            "assessment": "ok",
            "key_findings": [{"finding": "x", "evidence": "x", "caveat": "", "evidence_type": "text"}],
        },
        {
            "assessment": "ok",
            "key_findings": [{"finding": "x", "evidence": "x", "caveat": None, "evidence_type": "text", "evidence_locator": "p. 1"}],
        },
    ],
)
def test_single_pass_invalid_output_is_not_retried_and_records_validation_error(
    tmp_path: Path,
    monkeypatch: object,
    payload: object,
) -> None:
    pages = make_pages()
    _patch_single_pass_io(monkeypatch, pages)
    llm = FakeLLM([payload])
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    trace = runtime.run_single_pass_paper_judge(pdf_path=pdf_path, llm=llm)

    assert len(llm.calls) == 1
    assert trace.assessment is None
    assert trace.findings == ()
    assert trace.error is not None
    assert trace.error.startswith("invalid_single_pass_output:")
    assert trace.model_calls[0].validation_error is not None


def test_single_pass_keeps_unmatched_text_evidence_without_automatic_matching(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    pages = make_pages()
    _patch_single_pass_io(monkeypatch, pages)
    payload = {
        "assessment": "The paper makes a bounded contribution.",
        "key_findings": [
            {
                "finding": "A text finding.",
                "evidence": "A paraphrase that is not in the paper.",
                "caveat": "A caveat.",
                "evidence_type": "text",
                "evidence_locator": "paper.pdf, p. 1",
            },
            {
                "finding": "A table finding.",
                "evidence": "The table comparison.",
                "caveat": "",
                "evidence_type": "table",
                "evidence_locator": "Table 2, p. 3",
            },
        ],
    }
    llm = FakeLLM([payload])
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    trace = runtime.run_single_pass_paper_judge(pdf_path=pdf_path, llm=llm)

    assert trace.error is None
    assert trace.assessment == "The paper makes a bounded contribution."
    assert len(trace.findings) == 2
    assert not hasattr(trace.findings[0], "audit_warning")
    assert not hasattr(trace.findings[1], "audit_warning")
    assert trace.model_calls[0].validation_error is None
    assert "audit_warning" not in json.loads(trace.to_json())["findings"][0]


def test_single_pass_accepts_visual_evidence_without_text_match(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    pages = make_pages()
    _patch_single_pass_io(monkeypatch, pages)
    payload = {
        "assessment": "The table supports the comparison.",
        "key_findings": [
            {
                "finding": "The table reports a planner comparison.",
                "evidence": "A visual comparison in Table 2.",
                "caveat": "",
                "evidence_type": "table",
                "evidence_locator": "Table 2, p. 3",
            }
        ],
    }
    llm = FakeLLM([payload])
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    trace = runtime.run_single_pass_paper_judge(pdf_path=pdf_path, llm=llm)

    assert trace.error is None
    assert trace.findings[0].evidence_type == "table"


def test_single_pass_provider_error_keeps_one_record(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    pages = make_pages()
    _patch_single_pass_io(monkeypatch, pages)

    class FailingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete_json(
            self,
            prompt: str,
            *,
            system: str | None = None,
            image_urls: list[str] | None = None,
        ) -> object:
            self.calls += 1
            raise RuntimeError("provider down")

    llm = FailingLLM()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    trace = runtime.run_single_pass_paper_judge(pdf_path=pdf_path, llm=llm)

    assert llm.calls == 1
    assert trace.assessment is None
    assert trace.findings == ()
    assert trace.error == "single_pass_call_error:RuntimeError: provider down"
    assert len(trace.model_calls) == 1
    assert trace.model_calls[0].role == "single_pass"
    assert trace.model_calls[0].error == "RuntimeError: provider down"


def test_single_pass_reuses_llm_without_sharing_recorders(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    pages = make_pages()
    _patch_single_pass_io(monkeypatch, pages)
    llm = FakeLLM([_single_pass_payload(), _single_pass_payload()])
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"
    first_path.write_bytes(b"%PDF fake")
    second_path.write_bytes(b"%PDF fake")

    first = runtime.run_single_pass_paper_judge(pdf_path=first_path, llm=llm)
    second = runtime.run_single_pass_paper_judge(pdf_path=second_path, llm=llm)

    assert len(llm.calls) == 2
    assert [record.call_id for record in first.model_calls] == ["call-001"]
    assert [record.call_id for record in second.model_calls] == ["call-001"]
    assert first.source_document == "first.pdf"
    assert second.source_document == "second.pdf"


def test_text_only_single_pass_returns_the_shared_final_judgment_shape(
    tmp_path: Path, monkeypatch: object
) -> None:
    pages = make_pages()
    _patch_single_pass_io(monkeypatch, pages)
    llm = FakeLLM([
        {
            "assessment": "A bounded contribution.",
            "key_findings": [{
                "finding": "A supported finding.",
                "evidence": [{"content": pages[0].text, "evidence_type": "text", "locator": "p. 1"}],
                "caveat": "A caveat.",
            }],
            "unresolved_questions": [],
            "checklist_coverage": {"core_contribution": "covered"},
        }
    ])
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    trace = runtime.run_single_pass_paper_judge(
        pdf_path=pdf_path, llm=llm, text_only=True
    )

    assert len(llm.calls) == 1
    assert llm.calls[0]["image_urls"] is None
    assert trace.images_sent_to_model == 0
    assert trace.final_judgment is not None
    assert trace.final_judgment.assessment == "A bounded contribution."
    assert trace.final_judgment.key_findings[0].evidence[0].locator == "p. 1"


def test_evidence_worker_keeps_noncanonical_semantic_content() -> None:
    task = EvidenceTask("Which result is decisive?")
    result = runtime._parse_evidence_result(
        task=task,
        payload={
            "finding": {"claim": "A structured finding"},
            "evidence": [{"value": "A structured evidence item"}],
            "caveat": "A caveat.",
            "evidence_type": "table",
            "evidence_locator": "Table 1, p. 1",
        },
        selected_pages=(1,),
        location_rationale="table",
        selected_text={1: "text"},
    )

    assert result.error is None
    assert "structured finding" in result.finding
    assert "structured evidence" in result.evidence
    assert result.noncanonical_output_shape == "finding,evidence"


def test_evidence_worker_parses_zero_to_two_suggested_questions() -> None:
    task = EvidenceTask("Which mechanism is isolated?")
    result = runtime._parse_evidence_result(
        task=task,
        payload={
            "finding": "The comparison changes two components.",
            "evidence": "Table 1",
            "caveat": "The components are not isolated.",
            "evidence_type": "table",
            "evidence_locator": "Table 1, p. 1",
            "suggested_questions": [
                "Does a control change only the access path?",
                "Is the same budget used by both comparators?",
            ],
        },
        selected_pages=(1,),
        location_rationale="comparison",
        selected_text={1: "text"},
    )

    assert result.error is None
    assert result.suggested_questions == (
        "Does a control change only the access path?",
        "Is the same budget used by both comparators?",
    )


@pytest.mark.parametrize("suggested_questions", ["not a list", ["one", 2], ["one", "two", "three"]])
def test_evidence_worker_rejects_structurally_invalid_suggested_questions(
    suggested_questions: object,
) -> None:
    result = runtime._parse_evidence_result(
        task=EvidenceTask("Which mechanism is isolated?"),
        payload={
            "finding": "A finding.",
            "evidence": "Table 1",
            "caveat": "A caveat.",
            "evidence_type": "table",
            "evidence_locator": "Table 1, p. 1",
            "suggested_questions": suggested_questions,
        },
        selected_pages=(1,),
        location_rationale="comparison",
        selected_text={1: "text"},
    )

    assert result.error == "invalid_evidence_output:suggested_questions"


def test_state_payload_includes_action_history_but_not_parse_diagnostics() -> None:
    task = EvidenceTask("Question")
    result = WorkerResult(task=task, finding="Finding", evidence="Evidence", caveat="Caveat", noncanonical_output_shape="evidence", suggested_questions=("Check the missing control.",))
    state = AgentState(
        findings=(result,),
        unresolved_questions=("Open",),
        steps=(TraceStep(1, MasterAction("READ_PAPER", (task,), rationale="Why"), (result,)),),
        remaining_rounds=4,
    )
    payload = runtime._state_payload(state)
    assert payload["history"] == [{
        "round_number": 1,
        "kind": "READ_PAPER",
        "questions": ["Question"],
        "rationale": "Why",
        "assessment": None,
        "conclusion_at_risk": "",
        "missing_evidence": "",
        "expected_judgment_delta": "",
        "unresolved_questions": (),
        "checklist_coverage": None,
    }]
    assert "noncanonical_output_shape" not in json.dumps(payload)
    assert payload["worker_suggested_questions"] == [{
        "origin_question": "Question",
        "questions": ["Check the missing control."],
    }]


def test_master_receives_worker_suggestions_as_candidates_not_evidence() -> None:
    task = EvidenceTask("Inspect the main comparison.")
    result = WorkerResult(
        task=task,
        finding="The reported comparison has a caveat.",
        evidence="Table 1",
        caveat="The budget differs.",
        suggested_questions=("Does a matched-budget control exist?",),
    )
    llm = FakeLLM([{
        "kind": "READ_PAPER",
        "tasks": [
            {"question": "Is there a matched-budget control?", "source_scope": "paper"},
            {"question": "Does the access path change with the comparator?", "source_scope": "paper"},
        ],
        "conclusion_at_risk": "The comparison may be confounded by budget.",
        "missing_evidence": "A matched-budget control.",
        "expected_judgment_delta": "Bound the comparison claim.",
    }])
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview.",
        overview_images=(),
    )

    action = master(AgentState(findings=(result,), remaining_rounds=4))

    prompt = json.loads(llm.calls[0]["prompt"])
    assert prompt["state"]["worker_suggested_questions"] == [{
        "origin_question": "Inspect the main comparison.",
        "questions": ["Does a matched-budget control exist?"],
    }]
    assert len(action.tasks) == 2
    instructions = " ".join(prompt["instructions"])
    assert "candidate questions, not evidence" in instructions


def test_evidence_worker_accepts_findings_list_and_keeps_every_evidence() -> None:
    task = EvidenceTask("Which comparison matters?")
    result = runtime._parse_evidence_result(
        task=task,
        payload={"findings": [{
            "finding": "A finding.",
            "evidence": [
                {"content": "first", "evidence_type": "table", "locator": "Table 1, p. 1"},
                {"content": "second", "evidence_type": "figure", "locator": "Figure 1, p. 2"},
            ],
            "caveat": "A boundary.",
        }]},
        selected_pages=(1, 2),
        location_rationale="comparison",
        selected_text={1: "text", 2: "text"},
    )

    assert result.error is None
    assert "first" in result.evidence and "second" in result.evidence
    assert result.noncanonical_output_shape is None


def test_evidence_worker_keeps_every_item_from_findings_list() -> None:
    task = EvidenceTask("Which comparisons matter?")
    result = runtime._parse_evidence_result(
        task=task,
        payload={"findings": [
            {
                "finding": "First finding.",
                "evidence": [{"content": "first", "evidence_type": "table", "locator": "Table 1, p. 1"}],
                "caveat": "First boundary.",
            },
            {
                "finding": "Second finding.",
                "evidence": [{"content": "second", "evidence_type": "figure", "locator": "Figure 1, p. 2"}],
                "caveat": "Second boundary.",
            },
        ]},
        selected_pages=(1, 2),
        location_rationale="comparison",
        selected_text={1: "text", 2: "text"},
    )

    assert result.error is None
    assert "First finding." in result.finding
    assert "Second finding." in result.finding
    assert "First boundary." in result.caveat
    assert "Second boundary." in result.caveat


def test_evidence_worker_keeps_dict_finding_inside_findings_list() -> None:
    task = EvidenceTask("Which comparison matters?")
    result = runtime._parse_evidence_result(
        task=task,
        payload={"findings": [{
            "finding": {"claim": "Structured claim", "scope": "bounded"},
            "evidence": [{"content": "value", "evidence_type": "table", "locator": "Table 1, p. 1"}],
            "caveat": "Boundary.",
        }]},
        selected_pages=(1,),
        location_rationale="comparison",
        selected_text={1: "text"},
    )

    assert result.error is None
    assert "Structured claim" in result.finding
    assert result.noncanonical_output_shape == "finding"


def test_evidence_worker_keeps_dict_evidence_inside_findings_list() -> None:
    task = EvidenceTask("Which comparison matters?")
    result = runtime._parse_evidence_result(
        task=task,
        payload={"findings": [{
            "finding": "Structured evidence finding.",
            "evidence": {"content": "value", "evidence_type": "table", "locator": "Table 1, p. 1"},
            "caveat": "Boundary.",
        }]},
        selected_pages=(1,),
        location_rationale="comparison",
        selected_text={1: "text"},
    )

    assert result.error is None
    assert "value" in result.evidence
    assert result.noncanonical_output_shape == "evidence"


def test_evidence_prompt_advertises_the_canonical_findings_list_shape() -> None:
    prompt = json.loads(
        runtime._build_evidence_prompt(
            paper_name="paper.pdf",
            question="Which result is decisive?",
            location_rationale="table",
            selected_text={1: "Selected text."},
            image_inputs=(),
        )
    )

    finding = prompt["required_json_shape"]["findings"][0]
    assert isinstance(finding["evidence"], list)
    assert finding["evidence"][0]["locator"]
    assert prompt["required_json_shape"]["suggested_questions"] == [
        "one bounded follow-up question, if needed"
    ]
    assert prompt["mechanism_audit_principles"] == list(runtime.MECHANISM_AUDIT_PRINCIPLES)
    assert "additional local issue" in " ".join(prompt["instructions"])


def test_evidence_prompt_requests_all_distinct_grounded_local_issues_without_forcing_criticism() -> None:
    prompt = json.loads(
        runtime._build_evidence_prompt(
            paper_name="paper.pdf",
            question="Which result matters?",
            location_rationale="evaluation",
            selected_text={1: "Selected text."},
            image_inputs=(),
        )
    )

    instructions = " ".join(prompt["instructions"])
    assert "every distinct supported issue" in instructions
    assert "Do not stop after the first additional issue" in instructions
    assert "discrete boundary" in instructions
    assert "one parameter changes multiple components" in instructions
    assert "activation frequency dilutes an aggregate result" in instructions
    assert "preprocessing, decomposition, or reranking provides an alternative explanation" in instructions
    assert "metrics or reported configurations disagree" in instructions
    assert "Report each supported implication as a separate finding" in instructions
    assert "If none is supported" in instructions
    assert "Do not invent criticism" in instructions


def test_evidence_prompt_requires_structured_visual_observation_and_context_verification() -> None:
    prompt = json.loads(
        runtime._build_evidence_prompt(
            paper_name="paper.pdf",
            question="What does the ablation show?",
            location_rationale="table",
            selected_text={2: "Selected text."},
            image_inputs=(),
            research_context=(
                ResearchContext(
                    finding_id="r1-t1-f1",
                    question="Earlier question",
                    finding="Earlier finding",
                    evidence="Earlier evidence",
                    caveat="Earlier caveat",
                    evidence_type="table",
                    evidence_locator="Table 1, p. 1",
                    decision_relevance="Check the relationship.",
                ),
            ),
        )
    )

    instructions = " ".join(prompt["instructions"])
    assert "reported by another Worker" in instructions
    assert "independently verify" in instructions
    assert "row, column, metric, comparison, and configuration" in instructions
    assert "directly observed" in instructions
    assert "inference" in instructions


def test_final_judgment_keeps_multiple_evidence_items(tmp_path: Path, monkeypatch: object) -> None:
    pages = make_pages()
    _patch_single_pass_io(monkeypatch, pages)
    llm = FakeLLM([{"assessment": "Assessment.", "key_findings": [{"finding": "Finding.", "evidence": [{"content": pages[0].text, "evidence_type": "text", "locator": "p. 1"}, {"content": "visual", "evidence_type": "table", "locator": "Table 1, p. 2"}], "caveat": "Boundary."}]}])
    path = tmp_path / "paper.pdf"; path.write_bytes(b"%PDF fake")
    trace = runtime.run_single_pass_paper_judge(pdf_path=path, llm=llm, text_only=True)
    assert trace.final_judgment is not None
    assert len(trace.final_judgment.key_findings[0].evidence) == 2


def test_single_pass_and_synthesis_share_the_decision_checklist() -> None:
    single = json.loads(runtime._build_single_pass_prompt(paper_name="paper.pdf", pages=make_pages(), image_inputs=()))
    trace = AgentTrace(steps=(), outcome="NEEDS_HUMAN", assessment=None, stop_reason="stop")
    synthesis = json.loads(runtime._build_synthesis_prompt("paper.pdf", make_pages(), "[]", trace))
    assert single["decision_checklist"] == synthesis["decision_checklist"]
    assert single["required_json_shape"]["checklist_coverage"]["claim_boundary"] == "covered|unresolved|not_applicable"
    assert single["required_json_shape"] == synthesis["required_json_shape"]
    assert isinstance(single["required_json_shape"]["key_findings"][0]["evidence"], list)
    assert single["mechanism_audit_principles"] == list(runtime.MECHANISM_AUDIT_PRINCIPLES)
    assert synthesis["mechanism_audit_principles"] == list(runtime.MECHANISM_AUDIT_PRINCIPLES)


def test_text_only_single_pass_prompt_has_compact_index_and_no_image_references(
    tmp_path: Path, monkeypatch: object
) -> None:
    pages = make_pages()
    _patch_single_pass_io(monkeypatch, pages)
    llm = FakeLLM([_single_pass_payload()])
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF fake")

    runtime.run_single_pass_paper_judge(pdf_path=path, llm=llm, text_only=True)

    prompt = json.loads(llm.calls[0]["prompt"])
    assert prompt["image_inputs"] == []
    assert prompt["compact_page_index"]


def test_synthesis_prompt_keeps_complete_action_history() -> None:
    first = EvidenceTask("First question")
    second = EvidenceTask("Second question")
    result = WorkerResult(task=first, finding="Finding", evidence="Evidence", caveat="Caveat")
    trace = AgentTrace(
        steps=(
            TraceStep(
                1,
                MasterAction(
                    "READ_PAPER",
                    (first,),
                    unresolved_questions=("first unresolved",),
                    assessment="First provisional.",
                    rationale="Need first evidence.",
                    checklist_coverage={"core_contribution": "covered"},
                ),
                (result,),
            ),
            TraceStep(
                2,
                MasterAction(
                    "NEEDS_HUMAN",
                    unresolved_questions=("second unresolved",),
                    rationale="Need review.",
                    checklist_coverage={"main_evidence": "unresolved"},
                ),
            ),
        ),
        outcome="NEEDS_HUMAN",
        assessment=None,
        stop_reason="need review",
    )

    prompt = json.loads(runtime._build_synthesis_prompt("paper.pdf", make_pages(), "[]", trace))

    assert prompt["history"]["history"][0]["assessment"] == "First provisional."
    assert prompt["history"]["history"][0]["unresolved_questions"] == ["first unresolved"]
    assert prompt["history"]["history"][0]["checklist_coverage"] == {"core_contribution": "covered"}
    assert prompt["history"]["cumulative_unresolved_questions"] == ["first unresolved", "second unresolved"]
    assert prompt["history"]["checklist_coverage"]["core_contribution"] == "covered"
    assert prompt["history"]["findings"][0]["caveat"] == "Caveat"
    assert "Preserve material caveats and unresolved questions" in " ".join(
        prompt["instructions"]
    )
    assert prompt["mechanism_audit_principles"] == list(runtime.MECHANISM_AUDIT_PRINCIPLES)


def test_synthesis_prompt_requests_a_comprehensive_judgment_without_silent_omission() -> None:
    prompt = json.loads(
        runtime._build_synthesis_prompt(
            "paper.pdf",
            make_pages(),
            "[]",
            AgentTrace(
                steps=(),
                outcome="NEEDS_HUMAN",
                assessment=None,
                stop_reason="stop",
            ),
        )
    )

    instructions = " ".join(prompt["instructions"])
    assert "comprehensive paper judgment, not a short report or abstract" in instructions
    assert "Use as many key_findings as needed" in instructions
    assert "Do not silently omit" in instructions
    assert "negative result" in instructions
    assert "metric disagreement" in instructions
    assert "configuration inconsistency" in instructions
    assert "ablation confound" in instructions
    assert "alternative explanation" in instructions
    assert "missing control" in instructions
    assert "cost, scaling, or stability limitation" in instructions
    assert "does not change the overall assessment" in instructions
    assert "explicitly retain every distinct issue" in instructions
    assert "checklist item is covered" in instructions
    assert "independent full-paper gap check" in instructions
    assert "account for every distinct supported issue" in instructions
    assert "retained, explicitly merged, corrected, or left unresolved" in instructions
    assert "Do not claim independent visual verification" in instructions


def test_stability_audit_directions_are_explicit_and_shared() -> None:
    trace = AgentTrace(
        steps=(),
        outcome="NEEDS_HUMAN",
        assessment=None,
        stop_reason="stop",
    )
    synthesis = json.loads(
        runtime._build_synthesis_prompt("paper.pdf", make_pages(), "[]", trace)
    )
    single = json.loads(
        runtime._build_single_pass_prompt(
            paper_name="paper.pdf",
            pages=make_pages(),
            image_inputs=(),
        )
    )

    expected_keys = {
        "auxiliary_model_reliability",
        "generative_transformation_fidelity",
        "matched_resource_efficiency",
        "claim_result_consistency",
    }
    assert expected_keys <= set(runtime.DECISION_CHECKLIST)
    assert synthesis["decision_checklist"] == single["decision_checklist"]

    principles = " ".join(synthesis["mechanism_audit_principles"])
    assert "validator or judge" in principles
    assert "self-confirmation" in principles
    assert "Do not infer one auxiliary role's model identity" in principles
    assert "another role" in principles
    assert "summarization, rewrite, compression" in principles
    assert "fidelity" in principles
    assert "total model calls, tokens, latency, or cost" in principles
    assert "headline prose" in principles
    assert "table values" in principles

    instructions = " ".join(synthesis["instructions"])
    assert "Every applicable checklist item marked covered" in instructions
    assert "assessment or an explicit key finding" in instructions
    assert "mark it unresolved" in instructions
    assert "Checklist coverage records whether the audit was completed" in instructions
    assert "confirmed limitation or inconsistency counts as covered" in instructions


def test_single_pass_uses_the_same_comprehensive_audit_standard_as_synthesis() -> None:
    prompt = json.loads(
        runtime._build_single_pass_prompt(
            paper_name="paper.pdf",
            pages=make_pages(),
            image_inputs=(),
        )
    )

    instructions = " ".join(prompt["instructions"])
    assert "comprehensive paper judgment, not a short report or abstract" in instructions
    assert "Use as many key_findings as needed" in instructions
    assert "negative result" in instructions
    assert "metric disagreement" in instructions
    assert "configuration inconsistency" in instructions
    assert "ablation confound" in instructions
    assert "alternative explanation" in instructions
    assert "missing control" in instructions
    assert "cost, scaling, or stability limitation" in instructions
    assert "text-only input" in instructions
    assert "method's computational, storage, latency, and deployment costs" in instructions
    assert "adoption cost" not in runtime.SINGLE_PASS_SYSTEM_PROMPT


def test_master_and_single_pass_prompts_include_mechanism_audit_guidance() -> None:
    master = runtime.PaperAgentMaster(
        llm=FakeLLM([{"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None}]),
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview.",
        overview_images=(),
    )
    master(AgentState(remaining_rounds=5))
    master_prompt = json.loads(master.llm.calls[0]["prompt"])
    single_prompt = json.loads(runtime._build_single_pass_prompt(
        paper_name="paper.pdf", pages=make_pages(), image_inputs=()
    ))

    assert master_prompt["mechanism_audit_principles"] == list(runtime.MECHANISM_AUDIT_PRINCIPLES)
    assert single_prompt["mechanism_audit_principles"] == list(runtime.MECHANISM_AUDIT_PRINCIPLES)


def _incremental_master_context_state() -> AgentState:
    first_task = EvidenceTask("Exact first task question")
    failed_task = EvidenceTask("Exact failed task question")
    second_task = EvidenceTask("Exact latest task question")
    first_result = WorkerResult(
        task=first_task,
        finding="FIRST_FINDING_VERBATIM",
        evidence="RAW_EVIDENCE_FIRST_DO_NOT_SEND_TO_MASTER",
        caveat="FIRST_CAVEAT_VERBATIM",
        pages_read=(2, 3),
        evidence_type="text",
        evidence_locator="paper.pdf, p. 2",
        suggested_questions=("OLD_SUGGESTED_QUESTION",),
        structured_findings=(WorkerFinding(
            "FIRST_FINDING_VERBATIM",
            (FindingEvidence("EVIDENCE_ITEM_CONTENT_FIRST_DO_NOT_SEND", "text", "paper.pdf, p. 2"),),
            "FIRST_CAVEAT_VERBATIM",
        ),),
    )
    failed_result = WorkerResult(
        task=failed_task,
        error="WORKER_FAILURE_VERBATIM",
        pages_read=(4,),
        location_rationale="failed locator",
    )
    latest_result = WorkerResult(
        task=second_task,
        finding="LATEST_FINDING_VERBATIM",
        evidence="RAW_EVIDENCE_LATEST_DO_NOT_SEND_TO_MASTER",
        caveat="LATEST_CAVEAT_VERBATIM",
        pages_read=(5,),
        evidence_type="table",
        evidence_locator="Table 2, p. 5",
        suggested_questions=("LATEST_SUGGESTED_QUESTION",),
        structured_findings=(WorkerFinding(
            "LATEST_FINDING_VERBATIM",
            (FindingEvidence("EVIDENCE_ITEM_CONTENT_LATEST_DO_NOT_SEND", "table", "Table 2, p. 5"),),
            "LATEST_CAVEAT_VERBATIM",
        ),),
    )
    first_action = MasterAction(
        "READ_PAPER",
        (first_task, failed_task),
        assessment="OLD_ACTION_ASSESSMENT_DO_NOT_SEND",
        rationale="OLD_ACTION_RATIONALE_DO_NOT_SEND",
        conclusion_at_risk="OLD_CONCLUSION_AT_RISK_DO_NOT_SEND",
        missing_evidence="OLD_MISSING_EVIDENCE_DO_NOT_SEND",
        expected_judgment_delta="OLD_DELTA_DO_NOT_SEND",
    )
    latest_action = MasterAction(
        "READ_PAPER",
        (second_task,),
        assessment="LATEST_ACTION_ASSESSMENT_DO_NOT_SEND",
        rationale="LATEST_ACTION_RATIONALE_DO_NOT_SEND",
        conclusion_at_risk="LATEST_CONCLUSION_AT_RISK",
        missing_evidence="LATEST_MISSING_EVIDENCE",
        expected_judgment_delta="LATEST_EXPECTED_DELTA",
        unresolved_questions=("LATEST_ACTION_UNRESOLVED",),
        checklist_coverage={"core_contribution": "covered"},
    )
    return AgentState(
        findings=(first_result, failed_result, latest_result),
        provisional_assessment="CURRENT_ASSESSMENT_VERBATIM",
        unresolved_questions=("CURRENT_UNRESOLVED_VERBATIM",),
        reflection_reports=(
            ReflectionReport("post_method_model", ("r1-t1-f1",), "OLD_REFLECTION_MEMO_DO_NOT_SEND"),
            ReflectionReport("pre_decide", ("r2-t1-f1",), "LATEST_REFLECTION_MEMO_VERBATIM"),
        ),
        steps=(
            TraceStep(1, first_action, (first_result, failed_result)),
            TraceStep(2, latest_action, (latest_result,)),
        ),
        remaining_rounds=3,
    )


def test_incremental_master_context_keeps_only_the_later_round_projection() -> None:
    state = _incremental_master_context_state()
    response = {"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None}
    legacy_llm = FakeLLM([response])
    compact_llm = FakeLLM([response])
    legacy = runtime.PaperAgentMaster(
        llm=legacy_llm,
        paper_name="paper.pdf",
        page_index="[{\"page_number\": 2}]",
        overview_text="OVERVIEW_TEXT_MUST_REMAIN_IN_LEGACY",
        overview_images=("data:image/png;base64,overview",),
    )
    compact = runtime.PaperAgentMaster(
        llm=compact_llm,
        paper_name="paper.pdf",
        page_index="[{\"page_number\": 2}]",
        overview_text="OVERVIEW_TEXT_MUST_NOT_REPEAT",
        overview_images=("data:image/png;base64,overview",),
        master_context_mode="incremental-no-raw-evidence",
    )

    legacy(state)
    compact(state)

    legacy_prompt = json.loads(legacy_llm.calls[0]["prompt"])
    prompt = json.loads(compact_llm.calls[0]["prompt"])
    serialized = json.dumps(prompt)
    assert legacy_prompt["overview_pages"] == "OVERVIEW_TEXT_MUST_REMAIN_IN_LEGACY"
    assert legacy_llm.calls[0]["image_urls"] == ["data:image/png;base64,overview"]
    assert "overview_pages" not in prompt
    assert "image_inputs" not in prompt
    assert compact_llm.calls[0]["image_urls"] is None
    assert prompt["compact_page_index"] == [{"page_number": 2}]
    assert prompt["state"]["completed_tasks"] == [
        {
            "round_number": 1,
            "task_number": 1,
            "question": "Exact first task question",
            "pages_read": [2, 3],
            "status": "completed",
            "finding_ids": ["r1-t1-f1"],
        },
        {
            "round_number": 1,
            "task_number": 2,
            "question": "Exact failed task question",
            "pages_read": [4],
            "status": "failed",
            "finding_ids": [],
        },
        {
            "round_number": 2,
            "task_number": 1,
            "question": "Exact latest task question",
            "pages_read": [5],
            "status": "completed",
            "finding_ids": ["r2-t1-f1"],
        },
    ]
    assert "FIRST_FINDING_VERBATIM" in serialized
    assert "FIRST_CAVEAT_VERBATIM" in serialized
    assert "LATEST_FINDING_VERBATIM" in serialized
    assert "LATEST_CAVEAT_VERBATIM" in serialized
    assert "RAW_EVIDENCE_FIRST_DO_NOT_SEND_TO_MASTER" not in serialized
    assert "RAW_EVIDENCE_LATEST_DO_NOT_SEND_TO_MASTER" not in serialized
    assert "EVIDENCE_ITEM_CONTENT_FIRST_DO_NOT_SEND" not in serialized
    assert "EVIDENCE_ITEM_CONTENT_LATEST_DO_NOT_SEND" not in serialized
    assert prompt["state"]["latest_reflection_report"]["reflection_memo"] == "LATEST_REFLECTION_MEMO_VERBATIM"
    assert "OLD_REFLECTION_MEMO_DO_NOT_SEND" not in serialized
    assert prompt["state"]["provisional_assessment"] == "CURRENT_ASSESSMENT_VERBATIM"
    assert prompt["state"]["unresolved_questions"] == ["CURRENT_UNRESOLVED_VERBATIM"]
    assert prompt["state"]["checklist_coverage"]["core_contribution"] == "covered"
    assert prompt["state"]["worker_failures"][0]["error"] == "WORKER_FAILURE_VERBATIM"
    assert prompt["state"]["remaining_rounds"] == 3
    assert prompt["state"]["latest_action"]["conclusion_at_risk"] == "LATEST_CONCLUSION_AT_RISK"
    assert prompt["state"]["latest_action"]["missing_evidence"] == "LATEST_MISSING_EVIDENCE"
    assert prompt["state"]["latest_action"]["expected_judgment_delta"] == "LATEST_EXPECTED_DELTA"
    assert "OLD_ACTION_RATIONALE_DO_NOT_SEND" not in serialized
    assert "OLD_ACTION_ASSESSMENT_DO_NOT_SEND" not in serialized
    assert "OLD_CONCLUSION_AT_RISK_DO_NOT_SEND" not in serialized
    assert "LATEST_ACTION_ASSESSMENT_DO_NOT_SEND" not in serialized
    assert "LATEST_SUGGESTED_QUESTION" in serialized
    assert "OLD_SUGGESTED_QUESTION" not in serialized
    full_state = runtime._state_payload(state)
    assert "EVIDENCE_ITEM_CONTENT_FIRST_DO_NOT_SEND" in json.dumps(full_state)
    assert "EVIDENCE_ITEM_CONTENT_FIRST_DO_NOT_SEND" in json.dumps(full_state)
    assert "OLD_REFLECTION_MEMO_DO_NOT_SEND" in json.dumps(full_state)


def test_incremental_master_context_preserves_the_first_call_and_full_reflector_synthesis_state() -> None:
    state = _incremental_master_context_state()
    response = {"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None}
    llm = FakeLLM([response, response])
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="FIRST_CALL_OVERVIEW",
        overview_images=("data:image/png;base64,overview",),
        master_context_mode="incremental-no-raw-evidence",
    )

    master(AgentState(remaining_rounds=5))
    first_prompt = json.loads(llm.calls[0]["prompt"])
    assert first_prompt["overview_pages"] == "FIRST_CALL_OVERVIEW"
    assert first_prompt["image_inputs"]
    assert llm.calls[0]["image_urls"] == ["data:image/png;base64,overview"]

    reflector_llm = FakeLLM([{"reflection_memo": "Reflection response."}])
    reflector = runtime.PaperReflector(
        llm=reflector_llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview.",
    )
    reflector(state, trigger="post_method_model", finding_ids=("r2-t1-f1",))
    reflector_prompt = json.loads(reflector_llm.calls[0]["prompt"])
    assert "EVIDENCE_ITEM_CONTENT_FIRST_DO_NOT_SEND" in json.dumps(reflector_prompt["state"])
    assert "OLD_REFLECTION_MEMO_DO_NOT_SEND" in json.dumps(reflector_prompt["state"])
    trace = AgentTrace(
        steps=state.steps,
        outcome="DECIDE",
        assessment=None,
        stop_reason="stop",
        reflection_reports=state.reflection_reports,
    )
    synthesis_prompt = json.loads(runtime._build_synthesis_prompt("paper.pdf", make_pages(), "[]", trace))
    assert "EVIDENCE_ITEM_CONTENT_FIRST_DO_NOT_SEND" in json.dumps(synthesis_prompt["history"])
    assert "OLD_REFLECTION_MEMO_DO_NOT_SEND" in json.dumps(synthesis_prompt["history"])


def _context_ownership_pages() -> list[PaperPage]:
    return [
        PaperPage(1, "Abstract\nMAIN_BODY_MARKER", 25, False, False),
        PaperPage(2, "2 Method\nMETHOD_BODY_MARKER", 28, False, False),
        PaperPage(3, "References\nREFERENCE_MARKER", 27, False, False),
        PaperPage(4, "Appendix\nAPPENDIX_MARKER", 24, False, False),
    ]


def test_context_ownership_first_master_gets_main_text_and_default_keeps_overview(
    tmp_path: Path, monkeypatch: object
) -> None:
    pages = _context_ownership_pages()
    _patch_single_pass_io(monkeypatch, pages)
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF fake")
    profile_llm = FakeLLM([
        {"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None},
        _single_pass_payload(),
    ])

    trace = runtime.run_local_paper_agent(
        pdf_path=path,
        llm=profile_llm,
        max_rounds=1,
        paper_context_mode="master-main-text-history-only",
    )

    prompt = json.loads(profile_llm.calls[0]["prompt"])
    assert "MAIN_BODY_MARKER" in prompt["main_paper_text"]
    assert "METHOD_BODY_MARKER" in prompt["main_paper_text"]
    assert "REFERENCE_MARKER" not in prompt["main_paper_text"]
    assert "APPENDIX_MARKER" not in prompt["main_paper_text"]
    assert "overview_pages" not in prompt
    assert prompt["compact_page_index"]
    assert [ref.page_number for ref in trace.model_calls[0].image_inputs] == [1, 2]

    default_llm = FakeLLM([{"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None}])
    master = runtime.PaperAgentMaster(
        llm=default_llm,
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="DEFAULT_OVERVIEW_MARKER",
        overview_images=("data:image/png;base64,overview",),
    )
    master(AgentState())
    default_prompt = json.loads(default_llm.calls[0]["prompt"])
    assert default_prompt["overview_pages"] == "DEFAULT_OVERVIEW_MARKER"
    assert default_prompt["image_inputs"]
    assert default_llm.calls[0]["image_urls"] == ["data:image/png;base64,overview"]


def test_context_ownership_later_master_keeps_incremental_state_without_paper_payload() -> None:
    state = _incremental_master_context_state()
    llm = FakeLLM([{"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None}])
    master = runtime.PaperAgentMaster(
        llm=llm,
        paper_name="paper.pdf",
        page_index='[{"page_number": 2, "summary": "index"}]',
        overview_text="OVERVIEW_NOT_SENT",
        overview_images=("data:image/png;base64,overview",),
        paper_context_mode="master-main-text-history-only",
        main_paper_text="MAIN_TEXT_NOT_REPEATED",
    )

    master(state)

    prompt = json.loads(llm.calls[0]["prompt"])
    assert prompt["state"]["latest_reflection_report"]["reflection_memo"] == "LATEST_REFLECTION_MEMO_VERBATIM"
    assert "EVIDENCE_ITEM_CONTENT_FIRST_DO_NOT_SEND" not in json.dumps(prompt)
    assert "EVIDENCE_ITEM_CONTENT_LATEST_DO_NOT_SEND" not in json.dumps(prompt)
    for key in ("main_paper_text", "overview_pages", "image_inputs"):
        assert key not in prompt
    assert prompt["compact_page_index"]
    assert llm.calls[0]["image_urls"] is None


def test_context_ownership_synthesis_keeps_history_without_paper_or_index() -> None:
    state = _incremental_master_context_state()
    trace = AgentTrace(
        steps=state.steps,
        outcome="DECIDE",
        assessment=None,
        stop_reason="stop",
        reflection_reports=state.reflection_reports,
    )

    prompt = json.loads(
        runtime._build_synthesis_prompt(
            "paper.pdf",
            _context_ownership_pages(),
            '[{"page_number": 1}]',
            trace,
            paper_context_mode="master-main-text-history-only",
        )
    )

    serialized_history = json.dumps(prompt["history"])
    assert "EVIDENCE_ITEM_CONTENT_FIRST_DO_NOT_SEND" in serialized_history
    assert "full_paper_pages" not in prompt
    assert "compact_page_index" not in prompt
    instructions = " ".join(prompt["instructions"])
    assert "Do not perform an independent paper-reading pass" in instructions
    assert "independent full-paper gap check" not in instructions
    assert "Use the full paper to check Worker claims" not in instructions
    assert "proposer/evolver" in instructions
    assert "verifier/judge" in instructions
    assert "unresolved consequential difference as a confound" in instructions


def test_master_checklist_updates_do_not_erase_prior_coverage() -> None:
    first = runtime.parse_master_action(
        {
            "kind": "READ_PAPER",
            "tasks": [{"question": "Question", "source_scope": "paper"}],
            "checklist_coverage": {"core_contribution": "covered"},
            "conclusion_at_risk": "The contribution may be overstated.",
            "missing_evidence": "A bounded comparison.",
            "expected_judgment_delta": "Narrow the contribution claim.",
        }
    )
    second = runtime.parse_master_action(
        {"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None}
    )
    state = AgentState(
        steps=(TraceStep(1, first), TraceStep(2, second)),
        remaining_rounds=3,
    )

    payload = runtime._state_payload(state)

    assert payload["checklist_coverage"]["core_contribution"] == "covered"
    assert second.checklist_coverage == {}


def test_master_prompt_uses_the_shared_decision_checklist() -> None:
    master = runtime.PaperAgentMaster(
        llm=FakeLLM([{"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None}]),
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview.",
        overview_images=(),
    )

    master(AgentState(remaining_rounds=5))

    prompt = json.loads(master.llm.calls[0]["prompt"])
    assert prompt["decision_checklist"] == list(runtime.DECISION_CHECKLIST)


def test_master_prompt_separates_coverage_from_sufficiency_and_preserves_open_questions() -> None:
    master = runtime.PaperAgentMaster(
        llm=FakeLLM([{"kind": "NEEDS_HUMAN"}]),
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="Overview.",
        overview_images=(),
        worker_role_mode="discovery-cross-check",
    )

    master(AgentState(remaining_rounds=5))

    prompt = json.loads(master.llm.calls[0]["prompt"])
    instructions = " ".join(prompt["instructions"])
    principles = " ".join(prompt["mechanism_audit_principles"])
    assert "Count an issue as covered when it is mentioned" not in principles
    assert "Checklist coverage records what has been examined; it is not by itself evidence that the paper judgment is complete" in instructions
    assert "qualify or bound the assessment without reversing it" in instructions
    assert "Worker suggested questions and unreviewed paper directions" in instructions
    assert "distinct unreviewed evidence-bearing direction remains" in instructions
    assert "Budget remaining is not evidence value" in instructions
    assert "DECIDE may retain honest unresolved_questions" in instructions
    assert prompt["output_contract"]["READ_PAPER"]["assessment"] != None
    assert prompt["output_contract"]["DECIDE"]["unresolved_questions"]


def test_locator_prompt_is_navigation_only_and_treats_index_as_incomplete() -> None:
    prompt = json.loads(
        runtime._build_locator_prompt(
            question="Where is the relevant comparison?",
            page_index="[]",
            page_count=20,
            research_context=(
                ResearchContext(
                    finding_id="r1-t1-f1",
                    question="Earlier question",
                    finding="Earlier finding",
                    evidence="Earlier evidence",
                    caveat="Earlier caveat",
                    evidence_type="text",
                    evidence_locator="p. 1",
                    decision_relevance="Check the relation.",
                ),
            ),
        )
    )

    instructions = " ".join(prompt["instructions"])
    assert "incomplete navigation aid" in instructions
    assert "absence from the preview is not evidence of absence" in instructions
    assert "smallest sufficient page set" in instructions
    assert "confirm, qualify, contradict, or independently test" in instructions
    assert "Act as a cross-finding reviewer" not in instructions


def test_local_agent_runs_one_final_synthesis_after_needs_human(
    tmp_path: Path, monkeypatch: object
) -> None:
    pages = make_pages()
    monkeypatch.setattr(runtime, "extract_pdf_pages", lambda path: pages)
    monkeypatch.setattr(runtime, "render_pdf_pages", lambda path, numbers: {n: "data:image/png;base64,x" for n in numbers})
    llm = FakeLLM([
        {"kind": "NEEDS_HUMAN", "tasks": [], "assessment": None},
        {
            "assessment": "Final assessment.",
            "key_findings": [{"finding": "Final finding.", "evidence": [{"content": pages[0].text, "evidence_type": "text", "locator": "p. 1"}], "caveat": "Boundary."}],
            "unresolved_questions": ["Open question"],
            "checklist_coverage": {"core_contribution": "covered"},
        },
    ])
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF fake")

    trace = runtime.run_local_paper_agent(pdf_path=path, llm=llm, max_rounds=5)

    assert [call["image_urls"] for call in llm.calls] == [["data:image/png;base64,x", "data:image/png;base64,x"], None]
    assert trace.final_judgment is not None
    assert trace.assessment == "Final assessment."
    assert trace.model_calls[-1].role == "synthesis"


@pytest.mark.parametrize("first_action", [
    {
        "kind": "DECIDE",
        "tasks": [],
        "assessment": "Loop assessment.",
        "stop_reason_code": "evidence_sufficient",
    },
    {
        "kind": "READ_PAPER",
        "tasks": [{"question": "Question", "source_scope": "paper"}],
        "conclusion_at_risk": "The claim may change.",
        "missing_evidence": "A bounded paper fact.",
        "expected_judgment_delta": "Narrow the claim.",
    },
])
def test_local_agent_runs_synthesis_after_decide_or_budget(
    tmp_path: Path, monkeypatch: object, first_action: dict[str, object]
) -> None:
    pages = make_pages()
    monkeypatch.setattr(runtime, "extract_pdf_pages", lambda path: pages)
    monkeypatch.setattr(runtime, "render_pdf_pages", lambda path, numbers: {n: "data:image/png;base64,x" for n in numbers})
    responses: list[object] = [first_action]
    if first_action["kind"] == "READ_PAPER":
        responses += [{"page_ranges": [{"start": 1, "end": 1}], "rationale": "r"}, {"finding": "f", "evidence": pages[0].text, "caveat": "", "evidence_type": "text", "evidence_locator": "p. 1"}]
    responses += [{"assessment": "Final.", "key_findings": [{"finding": "f", "evidence": [{"content": pages[0].text, "evidence_type": "text", "locator": "p. 1"}], "caveat": ""}]}]
    llm = FakeLLM(responses); path = tmp_path / "paper.pdf"; path.write_bytes(b"%PDF fake")
    trace = runtime.run_local_paper_agent(pdf_path=path, llm=llm, max_rounds=1)
    assert [record.role for record in trace.model_calls].count("synthesis") == 1
    assert trace.final_judgment is not None


def test_failed_final_synthesis_does_not_leave_a_nonuniform_loop_assessment(
    tmp_path: Path, monkeypatch: object
) -> None:
    pages = make_pages()
    monkeypatch.setattr(runtime, "extract_pdf_pages", lambda path: pages)
    monkeypatch.setattr(runtime, "render_pdf_pages", lambda path, numbers: {n: "data:image/png;base64,x" for n in numbers})
    llm = FakeLLM([
        {
            "kind": "DECIDE",
            "tasks": [],
            "assessment": "Loop-only assessment.",
            "stop_reason_code": "evidence_sufficient",
        },
        {"assessment": "", "key_findings": []},
    ])
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"%PDF fake")

    trace = runtime.run_local_paper_agent(pdf_path=path, llm=llm, max_rounds=5)

    assert trace.final_judgment is None
    assert trace.assessment is None
    assert trace.model_calls[-1].role == "synthesis"
    assert trace.model_calls[-1].validation_error is not None


def test_master_parses_optional_context_selection_fields_strictly() -> None:
    action = runtime.parse_master_action(
        {
            "kind": "READ_PAPER",
            "tasks": [
                {
                    "question": "Cross-check the two reported results.",
                    "source_scope": "paper",
                    "related_finding_ids": ["r1-t1-f1"],
                    "decision_relevance": "The relation could change the conclusion.",
                }
            ],
            "conclusion_at_risk": "The relation may change the conclusion.",
            "missing_evidence": "A cross-check of the reported results.",
            "expected_judgment_delta": "Bound the conclusion.",
        }
    )

    assert action.tasks[0].related_finding_ids == ("r1-t1-f1",)
    assert action.tasks[0].decision_relevance == "The relation could change the conclusion."
    with pytest.raises(ValueError, match="related_finding_ids"):
        runtime.parse_master_action(
            {
                "kind": "READ_PAPER",
                "tasks": [
                    {"question": "q", "source_scope": "paper", "related_finding_ids": "r1"}
                ],
            }
        )


def test_selected_context_reaches_locator_and_evidence_without_full_master_state(
    tmp_path: Path,
) -> None:
    pages = make_pages()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "check relation"},
            {
                "finding": "Current answer.",
                "evidence": "Selected evidence: the ablation removes the planner.",
                "caveat": "bounded",
                "evidence_type": "text",
                "evidence_locator": "paper.pdf, p. 2",
            },
        ]
    )
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda _path, numbers: {
            page: f"data:image/png;base64,page-{page}" for page in numbers
        },
        worker_context_mode="selected-context",
    )
    context = ResearchContext(
        finding_id="r1-t1-f1",
        question="Earlier question",
        finding="Earlier finding",
        evidence="Earlier evidence",
        caveat="Earlier caveat",
        evidence_type="table",
        evidence_locator="Table 1, p. 1",
        decision_relevance="Check whether the results conflict.",
    )

    worker.set_research_context((context,), ())
    result = worker(EvidenceTask("Current bounded question"))

    locator_prompt = json.loads(llm.calls[0]["prompt"])
    evidence_prompt = json.loads(llm.calls[1]["prompt"])
    expected_context = {
        "finding_id": "r1-t1-f1",
        "question": "Earlier question",
        "finding": "Earlier finding",
        "evidence": "Earlier evidence",
        "caveat": "Earlier caveat",
        "evidence_type": "table",
        "evidence_locator": "Table 1, p. 1",
        "decision_relevance": "Check whether the results conflict.",
    }
    assert locator_prompt["research_context"] == [expected_context]
    assert evidence_prompt["research_context"] == [expected_context]
    assert "history" not in locator_prompt
    assert "checklist_coverage" not in evidence_prompt
    context_instructions = " ".join(evidence_prompt["instructions"])
    assert "reported by another Worker" in context_instructions
    assert "independently verify" in context_instructions
    assert result.error is None


def test_no_context_worker_omits_research_context_from_both_prompts(tmp_path: Path) -> None:
    pages = make_pages()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "local"},
            {
                "finding": "Current answer.",
                "evidence": "Selected evidence: the ablation removes the planner.",
                "caveat": "",
                "evidence_type": "text",
                "evidence_locator": "paper.pdf, p. 2",
            },
        ]
    )
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda _path, numbers: {
            page: f"data:image/png;base64,page-{page}" for page in numbers
        },
    )

    worker(EvidenceTask("Current bounded question"))

    assert "research_context" not in json.loads(llm.calls[0]["prompt"])
    assert "research_context" not in json.loads(llm.calls[1]["prompt"])


def test_worker_receives_task_decision_context_without_selected_findings(tmp_path: Path) -> None:
    pages = make_pages()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "check mechanism"},
            {
                "finding": "Current answer.",
                "evidence": "Selected evidence: the ablation removes the planner.",
                "caveat": "bounded",
                "evidence_type": "text",
                "evidence_locator": "paper.pdf, p. 2",
            },
        ]
    )
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda _path, numbers: {
            page: f"data:image/png;base64,page-{page}" for page in numbers
        },
    )
    task = EvidenceTask(
        "Does the reported comparison isolate the mechanism?",
        decision_relevance=(
            "A coupled configuration may explain both observations, which would bound "
            "the mechanism claim."
        ),
    )

    result = worker(task)

    locator_prompt = json.loads(llm.calls[0]["prompt"])
    evidence_prompt = json.loads(llm.calls[1]["prompt"])
    for prompt in (locator_prompt, evidence_prompt):
        assert prompt["question"] == task.question
        assert prompt["decision_context"] == task.decision_relevance
        instructions = " ".join(prompt["instructions"]).lower()
        assert "unverified" in instructions
        assert "not paper evidence" in instructions
    assert "research_context" not in locator_prompt
    assert "research_context" not in evidence_prompt
    assert result.error is None


def test_master_state_exposes_deterministic_finding_ids() -> None:
    task = EvidenceTask("Earlier question")
    result = WorkerResult(
        task=task,
        finding="Earlier finding",
        evidence="Earlier evidence",
        caveat="Earlier caveat",
        evidence_type="text",
        evidence_locator="p. 1",
    )
    state = AgentState(
        findings=(result,),
        steps=(TraceStep(1, MasterAction("READ_PAPER", (task,)), (result,)),),
        remaining_rounds=4,
    )

    payload = runtime._state_payload(state)

    assert payload["findings"][0]["finding_id"] == "r1-t1-f1"


def test_master_prompt_explains_context_selection_as_a_bounded_research_lead() -> None:
    pages = make_pages()
    master = runtime.PaperAgentMaster(
        llm=FakeLLM([{"kind": "NEEDS_HUMAN"}]),
        paper_name="paper.pdf",
        page_index=runtime.build_compact_page_index(pages),
        overview_text=pages[0].text,
        overview_images=(),
    )

    master(AgentState(remaining_rounds=1))

    prompt = json.loads(master.llm.calls[0]["prompt"])
    task_contract = prompt["output_contract"]["READ_PAPER"]["tasks"][0]
    assert task_contract["related_finding_ids"] == []
    assert "decision_relevance" in task_contract
    assert "unverified" in task_contract["decision_relevance"]
    assert "purpose" in task_contract["decision_relevance"]
    assert "related_finding_ids" in " ".join(prompt["instructions"])


def test_evidence_findings_list_preserves_individual_finding_evidence_pairs_for_context() -> None:
    task = EvidenceTask("inspect local evidence")
    result = runtime._parse_evidence_result(
        task=task,
        payload={
            "findings": [
                {
                    "finding": "first finding",
                    "evidence": [{"content": "first text", "evidence_type": "text", "locator": "p. 1"}],
                    "caveat": "first caveat",
                },
                {
                    "finding": "second finding",
                    "evidence": [{"content": "second table", "evidence_type": "table", "locator": "Table 2, p. 2"}],
                    "caveat": "second caveat",
                },
            ]
        },
        selected_pages=(1, 2),
        location_rationale="local",
        selected_text={1: "first text", 2: "selected page"},
    )
    state = AgentState(
        findings=(result,),
        steps=(TraceStep(1, MasterAction("READ_PAPER", (task,)), (result,)),),
    )

    assert [(item.finding, item.evidence[0].content, item.caveat) for item in result.structured_findings] == [
        ("first finding", "first text", "first caveat"),
        ("second finding", "second table", "second caveat"),
    ]
    payload = runtime._state_payload(state)
    assert [(item["finding_id"], item["finding"], item["evidence"], item["caveat"]) for item in payload["findings"]] == [
        ("r1-t1-f1", "first finding", "first text", "first caveat"),
        ("r1-t1-f2", "second finding", "second table", "second caveat"),
    ]


def test_state_payload_keeps_structured_evidence_and_worker_diagnostics() -> None:
    successful_task = EvidenceTask("Inspect the comparison")
    failed_task = EvidenceTask("Inspect the missing control")
    successful = WorkerResult(
        task=successful_task,
        finding="legacy summary",
        evidence="primary evidence",
        caveat="summary caveat",
        pages_read=(3, 4),
        location_rationale="comparison pages",
        structured_findings=(
            WorkerFinding(
                finding="first finding",
                evidence=(
                    FindingEvidence("first text", "text", "p. 3"),
                    FindingEvidence("first table", "table", "Table 1, p. 4"),
                ),
                caveat="first caveat",
            ),
            WorkerFinding(
                finding="second finding",
                evidence=(
                    FindingEvidence("second figure", "figure", "Figure 2, p. 4"),
                    FindingEvidence("second text", "text", "p. 4"),
                ),
                caveat="second caveat",
            ),
        ),
        noncanonical_output_shape="C:/local/page.png",
    )
    failed = WorkerResult(
        task=failed_task,
        error="invalid_evidence_output:findings",
        pages_read=(5,),
        location_rationale="missing control page",
    )
    state = AgentState(
        findings=(successful, failed),
        steps=(
            TraceStep(
                1,
                MasterAction("READ_PAPER", (successful_task, failed_task)),
                (successful, failed),
            ),
        ),
        remaining_rounds=4,
    )

    payload = runtime._state_payload(state)

    assert payload["findings"] == [
        {
            "finding_id": "r1-t1-f1",
            "question": "Inspect the comparison",
            "finding": "first finding",
            "evidence": "first text",
            "evidence_type": "text",
            "evidence_locator": "p. 3",
            "evidence_items": [
                {"content": "first text", "evidence_type": "text", "locator": "p. 3"},
                {"content": "first table", "evidence_type": "table", "locator": "Table 1, p. 4"},
            ],
            "caveat": "first caveat",
            "pages_read": [3, 4],
            "location_rationale": "comparison pages",
        },
        {
            "finding_id": "r1-t1-f2",
            "question": "Inspect the comparison",
            "finding": "second finding",
            "evidence": "second figure",
            "evidence_type": "figure",
            "evidence_locator": "Figure 2, p. 4",
            "evidence_items": [
                {"content": "second figure", "evidence_type": "figure", "locator": "Figure 2, p. 4"},
                {"content": "second text", "evidence_type": "text", "locator": "p. 4"},
            ],
            "caveat": "second caveat",
            "pages_read": [3, 4],
            "location_rationale": "comparison pages",
        },
    ]
    assert payload["worker_failures"] == [
        {
            "question": "Inspect the missing control",
            "error": "invalid_evidence_output:findings",
            "pages_read": [5],
            "location_rationale": "missing control page",
        }
    ]
    serialized = json.dumps(payload)
    assert "data:image" not in serialized
    assert "C:/local/page.png" not in serialized


def test_role_mixed_worker_uses_discovery_or_cross_check_prompt_from_task_context(
    tmp_path: Path,
) -> None:
    pages = make_pages()
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    llm = FakeLLM(
        [
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "local"},
            {"finding": "local", "evidence": "Selected evidence: the ablation removes the planner.", "caveat": "", "evidence_type": "text", "evidence_locator": "p. 2"},
            {"page_ranges": [{"start": 2, "end": 2}], "rationale": "relation"},
            {"finding": "relation", "evidence": "Selected evidence: the ablation removes the planner.", "caveat": "", "evidence_type": "text", "evidence_locator": "p. 2"},
        ]
    )
    worker = runtime.PaperEvidenceWorker(
        pdf_path=pdf_path,
        pages=pages,
        page_index=runtime.build_compact_page_index(pages),
        llm=llm,
        render_pages=lambda _path, numbers: {page: f"data:image/png;base64,{page}" for page in numbers},
        worker_context_mode="selected-context",
        worker_role_mode="discovery-cross-check",
    )
    worker(EvidenceTask("independent local question"))
    worker.set_research_context(
        (
            ResearchContext(
                finding_id="r1-t1-f1",
                question="prior question",
                finding="prior finding",
                evidence="prior evidence",
                caveat="prior caveat",
                evidence_type="text",
                evidence_locator="p. 1",
                decision_relevance="check the relation",
            ),
        ),
        (),
    )
    worker(
        EvidenceTask(
            "bounded relation question",
            related_finding_ids=("r1-t1-f1",),
            decision_relevance="check the relation",
        )
    )

    prompts = [json.loads(call["prompt"]) for call in llm.calls]
    assert "research_context" not in prompts[0]
    assert "independent local evidence analyst" not in " ".join(
        prompts[0].get("instructions", [])
    )
    assert "independent local evidence analyst" in " ".join(prompts[1]["instructions"])
    assert prompts[2]["research_context"][0]["finding_id"] == "r1-t1-f1"
    assert "cross-finding reviewer" not in " ".join(prompts[2]["instructions"])
    assert "cross-finding reviewer" in " ".join(prompts[3]["instructions"])
    assert "do not delete, overwrite, or rewrite historical findings" in " ".join(
        prompts[3]["instructions"]
    )


def test_role_mixed_master_prompt_allows_discovery_and_cross_check_tasks_without_forcing_either() -> None:
    master = runtime.PaperAgentMaster(
        llm=FakeLLM([{"kind": "NEEDS_HUMAN"}]),
        paper_name="paper.pdf",
        page_index="[]",
        overview_text="overview",
        overview_images=(),
        worker_role_mode="discovery-cross-check",
    )

    master(AgentState(remaining_rounds=5))

    instructions = " ".join(json.loads(master.llm.calls[0]["prompt"])["instructions"])
    assert "First round normally uses Discovery" in instructions
    assert "mix Discovery and Cross-check" in instructions
    assert "do not force one task of each kind" in instructions
