from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

import deep_research.paper_reading as paper_reading
from deep_research.paper_reading import (
    PaperPage,
    PaperSection,
    artifact_paths,
    build_translation_prompt,
    detect_sections,
    extract_main_text,
    prepare_reading_artifacts,
    render_judgment_template,
    render_reading_pack,
    translate_reading_pack,
    normalize_text,
    render_pdf_pages,
)


def test_extract_main_text_stops_before_references_and_appendix() -> None:
    pages = [
        PaperPage(
            page_number=1,
            text="""
            Abstract
            We study memory systems for LLM agents.

            1 Introduction
            Agent memory needs durable updating.
            """,
            char_count=92,
            low_text=False,
            used_fallback=False,
        ),
        PaperPage(
            page_number=2,
            text="""
            2 Method
            The system stores observations in a temporal graph.

            References
            [1] A background paper.
            Appendix A
            Extra proof details.
            """,
            char_count=148,
            low_text=False,
            used_fallback=False,
        ),
    ]

    main_text = extract_main_text(pages)

    assert "2 Method" in main_text
    assert "References" not in main_text
    assert "Appendix A" not in main_text
    assert "Extra proof details" not in main_text


def test_extract_main_text_does_not_stop_on_inline_appendix_reference() -> None:
    pages = [
        PaperPage(
            page_number=1,
            text="""
            2 Tracing and Attributing Errors in Memory Systems
            Appendix B for the full formalization.

            3 MemTraceBench Construction
            We construct a diagnostic benchmark.
            """,
            char_count=170,
            low_text=False,
            used_fallback=False,
        ),
        PaperPage(
            page_number=2,
            text="""
            5 Experiments
            MemTrace improves attribution.

            References
            [1] A paper.
            """,
            char_count=93,
            low_text=False,
            used_fallback=False,
        ),
    ]

    main_text = extract_main_text(pages)

    assert "Appendix B for the full formalization." in main_text
    assert "3 MemTraceBench Construction" in main_text
    assert "5 Experiments" in main_text
    assert "References" not in main_text


def test_normalize_text_removes_nul_control_characters() -> None:
    assert normalize_text("Memory\x00Graph\x0cPaper") == "MemoryGraph Paper"


def test_extract_pdf_pages_prefers_pypdf_for_research_papers(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    def fake_pypdf(path: Path, *, used_fallback: bool = False) -> list[PaperPage]:
        assert path == pdf_path
        assert used_fallback is False
        return [
            PaperPage(
                page_number=1,
                text="pypdf preserves the abstract order",
                char_count=34,
                low_text=True,
                used_fallback=False,
            )
        ]

    monkeypatch.setattr(paper_reading, "_extract_pdf_pages_with_pypdf", fake_pypdf)

    pages = paper_reading.extract_pdf_pages(pdf_path)

    assert pages[0].text == "pypdf preserves the abstract order"


def test_render_pdf_pages_only_renders_requested_pages(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    calls: list[int] = []

    class FakeImage:
        def __init__(self, page_number: int) -> None:
            self.page_number = page_number
            self.original = self

        def save(self, output: object, *, format: str) -> None:
            assert format == "PNG"
            output.write(f"png-page-{self.page_number}".encode())

    class FakePage:
        def __init__(self, page_number: int) -> None:
            self.page_number = page_number

        def to_image(self, *, resolution: int) -> FakeImage:
            assert resolution == 144
            calls.append(self.page_number)
            return FakeImage(self.page_number)

    class FakePdf:
        pages = [FakePage(1), FakePage(2), FakePage(3)]

        def __enter__(self) -> "FakePdf":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakePdfPlumber:
        @staticmethod
        def open(path: str) -> FakePdf:
            assert path == str(pdf_path)
            return FakePdf()

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    monkeypatch.setitem(sys.modules, "pdfplumber", FakePdfPlumber)

    rendered = render_pdf_pages(pdf_path, [3, 1, 3])

    assert set(rendered) == {1, 3}
    assert calls == [1, 3]
    assert base64.b64decode(rendered[1].split(",", 1)[1]) == b"png-page-1"
    assert rendered[3].startswith("data:image/png;base64,")


def test_render_pdf_pages_rejects_out_of_range_page(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    class FakePdf:
        pages = [object(), object()]

        def __enter__(self) -> "FakePdf":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class FakePdfPlumber:
        @staticmethod
        def open(path: str) -> FakePdf:
            return FakePdf()

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    monkeypatch.setitem(sys.modules, "pdfplumber", FakePdfPlumber)

    with pytest.raises(ValueError, match="page 3"):
        render_pdf_pages(pdf_path, [3])


def test_detect_sections_finds_research_paper_headings() -> None:
    text = """
    Abstract
    We propose REAL for agent memory.

    1 Introduction
    Long-term memory should preserve evolving facts.

    2 Related Work
    Prior memory systems overwrite old facts.

    3 Framework
    REAL stores temporal property graph snapshots.

    4 Experiments
    The benchmark compares retrieval and reasoning.

    5 Limitations
    The approach depends on extraction quality.

    6 Conclusion
    Graph memory is useful for long-running agents.
    """

    sections = detect_sections(text)

    assert [(section.kind, section.heading) for section in sections] == [
        ("abstract", "Abstract"),
        ("introduction", "1 Introduction"),
        ("related_work", "2 Related Work"),
        ("method", "3 Framework"),
        ("experiments", "4 Experiments"),
        ("limitations", "5 Limitations"),
        ("conclusion", "6 Conclusion"),
    ]
    assert sections[3].text == "REAL stores temporal property graph snapshots."


def test_detect_sections_handles_ieee_style_headings_from_pdf_text() -> None:
    text = """
    REAL: A Reasoning-Enhanced Graph Framework
    Abstract—We propose REAL for long-term memory management.

    I. INTRODUCTION outsidetheLLM,indexesitforefficientretrieval
    Long-term memory is difficult for finite context windows.

    III. SYSTEMOVERVIEW
    REAL constructs a temporal and confidence-aware graph.

    IV. EXPERIMENTSANDRESULTS
    REAL improves memory QA on long-context benchmarks.

    V. RELATEDWORK
    Prior systems use flat memory or destructive graph updates.

    VI. CONCLUSION
    REAL improves retrieval by reasoning over graph memory.
    """

    sections = detect_sections(text)

    assert [(section.kind, section.heading) for section in sections] == [
        ("abstract", "Abstract"),
        ("introduction", "I. INTRODUCTION"),
        ("method", "III. SYSTEMOVERVIEW"),
        ("experiments", "IV. EXPERIMENTSANDRESULTS"),
        ("related_work", "V. RELATEDWORK"),
        ("conclusion", "VI. CONCLUSION"),
    ]
    assert sections[0].text.startswith("We propose REAL")
    assert "finite context windows" in sections[1].text
    assert "temporal and confidence-aware graph" in sections[2].text


def test_render_reading_pack_keeps_english_sections_and_notes_exclusions() -> None:
    sections = [
        PaperSection("abstract", "Abstract", "We propose a temporal graph memory."),
        PaperSection("method", "3 Framework", "The framework stores parallel fact versions."),
        PaperSection("experiments", "4 Experiments", "Experiments compare memory retrieval."),
        PaperSection("limitations", "5 Limitations", "Extraction quality remains a bottleneck."),
    ]

    markdown = render_reading_pack(
        title="REAL: A Reasoning-Enhanced Graph Framework",
        source_pdf=Path("paper/real.pdf"),
        sections=sections,
        main_text="main text",
    )

    assert markdown.startswith("# REAL: A Reasoning-Enhanced Graph Framework\n")
    assert "- Source PDF: `paper/real.pdf`" in markdown
    assert "## Abstract\n\nWe propose a temporal graph memory." in markdown
    assert "## Method / Framework\n\nThe framework stores parallel fact versions." in markdown
    assert "## Experiments / Evaluation\n\nExperiments compare memory retrieval." in markdown
    assert "## Limitations\n\nExtraction quality remains a bottleneck." in markdown
    assert "Appendix and full references are excluded by default." in markdown
    assert "## Missed-Paper Audit Notes" in markdown


def test_render_reading_pack_combines_repeated_section_kinds() -> None:
    sections = [
        PaperSection("experiments", "5 Experiments", ""),
        PaperSection("experiments", "5.1 Experimental Setup", "The benchmark uses MemTraceBench."),
        PaperSection("results", "5.2 Results", "MemTrace improves error attribution."),
    ]

    markdown = render_reading_pack(
        title="MemTrace",
        source_pdf=Path("paper/memtrace.pdf"),
        sections=sections,
        main_text="main text",
    )

    assert "## Experiments / Evaluation\n\nThe benchmark uses MemTraceBench." in markdown
    assert "## Results\n\nMemTrace improves error attribution." in markdown


def test_render_judgment_template_matches_manual_reading_fields() -> None:
    markdown = render_judgment_template(
        title="REAL: A Reasoning-Enhanced Graph Framework",
        source_pdf=Path("paper/real.pdf"),
        deepresearch_source="overlap of A-MEM run 5 and MemOS run 6",
    )

    assert "## REAL: A Reasoning-Enhanced Graph Framework" in markdown
    assert "- Local PDF: `paper/real.pdf`" in markdown
    assert "- ScholarTrace source: overlap of A-MEM run 5 and MemOS run 6" in markdown
    assert "- Manual rating: `must_read | useful | instructive | not_useful`" in markdown
    assert "- Main value type: `architecture | benchmark | implementation | survey | security | background`" in markdown
    assert "### Reusable Ideas For Our Agent Project" in markdown
    assert "### Missed Papers, Benchmarks, Repos" in markdown
    assert "### ScholarTrace Feedback" in markdown


def test_build_translation_prompt_keeps_names_and_excludes_appendix_references() -> None:
    prompt = build_translation_prompt("# REAL\n\n## Method / Framework\n\nTemporal graph memory.")

    assert "保留英文论文标题、方法名、benchmark、dataset、repo 名称和引用标记" in prompt
    assert "不要补译 appendix、supplementary material 或完整 references" in prompt
    assert "只输出中文初译稿" in prompt
    assert "# REAL" in prompt
    assert "Temporal graph memory." in prompt


def test_translate_reading_pack_uses_injected_completion() -> None:
    calls: list[dict[str, object]] = []

    class FakeMessage:
        content = "中文初译"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_completion(**kwargs: object) -> FakeResponse:
        calls.append(kwargs)
        return FakeResponse()

    result = translate_reading_pack(
        "# REAL\n\n## Abstract\n\nMemory paper.",
        completion_func=fake_completion,
        model="test-model",
        api_key="test-key",
        api_base="https://example.test",
        request_timeout=12,
    )

    assert result == "中文初译"
    assert calls[0]["model"] == "test-model"
    assert calls[0]["api_key"] == "test-key"
    assert calls[0]["api_base"] == "https://example.test"
    assert calls[0]["request_timeout"] == 12
    assert calls[0]["max_retries"] == 0
    assert calls[0]["num_retries"] == 0


def test_artifact_paths_use_output_dir_and_stem() -> None:
    paths = artifact_paths(
        Path("paper/2026_REAL_Reasoning_Enhanced_Graph_Framework_for_Long_Term_Memory.pdf"),
        output_dir=Path("out"),
        stem="real",
    )

    assert paths.text == Path("out/real.txt")
    assert paths.reading_pack == Path("out/real.read.md")
    assert paths.chinese == Path("out/real.zh.md")
    assert paths.judgment == Path("out/real.judgment.md")


def test_prepare_reading_artifacts_writes_outputs_without_translation(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")

    def fake_extract_pdf_pages(path: Path) -> list[PaperPage]:
        assert path == pdf_path
        return [
            PaperPage(
                page_number=1,
                text="""
                Abstract
                A memory paper.

                1 Introduction
                It studies long-term memory.

                References
                [1] Hidden reference.
                """,
                char_count=120,
                low_text=False,
                used_fallback=False,
            )
        ]

    monkeypatch.setattr(paper_reading, "extract_pdf_pages", fake_extract_pdf_pages)

    paths = prepare_reading_artifacts(
        pdf_path,
        title="Memory Paper",
        output_dir=tmp_path / "out",
        stem="memory-paper",
        translate=False,
        deepresearch_source="test run",
    )

    assert paths.text.read_text(encoding="utf-8").startswith("===== Page 1 =====")
    assert "# Memory Paper" in paths.reading_pack.read_text(encoding="utf-8")
    assert "Hidden reference" not in paths.reading_pack.read_text(encoding="utf-8")
    judgment = paths.judgment.read_text(encoding="utf-8")
    assert f"- Local extracted text: `{paths.text.as_posix()}`" in judgment
    assert f"- Local English reading pack: `{paths.reading_pack.as_posix()}`" in judgment
    assert f"- Local Chinese draft translation: `{paths.chinese.as_posix()}`" in judgment
    assert "ScholarTrace source: test run" in judgment
    assert not paths.chinese.exists()


def test_prepare_reading_artifacts_writes_translation_when_requested(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF fake")
    monkeypatch.setattr(
        paper_reading,
        "extract_pdf_pages",
        lambda _path: [
            PaperPage(
                page_number=1,
                text="Abstract\nA memory paper.",
                char_count=24,
                low_text=True,
                used_fallback=False,
            )
        ],
    )

    paths = prepare_reading_artifacts(
        pdf_path,
        title="Memory Paper",
        output_dir=tmp_path / "out",
        stem="memory-paper",
        translate=True,
        translator=lambda reading_pack: "中文初译：" + reading_pack.splitlines()[0],
    )

    assert paths.chinese.read_text(encoding="utf-8") == "中文初译：# Memory Paper\n"
