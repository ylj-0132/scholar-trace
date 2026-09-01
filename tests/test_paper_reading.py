from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

import deep_research.paper_reading as paper_reading
from deep_research.paper_reading import (
    PaperPage,
    extract_main_text,
    normalize_text,
    render_pdf_pages,
    section_kind_for_heading,
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



@pytest.mark.parametrize(
    ("heading", "expected"),
    (
        ("Abstract", "abstract"),
        ("3 Method", "method"),
        ("IV. EXPERIMENTS", "experiments"),
        ("References", None),
    ),
)
def test_section_kind_for_heading(heading: str, expected: str | None) -> None:
    assert section_kind_for_heading(heading) == expected
