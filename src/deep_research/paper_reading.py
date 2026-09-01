"""Utilities for preparing local paper PDFs for manual ScholarTrace reading."""

from __future__ import annotations

import base64
from io import BytesIO
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence



LOW_TEXT_THRESHOLD = 50


@dataclass(frozen=True)
class PaperPage:
    page_number: int
    text: str
    char_count: int
    low_text: bool
    used_fallback: bool
    error: str | None = None



SECTION_KIND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abstract", re.compile(r"^abstract$", re.IGNORECASE)),
    (
        "introduction",
        re.compile(r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?introduction$", re.IGNORECASE),
    ),
    (
        "related_work",
        re.compile(
            r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?(?:related\s*work|background)$",
            re.IGNORECASE,
        ),
    ),
    (
        "method",
        re.compile(
            r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?"
            r"(?:method|methods|methodology|approach|framework|model|architecture|"
            r"system\s*design|system\s*overview)$",
            re.IGNORECASE,
        ),
    ),
    (
        "experiments",
        re.compile(
            r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?"
            r"(?:experiments?|evaluation|experimental\s*setup|empirical\s*evaluation|"
            r"experiments\s*and\s*results)$",
            re.IGNORECASE,
        ),
    ),
    (
        "results",
        re.compile(r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?(?:results?|findings)$", re.IGNORECASE),
    ),
    (
        "discussion",
        re.compile(r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?discussion$", re.IGNORECASE),
    ),
    (
        "limitations",
        re.compile(r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?limitations?$", re.IGNORECASE),
    ),
    (
        "conclusion",
        re.compile(r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?conclusions?$", re.IGNORECASE),
    ),
)


MAIN_TEXT_CUTOFF_RE = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(?:references|bibliography|appendix(?:\s+[A-Z0-9](?:\s*[:.\-].*)?)?|appendices|"
    r"supplementary(?:\s+(?:material|information))?|supplemental\s+(?:material|information)|"
    r"acknowledg(?:e)?ments?)\s*$"
)


def extract_pdf_pages(pdf_path: Path) -> list[PaperPage]:
    """Extract page text from a local PDF, tolerating page-level failures."""

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path}")

    try:
        return _extract_pdf_pages_with_pypdf(path, used_fallback=False)
    except Exception as pypdf_exc:
        try:
            return _extract_pdf_pages_with_pdfplumber(path, used_fallback=True)
        except Exception as pdfplumber_exc:
            raise RuntimeError(
                f"PDF text extraction failed with pypdf ({pypdf_exc}) "
                f"and pdfplumber ({pdfplumber_exc})"
            ) from pdfplumber_exc


def render_pdf_pages(
    pdf_path: Path,
    page_numbers: Sequence[int],
    *,
    dpi: int = 144,
) -> dict[int, str]:
    """Render selected 1-based PDF pages as PNG data URLs without writing files."""

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path}")
    requested = list(page_numbers)
    if not requested:
        raise ValueError("at least one page is required")
    if any(isinstance(page, bool) or not isinstance(page, int) or page < 1 for page in requested):
        raise ValueError("page numbers must be positive integers")

    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber is not available") from exc

    with pdfplumber.open(str(path)) as pdf:
        total_pages = len(pdf.pages)
        unique_pages = sorted(set(requested))
        if unique_pages[-1] > total_pages:
            raise ValueError(
                f"page {unique_pages[-1]} is out of range 1..{total_pages}"
            )
        rendered: dict[int, str] = {}
        for page_number in unique_pages:
            try:
                page_image = pdf.pages[page_number - 1].to_image(resolution=dpi)
                buffer = BytesIO()
                page_image.original.save(buffer, format="PNG")
            except Exception as exc:
                raise RuntimeError(
                    f"PDF page rendering failed for page {page_number}: {exc}"
                ) from exc
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            rendered[page_number] = f"data:image/png;base64,{encoded}"
    return rendered


def _extract_pdf_pages_with_pdfplumber(path: Path, *, used_fallback: bool) -> list[PaperPage]:
    try:
        import pdfplumber
    except ImportError:
        raise RuntimeError("pdfplumber is not available")

    pages: list[PaperPage] = []
    with pdfplumber.open(str(path)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            try:
                text = normalize_text(page.extract_text() or "")
                pages.append(_make_page(index, text, used_fallback=used_fallback))
            except Exception as exc:  # pragma: no cover - defensive path for malformed pages.
                pages.append(
                    PaperPage(
                        page_number=index,
                        text="",
                        char_count=0,
                        low_text=True,
                        used_fallback=used_fallback,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return pages


def _extract_pdf_pages_with_pypdf(path: Path, *, used_fallback: bool) -> list[PaperPage]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[PaperPage] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = normalize_text(page.extract_text() or "")
            pages.append(_make_page(index, text, used_fallback=used_fallback))
        except Exception as exc:  # pragma: no cover - defensive path for malformed pages.
            pages.append(
                PaperPage(
                    page_number=index,
                    text="",
                    char_count=0,
                    low_text=True,
                    used_fallback=used_fallback,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return pages


def _make_page(page_number: int, text: str, *, used_fallback: bool) -> PaperPage:
    char_count = len(text)
    return PaperPage(
        page_number=page_number,
        text=text,
        char_count=char_count,
        low_text=char_count < LOW_TEXT_THRESHOLD,
        used_fallback=used_fallback,
    )


def render_page_text(pages: Sequence[PaperPage]) -> str:
    chunks: list[str] = []
    for page in pages:
        chunks.append(f"===== Page {page.page_number} =====\n\n{page.text.strip()}")
    return "\n\n".join(chunks).strip() + "\n"


def extract_main_text(pages: Sequence[PaperPage]) -> str:
    text = normalize_text("\n\n".join(page.text for page in pages if page.text.strip()))
    cutoff = MAIN_TEXT_CUTOFF_RE.search(text)
    if cutoff:
        text = text[: cutoff.start()]
    return text.strip()


def section_kind_for_heading(heading: str) -> str | None:
    compact = re.sub(r"\s+", " ", heading.strip())
    for kind, pattern in SECTION_KIND_PATTERNS:
        if pattern.match(compact):
            return kind
    return None


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f]", " ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
