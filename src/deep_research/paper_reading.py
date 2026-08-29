"""Utilities for preparing local paper PDFs for manual ScholarTrace reading."""

from __future__ import annotations

import base64
from io import BytesIO
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from deep_research.config import llm_api_base, llm_api_key_for_model, llm_model, llm_request_timeout


LOW_TEXT_THRESHOLD = 50


@dataclass(frozen=True)
class PaperPage:
    page_number: int
    text: str
    char_count: int
    low_text: bool
    used_fallback: bool
    error: str | None = None


@dataclass(frozen=True)
class PaperSection:
    kind: str
    heading: str
    text: str


@dataclass(frozen=True)
class ReadingArtifactPaths:
    text: Path
    reading_pack: Path
    chinese: Path
    judgment: Path


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

ABSTRACT_PREFIX_RE = re.compile(r"^\s*(?P<heading>Abstract)\s*[—\-:]\s*", re.IGNORECASE)
IEEE_INLINE_HEADING_RE = re.compile(
    r"(?P<heading>\b[IVXLCDM]+\.\s*[A-Z][A-Z]+(?:\s*[A-Z]+)*)"
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


def detect_sections(text: str) -> list[PaperSection]:
    normalized = normalize_text(text)
    matches_by_start: dict[int, tuple[str, str, int, int]] = {}
    for line_match in re.finditer(r"(?m)^[^\n]+", normalized):
        line = line_match.group(0)
        line_start = line_match.start()
        stripped = line.strip()

        abstract_match = ABSTRACT_PREFIX_RE.match(line)
        if abstract_match:
            start = line_start + abstract_match.start("heading")
            matches_by_start[start] = (
                "abstract",
                "Abstract",
                start,
                line_start + abstract_match.end(),
            )

        kind = section_kind_for_heading(stripped)
        if kind:
            start = line_start + len(line) - len(line.lstrip())
            matches_by_start.setdefault(start, (kind, stripped, start, line_match.end()))

        for inline_match in IEEE_INLINE_HEADING_RE.finditer(line):
            heading = re.sub(r"\s+", " ", inline_match.group("heading").strip())
            kind = section_kind_for_heading(heading)
            if not kind:
                continue
            start = line_start + inline_match.start("heading")
            matches_by_start.setdefault(start, (kind, heading, start, line_start + inline_match.end()))

    matches = sorted(matches_by_start.values(), key=lambda item: item[2])

    sections: list[PaperSection] = []
    for index, (kind, heading, _start, content_start) in enumerate(matches):
        content_end = matches[index + 1][2] if index + 1 < len(matches) else len(normalized)
        section_text = normalized[content_start:content_end].strip()
        sections.append(PaperSection(kind=kind, heading=heading, text=section_text))
    return sections


def render_reading_pack(
    *,
    title: str,
    source_pdf: Path,
    sections: Sequence[PaperSection],
    main_text: str,
) -> str:
    section_map = first_section_by_kind(sections)
    lines = [
        f"# {title}",
        "",
        "## Metadata",
        "",
        f"- Source PDF: `{source_pdf.as_posix()}`",
        f"- Main-text characters: {len(main_text)}",
        "- Reading scope: Appendix and full references are excluded by default.",
        "",
        "## Reading Questions",
        "",
        "- What problem does this paper solve for LLM-agent memory?",
        "- What architecture, benchmark, implementation detail, or warning can be reused?",
        "- Was ScholarTrace's status, role, and reason correct?",
        "- Which important papers, benchmarks, datasets, or repos did ScholarTrace miss?",
        "",
    ]

    ordered_sections = [
        ("abstract", "Abstract"),
        ("introduction", "Introduction"),
        ("related_work", "Related Work"),
        ("method", "Method / Framework"),
        ("experiments", "Experiments / Evaluation"),
        ("results", "Results"),
        ("discussion", "Discussion"),
        ("limitations", "Limitations"),
        ("conclusion", "Conclusion"),
    ]
    for kind, label in ordered_sections:
        section = section_map.get(kind)
        if section is None:
            continue
        lines.extend([f"## {label}", "", section.text.strip(), ""])

    lines.extend(
        [
            "## Missed-Paper Audit Notes",
            "",
            "- Important related papers mentioned by this paper:",
            "- Baselines, benchmarks, datasets, or repos ScholarTrace should track:",
            "- Retrieval or judging misses to investigate:",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def render_judgment_template(
    *,
    title: str,
    source_pdf: Path,
    deepresearch_source: str | None = None,
    extracted_text: Path | None = None,
    reading_pack: Path | None = None,
    chinese_translation: Path | None = None,
) -> str:
    source = deepresearch_source or "manual reading pass"
    lines = [
        f"## {title}",
        "",
        f"- Paper: `{title}`",
        f"- Local PDF: `{source_pdf.as_posix()}`",
        f"- Local extracted text: `{extracted_text.as_posix() if extracted_text else ''}`",
        f"- Local English reading pack: `{reading_pack.as_posix() if reading_pack else ''}`",
        f"- Local Chinese draft translation: `{chinese_translation.as_posix() if chinese_translation else ''}`",
        f"- ScholarTrace source: {source}",
        "- ScholarTrace status/role: ``",
        "- Manual rating: `must_read | useful | instructive | not_useful`",
        "- Main value type: `architecture | benchmark | implementation | survey | security | background`",
        "",
        "### Core Innovation",
        "",
        "",
        "### Method",
        "",
        "",
        "### Main Findings",
        "",
        "",
        "### Critical Evaluation",
        "",
        "",
        "### Reusable Ideas For Our Agent Project",
        "",
        "- ",
        "",
        "### Missed Papers, Benchmarks, Repos",
        "",
        "- ",
        "",
        "### ScholarTrace Feedback",
        "",
        "- Status correct: ``",
        "- Role correct: ``",
        "- Reason correct: ``",
        "- Retrieval or judging changes suggested: ``",
        "",
    ]
    return "\n".join(lines)


def build_translation_prompt(reading_pack: str) -> str:
    return f"""请把下面英文论文阅读包和关键章节翻译成中文。

要求：
- 这是机器辅助初译，优先准确，不要扩写，不要补充原文没有的信息。
- 保留英文论文标题、方法名、benchmark、dataset、repo 名称和引用标记，例如 [1]、Table 2、Figure 3。
- 保留 Markdown 标题层级、项目符号、代码块和字段名。
- 不要补译 appendix、supplementary material 或完整 references；如果阅读包提到相关工作，只翻译和 missed-paper audit 有关的短句。
- 术语保持一致：LLM agent=LLM 智能体，long-term memory=长期记忆，retrieval=检索，benchmark=基准。
- 只输出中文初译稿，不要输出解释。

英文阅读包：
{reading_pack}
"""


CompletionFunc = Callable[..., Any]
TranslatorFunc = Callable[[str], str]


def translate_reading_pack(
    reading_pack: str,
    *,
    completion_func: CompletionFunc | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    request_timeout: float | None = None,
) -> str:
    selected_model = model or llm_model()
    selected_api_key = api_key if api_key is not None else llm_api_key_for_model(selected_model)
    selected_api_base = api_base if api_base is not None else llm_api_base()
    selected_timeout = request_timeout if request_timeout is not None else llm_request_timeout()

    if completion_func is None:
        import litellm

        completion_func = litellm.completion

    kwargs: dict[str, Any] = {
        "model": selected_model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful technical translator for AI research papers.",
            },
            {"role": "user", "content": build_translation_prompt(reading_pack)},
        ],
        "request_timeout": selected_timeout,
        "max_retries": 0,
        "num_retries": 0,
    }
    if selected_api_key:
        kwargs["api_key"] = selected_api_key
    if selected_api_base:
        kwargs["api_base"] = selected_api_base

    response = completion_func(**kwargs)
    try:
        return str(response.choices[0].message.content or "").strip()
    except (AttributeError, IndexError, KeyError):
        return ""


def artifact_paths(
    pdf_path: Path,
    *,
    output_dir: Path | None = None,
    stem: str | None = None,
) -> ReadingArtifactPaths:
    base_dir = output_dir or pdf_path.parent
    base_stem = stem or pdf_path.stem
    return ReadingArtifactPaths(
        text=base_dir / f"{base_stem}.txt",
        reading_pack=base_dir / f"{base_stem}.read.md",
        chinese=base_dir / f"{base_stem}.zh.md",
        judgment=base_dir / f"{base_stem}.judgment.md",
    )


def prepare_reading_artifacts(
    pdf_path: Path,
    *,
    title: str | None = None,
    output_dir: Path | None = None,
    stem: str | None = None,
    translate: bool = False,
    translator: TranslatorFunc | None = None,
    deepresearch_source: str | None = None,
) -> ReadingArtifactPaths:
    path = Path(pdf_path)
    selected_title = title or path.stem
    paths = artifact_paths(path, output_dir=output_dir, stem=stem)

    pages = extract_pdf_pages(path)
    main_text = extract_main_text(pages)
    sections = detect_sections(main_text)
    page_text = render_page_text(pages)
    reading_pack = render_reading_pack(
        title=selected_title,
        source_pdf=path,
        sections=sections,
        main_text=main_text,
    )
    judgment = render_judgment_template(
        title=selected_title,
        source_pdf=path,
        deepresearch_source=deepresearch_source,
        extracted_text=paths.text,
        reading_pack=paths.reading_pack,
        chinese_translation=paths.chinese,
    )

    for output_path in (paths.text, paths.reading_pack, paths.judgment):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    paths.text.write_text(page_text, encoding="utf-8", newline="\n")
    paths.reading_pack.write_text(reading_pack, encoding="utf-8", newline="\n")
    paths.judgment.write_text(judgment, encoding="utf-8", newline="\n")

    if translate:
        translate_func = translator or translate_reading_pack
        chinese = translate_func(reading_pack).strip()
        paths.chinese.parent.mkdir(parents=True, exist_ok=True)
        paths.chinese.write_text(chinese + "\n", encoding="utf-8", newline="\n")

    return paths


def first_section_by_kind(sections: Sequence[PaperSection]) -> dict[str, PaperSection]:
    result: dict[str, PaperSection] = {}
    for section in sections:
        existing = result.get(section.kind)
        if existing is None:
            result[section.kind] = section
            continue

        texts = [part for part in (existing.text.strip(), section.text.strip()) if part]
        result[section.kind] = PaperSection(
            kind=existing.kind,
            heading=existing.heading,
            text="\n\n".join(texts),
        )
    return result


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
