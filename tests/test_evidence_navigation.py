from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_research import paper_agent_runtime as runtime, paper_reading as reading
from deep_research.paper_reading import PaperPage


def make_page(page_number, text):
    return PaperPage(page_number=page_number, text=text, char_count=len(text), low_text=False, used_fallback=False)


@pytest.mark.parametrize("heading,caption", [
    ("D.1 Importance configuration", "Table C.5: Effective budget sensitivity"),
    ("3.2 Retrieval settings", "Figure A.2: Ablation results"),
    ("Appendix B Additional controls", "Table S1. Controls"),
])
def test_navigation_finds_internal_headings_and_lettered_captions(heading, caption):
    page = make_page(page_number=1, text="Ordinary content. " * 80 + f"\n{heading}\n{caption}\n")
    index = json.loads(runtime.build_compact_page_index([page], preview_chars=40))[0]
    assert heading in index["headings"]
    assert caption in index["captions"]
    assert heading not in index["preview"]
    assert len(index["preview"]) == 40


def test_query_navigation_is_bounded_and_searches_beyond_page_prefix():
    pages = [make_page(page_number=i, text="Unrelated introductory prose. " * 100
                       + f"\nSection {i}\nImportance rating configuration uses the recorded agent alias.\n"
                       + "Unrelated trailing prose. " * 100) for i in range(1, 11)]
    snippets = reading.question_navigation_snippets(pages, "What importance rating configuration is reported?")
    assert 1 <= len(snippets) <= 6
    assert [s["page_number"] for s in snippets] == sorted(s["page_number"] for s in snippets)
    assert all(len(s["text"]) <= 240 for s in snippets)
    assert all("Importance rating configuration" in s["text"] for s in snippets)
    assert snippets == reading.question_navigation_snippets(pages, "What importance rating configuration is reported?")
    assert reading.question_navigation_snippets(pages, "unmatchedquasar") == []


def test_navigation_handles_wrapped_titles_without_promoting_formula_lines():
    headings, captions = reading.page_navigation_markers(
        "D.1\nConfiguration settings\nTable C.5: Controls\nX i\n1 K\nK KX\n"
        "0.5. The clock is initialized at creation.\n3.2 Retrieval settings"
    )
    assert headings == ["D.1 Configuration settings", "3.2 Retrieval settings"]
    assert captions == ["Table C.5: Controls"]


def test_locator_receives_query_snippets_but_worker_still_reads_selected_pages():
    class LLM:
        def __init__(self): self.prompts = []
        def complete_json(self, prompt, **kwargs):
            self.prompts.append(json.loads(prompt))
            if len(self.prompts) == 1:
                return {"page_ranges": [{"start": 2, "end": 2}], "rationale": "configuration section"}
            return {"findings": [{"finding": "The configuration is specified.", "evidence": [
                {"content": "Importance rating configuration is specified.", "evidence_type": "text", "locator": "p. 2"}
            ], "caveat": "Selected page only."}]}
    pages = [make_page(page_number=1, text="Introduction."), make_page(page_number=2,
        text="Prelude " * 100 + "\nD.1 Configuration\nImportance rating configuration is specified.")]
    llm = LLM()
    worker = runtime.PaperEvidenceWorker(pdf_path=Path("not-opened.pdf"), pages=pages,
        page_index=runtime.build_compact_page_index(pages), llm=llm,
        render_pages=lambda p, ns: {n: "data:image/png;base64,x" for n in ns})
    result = worker(runtime.EvidenceTask("What importance rating configuration is specified?"))
    assert not result.error
    assert llm.prompts[0]["navigation_snippets"][0]["page_number"] == 2
    assert llm.prompts[1]["selected_pages"][0]["text"] == pages[1].text
    assert "navigation_snippets" not in llm.prompts[1]
    assert len(llm.prompts) == 2
