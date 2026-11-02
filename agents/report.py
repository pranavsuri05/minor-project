"""Report Generation Agent: writes the structured Markdown report and assembles the final, cited version."""

from __future__ import annotations

import json
import logging
import re

from services.citations import format_reference_list, parse_citation_groups, render_citations
from services.llm import LLMClient

logger = logging.getLogger(__name__)

# Section headings in the order they must appear. Quality review checks for these.
REQUIRED_SECTIONS = [
    "Abstract",
    "1. Introduction",
    "2. Research Objectives",
    "3. Literature Review",
    "4. Comparative Analysis",
    "5. Research Gap Identification",
    "6. Future Research Directions",
    "7. Conclusion",
]
REFERENCES_HEADING = "References"

SYSTEM_PROMPT = """You are the Report Generation Agent. Write a structured academic research report in Markdown
from the analysis, summaries and research gaps you are given.

Structure (use these headings exactly, in this order):
# <report title>
## Abstract
## 1. Introduction
## 2. Research Objectives
## 3. Literature Review
## 4. Comparative Analysis
## 5. Research Gap Identification
## 6. Future Research Directions
## 7. Conclusion

Rules:
- Cite every claim taken from a source with the marker of the passage it comes from, including the
  page: [S1, p. 4]. Use only the markers listed in the passage index, and never invent a citation.
- Distinguish evidence-supported gaps from speculation. Start every future-direction paragraph with
  "Speculative:". Gaps must be stated as "the reviewed sources do not ..." when grounded.
- Do not invent numbers, results, authors, years or DOIs. If something is not in the inputs, leave it out.
- Do not expand or define an abbreviation unless the passages expand it; write abbreviations exactly as the passages do.
- Academic tone, concise paragraphs, no bullet-point walls in the literature review.
- Do NOT write a References section; it is generated automatically."""


class ReportAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def write(
        self,
        question: str,
        registry: list[dict],
        analysis: dict,
        summaries: dict,
        gaps: dict,
        feedback: str = "",
        evidence: list[dict] | None = None,
    ) -> str:
        """Return the report body as Markdown, with [S#] markers still unrendered."""
        registry_lines = "\n".join(
            f"{e['source_id']}: {e['title']} ({e['year']}), {e['page_count']} pages" for e in registry
        )
        user = (
            f"Research question: {question}\n\nRegistry (the only valid citation ids):\n{registry_lines}\n\n"
            f"Analysis (JSON):\n{json.dumps(analysis, ensure_ascii=False, indent=1)[:18_000]}\n\n"
            f"Summaries (JSON):\n{json.dumps(summaries, ensure_ascii=False, indent=1)[:8_000]}\n\n"
            f"Research gaps (JSON):\n{json.dumps(gaps, ensure_ascii=False, indent=1)[:6_000]}"
        )
        if evidence:
            index = "\n".join(
                f"[{e['source_id']}, p. {e['page_number']}] {e['text'][:240]}"
                for e in sorted(evidence, key=lambda e: (e["source_id"], e["page_number"]))[:14]
            )
            user += f"\n\nPassage index (the only page-level citation markers; copy them exactly):\n{index}"
        if feedback:
            user += (
                "\n\nThis is a revision. The quality reviewer found problems in the previous draft. "
                f"Fix them and keep everything else:\n{feedback}"
            )
        raw = self.llm.complete("report", SYSTEM_PROMPT, user)
        body = _strip_fences(raw).strip()
        logger.info("Report body generated (%d chars)", len(body))
        return body

    @staticmethod
    def finalise(body: str, registry: list[dict], style: str, quality_notes: list[str] | None = None) -> dict:
        """Render citations in the chosen style, add a deterministic References section and optional notes.

        Returns {"markdown": str, "cited_source_ids": [...]}.
        """
        rendered, cited = render_citations(body, registry, style)
        parts = [rendered.rstrip()]
        if quality_notes:
            parts.append("## Quality Review Notes\n\n" + "\n".join(f"- {note}" for note in quality_notes))
        if cited:
            references = format_reference_list(registry, cited, style)
            parts.append(f"## {REFERENCES_HEADING}\n\n" + "\n\n".join(references))
        markdown = "\n\n".join(parts) + "\n"
        return {"markdown": markdown, "cited_source_ids": cited}


_MARKER_RE = re.compile(r"\[S\d+[^\[\]\n]*\]")


def attach_pages(body: str, evidence: list[dict], embedder, min_similarity: float = 0.35) -> tuple[str, int]:
    """Add a page number to citations the model left without one.

    The sentence before each unpaged citation is matched against that source's retrieved passages
    (embedding similarity). The page of the best passage is used only if the match is strong enough;
    otherwise the citation stays page-less. Returns (new_body, number_of_pages_attached).
    """
    by_source: dict[str, list[dict]] = {}
    for item in evidence:
        by_source.setdefault(item["source_id"], []).append(item)
    vectors: dict[str, list[list[float]]] = {
        sid: embedder.embed_documents([p["text"] for p in passages]) for sid, passages in by_source.items()
    }

    pieces: list[str] = []
    cursor = 0
    attached = 0
    for start, end, refs in parse_citation_groups(body):
        if all(ref.page is not None for ref in refs):
            continue
        prefix = _MARKER_RE.sub("", body[:start])
        sentence = prefix[max(prefix.rfind(". "), prefix.rfind("\n")) + 1 :].strip()
        query = embedder.embed_query(sentence) if sentence else None
        parts: list[str] = []
        for ref in refs:
            page = ref.page
            if page is None and query is not None and ref.source_id in by_source:
                scores = [sum(a * b for a, b in zip(query, vec)) for vec in vectors[ref.source_id]]
                best = max(range(len(scores)), key=scores.__getitem__)
                if scores[best] >= min_similarity:
                    page = by_source[ref.source_id][best]["page_number"]
                    attached += 1
            parts.append(ref.source_id + (f", p. {page}" if page is not None else ""))
        pieces.append(body[cursor:start])
        pieces.append("[" + ", ".join(parts) + "]")
        cursor = end
    pieces.append(body[cursor:])
    return "".join(pieces), attached


def extract_title(markdown: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```$", text, re.DOTALL)
    return match.group(1) if match else text
