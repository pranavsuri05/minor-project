"""Summarisation Agent: per-paper summaries plus a multi-document synthesis with similarities and differences."""

from __future__ import annotations

import json
import logging

from agents.common import as_str_list, format_registry, keep_valid_sources
from services.citations import strip_invalid_citations
from services.llm import LLMClient, LLMOutputError

logger = logging.getLogger(__name__)

MAX_ANALYSIS_CHARS = 24_000

SYSTEM_PROMPT = """You are the Summarisation Agent in a research literature review system.
You receive the structured analysis of the sources. Write in formal academic English.

Produce:
- One summary per source (3 to 5 sentences) covering its objective, method and main result.
  Only summarise sources that appear in the analysis.
- A multi-document synthesis of 2 to 3 paragraphs. Highlight methodological similarities and
  differences. Avoid repeating the same sentence across sources.
- Lists of similarities and differences, each with the source ids it relies on.

Cite sources inline with ids from the registry, for example [S1] or [S2, p. 4]. Use only ids that exist
in the registry. Do not add facts that are not in the analysis. Do not expand or define an abbreviation unless the passages expand it; write abbreviations exactly as the passages do.

Return JSON with this shape:
{
  "paper_summaries": [{"source_id": "S1", "summary": ""}],
  "synthesis": "paragraph one\n\nparagraph two",
  "similarities": [{"statement": "", "source_ids": ["S1", "S2"]}],
  "differences": [{"statement": "", "source_ids": ["S1", "S2"]}]
}"""


class SummarisationAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def summarise(self, question: str, registry: list[dict], analysis: dict) -> dict:
        valid_ids = {entry["source_id"] for entry in registry}
        analysis_json = json.dumps(analysis, ensure_ascii=False, indent=1)[:MAX_ANALYSIS_CHARS]
        user = (
            f"Research question: {question}\n\nSource registry:\n{format_registry(registry)}\n\n"
            f"Structured analysis (JSON):\n{analysis_json}"
        )
        data = self.llm.complete_json("summarisation", SYSTEM_PROMPT, user)
        return self._normalise(data, valid_ids)

    @staticmethod
    def _normalise(data, valid_ids: set[str]) -> dict:
        if not isinstance(data, dict):
            raise LLMOutputError("The summarisation response was not a JSON object.")
        removed = 0

        summaries = []
        for raw in data.get("paper_summaries") or []:
            if not isinstance(raw, dict) or raw.get("source_id") not in valid_ids:
                removed += 1
                continue
            text, lost = strip_invalid_citations(str(raw.get("summary", "")).strip(), valid_ids)
            removed += lost
            if text:
                summaries.append({"source_id": raw["source_id"], "summary": text})

        synthesis, lost = strip_invalid_citations(str(data.get("synthesis", "")).strip(), valid_ids)
        removed += lost

        def items(key: str) -> list[dict]:
            nonlocal removed
            raw_items = []
            for entry in data.get(key) or []:
                if isinstance(entry, dict):
                    raw_items.append({"statement": str(entry.get("statement", "")).strip(),
                                      "source_ids": as_str_list(entry.get("source_ids"))})
            kept, lost_items = keep_valid_sources([i for i in raw_items if i["statement"]], valid_ids)
            removed += lost_items
            return kept

        result = {
            "paper_summaries": summaries,
            "synthesis": synthesis,
            "similarities": items("similarities"),
            "differences": items("differences"),
            "dropped_items": removed,
        }
        if not summaries and not synthesis:
            raise LLMOutputError("The summarisation response contained no usable summary.")
        return result
