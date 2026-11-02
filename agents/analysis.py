"""Semantic Analysis Agent: extracts objectives, methods, findings and limitations, and compares papers."""

from __future__ import annotations

import logging

from agents.common import as_str_list, format_evidence, format_registry, keep_valid_sources
from services.llm import LLMClient, LLMOutputError

logger = logging.getLogger(__name__)

COMPARISON_TYPES = {"agreement", "contradiction", "difference"}

SYSTEM_PROMPT = """You are the Semantic Analysis Agent in a research literature review system.
You receive a research question, the source registry and retrieved passages from the sources.

Rules:
- Use only information stated in the passages. If a field is not stated, leave it empty or omit the item.
- Every item must cite the source ids it relies on, using the ids from the registry (for example "S2").
- Compare papers. Label each comparison as "agreement", "contradiction" or "difference".
- Do not invent numbers, results, authors or years. Do not expand or define an abbreviation unless the passages expand it; write abbreviations exactly as the passages do.

Return JSON with this shape:
{
  "papers": [
    {
      "source_id": "S1",
      "objective": "",
      "methodology": "",
      "contributions": [{"statement": "", "source_ids": ["S1"]}],
      "findings": [{"statement": "", "source_ids": ["S1"]}],
      "limitations": [{"statement": "", "source_ids": ["S1"]}]
    }
  ],
  "comparisons": [
    {"type": "agreement", "topic": "", "statement": "", "source_ids": ["S1", "S2"]}
  ]
}"""


class SemanticAnalysisAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def analyse(self, question: str, subtasks: list[dict], registry: list[dict], evidence: list[dict]) -> dict:
        valid_ids = {entry["source_id"] for entry in registry}
        subtask_lines = "\n".join(f"- {s['title']}: {s.get('description', '')}" for s in subtasks)
        user = (
            f"Research question: {question}\n\nSubtasks:\n{subtask_lines}\n\n"
            f"Source registry:\n{format_registry(registry)}\n\n"
            f"Retrieved passages (cite with the marker shown before each passage):\n"
            f"{format_evidence(evidence, registry)}"
        )
        data = self.llm.complete_json("analysis", SYSTEM_PROMPT, user)
        result = self._normalise(data, valid_ids)
        logger.info(
            "Analysis: %d papers, %d comparisons, %d items dropped for invalid citations",
            len(result["papers"]), len(result["comparisons"]), result["dropped_items"],
        )
        return result

    @staticmethod
    def _normalise(data, valid_ids: set[str]) -> dict:
        if not isinstance(data, dict):
            raise LLMOutputError("The analysis response was not a JSON object.")
        dropped = 0
        papers: list[dict] = []
        for raw in data.get("papers") or []:
            if not isinstance(raw, dict) or raw.get("source_id") not in valid_ids:
                dropped += 1
                continue
            sections = {}
            for key in ("contributions", "findings", "limitations"):
                kept, lost = keep_valid_sources(_as_items(raw.get(key)), valid_ids)
                sections[key] = kept
                dropped += lost
            papers.append(
                {
                    "source_id": raw["source_id"],
                    "objective": str(raw.get("objective", "")).strip(),
                    "methodology": str(raw.get("methodology", "")).strip(),
                    **sections,
                }
            )

        comparisons: list[dict] = []
        for raw in data.get("comparisons") or []:
            if not isinstance(raw, dict):
                dropped += 1
                continue
            kept, lost = keep_valid_sources([raw], valid_ids)
            dropped += lost
            if not kept or len(kept[0]["source_ids"]) < 2:
                # A comparison needs at least two distinct sources to be meaningful.
                if kept:
                    dropped += 1
                continue
            item = kept[0]
            item["type"] = str(item.get("type", "")).lower()
            if item["type"] not in COMPARISON_TYPES:
                item["type"] = "difference"
            item["topic"] = str(item.get("topic", "")).strip()
            comparisons.append(item)

        return {"papers": papers, "comparisons": comparisons, "dropped_items": dropped}


def _as_items(value) -> list:
    """Accept either {"statement", "source_ids"} objects or plain strings (which cannot be traced)."""
    items = []
    for entry in value or []:
        if isinstance(entry, dict):
            items.append({"statement": str(entry.get("statement", "")).strip(),
                          "source_ids": as_str_list(entry.get("source_ids"))})
        elif isinstance(entry, str):
            items.append({"statement": entry.strip(), "source_ids": []})
    return [i for i in items if i["statement"]]
