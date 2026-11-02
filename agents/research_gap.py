"""Research Gap Identification Agent: separates evidence-backed gaps from speculative directions."""

from __future__ import annotations

import json
import logging

from agents.common import as_str_list, format_registry, keep_valid_sources
from services.llm import LLMClient, LLMOutputError

logger = logging.getLogger(__name__)

EVIDENCE = "evidence_supported"
SPECULATIVE = "speculative"

SYSTEM_PROMPT = """You are the Research Gap Identification Agent in a literature review system.
You receive the source registry, the structured analysis and the summaries. Identify what the
reviewed literature leaves unresolved.

Two kinds of output are allowed:
1. Gaps with category "evidence_supported": grounded in explicit limitations, missing comparisons,
   untested conditions, or insufficient evidence REPORTED BY the listed sources. Each must list the
   source_ids that support it and a "basis" that quotes or paraphrases that limitation.
2. Future directions: possible research directions that go beyond the sources. They are always
   speculative. Give a rationale and, where relevant, the source_ids that motivate them.

Never invent limitations. If the sources do not show a gap, return fewer items.

Return JSON with this shape:
{
  "gaps": [
    {"title": "", "description": "", "category": "evidence_supported", "basis": "", "source_ids": ["S1"]}
  ],
  "future_directions": [
    {"direction": "", "rationale": "", "source_ids": ["S2"]}
  ]
}"""


class ResearchGapAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def identify(self, question: str, registry: list[dict], analysis: dict, summaries: dict) -> dict:
        valid_ids = {entry["source_id"] for entry in registry}
        user = (
            f"Research question: {question}\n\nSource registry:\n{format_registry(registry)}\n\n"
            f"Structured analysis (JSON):\n{json.dumps(analysis, ensure_ascii=False, indent=1)[:16_000]}\n\n"
            f"Summaries (JSON):\n{json.dumps(summaries, ensure_ascii=False, indent=1)[:8_000]}"
        )
        data = self.llm.complete_json("research_gaps", SYSTEM_PROMPT, user)
        return self._normalise(data, valid_ids)

    @staticmethod
    def _normalise(data, valid_ids: set[str]) -> dict:
        if not isinstance(data, dict):
            raise LLMOutputError("The research gap response was not a JSON object.")

        gaps: list[dict] = []
        for raw in data.get("gaps") or []:
            if not isinstance(raw, dict) or not str(raw.get("title", "")).strip():
                continue
            sources = [s for s in as_str_list(raw.get("source_ids")) if s in valid_ids]
            # A gap only counts as evidence-supported when it cites real sources and a basis for it.
            supported = bool(sources) and bool(str(raw.get("basis", "")).strip())
            gaps.append(
                {
                    "title": str(raw["title"]).strip(),
                    "description": str(raw.get("description", "")).strip(),
                    "category": EVIDENCE if supported else SPECULATIVE,
                    "basis": str(raw.get("basis", "")).strip(),
                    "source_ids": list(dict.fromkeys(sources)),
                }
            )

        directions: list[dict] = []
        for raw in data.get("future_directions") or []:
            if not isinstance(raw, dict) or not str(raw.get("direction", "")).strip():
                continue
            directions.append(
                {
                    "direction": str(raw["direction"]).strip(),
                    "rationale": str(raw.get("rationale", "")).strip(),
                    "category": SPECULATIVE,
                    "source_ids": list(dict.fromkeys(s for s in as_str_list(raw.get("source_ids")) if s in valid_ids)),
                }
            )
        return {"gaps": gaps, "future_directions": directions}
