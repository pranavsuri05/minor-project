"""Planning Agent: decomposes the research question into subtasks with focused retrieval queries."""

from __future__ import annotations

import logging

from agents.common import as_str_list
from services.llm import LLMClient, LLMOutputError

logger = logging.getLogger(__name__)

MAX_SUBTASKS = 5

SYSTEM_PROMPT = """You are the Planning Agent in a multi-agent scientific literature review system.
Given a research question, identify the research dimensions it touches and split it into 3 to 5
meaningful subtasks. For each subtask write a focused retrieval query: a short phrase (6-14 words)
that would retrieve matching passages from research papers with semantic search. Do not use
numbering in the queries, and do not invent facts.

Return JSON with this shape:
{
  "research_dimensions": ["..."],
  "subtasks": [
    {"title": "short label", "description": "one sentence on what to find", "retrieval_query": "focused phrase"}
  ]
}"""


class PlanningAgent:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def plan(self, question: str) -> dict:
        """Return {"research_dimensions": [...], "subtasks": [...]}; each subtask has title, description, retrieval_query."""
        if not question.strip():
            raise ValueError("The research question is empty.")
        data = self.llm.complete_json("planning", SYSTEM_PROMPT, f"Research question: {question.strip()}")
        subtasks = self._validate_subtasks(data)
        dimensions = as_str_list(data.get("research_dimensions")) if isinstance(data, dict) else []
        logger.info("Planning produced %d subtasks", len(subtasks))
        return {"research_dimensions": dimensions, "subtasks": subtasks}

    @staticmethod
    def _validate_subtasks(data) -> list[dict]:
        if not isinstance(data, dict) or not isinstance(data.get("subtasks"), list):
            raise LLMOutputError("The planning response did not contain a 'subtasks' list.")
        subtasks: list[dict] = []
        for raw in data["subtasks"]:
            if not isinstance(raw, dict):
                continue
            query = str(raw.get("retrieval_query", "")).strip()
            title = str(raw.get("title", "")).strip() or query[:60]
            if not query:
                continue
            subtasks.append(
                {
                    "title": title,
                    "description": str(raw.get("description", "")).strip(),
                    "retrieval_query": query,
                }
            )
        if not subtasks:
            raise LLMOutputError("The planning response contained no usable retrieval queries.")
        return subtasks[:MAX_SUBTASKS]
