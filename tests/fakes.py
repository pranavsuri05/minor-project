"""Deterministic test doubles: a scripted LLM and a hashing embedder (no model download, no network)."""

from __future__ import annotations

import json
import re
import zlib
from typing import Any, Callable

import fitz

from services.llm import parse_json_text

DIM = 256


class HashEmbedder:
    """Bag-of-words hashing into a fixed vector. Texts sharing words get high cosine similarity."""

    model_name = "hash-test"

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0] * DIM
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[zlib.crc32(token.encode()) % DIM] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class FakeLLM:
    """Scripted LLM. `responses` maps a task name to a value, a callable(user_prompt) or a list of values."""

    is_configured = True
    model_label = "fake:test"

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def _value(self, task: str, user: str) -> Any:
        self.calls.append(task)
        if task not in self.responses:
            raise AssertionError(f"Unexpected LLM task: {task}")
        value = self.responses[task]
        if isinstance(value, list):
            value = value.pop(0) if len(value) > 1 else value[0]
        if callable(value):
            value = value(user)
        if isinstance(value, Exception):
            raise value
        return value

    def complete(self, task: str, system: str, user: str) -> str:
        value = self._value(task, user)
        return value if isinstance(value, str) else json.dumps(value)

    def complete_json(self, task: str, system: str, user: str) -> Any:
        value = self._value(task, user)
        return parse_json_text(value) if isinstance(value, str) else value


def make_pdf(pages: list[str], title: str = "", author: str = "", creation: str = "") -> bytes:
    """Create a real PDF with one text block per page."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=9)
    meta = {"title": title, "author": author, "creationDate": creation}
    doc.set_metadata({k: v for k, v in meta.items() if v})
    data = doc.tobytes()
    doc.close()
    return data


def source_ids_in(text: str) -> list[str]:
    """Source ids named in a prompt, either as evidence headers ("### S1") or registry lines ("S1: ...")."""
    found = set(re.findall(r"### (S\d+)", text)) | set(re.findall(r"^(S\d+):", text, re.MULTILINE))
    return sorted(found, key=lambda s: int(s[1:]))
