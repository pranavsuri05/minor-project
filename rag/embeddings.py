"""Embedding interface. Production uses Sentence Transformers; tests inject a deterministic embedder."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """The embedding model could not be loaded. Shown to the user; retrieval never falls back silently."""


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers. The model is loaded lazily, once, and then reused."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self.model_name)
                    self._model = SentenceTransformer(self.model_name)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Embedding model %s failed to load", self.model_name)
                    raise EmbeddingUnavailableError(
                        f"The embedding model '{self.model_name}' could not be loaded. On the first run it is "
                        "downloaded from Hugging Face, so check your internet connection (or pre-download it). "
                        f"Details: {exc}"
                    ) from exc
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32)
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


_shared: dict[str, SentenceTransformerEmbedder] = {}


def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    """Process-wide embedder so the (large) model is loaded once per model name."""
    if model_name not in _shared:
        _shared[model_name] = SentenceTransformerEmbedder(model_name)
    return _shared[model_name]
