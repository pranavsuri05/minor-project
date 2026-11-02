"""ChromaDB-backed vector store and the reusable RAG facade used by every retrieval agent.

Chunks from all documents live in one collection. Each document is embedded once (its id is
its content hash), and every query is filtered to the document ids of the current session,
so papers from different research sessions never mix.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from rag.chunking import Chunk
from rag.embeddings import Embedder
from rag.ingestion import ParsedDocument

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"
_BATCH = 128


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    page_end: int
    text: str
    score: float  # cosine similarity, higher is more relevant
    chunk_index: int


class VectorStore:
    def __init__(self, persist_dir: Path, collection_name: str = COLLECTION_NAME) -> None:
        import chromadb

        persist_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._client = chromadb.PersistentClient(path=str(persist_dir))
            # Embeddings are always supplied by our own Embedder, so Chroma's default ONNX model is never used.
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
                embedding_function=None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("ChromaDB could not open %s", persist_dir)
            raise RuntimeError(f"The ChromaDB vector store at {persist_dir} could not be opened: {exc}") from exc

    def has_document(self, document_id: str, text_hash: str | None = None) -> bool:
        """True if the document is indexed. With `text_hash`, only if its extracted text is unchanged."""
        where = {"document_id": document_id}
        if text_hash is not None:
            where = {"$and": [{"document_id": document_id}, {"text_hash": text_hash}]}
        return bool(self._collection.get(where=where, limit=1, include=[])["ids"])

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]], text_hash: str = "") -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start : start + _BATCH],
                documents=[c.text for c in batch],
                metadatas=[{**c.to_metadata(), "text_hash": text_hash} for c in batch],
            )

    def query(self, vector: list[float], document_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        if not document_ids or top_k <= 0 or self._collection.count() == 0:
            return []
        where = (
            {"document_id": document_ids[0]}
            if len(document_ids) == 1
            else {"document_id": {"$in": document_ids}}
        )
        result = self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[RetrievedChunk] = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            hits.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=meta["document_id"],
                    document_name=meta["document_name"],
                    page_number=int(meta["page_number"]),
                    page_end=int(meta["page_end"]),
                    text=text,
                    score=round(1.0 - float(distance), 4),
                    chunk_index=int(meta["chunk_index"]),
                )
            )
        return hits


class RAGEngine:
    """Reusable RAG facade: index documents, then retrieve passages for any query."""

    def __init__(self, store: VectorStore, embedder: Embedder, chunk_size_words: int, overlap_words: int) -> None:
        self.store = store
        self.embedder = embedder
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def index_document(self, doc: ParsedDocument) -> tuple[list[Chunk], bool]:
        """Chunk, embed and store a document. Returns (chunks, newly_indexed).

        A document that is already in the store is not re-embedded; its chunks are rebuilt in
        memory so callers always get the same chunk ids and metadata.
        """
        from rag.chunking import chunk_document

        chunks = chunk_document(doc, self.chunk_size_words, self.overlap_words)
        if not chunks:
            raise ValueError(f"Document {doc.document_name} produced no chunks")
        # The text hash makes a change in text extraction (or chunking) visible: stale chunks are replaced.
        text_hash = hashlib.sha256("\n".join(c.text for c in chunks).encode("utf-8")).hexdigest()[:16]
        if self.store.has_document(doc.document_id, text_hash):
            logger.info("Document %s already indexed; reusing embeddings", doc.document_name)
            return chunks, False
        self.store.delete_document(doc.document_id)  # no-op for a new document
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add_chunks(chunks, vectors, text_hash)
        logger.info("Indexed %s: %d chunks", doc.document_name, len(chunks))
        return chunks, True

    def retrieve(self, query: str, document_ids: list[str], top_k: int) -> list[RetrievedChunk]:
        """Top-k passages across the given documents for one query."""
        if not query.strip():
            return []
        vector = self.embedder.embed_query(query)
        return self.store.query(vector, document_ids, top_k)

    def retrieve_per_document(self, query: str, document_ids: list[str], k: int = 1) -> list[RetrievedChunk]:
        """Best k passages from each document, so every paper stays represented in the evidence."""
        if not query.strip():
            return []
        vector = self.embedder.embed_query(query)
        results: list[RetrievedChunk] = []
        for document_id in document_ids:
            results.extend(self.store.query(vector, [document_id], k))
        return results
