"""Literature Retrieval Agent: ingests sources (uploads and arXiv) and retrieves evidence through RAG."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from rag.ingestion import IngestionError, ParsedDocument, parse_abstract, parse_pdf_bytes, validate_upload
from rag.vector_store import RAGEngine
from services.academic_search import AcademicSearchError, Paper, download_pdf
from services.citations import make_source_entry

logger = logging.getLogger(__name__)

MIN_RELEVANCE = 0.10  # cosine similarity below this is not treated as evidence for subtask queries


@dataclass
class IngestResult:
    registry: list[dict]
    issues: list[dict] = field(default_factory=list)  # {"name", "level": "error"|"warning"|"info", "message"}
    documents: list[ParsedDocument] = field(default_factory=list)
    chunk_count: int = 0


class LiteratureRetrievalAgent:
    def __init__(self, rag: RAGEngine, max_upload_mb: int = 50) -> None:
        self.rag = rag
        self.max_upload_mb = max_upload_mb

    # ------------------------------------------------------------------ ingestion

    def ingest_sources(self, inputs: list[dict[str, Any]]) -> IngestResult:
        """Parse, chunk and index each input. `inputs` items are either
        {"kind": "upload", "name": str, "data": bytes} or {"kind": "arxiv", "paper": Paper}.
        Bad files are reported and skipped; the rest of the session continues.
        """
        registry: list[dict] = []
        issues: list[dict] = []
        documents: list[ParsedDocument] = []
        seen: set[str] = set()
        chunk_count = 0

        for item in inputs:
            paper: Paper | None = item.get("paper") if item.get("kind") == "arxiv" else None
            name = paper.title if paper else item.get("name", "uploaded file")
            try:
                if paper is not None:
                    doc = self._load_arxiv(paper, issues)
                else:
                    validate_upload(item["name"], item["data"], self.max_upload_mb)
                    doc = parse_pdf_bytes(item["data"], item["name"])
            except IngestionError as exc:
                logger.warning("Skipping source %s: %s", name, exc)
                issues.append({"name": name, "level": "error", "message": str(exc)})
                continue

            if doc.document_id in seen:
                issues.append({"name": doc.document_name, "level": "info",
                               "message": "Duplicate of a source already in this session; it was ignored."})
                continue
            seen.add(doc.document_id)

            chunks, newly_indexed = self.rag.index_document(doc)
            chunk_count += len(chunks)
            if not newly_indexed:
                issues.append({"name": doc.document_name, "level": "info",
                               "message": "Already indexed from an earlier session; embeddings were reused."})
            entry = make_source_entry(len(registry) + 1, doc, paper)
            entry["char_count"] = doc.char_count
            entry["file_hash"] = doc.file_hash
            registry.append(entry)
            documents.append(doc)

        if not registry:
            raise IngestionError(
                "None of the sources could be used. " + " ".join(i["message"] for i in issues if i["level"] == "error")
            )
        logger.info("Ingested %d sources (%d chunks)", len(registry), chunk_count)
        return IngestResult(registry=registry, issues=issues, documents=documents, chunk_count=chunk_count)

    def _load_arxiv(self, paper: Paper, issues: list[dict]) -> ParsedDocument:
        """Try the full arXiv PDF; fall back to the abstract so the paper is still usable."""
        try:
            data = download_pdf(paper.pdf_url)
            doc = parse_pdf_bytes(data, f"arXiv-{paper.arxiv_id}.pdf")
            doc.source_type = "arxiv"
            return doc
        except (AcademicSearchError, IngestionError) as exc:
            logger.warning("Full text unavailable for arXiv:%s (%s); using abstract", paper.arxiv_id, exc)
            issues.append({"name": paper.title, "level": "warning",
                           "message": f"Full PDF unavailable ({exc}). The abstract was used instead."})
        return parse_abstract(paper.arxiv_id, paper.title, paper.abstract)

    # ------------------------------------------------------------------ retrieval

    def retrieve_evidence(
        self,
        question: str,
        subtasks: list[dict],
        registry: list[dict],
        top_k: int,
    ) -> list[dict]:
        """Subtask-driven RAG retrieval plus one guaranteed passage per document (for comparison)."""
        document_ids = [e["document_id"] for e in registry]
        per_query = max(1, math.ceil(top_k / max(1, len(subtasks))))
        evidence: dict[str, dict] = {}

        for subtask in subtasks:
            hits = self.rag.retrieve(subtask["retrieval_query"], document_ids, per_query)
            for hit in hits:
                if hit.score < MIN_RELEVANCE:
                    continue
                self._merge(evidence, hit, registry, subtask["title"])

        for hit in self.rag.retrieve_per_document(question, document_ids, k=1):
            self._merge(evidence, hit, registry, "Overall question (coverage)")

        ordered = sorted(evidence.values(), key=lambda e: e["score"], reverse=True)
        logger.info("Retrieved %d distinct passages for %d subtasks", len(ordered), len(subtasks))
        return ordered

    def retrieve_additional(
        self,
        queries: list[str],
        registry: list[dict],
        existing: list[dict],
        per_query: int = 3,
    ) -> list[dict]:
        """Extra retrieval requested by the Quality Agent for unsupported claims."""
        document_ids = [e["document_id"] for e in registry]
        evidence = {e["chunk_id"]: dict(e) for e in existing}
        for query in queries[:6]:
            for hit in self.rag.retrieve(query, document_ids, per_query):
                if hit.score >= MIN_RELEVANCE:
                    self._merge(evidence, hit, registry, "Quality review follow-up")
        return sorted(evidence.values(), key=lambda e: e["score"], reverse=True)

    @staticmethod
    def _merge(store: dict[str, dict], hit, registry: list[dict], label: str) -> None:
        source_id = next((e["source_id"] for e in registry if e["document_id"] == hit.document_id), None)
        if source_id is None:
            return
        existing = store.get(hit.chunk_id)
        if existing is None:
            store[hit.chunk_id] = {
                "chunk_id": hit.chunk_id,
                "source_id": source_id,
                "document_id": hit.document_id,
                "document_name": hit.document_name,
                "page_number": hit.page_number,
                "page_end": hit.page_end,
                "chunk_index": hit.chunk_index,
                "text": hit.text,
                "score": hit.score,
                "matched_subtasks": [label],
            }
            return
        existing["score"] = max(existing["score"], hit.score)
        if label not in existing["matched_subtasks"]:
            existing["matched_subtasks"].append(label)
