"""Sliding-window chunking that keeps exact page provenance for every chunk."""

from __future__ import annotations

from dataclasses import dataclass

from rag.ingestion import ParsedDocument


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int  # page where the chunk starts
    page_end: int  # page where the chunk ends (equal to page_number for most chunks)
    text: str
    chunk_index: int

    def to_metadata(self) -> dict:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_number": self.page_number,
            "page_end": self.page_end,
            "chunk_index": self.chunk_index,
        }


def chunk_document(doc: ParsedDocument, chunk_size_words: int = 600, overlap_words: int = 90) -> list[Chunk]:
    """Split a document into overlapping word windows.

    Each word remembers the page it came from, so a chunk that spans a page break still
    reports the correct first and last page. Word counts approximate LLM tokens (≈0.75 words/token).
    """
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if not 0 <= overlap_words < chunk_size_words:
        raise ValueError("overlap_words must be non-negative and smaller than chunk_size_words")

    tokens: list[tuple[str, int]] = []
    for page in doc.pages:
        tokens.extend((word, page.page_number) for word in page.text.split())
    if not tokens:
        return []

    step = chunk_size_words - overlap_words
    chunks: list[Chunk] = []
    start = 0
    while True:
        window = tokens[start : start + chunk_size_words]
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.document_id}-c{index:04d}",
                document_id=doc.document_id,
                document_name=doc.document_name,
                page_number=window[0][1],
                page_end=window[-1][1],
                text=" ".join(word for word, _ in window),
                chunk_index=index,
            )
        )
        if start + chunk_size_words >= len(tokens):
            break
        start += step
    return chunks
