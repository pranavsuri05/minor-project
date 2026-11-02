import pytest

from rag.embeddings import EmbeddingUnavailableError, SentenceTransformerEmbedder
from rag.ingestion import parse_pdf_bytes
from tests.fakes import make_pdf

RAG_TEXT = ("Retrieval augmented generation grounds language model answers in retrieved passages from an index. "
            "Dense retrievers embed queries and documents with a shared encoder for semantic similarity search. ")
CHEM_TEXT = ("Catalytic hydrogenation reduces alkenes using palladium catalysts under moderate pressure. "
             "Reaction kinetics depend on temperature, solvent polarity, and the surface area of the metal. ")


def test_index_and_retrieve_returns_traceable_chunks(rag):
    rag_doc = parse_pdf_bytes(make_pdf([RAG_TEXT * 3]), "rag.pdf")
    chem_doc = parse_pdf_bytes(make_pdf([CHEM_TEXT * 3]), "chem.pdf")
    rag.index_document(rag_doc)
    rag.index_document(chem_doc)

    hits = rag.retrieve("retrieval augmented generation passages", [rag_doc.document_id], top_k=3)
    assert hits, "expected at least one hit"
    assert all(h.document_id == rag_doc.document_id for h in hits)
    assert all(h.document_name == "rag.pdf" and h.page_number == 1 for h in hits)
    assert hits[0].score >= hits[-1].score


def test_retrieval_is_restricted_to_session_documents(rag):
    rag_doc = parse_pdf_bytes(make_pdf([RAG_TEXT * 3]), "rag.pdf")
    chem_doc = parse_pdf_bytes(make_pdf([CHEM_TEXT * 3]), "chem.pdf")
    rag.index_document(rag_doc)
    rag.index_document(chem_doc)
    hits = rag.retrieve("palladium catalysts hydrogenation", [rag_doc.document_id], top_k=5)
    assert all(h.document_id == rag_doc.document_id for h in hits)
    assert rag.retrieve("anything", [], top_k=5) == []


def test_reindexing_reuses_embeddings(rag):
    doc = parse_pdf_bytes(make_pdf([RAG_TEXT * 3]), "rag.pdf")
    _, first = rag.index_document(doc)
    chunks, second = rag.index_document(doc)
    assert first is True and second is False
    assert chunks and chunks[0].document_id == doc.document_id


def test_changed_text_replaces_stale_chunks(rag):
    from dataclasses import replace

    from rag.ingestion import PageText

    doc = parse_pdf_bytes(make_pdf([RAG_TEXT * 3]), "rag.pdf")
    rag.index_document(doc)
    # Same file identity, but text extraction now yields different text (for example references removed).
    changed = replace(doc, pages=[PageText(1, "Entirely different extracted content about catalytic palladium hydrogenation. " * 8)])
    chunks, newly_indexed = rag.index_document(changed)
    assert newly_indexed is True
    assert rag.store._collection.count() == len(chunks)  # only the new chunks remain
    hits = rag.retrieve("retrieval augmented generation passages", [doc.document_id], top_k=10)
    assert all("retrieval augmented" not in h.text.lower() for h in hits)


def test_per_document_coverage_returns_each_document(rag):
    a = parse_pdf_bytes(make_pdf([RAG_TEXT * 3]), "a.pdf")
    b = parse_pdf_bytes(make_pdf([CHEM_TEXT * 3]), "b.pdf")
    rag.index_document(a)
    rag.index_document(b)
    hits = rag.retrieve_per_document("retrieval passages and catalysts", [a.document_id, b.document_id], k=1)
    assert {h.document_id for h in hits} == {a.document_id, b.document_id}


def test_embedding_model_failure_is_reported_not_faked(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    embedder = SentenceTransformerEmbedder("definitely-not-a-real-model-xyz")
    with pytest.raises(EmbeddingUnavailableError, match="could not be loaded"):
        embedder.embed_query("hello")
