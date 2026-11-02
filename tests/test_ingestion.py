import pytest

from rag.chunking import chunk_document
from rag.ingestion import IngestionError, document_id_for, parse_pdf_bytes, validate_upload
from tests.fakes import make_pdf

LONG = "Retrieval augmented generation combines a retriever with a generator for grounded answers. "


def test_page_numbers_and_text_are_preserved():
    data = make_pdf([LONG * 4 + "ALPHA marker", LONG * 4 + "BETA marker", LONG * 4 + "GAMMA marker"],
                    title="Sample Paper", author="A. Writer", creation="D:20230105120000")
    doc = parse_pdf_bytes(data, "sample.pdf")
    assert doc.page_count == 3
    assert [p.page_number for p in doc.pages] == [1, 2, 3]
    assert "ALPHA" in doc.pages[0].text and "BETA" not in doc.pages[0].text
    assert "GAMMA" in doc.pages[2].text
    assert doc.metadata["title"] == "Sample Paper"


def test_document_identity_is_content_hash():
    data = make_pdf([LONG * 4])
    assert parse_pdf_bytes(data, "a.pdf").document_id == document_id_for(data)
    assert document_id_for(data) != document_id_for(make_pdf([LONG * 4 + "x"]))


def test_blank_pdf_is_rejected_as_no_text():
    with pytest.raises(IngestionError, match="no extractable text|scanned"):
        parse_pdf_bytes(make_pdf([""]), "blank.pdf")


def test_very_short_pdf_is_rejected():
    with pytest.raises(IngestionError, match="characters of text"):
        parse_pdf_bytes(make_pdf(["Only a few words here."]), "short.pdf")


def test_corrupted_bytes_are_rejected_on_validation():
    with pytest.raises(IngestionError, match="PDF header"):
        validate_upload("broken.pdf", b"not a pdf at all", max_mb=5)


def test_corrupted_pdf_with_valid_header_is_rejected_on_parse():
    with pytest.raises(IngestionError, match="could not be read|corrupted"):
        parse_pdf_bytes(b"%PDF-1.4\ngarbage that is not a real document", "fake.pdf")


def test_wrong_extension_and_size_limit():
    with pytest.raises(IngestionError, match="not a PDF"):
        validate_upload("notes.txt", make_pdf([LONG * 4]), max_mb=5)
    with pytest.raises(IngestionError, match="empty"):
        validate_upload("empty.pdf", b"", max_mb=5)
    with pytest.raises(IngestionError, match="upload limit"):
        validate_upload("big.pdf", b"%PDF-" + b"0" * (2 * 1024 * 1024), max_mb=1)


def test_chunks_keep_metadata_and_page_provenance():
    words_a = " ".join(f"alpha{i}" for i in range(50))
    words_b = " ".join(f"beta{i}" for i in range(50))
    doc = parse_pdf_bytes(make_pdf([words_a + " " + words_a, words_b + " " + words_b]), "two.pdf")
    chunks = chunk_document(doc, chunk_size_words=40, overlap_words=8)
    assert len(chunks) >= 3
    assert all(c.document_id == doc.document_id and c.document_name == "two.pdf" for c in chunks)
    assert chunks[0].page_number == 1
    assert chunks[-1].page_end == 2
    # Overlap: the second chunk starts with words that the first chunk also contains.
    first_words = chunks[0].text.split()
    assert chunks[1].text.split()[0] in first_words
    # No chunk mixes content it should not have: page numbers stay within the document.
    assert all(1 <= c.page_number <= c.page_end <= doc.page_count for c in chunks)


def test_chunk_parameter_validation():
    doc = parse_pdf_bytes(make_pdf([LONG * 4]), "x.pdf")
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_size_words=10, overlap_words=10)
