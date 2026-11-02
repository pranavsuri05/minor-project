import io
import zipfile

import fitz
from docx import Document

from services.report_export import build_docx, build_pdf, markdown_to_blocks

MARKDOWN = """# Test Report Title

## Abstract

This abstract has **bold** and *italic* text with a symbol → arrow [1, p. 2].

## 3. Literature Review

Paragraph one with a claim [1, p. 2].

- First bullet
- Second bullet

## References

[1] A. Lovelace, "Dense Retrieval Survey," arXiv, 2021. [Online]. Available: https://arxiv.org/abs/1

[2] "Uploaded Notes," uploaded document (notes.pdf), 2 pages, n.d.
"""


def test_markdown_blocks_are_classified():
    kinds = [(b.kind, b.text) for b in markdown_to_blocks(MARKDOWN)]
    assert kinds[0] == ("title", "Test Report Title")
    assert ("h2", "Abstract") in kinds
    assert sum(1 for k, _ in kinds if k == "bullet") == 2
    assert sum(1 for k, _ in kinds if k == "reference") == 2


def test_pdf_export_is_valid_and_contains_content():
    data = build_pdf(MARKDOWN, "Test Report Title", "2026-11-02")
    assert data.startswith(b"%PDF-")
    pdf = fitz.open(stream=data, filetype="pdf")
    text = "".join(page.get_text() for page in pdf)
    pdf.close()
    assert "Test Report Title" in text
    assert "Literature Review" in text and "References" in text
    assert "Page 1" in text


def test_docx_export_is_valid_and_has_headings():
    data = build_docx(MARKDOWN, "Test Report Title", "2026-11-02")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert "word/document.xml" in archive.namelist()
    document = Document(io.BytesIO(data))
    headings = [p.text for p in document.paragraphs if p.style.name.startswith("Heading")]
    assert "Abstract" in headings and "3. Literature Review" in headings
    bullets = [p for p in document.paragraphs if p.style.name == "List Bullet"]
    assert len(bullets) == 2
