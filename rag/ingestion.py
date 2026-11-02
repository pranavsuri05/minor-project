"""PDF ingestion with PyMuPDF. Preserves page numbers and gives each document a stable identity."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
MIN_EXTRACTABLE_CHARS = 200  # below this a PDF is treated as empty or scanned (no text layer)


class IngestionError(ValueError):
    """A document cannot be used. The message is shown to the user as-is."""


@dataclass
class PageText:
    page_number: int  # 1-based, matches the PDF viewer
    text: str


@dataclass
class ParsedDocument:
    document_id: str
    document_name: str
    file_hash: str
    pages: list[PageText]
    metadata: dict[str, str] = field(default_factory=dict)
    source_type: str = "upload"  # "upload" | "arxiv" | "abstract"

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document_id_for(data: bytes) -> str:
    """Identity is the content hash, so the same PDF uploaded twice maps to one document."""
    return file_sha256(data)[:16]


def validate_upload(name: str, data: bytes, max_mb: int) -> None:
    """Cheap checks run before any parsing. Raises IngestionError with a user-facing message."""
    if not name.lower().endswith(".pdf"):
        raise IngestionError(f"'{name}' is not a PDF file. Only .pdf documents are supported.")
    if not data:
        raise IngestionError(f"'{name}' is empty (0 bytes).")
    if len(data) > max_mb * 1024 * 1024:
        raise IngestionError(f"'{name}' is larger than the {max_mb} MB upload limit.")
    if not data.lstrip()[:8].startswith(PDF_MAGIC):
        raise IngestionError(f"'{name}' does not start with a PDF header, so it is probably corrupted or renamed.")


_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


_REFERENCES_HEADING_RE = re.compile(r"^\s*(references|bibliography|reference list)\s*$", re.IGNORECASE | re.MULTILINE)


def _blank_reference_list(pages: list[PageText]) -> list[PageText]:
    """Empty the reference list so bibliography entries are never retrieved as evidence.

    A heading only counts if the text after it contains several years, which separates a real
    bibliography from a body-text word. Page numbers are kept, so the page count still matches the PDF.
    Anything after the reference list (including appendices) is dropped with it.
    """
    for index in range(1, len(pages)):
        match = _REFERENCES_HEADING_RE.search(pages[index].text)
        if not match:
            continue
        following = pages[index].text[match.end():] + " " + " ".join(p.text for p in pages[index + 1 :])
        if len(re.findall(r"\b(?:19|20)\d{2}\b", following[:3000])) < 3:
            continue
        blanked = [PageText(p.page_number, p.text) for p in pages[:index]]
        blanked.append(PageText(pages[index].page_number, pages[index].text[: match.start()].strip()))
        blanked += [PageText(p.page_number, "") for p in pages[index + 1 :]]
        return blanked
    return pages


def parse_pdf_bytes(data: bytes, document_name: str) -> ParsedDocument:
    """Extract per-page text. Raises IngestionError for corrupted, encrypted, empty or scanned PDFs."""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - PyMuPDF raises several exception types
        logger.warning("Could not open %s: %s", document_name, exc)
        raise IngestionError(f"'{document_name}' could not be read. The file may be corrupted.") from exc

    try:
        if doc.needs_pass:
            raise IngestionError(f"'{document_name}' is password-protected. Remove the password and upload again.")
        if doc.page_count == 0:
            raise IngestionError(f"'{document_name}' has no pages.")

        pages: list[PageText] = []
        for index in range(doc.page_count):
            text = _clean(doc.load_page(index).get_text("text"))
            pages.append(PageText(page_number=index + 1, text=text))

        metadata = {k: v for k, v in (doc.metadata or {}).items() if v}
    except IngestionError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Text extraction failed for %s", document_name)
        raise IngestionError(f"Text extraction failed for '{document_name}': {exc}") from exc
    finally:
        doc.close()

    pages = _blank_reference_list(pages)
    total_chars = sum(len(p.text) for p in pages)
    if total_chars == 0:
        raise IngestionError(
            f"'{document_name}' contains no extractable text. It may be a scanned PDF without a text layer."
        )
    if total_chars < MIN_EXTRACTABLE_CHARS:
        raise IngestionError(
            f"'{document_name}' has only {total_chars} characters of text. "
            "It is probably empty or scanned, so it cannot be analysed."
        )

    logger.info("Parsed %s: %d pages, %d chars", document_name, len(pages), total_chars)
    return ParsedDocument(
        document_id=document_id_for(data),
        document_name=document_name,
        file_hash=file_sha256(data),
        pages=pages,
        metadata=metadata,
        source_type="upload",
    )


def parse_pdf_file(path: Path) -> ParsedDocument:
    data = path.read_bytes()
    validate_upload(path.name, data, max_mb=10_000)
    return parse_pdf_bytes(data, path.name)


def parse_abstract(arxiv_id: str, title: str, abstract: str) -> ParsedDocument:
    """Fallback when an arXiv PDF cannot be downloaded: the abstract is indexed as page 1."""
    text = _clean(f"{title}. {abstract}")
    if len(text) < 50:
        raise IngestionError(f"arXiv entry {arxiv_id} has no usable abstract.")
    key = f"abstract:{arxiv_id}:{title}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return ParsedDocument(
        document_id=digest[:16],
        document_name=f"arXiv {arxiv_id}",
        file_hash=digest,
        pages=[PageText(page_number=1, text=text)],
        metadata={"title": title},
        source_type="abstract",
    )
