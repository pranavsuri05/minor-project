"""arXiv search and PDF download. The rest of the app works without this module's network access."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
MAX_PDF_BYTES = 50 * 1024 * 1024


class AcademicSearchError(RuntimeError):
    """arXiv could not be queried. The UI reports this and keeps uploaded PDFs usable."""


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    published: str = ""  # ISO date from arXiv, e.g. 2023-05-01T...
    year: str = "n.d."
    url: str = ""
    pdf_url: str = ""
    source: str = "arxiv"

    @property
    def key(self) -> str:
        return self.arxiv_id


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def parse_arxiv_feed(xml_text: str) -> list[Paper]:
    """Parse an arXiv Atom feed into Paper objects. Raises AcademicSearchError on malformed XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise AcademicSearchError(f"arXiv returned an unreadable response: {exc}") from exc

    papers: list[Paper] = []
    for entry in root.findall("a:entry", ATOM_NS):
        entry_id = _clean(entry.findtext("a:id", default="", namespaces=ATOM_NS))
        if not entry_id:
            continue
        arxiv_id = re.sub(r"v\d+$", "", entry_id.rsplit("/abs/", 1)[-1])
        published = _clean(entry.findtext("a:published", default="", namespaces=ATOM_NS))
        pdf_url = ""
        for link in entry.findall("a:link", ATOM_NS):
            if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                pdf_url = link.get("href", "")
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_clean(entry.findtext("a:title", default="", namespaces=ATOM_NS)),
                authors=[
                    _clean(author.findtext("a:name", default="", namespaces=ATOM_NS))
                    for author in entry.findall("a:author", ATOM_NS)
                ],
                abstract=_clean(entry.findtext("a:summary", default="", namespaces=ATOM_NS)),
                published=published,
                year=published[:4] if published[:4].isdigit() else "n.d.",
                url=f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}",
            )
        )
    return papers


def search_arxiv(query: str, max_results: int = 8, timeout: float = 15.0) -> list[Paper]:
    """Search arXiv by relevance. Raises AcademicSearchError when the service is unreachable."""
    if not query.strip():
        return []
    params = {
        "search_query": f"all:{query.strip()}",
        "start": 0,
        "max_results": max(1, min(max_results, 25)),
        "sortBy": "relevance",
    }
    try:
        response = httpx.get(ARXIV_API, params=params, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("arXiv search failed: %s", exc)
        raise AcademicSearchError(f"arXiv is unavailable right now ({exc.__class__.__name__}). "
                                  "You can still analyse uploaded PDFs.") from exc
    return parse_arxiv_feed(response.text)


def download_pdf(url: str, timeout: float = 30.0) -> bytes:
    """Download a PDF. Raises AcademicSearchError if the response is not a PDF or is too large."""
    try:
        with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as response:
            response.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_bytes():
                total += len(chunk)
                if total > MAX_PDF_BYTES:
                    raise AcademicSearchError("The PDF is larger than the 50 MB limit.")
                chunks.append(chunk)
    except httpx.HTTPError as exc:
        raise AcademicSearchError(f"Could not download the PDF ({exc.__class__.__name__}).") from exc
    data = b"".join(chunks)
    if not data.lstrip()[:8].startswith(b"%PDF-"):
        raise AcademicSearchError("The download did not return a PDF file.")
    return data
