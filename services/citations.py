"""Source registry, citation markers and IEEE/APA rendering.

Agents write evidence markers such as `[S2, p. 4]` that point at registry entries. Only this
module turns markers into formatted citations, and reference entries come from registry
metadata, never from model output, so references cannot be invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rag.ingestion import ParsedDocument
from services.academic_search import Paper

CITATION_STYLES = ("IEEE", "APA")

_GROUP_RE = re.compile(r"\[(S\d+(?:[^\[\]\n]*))\]")
_TOKEN_RE = re.compile(r"S(\d+)|p{1,2}\.?\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class CitationRef:
    source_id: str
    page: int | None


def make_source_entry(index: int, doc: ParsedDocument, paper: Paper | None = None) -> dict:
    """Build one registry entry. Uploaded PDFs use embedded metadata when present, otherwise the file name."""
    if paper is not None:
        title = paper.title or doc.document_name
        authors = [a for a in paper.authors if a]
        year = paper.year or "n.d."
        url = paper.url
        source_type = doc.source_type  # "arxiv" (full PDF) or "abstract" (fallback)
    else:
        meta = doc.metadata
        title = (meta.get("title") or "").strip() or _guess_title(doc) or Path(doc.document_name).stem.replace("_", " ").replace("-", " ")
        raw_author = (meta.get("author") or "").strip()
        authors = [a.strip() for a in re.split(r";|,| and ", raw_author) if a.strip()][:6]
        year = _year_from_pdf_date(meta.get("creationDate", "")) or "n.d."
        url = ""
        source_type = "upload"
    return {
        "source_id": f"S{index}",
        "document_id": doc.document_id,
        "document_name": doc.document_name,
        "title": title,
        "authors": authors,
        "year": year,
        "source_type": source_type,
        "url": url,
        "page_count": doc.page_count,
    }


_NOT_TITLE_RE = re.compile(r"journal|issn|arxiv|proceedings|https?://|www\.|copyright|©|preprint|doi", re.IGNORECASE)


def _guess_title(doc: ParsedDocument) -> str:
    """Best-effort title for PDFs without title metadata: the first plausible line of page 1."""
    if not doc.pages:
        return ""
    for line in doc.pages[0].text.split("\n"):
        line = line.strip()
        if 15 <= len(line) <= 200 and not line.isdigit() and not _NOT_TITLE_RE.search(line):
            return line
    return ""


def _year_from_pdf_date(value: str) -> str:
    match = re.search(r"D:(\d{4})", value or "")
    return match.group(1) if match else ""


# --------------------------------------------------------------------------- markers

def parse_citation_groups(text: str) -> list[tuple[int, int, list[CitationRef]]]:
    """Return (start, end, refs) for each bracket group that starts with a source id."""
    groups = []
    for match in _GROUP_RE.finditer(text):
        refs: list[CitationRef] = []
        for token in _TOKEN_RE.finditer(match.group(1)):
            if token.group(1):
                refs.append(CitationRef(source_id=f"S{token.group(1)}", page=None))
            elif refs:
                # A page number belongs to the most recent source in the group.
                if refs[-1].page is None:
                    refs[-1] = CitationRef(refs[-1].source_id, int(token.group(2)))
        if refs:
            groups.append((match.start(), match.end(), refs))
    return groups


def validate_citations(text: str, registry: list[dict]) -> list[str]:
    """Return human-readable problems: unknown source ids and out-of-range page numbers."""
    by_id = {entry["source_id"]: entry for entry in registry}
    issues: list[str] = []
    for _, _, refs in parse_citation_groups(text):
        for ref in refs:
            entry = by_id.get(ref.source_id)
            if entry is None:
                issues.append(f"Citation {ref.source_id} does not exist in the source registry.")
                continue
            if ref.page is not None and not 1 <= ref.page <= entry["page_count"]:
                issues.append(
                    f"Page {ref.page} is outside {ref.source_id} ({entry['document_name']}, "
                    f"{entry['page_count']} pages)."
                )
    return issues


def strip_invalid_citations(text: str, valid_ids: set[str]) -> tuple[str, int]:
    """Remove citation groups that point at unknown sources. Returns (clean_text, removed_count)."""
    removed = 0

    def drop_unknown(match: re.Match) -> str:
        nonlocal removed
        groups = parse_citation_groups(match.group(0))
        if not groups:
            return match.group(0)
        _, _, refs = groups[0]
        if all(ref.source_id in valid_ids for ref in refs):
            return match.group(0)
        removed += 1
        return ""

    cleaned = _GROUP_RE.sub(drop_unknown, text)
    return re.sub(r"[ \t]+([.,;])", r"\1", cleaned), removed


def cited_source_ids(text: str) -> list[str]:
    """Source ids in order of first citation, ignoring invalid ids."""
    order: list[str] = []
    for _, _, refs in parse_citation_groups(text):
        for ref in refs:
            if ref.source_id not in order:
                order.append(ref.source_id)
    return order


# --------------------------------------------------------------------------- rendering

def _initials(name: str) -> str:
    parts = name.split()
    if len(parts) <= 1:
        return ""
    return " ".join(f"{p[0]}." for p in parts[:-1])


def _last_name(name: str) -> str:
    parts = name.split()
    return parts[-1] if parts else name


def _apa_author(entry: dict) -> str:
    authors = entry["authors"]
    if not authors:
        title = entry["title"]
        return title if len(title) <= 60 else title[:57].rstrip() + "..."
    if len(authors) == 1:
        return _last_name(authors[0])
    if len(authors) == 2:
        return f"{_last_name(authors[0])} & {_last_name(authors[1])}"
    return f"{_last_name(authors[0])} et al."


def _ieee_author_list(entry: dict) -> str:
    authors = entry["authors"]
    if not authors:
        return ""
    formatted = [f"{_initials(a)} {_last_name(a)}".strip() for a in authors]
    if len(formatted) > 3:
        return f"{formatted[0]} et al."
    return ", ".join(formatted)


def render_citations(text: str, registry: list[dict], style: str,
                     order: list[str] | None = None) -> tuple[str, list[str]]:
    """Replace evidence markers with IEEE numbers or APA author-date citations.

    `order` fixes the IEEE numbering (pass the report's order so every section agrees). Sources not in
    `order` are numbered after it, by first appearance in `text`.
    Returns (rendered_text, full_order).
    """
    if style not in CITATION_STYLES:
        raise ValueError(f"Unknown citation style: {style}")
    by_id = {entry["source_id"]: entry for entry in registry}
    base = list(order or [])
    order = base + [sid for sid in cited_source_ids(text) if sid not in base]
    number_of = {sid: i + 1 for i, sid in enumerate(order)}

    def replace(match: re.Match) -> str:
        groups = parse_citation_groups(match.group(0))
        if not groups:
            return match.group(0)
        _, _, refs = groups[0]
        parts: list[str] = []
        for ref in refs:
            entry = by_id.get(ref.source_id)
            if entry is None:
                continue  # invalid markers are reported by the quality agent and dropped here
            page_suffix = f", p. {ref.page}" if ref.page is not None else ""
            if style == "IEEE":
                parts.append(f"[{number_of[ref.source_id]}{page_suffix}]")
            else:
                parts.append(f"{_apa_author(entry)}, {entry['year']}{page_suffix}")
        if not parts:
            return ""
        if style == "IEEE":
            return ", ".join(parts)
        return "(" + "; ".join(parts) + ")"

    rendered_text = _GROUP_RE.sub(replace, text)

    def mention(match: re.Match) -> str:
        """Prose such as "S1 and S2 differ" follows the same numbering as the bracketed citations."""
        entry = by_id.get(match.group(0))
        if entry is None:
            return match.group(0)
        if style == "IEEE":
            if entry["source_id"] not in number_of:
                order.append(entry["source_id"])
                number_of[entry["source_id"]] = len(order)
            return f"[{number_of[entry['source_id']]}]"
        return _apa_author(entry)

    rendered_text = re.sub(r"\bS\d+\b", mention, rendered_text)
    return rendered_text, order


def format_reference_list(registry: list[dict], cited_ids: list[str], style: str) -> list[str]:
    """One formatted entry per cited source, in citation order."""
    by_id = {entry["source_id"]: entry for entry in registry}
    lines: list[str] = []
    for number, sid in enumerate(cited_ids, start=1):
        entry = by_id[sid]
        if style == "IEEE":
            lines.append(f"[{number}] {_ieee_reference(entry)}")
        else:
            lines.append(_apa_reference(entry))
    return lines


def _year_text(year: str) -> str:
    """Year followed by one full stop, without doubling a stop that is already part of 'n.d.'."""
    return year if year.endswith(".") else f"{year}."


def _ieee_reference(entry: dict) -> str:
    authors = _ieee_author_list(entry)
    prefix = f"{authors}, " if authors else ""
    if entry["source_type"] == "upload":
        return (
            f"{prefix}\"{entry['title']},\" uploaded document ({entry['document_name']}), "
            f"{entry['page_count']} pages, {_year_text(entry['year'])}"
        )
    label = "arXiv abstract only" if entry["source_type"] == "abstract" else "arXiv preprint"
    url = f" [Online]. Available: {entry['url']}" if entry["url"] else ""
    return f"{prefix}\"{entry['title']},\" {label}, {_year_text(entry['year'])}{url}"


def _apa_reference(entry: dict) -> str:
    authors = entry["authors"]
    if authors:
        names = [f"{_last_name(a)}, {_initials(a)}".rstrip(", ") for a in authors[:20]]
        author_text = names[0] if len(names) == 1 else ", ".join(names[:-1]) + f", & {names[-1]}"
    else:
        author_text = ""
    if entry["source_type"] == "upload":
        tail = f"{entry['title']} [Uploaded document: {entry['document_name']}]."
    else:
        suffix = " (abstract only)" if entry["source_type"] == "abstract" else ""
        tail = f"{entry['title']}{suffix}. arXiv."
    link = f" {entry['url']}" if entry["url"] else ""
    head = f"{author_text} " if author_text else ""
    return f"{head}({entry['year']}). {tail}{link}".strip()
