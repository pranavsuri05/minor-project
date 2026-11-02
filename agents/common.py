"""Helpers shared by the agents: prompt formatting and source-id enforcement."""

from __future__ import annotations

from typing import Any


def format_registry(registry: list[dict]) -> str:
    """One line per source so the model knows which ids exist and what they are."""
    lines = []
    for entry in registry:
        authors = ", ".join(entry["authors"][:4]) or "unknown authors"
        lines.append(
            f"{entry['source_id']}: \"{entry['title']}\" by {authors} ({entry['year']}); "
            f"type={entry['source_type']}; {entry['page_count']} pages"
        )
    return "\n".join(lines) or "(no sources)"


def format_evidence(evidence: list[dict], registry: list[dict] | None = None) -> str:
    """Group passages by source. Each passage header carries the exact citation marker to reuse."""
    titles = {e["source_id"]: e["title"] for e in (registry or [])}
    groups: dict[str, list[dict]] = {}
    for item in evidence:
        groups.setdefault(item["source_id"], []).append(item)
    blocks = []
    for source_id in sorted(groups, key=lambda s: int(s[1:])):
        header = f"### {source_id}" + (f" — {titles[source_id]}" if source_id in titles else "")
        passages = []
        for item in sorted(groups[source_id], key=lambda i: (i["page_number"], i["chunk_index"])):
            page = item["page_number"]
            page_label = f"[{source_id}, p. {page}]"
            passages.append(f"{page_label} {item['text']}")
        blocks.append(header + "\n" + "\n\n".join(passages))
    return "\n\n".join(blocks) or "(no evidence retrieved)"


def keep_valid_sources(items: list[Any], valid_ids: set[str], key: str = "source_ids") -> tuple[list[dict], int]:
    """Drop citations to unknown ids. Items left without any valid source are removed entirely.

    Returns (kept_items, number_of_items_dropped). This is how the pipeline guarantees that every
    finding traces back to a source in the registry.
    """
    kept: list[dict] = []
    dropped = 0
    for item in items or []:
        if not isinstance(item, dict):
            dropped += 1
            continue
        sources = item.get(key) or []
        if isinstance(sources, str):
            sources = [sources]
        valid = [s for s in sources if isinstance(s, str) and s in valid_ids]
        if not valid:
            dropped += 1
            continue
        item = dict(item)
        item[key] = list(dict.fromkeys(valid))
        kept.append(item)
    return kept, dropped


def as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []
