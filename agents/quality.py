"""Quality Review Agent: checks citations, sections and claims before a report is accepted.

Deterministic checks run first and cannot be talked out of a failure. The LLM claim check is
done against the retrieved passages only, and anything it cannot verify is flagged, not passed.
"""

from __future__ import annotations

import logging
import re

from agents.common import format_evidence
from agents.report import REQUIRED_SECTIONS
from services.citations import validate_citations
from services.llm import LLMClient, LLMOutputError

logger = logging.getLogger(__name__)

EVIDENCE_SECTIONS = ("Literature Review", "Comparative Analysis", "Research Gap Identification")
MAX_CLAIMS = 40
MAX_EVIDENCE_CHARS = 24_000

SYSTEM_PROMPT = """You are the Quality Review Agent. Check each factual claim in the report against the passages.
A claim is:
- "supported": the cited passages state it (or it follows directly from them).
- "unsupported": the claim states a fact, number, result or comparison that the passages do not show,
  or cites a source that does not say it.
- "unverifiable": the claim cannot be checked against the passages provided.
An expansion of an abbreviation that the passages do not give counts as "unsupported".
Ignore headings, the research question itself and clearly labelled speculation ("Speculative:").

Return JSON:
{"claims": [{"claim": "short quote or paraphrase", "source_ids": ["S1"], "status": "supported|unsupported|unverifiable", "reason": "one sentence"}]}"""


class QualityAgent:
    def __init__(self, llm: LLMClient, max_revisions: int = 2) -> None:
        self.llm = llm
        self.max_revisions = max_revisions

    def review(self, report_markdown: str, registry: list[dict], evidence: list[dict]) -> dict:
        checks: list[dict] = []

        # 1. Citation identifiers and page references must exist in the source registry.
        invalid = validate_citations(report_markdown, registry)
        checks.append(_check("citation_validity", "Citations map to real sources and pages", not invalid, invalid))

        # 2. Required sections must all be present.
        missing_sections = [s for s in REQUIRED_SECTIONS if not _has_heading(report_markdown, s)]
        checks.append(
            _check("required_sections", "All required report sections are present", not missing_sections,
                   [f"Missing section: {s}" for s in missing_sections])
        )

        # 3. Evidence-heavy paragraphs should carry at least one citation.
        uncited = _uncited_paragraphs(report_markdown)
        checks.append(
            _check("missing_citations", "Evidence paragraphs carry citations", not uncited,
                   [f"Uncited paragraph: {p[:120]}..." for p in uncited])
        )

        # 4. Claim verification against retrieved evidence (LLM-assisted).
        claim_status, claims, claim_error = self._verify_claims(report_markdown, registry, evidence)
        unsupported = [c for c in claims if c.get("status") == "unsupported"]
        unverifiable = [c for c in claims if c.get("status") == "unverifiable"]
        if claim_status == "completed":
            checks.append(
                _check("claim_verification", "Claims are supported by retrieved evidence", not unsupported,
                       [f"Unsupported: {c['claim'][:140]} ({c.get('reason', '')})" for c in unsupported])
            )
        else:
            checks.append(_check("claim_verification", "Claims are supported by retrieved evidence", False,
                                 [f"Claim verification did not run: {claim_error}"]))

        warnings = [f"Could not verify: {c['claim'][:140]}" for c in unverifiable]
        warnings += [d for c in checks if c["name"] == "missing_citations" for d in c["details"]]

        needs_revision = bool(invalid or missing_sections or unsupported)
        feedback = _revision_feedback(invalid, missing_sections, unsupported)
        follow_up_queries = [_query_from_claim(c["claim"]) for c in unsupported][:6]

        score = _score(invalid, missing_sections, unsupported, unverifiable, uncited, claim_status)
        if needs_revision:
            verdict = "Needs revision"
        elif claim_status != "completed":
            verdict = "Partially verified (claim check did not run)"
        elif warnings:
            verdict = "Passed with warnings"
        else:
            verdict = "Passed"
        result = {
            "score": score,
            "verdict": verdict,
            "checks": checks,
            "invalid_citations": invalid,
            "missing_sections": missing_sections,
            "unsupported_claims": [c["claim"] for c in unsupported],
            "unverifiable_claims": [c["claim"] for c in unverifiable],
            "warnings": warnings,
            "claim_check_status": claim_status,
            "claims_checked": len(claims),
            "needs_revision": needs_revision,
            "revision_feedback": feedback,
            "follow_up_queries": follow_up_queries,
        }
        logger.info("Quality review: score=%s verdict=%s", score, verdict)
        return result

    def _verify_claims(self, report: str, registry: list[dict], evidence: list[dict]) -> tuple[str, list[dict], str]:
        body = report.split("\n## References", 1)[0]
        # Share the budget between passages instead of cutting the text from the end, which would silently
        # drop the last source (sorted by id) and make its claims look unsupported.
        per_passage = max(600, MAX_EVIDENCE_CHARS // max(1, len(evidence)))
        trimmed = [{**item, "text": item["text"][:per_passage]} for item in evidence]
        evidence_text = format_evidence(trimmed, registry)
        user = f"Report:\n{body[:16_000]}\n\nRetrieved passages:\n{evidence_text}"
        raw_claims = None
        for attempt in (1, 2):  # the model occasionally answers in another shape; ask once more
            try:
                data = self.llm.complete_json("quality_claims", SYSTEM_PROMPT, user)
            except LLMOutputError as exc:
                logger.warning("Claim verification returned unusable output: %s", exc)
                return "not_run", [], str(exc)
            raw_claims = _extract_claims(data)
            if raw_claims is not None:
                break
            shape = list(data.keys()) if isinstance(data, dict) else type(data).__name__
            logger.warning("Claim verification attempt %d returned an unexpected shape: %s", attempt, shape)
        if raw_claims is None:
            return "not_run", [], "the reviewer returned no claim list"
        claims = []
        for raw in raw_claims[:MAX_CLAIMS]:
            if not isinstance(raw, dict) or not str(raw.get("claim", "")).strip():
                continue
            status = str(raw.get("status", "")).lower()
            if status not in {"supported", "unsupported", "unverifiable"}:
                status = "unverifiable"
            claims.append({"claim": str(raw["claim"]).strip(), "status": status,
                           "reason": str(raw.get("reason", "")).strip()})
        return "completed", claims, ""


def _extract_claims(data) -> list | None:
    """Accept {"claims": [...]}, a bare list, or a dict with exactly one list value."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("claims"), list):
            return data["claims"]
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    return None


def _check(name: str, label: str, passed: bool, details: list[str]) -> dict:
    return {"name": name, "label": label, "passed": passed, "details": details}


def _has_heading(markdown: str, section: str) -> bool:
    # Match on the heading text without its numbering, so "3. Literature Review" and "Literature Review" both count.
    bare = re.sub(r"^\d+\.\s*", "", section)
    return re.search(rf"^#{{1,3}}\s*(?:\d+\.\s*)?{re.escape(bare)}\s*$", markdown, re.MULTILINE | re.IGNORECASE) is not None


def _uncited_paragraphs(markdown: str) -> list[str]:
    """Paragraphs of 20+ words without a citation, inside the evidence-based sections only."""
    uncited: list[str] = []
    current = ""
    for block in re.split(r"\n\s*\n", markdown):
        heading = re.match(r"^#{1,3}\s*(.+)$", block.strip())
        if heading:
            current = re.sub(r"^\d+\.\s*", "", heading.group(1)).strip()
            continue
        if current not in EVIDENCE_SECTIONS:
            continue
        text = block.strip()
        if text.startswith(("-", "*", "|")) or len(text.split()) < 20:
            continue
        if not re.search(r"\[S\d+", text):
            uncited.append(text)
    return uncited


def _revision_feedback(invalid: list[str], missing: list[str], unsupported: list[dict]) -> str:
    lines: list[str] = []
    if invalid:
        lines.append("Invalid citations (use only registry ids and real page numbers):")
        lines += [f"- {issue}" for issue in invalid[:12]]
    if missing:
        lines.append("Missing sections: " + ", ".join(missing))
    if unsupported:
        lines.append("Rewrite these unsupported claims so they match the passages, or remove them:")
        lines += [f"- {c['claim'][:200]}" for c in unsupported[:12]]
    return "\n".join(lines)


def _query_from_claim(claim: str) -> str:
    words = re.sub(r"[^\w\s-]", " ", claim).split()
    return " ".join(words[:14])


def _score(invalid, missing_sections, unsupported, unverifiable, uncited, claim_status) -> int:
    score = 100
    score -= min(40, 10 * len(invalid))
    score -= min(25, 5 * len(missing_sections))
    score -= min(45, 15 * len(unsupported))
    score -= min(10, 2 * len(unverifiable))
    score -= min(10, len(uncited))
    if claim_status != "completed":
        score -= 10
    return max(0, score)
