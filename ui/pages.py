"""Streamlit pages. Business logic lives in agents/, rag/, services/ and database/; these only present it."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import streamlit as st

from agents.report import extract_title
from agents.supervisor import STAGE_LABELS, STAGES, ResearchError, ResearchSupervisor
from services.citations import render_citations
from config import Settings
from database.history import HistoryDB
from rag.ingestion import IngestionError, parse_pdf_bytes, validate_upload
from rag.vector_store import RAGEngine
from services.academic_search import AcademicSearchError, Paper, search_arxiv
from services.llm import LLMClient
from services.report_export import build_docx, build_pdf
from ui.components import (
    badge,
    card,
    check_list_html,
    empty_state,
    esc,
    page_header,
    section_label,
    stage_list_html,
    status_badge,
)

logger = logging.getLogger(__name__)

EXAMPLE_QUESTION = "What are the limitations of current Retrieval-Augmented Generation systems?"


@dataclass
class AppContext:
    settings: Settings
    llm: LLMClient
    rag: RAGEngine
    db: HistoryDB
    supervisor: ResearchSupervisor


# ---------------------------------------------------------------------- helpers

def _fmt_date(value: str | None) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).strftime("%d %b %Y, %H:%M")
    except ValueError:
        return value


def _open_session(session_id: str) -> None:
    st.session_state["open_session"] = session_id


def _go_to(page: str) -> None:
    st.session_state["nav"] = page
    st.session_state.pop("open_session", None)


@st.cache_data(show_spinner=False, max_entries=64)
def _precheck_pdf(name: str, data: bytes, max_mb: int) -> dict:
    """Validate an upload before the run so the user sees problems immediately."""
    try:
        validate_upload(name, data, max_mb)
        doc = parse_pdf_bytes(data, name)
        return {"ok": True, "pages": doc.page_count, "chars": doc.char_count, "message": ""}
    except IngestionError as exc:
        return {"ok": False, "pages": 0, "chars": 0, "message": str(exc)}


@st.cache_data(show_spinner=False, max_entries=32)
def _pdf_bytes(markdown: str, title: str, generated_on: str) -> bytes:
    return build_pdf(markdown, title, generated_on)


@st.cache_data(show_spinner=False, max_entries=32)
def _docx_bytes(markdown: str, title: str, generated_on: str) -> bytes:
    return build_docx(markdown, title, generated_on)


TEXT_KEYS = {"statement", "summary", "synthesis", "description", "basis", "rationale", "objective",
             "methodology", "direction", "topic"}


def _with_rendered_citations(session: dict) -> dict:
    """Return a copy of the session where every LLM text shows citations in the chosen style."""
    registry = session["registry"] or []
    style = session["citation_style"]
    order = session.get("citation_order") or []

    def render(value: str) -> str:
        return render_citations(value, registry, style, order)[0] if value else value

    def walk(value):
        if isinstance(value, dict):
            out = {k: render(v) if k in TEXT_KEYS and isinstance(v, str) else walk(v) for k, v in value.items()}
            if isinstance(value.get("source_ids"), list):  # citation label in the report's own numbering
                out["cites"] = render(" ".join(f"[{s}]" for s in value["source_ids"]))
            return out
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value

    rendered = dict(session)
    for key in ("analysis", "summaries", "gaps"):
        rendered[key] = walk(session.get(key) or {})
    return rendered


def _source_label(entry: dict) -> str:
    return f"{entry['source_id']} · {entry['title']} ({entry['year']})"


def _authors_text(authors: list[str], limit: int = 4) -> str:
    if not authors:
        return "Authors unknown"
    shown = ", ".join(authors[:limit])
    return shown + (" et al." if len(authors) > limit else "")


# ---------------------------------------------------------------------- overview

def render_overview(ctx: AppContext) -> None:
    page_header("Overview", "Your research sessions, processed papers and generated reports.")
    stats = ctx.db.stats()
    cols = st.columns(4)
    cols[0].metric("Sessions", stats["total_sessions"])
    cols[1].metric("Papers", stats["papers_processed"], help="Distinct documents used in any session")
    cols[2].metric("Reports", stats["reports_generated"])
    cols[3].metric("Completed", stats["completed_sessions"])

    # A callback runs before the rerun, so it may set the sidebar radio's key; an inline call may not.
    st.button("New research", type="primary", key="overview_new", on_click=_go_to, args=("New Research",))

    section_label("Recent research")
    sessions = ctx.db.list_sessions(limit=5)
    if not sessions:
        empty_state("No research yet", "Start a research session to analyse your papers and generate a cited report.")
        return
    for session in sessions:
        _session_row(session, key_prefix="overview")


def _session_row(session: dict, key_prefix: str) -> None:
    quality = (
        f"Quality {session['quality_score']}/100" if session.get("quality_score") is not None else "No quality score"
    )
    with st.container(border=False):
        left, right = st.columns([6, 1.3])
        with left:
            st.markdown(
                f'<div class="rp-card"><div class="rp-card-title">{esc(session["question"])}</div>'
                f'<div class="rp-meta">{esc(_fmt_date(session["created_at"]))} · {esc(session["citation_style"])} · '
                f'{session.get("document_count", 0)} sources · {esc(quality)}</div>'
                f'<div style="margin-top:0.5rem">{status_badge(session["status"])}</div></div>',
                unsafe_allow_html=True,
            )
        with right:
            st.write("")
            if session.get("has_report"):
                st.button("Open", key=f"{key_prefix}_{session['session_id']}", on_click=_open_session,
                          args=(session["session_id"],), use_container_width=True)


# ------------------------------------------------------------------ new research

def render_new_research(ctx: AppContext) -> None:
    page_header("New research", "Ask a research question, add papers, and let the agents build a cited report.")

    if not ctx.llm.is_configured:
        st.warning(ctx.settings.llm_setup_message, icon="⚙️")

    question = st.text_area(
        "Research question",
        value=st.session_state.get("question_draft", ""),
        placeholder=EXAMPLE_QUESTION,
        height=96,
        key="question",
        help="Be specific. The planner splits this into subtasks and retrieves evidence for each.",
    )

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        section_label("Your papers (PDF)")
        uploads = st.file_uploader(
            "Upload research PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="uploads",
            label_visibility="collapsed",
        )
        if uploads:
            for upload in uploads:
                info = _precheck_pdf(upload.name, upload.getvalue(), ctx.settings.max_upload_mb)
                if info["ok"]:
                    st.caption(f"✓ {upload.name} · {info['pages']} pages")
                else:
                    st.caption(f"✕ {upload.name} · {info['message']}")
        else:
            st.caption("Drop one or more PDFs. Scanned PDFs without a text layer cannot be analysed.")

    with right:
        section_label("Academic search (arXiv)")
        query_col, button_col = st.columns([2.6, 1])
        with query_col:
            arxiv_query = st.text_input("Search arXiv", key="arxiv_query", placeholder="e.g. retrieval augmented generation",
                                        label_visibility="collapsed")
        with button_col:
            search_clicked = st.button("Search", use_container_width=True, key="arxiv_search")
        if search_clicked and arxiv_query.strip():
            try:
                with st.spinner("Searching arXiv…"):
                    st.session_state["arxiv_results"] = search_arxiv(arxiv_query, ctx.settings.arxiv_max_results)
                st.session_state["arxiv_error"] = ""
            except AcademicSearchError as exc:
                st.session_state["arxiv_error"] = str(exc)
        if st.session_state.get("arxiv_error"):
            st.warning(st.session_state["arxiv_error"], icon="🌐")
        _render_arxiv_results(st.session_state.get("arxiv_results", []))

    section_label("Options")
    opt_left, opt_right = st.columns([1, 1.2])
    with opt_left:
        citation_style = st.radio("Citation style", ["IEEE", "APA"], horizontal=True, key="citation_style")
    with opt_right:
        top_k = st.slider(
            "Passages to analyse",
            min_value=4,
            max_value=30,
            value=12,
            key="top_k",
            help="Number of top-ranked passages retrieved per subtask (scaled by the number of subtasks).",
        )

    selected = _selected_papers()
    uploads = uploads or []
    total_sources = len(uploads) + len(selected)
    can_start = bool(question.strip()) and total_sources > 0 and ctx.llm.is_configured
    if not question.strip():
        st.caption("Enter a research question to begin.")
    elif total_sources == 0:
        st.caption("Add at least one PDF or select an arXiv paper.")

    if st.button(
        f"Start research · {total_sources} source{'s' if total_sources != 1 else ''}",
        type="primary",
        disabled=not can_start,
        key="start_research",
    ):
        _run_research(ctx, question.strip(), citation_style, top_k, uploads, selected)

    if st.session_state.get("last_run_error"):
        st.error(st.session_state["last_run_error"], icon="⚠️")


def _render_arxiv_results(results: list[Paper]) -> None:
    if not results:
        empty_state("Search arXiv", "Find open-access preprints and select them as sources.")
        return
    for paper in results:
        key = f"pick_{paper.arxiv_id}"
        with st.container(border=False):
            st.markdown(
                f'<div class="rp-card" style="margin-bottom:0.4rem"><div class="rp-card-title">{esc(paper.title)}</div>'
                f'<div class="rp-meta">{esc(_authors_text(paper.authors))} · {esc(paper.year)} · arXiv:{esc(paper.arxiv_id)}</div>'
                f'<div class="rp-meta" style="margin-top:0.35rem">{esc(paper.abstract[:260])}'
                f'{"…" if len(paper.abstract) > 260 else ""}</div></div>',
                unsafe_allow_html=True,
            )
            st.checkbox("Include in this research", key=key, value=st.session_state.get(key, False))


def _selected_papers() -> list[Paper]:
    return [p for p in st.session_state.get("arxiv_results", []) if st.session_state.get(f"pick_{p.arxiv_id}")]


def _run_research(ctx: AppContext, question: str, style: str, top_k: int, uploads, selected: list[Paper]) -> None:
    inputs = [{"kind": "upload", "name": f.name, "data": f.getvalue()} for f in uploads]
    inputs += [{"kind": "arxiv", "paper": paper} for paper in selected]
    st.session_state.pop("last_run_error", None)

    statuses: dict[str, str] = {key: "pending" for key, _, _ in STAGES}
    messages: dict[str, str] = {}
    index_of = {key: i for i, (key, _, _) in enumerate(STAGES)}

    section_label("Execution")
    progress_bar = st.progress(0.0, text="Starting agents…")
    board = st.empty()

    def render() -> None:
        board.markdown(stage_list_html(STAGES, statuses, messages, index_of), unsafe_allow_html=True)

    def progress(stage: str, status: str, message: str) -> None:
        statuses[stage] = status
        if message:
            messages[stage] = message
        done = sum(1 for s in statuses.values() if s == "completed")
        running = [STAGE_LABELS[k] for k, v in statuses.items() if v == "running"]
        progress_bar.progress(done / len(STAGES), text=f"{running[0]}…" if running else "Working…")
        render()

    render()
    try:
        with st.spinner(""):
            session = ctx.supervisor.run(question, style, top_k, inputs, progress=progress)
    except ResearchError as exc:
        progress_bar.progress(1.0, text="Stopped")
        st.session_state["last_run_error"] = (
            f"Research stopped at “{STAGE_LABELS.get(exc.stage, exc.stage)}”: {exc}. "
            "Partial results are saved in Research History."
        )
        logger.error("Research run failed: %s", exc)
        st.rerun()
        return
    progress_bar.progress(1.0, text="Completed")
    st.session_state["question_draft"] = question
    st.success("Research complete. The report is ready.", icon="✅")
    st.button("Open research report", type="primary", key="open_result",
              on_click=_open_session, args=(session["session_id"],))


# ------------------------------------------------------------------ results

def render_session_results(ctx: AppContext, session_id: str) -> None:
    session = ctx.db.get_session(session_id)
    if session is None:
        st.error("This research session no longer exists.")
        return
    if session["status"] != "completed":
        page_header("Research session", session["question"])
        st.error(session.get("error") or "This session did not finish.", icon="⚠️")
        return

    markdown = session["report_markdown"] or ""
    session = _with_rendered_citations(session)
    title = extract_title(markdown, session["question"])
    generated = _fmt_date(session["completed_at"])
    registry = session["registry"] or []
    analysis = session["analysis"] or {}
    summaries = session["summaries"] or {}
    gaps = session["gaps"] or {}
    quality = session["quality"] or {}

    head_left, head_right = st.columns([3.2, 1.5])
    with head_left:
        st.button("← Back to history", key="back_history", on_click=_go_to, args=("Research History",))
        page_header(title, session["question"])
        meta_badges = " ".join(
            [
                badge(session["citation_style"], "pending"),
                status_badge(session["status"]),
                badge(f"{len(registry)} sources", "pending"),
                badge(f"Quality {quality.get('score', '—')}/100", "ok" if (quality.get("score") or 0) >= 80 else "warn"),
            ]
        )
        st.markdown(f"{meta_badges}<div class='rp-meta' style='margin-top:0.6rem'>Generated {esc(generated)}</div>",
                    unsafe_allow_html=True)
    with head_right:
        st.write("")
        st.write("")
        slug = "research-report"
        st.download_button("Download PDF", _pdf_bytes(markdown, title, generated), file_name=f"{slug}.pdf",
                           mime="application/pdf", type="primary", use_container_width=True,
                           key=f"pdf_{session_id}")
        st.download_button("Download DOCX", _docx_bytes(markdown, title, generated), file_name=f"{slug}.docx",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                           use_container_width=True, key=f"docx_{session_id}")
        st.download_button("Download Markdown", markdown.encode("utf-8"), file_name=f"{slug}.md", mime="text/markdown",
                           use_container_width=True, key=f"md_{session_id}")

    tabs = st.tabs(["Overview", "Paper Summaries", "Comparative Analysis", "Research Gaps", "References",
                    "Quality Review", "Final Report"])

    with tabs[0]:
        _tab_overview(ctx, session)
    with tabs[1]:
        _tab_summaries(registry, analysis, summaries)
    with tabs[2]:
        _tab_comparison(registry, analysis, summaries)
    with tabs[3]:
        _tab_gaps(registry, gaps)
    with tabs[4]:
        _tab_references(markdown)
    with tabs[5]:
        _tab_quality(quality)
    with tabs[6]:
        st.markdown(markdown)


def _tab_overview(ctx: AppContext, session: dict) -> None:
    subtasks = (session.get("subtasks") or {}).get("subtasks", [])
    quality = session.get("quality") or {}
    cols = st.columns(4)
    cols[0].metric("Sources", len(session["registry"] or []))
    cols[1].metric("Chunks indexed", session.get("num_chunks") or 0)
    cols[2].metric("Subtasks", len(subtasks))
    cols[3].metric("Quality (of 100)", quality.get("score", "—"))

    synthesis = (session.get("summaries") or {}).get("synthesis", "")
    section_label("Synthesis")
    if synthesis:
        st.markdown(synthesis)
    else:
        empty_state("No synthesis", "The summarisation agent did not produce a cross-paper synthesis.")

    section_label("Research subtasks")
    for item in subtasks:
        card(item["title"], f"<div class='rp-meta'>{esc(item.get('description', ''))}</div>",
             meta=f"Query: {item['retrieval_query']}")

    if session.get("skipped"):
        with st.expander(f"Source notes ({len(session['skipped'])})"):
            for issue in session["skipped"]:
                level = {"error": "✕", "warning": "!", "info": "i"}.get(issue["level"], "•")
                st.markdown(f"**{level} {issue['name']}** — {issue['message']}")

    with st.expander("Agent processing log"):
        events = ctx.db.session_events(session["session_id"])
        if events:
            st.dataframe(pd.DataFrame(events)[["created_at", "agent", "status", "message"]],
                         hide_index=True, use_container_width=True)
        else:
            st.caption("No events recorded.")


def _tab_summaries(registry: list[dict], analysis: dict, summaries: dict) -> None:
    by_id = {e["source_id"]: e for e in registry}
    analysis_by_id = {p["source_id"]: p for p in analysis.get("papers", [])}
    summary_by_id = {s["source_id"]: s["summary"] for s in summaries.get("paper_summaries", [])}
    if not registry:
        empty_state("No sources", "No source documents were used in this session.")
        return
    for index, entry in enumerate(registry):
        sid = entry["source_id"]
        label = f"{sid} · {entry['title']}"
        with st.expander(label, expanded=index == 0):
            kind = "Uploaded document" if entry["source_type"] == "upload" else (
                "arXiv abstract only" if entry["source_type"] == "abstract" else "arXiv preprint")
            st.markdown(
                f"<div class='rp-meta'>{esc(_authors_text(entry['authors']))} · {esc(entry['year'])} · "
                f"{esc(kind)} · {entry['page_count']} pages</div>",
                unsafe_allow_html=True,
            )
            if sid in summary_by_id:
                st.markdown(summary_by_id[sid])
            else:
                st.caption("No summary was generated for this source.")
            detail = analysis_by_id.get(sid)
            if detail:
                if detail.get("objective"):
                    st.markdown(f"**Objective.** {detail['objective']}")
                if detail.get("methodology"):
                    st.markdown(f"**Methodology.** {detail['methodology']}")
                for heading, key in (("Key findings", "findings"), ("Limitations", "limitations")):
                    if detail.get(key):
                        st.markdown(f"**{heading}**")
                        for item in detail[key]:
                            cites = item.get("cites", "")
                            st.markdown(f"- {item['statement']} {cites}")
            else:
                st.caption("This source was not represented in the retrieved evidence.")


def _tab_comparison(registry: list[dict], analysis: dict, summaries: dict) -> None:
    by_id = {e["source_id"]: e for e in registry}

    def cite_list(ids: list[str]) -> str:
        return ", ".join(f"{i}" for i in ids)

    comparisons = analysis.get("comparisons", [])
    section_label("Agreements, contradictions and differences")
    if comparisons:
        for item in comparisons:
            tone = {"agreement": "ok", "contradiction": "fail", "difference": "warn"}.get(item["type"], "pending")
            st.markdown(
                f"{badge(item['type'].title(), tone)} **{esc(item.get('topic', ''))}** "
                f"<span class='rp-meta'>{esc(item.get('cites', ''))}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(item["statement"])
            st.write("")
    else:
        st.caption("No cross-paper comparisons were supported by the retrieved passages.")

    similarities = summaries.get("similarities", [])
    differences = summaries.get("differences", [])
    left, right = st.columns(2)
    with left:
        section_label("Similarities")
        for item in similarities or []:
            st.markdown(f"- {item['statement']} {item.get('cites', '')}")
        if not similarities:
            st.caption("None identified.")
    with right:
        section_label("Differences")
        for item in differences or []:
            st.markdown(f"- {item['statement']} {item.get('cites', '')}")
        if not differences:
            st.caption("None identified.")

    section_label("Method comparison")
    rows = []
    for paper in analysis.get("papers", []):
        entry = by_id.get(paper["source_id"])
        if entry:
            rows.append({"Source": paper["source_id"], "Title": entry["title"], "Objective": paper.get("objective", ""),
                         "Methodology": paper.get("methodology", "")})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _tab_gaps(registry: list[dict], gaps: dict) -> None:
    section_label("Evidence-supported gaps")
    evidence_gaps = [g for g in gaps.get("gaps", []) if g["category"] == "evidence_supported"]
    if evidence_gaps:
        for gap in evidence_gaps:
            body = f"<div>{esc(gap.get('description', ''))}</div>" \
                   f"<div class='rp-meta' style='margin-top:0.4rem'>Basis: {esc(gap.get('basis', ''))}</div>"
            sources = " · ".join(_source_label(next(e for e in registry if e["source_id"] == s))
                                 for s in gap["source_ids"] if any(e["source_id"] == s for e in registry))
            card(gap["title"], body + f"<div class='rp-meta' style='margin-top:0.4rem'>{esc(sources)}</div>")
    else:
        empty_state("No evidence-supported gaps", "The reviewed sources did not report limitations that yield a gap.")

    section_label("Speculative future directions")
    st.caption("These go beyond the reviewed sources and are not evidence-backed.")
    directions = gaps.get("future_directions", [])
    if directions:
        for item in directions:
            card(item["direction"], f"<div>{esc(item.get('rationale', ''))}</div>",
                 meta="Speculative")
    else:
        st.caption("No speculative directions were proposed.")


def _tab_references(markdown: str) -> None:
    section = markdown.split("\n## References", 1)
    if len(section) < 2:
        st.caption("No references were generated.")
        return
    entries = [chunk.strip() for chunk in section[1].split("\n\n") if chunk.strip()]
    section_label("References")
    for entry in entries:
        st.markdown(f"<div class='rp-reference'>{esc(entry)}</div>", unsafe_allow_html=True)


def _tab_quality(quality: dict) -> None:
    if not quality:
        empty_state("No quality review", "The quality review did not run for this session.")
        return
    left, right = st.columns([1, 2.4])
    with left:
        st.markdown(f"<div class='rp-meta'>Score</div><div class='rp-score'>{quality.get('score', '—')}/100</div>",
                    unsafe_allow_html=True)
        tone = "ok" if quality.get("verdict") == "Passed" else ("warn" if "Passed" in quality.get("verdict", "") or "Partially" in quality.get("verdict", "") else "fail")
        st.markdown(badge(quality.get("verdict", ""), tone), unsafe_allow_html=True)
        st.caption(f"Claims checked: {quality.get('claims_checked', 0)} · claim check: {quality.get('claim_check_status', '')}")
    with right:
        st.markdown(check_list_html(quality.get("checks", [])), unsafe_allow_html=True)

    if quality.get("unsupported_claims"):
        with st.expander(f"Unsupported claims ({len(quality['unsupported_claims'])})", expanded=True):
            for claim in quality["unsupported_claims"]:
                st.markdown(f"- {claim}")
    if quality.get("unverifiable_claims"):
        with st.expander(f"Claims that could not be verified ({len(quality['unverifiable_claims'])})"):
            for claim in quality["unverifiable_claims"]:
                st.markdown(f"- {claim}")
    if quality.get("invalid_citations"):
        with st.expander(f"Invalid citations ({len(quality['invalid_citations'])})"):
            for issue in quality["invalid_citations"]:
                st.markdown(f"- {issue}")


# ------------------------------------------------------------------ history

def render_history(ctx: AppContext) -> None:
    page_header("Research history", "Every session is stored locally in SQLite. Open any completed report.")
    sessions = ctx.db.list_sessions(limit=200)
    if not sessions:
        empty_state("Nothing here yet", "Completed and interrupted sessions appear here.")
        return
    for session in sessions:
        _session_row(session, key_prefix="history")
        if not session.get("has_report") and session.get("error"):
            st.caption(f"Stopped: {session['error']}")


# ------------------------------------------------------------------ saved papers

def render_saved_papers(ctx: AppContext) -> None:
    page_header("Saved papers", "The document library. Each paper is embedded once and reused across sessions.")
    documents = ctx.db.list_documents()
    if not documents:
        empty_state("Your library is empty", "PDFs and arXiv papers appear here after a research session uses them.")
        return
    rows = []
    for doc in documents:
        kind = {"upload": "Uploaded", "arxiv": "arXiv PDF", "abstract": "arXiv abstract"}.get(doc["source_type"], doc["source_type"])
        rows.append(
            {
                "Title": doc["title"] or doc["document_name"],
                "Authors": _authors_text(doc["authors"], limit=3),
                "Year": doc["year"] or "n.d.",
                "Type": kind,
                "Pages": doc["page_count"],
                "Sessions": doc["session_count"],
                "Added": _fmt_date(doc["created_at"]),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(f"{len(documents)} documents in library.")


# ------------------------------------------------------------------ settings

def render_settings(ctx: AppContext) -> None:
    settings = ctx.settings
    page_header("Settings", "Configuration status, connectivity checks and storage locations.")

    section_label("Language model")
    if ctx.llm.is_configured:
        st.markdown(f"{badge('Configured', 'ok')} Provider **{settings.llm_provider}**, model **{settings.llm_model}**",
                    unsafe_allow_html=True)
        if st.button("Test LLM connection", key="test_llm"):
            try:
                with st.spinner("Contacting the model…"):
                    reply = ctx.llm.complete("connection_test", "Reply with exactly one word.", "Reply with the word OK.")
                st.success(f"The model replied: {reply.strip()[:80]}")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
    else:
        st.markdown(badge("Not configured", "fail"), unsafe_allow_html=True)
        st.warning(settings.llm_setup_message)

    section_label("Embeddings and vector store")
    st.markdown(
        f"Model **{settings.embedding_model}** · chunks of ~{settings.chunk_size_words} words with "
        f"{settings.chunk_overlap_words} overlap · ChromaDB at `{settings.chroma_dir}`",
    )
    if st.button("Load embedding model", key="test_embedder", help="Downloads the model on first use"):
        try:
            with st.spinner("Loading all-MiniLM-L6-v2…"):
                ctx.rag.embedder.embed_query("connectivity check")
            st.success("Embedding model is ready.")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    section_label("Academic search")
    if st.button("Check arXiv connectivity", key="test_arxiv"):
        try:
            found = search_arxiv("retrieval augmented generation", max_results=1)
            st.success(f"arXiv responded ({len(found)} result).")
        except AcademicSearchError as exc:
            st.error(str(exc))

    section_label("Configuration reference")
    st.markdown(
        """
| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `gemini` (default) or `openai` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` | API key for the selected provider |
| `LLM_MODEL` | Optional model override |
| `EMBEDDING_MODEL` | Sentence-Transformers model (default `all-MiniLM-L6-v2`) |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | Chunking window (default 600 / 90 words) |
| `MAX_UPLOAD_MB` | Upload size limit (default 50) |
| `MAX_REVISIONS` | Quality-driven report revisions, at most 2 |
        """
    )
