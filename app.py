"""Streamlit entry point: streamlit run app.py"""

from __future__ import annotations

import streamlit as st

from config import configure_logging, get_settings
from database.history import HistoryDB
from rag.embeddings import get_embedder
from rag.vector_store import RAGEngine, VectorStore
from services.llm import LLMClient
from agents.supervisor import ResearchSupervisor
from ui.components import brand_header
from ui.pages import (
    AppContext,
    render_history,
    render_new_research,
    render_overview,
    render_saved_papers,
    render_session_results,
    render_settings,
)
from ui.styles import MODES, build_css

NAV_PAGES = ["Overview", "New Research", "Research History", "Saved Papers", "Settings"]


@st.cache_resource(show_spinner=False)
def get_context() -> AppContext:
    """One set of shared services per server process (LLM client, vector store, database, supervisor)."""
    settings = get_settings()
    configure_logging(settings.log_dir)
    db = HistoryDB(settings.db_path)
    db.mark_interrupted_sessions()
    llm = LLMClient(settings)
    store = VectorStore(settings.chroma_dir)
    rag = RAGEngine(store, get_embedder(settings.embedding_model), settings.chunk_size_words,
                    settings.chunk_overlap_words)
    supervisor = ResearchSupervisor(llm, rag, db, max_revisions=settings.max_revisions,
                                    max_upload_mb=settings.max_upload_mb)
    return AppContext(settings=settings, llm=llm, rag=rag, db=db, supervisor=supervisor)


def _clear_open_session() -> None:
    st.session_state.pop("open_session", None)


def main() -> None:
    st.set_page_config(page_title="Research Platform", page_icon="🔬", layout="wide",
                       initial_sidebar_state="expanded")
    if "nav" not in st.session_state:
        st.session_state["nav"] = "Overview"

    with st.sidebar:
        brand_header()
        st.radio("Navigate", NAV_PAGES, key="nav", label_visibility="collapsed", on_change=_clear_open_session)
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.radio("Appearance", MODES, horizontal=True, key="appearance")

    st.markdown(build_css(st.session_state.get("appearance", "Light")), unsafe_allow_html=True)

    ctx = get_context()
    if st.session_state.get("open_session"):
        render_session_results(ctx, st.session_state["open_session"])
        return

    page = st.session_state["nav"]
    if page == "Overview":
        render_overview(ctx)
    elif page == "New Research":
        render_new_research(ctx)
    elif page == "Research History":
        render_history(ctx)
    elif page == "Saved Papers":
        render_saved_papers(ctx)
    elif page == "Settings":
        render_settings(ctx)


main()
