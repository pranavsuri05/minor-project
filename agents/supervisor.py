"""Supervisor / Orchestrator: a LangGraph StateGraph that coordinates the specialised agents.

Flow:
    ingest -> plan -> retrieve -> analyse -> summarise -> gaps -> report -> quality
                                                             ^              |
                                                             |  (revise,    |
                                                             +--- at most 2 times, if the
                                                                  quality review fails)
    quality -> finalise -> END

Every stage reports real start/finish events through `progress`, and every session is saved to SQLite.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.analysis import SemanticAnalysisAgent
from agents.planning import PlanningAgent
from agents.quality import QualityAgent
from agents.report import ReportAgent, attach_pages
from agents.research_gap import ResearchGapAgent
from agents.retrieval import LiteratureRetrievalAgent
from agents.summarisation import SummarisationAgent
from database.history import HistoryDB
from rag.vector_store import RAGEngine
from services.llm import LLMClient

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str, str], None]  # (stage, status, message)

STAGES: list[tuple[str, str, str]] = [
    ("ingest", "Document Ingestion", "Extract PDF text with page numbers, chunk and embed it."),
    ("plan", "Planning Agent", "Break the research question into subtasks and retrieval queries."),
    ("retrieve", "Literature Retrieval (RAG)", "Semantic search over the indexed chunks for each subtask."),
    ("analyse", "Semantic Analysis Agent", "Extract objectives, methods, findings and limitations."),
    ("summarise", "Summarisation Agent", "Summarise each paper and synthesise similarities and differences."),
    ("gaps", "Research Gap Agent", "Find evidence-backed gaps and label speculative directions."),
    ("report", "Report Generation Agent", "Write the structured report from the evidence."),
    ("quality", "Quality Review Agent", "Validate citations, sections and claims against evidence."),
    ("finalise", "Finalisation", "Render citations in the chosen style and build references."),
]
STAGE_LABELS = {key: label for key, label, _ in STAGES}


class ResearchError(RuntimeError):
    """A research stage failed. `stage` names the failing stage for the UI."""

    def __init__(self, stage: str, message: str, session_id: str) -> None:
        super().__init__(message)
        self.stage = stage
        self.session_id = session_id


class ResearchState(TypedDict, total=False):
    session_id: str
    question: str
    citation_style: str
    top_k: int
    inputs: list[dict[str, Any]]
    registry: list[dict]
    issues: list[dict]
    chunk_count: int
    planning: dict
    evidence: list[dict]
    analysis: dict
    summaries: dict
    gaps: dict
    report_body: str
    quality: dict
    revision_feedback: str
    revision_count: int
    final: dict


class ResearchSupervisor:
    def __init__(self, llm: LLMClient, rag: RAGEngine, db: HistoryDB, max_revisions: int = 2,
                 max_upload_mb: int = 50) -> None:
        self.llm = llm
        self.rag = rag
        self.db = db
        self.max_revisions = min(max_revisions, 2)  # hard bound, never more than two revisions
        self.retrieval = LiteratureRetrievalAgent(rag, max_upload_mb=max_upload_mb)
        self.planner = PlanningAgent(llm)
        self.analyst = SemanticAnalysisAgent(llm)
        self.summariser = SummarisationAgent(llm)
        self.gap_agent = ResearchGapAgent(llm)
        self.report_agent = ReportAgent(llm)
        self.quality_agent = QualityAgent(llm, max_revisions=self.max_revisions)

    # ------------------------------------------------------------------ graph

    def build_graph(self, emit: Callable[[str, str, str], None]):
        """Compile the LangGraph for one research session. `emit` receives every stage event."""

        def guarded(stage: str, work: Callable[[ResearchState], dict]) -> Callable[[ResearchState], dict]:
            label = STAGE_LABELS[stage]

            def node(state: ResearchState) -> dict:
                emit(stage, "running", "")
                try:
                    update = work(state)
                except ResearchError:
                    raise
                except Exception as exc:  # noqa: BLE001 - every stage failure is reported with its name
                    logger.exception("Stage %s failed", stage)
                    emit(stage, "failed", str(exc))
                    raise ResearchError(stage, f"{label} failed: {exc}", state.get("session_id", "")) from exc
                emit(stage, "completed", update.pop("_message", ""))
                return update

            return node

        def ingest(state: ResearchState) -> dict:
            result = self.retrieval.ingest_sources(state["inputs"])
            for entry in result.registry:
                self.db.upsert_document(entry)
                self.db.link_session_document(state["session_id"], entry["document_id"])
            skipped = sum(1 for i in result.issues if i["level"] == "error")
            return {
                "registry": result.registry,
                "issues": result.issues,
                "chunk_count": result.chunk_count,
                "_message": f"{len(result.registry)} sources indexed, {result.chunk_count} chunks"
                            + (f", {skipped} skipped" if skipped else ""),
            }

        def plan(state: ResearchState) -> dict:
            planning = self.planner.plan(state["question"])
            titles = "; ".join(s["title"] for s in planning["subtasks"])
            return {"planning": planning, "_message": f"{len(planning['subtasks'])} subtasks: {titles}"}

        def retrieve(state: ResearchState) -> dict:
            evidence = self.retrieval.retrieve_evidence(
                state["question"], state["planning"]["subtasks"], state["registry"], state["top_k"]
            )
            if not evidence:
                raise ValueError("No relevant passages were retrieved from the uploaded sources. "
                                 "Try a question closer to the content of the papers.")
            return {"evidence": evidence, "_message": f"{len(evidence)} relevant passages retrieved"}

        def analyse(state: ResearchState) -> dict:
            analysis = self.analyst.analyse(
                state["question"], state["planning"]["subtasks"], state["registry"], state["evidence"]
            )
            return {"analysis": analysis,
                    "_message": f"{len(analysis['papers'])} papers analysed, {len(analysis['comparisons'])} comparisons"}

        def summarise(state: ResearchState) -> dict:
            summaries = self.summariser.summarise(state["question"], state["registry"], state["analysis"])
            return {"summaries": summaries, "_message": f"{len(summaries['paper_summaries'])} paper summaries"}

        def identify_gaps(state: ResearchState) -> dict:
            gaps = self.gap_agent.identify(state["question"], state["registry"], state["analysis"], state["summaries"])
            supported = sum(1 for g in gaps["gaps"] if g["category"] == "evidence_supported")
            return {"gaps": gaps, "_message": f"{supported} evidence-supported gaps, "
                                              f"{len(gaps['future_directions'])} speculative directions"}

        def write_report(state: ResearchState) -> dict:
            revision = state.get("revision_count", 0)
            body = self.report_agent.write(
                state["question"], state["registry"], state["analysis"], state["summaries"], state["gaps"],
                feedback=state.get("revision_feedback", ""), evidence=state.get("evidence"),
            )
            body, attached = attach_pages(body, state["evidence"], self.rag.embedder)
            message = "Report drafted" if revision == 0 else f"Report revised (revision {revision})"
            if attached:
                message += f"; {attached} page references attached from retrieved passages"
            return {"report_body": body, "_message": message}

        def review_quality(state: ResearchState) -> dict:
            quality = self.quality_agent.review(state["report_body"], state["registry"], state["evidence"])
            return {
                "quality": quality,
                "revision_feedback": quality["revision_feedback"],
                "_message": f"{quality['verdict']} (score {quality['score']}/100)",
            }

        def revise(state: ResearchState) -> dict:
            """Bounded revision: fetch extra evidence for unsupported claims, then rewrite the report."""
            revision = state.get("revision_count", 0) + 1
            queries = state["quality"].get("follow_up_queries", [])
            evidence = state["evidence"]
            if queries:
                evidence = self.retrieval.retrieve_additional(queries, state["registry"], evidence)
            return {
                "evidence": evidence,
                "revision_count": revision,
                "_message": f"Revision {revision}: {len(queries)} follow-up retrieval queries",
            }

        def finalise(state: ResearchState) -> dict:
            quality = state["quality"]
            notes = list(quality.get("warnings", []))
            if quality.get("needs_revision"):
                notes.insert(0, "The review still found problems after "
                                f"{state.get('revision_count', 0)} revision(s); they are listed here and must be checked by the author.")
            rendered = self.report_agent.finalise(state["report_body"], state["registry"],
                                                  state["citation_style"], notes)
            return {"final": rendered, "_message": f"{len(rendered['cited_source_ids'])} references generated"}

        graph = StateGraph(ResearchState)
        graph.add_node("ingest", guarded("ingest", ingest))
        graph.add_node("plan", guarded("plan", plan))
        graph.add_node("retrieve", guarded("retrieve", retrieve))
        graph.add_node("analyse", guarded("analyse", analyse))
        graph.add_node("summarise", guarded("summarise", summarise))
        graph.add_node("gaps", guarded("gaps", identify_gaps))
        graph.add_node("report", guarded("report", write_report))
        graph.add_node("quality", guarded("quality", review_quality))
        graph.add_node("revise", guarded("report", revise))
        graph.add_node("finalise", guarded("finalise", finalise))

        graph.add_edge(START, "ingest")
        graph.add_edge("ingest", "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "analyse")
        graph.add_edge("analyse", "summarise")
        graph.add_edge("summarise", "gaps")
        graph.add_edge("gaps", "report")
        graph.add_edge("report", "quality")
        graph.add_conditional_edges(
            "quality",
            self._route_after_quality,
            {"revise": "revise", "finalise": "finalise"},
        )
        graph.add_edge("revise", "report")
        graph.add_edge("finalise", END)
        return graph.compile()

    def _route_after_quality(self, state: ResearchState) -> str:
        needs_revision = state.get("quality", {}).get("needs_revision", False)
        if needs_revision and state.get("revision_count", 0) < self.max_revisions:
            return "revise"
        return "finalise"

    # ------------------------------------------------------------------ run

    def run(
        self,
        question: str,
        citation_style: str,
        top_k: int,
        inputs: list[dict[str, Any]],
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline and persist the session. Returns the stored session dict."""
        session_id = uuid.uuid4().hex[:12]
        notify = progress or (lambda stage, status, message: None)

        def emit(stage: str, status: str, message: str = "") -> None:
            self.db.record_event(session_id, stage, status, message)
            notify(stage, status, message)

        self.db.create_session(session_id, question, citation_style, top_k)
        graph = self.build_graph(emit)
        initial: ResearchState = {
            "session_id": session_id,
            "question": question.strip(),
            "citation_style": citation_style,
            "top_k": top_k,
            "inputs": inputs,
            "revision_count": 0,
            "revision_feedback": "",
        }
        logger.info("Research session %s started (%d inputs, style=%s)", session_id, len(inputs), citation_style)
        try:
            state: ResearchState = graph.invoke(initial)
        except ResearchError as exc:
            self.db.finish_session(session_id, "failed", error=str(exc))
            logger.error("Session %s failed at %s: %s", session_id, exc.stage, exc)
            raise
        except Exception as exc:  # noqa: BLE001
            self.db.finish_session(session_id, "failed", error=str(exc))
            logger.exception("Session %s failed unexpectedly", session_id)
            raise ResearchError("supervisor", f"Unexpected error: {exc}", session_id) from exc

        final = state["final"]
        self.db.finish_session(
            session_id,
            "completed",
            registry=state["registry"],
            skipped=state.get("issues", []),
            subtasks=state["planning"],
            analysis=state["analysis"],
            summaries=state["summaries"],
            gaps=state["gaps"],
            quality=state["quality"],
            citation_order=final["cited_source_ids"],
            report_markdown=final["markdown"],
            num_chunks=state.get("chunk_count", 0),
        )
        logger.info("Research session %s completed", session_id)
        return self.db.get_session(session_id)
