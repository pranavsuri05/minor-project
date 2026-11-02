"""End-to-end LangGraph pipeline with real PDF parsing, real ChromaDB and a scripted LLM."""

import re

import pytest

from agents.supervisor import STAGES, ResearchError, ResearchSupervisor
from services.llm import LLMError
from tests.fakes import FakeLLM, make_pdf, source_ids_in

RAG_PAPER = ("Retrieval augmented generation grounds answers in retrieved passages. A limitation is that retrieval "
             "accuracy drops for long documents, and the study did not evaluate multilingual corpora. ") * 3
SURVEY = ("Dense retrieval encoders improve recall on benchmark questions. The authors report that hallucination "
          "decreases when grounded passages are supplied, but context windows remain a constraint. ") * 3


def _planning(_user):
    return {"research_dimensions": ["retrieval", "generation"], "subtasks": [
        {"title": "Limitations", "description": "limits of RAG", "retrieval_query": "limitations of retrieval augmented generation"},
        {"title": "Recall", "description": "retrieval quality", "retrieval_query": "dense retrieval recall accuracy"},
        {"title": "Hallucination", "description": "grounding", "retrieval_query": "hallucination grounded passages"},
    ]}


def _analysis(user):
    sids = source_ids_in(user)
    papers = [{"source_id": s, "objective": "Study RAG", "methodology": "Retrieval plus generation",
               "contributions": [], "findings": [{"statement": f"{s} reports grounded answers", "source_ids": [s]}],
               "limitations": [{"statement": f"{s} reports a limitation", "source_ids": [s]}]} for s in sids]
    comparisons = []
    if len(sids) >= 2:
        comparisons = [{"type": "difference", "topic": "retrieval", "statement": "Methods differ",
                        "source_ids": sids[:2]}]
    return {"papers": papers, "comparisons": comparisons}


def _summaries(user):
    sids = source_ids_in(user)
    return {"paper_summaries": [{"source_id": s, "summary": f"Summary of {s} [{s}]."} for s in sids],
            "synthesis": "Across the sources " + " and ".join(f"[{s}]" for s in sids) + " retrieval helps.",
            "similarities": [], "differences": []}


def _gaps(user):
    sids = source_ids_in(user)
    return {"gaps": [{"title": "Multilingual evaluation", "description": "Not evaluated",
                      "category": "evidence_supported", "basis": "Limitation reported",
                      "source_ids": sids[:1]}],
            "future_directions": [{"direction": "Longer contexts", "rationale": "speculative", "source_ids": []}]}


def _report_text(user, first_id="S1"):
    body = (f"The reviewed work reports that retrieval accuracy drops for long documents [{first_id}, p. 1]. "
            "The sources agree that grounding reduces hallucination while differing in how they measure recall "
            "and evaluate context limits across benchmarks [S1, p. 1].")
    return "\n\n".join([
        "# Retrieval Augmented Generation Limits", "## Abstract", "This review summarises the evidence.",
        "## 1. Introduction", "Background text.", "## 2. Research Objectives", "Objectives text.",
        "## 3. Literature Review", body, "## 4. Comparative Analysis", body,
        "## 5. Research Gap Identification", body, "## 6. Future Research Directions",
        "Speculative: explore longer contexts.", "## 7. Conclusion", "Conclusion text."])


def _quality_ok(_user):
    return {"claims": [{"claim": "retrieval accuracy drops", "source_ids": ["S1"], "status": "supported", "reason": ""}]}


def _responses(report=_report_text, quality=_quality_ok):
    return {
        "planning": _planning, "analysis": _analysis, "summarisation": _summaries,
        "research_gaps": _gaps, "report": report, "quality_claims": quality,
    }


@pytest.fixture
def pdfs():
    return [
        {"kind": "upload", "name": "rag_paper.pdf", "data": make_pdf([RAG_PAPER], title="RAG Limits", author="Ada Lovelace")},
        {"kind": "upload", "name": "survey.pdf", "data": make_pdf([SURVEY, SURVEY], creation="D:20220101000000")},
    ]


def _supervisor(llm, rag, db, settings):
    return ResearchSupervisor(llm, rag, db, max_revisions=settings.max_revisions, max_upload_mb=5)


def test_full_pipeline_produces_cited_report_and_persists(rag, db, settings, pdfs):
    llm = FakeLLM(_responses())
    events = []
    session = _supervisor(llm, rag, db, settings).run(
        "What are the limitations of current RAG systems?", "IEEE", 12, pdfs,
        progress=lambda stage, status, msg: events.append((stage, status)),
    )

    assert session["status"] == "completed"
    assert [s["source_id"] for s in session["registry"]] == ["S1", "S2"]
    assert session["registry"][0]["title"] == "RAG Limits" and session["registry"][0]["authors"] == ["Ada Lovelace"]
    assert session["registry"][1]["year"] == "2022"  # from the PDF creation date

    report = session["report_markdown"]
    for heading in ["## Abstract", "## 3. Literature Review", "## 5. Research Gap Identification", "## References"]:
        assert heading in report
    assert "[1, p. 1]" in report                 # IEEE rendering with page number
    assert "[S1" not in report                    # no raw markers left behind
    assert "RAG Limits" in report.split("## References")[1]
    assert session["quality"]["verdict"] == "Passed"
    assert session["gaps"]["gaps"][0]["category"] == "evidence_supported"

    completed = {s for s, status in events if status == "completed"}
    assert {key for key, _, _ in STAGES} <= completed
    assert events[0] == ("ingest", "running")

    stored = db.get_session(session["session_id"])
    assert stored["report_markdown"] == report
    assert db.stats()["papers_processed"] == 2
    assert len(db.session_events(session["session_id"])) >= 2 * len(STAGES)


def test_quality_failure_triggers_bounded_revision(rag, db, settings, pdfs):
    failing = {"claims": [{"claim": "retrieval doubles accuracy", "source_ids": ["S1"],
                           "status": "unsupported", "reason": "not stated"}]}
    llm = FakeLLM(_responses(quality=[failing, _quality_ok]))
    session = _supervisor(llm, rag, db, settings).run("limits of RAG", "APA", 8, pdfs)
    assert session["status"] == "completed"
    assert llm.calls.count("report") == 2                 # one revision
    assert session["quality"]["verdict"] == "Passed"


def test_revisions_are_capped_and_unresolved_problems_are_flagged(rag, db, settings, pdfs):
    always_failing = {"claims": [{"claim": "made up number 42", "source_ids": ["S1"],
                                  "status": "unsupported", "reason": "x"}]}
    llm = FakeLLM(_responses(quality=always_failing))
    session = _supervisor(llm, rag, db, settings).run("limits of RAG", "IEEE", 8, pdfs)
    assert llm.calls.count("report") == 3                 # original + at most 2 revisions
    assert session["quality"]["needs_revision"] is True
    assert "Quality Review Notes" in session["report_markdown"]
    assert "still found problems" in session["report_markdown"]


def test_invalid_citation_triggers_revision_and_is_fixed(rag, db, settings, pdfs):
    def draft(user):
        # First draft cites a source that does not exist; the revision (which carries feedback) is correct.
        return _report_text(user, first_id="S9") if "This is a revision" not in user else _report_text(user)

    llm = FakeLLM(_responses(report=draft))
    session = _supervisor(llm, rag, db, settings).run("limits of RAG", "APA", 8, pdfs)
    assert session["status"] == "completed"
    assert llm.calls.count("report") == 2
    assert "S9" not in session["report_markdown"]
    assert session["quality"]["verdict"] == "Passed"


def test_llm_failure_is_recorded_with_stage(rag, db, settings, pdfs):
    llm = FakeLLM(_responses())
    llm.responses["planning"] = LLMError("The LLM provider rate limit or quota was reached.")
    with pytest.raises(ResearchError) as info:
        _supervisor(llm, rag, db, settings).run("limits of RAG", "IEEE", 8, pdfs)
    assert info.value.stage == "plan"
    failed = db.list_sessions()[0]
    assert failed["status"] == "failed" and "quota" in failed["error"]


def test_all_sources_invalid_stops_at_ingestion(rag, db, settings):
    bad = [{"kind": "upload", "name": "scan.pdf", "data": make_pdf([""])},
           {"kind": "upload", "name": "notes.txt", "data": b"hello"}]
    llm = FakeLLM(_responses())
    with pytest.raises(ResearchError) as info:
        _supervisor(llm, rag, db, settings).run("limits of RAG", "IEEE", 8, bad)
    assert info.value.stage == "ingest"
    assert "None of the sources could be used" in str(info.value)
    assert llm.calls == []


def test_one_bad_file_does_not_stop_the_session(rag, db, settings, pdfs):
    mixed = pdfs + [{"kind": "upload", "name": "broken.pdf", "data": b"%PDF-1.4 junk"}]
    session = _supervisor(FakeLLM(_responses()), rag, db, settings).run("limits of RAG", "IEEE", 8, mixed)
    assert session["status"] == "completed"
    assert any(i["name"] == "broken.pdf" and i["level"] == "error" for i in session["skipped"])


def test_duplicate_upload_is_ignored(rag, db, settings, pdfs):
    dup = pdfs + [{"kind": "upload", "name": "copy.pdf", "data": pdfs[0]["data"]}]
    session = _supervisor(FakeLLM(_responses()), rag, db, settings).run("limits of RAG", "IEEE", 8, dup)
    assert len(session["registry"]) == 2
    assert any("Duplicate" in i["message"] for i in session["skipped"])
