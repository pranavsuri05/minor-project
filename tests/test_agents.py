import pytest

from agents.analysis import SemanticAnalysisAgent
from agents.planning import PlanningAgent
from agents.quality import QualityAgent
from agents.report import REQUIRED_SECTIONS, ReportAgent
from agents.research_gap import ResearchGapAgent
from agents.summarisation import SummarisationAgent
from services.llm import LLMClient, LLMOutputError, parse_json_text
from tests.fakes import FakeLLM

REGISTRY = [
    {"source_id": "S1", "document_id": "d1", "document_name": "a.pdf", "title": "Paper A", "authors": ["X Y"],
     "year": "2022", "source_type": "upload", "url": "", "page_count": 5},
    {"source_id": "S2", "document_id": "d2", "document_name": "b.pdf", "title": "Paper B", "authors": ["Z W"],
     "year": "2023", "source_type": "arxiv", "url": "https://arxiv.org/abs/9", "page_count": 3},
]
EVIDENCE = [
    {"chunk_id": "c1", "source_id": "S1", "page_number": 2, "page_end": 2, "chunk_index": 0,
     "text": "Method A uses dense retrieval.", "score": 0.6, "document_id": "d1", "document_name": "a.pdf",
     "matched_subtasks": []},
]


def test_json_parser_handles_fences_and_prose():
    assert parse_json_text('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_text('Here you go: {"subtasks": []} thanks') == {"subtasks": []}
    with pytest.raises(LLMOutputError):
        parse_json_text("no json here")


def test_complete_json_repairs_once_then_fails(monkeypatch, settings):
    client = LLMClient(settings)
    replies = iter(["not json", '{"ok": true}'])
    monkeypatch.setattr(client, "complete", lambda task, system, user: next(replies))
    assert client.complete_json("t", "sys", "user") == {"ok": True}

    always_bad = LLMClient(settings)
    monkeypatch.setattr(always_bad, "complete", lambda task, system, user: "still bad")
    with pytest.raises(LLMOutputError):
        always_bad.complete_json("t", "sys", "user")


def test_missing_api_key_gives_setup_message(settings):
    from dataclasses import replace

    unconfigured = LLMClient(replace(settings, openai_api_key=""))
    assert not unconfigured.is_configured
    from services.llm import LLMNotConfiguredError

    with pytest.raises(LLMNotConfiguredError, match="OPENAI_API_KEY"):
        unconfigured.complete("t", "s", "u")


def test_planning_returns_structured_subtasks():
    llm = FakeLLM({"planning": {"research_dimensions": ["limits"], "subtasks": [
        {"title": "Retrieval accuracy", "description": "d", "retrieval_query": "retrieval accuracy limits"},
        {"title": "", "description": "", "retrieval_query": "hallucination grounding"},
        {"title": "empty", "description": "", "retrieval_query": ""},
    ]}})
    plan = PlanningAgent(llm).plan("What are the limits of RAG?")
    assert plan["research_dimensions"] == ["limits"]
    assert [s["retrieval_query"] for s in plan["subtasks"]] == ["retrieval accuracy limits", "hallucination grounding"]
    assert plan["subtasks"][1]["title"] == "hallucination grounding"


def test_planning_rejects_unusable_output():
    with pytest.raises(LLMOutputError):
        PlanningAgent(FakeLLM({"planning": {"subtasks": []}})).plan("question")


def test_analysis_drops_untraceable_items_and_requires_two_sources_for_comparisons():
    llm = FakeLLM({"analysis": {
        "papers": [
            {"source_id": "S1", "objective": "Study", "methodology": "Dense retrieval",
             "contributions": [], "limitations": [],
             "findings": [{"statement": "Improves recall", "source_ids": ["S1"]},
                          {"statement": "Invented", "source_ids": ["S99"]},
                          "plain string with no source"]},
            {"source_id": "S42", "findings": []},
        ],
        "comparisons": [
            {"type": "agreement", "topic": "recall", "statement": "Both agree", "source_ids": ["S1", "S2"]},
            {"type": "contradiction", "topic": "one-sided", "statement": "x", "source_ids": ["S1"]},
        ],
    }})
    result = SemanticAnalysisAgent(llm).analyse("q", [{"title": "t", "description": ""}], REGISTRY, EVIDENCE)
    assert [p["source_id"] for p in result["papers"]] == ["S1"]
    assert [f["statement"] for f in result["papers"][0]["findings"]] == ["Improves recall"]
    assert len(result["comparisons"]) == 1 and result["comparisons"][0]["source_ids"] == ["S1", "S2"]
    assert result["dropped_items"] >= 3


def test_summarisation_strips_unknown_citation_markers():
    llm = FakeLLM({"summarisation": {
        "paper_summaries": [{"source_id": "S1", "summary": "Paper A works well [S1, p. 2] [S8]."}],
        "synthesis": "Both papers differ [S1] and [S2, p. 1].",
        "similarities": [{"statement": "Shared idea", "source_ids": ["S1", "S2"]}],
        "differences": [{"statement": "Ghost", "source_ids": ["S8"]}],
    }})
    result = SummarisationAgent(llm).summarise("q", REGISTRY, {"papers": []})
    assert "S8" not in result["paper_summaries"][0]["summary"]
    assert result["paper_summaries"][0]["summary"].endswith("[S1, p. 2].")
    assert "[S2, p. 1]" in result["synthesis"]
    assert result["differences"] == []
    assert result["dropped_items"] >= 2


def test_gaps_are_evidence_supported_only_with_sources_and_basis():
    llm = FakeLLM({"research_gaps": {
        "gaps": [
            {"title": "No multilingual test", "description": "d", "basis": "S1 limitation: English only",
             "source_ids": ["S1"]},
            {"title": "Unsourced gap", "description": "d", "basis": "", "source_ids": []},
        ],
        "future_directions": [{"direction": "Try retrieval at scale", "rationale": "r", "source_ids": []}],
    }})
    gaps = ResearchGapAgent(llm).identify("q", REGISTRY, {}, {})
    categories = {g["title"]: g["category"] for g in gaps["gaps"]}
    assert categories == {"No multilingual test": "evidence_supported", "Unsourced gap": "speculative"}
    assert gaps["future_directions"][0]["category"] == "speculative"


def _good_report() -> str:
    paragraph = ("The reviewed work studies retrieval quality in depth and reports that dense retrieval improves "
                 "recall on the evaluated benchmark [S1, p. 2].")
    sections = ["# Limits of RAG", "## Abstract", "A short abstract.",
                "## 1. Introduction", "Intro text.", "## 2. Research Objectives", "Objectives.",
                "## 3. Literature Review", paragraph,
                "## 4. Comparative Analysis", paragraph,
                "## 5. Research Gap Identification", paragraph,
                "## 6. Future Research Directions", "Speculative: test longer contexts.",
                "## 7. Conclusion", "Done."]
    return "\n\n".join(sections)


def test_quality_passes_clean_report_and_flags_problems():
    llm = FakeLLM({"quality_claims": {"claims": [
        {"claim": "dense retrieval improves recall", "source_ids": ["S1"], "status": "supported", "reason": ""}]}})
    qa = QualityAgent(llm)
    result = qa.review(_good_report(), REGISTRY, EVIDENCE)
    assert result["verdict"] == "Passed" and result["score"] == 100 and not result["needs_revision"]

    broken = _good_report().replace("[S1, p. 2]", "[S9, p. 2]").replace("## 6. Future Research Directions", "## 6. Other")
    llm2 = FakeLLM({"quality_claims": {"claims": [
        {"claim": "dense retrieval doubles recall", "source_ids": ["S1"], "status": "unsupported",
         "reason": "passage says improves, not doubles"}]}})
    result = QualityAgent(llm2).review(broken, REGISTRY, EVIDENCE)
    assert result["needs_revision"] is True
    assert any("S9" in issue for issue in result["invalid_citations"])
    assert "6. Future Research Directions" in result["missing_sections"]
    assert result["unsupported_claims"] == ["dense retrieval doubles recall"]
    assert result["follow_up_queries"] and "Rewrite" in result["revision_feedback"]


def test_quality_reports_when_claim_check_cannot_run():
    llm = FakeLLM({"quality_claims": LLMOutputError("bad json")})
    result = QualityAgent(llm).review(_good_report(), REGISTRY, EVIDENCE)
    assert result["claim_check_status"] == "not_run"
    assert result["verdict"].startswith("Partially verified")
    assert result["score"] == 90


def test_quality_accepts_alternative_claim_shapes_and_asks_twice_on_bad_shape():
    claim = {"claim": "dense retrieval improves recall", "status": "supported", "reason": ""}
    # FakeLLM treats a list as a sequence of responses, so [[claim]] scripts one bare-list response.
    for scripted in ([[claim]], {"results": [claim]}):
        result = QualityAgent(FakeLLM({"quality_claims": scripted})).review(_good_report(), REGISTRY, EVIDENCE)
        assert result["claim_check_status"] == "completed" and result["claims_checked"] == 1

    llm = FakeLLM({"quality_claims": [{"verdict": "looks fine"}, {"claims": [claim]}]})
    result = QualityAgent(llm).review(_good_report(), REGISTRY, EVIDENCE)
    assert result["claim_check_status"] == "completed" and llm.calls.count("quality_claims") == 2

    never_valid = FakeLLM({"quality_claims": {"verdict": "looks fine"}})
    result = QualityAgent(never_valid).review(_good_report(), REGISTRY, EVIDENCE)
    assert result["claim_check_status"] == "not_run" and never_valid.calls.count("quality_claims") == 2


def test_quality_evidence_keeps_every_source_even_when_passages_are_long():
    seen = {}

    def capture(user):
        seen["prompt"] = user
        return {"claims": []}

    long_text = "retrieval " * 5000  # ~50k chars each, far over the evidence budget
    evidence = [
        {"chunk_id": f"c{i}", "source_id": f"S{i}", "page_number": 1, "page_end": 1, "chunk_index": 0,
         "text": long_text, "score": 0.5, "document_id": f"d{i}", "document_name": f"{i}.pdf", "matched_subtasks": []}
        for i in (1, 2, 3)
    ]
    registry = REGISTRY + [{"source_id": "S3", "document_id": "d3", "document_name": "c.pdf", "title": "Paper C",
                            "authors": [], "year": "2024", "source_type": "upload", "url": "", "page_count": 2}]
    QualityAgent(FakeLLM({"quality_claims": capture})).review(_good_report(), registry, evidence)
    for source_id in ("S1", "S2", "S3"):
        assert f"### {source_id}" in seen["prompt"], f"{source_id} was cut from the reviewer's evidence"


def test_attach_pages_uses_best_matching_passage_and_skips_weak_matches():
    from agents.report import attach_pages
    from tests.fakes import HashEmbedder

    evidence = [
        {"source_id": "S1", "page_number": 2, "text": "dense retrieval improves recall on benchmark questions"},
        {"source_id": "S1", "page_number": 7, "text": "catalytic hydrogenation of alkenes uses palladium"},
    ]
    body = ("Dense retrieval improves recall on benchmark questions [S1]. "
            "Already paged claim [S1, p. 4]. Something unrelated entirely [S1].")
    result, attached = attach_pages(body, evidence, HashEmbedder(), min_similarity=0.5)
    assert "improves recall on benchmark questions [S1, p. 2]" in result
    assert "[S1, p. 4]" in result                      # existing pages are never overwritten
    assert result.endswith("entirely [S1].")           # weak match: no guessed page
    assert attached == 1


def test_report_finalise_renders_style_and_references():
    body = _good_report()
    finalised = ReportAgent.finalise(body, REGISTRY, "IEEE", quality_notes=["Note one"])
    assert "[1, p. 2]" in finalised["markdown"]
    assert "## References" in finalised["markdown"] and "## Quality Review Notes" in finalised["markdown"]
    assert finalised["cited_source_ids"] == ["S1"]
    apa = ReportAgent.finalise(body, REGISTRY, "APA")
    assert "(Y, 2022, p. 2)" in apa["markdown"]
    assert all(section.split(" ", 1)[-1] in _good_report() for section in REQUIRED_SECTIONS[1:2])
