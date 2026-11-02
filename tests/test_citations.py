from services.citations import (
    cited_source_ids,
    format_reference_list,
    parse_citation_groups,
    render_citations,
    strip_invalid_citations,
    validate_citations,
)

REGISTRY = [
    {"source_id": "S1", "document_id": "d1", "document_name": "one.pdf", "title": "Dense Retrieval Survey",
     "authors": ["Ada Lovelace", "Alan Turing"], "year": "2021", "source_type": "arxiv",
     "url": "https://arxiv.org/abs/1234.5678", "page_count": 4},
    {"source_id": "S2", "document_id": "d2", "document_name": "notes.pdf", "title": "Uploaded Notes",
     "authors": [], "year": "n.d.", "source_type": "upload", "url": "", "page_count": 2},
]


def test_parse_groups_and_pages():
    groups = parse_citation_groups("See [S1, p. 3] and [S2] then [S1, p. 4; S2].")
    assert [[(r.source_id, r.page) for r in refs] for _, _, refs in groups] == [
        [("S1", 3)], [("S2", None)], [("S1", 4), ("S2", None)]
    ]


def test_validation_flags_unknown_ids_and_bad_pages():
    issues = validate_citations("Claim [S9]. Other [S1, p. 9]. Good [S2, p. 2].", REGISTRY)
    assert any("S9" in i for i in issues)
    assert any("Page 9" in i for i in issues)
    assert len(issues) == 2


def test_ieee_numbers_by_first_appearance_and_keeps_pages():
    text = "First [S2, p. 1] then [S1, p. 3] and again [S2]."
    rendered, order = render_citations(text, REGISTRY, "IEEE")
    assert order == ["S2", "S1"]
    assert rendered == "First [1, p. 1] then [2, p. 3] and again [1]."


def test_apa_uses_author_year_and_handles_missing_authors():
    text = "Retrieval improves grounding [S1, p. 2]; notes agree [S2]."
    rendered, _ = render_citations(text, REGISTRY, "APA")
    assert rendered == "Retrieval improves grounding (Lovelace & Turing, 2021, p. 2); notes agree (Uploaded Notes, n.d.)."


def test_bare_source_mentions_follow_the_same_numbering():
    text = "S2 and S1 differ [S2, p. 1]. Later S1 agrees."
    rendered, order = render_citations(text, REGISTRY, "IEEE", order=["S2", "S1"])
    assert rendered == "[1] and [2] differ [1, p. 1]. Later [2] agrees."
    assert order == ["S2", "S1"]
    apa, _ = render_citations("S1 and S2 differ.", REGISTRY, "APA")
    assert apa == "Lovelace & Turing and Uploaded Notes differ."


def test_reference_lists_are_built_from_registry_only():
    ieee = format_reference_list(REGISTRY, ["S1", "S2"], "IEEE")
    assert ieee[0].startswith('[1] A. Lovelace, A. Turing, "Dense Retrieval Survey,"')
    assert "https://arxiv.org/abs/1234.5678" in ieee[0]
    assert "uploaded document (notes.pdf)" in ieee[1]
    assert "n.d.." not in ieee[1]  # no doubled full stop after n.d.
    apa = format_reference_list(REGISTRY, ["S1"], "APA")
    assert apa == ["Lovelace, A., & Turing, A. (2021). Dense Retrieval Survey. arXiv. https://arxiv.org/abs/1234.5678"]


def test_strip_invalid_and_cited_order():
    cleaned, removed = strip_invalid_citations("Keep [S1]. Drop [S7, p. 1].", {"S1", "S2"})
    assert removed == 1 and "S7" not in cleaned and "[S1]" in cleaned
    assert cited_source_ids("[S2] then [S1] then [S2]") == ["S2", "S1"]
