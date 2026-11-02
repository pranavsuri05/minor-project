# Multi-Agent Autonomous Scientific Research Platform

> **Academic minor project** · B.Tech, Computer Science and Technology · 7th semester
> Author: Pranav Suri

A Streamlit application in which a LangGraph supervisor coordinates eight specialised agents. The
agents read uploaded research PDFs (and optionally arXiv papers), retrieve evidence with RAG, and produce a
cited literature review that is checked before it is delivered.

## Quick start

Requires Python 3.11 or newer.

```bash
cd research-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then put your API key in .env
streamlit run app.py
```

The first research run downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB) from Hugging Face.

### Configure the LLM

Set `LLM_PROVIDER` to `gemini_native` (Google's `google-genai` SDK), `gemini` (OpenAI-compatible endpoint) or `openai`,
and fill in the matching key. `gemini_native` and `gemini` both use `GEMINI_API_KEY`:

```
LLM_PROVIDER=gemini_native
LLM_MODEL=gemini-3.5-flash-lite
GEMINI_API_KEY=your-key
```

Model availability changes over time. A retired model returns HTTP 404, and an overloaded one returns 503
(the native provider retries 429/5xx twice). Pick a model your key can actually call.

With no key the app still starts. Uploads and retrieval work, and the New Research page shows a setup
message instead of generating anything. The app never returns fabricated results.

## Pipeline

```
Document ingestion ─► Planning ─► Literature retrieval (RAG) ─► Semantic analysis ─► Summarisation
                                                                                       │
Finalisation ◄── Quality review ◄── Report generation ◄── Research gap identification ◄─┘
                     │   (at most 2 revision loops: extra retrieval, rewrite)
                     └──────────────────────────────────────────────────────────────────┘
```

| Agent | File | Responsibility |
|---|---|---|
| Supervisor | `agents/supervisor.py` | LangGraph StateGraph, shared state, progress events, persistence |
| Planning | `agents/planning.py` | 3–5 subtasks with retrieval queries (structured JSON) |
| Literature Retrieval | `agents/retrieval.py` | PDF/arXiv ingestion, duplicate handling, per-subtask RAG |
| Semantic Analysis | `agents/analysis.py` | Objectives, methods, findings, limitations, cross-paper comparisons |
| Summarisation | `agents/summarisation.py` | Per-paper summaries, synthesis, similarities and differences |
| Research Gap | `agents/research_gap.py` | Evidence-supported gaps, separately labelled speculative directions |
| Report Generation | `agents/report.py` | Structured Markdown report; references are built from metadata |
| Quality Review | `agents/quality.py` | Citation/page validation, section checks, LLM claim verification |

Every finding keeps `source_ids`. Any item that cites an unknown source is dropped and counted, so a
claim cannot reach the report without a real source behind it.

## Evidence and citations

- Each chunk keeps `document_id`, `document_name`, `page_number` (and `page_end`), `chunk_id` and text.
- Sources are registered as `S1`, `S2`, … The LLM writes markers such as `[S2, p. 4]`. The quality agent
  checks that the source exists and the page is in range.
- IEEE (`[1, p. 4]`) and APA (`(Lovelace & Turing, 2021, p. 4)`) are rendered from the registry.
- References are generated from registry metadata. Uploaded PDFs without bibliographic metadata are
  labelled as uploaded documents.

## Project layout

```
app.py                  Streamlit entry point
config.py               Settings from environment variables; logging
agents/                 Supervisor and the eight agents, plus shared helpers
rag/                    PDF ingestion (PyMuPDF), chunking, embeddings, ChromaDB store and RAG facade
services/               Shared LLM client, arXiv search, citations, PDF and DOCX export
database/history.py     SQLite: sessions, agent event log, document library
ui/                     Theme CSS (Light/Dark/Auto), components, pages
tests/                  pytest suite (deterministic fakes, no network required)
data/                   SQLite database (created at runtime)
chroma_db/              Vector store (created at runtime)
outputs/                Exported reports
logs/app.log            Application log
```

## Tests

```bash
pytest
```

The suite uses a scripted LLM and a deterministic hashing embedder, so it needs neither an API key nor
model downloads. The end-to-end tests still parse real PDFs and write to a real ChromaDB store.

## Limitations

- Retrieval quality depends on the embedding model. `all-MiniLM-L6-v2` is small and general-purpose.
- Output quality depends on the LLM. The quality agent catches unsupported claims and bad citations, but it
  is itself an LLM check. Its verdicts are shown in full, not hidden.
- Scanned PDFs without a text layer are rejected. OCR is not included.
- Semantic Scholar is not integrated. arXiv is the only academic search source.
- Single-user, local deployment. No authentication, and no multi-user sessions.
- Token counts are approximated with word counts (about 0.75 words per token).
- **Page references are approximate.** If the model omits a page, the app attaches the page of the best-matching
  retrieved passage (embedding similarity, minimum 0.35). A passage can span a page break, and the match is a
  heuristic. In one live check, about 75–80% of cited sentences matched the text of their cited page closely.
- **Reference lists are removed from the evidence** (everything after a "References" heading, including appendices
  that follow it). PDFs without a recognisable heading are indexed in full.
- **Titles of uploaded PDFs without metadata** are guessed from the first line of page 1 and can be truncated or
  wrong. Authors are not guessed. Check the reference list before submitting a report.
- **The quality review is LLM-assisted and not exhaustive.** In live testing it once passed a report containing a
  wrongly expanded abbreviation. Treat the quality score as a signal, not a guarantee.

## Acknowledgements

This project is built on open-source software and public services:
[Streamlit](https://streamlit.io), [LangGraph](https://github.com/langchain-ai/langgraph) and
[LangChain](https://www.langchain.com), [ChromaDB](https://www.trychroma.com),
[Sentence-Transformers](https://www.sbert.net) (model `all-MiniLM-L6-v2`), [PyMuPDF](https://pymupdf.readthedocs.io),
[ReportLab](https://www.reportlab.com), [python-docx](https://python-docx.readthedocs.io), and the Google Gen AI SDK
(`google-genai`). Paper search uses the [arXiv API](https://info.arxiv.org/help/api/index.html); thank you to arXiv
for use of its open access interoperability. Each library is used under its own licence.
