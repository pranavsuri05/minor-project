import os

import pytest

from config import Settings
from database.history import HistoryDB
from rag.vector_store import RAGEngine, VectorStore
from tests.fakes import HashEmbedder

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")


@pytest.fixture
def settings(tmp_path):
    return Settings(
        llm_provider="openai", openai_api_key="test", gemini_api_key="", llm_model="test-model",
        llm_timeout_s=5, llm_max_retries=0, llm_max_tokens=500, embedding_model="hash-test",
        chunk_size_words=40, chunk_overlap_words=8, max_upload_mb=5, arxiv_max_results=3, max_revisions=2,
        data_dir=tmp_path / "data", outputs_dir=tmp_path / "out", chroma_dir=tmp_path / "chroma",
        db_path=tmp_path / "data" / "test.db", log_dir=tmp_path / "logs",
    )


@pytest.fixture
def rag(tmp_path, settings):
    store = VectorStore(tmp_path / "chroma")
    return RAGEngine(store, HashEmbedder(), settings.chunk_size_words, settings.chunk_overlap_words)


@pytest.fixture
def db(settings):
    return HistoryDB(settings.db_path)
