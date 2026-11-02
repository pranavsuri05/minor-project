"""Central configuration. Values come from environment variables (a .env file is loaded if present)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

# Gemini exposes an OpenAI-compatible endpoint, so one client class serves both providers.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

PROVIDER_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "gemini_native": "gemini-3.5-flash-lite",  # Google's google-genai SDK; uses GEMINI_API_KEY
}


def _secret(name: str) -> str:
    """Read a top-level Streamlit secret (Streamlit Community Cloud, or .streamlit/secrets.toml locally).

    Returns "" when there is no secrets file, the key is missing, or the code runs outside Streamlit.
    """
    try:
        import streamlit as st

        value = st.secrets.get(name)
    except Exception:  # noqa: BLE001 - no secrets configured is a normal situation
        return ""
    if value is None or hasattr(value, "keys"):  # ignore missing keys and nested tables
        return ""
    return str(value).strip()


def _env(name: str, default: str = "") -> str:
    """Setting lookup order: environment variable (including .env), then Streamlit secrets, then the default."""
    return os.getenv(name, "").strip() or _secret(name) or default


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning("Ignoring non-integer %s=%r; using %d", name, raw, default)
        return default


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    openai_api_key: str = field(repr=False)  # repr=False keeps keys out of logs and tracebacks
    gemini_api_key: str = field(repr=False)
    llm_model: str
    llm_timeout_s: int
    llm_max_retries: int
    llm_max_tokens: int
    embedding_model: str
    chunk_size_words: int
    chunk_overlap_words: int
    max_upload_mb: int
    arxiv_max_results: int
    max_revisions: int
    data_dir: Path
    outputs_dir: Path
    chroma_dir: Path
    db_path: Path
    log_dir: Path

    @property
    def active_api_key(self) -> str:
        return self.openai_api_key if self.llm_provider == "openai" else self.gemini_api_key

    @property
    def llm_base_url(self) -> str | None:
        return GEMINI_OPENAI_BASE_URL if self.llm_provider == "gemini" else None

    @property
    def llm_configured(self) -> bool:
        return self.llm_provider in PROVIDER_DEFAULT_MODELS and bool(self.active_api_key)

    @property
    def llm_setup_message(self) -> str:
        key_name = "OPENAI_API_KEY" if self.llm_provider == "openai" else "GEMINI_API_KEY"
        if self.llm_provider not in PROVIDER_DEFAULT_MODELS:
            return (
                f"Unknown LLM_PROVIDER '{self.llm_provider}'. Use 'gemini_native', 'gemini' or 'openai'."
            )
        return (
            f"No LLM API key is configured for provider '{self.llm_provider}'. Set {key_name} in a local .env "
            "file (copy .env.example) or in the Streamlit Community Cloud secrets. The app can still ingest PDFs, "
            "but research reports require an LLM."
        )


def build_settings(root: Path = ROOT_DIR) -> Settings:
    provider = _env("LLM_PROVIDER", "gemini_native").lower()
    data_dir = root / "data"
    return Settings(
        llm_provider=provider,
        openai_api_key=_env("OPENAI_API_KEY"),
        gemini_api_key=_env("GEMINI_API_KEY"),
        llm_model=_env("LLM_MODEL") or PROVIDER_DEFAULT_MODELS.get(provider, ""),
        llm_timeout_s=_env_int("LLM_TIMEOUT_S", 90),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 2),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 3000),
        embedding_model=_env("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        # all-MiniLM-L6-v2 works on ~256 word-pieces; ~600 whitespace words ≈ 800 LLM tokens.
        chunk_size_words=_env_int("CHUNK_SIZE_WORDS", 600),
        chunk_overlap_words=_env_int("CHUNK_OVERLAP_WORDS", 90),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 50),
        arxiv_max_results=_env_int("ARXIV_MAX_RESULTS", 8),
        max_revisions=min(_env_int("MAX_REVISIONS", 2), 2),
        data_dir=data_dir,
        outputs_dir=root / "outputs",
        chroma_dir=root / "chroma_db",
        db_path=data_dir / "research.db",
        log_dir=root / "logs",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = build_settings()
    for folder in (settings.data_dir, settings.outputs_dir, settings.chroma_dir, settings.log_dir):
        folder.mkdir(parents=True, exist_ok=True)
    return settings


_logging_configured = False


def configure_logging(log_dir: Path | None = None) -> None:
    """Send logs to stderr and logs/app.log once per process."""
    global _logging_configured
    if _logging_configured:
        return
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "app.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
    )
    _logging_configured = True
