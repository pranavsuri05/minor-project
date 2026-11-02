"""SQLite persistence: research sessions, agent event log, and the document library."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    document_name TEXT NOT NULL,
    title TEXT,
    authors_json TEXT,
    year TEXT,
    source_type TEXT,
    source_url TEXT,
    page_count INTEGER,
    char_count INTEGER,
    file_hash TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    citation_style TEXT NOT NULL,
    top_k INTEGER,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    num_chunks INTEGER DEFAULT 0,
    registry_json TEXT,
    skipped_json TEXT,
    subtasks_json TEXT,
    analysis_json TEXT,
    summaries_json TEXT,
    gaps_json TEXT,
    quality_json TEXT,
    citation_order_json TEXT,
    report_markdown TEXT
);
CREATE TABLE IF NOT EXISTS session_documents (
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    document_id TEXT NOT NULL REFERENCES documents(document_id),
    PRIMARY KEY (session_id, document_id)
);
CREATE TABLE IF NOT EXISTS agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    agent TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    created_at TEXT NOT NULL
);
"""

JSON_COLUMNS = {
    "registry_json": "registry",
    "skipped_json": "skipped",
    "subtasks_json": "subtasks",
    "analysis_json": "analysis",
    "summaries_json": "summaries",
    "gaps_json": "gaps",
    "quality_json": "quality",
    "citation_order_json": "citation_order",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HistoryDB:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ------------------------------------------------------------------ documents

    def upsert_document(self, entry: dict[str, Any]) -> None:
        """Insert or refresh a library document. `entry` uses registry field names."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (document_id, document_name, title, authors_json, year, source_type,
                                       source_url, page_count, char_count, file_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title = excluded.title,
                    authors_json = excluded.authors_json,
                    year = excluded.year,
                    source_url = excluded.source_url
                """,
                (
                    entry["document_id"],
                    entry["document_name"],
                    entry.get("title"),
                    json.dumps(entry.get("authors", [])),
                    entry.get("year"),
                    entry.get("source_type"),
                    entry.get("url") or None,
                    entry.get("page_count"),
                    entry.get("char_count", 0),
                    entry.get("file_hash"),
                    _now(),
                ),
            )

    def list_documents(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT d.*, COUNT(sd.session_id) AS session_count
                FROM documents d
                LEFT JOIN session_documents sd ON sd.document_id = d.document_id
                GROUP BY d.document_id
                ORDER BY d.created_at DESC
                """
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["authors"] = json.loads(item.pop("authors_json") or "[]")
            result.append(item)
        return result

    # ------------------------------------------------------------------ sessions

    def create_session(self, session_id: str, question: str, citation_style: str, top_k: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, question, citation_style, top_k, status, created_at) "
                "VALUES (?, ?, ?, ?, 'running', ?)",
                (session_id, question, citation_style, top_k, _now()),
            )

    def link_session_document(self, session_id: str, document_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO session_documents (session_id, document_id) VALUES (?, ?)",
                (session_id, document_id),
            )

    def record_event(self, session_id: str, agent: str, status: str, message: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO agent_events (session_id, agent, status, message, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, agent, status, message, _now()),
            )

    def finish_session(self, session_id: str, status: str, error: str | None = None, **fields: Any) -> None:
        """Store the final state of a session. Keyword names map to *_json columns or plain columns."""
        assignments = ["status = ?", "error = ?", "completed_at = ?"]
        values: list[Any] = [status, error, _now()]
        for key, value in fields.items():
            column = next((c for c, name in JSON_COLUMNS.items() if name == key), key)
            if column in JSON_COLUMNS:
                value = json.dumps(value, ensure_ascii=False)
            assignments.append(f"{column} = ?")
            values.append(value)
        values.append(session_id)
        with self._connect() as conn:
            conn.execute(f"UPDATE sessions SET {', '.join(assignments)} WHERE session_id = ?", values)

    def mark_interrupted_sessions(self) -> int:
        """Sessions still 'running' after a restart were interrupted; mark them so history is honest."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE sessions SET status = 'interrupted', error = 'The app stopped before this research finished.' "
                "WHERE status = 'running'"
            )
            return cursor.rowcount

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.question, s.citation_style, s.status, s.error, s.created_at,
                       s.completed_at, s.num_chunks, s.quality_json,
                       (SELECT COUNT(*) FROM session_documents sd WHERE sd.session_id = s.session_id) AS document_count,
                       (s.report_markdown IS NOT NULL) AS has_report
                FROM sessions s
                ORDER BY s.created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            quality = json.loads(item.pop("quality_json") or "null")
            item["quality_score"] = quality.get("score") if isinstance(quality, dict) else None
            item["has_report"] = bool(item["has_report"])
            result.append(item)
        return result

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            documents = conn.execute(
                "SELECT d.* FROM documents d JOIN session_documents sd ON sd.document_id = d.document_id "
                "WHERE sd.session_id = ?",
                (session_id,),
            ).fetchall()
        item = dict(row)
        for column, name in JSON_COLUMNS.items():
            item[name] = json.loads(item.pop(column) or "null")
        item["documents"] = [dict(d) for d in documents]
        return item

    def session_events(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT agent, status, message, created_at FROM agent_events WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM sessions) AS total_sessions,
                    (SELECT COUNT(*) FROM sessions WHERE status = 'completed') AS completed_sessions,
                    (SELECT COUNT(*) FROM sessions WHERE report_markdown IS NOT NULL) AS reports_generated,
                    (SELECT COUNT(DISTINCT document_id) FROM session_documents) AS papers_processed
                """
            ).fetchone()
        return dict(row)
