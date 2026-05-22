from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from ..config import settings
from ..schemas.memory import EpisodicEvent

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodic_events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    content    TEXT NOT NULL,
    paper_id   TEXT,
    paper_title TEXT,
    feedback   TEXT,
    timestamp  TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_event_type ON episodic_events(event_type);
CREATE INDEX IF NOT EXISTS idx_timestamp  ON episodic_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_paper_id   ON episodic_events(paper_id);

CREATE TABLE IF NOT EXISTS crawl_log (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    conference   TEXT NOT NULL,
    year         INTEGER NOT NULL,
    decision     TEXT,
    paper_count  INTEGER,
    crawled_at   TEXT NOT NULL,
    status       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_crawl_conf_year ON crawl_log(conference, year);
"""


class EpisodicStore:
    def __init__(self, db_path: str | None = None) -> None:
        self._db = db_path or settings.sqlite.db_path
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(self._db)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_event(self, event: EpisodicEvent) -> None:
        ts = event.timestamp.isoformat()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO episodic_events
                   (event_type, content, paper_id, paper_title, feedback, timestamp, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_type,
                    event.content,
                    event.paper_id,
                    event.paper_title,
                    event.feedback,
                    ts,
                    event.session_id,
                ),
            )

    def get_recent_queries(self, n: int = 20) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT content FROM episodic_events WHERE event_type='query' "
                "ORDER BY timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [r["content"] for r in rows]

    def get_liked_paper_ids(self, n: int = 50) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT paper_id FROM episodic_events "
                "WHERE event_type='like' AND paper_id IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [r["paper_id"] for r in rows]

    def get_disliked_paper_ids(self, n: int = 50) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT paper_id FROM episodic_events "
                "WHERE event_type='dislike' AND paper_id IS NOT NULL "
                "ORDER BY timestamp DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [r["paper_id"] for r in rows]

    def get_last_push_timestamp(self) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT timestamp FROM episodic_events WHERE event_type='push_sent' "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        return row["timestamp"] if row else None

    def log_crawl(
        self,
        conference: str,
        year: int,
        decision: str | None,
        paper_count: int,
        status: str,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO crawl_log (conference, year, decision, paper_count, crawled_at, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conference, year, decision, paper_count, ts, status),
            )

    def has_crawl_record(self, conference: str, year: int) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM crawl_log WHERE conference=? AND year=? AND status='success' LIMIT 1",
                (conference, year),
            ).fetchone()
        return row is not None
