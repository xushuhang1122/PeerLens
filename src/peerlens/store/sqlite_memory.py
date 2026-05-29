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

CREATE TABLE IF NOT EXISTS agent_sessions (
    session_id       TEXT PRIMARY KEY,
    agent_type       TEXT NOT NULL,
    timestamp        TEXT NOT NULL,
    input_summary    TEXT NOT NULL,
    output_summary   TEXT NOT NULL,
    key_findings     TEXT NOT NULL DEFAULT '[]',
    tags             TEXT NOT NULL DEFAULT '[]',
    full_report_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_agent_type ON agent_sessions(agent_type);
CREATE INDEX IF NOT EXISTS idx_sessions_timestamp  ON agent_sessions(timestamp);
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

    def save_agent_session(
        self,
        session_id: str,
        agent_type: str,
        timestamp: str,
        input_summary: str,
        output_summary: str,
        key_findings: list[str],
        tags: list[str],
        full_report_path: str = "",
    ) -> None:
        import json
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO agent_sessions
                   (session_id, agent_type, timestamp, input_summary, output_summary,
                    key_findings, tags, full_report_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    agent_type,
                    timestamp,
                    input_summary,
                    output_summary,
                    json.dumps(key_findings, ensure_ascii=False),
                    json.dumps(tags, ensure_ascii=False),
                    full_report_path,
                ),
            )

    def get_recent_sessions(
        self, agent_type: str | None = None, limit: int = 20
    ) -> list[dict]:
        import json
        with self._conn() as conn:
            if agent_type:
                rows = conn.execute(
                    "SELECT * FROM agent_sessions WHERE agent_type=? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (agent_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM agent_sessions ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["key_findings"] = json.loads(d.get("key_findings", "[]"))
            d["tags"] = json.loads(d.get("tags", "[]"))
            result.append(d)
        return result

    def get_session_by_id(self, session_id: str) -> dict | None:
        import json
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM agent_sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["key_findings"] = json.loads(d.get("key_findings", "[]"))
        d["tags"] = json.loads(d.get("tags", "[]"))
        return d
