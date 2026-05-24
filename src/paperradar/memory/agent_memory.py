from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import settings
from ..retrieval.embedder import Embedder
from ..schemas.memory import AgentSession
from ..store.chroma import ChromaManager
from ..store.sqlite_memory import EpisodicStore

_COMPRESS_SYSTEM = """\
You are a research assistant. Compress the given agent session into a concise JSON summary.

Return ONLY valid JSON with these exact keys:
{
  "input_summary": "...",    // <=100 words describing what was analyzed/queried
  "output_summary": "...",   // <=150 words describing the most important findings
  "key_findings": ["...", ...],  // 3-5 bullet-point findings
  "tags": ["...", ...]           // 5-8 domain/keyword tags (lowercase, short)
}

Be specific and factual. Include paper titles, domains, and key technical terms when relevant."""


class AgentMemoryManager:
    """Saves and retrieves structured cross-agent memory sessions."""

    def __init__(self) -> None:
        self._store = EpisodicStore()
        self._chroma = ChromaManager()
        self._embedder = Embedder()
        self._llm = ChatOpenAI(
            model=settings.llm.model,
            temperature=0,
            api_key=settings.llm.openai_api_key,
            **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
        )

    # ------------------------------------------------------------------
    # compress_session: full report → structured summary via LLM
    # ------------------------------------------------------------------
    def compress_session(
        self,
        agent_type: str,
        input_text: str,
        full_report_dict: dict,
    ) -> tuple[str, str, list[str], list[str]]:
        report_str = json.dumps(full_report_dict, ensure_ascii=False, default=str)
        # Trim to avoid token overflow
        report_excerpt = report_str[:6000]
        user_msg = (
            f"Agent type: {agent_type}\n\n"
            f"Input (first 400 chars): {input_text[:400]}\n\n"
            f"Output report (excerpt):\n{report_excerpt}"
        )
        try:
            response = self._llm.invoke(
                [SystemMessage(content=_COMPRESS_SYSTEM), HumanMessage(content=user_msg)]
            )
            raw = response.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            return (
                str(data.get("input_summary", "")),
                str(data.get("output_summary", "")),
                list(data.get("key_findings", [])),
                list(data.get("tags", [])),
            )
        except Exception:
            # Graceful fallback: minimal summaries
            domain = full_report_dict.get("detected_domain", "") or full_report_dict.get("title", "")
            return (
                f"{agent_type} session on: {input_text[:80]}",
                f"Agent completed {agent_type} analysis. Domain: {domain}",
                [],
                [],
            )

    # ------------------------------------------------------------------
    # save_session: persist to JSON + SQLite + ChromaDB
    # ------------------------------------------------------------------
    def save_session(
        self,
        agent_type: str,
        input_text: str,
        full_report_obj: Any,
        session_id: str | None = None,
    ) -> str:
        if session_id is None:
            session_id = uuid4().hex[:16]

        full_report_dict = (
            full_report_obj.model_dump() if hasattr(full_report_obj, "model_dump")
            else dict(full_report_obj)
        )

        input_summary, output_summary, key_findings, tags = self.compress_session(
            agent_type, input_text, full_report_dict
        )

        # 1. Archive full report as JSON
        os.makedirs(settings.memories_dir, exist_ok=True)
        full_report_path = os.path.join(settings.memories_dir, f"{session_id}.json")
        with open(full_report_path, "w", encoding="utf-8") as f:
            json.dump(full_report_dict, f, ensure_ascii=False, default=str, indent=2)

        # 2. Build embeddable document
        document = " ".join(filter(None, [
            input_summary,
            output_summary,
            *key_findings,
            *tags,
        ]))
        embedding = self._embedder.embed_query(document)

        # 3. Store in ChromaDB
        timestamp_str = datetime.now(timezone.utc).isoformat()
        metadata: dict[str, Any] = {
            "agent_type": agent_type,
            "timestamp": timestamp_str,
            "tags": ", ".join(tags),
            "input_summary": input_summary[:200],
        }
        self._chroma.upsert_agent_memory(session_id, embedding, document, metadata)

        # 4. Store in SQLite
        self._store.save_agent_session(
            session_id=session_id,
            agent_type=agent_type,
            timestamp=timestamp_str,
            input_summary=input_summary,
            output_summary=output_summary,
            key_findings=key_findings,
            tags=tags,
            full_report_path=full_report_path,
        )

        return session_id

    # ------------------------------------------------------------------
    # retrieve_relevant: vector search + recency re-scoring
    # ------------------------------------------------------------------
    def retrieve_relevant(
        self,
        query_text: str,
        agent_type_filter: str | None = None,
        n_results: int = 5,
        recency_weight: float = 0.3,
    ) -> list[AgentSession]:
        embedding = self._embedder.embed_query(query_text)
        where = {"agent_type": agent_type_filter} if agent_type_filter else None

        try:
            raw = self._chroma.query_agent_memory(
                embedding, n_results=min(n_results * 2, 20), where=where
            )
        except Exception:
            return []

        ids = raw.get("ids", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        if not ids:
            return []

        now = datetime.now(timezone.utc)
        scored: list[tuple[float, str]] = []
        for sid, dist, meta in zip(ids, distances, metadatas):
            semantic_score = max(0.0, 1.0 - float(dist))
            try:
                ts_str = meta.get("timestamp", "")
                ts = datetime.fromisoformat(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                days_elapsed = (now - ts).total_seconds() / 86400.0
                recency_score = math.exp(-days_elapsed / 30.0)
            except Exception:
                recency_score = 0.5
            final = semantic_score * (1 - recency_weight) + recency_score * recency_weight
            scored.append((final, sid))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_ids = [sid for _, sid in scored[:n_results]]

        sessions: list[AgentSession] = []
        for sid in top_ids:
            row = self._store.get_session_by_id(sid)
            if row is None:
                continue
            try:
                sessions.append(AgentSession(
                    session_id=row["session_id"],
                    agent_type=row["agent_type"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    input_summary=row["input_summary"],
                    output_summary=row["output_summary"],
                    key_findings=row["key_findings"],
                    tags=row["tags"],
                    full_report_path=row.get("full_report_path", ""),
                ))
            except Exception:
                continue
        return sessions

    # ------------------------------------------------------------------
    # build_memory_context: format sessions as LLM-readable text
    # ------------------------------------------------------------------
    def build_memory_context(
        self,
        sessions: list[AgentSession],
        current_agent_type: str,
    ) -> str:
        if not sessions:
            return ""
        lines = ["=== Your Research History ==="]
        for i, s in enumerate(sessions, 1):
            ts = s.timestamp.strftime("%Y-%m-%d")
            tags_str = ", ".join(s.tags[:6]) if s.tags else "N/A"
            lines.append(f"[{i}] [{s.agent_type}] {ts} | Tags: {tags_str}")
            lines.append(f"    Input: {s.input_summary}")
            if s.key_findings:
                lines.append("    Key findings:")
                for kf in s.key_findings:
                    lines.append(f"      - {kf}")
            else:
                lines.append(f"    Summary: {s.output_summary[:200]}")
            lines.append("")
        lines.append("===")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # get_all_sessions: for UI display
    # ------------------------------------------------------------------
    def get_all_sessions(self, limit: int = 50) -> list[dict]:
        return self._store.get_recent_sessions(limit=limit)
