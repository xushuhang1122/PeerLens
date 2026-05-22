from __future__ import annotations

from ..schemas.memory import EpisodicEvent
from ..schemas.tools import UserProfile
from ..store.sqlite_memory import EpisodicStore


class EpisodicMemory:
    def __init__(self) -> None:
        self._store = EpisodicStore()

    def record_query(self, query: str, session_id: str = "") -> None:
        self._store.log_event(
            EpisodicEvent(event_type="query", content=query, session_id=session_id)
        )

    def record_view(self, paper_id: str, title: str, session_id: str = "") -> None:
        self._store.log_event(
            EpisodicEvent(
                event_type="view",
                content=paper_id,
                paper_id=paper_id,
                paper_title=title,
                session_id=session_id,
            )
        )

    def record_feedback(
        self,
        paper_id: str,
        title: str,
        feedback: str,
        session_id: str = "",
    ) -> None:
        event_type = "like" if feedback == "up" else "dislike"
        self._store.log_event(
            EpisodicEvent(
                event_type=event_type,  # type: ignore[arg-type]
                content=paper_id,
                paper_id=paper_id,
                paper_title=title,
                feedback=feedback,  # type: ignore[arg-type]
                session_id=session_id,
            )
        )

    def get_recent_queries(self, n: int = 20) -> list[str]:
        return self._store.get_recent_queries(n)

    def get_liked_paper_ids(self, n: int = 50) -> list[str]:
        return self._store.get_liked_paper_ids(n)

    def get_disliked_paper_ids(self, n: int = 50) -> list[str]:
        return self._store.get_disliked_paper_ids(n)

    def has_crawl_record(self, conference: str, year: int) -> bool:
        return self._store.has_crawl_record(conference, year)

    def log_crawl(
        self,
        conference: str,
        year: int,
        decision: str | None,
        paper_count: int,
        status: str,
    ) -> None:
        self._store.log_crawl(conference, year, decision, paper_count, status)
