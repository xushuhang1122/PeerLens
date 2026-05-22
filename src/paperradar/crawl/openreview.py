from __future__ import annotations

import time
from typing import Iterator, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..schemas.paper import DecisionType, Paper

_HEADERS = {"User-Agent": settings.openreview.user_agent}


def _normalize_decision(venue: str, conference: str) -> DecisionType:
    v = venue.lower()
    if "oral" in v:
        return "oral"
    if "spotlight" in v or "notable" in v:
        return "spotlight"
    if "poster" in v:
        return "poster"
    if "accept" in v:
        return "accepted"
    if "reject" in v:
        return "rejected"
    # Papers with no venue string or "submitted to..." are unreviewed / rejected without full decision
    if not v or v.startswith("submitted") or "withdraw" in v:
        return "rejected"
    return "unknown"


def _extract_paper(note: dict, conference: str, year: int) -> Optional[Paper]:
    content = note.get("content", {})

    def _val(field: str) -> str:
        v = content.get(field, {})
        if isinstance(v, dict):
            return v.get("value", "") or ""
        return str(v) if v else ""

    def _list_val(field: str) -> list[str]:
        v = content.get(field, {})
        if isinstance(v, dict):
            v = v.get("value", [])
        if isinstance(v, list):
            return [str(x) for x in v]
        if isinstance(v, str):
            return [v] if v else []
        return []

    title = _val("title")
    if not title:
        return None

    venue = _val("venue") or _val("venueid")
    decision = _normalize_decision(venue, conference)

    return Paper(
        id=note.get("id", ""),
        number=note.get("number"),
        title=title,
        authors=_list_val("authors"),
        abstract=_val("abstract"),
        keywords=_list_val("keywords"),
        primary_area=_val("primary_area") or _val("area"),
        venue=venue,
        decision=decision,
        forum_url=f"https://openreview.net/forum?id={note.get('id', '')}",
        conference=conference,
        year=year,
    )


class OpenReviewClient:
    """Fetches papers from OpenReview API v2."""

    def __init__(self) -> None:
        self._cfg = settings.openreview

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def _get(self, params: dict) -> dict:
        resp = requests.get(
            self._cfg.base_url,
            params=params,
            headers=_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def paginate(
        self,
        venue_id: str,
        venue_pattern: Optional[str] = None,
    ) -> Iterator[dict]:
        """Yield raw note dicts from OpenReview with pagination."""
        offset = 0
        while True:
            params: dict = {
                "content.venueid": venue_id,
                "details": "replyCount,invitation",
                "sort": "number:desc",
                "limit": self._cfg.limit,
                "offset": offset,
            }
            if venue_pattern:
                params["content.venue"] = venue_pattern
            data = self._get(params)
            notes = data.get("notes", [])
            if not notes:
                break
            for note in notes:
                yield note
            if len(notes) < self._cfg.limit:
                break
            offset += self._cfg.limit
            time.sleep(self._cfg.batch_sleep)

    def fetch_papers(
        self,
        conference: str,
        year: int,
        decision: Optional[str] = None,
    ) -> list[Paper]:
        """Fetch papers for a conference/year, optionally filtered by decision."""
        conf_cfg = settings.conferences.CONFERENCES.get(conference)
        if not conf_cfg:
            raise ValueError(f"Unknown conference: {conference}")

        venue_id = conf_cfg["venue_id"].format(year=year)
        decision_patterns = conf_cfg.get("decisions", {})

        if decision and decision in decision_patterns:
            # Fetch a specific decision type using its known venue-string pattern
            venue_pattern = decision_patterns[decision].format(year=year)
            notes_iter = self.paginate(venue_id, venue_pattern)
        else:
            # Fetch ALL submissions for this venue (accepted + rejected + withdrawn).
            # Decision is auto-detected from content.venue via _normalize_decision.
            notes_iter = self.paginate(venue_id)

        papers = []
        for note in notes_iter:
            p = _extract_paper(note, conference, year)
            if p:
                papers.append(p)
        return papers
