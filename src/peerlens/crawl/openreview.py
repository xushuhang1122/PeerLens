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
    # Empty venue = paper was not accepted (rejected / withdrawn / no final decision)
    if not v or v.startswith("submitted") or "withdraw" in v:
        return "rejected"
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

    # Use content.venue (human-readable, e.g. "ICLR 2024 rejected") for decision detection.
    # Do NOT fall back to venueid — a venueid like "ICLR.cc/2024/Conference" is not a decision string.
    venue_str = _val("venue")
    decision = _normalize_decision(venue_str, conference)

    return Paper(
        id=note.get("id", ""),
        number=note.get("number"),
        title=title,
        authors=_list_val("authors"),
        abstract=_val("abstract"),
        keywords=_list_val("keywords"),
        primary_area=_val("primary_area") or _val("area"),
        venue=venue_str or _val("venueid"),
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
        skip_venue_id: bool = False,
    ) -> Iterator[dict]:
        """Yield raw note dicts from OpenReview with pagination.

        skip_venue_id: omit the content.venueid filter, querying by content.venue only.
        Needed for ICLR rejected papers which may have a different venueid than accepted ones.
        """
        offset = 0
        while True:
            params: dict = {
                "details": "replyCount,invitation",
                "sort": "number:desc",
                "limit": self._cfg.limit,
                "offset": offset,
            }
            if not skip_venue_id:
                params["content.venueid"] = venue_id
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
            venue_pattern = decision_patterns[decision].format(year=year)
            # Rejected papers on ICLR may live under a different venueid — skip the filter
            skip = (decision == "rejected")
            notes_iter = self.paginate(venue_id, venue_pattern, skip_venue_id=skip)
            papers = []
            for note in notes_iter:
                p = _extract_paper(note, conference, year)
                if p:
                    papers.append(p)
            return papers

        if decision_patterns:
            # Fetch each defined decision category separately.
            # For "rejected" entries, skip the venueid filter so papers with a
            # different venueid (common for ICLR rejected) are still found.
            papers: list[Paper] = []
            for dec, pattern in decision_patterns.items():
                vp = pattern.format(year=year)
                skip = (dec == "rejected")
                for note in self.paginate(venue_id, vp, skip_venue_id=skip):
                    p = _extract_paper(note, conference, year)
                    if p:
                        papers.append(p)
                time.sleep(self._cfg.single_sleep)
            return papers

        # No decision patterns defined (ICML, AISTATS, etc.): fetch all by venueid.
        # Decision is inferred from content.venue via _normalize_decision.
        papers = []
        for note in self.paginate(venue_id):
            p = _extract_paper(note, conference, year)
            if p:
                papers.append(p)
        return papers
