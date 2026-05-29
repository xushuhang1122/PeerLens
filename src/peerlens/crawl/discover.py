from __future__ import annotations

import time
from collections import Counter
from typing import Optional

import requests

from ..config import settings
from ..schemas.tools import ConferenceFieldInfo, DiscoverConferenceOutput

_HEADERS = {"User-Agent": settings.openreview.user_agent}
_BASE = "https://api2.openreview.net/notes"


def _probe_venue(venue_id: str, limit: int = 20) -> Optional[list[dict]]:
    try:
        resp = requests.get(
            _BASE,
            params={
                "content.venueid": venue_id,
                "limit": limit,
                "sort": "number:desc",
            },
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("notes", [])
    except Exception:
        return None


def _probe_forum_reviews(forum_id: str) -> list[str]:
    """Return invitation strings for the first forum's notes."""
    try:
        resp = requests.get(
            _BASE,
            params={"forum": forum_id, "limit": 30},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        notes = resp.json().get("notes", [])
        return [n.get("invitation", "") for n in notes if n.get("invitation")]
    except Exception:
        return []


def discover_venue(venue_id: str) -> DiscoverConferenceOutput:
    notes = _probe_venue(venue_id)

    if notes is None:
        return DiscoverConferenceOutput(
            found=False, venue_id=venue_id, error="API request failed."
        )
    if not notes:
        return DiscoverConferenceOutput(
            found=False, venue_id=venue_id, error="No papers found for this venue_id."
        )

    # Collect content fields across all sample papers
    field_counter: Counter = Counter()
    venue_values: Counter = Counter()
    titles: list[str] = []

    for note in notes:
        content = note.get("content", {})
        for k in content.keys():
            field_counter[k] += 1
        venue_val = content.get("venue", {})
        if isinstance(venue_val, dict):
            venue_val = venue_val.get("value", "")
        if venue_val:
            venue_values[str(venue_val)] += 1
        title = content.get("title", {})
        if isinstance(title, dict):
            title = title.get("value", "")
        if title and len(titles) < 5:
            titles.append(str(title))

    # Probe review invitations from first paper
    review_inv: Optional[str] = None
    if notes:
        invitations = _probe_forum_reviews(notes[0].get("id", ""))
        time.sleep(settings.openreview.single_sleep)
        review_invs = [i for i in invitations if "Review" in i or "review" in i]
        if review_invs:
            # Strip paper-specific prefix, keep invitation template
            review_inv = review_invs[0].split("/-/")[0] + "/-/Official_Review"

    # Summarize filterable fields
    common_fields = [f for f, c in field_counter.most_common(15)]
    decision_patterns = [v for v, _ in venue_values.most_common(10)]

    # Build human-readable summary
    lines = [
        f"Venue: {venue_id}",
        f"Sample size: {len(notes)} papers",
        f"",
        f"Available content fields: {', '.join(common_fields)}",
        f"",
        f"Detected venue/decision patterns:",
    ]
    for pat in decision_patterns:
        lines.append(f"  - \"{pat}\"")
    if review_inv:
        lines.append(f"")
        lines.append(f"Review invitation pattern: {review_inv}")
    lines.append(f"")
    lines.append(
        "To crawl this conference, use check_and_crawl_new_conference with the venue_id above. "
        "To filter by decision in search_papers, use the exact strings from 'venue/decision patterns'."
    )

    info = ConferenceFieldInfo(
        venue_id=venue_id,
        paper_count=len(notes),
        content_fields=common_fields,
        decision_patterns=decision_patterns,
        review_invitation_pattern=review_inv,
        sample_titles=titles,
        filterable_summary="\n".join(lines),
    )
    return DiscoverConferenceOutput(found=True, venue_id=venue_id, info=info)
