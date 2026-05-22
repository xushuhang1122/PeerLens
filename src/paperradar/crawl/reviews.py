from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import settings
from ..schemas.paper import Review

_HEADERS = {"User-Agent": settings.openreview.user_agent}
_BASE = "https://api2.openreview.net/notes"

# Invitation substrings that indicate non-review notes
_EXCLUDE_INVITATION = (
    "rebuttal",
    "response",
    "meta_review",
    "decision",
    "desk_rejection",
    "withdraw",
    "ethics_review",
    "ac_recommendation",
    "program_chair",
    "sac_report",
    "senior_area_chair",
    "author_checklist",
    "camera_ready",
    "supplementary",
)


def _is_review_invitation(invitation: str) -> bool:
    inv = invitation.lower()
    if "review" not in inv:
        return False
    return not any(pat in inv for pat in _EXCLUDE_INVITATION)


def _extract_review(note: dict, paper_id: str) -> Optional[Review]:
    # API v2 uses "invitations" (list); fall back to legacy "invitation" (string)
    raw_invitations = note.get("invitations") or []
    if not raw_invitations:
        legacy = note.get("invitation", "") or ""
        raw_invitations = [legacy] if legacy else []

    invitation = next(
        (inv for inv in raw_invitations if _is_review_invitation(inv)),
        None,
    )
    if invitation is None:
        return None

    content = note.get("content", {})

    def _val(*fields: str) -> str:
        for field in fields:
            v = content.get(field, {})
            if isinstance(v, dict):
                s = str(v.get("value", "") or "")
            else:
                s = str(v) if v else ""
            if s:
                return s
        return ""

    def _num(*fields: str) -> Optional[float]:
        for field in fields:
            v = content.get(field, {})
            if isinstance(v, dict):
                v = v.get("value")
            try:
                if v is not None and str(v).strip():
                    return float(str(v).split("/")[0].strip())
            except (ValueError, AttributeError):
                pass
        return None

    summary = _val("summary", "summary_of_the_paper", "paper_summary", "overview")

    # Handle venues that combine strengths/weaknesses into one field (e.g. older ICLR)
    combined_sw = _val("strength_and_weaknesses", "strengths_and_weaknesses")
    if combined_sw:
        strengths = combined_sw
        weaknesses = ""
    else:
        strengths = _val("strengths", "strength", "pros", "positive_aspects")
        weaknesses = _val("weaknesses", "weakness", "limitations", "cons", "negative_aspects")

    questions = _val(
        "questions",
        "questions_and_suggestions",
        "questions_for_the_authors",
        "additional_comments",
        "comments",
    )
    soundness = _val("soundness", "technical_quality", "correctness", "technical_correctness")
    presentation = _val("presentation", "clarity", "writing_quality", "clarity_and_writing")
    contribution = _val("contribution", "originality", "novelty_and_significance", "significance")

    parts = [s for s in [summary, strengths, weaknesses, questions] if s]
    full_text = "\n\n".join(parts)

    max_chars = settings.embedding.review_max_tokens * 4
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]

    # Skip notes with no usable text at all
    if not full_text.strip():
        return None

    return Review(
        id=note.get("id", ""),
        forum_id=note.get("forum", ""),
        paper_id=paper_id,
        reviewer_id=note.get("signatures", [""])[0] if note.get("signatures") else "",
        rating=_num("rating", "recommendation", "score", "overall"),
        confidence=_num("confidence", "reviewer_confidence"),
        summary=summary,
        soundness=soundness,
        presentation=presentation,
        contribution=contribution,
        strengths=strengths,
        weaknesses=weaknesses,
        questions=questions,
        full_text=full_text,
        invitation=invitation,
    )


async def _fetch_forum_notes_async(
    session: aiohttp.ClientSession, forum_id: str
) -> list[dict]:
    # limit=1000 prevents the default pagination cutoff (typically 25 notes)
    url = f"{_BASE}?forum={forum_id}&limit=1000"
    for attempt in range(4):
        try:
            async with session.get(url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                data = await resp.json()
                return data.get("notes", [])
        except aiohttp.ClientResponseError:
            if attempt == 3:
                raise
            await asyncio.sleep(2 ** attempt * 2)
    return []


async def _fetch_reviews_batch(
    paper_ids: list[str],
    concurrency: int,
) -> dict[str, list[Review]]:
    results: dict[str, list[Review]] = {}
    semaphore = asyncio.Semaphore(concurrency)

    async def _fetch_one(paper_id: str, session: aiohttp.ClientSession) -> None:
        async with semaphore:
            try:
                notes = await _fetch_forum_notes_async(session, paper_id)
                reviews = []
                for note in notes:
                    r = _extract_review(note, paper_id)
                    if r:
                        reviews.append(r)
                results[paper_id] = reviews
                await asyncio.sleep(settings.openreview.single_sleep)
            except Exception as exc:
                print(f"[reviews] fetch failed for {paper_id}: {exc}")
                results[paper_id] = []

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_one(pid, session) for pid in paper_ids]
        await asyncio.gather(*tasks)

    return results


class ReviewFetcher:
    """Async batch fetcher for OpenReview forum notes (reviews)."""

    def fetch_reviews_for_papers(
        self,
        paper_ids: list[str],
        concurrency: int | None = None,
    ) -> dict[str, list[Review]]:
        c = concurrency or settings.openreview.async_concurrency
        return asyncio.run(_fetch_reviews_batch(paper_ids, c))
