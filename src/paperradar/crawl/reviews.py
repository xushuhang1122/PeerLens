from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config import settings
from ..schemas.paper import Review

_HEADERS = {"User-Agent": settings.openreview.user_agent}
_BASE = "https://api2.openreview.net/notes"


def _extract_review(note: dict, paper_id: str) -> Optional[Review]:
    invitation = note.get("invitation", "") or ""
    if "Official_Review" not in invitation:
        return None

    content = note.get("content", {})

    def _val(field: str) -> str:
        v = content.get(field, {})
        if isinstance(v, dict):
            return str(v.get("value", "") or "")
        return str(v) if v else ""

    def _num(field: str) -> Optional[float]:
        v = content.get(field, {})
        if isinstance(v, dict):
            v = v.get("value")
        try:
            return float(str(v).split("/")[0].strip()) if v else None
        except (ValueError, AttributeError):
            return None

    strengths = _val("strengths")
    weaknesses = _val("weaknesses")
    questions = _val("questions_and_suggestions") or _val("questions")
    summary = _val("summary")
    soundness = _val("soundness")
    presentation = _val("presentation")
    contribution = _val("contribution")

    parts = [s for s in [summary, strengths, weaknesses, questions] if s]
    full_text = "\n\n".join(parts)

    max_chars = settings.embedding.review_max_tokens * 4
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]

    return Review(
        id=note.get("id", ""),
        forum_id=note.get("forum", ""),
        paper_id=paper_id,
        reviewer_id=note.get("signatures", [""])[0] if note.get("signatures") else "",
        rating=_num("rating"),
        confidence=_num("confidence"),
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
    url = f"{_BASE}?forum={forum_id}"
    async with session.get(url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return data.get("notes", [])


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
            except Exception:
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
