from __future__ import annotations

import io
import re


def extract_paper_text(pdf_bytes: bytes, max_words: int = 1500) -> str:
    """Extract plain text from a PDF, truncated to max_words."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: list[str] = []
    word_count = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        words = text.split()
        remaining = max_words - word_count
        if remaining <= 0:
            break
        parts.append(" ".join(words[:remaining]))
        word_count += min(len(words), remaining)

    return "\n".join(parts).strip()


def extract_title_abstract(text: str) -> dict[str, str]:
    """Heuristically extract title and abstract from paper text."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    title = lines[0] if lines else ""

    abstract = ""
    abstract_match = re.search(
        r"(?i)\babstract\b[\s\n:—-]*(.*?)(?=\n\s*\n|\n\s*(?:1\.?\s*introduction|keywords?)\b)",
        text,
        re.S,
    )
    if abstract_match:
        abstract = re.sub(r"\s+", " ", abstract_match.group(1)).strip()
    else:
        # Fallback: take the first 200 words after skipping title line
        body = " ".join(lines[1:])
        abstract = " ".join(body.split()[:200])

    return {"title": title, "abstract": abstract}
