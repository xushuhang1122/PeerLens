from __future__ import annotations

import io
import re


_BODY_END_PATTERNS = [
    re.compile(r"^\s*(Appendix|APPENDIX)\b", re.MULTILINE),
    re.compile(r"^\s*A\.\s+[A-Z]", re.MULTILINE),
    re.compile(r"^\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*$", re.MULTILINE),
    re.compile(r"^\s*Acknowledgements?\b", re.MULTILINE | re.IGNORECASE),
]


def extract_body_text(text: str, max_words: int = 10_000) -> str:
    """Strip appendix/references sections, then truncate to max_words."""
    cutoff = len(text)
    for pat in _BODY_END_PATTERNS:
        m = pat.search(text)
        if m and m.start() < cutoff:
            cutoff = m.start()
    body = text[:cutoff]
    words = body.split()
    if len(words) > max_words:
        body = " ".join(words[:max_words]) + "\n[truncated]"
    return body.strip()


def extract_paper_text(pdf_bytes: bytes, max_words: int = 1500) -> str:
    """Extract plain text from a PDF up to max_words, preserving line structure."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text: list[str] = []
    word_count = 0

    for page in reader.pages:
        if word_count >= max_words:
            break
        raw = page.extract_text() or ""
        lines = raw.splitlines()
        kept: list[str] = []
        for line in lines:
            words = line.split()
            if not words:
                kept.append("")
                continue
            remaining = max_words - word_count
            if remaining <= 0:
                break
            if len(words) <= remaining:
                kept.append(line)
                word_count += len(words)
            else:
                kept.append(" ".join(words[:remaining]))
                word_count += remaining
                break
        pages_text.append("\n".join(kept))

    return "\n".join(pages_text).strip()


# Patterns that signal the end of an abstract section
_ABSTRACT_STOP = re.compile(
    r"\n[ \t]*\n"                                           # blank line between paragraphs
    r"|(?:\n[ \t]*)(?:\d+\.?\s+)?introduction\b"           # Introduction / 1. Introduction
    r"|(?:\n[ \t]*)keywords?[ \t]*[:\-—\.]"           # Keywords:
    r"|(?:\n[ \t]*)index[ \t]+terms?[ \t]*[:\-—\.]"   # Index Terms:
    r"|(?:\n[ \t]*)(?:ccs[ \t]+concepts|acm[ \t]+reference[ \t]+format)\b"
    r"|(?:\n[ \t]*)\d+[ \t]+[A-Z][A-Za-z]",               # numbered section "1 Background"
    re.I,
)

_ABSTRACT_MAX_WORDS = 350


def extract_title_abstract(text: str) -> dict[str, str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # --- Title ---
    title = ""
    for line in lines[:15]:
        words = line.split()
        if len(words) < 3 or len(words) > 25:
            continue
        if re.match(
            r"^(preprint|arxiv|under review|submitted|published|proceedings|workshop|conference|journal|\d{4})\b",
            line, re.I,
        ):
            continue
        if sum(c.isalpha() for c in line) / max(len(line), 1) < 0.5:
            continue
        title = line
        break
    if not title:
        title = lines[0][:120] if lines else ""

    # --- Abstract ---
    # Match "Abstract" as a heading (start of line, or with colon/dash), not mid-sentence.
    abstract = ""
    m = re.search(r"(?im)(?:^|\n)[ \t]*abstract[ \t]*[:\-—]?[ \t]*\n?", text)
    if m:
        body = text[m.end():]
        stop = _ABSTRACT_STOP.search(body)
        raw = body[: stop.start()] if stop else body[:2000]
        abstract = re.sub(r"\s+", " ", raw).strip()
        words = abstract.split()
        if len(words) > _ABSTRACT_MAX_WORDS:
            abstract = " ".join(words[:_ABSTRACT_MAX_WORDS])

    if not abstract:
        # Fallback: skip short/email lines after the title, then grab first prose chunk
        title_pos = text.find(title) if title else 0
        body = text[title_pos + len(title) :] if title_pos >= 0 else text
        prose: list[str] = []
        for line in body.splitlines():
            line = line.strip()
            if not line or "@" in line:
                continue
            if not prose and len(line.split()) < 6:
                continue
            prose.append(line)
            if len(" ".join(prose).split()) >= 200:
                break
        abstract = " ".join(" ".join(prose).split()[:200])

    return {"title": title, "abstract": abstract}
