from __future__ import annotations

import io
import re
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Stage 0: rule-based PDF noise cleaning
# ---------------------------------------------------------------------------

_ALPHANUM_WHITELIST = frozenset({
    "GPT4", "GPT3", "GPT2", "T5", "H100", "A100", "A10",
    "COVID19", "COVID", "LLAMA3", "LLAMA2", "BERT", "ROBERTA",
    "LLAMA", "LLAVA", "DALL", "GPT",
})

_LINE_NUM_RE = re.compile(r"\b([A-Za-z]{4,})(\d{2,})\b")
_HYPHEN_RE = re.compile(r"(\w+)-\s*\n\s*(\w+)")
_FIGURE_CAPTION_RE = re.compile(
    r"(?im)^(fig(?:ure)?\.?\s+\d+[\.:)]|table\s+\d+[\.:)])[^\n]*$"
)


def _clean_line_artifacts(text: str) -> tuple[str, list[str]]:
    noise: list[str] = []

    def _replacer(m: re.Match) -> str:
        word, digits = m.group(1), m.group(2)
        upper = (word + digits).upper()
        if upper in _ALPHANUM_WHITELIST or (word + digits) in _ALPHANUM_WHITELIST:
            return m.group(0)
        noise.append(m.group(0))
        return word

    cleaned = _LINE_NUM_RE.sub(_replacer, text)
    return cleaned, noise


def _clean_hyphen_breaks(text: str) -> str:
    return _HYPHEN_RE.sub(r"\1\2", text)


def _clean_figure_captions(text: str) -> tuple[str, list[str]]:
    noise: list[str] = []

    def _replacer(m: re.Match) -> str:
        noise.append(m.group(0))
        return ""

    cleaned = _FIGURE_CAPTION_RE.sub(_replacer, text)
    return cleaned, noise


def _split_into_chunks(text: str) -> list[str]:
    section_re = re.compile(r"(?m)^(?:#+\s+|\d+\.?\s+[A-Z])")
    boundaries = [m.start() for m in section_re.finditer(text)]
    if len(boundaries) < 2:
        mid = len(text) // 2
        return [text[:mid], text[mid:]]
    chunks: list[str] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chunks.append(text[start:end])
    return chunks


def preprocess_paper_text(text: str, context_limit: int = 120_000) -> dict:
    """
    Stage 0 preprocessing: clean PDF extraction noise, optionally chunk.
    Returns {"clean_text", "is_chunked", "chunks", "noise_log"}.
    """
    noise_log: list[str] = []

    step1 = _clean_hyphen_breaks(text)
    step2, n2 = _clean_line_artifacts(step1)
    noise_log.extend(n2)
    step3, n3 = _clean_figure_captions(step2)
    noise_log.extend(n3)

    clean_text = step3
    token_estimate = len(clean_text.split()) * 1.3
    threshold = context_limit * 0.6

    if token_estimate > threshold:
        chunks = _split_into_chunks(clean_text)
        return {"clean_text": clean_text, "is_chunked": True, "chunks": chunks, "noise_log": noise_log}

    return {"clean_text": clean_text, "is_chunked": False, "chunks": [], "noise_log": noise_log}


# ---------------------------------------------------------------------------

_BODY_END_PATTERNS = [
    re.compile(r"^\s*(Appendix|APPENDIX)\b", re.MULTILINE),
    re.compile(r"^\s*(References|REFERENCES|Bibliography|BIBLIOGRAPHY)\s*$", re.MULTILINE),
    re.compile(r"^\s*Acknowledgements?\s*$", re.MULTILINE | re.IGNORECASE),
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


def _table_to_markdown(table: list) -> str:
    if not table:
        return ""
    rows = []
    for i, row in enumerate(table):
        cells = [str(cell or "").replace("\n", " ").strip() for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
        if i == 0:
            rows.append("|" + "|".join(["---"] * len(cells)) + "|")
    return "\n".join(rows)


def _extract_tables_markdown(pdf_bytes: bytes) -> str:
    """Use pdfplumber to extract tables and return them as numbered markdown blocks."""
    try:
        import pdfplumber
        sections: list[str] = []
        table_count = 0
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                for table in tables:
                    if not table or all(
                        all(cell is None or str(cell).strip() == "" for cell in row)
                        for row in table
                    ):
                        continue
                    table_count += 1
                    md = _table_to_markdown(table)
                    if md:
                        sections.append(f"### Table {table_count} (page {page_num})\n\n{md}")
        return "\n\n".join(sections)
    except Exception:
        return ""


def extract_full_paper_text(pdf_bytes: bytes) -> str:
    """Extract full paper body via pypdf, then append tables from pdfplumber."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    body = extract_body_text(raw, max_words=100_000)

    tables_md = _extract_tables_markdown(pdf_bytes)
    if tables_md:
        body += "\n\n---\n## Extracted Tables\n\n" + tables_md

    return body


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

_NUMBERED_SECTION_RE = re.compile(
    r"(?:^|\n)([1-9])\s*\.?\s+([A-Z][A-Za-z][A-Za-z &/\-:,]{1,50})(?=\n|\s{2}|$)",
    re.MULTILINE,
)

_KEYWORD_ANCHORS: list[tuple[str, str]] = [
    ("introduction", "Introduction"),
    ("related work", "Related Work"),
    ("background", "Background"),
    ("preliminaries", "Preliminaries"),
    ("method", "Method"),
    ("methodology", "Methodology"),
    ("approach", "Approach"),
    ("framework", "Framework"),
    ("model", "Model"),
    ("experiment", "Experiments"),
    ("evaluation", "Evaluation"),
    ("result", "Results"),
    ("analysis", "Analysis"),
    ("discussion", "Discussion"),
    ("ablation", "Ablation Study"),
    ("conclusion", "Conclusion"),
    ("limitation", "Limitations"),
]


def _words(text: str) -> list[str]:
    return text.split()


def _truncate(text: str, max_words: int) -> str:
    w = _words(text)
    return " ".join(w[:max_words]) + ("\n[truncated]" if len(w) > max_words else "")


def _build_sections_from_spans(
    text: str,
    spans: list[tuple[int, int, str]],
    max_words_per_section: int,
) -> "OrderedDict[str, str]":
    """Given (start, end, label) spans, return section dict."""
    result: OrderedDict[str, str] = OrderedDict()
    for start, end, label in spans:
        body = text[start:end].strip()
        result[label] = _truncate(body, max_words_per_section)
    return result


def split_into_sections(
    text: str,
    max_words_per_section: int = 2000,
) -> "OrderedDict[str, str]":
    """
    Split paper body text into named sections.
    Returns OrderedDict: section_label -> text (truncated to max_words_per_section).
    Three strategies in order: numbered headers, keyword headers, equal-chunk fallback.
    """
    # --- Strategy 1: numbered top-level sections (e.g. "1 Introduction", "2. Method") ---
    matches = list(_NUMBERED_SECTION_RE.finditer(text))
    if len(matches) >= 2:
        spans: list[tuple[int, int, str]] = []
        for i, m in enumerate(matches):
            label = f"{m.group(1)}. {m.group(2).strip()}"
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            spans.append((start, end, label))
        return _build_sections_from_spans(text, spans, max_words_per_section)

    # --- Strategy 2: keyword-based header detection ---
    found: list[tuple[int, str]] = []
    for keyword, label in _KEYWORD_ANCHORS:
        pat = re.compile(
            r"(?:^|\n)(?:\d+\.?\s+)?" + re.escape(keyword) + r"s?\s*(?:\n|$)",
            re.IGNORECASE,
        )
        for m in pat.finditer(text):
            found.append((m.end(), label))
    if len(found) >= 2:
        found.sort(key=lambda x: x[0])
        deduped: list[tuple[int, str]] = [found[0]]
        for pos, label in found[1:]:
            if pos - deduped[-1][0] > 80:
                deduped.append((pos, label))
        spans = []
        for i, (start, label) in enumerate(deduped):
            end = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
            spans.append((start, end, label))
        return _build_sections_from_spans(text, spans, max_words_per_section)

    # --- Strategy 3: equal-chunk fallback ---
    words = _words(text)
    n = max(len(words) // 4, 100)
    labels = ["Introduction", "Methods", "Experiments", "Conclusion"]
    result: OrderedDict[str, str] = OrderedDict()
    for i, lbl in enumerate(labels):
        chunk = words[i * n: (i + 1) * n]
        result[lbl] = _truncate(" ".join(chunk), max_words_per_section)
    return result


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
