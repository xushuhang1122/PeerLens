from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from typing import Any

import requests

from ..config import settings

_ATOM_NS = "http://www.w3.org/2005/Atom"
_OR_API = "https://api2.openreview.net"
_ARXIV_API = "http://export.arxiv.org/api/query"
_REQUEST_TIMEOUT = 20


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract up to 10000 words from a PDF using pypdf."""
    try:
        from ..utils.pdf_parser import extract_paper_text, extract_body_text
        # extract_paper_text has its own word limit; use a higher limit for full body
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        full_text = "\n".join(pages_text)
        return extract_body_text(full_text, max_words=10_000)
    except Exception:
        return ""


def fetch_paper_from_openreview(forum_id: str) -> dict[str, Any]:
    """
    Fetch paper metadata from OpenReview by forum ID.
    Returns dict with keys: title, authors, abstract, venue, full_text, source_url, paper_id.
    Falls back to abstract-only if PDF is unavailable.
    """
    source_url = f"https://openreview.net/forum?id={forum_id}"
    try:
        resp = requests.get(
            f"{_OR_API}/notes",
            params={"forum": forum_id, "details": "replyCount"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        notes = resp.json().get("notes", [])
    except Exception as e:
        return {
            "title": "", "authors": [], "abstract": "",
            "venue": "", "full_text": "", "source_url": source_url,
            "paper_id": forum_id, "error": str(e),
        }

    # The submission note is typically the first note with a "title" in content
    paper_note = None
    for note in notes:
        content = note.get("content", {})
        if isinstance(content.get("title"), dict) and content["title"].get("value"):
            paper_note = note
            break
        if isinstance(content.get("title"), str) and content["title"]:
            paper_note = note
            break

    if paper_note is None and notes:
        paper_note = notes[0]

    if paper_note is None:
        return {
            "title": "", "authors": [], "abstract": "",
            "venue": "", "full_text": "", "source_url": source_url,
            "paper_id": forum_id,
        }

    def _v(field: Any) -> str:
        if isinstance(field, dict):
            return str(field.get("value", ""))
        return str(field) if field else ""

    def _list_v(field: Any) -> list[str]:
        if isinstance(field, dict):
            val = field.get("value", [])
        else:
            val = field or []
        return [str(x) for x in (val if isinstance(val, list) else [val])]

    content = paper_note.get("content", {})
    title = _v(content.get("title", ""))
    abstract = _v(content.get("abstract", ""))
    authors = _list_v(content.get("authors", []))
    venue = _v(content.get("venue", ""))
    full_text = abstract

    # Attempt to fetch PDF for richer body text
    try:
        pdf_resp = requests.get(
            f"https://openreview.net/pdf?id={forum_id}",
            timeout=_REQUEST_TIMEOUT,
        )
        if pdf_resp.status_code == 200 and pdf_resp.headers.get("content-type", "").startswith("application/pdf"):
            full_text = _extract_pdf_text(pdf_resp.content) or abstract
    except Exception:
        pass

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "venue": venue,
        "full_text": full_text,
        "source_url": source_url,
        "paper_id": forum_id,
    }


def fetch_paper_from_arxiv(arxiv_id_or_url: str) -> dict[str, Any]:
    """
    Fetch paper metadata from ArXiv by ID or URL.
    Returns dict with keys: title, authors, abstract, venue, full_text, source_url, paper_id.
    """
    # Normalize to bare ID
    arxiv_id = arxiv_id_or_url.strip()
    arxiv_id = re.sub(r"https?://arxiv\.org/(abs|pdf)/", "", arxiv_id)
    arxiv_id = arxiv_id.replace(".pdf", "").strip("/")

    source_url = f"https://arxiv.org/abs/{arxiv_id}"

    try:
        resp = requests.get(
            _ARXIV_API,
            params={"id_list": arxiv_id, "max_results": "1"},
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as e:
        return {
            "title": "", "authors": [], "abstract": "",
            "venue": f"arXiv:{arxiv_id}", "full_text": "", "source_url": source_url,
            "paper_id": arxiv_id, "error": str(e),
        }

    ns = {"atom": _ATOM_NS}
    entry = root.find("atom:entry", ns)
    if entry is None:
        return {
            "title": "", "authors": [], "abstract": "",
            "venue": f"arXiv:{arxiv_id}", "full_text": "", "source_url": source_url,
            "paper_id": arxiv_id,
        }

    def _text(tag: str) -> str:
        el = entry.find(f"atom:{tag}", ns)
        return (el.text or "").strip() if el is not None else ""

    title = _text("title").replace("\n", " ")
    abstract = _text("summary").replace("\n", " ")
    authors = [
        (a.find("atom:name", ns).text or "").strip()
        for a in entry.findall("atom:author", ns)
        if a.find("atom:name", ns) is not None
    ]
    # Try to get published year for venue label
    published = _text("published")
    year = published[:4] if published else ""
    venue = f"arXiv:{arxiv_id}" + (f" ({year})" if year else "")
    full_text = abstract

    # Attempt PDF download
    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        pdf_resp = requests.get(pdf_url, timeout=_REQUEST_TIMEOUT)
        if pdf_resp.status_code == 200:
            full_text = _extract_pdf_text(pdf_resp.content) or abstract
    except Exception:
        pass

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "venue": venue,
        "full_text": full_text,
        "source_url": source_url,
        "paper_id": arxiv_id,
    }
