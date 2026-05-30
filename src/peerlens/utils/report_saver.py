from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def _slug(text: str, max_len: int = 50) -> str:
    s = re.sub(r"[^\w\s-]", "", text).strip()
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:max_len]


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(report_type: str) -> Path:
    p = Path("reports") / report_type
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------

def save_diagnosis_report(report, paper_title: str = "") -> Path:
    from datetime import datetime as _dt
    title = paper_title or report.detected_domain or "untitled"
    path = _ensure_dir("diagnosis") / f"{_ts()}_{_slug(title)}.md"

    _REPAIR_LABEL = {
        "one_day_revision": "Revision (< 1 day)",
        "needs_experiment": "Needs experiment",
        "needs_redesign": "Structural redesign",
    }
    _NATURE_LABEL = {
        "content_missing": "Content missing",
        "expression_issue": "Expression issue",
        "design_flaw": "Design flaw",
    }
    _REPAIR_ORDER = {"one_day_revision": 0, "needs_experiment": 1, "needs_redesign": 2}
    _CONF_ORDER = {"high": 0, "medium": 1, "low": 2}

    lines: list[str] = [
        f"# Diagnosis Report — {title}",
        "",
        f"_Generated: {_dt.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
    ]

    if report.detected_keywords:
        lines += [f"**Domain:** {report.detected_domain}  |  "
                  f"**Keywords:** {' · '.join(report.detected_keywords)}", ""]

    if report.executive_summary:
        lines += ["## Executive Summary", "", report.executive_summary, ""]

    if report.findings:
        lines += ["## Priority Fix List", ""]
        evidence_map = {u.finding_id: u for u in report.evidence_updates}
        sorted_findings = sorted(
            report.findings,
            key=lambda f: (
                _REPAIR_ORDER.get(f.repair_cost, 9),
                1 if "[counterevidence" in f.confidence_reason else 0,
                _CONF_ORDER.get(f.confidence, 9),
            ),
        )
        for f in sorted_findings:
            repair = _REPAIR_LABEL.get(f.repair_cost, f.repair_cost)
            nature = _NATURE_LABEL.get(f.nature, f.nature)
            primary_mark = " **(primary)**" if getattr(f, "is_primary", False) else ""
            lines.append(f"### {f.id}{primary_mark} — [{repair}] [{nature}]")
            lines.append(f"**Confidence:** {f.confidence}")
            lines.append("")
            lines.append(f.problem)
            if f.confidence_reason:
                lines += ["", f"_Note: {f.confidence_reason}_"]
            ev = evidence_map.get(f.id)
            if ev and ev.evidence_quote:
                lines += ["", f"> Evidence ({ev.verdict}): {ev.evidence_quote}"]
            lines.append("")

    quick_fixes = [f for f in report.findings if f.repair_cost == "one_day_revision"]
    if quick_fixes:
        lines += ["## Specific Suggestions", ""]
        for f in quick_fixes:
            steps = getattr(f, "action_steps", [])
            lines.append(f"**{f.id}:** {f.problem}")
            for i, step in enumerate(steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

    if report.writing_issues:
        lines += ["## Writing Issues", ""]
        for wi in report.writing_issues:
            if wi.quote:
                lines.append(f'- `"{wi.quote}"` — {wi.issue}')
                if wi.suggestion:
                    lines.append(f'  - Suggestion: {wi.suggestion}')
        lines.append("")

    if report.similar_accepted:
        lines += ["## Similar Accepted Papers", ""]
        for p in report.similar_accepted:
            lines.append(f"- **{p.title}** — {p.conference} {p.year} `{p.decision}`")
        lines.append("")

    if report.similar_rejected:
        lines += ["## Similar Rejected Papers", ""]
        for p in report.similar_rejected:
            lines.append(f"- **{p.title}** — {p.conference} {p.year} `{p.decision}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Research (survey)
# ---------------------------------------------------------------------------

def save_survey_report(report, query: str = "") -> Path:
    from datetime import datetime as _dt
    title = report.title or query or "untitled"
    path = _ensure_dir("research") / f"{_ts()}_{_slug(title)}.md"

    lines: list[str] = [
        f"# {report.title}",
        "",
        f"_Generated: {_dt.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_",
        f"_Query: {query}_",
        "",
    ]

    if report.background:
        lines += ["## Background", "", report.background, ""]

    for section in report.sections:
        lines += [f"## {section.heading}", "", section.content, ""]

    if report.open_questions:
        lines += ["## Open Questions", ""]
        lines += [f"- {q}" for q in report.open_questions]
        lines.append("")

    if report.submission_advice:
        lines += ["## Submission Advice", "", report.submission_advice, ""]

    if report.key_papers:
        lines += ["## Key Papers", ""]
        for p in report.key_papers[:10]:
            link = f"[{p.title}]({p.forum_url})" if p.forum_url else p.title
            lines.append(f"- **{link}** — {p.conference} {p.year} `{p.decision}`")
            if p.abstract:
                lines.append(f"  > {p.abstract[:200]}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def save_reading_report(report, source: str = "") -> Path:
    from datetime import datetime as _dt
    title = report.paper_title or "untitled"
    path = _ensure_dir("reading") / f"{_ts()}_{_slug(title)}.md"

    lines: list[str] = [f"# {title}", ""]
    if report.authors:
        lines.append(f"**Authors:** {', '.join(report.authors)}")
    if report.venue:
        lines.append(f"**Venue:** {report.venue}")
    if source or report.source_url:
        lines.append(f"**Source:** {source or report.source_url}")
    lines += ["", f"_Generated: {_dt.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_", ""]

    if report.tldr:
        lines += ["## TL;DR", "", report.tldr, ""]

    if report.problem_statement:
        lines += ["## Problem Statement", "", report.problem_statement, ""]

    if report.core_contributions:
        lines += ["## Core Contributions", ""]
        lines += [f"- {c}" for c in report.core_contributions]
        lines.append("")

    if report.methodology_summary:
        lines += ["## Methodology", "", report.methodology_summary, ""]

    if report.key_innovations:
        lines += ["## Key Innovations", ""]
        lines += [f"- {i}" for i in report.key_innovations]
        lines.append("")

    if report.main_results:
        lines += ["## Main Results", "", report.main_results, ""]

    if report.ablations:
        lines += ["## Ablations", "", report.ablations, ""]

    if report.limitations:
        lines += ["## Limitations", ""]
        lines += [f"- {l}" for l in report.limitations]
        lines.append("")

    if report.open_questions:
        lines += ["## Open Questions", ""]
        lines += [f"- {q}" for q in report.open_questions]
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
