import sys
sys.path.insert(0, ".")

import streamlit as st

from src.peerlens.utils.pdf_parser import extract_full_paper_text, extract_title_abstract
from src.peerlens.agent.diagnosis_runner import stream_diagnosis_agent

_KNOWN_VENUES = [
    "NeurIPS 2026", "ICML 2026", "ICLR 2026",
    "NeurIPS 2025", "ICML 2025", "ICLR 2025",
    "AISTATS 2026", "UAI 2026", "AAAI 2026",
    "CVPR 2026", "ECCV 2026",
    "ACL 2026", "EMNLP 2026",
    "JMLR",
]

_REPAIR_LABEL = {
    "one_day_revision": "Revision (< 1 day)",
    "needs_experiment": "Needs experiment",
    "needs_redesign": "Structural redesign",
}
_REPAIR_ORDER = {"one_day_revision": 0, "needs_experiment": 1, "needs_redesign": 2}
_REPAIR_COLOR = {"one_day_revision": "green", "needs_experiment": "orange", "needs_redesign": "red"}

_NATURE_LABEL = {
    "content_missing": "Content missing",
    "expression_issue": "Expression issue",
    "design_flaw": "Design flaw",
}
_NATURE_COLOR = {"content_missing": "orange", "expression_issue": "blue", "design_flaw": "red"}

_CONF_ORDER = {"high": 0, "medium": 1, "low": 2}


def _sort_findings(findings):
    return sorted(
        findings,
        key=lambda f: (
            _REPAIR_ORDER.get(f.repair_cost, 9),
            1 if f.confidence == "low" and "[counterevidence" in f.confidence_reason else 0,
            _CONF_ORDER.get(f.confidence, 9),
        ),
    )


def _report_to_markdown(report, paper_title: str = "") -> str:
    from datetime import datetime

    lines: list[str] = []
    title = paper_title or report.detected_domain
    lines += [f"# Diagnosis Report — {title}", ""]
    lines += [f"_Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}_", ""]

    if report.detected_keywords:
        lines += [f"**Domain:** {report.detected_domain}", ""]
        lines += ["**Keywords:** " + " · ".join(f"`{k}`" for k in report.detected_keywords), ""]

    # 1. Executive Summary
    if report.executive_summary:
        lines += ["## Executive Summary", "", report.executive_summary, ""]

    # 2. Priority Fix List
    if report.findings:
        lines += ["## Priority Fix List", ""]
        evidence_map = {u.finding_id: u for u in report.evidence_updates}
        for f in _sort_findings(report.findings):
            repair = _REPAIR_LABEL.get(f.repair_cost, f.repair_cost)
            nature = _NATURE_LABEL.get(f.nature, f.nature)
            lines.append(f"### [{repair}] {f.id}")
            lines.append(f"**Nature:** {nature}  |  **Confidence:** {f.confidence}")
            lines.append("")
            lines.append(f.problem)
            if f.confidence_reason:
                lines.append(f"")
                lines.append(f"_Note: {f.confidence_reason}_")
            ev = evidence_map.get(f.id)
            if ev and ev.evidence_quote:
                lines.append(f"")
                lines.append(f"> Evidence ({ev.verdict}): {ev.evidence_quote}")
            lines.append("")

    # 3. Specific Suggestions (one_day_revision only)
    quick_fixes = [f for f in report.findings if f.repair_cost == "one_day_revision"]
    if quick_fixes:
        lines += ["## Specific Suggestions", ""]
        evidence_map = {u.finding_id: u for u in report.evidence_updates}
        for f in quick_fixes:
            lines.append(f"**{f.id}:** {f.problem}")
            ev = evidence_map.get(f.id)
            if ev and ev.verdict == "confirmed" and ev.evidence_quote:
                lines.append(f"> {ev.evidence_quote}")
            lines.append("")

    # 4. Writing Issues
    if report.writing_issues:
        lines += ["## Writing Issues", ""]
        for wi in report.writing_issues:
            if wi.quote:
                lines.append(f'- `"{wi.quote}"` — {wi.issue}')
                if wi.suggestion:
                    lines.append(f'  - Suggestion: {wi.suggestion}')
        lines.append("")

    # Similar papers
    if report.similar_accepted:
        lines += ["## Similar Accepted Papers", ""]
        for p in report.similar_accepted:
            link = f"[{p.title}]({p.forum_url})" if p.forum_url else p.title
            lines.append(f"- **{link}** — {p.conference} {p.year} `{p.decision}`")
        lines.append("")

    if report.similar_rejected:
        lines += ["## Similar Rejected Papers", ""]
        for p in report.similar_rejected:
            link = f"[{p.title}]({p.forum_url})" if p.forum_url else p.title
            lines.append(f"- **{link}** — {p.conference} {p.year} `{p.decision}`")
        lines.append("")

    return "\n".join(lines)


def _render_report(report) -> None:
    st.subheader(f"Diagnosis Report — {report.detected_domain}")

    if report.detected_keywords:
        st.markdown("**Keywords:** " + " | ".join(f"`{k}`" for k in report.detected_keywords))

    # 1. Executive Summary
    if report.executive_summary:
        st.divider()
        st.subheader("Executive Summary")
        st.info(report.executive_summary)

    # 2. Priority Fix List
    if report.findings:
        st.divider()
        st.subheader("Priority Fix List")
        evidence_map = {u.finding_id: u for u in report.evidence_updates}

        repair_groups: dict[str, list] = {}
        for f in _sort_findings(report.findings):
            repair_groups.setdefault(f.repair_cost, []).append(f)

        for repair_key in ("one_day_revision", "needs_experiment", "needs_redesign"):
            group = repair_groups.get(repair_key, [])
            if not group:
                continue
            repair_label = _REPAIR_LABEL.get(repair_key, repair_key)
            repair_color = _REPAIR_COLOR.get(repair_key, "gray")
            st.markdown(f"**:{repair_color}[{repair_label}]**")
            for f in group:
                nature_label = _NATURE_LABEL.get(f.nature, f.nature)
                nature_color = _NATURE_COLOR.get(f.nature, "gray")
                is_refuted = "[counterevidence" in f.confidence_reason
                with st.container(border=True):
                    col_id, col_tags = st.columns([1, 4])
                    with col_id:
                        st.markdown(f"**{f.id}**")
                    with col_tags:
                        st.markdown(
                            f":{nature_color}[{nature_label}]"
                            f"  confidence: `{f.confidence}`"
                            + (" — :red[counterevidence found]" if is_refuted else "")
                        )
                    st.markdown(f.problem)
                    if f.confidence_reason and f.confidence != "high":
                        st.caption(f.confidence_reason)
                    ev = evidence_map.get(f.id)
                    if ev and ev.evidence_quote:
                        with st.expander(f"Evidence ({ev.verdict})"):
                            st.markdown(f"> {ev.evidence_quote}")

    # 3. Specific Suggestions (one_day_revision)
    quick_fixes = [f for f in report.findings if f.repair_cost == "one_day_revision"]
    if quick_fixes:
        st.divider()
        st.subheader("Specific Suggestions")
        evidence_map = {u.finding_id: u for u in report.evidence_updates}
        for f in quick_fixes:
            ev = evidence_map.get(f.id)
            with st.container(border=True):
                st.markdown(f"**{f.id}:** {f.problem}")
                if ev and ev.verdict == "confirmed" and ev.evidence_quote:
                    st.markdown(f"> {ev.evidence_quote}")

    # 4. Writing Issues
    if report.writing_issues:
        st.divider()
        st.subheader("Writing Issues")
        with st.expander(f"{len(report.writing_issues)} issue(s) found", expanded=False):
            for wi in report.writing_issues:
                if wi.quote:
                    st.markdown(f"- `\"{wi.quote}\"` — {wi.issue}")
                    if wi.suggestion:
                        st.caption(f"Suggestion: {wi.suggestion}")

    # Similar papers tabs
    st.divider()
    tab_acc, tab_rej = st.tabs([
        f"Similar Accepted Papers ({len(report.similar_accepted)})",
        f"Similar Rejected Papers ({len(report.similar_rejected)})",
    ])
    with tab_acc:
        if report.similar_accepted:
            for p in report.similar_accepted:
                st.markdown(
                    f"**[{p.title}]({p.forum_url})** — {p.conference} {p.year} `{p.decision}`"
                    if p.forum_url else
                    f"**{p.title}** — {p.conference} {p.year} `{p.decision}`"
                )
                if p.abstract:
                    st.caption(p.abstract[:250] + "..." if len(p.abstract) > 250 else p.abstract)
        else:
            st.info("No accepted papers found. Make sure the database has papers crawled for this domain.")
    with tab_rej:
        if report.similar_rejected:
            for p in report.similar_rejected:
                st.markdown(
                    f"**[{p.title}]({p.forum_url})** — {p.conference} {p.year} `{p.decision}`"
                    if p.forum_url else
                    f"**{p.title}** — {p.conference} {p.year} `{p.decision}`"
                )
                if p.abstract:
                    st.caption(p.abstract[:250] + "..." if len(p.abstract) > 250 else p.abstract)
        else:
            st.info("No rejected papers found in the database for this domain.")


# ------------------------------------------------------------------
# Page header
# ------------------------------------------------------------------
st.title("Paper Diagnosis")
st.caption(
    "Upload your paper (PDF). The agent will detect its domain, find similar accepted and rejected papers "
    "in the database, and give you a prioritized list of issues with repair cost estimates."
)

# ------------------------------------------------------------------
# If a report already exists, show it and offer to start over
# ------------------------------------------------------------------
existing_report = st.session_state.get("diagnosis_report")

if existing_report:
    meta = st.session_state.get("diagnosis_meta", {})
    col_info, col_btn = st.columns([5, 1])
    with col_info:
        if meta.get("title"):
            st.markdown(f"**Paper:** {meta['title']}")
    with col_btn:
        if st.button("New Diagnosis", type="secondary"):
            for k in ("diagnosis_report", "diagnosis_meta", "diagnosis_running"):
                st.session_state.pop(k, None)
            st.rerun()

    paper_title = meta.get("title", "")
    md_content = _report_to_markdown(existing_report, paper_title)
    safe_name = (paper_title or existing_report.detected_domain).replace(" ", "_")[:60]
    st.download_button(
        label="Download Report (.md)",
        data=md_content.encode("utf-8"),
        file_name=f"diagnosis_{safe_name}.md",
        mime="text/markdown",
    )

    st.divider()
    _render_report(existing_report)

else:
    # ------------------------------------------------------------------
    # Upload form
    # ------------------------------------------------------------------
    uploaded = st.file_uploader("Upload paper (PDF)", type=["pdf"])

    if uploaded is not None:
        pdf_bytes = uploaded.read()
        with st.spinner("Extracting text..."):
            paper_text = extract_full_paper_text(pdf_bytes)
            meta = extract_title_abstract(paper_text)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Detected title:** {meta['title']}")
        with col2:
            st.caption(f"{len(paper_text.split())} words extracted")

        with st.expander("Abstract preview"):
            st.write(meta["abstract"] or paper_text[:500])

        st.divider()

        target_venue = st.selectbox(
            "Target venue (optional)",
            options=[""] + _KNOWN_VENUES,
            format_func=lambda x: "Not specified (general review)" if x == "" else x,
            help="Select or type a venue to get feedback tailored to that conference's review criteria.",
        )
        custom_venue = st.text_input(
            "Or enter a custom venue",
            placeholder="e.g. COLM 2026, TMLR, CoRL 2026",
        )
        effective_venue = custom_venue.strip() if custom_venue.strip() else target_venue

        if "diagnosis_running" not in st.session_state:
            st.session_state.diagnosis_running = False

        if st.button("Run Diagnosis", type="primary", disabled=st.session_state.diagnosis_running):
            st.session_state.diagnosis_running = True
            st.session_state["diagnosis_meta"] = meta

            with st.status("Running diagnosis...", expanded=True) as status:
                try:
                    final_report = None

                    for event in stream_diagnosis_agent(paper_text, effective_venue):
                        if isinstance(event, dict) and "error" in event:
                            st.warning(f"Agent stopped: {event['error']}")
                            break

                        if not isinstance(event, dict):
                            continue

                        msgs = event.get("messages", [])
                        if msgs:
                            last = msgs[-1]
                            role = getattr(last, "type", "")
                            content = getattr(last, "content", "")
                            if role == "ai" and content and isinstance(content, str) and content.strip():
                                st.write(content)

                        report = event.get("report")
                        if report:
                            final_report = report

                    status.update(label="Done", state="complete")
                    st.session_state.diagnosis_report = final_report
                except Exception as e:
                    status.update(label="Error", state="error")
                    st.error(str(e))

            st.session_state.diagnosis_running = False
            st.rerun()

    else:
        st.info("Upload a PDF above to start the diagnosis.")
