import sys
sys.path.insert(0, ".")

import streamlit as st

from src.paperradar.utils.pdf_parser import extract_paper_text, extract_title_abstract
from src.paperradar.agent.diagnosis_runner import stream_diagnosis_agent
from src.paperradar.schemas.diagnosis import SimulatedReview

_KNOWN_VENUES = [
    "NeurIPS 2026", "ICML 2026", "ICLR 2026",
    "NeurIPS 2025", "ICML 2025", "ICLR 2025",
    "AISTATS 2026", "UAI 2026", "AAAI 2026",
    "CVPR 2026", "ECCV 2026",
    "ACL 2026", "EMNLP 2026",
    "JMLR",
]

_PRIORITY_COLOR = {"critical": "red", "important": "orange", "minor": "blue"}


def _render_score_chip(label: str, value: int, max_val: int) -> None:
    st.metric(label=label, value=f"{value}/{max_val}")


def _render_simulated_review(rev: SimulatedReview) -> None:
    venue_label = f" — {rev.venue}" if rev.venue else ""
    st.subheader(f"Simulated Peer Review{venue_label}")
    with st.container(border=True):
        cols = st.columns(5)
        with cols[0]:
            _render_score_chip(f"Overall ({rev.overall_scale})", rev.overall_score, int(rev.overall_scale.split("-")[-1]))
        with cols[1]:
            _render_score_chip(f"Soundness ({rev.soundness_scale})", rev.soundness, int(rev.soundness_scale.split("-")[-1]))
        with cols[2]:
            _render_score_chip(f"Presentation ({rev.presentation_scale})", rev.presentation, int(rev.presentation_scale.split("-")[-1]))
        with cols[3]:
            _render_score_chip(f"Contribution ({rev.contribution_scale})", rev.contribution, int(rev.contribution_scale.split("-")[-1]))
        with cols[4]:
            _render_score_chip(f"Confidence ({rev.confidence_scale})", rev.confidence, int(rev.confidence_scale.split("-")[-1]))

        if rev.score_interpretation:
            st.caption(f"Score interpretation: {rev.score_interpretation}")

        if rev.summary:
            st.markdown(f"**Verdict:** {rev.summary}")

        tab_s, tab_w, tab_q = st.tabs(["Strengths", "Weaknesses", "Questions to Authors"])
        with tab_s:
            for s in rev.strengths:
                st.markdown(f"- {s}")
        with tab_w:
            for w in rev.weaknesses:
                st.markdown(f"- {w}")
        with tab_q:
            for q in rev.questions:
                st.markdown(f"- {q}")


def _render_report(report) -> None:
    st.subheader(f"Diagnosis Report — {report.detected_domain}")

    if report.detected_keywords:
        st.markdown("**Keywords detected:** " + " | ".join(f"`{k}`" for k in report.detected_keywords))

    st.markdown(f"**Overall assessment:** {report.overall_assessment}")

    if report.simulated_review:
        st.divider()
        _render_simulated_review(report.simulated_review)

    st.divider()
    col_acc, col_rej = st.columns(2)
    with col_acc:
        st.markdown("**Acceptance patterns in this area**")
        for p in report.acceptance_patterns:
            st.markdown(f"- {p}")
    with col_rej:
        st.markdown("**Common rejection patterns**")
        for p in report.rejection_patterns:
            st.markdown(f"- {p}")

    if report.key_reviewer_concerns:
        st.markdown("**Key reviewer concerns in this domain**")
        for c in report.key_reviewer_concerns:
            st.markdown(f"- {c}")

    st.divider()
    st.subheader("Improvement Suggestions")

    priority_order = {"critical": 0, "important": 1, "minor": 2}
    sorted_suggestions = sorted(
        report.suggestions,
        key=lambda s: priority_order.get(s.priority.lower(), 3),
    )
    for s in sorted_suggestions:
        color = _PRIORITY_COLOR.get(s.priority.lower(), "gray")
        with st.container(border=True):
            st.markdown(f"**:{color}[{s.priority.upper()}]** — **{s.aspect.title()}**")
            if s.reviewer_comment:
                st.markdown(f"*Reviewer comment:* _{s.reviewer_comment}_")
            st.markdown(f"*Suggestion:* {s.suggestion}")

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
    "in the database, simulate a peer review, and give you specific improvement suggestions."
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
            paper_text = extract_paper_text(pdf_bytes, max_words=1500)
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
            help="Select or type a venue to get scoring and feedback tailored to that conference or journal's review criteria.",
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
