import json
import datetime

import streamlit as st

import sys
sys.path.insert(0, ".")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.paperradar.config import settings
from src.paperradar.agent.research_runner import stream_research_agent
from src.paperradar.memory.episodic import EpisodicMemory
from src.paperradar.schemas.survey import SurveyReport

# ------------------------------------------------------------------
# LLM for clarification (stateless, cached)
# ------------------------------------------------------------------

@st.cache_resource
def _get_clarify_llm():
    return ChatOpenAI(
        model=settings.llm.model,
        temperature=0.4,
        api_key=settings.llm.openai_api_key,
        **({"base_url": settings.llm.base_url} if settings.llm.base_url else {}),
    )


@st.cache_resource
def get_episodic():
    return EpisodicMemory()


_current_year = datetime.datetime.now().year
_default_years = [_current_year - 1, _current_year - 2, _current_year - 3]

_CLARIFY_SYSTEM = f"""\
You are helping a researcher scope a literature survey. Work through these phases in order.

PHASE 1 — FIRST RESPONSE (only for the very first user message):
Carefully read the user's message. Extract what is already specified:
  - Research topic/domain (usually present)
  - Time range / years (may or may not be mentioned)
  - Target conferences (may or may not be mentioned)
Ask EXACTLY ONE question that covers only the 1-2 most critical missing pieces.
Preferred format: combine time range and conference into a single question if both are missing.
Example: "Should I focus on papers from {_default_years[-1]}-{_default_years[0]}, and are NeurIPS/ICML/ICLR your target venues, or do you have other preferences?"
If the user already mentioned years AND conferences, skip this phase and go directly to PHASE 3.
Never ask about things the user already specified.

PHASE 2 — FOLLOW-UP (only if the research topic itself is still unclear after phase 1):
Ask at most ONE targeted question about the research scope or angle (not metadata).
Do NOT re-ask about years or conferences — use defaults: years={_default_years}, conferences=["NeurIPS","ICML","ICLR"].
Skip this phase entirely if the topic is already clear enough.

PHASE 3 — CONFIRMATION:
When you have enough context, output a plain-text summary starting with "CONFIRM:" followed by 2-3 sentences covering:
  - The research topic and specific angle
  - The time range (use defaults if not specified: {_default_years[-1]}-{_default_years[0]})
  - The target venues (use defaults if not specified: NeurIPS, ICML, ICLR)
Then ask the user to confirm or correct.

PHASE 4 — READY:
Only after the user explicitly confirms (says yes / ok / looks good / correct / etc.),
output a JSON block (and ONLY the JSON block):
{{"ready": true, "refined_query": "...", "focus": {{"conferences": [...], "years": [...], "decisions": ["oral","spotlight","poster","accepted"]}}}}

The refined_query must be a clear English research question suitable for a literature search.

RULES:
- Never output the JSON block before the user explicitly confirms.
- Never ask more than one question per turn.
- Silently apply defaults for any missing metadata (do not mention defaults unless in the CONFIRM summary).
- If the user says "start" / "go" / "begin" at any point, immediately output the ready JSON using your best understanding.
"""


def _run_clarify(messages: list[dict]) -> str:
    llm = _get_clarify_llm()
    lc_msgs: list = [SystemMessage(content=_CLARIFY_SYSTEM)]
    for m in messages:
        if m["role"] == "user":
            lc_msgs.append(HumanMessage(content=m["content"]))
        else:
            lc_msgs.append(AIMessage(content=m["content"]))
    resp = llm.invoke(lc_msgs)
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def _extract_json(text: str) -> dict | None:
    try:
        s = text.find("{")
        e = text.rfind("}") + 1
        if s >= 0 and e > s:
            return json.loads(text[s:e])
    except Exception:
        pass
    return None


def _force_extract_context() -> dict:
    """Extract refined_query + focus from conversation history via LLM when user force-starts."""
    existing = st.session_state.research_context
    if existing.get("refined_query"):
        return existing

    msgs = st.session_state.research_messages
    conversation = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in msgs
    )

    extract_prompt = (
        "Given this conversation, extract the researcher's intent.\n"
        "Output ONLY JSON with no other text:\n"
        '{"refined_query": "...", "focus": {"conferences": [...], "years": [...], "decisions": [...]}}\n\n'
        f"Defaults if not specified: years={_default_years}, "
        'conferences=["NeurIPS","ICML","ICLR"], decisions=["oral","spotlight","poster","accepted"].\n'
        "refined_query must be a clear English research question.\n\n"
        f"Conversation:\n{conversation}"
    )

    try:
        llm = _get_clarify_llm()
        resp = llm.invoke(extract_prompt)
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        parsed = _extract_json(raw)
        if parsed and parsed.get("refined_query"):
            return {"refined_query": parsed["refined_query"], "focus": parsed.get("focus", {})}
    except Exception:
        pass

    # Fallback: concatenate all user messages
    user_msgs = " ".join(m["content"] for m in msgs if m["role"] == "user")
    return {"refined_query": user_msgs or "research survey", "focus": {}}


# ------------------------------------------------------------------
# Survey rendering (reused across phases)
# ------------------------------------------------------------------

def _render_survey(survey: SurveyReport) -> None:
    st.divider()
    st.subheader(survey.title)
    st.markdown(f"_{survey.background}_")

    if survey.sections:
        for sec in survey.sections:
            st.markdown(f"### {sec.heading}")
            st.markdown(sec.content)

    if survey.key_papers:
        st.markdown("### Papers Referenced")
        for i, p in enumerate(survey.key_papers):
            with st.container(border=True):
                col_num, col_body = st.columns([1, 11])
                with col_num:
                    st.markdown(f"**[{i+1}]**")
                with col_body:
                    link = f"[{p.title}]({p.forum_url})" if p.forum_url else p.title
                    st.markdown(f"**{link}**")
                    st.caption(
                        f"{p.conference} {p.year} · `{p.decision}`"
                        + (f" · {p.primary_area}" if p.primary_area else "")
                    )
                    if p.abstract:
                        preview = p.abstract[:280].rstrip()
                        if len(p.abstract) > 280:
                            preview += "..."
                        st.markdown(f"> {preview}")

    if survey.open_questions:
        st.markdown("### Open Questions")
        for q in survey.open_questions:
            st.markdown(f"- {q}")

    if survey.submission_advice:
        st.info(f"**Submission advice:** {survey.submission_advice}")


# ------------------------------------------------------------------
# Session state init
# ------------------------------------------------------------------
if "research_phase" not in st.session_state:
    st.session_state.research_phase = "idle"
if "research_messages" not in st.session_state:
    st.session_state.research_messages = []
if "research_context" not in st.session_state:
    st.session_state.research_context = {}

# ------------------------------------------------------------------
# Page header
# ------------------------------------------------------------------
st.title("Research Agent")
st.caption(
    "Describe your research topic. The agent will ask a brief clarifying question, "
    "confirm its understanding, then search the paper database and write a structured mini survey."
)

phase = st.session_state.research_phase

# Render chat history
for msg in st.session_state.research_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# Force-start button (shown after first exchange)
# ------------------------------------------------------------------
if phase in ("clarifying", "confirming"):
    if st.button("Start Research now", type="secondary"):
        with st.spinner("Extracting research intent..."):
            ctx = _force_extract_context()
        st.session_state.research_context = ctx
        st.session_state.research_phase = "researching"
        st.rerun()

# ==================================================================
# PHASE: idle / clarifying / confirming
# ==================================================================
if phase in ("idle", "clarifying", "confirming"):
    query = st.chat_input("Describe your research topic...")

    if query:
        force = query.strip().lower() in ("start", "go", "begin")

        st.session_state.research_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        if force:
            with st.spinner("Extracting research intent..."):
                ctx = _force_extract_context()
            st.session_state.research_context = ctx
            st.session_state.research_phase = "researching"
            st.rerun()

        with st.spinner("Thinking..."):
            reply = _run_clarify(st.session_state.research_messages)

        parsed = _extract_json(reply)

        if parsed and parsed.get("ready"):
            refined = parsed.get("refined_query", query)
            focus = parsed.get("focus", {})
            st.session_state.research_context = {"refined_query": refined, "focus": focus}
            launch_msg = f"Starting research on: **{refined}**"
            st.session_state.research_messages.append({"role": "assistant", "content": launch_msg})
            with st.chat_message("assistant"):
                st.markdown(launch_msg)
            st.session_state.research_phase = "researching"
            st.rerun()

        elif reply.strip().upper().startswith("CONFIRM:"):
            st.session_state.research_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.research_phase = "confirming"

        else:
            st.session_state.research_messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state.research_phase = "clarifying"

# ==================================================================
# PHASE: researching
# ==================================================================
elif phase == "researching":
    ctx = st.session_state.research_context
    refined_query = ctx.get("refined_query", "")
    focus = ctx.get("focus", {})

    get_episodic().record_query(refined_query)

    with st.chat_message("assistant"):
        with st.status("Running research agent...", expanded=True) as status:
            try:
                survey = None
                for event in stream_research_agent(refined_query, focus):
                    if isinstance(event, dict) and "error" in event:
                        st.warning(f"Agent error: {event['error']}")
                        break

                    if not isinstance(event, dict):
                        continue

                    msgs = event.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        content = getattr(last, "content", "")
                        if content and isinstance(content, str) and content.strip():
                            st.caption(content)

                    if event.get("survey_report"):
                        survey = event["survey_report"]

                status.update(label="Done", state="complete")

            except Exception as e:
                status.update(label="Error", state="error")
                st.error(str(e))

        if survey:
            st.session_state["research_survey"] = survey
            _render_survey(survey)
            summary_md = f"## {survey.title}\n{survey.background}"
        else:
            st.session_state.pop("research_survey", None)
            st.warning("The agent did not produce a survey. Try a more specific topic.")
            summary_md = "Research complete (no survey generated)."

        st.session_state.research_messages.append(
            {"role": "assistant", "content": summary_md}
        )

    st.session_state.research_phase = "done"
    st.rerun()

# ==================================================================
# PHASE: done
# ==================================================================
elif phase == "done":
    col1, col2 = st.columns([2, 8])
    with col1:
        if st.button("New Research", type="primary"):
            for key in ("research_phase", "research_messages", "research_context", "research_survey"):
                st.session_state.pop(key, None)
            st.rerun()
    with col2:
        if st.button("Continue / Refine"):
            st.session_state.research_phase = "clarifying"
            st.rerun()

    survey = st.session_state.get("research_survey")
    if survey:
        _render_survey(survey)
