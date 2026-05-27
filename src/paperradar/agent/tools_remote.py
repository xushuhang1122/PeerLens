from __future__ import annotations

from typing import Any

from ..config import settings

_REMOTE_TOOL_NAMES = {
    "search_papers",
    "get_paper_reviews",
    "cluster_reviews",
    "analyze_temporal_distribution",
    "identify_research_gaps",
}

# Cache keyed by URL so switching servers doesn't serve stale tools.
_url_cache: dict[str, dict[str, Any]] = {}


def _active_url() -> str | None:
    """Return the MCP URL for this request.

    Priority: Streamlit session_state (per-user toggle) > env var (default).
    Falls back gracefully when called outside a Streamlit context.
    """
    try:
        import streamlit as st
        return st.session_state.get("remote_mcp_url", settings.remote_mcp.url)
    except Exception:
        return settings.remote_mcp.url


def is_remote_mode() -> bool:
    return bool(_active_url())


def _load_mcp_tools(url: str) -> dict[str, Any]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {"peerlens": {"url": url, "transport": "streamable_http"}}
    )
    tools = client.get_tools()
    return {t.name: t for t in tools if t.name in _REMOTE_TOOL_NAMES}


def get_remote_tool(name: str) -> Any:
    url = _active_url()
    if not url:
        raise RuntimeError("No remote MCP URL configured.")
    if url not in _url_cache:
        _url_cache[url] = _load_mcp_tools(url)
    tool = _url_cache[url].get(name)
    if tool is None:
        raise RuntimeError(
            f"Remote MCP tool '{name}' not found. "
            f"Available: {list(_url_cache[url].keys())}"
        )
    return tool


def resolve_tool(name: str, local_tool: Any) -> Any:
    if is_remote_mode():
        return get_remote_tool(name)
    return local_tool
