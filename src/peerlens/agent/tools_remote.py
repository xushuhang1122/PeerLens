from __future__ import annotations

import json as _json
from typing import Any

from ..config import settings

_REMOTE_TOOL_NAMES = {
    "search_papers",
    "get_paper_reviews",
}

# Cache keyed by URL so switching servers doesn't serve stale tools.
_url_cache: dict[str, dict[str, Any]] = {}


def _active_url() -> str | None:
    """Return the MCP URL for this request.

    Priority: Streamlit session_state (per-user toggle) > env var (default).
    diag_force_local overrides to None for the duration of a single diagnosis run.
    Falls back gracefully when called outside a Streamlit context.
    """
    try:
        import streamlit as st
        if st.session_state.get("diag_force_local"):
            return None
        return st.session_state.get("remote_mcp_url", settings.remote_mcp.url)
    except Exception:
        return settings.remote_mcp.url


def is_remote_mode() -> bool:
    return bool(_active_url())


def _call_mcp_tool(url: str, tool_name: str, arguments: dict) -> Any:
    """
    Call an MCP tool via raw synchronous HTTP (no async libraries).

    Implements the MCP streamable-http protocol directly:
    initialize → notifications/initialized → tools/call → parse SSE response.
    """
    import httpx

    hdrs = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    with httpx.Client(timeout=30) as http:
        # 1. Initialize session
        r_init = http.post(url, json={
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "peerlens", "version": "1.0"},
            },
        }, headers=hdrs)
        r_init.raise_for_status()
        session_id = r_init.headers.get("mcp-session-id", "")
        hdrs_s = {**hdrs, "Mcp-Session-Id": session_id}

        # 2. Notify initialized
        http.post(url, json={"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}, headers=hdrs_s)

        # 3. Call tool
        r_call = http.post(url, json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }, headers=hdrs_s, timeout=60)
        r_call.raise_for_status()

        # 4. Parse SSE response: extract "data: {...}" lines
        for line in r_call.text.splitlines():
            if line.startswith("data:"):
                payload = _json.loads(line[5:].strip())
                content = payload.get("result", {}).get("content", [])
                if content and content[0].get("type") == "text":
                    return _json.loads(content[0]["text"])

        # 5. Close session (best-effort)
        try:
            http.delete(url, headers=hdrs_s)
        except Exception:
            pass

    return {}


class _SyncTool:
    """Sync wrapper around a single remote MCP tool."""

    def __init__(self, url: str, name: str) -> None:
        self._url = url
        self.name = name

    def invoke(self, input: Any) -> Any:
        return _call_mcp_tool(self._url, self.name, input)


def _load_mcp_tools(url: str) -> dict[str, Any]:
    """Return sync tool stubs for all known remote tools (no network call yet)."""
    return {name: _SyncTool(url, name) for name in _REMOTE_TOOL_NAMES}


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
