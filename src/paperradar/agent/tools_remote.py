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

_remote_cache: dict[str, Any] | None = None


def is_remote_mode() -> bool:
    return bool(settings.remote_mcp.url)


def _load_mcp_tools() -> dict[str, Any]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(
        {
            "peerlens": {
                "url": settings.remote_mcp.url,
                "transport": "streamable_http",
            }
        }
    )
    tools = client.get_tools()
    return {t.name: t for t in tools if t.name in _REMOTE_TOOL_NAMES}


def get_remote_tool(name: str) -> Any:
    global _remote_cache
    if _remote_cache is None:
        _remote_cache = _load_mcp_tools()
    tool = _remote_cache.get(name)
    if tool is None:
        raise RuntimeError(
            f"Remote MCP tool '{name}' not found. "
            f"Available: {list(_remote_cache.keys())}"
        )
    return tool


def resolve_tool(name: str, local_tool: Any) -> Any:
    """Return the remote MCP tool if in remote mode, otherwise the local tool."""
    if is_remote_mode():
        return get_remote_tool(name)
    return local_tool
