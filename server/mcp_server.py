"""
PeerLens MCP Server — exposes read-only database tools over HTTP.

Run on the cloud machine that hosts the pre-crawled ChromaDB + BM25 data:

    python server/mcp_server.py

Required env vars on the server:
    EMBEDDING_API_KEY   — for query embedding
    EMBEDDING_BASE_URL  — optional, for custom embedding endpoint
    EMBEDDING_MODEL     — optional, defaults to text-embedding-3-large

The server itself does NOT need LLM_API_KEY; all LLM calls happen on the client.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP
from src.peerlens.agent.tools import (
    search_papers,
    get_paper_reviews,
)

mcp = FastMCP("PeerLens")


@mcp.tool(name="search_papers")
def _search_papers(
    query: str,
    top_k: int = 20,
    decision_filter: list[str] | None = None,
    conference_filter: list[str] | None = None,
    year_filter: list[int] | None = None,
) -> dict:
    """Search papers using hybrid BM25 + semantic retrieval."""
    result = search_papers.invoke(
        {
            "query": query,
            "top_k": top_k,
            "decision_filter": decision_filter,
            "conference_filter": conference_filter,
            "year_filter": year_filter,
        }
    )
    return result.model_dump()


@mcp.tool(name="get_paper_reviews")
def _get_paper_reviews(paper_ids: list[str]) -> dict:
    """Fetch stored review data for given paper IDs."""
    result = get_paper_reviews.invoke({"paper_ids": paper_ids})
    return result.model_dump()



if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8765"))
    host = os.getenv("MCP_HOST", "0.0.0.0")
    mcp.run(transport="streamable-http", host=host, port=port)
