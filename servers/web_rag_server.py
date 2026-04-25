"""
MCP Server: Web RAG (DuckDuckGo Search)
========================================
Exposes web search as an MCP tool. Any MCP client (including our
LangGraph agent) can discover and call this tool at runtime.

Equivalent of: langgraph_core/tools/web_rag/duckduckgo.py from v3
but now as an independent, decoupled MCP server.

Run standalone:  uv run servers/web_rag_server.py
"""

from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS

mcp = FastMCP("Web RAG Server")


@mcp.tool()
def web_search(query: str, top_k: int = 5) -> dict:
    """
    Search the web using DuckDuckGo and return relevant results.
    Use this for current events, market trends, competitor analysis,
    or any question that needs up-to-date web information.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=top_k))

        formatted = []
        for r in results:
            formatted.append(
                {
                    "title": r.get("title", ""),
                    "link": r.get("href", ""),
                    "content": r.get("body", ""),
                }
            )

        return {"data": formatted, "question": query, "result_count": len(formatted)}

    except Exception as e:
        return {"data": [], "question": query, "error": str(e)}


@mcp.resource("info://web-rag")
def server_info() -> str:
    """Information about the Web RAG MCP server."""
    return """
    Web RAG MCP Server
    ------------------
    Tool    : web_search(query, top_k)
    Source  : DuckDuckGo API
    Purpose : Search the web for current information
    """


if __name__ == "__main__":
    mcp.run(transport="stdio")
