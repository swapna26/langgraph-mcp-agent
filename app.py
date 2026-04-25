"""
FastAPI Application — LangGraph + MCP Agent
============================================
Exposes the multi-agent system via HTTP endpoints.

Equivalent of: main.py + api/v1/chat/routes.py from v3,
but simplified for learning.

Run with:  uv run uvicorn app:app --reload --port 8080
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent.graph import build_graph, MCPToolManager

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global references
graph = None
tool_manager: MCPToolManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start MCP connections on startup, close on shutdown."""
    global graph, tool_manager

    project_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Project directory: {project_dir}")

    graph, tool_manager = await build_graph(project_dir)
    logger.info("Agent graph ready!")

    yield

    # Cleanup on shutdown
    if tool_manager:
        await tool_manager.close_all()
        logger.info("MCP connections closed.")


app = FastAPI(
    title="LangGraph MCP Agent",
    description="Multi-agent system (v3 architecture) with MCP server tools",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., description="User's question")


class ChatResponse(BaseModel):
    answer: str
    agents_used: list[str] = []


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a query to the multi-agent system.
    The supervisor routes it to the appropriate agent(s).
    """
    logger.info(f"Query: {request.query}")

    result = await graph.ainvoke({"messages": [("user", request.query)]})

    # Extract the final response and which agents were used
    agents_used = []
    final_answer = ""

    for msg in result["messages"]:
        if hasattr(msg, "name") and msg.name:
            agents_used.append(msg.name)
            final_answer = msg.content

    if not final_answer:
        final_answer = result["messages"][-1].content

    return ChatResponse(answer=final_answer, agents_used=list(set(agents_used)))


@app.get("/health")
async def health():
    return {"status": "ok", "graph_ready": graph is not None}


@app.get("/agents")
async def list_agents():
    """List available agents and their MCP tools."""
    if not tool_manager:
        return {"error": "Not initialized"}

    agents = {}
    for server_name, tools in tool_manager._tools.items():
        agents[server_name] = [
            {"name": t.name, "description": t.description} for t in tools
        ]
    return {"agents": agents}
