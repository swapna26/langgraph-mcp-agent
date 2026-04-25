"""
LangGraph Agent with MCP Tools — v3 Architecture
=================================================
Builds the full LangGraph state graph with:
- Supervisor node (multi-agent routing, v3 style)
- Agent nodes (each connected to MCP server tools)
- Tool execution nodes

Key difference from v3 original:
- v3: Tools imported from tool_registry.py (hardcoded)
- THIS: Tools discovered at runtime from MCP servers (decoupled!)
"""

import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from .mcp_tool_loader import MCPToolManager
from .supervisor import create_supervisor

load_dotenv()
logger = logging.getLogger(__name__)


def create_llm():
    """Initialize Google Gemini LLM."""
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
    )


def create_agent_node(llm, tools: list, agent_name: str):
    """
    Creates an agent node that uses LLM with MCP-discovered tools.

    Same pattern as v3's agent_creation.py but tools come from MCP
    servers instead of the hardcoded tool_registry.
    """
    llm_with_tools = llm.bind_tools(tools)

    async def agent_node(state: MessagesState) -> Command:
        try:
            response = await llm_with_tools.ainvoke(state["messages"])

            # If the LLM wants to call tools, execute them
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_results = []
                tool_map = {t.name: t for t in tools}

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]

                    if tool_name in tool_map:
                        logger.info(
                            f"[{agent_name}] Calling MCP tool: {tool_name}({tool_args})"
                        )
                        result = await tool_map[tool_name].ainvoke(tool_args)
                        tool_results.append(f"[{tool_name}]: {result}")

                # Get final response from LLM with tool results
                tool_context = "\n".join(tool_results)
                followup = await llm.ainvoke(
                    state["messages"]
                    + [
                        AIMessage(content=response.content or "Calling tools..."),
                        HumanMessage(
                            content=f"Tool results:\n{tool_context}\n\nPlease provide a clear, helpful answer based on these results."
                        ),
                    ]
                )
                final_content = followup.content
            else:
                final_content = response.content

            return Command(
                update={
                    "messages": [HumanMessage(content=final_content, name=agent_name)]
                },
                goto=END,
            )

        except Exception as e:
            logger.error(f"Agent '{agent_name}' failed: {e}", exc_info=True)
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=f"Error in {agent_name}: {str(e)}", name=agent_name
                        )
                    ]
                },
                goto=END,
            )

    return agent_node


async def build_graph(project_dir: str) -> tuple:
    """
    Build the full LangGraph agent system with MCP tools.

    Returns the compiled graph and the tool manager (for cleanup).
    """
    logger.info("Building LangGraph agent with MCP tools...")

    # 1. Initialize LLM
    llm = create_llm()
    logger.info("LLM initialized (Google Gemini)")

    # 2. Connect to MCP servers and discover tools
    tool_manager = MCPToolManager(project_dir)
    all_tools = await tool_manager.connect_all()

    for server, tools in all_tools.items():
        tool_names = [t.name for t in tools]
        logger.info(f"  {server}: {tool_names}")

    # 3. Define agent-to-server mapping
    #    (which agent uses which MCP server's tools)
    agent_server_map = {
        "web_rag_agent": "web_rag",
        "doc_rag_agent": "doc_rag",
        "db_rag_agent": "db_rag",
        "sql_agent": "sql_rag",
    }
    members = list(agent_server_map.keys())

    # 4. Build the state graph (v3 architecture)
    builder = StateGraph(MessagesState)

    # Add supervisor node
    builder.add_node("supervisor", create_supervisor(llm, members))
    builder.add_edge(START, "supervisor")

    # Add agent nodes — each gets tools from its MCP server
    for agent_name, server_name in agent_server_map.items():
        tools = tool_manager.get_tools_by_server(server_name)
        if tools:
            builder.add_node(agent_name, create_agent_node(llm, tools, agent_name))
            logger.info(f"Added agent node: {agent_name} ({len(tools)} MCP tools)")
        else:
            logger.warning(f"No tools for {agent_name}, skipping")

    # 5. Compile and return
    graph = builder.compile()
    logger.info("Graph compiled successfully!")

    return graph, tool_manager
