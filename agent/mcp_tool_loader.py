"""
MCP Tool Loader
===============
Connects to MCP servers, discovers their tools at runtime, and converts
them into LangChain-compatible tools for use with LangGraph agents.

This is the KEY piece that replaces the hardcoded tool_registry.py from v3.
Instead of importing tools directly, we DISCOVER them from MCP servers.
"""

import json
import asyncio
import logging
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import StructuredTool
from pydantic import create_model

logger = logging.getLogger(__name__)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: list[str]


# --- Define the 3 MCP servers ---
MCP_SERVERS = [
    MCPServerConfig(
        name="web_rag",
        command="uv",
        args=["run", "--directory", "", "servers/web_rag_server.py"],
    ),
    MCPServerConfig(
        name="doc_rag",
        command="uv",
        args=["run", "--directory", "", "servers/doc_rag_server.py"],
    ),
    MCPServerConfig(
        name="db_rag",
        command="uv",
        args=["run", "--directory", "", "servers/db_rag_server.py"],
    ),
    MCPServerConfig(
        name="sql_rag",
        command="uv",
        args=["run", "--directory", "", "servers/sql_rag_server.py"],
    ),
]


class MCPToolManager:
    """
    Manages connections to multiple MCP servers and provides
    LangChain-compatible tools discovered from them.

    This replaces the hardcoded tool_registry.py from v3:
    - v3: Tools imported at code level, tightly coupled
    - MCP: Tools discovered at runtime, completely decoupled
    """

    def __init__(self, project_dir: str):
        self.project_dir = project_dir
        self._sessions: dict[str, tuple] = {}  # name -> (session, read, write, cm1, cm2)
        self._tools: dict[str, list[StructuredTool]] = {}  # server_name -> tools

    async def connect_all(self) -> dict[str, list[StructuredTool]]:
        """Connect to all MCP servers and discover their tools."""
        all_tools = {}
        for server_config in MCP_SERVERS:
            try:
                tools = await self._connect_server(server_config)
                all_tools[server_config.name] = tools
                logger.info(f"Connected to '{server_config.name}': {len(tools)} tools discovered")
            except Exception as e:
                logger.error(f"Failed to connect to '{server_config.name}': {e}")
                all_tools[server_config.name] = []
        self._tools = all_tools
        return all_tools

    async def _connect_server(self, config: MCPServerConfig) -> list[StructuredTool]:
        """Connect to a single MCP server and convert its tools to LangChain tools."""
        # Set the project directory in the args
        args = [a if a else self.project_dir for a in config.args]

        server_params = StdioServerParameters(
            command=config.command,
            args=args,
        )

        # Create the connection
        read_stream, write_stream = None, None
        cm1 = stdio_client(server_params)
        read_stream, write_stream = await cm1.__aenter__()

        cm2 = ClientSession(read_stream, write_stream)
        session = await cm2.__aenter__()
        await session.initialize()

        # Store for cleanup
        self._sessions[config.name] = (session, read_stream, write_stream, cm1, cm2)

        # Discover tools
        tools_result = await session.list_tools()
        langchain_tools = []

        for tool in tools_result.tools:
            lc_tool = self._mcp_to_langchain(session, tool, config.name)
            langchain_tools.append(lc_tool)

        return langchain_tools

    def _mcp_to_langchain(self, session: ClientSession, mcp_tool, server_name: str) -> StructuredTool:
        """Convert an MCP tool to a LangChain StructuredTool."""
        tool_name = mcp_tool.name
        tool_desc = mcp_tool.description or ""

        # Build Pydantic model from MCP tool schema for proper argument parsing
        schema = mcp_tool.inputSchema or {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        field_definitions = {}
        for prop_name, prop_info in properties.items():
            prop_type = prop_info.get("type", "string")
            type_map = {"string": str, "integer": int, "number": float, "boolean": bool}
            python_type = type_map.get(prop_type, str)

            default = ... if prop_name in required else prop_info.get("default", None)
            field_definitions[prop_name] = (python_type, default)

        ArgsModel = create_model(f"{tool_name}_args", **field_definitions)

        # Create the async wrapper that calls MCP
        captured_session = session
        captured_name = tool_name

        async def call_mcp_tool(**kwargs):
            result = await captured_session.call_tool(captured_name, kwargs)
            if result.content:
                return result.content[0].text
            return ""

        # Sync wrapper for LangChain compatibility
        def call_mcp_tool_sync(**kwargs):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, call_mcp_tool(**kwargs))
                    return future.result()
            return asyncio.run(call_mcp_tool(**kwargs))

        return StructuredTool(
            name=tool_name,
            description=tool_desc,
            args_schema=ArgsModel,
            coroutine=call_mcp_tool,
            func=call_mcp_tool_sync,
        )

    def get_tools_by_server(self, server_name: str) -> list[StructuredTool]:
        """Get tools for a specific server/agent."""
        return self._tools.get(server_name, [])

    def get_all_tools(self) -> list[StructuredTool]:
        """Get all tools from all servers as a flat list."""
        all_tools = []
        for tools in self._tools.values():
            all_tools.extend(tools)
        return all_tools

    async def close_all(self):
        """Close all MCP server connections."""
        for name, (session, read, write, cm1, cm2) in self._sessions.items():
            try:
                await cm2.__aexit__(None, None, None)
                await cm1.__aexit__(None, None, None)
                logger.info(f"Disconnected from '{name}'")
            except Exception as e:
                logger.warning(f"Error closing '{name}': {e}")
        self._sessions.clear()
