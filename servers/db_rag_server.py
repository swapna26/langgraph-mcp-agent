"""
MCP Server: DB RAG (Database Query)
====================================
Exposes SQL query capabilities over PostgreSQL as MCP tools.
Supports both direct SQL execution and natural-language-to-SQL
conversion using Gemini LLM.

Equivalent of: langgraph_core/tools/db_rag/databricks_sql.py from v3,
but now as an independent MCP server using PostgreSQL instead of Databricks.

Run standalone:  uv run servers/db_rag_server.py
"""

import os
import psycopg2
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("DB RAG Server")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://raguser:ragpassword@localhost:5432/agentic_rag"
)


def get_table_schemas() -> str:
    """Fetch schema info for all user tables in the database."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()
        # Get all user tables
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]

        schemas = []
        for table in tables:
            cur.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table}' AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
            columns = cur.fetchall()
            col_info = ", ".join(f"{col[0]} ({col[1]})" for col in columns)
            schemas.append(f"Table: {table}\n  Columns: {col_info}")

        return "\n\n".join(schemas)
    finally:
        conn.close()


def execute_sql(query: str) -> list[dict]:
    """Execute a read-only SQL query and return results."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        conn.set_session(readonly=True)
        cur = conn.cursor()
        cur.execute(query)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        return [dict(zip(columns, row)) for row in rows]
    finally:
        conn.close()


@mcp.tool()
def query_database(sql_query: str) -> dict:
    """
    Execute a READ-ONLY SQL query against the PostgreSQL database.
    Only SELECT statements are allowed. Use this when you already
    know the exact SQL query to run.

    Available tables:
    - conversations (id, user_id, title, created_at, updated_at, is_active)
    - conversation_messages (id, conversation_id, role, content, created_at, sources, processing_mode, model_used)
    - conversation_summaries (id, conversation_id, summary, message_count, created_at)
    - data_llamaindex_vectors (id, text, metadata_, node_id, embedding)
    - data_embeddings_gemini (id, text, metadata_, node_id, embedding)
    """
    # Safety: only allow SELECT queries
    stripped = sql_query.strip().upper()
    if not stripped.startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed for safety.", "data": []}

    try:
        results = execute_sql(sql_query)
        return {
            "data": results[:100],  # Limit to 100 rows
            "row_count": len(results),
            "query": sql_query,
        }
    except Exception as e:
        return {"error": str(e), "data": [], "query": sql_query}


@mcp.tool()
def describe_database() -> dict:
    """
    Get the complete database schema — all tables and their columns.
    Use this first to understand what data is available before writing queries.
    """
    try:
        schema = get_table_schemas()
        return {"schema": schema}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_table_sample(table_name: str, limit: int = 5) -> dict:
    """
    Get a sample of rows from a specific table.
    Useful for understanding the data format before writing queries.
    Only works with: conversations, conversation_messages, conversation_summaries
    """
    allowed_tables = {
        "conversations",
        "conversation_messages",
        "conversation_summaries",
    }
    if table_name not in allowed_tables:
        return {
            "error": f"Table '{table_name}' not allowed. Choose from: {allowed_tables}"
        }

    try:
        results = execute_sql(f"SELECT * FROM {table_name} LIMIT {min(limit, 10)}")
        return {"table": table_name, "data": results, "row_count": len(results)}
    except Exception as e:
        return {"error": str(e)}


@mcp.resource("info://db-rag")
def server_info() -> str:
    """Information about the DB RAG MCP server."""
    return """
    DB RAG MCP Server
    -----------------
    Tools   : query_database(sql), describe_database(), get_table_sample(table, limit)
    Backend : PostgreSQL 16
    Database: agentic_rag
    Tables  : conversations (272 rows), conversation_messages (554 rows),
              conversation_summaries, data_llamaindex_vectors (553 rows),
              data_embeddings_gemini (659 rows)
    Safety  : READ-ONLY queries only (no INSERT/UPDATE/DELETE)
    """


if __name__ == "__main__":
    mcp.run(transport="stdio")
