"""
MCP Server: SQL RAG (Text-to-SQL with Local LLM)
==================================================
Converts natural language questions to SQL using a locally-hosted
Mistral-7B model (served via Ollama), executes them on PostgreSQL,
and returns results.

This is the TEXT-TO-SQL pipeline:
  User question → Schema linking → Prompt building → LLM generates SQL
  → SQL validation → Execute on DB → Return results

The LLM runs ON-PREMISE (Ollama on localhost:11434), ensuring
zero external API calls and full data sovereignty.

Run standalone:  uv run servers/sql_rag_server.py
"""

import os
import json
import re
import httpx
import psycopg2
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("SQL RAG Server")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://raguser:ragpassword@localhost:5432/agentic_rag",
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

MAX_RETRIES = 2
MAX_ROWS = 100


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_connection():
    """Get a read-only PostgreSQL connection."""
    conn = psycopg2.connect(DATABASE_URL)
    conn.set_session(readonly=True)
    return conn


def _fetch_all_tables() -> list[dict]:
    """Return all public tables with their row counts."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]

        result = []
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            count = cur.fetchone()[0]
            result.append({"table_name": table, "row_count": count})
        return result
    finally:
        conn.close()


def _fetch_table_schema(table_name: str) -> dict:
    """Return column definitions and sample rows for a table."""
    conn = _get_connection()
    try:
        cur = conn.cursor()

        # Column info
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = %s AND table_schema = 'public'
            ORDER BY ordinal_position
        """, (table_name,))
        columns = [
            {"name": r[0], "type": r[1], "nullable": r[2]}
            for r in cur.fetchall()
        ]

        if not columns:
            return {"error": f"Table '{table_name}' not found."}

        # Sample rows (3)
        col_names = [c["name"] for c in columns]
        # Exclude large vector/embedding columns from sample
        safe_cols = [c for c in col_names if "embedding" not in c.lower() and "vector" not in c.lower()]
        if not safe_cols:
            safe_cols = col_names[:5]

        select_cols = ", ".join(safe_cols)
        cur.execute(f"SELECT {select_cols} FROM {table_name} LIMIT 3")  # noqa: S608
        sample_rows = []
        for row in cur.fetchall():
            sample_rows.append(dict(zip(safe_cols, [str(v) for v in row])))

        return {"table_name": table_name, "columns": columns, "sample_rows": sample_rows}
    finally:
        conn.close()


def _execute_sql(query: str) -> dict:
    """Execute a SELECT query and return results."""
    conn = _get_connection()
    try:
        cur = conn.cursor()
        cur.execute(query)
        col_names = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchmany(MAX_ROWS)
        total = cur.rowcount
        data = [dict(zip(col_names, [str(v) for v in row])) for row in rows]
        return {"data": data, "row_count": total, "columns": col_names}
    finally:
        conn.close()


def _validate_sql(sql: str) -> str | None:
    """Validate that the SQL is a safe SELECT query.
    Returns None if valid, or an error message if invalid.
    """
    cleaned = sql.strip().rstrip(";").strip()
    upper = cleaned.upper()

    if not upper.startswith("SELECT"):
        return "Only SELECT queries are allowed."

    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
                 "CREATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"]
    for keyword in forbidden:
        # Match as whole word to avoid false positives
        if re.search(rf'\b{keyword}\b', upper):
            return f"Forbidden keyword '{keyword}' detected in query."

    return None


# ---------------------------------------------------------------------------
# LLM helper — calls local Ollama (Mistral)
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, system_prompt: str = "") -> str:
    """Call the local Ollama LLM and return the response text."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = httpx.post(
        f"{OLLAMA_BASE_URL}/v1/chat/completions",
        json={"model": OLLAMA_MODEL, "messages": messages, "temperature": 0.1},
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _extract_sql(llm_output: str) -> str:
    """Extract SQL from the LLM output, handling markdown code blocks."""
    # Try to find SQL in ```sql ... ``` blocks
    match = re.search(r'```(?:sql)?\s*\n?(.*?)\n?```', llm_output, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Try to find a line starting with SELECT
    for line in llm_output.split("\n"):
        if line.strip().upper().startswith("SELECT"):
            # Grab from this line to the end (or until a non-SQL line)
            idx = llm_output.index(line)
            candidate = llm_output[idx:].strip()
            # Remove trailing explanation text after semicolon
            if ";" in candidate:
                candidate = candidate[:candidate.index(";") + 1]
            return candidate.strip().rstrip(";")

    # Fallback: return the whole thing stripped
    return llm_output.strip().rstrip(";")


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_tables() -> dict:
    """
    List all tables in the database with their row counts.
    Use this first to understand what data is available.
    """
    try:
        tables = _fetch_all_tables()
        return {"tables": tables, "count": len(tables)}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_table_schema(table_name: str) -> dict:
    """
    Get the schema (columns, types) and 3 sample rows for a specific table.
    Use this to understand a table's structure before asking questions about it.
    """
    try:
        return _fetch_table_schema(table_name)
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def text_to_sql(question: str) -> dict:
    """
    Convert a natural language question to SQL, execute it, and return results.

    This is the main Text-to-SQL tool. It:
    1. Discovers available tables and their schemas
    2. Sends the schema + question to a local LLM (Mistral via Ollama)
    3. The LLM generates a SQL query
    4. Validates the SQL (only SELECT allowed)
    5. Executes on PostgreSQL
    6. If there's a SQL error, retries with error feedback (max 2 retries)
    7. Returns the generated SQL, query results, and row count

    Example questions:
    - "How many conversations happened in the last 7 days?"
    - "What is the most active user by message count?"
    - "Show me the average messages per conversation"
    """
    try:
        # Step 1: Get all table schemas for context
        tables = _fetch_all_tables()
        table_names = [t["table_name"] for t in tables]

        schemas = []
        for t in table_names:
            schema = _fetch_table_schema(t)
            if "error" not in schema:
                cols = ", ".join(
                    f'{c["name"]} ({c["type"]})'
                    for c in schema["columns"]
                )
                schemas.append(f"Table: {t}\n  Columns: {cols}")

        schema_text = "\n\n".join(schemas)

        # Step 2: Build prompt for the LLM
        system_prompt = """You are a SQL expert. Given a database schema and a natural language question, generate a PostgreSQL SELECT query that answers the question.

Rules:
- Only generate SELECT queries. Never use DROP, DELETE, UPDATE, INSERT, ALTER, or any other DDL/DML.
- Return ONLY the SQL query, no explanations.
- Use proper PostgreSQL syntax.
- If you need to cast types, use PostgreSQL casting syntax (e.g., ::date, ::text).
- Limit results to 100 rows maximum using LIMIT 100.
- Wrap the SQL in a ```sql code block."""

        user_prompt = f"""Database Schema:
{schema_text}

Question: {question}

Generate the SQL query:"""

        # Step 3: Call local LLM to generate SQL
        llm_output = _call_ollama(user_prompt, system_prompt)
        generated_sql = _extract_sql(llm_output)

        # Step 4: Validate
        validation_error = _validate_sql(generated_sql)
        if validation_error:
            return {
                "error": validation_error,
                "generated_sql": generated_sql,
                "data": [],
            }

        # Step 5: Execute with retry logic
        last_error = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                result = _execute_sql(generated_sql)
                return {
                    "generated_sql": generated_sql,
                    "data": result["data"],
                    "row_count": result["row_count"],
                    "columns": result["columns"],
                    "attempts": attempt + 1,
                }
            except Exception as sql_err:
                last_error = str(sql_err)
                if attempt < MAX_RETRIES:
                    # Send error back to LLM for self-correction
                    retry_prompt = f"""The SQL query you generated caused an error.

Original question: {question}
Generated SQL: {generated_sql}
Error: {last_error}

Please fix the SQL query. Return ONLY the corrected SQL in a ```sql code block."""

                    llm_output = _call_ollama(retry_prompt, system_prompt)
                    generated_sql = _extract_sql(llm_output)

                    validation_error = _validate_sql(generated_sql)
                    if validation_error:
                        return {
                            "error": validation_error,
                            "generated_sql": generated_sql,
                            "data": [],
                            "attempts": attempt + 2,
                        }

        return {
            "error": f"SQL execution failed after {MAX_RETRIES + 1} attempts: {last_error}",
            "generated_sql": generated_sql,
            "data": [],
            "attempts": MAX_RETRIES + 1,
        }

    except httpx.ConnectError:
        return {
            "error": "Cannot connect to Ollama. Is it running? (docker start agentic_rag_ollama)",
            "data": [],
        }
    except Exception as e:
        return {"error": str(e), "data": []}


@mcp.resource("info://sql-rag")
def server_info() -> str:
    """Information about the SQL RAG MCP server."""
    return """
    SQL RAG MCP Server — Text-to-SQL with On-Premise LLM
    -----------------------------------------------------
    Tools   : list_tables(), get_table_schema(table), text_to_sql(question)
    LLM     : Mistral-7B via Ollama (localhost:11434) — fully on-premise
    Backend : PostgreSQL 16
    Database: agentic_rag
    Safety  : READ-ONLY queries only, SELECT statements only,
              forbidden keyword detection, max 100 rows
    Retry   : Auto-corrects SQL errors up to 2 times using LLM feedback
    """


if __name__ == "__main__":
    mcp.run(transport="stdio")
