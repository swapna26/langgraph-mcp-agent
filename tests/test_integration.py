"""
Integration Tests for SQL RAG Server
======================================
Tests the full Text-to-SQL pipeline with REAL PostgreSQL and REAL Ollama LLM.
These tests require both services to be running.

Run locally:
  - Start PostgreSQL: docker start agentic_rag_postgres
  - Start Ollama: ollama serve (with mistral or llama3.2:1b)
  - Run: OLLAMA_MODEL=llama3.2:1b uv run pytest tests/test_integration.py -v

In CI: These run automatically with Docker services (see ci.yml)
"""

import os
import pytest
import httpx
import psycopg2

# Skip all tests if services are not available
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://raguser:ragpassword@localhost:5432/agentic_rag"
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def is_postgres_available():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.close()
        return True
    except Exception:
        return False


def is_ollama_available():
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        return r.status_code == 200
    except Exception:
        return False


skip_no_postgres = pytest.mark.skipif(
    not is_postgres_available(), reason="PostgreSQL not available"
)
skip_no_ollama = pytest.mark.skipif(
    not is_ollama_available(), reason="Ollama not available"
)


@skip_no_postgres
class TestDatabaseIntegration:
    """Tests that directly hit the real PostgreSQL database."""

    def test_list_tables_real(self):
        from servers.sql_rag_server import list_tables

        result = list_tables()
        assert "error" not in result
        assert result["count"] > 0
        table_names = [t["table_name"] for t in result["tables"]]
        assert "conversations" in table_names

    def test_get_table_schema_real(self):
        from servers.sql_rag_server import get_table_schema

        result = get_table_schema("conversations")
        assert "error" not in result
        assert result["table_name"] == "conversations"
        col_names = [c["name"] for c in result["columns"]]
        assert "id" in col_names
        assert "title" in col_names

    def test_get_table_schema_invalid_table(self):
        from servers.sql_rag_server import get_table_schema

        result = get_table_schema("nonexistent_table_xyz")
        assert "error" in result


@skip_no_postgres
@skip_no_ollama
class TestTextToSQLIntegration:
    """Tests the full Text-to-SQL pipeline with real LLM and real DB."""

    def test_simple_count_query(self):
        """Ask a simple count question — LLM should generate valid SQL."""
        from servers.sql_rag_server import text_to_sql

        result = text_to_sql("How many conversations are there in total?")

        assert "error" not in result, f"Got error: {result.get('error')}"
        assert "generated_sql" in result
        assert "SELECT" in result["generated_sql"].upper()
        assert result["row_count"] >= 1
        assert len(result["data"]) >= 1

    def test_select_with_limit(self):
        """Ask for specific rows — LLM should generate a SELECT with LIMIT."""
        from servers.sql_rag_server import text_to_sql

        result = text_to_sql("Show me the first 3 conversations with their titles")

        assert "error" not in result, f"Got error: {result.get('error')}"
        assert "generated_sql" in result
        assert len(result["data"]) <= 5  # Should be a small result set
