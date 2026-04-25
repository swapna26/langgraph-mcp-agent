"""
Unit Tests for SQL RAG Server
==============================
Tests SQL validation, SQL extraction from LLM output, and the
text_to_sql pipeline using mocked database and LLM responses.

Run: uv run pytest tests/test_sql_rag_server.py -v
"""

from unittest.mock import patch
from servers.sql_rag_server import _validate_sql, _extract_sql, text_to_sql


# -------------------------------------------------------------------
# Test 1: SQL Validation (_validate_sql)
# -------------------------------------------------------------------


class TestSQLValidation:
    """Tests that only safe SELECT queries are allowed."""

    def test_valid_select(self):
        assert _validate_sql("SELECT * FROM conversations") is None

    def test_valid_select_with_join(self):
        assert (
            _validate_sql(
                "SELECT c.id FROM conversations c JOIN conversation_messages m ON c.id = m.conversation_id"
            )
            is None
        )

    def test_valid_select_with_aggregation(self):
        assert (
            _validate_sql("SELECT COUNT(*) FROM conversations GROUP BY user_id") is None
        )

    def test_reject_drop(self):
        assert _validate_sql("DROP TABLE conversations") is not None

    def test_reject_delete(self):
        assert _validate_sql("DELETE FROM conversations WHERE id = 1") is not None

    def test_reject_update(self):
        assert _validate_sql("UPDATE conversations SET title = 'hacked'") is not None

    def test_reject_insert(self):
        assert _validate_sql("INSERT INTO conversations VALUES ('test')") is not None

    def test_reject_alter(self):
        assert (
            _validate_sql("ALTER TABLE conversations ADD COLUMN hack text") is not None
        )

    def test_reject_truncate(self):
        assert _validate_sql("TRUNCATE conversations") is not None

    def test_reject_drop_inside_select(self):
        """DROP hidden inside a SELECT should still be caught."""
        result = _validate_sql("SELECT * FROM conversations; DROP TABLE conversations")
        assert result is not None

    def test_reject_non_select(self):
        result = _validate_sql("EXPLAIN SELECT * FROM conversations")
        assert result is not None
        assert "Only SELECT" in result


# -------------------------------------------------------------------
# Test 2: SQL Extraction from LLM Output (_extract_sql)
# -------------------------------------------------------------------


class TestSQLExtraction:
    """Tests that SQL is correctly extracted from various LLM output formats."""

    def test_extract_from_code_block(self):
        llm_output = """Here is the SQL query:
```sql
SELECT COUNT(*) FROM conversations
```
This will count all rows."""
        result = _extract_sql(llm_output)
        assert result == "SELECT COUNT(*) FROM conversations"

    def test_extract_from_plain_code_block(self):
        llm_output = """```
SELECT id, title FROM conversations LIMIT 5
```"""
        result = _extract_sql(llm_output)
        assert result == "SELECT id, title FROM conversations LIMIT 5"

    def test_extract_plain_sql(self):
        llm_output = "SELECT COUNT(*) FROM conversations;"
        result = _extract_sql(llm_output)
        assert result.startswith("SELECT")

    def test_extract_with_explanation_before(self):
        llm_output = """To answer your question, I'll count the conversations:
SELECT COUNT(*) FROM conversations;
This returns the total count."""
        result = _extract_sql(llm_output)
        assert "SELECT" in result
        assert "COUNT" in result

    def test_extracts_from_code_block_with_semicolon(self):
        llm_output = "```sql\nSELECT * FROM conversations;\n```"
        result = _extract_sql(llm_output)
        assert "SELECT" in result
        assert "conversations" in result


# -------------------------------------------------------------------
# Test 3: text_to_sql with Mocked LLM and Database
# -------------------------------------------------------------------


class TestTextToSQL:
    """Tests the full text_to_sql pipeline with mocked dependencies."""

    @patch("servers.sql_rag_server._execute_sql")
    @patch("servers.sql_rag_server._call_ollama")
    @patch("servers.sql_rag_server._fetch_table_schema")
    @patch("servers.sql_rag_server._fetch_all_tables")
    def test_successful_query(
        self, mock_tables, mock_schema, mock_ollama, mock_execute
    ):
        # Mock database tables
        mock_tables.return_value = [{"table_name": "conversations", "row_count": 272}]

        # Mock schema
        mock_schema.return_value = {
            "table_name": "conversations",
            "columns": [
                {"name": "id", "type": "character varying"},
                {"name": "title", "type": "character varying"},
                {"name": "created_at", "type": "timestamp with time zone"},
            ],
            "sample_rows": [],
        }

        # Mock LLM response
        mock_ollama.return_value = "```sql\nSELECT COUNT(*) FROM conversations\n```"

        # Mock SQL execution
        mock_execute.return_value = {
            "data": [{"count": "272"}],
            "row_count": 1,
            "columns": ["count"],
        }

        result = text_to_sql("How many conversations are there?")

        assert "error" not in result
        assert result["generated_sql"] == "SELECT COUNT(*) FROM conversations"
        assert result["data"] == [{"count": "272"}]
        assert result["row_count"] == 1
        assert result["attempts"] == 1

    @patch("servers.sql_rag_server._execute_sql")
    @patch("servers.sql_rag_server._call_ollama")
    @patch("servers.sql_rag_server._fetch_table_schema")
    @patch("servers.sql_rag_server._fetch_all_tables")
    def test_retry_on_sql_error(
        self, mock_tables, mock_schema, mock_ollama, mock_execute
    ):
        """Tests that text_to_sql retries when SQL execution fails."""
        mock_tables.return_value = [{"table_name": "conversations", "row_count": 272}]
        mock_schema.return_value = {
            "table_name": "conversations",
            "columns": [{"name": "id", "type": "character varying"}],
            "sample_rows": [],
        }

        # First LLM call returns bad SQL, second returns good SQL
        mock_ollama.side_effect = [
            "```sql\nSELECT * FROM wrong_table\n```",
            "```sql\nSELECT * FROM conversations\n```",
        ]

        # First execution fails, second succeeds
        mock_execute.side_effect = [
            Exception('relation "wrong_table" does not exist'),
            {"data": [{"id": "1"}], "row_count": 1, "columns": ["id"]},
        ]

        result = text_to_sql("Show me all conversations")

        assert "error" not in result
        assert result["attempts"] == 2
        assert result["generated_sql"] == "SELECT * FROM conversations"

    @patch("servers.sql_rag_server._call_ollama")
    @patch("servers.sql_rag_server._fetch_table_schema")
    @patch("servers.sql_rag_server._fetch_all_tables")
    def test_reject_dangerous_sql(self, mock_tables, mock_schema, mock_ollama):
        """Tests that dangerous SQL from LLM is rejected."""
        mock_tables.return_value = [{"table_name": "conversations", "row_count": 272}]
        mock_schema.return_value = {
            "table_name": "conversations",
            "columns": [{"name": "id", "type": "character varying"}],
            "sample_rows": [],
        }

        # LLM returns dangerous SQL
        mock_ollama.return_value = "```sql\nDROP TABLE conversations\n```"

        result = text_to_sql("Delete all conversations")

        assert "error" in result

    @patch("servers.sql_rag_server._fetch_all_tables")
    def test_ollama_connection_error(self, mock_tables):
        """Tests graceful handling when Ollama is not running."""
        import httpx

        mock_tables.side_effect = httpx.ConnectError("Connection refused")

        result = text_to_sql("How many conversations?")

        assert "error" in result


# -------------------------------------------------------------------
# Test 4: list_tables and get_table_schema with Mocked DB
# -------------------------------------------------------------------


class TestDatabaseTools:
    """Tests list_tables and get_table_schema with mocked database."""

    @patch("servers.sql_rag_server._fetch_all_tables")
    def test_list_tables(self, mock_fetch):
        from servers.sql_rag_server import list_tables

        mock_fetch.return_value = [
            {"table_name": "conversations", "row_count": 272},
            {"table_name": "conversation_messages", "row_count": 554},
        ]

        result = list_tables()
        assert result["count"] == 2
        assert len(result["tables"]) == 2

    @patch("servers.sql_rag_server._fetch_table_schema")
    def test_get_table_schema(self, mock_fetch):
        from servers.sql_rag_server import get_table_schema

        mock_fetch.return_value = {
            "table_name": "conversations",
            "columns": [
                {"name": "id", "type": "character varying", "nullable": "NO"},
            ],
            "sample_rows": [{"id": "test_1"}],
        }

        result = get_table_schema("conversations")
        assert result["table_name"] == "conversations"
        assert len(result["columns"]) == 1
