"""
MCP Server: Doc RAG (Document Semantic Search)
===============================================
Performs semantic vector search over indexed PDF documents stored
in PostgreSQL with pgvector. Uses Ollama for query embeddings and
pgvector for cosine similarity search.

Documents indexed: HR Bylaws, Procurement Manuals, Abu Dhabi Standards,
                   Information Security

Equivalent of: langgraph_core/tools/doc_rag/databricks_vector_search.py
from v3, but now as an independent MCP server.

Run standalone:  uv run servers/doc_rag_server.py
"""

import os
import json
import httpx
import psycopg2
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("Doc RAG Server")

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://raguser:ragpassword@localhost:5432/agentic_rag"
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:v1.5")

# --- Embedding table config ---
# Using the Ollama-indexed table (553 rows, 768-dim, has source_document metadata)
VECTOR_TABLE = "data_llamaindex_vectors"
EMBEDDING_DIM = 768


def get_embedding(text: str) -> list[float]:
    """Generate embedding using Ollama."""
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": OLLAMA_EMBEDDING_MODEL, "input": text},
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def vector_search(
    query_embedding: list[float], top_k: int = 5, filename_filter: str = None
) -> list[dict]:
    """Perform cosine similarity search on pgvector."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        if filename_filter:
            cur.execute(
                f"""
                SELECT text, metadata_, 1 - (embedding <=> %s::vector) as similarity
                FROM {VECTOR_TABLE}
                WHERE metadata_->>'source_document' = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_str, filename_filter, embedding_str, top_k),
            )
        else:
            cur.execute(
                f"""
                SELECT text, metadata_, 1 - (embedding <=> %s::vector) as similarity
                FROM {VECTOR_TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding_str, embedding_str, top_k),
            )

        rows = cur.fetchall()
        results = []
        for text, metadata, similarity in rows:
            meta = metadata if isinstance(metadata, dict) else json.loads(metadata)
            results.append(
                {
                    "text": text,
                    "source": meta.get(
                        "source_document", meta.get("filename", "unknown")
                    ),
                    "similarity": round(float(similarity), 4),
                }
            )
        return results
    finally:
        conn.close()


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> dict:
    """
    Search indexed PDF documents using semantic vector search.
    Use this for questions about HR policies, procurement manuals,
    Abu Dhabi procurement standards, or information security documents.

    Available documents:
    - HR Bylaws
    - Procurement Manual (Business Process)
    - Procurement Manual (Ariba Aligned)
    - Abu Dhabi Procurement Standards
    - Information Security
    """
    try:
        query_embedding = get_embedding(query)
        results = vector_search(query_embedding, top_k=top_k)
        return {
            "query": query,
            "results": results,
            "result_count": len(results),
        }
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}


@mcp.tool()
def list_indexed_documents() -> dict:
    """List all documents that have been indexed and are searchable."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT metadata_->>'source_document' as doc, count(*) as chunks
            FROM {VECTOR_TABLE}
            WHERE metadata_->>'source_document' IS NOT NULL
            GROUP BY metadata_->>'source_document'
            ORDER BY chunks DESC
            """
        )
        docs = [{"document": row[0], "chunks": row[1]} for row in cur.fetchall()]
        return {"documents": docs, "total_documents": len(docs)}
    finally:
        conn.close()


@mcp.resource("info://doc-rag")
def server_info() -> str:
    """Information about the Doc RAG MCP server."""
    return """
    Doc RAG MCP Server
    ------------------
    Tools   : search_documents(query, top_k), list_indexed_documents()
    Backend : PostgreSQL + pgvector (cosine similarity)
    Embeddings: Ollama nomic-embed-text:v1.5 (768-dim)
    Documents: HR Bylaws, Procurement Manuals, Abu Dhabi Standards, Info Security
    """


if __name__ == "__main__":
    mcp.run(transport="stdio")
