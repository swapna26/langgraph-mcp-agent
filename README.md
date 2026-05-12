# LangGraph MCP Agent

A multi-agent RAG system where AI agents discover and call tools at runtime via the Model Context Protocol (MCP). A supervisor routes user queries to specialized agents (web search, document search, database queries, text-to-SQL), each backed by independent MCP servers.

## Why This Project?

Traditional AI agents have tools hardcoded in their source code. Adding a new tool means changing agent code and redeploying. Tools can't be reused across agents easily.

This project solves that by using **MCP** (Model Context Protocol) — an open standard by Anthropic. Tools run as **independent servers**. Agents **discover tools at runtime** via the MCP protocol. Adding a new tool means deploying a new MCP server — zero changes to agent code.

```
Traditional approach:
  Agent code contains: search_web(), query_db(), read_docs()
  → Adding a tool = code change + redeploy

MCP approach:
  Agent code contains: connect_to_mcp_servers(), discover_tools()
  → Adding a tool = start a new MCP server (agent discovers it automatically)
```

---

## Architecture

```
User Question
    │
    ▼
FastAPI (app.py)
    │
    ▼
LangGraph State Machine (graph.py)
    │
    ▼
SUPERVISOR (Gemini LLM)
    │  Decides: which agent(s) should handle this?
    │  Can route to multiple agents simultaneously
    │
    ├──→ web_rag_agent ──→ MCP Server ──→ DuckDuckGo Search
    ├──→ doc_rag_agent ──→ MCP Server ──→ PostgreSQL + pgvector (semantic search)
    ├──→ db_rag_agent  ──→ MCP Server ──→ PostgreSQL (direct SQL)
    └──→ sql_agent     ──→ MCP Server ──→ Ollama Mistral (text-to-SQL)
```

### How a Query Flows

1. User sends `POST /chat {"query": "How many conversations are in the database?"}`
2. **Supervisor** (Gemini LLM) analyzes the query and decides: route to `sql_agent`
3. **sql_agent** connects to its MCP server, discovers available tools (`text_to_sql`, `list_tables`, etc.)
4. Agent's LLM decides which tool to call: `text_to_sql("How many conversations?")`
5. MCP server generates SQL via Ollama Mistral, validates it, executes against PostgreSQL
6. Result flows back: MCP server → agent → LangGraph → FastAPI → user
7. Response: `{"answer": "There are 272 conversations.", "agents_used": ["sql_agent"]}`

### Multi-Agent Routing

The supervisor can route to **multiple agents simultaneously**:

```
"Find procurement policies AND show conversation stats"
  → doc_rag_agent (searches procurement documents)
  → sql_agent (queries conversation database)
  → Both results combined in response
```

---

## What is MCP?

MCP (Model Context Protocol) is an open standard that defines how AI assistants communicate with external servers. Think of it like a USB port for AI — any MCP server can plug into any MCP client.

### How MCP Works Here

```
Agent (MCP Client)              MCP Server (independent process)
      │                               │
      │──── stdio pipes ─────────────│
      │                               │
      │  1. list_tools()              │
      │  ←── [web_search, ...]        │  Tools discovered at runtime
      │                               │
      │  2. call_tool("web_search",   │
      │     {"query": "AI news"})     │
      │  ←── search results           │  Tool executed on server
      │                               │
```

The agent launches each MCP server as a subprocess and communicates over stdin/stdout pipes. No HTTP, no ports — just process-level communication.

### Why This Matters

| Without MCP | With MCP |
|-------------|----------|
| Tools hardcoded in agent | Tools discovered at runtime |
| Adding tool = code change | Adding tool = new MCP server |
| Tools coupled to one agent | Tools reusable across agents |
| Monolithic deployment | Independent server deployment |

---

## What is LangGraph?

LangGraph is a state machine framework for building agentic workflows. It manages how data flows between nodes (agents) and handles routing, state, and execution.

### How LangGraph Is Used Here

```python
# Define the graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("supervisor", supervisor_node)       # Routes queries
builder.add_node("web_rag_agent", agent_node)          # Web search
builder.add_node("doc_rag_agent", agent_node)          # Document search
builder.add_node("db_rag_agent", agent_node)           # Database queries
builder.add_node("sql_agent", agent_node)              # Text-to-SQL

# Define flow: START → supervisor → agents → END
builder.add_edge(START, "supervisor")

# Compile into executable graph
graph = builder.compile()
```

The supervisor uses `Command(goto=["agent1", "agent2"])` to dynamically route to one or more agents. Each agent executes independently and returns results to END.

### Key LangGraph Concepts

- **StateGraph** — The computation graph definition
- **MessagesState** — Shared state (conversation messages) that flows between nodes
- **Command** — Dynamic routing instruction (which node(s) to go to next)
- **START / END** — Entry and exit points of the graph

---

## The Four MCP Servers

### 1. Web RAG Server (`servers/web_rag_server.py`)

Searches the web using DuckDuckGo.

| Tool | Arguments | What It Does |
|------|-----------|-------------|
| `web_search` | `query: str, top_k: int = 5` | Returns top search results with title, link, and content |

**Use case:** Current events, external knowledge, competitor research, anything not in your internal documents.

### 2. Doc RAG Server (`servers/doc_rag_server.py`)

Semantic search over indexed documents using PostgreSQL + pgvector.

| Tool | Arguments | What It Does |
|------|-----------|-------------|
| `search_documents` | `query: str, top_k: int = 5` | Converts query to embedding (Ollama), performs cosine similarity search |
| `list_indexed_documents` | — | Lists all indexed documents with chunk counts |

**How it works:**
1. User query → Ollama `nomic-embed-text:v1.5` → 768-dimensional vector
2. Vector compared against pre-indexed document chunks in pgvector
3. Top K most similar chunks returned with similarity scores

**Documents indexed:** HR Bylaws, Procurement Manuals, Abu Dhabi Procurement Standards, Information Security policies.

**Use case:** Internal document search, policy questions, procurement rules.

### 3. DB RAG Server (`servers/db_rag_server.py`)

Direct read-only SQL queries against PostgreSQL.

| Tool | Arguments | What It Does |
|------|-----------|-------------|
| `query_database` | `sql_query: str` | Executes read-only SELECT queries (max 100 rows) |
| `describe_database` | — | Returns full schema (all tables, columns, types) |
| `get_table_sample` | `table_name: str, limit: int = 5` | Sample rows from allowed tables |

**Safety:** Rejects any non-SELECT queries (no DROP, DELETE, UPDATE, INSERT).

**Use case:** When the LLM already knows the SQL or needs schema exploration.

### 4. SQL RAG Server (`servers/sql_rag_server.py`)

The most complex server — converts natural language to SQL using an on-premise LLM.

| Tool | Arguments | What It Does |
|------|-----------|-------------|
| `text_to_sql` | `question: str` | Full pipeline: schema linking → prompt → LLM → SQL → validation → execution |
| `list_tables` | — | Returns all tables with row counts |
| `get_table_schema` | `table_name: str` | Returns columns, types, and sample rows |

**The text-to-sql pipeline:**

```
User question: "How many active conversations?"
    │
    ▼
1. Schema Linking
   Fetch all tables + columns + sample rows from PostgreSQL
    │
    ▼
2. Prompt Building
   System: "You are a SQL expert..."
   User: "Schema: [tables...]\n\nQuestion: How many active conversations?"
    │
    ▼
3. LLM Call
   Ollama Mistral-7B generates SQL
    │
    ▼
4. SQL Extraction
   Parse SQL from markdown code blocks or plain text
    │
    ▼
5. Validation
   Only SELECT allowed. Rejects DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE
    │
    ▼
6. Execution
   Run against PostgreSQL (read-only, max 100 rows)
    │
    ▼
7. Self-Correction (if SQL fails)
   Send error back to LLM: "Your SQL had this error: ... Fix it."
   Retry up to 2 times
    │
    ▼
Return results + generated SQL + attempt count
```

**Use case:** Non-technical users asking data questions in plain English.

---

## The Supervisor (Multi-Agent Router)

The supervisor is a Gemini LLM that decides which agent(s) should handle each query.

**How routing works:**

| User Query | Routed To | Why |
|-----------|-----------|-----|
| "What are the procurement rules?" | `doc_rag_agent` | Internal document question |
| "Search the web for AI news" | `web_rag_agent` | External web search |
| "How many conversations?" | `sql_agent` | Database question (natural language) |
| "Show me the schema" | `db_rag_agent` | Direct database exploration |
| "Find policies AND show stats" | `doc_rag_agent` + `sql_agent` | Multi-agent dispatch |

The supervisor outputs structured data:
```python
{
    "next_agents": ["doc_rag_agent", "sql_agent"],
    "modified_queries": {
        "doc_rag_agent": "procurement approval policies",
        "sql_agent": "count of conversations by date"
    },
    "reasoning": "Query needs both document search and database stats"
}
```

**Version history:**
- v1: Routes to single agent
- v2: Routes to single agent + modifies query for better results
- v3 (current): Routes to **multiple agents** + modifies each query independently

---

## MCP Tool Discovery (The Key Innovation)

The `agent/mcp_tool_loader.py` handles runtime tool discovery:

```python
# 1. Define MCP servers (just name + command to launch them)
MCP_SERVERS = [
    MCPServerConfig("web_rag", "uv", ["run", "servers/web_rag_server.py"]),
    MCPServerConfig("doc_rag", "uv", ["run", "servers/doc_rag_server.py"]),
    # ...
]

# 2. Connect to all servers at startup
tool_manager = MCPToolManager()
tools_by_server = await tool_manager.connect_all()

# 3. Each server's tools are discovered automatically
# tools_by_server = {
#   "web_rag": [StructuredTool(name="web_search", ...)],
#   "doc_rag": [StructuredTool(name="search_documents", ...), ...],
#   ...
# }
```

**Conversion process:** MCP tools → LangChain StructuredTools

1. `session.list_tools()` — discover what's available
2. Build Pydantic model from each tool's `inputSchema`
3. Create async wrapper that calls `session.call_tool(name, args)`
4. Wrap in `StructuredTool` for LangChain compatibility

This means the agent code **never imports** any tool function. Tools are fetched, converted, and bound to the LLM at runtime.

---

## API Endpoints

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/chat` | POST | `{"query": "..."}` | `{"answer": "...", "agents_used": ["sql_agent"]}` |
| `/health` | GET | — | `{"status": "ok", "graph_ready": true}` |
| `/agents` | GET | — | `{"agents": {"web_rag": [{"name": "web_search", ...}], ...}}` |

```bash
# Ask a question
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How many conversations are in the database?"}'

# Check health
curl http://localhost:8080/health

# List available agents and their tools
curl http://localhost:8080/agents
```

---

## On-Premise LLM (Why Ollama + Mistral)

The text-to-SQL server uses Ollama running Mistral-7B locally instead of calling a cloud API.

**Why on-premise:**
- **Data privacy** — SQL queries and database schemas never leave your network
- **Cost** — No per-token API charges for high-volume SQL generation
- **Latency** — No network round-trip to cloud providers
- **Compliance** — Required for enterprise/government environments

**Ollama** is the runtime that loads and serves the model. **Mistral-7B** is the model that generates SQL from natural language.

---

## Docker Setup

Three containers orchestrated with Docker Compose:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  app         │    │  postgres    │    │  ollama      │
│  (FastAPI)   │───→│  (pgvector)  │    │  (Mistral)   │
│  port 8080   │    │  port 5432   │    │  port 11434  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       │          pgdata volume         ollama_models volume
       │          (persists data)       (persists models)
```

```bash
# Start everything
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f app
```

**Key details:**
- Container networking: services reference each other by name (`postgres:5432`, `ollama:11434`)
- Health checks: app waits for postgres to be ready before starting
- Volumes: database and model files persist across container restarts
- GPU support: can be enabled in docker-compose.yml for production

---

## Testing

### Unit Tests (22 tests)

```bash
uv run pytest tests/test_sql_rag_server.py -v
```

All external services mocked (database, LLM). Tests cover:

| Category | Tests | What They Verify |
|----------|-------|-----------------|
| SQL Validation | 8 | Accepts valid SELECT, rejects DROP/DELETE/UPDATE/INSERT/ALTER/TRUNCATE |
| SQL Extraction | 5 | Parses SQL from markdown code blocks, plain text, handles semicolons |
| Text-to-SQL | 4 | Successful queries, retry on error, dangerous SQL rejection, Ollama connection failure |
| Database Tools | 2 | list_tables(), get_table_schema() |

### Integration Tests (auto-skip if services unavailable)

```bash
uv run pytest tests/test_integration.py -v
```

Requires real PostgreSQL and Ollama running. Tests the full pipeline end-to-end.

---

## Project Structure

```
langgraph-mcp-agent/
├── app.py                    # FastAPI endpoints (/chat, /health, /agents)
├── agent/
│   ├── supervisor.py         # Multi-agent router (Gemini LLM)
│   ├── graph.py              # LangGraph state machine builder
│   └── mcp_tool_loader.py    # MCP connection & runtime tool discovery
├── servers/
│   ├── web_rag_server.py     # MCP Server: DuckDuckGo web search
│   ├── doc_rag_server.py     # MCP Server: pgvector document search
│   ├── db_rag_server.py      # MCP Server: direct SQL queries
│   └── sql_rag_server.py     # MCP Server: text-to-SQL pipeline
├── tests/
│   ├── test_sql_rag_server.py  # 22 unit tests (mocked)
│   └── test_integration.py     # Integration tests (real services)
├── data/docs/                # Documents for RAG indexing
├── Dockerfile                # Container image for FastAPI app
├── docker-compose.yml        # Orchestrates app + postgres + ollama
├── ARCHITECTURE.md           # Detailed architecture documentation
├── pyproject.toml            # Dependencies
└── test_supervisor.py        # Quick supervisor routing test
```

## Quick Start

```bash
# Option 1: Docker (recommended)
docker compose up -d

# Option 2: Local
# Requires: PostgreSQL with pgvector, Ollama with mistral model
uv sync
uv run uvicorn app:app --port 8080
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | LangGraph | State machine for multi-agent routing |
| Supervisor LLM | Google Gemini 2.5 Flash | Query routing and agent coordination |
| SQL LLM | Ollama Mistral-7B | Text-to-SQL generation (on-premise) |
| Embeddings | Ollama nomic-embed-text | Document vectorization for semantic search |
| Tool Protocol | MCP (Model Context Protocol) | Runtime tool discovery and execution |
| Vector DB | PostgreSQL + pgvector | Semantic similarity search |
| Database | PostgreSQL | Conversation and document storage |
| Web Search | DuckDuckGo | External web search |
| API | FastAPI | HTTP interface |
| Containerization | Docker + Compose | Multi-container orchestration |

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection | `postgresql://user:pass@localhost:5432/agentic_rag` |
| `OLLAMA_BASE_URL` | Ollama server address | `http://localhost:11434` |
| `OLLAMA_MODEL` | LLM for text-to-SQL | `mistral` |
| `OLLAMA_EMBEDDING_MODEL` | Embedding model | `nomic-embed-text:v1.5` |
| `GEMINI_API_KEY` | Google Gemini API key | `AIza...` |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash` |

## Requirements

- Python 3.12+
- PostgreSQL with pgvector extension
- Ollama with Mistral model
- Google Gemini API key (for supervisor routing)
