# End-to-End Architecture: Agentic RAG with On-Premise LLM

## Table of Contents
1. [System Overview](#system-overview)
2. [Application Flow](#application-flow)
3. [Text-to-SQL Pipeline](#text-to-sql-pipeline)
4. [MCP Protocol](#mcp-protocol)
5. [On-Premise LLM Deployment](#on-premise-llm-deployment)
6. [Docker & Containerization](#docker--containerization)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [Production Deployment](#production-deployment)
9. [Nginx, Reverse Proxy & SSL](#nginx-reverse-proxy--ssl)
10. [Environment Variables](#environment-variables)
11. [Interview Quick Reference](#interview-quick-reference)

---

## System Overview

```
User Question
     |
     v
+------------------+
|   FastAPI App     |   (app.py - Entry point)
|   POST /chat      |
+--------+---------+
         |
         v
+----------------------------------------------+
|          LangGraph State Machine              |
|                                               |
|  +-------------+                              |
|  | SUPERVISOR  |   (Gemini LLM decides which  |
|  |  (Router)   |    agent handles the question)|
|  +------+------+                              |
|         | Routes to one of:                   |
|    +----+----+----------+----------+          |
|    v         v          v          v          |
|  web_rag  doc_rag    db_rag    sql_agent      |
|  _agent   _agent     _agent                   |
+----+--------+----------+----------+-----------+
     |        |          |          |
     v        v          v          v
  Web RAG  Doc RAG    DB RAG    SQL RAG
  MCP      MCP        MCP       MCP
  Server   Server     Server    Server
                                  |
                         +--------+--------+
                         v                 v
                    Ollama/vLLM       PostgreSQL
                    (LLM on GPU)      (Database)
```

---

## Application Flow

### Step-by-Step

1. **User sends a question** via `POST /chat`
2. **Supervisor (LLM)** reads the question and routes to the right agent
3. **Agent** connects to its MCP Server via stdio pipe
4. **MCP Server** executes tools (search web, read docs, query DB, generate SQL)
5. **Response** flows back through the agent to the user

### Example Flow

```
User: "How many conversations happened last week?"
  |
  v
Supervisor: "This needs SQL" --> routes to sql_agent
  |
  v
sql_agent: calls text_to_sql() on SQL RAG MCP Server
  |
  v
SQL RAG Server:
  1. list_tables() --> finds "conversations" table
  2. get_table_schema("conversations") --> gets columns
  3. Builds prompt with schema context
  4. Calls Ollama LLM --> generates SQL
  5. Validates SQL (SELECT only)
  6. Executes on PostgreSQL
  7. Returns results
  |
  v
Response: "42 conversations happened in the last 7 days"
```

---

## Text-to-SQL Pipeline

The core of the SQL agent — converts natural language to SQL.

```
Step 1: Schema Linking
  - list_tables() --> find relevant tables
  - get_table_schema() --> get columns, types, sample data

Step 2: Prompt Building
  - System prompt: "You are a SQL expert"
  - Schema context: table names, columns, types
  - Few-shot examples: sample Q&A pairs
  - User question

Step 3: LLM Call (On-Premise)
  - Send prompt to Ollama at localhost:11434
  - LLM generates SQL query

Step 4: SQL Validation (Safety Gate)
  - Must start with SELECT
  - No DROP, DELETE, UPDATE, INSERT, ALTER, TRUNCATE
  - No hidden statements after semicolons

Step 5: Execute on PostgreSQL
  - Read-only connection
  - Max 100 rows returned

Step 6: Self-Correction (if SQL fails)
  - Send error back to LLM
  - LLM generates corrected SQL
  - Max 2 retry attempts
```

---

## MCP Protocol

### What is MCP?

MCP (Model Context Protocol) is a standard for connecting AI agents to tools. Instead of hardcoding tools in agent code, tools are **discovered at runtime**.

### Old Way (Hardcoded)

```python
from tools.sql_tool import text_to_sql
from tools.web_tool import search_web
```

### MCP Way (Runtime Discovery)

```python
tools = await mcp_session.list_tools()  # Server tells you what's available
result = await mcp_session.call_tool("text_to_sql", {"question": "..."})
```

### Benefits
- Add/remove tools without changing agent code
- Each MCP server is independent
- Plug-and-play architecture

---

## On-Premise LLM Deployment

### Why On-Premise Instead of Cloud API?

| Aspect | On-Premise | Cloud API (OpenAI) |
|--------|-----------|-------------------|
| Data Privacy | Data never leaves your servers | Data goes to third party |
| Cost | Fixed GPU cost, no per-token billing | Pay per token |
| Latency | No network hop | Network round-trip |
| Compliance | GDPR/SOC2 friendly | Requires DPA agreements |
| Control | Full control over model | Vendor lock-in |

### Ollama vs vLLM

| Aspect | Ollama | vLLM |
|--------|--------|------|
| Best for | Development, Mac/local | Production, Linux/GPU servers |
| GPU | Apple Metal, NVIDIA | NVIDIA only |
| Performance | Good | Optimized (batching, paged attention) |
| API | OpenAI-compatible | OpenAI-compatible |

Both expose the **same API format** — switching is just a URL change:

```
Ollama: http://ollama:11434/v1/chat/completions
vLLM:   http://vllm:8000/v1/chat/completions
OpenAI: https://api.openai.com/v1/chat/completions
```

### Multiple Models (Enterprise Setup like Suadeo)

Each model runs as a separate container with its own GPU:

```
chat-model     (Mistral-7B)     --> GPU 0, port 8000
audio-model    (Whisper)        --> GPU 1, port 8001
vision-model   (LLaVA)          --> GPU 2, port 8002
embedding-model (BGE-large)     --> GPU 3, port 8003
```

App code picks the right model via environment variables:

```python
chat_url = os.getenv("CHAT_MODEL_URL")          # http://chat-model:8000
embedding_url = os.getenv("EMBEDDING_MODEL_URL") # http://embedding-model:8003
```

---

## Docker & Containerization

### Key Concepts

```
Dockerfile  = Recipe/instructions to build an image
Image       = A package/blueprint (like a .zip of your app)
Container   = A running instance of an image (the actual app running)
Registry    = Storage for images (DockerHub, Harbor)
```

### Docker Build Flow

```
Dockerfile --> docker build --> Image --> docker run --> Container
```

### Base Image

```dockerfile
FROM python:3.12-slim   # Start with Linux + Python 3.12 (150MB)
```

| Image | Size | What's Inside |
|-------|------|--------------|
| python:3.12 | ~900MB | Full OS, build tools, everything |
| python:3.12-slim | ~150MB | Minimal OS, just Python |
| python:3.12-alpine | ~50MB | Ultra small, can have compatibility issues |

### Docker Compose

Runs multiple containers together with one command:

```bash
docker compose up -d     # Start all services
docker compose down      # Stop all services
docker compose pull      # Pull updated images
docker compose logs -f   # View logs
```

### Container Communication

Inside Docker network, containers talk by **service name**:

```python
# App connects to PostgreSQL
psycopg2.connect("postgresql://raguser:pass@postgres:5432/mydb")
#                                           ^^^^^^^ service name, not localhost

# App connects to Ollama
httpx.post("http://ollama:11434/v1/chat/completions")
#               ^^^^^^ service name
```

### Docker Volumes

Data that needs to persist across container restarts:

```yaml
volumes:
  pgdata:          # Database files survive restarts
  ollama_models:   # Downloaded models survive restarts
```

### Harbor vs DockerHub

```
DockerHub = Public registry (anyone can access)
Harbor    = Private registry (hosted inside company network)
```

Harbor is used in enterprises for data privacy — images never leave company servers.

---

## CI/CD Pipeline

### Pipeline Overview

```
git push to main
     |
     v
+-----------+   +---------------+
|   Lint    |   |  Unit Tests   |    <-- Run in parallel
+-----+-----+   +-------+-------+
      |                 |
      +--------+--------+
               v
      +------------------+
      | Integration Tests |    <-- Needs lint + unit tests
      | (real Postgres +  |
      |  real Ollama)     |
      +--------+---------+
               v
      +------------------+
      | Build & Push      |    <-- Builds Docker image
      | Docker Image      |       Pushes to DockerHub/Harbor
      +--------+---------+
               v
      +------------------+
      | Deploy            |    <-- SSH to production server
      | docker compose    |       Pull new image, restart app
      +------------------+
```

### Job Details

| Job | What It Does | When It Runs |
|-----|-------------|-------------|
| Lint | ruff check + format | Every push & PR |
| Unit Tests | 22 tests with mocked DB/LLM (0.2s) | Every push & PR |
| Integration Tests | Real PostgreSQL + Ollama in Docker | After lint + unit tests pass |
| Build & Push | Build Docker image, push to registry | After integration tests, main branch only |
| Deploy | SSH to server, pull image, restart | After build & push, main branch only |

### Unit Tests vs Integration Tests

```
Unit Tests:
  - Mock all external services (DB, LLM)
  - Fast (0.2 seconds for 22 tests)
  - Test logic only (SQL validation, SQL extraction)
  - No services needed

Integration Tests:
  - Use REAL PostgreSQL and REAL Ollama
  - Slower (depends on LLM response time)
  - Test full pipeline end-to-end
  - Skip automatically if services unavailable
```

### GitHub Secrets

Sensitive values stored in GitHub (never in code):

| Secret | Purpose |
|--------|---------|
| DOCKERHUB_USERNAME | DockerHub login |
| DOCKERHUB_TOKEN | DockerHub access token |
| SERVER_HOST | Production server IP (for deploy) |
| SERVER_USER | SSH username (for deploy) |
| SERVER_SSH_KEY | SSH private key (for deploy) |

Set at: GitHub Repo --> Settings --> Secrets and variables --> Actions

---

## Production Deployment

### Deploy Flow

```
CI/CD (GitHub Actions)
     |
     | SSH into production server
     v
Production Server:
     1. docker compose pull    <-- pulls NEW app image from registry
     2. docker compose up -d   <-- restarts ONLY the app container
     3. docker image prune -f  <-- deletes old unused images
```

### What Gets Restarted?

```
Before:  app v1.0 (old) + postgres (running) + ollama (running)
After:   app v1.1 (new) + postgres (running) + ollama (running)
                              |                    |
                         NOT restarted         NOT restarted
                         data preserved        model preserved
```

Only the **app container** gets replaced. Database and LLM keep running.

### Production Server Architecture

```
+----------------------------------------------------------+
|  Production Server (GPU Machine)                          |
|                                                           |
|  +----------+                                             |
|  |   App    | --> http://chat-model:8000  --> [GPU 0]     |
|  |  :8080   | --> http://audio-model:8001 --> [GPU 1]     |
|  |          | --> http://vision-model:8002--> [GPU 2]     |
|  |          | --> http://embed-model:8003 --> [GPU 3]     |
|  |          | --> postgresql://postgres:5432              |
|  +----------+                                             |
|                                                           |
|  All images pulled from Harbor (private registry)         |
|  All containers on same Docker network                    |
|  Each model gets dedicated GPU                            |
+----------------------------------------------------------+
```

---

## Nginx, Reverse Proxy & SSL

### Nginx

A web server that sits between the internet and your app. Like a **receptionist** at an office building.

```
Without Nginx:  Users --> directly access app (unsafe)
With Nginx:     Users --> Nginx (receptionist) --> forwards to app
```

### Reverse Proxy

Acts on behalf of the **server** (not the user):

```
Forward Proxy (VPN):   You --> Proxy --> Internet    (hides WHO is asking)
Reverse Proxy (Nginx): Internet --> Proxy --> App    (hides WHERE the app is)
```

Users see `https://ai.suadeo.com`, never `http://203.0.113.50:8080`.

Nginx can also route to multiple apps:

```
https://ai.suadeo.com/chat   --> Nginx --> localhost:8080 (AI app)
https://ai.suadeo.com/docs   --> Nginx --> localhost:3000 (Docs app)
https://ai.suadeo.com/admin  --> Nginx --> localhost:9090 (Admin app)
```

### SSL Certificate

Enables HTTPS — the lock icon in the browser.

```
HTTP:   Data sent as plain text    (like a postcard - anyone can read)
HTTPS:  Data is encrypted          (like a sealed envelope - only recipient reads)
```

How it works:

```
1. Browser connects to https://ai.suadeo.com
2. Nginx sends SSL certificate to browser
3. Browser verifies certificate is valid
4. Encrypted connection established
5. All data flows encrypted
```

Certificate providers:
- **Free:** Let's Encrypt (automated)
- **Paid:** DigiCert, Comodo (enterprise)

### Full Request Flow

```
User: https://ai.suadeo.com/chat
     |
     | HTTPS (encrypted by SSL)
     v
+---------+
|  Nginx  |  Reverse Proxy
|  :443   |  1. Decrypts HTTPS
|         |  2. Checks URL path
|         |  3. Forwards to app
+----+----+
     |
     | http://localhost:8080 (internal, plain HTTP - fine here)
     v
+---------+
|  App    |
|  :8080  |
+---------+
```

Between Nginx and your app it's plain HTTP — that's fine because it's internal and never leaves the server.

---

## Environment Variables

### Where They Come From (per environment)

| Environment | Source | Example |
|------------|--------|---------|
| Local dev | `.env` file on your laptop | `DATABASE_URL=...@localhost:5432` |
| CI/CD | `env:` in ci.yml + GitHub Secrets | `DATABASE_URL=...@localhost:5432` |
| Production | `docker-compose.yml` or `.env.production` on server | `DATABASE_URL=...@postgres:5432` |
| Docker image | **NEVER** — no secrets inside images | -- |

### Key Rule

`.env` file is **never** pushed to git or Docker image. Each environment manages its own configuration.

```
.env              --> .gitignore (not in git)
.env              --> .dockerignore (not in Docker image)
docker-compose    --> environment variables for production
GitHub Secrets    --> sensitive values for CI/CD
```

---

## Interview Quick Reference

### 30-Second Architecture Summary

> "I built a multi-agent system using LangGraph where a Supervisor routes user questions to specialized agents — web search, document retrieval, database lookup, and Text-to-SQL. Each agent connects to its tools via MCP protocol, so tools are discovered at runtime, not hardcoded. The SQL agent uses an on-premise LLM (Mistral-7B on GPU via Ollama) to convert natural language to SQL, validates it for safety (SELECT-only), executes against PostgreSQL, and has self-correction — if the SQL fails, it sends the error back to the LLM to fix."

### 30-Second CI/CD Summary

> "After CI passes all tests, the CD pipeline builds a Docker image of the application and pushes it to a container registry (Harbor). On the production server, Docker Compose orchestrates all services — the app, PostgreSQL, and LLM servers on GPUs. On each deploy, only the app container gets replaced — the database and LLM keep running with data preserved in Docker volumes."

### 30-Second Deployment Summary

> "Nginx acts as a reverse proxy in front of our application. Users access the public domain over HTTPS — Nginx handles SSL termination, decrypts the request, and forwards it to the app running internally on port 8080. The app server is never directly exposed to the internet, and all external traffic is encrypted."

### Key Technologies

| Component | Technology |
|-----------|-----------|
| Agent Framework | LangGraph (state machine with supervisor routing) |
| Tool Protocol | MCP (Model Context Protocol) |
| On-Premise LLM | Ollama (dev) / vLLM (production) |
| LLM Model | Mistral-7B |
| Database | PostgreSQL |
| API | FastAPI |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Registry | DockerHub (dev) / Harbor (enterprise) |
| Reverse Proxy | Nginx |
| Encryption | SSL/TLS certificates |
