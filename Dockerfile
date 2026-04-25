# Dockerfile for LangGraph MCP Agent
# ====================================
# Packages the FastAPI application into a Docker image.
# This image contains ONLY the app — PostgreSQL and Ollama
# run as separate containers (see docker-compose.yml).

FROM python:3.12-slim

# Install system dependencies needed by psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency files first (Docker layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Expose the app port
EXPOSE 8080

# Start the FastAPI application
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
