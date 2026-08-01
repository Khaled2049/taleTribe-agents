# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  && rm -rf /var/lib/apt/lists/*

# Copy Poetry manifests
COPY pyproject.toml poetry.lock ./

# Install Poetry + export plugin (export was removed from Poetry core in 2.0),
# export main deps to requirements, then remove Poetry
RUN pip install --no-cache-dir poetry poetry-plugin-export \
    && poetry export -f requirements.txt --only main --without-hashes -o /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip uninstall -y poetry poetry-plugin-export

# Copy the entire agents directory
COPY agents/ /app/agents/

# Copy the MCP server package (OAuth 2.1 AS + owner-scoped story tools).
# server.py falls back to an MCP-less app on ImportError, so forgetting this
# COPY ships a silently degraded image — keep it explicit.
COPY mcp_server/ /app/mcp_server/

# Copy the unified server and its root-level modules (imported by server.py)
COPY server.py config.py rate_limit.py /app/

# Disable local image generation in Docker (use external API instead)
ENV ENABLE_LOCAL_IMAGE_GENERATION=false

# Set Python path
ENV PYTHONPATH=/app

# Expose port (Cloud Run will set PORT env var, default to 8000 for local)
EXPOSE 8080

# Health check (Cloud Run uses its own health checks, but this is useful for local testing)
# Note: Cloud Run automatically sets PORT=8080, but we use env var for flexibility
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import os, httpx; port=os.getenv('PORT', '8000'); httpx.get(f'http://localhost:{port}/health', timeout=5)" || exit 1

# Run the unified server
CMD ["python", "server.py"]
