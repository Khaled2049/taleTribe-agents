# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
  gcc \
  && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the entire agents directory
COPY agents/ /app/agents/

# Copy the unified server
COPY server.py /app/server.py

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
