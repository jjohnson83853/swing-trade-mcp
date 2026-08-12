FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for health checks and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (pinned in requirements.txt)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application server
COPY server.py /app/server.py

# Configure environment variables for discoverability and logging
ENV MCP_SERVER_NAME="swing-trade-mcp"
ENV MCP_SERVER_VERSION="1.0.0"
ENV MCP_TOOL_COUNT_LOG="true"
ENV FASTMCP_HOST="0.0.0.0"
ENV FASTMCP_PORT="8000"

EXPOSE 8000

# Container healthcheck targeting the MCP health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "server.py"]
