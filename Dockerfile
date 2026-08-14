FROM python:3.12-slim

WORKDIR /app

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

# Container healthcheck targeting the MCP health endpoint.
# Uses stdlib urllib instead of curl/wget so no extra system package is needed.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["python", "server.py"]
