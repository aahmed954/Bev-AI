# BEV OSINT MCP Server Dockerfile
# Multi-stage build for production-ready container

# Build stage
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    MCP_SERVER_PORT=3010 \
    MCP_SERVER_HOST=0.0.0.0

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    dnsutils \
    whois \
    openssl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r mcp && useradd -r -g mcp -d /app -s /bin/bash mcp

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/config /var/log && \
    chown -R mcp:mcp /app /var/log

# Copy application code
COPY --chown=mcp:mcp src/mcp_server/ /app/mcp_server/
COPY --chown=mcp:mcp src/infrastructure/ /app/infrastructure/

# Copy configuration files
COPY --chown=mcp:mcp config/ /app/config/

# Create health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:$MCP_SERVER_PORT/health || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh && \
    chown mcp:mcp /app/healthcheck.sh

# Switch to non-root user
USER mcp

# Expose port
EXPOSE 3010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /app/healthcheck.sh

# Set up entry point
ENTRYPOINT ["python", "-m", "mcp_server.main"]

# Default command
CMD []

# Labels for metadata
LABEL maintainer="BEV OSINT Framework" \
      version="1.0.0" \
      description="Secure MCP Server for OSINT Operations" \
      security.features="JWT Auth, Rate Limiting, Input Validation, Audit Logging" \
      performance.features="Connection Pooling, Async Operations, Background Tasks" \
      integration.databases="PostgreSQL, Neo4j, Redis, Elasticsearch"