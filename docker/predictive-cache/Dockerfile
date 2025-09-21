# BEV OSINT Framework - Predictive Cache Service
# Multi-tier caching with ML-driven optimization and intelligent prefetching

FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    build-essential \
    libpq-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache /app/models

# Create non-root user
RUN groupadd -r predictive_cache && \
    useradd -r -g predictive_cache -s /bin/bash predictive_cache && \
    chown -R predictive_cache:predictive_cache /app

# Switch to non-root user
USER predictive_cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8044/health || exit 1

# Expose ports
EXPOSE 8044 9044

# Entry point
CMD ["python", "-m", "src.infrastructure.predictive_cache_service"]