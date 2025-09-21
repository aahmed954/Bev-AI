# Multi-stage build for Celery Pipeline Workers
FROM python:3.12-slim as builder

# Install system dependencies for ARM compatibility
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libhdf5-dev \
    libatlas-base-dev \
    gfortran \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-103 \
    libatlas3-base \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r celeryuser && useradd -r -g celeryuser celeryuser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY edge_worker.py .
COPY genetic_worker.py .
COPY knowledge_worker.py .
COPY toolmaster_worker.py .
COPY celery_app.py .
COPY --chown=celeryuser:celeryuser . .

# Create directories for processing
RUN mkdir -p /app/logs /app/cache /app/models && \
    chown -R celeryuser:celeryuser /app

# Switch to non-root user
USER celeryuser

# Expose port for monitoring
EXPOSE 5555

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD celery -A celery_app inspect ping || exit 1

# Default command (can be overridden for specific workers)
CMD ["celery", "-A", "celery_app", "worker", "--loglevel=info", "--concurrency=4"]