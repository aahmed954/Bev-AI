# Auto-Recovery Service Dockerfile
# ================================
#
# Multi-stage build for the BEV Auto-Recovery System
# Optimized for production deployment with security hardening

# Build stage
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    wget \
    ca-certificates \
    dumb-init \
    gosu \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r autorecovery \
    && useradd -r -g autorecovery -m -d /app -s /bin/bash autorecovery

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Docker CLI for container management
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian bullseye stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Install kubectl for Kubernetes management
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm kubectl

# Set working directory
WORKDIR /app

# Copy application code
COPY src/infrastructure/ ./infrastructure/
COPY config/auto_recovery.yaml ./config/auto_recovery.yaml
COPY docker/auto-recovery/entrypoint.sh ./entrypoint.sh

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/data \
    /app/backups \
    /app/snapshots \
    /app/config \
    && chown -R autorecovery:autorecovery /app

# Set permissions
RUN chmod +x ./entrypoint.sh

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO \
    WORKERS=4 \
    MAX_CONCURRENT_RECOVERIES=5 \
    RECOVERY_TIMEOUT=600 \
    CONFIG_PATH=/app/config/auto_recovery.yaml

# Expose ports
EXPOSE 8000 9000

# Labels for metadata
LABEL maintainer="BEV Infrastructure Team" \
      version="1.0.0" \
      description="Auto-Recovery Service for BEV OSINT Framework" \
      component="infrastructure" \
      criticality="critical"

# Use dumb-init as PID 1
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Switch to non-root user and start
USER autorecovery
CMD ["./entrypoint.sh"]