# Dockerfile for BEV Proxy Management Service
# Optimized for high-performance proxy pool management and geographic routing

FROM python:3.11-slim as builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    wget \
    iputils-ping \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    iputils-ping \
    dnsutils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Create directories
RUN mkdir -p /app/{src,config,data,logs} && \
    chown -R app:app /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=app:app ./src/ /app/src/
COPY --chown=app:app ./config/ /app/config/
COPY --chown=app:app ./app.py /app/

# Set permissions
RUN chmod +x /app/app.py

# Switch to app user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 9090

# Run application
CMD ["python", "/app/app.py"]