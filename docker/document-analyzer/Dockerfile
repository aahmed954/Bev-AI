# Multi-stage build for Document Analyzer with NLP processing
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
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm && \
    python -m spacy download en_core_web_lg

# Production stage
FROM python:3.12-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-103 \
    libatlas3-base \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r analyzer && useradd -r -g analyzer analyzer

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY analyzer.py .
COPY --chown=analyzer:analyzer . .

# Create directories for processing
RUN mkdir -p /app/models /app/cache && \
    chown -R analyzer:analyzer /app

# Switch to non-root user
USER analyzer

# Expose port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Start the application
CMD ["uvicorn", "analyzer:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "2"]