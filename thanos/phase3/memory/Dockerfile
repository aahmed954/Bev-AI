# Memory Management Service
FROM python:3.11-slim

LABEL maintainer="BEV OSINT Team"
LABEL description="Memory Management Service"

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install dependencies if requirements exist
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py"]
