FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy infrastructure requirements and install dependencies
COPY src/infrastructure/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy auto recovery source code
COPY src/infrastructure/ /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose auto recovery port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run auto recovery service
CMD ["python", "auto_recovery.py"]