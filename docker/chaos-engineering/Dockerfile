FROM python:3.11-slim

# Install system dependencies for chaos engineering tools
RUN apt-get update && apt-get install -y \
    iproute2 \
    iptables \
    stress-ng \
    curl \
    wget \
    net-tools \
    procps \
    htop \
    tcpdump \
    traceroute \
    iputils-ping \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/config /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the chaos engineering service
CMD ["python", "-m", "src.testing.chaos_engineer"]