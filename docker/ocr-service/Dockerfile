# Multi-stage build for OCR service with Tesseract
FROM python:3.12-slim as builder

# Install system dependencies for ARM compatibility
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    tesseract-ocr-rus \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-ara \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    poppler-utils \
    libmagic1 \
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
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    tesseract-ocr-deu \
    tesseract-ocr-spa \
    tesseract-ocr-ita \
    tesseract-ocr-por \
    tesseract-ocr-rus \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-jpn \
    tesseract-ocr-kor \
    tesseract-ocr-ara \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r ocruser && useradd -r -g ocruser ocruser

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY app.py .
COPY --chown=ocruser:ocruser . .

# Create directories for file processing
RUN mkdir -p /app/uploads /app/processed && \
    chown -R ocruser:ocruser /app

# Switch to non-root user
USER ocruser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]