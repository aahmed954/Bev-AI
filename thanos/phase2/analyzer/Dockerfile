FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir pytesseract pillow
CMD ["python", "-m", "http.server", "8000"]
