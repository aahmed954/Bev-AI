FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || pip install requests
COPY . .
CMD ["python", "analyzer.py"]
