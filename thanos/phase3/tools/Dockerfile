FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir redis celery
CMD ["python", "-m", "http.server", "8000"]
