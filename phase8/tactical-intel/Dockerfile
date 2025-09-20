FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir flask redis
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
