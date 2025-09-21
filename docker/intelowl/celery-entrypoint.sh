#!/bin/bash
set -e

# Wait for Django to be ready
echo "Waiting for Django service to be ready..."
sleep 10

# Start Celery based on the worker type
case "$CELERY_WORKER_TYPE" in
  "beat")
    echo "Starting Celery Beat..."
    exec celery -A intelowl beat -l INFO
    ;;
  "worker")
    echo "Starting Celery Worker..."
    exec celery -A intelowl worker -l INFO -Q default,high_priority,low_priority -c 4
    ;;
  *)
    echo "Unknown CELERY_WORKER_TYPE: $CELERY_WORKER_TYPE"
    exit 1
    ;;
esac