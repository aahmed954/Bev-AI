FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    python3-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    Django==3.2.20 \
    psycopg2-binary==2.9.9 \
    celery==5.2.7 \
    redis==4.5.4 \
    gunicorn==20.1.0 \
    djangorestframework==3.14.0 \
    django-cors-headers==3.14.0 \
    python-dotenv==1.0.0 \
    requests==2.31.0 \
    Pillow==9.5.0

# Create a basic Django project structure
RUN django-admin startproject intelowl .

# Create basic settings
RUN echo "from .settings import *" > /app/intelowl/local_settings.py && \
    echo "DATABASES = {" >> /app/intelowl/local_settings.py && \
    echo "    'default': {" >> /app/intelowl/local_settings.py && \
    echo "        'ENGINE': 'django.db.backends.postgresql'," >> /app/intelowl/local_settings.py && \
    echo "        'NAME': 'intelowl'," >> /app/intelowl/local_settings.py && \
    echo "        'USER': 'intelowl'," >> /app/intelowl/local_settings.py && \
    echo "        'PASSWORD': 'IntelOwl2024'," >> /app/intelowl/local_settings.py && \
    echo "        'HOST': 'bev_intelowl_postgres'," >> /app/intelowl/local_settings.py && \
    echo "        'PORT': '5432'," >> /app/intelowl/local_settings.py && \
    echo "    }" >> /app/intelowl/local_settings.py && \
    echo "}" >> /app/intelowl/local_settings.py && \
    echo "ALLOWED_HOSTS = ['*']" >> /app/intelowl/local_settings.py && \
    echo "CELERY_BROKER_URL = 'redis://bev_redis_standalone:6379/0'" >> /app/intelowl/local_settings.py && \
    echo "CELERY_RESULT_BACKEND = 'redis://bev_redis_standalone:6379/0'" >> /app/intelowl/local_settings.py

# Create entrypoint script
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'python manage.py migrate --noinput' >> /entrypoint.sh && \
    echo 'python manage.py collectstatic --noinput' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "intelowl.wsgi:application"]