#!/usr/bin/env python3
"""
Celery Application Configuration for ORACLE1 Workers
Centralized configuration and task routing
"""

import os
from celery import Celery
from pydantic_settings import BaseSettings

class CelerySettings(BaseSettings):
    """Celery configuration settings"""

    # Redis configuration
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""

    # Celery configuration
    broker_url: str = f"redis://{redis_host}:{redis_port}/{redis_db}"
    result_backend: str = f"redis://{redis_host}:{redis_port}/{redis_db}"

    # Task routing
    task_routes: dict = {
        'edge_worker.*': {'queue': 'edge_computing'},
        'genetic_worker.*': {'queue': 'genetic_optimization'},
        'knowledge_worker.*': {'queue': 'knowledge_synthesis'},
        'toolmaster_worker.*': {'queue': 'tool_orchestration'},
    }

    # Worker configuration
    worker_prefetch_multiplier: int = 1
    task_acks_late: bool = True
    worker_max_tasks_per_child: int = 100

    # Monitoring
    worker_send_task_events: bool = True
    task_send_sent_event: bool = True

    class Config:
        env_prefix = "CELERY_"

settings = CelerySettings()

# Create Celery application
app = Celery('oracle1_pipeline')

# Configure Celery
app.conf.update(
    broker_url=settings.broker_url,
    result_backend=settings.result_backend,
    task_routes=settings.task_routes,
    worker_prefetch_multiplier=settings.worker_prefetch_multiplier,
    task_acks_late=settings.task_acks_late,
    worker_max_tasks_per_child=settings.worker_max_tasks_per_child,
    worker_send_task_events=settings.worker_send_task_events,
    task_send_sent_event=settings.task_send_sent_event,

    # Task serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,

    # Task result expiration
    result_expires=3600,

    # Queue configuration
    task_default_queue='default',
    task_queues={
        'edge_computing': {
            'exchange': 'edge_computing',
            'exchange_type': 'direct',
            'routing_key': 'edge_computing',
        },
        'genetic_optimization': {
            'exchange': 'genetic_optimization',
            'exchange_type': 'direct',
            'routing_key': 'genetic_optimization',
        },
        'knowledge_synthesis': {
            'exchange': 'knowledge_synthesis',
            'exchange_type': 'direct',
            'routing_key': 'knowledge_synthesis',
        },
        'tool_orchestration': {
            'exchange': 'tool_orchestration',
            'exchange_type': 'direct',
            'routing_key': 'tool_orchestration',
        },
    },

    # Error handling
    task_reject_on_worker_lost=True,
    task_ignore_result=False,

    # Security
    worker_hijack_root_logger=False,
    worker_log_color=False,
)

# Auto-discover tasks
app.autodiscover_tasks([
    'edge_worker',
    'genetic_worker',
    'knowledge_worker',
    'toolmaster_worker'
])

if __name__ == '__main__':
    app.start()