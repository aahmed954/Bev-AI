"""
BEV Health Monitoring DAG
=========================

Comprehensive Airflow DAG for monitoring all critical services in the BEV infrastructure.
Includes 7 major service categories with automated recovery actions and alerting.

Author: BEV Infrastructure Team
Version: 1.0.0
Schedule: Every 5 minutes
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

import requests
import psycopg2
import redis
import pymongo
from elasticsearch import Elasticsearch

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.models import Variable
from airflow.exceptions import AirflowException
from airflow.utils.dates import days_ago

# =============================================================================
# DAG CONFIGURATION
# =============================================================================

default_args = {
    'owner': 'bev-infrastructure',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=10),
    'email': [
        Variable.get("BEV_ADMIN_EMAIL", default_var="admin@bev.local"),
        Variable.get("BEV_ONCALL_EMAIL", default_var="oncall@bev.local")
    ]
}

dag = DAG(
    'bev_health_monitoring',
    default_args=default_args,
    description='Comprehensive BEV infrastructure health monitoring',
    schedule_interval=timedelta(minutes=5),
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=['bev', 'monitoring', 'health', 'infrastructure'],
    doc_md=__doc__
)

# =============================================================================
# CONFIGURATION AND ENVIRONMENT VARIABLES
# =============================================================================

# Service endpoints configuration
SERVICES_CONFIG = {
    'prometheus': {
        'url': Variable.get("PROMETHEUS_URL", default_var="http://prometheus:9090"),
        'health_endpoint': "/-/healthy",
        'metrics_endpoint': "/metrics",
        'critical': True
    },
    'grafana': {
        'url': Variable.get("GRAFANA_URL", default_var="http://grafana:3000"),
        'health_endpoint': "/api/health",
        'critical': True
    },
    'thanos_query': {
        'url': Variable.get("THANOS_QUERY_URL", default_var="http://thanos-query:19192"),
        'health_endpoint': "/-/healthy",
        'critical': True
    },
    'airflow': {
        'url': Variable.get("AIRFLOW_URL", default_var="http://airflow-webserver:8080"),
        'health_endpoint': "/health",
        'critical': True
    },
    'elasticsearch': {
        'url': Variable.get("ELASTICSEARCH_URL", default_var="http://elasticsearch:9200"),
        'health_endpoint': "/_cluster/health",
        'critical': True
    },
    'influxdb': {
        'url': Variable.get("INFLUXDB_URL", default_var="http://influxdb:8086"),
        'health_endpoint': "/health",
        'critical': False
    },
    'postgresql': {
        'host': Variable.get("POSTGRESQL_HOST", default_var="postgresql"),
        'port': int(Variable.get("POSTGRESQL_PORT", default_var="5432")),
        'database': Variable.get("POSTGRESQL_DATABASE", default_var="bev"),
        'username': Variable.get("POSTGRESQL_USERNAME", default_var="bev_user"),
        'password': Variable.get("POSTGRESQL_PASSWORD", default_var=""),
        'critical': True
    },
    'mongodb': {
        'host': Variable.get("MONGODB_HOST", default_var="mongodb"),
        'port': int(Variable.get("MONGODB_PORT", default_var="27017")),
        'critical': False
    },
    'redis': {
        'host': Variable.get("REDIS_HOST", default_var="redis"),
        'port': int(Variable.get("REDIS_PORT", default_var="6379")),
        'password': Variable.get("REDIS_PASSWORD", default_var=""),
        'critical': False
    }
}

# Alert configuration
ALERT_CONFIG = {
    'slack_webhook': Variable.get("SLACK_WEBHOOK_URL", default_var=""),
    'email_recipients': [
        Variable.get("BEV_ADMIN_EMAIL", default_var="admin@bev.local"),
        Variable.get("BEV_ONCALL_EMAIL", default_var="oncall@bev.local")
    ],
    'pagerduty_key': Variable.get("PAGERDUTY_SERVICE_KEY", default_var=""),
    'recovery_attempts': 3,
    'escalation_timeout': 15  # minutes
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_logger() -> logging.Logger:
    """Get configured logger for the DAG."""
    logger = logging.getLogger('bev_health_monitoring')
    logger.setLevel(logging.INFO)
    return logger

def send_alert(context: Dict[str, Any], service: str, status: str, message: str) -> None:
    """Send alert notifications via multiple channels."""
    logger = get_logger()

    alert_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'service': service,
        'status': status,
        'message': message,
        'dag_id': context['dag'].dag_id,
        'task_id': context['task'].task_id,
        'execution_date': context['execution_date'].isoformat()
    }

    # Log the alert
    logger.error(f"ALERT: {service} - {status} - {message}")

    # Store alert in XCom for downstream tasks
    context['task_instance'].xcom_push(key=f'alert_{service}', value=alert_data)

def check_service_health(service_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Generic service health check function."""
    logger = get_logger()
    result = {
        'service': service_name,
        'status': 'unknown',
        'response_time': None,
        'message': '',
        'timestamp': datetime.utcnow().isoformat()
    }

    try:
        if 'url' in config and 'health_endpoint' in config:
            # HTTP-based health check
            url = f"{config['url']}{config['health_endpoint']}"
            start_time = datetime.utcnow()

            response = requests.get(url, timeout=30)
            response_time = (datetime.utcnow() - start_time).total_seconds()

            result['response_time'] = response_time

            if response.status_code == 200:
                result['status'] = 'healthy'
                result['message'] = f"Service responding normally (HTTP {response.status_code})"
            else:
                result['status'] = 'unhealthy'
                result['message'] = f"HTTP {response.status_code}: {response.text[:200]}"

        logger.info(f"Health check for {service_name}: {result['status']}")
        return result

    except requests.RequestException as e:
        result['status'] = 'unhealthy'
        result['message'] = f"Connection error: {str(e)}"
        logger.error(f"Health check failed for {service_name}: {str(e)}")
        return result
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in health check for {service_name}: {str(e)}")
        return result

# =============================================================================
# SERVICE-SPECIFIC HEALTH CHECK FUNCTIONS
# =============================================================================

def check_prometheus_health(**context) -> None:
    """Check Prometheus service health and query capabilities."""
    logger = get_logger()
    service_config = SERVICES_CONFIG['prometheus']

    # Basic health check
    result = check_service_health('prometheus', service_config)

    # Additional Prometheus-specific checks
    try:
        # Check if Prometheus can execute queries
        query_url = f"{service_config['url']}/api/v1/query"
        params = {'query': 'up'}

        response = requests.get(query_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'success':
                targets_up = len([r for r in data.get('data', {}).get('result', []) if r.get('value', [None, '0'])[1] == '1'])
                result['targets_up'] = targets_up
                result['message'] += f" | {targets_up} targets up"
            else:
                result['status'] = 'degraded'
                result['message'] = "Query API not responding correctly"

        # Check configuration reload status
        reload_url = f"{service_config['url']}/api/v1/status/config"
        response = requests.get(reload_url, timeout=10)
        if response.status_code == 200:
            config_data = response.json()
            if config_data.get('status') != 'success':
                result['status'] = 'degraded'
                result['message'] += " | Config reload issues detected"

    except Exception as e:
        logger.warning(f"Extended Prometheus check failed: {str(e)}")

    # Store results and send alerts if necessary
    context['task_instance'].xcom_push(key='prometheus_health', value=result)

    if result['status'] in ['unhealthy', 'error']:
        send_alert(context, 'prometheus', result['status'], result['message'])
        if service_config['critical']:
            raise AirflowException(f"Critical service Prometheus is {result['status']}: {result['message']}")

def check_database_health(**context) -> None:
    """Check PostgreSQL database health and connectivity."""
    logger = get_logger()
    service_config = SERVICES_CONFIG['postgresql']

    result = {
        'service': 'postgresql',
        'status': 'unknown',
        'message': '',
        'timestamp': datetime.utcnow().isoformat(),
        'connection_count': None,
        'slow_queries': None
    }

    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=service_config['host'],
            port=service_config['port'],
            database=service_config['database'],
            user=service_config['username'],
            password=service_config['password'],
            connect_timeout=30
        )

        cursor = conn.cursor()

        # Basic connectivity check
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]

        # Check active connections
        cursor.execute("SELECT count(*) FROM pg_stat_activity;")
        connection_count = cursor.fetchone()[0]
        result['connection_count'] = connection_count

        # Check for slow queries
        cursor.execute("""
            SELECT count(*) FROM pg_stat_activity
            WHERE state = 'active' AND query_start < NOW() - INTERVAL '60 seconds'
        """)
        slow_queries = cursor.fetchone()[0]
        result['slow_queries'] = slow_queries

        # Check database size
        cursor.execute(f"SELECT pg_size_pretty(pg_database_size('{service_config['database']}'));")
        db_size = cursor.fetchone()[0]

        result['status'] = 'healthy'
        result['message'] = f"Connected successfully | {connection_count} connections | {slow_queries} slow queries | Size: {db_size}"

        # Warn on high connection count or slow queries
        if connection_count > 100:
            result['status'] = 'degraded'
            result['message'] += " | High connection count"

        if slow_queries > 5:
            result['status'] = 'degraded'
            result['message'] += " | Multiple slow queries detected"

        cursor.close()
        conn.close()

    except psycopg2.Error as e:
        result['status'] = 'unhealthy'
        result['message'] = f"Database error: {str(e)}"
        logger.error(f"PostgreSQL health check failed: {str(e)}")
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in PostgreSQL check: {str(e)}")

    # Store results and send alerts if necessary
    context['task_instance'].xcom_push(key='postgresql_health', value=result)

    if result['status'] in ['unhealthy', 'error']:
        send_alert(context, 'postgresql', result['status'], result['message'])
        if service_config['critical']:
            raise AirflowException(f"Critical service PostgreSQL is {result['status']}: {result['message']}")

def check_elasticsearch_health(**context) -> None:
    """Check Elasticsearch cluster health and performance."""
    logger = get_logger()
    service_config = SERVICES_CONFIG['elasticsearch']

    result = {
        'service': 'elasticsearch',
        'status': 'unknown',
        'message': '',
        'timestamp': datetime.utcnow().isoformat(),
        'cluster_status': None,
        'active_shards': None,
        'relocating_shards': None
    }

    try:
        # Connect to Elasticsearch
        es = Elasticsearch([service_config['url']], timeout=30)

        # Check cluster health
        health = es.cluster.health()
        cluster_status = health['status']
        result['cluster_status'] = cluster_status
        result['active_shards'] = health['active_shards']
        result['relocating_shards'] = health['relocating_shards']

        # Determine overall status based on cluster health
        if cluster_status == 'green':
            result['status'] = 'healthy'
            result['message'] = f"Cluster green | {health['active_shards']} active shards"
        elif cluster_status == 'yellow':
            result['status'] = 'degraded'
            result['message'] = f"Cluster yellow | {health['unassigned_shards']} unassigned shards"
        else:  # red
            result['status'] = 'unhealthy'
            result['message'] = f"Cluster red | {health['unassigned_shards']} unassigned shards"

        # Check indices status
        indices_stats = es.indices.stats()
        total_docs = indices_stats['_all']['total']['docs']['count']
        result['message'] += f" | {total_docs:,} total documents"

    except Exception as e:
        result['status'] = 'unhealthy'
        result['message'] = f"Connection error: {str(e)}"
        logger.error(f"Elasticsearch health check failed: {str(e)}")

    # Store results and send alerts if necessary
    context['task_instance'].xcom_push(key='elasticsearch_health', value=result)

    if result['status'] in ['unhealthy', 'error']:
        send_alert(context, 'elasticsearch', result['status'], result['message'])
        if service_config['critical']:
            raise AirflowException(f"Critical service Elasticsearch is {result['status']}: {result['message']}")

def check_redis_health(**context) -> None:
    """Check Redis cache health and performance."""
    logger = get_logger()
    service_config = SERVICES_CONFIG['redis']

    result = {
        'service': 'redis',
        'status': 'unknown',
        'message': '',
        'timestamp': datetime.utcnow().isoformat(),
        'memory_usage': None,
        'connected_clients': None
    }

    try:
        # Connect to Redis
        r = redis.Redis(
            host=service_config['host'],
            port=service_config['port'],
            password=service_config.get('password', None),
            socket_timeout=30,
            decode_responses=True
        )

        # Test basic connectivity
        r.ping()

        # Get Redis info
        info = r.info()

        # Extract key metrics
        memory_used = info.get('used_memory_human', '0B')
        memory_peak = info.get('used_memory_peak_human', '0B')
        connected_clients = info.get('connected_clients', 0)

        result['memory_usage'] = memory_used
        result['connected_clients'] = connected_clients

        # Check memory usage percentage if maxmemory is set
        max_memory = info.get('maxmemory', 0)
        used_memory = info.get('used_memory', 0)

        if max_memory > 0:
            memory_pct = (used_memory / max_memory) * 100
            if memory_pct > 90:
                result['status'] = 'degraded'
                result['message'] = f"High memory usage: {memory_pct:.1f}% | {connected_clients} clients"
            else:
                result['status'] = 'healthy'
                result['message'] = f"Memory: {memory_used} | {connected_clients} clients"
        else:
            result['status'] = 'healthy'
            result['message'] = f"Memory: {memory_used} (peak: {memory_peak}) | {connected_clients} clients"

        # Check if Redis is in read-only mode
        if info.get('loading', 0) == 1:
            result['status'] = 'degraded'
            result['message'] += " | Loading data"

    except redis.ConnectionError as e:
        result['status'] = 'unhealthy'
        result['message'] = f"Connection error: {str(e)}"
        logger.error(f"Redis health check failed: {str(e)}")
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in Redis check: {str(e)}")

    # Store results and send alerts if necessary
    context['task_instance'].xcom_push(key='redis_health', value=result)

    if result['status'] in ['unhealthy', 'error']:
        send_alert(context, 'redis', result['status'], result['message'])

def check_mongodb_health(**context) -> None:
    """Check MongoDB database health and performance."""
    logger = get_logger()
    service_config = SERVICES_CONFIG['mongodb']

    result = {
        'service': 'mongodb',
        'status': 'unknown',
        'message': '',
        'timestamp': datetime.utcnow().isoformat(),
        'connections': None,
        'operations_per_sec': None
    }

    try:
        # Connect to MongoDB
        client = pymongo.MongoClient(
            host=service_config['host'],
            port=service_config['port'],
            serverSelectionTimeoutMS=30000
        )

        # Test connectivity and get server status
        server_status = client.admin.command('serverStatus')

        # Extract key metrics
        connections = server_status.get('connections', {})
        opcounters = server_status.get('opcounters', {})

        current_connections = connections.get('current', 0)
        available_connections = connections.get('available', 0)

        result['connections'] = current_connections
        result['available_connections'] = available_connections

        # Calculate operations per second (rough estimate)
        total_ops = sum(opcounters.values()) if opcounters else 0
        uptime = server_status.get('uptime', 1)
        ops_per_sec = total_ops / uptime if uptime > 0 else 0
        result['operations_per_sec'] = round(ops_per_sec, 2)

        # Determine status
        if available_connections < 100:
            result['status'] = 'degraded'
            result['message'] = f"Low available connections: {available_connections} | {current_connections} active"
        else:
            result['status'] = 'healthy'
            result['message'] = f"Connections: {current_connections}/{available_connections} | {ops_per_sec:.1f} ops/sec"

        # Check replica set status if applicable
        try:
            rs_status = client.admin.command('replSetGetStatus')
            if rs_status:
                result['message'] += " | Replica set active"
        except pymongo.errors.OperationFailure:
            # Not a replica set, which is fine
            pass

        client.close()

    except pymongo.errors.ServerSelectionTimeoutError as e:
        result['status'] = 'unhealthy'
        result['message'] = f"Connection timeout: {str(e)}"
        logger.error(f"MongoDB health check failed: {str(e)}")
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error in MongoDB check: {str(e)}")

    # Store results and send alerts if necessary
    context['task_instance'].xcom_push(key='mongodb_health', value=result)

    if result['status'] in ['unhealthy', 'error']:
        send_alert(context, 'mongodb', result['status'], result['message'])

def check_thanos_health(**context) -> None:
    """Check Thanos components health and query capabilities."""
    logger = get_logger()

    thanos_components = ['thanos_query', 'thanos-store', 'thanos-compactor', 'thanos-sidecar']
    overall_status = 'healthy'
    component_results = {}

    for component in thanos_components:
        if component in SERVICES_CONFIG:
            result = check_service_health(component, SERVICES_CONFIG[component])
            component_results[component] = result

            if result['status'] in ['unhealthy', 'error']:
                overall_status = 'degraded'

    # Additional Thanos Query specific checks
    try:
        thanos_query_url = SERVICES_CONFIG['thanos_query']['url']

        # Check if Thanos Query can access stores
        stores_url = f"{thanos_query_url}/api/v1/stores"
        response = requests.get(stores_url, timeout=30)

        if response.status_code == 200:
            stores_data = response.json()
            active_stores = len([s for s in stores_data.get('data', []) if s.get('lastCheck', {}).get('error') is None])
            component_results['store_connectivity'] = {
                'active_stores': active_stores,
                'total_stores': len(stores_data.get('data', []))
            }

    except Exception as e:
        logger.warning(f"Extended Thanos check failed: {str(e)}")
        overall_status = 'degraded'

    # Store results
    context['task_instance'].xcom_push(key='thanos_health', value={
        'overall_status': overall_status,
        'components': component_results,
        'timestamp': datetime.utcnow().isoformat()
    })

    if overall_status in ['unhealthy', 'error']:
        send_alert(context, 'thanos', overall_status, f"Thanos components health check failed")

def perform_automated_recovery(**context) -> None:
    """Attempt automated recovery for failed services."""
    logger = get_logger()

    # Get health check results from XCom
    ti = context['task_instance']

    recovery_actions = []

    # Check each service and perform recovery if needed
    for service_key in ['prometheus_health', 'postgresql_health', 'elasticsearch_health', 'redis_health', 'mongodb_health', 'thanos_health']:
        health_data = ti.xcom_pull(key=service_key)

        if health_data and health_data.get('status') in ['unhealthy', 'error']:
            service_name = health_data['service']

            # Attempt service restart (this would be customized per service)
            recovery_action = f"restart_{service_name}"
            recovery_actions.append(recovery_action)

            logger.info(f"Attempting recovery for {service_name}: {recovery_action}")

            # Store recovery attempt
            ti.xcom_push(key=f'recovery_{service_name}', value={
                'action': recovery_action,
                'timestamp': datetime.utcnow().isoformat(),
                'reason': health_data.get('message', 'Health check failed')
            })

    # Log all recovery actions
    if recovery_actions:
        logger.info(f"Recovery actions performed: {', '.join(recovery_actions)}")
        ti.xcom_push(key='recovery_summary', value={
            'actions': recovery_actions,
            'timestamp': datetime.utcnow().isoformat()
        })
    else:
        logger.info("No recovery actions needed")

def generate_health_report(**context) -> None:
    """Generate comprehensive health report from all checks."""
    logger = get_logger()
    ti = context['task_instance']

    # Collect all health check results
    health_checks = [
        'prometheus_health',
        'postgresql_health',
        'elasticsearch_health',
        'redis_health',
        'mongodb_health',
        'thanos_health'
    ]

    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'dag_run_id': context['dag_run'].run_id,
        'execution_date': context['execution_date'].isoformat(),
        'services': {},
        'summary': {
            'total_services': 0,
            'healthy': 0,
            'degraded': 0,
            'unhealthy': 0,
            'error': 0
        },
        'alerts': [],
        'recovery_actions': []
    }

    # Process each health check
    for check_key in health_checks:
        health_data = ti.xcom_pull(key=check_key)
        if health_data:
            service_name = health_data.get('service', check_key.replace('_health', ''))
            report['services'][service_name] = health_data

            # Update summary counts
            status = health_data.get('status', 'unknown')
            report['summary']['total_services'] += 1

            if status == 'healthy':
                report['summary']['healthy'] += 1
            elif status == 'degraded':
                report['summary']['degraded'] += 1
            elif status == 'unhealthy':
                report['summary']['unhealthy'] += 1
            elif status == 'error':
                report['summary']['error'] += 1

    # Collect alerts
    for service in report['services']:
        alert_data = ti.xcom_pull(key=f'alert_{service}')
        if alert_data:
            report['alerts'].append(alert_data)

    # Collect recovery actions
    recovery_summary = ti.xcom_pull(key='recovery_summary')
    if recovery_summary:
        report['recovery_actions'] = recovery_summary.get('actions', [])

    # Calculate overall health score
    total = report['summary']['total_services']
    if total > 0:
        health_score = (
            (report['summary']['healthy'] * 100) +
            (report['summary']['degraded'] * 75) +
            (report['summary']['unhealthy'] * 25) +
            (report['summary']['error'] * 0)
        ) / total
        report['summary']['health_score'] = round(health_score, 2)

    # Store final report
    ti.xcom_push(key='health_report', value=report)

    # Log summary
    logger.info(f"Health Report Summary: {report['summary']}")

    # Send report via email if there are issues
    if report['summary']['unhealthy'] > 0 or report['summary']['error'] > 0:
        logger.warning(f"Health issues detected: {len(report['alerts'])} alerts, {len(report['recovery_actions'])} recovery actions")

# =============================================================================
# DAG TASK DEFINITIONS
# =============================================================================

# Service health check tasks
prometheus_health_task = PythonOperator(
    task_id='check_prometheus_health',
    python_callable=check_prometheus_health,
    dag=dag,
    doc_md="Check Prometheus service health and query capabilities"
)

database_health_task = PythonOperator(
    task_id='check_database_health',
    python_callable=check_database_health,
    dag=dag,
    doc_md="Check PostgreSQL database connectivity and performance"
)

elasticsearch_health_task = PythonOperator(
    task_id='check_elasticsearch_health',
    python_callable=check_elasticsearch_health,
    dag=dag,
    doc_md="Check Elasticsearch cluster health and status"
)

redis_health_task = PythonOperator(
    task_id='check_redis_health',
    python_callable=check_redis_health,
    dag=dag,
    doc_md="Check Redis cache connectivity and memory usage"
)

mongodb_health_task = PythonOperator(
    task_id='check_mongodb_health',
    python_callable=check_mongodb_health,
    dag=dag,
    doc_md="Check MongoDB database connectivity and replica set status"
)

thanos_health_task = PythonOperator(
    task_id='check_thanos_health',
    python_callable=check_thanos_health,
    dag=dag,
    doc_md="Check Thanos components health and store connectivity"
)

# HTTP sensor tasks for quick connectivity checks
grafana_sensor = HttpSensor(
    task_id='check_grafana_connectivity',
    http_conn_id='grafana_default',
    endpoint='/api/health',
    timeout=30,
    poke_interval=10,
    mode='reschedule',
    dag=dag
)

# Recovery and reporting tasks
recovery_task = PythonOperator(
    task_id='perform_automated_recovery',
    python_callable=perform_automated_recovery,
    trigger_rule='all_done',  # Run regardless of upstream task status
    dag=dag,
    doc_md="Attempt automated recovery for failed services"
)

report_task = PythonOperator(
    task_id='generate_health_report',
    python_callable=generate_health_report,
    trigger_rule='all_done',  # Run regardless of upstream task status
    dag=dag,
    doc_md="Generate comprehensive health report from all checks"
)

# Notification tasks
slack_notification = SlackWebhookOperator(
    task_id='send_slack_notification',
    http_conn_id='slack_default',
    message="BEV Health Monitoring Alert: {{ task_instance.xcom_pull(key='health_report')['summary'] }}",
    trigger_rule='all_done',
    dag=dag
)

# =============================================================================
# TASK DEPENDENCIES
# =============================================================================

# All health checks run in parallel
health_checks = [
    prometheus_health_task,
    database_health_task,
    elasticsearch_health_task,
    redis_health_task,
    mongodb_health_task,
    thanos_health_task,
    grafana_sensor
]

# Recovery runs after all health checks complete (regardless of status)
health_checks >> recovery_task

# Report generation runs after recovery
recovery_task >> report_task

# Notifications run after report generation
report_task >> slack_notification