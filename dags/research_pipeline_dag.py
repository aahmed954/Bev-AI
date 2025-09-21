#!/usr/bin/env python3
"""
Airflow DAG: Research Pipeline Orchestration
Automated OSINT research, data collection, and intelligence synthesis
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.redis.operators.redis_publish import RedisPublishOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import json
import os

# Default arguments
default_args = {
    'owner': 'research_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}

# Research Pipeline DAG
research_dag = DAG(
    'research_pipeline',
    default_args=default_args,
    description='Automated OSINT research and intelligence gathering',
    schedule_interval='@hourly',
    catchup=False,
    max_active_runs=3,
    tags=['research', 'osint', 'intelligence']
)

def initialize_research_context(**context):
    """Initialize research context and targets"""
    import sys
    sys.path.append('/home/starlord/Bev/src')
    
    from agents.research_coordinator import ResearchCoordinator
    
    # Get research targets from Variable or use defaults
    targets = Variable.get('research_targets', deserialize_json=True, default_var=[])
    
    research_context = {
        'execution_date': context['execution_date'].isoformat(),
        'run_id': context['run_id'],
        'targets': targets,
        'depth': Variable.get('research_depth', default_var='deep'),
        'sources': ['osint', 'darknet', 'social', 'infrastructure'],
        'output_path': f"/home/starlord/Bev/data/research/{context['ds']}"
    }
    
    # Create output directory
    os.makedirs(research_context['output_path'], exist_ok=True)
    
    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='research_context', value=research_context)
    
    return research_context

def gather_osint_intelligence(**context):
    """Execute OSINT gathering"""
    import asyncio
    import sys
    sys.path.append('/home/starlord/Bev/src')
    
    from agents.research_coordinator import ResearchOracle
    
    research_context = context['task_instance'].xcom_pull(
        task_ids='initialize_context',
        key='research_context'
    )
    
    oracle = ResearchOracle({
        'dehashed_key': Variable.get('dehashed_api_key', default_var=''),
        'shodan_api_key': Variable.get('shodan_api_key', default_var=''),
        'censys_id': Variable.get('censys_api_id', default_var=''),
        'censys_secret': Variable.get('censys_api_secret', default_var='')
    })
    
    results = []
    
    async def run_investigation():
        for target in research_context['targets']:
            result = await oracle.investigate_target(
                target,
                depth=research_context['depth']
            )
            results.append(result)
    
    # Run async investigation
    asyncio.run(run_investigation())
    
    # Store results
    output_file = f"{research_context['output_path']}/osint_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    context['task_instance'].xcom_push(key='osint_results', value=output_file)
    
    return len(results)

# Initialize context task
init_task = PythonOperator(
    task_id='initialize_context',
    python_callable=initialize_research_context,
    dag=research_dag
)

# OSINT gathering task
osint_task = PythonOperator(
    task_id='osint_gathering',
    python_callable=gather_osint_intelligence,
    dag=research_dag
)

# Define task dependencies
init_task >> osint_task
