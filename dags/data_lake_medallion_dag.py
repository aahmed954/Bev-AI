"""
BEV Data Lake Medallion Architecture DAG
Complete Bronze → Silver → Gold data pipeline
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.s3_key_sensor import S3KeySensor
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json
import hashlib

default_args = {
    'owner': 'bev-swarm',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2)
}

# Medallion Architecture DAG
dag = DAG(
    'data_lake_medallion_architecture',
    default_args=default_args,
    description='Complete Bronze-Silver-Gold data pipeline',
    schedule_interval='@hourly',
    catchup=False,
    tags=['data-lake', 'medallion', 'etl']
)

def ingest_to_bronze(**context):
    """Ingest raw data to Bronze layer"""
    import boto3
    import requests
    from datetime import datetime
    
    s3 = boto3.client('s3')
    
    # Data sources
    sources = {
        'research': 'http://api.research.bev/latest',
        'security': 'http://api.security.bev/incidents',
        'metrics': 'http://api.metrics.bev/telemetry',
        'agents': 'http://api.agents.bev/activities'
    }
    
    ingested_files = []
    
    for source_name, url in sources.items():
        try:
            # Fetch data
            response = requests.get(url, timeout=30)
            data = response.json()
            
            # Add metadata
            enriched_data = {
                'source': source_name,
                'ingestion_timestamp': datetime.utcnow().isoformat(),
                'raw_data': data,
                'schema_version': '1.0'
            }
            
            # Generate unique key
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            file_key = f"bronze/{source_name}/year={datetime.utcnow().year}/month={datetime.utcnow().month:02d}/day={datetime.utcnow().day:02d}/{source_name}_{timestamp}.json"
            
            # Upload to S3
            s3.put_object(
                Bucket='bev-data-lake',
                Key=file_key,
                Body=json.dumps(enriched_data),
                ContentType='application/json'
            )
            
            ingested_files.append(file_key)
            print(f"Ingested {source_name} to {file_key}")
            
        except Exception as e:
            print(f"Failed to ingest {source_name}: {e}")
    
    # Push file list to XCom for next task
    context['task_instance'].xcom_push(key='bronze_files', value=ingested_files)
    
    return ingested_files

def process_to_silver(**context):
    """Process Bronze data to Silver layer with validation and deduplication"""
    import boto3
    import pandas as pd
    from pyspark.sql import SparkSession
    
    # Get bronze files from previous task
    bronze_files = context['task_instance'].xcom_pull(
        task_ids='ingest_bronze',
        key='bronze_files'
    )
    
    spark = SparkSession.builder \
        .appName("BEV_Silver_Processing") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
    
    processed_files = []
    
    for file_path in bronze_files:
        try:
            # Read from Bronze
            df = spark.read.json(f"s3a://bev-data-lake/{file_path}")
            
            # Data quality checks
            df = df.filter(df.raw_data.isNotNull())
            
            # Deduplication based on content hash
            df = df.withColumn(
                "content_hash",
                hash(df.raw_data.cast("string"))
            ).dropDuplicates(["content_hash"])
            
            # Schema validation
            required_fields = ['source', 'ingestion_timestamp', 'raw_data']
            for field in required_fields:
                if field not in df.columns:
                    raise ValueError(f"Missing required field: {field}")
            
            # Enrich with calculated fields
            df = df.withColumn("processing_timestamp", current_timestamp())
            df = df.withColumn("data_quality_score", lit(0.95))  # Placeholder
            
            # Partitioned write to Silver
            output_path = file_path.replace('bronze/', 'silver/').replace('.json', '')
            
            df.write \
                .mode("overwrite") \
                .partitionBy("source") \
                .parquet(f"s3a://bev-data-lake/{output_path}")
            
            processed_files.append(output_path)
            print(f"Processed to Silver: {output_path}")
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
    
    spark.stop()
    
    context['task_instance'].xcom_push(key='silver_files', value=processed_files)
    return processed_files

def aggregate_to_gold(**context):
    """Aggregate Silver data to Gold layer for analytics"""
    import boto3
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Get silver files
    silver_files = context['task_instance'].xcom_pull(
        task_ids='process_silver',
        key='silver_files'
    )
    
    s3 = boto3.client('s3')
    
    # Aggregate metrics
    aggregations = {
        'hourly_summary': {},
        'daily_metrics': {},
        'agent_performance': {},
        'security_incidents': {}
    }
    
    for file_path in silver_files:
        try:
            # Read Silver data
            response = s3.get_object(Bucket='bev-data-lake', Key=file_path)
            data = pd.read_parquet(response['Body'])
            
            # Source-specific aggregations
            source = data['source'].iloc[0] if len(data) > 0 else 'unknown'
            
            if source == 'research':
                # Research metrics
                agg = data.groupby('hour').agg({
                    'documents_processed': 'sum',
                    'entities_extracted': 'sum',
                    'quality_score': 'mean'
                })
                aggregations['hourly_summary']['research'] = agg.to_dict()
                
            elif source == 'security':
                # Security metrics
                incidents = data[data['severity'] == 'critical']
                aggregations['security_incidents'] = {
                    'critical_count': len(incidents),
                    'sources': incidents['source'].unique().tolist(),
                    'timestamp': datetime.utcnow().isoformat()
                }
                
            elif source == 'agents':
                # Agent performance
                perf = data.groupby('agent_id').agg({
                    'tasks_completed': 'sum',
                    'avg_latency': 'mean',
                    'error_rate': 'mean'
                })
                aggregations['agent_performance'] = perf.to_dict()
            
        except Exception as e:
            print(f"Failed to aggregate {file_path}: {e}")
    
    # Write Gold layer aggregations
    gold_path = f"gold/aggregations/date={datetime.utcnow().strftime('%Y%m%d')}/summary.json"
    
    s3.put_object(
        Bucket='bev-data-lake',
        Key=gold_path,
        Body=json.dumps(aggregations, default=str),
        ContentType='application/json'
    )
    
    print(f"Gold layer updated: {gold_path}")
    
    return gold_path

def data_quality_validation(**context):
    """Validate data quality across all layers"""
    quality_checks = {
        'completeness': 0,
        'uniqueness': 0,
        'consistency': 0,
        'validity': 0,
        'accuracy': 0,
        'timeliness': 0
    }
    
    # Run quality checks
    # Implementation here...
    
    # Calculate overall score
    overall_score = np.mean(list(quality_checks.values()))
    
    if overall_score < 0.8:
        raise ValueError(f"Data quality below threshold: {overall_score}")
    
    return quality_checks

# Define tasks
ingest_bronze = PythonOperator(
    task_id='ingest_bronze',
    python_callable=ingest_to_bronze,
    dag=dag
)

process_silver = PythonOperator(
    task_id='process_silver',
    python_callable=process_to_silver,
    dag=dag
)

aggregate_gold = PythonOperator(
    task_id='aggregate_gold',
    python_callable=aggregate_to_gold,
    dag=dag
)

quality_check = PythonOperator(
    task_id='quality_validation',
    python_callable=data_quality_validation,
    dag=dag
)

# Task dependencies
ingest_bronze >> process_silver >> aggregate_gold >> quality_check
