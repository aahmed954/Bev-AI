"""
Cost Optimization DAG
Monitor and optimize resource usage across the swarm
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
import pandas as pd
import numpy as np

default_args = {
    'owner': 'bev-finance',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['admin@bev.ai'],
    'email_on_failure': True,
    'retries': 1
}

dag = DAG(
    'cost_optimization',
    default_args=default_args,
    description='Monitor and optimize infrastructure costs',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    tags=['cost', 'optimization', 'monitoring']
)

def analyze_resource_usage(**context):
    """Analyze resource usage across all services"""
    import boto3
    from prometheus_client import CollectorRegistry, Gauge
    import requests
    
    # Collect metrics from various sources
    metrics = {
        'compute': {},
        'storage': {},
        'network': {},
        'api_calls': {}
    }
    
    # Prometheus metrics
    prom_response = requests.get('http://prometheus:9090/api/v1/query', params={
        'query': 'sum(rate(container_cpu_usage_seconds_total[5m])) by (container_name)'
    })
    
    cpu_usage = prom_response.json()['data']['result']
    
    for result in cpu_usage:
        container = result['metric']['container_name']
        usage = float(result['value'][1])
        metrics['compute'][container] = {
            'cpu_usage': usage,
            'cost_per_hour': usage * 0.05  # $0.05 per CPU hour
        }
    
    # Storage costs
    s3 = boto3.client('s3')
    
    for bucket_name in ['bev-data-lake', 'bev-models', 'bev-artifacts']:
        response = s3.list_objects_v2(Bucket=bucket_name)
        total_size = sum(obj['Size'] for obj in response.get('Contents', []))
        
        metrics['storage'][bucket_name] = {
            'size_gb': total_size / (1024**3),
            'cost_per_month': (total_size / (1024**3)) * 0.023  # $0.023 per GB
        }
    
    # API costs
    api_logs = pd.read_parquet('s3://bev-data-lake/gold/api_usage/latest.parquet')
    
    for model in ['gpt-4', 'claude-sonnet', 'llama-3']:
        model_usage = api_logs[api_logs['model'] == model]
        
        metrics['api_calls'][model] = {
            'total_calls': len(model_usage),
            'total_tokens': model_usage['tokens'].sum(),
            'total_cost': model_usage['cost'].sum()
        }
    
    # Calculate total costs
    total_compute = sum(m['cost_per_hour'] * 24 for m in metrics['compute'].values())
    total_storage = sum(m['cost_per_month'] / 30 for m in metrics['storage'].values())
    total_api = sum(m['total_cost'] for m in metrics['api_calls'].values())
    
    metrics['summary'] = {
        'daily_compute_cost': total_compute,
        'daily_storage_cost': total_storage,
        'daily_api_cost': total_api,
        'total_daily_cost': total_compute + total_storage + total_api
    }
    
    context['task_instance'].xcom_push(key='usage_metrics', value=metrics)
    
    return metrics

def identify_optimization_opportunities(**context):
    """Identify cost optimization opportunities"""
    metrics = context['task_instance'].xcom_pull(
        task_ids='analyze_usage',
        key='usage_metrics'
    )
    
    recommendations = []
    
    # Compute optimizations
    for container, usage in metrics['compute'].items():
        if usage['cpu_usage'] < 0.1:  # Less than 10% utilization
            recommendations.append({
                'type': 'compute',
                'resource': container,
                'issue': 'underutilized',
                'recommendation': 'Consider scaling down or using spot instances',
                'potential_savings': usage['cost_per_hour'] * 0.7
            })
    
    # Storage optimizations
    for bucket, storage in metrics['storage'].items():
        if 'archive' not in bucket and storage['size_gb'] > 100:
            recommendations.append({
                'type': 'storage',
                'resource': bucket,
                'issue': 'large_cold_storage',
                'recommendation': 'Move older data to Glacier',
                'potential_savings': storage['cost_per_month'] * 0.8
            })
    
    # API optimizations
    for model, usage in metrics['api_calls'].items():
        if model == 'gpt-4' and usage['total_calls'] > 1000:
            cheaper_cost = usage['total_cost'] * 0.1  # Claude is 10x cheaper
            
            recommendations.append({
                'type': 'api',
                'resource': model,
                'issue': 'expensive_model_overuse',
                'recommendation': 'Route simpler queries to Claude Sonnet or Llama-3',
                'potential_savings': usage['total_cost'] - cheaper_cost
            })
    
    # Calculate total potential savings
    total_savings = sum(r['potential_savings'] for r in recommendations)
    
    optimization_report = {
        'recommendations': recommendations,
        'total_potential_savings': total_savings,
        'current_daily_cost': metrics['summary']['total_daily_cost'],
        'optimized_daily_cost': metrics['summary']['total_daily_cost'] - total_savings,
        'savings_percentage': (total_savings / metrics['summary']['total_daily_cost']) * 100
    }
    
    context['task_instance'].xcom_push(key='optimization_report', value=optimization_report)
    
    return optimization_report

def implement_auto_scaling(**context):
    """Automatically implement scaling recommendations"""
    import docker
    import kubernetes
    
    report = context['task_instance'].xcom_pull(
        task_ids='identify_optimizations',
        key='optimization_report'
    )
    
    implemented = []
    
    for recommendation in report['recommendations']:
        if recommendation['type'] == 'compute' and recommendation['issue'] == 'underutilized':
            # Scale down container
            container_name = recommendation['resource']
            
            # Kubernetes scaling
            if 'k8s' in container_name:
                # kubectl scale deployment
                pass
            
            # Docker Swarm scaling
            else:
                client = docker.from_env()
                service = client.services.get(container_name)
                
                # Scale down by 50%
                current_replicas = service.attrs['Spec']['Mode']['Replicated']['Replicas']
                new_replicas = max(1, current_replicas // 2)
                
                service.update(replicas=new_replicas)
                
                implemented.append({
                    'action': 'scaled_down',
                    'resource': container_name,
                    'from': current_replicas,
                    'to': new_replicas
                })
    
    return implemented

def generate_cost_report(**context):
    """Generate comprehensive cost report"""
    metrics = context['task_instance'].xcom_pull(
        task_ids='analyze_usage',
        key='usage_metrics'
    )
    
    report = context['task_instance'].xcom_pull(
        task_ids='identify_optimizations',
        key='optimization_report'
    )
    
    # Create detailed report
    html_report = f"""
    <html>
    <head><title>BEV Cost Report</title></head>
    <body>
        <h1>Daily Cost Report - {datetime.now().strftime('%Y-%m-%d')}</h1>
        
        <h2>Current Costs</h2>
        <ul>
            <li>Compute: ${metrics['summary']['daily_compute_cost']:.2f}</li>
            <li>Storage: ${metrics['summary']['daily_storage_cost']:.2f}</li>
            <li>API Calls: ${metrics['summary']['daily_api_cost']:.2f}</li>
            <li><strong>Total: ${metrics['summary']['total_daily_cost']:.2f}</strong></li>
        </ul>
        
        <h2>Optimization Opportunities</h2>
        <p>Potential Savings: ${report['total_potential_savings']:.2f} ({report['savings_percentage']:.1f}%)</p>
        
        <h3>Recommendations:</h3>
        <ol>
        {''.join(f"<li>{r['recommendation']} - Save ${r['potential_savings']:.2f}</li>" for r in report['recommendations'])}
        </ol>
        
        <h2>Projected Monthly Cost</h2>
        <ul>
            <li>Current: ${metrics['summary']['total_daily_cost'] * 30:.2f}</li>
            <li>Optimized: ${report['optimized_daily_cost'] * 30:.2f}</li>
            <li>Savings: ${report['total_potential_savings'] * 30:.2f}</li>
        </ul>
    </body>
    </html>
    """
    
    # Save report
    with open('/tmp/cost_report.html', 'w') as f:
        f.write(html_report)
    
    return html_report

# Task definitions
analyze_usage = PythonOperator(
    task_id='analyze_usage',
    python_callable=analyze_resource_usage,
    dag=dag
)

identify_optimizations = PythonOperator(
    task_id='identify_optimizations',
    python_callable=identify_optimization_opportunities,
    dag=dag
)

auto_scale = PythonOperator(
    task_id='implement_auto_scaling',
    python_callable=implement_auto_scaling,
    dag=dag
)

generate_report = PythonOperator(
    task_id='generate_cost_report',
    python_callable=generate_cost_report,
    dag=dag
)

send_report = EmailOperator(
    task_id='send_report',
    to=['admin@bev.ai'],
    subject='BEV Daily Cost Report',
    html_content="{{ task_instance.xcom_pull(task_ids='generate_cost_report') }}",
    dag=dag
)

# Task dependencies
analyze_usage >> identify_optimizations >> [auto_scale, generate_report] >> send_report
