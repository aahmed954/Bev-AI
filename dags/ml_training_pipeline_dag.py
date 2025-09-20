"""
ML Model Training Pipeline DAG
Automated model training, evaluation, and deployment
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
import mlflow
import numpy as np
from sklearn.model_selection import cross_val_score
import json

default_args = {
    'owner': 'bev-ml',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=10)
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Automated ML model training and deployment',
    schedule_interval='@weekly',
    catchup=False,
    tags=['ml', 'training', 'deployment']
)

def prepare_training_data(**context):
    """Prepare training data from various sources"""
    import pandas as pd
    import boto3
    from sklearn.model_selection import train_test_split
    
    # Collect training data
    data_sources = [
        's3://bev-data-lake/gold/training/research_embeddings.parquet',
        's3://bev-data-lake/gold/training/agent_interactions.parquet',
        's3://bev-data-lake/gold/training/code_generation_samples.parquet'
    ]
    
    combined_data = []
    
    for source in data_sources:
        df = pd.read_parquet(source)
        combined_data.append(df)
    
    # Combine and preprocess
    full_dataset = pd.concat(combined_data, ignore_index=True)
    
    # Feature engineering
    full_dataset['text_length'] = full_dataset['text'].str.len()
    full_dataset['complexity_score'] = full_dataset['tokens'].apply(lambda x: len(x) / 100)
    
    # Split data
    X = full_dataset.drop(['target'], axis=1)
    y = full_dataset['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Save preprocessed data
    train_path = '/tmp/train_data.parquet'
    test_path = '/tmp/test_data.parquet'
    
    pd.DataFrame(X_train).to_parquet(train_path)
    pd.DataFrame(X_test).to_parquet(test_path)
    
    context['task_instance'].xcom_push(key='train_path', value=train_path)
    context['task_instance'].xcom_push(key='test_path', value=test_path)
    
    return {'train_size': len(X_train), 'test_size': len(X_test)}

def train_models(**context):
    """Train multiple models in parallel"""
    import joblib
    from sklearn.ensemble import RandomForestClassifier, XGBClassifier
    from sklearn.neural_network import MLPClassifier
    import torch
    import torch.nn as nn
    
    # Get data paths
    train_path = context['task_instance'].xcom_pull(
        task_ids='prepare_data',
        key='train_path'
    )
    
    # Load data
    X_train = pd.read_parquet(train_path)
    y_train = pd.read_parquet(train_path.replace('train', 'labels'))
    
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1
        ),
        'xgboost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        ),
        'neural_network': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=1000
        )
    }
    
    trained_models = {}
    
    # MLflow tracking
    mlflow.set_experiment("bev_model_training")
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"train_{model_name}"):
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            # Log metrics
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            
            # Save model
            model_path = f"/tmp/model_{model_name}.pkl"
            joblib.dump(model, model_path)
            
            # Log model
            mlflow.sklearn.log_model(model, model_name)
            
            trained_models[model_name] = {
                'path': model_path,
                'cv_score': cv_scores.mean()
            }
    
    context['task_instance'].xcom_push(key='trained_models', value=trained_models)
    
    return trained_models

def evaluate_models(**context):
    """Evaluate all trained models"""
    import joblib
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    # Get model info
    trained_models = context['task_instance'].xcom_pull(
        task_ids='train_models',
        key='trained_models'
    )
    
    test_path = context['task_instance'].xcom_pull(
        task_ids='prepare_data',
        key='test_path'
    )
    
    # Load test data
    X_test = pd.read_parquet(test_path)
    y_test = pd.read_parquet(test_path.replace('test', 'labels'))
    
    evaluation_results = {}
    
    for model_name, model_info in trained_models.items():
        # Load model
        model = joblib.load(model_info['path'])
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        evaluation_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_score': model_info['cv_score']
        }
        
        # Log to MLflow
        with mlflow.start_run(run_name=f"eval_{model_name}"):
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1", f1)
    
    # Select best model
    best_model = max(evaluation_results.items(), 
                    key=lambda x: x[1]['f1'])
    
    context['task_instance'].xcom_push(key='best_model', value=best_model[0])
    context['task_instance'].xcom_push(key='evaluation_results', value=evaluation_results)
    
    return evaluation_results

def deploy_best_model(**context):
    """Deploy the best performing model"""
    import docker
    import boto3
    
    best_model_name = context['task_instance'].xcom_pull(
        task_ids='evaluate_models',
        key='best_model'
    )
    
    trained_models = context['task_instance'].xcom_pull(
        task_ids='train_models',
        key='trained_models'
    )
    
    model_path = trained_models[best_model_name]['path']
    
    # Create model serving container
    client = docker.from_env()
    
    # Build Docker image for model serving
    dockerfile = f"""
    FROM python:3.9-slim
    
    WORKDIR /app
    
    COPY {model_path} /app/model.pkl
    
    RUN pip install fastapi uvicorn scikit-learn joblib
    
    COPY model_server.py /app/
    
    CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8000"]
    """
    
    # Build and push to registry
    image_name = f"bev-models/{best_model_name}:latest"
    
    client.images.build(
        path=".",
        tag=image_name,
        dockerfile=dockerfile
    )
    
    # Deploy to Kubernetes or Docker Swarm
    deployment_config = {
        'model': best_model_name,
        'image': image_name,
        'replicas': 3,
        'resources': {
            'cpu': '1000m',
            'memory': '2Gi'
        }
    }
    
    # Update model registry
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket='bev-models',
        Key=f'deployments/{best_model_name}/config.json',
        Body=json.dumps(deployment_config)
    )
    
    print(f"Deployed {best_model_name} successfully")
    
    return deployment_config

# Define tasks
prepare_data = PythonOperator(
    task_id='prepare_data',
    python_callable=prepare_training_data,
    dag=dag
)

train = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag
)

evaluate = PythonOperator(
    task_id='evaluate_models',
    python_callable=evaluate_models,
    dag=dag
)

deploy = PythonOperator(
    task_id='deploy_best_model',
    python_callable=deploy_best_model,
    dag=dag
)

# Task dependencies
prepare_data >> train >> evaluate >> deploy
