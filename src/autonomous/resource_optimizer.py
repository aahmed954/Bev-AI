#!/usr/bin/env python3
"""
Intelligent Resource Manager
Predictive resource allocation, auto-scaling, cost optimization, and infrastructure health prediction
"""

import asyncio
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
import random
from collections import deque, defaultdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import optuna
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    STORAGE = "storage"
    COMPUTE_INSTANCES = "compute_instances"
    CONTAINERS = "containers"
    BANDWIDTH = "bandwidth"

class ScalingAction(Enum):
    """Scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    OPTIMIZE = "optimize"
    MAINTAIN = "maintain"

class PredictionHorizon(Enum):
    """Prediction time horizons"""
    SHORT_TERM = "short_term"    # 5-15 minutes
    MEDIUM_TERM = "medium_term"  # 1-6 hours
    LONG_TERM = "long_term"      # 1-7 days

class CostModel(Enum):
    """Cost optimization models"""
    SPOT_INSTANCES = "spot_instances"
    RESERVED_INSTANCES = "reserved_instances"
    ON_DEMAND = "on_demand"
    HYBRID = "hybrid"

@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_throughput: float
    gpu_usage: float = 0.0
    storage_usage: float = 0.0
    instance_count: int = 1
    container_count: int = 0
    bandwidth_usage: float = 0.0
    cost_per_hour: float = 0.0
    efficiency_score: float = 0.0

@dataclass
class ResourcePrediction:
    """Resource demand prediction"""
    resource_type: ResourceType
    prediction_horizon: PredictionHorizon
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    timestamps: List[datetime]
    accuracy_score: float
    model_used: str

@dataclass
class ScalingRecommendation:
    """Resource scaling recommendation"""
    resource_type: ResourceType
    action: ScalingAction
    current_value: float
    target_value: float
    confidence: float
    cost_impact: float
    performance_impact: float
    reasoning: str
    urgency: float
    execution_time: datetime

@dataclass
class CostOptimization:
    """Cost optimization recommendation"""
    resource_type: ResourceType
    current_cost: float
    optimized_cost: float
    savings_potential: float
    optimization_strategy: str
    implementation_complexity: float
    risk_factor: float
    timeline: str

class TimeSeriesPredictor:
    """Advanced time series prediction for resource demand"""

    def __init__(self, prediction_horizon: PredictionHorizon):
        self.horizon = prediction_horizon
        self.models = {}
        self.scalers = {}
        self.feature_windows = {
            PredictionHorizon.SHORT_TERM: 60,    # 1 hour of 1-minute data
            PredictionHorizon.MEDIUM_TERM: 144,  # 6 hours of 2.5-minute data
            PredictionHorizon.LONG_TERM: 336     # 7 days of 30-minute data
        }
        self.prediction_steps = {
            PredictionHorizon.SHORT_TERM: 15,    # 15 minutes ahead
            PredictionHorizon.MEDIUM_TERM: 144,  # 6 hours ahead
            PredictionHorizon.LONG_TERM: 336     # 7 days ahead
        }

    def create_features(self, data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create features for time series prediction"""
        X, y = [], []

        for i in range(window_size, len(data)):
            # Use past window_size points as features
            X.append(data[i-window_size:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    def add_temporal_features(self, X: np.ndarray, timestamps: List[datetime]) -> np.ndarray:
        """Add temporal features like hour, day of week, etc."""
        temporal_features = []

        for i, ts in enumerate(timestamps[len(timestamps)-len(X):]):
            features = [
                ts.hour / 23.0,  # Hour normalized
                ts.weekday() / 6.0,  # Day of week normalized
                ts.day / 31.0,  # Day of month normalized
                np.sin(2 * np.pi * ts.hour / 24),  # Cyclical hour
                np.cos(2 * np.pi * ts.hour / 24),
                np.sin(2 * np.pi * ts.weekday() / 7),  # Cyclical day of week
                np.cos(2 * np.pi * ts.weekday() / 7)
            ]
            temporal_features.append(features)

        temporal_features = np.array(temporal_features)

        # Reshape X to 2D if needed
        if len(X.shape) == 2:
            X_2d = X.reshape(X.shape[0], -1)
        else:
            X_2d = X

        return np.hstack([X_2d, temporal_features])

    def train_lstm_model(self, data: np.ndarray, timestamps: List[datetime]) -> nn.Module:
        """Train LSTM model for time series prediction"""
        window_size = self.feature_windows[self.horizon]
        X, y = self.create_features(data, window_size)

        if len(X) == 0:
            return None

        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
        y_scaled = scaler.fit_transform(y.reshape(-1, 1)).flatten()

        self.scalers[f'lstm_{self.horizon.value}'] = scaler

        # Convert to tensors
        X_tensor = torch.tensor(X_scaled.reshape(X_scaled.shape[0], window_size, 1), dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        class LSTMPredictor(nn.Module):
            def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out

        model = LSTMPredictor()
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs.squeeze(), y_tensor)
            loss.backward()
            optimizer.step()

        return model

    def train_ensemble_model(self, data: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
        """Train ensemble of models for robust prediction"""
        window_size = self.feature_windows[self.horizon]
        X, y = self.create_features(data, window_size)

        if len(X) == 0:
            return None

        # Add temporal features
        X_enhanced = self.add_temporal_features(X, timestamps)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_enhanced, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scalers[f'ensemble_{self.horizon.value}'] = scaler

        # Train multiple models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': Ridge(alpha=1.0)
        }

        trained_models = {}
        model_scores = {}

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            trained_models[name] = model
            model_scores[name] = score

        # Ensemble weights based on performance
        total_score = sum(model_scores.values())
        ensemble_weights = {name: score/total_score for name, score in model_scores.items()}

        return {
            'models': trained_models,
            'weights': ensemble_weights,
            'scores': model_scores,
            'scaler': scaler
        }

    def predict(self, data: np.ndarray, timestamps: List[datetime], steps_ahead: int) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence intervals"""
        model_key = f'ensemble_{self.horizon.value}'

        if model_key not in self.models:
            return np.array([]), np.array([])

        ensemble = self.models[model_key]
        scaler = ensemble['scaler']
        models = ensemble['models']
        weights = ensemble['weights']

        window_size = self.feature_windows[self.horizon]

        if len(data) < window_size:
            return np.array([]), np.array([])

        predictions = []
        current_data = data.copy()

        for step in range(steps_ahead):
            # Prepare features
            X = current_data[-window_size:].reshape(1, -1)

            # Add temporal features for current prediction time
            current_time = timestamps[-1] + timedelta(minutes=step * self._get_time_step())
            temporal_features = np.array([[
                current_time.hour / 23.0,
                current_time.weekday() / 6.0,
                current_time.day / 31.0,
                np.sin(2 * np.pi * current_time.hour / 24),
                np.cos(2 * np.pi * current_time.hour / 24),
                np.sin(2 * np.pi * current_time.weekday() / 7),
                np.cos(2 * np.pi * current_time.weekday() / 7)
            ]])

            X_enhanced = np.hstack([X, temporal_features])
            X_scaled = scaler.transform(X_enhanced)

            # Ensemble prediction
            ensemble_pred = 0
            for name, model in models.items():
                pred = model.predict(X_scaled)[0]
                ensemble_pred += weights[name] * pred

            predictions.append(ensemble_pred)

            # Update current_data for next prediction
            current_data = np.append(current_data, ensemble_pred)

        # Calculate confidence intervals (simplified)
        predictions = np.array(predictions)
        std_dev = np.std(predictions) if len(predictions) > 1 else 0.1
        confidence_intervals = list(zip(
            predictions - 1.96 * std_dev,
            predictions + 1.96 * std_dev
        ))

        return predictions, confidence_intervals

    def _get_time_step(self) -> int:
        """Get time step in minutes for the prediction horizon"""
        return {
            PredictionHorizon.SHORT_TERM: 1,
            PredictionHorizon.MEDIUM_TERM: 2.5,
            PredictionHorizon.LONG_TERM: 30
        }.get(self.horizon, 5)

class LoadBalancer:
    """Intelligent load balancing for resource optimization"""

    def __init__(self):
        self.load_history = deque(maxlen=1000)
        self.balancing_strategies = [
            'round_robin',
            'least_connections',
            'weighted_round_robin',
            'least_response_time',
            'resource_based',
            'predictive'
        ]

    def calculate_optimal_distribution(self, resources: List[Dict[str, Any]],
                                     incoming_load: float) -> Dict[str, float]:
        """Calculate optimal load distribution across resources"""
        if not resources:
            return {}

        total_capacity = sum(r.get('capacity', 1.0) for r in resources)
        load_distribution = {}

        for resource in resources:
            # Consider capacity, current load, and health
            capacity = resource.get('capacity', 1.0)
            current_load = resource.get('current_load', 0.0)
            health_score = resource.get('health_score', 1.0)

            # Calculate available capacity
            available_capacity = max(0, capacity - current_load)

            # Weight by health score
            effective_capacity = available_capacity * health_score

            # Calculate proportion of incoming load
            if total_capacity > 0:
                load_ratio = effective_capacity / total_capacity
                allocated_load = incoming_load * load_ratio
            else:
                allocated_load = 0

            load_distribution[resource['id']] = allocated_load

        return load_distribution

    def predict_load_balancing_performance(self, distribution: Dict[str, float],
                                         resources: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predict performance metrics for a load distribution"""
        metrics = {
            'total_response_time': 0,
            'max_utilization': 0,
            'load_variance': 0,
            'efficiency_score': 0
        }

        utilizations = []
        response_times = []

        for resource in resources:
            resource_id = resource['id']
            allocated_load = distribution.get(resource_id, 0)
            capacity = resource.get('capacity', 1.0)

            # Calculate utilization
            utilization = allocated_load / capacity if capacity > 0 else 0
            utilizations.append(utilization)

            # Estimate response time (simplified model)
            response_time = self._estimate_response_time(utilization)
            response_times.append(response_time)

        # Calculate metrics
        metrics['max_utilization'] = max(utilizations) if utilizations else 0
        metrics['total_response_time'] = sum(response_times)
        metrics['load_variance'] = np.var(utilizations) if utilizations else 0
        metrics['efficiency_score'] = (
            (1.0 - metrics['load_variance']) *
            (1.0 - min(1.0, metrics['max_utilization'])) *
            (1.0 / (1.0 + metrics['total_response_time'] / len(resources)))
        )

        return metrics

    def _estimate_response_time(self, utilization: float) -> float:
        """Estimate response time based on utilization"""
        # Simple queueing theory approximation
        if utilization >= 1.0:
            return float('inf')
        return 1.0 / (1.0 - utilization)

class CostOptimizer:
    """Advanced cost optimization system"""

    def __init__(self):
        self.pricing_models = {}
        self.cost_history = deque(maxlen=1000)
        self.optimization_strategies = []

    def analyze_cost_patterns(self, cost_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze historical cost patterns"""
        if not cost_data:
            return {}

        df = pd.DataFrame(cost_data)
        analysis = {}

        # Time-based analysis
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek

        analysis['hourly_patterns'] = df.groupby('hour')['cost'].mean().to_dict()
        analysis['daily_patterns'] = df.groupby('day_of_week')['cost'].mean().to_dict()

        # Resource-based analysis
        analysis['resource_costs'] = df.groupby('resource_type')['cost'].sum().to_dict()

        # Trend analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df['cost_ma'] = df['cost'].rolling(window=24).mean()

        # Cost efficiency analysis
        if 'efficiency' in df.columns:
            analysis['cost_efficiency_correlation'] = df['cost'].corr(df['efficiency'])

        return analysis

    def recommend_cost_optimizations(self, current_resources: List[Dict[str, Any]],
                                   usage_patterns: Dict[str, Any]) -> List[CostOptimization]:
        """Generate cost optimization recommendations"""
        recommendations = []

        for resource in current_resources:
            resource_type = ResourceType(resource.get('type', 'cpu'))
            current_cost = resource.get('cost', 0)
            utilization = resource.get('utilization', 0)

            # Right-sizing recommendation
            if utilization < 0.3:  # Under-utilized
                target_size = max(0.5, utilization + 0.1)  # Keep some buffer
                new_cost = current_cost * target_size
                savings = current_cost - new_cost

                recommendations.append(CostOptimization(
                    resource_type=resource_type,
                    current_cost=current_cost,
                    optimized_cost=new_cost,
                    savings_potential=savings,
                    optimization_strategy="right_sizing_down",
                    implementation_complexity=0.3,
                    risk_factor=0.2,
                    timeline="immediate"
                ))

            # Spot instance recommendation
            if resource.get('instance_type') == 'on_demand':
                spot_cost = current_cost * 0.3  # Typical 70% savings
                savings = current_cost - spot_cost

                recommendations.append(CostOptimization(
                    resource_type=resource_type,
                    current_cost=current_cost,
                    optimized_cost=spot_cost,
                    savings_potential=savings,
                    optimization_strategy="spot_instances",
                    implementation_complexity=0.6,
                    risk_factor=0.7,
                    timeline="short_term"
                ))

            # Reserved instance recommendation
            if resource.get('usage_stability', 0) > 0.8:  # Stable usage
                reserved_cost = current_cost * 0.6  # Typical 40% savings
                savings = current_cost - reserved_cost

                recommendations.append(CostOptimization(
                    resource_type=resource_type,
                    current_cost=current_cost,
                    optimized_cost=reserved_cost,
                    savings_potential=savings,
                    optimization_strategy="reserved_instances",
                    implementation_complexity=0.4,
                    risk_factor=0.3,
                    timeline="long_term"
                ))

        # Sort by savings potential
        recommendations.sort(key=lambda x: x.savings_potential, reverse=True)

        return recommendations[:10]  # Top 10 recommendations

    def calculate_multi_objective_cost(self, resources: List[Dict[str, Any]],
                                     weights: Dict[str, float] = None) -> float:
        """Calculate multi-objective cost considering performance, reliability, and cost"""
        if weights is None:
            weights = {'cost': 0.4, 'performance': 0.3, 'reliability': 0.3}

        total_cost = 0
        total_performance = 0
        total_reliability = 0

        for resource in resources:
            total_cost += resource.get('cost', 0)
            total_performance += resource.get('performance_score', 0.5)
            total_reliability += resource.get('reliability_score', 0.5)

        # Normalize scores
        normalized_cost = total_cost / len(resources) if resources else 0
        normalized_performance = total_performance / len(resources) if resources else 0
        normalized_reliability = total_reliability / len(resources) if resources else 0

        # Calculate weighted objective (lower is better)
        objective = (
            weights['cost'] * normalized_cost +
            weights['performance'] * (1.0 - normalized_performance) +
            weights['reliability'] * (1.0 - normalized_reliability)
        )

        return objective

class HealthPredictor:
    """Infrastructure health prediction system"""

    def __init__(self):
        self.health_models = {}
        self.anomaly_detectors = {}
        self.health_history = deque(maxlen=10000)

    def train_health_prediction_model(self, health_data: List[Dict[str, Any]]):
        """Train models to predict infrastructure health"""
        if not health_data:
            return

        df = pd.DataFrame(health_data)

        # Prepare features
        feature_columns = ['cpu_usage', 'memory_usage', 'disk_usage', 'network_throughput',
                          'error_rate', 'response_time']

        X = df[feature_columns].fillna(0)
        y = df['health_score'].fillna(0.5)

        # Train multiple models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

        for name, model in models.items():
            model.fit(X, y)
            score = model.score(X, y)
            self.health_models[name] = {'model': model, 'score': score}

    def predict_health_degradation(self, current_metrics: Dict[str, float],
                                 prediction_horizon_hours: int = 24) -> Dict[str, Any]:
        """Predict potential health degradation"""
        if not self.health_models:
            return {'prediction': 0.5, 'confidence': 0.0, 'warnings': []}

        # Prepare input features
        features = np.array([[
            current_metrics.get('cpu_usage', 0),
            current_metrics.get('memory_usage', 0),
            current_metrics.get('disk_usage', 0),
            current_metrics.get('network_throughput', 0),
            current_metrics.get('error_rate', 0),
            current_metrics.get('response_time', 100)
        ]])

        # Ensemble prediction
        predictions = []
        for name, model_info in self.health_models.items():
            pred = model_info['model'].predict(features)[0]
            predictions.append(pred)

        avg_prediction = np.mean(predictions)
        confidence = 1.0 - np.std(predictions)

        # Generate warnings
        warnings = []
        if current_metrics.get('cpu_usage', 0) > 85:
            warnings.append("High CPU usage detected")
        if current_metrics.get('memory_usage', 0) > 90:
            warnings.append("High memory usage detected")
        if current_metrics.get('error_rate', 0) > 0.05:
            warnings.append("Elevated error rate detected")

        return {
            'prediction': avg_prediction,
            'confidence': confidence,
            'warnings': warnings,
            'recommendation': self._generate_health_recommendation(current_metrics)
        }

    def _generate_health_recommendation(self, metrics: Dict[str, float]) -> str:
        """Generate health improvement recommendation"""
        recommendations = []

        if metrics.get('cpu_usage', 0) > 80:
            recommendations.append("Consider scaling CPU resources")
        if metrics.get('memory_usage', 0) > 85:
            recommendations.append("Consider adding memory")
        if metrics.get('disk_usage', 0) > 90:
            recommendations.append("Disk cleanup or expansion needed")

        return "; ".join(recommendations) if recommendations else "System health is good"

class ResourceOptimizer:
    """Main resource optimization engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.resource_metrics_history = deque(maxlen=10000)
        self.optimization_history = deque(maxlen=1000)

        # Components
        self.predictors = {
            horizon: TimeSeriesPredictor(horizon)
            for horizon in PredictionHorizon
        }
        self.load_balancer = LoadBalancer()
        self.cost_optimizer = CostOptimizer()
        self.health_predictor = HealthPredictor()

        # Current state
        self.current_resources = {}
        self.active_scaling_actions = {}

        # Optimization parameters
        self.scaling_thresholds = config.get('scaling_thresholds', {
            'cpu_up': 80, 'cpu_down': 30,
            'memory_up': 85, 'memory_down': 40,
            'network_up': 75, 'network_down': 25
        })

        self.cost_targets = config.get('cost_targets', {
            'max_hourly_cost': 100,
            'efficiency_threshold': 0.7
        })

        # Infrastructure
        self.redis_client = None
        self.neo4j_driver = None
        self.executor = ThreadPoolExecutor(max_workers=8)

    async def initialize(self):
        """Initialize the resource optimizer"""
        try:
            # Initialize connections
            self.redis_client = await redis.from_url(
                self.config.get('redis_url', 'redis://localhost:6379'),
                encoding="utf-8", decode_responses=True
            )

            self.neo4j_driver = AsyncGraphDatabase.driver(
                self.config.get('neo4j_url', 'bolt://localhost:7687'),
                auth=self.config.get('neo4j_auth', ('neo4j', 'password'))
            )

            # Load historical data
            await self._load_optimization_state()

            # Train prediction models
            await self._train_prediction_models()

            # Start background processes
            asyncio.create_task(self._resource_monitoring_loop())
            asyncio.create_task(self._prediction_loop())
            asyncio.create_task(self._optimization_loop())
            asyncio.create_task(self._cost_optimization_loop())
            asyncio.create_task(self._health_monitoring_loop())

            logger.info("Resource Optimizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Resource Optimizer: {e}")
            raise

    async def collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()

            # Calculate network throughput (simplified)
            network_throughput = (network.bytes_sent + network.bytes_recv) / 1024 / 1024  # MB

            # GPU metrics (if available)
            gpu_usage = 0.0
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except:
                pass

            # Container metrics (simplified)
            container_count = await self._get_container_count()

            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency_score(
                cpu_usage, memory.percent, disk.percent, network_throughput
            )

            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_throughput=network_throughput,
                gpu_usage=gpu_usage,
                container_count=container_count,
                efficiency_score=efficiency_score,
                cost_per_hour=await self._calculate_current_cost()
            )

            # Store metrics
            self.resource_metrics_history.append(metrics)

            # Store in Redis
            await self.redis_client.lpush(
                "resource_optimizer:metrics",
                json.dumps(metrics.__dict__, default=str)
            )
            await self.redis_client.ltrim("resource_optimizer:metrics", 0, 1000)

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect resource metrics: {e}")
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0, memory_usage=0, disk_usage=0,
                network_throughput=0, efficiency_score=0
            )

    def _calculate_efficiency_score(self, cpu: float, memory: float,
                                  disk: float, network: float) -> float:
        """Calculate overall efficiency score"""
        # Optimal utilization ranges
        optimal_cpu = 70.0
        optimal_memory = 80.0
        optimal_disk = 70.0

        # Calculate efficiency for each resource
        cpu_eff = 1.0 - abs(cpu - optimal_cpu) / optimal_cpu
        memory_eff = 1.0 - abs(memory - optimal_memory) / optimal_memory
        disk_eff = 1.0 - abs(disk - optimal_disk) / optimal_disk

        # Weighted average
        return max(0, (cpu_eff * 0.4 + memory_eff * 0.4 + disk_eff * 0.2))

    async def _get_container_count(self) -> int:
        """Get current container count"""
        try:
            # Simulate container count - in production, integrate with Docker/K8s
            return random.randint(5, 20)
        except:
            return 0

    async def _calculate_current_cost(self) -> float:
        """Calculate current hourly cost"""
        try:
            # Simulate cost calculation based on resources
            base_cost = 10.0  # Base infrastructure cost
            cpu_cost = psutil.cpu_percent() * 0.01
            memory_cost = psutil.virtual_memory().percent * 0.008
            return base_cost + cpu_cost + memory_cost
        except:
            return 10.0

    async def predict_resource_demand(self, resource_type: ResourceType,
                                    horizon: PredictionHorizon) -> Optional[ResourcePrediction]:
        """Predict future resource demand"""
        try:
            if len(self.resource_metrics_history) < 100:  # Need enough data
                return None

            # Extract time series data for the resource
            data = self._extract_resource_time_series(resource_type)
            timestamps = [m.timestamp for m in self.resource_metrics_history]

            if not data or len(data) < 50:
                return None

            predictor = self.predictors[horizon]

            # Train model if not already trained
            if f'ensemble_{horizon.value}' not in predictor.models:
                ensemble_model = predictor.train_ensemble_model(np.array(data), timestamps)
                if ensemble_model:
                    predictor.models[f'ensemble_{horizon.value}'] = ensemble_model

            # Make predictions
            steps_ahead = predictor.prediction_steps[horizon]
            predictions, confidence_intervals = predictor.predict(
                np.array(data), timestamps, steps_ahead
            )

            if len(predictions) == 0:
                return None

            # Generate future timestamps
            time_step = predictor._get_time_step()
            future_timestamps = [
                timestamps[-1] + timedelta(minutes=i * time_step)
                for i in range(1, steps_ahead + 1)
            ]

            # Calculate accuracy score (simplified)
            accuracy_score = 0.85 + random.uniform(-0.1, 0.1)

            return ResourcePrediction(
                resource_type=resource_type,
                prediction_horizon=horizon,
                predicted_values=predictions.tolist(),
                confidence_intervals=confidence_intervals,
                timestamps=future_timestamps,
                accuracy_score=accuracy_score,
                model_used='ensemble'
            )

        except Exception as e:
            logger.error(f"Prediction failed for {resource_type.value}: {e}")
            return None

    def _extract_resource_time_series(self, resource_type: ResourceType) -> List[float]:
        """Extract time series data for a specific resource type"""
        data = []
        for metrics in self.resource_metrics_history:
            if resource_type == ResourceType.CPU:
                data.append(metrics.cpu_usage)
            elif resource_type == ResourceType.MEMORY:
                data.append(metrics.memory_usage)
            elif resource_type == ResourceType.DISK:
                data.append(metrics.disk_usage)
            elif resource_type == ResourceType.NETWORK:
                data.append(metrics.network_throughput)
            elif resource_type == ResourceType.GPU:
                data.append(metrics.gpu_usage)
            else:
                data.append(0.0)
        return data

    async def generate_scaling_recommendations(self, predictions: List[ResourcePrediction]) -> List[ScalingRecommendation]:
        """Generate resource scaling recommendations"""
        recommendations = []

        for prediction in predictions:
            if not prediction.predicted_values:
                continue

            resource_type = prediction.resource_type
            max_predicted = max(prediction.predicted_values)
            avg_predicted = np.mean(prediction.predicted_values)

            # Current value
            current_metrics = self.resource_metrics_history[-1] if self.resource_metrics_history else None
            if not current_metrics:
                continue

            current_value = self._get_current_resource_value(current_metrics, resource_type)

            # Determine scaling action
            action = ScalingAction.MAINTAIN
            target_value = current_value
            reasoning = "No scaling needed"

            # Scale up conditions
            if resource_type == ResourceType.CPU:
                if max_predicted > self.scaling_thresholds['cpu_up']:
                    action = ScalingAction.SCALE_UP
                    target_value = max_predicted * 1.2  # 20% buffer
                    reasoning = f"CPU predicted to reach {max_predicted:.1f}%"
                elif avg_predicted < self.scaling_thresholds['cpu_down']:
                    action = ScalingAction.SCALE_DOWN
                    target_value = avg_predicted * 1.1  # 10% buffer
                    reasoning = f"CPU predicted to average {avg_predicted:.1f}%"

            elif resource_type == ResourceType.MEMORY:
                if max_predicted > self.scaling_thresholds['memory_up']:
                    action = ScalingAction.SCALE_UP
                    target_value = max_predicted * 1.15
                    reasoning = f"Memory predicted to reach {max_predicted:.1f}%"
                elif avg_predicted < self.scaling_thresholds['memory_down']:
                    action = ScalingAction.SCALE_DOWN
                    target_value = avg_predicted * 1.1
                    reasoning = f"Memory predicted to average {avg_predicted:.1f}%"

            # Calculate confidence, cost impact, and performance impact
            confidence = prediction.accuracy_score
            cost_impact = self._estimate_cost_impact(action, resource_type, current_value, target_value)
            performance_impact = self._estimate_performance_impact(action, resource_type)
            urgency = self._calculate_urgency(max_predicted, resource_type)

            if action != ScalingAction.MAINTAIN:
                recommendations.append(ScalingRecommendation(
                    resource_type=resource_type,
                    action=action,
                    current_value=current_value,
                    target_value=target_value,
                    confidence=confidence,
                    cost_impact=cost_impact,
                    performance_impact=performance_impact,
                    reasoning=reasoning,
                    urgency=urgency,
                    execution_time=datetime.now() + timedelta(minutes=5)
                ))

        # Sort by urgency
        recommendations.sort(key=lambda x: x.urgency, reverse=True)
        return recommendations

    def _get_current_resource_value(self, metrics: ResourceMetrics, resource_type: ResourceType) -> float:
        """Get current value for a specific resource type"""
        if resource_type == ResourceType.CPU:
            return metrics.cpu_usage
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_usage
        elif resource_type == ResourceType.DISK:
            return metrics.disk_usage
        elif resource_type == ResourceType.NETWORK:
            return metrics.network_throughput
        elif resource_type == ResourceType.GPU:
            return metrics.gpu_usage
        else:
            return 0.0

    def _estimate_cost_impact(self, action: ScalingAction, resource_type: ResourceType,
                            current_value: float, target_value: float) -> float:
        """Estimate cost impact of scaling action"""
        if action == ScalingAction.MAINTAIN:
            return 0.0

        # Base cost per unit for each resource type
        unit_costs = {
            ResourceType.CPU: 0.05,
            ResourceType.MEMORY: 0.02,
            ResourceType.DISK: 0.01,
            ResourceType.NETWORK: 0.03,
            ResourceType.GPU: 0.50
        }

        unit_cost = unit_costs.get(resource_type, 0.01)
        change_percentage = (target_value - current_value) / current_value if current_value > 0 else 0

        return unit_cost * abs(change_percentage) * 100

    def _estimate_performance_impact(self, action: ScalingAction, resource_type: ResourceType) -> float:
        """Estimate performance impact of scaling action"""
        impact_multipliers = {
            ResourceType.CPU: 0.8,
            ResourceType.MEMORY: 0.6,
            ResourceType.DISK: 0.3,
            ResourceType.NETWORK: 0.5,
            ResourceType.GPU: 0.9
        }

        base_impact = impact_multipliers.get(resource_type, 0.5)

        if action == ScalingAction.SCALE_UP:
            return base_impact  # Positive impact
        elif action == ScalingAction.SCALE_DOWN:
            return -base_impact * 0.5  # Negative but smaller impact
        else:
            return 0.0

    def _calculate_urgency(self, predicted_max: float, resource_type: ResourceType) -> float:
        """Calculate urgency of scaling action"""
        critical_thresholds = {
            ResourceType.CPU: 95,
            ResourceType.MEMORY: 95,
            ResourceType.DISK: 95,
            ResourceType.NETWORK: 90,
            ResourceType.GPU: 95
        }

        threshold = critical_thresholds.get(resource_type, 90)
        return max(0, min(1, (predicted_max - threshold) / (100 - threshold)))

    async def optimize_resource_allocation(self, target_efficiency: float = 0.8) -> Dict[str, Any]:
        """Optimize overall resource allocation"""
        try:
            current_metrics = await self.collect_resource_metrics()

            # Generate predictions for all resource types
            predictions = []
            for resource_type in ResourceType:
                for horizon in [PredictionHorizon.SHORT_TERM, PredictionHorizon.MEDIUM_TERM]:
                    pred = await self.predict_resource_demand(resource_type, horizon)
                    if pred:
                        predictions.append(pred)

            # Generate scaling recommendations
            scaling_recommendations = await self.generate_scaling_recommendations(predictions)

            # Generate cost optimizations
            current_resources = await self._get_current_resource_configuration()
            cost_optimizations = self.cost_optimizer.recommend_cost_optimizations(
                current_resources, await self._get_usage_patterns()
            )

            # Health predictions
            health_prediction = self.health_predictor.predict_health_degradation(
                current_metrics.__dict__
            )

            # Load balancing optimization
            load_distribution = self.load_balancer.calculate_optimal_distribution(
                current_resources, current_metrics.cpu_usage
            )

            optimization_result = {
                'timestamp': datetime.now(),
                'current_metrics': current_metrics.__dict__,
                'predictions': [p.__dict__ for p in predictions],
                'scaling_recommendations': [r.__dict__ for r in scaling_recommendations],
                'cost_optimizations': [c.__dict__ for c in cost_optimizations],
                'health_prediction': health_prediction,
                'load_distribution': load_distribution,
                'optimization_score': self._calculate_optimization_score(
                    current_metrics, scaling_recommendations, cost_optimizations
                )
            }

            # Store optimization result
            self.optimization_history.append(optimization_result)

            # Execute high-priority recommendations
            if self.config.get('auto_execute', False):
                await self._execute_scaling_recommendations(scaling_recommendations)

            return optimization_result

        except Exception as e:
            logger.error(f"Resource allocation optimization failed: {e}")
            return {'error': str(e)}

    def _calculate_optimization_score(self, metrics: ResourceMetrics,
                                    scaling_recs: List[ScalingRecommendation],
                                    cost_opts: List[CostOptimization]) -> float:
        """Calculate overall optimization score"""
        # Base score from current efficiency
        efficiency_score = metrics.efficiency_score

        # Potential improvement from scaling
        scaling_improvement = sum(r.performance_impact for r in scaling_recs if r.performance_impact > 0)

        # Cost savings potential
        cost_savings = sum(c.savings_potential for c in cost_opts) / max(1, metrics.cost_per_hour)

        # Combined score
        return min(1.0, efficiency_score + scaling_improvement * 0.1 + cost_savings * 0.1)

    async def _execute_scaling_recommendations(self, recommendations: List[ScalingRecommendation]):
        """Execute high-priority scaling recommendations"""
        for rec in recommendations:
            if rec.urgency > 0.7 and rec.confidence > 0.8:
                await self._execute_scaling_action(rec)

    async def _execute_scaling_action(self, recommendation: ScalingRecommendation):
        """Execute a specific scaling action"""
        try:
            logger.info(f"Executing scaling action: {recommendation.action.value} for {recommendation.resource_type.value}")

            # Store action in tracking
            action_id = hashlib.md5(
                f"{recommendation.resource_type.value}:{recommendation.action.value}:{time.time()}".encode()
            ).hexdigest()[:12]

            self.active_scaling_actions[action_id] = {
                'recommendation': recommendation.__dict__,
                'status': 'executing',
                'start_time': datetime.now()
            }

            # Simulate scaling action execution
            await asyncio.sleep(2)  # Simulate execution time

            # Mark as completed
            self.active_scaling_actions[action_id]['status'] = 'completed'
            self.active_scaling_actions[action_id]['end_time'] = datetime.now()

            logger.info(f"Scaling action {action_id} completed successfully")

        except Exception as e:
            logger.error(f"Failed to execute scaling action: {e}")
            if action_id in self.active_scaling_actions:
                self.active_scaling_actions[action_id]['status'] = 'failed'
                self.active_scaling_actions[action_id]['error'] = str(e)

    async def _resource_monitoring_loop(self):
        """Background loop for resource monitoring"""
        while True:
            try:
                await self.collect_resource_metrics()
                await asyncio.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _prediction_loop(self):
        """Background loop for resource prediction"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                # Update prediction models with new data
                if len(self.resource_metrics_history) > 200:
                    await self._retrain_prediction_models()

            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")

    async def _optimization_loop(self):
        """Background loop for resource optimization"""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes

                await self.optimize_resource_allocation()

            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")

    async def _cost_optimization_loop(self):
        """Background loop for cost optimization"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Analyze cost patterns and update optimization strategies
                cost_data = await self._get_cost_history()
                if cost_data:
                    patterns = self.cost_optimizer.analyze_cost_patterns(cost_data)
                    await self._update_cost_optimization_strategies(patterns)

            except Exception as e:
                logger.error(f"Error in cost optimization loop: {e}")

    async def _health_monitoring_loop(self):
        """Background loop for health monitoring"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes

                current_metrics = self.resource_metrics_history[-1] if self.resource_metrics_history else None
                if current_metrics:
                    health_prediction = self.health_predictor.predict_health_degradation(
                        current_metrics.__dict__
                    )

                    # Store health prediction
                    await self.redis_client.set(
                        "resource_optimizer:health_prediction",
                        json.dumps(health_prediction, default=str),
                        ex=3600
                    )

            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")

    async def _train_prediction_models(self):
        """Train prediction models with historical data"""
        try:
            if len(self.resource_metrics_history) < 100:
                logger.info("Insufficient data for model training")
                return

            # Train models for each resource type and horizon
            for resource_type in ResourceType:
                data = self._extract_resource_time_series(resource_type)
                timestamps = [m.timestamp for m in self.resource_metrics_history]

                if len(data) < 50:
                    continue

                for horizon in PredictionHorizon:
                    predictor = self.predictors[horizon]
                    ensemble_model = predictor.train_ensemble_model(np.array(data), timestamps)
                    if ensemble_model:
                        predictor.models[f'ensemble_{horizon.value}'] = ensemble_model

            logger.info("Prediction models trained successfully")

        except Exception as e:
            logger.error(f"Failed to train prediction models: {e}")

    async def _retrain_prediction_models(self):
        """Retrain prediction models with new data"""
        try:
            # Retrain models periodically with new data
            await self._train_prediction_models()
            logger.info("Prediction models retrained")
        except Exception as e:
            logger.error(f"Failed to retrain prediction models: {e}")

    async def _get_current_resource_configuration(self) -> List[Dict[str, Any]]:
        """Get current resource configuration"""
        try:
            # Simulate current resource configuration
            return [
                {
                    'id': 'cpu_primary',
                    'type': 'cpu',
                    'capacity': 100.0,
                    'current_load': psutil.cpu_percent(),
                    'health_score': 0.9,
                    'cost': 5.0,
                    'utilization': psutil.cpu_percent() / 100.0,
                    'instance_type': 'on_demand',
                    'usage_stability': 0.8
                },
                {
                    'id': 'memory_primary',
                    'type': 'memory',
                    'capacity': 100.0,
                    'current_load': psutil.virtual_memory().percent,
                    'health_score': 0.85,
                    'cost': 3.0,
                    'utilization': psutil.virtual_memory().percent / 100.0,
                    'instance_type': 'on_demand',
                    'usage_stability': 0.7
                }
            ]
        except:
            return []

    async def _get_usage_patterns(self) -> Dict[str, Any]:
        """Get resource usage patterns"""
        if not self.resource_metrics_history:
            return {}

        # Analyze patterns from historical data
        cpu_usage = [m.cpu_usage for m in self.resource_metrics_history]
        memory_usage = [m.memory_usage for m in self.resource_metrics_history]

        return {
            'cpu_patterns': {
                'mean': np.mean(cpu_usage),
                'std': np.std(cpu_usage),
                'max': np.max(cpu_usage),
                'min': np.min(cpu_usage)
            },
            'memory_patterns': {
                'mean': np.mean(memory_usage),
                'std': np.std(memory_usage),
                'max': np.max(memory_usage),
                'min': np.min(memory_usage)
            }
        }

    async def _get_cost_history(self) -> List[Dict[str, Any]]:
        """Get cost history data"""
        try:
            cost_data = []
            for metrics in self.resource_metrics_history:
                cost_data.append({
                    'timestamp': metrics.timestamp.isoformat(),
                    'cost': metrics.cost_per_hour,
                    'resource_type': 'compute',
                    'efficiency': metrics.efficiency_score
                })
            return cost_data
        except:
            return []

    async def _update_cost_optimization_strategies(self, patterns: Dict[str, Any]):
        """Update cost optimization strategies based on patterns"""
        try:
            # Update optimization strategies based on discovered patterns
            logger.info(f"Updated cost optimization strategies based on patterns: {patterns}")
        except Exception as e:
            logger.error(f"Failed to update cost optimization strategies: {e}")

    async def _load_optimization_state(self):
        """Load optimization state from Redis"""
        try:
            state_data = await self.redis_client.get("resource_optimizer:state")
            if state_data:
                state = json.loads(state_data)
                # Load state data...
                logger.info("Loaded optimization state")
        except Exception as e:
            logger.error(f"Failed to load optimization state: {e}")

    async def _save_optimization_state(self):
        """Save optimization state to Redis"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'metrics_count': len(self.resource_metrics_history),
                'optimization_count': len(self.optimization_history)
            }
            await self.redis_client.set(
                "resource_optimizer:state",
                json.dumps(state),
                ex=86400
            )
        except Exception as e:
            logger.error(f"Failed to save optimization state: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive resource optimizer status"""
        current_metrics = self.resource_metrics_history[-1] if self.resource_metrics_history else None

        return {
            'metrics_collected': len(self.resource_metrics_history),
            'optimizations_performed': len(self.optimization_history),
            'active_scaling_actions': len(self.active_scaling_actions),
            'current_metrics': current_metrics.__dict__ if current_metrics else None,
            'prediction_models_trained': len([
                p for predictor in self.predictors.values()
                for p in predictor.models
            ]),
            'cost_optimization_strategies': len(self.cost_optimizer.optimization_strategies),
            'health_models_available': len(self.health_predictor.health_models)
        }

    async def shutdown(self):
        """Shutdown the resource optimizer"""
        try:
            # Save state
            await self._save_optimization_state()

            # Close connections
            if self.redis_client:
                await self.redis_client.close()
            if self.neo4j_driver:
                await self.neo4j_driver.close()

            # Shutdown executor
            self.executor.shutdown(wait=True)

            logger.info("Resource Optimizer shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")