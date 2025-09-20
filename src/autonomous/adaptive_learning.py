#!/usr/bin/env python3
"""
Adaptive Learning Engine
Online learning system with neural architecture search, continual learning, and model optimization
"""

import asyncio
import logging
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
import random
from collections import deque, defaultdict
import optuna
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pickle
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
import copy
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of machine learning models"""
    NEURAL_NETWORK = "neural_network"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    TRANSFORMER = "transformer"
    CNN = "cnn"
    LSTM = "lstm"
    AUTOENCODER = "autoencoder"
    REINFORCEMENT = "reinforcement"

class LearningMode(Enum):
    """Learning modes for the system"""
    ONLINE = "online"
    BATCH = "batch"
    CONTINUAL = "continual"
    TRANSFER = "transfer"
    META = "meta"
    FEDERATED = "federated"

class ArchitectureComponent(Enum):
    """Neural architecture components"""
    LINEAR = "linear"
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    LSTM = "lstm"
    GRU = "gru"
    ATTENTION = "attention"
    TRANSFORMER_BLOCK = "transformer_block"
    RESIDUAL_BLOCK = "residual_block"
    DROPOUT = "dropout"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"

@dataclass
class LearningTask:
    """Represents a learning task"""
    id: str
    name: str
    task_type: str  # classification, regression, forecasting, etc.
    data_source: str
    target_metric: str
    model_type: ModelType
    learning_mode: LearningMode
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    progress: float = 0.0
    best_score: float = 0.0
    current_model: Optional[Any] = None
    performance_history: List[Dict[str, float]] = field(default_factory=list)

@dataclass
class ModelArchitecture:
    """Represents a neural network architecture"""
    id: str
    name: str
    components: List[Dict[str, Any]]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_score: float = 0.0
    complexity_score: float = 0.0
    training_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

class NeuralArchitectureSearch:
    """Neural Architecture Search (NAS) system"""

    def __init__(self, search_space: Dict[str, Any]):
        self.search_space = search_space
        self.population_size = 20
        self.generations = 10
        self.mutation_rate = 0.3
        self.crossover_rate = 0.6
        self.elite_ratio = 0.2
        self.architectures_tested = {}
        self.best_architectures = []

    def create_random_architecture(self, input_shape: Tuple[int, ...],
                                 output_shape: Tuple[int, ...]) -> ModelArchitecture:
        """Create a random neural architecture"""
        components = []
        current_shape = input_shape

        # Input layer
        components.append({
            'type': ArchitectureComponent.LINEAR,
            'params': {
                'in_features': current_shape[-1],
                'out_features': random.choice([64, 128, 256, 512])
            }
        })

        # Hidden layers
        num_layers = random.randint(2, 6)
        for i in range(num_layers):
            layer_type = random.choice([
                ArchitectureComponent.LINEAR,
                ArchitectureComponent.DROPOUT,
                ArchitectureComponent.BATCH_NORM
            ])

            if layer_type == ArchitectureComponent.LINEAR:
                in_features = components[-1]['params'].get('out_features', current_shape[-1])
                out_features = random.choice([32, 64, 128, 256, 512])
                components.append({
                    'type': layer_type,
                    'params': {
                        'in_features': in_features,
                        'out_features': out_features
                    }
                })
            elif layer_type == ArchitectureComponent.DROPOUT:
                components.append({
                    'type': layer_type,
                    'params': {
                        'p': random.uniform(0.1, 0.5)
                    }
                })
            elif layer_type == ArchitectureComponent.BATCH_NORM:
                in_features = components[-1]['params'].get('out_features', current_shape[-1])
                components.append({
                    'type': layer_type,
                    'params': {
                        'num_features': in_features
                    }
                })

        # Output layer
        last_linear = None
        for comp in reversed(components):
            if comp['type'] == ArchitectureComponent.LINEAR:
                last_linear = comp
                break

        if last_linear:
            components.append({
                'type': ArchitectureComponent.LINEAR,
                'params': {
                    'in_features': last_linear['params']['out_features'],
                    'out_features': output_shape[-1]
                }
            })

        arch_id = hashlib.md5(str(components).encode()).hexdigest()[:12]

        return ModelArchitecture(
            id=arch_id,
            name=f"nas_arch_{arch_id}",
            components=components,
            input_shape=input_shape,
            output_shape=output_shape
        )

    def build_model_from_architecture(self, architecture: ModelArchitecture) -> nn.Module:
        """Build PyTorch model from architecture specification"""
        class DynamicModel(nn.Module):
            def __init__(self, components):
                super().__init__()
                self.layers = nn.ModuleList()

                for i, comp in enumerate(components):
                    if comp['type'] == ArchitectureComponent.LINEAR:
                        self.layers.append(nn.Linear(**comp['params']))
                    elif comp['type'] == ArchitectureComponent.DROPOUT:
                        self.layers.append(nn.Dropout(**comp['params']))
                    elif comp['type'] == ArchitectureComponent.BATCH_NORM:
                        self.layers.append(nn.BatchNorm1d(**comp['params']))

            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    if isinstance(layer, (nn.Dropout, nn.BatchNorm1d)):
                        x = layer(x)
                    else:
                        x = layer(x)
                        if i < len(self.layers) - 1:  # Don't apply activation to output layer
                            x = F.relu(x)
                return x

        return DynamicModel(architecture.components)

    def mutate_architecture(self, architecture: ModelArchitecture) -> ModelArchitecture:
        """Mutate an architecture"""
        new_components = copy.deepcopy(architecture.components)

        # Mutation strategies
        mutation_type = random.choice(['add_layer', 'remove_layer', 'modify_layer', 'change_activation'])

        if mutation_type == 'add_layer' and len(new_components) < 10:
            # Add a new layer
            insert_pos = random.randint(1, len(new_components) - 1)
            prev_layer = new_components[insert_pos - 1]

            if prev_layer['type'] == ArchitectureComponent.LINEAR:
                out_features = prev_layer['params']['out_features']
                new_layer = {
                    'type': ArchitectureComponent.LINEAR,
                    'params': {
                        'in_features': out_features,
                        'out_features': random.choice([32, 64, 128, 256])
                    }
                }
                new_components.insert(insert_pos, new_layer)

        elif mutation_type == 'remove_layer' and len(new_components) > 3:
            # Remove a layer (not input or output)
            remove_pos = random.randint(1, len(new_components) - 2)
            if new_components[remove_pos]['type'] == ArchitectureComponent.LINEAR:
                new_components.pop(remove_pos)

        elif mutation_type == 'modify_layer':
            # Modify layer parameters
            layer_idx = random.randint(0, len(new_components) - 1)
            layer = new_components[layer_idx]

            if layer['type'] == ArchitectureComponent.LINEAR:
                # Change output features
                layer['params']['out_features'] = random.choice([32, 64, 128, 256, 512])
            elif layer['type'] == ArchitectureComponent.DROPOUT:
                layer['params']['p'] = random.uniform(0.1, 0.5)

        # Fix connectivity issues
        self._fix_architecture_connectivity(new_components)

        arch_id = hashlib.md5(str(new_components).encode()).hexdigest()[:12]

        return ModelArchitecture(
            id=arch_id,
            name=f"mutated_arch_{arch_id}",
            components=new_components,
            input_shape=architecture.input_shape,
            output_shape=architecture.output_shape
        )

    def _fix_architecture_connectivity(self, components: List[Dict[str, Any]]):
        """Fix connectivity issues in architecture"""
        for i in range(1, len(components)):
            if (components[i]['type'] == ArchitectureComponent.LINEAR and
                components[i-1]['type'] == ArchitectureComponent.LINEAR):
                components[i]['params']['in_features'] = components[i-1]['params']['out_features']

    def crossover_architectures(self, parent1: ModelArchitecture,
                              parent2: ModelArchitecture) -> ModelArchitecture:
        """Crossover two architectures"""
        # Simple crossover: take parts from each parent
        components1 = parent1.components
        components2 = parent2.components

        crossover_point = random.randint(1, min(len(components1), len(components2)) - 1)

        new_components = (components1[:crossover_point] +
                         components2[crossover_point:])

        self._fix_architecture_connectivity(new_components)

        arch_id = hashlib.md5(str(new_components).encode()).hexdigest()[:12]

        return ModelArchitecture(
            id=arch_id,
            name=f"crossover_arch_{arch_id}",
            components=new_components,
            input_shape=parent1.input_shape,
            output_shape=parent1.output_shape
        )

    async def search_optimal_architecture(self, X: np.ndarray, y: np.ndarray,
                                        task_type: str = "regression") -> ModelArchitecture:
        """Search for optimal architecture using evolutionary algorithm"""
        input_shape = (X.shape[1],)
        output_shape = (1,) if task_type == "regression" else (len(np.unique(y)),)

        # Initialize population
        population = [
            self.create_random_architecture(input_shape, output_shape)
            for _ in range(self.population_size)
        ]

        best_architecture = None
        best_score = -float('inf')

        for generation in range(self.generations):
            # Evaluate population
            scores = []
            for arch in population:
                score = await self._evaluate_architecture(arch, X, y, task_type)
                arch.performance_score = score
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_architecture = arch

            # Selection and reproduction
            elite_count = int(self.population_size * self.elite_ratio)
            elite_indices = np.argsort(scores)[-elite_count:]
            elite_population = [population[i] for i in elite_indices]

            new_population = elite_population.copy()

            # Generate offspring
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_rate:
                    # Crossover
                    parent1, parent2 = random.sample(elite_population, 2)
                    offspring = self.crossover_architectures(parent1, parent2)
                else:
                    # Mutation
                    parent = random.choice(elite_population)
                    offspring = self.mutate_architecture(parent)

                new_population.append(offspring)

            population = new_population[:self.population_size]

            logger.info(f"NAS Generation {generation + 1}, Best Score: {best_score:.4f}")

        return best_architecture

    async def _evaluate_architecture(self, architecture: ModelArchitecture,
                                   X: np.ndarray, y: np.ndarray,
                                   task_type: str) -> float:
        """Evaluate architecture performance"""
        try:
            if architecture.id in self.architectures_tested:
                return self.architectures_tested[architecture.id]

            model = self.build_model_from_architecture(architecture)

            # Quick training for evaluation
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)

            if task_type == "classification":
                criterion = nn.CrossEntropyLoss()
                y_tensor = y_tensor.long()
            else:
                criterion = nn.MSELoss()
                y_tensor = y_tensor.unsqueeze(1)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for epoch in range(50):  # Quick training
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                if task_type == "classification":
                    predicted = torch.argmax(outputs, dim=1)
                    score = accuracy_score(y, predicted.numpy())
                else:
                    score = 1.0 / (1.0 + mean_squared_error(y, outputs.numpy().flatten()))

            # Consider complexity penalty
            num_params = sum(p.numel() for p in model.parameters())
            complexity_penalty = num_params / 1000000  # Penalty for large models
            final_score = score - 0.1 * complexity_penalty

            self.architectures_tested[architecture.id] = final_score
            return final_score

        except Exception as e:
            logger.error(f"Architecture evaluation failed: {e}")
            return 0.0

class ContinualLearningSystem:
    """Continual learning system for avoiding catastrophic forgetting"""

    def __init__(self):
        self.memory_buffer = deque(maxlen=10000)
        self.task_boundaries = []
        self.importance_weights = {}
        self.previous_models = []

    def add_experience(self, X: np.ndarray, y: np.ndarray, task_id: str):
        """Add experience to memory buffer"""
        for i in range(len(X)):
            self.memory_buffer.append({
                'x': X[i],
                'y': y[i],
                'task_id': task_id,
                'timestamp': datetime.now()
            })

    def replay_training(self, model: nn.Module, X_new: np.ndarray, y_new: np.ndarray,
                       replay_ratio: float = 0.3) -> nn.Module:
        """Train with experience replay"""
        if not self.memory_buffer:
            return self._train_standard(model, X_new, y_new)

        # Sample from memory buffer
        replay_size = int(len(X_new) * replay_ratio)
        replay_samples = random.sample(list(self.memory_buffer),
                                     min(replay_size, len(self.memory_buffer)))

        X_replay = np.array([s['x'] for s in replay_samples])
        y_replay = np.array([s['y'] for s in replay_samples])

        # Combine new and replay data
        X_combined = np.vstack([X_new, X_replay])
        y_combined = np.hstack([y_new, y_replay])

        return self._train_standard(model, X_combined, y_combined)

    def elastic_weight_consolidation(self, model: nn.Module, X_new: np.ndarray,
                                   y_new: np.ndarray, lambda_ewc: float = 1000) -> nn.Module:
        """Train with Elastic Weight Consolidation"""
        if not self.previous_models:
            return self._train_standard(model, X_new, y_new)

        # Calculate Fisher Information Matrix (simplified)
        fisher_information = self._calculate_fisher_information(model, X_new, y_new)

        # Training with EWC penalty
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.tensor(X_new, dtype=torch.float32)
        y_tensor = torch.tensor(y_new, dtype=torch.float32).unsqueeze(1)

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()

            # Standard loss
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)

            # EWC penalty
            ewc_penalty = 0
            if self.previous_models:
                prev_model = self.previous_models[-1]
                for (name, param), (_, prev_param) in zip(model.named_parameters(),
                                                         prev_model.named_parameters()):
                    if name in fisher_information:
                        ewc_penalty += (fisher_information[name] *
                                      (param - prev_param).pow(2)).sum()

            total_loss = loss + lambda_ewc * ewc_penalty
            total_loss.backward()
            optimizer.step()

        return model

    def _calculate_fisher_information(self, model: nn.Module, X: np.ndarray,
                                    y: np.ndarray) -> Dict[str, torch.Tensor]:
        """Calculate Fisher Information Matrix (simplified)"""
        fisher_info = {}

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        model.eval()
        criterion = nn.MSELoss()

        for name, param in model.named_parameters():
            fisher_info[name] = torch.zeros_like(param)

        for i in range(min(100, len(X))):  # Sample for efficiency
            model.zero_grad()
            output = model(X_tensor[i:i+1])
            loss = criterion(output, y_tensor[i:i+1])
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher_info[name] += param.grad.pow(2)

        # Normalize
        for name in fisher_info:
            fisher_info[name] /= min(100, len(X))

        return fisher_info

    def _train_standard(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> nn.Module:
        """Standard training procedure"""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        return model

class HyperparameterOptimizer:
    """Advanced hyperparameter optimization system"""

    def __init__(self):
        self.optimization_history = {}
        self.best_configurations = {}

    def optimize_model_hyperparameters(self, model_type: ModelType, X: np.ndarray,
                                     y: np.ndarray, n_trials: int = 100) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model type"""

        def objective(trial):
            if model_type == ModelType.NEURAL_NETWORK:
                return self._optimize_neural_network(trial, X, y)
            elif model_type == ModelType.RANDOM_FOREST:
                return self._optimize_random_forest(trial, X, y)
            elif model_type == ModelType.GRADIENT_BOOSTING:
                return self._optimize_gradient_boosting(trial, X, y)
            else:
                return 0.0

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        self.best_configurations[model_type] = best_params

        return best_params

    def _optimize_neural_network(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize neural network hyperparameters"""
        # Hyperparameters to optimize
        n_layers = trial.suggest_int('n_layers', 2, 5)
        hidden_sizes = []
        for i in range(n_layers):
            hidden_sizes.append(trial.suggest_categorical(f'hidden_size_{i}', [32, 64, 128, 256, 512]))

        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

        # Build and train model
        try:
            class OptimizedNN(nn.Module):
                def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
                    super().__init__()
                    layers = []
                    prev_size = input_size

                    for hidden_size in hidden_sizes:
                        layers.extend([
                            nn.Linear(prev_size, hidden_size),
                            nn.ReLU(),
                            nn.Dropout(dropout_rate)
                        ])
                        prev_size = hidden_size

                    layers.append(nn.Linear(prev_size, output_size))
                    self.network = nn.Sequential(*layers)

                def forward(self, x):
                    return self.network(x)

            model = OptimizedNN(X.shape[1], hidden_sizes, 1, dropout_rate)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()

            # Convert to tensors
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

            # Training
            model.train()
            for epoch in range(50):  # Quick training for optimization
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                mse = mean_squared_error(y, outputs.numpy().flatten())
                return 1.0 / (1.0 + mse)  # Convert to score to maximize

        except Exception as e:
            logger.error(f"Neural network optimization failed: {e}")
            return 0.0

    def _optimize_random_forest(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize Random Forest hyperparameters"""
        n_estimators = trial.suggest_int('n_estimators', 10, 200)
        max_depth = trial.suggest_int('max_depth', 3, 20)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])

        try:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )

            # Cross-validation
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            return -scores.mean()  # Convert to positive score

        except Exception as e:
            logger.error(f"Random Forest optimization failed: {e}")
            return 0.0

    def _optimize_gradient_boosting(self, trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optimize Gradient Boosting hyperparameters"""
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        subsample = trial.suggest_uniform('subsample', 0.6, 1.0)

        try:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                random_state=42
            )

            # Cross-validation
            scores = cross_val_score(model, X, y, cv=3, scoring='neg_mean_squared_error')
            return -scores.mean()

        except Exception as e:
            logger.error(f"Gradient Boosting optimization failed: {e}")
            return 0.0

class AdaptiveLearningEngine:
    """Main adaptive learning engine"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.learning_tasks: Dict[str, LearningTask] = {}
        self.models: Dict[str, Any] = {}
        self.data_sources: Dict[str, Any] = {}

        # Components
        self.nas_system = NeuralArchitectureSearch(config.get('nas_search_space', {}))
        self.continual_learning = ContinualLearningSystem()
        self.hyperparameter_optimizer = HyperparameterOptimizer()

        # Data processing
        self.scalers: Dict[str, Any] = {}
        self.feature_extractors: Dict[str, Any] = {}

        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.learning_metrics = {}

        # Infrastructure
        self.redis_client = None
        self.neo4j_driver = None
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Online learning
        self.online_models: Dict[str, Any] = {}
        self.data_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Model monitoring
        self.model_drift_detectors: Dict[str, Any] = {}
        self.performance_monitors: Dict[str, Any] = {}

    async def initialize(self):
        """Initialize the adaptive learning engine"""
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

            # Load existing state
            await self._load_learning_state()

            # Start background processes
            asyncio.create_task(self._online_learning_loop())
            asyncio.create_task(self._model_monitoring_loop())
            asyncio.create_task(self._hyperparameter_optimization_loop())
            asyncio.create_task(self._architecture_search_loop())

            logger.info("Adaptive Learning Engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Adaptive Learning Engine: {e}")
            raise

    async def create_learning_task(self, name: str, task_type: str, data_source: str,
                                 target_metric: str, model_type: ModelType,
                                 learning_mode: LearningMode,
                                 parameters: Dict[str, Any] = None) -> str:
        """Create a new learning task"""
        task_id = hashlib.md5(f"{name}:{task_type}:{time.time()}".encode()).hexdigest()[:12]

        task = LearningTask(
            id=task_id,
            name=name,
            task_type=task_type,
            data_source=data_source,
            target_metric=target_metric,
            model_type=model_type,
            learning_mode=learning_mode,
            parameters=parameters or {}
        )

        self.learning_tasks[task_id] = task
        await self._save_learning_state()

        # Start task execution
        asyncio.create_task(self._execute_learning_task(task))

        logger.info(f"Created learning task: {name} ({task_id})")
        return task_id

    async def _execute_learning_task(self, task: LearningTask):
        """Execute a learning task"""
        try:
            task.status = "running"

            # Load and prepare data
            X, y = await self._load_task_data(task)

            if X is None or y is None:
                task.status = "failed"
                return

            # Data preprocessing
            X_processed = await self._preprocess_data(X, task)

            # Model selection and training based on learning mode
            if task.learning_mode == LearningMode.ONLINE:
                model = await self._train_online_model(task, X_processed, y)
            elif task.learning_mode == LearningMode.BATCH:
                model = await self._train_batch_model(task, X_processed, y)
            elif task.learning_mode == LearningMode.CONTINUAL:
                model = await self._train_continual_model(task, X_processed, y)
            elif task.learning_mode == LearningMode.TRANSFER:
                model = await self._train_transfer_model(task, X_processed, y)
            elif task.learning_mode == LearningMode.META:
                model = await self._train_meta_model(task, X_processed, y)
            else:
                model = await self._train_batch_model(task, X_processed, y)

            if model is not None:
                task.current_model = model
                task.status = "completed"
                task.progress = 1.0

                # Evaluate model
                score = await self._evaluate_model(model, X_processed, y, task.task_type)
                task.best_score = score

                # Store model
                self.models[task.id] = model

                logger.info(f"Learning task {task.name} completed with score: {score:.4f}")
            else:
                task.status = "failed"

        except Exception as e:
            logger.error(f"Failed to execute learning task {task.id}: {e}")
            task.status = "failed"

    async def _load_task_data(self, task: LearningTask) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load data for a learning task"""
        try:
            # Simulate data loading - in production, this would connect to actual data sources
            data_source = task.data_source

            if data_source == "performance_metrics":
                # Load performance metrics from Redis
                metrics_data = await self.redis_client.get("system:performance_history")
                if metrics_data:
                    metrics = json.loads(metrics_data)
                    X = np.array([[m.get('cpu_usage', 0), m.get('memory_usage', 0),
                                 m.get('throughput', 0), m.get('response_time', 0)]
                                for m in metrics])
                    y = np.array([m.get('efficiency_score', 0) for m in metrics])
                    return X, y

            elif data_source == "task_completion":
                # Simulate task completion data
                n_samples = task.parameters.get('n_samples', 1000)
                n_features = task.parameters.get('n_features', 10)

                X = np.random.randn(n_samples, n_features)
                if task.task_type == "classification":
                    y = np.random.randint(0, 3, n_samples)
                else:
                    y = np.random.randn(n_samples)

                return X, y

            else:
                # Generate synthetic data for demonstration
                n_samples = 1000
                n_features = 10
                X = np.random.randn(n_samples, n_features)
                y = np.random.randn(n_samples)
                return X, y

        except Exception as e:
            logger.error(f"Failed to load data for task {task.id}: {e}")
            return None, None

    async def _preprocess_data(self, X: np.ndarray, task: LearningTask) -> np.ndarray:
        """Preprocess data for a learning task"""
        try:
            # Scaling
            scaler_key = f"{task.id}_scaler"
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                X_scaled = self.scalers[scaler_key].fit_transform(X)
            else:
                X_scaled = self.scalers[scaler_key].transform(X)

            # Feature extraction if needed
            if task.parameters.get('use_pca', False):
                pca_key = f"{task.id}_pca"
                if pca_key not in self.feature_extractors:
                    n_components = min(X.shape[1], task.parameters.get('pca_components', 5))
                    self.feature_extractors[pca_key] = PCA(n_components=n_components)
                    X_scaled = self.feature_extractors[pca_key].fit_transform(X_scaled)
                else:
                    X_scaled = self.feature_extractors[pca_key].transform(X_scaled)

            return X_scaled

        except Exception as e:
            logger.error(f"Data preprocessing failed for task {task.id}: {e}")
            return X

    async def _train_online_model(self, task: LearningTask, X: np.ndarray, y: np.ndarray) -> Any:
        """Train model using online learning"""
        try:
            # For online learning, we use incremental algorithms
            from sklearn.linear_model import SGDRegressor, SGDClassifier

            if task.task_type == "classification":
                model = SGDClassifier(random_state=42)
            else:
                model = SGDRegressor(random_state=42)

            # Online training with mini-batches
            batch_size = task.parameters.get('batch_size', 32)

            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]

                if i == 0:
                    model.fit(X_batch, y_batch)
                else:
                    model.partial_fit(X_batch, y_batch)

                # Update progress
                task.progress = min(1.0, (i + batch_size) / len(X))

            return model

        except Exception as e:
            logger.error(f"Online training failed for task {task.id}: {e}")
            return None

    async def _train_batch_model(self, task: LearningTask, X: np.ndarray, y: np.ndarray) -> Any:
        """Train model using batch learning"""
        try:
            if task.model_type == ModelType.NEURAL_NETWORK:
                # Use NAS to find optimal architecture
                if task.parameters.get('use_nas', False):
                    architecture = await self.nas_system.search_optimal_architecture(
                        X, y, task.task_type
                    )
                    model = self.nas_system.build_model_from_architecture(architecture)
                else:
                    # Use hyperparameter optimization
                    best_params = self.hyperparameter_optimizer.optimize_model_hyperparameters(
                        ModelType.NEURAL_NETWORK, X, y
                    )

                    # Build model with optimized parameters
                    class OptimizedNN(nn.Module):
                        def __init__(self, input_size, hidden_sizes, output_size, dropout_rate):
                            super().__init__()
                            layers = []
                            prev_size = input_size

                            for hidden_size in hidden_sizes:
                                layers.extend([
                                    nn.Linear(prev_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(dropout_rate)
                                ])
                                prev_size = hidden_size

                            layers.append(nn.Linear(prev_size, output_size))
                            self.network = nn.Sequential(*layers)

                        def forward(self, x):
                            return self.network(x)

                    hidden_sizes = [best_params.get(f'hidden_size_{i}', 64)
                                  for i in range(best_params.get('n_layers', 3))]
                    model = OptimizedNN(
                        X.shape[1], hidden_sizes, 1,
                        best_params.get('dropout_rate', 0.1)
                    )

                # Train the neural network
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=best_params.get('learning_rate', 0.001) if 'best_params' in locals() else 0.001
                )
                criterion = nn.MSELoss() if task.task_type == "regression" else nn.CrossEntropyLoss()

                X_tensor = torch.tensor(X, dtype=torch.float32)
                y_tensor = torch.tensor(y, dtype=torch.float32)
                if task.task_type == "regression":
                    y_tensor = y_tensor.unsqueeze(1)
                else:
                    y_tensor = y_tensor.long()

                model.train()
                epochs = task.parameters.get('epochs', 200)
                for epoch in range(epochs):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()

                    task.progress = (epoch + 1) / epochs

                return model

            elif task.model_type == ModelType.RANDOM_FOREST:
                best_params = self.hyperparameter_optimizer.optimize_model_hyperparameters(
                    ModelType.RANDOM_FOREST, X, y
                )

                if task.task_type == "classification":
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(**best_params, random_state=42)
                else:
                    model = RandomForestRegressor(**best_params, random_state=42)

                model.fit(X, y)
                task.progress = 1.0
                return model

            elif task.model_type == ModelType.GRADIENT_BOOSTING:
                best_params = self.hyperparameter_optimizer.optimize_model_hyperparameters(
                    ModelType.GRADIENT_BOOSTING, X, y
                )

                if task.task_type == "classification":
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(**best_params, random_state=42)
                else:
                    model = GradientBoostingRegressor(**best_params, random_state=42)

                model.fit(X, y)
                task.progress = 1.0
                return model

        except Exception as e:
            logger.error(f"Batch training failed for task {task.id}: {e}")
            return None

    async def _train_continual_model(self, task: LearningTask, X: np.ndarray, y: np.ndarray) -> Any:
        """Train model using continual learning"""
        try:
            # Initialize or load existing model
            if task.id in self.models:
                model = self.models[task.id]
            else:
                # Create new model
                model = nn.Sequential(
                    nn.Linear(X.shape[1], 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )

            # Add new experience to continual learning system
            self.continual_learning.add_experience(X, y, task.id)

            # Train with continual learning strategy
            if task.parameters.get('use_ewc', False):
                model = self.continual_learning.elastic_weight_consolidation(model, X, y)
            else:
                model = self.continual_learning.replay_training(model, X, y)

            # Save model for future continual learning
            self.continual_learning.previous_models.append(copy.deepcopy(model))

            task.progress = 1.0
            return model

        except Exception as e:
            logger.error(f"Continual learning failed for task {task.id}: {e}")
            return None

    async def _train_transfer_model(self, task: LearningTask, X: np.ndarray, y: np.ndarray) -> Any:
        """Train model using transfer learning"""
        try:
            # Find similar tasks for transfer learning
            source_task = await self._find_similar_task(task)

            if source_task and source_task.id in self.models:
                # Load pre-trained model
                source_model = self.models[source_task.id]

                if isinstance(source_model, nn.Module):
                    # Fine-tune neural network
                    model = copy.deepcopy(source_model)

                    # Freeze early layers
                    freeze_layers = task.parameters.get('freeze_layers', 2)
                    for i, param in enumerate(model.parameters()):
                        if i < freeze_layers:
                            param.requires_grad = False

                    # Fine-tune with lower learning rate
                    optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=0.0001  # Lower learning rate for fine-tuning
                    )
                    criterion = nn.MSELoss()

                    X_tensor = torch.tensor(X, dtype=torch.float32)
                    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

                    model.train()
                    for epoch in range(50):  # Fewer epochs for fine-tuning
                        optimizer.zero_grad()
                        outputs = model(X_tensor)
                        loss = criterion(outputs, y_tensor)
                        loss.backward()
                        optimizer.step()

                        task.progress = (epoch + 1) / 50

                    return model
                else:
                    # For sklearn models, use the model as feature extractor
                    # This is a simplified approach
                    return await self._train_batch_model(task, X, y)
            else:
                # No suitable source task found, use regular training
                return await self._train_batch_model(task, X, y)

        except Exception as e:
            logger.error(f"Transfer learning failed for task {task.id}: {e}")
            return None

    async def _train_meta_model(self, task: LearningTask, X: np.ndarray, y: np.ndarray) -> Any:
        """Train model using meta-learning"""
        try:
            # Implement a simple meta-learning approach
            # In practice, this would use more sophisticated algorithms like MAML

            # Collect data from similar tasks
            similar_tasks = await self._find_similar_tasks(task, max_tasks=5)

            if not similar_tasks:
                return await self._train_batch_model(task, X, y)

            # Meta-training: learn general initialization
            meta_model = nn.Sequential(
                nn.Linear(X.shape[1], 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

            meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)

            # Simulate meta-training on similar tasks
            for similar_task in similar_tasks:
                if similar_task.id in self.models:
                    # Create a copy for inner loop optimization
                    inner_model = copy.deepcopy(meta_model)
                    inner_optimizer = torch.optim.SGD(inner_model.parameters(), lr=0.01)

                    # Simulate inner loop training
                    X_sim, y_sim = await self._load_task_data(similar_task)
                    if X_sim is not None and y_sim is not None:
                        X_sim = await self._preprocess_data(X_sim, similar_task)

                        X_tensor = torch.tensor(X_sim[:100], dtype=torch.float32)  # Use subset
                        y_tensor = torch.tensor(y_sim[:100], dtype=torch.float32).unsqueeze(1)

                        # Inner loop
                        for _ in range(5):
                            inner_optimizer.zero_grad()
                            outputs = inner_model(X_tensor)
                            loss = F.mse_loss(outputs, y_tensor)
                            loss.backward()
                            inner_optimizer.step()

                        # Meta-update
                        meta_optimizer.zero_grad()
                        # Use the adapted model to compute meta-loss
                        meta_loss = F.mse_loss(inner_model(X_tensor), y_tensor)
                        meta_loss.backward()
                        meta_optimizer.step()

            # Fine-tune on current task
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

            optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = meta_model(X_tensor)
                loss = F.mse_loss(outputs, y_tensor)
                loss.backward()
                optimizer.step()

                task.progress = (epoch + 1) / 100

            return meta_model

        except Exception as e:
            logger.error(f"Meta-learning failed for task {task.id}: {e}")
            return None

    async def _find_similar_task(self, task: LearningTask) -> Optional[LearningTask]:
        """Find the most similar completed task"""
        similar_tasks = await self._find_similar_tasks(task, max_tasks=1)
        return similar_tasks[0] if similar_tasks else None

    async def _find_similar_tasks(self, task: LearningTask, max_tasks: int = 5) -> List[LearningTask]:
        """Find similar completed tasks"""
        completed_tasks = [t for t in self.learning_tasks.values()
                          if t.status == "completed" and t.id != task.id]

        if not completed_tasks:
            return []

        # Simple similarity based on task type and parameters
        similarities = []
        for completed_task in completed_tasks:
            similarity = 0.0

            # Task type similarity
            if completed_task.task_type == task.task_type:
                similarity += 0.5

            # Model type similarity
            if completed_task.model_type == task.model_type:
                similarity += 0.3

            # Parameter similarity (simplified)
            common_params = set(completed_task.parameters.keys()) & set(task.parameters.keys())
            if common_params:
                similarity += 0.2 * len(common_params) / max(len(completed_task.parameters), len(task.parameters))

            similarities.append((completed_task, similarity))

        # Sort by similarity and return top tasks
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [task for task, _ in similarities[:max_tasks]]

    async def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, task_type: str) -> float:
        """Evaluate model performance"""
        try:
            if isinstance(model, nn.Module):
                model.eval()
                X_tensor = torch.tensor(X, dtype=torch.float32)
                with torch.no_grad():
                    predictions = model(X_tensor).numpy().flatten()
            else:
                predictions = model.predict(X)

            if task_type == "classification":
                if isinstance(model, nn.Module):
                    predictions = np.argmax(predictions, axis=1) if len(predictions.shape) > 1 else predictions
                score = accuracy_score(y, predictions)
            else:
                score = 1.0 / (1.0 + mean_squared_error(y, predictions))

            return float(score)

        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return 0.0

    async def _online_learning_loop(self):
        """Background loop for online learning"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Process data buffers for online learning
                for task_id, buffer in self.data_buffers.items():
                    if len(buffer) > 100:  # Sufficient data for update
                        task = self.learning_tasks.get(task_id)
                        if task and task.learning_mode == LearningMode.ONLINE:
                            await self._update_online_model(task, buffer)

            except Exception as e:
                logger.error(f"Error in online learning loop: {e}")

    async def _model_monitoring_loop(self):
        """Background loop for model performance monitoring"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                for task_id, model in self.models.items():
                    await self._monitor_model_performance(task_id, model)

            except Exception as e:
                logger.error(f"Error in model monitoring loop: {e}")

    async def _hyperparameter_optimization_loop(self):
        """Background loop for continuous hyperparameter optimization"""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour

                # Optimize hyperparameters for active learning tasks
                active_tasks = [t for t in self.learning_tasks.values()
                              if t.status == "running" and t.model_type != ModelType.NEURAL_NETWORK]

                for task in active_tasks:
                    if task.parameters.get('auto_optimize', True):
                        await self._optimize_task_hyperparameters(task)

            except Exception as e:
                logger.error(f"Error in hyperparameter optimization loop: {e}")

    async def _architecture_search_loop(self):
        """Background loop for neural architecture search"""
        while True:
            try:
                await asyncio.sleep(7200)  # Check every 2 hours

                # Run NAS for neural network tasks
                nn_tasks = [t for t in self.learning_tasks.values()
                           if t.model_type == ModelType.NEURAL_NETWORK and t.status == "completed"]

                for task in nn_tasks:
                    if task.parameters.get('continuous_nas', False):
                        await self._improve_architecture(task)

            except Exception as e:
                logger.error(f"Error in architecture search loop: {e}")

    async def _update_online_model(self, task: LearningTask, buffer: deque):
        """Update online model with new data"""
        try:
            # Extract data from buffer
            data_points = list(buffer)
            X = np.array([dp['x'] for dp in data_points])
            y = np.array([dp['y'] for dp in data_points])

            if task.id in self.online_models:
                model = self.online_models[task.id]

                # Incremental learning
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(X, y)

                    # Evaluate and update performance
                    score = await self._evaluate_model(model, X, y, task.task_type)
                    task.performance_history.append({
                        'timestamp': datetime.now(),
                        'score': score,
                        'samples': len(X)
                    })

            # Clear processed data from buffer
            buffer.clear()

        except Exception as e:
            logger.error(f"Failed to update online model for task {task.id}: {e}")

    async def _monitor_model_performance(self, task_id: str, model: Any):
        """Monitor model performance and detect drift"""
        try:
            task = self.learning_tasks.get(task_id)
            if not task:
                return

            # Load recent data for monitoring
            X_recent, y_recent = await self._load_recent_data(task)

            if X_recent is not None and y_recent is not None:
                # Evaluate current performance
                current_score = await self._evaluate_model(model, X_recent, y_recent, task.task_type)

                # Compare with historical performance
                if task.performance_history:
                    avg_historical_score = np.mean([h['score'] for h in task.performance_history[-10:]])

                    # Detect performance degradation
                    if current_score < avg_historical_score * 0.9:  # 10% degradation threshold
                        logger.warning(f"Performance degradation detected for task {task_id}")

                        # Trigger retraining if configured
                        if task.parameters.get('auto_retrain', False):
                            await self._retrain_model(task)

                # Update performance history
                task.performance_history.append({
                    'timestamp': datetime.now(),
                    'score': current_score,
                    'type': 'monitoring'
                })

        except Exception as e:
            logger.error(f"Model monitoring failed for task {task_id}: {e}")

    async def _load_recent_data(self, task: LearningTask) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load recent data for model monitoring"""
        # This would load recent data specific to the task
        # For now, we'll simulate this
        return await self._load_task_data(task)

    async def _retrain_model(self, task: LearningTask):
        """Retrain a model that has degraded"""
        try:
            logger.info(f"Retraining model for task {task.id}")

            # Load fresh data
            X, y = await self._load_task_data(task)
            if X is not None and y is not None:
                X_processed = await self._preprocess_data(X, task)

                # Retrain based on learning mode
                if task.learning_mode == LearningMode.CONTINUAL:
                    model = await self._train_continual_model(task, X_processed, y)
                else:
                    model = await self._train_batch_model(task, X_processed, y)

                if model is not None:
                    self.models[task.id] = model
                    task.current_model = model

                    # Evaluate new model
                    score = await self._evaluate_model(model, X_processed, y, task.task_type)
                    task.best_score = max(task.best_score, score)

                    logger.info(f"Model retrained for task {task.id}, new score: {score:.4f}")

        except Exception as e:
            logger.error(f"Model retraining failed for task {task.id}: {e}")

    async def _optimize_task_hyperparameters(self, task: LearningTask):
        """Optimize hyperparameters for a specific task"""
        try:
            X, y = await self._load_task_data(task)
            if X is not None and y is not None:
                X_processed = await self._preprocess_data(X, task)

                # Run hyperparameter optimization
                best_params = self.hyperparameter_optimizer.optimize_model_hyperparameters(
                    task.model_type, X_processed, y, n_trials=50
                )

                # Update task parameters
                task.parameters.update(best_params)

                logger.info(f"Optimized hyperparameters for task {task.id}: {best_params}")

        except Exception as e:
            logger.error(f"Hyperparameter optimization failed for task {task.id}: {e}")

    async def _improve_architecture(self, task: LearningTask):
        """Improve neural architecture for a task"""
        try:
            if task.model_type != ModelType.NEURAL_NETWORK:
                return

            X, y = await self._load_task_data(task)
            if X is not None and y is not None:
                X_processed = await self._preprocess_data(X, task)

                # Run architecture search
                improved_arch = await self.nas_system.search_optimal_architecture(
                    X_processed, y, task.task_type
                )

                # Compare with current model
                current_score = task.best_score

                if improved_arch.performance_score > current_score:
                    # Build and train improved model
                    improved_model = self.nas_system.build_model_from_architecture(improved_arch)

                    # Train the improved model
                    optimizer = torch.optim.Adam(improved_model.parameters(), lr=0.001)
                    criterion = nn.MSELoss()

                    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
                    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

                    improved_model.train()
                    for epoch in range(100):
                        optimizer.zero_grad()
                        outputs = improved_model(X_tensor)
                        loss = criterion(outputs, y_tensor)
                        loss.backward()
                        optimizer.step()

                    # Update task with improved model
                    self.models[task.id] = improved_model
                    task.current_model = improved_model
                    task.best_score = improved_arch.performance_score

                    logger.info(f"Improved architecture for task {task.id}, new score: {improved_arch.performance_score:.4f}")

        except Exception as e:
            logger.error(f"Architecture improvement failed for task {task.id}: {e}")

    async def _load_learning_state(self):
        """Load learning state from Redis"""
        try:
            state_data = await self.redis_client.get("adaptive_learning:state")
            if state_data:
                state = json.loads(state_data)

                # Load learning tasks
                for task_data in state.get('learning_tasks', []):
                    task = LearningTask(**task_data)
                    self.learning_tasks[task.id] = task

                logger.info(f"Loaded {len(self.learning_tasks)} learning tasks")

        except Exception as e:
            logger.error(f"Failed to load learning state: {e}")

    async def _save_learning_state(self):
        """Save learning state to Redis"""
        try:
            state = {
                'learning_tasks': [
                    {
                        'id': task.id,
                        'name': task.name,
                        'task_type': task.task_type,
                        'data_source': task.data_source,
                        'target_metric': task.target_metric,
                        'model_type': task.model_type.value,
                        'learning_mode': task.learning_mode.value,
                        'parameters': task.parameters,
                        'created_at': task.created_at.isoformat(),
                        'status': task.status,
                        'progress': task.progress,
                        'best_score': task.best_score,
                        'performance_history': task.performance_history
                    }
                    for task in self.learning_tasks.values()
                ]
            }

            await self.redis_client.set(
                "adaptive_learning:state",
                json.dumps(state, default=str),
                ex=86400  # 24 hours
            )

        except Exception as e:
            logger.error(f"Failed to save learning state: {e}")

    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive learning engine status"""
        return {
            'total_tasks': len(self.learning_tasks),
            'running_tasks': len([t for t in self.learning_tasks.values() if t.status == "running"]),
            'completed_tasks': len([t for t in self.learning_tasks.values() if t.status == "completed"]),
            'failed_tasks': len([t for t in self.learning_tasks.values() if t.status == "failed"]),
            'total_models': len(self.models),
            'online_models': len(self.online_models),
            'architectures_tested': len(self.nas_system.architectures_tested),
            'best_configurations': len(self.hyperparameter_optimizer.best_configurations),
            'learning_metrics': self.learning_metrics
        }

    async def shutdown(self):
        """Shutdown the adaptive learning engine"""
        try:
            # Save state
            await self._save_learning_state()

            # Close connections
            if self.redis_client:
                await self.redis_client.close()
            if self.neo4j_driver:
                await self.neo4j_driver.close()

            # Shutdown executor
            self.executor.shutdown(wait=True)

            logger.info("Adaptive Learning Engine shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")