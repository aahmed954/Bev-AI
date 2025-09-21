"""
BEV OSINT Framework - ML-based Cache Prediction System
Advanced machine learning models for cache hit prediction and pattern analysis.
"""

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import aioredis
import psycopg2.pool
from prometheus_client import Gauge, Counter, Histogram


class PredictionType(Enum):
    HIT_PROBABILITY = "hit_probability"
    ACCESS_TIME = "access_time"
    USER_PATTERN = "user_pattern"
    POPULARITY = "popularity"


class ModelType(Enum):
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


@dataclass
class QueryPattern:
    """Represents a query pattern for ML analysis."""
    query_hash: str
    query_type: str
    user_id: Optional[str]
    timestamp: datetime
    response_size: int
    processing_time: float
    cache_hit: bool
    user_context: Dict[str, Any]
    query_features: Dict[str, float]


@dataclass
class PredictionResult:
    """ML prediction result with confidence metrics."""
    query_hash: str
    prediction_type: PredictionType
    predicted_value: float
    confidence: float
    model_used: ModelType
    features_used: List[str]
    timestamp: datetime


@dataclass
class UserBehaviorProfile:
    """User behavior profile for personalized caching."""
    user_id: str
    query_frequency: Dict[str, float]
    access_patterns: Dict[str, List[int]]  # Hour-of-day patterns
    preferred_data_types: List[str]
    cache_hit_rate: float
    last_updated: datetime


class MLPredictor:
    """
    Advanced ML-based cache prediction system.
    Uses multiple models for different prediction tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_url = config.get("redis_url", "redis://redis:6379/12")
        self.postgres_uri = config.get("postgres_uri")

        # Model storage
        self.models: Dict[PredictionType, Any] = {}
        self.scalers: Dict[PredictionType, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}

        # Training data
        self.training_data: List[QueryPattern] = []
        self.user_profiles: Dict[str, UserBehaviorProfile] = {}

        # Configuration
        self.min_training_samples = config.get("min_training_samples", 1000)
        self.retrain_interval = config.get("retrain_interval_hours", 6)
        self.feature_window_hours = config.get("feature_window_hours", 24)
        self.model_accuracy_threshold = config.get("model_accuracy_threshold", 0.8)

        # External connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

        # Metrics
        self._init_prometheus_metrics()

        # Logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for ML predictor."""
        logger = logging.getLogger('ml_predictor')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for ML predictor."""
        self.prom_prediction_accuracy = Gauge(
            'bev_cache_prediction_accuracy',
            'Cache prediction model accuracy',
            ['model_type']
        )
        self.prom_predictions_total = Counter(
            'bev_cache_predictions_total',
            'Total cache predictions made',
            ['prediction_type', 'model_type']
        )
        self.prom_training_time = Histogram(
            'bev_model_training_duration_seconds',
            'Model training duration',
            ['model_type']
        )
        self.prom_model_size = Gauge(
            'bev_model_size_bytes',
            'Model size in bytes',
            ['model_type']
        )

    async def initialize(self):
        """Initialize ML predictor with connections and pre-trained models."""
        try:
            # Initialize Redis connection
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                decode_responses=True
            )

            # Initialize database connection pool
            if self.postgres_uri:
                self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    dsn=self.postgres_uri
                )

            # Load existing models
            await self._load_models()

            # Load training data
            await self._load_training_data()

            # Load user profiles
            await self._load_user_profiles()

            self.logger.info("ML predictor initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize ML predictor: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown of ML predictor."""
        if self.redis_client:
            await self.redis_client.close()
        if self.db_pool:
            self.db_pool.closeall()
        self.logger.info("ML predictor shutdown completed")

    async def predict_cache_hit(self, query_hash: str, query_type: str,
                               user_id: Optional[str] = None,
                               query_features: Optional[Dict[str, float]] = None) -> PredictionResult:
        """
        Predict cache hit probability for a query.

        Args:
            query_hash: Hash of the query
            query_type: Type of query (osint, intelligence, etc.)
            user_id: Optional user identifier
            query_features: Optional pre-computed features

        Returns:
            PredictionResult with hit probability prediction
        """
        try:
            # Extract features
            features = await self._extract_features(
                query_hash, query_type, user_id, query_features
            )

            # Get model
            model = self.models.get(PredictionType.HIT_PROBABILITY)
            scaler = self.scalers.get(PredictionType.HIT_PROBABILITY)

            if not model or not scaler:
                # Use heuristic if model not available
                return await self._heuristic_hit_prediction(query_hash, query_type, user_id)

            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, PredictionType.HIT_PROBABILITY)
            scaled_features = scaler.transform([feature_vector])

            # Make prediction
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(scaled_features)[0][1]  # Probability of hit
            else:
                prediction = float(model.predict(scaled_features)[0])
                prediction = max(0.0, min(1.0, prediction))  # Clamp to [0,1]

            # Calculate confidence based on feature quality
            confidence = self._calculate_prediction_confidence(features, model)

            result = PredictionResult(
                query_hash=query_hash,
                prediction_type=PredictionType.HIT_PROBABILITY,
                predicted_value=prediction,
                confidence=confidence,
                model_used=ModelType.RANDOM_FOREST,
                features_used=list(features.keys()),
                timestamp=datetime.now(timezone.utc)
            )

            # Update metrics
            self.prom_predictions_total.labels(
                prediction_type=PredictionType.HIT_PROBABILITY.value,
                model_type=ModelType.RANDOM_FOREST.value
            ).inc()

            return result

        except Exception as e:
            self.logger.error(f"Error predicting cache hit for {query_hash}: {e}")
            return await self._heuristic_hit_prediction(query_hash, query_type, user_id)

    async def predict_access_time(self, query_hash: str, query_type: str,
                                 user_id: Optional[str] = None) -> PredictionResult:
        """
        Predict when a query will likely be accessed next.

        Args:
            query_hash: Hash of the query
            query_type: Type of query
            user_id: Optional user identifier

        Returns:
            PredictionResult with predicted access time (hours from now)
        """
        try:
            # Extract temporal features
            features = await self._extract_temporal_features(query_hash, query_type, user_id)

            # Get model
            model = self.models.get(PredictionType.ACCESS_TIME)
            scaler = self.scalers.get(PredictionType.ACCESS_TIME)

            if not model or not scaler:
                return await self._heuristic_access_prediction(query_hash, query_type, user_id)

            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, PredictionType.ACCESS_TIME)
            scaled_features = scaler.transform([feature_vector])

            # Make prediction (hours from now)
            predicted_hours = float(model.predict(scaled_features)[0])
            predicted_hours = max(0.1, min(168.0, predicted_hours))  # Clamp to [0.1, 168] hours

            confidence = self._calculate_prediction_confidence(features, model)

            result = PredictionResult(
                query_hash=query_hash,
                prediction_type=PredictionType.ACCESS_TIME,
                predicted_value=predicted_hours,
                confidence=confidence,
                model_used=ModelType.GRADIENT_BOOSTING,
                features_used=list(features.keys()),
                timestamp=datetime.now(timezone.utc)
            )

            self.prom_predictions_total.labels(
                prediction_type=PredictionType.ACCESS_TIME.value,
                model_type=ModelType.GRADIENT_BOOSTING.value
            ).inc()

            return result

        except Exception as e:
            self.logger.error(f"Error predicting access time for {query_hash}: {e}")
            return await self._heuristic_access_prediction(query_hash, query_type, user_id)

    async def analyze_user_patterns(self, user_id: str) -> UserBehaviorProfile:
        """
        Analyze user behavior patterns for personalized caching.

        Args:
            user_id: User identifier

        Returns:
            UserBehaviorProfile with user behavior analysis
        """
        try:
            # Get user's historical data
            user_data = await self._get_user_historical_data(user_id)

            if not user_data:
                return self._create_default_user_profile(user_id)

            # Analyze query frequency
            query_frequency = self._analyze_query_frequency(user_data)

            # Analyze access patterns
            access_patterns = self._analyze_access_patterns(user_data)

            # Determine preferred data types
            preferred_types = self._analyze_preferred_data_types(user_data)

            # Calculate cache hit rate
            hit_rate = self._calculate_user_hit_rate(user_data)

            profile = UserBehaviorProfile(
                user_id=user_id,
                query_frequency=query_frequency,
                access_patterns=access_patterns,
                preferred_data_types=preferred_types,
                cache_hit_rate=hit_rate,
                last_updated=datetime.now(timezone.utc)
            )

            # Store profile
            await self._store_user_profile(profile)
            self.user_profiles[user_id] = profile

            return profile

        except Exception as e:
            self.logger.error(f"Error analyzing patterns for user {user_id}: {e}")
            return self._create_default_user_profile(user_id)

    async def train_models(self) -> Dict[PredictionType, float]:
        """
        Train all ML models with latest data.

        Returns:
            Dictionary of model accuracies by prediction type
        """
        accuracies = {}

        try:
            # Ensure we have enough training data
            await self._collect_training_data()

            if len(self.training_data) < self.min_training_samples:
                self.logger.warning(
                    f"Insufficient training data: {len(self.training_data)} < {self.min_training_samples}"
                )
                return accuracies

            # Train hit probability model
            hit_accuracy = await self._train_hit_probability_model()
            accuracies[PredictionType.HIT_PROBABILITY] = hit_accuracy

            # Train access time model
            access_accuracy = await self._train_access_time_model()
            accuracies[PredictionType.ACCESS_TIME] = access_accuracy

            # Save models
            await self._save_models()

            self.logger.info(f"Model training completed. Accuracies: {accuracies}")

            return accuracies

        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            return accuracies

    async def _extract_features(self, query_hash: str, query_type: str,
                               user_id: Optional[str],
                               query_features: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Extract comprehensive features for cache prediction."""
        features = {}

        # Basic query features
        features['query_type_encoded'] = self._encode_query_type(query_type)
        features['query_complexity'] = len(query_hash) / 64.0  # Normalized complexity

        # Time-based features
        now = datetime.now(timezone.utc)
        features['hour_of_day'] = now.hour / 23.0
        features['day_of_week'] = now.weekday() / 6.0
        features['is_weekend'] = float(now.weekday() >= 5)

        # Historical features
        hist_data = await self._get_query_history(query_hash)
        if hist_data:
            features['access_frequency'] = len(hist_data) / 30.0  # Last 30 days
            features['avg_response_time'] = np.mean([d.get('response_time', 0) for d in hist_data])
            features['last_access_hours'] = self._hours_since_last_access(hist_data)
        else:
            features['access_frequency'] = 0.0
            features['avg_response_time'] = 0.0
            features['last_access_hours'] = 999.0

        # User-specific features
        if user_id:
            user_profile = self.user_profiles.get(user_id)
            if user_profile:
                features['user_hit_rate'] = user_profile.cache_hit_rate
                features['user_query_freq'] = user_profile.query_frequency.get(query_type, 0.0)
            else:
                features['user_hit_rate'] = 0.5  # Default
                features['user_query_freq'] = 0.0
        else:
            features['user_hit_rate'] = 0.5
            features['user_query_freq'] = 0.0

        # System load features
        system_load = await self._get_system_load()
        features['cpu_usage'] = system_load.get('cpu_usage', 0.0) / 100.0
        features['memory_usage'] = system_load.get('memory_usage', 0.0) / 100.0
        features['cache_usage'] = system_load.get('cache_usage', 0.0) / 100.0

        # Custom query features
        if query_features:
            for key, value in query_features.items():
                features[f'custom_{key}'] = float(value)

        return features

    async def _extract_temporal_features(self, query_hash: str, query_type: str,
                                       user_id: Optional[str]) -> Dict[str, float]:
        """Extract features specific to temporal access prediction."""
        features = {}

        # Get historical access pattern
        hist_data = await self._get_query_history(query_hash, hours=168)  # Last week

        if hist_data:
            # Access frequency patterns
            access_times = [datetime.fromisoformat(d['timestamp']) for d in hist_data]

            # Hour-of-day pattern
            hours = [t.hour for t in access_times]
            features['most_frequent_hour'] = max(set(hours), key=hours.count) / 23.0

            # Day-of-week pattern
            days = [t.weekday() for t in access_times]
            features['most_frequent_day'] = max(set(days), key=days.count) / 6.0

            # Access intervals
            if len(access_times) > 1:
                intervals = []
                for i in range(1, len(access_times)):
                    interval = (access_times[i] - access_times[i-1]).total_seconds() / 3600
                    intervals.append(interval)

                features['avg_access_interval'] = np.mean(intervals) / 168.0  # Normalized to weeks
                features['access_regularity'] = 1.0 / (1.0 + np.std(intervals) / np.mean(intervals))
            else:
                features['avg_access_interval'] = 1.0
                features['access_regularity'] = 0.0
        else:
            features['most_frequent_hour'] = 0.5
            features['most_frequent_day'] = 0.5
            features['avg_access_interval'] = 1.0
            features['access_regularity'] = 0.0

        # User temporal patterns
        if user_id and user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            user_patterns = profile.access_patterns.get(query_type, [0] * 24)
            current_hour = datetime.now(timezone.utc).hour
            features['user_hour_preference'] = user_patterns[current_hour] / max(user_patterns) if max(user_patterns) > 0 else 0.0
        else:
            features['user_hour_preference'] = 0.5

        return features

    async def _train_hit_probability_model(self) -> float:
        """Train the cache hit probability model."""
        try:
            start_time = time.time()

            # Prepare training data
            X, y = self._prepare_training_data_for_hit_prediction()

            if len(X) < 100:
                self.logger.warning("Insufficient data for hit probability model training")
                return 0.0

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Random Forest model
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            # Store model and scaler
            self.models[PredictionType.HIT_PROBABILITY] = model
            self.scalers[PredictionType.HIT_PROBABILITY] = scaler

            # Update metrics
            training_time = time.time() - start_time
            self.prom_training_time.labels(
                model_type=ModelType.RANDOM_FOREST.value
            ).observe(training_time)

            self.prom_prediction_accuracy.labels(
                model_type=ModelType.RANDOM_FOREST.value
            ).set(accuracy)

            # Calculate model size
            model_size = len(pickle.dumps(model))
            self.prom_model_size.labels(
                model_type=ModelType.RANDOM_FOREST.value
            ).set(model_size)

            self.logger.info(f"Hit probability model trained. Accuracy: {accuracy:.3f}, Time: {training_time:.2f}s")

            return accuracy

        except Exception as e:
            self.logger.error(f"Error training hit probability model: {e}")
            return 0.0

    async def _train_access_time_model(self) -> float:
        """Train the access time prediction model."""
        try:
            start_time = time.time()

            # Prepare training data
            X, y = self._prepare_training_data_for_access_prediction()

            if len(X) < 100:
                self.logger.warning("Insufficient data for access time model training")
                return 0.0

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Gradient Boosting model
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            accuracy = 1.0 / (1.0 + mse)  # Convert MSE to accuracy-like metric

            # Store model and scaler
            self.models[PredictionType.ACCESS_TIME] = model
            self.scalers[PredictionType.ACCESS_TIME] = scaler

            # Update metrics
            training_time = time.time() - start_time
            self.prom_training_time.labels(
                model_type=ModelType.GRADIENT_BOOSTING.value
            ).observe(training_time)

            self.prom_prediction_accuracy.labels(
                model_type=ModelType.GRADIENT_BOOSTING.value
            ).set(accuracy)

            model_size = len(pickle.dumps(model))
            self.prom_model_size.labels(
                model_type=ModelType.GRADIENT_BOOSTING.value
            ).set(model_size)

            self.logger.info(f"Access time model trained. Accuracy: {accuracy:.3f}, Time: {training_time:.2f}s")

            return accuracy

        except Exception as e:
            self.logger.error(f"Error training access time model: {e}")
            return 0.0

    def _prepare_training_data_for_hit_prediction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for hit probability prediction."""
        X, y = [], []

        for pattern in self.training_data:
            # Extract features
            feature_vector = self._pattern_to_feature_vector(pattern, PredictionType.HIT_PROBABILITY)
            X.append(feature_vector)
            y.append(1 if pattern.cache_hit else 0)

        return np.array(X), np.array(y)

    def _prepare_training_data_for_access_prediction(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for access time prediction."""
        X, y = [], []

        # Group patterns by query_hash to calculate next access times
        query_groups = {}
        for pattern in self.training_data:
            if pattern.query_hash not in query_groups:
                query_groups[pattern.query_hash] = []
            query_groups[pattern.query_hash].append(pattern)

        for query_hash, patterns in query_groups.items():
            # Sort by timestamp
            patterns.sort(key=lambda p: p.timestamp)

            # Create training samples
            for i in range(len(patterns) - 1):
                current_pattern = patterns[i]
                next_pattern = patterns[i + 1]

                # Time until next access (in hours)
                next_access_hours = (next_pattern.timestamp - current_pattern.timestamp).total_seconds() / 3600

                if 0.1 <= next_access_hours <= 168:  # Between 6 minutes and 1 week
                    feature_vector = self._pattern_to_feature_vector(current_pattern, PredictionType.ACCESS_TIME)
                    X.append(feature_vector)
                    y.append(next_access_hours)

        return np.array(X), np.array(y)

    def _pattern_to_feature_vector(self, pattern: QueryPattern, prediction_type: PredictionType) -> List[float]:
        """Convert QueryPattern to feature vector."""
        features = []

        # Basic features
        features.append(self._encode_query_type(pattern.query_type))
        features.append(pattern.timestamp.hour / 23.0)
        features.append(pattern.timestamp.weekday() / 6.0)
        features.append(float(pattern.timestamp.weekday() >= 5))
        features.append(pattern.response_size / 1000000.0)  # Normalize to MB
        features.append(min(pattern.processing_time / 10.0, 1.0))  # Cap at 10 seconds

        # Query-specific features
        if pattern.query_features:
            for key in sorted(pattern.query_features.keys()):
                features.append(pattern.query_features[key])

        # User context features
        if pattern.user_context:
            features.append(pattern.user_context.get('session_length', 0.0) / 3600.0)  # Hours
            features.append(float(pattern.user_context.get('is_repeat_user', False)))
        else:
            features.extend([0.0, 0.0])

        return features

    def _prepare_feature_vector(self, features: Dict[str, float], prediction_type: PredictionType) -> List[float]:
        """Prepare feature vector from feature dictionary."""
        # Define feature order for consistency
        feature_order = [
            'query_type_encoded', 'query_complexity', 'hour_of_day', 'day_of_week',
            'is_weekend', 'access_frequency', 'avg_response_time', 'last_access_hours',
            'user_hit_rate', 'user_query_freq', 'cpu_usage', 'memory_usage', 'cache_usage'
        ]

        if prediction_type == PredictionType.ACCESS_TIME:
            feature_order.extend([
                'most_frequent_hour', 'most_frequent_day', 'avg_access_interval',
                'access_regularity', 'user_hour_preference'
            ])

        vector = []
        for feature_name in feature_order:
            vector.append(features.get(feature_name, 0.0))

        # Add custom features
        for key, value in features.items():
            if key.startswith('custom_'):
                vector.append(value)

        return vector

    def _encode_query_type(self, query_type: str) -> float:
        """Encode query type as float."""
        type_mapping = {
            'osint': 0.1,
            'intelligence': 0.2,
            'blockchain': 0.3,
            'network': 0.4,
            'social': 0.5,
            'reputation': 0.6,
            'economic': 0.7,
            'tactical': 0.8,
            'fusion': 0.9,
            'unknown': 0.0
        }
        return type_mapping.get(query_type.lower(), 0.0)

    def _calculate_prediction_confidence(self, features: Dict[str, float], model: Any) -> float:
        """Calculate confidence score for prediction."""
        confidence = 0.8  # Base confidence

        # Adjust based on feature quality
        if features.get('access_frequency', 0) > 0.1:
            confidence += 0.1

        if features.get('last_access_hours', 999) < 24:
            confidence += 0.1

        # Adjust based on model type
        if hasattr(model, 'feature_importances_'):
            # More confidence if important features are present
            important_features = ['access_frequency', 'user_hit_rate', 'avg_response_time']
            for feature in important_features:
                if feature in features and features[feature] > 0:
                    confidence += 0.02

        return min(1.0, confidence)

    async def _heuristic_hit_prediction(self, query_hash: str, query_type: str,
                                       user_id: Optional[str]) -> PredictionResult:
        """Fallback heuristic prediction when ML model is not available."""
        # Simple heuristic based on recent access patterns
        recent_data = await self._get_query_history(query_hash, hours=24)

        if not recent_data:
            hit_probability = 0.1  # Low probability for new queries
        else:
            # Higher probability for recently accessed queries
            hours_since_last = self._hours_since_last_access(recent_data)
            if hours_since_last < 1:
                hit_probability = 0.9
            elif hours_since_last < 6:
                hit_probability = 0.7
            elif hours_since_last < 24:
                hit_probability = 0.5
            else:
                hit_probability = 0.2

        return PredictionResult(
            query_hash=query_hash,
            prediction_type=PredictionType.HIT_PROBABILITY,
            predicted_value=hit_probability,
            confidence=0.6,
            model_used=ModelType.ENSEMBLE,  # Heuristic
            features_used=['recent_access'],
            timestamp=datetime.now(timezone.utc)
        )

    async def _heuristic_access_prediction(self, query_hash: str, query_type: str,
                                         user_id: Optional[str]) -> PredictionResult:
        """Fallback heuristic prediction for access time."""
        # Simple heuristic based on query type and historical patterns
        type_intervals = {
            'osint': 6.0,
            'intelligence': 12.0,
            'blockchain': 4.0,
            'network': 8.0,
            'social': 2.0,
            'reputation': 24.0,
            'economic': 24.0,
            'tactical': 1.0,
            'fusion': 6.0
        }

        predicted_hours = type_intervals.get(query_type, 12.0)

        return PredictionResult(
            query_hash=query_hash,
            prediction_type=PredictionType.ACCESS_TIME,
            predicted_value=predicted_hours,
            confidence=0.5,
            model_used=ModelType.ENSEMBLE,
            features_used=['query_type'],
            timestamp=datetime.now(timezone.utc)
        )

    async def _load_models(self):
        """Load pre-trained models from storage."""
        try:
            if self.redis_client:
                for prediction_type in PredictionType:
                    model_key = f"ml_cache:model:{prediction_type.value}"
                    scaler_key = f"ml_cache:scaler:{prediction_type.value}"

                    model_data = await self.redis_client.get(model_key)
                    scaler_data = await self.redis_client.get(scaler_key)

                    if model_data and scaler_data:
                        self.models[prediction_type] = pickle.loads(model_data.encode('latin1'))
                        self.scalers[prediction_type] = pickle.loads(scaler_data.encode('latin1'))

                        self.logger.info(f"Loaded model for {prediction_type.value}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    async def _save_models(self):
        """Save trained models to storage."""
        try:
            if self.redis_client:
                for prediction_type, model in self.models.items():
                    if model and prediction_type in self.scalers:
                        model_key = f"ml_cache:model:{prediction_type.value}"
                        scaler_key = f"ml_cache:scaler:{prediction_type.value}"

                        model_data = pickle.dumps(model).decode('latin1')
                        scaler_data = pickle.dumps(self.scalers[prediction_type]).decode('latin1')

                        await self.redis_client.set(model_key, model_data, ex=86400 * 7)  # 7 days
                        await self.redis_client.set(scaler_key, scaler_data, ex=86400 * 7)

                        self.logger.info(f"Saved model for {prediction_type.value}")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    async def _collect_training_data(self):
        """Collect training data from cache access logs."""
        try:
            # Implementation would collect data from logs, database, etc.
            # For now, we'll simulate some training data
            pass
        except Exception as e:
            self.logger.error(f"Error collecting training data: {e}")

    async def _load_training_data(self):
        """Load historical training data."""
        try:
            if self.redis_client:
                training_key = "ml_cache:training_data"
                data = await self.redis_client.get(training_key)
                if data:
                    self.training_data = pickle.loads(data.encode('latin1'))
                    self.logger.info(f"Loaded {len(self.training_data)} training samples")
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")

    async def _load_user_profiles(self):
        """Load user behavior profiles."""
        try:
            if self.redis_client:
                profile_keys = await self.redis_client.keys("ml_cache:user_profile:*")
                for key in profile_keys:
                    profile_data = await self.redis_client.get(key)
                    if profile_data:
                        profile = pickle.loads(profile_data.encode('latin1'))
                        self.user_profiles[profile.user_id] = profile

                self.logger.info(f"Loaded {len(self.user_profiles)} user profiles")
        except Exception as e:
            self.logger.error(f"Error loading user profiles: {e}")

    async def _store_user_profile(self, profile: UserBehaviorProfile):
        """Store user behavior profile."""
        try:
            if self.redis_client:
                profile_key = f"ml_cache:user_profile:{profile.user_id}"
                profile_data = pickle.dumps(profile).decode('latin1')
                await self.redis_client.set(profile_key, profile_data, ex=86400 * 30)  # 30 days
        except Exception as e:
            self.logger.error(f"Error storing user profile: {e}")

    async def _get_query_history(self, query_hash: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical data for a query."""
        try:
            if self.redis_client:
                history_key = f"cache:history:{query_hash}"
                history_data = await self.redis_client.lrange(history_key, 0, -1)

                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

                filtered_data = []
                for data in history_data:
                    record = json.loads(data)
                    record_time = datetime.fromisoformat(record['timestamp'])
                    if record_time >= cutoff_time:
                        filtered_data.append(record)

                return filtered_data
        except Exception as e:
            self.logger.error(f"Error getting query history: {e}")

        return []

    def _hours_since_last_access(self, history_data: List[Dict[str, Any]]) -> float:
        """Calculate hours since last access."""
        if not history_data:
            return 999.0

        latest_time = max(datetime.fromisoformat(d['timestamp']) for d in history_data)
        return (datetime.now(timezone.utc) - latest_time).total_seconds() / 3600

    async def _get_system_load(self) -> Dict[str, float]:
        """Get current system load metrics."""
        try:
            if self.redis_client:
                load_key = "system:load"
                load_data = await self.redis_client.get(load_key)
                if load_data:
                    return json.loads(load_data)
        except Exception as e:
            self.logger.error(f"Error getting system load: {e}")

        return {'cpu_usage': 50.0, 'memory_usage': 60.0, 'cache_usage': 40.0}

    async def _get_user_historical_data(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's historical query data."""
        try:
            if self.redis_client:
                user_key = f"user:history:{user_id}"
                user_data = await self.redis_client.lrange(user_key, 0, -1)
                return [json.loads(data) for data in user_data]
        except Exception as e:
            self.logger.error(f"Error getting user history: {e}")

        return []

    def _create_default_user_profile(self, user_id: str) -> UserBehaviorProfile:
        """Create default user profile for new users."""
        return UserBehaviorProfile(
            user_id=user_id,
            query_frequency={},
            access_patterns={},
            preferred_data_types=[],
            cache_hit_rate=0.5,
            last_updated=datetime.now(timezone.utc)
        )

    def _analyze_query_frequency(self, user_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze user's query frequency patterns."""
        frequency = {}
        for record in user_data:
            query_type = record.get('query_type', 'unknown')
            frequency[query_type] = frequency.get(query_type, 0) + 1

        # Normalize frequencies
        total = sum(frequency.values())
        if total > 0:
            frequency = {k: v / total for k, v in frequency.items()}

        return frequency

    def _analyze_access_patterns(self, user_data: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Analyze user's time-based access patterns."""
        patterns = {}

        for record in user_data:
            query_type = record.get('query_type', 'unknown')
            timestamp = datetime.fromisoformat(record['timestamp'])
            hour = timestamp.hour

            if query_type not in patterns:
                patterns[query_type] = [0] * 24

            patterns[query_type][hour] += 1

        return patterns

    def _analyze_preferred_data_types(self, user_data: List[Dict[str, Any]]) -> List[str]:
        """Analyze user's preferred data types."""
        type_counts = {}
        for record in user_data:
            data_type = record.get('data_type', 'unknown')
            type_counts[data_type] = type_counts.get(data_type, 0) + 1

        # Return top 5 preferred types
        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_types[:5]]

    def _calculate_user_hit_rate(self, user_data: List[Dict[str, Any]]) -> float:
        """Calculate user's cache hit rate."""
        if not user_data:
            return 0.5

        hits = sum(1 for record in user_data if record.get('cache_hit', False))
        return hits / len(user_data)

    async def run_training_loop(self):
        """Run periodic model training loop."""
        self.logger.info(f"Starting ML training loop (interval: {self.retrain_interval} hours)")

        while True:
            try:
                await asyncio.sleep(self.retrain_interval * 3600)  # Convert to seconds

                self.logger.info("Starting scheduled model training")
                accuracies = await self.train_models()

                # Check model quality
                for prediction_type, accuracy in accuracies.items():
                    if accuracy < self.model_accuracy_threshold:
                        self.logger.warning(
                            f"Low accuracy for {prediction_type.value}: {accuracy:.3f} < {self.model_accuracy_threshold}"
                        )

                self.logger.info("Scheduled model training completed")

            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry


async def main():
    """Main entry point for ML predictor."""
    config = {
        "redis_url": "redis://redis:6379/12",
        "postgres_uri": "postgresql://user:pass@postgres:5432/bev_osint",
        "min_training_samples": 1000,
        "retrain_interval_hours": 6,
        "model_accuracy_threshold": 0.8
    }

    predictor = MLPredictor(config)

    try:
        await predictor.initialize()
        await predictor.run_training_loop()
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await predictor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())