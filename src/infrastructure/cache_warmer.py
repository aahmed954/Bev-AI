"""
BEV OSINT Framework - Intelligent Cache Warming System
Advanced cache warming with ML-driven prefetching and user behavior analysis.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import aiohttp
import psycopg2.pool
from prometheus_client import Gauge, Counter, Histogram
import numpy as np
from collections import defaultdict

from .ml_predictor import MLPredictor, PredictionType, UserBehaviorProfile
from .predictive_cache import PredictiveCache, CacheTier


class WarmingStrategy(Enum):
    USER_BASED = "user_based"
    TEMPORAL_BASED = "temporal_based"
    POPULARITY_BASED = "popularity_based"
    ML_PREDICTED = "ml_predicted"
    COLLABORATIVE = "collaborative"


class WarmingPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class WarmingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WarmingTask:
    """Represents a cache warming task."""
    task_id: str
    strategy: WarmingStrategy
    priority: WarmingPriority
    cache_key: str
    query_params: Dict[str, Any]
    user_id: Optional[str]
    query_type: str
    predicted_hit_rate: float
    estimated_size: int
    target_tier: CacheTier
    created_at: datetime
    scheduled_at: datetime
    status: WarmingStatus = WarmingStatus.PENDING
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class WarmingMetrics:
    """Cache warming performance metrics."""
    total_tasks_created: int
    tasks_completed: int
    tasks_failed: int
    tasks_skipped: int
    total_warming_time: float
    avg_task_time: float
    hit_rate_improvement: float
    prefetch_accuracy: float
    bandwidth_used: int
    cache_space_used: int
    timestamp: datetime


class CacheWarmer:
    """
    Intelligent cache warming system with ML-driven prefetching.
    Analyzes user patterns and proactively loads likely accessed data.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_url = config.get("redis_url", "redis://redis:6379/14")
        self.postgres_uri = config.get("postgres_uri")

        # Warming configuration
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 10)
        self.warming_interval = config.get("warming_interval_seconds", 300)  # 5 minutes
        self.user_analysis_window = config.get("user_analysis_window_hours", 24)
        self.popularity_threshold = config.get("popularity_threshold", 0.1)
        self.max_warming_bandwidth = config.get("max_warming_bandwidth_mbps", 100)

        # Task management
        self.pending_tasks: Dict[str, WarmingTask] = {}
        self.active_tasks: Dict[str, WarmingTask] = {}
        self.completed_tasks: List[WarmingTask] = []
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

        # User behavior tracking
        self.user_behaviors: Dict[str, Dict[str, Any]] = {}
        self.query_popularity: Dict[str, Dict[str, float]] = {}  # {query_type: {key: score}}
        self.temporal_patterns: Dict[str, List[float]] = {}  # {query_type: [hourly_scores]}

        # External connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

        # ML predictor and cache references
        self.ml_predictor: Optional[MLPredictor] = None
        self.cache_system: Optional[PredictiveCache] = None

        # Data source handlers
        self.data_handlers: Dict[str, Callable] = {}

        # Metrics
        self.metrics = WarmingMetrics(
            total_tasks_created=0,
            tasks_completed=0,
            tasks_failed=0,
            tasks_skipped=0,
            total_warming_time=0.0,
            avg_task_time=0.0,
            hit_rate_improvement=0.0,
            prefetch_accuracy=0.0,
            bandwidth_used=0,
            cache_space_used=0,
            timestamp=datetime.now(timezone.utc)
        )

        # Prometheus metrics
        self._init_prometheus_metrics()

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()

        # Logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for cache warmer."""
        logger = logging.getLogger('cache_warmer')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for cache warming."""
        self.prom_warming_tasks = Counter(
            'bev_cache_warming_tasks_total',
            'Total cache warming tasks',
            ['strategy', 'status']
        )
        self.prom_warming_time = Histogram(
            'bev_cache_warming_time_seconds',
            'Cache warming task duration',
            ['strategy']
        )
        self.prom_prefetch_accuracy = Gauge(
            'bev_cache_prefetch_accuracy',
            'Cache prefetch accuracy rate'
        )
        self.prom_warming_bandwidth = Gauge(
            'bev_cache_warming_bandwidth_bytes_per_second',
            'Cache warming bandwidth usage'
        )
        self.prom_pending_tasks = Gauge(
            'bev_cache_warming_pending_tasks',
            'Number of pending warming tasks'
        )
        self.prom_hit_improvement = Gauge(
            'bev_cache_hit_rate_improvement',
            'Cache hit rate improvement from warming'
        )

    async def initialize(self):
        """Initialize cache warming system."""
        try:
            # Initialize Redis connection
            self.redis_client = await aioredis.from_url(
                self.redis_url,
                decode_responses=True
            )

            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )

            # Initialize database connection pool
            if self.postgres_uri:
                self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=10,
                    dsn=self.postgres_uri
                )

            # Register default data handlers
            self._register_default_data_handlers()

            # Load existing state
            await self._load_warming_state()

            # Start background tasks
            await self._start_background_tasks()

            self.logger.info("Cache warming system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize cache warmer: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown of cache warming system."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Save state
        await self._save_warming_state()

        # Close connections
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.db_pool:
            self.db_pool.closeall()

        self.logger.info("Cache warmer shutdown completed")

    def set_ml_predictor(self, predictor: MLPredictor):
        """Set ML predictor reference."""
        self.ml_predictor = predictor

    def set_cache_system(self, cache_system: PredictiveCache):
        """Set cache system reference."""
        self.cache_system = cache_system

    def register_data_handler(self, query_type: str, handler: Callable):
        """Register data handler for specific query type."""
        self.data_handlers[query_type] = handler
        self.logger.info(f"Registered data handler for {query_type}")

    async def create_warming_task(self, strategy: WarmingStrategy, cache_key: str,
                                query_params: Dict[str, Any], query_type: str,
                                user_id: Optional[str] = None,
                                priority: WarmingPriority = WarmingPriority.MEDIUM,
                                scheduled_at: Optional[datetime] = None) -> str:
        """
        Create a new cache warming task.

        Args:
            strategy: Warming strategy to use
            cache_key: Cache key to warm
            query_params: Parameters for data retrieval
            query_type: Type of query
            user_id: Optional user ID for personalization
            priority: Task priority
            scheduled_at: Optional scheduled execution time

        Returns:
            Task ID
        """
        try:
            task_id = f"warm_{int(time.time() * 1000)}_{hash(cache_key) % 10000}"

            # Get ML prediction for hit rate
            predicted_hit_rate = 0.5  # Default
            if self.ml_predictor and user_id:
                try:
                    prediction = await self.ml_predictor.predict_cache_hit(
                        cache_key, query_type, user_id
                    )
                    predicted_hit_rate = prediction.predicted_value
                except Exception as e:
                    self.logger.warning(f"Failed to get ML prediction: {e}")

            # Estimate data size
            estimated_size = await self._estimate_data_size(query_type, query_params)

            # Determine target tier
            target_tier = await self._determine_target_tier(
                predicted_hit_rate, estimated_size, priority
            )

            task = WarmingTask(
                task_id=task_id,
                strategy=strategy,
                priority=priority,
                cache_key=cache_key,
                query_params=query_params,
                user_id=user_id,
                query_type=query_type,
                predicted_hit_rate=predicted_hit_rate,
                estimated_size=estimated_size,
                target_tier=target_tier,
                created_at=datetime.now(timezone.utc),
                scheduled_at=scheduled_at or datetime.now(timezone.utc)
            )

            self.pending_tasks[task_id] = task
            self.metrics.total_tasks_created += 1

            # Update Prometheus metrics
            self.prom_warming_tasks.labels(
                strategy=strategy.value,
                status='created'
            ).inc()

            self.logger.debug(f"Created warming task {task_id} for {cache_key}")

            return task_id

        except Exception as e:
            self.logger.error(f"Error creating warming task: {e}")
            raise

    async def analyze_user_patterns(self, user_id: str) -> List[str]:
        """
        Analyze user patterns and generate warming tasks.

        Args:
            user_id: User to analyze

        Returns:
            List of created task IDs
        """
        try:
            if not self.ml_predictor:
                self.logger.warning("ML predictor not available for user analysis")
                return []

            # Get user behavior profile
            profile = await self.ml_predictor.analyze_user_patterns(user_id)

            task_ids = []

            # Create tasks based on user's frequent query types
            for query_type, frequency in profile.query_frequency.items():
                if frequency > 0.1:  # 10% threshold
                    # Generate warming tasks for this query type
                    warming_keys = await self._generate_user_warming_keys(
                        user_id, query_type, profile
                    )

                    for cache_key, query_params in warming_keys:
                        task_id = await self.create_warming_task(
                            strategy=WarmingStrategy.USER_BASED,
                            cache_key=cache_key,
                            query_params=query_params,
                            query_type=query_type,
                            user_id=user_id,
                            priority=WarmingPriority.HIGH if frequency > 0.3 else WarmingPriority.MEDIUM
                        )
                        task_ids.append(task_id)

            # Create tasks based on user's access patterns
            current_hour = datetime.now(timezone.utc).hour
            for query_type, hourly_patterns in profile.access_patterns.items():
                if len(hourly_patterns) > current_hour and hourly_patterns[current_hour] > 0:
                    # User is likely to be active now
                    temporal_keys = await self._generate_temporal_warming_keys(
                        user_id, query_type, current_hour
                    )

                    for cache_key, query_params in temporal_keys:
                        task_id = await self.create_warming_task(
                            strategy=WarmingStrategy.TEMPORAL_BASED,
                            cache_key=cache_key,
                            query_params=query_params,
                            query_type=query_type,
                            user_id=user_id,
                            priority=WarmingPriority.HIGH
                        )
                        task_ids.append(task_id)

            self.logger.info(f"Created {len(task_ids)} warming tasks for user {user_id}")

            return task_ids

        except Exception as e:
            self.logger.error(f"Error analyzing user patterns for {user_id}: {e}")
            return []

    async def analyze_popularity_trends(self) -> List[str]:
        """
        Analyze popular queries and create warming tasks.

        Returns:
            List of created task IDs
        """
        try:
            # Analyze query popularity from recent cache access logs
            popular_queries = await self._analyze_query_popularity()

            task_ids = []

            for query_info in popular_queries:
                query_type = query_info['query_type']
                cache_key = query_info['cache_key']
                popularity_score = query_info['popularity_score']
                query_params = query_info.get('query_params', {})

                if popularity_score > self.popularity_threshold:
                    priority = (
                        WarmingPriority.CRITICAL if popularity_score > 0.8 else
                        WarmingPriority.HIGH if popularity_score > 0.5 else
                        WarmingPriority.MEDIUM
                    )

                    task_id = await self.create_warming_task(
                        strategy=WarmingStrategy.POPULARITY_BASED,
                        cache_key=cache_key,
                        query_params=query_params,
                        query_type=query_type,
                        priority=priority
                    )
                    task_ids.append(task_id)

            self.logger.info(f"Created {len(task_ids)} popularity-based warming tasks")

            return task_ids

        except Exception as e:
            self.logger.error(f"Error analyzing popularity trends: {e}")
            return []

    async def create_collaborative_warming_tasks(self) -> List[str]:
        """
        Create warming tasks based on collaborative filtering.

        Returns:
            List of created task IDs
        """
        try:
            # Find users with similar behavior patterns
            user_similarities = await self._calculate_user_similarities()

            task_ids = []

            for user_id, similar_users in user_similarities.items():
                # Get successful cache keys from similar users
                recommended_keys = await self._get_collaborative_recommendations(
                    user_id, similar_users
                )

                for recommendation in recommended_keys:
                    task_id = await self.create_warming_task(
                        strategy=WarmingStrategy.COLLABORATIVE,
                        cache_key=recommendation['cache_key'],
                        query_params=recommendation['query_params'],
                        query_type=recommendation['query_type'],
                        user_id=user_id,
                        priority=WarmingPriority.MEDIUM
                    )
                    task_ids.append(task_id)

            self.logger.info(f"Created {len(task_ids)} collaborative warming tasks")

            return task_ids

        except Exception as e:
            self.logger.error(f"Error creating collaborative warming tasks: {e}")
            return []

    async def execute_warming_task(self, task_id: str) -> bool:
        """
        Execute a specific warming task.

        Args:
            task_id: Task to execute

        Returns:
            True if successful
        """
        async with self.task_semaphore:
            task = self.pending_tasks.get(task_id)
            if not task:
                self.logger.warning(f"Task {task_id} not found")
                return False

            # Move to active tasks
            self.active_tasks[task_id] = task
            del self.pending_tasks[task_id]

            task.status = WarmingStatus.IN_PROGRESS
            task.last_attempt = datetime.now(timezone.utc)
            task.attempts += 1

            start_time = time.time()

            try:
                # Check if already cached
                if self.cache_system:
                    existing_value = await self.cache_system.get(
                        task.cache_key, task.user_id, task.query_type
                    )
                    if existing_value is not None:
                        task.status = WarmingStatus.SKIPPED
                        self.metrics.tasks_skipped += 1
                        await self._complete_task(task_id, task)
                        return True

                # Execute data retrieval
                data = await self._retrieve_data(task)

                if data is not None:
                    # Store in cache
                    if self.cache_system:
                        success = await self.cache_system.set(
                            key=task.cache_key,
                            value=data,
                            ttl=self._calculate_warming_ttl(task),
                            user_id=task.user_id,
                            query_type=task.query_type,
                            size_hint=task.estimated_size
                        )

                        if success:
                            task.status = WarmingStatus.COMPLETED
                            self.metrics.tasks_completed += 1
                            self.metrics.cache_space_used += task.estimated_size

                            execution_time = time.time() - start_time
                            self.metrics.total_warming_time += execution_time
                            self.metrics.avg_task_time = (
                                self.metrics.total_warming_time /
                                max(1, self.metrics.tasks_completed)
                            )

                            # Update Prometheus metrics
                            self.prom_warming_time.labels(
                                strategy=task.strategy.value
                            ).observe(execution_time)

                            self.logger.debug(f"Successfully warmed cache for {task.cache_key}")
                        else:
                            task.status = WarmingStatus.FAILED
                            task.error_message = "Failed to store in cache"
                            self.metrics.tasks_failed += 1
                    else:
                        task.status = WarmingStatus.FAILED
                        task.error_message = "Cache system not available"
                        self.metrics.tasks_failed += 1
                else:
                    task.status = WarmingStatus.FAILED
                    task.error_message = "Failed to retrieve data"
                    self.metrics.tasks_failed += 1

                await self._complete_task(task_id, task)

                # Update Prometheus metrics
                self.prom_warming_tasks.labels(
                    strategy=task.strategy.value,
                    status=task.status.value
                ).inc()

                return task.status == WarmingStatus.COMPLETED

            except Exception as e:
                task.status = WarmingStatus.FAILED
                task.error_message = str(e)
                self.metrics.tasks_failed += 1

                self.logger.error(f"Error executing warming task {task_id}: {e}")

                await self._complete_task(task_id, task)

                self.prom_warming_tasks.labels(
                    strategy=task.strategy.value,
                    status='failed'
                ).inc()

                return False

    async def _retrieve_data(self, task: WarmingTask) -> Optional[Any]:
        """Retrieve data for warming task."""
        try:
            # Get appropriate data handler
            handler = self.data_handlers.get(task.query_type)
            if not handler:
                self.logger.warning(f"No data handler for query type: {task.query_type}")
                return None

            # Execute data retrieval
            data = await handler(task.query_params)

            # Update bandwidth metrics
            if data:
                data_size = len(json.dumps(data, default=str).encode('utf-8'))
                self.metrics.bandwidth_used += data_size

            return data

        except Exception as e:
            self.logger.error(f"Error retrieving data for task {task.task_id}: {e}")
            return None

    async def _complete_task(self, task_id: str, task: WarmingTask):
        """Complete warming task and cleanup."""
        task.completion_time = datetime.now(timezone.utc)

        # Move to completed tasks
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]

        self.completed_tasks.append(task)

        # Keep only recent completed tasks
        if len(self.completed_tasks) > 1000:
            self.completed_tasks = self.completed_tasks[-1000:]

        # Store task result for analytics
        await self._store_task_result(task)

    async def _estimate_data_size(self, query_type: str, query_params: Dict[str, Any]) -> int:
        """Estimate data size for query."""
        # Simple heuristic-based estimation
        base_sizes = {
            'osint': 10 * 1024,      # 10KB
            'intelligence': 50 * 1024,   # 50KB
            'blockchain': 5 * 1024,      # 5KB
            'network': 20 * 1024,        # 20KB
            'social': 15 * 1024,         # 15KB
            'reputation': 8 * 1024,      # 8KB
            'economic': 30 * 1024,       # 30KB
            'tactical': 100 * 1024,      # 100KB
            'fusion': 200 * 1024         # 200KB
        }

        base_size = base_sizes.get(query_type, 20 * 1024)

        # Adjust based on query complexity
        complexity_factor = 1.0
        if 'limit' in query_params:
            complexity_factor *= query_params['limit'] / 100.0
        if 'depth' in query_params:
            complexity_factor *= query_params['depth']

        return int(base_size * complexity_factor)

    async def _determine_target_tier(self, predicted_hit_rate: float,
                                   estimated_size: int, priority: WarmingPriority) -> CacheTier:
        """Determine target cache tier for warming."""
        # High hit rate and high priority -> Hot tier
        if predicted_hit_rate > 0.7 and priority in [WarmingPriority.CRITICAL, WarmingPriority.HIGH]:
            return CacheTier.HOT

        # Medium hit rate or medium priority -> Warm tier
        elif predicted_hit_rate > 0.4 or priority == WarmingPriority.MEDIUM:
            return CacheTier.WARM

        # Low hit rate or large size -> Cold tier
        else:
            return CacheTier.COLD

    def _calculate_warming_ttl(self, task: WarmingTask) -> int:
        """Calculate TTL for warmed cache entry."""
        base_ttl = 3600  # 1 hour

        # Adjust based on strategy
        if task.strategy == WarmingStrategy.USER_BASED:
            base_ttl = 7200  # 2 hours for user-specific
        elif task.strategy == WarmingStrategy.POPULARITY_BASED:
            base_ttl = 14400  # 4 hours for popular items
        elif task.strategy == WarmingStrategy.TEMPORAL_BASED:
            base_ttl = 1800   # 30 minutes for temporal
        elif task.strategy == WarmingStrategy.ML_PREDICTED:
            # Use ML prediction for TTL
            if self.ml_predictor and task.user_id:
                try:
                    # This would use access time prediction
                    base_ttl = int(task.predicted_hit_rate * 7200)
                except:
                    pass

        # Adjust based on priority
        priority_multipliers = {
            WarmingPriority.CRITICAL: 2.0,
            WarmingPriority.HIGH: 1.5,
            WarmingPriority.MEDIUM: 1.0,
            WarmingPriority.LOW: 0.5
        }

        return int(base_ttl * priority_multipliers.get(task.priority, 1.0))

    async def _generate_user_warming_keys(self, user_id: str, query_type: str,
                                        profile: UserBehaviorProfile) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate cache keys for user-based warming."""
        keys = []

        # Generate keys based on user's preferred data types
        for data_type in profile.preferred_data_types[:3]:  # Top 3 preferences
            cache_key = f"{query_type}:{user_id}:{data_type}:recent"
            query_params = {
                "user_id": user_id,
                "data_type": data_type,
                "query_type": query_type,
                "limit": 50
            }
            keys.append((cache_key, query_params))

        return keys

    async def _generate_temporal_warming_keys(self, user_id: str, query_type: str,
                                            hour: int) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate cache keys for temporal-based warming."""
        keys = []

        # Generate time-specific keys
        time_contexts = ["hourly", "recent", "trending"]
        for context in time_contexts:
            cache_key = f"{query_type}:{context}:{hour}:{user_id}"
            query_params = {
                "user_id": user_id,
                "query_type": query_type,
                "time_context": context,
                "hour": hour,
                "limit": 20
            }
            keys.append((cache_key, query_params))

        return keys

    async def _analyze_query_popularity(self) -> List[Dict[str, Any]]:
        """Analyze query popularity from cache access logs."""
        try:
            if not self.redis_client:
                return []

            # Get recent cache events
            events = await self.redis_client.lrange("cache:events", 0, 999)

            # Count query frequencies
            query_counts = defaultdict(int)
            total_queries = 0

            for event_data in events:
                try:
                    event = json.loads(event_data)
                    if event.get('event_type') == 'hit':
                        key = event.get('key', '')
                        query_type = key.split(':')[0] if ':' in key else 'unknown'
                        query_counts[key] += 1
                        total_queries += 1
                except:
                    continue

            # Calculate popularity scores
            popular_queries = []
            for cache_key, count in query_counts.items():
                if count > 1:  # Only consider queries with multiple hits
                    popularity_score = count / total_queries if total_queries > 0 else 0
                    query_type = cache_key.split(':')[0] if ':' in cache_key else 'unknown'

                    popular_queries.append({
                        'cache_key': cache_key,
                        'query_type': query_type,
                        'popularity_score': popularity_score,
                        'hit_count': count,
                        'query_params': {}  # Would extract from key structure
                    })

            # Sort by popularity
            popular_queries.sort(key=lambda x: x['popularity_score'], reverse=True)

            return popular_queries[:50]  # Top 50

        except Exception as e:
            self.logger.error(f"Error analyzing query popularity: {e}")
            return []

    async def _calculate_user_similarities(self) -> Dict[str, List[str]]:
        """Calculate user similarity for collaborative filtering."""
        try:
            # This would implement user similarity calculation
            # For now, return empty dict
            return {}

        except Exception as e:
            self.logger.error(f"Error calculating user similarities: {e}")
            return {}

    async def _get_collaborative_recommendations(self, user_id: str,
                                               similar_users: List[str]) -> List[Dict[str, Any]]:
        """Get collaborative filtering recommendations."""
        try:
            # This would implement collaborative filtering recommendations
            # For now, return empty list
            return []

        except Exception as e:
            self.logger.error(f"Error getting collaborative recommendations: {e}")
            return []

    def _register_default_data_handlers(self):
        """Register default data handlers for common query types."""
        async def osint_handler(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # Mock OSINT data handler
            return {"type": "osint", "data": f"mock_osint_data_{params.get('user_id', 'anon')}"}

        async def intelligence_handler(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # Mock intelligence data handler
            return {"type": "intelligence", "data": f"mock_intel_data_{params.get('data_type', 'general')}"}

        async def blockchain_handler(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # Mock blockchain data handler
            return {"type": "blockchain", "data": f"mock_blockchain_data_{params.get('limit', 10)}"}

        # Register handlers
        self.data_handlers['osint'] = osint_handler
        self.data_handlers['intelligence'] = intelligence_handler
        self.data_handlers['blockchain'] = blockchain_handler

    async def _store_task_result(self, task: WarmingTask):
        """Store task result for analytics."""
        try:
            if self.redis_client:
                task_data = {
                    "task_id": task.task_id,
                    "strategy": task.strategy.value,
                    "status": task.status.value,
                    "query_type": task.query_type,
                    "predicted_hit_rate": task.predicted_hit_rate,
                    "estimated_size": task.estimated_size,
                    "execution_time": (
                        (task.completion_time - task.last_attempt).total_seconds()
                        if task.completion_time and task.last_attempt else 0
                    ),
                    "timestamp": task.completion_time.isoformat() if task.completion_time else None
                }

                await self.redis_client.lpush(
                    "warming:results",
                    json.dumps(task_data)
                )
                await self.redis_client.ltrim("warming:results", 0, 999)  # Keep last 1000

        except Exception as e:
            self.logger.error(f"Error storing task result: {e}")

    async def _start_background_tasks(self):
        """Start background tasks for cache warming."""
        tasks = [
            self._warming_scheduler_loop(),
            self._metrics_update_loop(),
            self._task_cleanup_loop(),
            self._pattern_analysis_loop()
        ]

        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _warming_scheduler_loop(self):
        """Main warming scheduler loop."""
        while True:
            try:
                await asyncio.sleep(self.warming_interval)

                # Analyze and create warming tasks
                await self._scheduled_analysis()

                # Execute pending tasks
                await self._execute_pending_tasks()

                # Update metrics
                self.prom_pending_tasks.set(len(self.pending_tasks))

            except Exception as e:
                self.logger.error(f"Error in warming scheduler: {e}")
                await asyncio.sleep(self.warming_interval)

    async def _scheduled_analysis(self):
        """Perform scheduled analysis and create warming tasks."""
        try:
            # Popularity-based warming
            await self.analyze_popularity_trends()

            # User-based warming for active users
            active_users = await self._get_active_users()
            for user_id in active_users[:10]:  # Limit to 10 users per cycle
                await self.analyze_user_patterns(user_id)

            # Collaborative filtering (less frequent)
            import random
            if random.random() < 0.1:  # 10% chance
                await self.create_collaborative_warming_tasks()

        except Exception as e:
            self.logger.error(f"Error in scheduled analysis: {e}")

    async def _execute_pending_tasks(self):
        """Execute pending warming tasks."""
        try:
            # Sort tasks by priority and scheduled time
            sorted_tasks = sorted(
                self.pending_tasks.items(),
                key=lambda x: (
                    x[1].priority.value,
                    x[1].scheduled_at
                )
            )

            # Execute tasks up to concurrency limit
            available_slots = self.max_concurrent_tasks - len(self.active_tasks)
            tasks_to_execute = sorted_tasks[:available_slots]

            execution_tasks = []
            for task_id, task in tasks_to_execute:
                if task.scheduled_at <= datetime.now(timezone.utc):
                    execution_tasks.append(
                        asyncio.create_task(self.execute_warming_task(task_id))
                    )

            if execution_tasks:
                await asyncio.gather(*execution_tasks, return_exceptions=True)

        except Exception as e:
            self.logger.error(f"Error executing pending tasks: {e}")

    async def _metrics_update_loop(self):
        """Update metrics and monitoring data."""
        while True:
            try:
                await asyncio.sleep(60)  # 1 minute

                # Calculate prefetch accuracy
                if self.metrics.tasks_completed > 0:
                    # This would analyze actual cache hits vs predictions
                    # For now, use a simple calculation
                    accuracy = min(0.95, self.metrics.tasks_completed /
                                 max(1, self.metrics.total_tasks_created))
                    self.metrics.prefetch_accuracy = accuracy
                    self.prom_prefetch_accuracy.set(accuracy)

                # Update bandwidth metrics
                bandwidth_mbps = self.metrics.bandwidth_used / (1024 * 1024)
                self.prom_warming_bandwidth.set(bandwidth_mbps)

                # Update hit rate improvement
                # This would compare hit rates before/after warming
                self.prom_hit_improvement.set(self.metrics.hit_rate_improvement)

                self.logger.debug(f"Updated warming metrics: {self.metrics.tasks_completed} completed")

            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(60)

    async def _task_cleanup_loop(self):
        """Clean up old completed tasks."""
        while True:
            try:
                await asyncio.sleep(3600)  # 1 hour

                # Remove old completed tasks
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                self.completed_tasks = [
                    task for task in self.completed_tasks
                    if task.completion_time and task.completion_time >= cutoff_time
                ]

                # Clean up failed tasks that are too old
                old_tasks = [
                    task_id for task_id, task in self.pending_tasks.items()
                    if task.created_at < cutoff_time and task.attempts > 3
                ]

                for task_id in old_tasks:
                    del self.pending_tasks[task_id]
                    self.metrics.tasks_failed += 1

                self.logger.debug(f"Cleaned up {len(old_tasks)} old tasks")

            except Exception as e:
                self.logger.error(f"Error in task cleanup: {e}")
                await asyncio.sleep(3600)

    async def _pattern_analysis_loop(self):
        """Analyze patterns for optimization."""
        while True:
            try:
                await asyncio.sleep(1800)  # 30 minutes

                # Analyze warming effectiveness
                await self._analyze_warming_effectiveness()

                # Update temporal patterns
                await self._update_temporal_patterns()

            except Exception as e:
                self.logger.error(f"Error in pattern analysis: {e}")
                await asyncio.sleep(1800)

    async def _analyze_warming_effectiveness(self):
        """Analyze how effective warming strategies are."""
        try:
            # Analyze completed tasks by strategy
            strategy_stats = defaultdict(lambda: {"total": 0, "successful": 0})

            for task in self.completed_tasks[-100:]:  # Last 100 tasks
                strategy_stats[task.strategy.value]["total"] += 1
                if task.status == WarmingStatus.COMPLETED:
                    strategy_stats[task.strategy.value]["successful"] += 1

            # Calculate effectiveness
            for strategy, stats in strategy_stats.items():
                if stats["total"] > 0:
                    effectiveness = stats["successful"] / stats["total"]
                    self.logger.info(f"Strategy {strategy} effectiveness: {effectiveness:.2f}")

        except Exception as e:
            self.logger.error(f"Error analyzing warming effectiveness: {e}")

    async def _update_temporal_patterns(self):
        """Update temporal access patterns."""
        try:
            # Analyze cache access events by hour
            if not self.redis_client:
                return

            events = await self.redis_client.lrange("cache:events", 0, 999)
            hourly_patterns = defaultdict(lambda: [0] * 24)

            for event_data in events:
                try:
                    event = json.loads(event_data)
                    if event.get('event_type') == 'hit':
                        timestamp = datetime.fromisoformat(event['timestamp'])
                        hour = timestamp.hour
                        key = event.get('key', '')
                        query_type = key.split(':')[0] if ':' in key else 'unknown'
                        hourly_patterns[query_type][hour] += 1
                except:
                    continue

            # Update patterns
            self.temporal_patterns = dict(hourly_patterns)

        except Exception as e:
            self.logger.error(f"Error updating temporal patterns: {e}")

    async def _get_active_users(self) -> List[str]:
        """Get list of currently active users."""
        try:
            if not self.redis_client:
                return []

            # Get recent cache events to identify active users
            events = await self.redis_client.lrange("cache:events", 0, 199)
            active_users = set()

            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)

            for event_data in events:
                try:
                    event = json.loads(event_data)
                    event_time = datetime.fromisoformat(event['timestamp'])
                    if event_time >= cutoff_time and event.get('user_id'):
                        active_users.add(event['user_id'])
                except:
                    continue

            return list(active_users)

        except Exception as e:
            self.logger.error(f"Error getting active users: {e}")
            return []

    async def _load_warming_state(self):
        """Load warming state from persistent storage."""
        try:
            if self.redis_client:
                # Load pending tasks
                tasks_data = await self.redis_client.get("warming:pending_tasks")
                if tasks_data:
                    tasks_dict = json.loads(tasks_data)
                    for task_id, task_data in tasks_dict.items():
                        # Convert datetime strings back to datetime objects
                        for key in ['created_at', 'scheduled_at', 'last_attempt', 'completion_time']:
                            if task_data.get(key):
                                task_data[key] = datetime.fromisoformat(task_data[key])

                        self.pending_tasks[task_id] = WarmingTask(**task_data)

                self.logger.info(f"Loaded {len(self.pending_tasks)} pending warming tasks")

        except Exception as e:
            self.logger.error(f"Error loading warming state: {e}")

    async def _save_warming_state(self):
        """Save warming state to persistent storage."""
        try:
            if self.redis_client:
                # Save pending tasks
                tasks_dict = {}
                for task_id, task in self.pending_tasks.items():
                    task_data = asdict(task)
                    # Convert datetime objects to strings
                    for key, value in task_data.items():
                        if isinstance(value, datetime):
                            task_data[key] = value.isoformat()

                    tasks_dict[task_id] = task_data

                await self.redis_client.set(
                    "warming:pending_tasks",
                    json.dumps(tasks_dict),
                    ex=86400  # 24 hours
                )

                self.logger.info("Saved warming state")

        except Exception as e:
            self.logger.error(f"Error saving warming state: {e}")

    async def get_warming_stats(self) -> Dict[str, Any]:
        """Get comprehensive warming statistics."""
        stats = {
            "metrics": asdict(self.metrics),
            "task_counts": {
                "pending": len(self.pending_tasks),
                "active": len(self.active_tasks),
                "completed": len(self.completed_tasks)
            },
            "strategy_distribution": {},
            "priority_distribution": {},
            "temporal_patterns": self.temporal_patterns,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Calculate strategy distribution
        for task in list(self.pending_tasks.values()) + list(self.active_tasks.values()):
            strategy = task.strategy.value
            stats["strategy_distribution"][strategy] = stats["strategy_distribution"].get(strategy, 0) + 1

        # Calculate priority distribution
        for task in list(self.pending_tasks.values()) + list(self.active_tasks.values()):
            priority = task.priority.value
            stats["priority_distribution"][priority] = stats["priority_distribution"].get(priority, 0) + 1

        return stats


async def main():
    """Main entry point for cache warmer."""
    config = {
        "redis_url": "redis://redis:6379/14",
        "postgres_uri": "postgresql://user:pass@postgres:5432/bev_osint",
        "max_concurrent_tasks": 10,
        "warming_interval_seconds": 300,
        "popularity_threshold": 0.1
    }

    warmer = CacheWarmer(config)

    try:
        await warmer.initialize()
        # Keep running (in real deployment, this would be part of a larger service)
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await warmer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())