import os
"""
BEV OSINT Framework - Predictive Cache System
ML-driven multi-tier caching with intelligent prefetching and adaptive optimization.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aioredis
import psycopg2.pool
from prometheus_client import Gauge, Counter, Histogram, Summary
import yaml

from .ml_predictor import MLPredictor, PredictionType, PredictionResult


class CacheTier(Enum):
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class CachePolicy(Enum):
    LRU = "lru"
    LFU = "lfu"
    ARC = "arc"
    ML_ADAPTIVE = "ml_adaptive"


class CacheEventType(Enum):
    HIT = "hit"
    MISS = "miss"
    EVICTION = "eviction"
    PREFETCH = "prefetch"
    WARMING = "warming"


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    tier: CacheTier
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int]
    prediction_score: float
    user_id: Optional[str] = None
    query_type: Optional[str] = None
    custom_metadata: Optional[Dict[str, Any]] = None


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    total_requests: int
    hits: int
    misses: int
    hit_rate: float
    avg_response_time: float
    hot_tier_usage: float
    warm_tier_usage: float
    cold_tier_usage: float
    evictions: int
    prefetch_hits: int
    ml_prediction_accuracy: float
    timestamp: datetime


@dataclass
class CacheConfiguration:
    """Cache configuration parameters."""
    hot_tier_size_gb: float = 4.0
    warm_tier_size_gb: float = 8.0
    cold_tier_persistent: bool = True
    default_ttl_seconds: int = 3600
    prefetch_threshold: float = 0.7
    eviction_policy: CachePolicy = CachePolicy.ML_ADAPTIVE
    ml_prediction_interval: int = 300  # 5 minutes
    enable_intelligent_warming: bool = True
    max_prefetch_batch_size: int = 100


class PredictiveCache:
    """
    Advanced predictive cache system with ML-driven optimization.
    Implements multi-tier caching with intelligent prefetching.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = CacheConfiguration(**config.get("cache", {}))
        self.redis_cluster_nodes = config.get("redis_cluster_nodes", [
            {"host": "redis-node-1", "port": 7001},
            {"host": "redis-node-2", "port": 7002},
            {"host": "redis-node-3", "port": 7003}
        ])
        self.redis_password = config.get("redis_password")
        self.postgres_uri = config.get("postgres_uri")

        # Cache storage by tier
        self.hot_cache: Dict[str, CacheEntry] = {}
        self.warm_cache: Dict[str, CacheEntry] = {}
        self.cold_cache_keys: Set[str] = set()

        # Access tracking
        self.access_patterns: Dict[str, List[datetime]] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}

        # ML predictor
        self.ml_predictor: Optional[MLPredictor] = None

        # External connections
        self.redis_cluster: Optional[aioredis.RedisCluster] = None
        self.redis_standalone: Optional[aioredis.Redis] = None
        self.db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

        # Metrics and monitoring
        self.metrics = CacheMetrics(
            total_requests=0,
            hits=0,
            misses=0,
            hit_rate=0.0,
            avg_response_time=0.0,
            hot_tier_usage=0.0,
            warm_tier_usage=0.0,
            cold_tier_usage=0.0,
            evictions=0,
            prefetch_hits=0,
            ml_prediction_accuracy=0.0,
            timestamp=datetime.now(timezone.utc)
        )

        # Prometheus metrics
        self._init_prometheus_metrics()

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()

        # Logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for predictive cache."""
        logger = logging.getLogger('predictive_cache')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for cache monitoring."""
        self.prom_cache_requests = Counter(
            'bev_cache_requests_total',
            'Total cache requests',
            ['tier', 'result']
        )
        self.prom_cache_hit_rate = Gauge(
            'bev_cache_hit_rate',
            'Cache hit rate by tier',
            ['tier']
        )
        self.prom_cache_response_time = Histogram(
            'bev_cache_response_time_seconds',
            'Cache response time',
            ['tier', 'operation']
        )
        self.prom_cache_size = Gauge(
            'bev_cache_size_bytes',
            'Cache size in bytes',
            ['tier']
        )
        self.prom_cache_entries = Gauge(
            'bev_cache_entries_total',
            'Number of cache entries',
            ['tier']
        )
        self.prom_evictions = Counter(
            'bev_cache_evictions_total',
            'Cache evictions',
            ['tier', 'reason']
        )
        self.prom_prefetch_hits = Counter(
            'bev_cache_prefetch_hits_total',
            'Prefetch hits'
        )
        self.prom_ml_prediction_accuracy = Gauge(
            'bev_cache_ml_accuracy',
            'ML prediction accuracy'
        )

    async def initialize(self):
        """Initialize predictive cache system."""
        try:
            # Initialize Redis cluster connection
            startup_nodes = [
                aioredis.from_url(f"redis://:{self.redis_password}@{node['host']}:{node['port']}")
                for node in self.redis_cluster_nodes
            ]
            self.redis_cluster = aioredis.RedisCluster(startup_nodes=startup_nodes)

            # Initialize standalone Redis for metadata
            self.redis_standalone = await aioredis.from_url(
                f"redis://:{self.redis_password}@redis:6379/13",
                decode_responses=True
            )

            # Initialize database connection pool
            if self.postgres_uri:
                self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=20,
                    dsn=self.postgres_uri
                )

            # Initialize ML predictor
            ml_config = {
                "redis_url": f"redis://:{self.redis_password}@redis:6379/12",
                "postgres_uri": self.postgres_uri,
                "min_training_samples": 500,
                "retrain_interval_hours": 4
            }
            self.ml_predictor = MLPredictor(ml_config)
            await self.ml_predictor.initialize()

            # Load existing cache state
            await self._load_cache_state()

            # Start background tasks
            await self._start_background_tasks()

            self.logger.info("Predictive cache system initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize predictive cache: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown of predictive cache."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Save cache state
        await self._save_cache_state()

        # Close connections
        if self.redis_cluster:
            await self.redis_cluster.close()
        if self.redis_standalone:
            await self.redis_standalone.close()
        if self.ml_predictor:
            await self.ml_predictor.shutdown()
        if self.db_pool:
            self.db_pool.closeall()

        self.logger.info("Predictive cache shutdown completed")

    async def get(self, key: str, user_id: Optional[str] = None,
                 query_type: Optional[str] = None) -> Optional[Any]:
        """
        Get value from cache with ML-based tier selection.

        Args:
            key: Cache key
            user_id: Optional user identifier for personalization
            query_type: Optional query type for optimization

        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        self.metrics.total_requests += 1

        try:
            # Check cache tiers in order: hot -> warm -> cold
            entry = await self._get_from_tier(key, CacheTier.HOT)
            if entry:
                await self._record_cache_event(CacheEventType.HIT, key, CacheTier.HOT, user_id)
                self.metrics.hits += 1
                self._update_access_pattern(key, user_id)
                return entry.value

            entry = await self._get_from_tier(key, CacheTier.WARM)
            if entry:
                await self._record_cache_event(CacheEventType.HIT, key, CacheTier.WARM, user_id)
                self.metrics.hits += 1
                self._update_access_pattern(key, user_id)

                # Promote to hot tier if frequently accessed
                if await self._should_promote_to_hot(entry):
                    await self._promote_entry(entry, CacheTier.HOT)

                return entry.value

            entry = await self._get_from_tier(key, CacheTier.COLD)
            if entry:
                await self._record_cache_event(CacheEventType.HIT, key, CacheTier.COLD, user_id)
                self.metrics.hits += 1
                self._update_access_pattern(key, user_id)

                # Promote based on ML prediction
                if await self._should_promote_based_on_ml(entry, user_id, query_type):
                    await self._promote_entry(entry, CacheTier.WARM)

                return entry.value

            # Cache miss
            await self._record_cache_event(CacheEventType.MISS, key, None, user_id)
            self.metrics.misses += 1

            # Trigger intelligent prefetching
            if user_id and query_type:
                await self._trigger_intelligent_prefetch(key, user_id, query_type)

            return None

        finally:
            response_time = time.time() - start_time
            self._update_response_time_metrics(response_time)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                 user_id: Optional[str] = None, query_type: Optional[str] = None,
                 size_hint: Optional[int] = None) -> bool:
        """
        Set value in cache with ML-based tier placement.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            user_id: Optional user identifier
            query_type: Optional query type
            size_hint: Optional size hint in bytes

        Returns:
            True if successfully cached
        """
        try:
            # Calculate size
            if size_hint:
                size_bytes = size_hint
            else:
                size_bytes = len(json.dumps(value, default=str).encode('utf-8'))

            # Get ML prediction for optimal tier placement
            if self.ml_predictor and user_id and query_type:
                prediction = await self.ml_predictor.predict_cache_hit(
                    key, query_type, user_id
                )
                tier = await self._determine_optimal_tier(prediction, size_bytes)
            else:
                tier = await self._heuristic_tier_selection(key, size_bytes)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                tier=tier,
                created_at=datetime.now(timezone.utc),
                last_accessed=datetime.now(timezone.utc),
                access_count=1,
                size_bytes=size_bytes,
                ttl_seconds=ttl or self.config.default_ttl_seconds,
                prediction_score=prediction.predicted_value if 'prediction' in locals() else 0.5,
                user_id=user_id,
                query_type=query_type
            )

            # Store in appropriate tier
            success = await self._store_in_tier(entry)

            if success:
                self._update_access_pattern(key, user_id)
                self.logger.debug(f"Stored {key} in {tier.value} tier (size: {size_bytes} bytes)")

            return success

        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from all cache tiers."""
        try:
            deleted = False

            # Remove from all tiers
            for tier in CacheTier:
                if await self._delete_from_tier(key, tier):
                    deleted = True

            if deleted:
                self.logger.debug(f"Deleted {key} from cache")

            return deleted

        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            return False

    async def _get_from_tier(self, key: str, tier: CacheTier) -> Optional[CacheEntry]:
        """Get entry from specific cache tier."""
        try:
            if tier == CacheTier.HOT:
                entry = self.hot_cache.get(key)
                if entry and not await self._is_expired(entry):
                    entry.last_accessed = datetime.now(timezone.utc)
                    entry.access_count += 1
                    return entry
                elif entry:
                    # Remove expired entry
                    del self.hot_cache[key]

            elif tier == CacheTier.WARM:
                entry = self.warm_cache.get(key)
                if entry and not await self._is_expired(entry):
                    entry.last_accessed = datetime.now(timezone.utc)
                    entry.access_count += 1
                    return entry
                elif entry:
                    del self.warm_cache[key]

            elif tier == CacheTier.COLD:
                if key in self.cold_cache_keys:
                    # Get from Redis cluster
                    data = await self.redis_cluster.get(f"cache:cold:{key}")
                    if data:
                        entry_dict = json.loads(data)
                        entry = CacheEntry(**entry_dict)
                        if not await self._is_expired(entry):
                            entry.last_accessed = datetime.now(timezone.utc)
                            entry.access_count += 1
                            return entry
                        else:
                            # Remove expired entry
                            await self.redis_cluster.delete(f"cache:cold:{key}")
                            self.cold_cache_keys.discard(key)

            return None

        except Exception as e:
            self.logger.error(f"Error getting {key} from {tier.value} tier: {e}")
            return None

    async def _store_in_tier(self, entry: CacheEntry) -> bool:
        """Store entry in specified tier."""
        try:
            if entry.tier == CacheTier.HOT:
                # Check capacity
                if await self._check_tier_capacity(CacheTier.HOT, entry.size_bytes):
                    self.hot_cache[entry.key] = entry
                    return True
                else:
                    # Try to evict and store in lower tier
                    await self._evict_from_hot_tier()
                    entry.tier = CacheTier.WARM
                    return await self._store_in_tier(entry)

            elif entry.tier == CacheTier.WARM:
                if await self._check_tier_capacity(CacheTier.WARM, entry.size_bytes):
                    self.warm_cache[entry.key] = entry
                    return True
                else:
                    await self._evict_from_warm_tier()
                    entry.tier = CacheTier.COLD
                    return await self._store_in_tier(entry)

            elif entry.tier == CacheTier.COLD:
                # Store in Redis cluster
                entry_dict = asdict(entry)
                # Convert datetime objects to ISO strings
                for key, value in entry_dict.items():
                    if isinstance(value, datetime):
                        entry_dict[key] = value.isoformat()

                await self.redis_cluster.set(
                    f"cache:cold:{entry.key}",
                    json.dumps(entry_dict, default=str),
                    ex=entry.ttl_seconds
                )
                self.cold_cache_keys.add(entry.key)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error storing {entry.key} in {entry.tier.value} tier: {e}")
            return False

    async def _delete_from_tier(self, key: str, tier: CacheTier) -> bool:
        """Delete entry from specific tier."""
        try:
            if tier == CacheTier.HOT and key in self.hot_cache:
                del self.hot_cache[key]
                return True
            elif tier == CacheTier.WARM and key in self.warm_cache:
                del self.warm_cache[key]
                return True
            elif tier == CacheTier.COLD and key in self.cold_cache_keys:
                await self.redis_cluster.delete(f"cache:cold:{key}")
                self.cold_cache_keys.discard(key)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error deleting {key} from {tier.value} tier: {e}")
            return False

    async def _check_tier_capacity(self, tier: CacheTier, entry_size: int) -> bool:
        """Check if tier has capacity for new entry."""
        current_size = await self._get_tier_size(tier)
        max_size = self._get_tier_max_size(tier)

        return (current_size + entry_size) <= max_size

    async def _get_tier_size(self, tier: CacheTier) -> int:
        """Get current size of cache tier in bytes."""
        if tier == CacheTier.HOT:
            return sum(entry.size_bytes for entry in self.hot_cache.values())
        elif tier == CacheTier.WARM:
            return sum(entry.size_bytes for entry in self.warm_cache.values())
        elif tier == CacheTier.COLD:
            # Estimate based on key count (would need Redis memory info for exact)
            return len(self.cold_cache_keys) * 1024  # Rough estimate

        return 0

    def _get_tier_max_size(self, tier: CacheTier) -> int:
        """Get maximum size for cache tier in bytes."""
        if tier == CacheTier.HOT:
            return int(self.config.hot_tier_size_gb * 1024 * 1024 * 1024)
        elif tier == CacheTier.WARM:
            return int(self.config.warm_tier_size_gb * 1024 * 1024 * 1024)
        elif tier == CacheTier.COLD:
            return float('inf')  # Unlimited for persistent storage

        return 0

    async def _determine_optimal_tier(self, prediction: PredictionResult, size_bytes: int) -> CacheTier:
        """Determine optimal cache tier based on ML prediction."""
        hit_probability = prediction.predicted_value
        confidence = prediction.confidence

        # High probability and high confidence -> Hot tier
        if hit_probability > 0.8 and confidence > 0.8:
            return CacheTier.HOT

        # Medium probability -> Warm tier
        elif hit_probability > 0.4:
            return CacheTier.WARM

        # Low probability or large size -> Cold tier
        else:
            return CacheTier.COLD

    async def _heuristic_tier_selection(self, key: str, size_bytes: int) -> CacheTier:
        """Fallback heuristic tier selection when ML prediction unavailable."""
        # Small items go to hot tier
        if size_bytes < 1024 * 1024:  # < 1MB
            return CacheTier.HOT

        # Medium items go to warm tier
        elif size_bytes < 10 * 1024 * 1024:  # < 10MB
            return CacheTier.WARM

        # Large items go to cold tier
        else:
            return CacheTier.COLD

    async def _should_promote_to_hot(self, entry: CacheEntry) -> bool:
        """Determine if entry should be promoted to hot tier."""
        # Promote if accessed multiple times recently
        recent_accesses = len([
            t for t in self.access_patterns.get(entry.key, [])
            if (datetime.now(timezone.utc) - t).total_seconds() < 3600
        ])

        return recent_accesses >= 3

    async def _should_promote_based_on_ml(self, entry: CacheEntry, user_id: Optional[str],
                                        query_type: Optional[str]) -> bool:
        """Use ML prediction to determine if entry should be promoted."""
        if not self.ml_predictor or not user_id or not query_type:
            return False

        try:
            prediction = await self.ml_predictor.predict_cache_hit(
                entry.key, query_type, user_id
            )
            return prediction.predicted_value > 0.6

        except Exception as e:
            self.logger.error(f"Error in ML promotion prediction: {e}")
            return False

    async def _promote_entry(self, entry: CacheEntry, target_tier: CacheTier):
        """Promote cache entry to higher tier."""
        try:
            # Remove from current tier
            await self._delete_from_tier(entry.key, entry.tier)

            # Update tier and store
            entry.tier = target_tier
            await self._store_in_tier(entry)

            self.logger.debug(f"Promoted {entry.key} to {target_tier.value} tier")

        except Exception as e:
            self.logger.error(f"Error promoting {entry.key}: {e}")

    async def _evict_from_hot_tier(self):
        """Evict entries from hot tier using adaptive policy."""
        if not self.hot_cache:
            return

        if self.config.eviction_policy == CachePolicy.ML_ADAPTIVE:
            await self._ml_adaptive_eviction(CacheTier.HOT)
        elif self.config.eviction_policy == CachePolicy.LRU:
            await self._lru_eviction(CacheTier.HOT)
        elif self.config.eviction_policy == CachePolicy.LFU:
            await self._lfu_eviction(CacheTier.HOT)

    async def _evict_from_warm_tier(self):
        """Evict entries from warm tier."""
        if not self.warm_cache:
            return

        if self.config.eviction_policy == CachePolicy.ML_ADAPTIVE:
            await self._ml_adaptive_eviction(CacheTier.WARM)
        elif self.config.eviction_policy == CachePolicy.LRU:
            await self._lru_eviction(CacheTier.WARM)
        elif self.config.eviction_policy == CachePolicy.LFU:
            await self._lfu_eviction(CacheTier.WARM)

    async def _ml_adaptive_eviction(self, tier: CacheTier):
        """ML-based adaptive eviction policy."""
        try:
            cache = self.hot_cache if tier == CacheTier.HOT else self.warm_cache

            # Get predictions for all entries
            predictions = {}
            for key, entry in cache.items():
                if self.ml_predictor and entry.user_id and entry.query_type:
                    try:
                        pred = await self.ml_predictor.predict_cache_hit(
                            key, entry.query_type, entry.user_id
                        )
                        predictions[key] = pred.predicted_value
                    except:
                        predictions[key] = 0.5  # Default

            # Sort by prediction score (ascending - evict lowest first)
            sorted_entries = sorted(
                cache.items(),
                key=lambda x: predictions.get(x[0], 0.5)
            )

            # Evict bottom 25%
            evict_count = max(1, len(sorted_entries) // 4)
            for i in range(evict_count):
                key, entry = sorted_entries[i]
                await self._delete_from_tier(key, tier)
                self.metrics.evictions += 1
                await self._record_cache_event(CacheEventType.EVICTION, key, tier, entry.user_id)

        except Exception as e:
            self.logger.error(f"Error in ML adaptive eviction: {e}")
            await self._lru_eviction(tier)  # Fallback

    async def _lru_eviction(self, tier: CacheTier):
        """Least Recently Used eviction policy."""
        cache = self.hot_cache if tier == CacheTier.HOT else self.warm_cache

        if not cache:
            return

        # Sort by last accessed time
        sorted_entries = sorted(
            cache.items(),
            key=lambda x: x[1].last_accessed
        )

        # Evict oldest 25%
        evict_count = max(1, len(sorted_entries) // 4)
        for i in range(evict_count):
            key, entry = sorted_entries[i]
            await self._delete_from_tier(key, tier)
            self.metrics.evictions += 1
            await self._record_cache_event(CacheEventType.EVICTION, key, tier, entry.user_id)

    async def _lfu_eviction(self, tier: CacheTier):
        """Least Frequently Used eviction policy."""
        cache = self.hot_cache if tier == CacheTier.HOT else self.warm_cache

        if not cache:
            return

        # Sort by access count
        sorted_entries = sorted(
            cache.items(),
            key=lambda x: x[1].access_count
        )

        # Evict least frequently used 25%
        evict_count = max(1, len(sorted_entries) // 4)
        for i in range(evict_count):
            key, entry = sorted_entries[i]
            await self._delete_from_tier(key, tier)
            self.metrics.evictions += 1
            await self._record_cache_event(CacheEventType.EVICTION, key, tier, entry.user_id)

    async def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if not entry.ttl_seconds:
            return False

        age = (datetime.now(timezone.utc) - entry.created_at).total_seconds()
        return age > entry.ttl_seconds

    def _update_access_pattern(self, key: str, user_id: Optional[str]):
        """Update access patterns for analytics."""
        now = datetime.now(timezone.utc)

        if key not in self.access_patterns:
            self.access_patterns[key] = []

        self.access_patterns[key].append(now)

        # Keep only last 24 hours
        cutoff = now - timedelta(hours=24)
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t >= cutoff
        ]

        # Update user session
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = {
                    "start_time": now,
                    "last_activity": now,
                    "queries": []
                }

            self.user_sessions[user_id]["last_activity"] = now
            self.user_sessions[user_id]["queries"].append({
                "key": key,
                "timestamp": now.isoformat()
            })

    async def _trigger_intelligent_prefetch(self, missed_key: str, user_id: str, query_type: str):
        """Trigger intelligent prefetching based on user patterns."""
        if not self.config.enable_intelligent_warming:
            return

        try:
            # Analyze user's current session
            session = self.user_sessions.get(user_id, {})
            recent_queries = session.get("queries", [])[-10:]  # Last 10 queries

            # Get similar patterns from ML predictor
            if self.ml_predictor:
                user_profile = await self.ml_predictor.analyze_user_patterns(user_id)

                # Predict likely next queries
                prefetch_candidates = await self._identify_prefetch_candidates(
                    user_profile, recent_queries, query_type
                )

                # Prefetch high-probability items
                await self._execute_prefetch_batch(prefetch_candidates[:self.config.max_prefetch_batch_size])

        except Exception as e:
            self.logger.error(f"Error in intelligent prefetch: {e}")

    async def _identify_prefetch_candidates(self, user_profile, recent_queries: List[Dict],
                                         current_query_type: str) -> List[str]:
        """Identify candidate keys for prefetching."""
        candidates = []

        # Based on user's query frequency patterns
        for query_type, frequency in user_profile.query_frequency.items():
            if frequency > 0.1:  # Frequent query type
                # Generate likely cache keys (this would be domain-specific)
                candidates.extend(await self._generate_likely_keys(query_type, user_profile.user_id))

        # Based on access patterns
        for query_type, patterns in user_profile.access_patterns.items():
            current_hour = datetime.now(timezone.utc).hour
            if patterns[current_hour] > 0:  # Active during this hour
                candidates.extend(await self._generate_time_based_keys(query_type, current_hour))

        return candidates[:50]  # Limit candidates

    async def _generate_likely_keys(self, query_type: str, user_id: str) -> List[str]:
        """Generate likely cache keys based on query type and user."""
        # This would be implemented based on your specific query patterns
        # For now, return empty list
        return []

    async def _generate_time_based_keys(self, query_type: str, hour: int) -> List[str]:
        """Generate time-based likely cache keys."""
        # This would be implemented based on temporal patterns
        return []

    async def _execute_prefetch_batch(self, candidates: List[str]):
        """Execute batch prefetching of candidate keys."""
        # This would fetch data and populate cache proactively
        # Implementation depends on your data sources
        pass

    async def _record_cache_event(self, event_type: CacheEventType, key: str,
                                 tier: Optional[CacheTier], user_id: Optional[str]):
        """Record cache event for analytics."""
        try:
            event = {
                "event_type": event_type.value,
                "key": key,
                "tier": tier.value if tier else None,
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Store in Redis for analytics
            if self.redis_standalone:
                await self.redis_standalone.lpush(
                    "cache:events",
                    json.dumps(event)
                )
                await self.redis_standalone.ltrim("cache:events", 0, 9999)  # Keep last 10k events

            # Update Prometheus metrics
            if event_type in [CacheEventType.HIT, CacheEventType.MISS]:
                self.prom_cache_requests.labels(
                    tier=tier.value if tier else "none",
                    result=event_type.value
                ).inc()

        except Exception as e:
            self.logger.error(f"Error recording cache event: {e}")

    def _update_response_time_metrics(self, response_time: float):
        """Update response time metrics."""
        # Update moving average
        alpha = 0.1  # Smoothing factor
        self.metrics.avg_response_time = (
            alpha * response_time +
            (1 - alpha) * self.metrics.avg_response_time
        )

        # Update hit rate
        if self.metrics.total_requests > 0:
            self.metrics.hit_rate = self.metrics.hits / self.metrics.total_requests

    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        tasks = [
            self._cleanup_expired_entries(),
            self._update_metrics_loop(),
            self._ml_prediction_loop(),
            self._cache_warming_loop()
        ]

        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _cleanup_expired_entries(self):
        """Background task to clean up expired cache entries."""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes

                # Clean hot tier
                expired_keys = [
                    key for key, entry in self.hot_cache.items()
                    if await self._is_expired(entry)
                ]
                for key in expired_keys:
                    del self.hot_cache[key]

                # Clean warm tier
                expired_keys = [
                    key for key, entry in self.warm_cache.items()
                    if await self._is_expired(entry)
                ]
                for key in expired_keys:
                    del self.warm_cache[key]

                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

            except Exception as e:
                self.logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(300)

    async def _update_metrics_loop(self):
        """Background task to update metrics."""
        while True:
            try:
                await asyncio.sleep(60)  # 1 minute

                # Update tier usage metrics
                for tier in CacheTier:
                    size = await self._get_tier_size(tier)
                    max_size = self._get_tier_max_size(tier)

                    if tier == CacheTier.HOT:
                        count = len(self.hot_cache)
                    elif tier == CacheTier.WARM:
                        count = len(self.warm_cache)
                    else:
                        count = len(self.cold_cache_keys)

                    self.prom_cache_size.labels(tier=tier.value).set(size)
                    self.prom_cache_entries.labels(tier=tier.value).set(count)

                    if max_size != float('inf'):
                        usage = size / max_size
                        if tier == CacheTier.HOT:
                            self.metrics.hot_tier_usage = usage
                        elif tier == CacheTier.WARM:
                            self.metrics.warm_tier_usage = usage

                # Update hit rates
                for tier in CacheTier:
                    # Calculate tier-specific hit rate from events
                    # This would require event analysis
                    pass

            except Exception as e:
                self.logger.error(f"Error in metrics update: {e}")
                await asyncio.sleep(60)

    async def _ml_prediction_loop(self):
        """Background task for ML prediction updates."""
        while True:
            try:
                await asyncio.sleep(self.config.ml_prediction_interval)

                # Update ML predictions for existing cache entries
                if self.ml_predictor:
                    await self._update_ml_predictions()

            except Exception as e:
                self.logger.error(f"Error in ML prediction loop: {e}")
                await asyncio.sleep(self.config.ml_prediction_interval)

    async def _cache_warming_loop(self):
        """Background task for intelligent cache warming."""
        while True:
            try:
                await asyncio.sleep(1800)  # 30 minutes

                if self.config.enable_intelligent_warming:
                    await self._perform_intelligent_warming()

            except Exception as e:
                self.logger.error(f"Error in cache warming: {e}")
                await asyncio.sleep(1800)

    async def _update_ml_predictions(self):
        """Update ML predictions for cache entries."""
        # Update predictions for hot and warm tier entries
        for cache in [self.hot_cache, self.warm_cache]:
            for key, entry in cache.items():
                if entry.user_id and entry.query_type and self.ml_predictor:
                    try:
                        prediction = await self.ml_predictor.predict_cache_hit(
                            key, entry.query_type, entry.user_id
                        )
                        entry.prediction_score = prediction.predicted_value
                    except:
                        pass  # Keep existing score

    async def _perform_intelligent_warming(self):
        """Perform intelligent cache warming based on patterns."""
        try:
            # Analyze user patterns and pre-load likely accessed data
            for user_id, session in self.user_sessions.items():
                if self.ml_predictor:
                    user_profile = await self.ml_predictor.analyze_user_patterns(user_id)

                    # Warm cache for user's frequent query types
                    for query_type, frequency in user_profile.query_frequency.items():
                        if frequency > 0.2:  # High frequency
                            await self._warm_cache_for_query_type(query_type, user_id)

        except Exception as e:
            self.logger.error(f"Error in intelligent warming: {e}")

    async def _warm_cache_for_query_type(self, query_type: str, user_id: str):
        """Warm cache for specific query type and user."""
        # This would pre-load common queries for the user/query type
        # Implementation depends on your specific data sources
        pass

    async def _load_cache_state(self):
        """Load cache state from persistent storage."""
        try:
            if self.redis_standalone:
                # Load cold cache keys
                keys = await self.redis_standalone.smembers("cache:cold_keys")
                self.cold_cache_keys = set(keys)

                self.logger.info(f"Loaded {len(self.cold_cache_keys)} cold cache keys")

        except Exception as e:
            self.logger.error(f"Error loading cache state: {e}")

    async def _save_cache_state(self):
        """Save cache state to persistent storage."""
        try:
            if self.redis_standalone:
                # Save cold cache keys
                if self.cold_cache_keys:
                    await self.redis_standalone.delete("cache:cold_keys")
                    await self.redis_standalone.sadd("cache:cold_keys", *self.cold_cache_keys)

                self.logger.info("Saved cache state")

        except Exception as e:
            self.logger.error(f"Error saving cache state: {e}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "metrics": asdict(self.metrics),
            "tier_stats": {
                "hot": {
                    "entries": len(self.hot_cache),
                    "size_bytes": await self._get_tier_size(CacheTier.HOT),
                    "max_size_bytes": self._get_tier_max_size(CacheTier.HOT)
                },
                "warm": {
                    "entries": len(self.warm_cache),
                    "size_bytes": await self._get_tier_size(CacheTier.WARM),
                    "max_size_bytes": self._get_tier_max_size(CacheTier.WARM)
                },
                "cold": {
                    "entries": len(self.cold_cache_keys),
                    "size_bytes": await self._get_tier_size(CacheTier.COLD),
                    "max_size_bytes": self._get_tier_max_size(CacheTier.COLD)
                }
            },
            "configuration": asdict(self.config),
            "active_users": len(self.user_sessions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return stats


async def main():
    """Main entry point for predictive cache."""
    config = {
        "cache": {
            "hot_tier_size_gb": 4.0,
            "warm_tier_size_gb": 8.0,
            "prefetch_threshold": 0.7,
            "enable_intelligent_warming": True
        },
        "redis_cluster_nodes": [
            {"host": "redis-node-1", "port": 7001},
            {"host": "redis-node-2", "port": 7002},
            {"host": "redis-node-3", "port": 7003}
        ],
        "redis_password": "your_redis_password",
        "postgres_uri": "postgresql://user:pass@postgres:5432/bev_osint"
    }

    cache = PredictiveCache(config)

    try:
        await cache.initialize()
        # Keep running (in real deployment, this would be part of a larger service)
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await cache.shutdown()


if __name__ == "__main__":
    asyncio.run(main())