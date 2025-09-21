"""
BEV OSINT Framework - Enterprise Cache Optimizer
Advanced caching optimization for 99% hit rate and <25ms response times
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import aioredis
import asyncpg
from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger(__name__)

class CacheTier(Enum):
    """Cache tier definitions"""
    L1_MEMORY = "l1_memory"      # Ultra-fast in-memory cache
    L2_REDIS = "l2_redis"         # Fast Redis cluster cache
    L3_DISTRIBUTED = "l3_dist"    # Distributed persistent cache
    L4_EDGE = "l4_edge"          # Edge location cache
    L5_COLD = "l5_cold"          # Cold storage cache

class PrefetchStrategy(Enum):
    """Prefetch strategies for predictive caching"""
    TEMPORAL = "temporal"           # Time-based patterns
    SEQUENTIAL = "sequential"       # Sequential access patterns
    COLLABORATIVE = "collaborative" # User similarity patterns
    SEMANTIC = "semantic"          # Content similarity patterns
    HYBRID = "hybrid"             # Combined approach

@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata"""
    key: str
    value: Any
    size_bytes: int
    tier: CacheTier
    access_count: int = 0
    last_access: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    creation_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ttl_seconds: Optional[int] = None
    user_id: Optional[str] = None
    query_type: Optional[str] = None
    semantic_vector: Optional[np.ndarray] = None
    prefetch_score: float = 0.0
    compression_ratio: float = 1.0
    hit_probability: float = 0.5

@dataclass
class AccessPattern:
    """User access pattern analysis"""
    user_id: str
    patterns: Dict[str, List[float]]
    temporal_distribution: np.ndarray
    query_types: Dict[str, int]
    avg_session_duration: float
    peak_hours: List[int]
    predictability_score: float

@dataclass
class CacheMetrics:
    """Comprehensive cache metrics"""
    hit_rate: float
    miss_rate: float
    eviction_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    memory_usage_percent: float
    bandwidth_usage_mbps: float
    prefetch_accuracy: float
    compression_ratio: float
    tier_distribution: Dict[str, float]

class EnterpriseCacheOptimizer:
    """
    Enterprise-grade cache optimizer for 99% hit rate target

    Features:
    - Multi-tier intelligent caching
    - ML-based predictive prefetching
    - Semantic content clustering
    - Adaptive TTL management
    - Real-time optimization
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Cache tiers with size limits (in bytes)
        self.tier_limits = {
            CacheTier.L1_MEMORY: 4 * 1024**3,      # 4 GB
            CacheTier.L2_REDIS: 64 * 1024**3,      # 64 GB
            CacheTier.L3_DISTRIBUTED: 256 * 1024**3, # 256 GB
            CacheTier.L4_EDGE: 128 * 1024**3,      # 128 GB per edge
            CacheTier.L5_COLD: float('inf')        # Unlimited
        }

        # Cache storage
        self.caches: Dict[CacheTier, Dict[str, CacheEntry]] = {
            tier: {} for tier in CacheTier
        }

        # Access tracking
        self.access_history = deque(maxlen=100000)
        self.user_patterns: Dict[str, AccessPattern] = {}
        self.query_frequencies = defaultdict(int)
        self.semantic_clusters: Dict[str, Set[str]] = {}

        # ML models for prediction
        self.hit_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.ttl_optimizer = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.model_trained = False

        # Performance tracking
        self.current_metrics = CacheMetrics(
            hit_rate=0.0, miss_rate=1.0, eviction_rate=0.0,
            avg_latency_ms=0.0, p50_latency_ms=0.0,
            p95_latency_ms=0.0, p99_latency_ms=0.0,
            memory_usage_percent=0.0, bandwidth_usage_mbps=0.0,
            prefetch_accuracy=0.0, compression_ratio=1.0,
            tier_distribution={}
        )

        # Prometheus metrics
        self.cache_hits = Counter('bev_cache_hits_total', 'Total cache hits', ['tier'])
        self.cache_misses = Counter('bev_cache_misses_total', 'Total cache misses')
        self.cache_latency = Histogram('bev_cache_latency_ms', 'Cache operation latency')
        self.hit_rate_gauge = Gauge('bev_cache_hit_rate', 'Current cache hit rate')
        self.memory_usage_gauge = Gauge('bev_cache_memory_usage_bytes', 'Memory usage', ['tier'])

        # Optimization state
        self.optimization_interval = config.get('optimization_interval_seconds', 300)
        self.last_optimization = datetime.now(timezone.utc)
        self.prefetch_queue = asyncio.Queue(maxsize=10000)
        self.active_prefetches: Set[str] = set()

        # External connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize cache optimizer with external connections"""
        # Initialize Redis connection
        redis_url = self.config.get('redis_url', 'redis://localhost:6379')
        self.redis_client = await aioredis.from_url(
            redis_url,
            decode_responses=False,
            max_connections=100
        )

        # Initialize database pool
        if self.config.get('postgres_uri'):
            self.db_pool = await asyncpg.create_pool(
                self.config['postgres_uri'],
                min_size=10,
                max_size=50,
                command_timeout=10
            )

        # Start background tasks
        asyncio.create_task(self._optimization_loop())
        asyncio.create_task(self._prefetch_loop())
        asyncio.create_task(self._metrics_loop())

        logger.info("Enterprise cache optimizer initialized")

    async def get(self, key: str, user_id: Optional[str] = None) -> Tuple[Optional[Any], CacheTier]:
        """
        Get value from cache with tier traversal

        Returns:
            Tuple of (value, tier) or (None, None) if not found
        """
        start_time = time.time()

        # Check each tier in order
        for tier in CacheTier:
            if key in self.caches[tier]:
                entry = self.caches[tier][key]

                # Check TTL
                if entry.ttl_seconds:
                    age = (datetime.now(timezone.utc) - entry.creation_time).total_seconds()
                    if age > entry.ttl_seconds:
                        # Expired, remove and continue
                        del self.caches[tier][key]
                        continue

                # Update access metadata
                entry.access_count += 1
                entry.last_access = datetime.now(timezone.utc)

                # Promote to higher tier if frequently accessed
                if await self._should_promote(entry, tier):
                    await self._promote_entry(entry, tier)

                # Record hit
                self.cache_hits.labels(tier=tier.value).inc()
                latency = (time.time() - start_time) * 1000
                self.cache_latency.observe(latency)

                # Track access pattern
                await self._track_access(key, user_id, tier, hit=True)

                # Trigger predictive prefetch
                asyncio.create_task(self._predictive_prefetch(key, user_id))

                return entry.value, tier

        # Cache miss
        self.cache_misses.inc()
        latency = (time.time() - start_time) * 1000
        self.cache_latency.observe(latency)

        await self._track_access(key, user_id, None, hit=False)

        return None, None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                  user_id: Optional[str] = None, query_type: Optional[str] = None,
                  semantic_vector: Optional[np.ndarray] = None) -> bool:
        """
        Set value in cache with intelligent tier placement
        """
        try:
            # Calculate entry size
            size_bytes = self._estimate_size(value)

            # Compress if beneficial
            compressed_value, compression_ratio = await self._compress_if_beneficial(value)

            # Predict hit probability
            hit_probability = await self._predict_hit_probability(
                key, user_id, query_type, semantic_vector
            )

            # Determine optimal tier based on prediction
            optimal_tier = self._determine_optimal_tier(
                size_bytes, hit_probability, query_type
            )

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=compressed_value,
                size_bytes=size_bytes,
                tier=optimal_tier,
                ttl_seconds=ttl or self._calculate_adaptive_ttl(hit_probability),
                user_id=user_id,
                query_type=query_type,
                semantic_vector=semantic_vector,
                hit_probability=hit_probability,
                compression_ratio=compression_ratio
            )

            # Ensure space in target tier
            await self._ensure_space(optimal_tier, size_bytes)

            # Store in cache
            self.caches[optimal_tier][key] = entry

            # Update memory usage metrics
            self._update_memory_metrics()

            # Add to semantic cluster if vector provided
            if semantic_vector is not None:
                await self._update_semantic_cluster(key, semantic_vector)

            # Trigger collaborative prefetch
            if user_id:
                asyncio.create_task(
                    self._collaborative_prefetch(user_id, query_type)
                )

            return True

        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {e}")
            return False

    async def _should_promote(self, entry: CacheEntry, current_tier: CacheTier) -> bool:
        """Determine if entry should be promoted to higher tier"""
        if current_tier == CacheTier.L1_MEMORY:
            return False  # Already at highest tier

        # Promotion criteria
        access_threshold = {
            CacheTier.L5_COLD: 2,
            CacheTier.L4_EDGE: 5,
            CacheTier.L3_DISTRIBUTED: 10,
            CacheTier.L2_REDIS: 20
        }

        # Check if access count exceeds threshold
        if entry.access_count >= access_threshold.get(current_tier, 10):
            # Additional check: recency of accesses
            recency = (datetime.now(timezone.utc) - entry.last_access).total_seconds()
            if recency < 300:  # Accessed within last 5 minutes
                return True

        return False

    async def _promote_entry(self, entry: CacheEntry, current_tier: CacheTier):
        """Promote entry to higher tier"""
        tier_order = list(CacheTier)
        current_index = tier_order.index(current_tier)

        if current_index > 0:
            target_tier = tier_order[current_index - 1]

            # Ensure space in target tier
            await self._ensure_space(target_tier, entry.size_bytes)

            # Move entry
            del self.caches[current_tier][entry.key]
            entry.tier = target_tier
            self.caches[target_tier][entry.key] = entry

            logger.debug(f"Promoted {entry.key} from {current_tier} to {target_tier}")

    async def _ensure_space(self, tier: CacheTier, required_bytes: int):
        """Ensure sufficient space in tier through intelligent eviction"""
        current_usage = sum(
            entry.size_bytes for entry in self.caches[tier].values()
        )

        limit = self.tier_limits[tier]

        while current_usage + required_bytes > limit:
            # Find best candidate for eviction
            eviction_candidate = await self._select_eviction_candidate(tier)

            if not eviction_candidate:
                break

            # Demote to lower tier if valuable
            if eviction_candidate.hit_probability > 0.3:
                await self._demote_entry(eviction_candidate, tier)
            else:
                # Evict completely
                del self.caches[tier][eviction_candidate.key]

            current_usage -= eviction_candidate.size_bytes

    async def _select_eviction_candidate(self, tier: CacheTier) -> Optional[CacheEntry]:
        """Select best candidate for eviction using advanced scoring"""
        if not self.caches[tier]:
            return None

        candidates = []
        now = datetime.now(timezone.utc)

        for entry in self.caches[tier].values():
            # Calculate eviction score (lower is better)
            recency = (now - entry.last_access).total_seconds()
            frequency = entry.access_count

            # TTL consideration
            if entry.ttl_seconds:
                remaining_ttl = entry.ttl_seconds - (now - entry.creation_time).total_seconds()
                ttl_factor = max(0, remaining_ttl / entry.ttl_seconds)
            else:
                ttl_factor = 1.0

            # Combined score
            score = (
                recency / 3600 +                    # Recency (hours)
                1 / (frequency + 1) * 10 +          # Frequency inverse
                (1 - entry.hit_probability) * 5 +   # Hit probability
                1 / (entry.size_bytes / 1024) +     # Size factor (prefer evicting larger)
                (1 - ttl_factor) * 2                # TTL factor
            )

            candidates.append((score, entry))

        # Sort by score and return worst candidate
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1] if candidates else None

    async def _demote_entry(self, entry: CacheEntry, current_tier: CacheTier):
        """Demote entry to lower tier"""
        tier_order = list(CacheTier)
        current_index = tier_order.index(current_tier)

        if current_index < len(tier_order) - 1:
            target_tier = tier_order[current_index + 1]

            # Move entry
            del self.caches[current_tier][entry.key]
            entry.tier = target_tier
            self.caches[target_tier][entry.key] = entry

    async def _predict_hit_probability(self, key: str, user_id: Optional[str],
                                      query_type: Optional[str],
                                      semantic_vector: Optional[np.ndarray]) -> float:
        """Predict hit probability using ML model"""
        if not self.model_trained:
            return 0.5  # Default probability

        try:
            # Extract features
            features = await self._extract_features(key, user_id, query_type, semantic_vector)

            # Scale features
            scaled_features = self.scaler.transform([features])

            # Predict
            probability = self.hit_predictor.predict(scaled_features)[0]

            return min(1.0, max(0.0, probability))

        except Exception as e:
            logger.error(f"Hit prediction failed: {e}")
            return 0.5

    async def _extract_features(self, key: str, user_id: Optional[str],
                               query_type: Optional[str],
                               semantic_vector: Optional[np.ndarray]) -> List[float]:
        """Extract features for ML prediction"""
        features = []

        # Key characteristics
        key_hash = hashlib.md5(key.encode()).hexdigest()
        features.append(int(key_hash[:8], 16) / (16**8))  # Normalized hash

        # User patterns
        if user_id and user_id in self.user_patterns:
            pattern = self.user_patterns[user_id]
            features.append(pattern.predictability_score)
            features.append(len(pattern.query_types))
            features.append(pattern.avg_session_duration / 3600)  # Hours
        else:
            features.extend([0.5, 1, 0.5])

        # Query type frequency
        if query_type:
            frequency = self.query_frequencies[query_type]
            features.append(min(1.0, frequency / 1000))
        else:
            features.append(0.0)

        # Time features
        now = datetime.now(timezone.utc)
        features.append(now.hour / 24)
        features.append(now.weekday() / 7)

        # Semantic similarity to hot items
        if semantic_vector is not None:
            similarity = await self._calculate_semantic_similarity(semantic_vector)
            features.append(similarity)
        else:
            features.append(0.0)

        return features

    def _determine_optimal_tier(self, size_bytes: int, hit_probability: float,
                               query_type: Optional[str]) -> CacheTier:
        """Determine optimal cache tier for entry"""
        # Size-based tier selection
        if size_bytes < 1024:  # < 1KB
            if hit_probability > 0.9:
                return CacheTier.L1_MEMORY
            elif hit_probability > 0.7:
                return CacheTier.L2_REDIS
            else:
                return CacheTier.L3_DISTRIBUTED
        elif size_bytes < 1024 * 1024:  # < 1MB
            if hit_probability > 0.8:
                return CacheTier.L2_REDIS
            else:
                return CacheTier.L3_DISTRIBUTED
        else:  # >= 1MB
            if hit_probability > 0.9:
                return CacheTier.L3_DISTRIBUTED
            elif hit_probability > 0.5:
                return CacheTier.L4_EDGE
            else:
                return CacheTier.L5_COLD

    def _calculate_adaptive_ttl(self, hit_probability: float) -> int:
        """Calculate adaptive TTL based on hit probability"""
        # Higher hit probability = longer TTL
        base_ttl = 300  # 5 minutes

        if hit_probability > 0.9:
            return base_ttl * 12  # 1 hour
        elif hit_probability > 0.7:
            return base_ttl * 6   # 30 minutes
        elif hit_probability > 0.5:
            return base_ttl * 2   # 10 minutes
        else:
            return base_ttl       # 5 minutes

    async def _predictive_prefetch(self, key: str, user_id: Optional[str]):
        """Trigger predictive prefetch based on access patterns"""
        try:
            # Temporal prefetch - predict next likely access
            temporal_candidates = await self._predict_temporal_sequence(key)

            # Sequential prefetch - predict related items
            sequential_candidates = await self._predict_sequential_items(key)

            # Collaborative prefetch - based on similar users
            if user_id:
                collaborative_candidates = await self._predict_collaborative_items(user_id)
            else:
                collaborative_candidates = []

            # Semantic prefetch - similar content
            semantic_candidates = await self._predict_semantic_items(key)

            # Combine and rank candidates
            all_candidates = (
                temporal_candidates +
                sequential_candidates +
                collaborative_candidates +
                semantic_candidates
            )

            # Deduplicate and sort by score
            seen = set()
            unique_candidates = []
            for score, candidate in sorted(all_candidates, reverse=True):
                if candidate not in seen and candidate not in self.active_prefetches:
                    seen.add(candidate)
                    unique_candidates.append((score, candidate))
                    if len(unique_candidates) >= 10:
                        break

            # Queue for prefetch
            for score, candidate in unique_candidates:
                if score > 0.7:  # Only prefetch high-confidence items
                    await self.prefetch_queue.put((score, candidate, user_id))
                    self.active_prefetches.add(candidate)

        except Exception as e:
            logger.error(f"Predictive prefetch failed: {e}")

    async def _prefetch_loop(self):
        """Background loop for executing prefetch operations"""
        while True:
            try:
                # Get next prefetch task
                score, key, user_id = await self.prefetch_queue.get()

                # Check if already cached
                cached_value, _ = await self.get(key, user_id)
                if cached_value is None:
                    # Fetch from source (would be implemented based on key pattern)
                    value = await self._fetch_from_source(key)
                    if value:
                        await self.set(key, value, user_id=user_id)

                # Remove from active set
                self.active_prefetches.discard(key)

            except Exception as e:
                logger.error(f"Prefetch loop error: {e}")
                await asyncio.sleep(1)

    async def _optimization_loop(self):
        """Background loop for cache optimization"""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)

                # Collect metrics
                metrics = await self._collect_metrics()
                self.current_metrics = metrics

                # Update Prometheus metrics
                self.hit_rate_gauge.set(metrics.hit_rate)

                # Retrain ML models if enough data
                if len(self.access_history) >= 10000:
                    await self._retrain_models()

                # Optimize tier distribution
                await self._optimize_tier_distribution()

                # Clean expired entries
                await self._cleanup_expired()

                logger.info(f"Cache optimization complete - Hit rate: {metrics.hit_rate:.2%}")

            except Exception as e:
                logger.error(f"Optimization loop error: {e}")

    async def _metrics_loop(self):
        """Background loop for metrics collection"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute

                # Log current metrics
                metrics = self.current_metrics
                logger.info(
                    f"Cache Metrics - Hit: {metrics.hit_rate:.2%}, "
                    f"P99: {metrics.p99_latency_ms:.1f}ms, "
                    f"Memory: {metrics.memory_usage_percent:.1%}, "
                    f"Prefetch: {metrics.prefetch_accuracy:.2%}"
                )

            except Exception as e:
                logger.error(f"Metrics loop error: {e}")

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        if isinstance(value, str):
            return len(value.encode('utf-8'))
        elif isinstance(value, bytes):
            return len(value)
        elif isinstance(value, (dict, list)):
            return len(json.dumps(value).encode('utf-8'))
        else:
            return 256  # Default estimate

    async def _compress_if_beneficial(self, value: Any) -> Tuple[Any, float]:
        """Compress value if beneficial"""
        # Implementation would use lz4 or zstd compression
        # For now, return uncompressed
        return value, 1.0

    async def _fetch_from_source(self, key: str) -> Optional[Any]:
        """Fetch value from original source"""
        # This would be implemented based on your data sources
        return None

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        tier_stats = {}
        total_entries = 0
        total_size = 0

        for tier in CacheTier:
            entries = len(self.caches[tier])
            size = sum(e.size_bytes for e in self.caches[tier].values())
            tier_stats[tier.value] = {
                'entries': entries,
                'size_bytes': size,
                'size_mb': size / (1024**2),
                'usage_percent': (size / self.tier_limits[tier] * 100)
                                if self.tier_limits[tier] != float('inf') else 0
            }
            total_entries += entries
            total_size += size

        return {
            'total_entries': total_entries,
            'total_size_mb': total_size / (1024**2),
            'tier_stats': tier_stats,
            'metrics': {
                'hit_rate': self.current_metrics.hit_rate,
                'p99_latency_ms': self.current_metrics.p99_latency_ms,
                'prefetch_accuracy': self.current_metrics.prefetch_accuracy,
                'compression_ratio': self.current_metrics.compression_ratio
            },
            'active_prefetches': len(self.active_prefetches),
            'model_trained': self.model_trained
        }