"""
BEV OSINT Framework - Cache Hit Rate Optimizer
Adaptive algorithms for optimizing cache performance with ML-driven strategies.
"""

import asyncio
import json
import logging
import time
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import aioredis
import psycopg2.pool
from prometheus_client import Gauge, Counter, Histogram
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from .ml_predictor import MLPredictor, PredictionResult
from .predictive_cache import PredictiveCache, CacheTier, CachePolicy
from .cache_warmer import CacheWarmer


class OptimizationStrategy(Enum):
    ADAPTIVE_REPLACEMENT = "adaptive_replacement"
    ML_GUIDED = "ml_guided"
    FREQUENCY_BASED = "frequency_based"
    RECENCY_BASED = "recency_based"
    SIZE_AWARE = "size_aware"
    COST_AWARE = "cost_aware"
    HYBRID = "hybrid"


class AdaptivePolicy(Enum):
    ARC = "arc"  # Adaptive Replacement Cache
    CAR = "car"  # Clock with Adaptive Replacement
    LIRS = "lirs"  # Low Inter-reference Recency Set
    ML_ARC = "ml_arc"  # ML-enhanced ARC
    DYNAMIC = "dynamic"  # Dynamic policy selection


class OptimizationMetric(Enum):
    HIT_RATE = "hit_rate"
    RESPONSE_TIME = "response_time"
    MEMORY_EFFICIENCY = "memory_efficiency"
    COST_EFFICIENCY = "cost_efficiency"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class CacheEntry:
    """Enhanced cache entry with optimization metadata."""
    key: str
    size: int
    frequency: int
    recency_score: float
    prediction_score: float
    cost_score: float
    last_access: datetime
    access_pattern: List[datetime]
    tier: CacheTier
    user_priority: float = 1.0
    query_type: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class OptimizationResult:
    """Result of cache optimization analysis."""
    strategy: OptimizationStrategy
    policy: AdaptivePolicy
    hit_rate_improvement: float
    memory_savings: float
    recommendations: List[str]
    eviction_candidates: List[str]
    promotion_candidates: List[str]
    tier_rebalancing: Dict[CacheTier, float]
    confidence: float
    timestamp: datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization analysis."""
    hit_rate: float
    miss_rate: float
    avg_response_time: float
    memory_utilization: float
    eviction_rate: float
    promotion_rate: float
    prefetch_accuracy: float
    cost_per_hit: float
    user_satisfaction_score: float
    timestamp: datetime


class ARCOptimizer:
    """Adaptive Replacement Cache implementation with ML enhancements."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.p = 0  # Target size for T1 (recency)

        # ARC lists
        self.T1 = deque()  # Recently used pages
        self.T2 = deque()  # Frequently used pages
        self.B1 = deque()  # Ghost list for T1
        self.B2 = deque()  # Ghost list for T2

        # Entry metadata
        self.entries: Dict[str, CacheEntry] = {}
        self.access_history: Dict[str, List[datetime]] = {}

    def adapt(self, key: str, in_b1: bool = False, in_b2: bool = False):
        """Adapt the target size p based on access patterns."""
        if in_b1:
            # Hit in B1 - increase recency component
            delta = max(len(self.B2) // len(self.B1), 1) if len(self.B1) > 0 else 1
            self.p = min(self.p + delta, self.max_size)
        elif in_b2:
            # Hit in B2 - increase frequency component
            delta = max(len(self.B1) // len(self.B2), 1) if len(self.B2) > 0 else 1
            self.p = max(self.p - delta, 0)

    def should_evict_from_t1(self) -> bool:
        """Determine if we should evict from T1 or T2."""
        if len(self.T1) >= max(1, self.p):
            return True
        return False


class CacheOptimizer:
    """
    Advanced cache optimizer with adaptive algorithms and ML-driven strategies.
    Optimizes cache performance through intelligent eviction, promotion, and rebalancing.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_url = config.get("redis_url", "redis://redis:6379/15")
        self.postgres_uri = config.get("postgres_uri")

        # Optimization configuration
        self.optimization_interval = config.get("optimization_interval_seconds", 300)  # 5 minutes
        self.analysis_window_hours = config.get("analysis_window_hours", 24)
        self.min_samples_for_ml = config.get("min_samples_for_ml", 100)
        self.target_hit_rate = config.get("target_hit_rate", 0.85)
        self.max_memory_utilization = config.get("max_memory_utilization", 0.9)

        # Adaptive algorithms
        self.arc_optimizers: Dict[CacheTier, ARCOptimizer] = {}
        self.current_strategy = OptimizationStrategy.HYBRID
        self.current_policy = AdaptivePolicy.ML_ARC

        # Performance tracking
        self.performance_history: List[PerformanceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []

        # ML models for optimization
        self.hit_rate_predictor: Optional[RandomForestRegressor] = None
        self.eviction_predictor: Optional[LinearRegression] = None

        # Cache system references
        self.cache_system: Optional[PredictiveCache] = None
        self.ml_predictor: Optional[MLPredictor] = None
        self.cache_warmer: Optional[CacheWarmer] = None

        # External connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

        # Metrics tracking
        self.tier_metrics: Dict[CacheTier, Dict[str, float]] = {
            tier: {"hit_rate": 0.0, "utilization": 0.0, "avg_size": 0.0}
            for tier in CacheTier
        }

        # Prometheus metrics
        self._init_prometheus_metrics()

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()

        # Logging
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for cache optimizer."""
        logger = logging.getLogger('cache_optimizer')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for cache optimization."""
        self.prom_hit_rate = Gauge(
            'bev_cache_hit_rate_optimized',
            'Optimized cache hit rate',
            ['tier']
        )
        self.prom_memory_efficiency = Gauge(
            'bev_cache_memory_efficiency',
            'Cache memory efficiency',
            ['tier']
        )
        self.prom_optimization_score = Gauge(
            'bev_cache_optimization_score',
            'Overall cache optimization score'
        )
        self.prom_evictions = Counter(
            'bev_cache_evictions_optimized_total',
            'Optimized cache evictions',
            ['tier', 'reason']
        )
        self.prom_promotions = Counter(
            'bev_cache_promotions_total',
            'Cache tier promotions',
            ['from_tier', 'to_tier']
        )
        self.prom_rebalancing_ops = Counter(
            'bev_cache_rebalancing_operations_total',
            'Cache rebalancing operations'
        )

    async def initialize(self):
        """Initialize cache optimizer."""
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
                    maxconn=5,
                    dsn=self.postgres_uri
                )

            # Initialize ARC optimizers for each tier
            tier_sizes = {
                CacheTier.HOT: int(4 * 1024 * 1024 * 1024),  # 4GB
                CacheTier.WARM: int(8 * 1024 * 1024 * 1024),  # 8GB
                CacheTier.COLD: int(float('inf'))  # Unlimited
            }

            for tier, size in tier_sizes.items():
                if size != float('inf'):
                    self.arc_optimizers[tier] = ARCOptimizer(size)

            # Load optimization state
            await self._load_optimization_state()

            # Start background tasks
            await self._start_background_tasks()

            self.logger.info("Cache optimizer initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize cache optimizer: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown of cache optimizer."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Save optimization state
        await self._save_optimization_state()

        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        if self.db_pool:
            self.db_pool.closeall()

        self.logger.info("Cache optimizer shutdown completed")

    def set_cache_system(self, cache_system: PredictiveCache):
        """Set cache system reference."""
        self.cache_system = cache_system

    def set_ml_predictor(self, predictor: MLPredictor):
        """Set ML predictor reference."""
        self.ml_predictor = predictor

    def set_cache_warmer(self, warmer: CacheWarmer):
        """Set cache warmer reference."""
        self.cache_warmer = warmer

    async def analyze_cache_performance(self) -> PerformanceMetrics:
        """
        Analyze current cache performance across all tiers.

        Returns:
            PerformanceMetrics with current performance data
        """
        try:
            if not self.cache_system:
                raise ValueError("Cache system not set")

            # Get current cache statistics
            cache_stats = await self.cache_system.get_cache_stats()

            # Calculate performance metrics
            metrics = cache_stats["metrics"]
            hit_rate = metrics.get("hit_rate", 0.0)
            miss_rate = 1.0 - hit_rate
            avg_response_time = metrics.get("avg_response_time", 0.0)

            # Calculate memory utilization
            tier_stats = cache_stats["tier_stats"]
            total_used = sum(
                tier["size_bytes"] for tier in tier_stats.values()
                if tier["max_size_bytes"] != float('inf')
            )
            total_max = sum(
                tier["max_size_bytes"] for tier in tier_stats.values()
                if tier["max_size_bytes"] != float('inf')
            )
            memory_utilization = total_used / total_max if total_max > 0 else 0.0

            # Get additional metrics from cache events
            eviction_rate = await self._calculate_eviction_rate()
            promotion_rate = await self._calculate_promotion_rate()
            prefetch_accuracy = await self._calculate_prefetch_accuracy()
            cost_per_hit = await self._calculate_cost_per_hit()
            user_satisfaction = await self._calculate_user_satisfaction()

            performance = PerformanceMetrics(
                hit_rate=hit_rate,
                miss_rate=miss_rate,
                avg_response_time=avg_response_time,
                memory_utilization=memory_utilization,
                eviction_rate=eviction_rate,
                promotion_rate=promotion_rate,
                prefetch_accuracy=prefetch_accuracy,
                cost_per_hit=cost_per_hit,
                user_satisfaction_score=user_satisfaction,
                timestamp=datetime.now(timezone.utc)
            )

            # Store in history
            self.performance_history.append(performance)
            if len(self.performance_history) > 288:  # Keep 24 hours at 5-min intervals
                self.performance_history = self.performance_history[-288:]

            return performance

        except Exception as e:
            self.logger.error(f"Error analyzing cache performance: {e}")
            raise

    async def optimize_cache_strategy(self) -> OptimizationResult:
        """
        Perform comprehensive cache optimization analysis.

        Returns:
            OptimizationResult with optimization recommendations
        """
        try:
            # Analyze current performance
            current_performance = await self.analyze_cache_performance()

            # Determine optimal strategy
            optimal_strategy = await self._determine_optimal_strategy(current_performance)

            # Determine optimal policy
            optimal_policy = await self._determine_optimal_policy(current_performance)

            # Generate specific recommendations
            recommendations = await self._generate_optimization_recommendations(
                current_performance, optimal_strategy, optimal_policy
            )

            # Identify eviction candidates
            eviction_candidates = await self._identify_eviction_candidates()

            # Identify promotion candidates
            promotion_candidates = await self._identify_promotion_candidates()

            # Calculate tier rebalancing
            tier_rebalancing = await self._calculate_tier_rebalancing()

            # Estimate improvement
            hit_rate_improvement = await self._estimate_hit_rate_improvement(
                optimal_strategy, optimal_policy
            )
            memory_savings = await self._estimate_memory_savings(eviction_candidates)

            # Calculate confidence
            confidence = self._calculate_optimization_confidence(current_performance)

            result = OptimizationResult(
                strategy=optimal_strategy,
                policy=optimal_policy,
                hit_rate_improvement=hit_rate_improvement,
                memory_savings=memory_savings,
                recommendations=recommendations,
                eviction_candidates=eviction_candidates,
                promotion_candidates=promotion_candidates,
                tier_rebalancing=tier_rebalancing,
                confidence=confidence,
                timestamp=datetime.now(timezone.utc)
            )

            # Store in history
            self.optimization_history.append(result)
            if len(self.optimization_history) > 100:
                self.optimization_history = self.optimization_history[-100:]

            return result

        except Exception as e:
            self.logger.error(f"Error optimizing cache strategy: {e}")
            raise

    async def apply_optimization(self, optimization: OptimizationResult) -> bool:
        """
        Apply optimization recommendations to the cache system.

        Args:
            optimization: OptimizationResult to apply

        Returns:
            True if successfully applied
        """
        try:
            if not self.cache_system:
                raise ValueError("Cache system not set")

            self.logger.info(f"Applying optimization: {optimization.strategy.value}")

            # Update current strategy and policy
            self.current_strategy = optimization.strategy
            self.current_policy = optimization.policy

            # Apply evictions
            eviction_success = await self._apply_evictions(optimization.eviction_candidates)

            # Apply promotions
            promotion_success = await self._apply_promotions(optimization.promotion_candidates)

            # Apply tier rebalancing
            rebalancing_success = await self._apply_tier_rebalancing(optimization.tier_rebalancing)

            # Update cache policies
            policy_success = await self._update_cache_policies(optimization.policy)

            success = all([eviction_success, promotion_success, rebalancing_success, policy_success])

            if success:
                self.logger.info("Successfully applied cache optimization")

                # Update Prometheus metrics
                self.prom_optimization_score.set(optimization.confidence)
                self.prom_rebalancing_ops.inc()

            else:
                self.logger.warning("Partial success in applying cache optimization")

            return success

        except Exception as e:
            self.logger.error(f"Error applying optimization: {e}")
            return False

    async def _determine_optimal_strategy(self, performance: PerformanceMetrics) -> OptimizationStrategy:
        """Determine optimal optimization strategy based on current performance."""
        try:
            # Analyze performance characteristics
            hit_rate = performance.hit_rate
            memory_util = performance.memory_utilization
            response_time = performance.avg_response_time
            user_satisfaction = performance.user_satisfaction_score

            # ML-guided strategy if we have enough data and good ML predictor
            if (self.ml_predictor and
                len(self.performance_history) >= self.min_samples_for_ml and
                hit_rate < self.target_hit_rate):
                return OptimizationStrategy.ML_GUIDED

            # Adaptive replacement if memory pressure is high
            elif memory_util > self.max_memory_utilization:
                return OptimizationStrategy.ADAPTIVE_REPLACEMENT

            # Frequency-based if hit rate is low
            elif hit_rate < 0.7:
                return OptimizationStrategy.FREQUENCY_BASED

            # Size-aware if memory efficiency is poor
            elif memory_util > 0.8 and hit_rate < 0.8:
                return OptimizationStrategy.SIZE_AWARE

            # Cost-aware if cost per hit is high
            elif performance.cost_per_hit > 1.0:
                return OptimizationStrategy.COST_AWARE

            # Hybrid for balanced scenarios
            else:
                return OptimizationStrategy.HYBRID

        except Exception as e:
            self.logger.error(f"Error determining optimal strategy: {e}")
            return OptimizationStrategy.HYBRID

    async def _determine_optimal_policy(self, performance: PerformanceMetrics) -> AdaptivePolicy:
        """Determine optimal adaptive policy based on performance."""
        try:
            # ML-enhanced ARC if ML predictor is available and performing well
            if (self.ml_predictor and
                performance.prefetch_accuracy > 0.7 and
                len(self.performance_history) >= 50):
                return AdaptivePolicy.ML_ARC

            # Standard ARC for general workloads
            elif performance.memory_utilization > 0.7:
                return AdaptivePolicy.ARC

            # LIRS for workloads with strong locality
            elif performance.hit_rate > 0.8:
                return AdaptivePolicy.LIRS

            # CAR for clock-based optimization
            elif performance.avg_response_time > 1.0:
                return AdaptivePolicy.CAR

            # Dynamic policy selection for varying workloads
            else:
                return AdaptivePolicy.DYNAMIC

        except Exception as e:
            self.logger.error(f"Error determining optimal policy: {e}")
            return AdaptivePolicy.ARC

    async def _generate_optimization_recommendations(self, performance: PerformanceMetrics,
                                                   strategy: OptimizationStrategy,
                                                   policy: AdaptivePolicy) -> List[str]:
        """Generate specific optimization recommendations."""
        recommendations = []

        try:
            # Hit rate recommendations
            if performance.hit_rate < self.target_hit_rate:
                diff = self.target_hit_rate - performance.hit_rate
                recommendations.append(
                    f"Increase cache hit rate by {diff:.2%} through improved prefetching"
                )

            # Memory utilization recommendations
            if performance.memory_utilization > self.max_memory_utilization:
                recommendations.append(
                    f"Reduce memory utilization from {performance.memory_utilization:.1%} "
                    f"to below {self.max_memory_utilization:.1%}"
                )

            # Response time recommendations
            if performance.avg_response_time > 0.5:
                recommendations.append(
                    f"Optimize response time from {performance.avg_response_time:.3f}s to <0.5s"
                )

            # Strategy-specific recommendations
            if strategy == OptimizationStrategy.ML_GUIDED:
                recommendations.append("Enable ML-guided cache optimization for intelligent eviction")
            elif strategy == OptimizationStrategy.ADAPTIVE_REPLACEMENT:
                recommendations.append("Implement adaptive replacement cache (ARC) algorithm")
            elif strategy == OptimizationStrategy.FREQUENCY_BASED:
                recommendations.append("Optimize based on access frequency patterns")
            elif strategy == OptimizationStrategy.SIZE_AWARE:
                recommendations.append("Implement size-aware caching with variable-size entries")

            # Policy-specific recommendations
            if policy == AdaptivePolicy.ML_ARC:
                recommendations.append("Use ML-enhanced ARC for adaptive cache replacement")
            elif policy == AdaptivePolicy.LIRS:
                recommendations.append("Implement LIRS algorithm for locality-aware caching")

            # Tier-specific recommendations
            for tier in CacheTier:
                tier_hit_rate = self.tier_metrics[tier]["hit_rate"]
                if tier_hit_rate < 0.8:
                    recommendations.append(
                        f"Optimize {tier.value} tier hit rate (current: {tier_hit_rate:.1%})"
                    )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]

    async def _identify_eviction_candidates(self) -> List[str]:
        """Identify cache entries that should be evicted."""
        candidates = []

        try:
            if not self.cache_system:
                return candidates

            # Get cache statistics
            cache_stats = await self.cache_system.get_cache_stats()

            # Analyze entries in hot and warm tiers
            for tier_name, tier_stats in cache_stats["tier_stats"].items():
                tier = CacheTier(tier_name)

                # Skip cold tier (persistent storage)
                if tier == CacheTier.COLD:
                    continue

                utilization = (
                    tier_stats["size_bytes"] / tier_stats["max_size_bytes"]
                    if tier_stats["max_size_bytes"] > 0 else 0
                )

                # If tier is over-utilized, identify eviction candidates
                if utilization > 0.9:
                    tier_candidates = await self._get_tier_eviction_candidates(tier)
                    candidates.extend(tier_candidates)

            return candidates[:50]  # Limit to top 50 candidates

        except Exception as e:
            self.logger.error(f"Error identifying eviction candidates: {e}")
            return []

    async def _identify_promotion_candidates(self) -> List[str]:
        """Identify cache entries that should be promoted to higher tiers."""
        candidates = []

        try:
            if not self.cache_system or not self.ml_predictor:
                return candidates

            # Get recent cache access events
            if self.redis_client:
                events = await self.redis_client.lrange("cache:events", 0, 499)

                # Analyze access patterns
                access_counts = defaultdict(int)
                recent_accesses = defaultdict(list)

                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)

                for event_data in events:
                    try:
                        event = json.loads(event_data)
                        if event.get('event_type') == 'hit':
                            key = event.get('key', '')
                            timestamp = datetime.fromisoformat(event['timestamp'])

                            if timestamp >= cutoff_time:
                                access_counts[key] += 1
                                recent_accesses[key].append(timestamp)
                    except:
                        continue

                # Identify frequently accessed items in lower tiers
                for key, count in access_counts.items():
                    if count >= 3:  # Accessed 3+ times in last hour
                        # Get ML prediction for promotion
                        try:
                            cache_value = await self.cache_system.get(key)
                            if cache_value:  # Item is cached in some tier
                                # Would need to determine current tier and check if promotion is beneficial
                                candidates.append(key)
                        except:
                            continue

            return candidates[:20]  # Limit to top 20 candidates

        except Exception as e:
            self.logger.error(f"Error identifying promotion candidates: {e}")
            return []

    async def _calculate_tier_rebalancing(self) -> Dict[CacheTier, float]:
        """Calculate optimal tier size rebalancing."""
        rebalancing = {}

        try:
            if not self.cache_system:
                return rebalancing

            # Get current tier utilizations
            cache_stats = await self.cache_system.get_cache_stats()
            tier_stats = cache_stats["tier_stats"]

            # Calculate optimal sizes based on access patterns
            total_accesses = sum(self.tier_metrics[tier]["hit_rate"] for tier in CacheTier)

            for tier in [CacheTier.HOT, CacheTier.WARM]:  # Skip COLD (unlimited)
                current_size = tier_stats[tier.value]["size_bytes"]
                max_size = tier_stats[tier.value]["max_size_bytes"]
                current_util = current_size / max_size if max_size > 0 else 0

                # Calculate optimal utilization based on hit rate
                hit_rate = self.tier_metrics[tier]["hit_rate"]
                optimal_util = min(0.85, 0.5 + (hit_rate * 0.4))  # 50-90% range

                # Calculate rebalancing factor
                if abs(current_util - optimal_util) > 0.1:
                    rebalancing[tier] = optimal_util - current_util

            return rebalancing

        except Exception as e:
            self.logger.error(f"Error calculating tier rebalancing: {e}")
            return {}

    async def _estimate_hit_rate_improvement(self, strategy: OptimizationStrategy,
                                           policy: AdaptivePolicy) -> float:
        """Estimate potential hit rate improvement."""
        try:
            # Use historical data to estimate improvement
            if len(self.performance_history) < 10:
                return 0.05  # Conservative estimate

            current_hit_rate = self.performance_history[-1].hit_rate
            target_improvement = 0.0

            # Strategy-based estimates
            if strategy == OptimizationStrategy.ML_GUIDED:
                target_improvement = 0.08  # 8% improvement with ML
            elif strategy == OptimizationStrategy.ADAPTIVE_REPLACEMENT:
                target_improvement = 0.05  # 5% with ARC
            elif strategy == OptimizationStrategy.FREQUENCY_BASED:
                target_improvement = 0.03  # 3% with frequency optimization
            else:
                target_improvement = 0.02  # 2% with other strategies

            # Policy-based adjustments
            if policy == AdaptivePolicy.ML_ARC:
                target_improvement *= 1.2  # 20% boost with ML-ARC
            elif policy == AdaptivePolicy.LIRS:
                target_improvement *= 1.1  # 10% boost with LIRS

            # Cap improvement at realistic levels
            max_possible = min(0.95 - current_hit_rate, 0.15)  # Max 15% or to 95%
            return min(target_improvement, max_possible)

        except Exception as e:
            self.logger.error(f"Error estimating hit rate improvement: {e}")
            return 0.02

    async def _estimate_memory_savings(self, eviction_candidates: List[str]) -> float:
        """Estimate memory savings from evictions."""
        try:
            if not eviction_candidates or not self.cache_system:
                return 0.0

            total_savings = 0
            for key in eviction_candidates[:10]:  # Estimate for top 10
                # Would need to get actual entry size
                # For now, use average estimate
                estimated_size = 50 * 1024  # 50KB average
                total_savings += estimated_size

            # Convert to percentage of total cache size
            cache_stats = await self.cache_system.get_cache_stats()
            total_cache_size = sum(
                tier["size_bytes"] for tier in cache_stats["tier_stats"].values()
            )

            if total_cache_size > 0:
                return total_savings / total_cache_size
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Error estimating memory savings: {e}")
            return 0.0

    def _calculate_optimization_confidence(self, performance: PerformanceMetrics) -> float:
        """Calculate confidence in optimization recommendations."""
        confidence = 0.8  # Base confidence

        try:
            # Increase confidence with more historical data
            if len(self.performance_history) > 50:
                confidence += 0.1

            # Increase confidence if ML predictor is available and accurate
            if self.ml_predictor and performance.prefetch_accuracy > 0.8:
                confidence += 0.1

            # Decrease confidence if performance is highly variable
            if len(self.performance_history) >= 10:
                hit_rates = [p.hit_rate for p in self.performance_history[-10:]]
                hit_rate_std = np.std(hit_rates)
                if hit_rate_std > 0.1:
                    confidence -= 0.2

            return max(0.5, min(1.0, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.7

    async def _apply_evictions(self, candidates: List[str]) -> bool:
        """Apply eviction recommendations."""
        try:
            if not candidates or not self.cache_system:
                return True

            evicted_count = 0
            for key in candidates[:10]:  # Limit to 10 evictions per optimization
                try:
                    success = await self.cache_system.delete(key)
                    if success:
                        evicted_count += 1
                        self.prom_evictions.labels(tier="optimized", reason="optimization").inc()
                except:
                    continue

            self.logger.info(f"Evicted {evicted_count} entries during optimization")
            return evicted_count > 0

        except Exception as e:
            self.logger.error(f"Error applying evictions: {e}")
            return False

    async def _apply_promotions(self, candidates: List[str]) -> bool:
        """Apply promotion recommendations."""
        try:
            if not candidates or not self.cache_system:
                return True

            promoted_count = 0
            for key in candidates[:5]:  # Limit to 5 promotions per optimization
                try:
                    # Get current value and re-set with higher priority
                    value = await self.cache_system.get(key)
                    if value:
                        # This would trigger tier promotion in the cache system
                        await self.cache_system.set(key, value)
                        promoted_count += 1
                        self.prom_promotions.labels(from_tier="warm", to_tier="hot").inc()
                except:
                    continue

            self.logger.info(f"Promoted {promoted_count} entries during optimization")
            return promoted_count > 0

        except Exception as e:
            self.logger.error(f"Error applying promotions: {e}")
            return False

    async def _apply_tier_rebalancing(self, rebalancing: Dict[CacheTier, float]) -> bool:
        """Apply tier rebalancing recommendations."""
        try:
            if not rebalancing:
                return True

            # This would involve adjusting tier size limits
            # For now, just log the recommendations
            for tier, adjustment in rebalancing.items():
                if abs(adjustment) > 0.05:  # 5% threshold
                    action = "increase" if adjustment > 0 else "decrease"
                    self.logger.info(
                        f"Recommend {action} {tier.value} tier size by {abs(adjustment):.1%}"
                    )

            return True

        except Exception as e:
            self.logger.error(f"Error applying tier rebalancing: {e}")
            return False

    async def _update_cache_policies(self, policy: AdaptivePolicy) -> bool:
        """Update cache policies based on optimization."""
        try:
            # This would update the cache system's eviction policies
            # For now, just update internal state
            self.current_policy = policy
            self.logger.info(f"Updated cache policy to {policy.value}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating cache policies: {e}")
            return False

    async def _get_tier_eviction_candidates(self, tier: CacheTier) -> List[str]:
        """Get eviction candidates for a specific tier."""
        candidates = []

        try:
            # This would analyze entries in the specific tier
            # For now, return empty list as implementation depends on cache internals
            return candidates

        except Exception as e:
            self.logger.error(f"Error getting tier eviction candidates: {e}")
            return []

    async def _calculate_eviction_rate(self) -> float:
        """Calculate recent eviction rate."""
        try:
            if not self.redis_client:
                return 0.0

            events = await self.redis_client.lrange("cache:events", 0, 199)
            eviction_count = 0
            total_events = 0

            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)

            for event_data in events:
                try:
                    event = json.loads(event_data)
                    event_time = datetime.fromisoformat(event['timestamp'])

                    if event_time >= cutoff_time:
                        total_events += 1
                        if event.get('event_type') == 'eviction':
                            eviction_count += 1
                except:
                    continue

            return eviction_count / total_events if total_events > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating eviction rate: {e}")
            return 0.0

    async def _calculate_promotion_rate(self) -> float:
        """Calculate recent promotion rate."""
        # Similar to eviction rate calculation
        return 0.0  # Placeholder

    async def _calculate_prefetch_accuracy(self) -> float:
        """Calculate prefetch accuracy."""
        try:
            if not self.cache_warmer:
                return 0.0

            # Get warming statistics
            warming_stats = await self.cache_warmer.get_warming_stats()
            return warming_stats["metrics"].get("prefetch_accuracy", 0.0)

        except Exception as e:
            self.logger.error(f"Error calculating prefetch accuracy: {e}")
            return 0.0

    async def _calculate_cost_per_hit(self) -> float:
        """Calculate cost per cache hit."""
        # This would calculate actual costs based on infrastructure
        return 0.001  # Placeholder: $0.001 per hit

    async def _calculate_user_satisfaction(self) -> float:
        """Calculate user satisfaction score."""
        try:
            # Based on response times and hit rates
            if not self.performance_history:
                return 0.8

            recent_perf = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history

            avg_hit_rate = np.mean([p.hit_rate for p in recent_perf])
            avg_response_time = np.mean([p.avg_response_time for p in recent_perf])

            # Satisfaction based on hit rate and response time
            hit_satisfaction = min(1.0, avg_hit_rate / 0.85)  # 85% target
            time_satisfaction = max(0.0, 1.0 - (avg_response_time / 2.0))  # 2s max acceptable

            return (hit_satisfaction + time_satisfaction) / 2.0

        except Exception as e:
            self.logger.error(f"Error calculating user satisfaction: {e}")
            return 0.8

    async def _start_background_tasks(self):
        """Start background optimization tasks."""
        tasks = [
            self._optimization_loop(),
            self._performance_monitoring_loop(),
            self._metrics_update_loop()
        ]

        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    async def _optimization_loop(self):
        """Main optimization loop."""
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)

                # Perform optimization analysis
                optimization = await self.optimize_cache_strategy()

                # Apply optimization if confidence is high enough
                if optimization.confidence > 0.7:
                    await self.apply_optimization(optimization)

                self.logger.info(
                    f"Optimization completed: {optimization.strategy.value} "
                    f"(confidence: {optimization.confidence:.2f})"
                )

            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(self.optimization_interval)

    async def _performance_monitoring_loop(self):
        """Performance monitoring loop."""
        while True:
            try:
                await asyncio.sleep(60)  # 1 minute

                # Analyze performance
                performance = await self.analyze_cache_performance()

                # Update tier metrics
                for tier in CacheTier:
                    if self.cache_system:
                        cache_stats = await self.cache_system.get_cache_stats()
                        tier_stats = cache_stats["tier_stats"].get(tier.value, {})

                        # Update Prometheus metrics
                        hit_rate = self.tier_metrics[tier]["hit_rate"]
                        self.prom_hit_rate.labels(tier=tier.value).set(hit_rate)

                        if tier_stats.get("max_size_bytes", 0) > 0:
                            utilization = tier_stats["size_bytes"] / tier_stats["max_size_bytes"]
                            self.prom_memory_efficiency.labels(tier=tier.value).set(utilization)

            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)

    async def _metrics_update_loop(self):
        """Metrics update loop."""
        while True:
            try:
                await asyncio.sleep(300)  # 5 minutes

                # Calculate overall optimization score
                if self.performance_history:
                    recent_hit_rate = self.performance_history[-1].hit_rate
                    optimization_score = min(1.0, recent_hit_rate / self.target_hit_rate)
                    self.prom_optimization_score.set(optimization_score)

            except Exception as e:
                self.logger.error(f"Error updating metrics: {e}")
                await asyncio.sleep(300)

    async def _load_optimization_state(self):
        """Load optimization state from persistent storage."""
        try:
            if self.redis_client:
                # Load performance history
                history_data = await self.redis_client.get("optimizer:performance_history")
                if history_data:
                    history_list = json.loads(history_data)
                    self.performance_history = [
                        PerformanceMetrics(**item) for item in history_list[-100:]
                    ]

                self.logger.info(f"Loaded {len(self.performance_history)} performance records")

        except Exception as e:
            self.logger.error(f"Error loading optimization state: {e}")

    async def _save_optimization_state(self):
        """Save optimization state to persistent storage."""
        try:
            if self.redis_client:
                # Save performance history
                history_data = [asdict(p) for p in self.performance_history[-100:]]
                await self.redis_client.set(
                    "optimizer:performance_history",
                    json.dumps(history_data, default=str),
                    ex=86400 * 7  # 7 days
                )

                self.logger.info("Saved optimization state")

        except Exception as e:
            self.logger.error(f"Error saving optimization state: {e}")

    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "current_strategy": self.current_strategy.value,
            "current_policy": self.current_policy.value,
            "performance_history_size": len(self.performance_history),
            "optimization_history_size": len(self.optimization_history),
            "tier_metrics": self.tier_metrics,
            "recent_performance": asdict(self.performance_history[-1]) if self.performance_history else None,
            "recent_optimization": asdict(self.optimization_history[-1]) if self.optimization_history else None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return stats


async def main():
    """Main entry point for cache optimizer."""
    config = {
        "redis_url": "redis://redis:6379/15",
        "postgres_uri": "postgresql://user:pass@postgres:5432/bev_osint",
        "optimization_interval_seconds": 300,
        "target_hit_rate": 0.85,
        "max_memory_utilization": 0.9
    }

    optimizer = CacheOptimizer(config)

    try:
        await optimizer.initialize()
        # Keep running (in real deployment, this would be part of a larger service)
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await optimizer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())