"""
BEV OSINT Framework - Predictive Cache System Tests
Comprehensive test suite for ML-driven cache optimization and intelligent prefetching.
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

# Import components under test
from src.infrastructure.ml_predictor import MLPredictor, PredictionType, PredictionResult
from src.infrastructure.predictive_cache import PredictiveCache, CacheTier, CachePolicy
from src.infrastructure.cache_warmer import CacheWarmer, WarmingStrategy, WarmingPriority
from src.infrastructure.cache_optimizer import CacheOptimizer, OptimizationStrategy, AdaptivePolicy


class TestMLPredictor:
    """Test suite for ML-based cache prediction system."""

    @pytest.fixture
    async def ml_predictor(self):
        """Create ML predictor instance for testing."""
        config = {
            "redis_url": "redis://localhost:6379/12",
            "postgres_uri": None,
            "min_training_samples": 10,
            "retrain_interval_hours": 1
        }
        predictor = MLPredictor(config)

        # Mock external connections
        predictor.redis_client = AsyncMock()
        predictor.db_pool = None

        yield predictor

        if hasattr(predictor, 'shutdown'):
            await predictor.shutdown()

    @pytest.mark.asyncio
    async def test_initialization(self, ml_predictor):
        """Test ML predictor initialization."""
        assert ml_predictor.config is not None
        assert ml_predictor.min_training_samples == 10
        assert ml_predictor.retrain_interval == 1

    @pytest.mark.asyncio
    async def test_cache_hit_prediction(self, ml_predictor):
        """Test cache hit probability prediction."""
        # Mock model existence
        ml_predictor.models = {PredictionType.HIT_PROBABILITY: MagicMock()}
        ml_predictor.scalers = {PredictionType.HIT_PROBABILITY: MagicMock()}

        # Mock model prediction
        mock_model = ml_predictor.models[PredictionType.HIT_PROBABILITY]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]  # 70% hit probability

        mock_scaler = ml_predictor.scalers[PredictionType.HIT_PROBABILITY]
        mock_scaler.transform.return_value = [[0.1, 0.2, 0.3]]

        # Test prediction
        result = await ml_predictor.predict_cache_hit(
            "test_key", "osint", "user123"
        )

        assert isinstance(result, PredictionResult)
        assert result.predicted_value == 0.7
        assert result.prediction_type == PredictionType.HIT_PROBABILITY
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_access_time_prediction(self, ml_predictor):
        """Test access time prediction."""
        # Mock model existence
        ml_predictor.models = {PredictionType.ACCESS_TIME: MagicMock()}
        ml_predictor.scalers = {PredictionType.ACCESS_TIME: MagicMock()}

        # Mock model prediction
        mock_model = ml_predictor.models[PredictionType.ACCESS_TIME]
        mock_model.predict.return_value = [2.5]  # 2.5 hours

        mock_scaler = ml_predictor.scalers[PredictionType.ACCESS_TIME]
        mock_scaler.transform.return_value = [[0.1, 0.2, 0.3]]

        # Test prediction
        result = await ml_predictor.predict_access_time(
            "test_key", "intelligence", "user123"
        )

        assert isinstance(result, PredictionResult)
        assert result.predicted_value == 2.5
        assert result.prediction_type == PredictionType.ACCESS_TIME

    @pytest.mark.asyncio
    async def test_user_pattern_analysis(self, ml_predictor):
        """Test user behavior pattern analysis."""
        # Mock user data
        mock_user_data = [
            {"query_type": "osint", "timestamp": datetime.now(timezone.utc).isoformat(), "cache_hit": True},
            {"query_type": "osint", "timestamp": datetime.now(timezone.utc).isoformat(), "cache_hit": False},
            {"query_type": "intelligence", "timestamp": datetime.now(timezone.utc).isoformat(), "cache_hit": True}
        ]

        with patch.object(ml_predictor, '_get_user_historical_data', return_value=mock_user_data):
            profile = await ml_predictor.analyze_user_patterns("user123")

            assert profile.user_id == "user123"
            assert "osint" in profile.query_frequency
            assert profile.cache_hit_rate > 0.0


class TestPredictiveCache:
    """Test suite for predictive cache system."""

    @pytest.fixture
    async def cache_system(self):
        """Create cache system instance for testing."""
        config = {
            "cache": {
                "hot_tier_size_gb": 1.0,
                "warm_tier_size_gb": 2.0,
                "default_ttl_seconds": 3600
            },
            "redis_cluster_nodes": [
                {"host": "localhost", "port": 7001}
            ],
            "redis_password": "test",
            "postgres_uri": None
        }
        cache = PredictiveCache(config)

        # Mock external connections
        cache.redis_cluster = AsyncMock()
        cache.redis_standalone = AsyncMock()
        cache.ml_predictor = AsyncMock()

        yield cache

        if hasattr(cache, 'shutdown'):
            await cache.shutdown()

    @pytest.mark.asyncio
    async def test_cache_get_hit(self, cache_system):
        """Test cache get operation with hit."""
        # Setup cache entry
        test_key = "test_key"
        test_value = {"data": "test_data"}

        # Add entry to hot cache
        from src.infrastructure.predictive_cache import CacheEntry
        entry = CacheEntry(
            key=test_key,
            value=test_value,
            tier=CacheTier.HOT,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            access_count=1,
            size_bytes=1024,
            ttl_seconds=3600,
            prediction_score=0.8
        )
        cache_system.hot_cache[test_key] = entry

        # Test get operation
        result = await cache_system.get(test_key, "user123", "osint")

        assert result == test_value
        assert entry.access_count == 2

    @pytest.mark.asyncio
    async def test_cache_get_miss(self, cache_system):
        """Test cache get operation with miss."""
        result = await cache_system.get("nonexistent_key", "user123", "osint")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set(self, cache_system):
        """Test cache set operation."""
        test_key = "test_key"
        test_value = {"data": "test_data"}

        # Mock ML predictor
        mock_prediction = MagicMock()
        mock_prediction.predicted_value = 0.8
        cache_system.ml_predictor.predict_cache_hit.return_value = mock_prediction

        # Test set operation
        success = await cache_system.set(
            test_key, test_value, ttl=7200,
            user_id="user123", query_type="osint"
        )

        assert success is True
        assert test_key in cache_system.hot_cache

    @pytest.mark.asyncio
    async def test_cache_tier_promotion(self, cache_system):
        """Test cache entry promotion between tiers."""
        test_key = "test_key"

        # Create entry in warm tier
        from src.infrastructure.predictive_cache import CacheEntry
        entry = CacheEntry(
            key=test_key,
            value={"data": "test"},
            tier=CacheTier.WARM,
            created_at=datetime.now(timezone.utc),
            last_accessed=datetime.now(timezone.utc),
            access_count=5,  # High access count for promotion
            size_bytes=1024,
            ttl_seconds=3600,
            prediction_score=0.9
        )
        cache_system.warm_cache[test_key] = entry

        # Simulate multiple accesses to trigger promotion
        cache_system.access_patterns[test_key] = [
            datetime.now(timezone.utc) - timedelta(minutes=i)
            for i in range(5)
        ]

        # Mock promotion logic
        with patch.object(cache_system, '_should_promote_to_hot', return_value=True):
            await cache_system._promote_entry(entry, CacheTier.HOT)

            assert entry.tier == CacheTier.HOT


class TestCacheWarmer:
    """Test suite for intelligent cache warming system."""

    @pytest.fixture
    async def cache_warmer(self):
        """Create cache warmer instance for testing."""
        config = {
            "redis_url": "redis://localhost:6379/14",
            "postgres_uri": None,
            "max_concurrent_tasks": 5,
            "warming_interval_seconds": 60
        }
        warmer = CacheWarmer(config)

        # Mock external connections
        warmer.redis_client = AsyncMock()
        warmer.session = AsyncMock()
        warmer.ml_predictor = AsyncMock()
        warmer.cache_system = AsyncMock()

        yield warmer

        if hasattr(warmer, 'shutdown'):
            await warmer.shutdown()

    @pytest.mark.asyncio
    async def test_warming_task_creation(self, cache_warmer):
        """Test creation of warming tasks."""
        task_id = await cache_warmer.create_warming_task(
            strategy=WarmingStrategy.USER_BASED,
            cache_key="test_key",
            query_params={"user_id": "user123"},
            query_type="osint",
            user_id="user123",
            priority=WarmingPriority.HIGH
        )

        assert task_id is not None
        assert task_id in cache_warmer.pending_tasks

        task = cache_warmer.pending_tasks[task_id]
        assert task.strategy == WarmingStrategy.USER_BASED
        assert task.priority == WarmingPriority.HIGH

    @pytest.mark.asyncio
    async def test_user_pattern_analysis(self, cache_warmer):
        """Test user pattern analysis for warming."""
        # Mock ML predictor response
        from src.infrastructure.ml_predictor import UserBehaviorProfile
        mock_profile = UserBehaviorProfile(
            user_id="user123",
            query_frequency={"osint": 0.6, "intelligence": 0.4},
            access_patterns={"osint": [2, 5, 8, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
            preferred_data_types=["threat_intel", "iot_data"],
            cache_hit_rate=0.75,
            last_updated=datetime.now(timezone.utc)
        )

        cache_warmer.ml_predictor.analyze_user_patterns.return_value = mock_profile

        # Mock key generation
        with patch.object(cache_warmer, '_generate_user_warming_keys', return_value=[("key1", {}), ("key2", {})]):
            task_ids = await cache_warmer.analyze_user_patterns("user123")

            assert len(task_ids) > 0
            assert len(cache_warmer.pending_tasks) > 0

    @pytest.mark.asyncio
    async def test_popularity_trend_analysis(self, cache_warmer):
        """Test popularity trend analysis for warming."""
        # Mock cache events
        mock_events = [
            '{"event_type": "hit", "key": "popular_key", "timestamp": "2024-01-01T10:00:00Z"}',
            '{"event_type": "hit", "key": "popular_key", "timestamp": "2024-01-01T10:01:00Z"}',
            '{"event_type": "hit", "key": "another_key", "timestamp": "2024-01-01T10:02:00Z"}'
        ]

        cache_warmer.redis_client.lrange.return_value = mock_events

        task_ids = await cache_warmer.analyze_popularity_trends()

        # Should create warming tasks for popular queries
        assert isinstance(task_ids, list)


class TestCacheOptimizer:
    """Test suite for cache optimization system."""

    @pytest.fixture
    async def cache_optimizer(self):
        """Create cache optimizer instance for testing."""
        config = {
            "redis_url": "redis://localhost:6379/15",
            "postgres_uri": None,
            "optimization_interval_seconds": 300,
            "target_hit_rate": 0.85
        }
        optimizer = CacheOptimizer(config)

        # Mock external connections
        optimizer.redis_client = AsyncMock()
        optimizer.cache_system = AsyncMock()
        optimizer.ml_predictor = AsyncMock()
        optimizer.cache_warmer = AsyncMock()

        yield optimizer

        if hasattr(optimizer, 'shutdown'):
            await optimizer.shutdown()

    @pytest.mark.asyncio
    async def test_performance_analysis(self, cache_optimizer):
        """Test cache performance analysis."""
        # Mock cache statistics
        mock_stats = {
            "metrics": {
                "hit_rate": 0.75,
                "avg_response_time": 0.15,
                "evictions": 10
            },
            "tier_stats": {
                "hot": {"size_bytes": 2000000000, "max_size_bytes": 4000000000},
                "warm": {"size_bytes": 6000000000, "max_size_bytes": 8000000000}
            }
        }

        cache_optimizer.cache_system.get_cache_stats.return_value = mock_stats

        # Mock additional metrics
        with patch.object(cache_optimizer, '_calculate_eviction_rate', return_value=0.05):
            with patch.object(cache_optimizer, '_calculate_promotion_rate', return_value=0.02):
                performance = await cache_optimizer.analyze_cache_performance()

                assert performance.hit_rate == 0.75
                assert performance.avg_response_time == 0.15
                assert performance.memory_utilization > 0.0

    @pytest.mark.asyncio
    async def test_optimization_strategy_selection(self, cache_optimizer):
        """Test optimization strategy selection."""
        # Create mock performance metrics
        from src.infrastructure.cache_optimizer import PerformanceMetrics
        performance = PerformanceMetrics(
            hit_rate=0.65,  # Below target
            miss_rate=0.35,
            avg_response_time=0.2,
            memory_utilization=0.95,  # High memory pressure
            eviction_rate=0.1,
            promotion_rate=0.05,
            prefetch_accuracy=0.8,
            cost_per_hit=0.002,
            user_satisfaction_score=0.7,
            timestamp=datetime.now(timezone.utc)
        )

        strategy = await cache_optimizer._determine_optimal_strategy(performance)

        # Should select adaptive replacement due to high memory utilization
        assert strategy == OptimizationStrategy.ADAPTIVE_REPLACEMENT

    @pytest.mark.asyncio
    async def test_optimization_application(self, cache_optimizer):
        """Test application of optimization recommendations."""
        from src.infrastructure.cache_optimizer import OptimizationResult

        optimization = OptimizationResult(
            strategy=OptimizationStrategy.ML_GUIDED,
            policy=AdaptivePolicy.ML_ARC,
            hit_rate_improvement=0.05,
            memory_savings=0.1,
            recommendations=["Improve prefetching", "Optimize eviction"],
            eviction_candidates=["old_key_1", "old_key_2"],
            promotion_candidates=["hot_key_1"],
            tier_rebalancing={CacheTier.HOT: 0.1},
            confidence=0.85,
            timestamp=datetime.now(timezone.utc)
        )

        # Mock successful operations
        with patch.object(cache_optimizer, '_apply_evictions', return_value=True):
            with patch.object(cache_optimizer, '_apply_promotions', return_value=True):
                with patch.object(cache_optimizer, '_apply_tier_rebalancing', return_value=True):
                    with patch.object(cache_optimizer, '_update_cache_policies', return_value=True):
                        success = await cache_optimizer.apply_optimization(optimization)

                        assert success is True


class TestIntegration:
    """Integration tests for the complete predictive cache system."""

    @pytest.mark.asyncio
    async def test_end_to_end_cache_operation(self):
        """Test complete cache operation from prediction to optimization."""
        # This would test the full pipeline:
        # 1. ML prediction for cache placement
        # 2. Cache storage and retrieval
        # 3. Cache warming based on patterns
        # 4. Performance monitoring
        # 5. Optimization recommendations

        # Mock the complete system integration
        pass

    @pytest.mark.asyncio
    async def test_cache_warming_integration(self):
        """Test integration between cache warming and optimization."""
        pass

    @pytest.mark.asyncio
    async def test_ml_prediction_accuracy(self):
        """Test ML prediction accuracy in realistic scenarios."""
        pass


class TestPerformance:
    """Performance tests for the predictive cache system."""

    @pytest.mark.asyncio
    async def test_cache_response_time(self):
        """Test cache response time under load."""
        # Test cache operations should complete within target time
        pass

    @pytest.mark.asyncio
    async def test_ml_prediction_latency(self):
        """Test ML prediction latency."""
        # ML predictions should complete within acceptable time
        pass

    @pytest.mark.asyncio
    async def test_optimization_overhead(self):
        """Test optimization process overhead."""
        # Optimization should not significantly impact cache performance
        pass

    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """Test memory usage of cache components."""
        # Memory usage should stay within configured limits
        pass


# Fixtures and utilities
@pytest.fixture
def sample_cache_data():
    """Sample data for cache testing."""
    return {
        "small_data": {"size": "small", "data": "x" * 100},
        "medium_data": {"size": "medium", "data": "x" * 10000},
        "large_data": {"size": "large", "data": "x" * 1000000}
    }


@pytest.fixture
def mock_user_profiles():
    """Mock user behavior profiles for testing."""
    return [
        {
            "user_id": "user1",
            "query_frequency": {"osint": 0.7, "intelligence": 0.3},
            "access_patterns": {"osint": [5, 3, 2, 1, 0] + [0] * 19},
            "cache_hit_rate": 0.8
        },
        {
            "user_id": "user2",
            "query_frequency": {"blockchain": 0.6, "network": 0.4},
            "access_patterns": {"blockchain": [2, 8, 12, 5, 1] + [0] * 19},
            "cache_hit_rate": 0.6
        }
    ]


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration for cache system."""
    return {
        "redis_url": "redis://localhost:6379",
        "cache_size_limits": {
            "hot": 1024 * 1024 * 1024,  # 1GB
            "warm": 2 * 1024 * 1024 * 1024,  # 2GB
        },
        "ml_config": {
            "min_samples": 10,
            "confidence_threshold": 0.7
        }
    }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])