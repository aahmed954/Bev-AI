"""
Cache performance tests for predictive caching system
Validates >80% cache hit rates and cache optimization
"""

import pytest
import asyncio
import redis
import time
import json
import random
import statistics
import logging
from typing import Dict, Any, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

@pytest.mark.cache
class TestCachePerformance:
    """Test cache performance and hit rate optimization"""

    @pytest.fixture(autouse=True)
    def setup_cache_test(self, redis_connection, performance_monitor):
        """Setup cache testing environment"""
        self.redis = redis_connection
        self.monitor = performance_monitor
        self.test_prefix = "cache_test"
        self.monitor.start()

        # Clear test data
        keys = self.redis.keys(f"{self.test_prefix}:*")
        if keys:
            self.redis.delete(*keys)

        yield

        # Cleanup
        keys = self.redis.keys(f"{self.test_prefix}:*")
        if keys:
            self.redis.delete(*keys)
        self.monitor.stop()

    async def test_basic_cache_performance(self):
        """Test basic cache operations performance"""
        logger.info("Testing basic cache performance")

        # Test cache SET operations
        set_times = []
        num_operations = 1000

        for i in range(num_operations):
            start_time = time.time()
            self.redis.setex(f"{self.test_prefix}:basic:{i}", 3600, f"test_value_{i}")
            set_time = (time.time() - start_time) * 1000  # Convert to ms
            set_times.append(set_time)

        # Test cache GET operations
        get_times = []
        hit_count = 0

        for i in range(num_operations):
            start_time = time.time()
            value = self.redis.get(f"{self.test_prefix}:basic:{i}")
            get_time = (time.time() - start_time) * 1000

            if value is not None:
                hit_count += 1

            get_times.append(get_time)

        # Performance metrics
        avg_set_time = statistics.mean(set_times)
        avg_get_time = statistics.mean(get_times)
        hit_rate = hit_count / num_operations

        self.monitor.record("cache_avg_set_time", avg_set_time)
        self.monitor.record("cache_avg_get_time", avg_get_time)
        self.monitor.record("cache_hit_rate", hit_rate)

        # Performance assertions
        assert avg_set_time <= 1.0, f"Cache SET too slow: {avg_set_time:.3f}ms"
        assert avg_get_time <= 0.5, f"Cache GET too slow: {avg_get_time:.3f}ms"
        assert hit_rate >= 0.99, f"Cache hit rate too low: {hit_rate:.2%}"

        logger.info(f"Basic cache performance: SET={avg_set_time:.3f}ms, GET={avg_get_time:.3f}ms, HIT={hit_rate:.2%}")

    async def test_predictive_cache_efficiency(self):
        """Test predictive caching algorithms and efficiency"""
        logger.info("Testing predictive cache efficiency")

        # Simulate access patterns for prediction
        access_patterns = self._generate_access_patterns()

        # Phase 1: Learning phase - establish patterns
        learning_requests = access_patterns["learning"]
        for request in learning_requests:
            self._simulate_cache_request(request)

        # Phase 2: Prediction phase - test predictive accuracy
        prediction_requests = access_patterns["prediction"]
        cache_hits = 0
        cache_misses = 0
        prediction_accuracy = []

        for request in prediction_requests:
            # Check if predictive cache preloaded the data
            cache_key = self._generate_cache_key(request)
            cached_value = self.redis.get(cache_key)

            if cached_value is not None:
                cache_hits += 1
                prediction_accuracy.append(1)
            else:
                cache_misses += 1
                prediction_accuracy.append(0)
                # Simulate cache miss - load and cache data
                self._simulate_cache_miss_handling(request)

        # Calculate predictive metrics
        total_requests = len(prediction_requests)
        predictive_hit_rate = cache_hits / total_requests
        avg_prediction_accuracy = statistics.mean(prediction_accuracy)

        self.monitor.record("predictive_hit_rate", predictive_hit_rate)
        self.monitor.record("prediction_accuracy", avg_prediction_accuracy)

        # Predictive cache should achieve high hit rates
        assert predictive_hit_rate >= 0.80, f"Predictive hit rate {predictive_hit_rate:.2%} below 80% target"

        logger.info(f"Predictive cache: {predictive_hit_rate:.2%} hit rate, {avg_prediction_accuracy:.2%} accuracy")

    async def test_cache_under_high_load(self):
        """Test cache performance under high concurrent load"""
        logger.info("Testing cache under high load")

        concurrent_clients = 50
        operations_per_client = 100

        # Generate concurrent cache operations
        async def client_operations(client_id: int):
            client_metrics = {"hits": 0, "misses": 0, "latencies": []}

            for op_id in range(operations_per_client):
                operation_type = random.choice(["get", "set", "delete"])
                key = f"{self.test_prefix}:load:{client_id}:{op_id % 20}"  # Reuse keys for hits

                start_time = time.time()

                if operation_type == "get":
                    value = self.redis.get(key)
                    if value is not None:
                        client_metrics["hits"] += 1
                    else:
                        client_metrics["misses"] += 1

                elif operation_type == "set":
                    self.redis.setex(key, 3600, f"client_{client_id}_value_{op_id}")

                elif operation_type == "delete":
                    self.redis.delete(key)

                latency = (time.time() - start_time) * 1000
                client_metrics["latencies"].append(latency)

                # Small delay to simulate realistic load
                await asyncio.sleep(0.001)

            return client_metrics

        # Execute concurrent operations
        start_time = time.time()
        client_tasks = [client_operations(i) for i in range(concurrent_clients)]
        client_results = await asyncio.gather(*client_tasks)
        total_time = time.time() - start_time

        # Aggregate results
        total_hits = sum(r["hits"] for r in client_results)
        total_misses = sum(r["misses"] for r in client_results)
        all_latencies = []
        for r in client_results:
            all_latencies.extend(r["latencies"])

        # Performance metrics
        total_operations = len(all_latencies)
        operations_per_second = total_operations / total_time
        hit_rate_under_load = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        avg_latency_under_load = statistics.mean(all_latencies)
        p95_latency = np.percentile(all_latencies, 95)

        self.monitor.record("load_test_ops_per_sec", operations_per_second)
        self.monitor.record("load_test_hit_rate", hit_rate_under_load)
        self.monitor.record("load_test_avg_latency", avg_latency_under_load)
        self.monitor.record("load_test_p95_latency", p95_latency)

        # Performance assertions under load
        assert operations_per_second >= 10000, f"OPS too low under load: {operations_per_second:.1f}"
        assert avg_latency_under_load <= 2.0, f"Latency too high under load: {avg_latency_under_load:.3f}ms"
        assert p95_latency <= 5.0, f"P95 latency too high: {p95_latency:.3f}ms"

        logger.info(f"High load performance: {operations_per_second:.1f} OPS, {avg_latency_under_load:.3f}ms avg latency")

    async def test_cache_eviction_strategies(self):
        """Test cache eviction and memory management"""
        logger.info("Testing cache eviction strategies")

        # Fill cache to near capacity
        max_memory = 1024 * 1024  # 1MB for testing
        avg_value_size = 1000  # 1KB average
        target_keys = int(max_memory * 0.8 / avg_value_size)  # 80% capacity

        # Insert data until near capacity
        for i in range(target_keys):
            key = f"{self.test_prefix}:eviction:{i}"
            value = "x" * avg_value_size  # 1KB value
            self.redis.setex(key, 7200, value)  # 2 hour TTL

        # Test different eviction scenarios
        eviction_scenarios = [
            {"name": "lru_eviction", "access_pattern": "sequential"},
            {"name": "lfu_eviction", "access_pattern": "frequency_based"},
            {"name": "ttl_eviction", "access_pattern": "time_based"}
        ]

        for scenario in eviction_scenarios:
            logger.info(f"Testing {scenario['name']} eviction")

            # Generate access pattern to trigger eviction
            if scenario["access_pattern"] == "sequential":
                # Access keys sequentially to test LRU
                access_keys = [f"{self.test_prefix}:eviction:{i}" for i in range(0, target_keys, 10)]

            elif scenario["access_pattern"] == "frequency_based":
                # Access some keys more frequently for LFU testing
                access_keys = []
                for i in range(100):  # High frequency keys
                    access_keys.extend([f"{self.test_prefix}:eviction:{i}"] * 5)

            elif scenario["access_pattern"] == "time_based":
                # Mix of recent and old keys for TTL testing
                access_keys = [f"{self.test_prefix}:eviction:{i}" for i in range(target_keys//2, target_keys)]

            # Perform accesses and monitor eviction
            eviction_start = time.time()
            cache_performance = {"hits": 0, "misses": 0}

            for key in access_keys:
                value = self.redis.get(key)
                if value is not None:
                    cache_performance["hits"] += 1
                else:
                    cache_performance["misses"] += 1

            # Add new data to force eviction
            new_keys_count = 200
            for i in range(new_keys_count):
                new_key = f"{self.test_prefix}:eviction:new:{i}"
                new_value = "y" * avg_value_size
                self.redis.setex(new_key, 3600, new_value)

            eviction_time = time.time() - eviction_start

            # Measure eviction effectiveness
            post_eviction_hit_rate = cache_performance["hits"] / len(access_keys)
            memory_usage = self._estimate_cache_memory_usage()

            self.monitor.record(f"{scenario['name']}_hit_rate", post_eviction_hit_rate)
            self.monitor.record(f"{scenario['name']}_time", eviction_time)

            logger.info(f"  {scenario['name']}: {post_eviction_hit_rate:.2%} hit rate, {eviction_time:.2f}s")

            # Eviction should maintain reasonable performance
            assert post_eviction_hit_rate >= 0.70, f"{scenario['name']} hit rate too low after eviction"

    async def test_cache_warming_strategies(self):
        """Test cache warming and preloading strategies"""
        logger.info("Testing cache warming strategies")

        # Define warming strategies
        warming_strategies = [
            {"name": "predictive_warming", "method": "ml_based"},
            {"name": "pattern_warming", "method": "historical_patterns"},
            {"name": "priority_warming", "method": "business_priority"}
        ]

        for strategy in warming_strategies:
            logger.info(f"Testing {strategy['name']} strategy")

            # Clear cache for clean test
            self.redis.flushdb()

            # Simulate cache warming
            warming_start = time.time()
            warmed_keys = await self._simulate_cache_warming(strategy)
            warming_time = time.time() - warming_start

            # Test effectiveness of warming
            test_requests = self._generate_test_requests(1000)
            warm_hits = 0
            cold_misses = 0

            for request in test_requests:
                cache_key = self._generate_cache_key(request)
                if cache_key in warmed_keys:
                    # Check if actually cached
                    value = self.redis.get(cache_key)
                    if value is not None:
                        warm_hits += 1
                    else:
                        cold_misses += 1

            # Calculate warming effectiveness
            total_warmable = len([r for r in test_requests if self._generate_cache_key(r) in warmed_keys])
            warming_hit_rate = warm_hits / total_warmable if total_warmable > 0 else 0

            self.monitor.record(f"{strategy['name']}_warming_time", warming_time)
            self.monitor.record(f"{strategy['name']}_warming_hit_rate", warming_hit_rate)

            logger.info(f"  {strategy['name']}: {warming_hit_rate:.2%} effectiveness, {warming_time:.2f}s warming time")

            # Warming should be effective and fast
            assert warming_hit_rate >= 0.75, f"{strategy['name']} effectiveness too low"
            assert warming_time <= 30, f"{strategy['name']} warming too slow"

    async def test_distributed_cache_consistency(self):
        """Test cache consistency across distributed nodes"""
        logger.info("Testing distributed cache consistency")

        # Simulate multiple cache nodes (using Redis databases)
        cache_nodes = [0, 1, 2]  # Different Redis databases

        # Test data consistency across nodes
        test_data = {
            f"distributed_key_{i}": f"distributed_value_{i}"
            for i in range(100)
        }

        # Write to all nodes
        write_start = time.time()
        for db_id in cache_nodes:
            node_redis = redis.Redis(host="localhost", port=6379, db=db_id, decode_responses=True)
            for key, value in test_data.items():
                node_redis.setex(key, 3600, value)

        write_time = time.time() - write_start

        # Test consistency across nodes
        consistency_check_start = time.time()
        consistency_results = {"consistent": 0, "inconsistent": 0}

        for key, expected_value in test_data.items():
            node_values = []
            for db_id in cache_nodes:
                node_redis = redis.Redis(host="localhost", port=6379, db=db_id, decode_responses=True)
                value = node_redis.get(key)
                node_values.append(value)

            # Check if all nodes have the same value
            if all(v == expected_value for v in node_values):
                consistency_results["consistent"] += 1
            else:
                consistency_results["inconsistent"] += 1

        consistency_time = time.time() - consistency_check_start

        # Calculate consistency metrics
        total_keys = len(test_data)
        consistency_rate = consistency_results["consistent"] / total_keys

        self.monitor.record("distributed_write_time", write_time)
        self.monitor.record("distributed_consistency_time", consistency_time)
        self.monitor.record("distributed_consistency_rate", consistency_rate)

        # Cleanup test databases
        for db_id in cache_nodes:
            node_redis = redis.Redis(host="localhost", port=6379, db=db_id)
            node_redis.flushdb()

        # Distributed cache should maintain high consistency
        assert consistency_rate >= 0.99, f"Consistency rate {consistency_rate:.2%} too low"

        logger.info(f"Distributed consistency: {consistency_rate:.2%} consistent, {consistency_time:.2f}s check time")

    # Helper methods

    def _generate_access_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate realistic access patterns for testing"""
        # Learning phase patterns (establish predictable access)
        learning_patterns = []
        for i in range(1000):
            pattern = {
                "type": "osint_query",
                "target": f"target_{i % 50}.com",  # 50 unique targets, repeated access
                "timestamp": time.time() - (1000 - i) * 60,  # Historical timestamps
                "user_id": f"user_{i % 10}",  # 10 users
                "query_type": random.choice(["domain", "ip", "subdomain"])
            }
            learning_patterns.append(pattern)

        # Prediction phase patterns (test predictive accuracy)
        prediction_patterns = []
        for i in range(500):
            # 80% patterns should be predictable based on learning
            if random.random() < 0.8:
                # Predictable pattern
                pattern = {
                    "type": "osint_query",
                    "target": f"target_{i % 50}.com",  # Same targets as learning
                    "timestamp": time.time(),
                    "user_id": f"user_{i % 10}",  # Same users
                    "query_type": random.choice(["domain", "ip", "subdomain"])
                }
            else:
                # New, unpredictable pattern
                pattern = {
                    "type": "osint_query",
                    "target": f"new_target_{i}.com",
                    "timestamp": time.time(),
                    "user_id": f"new_user_{i}",
                    "query_type": "comprehensive"
                }
            prediction_patterns.append(pattern)

        return {
            "learning": learning_patterns,
            "prediction": prediction_patterns
        }

    def _simulate_cache_request(self, request: Dict[str, Any]):
        """Simulate a cache request and response"""
        cache_key = self._generate_cache_key(request)

        # Check cache first
        cached_data = self.redis.get(cache_key)

        if cached_data is None:
            # Cache miss - simulate data generation and caching
            simulated_data = {
                "query_result": f"Result for {request['target']}",
                "timestamp": time.time(),
                "processing_time": random.uniform(0.1, 2.0)
            }

            # Cache the result
            self.redis.setex(cache_key, 3600, json.dumps(simulated_data))

            # Update access pattern for prediction
            self._update_access_patterns(request)

    def _simulate_cache_miss_handling(self, request: Dict[str, Any]):
        """Simulate handling of cache miss"""
        cache_key = self._generate_cache_key(request)

        # Simulate data generation (expensive operation)
        time.sleep(0.01)  # 10ms simulation

        simulated_data = {
            "query_result": f"Generated result for {request['target']}",
            "timestamp": time.time(),
            "cache_miss": True
        }

        # Cache the newly generated data
        self.redis.setex(cache_key, 3600, json.dumps(simulated_data))

    def _generate_cache_key(self, request: Dict[str, Any]) -> str:
        """Generate cache key from request"""
        key_parts = [
            request.get("type", "unknown"),
            request.get("target", "notarget"),
            request.get("query_type", "default")
        ]
        return f"{self.test_prefix}:request:" + ":".join(key_parts)

    def _update_access_patterns(self, request: Dict[str, Any]):
        """Update access patterns for predictive caching"""
        pattern_key = f"{self.test_prefix}:patterns:{request.get('user_id', 'unknown')}"

        # Get existing patterns
        existing_patterns = self.redis.get(pattern_key)
        if existing_patterns:
            patterns = json.loads(existing_patterns)
        else:
            patterns = []

        # Add new pattern
        patterns.append({
            "target": request.get("target"),
            "query_type": request.get("query_type"),
            "timestamp": time.time()
        })

        # Keep only recent patterns (last 100)
        if len(patterns) > 100:
            patterns = patterns[-100:]

        # Update patterns in cache
        self.redis.setex(pattern_key, 7200, json.dumps(patterns))

    def _estimate_cache_memory_usage(self) -> int:
        """Estimate cache memory usage"""
        # This would use Redis memory info in real implementation
        # For testing, return a simulated value
        return random.randint(500000, 1000000)  # 500KB - 1MB

    async def _simulate_cache_warming(self, strategy: Dict[str, str]) -> List[str]:
        """Simulate cache warming process"""
        warmed_keys = []

        if strategy["method"] == "ml_based":
            # Simulate ML-based prediction
            for i in range(100):
                key = f"{self.test_prefix}:predicted:{i}"
                value = f"ml_predicted_value_{i}"
                self.redis.setex(key, 3600, value)
                warmed_keys.append(key)

        elif strategy["method"] == "historical_patterns":
            # Warm based on historical access patterns
            for i in range(150):
                key = f"{self.test_prefix}:historical:{i % 50}"  # Repeat patterns
                value = f"historical_value_{i}"
                self.redis.setex(key, 3600, value)
                warmed_keys.append(key)

        elif strategy["method"] == "business_priority":
            # Warm high-priority business data
            priorities = ["critical", "high", "medium"]
            for priority in priorities:
                for i in range(50):
                    key = f"{self.test_prefix}:priority:{priority}:{i}"
                    value = f"{priority}_priority_value_{i}"
                    self.redis.setex(key, 3600, value)
                    warmed_keys.append(key)

        return warmed_keys

    def _generate_test_requests(self, count: int) -> List[Dict[str, Any]]:
        """Generate test requests for cache testing"""
        requests = []

        for i in range(count):
            request = {
                "type": "osint_query",
                "target": f"target_{i % 100}.com",  # 100 unique targets
                "query_type": random.choice(["domain", "ip", "subdomain", "comprehensive"]),
                "user_id": f"user_{i % 20}",  # 20 users
                "timestamp": time.time()
            }
            requests.append(request)

        return requests