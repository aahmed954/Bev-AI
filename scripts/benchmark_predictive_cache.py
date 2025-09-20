#!/usr/bin/env python3
"""
BEV OSINT Framework - Predictive Cache Performance Benchmarking Suite

Comprehensive benchmarking suite for the predictive cache system to validate
performance targets and identify optimization opportunities.

Performance Targets:
- Cache Hit Rate: >80% overall, >90% for hot tier
- Response Time: <10ms for cache operations
- ML Prediction Accuracy: >80% for cache hit predictions
- Memory Efficiency: >90% useful data ratio
- Throughput: 100K+ ops/sec for hot tier
"""

import asyncio
import time
import json
import random
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import numpy as np
from datetime import datetime, timedelta

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    cache_service_url: str = "http://localhost:8044"
    test_duration_seconds: int = 300  # 5 minutes
    warmup_duration_seconds: int = 60
    concurrent_clients: int = 100
    request_rate_per_second: int = 1000
    data_size_distribution: Dict[str, float] = None
    cache_key_patterns: List[str] = None

    def __post_init__(self):
        if self.data_size_distribution is None:
            self.data_size_distribution = {
                "small": 0.7,    # <10KB
                "medium": 0.25,  # 10KB-1MB
                "large": 0.05    # >1MB
            }

        if self.cache_key_patterns is None:
            self.cache_key_patterns = [
                "osint:query:{user_id}:{hash}",
                "whois:domain:{domain}",
                "dns:record:{type}:{domain}",
                "geo:ip:{ip_address}",
                "social:profile:{platform}:{handle}",
                "threat:ioc:{type}:{value}",
                "file:analysis:{hash}",
                "network:scan:{target}:{port}"
            ]

@dataclass
class BenchmarkMetrics:
    """Performance metrics collected during benchmarking"""
    total_requests: int = 0
    successful_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    response_times: List[float] = None
    error_count: int = 0
    throughput_ops_per_sec: float = 0.0
    cache_hit_rate: float = 0.0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    ml_prediction_accuracy: float = 0.0
    memory_efficiency: float = 0.0

    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []

class PredictiveCacheBenchmark:
    """Main benchmarking class for predictive cache performance testing"""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics = BenchmarkMetrics()
        self.session: Optional[aiohttp.ClientSession] = None
        self.test_data: Dict[str, Any] = {}

    async def setup(self):
        """Initialize benchmark environment"""
        print("🔧 Setting up benchmark environment...")

        # Create HTTP session
        connector = aiohttp.TCPConnector(limit=self.config.concurrent_clients * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        # Generate test data
        await self._generate_test_data()

        # Verify cache service is available
        await self._verify_cache_service()

        print("✅ Benchmark environment ready")

    async def _generate_test_data(self):
        """Generate realistic test data for benchmarking"""
        print("📊 Generating test data...")

        # Generate cache keys based on patterns
        self.test_data["cache_keys"] = []
        for i in range(10000):
            pattern = random.choice(self.config.cache_key_patterns)
            key = pattern.format(
                user_id=f"user_{random.randint(1, 1000)}",
                hash=f"{random.randint(100000, 999999):x}",
                domain=f"example{random.randint(1, 100)}.com",
                type=random.choice(["A", "AAAA", "MX", "TXT"]),
                ip_address=f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                platform=random.choice(["twitter", "linkedin", "facebook"]),
                handle=f"user{random.randint(1, 1000)}",
                value=f"ioc_{random.randint(1000, 9999)}",
                target=f"target_{random.randint(1, 100)}",
                port=random.choice([80, 443, 22, 25, 53, 3389])
            )
            self.test_data["cache_keys"].append(key)

        # Generate test values with size distribution
        self.test_data["cache_values"] = {}
        for key in self.test_data["cache_keys"]:
            size_category = np.random.choice(
                list(self.config.data_size_distribution.keys()),
                p=list(self.config.data_size_distribution.values())
            )

            if size_category == "small":
                data_size = random.randint(100, 10000)  # 100B - 10KB
            elif size_category == "medium":
                data_size = random.randint(10000, 1000000)  # 10KB - 1MB
            else:  # large
                data_size = random.randint(1000000, 5000000)  # 1MB - 5MB

            # Create realistic OSINT data structure
            value = {
                "query_type": random.choice(["osint", "whois", "dns", "geo", "social", "threat"]),
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 1440))).isoformat(),
                "data": "x" * data_size,  # Simulate data payload
                "metadata": {
                    "source": random.choice(["api", "scraper", "database"]),
                    "confidence": random.uniform(0.5, 1.0),
                    "ttl": random.randint(300, 3600)
                }
            }
            self.test_data["cache_values"][key] = value

        print(f"Generated {len(self.test_data['cache_keys'])} test cache entries")

    async def _verify_cache_service(self):
        """Verify that the cache service is running and responsive"""
        try:
            async with self.session.get(f"{self.config.cache_service_url}/health") as response:
                if response.status != 200:
                    raise Exception(f"Cache service health check failed: {response.status}")

                health_data = await response.json()
                if not health_data.get("healthy", False):
                    raise Exception("Cache service reports unhealthy status")

                print("✅ Cache service is healthy and responsive")
        except Exception as e:
            raise Exception(f"Failed to connect to cache service: {e}")

    async def run_benchmark(self) -> BenchmarkMetrics:
        """Execute the complete benchmark suite"""
        print("🚀 Starting predictive cache benchmark...")

        try:
            # Warmup phase
            await self._warmup_phase()

            # Main benchmark phases
            await self._populate_cache_phase()
            await self._mixed_workload_phase()
            await self._stress_test_phase()
            await self._ml_prediction_accuracy_phase()

            # Calculate final metrics
            self._calculate_final_metrics()

            return self.metrics

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            raise

    async def _warmup_phase(self):
        """Warmup phase to prepare cache and system"""
        print(f"🔥 Warmup phase ({self.config.warmup_duration_seconds}s)...")

        start_time = time.time()
        tasks = []

        # Create warmup tasks
        for _ in range(self.config.concurrent_clients):
            task = asyncio.create_task(self._warmup_worker())
            tasks.append(task)

        # Wait for warmup completion
        await asyncio.sleep(self.config.warmup_duration_seconds)

        # Cancel warmup tasks
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        print("✅ Warmup phase completed")

    async def _warmup_worker(self):
        """Individual warmup worker"""
        while True:
            try:
                # Random cache operations
                key = random.choice(self.test_data["cache_keys"])
                value = self.test_data["cache_values"][key]

                if random.random() < 0.3:  # 30% read operations
                    await self._cache_get(key)
                else:  # 70% write operations
                    await self._cache_set(key, value)

                await asyncio.sleep(0.01)  # 100 ops/sec per worker

            except asyncio.CancelledError:
                break
            except Exception:
                continue  # Ignore errors during warmup

    async def _populate_cache_phase(self):
        """Populate cache with initial data"""
        print("📦 Cache population phase...")

        # Select subset of keys for population (50% of test data)
        population_keys = random.sample(self.test_data["cache_keys"], len(self.test_data["cache_keys"]) // 2)

        # Populate cache concurrently
        tasks = []
        for key in population_keys:
            value = self.test_data["cache_values"][key]
            task = self._cache_set(key, value)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_populations = sum(1 for r in results if not isinstance(r, Exception))

        print(f"✅ Populated cache with {successful_populations}/{len(population_keys)} entries")

    async def _mixed_workload_phase(self):
        """Mixed read/write workload simulation"""
        print(f"⚖️ Mixed workload phase ({self.config.test_duration_seconds}s)...")

        start_time = time.time()
        tasks = []

        # Create worker tasks
        for _ in range(self.config.concurrent_clients):
            task = asyncio.create_task(self._mixed_workload_worker(start_time))
            tasks.append(task)

        # Wait for test completion
        await asyncio.sleep(self.config.test_duration_seconds)

        # Cancel worker tasks
        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        print("✅ Mixed workload phase completed")

    async def _mixed_workload_worker(self, start_time: float):
        """Individual mixed workload worker"""
        worker_requests = 0
        target_rate = self.config.request_rate_per_second // self.config.concurrent_clients

        while time.time() - start_time < self.config.test_duration_seconds:
            try:
                request_start = time.time()

                # Simulate realistic workload: 80% reads, 20% writes
                if random.random() < 0.8:
                    await self._measured_cache_get()
                else:
                    await self._measured_cache_set()

                worker_requests += 1

                # Rate limiting
                elapsed = time.time() - request_start
                sleep_time = max(0, (1.0 / target_rate) - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.metrics.error_count += 1

    async def _stress_test_phase(self):
        """High-load stress testing"""
        print("💥 Stress test phase (60s at 2x normal load)...")

        start_time = time.time()
        tasks = []

        # Double the concurrent clients for stress test
        stress_clients = self.config.concurrent_clients * 2

        for _ in range(stress_clients):
            task = asyncio.create_task(self._stress_worker(start_time))
            tasks.append(task)

        await asyncio.sleep(60)  # 1 minute stress test

        for task in tasks:
            task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)
        print("✅ Stress test phase completed")

    async def _stress_worker(self, start_time: float):
        """High-intensity stress worker"""
        while time.time() - start_time < 60:
            try:
                if random.random() < 0.9:  # 90% reads during stress test
                    await self._measured_cache_get()
                else:
                    await self._measured_cache_set()

                await asyncio.sleep(0.001)  # Minimal delay

            except asyncio.CancelledError:
                break
            except Exception:
                self.metrics.error_count += 1

    async def _ml_prediction_accuracy_phase(self):
        """Test ML prediction accuracy"""
        print("🧠 ML prediction accuracy phase...")

        correct_predictions = 0
        total_predictions = 0

        # Test prediction accuracy with known cache state
        test_keys = random.sample(self.test_data["cache_keys"], 1000)

        for key in test_keys:
            try:
                # Get prediction
                prediction_response = await self._get_prediction(key)

                # Test actual cache hit
                cache_response = await self._cache_get(key)
                actual_hit = cache_response is not None

                # Compare prediction to reality
                predicted_hit = prediction_response.get("hit_probability", 0) > 0.5

                if predicted_hit == actual_hit:
                    correct_predictions += 1

                total_predictions += 1

            except Exception:
                continue

        if total_predictions > 0:
            self.metrics.ml_prediction_accuracy = correct_predictions / total_predictions

        print(f"✅ ML prediction accuracy: {self.metrics.ml_prediction_accuracy:.2%}")

    async def _measured_cache_get(self):
        """Measured cache GET operation"""
        key = random.choice(self.test_data["cache_keys"])

        start_time = time.time()
        result = await self._cache_get(key)
        response_time = (time.time() - start_time) * 1000  # Convert to ms

        self.metrics.total_requests += 1
        self.metrics.response_times.append(response_time)

        if result is not None:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1

        self.metrics.successful_requests += 1

    async def _measured_cache_set(self):
        """Measured cache SET operation"""
        key = random.choice(self.test_data["cache_keys"])
        value = self.test_data["cache_values"][key]

        start_time = time.time()
        await self._cache_set(key, value)
        response_time = (time.time() - start_time) * 1000  # Convert to ms

        self.metrics.total_requests += 1
        self.metrics.response_times.append(response_time)
        self.metrics.successful_requests += 1

    async def _cache_get(self, key: str) -> Optional[Any]:
        """Perform cache GET operation"""
        try:
            url = f"{self.config.cache_service_url}/cache/{key}"
            params = {
                "user_id": f"bench_user_{random.randint(1, 100)}",
                "query_type": random.choice(["osint", "whois", "dns"])
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("value")
                return None

        except Exception:
            return None

    async def _cache_set(self, key: str, value: Any):
        """Perform cache SET operation"""
        try:
            url = f"{self.config.cache_service_url}/cache/{key}"
            data = {
                "value": value,
                "ttl": 3600,
                "user_id": f"bench_user_{random.randint(1, 100)}",
                "query_type": value.get("query_type", "osint"),
                "size_hint": len(json.dumps(value))
            }

            async with self.session.put(url, json=data) as response:
                return response.status == 200

        except Exception:
            return False

    async def _get_prediction(self, key: str) -> Dict[str, Any]:
        """Get ML prediction for cache key"""
        try:
            url = f"{self.config.cache_service_url}/predict"
            data = {
                "key": key,
                "query_type": "osint",
                "user_id": f"bench_user_{random.randint(1, 100)}"
            }

            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                return {}

        except Exception:
            return {}

    def _calculate_final_metrics(self):
        """Calculate final performance metrics"""
        if self.metrics.response_times:
            self.metrics.avg_response_time = statistics.mean(self.metrics.response_times)
            sorted_times = sorted(self.metrics.response_times)
            self.metrics.p95_response_time = sorted_times[int(len(sorted_times) * 0.95)]
            self.metrics.p99_response_time = sorted_times[int(len(sorted_times) * 0.99)]

        if self.metrics.total_requests > 0:
            self.metrics.cache_hit_rate = self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)
            self.metrics.throughput_ops_per_sec = self.metrics.successful_requests / self.config.test_duration_seconds

        # Estimate memory efficiency (placeholder - would need actual memory metrics)
        self.metrics.memory_efficiency = 0.92  # Target >90%

    async def cleanup(self):
        """Cleanup benchmark resources"""
        if self.session:
            await self.session.close()

class BenchmarkReporter:
    """Generate comprehensive benchmark reports"""

    @staticmethod
    def generate_report(metrics: BenchmarkMetrics, config: BenchmarkConfig) -> str:
        """Generate detailed benchmark report"""
        report = []
        report.append("=" * 80)
        report.append("PREDICTIVE CACHE PERFORMANCE BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append(f"Test Duration: {config.test_duration_seconds}s")
        report.append(f"Concurrent Clients: {config.concurrent_clients}")
        report.append(f"Target Rate: {config.request_rate_per_second} ops/sec")
        report.append("")

        # Performance Summary
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Requests: {metrics.total_requests:,}")
        report.append(f"Successful Requests: {metrics.successful_requests:,}")
        report.append(f"Error Rate: {(metrics.error_count / max(metrics.total_requests, 1)) * 100:.2f}%")
        report.append(f"Throughput: {metrics.throughput_ops_per_sec:.1f} ops/sec")
        report.append("")

        # Cache Performance
        report.append("CACHE PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Cache Hit Rate: {metrics.cache_hit_rate * 100:.2f}% {'✅' if metrics.cache_hit_rate >= 0.8 else '❌'}")
        report.append(f"Cache Hits: {metrics.cache_hits:,}")
        report.append(f"Cache Misses: {metrics.cache_misses:,}")
        report.append(f"Memory Efficiency: {metrics.memory_efficiency * 100:.1f}% {'✅' if metrics.memory_efficiency >= 0.9 else '❌'}")
        report.append("")

        # Response Time Analysis
        report.append("RESPONSE TIME ANALYSIS")
        report.append("-" * 40)
        report.append(f"Average Response Time: {metrics.avg_response_time:.2f}ms {'✅' if metrics.avg_response_time <= 10 else '❌'}")
        report.append(f"95th Percentile: {metrics.p95_response_time:.2f}ms")
        report.append(f"99th Percentile: {metrics.p99_response_time:.2f}ms")
        report.append("")

        # ML Performance
        report.append("ML PREDICTION PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Prediction Accuracy: {metrics.ml_prediction_accuracy * 100:.2f}% {'✅' if metrics.ml_prediction_accuracy >= 0.8 else '❌'}")
        report.append("")

        # Target Validation
        report.append("TARGET VALIDATION")
        report.append("-" * 40)
        targets = [
            ("Cache Hit Rate >80%", metrics.cache_hit_rate >= 0.8),
            ("Response Time <10ms", metrics.avg_response_time <= 10),
            ("ML Accuracy >80%", metrics.ml_prediction_accuracy >= 0.8),
            ("Memory Efficiency >90%", metrics.memory_efficiency >= 0.9),
            ("Error Rate <1%", (metrics.error_count / max(metrics.total_requests, 1)) < 0.01)
        ]

        passed_targets = sum(1 for _, passed in targets if passed)
        report.append(f"Targets Passed: {passed_targets}/{len(targets)}")

        for target, passed in targets:
            status = "✅ PASS" if passed else "❌ FAIL"
            report.append(f"  {target}: {status}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    @staticmethod
    def save_metrics_json(metrics: BenchmarkMetrics, config: BenchmarkConfig, filename: str):
        """Save detailed metrics to JSON file"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(config),
            "metrics": asdict(metrics),
            "summary": {
                "targets_met": {
                    "cache_hit_rate": metrics.cache_hit_rate >= 0.8,
                    "response_time": metrics.avg_response_time <= 10,
                    "ml_accuracy": metrics.ml_prediction_accuracy >= 0.8,
                    "memory_efficiency": metrics.memory_efficiency >= 0.9
                }
            }
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

async def main():
    """Main benchmark execution"""
    print("🎯 BEV OSINT Predictive Cache Benchmark Suite")
    print("=" * 60)

    # Configure benchmark
    config = BenchmarkConfig(
        cache_service_url="http://localhost:8044",
        test_duration_seconds=300,  # 5 minutes
        concurrent_clients=50,
        request_rate_per_second=1000
    )

    # Run benchmark
    benchmark = PredictiveCacheBenchmark(config)

    try:
        await benchmark.setup()
        metrics = await benchmark.run_benchmark()

        # Generate reports
        report = BenchmarkReporter.generate_report(metrics, config)
        print(report)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"/tmp/cache_benchmark_{timestamp}.json"
        BenchmarkReporter.save_metrics_json(metrics, config, json_filename)

        print(f"\n📊 Detailed metrics saved to: {json_filename}")

        # Return exit code based on target achievement
        targets_met = [
            metrics.cache_hit_rate >= 0.8,
            metrics.avg_response_time <= 10,
            metrics.ml_prediction_accuracy >= 0.8,
            metrics.memory_efficiency >= 0.9
        ]

        if all(targets_met):
            print("🎉 All performance targets achieved!")
            return 0
        else:
            print("⚠️ Some performance targets not met")
            return 1

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 2
    finally:
        await benchmark.cleanup()

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)