#!/usr/bin/env python3
"""
BEV OSINT Framework - Predictive Cache Load Testing Script

Comprehensive load testing suite for the predictive cache system to validate
performance under various load conditions and identify bottlenecks.

Test Scenarios:
- Baseline performance testing
- Stress testing with high concurrency
- Spike testing with sudden load increases
- Endurance testing for stability
- Cache warming efficiency testing
- ML prediction accuracy under load
"""

import asyncio
import json
import time
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

class LoadTestResult(NamedTuple):
    """Single load test operation result"""
    operation: str
    latency_ms: float
    success: bool
    error_type: Optional[str] = None
    response_size: int = 0
    cache_hit: bool = False

@dataclass
class LoadTestConfig:
    """Load test configuration"""
    cache_service_url: str = "http://localhost:8044"
    test_duration_seconds: int = 300
    warmup_duration_seconds: int = 60
    concurrent_users: int = 100
    operations_per_second: int = 1000
    read_write_ratio: float = 0.8  # 80% reads, 20% writes
    data_size_kb_min: int = 1
    data_size_kb_max: int = 100
    key_space_size: int = 10000
    hot_key_percentage: float = 0.2  # 20% of operations target hot keys

@dataclass
class LoadTestMetrics:
    """Aggregated load test metrics"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0
    avg_response_size_kb: float = 0.0

class PredictiveCacheLoadTester:
    """Load testing framework for predictive cache"""

    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.test_data: Dict[str, Any] = {}
        self.results: List[LoadTestResult] = []
        self.running = True
        self.start_time: float = 0.0

    async def setup(self):
        """Initialize load testing environment"""
        print("🔧 Setting up load test environment...")

        # Create HTTP session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.config.concurrent_users * 2,
            limit_per_host=self.config.concurrent_users * 2,
            keepalive_timeout=30
        )
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

        # Generate test data
        await self._generate_test_data()

        # Verify cache service availability
        await self._verify_cache_service()

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("✅ Load test environment ready")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.running = False
        print("\n🛑 Shutdown signal received, stopping load test...")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

    async def _generate_test_data(self):
        """Generate realistic test data for load testing"""
        print("📊 Generating test data...")

        # Generate cache keys with realistic OSINT patterns
        key_patterns = [
            "osint:query:{user_id}:{hash}",
            "whois:domain:{domain}",
            "dns:record:{type}:{domain}",
            "geo:ip:{ip_address}",
            "social:profile:{platform}:{handle}",
            "threat:ioc:{type}:{value}",
            "file:analysis:{hash}",
            "network:scan:{target}:{port}"
        ]

        self.test_data["keys"] = []
        for i in range(self.config.key_space_size):
            pattern = random.choice(key_patterns)
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
            self.test_data["keys"].append(key)

        # Identify hot keys (most frequently accessed)
        hot_key_count = int(len(self.test_data["keys"]) * self.config.hot_key_percentage)
        self.test_data["hot_keys"] = self.test_data["keys"][:hot_key_count]
        self.test_data["cold_keys"] = self.test_data["keys"][hot_key_count:]

        # Generate test values with varying sizes
        self.test_data["values"] = {}
        for key in self.test_data["keys"]:
            size_kb = random.randint(self.config.data_size_kb_min, self.config.data_size_kb_max)
            value = {
                "query_type": random.choice(["osint", "whois", "dns", "geo", "social", "threat"]),
                "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 1440))).isoformat(),
                "data": "x" * (size_kb * 1024),  # Payload data
                "metadata": {
                    "source": random.choice(["api", "scraper", "database"]),
                    "confidence": random.uniform(0.5, 1.0),
                    "ttl": random.randint(300, 3600)
                },
                "size_kb": size_kb
            }
            self.test_data["values"][key] = value

        print(f"Generated {len(self.test_data['keys'])} test keys")
        print(f"Hot keys: {len(self.test_data['hot_keys'])}, Cold keys: {len(self.test_data['cold_keys'])}")

    async def _verify_cache_service(self):
        """Verify cache service is available and responsive"""
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

    async def run_load_test_suite(self) -> Dict[str, LoadTestMetrics]:
        """Execute comprehensive load test suite"""
        print("🚀 Starting predictive cache load test suite...")

        test_results = {}

        try:
            # 1. Baseline Performance Test
            print("\n" + "="*60)
            print("1. BASELINE PERFORMANCE TEST")
            print("="*60)
            baseline_config = LoadTestConfig(
                cache_service_url=self.config.cache_service_url,
                test_duration_seconds=120,
                concurrent_users=10,
                operations_per_second=100
            )
            test_results["baseline"] = await self._run_single_test(baseline_config, "Baseline")

            # 2. Normal Load Test
            print("\n" + "="*60)
            print("2. NORMAL LOAD TEST")
            print("="*60)
            test_results["normal_load"] = await self._run_single_test(self.config, "Normal Load")

            # 3. High Concurrency Test
            print("\n" + "="*60)
            print("3. HIGH CONCURRENCY TEST")
            print("="*60)
            high_concurrency_config = LoadTestConfig(
                cache_service_url=self.config.cache_service_url,
                test_duration_seconds=180,
                concurrent_users=200,
                operations_per_second=self.config.operations_per_second
            )
            test_results["high_concurrency"] = await self._run_single_test(high_concurrency_config, "High Concurrency")

            # 4. Stress Test
            print("\n" + "="*60)
            print("4. STRESS TEST")
            print("="*60)
            stress_config = LoadTestConfig(
                cache_service_url=self.config.cache_service_url,
                test_duration_seconds=300,
                concurrent_users=500,
                operations_per_second=2000
            )
            test_results["stress"] = await self._run_single_test(stress_config, "Stress")

            # 5. Spike Test
            print("\n" + "="*60)
            print("5. SPIKE TEST")
            print("="*60)
            test_results["spike"] = await self._run_spike_test()

            # 6. Cache Warming Test
            print("\n" + "="*60)
            print("6. CACHE WARMING EFFICIENCY TEST")
            print("="*60)
            test_results["cache_warming"] = await self._run_cache_warming_test()

            return test_results

        except Exception as e:
            print(f"❌ Load test suite failed: {e}")
            raise

    async def _run_single_test(self, config: LoadTestConfig, test_name: str) -> LoadTestMetrics:
        """Run a single load test scenario"""
        print(f"🎯 Running {test_name} test...")
        print(f"Duration: {config.test_duration_seconds}s, Users: {config.concurrent_users}, Target OPS: {config.operations_per_second}")

        # Clear previous results
        self.results.clear()
        self.start_time = time.time()

        # Warmup phase
        await self._warmup_phase(config)

        # Main test phase
        await self._execute_load_test(config)

        # Calculate metrics
        metrics = self._calculate_metrics()

        print(f"✅ {test_name} test completed")
        print(f"   Throughput: {metrics.throughput_ops_per_sec:.1f} ops/sec")
        print(f"   Success Rate: {((metrics.successful_operations / max(metrics.total_operations, 1)) * 100):.2f}%")
        print(f"   Cache Hit Rate: {metrics.cache_hit_rate:.2%}")
        print(f"   P95 Latency: {metrics.p95_latency_ms:.2f}ms")

        return metrics

    async def _warmup_phase(self, config: LoadTestConfig):
        """Warmup phase to prepare system"""
        if config.warmup_duration_seconds <= 0:
            return

        print(f"🔥 Warmup phase ({config.warmup_duration_seconds}s)...")

        warmup_tasks = []
        for _ in range(min(config.concurrent_users, 20)):  # Limit warmup concurrency
            task = asyncio.create_task(self._warmup_worker())
            warmup_tasks.append(task)

        await asyncio.sleep(config.warmup_duration_seconds)

        for task in warmup_tasks:
            task.cancel()

        await asyncio.gather(*warmup_tasks, return_exceptions=True)
        print("✅ Warmup completed")

    async def _warmup_worker(self):
        """Warmup worker to prepare cache"""
        while True:
            try:
                # Populate cache with some data
                key = random.choice(self.test_data["keys"])
                value = self.test_data["values"][key]

                if random.random() < 0.3:  # 30% reads
                    await self._cache_get(key)
                else:  # 70% writes during warmup
                    await self._cache_set(key, value)

                await asyncio.sleep(0.05)  # 20 ops/sec per worker

            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _execute_load_test(self, config: LoadTestConfig):
        """Execute main load test with specified configuration"""
        print(f"⚡ Main test phase ({config.test_duration_seconds}s)...")

        # Create worker tasks
        workers = []
        ops_per_worker = config.operations_per_second // config.concurrent_users

        for worker_id in range(config.concurrent_users):
            worker = asyncio.create_task(
                self._load_test_worker(worker_id, ops_per_worker, config)
            )
            workers.append(worker)

        # Run test for specified duration
        await asyncio.sleep(config.test_duration_seconds)

        # Stop all workers
        for worker in workers:
            worker.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

    async def _load_test_worker(self, worker_id: int, ops_per_second: int, config: LoadTestConfig):
        """Individual load test worker"""
        worker_operations = 0
        last_rate_check = time.time()
        target_interval = 1.0 / max(ops_per_second, 1)

        while self.running:
            try:
                operation_start = time.time()

                # Choose operation type based on read/write ratio
                if random.random() < config.read_write_ratio:
                    result = await self._measured_cache_get()
                else:
                    result = await self._measured_cache_set()

                self.results.append(result)
                worker_operations += 1

                # Rate limiting
                operation_duration = time.time() - operation_start
                sleep_time = max(0, target_interval - operation_duration)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                # Adjust rate if falling behind
                if worker_operations % 100 == 0:
                    elapsed = time.time() - last_rate_check
                    if elapsed > 0:
                        actual_rate = 100 / elapsed
                        if actual_rate < ops_per_second * 0.8:  # If 20% below target
                            target_interval *= 0.9  # Increase rate slightly
                        last_rate_check = time.time()

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Record error
                error_result = LoadTestResult(
                    operation="error",
                    latency_ms=0.0,
                    success=False,
                    error_type=str(type(e).__name__)
                )
                self.results.append(error_result)

    async def _measured_cache_get(self) -> LoadTestResult:
        """Perform measured cache GET operation"""
        # Select key (favor hot keys)
        if random.random() < 0.8:  # 80% chance of hot key
            key = random.choice(self.test_data["hot_keys"])
        else:
            key = random.choice(self.test_data["cold_keys"])

        start_time = time.time()
        try:
            response_data = await self._cache_get(key)
            latency_ms = (time.time() - start_time) * 1000

            cache_hit = response_data is not None
            response_size = len(json.dumps(response_data)) if response_data else 0

            return LoadTestResult(
                operation="get",
                latency_ms=latency_ms,
                success=True,
                response_size=response_size,
                cache_hit=cache_hit
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LoadTestResult(
                operation="get",
                latency_ms=latency_ms,
                success=False,
                error_type=str(type(e).__name__)
            )

    async def _measured_cache_set(self) -> LoadTestResult:
        """Perform measured cache SET operation"""
        key = random.choice(self.test_data["keys"])
        value = self.test_data["values"][key]

        start_time = time.time()
        try:
            success = await self._cache_set(key, value)
            latency_ms = (time.time() - start_time) * 1000

            response_size = len(json.dumps(value))

            return LoadTestResult(
                operation="set",
                latency_ms=latency_ms,
                success=success,
                response_size=response_size
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LoadTestResult(
                operation="set",
                latency_ms=latency_ms,
                success=False,
                error_type=str(type(e).__name__)
            )

    async def _cache_get(self, key: str) -> Optional[Any]:
        """Perform cache GET operation"""
        url = f"{self.config.cache_service_url}/cache/{key}"
        params = {
            "user_id": f"loadtest_user_{random.randint(1, 100)}",
            "query_type": random.choice(["osint", "whois", "dns"])
        }

        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("value")
            return None

    async def _cache_set(self, key: str, value: Any) -> bool:
        """Perform cache SET operation"""
        url = f"{self.config.cache_service_url}/cache/{key}"
        data = {
            "value": value,
            "ttl": 3600,
            "user_id": f"loadtest_user_{random.randint(1, 100)}",
            "query_type": value.get("query_type", "osint"),
            "size_hint": len(json.dumps(value))
        }

        async with self.session.put(url, json=data) as response:
            return response.status == 200

    async def _run_spike_test(self) -> LoadTestMetrics:
        """Run spike test with sudden load increases"""
        print("⚡ Spike test: 30s normal → 60s spike → 30s normal...")

        self.results.clear()
        self.start_time = time.time()

        # Phase 1: Normal load (30s)
        normal_config = LoadTestConfig(
            cache_service_url=self.config.cache_service_url,
            test_duration_seconds=30,
            concurrent_users=50,
            operations_per_second=500
        )
        await self._execute_load_test(normal_config)

        # Phase 2: Spike load (60s)
        spike_config = LoadTestConfig(
            cache_service_url=self.config.cache_service_url,
            test_duration_seconds=60,
            concurrent_users=300,
            operations_per_second=3000
        )
        await self._execute_load_test(spike_config)

        # Phase 3: Return to normal (30s)
        await self._execute_load_test(normal_config)

        metrics = self._calculate_metrics()
        print("✅ Spike test completed")
        return metrics

    async def _run_cache_warming_test(self) -> LoadTestMetrics:
        """Test cache warming efficiency"""
        print("🔥 Testing cache warming efficiency...")

        self.results.clear()
        self.start_time = time.time()

        # Step 1: Clear cache and measure cold performance
        await self._clear_cache()
        await asyncio.sleep(2)

        # Cold performance baseline
        cold_start = time.time()
        cold_results = []
        for _ in range(100):
            key = random.choice(self.test_data["hot_keys"])
            result = await self._measured_cache_get()
            cold_results.append(result)

        cold_hit_rate = sum(1 for r in cold_results if r.cache_hit) / len(cold_results)
        print(f"Cold cache hit rate: {cold_hit_rate:.2%}")

        # Step 2: Trigger cache warming
        await self._trigger_cache_warming()
        await asyncio.sleep(30)  # Wait for warming to complete

        # Step 3: Measure warmed performance
        warm_results = []
        for _ in range(100):
            key = random.choice(self.test_data["hot_keys"])
            result = await self._measured_cache_get()
            warm_results.append(result)

        warm_hit_rate = sum(1 for r in warm_results if r.cache_hit) / len(warm_results)
        print(f"Warmed cache hit rate: {warm_hit_rate:.2%}")

        # Calculate warming efficiency
        self.results.extend(cold_results + warm_results)
        metrics = self._calculate_metrics()

        warming_efficiency = (warm_hit_rate - cold_hit_rate) / (1.0 - cold_hit_rate) if cold_hit_rate < 1.0 else 1.0
        print(f"Cache warming efficiency: {warming_efficiency:.2%}")

        return metrics

    async def _clear_cache(self):
        """Clear cache for warming test"""
        try:
            async with self.session.post(f"{self.config.cache_service_url}/admin/clear") as response:
                if response.status != 200:
                    print("⚠️ Failed to clear cache")
        except Exception as e:
            print(f"⚠️ Error clearing cache: {e}")

    async def _trigger_cache_warming(self):
        """Trigger cache warming"""
        try:
            warming_data = {
                "strategy": "popularity_based",
                "intensity": "high"
            }
            async with self.session.post(f"{self.config.cache_service_url}/warm", json=warming_data) as response:
                if response.status == 200:
                    print("✅ Cache warming triggered")
                else:
                    print("⚠️ Failed to trigger cache warming")
        except Exception as e:
            print(f"⚠️ Error triggering cache warming: {e}")

    def _calculate_metrics(self) -> LoadTestMetrics:
        """Calculate aggregated metrics from test results"""
        if not self.results:
            return LoadTestMetrics()

        # Filter valid results
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]

        # Calculate latency statistics
        latencies = [r.latency_ms for r in successful_results if r.latency_ms > 0]
        cache_hits = [r for r in successful_results if r.cache_hit]
        cache_misses = [r for r in successful_results if not r.cache_hit]

        metrics = LoadTestMetrics()
        metrics.total_operations = len(self.results)
        metrics.successful_operations = len(successful_results)
        metrics.failed_operations = len(failed_results)
        metrics.cache_hits = len(cache_hits)
        metrics.cache_misses = len(cache_misses)

        if latencies:
            metrics.total_latency_ms = sum(latencies)
            metrics.min_latency_ms = min(latencies)
            metrics.max_latency_ms = max(latencies)

            sorted_latencies = sorted(latencies)
            metrics.p50_latency_ms = sorted_latencies[len(sorted_latencies) // 2]
            metrics.p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            metrics.p99_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        # Calculate rates
        test_duration = time.time() - self.start_time if self.start_time > 0 else 1
        metrics.throughput_ops_per_sec = metrics.successful_operations / test_duration
        metrics.error_rate = metrics.failed_operations / max(metrics.total_operations, 1)
        metrics.cache_hit_rate = metrics.cache_hits / max(metrics.cache_hits + metrics.cache_misses, 1)

        # Calculate average response size
        response_sizes = [r.response_size for r in successful_results if r.response_size > 0]
        if response_sizes:
            metrics.avg_response_size_kb = statistics.mean(response_sizes) / 1024

        return metrics

def generate_load_test_report(test_results: Dict[str, LoadTestMetrics]) -> str:
    """Generate comprehensive load test report"""
    report = []
    report.append("=" * 80)
    report.append("PREDICTIVE CACHE LOAD TEST REPORT")
    report.append("=" * 80)
    report.append(f"Report Generated: {datetime.now().isoformat()}")
    report.append("")

    for test_name, metrics in test_results.items():
        report.append(f"{test_name.upper().replace('_', ' ')} TEST RESULTS")
        report.append("-" * 40)
        report.append(f"Total Operations: {metrics.total_operations:,}")
        report.append(f"Successful Operations: {metrics.successful_operations:,}")
        report.append(f"Failed Operations: {metrics.failed_operations:,}")
        report.append(f"Success Rate: {((metrics.successful_operations / max(metrics.total_operations, 1)) * 100):.2f}%")
        report.append(f"Error Rate: {(metrics.error_rate * 100):.2f}%")
        report.append("")

        report.append(f"Cache Performance:")
        report.append(f"  Hit Rate: {(metrics.cache_hit_rate * 100):.2f}%")
        report.append(f"  Cache Hits: {metrics.cache_hits:,}")
        report.append(f"  Cache Misses: {metrics.cache_misses:,}")
        report.append("")

        report.append(f"Performance Metrics:")
        report.append(f"  Throughput: {metrics.throughput_ops_per_sec:.1f} ops/sec")
        report.append(f"  Avg Response Size: {metrics.avg_response_size_kb:.2f} KB")
        report.append("")

        report.append(f"Latency Distribution:")
        report.append(f"  Min: {metrics.min_latency_ms:.2f}ms")
        report.append(f"  P50: {metrics.p50_latency_ms:.2f}ms")
        report.append(f"  P95: {metrics.p95_latency_ms:.2f}ms")
        report.append(f"  P99: {metrics.p99_latency_ms:.2f}ms")
        report.append(f"  Max: {metrics.max_latency_ms:.2f}ms")
        report.append("")

        # Target validation
        targets_met = []
        if test_name == "baseline":
            targets_met = [
                ("Cache Hit Rate >60%", metrics.cache_hit_rate >= 0.6),
                ("P95 Latency <20ms", metrics.p95_latency_ms <= 20),
                ("Error Rate <1%", metrics.error_rate <= 0.01)
            ]
        elif test_name in ["normal_load", "high_concurrency"]:
            targets_met = [
                ("Cache Hit Rate >75%", metrics.cache_hit_rate >= 0.75),
                ("P95 Latency <50ms", metrics.p95_latency_ms <= 50),
                ("Error Rate <2%", metrics.error_rate <= 0.02)
            ]
        elif test_name == "stress":
            targets_met = [
                ("Cache Hit Rate >60%", metrics.cache_hit_rate >= 0.6),
                ("P95 Latency <100ms", metrics.p95_latency_ms <= 100),
                ("Error Rate <5%", metrics.error_rate <= 0.05)
            ]

        if targets_met:
            passed = sum(1 for _, met in targets_met if met)
            report.append(f"Target Validation: {passed}/{len(targets_met)} passed")
            for target, met in targets_met:
                status = "✅ PASS" if met else "❌ FAIL"
                report.append(f"  {target}: {status}")
            report.append("")

    report.append("=" * 80)
    return "\n".join(report)

async def main():
    """Main load testing script"""
    parser = argparse.ArgumentParser(description="Predictive Cache Load Testing")
    parser.add_argument("--cache-url", default="http://localhost:8044",
                       help="Cache service URL")
    parser.add_argument("--duration", type=int, default=300,
                       help="Test duration in seconds")
    parser.add_argument("--users", type=int, default=100,
                       help="Concurrent users")
    parser.add_argument("--ops", type=int, default=1000,
                       help="Operations per second")
    parser.add_argument("--test", choices=["baseline", "normal", "stress", "spike", "warming", "all"],
                       default="all", help="Test type to run")

    args = parser.parse_args()

    config = LoadTestConfig(
        cache_service_url=args.cache_url,
        test_duration_seconds=args.duration,
        concurrent_users=args.users,
        operations_per_second=args.ops
    )

    tester = PredictiveCacheLoadTester(config)

    try:
        await tester.setup()

        if args.test == "all":
            results = await tester.run_load_test_suite()
        else:
            # Run single test
            if args.test == "baseline":
                baseline_config = LoadTestConfig(
                    cache_service_url=config.cache_service_url,
                    test_duration_seconds=120,
                    concurrent_users=10,
                    operations_per_second=100
                )
                results = {"baseline": await tester._run_single_test(baseline_config, "Baseline")}
            elif args.test == "normal":
                results = {"normal_load": await tester._run_single_test(config, "Normal Load")}
            elif args.test == "stress":
                stress_config = LoadTestConfig(
                    cache_service_url=config.cache_service_url,
                    test_duration_seconds=300,
                    concurrent_users=500,
                    operations_per_second=2000
                )
                results = {"stress": await tester._run_single_test(stress_config, "Stress")}
            elif args.test == "spike":
                results = {"spike": await tester._run_spike_test()}
            elif args.test == "warming":
                results = {"cache_warming": await tester._run_cache_warming_test()}

        # Generate and display report
        report = generate_load_test_report(results)
        print("\n" + report)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"/tmp/cache_load_test_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": asdict(config),
                "results": {k: asdict(v) for k, v in results.items()}
            }, f, indent=2)

        print(f"\n📊 Detailed results saved to: {results_file}")

        # Determine exit code based on critical targets
        critical_failures = 0
        for test_name, metrics in results.items():
            if metrics.error_rate > 0.1:  # >10% error rate
                critical_failures += 1
            if test_name != "stress" and metrics.cache_hit_rate < 0.5:  # <50% hit rate (except stress test)
                critical_failures += 1

        if critical_failures == 0:
            print("🎉 All load tests passed!")
            return 0
        else:
            print(f"⚠️ {critical_failures} critical performance issues detected")
            return 1

    except Exception as e:
        print(f"❌ Load test failed: {e}")
        return 2
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)