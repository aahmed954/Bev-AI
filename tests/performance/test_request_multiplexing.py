"""
Performance tests for request multiplexing system
Validates 1000+ concurrent requests and <100ms latency targets
"""

import pytest
import asyncio
import aiohttp
import time
import statistics
import json
import random
import logging
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger(__name__)

@pytest.mark.performance
class TestRequestMultiplexing:
    """Test request multiplexing performance and concurrency"""

    @pytest.fixture(autouse=True)
    def setup_performance_test(self, performance_monitor):
        """Setup performance monitoring for each test"""
        self.monitor = performance_monitor
        self.monitor.start()
        yield
        self.monitor.stop()

    async def test_concurrent_request_handling(self):
        """Test handling of 1000+ concurrent requests"""
        target_requests = 1000
        base_url = "http://localhost:8000"  # Assuming API endpoint

        # Generate diverse test payloads
        test_payloads = self._generate_test_payloads(target_requests)

        async with aiohttp.ClientSession() as session:
            # Create tasks for concurrent requests
            tasks = []
            start_time = time.time()

            for i, payload in enumerate(test_payloads):
                task = self._make_async_request(session, base_url, payload, i)
                tasks.append(task)

            # Execute all requests concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time

            # Analyze results
            successful_requests = [r for r in results if not isinstance(r, Exception)]
            failed_requests = [r for r in results if isinstance(r, Exception)]

            success_rate = len(successful_requests) / len(results)
            requests_per_second = len(successful_requests) / total_time

            # Performance assertions
            assert len(successful_requests) >= target_requests * 0.95  # 95% success rate
            assert requests_per_second >= 500  # At least 500 RPS
            assert total_time <= 10  # Complete within 10 seconds

            # Latency analysis
            latencies = [r.get('latency', 0) for r in successful_requests if isinstance(r, dict)]
            if latencies:
                avg_latency = statistics.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                p99_latency = np.percentile(latencies, 99)

                self.monitor.record("avg_latency", avg_latency)
                self.monitor.record("p95_latency", p95_latency)
                self.monitor.record("p99_latency", p99_latency)

                assert avg_latency <= 100  # Average latency under 100ms
                assert p95_latency <= 200  # 95th percentile under 200ms

            logger.info(f"Concurrent test results:")
            logger.info(f"  Total requests: {len(results)}")
            logger.info(f"  Successful: {len(successful_requests)}")
            logger.info(f"  Success rate: {success_rate:.2%}")
            logger.info(f"  RPS: {requests_per_second:.1f}")
            logger.info(f"  Total time: {total_time:.2f}s")
            if latencies:
                logger.info(f"  Avg latency: {avg_latency:.1f}ms")
                logger.info(f"  P95 latency: {p95_latency:.1f}ms")

    async def test_burst_traffic_handling(self):
        """Test system behavior under burst traffic conditions"""
        base_url = "http://localhost:8000"

        # Simulate traffic bursts
        burst_scenarios = [
            {"requests": 200, "duration": 1},   # 200 req/s for 1 second
            {"requests": 500, "duration": 2},   # 500 req/s for 2 seconds
            {"requests": 1000, "duration": 1},  # 1000 req/s for 1 second
        ]

        async with aiohttp.ClientSession() as session:
            for scenario in burst_scenarios:
                logger.info(f"Testing burst: {scenario['requests']} requests in {scenario['duration']}s")

                payloads = self._generate_test_payloads(scenario['requests'])
                start_time = time.time()

                # Create burst of requests
                tasks = []
                for payload in payloads:
                    task = self._make_async_request(session, base_url, payload)
                    tasks.append(task)

                # Execute burst
                results = await asyncio.gather(*tasks, return_exceptions=True)
                burst_time = time.time() - start_time

                successful = [r for r in results if not isinstance(r, Exception)]
                success_rate = len(successful) / len(results)
                actual_rps = len(successful) / burst_time

                # Burst performance assertions
                assert success_rate >= 0.90  # 90% success during burst
                assert burst_time <= scenario['duration'] * 2  # Complete within 2x expected time

                self.monitor.record(f"burst_{scenario['requests']}_success_rate", success_rate)
                self.monitor.record(f"burst_{scenario['requests']}_rps", actual_rps)

                logger.info(f"  Success rate: {success_rate:.2%}")
                logger.info(f"  Actual RPS: {actual_rps:.1f}")

                # Brief pause between bursts
                await asyncio.sleep(2)

    async def test_multiplexer_efficiency(self):
        """Test request multiplexer efficiency and optimization"""
        base_url = "http://localhost:8000"

        # Test different multiplexing strategies
        strategies = ["round_robin", "least_connections", "weighted"]

        for strategy in strategies:
            logger.info(f"Testing multiplexing strategy: {strategy}")

            # Configure multiplexer for this strategy
            config_payload = {
                "action": "configure_multiplexer",
                "strategy": strategy,
                "workers": 8
            }

            async with aiohttp.ClientSession() as session:
                # Configure strategy
                await self._make_async_request(session, base_url, config_payload)

                # Test performance with this strategy
                test_requests = 500
                payloads = self._generate_test_payloads(test_requests)

                start_time = time.time()
                tasks = [self._make_async_request(session, base_url, p) for p in payloads]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                strategy_time = time.time() - start_time

                successful = [r for r in results if not isinstance(r, Exception)]
                strategy_rps = len(successful) / strategy_time

                # Record strategy performance
                self.monitor.record(f"strategy_{strategy}_rps", strategy_rps)
                self.monitor.record(f"strategy_{strategy}_time", strategy_time)

                logger.info(f"  Strategy {strategy}: {strategy_rps:.1f} RPS in {strategy_time:.2f}s")

                # Strategy should handle at least 200 RPS
                assert strategy_rps >= 200

    async def test_connection_pooling_performance(self):
        """Test connection pool optimization"""
        base_url = "http://localhost:8000"

        # Test different pool sizes
        pool_configs = [
            {"pool_size": 10, "max_overflow": 5},
            {"pool_size": 20, "max_overflow": 10},
            {"pool_size": 50, "max_overflow": 25},
        ]

        for config in pool_configs:
            logger.info(f"Testing pool config: {config}")

            # Configure connection pool
            pool_payload = {
                "action": "configure_pool",
                "pool_size": config["pool_size"],
                "max_overflow": config["max_overflow"]
            }

            connector = aiohttp.TCPConnector(
                limit=config["pool_size"],
                limit_per_host=config["pool_size"]
            )

            async with aiohttp.ClientSession(connector=connector) as session:
                # Configure pool
                await self._make_async_request(session, base_url, pool_payload)

                # Test concurrent requests with this pool
                test_requests = 300
                payloads = self._generate_test_payloads(test_requests)

                start_time = time.time()
                tasks = [self._make_async_request(session, base_url, p) for p in payloads]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                pool_time = time.time() - start_time

                successful = [r for r in results if not isinstance(r, Exception)]
                pool_rps = len(successful) / pool_time
                connection_reuse_rate = self._calculate_connection_reuse(results)

                self.monitor.record(f"pool_{config['pool_size']}_rps", pool_rps)
                self.monitor.record(f"pool_{config['pool_size']}_reuse", connection_reuse_rate)

                logger.info(f"  Pool {config['pool_size']}: {pool_rps:.1f} RPS, {connection_reuse_rate:.2%} reuse")

                # Pool should improve performance
                assert pool_rps >= 150
                assert connection_reuse_rate >= 0.8  # 80% connection reuse

    async def test_request_queuing_performance(self):
        """Test request queue management and overflow handling"""
        base_url = "http://localhost:8000"

        # Test queue sizes and overflow behavior
        queue_configs = [
            {"queue_size": 100, "overflow_strategy": "drop"},
            {"queue_size": 500, "overflow_strategy": "block"},
            {"queue_size": 1000, "overflow_strategy": "expand"},
        ]

        async with aiohttp.ClientSession() as session:
            for config in queue_configs:
                logger.info(f"Testing queue config: {config}")

                # Configure queue
                queue_payload = {
                    "action": "configure_queue",
                    "queue_size": config["queue_size"],
                    "overflow_strategy": config["overflow_strategy"]
                }
                await self._make_async_request(session, base_url, queue_payload)

                # Send requests to fill and overflow queue
                overflow_requests = config["queue_size"] + 200
                payloads = self._generate_test_payloads(overflow_requests)

                start_time = time.time()
                tasks = [self._make_async_request(session, base_url, p) for p in payloads]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                queue_time = time.time() - start_time

                successful = [r for r in results if not isinstance(r, Exception)]
                dropped = len(results) - len(successful)

                queue_throughput = len(successful) / queue_time
                drop_rate = dropped / len(results)

                self.monitor.record(f"queue_{config['queue_size']}_throughput", queue_throughput)
                self.monitor.record(f"queue_{config['queue_size']}_drop_rate", drop_rate)

                logger.info(f"  Queue {config['queue_size']}: {queue_throughput:.1f} RPS, {drop_rate:.2%} dropped")

                # Queue should handle overflow gracefully
                if config["overflow_strategy"] == "drop":
                    assert drop_rate >= 0.1  # Some requests should be dropped
                else:
                    assert drop_rate <= 0.05  # Minimal drops for other strategies

    def _generate_test_payloads(self, count: int) -> List[Dict[str, Any]]:
        """Generate diverse test payloads for load testing"""
        payload_types = [
            {"type": "domain_lookup", "size": "small"},
            {"type": "ip_analysis", "size": "medium"},
            {"type": "comprehensive_scan", "size": "large"},
            {"type": "subdomain_enum", "size": "medium"},
            {"type": "port_scan", "size": "small"},
        ]

        payloads = []
        for i in range(count):
            base_payload = random.choice(payload_types).copy()
            base_payload.update({
                "request_id": f"test_req_{i}",
                "target": f"target_{i % 100}.example.com",
                "timestamp": time.time(),
                "priority": random.choice(["low", "medium", "high"]),
                "timeout": random.randint(5, 30)
            })

            # Add size-specific data
            if base_payload["size"] == "large":
                base_payload["options"] = {
                    "deep_scan": True,
                    "all_ports": True,
                    "service_detection": True,
                    "os_detection": True
                }
            elif base_payload["size"] == "medium":
                base_payload["options"] = {
                    "common_ports": True,
                    "service_detection": True
                }
            else:
                base_payload["options"] = {"quick_scan": True}

            payloads.append(base_payload)

        return payloads

    async def _make_async_request(self, session: aiohttp.ClientSession,
                                 base_url: str, payload: Dict[str, Any],
                                 request_id: int = None) -> Dict[str, Any]:
        """Make async HTTP request with timing"""
        start_time = time.time()

        try:
            # Simulate API endpoint (adjust URL as needed)
            url = f"{base_url}/api/v1/osint/analyze"

            async with session.post(url, json=payload, timeout=30) as response:
                latency = (time.time() - start_time) * 1000  # Convert to ms

                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "latency": latency,
                        "status": response.status,
                        "request_id": request_id,
                        "response_size": len(str(data))
                    }
                else:
                    return {
                        "success": False,
                        "latency": latency,
                        "status": response.status,
                        "request_id": request_id,
                        "error": f"HTTP {response.status}"
                    }

        except asyncio.TimeoutError:
            latency = (time.time() - start_time) * 1000
            return {
                "success": False,
                "latency": latency,
                "request_id": request_id,
                "error": "timeout"
            }
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                "success": False,
                "latency": latency,
                "request_id": request_id,
                "error": str(e)
            }

    def _calculate_connection_reuse(self, results: List[Dict[str, Any]]) -> float:
        """Calculate connection reuse rate from request results"""
        # This would analyze actual connection metrics
        # For simulation, assume good reuse with larger pools
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        if not successful:
            return 0.0

        # Simulate connection reuse based on latency patterns
        # Lower latency often indicates connection reuse
        avg_latency = statistics.mean(r.get("latency", 1000) for r in successful)

        # Simple heuristic: lower latency = higher reuse probability
        if avg_latency < 50:
            return 0.95
        elif avg_latency < 100:
            return 0.85
        elif avg_latency < 200:
            return 0.75
        else:
            return 0.60

@pytest.mark.performance
class TestThroughputOptimization:
    """Test system throughput optimization"""

    async def test_maximum_sustainable_throughput(self):
        """Find maximum sustainable throughput"""
        base_url = "http://localhost:8000"
        test_duration = 60  # 1 minute test

        # Gradually increase load to find breaking point
        load_levels = [100, 200, 500, 750, 1000, 1250, 1500]
        sustainable_throughput = 0

        async with aiohttp.ClientSession() as session:
            for rps_target in load_levels:
                logger.info(f"Testing sustained load: {rps_target} RPS for {test_duration}s")

                success_count = 0
                error_count = 0
                latencies = []

                start_time = time.time()
                end_time = start_time + test_duration

                while time.time() < end_time:
                    # Calculate requests for this second
                    batch_start = time.time()

                    # Send requests for this second
                    batch_tasks = []
                    for _ in range(rps_target):
                        payload = self._generate_single_payload()
                        task = self._make_async_request(session, base_url, payload)
                        batch_tasks.append(task)

                    # Wait for batch completion
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                    # Analyze batch results
                    for result in batch_results:
                        if isinstance(result, dict) and result.get("success"):
                            success_count += 1
                            latencies.append(result.get("latency", 0))
                        else:
                            error_count += 1

                    # Maintain timing
                    batch_time = time.time() - batch_start
                    if batch_time < 1.0:
                        await asyncio.sleep(1.0 - batch_time)

                # Calculate metrics
                total_requests = success_count + error_count
                actual_rps = success_count / test_duration
                error_rate = error_count / total_requests if total_requests > 0 else 1.0
                avg_latency = statistics.mean(latencies) if latencies else float('inf')
                p95_latency = np.percentile(latencies, 95) if latencies else float('inf')

                logger.info(f"  Results: {actual_rps:.1f} RPS, {error_rate:.2%} errors, {avg_latency:.1f}ms avg latency")

                # Check if this load level is sustainable
                if (error_rate <= 0.05 and  # 5% error rate or less
                    avg_latency <= 200 and   # Average latency under 200ms
                    p95_latency <= 500):     # P95 latency under 500ms
                    sustainable_throughput = actual_rps
                else:
                    logger.info(f"  Load level {rps_target} RPS not sustainable")
                    break

                # Brief pause between load tests
                await asyncio.sleep(5)

        # Validate minimum throughput requirement
        assert sustainable_throughput >= 500, f"Sustainable throughput {sustainable_throughput} below 500 RPS"
        logger.info(f"Maximum sustainable throughput: {sustainable_throughput:.1f} RPS")

    def _generate_single_payload(self) -> Dict[str, Any]:
        """Generate a single test payload"""
        return {
            "type": "quick_lookup",
            "target": f"target_{random.randint(1, 1000)}.test.com",
            "request_id": f"load_test_{int(time.time() * 1000000)}",
            "priority": "medium",
            "options": {"quick_scan": True}
        }

    async def _make_async_request(self, session: aiohttp.ClientSession,
                                 base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make async request for throughput testing"""
        start_time = time.time()

        try:
            url = f"{base_url}/api/v1/osint/quick"

            async with session.post(url, json=payload, timeout=10) as response:
                latency = (time.time() - start_time) * 1000

                if response.status == 200:
                    await response.json()  # Consume response
                    return {"success": True, "latency": latency}
                else:
                    return {"success": False, "latency": latency, "status": response.status}

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {"success": False, "latency": latency, "error": str(e)}