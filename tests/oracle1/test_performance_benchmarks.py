#!/usr/bin/env python3
"""
Performance Benchmarking Suite for ORACLE1 ARM Deployment

This module provides comprehensive performance testing including:
- Resource utilization monitoring
- Service response time benchmarking
- Concurrent load testing
- Memory and CPU efficiency validation
- Network performance testing
"""

import asyncio
import json
import logging
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp
import docker
import psutil
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceBenchmarker:
    """Comprehensive performance benchmarking for ORACLE1 services."""

    def __init__(self, project_root: str = "/home/starlord/Projects/Bev"):
        self.project_root = Path(project_root)
        self.docker_client = docker.from_env()
        self.results = {
            "timestamp": time.time(),
            "benchmarks": [],
            "system_info": self._get_system_info(),
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "warnings": 0
            }
        }

        # Performance thresholds (configurable)
        self.thresholds = {
            "max_response_time": 1000,  # milliseconds
            "max_cpu_usage": 80,        # percentage
            "max_memory_usage": 2048,   # MB per service
            "min_throughput": 100,      # requests per second
            "max_latency_p95": 2000,    # milliseconds
            "max_error_rate": 1.0,      # percentage
        }

        # Service endpoints for testing
        self.service_endpoints = {
            "nginx": "http://localhost:80",
            "n8n": "http://localhost:5678",
            "prometheus": "http://localhost:9090",
            "grafana": "http://localhost:3000",
            "alertmanager": "http://localhost:9093",
            "vault": "http://localhost:8200",
            "minio1": "http://localhost:9001",
            "minio2": "http://localhost:9002",
            "minio3": "http://localhost:9003",
            "litellm-1": "http://localhost:5000",
            "litellm-2": "http://localhost:5001",
            "litellm-3": "http://localhost:5002",
            "request-multiplexer": "http://localhost:8080",
            "influxdb-primary": "http://localhost:8086",
            "influxdb-replica": "http://localhost:8087"
        }

    def _get_system_info(self) -> Dict:
        """Get system information for benchmarking context."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_physical": psutil.cpu_count(logical=False),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_total": psutil.disk_usage('/').total,
            "disk_free": psutil.disk_usage('/').free,
            "platform": sys.platform,
            "architecture": "arm64"  # Known for ORACLE1
        }

    def log_benchmark_result(self, test_name: str, status: str, details: str = "",
                           metrics: Dict = None, execution_time: float = 0.0):
        """Log a benchmark result."""
        result = {
            "test_name": test_name,
            "status": status,  # "PASS", "FAIL", "WARN"
            "details": details,
            "metrics": metrics or {},
            "execution_time": execution_time,
            "timestamp": time.time()
        }

        self.results["benchmarks"].append(result)
        self.results["summary"]["total_tests"] += 1

        if status == "PASS":
            self.results["summary"]["passed_tests"] += 1
            logger.info(f"[PASS] {test_name}: {details}")
        elif status == "FAIL":
            self.results["summary"]["failed_tests"] += 1
            logger.error(f"[FAIL] {test_name}: {details}")
        elif status == "WARN":
            self.results["summary"]["warnings"] += 1
            logger.warning(f"[WARN] {test_name}: {details}")

    async def benchmark_system_resources(self) -> bool:
        """Benchmark overall system resource utilization."""
        logger.info("Benchmarking system resource utilization...")

        start_time = time.time()

        # Collect baseline metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        load_avg = psutil.getloadavg()

        # Network I/O
        net_io_start = psutil.net_io_counters()
        await asyncio.sleep(1)
        net_io_end = psutil.net_io_counters()

        net_bytes_sent = net_io_end.bytes_sent - net_io_start.bytes_sent
        net_bytes_recv = net_io_end.bytes_recv - net_io_start.bytes_recv

        execution_time = time.time() - start_time

        metrics = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / 1024 / 1024,
            "memory_available_mb": memory.available / 1024 / 1024,
            "disk_percent": (disk.used / disk.total) * 100,
            "disk_free_gb": disk.free / 1024 / 1024 / 1024,
            "load_1min": load_avg[0],
            "load_5min": load_avg[1],
            "load_15min": load_avg[2],
            "network_bytes_sent_per_sec": net_bytes_sent,
            "network_bytes_recv_per_sec": net_bytes_recv
        }

        # Evaluate against thresholds
        issues = []
        if cpu_percent > self.thresholds["max_cpu_usage"]:
            issues.append(f"CPU usage high: {cpu_percent:.1f}%")

        if memory.percent > 90:
            issues.append(f"Memory usage critical: {memory.percent:.1f}%")

        if disk.free < 10 * 1024 * 1024 * 1024:  # 10GB
            issues.append(f"Disk space low: {disk.free / 1024 / 1024 / 1024:.1f}GB")

        if load_avg[0] > psutil.cpu_count() * 2:
            issues.append(f"Load average high: {load_avg[0]:.2f}")

        if issues:
            self.log_benchmark_result(
                "system_resource_utilization",
                "WARN" if len(issues) <= 2 else "FAIL",
                f"Resource issues detected: {'; '.join(issues)}",
                metrics,
                execution_time
            )
            return len(issues) <= 2
        else:
            self.log_benchmark_result(
                "system_resource_utilization",
                "PASS",
                f"System resources optimal: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%, Load {load_avg[0]:.2f}",
                metrics,
                execution_time
            )
            return True

    async def benchmark_docker_performance(self) -> bool:
        """Benchmark Docker container performance."""
        logger.info("Benchmarking Docker container performance...")

        start_time = time.time()

        try:
            # Get running containers
            containers = self.docker_client.containers.list()
            container_metrics = {}

            for container in containers:
                try:
                    # Get container stats
                    stats = container.stats(stream=False)

                    # Parse CPU usage
                    cpu_stats = stats['cpu_stats']
                    precpu_stats = stats['precpu_stats']

                    cpu_usage = 0.0
                    if 'cpu_usage' in cpu_stats and 'cpu_usage' in precpu_stats:
                        cpu_delta = cpu_stats['cpu_usage']['total_usage'] - precpu_stats['cpu_usage']['total_usage']
                        system_delta = cpu_stats['system_cpu_usage'] - precpu_stats['system_cpu_usage']
                        online_cpus = len(cpu_stats['cpu_usage']['percpu_usage'])

                        if system_delta > 0:
                            cpu_usage = (cpu_delta / system_delta) * online_cpus * 100.0

                    # Parse memory usage
                    memory_stats = stats['memory_stats']
                    memory_usage = memory_stats.get('usage', 0)
                    memory_limit = memory_stats.get('limit', 1)
                    memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0

                    container_metrics[container.name] = {
                        "cpu_percent": cpu_usage,
                        "memory_usage_mb": memory_usage / 1024 / 1024,
                        "memory_percent": memory_percent,
                        "status": container.status
                    }

                except Exception as e:
                    logger.warning(f"Failed to get stats for container {container.name}: {str(e)}")

            execution_time = time.time() - start_time

            # Analyze container performance
            high_cpu_containers = []
            high_memory_containers = []
            total_memory_usage = 0

            for container_name, metrics in container_metrics.items():
                if metrics["cpu_percent"] > self.thresholds["max_cpu_usage"]:
                    high_cpu_containers.append(f"{container_name}: {metrics['cpu_percent']:.1f}%")

                if metrics["memory_usage_mb"] > self.thresholds["max_memory_usage"]:
                    high_memory_containers.append(f"{container_name}: {metrics['memory_usage_mb']:.1f}MB")

                total_memory_usage += metrics["memory_usage_mb"]

            issues = []
            if high_cpu_containers:
                issues.append(f"High CPU: {', '.join(high_cpu_containers)}")
            if high_memory_containers:
                issues.append(f"High Memory: {', '.join(high_memory_containers)}")

            metrics = {
                "container_count": len(containers),
                "total_memory_usage_mb": total_memory_usage,
                "average_memory_per_container": total_memory_usage / len(containers) if containers else 0,
                "container_details": container_metrics
            }

            if issues:
                self.log_benchmark_result(
                    "docker_container_performance",
                    "WARN",
                    f"Container performance issues: {'; '.join(issues)}",
                    metrics,
                    execution_time
                )
                return False
            else:
                self.log_benchmark_result(
                    "docker_container_performance",
                    "PASS",
                    f"All {len(containers)} containers performing well (avg memory: {total_memory_usage / len(containers) if containers else 0:.1f}MB)",
                    metrics,
                    execution_time
                )
                return True

        except Exception as e:
            self.log_benchmark_result(
                "docker_container_performance",
                "FAIL",
                f"Docker performance benchmark failed: {str(e)}",
                {},
                time.time() - start_time
            )
            return False

    async def benchmark_service_response_times(self) -> bool:
        """Benchmark service response times."""
        logger.info("Benchmarking service response times...")

        start_time = time.time()
        response_times = {}
        all_services_healthy = True

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for service_name, endpoint in self.service_endpoints.items():
                service_start = time.time()

                try:
                    # Test health/status endpoint first
                    health_endpoints = [
                        f"{endpoint}/health",
                        f"{endpoint}/-/healthy",
                        f"{endpoint}/api/health",
                        f"{endpoint}/v1/sys/health",
                        f"{endpoint}",
                    ]

                    response_time = None
                    status_code = None

                    for health_endpoint in health_endpoints:
                        try:
                            request_start = time.time()
                            async with session.get(health_endpoint) as response:
                                response_time = (time.time() - request_start) * 1000  # ms
                                status_code = response.status
                                if status_code in [200, 201, 204, 302]:
                                    break
                        except:
                            continue

                    if response_time is not None:
                        response_times[service_name] = {
                            "response_time_ms": response_time,
                            "status_code": status_code,
                            "status": "healthy" if status_code in [200, 201, 204, 302] else "unhealthy"
                        }

                        if response_time > self.thresholds["max_response_time"]:
                            all_services_healthy = False
                    else:
                        response_times[service_name] = {
                            "response_time_ms": None,
                            "status_code": None,
                            "status": "unreachable"
                        }
                        all_services_healthy = False

                except Exception as e:
                    response_times[service_name] = {
                        "response_time_ms": None,
                        "status_code": None,
                        "status": "error",
                        "error": str(e)
                    }
                    all_services_healthy = False

        execution_time = time.time() - start_time

        # Calculate statistics
        healthy_times = [rt["response_time_ms"] for rt in response_times.values()
                        if rt["response_time_ms"] is not None]

        if healthy_times:
            avg_response_time = statistics.mean(healthy_times)
            p95_response_time = statistics.quantiles(healthy_times, n=20)[18]  # 95th percentile
            max_response_time = max(healthy_times)
        else:
            avg_response_time = p95_response_time = max_response_time = 0

        metrics = {
            "service_count": len(self.service_endpoints),
            "healthy_services": len([rt for rt in response_times.values() if rt["status"] == "healthy"]),
            "average_response_time_ms": avg_response_time,
            "p95_response_time_ms": p95_response_time,
            "max_response_time_ms": max_response_time,
            "service_details": response_times
        }

        if all_services_healthy and p95_response_time <= self.thresholds["max_latency_p95"]:
            self.log_benchmark_result(
                "service_response_times",
                "PASS",
                f"All services responsive (avg: {avg_response_time:.1f}ms, p95: {p95_response_time:.1f}ms)",
                metrics,
                execution_time
            )
            return True
        else:
            unhealthy_services = [name for name, rt in response_times.items() if rt["status"] != "healthy"]
            self.log_benchmark_result(
                "service_response_times",
                "FAIL" if len(unhealthy_services) > 3 else "WARN",
                f"Service issues detected. Unhealthy: {', '.join(unhealthy_services)} (p95: {p95_response_time:.1f}ms)",
                metrics,
                execution_time
            )
            return len(unhealthy_services) <= 3

    async def benchmark_concurrent_load(self) -> bool:
        """Benchmark concurrent load handling."""
        logger.info("Benchmarking concurrent load handling...")

        start_time = time.time()

        # Test concurrent requests to nginx (main entry point)
        nginx_endpoint = "http://localhost:80"
        concurrent_requests = 50
        request_timeout = 5

        async def make_request(session, request_id):
            try:
                request_start = time.time()
                async with session.get(nginx_endpoint, timeout=aiohttp.ClientTimeout(total=request_timeout)) as response:
                    request_time = (time.time() - request_start) * 1000
                    return {
                        "request_id": request_id,
                        "response_time_ms": request_time,
                        "status_code": response.status,
                        "success": response.status < 400
                    }
            except asyncio.TimeoutError:
                return {
                    "request_id": request_id,
                    "response_time_ms": request_timeout * 1000,
                    "status_code": 408,
                    "success": False,
                    "error": "timeout"
                }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "response_time_ms": None,
                    "status_code": 500,
                    "success": False,
                    "error": str(e)
                }

        # Execute concurrent requests
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session, i) for i in range(concurrent_requests)]
            results = await asyncio.gather(*tasks)

        execution_time = time.time() - start_time

        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]

        success_rate = len(successful_requests) / len(results) * 100
        throughput = len(successful_requests) / execution_time  # requests per second

        if successful_requests:
            response_times = [r["response_time_ms"] for r in successful_requests]
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times)
        else:
            avg_response_time = p95_response_time = 0

        metrics = {
            "concurrent_requests": concurrent_requests,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate_percent": success_rate,
            "throughput_rps": throughput,
            "average_response_time_ms": avg_response_time,
            "p95_response_time_ms": p95_response_time,
            "execution_time_s": execution_time
        }

        # Evaluate performance
        if (success_rate >= 99.0 and
            throughput >= self.thresholds["min_throughput"] and
            p95_response_time <= self.thresholds["max_latency_p95"]):
            self.log_benchmark_result(
                "concurrent_load_handling",
                "PASS",
                f"Excellent load handling: {success_rate:.1f}% success, {throughput:.1f} RPS, {p95_response_time:.1f}ms p95",
                metrics,
                execution_time
            )
            return True
        elif success_rate >= 95.0 and throughput >= self.thresholds["min_throughput"] * 0.8:
            self.log_benchmark_result(
                "concurrent_load_handling",
                "WARN",
                f"Acceptable load handling: {success_rate:.1f}% success, {throughput:.1f} RPS, {p95_response_time:.1f}ms p95",
                metrics,
                execution_time
            )
            return True
        else:
            self.log_benchmark_result(
                "concurrent_load_handling",
                "FAIL",
                f"Poor load handling: {success_rate:.1f}% success, {throughput:.1f} RPS, {p95_response_time:.1f}ms p95",
                metrics,
                execution_time
            )
            return False

    async def benchmark_redis_performance(self) -> bool:
        """Benchmark Redis performance."""
        logger.info("Benchmarking Redis performance...")

        start_time = time.time()

        try:
            # Connect to Redis
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

            # Test Redis connectivity
            ping_start = time.time()
            redis_client.ping()
            ping_time = (time.time() - ping_start) * 1000

            # Benchmark basic operations
            operations = {
                "set": [],
                "get": [],
                "delete": []
            }

            # Perform benchmark operations
            for i in range(100):
                key = f"benchmark_key_{i}"
                value = f"benchmark_value_{i}"

                # SET operation
                set_start = time.time()
                redis_client.set(key, value)
                operations["set"].append((time.time() - set_start) * 1000)

                # GET operation
                get_start = time.time()
                retrieved_value = redis_client.get(key)
                operations["get"].append((time.time() - get_start) * 1000)

                # DELETE operation
                del_start = time.time()
                redis_client.delete(key)
                operations["delete"].append((time.time() - del_start) * 1000)

            # Calculate statistics
            avg_set_time = statistics.mean(operations["set"])
            avg_get_time = statistics.mean(operations["get"])
            avg_del_time = statistics.mean(operations["delete"])

            p95_set_time = statistics.quantiles(operations["set"], n=20)[18]
            p95_get_time = statistics.quantiles(operations["get"], n=20)[18]
            p95_del_time = statistics.quantiles(operations["delete"], n=20)[18]

            execution_time = time.time() - start_time

            metrics = {
                "ping_time_ms": ping_time,
                "operations_tested": 100,
                "avg_set_time_ms": avg_set_time,
                "avg_get_time_ms": avg_get_time,
                "avg_delete_time_ms": avg_del_time,
                "p95_set_time_ms": p95_set_time,
                "p95_get_time_ms": p95_get_time,
                "p95_delete_time_ms": p95_del_time,
                "total_operations": 300
            }

            # Evaluate Redis performance
            if (ping_time < 5 and avg_get_time < 1 and avg_set_time < 2):
                self.log_benchmark_result(
                    "redis_performance",
                    "PASS",
                    f"Redis performance excellent: ping {ping_time:.2f}ms, get {avg_get_time:.2f}ms, set {avg_set_time:.2f}ms",
                    metrics,
                    execution_time
                )
                return True
            elif ping_time < 20 and avg_get_time < 5:
                self.log_benchmark_result(
                    "redis_performance",
                    "WARN",
                    f"Redis performance acceptable: ping {ping_time:.2f}ms, get {avg_get_time:.2f}ms, set {avg_set_time:.2f}ms",
                    metrics,
                    execution_time
                )
                return True
            else:
                self.log_benchmark_result(
                    "redis_performance",
                    "FAIL",
                    f"Redis performance poor: ping {ping_time:.2f}ms, get {avg_get_time:.2f}ms, set {avg_set_time:.2f}ms",
                    metrics,
                    execution_time
                )
                return False

        except Exception as e:
            self.log_benchmark_result(
                "redis_performance",
                "FAIL",
                f"Redis benchmark failed: {str(e)}",
                {},
                time.time() - start_time
            )
            return False

    async def benchmark_storage_performance(self) -> bool:
        """Benchmark storage I/O performance."""
        logger.info("Benchmarking storage I/O performance...")

        start_time = time.time()

        try:
            # Create test directory
            test_dir = Path("/tmp/bev_storage_benchmark")
            test_dir.mkdir(exist_ok=True)

            # Write test
            write_times = []
            test_data = "x" * 1024 * 1024  # 1MB of data

            for i in range(10):
                write_start = time.time()
                with open(test_dir / f"test_file_{i}.txt", 'w') as f:
                    f.write(test_data)
                    f.flush()
                write_times.append((time.time() - write_start) * 1000)

            # Read test
            read_times = []
            for i in range(10):
                read_start = time.time()
                with open(test_dir / f"test_file_{i}.txt", 'r') as f:
                    data = f.read()
                read_times.append((time.time() - read_start) * 1000)

            # Calculate statistics
            avg_write_time = statistics.mean(write_times)
            avg_read_time = statistics.mean(read_times)
            write_throughput_mbps = (1 / (avg_write_time / 1000))  # MB/s
            read_throughput_mbps = (1 / (avg_read_time / 1000))   # MB/s

            # Cleanup
            for i in range(10):
                (test_dir / f"test_file_{i}.txt").unlink()
            test_dir.rmdir()

            execution_time = time.time() - start_time

            metrics = {
                "avg_write_time_ms": avg_write_time,
                "avg_read_time_ms": avg_read_time,
                "write_throughput_mbps": write_throughput_mbps,
                "read_throughput_mbps": read_throughput_mbps,
                "test_file_size_mb": 1,
                "test_iterations": 10
            }

            # Evaluate storage performance
            if write_throughput_mbps >= 50 and read_throughput_mbps >= 100:
                self.log_benchmark_result(
                    "storage_io_performance",
                    "PASS",
                    f"Storage performance excellent: write {write_throughput_mbps:.1f}MB/s, read {read_throughput_mbps:.1f}MB/s",
                    metrics,
                    execution_time
                )
                return True
            elif write_throughput_mbps >= 20 and read_throughput_mbps >= 50:
                self.log_benchmark_result(
                    "storage_io_performance",
                    "WARN",
                    f"Storage performance acceptable: write {write_throughput_mbps:.1f}MB/s, read {read_throughput_mbps:.1f}MB/s",
                    metrics,
                    execution_time
                )
                return True
            else:
                self.log_benchmark_result(
                    "storage_io_performance",
                    "FAIL",
                    f"Storage performance poor: write {write_throughput_mbps:.1f}MB/s, read {read_throughput_mbps:.1f}MB/s",
                    metrics,
                    execution_time
                )
                return False

        except Exception as e:
            self.log_benchmark_result(
                "storage_io_performance",
                "FAIL",
                f"Storage I/O benchmark failed: {str(e)}",
                {},
                time.time() - start_time
            )
            return False

    async def run_all_benchmarks(self) -> Dict:
        """Run all performance benchmarks."""
        logger.info("Starting comprehensive performance benchmarking...")

        start_time = time.time()

        # Run all benchmark phases
        benchmark_phases = [
            ("System Resources", self.benchmark_system_resources),
            ("Docker Performance", self.benchmark_docker_performance),
            ("Service Response Times", self.benchmark_service_response_times),
            ("Concurrent Load", self.benchmark_concurrent_load),
            ("Redis Performance", self.benchmark_redis_performance),
            ("Storage I/O Performance", self.benchmark_storage_performance)
        ]

        for phase_name, benchmark_func in benchmark_phases:
            logger.info(f"Running benchmark phase: {phase_name}")
            try:
                await benchmark_func()
            except Exception as e:
                logger.error(f"Benchmark phase {phase_name} failed with exception: {str(e)}")
                self.log_benchmark_result(
                    f"phase_{phase_name.lower().replace(' ', '_')}",
                    "FAIL",
                    f"Phase failed with exception: {str(e)}"
                )

        # Calculate final results
        total_time = time.time() - start_time
        self.results["total_execution_time"] = total_time

        success_rate = (self.results["summary"]["passed_tests"] /
                       self.results["summary"]["total_tests"] * 100) if self.results["summary"]["total_tests"] > 0 else 0

        self.results["summary"]["success_rate"] = success_rate

        logger.info(f"Performance benchmarking completed in {total_time:.2f}s")
        logger.info(f"Results: {self.results['summary']['passed_tests']}/{self.results['summary']['total_tests']} passed "
                   f"({success_rate:.1f}% success rate)")

        return self.results

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save benchmark results to JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"/home/starlord/Projects/Bev/validation_results/performance_benchmarks_{timestamp}.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Performance benchmark results saved to: {output_path}")
        return str(output_path)


async def main():
    """Main entry point for performance benchmarking."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "/home/starlord/Projects/Bev"

    benchmarker = PerformanceBenchmarker(project_root)

    try:
        results = await benchmarker.run_all_benchmarks()
        output_file = benchmarker.save_results()

        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE BENCHMARKING SUMMARY")
        print("="*60)
        print(f"Total Tests: {results['summary']['total_tests']}")
        print(f"Passed: {results['summary']['passed_tests']}")
        print(f"Failed: {results['summary']['failed_tests']}")
        print(f"Warnings: {results['summary']['warnings']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Execution Time: {results['total_execution_time']:.2f}s")
        print(f"Results saved to: {output_file}")
        print("="*60)

        # Exit with appropriate code
        if results['summary']['failed_tests'] == 0:
            print("✅ All performance benchmarks passed!")
            sys.exit(0)
        else:
            print("❌ Some performance benchmarks failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Performance benchmarking interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Performance benchmarking failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())