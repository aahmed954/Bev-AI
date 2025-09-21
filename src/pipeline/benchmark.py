#!/usr/bin/env python3
"""
BEV OSINT Framework - Request Multiplexer Performance Benchmark
Comprehensive performance testing and benchmarking suite for the
request multiplexing system.
"""

import asyncio
import time
import statistics
import json
import argparse
import sys
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

from request_multiplexer import RequestMultiplexer, Request, RequestPriority, create_multiplexer


@dataclass
class BenchmarkResult:
    """Benchmark result container"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_latency: float
    median_latency: float
    p95_latency: float
    p99_latency: float
    min_latency: float
    max_latency: float
    throughput: float
    success_rate: float
    peak_concurrency: int
    memory_usage_mb: float
    errors: List[str]


class PerformanceBenchmark:
    """Performance benchmark suite for RequestMultiplexer"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.multiplexer: RequestMultiplexer = None
        self.results: List[BenchmarkResult] = []

    async def setup(self):
        """Setup the benchmark environment"""
        print("🚀 Setting up RequestMultiplexer benchmark environment...")

        self.multiplexer = create_multiplexer(self.config)
        await self.multiplexer.start()

        print("✅ Benchmark environment ready")

    async def teardown(self):
        """Cleanup benchmark environment"""
        print("🧹 Cleaning up benchmark environment...")

        if self.multiplexer:
            await self.multiplexer.stop()

        print("✅ Cleanup complete")

    async def benchmark_basic_throughput(self, num_requests: int = 100) -> BenchmarkResult:
        """Benchmark basic throughput with simple GET requests"""
        print(f"📊 Running basic throughput test with {num_requests} requests...")

        # Prepare requests
        requests = [
            Request(url=f'https://httpbin.org/get?id={i}', method='GET')
            for i in range(num_requests)
        ]

        latencies = []
        errors = []

        start_time = time.time()

        try:
            # Submit all requests
            request_ids = await self.multiplexer.bulk_submit(requests)

            # Wait for completion
            completed = await self.multiplexer.wait_for_completion(request_ids, timeout=300.0)

            end_time = time.time()
            total_time = end_time - start_time

            # Collect latencies
            successful = 0
            failed = 0

            for req in completed:
                if req.latency:
                    latencies.append(req.latency)

                if req.result and 'status_code' in req.result:
                    if req.result['status_code'] < 400:
                        successful += 1
                    else:
                        failed += 1
                        errors.append(f"HTTP {req.result['status_code']}")
                else:
                    failed += 1
                    errors.append(req.last_error or "Unknown error")

            # Calculate statistics
            if latencies:
                avg_latency = statistics.mean(latencies)
                median_latency = statistics.median(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)

                if len(latencies) >= 20:
                    quantiles = statistics.quantiles(latencies, n=100)
                    p95_latency = quantiles[94]
                    p99_latency = quantiles[98]
                else:
                    p95_latency = max_latency
                    p99_latency = max_latency
            else:
                avg_latency = median_latency = min_latency = max_latency = 0.0
                p95_latency = p99_latency = 0.0

            # Get system stats
            stats = self.multiplexer.get_statistics()
            peak_concurrency = stats['performance']['peak_concurrency']

            # Memory usage (simplified)
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_usage_mb = process.memory_info().rss / 1024 / 1024

            return BenchmarkResult(
                test_name="Basic Throughput",
                total_requests=num_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time=total_time,
                avg_latency=avg_latency,
                median_latency=median_latency,
                p95_latency=p95_latency,
                p99_latency=p99_latency,
                min_latency=min_latency,
                max_latency=max_latency,
                throughput=successful / total_time if total_time > 0 else 0,
                success_rate=successful / num_requests if num_requests > 0 else 0,
                peak_concurrency=peak_concurrency,
                memory_usage_mb=memory_usage_mb,
                errors=list(set(errors))  # Unique errors
            )

        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return BenchmarkResult(
                test_name="Basic Throughput",
                total_requests=num_requests,
                successful_requests=0,
                failed_requests=num_requests,
                total_time=0,
                avg_latency=0,
                median_latency=0,
                p95_latency=0,
                p99_latency=0,
                min_latency=0,
                max_latency=0,
                throughput=0,
                success_rate=0,
                peak_concurrency=0,
                memory_usage_mb=0,
                errors=[str(e)]
            )

    async def benchmark_concurrency_scaling(self) -> List[BenchmarkResult]:
        """Benchmark performance across different concurrency levels"""
        print("📈 Running concurrency scaling benchmark...")

        concurrency_levels = [10, 25, 50, 100, 200, 500]
        results = []

        for concurrency in concurrency_levels:
            if concurrency > self.config['max_concurrency']:
                print(f"⏭️  Skipping concurrency {concurrency} (exceeds max_concurrency)")
                continue

            print(f"  Testing concurrency level: {concurrency}")

            # Create requests with staggered submission
            requests = [
                Request(url=f'https://httpbin.org/delay/1?id={i}', method='GET')
                for i in range(concurrency)
            ]

            latencies = []
            start_time = time.time()

            try:
                # Submit requests in batches to simulate real concurrency
                batch_size = min(20, concurrency)
                request_ids = []

                for i in range(0, len(requests), batch_size):
                    batch = requests[i:i + batch_size]
                    batch_ids = await self.multiplexer.bulk_submit(batch)
                    request_ids.extend(batch_ids)

                    # Small delay between batches
                    if i + batch_size < len(requests):
                        await asyncio.sleep(0.1)

                # Wait for all to complete
                completed = await self.multiplexer.wait_for_completion(request_ids, timeout=120.0)
                end_time = time.time()

                # Process results
                successful = sum(1 for req in completed if req.result and req.result.get('status_code', 0) < 400)
                failed = len(completed) - successful

                for req in completed:
                    if req.latency:
                        latencies.append(req.latency)

                # Calculate metrics
                total_time = end_time - start_time
                throughput = successful / total_time if total_time > 0 else 0

                if latencies:
                    avg_latency = statistics.mean(latencies)
                    p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies)
                else:
                    avg_latency = p95_latency = 0

                result = BenchmarkResult(
                    test_name=f"Concurrency {concurrency}",
                    total_requests=concurrency,
                    successful_requests=successful,
                    failed_requests=failed,
                    total_time=total_time,
                    avg_latency=avg_latency,
                    median_latency=statistics.median(latencies) if latencies else 0,
                    p95_latency=p95_latency,
                    p99_latency=p95_latency,  # Simplified
                    min_latency=min(latencies) if latencies else 0,
                    max_latency=max(latencies) if latencies else 0,
                    throughput=throughput,
                    success_rate=successful / concurrency,
                    peak_concurrency=self.multiplexer.get_statistics()['performance']['peak_concurrency'],
                    memory_usage_mb=0,  # Skip for this test
                    errors=[]
                )

                results.append(result)
                print(f"    ✅ Throughput: {throughput:.2f} req/s, Success Rate: {result.success_rate:.2%}")

            except Exception as e:
                print(f"    ❌ Failed: {e}")

            # Cool down between tests
            await asyncio.sleep(2)

        return results

    async def benchmark_mixed_workload(self, duration_seconds: int = 60) -> BenchmarkResult:
        """Benchmark with mixed request types and priorities"""
        print(f"🔄 Running mixed workload benchmark for {duration_seconds} seconds...")

        start_time = time.time()
        end_time = start_time + duration_seconds

        request_count = 0
        latencies = []
        errors = []
        request_ids = []

        try:
            while time.time() < end_time:
                # Generate different types of requests
                request_type = request_count % 4

                if request_type == 0:
                    # Simple GET
                    request = Request(
                        url='https://httpbin.org/get',
                        method='GET',
                        priority=RequestPriority.MEDIUM
                    )
                elif request_type == 1:
                    # POST with data
                    request = Request(
                        url='https://httpbin.org/post',
                        method='POST',
                        data={'test': f'data_{request_count}'},
                        priority=RequestPriority.MEDIUM
                    )
                elif request_type == 2:
                    # Delayed request
                    request = Request(
                        url='https://httpbin.org/delay/1',
                        method='GET',
                        priority=RequestPriority.LOW
                    )
                else:
                    # High priority request
                    request = Request(
                        url='https://httpbin.org/uuid',
                        method='GET',
                        priority=RequestPriority.HIGH
                    )

                request_id = await self.multiplexer.submit_request(request)
                request_ids.append(request_id)
                request_count += 1

                # Vary submission rate
                await asyncio.sleep(0.05 + (request_count % 10) * 0.01)

            # Wait for all requests to complete
            print("  Waiting for all requests to complete...")
            completed = await self.multiplexer.wait_for_completion(request_ids, timeout=300.0)

            # Process results
            successful = 0
            failed = 0

            for req in completed:
                if req.latency:
                    latencies.append(req.latency)

                if req.result and req.result.get('status_code', 0) < 400:
                    successful += 1
                else:
                    failed += 1
                    if req.last_error:
                        errors.append(req.last_error)

            total_time = time.time() - start_time

            # Calculate statistics
            if latencies:
                avg_latency = statistics.mean(latencies)
                median_latency = statistics.median(latencies)
                p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies)
                p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) > 100 else max(latencies)
            else:
                avg_latency = median_latency = p95_latency = p99_latency = 0

            return BenchmarkResult(
                test_name="Mixed Workload",
                total_requests=request_count,
                successful_requests=successful,
                failed_requests=failed,
                total_time=total_time,
                avg_latency=avg_latency,
                median_latency=median_latency,
                p95_latency=p95_latency,
                p99_latency=p99_latency,
                min_latency=min(latencies) if latencies else 0,
                max_latency=max(latencies) if latencies else 0,
                throughput=successful / total_time if total_time > 0 else 0,
                success_rate=successful / request_count if request_count > 0 else 0,
                peak_concurrency=self.multiplexer.get_statistics()['performance']['peak_concurrency'],
                memory_usage_mb=0,
                errors=list(set(errors))[:10]  # First 10 unique errors
            )

        except Exception as e:
            print(f"❌ Mixed workload benchmark failed: {e}")
            return BenchmarkResult(
                test_name="Mixed Workload",
                total_requests=request_count,
                successful_requests=0,
                failed_requests=request_count,
                total_time=0,
                avg_latency=0,
                median_latency=0,
                p95_latency=0,
                p99_latency=0,
                min_latency=0,
                max_latency=0,
                throughput=0,
                success_rate=0,
                peak_concurrency=0,
                memory_usage_mb=0,
                errors=[str(e)]
            )

    async def run_all_benchmarks(self):
        """Run all benchmark tests"""
        print("🎯 Starting comprehensive performance benchmarks...\n")

        await self.setup()

        try:
            # Basic throughput test
            result1 = await self.benchmark_basic_throughput(100)
            self.results.append(result1)

            # Concurrency scaling
            scaling_results = await self.benchmark_concurrency_scaling()
            self.results.extend(scaling_results)

            # Mixed workload
            result3 = await self.benchmark_mixed_workload(30)  # 30 seconds
            self.results.append(result3)

        finally:
            await self.teardown()

    def print_results(self):
        """Print benchmark results in a formatted way"""
        print("\n" + "="*80)
        print("🏆 BENCHMARK RESULTS")
        print("="*80)

        for result in self.results:
            print(f"\n📊 {result.test_name}")
            print(f"   Total Requests: {result.total_requests}")
            print(f"   Successful: {result.successful_requests}")
            print(f"   Failed: {result.failed_requests}")
            print(f"   Success Rate: {result.success_rate:.2%}")
            print(f"   Total Time: {result.total_time:.2f}s")
            print(f"   Throughput: {result.throughput:.2f} req/s")
            print(f"   Average Latency: {result.avg_latency:.3f}s")
            print(f"   Median Latency: {result.median_latency:.3f}s")
            print(f"   95th Percentile: {result.p95_latency:.3f}s")
            print(f"   Peak Concurrency: {result.peak_concurrency}")

            if result.memory_usage_mb > 0:
                print(f"   Memory Usage: {result.memory_usage_mb:.2f} MB")

            if result.errors:
                print(f"   Errors: {', '.join(result.errors[:3])}")
                if len(result.errors) > 3:
                    print(f"           ... and {len(result.errors) - 3} more")

        # Summary
        print("\n📈 PERFORMANCE SUMMARY")
        print("-" * 40)

        throughputs = [r.throughput for r in self.results if r.throughput > 0]
        if throughputs:
            print(f"   Best Throughput: {max(throughputs):.2f} req/s")
            print(f"   Average Throughput: {statistics.mean(throughputs):.2f} req/s")

        latencies = [r.avg_latency for r in self.results if r.avg_latency > 0]
        if latencies:
            print(f"   Best Avg Latency: {min(latencies):.3f}s")
            print(f"   Overall Avg Latency: {statistics.mean(latencies):.3f}s")

        success_rates = [r.success_rate for r in self.results]
        print(f"   Overall Success Rate: {statistics.mean(success_rates):.2%}")

    def save_results(self, filename: str):
        """Save results to JSON file"""
        results_data = []
        for result in self.results:
            results_data.append({
                'test_name': result.test_name,
                'total_requests': result.total_requests,
                'successful_requests': result.successful_requests,
                'failed_requests': result.failed_requests,
                'total_time': result.total_time,
                'avg_latency': result.avg_latency,
                'median_latency': result.median_latency,
                'p95_latency': result.p95_latency,
                'p99_latency': result.p99_latency,
                'min_latency': result.min_latency,
                'max_latency': result.max_latency,
                'throughput': result.throughput,
                'success_rate': result.success_rate,
                'peak_concurrency': result.peak_concurrency,
                'memory_usage_mb': result.memory_usage_mb,
                'errors': result.errors
            })

        benchmark_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'results': results_data
        }

        with open(filename, 'w') as f:
            json.dump(benchmark_data, f, indent=2)

        print(f"💾 Results saved to {filename}")


def create_benchmark_config() -> Dict[str, Any]:
    """Create benchmark configuration"""
    return {
        'max_concurrency': 1000,
        'worker_count': 50,
        'request_timeout': 30.0,
        'retry_attempts': 2,
        'enable_caching': True,
        'cache_ttl': 300,
        'connection_pool': {
            'max_connections': 500,
            'max_connections_per_host': 50,
            'connection_timeout': 30.0,
            'idle_timeout': 300,
            'enable_proxy_rotation': False  # Disabled for consistent benchmarking
        },
        'rate_limiting': {
            'global_limit': 2000,
            'global_window': 60,
            'per_host_limit': 200,
            'per_host_window': 60
        },
        'queue_manager': {
            'queue_type': 'memory',
            'max_queue_size': 10000,
            'enable_backpressure': True,
            'backpressure_threshold': 8000
        }
    }


async def main():
    """Main benchmark execution"""
    parser = argparse.ArgumentParser(description='RequestMultiplexer Performance Benchmark')
    parser.add_argument('--output', '-o', default='benchmark_results.json',
                       help='Output file for results (default: benchmark_results.json)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmark with reduced load')

    args = parser.parse_args()

    # Create configuration
    config = create_benchmark_config()

    if args.quick:
        # Reduce load for quick testing
        config['max_concurrency'] = 100
        config['worker_count'] = 10

    # Run benchmarks
    benchmark = PerformanceBenchmark(config)

    try:
        await benchmark.run_all_benchmarks()
        benchmark.print_results()
        benchmark.save_results(args.output)

        print(f"\n✅ Benchmark completed successfully!")

        # Performance validation
        successful_results = [r for r in benchmark.results if r.success_rate > 0.8]
        if successful_results:
            best_throughput = max(r.throughput for r in successful_results)
            avg_latency = statistics.mean(r.avg_latency for r in successful_results if r.avg_latency > 0)

            print(f"🎯 Performance Summary:")
            print(f"   Peak Throughput: {best_throughput:.2f} req/s")
            print(f"   Average Latency: {avg_latency:.3f}s")

            # Validate against requirements
            if best_throughput >= 50:  # 50 req/s minimum
                print("✅ Throughput requirement MET")
            else:
                print("❌ Throughput requirement NOT MET")

            if avg_latency <= 2.0:  # 2 second max average
                print("✅ Latency requirement MET")
            else:
                print("❌ Latency requirement NOT MET")

    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
        await benchmark.teardown()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        await benchmark.teardown()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())