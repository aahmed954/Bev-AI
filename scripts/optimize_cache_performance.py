#!/usr/bin/env python3
"""
BEV OSINT Framework - Predictive Cache Performance Optimization Script

Automated optimization script for the predictive cache system that analyzes
performance metrics and applies optimizations to improve cache efficiency.

Key Optimization Areas:
- Tier size allocation based on workload patterns
- ML model retraining for improved predictions
- Cache warming strategy optimization
- Memory utilization optimization
- Configuration parameter tuning
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import aiohttp
import yaml
import argparse

@dataclass
class OptimizationConfig:
    """Configuration for optimization operations"""
    cache_service_url: str = "http://localhost:8044"
    prometheus_url: str = "http://localhost:9090"
    optimization_interval_minutes: int = 15
    min_samples_for_optimization: int = 1000
    target_hit_rate: float = 0.85
    target_response_time_ms: float = 10.0
    memory_utilization_threshold: float = 0.85
    ml_accuracy_threshold: float = 0.8

@dataclass
class PerformanceMetrics:
    """Current performance metrics"""
    cache_hit_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    memory_utilization: Dict[str, float] = None
    ml_prediction_accuracy: float = 0.0
    request_rate: float = 0.0
    error_rate: float = 0.0
    tier_hit_rates: Dict[str, float] = None

    def __post_init__(self):
        if self.memory_utilization is None:
            self.memory_utilization = {"hot": 0.0, "warm": 0.0, "cold": 0.0}
        if self.tier_hit_rates is None:
            self.tier_hit_rates = {"hot": 0.0, "warm": 0.0, "cold": 0.0}

@dataclass
class OptimizationResult:
    """Results of optimization operations"""
    timestamp: str
    optimizations_applied: List[str]
    metrics_before: PerformanceMetrics
    metrics_after: Optional[PerformanceMetrics] = None
    improvement_percentage: float = 0.0
    success: bool = False

class CachePerformanceOptimizer:
    """Main optimization engine for predictive cache"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.optimization_history: List[OptimizationResult] = []

    async def setup(self):
        """Initialize optimizer"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

    async def collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics from cache service and Prometheus"""
        print("📊 Collecting performance metrics...")

        # Get cache service metrics
        cache_metrics = await self._get_cache_service_metrics()

        # Get Prometheus metrics (if available)
        prometheus_metrics = await self._get_prometheus_metrics()

        # Combine metrics
        metrics = PerformanceMetrics(
            cache_hit_rate=cache_metrics.get("cache_hit_rate", 0.0),
            avg_response_time_ms=cache_metrics.get("avg_response_time_ms", 0.0),
            p95_response_time_ms=cache_metrics.get("p95_response_time_ms", 0.0),
            memory_utilization=cache_metrics.get("memory_utilization", {}),
            ml_prediction_accuracy=cache_metrics.get("ml_prediction_accuracy", 0.0),
            request_rate=prometheus_metrics.get("request_rate", 0.0),
            error_rate=prometheus_metrics.get("error_rate", 0.0),
            tier_hit_rates=cache_metrics.get("tier_hit_rates", {})
        )

        return metrics

    async def _get_cache_service_metrics(self) -> Dict[str, Any]:
        """Get metrics from cache service"""
        try:
            async with self.session.get(f"{self.config.cache_service_url}/stats") as response:
                if response.status == 200:
                    return await response.json()
                return {}
        except Exception as e:
            print(f"⚠️ Failed to get cache service metrics: {e}")
            return {}

    async def _get_prometheus_metrics(self) -> Dict[str, Any]:
        """Get metrics from Prometheus (if available)"""
        try:
            # Query Prometheus for cache metrics
            queries = {
                "request_rate": "rate(bev_cache_requests_total[5m])",
                "error_rate": "rate(bev_cache_errors_total[5m]) / rate(bev_cache_requests_total[5m])"
            }

            metrics = {}
            for metric_name, query in queries.items():
                url = f"{self.config.prometheus_url}/api/v1/query"
                params = {"query": query}

                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success" and data.get("data", {}).get("result"):
                            result = data["data"]["result"][0]
                            metrics[metric_name] = float(result["value"][1])

            return metrics

        except Exception as e:
            print(f"⚠️ Failed to get Prometheus metrics: {e}")
            return {}

    async def analyze_optimization_opportunities(self, metrics: PerformanceMetrics) -> List[str]:
        """Analyze current metrics and identify optimization opportunities"""
        opportunities = []

        # Cache hit rate optimization
        if metrics.cache_hit_rate < self.config.target_hit_rate:
            if metrics.ml_prediction_accuracy < self.config.ml_accuracy_threshold:
                opportunities.append("retrain_ml_models")

            if metrics.tier_hit_rates.get("hot", 0) < 0.9:
                opportunities.append("optimize_hot_tier_allocation")

            opportunities.append("optimize_cache_warming")

        # Response time optimization
        if metrics.avg_response_time_ms > self.config.target_response_time_ms:
            if metrics.memory_utilization.get("hot", 0) > 0.9:
                opportunities.append("increase_hot_tier_size")

            if metrics.tier_hit_rates.get("warm", 0) < 0.7:
                opportunities.append("optimize_warm_tier_strategy")

            opportunities.append("optimize_eviction_policy")

        # Memory optimization
        total_memory_util = sum(metrics.memory_utilization.values()) / len(metrics.memory_utilization)
        if total_memory_util > self.config.memory_utilization_threshold:
            opportunities.append("optimize_memory_allocation")

        # Error rate optimization
        if metrics.error_rate > 0.01:  # >1% error rate
            opportunities.append("optimize_error_handling")

        return opportunities

    async def apply_optimizations(self, opportunities: List[str], metrics: PerformanceMetrics) -> OptimizationResult:
        """Apply identified optimizations"""
        print(f"🔧 Applying {len(opportunities)} optimizations...")

        result = OptimizationResult(
            timestamp=datetime.now().isoformat(),
            optimizations_applied=opportunities,
            metrics_before=metrics
        )

        applied_optimizations = []

        for optimization in opportunities:
            try:
                if await self._apply_single_optimization(optimization, metrics):
                    applied_optimizations.append(optimization)
                    print(f"✅ Applied: {optimization}")
                else:
                    print(f"❌ Failed: {optimization}")
            except Exception as e:
                print(f"❌ Error applying {optimization}: {e}")

        result.optimizations_applied = applied_optimizations
        result.success = len(applied_optimizations) > 0

        return result

    async def _apply_single_optimization(self, optimization: str, metrics: PerformanceMetrics) -> bool:
        """Apply a single optimization"""
        optimization_methods = {
            "retrain_ml_models": self._retrain_ml_models,
            "optimize_hot_tier_allocation": self._optimize_hot_tier_allocation,
            "optimize_cache_warming": self._optimize_cache_warming,
            "increase_hot_tier_size": self._increase_hot_tier_size,
            "optimize_warm_tier_strategy": self._optimize_warm_tier_strategy,
            "optimize_eviction_policy": self._optimize_eviction_policy,
            "optimize_memory_allocation": self._optimize_memory_allocation,
            "optimize_error_handling": self._optimize_error_handling
        }

        method = optimization_methods.get(optimization)
        if method:
            return await method(metrics)
        return False

    async def _retrain_ml_models(self, metrics: PerformanceMetrics) -> bool:
        """Trigger ML model retraining"""
        try:
            async with self.session.post(f"{self.config.cache_service_url}/admin/retrain") as response:
                return response.status == 200
        except Exception:
            return False

    async def _optimize_hot_tier_allocation(self, metrics: PerformanceMetrics) -> bool:
        """Optimize hot tier allocation strategy"""
        try:
            # Calculate optimal allocation based on access patterns
            optimization_params = {
                "strategy": "ml_guided",
                "target_hit_rate": 0.95,
                "eviction_policy": "adaptive"
            }

            async with self.session.post(
                f"{self.config.cache_service_url}/admin/optimize_tier",
                json={"tier": "hot", "params": optimization_params}
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def _optimize_cache_warming(self, metrics: PerformanceMetrics) -> bool:
        """Optimize cache warming strategy"""
        try:
            # Determine best warming strategy based on current performance
            if metrics.tier_hit_rates.get("hot", 0) < 0.8:
                strategy = "user_based"
            elif metrics.avg_response_time_ms > 15:
                strategy = "popularity_based"
            else:
                strategy = "temporal_based"

            async with self.session.post(
                f"{self.config.cache_service_url}/warm",
                json={"strategy": strategy, "intensity": "high"}
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def _increase_hot_tier_size(self, metrics: PerformanceMetrics) -> bool:
        """Increase hot tier size if memory allows"""
        try:
            # Calculate new size (increase by 25% if possible)
            current_utilization = metrics.memory_utilization.get("hot", 0)
            if current_utilization > 0.9:
                new_size_factor = 1.25
            else:
                new_size_factor = 1.1

            async with self.session.post(
                f"{self.config.cache_service_url}/admin/resize_tier",
                json={"tier": "hot", "size_factor": new_size_factor}
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def _optimize_warm_tier_strategy(self, metrics: PerformanceMetrics) -> bool:
        """Optimize warm tier strategy"""
        try:
            # Switch to ARC if not already using it
            async with self.session.post(
                f"{self.config.cache_service_url}/admin/set_eviction_policy",
                json={"tier": "warm", "policy": "arc"}
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def _optimize_eviction_policy(self, metrics: PerformanceMetrics) -> bool:
        """Optimize eviction policy based on workload"""
        try:
            # Choose optimal eviction policy
            if metrics.request_rate > 1000:  # High throughput
                policy = "ml_adaptive"
            elif metrics.avg_response_time_ms > 20:  # High latency
                policy = "lru"
            else:
                policy = "arc"

            async with self.session.post(
                f"{self.config.cache_service_url}/admin/set_eviction_policy",
                json={"policy": policy}
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def _optimize_memory_allocation(self, metrics: PerformanceMetrics) -> bool:
        """Optimize memory allocation across tiers"""
        try:
            # Rebalance tier sizes based on hit rates and utilization
            total_util = sum(metrics.memory_utilization.values())
            hot_hit_rate = metrics.tier_hit_rates.get("hot", 0)
            warm_hit_rate = metrics.tier_hit_rates.get("warm", 0)

            # If hot tier hit rate is high, allocate more memory to it
            if hot_hit_rate > 0.9 and metrics.memory_utilization.get("hot", 0) > 0.8:
                reallocation = {"hot": 1.2, "warm": 0.9}
            elif warm_hit_rate < 0.5:
                reallocation = {"hot": 1.1, "warm": 1.2}
            else:
                return True  # No reallocation needed

            async with self.session.post(
                f"{self.config.cache_service_url}/admin/reallocate_memory",
                json={"allocation": reallocation}
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def _optimize_error_handling(self, metrics: PerformanceMetrics) -> bool:
        """Optimize error handling and circuit breaker settings"""
        try:
            # Adjust circuit breaker thresholds based on error rate
            if metrics.error_rate > 0.05:  # >5% error rate
                threshold_adjustment = {"failure_threshold": 3, "timeout": 30}
            else:
                threshold_adjustment = {"failure_threshold": 5, "timeout": 60}

            async with self.session.post(
                f"{self.config.cache_service_url}/admin/configure_circuit_breaker",
                json=threshold_adjustment
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def measure_optimization_impact(self, result: OptimizationResult) -> OptimizationResult:
        """Measure the impact of applied optimizations"""
        print("📈 Measuring optimization impact...")

        # Wait for optimizations to take effect
        await asyncio.sleep(60)

        # Collect new metrics
        new_metrics = await self.collect_performance_metrics()
        result.metrics_after = new_metrics

        # Calculate improvement
        if result.metrics_before.cache_hit_rate > 0:
            hit_rate_improvement = (
                (new_metrics.cache_hit_rate - result.metrics_before.cache_hit_rate) /
                result.metrics_before.cache_hit_rate
            ) * 100

            response_time_improvement = (
                (result.metrics_before.avg_response_time_ms - new_metrics.avg_response_time_ms) /
                result.metrics_before.avg_response_time_ms
            ) * 100

            # Overall improvement score
            result.improvement_percentage = (hit_rate_improvement + response_time_improvement) / 2

        return result

    async def run_continuous_optimization(self):
        """Run continuous optimization loop"""
        print("🔄 Starting continuous cache optimization...")

        iteration = 0
        while True:
            try:
                iteration += 1
                print(f"\n{'='*60}")
                print(f"OPTIMIZATION ITERATION {iteration}")
                print(f"{'='*60}")

                # Collect current metrics
                current_metrics = await self.collect_performance_metrics()
                print(f"Current hit rate: {current_metrics.cache_hit_rate:.2%}")
                print(f"Current response time: {current_metrics.avg_response_time_ms:.2f}ms")

                # Check if optimization is needed
                needs_optimization = (
                    current_metrics.cache_hit_rate < self.config.target_hit_rate or
                    current_metrics.avg_response_time_ms > self.config.target_response_time_ms or
                    current_metrics.ml_prediction_accuracy < self.config.ml_accuracy_threshold
                )

                if not needs_optimization:
                    print("✅ Performance targets met, no optimization needed")
                else:
                    # Analyze and apply optimizations
                    opportunities = await self.analyze_optimization_opportunities(current_metrics)

                    if opportunities:
                        optimization_result = await self.apply_optimizations(opportunities, current_metrics)
                        optimization_result = await self.measure_optimization_impact(optimization_result)

                        self.optimization_history.append(optimization_result)

                        if optimization_result.success:
                            print(f"✅ Optimization successful: {optimization_result.improvement_percentage:.1f}% improvement")
                        else:
                            print("❌ Optimization failed")
                    else:
                        print("ℹ️ No optimization opportunities identified")

                # Wait for next optimization cycle
                print(f"⏳ Waiting {self.config.optimization_interval_minutes} minutes for next cycle...")
                await asyncio.sleep(self.config.optimization_interval_minutes * 60)

            except KeyboardInterrupt:
                print("\n🛑 Optimization stopped by user")
                break
            except Exception as e:
                print(f"❌ Optimization cycle failed: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def generate_optimization_report(self) -> str:
        """Generate comprehensive optimization report"""
        if not self.optimization_history:
            return "No optimizations performed yet."

        report = []
        report.append("=" * 80)
        report.append("CACHE OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append(f"Total Optimization Cycles: {len(self.optimization_history)}")
        report.append(f"Report Generated: {datetime.now().isoformat()}")
        report.append("")

        # Summary statistics
        successful_optimizations = [r for r in self.optimization_history if r.success]
        if successful_optimizations:
            avg_improvement = statistics.mean([r.improvement_percentage for r in successful_optimizations])
            report.append("OPTIMIZATION SUMMARY")
            report.append("-" * 40)
            report.append(f"Successful Optimizations: {len(successful_optimizations)}")
            report.append(f"Average Improvement: {avg_improvement:.2f}%")
            report.append("")

        # Recent optimizations
        report.append("RECENT OPTIMIZATIONS")
        report.append("-" * 40)
        for i, result in enumerate(self.optimization_history[-5:], 1):
            report.append(f"{i}. {result.timestamp}")
            report.append(f"   Applied: {', '.join(result.optimizations_applied)}")
            report.append(f"   Improvement: {result.improvement_percentage:.2f}%")
            report.append(f"   Success: {'✅' if result.success else '❌'}")
            report.append("")

        return "\n".join(report)

async def main():
    """Main optimization script"""
    parser = argparse.ArgumentParser(description="Predictive Cache Performance Optimizer")
    parser.add_argument("--mode", choices=["single", "continuous"], default="single",
                       help="Optimization mode: single run or continuous")
    parser.add_argument("--cache-url", default="http://localhost:8044",
                       help="Cache service URL")
    parser.add_argument("--prometheus-url", default="http://localhost:9090",
                       help="Prometheus URL")
    parser.add_argument("--interval", type=int, default=15,
                       help="Optimization interval in minutes (continuous mode)")

    args = parser.parse_args()

    config = OptimizationConfig(
        cache_service_url=args.cache_url,
        prometheus_url=args.prometheus_url,
        optimization_interval_minutes=args.interval
    )

    optimizer = CachePerformanceOptimizer(config)

    try:
        await optimizer.setup()

        if args.mode == "single":
            print("🎯 Running single optimization cycle...")

            # Collect metrics
            metrics = await optimizer.collect_performance_metrics()

            # Analyze and apply optimizations
            opportunities = await optimizer.analyze_optimization_opportunities(metrics)

            if opportunities:
                result = await optimizer.apply_optimizations(opportunities, metrics)
                result = await optimizer.measure_optimization_impact(result)

                print("\n" + optimizer.generate_optimization_report())

                return 0 if result.success else 1
            else:
                print("✅ No optimizations needed")
                return 0

        else:  # continuous mode
            await optimizer.run_continuous_optimization()
            return 0

    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return 2
    finally:
        await optimizer.cleanup()

if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)