#!/usr/bin/env python3
"""
BEV OSINT Framework - Predictive Cache Performance Monitor

Real-time monitoring script for predictive cache performance with alerting,
trend analysis, and automated health checks.

Features:
- Real-time performance monitoring
- Threshold-based alerting
- Performance trend analysis
- Health status reporting
- Integration with Prometheus metrics
- Automated remediation suggestions
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
import aiohttp
import argparse
from collections import deque
import signal
import sys

@dataclass
class AlertThresholds:
    """Alert threshold configuration"""
    cache_hit_rate_critical: float = 0.6
    cache_hit_rate_warning: float = 0.75
    response_time_critical_ms: float = 50.0
    response_time_warning_ms: float = 25.0
    ml_accuracy_critical: float = 0.6
    ml_accuracy_warning: float = 0.75
    memory_utilization_critical: float = 0.95
    memory_utilization_warning: float = 0.85
    error_rate_critical: float = 0.05
    error_rate_warning: float = 0.02

@dataclass
class MonitoringConfig:
    """Configuration for monitoring operations"""
    cache_service_url: str = "http://localhost:8044"
    prometheus_url: str = "http://localhost:9090"
    monitoring_interval_seconds: int = 30
    trend_analysis_window_minutes: int = 60
    alert_cooldown_minutes: int = 5
    enable_notifications: bool = True
    log_file: Optional[str] = None

@dataclass
class PerformanceSnapshot:
    """Single point-in-time performance measurement"""
    timestamp: datetime
    cache_hit_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    memory_utilization: Dict[str, float] = field(default_factory=dict)
    ml_prediction_accuracy: float = 0.0
    request_rate: float = 0.0
    error_rate: float = 0.0
    tier_hit_rates: Dict[str, float] = field(default_factory=dict)
    active_connections: int = 0
    cache_warming_tasks: int = 0

@dataclass
class Alert:
    """Performance alert"""
    timestamp: datetime
    severity: str  # critical, warning, info
    component: str
    message: str
    metric_value: float
    threshold: float
    suggested_action: str

@dataclass
class TrendAnalysis:
    """Performance trend analysis results"""
    metric_name: str
    trend_direction: str  # improving, degrading, stable
    trend_strength: float  # 0.0 to 1.0
    slope: float
    r_squared: float
    forecast_1h: float
    recommendation: str

class CachePerformanceMonitor:
    """Real-time cache performance monitoring system"""

    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.thresholds = AlertThresholds()
        self.session: Optional[aiohttp.ClientSession] = None
        self.performance_history: deque = deque(maxlen=1000)  # Keep last 1000 measurements
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_alert_times: Dict[str, datetime] = {}
        self.running = True

    async def setup(self):
        """Initialize monitoring system"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print("🖥️ Cache Performance Monitor initialized")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.running = False
        print("\n🛑 Shutdown signal received, stopping monitor...")

    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()

    async def collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance metrics"""
        # Get cache service metrics
        cache_metrics = await self._get_cache_service_metrics()

        # Get Prometheus metrics
        prometheus_metrics = await self._get_prometheus_metrics()

        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            cache_hit_rate=cache_metrics.get("cache_hit_rate", 0.0),
            avg_response_time_ms=cache_metrics.get("avg_response_time_ms", 0.0),
            p95_response_time_ms=cache_metrics.get("p95_response_time_ms", 0.0),
            p99_response_time_ms=cache_metrics.get("p99_response_time_ms", 0.0),
            memory_utilization=cache_metrics.get("memory_utilization", {}),
            ml_prediction_accuracy=cache_metrics.get("ml_prediction_accuracy", 0.0),
            request_rate=prometheus_metrics.get("request_rate", 0.0),
            error_rate=prometheus_metrics.get("error_rate", 0.0),
            tier_hit_rates=cache_metrics.get("tier_hit_rates", {}),
            active_connections=cache_metrics.get("active_connections", 0),
            cache_warming_tasks=cache_metrics.get("cache_warming_tasks", 0)
        )

        return snapshot

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
        """Get metrics from Prometheus"""
        try:
            queries = {
                "request_rate": "rate(bev_cache_requests_total[1m])",
                "error_rate": "rate(bev_cache_errors_total[1m]) / rate(bev_cache_requests_total[1m])",
                "cache_hit_rate": "bev_cache_hit_rate",
                "response_time": "bev_cache_response_time_seconds"
            }

            metrics = {}
            for metric_name, query in queries.items():
                try:
                    url = f"{self.config.prometheus_url}/api/v1/query"
                    params = {"query": query}

                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("status") == "success" and data.get("data", {}).get("result"):
                                result = data["data"]["result"][0]
                                metrics[metric_name] = float(result["value"][1])
                except Exception:
                    continue

            return metrics

        except Exception as e:
            print(f"⚠️ Failed to get Prometheus metrics: {e}")
            return {}

    def analyze_performance_alerts(self, snapshot: PerformanceSnapshot) -> List[Alert]:
        """Analyze current performance and generate alerts"""
        alerts = []

        # Cache hit rate alerts
        if snapshot.cache_hit_rate < self.thresholds.cache_hit_rate_critical:
            alerts.append(Alert(
                timestamp=snapshot.timestamp,
                severity="critical",
                component="cache_hit_rate",
                message=f"Cache hit rate critically low: {snapshot.cache_hit_rate:.2%}",
                metric_value=snapshot.cache_hit_rate,
                threshold=self.thresholds.cache_hit_rate_critical,
                suggested_action="Check ML model accuracy, optimize warming strategy, review tier allocation"
            ))
        elif snapshot.cache_hit_rate < self.thresholds.cache_hit_rate_warning:
            alerts.append(Alert(
                timestamp=snapshot.timestamp,
                severity="warning",
                component="cache_hit_rate",
                message=f"Cache hit rate below target: {snapshot.cache_hit_rate:.2%}",
                metric_value=snapshot.cache_hit_rate,
                threshold=self.thresholds.cache_hit_rate_warning,
                suggested_action="Monitor trend, consider cache warming optimization"
            ))

        # Response time alerts
        if snapshot.avg_response_time_ms > self.thresholds.response_time_critical_ms:
            alerts.append(Alert(
                timestamp=snapshot.timestamp,
                severity="critical",
                component="response_time",
                message=f"Response time critically high: {snapshot.avg_response_time_ms:.2f}ms",
                metric_value=snapshot.avg_response_time_ms,
                threshold=self.thresholds.response_time_critical_ms,
                suggested_action="Check memory utilization, optimize tier allocation, review eviction policy"
            ))
        elif snapshot.avg_response_time_ms > self.thresholds.response_time_warning_ms:
            alerts.append(Alert(
                timestamp=snapshot.timestamp,
                severity="warning",
                component="response_time",
                message=f"Response time elevated: {snapshot.avg_response_time_ms:.2f}ms",
                metric_value=snapshot.avg_response_time_ms,
                threshold=self.thresholds.response_time_warning_ms,
                suggested_action="Monitor trend, check for memory pressure"
            ))

        # ML accuracy alerts
        if snapshot.ml_prediction_accuracy < self.thresholds.ml_accuracy_critical:
            alerts.append(Alert(
                timestamp=snapshot.timestamp,
                severity="critical",
                component="ml_accuracy",
                message=f"ML prediction accuracy critically low: {snapshot.ml_prediction_accuracy:.2%}",
                metric_value=snapshot.ml_prediction_accuracy,
                threshold=self.thresholds.ml_accuracy_critical,
                suggested_action="Retrain ML models immediately, check training data quality"
            ))
        elif snapshot.ml_prediction_accuracy < self.thresholds.ml_accuracy_warning:
            alerts.append(Alert(
                timestamp=snapshot.timestamp,
                severity="warning",
                component="ml_accuracy",
                message=f"ML prediction accuracy below target: {snapshot.ml_prediction_accuracy:.2%}",
                metric_value=snapshot.ml_prediction_accuracy,
                threshold=self.thresholds.ml_accuracy_warning,
                suggested_action="Schedule ML model retraining, review feature engineering"
            ))

        # Memory utilization alerts
        for tier, utilization in snapshot.memory_utilization.items():
            if utilization > self.thresholds.memory_utilization_critical:
                alerts.append(Alert(
                    timestamp=snapshot.timestamp,
                    severity="critical",
                    component=f"memory_{tier}",
                    message=f"{tier.title()} tier memory critically high: {utilization:.1%}",
                    metric_value=utilization,
                    threshold=self.thresholds.memory_utilization_critical,
                    suggested_action=f"Increase {tier} tier size or optimize eviction policy"
                ))
            elif utilization > self.thresholds.memory_utilization_warning:
                alerts.append(Alert(
                    timestamp=snapshot.timestamp,
                    severity="warning",
                    component=f"memory_{tier}",
                    message=f"{tier.title()} tier memory high: {utilization:.1%}",
                    metric_value=utilization,
                    threshold=self.thresholds.memory_utilization_warning,
                    suggested_action=f"Monitor {tier} tier memory usage trend"
                ))

        # Error rate alerts
        if snapshot.error_rate > self.thresholds.error_rate_critical:
            alerts.append(Alert(
                timestamp=snapshot.timestamp,
                severity="critical",
                component="error_rate",
                message=f"Error rate critically high: {snapshot.error_rate:.2%}",
                metric_value=snapshot.error_rate,
                threshold=self.thresholds.error_rate_critical,
                suggested_action="Check service health, review error logs, verify Redis connectivity"
            ))
        elif snapshot.error_rate > self.thresholds.error_rate_warning:
            alerts.append(Alert(
                timestamp=snapshot.timestamp,
                severity="warning",
                component="error_rate",
                message=f"Error rate elevated: {snapshot.error_rate:.2%}",
                metric_value=snapshot.error_rate,
                threshold=self.thresholds.error_rate_warning,
                suggested_action="Monitor error trends, check for network issues"
            ))

        return alerts

    def process_alerts(self, new_alerts: List[Alert]):
        """Process new alerts with cooldown and deduplication"""
        for alert in new_alerts:
            alert_key = f"{alert.component}_{alert.severity}"
            now = datetime.now()

            # Check if alert is in cooldown period
            last_alert_time = self.last_alert_times.get(alert_key)
            if last_alert_time:
                time_since_last = (now - last_alert_time).total_seconds() / 60
                if time_since_last < self.config.alert_cooldown_minutes:
                    continue  # Skip alert due to cooldown

            # Add to active alerts and history
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            self.last_alert_times[alert_key] = now

            # Print alert
            severity_icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
            icon = severity_icon.get(alert.severity, "📊")
            print(f"\n{icon} ALERT [{alert.severity.upper()}] {alert.component}")
            print(f"   {alert.message}")
            print(f"   Suggested Action: {alert.suggested_action}")

    def analyze_trends(self) -> List[TrendAnalysis]:
        """Analyze performance trends over time"""
        if len(self.performance_history) < 10:
            return []

        trend_analyses = []

        # Analyze key metrics
        metrics_to_analyze = [
            ("cache_hit_rate", "higher is better"),
            ("avg_response_time_ms", "lower is better"),
            ("ml_prediction_accuracy", "higher is better"),
            ("error_rate", "lower is better")
        ]

        for metric_name, direction in metrics_to_analyze:
            try:
                # Extract metric values and timestamps
                values = []
                timestamps = []

                for snapshot in list(self.performance_history)[-60:]:  # Last 60 measurements
                    if hasattr(snapshot, metric_name):
                        value = getattr(snapshot, metric_name)
                        if value > 0:  # Skip zero values
                            values.append(value)
                            timestamps.append(snapshot.timestamp.timestamp())

                if len(values) < 5:
                    continue

                # Calculate trend
                trend_analysis = self._calculate_trend(metric_name, values, timestamps, direction)
                if trend_analysis:
                    trend_analyses.append(trend_analysis)

            except Exception as e:
                print(f"⚠️ Failed to analyze trend for {metric_name}: {e}")

        return trend_analyses

    def _calculate_trend(self, metric_name: str, values: List[float],
                        timestamps: List[float], direction: str) -> Optional[TrendAnalysis]:
        """Calculate trend statistics for a metric"""
        try:
            import numpy as np
            from sklearn.linear_model import LinearRegression

            # Prepare data
            X = np.array(timestamps).reshape(-1, 1)
            y = np.array(values)

            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)

            # Calculate R-squared
            r_squared = model.score(X, y)
            slope = model.coef_[0]

            # Determine trend direction
            if abs(slope) < 0.001:  # Very small slope
                trend_direction = "stable"
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = "increasing"
                trend_strength = min(abs(slope) * 1000, 1.0)
            else:
                trend_direction = "decreasing"
                trend_strength = min(abs(slope) * 1000, 1.0)

            # Forecast 1 hour ahead
            future_timestamp = timestamps[-1] + 3600  # 1 hour
            forecast_1h = model.predict([[future_timestamp]])[0]

            # Generate recommendation
            recommendation = self._generate_trend_recommendation(
                metric_name, trend_direction, trend_strength, direction, forecast_1h, values[-1]
            )

            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                slope=slope,
                r_squared=r_squared,
                forecast_1h=forecast_1h,
                recommendation=recommendation
            )

        except ImportError:
            # Fallback to simple trend calculation without sklearn
            if len(values) < 2:
                return None

            # Simple slope calculation
            slope = (values[-1] - values[0]) / (timestamps[-1] - timestamps[0])

            if abs(slope) < 0.001:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"

            trend_strength = min(abs(slope) * 1000, 1.0)
            forecast_1h = values[-1] + (slope * 3600)

            recommendation = self._generate_trend_recommendation(
                metric_name, trend_direction, trend_strength, direction, forecast_1h, values[-1]
            )

            return TrendAnalysis(
                metric_name=metric_name,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                slope=slope,
                r_squared=0.0,
                forecast_1h=forecast_1h,
                recommendation=recommendation
            )

        except Exception:
            return None

    def _generate_trend_recommendation(self, metric_name: str, trend_direction: str,
                                     trend_strength: float, preferred_direction: str,
                                     forecast: float, current: float) -> str:
        """Generate recommendation based on trend analysis"""
        if trend_direction == "stable":
            return "Metric is stable, continue monitoring"

        is_good_trend = (
            (preferred_direction == "higher is better" and trend_direction == "increasing") or
            (preferred_direction == "lower is better" and trend_direction == "decreasing")
        )

        if is_good_trend:
            if trend_strength > 0.5:
                return f"Excellent {trend_direction} trend, monitor to ensure sustainability"
            else:
                return f"Positive {trend_direction} trend, continue current optimization"
        else:
            if trend_strength > 0.7:
                return f"Critical: Strong negative trend, immediate intervention required"
            elif trend_strength > 0.3:
                return f"Warning: Negative trend detected, investigate and optimize"
            else:
                return f"Minor negative trend, monitor closely"

    def generate_status_report(self) -> str:
        """Generate comprehensive status report"""
        if not self.performance_history:
            return "No performance data available"

        latest = self.performance_history[-1]
        report = []

        report.append("=" * 80)
        report.append("PREDICTIVE CACHE PERFORMANCE STATUS")
        report.append("=" * 80)
        report.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Last Update: {latest.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Current Performance
        report.append("CURRENT PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Cache Hit Rate: {latest.cache_hit_rate:.2%} {'✅' if latest.cache_hit_rate >= 0.8 else '❌'}")
        report.append(f"Avg Response Time: {latest.avg_response_time_ms:.2f}ms {'✅' if latest.avg_response_time_ms <= 10 else '❌'}")
        report.append(f"P95 Response Time: {latest.p95_response_time_ms:.2f}ms")
        report.append(f"P99 Response Time: {latest.p99_response_time_ms:.2f}ms")
        report.append(f"ML Prediction Accuracy: {latest.ml_prediction_accuracy:.2%} {'✅' if latest.ml_prediction_accuracy >= 0.8 else '❌'}")
        report.append(f"Request Rate: {latest.request_rate:.1f} req/sec")
        report.append(f"Error Rate: {latest.error_rate:.2%} {'✅' if latest.error_rate <= 0.01 else '❌'}")
        report.append("")

        # Tier Performance
        report.append("TIER PERFORMANCE")
        report.append("-" * 40)
        for tier, hit_rate in latest.tier_hit_rates.items():
            target = {"hot": 0.9, "warm": 0.7, "cold": 0.5}.get(tier, 0.5)
            status = "✅" if hit_rate >= target else "❌"
            report.append(f"{tier.title()} Tier Hit Rate: {hit_rate:.2%} {status}")

        for tier, utilization in latest.memory_utilization.items():
            status = "✅" if utilization <= 0.85 else "⚠️" if utilization <= 0.95 else "❌"
            report.append(f"{tier.title()} Tier Memory: {utilization:.1%} {status}")
        report.append("")

        # Active Alerts
        if self.active_alerts:
            report.append("ACTIVE ALERTS")
            report.append("-" * 40)
            for alert_key, alert in self.active_alerts.items():
                severity_icon = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
                icon = severity_icon.get(alert.severity, "📊")
                report.append(f"{icon} {alert.message}")
            report.append("")

        # Trend Analysis
        trends = self.analyze_trends()
        if trends:
            report.append("PERFORMANCE TRENDS")
            report.append("-" * 40)
            for trend in trends:
                direction_icon = {"increasing": "📈", "decreasing": "📉", "stable": "📊"}
                icon = direction_icon.get(trend.trend_direction, "📊")
                report.append(f"{icon} {trend.metric_name}: {trend.trend_direction}")
                if trend.trend_strength > 0.3:
                    report.append(f"   Strength: {trend.trend_strength:.2f}")
                    report.append(f"   Recommendation: {trend.recommendation}")
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)

    async def run_monitoring_loop(self):
        """Main monitoring loop"""
        print("🚀 Starting cache performance monitoring...")
        print(f"Monitoring interval: {self.config.monitoring_interval_seconds} seconds")

        while self.running:
            try:
                # Collect performance snapshot
                snapshot = await self.collect_performance_snapshot()
                self.performance_history.append(snapshot)

                # Analyze for alerts
                new_alerts = self.analyze_performance_alerts(snapshot)
                if new_alerts:
                    self.process_alerts(new_alerts)

                # Print periodic status
                if len(self.performance_history) % 10 == 0:  # Every 10 measurements
                    print(f"\n📊 Status: Hit Rate {snapshot.cache_hit_rate:.2%} | "
                          f"Response Time {snapshot.avg_response_time_ms:.1f}ms | "
                          f"ML Accuracy {snapshot.ml_prediction_accuracy:.2%}")

                # Log to file if configured
                if self.config.log_file:
                    await self._log_snapshot(snapshot)

                # Wait for next cycle
                await asyncio.sleep(self.config.monitoring_interval_seconds)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Monitoring cycle failed: {e}")
                await asyncio.sleep(5)  # Wait before retrying

        print("🛑 Monitoring stopped")

    async def _log_snapshot(self, snapshot: PerformanceSnapshot):
        """Log performance snapshot to file"""
        try:
            log_entry = {
                "timestamp": snapshot.timestamp.isoformat(),
                "metrics": asdict(snapshot)
            }

            with open(self.config.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        except Exception as e:
            print(f"⚠️ Failed to log snapshot: {e}")

async def main():
    """Main monitoring script"""
    parser = argparse.ArgumentParser(description="Predictive Cache Performance Monitor")
    parser.add_argument("--cache-url", default="http://localhost:8044",
                       help="Cache service URL")
    parser.add_argument("--prometheus-url", default="http://localhost:9090",
                       help="Prometheus URL")
    parser.add_argument("--interval", type=int, default=30,
                       help="Monitoring interval in seconds")
    parser.add_argument("--log-file", help="Log file path for performance data")
    parser.add_argument("--report", action="store_true",
                       help="Generate single status report and exit")

    args = parser.parse_args()

    config = MonitoringConfig(
        cache_service_url=args.cache_url,
        prometheus_url=args.prometheus_url,
        monitoring_interval_seconds=args.interval,
        log_file=args.log_file
    )

    monitor = CachePerformanceMonitor(config)

    try:
        await monitor.setup()

        if args.report:
            # Single report mode
            snapshot = await monitor.collect_performance_snapshot()
            monitor.performance_history.append(snapshot)
            print(monitor.generate_status_report())
            return 0
        else:
            # Continuous monitoring mode
            await monitor.run_monitoring_loop()
            return 0

    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        return 1
    finally:
        await monitor.cleanup()

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))