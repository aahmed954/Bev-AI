"""
BEV OSINT Framework - Monitoring API Service
Unified API for health monitoring, metrics collection, and alerting.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import asdict
import aiohttp
from aiohttp import web
import aioredis
import asyncpg
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from health_monitor import HealthMonitor, ServiceMetrics, HealthStatus
from metrics_collector import MetricsCollector, MetricValue
from alert_system import AlertSystem, Alert, AlertSeverity, AlertState


class MonitoringAPI:
    """
    Unified monitoring API that provides endpoints for health monitoring,
    metrics collection, and alert management.
    """

    def __init__(self):
        self.health_monitor = HealthMonitor()
        self.metrics_collector = MetricsCollector()
        self.alert_system = AlertSystem()

        self.app = web.Application()
        self.setup_routes()

        self.logger = logging.getLogger('monitoring_api')

    def setup_routes(self):
        """Setup API routes."""
        # Health monitoring endpoints
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/health/services', self.get_service_health)
        self.app.router.add_get('/health/services/{service_name}', self.get_service_health_detail)
        self.app.router.add_get('/health/summary', self.get_health_summary)

        # Metrics endpoints
        self.app.router.add_get('/metrics', self.get_prometheus_metrics)
        self.app.router.add_get('/metrics/services', self.get_service_metrics)
        self.app.router.add_get('/metrics/services/{service_name}', self.get_service_metrics_detail)
        self.app.router.add_get('/metrics/history/{service_name}', self.get_service_metrics_history)
        self.app.router.add_get('/metrics/aggregated', self.get_aggregated_metrics)

        # Alert endpoints
        self.app.router.add_get('/alerts', self.get_active_alerts)
        self.app.router.add_get('/alerts/history', self.get_alert_history)
        self.app.router.add_post('/alerts/{alert_id}/acknowledge', self.acknowledge_alert)
        self.app.router.add_get('/alerts/stats', self.get_alert_stats)

        # Dashboard endpoints
        self.app.router.add_get('/dashboard/overview', self.get_dashboard_overview)
        self.app.router.add_get('/dashboard/services', self.get_services_dashboard)
        self.app.router.add_get('/dashboard/alerts', self.get_alerts_dashboard)

        # System endpoints
        self.app.router.add_get('/status', self.get_system_status)
        self.app.router.add_get('/config', self.get_configuration)
        self.app.router.add_post('/config/reload', self.reload_configuration)

    async def initialize(self):
        """Initialize all monitoring components."""
        try:
            await self.health_monitor.initialize()
            await self.metrics_collector.initialize()
            await self.alert_system.initialize()

            # Start background tasks
            await self.health_monitor.run_monitoring_loop()
            await self.metrics_collector.start_collection_tasks()
            await self.alert_system.start_alert_evaluation_loop()

            self.logger.info("Monitoring API initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring API: {e}")
            raise

    # Health monitoring endpoints
    async def health_check(self, request):
        """Basic health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "1.0.0",
            "components": {
                "health_monitor": "running",
                "metrics_collector": "running",
                "alert_system": "running"
            }
        })

    async def get_service_health(self, request):
        """Get health status of all services."""
        try:
            services = await self.health_monitor.get_all_service_metrics()

            response_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_services": len(services),
                "healthy_services": sum(1 for s in services.values() if s.status == HealthStatus.HEALTHY),
                "services": {}
            }

            for service_name, metrics in services.items():
                response_data["services"][service_name] = {
                    "status": metrics.status.value,
                    "response_time": metrics.response_time,
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "last_check": metrics.last_check.isoformat(),
                    "uptime": metrics.uptime
                }

            return web.json_response(response_data)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get service health: {e}"},
                status=500
            )

    async def get_service_health_detail(self, request):
        """Get detailed health information for a specific service."""
        service_name = request.match_info['service_name']

        try:
            metrics = await self.health_monitor.get_service_metrics(service_name)
            if not metrics:
                return web.json_response(
                    {"error": f"Service {service_name} not found"},
                    status=404
                )

            # Get historical data
            history = await self.health_monitor.get_service_history(service_name, hours=24)

            response_data = {
                "service_name": service_name,
                "current_status": asdict(metrics),
                "history_24h": [asdict(h) for h in history[-100:]],  # Last 100 entries
                "summary": {
                    "avg_response_time": sum(h.response_time for h in history) / len(history) if history else 0,
                    "avg_cpu_usage": sum(h.cpu_usage for h in history) / len(history) if history else 0,
                    "avg_memory_usage": sum(h.memory_usage for h in history) / len(history) if history else 0,
                    "uptime_percentage": sum(1 for h in history if h.status == HealthStatus.HEALTHY) / len(history) * 100 if history else 0
                }
            }

            return web.json_response(response_data)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get service health detail: {e}"},
                status=500
            )

    async def get_health_summary(self, request):
        """Get overall health summary."""
        try:
            services = await self.health_monitor.get_all_service_metrics()
            alerts = await self.alert_system.get_active_alerts()

            # Calculate statistics
            total_services = len(services)
            healthy_services = sum(1 for s in services.values() if s.status == HealthStatus.HEALTHY)
            degraded_services = sum(1 for s in services.values() if s.status == HealthStatus.DEGRADED)
            unhealthy_services = sum(1 for s in services.values() if s.status == HealthStatus.UNHEALTHY)

            critical_alerts = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
            warning_alerts = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)

            # Calculate overall health score
            health_score = (healthy_services / total_services * 100) if total_services > 0 else 0

            response_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "overall_health": {
                    "score": round(health_score, 2),
                    "status": "healthy" if health_score > 90 else "degraded" if health_score > 70 else "unhealthy"
                },
                "services": {
                    "total": total_services,
                    "healthy": healthy_services,
                    "degraded": degraded_services,
                    "unhealthy": unhealthy_services
                },
                "alerts": {
                    "total": len(alerts),
                    "critical": critical_alerts,
                    "warning": warning_alerts
                },
                "performance": {
                    "avg_response_time": sum(s.response_time for s in services.values()) / len(services) if services else 0,
                    "avg_cpu_usage": sum(s.cpu_usage for s in services.values()) / len(services) if services else 0,
                    "avg_memory_usage": sum(s.memory_usage for s in services.values()) / len(services) if services else 0
                }
            }

            return web.json_response(response_data)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get health summary: {e}"},
                status=500
            )

    # Metrics endpoints
    async def get_prometheus_metrics(self, request):
        """Get Prometheus metrics from all monitoring components."""
        try:
            metrics_data = generate_latest()
            return web.Response(
                text=metrics_data.decode('utf-8'),
                content_type=CONTENT_TYPE_LATEST
            )
        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get Prometheus metrics: {e}"},
                status=500
            )

    async def get_service_metrics(self, request):
        """Get current metrics for all services."""
        try:
            services = await self.health_monitor.get_all_service_metrics()

            response_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "services": {}
            }

            for service_name, metrics in services.items():
                response_data["services"][service_name] = {
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "response_time": metrics.response_time,
                    "error_count": metrics.error_count,
                    "success_count": metrics.success_count,
                    "uptime": metrics.uptime,
                    "disk_usage": metrics.disk_usage,
                    "network_io": metrics.network_io,
                    "custom_metrics": metrics.custom_metrics
                }

            return web.json_response(response_data)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get service metrics: {e}"},
                status=500
            )

    async def get_service_metrics_detail(self, request):
        """Get detailed metrics for a specific service."""
        service_name = request.match_info['service_name']
        hours = int(request.query.get('hours', 1))

        try:
            # Get current metrics
            current_metrics = await self.health_monitor.get_service_metrics(service_name)
            if not current_metrics:
                return web.json_response(
                    {"error": f"Service {service_name} not found"},
                    status=404
                )

            # Get historical metrics
            history = await self.metrics_collector.get_metric_history(
                "service_response_time", service_name, hours
            )

            response_data = {
                "service_name": service_name,
                "current_metrics": asdict(current_metrics),
                "historical_data": {
                    "response_time": [asdict(h) for h in history],
                    "period_hours": hours
                }
            }

            return web.json_response(response_data)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get service metrics detail: {e}"},
                status=500
            )

    async def get_service_metrics_history(self, request):
        """Get historical metrics for a service."""
        service_name = request.match_info['service_name']
        metric_name = request.query.get('metric', 'service_response_time')
        hours = int(request.query.get('hours', 24))

        try:
            history = await self.metrics_collector.get_metric_history(
                metric_name, service_name, hours
            )

            response_data = {
                "service_name": service_name,
                "metric_name": metric_name,
                "period_hours": hours,
                "data_points": len(history),
                "history": [asdict(h) for h in history]
            }

            return web.json_response(response_data)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get metrics history: {e}"},
                status=500
            )

    async def get_aggregated_metrics(self, request):
        """Get aggregated metrics across all services."""
        metric_name = request.query.get('metric', 'service_response_time')
        time_window = int(request.query.get('window', 3600))  # 1 hour default

        try:
            aggregated = await self.metrics_collector.aggregate_metrics(metric_name, time_window)

            if not aggregated:
                return web.json_response(
                    {"error": f"No aggregated data found for metric {metric_name}"},
                    status=404
                )

            return web.json_response(asdict(aggregated))

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get aggregated metrics: {e}"},
                status=500
            )

    # Alert endpoints
    async def get_active_alerts(self, request):
        """Get currently active alerts."""
        service_name = request.query.get('service')
        severity = request.query.get('severity')

        try:
            severity_filter = AlertSeverity(severity) if severity else None
            alerts = await self.alert_system.get_active_alerts(service_name, severity_filter)

            response_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_alerts": len(alerts),
                "alerts": [asdict(alert) for alert in alerts]
            }

            return web.json_response(response_data)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get active alerts: {e}"},
                status=500
            )

    async def get_alert_history(self, request):
        """Get alert history."""
        hours = int(request.query.get('hours', 24))
        service_name = request.query.get('service')

        try:
            history = await self.alert_system.get_alert_history(hours, service_name)

            response_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "period_hours": hours,
                "total_alerts": len(history),
                "alerts": [asdict(alert) for alert in history]
            }

            return web.json_response(response_data)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get alert history: {e}"},
                status=500
            )

    async def acknowledge_alert(self, request):
        """Acknowledge an active alert."""
        alert_id = request.match_info['alert_id']

        try:
            data = await request.json()
            acknowledged_by = data.get('acknowledged_by', 'unknown')

            success = await self.alert_system.acknowledge_alert(alert_id, acknowledged_by)

            if success:
                return web.json_response({
                    "status": "success",
                    "message": f"Alert {alert_id} acknowledged by {acknowledged_by}"
                })
            else:
                return web.json_response(
                    {"error": f"Alert {alert_id} not found or already acknowledged"},
                    status=404
                )

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to acknowledge alert: {e}"},
                status=500
            )

    async def get_alert_stats(self, request):
        """Get alert statistics."""
        try:
            active_alerts = await self.alert_system.get_active_alerts()
            history_24h = await self.alert_system.get_alert_history(24)

            # Calculate statistics
            stats = {
                "active": {
                    "total": len(active_alerts),
                    "critical": sum(1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL),
                    "warning": sum(1 for a in active_alerts if a.severity == AlertSeverity.WARNING),
                    "info": sum(1 for a in active_alerts if a.severity == AlertSeverity.INFO)
                },
                "last_24h": {
                    "total": len(history_24h),
                    "resolved": sum(1 for a in history_24h if a.state == AlertState.RESOLVED),
                    "acknowledged": sum(1 for a in history_24h if a.state == AlertState.ACKNOWLEDGED)
                }
            }

            return web.json_response(stats)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get alert statistics: {e}"},
                status=500
            )

    # Dashboard endpoints
    async def get_dashboard_overview(self, request):
        """Get overview dashboard data."""
        try:
            # Get data from all components
            services = await self.health_monitor.get_all_service_metrics()
            alerts = await self.alert_system.get_active_alerts()

            # Prepare dashboard data
            dashboard_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": {
                    "total_services": len(services),
                    "healthy_services": sum(1 for s in services.values() if s.status == HealthStatus.HEALTHY),
                    "total_alerts": len(alerts),
                    "critical_alerts": sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
                },
                "top_issues": [
                    {
                        "service": name,
                        "issue": "High CPU usage" if metrics.cpu_usage > 85 else "High memory usage" if metrics.memory_usage > 90 else "Slow response",
                        "value": max(metrics.cpu_usage, metrics.memory_usage, metrics.response_time),
                        "severity": "critical" if max(metrics.cpu_usage, metrics.memory_usage) > 95 else "warning"
                    }
                    for name, metrics in services.items()
                    if metrics.cpu_usage > 85 or metrics.memory_usage > 90 or metrics.response_time > 5
                ][:10],  # Top 10 issues
                "service_status": {
                    name: {
                        "status": metrics.status.value,
                        "response_time": metrics.response_time,
                        "cpu_usage": metrics.cpu_usage,
                        "memory_usage": metrics.memory_usage
                    }
                    for name, metrics in services.items()
                }
            }

            return web.json_response(dashboard_data)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get dashboard overview: {e}"},
                status=500
            )

    async def get_services_dashboard(self, request):
        """Get services dashboard data."""
        try:
            services = await self.health_monitor.get_all_service_metrics()

            # Group services by type
            service_groups = {}
            for name, metrics in services.items():
                service_type = self.health_monitor.services.get(name, {}).get('type', 'unknown')
                if service_type not in service_groups:
                    service_groups[service_type] = []

                service_groups[service_type].append({
                    "name": name,
                    "status": metrics.status.value,
                    "response_time": metrics.response_time,
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage,
                    "uptime": metrics.uptime
                })

            return web.json_response({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service_groups": service_groups
            })

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get services dashboard: {e}"},
                status=500
            )

    async def get_alerts_dashboard(self, request):
        """Get alerts dashboard data."""
        try:
            active_alerts = await self.alert_system.get_active_alerts()
            recent_history = await self.alert_system.get_alert_history(6)  # Last 6 hours

            # Group alerts by service and severity
            alerts_by_service = {}
            alerts_by_severity = {"critical": 0, "warning": 0, "info": 0}

            for alert in active_alerts:
                if alert.service_name not in alerts_by_service:
                    alerts_by_service[alert.service_name] = []
                alerts_by_service[alert.service_name].append(asdict(alert))
                alerts_by_severity[alert.severity.value] += 1

            return web.json_response({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "active_alerts": {
                    "total": len(active_alerts),
                    "by_severity": alerts_by_severity,
                    "by_service": alerts_by_service
                },
                "recent_history": [asdict(alert) for alert in recent_history[:50]]  # Last 50
            })

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get alerts dashboard: {e}"},
                status=500
            )

    # System endpoints
    async def get_system_status(self, request):
        """Get overall system status."""
        try:
            status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "health_monitor": {
                        "status": "running",
                        "services_monitored": len(self.health_monitor.services),
                        "last_check": self.health_monitor.collection_stats.get("last_collection")
                    },
                    "metrics_collector": {
                        "status": "running",
                        "metrics_collected": self.metrics_collector.collection_stats.get("metrics_collected", 0),
                        "collection_errors": self.metrics_collector.collection_stats.get("collection_errors", 0)
                    },
                    "alert_system": {
                        "status": "running",
                        "active_alerts": len(self.alert_system.active_alerts),
                        "alert_rules": len(self.alert_system.alert_rules)
                    }
                }
            }

            return web.json_response(status)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get system status: {e}"},
                status=500
            )

    async def get_configuration(self, request):
        """Get current configuration."""
        try:
            config = {
                "health_monitor": self.health_monitor.config,
                "metrics_collector": self.metrics_collector.config,
                "alert_system": self.alert_system.config
            }

            return web.json_response(config)

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to get configuration: {e}"},
                status=500
            )

    async def reload_configuration(self, request):
        """Reload configuration for all components."""
        try:
            # This would reload configuration from files
            # Implementation depends on specific requirements

            return web.json_response({
                "status": "success",
                "message": "Configuration reloaded successfully",
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            return web.json_response(
                {"error": f"Failed to reload configuration: {e}"},
                status=500
            )

    async def start_server(self, host='0.0.0.0', port=8000):
        """Start the monitoring API server."""
        await self.initialize()

        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, host, port)
        await site.start()

        self.logger.info(f"Monitoring API server started on {host}:{port}")

        try:
            # Keep the server running
            while True:
                await asyncio.sleep(3600)
        except KeyboardInterrupt:
            self.logger.info("Shutting down monitoring API server")
        finally:
            await runner.cleanup()
            await self.shutdown()

    async def shutdown(self):
        """Graceful shutdown of all monitoring components."""
        await self.health_monitor.shutdown()
        await self.metrics_collector.shutdown()
        await self.alert_system.shutdown()


async def main():
    """Main entry point for monitoring API."""
    api = MonitoringAPI()
    await api.start_server()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())