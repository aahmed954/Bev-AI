import os
"""
BEV OSINT Framework - Health Monitoring System
Comprehensive health monitoring with 30-second intervals for all services.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import aioredis
import psutil
import docker
from prometheus_client import Gauge, Counter, Histogram, start_http_server
import yaml


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ServiceMetrics:
    """Comprehensive service metrics collection."""
    service_name: str
    status: HealthStatus
    response_time: float
    cpu_usage: float
    memory_usage: float
    error_count: int
    success_count: int
    last_check: datetime
    uptime: float
    disk_usage: Optional[float] = None
    network_io: Optional[Dict[str, float]] = None
    custom_metrics: Optional[Dict[str, Any]] = None


@dataclass
class HealthAlert:
    """Health alert configuration and tracking."""
    service_name: str
    alert_type: str
    severity: AlertSeverity
    threshold_value: float
    current_value: float
    message: str
    timestamp: datetime
    acknowledged: bool = False


class HealthMonitor:
    """
    Advanced health monitoring system for BEV OSINT framework.
    Monitors 50+ services with comprehensive metrics and alerting.
    """

    def __init__(self, config_path: str = "/app/config/health_monitor.yml"):
        self.config = self._load_config(config_path)
        self.services = self.config.get("services", {})
        self.alert_config = self.config.get("alerts", {})
        self.check_interval = self.config.get("check_interval", 30)

        # Monitoring state
        self.service_metrics: Dict[str, ServiceMetrics] = {}
        self.active_alerts: Dict[str, List[HealthAlert]] = {}
        self.service_history: Dict[str, List[ServiceMetrics]] = {}

        # External connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.docker_client = docker.from_env()
        self.session: Optional[aiohttp.ClientSession] = None

        # Prometheus metrics
        self._init_prometheus_metrics()

        # Logging setup
        self.logger = self._setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load health monitoring configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for health monitoring."""
        return {
            "services": {
                # Core infrastructure
                "postgres": {"type": "database", "port": 5432, "health_endpoint": None},
                "neo4j": {"type": "database", "port": 7474, "health_endpoint": "/"},
                "redis": {"type": "cache", "port": 6379, "health_endpoint": None},
                "elasticsearch": {"type": "search", "port": 9200, "health_endpoint": "/_cluster/health"},
                "influxdb": {"type": "timeseries", "port": 8086, "health_endpoint": "/health"},

                # Message queues
                "kafka-1": {"type": "queue", "port": 19092, "health_endpoint": None},
                "kafka-2": {"type": "queue", "port": 29092, "health_endpoint": None},
                "kafka-3": {"type": "queue", "port": 39092, "health_endpoint": None},
                "rabbitmq-1": {"type": "queue", "port": 15672, "health_endpoint": "/api/health/checks/alarms"},
                "rabbitmq-2": {"type": "queue", "port": 15673, "health_endpoint": "/api/health/checks/alarms"},
                "rabbitmq-3": {"type": "queue", "port": 15674, "health_endpoint": "/api/health/checks/alarms"},

                # IntelOwl stack
                "intelowl-django": {"type": "application", "port": 8000, "health_endpoint": "/api/health"},
                "intelowl-nginx": {"type": "proxy", "port": 80, "health_endpoint": "/"},
                "cytoscape-server": {"type": "application", "port": 3000, "health_endpoint": "/health"},

                # Phase 7 services
                "dm-crawler": {"type": "intelligence", "port": 8001, "health_endpoint": "/health"},
                "crypto-intel": {"type": "intelligence", "port": 8002, "health_endpoint": "/health"},
                "reputation-analyzer": {"type": "intelligence", "port": 8003, "health_endpoint": "/health"},
                "economics-processor": {"type": "intelligence", "port": 8004, "health_endpoint": "/health"},

                # Phase 8 services
                "tactical-intel": {"type": "security", "port": 8005, "health_endpoint": "/health"},
                "defense-automation": {"type": "security", "port": 8006, "health_endpoint": "/health"},
                "opsec-enforcer": {"type": "security", "port": 8007, "health_endpoint": "/health"},
                "intel-fusion": {"type": "security", "port": 8008, "health_endpoint": "/health"},

                # Predictive Cache Infrastructure
                "predictive-cache": {"type": "infrastructure", "port": 8044, "health_endpoint": "/health"},

                # Phase 9 services
                "autonomous-coordinator": {"type": "autonomous", "port": 8009, "health_endpoint": "/health"},
                "adaptive-learning": {"type": "autonomous", "port": 8010, "health_endpoint": "/health"},
                "resource-manager": {"type": "autonomous", "port": 8011, "health_endpoint": "/health"},
                "knowledge-evolution": {"type": "autonomous", "port": 8012, "health_endpoint": "/health"},
            },
            "alerts": {
                "response_time_threshold": 5.0,
                "cpu_threshold": 85.0,
                "memory_threshold": 90.0,
                "error_rate_threshold": 5.0,
                "disk_threshold": 95.0,
                # Predictive cache specific alerts
                "cache_hit_rate_threshold": 0.7,
                "cache_response_time_threshold": 0.1,
                "ml_prediction_accuracy_threshold": 0.8,
                "cache_memory_utilization_threshold": 0.9,
            },
            "check_interval": 30,
            "history_retention_hours": 24,
            "prometheus_port": 9091,
        }

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics for health monitoring."""
        self.prom_service_up = Gauge('bev_service_up', 'Service availability', ['service', 'type'])
        self.prom_response_time = Gauge('bev_service_response_time_seconds', 'Service response time', ['service'])
        self.prom_cpu_usage = Gauge('bev_service_cpu_usage_percent', 'Service CPU usage', ['service'])
        self.prom_memory_usage = Gauge('bev_service_memory_usage_percent', 'Service memory usage', ['service'])
        self.prom_error_count = Counter('bev_service_errors_total', 'Service error count', ['service'])
        self.prom_check_duration = Histogram('bev_health_check_duration_seconds', 'Health check duration', ['service'])
        self.prom_active_alerts = Gauge('bev_active_alerts_total', 'Active alerts count', ['service', 'severity'])

        # Predictive cache specific metrics
        self.prom_cache_hit_rate = Gauge('bev_cache_hit_rate', 'Cache hit rate', ['service', 'tier'])
        self.prom_cache_memory_utilization = Gauge('bev_cache_memory_utilization', 'Cache memory utilization', ['service', 'tier'])
        self.prom_ml_prediction_accuracy = Gauge('bev_ml_prediction_accuracy', 'ML prediction accuracy', ['service'])
        self.prom_cache_warming_tasks = Gauge('bev_cache_warming_tasks', 'Active cache warming tasks', ['service'])
        self.prom_cache_optimization_score = Gauge('bev_cache_optimization_score', 'Cache optimization score', ['service'])

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for health monitoring."""
        logger = logging.getLogger('health_monitor')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    async def initialize(self):
        """Initialize external connections and start monitoring."""
        try:
            # Initialize Redis connection
            self.redis_client = await aioredis.from_url(
                "redis://redis:6379",
                password=self.config.get("redis_password"),
                decode_responses=True
            )

            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )

            # Start Prometheus metrics server
            start_http_server(self.config.get("prometheus_port", 9091))

            self.logger.info("Health monitor initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize health monitor: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown of health monitor."""
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
        self.logger.info("Health monitor shutdown completed")

    async def check_service_health(self, service_name: str, service_config: Dict[str, Any]) -> ServiceMetrics:
        """
        Perform comprehensive health check for a single service.

        Args:
            service_name: Name of the service to check
            service_config: Service configuration dictionary

        Returns:
            ServiceMetrics: Complete metrics for the service
        """
        start_time = time.time()

        try:
            # Get container metrics if available
            container_metrics = await self._get_container_metrics(service_name)

            # Check service endpoint if configured
            endpoint_metrics = await self._check_service_endpoint(service_name, service_config)

            # Combine metrics
            status = self._determine_service_status(endpoint_metrics, container_metrics)

            metrics = ServiceMetrics(
                service_name=service_name,
                status=status,
                response_time=endpoint_metrics.get("response_time", 0.0),
                cpu_usage=container_metrics.get("cpu_usage", 0.0),
                memory_usage=container_metrics.get("memory_usage", 0.0),
                error_count=endpoint_metrics.get("error_count", 0),
                success_count=endpoint_metrics.get("success_count", 0),
                last_check=datetime.now(timezone.utc),
                uptime=container_metrics.get("uptime", 0.0),
                disk_usage=container_metrics.get("disk_usage"),
                network_io=container_metrics.get("network_io"),
                custom_metrics=endpoint_metrics.get("custom_metrics")
            )

            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)

            # Store metrics in history
            await self._store_metrics(metrics)

            # Check for alerts
            await self._check_alerts(metrics)

            check_duration = time.time() - start_time
            self.prom_check_duration.labels(service=service_name).observe(check_duration)

            return metrics

        except Exception as e:
            self.logger.error(f"Error checking {service_name} health: {e}")

            error_metrics = ServiceMetrics(
                service_name=service_name,
                status=HealthStatus.UNKNOWN,
                response_time=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                error_count=1,
                success_count=0,
                last_check=datetime.now(timezone.utc),
                uptime=0.0
            )

            self._update_prometheus_metrics(error_metrics)
            return error_metrics

    async def _get_container_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get Docker container metrics for service."""
        try:
            container_name = f"bev_{service_name.replace('-', '_')}"
            container = self.docker_client.containers.get(container_name)

            # Get container stats
            stats = container.stats(stream=False)

            # Calculate CPU usage
            cpu_usage = self._calculate_cpu_usage(stats)

            # Calculate memory usage
            memory_usage = self._calculate_memory_usage(stats)

            # Get uptime
            started_at = container.attrs['State']['StartedAt']
            uptime = self._calculate_uptime(started_at)

            # Get network I/O
            network_io = self._extract_network_io(stats)

            return {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "uptime": uptime,
                "network_io": network_io,
                "status": container.status
            }

        except docker.errors.NotFound:
            self.logger.warning(f"Container for service {service_name} not found")
            return {}
        except Exception as e:
            self.logger.error(f"Error getting container metrics for {service_name}: {e}")
            return {}

    async def _check_service_endpoint(self, service_name: str, service_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check service health via HTTP endpoint."""
        health_endpoint = service_config.get("health_endpoint")
        if not health_endpoint:
            return {"response_time": 0.0, "success_count": 1, "error_count": 0}

        port = service_config.get("port", 80)
        url = f"http://{service_name}:{port}{health_endpoint}"

        start_time = time.time()

        try:
            async with self.session.get(url) as response:
                response_time = time.time() - start_time

                if response.status == 200:
                    # Try to parse response for additional metrics
                    try:
                        data = await response.json()
                        custom_metrics = self._extract_custom_metrics(data)
                    except:
                        custom_metrics = None

                    return {
                        "response_time": response_time,
                        "success_count": 1,
                        "error_count": 0,
                        "status_code": response.status,
                        "custom_metrics": custom_metrics
                    }
                else:
                    return {
                        "response_time": response_time,
                        "success_count": 0,
                        "error_count": 1,
                        "status_code": response.status
                    }

        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"Health check failed for {service_name}: {e}")

            return {
                "response_time": response_time,
                "success_count": 0,
                "error_count": 1,
                "error": str(e)
            }

    def _calculate_cpu_usage(self, stats: Dict[str, Any]) -> float:
        """Calculate CPU usage percentage from Docker stats."""
        try:
            cpu_stats = stats['cpu_stats']
            prev_cpu_stats = stats['precpu_stats']

            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - prev_cpu_stats['cpu_usage']['total_usage']
            system_delta = cpu_stats['system_cpu_usage'] - prev_cpu_stats['system_cpu_usage']

            if system_delta > 0:
                cpu_usage = (cpu_delta / system_delta) * len(cpu_stats['cpu_usage']['percpu_usage']) * 100.0
                return round(cpu_usage, 2)

        except (KeyError, ZeroDivisionError):
            pass

        return 0.0

    def _calculate_memory_usage(self, stats: Dict[str, Any]) -> float:
        """Calculate memory usage percentage from Docker stats."""
        try:
            memory_stats = stats['memory_stats']
            usage = memory_stats['usage']
            limit = memory_stats['limit']

            if limit > 0:
                memory_usage = (usage / limit) * 100.0
                return round(memory_usage, 2)

        except KeyError:
            pass

        return 0.0

    def _calculate_uptime(self, started_at: str) -> float:
        """Calculate service uptime in seconds."""
        try:
            start_time = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            uptime = (datetime.now(timezone.utc) - start_time).total_seconds()
            return uptime
        except:
            return 0.0

    def _extract_network_io(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """Extract network I/O statistics."""
        try:
            networks = stats['networks']
            total_rx = sum(net['rx_bytes'] for net in networks.values())
            total_tx = sum(net['tx_bytes'] for net in networks.values())

            return {
                "rx_bytes": total_rx,
                "tx_bytes": total_tx
            }
        except KeyError:
            return {}

    def _extract_custom_metrics(self, health_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract custom metrics from service health endpoint."""
        custom_metrics = {}

        # Common health endpoint patterns
        if 'metrics' in health_data:
            custom_metrics.update(health_data['metrics'])

        if 'database' in health_data:
            db_info = health_data['database']
            if 'connections' in db_info:
                custom_metrics['db_connections'] = db_info['connections']

        if 'queue' in health_data:
            queue_info = health_data['queue']
            if 'messages' in queue_info:
                custom_metrics['queue_size'] = queue_info['messages']

        # Predictive cache specific metrics
        if 'cache' in health_data:
            cache_info = health_data['cache']
            custom_metrics['cache_hit_rate'] = cache_info.get('hit_rate', 0.0)
            custom_metrics['cache_response_time'] = cache_info.get('response_time', 0.0)
            custom_metrics['cache_memory_utilization'] = cache_info.get('memory_utilization', 0.0)

        if 'ml' in health_data:
            ml_info = health_data['ml']
            custom_metrics['ml_prediction_accuracy'] = ml_info.get('prediction_accuracy', 0.0)
            custom_metrics['ml_models_loaded'] = ml_info.get('models_loaded', 0)

        if 'warming' in health_data:
            warming_info = health_data['warming']
            custom_metrics['warming_tasks_active'] = warming_info.get('tasks_active', 0)
            custom_metrics['warming_tasks_completed'] = warming_info.get('tasks_completed', 0)

        if 'optimization' in health_data:
            opt_info = health_data['optimization']
            custom_metrics['optimization_score'] = opt_info.get('score', 0.0)
            custom_metrics['optimization_last_run'] = opt_info.get('last_run', 0)

        return custom_metrics if custom_metrics else None

    def _determine_service_status(self, endpoint_metrics: Dict[str, Any], container_metrics: Dict[str, Any]) -> HealthStatus:
        """Determine overall service health status."""
        # Check if container is running
        container_status = container_metrics.get("status", "").lower()
        if container_status not in ["running", ""]:
            return HealthStatus.UNHEALTHY

        # Check endpoint response
        if endpoint_metrics.get("error_count", 0) > 0:
            return HealthStatus.DEGRADED

        # Check resource usage
        cpu_usage = container_metrics.get("cpu_usage", 0)
        memory_usage = container_metrics.get("memory_usage", 0)

        if cpu_usage > 95 or memory_usage > 95:
            return HealthStatus.UNHEALTHY
        elif cpu_usage > 85 or memory_usage > 90:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _update_prometheus_metrics(self, metrics: ServiceMetrics):
        """Update Prometheus metrics with latest service metrics."""
        service_type = self.services.get(metrics.service_name, {}).get("type", "unknown")

        # Service availability
        status_value = 1 if metrics.status == HealthStatus.HEALTHY else 0
        self.prom_service_up.labels(service=metrics.service_name, type=service_type).set(status_value)

        # Performance metrics
        self.prom_response_time.labels(service=metrics.service_name).set(metrics.response_time)
        self.prom_cpu_usage.labels(service=metrics.service_name).set(metrics.cpu_usage)
        self.prom_memory_usage.labels(service=metrics.service_name).set(metrics.memory_usage)

        # Error tracking
        if metrics.error_count > 0:
            self.prom_error_count.labels(service=metrics.service_name).inc(metrics.error_count)

        # Predictive cache specific metrics
        if metrics.service_name == "predictive-cache" and metrics.custom_metrics:
            custom = metrics.custom_metrics

            # Cache hit rate by tier (if available)
            if 'cache_hit_rate' in custom:
                self.prom_cache_hit_rate.labels(
                    service=metrics.service_name, tier="overall"
                ).set(custom['cache_hit_rate'])

            # Cache memory utilization
            if 'cache_memory_utilization' in custom:
                self.prom_cache_memory_utilization.labels(
                    service=metrics.service_name, tier="overall"
                ).set(custom['cache_memory_utilization'])

            # ML prediction accuracy
            if 'ml_prediction_accuracy' in custom:
                self.prom_ml_prediction_accuracy.labels(
                    service=metrics.service_name
                ).set(custom['ml_prediction_accuracy'])

            # Cache warming tasks
            if 'warming_tasks_active' in custom:
                self.prom_cache_warming_tasks.labels(
                    service=metrics.service_name
                ).set(custom['warming_tasks_active'])

            # Cache optimization score
            if 'optimization_score' in custom:
                self.prom_cache_optimization_score.labels(
                    service=metrics.service_name
                ).set(custom['optimization_score'])

    async def _store_metrics(self, metrics: ServiceMetrics):
        """Store metrics in Redis and in-memory history."""
        if self.redis_client:
            try:
                # Store current metrics
                metrics_key = f"health:metrics:{metrics.service_name}"
                await self.redis_client.hset(metrics_key, mapping=asdict(metrics))
                await self.redis_client.expire(metrics_key, 3600)  # 1 hour TTL

                # Store in time series
                ts_key = f"health:timeseries:{metrics.service_name}"
                timestamp = int(metrics.last_check.timestamp())

                ts_data = {
                    "timestamp": timestamp,
                    "status": metrics.status.value,
                    "response_time": metrics.response_time,
                    "cpu_usage": metrics.cpu_usage,
                    "memory_usage": metrics.memory_usage
                }

                await self.redis_client.zadd(ts_key, {json.dumps(ts_data): timestamp})

                # Keep only last 24 hours
                cutoff = timestamp - (24 * 3600)
                await self.redis_client.zremrangebyscore(ts_key, 0, cutoff)

            except Exception as e:
                self.logger.error(f"Error storing metrics for {metrics.service_name}: {e}")

        # Update in-memory history
        if metrics.service_name not in self.service_history:
            self.service_history[metrics.service_name] = []

        self.service_history[metrics.service_name].append(metrics)

        # Limit history size
        max_history = 2880  # 24 hours * 60 minutes * 2 (30-second intervals)
        if len(self.service_history[metrics.service_name]) > max_history:
            self.service_history[metrics.service_name] = self.service_history[metrics.service_name][-max_history:]

    async def _check_alerts(self, metrics: ServiceMetrics):
        """Check for alert conditions and trigger alerts."""
        alerts = []
        service_name = metrics.service_name

        # Response time alert
        response_threshold = self.alert_config.get("response_time_threshold", 5.0)
        if metrics.response_time > response_threshold:
            alerts.append(HealthAlert(
                service_name=service_name,
                alert_type="response_time",
                severity=AlertSeverity.WARNING if metrics.response_time < response_threshold * 2 else AlertSeverity.CRITICAL,
                threshold_value=response_threshold,
                current_value=metrics.response_time,
                message=f"High response time: {metrics.response_time:.2f}s",
                timestamp=datetime.now(timezone.utc)
            ))

        # CPU usage alert
        cpu_threshold = self.alert_config.get("cpu_threshold", 85.0)
        if metrics.cpu_usage > cpu_threshold:
            alerts.append(HealthAlert(
                service_name=service_name,
                alert_type="cpu_usage",
                severity=AlertSeverity.WARNING if metrics.cpu_usage < 95 else AlertSeverity.CRITICAL,
                threshold_value=cpu_threshold,
                current_value=metrics.cpu_usage,
                message=f"High CPU usage: {metrics.cpu_usage:.1f}%",
                timestamp=datetime.now(timezone.utc)
            ))

        # Memory usage alert
        memory_threshold = self.alert_config.get("memory_threshold", 90.0)
        if metrics.memory_usage > memory_threshold:
            alerts.append(HealthAlert(
                service_name=service_name,
                alert_type="memory_usage",
                severity=AlertSeverity.WARNING if metrics.memory_usage < 95 else AlertSeverity.CRITICAL,
                threshold_value=memory_threshold,
                current_value=metrics.memory_usage,
                message=f"High memory usage: {metrics.memory_usage:.1f}%",
                timestamp=datetime.now(timezone.utc)
            ))

        # Service health alert
        if metrics.status in [HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN]:
            alerts.append(HealthAlert(
                service_name=service_name,
                alert_type="service_health",
                severity=AlertSeverity.CRITICAL,
                threshold_value=1.0,
                current_value=0.0,
                message=f"Service is {metrics.status.value}",
                timestamp=datetime.now(timezone.utc)
            ))

        # Predictive cache specific alerts
        if service_name == "predictive-cache" and metrics.custom_metrics:
            custom = metrics.custom_metrics

            # Cache hit rate alert
            hit_rate_threshold = self.alert_config.get("cache_hit_rate_threshold", 0.7)
            if 'cache_hit_rate' in custom and custom['cache_hit_rate'] < hit_rate_threshold:
                alerts.append(HealthAlert(
                    service_name=service_name,
                    alert_type="cache_hit_rate",
                    severity=AlertSeverity.WARNING if custom['cache_hit_rate'] > hit_rate_threshold * 0.8 else AlertSeverity.CRITICAL,
                    threshold_value=hit_rate_threshold,
                    current_value=custom['cache_hit_rate'],
                    message=f"Low cache hit rate: {custom['cache_hit_rate']:.1%}",
                    timestamp=datetime.now(timezone.utc)
                ))

            # Cache response time alert
            cache_response_threshold = self.alert_config.get("cache_response_time_threshold", 0.1)
            if 'cache_response_time' in custom and custom['cache_response_time'] > cache_response_threshold:
                alerts.append(HealthAlert(
                    service_name=service_name,
                    alert_type="cache_response_time",
                    severity=AlertSeverity.WARNING if custom['cache_response_time'] < cache_response_threshold * 2 else AlertSeverity.CRITICAL,
                    threshold_value=cache_response_threshold,
                    current_value=custom['cache_response_time'],
                    message=f"High cache response time: {custom['cache_response_time']:.3f}s",
                    timestamp=datetime.now(timezone.utc)
                ))

            # ML prediction accuracy alert
            ml_accuracy_threshold = self.alert_config.get("ml_prediction_accuracy_threshold", 0.8)
            if 'ml_prediction_accuracy' in custom and custom['ml_prediction_accuracy'] < ml_accuracy_threshold:
                alerts.append(HealthAlert(
                    service_name=service_name,
                    alert_type="ml_prediction_accuracy",
                    severity=AlertSeverity.WARNING,
                    threshold_value=ml_accuracy_threshold,
                    current_value=custom['ml_prediction_accuracy'],
                    message=f"Low ML prediction accuracy: {custom['ml_prediction_accuracy']:.1%}",
                    timestamp=datetime.now(timezone.utc)
                ))

            # Cache memory utilization alert
            cache_memory_threshold = self.alert_config.get("cache_memory_utilization_threshold", 0.9)
            if 'cache_memory_utilization' in custom and custom['cache_memory_utilization'] > cache_memory_threshold:
                alerts.append(HealthAlert(
                    service_name=service_name,
                    alert_type="cache_memory_utilization",
                    severity=AlertSeverity.WARNING if custom['cache_memory_utilization'] < 0.95 else AlertSeverity.CRITICAL,
                    threshold_value=cache_memory_threshold,
                    current_value=custom['cache_memory_utilization'],
                    message=f"High cache memory utilization: {custom['cache_memory_utilization']:.1%}",
                    timestamp=datetime.now(timezone.utc)
                ))

        # Store and process alerts
        if alerts:
            if service_name not in self.active_alerts:
                self.active_alerts[service_name] = []

            for alert in alerts:
                # Check if this alert already exists
                existing = any(
                    a.alert_type == alert.alert_type and not a.acknowledged
                    for a in self.active_alerts[service_name]
                )

                if not existing:
                    self.active_alerts[service_name].append(alert)
                    await self._send_alert(alert)

                    # Update Prometheus metrics
                    self.prom_active_alerts.labels(
                        service=service_name,
                        severity=alert.severity.value
                    ).inc()

    async def _send_alert(self, alert: HealthAlert):
        """Send alert notification."""
        self.logger.warning(
            f"ALERT [{alert.severity.value.upper()}] {alert.service_name}: {alert.message}"
        )

        # Store alert in Redis
        if self.redis_client:
            try:
                alert_key = f"health:alerts:{alert.service_name}"
                alert_data = asdict(alert)
                await self.redis_client.lpush(alert_key, json.dumps(alert_data, default=str))
                await self.redis_client.ltrim(alert_key, 0, 99)  # Keep last 100 alerts
                await self.redis_client.expire(alert_key, 86400)  # 24 hours TTL
            except Exception as e:
                self.logger.error(f"Error storing alert: {e}")

    async def get_service_metrics(self, service_name: str) -> Optional[ServiceMetrics]:
        """Get current metrics for a specific service."""
        return self.service_metrics.get(service_name)

    async def get_all_service_metrics(self) -> Dict[str, ServiceMetrics]:
        """Get current metrics for all services."""
        return self.service_metrics.copy()

    async def get_service_history(self, service_name: str, hours: int = 1) -> List[ServiceMetrics]:
        """Get historical metrics for a service."""
        if service_name not in self.service_history:
            return []

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [
            metric for metric in self.service_history[service_name]
            if metric.last_check >= cutoff_time
        ]

    async def acknowledge_alert(self, service_name: str, alert_type: str) -> bool:
        """Acknowledge an active alert."""
        if service_name in self.active_alerts:
            for alert in self.active_alerts[service_name]:
                if alert.alert_type == alert_type and not alert.acknowledged:
                    alert.acknowledged = True
                    self.logger.info(f"Alert acknowledged: {service_name} - {alert_type}")
                    return True
        return False

    async def run_monitoring_loop(self):
        """Main monitoring loop - runs health checks every 30 seconds."""
        self.logger.info(f"Starting health monitoring loop (interval: {self.check_interval}s)")

        while True:
            try:
                start_time = time.time()

                # Create tasks for parallel health checks
                tasks = []
                for service_name, service_config in self.services.items():
                    task = asyncio.create_task(
                        self.check_service_health(service_name, service_config)
                    )
                    tasks.append((service_name, task))

                # Wait for all checks to complete
                for service_name, task in tasks:
                    try:
                        metrics = await task
                        self.service_metrics[service_name] = metrics
                    except Exception as e:
                        self.logger.error(f"Health check failed for {service_name}: {e}")

                # Log summary
                total_services = len(self.services)
                healthy_services = sum(
                    1 for m in self.service_metrics.values()
                    if m.status == HealthStatus.HEALTHY
                )

                elapsed_time = time.time() - start_time
                self.logger.info(
                    f"Health check completed: {healthy_services}/{total_services} healthy "
                    f"(took {elapsed_time:.2f}s)"
                )

                # Wait for next interval
                sleep_time = max(0, self.check_interval - elapsed_time)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)


async def main():
    """Main entry point for health monitor."""
    health_monitor = HealthMonitor()

    try:
        await health_monitor.initialize()
        await health_monitor.run_monitoring_loop()
    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await health_monitor.shutdown()


if __name__ == "__main__":
    asyncio.run(main())