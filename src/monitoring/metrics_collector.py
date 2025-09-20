import os
"""
BEV OSINT Framework - Advanced Metrics Collection System
Comprehensive metrics collection with time-series data and performance analytics.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import aioredis
import asyncpg
import psutil
import docker
from prometheus_client import Gauge, Counter, Histogram, Summary, Info
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import numpy as np
from collections import defaultdict, deque
import yaml


class MetricType(Enum):
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class MetricCategory(Enum):
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"
    RELIABILITY = "reliability"
    SECURITY = "security"
    BUSINESS = "business"


@dataclass
class MetricDefinition:
    """Definition of a metric including metadata and collection configuration."""
    name: str
    metric_type: MetricType
    category: MetricCategory
    description: str
    unit: str
    labels: List[str]
    collection_interval: int
    retention_hours: int
    alert_enabled: bool = True
    aggregation_functions: List[str] = None


@dataclass
class MetricValue:
    """Single metric value with metadata."""
    metric_name: str
    value: Union[float, int, str]
    labels: Dict[str, str]
    timestamp: datetime
    source_service: str
    tags: Optional[Dict[str, str]] = None


@dataclass
class AggregatedMetric:
    """Aggregated metric with statistical data."""
    metric_name: str
    avg: float
    min: float
    max: float
    sum: float
    count: int
    percentiles: Dict[int, float]
    period_start: datetime
    period_end: datetime
    labels: Dict[str, str]


class MetricsCollector:
    """
    Advanced metrics collection system for BEV OSINT framework.
    Provides comprehensive metrics collection, aggregation, and time-series storage.
    """

    def __init__(self, config_path: str = "/app/config/metrics_collector.yml"):
        self.config = self._load_config(config_path)
        self.metric_definitions = self._load_metric_definitions()

        # Collection state
        self.metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.aggregated_metrics: Dict[str, List[AggregatedMetric]] = defaultdict(list)
        self.collection_tasks: Dict[str, asyncio.Task] = {}

        # External connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.postgres_pool: Optional[asyncpg.Pool] = None
        self.influx_client: Optional[influxdb_client.InfluxDBClient] = None
        self.docker_client = docker.from_env()
        self.session: Optional[aiohttp.ClientSession] = None

        # Prometheus collectors
        self.prometheus_metrics: Dict[str, Any] = {}

        # Statistics tracking
        self.collection_stats = {
            "metrics_collected": 0,
            "collection_errors": 0,
            "last_collection": None,
            "average_collection_time": 0.0
        }

        self.logger = self._setup_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load metrics collector configuration."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for metrics collection."""
        return {
            "collection": {
                "default_interval": 30,
                "batch_size": 1000,
                "max_memory_metrics": 100000,
                "aggregation_intervals": [60, 300, 900, 3600],  # 1m, 5m, 15m, 1h
            },
            "storage": {
                "influxdb": {
                    "url": "http://influxdb:8086",
                    "token": "${INFLUXDB_TOKEN}",
                    "org": "bev-osint",
                    "bucket": "metrics",
                    "measurement": "bev_metrics"
                },
                "redis": {
                    "url": "redis://redis:6379",
                    "db": 2,
                    "ttl": 86400
                },
                "postgres": {
                    "host": "postgres",
                    "port": 5432,
                    "database": "osint",
                    "table": "metrics_timeseries"
                }
            },
            "sources": {
                "docker_stats": True,
                "system_metrics": True,
                "application_metrics": True,
                "custom_endpoints": True
            },
            "aggregation": {
                "percentiles": [50, 90, 95, 99],
                "enable_real_time": True,
                "batch_aggregation": True
            }
        }

    def _load_metric_definitions(self) -> Dict[str, MetricDefinition]:
        """Load comprehensive metric definitions."""
        definitions = {}

        # Performance metrics
        performance_metrics = [
            MetricDefinition(
                name="service_response_time",
                metric_type=MetricType.HISTOGRAM,
                category=MetricCategory.PERFORMANCE,
                description="Service response time in seconds",
                unit="seconds",
                labels=["service", "endpoint", "method"],
                collection_interval=30,
                retention_hours=168,  # 1 week
                aggregation_functions=["avg", "p95", "p99"]
            ),
            MetricDefinition(
                name="service_throughput",
                metric_type=MetricType.GAUGE,
                description="Service throughput in requests per second",
                unit="req/s",
                labels=["service"],
                collection_interval=30,
                retention_hours=168
            ),
            MetricDefinition(
                name="cpu_usage_percent",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.PERFORMANCE,
                description="CPU usage percentage",
                unit="percent",
                labels=["service", "container"],
                collection_interval=15,
                retention_hours=72
            ),
            MetricDefinition(
                name="memory_usage_percent",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.PERFORMANCE,
                description="Memory usage percentage",
                unit="percent",
                labels=["service", "container"],
                collection_interval=15,
                retention_hours=72
            ),
            MetricDefinition(
                name="disk_io_ops",
                metric_type=MetricType.COUNTER,
                category=MetricCategory.PERFORMANCE,
                description="Disk I/O operations",
                unit="ops",
                labels=["service", "operation"],
                collection_interval=30,
                retention_hours=24
            ),
            MetricDefinition(
                name="network_bytes",
                metric_type=MetricType.COUNTER,
                category=MetricCategory.PERFORMANCE,
                description="Network bytes transferred",
                unit="bytes",
                labels=["service", "direction"],
                collection_interval=30,
                retention_hours=24
            )
        ]

        # Availability metrics
        availability_metrics = [
            MetricDefinition(
                name="service_up",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.AVAILABILITY,
                description="Service availability (1=up, 0=down)",
                unit="boolean",
                labels=["service", "type"],
                collection_interval=30,
                retention_hours=720  # 30 days
            ),
            MetricDefinition(
                name="service_uptime",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.AVAILABILITY,
                description="Service uptime in seconds",
                unit="seconds",
                labels=["service"],
                collection_interval=60,
                retention_hours=720
            ),
            MetricDefinition(
                name="health_check_success",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.AVAILABILITY,
                description="Health check success rate",
                unit="percent",
                labels=["service", "check_type"],
                collection_interval=30,
                retention_hours=168
            )
        ]

        # Reliability metrics
        reliability_metrics = [
            MetricDefinition(
                name="error_rate",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.RELIABILITY,
                description="Error rate percentage",
                unit="percent",
                labels=["service", "error_type"],
                collection_interval=30,
                retention_hours=168
            ),
            MetricDefinition(
                name="request_errors",
                metric_type=MetricType.COUNTER,
                category=MetricCategory.RELIABILITY,
                description="Total request errors",
                unit="count",
                labels=["service", "status_code"],
                collection_interval=30,
                retention_hours=168
            ),
            MetricDefinition(
                name="timeout_count",
                metric_type=MetricType.COUNTER,
                category=MetricCategory.RELIABILITY,
                description="Request timeout count",
                unit="count",
                labels=["service"],
                collection_interval=30,
                retention_hours=168
            )
        ]

        # Security metrics
        security_metrics = [
            MetricDefinition(
                name="authentication_failures",
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                description="Authentication failure count",
                unit="count",
                labels=["service", "source_ip"],
                collection_interval=30,
                retention_hours=720
            ),
            MetricDefinition(
                name="suspicious_activity",
                metric_type=MetricType.COUNTER,
                category=MetricCategory.SECURITY,
                description="Suspicious activity count",
                unit="count",
                labels=["service", "activity_type"],
                collection_interval=30,
                retention_hours=720
            )
        ]

        # Business metrics
        business_metrics = [
            MetricDefinition(
                name="active_users",
                metric_type=MetricType.GAUGE,
                category=MetricCategory.BUSINESS,
                description="Active user count",
                unit="count",
                labels=["service"],
                collection_interval=60,
                retention_hours=720
            ),
            MetricDefinition(
                name="api_requests",
                metric_type=MetricType.COUNTER,
                category=MetricCategory.BUSINESS,
                description="Total API requests",
                unit="count",
                labels=["service", "endpoint"],
                collection_interval=30,
                retention_hours=168
            ),
            MetricDefinition(
                name="data_processed",
                metric_type=MetricType.COUNTER,
                category=MetricCategory.BUSINESS,
                description="Data processed volume",
                unit="bytes",
                labels=["service", "data_type"],
                collection_interval=60,
                retention_hours=168
            )
        ]

        # Combine all metrics
        all_metrics = (performance_metrics + availability_metrics +
                      reliability_metrics + security_metrics + business_metrics)

        for metric in all_metrics:
            definitions[metric.name] = metric

        return definitions

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for metrics collection."""
        logger = logging.getLogger('metrics_collector')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    async def initialize(self):
        """Initialize external connections and metric collectors."""
        try:
            # Initialize Redis connection
            redis_config = self.config["storage"]["redis"]
            self.redis_client = await aioredis.from_url(
                redis_config["url"],
                db=redis_config.get("db", 2),
                decode_responses=True
            )

            # Initialize PostgreSQL connection pool
            postgres_config = self.config["storage"]["postgres"]
            self.postgres_pool = await asyncpg.create_pool(
                host=postgres_config["host"],
                port=postgres_config["port"],
                database=postgres_config["database"],
                user="bev",
                password=os.getenv('DB_PASSWORD', 'dev_password'),  # Should be from env
                min_size=2,
                max_size=10
            )

            # Initialize InfluxDB client
            influx_config = self.config["storage"]["influxdb"]
            self.influx_client = influxdb_client.InfluxDBClient(
                url=influx_config["url"],
                token=influx_config["token"],
                org=influx_config["org"]
            )

            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )

            # Initialize Prometheus metrics
            self._init_prometheus_metrics()

            # Create database tables if needed
            await self._ensure_database_schema()

            self.logger.info("Metrics collector initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize metrics collector: {e}")
            raise

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metric collectors."""
        for name, definition in self.metric_definitions.items():
            labels = definition.labels

            if definition.metric_type == MetricType.GAUGE:
                self.prometheus_metrics[name] = Gauge(
                    f"bev_{name}",
                    definition.description,
                    labels
                )
            elif definition.metric_type == MetricType.COUNTER:
                self.prometheus_metrics[name] = Counter(
                    f"bev_{name}_total",
                    definition.description,
                    labels
                )
            elif definition.metric_type == MetricType.HISTOGRAM:
                self.prometheus_metrics[name] = Histogram(
                    f"bev_{name}",
                    definition.description,
                    labels
                )
            elif definition.metric_type == MetricType.SUMMARY:
                self.prometheus_metrics[name] = Summary(
                    f"bev_{name}",
                    definition.description,
                    labels
                )

    async def _ensure_database_schema(self):
        """Ensure database schema exists for metrics storage."""
        if self.postgres_pool:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS metrics_timeseries (
                        id SERIAL PRIMARY KEY,
                        metric_name VARCHAR(255) NOT NULL,
                        service_name VARCHAR(255) NOT NULL,
                        value DOUBLE PRECISION NOT NULL,
                        labels JSONB,
                        tags JSONB,
                        timestamp TIMESTAMPTZ NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    )
                """)

                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_name_service_time
                    ON metrics_timeseries(metric_name, service_name, timestamp)
                """)

                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                    ON metrics_timeseries(timestamp)
                """)

    async def collect_metric(self, metric_name: str, value: Union[float, int, str],
                           labels: Dict[str, str], source_service: str,
                           tags: Optional[Dict[str, str]] = None) -> bool:
        """
        Collect a single metric value.

        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Metric labels
            source_service: Source service name
            tags: Additional tags

        Returns:
            bool: Success status
        """
        try:
            metric_value = MetricValue(
                metric_name=metric_name,
                value=value,
                labels=labels,
                timestamp=datetime.now(timezone.utc),
                source_service=source_service,
                tags=tags
            )

            # Store in memory buffer
            self.metric_values[metric_name].append(metric_value)

            # Update Prometheus metric
            await self._update_prometheus_metric(metric_value)

            # Store in external systems
            await self._store_metric_value(metric_value)

            self.collection_stats["metrics_collected"] += 1
            return True

        except Exception as e:
            self.logger.error(f"Error collecting metric {metric_name}: {e}")
            self.collection_stats["collection_errors"] += 1
            return False

    async def _update_prometheus_metric(self, metric_value: MetricValue):
        """Update Prometheus metric with new value."""
        metric_name = metric_value.metric_name
        if metric_name not in self.prometheus_metrics:
            return

        prom_metric = self.prometheus_metrics[metric_name]
        label_values = [metric_value.labels.get(label, "") for label in
                       self.metric_definitions[metric_name].labels]

        if isinstance(prom_metric, Gauge):
            prom_metric.labels(*label_values).set(float(metric_value.value))
        elif isinstance(prom_metric, Counter):
            prom_metric.labels(*label_values).inc(float(metric_value.value))
        elif isinstance(prom_metric, Histogram):
            prom_metric.labels(*label_values).observe(float(metric_value.value))
        elif isinstance(prom_metric, Summary):
            prom_metric.labels(*label_values).observe(float(metric_value.value))

    async def _store_metric_value(self, metric_value: MetricValue):
        """Store metric value in external storage systems."""
        # Store in InfluxDB
        if self.influx_client:
            try:
                write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)

                point = influxdb_client.Point("bev_metrics") \
                    .tag("service", metric_value.source_service) \
                    .tag("metric", metric_value.metric_name)

                # Add labels as tags
                for key, value in metric_value.labels.items():
                    point = point.tag(key, value)

                # Add custom tags
                if metric_value.tags:
                    for key, value in metric_value.tags.items():
                        point = point.tag(key, value)

                point = point.field("value", float(metric_value.value)) \
                    .time(metric_value.timestamp)

                write_api.write(
                    bucket=self.config["storage"]["influxdb"]["bucket"],
                    record=point
                )

            except Exception as e:
                self.logger.error(f"Error writing to InfluxDB: {e}")

        # Store in Redis for fast access
        if self.redis_client:
            try:
                key = f"metrics:{metric_value.metric_name}:{metric_value.source_service}"
                data = {
                    "value": metric_value.value,
                    "labels": json.dumps(metric_value.labels),
                    "timestamp": metric_value.timestamp.isoformat()
                }

                await self.redis_client.hset(key, mapping=data)
                await self.redis_client.expire(key, self.config["storage"]["redis"]["ttl"])

            except Exception as e:
                self.logger.error(f"Error writing to Redis: {e}")

        # Store in PostgreSQL for long-term storage
        if self.postgres_pool:
            try:
                async with self.postgres_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO metrics_timeseries
                        (metric_name, service_name, value, labels, tags, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                    metric_value.metric_name,
                    metric_value.source_service,
                    float(metric_value.value),
                    json.dumps(metric_value.labels),
                    json.dumps(metric_value.tags) if metric_value.tags else None,
                    metric_value.timestamp
                    )

            except Exception as e:
                self.logger.error(f"Error writing to PostgreSQL: {e}")

    async def collect_docker_metrics(self) -> List[MetricValue]:
        """Collect comprehensive Docker container metrics."""
        metrics = []

        try:
            for container in self.docker_client.containers.list():
                container_name = container.name
                service_name = container_name.replace("bev_", "")

                # Get container stats
                stats = container.stats(stream=False)

                # CPU metrics
                cpu_usage = self._calculate_cpu_usage(stats)
                metrics.append(MetricValue(
                    metric_name="cpu_usage_percent",
                    value=cpu_usage,
                    labels={"service": service_name, "container": container_name},
                    timestamp=datetime.now(timezone.utc),
                    source_service=service_name
                ))

                # Memory metrics
                memory_usage = self._calculate_memory_usage(stats)
                metrics.append(MetricValue(
                    metric_name="memory_usage_percent",
                    value=memory_usage,
                    labels={"service": service_name, "container": container_name},
                    timestamp=datetime.now(timezone.utc),
                    source_service=service_name
                ))

                # Network metrics
                network_stats = self._extract_network_stats(stats)
                for direction, bytes_count in network_stats.items():
                    metrics.append(MetricValue(
                        metric_name="network_bytes",
                        value=bytes_count,
                        labels={"service": service_name, "direction": direction},
                        timestamp=datetime.now(timezone.utc),
                        source_service=service_name
                    ))

                # Disk I/O metrics
                disk_stats = self._extract_disk_stats(stats)
                for operation, ops_count in disk_stats.items():
                    metrics.append(MetricValue(
                        metric_name="disk_io_ops",
                        value=ops_count,
                        labels={"service": service_name, "operation": operation},
                        timestamp=datetime.now(timezone.utc),
                        source_service=service_name
                    ))

        except Exception as e:
            self.logger.error(f"Error collecting Docker metrics: {e}")

        return metrics

    async def collect_system_metrics(self) -> List[MetricValue]:
        """Collect system-level metrics."""
        metrics = []
        timestamp = datetime.now(timezone.utc)

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricValue(
                metric_name="cpu_usage_percent",
                value=cpu_percent,
                labels={"service": "system", "container": "host"},
                timestamp=timestamp,
                source_service="system"
            ))

            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(MetricValue(
                metric_name="memory_usage_percent",
                value=memory.percent,
                labels={"service": "system", "container": "host"},
                timestamp=timestamp,
                source_service="system"
            ))

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(MetricValue(
                metric_name="disk_usage_percent",
                value=disk_percent,
                labels={"service": "system", "container": "host"},
                timestamp=timestamp,
                source_service="system"
            ))

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

        return metrics

    async def collect_application_metrics(self, service_name: str, endpoint: str) -> List[MetricValue]:
        """Collect application-specific metrics from service endpoints."""
        metrics = []

        try:
            url = f"http://{service_name}:8000/metrics"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Parse application metrics
                    for metric_name, metric_data in data.items():
                        if isinstance(metric_data, dict):
                            value = metric_data.get("value", 0)
                            labels = metric_data.get("labels", {})
                        else:
                            value = metric_data
                            labels = {}

                        labels["service"] = service_name

                        metrics.append(MetricValue(
                            metric_name=metric_name,
                            value=value,
                            labels=labels,
                            timestamp=datetime.now(timezone.utc),
                            source_service=service_name
                        ))

        except Exception as e:
            self.logger.debug(f"Could not collect application metrics from {service_name}: {e}")

        return metrics

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

    def _extract_network_stats(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """Extract network I/O statistics."""
        try:
            networks = stats['networks']
            total_rx = sum(net['rx_bytes'] for net in networks.values())
            total_tx = sum(net['tx_bytes'] for net in networks.values())

            return {
                "rx": total_rx,
                "tx": total_tx
            }
        except KeyError:
            return {"rx": 0.0, "tx": 0.0}

    def _extract_disk_stats(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """Extract disk I/O statistics."""
        try:
            blkio_stats = stats['blkio_stats']

            reads = 0
            writes = 0

            for entry in blkio_stats.get('io_service_bytes_recursive', []):
                if entry['op'] == 'Read':
                    reads += entry['value']
                elif entry['op'] == 'Write':
                    writes += entry['value']

            return {
                "read": reads,
                "write": writes
            }
        except KeyError:
            return {"read": 0.0, "write": 0.0}

    async def aggregate_metrics(self, metric_name: str, time_window: int) -> Optional[AggregatedMetric]:
        """
        Aggregate metrics over a time window.

        Args:
            metric_name: Name of the metric to aggregate
            time_window: Time window in seconds

        Returns:
            AggregatedMetric: Aggregated metric data
        """
        if metric_name not in self.metric_values:
            return None

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=time_window)

        # Filter values within time window
        values = [
            mv.value for mv in self.metric_values[metric_name]
            if start_time <= mv.timestamp <= end_time and isinstance(mv.value, (int, float))
        ]

        if not values:
            return None

        values_array = np.array(values, dtype=float)

        # Calculate percentiles
        percentiles = {}
        for p in self.config["aggregation"]["percentiles"]:
            percentiles[p] = float(np.percentile(values_array, p))

        # Get labels from the most recent metric
        recent_metric = max(
            [mv for mv in self.metric_values[metric_name] if start_time <= mv.timestamp <= end_time],
            key=lambda x: x.timestamp,
            default=None
        )

        labels = recent_metric.labels if recent_metric else {}

        return AggregatedMetric(
            metric_name=metric_name,
            avg=float(np.mean(values_array)),
            min=float(np.min(values_array)),
            max=float(np.max(values_array)),
            sum=float(np.sum(values_array)),
            count=len(values),
            percentiles=percentiles,
            period_start=start_time,
            period_end=end_time,
            labels=labels
        )

    async def get_metric_history(self, metric_name: str, service_name: str,
                                hours: int = 24) -> List[MetricValue]:
        """Get historical metric data from storage."""
        if self.postgres_pool:
            try:
                async with self.postgres_pool.acquire() as conn:
                    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

                    rows = await conn.fetch("""
                        SELECT metric_name, service_name, value, labels, tags, timestamp
                        FROM metrics_timeseries
                        WHERE metric_name = $1 AND service_name = $2 AND timestamp >= $3
                        ORDER BY timestamp DESC
                        LIMIT 10000
                    """, metric_name, service_name, cutoff_time)

                    return [
                        MetricValue(
                            metric_name=row['metric_name'],
                            value=row['value'],
                            labels=json.loads(row['labels']) if row['labels'] else {},
                            timestamp=row['timestamp'],
                            source_service=row['service_name'],
                            tags=json.loads(row['tags']) if row['tags'] else None
                        )
                        for row in rows
                    ]

            except Exception as e:
                self.logger.error(f"Error getting metric history: {e}")

        return []

    async def cleanup_old_metrics(self):
        """Clean up old metrics from storage."""
        try:
            # Clean up PostgreSQL
            if self.postgres_pool:
                async with self.postgres_pool.acquire() as conn:
                    cutoff_time = datetime.now(timezone.utc) - timedelta(days=30)
                    result = await conn.execute("""
                        DELETE FROM metrics_timeseries
                        WHERE timestamp < $1
                    """, cutoff_time)
                    self.logger.info(f"Cleaned up old metrics from PostgreSQL: {result}")

            # Clean up in-memory metrics
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
            for metric_name in self.metric_values:
                original_count = len(self.metric_values[metric_name])
                self.metric_values[metric_name] = deque(
                    [mv for mv in self.metric_values[metric_name] if mv.timestamp >= cutoff_time],
                    maxlen=10000
                )
                cleaned_count = original_count - len(self.metric_values[metric_name])
                if cleaned_count > 0:
                    self.logger.debug(f"Cleaned {cleaned_count} old {metric_name} metrics from memory")

        except Exception as e:
            self.logger.error(f"Error during metrics cleanup: {e}")

    async def start_collection_tasks(self):
        """Start background metric collection tasks."""
        # Docker metrics collection
        if self.config["sources"]["docker_stats"]:
            self.collection_tasks["docker"] = asyncio.create_task(
                self._docker_collection_loop()
            )

        # System metrics collection
        if self.config["sources"]["system_metrics"]:
            self.collection_tasks["system"] = asyncio.create_task(
                self._system_collection_loop()
            )

        # Application metrics collection
        if self.config["sources"]["application_metrics"]:
            self.collection_tasks["application"] = asyncio.create_task(
                self._application_collection_loop()
            )

        # Aggregation task
        if self.config["aggregation"]["enable_real_time"]:
            self.collection_tasks["aggregation"] = asyncio.create_task(
                self._aggregation_loop()
            )

        # Cleanup task
        self.collection_tasks["cleanup"] = asyncio.create_task(
            self._cleanup_loop()
        )

        self.logger.info(f"Started {len(self.collection_tasks)} collection tasks")

    async def _docker_collection_loop(self):
        """Background loop for Docker metrics collection."""
        while True:
            try:
                start_time = time.time()
                metrics = await self.collect_docker_metrics()

                for metric in metrics:
                    await self.collect_metric(
                        metric.metric_name,
                        metric.value,
                        metric.labels,
                        metric.source_service,
                        metric.tags
                    )

                collection_time = time.time() - start_time
                self.logger.debug(f"Collected {len(metrics)} Docker metrics in {collection_time:.2f}s")

                await asyncio.sleep(self.config["collection"]["default_interval"])

            except Exception as e:
                self.logger.error(f"Error in Docker collection loop: {e}")
                await asyncio.sleep(60)

    async def _system_collection_loop(self):
        """Background loop for system metrics collection."""
        while True:
            try:
                metrics = await self.collect_system_metrics()

                for metric in metrics:
                    await self.collect_metric(
                        metric.metric_name,
                        metric.value,
                        metric.labels,
                        metric.source_service,
                        metric.tags
                    )

                await asyncio.sleep(60)  # System metrics every minute

            except Exception as e:
                self.logger.error(f"Error in system collection loop: {e}")
                await asyncio.sleep(60)

    async def _application_collection_loop(self):
        """Background loop for application metrics collection."""
        services = [
            "dm-crawler", "crypto-intel", "reputation-analyzer", "economics-processor",
            "tactical-intel", "defense-automation", "opsec-enforcer", "intel-fusion",
            "autonomous-coordinator", "adaptive-learning", "resource-manager", "knowledge-evolution"
        ]

        while True:
            try:
                for service in services:
                    metrics = await self.collect_application_metrics(service, "/metrics")

                    for metric in metrics:
                        await self.collect_metric(
                            metric.metric_name,
                            metric.value,
                            metric.labels,
                            metric.source_service,
                            metric.tags
                        )

                await asyncio.sleep(self.config["collection"]["default_interval"])

            except Exception as e:
                self.logger.error(f"Error in application collection loop: {e}")
                await asyncio.sleep(60)

    async def _aggregation_loop(self):
        """Background loop for metrics aggregation."""
        while True:
            try:
                for interval in self.config["aggregation"]["aggregation_intervals"]:
                    for metric_name in self.metric_definitions:
                        aggregated = await self.aggregate_metrics(metric_name, interval)
                        if aggregated:
                            self.aggregated_metrics[metric_name].append(aggregated)

                            # Limit aggregated metrics history
                            if len(self.aggregated_metrics[metric_name]) > 1000:
                                self.aggregated_metrics[metric_name] = self.aggregated_metrics[metric_name][-1000:]

                await asyncio.sleep(300)  # Aggregate every 5 minutes

            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(300)

    async def _cleanup_loop(self):
        """Background loop for metrics cleanup."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.cleanup_old_metrics()

            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)

    async def shutdown(self):
        """Graceful shutdown of metrics collector."""
        # Cancel all collection tasks
        for task_name, task in self.collection_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                self.logger.info(f"Cancelled {task_name} collection task")

        # Close connections
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()
        if self.postgres_pool:
            await self.postgres_pool.close()
        if self.influx_client:
            self.influx_client.close()

        self.logger.info("Metrics collector shutdown completed")


async def main():
    """Main entry point for metrics collector."""
    collector = MetricsCollector()

    try:
        await collector.initialize()
        await collector.start_collection_tasks()

        # Keep running
        while True:
            await asyncio.sleep(60)

    except KeyboardInterrupt:
        logging.info("Received shutdown signal")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
    finally:
        await collector.shutdown()


if __name__ == "__main__":
    asyncio.run(main())