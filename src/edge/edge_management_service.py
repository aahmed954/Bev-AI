import os
"""
Edge Management Service for BEV OSINT Framework

Comprehensive edge computing management service that orchestrates
edge computing network, model synchronization, geographic routing,
and integration with multiplexing and caching systems.
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
from aiohttp import web
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest

# Import edge computing components
from .edge_compute_network import EdgeComputeNetwork, EdgeRequest, EdgeResponse, RequestType
from .edge_node_manager import EdgeNodeManager, NodeConfiguration, NodeStatus
from .model_synchronizer import ModelSynchronizer, ModelVersion, SyncPriority
from .geo_router import GeoRouter, RoutingRequest, RoutingStrategy

class ServiceStatus(Enum):
    """Edge management service status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    STOPPED = "stopped"

@dataclass
class EdgeManagementConfig:
    """Configuration for edge management service"""
    service_port: int = 8080
    admin_port: int = 8081
    metrics_port: int = 9090
    log_level: str = "INFO"
    enable_auto_scaling: bool = True
    enable_model_sync: bool = True
    enable_geo_routing: bool = True
    health_check_interval: int = 30
    performance_optimization_interval: int = 300
    max_concurrent_requests: int = 1000

class EdgeManagementService:
    """
    Edge Management Service

    Central orchestration service for the edge computing network that manages:
    - Edge compute network coordination
    - Model synchronization across regions
    - Geographic routing optimization
    - Integration with multiplexing and caching
    - Performance monitoring and optimization
    - Auto-scaling and load balancing
    """

    def __init__(self, config: EdgeManagementConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.status = ServiceStatus.INITIALIZING

        # Core components
        self.edge_network: Optional[EdgeComputeNetwork] = None
        self.model_synchronizer: Optional[ModelSynchronizer] = None
        self.geo_router: Optional[GeoRouter] = None
        self.node_managers: Dict[str, EdgeNodeManager] = {}

        # Service state
        self.active_requests: Dict[str, EdgeRequest] = {}
        self.request_queue = asyncio.Queue(maxsize=self.config.max_concurrent_requests)
        self.performance_metrics: Dict[str, float] = {}

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.service_requests_total = Counter(
            'edge_service_requests_total',
            'Total service requests',
            ['request_type', 'region', 'status'],
            registry=self.registry
        )
        self.service_latency_histogram = Histogram(
            'edge_service_latency_seconds',
            'Service request latency',
            ['request_type', 'region'],
            registry=self.registry
        )
        self.active_requests_gauge = Gauge(
            'edge_service_active_requests',
            'Currently active requests',
            registry=self.registry
        )
        self.service_health_gauge = Gauge(
            'edge_service_health',
            'Service health score',
            registry=self.registry
        )

        # Web applications
        self.app: Optional[web.Application] = None
        self.admin_app: Optional[web.Application] = None
        self.metrics_app: Optional[web.Application] = None

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize edge management service"""
        try:
            self.logger.info("Initializing Edge Management Service")

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=60)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                user="postgres",
                password=os.getenv('DB_PASSWORD', 'dev_password'),
                database="bev_osint",
                min_size=5,
                max_size=20
            )

            # Create database tables
            await self._create_tables()

            # Initialize core components
            await self._initialize_components()

            # Setup web services
            await self._setup_web_services()

            # Start background tasks
            await self._start_background_tasks()

            self.status = ServiceStatus.RUNNING
            self.logger.info("Edge Management Service initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Edge Management Service: {e}")
            self.status = ServiceStatus.STOPPED
            raise

    async def _create_tables(self):
        """Create database tables for edge management"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_service_requests (
                    request_id VARCHAR(64) PRIMARY KEY,
                    request_type VARCHAR(32) NOT NULL,
                    client_ip INET NOT NULL,
                    payload JSONB NOT NULL,
                    priority INTEGER NOT NULL,
                    max_latency_ms INTEGER NOT NULL,
                    model_preference VARCHAR(32),
                    routing_strategy VARCHAR(32),
                    selected_region VARCHAR(32),
                    selected_node VARCHAR(64),
                    processing_time_ms INTEGER,
                    total_latency_ms INTEGER,
                    cache_hit BOOLEAN,
                    success BOOLEAN,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_service_metrics (
                    metric_name VARCHAR(64) NOT NULL,
                    metric_value DECIMAL(10,2) NOT NULL,
                    region VARCHAR(32),
                    node_id VARCHAR(64),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (metric_name, timestamp)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_service_events (
                    event_id SERIAL PRIMARY KEY,
                    event_type VARCHAR(32) NOT NULL,
                    event_data JSONB NOT NULL,
                    severity VARCHAR(16) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def _initialize_components(self):
        """Initialize core edge computing components"""
        try:
            # Initialize Edge Compute Network
            self.edge_network = EdgeComputeNetwork()
            await self.edge_network.initialize()

            # Initialize Model Synchronizer
            if self.config.enable_model_sync:
                self.model_synchronizer = ModelSynchronizer()
                await self.model_synchronizer.initialize()

            # Initialize Geographic Router
            if self.config.enable_geo_routing:
                self.geo_router = GeoRouter()
                await self.geo_router.initialize()

            self.logger.info("Core components initialized successfully")

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise

    async def _setup_web_services(self):
        """Setup web services for API and admin interfaces"""
        # Main service API
        self.app = web.Application()
        self.app.router.add_post('/api/v1/process', self._handle_process_request)
        self.app.router.add_get('/api/v1/status', self._handle_service_status)
        self.app.router.add_get('/api/v1/regions', self._handle_list_regions)
        self.app.router.add_get('/api/v1/models', self._handle_list_models)
        self.app.router.add_post('/api/v1/sync_model', self._handle_sync_model)
        self.app.router.add_get('/health', self._handle_health)

        # Admin interface
        self.admin_app = web.Application()
        self.admin_app.router.add_get('/admin/dashboard', self._handle_admin_dashboard)
        self.admin_app.router.add_get('/admin/nodes', self._handle_admin_nodes)
        self.admin_app.router.add_post('/admin/scale', self._handle_admin_scale)
        self.admin_app.router.add_post('/admin/maintenance', self._handle_admin_maintenance)
        self.admin_app.router.add_get('/admin/metrics', self._handle_admin_metrics)

        # Metrics interface
        self.metrics_app = web.Application()
        self.metrics_app.router.add_get('/metrics', self._handle_prometheus_metrics)

        # Start web servers
        await self._start_web_servers()

    async def _start_web_servers(self):
        """Start web servers for API, admin, and metrics"""
        # Start main service API
        service_runner = web.AppRunner(self.app)
        await service_runner.setup()
        service_site = web.TCPSite(service_runner, '0.0.0.0', self.config.service_port)
        await service_site.start()

        # Start admin interface
        admin_runner = web.AppRunner(self.admin_app)
        await admin_runner.setup()
        admin_site = web.TCPSite(admin_runner, '0.0.0.0', self.config.admin_port)
        await admin_site.start()

        # Start metrics interface
        metrics_runner = web.AppRunner(self.metrics_app)
        await metrics_runner.setup()
        metrics_site = web.TCPSite(metrics_runner, '0.0.0.0', self.config.metrics_port)
        await metrics_site.start()

        self.logger.info(f"Web servers started - Service: {self.config.service_port}, "
                        f"Admin: {self.config.admin_port}, Metrics: {self.config.metrics_port}")

    async def _start_background_tasks(self):
        """Start background monitoring and optimization tasks"""
        asyncio.create_task(self._request_processor())
        asyncio.create_task(self._health_monitor())
        asyncio.create_task(self._performance_optimizer())
        asyncio.create_task(self._metrics_collector())

        if self.config.enable_auto_scaling:
            asyncio.create_task(self._auto_scaler())

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process edge computing request through the full pipeline"""
        start_time = time.time()
        request_id = request_data.get("request_id", f"edge_{int(time.time() * 1000)}")

        try:
            # Create edge request
            edge_request = EdgeRequest(
                request_id=request_id,
                request_type=RequestType(request_data.get("request_type", "inference")),
                payload=request_data.get("payload", {}),
                priority=request_data.get("priority", 1),
                max_latency_ms=request_data.get("max_latency_ms", 5000),
                client_ip=request_data.get("client_ip", "127.0.0.1"),
                client_region=request_data.get("client_region"),
                timestamp=datetime.utcnow(),
                model_preference=request_data.get("model_preference"),
                cache_key=request_data.get("cache_key")
            )

            # Store in active requests
            self.active_requests[request_id] = edge_request
            self.active_requests_gauge.set(len(self.active_requests))

            # Geographic routing (if enabled)
            selected_region = None
            if self.config.enable_geo_routing and self.geo_router:
                routing_request = RoutingRequest(
                    request_id=request_id,
                    client_ip=edge_request.client_ip,
                    client_location=None,
                    request_type=edge_request.request_type.value,
                    model_preference=edge_request.model_preference,
                    max_latency_ms=edge_request.max_latency_ms,
                    priority=edge_request.priority,
                    timestamp=edge_request.timestamp
                )

                routing_result = await self.geo_router.route_request(
                    routing_request,
                    RoutingStrategy(request_data.get("routing_strategy", "hybrid"))
                )

                if routing_result.selected_node:
                    selected_region = routing_result.selected_region
                    edge_request.client_region = selected_region

            # Process through edge network
            response = await self.edge_network.process_request(edge_request)

            # Integration with multiplexing system
            await self._integrate_with_multiplexing(edge_request, response)

            total_time = (time.time() - start_time) * 1000

            # Store results
            await self._store_request_result(edge_request, response, total_time)

            # Update metrics
            self.service_requests_total.labels(
                request_type=edge_request.request_type.value,
                region=response.region,
                status="success" if response.success else "failed"
            ).inc()

            self.service_latency_histogram.labels(
                request_type=edge_request.request_type.value,
                region=response.region
            ).observe(total_time / 1000.0)

            # Prepare response
            result = {
                "request_id": request_id,
                "success": response.success,
                "region": response.region,
                "model_used": response.model_used,
                "processing_time_ms": response.processing_time_ms,
                "total_latency_ms": int(total_time),
                "cache_hit": response.cache_hit,
                "response_data": response.response_data,
                "timestamp": response.timestamp.isoformat()
            }

            if not response.success:
                result["error_message"] = response.error_message

            return result

        except Exception as e:
            total_time = (time.time() - start_time) * 1000

            self.logger.error(f"Request processing failed for {request_id}: {e}")

            # Update error metrics
            self.service_requests_total.labels(
                request_type=request_data.get("request_type", "unknown"),
                region="error",
                status="failed"
            ).inc()

            return {
                "request_id": request_id,
                "success": False,
                "error_message": str(e),
                "total_latency_ms": int(total_time),
                "timestamp": datetime.utcnow().isoformat()
            }

        finally:
            # Cleanup active requests
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            self.active_requests_gauge.set(len(self.active_requests))

    async def _integrate_with_multiplexing(self, request: EdgeRequest, response: EdgeResponse):
        """Integrate with request multiplexing system"""
        try:
            # Send result to multiplexing system for request coordination
            multiplex_data = {
                "request_id": request.request_id,
                "edge_region": response.region,
                "processing_time_ms": response.processing_time_ms,
                "success": response.success,
                "cache_hit": response.cache_hit,
                "timestamp": response.timestamp.isoformat()
            }

            async with self.session.post(
                "http://172.30.0.42:8080/api/edge_result",
                json=multiplex_data
            ) as resp:
                if resp.status != 200:
                    self.logger.warning(f"Failed to notify multiplexing system: {resp.status}")

        except Exception as e:
            self.logger.warning(f"Multiplexing integration failed: {e}")

    async def _store_request_result(self, request: EdgeRequest, response: EdgeResponse, total_time: float):
        """Store request result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO edge_service_requests
                    (request_id, request_type, client_ip, payload, priority, max_latency_ms,
                     model_preference, selected_region, selected_node, processing_time_ms,
                     total_latency_ms, cache_hit, success, error_message, timestamp, completed_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """, request.request_id, request.request_type.value, request.client_ip,
                json.dumps(request.payload), request.priority, request.max_latency_ms,
                request.model_preference, response.region, response.region,  # Using region as node for now
                response.processing_time_ms, int(total_time), response.cache_hit,
                response.success, response.error_message, request.timestamp, datetime.utcnow())

        except Exception as e:
            self.logger.error(f"Failed to store request result: {e}")

    async def sync_model_to_regions(self, model_name: str, version: str, regions: List[str], priority: str = "normal") -> str:
        """Synchronize model to specified regions"""
        if not self.model_synchronizer:
            raise Exception("Model synchronization is disabled")

        try:
            sync_priority = SyncPriority.NORMAL
            if priority.lower() == "high":
                sync_priority = SyncPriority.HIGH
            elif priority.lower() == "critical":
                sync_priority = SyncPriority.CRITICAL
            elif priority.lower() == "low":
                sync_priority = SyncPriority.LOW

            task_id = await self.model_synchronizer.sync_model_to_regions(
                model_name, version, regions, sync_priority
            )

            # Log event
            await self._log_event("model_sync_started", {
                "task_id": task_id,
                "model_name": model_name,
                "version": version,
                "regions": regions,
                "priority": priority
            }, "info")

            return task_id

        except Exception as e:
            self.logger.error(f"Model synchronization failed: {e}")
            raise

    async def _request_processor(self):
        """Background request processor"""
        while True:
            try:
                # Process queued requests
                if not self.request_queue.empty():
                    request_data = await self.request_queue.get()
                    await self.process_request(request_data)
                    self.request_queue.task_done()
                else:
                    await asyncio.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Request processor error: {e}")
                await asyncio.sleep(1)

    async def _health_monitor(self):
        """Monitor overall service health"""
        while True:
            try:
                await self._check_service_health()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.config.health_check_interval)

    async def _check_service_health(self):
        """Check overall service health"""
        try:
            health_score = 1.0

            # Check edge network health
            if self.edge_network:
                network_status = await self.edge_network.get_network_status()
                if network_status["active_regions"] == 0:
                    health_score *= 0.0  # No active regions
                elif network_status["active_regions"] < network_status["total_regions"] * 0.5:
                    health_score *= 0.5  # Less than 50% regions active

            # Check component health
            if self.model_synchronizer:
                sync_status = await self.model_synchronizer.get_sync_status()
                failed_tasks = sync_status.get("tasks_by_status", {}).get("failed", 0)
                total_tasks = sync_status.get("total_tasks", 1)
                if failed_tasks / total_tasks > 0.1:  # More than 10% failed
                    health_score *= 0.8

            if self.geo_router:
                router_stats = await self.geo_router.get_routing_statistics()
                success_rate = router_stats.get("recent_performance", {}).get("success_rate", 1.0)
                health_score *= success_rate

            # Update health metrics
            self.service_health_gauge.set(health_score)

            # Update service status
            if health_score >= 0.9:
                self.status = ServiceStatus.RUNNING
            elif health_score >= 0.5:
                self.status = ServiceStatus.DEGRADED
            else:
                self.status = ServiceStatus.DEGRADED

            self.logger.debug(f"Service health score: {health_score:.2f}")

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.service_health_gauge.set(0.0)

    async def _performance_optimizer(self):
        """Optimize service performance"""
        while True:
            try:
                await self._optimize_performance()
                await asyncio.sleep(self.config.performance_optimization_interval)
            except Exception as e:
                self.logger.error(f"Performance optimization error: {e}")
                await asyncio.sleep(self.config.performance_optimization_interval)

    async def _optimize_performance(self):
        """Optimize overall performance based on metrics"""
        try:
            # Analyze request patterns
            async with self.db_pool.acquire() as conn:
                # Get recent performance metrics
                metrics = await conn.fetchrow("""
                    SELECT
                        AVG(total_latency_ms) as avg_latency,
                        COUNT(*) as total_requests,
                        COUNT(*) FILTER (WHERE success = TRUE) as successful_requests,
                        COUNT(*) FILTER (WHERE cache_hit = TRUE) as cache_hits
                    FROM edge_service_requests
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                """)

                if metrics and metrics['total_requests'] > 0:
                    success_rate = metrics['successful_requests'] / metrics['total_requests']
                    cache_hit_rate = metrics['cache_hits'] / metrics['total_requests']
                    avg_latency = float(metrics['avg_latency'] or 0)

                    self.performance_metrics.update({
                        "success_rate": success_rate,
                        "cache_hit_rate": cache_hit_rate,
                        "avg_latency_ms": avg_latency
                    })

                    # Optimization decisions
                    if success_rate < 0.95:
                        await self._trigger_healing_actions()

                    if avg_latency > 2000:  # > 2 seconds
                        await self._optimize_latency()

                    if cache_hit_rate < 0.3:  # < 30% cache hit rate
                        await self._optimize_caching()

        except Exception as e:
            self.logger.error(f"Performance optimization failed: {e}")

    async def _trigger_healing_actions(self):
        """Trigger self-healing actions for low success rates"""
        try:
            # Restart unhealthy nodes
            if self.edge_network:
                network_status = await self.edge_network.get_network_status()
                for region, region_info in network_status["regions"].items():
                    if not region_info["active"] or region_info["current_load"] > 0.9:
                        await self._restart_region_nodes(region)

            await self._log_event("healing_action", {
                "action": "restart_unhealthy_nodes",
                "trigger": "low_success_rate"
            }, "warning")

        except Exception as e:
            self.logger.error(f"Healing actions failed: {e}")

    async def _optimize_latency(self):
        """Optimize system latency"""
        try:
            # Update routing weights for latency optimization
            if self.geo_router:
                self.geo_router.latency_weight = min(0.6, self.geo_router.latency_weight + 0.1)
                self.geo_router.load_weight = max(0.2, self.geo_router.load_weight - 0.05)

            await self._log_event("latency_optimization", {
                "action": "adjust_routing_weights",
                "new_latency_weight": self.geo_router.latency_weight if self.geo_router else 0
            }, "info")

        except Exception as e:
            self.logger.error(f"Latency optimization failed: {e}")

    async def _optimize_caching(self):
        """Optimize caching performance"""
        try:
            # Trigger cache warming for popular requests
            await self._warm_popular_caches()

            await self._log_event("cache_optimization", {
                "action": "warm_popular_caches",
                "trigger": "low_cache_hit_rate"
            }, "info")

        except Exception as e:
            self.logger.error(f"Cache optimization failed: {e}")

    async def _warm_popular_caches(self):
        """Warm caches for popular requests"""
        try:
            # Get popular request patterns
            async with self.db_pool.acquire() as conn:
                popular_requests = await conn.fetch("""
                    SELECT payload, COUNT(*) as request_count
                    FROM edge_service_requests
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                    AND cache_hit = FALSE
                    GROUP BY payload
                    ORDER BY request_count DESC
                    LIMIT 10
                """)

                # Pre-warm these requests
                for request in popular_requests:
                    # This would involve pre-processing common requests
                    # and storing results in cache
                    pass

        except Exception as e:
            self.logger.error(f"Cache warming failed: {e}")

    async def _restart_region_nodes(self, region: str):
        """Restart nodes in a specific region"""
        # Implementation would depend on deployment infrastructure
        self.logger.info(f"Triggering restart for region {region} nodes")

    async def _metrics_collector(self):
        """Collect and store performance metrics"""
        while True:
            try:
                await self._collect_metrics()
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)

    async def _collect_metrics(self):
        """Collect performance metrics from all components"""
        try:
            timestamp = datetime.utcnow()

            # Collect edge network metrics
            if self.edge_network:
                network_status = await self.edge_network.get_network_status()
                await self._store_metric("edge_active_regions", network_status["active_regions"], timestamp)
                await self._store_metric("edge_average_latency", network_status["average_latency"], timestamp)

            # Collect service metrics
            await self._store_metric("service_active_requests", len(self.active_requests), timestamp)
            await self._store_metric("service_health_score", self.service_health_gauge._value._value, timestamp)

            for metric_name, metric_value in self.performance_metrics.items():
                await self._store_metric(f"service_{metric_name}", metric_value, timestamp)

        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")

    async def _store_metric(self, name: str, value: float, timestamp: datetime, region: str = None, node_id: str = None):
        """Store metric in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO edge_service_metrics
                    (metric_name, metric_value, region, node_id, timestamp)
                    VALUES ($1, $2, $3, $4, $5)
                """, name, value, region, node_id, timestamp)

        except Exception as e:
            self.logger.error(f"Failed to store metric {name}: {e}")

    async def _auto_scaler(self):
        """Auto-scaling based on load and performance"""
        while True:
            try:
                await self._check_scaling_needs()
                await asyncio.sleep(120)  # Check every 2 minutes
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(120)

    async def _check_scaling_needs(self):
        """Check if scaling up or down is needed"""
        try:
            # Get current load metrics
            if self.edge_network:
                network_status = await self.edge_network.get_network_status()

                for region, region_info in network_status["regions"].items():
                    load = region_info.get("current_load", 0)

                    # Scale up if load > 80%
                    if load > 0.8:
                        await self._scale_up_region(region)

                    # Scale down if load < 20% for extended period
                    elif load < 0.2:
                        await self._consider_scale_down(region)

        except Exception as e:
            self.logger.error(f"Scaling check failed: {e}")

    async def _scale_up_region(self, region: str):
        """Scale up resources in a region"""
        self.logger.info(f"Scaling up region {region}")
        await self._log_event("auto_scale_up", {"region": region}, "info")

    async def _consider_scale_down(self, region: str):
        """Consider scaling down resources in a region"""
        self.logger.info(f"Considering scale down for region {region}")
        await self._log_event("auto_scale_down_considered", {"region": region}, "info")

    async def _log_event(self, event_type: str, event_data: Dict[str, Any], severity: str):
        """Log service event"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO edge_service_events
                    (event_type, event_data, severity)
                    VALUES ($1, $2, $3)
                """, event_type, json.dumps(event_data), severity)

        except Exception as e:
            self.logger.error(f"Failed to log event: {e}")

    # HTTP Handlers
    async def _handle_process_request(self, request):
        """Handle edge processing request"""
        try:
            data = await request.json()
            result = await self.process_request(data)
            return web.json_response(result)

        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def _handle_service_status(self, request):
        """Handle service status request"""
        try:
            status = {
                "service_status": self.status.value,
                "timestamp": datetime.utcnow().isoformat(),
                "active_requests": len(self.active_requests),
                "performance_metrics": self.performance_metrics,
                "components": {
                    "edge_network": self.edge_network is not None,
                    "model_synchronizer": self.model_synchronizer is not None,
                    "geo_router": self.geo_router is not None
                }
            }

            if self.edge_network:
                status["edge_network_status"] = await self.edge_network.get_network_status()

            return web.json_response(status)

        except Exception as e:
            return web.json_response({
                "error": str(e)
            }, status=500)

    async def _handle_list_regions(self, request):
        """Handle list regions request"""
        try:
            if self.edge_network:
                network_status = await self.edge_network.get_network_status()
                return web.json_response(network_status["regions"])
            else:
                return web.json_response({})

        except Exception as e:
            return web.json_response({
                "error": str(e)
            }, status=500)

    async def _handle_list_models(self, request):
        """Handle list models request"""
        try:
            if self.model_synchronizer:
                sync_status = await self.model_synchronizer.get_sync_status()
                return web.json_response(sync_status)
            else:
                return web.json_response({"models": []})

        except Exception as e:
            return web.json_response({
                "error": str(e)
            }, status=500)

    async def _handle_sync_model(self, request):
        """Handle model synchronization request"""
        try:
            data = await request.json()

            task_id = await self.sync_model_to_regions(
                data["model_name"],
                data["version"],
                data["regions"],
                data.get("priority", "normal")
            )

            return web.json_response({
                "success": True,
                "task_id": task_id
            })

        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def _handle_health(self, request):
        """Handle health check request"""
        health_status = {
            "status": self.status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "health_score": self.service_health_gauge._value._value,
            "active_requests": len(self.active_requests)
        }

        status_code = 200 if self.status in [ServiceStatus.RUNNING, ServiceStatus.DEGRADED] else 503
        return web.json_response(health_status, status=status_code)

    async def _handle_admin_dashboard(self, request):
        """Handle admin dashboard request"""
        # Return HTML dashboard or redirect to dashboard UI
        return web.Response(text="Admin Dashboard - Edge Computing Management", content_type='text/html')

    async def _handle_admin_nodes(self, request):
        """Handle admin nodes management"""
        try:
            if self.edge_network:
                network_status = await self.edge_network.get_network_status()
                return web.json_response(network_status)
            else:
                return web.json_response({"nodes": []})

        except Exception as e:
            return web.json_response({
                "error": str(e)
            }, status=500)

    async def _handle_admin_scale(self, request):
        """Handle admin scaling request"""
        try:
            data = await request.json()
            action = data.get("action")  # "scale_up" or "scale_down"
            region = data.get("region")

            if action == "scale_up":
                await self._scale_up_region(region)
            elif action == "scale_down":
                await self._consider_scale_down(region)

            return web.json_response({
                "success": True,
                "action": action,
                "region": region
            })

        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def _handle_admin_maintenance(self, request):
        """Handle admin maintenance request"""
        try:
            data = await request.json()
            maintenance_type = data.get("type")

            if maintenance_type == "restart_unhealthy":
                await self._trigger_healing_actions()
            elif maintenance_type == "optimize_performance":
                await self._optimize_performance()
            elif maintenance_type == "warm_caches":
                await self._warm_popular_caches()

            return web.json_response({
                "success": True,
                "maintenance_type": maintenance_type
            })

        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def _handle_admin_metrics(self, request):
        """Handle admin metrics request"""
        try:
            # Get recent metrics from database
            async with self.db_pool.acquire() as conn:
                metrics = await conn.fetch("""
                    SELECT metric_name, metric_value, region, node_id, timestamp
                    FROM edge_service_metrics
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                    ORDER BY timestamp DESC
                    LIMIT 1000
                """)

                metrics_data = [
                    {
                        "name": row['metric_name'],
                        "value": float(row['metric_value']),
                        "region": row['region'],
                        "node_id": row['node_id'],
                        "timestamp": row['timestamp'].isoformat()
                    }
                    for row in metrics
                ]

                return web.json_response({"metrics": metrics_data})

        except Exception as e:
            return web.json_response({
                "error": str(e)
            }, status=500)

    async def _handle_prometheus_metrics(self, request):
        """Handle Prometheus metrics request"""
        metrics_data = generate_latest(self.registry)
        return web.Response(text=metrics_data.decode('utf-8'), content_type='text/plain')

    async def cleanup(self):
        """Cleanup service resources"""
        try:
            self.status = ServiceStatus.STOPPED

            # Cleanup components
            if self.edge_network:
                await self.edge_network.cleanup()

            if self.model_synchronizer:
                await self.model_synchronizer.cleanup()

            if self.geo_router:
                await self.geo_router.cleanup()

            # Close HTTP session
            if self.session:
                await self.session.close()

            # Close database connections
            if self.db_pool:
                await self.db_pool.close()

            self.logger.info("Edge Management Service cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

# Example usage
async def main():
    """Example usage of EdgeManagementService"""
    config = EdgeManagementConfig(
        service_port=8080,
        admin_port=8081,
        metrics_port=9090,
        log_level="INFO",
        enable_auto_scaling=True,
        enable_model_sync=True,
        enable_geo_routing=True
    )

    service = EdgeManagementService(config)

    try:
        await service.initialize()

        # Example request
        request_data = {
            "request_type": "osint_analysis",
            "payload": {
                "target": "example.com",
                "analysis_type": "domain_intelligence"
            },
            "priority": 1,
            "max_latency_ms": 5000,
            "client_ip": "192.168.1.100",
            "model_preference": "llama-3-8b"
        }

        result = await service.process_request(request_data)
        print(f"Processing result: {result}")

        # Keep service running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await service.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())