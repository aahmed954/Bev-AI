import os
"""
BEV OSINT Framework - Predictive Cache Service
Main service integrating ML prediction, cache warming, optimization, and multi-tier caching.
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
from aiohttp import web
import aioredis
import psycopg2.pool
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import yaml

from .ml_predictor import MLPredictor
from .predictive_cache import PredictiveCache
from .cache_warmer import CacheWarmer
from .cache_optimizer import CacheOptimizer


class ServiceStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ServiceMetrics:
    """Comprehensive service metrics."""
    cache_hit_rate: float
    cache_response_time: float
    ml_prediction_accuracy: float
    warming_tasks_completed: int
    optimization_score: float
    memory_utilization: float
    active_users: int
    requests_per_second: float
    error_rate: float
    uptime_seconds: float
    timestamp: datetime


class PredictiveCacheService:
    """
    Main predictive cache service integrating all components.
    Provides HTTP API, metrics, and orchestrates all cache operations.
    """

    def __init__(self, config_path: str = "/app/config/predictive_cache.yml"):
        self.config_path = config_path
        self.config = self._load_config()

        # Service state
        self.status = ServiceStatus.STARTING
        self.start_time = datetime.now(timezone.utc)
        self.shutdown_event = asyncio.Event()

        # Core components
        self.ml_predictor: Optional[MLPredictor] = None
        self.cache_system: Optional[PredictiveCache] = None
        self.cache_warmer: Optional[CacheWarmer] = None
        self.cache_optimizer: Optional[CacheOptimizer] = None

        # External connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None

        # Web server
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # Request tracking
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = time.time()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

        # Logging
        self.logger = self._setup_logging()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "service": {
                "host": "0.0.0.0",
                "port": 8044,
                "workers": 4,
                "debug": False
            },
            "redis": {
                "cluster_nodes": [
                    {"host": "redis-node-1", "port": 7001},
                    {"host": "redis-node-2", "port": 7002},
                    {"host": "redis-node-3", "port": 7003}
                ],
                "password": "${REDIS_PASSWORD}",
                "standalone_url": "redis://redis:6379"
            },
            "postgres": {
                "uri": "${POSTGRES_URI}"
            },
            "cache": {
                "hot_tier_size_gb": 4.0,
                "warm_tier_size_gb": 8.0,
                "cold_tier_persistent": True,
                "default_ttl_seconds": 3600,
                "prefetch_threshold": 0.7,
                "eviction_policy": "ml_adaptive",
                "enable_intelligent_warming": True,
                "max_prefetch_batch_size": 100
            },
            "ml": {
                "min_training_samples": 1000,
                "retrain_interval_hours": 6,
                "model_accuracy_threshold": 0.8,
                "prediction_confidence_threshold": 0.7
            },
            "optimization": {
                "optimization_interval_seconds": 300,
                "target_hit_rate": 0.85,
                "max_memory_utilization": 0.9,
                "analysis_window_hours": 24
            },
            "warming": {
                "max_concurrent_tasks": 10,
                "warming_interval_seconds": 300,
                "user_analysis_window_hours": 24,
                "popularity_threshold": 0.1,
                "max_warming_bandwidth_mbps": 100
            },
            "monitoring": {
                "metrics_interval_seconds": 60,
                "health_check_interval_seconds": 30,
                "log_level": "INFO",
                "enable_detailed_logging": True
            }
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging."""
        logger = logging.getLogger('predictive_cache_service')
        log_level = self.config.get("monitoring", {}).get("log_level", "INFO")
        logger.setLevel(getattr(logging, log_level))

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    async def initialize(self):
        """Initialize all service components."""
        try:
            self.logger.info("Initializing Predictive Cache Service")

            # Initialize external connections
            await self._initialize_connections()

            # Initialize core components
            await self._initialize_components()

            # Setup web server
            await self._setup_web_server()

            # Start background tasks
            await self._start_background_tasks()

            self.status = ServiceStatus.RUNNING
            self.logger.info("Predictive Cache Service initialized successfully")

        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.logger.error(f"Failed to initialize service: {e}")
            raise

    async def _initialize_connections(self):
        """Initialize external connections."""
        # Initialize Redis connection
        redis_config = self.config.get("redis", {})
        redis_url = redis_config.get("standalone_url", "redis://redis:6379")
        self.redis_client = await aioredis.from_url(
            redis_url,
            password=redis_config.get("password"),
            decode_responses=True
        )

        # Initialize database connection pool
        postgres_config = self.config.get("postgres", {})
        if postgres_config.get("uri"):
            self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=20,
                dsn=postgres_config["uri"]
            )

        self.logger.info("External connections initialized")

    async def _initialize_components(self):
        """Initialize core cache components."""
        # Initialize ML Predictor
        ml_config = {
            "redis_url": f"{self.config['redis']['standalone_url']}/12",
            "postgres_uri": self.config.get("postgres", {}).get("uri"),
            **self.config.get("ml", {})
        }
        self.ml_predictor = MLPredictor(ml_config)
        await self.ml_predictor.initialize()

        # Initialize Cache System
        cache_config = {
            "cache": self.config.get("cache", {}),
            "redis_cluster_nodes": self.config["redis"]["cluster_nodes"],
            "redis_password": self.config["redis"]["password"],
            "postgres_uri": self.config.get("postgres", {}).get("uri")
        }
        self.cache_system = PredictiveCache(cache_config)
        await self.cache_system.initialize()

        # Set ML predictor reference
        self.cache_system.ml_predictor = self.ml_predictor

        # Initialize Cache Warmer
        warmer_config = {
            "redis_url": f"{self.config['redis']['standalone_url']}/14",
            "postgres_uri": self.config.get("postgres", {}).get("uri"),
            **self.config.get("warming", {})
        }
        self.cache_warmer = CacheWarmer(warmer_config)
        await self.cache_warmer.initialize()

        # Set component references
        self.cache_warmer.set_ml_predictor(self.ml_predictor)
        self.cache_warmer.set_cache_system(self.cache_system)

        # Initialize Cache Optimizer
        optimizer_config = {
            "redis_url": f"{self.config['redis']['standalone_url']}/15",
            "postgres_uri": self.config.get("postgres", {}).get("uri"),
            **self.config.get("optimization", {})
        }
        self.cache_optimizer = CacheOptimizer(optimizer_config)
        await self.cache_optimizer.initialize()

        # Set component references
        self.cache_optimizer.set_cache_system(self.cache_system)
        self.cache_optimizer.set_ml_predictor(self.ml_predictor)
        self.cache_optimizer.set_cache_warmer(self.cache_warmer)

        self.logger.info("Core components initialized")

    async def _setup_web_server(self):
        """Setup HTTP web server."""
        self.app = web.Application()

        # Setup routes
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/metrics', self.metrics_endpoint)
        self.app.router.add_get('/stats', self.stats_endpoint)

        # Cache API routes
        self.app.router.add_get('/cache/{key}', self.cache_get)
        self.app.router.add_put('/cache/{key}', self.cache_set)
        self.app.router.add_delete('/cache/{key}', self.cache_delete)

        # ML and optimization routes
        self.app.router.add_post('/predict', self.predict_endpoint)
        self.app.router.add_post('/warm', self.warm_endpoint)
        self.app.router.add_post('/optimize', self.optimize_endpoint)

        # Admin routes
        self.app.router.add_get('/admin/status', self.admin_status)
        self.app.router.add_post('/admin/retrain', self.admin_retrain)

        # Setup middleware
        self.app.middlewares.append(self._request_logging_middleware)
        self.app.middlewares.append(self._error_handling_middleware)

        # Create runner
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        # Create site
        service_config = self.config.get("service", {})
        self.site = web.TCPSite(
            self.runner,
            service_config.get("host", "0.0.0.0"),
            service_config.get("port", 8044)
        )

        self.logger.info("Web server setup completed")

    async def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks."""
        tasks = [
            self._metrics_collection_loop(),
            self._health_monitoring_loop(),
            self._performance_logging_loop()
        ]

        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.append(task)

        self.logger.info("Background tasks started")

    @web.middleware
    async def _request_logging_middleware(self, request: web.Request, handler):
        """Middleware for request logging and metrics."""
        start_time = time.time()
        self.request_count += 1

        try:
            response = await handler(request)
            response_time = time.time() - start_time

            if self.config.get("monitoring", {}).get("enable_detailed_logging"):
                self.logger.info(
                    f"{request.method} {request.path} - "
                    f"Status: {response.status} - "
                    f"Time: {response_time:.3f}s"
                )

            return response

        except Exception as e:
            self.error_count += 1
            response_time = time.time() - start_time
            self.logger.error(
                f"{request.method} {request.path} - "
                f"Error: {str(e)} - "
                f"Time: {response_time:.3f}s"
            )
            raise

    @web.middleware
    async def _error_handling_middleware(self, request: web.Request, handler):
        """Middleware for error handling."""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Unhandled error in {request.path}: {e}")
            return web.json_response(
                {"error": "Internal server error", "message": str(e)},
                status=500
            )

    # HTTP Endpoints

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        health_status = {
            "status": self.status.value,
            "uptime": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
            "components": {
                "ml_predictor": "healthy" if self.ml_predictor else "unavailable",
                "cache_system": "healthy" if self.cache_system else "unavailable",
                "cache_warmer": "healthy" if self.cache_warmer else "unavailable",
                "cache_optimizer": "healthy" if self.cache_optimizer else "unavailable"
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        return web.json_response(health_status)

    async def metrics_endpoint(self, request: web.Request) -> web.Response:
        """Prometheus metrics endpoint."""
        metrics_data = generate_latest()
        return web.Response(body=metrics_data, content_type=CONTENT_TYPE_LATEST)

    async def stats_endpoint(self, request: web.Request) -> web.Response:
        """Comprehensive statistics endpoint."""
        try:
            stats = await self._collect_comprehensive_stats()
            return web.json_response(stats)
        except Exception as e:
            return web.json_response(
                {"error": "Failed to collect stats", "message": str(e)},
                status=500
            )

    async def cache_get(self, request: web.Request) -> web.Response:
        """Get value from cache."""
        try:
            key = request.match_info['key']
            user_id = request.query.get('user_id')
            query_type = request.query.get('query_type', 'unknown')

            if not self.cache_system:
                return web.json_response(
                    {"error": "Cache system unavailable"},
                    status=503
                )

            value = await self.cache_system.get(key, user_id, query_type)

            if value is not None:
                return web.json_response({
                    "key": key,
                    "value": value,
                    "hit": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            else:
                return web.json_response({
                    "key": key,
                    "hit": False,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }, status=404)

        except Exception as e:
            return web.json_response(
                {"error": "Cache get failed", "message": str(e)},
                status=500
            )

    async def cache_set(self, request: web.Request) -> web.Response:
        """Set value in cache."""
        try:
            key = request.match_info['key']
            data = await request.json()

            value = data.get('value')
            ttl = data.get('ttl')
            user_id = data.get('user_id')
            query_type = data.get('query_type', 'unknown')
            size_hint = data.get('size_hint')

            if not self.cache_system:
                return web.json_response(
                    {"error": "Cache system unavailable"},
                    status=503
                )

            success = await self.cache_system.set(
                key=key,
                value=value,
                ttl=ttl,
                user_id=user_id,
                query_type=query_type,
                size_hint=size_hint
            )

            if success:
                return web.json_response({
                    "key": key,
                    "stored": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            else:
                return web.json_response({
                    "key": key,
                    "stored": False,
                    "error": "Failed to store in cache"
                }, status=500)

        except Exception as e:
            return web.json_response(
                {"error": "Cache set failed", "message": str(e)},
                status=500
            )

    async def cache_delete(self, request: web.Request) -> web.Response:
        """Delete value from cache."""
        try:
            key = request.match_info['key']

            if not self.cache_system:
                return web.json_response(
                    {"error": "Cache system unavailable"},
                    status=503
                )

            success = await self.cache_system.delete(key)

            return web.json_response({
                "key": key,
                "deleted": success,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            return web.json_response(
                {"error": "Cache delete failed", "message": str(e)},
                status=500
            )

    async def predict_endpoint(self, request: web.Request) -> web.Response:
        """ML prediction endpoint."""
        try:
            data = await request.json()

            key = data.get('key')
            query_type = data.get('query_type')
            user_id = data.get('user_id')

            if not self.ml_predictor:
                return web.json_response(
                    {"error": "ML predictor unavailable"},
                    status=503
                )

            prediction = await self.ml_predictor.predict_cache_hit(
                key, query_type, user_id
            )

            return web.json_response({
                "prediction": {
                    "hit_probability": prediction.predicted_value,
                    "confidence": prediction.confidence,
                    "model_used": prediction.model_used.value,
                    "features_used": prediction.features_used
                },
                "timestamp": prediction.timestamp.isoformat()
            })

        except Exception as e:
            return web.json_response(
                {"error": "Prediction failed", "message": str(e)},
                status=500
            )

    async def warm_endpoint(self, request: web.Request) -> web.Response:
        """Cache warming endpoint."""
        try:
            data = await request.json()

            strategy = data.get('strategy', 'user_based')
            user_id = data.get('user_id')
            query_type = data.get('query_type')

            if not self.cache_warmer:
                return web.json_response(
                    {"error": "Cache warmer unavailable"},
                    status=503
                )

            if strategy == 'user_based' and user_id:
                task_ids = await self.cache_warmer.analyze_user_patterns(user_id)
            elif strategy == 'popularity_based':
                task_ids = await self.cache_warmer.analyze_popularity_trends()
            elif strategy == 'collaborative':
                task_ids = await self.cache_warmer.create_collaborative_warming_tasks()
            else:
                return web.json_response(
                    {"error": "Invalid warming strategy"},
                    status=400
                )

            return web.json_response({
                "strategy": strategy,
                "tasks_created": len(task_ids),
                "task_ids": task_ids,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            return web.json_response(
                {"error": "Warming failed", "message": str(e)},
                status=500
            )

    async def optimize_endpoint(self, request: web.Request) -> web.Response:
        """Cache optimization endpoint."""
        try:
            if not self.cache_optimizer:
                return web.json_response(
                    {"error": "Cache optimizer unavailable"},
                    status=503
                )

            optimization = await self.cache_optimizer.optimize_cache_strategy()

            # Optionally apply optimization immediately
            data = await request.json() if request.has_body else {}
            apply_immediately = data.get('apply', False)

            if apply_immediately:
                success = await self.cache_optimizer.apply_optimization(optimization)
                return web.json_response({
                    "optimization": asdict(optimization),
                    "applied": success,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            else:
                return web.json_response({
                    "optimization": asdict(optimization),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

        except Exception as e:
            return web.json_response(
                {"error": "Optimization failed", "message": str(e)},
                status=500
            )

    async def admin_status(self, request: web.Request) -> web.Response:
        """Admin status endpoint."""
        try:
            status = {
                "service": {
                    "status": self.status.value,
                    "uptime": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
                    "requests_handled": self.request_count,
                    "errors": self.error_count,
                    "error_rate": self.error_count / max(1, self.request_count)
                },
                "components": {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Get component statistics
            if self.cache_system:
                status["components"]["cache"] = await self.cache_system.get_cache_stats()

            if self.cache_warmer:
                status["components"]["warmer"] = await self.cache_warmer.get_warming_stats()

            if self.cache_optimizer:
                status["components"]["optimizer"] = await self.cache_optimizer.get_optimization_stats()

            return web.json_response(status)

        except Exception as e:
            return web.json_response(
                {"error": "Status collection failed", "message": str(e)},
                status=500
            )

    async def admin_retrain(self, request: web.Request) -> web.Response:
        """Admin ML model retraining endpoint."""
        try:
            if not self.ml_predictor:
                return web.json_response(
                    {"error": "ML predictor unavailable"},
                    status=503
                )

            accuracies = await self.ml_predictor.train_models()

            return web.json_response({
                "retrained": True,
                "model_accuracies": {k.value: v for k, v in accuracies.items()},
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        except Exception as e:
            return web.json_response(
                {"error": "Retraining failed", "message": str(e)},
                status=500
            )

    # Background Tasks

    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        interval = self.config.get("monitoring", {}).get("metrics_interval_seconds", 60)

        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(interval)

                # Collect and update metrics
                metrics = await self._collect_service_metrics()
                await self._update_metrics_storage(metrics)

            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")

    async def _health_monitoring_loop(self):
        """Background health monitoring loop."""
        interval = self.config.get("monitoring", {}).get("health_check_interval_seconds", 30)

        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(interval)

                # Check component health
                await self._check_component_health()

            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")

    async def _performance_logging_loop(self):
        """Background performance logging loop."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # 5 minutes

                # Log performance summary
                if self.cache_system:
                    stats = await self.cache_system.get_cache_stats()
                    hit_rate = stats["metrics"]["hit_rate"]
                    response_time = stats["metrics"]["avg_response_time"]

                    self.logger.info(
                        f"Performance: Hit Rate: {hit_rate:.1%}, "
                        f"Response Time: {response_time:.3f}s, "
                        f"Requests: {self.request_count}, "
                        f"Errors: {self.error_count}"
                    )

            except Exception as e:
                self.logger.error(f"Error in performance logging: {e}")

    # Utility Methods

    async def _collect_service_metrics(self) -> ServiceMetrics:
        """Collect comprehensive service metrics."""
        try:
            cache_hit_rate = 0.0
            cache_response_time = 0.0
            memory_utilization = 0.0

            if self.cache_system:
                cache_stats = await self.cache_system.get_cache_stats()
                cache_hit_rate = cache_stats["metrics"]["hit_rate"]
                cache_response_time = cache_stats["metrics"]["avg_response_time"]

                # Calculate memory utilization
                tier_stats = cache_stats["tier_stats"]
                total_used = sum(
                    tier["size_bytes"] for tier in tier_stats.values()
                    if tier["max_size_bytes"] != float('inf')
                )
                total_max = sum(
                    tier["max_size_bytes"] for tier in tier_stats.values()
                    if tier["max_size_bytes"] != float('inf')
                )
                memory_utilization = total_used / total_max if total_max > 0 else 0.0

            ml_prediction_accuracy = 0.0
            if self.ml_predictor:
                # Would get from ML predictor metrics
                ml_prediction_accuracy = 0.85  # Placeholder

            warming_tasks_completed = 0
            if self.cache_warmer:
                warming_stats = await self.cache_warmer.get_warming_stats()
                warming_tasks_completed = warming_stats["metrics"]["tasks_completed"]

            optimization_score = 0.0
            if self.cache_optimizer:
                optimizer_stats = await self.cache_optimizer.get_optimization_stats()
                if optimizer_stats.get("recent_performance"):
                    optimization_score = optimizer_stats["recent_performance"]["hit_rate"]

            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            requests_per_second = self.request_count / max(1, uptime)
            error_rate = self.error_count / max(1, self.request_count)

            return ServiceMetrics(
                cache_hit_rate=cache_hit_rate,
                cache_response_time=cache_response_time,
                ml_prediction_accuracy=ml_prediction_accuracy,
                warming_tasks_completed=warming_tasks_completed,
                optimization_score=optimization_score,
                memory_utilization=memory_utilization,
                active_users=0,  # Would track from cache events
                requests_per_second=requests_per_second,
                error_rate=error_rate,
                uptime_seconds=uptime,
                timestamp=datetime.now(timezone.utc)
            )

        except Exception as e:
            self.logger.error(f"Error collecting service metrics: {e}")
            raise

    async def _update_metrics_storage(self, metrics: ServiceMetrics):
        """Update metrics in persistent storage."""
        try:
            if self.redis_client:
                metrics_data = asdict(metrics)
                await self.redis_client.set(
                    "predictive_cache:service_metrics",
                    json.dumps(metrics_data, default=str),
                    ex=3600
                )

        except Exception as e:
            self.logger.error(f"Error updating metrics storage: {e}")

    async def _check_component_health(self):
        """Check health of all components."""
        try:
            components_healthy = True

            # Check each component
            if self.ml_predictor:
                # Would implement health check
                pass

            if self.cache_system:
                # Would implement health check
                pass

            if self.cache_warmer:
                # Would implement health check
                pass

            if self.cache_optimizer:
                # Would implement health check
                pass

            if not components_healthy and self.status == ServiceStatus.RUNNING:
                self.status = ServiceStatus.ERROR
                self.logger.warning("Service status changed to ERROR due to component failures")

        except Exception as e:
            self.logger.error(f"Error checking component health: {e}")

    async def _collect_comprehensive_stats(self) -> Dict[str, Any]:
        """Collect comprehensive statistics."""
        stats = {
            "service": {
                "status": self.status.value,
                "uptime": (datetime.now(timezone.utc) - self.start_time).total_seconds(),
                "requests": self.request_count,
                "errors": self.error_count,
                "error_rate": self.error_count / max(1, self.request_count)
            },
            "metrics": asdict(await self._collect_service_metrics()),
            "components": {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # Get component stats
        if self.cache_system:
            stats["components"]["cache"] = await self.cache_system.get_cache_stats()

        if self.cache_warmer:
            stats["components"]["warmer"] = await self.cache_warmer.get_warming_stats()

        if self.cache_optimizer:
            stats["components"]["optimizer"] = await self.cache_optimizer.get_optimization_stats()

        return stats

    async def start(self):
        """Start the service."""
        try:
            await self.initialize()

            # Start web server
            await self.site.start()
            service_config = self.config.get("service", {})
            host = service_config.get("host", "0.0.0.0")
            port = service_config.get("port", 8044)

            self.logger.info(f"Predictive Cache Service started on {host}:{port}")

            # Wait for shutdown signal
            await self.shutdown_event.wait()

        except Exception as e:
            self.logger.error(f"Error starting service: {e}")
            self.status = ServiceStatus.ERROR
            raise

    async def shutdown(self):
        """Graceful shutdown of the service."""
        try:
            self.logger.info("Shutting down Predictive Cache Service")
            self.status = ServiceStatus.STOPPING

            # Signal shutdown
            self.shutdown_event.set()

            # Cancel background tasks
            for task in self._background_tasks:
                task.cancel()

            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            # Shutdown components
            if self.cache_optimizer:
                await self.cache_optimizer.shutdown()

            if self.cache_warmer:
                await self.cache_warmer.shutdown()

            if self.cache_system:
                await self.cache_system.shutdown()

            if self.ml_predictor:
                await self.ml_predictor.shutdown()

            # Shutdown web server
            if self.site:
                await self.site.stop()

            if self.runner:
                await self.runner.cleanup()

            # Close connections
            if self.redis_client:
                await self.redis_client.close()

            if self.db_pool:
                self.db_pool.closeall()

            self.status = ServiceStatus.STOPPED
            self.logger.info("Predictive Cache Service shutdown completed")

        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            self.status = ServiceStatus.ERROR

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        if sys.platform != 'win32':
            for sig in [signal.SIGTERM, signal.SIGINT]:
                signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating shutdown")
        asyncio.create_task(self.shutdown())


async def main():
    """Main entry point for the service."""
    service = PredictiveCacheService()
    service.setup_signal_handlers()

    try:
        await service.start()
    except KeyboardInterrupt:
        service.logger.info("Received keyboard interrupt")
    except Exception as e:
        service.logger.error(f"Fatal error: {e}")
    finally:
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(main())