import os
"""
Edge Computing Integration for BEV OSINT Framework

Integrates edge computing network with existing multiplexing and caching infrastructure
to provide seamless request routing, processing, and caching across geographic regions.
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
import asyncpg
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Import existing infrastructure components
try:
    from ..pipeline.request_multiplexer import RequestMultiplexer, Request, RequestPriority
    from ..infrastructure.predictive_cache import PredictiveCache
except ImportError:
    # Fallback for standalone testing
    pass

# Import edge computing components
from .edge_compute_network import EdgeComputeNetwork, EdgeRequest, EdgeResponse, RequestType
from .geo_router import GeoRouter, RoutingRequest, RoutingStrategy

class IntegrationMode(Enum):
    """Integration operation modes"""
    EDGE_FIRST = "edge_first"          # Try edge first, fallback to traditional
    TRADITIONAL_FIRST = "traditional_first"  # Try traditional first, fallback to edge
    EDGE_ONLY = "edge_only"            # Only use edge computing
    LOAD_BALANCED = "load_balanced"    # Balance between edge and traditional
    INTELLIGENT = "intelligent"       # AI-driven routing decisions

@dataclass
class IntegrationConfig:
    """Configuration for edge integration"""
    mode: IntegrationMode = IntegrationMode.INTELLIGENT
    edge_preference_threshold: float = 0.7  # Preference score threshold for edge routing
    cache_sync_enabled: bool = True
    multiplexer_integration: bool = True
    failover_timeout_ms: int = 5000
    health_check_interval: int = 30
    performance_optimization: bool = True
    geo_routing_enabled: bool = True

@dataclass
class ProcessingResult:
    """Result of integrated request processing"""
    request_id: str
    success: bool
    processing_path: str  # "edge", "traditional", "hybrid"
    region: Optional[str]
    processing_time_ms: int
    latency_ms: int
    cache_hit: bool
    response_data: Dict[str, Any]
    error_message: Optional[str] = None

class EdgeIntegrationManager:
    """
    Edge Integration Manager

    Orchestrates integration between edge computing network and existing
    BEV OSINT infrastructure (multiplexing and caching systems).
    """

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Core components
        self.edge_network: Optional[EdgeComputeNetwork] = None
        self.geo_router: Optional[GeoRouter] = None
        self.request_multiplexer: Optional[RequestMultiplexer] = None
        self.predictive_cache: Optional[PredictiveCache] = None

        # Integration state
        self.request_routing_history: List[Tuple[str, str, float]] = []  # (request_id, path, score)
        self.performance_metrics: Dict[str, float] = {}
        self.active_integrations: Dict[str, Dict[str, Any]] = {}

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.integration_requests_total = Counter(
            'edge_integration_requests_total',
            'Total integration requests',
            ['processing_path', 'region', 'success'],
            registry=self.registry
        )
        self.integration_latency_histogram = Histogram(
            'edge_integration_latency_seconds',
            'Integration request latency',
            ['processing_path', 'region'],
            registry=self.registry
        )
        self.routing_decision_counter = Counter(
            'edge_integration_routing_decisions_total',
            'Routing decisions made',
            ['decision', 'mode'],
            registry=self.registry
        )
        self.cache_sync_counter = Counter(
            'edge_integration_cache_sync_total',
            'Cache synchronization operations',
            ['operation', 'success'],
            registry=self.registry
        )

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize edge integration manager"""
        try:
            self.logger.info("Initializing Edge Integration Manager")

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Initialize database connection
            self.db_pool = await asyncpg.create_pool(
                host="localhost",
                port=5432,
                user="postgres",
                password=os.getenv('DB_PASSWORD', 'dev_password'),
                database="bev_osint",
                min_size=3,
                max_size=10
            )

            # Create database tables
            await self._create_tables()

            # Initialize edge computing components
            await self._initialize_edge_components()

            # Initialize traditional infrastructure components
            await self._initialize_traditional_components()

            # Setup integration monitoring
            await self._setup_integration_monitoring()

            self.logger.info("Edge Integration Manager initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize Edge Integration Manager: {e}")
            raise

    async def _create_tables(self):
        """Create database tables for integration tracking"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_integration_requests (
                    request_id VARCHAR(64) PRIMARY KEY,
                    processing_path VARCHAR(32) NOT NULL,
                    region VARCHAR(32),
                    client_ip INET,
                    request_type VARCHAR(32) NOT NULL,
                    payload JSONB NOT NULL,
                    routing_decision VARCHAR(32) NOT NULL,
                    routing_score DECIMAL(5,2),
                    processing_time_ms INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    cache_hit BOOLEAN NOT NULL,
                    success BOOLEAN NOT NULL,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_integration_performance (
                    metric_name VARCHAR(64) NOT NULL,
                    metric_value DECIMAL(10,2) NOT NULL,
                    processing_path VARCHAR(32) NOT NULL,
                    region VARCHAR(32),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (metric_name, processing_path, timestamp)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_cache_sync_events (
                    event_id SERIAL PRIMARY KEY,
                    sync_type VARCHAR(32) NOT NULL,
                    source_region VARCHAR(32),
                    target_region VARCHAR(32),
                    cache_key VARCHAR(128) NOT NULL,
                    sync_success BOOLEAN NOT NULL,
                    sync_time_ms INTEGER,
                    error_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    async def _initialize_edge_components(self):
        """Initialize edge computing components"""
        try:
            # Initialize edge compute network
            self.edge_network = EdgeComputeNetwork()
            await self.edge_network.initialize()

            # Initialize geographic router
            if self.config.geo_routing_enabled:
                self.geo_router = GeoRouter()
                await self.geo_router.initialize()

            self.logger.info("Edge computing components initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize edge components: {e}")
            raise

    async def _initialize_traditional_components(self):
        """Initialize traditional infrastructure components"""
        try:
            # Initialize request multiplexer (if available)
            if self.config.multiplexer_integration:
                try:
                    # This would be the actual multiplexer initialization
                    # For now, we'll simulate the connection
                    self.logger.info("Request multiplexer integration enabled")
                except Exception as e:
                    self.logger.warning(f"Could not initialize request multiplexer: {e}")

            # Initialize predictive cache (if available)
            if self.config.cache_sync_enabled:
                try:
                    # This would be the actual cache initialization
                    # For now, we'll simulate the connection
                    self.logger.info("Predictive cache integration enabled")
                except Exception as e:
                    self.logger.warning(f"Could not initialize predictive cache: {e}")

            self.logger.info("Traditional infrastructure components initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize traditional components: {e}")

    async def _setup_integration_monitoring(self):
        """Setup integration monitoring and optimization"""
        # Start background tasks
        asyncio.create_task(self._performance_monitor())
        asyncio.create_task(self._cache_synchronizer())
        asyncio.create_task(self._routing_optimizer())

    async def process_integrated_request(self, request_data: Dict[str, Any]) -> ProcessingResult:
        """Process request through integrated edge and traditional infrastructure"""
        start_time = time.time()
        request_id = request_data.get("request_id", f"int_{int(time.time() * 1000)}")

        try:
            # Make routing decision
            routing_decision = await self._make_routing_decision(request_data)

            # Process based on routing decision
            if routing_decision["path"] == "edge":
                result = await self._process_via_edge(request_data, routing_decision)
            elif routing_decision["path"] == "traditional":
                result = await self._process_via_traditional(request_data, routing_decision)
            elif routing_decision["path"] == "hybrid":
                result = await self._process_via_hybrid(request_data, routing_decision)
            else:
                raise Exception(f"Unknown routing path: {routing_decision['path']}")

            # Update metrics
            self.integration_requests_total.labels(
                processing_path=result.processing_path,
                region=result.region or "unknown",
                success="true" if result.success else "false"
            ).inc()

            self.integration_latency_histogram.labels(
                processing_path=result.processing_path,
                region=result.region or "unknown"
            ).observe((time.time() - start_time))

            # Store integration result
            await self._store_integration_result(request_data, routing_decision, result)

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            self.logger.error(f"Integrated request processing failed for {request_id}: {e}")

            # Create error result
            result = ProcessingResult(
                request_id=request_id,
                success=False,
                processing_path="error",
                region=None,
                processing_time_ms=int(processing_time),
                latency_ms=int(processing_time),
                cache_hit=False,
                response_data={"error": str(e)},
                error_message=str(e)
            )

            # Update error metrics
            self.integration_requests_total.labels(
                processing_path="error",
                region="unknown",
                success="false"
            ).inc()

            return result

    async def _make_routing_decision(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent routing decision based on configuration and conditions"""
        decision = {
            "path": "edge",  # Default to edge
            "region": None,
            "score": 0.0,
            "reasoning": [],
            "fallback": "traditional"
        }

        try:
            if self.config.mode == IntegrationMode.EDGE_ONLY:
                decision["path"] = "edge"
                decision["reasoning"].append("edge_only_mode")

            elif self.config.mode == IntegrationMode.TRADITIONAL_FIRST:
                decision["path"] = "traditional"
                decision["fallback"] = "edge"
                decision["reasoning"].append("traditional_first_mode")

            elif self.config.mode == IntegrationMode.LOAD_BALANCED:
                # Simple load balancing based on current queue sizes
                edge_load = await self._get_edge_load()
                traditional_load = await self._get_traditional_load()

                if edge_load < traditional_load:
                    decision["path"] = "edge"
                    decision["score"] = 1.0 - edge_load
                else:
                    decision["path"] = "traditional"
                    decision["score"] = 1.0 - traditional_load

                decision["reasoning"].append(f"load_balanced_edge={edge_load:.2f}_traditional={traditional_load:.2f}")

            elif self.config.mode == IntegrationMode.INTELLIGENT:
                # AI-driven routing decision
                score = await self._calculate_routing_score(request_data)
                decision["score"] = score

                if score >= self.config.edge_preference_threshold:
                    decision["path"] = "edge"
                    decision["reasoning"].append(f"intelligent_edge_score={score:.2f}")
                else:
                    decision["path"] = "traditional"
                    decision["reasoning"].append(f"intelligent_traditional_score={score:.2f}")

            # Geographic routing for edge path
            if decision["path"] == "edge" and self.geo_router:
                routing_request = RoutingRequest(
                    request_id=request_data.get("request_id", "unknown"),
                    client_ip=request_data.get("client_ip", "127.0.0.1"),
                    client_location=None,
                    request_type=request_data.get("request_type", "inference"),
                    model_preference=request_data.get("model_preference"),
                    max_latency_ms=request_data.get("max_latency_ms", 5000),
                    priority=request_data.get("priority", 1),
                    timestamp=datetime.utcnow()
                )

                routing_result = await self.geo_router.route_request(routing_request)
                if routing_result.selected_region:
                    decision["region"] = routing_result.selected_region
                    decision["reasoning"].append(f"geo_routed_to_{routing_result.selected_region}")

            # Update routing decision metrics
            self.routing_decision_counter.labels(
                decision=decision["path"],
                mode=self.config.mode.value
            ).inc()

            return decision

        except Exception as e:
            self.logger.error(f"Routing decision failed: {e}")
            decision["path"] = "edge"  # Default fallback
            decision["reasoning"].append(f"error_fallback_{str(e)}")
            return decision

    async def _calculate_routing_score(self, request_data: Dict[str, Any]) -> float:
        """Calculate intelligent routing score (0.0 = traditional, 1.0 = edge)"""
        score = 0.5  # Base score

        try:
            # Factor 1: Request type (some types work better on edge)
            request_type = request_data.get("request_type", "")
            if request_type in ["inference", "analysis", "pattern_recognition"]:
                score += 0.2
            elif request_type in ["data_processing", "aggregation"]:
                score -= 0.1

            # Factor 2: Client location proximity to edge nodes
            client_ip = request_data.get("client_ip")
            if client_ip and self.geo_router:
                client_location = self.geo_router.resolve_client_location(client_ip)
                if client_location:
                    # Calculate proximity to nearest edge node
                    min_distance = float('inf')
                    for region_code, region in self.geo_router.edge_nodes.items():
                        distance = self.geo_router.calculate_geographic_distance(client_location, region.location)
                        min_distance = min(min_distance, distance)

                    # Closer = higher edge score
                    if min_distance < 1000:  # Less than 1000km
                        score += 0.2
                    elif min_distance < 5000:  # Less than 5000km
                        score += 0.1

            # Factor 3: Current system loads
            edge_load = await self._get_edge_load()
            traditional_load = await self._get_traditional_load()

            if edge_load < 0.7:  # Edge not overloaded
                score += 0.1
            if traditional_load > 0.8:  # Traditional system overloaded
                score += 0.2

            # Factor 4: Historical performance
            historical_perf = await self._get_historical_performance()
            if historical_perf["edge_avg_latency"] < historical_perf["traditional_avg_latency"]:
                score += 0.1

            # Factor 5: Cache hit probability
            cache_key = request_data.get("cache_key")
            if cache_key:
                cache_hit_prob = await self._estimate_cache_hit_probability(cache_key)
                if cache_hit_prob > 0.7:  # High cache hit probability
                    score += 0.1

            # Ensure score is within bounds
            score = max(0.0, min(1.0, score))

            return score

        except Exception as e:
            self.logger.error(f"Routing score calculation failed: {e}")
            return 0.5  # Default to neutral score

    async def _process_via_edge(self, request_data: Dict[str, Any], routing_decision: Dict[str, Any]) -> ProcessingResult:
        """Process request via edge computing network"""
        start_time = time.time()

        try:
            # Create edge request
            edge_request = EdgeRequest(
                request_id=request_data.get("request_id", f"edge_{int(time.time() * 1000)}"),
                request_type=RequestType(request_data.get("request_type", "inference")),
                payload=request_data.get("payload", {}),
                priority=request_data.get("priority", 1),
                max_latency_ms=request_data.get("max_latency_ms", 5000),
                client_ip=request_data.get("client_ip", "127.0.0.1"),
                client_region=routing_decision.get("region"),
                timestamp=datetime.utcnow(),
                model_preference=request_data.get("model_preference"),
                cache_key=request_data.get("cache_key")
            )

            # Process through edge network
            edge_response = await self.edge_network.process_request(edge_request)

            processing_time = (time.time() - start_time) * 1000

            # Create result
            result = ProcessingResult(
                request_id=edge_request.request_id,
                success=edge_response.success,
                processing_path="edge",
                region=edge_response.region,
                processing_time_ms=edge_response.processing_time_ms,
                latency_ms=int(processing_time),
                cache_hit=edge_response.cache_hit,
                response_data=edge_response.response_data,
                error_message=edge_response.error_message if not edge_response.success else None
            )

            # Sync cache to traditional infrastructure if enabled
            if self.config.cache_sync_enabled and edge_response.cache_hit:
                await self._sync_cache_to_traditional(edge_request.cache_key, edge_response.response_data)

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            # Attempt failover to traditional if configured
            if routing_decision.get("fallback") == "traditional":
                self.logger.warning(f"Edge processing failed, attempting traditional fallback: {e}")
                return await self._process_via_traditional(request_data, {"path": "traditional", "fallback": None})

            return ProcessingResult(
                request_id=request_data.get("request_id", "unknown"),
                success=False,
                processing_path="edge",
                region=None,
                processing_time_ms=int(processing_time),
                latency_ms=int(processing_time),
                cache_hit=False,
                response_data={"error": str(e)},
                error_message=str(e)
            )

    async def _process_via_traditional(self, request_data: Dict[str, Any], routing_decision: Dict[str, Any]) -> ProcessingResult:
        """Process request via traditional infrastructure"""
        start_time = time.time()

        try:
            # Simulate traditional processing
            # In a real implementation, this would interface with the actual multiplexer and cache

            # Check traditional cache first
            cache_hit = False
            cached_response = None
            cache_key = request_data.get("cache_key")

            if cache_key:
                cached_response = await self._check_traditional_cache(cache_key)
                if cached_response:
                    cache_hit = True

            if not cache_hit:
                # Process through traditional infrastructure
                # This would be the actual request processing logic
                await asyncio.sleep(0.1)  # Simulate processing time

                # Simulate response
                response_data = {
                    "result": f"Traditional processing for {request_data.get('request_type', 'unknown')}",
                    "processed_by": "traditional_infrastructure",
                    "timestamp": datetime.utcnow().isoformat()
                }

                # Cache response if applicable
                if cache_key:
                    await self._cache_traditional_response(cache_key, response_data)

            else:
                response_data = cached_response

            processing_time = (time.time() - start_time) * 1000

            result = ProcessingResult(
                request_id=request_data.get("request_id", f"trad_{int(time.time() * 1000)}"),
                success=True,
                processing_path="traditional",
                region=None,
                processing_time_ms=int(processing_time),
                latency_ms=int(processing_time),
                cache_hit=cache_hit,
                response_data=response_data
            )

            return result

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            # Attempt failover to edge if configured
            if routing_decision.get("fallback") == "edge":
                self.logger.warning(f"Traditional processing failed, attempting edge fallback: {e}")
                return await self._process_via_edge(request_data, {"path": "edge", "fallback": None})

            return ProcessingResult(
                request_id=request_data.get("request_id", "unknown"),
                success=False,
                processing_path="traditional",
                region=None,
                processing_time_ms=int(processing_time),
                latency_ms=int(processing_time),
                cache_hit=False,
                response_data={"error": str(e)},
                error_message=str(e)
            )

    async def _process_via_hybrid(self, request_data: Dict[str, Any], routing_decision: Dict[str, Any]) -> ProcessingResult:
        """Process request via hybrid edge and traditional approach"""
        # For hybrid processing, we could:
        # 1. Split the request and process parts on edge and traditional
        # 2. Process on both and return the fastest result
        # 3. Use edge for inference and traditional for data aggregation

        # For simplicity, we'll try edge first and fallback to traditional
        try:
            return await self._process_via_edge(request_data, {"path": "edge", "fallback": "traditional"})
        except Exception:
            return await self._process_via_traditional(request_data, {"path": "traditional", "fallback": None})

    async def _get_edge_load(self) -> float:
        """Get current edge computing load (0.0 = no load, 1.0 = full load)"""
        try:
            if self.edge_network:
                status = await self.edge_network.get_network_status()
                total_capacity = status.get("total_capacity", 1)
                active_regions = status.get("active_regions", 0)

                # Simple load calculation based on active regions vs total
                return min(1.0, active_regions / max(total_capacity / 100, 1))
            return 0.5
        except Exception:
            return 0.5

    async def _get_traditional_load(self) -> float:
        """Get current traditional infrastructure load"""
        try:
            # This would query the actual multiplexer for current load
            # For now, simulate a load value
            return 0.3
        except Exception:
            return 0.5

    async def _get_historical_performance(self) -> Dict[str, float]:
        """Get historical performance metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                # Get average latency for each processing path
                edge_perf = await conn.fetchval("""
                    SELECT AVG(latency_ms) FROM edge_integration_requests
                    WHERE processing_path = 'edge' AND timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                """) or 100.0

                traditional_perf = await conn.fetchval("""
                    SELECT AVG(latency_ms) FROM edge_integration_requests
                    WHERE processing_path = 'traditional' AND timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                """) or 150.0

                return {
                    "edge_avg_latency": float(edge_perf),
                    "traditional_avg_latency": float(traditional_perf)
                }
        except Exception:
            return {"edge_avg_latency": 100.0, "traditional_avg_latency": 150.0}

    async def _estimate_cache_hit_probability(self, cache_key: str) -> float:
        """Estimate probability of cache hit for given key"""
        try:
            # This would analyze cache patterns and predict hit probability
            # For now, return a simulated probability
            return 0.6
        except Exception:
            return 0.5

    async def _check_traditional_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check traditional cache for cached response"""
        try:
            # This would interface with the actual predictive cache
            # For now, simulate cache miss/hit
            return None
        except Exception:
            return None

    async def _cache_traditional_response(self, cache_key: str, response_data: Dict[str, Any]):
        """Cache response in traditional cache system"""
        try:
            # This would interface with the actual predictive cache
            pass
        except Exception:
            pass

    async def _sync_cache_to_traditional(self, cache_key: str, response_data: Dict[str, Any]):
        """Sync edge cache response to traditional cache"""
        try:
            if self.config.cache_sync_enabled:
                await self._cache_traditional_response(cache_key, response_data)

                # Record sync event
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO edge_cache_sync_events
                        (sync_type, cache_key, sync_success, sync_time_ms)
                        VALUES ($1, $2, $3, $4)
                    """, "edge_to_traditional", cache_key, True, 10)

                self.cache_sync_counter.labels(operation="edge_to_traditional", success="true").inc()

        except Exception as e:
            self.logger.error(f"Cache sync failed: {e}")
            self.cache_sync_counter.labels(operation="edge_to_traditional", success="false").inc()

    async def _store_integration_result(self, request_data: Dict[str, Any], routing_decision: Dict[str, Any], result: ProcessingResult):
        """Store integration result in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO edge_integration_requests
                    (request_id, processing_path, region, client_ip, request_type, payload,
                     routing_decision, routing_score, processing_time_ms, latency_ms,
                     cache_hit, success, error_message)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """, result.request_id, result.processing_path, result.region,
                request_data.get("client_ip"), request_data.get("request_type"),
                json.dumps(request_data.get("payload", {})), routing_decision["path"],
                routing_decision["score"], result.processing_time_ms, result.latency_ms,
                result.cache_hit, result.success, result.error_message)

        except Exception as e:
            self.logger.error(f"Failed to store integration result: {e}")

    async def _performance_monitor(self):
        """Monitor integration performance continuously"""
        while True:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(60)  # Collect every minute
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)

    async def _collect_performance_metrics(self):
        """Collect and store performance metrics"""
        try:
            # Collect metrics for each processing path
            async with self.db_pool.acquire() as conn:
                for path in ["edge", "traditional", "hybrid"]:
                    # Average latency
                    avg_latency = await conn.fetchval("""
                        SELECT AVG(latency_ms) FROM edge_integration_requests
                        WHERE processing_path = $1 AND timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                    """, path) or 0

                    # Success rate
                    success_rate = await conn.fetchval("""
                        SELECT AVG(CASE WHEN success THEN 1.0 ELSE 0.0 END) FROM edge_integration_requests
                        WHERE processing_path = $1 AND timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                    """, path) or 0

                    # Cache hit rate
                    cache_hit_rate = await conn.fetchval("""
                        SELECT AVG(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END) FROM edge_integration_requests
                        WHERE processing_path = $1 AND timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                    """, path) or 0

                    # Store metrics
                    await conn.execute("""
                        INSERT INTO edge_integration_performance
                        (metric_name, metric_value, processing_path)
                        VALUES ($1, $2, $3), ($4, $5, $3), ($6, $7, $3)
                    """, "avg_latency_ms", avg_latency, path,
                    "success_rate", success_rate, path,
                    "cache_hit_rate", cache_hit_rate, path)

        except Exception as e:
            self.logger.error(f"Performance metrics collection failed: {e}")

    async def _cache_synchronizer(self):
        """Synchronize cache between edge and traditional infrastructure"""
        while True:
            try:
                if self.config.cache_sync_enabled:
                    await self._perform_cache_sync()
                await asyncio.sleep(300)  # Sync every 5 minutes
            except Exception as e:
                self.logger.error(f"Cache synchronization error: {e}")
                await asyncio.sleep(300)

    async def _perform_cache_sync(self):
        """Perform bidirectional cache synchronization"""
        try:
            # This would implement the actual cache synchronization logic
            # between edge nodes and traditional cache systems
            self.logger.debug("Performing cache synchronization")

        except Exception as e:
            self.logger.error(f"Cache sync failed: {e}")

    async def _routing_optimizer(self):
        """Optimize routing decisions based on performance data"""
        while True:
            try:
                await self._optimize_routing_strategy()
                await asyncio.sleep(600)  # Optimize every 10 minutes
            except Exception as e:
                self.logger.error(f"Routing optimization error: {e}")
                await asyncio.sleep(600)

    async def _optimize_routing_strategy(self):
        """Optimize routing strategy based on performance metrics"""
        try:
            if self.config.performance_optimization:
                # Analyze performance data and adjust routing parameters
                historical_perf = await self._get_historical_performance()

                # Adjust edge preference threshold based on performance
                if historical_perf["edge_avg_latency"] < historical_perf["traditional_avg_latency"] * 0.8:
                    # Edge is significantly faster, increase preference
                    self.config.edge_preference_threshold = max(0.5, self.config.edge_preference_threshold - 0.05)
                elif historical_perf["edge_avg_latency"] > historical_perf["traditional_avg_latency"] * 1.2:
                    # Edge is slower, decrease preference
                    self.config.edge_preference_threshold = min(0.9, self.config.edge_preference_threshold + 0.05)

                self.logger.debug(f"Optimized edge preference threshold to {self.config.edge_preference_threshold:.2f}")

        except Exception as e:
            self.logger.error(f"Routing optimization failed: {e}")

    async def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and metrics"""
        try:
            status = {
                "integration_mode": self.config.mode.value,
                "edge_preference_threshold": self.config.edge_preference_threshold,
                "cache_sync_enabled": self.config.cache_sync_enabled,
                "geo_routing_enabled": self.config.geo_routing_enabled,
                "components": {
                    "edge_network": self.edge_network is not None,
                    "geo_router": self.geo_router is not None,
                    "request_multiplexer": self.request_multiplexer is not None,
                    "predictive_cache": self.predictive_cache is not None
                },
                "performance": await self._get_historical_performance(),
                "timestamp": datetime.utcnow().isoformat()
            }

            # Add edge network status if available
            if self.edge_network:
                edge_status = await self.edge_network.get_network_status()
                status["edge_network_status"] = edge_status

            return status

        except Exception as e:
            self.logger.error(f"Failed to get integration status: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup integration manager resources"""
        try:
            # Cleanup edge components
            if self.edge_network:
                await self.edge_network.cleanup()

            if self.geo_router:
                await self.geo_router.cleanup()

            # Close HTTP session
            if self.session:
                await self.session.close()

            # Close database connections
            if self.db_pool:
                await self.db_pool.close()

            self.logger.info("Edge Integration Manager cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

# Example usage
async def main():
    """Example usage of EdgeIntegrationManager"""
    config = IntegrationConfig(
        mode=IntegrationMode.INTELLIGENT,
        edge_preference_threshold=0.7,
        cache_sync_enabled=True,
        multiplexer_integration=True,
        geo_routing_enabled=True
    )

    integration_manager = EdgeIntegrationManager(config)

    try:
        await integration_manager.initialize()

        # Example request
        request_data = {
            "request_id": "test-integration-001",
            "request_type": "osint_analysis",
            "payload": {
                "target": "example.com",
                "analysis_type": "domain_intelligence"
            },
            "priority": 1,
            "max_latency_ms": 5000,
            "client_ip": "192.168.1.100",
            "model_preference": "llama-3-8b",
            "cache_key": "osint_domain_example.com"
        }

        # Process integrated request
        result = await integration_manager.process_integrated_request(request_data)
        print(f"Integration result: {result}")

        # Get status
        status = await integration_manager.get_integration_status()
        print(f"Integration status: {status}")

    finally:
        await integration_manager.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())