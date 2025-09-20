import os
"""
Geographic Router for BEV OSINT Framework

Implements intelligent geographic routing with latency optimization,
load balancing, and failover capabilities for edge computing network.
"""

import asyncio
import time
import logging
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import aiohttp
import asyncpg
import geoip2.database
import geoip2.errors
from geopy.distance import geodesic
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import redis.asyncio as redis

class RoutingStrategy(Enum):
    """Routing strategy types"""
    LATENCY_OPTIMIZED = "latency_optimized"
    LOAD_BALANCED = "load_balanced"
    GEOGRAPHIC_PROXIMITY = "geographic_proximity"
    COST_OPTIMIZED = "cost_optimized"
    HYBRID = "hybrid"

class HealthStatus(Enum):
    """Edge node health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

@dataclass
class GeographicLocation:
    """Geographic location information"""
    latitude: float
    longitude: float
    country: str
    region: str
    city: str
    timezone: str
    continent: str

@dataclass
class EdgeNodeInfo:
    """Edge node information for routing"""
    node_id: str
    region: str
    endpoint: str
    location: GeographicLocation
    current_load: float
    capacity: int
    avg_latency_ms: float
    health_status: HealthStatus
    last_health_check: datetime
    supported_models: List[str]
    cost_per_request: float
    priority_score: float

@dataclass
class RoutingRequest:
    """Request for geographic routing"""
    request_id: str
    client_ip: str
    client_location: Optional[GeographicLocation]
    request_type: str
    model_preference: Optional[str]
    max_latency_ms: int
    priority: int
    timestamp: datetime

@dataclass
class RoutingResult:
    """Result of geographic routing"""
    request_id: str
    selected_node: str
    selected_region: str
    estimated_latency_ms: float
    confidence_score: float
    routing_strategy: RoutingStrategy
    fallback_nodes: List[str]
    routing_time_ms: float

class GeoRouter:
    """
    Geographic Router

    Implements intelligent routing decisions based on geographic proximity,
    latency optimization, load balancing, and cost considerations.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.edge_nodes: Dict[str, EdgeNodeInfo] = {}
        self.routing_history: List[Tuple[RoutingRequest, RoutingResult]] = []
        self.latency_matrix: Dict[str, Dict[str, float]] = {}
        self.geoip_db: Optional[geoip2.database.Reader] = None
        self.redis_client: Optional[redis.Redis] = None

        # Routing configuration
        self.default_strategy = RoutingStrategy.HYBRID
        self.max_routing_time_ms = 100
        self.latency_weight = 0.4
        self.load_weight = 0.3
        self.geographic_weight = 0.2
        self.cost_weight = 0.1
        self.health_check_interval = 30
        self.latency_measurement_interval = 60

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.routing_counter = Counter(
            'geo_routing_total',
            'Total routing decisions',
            ['strategy', 'region', 'success'],
            registry=self.registry
        )
        self.routing_latency_histogram = Histogram(
            'geo_routing_latency_seconds',
            'Routing decision latency',
            ['strategy'],
            registry=self.registry
        )
        self.selected_node_counter = Counter(
            'geo_selected_node_total',
            'Nodes selected for routing',
            ['node_id', 'region'],
            registry=self.registry
        )
        self.edge_latency_gauge = Gauge(
            'geo_edge_latency_ms',
            'Edge node latency',
            ['node_id', 'region'],
            registry=self.registry
        )
        self.edge_load_gauge = Gauge(
            'geo_edge_load_percent',
            'Edge node load percentage',
            ['node_id', 'region'],
            registry=self.registry
        )

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize geographic router"""
        try:
            self.logger.info("Initializing Geographic Router")

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

            # Initialize Redis connection for caching
            try:
                self.redis_client = redis.Redis(
                    host="localhost",
                    port=6379,
                    decode_responses=True
                )
                await self.redis_client.ping()
                self.logger.info("Redis connection established")
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None

            # Create database tables
            await self._create_tables()

            # Initialize GeoIP database
            await self._initialize_geoip()

            # Load edge nodes configuration
            await self._load_edge_nodes()

            # Initialize latency matrix
            await self._initialize_latency_matrix()

            # Start background tasks
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._latency_monitor())
            asyncio.create_task(self._routing_optimizer())

            self.logger.info("Geographic Router initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize geographic router: {e}")
            raise

    async def _create_tables(self):
        """Create database tables for routing"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS geo_routing_requests (
                    request_id VARCHAR(64) PRIMARY KEY,
                    client_ip INET NOT NULL,
                    client_country VARCHAR(32),
                    client_region VARCHAR(32),
                    client_city VARCHAR(32),
                    request_type VARCHAR(32) NOT NULL,
                    model_preference VARCHAR(32),
                    max_latency_ms INTEGER NOT NULL,
                    priority INTEGER NOT NULL,
                    selected_node VARCHAR(64) NOT NULL,
                    selected_region VARCHAR(32) NOT NULL,
                    estimated_latency_ms DECIMAL(8,2) NOT NULL,
                    actual_latency_ms DECIMAL(8,2),
                    routing_strategy VARCHAR(32) NOT NULL,
                    confidence_score DECIMAL(5,2) NOT NULL,
                    routing_time_ms DECIMAL(8,2) NOT NULL,
                    success BOOLEAN,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_node_metrics (
                    node_id VARCHAR(64) NOT NULL,
                    region VARCHAR(32) NOT NULL,
                    metric_type VARCHAR(32) NOT NULL,
                    metric_value DECIMAL(10,2) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (node_id, metric_type, timestamp)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS latency_measurements (
                    source_ip INET NOT NULL,
                    target_node VARCHAR(64) NOT NULL,
                    target_region VARCHAR(32) NOT NULL,
                    latency_ms DECIMAL(8,2) NOT NULL,
                    measurement_type VARCHAR(32) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_ip, target_node, timestamp)
                )
            """)

    async def _initialize_geoip(self):
        """Initialize GeoIP database for location resolution"""
        try:
            # Try to load GeoIP database (MaxMind GeoLite2)
            geoip_paths = [
                "/usr/share/GeoIP/GeoLite2-City.mmdb",
                "/opt/geoip/GeoLite2-City.mmdb",
                "./GeoLite2-City.mmdb"
            ]

            for path in geoip_paths:
                try:
                    self.geoip_db = geoip2.database.Reader(path)
                    self.logger.info(f"GeoIP database loaded from {path}")
                    break
                except FileNotFoundError:
                    continue

            if not self.geoip_db:
                self.logger.warning("GeoIP database not found, using fallback location detection")

        except Exception as e:
            self.logger.warning(f"GeoIP initialization failed: {e}")

    async def _load_edge_nodes(self):
        """Load edge nodes configuration from database"""
        try:
            async with self.db_pool.acquire() as conn:
                nodes = await conn.fetch("""
                    SELECT node_id, region, endpoint, model_endpoint, is_active
                    FROM edge_nodes
                    WHERE is_active = TRUE
                """)

                for node in nodes:
                    # Get location for region
                    location = self._get_region_location(node['region'])

                    node_info = EdgeNodeInfo(
                        node_id=node['node_id'],
                        region=node['region'],
                        endpoint=node['endpoint'],
                        location=location,
                        current_load=0.0,
                        capacity=1000,  # Default capacity
                        avg_latency_ms=100.0,
                        health_status=HealthStatus.HEALTHY,
                        last_health_check=datetime.utcnow(),
                        supported_models=["llama-3-8b", "phi-3-mini", "mistral-7b"],
                        cost_per_request=0.001,  # Default cost
                        priority_score=1.0
                    )

                    self.edge_nodes[node['node_id']] = node_info

            self.logger.info(f"Loaded {len(self.edge_nodes)} edge nodes")

        except Exception as e:
            self.logger.error(f"Failed to load edge nodes: {e}")

    def _get_region_location(self, region: str) -> GeographicLocation:
        """Get geographic location for a region"""
        region_locations = {
            "us-east": GeographicLocation(
                latitude=40.7128, longitude=-74.0060,
                country="US", region="New York", city="New York",
                timezone="America/New_York", continent="North America"
            ),
            "us-west": GeographicLocation(
                latitude=37.7749, longitude=-122.4194,
                country="US", region="California", city="San Francisco",
                timezone="America/Los_Angeles", continent="North America"
            ),
            "eu-central": GeographicLocation(
                latitude=52.5200, longitude=13.4050,
                country="DE", region="Berlin", city="Berlin",
                timezone="Europe/Berlin", continent="Europe"
            ),
            "asia-pacific": GeographicLocation(
                latitude=1.3521, longitude=103.8198,
                country="SG", region="Singapore", city="Singapore",
                timezone="Asia/Singapore", continent="Asia"
            )
        }

        return region_locations.get(region, region_locations["us-east"])

    async def _initialize_latency_matrix(self):
        """Initialize latency measurement matrix"""
        try:
            # Load historical latency data
            async with self.db_pool.acquire() as conn:
                measurements = await conn.fetch("""
                    SELECT target_node, target_region, AVG(latency_ms) as avg_latency
                    FROM latency_measurements
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                    GROUP BY target_node, target_region
                """)

                for measurement in measurements:
                    node_id = measurement['target_node']
                    region = measurement['target_region']
                    avg_latency = float(measurement['avg_latency'])

                    if node_id not in self.latency_matrix:
                        self.latency_matrix[node_id] = {}

                    # Store latency data
                    if node_id in self.edge_nodes:
                        self.edge_nodes[node_id].avg_latency_ms = avg_latency

        except Exception as e:
            self.logger.error(f"Failed to initialize latency matrix: {e}")

    def resolve_client_location(self, client_ip: str) -> Optional[GeographicLocation]:
        """Resolve client geographic location from IP address"""
        try:
            if not self.geoip_db:
                return None

            response = self.geoip_db.city(client_ip)

            location = GeographicLocation(
                latitude=float(response.location.latitude) if response.location.latitude else 0.0,
                longitude=float(response.location.longitude) if response.location.longitude else 0.0,
                country=response.country.iso_code or "Unknown",
                region=response.subdivisions.most_specific.name or "Unknown",
                city=response.city.name or "Unknown",
                timezone=response.location.time_zone or "UTC",
                continent=response.continent.name or "Unknown"
            )

            return location

        except geoip2.errors.AddressNotFoundError:
            self.logger.debug(f"IP address {client_ip} not found in GeoIP database")
        except Exception as e:
            self.logger.warning(f"GeoIP lookup failed for {client_ip}: {e}")

        return None

    def calculate_geographic_distance(self, loc1: GeographicLocation, loc2: GeographicLocation) -> float:
        """Calculate geographic distance between two locations in kilometers"""
        try:
            distance = geodesic(
                (loc1.latitude, loc1.longitude),
                (loc2.latitude, loc2.longitude)
            ).kilometers

            return distance

        except Exception as e:
            self.logger.warning(f"Distance calculation failed: {e}")
            return 10000.0  # Default large distance

    def estimate_latency_from_distance(self, distance_km: float) -> float:
        """Estimate latency from geographic distance"""
        # Rough estimation: ~0.1ms per 10km + base latency
        # This is a simplified model; real-world latency depends on many factors
        base_latency = 20.0  # Base latency in ms
        distance_latency = distance_km * 0.01  # 0.01ms per km

        return base_latency + distance_latency

    async def route_request(self, request: RoutingRequest, strategy: Optional[RoutingStrategy] = None) -> RoutingResult:
        """Route request to optimal edge node"""
        start_time = time.time()

        try:
            routing_strategy = strategy or self.default_strategy

            # Resolve client location if not provided
            if not request.client_location:
                request.client_location = self.resolve_client_location(request.client_ip)

            # Get available nodes
            available_nodes = await self._get_available_nodes(request)

            if not available_nodes:
                raise Exception("No available edge nodes")

            # Apply routing strategy
            if routing_strategy == RoutingStrategy.LATENCY_OPTIMIZED:
                selected_node, confidence = await self._route_by_latency(request, available_nodes)
            elif routing_strategy == RoutingStrategy.LOAD_BALANCED:
                selected_node, confidence = await self._route_by_load(request, available_nodes)
            elif routing_strategy == RoutingStrategy.GEOGRAPHIC_PROXIMITY:
                selected_node, confidence = await self._route_by_geography(request, available_nodes)
            elif routing_strategy == RoutingStrategy.COST_OPTIMIZED:
                selected_node, confidence = await self._route_by_cost(request, available_nodes)
            else:  # HYBRID
                selected_node, confidence = await self._route_hybrid(request, available_nodes)

            # Generate fallback nodes
            fallback_nodes = await self._generate_fallback_nodes(request, available_nodes, selected_node)

            routing_time = (time.time() - start_time) * 1000

            # Create routing result
            result = RoutingResult(
                request_id=request.request_id,
                selected_node=selected_node,
                selected_region=self.edge_nodes[selected_node].region,
                estimated_latency_ms=self.edge_nodes[selected_node].avg_latency_ms,
                confidence_score=confidence,
                routing_strategy=routing_strategy,
                fallback_nodes=fallback_nodes,
                routing_time_ms=routing_time
            )

            # Store routing decision
            await self._store_routing_decision(request, result)

            # Update metrics
            self.routing_counter.labels(
                strategy=routing_strategy.value,
                region=result.selected_region,
                success="true"
            ).inc()

            self.routing_latency_histogram.labels(
                strategy=routing_strategy.value
            ).observe(routing_time / 1000.0)

            self.selected_node_counter.labels(
                node_id=selected_node,
                region=result.selected_region
            ).inc()

            self.logger.debug(f"Routed request {request.request_id} to {selected_node} in {routing_time:.2f}ms")

            return result

        except Exception as e:
            routing_time = (time.time() - start_time) * 1000

            self.logger.error(f"Routing failed for request {request.request_id}: {e}")

            # Update error metrics
            self.routing_counter.labels(
                strategy=(strategy or self.default_strategy).value,
                region="error",
                success="false"
            ).inc()

            # Return error result
            return RoutingResult(
                request_id=request.request_id,
                selected_node="",
                selected_region="",
                estimated_latency_ms=0.0,
                confidence_score=0.0,
                routing_strategy=strategy or self.default_strategy,
                fallback_nodes=[],
                routing_time_ms=routing_time
            )

    async def _get_available_nodes(self, request: RoutingRequest) -> List[str]:
        """Get list of available edge nodes for request"""
        available_nodes = []

        for node_id, node_info in self.edge_nodes.items():
            # Check health status
            if node_info.health_status == HealthStatus.OFFLINE:
                continue

            # Check model availability
            if request.model_preference and request.model_preference not in node_info.supported_models:
                continue

            # Check capacity
            if node_info.current_load > 0.95:  # Over 95% capacity
                continue

            # Check latency requirements
            if node_info.avg_latency_ms > request.max_latency_ms:
                continue

            available_nodes.append(node_id)

        return available_nodes

    async def _route_by_latency(self, request: RoutingRequest, available_nodes: List[str]) -> Tuple[str, float]:
        """Route based on latency optimization"""
        best_node = None
        best_latency = float('inf')

        for node_id in available_nodes:
            node_info = self.edge_nodes[node_id]

            # Use cached latency or estimate from distance
            latency = node_info.avg_latency_ms

            if request.client_location:
                distance = self.calculate_geographic_distance(request.client_location, node_info.location)
                estimated_latency = self.estimate_latency_from_distance(distance)
                latency = min(latency, estimated_latency)

            if latency < best_latency:
                best_latency = latency
                best_node = node_id

        confidence = 0.9 if best_node else 0.0
        return best_node or available_nodes[0], confidence

    async def _route_by_load(self, request: RoutingRequest, available_nodes: List[str]) -> Tuple[str, float]:
        """Route based on load balancing"""
        best_node = None
        best_load_score = float('inf')

        for node_id in available_nodes:
            node_info = self.edge_nodes[node_id]

            # Calculate load score (lower is better)
            load_score = node_info.current_load * 100 + (node_info.avg_latency_ms / 10)

            if load_score < best_load_score:
                best_load_score = load_score
                best_node = node_id

        confidence = 0.8 if best_node else 0.0
        return best_node or available_nodes[0], confidence

    async def _route_by_geography(self, request: RoutingRequest, available_nodes: List[str]) -> Tuple[str, float]:
        """Route based on geographic proximity"""
        if not request.client_location:
            # Fallback to load balancing
            return await self._route_by_load(request, available_nodes)

        best_node = None
        best_distance = float('inf')

        for node_id in available_nodes:
            node_info = self.edge_nodes[node_id]
            distance = self.calculate_geographic_distance(request.client_location, node_info.location)

            if distance < best_distance:
                best_distance = distance
                best_node = node_id

        confidence = 0.85 if best_node else 0.0
        return best_node or available_nodes[0], confidence

    async def _route_by_cost(self, request: RoutingRequest, available_nodes: List[str]) -> Tuple[str, float]:
        """Route based on cost optimization"""
        best_node = None
        best_cost_score = float('inf')

        for node_id in available_nodes:
            node_info = self.edge_nodes[node_id]

            # Calculate cost score (cost + latency penalty)
            latency_penalty = max(0, node_info.avg_latency_ms - request.max_latency_ms) * 0.001
            cost_score = node_info.cost_per_request + latency_penalty

            if cost_score < best_cost_score:
                best_cost_score = cost_score
                best_node = node_id

        confidence = 0.7 if best_node else 0.0
        return best_node or available_nodes[0], confidence

    async def _route_hybrid(self, request: RoutingRequest, available_nodes: List[str]) -> Tuple[str, float]:
        """Route using hybrid strategy (weighted combination)"""
        best_node = None
        best_score = float('inf')

        for node_id in available_nodes:
            node_info = self.edge_nodes[node_id]

            # Calculate weighted score
            latency_score = node_info.avg_latency_ms * self.latency_weight
            load_score = node_info.current_load * 100 * self.load_weight

            # Geographic score
            geographic_score = 0
            if request.client_location:
                distance = self.calculate_geographic_distance(request.client_location, node_info.location)
                geographic_score = distance * self.geographic_weight

            cost_score = node_info.cost_per_request * 1000 * self.cost_weight

            # Health penalty
            health_penalty = 0
            if node_info.health_status == HealthStatus.DEGRADED:
                health_penalty = 20
            elif node_info.health_status == HealthStatus.UNHEALTHY:
                health_penalty = 100

            total_score = latency_score + load_score + geographic_score + cost_score + health_penalty

            if total_score < best_score:
                best_score = total_score
                best_node = node_id

        confidence = 0.95 if best_node else 0.0
        return best_node or available_nodes[0], confidence

    async def _generate_fallback_nodes(self, request: RoutingRequest, available_nodes: List[str], selected_node: str) -> List[str]:
        """Generate fallback nodes for redundancy"""
        fallback_nodes = []

        # Remove selected node from available nodes
        remaining_nodes = [node for node in available_nodes if node != selected_node]

        # Sort by composite score
        scored_nodes = []
        for node_id in remaining_nodes:
            node_info = self.edge_nodes[node_id]
            score = node_info.avg_latency_ms + (node_info.current_load * 100)
            scored_nodes.append((node_id, score))

        # Sort by score and take top 2
        scored_nodes.sort(key=lambda x: x[1])
        fallback_nodes = [node_id for node_id, _ in scored_nodes[:2]]

        return fallback_nodes

    async def _store_routing_decision(self, request: RoutingRequest, result: RoutingResult):
        """Store routing decision in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO geo_routing_requests
                    (request_id, client_ip, client_country, client_region, client_city,
                     request_type, model_preference, max_latency_ms, priority,
                     selected_node, selected_region, estimated_latency_ms,
                     routing_strategy, confidence_score, routing_time_ms, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                """, request.request_id, request.client_ip,
                request.client_location.country if request.client_location else None,
                request.client_location.region if request.client_location else None,
                request.client_location.city if request.client_location else None,
                request.request_type, request.model_preference, request.max_latency_ms,
                request.priority, result.selected_node, result.selected_region,
                result.estimated_latency_ms, result.routing_strategy.value,
                result.confidence_score, result.routing_time_ms, request.timestamp)

        except Exception as e:
            self.logger.error(f"Failed to store routing decision: {e}")

    async def update_node_metrics(self, node_id: str, metrics: Dict[str, float]):
        """Update node metrics for routing decisions"""
        try:
            if node_id not in self.edge_nodes:
                return

            node_info = self.edge_nodes[node_id]

            # Update metrics
            if "load" in metrics:
                node_info.current_load = metrics["load"]
                self.edge_load_gauge.labels(
                    node_id=node_id,
                    region=node_info.region
                ).set(metrics["load"] * 100)

            if "latency" in metrics:
                node_info.avg_latency_ms = metrics["latency"]
                self.edge_latency_gauge.labels(
                    node_id=node_id,
                    region=node_info.region
                ).set(metrics["latency"])

            if "health" in metrics:
                health_value = metrics["health"]
                if health_value > 0.9:
                    node_info.health_status = HealthStatus.HEALTHY
                elif health_value > 0.7:
                    node_info.health_status = HealthStatus.DEGRADED
                elif health_value > 0.3:
                    node_info.health_status = HealthStatus.UNHEALTHY
                else:
                    node_info.health_status = HealthStatus.OFFLINE

            # Store in database
            async with self.db_pool.acquire() as conn:
                for metric_name, metric_value in metrics.items():
                    await conn.execute("""
                        INSERT INTO edge_node_metrics
                        (node_id, region, metric_type, metric_value)
                        VALUES ($1, $2, $3, $4)
                    """, node_id, node_info.region, metric_name, metric_value)

        except Exception as e:
            self.logger.error(f"Failed to update node metrics for {node_id}: {e}")

    async def _health_monitor(self):
        """Monitor edge node health continuously"""
        while True:
            try:
                await self._check_node_health()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.health_check_interval)

    async def _check_node_health(self):
        """Check health of all edge nodes"""
        health_tasks = []

        for node_id, node_info in self.edge_nodes.items():
            task = self._check_single_node_health(node_id, node_info)
            health_tasks.append(task)

        await asyncio.gather(*health_tasks, return_exceptions=True)

    async def _check_single_node_health(self, node_id: str, node_info: EdgeNodeInfo):
        """Check health of a single edge node"""
        try:
            start_time = time.time()

            async with self.session.get(f"{node_info.endpoint}/health") as response:
                latency = (time.time() - start_time) * 1000

                if response.status == 200:
                    health_data = await response.json()

                    # Update metrics
                    await self.update_node_metrics(node_id, {
                        "latency": latency,
                        "health": 1.0,
                        "load": health_data.get("memory_usage_percent", 0) / 100.0
                    })

                    node_info.last_health_check = datetime.utcnow()
                else:
                    await self.update_node_metrics(node_id, {
                        "latency": latency,
                        "health": 0.5
                    })

        except Exception as e:
            self.logger.warning(f"Health check failed for {node_id}: {e}")
            await self.update_node_metrics(node_id, {
                "health": 0.0
            })

    async def _latency_monitor(self):
        """Monitor latency between clients and edge nodes"""
        while True:
            try:
                await self._measure_latencies()
                await asyncio.sleep(self.latency_measurement_interval)
            except Exception as e:
                self.logger.error(f"Latency monitoring error: {e}")
                await asyncio.sleep(self.latency_measurement_interval)

    async def _measure_latencies(self):
        """Measure latencies to edge nodes"""
        # Get recent client IPs for latency testing
        try:
            async with self.db_pool.acquire() as conn:
                recent_ips = await conn.fetch("""
                    SELECT DISTINCT client_ip
                    FROM geo_routing_requests
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                    LIMIT 10
                """)

                for ip_record in recent_ips:
                    client_ip = str(ip_record['client_ip'])
                    await self._measure_latency_from_ip(client_ip)

        except Exception as e:
            self.logger.error(f"Latency measurement failed: {e}")

    async def _measure_latency_from_ip(self, client_ip: str):
        """Measure latency from a specific client IP to all edge nodes"""
        # This would typically involve:
        # 1. Ping/traceroute measurements
        # 2. HTTP request timing
        # 3. Network topology analysis

        # For now, we'll simulate based on geographic distance
        client_location = self.resolve_client_location(client_ip)
        if not client_location:
            return

        for node_id, node_info in self.edge_nodes.items():
            try:
                distance = self.calculate_geographic_distance(client_location, node_info.location)
                estimated_latency = self.estimate_latency_from_distance(distance)

                # Store latency measurement
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO latency_measurements
                        (source_ip, target_node, target_region, latency_ms, measurement_type)
                        VALUES ($1, $2, $3, $4, $5)
                    """, client_ip, node_id, node_info.region, estimated_latency, "estimated")

            except Exception as e:
                self.logger.warning(f"Latency measurement failed for {node_id}: {e}")

    async def _routing_optimizer(self):
        """Optimize routing strategies based on performance data"""
        while True:
            try:
                await self._analyze_routing_performance()
                await self._optimize_routing_weights()
                await asyncio.sleep(300)  # Optimize every 5 minutes
            except Exception as e:
                self.logger.error(f"Routing optimization error: {e}")
                await asyncio.sleep(300)

    async def _analyze_routing_performance(self):
        """Analyze routing performance and adjust strategies"""
        try:
            async with self.db_pool.acquire() as conn:
                # Analyze success rates by strategy
                strategy_performance = await conn.fetch("""
                    SELECT routing_strategy,
                           COUNT(*) as total_requests,
                           AVG(confidence_score) as avg_confidence,
                           AVG(routing_time_ms) as avg_routing_time,
                           COUNT(*) FILTER (WHERE success = TRUE) as successful_requests
                    FROM geo_routing_requests
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                    GROUP BY routing_strategy
                """)

                for performance in strategy_performance:
                    strategy = performance['routing_strategy']
                    success_rate = performance['successful_requests'] / max(performance['total_requests'], 1)

                    self.logger.info(f"Strategy {strategy}: {success_rate:.2%} success rate, "
                                   f"{performance['avg_confidence']:.2f} avg confidence, "
                                   f"{performance['avg_routing_time']:.2f}ms avg routing time")

        except Exception as e:
            self.logger.error(f"Routing performance analysis failed: {e}")

    async def _optimize_routing_weights(self):
        """Optimize routing weights based on performance data"""
        try:
            # Analyze which factors contribute most to successful routing
            async with self.db_pool.acquire() as conn:
                # Get correlation between routing factors and success
                correlation_data = await conn.fetch("""
                    SELECT
                        AVG(estimated_latency_ms) as avg_latency,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(*) FILTER (WHERE success = TRUE) as successful_requests,
                        COUNT(*) as total_requests
                    FROM geo_routing_requests
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours'
                    AND routing_strategy = 'hybrid'
                    GROUP BY selected_region
                """)

                # Simple adaptive weight adjustment
                if correlation_data:
                    total_success_rate = sum(row['successful_requests'] for row in correlation_data) / \
                                       max(sum(row['total_requests'] for row in correlation_data), 1)

                    if total_success_rate < 0.9:  # If success rate is below 90%
                        # Increase latency weight for better performance
                        self.latency_weight = min(0.5, self.latency_weight + 0.05)
                        self.load_weight = max(0.2, self.load_weight - 0.02)
                    elif total_success_rate > 0.95:  # If success rate is high
                        # Optimize for cost
                        self.cost_weight = min(0.2, self.cost_weight + 0.02)
                        self.latency_weight = max(0.3, self.latency_weight - 0.02)

        except Exception as e:
            self.logger.error(f"Weight optimization failed: {e}")

    async def get_routing_statistics(self) -> Dict[str, Any]:
        """Get routing statistics and performance metrics"""
        try:
            stats = {
                "total_nodes": len(self.edge_nodes),
                "healthy_nodes": len([n for n in self.edge_nodes.values() if n.health_status == HealthStatus.HEALTHY]),
                "routing_weights": {
                    "latency": self.latency_weight,
                    "load": self.load_weight,
                    "geographic": self.geographic_weight,
                    "cost": self.cost_weight
                },
                "node_status": {},
                "recent_performance": {}
            }

            # Node status
            for node_id, node_info in self.edge_nodes.items():
                stats["node_status"][node_id] = {
                    "region": node_info.region,
                    "health": node_info.health_status.value,
                    "load": node_info.current_load,
                    "latency_ms": node_info.avg_latency_ms,
                    "last_check": node_info.last_health_check.isoformat()
                }

            # Recent performance
            async with self.db_pool.acquire() as conn:
                performance = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_requests,
                        AVG(routing_time_ms) as avg_routing_time,
                        AVG(confidence_score) as avg_confidence,
                        COUNT(*) FILTER (WHERE success = TRUE) as successful_requests
                    FROM geo_routing_requests
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                """)

                if performance:
                    stats["recent_performance"] = {
                        "total_requests": performance['total_requests'],
                        "avg_routing_time_ms": float(performance['avg_routing_time'] or 0),
                        "avg_confidence": float(performance['avg_confidence'] or 0),
                        "success_rate": performance['successful_requests'] / max(performance['total_requests'], 1)
                    }

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get routing statistics: {e}")
            return {"error": str(e)}

    async def cleanup(self):
        """Cleanup router resources"""
        try:
            # Close HTTP session
            if self.session:
                await self.session.close()

            # Close database connections
            if self.db_pool:
                await self.db_pool.close()

            # Close Redis connection
            if self.redis_client:
                await self.redis_client.aclose()

            # Close GeoIP database
            if self.geoip_db:
                self.geoip_db.close()

            self.logger.info("Geographic Router cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

# Example usage
async def main():
    """Example usage of GeoRouter"""
    router = GeoRouter()

    try:
        await router.initialize()

        # Example routing request
        request = RoutingRequest(
            request_id="route-test-001",
            client_ip="192.168.1.100",
            client_location=None,  # Will be resolved from IP
            request_type="osint_analysis",
            model_preference="llama-3-8b",
            max_latency_ms=5000,
            priority=1,
            timestamp=datetime.utcnow()
        )

        # Route request
        result = await router.route_request(request)
        print(f"Routing result: {result}")

        # Get statistics
        stats = await router.get_routing_statistics()
        print(f"Router statistics: {stats}")

    finally:
        await router.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())