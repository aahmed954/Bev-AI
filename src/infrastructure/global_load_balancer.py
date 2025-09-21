"""
BEV OSINT Framework - Global Load Balancer with Geo-Routing
Enterprise-grade load balancing for 2000+ concurrent users with <50ms global latency
"""

import asyncio
import hashlib
import heapq
import json
import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import aiohttp
import asyncpg
from geopy.distance import geodesic
from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger(__name__)

class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    CONSISTENT_HASH = "consistent_hash"
    GEO_PROXIMITY = "geo_proximity"
    ADAPTIVE = "adaptive"
    LATENCY_AWARE = "latency_aware"

class NodeHealth(Enum):
    """Node health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"

class Region(Enum):
    """Global regions"""
    US_EAST = "us-east"
    US_WEST = "us-west"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia-pacific"
    SOUTH_AMERICA = "south-america"
    MIDDLE_EAST = "middle-east"
    AFRICA = "africa"

@dataclass
class EdgeNode:
    """Edge node configuration and state"""
    node_id: str
    region: Region
    hostname: str
    ip_address: str
    port: int
    capacity: int  # Max concurrent connections
    weight: float = 1.0
    latitude: float = 0.0
    longitude: float = 0.0

    # Runtime state
    active_connections: int = 0
    total_requests: int = 0
    error_count: int = 0
    health: NodeHealth = NodeHealth.HEALTHY
    last_health_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Performance metrics
    response_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    avg_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    bandwidth_mbps: float = 0.0

    # Circuit breaker
    failure_count: int = 0
    circuit_open: bool = False
    circuit_open_until: Optional[datetime] = None

@dataclass
class RoutingRequest:
    """Request routing information"""
    request_id: str
    client_ip: str
    client_latitude: Optional[float] = None
    client_longitude: Optional[float] = None
    client_region: Optional[Region] = None
    request_type: str = "general"
    priority: int = 0
    sticky_session: bool = False
    session_id: Optional[str] = None
    preferred_regions: List[Region] = field(default_factory=list)
    excluded_nodes: Set[str] = field(default_factory=set)
    max_latency_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class RoutingDecision:
    """Load balancer routing decision"""
    request_id: str
    selected_node: EdgeNode
    algorithm_used: LoadBalancingAlgorithm
    decision_score: float
    expected_latency_ms: float
    alternatives: List[EdgeNode]
    decision_time_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class GlobalLoadBalancer:
    """
    Enterprise-grade global load balancer with geo-routing

    Features:
    - Multiple load balancing algorithms
    - Geographic proximity routing
    - Health checking and circuit breaking
    - Adaptive algorithm selection
    - Session affinity support
    - Real-time performance monitoring
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Node management
        self.nodes: Dict[str, EdgeNode] = {}
        self.nodes_by_region: Dict[Region, List[EdgeNode]] = defaultdict(list)
        self.consistent_hash_ring: Dict[int, str] = {}

        # Session affinity
        self.session_mapping: Dict[str, str] = {}  # session_id -> node_id
        self.session_ttl = timedelta(hours=1)
        self.session_timestamps: Dict[str, datetime] = {}

        # Algorithm configuration
        self.default_algorithm = LoadBalancingAlgorithm.ADAPTIVE
        self.algorithm_weights = {
            LoadBalancingAlgorithm.LEAST_RESPONSE_TIME: 0.4,
            LoadBalancingAlgorithm.LEAST_CONNECTIONS: 0.3,
            LoadBalancingAlgorithm.GEO_PROXIMITY: 0.2,
            LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN: 0.1
        }

        # Round-robin state
        self.round_robin_counters: Dict[Region, int] = defaultdict(int)

        # Performance tracking
        self.request_history = deque(maxlen=10000)
        self.routing_decisions = deque(maxlen=10000)
        self.algorithm_performance: Dict[LoadBalancingAlgorithm, Dict[str, float]] = defaultdict(
            lambda: {"success_rate": 0.0, "avg_latency": 0.0, "usage_count": 0}
        )

        # Health checking
        self.health_check_interval = config.get('health_check_interval_seconds', 5)
        self.unhealthy_threshold = config.get('unhealthy_threshold', 3)
        self.healthy_threshold = config.get('healthy_threshold', 2)
        self.circuit_breaker_timeout = timedelta(
            seconds=config.get('circuit_breaker_timeout_seconds', 30)
        )

        # Geo-routing configuration
        self.geo_routing_enabled = config.get('enable_geo_routing', True)
        self.cross_region_penalty_ms = config.get('cross_region_penalty_ms', 50)

        # Prometheus metrics
        self.requests_routed = Counter(
            'bev_lb_requests_routed_total',
            'Total requests routed',
            ['region', 'algorithm', 'status']
        )
        self.routing_latency = Histogram(
            'bev_lb_routing_latency_ms',
            'Routing decision latency'
        )
        self.node_health_gauge = Gauge(
            'bev_lb_node_health',
            'Node health status',
            ['node_id', 'region']
        )
        self.active_connections_gauge = Gauge(
            'bev_lb_active_connections',
            'Active connections per node',
            ['node_id', 'region']
        )

    async def initialize(self):
        """Initialize load balancer with node configuration"""
        # Load node configuration
        await self._load_node_configuration()

        # Build consistent hash ring
        self._build_consistent_hash_ring()

        # Start background tasks
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._session_cleanup_loop())
        asyncio.create_task(self._algorithm_optimization_loop())

        logger.info(f"Global load balancer initialized with {len(self.nodes)} nodes")

    async def route_request(self, request: RoutingRequest) -> RoutingDecision:
        """
        Route request to optimal node

        Returns:
            RoutingDecision with selected node and metadata
        """
        start_time = time.time()

        # Check session affinity
        if request.sticky_session and request.session_id:
            node = await self._get_session_node(request.session_id)
            if node and node.health == NodeHealth.HEALTHY:
                return self._create_routing_decision(
                    request, node, LoadBalancingAlgorithm.ROUND_ROBIN,
                    time.time() - start_time
                )

        # Get available nodes
        available_nodes = await self._get_available_nodes(request)

        if not available_nodes:
            raise Exception("No available nodes for routing")

        # Select algorithm
        algorithm = await self._select_algorithm(request, available_nodes)

        # Route based on selected algorithm
        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            selected = await self._round_robin_select(available_nodes, request)
        elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            selected = await self._least_connections_select(available_nodes)
        elif algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            selected = await self._least_response_time_select(available_nodes)
        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            selected = await self._weighted_round_robin_select(available_nodes, request)
        elif algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
            selected = await self._consistent_hash_select(request)
        elif algorithm == LoadBalancingAlgorithm.GEO_PROXIMITY:
            selected = await self._geo_proximity_select(available_nodes, request)
        elif algorithm == LoadBalancingAlgorithm.LATENCY_AWARE:
            selected = await self._latency_aware_select(available_nodes, request)
        else:  # ADAPTIVE
            selected = await self._adaptive_select(available_nodes, request)

        # Update session mapping if sticky
        if request.sticky_session and request.session_id:
            await self._update_session_mapping(request.session_id, selected.node_id)

        # Update node state
        selected.active_connections += 1
        selected.total_requests += 1

        # Create routing decision
        decision = self._create_routing_decision(
            request, selected, algorithm,
            time.time() - start_time, available_nodes
        )

        # Track decision
        self.routing_decisions.append(decision)

        # Update metrics
        self.requests_routed.labels(
            region=selected.region.value,
            algorithm=algorithm.value,
            status="success"
        ).inc()
        self.routing_latency.observe(decision.decision_time_ms)

        return decision

    async def _get_available_nodes(self, request: RoutingRequest) -> List[EdgeNode]:
        """Get list of available nodes for routing"""
        available = []

        for node in self.nodes.values():
            # Skip excluded nodes
            if node.node_id in request.excluded_nodes:
                continue

            # Skip unhealthy nodes
            if node.health in [NodeHealth.UNHEALTHY, NodeHealth.MAINTENANCE]:
                continue

            # Skip nodes with open circuit breaker
            if node.circuit_open:
                if node.circuit_open_until and datetime.now(timezone.utc) > node.circuit_open_until:
                    # Close circuit breaker
                    node.circuit_open = False
                    node.circuit_open_until = None
                    node.failure_count = 0
                else:
                    continue

            # Skip nodes at capacity
            if node.active_connections >= node.capacity:
                continue

            # Check region preference
            if request.preferred_regions and node.region not in request.preferred_regions:
                # Allow with penalty
                node._temp_penalty = self.cross_region_penalty_ms
            else:
                node._temp_penalty = 0

            available.append(node)

        return available

    async def _select_algorithm(self, request: RoutingRequest,
                               available_nodes: List[EdgeNode]) -> LoadBalancingAlgorithm:
        """Select optimal algorithm based on request and system state"""
        if self.default_algorithm != LoadBalancingAlgorithm.ADAPTIVE:
            return self.default_algorithm

        # Analyze request characteristics
        has_geo_data = request.client_latitude is not None
        is_priority = request.priority > 0
        has_latency_requirement = request.max_latency_ms is not None

        # Analyze system state
        avg_load = sum(n.active_connections / n.capacity for n in available_nodes) / len(available_nodes)
        load_variance = self._calculate_load_variance(available_nodes)

        # Decision logic
        if has_latency_requirement and request.max_latency_ms < 50:
            return LoadBalancingAlgorithm.LEAST_RESPONSE_TIME
        elif has_geo_data and self.geo_routing_enabled:
            return LoadBalancingAlgorithm.GEO_PROXIMITY
        elif load_variance > 0.3:
            return LoadBalancingAlgorithm.LEAST_CONNECTIONS
        elif is_priority:
            return LoadBalancingAlgorithm.LEAST_RESPONSE_TIME
        else:
            # Use weighted selection based on past performance
            return await self._select_by_performance()

    async def _round_robin_select(self, nodes: List[EdgeNode],
                                 request: RoutingRequest) -> EdgeNode:
        """Round-robin selection"""
        region = request.client_region or Region.US_EAST

        # Get nodes for region
        region_nodes = [n for n in nodes if n.region == region]
        if not region_nodes:
            region_nodes = nodes

        # Get counter for region
        counter = self.round_robin_counters[region]
        selected = region_nodes[counter % len(region_nodes)]
        self.round_robin_counters[region] = counter + 1

        return selected

    async def _least_connections_select(self, nodes: List[EdgeNode]) -> EdgeNode:
        """Select node with least connections"""
        return min(nodes, key=lambda n: n.active_connections / n.capacity)

    async def _least_response_time_select(self, nodes: List[EdgeNode]) -> EdgeNode:
        """Select node with lowest response time"""
        # Factor in both average and P95 response times
        def score_node(node: EdgeNode) -> float:
            base_score = node.avg_response_time_ms * 0.7 + node.p95_response_time_ms * 0.3
            # Add penalty if present
            return base_score + getattr(node, '_temp_penalty', 0)

        return min(nodes, key=score_node)

    async def _weighted_round_robin_select(self, nodes: List[EdgeNode],
                                          request: RoutingRequest) -> EdgeNode:
        """Weighted round-robin selection"""
        # Build weighted list
        weighted_nodes = []
        for node in nodes:
            weight = int(node.weight * 10)
            weighted_nodes.extend([node] * weight)

        if not weighted_nodes:
            return nodes[0]

        # Select based on hash of request ID for consistency
        hash_value = int(hashlib.md5(request.request_id.encode()).hexdigest(), 16)
        return weighted_nodes[hash_value % len(weighted_nodes)]

    async def _consistent_hash_select(self, request: RoutingRequest) -> EdgeNode:
        """Consistent hash selection for cache affinity"""
        # Hash the request ID
        hash_value = int(hashlib.md5(request.request_id.encode()).hexdigest(), 16)

        # Find the node in the hash ring
        keys = sorted(self.consistent_hash_ring.keys())
        for key in keys:
            if hash_value <= key:
                node_id = self.consistent_hash_ring[key]
                return self.nodes[node_id]

        # Wrap around to first node
        node_id = self.consistent_hash_ring[keys[0]]
        return self.nodes[node_id]

    async def _geo_proximity_select(self, nodes: List[EdgeNode],
                                   request: RoutingRequest) -> EdgeNode:
        """Select geographically closest node"""
        if not request.client_latitude or not request.client_longitude:
            # Fallback to least connections
            return await self._least_connections_select(nodes)

        client_location = (request.client_latitude, request.client_longitude)

        # Calculate distances
        node_distances = []
        for node in nodes:
            node_location = (node.latitude, node.longitude)
            distance = geodesic(client_location, node_location).kilometers

            # Factor in node load
            load_factor = 1 + (node.active_connections / node.capacity)
            adjusted_distance = distance * load_factor

            node_distances.append((adjusted_distance, node))

        # Sort by distance and return closest
        node_distances.sort(key=lambda x: x[0])
        return node_distances[0][1]

    async def _latency_aware_select(self, nodes: List[EdgeNode],
                                   request: RoutingRequest) -> EdgeNode:
        """Select based on expected end-to-end latency"""
        client_location = None
        if request.client_latitude and request.client_longitude:
            client_location = (request.client_latitude, request.client_longitude)

        best_node = None
        best_latency = float('inf')

        for node in nodes:
            # Calculate network latency
            if client_location:
                node_location = (node.latitude, node.longitude)
                distance_km = geodesic(client_location, node_location).kilometers
                # Rough estimate: 5ms per 1000km
                network_latency = (distance_km / 1000) * 5
            else:
                network_latency = 10  # Default estimate

            # Add processing latency
            processing_latency = node.avg_response_time_ms

            # Add queueing delay
            queue_delay = (node.active_connections / node.capacity) * 10

            # Total expected latency
            total_latency = network_latency + processing_latency + queue_delay

            # Add penalty if present
            total_latency += getattr(node, '_temp_penalty', 0)

            if total_latency < best_latency:
                best_latency = total_latency
                best_node = node

        return best_node or nodes[0]

    async def _adaptive_select(self, nodes: List[EdgeNode],
                              request: RoutingRequest) -> EdgeNode:
        """Adaptive selection using multiple algorithms with weighting"""
        scores = defaultdict(float)

        # Get selections from different algorithms
        algorithms = [
            (LoadBalancingAlgorithm.LEAST_CONNECTIONS, 0.3),
            (LoadBalancingAlgorithm.LEAST_RESPONSE_TIME, 0.3),
            (LoadBalancingAlgorithm.GEO_PROXIMITY, 0.2),
            (LoadBalancingAlgorithm.LATENCY_AWARE, 0.2)
        ]

        for algorithm, weight in algorithms:
            if algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                selected = await self._least_connections_select(nodes)
            elif algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
                selected = await self._least_response_time_select(nodes)
            elif algorithm == LoadBalancingAlgorithm.GEO_PROXIMITY:
                selected = await self._geo_proximity_select(nodes, request)
            elif algorithm == LoadBalancingAlgorithm.LATENCY_AWARE:
                selected = await self._latency_aware_select(nodes, request)
            else:
                continue

            scores[selected.node_id] += weight

        # Select node with highest combined score
        best_node_id = max(scores, key=scores.get)
        return self.nodes[best_node_id]

    async def _health_check_loop(self):
        """Background loop for health checking nodes"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                # Check all nodes
                tasks = [self._check_node_health(node) for node in self.nodes.values()]
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    async def _check_node_health(self, node: EdgeNode):
        """Check health of individual node"""
        try:
            # Perform health check
            async with aiohttp.ClientSession() as session:
                url = f"http://{node.hostname}:{node.port}/health"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Update metrics
                        node.cpu_usage = data.get('cpu_usage', 0)
                        node.memory_usage = data.get('memory_usage', 0)
                        node.bandwidth_mbps = data.get('bandwidth_mbps', 0)

                        # Mark as healthy
                        if node.health != NodeHealth.HEALTHY:
                            node.failure_count = max(0, node.failure_count - 1)
                            if node.failure_count == 0:
                                node.health = NodeHealth.HEALTHY
                                logger.info(f"Node {node.node_id} is now healthy")
                    else:
                        await self._handle_health_check_failure(node)

        except Exception as e:
            await self._handle_health_check_failure(node)

    async def _handle_health_check_failure(self, node: EdgeNode):
        """Handle health check failure"""
        node.failure_count += 1

        if node.failure_count >= self.unhealthy_threshold:
            if node.health != NodeHealth.UNHEALTHY:
                node.health = NodeHealth.UNHEALTHY
                logger.warning(f"Node {node.node_id} marked as unhealthy")

                # Open circuit breaker
                node.circuit_open = True
                node.circuit_open_until = datetime.now(timezone.utc) + self.circuit_breaker_timeout
        elif node.failure_count >= self.unhealthy_threshold // 2:
            node.health = NodeHealth.DEGRADED

        node.last_health_check = datetime.now(timezone.utc)

    def _build_consistent_hash_ring(self):
        """Build consistent hash ring for node distribution"""
        self.consistent_hash_ring.clear()

        # Add virtual nodes for better distribution
        virtual_nodes = 150

        for node in self.nodes.values():
            for i in range(virtual_nodes):
                virtual_key = f"{node.node_id}:{i}"
                hash_value = int(hashlib.md5(virtual_key.encode()).hexdigest(), 16)
                self.consistent_hash_ring[hash_value] = node.node_id

    def _calculate_load_variance(self, nodes: List[EdgeNode]) -> float:
        """Calculate load variance across nodes"""
        if not nodes:
            return 0.0

        loads = [n.active_connections / n.capacity for n in nodes]
        mean_load = sum(loads) / len(loads)
        variance = sum((load - mean_load) ** 2 for load in loads) / len(loads)
        return variance ** 0.5  # Standard deviation

    def _create_routing_decision(self, request: RoutingRequest, selected: EdgeNode,
                                algorithm: LoadBalancingAlgorithm,
                                decision_time: float,
                                alternatives: List[EdgeNode] = None) -> RoutingDecision:
        """Create routing decision object"""
        # Calculate expected latency
        if request.client_latitude and request.client_longitude:
            client_location = (request.client_latitude, request.client_longitude)
            node_location = (selected.latitude, selected.longitude)
            distance_km = geodesic(client_location, node_location).kilometers
            network_latency = (distance_km / 1000) * 5
        else:
            network_latency = 10

        expected_latency = network_latency + selected.avg_response_time_ms

        return RoutingDecision(
            request_id=request.request_id,
            selected_node=selected,
            algorithm_used=algorithm,
            decision_score=1.0,
            expected_latency_ms=expected_latency,
            alternatives=alternatives[:3] if alternatives else [],
            decision_time_ms=decision_time * 1000
        )

    async def _load_node_configuration(self):
        """Load node configuration from config"""
        # US East nodes
        self.nodes["thanos-primary"] = EdgeNode(
            node_id="thanos-primary",
            region=Region.US_EAST,
            hostname="thanos.bev.internal",
            ip_address="10.0.1.10",
            port=8080,
            capacity=800,
            weight=2.0,
            latitude=40.7128,
            longitude=-74.0060
        )

        self.nodes["thanos-secondary"] = EdgeNode(
            node_id="thanos-secondary",
            region=Region.US_EAST,
            hostname="thanos-2.bev.internal",
            ip_address="10.0.1.11",
            port=8080,
            capacity=600,
            weight=1.5,
            latitude=40.7128,
            longitude=-74.0060
        )

        # US West nodes
        self.nodes["oracle1"] = EdgeNode(
            node_id="oracle1",
            region=Region.US_WEST,
            hostname="oracle1.bev.internal",
            ip_address="10.0.2.10",
            port=8080,
            capacity=600,
            weight=1.5,
            latitude=37.7749,
            longitude=-122.4194
        )

        # Europe nodes
        self.nodes["edge-eu-1"] = EdgeNode(
            node_id="edge-eu-1",
            region=Region.EUROPE,
            hostname="edge-eu-1.bev.internal",
            ip_address="10.0.3.10",
            port=8080,
            capacity=400,
            weight=1.0,
            latitude=51.5074,
            longitude=-0.1278
        )

        # Asia Pacific nodes
        self.nodes["edge-ap-1"] = EdgeNode(
            node_id="edge-ap-1",
            region=Region.ASIA_PACIFIC,
            hostname="edge-ap-1.bev.internal",
            ip_address="10.0.4.10",
            port=8080,
            capacity=400,
            weight=1.0,
            latitude=35.6762,
            longitude=139.6503
        )

        # Build region mapping
        for node in self.nodes.values():
            self.nodes_by_region[node.region].append(node)

    async def _session_cleanup_loop(self):
        """Clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                now = datetime.now(timezone.utc)
                expired_sessions = []

                for session_id, timestamp in self.session_timestamps.items():
                    if now - timestamp > self.session_ttl:
                        expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    del self.session_mapping[session_id]
                    del self.session_timestamps[session_id]

                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    async def _algorithm_optimization_loop(self):
        """Optimize algorithm weights based on performance"""
        while True:
            try:
                await asyncio.sleep(600)  # Every 10 minutes

                # Analyze recent routing decisions
                if len(self.routing_decisions) >= 100:
                    await self._optimize_algorithm_weights()

            except Exception as e:
                logger.error(f"Algorithm optimization error: {e}")

    async def _optimize_algorithm_weights(self):
        """Optimize algorithm weights based on performance data"""
        # Calculate performance scores for each algorithm
        algorithm_scores = {}

        for algorithm in LoadBalancingAlgorithm:
            perf = self.algorithm_performance[algorithm]
            if perf['usage_count'] > 0:
                # Score based on success rate and latency
                score = (
                    perf['success_rate'] * 0.7 +
                    (1 - min(perf['avg_latency'] / 100, 1)) * 0.3
                )
                algorithm_scores[algorithm] = score

        # Update weights
        if algorithm_scores:
            total_score = sum(algorithm_scores.values())
            if total_score > 0:
                for algorithm, score in algorithm_scores.items():
                    if algorithm in self.algorithm_weights:
                        # Adjust weight gradually
                        new_weight = score / total_score
                        old_weight = self.algorithm_weights[algorithm]
                        self.algorithm_weights[algorithm] = old_weight * 0.7 + new_weight * 0.3

    async def _get_session_node(self, session_id: str) -> Optional[EdgeNode]:
        """Get node for existing session"""
        node_id = self.session_mapping.get(session_id)
        if node_id and node_id in self.nodes:
            return self.nodes[node_id]
        return None

    async def _update_session_mapping(self, session_id: str, node_id: str):
        """Update session to node mapping"""
        self.session_mapping[session_id] = node_id
        self.session_timestamps[session_id] = datetime.now(timezone.utc)

    async def _select_by_performance(self) -> LoadBalancingAlgorithm:
        """Select algorithm based on historical performance"""
        # Use weighted random selection
        algorithms = list(self.algorithm_weights.keys())
        weights = list(self.algorithm_weights.values())

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return LoadBalancingAlgorithm.ROUND_ROBIN

        weights = [w / total_weight for w in weights]

        # Weighted random selection
        return random.choices(algorithms, weights=weights)[0]

    async def _metrics_collection_loop(self):
        """Collect and update metrics"""
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds

                # Update node metrics
                for node in self.nodes.values():
                    self.node_health_gauge.labels(
                        node_id=node.node_id,
                        region=node.region.value
                    ).set(1 if node.health == NodeHealth.HEALTHY else 0)

                    self.active_connections_gauge.labels(
                        node_id=node.node_id,
                        region=node.region.value
                    ).set(node.active_connections)

                    # Calculate response time percentiles
                    if node.response_times:
                        times = list(node.response_times)
                        times.sort()
                        node.avg_response_time_ms = sum(times) / len(times)
                        node.p95_response_time_ms = times[int(len(times) * 0.95)]
                        node.p99_response_time_ms = times[int(len(times) * 0.99)]

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")

    async def report_request_completion(self, request_id: str, node_id: str,
                                       response_time_ms: float, success: bool):
        """Report request completion for metrics tracking"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.active_connections = max(0, node.active_connections - 1)

            if success:
                node.response_times.append(response_time_ms)
            else:
                node.error_count += 1

                # Check if circuit breaker should open
                error_rate = node.error_count / max(1, node.total_requests)
                if error_rate > 0.5 and node.total_requests > 10:
                    node.circuit_open = True
                    node.circuit_open_until = datetime.now(timezone.utc) + self.circuit_breaker_timeout
                    logger.warning(f"Circuit breaker opened for node {node_id}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics"""
        total_capacity = sum(n.capacity for n in self.nodes.values())
        total_active = sum(n.active_connections for n in self.nodes.values())
        healthy_nodes = sum(1 for n in self.nodes.values() if n.health == NodeHealth.HEALTHY)

        region_stats = {}
        for region in Region:
            region_nodes = self.nodes_by_region[region]
            if region_nodes:
                region_stats[region.value] = {
                    'nodes': len(region_nodes),
                    'healthy': sum(1 for n in region_nodes if n.health == NodeHealth.HEALTHY),
                    'capacity': sum(n.capacity for n in region_nodes),
                    'active_connections': sum(n.active_connections for n in region_nodes),
                    'avg_response_time_ms': sum(n.avg_response_time_ms for n in region_nodes) / len(region_nodes)
                }

        return {
            'total_nodes': len(self.nodes),
            'healthy_nodes': healthy_nodes,
            'total_capacity': total_capacity,
            'active_connections': total_active,
            'utilization': total_active / total_capacity if total_capacity > 0 else 0,
            'region_stats': region_stats,
            'algorithm_weights': dict(self.algorithm_weights),
            'active_sessions': len(self.session_mapping),
            'recent_decisions': len(self.routing_decisions)
        }