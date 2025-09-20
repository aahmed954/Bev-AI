import os
"""
Edge Computing Network for BEV OSINT Framework

Provides geographic distribution, edge inference, and latency optimization
for the BEV OSINT framework with integration to multiplexing and caching systems.
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

# Edge Computing Network Configuration
@dataclass
class EdgeRegion:
    """Configuration for an edge computing region"""
    name: str
    code: str
    country: str
    timezone: str
    inference_endpoint: str
    model_endpoint: str
    cache_endpoint: str
    multiplex_endpoint: str
    ip_address: str
    latitude: float
    longitude: float
    capacity: int
    model_variants: List[str]

# Geographic regions configuration
EDGE_REGIONS = {
    "us-east": EdgeRegion(
        name="US East",
        code="us-east",
        country="USA",
        timezone="America/New_York",
        inference_endpoint="http://172.30.0.47:8000",
        model_endpoint="http://172.30.0.47:8001",
        cache_endpoint="http://172.30.0.44:6379",
        multiplex_endpoint="http://172.30.0.42:8080",
        ip_address="172.30.0.47",
        latitude=40.7128,
        longitude=-74.0060,
        capacity=1000,
        model_variants=["llama-3-8b", "mistral-7b", "phi-3-mini"]
    ),
    "us-west": EdgeRegion(
        name="US West",
        code="us-west",
        country="USA",
        timezone="America/Los_Angeles",
        inference_endpoint="http://172.30.0.48:8000",
        model_endpoint="http://172.30.0.48:8001",
        cache_endpoint="http://172.30.0.44:6379",
        multiplex_endpoint="http://172.30.0.42:8080",
        ip_address="172.30.0.48",
        latitude=37.7749,
        longitude=-122.4194,
        capacity=1000,
        model_variants=["llama-3-8b", "mistral-7b", "phi-3-mini"]
    ),
    "eu-central": EdgeRegion(
        name="EU Central",
        code="eu-central",
        country="Germany",
        timezone="Europe/Berlin",
        inference_endpoint="http://172.30.0.49:8000",
        model_endpoint="http://172.30.0.49:8001",
        cache_endpoint="http://172.30.0.44:6379",
        multiplex_endpoint="http://172.30.0.42:8080",
        ip_address="172.30.0.49",
        latitude=52.5200,
        longitude=13.4050,
        capacity=800,
        model_variants=["llama-3-8b", "mistral-7b"]
    ),
    "asia-pacific": EdgeRegion(
        name="Asia Pacific",
        code="asia-pacific",
        country="Singapore",
        timezone="Asia/Singapore",
        inference_endpoint="http://172.30.0.50:8000",
        model_endpoint="http://172.30.0.50:8001",
        cache_endpoint="http://172.30.0.44:6379",
        multiplex_endpoint="http://172.30.0.42:8080",
        ip_address="172.30.0.50",
        latitude=1.3521,
        longitude=103.8198,
        capacity=600,
        model_variants=["llama-3-8b", "phi-3-mini"]
    )
}

class RequestType(Enum):
    """Types of requests that can be processed by edge nodes"""
    INFERENCE = "inference"
    OSINT_ANALYSIS = "osint_analysis"
    DATA_PROCESSING = "data_processing"
    PATTERN_RECOGNITION = "pattern_recognition"
    THREAT_DETECTION = "threat_detection"

@dataclass
class EdgeRequest:
    """Request to be processed by edge computing network"""
    request_id: str
    request_type: RequestType
    payload: Dict[str, Any]
    priority: int
    max_latency_ms: int
    client_ip: str
    client_region: Optional[str]
    timestamp: datetime
    model_preference: Optional[str] = None
    cache_key: Optional[str] = None

@dataclass
class EdgeResponse:
    """Response from edge computing network"""
    request_id: str
    region: str
    response_data: Dict[str, Any]
    processing_time_ms: int
    latency_ms: int
    model_used: str
    cache_hit: bool
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

class EdgeComputeNetwork:
    """
    Edge Computing Network Manager

    Manages geographic distribution, load balancing, and edge inference
    across multiple regions with integration to multiplexing and caching.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.regions = EDGE_REGIONS.copy()
        self.active_regions: Dict[str, bool] = {}
        self.region_loads: Dict[str, float] = {}
        self.region_latencies: Dict[str, float] = {}
        self.request_history: List[EdgeRequest] = []
        self.response_history: List[EdgeResponse] = []

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.request_counter = Counter(
            'edge_requests_total',
            'Total edge requests',
            ['region', 'request_type', 'success'],
            registry=self.registry
        )
        self.latency_histogram = Histogram(
            'edge_request_latency_seconds',
            'Edge request latency',
            ['region', 'request_type'],
            registry=self.registry
        )
        self.active_connections = Gauge(
            'edge_active_connections',
            'Active connections per region',
            ['region'],
            registry=self.registry
        )
        self.model_usage = Counter(
            'edge_model_usage_total',
            'Model usage counter',
            ['region', 'model'],
            registry=self.registry
        )

        # Session management
        self.session: Optional[aiohttp.ClientSession] = None
        self.db_pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize edge computing network"""
        try:
            self.logger.info("Initializing Edge Computing Network")

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
                min_size=5,
                max_size=20
            )

            # Create database tables
            await self._create_tables()

            # Check region health
            await self._check_region_health()

            # Initialize region loads
            for region_code in self.regions.keys():
                self.region_loads[region_code] = 0.0
                self.region_latencies[region_code] = 100.0  # Default 100ms

            self.logger.info(f"Edge network initialized with {len(self.active_regions)} regions")

        except Exception as e:
            self.logger.error(f"Failed to initialize edge network: {e}")
            raise

    async def _create_tables(self):
        """Create database tables for edge computing"""
        async with self.db_pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_requests (
                    request_id VARCHAR(64) PRIMARY KEY,
                    request_type VARCHAR(32) NOT NULL,
                    payload JSONB NOT NULL,
                    priority INTEGER NOT NULL,
                    max_latency_ms INTEGER NOT NULL,
                    client_ip INET NOT NULL,
                    client_region VARCHAR(32),
                    model_preference VARCHAR(32),
                    cache_key VARCHAR(128),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_region VARCHAR(32),
                    processing_time_ms INTEGER,
                    latency_ms INTEGER,
                    success BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS edge_regions (
                    region_code VARCHAR(32) PRIMARY KEY,
                    region_name VARCHAR(64) NOT NULL,
                    country VARCHAR(32) NOT NULL,
                    ip_address INET NOT NULL,
                    inference_endpoint VARCHAR(128) NOT NULL,
                    model_endpoint VARCHAR(128) NOT NULL,
                    latitude DECIMAL(10,8) NOT NULL,
                    longitude DECIMAL(11,8) NOT NULL,
                    capacity INTEGER NOT NULL,
                    current_load DECIMAL(5,2) DEFAULT 0.0,
                    avg_latency_ms DECIMAL(8,2) DEFAULT 100.0,
                    is_active BOOLEAN DEFAULT TRUE,
                    last_health_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    model_variants JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Insert region data
            for region_code, region in self.regions.items():
                await conn.execute("""
                    INSERT INTO edge_regions
                    (region_code, region_name, country, ip_address, inference_endpoint,
                     model_endpoint, latitude, longitude, capacity, model_variants)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (region_code) DO UPDATE SET
                        region_name = EXCLUDED.region_name,
                        country = EXCLUDED.country,
                        ip_address = EXCLUDED.ip_address,
                        inference_endpoint = EXCLUDED.inference_endpoint,
                        model_endpoint = EXCLUDED.model_endpoint,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        capacity = EXCLUDED.capacity,
                        model_variants = EXCLUDED.model_variants,
                        updated_at = CURRENT_TIMESTAMP
                """, region_code, region.name, region.country, region.ip_address,
                region.inference_endpoint, region.model_endpoint,
                region.latitude, region.longitude, region.capacity,
                json.dumps(region.model_variants))

    async def _check_region_health(self):
        """Check health status of all edge regions"""
        health_tasks = []
        for region_code, region in self.regions.items():
            task = self._check_single_region_health(region_code, region)
            health_tasks.append(task)

        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)

        for i, (region_code, result) in enumerate(zip(self.regions.keys(), health_results)):
            if isinstance(result, Exception):
                self.active_regions[region_code] = False
                self.logger.warning(f"Region {region_code} health check failed: {result}")
            else:
                self.active_regions[region_code] = result

    async def _check_single_region_health(self, region_code: str, region: EdgeRegion) -> bool:
        """Check health of a single region"""
        try:
            # Health check with timeout
            start_time = time.time()
            async with self.session.get(f"{region.inference_endpoint}/health") as response:
                if response.status == 200:
                    latency = (time.time() - start_time) * 1000  # Convert to milliseconds
                    self.region_latencies[region_code] = latency

                    # Update database
                    async with self.db_pool.acquire() as conn:
                        await conn.execute("""
                            UPDATE edge_regions
                            SET avg_latency_ms = $1, is_active = TRUE, last_health_check = CURRENT_TIMESTAMP
                            WHERE region_code = $2
                        """, latency, region_code)

                    return True

        except Exception as e:
            self.logger.warning(f"Health check failed for region {region_code}: {e}")

            # Update database
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        UPDATE edge_regions
                        SET is_active = FALSE, last_health_check = CURRENT_TIMESTAMP
                        WHERE region_code = $2
                    """, region_code)
            except Exception as db_error:
                self.logger.error(f"Database update failed for region {region_code}: {db_error}")

        return False

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points"""
        import math

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Earth's radius in kilometers
        r = 6371

        return c * r

    def select_optimal_region(self, request: EdgeRequest) -> str:
        """
        Select optimal edge region for request processing

        Factors considered:
        1. Geographic proximity (if client location available)
        2. Current load
        3. Average latency
        4. Model availability
        5. Regional capacity
        """
        scores = {}

        for region_code, region in self.regions.items():
            if not self.active_regions.get(region_code, False):
                continue

            score = 100.0  # Base score

            # Geographic proximity (if available)
            if request.client_region and request.client_region in self.regions:
                client_region = self.regions[request.client_region]
                distance = self.calculate_distance(
                    client_region.latitude, client_region.longitude,
                    region.latitude, region.longitude
                )
                # Penalize distant regions
                score -= min(distance / 100.0, 50.0)  # Max penalty of 50 points

            # Load balancing (lower load = higher score)
            current_load = self.region_loads.get(region_code, 0.0)
            score -= current_load * 30.0  # Up to 30 point penalty for high load

            # Latency consideration (lower latency = higher score)
            avg_latency = self.region_latencies.get(region_code, 100.0)
            score -= min(avg_latency / 10.0, 20.0)  # Up to 20 point penalty for high latency

            # Model availability
            if request.model_preference and request.model_preference in region.model_variants:
                score += 20.0  # Bonus for having preferred model
            elif request.model_preference and request.model_preference not in region.model_variants:
                score -= 30.0  # Penalty for not having preferred model

            # Capacity consideration
            if current_load > 0.8:  # If over 80% capacity
                score -= 40.0  # Heavy penalty
            elif current_load > 0.6:  # If over 60% capacity
                score -= 20.0  # Medium penalty

            scores[region_code] = score

        # Select region with highest score
        if not scores:
            raise Exception("No active regions available")

        optimal_region = max(scores.items(), key=lambda x: x[1])[0]

        self.logger.debug(f"Region selection scores: {scores}, selected: {optimal_region}")
        return optimal_region

    async def process_request(self, request: EdgeRequest) -> EdgeResponse:
        """Process request through edge computing network"""
        start_time = time.time()

        try:
            # Select optimal region
            selected_region = self.select_optimal_region(request)
            region = self.regions[selected_region]

            # Update region load
            self.region_loads[selected_region] += 0.1
            self.active_connections.labels(region=selected_region).inc()

            # Check cache first (integrate with predictive cache system)
            cache_hit = False
            cached_response = None

            if request.cache_key:
                cached_response = await self._check_cache(request.cache_key, region)
                if cached_response:
                    cache_hit = True

            if not cache_hit:
                # Process request through edge inference
                response_data = await self._process_inference_request(request, region)

                # Cache response if applicable
                if request.cache_key and response_data:
                    await self._cache_response(request.cache_key, response_data, region)

            else:
                response_data = cached_response

            processing_time = (time.time() - start_time) * 1000

            # Create response
            response = EdgeResponse(
                request_id=request.request_id,
                region=selected_region,
                response_data=response_data,
                processing_time_ms=int(processing_time),
                latency_ms=int(self.region_latencies.get(selected_region, 100.0)),
                model_used=request.model_preference or "llama-3-8b",
                cache_hit=cache_hit,
                timestamp=datetime.utcnow(),
                success=True
            )

            # Update metrics
            self.request_counter.labels(
                region=selected_region,
                request_type=request.request_type.value,
                success="true"
            ).inc()

            self.latency_histogram.labels(
                region=selected_region,
                request_type=request.request_type.value
            ).observe(processing_time / 1000.0)

            self.model_usage.labels(
                region=selected_region,
                model=response.model_used
            ).inc()

            # Store request/response in database
            await self._store_request_response(request, response)

            return response

        except Exception as e:
            self.logger.error(f"Failed to process request {request.request_id}: {e}")

            # Create error response
            response = EdgeResponse(
                request_id=request.request_id,
                region="error",
                response_data={"error": str(e)},
                processing_time_ms=int((time.time() - start_time) * 1000),
                latency_ms=0,
                model_used="none",
                cache_hit=False,
                timestamp=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )

            # Update error metrics
            self.request_counter.labels(
                region="error",
                request_type=request.request_type.value,
                success="false"
            ).inc()

            return response

        finally:
            # Cleanup region load tracking
            if 'selected_region' in locals():
                self.region_loads[selected_region] = max(0.0, self.region_loads[selected_region] - 0.1)
                self.active_connections.labels(region=selected_region).dec()

    async def _check_cache(self, cache_key: str, region: EdgeRegion) -> Optional[Dict[str, Any]]:
        """Check predictive cache for cached response"""
        try:
            async with self.session.get(f"{region.cache_endpoint}/get/{cache_key}") as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            self.logger.debug(f"Cache check failed: {e}")
        return None

    async def _cache_response(self, cache_key: str, response_data: Dict[str, Any], region: EdgeRegion):
        """Cache response in predictive cache system"""
        try:
            cache_payload = {
                "key": cache_key,
                "value": response_data,
                "ttl": 3600  # 1 hour TTL
            }
            async with self.session.post(f"{region.cache_endpoint}/set", json=cache_payload) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to cache response: {response.status}")
        except Exception as e:
            self.logger.debug(f"Cache storage failed: {e}")

    async def _process_inference_request(self, request: EdgeRequest, region: EdgeRegion) -> Dict[str, Any]:
        """Process inference request through edge node"""
        inference_payload = {
            "request_id": request.request_id,
            "request_type": request.request_type.value,
            "payload": request.payload,
            "model": request.model_preference or "llama-3-8b",
            "max_tokens": 2048,
            "temperature": 0.7
        }

        async with self.session.post(f"{region.inference_endpoint}/inference", json=inference_payload) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Inference failed with status {response.status}: {error_text}")

    async def _store_request_response(self, request: EdgeRequest, response: EdgeResponse):
        """Store request and response in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO edge_requests
                    (request_id, request_type, payload, priority, max_latency_ms,
                     client_ip, client_region, model_preference, cache_key, timestamp,
                     processed_region, processing_time_ms, latency_ms, success)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                """, request.request_id, request.request_type.value, json.dumps(request.payload),
                request.priority, request.max_latency_ms, request.client_ip, request.client_region,
                request.model_preference, request.cache_key, request.timestamp,
                response.region, response.processing_time_ms, response.latency_ms, response.success)
        except Exception as e:
            self.logger.error(f"Failed to store request/response: {e}")

    async def get_network_status(self) -> Dict[str, Any]:
        """Get current network status and metrics"""
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "regions": {},
            "total_regions": len(self.regions),
            "active_regions": sum(1 for active in self.active_regions.values() if active),
            "total_capacity": sum(region.capacity for region in self.regions.values()),
            "average_latency": sum(self.region_latencies.values()) / len(self.region_latencies) if self.region_latencies else 0,
            "request_count_last_hour": 0,
            "cache_hit_rate": 0.0
        }

        # Get detailed region status
        for region_code, region in self.regions.items():
            status["regions"][region_code] = {
                "name": region.name,
                "active": self.active_regions.get(region_code, False),
                "current_load": self.region_loads.get(region_code, 0.0),
                "avg_latency_ms": self.region_latencies.get(region_code, 100.0),
                "capacity": region.capacity,
                "model_variants": region.model_variants,
                "endpoints": {
                    "inference": region.inference_endpoint,
                    "model": region.model_endpoint,
                    "cache": region.cache_endpoint,
                    "multiplex": region.multiplex_endpoint
                }
            }

        # Get request statistics
        try:
            async with self.db_pool.acquire() as conn:
                # Request count last hour
                result = await conn.fetchval("""
                    SELECT COUNT(*) FROM edge_requests
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                """)
                status["request_count_last_hour"] = result or 0

                # Cache hit rate
                cache_stats = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_requests,
                        COUNT(*) FILTER (WHERE processed_region = 'cache') as cache_hits
                    FROM edge_requests
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
                """)

                if cache_stats and cache_stats['total_requests'] > 0:
                    status["cache_hit_rate"] = cache_stats['cache_hits'] / cache_stats['total_requests']

        except Exception as e:
            self.logger.error(f"Failed to get request statistics: {e}")

        return status

    async def optimize_network(self):
        """Optimize network performance based on metrics"""
        try:
            # Update region health
            await self._check_region_health()

            # Load balancing optimization
            await self._optimize_load_balancing()

            # Cache optimization
            await self._optimize_cache_distribution()

            self.logger.info("Network optimization completed")

        except Exception as e:
            self.logger.error(f"Network optimization failed: {e}")

    async def _optimize_load_balancing(self):
        """Optimize load balancing across regions"""
        # Implement load balancing optimization logic
        # This could include:
        # - Redistributing traffic based on current loads
        # - Scaling up/down regional capacity
        # - Updating routing preferences
        pass

    async def _optimize_cache_distribution(self):
        """Optimize cache distribution across regions"""
        # Implement cache optimization logic
        # This could include:
        # - Prewarming frequently accessed data
        # - Redistributing cache contents based on access patterns
        # - Adjusting cache TTL based on regional patterns
        pass

    async def cleanup(self):
        """Cleanup network resources"""
        try:
            if self.session:
                await self.session.close()

            if self.db_pool:
                await self.db_pool.close()

            self.logger.info("Edge compute network cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

# Example usage
async def main():
    """Example usage of EdgeComputeNetwork"""
    network = EdgeComputeNetwork()

    try:
        await network.initialize()

        # Example request
        request = EdgeRequest(
            request_id="test-001",
            request_type=RequestType.OSINT_ANALYSIS,
            payload={
                "target": "example.com",
                "analysis_type": "domain_intelligence",
                "depth": "comprehensive"
            },
            priority=1,
            max_latency_ms=5000,
            client_ip="192.168.1.100",
            client_region="us-east",
            timestamp=datetime.utcnow(),
            model_preference="llama-3-8b",
            cache_key="osint_domain_example.com"
        )

        # Process request
        response = await network.process_request(request)
        print(f"Response: {response}")

        # Get network status
        status = await network.get_network_status()
        print(f"Network Status: {status}")

    finally:
        await network.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())