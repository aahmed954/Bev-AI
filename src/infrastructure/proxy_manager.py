import os
#!/usr/bin/env python3
"""
Comprehensive Proxy Pool Management System for BEV OSINT Framework
Supports 10K+ residential proxies with geographic distribution and health checking

Features:
- Residential, datacenter, and rotating proxy pool support
- Geographic distribution (US-East, US-West, EU-Central, Asia-Pacific)
- Health checking with automatic failover (30-second intervals)
- Load balancing with multiple strategies (round-robin, least-connections, weighted)
- Integration with existing Tor infrastructure
- Performance target: 1000+ concurrent connections
- Real-time metrics and monitoring

Place in: /home/starlord/Projects/Bev/src/infrastructure/proxy_manager.py
"""

import asyncio
import aiohttp
import json
import logging
import time
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, AsyncGenerator
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import asyncpg
import aioredis
from urllib.parse import urlparse
import ssl
import socket
import struct
import ipaddress
import statistics
from concurrent.futures import ThreadPoolExecutor
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProxyType(Enum):
    """Proxy type enumeration"""
    RESIDENTIAL = "residential"
    DATACENTER = "datacenter"
    ROTATING = "rotating"
    TOR = "tor"
    MOBILE = "mobile"

class ProxyRegion(Enum):
    """Geographic regions for proxy distribution"""
    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_CENTRAL = "eu-central"
    EU_WEST = "eu-west"
    ASIA_PACIFIC = "asia-pacific"
    GLOBAL = "global"

class ProxyStatus(Enum):
    """Proxy health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    TESTING = "testing"
    DISABLED = "disabled"

class LoadBalanceStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"
    GEOGRAPHIC = "geographic"

@dataclass
class ProxyEndpoint:
    """Individual proxy endpoint configuration"""
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    proxy_type: ProxyType = ProxyType.DATACENTER
    region: ProxyRegion = ProxyRegion.GLOBAL
    weight: float = 1.0
    max_connections: int = 100
    rotation_interval: Optional[int] = None  # seconds

    # Health metrics
    status: ProxyStatus = ProxyStatus.TESTING
    last_check: Optional[datetime] = None
    response_time: Optional[float] = None
    success_rate: float = 0.0
    failure_count: int = 0
    total_requests: int = 0
    active_connections: int = 0

    # Provider information
    provider: Optional[str] = None
    provider_pool_id: Optional[str] = None
    cost_per_gb: Optional[float] = None

    @property
    def proxy_url(self) -> str:
        """Generate proxy URL"""
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"socks5://{auth}{self.host}:{self.port}"

    @property
    def http_proxy_url(self) -> str:
        """Generate HTTP proxy URL"""
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"http://{auth}{self.host}:{self.port}"

    @property
    def is_healthy(self) -> bool:
        """Check if proxy is healthy"""
        return self.status == ProxyStatus.HEALTHY

    @property
    def utilization(self) -> float:
        """Calculate current utilization percentage"""
        if self.max_connections <= 0:
            return 0.0
        return (self.active_connections / self.max_connections) * 100

class ProxyHealthChecker:
    """Health checking system for proxy endpoints"""

    def __init__(self, check_interval: int = 30, timeout: int = 10):
        self.check_interval = check_interval
        self.timeout = timeout
        self.test_urls = [
            "http://httpbin.org/ip",
            "https://api.ipify.org",
            "http://icanhazip.com"
        ]
        self.running = False
        self._check_tasks = weakref.WeakSet()

    async def start_health_checks(self, proxy_manager):
        """Start continuous health checking"""
        self.running = True
        logger.info(f"Starting health checks with {self.check_interval}s interval")

        while self.running:
            try:
                await self._run_health_check_cycle(proxy_manager)
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check cycle error: {e}")
                await asyncio.sleep(5)

    async def stop_health_checks(self):
        """Stop health checking"""
        self.running = False
        logger.info("Stopping health checks")

    async def _run_health_check_cycle(self, proxy_manager):
        """Run a complete health check cycle"""
        start_time = time.time()

        # Get all proxies grouped by region
        all_proxies = await proxy_manager.get_all_proxies()
        proxy_groups = defaultdict(list)

        for proxy in all_proxies:
            proxy_groups[proxy.region].append(proxy)

        # Check each region in parallel
        tasks = []
        for region, proxies in proxy_groups.items():
            task = asyncio.create_task(
                self._check_region_proxies(proxies, proxy_manager)
            )
            self._check_tasks.add(task)
            tasks.append(task)

        # Wait for all checks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        total_checked = 0
        total_healthy = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Region check failed: {result}")
            else:
                checked, healthy = result
                total_checked += checked
                total_healthy += healthy

        duration = time.time() - start_time
        health_rate = (total_healthy / total_checked) * 100 if total_checked > 0 else 0

        logger.info(
            f"Health check cycle completed: {total_healthy}/{total_checked} "
            f"healthy ({health_rate:.1f}%) in {duration:.2f}s"
        )

    async def _check_region_proxies(self, proxies: List[ProxyEndpoint], proxy_manager) -> Tuple[int, int]:
        """Check health of proxies in a specific region"""
        if not proxies:
            return 0, 0

        # Limit concurrent checks per region
        semaphore = asyncio.Semaphore(20)

        async def check_single_proxy(proxy):
            async with semaphore:
                return await self._check_proxy_health(proxy, proxy_manager)

        # Run health checks
        tasks = [check_single_proxy(proxy) for proxy in proxies]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        checked = len([r for r in results if not isinstance(r, Exception)])
        healthy = len([r for r in results if r is True])

        return checked, healthy

    async def _check_proxy_health(self, proxy: ProxyEndpoint, proxy_manager) -> bool:
        """Check health of a single proxy"""
        start_time = time.time()

        try:
            # Set proxy to testing status
            proxy.status = ProxyStatus.TESTING
            proxy.last_check = datetime.now()

            # Test proxy connectivity
            success = await self._test_proxy_connectivity(proxy)
            response_time = time.time() - start_time

            # Update proxy metrics
            proxy.response_time = response_time
            proxy.total_requests += 1

            if success:
                proxy.failure_count = 0
                proxy.success_rate = min(proxy.success_rate + 0.1, 1.0)

                # Determine status based on response time and success rate
                if response_time < 2.0 and proxy.success_rate >= 0.9:
                    proxy.status = ProxyStatus.HEALTHY
                elif response_time < 5.0 and proxy.success_rate >= 0.7:
                    proxy.status = ProxyStatus.DEGRADED
                else:
                    proxy.status = ProxyStatus.UNHEALTHY
            else:
                proxy.failure_count += 1
                proxy.success_rate = max(proxy.success_rate - 0.2, 0.0)

                # Mark as unhealthy after multiple failures
                if proxy.failure_count >= 3:
                    proxy.status = ProxyStatus.UNHEALTHY
                else:
                    proxy.status = ProxyStatus.DEGRADED

            # Update proxy in storage
            await proxy_manager.update_proxy_metrics(proxy)

            return success

        except Exception as e:
            logger.error(f"Health check failed for {proxy.host}:{proxy.port}: {e}")
            proxy.status = ProxyStatus.UNHEALTHY
            proxy.failure_count += 1
            await proxy_manager.update_proxy_metrics(proxy)
            return False

    async def _test_proxy_connectivity(self, proxy: ProxyEndpoint) -> bool:
        """Test proxy connectivity using multiple endpoints"""
        test_url = random.choice(self.test_urls)

        try:
            # Configure proxy for aiohttp
            proxy_url = proxy.http_proxy_url
            connector = aiohttp.ProxyConnector.from_url(proxy_url)

            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                async with session.get(test_url) as response:
                    if response.status == 200:
                        # Verify we're using the proxy by checking returned IP
                        content = await response.text()
                        return self._validate_proxy_response(content, proxy)
                    return False

        except Exception as e:
            logger.debug(f"Proxy test failed for {proxy.host}:{proxy.port}: {e}")
            return False

    def _validate_proxy_response(self, response_content: str, proxy: ProxyEndpoint) -> bool:
        """Validate that the response came through the proxy"""
        try:
            # Basic validation - if we got a response, proxy is working
            # In production, you might want to validate the returned IP
            return len(response_content.strip()) > 0
        except:
            return False

class LoadBalancer:
    """Advanced load balancing for proxy distribution"""

    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.LEAST_CONNECTIONS):
        self.strategy = strategy
        self.region_weights = {
            ProxyRegion.US_EAST: 1.0,
            ProxyRegion.US_WEST: 1.0,
            ProxyRegion.EU_CENTRAL: 0.8,
            ProxyRegion.EU_WEST: 0.8,
            ProxyRegion.ASIA_PACIFIC: 0.6,
            ProxyRegion.GLOBAL: 0.5
        }
        self.round_robin_counters = defaultdict(int)
        self.response_time_history = defaultdict(lambda: deque(maxlen=100))

    async def select_proxy(self,
                          available_proxies: List[ProxyEndpoint],
                          region_preference: Optional[ProxyRegion] = None,
                          proxy_type_preference: Optional[ProxyType] = None) -> Optional[ProxyEndpoint]:
        """Select optimal proxy based on load balancing strategy"""

        if not available_proxies:
            return None

        # Filter by preferences
        filtered_proxies = self._filter_proxies(
            available_proxies, region_preference, proxy_type_preference
        )

        if not filtered_proxies:
            filtered_proxies = available_proxies

        # Apply load balancing strategy
        if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(filtered_proxies)
        elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(filtered_proxies)
        elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(filtered_proxies)
        elif self.strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(filtered_proxies)
        elif self.strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(filtered_proxies)
        elif self.strategy == LoadBalanceStrategy.GEOGRAPHIC:
            return self._geographic_select(filtered_proxies, region_preference)
        else:
            return self._least_connections_select(filtered_proxies)

    def _filter_proxies(self,
                       proxies: List[ProxyEndpoint],
                       region_preference: Optional[ProxyRegion],
                       proxy_type_preference: Optional[ProxyType]) -> List[ProxyEndpoint]:
        """Filter proxies based on preferences"""

        filtered = [p for p in proxies if p.is_healthy and p.utilization < 95]

        if region_preference:
            region_filtered = [p for p in filtered if p.region == region_preference]
            if region_filtered:
                filtered = region_filtered

        if proxy_type_preference:
            type_filtered = [p for p in filtered if p.proxy_type == proxy_type_preference]
            if type_filtered:
                filtered = type_filtered

        return filtered

    def _round_robin_select(self, proxies: List[ProxyEndpoint]) -> ProxyEndpoint:
        """Round-robin selection"""
        if not proxies:
            return None

        key = "global"
        counter = self.round_robin_counters[key]
        selected = proxies[counter % len(proxies)]
        self.round_robin_counters[key] = (counter + 1) % len(proxies)

        return selected

    def _least_connections_select(self, proxies: List[ProxyEndpoint]) -> ProxyEndpoint:
        """Select proxy with least active connections"""
        return min(proxies, key=lambda p: p.active_connections)

    def _weighted_round_robin_select(self, proxies: List[ProxyEndpoint]) -> ProxyEndpoint:
        """Weighted round-robin based on proxy weight and region"""

        # Calculate total weight
        total_weight = sum(
            p.weight * self.region_weights.get(p.region, 1.0)
            for p in proxies
        )

        if total_weight <= 0:
            return random.choice(proxies)

        # Random selection based on weights
        r = random.uniform(0, total_weight)
        current_weight = 0

        for proxy in proxies:
            current_weight += proxy.weight * self.region_weights.get(proxy.region, 1.0)
            if r <= current_weight:
                return proxy

        return proxies[-1]  # Fallback

    def _least_response_time_select(self, proxies: List[ProxyEndpoint]) -> ProxyEndpoint:
        """Select proxy with best average response time"""

        def avg_response_time(proxy):
            if proxy.response_time is None:
                return float('inf')

            history = self.response_time_history[f"{proxy.host}:{proxy.port}"]
            if not history:
                return proxy.response_time

            return statistics.mean(history)

        return min(proxies, key=avg_response_time)

    def _geographic_select(self,
                          proxies: List[ProxyEndpoint],
                          region_preference: Optional[ProxyRegion]) -> ProxyEndpoint:
        """Geographic-aware selection"""

        if region_preference:
            # Prefer proxies in the requested region
            region_proxies = [p for p in proxies if p.region == region_preference]
            if region_proxies:
                return self._least_connections_select(region_proxies)

        # Fallback to weighted selection based on region weights
        return self._weighted_round_robin_select(proxies)

    def update_response_time(self, proxy: ProxyEndpoint, response_time: float):
        """Update response time history for proxy"""
        key = f"{proxy.host}:{proxy.port}"
        self.response_time_history[key].append(response_time)

class ProxyRotationManager:
    """Manages automatic proxy rotation"""

    def __init__(self, default_rotation_interval: int = 1800):  # 30 minutes
        self.default_rotation_interval = default_rotation_interval
        self.rotation_tasks = {}
        self.rotation_history = defaultdict(list)

    async def start_rotation(self, proxy_manager):
        """Start automatic proxy rotation"""
        logger.info("Starting proxy rotation manager")

        while True:
            try:
                await self._check_rotation_needs(proxy_manager)
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Rotation manager error: {e}")
                await asyncio.sleep(30)

    async def _check_rotation_needs(self, proxy_manager):
        """Check which proxies need rotation"""
        current_time = datetime.now()

        # Get all rotating proxies
        all_proxies = await proxy_manager.get_all_proxies()
        rotating_proxies = [
            p for p in all_proxies
            if p.proxy_type == ProxyType.ROTATING and p.rotation_interval
        ]

        for proxy in rotating_proxies:
            if not proxy.last_check:
                continue

            rotation_interval = proxy.rotation_interval or self.default_rotation_interval
            time_since_last_rotation = (current_time - proxy.last_check).total_seconds()

            if time_since_last_rotation >= rotation_interval:
                await self._rotate_proxy(proxy, proxy_manager)

    async def _rotate_proxy(self, proxy: ProxyEndpoint, proxy_manager):
        """Rotate a specific proxy endpoint"""
        try:
            logger.info(f"Rotating proxy {proxy.host}:{proxy.port}")

            # Request new endpoint from provider (placeholder implementation)
            new_endpoint = await self._request_new_endpoint(proxy)

            if new_endpoint:
                # Update proxy configuration
                proxy.host = new_endpoint['host']
                proxy.port = new_endpoint['port']
                proxy.username = new_endpoint.get('username')
                proxy.password = new_endpoint.get('password')
                proxy.last_check = datetime.now()
                proxy.status = ProxyStatus.TESTING

                # Update in storage
                await proxy_manager.update_proxy_metrics(proxy)

                # Record rotation
                self.rotation_history[f"{proxy.provider}:{proxy.provider_pool_id}"].append(
                    datetime.now()
                )

                logger.info(f"Proxy rotated successfully: {proxy.host}:{proxy.port}")

        except Exception as e:
            logger.error(f"Failed to rotate proxy {proxy.host}:{proxy.port}: {e}")

    async def _request_new_endpoint(self, proxy: ProxyEndpoint) -> Optional[Dict]:
        """Request new endpoint from proxy provider"""
        # Placeholder implementation
        # In production, this would integrate with proxy provider APIs
        return {
            'host': f"new-{proxy.host}",
            'port': proxy.port,
            'username': proxy.username,
            'password': proxy.password
        }

class ProxyManager:
    """Main proxy pool management system"""

    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 postgres_url: str = "postgresql://localhost:5432/osint",
                 max_pool_size: int = 10000):

        self.redis_url = redis_url
        self.postgres_url = postgres_url
        self.max_pool_size = max_pool_size

        # Core components
        self.health_checker = ProxyHealthChecker()
        self.load_balancer = LoadBalancer()
        self.rotation_manager = ProxyRotationManager()

        # Storage
        self.redis_pool = None
        self.postgres_pool = None

        # In-memory proxy cache
        self.proxy_cache = {}
        self.cache_lock = asyncio.Lock()

        # Metrics
        self.metrics = {
            'total_proxies': 0,
            'healthy_proxies': 0,
            'total_requests': 0,
            'successful_requests': 0,
            'average_response_time': 0.0,
            'active_connections': 0
        }

        # Background tasks
        self.background_tasks = set()

        logger.info(f"ProxyManager initialized with max pool size: {max_pool_size}")

    async def initialize(self):
        """Initialize proxy manager and all components"""
        logger.info("Initializing ProxyManager...")

        try:
            # Initialize storage connections
            await self._initialize_storage()

            # Setup database schema
            await self._setup_database_schema()

            # Load proxy configurations
            await self._load_proxy_configurations()

            # Start background tasks
            await self._start_background_tasks()

            logger.info("ProxyManager initialization completed successfully")

        except Exception as e:
            logger.error(f"ProxyManager initialization failed: {e}")
            raise

    async def shutdown(self):
        """Graceful shutdown of proxy manager"""
        logger.info("Shutting down ProxyManager...")

        # Stop background tasks
        await self.health_checker.stop_health_checks()

        for task in self.background_tasks:
            task.cancel()

        await asyncio.gather(*self.background_tasks, return_exceptions=True)

        # Close storage connections
        if self.redis_pool:
            await self.redis_pool.close()

        if self.postgres_pool:
            await self.postgres_pool.close()

        logger.info("ProxyManager shutdown completed")

    async def _initialize_storage(self):
        """Initialize Redis and PostgreSQL connections"""

        # Initialize Redis
        self.redis_pool = aioredis.from_url(
            self.redis_url,
            max_connections=20,
            retry_on_timeout=True
        )

        # Test Redis connection
        await self.redis_pool.ping()
        logger.info("Redis connection established")

        # Initialize PostgreSQL
        self.postgres_pool = await asyncpg.create_pool(
            self.postgres_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )

        logger.info("PostgreSQL connection pool established")

    async def _setup_database_schema(self):
        """Setup PostgreSQL schema for proxy management"""

        schema_sql = """
        CREATE TABLE IF NOT EXISTS proxy_endpoints (
            id SERIAL PRIMARY KEY,
            host VARCHAR(255) NOT NULL,
            port INTEGER NOT NULL,
            username VARCHAR(255),
            password VARCHAR(255),
            proxy_type VARCHAR(50) NOT NULL,
            region VARCHAR(50) NOT NULL,
            weight REAL DEFAULT 1.0,
            max_connections INTEGER DEFAULT 100,
            rotation_interval INTEGER,
            status VARCHAR(50) DEFAULT 'testing',
            last_check TIMESTAMP,
            response_time REAL,
            success_rate REAL DEFAULT 0.0,
            failure_count INTEGER DEFAULT 0,
            total_requests INTEGER DEFAULT 0,
            active_connections INTEGER DEFAULT 0,
            provider VARCHAR(255),
            provider_pool_id VARCHAR(255),
            cost_per_gb REAL,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(host, port)
        );

        CREATE TABLE IF NOT EXISTS proxy_metrics (
            id SERIAL PRIMARY KEY,
            proxy_id INTEGER REFERENCES proxy_endpoints(id),
            timestamp TIMESTAMP DEFAULT NOW(),
            response_time REAL,
            success BOOLEAN,
            error_message TEXT,
            bytes_transferred BIGINT,
            request_type VARCHAR(50)
        );

        CREATE TABLE IF NOT EXISTS proxy_usage_stats (
            id SERIAL PRIMARY KEY,
            proxy_id INTEGER REFERENCES proxy_endpoints(id),
            date DATE DEFAULT CURRENT_DATE,
            total_requests INTEGER DEFAULT 0,
            successful_requests INTEGER DEFAULT 0,
            failed_requests INTEGER DEFAULT 0,
            bytes_transferred BIGINT DEFAULT 0,
            average_response_time REAL DEFAULT 0.0,
            max_concurrent_connections INTEGER DEFAULT 0,
            UNIQUE(proxy_id, date)
        );

        CREATE INDEX IF NOT EXISTS idx_proxy_endpoints_status ON proxy_endpoints(status);
        CREATE INDEX IF NOT EXISTS idx_proxy_endpoints_region ON proxy_endpoints(region);
        CREATE INDEX IF NOT EXISTS idx_proxy_endpoints_type ON proxy_endpoints(proxy_type);
        CREATE INDEX IF NOT EXISTS idx_proxy_metrics_timestamp ON proxy_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_proxy_usage_stats_date ON proxy_usage_stats(date);
        """

        async with self.postgres_pool.acquire() as conn:
            await conn.execute(schema_sql)

        logger.info("Database schema setup completed")

    async def _load_proxy_configurations(self):
        """Load proxy configurations from database"""

        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM proxy_endpoints
                WHERE status != 'disabled'
                ORDER BY region, proxy_type
            """)

        async with self.cache_lock:
            self.proxy_cache.clear()

            for row in rows:
                proxy = ProxyEndpoint(
                    host=row['host'],
                    port=row['port'],
                    username=row['username'],
                    password=row['password'],
                    proxy_type=ProxyType(row['proxy_type']),
                    region=ProxyRegion(row['region']),
                    weight=row['weight'],
                    max_connections=row['max_connections'],
                    rotation_interval=row['rotation_interval'],
                    status=ProxyStatus(row['status']),
                    last_check=row['last_check'],
                    response_time=row['response_time'],
                    success_rate=row['success_rate'],
                    failure_count=row['failure_count'],
                    total_requests=row['total_requests'],
                    active_connections=row['active_connections'],
                    provider=row['provider'],
                    provider_pool_id=row['provider_pool_id'],
                    cost_per_gb=row['cost_per_gb']
                )

                cache_key = f"{proxy.host}:{proxy.port}"
                self.proxy_cache[cache_key] = proxy

        logger.info(f"Loaded {len(self.proxy_cache)} proxy configurations")

    async def _start_background_tasks(self):
        """Start background maintenance tasks"""

        # Health checking
        health_task = asyncio.create_task(
            self.health_checker.start_health_checks(self)
        )
        self.background_tasks.add(health_task)

        # Proxy rotation
        rotation_task = asyncio.create_task(
            self.rotation_manager.start_rotation(self)
        )
        self.background_tasks.add(rotation_task)

        # Metrics collection
        metrics_task = asyncio.create_task(self._collect_metrics())
        self.background_tasks.add(metrics_task)

        # Cache refresh
        cache_task = asyncio.create_task(self._refresh_cache())
        self.background_tasks.add(cache_task)

        logger.info("Background tasks started")

    async def get_proxy(self,
                       region_preference: Optional[ProxyRegion] = None,
                       proxy_type_preference: Optional[ProxyType] = None,
                       load_balance_strategy: Optional[LoadBalanceStrategy] = None) -> Optional[ProxyEndpoint]:
        """Get an optimal proxy endpoint"""

        try:
            # Set load balancing strategy if provided
            if load_balance_strategy:
                self.load_balancer.strategy = load_balance_strategy

            # Get available proxies
            available_proxies = await self.get_healthy_proxies()

            if not available_proxies:
                logger.warning("No healthy proxies available")
                return None

            # Select optimal proxy
            selected_proxy = await self.load_balancer.select_proxy(
                available_proxies, region_preference, proxy_type_preference
            )

            if selected_proxy:
                # Increment active connections
                selected_proxy.active_connections += 1
                await self.update_proxy_metrics(selected_proxy)

                # Cache in Redis for quick access
                cache_key = f"active_proxy:{selected_proxy.host}:{selected_proxy.port}"
                await self.redis_pool.setex(
                    cache_key, 300, json.dumps(asdict(selected_proxy), default=str)
                )

                logger.debug(
                    f"Selected proxy: {selected_proxy.host}:{selected_proxy.port} "
                    f"({selected_proxy.region.value}, {selected_proxy.proxy_type.value})"
                )

            return selected_proxy

        except Exception as e:
            logger.error(f"Error selecting proxy: {e}")
            return None

    async def release_proxy(self, proxy: ProxyEndpoint, success: bool = True, response_time: Optional[float] = None):
        """Release a proxy endpoint after use"""

        try:
            # Decrement active connections
            proxy.active_connections = max(0, proxy.active_connections - 1)

            # Update metrics
            if response_time:
                proxy.response_time = response_time
                self.load_balancer.update_response_time(proxy, response_time)

            if success:
                proxy.success_rate = min(proxy.success_rate + 0.01, 1.0)
            else:
                proxy.success_rate = max(proxy.success_rate - 0.05, 0.0)
                proxy.failure_count += 1

            proxy.total_requests += 1

            # Update in storage
            await self.update_proxy_metrics(proxy)

            # Update global metrics
            self.metrics['total_requests'] += 1
            if success:
                self.metrics['successful_requests'] += 1

            if response_time:
                # Update average response time
                total_successful = self.metrics['successful_requests']
                if total_successful > 0:
                    current_avg = self.metrics['average_response_time']
                    self.metrics['average_response_time'] = (
                        (current_avg * (total_successful - 1) + response_time) / total_successful
                    )

            logger.debug(f"Released proxy: {proxy.host}:{proxy.port} (success: {success})")

        except Exception as e:
            logger.error(f"Error releasing proxy: {e}")

    async def get_healthy_proxies(self) -> List[ProxyEndpoint]:
        """Get all healthy proxy endpoints"""

        async with self.cache_lock:
            healthy_proxies = [
                proxy for proxy in self.proxy_cache.values()
                if proxy.status in [ProxyStatus.HEALTHY, ProxyStatus.DEGRADED]
                and proxy.utilization < 95
            ]

        return healthy_proxies

    async def get_all_proxies(self) -> List[ProxyEndpoint]:
        """Get all proxy endpoints"""

        async with self.cache_lock:
            return list(self.proxy_cache.values())

    async def add_proxy(self, proxy: ProxyEndpoint) -> bool:
        """Add a new proxy endpoint"""

        try:
            # Check if proxy already exists
            cache_key = f"{proxy.host}:{proxy.port}"
            if cache_key in self.proxy_cache:
                logger.warning(f"Proxy {proxy.host}:{proxy.port} already exists")
                return False

            # Check pool size limit
            if len(self.proxy_cache) >= self.max_pool_size:
                logger.warning(f"Proxy pool size limit reached ({self.max_pool_size})")
                return False

            # Insert into database
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO proxy_endpoints
                    (host, port, username, password, proxy_type, region, weight,
                     max_connections, rotation_interval, provider, provider_pool_id, cost_per_gb)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                    proxy.host, proxy.port, proxy.username, proxy.password,
                    proxy.proxy_type.value, proxy.region.value, proxy.weight,
                    proxy.max_connections, proxy.rotation_interval,
                    proxy.provider, proxy.provider_pool_id, proxy.cost_per_gb
                )

            # Add to cache
            async with self.cache_lock:
                self.proxy_cache[cache_key] = proxy

            logger.info(f"Added proxy: {proxy.host}:{proxy.port}")
            return True

        except Exception as e:
            logger.error(f"Error adding proxy {proxy.host}:{proxy.port}: {e}")
            return False

    async def remove_proxy(self, host: str, port: int) -> bool:
        """Remove a proxy endpoint"""

        try:
            cache_key = f"{host}:{port}"

            # Remove from database
            async with self.postgres_pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM proxy_endpoints
                    WHERE host = $1 AND port = $2
                """, host, port)

            # Remove from cache
            async with self.cache_lock:
                if cache_key in self.proxy_cache:
                    del self.proxy_cache[cache_key]

            logger.info(f"Removed proxy: {host}:{port}")
            return True

        except Exception as e:
            logger.error(f"Error removing proxy {host}:{port}: {e}")
            return False

    async def update_proxy_metrics(self, proxy: ProxyEndpoint):
        """Update proxy metrics in storage"""

        try:
            async with self.postgres_pool.acquire() as conn:
                await conn.execute("""
                    UPDATE proxy_endpoints
                    SET status = $1, last_check = $2, response_time = $3,
                        success_rate = $4, failure_count = $5, total_requests = $6,
                        active_connections = $7, updated_at = NOW()
                    WHERE host = $8 AND port = $9
                """,
                    proxy.status.value, proxy.last_check, proxy.response_time,
                    proxy.success_rate, proxy.failure_count, proxy.total_requests,
                    proxy.active_connections, proxy.host, proxy.port
                )

            # Update cache
            cache_key = f"{proxy.host}:{proxy.port}"
            async with self.cache_lock:
                if cache_key in self.proxy_cache:
                    self.proxy_cache[cache_key] = proxy

        except Exception as e:
            logger.error(f"Error updating proxy metrics: {e}")

    async def get_proxy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive proxy statistics"""

        try:
            stats = {
                'pool_size': len(self.proxy_cache),
                'by_status': defaultdict(int),
                'by_region': defaultdict(int),
                'by_type': defaultdict(int),
                'utilization': {
                    'total_capacity': 0,
                    'active_connections': 0,
                    'utilization_percentage': 0.0
                },
                'performance': {
                    'average_response_time': 0.0,
                    'success_rate': 0.0,
                    'top_performers': [],
                    'worst_performers': []
                }
            }

            async with self.cache_lock:
                total_capacity = 0
                total_active = 0
                response_times = []
                success_rates = []

                proxy_performance = []

                for proxy in self.proxy_cache.values():
                    # Count by status
                    stats['by_status'][proxy.status.value] += 1

                    # Count by region
                    stats['by_region'][proxy.region.value] += 1

                    # Count by type
                    stats['by_type'][proxy.proxy_type.value] += 1

                    # Utilization
                    total_capacity += proxy.max_connections
                    total_active += proxy.active_connections

                    # Performance metrics
                    if proxy.response_time:
                        response_times.append(proxy.response_time)

                    success_rates.append(proxy.success_rate)

                    proxy_performance.append({
                        'host': proxy.host,
                        'port': proxy.port,
                        'region': proxy.region.value,
                        'response_time': proxy.response_time or 0,
                        'success_rate': proxy.success_rate,
                        'utilization': proxy.utilization
                    })

                # Calculate utilization
                stats['utilization']['total_capacity'] = total_capacity
                stats['utilization']['active_connections'] = total_active
                if total_capacity > 0:
                    stats['utilization']['utilization_percentage'] = (total_active / total_capacity) * 100

                # Calculate performance metrics
                if response_times:
                    stats['performance']['average_response_time'] = statistics.mean(response_times)

                if success_rates:
                    stats['performance']['success_rate'] = statistics.mean(success_rates)

                # Top and worst performers
                proxy_performance.sort(key=lambda x: (x['success_rate'], -x['response_time']), reverse=True)
                stats['performance']['top_performers'] = proxy_performance[:10]
                stats['performance']['worst_performers'] = proxy_performance[-10:]

            # Add global metrics
            stats['global_metrics'] = self.metrics.copy()

            return stats

        except Exception as e:
            logger.error(f"Error getting proxy statistics: {e}")
            return {}

    async def _collect_metrics(self):
        """Background task to collect and update metrics"""

        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute

                async with self.cache_lock:
                    # Update global metrics
                    self.metrics['total_proxies'] = len(self.proxy_cache)
                    self.metrics['healthy_proxies'] = len([
                        p for p in self.proxy_cache.values()
                        if p.status == ProxyStatus.HEALTHY
                    ])
                    self.metrics['active_connections'] = sum(
                        p.active_connections for p in self.proxy_cache.values()
                    )

                # Store metrics in Redis
                await self.redis_pool.hset(
                    "proxy_manager_metrics",
                    mapping={k: str(v) for k, v in self.metrics.items()}
                )

                # Update daily usage stats in PostgreSQL
                await self._update_daily_usage_stats()

            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")

    async def _update_daily_usage_stats(self):
        """Update daily usage statistics"""

        try:
            async with self.postgres_pool.acquire() as conn:
                for proxy in self.proxy_cache.values():
                    await conn.execute("""
                        INSERT INTO proxy_usage_stats
                        (proxy_id, total_requests, successful_requests, failed_requests,
                         average_response_time, max_concurrent_connections)
                        SELECT
                            (SELECT id FROM proxy_endpoints WHERE host = $1 AND port = $2),
                            $3, $4, $5, $6, $7
                        ON CONFLICT (proxy_id, date) DO UPDATE SET
                            total_requests = EXCLUDED.total_requests,
                            successful_requests = EXCLUDED.successful_requests,
                            failed_requests = EXCLUDED.failed_requests,
                            average_response_time = EXCLUDED.average_response_time,
                            max_concurrent_connections = GREATEST(
                                proxy_usage_stats.max_concurrent_connections,
                                EXCLUDED.max_concurrent_connections
                            )
                    """,
                        proxy.host, proxy.port, proxy.total_requests,
                        int(proxy.total_requests * proxy.success_rate),
                        proxy.total_requests - int(proxy.total_requests * proxy.success_rate),
                        proxy.response_time or 0.0, proxy.active_connections
                    )

        except Exception as e:
            logger.error(f"Error updating daily usage stats: {e}")

    async def _refresh_cache(self):
        """Background task to refresh proxy cache from database"""

        while True:
            try:
                await asyncio.sleep(300)  # Refresh every 5 minutes
                await self._load_proxy_configurations()
                logger.debug("Proxy cache refreshed")

            except Exception as e:
                logger.error(f"Error refreshing cache: {e}")

# Factory function for easy initialization
async def create_proxy_manager(
    redis_url: str = "redis://localhost:6379",
    postgres_url: str = "postgresql://localhost:5432/osint",
    max_pool_size: int = 10000
) -> ProxyManager:
    """Create and initialize a ProxyManager instance"""

    manager = ProxyManager(redis_url, postgres_url, max_pool_size)
    await manager.initialize()
    return manager

if __name__ == "__main__":
    # Example usage and testing
    async def main():
        # Create proxy manager
        manager = await create_proxy_manager()

        try:
            # Add some test proxies
            test_proxies = [
                ProxyEndpoint(
                    host="proxy1.example.com", port=8080,
                    proxy_type=ProxyType.RESIDENTIAL,
                    region=ProxyRegion.US_EAST,
                    provider="Example Provider 1"
                ),
                ProxyEndpoint(
                    host="proxy2.example.com", port=8080,
                    proxy_type=ProxyType.DATACENTER,
                    region=ProxyRegion.EU_CENTRAL,
                    provider="Example Provider 2"
                )
            ]

            for proxy in test_proxies:
                await manager.add_proxy(proxy)

            # Test proxy selection
            for i in range(5):
                proxy = await manager.get_proxy(region_preference=ProxyRegion.US_EAST)
                if proxy:
                    print(f"Selected proxy: {proxy.host}:{proxy.port}")

                    # Simulate usage
                    await asyncio.sleep(1)
                    await manager.release_proxy(proxy, success=True, response_time=0.5)
                else:
                    print("No proxy available")

            # Get statistics
            stats = await manager.get_proxy_statistics()
            print(f"Proxy statistics: {json.dumps(stats, indent=2, default=str)}")

        finally:
            await manager.shutdown()

    # Run example
    asyncio.run(main())