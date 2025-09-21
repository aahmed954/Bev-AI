#!/usr/bin/env python3
"""
BEV OSINT Framework - Connection Pool Manager
Intelligent connection pooling with resource management, proxy rotation,
and performance optimization for high-concurrency request processing.
"""

import asyncio
import aiohttp
import time
import ssl
import weakref
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from urllib.parse import urlparse
from collections import defaultdict, deque
import statistics
import certifi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state enumeration"""
    IDLE = "idle"
    ACTIVE = "active"
    STALE = "stale"
    ERROR = "error"
    CLOSING = "closing"


@dataclass
class ConnectionInfo:
    """Connection information tracking"""
    connection_id: str
    endpoint: str
    proxy: Optional[str] = None
    state: ConnectionState = ConnectionState.IDLE
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    requests_served: int = 0
    errors_count: int = 0
    latency_stats: List[float] = field(default_factory=list)
    user_agent: Optional[str] = None

    @property
    def age(self) -> float:
        """Connection age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()

    @property
    def idle_time(self) -> float:
        """Idle time in seconds"""
        return (datetime.now() - self.last_used).total_seconds()

    @property
    def average_latency(self) -> float:
        """Average latency for this connection"""
        return statistics.mean(self.latency_stats) if self.latency_stats else 0.0

    @property
    def error_rate(self) -> float:
        """Error rate for this connection"""
        total = self.requests_served + self.errors_count
        return self.errors_count / total if total > 0 else 0.0


@dataclass
class PoolStats:
    """Connection pool statistics"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    stale_connections: int = 0
    error_connections: int = 0
    total_requests: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    connections_created: int = 0
    connections_closed: int = 0
    proxy_rotations: int = 0

    @property
    def hit_rate(self) -> float:
        """Connection pool hit rate"""
        total = self.pool_hits + self.pool_misses
        return self.pool_hits / total if total > 0 else 0.0


class ProxyRotator:
    """Intelligent proxy rotation system"""

    def __init__(self, proxy_pool_config: Dict[str, Any]):
        self.config = proxy_pool_config
        self.proxy_pool_url = proxy_pool_config.get('proxy_pool_url', 'http://172.30.0.40:9090')
        self.rotation_strategy = proxy_pool_config.get('strategy', 'round_robin')
        self.health_check_interval = proxy_pool_config.get('health_check_interval', 60)

        # Proxy management
        self.available_proxies: List[Dict] = []
        self.proxy_stats: Dict[str, Dict] = defaultdict(lambda: {
            'requests': 0,
            'errors': 0,
            'avg_latency': 0.0,
            'last_used': None,
            'blacklisted': False
        })
        self.current_index = 0

        # Health monitoring
        self.last_health_check = datetime.now()

    async def initialize(self):
        """Initialize proxy rotator"""
        await self.refresh_proxy_pool()
        logger.info(f"ProxyRotator initialized with {len(self.available_proxies)} proxies")

    async def refresh_proxy_pool(self):
        """Refresh proxy pool from proxy pool manager"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.proxy_pool_url}/api/proxies") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.available_proxies = data.get('proxies', [])
                        logger.debug(f"Refreshed proxy pool: {len(self.available_proxies)} proxies")
        except Exception as e:
            logger.warning(f"Failed to refresh proxy pool: {e}")

    async def get_proxy(self, endpoint: str = None) -> Optional[str]:
        """Get next proxy based on rotation strategy"""
        # Check if we need to refresh proxy pool
        if (datetime.now() - self.last_health_check).seconds > self.health_check_interval:
            await self.refresh_proxy_pool()
            self.last_health_check = datetime.now()

        if not self.available_proxies:
            return None

        if self.rotation_strategy == 'round_robin':
            return self._round_robin_proxy()
        elif self.rotation_strategy == 'least_used':
            return self._least_used_proxy()
        elif self.rotation_strategy == 'best_performance':
            return self._best_performance_proxy()
        elif self.rotation_strategy == 'geographic':
            return self._geographic_proxy(endpoint)
        else:
            return self._round_robin_proxy()

    def _round_robin_proxy(self) -> str:
        """Round-robin proxy selection"""
        if not self.available_proxies:
            return None

        proxy = self.available_proxies[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.available_proxies)
        return proxy.get('url')

    def _least_used_proxy(self) -> str:
        """Select least used proxy"""
        if not self.available_proxies:
            return None

        # Sort by request count
        sorted_proxies = sorted(
            self.available_proxies,
            key=lambda p: self.proxy_stats[p.get('url', '')]['requests']
        )
        return sorted_proxies[0].get('url')

    def _best_performance_proxy(self) -> str:
        """Select best performing proxy"""
        if not self.available_proxies:
            return None

        # Score based on error rate and latency
        best_proxy = None
        best_score = float('inf')

        for proxy in self.available_proxies:
            proxy_url = proxy.get('url', '')
            stats = self.proxy_stats[proxy_url]

            if stats['blacklisted']:
                continue

            # Calculate score (lower is better)
            error_rate = stats['errors'] / max(stats['requests'], 1)
            latency = stats['avg_latency']
            score = error_rate * 1000 + latency  # Weight error rate heavily

            if score < best_score:
                best_score = score
                best_proxy = proxy

        return best_proxy.get('url') if best_proxy else self.available_proxies[0].get('url')

    def _geographic_proxy(self, endpoint: str) -> str:
        """Select proxy based on geographic proximity to endpoint"""
        if not self.available_proxies or not endpoint:
            return self._round_robin_proxy()

        # Extract country from endpoint (simplified)
        parsed = urlparse(endpoint)
        domain = parsed.netloc.lower()

        # Country-specific proxy selection
        country_preferences = {
            '.uk': 'GB',
            '.de': 'DE',
            '.fr': 'FR',
            '.jp': 'JP',
            '.au': 'AU',
            '.ca': 'CA'
        }

        preferred_country = 'US'  # Default
        for suffix, country in country_preferences.items():
            if suffix in domain:
                preferred_country = country
                break

        # Find proxy in preferred country
        for proxy in self.available_proxies:
            if proxy.get('country') == preferred_country:
                return proxy.get('url')

        # Fallback to round-robin
        return self._round_robin_proxy()

    def record_request(self, proxy_url: str, latency: float, error: bool = False):
        """Record proxy usage statistics"""
        stats = self.proxy_stats[proxy_url]
        stats['requests'] += 1
        stats['last_used'] = datetime.now()

        if error:
            stats['errors'] += 1
        else:
            # Update average latency
            current_avg = stats['avg_latency']
            request_count = stats['requests'] - stats['errors']
            if request_count > 0:
                stats['avg_latency'] = ((current_avg * (request_count - 1)) + latency) / request_count

        # Blacklist proxy if error rate is too high
        error_rate = stats['errors'] / stats['requests']
        if stats['requests'] > 10 and error_rate > 0.5:
            stats['blacklisted'] = True
            logger.warning(f"Blacklisted proxy {proxy_url} due to high error rate: {error_rate:.2%}")


class ConnectionPoolManager:
    """
    Intelligent connection pool manager with proxy rotation,
    adaptive sizing, and performance optimization.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Pool configuration
        self.max_connections = config.get('max_connections', 500)
        self.max_connections_per_host = config.get('max_connections_per_host', 50)
        self.connection_timeout = config.get('connection_timeout', 30.0)
        self.read_timeout = config.get('read_timeout', 30.0)
        self.idle_timeout = config.get('idle_timeout', 300)  # 5 minutes
        self.max_connection_age = config.get('max_connection_age', 3600)  # 1 hour
        self.enable_proxy_rotation = config.get('enable_proxy_rotation', True)

        # SSL configuration
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE

        # Connection pools by endpoint
        self.connection_pools: Dict[str, aiohttp.ClientSession] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        self.host_connection_counts: Dict[str, int] = defaultdict(int)

        # Proxy rotation
        self.proxy_rotator = None
        if self.enable_proxy_rotation:
            self.proxy_rotator = ProxyRotator(config.get('proxy_rotation', {}))

        # Performance tracking
        self.stats = PoolStats()
        self.latency_window = deque(maxlen=1000)

        # Cleanup task
        self.cleanup_task = None
        self._shutdown = False

        logger.info(f"ConnectionPoolManager initialized: {self.max_connections} max connections")

    async def initialize(self):
        """Initialize the connection pool manager"""
        if self.proxy_rotator:
            await self.proxy_rotator.initialize()

        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_stale_connections())

        logger.info("ConnectionPoolManager initialized")

    async def close(self):
        """Close all connections and cleanup"""
        self._shutdown = True

        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()

        # Close all sessions
        close_tasks = []
        for session in self.connection_pools.values():
            close_tasks.append(session.close())

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        self.connection_pools.clear()
        self.connection_info.clear()

        logger.info("ConnectionPoolManager closed")

    async def get_session(self, endpoint: str) -> aiohttp.ClientSession:
        """Get or create a session for the endpoint"""
        parsed = urlparse(endpoint)
        host_key = f"{parsed.scheme}://{parsed.netloc}"

        # Check if we have an existing session
        session = self.connection_pools.get(host_key)

        if session and not session.closed:
            # Check connection limits
            if self.host_connection_counts[host_key] < self.max_connections_per_host:
                self.stats.pool_hits += 1
                self._update_connection_usage(host_key)
                return session
            else:
                # Too many connections for this host, wait or create new
                logger.debug(f"Connection limit reached for {host_key}")

        # Need to create new session
        self.stats.pool_misses += 1
        return await self._create_session(host_key, endpoint)

    async def _create_session(self, host_key: str, endpoint: str) -> aiohttp.ClientSession:
        """Create a new session for the host"""
        # Check global connection limit
        if self.stats.total_connections >= self.max_connections:
            # Clean up stale connections first
            await self._cleanup_stale_connections()

            # If still at limit, wait for a connection to become available
            if self.stats.total_connections >= self.max_connections:
                logger.warning("Connection pool exhausted, waiting for available connection")
                await asyncio.sleep(0.1)
                return await self.get_session(endpoint)

        # Configure session
        connector_kwargs = {
            'limit': self.max_connections_per_host,
            'ttl_dns_cache': 300,
            'use_dns_cache': True,
            'ssl': self.ssl_context,
            'enable_cleanup_closed': True
        }

        # Add proxy if available
        proxy_url = None
        if self.proxy_rotator:
            proxy_url = await self.proxy_rotator.get_proxy(endpoint)
            if proxy_url:
                connector_kwargs['proxy'] = proxy_url

        # Create connector and session
        connector = aiohttp.TCPConnector(**connector_kwargs)

        timeout = aiohttp.ClientTimeout(
            total=self.connection_timeout,
            connect=self.connection_timeout,
            sock_read=self.read_timeout
        )

        session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': self._get_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )

        # Track the proxy used
        if proxy_url:
            session._proxy_used = proxy_url

        # Store session and info
        self.connection_pools[host_key] = session
        self.connection_info[host_key] = ConnectionInfo(
            connection_id=f"{host_key}_{int(time.time())}",
            endpoint=endpoint,
            proxy=proxy_url,
            state=ConnectionState.ACTIVE
        )

        # Update statistics
        self.stats.total_connections += 1
        self.stats.connections_created += 1
        self.host_connection_counts[host_key] += 1

        if proxy_url:
            self.stats.proxy_rotations += 1

        logger.debug(f"Created new session for {host_key} with proxy: {proxy_url}")
        return session

    def _update_connection_usage(self, host_key: str):
        """Update connection usage statistics"""
        conn_info = self.connection_info.get(host_key)
        if conn_info:
            conn_info.last_used = datetime.now()
            conn_info.requests_served += 1
            conn_info.state = ConnectionState.ACTIVE

    def record_request_latency(self, endpoint: str, latency: float, error: bool = False):
        """Record request latency and update connection stats"""
        parsed = urlparse(endpoint)
        host_key = f"{parsed.scheme}://{parsed.netloc}"

        conn_info = self.connection_info.get(host_key)
        if conn_info:
            conn_info.latency_stats.append(latency)
            # Keep only last 100 latencies per connection
            if len(conn_info.latency_stats) > 100:
                conn_info.latency_stats = conn_info.latency_stats[-100:]

            if error:
                conn_info.errors_count += 1

        # Update proxy statistics
        if self.proxy_rotator and conn_info and conn_info.proxy:
            self.proxy_rotator.record_request(conn_info.proxy, latency, error)

        self.latency_window.append(latency)
        self.stats.total_requests += 1

    async def _cleanup_stale_connections(self):
        """Clean up stale and old connections"""
        while not self._shutdown:
            current_time = datetime.now()
            stale_connections = []

            for host_key, conn_info in self.connection_info.items():
                # Check if connection is stale
                if (conn_info.idle_time > self.idle_timeout or
                    conn_info.age > self.max_connection_age or
                    conn_info.error_rate > 0.5):
                    stale_connections.append(host_key)

            # Close stale connections
            for host_key in stale_connections:
                await self._close_connection(host_key)

            if stale_connections:
                logger.debug(f"Cleaned up {len(stale_connections)} stale connections")

            # Update connection states
            for conn_info in self.connection_info.values():
                if conn_info.idle_time > 60:  # 1 minute idle
                    conn_info.state = ConnectionState.IDLE
                elif conn_info.error_rate > 0.3:
                    conn_info.state = ConnectionState.ERROR

            # Sleep before next cleanup
            await asyncio.sleep(30)  # Check every 30 seconds

    async def _close_connection(self, host_key: str):
        """Close a specific connection"""
        session = self.connection_pools.get(host_key)
        if session:
            try:
                await session.close()
            except Exception as e:
                logger.debug(f"Error closing session for {host_key}: {e}")

            del self.connection_pools[host_key]
            self.stats.connections_closed += 1

        if host_key in self.connection_info:
            del self.connection_info[host_key]

        self.host_connection_counts[host_key] = max(0, self.host_connection_counts[host_key] - 1)
        self.stats.total_connections = max(0, self.stats.total_connections - 1)

    def _get_user_agent(self) -> str:
        """Get a realistic user agent string"""
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0'
        ]
        return user_agents[int(time.time()) % len(user_agents)]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive connection pool statistics"""
        # Calculate connection state distribution
        state_counts = defaultdict(int)
        for conn_info in self.connection_info.values():
            state_counts[conn_info.state.value] += 1

        # Calculate average latencies by host
        host_latencies = {}
        for host_key, conn_info in self.connection_info.items():
            if conn_info.latency_stats:
                host_latencies[host_key] = {
                    'average': statistics.mean(conn_info.latency_stats),
                    'median': statistics.median(conn_info.latency_stats),
                    'p95': statistics.quantiles(conn_info.latency_stats, n=20)[18] if len(conn_info.latency_stats) > 20 else 0
                }

        return {
            'pool_stats': {
                'total_connections': self.stats.total_connections,
                'active_connections': state_counts['active'],
                'idle_connections': state_counts['idle'],
                'stale_connections': state_counts['stale'],
                'error_connections': state_counts['error'],
                'hit_rate': self.stats.hit_rate,
                'connections_created': self.stats.connections_created,
                'connections_closed': self.stats.connections_closed
            },
            'performance': {
                'total_requests': self.stats.total_requests,
                'average_latency': statistics.mean(self.latency_window) if self.latency_window else 0,
                'proxy_rotations': self.stats.proxy_rotations
            },
            'host_distribution': dict(self.host_connection_counts),
            'host_latencies': host_latencies,
            'proxy_stats': dict(self.proxy_rotator.proxy_stats) if self.proxy_rotator else {}
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on connection pool"""
        healthy_connections = 0
        unhealthy_connections = 0

        for conn_info in self.connection_info.values():
            if conn_info.state in [ConnectionState.ACTIVE, ConnectionState.IDLE]:
                healthy_connections += 1
            else:
                unhealthy_connections += 1

        return {
            'status': 'healthy' if healthy_connections > unhealthy_connections else 'degraded',
            'healthy_connections': healthy_connections,
            'unhealthy_connections': unhealthy_connections,
            'total_connections': len(self.connection_info),
            'utilization': len(self.connection_info) / self.max_connections,
            'proxy_pool_available': len(self.proxy_rotator.available_proxies) if self.proxy_rotator else 0
        }


# Factory function
def create_connection_pool_manager(config: Dict[str, Any]) -> ConnectionPoolManager:
    """Create and configure a ConnectionPoolManager instance"""
    return ConnectionPoolManager(config)


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'max_connections': 500,
            'max_connections_per_host': 50,
            'connection_timeout': 30.0,
            'read_timeout': 30.0,
            'idle_timeout': 300,
            'max_connection_age': 3600,
            'enable_proxy_rotation': True,
            'proxy_rotation': {
                'proxy_pool_url': 'http://172.30.0.40:9090',
                'strategy': 'best_performance',
                'health_check_interval': 60
            }
        }

        pool_manager = create_connection_pool_manager(config)
        await pool_manager.initialize()

        try:
            # Test getting sessions
            session1 = await pool_manager.get_session('https://httpbin.org/get')
            session2 = await pool_manager.get_session('https://jsonplaceholder.typicode.com/posts')

            # Simulate some requests
            start_time = time.time()
            async with session1.get('https://httpbin.org/get') as response:
                await response.read()
            latency = time.time() - start_time

            pool_manager.record_request_latency('https://httpbin.org/get', latency)

            # Get statistics
            stats = pool_manager.get_statistics()
            print(f"Pool Statistics: {stats}")

            # Health check
            health = await pool_manager.health_check()
            print(f"Health Check: {health}")

        finally:
            await pool_manager.close()

    asyncio.run(main())