#!/usr/bin/env python3
"""
BEV OSINT Framework - Request Multiplexer
High-performance request multiplexing system supporting 1000+ concurrent requests
with intelligent connection pooling, rate limiting, and queue management.
"""

import asyncio
import aiohttp
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
from urllib.parse import urlparse
import weakref
from collections import defaultdict, deque
import statistics

from .connection_pool import ConnectionPoolManager
from .rate_limiter import RateLimitEngine
from .queue_manager import QueueManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """Request priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class RequestStatus(Enum):
    """Request execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


@dataclass
class Request:
    """Individual request container"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    data: Any = None
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    priority: RequestPriority = RequestPriority.MEDIUM
    retries: int = 3
    retry_delay: float = 1.0
    callback: Optional[Callable] = None
    context: Dict[str, Any] = field(default_factory=dict)

    # Runtime attributes
    status: RequestStatus = RequestStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempts: int = 0
    last_error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    proxy_used: Optional[str] = None
    latency: Optional[float] = None


@dataclass
class Response:
    """Response container"""
    request_id: str
    status_code: int
    headers: Dict[str, str]
    content: bytes
    text: str
    elapsed: float
    proxy_used: Optional[str] = None
    cached: bool = False
    error: Optional[str] = None


@dataclass
class MultiplexerStats:
    """Multiplexer performance statistics"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    retried_requests: int = 0
    rate_limited_requests: int = 0
    cached_responses: int = 0
    total_latency: float = 0.0
    peak_concurrency: int = 0
    current_concurrency: int = 0
    uptime: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.completed_requests / self.total_requests

    @property
    def average_latency(self) -> float:
        if self.completed_requests == 0:
            return 0.0
        return self.total_latency / self.completed_requests

    @property
    def requests_per_second(self) -> float:
        runtime = (datetime.now() - self.start_time).total_seconds()
        if runtime == 0:
            return 0.0
        return self.total_requests / runtime


class RequestMultiplexer:
    """
    High-performance request multiplexer supporting 1000+ concurrent requests
    with intelligent resource management and advanced features.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_concurrency = config.get('max_concurrency', 1000)
        self.request_timeout = config.get('request_timeout', 30.0)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.enable_caching = config.get('enable_caching', True)
        self.cache_ttl = config.get('cache_ttl', 300)  # 5 minutes

        # Core components
        self.connection_pool = ConnectionPoolManager(config.get('connection_pool', {}))
        self.rate_limiter = RateLimitEngine(config.get('rate_limiting', {}))
        self.queue_manager = QueueManager(config.get('queue_manager', {}))

        # Request management
        self.active_requests: Dict[str, Request] = {}
        self.request_history: deque = deque(maxlen=10000)
        self.response_cache: Dict[str, Tuple[Response, datetime]] = {}

        # Concurrency control
        self.semaphore = asyncio.Semaphore(self.max_concurrency)
        self.request_workers = []
        self.worker_count = config.get('worker_count', 50)

        # Performance tracking
        self.stats = MultiplexerStats()
        self.latency_window = deque(maxlen=1000)
        self.throughput_window = deque(maxlen=60)  # Last 60 seconds

        # Circuit breaker for endpoints
        self.circuit_breakers: Dict[str, Dict] = defaultdict(lambda: {
            'failures': 0,
            'last_failure': None,
            'state': 'closed',  # closed, open, half-open
            'next_attempt': None
        })

        # Event callbacks
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Shutdown flag
        self._shutdown = False

        logger.info(f"RequestMultiplexer initialized with max_concurrency={self.max_concurrency}")

    async def start(self):
        """Start the multiplexer and its components"""
        logger.info("Starting RequestMultiplexer...")

        # Initialize components
        await self.connection_pool.initialize()
        await self.rate_limiter.initialize()
        await self.queue_manager.initialize()

        # Start request workers
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._request_worker(i))
            self.request_workers.append(worker)

        # Start performance monitoring
        asyncio.create_task(self._performance_monitor())

        # Start cache cleanup
        if self.enable_caching:
            asyncio.create_task(self._cache_cleanup())

        logger.info(f"RequestMultiplexer started with {self.worker_count} workers")

    async def stop(self):
        """Stop the multiplexer gracefully"""
        logger.info("Stopping RequestMultiplexer...")

        self._shutdown = True

        # Cancel workers
        for worker in self.request_workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.request_workers, return_exceptions=True)

        # Stop components
        await self.connection_pool.close()
        await self.rate_limiter.close()
        await self.queue_manager.close()

        logger.info("RequestMultiplexer stopped")

    async def submit_request(self, request: Request) -> str:
        """Submit a request for processing"""
        # Generate unique ID if not provided
        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        # Validate request
        if not request.url:
            raise ValueError("Request URL is required")

        # Check rate limiting
        endpoint = self._get_endpoint_key(request.url)
        if not await self.rate_limiter.allow_request(endpoint, request.priority):
            request.status = RequestStatus.RATE_LIMITED
            self.stats.rate_limited_requests += 1
            await self._emit_event('request_rate_limited', request)
            raise Exception(f"Rate limit exceeded for {endpoint}")

        # Check circuit breaker
        if self._is_circuit_open(endpoint):
            request.status = RequestStatus.FAILED
            request.last_error = f"Circuit breaker open for {endpoint}"
            self.stats.failed_requests += 1
            await self._emit_event('request_circuit_open', request)
            raise Exception(request.last_error)

        # Add to active requests
        self.active_requests[request.request_id] = request
        self.stats.total_requests += 1

        # Queue the request
        await self.queue_manager.enqueue(request)
        request.status = RequestStatus.QUEUED

        await self._emit_event('request_submitted', request)

        logger.debug(f"Request {request.request_id} submitted for {request.url}")
        return request.request_id

    async def get_request_status(self, request_id: str) -> Optional[Request]:
        """Get status of a specific request"""
        return self.active_requests.get(request_id)

    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending or processing request"""
        request = self.active_requests.get(request_id)
        if not request:
            return False

        if request.status in [RequestStatus.PENDING, RequestStatus.QUEUED]:
            request.status = RequestStatus.FAILED
            request.last_error = "Cancelled by user"
            await self._emit_event('request_cancelled', request)
            return True

        return False

    async def _request_worker(self, worker_id: int):
        """Worker coroutine to process requests"""
        logger.debug(f"Request worker {worker_id} started")

        while not self._shutdown:
            try:
                # Get next request from queue
                request = await self.queue_manager.dequeue(timeout=1.0)
                if not request:
                    continue

                # Process the request
                await self._process_request(request)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)

        logger.debug(f"Request worker {worker_id} stopped")

    async def _process_request(self, request: Request):
        """Process a single request"""
        request.status = RequestStatus.PROCESSING
        request.started_at = datetime.now()
        self.stats.current_concurrency += 1

        # Update peak concurrency
        if self.stats.current_concurrency > self.stats.peak_concurrency:
            self.stats.peak_concurrency = self.stats.current_concurrency

        try:
            async with self.semaphore:
                await self._execute_request(request)
        finally:
            self.stats.current_concurrency -= 1
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            self.request_history.append(request)

    async def _execute_request(self, request: Request):
        """Execute the actual HTTP request"""
        endpoint = self._get_endpoint_key(request.url)

        # Check cache first
        if self.enable_caching and request.method == 'GET':
            cached_response = self._get_cached_response(request)
            if cached_response:
                request.result = cached_response.__dict__
                request.status = RequestStatus.COMPLETED
                request.completed_at = datetime.now()
                self.stats.cached_responses += 1
                await self._emit_event('request_completed', request)
                return

        # Execute with retries
        for attempt in range(request.retries + 1):
            request.attempts = attempt + 1

            try:
                start_time = time.time()

                # Get connection from pool
                session = await self.connection_pool.get_session(endpoint)

                # Execute request
                async with session.request(
                    method=request.method,
                    url=request.url,
                    headers=request.headers,
                    data=request.data,
                    timeout=aiohttp.ClientTimeout(total=request.timeout),
                    **request.params
                ) as response:
                    # Read response
                    content = await response.read()
                    text = await response.text()
                    elapsed = time.time() - start_time

                    # Create response object
                    resp = Response(
                        request_id=request.request_id,
                        status_code=response.status,
                        headers=dict(response.headers),
                        content=content,
                        text=text,
                        elapsed=elapsed,
                        proxy_used=getattr(session, '_proxy_used', None)
                    )

                    # Update request
                    request.result = resp.__dict__
                    request.latency = elapsed
                    request.status = RequestStatus.COMPLETED
                    request.completed_at = datetime.now()

                    # Update statistics
                    self.stats.completed_requests += 1
                    self.stats.total_latency += elapsed
                    self.latency_window.append(elapsed)

                    # Cache response if applicable
                    if self.enable_caching and request.method == 'GET' and response.status == 200:
                        self._cache_response(request, resp)

                    # Reset circuit breaker on success
                    self._reset_circuit_breaker(endpoint)

                    await self._emit_event('request_completed', request)
                    return

            except asyncio.TimeoutError:
                request.last_error = f"Timeout after {request.timeout}s"
                if attempt == request.retries:
                    request.status = RequestStatus.TIMEOUT
                    self.stats.failed_requests += 1
                    self._record_circuit_failure(endpoint)
                    await self._emit_event('request_timeout', request)
                    return

            except Exception as e:
                request.last_error = str(e)
                if attempt == request.retries:
                    request.status = RequestStatus.FAILED
                    self.stats.failed_requests += 1
                    self._record_circuit_failure(endpoint)
                    await self._emit_event('request_failed', request)
                    return

                # Retry with exponential backoff
                if attempt < request.retries:
                    request.status = RequestStatus.RETRYING
                    self.stats.retried_requests += 1
                    retry_delay = request.retry_delay * (2 ** attempt)
                    await asyncio.sleep(retry_delay)
                    await self._emit_event('request_retry', request)

    def _get_endpoint_key(self, url: str) -> str:
        """Extract endpoint key for rate limiting and circuit breaking"""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _get_cached_response(self, request: Request) -> Optional[Response]:
        """Get cached response if available and valid"""
        cache_key = self._get_cache_key(request)
        cached = self.response_cache.get(cache_key)

        if cached:
            response, timestamp = cached
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                response.cached = True
                return response
            else:
                # Remove expired cache entry
                del self.response_cache[cache_key]

        return None

    def _cache_response(self, request: Request, response: Response):
        """Cache a response"""
        cache_key = self._get_cache_key(request)
        self.response_cache[cache_key] = (response, datetime.now())

    def _get_cache_key(self, request: Request) -> str:
        """Generate cache key for request"""
        # Simple cache key based on URL and headers
        key_data = f"{request.method}:{request.url}"
        if request.headers:
            sorted_headers = sorted(request.headers.items())
            key_data += ":" + str(sorted_headers)
        return key_data

    def _is_circuit_open(self, endpoint: str) -> bool:
        """Check if circuit breaker is open for endpoint"""
        breaker = self.circuit_breakers[endpoint]

        if breaker['state'] == 'open':
            if breaker['next_attempt'] and datetime.now() > breaker['next_attempt']:
                # Move to half-open state
                breaker['state'] = 'half-open'
                return False
            return True

        return False

    def _record_circuit_failure(self, endpoint: str):
        """Record a failure for circuit breaker"""
        breaker = self.circuit_breakers[endpoint]
        breaker['failures'] += 1
        breaker['last_failure'] = datetime.now()

        # Open circuit if too many failures
        failure_threshold = self.config.get('circuit_breaker_threshold', 5)
        if breaker['failures'] >= failure_threshold:
            breaker['state'] = 'open'
            # Set next attempt time (60 seconds later)
            breaker['next_attempt'] = datetime.now() + timedelta(seconds=60)

    def _reset_circuit_breaker(self, endpoint: str):
        """Reset circuit breaker on successful request"""
        breaker = self.circuit_breakers[endpoint]
        breaker['failures'] = 0
        breaker['state'] = 'closed'
        breaker['next_attempt'] = None

    async def _cache_cleanup(self):
        """Periodically clean up expired cache entries"""
        while not self._shutdown:
            await asyncio.sleep(60)  # Check every minute

            current_time = datetime.now()
            expired_keys = []

            for key, (response, timestamp) in self.response_cache.items():
                if (current_time - timestamp).total_seconds() > self.cache_ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.response_cache[key]

            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    async def _performance_monitor(self):
        """Monitor and log performance metrics"""
        while not self._shutdown:
            await asyncio.sleep(10)  # Update every 10 seconds

            # Update uptime
            self.stats.uptime = (datetime.now() - self.stats.start_time).total_seconds()

            # Log performance summary
            logger.info(
                f"Performance: {self.stats.completed_requests}/{self.stats.total_requests} requests, "
                f"{self.stats.success_rate:.2%} success rate, "
                f"{self.stats.average_latency:.3f}s avg latency, "
                f"{self.stats.current_concurrency} active"
            )

    async def _emit_event(self, event_type: str, request: Request):
        """Emit event to registered handlers"""
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(request)
                else:
                    handler(request)
            except Exception as e:
                logger.error(f"Event handler error for {event_type}: {e}")

    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler for specific event type"""
        self.event_handlers[event_type].append(handler)

    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove event handler"""
        if handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            'requests': {
                'total': self.stats.total_requests,
                'completed': self.stats.completed_requests,
                'failed': self.stats.failed_requests,
                'retried': self.stats.retried_requests,
                'rate_limited': self.stats.rate_limited_requests,
                'cached': self.stats.cached_responses,
                'success_rate': self.stats.success_rate,
                'requests_per_second': self.stats.requests_per_second
            },
            'performance': {
                'average_latency': self.stats.average_latency,
                'current_concurrency': self.stats.current_concurrency,
                'peak_concurrency': self.stats.peak_concurrency,
                'uptime': self.stats.uptime
            },
            'components': {
                'connection_pool': self.connection_pool.get_statistics(),
                'rate_limiter': self.rate_limiter.get_statistics(),
                'queue_manager': self.queue_manager.get_statistics()
            },
            'cache': {
                'entries': len(self.response_cache),
                'hit_rate': self.stats.cached_responses / max(1, self.stats.completed_requests)
            },
            'circuit_breakers': {
                endpoint: {
                    'state': data['state'],
                    'failures': data['failures'],
                    'last_failure': data['last_failure'].isoformat() if data['last_failure'] else None
                }
                for endpoint, data in self.circuit_breakers.items()
            }
        }

    # Convenience methods for common operations

    async def get(self, url: str, **kwargs) -> str:
        """Convenience method for GET requests"""
        request = Request(url=url, method='GET', **kwargs)
        return await self.submit_request(request)

    async def post(self, url: str, data: Any = None, **kwargs) -> str:
        """Convenience method for POST requests"""
        request = Request(url=url, method='POST', data=data, **kwargs)
        return await self.submit_request(request)

    async def put(self, url: str, data: Any = None, **kwargs) -> str:
        """Convenience method for PUT requests"""
        request = Request(url=url, method='PUT', data=data, **kwargs)
        return await self.submit_request(request)

    async def delete(self, url: str, **kwargs) -> str:
        """Convenience method for DELETE requests"""
        request = Request(url=url, method='DELETE', **kwargs)
        return await self.submit_request(request)

    async def bulk_submit(self, requests: List[Request]) -> List[str]:
        """Submit multiple requests in bulk"""
        request_ids = []
        for request in requests:
            request_id = await self.submit_request(request)
            request_ids.append(request_id)
        return request_ids

    async def wait_for_completion(self, request_ids: List[str], timeout: float = 300.0) -> List[Request]:
        """Wait for multiple requests to complete"""
        start_time = time.time()
        completed_requests = []

        while request_ids and (time.time() - start_time) < timeout:
            for request_id in list(request_ids):
                request = await self.get_request_status(request_id)
                if request and request.status in [RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.TIMEOUT]:
                    completed_requests.append(request)
                    request_ids.remove(request_id)

            if request_ids:
                await asyncio.sleep(0.1)

        return completed_requests


# Factory function for easy instantiation
def create_multiplexer(config: Dict[str, Any]) -> RequestMultiplexer:
    """Create and configure a RequestMultiplexer instance"""
    return RequestMultiplexer(config)


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'max_concurrency': 1000,
            'worker_count': 50,
            'request_timeout': 30.0,
            'retry_attempts': 3,
            'enable_caching': True,
            'cache_ttl': 300,
            'connection_pool': {
                'max_connections': 500,
                'max_connections_per_host': 50
            },
            'rate_limiting': {
                'default_limit': 100,
                'window_size': 60
            },
            'queue_manager': {
                'max_queue_size': 10000,
                'priority_enabled': True
            }
        }

        multiplexer = create_multiplexer(config)
        await multiplexer.start()

        try:
            # Example requests
            request_id1 = await multiplexer.get('https://httpbin.org/get')
            request_id2 = await multiplexer.post('https://httpbin.org/post', data={'test': 'data'})

            # Wait for completion
            completed = await multiplexer.wait_for_completion([request_id1, request_id2])

            for request in completed:
                print(f"Request {request.request_id}: {request.status}")
                if request.result:
                    print(f"  Status Code: {request.result.get('status_code')}")
                    print(f"  Latency: {request.latency:.3f}s")

            # Print statistics
            stats = multiplexer.get_statistics()
            print(f"\nStatistics: {json.dumps(stats, indent=2, default=str)}")

        finally:
            await multiplexer.stop()

    asyncio.run(main())