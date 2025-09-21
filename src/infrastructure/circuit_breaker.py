"""
Circuit Breaker Implementation for BEV Auto-Recovery System
==========================================================

Advanced circuit breaker pattern with configurable failure thresholds,
adaptive timeouts, and state management for microservices reliability.

Features:
- Configurable failure thresholds and timeout periods
- Exponential backoff with jitter for retry timing
- Health check integration with service discovery
- State persistence across service restarts
- Metrics collection and alerting integration
- Bulkhead pattern support for resource isolation

Author: BEV Infrastructure Team
Version: 1.0.0
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from functools import wraps
import json
import redis
from contextlib import asynccontextmanager
import aiohttp
import backoff


class CircuitBreakerState(Enum):
    """Circuit breaker states following the classic pattern."""
    CLOSED = "closed"        # Normal operation, requests pass through
    OPEN = "open"           # Failing fast, requests rejected immediately
    HALF_OPEN = "half_open" # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    # Failure detection
    failure_threshold: int = 5                    # Number of failures to trigger open state
    failure_rate_threshold: float = 0.5          # Failure rate (0.0-1.0) to trigger open
    minimum_request_threshold: int = 10          # Minimum requests before rate calculation

    # Timing configuration
    timeout_duration: float = 60.0               # How long to stay open (seconds)
    half_open_max_calls: int = 3                 # Max calls allowed in half-open state
    half_open_success_threshold: int = 2         # Successes needed to close from half-open

    # Request timeout
    request_timeout: float = 30.0                # Individual request timeout

    # Exponential backoff
    backoff_multiplier: float = 2.0              # Backoff multiplier for retries
    max_backoff_time: float = 300.0              # Maximum backoff time (5 minutes)
    jitter_enabled: bool = True                  # Add randomness to backoff

    # Bulkhead settings
    max_concurrent_calls: int = 100              # Maximum concurrent requests
    queue_size: int = 1000                       # Queue size for waiting requests

    # Health check
    health_check_interval: float = 30.0          # Health check frequency (seconds)
    health_check_timeout: float = 10.0           # Health check timeout

    # State persistence
    state_persistence_enabled: bool = True       # Persist state to Redis
    state_key_prefix: str = "cb_state"          # Redis key prefix


@dataclass
class CallResult:
    """Result of a circuit breaker call."""
    success: bool
    response: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    circuit_breaker_state: CircuitBreakerState = CircuitBreakerState.CLOSED


class CircuitBreakerMetrics:
    """Metrics collection for circuit breaker monitoring."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self.reset_metrics()
        self._lock = threading.Lock()

    def reset_metrics(self):
        """Reset all metrics to initial state."""
        with self._lock:
            self.total_requests = 0
            self.successful_requests = 0
            self.failed_requests = 0
            self.rejected_requests = 0
            self.timeout_requests = 0
            self.state_changes = 0
            self.last_failure_time: Optional[datetime] = None
            self.last_success_time: Optional[datetime] = None
            self.circuit_open_time: Optional[datetime] = None
            self.total_downtime = timedelta()
            self.average_response_time = 0.0
            self.response_times: List[float] = []

    def record_request(self, result: CallResult):
        """Record a request result."""
        with self._lock:
            self.total_requests += 1
            self.response_times.append(result.execution_time)

            # Keep only last 100 response times for memory efficiency
            if len(self.response_times) > 100:
                self.response_times = self.response_times[-100:]

            # Update average response time
            self.average_response_time = sum(self.response_times) / len(self.response_times)

            if result.success:
                self.successful_requests += 1
                self.last_success_time = result.timestamp
            else:
                self.failed_requests += 1
                self.last_failure_time = result.timestamp

    def record_rejection(self):
        """Record a rejected request."""
        with self._lock:
            self.rejected_requests += 1

    def record_timeout(self):
        """Record a timeout."""
        with self._lock:
            self.timeout_requests += 1

    def record_state_change(self, new_state: CircuitBreakerState):
        """Record a state change."""
        with self._lock:
            self.state_changes += 1
            if new_state == CircuitBreakerState.OPEN:
                self.circuit_open_time = datetime.utcnow()
            elif new_state == CircuitBreakerState.CLOSED and self.circuit_open_time:
                self.total_downtime += datetime.utcnow() - self.circuit_open_time
                self.circuit_open_time = None

    def get_failure_rate(self) -> float:
        """Calculate current failure rate."""
        with self._lock:
            if self.total_requests == 0:
                return 0.0
            return self.failed_requests / self.total_requests

    def get_availability(self) -> float:
        """Calculate service availability percentage."""
        with self._lock:
            if self.total_requests == 0:
                return 100.0
            return (self.successful_requests / self.total_requests) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        with self._lock:
            return {
                'service_name': self.service_name,
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'rejected_requests': self.rejected_requests,
                'timeout_requests': self.timeout_requests,
                'state_changes': self.state_changes,
                'failure_rate': self.get_failure_rate(),
                'availability': self.get_availability(),
                'average_response_time': self.average_response_time,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'last_success_time': self.last_success_time.isoformat() if self.last_success_time else None,
                'total_downtime_seconds': self.total_downtime.total_seconds()
            }


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    def __init__(self, service_name: str, state: CircuitBreakerState):
        self.service_name = service_name
        self.state = state
        super().__init__(f"Circuit breaker for {service_name} is {state.value}")


class CircuitBreaker:
    """
    Advanced circuit breaker implementation with comprehensive monitoring.

    Provides protection against cascading failures in microservices architecture
    with configurable thresholds, state persistence, and health checking.
    """

    def __init__(self,
                 service_name: str,
                 config: Optional[CircuitBreakerConfig] = None,
                 redis_client: Optional[redis.Redis] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize circuit breaker.

        Args:
            service_name: Unique identifier for the protected service
            config: Circuit breaker configuration
            redis_client: Redis client for state persistence
            logger: Custom logger instance
        """
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        self.redis_client = redis_client
        self.logger = logger or logging.getLogger(f"circuit_breaker.{service_name}")

        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        self.half_open_calls = 0

        # Concurrency control
        self._lock = threading.RLock()
        self.active_calls = 0
        self.call_queue = asyncio.Queue(maxsize=self.config.queue_size)

        # Metrics and monitoring
        self.metrics = CircuitBreakerMetrics(service_name)

        # Health checking
        self.health_check_task: Optional[asyncio.Task] = None
        self.health_check_url: Optional[str] = None

        # Load persisted state
        if self.config.state_persistence_enabled and self.redis_client:
            self._load_state()

        self.logger.info(f"Circuit breaker initialized for {service_name}")

    def _get_state_key(self) -> str:
        """Get Redis key for state persistence."""
        return f"{self.config.state_key_prefix}:{self.service_name}"

    def _save_state(self):
        """Save current state to Redis."""
        if not (self.config.state_persistence_enabled and self.redis_client):
            return

        try:
            state_data = {
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
                'next_attempt_time': self.next_attempt_time.isoformat() if self.next_attempt_time else None,
                'half_open_calls': self.half_open_calls,
                'metrics': self.metrics.to_dict()
            }

            self.redis_client.setex(
                self._get_state_key(),
                3600,  # 1 hour TTL
                json.dumps(state_data, default=str)
            )
        except Exception as e:
            self.logger.warning(f"Failed to save circuit breaker state: {e}")

    def _load_state(self):
        """Load state from Redis."""
        if not (self.config.state_persistence_enabled and self.redis_client):
            return

        try:
            state_data = self.redis_client.get(self._get_state_key())
            if not state_data:
                return

            data = json.loads(state_data)

            self.state = CircuitBreakerState(data['state'])
            self.failure_count = data['failure_count']
            self.success_count = data['success_count']
            self.half_open_calls = data['half_open_calls']

            if data['last_failure_time']:
                self.last_failure_time = datetime.fromisoformat(data['last_failure_time'])

            if data['next_attempt_time']:
                self.next_attempt_time = datetime.fromisoformat(data['next_attempt_time'])

            self.logger.info(f"Loaded circuit breaker state: {self.state.value}")

        except Exception as e:
            self.logger.warning(f"Failed to load circuit breaker state: {e}")

    def _calculate_backoff_time(self) -> float:
        """Calculate exponential backoff time with jitter."""
        base_time = min(
            self.config.timeout_duration * (self.config.backoff_multiplier ** self.failure_count),
            self.config.max_backoff_time
        )

        if self.config.jitter_enabled:
            import random
            # Add ±25% jitter
            jitter = base_time * 0.25 * (2 * random.random() - 1)
            return max(0, base_time + jitter)

        return base_time

    def _should_attempt_call(self) -> bool:
        """Determine if a call should be attempted based on current state."""
        now = datetime.utcnow()

        if self.state == CircuitBreakerState.CLOSED:
            return True

        elif self.state == CircuitBreakerState.OPEN:
            if self.next_attempt_time and now >= self.next_attempt_time:
                self._transition_to_half_open()
                return True
            return False

        elif self.state == CircuitBreakerState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls

        return False

    def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        if self.state != CircuitBreakerState.OPEN:
            old_state = self.state
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = datetime.utcnow() + timedelta(seconds=self._calculate_backoff_time())
            self.metrics.record_state_change(self.state)
            self._save_state()

            self.logger.warning(
                f"Circuit breaker {old_state.value} -> {self.state.value} "
                f"(failures: {self.failure_count}, next attempt: {self.next_attempt_time})"
            )

    def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        if self.state != CircuitBreakerState.HALF_OPEN:
            old_state = self.state
            self.state = CircuitBreakerState.HALF_OPEN
            self.half_open_calls = 0
            self.success_count = 0
            self.metrics.record_state_change(self.state)
            self._save_state()

            self.logger.info(f"Circuit breaker {old_state.value} -> {self.state.value}")

    def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        if self.state != CircuitBreakerState.CLOSED:
            old_state = self.state
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.next_attempt_time = None
            self.metrics.record_state_change(self.state)
            self._save_state()

            self.logger.info(f"Circuit breaker {old_state.value} -> {self.state.value}")

    def _record_success(self):
        """Record a successful call and update state accordingly."""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.half_open_success_threshold:
                    self._transition_to_closed()
            elif self.state == CircuitBreakerState.CLOSED:
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)

    def _record_failure(self):
        """Record a failed call and update state accordingly."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()

            if self.state == CircuitBreakerState.HALF_OPEN:
                # Any failure in half-open state transitions back to open
                self._transition_to_open()
            elif self.state == CircuitBreakerState.CLOSED:
                # Check if we should open the circuit
                should_open = False

                # Check failure count threshold
                if self.failure_count >= self.config.failure_threshold:
                    should_open = True

                # Check failure rate threshold
                if (self.metrics.total_requests >= self.config.minimum_request_threshold and
                    self.metrics.get_failure_rate() >= self.config.failure_rate_threshold):
                    should_open = True

                if should_open:
                    self._transition_to_open()

    @asynccontextmanager
    async def _acquire_call_slot(self):
        """Acquire a slot for making a call (bulkhead pattern)."""
        if self.active_calls >= self.config.max_concurrent_calls:
            try:
                await asyncio.wait_for(
                    self.call_queue.put(None),
                    timeout=self.config.request_timeout
                )
            except asyncio.TimeoutError:
                self.metrics.record_rejection()
                raise CircuitBreakerException(self.service_name, self.state)

        self.active_calls += 1
        try:
            yield
        finally:
            self.active_calls -= 1
            try:
                self.call_queue.get_nowait()
                self.call_queue.task_done()
            except asyncio.QueueEmpty:
                pass

    async def call(self, func: Callable, *args, **kwargs) -> CallResult:
        """
        Execute a function call through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            CallResult: Result of the function call

        Raises:
            CircuitBreakerException: If circuit breaker is open
        """
        with self._lock:
            if not self._should_attempt_call():
                self.metrics.record_rejection()
                raise CircuitBreakerException(self.service_name, self.state)

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.half_open_calls += 1

        start_time = time.time()
        result = CallResult(success=False, circuit_breaker_state=self.state)

        try:
            async with self._acquire_call_slot():
                # Execute the function with timeout
                if asyncio.iscoroutinefunction(func):
                    response = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.request_timeout
                    )
                else:
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: func(*args, **kwargs)
                    )

                result.success = True
                result.response = response

        except asyncio.TimeoutError:
            result.error = Exception(f"Request timeout after {self.config.request_timeout}s")
            self.metrics.record_timeout()

        except Exception as e:
            result.error = e

        finally:
            result.execution_time = time.time() - start_time
            result.timestamp = datetime.utcnow()

            # Record metrics and update state
            self.metrics.record_request(result)

            if result.success:
                self._record_success()
            else:
                self._record_failure()

            self._save_state()

        return result

    def set_health_check_url(self, url: str):
        """Set health check URL for automatic recovery testing."""
        self.health_check_url = url
        self.logger.info(f"Health check URL set: {url}")

    async def perform_health_check(self) -> bool:
        """
        Perform health check against the service.

        Returns:
            bool: True if service is healthy, False otherwise
        """
        if not self.health_check_url:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.health_check_url,
                    timeout=aiohttp.ClientTimeout(total=self.config.health_check_timeout)
                ) as response:
                    return response.status == 200
        except Exception as e:
            self.logger.debug(f"Health check failed: {e}")
            return False

    async def start_health_monitoring(self):
        """Start background health check monitoring."""
        if self.health_check_task:
            return

        async def health_check_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.health_check_interval)

                    if self.state == CircuitBreakerState.OPEN:
                        is_healthy = await self.perform_health_check()
                        if is_healthy:
                            self.logger.info("Health check passed, attempting transition to half-open")
                            with self._lock:
                                self.next_attempt_time = datetime.utcnow()

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Health check loop error: {e}")

        self.health_check_task = asyncio.create_task(health_check_loop())
        self.logger.info("Health monitoring started")

    def stop_health_monitoring(self):
        """Stop background health check monitoring."""
        if self.health_check_task:
            self.health_check_task.cancel()
            self.health_check_task = None
            self.logger.info("Health monitoring stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status and metrics."""
        return {
            'service_name': self.service_name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'half_open_calls': self.half_open_calls,
            'active_calls': self.active_calls,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'next_attempt_time': self.next_attempt_time.isoformat() if self.next_attempt_time else None,
            'metrics': self.metrics.to_dict(),
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'failure_rate_threshold': self.config.failure_rate_threshold,
                'timeout_duration': self.config.timeout_duration,
                'max_concurrent_calls': self.config.max_concurrent_calls,
                'request_timeout': self.config.request_timeout
            }
        }

    def reset(self):
        """Reset circuit breaker to initial state."""
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_calls = 0
            self.last_failure_time = None
            self.next_attempt_time = None
            self.metrics.reset_metrics()
            self._save_state()

            self.logger.info("Circuit breaker reset to initial state")


def circuit_breaker(service_name: str,
                   config: Optional[CircuitBreakerConfig] = None,
                   redis_client: Optional[redis.Redis] = None):
    """
    Decorator for applying circuit breaker pattern to functions.

    Args:
        service_name: Name of the service being protected
        config: Circuit breaker configuration
        redis_client: Redis client for state persistence
    """
    def decorator(func: Callable):
        cb = CircuitBreaker(service_name, config, redis_client)

        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await cb.call(func, *args, **kwargs)
            if result.success:
                return result.response
            else:
                raise result.error

        # Attach circuit breaker instance to function for monitoring
        wrapper.circuit_breaker = cb
        return wrapper

    return decorator


# Utility function for creating circuit breaker instances
def create_circuit_breaker(service_name: str,
                          failure_threshold: int = 5,
                          timeout_duration: float = 60.0,
                          redis_url: Optional[str] = None) -> CircuitBreaker:
    """
    Convenience function to create a circuit breaker with common settings.

    Args:
        service_name: Name of the service being protected
        failure_threshold: Number of failures to trigger open state
        timeout_duration: How long to stay open (seconds)
        redis_url: Redis connection URL for state persistence

    Returns:
        CircuitBreaker: Configured circuit breaker instance
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout_duration=timeout_duration
    )

    redis_client = None
    if redis_url:
        redis_client = redis.from_url(redis_url)

    return CircuitBreaker(service_name, config, redis_client)