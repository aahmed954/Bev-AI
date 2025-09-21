#!/usr/bin/env python3
"""
BEV OSINT Framework - Rate Limiting Engine
Advanced rate limiting with multiple algorithms including token bucket,
sliding window, fixed window, and adaptive limiting strategies.
"""

import asyncio
import time
import math
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
import redis.asyncio as redis
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithm types"""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"


class RequestPriority(Enum):
    """Request priority levels for rate limiting"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    algorithm: RateLimitAlgorithm
    limit: int  # requests
    window: int  # seconds
    burst_limit: Optional[int] = None
    priority_weights: Dict[RequestPriority, float] = field(default_factory=lambda: {
        RequestPriority.CRITICAL: 1.0,
        RequestPriority.HIGH: 0.8,
        RequestPriority.MEDIUM: 0.6,
        RequestPriority.LOW: 0.4,
        RequestPriority.BACKGROUND: 0.2
    })
    adaptive_enabled: bool = False
    adaptive_min_limit: int = 1
    adaptive_max_limit: int = 1000
    backoff_factor: float = 2.0
    enabled: bool = True


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[float] = None
    algorithm_used: str = ""
    rule_name: str = ""


@dataclass
class RateLimitStats:
    """Rate limiting statistics"""
    total_requests: int = 0
    allowed_requests: int = 0
    denied_requests: int = 0
    adaptive_adjustments: int = 0
    rules_triggered: Dict[str, int] = field(default_factory=dict)
    algorithm_usage: Dict[str, int] = field(default_factory=dict)

    @property
    def denial_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.denied_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.allowed_requests / self.total_requests


class RateLimiter(ABC):
    """Abstract base class for rate limiters"""

    def __init__(self, rule: RateLimitRule):
        self.rule = rule

    @abstractmethod
    async def check_limit(self, key: str, priority: RequestPriority = RequestPriority.MEDIUM) -> RateLimitResult:
        """Check if request is within rate limit"""
        pass

    @abstractmethod
    async def reset(self, key: str):
        """Reset rate limit for a key"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get limiter statistics"""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter implementation"""

    def __init__(self, rule: RateLimitRule, redis_client: Optional[redis.Redis] = None):
        super().__init__(rule)
        self.redis_client = redis_client
        self.local_buckets = {}  # Local fallback if no Redis

    async def check_limit(self, key: str, priority: RequestPriority = RequestPriority.MEDIUM) -> RateLimitResult:
        """Check token bucket limit"""
        now = time.time()

        # Calculate effective limit based on priority
        priority_weight = self.rule.priority_weights.get(priority, 1.0)
        effective_limit = int(self.rule.limit * priority_weight)

        if self.redis_client:
            return await self._check_redis_bucket(key, effective_limit, now)
        else:
            return await self._check_local_bucket(key, effective_limit, now)

    async def _check_redis_bucket(self, key: str, limit: int, now: float) -> RateLimitResult:
        """Redis-based token bucket"""
        bucket_key = f"rate_limit:token_bucket:{self.rule.name}:{key}"

        # Lua script for atomic token bucket operation
        script = """
        local bucket_key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        local now = tonumber(ARGV[3])
        local tokens_to_add = tonumber(ARGV[4])

        local bucket = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or limit
        local last_refill = tonumber(bucket[2]) or now

        -- Calculate tokens to add based on time elapsed
        local time_elapsed = now - last_refill
        local new_tokens = math.min(limit, tokens + math.floor(time_elapsed * tokens_to_add))

        local allowed = 0
        local remaining = new_tokens

        if new_tokens >= 1 then
            new_tokens = new_tokens - 1
            remaining = new_tokens
            allowed = 1
        end

        -- Update bucket
        redis.call('HMSET', bucket_key, 'tokens', new_tokens, 'last_refill', now)
        redis.call('EXPIRE', bucket_key, window * 2)

        return {allowed, remaining, now + window}
        """

        tokens_per_second = limit / self.rule.window
        result = await self.redis_client.eval(
            script, 1, bucket_key, limit, self.rule.window, now, tokens_per_second
        )

        allowed, remaining, reset_time = result
        return RateLimitResult(
            allowed=bool(allowed),
            remaining=int(remaining),
            reset_time=datetime.fromtimestamp(reset_time),
            algorithm_used="token_bucket",
            rule_name=self.rule.name
        )

    async def _check_local_bucket(self, key: str, limit: int, now: float) -> RateLimitResult:
        """Local memory token bucket"""
        if key not in self.local_buckets:
            self.local_buckets[key] = {
                'tokens': limit,
                'last_refill': now
            }

        bucket = self.local_buckets[key]

        # Calculate tokens to add
        time_elapsed = now - bucket['last_refill']
        tokens_per_second = limit / self.rule.window
        tokens_to_add = time_elapsed * tokens_per_second

        bucket['tokens'] = min(limit, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now

        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return RateLimitResult(
                allowed=True,
                remaining=int(bucket['tokens']),
                reset_time=datetime.fromtimestamp(now + self.rule.window),
                algorithm_used="token_bucket",
                rule_name=self.rule.name
            )
        else:
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=datetime.fromtimestamp(now + self.rule.window),
                retry_after=1.0 / tokens_per_second,
                algorithm_used="token_bucket",
                rule_name=self.rule.name
            )

    async def reset(self, key: str):
        """Reset token bucket"""
        if self.redis_client:
            bucket_key = f"rate_limit:token_bucket:{self.rule.name}:{key}"
            await self.redis_client.delete(bucket_key)
        else:
            self.local_buckets.pop(key, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get token bucket statistics"""
        return {
            'algorithm': 'token_bucket',
            'rule_name': self.rule.name,
            'limit': self.rule.limit,
            'window': self.rule.window,
            'active_buckets': len(self.local_buckets)
        }


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter implementation"""

    def __init__(self, rule: RateLimitRule, redis_client: Optional[redis.Redis] = None):
        super().__init__(rule)
        self.redis_client = redis_client
        self.local_windows = defaultdict(deque)

    async def check_limit(self, key: str, priority: RequestPriority = RequestPriority.MEDIUM) -> RateLimitResult:
        """Check sliding window limit"""
        now = time.time()
        window_start = now - self.rule.window

        # Calculate effective limit based on priority
        priority_weight = self.rule.priority_weights.get(priority, 1.0)
        effective_limit = int(self.rule.limit * priority_weight)

        if self.redis_client:
            return await self._check_redis_window(key, effective_limit, now, window_start)
        else:
            return await self._check_local_window(key, effective_limit, now, window_start)

    async def _check_redis_window(self, key: str, limit: int, now: float, window_start: float) -> RateLimitResult:
        """Redis-based sliding window"""
        window_key = f"rate_limit:sliding_window:{self.rule.name}:{key}"

        # Lua script for atomic sliding window operation
        script = """
        local window_key = KEYS[1]
        local now = tonumber(ARGV[1])
        local window_start = tonumber(ARGV[2])
        local limit = tonumber(ARGV[3])
        local window_size = tonumber(ARGV[4])

        -- Remove old entries
        redis.call('ZREMRANGEBYSCORE', window_key, 0, window_start)

        -- Count current entries
        local current_count = redis.call('ZCARD', window_key)

        local allowed = 0
        local remaining = math.max(0, limit - current_count)

        if current_count < limit then
            -- Add new entry
            redis.call('ZADD', window_key, now, now)
            allowed = 1
            remaining = remaining - 1
        end

        -- Set expiration
        redis.call('EXPIRE', window_key, window_size * 2)

        return {allowed, remaining, current_count}
        """

        result = await self.redis_client.eval(
            script, 1, window_key, now, window_start, limit, self.rule.window
        )

        allowed, remaining, current_count = result
        return RateLimitResult(
            allowed=bool(allowed),
            remaining=int(remaining),
            reset_time=datetime.fromtimestamp(now + self.rule.window),
            algorithm_used="sliding_window",
            rule_name=self.rule.name
        )

    async def _check_local_window(self, key: str, limit: int, now: float, window_start: float) -> RateLimitResult:
        """Local memory sliding window"""
        window = self.local_windows[key]

        # Remove old entries
        while window and window[0] < window_start:
            window.popleft()

        current_count = len(window)

        if current_count < limit:
            window.append(now)
            return RateLimitResult(
                allowed=True,
                remaining=limit - current_count - 1,
                reset_time=datetime.fromtimestamp(now + self.rule.window),
                algorithm_used="sliding_window",
                rule_name=self.rule.name
            )
        else:
            # Calculate retry after based on oldest entry
            oldest_entry = window[0] if window else now
            retry_after = (oldest_entry + self.rule.window) - now

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=datetime.fromtimestamp(oldest_entry + self.rule.window),
                retry_after=max(0, retry_after),
                algorithm_used="sliding_window",
                rule_name=self.rule.name
            )

    async def reset(self, key: str):
        """Reset sliding window"""
        if self.redis_client:
            window_key = f"rate_limit:sliding_window:{self.rule.name}:{key}"
            await self.redis_client.delete(window_key)
        else:
            self.local_windows[key].clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get sliding window statistics"""
        total_entries = sum(len(window) for window in self.local_windows.values())
        return {
            'algorithm': 'sliding_window',
            'rule_name': self.rule.name,
            'limit': self.rule.limit,
            'window': self.rule.window,
            'active_windows': len(self.local_windows),
            'total_entries': total_entries
        }


class FixedWindowLimiter(RateLimiter):
    """Fixed window rate limiter implementation"""

    def __init__(self, rule: RateLimitRule, redis_client: Optional[redis.Redis] = None):
        super().__init__(rule)
        self.redis_client = redis_client
        self.local_counters = {}

    async def check_limit(self, key: str, priority: RequestPriority = RequestPriority.MEDIUM) -> RateLimitResult:
        """Check fixed window limit"""
        now = time.time()
        window_id = int(now // self.rule.window)

        # Calculate effective limit based on priority
        priority_weight = self.rule.priority_weights.get(priority, 1.0)
        effective_limit = int(self.rule.limit * priority_weight)

        if self.redis_client:
            return await self._check_redis_counter(key, effective_limit, window_id, now)
        else:
            return await self._check_local_counter(key, effective_limit, window_id, now)

    async def _check_redis_counter(self, key: str, limit: int, window_id: int, now: float) -> RateLimitResult:
        """Redis-based fixed window"""
        counter_key = f"rate_limit:fixed_window:{self.rule.name}:{key}:{window_id}"

        # Atomic increment and check
        pipeline = self.redis_client.pipeline()
        pipeline.incr(counter_key)
        pipeline.expire(counter_key, self.rule.window * 2)
        result = await pipeline.execute()

        current_count = result[0]

        if current_count <= limit:
            return RateLimitResult(
                allowed=True,
                remaining=limit - current_count,
                reset_time=datetime.fromtimestamp((window_id + 1) * self.rule.window),
                algorithm_used="fixed_window",
                rule_name=self.rule.name
            )
        else:
            # Decrement since we exceeded the limit
            await self.redis_client.decr(counter_key)

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=datetime.fromtimestamp((window_id + 1) * self.rule.window),
                retry_after=(window_id + 1) * self.rule.window - now,
                algorithm_used="fixed_window",
                rule_name=self.rule.name
            )

    async def _check_local_counter(self, key: str, limit: int, window_id: int, now: float) -> RateLimitResult:
        """Local memory fixed window"""
        counter_key = f"{key}:{window_id}"

        if counter_key not in self.local_counters:
            self.local_counters[counter_key] = 0

        self.local_counters[counter_key] += 1
        current_count = self.local_counters[counter_key]

        # Clean up old windows
        old_windows = [k for k in self.local_counters.keys()
                      if int(k.split(':')[-1]) < window_id - 1]
        for old_key in old_windows:
            del self.local_counters[old_key]

        if current_count <= limit:
            return RateLimitResult(
                allowed=True,
                remaining=limit - current_count,
                reset_time=datetime.fromtimestamp((window_id + 1) * self.rule.window),
                algorithm_used="fixed_window",
                rule_name=self.rule.name
            )
        else:
            # Rollback the increment
            self.local_counters[counter_key] -= 1

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=datetime.fromtimestamp((window_id + 1) * self.rule.window),
                retry_after=(window_id + 1) * self.rule.window - now,
                algorithm_used="fixed_window",
                rule_name=self.rule.name
            )

    async def reset(self, key: str):
        """Reset fixed window counters"""
        if self.redis_client:
            pattern = f"rate_limit:fixed_window:{self.rule.name}:{key}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
        else:
            keys_to_delete = [k for k in self.local_counters.keys() if k.startswith(f"{key}:")]
            for k in keys_to_delete:
                del self.local_counters[k]

    def get_stats(self) -> Dict[str, Any]:
        """Get fixed window statistics"""
        return {
            'algorithm': 'fixed_window',
            'rule_name': self.rule.name,
            'limit': self.rule.limit,
            'window': self.rule.window,
            'active_counters': len(self.local_counters)
        }


class AdaptiveLimiter(RateLimiter):
    """Adaptive rate limiter that adjusts limits based on system performance"""

    def __init__(self, rule: RateLimitRule, redis_client: Optional[redis.Redis] = None):
        super().__init__(rule)
        self.redis_client = redis_client
        self.base_limiter = SlidingWindowLimiter(rule, redis_client)

        # Adaptive parameters
        self.current_limit = rule.limit
        self.last_adjustment = time.time()
        self.adjustment_interval = 60  # 1 minute
        self.performance_window = deque(maxlen=100)
        self.error_window = deque(maxlen=100)

    async def check_limit(self, key: str, priority: RequestPriority = RequestPriority.MEDIUM) -> RateLimitResult:
        """Check adaptive limit with dynamic adjustment"""
        # Perform adaptive adjustment if needed
        await self._adjust_limit_if_needed()

        # Create temporary rule with current limit
        temp_rule = RateLimitRule(
            name=self.rule.name,
            algorithm=self.rule.algorithm,
            limit=self.current_limit,
            window=self.rule.window,
            priority_weights=self.rule.priority_weights
        )

        temp_limiter = SlidingWindowLimiter(temp_rule, self.redis_client)
        result = await temp_limiter.check_limit(key, priority)
        result.algorithm_used = "adaptive"

        return result

    async def _adjust_limit_if_needed(self):
        """Adjust rate limit based on performance metrics"""
        now = time.time()

        if now - self.last_adjustment < self.adjustment_interval:
            return

        if len(self.performance_window) < 10:  # Need minimum data
            return

        # Calculate performance metrics
        avg_response_time = sum(self.performance_window) / len(self.performance_window)
        error_rate = sum(self.error_window) / len(self.error_window)

        # Adjustment logic
        if error_rate > 0.1:  # High error rate, decrease limit
            new_limit = max(self.rule.adaptive_min_limit,
                          int(self.current_limit * 0.8))
        elif avg_response_time > 5.0:  # High latency, decrease limit
            new_limit = max(self.rule.adaptive_min_limit,
                          int(self.current_limit * 0.9))
        elif error_rate < 0.01 and avg_response_time < 1.0:  # Good performance, increase limit
            new_limit = min(self.rule.adaptive_max_limit,
                          int(self.current_limit * 1.1))
        else:
            new_limit = self.current_limit

        if new_limit != self.current_limit:
            logger.info(f"Adaptive limiter adjusted limit: {self.current_limit} -> {new_limit}")
            self.current_limit = new_limit
            self.last_adjustment = now

    def record_performance(self, response_time: float, error: bool):
        """Record performance metrics for adaptive adjustment"""
        self.performance_window.append(response_time)
        self.error_window.append(1.0 if error else 0.0)

    async def reset(self, key: str):
        """Reset adaptive limiter"""
        await self.base_limiter.reset(key)

    def get_stats(self) -> Dict[str, Any]:
        """Get adaptive limiter statistics"""
        base_stats = self.base_limiter.get_stats()
        base_stats.update({
            'algorithm': 'adaptive',
            'current_limit': self.current_limit,
            'original_limit': self.rule.limit,
            'avg_response_time': sum(self.performance_window) / len(self.performance_window) if self.performance_window else 0,
            'error_rate': sum(self.error_window) / len(self.error_window) if self.error_window else 0
        })
        return base_stats


class RateLimitEngine:
    """
    Advanced rate limiting engine supporting multiple algorithms
    and intelligent rule management.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_url = config.get('redis_url', 'redis://redis:6379/0')
        self.redis_client: Optional[redis.Redis] = None

        # Rate limiting rules
        self.rules: Dict[str, RateLimitRule] = {}
        self.limiters: Dict[str, RateLimiter] = {}

        # Global statistics
        self.stats = RateLimitStats()

        # Default rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default rate limiting rules"""
        default_rules = [
            RateLimitRule(
                name="global",
                algorithm=RateLimitAlgorithm.SLIDING_WINDOW,
                limit=self.config.get('global_limit', 1000),
                window=self.config.get('global_window', 60)
            ),
            RateLimitRule(
                name="per_host",
                algorithm=RateLimitAlgorithm.TOKEN_BUCKET,
                limit=self.config.get('per_host_limit', 100),
                window=self.config.get('per_host_window', 60)
            ),
            RateLimitRule(
                name="burst_protection",
                algorithm=RateLimitAlgorithm.FIXED_WINDOW,
                limit=self.config.get('burst_limit', 50),
                window=self.config.get('burst_window', 10)
            )
        ]

        for rule in default_rules:
            self.add_rule(rule)

    async def initialize(self):
        """Initialize the rate limiting engine"""
        # Initialize Redis connection
        if self.redis_url:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("Connected to Redis for rate limiting")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, using local storage")
                self.redis_client = None

        # Initialize limiters
        for rule_name, rule in self.rules.items():
            self.limiters[rule_name] = self._create_limiter(rule)

        logger.info(f"RateLimitEngine initialized with {len(self.rules)} rules")

    async def close(self):
        """Close the rate limiting engine"""
        if self.redis_client:
            await self.redis_client.close()

    def add_rule(self, rule: RateLimitRule):
        """Add a rate limiting rule"""
        self.rules[rule.name] = rule
        if hasattr(self, 'limiters'):  # Only create limiter if engine is initialized
            self.limiters[rule.name] = self._create_limiter(rule)
        logger.info(f"Added rate limiting rule: {rule.name}")

    def remove_rule(self, rule_name: str):
        """Remove a rate limiting rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            if rule_name in self.limiters:
                del self.limiters[rule_name]
            logger.info(f"Removed rate limiting rule: {rule_name}")

    def _create_limiter(self, rule: RateLimitRule) -> RateLimiter:
        """Create appropriate limiter for rule"""
        if rule.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return TokenBucketLimiter(rule, self.redis_client)
        elif rule.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return SlidingWindowLimiter(rule, self.redis_client)
        elif rule.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return FixedWindowLimiter(rule, self.redis_client)
        elif rule.algorithm == RateLimitAlgorithm.ADAPTIVE:
            return AdaptiveLimiter(rule, self.redis_client)
        else:
            # Default to sliding window
            return SlidingWindowLimiter(rule, self.redis_client)

    async def allow_request(self, endpoint: str, priority: RequestPriority = RequestPriority.MEDIUM,
                          rules: Optional[List[str]] = None) -> bool:
        """Check if request is allowed by rate limiting rules"""
        self.stats.total_requests += 1

        # Use specified rules or all enabled rules
        rules_to_check = rules or [name for name, rule in self.rules.items() if rule.enabled]

        for rule_name in rules_to_check:
            if rule_name not in self.limiters:
                continue

            limiter = self.limiters[rule_name]
            result = await limiter.check_limit(endpoint, priority)

            # Update statistics
            self.stats.algorithm_usage[result.algorithm_used] = \
                self.stats.algorithm_usage.get(result.algorithm_used, 0) + 1

            if not result.allowed:
                self.stats.denied_requests += 1
                self.stats.rules_triggered[rule_name] = \
                    self.stats.rules_triggered.get(rule_name, 0) + 1
                logger.debug(f"Request denied by rule '{rule_name}' for endpoint '{endpoint}'")
                return False

        self.stats.allowed_requests += 1
        return True

    async def get_limit_status(self, endpoint: str, rule_name: str) -> Optional[RateLimitResult]:
        """Get current rate limit status for endpoint and rule"""
        if rule_name not in self.limiters:
            return None

        limiter = self.limiters[rule_name]
        # This is a check-only operation, we'll use a special key
        check_key = f"status_check:{endpoint}"
        return await limiter.check_limit(check_key, RequestPriority.LOW)

    async def reset_limits(self, endpoint: str, rules: Optional[List[str]] = None):
        """Reset rate limits for endpoint"""
        rules_to_reset = rules or list(self.limiters.keys())

        for rule_name in rules_to_reset:
            if rule_name in self.limiters:
                await self.limiters[rule_name].reset(endpoint)

        logger.info(f"Reset rate limits for endpoint '{endpoint}'")

    def record_performance(self, response_time: float, error: bool):
        """Record performance metrics for adaptive limiters"""
        for limiter in self.limiters.values():
            if isinstance(limiter, AdaptiveLimiter):
                limiter.record_performance(response_time, error)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive rate limiting statistics"""
        limiter_stats = {}
        for name, limiter in self.limiters.items():
            limiter_stats[name] = limiter.get_stats()

        return {
            'global_stats': {
                'total_requests': self.stats.total_requests,
                'allowed_requests': self.stats.allowed_requests,
                'denied_requests': self.stats.denied_requests,
                'denial_rate': self.stats.denial_rate,
                'success_rate': self.stats.success_rate
            },
            'rules_triggered': dict(self.stats.rules_triggered),
            'algorithm_usage': dict(self.stats.algorithm_usage),
            'limiter_stats': limiter_stats,
            'active_rules': list(self.rules.keys())
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on rate limiting engine"""
        health_status = {
            'status': 'healthy',
            'redis_connected': self.redis_client is not None,
            'active_rules': len(self.rules),
            'active_limiters': len(self.limiters),
            'issues': []
        }

        # Check Redis connection
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health_status['redis_latency'] = 'ok'
            except Exception as e:
                health_status['redis_connected'] = False
                health_status['issues'].append(f"Redis connection failed: {e}")
                health_status['status'] = 'degraded'

        # Check if denial rate is too high
        if self.stats.denial_rate > 0.5:
            health_status['issues'].append(f"High denial rate: {self.stats.denial_rate:.2%}")
            health_status['status'] = 'warning'

        return health_status


# Factory function
def create_rate_limit_engine(config: Dict[str, Any]) -> RateLimitEngine:
    """Create and configure a RateLimitEngine instance"""
    return RateLimitEngine(config)


if __name__ == "__main__":
    # Example usage
    async def main():
        config = {
            'redis_url': 'redis://localhost:6379/0',
            'global_limit': 1000,
            'global_window': 60,
            'per_host_limit': 100,
            'per_host_window': 60,
            'burst_limit': 50,
            'burst_window': 10
        }

        engine = create_rate_limit_engine(config)
        await engine.initialize()

        try:
            # Test rate limiting
            endpoint = "https://example.com"

            for i in range(10):
                allowed = await engine.allow_request(endpoint, RequestPriority.MEDIUM)
                print(f"Request {i+1}: {'Allowed' if allowed else 'Denied'}")

                if not allowed:
                    # Check status
                    status = await engine.get_limit_status(endpoint, "per_host")
                    if status:
                        print(f"  Remaining: {status.remaining}, Reset: {status.reset_time}")

            # Get statistics
            stats = engine.get_statistics()
            print(f"\nStatistics: {json.dumps(stats, indent=2, default=str)}")

            # Health check
            health = await engine.health_check()
            print(f"\nHealth: {health}")

        finally:
            await engine.close()

    asyncio.run(main())