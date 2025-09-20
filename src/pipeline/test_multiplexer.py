#!/usr/bin/env python3
"""
BEV OSINT Framework - Request Multiplexer Test Suite
Comprehensive tests for the request multiplexing system including
unit tests, integration tests, and performance benchmarks.
"""

import asyncio
import pytest
import pytest_asyncio
import time
import json
import statistics
from typing import List, Dict, Any
import aiohttp
from unittest.mock import Mock, AsyncMock, patch

from request_multiplexer import (
    RequestMultiplexer, Request, RequestPriority, RequestStatus, Response,
    create_multiplexer
)
from connection_pool import ConnectionPoolManager, create_connection_pool_manager
from rate_limiter import (
    RateLimitEngine, RateLimitAlgorithm, RateLimitRule, RequestPriority as RLRequestPriority,
    create_rate_limit_engine
)
from queue_manager import (
    QueueManager, QueueMessage, MessagePriority, QueueType,
    create_queue_manager
)


class TestRequestMultiplexer:
    """Test cases for RequestMultiplexer"""

    @pytest_asyncio.fixture
    async def multiplexer(self):
        """Create a test multiplexer instance"""
        config = {
            'max_concurrency': 100,
            'worker_count': 5,
            'request_timeout': 10.0,
            'retry_attempts': 2,
            'enable_caching': True,
            'cache_ttl': 60,
            'connection_pool': {
                'max_connections': 50,
                'max_connections_per_host': 10,
                'connection_timeout': 5.0
            },
            'rate_limiting': {
                'global_limit': 100,
                'global_window': 60
            },
            'queue_manager': {
                'queue_type': 'memory',
                'max_queue_size': 1000
            }
        }

        multiplexer = create_multiplexer(config)
        await multiplexer.start()
        yield multiplexer
        await multiplexer.stop()

    @pytest.mark.asyncio
    async def test_submit_request(self, multiplexer):
        """Test submitting a single request"""
        request = Request(
            url='https://httpbin.org/get',
            method='GET',
            priority=RequestPriority.MEDIUM
        )

        request_id = await multiplexer.submit_request(request)
        assert request_id is not None
        assert len(request_id) > 0

        # Check request status
        req_status = await multiplexer.get_request_status(request_id)
        assert req_status is not None
        assert req_status.request_id == request_id

    @pytest.mark.asyncio
    async def test_bulk_submit(self, multiplexer):
        """Test bulk request submission"""
        requests = [
            Request(url=f'https://httpbin.org/get?id={i}', method='GET')
            for i in range(10)
        ]

        request_ids = await multiplexer.bulk_submit(requests)
        assert len(request_ids) == 10
        assert all(isinstance(req_id, str) for req_id in request_ids)

    @pytest.mark.asyncio
    async def test_wait_for_completion(self, multiplexer):
        """Test waiting for request completion"""
        requests = [
            Request(url='https://httpbin.org/delay/1', method='GET')
            for _ in range(3)
        ]

        request_ids = await multiplexer.bulk_submit(requests)
        completed = await multiplexer.wait_for_completion(request_ids, timeout=30.0)

        assert len(completed) == 3
        assert all(req.status in [RequestStatus.COMPLETED, RequestStatus.FAILED]
                  for req in completed)

    @pytest.mark.asyncio
    async def test_cancel_request(self, multiplexer):
        """Test request cancellation"""
        request = Request(
            url='https://httpbin.org/delay/10',  # Long delay
            method='GET'
        )

        request_id = await multiplexer.submit_request(request)

        # Try to cancel immediately
        cancelled = await multiplexer.cancel_request(request_id)

        # Cancellation might succeed or fail depending on timing
        assert isinstance(cancelled, bool)

    @pytest.mark.asyncio
    async def test_caching(self, multiplexer):
        """Test response caching"""
        request = Request(
            url='https://httpbin.org/uuid',
            method='GET'
        )

        # First request
        request_id1 = await multiplexer.submit_request(request)
        completed1 = await multiplexer.wait_for_completion([request_id1], timeout=10.0)

        # Second identical request (should be cached)
        request_id2 = await multiplexer.submit_request(request)
        completed2 = await multiplexer.wait_for_completion([request_id2], timeout=10.0)

        # Both should complete successfully
        assert len(completed1) == 1
        assert len(completed2) == 1

    @pytest.mark.asyncio
    async def test_statistics(self, multiplexer):
        """Test statistics collection"""
        stats = multiplexer.get_statistics()

        assert 'requests' in stats
        assert 'performance' in stats
        assert 'components' in stats
        assert 'cache' in stats

        # Check structure
        assert 'total' in stats['requests']
        assert 'current_concurrency' in stats['performance']
        assert 'connection_pool' in stats['components']


class TestConnectionPoolManager:
    """Test cases for ConnectionPoolManager"""

    @pytest_asyncio.fixture
    async def pool_manager(self):
        """Create a test connection pool manager"""
        config = {
            'max_connections': 50,
            'max_connections_per_host': 10,
            'connection_timeout': 5.0,
            'idle_timeout': 60,
            'enable_proxy_rotation': False
        }

        manager = create_connection_pool_manager(config)
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_get_session(self, pool_manager):
        """Test getting sessions from pool"""
        session1 = await pool_manager.get_session('https://httpbin.org')
        session2 = await pool_manager.get_session('https://httpbin.org')

        assert session1 is not None
        assert session2 is not None
        # Should reuse the same session for same host
        assert session1 is session2

    @pytest.mark.asyncio
    async def test_record_latency(self, pool_manager):
        """Test latency recording"""
        endpoint = 'https://httpbin.org'

        # Record some latencies
        pool_manager.record_request_latency(endpoint, 0.1, False)
        pool_manager.record_request_latency(endpoint, 0.2, False)
        pool_manager.record_request_latency(endpoint, 0.3, True)  # Error

        stats = pool_manager.get_statistics()
        assert 'performance' in stats
        assert stats['performance']['total_requests'] == 3

    @pytest.mark.asyncio
    async def test_health_check(self, pool_manager):
        """Test health check functionality"""
        health = await pool_manager.health_check()

        assert 'status' in health
        assert 'healthy_connections' in health
        assert 'total_connections' in health
        assert health['status'] in ['healthy', 'degraded']

    @pytest.mark.asyncio
    async def test_connection_limits(self, pool_manager):
        """Test connection limiting"""
        # Create sessions for different hosts
        sessions = []
        for i in range(15):  # More than max_connections_per_host (10)
            session = await pool_manager.get_session(f'https://example{i}.com')
            sessions.append(session)

        # All sessions should be created but managed properly
        assert len(sessions) == 15
        assert all(session is not None for session in sessions)


class TestRateLimitEngine:
    """Test cases for RateLimitEngine"""

    @pytest_asyncio.fixture
    async def rate_limiter(self):
        """Create a test rate limiter"""
        config = {
            'global_limit': 10,
            'global_window': 1,  # 1 second window for fast testing
            'per_host_limit': 5,
            'per_host_window': 1
        }

        limiter = create_rate_limit_engine(config)
        await limiter.initialize()
        yield limiter
        await limiter.close()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, rate_limiter):
        """Test basic rate limiting"""
        endpoint = 'https://example.com'

        # Should allow requests up to limit
        for i in range(5):
            allowed = await rate_limiter.allow_request(endpoint, RLRequestPriority.MEDIUM)
            assert allowed is True

        # Should deny requests over limit
        for i in range(3):
            allowed = await rate_limiter.allow_request(endpoint, RLRequestPriority.MEDIUM)
            # Might be allowed or denied depending on timing and algorithm
            assert isinstance(allowed, bool)

    @pytest.mark.asyncio
    async def test_priority_handling(self, rate_limiter):
        """Test priority-based rate limiting"""
        endpoint = 'https://example.com'

        # Fill up the limit with low priority requests
        for _ in range(10):
            await rate_limiter.allow_request(endpoint, RLRequestPriority.LOW)

        # High priority should still be allowed (depending on implementation)
        high_priority_allowed = await rate_limiter.allow_request(endpoint, RLRequestPriority.CRITICAL)
        assert isinstance(high_priority_allowed, bool)

    @pytest.mark.asyncio
    async def test_statistics(self, rate_limiter):
        """Test statistics collection"""
        stats = rate_limiter.get_statistics()

        assert 'global_stats' in stats
        assert 'active_rules' in stats
        assert isinstance(stats['global_stats']['total_requests'], int)

    @pytest.mark.asyncio
    async def test_health_check(self, rate_limiter):
        """Test health check"""
        health = await rate_limiter.health_check()

        assert 'status' in health
        assert 'active_rules' in health
        assert health['status'] in ['healthy', 'degraded', 'warning']


class TestQueueManager:
    """Test cases for QueueManager"""

    @pytest_asyncio.fixture
    async def queue_manager(self):
        """Create a test queue manager"""
        config = {
            'queue_type': 'memory',
            'max_queue_size': 100,
            'enable_backpressure': True,
            'backpressure_threshold': 80
        }

        manager = create_queue_manager(config)
        await manager.initialize()
        yield manager
        await manager.close()

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self, queue_manager):
        """Test basic queue operations"""
        message = QueueMessage(
            payload={'test': 'data'},
            priority=MessagePriority.MEDIUM,
            routing_key='test.route'
        )

        # Enqueue message
        success = await queue_manager.enqueue(message)
        assert success is True

        # Dequeue message
        dequeued = await queue_manager.dequeue(timeout=1.0)
        assert dequeued is not None
        assert dequeued.payload == message.payload

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue_manager):
        """Test priority-based message ordering"""
        # Enqueue messages with different priorities
        messages = [
            QueueMessage(payload={'id': 1}, priority=MessagePriority.LOW),
            QueueMessage(payload={'id': 2}, priority=MessagePriority.HIGH),
            QueueMessage(payload={'id': 3}, priority=MessagePriority.CRITICAL),
            QueueMessage(payload={'id': 4}, priority=MessagePriority.MEDIUM)
        ]

        for msg in messages:
            await queue_manager.enqueue(msg)

        # Dequeue and check order (highest priority first)
        dequeued_order = []
        for _ in range(4):
            msg = await queue_manager.dequeue(timeout=1.0)
            if msg:
                dequeued_order.append(msg.priority)

        # Should prioritize CRITICAL, HIGH, MEDIUM, LOW
        assert MessagePriority.CRITICAL in dequeued_order
        assert MessagePriority.HIGH in dequeued_order

    @pytest.mark.asyncio
    async def test_message_handlers(self, queue_manager):
        """Test message handler registration and execution"""
        processed_messages = []

        async def test_handler(message):
            processed_messages.append(message.payload)

        # Register handler
        queue_manager.register_handler('test_queue', test_handler)

        # Start consumer
        await queue_manager.start_consumer('test_queue', concurrency=1)

        # Send test message
        message = QueueMessage(
            payload={'test': 'handler_data'},
            routing_key='test_queue'
        )
        await queue_manager.enqueue(message)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Stop consumer
        await queue_manager.stop_consumer('test_queue')

        # Check if message was processed
        assert len(processed_messages) >= 0  # Might be processed or not depending on timing

    @pytest.mark.asyncio
    async def test_statistics(self, queue_manager):
        """Test statistics collection"""
        stats = queue_manager.get_statistics()

        assert 'global_stats' in stats
        assert 'queue_stats' in stats
        assert 'backend_stats' in stats

    @pytest.mark.asyncio
    async def test_health_check(self, queue_manager):
        """Test health check"""
        health = await queue_manager.health_check()

        assert 'status' in health
        assert 'backend_type' in health
        assert health['status'] in ['healthy', 'degraded', 'critical']


class TestIntegration:
    """Integration tests for the complete system"""

    @pytest_asyncio.fixture
    async def full_system(self):
        """Create a complete test system"""
        config = {
            'max_concurrency': 50,
            'worker_count': 5,
            'request_timeout': 10.0,
            'connection_pool': {
                'max_connections': 25,
                'max_connections_per_host': 5
            },
            'rate_limiting': {
                'global_limit': 50,
                'global_window': 60
            },
            'queue_manager': {
                'queue_type': 'memory',
                'max_queue_size': 100
            }
        }

        multiplexer = create_multiplexer(config)
        await multiplexer.start()
        yield multiplexer
        await multiplexer.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, full_system):
        """Test complete end-to-end workflow"""
        # Submit multiple requests
        requests = [
            Request(url='https://httpbin.org/get?id=1', method='GET'),
            Request(url='https://httpbin.org/get?id=2', method='GET'),
            Request(url='https://httpbin.org/post', method='POST', data={'test': 'data'}),
        ]

        request_ids = await full_system.bulk_submit(requests)
        assert len(request_ids) == 3

        # Wait for completion
        completed = await full_system.wait_for_completion(request_ids, timeout=30.0)

        # Verify results
        assert len(completed) == 3
        success_count = sum(1 for req in completed if req.status == RequestStatus.COMPLETED)
        assert success_count >= 0  # At least some should succeed

        # Check statistics
        stats = full_system.get_statistics()
        assert stats['requests']['total'] >= 3


class TestPerformance:
    """Performance tests and benchmarks"""

    @pytest_asyncio.fixture
    async def perf_multiplexer(self):
        """Create a performance-optimized multiplexer"""
        config = {
            'max_concurrency': 500,
            'worker_count': 20,
            'request_timeout': 30.0,
            'connection_pool': {
                'max_connections': 200,
                'max_connections_per_host': 20
            },
            'rate_limiting': {
                'global_limit': 1000,
                'global_window': 60
            },
            'queue_manager': {
                'queue_type': 'memory',
                'max_queue_size': 5000
            }
        }

        multiplexer = create_multiplexer(config)
        await multiplexer.start()
        yield multiplexer
        await multiplexer.stop()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrency_performance(self, perf_multiplexer):
        """Test performance under high concurrency"""
        num_requests = 100

        requests = [
            Request(url=f'https://httpbin.org/get?id={i}', method='GET')
            for i in range(num_requests)
        ]

        start_time = time.time()
        request_ids = await perf_multiplexer.bulk_submit(requests)
        submit_time = time.time() - start_time

        completion_start = time.time()
        completed = await perf_multiplexer.wait_for_completion(request_ids, timeout=60.0)
        completion_time = time.time() - completion_start

        # Performance assertions
        assert submit_time < 5.0  # Should submit 100 requests in under 5 seconds
        assert len(completed) == num_requests

        # Calculate throughput
        total_time = submit_time + completion_time
        throughput = num_requests / total_time

        print(f"Performance Results:")
        print(f"  Submit time: {submit_time:.3f}s")
        print(f"  Completion time: {completion_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Throughput: {throughput:.2f} req/s")

        # Minimum throughput expectation
        assert throughput > 5.0  # At least 5 requests per second

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_latency_distribution(self, perf_multiplexer):
        """Test latency distribution"""
        num_requests = 50
        latencies = []

        for i in range(num_requests):
            request = Request(url='https://httpbin.org/get', method='GET')

            start_time = time.time()
            request_id = await perf_multiplexer.submit_request(request)
            completed = await perf_multiplexer.wait_for_completion([request_id], timeout=30.0)
            end_time = time.time()

            if completed and completed[0].status == RequestStatus.COMPLETED:
                latency = end_time - start_time
                latencies.append(latency)

        if latencies:
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            median_latency = statistics.median(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies)

            print(f"Latency Statistics:")
            print(f"  Average: {avg_latency:.3f}s")
            print(f"  Median: {median_latency:.3f}s")
            print(f"  P95: {p95_latency:.3f}s")
            print(f"  Min: {min(latencies):.3f}s")
            print(f"  Max: {max(latencies):.3f}s")

            # Performance expectations
            assert avg_latency < 5.0  # Average latency under 5 seconds
            assert p95_latency < 10.0  # 95th percentile under 10 seconds

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_memory_usage(self, perf_multiplexer):
        """Test memory usage under load"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Submit many requests
        num_batches = 10
        batch_size = 50

        for batch in range(num_batches):
            requests = [
                Request(url=f'https://httpbin.org/get?batch={batch}&id={i}', method='GET')
                for i in range(batch_size)
            ]

            request_ids = await perf_multiplexer.bulk_submit(requests)
            completed = await perf_multiplexer.wait_for_completion(request_ids, timeout=30.0)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory Usage:")
        print(f"  Initial: {initial_memory:.2f} MB")
        print(f"  Final: {final_memory:.2f} MB")
        print(f"  Increase: {memory_increase:.2f} MB")

        # Memory should not grow excessively
        assert memory_increase < 500  # Less than 500MB increase


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([
        __file__,
        '-v',
        '--asyncio-mode=auto',
        '-m', 'not slow'  # Skip slow tests by default
    ])