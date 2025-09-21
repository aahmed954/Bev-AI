#!/usr/bin/env python3
"""
BEV OSINT Framework - Request Multiplexer Service
Main service module that provides a complete HTTP API for the request multiplexing system.
"""

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List

import uvloop
from aiohttp import web, ClientTimeout
from aiohttp.web_middlewares import cors_handler
from aiohttp_cors import setup as cors_setup, ResourceOptions
import structlog

from request_multiplexer import RequestMultiplexer, Request, RequestPriority, create_multiplexer
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('multiplexer_requests_total', 'Total requests processed', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('multiplexer_request_duration_seconds', 'Request processing duration')
ACTIVE_REQUESTS = Gauge('multiplexer_active_requests', 'Currently active requests')
QUEUE_SIZE = Gauge('multiplexer_queue_size', 'Current queue size')
ERROR_COUNT = Counter('multiplexer_errors_total', 'Total errors', ['error_type'])


class RequestMultiplexerService:
    """HTTP service wrapper for the RequestMultiplexer"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.multiplexer: Optional[RequestMultiplexer] = None
        self.app = web.Application()
        self.setup_routes()
        self.setup_cors()
        self.setup_middlewares()

    def setup_routes(self):
        """Setup HTTP routes"""
        # Health and status
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_get('/status', self.get_status)
        self.app.router.add_get('/metrics', self.get_metrics)

        # Core multiplexer API
        self.app.router.add_post('/api/v1/request', self.submit_request)
        self.app.router.add_get('/api/v1/request/{request_id}', self.get_request_status)
        self.app.router.add_delete('/api/v1/request/{request_id}', self.cancel_request)
        self.app.router.add_post('/api/v1/requests/bulk', self.submit_bulk_requests)
        self.app.router.add_post('/api/v1/requests/wait', self.wait_for_requests)

        # Convenience endpoints
        self.app.router.add_get('/api/v1/get', self.get_request)
        self.app.router.add_post('/api/v1/post', self.post_request)
        self.app.router.add_put('/api/v1/put', self.put_request)
        self.app.router.add_delete('/api/v1/delete', self.delete_request)

        # Statistics and monitoring
        self.app.router.add_get('/api/v1/statistics', self.get_statistics)
        self.app.router.add_get('/api/v1/performance', self.get_performance)

    def setup_cors(self):
        """Setup CORS for cross-origin requests"""
        cors = cors_setup(self.app, defaults={
            "*": ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })

        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)

    def setup_middlewares(self):
        """Setup HTTP middlewares"""
        self.app.middlewares.append(self.logging_middleware)
        self.app.middlewares.append(self.metrics_middleware)
        self.app.middlewares.append(self.error_middleware)

    @web.middleware
    async def logging_middleware(self, request, handler):
        """Log all requests"""
        start_time = asyncio.get_event_loop().time()

        try:
            response = await handler(request)
            duration = asyncio.get_event_loop().time() - start_time

            logger.info(
                "HTTP request processed",
                method=request.method,
                path=request.path,
                status=response.status,
                duration=duration,
                remote=request.remote
            )

            return response
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time

            logger.error(
                "HTTP request failed",
                method=request.method,
                path=request.path,
                duration=duration,
                error=str(e),
                remote=request.remote
            )

            raise

    @web.middleware
    async def metrics_middleware(self, request, handler):
        """Collect Prometheus metrics"""
        start_time = asyncio.get_event_loop().time()

        with REQUEST_DURATION.time():
            try:
                response = await handler(request)
                REQUEST_COUNT.labels(
                    method=request.method,
                    endpoint=request.path,
                    status=response.status
                ).inc()
                return response
            except Exception as e:
                ERROR_COUNT.labels(error_type=type(e).__name__).inc()
                raise

    @web.middleware
    async def error_middleware(self, request, handler):
        """Handle and format errors"""
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except Exception as e:
            logger.error("Unhandled error", error=str(e), path=request.path)
            return web.json_response(
                {'error': 'Internal server error', 'message': str(e)},
                status=500
            )

    async def health_check(self, request):
        """Health check endpoint"""
        if not self.multiplexer:
            return web.json_response({'status': 'unhealthy', 'reason': 'Multiplexer not initialized'}, status=503)

        try:
            stats = self.multiplexer.get_statistics()
            health_status = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'uptime': stats['performance']['uptime'],
                'active_requests': stats['performance']['current_concurrency'],
                'components': {
                    'connection_pool': 'healthy',
                    'rate_limiter': 'healthy',
                    'queue_manager': 'healthy'
                }
            }

            # Check component health
            if stats['performance']['current_concurrency'] > self.config.get('max_concurrency', 1000) * 0.9:
                health_status['status'] = 'degraded'
                health_status['warnings'] = ['High concurrency usage']

            status_code = 200 if health_status['status'] == 'healthy' else 503
            return web.json_response(health_status, status=status_code)

        except Exception as e:
            return web.json_response({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=503)

    async def get_status(self, request):
        """Get detailed system status"""
        if not self.multiplexer:
            return web.json_response({'error': 'Multiplexer not initialized'}, status=503)

        stats = self.multiplexer.get_statistics()
        return web.json_response(stats)

    async def get_metrics(self, request):
        """Prometheus metrics endpoint"""
        # Update dynamic metrics
        if self.multiplexer:
            stats = self.multiplexer.get_statistics()
            ACTIVE_REQUESTS.set(stats['performance']['current_concurrency'])
            QUEUE_SIZE.set(stats['components']['queue_manager'].get('queue_size', 0))

        metrics_output = generate_latest()
        return web.Response(text=metrics_output.decode('utf-8'), content_type=CONTENT_TYPE_LATEST)

    async def submit_request(self, request):
        """Submit a single request for processing"""
        if not self.multiplexer:
            return web.json_response({'error': 'Multiplexer not initialized'}, status=503)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON'}, status=400)

        # Validate required fields
        if 'url' not in data:
            return web.json_response({'error': 'URL is required'}, status=400)

        # Create request object
        try:
            req = Request(
                url=data['url'],
                method=data.get('method', 'GET'),
                headers=data.get('headers', {}),
                data=data.get('data'),
                params=data.get('params', {}),
                timeout=data.get('timeout', 30.0),
                priority=RequestPriority(data.get('priority', RequestPriority.MEDIUM.value)),
                retries=data.get('retries', 3),
                retry_delay=data.get('retry_delay', 1.0),
                context=data.get('context', {})
            )

            request_id = await self.multiplexer.submit_request(req)

            return web.json_response({
                'request_id': request_id,
                'status': 'submitted',
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            logger.error("Failed to submit request", error=str(e))
            return web.json_response({'error': str(e)}, status=400)

    async def get_request_status(self, request):
        """Get status of a specific request"""
        if not self.multiplexer:
            return web.json_response({'error': 'Multiplexer not initialized'}, status=503)

        request_id = request.match_info['request_id']
        req = await self.multiplexer.get_request_status(request_id)

        if not req:
            return web.json_response({'error': 'Request not found'}, status=404)

        # Convert request to JSON-serializable format
        req_data = {
            'request_id': req.request_id,
            'url': req.url,
            'method': req.method,
            'status': req.status.value,
            'created_at': req.created_at.isoformat(),
            'started_at': req.started_at.isoformat() if req.started_at else None,
            'completed_at': req.completed_at.isoformat() if req.completed_at else None,
            'attempts': req.attempts,
            'last_error': req.last_error,
            'result': req.result,
            'latency': req.latency
        }

        return web.json_response(req_data)

    async def cancel_request(self, request):
        """Cancel a pending request"""
        if not self.multiplexer:
            return web.json_response({'error': 'Multiplexer not initialized'}, status=503)

        request_id = request.match_info['request_id']
        cancelled = await self.multiplexer.cancel_request(request_id)

        if cancelled:
            return web.json_response({'status': 'cancelled', 'request_id': request_id})
        else:
            return web.json_response({'error': 'Cannot cancel request'}, status=400)

    async def submit_bulk_requests(self, request):
        """Submit multiple requests in bulk"""
        if not self.multiplexer:
            return web.json_response({'error': 'Multiplexer not initialized'}, status=503)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON'}, status=400)

        if 'requests' not in data:
            return web.json_response({'error': 'requests field is required'}, status=400)

        requests_data = data['requests']
        if not isinstance(requests_data, list):
            return web.json_response({'error': 'requests must be a list'}, status=400)

        # Create request objects
        requests = []
        for req_data in requests_data:
            if 'url' not in req_data:
                continue

            req = Request(
                url=req_data['url'],
                method=req_data.get('method', 'GET'),
                headers=req_data.get('headers', {}),
                data=req_data.get('data'),
                params=req_data.get('params', {}),
                timeout=req_data.get('timeout', 30.0),
                priority=RequestPriority(req_data.get('priority', RequestPriority.MEDIUM.value)),
                retries=req_data.get('retries', 3),
                retry_delay=req_data.get('retry_delay', 1.0),
                context=req_data.get('context', {})
            )
            requests.append(req)

        # Submit all requests
        try:
            request_ids = await self.multiplexer.bulk_submit(requests)

            return web.json_response({
                'request_ids': request_ids,
                'count': len(request_ids),
                'status': 'submitted',
                'timestamp': datetime.now().isoformat()
            })

        except Exception as e:
            logger.error("Failed to submit bulk requests", error=str(e))
            return web.json_response({'error': str(e)}, status=400)

    async def wait_for_requests(self, request):
        """Wait for multiple requests to complete"""
        if not self.multiplexer:
            return web.json_response({'error': 'Multiplexer not initialized'}, status=503)

        try:
            data = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON'}, status=400)

        if 'request_ids' not in data:
            return web.json_response({'error': 'request_ids field is required'}, status=400)

        request_ids = data['request_ids']
        timeout = data.get('timeout', 300.0)

        try:
            completed_requests = await self.multiplexer.wait_for_completion(request_ids, timeout)

            # Convert to JSON-serializable format
            results = []
            for req in completed_requests:
                req_data = {
                    'request_id': req.request_id,
                    'url': req.url,
                    'status': req.status.value,
                    'result': req.result,
                    'latency': req.latency,
                    'last_error': req.last_error
                }
                results.append(req_data)

            return web.json_response({
                'results': results,
                'count': len(results),
                'completed_count': len([r for r in results if r['status'] == 'completed']),
                'failed_count': len([r for r in results if r['status'] == 'failed'])
            })

        except Exception as e:
            logger.error("Failed to wait for requests", error=str(e))
            return web.json_response({'error': str(e)}, status=400)

    # Convenience endpoints
    async def get_request(self, request):
        """Convenience GET request endpoint"""
        url = request.query.get('url')
        if not url:
            return web.json_response({'error': 'url parameter is required'}, status=400)

        req = Request(url=url, method='GET')
        request_id = await self.multiplexer.submit_request(req)

        # Wait for completion
        completed = await self.multiplexer.wait_for_completion([request_id], timeout=30.0)
        if completed:
            return web.json_response(completed[0].result)
        else:
            return web.json_response({'error': 'Request timeout'}, status=408)

    async def post_request(self, request):
        """Convenience POST request endpoint"""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON'}, status=400)

        url = data.get('url')
        if not url:
            return web.json_response({'error': 'url field is required'}, status=400)

        req = Request(
            url=url,
            method='POST',
            data=data.get('data'),
            headers=data.get('headers', {})
        )
        request_id = await self.multiplexer.submit_request(req)

        # Wait for completion
        completed = await self.multiplexer.wait_for_completion([request_id], timeout=30.0)
        if completed:
            return web.json_response(completed[0].result)
        else:
            return web.json_response({'error': 'Request timeout'}, status=408)

    async def put_request(self, request):
        """Convenience PUT request endpoint"""
        try:
            data = await request.json()
        except Exception:
            return web.json_response({'error': 'Invalid JSON'}, status=400)

        url = data.get('url')
        if not url:
            return web.json_response({'error': 'url field is required'}, status=400)

        req = Request(
            url=url,
            method='PUT',
            data=data.get('data'),
            headers=data.get('headers', {})
        )
        request_id = await self.multiplexer.submit_request(req)

        # Wait for completion
        completed = await self.multiplexer.wait_for_completion([request_id], timeout=30.0)
        if completed:
            return web.json_response(completed[0].result)
        else:
            return web.json_response({'error': 'Request timeout'}, status=408)

    async def delete_request(self, request):
        """Convenience DELETE request endpoint"""
        url = request.query.get('url')
        if not url:
            return web.json_response({'error': 'url parameter is required'}, status=400)

        req = Request(url=url, method='DELETE')
        request_id = await self.multiplexer.submit_request(req)

        # Wait for completion
        completed = await self.multiplexer.wait_for_completion([request_id], timeout=30.0)
        if completed:
            return web.json_response(completed[0].result)
        else:
            return web.json_response({'error': 'Request timeout'}, status=408)

    async def get_statistics(self, request):
        """Get comprehensive statistics"""
        if not self.multiplexer:
            return web.json_response({'error': 'Multiplexer not initialized'}, status=503)

        stats = self.multiplexer.get_statistics()
        return web.json_response(stats)

    async def get_performance(self, request):
        """Get performance metrics"""
        if not self.multiplexer:
            return web.json_response({'error': 'Multiplexer not initialized'}, status=503)

        stats = self.multiplexer.get_statistics()
        performance = {
            'requests_per_second': stats['requests']['requests_per_second'],
            'average_latency': stats['performance']['average_latency'],
            'success_rate': stats['requests']['success_rate'],
            'current_concurrency': stats['performance']['current_concurrency'],
            'peak_concurrency': stats['performance']['peak_concurrency'],
            'uptime': stats['performance']['uptime']
        }

        return web.json_response(performance)

    async def start(self):
        """Start the multiplexer service"""
        logger.info("Starting RequestMultiplexer service")

        # Initialize multiplexer
        self.multiplexer = create_multiplexer(self.config)
        await self.multiplexer.start()

        logger.info("RequestMultiplexer service started successfully")

    async def stop(self):
        """Stop the multiplexer service"""
        logger.info("Stopping RequestMultiplexer service")

        if self.multiplexer:
            await self.multiplexer.stop()

        logger.info("RequestMultiplexer service stopped")


def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables"""
    config = {
        # Multiplexer settings
        'max_concurrency': int(os.getenv('MAX_CONCURRENCY', '1000')),
        'worker_count': int(os.getenv('WORKER_COUNT', '50')),
        'request_timeout': float(os.getenv('REQUEST_TIMEOUT', '30.0')),
        'retry_attempts': int(os.getenv('RETRY_ATTEMPTS', '3')),
        'enable_caching': os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
        'cache_ttl': int(os.getenv('CACHE_TTL', '300')),

        # Connection pool
        'connection_pool': {
            'max_connections': int(os.getenv('MAX_CONNECTIONS', '500')),
            'max_connections_per_host': int(os.getenv('MAX_CONNECTIONS_PER_HOST', '50')),
            'connection_timeout': float(os.getenv('CONNECTION_TIMEOUT', '30.0')),
            'idle_timeout': int(os.getenv('IDLE_TIMEOUT', '300')),
            'enable_proxy_rotation': os.getenv('ENABLE_PROXY_ROTATION', 'true').lower() == 'true',
            'proxy_rotation': {
                'proxy_pool_url': os.getenv('PROXY_POOL_URL', 'http://proxy-manager:8000'),
                'strategy': 'best_performance',
                'health_check_interval': 60
            }
        },

        # Rate limiting
        'rate_limiting': {
            'redis_url': os.getenv('REDIS_URL', 'redis://redis:6379/12'),
            'global_limit': int(os.getenv('GLOBAL_LIMIT', '1000')),
            'global_window': int(os.getenv('GLOBAL_WINDOW', '60')),
            'per_host_limit': int(os.getenv('PER_HOST_LIMIT', '100')),
            'per_host_window': int(os.getenv('PER_HOST_WINDOW', '60')),
            'burst_limit': int(os.getenv('BURST_LIMIT', '50')),
            'burst_window': int(os.getenv('BURST_WINDOW', '10'))
        },

        # Queue management
        'queue_manager': {
            'queue_type': os.getenv('QUEUE_TYPE', 'rabbitmq'),
            'rabbitmq_url': os.getenv('RABBITMQ_URL', 'amqp://guest:guest@rabbitmq-1:5672/'),
            'kafka_brokers': os.getenv('KAFKA_BROKERS', 'kafka-1:9092,kafka-2:9092,kafka-3:9092'),
            'redis_url': os.getenv('REDIS_URL', 'redis://redis:6379/12'),
            'max_queue_size': int(os.getenv('MAX_QUEUE_SIZE', '10000')),
            'enable_backpressure': os.getenv('ENABLE_BACKPRESSURE', 'true').lower() == 'true',
            'backpressure_threshold': int(os.getenv('BACKPRESSURE_THRESHOLD', '1000')),
            'backpressure_strategy': os.getenv('BACKPRESSURE_STRATEGY', 'drop_low_priority')
        },

        # Circuit breaker
        'circuit_breaker_threshold': int(os.getenv('CIRCUIT_BREAKER_THRESHOLD', '5')),
        'circuit_breaker_timeout': int(os.getenv('CIRCUIT_BREAKER_TIMEOUT', '60'))
    }

    return config


async def create_app():
    """Create and configure the application"""
    config = load_config()
    service = RequestMultiplexerService(config)
    await service.start()
    return service.app


def main():
    """Main entry point"""
    # Use uvloop for better performance
    if sys.platform != 'win32':
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received signal", signal=signum)
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run the service
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))

    logger.info("Starting RequestMultiplexer service", host=host, port=port)

    # Run with aiohttp
    web.run_app(
        create_app(),
        host=host,
        port=port,
        access_log=logger
    )


if __name__ == '__main__':
    main()