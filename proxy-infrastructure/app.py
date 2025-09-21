#!/usr/bin/env python3
"""
BEV Proxy Management Service Main Application
High-performance proxy pool management with geographic routing
"""

import asyncio
import os
import signal
import sys
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

# Add src directory to path
sys.path.insert(0, '/app/src')

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    from fastapi.responses import PlainTextResponse

    # Import our proxy management components
    from infrastructure.proxy_manager import (
        ProxyManager, ProxyEndpoint, ProxyType, ProxyRegion, ProxyStatus,
        LoadBalanceStrategy, create_proxy_manager
    )
    from infrastructure.geo_router import (
        GeoRouter, TargetRegion, ComplianceRegion, create_geo_router
    )

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PROXY_REQUESTS_TOTAL = Counter('proxy_requests_total', 'Total proxy requests', ['region', 'type', 'status'])
PROXY_RESPONSE_TIME = Histogram('proxy_response_time_seconds', 'Proxy response time', ['region', 'type'])
ACTIVE_PROXIES = Gauge('active_proxies_total', 'Number of active proxies', ['region', 'type', 'status'])
PROXY_POOL_SIZE = Gauge('proxy_pool_size_total', 'Total proxy pool size')

# Global application state
app_state = {
    'proxy_manager': None,
    'geo_router': None,
    'shutdown_event': asyncio.Event()
}

# Pydantic models for API
class ProxyRequest(BaseModel):
    """Proxy request model"""
    target: Optional[str] = Field(None, description="Target hostname/URL for geographic optimization")
    region_preference: Optional[str] = Field(None, description="Preferred proxy region")
    proxy_type_preference: Optional[str] = Field(None, description="Preferred proxy type")
    operation_type: str = Field("osint", description="Operation type for optimization")
    load_balance_strategy: Optional[str] = Field(None, description="Load balancing strategy")

class ProxyResponse(BaseModel):
    """Proxy response model"""
    proxy_id: str
    host: str
    port: int
    proxy_url: str
    http_proxy_url: str
    region: str
    proxy_type: str
    weight: float
    response_time: Optional[float]
    utilization: float

class ProxyReleaseRequest(BaseModel):
    """Proxy release request model"""
    proxy_id: str
    success: bool = True
    response_time: Optional[float] = None

class AddProxyRequest(BaseModel):
    """Add proxy request model"""
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None
    proxy_type: str = "datacenter"
    region: str = "global"
    weight: float = 1.0
    max_connections: int = 100
    rotation_interval: Optional[int] = None
    provider: Optional[str] = None
    provider_pool_id: Optional[str] = None
    cost_per_gb: Optional[float] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    proxy_pool_size: int
    healthy_proxies: int
    regions_available: List[str]
    uptime_seconds: float

# Application lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting BEV Proxy Management Service...")

    # Initialize services
    try:
        # Get configuration from environment
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/11')
        postgres_url = os.getenv('POSTGRES_URI', 'postgresql://localhost:5432/osint')
        geoip_db_path = os.getenv('GEOIP_DB_PATH', '/app/data/GeoLite2-City.mmdb')
        max_pool_size = int(os.getenv('MAX_POOL_SIZE', '10000'))

        # Initialize proxy manager
        logger.info("Initializing proxy manager...")
        app_state['proxy_manager'] = await create_proxy_manager(
            redis_url=redis_url,
            postgres_url=postgres_url,
            max_pool_size=max_pool_size
        )

        # Initialize geo router
        logger.info("Initializing geo router...")
        app_state['geo_router'] = await create_geo_router(
            redis_url=redis_url,
            postgres_url=postgres_url,
            geoip_db_path=geoip_db_path
        )

        # Start background tasks
        asyncio.create_task(update_metrics_task())

        logger.info("Proxy Management Service started successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise

    finally:
        # Cleanup
        logger.info("Shutting down Proxy Management Service...")
        app_state['shutdown_event'].set()

        if app_state['proxy_manager']:
            await app_state['proxy_manager'].shutdown()

        if app_state['geo_router']:
            await app_state['geo_router'].shutdown()

        logger.info("Service shutdown completed")

# Create FastAPI application
app = FastAPI(
    title="BEV Proxy Management Service",
    description="High-performance proxy pool management with geographic routing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
async def get_proxy_manager() -> ProxyManager:
    """Get proxy manager instance"""
    if not app_state['proxy_manager']:
        raise HTTPException(status_code=503, detail="Proxy manager not initialized")
    return app_state['proxy_manager']

async def get_geo_router() -> GeoRouter:
    """Get geo router instance"""
    if not app_state['geo_router']:
        raise HTTPException(status_code=503, detail="Geo router not initialized")
    return app_state['geo_router']

# API Routes

@app.get("/health", response_model=HealthResponse)
async def health_check(
    proxy_manager: ProxyManager = Depends(get_proxy_manager),
    geo_router: GeoRouter = Depends(get_geo_router)
):
    """Health check endpoint"""
    try:
        stats = await proxy_manager.get_proxy_statistics()

        return HealthResponse(
            status="healthy",
            version="1.0.0",
            proxy_pool_size=stats.get('pool_size', 0),
            healthy_proxies=stats.get('by_status', {}).get('healthy', 0),
            regions_available=list(stats.get('by_region', {}).keys()),
            uptime_seconds=0.0  # TODO: Track actual uptime
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

@app.post("/proxy/get", response_model=ProxyResponse)
async def get_proxy(
    request: ProxyRequest,
    proxy_manager: ProxyManager = Depends(get_proxy_manager),
    geo_router: GeoRouter = Depends(get_geo_router)
):
    """Get optimal proxy endpoint"""
    try:
        # Determine optimal regions if target is provided
        optimal_regions = None
        if request.target:
            optimal_regions = await geo_router.get_optimal_regions(
                request.target, request.operation_type
            )
            if optimal_regions:
                request.region_preference = optimal_regions[0].value

        # Parse preferences
        region_pref = None
        if request.region_preference:
            try:
                region_pref = ProxyRegion(request.region_preference)
            except ValueError:
                pass

        type_pref = None
        if request.proxy_type_preference:
            try:
                type_pref = ProxyType(request.proxy_type_preference)
            except ValueError:
                pass

        strategy = None
        if request.load_balance_strategy:
            try:
                strategy = LoadBalanceStrategy(request.load_balance_strategy)
            except ValueError:
                pass

        # Get proxy from manager
        proxy = await proxy_manager.get_proxy(
            region_preference=region_pref,
            proxy_type_preference=type_pref,
            load_balance_strategy=strategy
        )

        if not proxy:
            PROXY_REQUESTS_TOTAL.labels(
                region=request.region_preference or "any",
                type=request.proxy_type_preference or "any",
                status="no_proxy_available"
            ).inc()
            raise HTTPException(status_code=503, detail="No proxy available")

        # Record metrics
        PROXY_REQUESTS_TOTAL.labels(
            region=proxy.region.value,
            type=proxy.proxy_type.value,
            status="success"
        ).inc()

        # Generate proxy ID for tracking
        proxy_id = f"{proxy.host}:{proxy.port}"

        return ProxyResponse(
            proxy_id=proxy_id,
            host=proxy.host,
            port=proxy.port,
            proxy_url=proxy.proxy_url,
            http_proxy_url=proxy.http_proxy_url,
            region=proxy.region.value,
            proxy_type=proxy.proxy_type.value,
            weight=proxy.weight,
            response_time=proxy.response_time,
            utilization=proxy.utilization
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting proxy: {e}")
        PROXY_REQUESTS_TOTAL.labels(
            region=request.region_preference or "any",
            type=request.proxy_type_preference or "any",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/proxy/release")
async def release_proxy(
    request: ProxyReleaseRequest,
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """Release proxy endpoint after use"""
    try:
        # Parse proxy ID
        host, port = request.proxy_id.split(':')
        port = int(port)

        # Find proxy in manager
        all_proxies = await proxy_manager.get_all_proxies()
        proxy = None
        for p in all_proxies:
            if p.host == host and p.port == port:
                proxy = p
                break

        if not proxy:
            raise HTTPException(status_code=404, detail="Proxy not found")

        # Release proxy
        await proxy_manager.release_proxy(
            proxy, request.success, request.response_time
        )

        # Record metrics
        if request.response_time:
            PROXY_RESPONSE_TIME.labels(
                region=proxy.region.value,
                type=proxy.proxy_type.value
            ).observe(request.response_time)

        return {"status": "released", "proxy_id": request.proxy_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error releasing proxy: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/proxy/add")
async def add_proxy(
    request: AddProxyRequest,
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """Add new proxy endpoint to pool"""
    try:
        # Create proxy endpoint
        proxy = ProxyEndpoint(
            host=request.host,
            port=request.port,
            username=request.username,
            password=request.password,
            proxy_type=ProxyType(request.proxy_type),
            region=ProxyRegion(request.region),
            weight=request.weight,
            max_connections=request.max_connections,
            rotation_interval=request.rotation_interval,
            provider=request.provider,
            provider_pool_id=request.provider_pool_id,
            cost_per_gb=request.cost_per_gb
        )

        # Add to manager
        success = await proxy_manager.add_proxy(proxy)

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to add proxy (may already exist or pool is full)"
            )

        return {
            "status": "added",
            "proxy_id": f"{proxy.host}:{proxy.port}",
            "message": "Proxy added successfully"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    except Exception as e:
        logger.error(f"Error adding proxy: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.delete("/proxy/{host}/{port}")
async def remove_proxy(
    host: str,
    port: int,
    proxy_manager: ProxyManager = Depends(get_proxy_manager)
):
    """Remove proxy endpoint from pool"""
    try:
        success = await proxy_manager.remove_proxy(host, port)

        if not success:
            raise HTTPException(status_code=404, detail="Proxy not found")

        return {
            "status": "removed",
            "proxy_id": f"{host}:{port}",
            "message": "Proxy removed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing proxy: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/proxy/statistics")
async def get_statistics(
    proxy_manager: ProxyManager = Depends(get_proxy_manager),
    geo_router: GeoRouter = Depends(get_geo_router)
):
    """Get comprehensive proxy statistics"""
    try:
        proxy_stats = await proxy_manager.get_proxy_statistics()
        geo_stats = await geo_router.get_geo_statistics()

        return {
            "proxy_statistics": proxy_stats,
            "geographic_statistics": geo_stats,
            "timestamp": asyncio.get_event_loop().time()
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/proxy/regions")
async def get_optimal_regions(
    target: str,
    operation_type: str = "osint",
    geo_router: GeoRouter = Depends(get_geo_router)
):
    """Get optimal proxy regions for a target"""
    try:
        regions = await geo_router.get_optimal_regions(target, operation_type)
        compliance = await geo_router.get_compliance_requirements(target)

        return {
            "target": target,
            "operation_type": operation_type,
            "optimal_regions": [r.value for r in regions],
            "compliance_requirements": [c.value for c in compliance]
        }

    except Exception as e:
        logger.error(f"Error getting optimal regions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return PlainTextResponse(generate_latest())

# Background tasks

async def update_metrics_task():
    """Background task to update Prometheus metrics"""
    while not app_state['shutdown_event'].is_set():
        try:
            if app_state['proxy_manager']:
                stats = await app_state['proxy_manager'].get_proxy_statistics()

                # Update pool size
                PROXY_POOL_SIZE.set(stats.get('pool_size', 0))

                # Update proxy counts by status, region, and type
                ACTIVE_PROXIES.clear()

                # Status distribution
                for status, count in stats.get('by_status', {}).items():
                    ACTIVE_PROXIES.labels(region="all", type="all", status=status).set(count)

                # Region distribution
                for region, count in stats.get('by_region', {}).items():
                    ACTIVE_PROXIES.labels(region=region, type="all", status="all").set(count)

                # Type distribution
                for proxy_type, count in stats.get('by_type', {}).items():
                    ACTIVE_PROXIES.labels(region="all", type=proxy_type, status="all").set(count)

            # Wait before next update
            await asyncio.sleep(30)

        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            await asyncio.sleep(60)

# Signal handlers
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    app_state['shutdown_event'].set()

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Main application entry point
if __name__ == "__main__":
    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=int(os.getenv('PORT', '8000')),
        log_level=os.getenv('LOG_LEVEL', 'info').lower(),
        access_log=True,
        reload=False,
        workers=1  # Single worker for shared state
    )

    server = uvicorn.Server(config)

    logger.info("Starting BEV Proxy Management Service...")
    server.run()