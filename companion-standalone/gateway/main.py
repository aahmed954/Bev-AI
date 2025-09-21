"""
AI Companion Integration Gateway
Provides optional integration with core OSINT platform while maintaining standalone capability
"""

import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aioredis
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GATEWAY_MODE = os.getenv("GATEWAY_MODE", "standalone")
CORE_PLATFORM_DETECTION = os.getenv("CORE_PLATFORM_DETECTION", "auto") == "auto"
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
FAILOVER_ENABLED = os.getenv("FAILOVER_ENABLED", "true").lower() == "true"

# Core Platform Integration Settings
THANOS_HOST = os.getenv("THANOS_HOST", "172.21.0.10")
ORACLE1_HOST = os.getenv("ORACLE1_HOST", "172.21.0.20")
OSINT_API_ENDPOINT = os.getenv("OSINT_API_ENDPOINT")

# Graceful Degradation Settings
STANDALONE_FALLBACK = os.getenv("STANDALONE_FALLBACK", "true").lower() == "true"
INTEGRATION_TIMEOUT = int(os.getenv("INTEGRATION_TIMEOUT", "5"))

app = FastAPI(
    title="AI Companion Integration Gateway",
    description="Optional integration bridge between companion and core OSINT platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
integration_status = {
    "mode": GATEWAY_MODE,
    "core_platform_available": False,
    "thanos_available": False,
    "oracle1_available": False,
    "osint_api_available": False,
    "last_check": None,
    "connection_history": []
}

redis_client: Optional[aioredis.Redis] = None

# Data Models
class IntegrationStatus(BaseModel):
    mode: str
    core_platform_available: bool
    thanos_available: bool
    oracle1_available: bool
    osint_api_available: bool
    last_check: Optional[datetime]
    uptime_seconds: int

class OSINTRequest(BaseModel):
    query: str
    query_type: str
    parameters: Dict[str, Any] = {}
    priority: str = "normal"

class OSINTResponse(BaseModel):
    success: bool
    data: Dict[str, Any] = {}
    error: Optional[str] = None
    source: str
    timestamp: datetime

async def get_redis():
    """Get Redis connection"""
    global redis_client
    if redis_client is None:
        try:
            redis_client = aioredis.from_url("redis://companion-redis:6379/2")
            await redis_client.ping()
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            redis_client = None
    return redis_client

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def check_service_health(host: str, port: int, path: str = "/health") -> bool:
    """Check if a service is healthy"""
    try:
        async with httpx.AsyncClient(timeout=INTEGRATION_TIMEOUT) as client:
            response = await client.get(f"http://{host}:{port}{path}")
            return response.status_code == 200
    except Exception as e:
        logger.debug(f"Health check failed for {host}:{port} - {e}")
        return False

async def detect_core_platform():
    """Auto-detect availability of core platform services"""
    logger.info("Detecting core platform availability...")

    # Check THANOS availability
    thanos_available = await check_service_health(THANOS_HOST, 8000)
    integration_status["thanos_available"] = thanos_available

    # Check ORACLE1 availability
    oracle1_available = await check_service_health(ORACLE1_HOST, 8000)
    integration_status["oracle1_available"] = oracle1_available

    # Check OSINT API availability
    osint_available = False
    if OSINT_API_ENDPOINT:
        try:
            async with httpx.AsyncClient(timeout=INTEGRATION_TIMEOUT) as client:
                response = await client.get(f"{OSINT_API_ENDPOINT}/health")
                osint_available = response.status_code == 200
        except Exception:
            pass
    integration_status["osint_api_available"] = osint_available

    # Update overall status
    integration_status["core_platform_available"] = any([
        thanos_available, oracle1_available, osint_available
    ])
    integration_status["last_check"] = datetime.utcnow()

    # Log connection history
    integration_status["connection_history"].append({
        "timestamp": datetime.utcnow().isoformat(),
        "thanos": thanos_available,
        "oracle1": oracle1_available,
        "osint_api": osint_available
    })

    # Keep only last 100 history entries
    if len(integration_status["connection_history"]) > 100:
        integration_status["connection_history"] = integration_status["connection_history"][-100:]

    logger.info(f"Platform detection complete: THANOS={thanos_available}, ORACLE1={oracle1_available}, OSINT={osint_available}")

async def background_health_monitor():
    """Background task to monitor core platform health"""
    while True:
        try:
            if CORE_PLATFORM_DETECTION:
                await detect_core_platform()

            # Cache status in Redis
            redis = await get_redis()
            if redis:
                await redis.setex(
                    "companion:integration_status",
                    300,  # 5 minutes TTL
                    str(integration_status)
                )

        except Exception as e:
            logger.error(f"Health monitor error: {e}")

        await asyncio.sleep(HEALTH_CHECK_INTERVAL)

@app.on_event("startup")
async def startup_event():
    """Initialize gateway on startup"""
    logger.info("Starting AI Companion Integration Gateway...")

    # Initial platform detection
    if CORE_PLATFORM_DETECTION:
        await detect_core_platform()

    # Start background health monitoring
    asyncio.create_task(background_health_monitor())

    logger.info(f"Gateway started in {GATEWAY_MODE} mode")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mode": GATEWAY_MODE,
        "timestamp": datetime.utcnow().isoformat(),
        "integration_enabled": CORE_PLATFORM_DETECTION
    }

@app.get("/integration/status", response_model=IntegrationStatus)
async def get_integration_status():
    """Get current integration status"""
    uptime = int(time.time() - app.extra.get("start_time", time.time()))

    return IntegrationStatus(
        mode=integration_status["mode"],
        core_platform_available=integration_status["core_platform_available"],
        thanos_available=integration_status["thanos_available"],
        oracle1_available=integration_status["oracle1_available"],
        osint_api_available=integration_status["osint_api_available"],
        last_check=integration_status["last_check"],
        uptime_seconds=uptime
    )

@app.post("/integration/detect")
async def trigger_detection():
    """Manually trigger core platform detection"""
    if not CORE_PLATFORM_DETECTION:
        raise HTTPException(status_code=400, detail="Auto-detection is disabled")

    await detect_core_platform()
    return {"message": "Detection triggered", "status": integration_status}

@app.get("/integration/history")
async def get_connection_history():
    """Get connection history"""
    return {
        "history": integration_status["connection_history"],
        "total_entries": len(integration_status["connection_history"])
    }

@app.post("/osint/query", response_model=OSINTResponse)
async def osint_query(request: OSINTRequest):
    """
    Route OSINT query to available platform or provide standalone response
    """
    logger.info(f"OSINT query received: {request.query_type}")

    # Try to route to core platform if available
    if integration_status["core_platform_available"] and integration_status["osint_api_available"]:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{OSINT_API_ENDPOINT}/query",
                    json=request.dict()
                )

                if response.status_code == 200:
                    data = response.json()
                    return OSINTResponse(
                        success=True,
                        data=data,
                        source="core_platform",
                        timestamp=datetime.utcnow()
                    )
        except Exception as e:
            logger.warning(f"Core platform query failed: {e}")

    # Fallback to standalone mode
    if STANDALONE_FALLBACK:
        return await handle_standalone_query(request)
    else:
        raise HTTPException(
            status_code=503,
            detail="Core platform unavailable and standalone fallback disabled"
        )

async def handle_standalone_query(request: OSINTRequest) -> OSINTResponse:
    """Handle OSINT query in standalone mode"""
    logger.info(f"Handling query in standalone mode: {request.query_type}")

    # Basic standalone implementations
    standalone_handlers = {
        "domain": handle_domain_query,
        "ip": handle_ip_query,
        "email": handle_email_query,
        "phone": handle_phone_query,
        "hash": handle_hash_query,
        "crypto": handle_crypto_query
    }

    handler = standalone_handlers.get(request.query_type)
    if not handler:
        return OSINTResponse(
            success=False,
            error=f"Query type '{request.query_type}' not supported in standalone mode",
            source="standalone",
            timestamp=datetime.utcnow()
        )

    try:
        data = await handler(request.query, request.parameters)
        return OSINTResponse(
            success=True,
            data=data,
            source="standalone",
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        return OSINTResponse(
            success=False,
            error=str(e),
            source="standalone",
            timestamp=datetime.utcnow()
        )

# Standalone query handlers
async def handle_domain_query(query: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Basic domain analysis in standalone mode"""
    return {
        "query": query,
        "type": "domain",
        "note": "Limited standalone analysis - connect to core platform for full capabilities",
        "basic_info": {
            "domain": query,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

async def handle_ip_query(query: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Basic IP analysis in standalone mode"""
    return {
        "query": query,
        "type": "ip",
        "note": "Limited standalone analysis - connect to core platform for full capabilities",
        "basic_info": {
            "ip": query,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

async def handle_email_query(query: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Basic email analysis in standalone mode"""
    return {
        "query": query,
        "type": "email",
        "note": "Limited standalone analysis - connect to core platform for full capabilities",
        "basic_info": {
            "email": query,
            "domain": query.split("@")[-1] if "@" in query else "unknown",
            "timestamp": datetime.utcnow().isoformat()
        }
    }

async def handle_phone_query(query: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Basic phone analysis in standalone mode"""
    return {
        "query": query,
        "type": "phone",
        "note": "Limited standalone analysis - connect to core platform for full capabilities",
        "basic_info": {
            "phone": query,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

async def handle_hash_query(query: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Basic hash analysis in standalone mode"""
    return {
        "query": query,
        "type": "hash",
        "note": "Limited standalone analysis - connect to core platform for full capabilities",
        "basic_info": {
            "hash": query,
            "length": len(query),
            "probable_type": "md5" if len(query) == 32 else "sha1" if len(query) == 40 else "sha256" if len(query) == 64 else "unknown",
            "timestamp": datetime.utcnow().isoformat()
        }
    }

async def handle_crypto_query(query: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Basic cryptocurrency analysis in standalone mode"""
    return {
        "query": query,
        "type": "crypto",
        "note": "Limited standalone analysis - connect to core platform for full capabilities",
        "basic_info": {
            "address": query,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    uptime = int(time.time() - app.extra.get("start_time", time.time()))

    metrics = f"""# HELP companion_gateway_uptime_seconds Gateway uptime in seconds
# TYPE companion_gateway_uptime_seconds counter
companion_gateway_uptime_seconds {uptime}

# HELP companion_core_platform_available Core platform availability
# TYPE companion_core_platform_available gauge
companion_core_platform_available {int(integration_status["core_platform_available"])}

# HELP companion_thanos_available THANOS service availability
# TYPE companion_thanos_available gauge
companion_thanos_available {int(integration_status["thanos_available"])}

# HELP companion_oracle1_available ORACLE1 service availability
# TYPE companion_oracle1_available gauge
companion_oracle1_available {int(integration_status["oracle1_available"])}

# HELP companion_osint_api_available OSINT API availability
# TYPE companion_osint_api_available gauge
companion_osint_api_available {int(integration_status["osint_api_available"])}
"""

    return metrics

# Store start time for uptime calculation
app.extra = {"start_time": time.time()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)