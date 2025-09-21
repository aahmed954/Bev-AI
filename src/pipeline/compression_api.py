#!/usr/bin/env python3
"""
Context Compression API - FastAPI endpoints for the BEV OSINT Context Compression Engine
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import os

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

# Import our compression components
from .context_compressor import (
    ContextCompressor,
    CompressionConfig,
    CompressionStrategy,
    ContextCompressionResult,
    CompressionMetrics
)
from .quality_validator import ValidationConfig
import redis
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('compression_requests_total', 'Total compression requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('compression_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
COMPRESSION_RATIO = Histogram('compression_ratio', 'Compression ratio achieved')
INFORMATION_LOSS = Histogram('information_loss_score', 'Information loss score')
ACTIVE_COMPRESSIONS = Gauge('active_compressions', 'Currently active compression operations')
CACHE_HITS = Counter('cache_hits_total', 'Total cache hits')
CACHE_MISSES = Counter('cache_misses_total', 'Total cache misses')

# API Models
class CompressionRequest(BaseModel):
    """Request model for compression operations"""
    content: Union[str, List[str], List[Dict[str, Any]]]
    context_id: Optional[str] = None
    strategy: Optional[str] = "balanced"
    target_compression_ratio: Optional[float] = Field(None, ge=0.1, le=0.9)
    max_information_loss: Optional[float] = Field(None, ge=0.0, le=0.3)
    preserve_semantics: bool = True
    enable_caching: bool = True
    quality_validation: bool = True

    @validator('strategy')
    def validate_strategy(cls, v):
        valid_strategies = ['conservative', 'balanced', 'aggressive', 'semantic_only', 'entropy_only']
        if v not in valid_strategies:
            raise ValueError(f'Strategy must be one of: {valid_strategies}')
        return v

class CompressionResponse(BaseModel):
    """Response model for compression operations"""
    success: bool
    compressed_content: List[str]
    original_size: int
    compressed_size: int
    compression_ratio: float
    token_compression_ratio: float
    information_loss_score: float
    semantic_similarity_score: float
    coherence_score: float
    processing_time: float
    cache_key: Optional[str]
    metadata: Dict[str, Any]

class CompressionStatus(BaseModel):
    """Status model for compression operations"""
    operation_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    started_at: datetime
    estimated_completion: Optional[datetime]
    result: Optional[CompressionResponse]
    error: Optional[str]

class AnalysisRequest(BaseModel):
    """Request model for content analysis"""
    content: Union[str, List[str]]

class AnalysisResponse(BaseModel):
    """Response model for content analysis"""
    content_count: int
    total_size_bytes: int
    total_token_count: int
    average_content_length: float
    estimated_compression_potential: float
    content_diversity_score: float
    recommended_strategy: str

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    version: str
    uptime: float
    services: Dict[str, str]
    metrics: Dict[str, Any]

# Configuration
class CompressionAPIConfig:
    """Configuration for the compression API"""
    def __init__(self):
        # Load from environment variables
        self.host = os.getenv('API_HOST', '0.0.0.0')
        self.port = int(os.getenv('API_PORT', 8000))
        self.workers = int(os.getenv('API_WORKERS', 4))
        self.debug = os.getenv('DEBUG', 'false').lower() == 'true'
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

        # Redis configuration
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))

        # Compression configuration
        self.default_strategy = CompressionStrategy(os.getenv('COMPRESSION_STRATEGY', 'balanced'))
        self.default_compression_ratio = float(os.getenv('TARGET_COMPRESSION_RATIO', 0.4))
        self.default_max_loss = float(os.getenv('MAX_INFORMATION_LOSS', 0.05))

        # Infrastructure configuration
        self.infrastructure_config = {
            'redis_host': self.redis_host,
            'redis_port': self.redis_port,
            'mongodb_url': os.getenv('MONGODB_URL', 'mongodb://localhost:27017/'),
            'qdrant_host': os.getenv('QDRANT_HOST', 'localhost'),
            'qdrant_port': int(os.getenv('QDRANT_PORT', 6333)),
            'weaviate_host': os.getenv('WEAVIATE_HOST', 'localhost'),
            'weaviate_port': int(os.getenv('WEAVIATE_PORT', 8080))
        }

# Global configuration and compressor instance
config = CompressionAPIConfig()
compressor = None
redis_client = None
operation_status = {}  # In-memory status tracking
app_start_time = time.time()

# Initialize FastAPI app
app = FastAPI(
    title="BEV Context Compression API",
    description="Advanced context compression engine with semantic deduplication and entropy-based compression",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Middleware for metrics collection
@app.middleware("http")
async def metrics_middleware(request, call_next):
    method = request.method
    endpoint = request.url.path

    REQUEST_COUNT.labels(method=method, endpoint=endpoint).inc()

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

    return response

@app.on_event("startup")
async def startup_event():
    """Initialize the compression engine and connections"""
    global compressor, redis_client

    try:
        # Initialize Redis client
        redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )

        # Test Redis connection
        redis_client.ping()
        logger.info("Redis connection established")

        # Initialize compression engine
        compression_config = CompressionConfig(
            strategy=config.default_strategy,
            target_compression_ratio=config.default_compression_ratio,
            max_information_loss=config.default_max_loss,
            enable_caching=True,
            vector_db_integration=True,
            quality_validation=True
        )

        compressor = ContextCompressor(compression_config, config.infrastructure_config)
        logger.info("Context compression engine initialized")

    except Exception as e:
        logger.error(f"Failed to initialize compression API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources"""
    logger.info("Shutting down compression API")

# Helper functions
def generate_operation_id() -> str:
    """Generate unique operation ID"""
    return f"comp_{int(time.time() * 1000)}_{hash(time.time()) % 10000}"

async def get_compressor():
    """Dependency to get compressor instance"""
    if compressor is None:
        raise HTTPException(status_code=503, detail="Compression engine not initialized")
    return compressor

# API Endpoints

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test Redis connection
        redis_status = "healthy" if redis_client.ping() else "unhealthy"

        # Test compression engine
        compressor_status = "healthy" if compressor is not None else "unhealthy"

        # Get statistics
        stats = await compressor.get_statistics() if compressor else {}

        uptime = time.time() - app_start_time

        return HealthResponse(
            status="healthy" if all([redis_status == "healthy", compressor_status == "healthy"]) else "unhealthy",
            version="1.0.0",
            uptime=uptime,
            services={
                "redis": redis_status,
                "compressor": compressor_status
            },
            metrics={
                "total_compressions": stats.get('total_compressions', 0),
                "avg_compression_ratio": stats.get('avg_compression_ratio', 0.0),
                "cache_hit_rate": stats.get('cache_hit_rate', 0.0)
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Health check failed")

@app.post("/compress", response_model=CompressionResponse)
async def compress_content(
    request: CompressionRequest,
    background_tasks: BackgroundTasks,
    compressor: ContextCompressor = Depends(get_compressor)
):
    """Compress content using specified strategy"""

    ACTIVE_COMPRESSIONS.inc()

    try:
        # Parse strategy
        strategy = CompressionStrategy(request.strategy)

        # Override compression config if specified
        if request.target_compression_ratio or request.max_information_loss:
            compression_config = CompressionConfig(
                strategy=strategy,
                target_compression_ratio=request.target_compression_ratio or config.default_compression_ratio,
                max_information_loss=request.max_information_loss or config.default_max_loss,
                preserve_semantics=request.preserve_semantics,
                enable_caching=request.enable_caching,
                quality_validation=request.quality_validation
            )

            # Create new compressor instance with custom config
            custom_compressor = ContextCompressor(compression_config, config.infrastructure_config)
            result = await custom_compressor.compress_context(
                request.content,
                request.context_id,
                strategy
            )
        else:
            # Use default compressor
            result = await compressor.compress_context(
                request.content,
                request.context_id,
                strategy
            )

        # Record metrics
        COMPRESSION_RATIO.observe(result.metrics.compression_ratio)
        INFORMATION_LOSS.observe(result.metrics.information_loss_score)

        if result.cache_key:
            CACHE_HITS.inc()
        else:
            CACHE_MISSES.inc()

        response = CompressionResponse(
            success=True,
            compressed_content=result.compressed_content,
            original_size=result.metrics.original_size,
            compressed_size=result.metrics.compressed_size,
            compression_ratio=result.metrics.compression_ratio,
            token_compression_ratio=result.metrics.token_compression_ratio,
            information_loss_score=result.metrics.information_loss_score,
            semantic_similarity_score=result.metrics.semantic_similarity_score,
            coherence_score=result.metrics.coherence_score,
            processing_time=result.metrics.processing_time,
            cache_key=result.cache_key,
            metadata=result.compression_metadata
        )

        return response

    except Exception as e:
        logger.error(f"Compression failed: {e}")
        raise HTTPException(status_code=500, detail=f"Compression failed: {str(e)}")

    finally:
        ACTIVE_COMPRESSIONS.dec()

@app.post("/compress/async")
async def compress_content_async(
    request: CompressionRequest,
    background_tasks: BackgroundTasks,
    compressor: ContextCompressor = Depends(get_compressor)
):
    """Start asynchronous compression operation"""

    operation_id = generate_operation_id()

    # Initialize operation status
    operation_status[operation_id] = CompressionStatus(
        operation_id=operation_id,
        status="pending",
        progress=0.0,
        started_at=datetime.now(),
        estimated_completion=None,
        result=None,
        error=None
    )

    # Start background compression
    background_tasks.add_task(
        perform_async_compression,
        operation_id,
        request,
        compressor
    )

    return {"operation_id": operation_id, "status": "pending"}

async def perform_async_compression(
    operation_id: str,
    request: CompressionRequest,
    compressor: ContextCompressor
):
    """Perform compression in background"""

    try:
        # Update status
        operation_status[operation_id].status = "processing"
        operation_status[operation_id].progress = 0.1

        ACTIVE_COMPRESSIONS.inc()

        # Parse strategy
        strategy = CompressionStrategy(request.strategy)

        # Perform compression
        result = await compressor.compress_context(
            request.content,
            request.context_id,
            strategy
        )

        # Create response
        response = CompressionResponse(
            success=True,
            compressed_content=result.compressed_content,
            original_size=result.metrics.original_size,
            compressed_size=result.metrics.compressed_size,
            compression_ratio=result.metrics.compression_ratio,
            token_compression_ratio=result.metrics.token_compression_ratio,
            information_loss_score=result.metrics.information_loss_score,
            semantic_similarity_score=result.metrics.semantic_similarity_score,
            coherence_score=result.metrics.coherence_score,
            processing_time=result.metrics.processing_time,
            cache_key=result.cache_key,
            metadata=result.compression_metadata
        )

        # Update status
        operation_status[operation_id].status = "completed"
        operation_status[operation_id].progress = 1.0
        operation_status[operation_id].result = response

        # Record metrics
        COMPRESSION_RATIO.observe(result.metrics.compression_ratio)
        INFORMATION_LOSS.observe(result.metrics.information_loss_score)

    except Exception as e:
        logger.error(f"Async compression failed: {e}")
        operation_status[operation_id].status = "failed"
        operation_status[operation_id].error = str(e)

    finally:
        ACTIVE_COMPRESSIONS.dec()

@app.get("/compress/status/{operation_id}", response_model=CompressionStatus)
async def get_compression_status(operation_id: str):
    """Get status of asynchronous compression operation"""

    if operation_id not in operation_status:
        raise HTTPException(status_code=404, detail="Operation not found")

    return operation_status[operation_id]

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_content(
    request: AnalysisRequest,
    compressor: ContextCompressor = Depends(get_compressor)
):
    """Analyze content complexity and compression potential"""

    try:
        # Normalize content to list
        if isinstance(request.content, str):
            content_list = [request.content]
        else:
            content_list = request.content

        # Perform analysis
        analysis = await compressor.analyze_content_complexity(content_list)

        return AnalysisResponse(**analysis)

    except Exception as e:
        logger.error(f"Content analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/decompress")
async def decompress_content(
    compression_result: Dict[str, Any],
    compressor: ContextCompressor = Depends(get_compressor)
):
    """Decompress previously compressed content"""

    try:
        # Reconstruct compression result object
        # This is a simplified version - in practice, you'd want to properly
        # deserialize the compression result from the stored format

        decompressed = await compressor.decompress_context(compression_result)

        return {
            "success": True,
            "decompressed_content": decompressed,
            "content_count": len(decompressed)
        }

    except Exception as e:
        logger.error(f"Decompression failed: {e}")
        raise HTTPException(status_code=500, detail=f"Decompression failed: {str(e)}")

@app.get("/statistics")
async def get_statistics(compressor: ContextCompressor = Depends(get_compressor)):
    """Get compression engine statistics"""

    try:
        stats = await compressor.get_statistics()

        # Add API-specific statistics
        api_stats = {
            "active_operations": len([s for s in operation_status.values() if s.status == "processing"]),
            "total_operations": len(operation_status),
            "uptime": time.time() - app_start_time
        }

        return {
            "compression_stats": stats,
            "api_stats": api_stats
        }

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(
        generate_latest(prometheus_client.REGISTRY),
        media_type="text/plain"
    )

@app.get("/config")
async def get_configuration():
    """Get current compression configuration"""

    return {
        "api_config": {
            "host": config.host,
            "port": config.port,
            "workers": config.workers,
            "debug": config.debug
        },
        "compression_config": {
            "default_strategy": config.default_strategy.value,
            "default_compression_ratio": config.default_compression_ratio,
            "default_max_loss": config.default_max_loss
        },
        "infrastructure_config": config.infrastructure_config
    }

@app.post("/config/validate")
async def validate_configuration(config_data: Dict[str, Any]):
    """Validate compression configuration"""

    try:
        # Validate compression strategy
        if 'strategy' in config_data:
            CompressionStrategy(config_data['strategy'])

        # Validate numeric ranges
        if 'target_compression_ratio' in config_data:
            ratio = config_data['target_compression_ratio']
            if not 0.1 <= ratio <= 0.9:
                raise ValueError("Compression ratio must be between 0.1 and 0.9")

        if 'max_information_loss' in config_data:
            loss = config_data['max_information_loss']
            if not 0.0 <= loss <= 0.3:
                raise ValueError("Max information loss must be between 0.0 and 0.3")

        return {"valid": True, "message": "Configuration is valid"}

    except Exception as e:
        return {"valid": False, "message": str(e)}

# Cleanup task for operation status
@app.on_event("startup")
async def start_cleanup_task():
    """Start background task to cleanup old operation statuses"""

    async def cleanup_operations():
        while True:
            try:
                # Remove operations older than 1 hour
                cutoff_time = datetime.now() - timedelta(hours=1)

                to_remove = [
                    op_id for op_id, status in operation_status.items()
                    if status.started_at < cutoff_time and status.status in ["completed", "failed"]
                ]

                for op_id in to_remove:
                    del operation_status[op_id]

                if to_remove:
                    logger.info(f"Cleaned up {len(to_remove)} old operations")

                # Wait 10 minutes before next cleanup
                await asyncio.sleep(600)

            except Exception as e:
                logger.error(f"Cleanup task failed: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute

    # Start cleanup task
    asyncio.create_task(cleanup_operations())

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid input", "detail": str(exc)}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "compression_api:app",
        host=config.host,
        port=config.port,
        workers=config.workers,
        log_level=config.log_level.lower(),
        reload=config.debug
    )