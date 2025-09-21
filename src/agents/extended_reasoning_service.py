"""
Extended Reasoning Service
FastAPI wrapper for the Extended Reasoning Pipeline
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.responses import Response

from extended_reasoning import ExtendedReasoningPipeline
from research_workflow import ResearchWorkflowEngine
from counterfactual_analyzer import CounterfactualAnalyzer
from knowledge_synthesizer import KnowledgeSynthesizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('extended_reasoning_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('extended_reasoning_request_duration_seconds', 'Request duration')
ACTIVE_PROCESSES = Gauge('extended_reasoning_active_processes', 'Active reasoning processes')
TOKEN_PROCESSED = Counter('extended_reasoning_tokens_processed_total', 'Total tokens processed')
ERROR_COUNT = Counter('extended_reasoning_errors_total', 'Total errors', ['error_type'])

# Global instances
pipeline: Optional[ExtendedReasoningPipeline] = None
active_processes: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global pipeline

    # Startup
    logger.info("Starting Extended Reasoning Service")

    try:
        # Initialize configuration
        config = {
            'compression_endpoint': 'http://172.30.0.43:8000',
            'vector_db_endpoint': 'http://172.30.0.44:8000',
            'max_tokens': 100000,
            'chunk_size': 8000,
            'overlap_ratio': 0.1,
            'min_confidence': 0.6,
            'max_processing_time': 600,
            'entity_confidence_threshold': 0.6,
            'relationship_confidence_threshold': 0.5,
            'pattern_significance_threshold': 0.7,
            'max_hypotheses': 10,
            'min_hypothesis_strength': 0.3,
            'evidence_weight_threshold': 0.6,
            'relationship_variation_factor': 0.8,
            'min_cluster_size': 3,
            'max_clusters': 10,
            'centrality_threshold': 0.1,
            'evidence_convergence_threshold': 0.7,
            'min_causal_strength': 0.6
        }

        # Initialize pipeline
        pipeline = ExtendedReasoningPipeline(config)

        logger.info("Extended Reasoning Service started successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to start Extended Reasoning Service: {str(e)}")
        raise

    # Shutdown
    logger.info("Shutting down Extended Reasoning Service")

    # Clean up active processes
    for process_id in list(active_processes.keys()):
        active_processes.pop(process_id, None)
        ACTIVE_PROCESSES.dec()

# FastAPI app
app = FastAPI(
    title="Extended Reasoning Pipeline",
    description="Advanced reasoning system for OSINT analysis with 100K+ token support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ReasoningRequest(BaseModel):
    content: str = Field(..., description="Content to analyze", min_length=10)
    context_id: Optional[str] = Field(None, description="Optional context identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    processing_options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing options")

class ReasoningResponse(BaseModel):
    process_id: str
    status: str
    message: str
    estimated_completion_time: Optional[float] = None

class ProcessStatus(BaseModel):
    process_id: str
    status: str
    current_phase: Optional[str] = None
    progress: float
    estimated_completion: Optional[float] = None
    error_message: Optional[str] = None

class ReasoningResult(BaseModel):
    process_id: str
    context_id: str
    final_synthesis: str
    confidence_score: float
    uncertainty_factors: List[str]
    phase_outputs: Dict[str, Any]
    processing_time: float
    token_efficiency: float
    verification_results: Dict[str, Any]
    counterfactual_analysis: Dict[str, Any]
    recommendations: List[str]
    network_analysis: Dict[str, Any]
    key_insights: List[Dict[str, Any]]
    knowledge_clusters: List[Dict[str, Any]]
    causal_chains: List[Dict[str, Any]]

# Dependency for metrics
def track_request(endpoint: str):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            REQUEST_COUNT.labels(method='POST', endpoint=endpoint).inc()

            with REQUEST_DURATION.time():
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    ERROR_COUNT.labels(error_type=type(e).__name__).inc()
                    raise
        return wrapper
    return decorator

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        health_status = await pipeline.health_check()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "pipeline_health": health_status,
            "active_processes": len(active_processes),
            "service_info": {
                "name": "Extended Reasoning Pipeline",
                "version": "1.0.0"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

# Main reasoning endpoint
@app.post("/analyze", response_model=ReasoningResponse)
@track_request("analyze")
async def analyze_content(
    request: ReasoningRequest,
    background_tasks: BackgroundTasks
):
    """
    Start extended reasoning analysis
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    process_id = str(uuid.uuid4())
    context_id = request.context_id or f"ctx_{int(time.time())}"

    # Estimate token count
    estimated_tokens = len(request.content.split()) * 1.3
    TOKEN_PROCESSED.inc(estimated_tokens)

    # Estimate completion time based on token count
    estimated_time = min(600, max(60, estimated_tokens / 1000 * 10))  # Rough estimate

    # Initialize process tracking
    active_processes[process_id] = {
        'status': 'queued',
        'context_id': context_id,
        'start_time': time.time(),
        'estimated_completion': time.time() + estimated_time,
        'current_phase': None,
        'progress': 0.0,
        'result': None,
        'error': None
    }

    ACTIVE_PROCESSES.inc()

    # Start background processing
    background_tasks.add_task(
        process_reasoning,
        process_id,
        request.content,
        context_id,
        request.metadata,
        request.processing_options
    )

    return ReasoningResponse(
        process_id=process_id,
        status="queued",
        message="Analysis started successfully",
        estimated_completion_time=estimated_time
    )

async def process_reasoning(
    process_id: str,
    content: str,
    context_id: str,
    metadata: Dict[str, Any],
    processing_options: Dict[str, Any]
):
    """Background reasoning process"""
    try:
        # Update status
        active_processes[process_id]['status'] = 'processing'
        active_processes[process_id]['current_phase'] = 'initialization'
        active_processes[process_id]['progress'] = 0.1

        logger.info(f"Starting reasoning process {process_id} for context {context_id}")

        # Process content through pipeline
        result = await pipeline.process_context(
            content=content,
            context_id=context_id,
            metadata=metadata
        )

        # Update final status
        active_processes[process_id]['status'] = 'completed'
        active_processes[process_id]['current_phase'] = 'completed'
        active_processes[process_id]['progress'] = 1.0
        active_processes[process_id]['result'] = result

        logger.info(f"Completed reasoning process {process_id}")

    except Exception as e:
        logger.error(f"Error in reasoning process {process_id}: {str(e)}")

        active_processes[process_id]['status'] = 'failed'
        active_processes[process_id]['error'] = str(e)
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()

    finally:
        ACTIVE_PROCESSES.dec()

# Status check endpoint
@app.get("/status/{process_id}", response_model=ProcessStatus)
async def get_process_status(process_id: str):
    """Get processing status"""
    if process_id not in active_processes:
        raise HTTPException(status_code=404, detail="Process not found")

    process_info = active_processes[process_id]

    return ProcessStatus(
        process_id=process_id,
        status=process_info['status'],
        current_phase=process_info.get('current_phase'),
        progress=process_info['progress'],
        estimated_completion=process_info.get('estimated_completion'),
        error_message=process_info.get('error')
    )

# Results endpoint
@app.get("/result/{process_id}", response_model=ReasoningResult)
async def get_reasoning_result(process_id: str):
    """Get reasoning results"""
    if process_id not in active_processes:
        raise HTTPException(status_code=404, detail="Process not found")

    process_info = active_processes[process_id]

    if process_info['status'] == 'failed':
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {process_info.get('error', 'Unknown error')}"
        )

    if process_info['status'] != 'completed':
        raise HTTPException(
            status_code=202,
            detail=f"Processing not complete. Status: {process_info['status']}"
        )

    result = process_info['result']
    if not result:
        raise HTTPException(status_code=500, detail="No result available")

    # Convert result to response model
    return ReasoningResult(
        process_id=process_id,
        context_id=result.context_id,
        final_synthesis=result.final_synthesis,
        confidence_score=result.confidence_score,
        uncertainty_factors=result.uncertainty_factors,
        phase_outputs=result.phase_outputs,
        processing_time=result.processing_time,
        token_efficiency=result.token_efficiency,
        verification_results=result.verification_results,
        counterfactual_analysis=result.counterfactual_analysis,
        recommendations=result.recommendations,
        network_analysis=result.phase_outputs.get('synthesis', {}).get('network_analysis', {}),
        key_insights=result.phase_outputs.get('synthesis', {}).get('key_insights', []),
        knowledge_clusters=result.phase_outputs.get('synthesis', {}).get('knowledge_clusters', []),
        causal_chains=result.phase_outputs.get('synthesis', {}).get('causal_chains', [])
    )

# List active processes
@app.get("/processes")
async def list_processes():
    """List all active processes"""
    return {
        "active_processes": len(active_processes),
        "processes": [
            {
                "process_id": pid,
                "status": info['status'],
                "start_time": info['start_time'],
                "context_id": info['context_id']
            }
            for pid, info in active_processes.items()
        ]
    }

# Cancel process
@app.delete("/process/{process_id}")
async def cancel_process(process_id: str):
    """Cancel a processing task"""
    if process_id not in active_processes:
        raise HTTPException(status_code=404, detail="Process not found")

    process_info = active_processes[process_id]

    if process_info['status'] in ['completed', 'failed']:
        raise HTTPException(status_code=400, detail="Process already finished")

    # Mark as cancelled
    active_processes[process_id]['status'] = 'cancelled'
    active_processes[process_id]['error'] = 'Cancelled by user'

    return {"message": f"Process {process_id} cancelled"}

# Performance metrics
@app.get("/performance")
async def get_performance_metrics():
    """Get performance metrics"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        metrics = await pipeline.get_processing_metrics()
        return {
            "pipeline_metrics": metrics,
            "service_metrics": {
                "active_processes": len(active_processes),
                "total_processes": len(active_processes),  # Would track historically
                "average_processing_time": 0.0,  # Would calculate from history
                "success_rate": 0.95  # Would calculate from history
            }
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

# Cleanup completed processes periodically
@app.on_event("startup")
async def start_cleanup_task():
    """Start periodic cleanup of completed processes"""
    async def cleanup_completed_processes():
        while True:
            try:
                current_time = time.time()
                to_remove = []

                for process_id, info in active_processes.items():
                    # Remove completed processes older than 1 hour
                    if (info['status'] in ['completed', 'failed', 'cancelled'] and
                        current_time - info['start_time'] > 3600):
                        to_remove.append(process_id)

                for process_id in to_remove:
                    active_processes.pop(process_id, None)
                    logger.info(f"Cleaned up completed process {process_id}")

                await asyncio.sleep(300)  # Clean up every 5 minutes

            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    asyncio.create_task(cleanup_completed_processes())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)