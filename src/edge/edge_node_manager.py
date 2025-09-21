"""
Edge Node Manager for BEV OSINT Framework

Manages individual edge computing nodes with local model deployment,
inference capability, and health monitoring.
"""

import asyncio
import time
import logging
import json
import psutil
import GPUtil
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import aiohttp
from aiohttp import web
import asyncpg
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import nvidia_ml_py3 as nvml

class NodeStatus(Enum):
    """Edge node status states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

class ModelType(Enum):
    """Supported model types for edge inference"""
    LLAMA_3_8B = "llama-3-8b"
    MISTRAL_7B = "mistral-7b"
    PHI_3_MINI = "phi-3-mini"
    CUSTOM = "custom"

@dataclass
class NodeConfiguration:
    """Configuration for an edge node"""
    node_id: str
    region: str
    ip_address: str
    port: int
    max_concurrent_requests: int
    memory_limit_gb: int
    gpu_memory_limit_gb: int
    model_cache_size: int
    health_check_interval: int
    model_sync_interval: int
    log_level: str

@dataclass
class ModelDeployment:
    """Model deployment configuration"""
    model_name: str
    model_type: ModelType
    model_path: str
    tokenizer_path: str
    max_tokens: int
    temperature: float
    top_p: float
    memory_usage_mb: int
    load_time_seconds: float
    is_loaded: bool
    last_used: datetime

@dataclass
class InferenceRequest:
    """Inference request for edge node"""
    request_id: str
    model_name: str
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    stop_sequences: List[str]
    stream: bool
    timeout_seconds: int

@dataclass
class InferenceResponse:
    """Inference response from edge node"""
    request_id: str
    model_name: str
    generated_text: str
    tokens_generated: int
    processing_time_ms: int
    memory_used_mb: int
    success: bool
    error_message: Optional[str] = None

class EdgeNodeManager:
    """
    Edge Node Manager

    Manages individual edge computing node with local model deployment,
    inference processing, and resource monitoring.
    """

    def __init__(self, config: NodeConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.status = NodeStatus.INITIALIZING
        self.models: Dict[str, ModelDeployment] = {}
        self.loaded_models: Dict[str, Any] = {}
        self.model_pipelines: Dict[str, Any] = {}
        self.request_queue = asyncio.Queue()
        self.active_requests: Dict[str, InferenceRequest] = {}

        # Performance metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.average_processing_time = 0.0
        self.current_memory_usage = 0.0
        self.current_gpu_usage = 0.0

        # Prometheus metrics
        self.registry = CollectorRegistry()
        self.request_counter = Counter(
            'edge_node_requests_total',
            'Total requests processed',
            ['model', 'success'],
            registry=self.registry
        )
        self.processing_time_histogram = Histogram(
            'edge_node_processing_time_seconds',
            'Request processing time',
            ['model'],
            registry=self.registry
        )
        self.memory_usage_gauge = Gauge(
            'edge_node_memory_usage_bytes',
            'Current memory usage',
            registry=self.registry
        )
        self.gpu_usage_gauge = Gauge(
            'edge_node_gpu_usage_percent',
            'Current GPU usage',
            registry=self.registry
        )
        self.model_load_counter = Counter(
            'edge_node_model_loads_total',
            'Total model loads',
            ['model'],
            registry=self.registry
        )

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # Initialize NVIDIA monitoring
        try:
            nvml.nvmlInit()
            self.gpu_available = True
            self.gpu_count = nvml.nvmlDeviceGetCount()
        except Exception:
            self.gpu_available = False
            self.gpu_count = 0

    async def initialize(self):
        """Initialize edge node manager"""
        try:
            self.logger.info(f"Initializing Edge Node {self.config.node_id}")

            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

            # Initialize web application
            await self._setup_web_server()

            # Load default models
            await self._load_default_models()

            # Start background tasks
            asyncio.create_task(self._health_monitor())
            asyncio.create_task(self._request_processor())
            asyncio.create_task(self._resource_monitor())

            self.status = NodeStatus.ACTIVE
            self.logger.info(f"Edge Node {self.config.node_id} initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize edge node: {e}")
            self.status = NodeStatus.OFFLINE
            raise

    async def _setup_web_server(self):
        """Setup web server for node API"""
        self.app = web.Application()

        # Add routes
        self.app.router.add_get('/health', self._handle_health)
        self.app.router.add_get('/status', self._handle_status)
        self.app.router.add_get('/models', self._handle_models)
        self.app.router.add_post('/inference', self._handle_inference)
        self.app.router.add_post('/load_model', self._handle_load_model)
        self.app.router.add_post('/unload_model', self._handle_unload_model)
        self.app.router.add_get('/metrics', self._handle_metrics)

        # Setup and start runner
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(
            self.runner,
            self.config.ip_address,
            self.config.port
        )
        await self.site.start()

        self.logger.info(f"Web server started on {self.config.ip_address}:{self.config.port}")

    async def _load_default_models(self):
        """Load default models for the edge node"""
        default_models = [
            {
                "name": "llama-3-8b",
                "type": ModelType.LLAMA_3_8B,
                "path": "microsoft/Llama-3-8b-chat-hf",
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9
            },
            {
                "name": "phi-3-mini",
                "type": ModelType.PHI_3_MINI,
                "path": "microsoft/Phi-3-mini-4k-instruct",
                "max_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9
            }
        ]

        for model_config in default_models:
            try:
                await self._load_model(
                    model_config["name"],
                    model_config["type"],
                    model_config["path"],
                    model_config["max_tokens"],
                    model_config["temperature"],
                    model_config["top_p"]
                )
            except Exception as e:
                self.logger.warning(f"Failed to load default model {model_config['name']}: {e}")

    async def _load_model(self, name: str, model_type: ModelType, model_path: str,
                         max_tokens: int, temperature: float, top_p: float) -> bool:
        """Load a model for inference"""
        try:
            self.logger.info(f"Loading model {name} from {model_path}")
            start_time = time.time()

            # Check if model is already loaded
            if name in self.loaded_models:
                self.logger.info(f"Model {name} already loaded")
                return True

            # Check memory availability
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            if available_memory < 4.0:  # Require at least 4GB free
                raise Exception(f"Insufficient memory: {available_memory:.1f}GB available")

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

            # Configure model loading based on available resources
            device = "cuda" if torch.cuda.is_available() and self.gpu_available else "cpu"

            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32
                )
                model = model.to(device)

            # Create inference pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )

            load_time = time.time() - start_time

            # Calculate memory usage
            memory_usage = self._get_model_memory_usage(model)

            # Store model deployment info
            deployment = ModelDeployment(
                model_name=name,
                model_type=model_type,
                model_path=model_path,
                tokenizer_path=model_path,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                memory_usage_mb=memory_usage,
                load_time_seconds=load_time,
                is_loaded=True,
                last_used=datetime.utcnow()
            )

            self.models[name] = deployment
            self.loaded_models[name] = model
            self.model_pipelines[name] = pipe

            # Update metrics
            self.model_load_counter.labels(model=name).inc()

            self.logger.info(f"Model {name} loaded successfully in {load_time:.2f}s, using {memory_usage}MB")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model {name}: {e}")
            return False

    def _get_model_memory_usage(self, model) -> int:
        """Get memory usage of a loaded model in MB"""
        try:
            if hasattr(model, 'get_memory_footprint'):
                return model.get_memory_footprint() // (1024 * 1024)
            else:
                # Estimate based on parameter count
                total_params = sum(p.numel() for p in model.parameters())
                # Rough estimate: 4 bytes per parameter (float32) or 2 bytes (float16)
                bytes_per_param = 2 if hasattr(model, 'dtype') and model.dtype == torch.float16 else 4
                return (total_params * bytes_per_param) // (1024 * 1024)
        except Exception:
            return 0  # Unknown

    async def _unload_model(self, name: str) -> bool:
        """Unload a model to free memory"""
        try:
            if name not in self.loaded_models:
                return True

            # Remove from memory
            del self.loaded_models[name]
            del self.model_pipelines[name]

            if name in self.models:
                self.models[name].is_loaded = False

            # Force garbage collection
            import gc
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info(f"Model {name} unloaded successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to unload model {name}: {e}")
            return False

    async def process_inference_request(self, request: InferenceRequest) -> InferenceResponse:
        """Process an inference request"""
        start_time = time.time()

        try:
            # Check if model is loaded
            if request.model_name not in self.model_pipelines:
                raise Exception(f"Model {request.model_name} not loaded")

            # Get model pipeline
            pipe = self.model_pipelines[request.model_name]
            model_deployment = self.models[request.model_name]

            # Update last used timestamp
            model_deployment.last_used = datetime.utcnow()

            # Prepare generation parameters
            generation_params = {
                "max_new_tokens": min(request.max_tokens, model_deployment.max_tokens),
                "temperature": request.temperature,
                "top_p": request.top_p,
                "do_sample": True,
                "pad_token_id": pipe.tokenizer.eos_token_id
            }

            if request.stop_sequences:
                generation_params["stop_sequences"] = request.stop_sequences

            # Generate response
            self.logger.debug(f"Processing inference request {request.request_id}")

            result = pipe(
                request.prompt,
                **generation_params
            )

            generated_text = result[0]["generated_text"]

            # Remove the original prompt from the generated text
            if generated_text.startswith(request.prompt):
                generated_text = generated_text[len(request.prompt):].strip()

            processing_time = (time.time() - start_time) * 1000

            # Count tokens (approximate)
            tokens_generated = len(generated_text.split())

            # Get current memory usage
            memory_used = psutil.virtual_memory().used // (1024 * 1024)

            response = InferenceResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                generated_text=generated_text,
                tokens_generated=tokens_generated,
                processing_time_ms=int(processing_time),
                memory_used_mb=memory_used,
                success=True
            )

            # Update metrics
            self.request_counter.labels(model=request.model_name, success="true").inc()
            self.processing_time_histogram.labels(model=request.model_name).observe(processing_time / 1000.0)

            self.total_requests += 1
            self.successful_requests += 1
            self.average_processing_time = (
                (self.average_processing_time * (self.successful_requests - 1) + processing_time) /
                self.successful_requests
            )

            return response

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000

            self.logger.error(f"Inference request {request.request_id} failed: {e}")

            response = InferenceResponse(
                request_id=request.request_id,
                model_name=request.model_name,
                generated_text="",
                tokens_generated=0,
                processing_time_ms=int(processing_time),
                memory_used_mb=0,
                success=False,
                error_message=str(e)
            )

            # Update error metrics
            self.request_counter.labels(model=request.model_name, success="false").inc()
            self.failed_requests += 1
            self.total_requests += 1

            return response

    async def _health_monitor(self):
        """Monitor node health continuously"""
        while True:
            try:
                await self._check_health()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(30)

    async def _check_health(self):
        """Check node health status"""
        try:
            # Check memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Check CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)

            # Check GPU usage (if available)
            gpu_usage = 0.0
            if self.gpu_available:
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_usage = gpu_util.gpu
                except Exception:
                    pass

            # Update metrics
            self.current_memory_usage = memory_usage_percent
            self.current_gpu_usage = gpu_usage
            self.memory_usage_gauge.set(memory.used)
            self.gpu_usage_gauge.set(gpu_usage)

            # Determine status based on resource usage
            if memory_usage_percent > 90 or cpu_usage > 95:
                if self.status == NodeStatus.ACTIVE:
                    self.status = NodeStatus.DEGRADED
                    self.logger.warning(f"Node status degraded: Memory={memory_usage_percent:.1f}%, CPU={cpu_usage:.1f}%")
            elif memory_usage_percent < 80 and cpu_usage < 85:
                if self.status == NodeStatus.DEGRADED:
                    self.status = NodeStatus.ACTIVE
                    self.logger.info("Node status restored to active")

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            if self.status not in [NodeStatus.MAINTENANCE, NodeStatus.OFFLINE]:
                self.status = NodeStatus.DEGRADED

    async def _request_processor(self):
        """Process inference requests from queue"""
        while True:
            try:
                # Wait for request with timeout
                request = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)

                # Add to active requests
                self.active_requests[request.request_id] = request

                try:
                    response = await self.process_inference_request(request)
                    # Response would be sent back via HTTP response
                finally:
                    # Remove from active requests
                    if request.request_id in self.active_requests:
                        del self.active_requests[request.request_id]

                self.request_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Request processing error: {e}")

    async def _resource_monitor(self):
        """Monitor resource usage and optimize"""
        while True:
            try:
                await self._optimize_resource_usage()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(60)

    async def _optimize_resource_usage(self):
        """Optimize resource usage by managing loaded models"""
        # Check if memory usage is high
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            # Find least recently used models to unload
            lru_models = sorted(
                [(name, deployment.last_used) for name, deployment in self.models.items() if deployment.is_loaded],
                key=lambda x: x[1]
            )

            # Unload oldest models until memory usage is acceptable
            for model_name, _ in lru_models:
                if memory.percent < 75:
                    break
                await self._unload_model(model_name)
                memory = psutil.virtual_memory()

    # HTTP Handlers
    async def _handle_health(self, request):
        """Health check endpoint"""
        health_status = {
            "status": self.status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "node_id": self.config.node_id,
            "region": self.config.region,
            "uptime_seconds": time.time() - getattr(self, 'start_time', time.time()),
            "memory_usage_percent": self.current_memory_usage,
            "gpu_usage_percent": self.current_gpu_usage,
            "loaded_models": len([m for m in self.models.values() if m.is_loaded]),
            "active_requests": len(self.active_requests),
            "total_requests": self.total_requests,
            "success_rate": self.successful_requests / max(self.total_requests, 1) * 100
        }

        status_code = 200 if self.status in [NodeStatus.ACTIVE, NodeStatus.DEGRADED] else 503
        return web.json_response(health_status, status=status_code)

    async def _handle_status(self, request):
        """Detailed status endpoint"""
        status = {
            "node_id": self.config.node_id,
            "region": self.config.region,
            "status": self.status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "models": {
                name: {
                    "type": deployment.model_type.value,
                    "is_loaded": deployment.is_loaded,
                    "memory_usage_mb": deployment.memory_usage_mb,
                    "last_used": deployment.last_used.isoformat(),
                    "max_tokens": deployment.max_tokens
                }
                for name, deployment in self.models.items()
            },
            "resources": {
                "memory_usage_percent": self.current_memory_usage,
                "gpu_usage_percent": self.current_gpu_usage,
                "gpu_available": self.gpu_available,
                "gpu_count": self.gpu_count
            },
            "performance": {
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "average_processing_time_ms": self.average_processing_time,
                "active_requests": len(self.active_requests)
            }
        }
        return web.json_response(status)

    async def _handle_models(self, request):
        """List available models endpoint"""
        models_info = {
            name: {
                "type": deployment.model_type.value,
                "is_loaded": deployment.is_loaded,
                "memory_usage_mb": deployment.memory_usage_mb,
                "load_time_seconds": deployment.load_time_seconds,
                "last_used": deployment.last_used.isoformat(),
                "max_tokens": deployment.max_tokens,
                "temperature": deployment.temperature,
                "top_p": deployment.top_p
            }
            for name, deployment in self.models.items()
        }
        return web.json_response(models_info)

    async def _handle_inference(self, request):
        """Handle inference request"""
        try:
            data = await request.json()

            inference_request = InferenceRequest(
                request_id=data.get("request_id", f"req_{int(time.time() * 1000)}"),
                model_name=data.get("model", "llama-3-8b"),
                prompt=data["prompt"],
                max_tokens=data.get("max_tokens", 1024),
                temperature=data.get("temperature", 0.7),
                top_p=data.get("top_p", 0.9),
                stop_sequences=data.get("stop_sequences", []),
                stream=data.get("stream", False),
                timeout_seconds=data.get("timeout", 30)
            )

            response = await self.process_inference_request(inference_request)

            return web.json_response({
                "request_id": response.request_id,
                "success": response.success,
                "generated_text": response.generated_text,
                "tokens_generated": response.tokens_generated,
                "processing_time_ms": response.processing_time_ms,
                "model_used": response.model_name,
                "error": response.error_message
            })

        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def _handle_load_model(self, request):
        """Handle model loading request"""
        try:
            data = await request.json()

            success = await self._load_model(
                data["name"],
                ModelType(data.get("type", "custom")),
                data["path"],
                data.get("max_tokens", 2048),
                data.get("temperature", 0.7),
                data.get("top_p", 0.9)
            )

            return web.json_response({"success": success})

        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def _handle_unload_model(self, request):
        """Handle model unloading request"""
        try:
            data = await request.json()
            success = await self._unload_model(data["name"])
            return web.json_response({"success": success})

        except Exception as e:
            return web.json_response({
                "success": False,
                "error": str(e)
            }, status=500)

    async def _handle_metrics(self, request):
        """Prometheus metrics endpoint"""
        metrics_data = generate_latest(self.registry)
        return web.Response(text=metrics_data.decode('utf-8'), content_type='text/plain')

    async def cleanup(self):
        """Cleanup node resources"""
        try:
            # Update status
            self.status = NodeStatus.OFFLINE

            # Unload all models
            for model_name in list(self.loaded_models.keys()):
                await self._unload_model(model_name)

            # Close web server
            if self.site:
                await self.site.stop()
            if self.runner:
                await self.runner.cleanup()

            # Close HTTP session
            if self.session:
                await self.session.close()

            self.logger.info(f"Edge Node {self.config.node_id} cleanup completed")

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

# Example usage
async def main():
    """Example usage of EdgeNodeManager"""
    config = NodeConfiguration(
        node_id="edge-us-east-001",
        region="us-east",
        ip_address="172.30.0.47",
        port=8000,
        max_concurrent_requests=10,
        memory_limit_gb=16,
        gpu_memory_limit_gb=8,
        model_cache_size=3,
        health_check_interval=30,
        model_sync_interval=300,
        log_level="INFO"
    )

    node_manager = EdgeNodeManager(config)

    try:
        await node_manager.initialize()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        await node_manager.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())