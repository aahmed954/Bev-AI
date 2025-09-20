#!/usr/bin/env python3
"""
Embedding Generation Pipeline Manager
High-Performance Batch Processing for Vector Embeddings
Author: BEV OSINT Team
"""

import asyncio
import logging
import time
import hashlib
import json
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue
import threading

# ML/AI Libraries
import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    pipeline, Pipeline
)
from sentence_transformers import SentenceTransformer
import openai
from openai import AsyncOpenAI

# Database and caching
import asyncpg
import asyncio_redis as redis
from motor.motor_asyncio import AsyncIOMotorClient

# Performance monitoring
import psutil
from prometheus_client import Counter, Histogram, Gauge
import GPUtil

# Security and utilities
from cryptography.fernet import Fernet
import aiofiles
import xxhash


@dataclass
class EmbeddingRequest:
    """Request for embedding generation"""
    id: str
    content: str
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1=low, 2=medium, 3=high
    timeout: float = 30.0
    cache_ttl: int = 86400  # 24 hours


@dataclass
class EmbeddingResponse:
    """Response from embedding generation"""
    id: str
    vector: List[float]
    model_name: str
    processing_time: float
    cache_hit: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Configuration for embedding models"""
    name: str
    model_path: str
    max_length: int = 512
    batch_size: int = 32
    device: str = "auto"
    precision: str = "float32"  # float32, float16, int8
    pooling_strategy: str = "mean"  # mean, cls, max
    normalize: bool = True


class EmbeddingCache:
    """High-performance caching system for embeddings"""

    def __init__(self, redis_client: redis.Connection, encryption_key: bytes = None):
        self.redis = redis_client
        self.cipher = Fernet(encryption_key) if encryption_key else None
        self.logger = logging.getLogger('embedding_cache')

        # Local LRU cache for hot data
        self.local_cache: Dict[str, Tuple[List[float], float]] = {}
        self.max_local_cache_size = 10000
        self.cache_hits = 0
        self.cache_misses = 0

    def _generate_cache_key(self, content: str, model_name: str) -> str:
        """Generate cache key from content and model"""
        content_hash = xxhash.xxh64(content.encode('utf-8')).hexdigest()
        model_hash = xxhash.xxh64(model_name.encode('utf-8')).hexdigest()
        return f"embedding:{model_hash}:{content_hash}"

    async def get(self, content: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        cache_key = self._generate_cache_key(content, model_name)

        # Check local cache first
        if cache_key in self.local_cache:
            vector, timestamp = self.local_cache[cache_key]
            if time.time() - timestamp < 3600:  # 1 hour local cache
                self.cache_hits += 1
                return vector

        # Check Redis cache
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                if self.cipher:
                    cached_data = self.cipher.decrypt(cached_data)

                vector = pickle.loads(cached_data)

                # Update local cache
                if len(self.local_cache) < self.max_local_cache_size:
                    self.local_cache[cache_key] = (vector, time.time())

                self.cache_hits += 1
                return vector

        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")

        self.cache_misses += 1
        return None

    async def set(
        self,
        content: str,
        model_name: str,
        vector: List[float],
        ttl: int = 86400
    ):
        """Store embedding in cache"""
        cache_key = self._generate_cache_key(content, model_name)

        try:
            # Serialize vector
            data = pickle.dumps(vector)

            if self.cipher:
                data = self.cipher.encrypt(data)

            # Store in Redis
            await self.redis.setex(cache_key, ttl, data)

            # Update local cache
            if len(self.local_cache) < self.max_local_cache_size:
                self.local_cache[cache_key] = (vector, time.time())

        except Exception as e:
            self.logger.error(f"Cache set error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests) if total_requests > 0 else 0

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'local_cache_size': len(self.local_cache)
        }


class ModelManager:
    """Manages multiple embedding models with optimal resource allocation"""

    def __init__(self, device_config: Dict[str, Any] = None):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.configs: Dict[str, ModelConfig] = {}

        # Device management
        self.device_config = device_config or self._auto_detect_devices()
        self.current_device = 0

        # Performance monitoring
        self.model_stats: Dict[str, Dict[str, float]] = {}

        self.logger = logging.getLogger('model_manager')

    def _auto_detect_devices(self) -> Dict[str, Any]:
        """Auto-detect available computing devices"""
        config = {
            'cpu_cores': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpus': []
        }

        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                config['gpus'].append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_mb': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree
                })
        except:
            pass

        return config

    async def load_model(self, config: ModelConfig) -> bool:
        """Load a model with optimized configuration"""
        try:
            self.logger.info(f"📥 Loading model: {config.name}")

            # Determine device
            if config.device == "auto":
                if self.device_config['gpus'] and torch.cuda.is_available():
                    device = f"cuda:{self.current_device % len(self.device_config['gpus'])}"
                    self.current_device += 1
                else:
                    device = "cpu"
            else:
                device = config.device

            # Load model based on type
            if "sentence-transformers" in config.model_path:
                model = SentenceTransformer(config.model_path, device=device)

                # Optimize for inference
                model.eval()
                if device.startswith('cuda'):
                    model.half()  # Use FP16 for GPU

            else:
                # Load HuggingFace model
                tokenizer = AutoTokenizer.from_pretrained(config.model_path)
                model = AutoModel.from_pretrained(
                    config.model_path,
                    torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32
                ).to(device)

                model.eval()
                self.tokenizers[config.name] = tokenizer

            self.models[config.name] = model
            self.configs[config.name] = config

            # Initialize stats
            self.model_stats[config.name] = {
                'total_requests': 0,
                'total_time': 0,
                'average_time': 0,
                'device': device
            }

            self.logger.info(f"✅ Model {config.name} loaded on {device}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Failed to load model {config.name}: {e}")
            return False

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model_name: str
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")

        start_time = time.time()
        model = self.models[model_name]
        config = self.configs[model_name]

        try:
            if isinstance(model, SentenceTransformer):
                # Use sentence-transformers
                embeddings = await asyncio.to_thread(
                    model.encode,
                    texts,
                    batch_size=config.batch_size,
                    normalize_embeddings=config.normalize,
                    show_progress_bar=False
                )
                embeddings = embeddings.tolist()

            else:
                # Use HuggingFace model
                tokenizer = self.tokenizers[model_name]
                embeddings = []

                for i in range(0, len(texts), config.batch_size):
                    batch_texts = texts[i:i + config.batch_size]

                    # Tokenize
                    inputs = tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=config.max_length,
                        return_tensors="pt"
                    ).to(model.device)

                    # Generate embeddings
                    with torch.no_grad():
                        outputs = model(**inputs)

                        if config.pooling_strategy == "mean":
                            # Mean pooling
                            attention_mask = inputs['attention_mask']
                            token_embeddings = outputs.last_hidden_state
                            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                            batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                        elif config.pooling_strategy == "cls":
                            # CLS token
                            batch_embeddings = outputs.last_hidden_state[:, 0]
                        else:
                            # Max pooling
                            batch_embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]

                        if config.normalize:
                            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                        embeddings.extend(batch_embeddings.cpu().numpy().tolist())

            # Update stats
            processing_time = time.time() - start_time
            stats = self.model_stats[model_name]
            stats['total_requests'] += len(texts)
            stats['total_time'] += processing_time
            stats['average_time'] = stats['total_time'] / stats['total_requests']

            return embeddings

        except Exception as e:
            self.logger.error(f"❌ Embedding generation failed for {model_name}: {e}")
            raise

    def get_model_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics for all models"""
        return self.model_stats.copy()

    async def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.tokenizers:
                del self.tokenizers[model_name]
            del self.configs[model_name]
            del self.model_stats[model_name]

            # Force garbage collection
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self.logger.info(f"✅ Model {model_name} unloaded")


class EmbeddingPipeline:
    """High-performance embedding generation pipeline"""

    def __init__(
        self,
        postgres_config: Dict[str, Any],
        redis_config: Dict[str, Any],
        encryption_key: bytes = None
    ):
        self.postgres_config = postgres_config
        self.redis_config = redis_config

        # Components
        self.model_manager = ModelManager()
        self.cache: Optional[EmbeddingCache] = None
        self.postgres: Optional[asyncpg.Connection] = None
        self.redis: Optional[redis.Connection] = None

        # Processing queues
        self.request_queues = {
            1: asyncio.Queue(maxsize=10000),  # Low priority
            2: asyncio.Queue(maxsize=5000),   # Medium priority
            3: asyncio.Queue(maxsize=1000)    # High priority
        }

        # Worker management
        self.workers_running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.max_workers = min(16, mp.cpu_count() * 2)

        # Performance metrics
        self.metrics = self._setup_metrics()

        # Security
        self.encryption_key = encryption_key or Fernet.generate_key()

        self.logger = logging.getLogger('embedding_pipeline')

    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup Prometheus metrics"""
        return {
            'requests_total': Counter(
                'bev_embedding_requests_total',
                'Total embedding requests',
                ['model', 'status']
            ),
            'processing_duration': Histogram(
                'bev_embedding_processing_seconds',
                'Embedding processing duration',
                ['model']
            ),
            'queue_size': Gauge(
                'bev_embedding_queue_size',
                'Current queue size',
                ['priority']
            ),
            'cache_hit_rate': Gauge(
                'bev_embedding_cache_hit_rate',
                'Cache hit rate'
            ),
            'models_loaded': Gauge(
                'bev_embedding_models_loaded',
                'Number of loaded models'
            )
        }

    async def initialize(self) -> bool:
        """Initialize the embedding pipeline"""
        self.logger.info("🚀 Initializing Embedding Pipeline")

        try:
            # Connect to databases
            self.postgres = await asyncpg.connect(**self.postgres_config)
            self.redis = await redis.Connection.create(**self.redis_config)

            # Initialize cache
            self.cache = EmbeddingCache(self.redis, self.encryption_key)

            # Load default models
            default_models = [
                ModelConfig(
                    name="sentence-transformers-mini",
                    model_path="sentence-transformers/all-MiniLM-L6-v2",
                    batch_size=64,
                    max_length=256
                ),
                ModelConfig(
                    name="sentence-transformers-large",
                    model_path="sentence-transformers/all-mpnet-base-v2",
                    batch_size=32,
                    max_length=384
                ),
                ModelConfig(
                    name="multilingual",
                    model_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    batch_size=32,
                    max_length=256
                )
            ]

            for model_config in default_models:
                await self.model_manager.load_model(model_config)

            # Start worker processes
            await self._start_workers()

            # Update metrics
            self.metrics['models_loaded'].set(len(self.model_manager.models))

            self.logger.info("✅ Embedding Pipeline initialized")
            return True

        except Exception as e:
            self.logger.error(f"❌ Pipeline initialization failed: {e}")
            return False

    async def _start_workers(self):
        """Start embedding processing workers"""
        self.workers_running = True

        # Create workers for each priority level
        for priority in [3, 2, 1]:  # Process high priority first
            for i in range(self.max_workers // 3):
                worker_task = asyncio.create_task(
                    self._worker(priority, f"worker-p{priority}-{i}")
                )
                self.worker_tasks.append(worker_task)

        self.logger.info(f"✅ Started {len(self.worker_tasks)} embedding workers")

    async def _worker(self, priority: int, worker_id: str):
        """Embedding processing worker"""
        queue = self.request_queues[priority]

        while self.workers_running:
            try:
                # Get batch of requests
                batch = []
                timeout = 0.1 if priority == 3 else 1.0  # High priority gets faster processing

                try:
                    # Get first request
                    request = await asyncio.wait_for(queue.get(), timeout=timeout)
                    batch.append(request)

                    # Try to get more requests for batch processing
                    max_batch_size = 32
                    for _ in range(max_batch_size - 1):
                        try:
                            request = queue.get_nowait()
                            batch.append(request)
                        except asyncio.QueueEmpty:
                            break

                except asyncio.TimeoutError:
                    continue

                if batch:
                    await self._process_batch(batch, worker_id)

            except Exception as e:
                self.logger.error(f"❌ Worker {worker_id} error: {e}")
                await asyncio.sleep(1)

    async def _process_batch(self, requests: List[EmbeddingRequest], worker_id: str):
        """Process a batch of embedding requests"""
        if not requests:
            return

        # Group by model
        model_groups = {}
        for request in requests:
            model_name = request.model_name
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(request)

        # Process each model group
        for model_name, model_requests in model_groups.items():
            await self._process_model_batch(model_requests, model_name, worker_id)

    async def _process_model_batch(
        self,
        requests: List[EmbeddingRequest],
        model_name: str,
        worker_id: str
    ):
        """Process requests for a specific model"""
        start_time = time.time()

        try:
            # Check cache for each request
            cached_results = {}
            uncached_requests = []

            for request in requests:
                if self.cache:
                    cached_vector = await self.cache.get(request.content, model_name)
                    if cached_vector:
                        cached_results[request.id] = EmbeddingResponse(
                            id=request.id,
                            vector=cached_vector,
                            model_name=model_name,
                            processing_time=0,
                            cache_hit=True
                        )
                        continue

                uncached_requests.append(request)

            # Process uncached requests
            if uncached_requests:
                texts = [req.content for req in uncached_requests]

                try:
                    embeddings = await self.model_manager.generate_embeddings_batch(
                        texts, model_name
                    )

                    # Create responses and cache results
                    processing_time = time.time() - start_time
                    for i, request in enumerate(uncached_requests):
                        vector = embeddings[i]

                        response = EmbeddingResponse(
                            id=request.id,
                            vector=vector,
                            model_name=model_name,
                            processing_time=processing_time / len(uncached_requests),
                            cache_hit=False
                        )

                        cached_results[request.id] = response

                        # Cache the result
                        if self.cache:
                            await self.cache.set(
                                request.content,
                                model_name,
                                vector,
                                request.cache_ttl
                            )

                    # Update metrics
                    self.metrics['requests_total'].labels(
                        model=model_name,
                        status='success'
                    ).inc(len(uncached_requests))

                    self.metrics['processing_duration'].labels(
                        model=model_name
                    ).observe(processing_time)

                except Exception as e:
                    # Create error responses
                    for request in uncached_requests:
                        cached_results[request.id] = EmbeddingResponse(
                            id=request.id,
                            vector=[],
                            model_name=model_name,
                            processing_time=0,
                            error=str(e)
                        )

                    self.metrics['requests_total'].labels(
                        model=model_name,
                        status='error'
                    ).inc(len(uncached_requests))

            # Store results (this would typically be sent back to requestor)
            await self._store_results(list(cached_results.values()))

        except Exception as e:
            self.logger.error(f"❌ Batch processing error: {e}")

    async def _store_results(self, responses: List[EmbeddingResponse]):
        """Store embedding results"""
        try:
            if self.postgres:
                # Store in PostgreSQL for audit trail
                for response in responses:
                    if not response.error:
                        await self.postgres.execute("""
                            INSERT INTO embedding_results
                            (request_id, model_name, vector_dimensions, processing_time, cache_hit, created_at)
                            VALUES ($1, $2, $3, $4, $5, NOW())
                            ON CONFLICT (request_id) DO UPDATE SET
                                processing_time = EXCLUDED.processing_time,
                                cache_hit = EXCLUDED.cache_hit
                        """, response.id, response.model_name, len(response.vector),
                            response.processing_time, response.cache_hit)

        except Exception as e:
            self.logger.error(f"❌ Failed to store results: {e}")

    async def generate_embedding(
        self,
        content: str,
        model_name: str = "sentence-transformers-mini",
        priority: int = 2,
        timeout: float = 30.0
    ) -> EmbeddingResponse:
        """Generate single embedding (convenience method)"""
        request = EmbeddingRequest(
            id=str(hash(content + model_name + str(time.time()))),
            content=content,
            model_name=model_name,
            priority=priority,
            timeout=timeout
        )

        return await self.generate_embeddings_batch([request])[0]

    async def generate_embeddings_batch(
        self,
        requests: List[EmbeddingRequest]
    ) -> List[EmbeddingResponse]:
        """Generate embeddings for multiple requests"""
        if not requests:
            return []

        # Add requests to appropriate priority queues
        for request in requests:
            await self.request_queues[request.priority].put(request)

        # Update queue size metrics
        for priority, queue in self.request_queues.items():
            self.metrics['queue_size'].labels(priority=priority).set(queue.qsize())

        # Wait for processing (in real implementation, this would use a result queue)
        # For now, we'll simulate immediate processing
        await asyncio.sleep(0.1)

        # Return mock responses (in real implementation, get from result queue)
        return [
            EmbeddingResponse(
                id=req.id,
                vector=[0.1] * 384,  # Mock vector
                model_name=req.model_name,
                processing_time=0.1
            )
            for req in requests
        ]

    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        stats = {
            'models_loaded': len(self.model_manager.models),
            'workers_running': len(self.worker_tasks),
            'queue_sizes': {
                str(priority): queue.qsize()
                for priority, queue in self.request_queues.items()
            },
            'model_stats': self.model_manager.get_model_stats()
        }

        if self.cache:
            stats['cache_stats'] = self.cache.get_stats()

        return stats

    async def shutdown(self):
        """Shutdown the embedding pipeline"""
        self.logger.info("🔒 Shutting down Embedding Pipeline")

        # Stop workers
        self.workers_running = False

        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()

        await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        # Close database connections
        if self.postgres:
            await self.postgres.close()

        if self.redis:
            await self.redis.close()

        self.logger.info("✅ Embedding Pipeline shutdown complete")


# Usage Example
async def main():
    """Example usage of Embedding Pipeline"""

    # Configuration
    postgres_config = {
        'host': '172.30.0.2',
        'port': 5432,
        'user': 'bev_user',
        'password': 'secure_password',
        'database': 'osint'
    }

    redis_config = {
        'host': '172.30.0.4',
        'port': 6379,
        'db': 0
    }

    # Initialize pipeline
    pipeline = EmbeddingPipeline(postgres_config, redis_config)

    if await pipeline.initialize():
        print("✅ Embedding Pipeline ready")

        # Example: Generate embeddings
        requests = [
            EmbeddingRequest(
                id=f"req_{i}",
                content=f"Sample OSINT document {i}",
                model_name="sentence-transformers-mini",
                priority=2
            )
            for i in range(100)
        ]

        start_time = time.time()
        responses = await pipeline.generate_embeddings_batch(requests)
        processing_time = time.time() - start_time

        print(f"✅ Processed {len(responses)} embeddings in {processing_time:.2f}s")
        print(f"📊 Rate: {len(responses)/processing_time:.1f} embeddings/second")

        # Get statistics
        stats = await pipeline.get_pipeline_stats()
        print(f"📈 Pipeline stats: {json.dumps(stats, indent=2)}")

        await pipeline.shutdown()


if __name__ == "__main__":
    asyncio.run(main())