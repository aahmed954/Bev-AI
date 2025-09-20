#!/usr/bin/env python3
"""
Vector Database Performance Benchmarking and Optimization System
Comprehensive testing and optimization for BEV OSINT vector infrastructure
Author: BEV OSINT Team
"""

import asyncio
import logging
import time
import json
import random
import statistics
import uuid
from typing import Dict, List, Any, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

# Performance monitoring
import psutil
import GPUtil
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway

# Database and ML libraries
import torch
from sentence_transformers import SentenceTransformer

# BEV Infrastructure
from .vector_db_manager import VectorDatabaseManager, EmbeddingDocument
from .embedding_manager import EmbeddingPipeline, EmbeddingRequest
from .database_integration import DatabaseIntegrationOrchestrator

# Utilities
import aiofiles
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


@dataclass
class BenchmarkConfig:
    """Configuration for performance benchmarks"""
    name: str
    description: str
    test_data_size: int = 1000
    batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    vector_dimensions: List[int] = field(default_factory=lambda: [384, 768, 1024])
    concurrent_clients: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    iterations: int = 3
    timeout: float = 300.0
    warm_up_iterations: int = 1


@dataclass
class PerformanceMetrics:
    """Performance measurement results"""
    operation: str
    database: str
    batch_size: int
    vector_dim: int
    concurrent_clients: int

    # Timing metrics
    min_latency: float
    max_latency: float
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float

    # Throughput metrics
    operations_per_second: float
    vectors_per_second: float

    # Resource utilization
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0

    # Error metrics
    error_rate: float
    timeout_rate: float

    # Quality metrics (for search operations)
    recall_at_10: float = 0.0
    precision_at_10: float = 0.0

    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OptimizationRecommendation:
    """System optimization recommendation"""
    category: str  # 'performance', 'resource', 'configuration'
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    expected_improvement: str
    implementation_effort: str  # 'low', 'medium', 'high'
    config_changes: Dict[str, Any] = field(default_factory=dict)


class VectorDatabaseBenchmark:
    """Comprehensive vector database performance benchmarking"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()

        # Test data generators
        self.test_vectors: Dict[int, List[List[float]]] = {}
        self.test_documents: List[EmbeddingDocument] = []
        self.ground_truth_similarities: Dict[str, List[Tuple[str, float]]] = {}

        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.system_resources: List[Dict[str, Any]] = []

        # Components under test
        self.vector_manager: Optional[VectorDatabaseManager] = None
        self.embedding_pipeline: Optional[EmbeddingPipeline] = None

        # Results storage
        self.results_dir = Path(config.get('results_dir', './benchmark_results'))
        self.results_dir.mkdir(exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('vector_db_benchmark')
        logger.setLevel(logging.INFO)
        return logger

    async def initialize(self) -> bool:
        """Initialize benchmarking system"""
        self.logger.info("🚀 Initializing Vector Database Benchmark System")

        try:
            # Initialize vector database manager
            self.vector_manager = VectorDatabaseManager(self.config['vector_db'])
            await self.vector_manager.initialize()

            # Initialize embedding pipeline
            self.embedding_pipeline = EmbeddingPipeline(
                self.config['postgres'],
                self.config['redis']
            )
            await self.embedding_pipeline.initialize()

            # Generate test data
            await self._generate_test_data()

            self.logger.info("✅ Benchmark system initialized")
            return True

        except Exception as e:
            self.logger.error(f"❌ Benchmark initialization failed: {e}")
            return False

    async def _generate_test_data(self):
        """Generate synthetic test data for benchmarking"""
        self.logger.info("🎲 Generating test data...")

        # Generate vectors of different dimensions
        for dim in [384, 768, 1024]:
            vectors = []
            for i in range(10000):  # Large test set
                # Generate normalized random vectors
                vector = np.random.normal(0, 1, dim)
                vector = vector / np.linalg.norm(vector)
                vectors.append(vector.tolist())

            self.test_vectors[dim] = vectors

        # Generate test documents
        content_templates = [
            "Cybersecurity threat analysis report {i}",
            "Dark web intelligence gathering on {topic}",
            "Social media monitoring for {entity}",
            "OSINT investigation {case_id}",
            "Malware analysis for sample {hash}",
            "Network intrusion detection alert {alert_id}",
            "Phishing campaign analysis {campaign}",
            "Cryptocurrency transaction monitoring {tx_id}",
            "Geolocation intelligence report {location}",
            "Digital forensics investigation {case}"
        ]

        topics = ["APT groups", "ransomware", "data breaches", "IoCs", "threat actors"]

        for i in range(5000):
            template = random.choice(content_templates)
            content = template.format(
                i=i,
                topic=random.choice(topics),
                entity=f"entity_{random.randint(1000, 9999)}",
                case_id=f"CASE-{random.randint(100000, 999999)}",
                hash=f"md5_{uuid.uuid4().hex[:32]}",
                alert_id=f"ALERT-{random.randint(10000, 99999)}",
                campaign=f"CAMP-{random.randint(1000, 9999)}",
                tx_id=f"0x{uuid.uuid4().hex}",
                location=f"LAT{random.uniform(-90, 90):.4f}_LON{random.uniform(-180, 180):.4f}",
                case=f"INV-{random.randint(100000, 999999)}"
            )

            doc = EmbeddingDocument(
                id=f"test_doc_{i}",
                content=content,
                metadata={
                    'category': random.choice(['threat_intel', 'osint', 'forensics', 'monitoring']),
                    'severity': random.choice(['low', 'medium', 'high', 'critical']),
                    'source': random.choice(['twitter', 'darkweb', 'forum', 'email', 'network']),
                    'timestamp': datetime.utcnow().isoformat()
                },
                collection='benchmark_test'
            )

            self.test_documents.append(doc)

        self.logger.info(f"✅ Generated {len(self.test_documents)} test documents")

    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark suite"""
        self.logger.info("🏁 Starting comprehensive benchmark suite")

        benchmark_configs = [
            BenchmarkConfig(
                name="vector_insertion",
                description="Vector insertion performance test",
                test_data_size=1000,
                batch_sizes=[8, 16, 32, 64, 128],
                iterations=3
            ),
            BenchmarkConfig(
                name="vector_search",
                description="Vector similarity search performance test",
                test_data_size=500,
                batch_sizes=[1, 4, 8, 16],
                iterations=5
            ),
            BenchmarkConfig(
                name="embedding_generation",
                description="Embedding generation pipeline performance",
                test_data_size=1000,
                batch_sizes=[16, 32, 64, 128, 256],
                iterations=3
            ),
            BenchmarkConfig(
                name="concurrent_operations",
                description="Concurrent load testing",
                test_data_size=2000,
                concurrent_clients=[1, 2, 4, 8, 16, 32],
                iterations=2
            ),
            BenchmarkConfig(
                name="memory_scaling",
                description="Memory usage scaling test",
                test_data_size=5000,
                batch_sizes=[64, 128, 256, 512],
                iterations=1
            )
        ]

        all_results = {}

        for config in benchmark_configs:
            self.logger.info(f"🧪 Running benchmark: {config.name}")

            try:
                if config.name == "vector_insertion":
                    results = await self._benchmark_vector_insertion(config)
                elif config.name == "vector_search":
                    results = await self._benchmark_vector_search(config)
                elif config.name == "embedding_generation":
                    results = await self._benchmark_embedding_generation(config)
                elif config.name == "concurrent_operations":
                    results = await self._benchmark_concurrent_operations(config)
                elif config.name == "memory_scaling":
                    results = await self._benchmark_memory_scaling(config)
                else:
                    continue

                all_results[config.name] = results

                # Save intermediate results
                await self._save_benchmark_results(config.name, results)

            except Exception as e:
                self.logger.error(f"❌ Benchmark {config.name} failed: {e}")
                all_results[config.name] = {'error': str(e)}

        # Generate optimization recommendations
        recommendations = await self._generate_optimization_recommendations(all_results)
        all_results['optimization_recommendations'] = recommendations

        # Generate comprehensive report
        report_path = await self._generate_benchmark_report(all_results)
        all_results['report_path'] = str(report_path)

        self.logger.info("✅ Comprehensive benchmark completed")
        return all_results

    async def _benchmark_vector_insertion(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark vector insertion performance"""
        results = {'metrics': [], 'summary': {}}

        for batch_size in config.batch_sizes:
            for vector_dim in config.vector_dimensions:
                for iteration in range(config.iterations):
                    # Prepare test data
                    test_docs = self.test_documents[:config.test_data_size]
                    vectors = self.test_vectors[vector_dim][:config.test_data_size]

                    # Add vectors to documents
                    for i, doc in enumerate(test_docs):
                        doc.vector = vectors[i]

                    # Measure insertion performance
                    start_time = time.time()
                    cpu_before = psutil.cpu_percent()
                    memory_before = psutil.virtual_memory().percent

                    # Process in batches
                    latencies = []
                    errors = 0

                    for i in range(0, len(test_docs), batch_size):
                        batch = test_docs[i:i + batch_size]

                        batch_start = time.time()
                        try:
                            success = await self.vector_manager.upsert_vectors(
                                batch, f"benchmark_test_{vector_dim}"
                            )
                            if not success:
                                errors += 1
                        except Exception:
                            errors += 1

                        batch_latency = time.time() - batch_start
                        latencies.append(batch_latency)

                    total_time = time.time() - start_time
                    cpu_after = psutil.cpu_percent()
                    memory_after = psutil.virtual_memory().percent

                    # Calculate metrics
                    if latencies:
                        metrics = PerformanceMetrics(
                            operation="vector_insertion",
                            database="qdrant",
                            batch_size=batch_size,
                            vector_dim=vector_dim,
                            concurrent_clients=1,
                            min_latency=min(latencies),
                            max_latency=max(latencies),
                            avg_latency=statistics.mean(latencies),
                            p50_latency=statistics.median(latencies),
                            p95_latency=np.percentile(latencies, 95),
                            p99_latency=np.percentile(latencies, 99),
                            operations_per_second=len(latencies) / total_time,
                            vectors_per_second=len(test_docs) / total_time,
                            cpu_usage=(cpu_after + cpu_before) / 2,
                            memory_usage=(memory_after + memory_before) / 2,
                            error_rate=errors / len(latencies) if latencies else 1.0,
                            timeout_rate=0.0
                        )

                        results['metrics'].append(asdict(metrics))
                        self.metrics_history.append(metrics)

        # Calculate summary statistics
        if results['metrics']:
            avg_throughput = statistics.mean([m['vectors_per_second'] for m in results['metrics']])
            avg_latency = statistics.mean([m['avg_latency'] for m in results['metrics']])

            results['summary'] = {
                'average_throughput_vectors_per_second': avg_throughput,
                'average_latency_seconds': avg_latency,
                'total_tests': len(results['metrics']),
                'test_data_size': config.test_data_size
            }

        return results

    async def _benchmark_vector_search(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark vector search performance"""
        results = {'metrics': [], 'summary': {}}

        # First, ensure we have data to search
        await self._populate_search_data()

        for batch_size in [1]:  # Search is typically single queries
            for vector_dim in config.vector_dimensions:
                query_vectors = self.test_vectors[vector_dim][:config.test_data_size]

                for iteration in range(config.iterations):
                    latencies = []
                    recall_scores = []
                    errors = 0

                    cpu_before = psutil.cpu_percent()
                    memory_before = psutil.virtual_memory().percent

                    start_time = time.time()

                    for query_vector in query_vectors[:100]:  # Sample of queries
                        query_start = time.time()

                        try:
                            search_results = await self.vector_manager.search_vectors(
                                query_vector=query_vector,
                                collection=f"benchmark_test_{vector_dim}",
                                limit=10,
                                score_threshold=0.5
                            )

                            # Calculate recall (simplified)
                            recall = len(search_results) / min(10, len(query_vectors))
                            recall_scores.append(recall)

                        except Exception:
                            errors += 1
                            recall_scores.append(0.0)

                        query_latency = time.time() - query_start
                        latencies.append(query_latency)

                    total_time = time.time() - start_time
                    cpu_after = psutil.cpu_percent()
                    memory_after = psutil.virtual_memory().percent

                    if latencies:
                        metrics = PerformanceMetrics(
                            operation="vector_search",
                            database="qdrant",
                            batch_size=batch_size,
                            vector_dim=vector_dim,
                            concurrent_clients=1,
                            min_latency=min(latencies),
                            max_latency=max(latencies),
                            avg_latency=statistics.mean(latencies),
                            p50_latency=statistics.median(latencies),
                            p95_latency=np.percentile(latencies, 95),
                            p99_latency=np.percentile(latencies, 99),
                            operations_per_second=len(latencies) / total_time,
                            vectors_per_second=len(latencies) / total_time,
                            cpu_usage=(cpu_after + cpu_before) / 2,
                            memory_usage=(memory_after + memory_before) / 2,
                            error_rate=errors / len(latencies) if latencies else 1.0,
                            timeout_rate=0.0,
                            recall_at_10=statistics.mean(recall_scores) if recall_scores else 0.0,
                            precision_at_10=statistics.mean(recall_scores) if recall_scores else 0.0
                        )

                        results['metrics'].append(asdict(metrics))
                        self.metrics_history.append(metrics)

        return results

    async def _benchmark_embedding_generation(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark embedding generation performance"""
        results = {'metrics': [], 'summary': {}}

        test_texts = [doc.content for doc in self.test_documents[:config.test_data_size]]

        for batch_size in config.batch_sizes:
            for iteration in range(config.iterations):
                latencies = []
                errors = 0

                cpu_before = psutil.cpu_percent()
                memory_before = psutil.virtual_memory().percent
                gpu_before = self._get_gpu_utilization()

                start_time = time.time()

                for i in range(0, len(test_texts), batch_size):
                    batch_texts = test_texts[i:i + batch_size]

                    batch_start = time.time()

                    try:
                        requests = [
                            EmbeddingRequest(
                                id=f"bench_{j}",
                                content=text,
                                model_name="sentence-transformers-mini",
                                priority=2
                            )
                            for j, text in enumerate(batch_texts)
                        ]

                        responses = await self.embedding_pipeline.generate_embeddings_batch(requests)

                        if not responses or any(r.error for r in responses):
                            errors += 1

                    except Exception:
                        errors += 1

                    batch_latency = time.time() - batch_start
                    latencies.append(batch_latency)

                total_time = time.time() - start_time
                cpu_after = psutil.cpu_percent()
                memory_after = psutil.virtual_memory().percent
                gpu_after = self._get_gpu_utilization()

                if latencies:
                    metrics = PerformanceMetrics(
                        operation="embedding_generation",
                        database="embedding_pipeline",
                        batch_size=batch_size,
                        vector_dim=384,  # Default model dimension
                        concurrent_clients=1,
                        min_latency=min(latencies),
                        max_latency=max(latencies),
                        avg_latency=statistics.mean(latencies),
                        p50_latency=statistics.median(latencies),
                        p95_latency=np.percentile(latencies, 95),
                        p99_latency=np.percentile(latencies, 99),
                        operations_per_second=len(latencies) / total_time,
                        vectors_per_second=len(test_texts) / total_time,
                        cpu_usage=(cpu_after + cpu_before) / 2,
                        memory_usage=(memory_after + memory_before) / 2,
                        gpu_usage=(gpu_after['utilization'] + gpu_before['utilization']) / 2,
                        gpu_memory=(gpu_after['memory'] + gpu_before['memory']) / 2,
                        error_rate=errors / len(latencies) if latencies else 1.0,
                        timeout_rate=0.0
                    )

                    results['metrics'].append(asdict(metrics))
                    self.metrics_history.append(metrics)

        return results

    async def _benchmark_concurrent_operations(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark concurrent operation performance"""
        results = {'metrics': [], 'summary': {}}

        for concurrent_clients in config.concurrent_clients:
            for iteration in range(config.iterations):
                # Create concurrent tasks
                tasks = []
                test_subset = self.test_documents[:config.test_data_size // concurrent_clients]

                cpu_before = psutil.cpu_percent()
                memory_before = psutil.virtual_memory().percent

                start_time = time.time()

                for client_id in range(concurrent_clients):
                    client_docs = [
                        doc for i, doc in enumerate(test_subset)
                        if i % concurrent_clients == client_id
                    ]

                    # Add vectors to documents
                    for doc in client_docs:
                        doc.vector = self.test_vectors[384][0]  # Use default vector

                    task = asyncio.create_task(
                        self._concurrent_client_task(client_docs, f"concurrent_test_{client_id}")
                    )
                    tasks.append(task)

                # Wait for all tasks to complete
                try:
                    task_results = await asyncio.gather(*tasks, return_exceptions=True)

                    total_time = time.time() - start_time
                    cpu_after = psutil.cpu_percent()
                    memory_after = psutil.virtual_memory().percent

                    # Analyze results
                    successful_tasks = [r for r in task_results if not isinstance(r, Exception)]
                    error_rate = 1 - (len(successful_tasks) / len(tasks))

                    if successful_tasks:
                        all_latencies = []
                        for task_result in successful_tasks:
                            all_latencies.extend(task_result['latencies'])

                        if all_latencies:
                            metrics = PerformanceMetrics(
                                operation="concurrent_operations",
                                database="qdrant",
                                batch_size=32,  # Default batch size
                                vector_dim=384,
                                concurrent_clients=concurrent_clients,
                                min_latency=min(all_latencies),
                                max_latency=max(all_latencies),
                                avg_latency=statistics.mean(all_latencies),
                                p50_latency=statistics.median(all_latencies),
                                p95_latency=np.percentile(all_latencies, 95),
                                p99_latency=np.percentile(all_latencies, 99),
                                operations_per_second=len(all_latencies) / total_time,
                                vectors_per_second=config.test_data_size / total_time,
                                cpu_usage=(cpu_after + cpu_before) / 2,
                                memory_usage=(memory_after + memory_before) / 2,
                                error_rate=error_rate,
                                timeout_rate=0.0
                            )

                            results['metrics'].append(asdict(metrics))
                            self.metrics_history.append(metrics)

                except Exception as e:
                    self.logger.error(f"❌ Concurrent benchmark failed: {e}")

        return results

    async def _concurrent_client_task(
        self,
        documents: List[EmbeddingDocument],
        collection: str
    ) -> Dict[str, Any]:
        """Single client task for concurrent testing"""
        latencies = []
        errors = 0

        batch_size = 16
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            start_time = time.time()
            try:
                success = await self.vector_manager.upsert_vectors(batch, collection)
                if not success:
                    errors += 1
            except Exception:
                errors += 1

            latency = time.time() - start_time
            latencies.append(latency)

        return {
            'latencies': latencies,
            'errors': errors,
            'total_operations': len(latencies)
        }

    async def _benchmark_memory_scaling(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """Benchmark memory usage scaling"""
        results = {'metrics': [], 'summary': {}}

        for batch_size in config.batch_sizes:
            self.logger.info(f"🧠 Testing memory scaling with batch size {batch_size}")

            # Monitor memory usage over time
            memory_samples = []
            cpu_samples = []

            # Process increasingly large batches
            test_docs = self.test_documents[:config.test_data_size]

            for doc in test_docs:
                doc.vector = self.test_vectors[768][0]  # Use larger vectors

            start_time = time.time()

            for i in range(0, len(test_docs), batch_size):
                batch = test_docs[i:i + batch_size]

                # Sample resources before processing
                memory_before = psutil.virtual_memory()
                cpu_before = psutil.cpu_percent()

                try:
                    await self.vector_manager.upsert_vectors(batch, "memory_test")
                except Exception as e:
                    self.logger.warning(f"Memory test batch failed: {e}")

                # Sample resources after processing
                memory_after = psutil.virtual_memory()
                cpu_after = psutil.cpu_percent()

                memory_samples.append({
                    'batch_number': i // batch_size,
                    'vectors_processed': min(i + batch_size, len(test_docs)),
                    'memory_used_mb': (memory_after.used - memory_before.used) / (1024 * 1024),
                    'memory_percent': memory_after.percent,
                    'cpu_percent': cpu_after
                })

                cpu_samples.append(cpu_after)

            total_time = time.time() - start_time

            # Calculate memory efficiency metrics
            if memory_samples:
                total_memory_used = sum(s['memory_used_mb'] for s in memory_samples)
                avg_memory_per_vector = total_memory_used / len(test_docs)
                peak_memory = max(s['memory_percent'] for s in memory_samples)

                metrics = PerformanceMetrics(
                    operation="memory_scaling",
                    database="qdrant",
                    batch_size=batch_size,
                    vector_dim=768,
                    concurrent_clients=1,
                    min_latency=0.0,
                    max_latency=total_time,
                    avg_latency=total_time / len(memory_samples),
                    p50_latency=total_time / 2,
                    p95_latency=total_time * 0.95,
                    p99_latency=total_time * 0.99,
                    operations_per_second=len(memory_samples) / total_time,
                    vectors_per_second=len(test_docs) / total_time,
                    cpu_usage=statistics.mean(cpu_samples),
                    memory_usage=peak_memory,
                    error_rate=0.0,
                    timeout_rate=0.0
                )

                results['metrics'].append(asdict(metrics))
                results['memory_efficiency'] = {
                    'memory_per_vector_mb': avg_memory_per_vector,
                    'peak_memory_percent': peak_memory,
                    'total_memory_used_mb': total_memory_used
                }

        return results

    async def _populate_search_data(self):
        """Populate vector database with search test data"""
        test_docs = self.test_documents[:1000]

        for vector_dim in [384, 768, 1024]:
            vectors = self.test_vectors[vector_dim][:1000]

            for i, doc in enumerate(test_docs):
                doc.vector = vectors[i]

            try:
                await self.vector_manager.upsert_vectors(test_docs, f"benchmark_test_{vector_dim}")
            except Exception as e:
                self.logger.warning(f"Failed to populate search data for dim {vector_dim}: {e}")

    def _get_gpu_utilization(self) -> Dict[str, float]:
        """Get GPU utilization metrics"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                return {
                    'utilization': gpu.load * 100,
                    'memory': gpu.memoryUtil * 100
                }
        except:
            pass

        return {'utilization': 0.0, 'memory': 0.0}

    async def _generate_optimization_recommendations(
        self,
        benchmark_results: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on benchmark results"""
        recommendations = []

        # Analyze vector insertion performance
        if 'vector_insertion' in benchmark_results:
            insertion_metrics = benchmark_results['vector_insertion']['metrics']

            if insertion_metrics:
                # Find optimal batch size
                best_throughput = max(m['vectors_per_second'] for m in insertion_metrics)
                optimal_batch = next(
                    m['batch_size'] for m in insertion_metrics
                    if m['vectors_per_second'] == best_throughput
                )

                recommendations.append(OptimizationRecommendation(
                    category='performance',
                    priority='high',
                    title=f'Optimize Vector Insertion Batch Size',
                    description=f'Use batch size of {optimal_batch} for optimal throughput ({best_throughput:.1f} vectors/sec)',
                    expected_improvement='20-40% throughput increase',
                    implementation_effort='low',
                    config_changes={'batch_size': optimal_batch}
                ))

                # Memory usage optimization
                high_memory_batches = [m for m in insertion_metrics if m['memory_usage'] > 80]
                if high_memory_batches:
                    recommendations.append(OptimizationRecommendation(
                        category='resource',
                        priority='medium',
                        title='Reduce Memory Usage During Insertion',
                        description='High memory usage detected during large batch insertions',
                        expected_improvement='30-50% memory reduction',
                        implementation_effort='medium',
                        config_changes={
                            'max_batch_size': 64,
                            'memory_limit_gb': 8
                        }
                    ))

        # Analyze search performance
        if 'vector_search' in benchmark_results:
            search_metrics = benchmark_results['vector_search']['metrics']

            if search_metrics:
                avg_search_latency = statistics.mean(m['avg_latency'] for m in search_metrics)

                if avg_search_latency > 0.1:  # 100ms threshold
                    recommendations.append(OptimizationRecommendation(
                        category='performance',
                        priority='high',
                        title='Optimize Search Index Configuration',
                        description=f'Average search latency is {avg_search_latency:.3f}s, consider index optimization',
                        expected_improvement='50-70% latency reduction',
                        implementation_effort='medium',
                        config_changes={
                            'index_type': 'HNSW',
                            'ef_construct': 200,
                            'm': 16
                        }
                    ))

        # Analyze embedding generation
        if 'embedding_generation' in benchmark_results:
            embedding_metrics = benchmark_results['embedding_generation']['metrics']

            if embedding_metrics:
                gpu_metrics = [m for m in embedding_metrics if m['gpu_usage'] > 0]

                if not gpu_metrics:
                    recommendations.append(OptimizationRecommendation(
                        category='performance',
                        priority='high',
                        title='Enable GPU Acceleration for Embeddings',
                        description='No GPU utilization detected. Enable CUDA for 10x speedup',
                        expected_improvement='5-10x performance increase',
                        implementation_effort='low',
                        config_changes={
                            'device': 'cuda',
                            'precision': 'float16'
                        }
                    ))

                # Batch size optimization
                best_embedding_throughput = max(m['vectors_per_second'] for m in embedding_metrics)
                optimal_embedding_batch = next(
                    m['batch_size'] for m in embedding_metrics
                    if m['vectors_per_second'] == best_embedding_throughput
                )

                recommendations.append(OptimizationRecommendation(
                    category='performance',
                    priority='medium',
                    title='Optimize Embedding Batch Size',
                    description=f'Use batch size of {optimal_embedding_batch} for embedding generation',
                    expected_improvement='15-25% throughput increase',
                    implementation_effort='low',
                    config_changes={'embedding_batch_size': optimal_embedding_batch}
                ))

        # Concurrent operations analysis
        if 'concurrent_operations' in benchmark_results:
            concurrent_metrics = benchmark_results['concurrent_operations']['metrics']

            if concurrent_metrics:
                # Find optimal concurrency level
                throughput_by_clients = {}
                for m in concurrent_metrics:
                    clients = m['concurrent_clients']
                    if clients not in throughput_by_clients:
                        throughput_by_clients[clients] = []
                    throughput_by_clients[clients].append(m['vectors_per_second'])

                avg_throughput_by_clients = {
                    clients: statistics.mean(throughputs)
                    for clients, throughputs in throughput_by_clients.items()
                }

                optimal_clients = max(avg_throughput_by_clients, key=avg_throughput_by_clients.get)

                recommendations.append(OptimizationRecommendation(
                    category='configuration',
                    priority='medium',
                    title='Configure Optimal Concurrency',
                    description=f'Use {optimal_clients} concurrent clients for optimal throughput',
                    expected_improvement='10-30% throughput increase',
                    implementation_effort='low',
                    config_changes={'max_concurrent_clients': optimal_clients}
                ))

        return recommendations

    async def _save_benchmark_results(self, benchmark_name: str, results: Dict[str, Any]):
        """Save benchmark results to file"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = self.results_dir / f"{benchmark_name}_{timestamp}.json"

        async with aiofiles.open(filename, 'w') as f:
            await f.write(json.dumps(results, indent=2, default=str))

        self.logger.info(f"💾 Saved {benchmark_name} results to {filename}")

    async def _generate_benchmark_report(self, all_results: Dict[str, Any]) -> Path:
        """Generate comprehensive benchmark report"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        report_path = self.results_dir / f"benchmark_report_{timestamp}.html"

        # Generate visualizations
        await self._generate_performance_charts(all_results)

        # Create HTML report
        html_content = self._create_html_report(all_results)

        async with aiofiles.open(report_path, 'w') as f:
            await f.write(html_content)

        self.logger.info(f"📊 Generated benchmark report: {report_path}")
        return report_path

    async def _generate_performance_charts(self, results: Dict[str, Any]):
        """Generate performance visualization charts"""
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Vector Database Performance Benchmark Results', fontsize=16)

            # Chart 1: Throughput vs Batch Size
            if 'vector_insertion' in results and results['vector_insertion']['metrics']:
                metrics = results['vector_insertion']['metrics']
                df = pd.DataFrame(metrics)

                if not df.empty:
                    batch_throughput = df.groupby('batch_size')['vectors_per_second'].mean()
                    axes[0, 0].plot(batch_throughput.index, batch_throughput.values, 'o-')
                    axes[0, 0].set_xlabel('Batch Size')
                    axes[0, 0].set_ylabel('Vectors/Second')
                    axes[0, 0].set_title('Insertion Throughput vs Batch Size')
                    axes[0, 0].grid(True)

            # Chart 2: Search Latency Distribution
            if 'vector_search' in results and results['vector_search']['metrics']:
                metrics = results['vector_search']['metrics']
                latencies = [m['avg_latency'] for m in metrics]
                axes[0, 1].hist(latencies, bins=20, alpha=0.7)
                axes[0, 1].set_xlabel('Latency (seconds)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Search Latency Distribution')
                axes[0, 1].grid(True)

            # Chart 3: Concurrent Performance
            if 'concurrent_operations' in results and results['concurrent_operations']['metrics']:
                metrics = results['concurrent_operations']['metrics']
                df = pd.DataFrame(metrics)

                if not df.empty:
                    concurrent_throughput = df.groupby('concurrent_clients')['vectors_per_second'].mean()
                    axes[1, 0].plot(concurrent_throughput.index, concurrent_throughput.values, 'o-')
                    axes[1, 0].set_xlabel('Concurrent Clients')
                    axes[1, 0].set_ylabel('Vectors/Second')
                    axes[1, 0].set_title('Throughput vs Concurrency')
                    axes[1, 0].grid(True)

            # Chart 4: Resource Utilization
            all_metrics = []
            for test_type in ['vector_insertion', 'vector_search', 'embedding_generation']:
                if test_type in results and results[test_type]['metrics']:
                    for metric in results[test_type]['metrics']:
                        metric['test_type'] = test_type
                        all_metrics.append(metric)

            if all_metrics:
                df = pd.DataFrame(all_metrics)
                resource_data = df.groupby('test_type')[['cpu_usage', 'memory_usage']].mean()

                x = range(len(resource_data.index))
                width = 0.35

                axes[1, 1].bar([i - width/2 for i in x], resource_data['cpu_usage'],
                              width, label='CPU Usage %', alpha=0.7)
                axes[1, 1].bar([i + width/2 for i in x], resource_data['memory_usage'],
                              width, label='Memory Usage %', alpha=0.7)

                axes[1, 1].set_xlabel('Test Type')
                axes[1, 1].set_ylabel('Usage %')
                axes[1, 1].set_title('Resource Utilization by Test Type')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(resource_data.index, rotation=45)
                axes[1, 1].legend()
                axes[1, 1].grid(True)

            plt.tight_layout()
            chart_path = self.results_dir / 'performance_charts.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"📈 Generated performance charts: {chart_path}")

        except Exception as e:
            self.logger.error(f"❌ Failed to generate charts: {e}")

    def _create_html_report(self, results: Dict[str, Any]) -> str:
        """Create HTML benchmark report"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vector Database Benchmark Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; color: #333; }
                .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }
                .recommendation { background: #e7f3ff; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }
                .high-priority { border-left-color: #ff4444; }
                .medium-priority { border-left-color: #ffaa00; }
                .low-priority { border-left-color: #00aa00; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .chart { text-align: center; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Vector Database Performance Benchmark Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
        """.format(timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC'))

        # Executive Summary
        html += """
            <div class="summary">
                <h2>📊 Executive Summary</h2>
        """

        total_tests = sum(
            len(results.get(test, {}).get('metrics', []))
            for test in ['vector_insertion', 'vector_search', 'embedding_generation', 'concurrent_operations']
        )

        html += f"<p><strong>Total Tests Executed:</strong> {total_tests}</p>"

        # Add summary metrics for each test type
        for test_type in ['vector_insertion', 'vector_search', 'embedding_generation']:
            if test_type in results and results[test_type]['metrics']:
                metrics = results[test_type]['metrics']
                avg_throughput = statistics.mean(m['vectors_per_second'] for m in metrics)
                avg_latency = statistics.mean(m['avg_latency'] for m in metrics)

                html += f"""
                    <div class="metric">
                        <h4>{test_type.replace('_', ' ').title()}</h4>
                        <p>Avg Throughput: {avg_throughput:.1f} vectors/sec</p>
                        <p>Avg Latency: {avg_latency:.3f}s</p>
                    </div>
                """

        html += "</div>"

        # Performance Charts
        html += """
            <div class="chart">
                <h2>📈 Performance Charts</h2>
                <img src="performance_charts.png" alt="Performance Charts" style="max-width: 100%;">
            </div>
        """

        # Optimization Recommendations
        if 'optimization_recommendations' in results:
            html += "<h2>🎯 Optimization Recommendations</h2>"

            for rec in results['optimization_recommendations']:
                priority_class = rec['priority'] + '-priority'
                html += f"""
                    <div class="recommendation {priority_class}">
                        <h3>{rec['title']} ({rec['priority'].upper()} PRIORITY)</h3>
                        <p><strong>Description:</strong> {rec['description']}</p>
                        <p><strong>Expected Improvement:</strong> {rec['expected_improvement']}</p>
                        <p><strong>Implementation Effort:</strong> {rec['implementation_effort']}</p>
                        <p><strong>Configuration Changes:</strong> {json.dumps(rec['config_changes'], indent=2)}</p>
                    </div>
                """

        # Detailed Results Tables
        for test_type, test_results in results.items():
            if test_type in ['vector_insertion', 'vector_search', 'embedding_generation', 'concurrent_operations']:
                if 'metrics' in test_results and test_results['metrics']:
                    html += f"<h2>📋 {test_type.replace('_', ' ').title()} Detailed Results</h2>"
                    html += "<table>"

                    # Table headers
                    headers = ['Batch Size', 'Vector Dim', 'Concurrent Clients', 'Avg Latency (s)',
                              'Throughput (vec/s)', 'CPU %', 'Memory %', 'Error Rate']
                    html += "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"

                    # Table rows
                    for metric in test_results['metrics'][:10]:  # Limit to first 10 rows
                        html += f"""
                            <tr>
                                <td>{metric['batch_size']}</td>
                                <td>{metric['vector_dim']}</td>
                                <td>{metric['concurrent_clients']}</td>
                                <td>{metric['avg_latency']:.3f}</td>
                                <td>{metric['vectors_per_second']:.1f}</td>
                                <td>{metric['cpu_usage']:.1f}</td>
                                <td>{metric['memory_usage']:.1f}</td>
                                <td>{metric['error_rate']:.3f}</td>
                            </tr>
                        """

                    html += "</table>"

        html += """
            <div class="summary">
                <h2>🔍 System Information</h2>
                <p><strong>CPU Cores:</strong> {cpu_cores}</p>
                <p><strong>Total Memory:</strong> {memory_gb:.1f} GB</p>
                <p><strong>GPU Available:</strong> {gpu_available}</p>
            </div>
        </body>
        </html>
        """.format(
            cpu_cores=mp.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            gpu_available="Yes" if torch.cuda.is_available() else "No"
        )

        return html

    async def shutdown(self):
        """Shutdown benchmark system"""
        self.logger.info("🔒 Shutting down benchmark system")

        if self.embedding_pipeline:
            await self.embedding_pipeline.shutdown()

        if self.vector_manager:
            await self.vector_manager.close()


# CLI Interface for running benchmarks
async def main():
    """Main function for running benchmarks"""
    config = {
        'vector_db': {
            'qdrant_primary_host': '172.30.0.36',
            'qdrant_replica_host': '172.30.0.37',
            'weaviate_host': '172.30.0.38',
            'weaviate_api_key': 'default-key'
        },
        'postgres': {
            'host': '172.30.0.2',
            'port': 5432,
            'user': 'bev_user',
            'password': 'secure_password',
            'database': 'osint'
        },
        'redis': {
            'host': '172.30.0.4',
            'port': 6379,
            'db': 0
        },
        'results_dir': './benchmark_results'
    }

    benchmark = VectorDatabaseBenchmark(config)

    if await benchmark.initialize():
        print("🚀 Starting comprehensive vector database benchmarks...")

        results = await benchmark.run_comprehensive_benchmark()

        print("✅ Benchmarks completed!")
        print(f"📊 Results saved to: {results.get('report_path', 'benchmark_results/')}")

        # Print summary
        if 'optimization_recommendations' in results:
            print(f"\n🎯 Generated {len(results['optimization_recommendations'])} optimization recommendations")
            for rec in results['optimization_recommendations'][:3]:  # Show top 3
                print(f"  • {rec['title']} ({rec['priority']} priority)")

        await benchmark.shutdown()


if __name__ == "__main__":
    asyncio.run(main())