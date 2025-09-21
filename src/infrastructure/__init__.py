#!/usr/bin/env python3
"""
BEV OSINT Vector Database Infrastructure
Complete vector database infrastructure for the BEV OSINT framework
Author: BEV OSINT Team
"""

from .vector_db_manager import VectorDatabaseManager, EmbeddingDocument, VectorSearchResult
from .embedding_manager import EmbeddingPipeline, EmbeddingRequest, EmbeddingResponse
from .database_integration import DatabaseIntegrationOrchestrator, IntegratedSearchResult
from .performance_benchmarks import VectorDatabaseBenchmark, PerformanceMetrics, OptimizationRecommendation

__all__ = [
    'VectorDatabaseManager',
    'EmbeddingDocument',
    'VectorSearchResult',
    'EmbeddingPipeline',
    'EmbeddingRequest',
    'EmbeddingResponse',
    'DatabaseIntegrationOrchestrator',
    'IntegratedSearchResult',
    'VectorDatabaseBenchmark',
    'PerformanceMetrics',
    'OptimizationRecommendation'
]

__version__ = "1.0.0"