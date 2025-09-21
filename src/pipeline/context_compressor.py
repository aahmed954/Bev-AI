#!/usr/bin/env python3
"""
Context Compression Engine - Comprehensive Context Compression with Information Loss Monitoring
Main coordinator for the BEV OSINT Context Compression system
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import tiktoken
from enum import Enum
import hashlib
import redis
from pymongo import MongoClient
import qdrant_client
from qdrant_client.models import Distance, VectorParams, PointStruct
import traceback

# Import our compression components
from .semantic_deduplicator import SemanticDeduplicator, DeduplicationResult, ContentFragment
from .entropy_compressor import EntropyCompressor, CompressionResult, CompressionBlock
from .quality_validator import QualityValidator, ValidationResult, ValidationConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompressionStrategy(Enum):
    """Available compression strategies"""
    CONSERVATIVE = "conservative"  # Minimal compression, maximum quality preservation
    BALANCED = "balanced"         # Balanced compression and quality
    AGGRESSIVE = "aggressive"     # Maximum compression, acceptable quality loss
    SEMANTIC_ONLY = "semantic_only"  # Only semantic deduplication
    ENTROPY_ONLY = "entropy_only"   # Only entropy-based compression

@dataclass
class CompressionConfig:
    """Configuration for compression pipeline"""
    strategy: CompressionStrategy = CompressionStrategy.BALANCED
    target_compression_ratio: float = 0.4  # 40% compression target
    max_information_loss: float = 0.05      # 5% max information loss
    preserve_semantics: bool = True
    enable_caching: bool = True
    vector_db_integration: bool = True
    quality_validation: bool = True

    # Semantic deduplication settings
    semantic_similarity_threshold: float = 0.85
    semantic_clustering_eps: float = 0.3

    # Entropy compression settings
    entropy_block_size: int = 1024
    entropy_target_ratio: float = 0.5

    # Quality thresholds
    min_similarity_score: float = 0.95
    min_coherence_score: float = 0.8

@dataclass
class CompressionMetrics:
    """Comprehensive metrics for compression operation"""
    original_size: int
    compressed_size: int
    original_token_count: int
    compressed_token_count: int
    compression_ratio: float
    token_compression_ratio: float

    # Quality metrics
    information_loss_score: float
    semantic_similarity_score: float
    coherence_score: float
    reconstruction_accuracy: float

    # Performance metrics
    processing_time: float
    deduplication_time: float
    compression_time: float
    validation_time: float

    # Component results
    deduplication_result: Optional[DeduplicationResult] = None
    entropy_result: Optional[CompressionResult] = None
    validation_result: Optional['ValidationResult'] = None

@dataclass
class ContextCompressionResult:
    """Complete result of context compression"""
    compressed_content: List[str]
    original_content: List[str]
    compression_metadata: Dict[str, Any]
    metrics: CompressionMetrics
    recovery_data: Dict[str, Any]  # Data needed for decompression
    cache_key: Optional[str] = None

class ContextCompressor:
    """
    Main context compression engine that orchestrates semantic deduplication,
    entropy compression, and quality validation for optimal context compression
    """

    def __init__(self, config: CompressionConfig, infrastructure_config: Dict[str, Any]):
        self.config = config
        self.infrastructure_config = infrastructure_config

        # Initialize tokenizer
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize storage clients
        self._initialize_storage()

        # Initialize compression components
        self._initialize_components()

        # Statistics tracking
        self.stats = {
            'total_compressions': 0,
            'total_bytes_processed': 0,
            'total_bytes_saved': 0,
            'avg_compression_ratio': 0.0,
            'avg_information_loss': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def _initialize_storage(self):
        """Initialize storage connections"""
        try:
            # Redis for caching
            if self.config.enable_caching:
                self.redis_client = redis.Redis(
                    host=self.infrastructure_config.get('redis_host', 'localhost'),
                    port=self.infrastructure_config.get('redis_port', 6379),
                    decode_responses=True
                )

            # Vector database for semantic operations
            if self.config.vector_db_integration:
                self.vector_client = qdrant_client.QdrantClient(
                    host=self.infrastructure_config.get('qdrant_host', 'localhost'),
                    port=self.infrastructure_config.get('qdrant_port', 6333)
                )

                # Ensure compression collection exists
                try:
                    self.vector_client.create_collection(
                        collection_name="compression_cache",
                        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                    )
                except Exception:
                    pass  # Collection might already exist

            # MongoDB for metadata storage
            self.mongo_client = MongoClient(
                self.infrastructure_config.get('mongodb_url', 'mongodb://localhost:27017/')
            )
            self.compression_db = self.mongo_client.bev_compression

            logger.info("Storage connections initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize storage: {e}")
            raise

    def _initialize_components(self):
        """Initialize compression components"""
        try:
            # Semantic deduplicator configuration
            dedup_config = {
                'similarity_threshold': self.config.semantic_similarity_threshold,
                'cluster_eps': self.config.semantic_clustering_eps,
                'redis_host': self.infrastructure_config.get('redis_host', 'localhost'),
                'redis_port': self.infrastructure_config.get('redis_port', 6379)
            }
            self.semantic_deduplicator = SemanticDeduplicator(dedup_config)

            # Entropy compressor configuration
            entropy_config = {
                'target_compression_ratio': self.config.entropy_target_ratio,
                'block_size': self.config.entropy_block_size,
                'max_information_loss': self.config.max_information_loss
            }
            self.entropy_compressor = EntropyCompressor(entropy_config)

            # Quality validator configuration
            quality_config = ValidationConfig(
                min_information_preservation=self.config.min_information_loss,
                min_semantic_similarity=self.config.min_similarity_score,
                min_structural_coherence=self.config.min_coherence_score,
                enable_deep_analysis=True,
                enable_linguistic_analysis=True,
                enable_semantic_analysis=True,
                enable_structural_analysis=True
            )
            self.quality_validator = QualityValidator(quality_config)

            logger.info("Compression components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize compression components: {e}")
            raise

    async def compress_context(self,
                             content: Union[str, List[str], List[Dict[str, Any]]],
                             context_id: Optional[str] = None,
                             strategy_override: Optional[CompressionStrategy] = None) -> ContextCompressionResult:
        """
        Main context compression pipeline

        Args:
            content: Content to compress (various formats supported)
            context_id: Optional identifier for caching
            strategy_override: Override default compression strategy

        Returns:
            ContextCompressionResult with compressed content and metrics
        """
        start_time = time.time()

        # Use strategy override if provided
        active_strategy = strategy_override or self.config.strategy

        try:
            # Step 1: Prepare and normalize content
            normalized_content = await self._normalize_content(content)
            original_size = sum(len(c.encode('utf-8')) for c in normalized_content)
            original_tokens = sum(len(self.encoding.encode(c)) for c in normalized_content)

            logger.info(f"Starting compression: {len(normalized_content)} items, "
                       f"{original_size} bytes, {original_tokens} tokens")

            # Step 2: Check cache if enabled
            cache_key = None
            if self.config.enable_caching and context_id:
                cache_key = self._generate_cache_key(normalized_content, active_strategy)
                cached_result = await self._check_cache(cache_key)
                if cached_result:
                    self.stats['cache_hits'] += 1
                    logger.info(f"Cache hit for key: {cache_key}")
                    return cached_result
                self.stats['cache_misses'] += 1

            # Step 3: Apply compression strategy
            compression_result = await self._apply_compression_strategy(
                normalized_content, active_strategy
            )

            # Step 4: Calculate comprehensive metrics
            metrics = await self._calculate_comprehensive_metrics(
                normalized_content, compression_result, start_time
            )

            # Step 5: Quality validation
            if self.config.quality_validation:
                validation_start = time.time()
                validation_result = await self._validate_compression_quality(
                    normalized_content, compression_result, metrics
                )
                metrics.validation_time = time.time() - validation_start

                # Apply quality-based adjustments if needed
                if not validation_result.passes_quality_thresholds:
                    logger.warning("Compression failed quality validation, applying fallback")
                    compression_result = await self._apply_fallback_compression(
                        normalized_content, validation_result
                    )
                    metrics = await self._calculate_comprehensive_metrics(
                        normalized_content, compression_result, start_time
                    )

            # Step 6: Create final result
            final_result = ContextCompressionResult(
                compressed_content=compression_result['compressed_content'],
                original_content=normalized_content,
                compression_metadata=compression_result['metadata'],
                metrics=metrics,
                recovery_data=compression_result['recovery_data'],
                cache_key=cache_key
            )

            # Step 7: Cache result if enabled
            if self.config.enable_caching and cache_key:
                await self._cache_result(cache_key, final_result)

            # Step 8: Update statistics
            await self._update_statistics(metrics)

            logger.info(f"Compression completed: {metrics.compression_ratio:.2%} ratio, "
                       f"{metrics.information_loss_score:.2%} information loss, "
                       f"{metrics.processing_time:.2f}s")

            return final_result

        except Exception as e:
            logger.error(f"Compression failed: {e}")
            logger.error(traceback.format_exc())
            raise

    async def _normalize_content(self, content: Union[str, List[str], List[Dict[str, Any]]]) -> List[str]:
        """Normalize input content to standard format"""
        if isinstance(content, str):
            return [content]
        elif isinstance(content, list):
            normalized = []
            for item in content:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict):
                    # Extract content from dictionary
                    if 'content' in item:
                        normalized.append(str(item['content']))
                    elif 'text' in item:
                        normalized.append(str(item['text']))
                    else:
                        normalized.append(json.dumps(item))
                else:
                    normalized.append(str(item))
            return normalized
        else:
            return [str(content)]

    def _generate_cache_key(self, content: List[str], strategy: CompressionStrategy) -> str:
        """Generate cache key for content and strategy"""
        content_hash = hashlib.sha256(
            ''.join(content).encode('utf-8')
        ).hexdigest()[:16]

        strategy_hash = hashlib.md5(
            f"{strategy.value}_{self.config.target_compression_ratio}_{self.config.max_information_loss}".encode()
        ).hexdigest()[:8]

        return f"compress_{content_hash}_{strategy_hash}"

    async def _check_cache(self, cache_key: str) -> Optional[ContextCompressionResult]:
        """Check if compression result is cached"""
        try:
            cached_data = self.redis_client.get(f"compression:{cache_key}")
            if cached_data:
                return self._deserialize_result(json.loads(cached_data))
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")
        return None

    async def _apply_compression_strategy(self,
                                        content: List[str],
                                        strategy: CompressionStrategy) -> Dict[str, Any]:
        """Apply the specified compression strategy"""

        if strategy == CompressionStrategy.SEMANTIC_ONLY:
            return await self._apply_semantic_only_compression(content)
        elif strategy == CompressionStrategy.ENTROPY_ONLY:
            return await self._apply_entropy_only_compression(content)
        elif strategy == CompressionStrategy.CONSERVATIVE:
            return await self._apply_conservative_compression(content)
        elif strategy == CompressionStrategy.BALANCED:
            return await self._apply_balanced_compression(content)
        elif strategy == CompressionStrategy.AGGRESSIVE:
            return await self._apply_aggressive_compression(content)
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")

    async def _apply_semantic_only_compression(self, content: List[str]) -> Dict[str, Any]:
        """Apply only semantic deduplication"""
        dedup_start = time.time()

        # Prepare content for semantic deduplication
        content_fragments = [
            {
                'id': f'fragment_{i}',
                'content': text,
                'type': 'text',
                'source': 'context',
                'timestamp': datetime.now().isoformat(),
                'importance': 1.0
            }
            for i, text in enumerate(content)
        ]

        # Perform semantic deduplication
        dedup_result = await self.semantic_deduplicator.deduplicate_content(content_fragments)

        dedup_time = time.time() - dedup_start

        # Extract compressed content
        compressed_content = [f.content for f in dedup_result.deduplicated_fragments]

        return {
            'compressed_content': compressed_content,
            'metadata': {
                'strategy': 'semantic_only',
                'deduplication_clusters': dedup_result.similarity_clusters,
                'deduplication_time': dedup_time
            },
            'recovery_data': {
                'original_fragment_mapping': {f.id: f.content for f in dedup_result.original_fragments},
                'deduplication_result': dedup_result
            }
        }

    async def _apply_entropy_only_compression(self, content: List[str]) -> Dict[str, Any]:
        """Apply only entropy-based compression"""
        compression_start = time.time()

        # Apply entropy compression
        entropy_result = await self.entropy_compressor.compress_content(
            content,
            compression_ratio=self.config.target_compression_ratio,
            preserve_semantics=self.config.preserve_semantics
        )

        compression_time = time.time() - compression_start

        # Decompress for final content
        compressed_content = await self.entropy_compressor.decompress_content(
            entropy_result.compressed_blocks
        )

        return {
            'compressed_content': compressed_content,
            'metadata': {
                'strategy': 'entropy_only',
                'compression_method': entropy_result.compression_method,
                'compression_time': compression_time
            },
            'recovery_data': {
                'compressed_blocks': entropy_result.compressed_blocks,
                'entropy_result': entropy_result
            }
        }

    async def _apply_conservative_compression(self, content: List[str]) -> Dict[str, Any]:
        """Apply conservative compression with minimal information loss"""
        # Use lower thresholds for conservative approach
        conservative_config = {
            'semantic_similarity_threshold': 0.95,  # Very high similarity required
            'target_compression_ratio': 0.2,       # Lower compression target
            'preserve_semantics': True
        }

        return await self._apply_two_stage_compression(content, conservative_config)

    async def _apply_balanced_compression(self, content: List[str]) -> Dict[str, Any]:
        """Apply balanced compression with moderate compression and quality"""
        balanced_config = {
            'semantic_similarity_threshold': self.config.semantic_similarity_threshold,
            'target_compression_ratio': self.config.target_compression_ratio,
            'preserve_semantics': self.config.preserve_semantics
        }

        return await self._apply_two_stage_compression(content, balanced_config)

    async def _apply_aggressive_compression(self, content: List[str]) -> Dict[str, Any]:
        """Apply aggressive compression with maximum compression ratio"""
        aggressive_config = {
            'semantic_similarity_threshold': 0.75,  # Lower similarity threshold
            'target_compression_ratio': 0.6,       # Higher compression target
            'preserve_semantics': True
        }

        return await self._apply_two_stage_compression(content, aggressive_config)

    async def _apply_two_stage_compression(self, content: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply two-stage compression: semantic deduplication followed by entropy compression"""

        # Stage 1: Semantic deduplication
        dedup_start = time.time()

        content_fragments = [
            {
                'id': f'fragment_{i}',
                'content': text,
                'type': 'text',
                'source': 'context',
                'timestamp': datetime.now().isoformat(),
                'importance': 1.0
            }
            for i, text in enumerate(content)
        ]

        # Temporarily adjust deduplicator threshold
        original_threshold = self.semantic_deduplicator.similarity_threshold
        self.semantic_deduplicator.similarity_threshold = config['semantic_similarity_threshold']

        dedup_result = await self.semantic_deduplicator.deduplicate_content(content_fragments)

        # Restore original threshold
        self.semantic_deduplicator.similarity_threshold = original_threshold

        dedup_time = time.time() - dedup_start

        # Stage 2: Entropy compression on deduplicated content
        compression_start = time.time()

        deduplicated_content = [f.content for f in dedup_result.deduplicated_fragments]

        entropy_result = await self.entropy_compressor.compress_content(
            deduplicated_content,
            compression_ratio=config['target_compression_ratio'],
            preserve_semantics=config['preserve_semantics']
        )

        compression_time = time.time() - compression_start

        # Get final compressed content
        final_compressed = await self.entropy_compressor.decompress_content(
            entropy_result.compressed_blocks
        )

        return {
            'compressed_content': final_compressed,
            'metadata': {
                'strategy': 'two_stage',
                'deduplication_clusters': dedup_result.similarity_clusters,
                'compression_method': entropy_result.compression_method,
                'deduplication_time': dedup_time,
                'compression_time': compression_time,
                'stage1_compression': dedup_result.compression_ratio,
                'stage2_compression': entropy_result.compression_ratio
            },
            'recovery_data': {
                'deduplication_result': dedup_result,
                'entropy_result': entropy_result,
                'stage1_mapping': {f.id: f.content for f in dedup_result.original_fragments}
            }
        }

    async def _calculate_comprehensive_metrics(self,
                                             original_content: List[str],
                                             compression_result: Dict[str, Any],
                                             start_time: float) -> CompressionMetrics:
        """Calculate comprehensive metrics for compression operation"""

        compressed_content = compression_result['compressed_content']

        # Size metrics
        original_size = sum(len(c.encode('utf-8')) for c in original_content)
        compressed_size = sum(len(c.encode('utf-8')) for c in compressed_content)
        compression_ratio = 1 - (compressed_size / max(1, original_size))

        # Token metrics
        original_tokens = sum(len(self.encoding.encode(c)) for c in original_content)
        compressed_tokens = sum(len(self.encoding.encode(c)) for c in compressed_content)
        token_compression_ratio = 1 - (compressed_tokens / max(1, original_tokens))

        # Performance metrics
        total_time = time.time() - start_time
        dedup_time = compression_result['metadata'].get('deduplication_time', 0.0)
        comp_time = compression_result['metadata'].get('compression_time', 0.0)

        # Quality metrics (basic estimates - will be refined by quality validator)
        information_loss = await self._estimate_information_loss(original_content, compressed_content)
        semantic_similarity = await self._calculate_semantic_similarity(original_content, compressed_content)
        coherence_score = await self._calculate_coherence(compressed_content)

        return CompressionMetrics(
            original_size=original_size,
            compressed_size=compressed_size,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compression_ratio,
            token_compression_ratio=token_compression_ratio,
            information_loss_score=information_loss,
            semantic_similarity_score=semantic_similarity,
            coherence_score=coherence_score,
            reconstruction_accuracy=1.0 - information_loss,  # Estimate
            processing_time=total_time,
            deduplication_time=dedup_time,
            compression_time=comp_time,
            validation_time=0.0  # Will be set later if validation is performed
        )

    async def _estimate_information_loss(self, original: List[str], compressed: List[str]) -> float:
        """Estimate information loss between original and compressed content"""
        if not original or not compressed:
            return 1.0 if not compressed else 0.0

        # Simple token-based estimation
        original_tokens = set()
        for content in original:
            original_tokens.update(self.encoding.encode(content))

        compressed_tokens = set()
        for content in compressed:
            compressed_tokens.update(self.encoding.encode(content))

        if not original_tokens:
            return 0.0

        preserved_tokens = len(original_tokens.intersection(compressed_tokens))
        loss_ratio = 1 - (preserved_tokens / len(original_tokens))

        return max(0.0, min(1.0, loss_ratio))

    async def _calculate_semantic_similarity(self, original: List[str], compressed: List[str]) -> float:
        """Calculate semantic similarity between original and compressed content"""
        if not original or not compressed:
            return 0.0 if original != compressed else 1.0

        # Use word overlap as a simple similarity metric
        original_words = set()
        for content in original:
            original_words.update(content.lower().split())

        compressed_words = set()
        for content in compressed:
            compressed_words.update(content.lower().split())

        if not original_words:
            return 1.0 if not compressed_words else 0.0

        intersection = len(original_words.intersection(compressed_words))
        union = len(original_words.union(compressed_words))

        return intersection / max(1, union)

    async def _calculate_coherence(self, content: List[str]) -> float:
        """Calculate coherence score of content"""
        if len(content) < 2:
            return 1.0

        # Simple coherence metric based on word overlap between adjacent items
        coherence_scores = []

        for i in range(len(content) - 1):
            words1 = set(content[i].lower().split())
            words2 = set(content[i + 1].lower().split())

            if not words1 or not words2:
                coherence_scores.append(0.0)
                continue

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            coherence_scores.append(intersection / max(1, union))

        return np.mean(coherence_scores) if coherence_scores else 1.0

    async def _validate_compression_quality(self,
                                          original: List[str],
                                          compression_result: Dict[str, Any],
                                          metrics: CompressionMetrics) -> ValidationResult:
        """Validate compression quality using QualityValidator"""
        try:
            compressed_content = compression_result['compressed_content']
            validation_result = await self.quality_validator.validate_compression(
                original,
                compressed_content,
                compression_result['metadata']
            )
            return validation_result

        except Exception as e:
            logger.warning(f"Quality validation failed: {e}, using fallback validation")

            # Fallback validation
            class FallbackValidationResult:
                def __init__(self):
                    self.passes_quality_thresholds = (
                        metrics.information_loss_score <= self.config.max_information_loss and
                        metrics.semantic_similarity_score >= self.config.min_similarity_score and
                        metrics.coherence_score >= self.config.min_coherence_score
                    )
                    self.quality_score = (
                        metrics.semantic_similarity_score * 0.4 +
                        metrics.coherence_score * 0.3 +
                        (1 - metrics.information_loss_score) * 0.3
                    )
                    self.recommendations = []

                    if not self.passes_quality_thresholds:
                        if metrics.information_loss_score > self.config.max_information_loss:
                            self.recommendations.append("Reduce compression ratio to preserve more information")
                        if metrics.semantic_similarity_score < self.config.min_similarity_score:
                            self.recommendations.append("Increase semantic similarity threshold")
                        if metrics.coherence_score < self.config.min_coherence_score:
                            self.recommendations.append("Improve content coherence preservation")

            return FallbackValidationResult()

    async def _apply_fallback_compression(self,
                                        content: List[str],
                                        validation_result: 'ValidationResult') -> Dict[str, Any]:
        """Apply fallback compression strategy when quality validation fails"""
        logger.info("Applying fallback compression strategy")

        # Use conservative strategy as fallback
        return await self._apply_conservative_compression(content)

    async def _cache_result(self, cache_key: str, result: ContextCompressionResult):
        """Cache compression result"""
        try:
            serialized = self._serialize_result(result)
            self.redis_client.setex(
                f"compression:{cache_key}",
                3600,  # 1 hour TTL
                json.dumps(serialized)
            )
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")

    def _serialize_result(self, result: ContextCompressionResult) -> Dict[str, Any]:
        """Serialize compression result for caching"""
        return {
            'compressed_content': result.compressed_content,
            'original_content': result.original_content,
            'compression_metadata': result.compression_metadata,
            'metrics': {
                'original_size': result.metrics.original_size,
                'compressed_size': result.metrics.compressed_size,
                'original_token_count': result.metrics.original_token_count,
                'compressed_token_count': result.metrics.compressed_token_count,
                'compression_ratio': result.metrics.compression_ratio,
                'token_compression_ratio': result.metrics.token_compression_ratio,
                'information_loss_score': result.metrics.information_loss_score,
                'semantic_similarity_score': result.metrics.semantic_similarity_score,
                'coherence_score': result.metrics.coherence_score,
                'reconstruction_accuracy': result.metrics.reconstruction_accuracy,
                'processing_time': result.metrics.processing_time,
                'deduplication_time': result.metrics.deduplication_time,
                'compression_time': result.metrics.compression_time,
                'validation_time': result.metrics.validation_time
            },
            'recovery_data': result.recovery_data,
            'cache_key': result.cache_key
        }

    def _deserialize_result(self, data: Dict[str, Any]) -> ContextCompressionResult:
        """Deserialize compression result from cache"""
        metrics_data = data['metrics']
        metrics = CompressionMetrics(
            original_size=metrics_data['original_size'],
            compressed_size=metrics_data['compressed_size'],
            original_token_count=metrics_data['original_token_count'],
            compressed_token_count=metrics_data['compressed_token_count'],
            compression_ratio=metrics_data['compression_ratio'],
            token_compression_ratio=metrics_data['token_compression_ratio'],
            information_loss_score=metrics_data['information_loss_score'],
            semantic_similarity_score=metrics_data['semantic_similarity_score'],
            coherence_score=metrics_data['coherence_score'],
            reconstruction_accuracy=metrics_data['reconstruction_accuracy'],
            processing_time=metrics_data['processing_time'],
            deduplication_time=metrics_data['deduplication_time'],
            compression_time=metrics_data['compression_time'],
            validation_time=metrics_data['validation_time']
        )

        return ContextCompressionResult(
            compressed_content=data['compressed_content'],
            original_content=data['original_content'],
            compression_metadata=data['compression_metadata'],
            metrics=metrics,
            recovery_data=data['recovery_data'],
            cache_key=data['cache_key']
        )

    async def _update_statistics(self, metrics: CompressionMetrics):
        """Update internal statistics"""
        self.stats['total_compressions'] += 1
        self.stats['total_bytes_processed'] += metrics.original_size
        self.stats['total_bytes_saved'] += (metrics.original_size - metrics.compressed_size)

        # Update running averages
        total = self.stats['total_compressions']
        self.stats['avg_compression_ratio'] = (
            (self.stats['avg_compression_ratio'] * (total - 1) + metrics.compression_ratio) / total
        )
        self.stats['avg_information_loss'] = (
            (self.stats['avg_information_loss'] * (total - 1) + metrics.information_loss_score) / total
        )

    async def decompress_context(self, compression_result: ContextCompressionResult) -> List[str]:
        """Decompress context using recovery data"""
        try:
            recovery_data = compression_result.recovery_data

            # Check if we have entropy compression data
            if 'entropy_result' in recovery_data:
                entropy_result = recovery_data['entropy_result']
                return await self.entropy_compressor.decompress_content(entropy_result.compressed_blocks)

            # Check if we have deduplication data only
            elif 'deduplication_result' in recovery_data:
                dedup_result = recovery_data['deduplication_result']
                return [f.content for f in dedup_result.original_fragments]

            # Fallback to compressed content
            else:
                logger.warning("No recovery data found, returning compressed content")
                return compression_result.compressed_content

        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return compression_result.compressed_content

    async def get_statistics(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return {
            **self.stats,
            'compression_efficiency': (
                self.stats['total_bytes_saved'] / max(1, self.stats['total_bytes_processed'])
            ),
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
            )
        }

    async def analyze_content_complexity(self, content: List[str]) -> Dict[str, Any]:
        """Analyze content complexity and compression potential"""
        if not content:
            return {}

        # Basic content analysis
        total_size = sum(len(c.encode('utf-8')) for c in content)
        total_tokens = sum(len(self.encoding.encode(c)) for c in content)
        avg_length = np.mean([len(c) for c in content])

        # Estimate compression potential
        compression_potential = await self._estimate_compression_potential(content)

        # Content diversity analysis
        diversity_score = await self._analyze_content_diversity(content)

        return {
            'content_count': len(content),
            'total_size_bytes': total_size,
            'total_token_count': total_tokens,
            'average_content_length': avg_length,
            'estimated_compression_potential': compression_potential,
            'content_diversity_score': diversity_score,
            'recommended_strategy': self._recommend_strategy(compression_potential, diversity_score)
        }

    async def _estimate_compression_potential(self, content: List[str]) -> float:
        """Estimate compression potential of content"""
        if not content:
            return 0.0

        # Analyze redundancy and repetition
        all_text = ' '.join(content)
        unique_chars = len(set(all_text))
        total_chars = len(all_text)

        char_diversity = unique_chars / max(1, total_chars)

        # Analyze word repetition
        words = all_text.split()
        unique_words = len(set(words))
        total_words = len(words)

        word_diversity = unique_words / max(1, total_words)

        # Compression potential is inverse of diversity
        potential = 1 - (char_diversity * 0.3 + word_diversity * 0.7)

        return max(0.0, min(1.0, potential))

    async def _analyze_content_diversity(self, content: List[str]) -> float:
        """Analyze content diversity"""
        if len(content) < 2:
            return 1.0

        # Calculate pairwise similarities
        similarities = []

        for i in range(len(content)):
            for j in range(i + 1, len(content)):
                similarity = await self._calculate_semantic_similarity([content[i]], [content[j]])
                similarities.append(similarity)

        # Diversity is inverse of average similarity
        avg_similarity = np.mean(similarities) if similarities else 0.0
        diversity = 1 - avg_similarity

        return max(0.0, min(1.0, diversity))

    def _recommend_strategy(self, compression_potential: float, diversity_score: float) -> str:
        """Recommend compression strategy based on content analysis"""
        if compression_potential > 0.7 and diversity_score < 0.3:
            return "aggressive"  # High redundancy, low diversity
        elif compression_potential > 0.4 and diversity_score < 0.5:
            return "balanced"   # Moderate redundancy and diversity
        elif compression_potential < 0.2 or diversity_score > 0.8:
            return "conservative"  # Low redundancy or high diversity
        else:
            return "balanced"   # Default recommendation