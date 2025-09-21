#!/usr/bin/env python3
"""
Entropy-Based Compression Engine - Advanced Context Compression using Information Theory
Part of the BEV OSINT Context Compression Engine
"""

import asyncio
import json
import logging
import math
import zlib
import gzip
import lzma
import bz2
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import tiktoken
from collections import Counter, defaultdict
import re
import pickle
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompressionBlock:
    """Represents a block of content for compression analysis"""
    id: str
    content: str
    original_size: int
    entropy: float
    redundancy_score: float
    compression_potential: float
    token_count: int
    importance_weight: float = 1.0
    content_type: str = 'text'
    compression_method: Optional[str] = None
    compressed_content: Optional[bytes] = None
    compression_ratio: float = 0.0

@dataclass
class EntropyAnalysis:
    """Results of entropy analysis for content"""
    shannon_entropy: float
    conditional_entropy: float
    mutual_information: float
    compression_complexity: float
    information_density: float
    redundancy_percentage: float
    optimal_compression_ratio: float

@dataclass
class CompressionResult:
    """Results of the compression process"""
    original_blocks: List[CompressionBlock]
    compressed_blocks: List[CompressionBlock]
    total_original_size: int
    total_compressed_size: int
    compression_ratio: float
    entropy_reduction: float
    information_loss_estimate: float
    processing_time: float
    compression_method: str
    quality_metrics: Dict[str, float]

class EntropyCompressor:
    """
    Advanced entropy-based compression engine using information theory principles
    for intelligent context compression with minimal information loss
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_compression_ratio = config.get('target_compression_ratio', 0.5)
        self.max_information_loss = config.get('max_information_loss', 0.05)
        self.block_size = config.get('block_size', 1024)
        self.overlap_size = config.get('overlap_size', 128)

        # Compression methods configuration
        self.compression_methods = {
            'gzip': {'compressor': gzip.compress, 'decompressor': gzip.decompress},
            'lzma': {'compressor': lzma.compress, 'decompressor': lzma.decompress},
            'bz2': {'compressor': bz2.compress, 'decompressor': bz2.decompress},
            'zlib': {'compressor': zlib.compress, 'decompressor': zlib.decompress}
        }

        # Initialize tokenizer for accurate token counting
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Initialize models for semantic compression
        self._initialize_models()

        # Statistics tracking
        self.stats = {
            'total_compressed': 0,
            'total_savings': 0,
            'avg_compression_ratio': 0.0,
            'avg_entropy_reduction': 0.0
        }

    def _initialize_models(self):
        """Initialize ML models for semantic compression"""
        try:
            # TF-IDF for identifying redundant patterns
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 3),
                stop_words='english'
            )

            # SVD for dimensionality reduction
            self.svd = TruncatedSVD(n_components=100)

            logger.info("Initialized entropy compression models")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    async def compress_content(self,
                             content: Union[str, List[str]],
                             compression_ratio: Optional[float] = None,
                             preserve_semantics: bool = True) -> CompressionResult:
        """
        Main compression pipeline using entropy analysis and information theory

        Args:
            content: Content to compress (string or list of strings)
            compression_ratio: Target compression ratio (0.0-1.0)
            preserve_semantics: Whether to preserve semantic meaning

        Returns:
            CompressionResult with compressed content and metrics
        """
        start_time = datetime.now()

        # Use provided ratio or default
        target_ratio = compression_ratio or self.target_compression_ratio

        # Prepare content blocks
        if isinstance(content, str):
            content_list = [content]
        else:
            content_list = content

        # Step 1: Create compression blocks
        blocks = await self._create_compression_blocks(content_list)

        # Step 2: Analyze entropy and redundancy
        await self._analyze_entropy(blocks)

        # Step 3: Determine optimal compression strategy
        compression_strategy = await self._determine_compression_strategy(blocks, target_ratio)

        # Step 4: Apply compression
        compressed_blocks = await self._apply_compression(blocks, compression_strategy, preserve_semantics)

        # Step 5: Quality validation
        quality_metrics = await self._validate_compression_quality(blocks, compressed_blocks)

        # Calculate final metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        result = await self._calculate_compression_metrics(
            blocks, compressed_blocks, processing_time, compression_strategy['method'], quality_metrics
        )

        # Update statistics
        self._update_statistics(result)

        logger.info(f"Compression completed: {result.compression_ratio:.2%} compression ratio, "
                   f"{result.information_loss_estimate:.2%} estimated information loss")

        return result

    async def _create_compression_blocks(self, content_list: List[str]) -> List[CompressionBlock]:
        """Create compression blocks from input content"""
        blocks = []

        for i, content in enumerate(content_list):
            # Split content into manageable blocks
            content_blocks = self._split_content(content)

            for j, block_content in enumerate(content_blocks):
                token_count = len(self.encoding.encode(block_content))

                block = CompressionBlock(
                    id=f"block_{i}_{j}",
                    content=block_content,
                    original_size=len(block_content.encode('utf-8')),
                    entropy=0.0,  # Will be calculated later
                    redundancy_score=0.0,
                    compression_potential=0.0,
                    token_count=token_count,
                    content_type=self._detect_content_type(block_content)
                )
                blocks.append(block)

        return blocks

    def _split_content(self, content: str) -> List[str]:
        """Split content into optimal blocks for compression"""
        if len(content) <= self.block_size:
            return [content]

        blocks = []
        start = 0

        while start < len(content):
            end = min(start + self.block_size, len(content))

            # Try to break at natural boundaries (sentences, paragraphs)
            if end < len(content):
                # Look for sentence endings within overlap region
                overlap_start = max(end - self.overlap_size, start)
                overlap_text = content[overlap_start:end + self.overlap_size]

                # Find best break point
                break_points = [
                    m.end() + overlap_start for m in re.finditer(r'[.!?]\s+', overlap_text)
                ]

                if break_points:
                    # Choose break point closest to target
                    target = end
                    best_break = min(break_points, key=lambda x: abs(x - target))
                    if overlap_start <= best_break <= end + self.overlap_size:
                        end = best_break

            blocks.append(content[start:end].strip())
            start = end

        return [block for block in blocks if block]

    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content for optimal compression strategy"""
        # Simple heuristics for content type detection
        code_patterns = [r'function\s+\w+', r'class\s+\w+', r'import\s+\w+', r'def\s+\w+']
        if any(re.search(pattern, content) for pattern in code_patterns):
            return 'code'

        if re.search(r'^\s*[{[]', content.strip()) and re.search(r'[}\]]\s*$', content.strip()):
            return 'structured'

        if len(content.split()) / len(content) < 0.1:  # Few spaces, likely structured data
            return 'structured'

        return 'text'

    async def _analyze_entropy(self, blocks: List[CompressionBlock]):
        """Analyze entropy and information content of blocks"""
        for block in blocks:
            analysis = await self._calculate_entropy_metrics(block.content)

            block.entropy = analysis.shannon_entropy
            block.redundancy_score = analysis.redundancy_percentage
            block.compression_potential = analysis.optimal_compression_ratio

    async def _calculate_entropy_metrics(self, content: str) -> EntropyAnalysis:
        """Calculate comprehensive entropy metrics for content"""
        # Shannon entropy
        shannon_entropy = self._calculate_shannon_entropy(content)

        # Character-level conditional entropy
        conditional_entropy = self._calculate_conditional_entropy(content)

        # Word-level analysis
        words = content.split()
        word_entropy = self._calculate_shannon_entropy(' '.join(words)) if words else 0

        # Compression complexity (based on structure)
        compression_complexity = self._estimate_compression_complexity(content)

        # Information density
        information_density = shannon_entropy / max(1, len(content))

        # Redundancy percentage
        redundancy_percentage = max(0, 1 - (shannon_entropy / math.log2(256)))

        # Optimal compression ratio estimate
        optimal_ratio = min(0.9, redundancy_percentage + 0.1)

        return EntropyAnalysis(
            shannon_entropy=shannon_entropy,
            conditional_entropy=conditional_entropy,
            mutual_information=max(0, shannon_entropy - conditional_entropy),
            compression_complexity=compression_complexity,
            information_density=information_density,
            redundancy_percentage=redundancy_percentage,
            optimal_compression_ratio=optimal_ratio
        )

    def _calculate_shannon_entropy(self, content: str) -> float:
        """Calculate Shannon entropy of content"""
        if not content:
            return 0.0

        # Count character frequencies
        char_counts = Counter(content)
        total_chars = len(content)

        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def _calculate_conditional_entropy(self, content: str, order: int = 2) -> float:
        """Calculate conditional entropy (Markov chain model)"""
        if len(content) < order + 1:
            return 0.0

        # Build conditional probability model
        context_counts = defaultdict(Counter)

        for i in range(len(content) - order):
            context = content[i:i + order]
            next_char = content[i + order]
            context_counts[context][next_char] += 1

        # Calculate conditional entropy
        conditional_entropy = 0.0
        total_contexts = 0

        for context, next_chars in context_counts.items():
            context_total = sum(next_chars.values())
            context_prob = context_total / (len(content) - order)

            context_entropy = 0.0
            for char, count in next_chars.items():
                char_prob = count / context_total
                context_entropy -= char_prob * math.log2(char_prob)

            conditional_entropy += context_prob * context_entropy
            total_contexts += 1

        return conditional_entropy

    def _estimate_compression_complexity(self, content: str) -> float:
        """Estimate compression complexity based on content structure"""
        # Various complexity indicators

        # Character diversity
        unique_chars = len(set(content))
        char_diversity = unique_chars / max(1, len(content))

        # Pattern repetition
        pattern_score = 0.0
        for pattern_length in [2, 3, 4, 5]:
            patterns = defaultdict(int)
            for i in range(len(content) - pattern_length + 1):
                pattern = content[i:i + pattern_length]
                patterns[pattern] += 1

            if patterns:
                max_repetition = max(patterns.values())
                pattern_score += max_repetition / len(content)

        # Structure indicators
        structure_score = 0.0
        if re.search(r'\n\s*\n', content):  # Paragraphs
            structure_score += 0.1
        if re.search(r'^\s*[-*]\s', content, re.MULTILINE):  # Lists
            structure_score += 0.1
        if re.search(r'[{}\[\]()]', content):  # Brackets
            structure_score += 0.1

        # Combine scores (lower is more compressible)
        complexity = (char_diversity * 0.4 +
                     (1 - pattern_score) * 0.4 +
                     structure_score * 0.2)

        return min(1.0, max(0.0, complexity))

    async def _determine_compression_strategy(self,
                                            blocks: List[CompressionBlock],
                                            target_ratio: float) -> Dict[str, Any]:
        """Determine optimal compression strategy based on content analysis"""

        # Analyze block characteristics
        avg_entropy = np.mean([block.entropy for block in blocks])
        avg_redundancy = np.mean([block.redundancy_score for block in blocks])
        content_types = Counter(block.content_type for block in blocks)

        # Choose compression method based on content
        if content_types.get('code', 0) > len(blocks) * 0.3:
            primary_method = 'lzma'  # Good for code
        elif content_types.get('structured', 0) > len(blocks) * 0.3:
            primary_method = 'gzip'  # Good for structured data
        else:
            primary_method = 'zlib'  # General purpose

        # Determine compression aggressiveness
        if target_ratio > 0.7:  # Aggressive compression
            aggressiveness = 'high'
            semantic_compression = True
        elif target_ratio > 0.4:  # Moderate compression
            aggressiveness = 'medium'
            semantic_compression = True
        else:  # Conservative compression
            aggressiveness = 'low'
            semantic_compression = False

        return {
            'method': primary_method,
            'aggressiveness': aggressiveness,
            'semantic_compression': semantic_compression,
            'target_ratio': target_ratio,
            'estimated_achievable_ratio': min(target_ratio, avg_redundancy + 0.1)
        }

    async def _apply_compression(self,
                               blocks: List[CompressionBlock],
                               strategy: Dict[str, Any],
                               preserve_semantics: bool) -> List[CompressionBlock]:
        """Apply compression using the determined strategy"""
        compressed_blocks = []

        for block in blocks:
            compressed_block = await self._compress_block(block, strategy, preserve_semantics)
            compressed_blocks.append(compressed_block)

        return compressed_blocks

    async def _compress_block(self,
                            block: CompressionBlock,
                            strategy: Dict[str, Any],
                            preserve_semantics: bool) -> CompressionBlock:
        """Compress an individual block"""
        # Try multiple compression methods and choose best
        best_compression = None
        best_ratio = 0.0

        content_bytes = block.content.encode('utf-8')

        for method_name, method_config in self.compression_methods.items():
            try:
                compressed_data = method_config['compressor'](content_bytes)
                compression_ratio = 1 - (len(compressed_data) / len(content_bytes))

                if compression_ratio > best_ratio:
                    best_ratio = compression_ratio
                    best_compression = {
                        'method': method_name,
                        'data': compressed_data,
                        'ratio': compression_ratio
                    }

            except Exception as e:
                logger.warning(f"Compression method {method_name} failed for block {block.id}: {e}")

        # Apply semantic compression if enabled and beneficial
        if preserve_semantics and strategy['semantic_compression']:
            semantic_compressed = await self._apply_semantic_compression(block, strategy)
            if semantic_compressed and semantic_compressed['ratio'] > best_ratio:
                best_compression = semantic_compressed

        # Update block with best compression
        if best_compression:
            compressed_block = CompressionBlock(
                id=block.id,
                content=block.content,  # Keep original for validation
                original_size=block.original_size,
                entropy=block.entropy,
                redundancy_score=block.redundancy_score,
                compression_potential=block.compression_potential,
                token_count=block.token_count,
                importance_weight=block.importance_weight,
                content_type=block.content_type,
                compression_method=best_compression['method'],
                compressed_content=best_compression['data'],
                compression_ratio=best_compression['ratio']
            )
        else:
            # No compression applied
            compressed_block = CompressionBlock(
                id=block.id,
                content=block.content,
                original_size=block.original_size,
                entropy=block.entropy,
                redundancy_score=block.redundancy_score,
                compression_potential=block.compression_potential,
                token_count=block.token_count,
                importance_weight=block.importance_weight,
                content_type=block.content_type,
                compression_method='none',
                compressed_content=content_bytes,
                compression_ratio=0.0
            )

        return compressed_block

    async def _apply_semantic_compression(self,
                                        block: CompressionBlock,
                                        strategy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply semantic-level compression techniques"""
        try:
            content = block.content

            # Remove redundant whitespace
            content = re.sub(r'\s+', ' ', content)
            content = re.sub(r'\n\s*\n', '\n', content)

            # Remove redundant punctuation
            content = re.sub(r'[.]{2,}', '...', content)
            content = re.sub(r'[!]{2,}', '!', content)
            content = re.sub(r'[?]{2,}', '?', content)

            # Compress repetitive patterns
            content = self._compress_repetitive_patterns(content)

            # Apply abbreviations for common terms (if aggressive)
            if strategy['aggressiveness'] == 'high':
                content = self._apply_abbreviations(content)

            compressed_bytes = content.encode('utf-8')
            compression_ratio = 1 - (len(compressed_bytes) / block.original_size)

            if compression_ratio > 0.05:  # Only use if meaningful compression
                return {
                    'method': 'semantic',
                    'data': compressed_bytes,
                    'ratio': compression_ratio
                }

        except Exception as e:
            logger.warning(f"Semantic compression failed for block {block.id}: {e}")

        return None

    def _compress_repetitive_patterns(self, content: str) -> str:
        """Compress repetitive patterns in text"""
        # Find and compress repetitive phrases
        words = content.split()
        if len(words) < 4:
            return content

        # Look for repeated phrases
        compressed_words = []
        i = 0

        while i < len(words):
            # Look for repetitions of 2-5 word phrases
            found_repetition = False

            for phrase_len in range(2, min(6, len(words) - i + 1)):
                phrase = words[i:i + phrase_len]
                phrase_str = ' '.join(phrase)

                # Count consecutive repetitions
                repetitions = 0
                j = i

                while j + phrase_len <= len(words) and words[j:j + phrase_len] == phrase:
                    repetitions += 1
                    j += phrase_len

                # If found 2+ repetitions, compress
                if repetitions >= 2:
                    if repetitions <= 3:
                        compressed_words.extend(phrase * 2)  # Keep 2 instances
                    else:
                        compressed_words.extend(phrase)
                        compressed_words.append(f"[x{repetitions}]")

                    i = j
                    found_repetition = True
                    break

            if not found_repetition:
                compressed_words.append(words[i])
                i += 1

        return ' '.join(compressed_words)

    def _apply_abbreviations(self, content: str) -> str:
        """Apply common abbreviations to reduce content size"""
        abbreviations = {
            ' and ': ' & ',
            ' with ': ' w/ ',
            ' without ': ' w/o ',
            ' through ': ' thru ',
            ' because ': ' bc ',
            ' therefore ': ' ∴ ',
            ' approximately ': ' ~',
            ' approximately ': ' ≈',
            'information': 'info',
            'application': 'app',
            'development': 'dev',
            'configuration': 'config',
            'implementation': 'impl',
            'organization': 'org',
            'administration': 'admin',
            'management': 'mgmt',
            'performance': 'perf'
        }

        for full_form, abbrev in abbreviations.items():
            content = content.replace(full_form, abbrev)

        return content

    async def _validate_compression_quality(self,
                                          original_blocks: List[CompressionBlock],
                                          compressed_blocks: List[CompressionBlock]) -> Dict[str, float]:
        """Validate compression quality and estimate information loss"""

        # Test decompression
        decompression_success_rate = 0.0
        reconstructed_content = []

        for block in compressed_blocks:
            try:
                if block.compression_method != 'none' and block.compressed_content:
                    if block.compression_method == 'semantic':
                        # Semantic compression stores the compressed text directly
                        reconstructed = block.compressed_content.decode('utf-8')
                    else:
                        # Standard compression methods
                        method_config = self.compression_methods[block.compression_method]
                        reconstructed = method_config['decompressor'](block.compressed_content).decode('utf-8')

                    reconstructed_content.append(reconstructed)
                    decompression_success_rate += 1
                else:
                    reconstructed_content.append(block.content)
                    decompression_success_rate += 1

            except Exception as e:
                logger.warning(f"Decompression failed for block {block.id}: {e}")
                reconstructed_content.append("")

        decompression_success_rate /= len(compressed_blocks) if compressed_blocks else 1

        # Calculate content similarity
        original_text = ' '.join(block.content for block in original_blocks)
        reconstructed_text = ' '.join(reconstructed_content)

        similarity_score = self._calculate_text_similarity(original_text, reconstructed_text)

        # Information preservation score
        original_tokens = sum(block.token_count for block in original_blocks)
        reconstructed_tokens = len(self.encoding.encode(reconstructed_text))
        token_preservation = reconstructed_tokens / max(1, original_tokens)

        return {
            'decompression_success_rate': decompression_success_rate,
            'content_similarity': similarity_score,
            'token_preservation': token_preservation,
            'information_preservation': (similarity_score + token_preservation) / 2
        }

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0 if text1 != text2 else 1.0

        # Use character-level similarity for short texts
        if len(text1) < 100 and len(text2) < 100:
            import difflib
            return difflib.SequenceMatcher(None, text1, text2).ratio()

        # Use word-level similarity for longer texts
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / max(1, union)

    async def _calculate_compression_metrics(self,
                                           original_blocks: List[CompressionBlock],
                                           compressed_blocks: List[CompressionBlock],
                                           processing_time: float,
                                           method: str,
                                           quality_metrics: Dict[str, float]) -> CompressionResult:
        """Calculate comprehensive compression metrics"""

        total_original_size = sum(block.original_size for block in original_blocks)
        total_compressed_size = sum(
            len(block.compressed_content) if block.compressed_content else block.original_size
            for block in compressed_blocks
        )

        compression_ratio = 1 - (total_compressed_size / max(1, total_original_size))

        # Entropy reduction
        original_entropy = np.mean([block.entropy for block in original_blocks])
        compressed_entropy = np.mean([block.entropy for block in compressed_blocks])
        entropy_reduction = (original_entropy - compressed_entropy) / max(1, original_entropy)

        # Information loss estimate
        information_loss = 1 - quality_metrics.get('information_preservation', 0.0)

        return CompressionResult(
            original_blocks=original_blocks,
            compressed_blocks=compressed_blocks,
            total_original_size=total_original_size,
            total_compressed_size=total_compressed_size,
            compression_ratio=compression_ratio,
            entropy_reduction=entropy_reduction,
            information_loss_estimate=information_loss,
            processing_time=processing_time,
            compression_method=method,
            quality_metrics=quality_metrics
        )

    def _update_statistics(self, result: CompressionResult):
        """Update internal statistics"""
        self.stats['total_compressed'] += 1
        self.stats['total_savings'] += result.total_original_size - result.total_compressed_size
        self.stats['avg_compression_ratio'] = (
            (self.stats['avg_compression_ratio'] * (self.stats['total_compressed'] - 1) +
             result.compression_ratio) / self.stats['total_compressed']
        )
        self.stats['avg_entropy_reduction'] = (
            (self.stats['avg_entropy_reduction'] * (self.stats['total_compressed'] - 1) +
             result.entropy_reduction) / self.stats['total_compressed']
        )

    async def decompress_content(self, compressed_blocks: List[CompressionBlock]) -> List[str]:
        """Decompress compressed content blocks"""
        decompressed_content = []

        for block in compressed_blocks:
            try:
                if block.compression_method == 'none':
                    decompressed_content.append(block.content)
                elif block.compression_method == 'semantic':
                    # Semantic compression stores modified text directly
                    content = block.compressed_content.decode('utf-8')
                    decompressed_content.append(content)
                elif block.compression_method in self.compression_methods:
                    # Standard compression methods
                    method_config = self.compression_methods[block.compression_method]
                    decompressed = method_config['decompressor'](block.compressed_content)
                    decompressed_content.append(decompressed.decode('utf-8'))
                else:
                    logger.warning(f"Unknown compression method: {block.compression_method}")
                    decompressed_content.append(block.content)

            except Exception as e:
                logger.error(f"Failed to decompress block {block.id}: {e}")
                decompressed_content.append(block.content)  # Fallback to original

        return decompressed_content

    async def get_statistics(self) -> Dict[str, Any]:
        """Get compression statistics"""
        return {
            **self.stats,
            'average_savings_bytes': (
                self.stats['total_savings'] / max(1, self.stats['total_compressed'])
            )
        }

    async def benchmark_compression_methods(self, test_content: str) -> Dict[str, Dict[str, float]]:
        """Benchmark different compression methods on test content"""
        results = {}
        content_bytes = test_content.encode('utf-8')
        original_size = len(content_bytes)

        for method_name, method_config in self.compression_methods.items():
            try:
                start_time = datetime.now()
                compressed = method_config['compressor'](content_bytes)
                compression_time = (datetime.now() - start_time).total_seconds()

                start_time = datetime.now()
                decompressed = method_config['decompressor'](compressed)
                decompression_time = (datetime.now() - start_time).total_seconds()

                compression_ratio = 1 - (len(compressed) / original_size)
                is_lossless = decompressed == content_bytes

                results[method_name] = {
                    'compression_ratio': compression_ratio,
                    'compression_time': compression_time,
                    'decompression_time': decompression_time,
                    'compressed_size': len(compressed),
                    'is_lossless': is_lossless
                }

            except Exception as e:
                results[method_name] = {
                    'error': str(e),
                    'compression_ratio': 0.0,
                    'compression_time': 0.0,
                    'decompression_time': 0.0,
                    'compressed_size': original_size,
                    'is_lossless': False
                }

        return results