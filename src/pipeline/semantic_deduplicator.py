#!/usr/bin/env python3
"""
Semantic Deduplicator - Advanced Content Deduplication using Embeddings and Similarity Analysis
Part of the BEV OSINT Context Compression Engine
"""

import asyncio
import hashlib
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import spacy
import redis
from pymongo import MongoClient
import tiktoken
import difflib
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContentFragment:
    """Represents a fragment of content with metadata and embeddings"""
    id: str
    content: str
    content_type: str  # 'text', 'code', 'structured', 'metadata'
    source: str
    timestamp: datetime
    token_count: int
    hash_md5: str = field(init=False)
    embedding: Optional[np.ndarray] = None
    tfidf_vector: Optional[np.ndarray] = None
    semantic_cluster: Optional[int] = None
    importance_score: float = 1.0
    similarity_threshold: float = 0.85

    def __post_init__(self):
        self.hash_md5 = hashlib.md5(self.content.encode()).hexdigest()

@dataclass
class DeduplicationResult:
    """Results of the deduplication process"""
    original_fragments: List[ContentFragment]
    deduplicated_fragments: List[ContentFragment]
    removed_duplicates: List[ContentFragment]
    similarity_clusters: Dict[int, List[str]]  # cluster_id -> fragment_ids
    compression_ratio: float
    information_loss_score: float
    semantic_coherence_score: float
    processing_time: float

class SemanticDeduplicator:
    """
    Advanced semantic deduplication engine using multiple similarity metrics
    and clustering algorithms for intelligent content compression
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.similarity_threshold = config.get('similarity_threshold', 0.85)
        self.semantic_model_name = config.get('semantic_model', 'all-MiniLM-L6-v2')
        self.cluster_eps = config.get('cluster_eps', 0.3)
        self.min_samples = config.get('min_samples', 2)
        self.preserve_unique_threshold = config.get('preserve_unique_threshold', 0.7)

        # Initialize models
        self._initialize_models()

        # Initialize storage
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            decode_responses=True
        )

        # Token counter
        self.encoding = tiktoken.get_encoding("cl100k_base")

        # Statistics
        self.stats = {
            'total_processed': 0,
            'duplicates_removed': 0,
            'clusters_formed': 0,
            'total_compression_ratio': 0.0
        }

    def _initialize_models(self):
        """Initialize semantic models and NLP components"""
        try:
            # Sentence transformer for semantic embeddings
            self.semantic_model = SentenceTransformer(self.semantic_model_name)

            # TF-IDF vectorizer for content similarity
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )

            # SpaCy for advanced text processing
            self.nlp = spacy.load("en_core_web_sm")

            logger.info(f"Initialized semantic models: {self.semantic_model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    async def deduplicate_content(self,
                                 content_fragments: List[Dict[str, Any]],
                                 preserve_order: bool = True) -> DeduplicationResult:
        """
        Main deduplication pipeline with semantic analysis and clustering

        Args:
            content_fragments: List of content dictionaries
            preserve_order: Whether to maintain original ordering

        Returns:
            DeduplicationResult with deduplicated content and metrics
        """
        start_time = datetime.now()

        # Convert to ContentFragment objects
        fragments = await self._prepare_fragments(content_fragments)

        # Step 1: Hash-based exact duplicate removal
        fragments = await self._remove_exact_duplicates(fragments)

        # Step 2: Generate embeddings and similarity vectors
        await self._generate_embeddings(fragments)

        # Step 3: Perform semantic clustering
        clusters = await self._perform_semantic_clustering(fragments)

        # Step 4: Intelligent deduplication within clusters
        deduplicated_fragments = await self._deduplicate_clusters(clusters, fragments)

        # Step 5: Quality assessment and final optimization
        final_fragments = await self._optimize_final_selection(deduplicated_fragments, preserve_order)

        # Calculate metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        result = await self._calculate_metrics(
            fragments, final_fragments, clusters, processing_time
        )

        # Update statistics
        self._update_statistics(result)

        logger.info(f"Deduplication completed: {len(fragments)} -> {len(final_fragments)} fragments "
                   f"({result.compression_ratio:.2%} compression)")

        return result

    async def _prepare_fragments(self, content_data: List[Dict[str, Any]]) -> List[ContentFragment]:
        """Convert input data to ContentFragment objects"""
        fragments = []

        for i, data in enumerate(content_data):
            content = data.get('content', str(data))

            # Count tokens
            token_count = len(self.encoding.encode(content))

            fragment = ContentFragment(
                id=data.get('id', f"fragment_{i}"),
                content=content,
                content_type=data.get('type', 'text'),
                source=data.get('source', 'unknown'),
                timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                token_count=token_count,
                importance_score=data.get('importance', 1.0)
            )
            fragments.append(fragment)

        return fragments

    async def _remove_exact_duplicates(self, fragments: List[ContentFragment]) -> List[ContentFragment]:
        """Remove exact duplicates based on content hash"""
        seen_hashes = set()
        unique_fragments = []

        for fragment in fragments:
            if fragment.hash_md5 not in seen_hashes:
                seen_hashes.add(fragment.hash_md5)
                unique_fragments.append(fragment)

        removed_count = len(fragments) - len(unique_fragments)
        if removed_count > 0:
            logger.info(f"Removed {removed_count} exact duplicates")

        return unique_fragments

    async def _generate_embeddings(self, fragments: List[ContentFragment]):
        """Generate semantic embeddings and TF-IDF vectors for all fragments"""
        contents = [f.content for f in fragments]

        # Generate semantic embeddings
        embeddings = self.semantic_model.encode(contents, show_progress_bar=True)

        # Generate TF-IDF vectors
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(contents)
            tfidf_vectors = tfidf_matrix.toarray()
        except Exception as e:
            logger.warning(f"TF-IDF generation failed: {e}, using zero vectors")
            tfidf_vectors = np.zeros((len(contents), 100))

        # Assign to fragments
        for i, fragment in enumerate(fragments):
            fragment.embedding = embeddings[i]
            fragment.tfidf_vector = tfidf_vectors[i]

    async def _perform_semantic_clustering(self, fragments: List[ContentFragment]) -> Dict[int, List[ContentFragment]]:
        """Perform DBSCAN clustering on semantic embeddings"""
        if not fragments:
            return {}

        # Prepare embedding matrix
        embeddings = np.array([f.embedding for f in fragments])

        # Perform clustering
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.min_samples,
            metric='cosine'
        ).fit(embeddings)

        # Group fragments by cluster
        clusters = defaultdict(list)

        for i, fragment in enumerate(fragments):
            cluster_id = clustering.labels_[i]
            fragment.semantic_cluster = cluster_id
            clusters[cluster_id].append(fragment)

        logger.info(f"Formed {len([k for k in clusters.keys() if k != -1])} semantic clusters")

        return dict(clusters)

    async def _deduplicate_clusters(self,
                                   clusters: Dict[int, List[ContentFragment]],
                                   all_fragments: List[ContentFragment]) -> List[ContentFragment]:
        """Intelligently deduplicate within each semantic cluster"""
        deduplicated = []

        for cluster_id, cluster_fragments in clusters.items():
            if cluster_id == -1:  # Noise cluster - keep all unique fragments
                deduplicated.extend(cluster_fragments)
                continue

            if len(cluster_fragments) == 1:
                deduplicated.extend(cluster_fragments)
                continue

            # Within cluster deduplication
            cluster_deduplicated = await self._deduplicate_similar_fragments(cluster_fragments)
            deduplicated.extend(cluster_deduplicated)

        return deduplicated

    async def _deduplicate_similar_fragments(self, fragments: List[ContentFragment]) -> List[ContentFragment]:
        """Deduplicate highly similar fragments within a cluster"""
        if len(fragments) <= 1:
            return fragments

        # Calculate pairwise similarities
        embeddings = np.array([f.embedding for f in fragments])
        similarity_matrix = cosine_similarity(embeddings)

        # Find groups of highly similar fragments
        to_remove = set()

        for i in range(len(fragments)):
            if i in to_remove:
                continue

            for j in range(i + 1, len(fragments)):
                if j in to_remove:
                    continue

                similarity = similarity_matrix[i][j]

                if similarity > self.similarity_threshold:
                    # Keep the more important or longer fragment
                    fragment_i = fragments[i]
                    fragment_j = fragments[j]

                    # Scoring criteria: importance > length > recency
                    score_i = (fragment_i.importance_score * 2 +
                              len(fragment_i.content) / 1000 +
                              fragment_i.timestamp.timestamp() / 1e10)

                    score_j = (fragment_j.importance_score * 2 +
                              len(fragment_j.content) / 1000 +
                              fragment_j.timestamp.timestamp() / 1e10)

                    if score_i >= score_j:
                        to_remove.add(j)
                    else:
                        to_remove.add(i)
                        break

        # Return fragments not marked for removal
        result = [f for i, f in enumerate(fragments) if i not in to_remove]

        if len(result) < len(fragments):
            logger.debug(f"Cluster deduplication: {len(fragments)} -> {len(result)} fragments")

        return result

    async def _optimize_final_selection(self,
                                       fragments: List[ContentFragment],
                                       preserve_order: bool) -> List[ContentFragment]:
        """Final optimization pass to ensure quality and coherence"""
        if not fragments:
            return fragments

        # Sort by importance and timestamp if not preserving order
        if not preserve_order:
            fragments.sort(key=lambda f: (f.importance_score, f.timestamp), reverse=True)

        # Apply final filtering based on content quality
        filtered_fragments = []

        for fragment in fragments:
            # Quality checks
            if self._is_quality_content(fragment):
                filtered_fragments.append(fragment)

        return filtered_fragments

    def _is_quality_content(self, fragment: ContentFragment) -> bool:
        """Check if content meets quality thresholds"""
        content = fragment.content.strip()

        # Minimum length check
        if len(content) < 10:
            return False

        # Check for meaningful content (not just punctuation/whitespace)
        meaningful_chars = re.sub(r'[^\w\s]', '', content)
        if len(meaningful_chars) < 5:
            return False

        # Check for repetitive content
        if self._is_repetitive_content(content):
            return False

        return True

    def _is_repetitive_content(self, content: str) -> bool:
        """Detect repetitive or low-value content"""
        # Check for excessive repetition of characters or words
        words = content.split()
        if len(words) < 3:
            return False

        # Check for repeated words
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word] += 1

        # If any word appears more than 50% of the time, it's likely repetitive
        max_count = max(word_counts.values())
        if max_count / len(words) > 0.5:
            return True

        return False

    async def _calculate_metrics(self,
                               original_fragments: List[ContentFragment],
                               final_fragments: List[ContentFragment],
                               clusters: Dict[int, List[ContentFragment]],
                               processing_time: float) -> DeduplicationResult:
        """Calculate comprehensive metrics for the deduplication process"""

        # Basic counts
        original_count = len(original_fragments)
        final_count = len(final_fragments)
        removed_count = original_count - final_count

        # Token counts
        original_tokens = sum(f.token_count for f in original_fragments)
        final_tokens = sum(f.token_count for f in final_fragments)

        # Compression ratio
        compression_ratio = 1 - (final_tokens / original_tokens) if original_tokens > 0 else 0

        # Information loss estimation (based on semantic coverage)
        information_loss = await self._estimate_information_loss(original_fragments, final_fragments)

        # Semantic coherence score
        coherence_score = await self._calculate_semantic_coherence(final_fragments)

        # Cluster information
        cluster_info = {k: [f.id for f in v] for k, v in clusters.items() if k != -1}

        # Removed fragments
        removed_fragment_ids = set(f.id for f in original_fragments) - set(f.id for f in final_fragments)
        removed_fragments = [f for f in original_fragments if f.id in removed_fragment_ids]

        return DeduplicationResult(
            original_fragments=original_fragments,
            deduplicated_fragments=final_fragments,
            removed_duplicates=removed_fragments,
            similarity_clusters=cluster_info,
            compression_ratio=compression_ratio,
            information_loss_score=information_loss,
            semantic_coherence_score=coherence_score,
            processing_time=processing_time
        )

    async def _estimate_information_loss(self,
                                       original: List[ContentFragment],
                                       final: List[ContentFragment]) -> float:
        """Estimate information loss through semantic coverage analysis"""
        if not original or not final:
            return 1.0 if not final else 0.0

        # Create embeddings for both sets
        original_embeddings = np.array([f.embedding for f in original])
        final_embeddings = np.array([f.embedding for f in final])

        # Calculate coverage: how well final set covers original semantic space
        similarities = cosine_similarity(original_embeddings, final_embeddings)
        max_similarities = np.max(similarities, axis=1)

        # Information loss is the complement of average maximum similarity
        coverage = np.mean(max_similarities)
        information_loss = 1 - coverage

        return max(0.0, min(1.0, information_loss))

    async def _calculate_semantic_coherence(self, fragments: List[ContentFragment]) -> float:
        """Calculate semantic coherence of the final fragment set"""
        if len(fragments) < 2:
            return 1.0

        embeddings = np.array([f.embedding for f in fragments])
        similarities = cosine_similarity(embeddings)

        # Remove diagonal (self-similarities)
        mask = np.eye(similarities.shape[0], dtype=bool)
        similarities_no_diag = similarities[~mask]

        # Coherence is the mean pairwise similarity
        coherence = np.mean(similarities_no_diag)

        return max(0.0, min(1.0, coherence))

    def _update_statistics(self, result: DeduplicationResult):
        """Update internal statistics"""
        self.stats['total_processed'] += len(result.original_fragments)
        self.stats['duplicates_removed'] += len(result.removed_duplicates)
        self.stats['clusters_formed'] += len(result.similarity_clusters)
        self.stats['total_compression_ratio'] += result.compression_ratio

    async def get_statistics(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return {
            **self.stats,
            'average_compression_ratio': (
                self.stats['total_compression_ratio'] / max(1, self.stats['total_processed'])
            )
        }

    async def analyze_content_diversity(self, fragments: List[ContentFragment]) -> Dict[str, Any]:
        """Analyze content diversity and semantic distribution"""
        if not fragments:
            return {}

        embeddings = np.array([f.embedding for f in fragments])

        # Calculate pairwise distances
        distances = 1 - cosine_similarity(embeddings)

        # Diversity metrics
        avg_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
        max_distance = np.max(distances)
        min_distance = np.min(distances[np.triu_indices_from(distances, k=1)])

        return {
            'content_count': len(fragments),
            'average_semantic_distance': float(avg_distance),
            'max_semantic_distance': float(max_distance),
            'min_semantic_distance': float(min_distance),
            'semantic_variance': float(np.var(distances)),
            'diversity_score': float(avg_distance)  # Higher is more diverse
        }

    async def save_cache(self, key: str, fragments: List[ContentFragment]):
        """Save processed fragments to cache"""
        try:
            serialized = [
                {
                    'id': f.id,
                    'content': f.content,
                    'content_type': f.content_type,
                    'source': f.source,
                    'timestamp': f.timestamp.isoformat(),
                    'token_count': f.token_count,
                    'hash_md5': f.hash_md5,
                    'embedding': f.embedding.tolist() if f.embedding is not None else None,
                    'semantic_cluster': f.semantic_cluster,
                    'importance_score': f.importance_score
                }
                for f in fragments
            ]

            self.redis_client.setex(
                f"dedup_cache:{key}",
                3600,  # 1 hour TTL
                json.dumps(serialized)
            )

        except Exception as e:
            logger.warning(f"Failed to save cache for key {key}: {e}")

    async def load_cache(self, key: str) -> Optional[List[ContentFragment]]:
        """Load processed fragments from cache"""
        try:
            cached_data = self.redis_client.get(f"dedup_cache:{key}")
            if not cached_data:
                return None

            serialized = json.loads(cached_data)
            fragments = []

            for data in serialized:
                fragment = ContentFragment(
                    id=data['id'],
                    content=data['content'],
                    content_type=data['content_type'],
                    source=data['source'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    token_count=data['token_count'],
                    importance_score=data['importance_score']
                )
                fragment.hash_md5 = data['hash_md5']
                fragment.semantic_cluster = data['semantic_cluster']

                if data['embedding']:
                    fragment.embedding = np.array(data['embedding'])

                fragments.append(fragment)

            return fragments

        except Exception as e:
            logger.warning(f"Failed to load cache for key {key}: {e}")
            return None