"""
Integration Client for Extended Reasoning Pipeline
Handles integration with context compression and vector database systems
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
import aiohttp
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class IntegrationConfig:
    """Configuration for integration services"""
    compression_endpoint: str
    vector_db_endpoint: str
    qdrant_endpoint: str
    weaviate_endpoint: str
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

class IntegrationClient:
    """
    Client for integrating with context compression and vector database systems
    Provides unified interface for the extended reasoning pipeline
    """

    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def _make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with retry logic"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use as async context manager.")

        last_exception = None

        for attempt in range(self.config.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Request failed with status {response.status}: {url}")
                        if response.status >= 500:  # Server error, retry
                            raise aiohttp.ClientError(f"Server error: {response.status}")
                        else:  # Client error, don't retry
                            response_text = await response.text()
                            raise aiohttp.ClientError(f"Client error {response.status}: {response_text}")

            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                    logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}, retrying...")
                else:
                    logger.error(f"All {self.config.max_retries} request attempts failed")

        raise last_exception or Exception("Request failed after all retries")

    # Context Compression Integration
    async def compress_context(self, content: str, target_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Compress large context using the context compression service

        Args:
            content: Raw content to compress
            target_ratio: Target compression ratio (0.5 = 50% reduction)

        Returns:
            Compressed content and metadata
        """
        try:
            url = f"{self.config.compression_endpoint}/compress"
            payload = {
                'content': content,
                'target_ratio': target_ratio,
                'preserve_key_information': True,
                'maintain_structure': True
            }

            result = await self._make_request('POST', url, json=payload)

            return {
                'compressed_content': result.get('compressed_content', content),
                'compression_ratio': result.get('compression_ratio', 1.0),
                'key_information_preserved': result.get('key_information', []),
                'compression_metadata': result.get('metadata', {}),
                'original_size': len(content),
                'compressed_size': len(result.get('compressed_content', content))
            }

        except Exception as e:
            logger.error(f"Context compression failed: {str(e)}")
            # Fallback: return original content
            return {
                'compressed_content': content,
                'compression_ratio': 1.0,
                'key_information_preserved': [],
                'compression_metadata': {},
                'original_size': len(content),
                'compressed_size': len(content),
                'error': str(e)
            }

    async def decompress_context(self, compressed_content: str, metadata: Dict[str, Any]) -> str:
        """
        Decompress context using the compression service

        Args:
            compressed_content: Compressed content
            metadata: Compression metadata

        Returns:
            Decompressed content
        """
        try:
            url = f"{self.config.compression_endpoint}/decompress"
            payload = {
                'compressed_content': compressed_content,
                'metadata': metadata
            }

            result = await self._make_request('POST', url, json=payload)
            return result.get('decompressed_content', compressed_content)

        except Exception as e:
            logger.error(f"Context decompression failed: {str(e)}")
            return compressed_content

    async def intelligent_chunk(self, content: str, chunk_size: int = 8000,
                              overlap_ratio: float = 0.1) -> List[str]:
        """
        Intelligent chunking using compression service

        Args:
            content: Content to chunk
            chunk_size: Target chunk size
            overlap_ratio: Overlap between chunks

        Returns:
            List of intelligently chunked content
        """
        try:
            url = f"{self.config.compression_endpoint}/chunk"
            payload = {
                'content': content,
                'chunk_size': chunk_size,
                'overlap_ratio': overlap_ratio,
                'preserve_semantics': True,
                'respect_boundaries': True
            }

            result = await self._make_request('POST', url, json=payload)
            return result.get('chunks', [content])

        except Exception as e:
            logger.error(f"Intelligent chunking failed: {str(e)}")
            # Fallback: simple word-based chunking
            words = content.split()
            chunk_words = chunk_size // 4  # Rough word estimate
            overlap_words = int(chunk_words * overlap_ratio)

            chunks = []
            start = 0
            while start < len(words):
                end = min(start + chunk_words, len(words))
                chunk = ' '.join(words[start:end])
                chunks.append(chunk)

                if end >= len(words):
                    break

                start = end - overlap_words

            return chunks

    # Vector Database Integration
    async def store_embeddings(self, texts: List[str], metadata: List[Dict[str, Any]],
                             collection_name: str = "reasoning_context") -> List[str]:
        """
        Store text embeddings in vector database

        Args:
            texts: List of texts to embed and store
            metadata: Metadata for each text
            collection_name: Vector collection name

        Returns:
            List of document IDs
        """
        try:
            # Use Qdrant for primary storage
            url = f"{self.config.vector_db_endpoint}/collections/{collection_name}/points"

            # Prepare points for insertion
            points = []
            for i, (text, meta) in enumerate(zip(texts, metadata)):
                point_id = f"doc_{int(time.time())}_{i}"
                points.append({
                    'id': point_id,
                    'text': text,
                    'metadata': meta
                })

            payload = {'points': points}
            result = await self._make_request('POST', url, json=payload)

            return result.get('ids', [f"doc_{int(time.time())}_{i}" for i in range(len(texts))])

        except Exception as e:
            logger.error(f"Vector storage failed: {str(e)}")
            return []

    async def search_similar(self, query_text: str, limit: int = 10,
                           collection_name: str = "reasoning_context",
                           threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar content in vector database

        Args:
            query_text: Query text
            limit: Maximum results to return
            collection_name: Collection to search
            threshold: Similarity threshold

        Returns:
            List of similar documents with scores
        """
        try:
            url = f"{self.config.vector_db_endpoint}/collections/{collection_name}/search"
            payload = {
                'query': query_text,
                'limit': limit,
                'threshold': threshold
            }

            result = await self._make_request('POST', url, json=payload)
            return result.get('results', [])

        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for texts

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            url = f"{self.config.vector_db_endpoint}/embed"
            payload = {'texts': texts}

            result = await self._make_request('POST', url, json=payload)
            return result.get('embeddings', [])

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return []

    # Knowledge Graph Integration
    async def store_knowledge_graph(self, entities: List[Dict[str, Any]],
                                  relationships: List[Dict[str, Any]],
                                  context_id: str) -> bool:
        """
        Store knowledge graph in vector database

        Args:
            entities: List of entities
            relationships: List of relationships
            context_id: Context identifier

        Returns:
            Success status
        """
        try:
            # Prepare graph data for storage
            graph_data = {
                'context_id': context_id,
                'entities': entities,
                'relationships': relationships,
                'timestamp': time.time()
            }

            # Store as a special document
            url = f"{self.config.vector_db_endpoint}/collections/knowledge_graphs/points"
            payload = {
                'points': [{
                    'id': f"graph_{context_id}",
                    'text': json.dumps(graph_data),
                    'metadata': {
                        'type': 'knowledge_graph',
                        'context_id': context_id,
                        'entity_count': len(entities),
                        'relationship_count': len(relationships)
                    }
                }]
            }

            await self._make_request('POST', url, json=payload)
            return True

        except Exception as e:
            logger.error(f"Knowledge graph storage failed: {str(e)}")
            return False

    async def retrieve_knowledge_graph(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve knowledge graph by context ID

        Args:
            context_id: Context identifier

        Returns:
            Knowledge graph data
        """
        try:
            url = f"{self.config.vector_db_endpoint}/collections/knowledge_graphs/points/graph_{context_id}"
            result = await self._make_request('GET', url)

            if result and 'text' in result:
                return json.loads(result['text'])

            return None

        except Exception as e:
            logger.error(f"Knowledge graph retrieval failed: {str(e)}")
            return None

    # Memory and Caching Integration
    async def cache_reasoning_result(self, context_id: str, result: Dict[str, Any]) -> bool:
        """
        Cache reasoning result for future use

        Args:
            context_id: Context identifier
            result: Reasoning result to cache

        Returns:
            Success status
        """
        try:
            url = f"{self.config.compression_endpoint}/cache"
            payload = {
                'key': f"reasoning_result_{context_id}",
                'data': result,
                'ttl': 3600 * 24  # 24 hours
            }

            await self._make_request('POST', url, json=payload)
            return True

        except Exception as e:
            logger.error(f"Result caching failed: {str(e)}")
            return False

    async def get_cached_result(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached reasoning result

        Args:
            context_id: Context identifier

        Returns:
            Cached result if available
        """
        try:
            url = f"{self.config.compression_endpoint}/cache/reasoning_result_{context_id}"
            result = await self._make_request('GET', url)
            return result.get('data')

        except Exception as e:
            logger.error(f"Cache retrieval failed: {str(e)}")
            return None

    # Contextual Enhancement
    async def enhance_context(self, content: str, context_type: str = "osint") -> Dict[str, Any]:
        """
        Enhance context with additional information

        Args:
            content: Original content
            context_type: Type of context (osint, investigation, etc.)

        Returns:
            Enhanced context data
        """
        try:
            # Search for related content
            similar_content = await self.search_similar(
                query_text=content[:1000],  # Use first 1000 chars as query
                limit=5,
                collection_name="context_enhancement"
            )

            # Get relevant entities from previous analyses
            entities_search = await self.search_similar(
                query_text=content,
                limit=10,
                collection_name="extracted_entities"
            )

            return {
                'original_content': content,
                'similar_content': similar_content,
                'related_entities': entities_search,
                'context_type': context_type,
                'enhancement_timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Context enhancement failed: {str(e)}")
            return {
                'original_content': content,
                'similar_content': [],
                'related_entities': [],
                'context_type': context_type,
                'enhancement_timestamp': time.time(),
                'error': str(e)
            }

    # Health and Status Checks
    async def check_service_health(self) -> Dict[str, bool]:
        """
        Check health of all integrated services

        Returns:
            Health status of each service
        """
        services = {
            'compression': self.config.compression_endpoint,
            'vector_db': self.config.vector_db_endpoint
        }

        health_status = {}

        for service_name, endpoint in services.items():
            try:
                health_url = f"{endpoint}/health"
                result = await self._make_request('GET', health_url)
                health_status[service_name] = result.get('status') == 'healthy'
            except Exception as e:
                logger.warning(f"Health check failed for {service_name}: {str(e)}")
                health_status[service_name] = False

        return health_status

    async def get_service_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from integrated services

        Returns:
            Combined metrics from all services
        """
        metrics = {}

        try:
            # Compression service metrics
            comp_url = f"{self.config.compression_endpoint}/metrics"
            comp_metrics = await self._make_request('GET', comp_url)
            metrics['compression'] = comp_metrics

        except Exception as e:
            logger.warning(f"Failed to get compression metrics: {str(e)}")
            metrics['compression'] = {}

        try:
            # Vector database metrics
            vdb_url = f"{self.config.vector_db_endpoint}/metrics"
            vdb_metrics = await self._make_request('GET', vdb_url)
            metrics['vector_db'] = vdb_metrics

        except Exception as e:
            logger.warning(f"Failed to get vector DB metrics: {str(e)}")
            metrics['vector_db'] = {}

        return metrics

# Factory function for creating integration client
async def create_integration_client(
    compression_endpoint: str = "http://172.30.0.43:8000",
    vector_db_endpoint: str = "http://172.30.0.44:8000",
    **kwargs
) -> IntegrationClient:
    """
    Factory function to create and initialize integration client

    Args:
        compression_endpoint: Context compression service endpoint
        vector_db_endpoint: Vector database service endpoint
        **kwargs: Additional configuration options

    Returns:
        Initialized IntegrationClient
    """
    config = IntegrationConfig(
        compression_endpoint=compression_endpoint,
        vector_db_endpoint=vector_db_endpoint,
        qdrant_endpoint=kwargs.get('qdrant_endpoint', 'http://172.30.0.44:6333'),
        weaviate_endpoint=kwargs.get('weaviate_endpoint', 'http://172.30.0.44:8080'),
        timeout=kwargs.get('timeout', 30),
        max_retries=kwargs.get('max_retries', 3),
        retry_delay=kwargs.get('retry_delay', 1.0)
    )

    client = IntegrationClient(config)
    return client

# Utility functions for common integration patterns
async def process_with_compression(content: str, target_ratio: float = 0.5) -> Tuple[str, Dict[str, Any]]:
    """
    Utility function to process content with compression

    Args:
        content: Content to process
        target_ratio: Compression target ratio

    Returns:
        Compressed content and metadata
    """
    async with await create_integration_client() as client:
        result = await client.compress_context(content, target_ratio)
        return result['compressed_content'], result

async def store_and_search(texts: List[str], metadata: List[Dict[str, Any]],
                          query: str) -> List[Dict[str, Any]]:
    """
    Utility function to store texts and search for similar content

    Args:
        texts: Texts to store
        metadata: Metadata for texts
        query: Search query

    Returns:
        Search results
    """
    async with await create_integration_client() as client:
        await client.store_embeddings(texts, metadata)
        return await client.search_similar(query)

async def enhance_reasoning_context(content: str) -> Dict[str, Any]:
    """
    Utility function to enhance reasoning context

    Args:
        content: Original content

    Returns:
        Enhanced context data
    """
    async with await create_integration_client() as client:
        return await client.enhance_context(content)