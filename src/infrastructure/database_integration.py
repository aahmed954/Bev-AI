#!/usr/bin/env python3
"""
Database Integration System
Seamless integration between Vector DBs, PostgreSQL, and Neo4j
Author: BEV OSINT Team
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import hashlib

# Database connections
import asyncpg
from neo4j import AsyncGraphDatabase
import asyncio_redis as redis

# Vector database integration
from .vector_db_manager import VectorDatabaseManager, EmbeddingDocument, VectorSearchResult
from .embedding_manager import EmbeddingPipeline, EmbeddingRequest

# Message queue and event streaming
import aio_pika
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

# Performance monitoring
from prometheus_client import Counter, Histogram, Gauge
import psutil

# Security and utilities
from cryptography.fernet import Fernet
import xxhash


@dataclass
class DataIntegrationEvent:
    """Event for cross-database data integration"""
    id: str
    event_type: str  # 'create', 'update', 'delete', 'search'
    source_db: str   # 'postgres', 'neo4j', 'qdrant', 'weaviate'
    target_dbs: List[str]
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    priority: int = 1


@dataclass
class EntityRelationship:
    """Graph relationship for Neo4j integration"""
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any]
    confidence: float = 1.0
    source: str = "system"


@dataclass
class IntegratedSearchResult:
    """Combined search result from multiple databases"""
    id: str
    content: str
    vector_similarity: float
    metadata: Dict[str, Any]
    relationships: List[EntityRelationship]
    entities: List[Dict[str, Any]]
    source_databases: List[str]
    relevance_score: float
    timestamp: datetime


class DatabaseSyncManager:
    """Manages synchronization between databases"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()

        # Sync state tracking
        self.sync_state = {
            'postgres_to_vector': {'last_sync': None, 'sync_count': 0},
            'neo4j_to_vector': {'last_sync': None, 'sync_count': 0},
            'vector_to_postgres': {'last_sync': None, 'sync_count': 0},
            'vector_to_neo4j': {'last_sync': None, 'sync_count': 0}
        }

        # Performance tracking
        self.sync_metrics = {
            'records_synced': 0,
            'sync_duration': 0,
            'sync_errors': 0,
            'last_error': None
        }

        # Change tracking
        self.change_log = asyncio.Queue(maxsize=10000)

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('db_sync_manager')
        logger.setLevel(logging.INFO)
        return logger

    async def sync_postgres_to_vector(
        self,
        postgres_conn: asyncpg.Connection,
        vector_manager: VectorDatabaseManager,
        embedding_pipeline: EmbeddingPipeline,
        table_name: str,
        content_field: str,
        batch_size: int = 100
    ) -> int:
        """Sync PostgreSQL data to vector databases"""
        start_time = time.time()
        synced_count = 0

        try:
            # Get last sync timestamp
            last_sync = self.sync_state['postgres_to_vector']['last_sync']
            where_clause = ""
            params = []

            if last_sync:
                where_clause = "WHERE updated_at > $1"
                params.append(last_sync)

            # Query new/updated records
            query = f"""
                SELECT id, {content_field}, metadata, created_at, updated_at
                FROM {table_name}
                {where_clause}
                ORDER BY updated_at
                LIMIT {batch_size}
            """

            rows = await postgres_conn.fetch(query, *params)

            if rows:
                # Convert to embedding documents
                documents = []
                for row in rows:
                    doc = EmbeddingDocument(
                        id=str(row['id']),
                        content=row[content_field],
                        metadata=row['metadata'] or {},
                        collection=table_name,
                        timestamp=row['created_at']
                    )
                    documents.append(doc)

                # Generate embeddings
                embedding_requests = [
                    EmbeddingRequest(
                        id=doc.id,
                        content=doc.content,
                        model_name="sentence-transformers-mini",
                        priority=2
                    )
                    for doc in documents
                ]

                embedding_responses = await embedding_pipeline.generate_embeddings_batch(
                    embedding_requests
                )

                # Update documents with embeddings
                for i, response in enumerate(embedding_responses):
                    if not response.error:
                        documents[i].vector = response.vector

                # Store in vector database
                success = await vector_manager.upsert_vectors(documents, table_name)

                if success:
                    synced_count = len(documents)
                    self.sync_state['postgres_to_vector']['last_sync'] = datetime.utcnow()
                    self.sync_state['postgres_to_vector']['sync_count'] += synced_count

            duration = time.time() - start_time
            self.sync_metrics['sync_duration'] += duration

            self.logger.info(f"✅ Synced {synced_count} records from {table_name} to vector DB in {duration:.2f}s")
            return synced_count

        except Exception as e:
            self.sync_metrics['sync_errors'] += 1
            self.sync_metrics['last_error'] = str(e)
            self.logger.error(f"❌ Sync failed for {table_name}: {e}")
            return 0

    async def sync_neo4j_to_vector(
        self,
        neo4j_driver,
        vector_manager: VectorDatabaseManager,
        embedding_pipeline: EmbeddingPipeline,
        node_labels: List[str],
        batch_size: int = 100
    ) -> int:
        """Sync Neo4j nodes to vector databases"""
        start_time = time.time()
        synced_count = 0

        try:
            async with neo4j_driver.session() as session:
                for label in node_labels:
                    # Get nodes with their properties
                    query = f"""
                        MATCH (n:{label})
                        WHERE n.updated_at > $last_sync OR NOT EXISTS(n.vector_synced)
                        RETURN n.id AS id, n.content AS content,
                               n.created_at AS created_at, n.updated_at AS updated_at,
                               properties(n) AS properties
                        ORDER BY n.updated_at
                        LIMIT {batch_size}
                    """

                    last_sync = self.sync_state['neo4j_to_vector']['last_sync']
                    if not last_sync:
                        last_sync = datetime(1970, 1, 1)

                    result = await session.run(query, last_sync=last_sync)
                    records = await result.data()

                    if records:
                        # Convert to embedding documents
                        documents = []
                        for record in records:
                            if record['content']:
                                doc = EmbeddingDocument(
                                    id=f"{label}_{record['id']}",
                                    content=record['content'],
                                    metadata={
                                        'label': label,
                                        'neo4j_id': record['id'],
                                        **record['properties']
                                    },
                                    collection=f"neo4j_{label.lower()}",
                                    timestamp=record['created_at']
                                )
                                documents.append(doc)

                        if documents:
                            # Generate embeddings
                            embedding_requests = [
                                EmbeddingRequest(
                                    id=doc.id,
                                    content=doc.content,
                                    model_name="sentence-transformers-mini",
                                    priority=2
                                )
                                for doc in documents
                            ]

                            embedding_responses = await embedding_pipeline.generate_embeddings_batch(
                                embedding_requests
                            )

                            # Update documents with embeddings
                            for i, response in enumerate(embedding_responses):
                                if not response.error:
                                    documents[i].vector = response.vector

                            # Store in vector database
                            success = await vector_manager.upsert_vectors(
                                documents, f"neo4j_{label.lower()}"
                            )

                            if success:
                                # Mark nodes as synced
                                update_query = f"""
                                    MATCH (n:{label})
                                    WHERE n.id IN $ids
                                    SET n.vector_synced = true, n.vector_sync_date = datetime()
                                """
                                ids = [doc.metadata['neo4j_id'] for doc in documents]
                                await session.run(update_query, ids=ids)

                                synced_count += len(documents)

            if synced_count > 0:
                self.sync_state['neo4j_to_vector']['last_sync'] = datetime.utcnow()
                self.sync_state['neo4j_to_vector']['sync_count'] += synced_count

            duration = time.time() - start_time
            self.logger.info(f"✅ Synced {synced_count} Neo4j nodes to vector DB in {duration:.2f}s")
            return synced_count

        except Exception as e:
            self.sync_metrics['sync_errors'] += 1
            self.sync_metrics['last_error'] = str(e)
            self.logger.error(f"❌ Neo4j sync failed: {e}")
            return 0

    def get_sync_status(self) -> Dict[str, Any]:
        """Get synchronization status"""
        return {
            'sync_state': self.sync_state,
            'metrics': self.sync_metrics,
            'queue_size': self.change_log.qsize()
        }


class CrossDatabaseSearchEngine:
    """Advanced search engine across all databases"""

    def __init__(
        self,
        postgres_conn: asyncpg.Connection,
        neo4j_driver,
        vector_manager: VectorDatabaseManager,
        embedding_pipeline: EmbeddingPipeline,
        redis_client
    ):
        self.postgres = postgres_conn
        self.neo4j = neo4j_driver
        self.vector_manager = vector_manager
        self.embedding_pipeline = embedding_pipeline
        self.redis = redis_client

        self.logger = logging.getLogger('cross_db_search')

        # Search metrics
        self.search_metrics = {
            'total_searches': 0,
            'average_latency': 0,
            'cache_hits': 0,
            'cross_db_queries': 0
        }

    async def search(
        self,
        query: str,
        search_type: str = "semantic",  # "semantic", "exact", "hybrid"
        databases: List[str] = None,
        limit: int = 10,
        include_relationships: bool = True,
        similarity_threshold: float = 0.7
    ) -> List[IntegratedSearchResult]:
        """
        Perform cross-database search

        Args:
            query: Search query text
            search_type: Type of search to perform
            databases: List of databases to search ('postgres', 'neo4j', 'qdrant', 'weaviate')
            limit: Maximum results to return
            include_relationships: Whether to include graph relationships
            similarity_threshold: Minimum similarity score for vector search

        Returns:
            List of integrated search results
        """
        start_time = time.time()
        self.search_metrics['total_searches'] += 1

        if databases is None:
            databases = ['postgres', 'neo4j', 'qdrant', 'weaviate']

        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, search_type, databases, limit)
            cached_results = await self._get_cached_results(cache_key)

            if cached_results:
                self.search_metrics['cache_hits'] += 1
                return cached_results

            results = []

            # Vector similarity search
            if search_type in ["semantic", "hybrid"] and any(db in databases for db in ['qdrant', 'weaviate']):
                vector_results = await self._vector_search(query, limit, similarity_threshold)
                results.extend(vector_results)

            # PostgreSQL search
            if "postgres" in databases:
                postgres_results = await self._postgres_search(query, search_type, limit)
                results.extend(postgres_results)

            # Neo4j graph search
            if "neo4j" in databases:
                neo4j_results = await self._neo4j_search(query, search_type, limit, include_relationships)
                results.extend(neo4j_results)

            # Merge and rank results
            merged_results = await self._merge_and_rank_results(results, query, limit)

            # Enhance with relationships if requested
            if include_relationships:
                for result in merged_results:
                    result.relationships = await self._get_entity_relationships(result.id)

            # Cache results
            await self._cache_results(cache_key, merged_results)

            # Update metrics
            duration = time.time() - start_time
            self.search_metrics['average_latency'] = (
                (self.search_metrics['average_latency'] * (self.search_metrics['total_searches'] - 1) + duration)
                / self.search_metrics['total_searches']
            )

            if len(databases) > 1:
                self.search_metrics['cross_db_queries'] += 1

            self.logger.info(f"🔍 Cross-DB search completed: {len(merged_results)} results in {duration:.2f}s")
            return merged_results

        except Exception as e:
            self.logger.error(f"❌ Cross-database search failed: {e}")
            return []

    async def _vector_search(
        self,
        query: str,
        limit: int,
        similarity_threshold: float
    ) -> List[IntegratedSearchResult]:
        """Perform vector similarity search"""
        try:
            # Generate query embedding
            embedding_request = EmbeddingRequest(
                id=f"search_{hash(query)}",
                content=query,
                model_name="sentence-transformers-mini",
                priority=3  # High priority for search
            )

            responses = await self.embedding_pipeline.generate_embeddings_batch([embedding_request])

            if responses and not responses[0].error:
                query_vector = responses[0].vector

                # Search all collections
                collections = ['osint_intel', 'threat_indicators', 'social_media', 'dark_web']
                all_results = []

                for collection in collections:
                    search_results = await self.vector_manager.search_vectors(
                        query_vector=query_vector,
                        collection=collection,
                        limit=limit,
                        score_threshold=similarity_threshold
                    )

                    for result in search_results:
                        integrated_result = IntegratedSearchResult(
                            id=result.id,
                            content=result.payload.get('content', ''),
                            vector_similarity=result.score,
                            metadata=result.payload,
                            relationships=[],
                            entities=[],
                            source_databases=['vector_db'],
                            relevance_score=result.score,
                            timestamp=datetime.utcnow()
                        )
                        all_results.append(integrated_result)

                return sorted(all_results, key=lambda x: x.vector_similarity, reverse=True)[:limit]

        except Exception as e:
            self.logger.error(f"❌ Vector search failed: {e}")

        return []

    async def _postgres_search(
        self,
        query: str,
        search_type: str,
        limit: int
    ) -> List[IntegratedSearchResult]:
        """Perform PostgreSQL search"""
        try:
            results = []

            if search_type in ["exact", "hybrid"]:
                # Full-text search
                search_query = """
                    SELECT id, content, metadata, created_at,
                           ts_rank(to_tsvector('english', content), plainto_tsquery('english', $1)) as rank
                    FROM osint_data
                    WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $1)
                    ORDER BY rank DESC
                    LIMIT $2
                """

                rows = await self.postgres.fetch(search_query, query, limit)

                for row in rows:
                    result = IntegratedSearchResult(
                        id=str(row['id']),
                        content=row['content'],
                        vector_similarity=0.0,
                        metadata=row['metadata'] or {},
                        relationships=[],
                        entities=[],
                        source_databases=['postgres'],
                        relevance_score=float(row['rank']),
                        timestamp=row['created_at']
                    )
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"❌ PostgreSQL search failed: {e}")
            return []

    async def _neo4j_search(
        self,
        query: str,
        search_type: str,
        limit: int,
        include_relationships: bool
    ) -> List[IntegratedSearchResult]:
        """Perform Neo4j graph search"""
        try:
            results = []

            async with self.neo4j.session() as session:
                # Full-text search with graph context
                search_query = """
                    CALL db.index.fulltext.queryNodes('contentIndex', $query)
                    YIELD node, score
                    OPTIONAL MATCH (node)-[r]-(connected)
                    RETURN node.id AS id, node.content AS content,
                           properties(node) AS properties, score,
                           COLLECT({type: type(r), target: connected.id, properties: properties(r)}) AS relationships
                    ORDER BY score DESC
                    LIMIT $limit
                """

                result = await session.run(search_query, query=query, limit=limit)
                records = await result.data()

                for record in records:
                    relationships = []
                    if include_relationships:
                        for rel in record['relationships']:
                            if rel['target']:  # Valid relationship
                                relationships.append(EntityRelationship(
                                    source_id=record['id'],
                                    target_id=rel['target'],
                                    relationship_type=rel['type'],
                                    properties=rel['properties'],
                                    confidence=1.0,
                                    source="neo4j"
                                ))

                    result = IntegratedSearchResult(
                        id=str(record['id']),
                        content=record['content'] or '',
                        vector_similarity=0.0,
                        metadata=record['properties'],
                        relationships=relationships,
                        entities=[],
                        source_databases=['neo4j'],
                        relevance_score=record['score'],
                        timestamp=datetime.utcnow()
                    )
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"❌ Neo4j search failed: {e}")
            return []

    async def _merge_and_rank_results(
        self,
        results: List[IntegratedSearchResult],
        query: str,
        limit: int
    ) -> List[IntegratedSearchResult]:
        """Merge and rank results from multiple databases"""
        try:
            # Group results by ID (same entity from different databases)
            grouped_results = {}

            for result in results:
                if result.id in grouped_results:
                    # Merge with existing result
                    existing = grouped_results[result.id]
                    existing.source_databases.extend(result.source_databases)
                    existing.metadata.update(result.metadata)
                    existing.relationships.extend(result.relationships)

                    # Use highest relevance score
                    if result.relevance_score > existing.relevance_score:
                        existing.relevance_score = result.relevance_score

                    # Use highest vector similarity
                    if result.vector_similarity > existing.vector_similarity:
                        existing.vector_similarity = result.vector_similarity

                else:
                    grouped_results[result.id] = result

            # Calculate composite relevance score
            for result in grouped_results.values():
                # Combine vector similarity and text relevance
                composite_score = (
                    result.vector_similarity * 0.6 +
                    result.relevance_score * 0.4 +
                    len(result.source_databases) * 0.1  # Bonus for cross-database presence
                )
                result.relevance_score = composite_score

            # Sort by composite relevance score
            sorted_results = sorted(
                grouped_results.values(),
                key=lambda x: x.relevance_score,
                reverse=True
            )

            return sorted_results[:limit]

        except Exception as e:
            self.logger.error(f"❌ Result merging failed: {e}")
            return results[:limit]

    async def _get_entity_relationships(self, entity_id: str) -> List[EntityRelationship]:
        """Get relationships for an entity from Neo4j"""
        try:
            relationships = []

            async with self.neo4j.session() as session:
                query = """
                    MATCH (n)-[r]-(connected)
                    WHERE n.id = $entity_id
                    RETURN type(r) AS rel_type, connected.id AS target_id,
                           properties(r) AS properties
                    LIMIT 50
                """

                result = await session.run(query, entity_id=entity_id)
                records = await result.data()

                for record in records:
                    relationship = EntityRelationship(
                        source_id=entity_id,
                        target_id=record['target_id'],
                        relationship_type=record['rel_type'],
                        properties=record['properties'],
                        confidence=1.0,
                        source="neo4j"
                    )
                    relationships.append(relationship)

            return relationships

        except Exception as e:
            self.logger.error(f"❌ Failed to get relationships for {entity_id}: {e}")
            return []

    def _generate_cache_key(
        self,
        query: str,
        search_type: str,
        databases: List[str],
        limit: int
    ) -> str:
        """Generate cache key for search results"""
        cache_input = f"{query}:{search_type}:{':'.join(sorted(databases))}:{limit}"
        return xxhash.xxh64(cache_input.encode()).hexdigest()

    async def _get_cached_results(self, cache_key: str) -> Optional[List[IntegratedSearchResult]]:
        """Get results from cache"""
        try:
            cached_data = await self.redis.get(f"search_cache:{cache_key}")
            if cached_data:
                data = json.loads(cached_data)
                return [IntegratedSearchResult(**item) for item in data]
        except Exception as e:
            self.logger.warning(f"Cache get error: {e}")
        return None

    async def _cache_results(self, cache_key: str, results: List[IntegratedSearchResult]):
        """Cache search results"""
        try:
            # Convert to JSON-serializable format
            serializable_results = []
            for result in results:
                result_dict = asdict(result)
                result_dict['timestamp'] = result.timestamp.isoformat()
                result_dict['relationships'] = [asdict(rel) for rel in result.relationships]
                serializable_results.append(result_dict)

            await self.redis.setex(
                f"search_cache:{cache_key}",
                3600,  # 1 hour cache
                json.dumps(serializable_results)
            )
        except Exception as e:
            self.logger.warning(f"Cache set error: {e}")

    def get_search_metrics(self) -> Dict[str, Any]:
        """Get search performance metrics"""
        return self.search_metrics.copy()


class DatabaseIntegrationOrchestrator:
    """Main orchestrator for database integration operations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()

        # Database connections
        self.postgres: Optional[asyncpg.Connection] = None
        self.neo4j = None
        self.redis: Optional = None

        # Integration components
        self.vector_manager: Optional[VectorDatabaseManager] = None
        self.embedding_pipeline: Optional[EmbeddingPipeline] = None
        self.sync_manager: Optional[DatabaseSyncManager] = None
        self.search_engine: Optional[CrossDatabaseSearchEngine] = None

        # Background tasks
        self.sync_tasks: List[asyncio.Task] = []
        self.running = False

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('db_integration')
        logger.setLevel(logging.INFO)
        return logger

    async def initialize(self) -> bool:
        """Initialize all integration components"""
        self.logger.info("🚀 Initializing Database Integration System")

        try:
            # Initialize database connections
            await self._connect_databases()

            # Initialize vector components
            self.vector_manager = VectorDatabaseManager(self.config['vector_db'])
            await self.vector_manager.initialize()

            self.embedding_pipeline = EmbeddingPipeline(
                self.config['postgres'],
                self.config['redis']
            )
            await self.embedding_pipeline.initialize()

            # Initialize integration components
            self.sync_manager = DatabaseSyncManager(self.config)

            self.search_engine = CrossDatabaseSearchEngine(
                self.postgres,
                self.neo4j,
                self.vector_manager,
                self.embedding_pipeline,
                self.redis
            )

            # Start background sync tasks
            await self._start_sync_tasks()

            self.running = True
            self.logger.info("✅ Database Integration System initialized")
            return True

        except Exception as e:
            self.logger.error(f"❌ Integration initialization failed: {e}")
            return False

    async def _connect_databases(self):
        """Connect to all databases"""
        # PostgreSQL
        self.postgres = await asyncpg.connect(**self.config['postgres'])

        # Neo4j
        self.neo4j = AsyncGraphDatabase.driver(
            self.config['neo4j']['uri'],
            auth=(self.config['neo4j']['user'], self.config['neo4j']['password'])
        )

        # Redis
        self.redis = await redis.Connection.create(**self.config['redis'])

    async def _start_sync_tasks(self):
        """Start background synchronization tasks"""
        self.sync_tasks = [
            asyncio.create_task(self._periodic_postgres_sync()),
            asyncio.create_task(self._periodic_neo4j_sync()),
            asyncio.create_task(self._monitor_sync_health())
        ]

        self.logger.info("✅ Started background sync tasks")

    async def _periodic_postgres_sync(self):
        """Periodic PostgreSQL to vector sync"""
        while self.running:
            try:
                await self.sync_manager.sync_postgres_to_vector(
                    self.postgres,
                    self.vector_manager,
                    self.embedding_pipeline,
                    'osint_data',
                    'content'
                )
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                self.logger.error(f"❌ PostgreSQL sync error: {e}")
                await asyncio.sleep(60)

    async def _periodic_neo4j_sync(self):
        """Periodic Neo4j to vector sync"""
        while self.running:
            try:
                await self.sync_manager.sync_neo4j_to_vector(
                    self.neo4j,
                    self.vector_manager,
                    self.embedding_pipeline,
                    ['Entity', 'Event', 'Location', 'Person']
                )
                await asyncio.sleep(600)  # Every 10 minutes
            except Exception as e:
                self.logger.error(f"❌ Neo4j sync error: {e}")
                await asyncio.sleep(120)

    async def _monitor_sync_health(self):
        """Monitor synchronization health"""
        while self.running:
            try:
                status = self.sync_manager.get_sync_status()
                self.logger.info(f"📊 Sync status: {status}")
                await asyncio.sleep(1800)  # Every 30 minutes
            except Exception as e:
                self.logger.error(f"❌ Sync monitoring error: {e}")
                await asyncio.sleep(300)

    async def search(self, **kwargs) -> List[IntegratedSearchResult]:
        """Perform integrated search across all databases"""
        if not self.search_engine:
            raise RuntimeError("Search engine not initialized")
        return await self.search_engine.search(**kwargs)

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            'running': self.running,
            'databases': {
                'postgres': self.postgres is not None,
                'neo4j': self.neo4j is not None,
                'redis': self.redis is not None
            },
            'components': {
                'vector_manager': self.vector_manager is not None,
                'embedding_pipeline': self.embedding_pipeline is not None,
                'sync_manager': self.sync_manager is not None,
                'search_engine': self.search_engine is not None
            },
            'background_tasks': len(self.sync_tasks)
        }

        if self.sync_manager:
            status['sync_status'] = self.sync_manager.get_sync_status()

        if self.search_engine:
            status['search_metrics'] = self.search_engine.get_search_metrics()

        return status

    async def shutdown(self):
        """Shutdown all integration components"""
        self.logger.info("🔒 Shutting down Database Integration System")

        self.running = False

        # Cancel background tasks
        for task in self.sync_tasks:
            task.cancel()

        await asyncio.gather(*self.sync_tasks, return_exceptions=True)

        # Shutdown components
        if self.embedding_pipeline:
            await self.embedding_pipeline.shutdown()

        if self.vector_manager:
            await self.vector_manager.close()

        # Close database connections
        if self.postgres:
            await self.postgres.close()

        if self.neo4j:
            await self.neo4j.close()

        if self.redis:
            await self.redis.close()

        self.logger.info("✅ Database Integration System shutdown complete")


# Usage Example
async def main():
    """Example usage of Database Integration System"""

    config = {
        'postgres': {
            'host': '172.30.0.2',
            'port': 5432,
            'user': 'bev_user',
            'password': 'secure_password',
            'database': 'osint'
        },
        'neo4j': {
            'uri': 'bolt://172.30.0.3:7687',
            'user': 'neo4j',
            'password': 'secure_password'
        },
        'redis': {
            'host': '172.30.0.4',
            'port': 6379,
            'db': 0
        },
        'vector_db': {
            'qdrant_primary_host': '172.30.0.36',
            'weaviate_host': '172.30.0.38'
        }
    }

    # Initialize orchestrator
    orchestrator = DatabaseIntegrationOrchestrator(config)

    if await orchestrator.initialize():
        print("✅ Database Integration System ready")

        # Example search
        results = await orchestrator.search(
            query="cybersecurity threat intelligence",
            search_type="hybrid",
            databases=['postgres', 'neo4j', 'qdrant'],
            limit=10,
            include_relationships=True
        )

        print(f"🔍 Found {len(results)} integrated results")

        # Get system status
        status = await orchestrator.get_system_status()
        print(f"📊 System status: {json.dumps(status, indent=2)}")

        await orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())