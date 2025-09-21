import os
#!/usr/bin/env python3
"""
Vector Database Infrastructure Manager
Comprehensive Qdrant and Weaviate Management for BEV OSINT Framework
Author: BEV OSINT Team
"""

import asyncio
import logging
import time
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import numpy as np

# Vector Database Clients
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import weaviate
from weaviate.client import Client as WeaviateClient
from weaviate import Config

# Database connections
import asyncpg
import asyncio_redis as redis
from neo4j import AsyncGraphDatabase

# Monitoring and metrics
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry

# Security and encryption
from cryptography.fernet import Fernet
import ssl


@dataclass
class VectorSearchResult:
    """Standardized vector search result"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingDocument:
    """Document structure for embedding operations"""
    id: str
    content: str
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None
    collection: str = "default"
    timestamp: Optional[datetime] = None


class VectorDatabaseManager:
    """
    Comprehensive Vector Database Management System

    Features:
    - Dual vector database support (Qdrant + Weaviate)
    - Automatic failover and load balancing
    - Performance monitoring and optimization
    - Security integration with BEV security framework
    - Batch processing for high-throughput operations
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self._setup_logging()

        # Vector database clients
        self.qdrant_primary: Optional[QdrantClient] = None
        self.qdrant_replica: Optional[QdrantClient] = None
        self.weaviate: Optional[WeaviateClient] = None

        # Traditional database connections
        self.postgres: Optional[asyncpg.Connection] = None
        self.redis: Optional[redis.Connection] = None
        self.neo4j = None

        # Performance monitoring
        self.metrics = self._setup_metrics()

        # Security
        self.encryption_key = config.get('encryption_key', Fernet.generate_key())
        self.cipher = Fernet(self.encryption_key)

        # Connection health tracking
        self.health_status = {
            'qdrant_primary': False,
            'qdrant_replica': False,
            'weaviate': False,
            'postgres': False,
            'redis': False,
            'neo4j': False
        }

        # Performance optimization
        self.batch_size = config.get('batch_size', 32)
        self.connection_pool_size = config.get('connection_pool_size', 10)

    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger('bev_vector_db')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_metrics(self) -> Dict[str, Any]:
        """Setup Prometheus metrics for monitoring"""
        registry = CollectorRegistry()

        metrics = {
            'vector_operations_total': Counter(
                'bev_vector_operations_total',
                'Total vector database operations',
                ['operation', 'database', 'status'],
                registry=registry
            ),
            'vector_search_duration': Histogram(
                'bev_vector_search_duration_seconds',
                'Vector search operation duration',
                ['database'],
                registry=registry
            ),
            'vector_insert_duration': Histogram(
                'bev_vector_insert_duration_seconds',
                'Vector insert operation duration',
                ['database'],
                registry=registry
            ),
            'active_connections': Gauge(
                'bev_vector_db_connections',
                'Active database connections',
                ['database'],
                registry=registry
            ),
            'collection_size': Gauge(
                'bev_vector_collection_size',
                'Number of vectors in collection',
                ['database', 'collection'],
                registry=registry
            )
        }

        return metrics

    async def initialize(self) -> bool:
        """Initialize all database connections"""
        self.logger.info("🚀 Initializing Vector Database Manager")

        tasks = [
            self._connect_qdrant(),
            self._connect_weaviate(),
            self._connect_postgres(),
            self._connect_redis(),
            self._connect_neo4j()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Log connection results
        for i, result in enumerate(results):
            db_name = ['qdrant', 'weaviate', 'postgres', 'redis', 'neo4j'][i]
            if isinstance(result, Exception):
                self.logger.error(f"❌ Failed to connect to {db_name}: {result}")
            else:
                self.logger.info(f"✅ Connected to {db_name}")

        # Check if we have at least one vector database
        if not (self.health_status['qdrant_primary'] or self.health_status['weaviate']):
            self.logger.error("❌ No vector databases available")
            return False

        # Initialize collections/schemas
        await self._initialize_collections()

        # Start health monitoring
        asyncio.create_task(self._health_monitor())

        self.logger.info("✅ Vector Database Manager initialized")
        return True

    async def _connect_qdrant(self) -> bool:
        """Connect to Qdrant cluster"""
        try:
            # Primary Qdrant instance
            self.qdrant_primary = QdrantClient(
                host=self.config.get('qdrant_primary_host', '172.30.0.36'),
                port=self.config.get('qdrant_primary_port', 6333),
                timeout=30,
                prefer_grpc=True
            )

            # Test connection
            collections = await asyncio.to_thread(self.qdrant_primary.get_collections)
            self.health_status['qdrant_primary'] = True
            self.metrics['active_connections'].labels(database='qdrant_primary').set(1)

            # Replica Qdrant instance
            try:
                self.qdrant_replica = QdrantClient(
                    host=self.config.get('qdrant_replica_host', '172.30.0.37'),
                    port=self.config.get('qdrant_replica_port', 6336),
                    timeout=30,
                    prefer_grpc=True
                )

                collections = await asyncio.to_thread(self.qdrant_replica.get_collections)
                self.health_status['qdrant_replica'] = True
                self.metrics['active_connections'].labels(database='qdrant_replica').set(1)

            except Exception as e:
                self.logger.warning(f"⚠️ Qdrant replica unavailable: {e}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Qdrant connection failed: {e}")
            return False

    async def _connect_weaviate(self) -> bool:
        """Connect to Weaviate cluster"""
        try:
            # Configure authentication
            auth_config = weaviate.AuthApiKey(
                api_key=self.config.get('weaviate_api_key', 'default-key')
            )

            self.weaviate = weaviate.Client(
                url=f"http://{self.config.get('weaviate_host', '172.30.0.38')}:8080",
                auth_client_secret=auth_config,
                timeout_config=(10, 60),
                additional_headers={
                    "X-OpenAI-Api-Key": self.config.get('openai_api_key', ''),
                }
            )

            # Test connection
            schema = self.weaviate.schema.get()
            self.health_status['weaviate'] = True
            self.metrics['active_connections'].labels(database='weaviate').set(1)

            return True

        except Exception as e:
            self.logger.error(f"❌ Weaviate connection failed: {e}")
            return False

    async def _connect_postgres(self) -> bool:
        """Connect to PostgreSQL with pgvector"""
        try:
            self.postgres = await asyncpg.connect(
                host=self.config.get('postgres_host', '172.30.0.2'),
                port=self.config.get('postgres_port', 5432),
                user=self.config.get('postgres_user', 'bev_user'),
                password=self.config.get('postgres_password', 'secure_password'),
                database=self.config.get('postgres_db', 'osint'),
                ssl='prefer'
            )

            # Enable pgvector extension
            await self.postgres.execute('CREATE EXTENSION IF NOT EXISTS vector;')

            self.health_status['postgres'] = True
            self.metrics['active_connections'].labels(database='postgres').set(1)

            return True

        except Exception as e:
            self.logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False

    async def _connect_redis(self) -> bool:
        """Connect to Redis for caching"""
        try:
            self.redis = await redis.Connection.create(
                host=self.config.get('redis_host', '172.30.0.4'),
                port=self.config.get('redis_port', 6379),
                password=self.config.get('redis_password', ''),
                db=self.config.get('redis_db', 0)
            )

            await self.redis.ping()
            self.health_status['redis'] = True
            self.metrics['active_connections'].labels(database='redis').set(1)

            return True

        except Exception as e:
            self.logger.error(f"❌ Redis connection failed: {e}")
            return False

    async def _connect_neo4j(self) -> bool:
        """Connect to Neo4j graph database"""
        try:
            self.neo4j = AsyncGraphDatabase.driver(
                f"bolt://{self.config.get('neo4j_host', '172.30.0.3')}:7687",
                auth=(
                    self.config.get('neo4j_user', 'neo4j'),
                    self.config.get('neo4j_password', 'secure_password')
                ),
                encrypted=False
            )

            # Test connection
            async with self.neo4j.session() as session:
                result = await session.run("RETURN 1")
                await result.consume()

            self.health_status['neo4j'] = True
            self.metrics['active_connections'].labels(database='neo4j').set(1)

            return True

        except Exception as e:
            self.logger.error(f"❌ Neo4j connection failed: {e}")
            return False

    async def _initialize_collections(self):
        """Initialize vector collections and schemas"""
        # Qdrant collections
        if self.qdrant_primary:
            await self._create_qdrant_collections()

        # Weaviate schemas
        if self.weaviate:
            await self._create_weaviate_schemas()

        # PostgreSQL tables
        if self.postgres:
            await self._create_postgres_tables()

    async def _create_qdrant_collections(self):
        """Create Qdrant collections for different data types"""
        collections = [
            {
                'name': 'osint_intel',
                'vector_size': 768,
                'distance': Distance.COSINE,
                'description': 'OSINT intelligence data'
            },
            {
                'name': 'threat_indicators',
                'vector_size': 384,
                'distance': Distance.COSINE,
                'description': 'Threat indicators and IOCs'
            },
            {
                'name': 'social_media',
                'vector_size': 768,
                'distance': Distance.COSINE,
                'description': 'Social media content analysis'
            },
            {
                'name': 'dark_web',
                'vector_size': 768,
                'distance': Distance.COSINE,
                'description': 'Dark web intelligence'
            }
        ]

        for collection_config in collections:
            try:
                # Check if collection exists
                existing = await asyncio.to_thread(
                    self.qdrant_primary.get_collections
                )

                collection_names = [c.name for c in existing.collections]

                if collection_config['name'] not in collection_names:
                    await asyncio.to_thread(
                        self.qdrant_primary.create_collection,
                        collection_name=collection_config['name'],
                        vectors_config=VectorParams(
                            size=collection_config['vector_size'],
                            distance=collection_config['distance']
                        )
                    )

                    self.logger.info(f"✅ Created Qdrant collection: {collection_config['name']}")

            except Exception as e:
                self.logger.error(f"❌ Failed to create Qdrant collection {collection_config['name']}: {e}")

    async def _create_weaviate_schemas(self):
        """Create Weaviate schemas for different data types"""
        schemas = [
            {
                "class": "OSINTIntel",
                "description": "OSINT intelligence documents",
                "vectorizer": "text2vec-transformers",
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "source", "dataType": ["string"]},
                    {"name": "timestamp", "dataType": ["date"]},
                    {"name": "classification", "dataType": ["string"]},
                    {"name": "confidence", "dataType": ["number"]},
                    {"name": "tags", "dataType": ["string[]"]},
                ]
            },
            {
                "class": "ThreatIndicator",
                "description": "Threat indicators and IOCs",
                "vectorizer": "text2vec-transformers",
                "properties": [
                    {"name": "indicator", "dataType": ["text"]},
                    {"name": "type", "dataType": ["string"]},
                    {"name": "malware_family", "dataType": ["string"]},
                    {"name": "severity", "dataType": ["string"]},
                    {"name": "first_seen", "dataType": ["date"]},
                    {"name": "last_seen", "dataType": ["date"]},
                    {"name": "sources", "dataType": ["string[]"]},
                ]
            }
        ]

        for schema in schemas:
            try:
                # Check if class exists
                existing = self.weaviate.schema.get()
                class_names = [cls['class'] for cls in existing.get('classes', [])]

                if schema['class'] not in class_names:
                    self.weaviate.schema.create_class(schema)
                    self.logger.info(f"✅ Created Weaviate schema: {schema['class']}")

            except Exception as e:
                self.logger.error(f"❌ Failed to create Weaviate schema {schema['class']}: {e}")

    async def _create_postgres_tables(self):
        """Create PostgreSQL tables for vector metadata"""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS vector_metadata (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                collection_name VARCHAR(255) NOT NULL,
                vector_id VARCHAR(255) NOT NULL,
                source_type VARCHAR(100),
                source_id VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB,
                tags TEXT[],
                INDEX (collection_name, vector_id),
                INDEX (source_type, source_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                content_hash VARCHAR(64) PRIMARY KEY,
                embedding VECTOR(768),
                model_name VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX (content_hash),
                INDEX (model_name)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS vector_operations_log (
                id SERIAL PRIMARY KEY,
                operation_type VARCHAR(50),
                collection_name VARCHAR(255),
                vector_count INTEGER,
                duration_ms INTEGER,
                status VARCHAR(20),
                error_message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX (operation_type),
                INDEX (collection_name),
                INDEX (created_at)
            );
            """
        ]

        for table_sql in tables:
            try:
                await self.postgres.execute(table_sql)
                self.logger.info("✅ Created PostgreSQL table")
            except Exception as e:
                self.logger.error(f"❌ Failed to create PostgreSQL table: {e}")

    async def _health_monitor(self):
        """Continuous health monitoring for all connections"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Check Qdrant
                if self.qdrant_primary:
                    try:
                        await asyncio.to_thread(self.qdrant_primary.get_collections)
                        self.health_status['qdrant_primary'] = True
                    except:
                        self.health_status['qdrant_primary'] = False
                        self.logger.warning("⚠️ Qdrant primary connection lost")

                # Check Weaviate
                if self.weaviate:
                    try:
                        self.weaviate.schema.get()
                        self.health_status['weaviate'] = True
                    except:
                        self.health_status['weaviate'] = False
                        self.logger.warning("⚠️ Weaviate connection lost")

                # Update metrics
                for db, status in self.health_status.items():
                    self.metrics['active_connections'].labels(database=db).set(1 if status else 0)

            except Exception as e:
                self.logger.error(f"❌ Health monitor error: {e}")

    async def get_preferred_vector_db(self) -> Tuple[str, Any]:
        """Get the best available vector database"""
        if self.health_status['qdrant_primary'] and self.qdrant_primary:
            return 'qdrant_primary', self.qdrant_primary
        elif self.health_status['qdrant_replica'] and self.qdrant_replica:
            return 'qdrant_replica', self.qdrant_replica
        elif self.health_status['weaviate'] and self.weaviate:
            return 'weaviate', self.weaviate
        else:
            raise Exception("No vector databases available")

    async def upsert_vectors(
        self,
        documents: List[EmbeddingDocument],
        collection: str = "default"
    ) -> bool:
        """
        Insert or update vectors in the preferred database

        Args:
            documents: List of documents with embeddings
            collection: Target collection name

        Returns:
            bool: Success status
        """
        start_time = time.time()

        try:
            db_name, db_client = await self.get_preferred_vector_db()

            if db_name.startswith('qdrant'):
                success = await self._upsert_qdrant(db_client, documents, collection)
            else:  # weaviate
                success = await self._upsert_weaviate(db_client, documents, collection)

            duration = time.time() - start_time
            status = 'success' if success else 'error'

            # Update metrics
            self.metrics['vector_operations_total'].labels(
                operation='upsert',
                database=db_name,
                status=status
            ).inc()

            self.metrics['vector_insert_duration'].labels(
                database=db_name
            ).observe(duration)

            # Log operation
            if self.postgres:
                await self._log_operation(
                    operation_type='upsert',
                    collection_name=collection,
                    vector_count=len(documents),
                    duration_ms=int(duration * 1000),
                    status=status
                )

            return success

        except Exception as e:
            self.logger.error(f"❌ Vector upsert failed: {e}")
            self.metrics['vector_operations_total'].labels(
                operation='upsert',
                database='unknown',
                status='error'
            ).inc()
            return False

    async def _upsert_qdrant(
        self,
        client: QdrantClient,
        documents: List[EmbeddingDocument],
        collection: str
    ) -> bool:
        """Upsert vectors in Qdrant"""
        try:
            points = []
            for doc in documents:
                if not doc.vector:
                    continue

                point = PointStruct(
                    id=doc.id,
                    vector=doc.vector,
                    payload={
                        'content': doc.content[:1000],  # Limit content size
                        'metadata': doc.metadata,
                        'timestamp': doc.timestamp.isoformat() if doc.timestamp else None,
                        'collection': doc.collection
                    }
                )
                points.append(point)

            if points:
                # Process in batches
                for i in range(0, len(points), self.batch_size):
                    batch = points[i:i + self.batch_size]
                    await asyncio.to_thread(
                        client.upsert,
                        collection_name=collection,
                        points=batch
                    )

            return True

        except Exception as e:
            self.logger.error(f"❌ Qdrant upsert failed: {e}")
            return False

    async def _upsert_weaviate(
        self,
        client: WeaviateClient,
        documents: List[EmbeddingDocument],
        collection: str
    ) -> bool:
        """Upsert vectors in Weaviate"""
        try:
            # Map collection to Weaviate class
            class_name = collection.replace('_', '').title()

            with client.batch as batch:
                for doc in documents:
                    properties = {
                        'content': doc.content,
                        **doc.metadata
                    }

                    if doc.timestamp:
                        properties['timestamp'] = doc.timestamp.isoformat()

                    batch.add_data_object(
                        data_object=properties,
                        class_name=class_name,
                        uuid=doc.id,
                        vector=doc.vector
                    )

            return True

        except Exception as e:
            self.logger.error(f"❌ Weaviate upsert failed: {e}")
            return False

    async def search_vectors(
        self,
        query_vector: List[float],
        collection: str = "default",
        limit: int = 10,
        score_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors

        Args:
            query_vector: Query vector for similarity search
            collection: Collection to search in
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of search results
        """
        start_time = time.time()

        try:
            db_name, db_client = await self.get_preferred_vector_db()

            if db_name.startswith('qdrant'):
                results = await self._search_qdrant(
                    db_client, query_vector, collection, limit, score_threshold
                )
            else:  # weaviate
                results = await self._search_weaviate(
                    db_client, query_vector, collection, limit, score_threshold
                )

            duration = time.time() - start_time

            # Update metrics
            self.metrics['vector_operations_total'].labels(
                operation='search',
                database=db_name,
                status='success'
            ).inc()

            self.metrics['vector_search_duration'].labels(
                database=db_name
            ).observe(duration)

            return results

        except Exception as e:
            self.logger.error(f"❌ Vector search failed: {e}")
            self.metrics['vector_operations_total'].labels(
                operation='search',
                database='unknown',
                status='error'
            ).inc()
            return []

    async def _search_qdrant(
        self,
        client: QdrantClient,
        query_vector: List[float],
        collection: str,
        limit: int,
        score_threshold: float
    ) -> List[VectorSearchResult]:
        """Search vectors in Qdrant"""
        try:
            search_result = await asyncio.to_thread(
                client.search,
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )

            results = []
            for point in search_result:
                result = VectorSearchResult(
                    id=str(point.id),
                    score=point.score,
                    payload=point.payload,
                    vector=point.vector
                )
                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"❌ Qdrant search failed: {e}")
            return []

    async def _search_weaviate(
        self,
        client: WeaviateClient,
        query_vector: List[float],
        collection: str,
        limit: int,
        score_threshold: float
    ) -> List[VectorSearchResult]:
        """Search vectors in Weaviate"""
        try:
            class_name = collection.replace('_', '').title()

            result = (
                client.query
                .get(class_name, ["content", "_additional {id, certainty}"])
                .with_near_vector({"vector": query_vector})
                .with_limit(limit)
                .with_where({
                    "path": ["_additional", "certainty"],
                    "operator": "GreaterThan",
                    "valueNumber": score_threshold
                })
                .do()
            )

            results = []
            data = result.get('data', {}).get('Get', {}).get(class_name, [])

            for item in data:
                additional = item.get('_additional', {})
                result = VectorSearchResult(
                    id=additional.get('id', ''),
                    score=additional.get('certainty', 0.0),
                    payload=item,
                    metadata={'class': class_name}
                )
                results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"❌ Weaviate search failed: {e}")
            return []

    async def _log_operation(
        self,
        operation_type: str,
        collection_name: str,
        vector_count: int,
        duration_ms: int,
        status: str,
        error_message: str = None
    ):
        """Log vector operations to PostgreSQL"""
        try:
            await self.postgres.execute("""
                INSERT INTO vector_operations_log
                (operation_type, collection_name, vector_count, duration_ms, status, error_message)
                VALUES ($1, $2, $3, $4, $5, $6)
            """, operation_type, collection_name, vector_count, duration_ms, status, error_message)
        except Exception as e:
            self.logger.error(f"❌ Failed to log operation: {e}")

    async def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """Get statistics for a collection"""
        try:
            db_name, db_client = await self.get_preferred_vector_db()

            if db_name.startswith('qdrant'):
                info = await asyncio.to_thread(
                    db_client.get_collection,
                    collection_name=collection
                )
                return {
                    'name': collection,
                    'vector_count': info.vectors_count,
                    'indexed_vectors': info.indexed_vectors_count,
                    'points_count': info.points_count,
                    'segments_count': info.segments_count,
                    'database': 'qdrant'
                }

            else:  # weaviate
                class_name = collection.replace('_', '').title()
                result = client.query.aggregate(class_name).with_meta_count().do()

                count = 0
                if 'data' in result and 'Aggregate' in result['data']:
                    agg_data = result['data']['Aggregate'].get(class_name, [])
                    if agg_data:
                        count = agg_data[0].get('meta', {}).get('count', 0)

                return {
                    'name': collection,
                    'vector_count': count,
                    'database': 'weaviate'
                }

        except Exception as e:
            self.logger.error(f"❌ Failed to get collection stats: {e}")
            return {'name': collection, 'error': str(e)}

    async def close(self):
        """Close all database connections"""
        self.logger.info("🔒 Closing vector database connections")

        if self.postgres:
            await self.postgres.close()

        if self.redis:
            await self.redis.close()

        if self.neo4j:
            await self.neo4j.close()

        # Qdrant and Weaviate clients close automatically

        self.logger.info("✅ All connections closed")


# Usage Example and Configuration
async def main():
    """Example usage of Vector Database Manager"""

    config = {
        'qdrant_primary_host': '172.30.0.36',
        'qdrant_primary_port': 6333,
        'qdrant_replica_host': '172.30.0.37',
        'qdrant_replica_port': 6336,
        'weaviate_host': '172.30.0.38',
        'weaviate_api_key': 'default-key',
        'postgres_host': '172.30.0.2',
        'postgres_user': 'bev_user',
        'postgres_password': 'secure_password',
        'redis_host': '172.30.0.4',
        'neo4j_host': '172.30.0.3',
        'batch_size': 32,
        'connection_pool_size': 10
    }

    # Initialize manager
    manager = VectorDatabaseManager(config)

    if await manager.initialize():
        print("✅ Vector Database Manager ready")

        # Example: Insert vectors
        documents = [
            EmbeddingDocument(
                id=str(uuid.uuid4()),
                content="Sample OSINT intelligence document",
                metadata={'source': 'twitter', 'classification': 'unclassified'},
                vector=[0.1] * 768,  # Example embedding
                collection='osint_intel'
            )
        ]

        await manager.upsert_vectors(documents, 'osint_intel')

        # Example: Search vectors
        query_vector = [0.1] * 768
        results = await manager.search_vectors(query_vector, 'osint_intel')

        print(f"Found {len(results)} similar vectors")

        await manager.close()


if __name__ == "__main__":
    asyncio.run(main())