import os
"""
ORACLE1 Memory Manager - Vector-based Memory System
Neo4j Graph + Redis Cache + PostgreSQL Persistence
365-day retention with intelligent memory compression
"""

import asyncio
import hashlib
import json
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import neo4j
import numpy as np
import pandas as pd
import redis.asyncio as redis
import structlog
from fastapi import FastAPI, HTTPException
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure structured logging
logger = structlog.get_logger()

# Metrics
memory_operations = Counter('memory_operations_total', 'Total memory operations', ['operation'])
memory_size = Gauge('memory_size_bytes', 'Memory size in bytes', ['type'])
query_latency = Histogram('memory_query_latency_seconds', 'Memory query latency')
cache_hits = Counter('cache_hits_total', 'Cache hit count')
cache_misses = Counter('cache_misses_total', 'Cache miss count')

app = FastAPI(title="ORACLE1 Memory Manager", version="3.0.0")

# Database models
Base = declarative_base()


class MemoryType(str, Enum):
    """Types of memory storage"""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    LONG_TERM = "long_term"
    SHORT_TERM = "short_term"


class MemoryPriority(str, Enum):
    """Memory priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ARCHIVE = "archive"


@dataclass
class Memory:
    """Represents a memory unit"""
    id: str
    type: MemoryType
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    priority: MemoryPriority = MemoryPriority.MEDIUM
    ttl: Optional[int] = None  # Time to live in seconds
    relationships: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)


class MemoryRecord(Base):
    """SQLAlchemy model for PostgreSQL storage"""
    __tablename__ = "memories"

    id = Column(String, primary_key=True)
    type = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(JSONB)
    metadata = Column(JSONB)
    timestamp = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    priority = Column(String, default="medium")
    ttl = Column(Integer, nullable=True)
    relationships = Column(JSONB, default=list)
    tags = Column(JSONB, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class VectorMemoryStore:
    """Vector-based memory storage using FAISS"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map = {}
        self.reverse_map = {}
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def add_memory(self, memory: Memory):
        """Add memory to vector store"""
        if memory.embedding is None:
            memory.embedding = self.embedder.encode(memory.content)

        # Add to FAISS index
        self.index.add(np.array([memory.embedding]))
        idx = self.index.ntotal - 1
        self.id_map[idx] = memory.id
        self.reverse_map[memory.id] = idx

    def search_similar(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar memories"""
        query_embedding = self.embedder.encode(query)
        distances, indices = self.index.search(np.array([query_embedding]), k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx in self.id_map:
                memory_id = self.id_map[idx]
                distance = distances[0][i]
                results.append((memory_id, float(distance)))

        return results

    def remove_memory(self, memory_id: str):
        """Remove memory from vector store"""
        if memory_id in self.reverse_map:
            # FAISS doesn't support removal, so we track deleted indices
            idx = self.reverse_map[memory_id]
            del self.id_map[idx]
            del self.reverse_map[memory_id]


class GraphMemoryStore:
    """Graph-based memory storage using Neo4j"""

    def __init__(self, uri: str, user: str, password: str):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

    async def add_memory_node(self, memory: Memory):
        """Add memory as node in graph"""
        with self.driver.session() as session:
            query = """
            CREATE (m:Memory {
                id: $id,
                type: $type,
                content: $content,
                priority: $priority,
                timestamp: $timestamp
            })
            """
            session.run(
                query,
                id=memory.id,
                type=memory.type.value,
                content=memory.content,
                priority=memory.priority.value,
                timestamp=memory.timestamp.isoformat()
            )

    async def add_relationship(self, from_id: str, to_id: str, relationship_type: str):
        """Add relationship between memories"""
        with self.driver.session() as session:
            query = """
            MATCH (a:Memory {id: $from_id})
            MATCH (b:Memory {id: $to_id})
            CREATE (a)-[r:RELATES {type: $rel_type}]->(b)
            """
            session.run(
                query,
                from_id=from_id,
                to_id=to_id,
                rel_type=relationship_type
            )

    async def find_connected_memories(self, memory_id: str, depth: int = 2) -> List[Dict]:
        """Find memories connected to given memory"""
        with self.driver.session() as session:
            query = """
            MATCH (m:Memory {id: $id})-[*1..$depth]-(connected:Memory)
            RETURN DISTINCT connected
            """
            result = session.run(query, id=memory_id, depth=depth)
            return [dict(record["connected"]) for record in result]

    async def find_paths(self, start_id: str, end_id: str) -> List[List[str]]:
        """Find paths between two memories"""
        with self.driver.session() as session:
            query = """
            MATCH path = shortestPath((a:Memory {id: $start})-[*]-(b:Memory {id: $end}))
            RETURN [n in nodes(path) | n.id] as path_ids
            """
            result = session.run(query, start=start_id, end=end_id)
            paths = []
            for record in result:
                paths.append(record["path_ids"])
            return paths

    def close(self):
        """Close Neo4j connection"""
        self.driver.close()


class MemoryCompressor:
    """Memory compression and consolidation"""

    def __init__(self):
        self.compression_ratio = 0.5
        self.min_similarity = 0.85

    async def compress_memories(self, memories: List[Memory]) -> List[Memory]:
        """Compress similar memories"""
        if len(memories) < 2:
            return memories

        # Group similar memories
        groups = self._cluster_memories(memories)

        # Compress each group
        compressed = []
        for group in groups:
            if len(group) > 1:
                compressed.append(self._merge_memories(group))
            else:
                compressed.extend(group)

        return compressed

    def _cluster_memories(self, memories: List[Memory]) -> List[List[Memory]]:
        """Cluster similar memories"""
        # Simple clustering based on content similarity
        clusters = []
        used = set()

        for i, mem1 in enumerate(memories):
            if i in used:
                continue

            cluster = [mem1]
            used.add(i)

            for j, mem2 in enumerate(memories[i+1:], i+1):
                if j not in used:
                    similarity = self._calculate_similarity(mem1, mem2)
                    if similarity > self.min_similarity:
                        cluster.append(mem2)
                        used.add(j)

            clusters.append(cluster)

        return clusters

    def _calculate_similarity(self, mem1: Memory, mem2: Memory) -> float:
        """Calculate similarity between two memories"""
        # Simplified similarity calculation
        if mem1.embedding is not None and mem2.embedding is not None:
            return float(np.dot(mem1.embedding, mem2.embedding) /
                        (np.linalg.norm(mem1.embedding) * np.linalg.norm(mem2.embedding)))
        return 0.0

    def _merge_memories(self, memories: List[Memory]) -> Memory:
        """Merge multiple memories into one"""
        # Create consolidated memory
        contents = [m.content for m in memories]
        merged_content = " | ".join(contents)

        # Combine metadata
        merged_metadata = {}
        for mem in memories:
            merged_metadata.update(mem.metadata)

        # Use highest priority
        max_priority = max(memories, key=lambda m: self._priority_value(m.priority))

        return Memory(
            id=hashlib.sha256(merged_content.encode()).hexdigest()[:16],
            type=memories[0].type,
            content=merged_content,
            metadata=merged_metadata,
            priority=max_priority.priority,
            tags=set().union(*[m.tags for m in memories])
        )

    def _priority_value(self, priority: MemoryPriority) -> int:
        """Get numeric value for priority"""
        values = {
            MemoryPriority.CRITICAL: 5,
            MemoryPriority.HIGH: 4,
            MemoryPriority.MEDIUM: 3,
            MemoryPriority.LOW: 2,
            MemoryPriority.ARCHIVE: 1
        }
        return values.get(priority, 0)


class MemoryManager:
    """Main memory management system"""

    def __init__(self):
        self.vector_store = VectorMemoryStore()
        self.graph_store = None
        self.redis_client = None
        self.db_engine = None
        self.compressor = MemoryCompressor()
        self.memories: Dict[str, Memory] = {}
        self.working_memory = deque(maxlen=100)
        self.retention_days = 365

    async def initialize(self):
        """Initialize memory manager"""
        try:
            # Initialize Redis
            self.redis_client = await redis.from_url(
                "redis://localhost:6379",
                encoding="utf-8",
                decode_responses=False
            )

            # Initialize Neo4j
            self.graph_store = GraphMemoryStore(
                uri="bolt://localhost:7687",
                user="neo4j",
                password=os.getenv('DB_PASSWORD', 'dev_password')
            )

            # Initialize PostgreSQL
            self.db_engine = create_async_engine(
                "postgresql+asyncpg://memory_user:password@localhost/memory"
            )

            # Create tables
            async with self.db_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            # Start background tasks
            asyncio.create_task(self._memory_maintenance())
            asyncio.create_task(self._cache_sync())

            logger.info("Memory manager initialized")

        except Exception as e:
            logger.error("Failed to initialize memory manager", error=str(e))
            raise

    async def store_memory(self, memory: Memory) -> str:
        """Store a new memory"""
        start_time = time.time()

        try:
            # Add to vector store
            self.vector_store.add_memory(memory)

            # Add to graph store
            await self.graph_store.add_memory_node(memory)

            # Cache in Redis
            await self._cache_memory(memory)

            # Persist to PostgreSQL
            await self._persist_memory(memory)

            # Add to local storage
            self.memories[memory.id] = memory

            # Add to working memory if high priority
            if memory.priority in [MemoryPriority.CRITICAL, MemoryPriority.HIGH]:
                self.working_memory.append(memory.id)

            memory_operations.labels(operation="store").inc()
            query_latency.observe(time.time() - start_time)

            return memory.id

        except Exception as e:
            logger.error("Failed to store memory", error=str(e))
            raise

    async def recall_memory(self, memory_id: str) -> Optional[Memory]:
        """Recall a specific memory"""
        # Check cache first
        cached = await self._get_cached_memory(memory_id)
        if cached:
            cache_hits.inc()
            return cached

        cache_misses.inc()

        # Check local storage
        if memory_id in self.memories:
            memory = self.memories[memory_id]
            memory.access_count += 1
            await self._cache_memory(memory)
            return memory

        # Load from database
        memory = await self._load_from_database(memory_id)
        if memory:
            self.memories[memory_id] = memory
            memory.access_count += 1
            await self._cache_memory(memory)

        return memory

    async def search_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 10
    ) -> List[Memory]:
        """Search for relevant memories"""
        start_time = time.time()

        # Vector similarity search
        similar = self.vector_store.search_similar(query, limit * 2)

        # Filter by type if specified
        results = []
        for memory_id, score in similar:
            memory = await self.recall_memory(memory_id)
            if memory and (memory_type is None or memory.type == memory_type):
                results.append(memory)
                if len(results) >= limit:
                    break

        query_latency.observe(time.time() - start_time)
        memory_operations.labels(operation="search").inc()

        return results

    async def find_related_memories(self, memory_id: str, depth: int = 2) -> List[Memory]:
        """Find memories related to given memory"""
        # Get connected memories from graph
        connected = await self.graph_store.find_connected_memories(memory_id, depth)

        # Load full memory objects
        related = []
        for node_data in connected:
            memory = await self.recall_memory(node_data["id"])
            if memory:
                related.append(memory)

        return related

    async def consolidate_memories(self, memory_type: MemoryType):
        """Consolidate memories of specific type"""
        # Get memories of type
        type_memories = [m for m in self.memories.values() if m.type == memory_type]

        # Compress similar memories
        compressed = await self.compressor.compress_memories(type_memories)

        # Replace with compressed versions
        for memory in compressed:
            await self.store_memory(memory)

        # Remove old memories
        for old_memory in type_memories:
            if old_memory.id not in [m.id for m in compressed]:
                await self.forget_memory(old_memory.id)

        logger.info(f"Consolidated {len(type_memories)} memories to {len(compressed)}")

    async def forget_memory(self, memory_id: str):
        """Remove a memory"""
        try:
            # Remove from all stores
            self.vector_store.remove_memory(memory_id)

            # Remove from cache
            await self.redis_client.delete(f"memory:{memory_id}")

            # Remove from local storage
            if memory_id in self.memories:
                del self.memories[memory_id]

            memory_operations.labels(operation="forget").inc()

        except Exception as e:
            logger.error("Failed to forget memory", memory_id=memory_id, error=str(e))

    async def _cache_memory(self, memory: Memory):
        """Cache memory in Redis"""
        try:
            # Serialize memory
            data = {
                "id": memory.id,
                "type": memory.type.value,
                "content": memory.content,
                "metadata": json.dumps(memory.metadata),
                "priority": memory.priority.value,
                "timestamp": memory.timestamp.isoformat(),
                "access_count": memory.access_count
            }

            # Set with TTL based on priority
            ttl = self._get_cache_ttl(memory.priority)
            await self.redis_client.setex(
                f"memory:{memory.id}",
                ttl,
                pickle.dumps(data)
            )

        except Exception as e:
            logger.error("Failed to cache memory", error=str(e))

    async def _get_cached_memory(self, memory_id: str) -> Optional[Memory]:
        """Get memory from cache"""
        try:
            data = await self.redis_client.get(f"memory:{memory_id}")
            if data:
                memory_data = pickle.loads(data)
                return Memory(
                    id=memory_data["id"],
                    type=MemoryType(memory_data["type"]),
                    content=memory_data["content"],
                    metadata=json.loads(memory_data["metadata"]),
                    priority=MemoryPriority(memory_data["priority"]),
                    timestamp=datetime.fromisoformat(memory_data["timestamp"]),
                    access_count=memory_data["access_count"]
                )
        except Exception as e:
            logger.error("Failed to get cached memory", error=str(e))
        return None

    def _get_cache_ttl(self, priority: MemoryPriority) -> int:
        """Get cache TTL based on priority"""
        ttl_map = {
            MemoryPriority.CRITICAL: 86400 * 7,  # 7 days
            MemoryPriority.HIGH: 86400 * 3,      # 3 days
            MemoryPriority.MEDIUM: 86400,        # 1 day
            MemoryPriority.LOW: 3600,            # 1 hour
            MemoryPriority.ARCHIVE: 300           # 5 minutes
        }
        return ttl_map.get(priority, 3600)

    async def _persist_memory(self, memory: Memory):
        """Persist memory to PostgreSQL"""
        try:
            async with AsyncSession(self.db_engine) as session:
                record = MemoryRecord(
                    id=memory.id,
                    type=memory.type.value,
                    content=memory.content,
                    embedding=memory.embedding.tolist() if memory.embedding is not None else None,
                    metadata=memory.metadata,
                    timestamp=memory.timestamp,
                    access_count=memory.access_count,
                    priority=memory.priority.value,
                    ttl=memory.ttl,
                    relationships=memory.relationships,
                    tags=list(memory.tags)
                )
                session.add(record)
                await session.commit()

        except Exception as e:
            logger.error("Failed to persist memory", error=str(e))

    async def _load_from_database(self, memory_id: str) -> Optional[Memory]:
        """Load memory from PostgreSQL"""
        try:
            async with AsyncSession(self.db_engine) as session:
                result = await session.execute(
                    f"SELECT * FROM memories WHERE id = '{memory_id}'"
                )
                record = result.first()

                if record:
                    return Memory(
                        id=record.id,
                        type=MemoryType(record.type),
                        content=record.content,
                        embedding=np.array(record.embedding) if record.embedding else None,
                        metadata=record.metadata or {},
                        timestamp=record.timestamp,
                        access_count=record.access_count,
                        priority=MemoryPriority(record.priority),
                        ttl=record.ttl,
                        relationships=record.relationships or [],
                        tags=set(record.tags or [])
                    )
        except Exception as e:
            logger.error("Failed to load from database", error=str(e))
        return None

    async def _memory_maintenance(self):
        """Background task for memory maintenance"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run hourly

                # Clean expired memories
                cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
                expired = [
                    m.id for m in self.memories.values()
                    if m.timestamp < cutoff_date and m.priority == MemoryPriority.LOW
                ]

                for memory_id in expired:
                    await self.forget_memory(memory_id)

                # Consolidate if needed
                if len(self.memories) > 10000:
                    await self.consolidate_memories(MemoryType.SEMANTIC)

                logger.info(f"Memory maintenance: removed {len(expired)} expired memories")

            except Exception as e:
                logger.error("Memory maintenance failed", error=str(e))

    async def _cache_sync(self):
        """Sync cache with database"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Update memory sizes
                memory_size.labels(type="vector").set(self.vector_store.index.ntotal)
                memory_size.labels(type="local").set(len(self.memories))
                memory_size.labels(type="working").set(len(self.working_memory))

            except Exception as e:
                logger.error("Cache sync failed", error=str(e))

    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = {
            "total_memories": len(self.memories),
            "vector_store_size": self.vector_store.index.ntotal,
            "working_memory_size": len(self.working_memory),
            "memory_types": {},
            "priority_distribution": {}
        }

        # Count by type
        for memory in self.memories.values():
            stats["memory_types"][memory.type.value] = \
                stats["memory_types"].get(memory.type.value, 0) + 1
            stats["priority_distribution"][memory.priority.value] = \
                stats["priority_distribution"].get(memory.priority.value, 0) + 1

        return stats

    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            await self.redis_client.close()
        if self.graph_store:
            self.graph_store.close()
        if self.db_engine:
            await self.db_engine.dispose()


# Global memory manager instance
memory_manager = MemoryManager()


@app.on_event("startup")
async def startup():
    """Initialize memory manager on startup"""
    await memory_manager.initialize()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    await memory_manager.cleanup()


@app.get("/health")
async def health():
    """Health check endpoint"""
    stats = await memory_manager.get_statistics()
    return {"status": "healthy", **stats}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


class MemoryRequest(BaseModel):
    """Memory storage request"""
    content: str
    type: MemoryType = MemoryType.SEMANTIC
    priority: MemoryPriority = MemoryPriority.MEDIUM
    metadata: Dict[str, Any] = {}
    tags: List[str] = []
    ttl: Optional[int] = None


@app.post("/memory/store")
async def store_memory(request: MemoryRequest):
    """Store a new memory"""
    memory = Memory(
        id=hashlib.sha256(f"{request.content}{time.time()}".encode()).hexdigest()[:16],
        type=request.type,
        content=request.content,
        priority=request.priority,
        metadata=request.metadata,
        tags=set(request.tags),
        ttl=request.ttl
    )

    memory_id = await memory_manager.store_memory(memory)
    return {"memory_id": memory_id, "status": "stored"}


@app.get("/memory/{memory_id}")
async def get_memory(memory_id: str):
    """Retrieve a memory"""
    memory = await memory_manager.recall_memory(memory_id)

    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    return {
        "id": memory.id,
        "type": memory.type.value,
        "content": memory.content,
        "metadata": memory.metadata,
        "priority": memory.priority.value,
        "access_count": memory.access_count,
        "timestamp": memory.timestamp.isoformat()
    }


@app.post("/memory/search")
async def search_memories(query: str, memory_type: Optional[MemoryType] = None, limit: int = 10):
    """Search for memories"""
    results = await memory_manager.search_memories(query, memory_type, limit)

    return {
        "query": query,
        "results": [
            {
                "id": m.id,
                "content": m.content,
                "type": m.type.value,
                "priority": m.priority.value
            }
            for m in results
        ]
    }


@app.get("/memory/{memory_id}/related")
async def get_related_memories(memory_id: str, depth: int = 2):
    """Get memories related to given memory"""
    related = await memory_manager.find_related_memories(memory_id, depth)

    return {
        "memory_id": memory_id,
        "related": [
            {
                "id": m.id,
                "content": m.content,
                "type": m.type.value
            }
            for m in related
        ]
    }


@app.delete("/memory/{memory_id}")
async def forget_memory(memory_id: str):
    """Forget a memory"""
    await memory_manager.forget_memory(memory_id)
    return {"status": "forgotten", "memory_id": memory_id}


@app.post("/memory/consolidate")
async def consolidate_memories(memory_type: MemoryType):
    """Consolidate memories of specific type"""
    await memory_manager.consolidate_memories(memory_type)
    return {"status": "consolidated", "type": memory_type.value}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300)