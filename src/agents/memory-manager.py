#!/usr/bin/env python3
"""
Memory Keeper - Multi-Tier Autonomous Memory Management
Handles Neo4j graphs, pgvector embeddings, and Redis caching
"""

import asyncio
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from neo4j import AsyncGraphDatabase
import asyncpg
import redis.asyncio as redis
import aiofiles
from sentence_transformers import SentenceTransformer
import faiss
import msgpack
import lz4.frame

@dataclass
class MemoryNode:
    """Knowledge graph node"""
    id: str
    type: str
    content: Dict[str, Any]
    embeddings: np.ndarray
    relationships: List[Dict[str, Any]]
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    importance_score: float = 1.0
    decay_rate: float = 0.01

@dataclass
class MemoryQuery:
    """Query for memory retrieval"""
    query_text: str
    query_type: str  # 'semantic', 'graph', 'exact', 'temporal'
    filters: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 10
    include_relationships: bool = True
    time_range: Optional[Tuple[datetime, datetime]] = None

class EmbeddingManager:
    """Manages vector embeddings for semantic search"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = 384  # Model output dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_map = {}
        self.index_counter = 0
        
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        return await asyncio.to_thread(self.model.encode, text)
    
    async def add_to_index(self, node_id: str, embedding: np.ndarray):
        """Add embedding to FAISS index"""
        self.index.add(embedding.reshape(1, -1))
        self.id_map[self.index_counter] = node_id
        self.index_counter += 1
    
    async def search_similar(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar embeddings"""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx in self.id_map:
                results.append((self.id_map[idx], float(dist)))
        
        return results

class GraphMemory:
    """Neo4j graph database manager"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        
    async def initialize_schema(self):
        """Create indexes and constraints"""
        async with self.driver.session() as session:
            # Create constraints
            await session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) 
                REQUIRE n.id IS UNIQUE
            """)
            
            await session.run("""
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:Concept) 
                REQUIRE n.id IS UNIQUE
            """)
            
            # Create indexes
            await session.run("""
                CREATE INDEX IF NOT EXISTS FOR (n:Entity) 
                ON (n.type, n.created_at)
            """)
            
            await session.run("""
                CREATE INDEX IF NOT EXISTS FOR (n:Concept) 
                ON (n.importance_score)
            """)
    
    async def create_node(self, node: MemoryNode) -> str:
        """Create node in graph"""
        async with self.driver.session() as session:
            result = await session.run("""
                CREATE (n:Entity {
                    id: $id,
                    type: $type,
                    content: $content,
                    created_at: datetime(),
                    importance_score: $importance,
                    access_count: 0
                })
                RETURN n.id as id
            """, {
                'id': node.id,
                'type': node.type,
                'content': json.dumps(node.content),
                'importance': node.importance_score
            })
            
            record = await result.single()
            return record['id']
    
    async def create_relationship(self, source_id: str, target_id: str, 
                                rel_type: str, properties: Dict = None):
        """Create relationship between nodes"""
        async with self.driver.session() as session:
            await session.run(f"""
                MATCH (a:Entity {{id: $source_id}})
                MATCH (b:Entity {{id: $target_id}})
                CREATE (a)-[r:{rel_type} $properties]->(b)
            """, {
                'source_id': source_id,
                'target_id': target_id,
                'properties': properties or {}
            })
    
    async def traverse_graph(self, start_id: str, depth: int = 3) -> List[Dict]:
        """Traverse graph from starting node"""
        async with self.driver.session() as session:
            result = await session.run("""
                MATCH path = (start:Entity {id: $start_id})-[*1..$depth]-(connected)
                RETURN path, connected.id as id, connected.type as type,
                       connected.content as content
                LIMIT 100
            """, {
                'start_id': start_id,
                'depth': depth
            })
            
            nodes = []
            async for record in result:
                nodes.append({
                    'id': record['id'],
                    'type': record['type'],
                    'content': json.loads(record['content'])
                })
            
            return nodes
    
    async def update_access_stats(self, node_id: str):
        """Update node access statistics"""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (n:Entity {id: $node_id})
                SET n.access_count = n.access_count + 1,
                    n.last_accessed = datetime()
            """, {'node_id': node_id})
    
    async def prune_old_memories(self, days: int = 90):
        """Remove old, unimportant memories"""
        async with self.driver.session() as session:
            await session.run("""
                MATCH (n:Entity)
                WHERE n.last_accessed < datetime() - duration({days: $days})
                  AND n.importance_score < 0.3
                  AND n.access_count < 5
                DETACH DELETE n
            """, {'days': days})

class VectorMemory:
    """PostgreSQL with pgvector for embeddings"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
        
    async def initialize(self):
        """Initialize connection pool and schema"""
        self.pool = await asyncpg.create_pool(self.connection_string)
        
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
            
            # Create memories table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content JSONB NOT NULL,
                    embedding vector(384),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    accessed_at TIMESTAMPTZ DEFAULT NOW(),
                    importance FLOAT DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)
            
            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding 
                ON memories USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type 
                ON memories(type)
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_importance 
                ON memories(importance DESC)
            """)
    
    async def store_memory(self, node: MemoryNode):
        """Store memory with embedding"""
        async with self.pool.acquire() as conn:
            embedding_list = node.embeddings.tolist()
            
            await conn.execute("""
                INSERT INTO memories (id, type, content, embedding, importance, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    accessed_at = NOW(),
                    access_count = memories.access_count + 1
            """, node.id, node.type, json.dumps(node.content), 
                embedding_list, node.importance_score, {})
    
    async def semantic_search(self, query_embedding: np.ndarray, 
                            limit: int = 10) -> List[Dict]:
        """Search by semantic similarity"""
        async with self.pool.acquire() as conn:
            embedding_list = query_embedding.tolist()
            
            results = await conn.fetch("""
                SELECT id, type, content, 
                       1 - (embedding <=> $1::vector) as similarity,
                       importance, access_count
                FROM memories
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """, embedding_list, limit)
            
            return [dict(r) for r in results]
    
    async def hybrid_search(self, query_embedding: np.ndarray, 
                          filters: Dict, limit: int = 10) -> List[Dict]:
        """Combined semantic and metadata search"""
        async with self.pool.acquire() as conn:
            embedding_list = query_embedding.tolist()
            
            # Build dynamic query with filters
            where_clauses = []
            params = [embedding_list, limit]
            param_count = 2
            
            for key, value in filters.items():
                param_count += 1
                where_clauses.append(f"content->>${param_count}::text = ${param_count+1}")
                params.extend([key, value])
                param_count += 1
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "TRUE"
            
            query = f"""
                SELECT id, type, content,
                       1 - (embedding <=> $1::vector) as similarity,
                       importance
                FROM memories
                WHERE {where_sql}
                ORDER BY embedding <=> $1::vector
                LIMIT $2
            """
            
            results = await conn.fetch(query, *params)
            return [dict(r) for r in results]

class CacheManager:
    """Redis-based caching layer"""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url, decode_responses=False)
        self.ttl = 3600  # 1 hour default TTL
        
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        data = await self.redis.get(key)
        if data:
            return msgpack.unpackb(lz4.frame.decompress(data), raw=False)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        """Store in cache with compression"""
        compressed = lz4.frame.compress(msgpack.packb(value))
        await self.redis.setex(key, ttl or self.ttl, compressed)
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(cursor, match=pattern)
            if keys:
                await self.redis.delete(*keys)
            if cursor == 0:
                break

class MemoryKeeper:
    """Main memory management orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        self.graph = GraphMemory(
            config['neo4j_uri'],
            config['neo4j_user'],
            config['neo4j_password']
        )
        self.vector = VectorMemory(config['postgres_uri'])
        self.cache = CacheManager(config['redis_url'])
        self.embeddings = EmbeddingManager()
        
        self.memory_tiers = {
            'hot': [],      # Most recent, frequently accessed
            'warm': [],     # Recent or moderately accessed
            'cold': [],     # Old, rarely accessed
            'archive': []   # Very old, candidate for deletion
        }
        
    async def initialize(self):
        """Initialize all storage systems"""
        await self.graph.initialize_schema()
        await self.vector.initialize()
        print("✅ Memory systems initialized")
    
    async def store(self, content: Any, memory_type: str = 'general',
                   relationships: List[Dict] = None) -> str:
        """Store memory across all tiers"""
        # Generate unique ID
        content_str = json.dumps(content, sort_keys=True)
        memory_id = hashlib.sha256(content_str.encode()).hexdigest()[:16]
        
        # Generate embedding
        embedding = await self.embeddings.generate_embedding(content_str)
        
        # Create memory node
        node = MemoryNode(
            id=memory_id,
            type=memory_type,
            content=content if isinstance(content, dict) else {'data': content},
            embeddings=embedding,
            relationships=relationships or []
        )
        
        # Store in graph database
        await self.graph.create_node(node)
        
        # Store in vector database
        await self.vector.store_memory(node)
        
        # Add to embedding index
        await self.embeddings.add_to_index(memory_id, embedding)
        
        # Cache for quick access
        await self.cache.set(f"memory:{memory_id}", content)
        
        # Add to hot tier
        self.memory_tiers['hot'].append(memory_id)
        
        print(f"💾 Stored memory {memory_id} across all tiers")
        return memory_id
    
    async def recall(self, query: MemoryQuery) -> List[Dict[str, Any]]:
        """Retrieve memories based on query"""
        # Check cache first
        cache_key = f"query:{hashlib.md5(str(query).encode()).hexdigest()}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        results = []
        
        if query.query_type == 'semantic':
            # Semantic search using embeddings
            query_embedding = await self.embeddings.generate_embedding(query.query_text)
            
            # Search in FAISS index
            similar_ids = await self.embeddings.search_similar(
                query_embedding, 
                query.max_results * 2  # Get more for filtering
            )
            
            # Retrieve full memories from vector DB
            vector_results = await self.vector.semantic_search(
                query_embedding,
                query.max_results
            )
            
            results = vector_results
            
        elif query.query_type == 'graph':
            # Graph traversal search
            if 'start_node' in query.filters:
                graph_results = await self.graph.traverse_graph(
                    query.filters['start_node'],
                    query.filters.get('depth', 3)
                )
                results = graph_results[:query.max_results]
        
        elif query.query_type == 'hybrid':
            # Combined semantic + metadata search
            query_embedding = await self.embeddings.generate_embedding(query.query_text)
            results = await self.vector.hybrid_search(
                query_embedding,
                query.filters,
                query.max_results
            )
        
        # Update access statistics
        for result in results:
            await self.graph.update_access_stats(result.get('id'))
        
        # Cache results
        await self.cache.set(cache_key, results, ttl=300)
        
        return results
    
    async def consolidate_memories(self):
        """Consolidate related memories for efficiency"""
        # Find clusters of related memories
        all_embeddings = []
        memory_ids = []
        
        async with self.vector.pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT id, embedding FROM memories
                WHERE importance > 0.5
                LIMIT 1000
            """)
            
            for row in results:
                memory_ids.append(row['id'])
                all_embeddings.append(np.array(row['embedding']))
        
        if len(all_embeddings) > 10:
            # Cluster embeddings
            embeddings_matrix = np.vstack(all_embeddings)
            
            # Use DBSCAN for clustering
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.3, min_samples=3)
            clusters = clustering.fit_predict(embeddings_matrix)
            
            # Merge memories in same cluster
            for cluster_id in set(clusters):
                if cluster_id != -1:  # Skip noise points
                    cluster_indices = np.where(clusters == cluster_id)[0]
                    cluster_memory_ids = [memory_ids[i] for i in cluster_indices]
                    
                    # Create consolidated memory
                    await self._merge_memories(cluster_memory_ids)
    
    async def _merge_memories(self, memory_ids: List[str]):
        """Merge multiple memories into consolidated memory"""
        consolidated_content = {
            'merged_from': memory_ids,
            'merged_at': datetime.now().isoformat(),
            'sub_memories': []
        }
        
        async with self.vector.pool.acquire() as conn:
            for mem_id in memory_ids:
                result = await conn.fetchrow(
                    "SELECT content FROM memories WHERE id = $1",
                    mem_id
                )
                if result:
                    consolidated_content['sub_memories'].append(result['content'])
        
        # Store consolidated memory
        await self.store(consolidated_content, 'consolidated')
    
    async def age_memories(self):
        """Move memories between tiers based on access patterns"""
        # Analyze access patterns
        async with self.vector.pool.acquire() as conn:
            # Move hot to warm (accessed < 1 hour ago)
            await conn.execute("""
                UPDATE memories 
                SET metadata = jsonb_set(metadata, '{tier}', '"warm"')
                WHERE accessed_at < NOW() - INTERVAL '1 hour'
                  AND metadata->>'tier' = 'hot'
            """)
            
            # Move warm to cold (accessed < 1 day ago)
            await conn.execute("""
                UPDATE memories
                SET metadata = jsonb_set(metadata, '{tier}', '"cold"')
                WHERE accessed_at < NOW() - INTERVAL '1 day'
                  AND metadata->>'tier' = 'warm'
            """)
            
            # Move cold to archive (accessed < 1 week ago)
            await conn.execute("""
                UPDATE memories
                SET metadata = jsonb_set(metadata, '{tier}', '"archive"')
                WHERE accessed_at < NOW() - INTERVAL '7 days'
                  AND metadata->>'tier' = 'cold'
                  AND importance < 0.5
            """)
    
    async def autonomous_optimization_loop(self):
        """Continuously optimize memory storage"""
        while True:
            try:
                # Age memories between tiers
                await self.age_memories()
                
                # Consolidate related memories
                await self.consolidate_memories()
                
                # Prune old memories from graph
                await self.graph.prune_old_memories()
                
                # Clear expired cache entries
                await self.cache.invalidate_pattern("query:*")
                
                print("🔄 Memory optimization cycle complete")
                
            except Exception as e:
                print(f"❌ Memory optimization error: {e}")
            
            await asyncio.sleep(3600)  # Run hourly

# Example usage
async def main():
    config = {
        'neo4j_uri': 'neo4j://thanos:7687',
        'neo4j_user': 'neo4j',
        'neo4j_password': 'research_graph_2024',
        'postgres_uri': 'postgresql://researcher:secure_research_2024@thanos:5432/research_db',
        'redis_url': 'redis://thanos:6379'
    }
    
    keeper = MemoryKeeper(config)
    await keeper.initialize()
    
    # Store some memories
    memory_id = await keeper.store({
        'research': 'Advanced watermark removal techniques',
        'findings': 'Frequency domain analysis shows promise',
        'timestamp': datetime.now().isoformat()
    }, memory_type='research')
    
    # Recall memories
    query = MemoryQuery(
        query_text="watermark removal techniques",
        query_type="semantic",
        max_results=5
    )
    
    results = await keeper.recall(query)
    print(f"Found {len(results)} relevant memories")
    
    # Start optimization loop
    asyncio.create_task(keeper.autonomous_optimization_loop())

if __name__ == "__main__":
    asyncio.run(main())
