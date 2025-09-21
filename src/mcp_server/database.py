"""
Database Integration Manager
===========================

Secure database connections with connection pooling for PostgreSQL, Neo4j, Redis, and Elasticsearch.
"""

import asyncio
import asyncpg
import redis.asyncio as aioredis
from neo4j import AsyncGraphDatabase
from elasticsearch import AsyncElasticsearch
import logging
from typing import Dict, List, Optional, Any, AsyncContextManager
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from datetime import datetime

from .models import OSINTResult, ThreatIntelligence, GraphNode, GraphRelationship, CryptoTransaction
from .security import SecurityManager, InputValidator


logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    postgres_uri: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    elasticsearch_host: str = "elasticsearch"
    elasticsearch_port: int = 9200
    
    # Connection pool settings
    postgres_min_connections: int = 5
    postgres_max_connections: int = 20
    redis_max_connections: int = 20
    
    # Timeouts
    connection_timeout: int = 30
    query_timeout: int = 300


class DatabaseError(Exception):
    """Database operation errors"""
    pass


class PostgreSQLManager:
    """PostgreSQL connection manager with security features"""
    
    def __init__(self, config: DatabaseConfig, security_manager: SecurityManager):
        self.config = config
        self.security = security_manager
        self.pool: Optional[asyncpg.Pool] = None
        self.validator = InputValidator()
    
    async def initialize(self):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.config.postgres_uri,
                min_size=self.config.postgres_min_connections,
                max_size=self.config.postgres_max_connections,
                command_timeout=self.config.query_timeout,
                server_settings={
                    'jit': 'off',  # Disable JIT for security
                    'log_statement': 'all'  # Log all statements for audit
                }
            )
            logger.info("PostgreSQL connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise DatabaseError(f"PostgreSQL initialization failed: {e}")
    
    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self) -> AsyncContextManager[asyncpg.Connection]:
        """Get connection from pool with security validation"""
        if not self.pool:
            raise DatabaseError("PostgreSQL pool not initialized")
        
        async with self.pool.acquire() as conn:
            # Set connection-level security settings
            await conn.execute("SET statement_timeout = $1", self.config.query_timeout * 1000)
            await conn.execute("SET lock_timeout = $1", 30000)  # 30 second lock timeout
            yield conn
    
    async def execute_query(self, query: str, params: tuple = (), fetch: str = "none") -> Any:
        """Execute query with security validation"""
        # Validate query for injection patterns
        threats = self.validator.check_injection_patterns(query)
        if threats:
            raise DatabaseError(f"Query contains potential threats: {threats}")
        
        async with self.get_connection() as conn:
            try:
                if fetch == "all":
                    return await conn.fetch(query, *params)
                elif fetch == "one":
                    return await conn.fetchrow(query, *params)
                elif fetch == "val":
                    return await conn.fetchval(query, *params)
                else:
                    return await conn.execute(query, *params)
            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise DatabaseError(f"Query execution failed: {e}")
    
    async def store_osint_result(self, result: OSINTResult) -> str:
        """Store OSINT investigation result"""
        query = """
            INSERT INTO osint_results (
                result_id, target_type, target_value, tool_name, 
                data, confidence_score, risk_score, sources, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING result_id
        """
        
        params = (
            result.result_id,
            result.target.target_type,
            result.target.value,
            result.tool_name,
            json.dumps(result.data),
            result.confidence_score,
            result.risk_score,
            result.sources,
            result.created_at
        )
        
        return await self.execute_query(query, params, fetch="val")
    
    async def store_threat_intelligence(self, threat: ThreatIntelligence) -> bool:
        """Store threat intelligence data"""
        query = """
            INSERT INTO threat_intelligence (
                ioc_type, value, threat_types, confidence, severity,
                sources, first_seen, last_seen, metadata
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (ioc_type, value) 
            DO UPDATE SET 
                threat_types = EXCLUDED.threat_types,
                confidence = EXCLUDED.confidence,
                severity = EXCLUDED.severity,
                sources = EXCLUDED.sources,
                last_seen = EXCLUDED.last_seen,
                metadata = EXCLUDED.metadata
        """
        
        params = (
            threat.ioc_type,
            threat.value,
            threat.threat_types,
            threat.confidence,
            threat.severity.value,
            threat.sources,
            threat.first_seen,
            threat.last_seen,
            json.dumps(threat.metadata)
        )
        
        await self.execute_query(query, params)
        return True
    
    async def get_threat_intelligence(self, ioc_type: str, value: str) -> Optional[Dict[str, Any]]:
        """Retrieve threat intelligence for IOC"""
        query = """
            SELECT * FROM threat_intelligence 
            WHERE ioc_type = $1 AND value = $2
        """
        
        result = await self.execute_query(query, (ioc_type, value), fetch="one")
        return dict(result) if result else None


class Neo4jManager:
    """Neo4j graph database manager"""
    
    def __init__(self, config: DatabaseConfig, security_manager: SecurityManager):
        self.config = config
        self.security = security_manager
        self.driver = None
        self.validator = InputValidator()
    
    async def initialize(self):
        """Initialize Neo4j driver"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_timeout=self.config.connection_timeout
            )
            
            # Test connection
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            
            logger.info("Neo4j driver initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise DatabaseError(f"Neo4j initialization failed: {e}")
    
    async def close(self):
        """Close Neo4j driver"""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j driver closed")
    
    async def execute_cypher(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute Cypher query with security validation"""
        if parameters is None:
            parameters = {}
        
        # Basic validation for Cypher injection
        dangerous_patterns = [
            r"CALL\s+apoc\.",  # APOC procedures can be dangerous
            r"LOAD\s+CSV\s+FROM",  # File loading
            r"dbms\.",  # System procedures
        ]
        
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower):
                raise DatabaseError(f"Query contains potentially dangerous pattern: {pattern}")
        
        if not self.driver:
            raise DatabaseError("Neo4j driver not initialized")
        
        async with self.driver.session() as session:
            try:
                result = await session.run(query, parameters)
                return [record.data() async for record in result]
            except Exception as e:
                logger.error(f"Cypher query failed: {e}")
                raise DatabaseError(f"Cypher query failed: {e}")
    
    async def create_node(self, labels: List[str], properties: Dict[str, Any]) -> str:
        """Create graph node"""
        # Sanitize labels and properties
        safe_labels = [self.validator.sanitize_input(label, 50) for label in labels]
        safe_properties = {
            k: self.validator.sanitize_input(str(v), 1000) if isinstance(v, str) else v
            for k, v in properties.items()
        }
        
        labels_str = ":".join(safe_labels)
        query = f"CREATE (n:{labels_str} $props) RETURN elementId(n) as node_id"
        
        result = await self.execute_cypher(query, {"props": safe_properties})
        return result[0]["node_id"] if result else None
    
    async def create_relationship(self, start_node_id: str, end_node_id: str, 
                                relationship_type: str, properties: Dict[str, Any] = None) -> str:
        """Create relationship between nodes"""
        if properties is None:
            properties = {}
        
        # Sanitize relationship type and properties
        safe_rel_type = self.validator.sanitize_input(relationship_type, 50)
        safe_properties = {
            k: self.validator.sanitize_input(str(v), 1000) if isinstance(v, str) else v
            for k, v in properties.items()
        }
        
        query = f"""
            MATCH (a), (b) 
            WHERE elementId(a) = $start_id AND elementId(b) = $end_id
            CREATE (a)-[r:{safe_rel_type} $props]->(b)
            RETURN elementId(r) as rel_id
        """
        
        params = {
            "start_id": start_node_id,
            "end_id": end_node_id,
            "props": safe_properties
        }
        
        result = await self.execute_cypher(query, params)
        return result[0]["rel_id"] if result else None
    
    async def find_related_entities(self, entity_value: str, max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find entities related to a given entity"""
        query = """
            MATCH path = (start)-[*1..3]-(related)
            WHERE start.value = $entity_value
            RETURN DISTINCT related.value as value, 
                   labels(related) as labels,
                   length(path) as distance
            ORDER BY distance
            LIMIT 100
        """
        
        return await self.execute_cypher(query, {"entity_value": entity_value})


class RedisManager:
    """Redis manager for caching and session storage"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client: Optional[aioredis.Redis] = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.client = aioredis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                max_connections=self.config.redis_max_connections,
                socket_timeout=self.config.connection_timeout,
                decode_responses=True
            )
            
            # Test connection
            await self.client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise DatabaseError(f"Redis initialization failed: {e}")
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")
    
    async def cache_result(self, key: str, value: Any, expiry: int = 3600):
        """Cache result with expiration"""
        if not self.client:
            raise DatabaseError("Redis client not initialized")
        
        await self.client.setex(key, expiry, json.dumps(value, default=str))
    
    async def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached result"""
        if not self.client:
            return None
        
        cached = await self.client.get(key)
        return json.loads(cached) if cached else None
    
    async def invalidate_cache(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        if not self.client:
            return
        
        keys = await self.client.keys(pattern)
        if keys:
            await self.client.delete(*keys)


class ElasticsearchManager:
    """Elasticsearch manager for search and analytics"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client: Optional[AsyncElasticsearch] = None
    
    async def initialize(self):
        """Initialize Elasticsearch client"""
        try:
            self.client = AsyncElasticsearch(
                hosts=[f"http://{self.config.elasticsearch_host}:{self.config.elasticsearch_port}"],
                timeout=self.config.connection_timeout,
                max_retries=3,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.client.ping()
            logger.info("Elasticsearch connection established")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise DatabaseError(f"Elasticsearch initialization failed: {e}")
    
    async def close(self):
        """Close Elasticsearch client"""
        if self.client:
            await self.client.close()
            logger.info("Elasticsearch connection closed")
    
    async def index_document(self, index: str, doc_id: str, document: Dict[str, Any]):
        """Index document in Elasticsearch"""
        if not self.client:
            raise DatabaseError("Elasticsearch client not initialized")
        
        await self.client.index(index=index, id=doc_id, document=document)
    
    async def search_documents(self, index: str, query: Dict[str, Any], size: int = 100) -> List[Dict[str, Any]]:
        """Search documents in Elasticsearch"""
        if not self.client:
            raise DatabaseError("Elasticsearch client not initialized")
        
        response = await self.client.search(index=index, query=query, size=size)
        return [hit["_source"] for hit in response["hits"]["hits"]]


class DatabaseManager:
    """Main database manager coordinating all database connections"""
    
    def __init__(self, config: DatabaseConfig, security_manager: SecurityManager):
        self.config = config
        self.security = security_manager
        
        self.postgres = PostgreSQLManager(config, security_manager)
        self.neo4j = Neo4jManager(config, security_manager)
        self.redis = RedisManager(config)
        self.elasticsearch = ElasticsearchManager(config)
    
    async def initialize_all(self):
        """Initialize all database connections"""
        await asyncio.gather(
            self.postgres.initialize(),
            self.neo4j.initialize(),
            self.redis.initialize(),
            self.elasticsearch.initialize(),
            return_exceptions=True
        )
        logger.info("All database connections initialized")
    
    async def close_all(self):
        """Close all database connections"""
        await asyncio.gather(
            self.postgres.close(),
            self.neo4j.close(),
            self.redis.close(),
            self.elasticsearch.close(),
            return_exceptions=True
        )
        logger.info("All database connections closed")
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all database connections"""
        health = {}
        
        try:
            await self.postgres.execute_query("SELECT 1", fetch="val")
            health["postgresql"] = True
        except:
            health["postgresql"] = False
        
        try:
            await self.neo4j.execute_cypher("RETURN 1")
            health["neo4j"] = True
        except:
            health["neo4j"] = False
        
        try:
            await self.redis.client.ping()
            health["redis"] = True
        except:
            health["redis"] = False
        
        try:
            await self.elasticsearch.client.ping()
            health["elasticsearch"] = True
        except:
            health["elasticsearch"] = False
        
        return health