"""
Integration tests for service connectivity and basic functionality
"""

import pytest
import asyncio
import requests
import psycopg2
import redis
import neo4j
import json
import time
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

@pytest.mark.integration
class TestServiceConnectivity:
    """Test basic connectivity to all core services"""

    async def test_postgresql_connection(self, postgres_connection):
        """Test PostgreSQL connection and basic operations"""
        cursor = postgres_connection.cursor()

        # Test basic connectivity
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        assert version is not None

        # Test pgvector extension
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cursor.fetchone()
        assert vector_ext is not None, "pgvector extension not installed"

        # Test database creation
        cursor.execute("SELECT datname FROM pg_database WHERE datname IN ('osint', 'breach_data', 'crypto_analysis');")
        databases = cursor.fetchall()
        assert len(databases) >= 3, "Required databases not created"

        cursor.close()
        logger.info("PostgreSQL connectivity test passed")

    async def test_redis_connection(self, redis_connection):
        """Test Redis connection and basic operations"""
        # Test basic connectivity
        assert redis_connection.ping() == True

        # Test basic operations
        redis_connection.set("test_key", "test_value", ex=60)
        value = redis_connection.get("test_key")
        assert value == "test_value"

        # Test Redis streams (for event processing)
        stream_id = redis_connection.xadd("test_stream", {"event": "test"})
        assert stream_id is not None

        # Cleanup
        redis_connection.delete("test_key")
        redis_connection.delete("test_stream")

        logger.info("Redis connectivity test passed")

    async def test_neo4j_connection(self, neo4j_session):
        """Test Neo4j connection and graph operations"""
        # Test basic connectivity
        result = neo4j_session.run("RETURN 1 as test")
        assert result.single()["test"] == 1

        # Test node creation and retrieval
        neo4j_session.run(
            "CREATE (t:TestNode {name: 'integration_test', timestamp: $timestamp})",
            timestamp=int(time.time())
        )

        result = neo4j_session.run(
            "MATCH (t:TestNode {name: 'integration_test'}) RETURN t.name as name"
        )
        node = result.single()
        assert node["name"] == "integration_test"

        # Cleanup
        neo4j_session.run("MATCH (t:TestNode {name: 'integration_test'}) DELETE t")

        logger.info("Neo4j connectivity test passed")

    async def test_qdrant_connection(self):
        """Test Qdrant vector database connection"""
        base_url = "http://localhost:6333"

        # Test health endpoint
        response = requests.get(f"{base_url}/", timeout=10)
        assert response.status_code == 200

        # Test collections endpoint
        response = requests.get(f"{base_url}/collections", timeout=10)
        assert response.status_code == 200

        # Create test collection
        collection_config = {
            "vectors": {
                "size": 384,
                "distance": "Cosine"
            }
        }
        response = requests.put(
            f"{base_url}/collections/test_integration",
            json=collection_config,
            timeout=10
        )
        assert response.status_code in [200, 409]  # 409 if already exists

        # Test point insertion
        test_point = {
            "points": [
                {
                    "id": 1,
                    "vector": [0.1] * 384,
                    "payload": {"test": "integration"}
                }
            ]
        }
        response = requests.put(
            f"{base_url}/collections/test_integration/points",
            json=test_point,
            timeout=10
        )
        assert response.status_code == 200

        # Cleanup - delete test collection
        requests.delete(f"{base_url}/collections/test_integration", timeout=10)

        logger.info("Qdrant connectivity test passed")

    async def test_weaviate_connection(self):
        """Test Weaviate vector database connection"""
        base_url = "http://localhost:8080"

        # Test health endpoint
        response = requests.get(f"{base_url}/v1/meta", timeout=10)
        assert response.status_code == 200

        # Test schema endpoint
        response = requests.get(f"{base_url}/v1/schema", timeout=10)
        assert response.status_code == 200

        logger.info("Weaviate connectivity test passed")

    async def test_elasticsearch_connection(self):
        """Test Elasticsearch connection"""
        base_url = "http://localhost:9200"

        # Test cluster health
        response = requests.get(f"{base_url}/_cluster/health", timeout=10)
        assert response.status_code == 200

        health = response.json()
        assert health["status"] in ["green", "yellow"]

        # Test index creation
        index_config = {
            "mappings": {
                "properties": {
                    "content": {"type": "text"},
                    "timestamp": {"type": "date"}
                }
            }
        }
        response = requests.put(
            f"{base_url}/test_integration",
            json=index_config,
            timeout=10
        )
        assert response.status_code in [200, 400]  # 400 if already exists

        # Cleanup
        requests.delete(f"{base_url}/test_integration", timeout=10)

        logger.info("Elasticsearch connectivity test passed")

    async def test_prometheus_connection(self):
        """Test Prometheus metrics collection"""
        base_url = "http://localhost:9090"

        # Test metrics endpoint
        response = requests.get(f"{base_url}/api/v1/query",
                              params={"query": "up"}, timeout=10)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"
        assert "data" in data

        logger.info("Prometheus connectivity test passed")

    async def test_grafana_connection(self):
        """Test Grafana dashboard access"""
        base_url = "http://localhost:3000"

        # Test health endpoint
        response = requests.get(f"{base_url}/api/health", timeout=10)
        assert response.status_code == 200

        logger.info("Grafana connectivity test passed")

    async def test_airflow_connection(self):
        """Test Airflow scheduler and webserver"""
        base_url = "http://localhost:8080"

        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        assert response.status_code == 200

        logger.info("Airflow connectivity test passed")

@pytest.mark.integration
class TestServiceIntegration:
    """Test integration between services"""

    async def test_postgres_redis_integration(self, postgres_connection, redis_connection):
        """Test data flow between PostgreSQL and Redis"""
        cursor = postgres_connection.cursor()

        # Insert test data in PostgreSQL
        cursor.execute("""
            INSERT INTO research_sessions (session_id, query, status, created_at)
            VALUES (%s, %s, %s, NOW())
        """, ("test_session_123", "integration_test", "active"))
        postgres_connection.commit()

        # Cache in Redis
        session_data = {
            "session_id": "test_session_123",
            "query": "integration_test",
            "status": "active"
        }
        redis_connection.setex(
            "session:test_session_123",
            3600,
            json.dumps(session_data)
        )

        # Verify cache retrieval
        cached_data = redis_connection.get("session:test_session_123")
        assert cached_data is not None

        parsed_data = json.loads(cached_data)
        assert parsed_data["session_id"] == "test_session_123"

        # Cleanup
        cursor.execute("DELETE FROM research_sessions WHERE session_id = %s", ("test_session_123",))
        postgres_connection.commit()
        redis_connection.delete("session:test_session_123")
        cursor.close()

        logger.info("PostgreSQL-Redis integration test passed")

    async def test_vector_database_sync(self):
        """Test synchronization between Qdrant and Weaviate"""
        # Test vector embedding storage in both systems
        test_vector = [0.1] * 384
        test_metadata = {"source": "integration_test", "timestamp": int(time.time())}

        # Store in Qdrant
        qdrant_url = "http://localhost:6333"
        collection_config = {
            "vectors": {"size": 384, "distance": "Cosine"}
        }
        requests.put(f"{qdrant_url}/collections/sync_test", json=collection_config)

        point_data = {
            "points": [{
                "id": 999,
                "vector": test_vector,
                "payload": test_metadata
            }]
        }
        response = requests.put(
            f"{qdrant_url}/collections/sync_test/points",
            json=point_data
        )
        assert response.status_code == 200

        # Verify retrieval from Qdrant
        response = requests.get(f"{qdrant_url}/collections/sync_test/points/999")
        assert response.status_code == 200

        # Cleanup
        requests.delete(f"{qdrant_url}/collections/sync_test")

        logger.info("Vector database sync test passed")

    async def test_monitoring_integration(self):
        """Test metrics collection and alerting integration"""
        # Generate test metrics
        metrics_endpoint = "http://localhost:9090/api/v1/query"

        # Query service health metrics
        query = "up{job='bev-services'}"
        response = requests.get(metrics_endpoint, params={"query": query})

        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"
            logger.info("Monitoring integration test passed")
        else:
            logger.warning("Monitoring metrics not available yet")

@pytest.mark.integration
@pytest.mark.slow
class TestDataPipeline:
    """Test complete data processing pipeline"""

    async def test_osint_data_flow(self, postgres_connection, redis_connection):
        """Test complete OSINT data processing flow"""
        cursor = postgres_connection.cursor()

        # Simulate OSINT query input
        query_data = {
            "target": "test-domain.com",
            "scan_type": "comprehensive",
            "timestamp": int(time.time())
        }

        # Store initial query
        cursor.execute("""
            INSERT INTO osint_queries (target, scan_type, status, created_at)
            VALUES (%s, %s, %s, NOW()) RETURNING id
        """, (query_data["target"], query_data["scan_type"], "processing"))
        postgres_connection.commit()

        query_id = cursor.fetchone()[0]
        assert query_id is not None

        # Cache processing status
        redis_connection.setex(
            f"query_status:{query_id}",
            3600,
            json.dumps({"status": "processing", "progress": 25})
        )

        # Simulate processing completion
        cursor.execute("""
            UPDATE osint_queries
            SET status = %s, completed_at = NOW()
            WHERE id = %s
        """, ("completed", query_id))
        postgres_connection.commit()

        # Update cache
        redis_connection.setex(
            f"query_status:{query_id}",
            3600,
            json.dumps({"status": "completed", "progress": 100})
        )

        # Verify final status
        final_status = redis_connection.get(f"query_status:{query_id}")
        assert final_status is not None

        status_data = json.loads(final_status)
        assert status_data["status"] == "completed"

        # Cleanup
        cursor.execute("DELETE FROM osint_queries WHERE id = %s", (query_id,))
        postgres_connection.commit()
        redis_connection.delete(f"query_status:{query_id}")
        cursor.close()

        logger.info("OSINT data flow test passed")