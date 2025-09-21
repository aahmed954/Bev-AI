"""
Vector database integration tests for Qdrant and Weaviate
"""

import pytest
import asyncio
import requests
import numpy as np
import json
import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

@pytest.mark.vector_db
class TestQdrantOperations:
    """Comprehensive Qdrant vector database tests"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        self.base_url = "http://localhost:6333"
        self.test_collection = "test_qdrant_integration"

        # Cleanup before test
        requests.delete(f"{self.base_url}/collections/{self.test_collection}")
        yield
        # Cleanup after test
        requests.delete(f"{self.base_url}/collections/{self.test_collection}")

    async def test_collection_management(self):
        """Test collection creation, configuration, and deletion"""
        # Create collection with specific configuration
        collection_config = {
            "vectors": {
                "size": 768,
                "distance": "Cosine"
            },
            "optimizers_config": {
                "default_segment_number": 2
            },
            "replication_factor": 1,
            "write_consistency_factor": 1
        }

        response = requests.put(
            f"{self.base_url}/collections/{self.test_collection}",
            json=collection_config,
            timeout=30
        )
        assert response.status_code == 200

        # Verify collection info
        response = requests.get(f"{self.base_url}/collections/{self.test_collection}")
        assert response.status_code == 200

        collection_info = response.json()["result"]
        assert collection_info["config"]["params"]["vectors"]["size"] == 768
        assert collection_info["config"]["params"]["vectors"]["distance"] == "Cosine"

        logger.info("Qdrant collection management test passed")

    async def test_point_operations(self):
        """Test point insertion, retrieval, and deletion"""
        # Create collection first
        collection_config = {
            "vectors": {"size": 384, "distance": "Cosine"}
        }
        requests.put(f"{self.base_url}/collections/{self.test_collection}", json=collection_config)

        # Generate test vectors
        test_points = []
        for i in range(10):
            vector = np.random.random(384).tolist()
            test_points.append({
                "id": i,
                "vector": vector,
                "payload": {
                    "text": f"test_document_{i}",
                    "category": "integration_test",
                    "timestamp": int(time.time())
                }
            })

        # Batch insert points
        response = requests.put(
            f"{self.base_url}/collections/{self.test_collection}/points",
            json={"points": test_points},
            timeout=30
        )
        assert response.status_code == 200

        # Wait for indexing
        await asyncio.sleep(2)

        # Retrieve specific point
        response = requests.get(
            f"{self.base_url}/collections/{self.test_collection}/points/5"
        )
        assert response.status_code == 200

        point_data = response.json()["result"]
        assert point_data["id"] == 5
        assert point_data["payload"]["text"] == "test_document_5"

        # Search for similar vectors
        search_query = {
            "vector": test_points[0]["vector"],
            "limit": 5,
            "with_payload": True
        }
        response = requests.post(
            f"{self.base_url}/collections/{self.test_collection}/points/search",
            json=search_query,
            timeout=30
        )
        assert response.status_code == 200

        search_results = response.json()["result"]
        assert len(search_results) > 0
        assert search_results[0]["id"] == 0  # Should find exact match first

        logger.info("Qdrant point operations test passed")

    async def test_filtered_search(self):
        """Test search with payload filtering"""
        # Create collection and insert test data
        collection_config = {"vectors": {"size": 384, "distance": "Cosine"}}
        requests.put(f"{self.base_url}/collections/{self.test_collection}", json=collection_config)

        # Insert points with different categories
        test_points = []
        categories = ["documents", "images", "videos", "audio"]

        for i in range(20):
            vector = np.random.random(384).tolist()
            test_points.append({
                "id": i,
                "vector": vector,
                "payload": {
                    "category": categories[i % 4],
                    "confidence": np.random.uniform(0.5, 1.0),
                    "timestamp": int(time.time()) - (i * 3600)  # Different timestamps
                }
            })

        requests.put(
            f"{self.base_url}/collections/{self.test_collection}/points",
            json={"points": test_points}
        )

        await asyncio.sleep(2)

        # Search with category filter
        search_query = {
            "vector": test_points[0]["vector"],
            "limit": 10,
            "filter": {
                "must": [
                    {"key": "category", "match": {"value": "documents"}}
                ]
            },
            "with_payload": True
        }

        response = requests.post(
            f"{self.base_url}/collections/{self.test_collection}/points/search",
            json=search_query
        )
        assert response.status_code == 200

        results = response.json()["result"]
        assert len(results) > 0

        # Verify all results have correct category
        for result in results:
            assert result["payload"]["category"] == "documents"

        # Search with confidence range filter
        confidence_search = {
            "vector": test_points[5]["vector"],
            "limit": 10,
            "filter": {
                "must": [
                    {
                        "key": "confidence",
                        "range": {"gte": 0.8}
                    }
                ]
            },
            "with_payload": True
        }

        response = requests.post(
            f"{self.base_url}/collections/{self.test_collection}/points/search",
            json=confidence_search
        )
        assert response.status_code == 200

        results = response.json()["result"]
        for result in results:
            assert result["payload"]["confidence"] >= 0.8

        logger.info("Qdrant filtered search test passed")

    async def test_performance_metrics(self):
        """Test vector database performance metrics"""
        # Create collection optimized for performance
        collection_config = {
            "vectors": {
                "size": 768,
                "distance": "Dot"
            },
            "optimizers_config": {
                "default_segment_number": 4,
                "max_segment_size": 50000
            },
            "hnsw_config": {
                "m": 16,
                "ef_construct": 100,
                "full_scan_threshold": 10000
            }
        }

        requests.put(f"{self.base_url}/collections/{self.test_collection}", json=collection_config)

        # Insert larger batch for performance testing
        batch_size = 1000
        test_points = []

        for i in range(batch_size):
            vector = np.random.random(768).tolist()
            test_points.append({
                "id": i,
                "vector": vector,
                "payload": {"batch_id": i // 100}
            })

        # Measure insertion time
        start_time = time.time()
        response = requests.put(
            f"{self.base_url}/collections/{self.test_collection}/points",
            json={"points": test_points},
            timeout=60
        )
        insertion_time = time.time() - start_time

        assert response.status_code == 200
        assert insertion_time < 30  # Should complete within 30 seconds

        # Wait for indexing
        await asyncio.sleep(5)

        # Measure search time
        search_vector = np.random.random(768).tolist()
        start_time = time.time()

        for _ in range(10):  # Multiple searches for average
            response = requests.post(
                f"{self.base_url}/collections/{self.test_collection}/points/search",
                json={
                    "vector": search_vector,
                    "limit": 20,
                    "with_payload": False
                }
            )
            assert response.status_code == 200

        avg_search_time = (time.time() - start_time) / 10
        assert avg_search_time < 0.1  # Each search should be under 100ms

        logger.info(f"Qdrant performance: insertion={insertion_time:.2f}s, search={avg_search_time:.3f}s")

@pytest.mark.vector_db
class TestWeaviateOperations:
    """Comprehensive Weaviate vector database tests"""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test"""
        self.base_url = "http://localhost:8080"
        self.test_class = "TestIntegration"

        # Cleanup before test
        requests.delete(f"{self.base_url}/v1/schema/{self.test_class}")
        yield
        # Cleanup after test
        requests.delete(f"{self.base_url}/v1/schema/{self.test_class}")

    async def test_schema_management(self):
        """Test schema creation and management"""
        # Create test class schema
        schema = {
            "class": self.test_class,
            "vectorizer": "none",  # We'll provide vectors manually
            "properties": [
                {
                    "name": "content",
                    "dataType": ["text"]
                },
                {
                    "name": "category",
                    "dataType": ["string"]
                },
                {
                    "name": "confidence",
                    "dataType": ["number"]
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"]
                }
            ]
        }

        response = requests.post(
            f"{self.base_url}/v1/schema",
            json=schema,
            timeout=30
        )
        assert response.status_code == 200

        # Verify schema creation
        response = requests.get(f"{self.base_url}/v1/schema/{self.test_class}")
        assert response.status_code == 200

        class_schema = response.json()
        assert class_schema["class"] == self.test_class
        assert len(class_schema["properties"]) == 4

        logger.info("Weaviate schema management test passed")

    async def test_object_operations(self):
        """Test object insertion, retrieval, and deletion"""
        # Create schema first
        schema = {
            "class": self.test_class,
            "vectorizer": "none",
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "category", "dataType": ["string"]}
            ]
        }
        requests.post(f"{self.base_url}/v1/schema", json=schema)

        # Insert test objects
        test_objects = []
        for i in range(5):
            vector = np.random.random(384).tolist()
            obj = {
                "class": self.test_class,
                "properties": {
                    "content": f"Test document {i}",
                    "category": "integration_test"
                },
                "vector": vector
            }

            response = requests.post(
                f"{self.base_url}/v1/objects",
                json=obj,
                timeout=30
            )
            assert response.status_code == 200

            obj_id = response.json()["id"]
            test_objects.append(obj_id)

        # Retrieve object by ID
        response = requests.get(
            f"{self.base_url}/v1/objects/{test_objects[0]}"
        )
        assert response.status_code == 200

        obj_data = response.json()
        assert obj_data["properties"]["content"] == "Test document 0"

        # Search for objects
        query = {
            "query": f"""
            {{
                Get {{
                    {self.test_class}(limit: 3) {{
                        content
                        category
                        _additional {{
                            id
                            certainty
                        }}
                    }}
                }}
            }}
            """
        }

        response = requests.post(
            f"{self.base_url}/v1/graphql",
            json=query,
            timeout=30
        )
        assert response.status_code == 200

        results = response.json()["data"]["Get"][self.test_class]
        assert len(results) > 0

        logger.info("Weaviate object operations test passed")

    async def test_semantic_search(self):
        """Test semantic search capabilities"""
        # Create schema with text vectorization
        schema = {
            "class": self.test_class,
            "vectorizer": "none",
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "title", "dataType": ["string"]},
                {"name": "tags", "dataType": ["string[]"]}
            ]
        }
        requests.post(f"{self.base_url}/v1/schema", json=schema)

        # Insert semantically meaningful test data
        test_documents = [
            {
                "content": "Machine learning algorithms for data analysis",
                "title": "ML Guide",
                "tags": ["machine-learning", "data-science", "algorithms"]
            },
            {
                "content": "Cybersecurity threats and defense mechanisms",
                "title": "Security Handbook",
                "tags": ["cybersecurity", "threats", "defense"]
            },
            {
                "content": "Web development with modern frameworks",
                "title": "Web Dev Tutorial",
                "tags": ["web-development", "javascript", "frameworks"]
            }
        ]

        for doc in test_documents:
            # Generate content-based vector (simplified approach)
            vector = np.random.random(384).tolist()

            obj = {
                "class": self.test_class,
                "properties": doc,
                "vector": vector
            }

            response = requests.post(f"{self.base_url}/v1/objects", json=obj)
            assert response.status_code == 200

        # Search with semantic query
        search_query = {
            "query": f"""
            {{
                Get {{
                    {self.test_class}(
                        where: {{
                            path: ["tags"]
                            operator: ContainsAny
                            valueStringArray: ["machine-learning", "data-science"]
                        }}
                        limit: 5
                    ) {{
                        content
                        title
                        tags
                        _additional {{
                            certainty
                        }}
                    }}
                }}
            }}
            """
        }

        response = requests.post(f"{self.base_url}/v1/graphql", json=search_query)
        assert response.status_code == 200

        results = response.json()["data"]["Get"][self.test_class]
        assert len(results) > 0

        # Verify relevant results
        found_ml = any("machine-learning" in result.get("tags", []) for result in results)
        assert found_ml

        logger.info("Weaviate semantic search test passed")

@pytest.mark.vector_db
class TestVectorDatabaseSync:
    """Test synchronization between vector databases"""

    async def test_cross_database_replication(self):
        """Test data replication between Qdrant and Weaviate"""
        # Test data
        test_vectors = [np.random.random(384).tolist() for _ in range(5)]
        test_metadata = [
            {"id": i, "content": f"Document {i}", "category": "sync_test"}
            for i in range(5)
        ]

        # Store in Qdrant
        qdrant_url = "http://localhost:6333"
        collection_name = "sync_test_collection"

        collection_config = {"vectors": {"size": 384, "distance": "Cosine"}}
        requests.put(f"{qdrant_url}/collections/{collection_name}", json=collection_config)

        qdrant_points = [
            {
                "id": i,
                "vector": test_vectors[i],
                "payload": test_metadata[i]
            }
            for i in range(5)
        ]

        response = requests.put(
            f"{qdrant_url}/collections/{collection_name}/points",
            json={"points": qdrant_points}
        )
        assert response.status_code == 200

        # Store in Weaviate
        weaviate_url = "http://localhost:8080"
        class_name = "SyncTest"

        schema = {
            "class": class_name,
            "vectorizer": "none",
            "properties": [
                {"name": "content", "dataType": ["text"]},
                {"name": "category", "dataType": ["string"]}
            ]
        }
        requests.post(f"{weaviate_url}/v1/schema", json=schema)

        for i in range(5):
            obj = {
                "class": class_name,
                "properties": {
                    "content": test_metadata[i]["content"],
                    "category": test_metadata[i]["category"]
                },
                "vector": test_vectors[i]
            }
            response = requests.post(f"{weaviate_url}/v1/objects", json=obj)
            assert response.status_code == 200

        # Verify data integrity in both systems
        await asyncio.sleep(2)

        # Check Qdrant
        response = requests.get(f"{qdrant_url}/collections/{collection_name}/points/1")
        assert response.status_code == 200
        qdrant_point = response.json()["result"]
        assert qdrant_point["payload"]["content"] == "Document 1"

        # Check Weaviate
        query = {
            "query": f"""
            {{
                Get {{
                    {class_name}(limit: 5) {{
                        content
                        category
                    }}
                }}
            }}
            """
        }
        response = requests.post(f"{weaviate_url}/v1/graphql", json=query)
        assert response.status_code == 200

        weaviate_results = response.json()["data"]["Get"][class_name]
        assert len(weaviate_results) == 5

        # Cleanup
        requests.delete(f"{qdrant_url}/collections/{collection_name}")
        requests.delete(f"{weaviate_url}/v1/schema/{class_name}")

        logger.info("Vector database synchronization test passed")

    async def test_performance_comparison(self):
        """Compare performance between Qdrant and Weaviate"""
        vector_size = 768
        num_vectors = 500

        # Generate test data
        test_vectors = [np.random.random(vector_size).tolist() for _ in range(num_vectors)]

        # Test Qdrant performance
        qdrant_url = "http://localhost:6333"
        qdrant_collection = "perf_test_qdrant"

        collection_config = {"vectors": {"size": vector_size, "distance": "Cosine"}}
        requests.put(f"{qdrant_url}/collections/{qdrant_collection}", json=collection_config)

        qdrant_points = [
            {"id": i, "vector": test_vectors[i], "payload": {"index": i}}
            for i in range(num_vectors)
        ]

        start_time = time.time()
        response = requests.put(
            f"{qdrant_url}/collections/{qdrant_collection}/points",
            json={"points": qdrant_points},
            timeout=60
        )
        qdrant_insert_time = time.time() - start_time
        assert response.status_code == 200

        # Test Weaviate performance
        weaviate_url = "http://localhost:8080"
        weaviate_class = "PerfTestWeaviate"

        schema = {
            "class": weaviate_class,
            "vectorizer": "none",
            "properties": [{"name": "index", "dataType": ["int"]}]
        }
        requests.post(f"{weaviate_url}/v1/schema", json=schema)

        start_time = time.time()
        for i in range(num_vectors):
            obj = {
                "class": weaviate_class,
                "properties": {"index": i},
                "vector": test_vectors[i]
            }
            requests.post(f"{weaviate_url}/v1/objects", json=obj)
        weaviate_insert_time = time.time() - start_time

        # Performance comparison
        logger.info(f"Performance comparison:")
        logger.info(f"  Qdrant batch insert: {qdrant_insert_time:.2f}s")
        logger.info(f"  Weaviate individual inserts: {weaviate_insert_time:.2f}s")

        # Both should complete within reasonable time
        assert qdrant_insert_time < 30
        assert weaviate_insert_time < 60

        # Cleanup
        requests.delete(f"{qdrant_url}/collections/{qdrant_collection}")
        requests.delete(f"{weaviate_url}/v1/schema/{weaviate_class}")

        logger.info("Vector database performance comparison completed")