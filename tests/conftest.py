"""
Pytest configuration for BEV OSINT testing framework
"""

import pytest
import asyncio
import docker
import time
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path
import psycopg2
import redis
import neo4j
from prometheus_client.parser import text_string_to_metric_families
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test configuration
TEST_SERVICES = {
    "postgres": {"host": "localhost", "port": 5432, "timeout": 30},
    "redis": {"host": "localhost", "port": 6379, "timeout": 10},
    "neo4j": {"host": "localhost", "port": 7687, "timeout": 30},
    "qdrant": {"host": "localhost", "port": 6333, "timeout": 20},
    "weaviate": {"host": "localhost", "port": 8080, "timeout": 20},
    "prometheus": {"host": "localhost", "port": 9090, "timeout": 10},
    "grafana": {"host": "localhost", "port": 3000, "timeout": 10},
    "airflow": {"host": "localhost", "port": 8080, "timeout": 30},
    "kafka": {"host": "localhost", "port": 9092, "timeout": 20},
    "elasticsearch": {"host": "localhost", "port": 9200, "timeout": 20}
}

@pytest.fixture(scope="session")
def docker_client():
    """Docker client for container management"""
    return docker.from_env()

@pytest.fixture(scope="session")
def test_config():
    """Test configuration fixture"""
    return {
        "performance_targets": {
            "concurrent_requests": 1000,
            "max_latency_ms": 100,
            "cache_hit_rate": 0.80,
            "chaos_recovery_minutes": 5,
            "availability_target": 0.999
        },
        "services": TEST_SERVICES,
        "test_data_path": Path(__file__).parent / "data",
        "reports_path": Path(__file__).parent / "reports"
    }

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def service_health_check(test_config):
    """Verify all required services are running"""
    logger.info("Checking service health...")
    health_status = {}

    for service_name, config in test_config["services"].items():
        try:
            if service_name == "postgres":
                conn = psycopg2.connect(
                    host=config["host"],
                    port=config["port"],
                    user="researcher",
                    password="research_db_2024",
                    database="osint",
                    connect_timeout=config["timeout"]
                )
                conn.close()
                health_status[service_name] = "healthy"

            elif service_name == "redis":
                r = redis.Redis(
                    host=config["host"],
                    port=config["port"],
                    socket_connect_timeout=config["timeout"]
                )
                r.ping()
                health_status[service_name] = "healthy"

            elif service_name == "neo4j":
                driver = neo4j.GraphDatabase.driver(
                    f"bolt://{config['host']}:{config['port']}",
                    auth=("neo4j", "research_graph_2024")
                )
                with driver.session() as session:
                    result = session.run("RETURN 1")
                    result.single()
                driver.close()
                health_status[service_name] = "healthy"

            else:
                # HTTP-based services
                port_map = {
                    "qdrant": 6333,
                    "weaviate": 8080,
                    "prometheus": 9090,
                    "grafana": 3000,
                    "airflow": 8080,
                    "elasticsearch": 9200
                }

                if service_name in port_map:
                    url = f"http://{config['host']}:{port_map[service_name]}"
                    if service_name == "qdrant":
                        url += "/collections"
                    elif service_name == "weaviate":
                        url += "/v1/meta"
                    elif service_name == "prometheus":
                        url += "/api/v1/query?query=up"
                    elif service_name == "elasticsearch":
                        url += "/_cluster/health"

                    response = requests.get(url, timeout=config["timeout"])
                    if response.status_code in [200, 404]:  # 404 acceptable for empty endpoints
                        health_status[service_name] = "healthy"
                    else:
                        health_status[service_name] = f"unhealthy: {response.status_code}"
                else:
                    health_status[service_name] = "not_checked"

        except Exception as e:
            health_status[service_name] = f"error: {str(e)}"
            logger.warning(f"Service {service_name} health check failed: {e}")

    # Log health status
    logger.info("Service Health Status:")
    for service, status in health_status.items():
        logger.info(f"  {service}: {status}")

    return health_status

@pytest.fixture
def postgres_connection():
    """PostgreSQL connection fixture"""
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        user="researcher",
        password="research_db_2024",
        database="osint"
    )
    yield conn
    conn.close()

@pytest.fixture
def redis_connection():
    """Redis connection fixture"""
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    yield r
    r.close()

@pytest.fixture
def neo4j_session():
    """Neo4j session fixture"""
    driver = neo4j.GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "research_graph_2024")
    )
    session = driver.session()
    yield session
    session.close()
    driver.close()

@pytest.fixture
def test_data_generator():
    """Generate test data for various scenarios"""
    def generate_osint_payload(size="small"):
        payloads = {
            "small": {
                "query": "test_domain.com",
                "type": "domain_analysis",
                "options": {"deep_scan": False}
            },
            "medium": {
                "query": "192.168.1.0/24",
                "type": "network_scan",
                "options": {"port_scan": True, "service_detection": True}
            },
            "large": {
                "query": "comprehensive_analysis",
                "type": "multi_source",
                "options": {
                    "sources": ["whois", "dns", "certificates", "subdomains"],
                    "deep_scan": True,
                    "export_format": "json"
                }
            }
        }
        return payloads.get(size, payloads["small"])

    return generate_osint_payload

@pytest.fixture
def metrics_collector():
    """Collect metrics from Prometheus"""
    def collect_metrics():
        try:
            response = requests.get("http://localhost:9090/api/v1/query",
                                  params={"query": "up"}, timeout=10)
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            return {}

    return collect_metrics

@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during tests"""
    class PerformanceMonitor:
        def __init__(self):
            self.metrics = []
            self.start_time = None

        def start(self):
            self.start_time = time.time()

        def record(self, metric_name: str, value: float, tags: Dict[str, str] = None):
            self.metrics.append({
                "timestamp": time.time(),
                "metric": metric_name,
                "value": value,
                "tags": tags or {}
            })

        def stop(self):
            if self.start_time:
                duration = time.time() - self.start_time
                self.record("test_duration", duration)

        def get_report(self):
            return {
                "total_metrics": len(self.metrics),
                "metrics": self.metrics,
                "summary": self._generate_summary()
            }

        def _generate_summary(self):
            if not self.metrics:
                return {}

            latencies = [m["value"] for m in self.metrics if m["metric"] == "latency"]
            return {
                "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
                "max_latency": max(latencies) if latencies else 0,
                "min_latency": min(latencies) if latencies else 0
            }

    return PerformanceMonitor()

@pytest.fixture
def chaos_controller():
    """Control chaos engineering scenarios"""
    class ChaosController:
        def __init__(self, docker_client):
            self.docker = docker_client
            self.affected_containers = []

        def kill_random_service(self, service_type="worker"):
            """Kill a random service container"""
            try:
                containers = self.docker.containers.list(
                    filters={"label": f"service_type={service_type}"}
                )
                if containers:
                    container = containers[0]  # Take first for deterministic testing
                    container.kill()
                    self.affected_containers.append(container.id)
                    logger.info(f"Killed container: {container.name}")
                    return container.id
            except Exception as e:
                logger.error(f"Failed to kill container: {e}")
            return None

        def network_partition(self, duration=30):
            """Simulate network partition"""
            # Implementation would use Docker network manipulation
            logger.info(f"Simulating network partition for {duration}s")
            time.sleep(duration)

        def resource_exhaustion(self, container_name, cpu_limit="50%"):
            """Simulate resource exhaustion"""
            try:
                container = self.docker.containers.get(container_name)
                container.update(cpu_period=100000, cpu_quota=50000)  # 50% CPU
                logger.info(f"Applied resource limits to {container_name}")
            except Exception as e:
                logger.error(f"Failed to apply resource limits: {e}")

        def cleanup(self):
            """Restore affected containers"""
            for container_id in self.affected_containers:
                try:
                    container = self.docker.containers.get(container_id)
                    container.restart()
                    logger.info(f"Restored container: {container_id}")
                except Exception as e:
                    logger.error(f"Failed to restore container {container_id}: {e}")
            self.affected_containers.clear()

    return ChaosController

# Test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "integration: Integration tests for service connectivity"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "chaos: Chaos engineering and resilience tests"
    )
    config.addinivalue_line(
        "markers", "end_to_end: Complete workflow tests"
    )
    config.addinivalue_line(
        "markers", "vector_db: Vector database specific tests"
    )
    config.addinivalue_line(
        "markers", "cache: Cache performance tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take longer than 30 seconds"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location"""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        if "chaos" in str(item.fspath) or "resilience" in str(item.fspath):
            item.add_marker(pytest.mark.chaos)
        if "end_to_end" in str(item.fspath):
            item.add_marker(pytest.mark.end_to_end)
        if "vector" in str(item.fspath):
            item.add_marker(pytest.mark.vector_db)
        if "cache" in str(item.fspath):
            item.add_marker(pytest.mark.cache)