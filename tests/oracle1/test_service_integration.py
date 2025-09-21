#!/usr/bin/env python3
"""
Service Integration and Communication Testing Suite for ORACLE1

This module provides comprehensive testing for:
- Service startup and health checks
- Inter-service communication
- Network connectivity and routing
- Service dependency validation
- API endpoint functionality
- Message queue communication
- Database connectivity
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
import docker
import redis
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceIntegrationTester:
    """Comprehensive service integration testing for ORACLE1."""

    def __init__(self, project_root: str = "/home/starlord/Projects/Bev"):
        self.project_root = Path(project_root)
        self.docker_client = docker.from_env()
        self.compose_file = self.project_root / "docker-compose-oracle1-unified.yml"
        self.results = {
            "timestamp": time.time(),
            "tests": [],
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "warnings": 0
            }
        }

        # Service configuration
        self.core_services = [
            "redis-arm", "nginx", "n8n"
        ]

        self.monitoring_services = [
            "prometheus", "grafana", "alertmanager", "vault"
        ]

        self.storage_services = [
            "minio1", "minio2", "minio3", "influxdb-primary", "influxdb-replica"
        ]

        self.processing_services = [
            "litellm-gateway-1", "litellm-gateway-2", "litellm-gateway-3",
            "request-multiplexer", "genetic-optimizer", "knowledge-synthesis",
            "toolmaster-orchestrator"
        ]

        # Service endpoints with expected status codes
        self.service_endpoints = {
            "redis-arm": {"port": 6379, "protocol": "tcp", "health_check": "redis_ping"},
            "nginx": {"port": 80, "protocol": "http", "path": "/", "expected_codes": [200, 404, 502]},
            "n8n": {"port": 5678, "protocol": "http", "path": "/", "expected_codes": [200, 401]},
            "prometheus": {"port": 9090, "protocol": "http", "path": "/-/healthy", "expected_codes": [200]},
            "grafana": {"port": 3000, "protocol": "http", "path": "/api/health", "expected_codes": [200]},
            "alertmanager": {"port": 9093, "protocol": "http", "path": "/-/healthy", "expected_codes": [200]},
            "vault": {"port": 8200, "protocol": "http", "path": "/v1/sys/health", "expected_codes": [200, 429, 501]},
            "minio1": {"port": 9001, "protocol": "http", "path": "/minio/health/live", "expected_codes": [200]},
            "minio2": {"port": 9002, "protocol": "http", "path": "/minio/health/live", "expected_codes": [200]},
            "minio3": {"port": 9003, "protocol": "http", "path": "/minio/health/live", "expected_codes": [200]},
            "influxdb-primary": {"port": 8086, "protocol": "http", "path": "/ping", "expected_codes": [200, 204]},
            "influxdb-replica": {"port": 8087, "protocol": "http", "path": "/ping", "expected_codes": [200, 204]},
            "litellm-gateway-1": {"port": 5000, "protocol": "http", "path": "/health", "expected_codes": [200, 404]},
            "litellm-gateway-2": {"port": 5001, "protocol": "http", "path": "/health", "expected_codes": [200, 404]},
            "litellm-gateway-3": {"port": 5002, "protocol": "http", "path": "/health", "expected_codes": [200, 404]},
            "request-multiplexer": {"port": 8080, "protocol": "http", "path": "/health", "expected_codes": [200, 404]}
        }

        # Network connectivity tests
        self.network_tests = {
            "thanos_connectivity": {
                "host": "100.122.12.54",
                "ports": [5432, 7687, 6379, 9092, 8000],
                "description": "THANOS server connectivity"
            },
            "external_dns": {
                "hosts": ["8.8.8.8", "1.1.1.1"],
                "description": "External DNS connectivity"
            }
        }

    def log_test_result(self, test_name: str, status: str, details: str = "",
                       execution_time: float = 0.0, metadata: Dict = None):
        """Log a test result."""
        result = {
            "test_name": test_name,
            "status": status,  # "PASS", "FAIL", "WARN"
            "details": details,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        self.results["tests"].append(result)
        self.results["summary"]["total_tests"] += 1

        if status == "PASS":
            self.results["summary"]["passed_tests"] += 1
            logger.info(f"[PASS] {test_name}: {details}")
        elif status == "FAIL":
            self.results["summary"]["failed_tests"] += 1
            logger.error(f"[FAIL] {test_name}: {details}")
        elif status == "WARN":
            self.results["summary"]["warnings"] += 1
            logger.warning(f"[WARN] {test_name}: {details}")

    async def test_docker_compose_configuration(self) -> bool:
        """Test Docker Compose configuration validity."""
        logger.info("Testing Docker Compose configuration...")

        start_time = time.time()

        try:
            # Validate compose file syntax
            result = subprocess.run([
                "docker", "compose", "-f", str(self.compose_file), "config"
            ], capture_output=True, text=True, cwd=str(self.project_root))

            execution_time = time.time() - start_time

            if result.returncode == 0:
                # Parse the validated config to check service count
                try:
                    config_data = yaml.safe_load(result.stdout)
                    service_count = len(config_data.get('services', {}))
                    network_count = len(config_data.get('networks', {}))
                    volume_count = len(config_data.get('volumes', {}))

                    self.log_test_result(
                        "docker_compose_configuration",
                        "PASS",
                        f"Compose config valid: {service_count} services, {network_count} networks, {volume_count} volumes",
                        execution_time,
                        {
                            "service_count": service_count,
                            "network_count": network_count,
                            "volume_count": volume_count
                        }
                    )
                    return True
                except yaml.YAMLError as e:
                    self.log_test_result(
                        "docker_compose_configuration",
                        "FAIL",
                        f"Invalid YAML in compose output: {str(e)}",
                        execution_time
                    )
                    return False
            else:
                self.log_test_result(
                    "docker_compose_configuration",
                    "FAIL",
                    f"Compose config validation failed: {result.stderr.strip()}",
                    execution_time
                )
                return False

        except Exception as e:
            self.log_test_result(
                "docker_compose_configuration",
                "FAIL",
                f"Compose config test failed: {str(e)}",
                time.time() - start_time
            )
            return False

    async def test_service_startup_sequence(self) -> bool:
        """Test service startup in proper dependency order."""
        logger.info("Testing service startup sequence...")

        start_time = time.time()

        try:
            # Start core services first
            for service_group, services in [
                ("core", self.core_services),
                ("monitoring", self.monitoring_services),
                ("storage", self.storage_services)
            ]:
                logger.info(f"Starting {service_group} services: {', '.join(services)}")

                for service in services:
                    service_start_time = time.time()

                    try:
                        # Start individual service
                        result = subprocess.run([
                            "docker", "compose", "-f", str(self.compose_file),
                            "up", "-d", service
                        ], capture_output=True, text=True, cwd=str(self.project_root), timeout=120)

                        service_time = time.time() - service_start_time

                        if result.returncode == 0:
                            # Wait for service to stabilize
                            await asyncio.sleep(5)

                            # Check if service is running
                            ps_result = subprocess.run([
                                "docker", "compose", "-f", str(self.compose_file),
                                "ps", service
                            ], capture_output=True, text=True, cwd=str(self.project_root))

                            if ps_result.returncode == 0 and "Up" in ps_result.stdout:
                                self.log_test_result(
                                    f"service_startup_{service}",
                                    "PASS",
                                    f"Service started successfully in {service_time:.1f}s",
                                    service_time,
                                    {"service_group": service_group}
                                )
                            else:
                                self.log_test_result(
                                    f"service_startup_{service}",
                                    "FAIL",
                                    f"Service not running after startup",
                                    service_time
                                )
                        else:
                            self.log_test_result(
                                f"service_startup_{service}",
                                "FAIL",
                                f"Service startup failed: {result.stderr.strip()[:200]}",
                                service_time
                            )

                    except subprocess.TimeoutExpired:
                        self.log_test_result(
                            f"service_startup_{service}",
                            "FAIL",
                            "Service startup timeout (>120s)",
                            120.0
                        )
                    except Exception as e:
                        self.log_test_result(
                            f"service_startup_{service}",
                            "FAIL",
                            f"Service startup error: {str(e)}",
                            time.time() - service_start_time
                        )

                # Wait between service groups
                await asyncio.sleep(10)

            execution_time = time.time() - start_time
            return True

        except Exception as e:
            self.log_test_result(
                "service_startup_sequence",
                "FAIL",
                f"Startup sequence test failed: {str(e)}",
                time.time() - start_time
            )
            return False

    async def test_service_health_checks(self) -> bool:
        """Test individual service health endpoints."""
        logger.info("Testing service health checks...")

        start_time = time.time()
        all_healthy = True

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for service_name, endpoint_config in self.service_endpoints.items():
                test_start = time.time()

                try:
                    if endpoint_config.get("health_check") == "redis_ping":
                        # Special case for Redis
                        try:
                            redis_client = redis.Redis(
                                host='localhost',
                                port=endpoint_config["port"],
                                socket_timeout=5,
                                decode_responses=True
                            )
                            redis_client.ping()

                            self.log_test_result(
                                f"health_check_{service_name}",
                                "PASS",
                                "Redis ping successful",
                                time.time() - test_start
                            )
                        except Exception as e:
                            self.log_test_result(
                                f"health_check_{service_name}",
                                "FAIL",
                                f"Redis ping failed: {str(e)}",
                                time.time() - test_start
                            )
                            all_healthy = False

                    elif endpoint_config["protocol"] == "http":
                        # HTTP health check
                        port = endpoint_config["port"]
                        path = endpoint_config.get("path", "/")
                        expected_codes = endpoint_config.get("expected_codes", [200])

                        url = f"http://localhost:{port}{path}"

                        try:
                            async with session.get(url) as response:
                                test_time = time.time() - test_start

                                if response.status in expected_codes:
                                    self.log_test_result(
                                        f"health_check_{service_name}",
                                        "PASS",
                                        f"HTTP {response.status} response in {test_time:.3f}s",
                                        test_time,
                                        {"url": url, "status_code": response.status}
                                    )
                                else:
                                    self.log_test_result(
                                        f"health_check_{service_name}",
                                        "FAIL",
                                        f"HTTP {response.status} (expected {expected_codes})",
                                        test_time,
                                        {"url": url, "status_code": response.status}
                                    )
                                    all_healthy = False

                        except aiohttp.ClientError as e:
                            self.log_test_result(
                                f"health_check_{service_name}",
                                "FAIL",
                                f"HTTP request failed: {str(e)}",
                                time.time() - test_start,
                                {"url": url}
                            )
                            all_healthy = False

                    elif endpoint_config["protocol"] == "tcp":
                        # TCP connectivity check
                        port = endpoint_config["port"]

                        try:
                            # Simple TCP connection test
                            reader, writer = await asyncio.wait_for(
                                asyncio.open_connection('localhost', port),
                                timeout=5
                            )
                            writer.close()
                            await writer.wait_closed()

                            self.log_test_result(
                                f"health_check_{service_name}",
                                "PASS",
                                f"TCP connection successful on port {port}",
                                time.time() - test_start,
                                {"port": port}
                            )

                        except Exception as e:
                            self.log_test_result(
                                f"health_check_{service_name}",
                                "FAIL",
                                f"TCP connection failed: {str(e)}",
                                time.time() - test_start,
                                {"port": port}
                            )
                            all_healthy = False

                except Exception as e:
                    self.log_test_result(
                        f"health_check_{service_name}",
                        "FAIL",
                        f"Health check error: {str(e)}",
                        time.time() - test_start
                    )
                    all_healthy = False

        execution_time = time.time() - start_time
        return all_healthy

    async def test_inter_service_communication(self) -> bool:
        """Test communication between services."""
        logger.info("Testing inter-service communication...")

        start_time = time.time()
        communication_tests = []

        # Test Redis connectivity from within containers
        try:
            result = subprocess.run([
                "docker", "exec", "bev_n8n",
                "sh", "-c", "echo 'PING' | nc redis-arm 6379 | grep PONG"
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and "PONG" in result.stdout:
                communication_tests.append(("n8n_to_redis", "PASS", "N8N can reach Redis"))
            else:
                communication_tests.append(("n8n_to_redis", "FAIL", "N8N cannot reach Redis"))

        except Exception as e:
            communication_tests.append(("n8n_to_redis", "FAIL", f"Test failed: {str(e)}"))

        # Test MinIO cluster communication
        try:
            result = subprocess.run([
                "docker", "exec", "bev_minio1",
                "mc", "admin", "info", "local"
            ], capture_output=True, text=True, timeout=15)

            if result.returncode == 0:
                communication_tests.append(("minio_cluster", "PASS", "MinIO cluster communication working"))
            else:
                communication_tests.append(("minio_cluster", "WARN", "MinIO cluster info unavailable"))

        except Exception as e:
            communication_tests.append(("minio_cluster", "FAIL", f"MinIO test failed: {str(e)}"))

        # Test Prometheus scraping
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:9090/api/v1/targets") as response:
                    if response.status == 200:
                        data = await response.json()
                        active_targets = len([t for t in data.get('data', {}).get('activeTargets', [])
                                            if t.get('health') == 'up'])
                        communication_tests.append((
                            "prometheus_scraping",
                            "PASS",
                            f"Prometheus scraping {active_targets} targets"
                        ))
                    else:
                        communication_tests.append((
                            "prometheus_scraping",
                            "FAIL",
                            f"Prometheus API returned {response.status}"
                        ))

        except Exception as e:
            communication_tests.append(("prometheus_scraping", "FAIL", f"Prometheus test failed: {str(e)}"))

        # Log all communication test results
        execution_time = time.time() - start_time
        all_passed = True

        for test_name, status, details in communication_tests:
            self.log_test_result(
                f"inter_service_{test_name}",
                status,
                details,
                execution_time / len(communication_tests)  # Approximate time per test
            )
            if status == "FAIL":
                all_passed = False

        return all_passed

    async def test_external_connectivity(self) -> bool:
        """Test external network connectivity."""
        logger.info("Testing external network connectivity...")

        start_time = time.time()
        connectivity_tests = []

        # Test THANOS connectivity
        thanos_config = self.network_tests["thanos_connectivity"]
        thanos_host = thanos_config["host"]

        for port in thanos_config["ports"]:
            try:
                # Test TCP connectivity
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(thanos_host, port),
                    timeout=5
                )
                writer.close()
                await writer.wait_closed()

                connectivity_tests.append((
                    f"thanos_port_{port}",
                    "PASS",
                    f"THANOS port {port} reachable"
                ))

            except Exception as e:
                connectivity_tests.append((
                    f"thanos_port_{port}",
                    "FAIL",
                    f"THANOS port {port} unreachable: {str(e)}"
                ))

        # Test external DNS
        for dns_host in self.network_tests["external_dns"]["hosts"]:
            try:
                result = subprocess.run([
                    "ping", "-c", "3", "-W", "5", dns_host
                ], capture_output=True, text=True, timeout=20)

                if result.returncode == 0:
                    connectivity_tests.append((
                        f"external_dns_{dns_host.replace('.', '_')}",
                        "PASS",
                        f"External DNS {dns_host} reachable"
                    ))
                else:
                    connectivity_tests.append((
                        f"external_dns_{dns_host.replace('.', '_')}",
                        "FAIL",
                        f"External DNS {dns_host} unreachable"
                    ))

            except Exception as e:
                connectivity_tests.append((
                    f"external_dns_{dns_host.replace('.', '_')}",
                    "FAIL",
                    f"DNS test failed: {str(e)}"
                ))

        # Test container-to-external connectivity
        try:
            result = subprocess.run([
                "docker", "run", "--rm", "--network", "bev_oracle",
                "alpine:latest", "ping", "-c", "3", "8.8.8.8"
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                connectivity_tests.append((
                    "container_external",
                    "PASS",
                    "Container can reach external networks"
                ))
            else:
                connectivity_tests.append((
                    "container_external",
                    "FAIL",
                    "Container cannot reach external networks"
                ))

        except Exception as e:
            connectivity_tests.append((
                "container_external",
                "FAIL",
                f"Container external test failed: {str(e)}"
            ))

        # Log all connectivity test results
        execution_time = time.time() - start_time
        all_passed = True

        for test_name, status, details in connectivity_tests:
            self.log_test_result(
                f"external_connectivity_{test_name}",
                status,
                details,
                execution_time / len(connectivity_tests)
            )
            if status == "FAIL":
                all_passed = False

        return all_passed

    async def test_service_dependencies(self) -> bool:
        """Test service dependency relationships."""
        logger.info("Testing service dependencies...")

        start_time = time.time()

        try:
            # Parse compose file to understand dependencies
            with open(self.compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)

            services = compose_data.get('services', {})
            dependency_tests = []

            # Check key dependency relationships
            dependencies_to_test = [
                ("prometheus", "redis-arm", "Prometheus depends on Redis"),
                ("grafana", "prometheus", "Grafana depends on Prometheus"),
                ("alertmanager", "prometheus", "AlertManager depends on Prometheus"),
                ("vault", "redis-arm", "Vault depends on Redis"),
                ("minio2", "minio1", "MinIO2 depends on MinIO1"),
                ("minio3", "minio1", "MinIO3 depends on MinIO1")
            ]

            for dependent_service, dependency_service, description in dependencies_to_test:
                # Check if both services exist in compose
                if dependent_service in services and dependency_service in services:
                    # Check if dependency is defined
                    depends_on = services[dependent_service].get('depends_on', [])

                    if dependency_service in depends_on:
                        dependency_tests.append((
                            f"dependency_{dependent_service}_{dependency_service}",
                            "PASS",
                            f"Dependency correctly defined: {description}"
                        ))
                    else:
                        dependency_tests.append((
                            f"dependency_{dependent_service}_{dependency_service}",
                            "WARN",
                            f"Dependency not explicitly defined: {description}"
                        ))
                else:
                    dependency_tests.append((
                        f"dependency_{dependent_service}_{dependency_service}",
                        "FAIL",
                        f"Service not found in compose: {description}"
                    ))

            # Check if services are actually running in correct order
            running_services = []
            try:
                ps_result = subprocess.run([
                    "docker", "compose", "-f", str(self.compose_file), "ps", "--services", "--filter", "status=running"
                ], capture_output=True, text=True, cwd=str(self.project_root))

                if ps_result.returncode == 0:
                    running_services = ps_result.stdout.strip().split('\n')

                for dependent_service, dependency_service, description in dependencies_to_test:
                    if dependent_service in running_services and dependency_service in running_services:
                        dependency_tests.append((
                            f"runtime_dependency_{dependent_service}_{dependency_service}",
                            "PASS",
                            f"Both services running: {description}"
                        ))
                    elif dependent_service in running_services and dependency_service not in running_services:
                        dependency_tests.append((
                            f"runtime_dependency_{dependent_service}_{dependency_service}",
                            "FAIL",
                            f"Dependent service running but dependency not: {description}"
                        ))

            except Exception as e:
                logger.warning(f"Could not check runtime dependencies: {str(e)}")

            # Log all dependency test results
            execution_time = time.time() - start_time
            all_passed = True

            for test_name, status, details in dependency_tests:
                self.log_test_result(
                    test_name,
                    status,
                    details,
                    execution_time / len(dependency_tests)
                )
                if status == "FAIL":
                    all_passed = False

            return all_passed

        except Exception as e:
            self.log_test_result(
                "service_dependencies",
                "FAIL",
                f"Dependency test failed: {str(e)}",
                time.time() - start_time
            )
            return False

    async def run_all_tests(self) -> Dict:
        """Run all service integration tests."""
        logger.info("Starting comprehensive service integration testing...")

        start_time = time.time()

        # Run all test phases
        test_phases = [
            ("Docker Compose Configuration", self.test_docker_compose_configuration),
            ("Service Startup Sequence", self.test_service_startup_sequence),
            ("Service Health Checks", self.test_service_health_checks),
            ("Inter-Service Communication", self.test_inter_service_communication),
            ("External Connectivity", self.test_external_connectivity),
            ("Service Dependencies", self.test_service_dependencies)
        ]

        for phase_name, test_func in test_phases:
            logger.info(f"Running test phase: {phase_name}")
            try:
                await test_func()
            except Exception as e:
                logger.error(f"Test phase {phase_name} failed with exception: {str(e)}")
                self.log_test_result(
                    f"phase_{phase_name.lower().replace(' ', '_')}",
                    "FAIL",
                    f"Phase failed with exception: {str(e)}"
                )

        # Calculate final results
        total_time = time.time() - start_time
        self.results["total_execution_time"] = total_time

        success_rate = (self.results["summary"]["passed_tests"] /
                       self.results["summary"]["total_tests"] * 100) if self.results["summary"]["total_tests"] > 0 else 0

        self.results["summary"]["success_rate"] = success_rate

        logger.info(f"Service integration testing completed in {total_time:.2f}s")
        logger.info(f"Results: {self.results['summary']['passed_tests']}/{self.results['summary']['total_tests']} passed "
                   f"({success_rate:.1f}% success rate)")

        return self.results

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save test results to JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"/home/starlord/Projects/Bev/validation_results/service_integration_{timestamp}.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Service integration test results saved to: {output_path}")
        return str(output_path)


async def main():
    """Main entry point for service integration testing."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "/home/starlord/Projects/Bev"

    tester = ServiceIntegrationTester(project_root)

    try:
        results = await tester.run_all_tests()
        output_file = tester.save_results()

        # Print summary
        print("\n" + "="*60)
        print("SERVICE INTEGRATION TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {results['summary']['total_tests']}")
        print(f"Passed: {results['summary']['passed_tests']}")
        print(f"Failed: {results['summary']['failed_tests']}")
        print(f"Warnings: {results['summary']['warnings']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Execution Time: {results['total_execution_time']:.2f}s")
        print(f"Results saved to: {output_file}")
        print("="*60)

        # Exit with appropriate code
        if results['summary']['failed_tests'] == 0:
            print("✅ All service integration tests passed!")
            sys.exit(0)
        else:
            print("❌ Some service integration tests failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Service integration testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Service integration testing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())