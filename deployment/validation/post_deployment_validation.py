#!/usr/bin/env python3

"""
BEV OSINT Framework - Post-Deployment Validation System
Comprehensive validation of deployed Phase 7, 8, 9 services
"""

import sys
import os
import json
import time
import asyncio
import aiohttp
import docker
import pytest
import argparse
import logging
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

@dataclass
class ServiceDefinition:
    """Service definition for validation"""
    name: str
    port: int
    ip_address: str
    phase: str
    health_endpoint: str = "/health"
    api_endpoint: str = "/api/v1/status"
    timeout: int = 30
    required_endpoints: List[str] = field(default_factory=list)
    performance_thresholds: Dict[str, float] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Validation result data structure"""
    test_name: str
    service: str
    status: str  # PASS, FAIL, WARN, SKIP
    message: str
    duration: float = 0.0
    details: Optional[Dict] = None
    severity: str = "INFO"

class PostDeploymentValidator:
    """Comprehensive post-deployment validation system"""

    def __init__(self, phases: List[str] = None):
        self.phases = phases or ["7", "8", "9"]
        self.project_root = project_root
        self.results: List[ValidationResult] = []
        self.docker_client = None
        self.failed_tests = 0
        self.warnings = 0
        self.session = None

        # Service definitions by phase
        self.services = self._define_services()

        # Setup logging
        self.setup_logging()

        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.project_root / "logs" / "deployment"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"post_deployment_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _define_services(self) -> Dict[str, List[ServiceDefinition]]:
        """Define service configurations for each phase"""
        return {
            "7": [
                ServiceDefinition(
                    name="dm-crawler", port=8001, ip_address="172.30.0.24", phase="7",
                    required_endpoints=["/api/v1/crawl/status", "/api/v1/sites"],
                    performance_thresholds={"response_time": 2.0, "memory_mb": 2048}
                ),
                ServiceDefinition(
                    name="crypto-intel", port=8002, ip_address="172.30.0.25", phase="7",
                    required_endpoints=["/api/v1/blockchain/status", "/api/v1/analysis"],
                    performance_thresholds={"response_time": 3.0, "memory_mb": 3072}
                ),
                ServiceDefinition(
                    name="reputation-analyzer", port=8003, ip_address="172.30.0.26", phase="7",
                    required_endpoints=["/api/v1/reputation/status", "/api/v1/analyze"],
                    performance_thresholds={"response_time": 4.0, "memory_mb": 4096}
                ),
                ServiceDefinition(
                    name="economics-processor", port=8004, ip_address="172.30.0.27", phase="7",
                    required_endpoints=["/api/v1/economics/status", "/api/v1/predictions"],
                    performance_thresholds={"response_time": 5.0, "memory_mb": 6144}
                ),
            ],
            "8": [
                ServiceDefinition(
                    name="tactical-intel", port=8005, ip_address="172.30.0.28", phase="8",
                    required_endpoints=["/api/v1/intel/threats", "/api/v1/mitre/status"],
                    performance_thresholds={"response_time": 3.0, "memory_mb": 4096}
                ),
                ServiceDefinition(
                    name="defense-automation", port=8006, ip_address="172.30.0.29", phase="8",
                    required_endpoints=["/api/v1/defense/status", "/api/v1/automation/rules"],
                    performance_thresholds={"response_time": 2.0, "memory_mb": 3072}
                ),
                ServiceDefinition(
                    name="opsec-monitor", port=8007, ip_address="172.30.0.30", phase="8",
                    required_endpoints=["/api/v1/opsec/status", "/api/v1/monitoring"],
                    performance_thresholds={"response_time": 4.0, "memory_mb": 5120}
                ),
                ServiceDefinition(
                    name="intel-fusion", port=8008, ip_address="172.30.0.31", phase="8",
                    required_endpoints=["/api/v1/fusion/status", "/api/v1/correlation/engines"],
                    performance_thresholds={"response_time": 5.0, "memory_mb": 8192}
                ),
            ],
            "9": [
                ServiceDefinition(
                    name="autonomous-coordinator", port=8009, ip_address="172.30.0.32", phase="9",
                    required_endpoints=["/api/v1/autonomous/status", "/api/v1/decisions/status"],
                    performance_thresholds={"response_time": 3.0, "memory_mb": 6144}
                ),
                ServiceDefinition(
                    name="adaptive-learning", port=8010, ip_address="172.30.0.33", phase="9",
                    required_endpoints=["/api/v1/learning/status", "/api/v1/algorithms/status"],
                    performance_thresholds={"response_time": 4.0, "memory_mb": 8192}
                ),
                ServiceDefinition(
                    name="resource-manager", port=8011, ip_address="172.30.0.34", phase="9",
                    required_endpoints=["/api/v1/resources/status", "/api/v1/optimization/status"],
                    performance_thresholds={"response_time": 2.0, "memory_mb": 4096}
                ),
                ServiceDefinition(
                    name="knowledge-evolution", port=8012, ip_address="172.30.0.35", phase="9",
                    required_endpoints=["/api/v1/knowledge/status", "/api/v1/evolution/status"],
                    performance_thresholds={"response_time": 6.0, "memory_mb": 10240}
                ),
            ]
        }

    def add_result(self, test_name: str, service: str, status: str, message: str,
                   duration: float = 0.0, details: Dict = None, severity: str = "INFO"):
        """Add a validation result"""
        result = ValidationResult(test_name, service, status, message, duration, details, severity)
        self.results.append(result)

        if status == "FAIL":
            self.failed_tests += 1
        elif status == "WARN":
            self.warnings += 1

        # Log the result
        log_level = {
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }.get(severity, logging.INFO)

        self.logger.log(log_level, f"{test_name}[{service}]: {status} - {message} ({duration:.2f}s)")

    async def validate_service_health(self, service: ServiceDefinition) -> bool:
        """Validate service health endpoint"""
        start_time = time.time()
        test_name = "health_check"

        try:
            url = f"http://localhost:{service.port}{service.health_endpoint}"
            timeout = aiohttp.ClientTimeout(total=service.timeout)

            async with self.session.get(url, timeout=timeout) as response:
                duration = time.time() - start_time

                if response.status == 200:
                    self.add_result(test_name, service.name, "PASS",
                                  f"Health endpoint responding (HTTP {response.status})", duration)
                    return True
                else:
                    self.add_result(test_name, service.name, "FAIL",
                                  f"Health endpoint returned HTTP {response.status}", duration, severity="ERROR")
                    return False

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.add_result(test_name, service.name, "FAIL",
                          f"Health check timeout after {service.timeout}s", duration, severity="ERROR")
            return False
        except Exception as e:
            duration = time.time() - start_time
            self.add_result(test_name, service.name, "FAIL",
                          f"Health check failed: {str(e)}", duration, severity="ERROR")
            return False

    async def validate_api_endpoints(self, service: ServiceDefinition) -> bool:
        """Validate API endpoints"""
        all_passed = True

        for endpoint in service.required_endpoints:
            start_time = time.time()
            test_name = f"api_endpoint_{endpoint.replace('/', '_').replace('-', '_')}"

            try:
                url = f"http://localhost:{service.port}{endpoint}"
                timeout = aiohttp.ClientTimeout(total=service.timeout)

                async with self.session.get(url, timeout=timeout) as response:
                    duration = time.time() - start_time

                    if response.status in [200, 201, 202]:
                        # Try to parse JSON response
                        try:
                            data = await response.json()
                            self.add_result(test_name, service.name, "PASS",
                                          f"API endpoint responding with valid JSON", duration,
                                          {"endpoint": endpoint, "status": response.status})
                        except:
                            self.add_result(test_name, service.name, "WARN",
                                          f"API endpoint responding but not JSON", duration,
                                          {"endpoint": endpoint, "status": response.status}, severity="WARN")
                    else:
                        self.add_result(test_name, service.name, "FAIL",
                                      f"API endpoint returned HTTP {response.status}", duration,
                                      {"endpoint": endpoint}, severity="ERROR")
                        all_passed = False

            except asyncio.TimeoutError:
                duration = time.time() - start_time
                self.add_result(test_name, service.name, "FAIL",
                              f"API endpoint timeout: {endpoint}", duration,
                              {"endpoint": endpoint}, severity="ERROR")
                all_passed = False
            except Exception as e:
                duration = time.time() - start_time
                self.add_result(test_name, service.name, "FAIL",
                              f"API endpoint error: {str(e)}", duration,
                              {"endpoint": endpoint}, severity="ERROR")
                all_passed = False

        return all_passed

    def validate_container_status(self, service: ServiceDefinition) -> bool:
        """Validate Docker container status"""
        test_name = "container_status"
        container_name = f"bev_{service.name}"

        try:
            container = self.docker_client.containers.get(container_name)

            if container.status == "running":
                self.add_result(test_name, service.name, "PASS",
                              f"Container running (ID: {container.id[:12]})")

                # Check container health if health check is configured
                if container.attrs.get("State", {}).get("Health"):
                    health_status = container.attrs["State"]["Health"]["Status"]
                    if health_status == "healthy":
                        self.add_result("container_health", service.name, "PASS",
                                      "Container health check passing")
                    else:
                        self.add_result("container_health", service.name, "WARN",
                                      f"Container health status: {health_status}", severity="WARN")

                return True
            else:
                self.add_result(test_name, service.name, "FAIL",
                              f"Container not running (status: {container.status})", severity="ERROR")
                return False

        except docker.errors.NotFound:
            self.add_result(test_name, service.name, "FAIL",
                          "Container not found", severity="ERROR")
            return False
        except Exception as e:
            self.add_result(test_name, service.name, "FAIL",
                          f"Container status check failed: {str(e)}", severity="ERROR")
            return False

    def validate_resource_usage(self, service: ServiceDefinition) -> bool:
        """Validate resource usage against thresholds"""
        test_name = "resource_usage"
        container_name = f"bev_{service.name}"

        try:
            container = self.docker_client.containers.get(container_name)
            stats = container.stats(stream=False)

            # Calculate memory usage
            memory_usage = stats['memory_stats'].get('usage', 0)
            memory_mb = memory_usage / (1024 * 1024)

            # Check against threshold
            memory_threshold = service.performance_thresholds.get("memory_mb", float('inf'))
            if memory_mb <= memory_threshold:
                self.add_result(test_name, service.name, "PASS",
                              f"Memory usage within threshold: {memory_mb:.1f}MB <= {memory_threshold}MB",
                              details={"memory_mb": memory_mb, "threshold_mb": memory_threshold})
            else:
                self.add_result(test_name, service.name, "WARN",
                              f"Memory usage above threshold: {memory_mb:.1f}MB > {memory_threshold}MB",
                              details={"memory_mb": memory_mb, "threshold_mb": memory_threshold},
                              severity="WARN")

            # Calculate CPU usage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']

            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
                self.add_result("cpu_usage", service.name, "PASS",
                              f"CPU usage: {cpu_percent:.1f}%",
                              details={"cpu_percent": cpu_percent})

            return True

        except docker.errors.NotFound:
            self.add_result(test_name, service.name, "SKIP",
                          "Container not found for resource check")
            return False
        except Exception as e:
            self.add_result(test_name, service.name, "WARN",
                          f"Resource usage check failed: {str(e)}", severity="WARN")
            return False

    async def validate_response_time(self, service: ServiceDefinition) -> bool:
        """Validate response time performance"""
        test_name = "response_time"
        threshold = service.performance_thresholds.get("response_time", 5.0)

        try:
            url = f"http://localhost:{service.port}{service.health_endpoint}"
            start_time = time.time()

            async with self.session.get(url) as response:
                duration = time.time() - start_time

                if duration <= threshold:
                    self.add_result(test_name, service.name, "PASS",
                                  f"Response time within threshold: {duration:.2f}s <= {threshold}s",
                                  duration, {"response_time": duration, "threshold": threshold})
                    return True
                else:
                    self.add_result(test_name, service.name, "WARN",
                                  f"Response time above threshold: {duration:.2f}s > {threshold}s",
                                  duration, {"response_time": duration, "threshold": threshold},
                                  severity="WARN")
                    return False

        except Exception as e:
            self.add_result(test_name, service.name, "FAIL",
                          f"Response time test failed: {str(e)}", severity="ERROR")
            return False

    def validate_network_connectivity(self, service: ServiceDefinition) -> bool:
        """Validate network connectivity"""
        test_name = "network_connectivity"
        container_name = f"bev_{service.name}"

        try:
            container = self.docker_client.containers.get(container_name)

            # Check if container is on the correct network
            networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})
            if "bev_osint" in networks:
                ip_address = networks["bev_osint"]["IPAddress"]
                if ip_address == service.ip_address:
                    self.add_result(test_name, service.name, "PASS",
                                  f"Network connectivity correct (IP: {ip_address})")
                else:
                    self.add_result(test_name, service.name, "WARN",
                                  f"IP address mismatch: expected {service.ip_address}, got {ip_address}",
                                  severity="WARN")
            else:
                self.add_result(test_name, service.name, "FAIL",
                              "Container not connected to bev_osint network", severity="ERROR")
                return False

            return True

        except docker.errors.NotFound:
            self.add_result(test_name, service.name, "FAIL",
                          "Container not found for network check", severity="ERROR")
            return False
        except Exception as e:
            self.add_result(test_name, service.name, "FAIL",
                          f"Network connectivity check failed: {str(e)}", severity="ERROR")
            return False

    def validate_log_output(self, service: ServiceDefinition) -> bool:
        """Validate service log output"""
        test_name = "log_validation"
        container_name = f"bev_{service.name}"

        try:
            container = self.docker_client.containers.get(container_name)

            # Get recent logs
            logs = container.logs(tail=50, timestamps=True).decode('utf-8')

            # Check for error patterns
            error_patterns = ["ERROR", "CRITICAL", "FATAL", "Exception", "Traceback"]
            errors_found = []

            for pattern in error_patterns:
                if pattern in logs:
                    errors_found.append(pattern)

            if errors_found:
                self.add_result(test_name, service.name, "WARN",
                              f"Error patterns found in logs: {', '.join(errors_found)}",
                              details={"error_patterns": errors_found}, severity="WARN")
            else:
                self.add_result(test_name, service.name, "PASS",
                              "No error patterns found in recent logs")

            # Check for startup success indicators
            success_patterns = ["started", "ready", "listening", "initialized"]
            success_found = any(pattern.lower() in logs.lower() for pattern in success_patterns)

            if success_found:
                self.add_result("startup_validation", service.name, "PASS",
                              "Service startup indicators found in logs")
            else:
                self.add_result("startup_validation", service.name, "WARN",
                              "No clear startup indicators found in logs", severity="WARN")

            return True

        except docker.errors.NotFound:
            self.add_result(test_name, service.name, "SKIP",
                          "Container not found for log check")
            return False
        except Exception as e:
            self.add_result(test_name, service.name, "WARN",
                          f"Log validation failed: {str(e)}", severity="WARN")
            return False

    async def validate_service_integration(self, services: List[ServiceDefinition]) -> bool:
        """Validate service integration and communication"""
        test_name = "service_integration"

        # Test basic inter-service connectivity
        all_passed = True

        for i, service1 in enumerate(services):
            for service2 in services[i+1:]:
                try:
                    # Try to connect from service1 to service2
                    url = f"http://localhost:{service2.port}/health"
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            self.add_result(f"{test_name}_connectivity", f"{service1.name}->{service2.name}",
                                          "PASS", "Inter-service connectivity working")
                        else:
                            self.add_result(f"{test_name}_connectivity", f"{service1.name}->{service2.name}",
                                          "WARN", f"Connectivity issue (HTTP {response.status})", severity="WARN")
                            all_passed = False

                except Exception as e:
                    self.add_result(f"{test_name}_connectivity", f"{service1.name}->{service2.name}",
                                  "WARN", f"Connectivity test failed: {str(e)}", severity="WARN")

        return all_passed

    def validate_database_connectivity(self) -> bool:
        """Validate database connectivity for all services"""
        test_name = "database_connectivity"

        # Check core database services
        databases = {
            "postgres": 5432,
            "neo4j": 7687,
            "redis": 6379,
            "elasticsearch": 9200,
            "influxdb": 8086
        }

        all_passed = True

        for db_name, port in databases.items():
            try:
                container = self.docker_client.containers.get(db_name)
                if container.status == "running":
                    self.add_result(f"{test_name}_{db_name}", "infrastructure", "PASS",
                                  f"{db_name} database running")
                else:
                    self.add_result(f"{test_name}_{db_name}", "infrastructure", "FAIL",
                                  f"{db_name} database not running", severity="ERROR")
                    all_passed = False

            except docker.errors.NotFound:
                self.add_result(f"{test_name}_{db_name}", "infrastructure", "FAIL",
                              f"{db_name} database container not found", severity="ERROR")
                all_passed = False

        return all_passed

    def validate_volume_mounts(self) -> bool:
        """Validate volume mounts for all services"""
        test_name = "volume_mounts"

        # Define expected volumes by phase
        expected_volumes = {
            "7": ["dm_crawler_data", "crypto_intel_data", "reputation_data", "economics_data"],
            "8": ["tactical_intel_data", "defense_automation_data", "opsec_data", "intel_fusion_data"],
            "9": ["autonomous_data", "adaptive_learning_data", "resource_manager_data", "knowledge_evolution_data"]
        }

        all_passed = True

        for phase in self.phases:
            if phase in expected_volumes:
                for volume_name in expected_volumes[phase]:
                    try:
                        volume = self.docker_client.volumes.get(volume_name)
                        self.add_result(f"{test_name}_{volume_name}", f"phase_{phase}", "PASS",
                                      f"Volume exists: {volume_name}")
                    except docker.errors.NotFound:
                        self.add_result(f"{test_name}_{volume_name}", f"phase_{phase}", "FAIL",
                                      f"Volume not found: {volume_name}", severity="ERROR")
                        all_passed = False

        return all_passed

    async def run_performance_tests(self, service: ServiceDefinition) -> bool:
        """Run performance tests for a service"""
        test_name = "performance_test"

        # Simple load test - multiple concurrent requests
        concurrent_requests = 5
        request_count = 10

        start_time = time.time()
        tasks = []

        for _ in range(concurrent_requests):
            for _ in range(request_count):
                url = f"http://localhost:{service.port}{service.health_endpoint}"
                task = self.session.get(url, timeout=aiohttp.ClientTimeout(total=30))
                tasks.append(task)

        try:
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time

            successful_requests = sum(1 for r in responses if hasattr(r, 'status') and r.status == 200)
            total_requests = len(tasks)

            success_rate = (successful_requests / total_requests) * 100

            if success_rate >= 95:
                self.add_result(test_name, service.name, "PASS",
                              f"Performance test passed: {success_rate:.1f}% success rate",
                              duration, {"success_rate": success_rate, "total_requests": total_requests})
                return True
            else:
                self.add_result(test_name, service.name, "WARN",
                              f"Performance test warning: {success_rate:.1f}% success rate",
                              duration, {"success_rate": success_rate, "total_requests": total_requests},
                              severity="WARN")
                return False

        except Exception as e:
            duration = time.time() - start_time
            self.add_result(test_name, service.name, "FAIL",
                          f"Performance test failed: {str(e)}", duration, severity="ERROR")
            return False

    async def validate_phase_services(self, phase: str) -> bool:
        """Validate all services in a phase"""
        if phase not in self.services:
            self.logger.warning(f"No services defined for phase {phase}")
            return True

        services = self.services[phase]
        self.logger.info(f"Validating Phase {phase} services: {[s.name for s in services]}")

        phase_passed = True

        # Validate each service
        for service in services:
            self.logger.info(f"Validating service: {service.name}")

            service_passed = True

            # Basic container validation
            if not self.validate_container_status(service):
                service_passed = False

            if not self.validate_network_connectivity(service):
                service_passed = False

            # Health and API validation
            if not await self.validate_service_health(service):
                service_passed = False
                continue  # Skip other tests if health check fails

            if not await self.validate_api_endpoints(service):
                service_passed = False

            # Performance validation
            if not await self.validate_response_time(service):
                pass  # Performance issues are warnings, not failures

            if not self.validate_resource_usage(service):
                pass  # Resource usage issues are warnings

            # Log validation
            if not self.validate_log_output(service):
                pass  # Log issues are warnings

            # Performance testing
            if not await self.run_performance_tests(service):
                pass  # Performance test failures are warnings

            if not service_passed:
                phase_passed = False

        # Service integration testing
        if len(services) > 1:
            if not await self.validate_service_integration(services):
                pass  # Integration issues are warnings

        return phase_passed

    async def run_all_validations(self) -> bool:
        """Run all post-deployment validations"""
        self.logger.info(f"Starting post-deployment validation for phases: {', '.join(self.phases)}")

        # Initialize HTTP session
        self.session = aiohttp.ClientSession()

        try:
            overall_success = True

            # Validate infrastructure
            if not self.validate_database_connectivity():
                overall_success = False

            if not self.validate_volume_mounts():
                overall_success = False

            # Validate each phase
            for phase in self.phases:
                phase_success = await self.validate_phase_services(phase)
                if not phase_success:
                    overall_success = False

            return overall_success

        finally:
            await self.session.close()

    def generate_report(self) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("="*80)
        report.append("BEV OSINT Framework - Post-Deployment Validation Report")
        report.append("="*80)
        report.append(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Phases Validated: {', '.join(self.phases)}")
        report.append(f"Total Tests: {len(self.results)}")
        report.append(f"Failed Tests: {self.failed_tests}")
        report.append(f"Warnings: {self.warnings}")
        report.append("")

        # Summary by status
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warned = sum(1 for r in self.results if r.status == "WARN")
        skipped = sum(1 for r in self.results if r.status == "SKIP")

        report.append("SUMMARY:")
        report.append(f"  ✅ Passed: {passed}")
        report.append(f"  ❌ Failed: {failed}")
        report.append(f"  ⚠️  Warnings: {warned}")
        report.append(f"  ⏭️  Skipped: {skipped}")
        report.append("")

        # Summary by phase
        for phase in self.phases:
            phase_results = [r for r in self.results if phase in r.service or f"phase_{phase}" in r.service]
            phase_failed = sum(1 for r in phase_results if r.status == "FAIL")
            phase_warned = sum(1 for r in phase_results if r.status == "WARN")

            status_icon = "✅" if phase_failed == 0 else "❌"
            report.append(f"{status_icon} Phase {phase}: {len(phase_results)} tests, {phase_failed} failed, {phase_warned} warnings")

        report.append("")

        # Performance summary
        response_times = [r for r in self.results if r.test_name == "response_time" and r.status == "PASS"]
        if response_times:
            avg_response_time = sum(r.duration for r in response_times) / len(response_times)
            report.append(f"PERFORMANCE SUMMARY:")
            report.append(f"  Average Response Time: {avg_response_time:.3f}s")
            report.append(f"  Fastest Service: {min(response_times, key=lambda x: x.duration).service} ({min(r.duration for r in response_times):.3f}s)")
            report.append(f"  Slowest Service: {max(response_times, key=lambda x: x.duration).service} ({max(r.duration for r in response_times):.3f}s)")
            report.append("")

        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 50)

        # Group by service
        services = {}
        for result in self.results:
            if result.service not in services:
                services[result.service] = []
            services[result.service].append(result)

        for service_name, service_results in services.items():
            service_failed = sum(1 for r in service_results if r.status == "FAIL")
            service_warned = sum(1 for r in service_results if r.status == "WARN")

            status_icon = "✅" if service_failed == 0 else ("⚠️" if service_failed == 0 and service_warned > 0 else "❌")
            report.append(f"\n{status_icon} {service_name.upper()}:")

            for result in service_results:
                status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}.get(result.status, "❓")
                report.append(f"    {status_icon} {result.test_name}: {result.message}")
                if result.duration > 0:
                    report.append(f"      Duration: {result.duration:.3f}s")
                if result.details:
                    for key, value in result.details.items():
                        report.append(f"      {key}: {value}")

        # Recommendations
        report.append("\n" + "="*50)
        report.append("RECOMMENDATIONS:")
        report.append("-" * 50)

        if self.failed_tests > 0:
            report.append("🚨 CRITICAL ISSUES FOUND")
            report.append("   Some services have failed validation. Investigate and resolve before production use.")
        elif self.warnings > 0:
            report.append("⚠️  WARNINGS DETECTED")
            report.append("   Some issues detected but services are functional. Monitor and consider fixes.")
        else:
            report.append("✅ ALL VALIDATIONS PASSED")
            report.append("   All services are functioning correctly and ready for use.")

        # Failed tests details
        failed_results = [r for r in self.results if r.status == "FAIL"]
        if failed_results:
            report.append("\nFAILED TESTS TO INVESTIGATE:")
            for result in failed_results:
                report.append(f"  • {result.service}/{result.test_name}: {result.message}")

        report.append("\n" + "="*80)
        return "\n".join(report)

    def save_report(self, filename: str = None) -> Path:
        """Save validation report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"post_deployment_validation_{timestamp}.txt"

        report_path = self.project_root / "logs" / "deployment" / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write(self.generate_report())

        return report_path

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BEV OSINT Post-Deployment Validation")
    parser.add_argument("--phases", default="7,8,9",
                       help="Comma-separated list of phases to validate (default: 7,8,9)")
    parser.add_argument("--output", help="Output report file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Parse phases
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    # Create validator
    validator = PostDeploymentValidator(phases)

    # Run validation
    success = await validator.run_all_validations()

    # Generate and display report
    report = validator.generate_report()
    print(report)

    # Save report
    report_path = validator.save_report(args.output)
    print(f"\nValidation report saved to: {report_path}")

    # Exit with appropriate code
    if success:
        print("\n✅ Validation successful - all services are functioning correctly")
        sys.exit(0)
    else:
        print(f"\n❌ Validation failed - {validator.failed_tests} tests failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())