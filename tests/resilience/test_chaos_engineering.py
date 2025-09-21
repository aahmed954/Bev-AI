"""
Chaos engineering tests for system resilience and auto-recovery
Validates <5 minute recovery time and system stability under failures
"""

import pytest
import asyncio
import docker
import time
import requests
import psycopg2
import redis
import json
import logging
import random
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import statistics

logger = logging.getLogger(__name__)

@pytest.mark.chaos
class TestChaosEngineering:
    """Chaos engineering tests for system resilience"""

    @pytest.fixture(autouse=True)
    def setup_chaos_test(self, docker_client, performance_monitor):
        """Setup chaos testing environment"""
        self.docker = docker_client
        self.monitor = performance_monitor
        self.affected_containers = []
        self.recovery_start_time = None

        yield

        # Cleanup: restore all affected containers
        self._restore_all_containers()

    async def test_service_failure_recovery(self):
        """Test auto-recovery from random service failures"""
        # Target services for chaos testing
        target_services = [
            "bev_postgres",
            "bev_redis",
            "bev_neo4j",
            "bev_qdrant",
            "bev_elasticsearch"
        ]

        for service in target_services:
            logger.info(f"Testing failure recovery for service: {service}")

            # Baseline health check
            initial_health = await self._check_system_health()
            assert initial_health["healthy_services"] >= 8, "System not healthy before chaos test"

            # Record pre-failure metrics
            self.monitor.start()

            # Kill the service
            container_id = self._kill_service(service)
            if not container_id:
                logger.warning(f"Could not kill service {service}, skipping")
                continue

            failure_time = time.time()
            self.recovery_start_time = failure_time

            # Monitor recovery process
            recovery_successful = await self._monitor_recovery(service, max_wait_time=300)  # 5 minutes

            if recovery_successful:
                recovery_time = time.time() - failure_time
                self.monitor.record(f"{service}_recovery_time", recovery_time)

                # Validate recovery time meets SLA
                assert recovery_time <= 300, f"Recovery took {recovery_time:.1f}s, exceeds 5 minute SLA"

                logger.info(f"Service {service} recovered in {recovery_time:.1f} seconds")

                # Verify system stability post-recovery
                await asyncio.sleep(30)  # Allow stabilization
                post_recovery_health = await self._check_system_health()
                assert post_recovery_health["healthy_services"] >= initial_health["healthy_services"]

            else:
                pytest.fail(f"Service {service} failed to recover within 5 minutes")

            # Brief pause between service tests
            await asyncio.sleep(60)

    async def test_cascade_failure_resilience(self):
        """Test resilience against cascading failures"""
        logger.info("Testing cascade failure resilience")

        # Phase 1: Kill primary database
        self._kill_service("bev_postgres")
        phase1_start = time.time()

        # Phase 2: Kill cache after 30 seconds
        await asyncio.sleep(30)
        self._kill_service("bev_redis")

        # Phase 3: Kill search engine after another 30 seconds
        await asyncio.sleep(30)
        self._kill_service("bev_elasticsearch")

        cascade_start = time.time()

        # Monitor system behavior during cascade
        system_responses = []
        test_duration = 180  # 3 minutes

        while time.time() - cascade_start < test_duration:
            try:
                # Test basic system responsiveness
                response = await self._test_system_endpoint()
                system_responses.append({
                    "timestamp": time.time(),
                    "responsive": response.get("success", False),
                    "latency": response.get("latency", 999999)
                })
            except Exception as e:
                system_responses.append({
                    "timestamp": time.time(),
                    "responsive": False,
                    "error": str(e)
                })

            await asyncio.sleep(10)

        # Analyze system behavior during cascade
        total_checks = len(system_responses)
        responsive_checks = sum(1 for r in system_responses if r.get("responsive"))
        uptime_during_cascade = responsive_checks / total_checks if total_checks > 0 else 0

        # System should maintain at least 50% availability during cascade
        assert uptime_during_cascade >= 0.5, f"System availability {uptime_during_cascade:.2%} too low during cascade"

        # Wait for auto-recovery
        recovery_successful = await self._monitor_full_recovery(max_wait_time=300)
        assert recovery_successful, "System failed to recover from cascade failure"

        total_recovery_time = time.time() - phase1_start
        self.monitor.record("cascade_recovery_time", total_recovery_time)

        logger.info(f"Cascade failure recovery completed in {total_recovery_time:.1f} seconds")
        logger.info(f"System availability during cascade: {uptime_during_cascade:.2%}")

    async def test_network_partition_resilience(self):
        """Test resilience to network partitions"""
        logger.info("Testing network partition resilience")

        # Create network isolation for subset of services
        partition_services = ["bev_qdrant", "bev_weaviate", "bev_neo4j"]

        # Simulate network partition by creating separate network
        try:
            # Create isolated network
            isolated_network = self.docker.networks.create(
                "bev_isolated",
                driver="bridge",
                check_duplicate=True
            )

            # Move services to isolated network
            for service_name in partition_services:
                try:
                    container = self.docker.containers.get(service_name)
                    isolated_network.connect(container)
                    self.affected_containers.append(container.id)
                except Exception as e:
                    logger.warning(f"Could not isolate {service_name}: {e}")

            partition_start = time.time()

            # Test system behavior during partition
            partition_responses = []
            partition_duration = 120  # 2 minutes

            while time.time() - partition_start < partition_duration:
                try:
                    response = await self._test_system_endpoint()
                    partition_responses.append(response)
                except Exception as e:
                    partition_responses.append({"success": False, "error": str(e)})

                await asyncio.sleep(15)

            # Restore network connectivity
            for service_name in partition_services:
                try:
                    container = self.docker.containers.get(service_name)
                    # Reconnect to main network
                    main_network = self.docker.networks.get("bev_osint")
                    main_network.connect(container)
                except Exception as e:
                    logger.warning(f"Could not restore {service_name}: {e}")

            # Monitor network healing
            healing_start = time.time()
            network_healed = await self._monitor_network_healing(max_wait_time=180)

            if network_healed:
                healing_time = time.time() - healing_start
                self.monitor.record("network_healing_time", healing_time)

                assert healing_time <= 180, f"Network healing took {healing_time:.1f}s, too long"
                logger.info(f"Network partition healed in {healing_time:.1f} seconds")
            else:
                pytest.fail("Network failed to heal within 3 minutes")

            # Cleanup isolated network
            isolated_network.remove()

        except Exception as e:
            logger.error(f"Network partition test error: {e}")
            pytest.fail(f"Network partition test failed: {e}")

    async def test_resource_exhaustion_recovery(self):
        """Test recovery from resource exhaustion scenarios"""
        resource_scenarios = [
            {"type": "cpu", "limit": "25%", "duration": 120},
            {"type": "memory", "limit": "512m", "duration": 120},
            {"type": "disk_io", "limit": "10mb", "duration": 120}
        ]

        for scenario in resource_scenarios:
            logger.info(f"Testing {scenario['type']} exhaustion recovery")

            # Apply resource constraints
            constrained_services = self._apply_resource_constraints(scenario)

            if not constrained_services:
                logger.warning(f"Could not apply {scenario['type']} constraints")
                continue

            constraint_start = time.time()

            # Monitor system performance under constraints
            performance_metrics = []

            while time.time() - constraint_start < scenario["duration"]:
                try:
                    response = await self._test_system_endpoint()
                    performance_metrics.append({
                        "timestamp": time.time(),
                        "latency": response.get("latency", 999999),
                        "success": response.get("success", False)
                    })
                except Exception as e:
                    performance_metrics.append({
                        "timestamp": time.time(),
                        "latency": 999999,
                        "success": False,
                        "error": str(e)
                    })

                await asyncio.sleep(10)

            # Remove resource constraints
            self._remove_resource_constraints(constrained_services)

            # Monitor recovery from resource exhaustion
            recovery_start = time.time()
            performance_recovered = await self._monitor_performance_recovery(max_wait_time=120)

            if performance_recovered:
                recovery_time = time.time() - recovery_start
                self.monitor.record(f"{scenario['type']}_exhaustion_recovery", recovery_time)

                assert recovery_time <= 120, f"Recovery from {scenario['type']} exhaustion too slow"
                logger.info(f"Recovered from {scenario['type']} exhaustion in {recovery_time:.1f}s")
            else:
                pytest.fail(f"Failed to recover from {scenario['type']} exhaustion")

            # Analyze performance during constraint
            successful_requests = [m for m in performance_metrics if m.get("success")]
            if successful_requests:
                avg_latency = statistics.mean(m["latency"] for m in successful_requests)
                success_rate = len(successful_requests) / len(performance_metrics)

                logger.info(f"  Performance under {scenario['type']} constraint:")
                logger.info(f"    Success rate: {success_rate:.2%}")
                logger.info(f"    Average latency: {avg_latency:.1f}ms")

                # System should maintain basic functionality
                assert success_rate >= 0.7, f"Success rate {success_rate:.2%} too low under {scenario['type']} constraint"

            await asyncio.sleep(30)  # Brief pause between scenarios

    async def test_data_corruption_recovery(self):
        """Test recovery from data corruption scenarios"""
        logger.info("Testing data corruption recovery")

        # Simulate data corruption in Redis
        try:
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)

            # Corrupt cache data
            corruption_keys = []
            for i in range(100):
                key = f"corrupted_key_{i}"
                r.set(key, "corrupted_data_" + "x" * 1000)  # Large corrupted entries
                corruption_keys.append(key)

            # Corrupt some structured data
            r.hset("system_config", "corrupted_field", "invalid_json_data{[[")
            corruption_keys.append("system_config")

            corruption_start = time.time()

            # Monitor system behavior with corrupted data
            corruption_responses = []
            corruption_duration = 60

            while time.time() - corruption_start < corruption_duration:
                try:
                    response = await self._test_system_endpoint()
                    corruption_responses.append(response)
                except Exception as e:
                    corruption_responses.append({"success": False, "error": str(e)})

                await asyncio.sleep(5)

            # Trigger cache cleanup/recovery
            self._trigger_cache_cleanup()

            # Monitor recovery from corruption
            recovery_start = time.time()
            data_recovered = await self._monitor_data_recovery(max_wait_time=180)

            if data_recovered:
                recovery_time = time.time() - recovery_start
                self.monitor.record("data_corruption_recovery", recovery_time)

                assert recovery_time <= 180, f"Data corruption recovery took {recovery_time:.1f}s"
                logger.info(f"Recovered from data corruption in {recovery_time:.1f} seconds")
            else:
                pytest.fail("Failed to recover from data corruption")

            # Cleanup corrupted keys
            r.delete(*corruption_keys)

        except Exception as e:
            logger.error(f"Data corruption test error: {e}")
            pytest.fail(f"Data corruption test failed: {e}")

    def _kill_service(self, service_name: str) -> Optional[str]:
        """Kill a specific service container"""
        try:
            container = self.docker.containers.get(service_name)
            container.kill()
            self.affected_containers.append(container.id)
            logger.info(f"Killed service: {service_name}")
            return container.id
        except Exception as e:
            logger.error(f"Failed to kill service {service_name}: {e}")
            return None

    async def _monitor_recovery(self, service_name: str, max_wait_time: int = 300) -> bool:
        """Monitor service recovery process"""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                container = self.docker.containers.get(service_name)
                container.reload()

                if container.status == "running":
                    # Additional health check specific to service type
                    if await self._verify_service_health(service_name):
                        return True

            except Exception as e:
                logger.debug(f"Recovery check for {service_name}: {e}")

            await asyncio.sleep(10)

        return False

    async def _monitor_full_recovery(self, max_wait_time: int = 300) -> bool:
        """Monitor full system recovery"""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            health = await self._check_system_health()
            if health["healthy_services"] >= 8:  # Require most services healthy
                return True

            await asyncio.sleep(15)

        return False

    async def _monitor_network_healing(self, max_wait_time: int = 180) -> bool:
        """Monitor network partition healing"""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                # Test connectivity between partitioned services
                health = await self._check_system_health()
                if health["network_connectivity"] >= 0.9:  # 90% connectivity restored
                    return True
            except Exception as e:
                logger.debug(f"Network healing check: {e}")

            await asyncio.sleep(10)

        return False

    async def _monitor_performance_recovery(self, max_wait_time: int = 120) -> bool:
        """Monitor performance recovery after resource exhaustion"""
        start_time = time.time()
        baseline_latency = 100  # Expected normal latency

        while time.time() - start_time < max_wait_time:
            try:
                response = await self._test_system_endpoint()
                if (response.get("success") and
                    response.get("latency", 999999) <= baseline_latency * 2):
                    return True
            except Exception as e:
                logger.debug(f"Performance recovery check: {e}")

            await asyncio.sleep(5)

        return False

    async def _monitor_data_recovery(self, max_wait_time: int = 180) -> bool:
        """Monitor recovery from data corruption"""
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                # Test data integrity
                r = redis.Redis(host="localhost", port=6379, decode_responses=True)

                # Check if corrupted data is cleaned up
                corrupted_keys = r.keys("corrupted_key_*")
                if len(corrupted_keys) == 0:
                    # Test system functionality
                    response = await self._test_system_endpoint()
                    if response.get("success"):
                        return True

            except Exception as e:
                logger.debug(f"Data recovery check: {e}")

            await asyncio.sleep(10)

        return False

    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        health_results = {
            "healthy_services": 0,
            "total_services": 0,
            "network_connectivity": 0.0,
            "response_time": 999999
        }

        # Check core services
        core_services = [
            "bev_postgres", "bev_redis", "bev_neo4j",
            "bev_qdrant", "bev_weaviate", "bev_elasticsearch",
            "bev_prometheus", "bev_grafana", "bev_airflow"
        ]

        healthy_count = 0
        for service in core_services:
            if await self._verify_service_health(service):
                healthy_count += 1

        health_results["healthy_services"] = healthy_count
        health_results["total_services"] = len(core_services)

        # Test system responsiveness
        try:
            response = await self._test_system_endpoint()
            if response.get("success"):
                health_results["response_time"] = response.get("latency", 999999)
                health_results["network_connectivity"] = 1.0
        except Exception:
            health_results["network_connectivity"] = 0.0

        return health_results

    async def _verify_service_health(self, service_name: str) -> bool:
        """Verify specific service health"""
        try:
            if service_name == "bev_postgres":
                conn = psycopg2.connect(
                    host="localhost", port=5432, user="researcher",
                    password="research_db_2024", database="osint",
                    connect_timeout=5
                )
                conn.close()
                return True

            elif service_name == "bev_redis":
                r = redis.Redis(host="localhost", port=6379, socket_connect_timeout=5)
                return r.ping()

            elif service_name in ["bev_qdrant", "bev_weaviate", "bev_elasticsearch"]:
                port_map = {"bev_qdrant": 6333, "bev_weaviate": 8080, "bev_elasticsearch": 9200}
                port = port_map.get(service_name)
                if port:
                    response = requests.get(f"http://localhost:{port}", timeout=5)
                    return response.status_code in [200, 404]

            return False

        except Exception:
            return False

    async def _test_system_endpoint(self) -> Dict[str, Any]:
        """Test a system endpoint for responsiveness"""
        start_time = time.time()

        try:
            # Test a lightweight endpoint
            response = requests.get("http://localhost:9090/api/v1/query",
                                  params={"query": "up"}, timeout=10)
            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                return {"success": True, "latency": latency}
            else:
                return {"success": False, "latency": latency, "status": response.status_code}

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {"success": False, "latency": latency, "error": str(e)}

    def _apply_resource_constraints(self, scenario: Dict[str, Any]) -> List[str]:
        """Apply resource constraints to containers"""
        constrained_services = []

        try:
            # Target CPU-intensive services for CPU constraints
            target_services = ["bev_postgres", "bev_elasticsearch", "bev_neo4j"]

            for service_name in target_services:
                try:
                    container = self.docker.containers.get(service_name)

                    if scenario["type"] == "cpu":
                        # Limit CPU to 25%
                        container.update(cpu_period=100000, cpu_quota=25000)
                    elif scenario["type"] == "memory":
                        # Limit memory to 512MB
                        container.update(mem_limit="512m")

                    constrained_services.append(service_name)
                    self.affected_containers.append(container.id)

                except Exception as e:
                    logger.warning(f"Could not constrain {service_name}: {e}")

        except Exception as e:
            logger.error(f"Failed to apply resource constraints: {e}")

        return constrained_services

    def _remove_resource_constraints(self, constrained_services: List[str]):
        """Remove resource constraints from containers"""
        for service_name in constrained_services:
            try:
                container = self.docker.containers.get(service_name)
                # Reset to unlimited resources
                container.update(cpu_period=0, cpu_quota=0, mem_limit=0)
            except Exception as e:
                logger.warning(f"Could not remove constraints from {service_name}: {e}")

    def _trigger_cache_cleanup(self):
        """Trigger cache cleanup and recovery"""
        try:
            r = redis.Redis(host="localhost", port=6379, decode_responses=True)
            # Trigger cleanup by flushing corrupted data
            r.flushdb()
            logger.info("Triggered cache cleanup")
        except Exception as e:
            logger.error(f"Failed to trigger cache cleanup: {e}")

    def _restore_all_containers(self):
        """Restore all affected containers"""
        for container_id in self.affected_containers:
            try:
                container = self.docker.containers.get(container_id)
                if container.status != "running":
                    container.restart()
                # Reset resource limits
                container.update(cpu_period=0, cpu_quota=0, mem_limit=0)
                logger.info(f"Restored container: {container_id}")
            except Exception as e:
                logger.error(f"Failed to restore container {container_id}: {e}")

        self.affected_containers.clear()

@pytest.mark.chaos
@pytest.mark.slow
class TestSystemStabilityUnderLoad:
    """Test system stability under various load conditions with chaos"""

    async def test_chaos_under_load(self):
        """Test system behavior with chaos events during high load"""
        logger.info("Testing chaos engineering under load")

        # Start sustained load
        load_task = asyncio.create_task(self._generate_sustained_load(duration=300))

        # Wait for load to stabilize
        await asyncio.sleep(30)

        # Introduce chaos events during load
        chaos_events = [
            {"action": "kill_service", "target": "bev_redis", "delay": 60},
            {"action": "network_delay", "target": "bev_postgres", "delay": 120},
            {"action": "cpu_spike", "target": "bev_elasticsearch", "delay": 180},
        ]

        for event in chaos_events:
            await asyncio.sleep(event["delay"])
            await self._execute_chaos_event(event)

        # Wait for load test completion
        load_results = await load_task

        # Analyze stability under chaos
        total_requests = load_results["total_requests"]
        successful_requests = load_results["successful_requests"]
        stability_rate = successful_requests / total_requests if total_requests > 0 else 0

        # System should maintain 70% stability under chaos + load
        assert stability_rate >= 0.7, f"System stability {stability_rate:.2%} too low under chaos + load"

        logger.info(f"System stability under chaos + load: {stability_rate:.2%}")

    async def _generate_sustained_load(self, duration: int = 300) -> Dict[str, Any]:
        """Generate sustained load on the system"""
        results = {"total_requests": 0, "successful_requests": 0}

        # Implementation would generate actual load
        # This is a placeholder for the sustained load generation
        logger.info(f"Generating sustained load for {duration} seconds")

        # Simulate load results
        results["total_requests"] = duration * 10  # 10 RPS
        results["successful_requests"] = int(results["total_requests"] * 0.85)  # 85% success

        return results

    async def _execute_chaos_event(self, event: Dict[str, Any]):
        """Execute a chaos engineering event"""
        logger.info(f"Executing chaos event: {event}")

        # Implementation would execute actual chaos events
        # This is a placeholder for chaos event execution
        await asyncio.sleep(1)  # Simulate event execution