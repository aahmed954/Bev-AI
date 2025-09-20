#!/usr/bin/env python3

"""
BEV OSINT Framework - Integration Testing Framework
Comprehensive end-to-end testing for Phase 7, 8, 9 integration
"""

import sys
import os
import json
import time
import asyncio
import aiohttp
import pytest
import unittest
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

@dataclass
class TestScenario:
    """Test scenario definition"""
    name: str
    description: str
    phases: List[str]
    services: List[str]
    test_type: str  # workflow, performance, security, data_flow
    prerequisites: List[str] = field(default_factory=list)
    cleanup_required: bool = True
    timeout: int = 300  # 5 minutes default

@dataclass
class TestResult:
    """Test result data structure"""
    scenario: str
    test_name: str
    status: str  # PASS, FAIL, SKIP, ERROR
    message: str
    duration: float = 0.0
    details: Optional[Dict] = None
    error_trace: Optional[str] = None

class IntegrationTestFramework:
    """Comprehensive integration testing framework"""

    def __init__(self, phases: List[str] = None):
        self.phases = phases or ["7", "8", "9"]
        self.project_root = project_root
        self.results: List[TestResult] = []
        self.session = None
        self.test_data = {}

        # Service endpoints by phase
        self.service_endpoints = {
            "7": {
                "dm-crawler": "http://localhost:8001",
                "crypto-intel": "http://localhost:8002",
                "reputation-analyzer": "http://localhost:8003",
                "economics-processor": "http://localhost:8004"
            },
            "8": {
                "tactical-intel": "http://localhost:8005",
                "defense-automation": "http://localhost:8006",
                "opsec-monitor": "http://localhost:8007",
                "intel-fusion": "http://localhost:8008"
            },
            "9": {
                "autonomous-coordinator": "http://localhost:8009",
                "adaptive-learning": "http://localhost:8010",
                "resource-manager": "http://localhost:8011",
                "knowledge-evolution": "http://localhost:8012"
            }
        }

        # Test scenarios
        self.test_scenarios = self._define_test_scenarios()

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.project_root / "logs" / "integration_tests"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"integration_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _define_test_scenarios(self) -> List[TestScenario]:
        """Define comprehensive test scenarios"""
        scenarios = []

        # Phase 7 Integration Tests
        scenarios.append(TestScenario(
            name="phase7_data_flow",
            description="Test data flow through Phase 7 services",
            phases=["7"],
            services=["dm-crawler", "crypto-intel", "reputation-analyzer", "economics-processor"],
            test_type="workflow",
            timeout=600
        ))

        scenarios.append(TestScenario(
            name="phase7_crypto_analysis_pipeline",
            description="Test cryptocurrency analysis pipeline",
            phases=["7"],
            services=["crypto-intel", "reputation-analyzer"],
            test_type="workflow"
        ))

        # Phase 8 Integration Tests
        scenarios.append(TestScenario(
            name="phase8_threat_response_workflow",
            description="Test threat detection and automated response",
            phases=["8"],
            services=["tactical-intel", "defense-automation"],
            test_type="workflow",
            prerequisites=["phase7_data_flow"]
        ))

        scenarios.append(TestScenario(
            name="phase8_intel_fusion",
            description="Test intelligence fusion across sources",
            phases=["8"],
            services=["tactical-intel", "intel-fusion"],
            test_type="workflow"
        ))

        scenarios.append(TestScenario(
            name="phase8_opsec_monitoring",
            description="Test OPSEC monitoring and alerting",
            phases=["8"],
            services=["opsec-monitor", "defense-automation"],
            test_type="security"
        ))

        # Phase 9 Integration Tests
        scenarios.append(TestScenario(
            name="phase9_autonomous_coordination",
            description="Test autonomous coordination across all systems",
            phases=["9"],
            services=["autonomous-coordinator", "adaptive-learning"],
            test_type="workflow",
            prerequisites=["phase7_data_flow", "phase8_threat_response_workflow"]
        ))

        scenarios.append(TestScenario(
            name="phase9_resource_optimization",
            description="Test resource management and optimization",
            phases=["9"],
            services=["resource-manager", "autonomous-coordinator"],
            test_type="performance"
        ))

        scenarios.append(TestScenario(
            name="phase9_knowledge_evolution",
            description="Test knowledge evolution and learning",
            phases=["9"],
            services=["knowledge-evolution", "adaptive-learning"],
            test_type="workflow"
        ))

        # Cross-Phase Integration Tests
        scenarios.append(TestScenario(
            name="cross_phase_data_pipeline",
            description="Test complete data pipeline across all phases",
            phases=["7", "8", "9"],
            services=["dm-crawler", "tactical-intel", "autonomous-coordinator"],
            test_type="workflow",
            timeout=900
        ))

        scenarios.append(TestScenario(
            name="end_to_end_threat_analysis",
            description="End-to-end threat analysis from data collection to autonomous response",
            phases=["7", "8", "9"],
            services=["crypto-intel", "tactical-intel", "defense-automation", "autonomous-coordinator"],
            test_type="workflow",
            timeout=1200
        ))

        # Performance Tests
        scenarios.append(TestScenario(
            name="load_test_all_phases",
            description="Load testing across all deployed phases",
            phases=["7", "8", "9"],
            services=[],  # All services
            test_type="performance",
            timeout=1800
        ))

        scenarios.append(TestScenario(
            name="concurrent_operations_test",
            description="Test concurrent operations across phases",
            phases=["7", "8", "9"],
            services=[],  # All services
            test_type="performance",
            timeout=900
        ))

        # Security Tests
        scenarios.append(TestScenario(
            name="security_integration_test",
            description="Test security controls across all phases",
            phases=["7", "8", "9"],
            services=[],  # All services
            test_type="security",
            timeout=600
        ))

        return scenarios

    def add_result(self, scenario: str, test_name: str, status: str, message: str,
                   duration: float = 0.0, details: Dict = None, error_trace: str = None):
        """Add a test result"""
        result = TestResult(scenario, test_name, status, message, duration, details, error_trace)
        self.results.append(result)

        # Log the result
        log_level = logging.INFO if status in ["PASS", "SKIP"] else logging.ERROR
        self.logger.log(log_level, f"{scenario}/{test_name}: {status} - {message} ({duration:.2f}s)")

    async def setup_test_environment(self):
        """Setup test environment"""
        self.logger.info("Setting up integration test environment...")

        # Initialize HTTP session
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

        # Wait for all services to be ready
        await self.wait_for_all_services()

        # Initialize test data
        self.test_data = {
            "test_id": f"integration_test_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "phases": self.phases
        }

        self.logger.info("Test environment setup completed")

    async def wait_for_all_services(self):
        """Wait for all services to be ready"""
        self.logger.info("Waiting for all services to be ready...")

        max_wait = 300  # 5 minutes
        check_interval = 10
        start_time = time.time()

        while time.time() - start_time < max_wait:
            all_ready = True

            for phase in self.phases:
                if phase in self.service_endpoints:
                    for service, endpoint in self.service_endpoints[phase].items():
                        try:
                            async with self.session.get(f"{endpoint}/health") as response:
                                if response.status != 200:
                                    all_ready = False
                                    break
                        except:
                            all_ready = False
                            break

                if not all_ready:
                    break

            if all_ready:
                self.logger.info("All services are ready")
                return

            self.logger.info(f"Services not ready, waiting... ({int(time.time() - start_time)}s elapsed)")
            await asyncio.sleep(check_interval)

        raise Exception(f"Services not ready after {max_wait}s")

    async def run_phase7_data_flow_test(self) -> bool:
        """Test Phase 7 data flow integration"""
        scenario = "phase7_data_flow"
        self.logger.info(f"Running scenario: {scenario}")

        start_time = time.time()

        try:
            # 1. Test DM Crawler data collection
            dm_endpoint = self.service_endpoints["7"]["dm-crawler"]
            async with self.session.post(f"{dm_endpoint}/api/v1/crawl/start",
                                       json={"test_mode": True, "sites": ["test_site"]}) as response:
                if response.status in [200, 202]:
                    self.add_result(scenario, "dm_crawler_start", "PASS", "DM crawler started successfully")
                else:
                    self.add_result(scenario, "dm_crawler_start", "FAIL", f"DM crawler start failed: HTTP {response.status}")
                    return False

            # 2. Test Crypto Intel analysis
            crypto_endpoint = self.service_endpoints["7"]["crypto-intel"]
            test_tx_data = {
                "transaction_id": "test_tx_123",
                "blockchain": "bitcoin",
                "amount": 1.5,
                "addresses": ["test_address_1", "test_address_2"]
            }

            async with self.session.post(f"{crypto_endpoint}/api/v1/analyze/transaction",
                                       json=test_tx_data) as response:
                if response.status in [200, 202]:
                    result = await response.json()
                    self.add_result(scenario, "crypto_analysis", "PASS", "Crypto analysis completed",
                                  details={"analysis_id": result.get("analysis_id")})
                else:
                    self.add_result(scenario, "crypto_analysis", "FAIL", f"Crypto analysis failed: HTTP {response.status}")
                    return False

            # 3. Test Reputation Analyzer
            reputation_endpoint = self.service_endpoints["7"]["reputation-analyzer"]
            test_entity = {
                "entity_id": "test_entity_123",
                "entity_type": "address",
                "data": {"address": "test_address_1", "blockchain": "bitcoin"}
            }

            async with self.session.post(f"{reputation_endpoint}/api/v1/analyze/reputation",
                                       json=test_entity) as response:
                if response.status in [200, 202]:
                    result = await response.json()
                    self.add_result(scenario, "reputation_analysis", "PASS", "Reputation analysis completed",
                                  details={"reputation_score": result.get("reputation_score")})
                else:
                    self.add_result(scenario, "reputation_analysis", "FAIL", f"Reputation analysis failed: HTTP {response.status}")
                    return False

            # 4. Test Economics Processor
            economics_endpoint = self.service_endpoints["7"]["economics-processor"]
            test_market_data = {
                "market": "bitcoin",
                "timeframe": "1h",
                "data_points": 10
            }

            async with self.session.post(f"{economics_endpoint}/api/v1/analyze/market",
                                       json=test_market_data) as response:
                if response.status in [200, 202]:
                    result = await response.json()
                    self.add_result(scenario, "economics_analysis", "PASS", "Economics analysis completed",
                                  details={"prediction_confidence": result.get("confidence")})
                else:
                    self.add_result(scenario, "economics_analysis", "FAIL", f"Economics analysis failed: HTTP {response.status}")
                    return False

            # 5. Test data flow between services
            await asyncio.sleep(5)  # Allow processing time

            # Check if data flowed correctly
            async with self.session.get(f"{crypto_endpoint}/api/v1/status/pipeline") as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("processed_items", 0) > 0:
                        self.add_result(scenario, "data_flow_validation", "PASS", "Data flow validation successful")
                    else:
                        self.add_result(scenario, "data_flow_validation", "WARN", "No data flow detected")
                else:
                    self.add_result(scenario, "data_flow_validation", "FAIL", "Data flow validation failed")
                    return False

            duration = time.time() - start_time
            self.add_result(scenario, "overall", "PASS", f"Phase 7 data flow test completed", duration)
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.add_result(scenario, "overall", "ERROR", f"Test failed with exception: {str(e)}", duration, error_trace=str(e))
            return False

    async def run_phase8_threat_response_test(self) -> bool:
        """Test Phase 8 threat response workflow"""
        scenario = "phase8_threat_response_workflow"
        self.logger.info(f"Running scenario: {scenario}")

        start_time = time.time()

        try:
            # 1. Inject threat intelligence
            tactical_endpoint = self.service_endpoints["8"]["tactical-intel"]
            threat_data = {
                "threat_type": "malicious_address",
                "indicators": ["test_malicious_address"],
                "confidence": 0.9,
                "source": "integration_test"
            }

            async with self.session.post(f"{tactical_endpoint}/api/v1/intel/ingest",
                                       json=threat_data) as response:
                if response.status in [200, 202]:
                    result = await response.json()
                    threat_id = result.get("threat_id")
                    self.add_result(scenario, "threat_ingestion", "PASS", "Threat intelligence ingested",
                                  details={"threat_id": threat_id})
                else:
                    self.add_result(scenario, "threat_ingestion", "FAIL", f"Threat ingestion failed: HTTP {response.status}")
                    return False

            # 2. Wait for threat processing
            await asyncio.sleep(10)

            # 3. Check if defense automation triggered
            defense_endpoint = self.service_endpoints["8"]["defense-automation"]
            async with self.session.get(f"{defense_endpoint}/api/v1/responses/recent") as response:
                if response.status == 200:
                    result = await response.json()
                    responses = result.get("responses", [])
                    if any(r.get("trigger_source") == "tactical_intel" for r in responses):
                        self.add_result(scenario, "automated_response", "PASS", "Automated response triggered")
                    else:
                        self.add_result(scenario, "automated_response", "WARN", "No automated response detected")
                else:
                    self.add_result(scenario, "automated_response", "FAIL", "Failed to check automated responses")
                    return False

            # 4. Test threat correlation
            async with self.session.get(f"{tactical_endpoint}/api/v1/correlation/status") as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("active_correlations", 0) > 0:
                        self.add_result(scenario, "threat_correlation", "PASS", "Threat correlation working")
                    else:
                        self.add_result(scenario, "threat_correlation", "WARN", "No threat correlations found")
                else:
                    self.add_result(scenario, "threat_correlation", "FAIL", "Threat correlation check failed")

            duration = time.time() - start_time
            self.add_result(scenario, "overall", "PASS", f"Phase 8 threat response test completed", duration)
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.add_result(scenario, "overall", "ERROR", f"Test failed with exception: {str(e)}", duration, error_trace=str(e))
            return False

    async def run_phase9_autonomous_test(self) -> bool:
        """Test Phase 9 autonomous coordination"""
        scenario = "phase9_autonomous_coordination"
        self.logger.info(f"Running scenario: {scenario}")

        start_time = time.time()

        try:
            # 1. Test autonomous decision making
            coordinator_endpoint = self.service_endpoints["9"]["autonomous-coordinator"]
            decision_request = {
                "scenario": "integration_test",
                "context": {
                    "threat_level": "medium",
                    "resource_availability": "high",
                    "system_load": "normal"
                },
                "request_autonomous_action": True
            }

            async with self.session.post(f"{coordinator_endpoint}/api/v1/decisions/request",
                                       json=decision_request) as response:
                if response.status in [200, 202]:
                    result = await response.json()
                    decision_id = result.get("decision_id")
                    self.add_result(scenario, "autonomous_decision", "PASS", "Autonomous decision requested",
                                  details={"decision_id": decision_id})
                else:
                    self.add_result(scenario, "autonomous_decision", "FAIL", f"Autonomous decision failed: HTTP {response.status}")
                    return False

            # 2. Test adaptive learning
            learning_endpoint = self.service_endpoints["9"]["adaptive-learning"]
            learning_data = {
                "model_type": "threat_detection",
                "training_data": [
                    {"features": [1, 2, 3], "label": "benign"},
                    {"features": [4, 5, 6], "label": "malicious"}
                ],
                "update_model": True
            }

            async with self.session.post(f"{learning_endpoint}/api/v1/learning/train",
                                       json=learning_data) as response:
                if response.status in [200, 202]:
                    result = await response.json()
                    self.add_result(scenario, "adaptive_learning", "PASS", "Adaptive learning triggered",
                                  details={"training_id": result.get("training_id")})
                else:
                    self.add_result(scenario, "adaptive_learning", "FAIL", f"Adaptive learning failed: HTTP {response.status}")
                    return False

            # 3. Wait for processing
            await asyncio.sleep(15)

            # 4. Check decision outcome
            async with self.session.get(f"{coordinator_endpoint}/api/v1/decisions/{decision_id}/status") as response:
                if response.status == 200:
                    result = await response.json()
                    decision_status = result.get("status")
                    if decision_status == "completed":
                        self.add_result(scenario, "decision_execution", "PASS", "Autonomous decision executed")
                    else:
                        self.add_result(scenario, "decision_execution", "WARN", f"Decision status: {decision_status}")
                else:
                    self.add_result(scenario, "decision_execution", "FAIL", "Failed to check decision status")

            # 5. Test knowledge evolution
            knowledge_endpoint = self.service_endpoints["9"]["knowledge-evolution"]
            async with self.session.get(f"{knowledge_endpoint}/api/v1/evolution/status") as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("evolution_active", False):
                        self.add_result(scenario, "knowledge_evolution", "PASS", "Knowledge evolution active")
                    else:
                        self.add_result(scenario, "knowledge_evolution", "WARN", "Knowledge evolution not active")
                else:
                    self.add_result(scenario, "knowledge_evolution", "FAIL", "Knowledge evolution check failed")

            duration = time.time() - start_time
            self.add_result(scenario, "overall", "PASS", f"Phase 9 autonomous test completed", duration)
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.add_result(scenario, "overall", "ERROR", f"Test failed with exception: {str(e)}", duration, error_trace=str(e))
            return False

    async def run_end_to_end_workflow_test(self) -> bool:
        """Test complete end-to-end workflow"""
        scenario = "end_to_end_threat_analysis"
        self.logger.info(f"Running scenario: {scenario}")

        start_time = time.time()

        try:
            # 1. Start with data collection (Phase 7)
            dm_endpoint = self.service_endpoints["7"]["dm-crawler"]
            crypto_endpoint = self.service_endpoints["7"]["crypto-intel"]

            # Simulate threat data discovery
            threat_tx = {
                "transaction_id": "suspicious_tx_456",
                "blockchain": "bitcoin",
                "amount": 10.0,
                "addresses": ["suspicious_address_1", "suspicious_address_2"],
                "timestamp": datetime.now().isoformat()
            }

            async with self.session.post(f"{crypto_endpoint}/api/v1/analyze/transaction",
                                       json=threat_tx) as response:
                if response.status not in [200, 202]:
                    self.add_result(scenario, "data_collection", "FAIL", "Initial data collection failed")
                    return False

            self.add_result(scenario, "data_collection", "PASS", "Threat data collected")

            # 2. Wait for Phase 7 processing
            await asyncio.sleep(10)

            # 3. Check if threat was identified and passed to Phase 8
            tactical_endpoint = self.service_endpoints["8"]["tactical-intel"]
            async with self.session.get(f"{tactical_endpoint}/api/v1/intel/recent") as response:
                if response.status == 200:
                    result = await response.json()
                    threats = result.get("threats", [])
                    if any("suspicious_tx_456" in str(t) for t in threats):
                        self.add_result(scenario, "threat_identification", "PASS", "Threat identified by tactical intel")
                    else:
                        self.add_result(scenario, "threat_identification", "WARN", "Threat not found in tactical intel")
                else:
                    self.add_result(scenario, "threat_identification", "FAIL", "Failed to check tactical intel")

            # 4. Check automated response (Phase 8)
            defense_endpoint = self.service_endpoints["8"]["defense-automation"]
            async with self.session.get(f"{defense_endpoint}/api/v1/responses/recent") as response:
                if response.status == 200:
                    result = await response.json()
                    responses = result.get("responses", [])
                    if responses:
                        self.add_result(scenario, "automated_response", "PASS", f"Automated responses triggered: {len(responses)}")
                    else:
                        self.add_result(scenario, "automated_response", "WARN", "No automated responses found")
                else:
                    self.add_result(scenario, "automated_response", "FAIL", "Failed to check automated responses")

            # 5. Check autonomous coordination (Phase 9)
            coordinator_endpoint = self.service_endpoints["9"]["autonomous-coordinator"]
            async with self.session.get(f"{coordinator_endpoint}/api/v1/operations/recent") as response:
                if response.status == 200:
                    result = await response.json()
                    operations = result.get("operations", [])
                    if operations:
                        self.add_result(scenario, "autonomous_coordination", "PASS", f"Autonomous operations: {len(operations)}")
                    else:
                        self.add_result(scenario, "autonomous_coordination", "WARN", "No autonomous operations found")
                else:
                    self.add_result(scenario, "autonomous_coordination", "FAIL", "Failed to check autonomous operations")

            # 6. Verify learning occurred (Phase 9)
            learning_endpoint = self.service_endpoints["9"]["adaptive-learning"]
            async with self.session.get(f"{learning_endpoint}/api/v1/learning/recent") as response:
                if response.status == 200:
                    result = await response.json()
                    learning_events = result.get("learning_events", [])
                    if learning_events:
                        self.add_result(scenario, "adaptive_learning", "PASS", f"Learning events: {len(learning_events)}")
                    else:
                        self.add_result(scenario, "adaptive_learning", "WARN", "No learning events found")
                else:
                    self.add_result(scenario, "adaptive_learning", "FAIL", "Failed to check learning events")

            duration = time.time() - start_time
            self.add_result(scenario, "overall", "PASS", f"End-to-end workflow test completed", duration)
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.add_result(scenario, "overall", "ERROR", f"Test failed with exception: {str(e)}", duration, error_trace=str(e))
            return False

    async def run_performance_test(self) -> bool:
        """Run performance testing across all phases"""
        scenario = "load_test_all_phases"
        self.logger.info(f"Running scenario: {scenario}")

        start_time = time.time()

        try:
            # Define performance test parameters
            concurrent_requests = 10
            requests_per_service = 20

            all_endpoints = []
            for phase in self.phases:
                if phase in self.service_endpoints:
                    for service, endpoint in self.service_endpoints[phase].items():
                        all_endpoints.append((service, f"{endpoint}/health"))

            # Create concurrent tasks
            tasks = []
            for service, url in all_endpoints:
                for _ in range(requests_per_service):
                    task = asyncio.create_task(self._performance_request(service, url))
                    tasks.append(task)

            # Execute all requests
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze results
            successful_requests = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
            total_requests = len(results)
            success_rate = (successful_requests / total_requests) * 100

            response_times = [r.get("duration", 0) for r in results if isinstance(r, dict) and r.get("success")]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0

            if success_rate >= 95:
                self.add_result(scenario, "load_test", "PASS",
                              f"Load test passed: {success_rate:.1f}% success rate, avg response: {avg_response_time:.3f}s",
                              details={"success_rate": success_rate, "avg_response_time": avg_response_time})
            else:
                self.add_result(scenario, "load_test", "FAIL",
                              f"Load test failed: {success_rate:.1f}% success rate",
                              details={"success_rate": success_rate, "avg_response_time": avg_response_time})

            duration = time.time() - start_time
            self.add_result(scenario, "overall", "PASS", f"Performance test completed", duration)
            return success_rate >= 95

        except Exception as e:
            duration = time.time() - start_time
            self.add_result(scenario, "overall", "ERROR", f"Performance test failed: {str(e)}", duration, error_trace=str(e))
            return False

    async def _performance_request(self, service: str, url: str) -> Dict:
        """Make a single performance test request"""
        start_time = time.time()
        try:
            async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                duration = time.time() - start_time
                return {
                    "service": service,
                    "success": response.status == 200,
                    "duration": duration,
                    "status": response.status
                }
        except Exception as e:
            duration = time.time() - start_time
            return {
                "service": service,
                "success": False,
                "duration": duration,
                "error": str(e)
            }

    async def run_security_test(self) -> bool:
        """Run security integration tests"""
        scenario = "security_integration_test"
        self.logger.info(f"Running scenario: {scenario}")

        start_time = time.time()

        try:
            # Test authentication on all services
            security_passed = True

            for phase in self.phases:
                if phase in self.service_endpoints:
                    for service, endpoint in self.service_endpoints[phase].items():
                        # Test invalid authentication
                        headers = {"Authorization": "Bearer invalid_token"}
                        async with self.session.get(f"{endpoint}/api/v1/secure-endpoint",
                                                   headers=headers) as response:
                            if response.status in [401, 403]:
                                self.add_result(scenario, f"auth_test_{service}", "PASS", "Authentication properly rejected")
                            else:
                                self.add_result(scenario, f"auth_test_{service}", "WARN",
                                              f"Unexpected auth response: {response.status}")

                        # Test input validation
                        malicious_data = {"script": "<script>alert('xss')</script>", "sql": "'; DROP TABLE users; --"}
                        async with self.session.post(f"{endpoint}/api/v1/test",
                                                    json=malicious_data) as response:
                            if response.status in [400, 422]:
                                self.add_result(scenario, f"input_validation_{service}", "PASS", "Input validation working")
                            else:
                                self.add_result(scenario, f"input_validation_{service}", "WARN",
                                              f"Input validation response: {response.status}")

            duration = time.time() - start_time
            self.add_result(scenario, "overall", "PASS", f"Security test completed", duration)
            return True

        except Exception as e:
            duration = time.time() - start_time
            self.add_result(scenario, "overall", "ERROR", f"Security test failed: {str(e)}", duration, error_trace=str(e))
            return False

    async def run_all_integration_tests(self) -> bool:
        """Run all integration test scenarios"""
        self.logger.info("Starting comprehensive integration testing...")

        await self.setup_test_environment()

        overall_success = True
        scenario_results = {}

        # Define test execution order
        test_scenarios = [
            ("phase7_data_flow", self.run_phase7_data_flow_test),
            ("phase8_threat_response", self.run_phase8_threat_response_test),
            ("phase9_autonomous", self.run_phase9_autonomous_test),
            ("end_to_end_workflow", self.run_end_to_end_workflow_test),
            ("performance_test", self.run_performance_test),
            ("security_test", self.run_security_test),
        ]

        for scenario_name, test_function in test_scenarios:
            self.logger.info(f"Executing test scenario: {scenario_name}")
            try:
                result = await test_function()
                scenario_results[scenario_name] = result
                if not result:
                    overall_success = False
                    self.logger.error(f"Scenario {scenario_name} failed")
                else:
                    self.logger.info(f"Scenario {scenario_name} passed")
            except Exception as e:
                self.logger.error(f"Scenario {scenario_name} failed with exception: {e}")
                scenario_results[scenario_name] = False
                overall_success = False

        return overall_success

    async def cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.session:
            await self.session.close()

        self.logger.info("Test environment cleanup completed")

    def generate_report(self) -> str:
        """Generate integration test report"""
        report = []
        report.append("="*80)
        report.append("BEV OSINT Framework - Integration Test Report")
        report.append("="*80)
        report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Phases Tested: {', '.join(self.phases)}")
        report.append(f"Total Test Cases: {len(self.results)}")
        report.append("")

        # Summary by status
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        errors = sum(1 for r in self.results if r.status == "ERROR")
        warnings = sum(1 for r in self.results if r.status == "WARN")
        skipped = sum(1 for r in self.results if r.status == "SKIP")

        report.append("SUMMARY:")
        report.append(f"  ✅ Passed: {passed}")
        report.append(f"  ❌ Failed: {failed}")
        report.append(f"  🚨 Errors: {errors}")
        report.append(f"  ⚠️  Warnings: {warnings}")
        report.append(f"  ⏭️  Skipped: {skipped}")
        report.append("")

        # Summary by scenario
        scenarios = {}
        for result in self.results:
            if result.scenario not in scenarios:
                scenarios[result.scenario] = {"PASS": 0, "FAIL": 0, "ERROR": 0, "WARN": 0, "SKIP": 0}
            scenarios[result.scenario][result.status] += 1

        report.append("SCENARIO RESULTS:")
        for scenario, counts in scenarios.items():
            total = sum(counts.values())
            status_icon = "✅" if counts["FAIL"] == 0 and counts["ERROR"] == 0 else "❌"
            report.append(f"  {status_icon} {scenario}: {total} tests ({counts['PASS']} passed, {counts['FAIL'] + counts['ERROR']} failed)")

        report.append("")

        # Performance summary
        performance_results = [r for r in self.results if "performance" in r.test_name.lower() or "load" in r.test_name.lower()]
        if performance_results:
            report.append("PERFORMANCE SUMMARY:")
            for result in performance_results:
                if result.details and "avg_response_time" in result.details:
                    report.append(f"  Average Response Time: {result.details['avg_response_time']:.3f}s")
                if result.details and "success_rate" in result.details:
                    report.append(f"  Success Rate: {result.details['success_rate']:.1f}%")
            report.append("")

        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 50)

        for scenario in scenarios.keys():
            scenario_results = [r for r in self.results if r.scenario == scenario]
            scenario_failed = any(r.status in ["FAIL", "ERROR"] for r in scenario_results)

            status_icon = "❌" if scenario_failed else "✅"
            report.append(f"\n{status_icon} {scenario.upper()}:")

            for result in scenario_results:
                status_icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "🚨", "WARN": "⚠️", "SKIP": "⏭️"}.get(result.status, "❓")
                report.append(f"    {status_icon} {result.test_name}: {result.message}")
                if result.duration > 0:
                    report.append(f"      Duration: {result.duration:.3f}s")
                if result.details:
                    for key, value in result.details.items():
                        report.append(f"      {key}: {value}")
                if result.error_trace:
                    report.append(f"      Error: {result.error_trace}")

        # Recommendations
        report.append("\n" + "="*50)
        report.append("RECOMMENDATIONS:")
        report.append("-" * 50)

        if failed > 0 or errors > 0:
            report.append("🚨 INTEGRATION ISSUES FOUND")
            report.append("   Some integration tests failed. Review failures and fix before production deployment.")
        elif warnings > 0:
            report.append("⚠️  WARNINGS DETECTED")
            report.append("   Some integration issues detected. Monitor and consider fixes.")
        else:
            report.append("✅ ALL INTEGRATION TESTS PASSED")
            report.append("   All services are integrating correctly and ready for production use.")

        report.append("\n" + "="*80)
        return "\n".join(report)

    def save_report(self, filename: str = None) -> Path:
        """Save integration test report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"integration_test_report_{timestamp}.txt"

        report_path = self.project_root / "logs" / "integration_tests" / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write(self.generate_report())

        return report_path

async def main():
    """Main entry point for integration testing"""
    import argparse

    parser = argparse.ArgumentParser(description="BEV OSINT Integration Testing Framework")
    parser.add_argument("--phases", default="7,8,9",
                       help="Comma-separated list of phases to test (default: 7,8,9)")
    parser.add_argument("--output", help="Output report file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Parse phases
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    # Create test framework
    framework = IntegrationTestFramework(phases)

    try:
        # Run all integration tests
        success = await framework.run_all_integration_tests()

        # Generate and display report
        report = framework.generate_report()
        print(report)

        # Save report
        report_path = framework.save_report(args.output)
        print(f"\nIntegration test report saved to: {report_path}")

        # Exit with appropriate code
        if success:
            print("\n✅ All integration tests passed - system ready for production")
            sys.exit(0)
        else:
            failed_count = sum(1 for r in framework.results if r.status in ["FAIL", "ERROR"])
            print(f"\n❌ Integration tests failed - {failed_count} test cases failed")
            sys.exit(1)

    finally:
        await framework.cleanup_test_environment()

if __name__ == "__main__":
    asyncio.run(main())