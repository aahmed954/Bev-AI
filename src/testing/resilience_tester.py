"""
Resilience Testing Framework for BEV OSINT Framework
===================================================

Comprehensive resilience testing with automated experiment orchestration,
recovery validation, and performance analysis system.

Features:
- Automated resilience experiment orchestration
- Multi-dimensional resilience metrics collection
- Recovery time and performance impact analysis
- Integration with chaos engineering and monitoring systems
- Comprehensive test scenario execution and validation

Author: BEV Infrastructure Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
import statistics
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
import aiohttp
import aioredis
import yaml
import numpy as np
from scipy import stats


class ResilienceMetric(Enum):
    """Types of resilience metrics to measure."""
    AVAILABILITY = "availability"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    RECOVERY_TIME = "recovery_time"
    MTTR = "mean_time_to_recovery"
    MTBF = "mean_time_between_failures"
    CASCADING_FAILURES = "cascading_failures"
    DATA_CONSISTENCY = "data_consistency"
    RESOURCE_UTILIZATION = "resource_utilization"


class TestPhase(Enum):
    """Phases of resilience testing."""
    PREPARATION = "preparation"
    BASELINE = "baseline"
    STRESS = "stress"
    FAILURE = "failure"
    RECOVERY = "recovery"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    COMPLETED = "completed"
    FAILED = "failed"


class ResilienceLevel(Enum):
    """Levels of system resilience."""
    FRAGILE = "fragile"      # Fails easily, slow recovery
    ROBUST = "robust"        # Withstands stress, normal recovery
    ANTIFRAGILE = "antifragile"  # Improves under stress


@dataclass
class ResilienceTestConfig:
    """Configuration for a resilience test."""
    name: str
    description: str
    target_services: List[str]
    test_duration: float = 300.0  # 5 minutes default

    # Test phases configuration
    baseline_duration: float = 60.0
    stress_duration: float = 120.0
    failure_duration: float = 60.0
    recovery_timeout: float = 180.0

    # Metrics to collect
    metrics_to_collect: List[ResilienceMetric] = field(default_factory=lambda: [
        ResilienceMetric.AVAILABILITY,
        ResilienceMetric.RESPONSE_TIME,
        ResilienceMetric.ERROR_RATE,
        ResilienceMetric.RECOVERY_TIME
    ])

    # Failure scenarios
    failure_scenarios: List[Dict[str, Any]] = field(default_factory=list)

    # Success criteria
    availability_threshold: float = 99.0  # Minimum availability %
    recovery_time_threshold: float = 60.0  # Maximum recovery time in seconds
    performance_degradation_threshold: float = 0.2  # 20% max degradation

    # Load testing configuration
    load_test_enabled: bool = True
    concurrent_users: int = 100
    requests_per_second: int = 50

    # Metadata
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: Optional[datetime] = None


@dataclass
class MetricSample:
    """A single metric measurement."""
    metric_type: ResilienceMetric
    service_name: str
    value: float
    timestamp: datetime
    phase: TestPhase
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseMetrics:
    """Metrics collected during a test phase."""
    phase: TestPhase
    start_time: datetime
    end_time: Optional[datetime]
    duration: float = 0.0

    # Collected samples
    samples: List[MetricSample] = field(default_factory=list)

    # Aggregated statistics
    availability: float = 0.0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0

    # Resource utilization
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    peak_cpu_usage: float = 0.0
    peak_memory_usage: float = 0.0


@dataclass
class ResilienceTestResult:
    """Results of a complete resilience test."""
    test_name: str
    start_time: datetime
    end_time: Optional[datetime]
    total_duration: float = 0.0
    success: bool = False

    # Phase results
    phase_metrics: Dict[TestPhase, PhaseMetrics] = field(default_factory=dict)

    # Overall metrics
    overall_availability: float = 0.0
    recovery_time: float = 0.0
    resilience_score: float = 0.0
    resilience_level: ResilienceLevel = ResilienceLevel.FRAGILE

    # Performance analysis
    baseline_performance: Dict[str, float] = field(default_factory=dict)
    degraded_performance: Dict[str, float] = field(default_factory=dict)
    performance_impact: float = 0.0

    # Failure analysis
    failures_detected: List[Dict[str, Any]] = field(default_factory=list)
    cascading_failures: List[Dict[str, Any]] = field(default_factory=list)
    recovery_events: List[Dict[str, Any]] = field(default_factory=list)

    # Insights and recommendations
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    # Raw data
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects various resilience metrics from services."""

    def __init__(self, health_monitor_url: str = "http://172.30.0.38:8080"):
        self.health_monitor_url = health_monitor_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger("metrics_collector")

    async def initialize(self):
        """Initialize the metrics collector."""
        self.session = aiohttp.ClientSession()

    async def shutdown(self):
        """Shutdown the metrics collector."""
        if self.session:
            await self.session.close()

    async def collect_availability(self, service_name: str) -> float:
        """Collect availability metric for a service."""
        try:
            async with self.session.get(f"{self.health_monitor_url}/health/{service_name}") as response:
                if response.status == 200:
                    health_data = await response.json()
                    return 100.0 if health_data.get('status') == 'healthy' else 0.0
                return 0.0
        except Exception:
            return 0.0

    async def collect_response_time(self, service_name: str, endpoint: str = "/health") -> float:
        """Collect response time metric for a service."""
        try:
            start_time = time.time()
            async with self.session.get(f"http://{service_name}:8080{endpoint}", timeout=aiohttp.ClientTimeout(total=10)) as response:
                end_time = time.time()
                if response.status == 200:
                    return (end_time - start_time) * 1000  # Convert to milliseconds
                return float('inf')  # Infinite response time for errors
        except Exception:
            return float('inf')

    async def collect_error_rate(self, service_name: str, sample_requests: int = 10) -> float:
        """Collect error rate by making sample requests."""
        errors = 0
        total = sample_requests

        for _ in range(sample_requests):
            try:
                async with self.session.get(f"http://{service_name}:8080/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status >= 400:
                        errors += 1
            except Exception:
                errors += 1

        return (errors / total) * 100.0

    async def collect_throughput(self, service_name: str, duration: float = 10.0) -> float:
        """Collect throughput metric by measuring requests per second."""
        start_time = time.time()
        successful_requests = 0

        while time.time() - start_time < duration:
            try:
                async with self.session.get(f"http://{service_name}:8080/health", timeout=aiohttp.ClientTimeout(total=1)) as response:
                    if response.status == 200:
                        successful_requests += 1
            except Exception:
                pass

            # Small delay to avoid overwhelming
            await asyncio.sleep(0.1)

        actual_duration = time.time() - start_time
        return successful_requests / actual_duration

    async def collect_resource_utilization(self, service_name: str) -> Dict[str, float]:
        """Collect resource utilization metrics."""
        try:
            async with self.session.get(f"{self.health_monitor_url}/metrics/{service_name}") as response:
                if response.status == 200:
                    metrics = await response.json()
                    return {
                        'cpu_usage': metrics.get('cpu_usage_percent', 0.0),
                        'memory_usage': metrics.get('memory_usage_percent', 0.0),
                        'disk_usage': metrics.get('disk_usage_percent', 0.0),
                        'network_io': metrics.get('network_io_mbps', 0.0)
                    }
        except Exception:
            pass

        return {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_io': 0.0
        }

    async def collect_metric(self, metric_type: ResilienceMetric, service_name: str) -> float:
        """Collect a specific metric type."""
        if metric_type == ResilienceMetric.AVAILABILITY:
            return await self.collect_availability(service_name)
        elif metric_type == ResilienceMetric.RESPONSE_TIME:
            return await self.collect_response_time(service_name)
        elif metric_type == ResilienceMetric.ERROR_RATE:
            return await self.collect_error_rate(service_name)
        elif metric_type == ResilienceMetric.THROUGHPUT:
            return await self.collect_throughput(service_name)
        else:
            return 0.0


class LoadGenerator:
    """Generates synthetic load for resilience testing."""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_load_tasks: List[asyncio.Task] = []
        self.logger = logging.getLogger("load_generator")

    async def initialize(self):
        """Initialize the load generator."""
        self.session = aiohttp.ClientSession()

    async def shutdown(self):
        """Shutdown the load generator."""
        await self.stop_all_load()
        if self.session:
            await self.session.close()

    async def generate_load(self, service_name: str, requests_per_second: int,
                          duration: float, endpoint: str = "/health") -> Dict[str, Any]:
        """Generate load against a service."""
        start_time = time.time()
        request_count = 0
        success_count = 0
        error_count = 0
        response_times = []

        interval = 1.0 / requests_per_second
        end_time = start_time + duration

        while time.time() < end_time:
            request_start = time.time()

            try:
                async with self.session.get(f"http://{service_name}:8080{endpoint}") as response:
                    request_time = (time.time() - request_start) * 1000
                    response_times.append(request_time)
                    request_count += 1

                    if response.status == 200:
                        success_count += 1
                    else:
                        error_count += 1

            except Exception:
                error_count += 1
                request_count += 1

            # Wait for next request interval
            sleep_time = max(0, interval - (time.time() - request_start))
            await asyncio.sleep(sleep_time)

        return {
            'duration': time.time() - start_time,
            'total_requests': request_count,
            'successful_requests': success_count,
            'failed_requests': error_count,
            'requests_per_second': request_count / (time.time() - start_time),
            'avg_response_time': statistics.mean(response_times) if response_times else 0,
            'p95_response_time': np.percentile(response_times, 95) if response_times else 0,
            'p99_response_time': np.percentile(response_times, 99) if response_times else 0,
            'error_rate': (error_count / request_count * 100) if request_count > 0 else 0
        }

    async def start_background_load(self, service_name: str, requests_per_second: int,
                                  endpoint: str = "/health") -> str:
        """Start background load generation."""
        load_id = f"load_{service_name}_{int(time.time())}"

        async def load_task():
            while True:
                try:
                    await self.generate_load(service_name, requests_per_second, 60.0, endpoint)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Load generation error: {e}")
                    await asyncio.sleep(5)

        task = asyncio.create_task(load_task())
        self.active_load_tasks.append(task)
        self.logger.info(f"Started background load for {service_name}: {load_id}")

        return load_id

    async def stop_all_load(self):
        """Stop all active load generation."""
        for task in self.active_load_tasks:
            task.cancel()

        if self.active_load_tasks:
            await asyncio.gather(*self.active_load_tasks, return_exceptions=True)

        self.active_load_tasks.clear()
        self.logger.info("Stopped all load generation")


class FailureDetector:
    """Detects and analyzes failures during resilience testing."""

    def __init__(self, auto_recovery_url: str = "http://172.30.0.41:8080"):
        self.auto_recovery_url = auto_recovery_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.detected_failures: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("failure_detector")

    async def initialize(self):
        """Initialize the failure detector."""
        self.session = aiohttp.ClientSession()

    async def shutdown(self):
        """Shutdown the failure detector."""
        if self.session:
            await self.session.close()

    async def detect_service_failure(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Detect if a service has failed."""
        try:
            # Check service health
            async with self.session.get(f"http://{service_name}:8080/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status != 200:
                    failure = {
                        'service': service_name,
                        'type': 'service_unavailable',
                        'timestamp': datetime.utcnow(),
                        'details': f'HTTP {response.status}'
                    }
                    self.detected_failures.append(failure)
                    return failure

        except asyncio.TimeoutError:
            failure = {
                'service': service_name,
                'type': 'timeout',
                'timestamp': datetime.utcnow(),
                'details': 'Health check timeout'
            }
            self.detected_failures.append(failure)
            return failure

        except Exception as e:
            failure = {
                'service': service_name,
                'type': 'connection_error',
                'timestamp': datetime.utcnow(),
                'details': str(e)
            }
            self.detected_failures.append(failure)
            return failure

        return None

    async def detect_cascading_failures(self, services: List[str]) -> List[Dict[str, Any]]:
        """Detect cascading failures across services."""
        cascading = []
        failure_times = {}

        # Check each service for failures
        for service in services:
            failure = await self.detect_service_failure(service)
            if failure:
                failure_times[service] = failure['timestamp']

        # Look for cascading patterns (failures within short time windows)
        if len(failure_times) > 1:
            sorted_failures = sorted(failure_times.items(), key=lambda x: x[1])

            for i in range(1, len(sorted_failures)):
                prev_service, prev_time = sorted_failures[i-1]
                curr_service, curr_time = sorted_failures[i]

                # If failures occurred within 30 seconds, consider it cascading
                if (curr_time - prev_time).total_seconds() < 30:
                    cascading.append({
                        'type': 'cascading_failure',
                        'trigger_service': prev_service,
                        'affected_service': curr_service,
                        'time_delta': (curr_time - prev_time).total_seconds(),
                        'timestamp': curr_time
                    })

        return cascading

    async def monitor_recovery(self, service_name: str, timeout: float = 180.0) -> Dict[str, Any]:
        """Monitor service recovery after failure."""
        start_time = time.time()
        recovery_detected = False
        recovery_time = 0.0

        while time.time() - start_time < timeout:
            try:
                async with self.session.get(f"http://{service_name}:8080/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        recovery_time = time.time() - start_time
                        recovery_detected = True
                        break
            except Exception:
                pass

            await asyncio.sleep(5)

        return {
            'service': service_name,
            'recovery_detected': recovery_detected,
            'recovery_time': recovery_time,
            'timeout': timeout,
            'timestamp': datetime.utcnow()
        }

    def get_failure_summary(self) -> Dict[str, Any]:
        """Get summary of detected failures."""
        return {
            'total_failures': len(self.detected_failures),
            'failure_types': list(set(f['type'] for f in self.detected_failures)),
            'affected_services': list(set(f['service'] for f in self.detected_failures)),
            'failures': self.detected_failures
        }


class ResilienceTester:
    """
    Main resilience testing framework with comprehensive test orchestration.
    """

    def __init__(self,
                 chaos_engineer_url: str = "http://172.30.0.45:8080",
                 auto_recovery_url: str = "http://172.30.0.41:8080",
                 health_monitor_url: str = "http://172.30.0.38:8080",
                 redis_url: str = "redis://redis:6379/13"):
        """
        Initialize the resilience testing system.

        Args:
            chaos_engineer_url: URL for chaos engineering system
            auto_recovery_url: URL for auto-recovery system
            health_monitor_url: URL for health monitoring system
            redis_url: Redis connection URL for state storage
        """
        self.chaos_engineer_url = chaos_engineer_url
        self.auto_recovery_url = auto_recovery_url
        self.health_monitor_url = health_monitor_url
        self.redis_url = redis_url

        # Initialize components
        self.metrics_collector = MetricsCollector(health_monitor_url)
        self.load_generator = LoadGenerator()
        self.failure_detector = FailureDetector(auto_recovery_url)

        # State management
        self.session: Optional[aiohttp.ClientSession] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.active_tests: Dict[str, ResilienceTestResult] = {}
        self.test_history: List[ResilienceTestResult] = []

        # Logging
        self.logger = logging.getLogger("resilience_tester")

    async def initialize(self):
        """Initialize the resilience testing system."""
        self.session = aiohttp.ClientSession()
        self.redis_client = aioredis.from_url(self.redis_url)

        await self.metrics_collector.initialize()
        await self.load_generator.initialize()
        await self.failure_detector.initialize()

        self.logger.info("Resilience testing system initialized")

    async def shutdown(self):
        """Shutdown the resilience testing system."""
        # Stop all active tests
        for test_name in list(self.active_tests.keys()):
            await self.stop_test(test_name)

        # Shutdown components
        await self.metrics_collector.shutdown()
        await self.load_generator.shutdown()
        await self.failure_detector.shutdown()

        # Close connections
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()

        self.logger.info("Resilience testing system shutdown complete")

    async def run_resilience_test(self, config: ResilienceTestConfig) -> ResilienceTestResult:
        """
        Run a complete resilience test with all phases.

        Args:
            config: Test configuration

        Returns:
            ResilienceTestResult: Comprehensive test results
        """
        if config.name in self.active_tests:
            raise RuntimeError(f"Test already running: {config.name}")

        # Create test result
        result = ResilienceTestResult(
            test_name=config.name,
            start_time=datetime.utcnow()
        )

        self.active_tests[config.name] = result

        try:
            self.logger.info(f"Starting resilience test: {config.name}")

            # Phase 1: Preparation
            await self._preparation_phase(config, result)

            # Phase 2: Baseline metrics collection
            await self._baseline_phase(config, result)

            # Phase 3: Stress testing (if load testing enabled)
            if config.load_test_enabled:
                await self._stress_phase(config, result)

            # Phase 4: Failure injection
            await self._failure_phase(config, result)

            # Phase 5: Recovery monitoring
            await self._recovery_phase(config, result)

            # Phase 6: Validation
            await self._validation_phase(config, result)

            # Phase 7: Analysis
            await self._analysis_phase(config, result)

            result.success = True
            result.end_time = datetime.utcnow()
            result.total_duration = (result.end_time - result.start_time).total_seconds()

            self.logger.info(f"Resilience test completed: {config.name}")

        except Exception as e:
            result.success = False
            result.end_time = datetime.utcnow()
            result.total_duration = (result.end_time - result.start_time).total_seconds()
            self.logger.error(f"Resilience test failed: {config.name} - {e}")

        finally:
            # Move to history and cleanup
            self.test_history.append(result)
            self.active_tests.pop(config.name, None)

            # Save results
            await self._save_test_result(result)

        return result

    async def _preparation_phase(self, config: ResilienceTestConfig, result: ResilienceTestResult):
        """Preparation phase - validate environment and services."""
        self.logger.info(f"Starting preparation phase for {config.name}")

        phase_metrics = PhaseMetrics(
            phase=TestPhase.PREPARATION,
            start_time=datetime.utcnow()
        )

        # Validate all target services are healthy
        for service in config.target_services:
            availability = await self.metrics_collector.collect_availability(service)
            if availability < 100.0:
                raise RuntimeError(f"Service {service} is not healthy before test")

        # Validate chaos engineering system is available
        try:
            async with self.session.get(f"{self.chaos_engineer_url}/status") as response:
                if response.status != 200:
                    raise RuntimeError("Chaos engineering system not available")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to chaos engineering system: {e}")

        phase_metrics.end_time = datetime.utcnow()
        phase_metrics.duration = (phase_metrics.end_time - phase_metrics.start_time).total_seconds()
        result.phase_metrics[TestPhase.PREPARATION] = phase_metrics

        self.logger.info(f"Preparation phase completed for {config.name}")

    async def _baseline_phase(self, config: ResilienceTestConfig, result: ResilienceTestResult):
        """Baseline phase - collect normal operation metrics."""
        self.logger.info(f"Starting baseline phase for {config.name}")

        phase_metrics = PhaseMetrics(
            phase=TestPhase.BASELINE,
            start_time=datetime.utcnow()
        )

        # Collect baseline metrics
        start_time = time.time()
        while time.time() - start_time < config.baseline_duration:
            timestamp = datetime.utcnow()

            for service in config.target_services:
                for metric_type in config.metrics_to_collect:
                    value = await self.metrics_collector.collect_metric(metric_type, service)

                    sample = MetricSample(
                        metric_type=metric_type,
                        service_name=service,
                        value=value,
                        timestamp=timestamp,
                        phase=TestPhase.BASELINE
                    )
                    phase_metrics.samples.append(sample)

            await asyncio.sleep(5)  # 5-second sampling interval

        # Calculate baseline statistics
        self._calculate_phase_statistics(phase_metrics)

        # Store baseline performance for comparison
        result.baseline_performance = {
            'availability': phase_metrics.availability,
            'avg_response_time': phase_metrics.avg_response_time,
            'error_rate': phase_metrics.error_rate,
            'throughput': phase_metrics.throughput
        }

        phase_metrics.end_time = datetime.utcnow()
        phase_metrics.duration = (phase_metrics.end_time - phase_metrics.start_time).total_seconds()
        result.phase_metrics[TestPhase.BASELINE] = phase_metrics

        self.logger.info(f"Baseline phase completed for {config.name}")

    async def _stress_phase(self, config: ResilienceTestConfig, result: ResilienceTestResult):
        """Stress phase - apply load to test system capacity."""
        self.logger.info(f"Starting stress phase for {config.name}")

        phase_metrics = PhaseMetrics(
            phase=TestPhase.STRESS,
            start_time=datetime.utcnow()
        )

        # Start load generation for all services
        load_tasks = []
        for service in config.target_services:
            load_id = await self.load_generator.start_background_load(
                service, config.requests_per_second
            )
            load_tasks.append(load_id)

        try:
            # Monitor metrics during stress
            start_time = time.time()
            while time.time() - start_time < config.stress_duration:
                timestamp = datetime.utcnow()

                for service in config.target_services:
                    for metric_type in config.metrics_to_collect:
                        value = await self.metrics_collector.collect_metric(metric_type, service)

                        sample = MetricSample(
                            metric_type=metric_type,
                            service_name=service,
                            value=value,
                            timestamp=timestamp,
                            phase=TestPhase.STRESS
                        )
                        phase_metrics.samples.append(sample)

                await asyncio.sleep(5)

        finally:
            # Stop load generation
            await self.load_generator.stop_all_load()

        # Calculate stress statistics
        self._calculate_phase_statistics(phase_metrics)

        phase_metrics.end_time = datetime.utcnow()
        phase_metrics.duration = (phase_metrics.end_time - phase_metrics.start_time).total_seconds()
        result.phase_metrics[TestPhase.STRESS] = phase_metrics

        self.logger.info(f"Stress phase completed for {config.name}")

    async def _failure_phase(self, config: ResilienceTestConfig, result: ResilienceTestResult):
        """Failure phase - inject failures and monitor impact."""
        self.logger.info(f"Starting failure phase for {config.name}")

        phase_metrics = PhaseMetrics(
            phase=TestPhase.FAILURE,
            start_time=datetime.utcnow()
        )

        injected_faults = []

        try:
            # Inject configured failures
            for scenario in config.failure_scenarios:
                fault_response = await self._inject_fault(scenario)
                if fault_response:
                    injected_faults.append(fault_response)

            # Monitor metrics during failure
            start_time = time.time()
            while time.time() - start_time < config.failure_duration:
                timestamp = datetime.utcnow()

                # Collect metrics
                for service in config.target_services:
                    for metric_type in config.metrics_to_collect:
                        value = await self.metrics_collector.collect_metric(metric_type, service)

                        sample = MetricSample(
                            metric_type=metric_type,
                            service_name=service,
                            value=value,
                            timestamp=timestamp,
                            phase=TestPhase.FAILURE
                        )
                        phase_metrics.samples.append(sample)

                # Detect failures
                for service in config.target_services:
                    failure = await self.failure_detector.detect_service_failure(service)
                    if failure:
                        result.failures_detected.append(failure)

                # Detect cascading failures
                cascading = await self.failure_detector.detect_cascading_failures(config.target_services)
                result.cascading_failures.extend(cascading)

                await asyncio.sleep(5)

        finally:
            # Remove injected faults
            for fault_id in injected_faults:
                await self._remove_fault(fault_id)

        # Calculate failure statistics
        self._calculate_phase_statistics(phase_metrics)

        phase_metrics.end_time = datetime.utcnow()
        phase_metrics.duration = (phase_metrics.end_time - phase_metrics.start_time).total_seconds()
        result.phase_metrics[TestPhase.FAILURE] = phase_metrics

        self.logger.info(f"Failure phase completed for {config.name}")

    async def _recovery_phase(self, config: ResilienceTestConfig, result: ResilienceTestResult):
        """Recovery phase - monitor system recovery after failures."""
        self.logger.info(f"Starting recovery phase for {config.name}")

        phase_metrics = PhaseMetrics(
            phase=TestPhase.RECOVERY,
            start_time=datetime.utcnow()
        )

        recovery_start_time = time.time()
        services_recovered = set()

        # Monitor recovery for each service
        while time.time() - recovery_start_time < config.recovery_timeout:
            timestamp = datetime.utcnow()

            for service in config.target_services:
                # Collect metrics
                for metric_type in config.metrics_to_collect:
                    value = await self.metrics_collector.collect_metric(metric_type, service)

                    sample = MetricSample(
                        metric_type=metric_type,
                        service_name=service,
                        value=value,
                        timestamp=timestamp,
                        phase=TestPhase.RECOVERY
                    )
                    phase_metrics.samples.append(sample)

                # Check if service has recovered
                if service not in services_recovered:
                    availability = await self.metrics_collector.collect_availability(service)
                    if availability >= 95.0:  # 95% threshold for recovery
                        recovery_time = time.time() - recovery_start_time
                        services_recovered.add(service)

                        recovery_event = {
                            'service': service,
                            'recovery_time': recovery_time,
                            'timestamp': timestamp
                        }
                        result.recovery_events.append(recovery_event)

            # Check if all services have recovered
            if len(services_recovered) == len(config.target_services):
                result.recovery_time = time.time() - recovery_start_time
                break

            await asyncio.sleep(5)

        # If not all services recovered within timeout
        if result.recovery_time == 0.0:
            result.recovery_time = config.recovery_timeout

        # Calculate recovery statistics
        self._calculate_phase_statistics(phase_metrics)

        phase_metrics.end_time = datetime.utcnow()
        phase_metrics.duration = (phase_metrics.end_time - phase_metrics.start_time).total_seconds()
        result.phase_metrics[TestPhase.RECOVERY] = phase_metrics

        self.logger.info(f"Recovery phase completed for {config.name}")

    async def _validation_phase(self, config: ResilienceTestConfig, result: ResilienceTestResult):
        """Validation phase - validate system has returned to normal operation."""
        self.logger.info(f"Starting validation phase for {config.name}")

        phase_metrics = PhaseMetrics(
            phase=TestPhase.VALIDATION,
            start_time=datetime.utcnow()
        )

        # Collect validation metrics
        start_time = time.time()
        while time.time() - start_time < 60.0:  # 1 minute validation
            timestamp = datetime.utcnow()

            for service in config.target_services:
                for metric_type in config.metrics_to_collect:
                    value = await self.metrics_collector.collect_metric(metric_type, service)

                    sample = MetricSample(
                        metric_type=metric_type,
                        service_name=service,
                        value=value,
                        timestamp=timestamp,
                        phase=TestPhase.VALIDATION
                    )
                    phase_metrics.samples.append(sample)

            await asyncio.sleep(5)

        # Calculate validation statistics
        self._calculate_phase_statistics(phase_metrics)

        # Compare with baseline
        if TestPhase.BASELINE in result.phase_metrics:
            baseline = result.phase_metrics[TestPhase.BASELINE]

            # Check if performance has returned to acceptable levels
            availability_recovered = phase_metrics.availability >= baseline.availability * 0.95
            response_time_recovered = phase_metrics.avg_response_time <= baseline.avg_response_time * 1.2
            error_rate_recovered = phase_metrics.error_rate <= baseline.error_rate * 2.0

            if availability_recovered and response_time_recovered and error_rate_recovered:
                result.strengths.append("System recovered to baseline performance")
            else:
                result.weaknesses.append("System did not fully recover to baseline performance")

        phase_metrics.end_time = datetime.utcnow()
        phase_metrics.duration = (phase_metrics.end_time - phase_metrics.start_time).total_seconds()
        result.phase_metrics[TestPhase.VALIDATION] = phase_metrics

        self.logger.info(f"Validation phase completed for {config.name}")

    async def _analysis_phase(self, config: ResilienceTestConfig, result: ResilienceTestResult):
        """Analysis phase - analyze results and generate insights."""
        self.logger.info(f"Starting analysis phase for {config.name}")

        # Calculate overall availability
        all_availability_samples = []
        for phase_metrics in result.phase_metrics.values():
            availability_samples = [s.value for s in phase_metrics.samples
                                  if s.metric_type == ResilienceMetric.AVAILABILITY]
            all_availability_samples.extend(availability_samples)

        if all_availability_samples:
            result.overall_availability = statistics.mean(all_availability_samples)

        # Calculate performance impact
        if TestPhase.BASELINE in result.phase_metrics and TestPhase.FAILURE in result.phase_metrics:
            baseline = result.phase_metrics[TestPhase.BASELINE]
            failure = result.phase_metrics[TestPhase.FAILURE]

            performance_impact = 0.0
            if baseline.avg_response_time > 0:
                response_time_impact = (failure.avg_response_time - baseline.avg_response_time) / baseline.avg_response_time
                performance_impact = max(performance_impact, response_time_impact)

            if baseline.throughput > 0:
                throughput_impact = (baseline.throughput - failure.throughput) / baseline.throughput
                performance_impact = max(performance_impact, throughput_impact)

            result.performance_impact = performance_impact

            # Store degraded performance
            result.degraded_performance = {
                'availability': failure.availability,
                'avg_response_time': failure.avg_response_time,
                'error_rate': failure.error_rate,
                'throughput': failure.throughput
            }

        # Calculate resilience score (0-100)
        resilience_score = 0.0

        # Availability component (40% weight)
        if result.overall_availability >= 99.0:
            resilience_score += 40.0
        elif result.overall_availability >= 95.0:
            resilience_score += 30.0
        elif result.overall_availability >= 90.0:
            resilience_score += 20.0
        else:
            resilience_score += 10.0

        # Recovery time component (30% weight)
        if result.recovery_time <= 30.0:
            resilience_score += 30.0
        elif result.recovery_time <= 60.0:
            resilience_score += 25.0
        elif result.recovery_time <= 120.0:
            resilience_score += 20.0
        else:
            resilience_score += 10.0

        # Performance impact component (20% weight)
        if result.performance_impact <= 0.1:
            resilience_score += 20.0
        elif result.performance_impact <= 0.3:
            resilience_score += 15.0
        elif result.performance_impact <= 0.5:
            resilience_score += 10.0
        else:
            resilience_score += 5.0

        # Failure handling component (10% weight)
        if not result.cascading_failures:
            resilience_score += 10.0
        elif len(result.cascading_failures) <= 1:
            resilience_score += 7.0
        else:
            resilience_score += 3.0

        result.resilience_score = resilience_score

        # Determine resilience level
        if resilience_score >= 80.0:
            result.resilience_level = ResilienceLevel.ANTIFRAGILE
        elif resilience_score >= 60.0:
            result.resilience_level = ResilienceLevel.ROBUST
        else:
            result.resilience_level = ResilienceLevel.FRAGILE

        # Generate recommendations
        await self._generate_recommendations(config, result)

        self.logger.info(f"Analysis phase completed for {config.name}")

    async def _generate_recommendations(self, config: ResilienceTestConfig, result: ResilienceTestResult):
        """Generate recommendations based on test results."""

        # Availability recommendations
        if result.overall_availability < config.availability_threshold:
            result.recommendations.append(
                f"Improve system availability (current: {result.overall_availability:.1f}%, "
                f"target: {config.availability_threshold:.1f}%)"
            )

        # Recovery time recommendations
        if result.recovery_time > config.recovery_time_threshold:
            result.recommendations.append(
                f"Reduce recovery time (current: {result.recovery_time:.1f}s, "
                f"target: {config.recovery_time_threshold:.1f}s)"
            )

        # Performance recommendations
        if result.performance_impact > config.performance_degradation_threshold:
            result.recommendations.append(
                f"Reduce performance impact during failures (current: {result.performance_impact:.1%}, "
                f"target: {config.performance_degradation_threshold:.1%})"
            )

        # Cascading failure recommendations
        if result.cascading_failures:
            result.recommendations.append(
                "Implement circuit breakers to prevent cascading failures"
            )

        # Resilience level specific recommendations
        if result.resilience_level == ResilienceLevel.FRAGILE:
            result.recommendations.extend([
                "Implement comprehensive monitoring and alerting",
                "Add redundancy and failover mechanisms",
                "Improve error handling and graceful degradation"
            ])
        elif result.resilience_level == ResilienceLevel.ROBUST:
            result.recommendations.extend([
                "Consider chaos engineering practices",
                "Implement auto-scaling capabilities",
                "Add performance optimization"
            ])

    def _calculate_phase_statistics(self, phase_metrics: PhaseMetrics):
        """Calculate aggregated statistics for a phase."""
        if not phase_metrics.samples:
            return

        # Group samples by metric type
        metric_groups = {}
        for sample in phase_metrics.samples:
            if sample.metric_type not in metric_groups:
                metric_groups[sample.metric_type] = []
            metric_groups[sample.metric_type].append(sample.value)

        # Calculate statistics for each metric type
        for metric_type, values in metric_groups.items():
            if metric_type == ResilienceMetric.AVAILABILITY:
                phase_metrics.availability = statistics.mean(values)
            elif metric_type == ResilienceMetric.RESPONSE_TIME:
                valid_values = [v for v in values if v != float('inf')]
                if valid_values:
                    phase_metrics.avg_response_time = statistics.mean(valid_values)
                    phase_metrics.p95_response_time = np.percentile(valid_values, 95)
                    phase_metrics.p99_response_time = np.percentile(valid_values, 99)
            elif metric_type == ResilienceMetric.ERROR_RATE:
                phase_metrics.error_rate = statistics.mean(values)
            elif metric_type == ResilienceMetric.THROUGHPUT:
                phase_metrics.throughput = statistics.mean(values)

    async def _inject_fault(self, scenario: Dict[str, Any]) -> Optional[str]:
        """Inject a fault using the chaos engineering system."""
        try:
            async with self.session.post(f"{self.chaos_engineer_url}/inject", json=scenario) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('fault_id')
                return None
        except Exception as e:
            self.logger.error(f"Failed to inject fault: {e}")
            return None

    async def _remove_fault(self, fault_id: str) -> bool:
        """Remove an injected fault."""
        try:
            async with self.session.delete(f"{self.chaos_engineer_url}/fault/{fault_id}") as response:
                return response.status == 200
        except Exception as e:
            self.logger.error(f"Failed to remove fault {fault_id}: {e}")
            return False

    async def stop_test(self, test_name: str) -> bool:
        """Stop an active resilience test."""
        if test_name not in self.active_tests:
            return False

        result = self.active_tests[test_name]
        result.success = False
        result.end_time = datetime.utcnow()
        result.total_duration = (result.end_time - result.start_time).total_seconds()

        # Move to history
        self.test_history.append(result)
        self.active_tests.pop(test_name, None)

        self.logger.info(f"Stopped resilience test: {test_name}")
        return True

    async def _save_test_result(self, result: ResilienceTestResult):
        """Save test result to storage."""
        try:
            if self.redis_client:
                key = f"resilience:test:{result.test_name}:{int(result.start_time.timestamp())}"
                await self.redis_client.set(key, json.dumps(asdict(result), default=str))
                await self.redis_client.expire(key, 86400 * 30)  # 30 days retention

        except Exception as e:
            self.logger.error(f"Failed to save test result: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get resilience testing system status."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'active_tests': len(self.active_tests),
            'test_history_count': len(self.test_history),
            'active_test_names': list(self.active_tests.keys()),
            'components_status': {
                'metrics_collector': 'active',
                'load_generator': 'active',
                'failure_detector': 'active'
            }
        }


# Example usage
async def example_resilience_test():
    """Example of how to use the resilience testing system."""
    tester = ResilienceTester()

    try:
        await tester.initialize()

        # Configure test
        config = ResilienceTestConfig(
            name="web_service_resilience_test",
            description="Test web service resilience under various failure conditions",
            target_services=["web-server", "api-server"],
            test_duration=600.0,  # 10 minutes
            failure_scenarios=[
                {
                    "injector_name": "network_delay",
                    "target_service": "web-server",
                    "profile_name": "network_delay_light",
                    "parameters": {"delay_ms": 200, "jitter_ms": 50}
                }
            ],
            availability_threshold=99.0,
            recovery_time_threshold=60.0
        )

        # Run test
        result = await tester.run_resilience_test(config)

        print(f"Test completed: {result.success}")
        print(f"Resilience score: {result.resilience_score:.1f}")
        print(f"Resilience level: {result.resilience_level.value}")
        print(f"Recovery time: {result.recovery_time:.1f}s")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        await tester.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_resilience_test())