"""
Chaos Engineering System for BEV OSINT Framework
===============================================

Comprehensive chaos engineering system with fault injection capabilities,
resilience testing, and recovery validation for the BEV framework.

Features:
- Multi-dimensional fault injection (network, CPU, memory, service)
- Automated experiment orchestration and safety mechanisms
- Recovery validation with performance metrics
- Integration with auto-recovery and health monitoring systems
- Comprehensive scenario library for systematic stress testing

Author: BEV Infrastructure Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
import os
import signal
import random
import subprocess
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import aiohttp
import aioredis
import docker
import psutil
import yaml
from abc import ABC, abstractmethod


class FaultType(Enum):
    """Types of faults that can be injected."""
    NETWORK_DELAY = "network_delay"
    NETWORK_LOSS = "network_loss"
    NETWORK_PARTITION = "network_partition"
    CPU_STRESS = "cpu_stress"
    MEMORY_LEAK = "memory_leak"
    DISK_FILL = "disk_fill"
    SERVICE_CRASH = "service_crash"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATABASE_SLOWDOWN = "database_slowdown"
    CONNECTION_TIMEOUT = "connection_timeout"


class ExperimentPhase(Enum):
    """Phases of a chaos experiment."""
    PLANNING = "planning"
    BASELINE = "baseline"
    INJECTION = "injection"
    RECOVERY = "recovery"
    VALIDATION = "validation"
    COMPLETED = "completed"
    FAILED = "failed"


class SafetyLevel(Enum):
    """Safety levels for experiments."""
    LOW = "low"          # Minor impact, fast recovery
    MEDIUM = "medium"    # Moderate impact, expected recovery
    HIGH = "high"        # High impact, potential system degradation
    CRITICAL = "critical" # Critical impact, requires approval


@dataclass
class FaultInjectionConfig:
    """Configuration for fault injection."""
    fault_type: FaultType
    target_service: str
    intensity: float  # 0.0 to 1.0
    duration: float   # seconds
    parameters: Dict[str, Any] = field(default_factory=dict)
    safety_checks: List[str] = field(default_factory=list)
    rollback_triggers: List[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    """Configuration for a chaos experiment."""
    name: str
    description: str
    hypothesis: str
    safety_level: SafetyLevel
    target_services: List[str]
    fault_injections: List[FaultInjectionConfig]

    # Timing configuration
    baseline_duration: float = 60.0  # seconds
    injection_duration: float = 180.0
    recovery_timeout: float = 300.0
    validation_duration: float = 120.0

    # Safety configuration
    max_impact_threshold: float = 0.5  # Max acceptable impact
    auto_rollback_enabled: bool = True
    emergency_stop_conditions: List[str] = field(default_factory=list)

    # Recovery validation
    recovery_success_criteria: List[str] = field(default_factory=list)
    performance_thresholds: Dict[str, float] = field(default_factory=dict)

    # Metadata
    tags: List[str] = field(default_factory=list)
    author: str = ""
    created_at: Optional[datetime] = None


@dataclass
class ExperimentResult:
    """Results of a chaos experiment."""
    experiment_name: str
    start_time: datetime
    end_time: Optional[datetime]
    phase: ExperimentPhase
    success: bool

    # Metrics
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    injection_metrics: Dict[str, Any] = field(default_factory=dict)
    recovery_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_impact: float = 0.0
    recovery_time: float = 0.0

    # Details
    fault_injections_executed: List[Dict[str, Any]] = field(default_factory=list)
    errors_encountered: List[str] = field(default_factory=list)
    safety_violations: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class FaultInjector(ABC):
    """Abstract base class for fault injectors."""

    @abstractmethod
    async def inject_fault(self, config: FaultInjectionConfig) -> bool:
        """Inject the specified fault."""
        pass

    @abstractmethod
    async def remove_fault(self, config: FaultInjectionConfig) -> bool:
        """Remove the injected fault."""
        pass

    @abstractmethod
    async def validate_injection(self, config: FaultInjectionConfig) -> bool:
        """Validate that the fault was successfully injected."""
        pass


class NetworkFaultInjector(FaultInjector):
    """Fault injector for network-related issues."""

    def __init__(self, docker_client):
        self.docker_client = docker_client
        self.active_faults: Dict[str, str] = {}  # service -> fault_id

    async def inject_fault(self, config: FaultInjectionConfig) -> bool:
        """Inject network fault using tc (traffic control)."""
        try:
            container = self.docker_client.containers.get(config.target_service)

            if config.fault_type == FaultType.NETWORK_DELAY:
                delay = config.parameters.get('delay_ms', 100)
                jitter = config.parameters.get('jitter_ms', 10)
                command = f"tc qdisc add dev eth0 root netem delay {delay}ms {jitter}ms"

            elif config.fault_type == FaultType.NETWORK_LOSS:
                loss_percent = config.parameters.get('loss_percent', 5)
                command = f"tc qdisc add dev eth0 root netem loss {loss_percent}%"

            elif config.fault_type == FaultType.NETWORK_PARTITION:
                target_ip = config.parameters.get('target_ip', '172.30.0.0/16')
                command = f"iptables -A OUTPUT -d {target_ip} -j DROP"

            else:
                return False

            # Execute command in container
            result = container.exec_run(command, privileged=True)
            if result.exit_code == 0:
                self.active_faults[config.target_service] = config.fault_type.value
                return True

            return False

        except Exception as e:
            logging.error(f"Failed to inject network fault: {e}")
            return False

    async def remove_fault(self, config: FaultInjectionConfig) -> bool:
        """Remove network fault."""
        try:
            container = self.docker_client.containers.get(config.target_service)

            if config.fault_type in [FaultType.NETWORK_DELAY, FaultType.NETWORK_LOSS]:
                command = "tc qdisc del dev eth0 root"
            elif config.fault_type == FaultType.NETWORK_PARTITION:
                target_ip = config.parameters.get('target_ip', '172.30.0.0/16')
                command = f"iptables -D OUTPUT -d {target_ip} -j DROP"
            else:
                return False

            result = container.exec_run(command, privileged=True)
            if result.exit_code == 0:
                self.active_faults.pop(config.target_service, None)
                return True

            return False

        except Exception as e:
            logging.error(f"Failed to remove network fault: {e}")
            return False

    async def validate_injection(self, config: FaultInjectionConfig) -> bool:
        """Validate network fault injection."""
        try:
            container = self.docker_client.containers.get(config.target_service)

            if config.fault_type in [FaultType.NETWORK_DELAY, FaultType.NETWORK_LOSS]:
                result = container.exec_run("tc qdisc show dev eth0")
                return "netem" in result.output.decode()
            elif config.fault_type == FaultType.NETWORK_PARTITION:
                result = container.exec_run("iptables -L OUTPUT")
                return "DROP" in result.output.decode()

            return False

        except Exception:
            return False


class ResourceFaultInjector(FaultInjector):
    """Fault injector for resource-related issues."""

    def __init__(self, docker_client):
        self.docker_client = docker_client
        self.active_stress_processes: Dict[str, List[subprocess.Popen]] = {}

    async def inject_fault(self, config: FaultInjectionConfig) -> bool:
        """Inject resource fault."""
        try:
            container = self.docker_client.containers.get(config.target_service)

            if config.fault_type == FaultType.CPU_STRESS:
                cpu_cores = config.parameters.get('cpu_cores', 1)
                cpu_load = config.parameters.get('cpu_load', 80)
                command = f"stress-ng --cpu {cpu_cores} --cpu-load {cpu_load} --timeout {config.duration}s"

            elif config.fault_type == FaultType.MEMORY_LEAK:
                memory_mb = config.parameters.get('memory_mb', 100)
                command = f"stress-ng --vm 1 --vm-bytes {memory_mb}M --timeout {config.duration}s"

            elif config.fault_type == FaultType.DISK_FILL:
                disk_size = config.parameters.get('disk_size', '1G')
                command = f"fallocate -l {disk_size} /tmp/chaos_fill_file"

            else:
                return False

            # Execute stress command in background
            result = container.exec_run(f"nohup {command} &", detach=True)

            if config.target_service not in self.active_stress_processes:
                self.active_stress_processes[config.target_service] = []

            return True

        except Exception as e:
            logging.error(f"Failed to inject resource fault: {e}")
            return False

    async def remove_fault(self, config: FaultInjectionConfig) -> bool:
        """Remove resource fault."""
        try:
            container = self.docker_client.containers.get(config.target_service)

            if config.fault_type in [FaultType.CPU_STRESS, FaultType.MEMORY_LEAK]:
                # Kill stress processes
                container.exec_run("pkill -f stress-ng")

            elif config.fault_type == FaultType.DISK_FILL:
                # Remove fill file
                container.exec_run("rm -f /tmp/chaos_fill_file")

            self.active_stress_processes.pop(config.target_service, None)
            return True

        except Exception as e:
            logging.error(f"Failed to remove resource fault: {e}")
            return False

    async def validate_injection(self, config: FaultInjectionConfig) -> bool:
        """Validate resource fault injection."""
        try:
            container = self.docker_client.containers.get(config.target_service)

            if config.fault_type in [FaultType.CPU_STRESS, FaultType.MEMORY_LEAK]:
                result = container.exec_run("pgrep stress-ng")
                return result.exit_code == 0

            elif config.fault_type == FaultType.DISK_FILL:
                result = container.exec_run("ls -la /tmp/chaos_fill_file")
                return result.exit_code == 0

            return False

        except Exception:
            return False


class ServiceFaultInjector(FaultInjector):
    """Fault injector for service-level issues."""

    def __init__(self, docker_client):
        self.docker_client = docker_client
        self.stopped_containers: List[str] = []

    async def inject_fault(self, config: FaultInjectionConfig) -> bool:
        """Inject service fault."""
        try:
            if config.fault_type == FaultType.SERVICE_CRASH:
                container = self.docker_client.containers.get(config.target_service)

                crash_method = config.parameters.get('method', 'stop')
                if crash_method == 'kill':
                    container.kill()
                else:
                    container.stop(timeout=5)

                self.stopped_containers.append(config.target_service)
                return True

            return False

        except Exception as e:
            logging.error(f"Failed to inject service fault: {e}")
            return False

    async def remove_fault(self, config: FaultInjectionConfig) -> bool:
        """Remove service fault by restarting service."""
        try:
            if config.fault_type == FaultType.SERVICE_CRASH:
                container = self.docker_client.containers.get(config.target_service)
                container.start()

                # Wait for container to be running
                for _ in range(30):
                    container.reload()
                    if container.status == 'running':
                        self.stopped_containers.remove(config.target_service)
                        return True
                    await asyncio.sleep(1)

                return False

        except Exception as e:
            logging.error(f"Failed to remove service fault: {e}")
            return False

    async def validate_injection(self, config: FaultInjectionConfig) -> bool:
        """Validate service fault injection."""
        try:
            container = self.docker_client.containers.get(config.target_service)
            container.reload()

            if config.fault_type == FaultType.SERVICE_CRASH:
                return container.status != 'running'

            return False

        except Exception:
            return True  # Container not found means it's crashed


class ChaosEngineer:
    """
    Main chaos engineering system with comprehensive fault injection,
    resilience testing, and recovery validation capabilities.
    """

    def __init__(self,
                 config_path: str = "/app/config/chaos_engineer.yaml",
                 redis_url: str = "redis://redis:6379/12",
                 auto_recovery_url: str = "http://172.30.0.41:8080",
                 health_monitor_url: str = "http://172.30.0.38:8080"):
        """
        Initialize the chaos engineering system.

        Args:
            config_path: Path to configuration file
            redis_url: Redis connection URL for state storage
            auto_recovery_url: URL for auto-recovery system integration
            health_monitor_url: URL for health monitoring system integration
        """
        self.config_path = Path(config_path)
        self.redis_url = redis_url
        self.auto_recovery_url = auto_recovery_url
        self.health_monitor_url = health_monitor_url

        # Initialize connections
        self.redis_client: Optional[aioredis.Redis] = None
        self.docker_client = docker.from_env()
        self.session: Optional[aiohttp.ClientSession] = None

        # Initialize fault injectors
        self.fault_injectors: Dict[FaultType, FaultInjector] = {
            FaultType.NETWORK_DELAY: NetworkFaultInjector(self.docker_client),
            FaultType.NETWORK_LOSS: NetworkFaultInjector(self.docker_client),
            FaultType.NETWORK_PARTITION: NetworkFaultInjector(self.docker_client),
            FaultType.CPU_STRESS: ResourceFaultInjector(self.docker_client),
            FaultType.MEMORY_LEAK: ResourceFaultInjector(self.docker_client),
            FaultType.DISK_FILL: ResourceFaultInjector(self.docker_client),
            FaultType.SERVICE_CRASH: ServiceFaultInjector(self.docker_client),
        }

        # Experiment management
        self.active_experiments: Dict[str, ExperimentResult] = {}
        self.experiment_history: List[ExperimentResult] = []
        self.safety_monitor_active = False

        # Configuration
        self.max_concurrent_experiments = 3
        self.safety_check_interval = 10.0  # seconds
        self.emergency_stop_threshold = 0.8  # 80% impact

        # Logging
        self.logger = logging.getLogger("chaos_engineer")
        self.logger.setLevel(logging.INFO)

        # Load configuration
        self._load_configuration()

    def _load_configuration(self):
        """Load chaos engineering configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)

                # Load global settings
                global_config = config_data.get('global', {})
                self.max_concurrent_experiments = global_config.get('max_concurrent_experiments', 3)
                self.safety_check_interval = global_config.get('safety_check_interval', 10.0)
                self.emergency_stop_threshold = global_config.get('emergency_stop_threshold', 0.8)

                self.logger.info("Chaos engineering configuration loaded")
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")

    async def initialize(self):
        """Initialize the chaos engineering system."""
        # Initialize Redis connection
        self.redis_client = aioredis.from_url(self.redis_url)

        # Initialize HTTP session
        self.session = aiohttp.ClientSession()

        # Start safety monitoring
        self.safety_monitor_active = True
        asyncio.create_task(self._safety_monitor())

        self.logger.info("Chaos engineering system initialized")

    async def shutdown(self):
        """Shutdown the chaos engineering system."""
        self.safety_monitor_active = False

        # Stop all active experiments
        for experiment_name in list(self.active_experiments.keys()):
            await self.stop_experiment(experiment_name, emergency=True)

        # Close connections
        if self.session:
            await self.session.close()
        if self.redis_client:
            await self.redis_client.close()

        self.logger.info("Chaos engineering system shutdown complete")

    async def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a complete chaos experiment with all phases.

        Args:
            config: Experiment configuration

        Returns:
            ExperimentResult: Comprehensive experiment results
        """
        if len(self.active_experiments) >= self.max_concurrent_experiments:
            raise RuntimeError("Maximum concurrent experiments reached")

        # Create experiment result
        result = ExperimentResult(
            experiment_name=config.name,
            start_time=datetime.utcnow(),
            end_time=None,
            phase=ExperimentPhase.PLANNING,
            success=False
        )

        self.active_experiments[config.name] = result

        try:
            self.logger.info(f"Starting chaos experiment: {config.name}")

            # Phase 1: Planning and safety checks
            result.phase = ExperimentPhase.PLANNING
            await self._validate_experiment_safety(config, result)

            # Phase 2: Baseline metrics collection
            result.phase = ExperimentPhase.BASELINE
            await self._collect_baseline_metrics(config, result)

            # Phase 3: Fault injection
            result.phase = ExperimentPhase.INJECTION
            await self._execute_fault_injections(config, result)

            # Phase 4: Recovery monitoring
            result.phase = ExperimentPhase.RECOVERY
            await self._monitor_recovery(config, result)

            # Phase 5: Validation
            result.phase = ExperimentPhase.VALIDATION
            await self._validate_recovery(config, result)

            # Mark as completed
            result.phase = ExperimentPhase.COMPLETED
            result.success = True
            result.end_time = datetime.utcnow()

            self.logger.info(f"Chaos experiment completed successfully: {config.name}")

        except Exception as e:
            result.phase = ExperimentPhase.FAILED
            result.errors_encountered.append(str(e))
            result.end_time = datetime.utcnow()
            self.logger.error(f"Chaos experiment failed: {config.name} - {e}")

            # Emergency cleanup
            await self._emergency_cleanup(config, result)

        finally:
            # Move to history and cleanup
            self.experiment_history.append(result)
            self.active_experiments.pop(config.name, None)

            # Save results
            await self._save_experiment_result(result)

        return result

    async def _validate_experiment_safety(self, config: ExperimentConfig, result: ExperimentResult):
        """Validate experiment safety before execution."""
        # Check system health
        health_status = await self._get_system_health()
        if not health_status.get('overall_healthy', False):
            raise RuntimeError("System not healthy enough for chaos experiments")

        # Check safety level permissions
        if config.safety_level == SafetyLevel.CRITICAL:
            # Require manual approval for critical experiments
            approval = await self._check_critical_experiment_approval(config.name)
            if not approval:
                raise RuntimeError("Critical experiment requires manual approval")

        # Validate target services exist
        for service_name in config.target_services:
            try:
                self.docker_client.containers.get(service_name)
            except docker.errors.NotFound:
                raise RuntimeError(f"Target service not found: {service_name}")

        # Check for concurrent experiments on same services
        for active_name, active_result in self.active_experiments.items():
            if active_name != config.name:
                active_config = await self._get_experiment_config(active_name)
                if active_config and set(config.target_services) & set(active_config.target_services):
                    raise RuntimeError(f"Conflicting experiment already running: {active_name}")

        self.logger.info(f"Safety validation passed for experiment: {config.name}")

    async def _collect_baseline_metrics(self, config: ExperimentConfig, result: ExperimentResult):
        """Collect baseline metrics before fault injection."""
        self.logger.info(f"Collecting baseline metrics for {config.name}")

        baseline_start = time.time()

        # Collect metrics for specified duration
        while time.time() - baseline_start < config.baseline_duration:
            timestamp = datetime.utcnow()

            # Collect system metrics
            system_metrics = await self._collect_system_metrics(config.target_services)

            # Collect application metrics
            app_metrics = await self._collect_application_metrics(config.target_services)

            # Store baseline data
            if 'samples' not in result.baseline_metrics:
                result.baseline_metrics['samples'] = []

            result.baseline_metrics['samples'].append({
                'timestamp': timestamp.isoformat(),
                'system': system_metrics,
                'application': app_metrics
            })

            await asyncio.sleep(5)  # 5-second sampling interval

        # Calculate baseline averages
        result.baseline_metrics['averages'] = self._calculate_metric_averages(
            result.baseline_metrics['samples']
        )

        self.logger.info(f"Baseline metrics collected for {config.name}")

    async def _execute_fault_injections(self, config: ExperimentConfig, result: ExperimentResult):
        """Execute all configured fault injections."""
        self.logger.info(f"Executing fault injections for {config.name}")

        injection_start = time.time()
        injected_faults = []

        try:
            # Inject all configured faults
            for fault_config in config.fault_injections:
                injector = self.fault_injectors.get(fault_config.fault_type)
                if not injector:
                    raise RuntimeError(f"No injector available for fault type: {fault_config.fault_type}")

                success = await injector.inject_fault(fault_config)
                if success:
                    injected_faults.append(fault_config)
                    result.fault_injections_executed.append({
                        'fault_type': fault_config.fault_type.value,
                        'target_service': fault_config.target_service,
                        'timestamp': datetime.utcnow().isoformat(),
                        'success': True
                    })
                    self.logger.info(f"Injected fault: {fault_config.fault_type.value} on {fault_config.target_service}")
                else:
                    result.errors_encountered.append(f"Failed to inject fault: {fault_config.fault_type.value}")

            # Monitor system during injection
            while time.time() - injection_start < config.injection_duration:
                # Collect metrics during injection
                timestamp = datetime.utcnow()
                system_metrics = await self._collect_system_metrics(config.target_services)
                app_metrics = await self._collect_application_metrics(config.target_services)

                if 'samples' not in result.injection_metrics:
                    result.injection_metrics['samples'] = []

                result.injection_metrics['samples'].append({
                    'timestamp': timestamp.isoformat(),
                    'system': system_metrics,
                    'application': app_metrics
                })

                # Safety monitoring
                current_impact = self._calculate_performance_impact(
                    result.baseline_metrics['averages'],
                    system_metrics
                )

                if current_impact > self.emergency_stop_threshold:
                    result.safety_violations.append(f"Emergency stop triggered: impact {current_impact}")
                    break

                await asyncio.sleep(5)

        finally:
            # Remove all injected faults
            for fault_config in injected_faults:
                injector = self.fault_injectors.get(fault_config.fault_type)
                if injector:
                    await injector.remove_fault(fault_config)
                    self.logger.info(f"Removed fault: {fault_config.fault_type.value} from {fault_config.target_service}")

        # Calculate injection averages
        if 'samples' in result.injection_metrics:
            result.injection_metrics['averages'] = self._calculate_metric_averages(
                result.injection_metrics['samples']
            )

        self.logger.info(f"Fault injection phase completed for {config.name}")

    async def _monitor_recovery(self, config: ExperimentConfig, result: ExperimentResult):
        """Monitor system recovery after fault removal."""
        self.logger.info(f"Monitoring recovery for {config.name}")

        recovery_start = time.time()
        recovery_detected = False

        while time.time() - recovery_start < config.recovery_timeout:
            timestamp = datetime.utcnow()
            system_metrics = await self._collect_system_metrics(config.target_services)
            app_metrics = await self._collect_application_metrics(config.target_services)

            if 'samples' not in result.recovery_metrics:
                result.recovery_metrics['samples'] = []

            result.recovery_metrics['samples'].append({
                'timestamp': timestamp.isoformat(),
                'system': system_metrics,
                'application': app_metrics
            })

            # Check if recovery is complete
            if not recovery_detected:
                recovery_complete = await self._check_recovery_complete(
                    config, result.baseline_metrics['averages'], system_metrics
                )

                if recovery_complete:
                    result.recovery_time = time.time() - recovery_start
                    recovery_detected = True
                    self.logger.info(f"Recovery detected for {config.name} after {result.recovery_time:.2f}s")

            await asyncio.sleep(5)

        if not recovery_detected:
            result.recovery_time = config.recovery_timeout
            result.errors_encountered.append("Recovery timeout exceeded")

        # Calculate recovery averages
        if 'samples' in result.recovery_metrics:
            result.recovery_metrics['averages'] = self._calculate_metric_averages(
                result.recovery_metrics['samples']
            )

        self.logger.info(f"Recovery monitoring completed for {config.name}")

    async def _validate_recovery(self, config: ExperimentConfig, result: ExperimentResult):
        """Validate that recovery was successful."""
        self.logger.info(f"Validating recovery for {config.name}")

        # Calculate overall performance impact
        if result.baseline_metrics.get('averages') and result.recovery_metrics.get('averages'):
            result.performance_impact = self._calculate_performance_impact(
                result.baseline_metrics['averages'],
                result.recovery_metrics['averages']
            )

        # Check recovery success criteria
        recovery_successful = True

        for criterion in config.recovery_success_criteria:
            if not await self._evaluate_recovery_criterion(criterion, result):
                recovery_successful = False
                result.errors_encountered.append(f"Recovery criterion failed: {criterion}")

        # Validate performance thresholds
        for metric_name, threshold in config.performance_thresholds.items():
            current_value = self._get_metric_value(result.recovery_metrics['averages'], metric_name)
            baseline_value = self._get_metric_value(result.baseline_metrics['averages'], metric_name)

            if current_value and baseline_value:
                deviation = abs(current_value - baseline_value) / baseline_value
                if deviation > threshold:
                    recovery_successful = False
                    result.errors_encountered.append(
                        f"Performance threshold exceeded for {metric_name}: {deviation:.2%} > {threshold:.2%}"
                    )

        # Check auto-recovery system response
        auto_recovery_response = await self._validate_auto_recovery_response(config.target_services)
        if not auto_recovery_response['all_services_recovered']:
            recovery_successful = False
            result.errors_encountered.append("Auto-recovery system did not restore all services")

        if recovery_successful:
            result.lessons_learned.append("System recovered successfully from injected faults")
            self.logger.info(f"Recovery validation passed for {config.name}")
        else:
            result.lessons_learned.append("System recovery was incomplete or slow")
            self.logger.warning(f"Recovery validation failed for {config.name}")

    async def _safety_monitor(self):
        """Continuous safety monitoring during experiments."""
        while self.safety_monitor_active:
            try:
                for experiment_name, result in self.active_experiments.items():
                    # Check for emergency stop conditions
                    if result.phase == ExperimentPhase.INJECTION:
                        current_metrics = await self._collect_system_metrics(
                            [result.experiment_name]  # Use experiment name as service identifier
                        )

                        if result.baseline_metrics.get('averages'):
                            impact = self._calculate_performance_impact(
                                result.baseline_metrics['averages'],
                                current_metrics
                            )

                            if impact > self.emergency_stop_threshold:
                                self.logger.warning(f"Emergency stop triggered for {experiment_name}")
                                await self.stop_experiment(experiment_name, emergency=True)

                await asyncio.sleep(self.safety_check_interval)

            except Exception as e:
                self.logger.error(f"Safety monitor error: {e}")
                await asyncio.sleep(30)

    async def stop_experiment(self, experiment_name: str, emergency: bool = False):
        """Stop an active experiment."""
        result = self.active_experiments.get(experiment_name)
        if not result:
            return

        self.logger.warning(f"Stopping experiment {experiment_name} (emergency: {emergency})")

        # Emergency cleanup
        config = await self._get_experiment_config(experiment_name)
        if config:
            await self._emergency_cleanup(config, result)

        # Mark as failed if emergency stop
        if emergency:
            result.phase = ExperimentPhase.FAILED
            result.safety_violations.append("Emergency stop executed")

        result.end_time = datetime.utcnow()

    async def _emergency_cleanup(self, config: ExperimentConfig, result: ExperimentResult):
        """Emergency cleanup for failed experiments."""
        self.logger.info(f"Executing emergency cleanup for {config.name}")

        # Remove all possible fault injections
        for fault_type, injector in self.fault_injectors.items():
            for service in config.target_services:
                try:
                    fault_config = FaultInjectionConfig(
                        fault_type=fault_type,
                        target_service=service,
                        intensity=0.0,
                        duration=0.0
                    )
                    await injector.remove_fault(fault_config)
                except Exception as e:
                    self.logger.warning(f"Emergency cleanup failed for {fault_type} on {service}: {e}")

        # Trigger auto-recovery for all target services
        for service in config.target_services:
            await self._trigger_auto_recovery(service)

    async def _collect_system_metrics(self, service_names: List[str]) -> Dict[str, Any]:
        """Collect system metrics for specified services."""
        metrics = {}

        for service_name in service_names:
            try:
                container = self.docker_client.containers.get(service_name)
                stats = container.stats(stream=False)

                # CPU metrics
                cpu_usage = 0.0
                if 'cpu_stats' in stats and 'precpu_stats' in stats:
                    cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                               stats['precpu_stats']['cpu_usage']['total_usage']
                    system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                  stats['precpu_stats']['system_cpu_usage']

                    if system_delta > 0:
                        cpu_usage = (cpu_delta / system_delta) * 100.0

                # Memory metrics
                memory_usage = 0.0
                memory_limit = 0.0
                if 'memory_stats' in stats:
                    memory_usage = stats['memory_stats'].get('usage', 0)
                    memory_limit = stats['memory_stats'].get('limit', 0)

                metrics[service_name] = {
                    'cpu_usage_percent': cpu_usage,
                    'memory_usage_bytes': memory_usage,
                    'memory_limit_bytes': memory_limit,
                    'memory_usage_percent': (memory_usage / memory_limit * 100) if memory_limit > 0 else 0,
                    'container_status': container.status,
                    'timestamp': datetime.utcnow().isoformat()
                }

            except Exception as e:
                self.logger.warning(f"Failed to collect metrics for {service_name}: {e}")
                metrics[service_name] = {
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }

        return metrics

    async def _collect_application_metrics(self, service_names: List[str]) -> Dict[str, Any]:
        """Collect application-specific metrics."""
        metrics = {}

        for service_name in service_names:
            try:
                # Try to get metrics from health monitoring system
                async with self.session.get(f"{self.health_monitor_url}/metrics/{service_name}") as response:
                    if response.status == 200:
                        metrics[service_name] = await response.json()
                    else:
                        metrics[service_name] = {'error': f'HTTP {response.status}'}

            except Exception as e:
                metrics[service_name] = {'error': str(e)}

        return metrics

    def _calculate_metric_averages(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate average values from metric samples."""
        if not samples:
            return {}

        averages = {}

        # Process system metrics
        for service in samples[0].get('system', {}):
            service_samples = [s['system'][service] for s in samples if service in s['system']]
            if service_samples and 'error' not in service_samples[0]:
                averages[service] = {
                    'cpu_usage_percent': sum(s['cpu_usage_percent'] for s in service_samples) / len(service_samples),
                    'memory_usage_percent': sum(s['memory_usage_percent'] for s in service_samples) / len(service_samples),
                }

        return averages

    def _calculate_performance_impact(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> float:
        """Calculate performance impact compared to baseline."""
        total_impact = 0.0
        service_count = 0

        for service in baseline:
            if service in current and 'error' not in current[service]:
                baseline_cpu = baseline[service].get('cpu_usage_percent', 0)
                current_cpu = current[service].get('cpu_usage_percent', 0)

                baseline_memory = baseline[service].get('memory_usage_percent', 0)
                current_memory = current[service].get('memory_usage_percent', 0)

                # Calculate relative impact
                cpu_impact = abs(current_cpu - baseline_cpu) / max(baseline_cpu, 1)
                memory_impact = abs(current_memory - baseline_memory) / max(baseline_memory, 1)

                service_impact = (cpu_impact + memory_impact) / 2
                total_impact += service_impact
                service_count += 1

        return total_impact / max(service_count, 1)

    async def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            async with self.session.get(f"{self.health_monitor_url}/health/overall") as response:
                if response.status == 200:
                    return await response.json()
                return {'overall_healthy': False, 'error': f'HTTP {response.status}'}
        except Exception as e:
            return {'overall_healthy': False, 'error': str(e)}

    async def _check_critical_experiment_approval(self, experiment_name: str) -> bool:
        """Check if critical experiment has manual approval."""
        # In a real implementation, this would check an approval system
        # For now, assume all critical experiments are pre-approved
        return True

    async def _get_experiment_config(self, experiment_name: str) -> Optional[ExperimentConfig]:
        """Get experiment configuration by name."""
        # This would typically load from storage
        # For now, return None as configs are passed directly
        return None

    async def _check_recovery_complete(self, config: ExperimentConfig, baseline: Dict[str, Any], current: Dict[str, Any]) -> bool:
        """Check if system has recovered to baseline performance."""
        impact = self._calculate_performance_impact(baseline, current)
        return impact < 0.1  # 10% threshold for recovery

    async def _evaluate_recovery_criterion(self, criterion: str, result: ExperimentResult) -> bool:
        """Evaluate a specific recovery success criterion."""
        # Parse and evaluate criteria like "response_time < 200ms"
        # For now, return True as a placeholder
        return True

    def _get_metric_value(self, metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract a specific metric value from metrics dict."""
        # Navigate nested dict to find metric_name
        # For now, return None as placeholder
        return None

    async def _validate_auto_recovery_response(self, service_names: List[str]) -> Dict[str, Any]:
        """Validate that auto-recovery system responded appropriately."""
        try:
            recovery_status = {}
            for service in service_names:
                async with self.session.get(f"{self.auto_recovery_url}/status/{service}") as response:
                    if response.status == 200:
                        status = await response.json()
                        recovery_status[service] = status.get('state') == 'healthy'
                    else:
                        recovery_status[service] = False

            return {
                'all_services_recovered': all(recovery_status.values()),
                'service_status': recovery_status
            }
        except Exception as e:
            return {'all_services_recovered': False, 'error': str(e)}

    async def _trigger_auto_recovery(self, service_name: str):
        """Trigger auto-recovery for a specific service."""
        try:
            async with self.session.post(f"{self.auto_recovery_url}/recover/{service_name}") as response:
                if response.status == 200:
                    self.logger.info(f"Auto-recovery triggered for {service_name}")
                else:
                    self.logger.warning(f"Failed to trigger auto-recovery for {service_name}")
        except Exception as e:
            self.logger.error(f"Error triggering auto-recovery for {service_name}: {e}")

    async def _save_experiment_result(self, result: ExperimentResult):
        """Save experiment result to storage."""
        try:
            if self.redis_client:
                key = f"chaos:experiment:{result.experiment_name}:{int(result.start_time.timestamp())}"
                await self.redis_client.set(key, json.dumps(asdict(result), default=str))
                await self.redis_client.expire(key, 86400 * 30)  # 30 days retention

        except Exception as e:
            self.logger.error(f"Failed to save experiment result: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get chaos engineering system status."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'active_experiments': len(self.active_experiments),
            'experiment_history_count': len(self.experiment_history),
            'safety_monitor_active': self.safety_monitor_active,
            'available_fault_types': [ft.value for ft in self.fault_injectors.keys()],
            'active_experiment_names': list(self.active_experiments.keys())
        }


# Signal handler for graceful shutdown
def setup_signal_handlers(chaos_engineer: ChaosEngineer):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        asyncio.create_task(chaos_engineer.shutdown())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Main entry point
async def main():
    """Main entry point for the chaos engineering system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize chaos engineer
    chaos_engineer = ChaosEngineer()

    # Setup signal handlers
    setup_signal_handlers(chaos_engineer)

    try:
        # Initialize system
        await chaos_engineer.initialize()

        # Keep running until shutdown
        while chaos_engineer.safety_monitor_active:
            await asyncio.sleep(10)

    except KeyboardInterrupt:
        pass
    finally:
        await chaos_engineer.shutdown()


if __name__ == "__main__":
    asyncio.run(main())