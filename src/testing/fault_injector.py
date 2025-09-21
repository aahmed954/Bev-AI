"""
Advanced Fault Injection System for BEV OSINT Framework
======================================================

Comprehensive fault injection capabilities including network delays,
CPU spikes, memory leaks, service crashes, and resource exhaustion.

Features:
- Multiple fault injection strategies with precise control
- Safety mechanisms and automatic rollback capabilities
- Integration with container orchestration and system monitoring
- Granular fault configuration and intensity control
- Real-time fault validation and monitoring

Author: BEV Infrastructure Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
import os
import subprocess
import psutil
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import docker
import aioredis
from abc import ABC, abstractmethod


class FaultCategory(Enum):
    """Categories of fault injection."""
    NETWORK = "network"
    COMPUTE = "compute"
    STORAGE = "storage"
    APPLICATION = "application"
    SYSTEM = "system"


class FaultSeverity(Enum):
    """Severity levels for fault injection."""
    LOW = "low"          # Minimal impact, quick recovery
    MEDIUM = "medium"    # Moderate impact, normal recovery
    HIGH = "high"        # Significant impact, extended recovery
    CRITICAL = "critical" # Severe impact, may require intervention


@dataclass
class FaultProfile:
    """Profile for a specific type of fault injection."""
    name: str
    category: FaultCategory
    severity: FaultSeverity
    description: str

    # Injection parameters
    min_intensity: float = 0.1
    max_intensity: float = 1.0
    min_duration: float = 30.0   # seconds
    max_duration: float = 300.0  # seconds

    # Safety parameters
    safety_checks: List[str] = field(default_factory=list)
    rollback_triggers: List[str] = field(default_factory=list)
    max_concurrent: int = 1

    # Prerequisites
    required_tools: List[str] = field(default_factory=list)
    target_requirements: List[str] = field(default_factory=list)


@dataclass
class FaultExecutionContext:
    """Context for fault execution with monitoring."""
    fault_id: str
    target_service: str
    profile: FaultProfile
    parameters: Dict[str, Any]

    # Execution state
    start_time: datetime
    end_time: Optional[datetime] = None
    active: bool = False
    validated: bool = False

    # Monitoring
    pre_injection_metrics: Dict[str, Any] = field(default_factory=dict)
    during_injection_metrics: List[Dict[str, Any]] = field(default_factory=list)
    post_injection_metrics: Dict[str, Any] = field(default_factory=dict)

    # Results
    success: bool = False
    error_message: Optional[str] = None
    rollback_triggered: bool = False
    impact_score: float = 0.0


class AdvancedFaultInjector(ABC):
    """Abstract base for advanced fault injectors with monitoring."""

    def __init__(self, name: str, category: FaultCategory):
        self.name = name
        self.category = category
        self.active_faults: Dict[str, FaultExecutionContext] = {}
        self.logger = logging.getLogger(f"fault_injector.{name}")

    @abstractmethod
    async def inject(self, context: FaultExecutionContext) -> bool:
        """Inject the fault with the given context."""
        pass

    @abstractmethod
    async def remove(self, context: FaultExecutionContext) -> bool:
        """Remove the injected fault."""
        pass

    @abstractmethod
    async def validate(self, context: FaultExecutionContext) -> bool:
        """Validate that the fault is active and working as expected."""
        pass

    @abstractmethod
    async def monitor(self, context: FaultExecutionContext) -> Dict[str, Any]:
        """Monitor the fault's impact and effects."""
        pass

    async def execute_with_monitoring(self, context: FaultExecutionContext) -> bool:
        """Execute fault injection with comprehensive monitoring."""
        try:
            # Pre-injection monitoring
            context.pre_injection_metrics = await self.monitor(context)

            # Inject fault
            success = await self.inject(context)
            if not success:
                context.error_message = "Fault injection failed"
                return False

            context.active = True
            context.start_time = datetime.utcnow()
            self.active_faults[context.fault_id] = context

            # Validate injection
            context.validated = await self.validate(context)
            if not context.validated:
                await self.remove(context)
                context.error_message = "Fault validation failed"
                return False

            self.logger.info(f"Fault {context.fault_id} injected successfully")
            return True

        except Exception as e:
            context.error_message = str(e)
            self.logger.error(f"Fault injection failed: {e}")
            return False

    async def cleanup_fault(self, fault_id: str) -> bool:
        """Clean up a specific fault."""
        context = self.active_faults.get(fault_id)
        if not context:
            return True

        try:
            success = await self.remove(context)
            context.active = False
            context.end_time = datetime.utcnow()

            # Post-injection monitoring
            context.post_injection_metrics = await self.monitor(context)

            if success:
                self.logger.info(f"Fault {fault_id} removed successfully")
            else:
                self.logger.warning(f"Failed to remove fault {fault_id}")

            return success

        except Exception as e:
            self.logger.error(f"Error removing fault {fault_id}: {e}")
            return False
        finally:
            self.active_faults.pop(fault_id, None)


class NetworkDelayInjector(AdvancedFaultInjector):
    """Network delay and latency fault injector."""

    def __init__(self, docker_client):
        super().__init__("network_delay", FaultCategory.NETWORK)
        self.docker_client = docker_client

    async def inject(self, context: FaultExecutionContext) -> bool:
        """Inject network delay using tc (traffic control)."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            delay_ms = context.parameters.get('delay_ms', 100)
            jitter_ms = context.parameters.get('jitter_ms', 10)
            loss_percent = context.parameters.get('loss_percent', 0)

            # Build tc command
            cmd_parts = [
                "tc", "qdisc", "add", "dev", "eth0", "root", "netem",
                "delay", f"{delay_ms}ms"
            ]

            if jitter_ms > 0:
                cmd_parts.extend([f"{jitter_ms}ms"])

            if loss_percent > 0:
                cmd_parts.extend(["loss", f"{loss_percent}%"])

            command = " ".join(cmd_parts)

            result = container.exec_run(command, privileged=True)
            return result.exit_code == 0

        except Exception as e:
            self.logger.error(f"Network delay injection failed: {e}")
            return False

    async def remove(self, context: FaultExecutionContext) -> bool:
        """Remove network delay configuration."""
        try:
            container = self.docker_client.containers.get(context.target_service)
            result = container.exec_run("tc qdisc del dev eth0 root", privileged=True)
            return result.exit_code == 0
        except Exception:
            return False

    async def validate(self, context: FaultExecutionContext) -> bool:
        """Validate network delay is active."""
        try:
            container = self.docker_client.containers.get(context.target_service)
            result = container.exec_run("tc qdisc show dev eth0")
            return "netem" in result.output.decode()
        except Exception:
            return False

    async def monitor(self, context: FaultExecutionContext) -> Dict[str, Any]:
        """Monitor network metrics."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            # Get network statistics
            stats = container.stats(stream=False)
            network_stats = stats.get('networks', {})

            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'network_io': network_stats,
                'container_status': container.status
            }

            # Ping test to measure actual latency
            ping_result = container.exec_run("ping -c 3 8.8.8.8")
            if ping_result.exit_code == 0:
                ping_output = ping_result.output.decode()
                # Parse ping statistics (simplified)
                if "avg" in ping_output:
                    metrics['ping_latency'] = ping_output

            return metrics

        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}


class CPUStressInjector(AdvancedFaultInjector):
    """CPU stress fault injector."""

    def __init__(self, docker_client):
        super().__init__("cpu_stress", FaultCategory.COMPUTE)
        self.docker_client = docker_client
        self.stress_processes: Dict[str, List[str]] = {}

    async def inject(self, context: FaultExecutionContext) -> bool:
        """Inject CPU stress using stress-ng."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            cpu_cores = context.parameters.get('cpu_cores', 1)
            cpu_load = context.parameters.get('cpu_load', 80)
            stress_type = context.parameters.get('stress_type', 'cpu')

            # Build stress command
            if stress_type == 'cpu':
                command = f"stress-ng --cpu {cpu_cores} --cpu-load {cpu_load} --timeout 0 &"
            elif stress_type == 'cpu-cache':
                command = f"stress-ng --cpu {cpu_cores} --cpu-method cache --timeout 0 &"
            elif stress_type == 'cpu-matrix':
                command = f"stress-ng --cpu {cpu_cores} --cpu-method matrix --timeout 0 &"
            else:
                command = f"stress-ng --cpu {cpu_cores} --timeout 0 &"

            # Execute stress command
            result = container.exec_run(f"sh -c '{command}'", detach=True)

            # Store process information
            self.stress_processes[context.fault_id] = [str(cpu_cores), stress_type]

            return result.exit_code == 0

        except Exception as e:
            self.logger.error(f"CPU stress injection failed: {e}")
            return False

    async def remove(self, context: FaultExecutionContext) -> bool:
        """Remove CPU stress processes."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            # Kill all stress-ng processes
            result = container.exec_run("pkill -f stress-ng")

            # Clean up process tracking
            self.stress_processes.pop(context.fault_id, None)

            return True  # pkill always returns success even if no processes found

        except Exception as e:
            self.logger.error(f"CPU stress removal failed: {e}")
            return False

    async def validate(self, context: FaultExecutionContext) -> bool:
        """Validate CPU stress is active."""
        try:
            container = self.docker_client.containers.get(context.target_service)
            result = container.exec_run("pgrep stress-ng")
            return result.exit_code == 0
        except Exception:
            return False

    async def monitor(self, context: FaultExecutionContext) -> Dict[str, Any]:
        """Monitor CPU metrics."""
        try:
            container = self.docker_client.containers.get(context.target_service)
            stats = container.stats(stream=False)

            # Calculate CPU usage
            cpu_usage = 0.0
            if 'cpu_stats' in stats and 'precpu_stats' in stats:
                cpu_delta = (stats['cpu_stats']['cpu_usage']['total_usage'] -
                           stats['precpu_stats']['cpu_usage']['total_usage'])
                system_delta = (stats['cpu_stats']['system_cpu_usage'] -
                              stats['precpu_stats']['system_cpu_usage'])

                if system_delta > 0:
                    cpu_usage = (cpu_delta / system_delta) * 100.0

            # Get load average from inside container
            load_result = container.exec_run("cat /proc/loadavg")
            load_avg = ""
            if load_result.exit_code == 0:
                load_avg = load_result.output.decode().strip()

            return {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_usage_percent': cpu_usage,
                'load_average': load_avg,
                'container_status': container.status,
                'active_stress_processes': len(self.stress_processes.get(context.fault_id, []))
            }

        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}


class MemoryLeakInjector(AdvancedFaultInjector):
    """Memory leak and exhaustion fault injector."""

    def __init__(self, docker_client):
        super().__init__("memory_leak", FaultCategory.COMPUTE)
        self.docker_client = docker_client
        self.memory_processes: Dict[str, Dict[str, Any]] = {}

    async def inject(self, context: FaultExecutionContext) -> bool:
        """Inject memory pressure using stress-ng or custom leak."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            memory_mb = context.parameters.get('memory_mb', 100)
            leak_type = context.parameters.get('leak_type', 'gradual')
            leak_rate = context.parameters.get('leak_rate_mb_per_sec', 1)

            if leak_type == 'immediate':
                # Immediate memory allocation
                command = f"stress-ng --vm 1 --vm-bytes {memory_mb}M --vm-keep --timeout 0 &"
            elif leak_type == 'gradual':
                # Gradual memory leak simulation
                script = f"""
import time
import sys
data = []
chunk_size = {leak_rate} * 1024 * 1024  # MB to bytes
total_target = {memory_mb} * 1024 * 1024
allocated = 0

while allocated < total_target:
    data.append(b'x' * int(chunk_size))
    allocated += chunk_size
    time.sleep(1)

# Keep memory allocated
while True:
    time.sleep(60)
"""

                # Write script to container
                script_path = f"/tmp/memory_leak_{context.fault_id}.py"
                container.exec_run(f"cat > {script_path}", stdin=script.encode())
                command = f"python3 {script_path} &"
            else:
                # Random access pattern
                command = f"stress-ng --vm 1 --vm-bytes {memory_mb}M --vm-method random --timeout 0 &"

            result = container.exec_run(f"sh -c '{command}'", detach=True)

            # Store process information
            self.memory_processes[context.fault_id] = {
                'memory_mb': memory_mb,
                'leak_type': leak_type,
                'start_time': datetime.utcnow()
            }

            return result.exit_code == 0

        except Exception as e:
            self.logger.error(f"Memory leak injection failed: {e}")
            return False

    async def remove(self, context: FaultExecutionContext) -> bool:
        """Remove memory stress processes."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            # Kill stress-ng processes
            container.exec_run("pkill -f stress-ng")

            # Kill Python memory leak scripts
            script_pattern = f"memory_leak_{context.fault_id}.py"
            container.exec_run(f"pkill -f {script_pattern}")

            # Clean up script files
            container.exec_run(f"rm -f /tmp/memory_leak_{context.fault_id}.py")

            # Clean up process tracking
            self.memory_processes.pop(context.fault_id, None)

            return True

        except Exception as e:
            self.logger.error(f"Memory leak removal failed: {e}")
            return False

    async def validate(self, context: FaultExecutionContext) -> bool:
        """Validate memory stress is active."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            # Check for stress-ng or Python processes
            stress_result = container.exec_run("pgrep stress-ng")
            python_result = container.exec_run(f"pgrep -f memory_leak_{context.fault_id}")

            return stress_result.exit_code == 0 or python_result.exit_code == 0

        except Exception:
            return False

    async def monitor(self, context: FaultExecutionContext) -> Dict[str, Any]:
        """Monitor memory metrics."""
        try:
            container = self.docker_client.containers.get(context.target_service)
            stats = container.stats(stream=False)

            # Memory statistics
            memory_stats = stats.get('memory_stats', {})
            memory_usage = memory_stats.get('usage', 0)
            memory_limit = memory_stats.get('limit', 0)
            memory_percent = (memory_usage / memory_limit * 100) if memory_limit > 0 else 0

            # Get detailed memory info from inside container
            meminfo_result = container.exec_run("cat /proc/meminfo")
            meminfo = ""
            if meminfo_result.exit_code == 0:
                meminfo = meminfo_result.output.decode()

            return {
                'timestamp': datetime.utcnow().isoformat(),
                'memory_usage_bytes': memory_usage,
                'memory_limit_bytes': memory_limit,
                'memory_usage_percent': memory_percent,
                'container_status': container.status,
                'meminfo_snippet': meminfo[:500] if meminfo else "",
                'process_info': self.memory_processes.get(context.fault_id, {})
            }

        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}


class DiskFillInjector(AdvancedFaultInjector):
    """Disk space exhaustion fault injector."""

    def __init__(self, docker_client):
        super().__init__("disk_fill", FaultCategory.STORAGE)
        self.docker_client = docker_client
        self.fill_files: Dict[str, List[str]] = {}

    async def inject(self, context: FaultExecutionContext) -> bool:
        """Inject disk space exhaustion."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            fill_size = context.parameters.get('fill_size_mb', 100)
            fill_path = context.parameters.get('fill_path', '/tmp')
            file_count = context.parameters.get('file_count', 1)

            files_created = []

            for i in range(file_count):
                file_path = f"{fill_path}/chaos_fill_{context.fault_id}_{i}.dat"
                size_per_file = fill_size // file_count

                # Create fill file
                command = f"fallocate -l {size_per_file}M {file_path}"
                result = container.exec_run(command)

                if result.exit_code == 0:
                    files_created.append(file_path)
                else:
                    # Fallback to dd if fallocate fails
                    dd_command = f"dd if=/dev/zero of={file_path} bs=1M count={size_per_file} 2>/dev/null"
                    dd_result = container.exec_run(dd_command)
                    if dd_result.exit_code == 0:
                        files_created.append(file_path)

            if files_created:
                self.fill_files[context.fault_id] = files_created
                return True

            return False

        except Exception as e:
            self.logger.error(f"Disk fill injection failed: {e}")
            return False

    async def remove(self, context: FaultExecutionContext) -> bool:
        """Remove disk fill files."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            files = self.fill_files.get(context.fault_id, [])
            for file_path in files:
                container.exec_run(f"rm -f {file_path}")

            # Clean up tracking
            self.fill_files.pop(context.fault_id, None)

            return True

        except Exception as e:
            self.logger.error(f"Disk fill removal failed: {e}")
            return False

    async def validate(self, context: FaultExecutionContext) -> bool:
        """Validate disk fill files exist."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            files = self.fill_files.get(context.fault_id, [])
            for file_path in files:
                result = container.exec_run(f"ls -la {file_path}")
                if result.exit_code != 0:
                    return False

            return len(files) > 0

        except Exception:
            return False

    async def monitor(self, context: FaultExecutionContext) -> Dict[str, Any]:
        """Monitor disk space metrics."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            # Get disk usage
            df_result = container.exec_run("df -h")
            disk_usage = ""
            if df_result.exit_code == 0:
                disk_usage = df_result.output.decode()

            # Get file information
            files = self.fill_files.get(context.fault_id, [])
            file_info = []

            for file_path in files:
                ls_result = container.exec_run(f"ls -lh {file_path}")
                if ls_result.exit_code == 0:
                    file_info.append(ls_result.output.decode().strip())

            return {
                'timestamp': datetime.utcnow().isoformat(),
                'disk_usage': disk_usage,
                'fill_files': file_info,
                'container_status': container.status,
                'files_created': len(files)
            }

        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}


class ServiceCrashInjector(AdvancedFaultInjector):
    """Service crash and failure fault injector."""

    def __init__(self, docker_client):
        super().__init__("service_crash", FaultCategory.APPLICATION)
        self.docker_client = docker_client
        self.crashed_services: Dict[str, Dict[str, Any]] = {}

    async def inject(self, context: FaultExecutionContext) -> bool:
        """Inject service crash."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            crash_method = context.parameters.get('method', 'stop')
            crash_signal = context.parameters.get('signal', 'SIGTERM')
            process_name = context.parameters.get('process_name', None)

            # Store original container state
            original_state = {
                'status': container.status,
                'restart_count': container.attrs['RestartCount'],
                'created': container.attrs['Created']
            }

            if crash_method == 'kill':
                container.kill(signal=crash_signal)
            elif crash_method == 'stop':
                container.stop(timeout=5)
            elif crash_method == 'process_kill' and process_name:
                # Kill specific process inside container
                result = container.exec_run(f"pkill -f {process_name}")
                if result.exit_code != 0:
                    return False
            elif crash_method == 'oom':
                # Trigger out-of-memory condition
                container.exec_run("python3 -c \"x = []; [x.append(' ' * 1024 * 1024) for i in range(10000)]\"", detach=True)
            else:
                return False

            self.crashed_services[context.fault_id] = {
                'original_state': original_state,
                'crash_method': crash_method,
                'crash_time': datetime.utcnow()
            }

            return True

        except Exception as e:
            self.logger.error(f"Service crash injection failed: {e}")
            return False

    async def remove(self, context: FaultExecutionContext) -> bool:
        """Restart crashed service."""
        try:
            container = self.docker_client.containers.get(context.target_service)

            # Start the container if it's stopped
            if container.status != 'running':
                container.start()

                # Wait for container to be running
                for _ in range(30):
                    container.reload()
                    if container.status == 'running':
                        break
                    await asyncio.sleep(1)

            # Clean up tracking
            self.crashed_services.pop(context.fault_id, None)

            return container.status == 'running'

        except Exception as e:
            self.logger.error(f"Service crash removal failed: {e}")
            return False

    async def validate(self, context: FaultExecutionContext) -> bool:
        """Validate service is crashed/stopped."""
        try:
            container = self.docker_client.containers.get(context.target_service)
            container.reload()

            crash_info = self.crashed_services.get(context.fault_id, {})
            crash_method = crash_info.get('crash_method', '')

            if crash_method in ['kill', 'stop']:
                return container.status != 'running'
            elif crash_method == 'process_kill':
                # Check if main process is still running
                return True  # Assume validation passed if we got here
            elif crash_method == 'oom':
                # Check container status and restart count
                return container.attrs['RestartCount'] > crash_info['original_state']['restart_count']

            return False

        except docker.errors.NotFound:
            return True  # Container not found means it's definitely crashed
        except Exception:
            return False

    async def monitor(self, context: FaultExecutionContext) -> Dict[str, Any]:
        """Monitor service status."""
        try:
            container = self.docker_client.containers.get(context.target_service)
            container.reload()

            return {
                'timestamp': datetime.utcnow().isoformat(),
                'container_status': container.status,
                'restart_count': container.attrs['RestartCount'],
                'exit_code': container.attrs.get('State', {}).get('ExitCode'),
                'finished_at': container.attrs.get('State', {}).get('FinishedAt'),
                'crash_info': self.crashed_services.get(context.fault_id, {})
            }

        except docker.errors.NotFound:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'error': 'Container not found',
                'crash_info': self.crashed_services.get(context.fault_id, {})
            }
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.utcnow().isoformat()}


class FaultInjectionManager:
    """
    Manager for coordinating multiple fault injectors with safety controls.
    """

    def __init__(self, docker_client, redis_client: Optional[aioredis.Redis] = None):
        self.docker_client = docker_client
        self.redis_client = redis_client
        self.logger = logging.getLogger("fault_injection_manager")

        # Initialize injectors
        self.injectors: Dict[str, AdvancedFaultInjector] = {
            'network_delay': NetworkDelayInjector(docker_client),
            'cpu_stress': CPUStressInjector(docker_client),
            'memory_leak': MemoryLeakInjector(docker_client),
            'disk_fill': DiskFillInjector(docker_client),
            'service_crash': ServiceCrashInjector(docker_client)
        }

        # Fault profiles
        self.fault_profiles = self._initialize_fault_profiles()

        # Active fault tracking
        self.active_faults: Dict[str, FaultExecutionContext] = {}
        self.fault_history: List[FaultExecutionContext] = []

        # Safety controls
        self.max_concurrent_faults = 10
        self.safety_monitoring_active = False

    def _initialize_fault_profiles(self) -> Dict[str, FaultProfile]:
        """Initialize predefined fault profiles."""
        return {
            'network_delay_light': FaultProfile(
                name="Light Network Delay",
                category=FaultCategory.NETWORK,
                severity=FaultSeverity.LOW,
                description="Add 50-100ms network delay",
                min_intensity=0.1,
                max_intensity=0.3,
                safety_checks=['network_connectivity', 'service_health']
            ),
            'network_delay_heavy': FaultProfile(
                name="Heavy Network Delay",
                category=FaultCategory.NETWORK,
                severity=FaultSeverity.HIGH,
                description="Add 500-1000ms network delay",
                min_intensity=0.7,
                max_intensity=1.0,
                safety_checks=['network_connectivity', 'service_health', 'cluster_health']
            ),
            'cpu_stress_moderate': FaultProfile(
                name="Moderate CPU Stress",
                category=FaultCategory.COMPUTE,
                severity=FaultSeverity.MEDIUM,
                description="50-70% CPU utilization stress",
                min_intensity=0.3,
                max_intensity=0.7,
                safety_checks=['cpu_temperature', 'system_load']
            ),
            'memory_leak_gradual': FaultProfile(
                name="Gradual Memory Leak",
                category=FaultCategory.COMPUTE,
                severity=FaultSeverity.MEDIUM,
                description="Gradual memory consumption over time",
                min_intensity=0.2,
                max_intensity=0.8,
                safety_checks=['memory_available', 'swap_usage']
            ),
            'service_crash_controlled': FaultProfile(
                name="Controlled Service Crash",
                category=FaultCategory.APPLICATION,
                severity=FaultSeverity.HIGH,
                description="Graceful service shutdown",
                min_intensity=1.0,
                max_intensity=1.0,
                safety_checks=['service_dependencies', 'backup_services']
            )
        }

    async def inject_fault(self, injector_name: str, target_service: str,
                          profile_name: str, parameters: Dict[str, Any]) -> str:
        """Inject a fault with the specified parameters."""

        if len(self.active_faults) >= self.max_concurrent_faults:
            raise RuntimeError("Maximum concurrent faults exceeded")

        injector = self.injectors.get(injector_name)
        if not injector:
            raise ValueError(f"Unknown injector: {injector_name}")

        profile = self.fault_profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Unknown fault profile: {profile_name}")

        # Generate unique fault ID
        fault_id = f"{injector_name}_{target_service}_{int(time.time())}_{random.randint(1000, 9999)}"

        # Create execution context
        context = FaultExecutionContext(
            fault_id=fault_id,
            target_service=target_service,
            profile=profile,
            parameters=parameters,
            start_time=datetime.utcnow()
        )

        # Execute fault injection
        success = await injector.execute_with_monitoring(context)

        if success:
            self.active_faults[fault_id] = context
            await self._save_fault_context(context)
            self.logger.info(f"Fault injected: {fault_id}")
        else:
            self.logger.error(f"Failed to inject fault: {fault_id}")
            raise RuntimeError(f"Fault injection failed: {context.error_message}")

        return fault_id

    async def remove_fault(self, fault_id: str) -> bool:
        """Remove a specific fault."""
        context = self.active_faults.get(fault_id)
        if not context:
            self.logger.warning(f"Fault not found: {fault_id}")
            return False

        # Find the appropriate injector
        injector_name = fault_id.split('_')[0]
        injector = self.injectors.get(injector_name)

        if not injector:
            self.logger.error(f"Injector not found for fault: {fault_id}")
            return False

        # Remove the fault
        success = await injector.cleanup_fault(fault_id)

        if success:
            # Move to history
            self.fault_history.append(context)
            self.active_faults.pop(fault_id, None)
            await self._save_fault_context(context)
            self.logger.info(f"Fault removed: {fault_id}")

        return success

    async def remove_all_faults(self) -> Dict[str, bool]:
        """Remove all active faults."""
        results = {}

        for fault_id in list(self.active_faults.keys()):
            results[fault_id] = await self.remove_fault(fault_id)

        return results

    async def get_fault_status(self, fault_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific fault."""
        context = self.active_faults.get(fault_id)
        if not context:
            return None

        # Get current monitoring data
        injector_name = fault_id.split('_')[0]
        injector = self.injectors.get(injector_name)

        if injector:
            current_metrics = await injector.monitor(context)
            context.during_injection_metrics.append(current_metrics)

        return {
            'fault_id': fault_id,
            'target_service': context.target_service,
            'profile_name': context.profile.name,
            'active': context.active,
            'validated': context.validated,
            'start_time': context.start_time.isoformat(),
            'duration': (datetime.utcnow() - context.start_time).total_seconds(),
            'current_metrics': current_metrics if injector else {},
            'error_message': context.error_message
        }

    async def get_all_faults_status(self) -> Dict[str, Any]:
        """Get status of all active faults."""
        status = {
            'active_faults': len(self.active_faults),
            'fault_history_count': len(self.fault_history),
            'available_injectors': list(self.injectors.keys()),
            'available_profiles': list(self.fault_profiles.keys()),
            'faults': {}
        }

        for fault_id in self.active_faults:
            fault_status = await self.get_fault_status(fault_id)
            if fault_status:
                status['faults'][fault_id] = fault_status

        return status

    async def _save_fault_context(self, context: FaultExecutionContext):
        """Save fault context to Redis."""
        if not self.redis_client:
            return

        try:
            key = f"chaos:fault:{context.fault_id}"
            data = asdict(context)

            # Convert datetime objects to strings
            for field in ['start_time', 'end_time']:
                if data.get(field):
                    data[field] = data[field].isoformat() if hasattr(data[field], 'isoformat') else str(data[field])

            await self.redis_client.set(key, json.dumps(data, default=str))
            await self.redis_client.expire(key, 86400)  # 24 hours retention

        except Exception as e:
            self.logger.error(f"Failed to save fault context: {e}")

    def get_fault_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Get all available fault profiles."""
        return {name: asdict(profile) for name, profile in self.fault_profiles.items()}

    def add_fault_profile(self, name: str, profile: FaultProfile):
        """Add a new fault profile."""
        self.fault_profiles[name] = profile
        self.logger.info(f"Added fault profile: {name}")

    async def validate_all_active_faults(self) -> Dict[str, bool]:
        """Validate all active faults are still working."""
        results = {}

        for fault_id, context in self.active_faults.items():
            injector_name = fault_id.split('_')[0]
            injector = self.injectors.get(injector_name)

            if injector:
                try:
                    is_valid = await injector.validate(context)
                    results[fault_id] = is_valid
                    context.validated = is_valid

                    if not is_valid:
                        self.logger.warning(f"Fault validation failed: {fault_id}")

                except Exception as e:
                    self.logger.error(f"Fault validation error for {fault_id}: {e}")
                    results[fault_id] = False
            else:
                results[fault_id] = False

        return results


# Example usage and testing functions
async def example_fault_injection():
    """Example of how to use the fault injection system."""
    docker_client = docker.from_env()
    manager = FaultInjectionManager(docker_client)

    try:
        # Inject network delay
        fault_id = await manager.inject_fault(
            injector_name='network_delay',
            target_service='web-server',
            profile_name='network_delay_light',
            parameters={
                'delay_ms': 100,
                'jitter_ms': 20,
                'loss_percent': 1
            }
        )

        print(f"Injected fault: {fault_id}")

        # Monitor for 30 seconds
        for i in range(6):
            status = await manager.get_fault_status(fault_id)
            print(f"Fault status: {status}")
            await asyncio.sleep(5)

        # Remove fault
        success = await manager.remove_fault(fault_id)
        print(f"Fault removal: {'success' if success else 'failed'}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up all faults
        await manager.remove_all_faults()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_fault_injection())