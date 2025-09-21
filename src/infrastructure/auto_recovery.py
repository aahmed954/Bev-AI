"""
Auto-Recovery System for BEV OSINT Framework
===========================================

Comprehensive auto-recovery system with circuit breaker integration,
multiple recovery strategies, and state preservation capabilities.

Features:
- Multiple recovery strategies: restart, rollback, circuit break
- Intelligent service discovery and health monitoring
- State preservation and rollback capabilities
- Integration with Docker container orchestration
- Advanced logging and alerting system
- Performance monitoring and SLA compliance

Author: BEV Infrastructure Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
import os
import signal
import shutil
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import aiohttp
import docker
import redis
import psycopg2
from sqlalchemy import create_engine, text
import yaml
import consul
from kubernetes import client, config as k8s_config
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RESTART = "restart"                    # Simple service restart
    ROLLBACK = "rollback"                 # Rollback to previous state
    CIRCUIT_BREAK = "circuit_break"       # Enable circuit breaker protection
    SCALE_UP = "scale_up"                 # Increase resource allocation
    SCALE_DOWN = "scale_down"             # Reduce resource allocation
    FAILOVER = "failover"                 # Switch to backup instance
    RECREATE = "recreate"                 # Full container recreation
    HYBRID = "hybrid"                     # Combination of strategies


class ServiceState(Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    RECOVERING = "recovering"
    FAILED = "failed"


class RecoveryResult(Enum):
    """Recovery operation results."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ServiceConfig:
    """Configuration for a service in the auto-recovery system."""
    name: str
    container_name: str
    image: str

    # Health check configuration
    health_check_url: Optional[str] = None
    health_check_command: Optional[str] = None
    health_check_interval: float = 30.0
    health_check_timeout: float = 10.0
    health_check_retries: int = 3

    # Recovery configuration
    recovery_strategies: List[RecoveryStrategy] = field(default_factory=lambda: [RecoveryStrategy.RESTART])
    max_recovery_attempts: int = 3
    recovery_timeout: float = 300.0
    backoff_multiplier: float = 2.0

    # Dependency configuration
    dependencies: List[str] = field(default_factory=list)
    dependents: List[str] = field(default_factory=list)

    # Resource limits
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    restart_policy: str = "unless-stopped"

    # Circuit breaker configuration
    circuit_breaker_enabled: bool = True
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None

    # State management
    state_backup_enabled: bool = True
    state_backup_paths: List[str] = field(default_factory=list)
    data_volume_paths: List[str] = field(default_factory=list)

    # Priority and criticality
    criticality: str = "normal"  # critical, high, normal, low
    priority: int = 50  # 0-100, higher = more important

    # SLA requirements
    target_availability: float = 99.9  # Target availability percentage
    max_downtime_minutes: float = 5.0  # Maximum acceptable downtime

    # Alerting
    alert_channels: List[str] = field(default_factory=list)
    escalation_levels: List[str] = field(default_factory=list)


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""
    service_name: str
    strategy: RecoveryStrategy
    timestamp: datetime
    result: RecoveryResult
    duration: float
    error_message: Optional[str] = None
    previous_state: Optional[ServiceState] = None
    new_state: Optional[ServiceState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceStateSnapshot:
    """Snapshot of service state for rollback purposes."""
    service_name: str
    timestamp: datetime
    container_id: str
    image_id: str
    environment: Dict[str, str]
    volumes: Dict[str, str]
    network_settings: Dict[str, Any]
    resource_limits: Dict[str, Any]
    labels: Dict[str, str]
    backup_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AutoRecoverySystem:
    """
    Main auto-recovery system with comprehensive service management.

    Provides intelligent recovery strategies, state management,
    and integration with container orchestration systems.
    """

    def __init__(self,
                 config_path: str = "/app/config/auto_recovery.yaml",
                 redis_url: str = "redis://redis:6379/11",
                 postgres_url: str = None,
                 consul_host: str = "consul:8500",
                 docker_socket: str = "unix://var/run/docker.sock"):
        """
        Initialize the auto-recovery system.

        Args:
            config_path: Path to configuration file
            redis_url: Redis connection URL for state storage
            postgres_url: PostgreSQL connection URL for persistence
            consul_host: Consul host for service discovery
            docker_socket: Docker socket path
        """
        self.config_path = Path(config_path)
        self.redis_url = redis_url
        self.postgres_url = postgres_url

        # Initialize connections
        self.redis_client = redis.from_url(redis_url)
        self.docker_client = docker.from_env()
        self.consul_client = consul.Consul(host=consul_host.split(':')[0])

        # Initialize PostgreSQL if available
        self.postgres_engine = None
        if postgres_url:
            self.postgres_engine = create_engine(postgres_url)
            self._init_database()

        # Service management
        self.services: Dict[str, ServiceConfig] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.service_states: Dict[str, ServiceState] = {}
        self.recovery_history: List[RecoveryAttempt] = []
        self.state_snapshots: Dict[str, List[ServiceStateSnapshot]] = {}

        # Monitoring and control
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.recovery_locks: Dict[str, asyncio.Lock] = {}
        self.shutdown_event = asyncio.Event()

        # Configuration
        self.max_concurrent_recoveries = 5
        self.global_recovery_timeout = 600.0  # 10 minutes
        self.snapshot_retention_days = 7
        self.metrics_retention_days = 30

        # Logging
        self.logger = logging.getLogger("auto_recovery")
        self.logger.setLevel(logging.INFO)

        # Load configuration
        self._load_configuration()

        # Initialize services
        self._initialize_services()

    def _init_database(self):
        """Initialize PostgreSQL database tables."""
        if not self.postgres_engine:
            return

        try:
            with self.postgres_engine.connect() as conn:
                # Create tables for recovery tracking
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS recovery_attempts (
                        id SERIAL PRIMARY KEY,
                        service_name VARCHAR(255) NOT NULL,
                        strategy VARCHAR(50) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        result VARCHAR(50) NOT NULL,
                        duration FLOAT NOT NULL,
                        error_message TEXT,
                        previous_state VARCHAR(50),
                        new_state VARCHAR(50),
                        metadata JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """))

                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS service_snapshots (
                        id SERIAL PRIMARY KEY,
                        service_name VARCHAR(255) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        container_id VARCHAR(255),
                        image_id VARCHAR(255),
                        snapshot_data JSONB NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """))

                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_recovery_service_time
                    ON recovery_attempts(service_name, timestamp)
                """))

                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_snapshots_service_time
                    ON service_snapshots(service_name, timestamp)
                """))

                conn.commit()
                self.logger.info("Database tables initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")

    def _load_configuration(self):
        """Load service configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)

                # Load services configuration
                for service_data in config_data.get('services', []):
                    service_config = ServiceConfig(**service_data)
                    self.services[service_config.name] = service_config

                # Load global settings
                global_config = config_data.get('global', {})
                self.max_concurrent_recoveries = global_config.get('max_concurrent_recoveries', 5)
                self.global_recovery_timeout = global_config.get('global_recovery_timeout', 600.0)
                self.snapshot_retention_days = global_config.get('snapshot_retention_days', 7)

                self.logger.info(f"Loaded configuration for {len(self.services)} services")
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def _initialize_services(self):
        """Initialize circuit breakers and recovery locks for all services."""
        for service_name, service_config in self.services.items():
            # Initialize circuit breaker
            if service_config.circuit_breaker_enabled:
                cb_config = service_config.circuit_breaker_config or CircuitBreakerConfig()
                circuit_breaker = CircuitBreaker(
                    service_name=service_name,
                    config=cb_config,
                    redis_client=self.redis_client
                )

                # Set health check URL if available
                if service_config.health_check_url:
                    circuit_breaker.set_health_check_url(service_config.health_check_url)

                self.circuit_breakers[service_name] = circuit_breaker

            # Initialize recovery lock
            self.recovery_locks[service_name] = asyncio.Lock()

            # Initialize service state
            self.service_states[service_name] = ServiceState.UNKNOWN

            # Initialize snapshot storage
            self.state_snapshots[service_name] = []

    async def start_monitoring(self):
        """Start health monitoring for all services."""
        self.logger.info("Starting auto-recovery monitoring system")

        # Start health monitoring tasks for each service
        for service_name, service_config in self.services.items():
            task = asyncio.create_task(
                self._monitor_service_health(service_name, service_config)
            )
            self.monitoring_tasks[service_name] = task

        # Start circuit breaker health monitoring
        for service_name, circuit_breaker in self.circuit_breakers.items():
            await circuit_breaker.start_health_monitoring()

        # Start background maintenance tasks
        asyncio.create_task(self._cleanup_old_data())
        asyncio.create_task(self._update_service_discovery())

        self.logger.info("Auto-recovery monitoring started successfully")

    async def stop_monitoring(self):
        """Stop all monitoring tasks."""
        self.logger.info("Stopping auto-recovery monitoring system")

        # Signal shutdown
        self.shutdown_event.set()

        # Cancel monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()

        # Stop circuit breaker monitoring
        for circuit_breaker in self.circuit_breakers.values():
            circuit_breaker.stop_health_monitoring()

        # Wait for tasks to complete
        await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)

        self.logger.info("Auto-recovery monitoring stopped")

    async def _monitor_service_health(self, service_name: str, service_config: ServiceConfig):
        """Monitor health of a specific service."""
        self.logger.info(f"Starting health monitoring for {service_name}")

        while not self.shutdown_event.is_set():
            try:
                # Perform health check
                health_status = await self._check_service_health(service_name, service_config)
                previous_state = self.service_states.get(service_name, ServiceState.UNKNOWN)
                self.service_states[service_name] = health_status

                # Log state changes
                if previous_state != health_status:
                    self.logger.info(f"Service {service_name} state changed: {previous_state.value} -> {health_status.value}")

                # Trigger recovery if needed
                if health_status in [ServiceState.UNHEALTHY, ServiceState.FAILED]:
                    await self._trigger_recovery(service_name, service_config, health_status)
                elif health_status == ServiceState.HEALTHY and previous_state == ServiceState.RECOVERING:
                    self.logger.info(f"Service {service_name} successfully recovered")
                    await self._create_state_snapshot(service_name)

                # Wait for next check
                await asyncio.sleep(service_config.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error monitoring {service_name}: {e}")
                await asyncio.sleep(30)  # Back off on errors

    async def _check_service_health(self, service_name: str, service_config: ServiceConfig) -> ServiceState:
        """Check health of a specific service."""
        try:
            # Check Docker container status
            container_healthy = await self._check_container_health(service_config.container_name)
            if not container_healthy:
                return ServiceState.FAILED

            # Perform HTTP health check if configured
            if service_config.health_check_url:
                http_healthy = await self._perform_http_health_check(
                    service_config.health_check_url,
                    service_config.health_check_timeout
                )
                if not http_healthy:
                    return ServiceState.UNHEALTHY

            # Perform command-based health check if configured
            if service_config.health_check_command:
                command_healthy = await self._perform_command_health_check(
                    service_config.container_name,
                    service_config.health_check_command
                )
                if not command_healthy:
                    return ServiceState.UNHEALTHY

            # Check circuit breaker state
            if service_name in self.circuit_breakers:
                cb_state = self.circuit_breakers[service_name].state
                if cb_state == CircuitBreakerState.OPEN:
                    return ServiceState.DEGRADED

            return ServiceState.HEALTHY

        except Exception as e:
            self.logger.error(f"Health check failed for {service_name}: {e}")
            return ServiceState.UNKNOWN

    async def _check_container_health(self, container_name: str) -> bool:
        """Check if Docker container is running and healthy."""
        try:
            container = self.docker_client.containers.get(container_name)

            # Check if container is running
            if container.status != 'running':
                return False

            # Check Docker health check if available
            health = container.attrs.get('State', {}).get('Health', {})
            if health and health.get('Status') == 'unhealthy':
                return False

            return True

        except docker.errors.NotFound:
            self.logger.warning(f"Container {container_name} not found")
            return False
        except Exception as e:
            self.logger.error(f"Error checking container {container_name}: {e}")
            return False

    async def _perform_http_health_check(self, url: str, timeout: float) -> bool:
        """Perform HTTP health check."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _perform_command_health_check(self, container_name: str, command: str) -> bool:
        """Perform command-based health check inside container."""
        try:
            container = self.docker_client.containers.get(container_name)
            result = container.exec_run(command, demux=True)
            return result.exit_code == 0
        except Exception:
            return False

    async def _trigger_recovery(self, service_name: str, service_config: ServiceConfig, current_state: ServiceState):
        """Trigger recovery process for a failed service."""
        # Check if recovery is already in progress
        async with self.recovery_locks[service_name]:
            if self.service_states[service_name] == ServiceState.RECOVERING:
                return

            self.service_states[service_name] = ServiceState.RECOVERING

        self.logger.warning(f"Triggering recovery for {service_name} (state: {current_state.value})")

        # Execute recovery strategies
        recovery_success = False
        for strategy in service_config.recovery_strategies:
            try:
                result = await self._execute_recovery_strategy(
                    service_name, service_config, strategy, current_state
                )

                if result.result == RecoveryResult.SUCCESS:
                    recovery_success = True
                    break
                elif result.result == RecoveryResult.PARTIAL_SUCCESS:
                    # Continue with next strategy
                    continue

            except Exception as e:
                self.logger.error(f"Recovery strategy {strategy.value} failed for {service_name}: {e}")

        if not recovery_success:
            self.logger.error(f"All recovery strategies failed for {service_name}")
            self.service_states[service_name] = ServiceState.FAILED
            await self._send_alert(service_name, "Recovery failed", "critical")

    async def _execute_recovery_strategy(self,
                                       service_name: str,
                                       service_config: ServiceConfig,
                                       strategy: RecoveryStrategy,
                                       previous_state: ServiceState) -> RecoveryAttempt:
        """Execute a specific recovery strategy."""
        start_time = time.time()
        attempt = RecoveryAttempt(
            service_name=service_name,
            strategy=strategy,
            timestamp=datetime.utcnow(),
            result=RecoveryResult.FAILURE,
            duration=0.0,
            previous_state=previous_state
        )

        try:
            self.logger.info(f"Executing {strategy.value} recovery for {service_name}")

            # Create snapshot before recovery
            if service_config.state_backup_enabled:
                await self._create_state_snapshot(service_name)

            # Execute strategy
            if strategy == RecoveryStrategy.RESTART:
                success = await self._restart_service(service_name, service_config)
            elif strategy == RecoveryStrategy.ROLLBACK:
                success = await self._rollback_service(service_name, service_config)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAK:
                success = await self._enable_circuit_breaker(service_name)
            elif strategy == RecoveryStrategy.SCALE_UP:
                success = await self._scale_service(service_name, service_config, scale_up=True)
            elif strategy == RecoveryStrategy.SCALE_DOWN:
                success = await self._scale_service(service_name, service_config, scale_up=False)
            elif strategy == RecoveryStrategy.FAILOVER:
                success = await self._failover_service(service_name, service_config)
            elif strategy == RecoveryStrategy.RECREATE:
                success = await self._recreate_service(service_name, service_config)
            elif strategy == RecoveryStrategy.HYBRID:
                success = await self._hybrid_recovery(service_name, service_config)
            else:
                raise ValueError(f"Unknown recovery strategy: {strategy}")

            # Determine result
            if success:
                attempt.result = RecoveryResult.SUCCESS
                attempt.new_state = ServiceState.HEALTHY
            else:
                attempt.result = RecoveryResult.FAILURE
                attempt.new_state = ServiceState.FAILED

            # Wait for service to stabilize and verify recovery
            await asyncio.sleep(10)
            final_state = await self._check_service_health(service_name, service_config)
            attempt.new_state = final_state

            if final_state == ServiceState.HEALTHY:
                attempt.result = RecoveryResult.SUCCESS
            elif final_state in [ServiceState.DEGRADED, ServiceState.RECOVERING]:
                attempt.result = RecoveryResult.PARTIAL_SUCCESS

        except asyncio.TimeoutError:
            attempt.result = RecoveryResult.TIMEOUT
            attempt.error_message = f"Recovery timeout after {service_config.recovery_timeout}s"

        except Exception as e:
            attempt.result = RecoveryResult.FAILURE
            attempt.error_message = str(e)
            self.logger.error(f"Recovery strategy {strategy.value} failed for {service_name}: {e}")

        finally:
            attempt.duration = time.time() - start_time

            # Record recovery attempt
            self.recovery_history.append(attempt)
            await self._save_recovery_attempt(attempt)

            self.logger.info(
                f"Recovery {strategy.value} for {service_name}: {attempt.result.value} "
                f"({attempt.duration:.2f}s)"
            )

        return attempt

    async def _restart_service(self, service_name: str, service_config: ServiceConfig) -> bool:
        """Restart a service using Docker."""
        try:
            container = self.docker_client.containers.get(service_config.container_name)

            self.logger.info(f"Restarting container {service_config.container_name}")
            container.restart(timeout=30)

            # Wait for container to be running
            for _ in range(10):
                container.reload()
                if container.status == 'running':
                    return True
                await asyncio.sleep(2)

            return False

        except Exception as e:
            self.logger.error(f"Failed to restart {service_name}: {e}")
            return False

    async def _rollback_service(self, service_name: str, service_config: ServiceConfig) -> bool:
        """Rollback service to previous healthy state."""
        try:
            # Find latest healthy snapshot
            snapshots = self.state_snapshots.get(service_name, [])
            if not snapshots:
                self.logger.warning(f"No snapshots available for {service_name} rollback")
                return False

            # Get the most recent snapshot
            snapshot = snapshots[-1]

            self.logger.info(f"Rolling back {service_name} to snapshot from {snapshot.timestamp}")

            # Stop current container
            container = self.docker_client.containers.get(service_config.container_name)
            container.stop(timeout=30)
            container.remove()

            # Restore data volumes if configured
            for backup_path in snapshot.backup_paths:
                if os.path.exists(backup_path):
                    restore_path = backup_path.replace('.backup', '')
                    shutil.copytree(backup_path, restore_path, dirs_exist_ok=True)

            # Recreate container with snapshot configuration
            new_container = self.docker_client.containers.run(
                image=snapshot.image_id,
                name=service_config.container_name,
                environment=snapshot.environment,
                volumes=snapshot.volumes,
                labels=snapshot.labels,
                detach=True,
                restart_policy={'Name': service_config.restart_policy}
            )

            # Wait for container to be running
            for _ in range(15):
                new_container.reload()
                if new_container.status == 'running':
                    return True
                await asyncio.sleep(2)

            return False

        except Exception as e:
            self.logger.error(f"Failed to rollback {service_name}: {e}")
            return False

    async def _enable_circuit_breaker(self, service_name: str) -> bool:
        """Enable circuit breaker protection for a service."""
        try:
            if service_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[service_name]
                circuit_breaker.reset()
                self.logger.info(f"Circuit breaker enabled for {service_name}")
                return True
            else:
                self.logger.warning(f"No circuit breaker configured for {service_name}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to enable circuit breaker for {service_name}: {e}")
            return False

    async def _scale_service(self, service_name: str, service_config: ServiceConfig, scale_up: bool) -> bool:
        """Scale service resources up or down."""
        try:
            container = self.docker_client.containers.get(service_config.container_name)

            # Get current resource limits
            current_config = container.attrs['HostConfig']

            if scale_up:
                # Increase CPU and memory limits
                new_cpu_limit = int(current_config.get('CpuQuota', 100000) * 1.5)
                new_memory_limit = int(current_config.get('Memory', 1073741824) * 1.5)
                self.logger.info(f"Scaling up {service_name}")
            else:
                # Decrease CPU and memory limits
                new_cpu_limit = int(current_config.get('CpuQuota', 100000) * 0.75)
                new_memory_limit = int(current_config.get('Memory', 1073741824) * 0.75)
                self.logger.info(f"Scaling down {service_name}")

            # Update container configuration
            container.update(
                cpu_quota=new_cpu_limit,
                mem_limit=new_memory_limit
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to scale {service_name}: {e}")
            return False

    async def _failover_service(self, service_name: str, service_config: ServiceConfig) -> bool:
        """Implement failover to backup instance."""
        try:
            # This would typically involve:
            # 1. Starting a backup instance
            # 2. Updating load balancer configuration
            # 3. Switching traffic to backup

            # For now, implement as container recreation on different node
            self.logger.info(f"Executing failover for {service_name}")

            # Stop current container
            container = self.docker_client.containers.get(service_config.container_name)
            container.stop(timeout=30)

            # Start new container with failover configuration
            new_container = self.docker_client.containers.run(
                image=service_config.image,
                name=f"{service_config.container_name}_failover",
                environment={"FAILOVER_MODE": "true"},
                detach=True,
                restart_policy={'Name': service_config.restart_policy}
            )

            # Wait for new container to be ready
            for _ in range(15):
                new_container.reload()
                if new_container.status == 'running':
                    # Remove old container
                    container.remove()
                    return True
                await asyncio.sleep(2)

            return False

        except Exception as e:
            self.logger.error(f"Failed to failover {service_name}: {e}")
            return False

    async def _recreate_service(self, service_name: str, service_config: ServiceConfig) -> bool:
        """Completely recreate service container."""
        try:
            self.logger.info(f"Recreating container for {service_name}")

            # Get current container configuration
            container = self.docker_client.containers.get(service_config.container_name)
            container_config = container.attrs

            # Stop and remove current container
            container.stop(timeout=30)
            container.remove()

            # Pull latest image
            self.docker_client.images.pull(service_config.image)

            # Recreate container with fresh configuration
            new_container = self.docker_client.containers.run(
                image=service_config.image,
                name=service_config.container_name,
                environment=container_config['Config'].get('Env', []),
                volumes=container_config['Mounts'],
                ports=container_config['NetworkSettings'].get('Ports', {}),
                labels=container_config['Config'].get('Labels', {}),
                detach=True,
                restart_policy={'Name': service_config.restart_policy}
            )

            # Wait for container to be running
            for _ in range(20):
                new_container.reload()
                if new_container.status == 'running':
                    return True
                await asyncio.sleep(3)

            return False

        except Exception as e:
            self.logger.error(f"Failed to recreate {service_name}: {e}")
            return False

    async def _hybrid_recovery(self, service_name: str, service_config: ServiceConfig) -> bool:
        """Execute hybrid recovery strategy combining multiple approaches."""
        try:
            self.logger.info(f"Executing hybrid recovery for {service_name}")

            # Step 1: Enable circuit breaker
            await self._enable_circuit_breaker(service_name)

            # Step 2: Try restart first
            if await self._restart_service(service_name, service_config):
                return True

            # Step 3: If restart fails, try scaling up
            if await self._scale_service(service_name, service_config, scale_up=True):
                return True

            # Step 4: If scaling fails, try rollback
            if await self._rollback_service(service_name, service_config):
                return True

            # Step 5: Last resort - recreate
            return await self._recreate_service(service_name, service_config)

        except Exception as e:
            self.logger.error(f"Hybrid recovery failed for {service_name}: {e}")
            return False

    async def _create_state_snapshot(self, service_name: str):
        """Create a state snapshot for rollback purposes."""
        try:
            service_config = self.services.get(service_name)
            if not service_config or not service_config.state_backup_enabled:
                return

            container = self.docker_client.containers.get(service_config.container_name)

            # Create snapshot
            snapshot = ServiceStateSnapshot(
                service_name=service_name,
                timestamp=datetime.utcnow(),
                container_id=container.id,
                image_id=container.image.id,
                environment=dict(container.attrs['Config'].get('Env', [])),
                volumes={mount['Source']: mount['Destination']
                        for mount in container.attrs.get('Mounts', [])},
                network_settings=container.attrs['NetworkSettings'],
                resource_limits=container.attrs['HostConfig'],
                labels=container.attrs['Config'].get('Labels', {})
            )

            # Backup data volumes
            backup_paths = []
            for path in service_config.state_backup_paths:
                if os.path.exists(path):
                    backup_path = f"{path}.backup.{int(time.time())}"
                    shutil.copytree(path, backup_path, dirs_exist_ok=True)
                    backup_paths.append(backup_path)

            snapshot.backup_paths = backup_paths

            # Store snapshot
            snapshots = self.state_snapshots.get(service_name, [])
            snapshots.append(snapshot)

            # Keep only recent snapshots
            snapshots = snapshots[-10:]  # Keep last 10 snapshots
            self.state_snapshots[service_name] = snapshots

            # Save to database
            await self._save_state_snapshot(snapshot)

            self.logger.info(f"Created state snapshot for {service_name}")

        except Exception as e:
            self.logger.error(f"Failed to create snapshot for {service_name}: {e}")

    async def _save_recovery_attempt(self, attempt: RecoveryAttempt):
        """Save recovery attempt to database."""
        if not self.postgres_engine:
            return

        try:
            with self.postgres_engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO recovery_attempts
                    (service_name, strategy, timestamp, result, duration,
                     error_message, previous_state, new_state, metadata)
                    VALUES
                    (:service_name, :strategy, :timestamp, :result, :duration,
                     :error_message, :previous_state, :new_state, :metadata)
                """), {
                    'service_name': attempt.service_name,
                    'strategy': attempt.strategy.value,
                    'timestamp': attempt.timestamp,
                    'result': attempt.result.value,
                    'duration': attempt.duration,
                    'error_message': attempt.error_message,
                    'previous_state': attempt.previous_state.value if attempt.previous_state else None,
                    'new_state': attempt.new_state.value if attempt.new_state else None,
                    'metadata': json.dumps(attempt.metadata)
                })
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save recovery attempt: {e}")

    async def _save_state_snapshot(self, snapshot: ServiceStateSnapshot):
        """Save state snapshot to database."""
        if not self.postgres_engine:
            return

        try:
            with self.postgres_engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO service_snapshots
                    (service_name, timestamp, container_id, image_id, snapshot_data)
                    VALUES
                    (:service_name, :timestamp, :container_id, :image_id, :snapshot_data)
                """), {
                    'service_name': snapshot.service_name,
                    'timestamp': snapshot.timestamp,
                    'container_id': snapshot.container_id,
                    'image_id': snapshot.image_id,
                    'snapshot_data': json.dumps(asdict(snapshot), default=str)
                })
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to save state snapshot: {e}")

    async def _send_alert(self, service_name: str, message: str, severity: str):
        """Send alert notification."""
        service_config = self.services.get(service_name)
        if not service_config:
            return

        alert_data = {
            'service': service_name,
            'message': message,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            'state': self.service_states.get(service_name, ServiceState.UNKNOWN).value
        }

        # Send to configured alert channels
        for channel in service_config.alert_channels:
            try:
                if channel.startswith('slack://'):
                    await self._send_slack_alert(channel, alert_data)
                elif channel.startswith('email://'):
                    await self._send_email_alert(channel, alert_data)
                elif channel.startswith('webhook://'):
                    await self._send_webhook_alert(channel, alert_data)
            except Exception as e:
                self.logger.error(f"Failed to send alert to {channel}: {e}")

    async def _send_slack_alert(self, webhook_url: str, alert_data: Dict[str, Any]):
        """Send alert to Slack."""
        # Implementation would send to Slack webhook
        pass

    async def _send_email_alert(self, email_config: str, alert_data: Dict[str, Any]):
        """Send alert via email."""
        # Implementation would send email alert
        pass

    async def _send_webhook_alert(self, webhook_url: str, alert_data: Dict[str, Any]):
        """Send alert to webhook."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=alert_data) as response:
                    if response.status == 200:
                        self.logger.info(f"Alert sent to webhook: {webhook_url}")
                    else:
                        self.logger.warning(f"Webhook alert failed: {response.status}")
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")

    async def _cleanup_old_data(self):
        """Clean up old snapshots and recovery data."""
        while not self.shutdown_event.is_set():
            try:
                # Clean up old snapshots from memory
                cutoff_time = datetime.utcnow() - timedelta(days=self.snapshot_retention_days)

                for service_name in self.state_snapshots:
                    snapshots = self.state_snapshots[service_name]
                    self.state_snapshots[service_name] = [
                        s for s in snapshots if s.timestamp > cutoff_time
                    ]

                # Clean up old database records
                if self.postgres_engine:
                    with self.postgres_engine.connect() as conn:
                        # Clean old recovery attempts
                        conn.execute(text("""
                            DELETE FROM recovery_attempts
                            WHERE created_at < NOW() - INTERVAL ':days days'
                        """), {'days': self.metrics_retention_days})

                        # Clean old snapshots
                        conn.execute(text("""
                            DELETE FROM service_snapshots
                            WHERE created_at < NOW() - INTERVAL ':days days'
                        """), {'days': self.snapshot_retention_days})

                        conn.commit()

                # Sleep for 1 hour before next cleanup
                await asyncio.sleep(3600)

            except Exception as e:
                self.logger.error(f"Cleanup task error: {e}")
                await asyncio.sleep(300)  # Back off on errors

    async def _update_service_discovery(self):
        """Update service discovery with current service states."""
        while not self.shutdown_event.is_set():
            try:
                for service_name, state in self.service_states.items():
                    service_config = self.services.get(service_name)
                    if not service_config:
                        continue

                    # Update Consul service registry
                    try:
                        self.consul_client.agent.service.register(
                            name=service_name,
                            service_id=f"{service_name}_auto_recovery",
                            tags=[state.value, service_config.criticality],
                            check=consul.Check.http(
                                service_config.health_check_url,
                                timeout="10s",
                                interval="30s"
                            ) if service_config.health_check_url else None
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to update Consul for {service_name}: {e}")

                # Sleep for 5 minutes before next update
                await asyncio.sleep(300)

            except Exception as e:
                self.logger.error(f"Service discovery update error: {e}")
                await asyncio.sleep(60)

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        circuit_breaker_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_breaker_status[name] = cb.get_status()

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'services': {
                name: {
                    'state': state.value,
                    'config': asdict(config)
                }
                for name, (state, config) in zip(
                    self.service_states.keys(),
                    [(self.service_states[name], self.services[name]) for name in self.service_states.keys()]
                )
            },
            'circuit_breakers': circuit_breaker_status,
            'recovery_history': [asdict(attempt) for attempt in self.recovery_history[-10:]],
            'system_metrics': {
                'total_services': len(self.services),
                'healthy_services': sum(1 for state in self.service_states.values() if state == ServiceState.HEALTHY),
                'unhealthy_services': sum(1 for state in self.service_states.values() if state in [ServiceState.UNHEALTHY, ServiceState.FAILED]),
                'recovering_services': sum(1 for state in self.service_states.values() if state == ServiceState.RECOVERING),
                'total_recovery_attempts': len(self.recovery_history),
                'successful_recoveries': sum(1 for attempt in self.recovery_history if attempt.result == RecoveryResult.SUCCESS)
            }
        }

    async def force_recovery(self, service_name: str, strategy: RecoveryStrategy) -> RecoveryAttempt:
        """Force recovery of a specific service with a specific strategy."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")

        service_config = self.services[service_name]
        current_state = self.service_states.get(service_name, ServiceState.UNKNOWN)

        self.logger.info(f"Force recovery requested for {service_name} using {strategy.value}")

        return await self._execute_recovery_strategy(
            service_name, service_config, strategy, current_state
        )


# Signal handler for graceful shutdown
def setup_signal_handlers(auto_recovery_system: AutoRecoverySystem):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        asyncio.create_task(auto_recovery_system.stop_monitoring())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Main entry point
async def main():
    """Main entry point for the auto-recovery system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Initialize auto-recovery system
    auto_recovery = AutoRecoverySystem()

    # Setup signal handlers
    setup_signal_handlers(auto_recovery)

    try:
        # Start monitoring
        await auto_recovery.start_monitoring()

        # Keep running until shutdown
        await auto_recovery.shutdown_event.wait()

    except KeyboardInterrupt:
        pass
    finally:
        await auto_recovery.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(main())