"""
Chaos Engineering Scenario Library for BEV OSINT Framework
==========================================================

Comprehensive library of chaos engineering scenarios for systematic
stress testing of the BEV framework's resilience and robustness.

Features:
- Pre-defined chaos engineering scenarios with varying complexity
- Systematic stress testing patterns and progressions
- Domain-specific scenarios for OSINT workloads
- Scenario composition and orchestration capabilities
- Performance regression and reliability testing scenarios

Author: BEV Infrastructure Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml
from abc import ABC, abstractmethod


class ScenarioCategory(Enum):
    """Categories of chaos engineering scenarios."""
    NETWORK = "network"
    COMPUTE = "compute"
    STORAGE = "storage"
    APPLICATION = "application"
    DATA = "data"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    INTEGRATION = "integration"


class ScenarioComplexity(Enum):
    """Complexity levels for scenarios."""
    SIMPLE = "simple"          # Single fault, single service
    MODERATE = "moderate"      # Multiple faults, single service
    COMPLEX = "complex"        # Multiple faults, multiple services
    ADVANCED = "advanced"      # Cascading faults, system-wide impact


class ScenarioOutcome(Enum):
    """Expected outcomes for scenarios."""
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAST_RECOVERY = "fast_recovery"
    NO_DATA_LOSS = "no_data_loss"
    MAINTAINED_AVAILABILITY = "maintained_availability"
    CIRCUIT_BREAKER_ACTIVATION = "circuit_breaker_activation"
    AUTO_SCALING = "auto_scaling"
    FAILOVER = "failover"


@dataclass
class ScenarioStep:
    """A single step in a chaos engineering scenario."""
    name: str
    action_type: str  # inject_fault, wait, validate, cleanup
    parameters: Dict[str, Any] = field(default_factory=dict)
    delay_before: float = 0.0  # seconds to wait before this step
    delay_after: float = 0.0   # seconds to wait after this step
    continue_on_failure: bool = True
    validation_criteria: List[str] = field(default_factory=list)


@dataclass
class ChaosScenario:
    """Definition of a chaos engineering scenario."""
    name: str
    description: str
    category: ScenarioCategory
    complexity: ScenarioComplexity

    # Target configuration
    target_services: List[str] = field(default_factory=list)
    target_components: List[str] = field(default_factory=list)

    # Scenario execution
    steps: List[ScenarioStep] = field(default_factory=list)
    total_duration: float = 300.0  # 5 minutes default

    # Expected outcomes
    expected_outcomes: List[ScenarioOutcome] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)

    # Safety and constraints
    max_impact_threshold: float = 0.5  # Maximum acceptable impact
    abort_conditions: List[str] = field(default_factory=list)
    cleanup_required: bool = True

    # Metadata
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    estimated_recovery_time: float = 60.0
    documentation_url: str = ""

    # Versioning and authoring
    version: str = "1.0.0"
    author: str = ""
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None


@dataclass
class ScenarioExecution:
    """Tracking of scenario execution."""
    scenario_name: str
    execution_id: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Execution state
    current_step: int = 0
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)

    # Results
    success: bool = False
    outcomes_achieved: List[ScenarioOutcome] = field(default_factory=list)
    impact_score: float = 0.0
    recovery_time: float = 0.0

    # Metrics and data
    metrics_collected: Dict[str, Any] = field(default_factory=dict)
    errors_encountered: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)


class ScenarioLibrary:
    """
    Comprehensive library of chaos engineering scenarios with
    systematic stress testing capabilities.
    """

    def __init__(self, config_path: str = "/app/config/scenarios.yaml"):
        self.config_path = Path(config_path)
        self.scenarios: Dict[str, ChaosScenario] = {}
        self.scenario_suites: Dict[str, List[str]] = {}
        self.execution_history: List[ScenarioExecution] = []

        self.logger = logging.getLogger("scenario_library")

        # Initialize with built-in scenarios
        self._initialize_builtin_scenarios()

        # Load additional scenarios from config
        self._load_scenarios_from_config()

    def _initialize_builtin_scenarios(self):
        """Initialize the library with built-in scenarios."""

        # === NETWORK SCENARIOS ===

        # Network Latency Scenarios
        self.scenarios["network_latency_light"] = ChaosScenario(
            name="network_latency_light",
            description="Light network latency to test service resilience",
            category=ScenarioCategory.NETWORK,
            complexity=ScenarioComplexity.SIMPLE,
            target_services=["web-server", "api-server"],
            steps=[
                ScenarioStep(
                    name="inject_network_delay",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "network_delay",
                        "profile_name": "network_delay_light",
                        "delay_ms": 100,
                        "jitter_ms": 20
                    },
                    delay_after=10.0
                ),
                ScenarioStep(
                    name="monitor_performance",
                    action_type="validate",
                    parameters={"duration": 120.0},
                    validation_criteria=["response_time < 500ms", "availability > 95%"]
                ),
                ScenarioStep(
                    name="cleanup_faults",
                    action_type="cleanup",
                    parameters={}
                )
            ],
            total_duration=180.0,
            expected_outcomes=[ScenarioOutcome.GRACEFUL_DEGRADATION],
            success_criteria=["availability > 95%", "no_cascading_failures"],
            tags=["network", "latency", "simple"]
        )

        self.scenarios["network_partition"] = ChaosScenario(
            name="network_partition",
            description="Network partition between services to test isolation",
            category=ScenarioCategory.NETWORK,
            complexity=ScenarioComplexity.COMPLEX,
            target_services=["database", "api-server"],
            steps=[
                ScenarioStep(
                    name="create_network_partition",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "network_partition",
                        "target_service": "database",
                        "parameters": {"target_ip": "172.30.0.0/24"}
                    },
                    delay_after=30.0
                ),
                ScenarioStep(
                    name="validate_circuit_breaker",
                    action_type="validate",
                    parameters={"duration": 60.0},
                    validation_criteria=["circuit_breaker_active", "fallback_active"]
                ),
                ScenarioStep(
                    name="restore_network",
                    action_type="cleanup",
                    parameters={}
                ),
                ScenarioStep(
                    name="verify_recovery",
                    action_type="validate",
                    parameters={"duration": 60.0},
                    validation_criteria=["all_services_healthy", "data_consistency"]
                )
            ],
            total_duration=240.0,
            expected_outcomes=[ScenarioOutcome.CIRCUIT_BREAKER_ACTIVATION, ScenarioOutcome.FAST_RECOVERY],
            success_criteria=["circuit_breaker_triggered", "recovery_time < 90s"],
            tags=["network", "partition", "circuit-breaker"]
        )

        # === COMPUTE SCENARIOS ===

        self.scenarios["cpu_stress_progressive"] = ChaosScenario(
            name="cpu_stress_progressive",
            description="Progressive CPU stress to test auto-scaling",
            category=ScenarioCategory.COMPUTE,
            complexity=ScenarioComplexity.MODERATE,
            target_services=["worker-service"],
            steps=[
                ScenarioStep(
                    name="light_cpu_stress",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "cpu_stress",
                        "profile_name": "cpu_stress_moderate",
                        "cpu_cores": 1,
                        "cpu_load": 50
                    },
                    delay_after=60.0
                ),
                ScenarioStep(
                    name="validate_performance",
                    action_type="validate",
                    parameters={"duration": 30.0},
                    validation_criteria=["response_time < 1000ms"]
                ),
                ScenarioStep(
                    name="heavy_cpu_stress",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "cpu_stress",
                        "profile_name": "cpu_stress_heavy",
                        "cpu_cores": 2,
                        "cpu_load": 90
                    },
                    delay_after=90.0
                ),
                ScenarioStep(
                    name="validate_auto_scaling",
                    action_type="validate",
                    parameters={"duration": 60.0},
                    validation_criteria=["auto_scaling_triggered", "additional_instances > 0"]
                ),
                ScenarioStep(
                    name="cleanup_stress",
                    action_type="cleanup",
                    parameters={}
                )
            ],
            total_duration=360.0,
            expected_outcomes=[ScenarioOutcome.AUTO_SCALING, ScenarioOutcome.MAINTAINED_AVAILABILITY],
            success_criteria=["auto_scaling_triggered", "availability > 98%"],
            tags=["compute", "cpu", "auto-scaling"]
        )

        self.scenarios["memory_exhaustion"] = ChaosScenario(
            name="memory_exhaustion",
            description="Memory exhaustion to test memory management",
            category=ScenarioCategory.COMPUTE,
            complexity=ScenarioComplexity.MODERATE,
            target_services=["data-processor"],
            steps=[
                ScenarioStep(
                    name="gradual_memory_leak",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "memory_leak",
                        "profile_name": "memory_leak_gradual",
                        "memory_mb": 200,
                        "leak_type": "gradual",
                        "leak_rate_mb_per_sec": 2
                    },
                    delay_after=120.0
                ),
                ScenarioStep(
                    name="monitor_oom_handling",
                    action_type="validate",
                    parameters={"duration": 180.0},
                    validation_criteria=["no_oom_kills", "graceful_degradation"]
                ),
                ScenarioStep(
                    name="cleanup_memory_stress",
                    action_type="cleanup",
                    parameters={}
                )
            ],
            total_duration=360.0,
            expected_outcomes=[ScenarioOutcome.GRACEFUL_DEGRADATION],
            success_criteria=["no_oom_kills", "recovery_time < 60s"],
            tags=["compute", "memory", "oom"]
        )

        # === STORAGE SCENARIOS ===

        self.scenarios["disk_space_exhaustion"] = ChaosScenario(
            name="disk_space_exhaustion",
            description="Disk space exhaustion to test storage management",
            category=ScenarioCategory.STORAGE,
            complexity=ScenarioComplexity.SIMPLE,
            target_services=["log-processor", "database"],
            steps=[
                ScenarioStep(
                    name="fill_disk_space",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "disk_fill",
                        "profile_name": "disk_fill_gradual",
                        "fill_size_mb": 500,
                        "fill_path": "/tmp"
                    },
                    delay_after=60.0
                ),
                ScenarioStep(
                    name="validate_disk_cleanup",
                    action_type="validate",
                    parameters={"duration": 120.0},
                    validation_criteria=["log_rotation_triggered", "temp_files_cleaned"]
                ),
                ScenarioStep(
                    name="cleanup_disk_fill",
                    action_type="cleanup",
                    parameters={}
                )
            ],
            total_duration=240.0,
            expected_outcomes=[ScenarioOutcome.GRACEFUL_DEGRADATION],
            success_criteria=["disk_cleanup_triggered", "service_available"],
            tags=["storage", "disk", "cleanup"]
        )

        # === APPLICATION SCENARIOS ===

        self.scenarios["service_crash_cascade"] = ChaosScenario(
            name="service_crash_cascade",
            description="Service crash to test dependency resilience",
            category=ScenarioCategory.APPLICATION,
            complexity=ScenarioComplexity.COMPLEX,
            target_services=["authentication-service", "api-server", "web-server"],
            steps=[
                ScenarioStep(
                    name="crash_auth_service",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "service_crash",
                        "target_service": "authentication-service",
                        "profile_name": "service_crash_controlled",
                        "method": "stop"
                    },
                    delay_after=30.0
                ),
                ScenarioStep(
                    name="monitor_dependent_services",
                    action_type="validate",
                    parameters={"duration": 90.0},
                    validation_criteria=["no_cascading_failures", "fallback_auth_active"]
                ),
                ScenarioStep(
                    name="trigger_auto_recovery",
                    action_type="validate",
                    parameters={"duration": 120.0},
                    validation_criteria=["auto_recovery_triggered", "service_restarted"]
                ),
                ScenarioStep(
                    name="verify_full_recovery",
                    action_type="validate",
                    parameters={"duration": 60.0},
                    validation_criteria=["all_services_healthy", "auth_fully_restored"]
                )
            ],
            total_duration=360.0,
            expected_outcomes=[ScenarioOutcome.FAST_RECOVERY, ScenarioOutcome.NO_DATA_LOSS],
            success_criteria=["no_cascading_failures", "recovery_time < 180s"],
            tags=["application", "crash", "dependencies"]
        )

        self.scenarios["database_connection_exhaustion"] = ChaosScenario(
            name="database_connection_exhaustion",
            description="Database connection pool exhaustion",
            category=ScenarioCategory.APPLICATION,
            complexity=ScenarioComplexity.MODERATE,
            target_services=["database", "api-server"],
            steps=[
                ScenarioStep(
                    name="exhaust_db_connections",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "connection_exhaustion",
                        "target_service": "database",
                        "max_connections": 100,
                        "hold_connections": True
                    },
                    delay_after=60.0
                ),
                ScenarioStep(
                    name="validate_connection_pooling",
                    action_type="validate",
                    parameters={"duration": 120.0},
                    validation_criteria=["connection_pool_healthy", "queue_management_active"]
                ),
                ScenarioStep(
                    name="cleanup_connections",
                    action_type="cleanup",
                    parameters={}
                )
            ],
            total_duration=240.0,
            expected_outcomes=[ScenarioOutcome.GRACEFUL_DEGRADATION],
            success_criteria=["connection_pool_recovered", "no_request_drops"],
            tags=["application", "database", "connections"]
        )

        # === DATA SCENARIOS ===

        self.scenarios["data_corruption_simulation"] = ChaosScenario(
            name="data_corruption_simulation",
            description="Simulate data corruption to test validation",
            category=ScenarioCategory.DATA,
            complexity=ScenarioComplexity.ADVANCED,
            target_services=["database", "data-processor"],
            steps=[
                ScenarioStep(
                    name="create_data_backup",
                    action_type="validate",
                    parameters={"create_backup": True},
                    validation_criteria=["backup_created"]
                ),
                ScenarioStep(
                    name="introduce_data_inconsistency",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "data_corruption",
                        "corruption_type": "checksum_mismatch",
                        "affected_percentage": 1.0
                    },
                    delay_after=30.0
                ),
                ScenarioStep(
                    name="validate_corruption_detection",
                    action_type="validate",
                    parameters={"duration": 60.0},
                    validation_criteria=["corruption_detected", "alerts_triggered"]
                ),
                ScenarioStep(
                    name="validate_auto_repair",
                    action_type="validate",
                    parameters={"duration": 120.0},
                    validation_criteria=["auto_repair_triggered", "data_consistency_restored"]
                ),
                ScenarioStep(
                    name="cleanup_corruption",
                    action_type="cleanup",
                    parameters={}
                )
            ],
            total_duration=300.0,
            expected_outcomes=[ScenarioOutcome.NO_DATA_LOSS],
            success_criteria=["corruption_detected", "data_restored"],
            tags=["data", "corruption", "validation"]
        )

        # === INTEGRATION SCENARIOS ===

        self.scenarios["external_api_failure"] = ChaosScenario(
            name="external_api_failure",
            description="External API failure to test integration resilience",
            category=ScenarioCategory.INTEGRATION,
            complexity=ScenarioComplexity.MODERATE,
            target_services=["osint-collector", "threat-analyzer"],
            steps=[
                ScenarioStep(
                    name="block_external_apis",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "network_partition",
                        "target_ips": ["8.8.8.8", "1.1.1.1"],
                        "block_external": True
                    },
                    delay_after=60.0
                ),
                ScenarioStep(
                    name="validate_fallback_mechanisms",
                    action_type="validate",
                    parameters={"duration": 120.0},
                    validation_criteria=["fallback_sources_active", "cached_data_used"]
                ),
                ScenarioStep(
                    name="restore_external_access",
                    action_type="cleanup",
                    parameters={}
                ),
                ScenarioStep(
                    name="validate_api_recovery",
                    action_type="validate",
                    parameters={"duration": 60.0},
                    validation_criteria=["external_apis_restored", "data_sync_resumed"]
                )
            ],
            total_duration=300.0,
            expected_outcomes=[ScenarioOutcome.GRACEFUL_DEGRADATION, ScenarioOutcome.FAST_RECOVERY],
            success_criteria=["fallback_activated", "recovery_time < 90s"],
            tags=["integration", "external-api", "fallback"]
        )

        # === ADVANCED SCENARIOS ===

        self.scenarios["multi_tier_cascade_failure"] = ChaosScenario(
            name="multi_tier_cascade_failure",
            description="Multi-tier cascading failure simulation",
            category=ScenarioCategory.INFRASTRUCTURE,
            complexity=ScenarioComplexity.ADVANCED,
            target_services=["load-balancer", "web-server", "api-server", "database"],
            steps=[
                ScenarioStep(
                    name="overload_database",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "cpu_stress",
                        "target_service": "database",
                        "cpu_load": 95
                    },
                    delay_after=45.0
                ),
                ScenarioStep(
                    name="add_network_latency",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "network_delay",
                        "target_service": "api-server",
                        "delay_ms": 300
                    },
                    delay_after=60.0
                ),
                ScenarioStep(
                    name="crash_web_server_instance",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "service_crash",
                        "target_service": "web-server",
                        "method": "kill"
                    },
                    delay_after=30.0
                ),
                ScenarioStep(
                    name="monitor_cascade_containment",
                    action_type="validate",
                    parameters={"duration": 180.0},
                    validation_criteria=[
                        "circuit_breakers_active",
                        "load_balancer_healthy",
                        "some_capacity_available"
                    ]
                ),
                ScenarioStep(
                    name="cleanup_all_faults",
                    action_type="cleanup",
                    parameters={}
                ),
                ScenarioStep(
                    name="validate_full_recovery",
                    action_type="validate",
                    parameters={"duration": 120.0},
                    validation_criteria=["all_tiers_healthy", "performance_restored"]
                )
            ],
            total_duration=600.0,
            expected_outcomes=[
                ScenarioOutcome.CIRCUIT_BREAKER_ACTIVATION,
                ScenarioOutcome.GRACEFUL_DEGRADATION,
                ScenarioOutcome.FAST_RECOVERY
            ],
            success_criteria=[
                "cascade_contained",
                "availability > 60%",
                "recovery_time < 300s"
            ],
            max_impact_threshold=0.8,
            tags=["advanced", "cascade", "multi-tier"]
        )

        # === OSINT-SPECIFIC SCENARIOS ===

        self.scenarios["osint_data_pipeline_stress"] = ChaosScenario(
            name="osint_data_pipeline_stress",
            description="OSINT data pipeline under stress conditions",
            category=ScenarioCategory.DATA,
            complexity=ScenarioComplexity.COMPLEX,
            target_services=["data-collector", "data-processor", "threat-analyzer"],
            steps=[
                ScenarioStep(
                    name="simulate_high_volume_ingestion",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "load_generator",
                        "target_service": "data-collector",
                        "data_volume_multiplier": 10,
                        "burst_pattern": True
                    },
                    delay_after=120.0
                ),
                ScenarioStep(
                    name="add_processing_delays",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "cpu_stress",
                        "target_service": "data-processor",
                        "cpu_load": 85
                    },
                    delay_after=90.0
                ),
                ScenarioStep(
                    name="validate_queue_management",
                    action_type="validate",
                    parameters={"duration": 180.0},
                    validation_criteria=[
                        "queue_not_overflowing",
                        "data_quality_maintained",
                        "processing_rate_adaptive"
                    ]
                ),
                ScenarioStep(
                    name="cleanup_stress_conditions",
                    action_type="cleanup",
                    parameters={}
                )
            ],
            total_duration=450.0,
            expected_outcomes=[ScenarioOutcome.GRACEFUL_DEGRADATION, ScenarioOutcome.NO_DATA_LOSS],
            success_criteria=["no_data_loss", "queue_stable", "processing_resumed"],
            tags=["osint", "data-pipeline", "stress"]
        )

        self.scenarios["threat_intel_system_failure"] = ChaosScenario(
            name="threat_intel_system_failure",
            description="Threat intelligence system failure simulation",
            category=ScenarioCategory.APPLICATION,
            complexity=ScenarioComplexity.COMPLEX,
            target_services=["threat-analyzer", "intel-correlator", "alert-manager"],
            steps=[
                ScenarioStep(
                    name="crash_threat_analyzer",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "service_crash",
                        "target_service": "threat-analyzer",
                        "method": "stop"
                    },
                    delay_after=30.0
                ),
                ScenarioStep(
                    name="corrupt_intel_database",
                    action_type="inject_fault",
                    parameters={
                        "injector_name": "data_corruption",
                        "target_service": "intel-correlator",
                        "corruption_type": "index_corruption"
                    },
                    delay_after=60.0
                ),
                ScenarioStep(
                    name="validate_threat_detection_continuity",
                    action_type="validate",
                    parameters={"duration": 180.0},
                    validation_criteria=[
                        "backup_analyzer_active",
                        "intel_correlation_degraded_but_functional",
                        "critical_alerts_still_generated"
                    ]
                ),
                ScenarioStep(
                    name="trigger_system_recovery",
                    action_type="cleanup",
                    parameters={}
                ),
                ScenarioStep(
                    name="validate_threat_detection_restoration",
                    action_type="validate",
                    parameters={"duration": 120.0},
                    validation_criteria=["full_threat_detection_restored", "intel_database_repaired"]
                )
            ],
            total_duration=450.0,
            expected_outcomes=[ScenarioOutcome.FAST_RECOVERY, ScenarioOutcome.NO_DATA_LOSS],
            success_criteria=["threat_detection_maintained", "recovery_time < 240s"],
            tags=["osint", "threat-intel", "security"]
        )

        # Initialize scenario suites
        self._initialize_scenario_suites()

    def _initialize_scenario_suites(self):
        """Initialize predefined scenario suites."""

        self.scenario_suites = {
            "basic_resilience": [
                "network_latency_light",
                "cpu_stress_progressive",
                "service_crash_cascade"
            ],

            "comprehensive_stress": [
                "network_latency_light",
                "network_partition",
                "cpu_stress_progressive",
                "memory_exhaustion",
                "disk_space_exhaustion",
                "service_crash_cascade"
            ],

            "osint_specific": [
                "osint_data_pipeline_stress",
                "threat_intel_system_failure",
                "external_api_failure"
            ],

            "advanced_chaos": [
                "multi_tier_cascade_failure",
                "data_corruption_simulation",
                "database_connection_exhaustion"
            ],

            "progressive_complexity": [
                "network_latency_light",
                "cpu_stress_progressive",
                "service_crash_cascade",
                "network_partition",
                "multi_tier_cascade_failure"
            ]
        }

    def _load_scenarios_from_config(self):
        """Load additional scenarios from configuration file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config_data = yaml.safe_load(f)

                # Load custom scenarios
                for scenario_data in config_data.get('scenarios', []):
                    scenario = ChaosScenario(**scenario_data)
                    self.scenarios[scenario.name] = scenario

                # Load custom suites
                custom_suites = config_data.get('scenario_suites', {})
                self.scenario_suites.update(custom_suites)

                self.logger.info(f"Loaded {len(config_data.get('scenarios', []))} custom scenarios")

        except Exception as e:
            self.logger.error(f"Failed to load scenarios from config: {e}")

    def get_scenario(self, name: str) -> Optional[ChaosScenario]:
        """Get a scenario by name."""
        return self.scenarios.get(name)

    def get_scenarios_by_category(self, category: ScenarioCategory) -> List[ChaosScenario]:
        """Get all scenarios in a specific category."""
        return [s for s in self.scenarios.values() if s.category == category]

    def get_scenarios_by_complexity(self, complexity: ScenarioComplexity) -> List[ChaosScenario]:
        """Get all scenarios of a specific complexity level."""
        return [s for s in self.scenarios.values() if s.complexity == complexity]

    def get_scenarios_by_tags(self, tags: List[str]) -> List[ChaosScenario]:
        """Get scenarios that match any of the provided tags."""
        matching_scenarios = []
        for scenario in self.scenarios.values():
            if any(tag in scenario.tags for tag in tags):
                matching_scenarios.append(scenario)
        return matching_scenarios

    def get_scenario_suite(self, suite_name: str) -> List[ChaosScenario]:
        """Get all scenarios in a predefined suite."""
        scenario_names = self.scenario_suites.get(suite_name, [])
        return [self.scenarios[name] for name in scenario_names if name in self.scenarios]

    def list_available_scenarios(self) -> Dict[str, Any]:
        """List all available scenarios with metadata."""
        scenario_list = {}

        for name, scenario in self.scenarios.items():
            scenario_list[name] = {
                'description': scenario.description,
                'category': scenario.category.value,
                'complexity': scenario.complexity.value,
                'duration': scenario.total_duration,
                'target_services': scenario.target_services,
                'tags': scenario.tags,
                'expected_outcomes': [outcome.value for outcome in scenario.expected_outcomes]
            }

        return scenario_list

    def list_scenario_suites(self) -> Dict[str, Any]:
        """List all available scenario suites."""
        suite_info = {}

        for suite_name, scenario_names in self.scenario_suites.items():
            scenarios = [self.scenarios[name] for name in scenario_names if name in self.scenarios]

            suite_info[suite_name] = {
                'scenario_count': len(scenarios),
                'scenarios': scenario_names,
                'total_duration': sum(s.total_duration for s in scenarios),
                'complexity_levels': list(set(s.complexity.value for s in scenarios)),
                'categories': list(set(s.category.value for s in scenarios))
            }

        return suite_info

    def create_custom_scenario(self, scenario: ChaosScenario) -> bool:
        """Add a custom scenario to the library."""
        try:
            # Validate scenario
            if not scenario.name or not scenario.steps:
                raise ValueError("Scenario must have a name and at least one step")

            # Set timestamps
            if not scenario.created_at:
                scenario.created_at = datetime.utcnow()
            scenario.last_modified = datetime.utcnow()

            # Add to library
            self.scenarios[scenario.name] = scenario

            self.logger.info(f"Added custom scenario: {scenario.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create custom scenario: {e}")
            return False

    def create_scenario_suite(self, suite_name: str, scenario_names: List[str]) -> bool:
        """Create a custom scenario suite."""
        try:
            # Validate that all scenarios exist
            missing_scenarios = [name for name in scenario_names if name not in self.scenarios]
            if missing_scenarios:
                raise ValueError(f"Unknown scenarios: {missing_scenarios}")

            self.scenario_suites[suite_name] = scenario_names

            self.logger.info(f"Created scenario suite: {suite_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create scenario suite: {e}")
            return False

    def generate_progressive_suite(self, target_services: List[str],
                                 max_complexity: ScenarioComplexity = ScenarioComplexity.COMPLEX) -> List[ChaosScenario]:
        """Generate a progressive test suite starting simple and increasing complexity."""

        # Filter scenarios by target services and complexity
        applicable_scenarios = []

        for scenario in self.scenarios.values():
            # Check if any target service matches
            if any(service in scenario.target_services for service in target_services):
                # Check complexity level
                complexity_order = [ScenarioComplexity.SIMPLE, ScenarioComplexity.MODERATE,
                                  ScenarioComplexity.COMPLEX, ScenarioComplexity.ADVANCED]

                if complexity_order.index(scenario.complexity) <= complexity_order.index(max_complexity):
                    applicable_scenarios.append(scenario)

        # Sort by complexity
        applicable_scenarios.sort(key=lambda s: [ScenarioComplexity.SIMPLE, ScenarioComplexity.MODERATE,
                                               ScenarioComplexity.COMPLEX, ScenarioComplexity.ADVANCED].index(s.complexity))

        return applicable_scenarios

    def generate_random_suite(self, count: int = 5, category: Optional[ScenarioCategory] = None,
                            complexity: Optional[ScenarioComplexity] = None) -> List[ChaosScenario]:
        """Generate a random suite of scenarios."""

        available_scenarios = list(self.scenarios.values())

        # Filter by category if specified
        if category:
            available_scenarios = [s for s in available_scenarios if s.category == category]

        # Filter by complexity if specified
        if complexity:
            available_scenarios = [s for s in available_scenarios if s.complexity == complexity]

        # Randomly select scenarios
        selected_count = min(count, len(available_scenarios))
        return random.sample(available_scenarios, selected_count)

    def validate_scenario(self, scenario: ChaosScenario) -> List[str]:
        """Validate a scenario configuration."""
        issues = []

        # Basic validation
        if not scenario.name:
            issues.append("Scenario must have a name")

        if not scenario.steps:
            issues.append("Scenario must have at least one step")

        if scenario.total_duration <= 0:
            issues.append("Total duration must be positive")

        # Step validation
        for i, step in enumerate(scenario.steps):
            if not step.name:
                issues.append(f"Step {i+1} must have a name")

            if not step.action_type:
                issues.append(f"Step {i+1} must have an action type")

            if step.delay_before < 0 or step.delay_after < 0:
                issues.append(f"Step {i+1} delays must be non-negative")

        # Timing validation
        total_step_time = sum(step.delay_before + step.delay_after for step in scenario.steps)
        if total_step_time > scenario.total_duration:
            issues.append("Sum of step delays exceeds total duration")

        return issues

    def estimate_scenario_duration(self, scenario: ChaosScenario) -> float:
        """Estimate the actual duration of a scenario including all delays."""

        total_duration = 0.0

        for step in scenario.steps:
            total_duration += step.delay_before + step.delay_after

            # Add estimated execution time based on action type
            if step.action_type == "inject_fault":
                total_duration += 10.0  # Fault injection overhead
            elif step.action_type == "validate":
                total_duration += step.parameters.get("duration", 30.0)
            elif step.action_type == "cleanup":
                total_duration += 15.0  # Cleanup overhead
            elif step.action_type == "wait":
                total_duration += step.parameters.get("duration", 60.0)

        return max(total_duration, scenario.total_duration)

    def get_scenario_statistics(self) -> Dict[str, Any]:
        """Get statistics about the scenario library."""

        categories = {}
        complexities = {}
        total_scenarios = len(self.scenarios)

        for scenario in self.scenarios.values():
            # Category statistics
            cat = scenario.category.value
            categories[cat] = categories.get(cat, 0) + 1

            # Complexity statistics
            comp = scenario.complexity.value
            complexities[comp] = complexities.get(comp, 0) + 1

        return {
            'total_scenarios': total_scenarios,
            'total_suites': len(self.scenario_suites),
            'categories': categories,
            'complexities': complexities,
            'average_duration': sum(s.total_duration for s in self.scenarios.values()) / total_scenarios if total_scenarios > 0 else 0,
            'execution_history_count': len(self.execution_history)
        }

    def export_scenarios(self, file_path: str, scenario_names: Optional[List[str]] = None):
        """Export scenarios to a YAML file."""
        try:
            scenarios_to_export = {}

            if scenario_names:
                for name in scenario_names:
                    if name in self.scenarios:
                        scenarios_to_export[name] = asdict(self.scenarios[name])
            else:
                scenarios_to_export = {name: asdict(scenario) for name, scenario in self.scenarios.items()}

            export_data = {
                'scenarios': list(scenarios_to_export.values()),
                'scenario_suites': self.scenario_suites,
                'exported_at': datetime.utcnow().isoformat(),
                'version': '1.0.0'
            }

            with open(file_path, 'w') as f:
                yaml.dump(export_data, f, default_flow_style=False, indent=2)

            self.logger.info(f"Exported {len(scenarios_to_export)} scenarios to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to export scenarios: {e}")

    def import_scenarios(self, file_path: str) -> bool:
        """Import scenarios from a YAML file."""
        try:
            with open(file_path, 'r') as f:
                import_data = yaml.safe_load(f)

            imported_count = 0

            # Import scenarios
            for scenario_data in import_data.get('scenarios', []):
                scenario = ChaosScenario(**scenario_data)
                self.scenarios[scenario.name] = scenario
                imported_count += 1

            # Import scenario suites
            imported_suites = import_data.get('scenario_suites', {})
            self.scenario_suites.update(imported_suites)

            self.logger.info(f"Imported {imported_count} scenarios and {len(imported_suites)} suites")
            return True

        except Exception as e:
            self.logger.error(f"Failed to import scenarios: {e}")
            return False


# Example usage
def example_scenario_usage():
    """Example of how to use the scenario library."""

    # Initialize library
    library = ScenarioLibrary()

    # List available scenarios
    print("Available scenarios:")
    scenarios = library.list_available_scenarios()
    for name, info in scenarios.items():
        print(f"  {name}: {info['description']}")

    # Get scenarios by category
    network_scenarios = library.get_scenarios_by_category(ScenarioCategory.NETWORK)
    print(f"\nNetwork scenarios: {len(network_scenarios)}")

    # Get a scenario suite
    basic_suite = library.get_scenario_suite("basic_resilience")
    print(f"\nBasic resilience suite has {len(basic_suite)} scenarios")

    # Generate progressive suite
    progressive = library.generate_progressive_suite(
        target_services=["web-server", "api-server"],
        max_complexity=ScenarioComplexity.COMPLEX
    )
    print(f"\nProgressive suite has {len(progressive)} scenarios")

    # Create custom scenario
    custom_scenario = ChaosScenario(
        name="custom_test",
        description="Custom test scenario",
        category=ScenarioCategory.NETWORK,
        complexity=ScenarioComplexity.SIMPLE,
        target_services=["test-service"],
        steps=[
            ScenarioStep(
                name="test_step",
                action_type="inject_fault",
                parameters={"test": "value"}
            )
        ]
    )

    library.create_custom_scenario(custom_scenario)
    print(f"\nCreated custom scenario: {custom_scenario.name}")

    # Get statistics
    stats = library.get_scenario_statistics()
    print(f"\nLibrary statistics: {stats}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_scenario_usage()