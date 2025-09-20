"""
BEV OSINT Framework - Chaos Engineering Testing Suite
====================================================

Comprehensive chaos engineering and resilience testing system for the
BEV OSINT framework with automated fault injection, recovery validation,
and performance analysis.

Components:
- ChaosEngineer: Main chaos engineering orchestration
- FaultInjectionManager: Advanced fault injection capabilities
- ResilienceTester: Comprehensive resilience testing framework
- ScenarioLibrary: Predefined and custom chaos scenarios
- ChaosEngineeringAPI: RESTful API for system integration

Author: BEV Infrastructure Team
Version: 1.0.0
"""

from .chaos_engineer import ChaosEngineer, ExperimentConfig, ExperimentResult
from .fault_injector import (
    FaultInjectionManager,
    FaultInjectionConfig,
    FaultExecutionContext,
    FaultType,
    FaultCategory,
    FaultSeverity
)
from .resilience_tester import (
    ResilienceTester,
    ResilienceTestConfig,
    ResilienceTestResult,
    ResilienceMetric,
    ResilienceLevel
)
from .scenario_library import (
    ScenarioLibrary,
    ChaosScenario,
    ScenarioCategory,
    ScenarioComplexity,
    ScenarioStep
)
from .chaos_api import ChaosEngineeringAPI

__all__ = [
    'ChaosEngineer',
    'ExperimentConfig',
    'ExperimentResult',
    'FaultInjectionManager',
    'FaultInjectionConfig',
    'FaultExecutionContext',
    'FaultType',
    'FaultCategory',
    'FaultSeverity',
    'ResilienceTester',
    'ResilienceTestConfig',
    'ResilienceTestResult',
    'ResilienceMetric',
    'ResilienceLevel',
    'ScenarioLibrary',
    'ChaosScenario',
    'ScenarioCategory',
    'ScenarioComplexity',
    'ScenarioStep',
    'ChaosEngineeringAPI'
]

__version__ = "1.0.0"