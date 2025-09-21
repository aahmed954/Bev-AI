# BEV OSINT Framework - Chaos Engineering System

Comprehensive chaos engineering system for testing resilience, fault tolerance, and recovery capabilities of the BEV OSINT framework.

## Overview

The BEV Chaos Engineering System provides a complete suite of tools for systematically testing system resilience through controlled fault injection, automated recovery validation, and performance analysis.

### Key Features

- **Comprehensive Fault Injection**: Network delays, CPU stress, memory leaks, service crashes, and more
- **Automated Resilience Testing**: Multi-phase testing with baseline, stress, failure, and recovery validation
- **Scenario Library**: Pre-built and customizable chaos engineering scenarios
- **Recovery Validation**: Integration with auto-recovery system (172.30.0.41) for validation
- **Performance Analysis**: Real-time metrics collection and impact assessment
- **Safety Mechanisms**: Emergency stop, safety checks, and controlled experiment boundaries
- **RESTful API**: Complete API for integration with monitoring and management systems

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Chaos Engineering API                    │
│                    (172.30.0.45:8080)                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     │         Core Components               │
│  ┌─────────────────┐│┌─────────────────┐┌──────────────────┐ │
│  │ ChaosEngineer   │││ FaultInjector   ││ ResilienceTester │ │
│  │                 │││                 ││                  │ │
│  │ - Orchestration │││ - Network Faults││ - Multi-phase    │ │
│  │ - Safety        │││ - Resource      ││   Testing        │ │
│  │ - Experiments   │││ - Service Faults││ - Metrics        │ │
│  └─────────────────┘││└─────────────────┘│ - Validation     │ │
│                     ││                   └──────────────────┘ │
│  ┌─────────────────┐││                                        │
│  │ ScenarioLibrary │││                                        │
│  │                 │││                                        │
│  │ - Built-in      │││                                        │
│  │ - Custom        │││                                        │
│  │ - Suites        │││                                        │
│  └─────────────────┘││                                        │
└─────────────────────┼───────────────────────────────────────┘
                      │
┌─────────────────────┼───────────────────────────────────────┐
│                     │         Integration Layer             │
│                     │                                       │
│  ┌─────────────────┐│┌─────────────────┐┌──────────────────┐ │
│  │ Auto-Recovery   │││ Health Monitor  ││ Target Services  │ │
│  │ (172.30.0.41)   │││ (172.30.0.38)   ││ (BEV Framework)  │ │
│  │                 │││                 ││                  │ │
│  │ - Validation    │││ - Metrics       ││ - All Services   │ │
│  │ - Recovery      │││ - Health Checks ││ - Containers     │ │
│  └─────────────────┘││└─────────────────┘│ - Infrastructure │ │
│                     ││                   └──────────────────┘ │
└─────────────────────┼───────────────────────────────────────┘
```

## Components

### 1. Chaos Engineer (`chaos_engineer.py`)

Main orchestration component that manages chaos experiments with comprehensive fault injection and recovery validation.

**Key Features:**
- Experiment lifecycle management
- Multi-phase experiment execution (baseline → injection → recovery → validation)
- Safety monitoring and emergency stop capabilities
- Integration with auto-recovery system validation

**Example Usage:**
```python
from src.testing import ChaosEngineer, ExperimentConfig

chaos_engineer = ChaosEngineer()
await chaos_engineer.initialize()

config = ExperimentConfig(
    name="web_service_resilience",
    target_services=["web-server", "api-server"],
    fault_injections=[...],
    safety_level=SafetyLevel.MEDIUM
)

result = await chaos_engineer.run_experiment(config)
```

### 2. Fault Injector (`fault_injector.py`)

Advanced fault injection system with multiple fault types and precise control mechanisms.

**Supported Fault Types:**
- **Network**: Latency, packet loss, partitions
- **Compute**: CPU stress, memory leaks, resource exhaustion
- **Storage**: Disk space exhaustion, I/O delays
- **Application**: Service crashes, connection exhaustion

**Example Usage:**
```python
from src.testing import FaultInjectionManager

fault_manager = FaultInjectionManager(docker_client, redis_client)

fault_id = await fault_manager.inject_fault(
    injector_name="network_delay",
    target_service="web-server",
    profile_name="network_delay_light",
    parameters={"delay_ms": 100, "jitter_ms": 20}
)
```

### 3. Resilience Tester (`resilience_tester.py`)

Comprehensive resilience testing framework with automated multi-phase testing and performance analysis.

**Testing Phases:**
1. **Preparation**: Environment validation
2. **Baseline**: Normal operation metrics
3. **Stress**: Load testing (optional)
4. **Failure**: Controlled fault injection
5. **Recovery**: Recovery monitoring
6. **Validation**: Performance restoration validation
7. **Analysis**: Comprehensive results analysis

**Example Usage:**
```python
from src.testing import ResilienceTester, ResilienceTestConfig

tester = ResilienceTester()
await tester.initialize()

config = ResilienceTestConfig(
    name="osint_pipeline_resilience",
    target_services=["data-collector", "threat-analyzer"],
    failure_scenarios=[...],
    availability_threshold=99.0
)

result = await tester.run_resilience_test(config)
```

### 4. Scenario Library (`scenario_library.py`)

Comprehensive library of pre-defined chaos engineering scenarios with OSINT-specific test cases.

**Built-in Scenarios:**
- Network latency and partition scenarios
- CPU and memory stress testing
- Service crash and dependency testing
- OSINT-specific data pipeline stress tests
- Multi-tier cascading failure simulations

**Scenario Suites:**
- `basic_resilience`: Essential resilience tests
- `osint_specific`: OSINT framework focused tests
- `comprehensive_stress`: Full system stress testing
- `advanced_chaos`: Complex multi-service scenarios

**Example Usage:**
```python
from src.testing import ScenarioLibrary

library = ScenarioLibrary()

# Get a specific scenario
scenario = library.get_scenario("network_latency_light")

# Get scenarios by category
network_scenarios = library.get_scenarios_by_category(ScenarioCategory.NETWORK)

# Get a scenario suite
basic_suite = library.get_scenario_suite("basic_resilience")
```

### 5. Chaos Engineering API (`chaos_api.py`)

RESTful API server providing complete access to chaos engineering functionality with real-time monitoring.

**API Endpoints:**
- `GET /health` - System health check
- `GET /status` - Comprehensive system status
- `POST /experiments` - Create new experiment
- `POST /faults` - Inject fault
- `GET /scenarios` - List available scenarios
- `POST /emergency-stop` - Emergency stop all activities
- `WebSocket /ws` - Real-time monitoring

**Example API Usage:**
```bash
# Inject a network delay fault
curl -X POST http://172.30.0.45:8080/faults \
  -H "Content-Type: application/json" \
  -d '{
    "injector_name": "network_delay",
    "target_service": "web-server",
    "profile_name": "network_delay_light",
    "parameters": {"delay_ms": 150, "jitter_ms": 30}
  }'

# Emergency stop all chaos activities
curl -X POST http://172.30.0.45:8080/emergency-stop

# Get system status
curl http://172.30.0.45:8080/status
```

## Quick Start

### 1. Start the Chaos Engineering System

```bash
# Using Docker Compose
docker-compose up chaos-engineer

# Or manually
cd /home/starlord/Projects/Bev
python -m src.testing
```

### 2. Access the API

- **API Base URL**: `http://172.30.0.45:8080`
- **WebSocket Monitoring**: `ws://172.30.0.45:8080/ws`
- **Health Check**: `http://172.30.0.45:8080/health`

### 3. Run a Simple Test

```python
import asyncio
from src.testing import ScenarioLibrary, ChaosEngineer

async def run_basic_test():
    # Initialize components
    chaos_engineer = ChaosEngineer()
    await chaos_engineer.initialize()

    scenario_library = ScenarioLibrary()

    # Get a basic scenario
    scenario = scenario_library.get_scenario("network_latency_light")

    # Convert to experiment config and run
    # Implementation depends on your specific needs

    print("Basic chaos test completed!")

# Run the test
asyncio.run(run_basic_test())
```

## Safety and Best Practices

### Safety Mechanisms

1. **Emergency Stop**: Immediately stops all chaos activities
2. **Safety Monitoring**: Continuous monitoring of system impact
3. **Automatic Rollback**: Auto-removal of faults if safety thresholds exceeded
4. **Experiment Boundaries**: Configurable impact limits and timeouts

### Best Practices

1. **Start Small**: Begin with low-impact scenarios
2. **Monitor Closely**: Use real-time monitoring during experiments
3. **Test in Stages**: Gradually increase complexity and impact
4. **Validate Recovery**: Ensure auto-recovery systems work as expected
5. **Document Results**: Keep detailed logs of experiments and learnings

### Safety Configuration

```yaml
safety:
  production_mode: false
  require_approval_for_critical: true
  auto_rollback_enabled: true
  safety_monitoring_interval: 5.0

  # Safety thresholds
  max_cpu_usage: 90.0
  max_memory_usage: 85.0
  max_error_rate: 50.0
  max_response_time: 5000.0
```

## Integration with BEV Framework

### Auto-Recovery System Integration

The chaos engineering system integrates with the auto-recovery system (172.30.0.41) to:
- Validate recovery mechanisms during experiments
- Test auto-recovery response times
- Verify service restoration capabilities

### Health Monitoring Integration

Integration with the health monitoring system (172.30.0.38) provides:
- Real-time metrics collection during experiments
- Baseline performance measurement
- Impact assessment and trend analysis

### Target Services

The system can test any service in the BEV framework:
- **Core Services**: web-server, api-server, database
- **OSINT Components**: data-collector, threat-analyzer, osint-processor
- **Infrastructure**: load-balancers, caches, message queues

## Configuration

### Main Configuration (`chaos_engineer.yaml`)

```yaml
global:
  max_concurrent_experiments: 3
  safety_check_interval: 10.0
  emergency_stop_threshold: 0.8

fault_injectors:
  network_delay:
    enabled: true
    max_delay_ms: 2000

  cpu_stress:
    enabled: true
    max_cpu_load: 95

integration:
  auto_recovery_url: "http://172.30.0.41:8080"
  health_monitor_url: "http://172.30.0.38:8080"
```

### Custom Scenarios (`scenarios.yaml`)

```yaml
scenarios:
  - name: "custom_osint_test"
    description: "Custom OSINT resilience test"
    category: "integration"
    complexity: "moderate"
    target_services: ["osint-collector", "threat-analyzer"]
    steps:
      - name: "inject_network_delay"
        action_type: "inject_fault"
        parameters:
          injector_name: "network_delay"
          delay_ms: 200
```

## Performance Targets

The chaos engineering system is designed to meet these performance criteria:

- **Experiment Completion**: <5 minutes for complete chaos experiments
- **Fault Injection Time**: <10 seconds to inject any fault type
- **Recovery Validation**: <3 minutes to validate full system recovery
- **API Response Time**: <200ms for all API endpoints
- **Safety Response**: <5 seconds to execute emergency stop

## Monitoring and Alerting

### Metrics Collection

The system collects comprehensive metrics:
- Experiment success/failure rates
- Fault injection effectiveness
- Recovery times and success rates
- System performance impact measurements

### Integration Points

- **Prometheus**: Metrics export at `/metrics` endpoint
- **WebSocket**: Real-time event streaming
- **Logs**: Structured logging to `/app/logs/chaos_engineering.log`

## Troubleshooting

### Common Issues

1. **Permission Errors**: Ensure the chaos engineering container has privileged access
2. **Network Isolation**: Verify container can access target services
3. **Resource Limits**: Check if fault injection is hitting resource constraints
4. **Safety Stops**: Review safety thresholds if experiments stop unexpectedly

### Debug Mode

Enable debug logging:
```bash
python -m src.testing --log-level DEBUG
```

### Health Checks

Monitor system health:
```bash
curl http://172.30.0.45:8080/health
curl http://172.30.0.45:8080/status
```

## Contributing

To add new fault injectors or scenarios:

1. **New Fault Types**: Extend `FaultInjector` abstract class in `fault_injector.py`
2. **Custom Scenarios**: Add to `scenarios.yaml` or create programmatically
3. **API Extensions**: Add new endpoints to `chaos_api.py`

## License

This chaos engineering system is part of the BEV OSINT Framework and follows the same licensing terms.