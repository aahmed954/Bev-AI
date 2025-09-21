# BEV Auto-Recovery System

## Overview

The BEV Auto-Recovery System is a comprehensive, enterprise-grade infrastructure component that provides intelligent service recovery, circuit breaker protection, and state management for the BEV OSINT Framework. It implements multiple recovery strategies with configurable thresholds and integrates seamlessly with Docker container orchestration.

## Features

### Core Capabilities
- **Circuit Breaker Pattern**: Prevents cascading failures with configurable thresholds
- **Multiple Recovery Strategies**: Restart, rollback, circuit break, scale up/down, failover, recreate, and hybrid approaches
- **State Preservation**: Automatic snapshots and rollback capabilities
- **Intelligent Health Monitoring**: Continuous service health assessment with multiple check types
- **Performance Monitoring**: Real-time metrics collection and SLA compliance tracking
- **Comprehensive Logging**: Structured logging with distributed tracing support
- **Multi-Channel Alerting**: Email, Slack, webhook, SMS, and PagerDuty integrations

### Technical Features
- **Recovery Time**: < 60 seconds (target)
- **Health Check Response**: < 5 seconds
- **Circuit Breaker Response**: < 1 second
- **State Snapshot Time**: < 30 seconds
- **Rollback Time**: < 2 minutes
- **Concurrent Operations**: Up to 100 simultaneous calls per service
- **Service Discovery**: Automatic registration and deregistration
- **Container Orchestration**: Docker and Kubernetes support

## Architecture

### Components

1. **Circuit Breaker (`circuit_breaker.py`)**
   - Advanced circuit breaker implementation with exponential backoff
   - State persistence across service restarts
   - Bulkhead pattern for resource isolation
   - Health check integration

2. **Auto-Recovery Service (`auto_recovery.py`)**
   - Main orchestration engine for recovery operations
   - Multiple recovery strategies with intelligent selection
   - Service dependency management
   - Container lifecycle management

3. **Logging & Alerting (`logging_alerting.py`)**
   - Structured logging with OpenTelemetry integration
   - Multi-channel alerting with escalation policies
   - Prometheus metrics collection
   - Audit trail and compliance logging

4. **Validation Framework (`recovery_validator.py`)**
   - Comprehensive testing and validation
   - Performance benchmarking
   - Chaos engineering capabilities
   - Compliance verification

## Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- PostgreSQL (for state persistence)
- Redis (for caching and state management)

### Setup

1. **Configure Environment Variables**:
   ```bash
   export POSTGRES_URI="postgresql://user:pass@localhost:5432/bev"
   export REDIS_PASSWORD="your_redis_password"
   export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
   export BEV_ADMIN_EMAIL="admin@yourdomain.com"
   ```

2. **Deploy with Docker Compose**:
   ```bash
   docker-compose -f docker-compose.complete.yml up -d auto-recovery
   ```

3. **Verify Installation**:
   ```bash
   curl http://localhost:8014/health
   ```

## Configuration

### Service Configuration (`config/auto_recovery.yaml`)

```yaml
services:
  - name: postgres
    container_name: bev_postgres
    health_check_url: "http://172.30.0.2:5432"
    recovery_strategies: [restart, rollback, recreate]
    max_recovery_attempts: 3
    target_availability: 99.95
    criticality: critical
    circuit_breaker_enabled: true
    state_backup_enabled: true
```

### Circuit Breaker Configuration

```python
config = CircuitBreakerConfig(
    failure_threshold=5,          # Failures before opening
    failure_rate_threshold=0.5,   # 50% failure rate threshold
    timeout_duration=60.0,        # Open state duration
    request_timeout=30.0,         # Individual request timeout
    max_concurrent_calls=100      # Bulkhead limit
)
```

### Recovery Strategies

1. **RESTART**: Simple container restart
2. **ROLLBACK**: Restore from previous state snapshot
3. **CIRCUIT_BREAK**: Enable circuit breaker protection
4. **SCALE_UP/DOWN**: Adjust resource allocation
5. **FAILOVER**: Switch to backup instance
6. **RECREATE**: Complete container recreation
7. **HYBRID**: Combination of multiple strategies

## Usage

### Basic Operations

#### Health Check
```bash
curl http://localhost:8014/health
```

#### System Status
```bash
curl http://localhost:8014/status
```

#### Force Recovery
```bash
curl -X POST http://localhost:8014/recover/service_name \
  -H "Content-Type: application/json" \
  -d '{"strategy": "restart"}'
```

### Programmatic Usage

```python
from auto_recovery import AutoRecoverySystem
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# Initialize auto-recovery system
auto_recovery = AutoRecoverySystem(
    config_path="/app/config/auto_recovery.yaml"
)

# Start monitoring
await auto_recovery.start_monitoring()

# Force recovery if needed
await auto_recovery.force_recovery("service_name", RecoveryStrategy.RESTART)
```

### Circuit Breaker Usage

```python
from circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# Create circuit breaker
config = CircuitBreakerConfig(failure_threshold=5)
cb = CircuitBreaker("my_service", config)

# Protect function calls
@circuit_breaker("my_service", config)
async def protected_function():
    # Your code here
    pass

# Or use directly
result = await cb.call(my_function, arg1, arg2)
```

## Monitoring and Metrics

### Prometheus Metrics

The system exposes comprehensive metrics at `http://localhost:9091/metrics`:

- `recovery_attempts_total`: Total recovery attempts by service and strategy
- `recovery_duration_seconds`: Recovery operation duration
- `service_health_status`: Current service health (1=healthy, 0=unhealthy)
- `circuit_breaker_state`: Circuit breaker state (0=closed, 1=half-open, 2=open)
- `active_connections`: Number of active connections per service
- `error_rate`: Current error rate percentage

### Health Monitoring

Services are continuously monitored using:
- HTTP health check endpoints
- Container status verification
- Command-based health checks
- Circuit breaker state monitoring

### Alerting Channels

Configured alert channels include:
- **Slack**: Real-time notifications with rich formatting
- **Email**: HTML-formatted alerts with full context
- **Webhook**: JSON payload to custom endpoints
- **PagerDuty**: Integration with incident management

## Validation and Testing

### Running Validation

```bash
# Run comprehensive validation
./scripts/validate_auto_recovery.sh

# Run specific test categories
python src/infrastructure/recovery_validator.py --config config/auto_recovery.yaml
```

### Test Categories

1. **Performance Tests**: Measure recovery times and response latencies
2. **Reliability Tests**: Multiple failure scenarios and dependency handling
3. **Recovery Tests**: Validate each recovery strategy effectiveness
4. **Circuit Breaker Tests**: State transitions and failure detection
5. **State Management Tests**: Snapshot creation and rollback functionality
6. **Integration Tests**: Health monitoring and logging integration
7. **Chaos Tests**: Network partitions and resource exhaustion
8. **Compliance Tests**: Audit logging and data protection

### Performance Requirements

- Recovery Time: < 60 seconds
- Health Check Response: < 5 seconds
- Circuit Breaker Response: < 1 second
- State Snapshot Time: < 30 seconds
- Rollback Time: < 2 minutes

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   ```bash
   # Check dependencies
   docker-compose logs auto-recovery

   # Verify configuration
   python -c "import yaml; yaml.safe_load(open('config/auto_recovery.yaml'))"
   ```

2. **Recovery Not Triggering**
   ```bash
   # Check service configuration
   curl http://localhost:8014/status | jq '.services["service_name"]'

   # Verify health check endpoints
   curl -v http://service:port/health
   ```

3. **Circuit Breaker Not Working**
   ```bash
   # Check circuit breaker status
   curl http://localhost:8014/status | jq '.circuit_breakers["service_name"]'

   # Reset circuit breaker
   curl -X POST http://localhost:8014/circuit-breaker/service_name/reset
   ```

### Debugging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
docker-compose restart auto-recovery
```

View detailed logs:
```bash
docker-compose logs -f auto-recovery
```

Check metrics:
```bash
curl http://localhost:9091/metrics | grep recovery
```

## Performance Tuning

### Circuit Breaker Tuning

- **High Traffic Services**: Increase `max_concurrent_calls`
- **Flaky Services**: Lower `failure_threshold`
- **Critical Services**: Longer `timeout_duration`
- **Fast Recovery**: Enable `health_check_url`

### Recovery Strategy Optimization

- **Database Services**: Use ROLLBACK for data consistency
- **Stateless Services**: Use RESTART for speed
- **Resource-Intensive Services**: Use SCALE_UP/DOWN
- **Critical Services**: Use HYBRID approach

### Resource Optimization

```yaml
# Reduce memory usage
max_concurrent_recoveries: 3
snapshot_retention_days: 3

# Increase performance
health_check_interval: 15.0
recovery_timeout: 300.0
```

## Security Considerations

- **Container Security**: Runs as non-root user (1000:1000)
- **Network Security**: Internal Docker network communication
- **Data Protection**: Encrypted state snapshots
- **Access Control**: Authentication for administrative endpoints
- **Audit Logging**: Complete audit trail of all operations

## Integration

### Health Monitoring DAG

The system integrates with the existing Airflow health monitoring DAG:

```python
# In bev_health_monitoring.py
from auto_recovery import AutoRecoverySystem

auto_recovery = AutoRecoverySystem()
await auto_recovery.force_recovery(service_name, strategy)
```

### Custom Recovery Strategies

```python
class CustomRecoveryStrategy:
    async def execute(self, service_name, service_config):
        # Your custom recovery logic
        return success_boolean

# Register custom strategy
auto_recovery.register_strategy("custom", CustomRecoveryStrategy())
```

## API Reference

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/status` | GET | Comprehensive system status |
| `/metrics` | GET | Prometheus metrics |
| `/recover/{service}` | POST | Force service recovery |
| `/circuit-breaker/{service}/reset` | POST | Reset circuit breaker |
| `/alerts/acknowledge/{id}` | POST | Acknowledge alert |
| `/snapshots/{service}` | GET | List service snapshots |

### WebSocket Events

Real-time events are available via WebSocket at `/ws/events`:

```javascript
const ws = new WebSocket('ws://localhost:8014/ws/events');
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Recovery event:', data);
};
```

## Contributing

1. Follow the existing code structure and patterns
2. Add comprehensive tests for new features
3. Update documentation for any API changes
4. Ensure all validation tests pass
5. Follow security best practices

## License

This software is part of the BEV OSINT Framework and is subject to the project's licensing terms.

## Support

For technical support and questions:
- Internal Documentation: `/docs/`
- Issue Tracking: Internal project management system
- Emergency Contact: BEV Operations Team