# BEV OSINT Framework - Testing & Quality Commands

## Primary Test Commands

### Master Test Runner
```bash
# Run complete test suite
./run_all_tests.sh

# Run specific test categories
./run_all_tests.sh --skip-performance --skip-security
./run_all_tests.sh --parallel  # Run tests in parallel
./run_all_tests.sh --quick --fail-fast  # Quick mode with early exit
```

### Individual Test Suites
```bash
# System validation
./validate_bev_deployment.sh

# Integration tests
cd tests && ./integration_tests.sh

# Performance tests
cd tests && ./performance_tests.sh

# Security tests
cd tests && ./security_tests.sh

# Monitoring tests
cd tests && ./monitoring_tests.sh
```

### Python Testing (pytest)
```bash
# Run all tests with pytest
cd tests
pytest -v

# Run specific test categories
pytest -v -m integration tests/integration/
pytest -v -m performance tests/performance/
pytest -v -m chaos tests/resilience/

# Run with coverage
pytest --cov=src tests/

# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/

# Parallel execution
pytest -v -n auto tests/vector_db/

# Generate HTML test report
pytest --html=reports/test_results.html tests/
```

### System Validation
```bash
# Complete system health check
python tests/validate_system.py

# Run test runner with automation
python tests/test_runner.py
python tests/test_runner.py --suite integration
python tests/test_runner.py --exclude-slow
```

## Code Quality Commands

### Formatting
```bash
# Format Python code
python -m black .
python -m black src/ tests/

# Check formatting without changes
python -m black --check .
```

### Linting
```bash
# Run flake8 linting
python -m flake8 src/ tests/
flake8 --max-line-length=88 src/

# Run with specific rules
flake8 --select=E,W,F src/
```

### Type Checking
```bash
# Run mypy type checking
python -m mypy src/
mypy --strict src/

# Check specific modules
mypy src/agents/ src/pipeline/
```

## Docker & Service Commands

### Service Management
```bash
# Start all services
docker-compose -f docker-compose.complete.yml up -d

# Check service health
./scripts/health_check.sh

# View logs
docker-compose -f docker-compose.complete.yml logs -f

# Restart specific service
docker-compose -f docker-compose.complete.yml restart intelowl-django

# Stop all services
docker-compose -f docker-compose.complete.yml down
```

### Container Operations
```bash
# Access container shell
docker exec -it bev_intelowl_django bash
docker exec -it bev_postgres psql -U researcher -d osint

# Check container stats
docker stats

# Clean up containers
docker-compose -f docker-compose.complete.yml down -v
```

## Performance & Monitoring

### Performance Testing
```bash
# Run comprehensive performance tests
cd tests && python -m pytest performance/ -v

# Specific performance metrics
python tests/performance/test_request_multiplexing.py
python tests/performance/test_cache_performance.py
```

### Monitoring Commands
```bash
# Check Prometheus metrics
curl http://localhost:9090/api/v1/label/__name__/values

# Grafana health
curl http://localhost:3000/api/health

# Custom metrics validation
python tests/monitoring/test_metrics_integration.py
```

## Development Workflow

### Before Committing
```bash
# Run complete quality check
python -m black .
python -m flake8 src/ tests/
python -m mypy src/
pytest --cov=src tests/

# Quick validation
./validate_bev_deployment.sh
```

### CI/CD Pipeline Commands
```bash
# Full pipeline simulation
./run_all_tests.sh --parallel
python -m black --check .
python -m flake8 src/ tests/
python -m mypy src/
pytest --cov=src --cov-report=xml tests/
```

## Test Configuration

### Environment Setup
```bash
# Test environment variables in .env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
REDIS_HOST=localhost
QDRANT_URL=http://localhost:6333
PROMETHEUS_URL=http://localhost:9090
```

### Test Data Management
```bash
# Setup test environment
cd tests && ./setup_test_environment.sh

# Clean test data
python tests/conftest.py --clean-test-data
```

## Performance Targets
- **Concurrent Requests**: 1000+ simultaneous
- **Response Latency**: <100ms average
- **Cache Hit Rate**: >80% efficiency
- **Recovery Time**: <5 minutes after failures
- **System Availability**: 99.9% uptime
- **Vector Search**: <50ms query response