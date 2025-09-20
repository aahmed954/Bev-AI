# BEV OSINT Framework - Essential Commands Reference

## 🚀 Quick Start Commands

### System Startup
```bash
# Start complete BEV system
./deploy_everything.sh

# Verify system health
./validate_bev_deployment.sh

# Check all services running
docker-compose -f docker-compose.complete.yml ps
```

### Daily Development
```bash
# Format and lint code
python -m black . && python -m flake8 src/ tests/

# Run quick tests
python tests/validate_system.py

# Type checking
python -m mypy src/
```

## 🧪 Testing Commands

### Master Test Suite
```bash
# Run all tests
./run_all_tests.sh

# Quick parallel execution
./run_all_tests.sh --parallel --quick

# Specific test categories
./run_all_tests.sh --skip-performance --skip-security
```

### Individual Testing
```bash
# System validation
./validate_bev_deployment.sh

# Python tests with coverage
pytest --cov=src tests/

# Performance tests
pytest -v tests/performance/

# Generate HTML test report
pytest --html=reports/test_results.html tests/
```

## 🔧 Service Management

### Docker Operations
```bash
# Start all services
docker-compose -f docker-compose.complete.yml up -d

# Stop all services
docker-compose -f docker-compose.complete.yml down

# View logs
docker-compose -f docker-compose.complete.yml logs -f

# Restart specific service
docker-compose -f docker-compose.complete.yml restart bev_postgres
```

### Database Access
```bash
# PostgreSQL
docker exec -it bev_postgres psql -U researcher -d osint

# Neo4j browser: http://localhost:7474 (neo4j/BevGraphMaster2024)

# Redis CLI
docker exec -it bev_redis redis-cli
```

## 📊 Monitoring & Health

### System Health
```bash
# Complete health check
./scripts/health_check.sh

# Service status
docker stats

# Application metrics
curl http://localhost:9090/metrics | grep bev_
```

### Performance Monitoring
```bash
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000

# Performance tests
python tests/performance/test_request_multiplexing.py
```

## 🛠 Development Workflow

### Code Quality
```bash
# Complete quality check (run before committing)
python -m black . && \
python -m flake8 src/ tests/ && \
python -m mypy src/ && \
pytest --cov=src tests/
```

### Git Operations
```bash
# Standard workflow
git status && git add . && git commit -m "description"

# Create feature branch
git checkout -b feature/analyzer-name
```

## 🔍 Access Points

### Web Interfaces
- **IntelOwl Dashboard**: http://localhost
- **Cytoscape Graph**: http://localhost/cytoscape  
- **Neo4j Browser**: http://localhost:7474
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **RabbitMQ**: http://localhost:15672

### Database Connections
```bash
# PostgreSQL: postgresql://bev:BevOSINT2024@localhost:5432/osint
# Neo4j: bolt://localhost:7687 (neo4j/BevGraphMaster2024)
# Redis: redis://:BevCacheMaster@localhost:6379
```

## 🚨 Troubleshooting

### Common Issues
```bash
# Service not starting
docker logs bev_service_name

# Database connection issues
docker exec bev_postgres pg_isready -U researcher

# Clear Redis cache
docker exec bev_redis redis-cli FLUSHALL

# Restart problematic services
docker-compose restart bev_postgres bev_redis
```

### Emergency Commands
```bash
# Stop everything
docker-compose -f docker-compose.complete.yml down

# Clean restart
docker-compose -f docker-compose.complete.yml down && \
./deploy_everything.sh

# System recovery
./verify_deployment.sh
```

## 📋 Task Completion Checklist

### Before Task Completion
1. **Code Quality**: `python -m black . && python -m flake8 src/`
2. **Type Checking**: `python -m mypy src/`
3. **Testing**: `pytest --cov=src tests/`
4. **System Validation**: `./validate_bev_deployment.sh`
5. **Integration Tests**: `pytest tests/integration/ -v`

### When Task is Complete
1. **Final Test Suite**: `./run_all_tests.sh --quick`
2. **Health Check**: `./scripts/health_check.sh`
3. **Documentation**: Update relevant docs if needed
4. **Git Commit**: Clear commit message describing changes

## 💡 Pro Tips

- **Always validate** system health before and after changes
- **Use parallel testing** for faster feedback: `./run_all_tests.sh --parallel`
- **Monitor logs** during development: `docker-compose logs -f`
- **Keep services running** - no need to restart Docker containers frequently
- **Use test-driven development** - write tests first when adding new features
- **Check performance impact** with performance tests after major changes