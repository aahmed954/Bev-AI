# BEV OSINT Framework - Development Workflow & Commands

## Daily Development Commands

### Project Startup
```bash
# Start the complete BEV system
./deploy_everything.sh

# Start individual services
docker-compose -f docker-compose.complete.yml up -d

# Verify system health
./validate_bev_deployment.sh

# Check service status
docker-compose -f docker-compose.complete.yml ps
```

### Development Environment
```bash
# Activate Python environment (if using virtual env)
source ~/ml/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Install test dependencies
pip install -r tests/requirements.txt
```

### Code Quality Workflow
```bash
# 1. Format code
python -m black src/ tests/

# 2. Lint code
python -m flake8 src/ tests/

# 3. Type checking
python -m mypy src/

# 4. Run tests
pytest tests/ -v

# 5. Run with coverage
pytest --cov=src tests/
```

### Testing Workflow
```bash
# Quick system validation
python tests/validate_system.py

# Run specific test suites
pytest -v tests/integration/
pytest -v tests/performance/
pytest -v tests/security/

# Run complete test suite
./run_all_tests.sh

# Generate test reports
pytest --html=reports/test_results.html tests/
```

## Service Management Commands

### Docker Operations
```bash
# Start all services
docker-compose -f docker-compose.complete.yml up -d

# Stop all services  
docker-compose -f docker-compose.complete.yml down

# Restart specific service
docker-compose -f docker-compose.complete.yml restart bev_postgres

# View logs
docker-compose -f docker-compose.complete.yml logs -f

# Execute commands in containers
docker exec -it bev_intelowl_django bash
docker exec -it bev_postgres psql -U researcher -d osint
```

### Health Monitoring
```bash
# System health check
./scripts/health_check.sh

# Service connectivity
python tests/integration/test_service_connectivity.py

# Performance monitoring
curl http://localhost:9090/metrics  # Prometheus
curl http://localhost:3000/api/health  # Grafana
```

## Database Operations

### PostgreSQL
```bash
# Connect to database
docker exec -it bev_postgres psql -U researcher -d osint

# Database backup
docker exec bev_postgres pg_dump -U researcher osint > backup.sql

# Database restore
docker exec -i bev_postgres psql -U researcher osint < backup.sql
```

### Neo4j
```bash
# Neo4j browser access
# http://localhost:7474 (neo4j/BevGraphMaster2024)

# Backup Neo4j
docker exec bev_neo4j neo4j-admin dump --to=/data/backup.dump

# Cypher queries
docker exec -it bev_neo4j cypher-shell -u neo4j -p BevGraphMaster2024
```

### Redis
```bash
# Redis CLI
docker exec -it bev_redis redis-cli

# Clear cache
docker exec bev_redis redis-cli FLUSHALL

# Monitor Redis
docker exec bev_redis redis-cli MONITOR
```

## Development Tasks

### Code Organization
- **Follow existing patterns** in the codebase
- **Use descriptive naming** for functions and variables
- **Type hints** for all Python functions
- **Docstrings** for public functions and classes
- **Error handling** with specific exceptions

### Code Style Guidelines
```python
# Python conventions
- Use type hints: def process_data(data: List[Dict]) -> Dict:
- Use f-strings: f"Processing {len(data)} items"
- Use pathlib: from pathlib import Path
- Async/await: async def fetch_data() -> Dict:
- Exception handling: raise ValueError("Invalid data format")
```

### Testing Guidelines
- **Write tests first** (TDD approach when possible)
- **Test at multiple levels**: unit, integration, end-to-end
- **Use fixtures** for test data setup
- **Mock external services** in unit tests
- **Performance tests** for critical paths

### Git Workflow
```bash
# Standard workflow
git status
git add .
git commit -m "feat: add breach database analyzer"
git push origin feature-branch

# Create feature branch
git checkout -b feature/new-analyzer
git checkout -b fix/performance-issue
git checkout -b docs/api-reference
```

## Production Deployment

### Pre-deployment Checklist
```bash
# 1. Run complete test suite
./run_all_tests.sh

# 2. Validate deployment
./validate_bev_deployment.sh

# 3. Security checks
./run_security_tests.py

# 4. Performance validation
pytest tests/performance/ -v

# 5. Generate deployment report
./verify_deployment.sh
```

### Deployment Commands
```bash
# Complete system deployment
./deploy_complete_system.sh

# Verify deployment
./verify_completeness.sh

# Monitor deployment
./deploy_bev_complete.sh --monitor
```

## Troubleshooting Commands

### Service Issues
```bash
# Check container status
docker ps -a

# View service logs
docker logs bev_intelowl_django
docker logs bev_postgres
docker logs bev_neo4j

# Restart problematic services
docker-compose restart bev_postgres bev_redis
```

### Database Issues
```bash
# PostgreSQL connection test
docker exec bev_postgres pg_isready -U researcher

# Neo4j status
docker exec bev_neo4j neo4j status

# Redis connectivity
docker exec bev_redis redis-cli ping
```

### Performance Issues
```bash
# System resource monitoring
docker stats

# Application metrics
curl http://localhost:9090/api/v1/query?query=up

# Performance profiling
python -m cProfile -o profile.stats your_script.py
```

## Maintenance Tasks

### Regular Maintenance
```bash
# Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete

# Database maintenance
docker exec bev_postgres vacuumdb -U researcher osint

# Docker cleanup
docker system prune -f
docker volume prune -f
```

### Backup Operations
```bash
# Complete system backup
./backups/create_backup.sh

# Database backups
./backups/backup_databases.sh

# Configuration backup
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
```

## Performance Optimization

### Monitoring Performance
```bash
# System metrics
htop
iotop
nethogs

# Application metrics
curl http://localhost:9090/metrics | grep bev_

# Database performance
docker exec bev_postgres pg_stat_activity
```

### Optimization Tasks
- **Database indexing**: Optimize query performance
- **Cache tuning**: Redis configuration optimization  
- **Connection pooling**: Optimize database connections
- **Memory management**: Monitor and tune memory usage