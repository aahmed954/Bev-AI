# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The BEV OSINT Framework is a comprehensive cybersecurity research platform that integrates IntelOwl, Neo4j, and custom analyzers for intelligence gathering and threat analysis. It operates as a single-user deployment optimized for performance with no authentication, designed specifically for authorized security research and academic purposes.

**⚠️ Security Note**: This system has NO AUTHENTICATION and should NEVER be exposed to public networks. It's designed for private network deployment and authorized security research only.

## Essential Commands

### System Lifecycle
```bash
# Complete system deployment
./deploy_everything.sh

# System health validation
./validate_bev_deployment.sh

# Complete system shutdown
docker-compose -f docker-compose.complete.yml down

# Emergency recovery
./verify_deployment.sh
```

### Development Workflow
```bash
# Code quality pipeline (run before commits)
python -m black . && python -m flake8 src/ tests/ && python -m mypy src/

# Complete test suite
./run_all_tests.sh

# Quick parallel testing
./run_all_tests.sh --parallel --quick

# System validation during development
python tests/validate_system.py
```

### Service Management
```bash
# Monitor all services
docker-compose -f docker-compose.complete.yml ps

# View aggregated logs
docker-compose -f docker-compose.complete.yml logs -f

# Restart specific service
docker-compose -f docker-compose.complete.yml restart bev_postgres
```

## Architecture Overview

### Core Components
- **IntelOwl Platform**: Web interface at http://localhost with dark theme
- **Custom Analyzers**: BreachDatabase, DarknetMarket, CryptoTracker, SocialMedia analyzers in `intelowl/custom_analyzers/`
- **MCP Server**: FastAPI-based server in `src/mcp_server/` providing OSINT tools via WebSocket/REST
- **Graph Visualization**: Cytoscape.js integration at http://localhost/cytoscape
- **Data Storage**: PostgreSQL (primary), Neo4j (graphs), Redis (cache), Elasticsearch (search)

### Service Architecture
The system runs as a microservices architecture orchestrated via Docker Compose:
- **Frontend Layer**: IntelOwl web interface + Cytoscape visualization
- **API Layer**: MCP server (`src/mcp_server/`) with tool registry and protocol handling
- **Processing Layer**: Custom analyzers + Celery workers + RabbitMQ messaging
- **Storage Layer**: Multi-database architecture (PostgreSQL, Neo4j, Redis, Elasticsearch)
- **Monitoring Layer**: Prometheus + Grafana stack
- **Security Layer**: Tor proxy integration for anonymized requests

### Key Service Endpoints
- IntelOwl Dashboard: http://localhost
- Neo4j Browser: http://localhost:7474 (neo4j/BevGraphMaster2024)
- Grafana Monitoring: http://localhost:3000 (admin/admin)
- MCP API Server: http://localhost:3010
- Prometheus Metrics: http://localhost:9090

## Development Patterns

### Source Code Organization
- `src/mcp_server/`: FastAPI-based MCP server with tool registry
- `src/pipeline/`: Data processing pipelines for OSINT workflows
- `src/security/`: Security modules and authentication (though disabled by default)
- `src/agents/`: AI agents for automated analysis
- `intelowl/custom_analyzers/`: OSINT analyzers (Breach, Darknet, Crypto, Social)

### Testing Framework
The project uses a comprehensive testing suite in `tests/` with specialized categories:
- `integration/`: Service connectivity and database integration
- `performance/`: Load testing (target: 1000+ concurrent requests, <100ms latency)
- `resilience/`: Chaos engineering and failure recovery
- `end_to_end/`: Complete OSINT investigation workflows
- `security/`: Security validation and penetration testing

### Configuration Management
- `.env`: API keys and database credentials (never commit sensitive values)
- `docker-compose.complete.yml`: Complete service orchestration
- `tests/test_config.yaml`: Testing parameters and performance targets
- Individual service configs in respective directories

## Database Architecture

### Multi-Database Design
- **PostgreSQL**: Primary data store with pgvector for semantic search
- **Neo4j**: Graph relationships and network analysis (bolt://localhost:7687)
- **Redis**: Session storage, caching, and rate limiting
- **Elasticsearch**: Full-text search and analytics indexing

### Database Access Patterns
```bash
# PostgreSQL (primary data)
docker exec -it bev_postgres psql -U researcher -d osint

# Neo4j (graph data)
# Web: http://localhost:7474 or bolt://localhost:7687

# Redis (cache/sessions)
docker exec -it bev_redis redis-cli
```

## Custom Analyzer Development

### Analyzer Structure
Custom analyzers inherit from IntelOwl's base analyzer class and are located in `intelowl/custom_analyzers/`:
- **BreachDatabaseAnalyzer**: Searches Dehashed, Snusbase, WeLeakInfo
- **DarknetMarketAnalyzer**: Scrapes AlphaBay, White House, Torrez via Tor
- **CryptoTrackerAnalyzer**: Bitcoin/Ethereum transaction analysis
- **SocialMediaAnalyzer**: Instagram, Twitter, LinkedIn profiling

### MCP Tool Development
OSINT tools are registered in `src/mcp_server/tools.py` and exposed via the MCP protocol:
- Inherit from `OSINTToolBase`
- Implement async `execute()` method
- Register with `OSINTToolRegistry`
- Support WebSocket and REST interfaces

## Performance Requirements

### System Targets
- **Concurrent Requests**: 1000+ simultaneous connections
- **Response Latency**: <100ms average response time
- **Cache Hit Rate**: >80% efficiency with predictive caching
- **Recovery Time**: <5 minutes after chaos engineering failures
- **System Availability**: 99.9% uptime target

### Performance Validation
```bash
# Performance test suite
pytest tests/performance/ -v

# Specific performance metrics
python tests/performance/test_request_multiplexing.py

# System resource monitoring
docker stats
```

## Security Considerations

### Network Security
- Tor integration via SOCKS5 proxy (socks5://localhost:9050)
- No external authentication (single-user deployment)
- All traffic should route through private networks only
- Firewall rules to block external access

### Data Protection
- Sensitive OSINT data encrypted at rest
- API keys managed via environment variables
- No logging of sensitive intelligence data
- Automatic data retention policies

### Operational Security
- Never expose to public internet
- Use only on isolated research networks
- Follow responsible disclosure for discovered vulnerabilities
- Maintain audit logs for compliance purposes

## Monitoring and Observability

### Health Monitoring
```bash
# Comprehensive health check
./scripts/health_check.sh

# Individual service health
docker-compose -f docker-compose.complete.yml ps

# Performance metrics
curl http://localhost:9090/metrics | grep bev_
```

### Key Metrics
- `bev_request_count`: API request volume
- `bev_tool_executions`: OSINT tool usage
- `bev_osint_analyses_total`: Investigation volume
- `bev_cache_hit_rate`: Cache efficiency
- `bev_threat_detections`: Security alerts

## Troubleshooting

### Common Issues
- **Services not starting**: Check Docker daemon and run `./validate_bev_deployment.sh`
- **Database connection failures**: Verify credentials in `.env` and service health
- **Poor performance**: Monitor resource usage with `docker stats` and Grafana
- **Tor connectivity issues**: Restart Tor service with `docker-compose restart bev_tor`

### Emergency Procedures
```bash
# Complete system reset
docker-compose -f docker-compose.complete.yml down -v && ./deploy_everything.sh

# Database recovery
./scripts/backup_databases.sh  # (run regularly)
./scripts/restore_databases.sh

# Security incident response
./scripts/emergency_isolation.sh
```

## Development Environment

### Prerequisites
- Ubuntu 20.04+ or compatible Linux
- Docker 20.10+ with Docker Compose v2
- 16GB RAM minimum (32GB recommended)
- 500GB SSD storage
- Network access for Tor proxy capabilities

### Quality Gates
All changes must pass:
1. Code formatting (`python -m black .`)
2. Linting (`python -m flake8 src/ tests/`)
3. Type checking (`python -m mypy src/`)
4. Unit tests (`pytest tests/ --cov=src`)
5. System validation (`./validate_bev_deployment.sh`)
6. Integration tests (`pytest tests/integration/ -v`)

## Legal and Compliance

This framework is designed for:
- Authorized cybersecurity research
- Academic and educational purposes
- Professional threat intelligence analysis
- Compliance with applicable laws and regulations

**Important**: Users are responsible for ensuring all activities comply with local laws, institutional policies, and ethical guidelines for security research.