# BEV OSINT Framework - Enterprise CLAUDE.md

**Enterprise-Grade Cybersecurity Intelligence Platform**

This file provides comprehensive guidance to Claude Code when working with the BEV OSINT Framework - a production-ready, enterprise-scale platform comparable to Palantir Gotham and Maltego.

## 🏢 Platform Overview

The BEV OSINT Framework is a **comprehensive enterprise cybersecurity intelligence platform** with multi-node distributed architecture, desktop applications, global edge computing, and advanced automation. With **130,534+ lines of production code** across **151+ microservices**, this represents a complete enterprise ecosystem for authorized cybersecurity research and threat intelligence operations.

**⚠️ Enterprise Security Note**: This platform features sophisticated multi-node architecture with HashiCorp Vault credential management, Tor anonymization, and enterprise-grade security. It's designed exclusively for authorized cybersecurity research in secure environments.

## 🚀 Essential Commands

### Enterprise Deployment
```bash
# Production deployment (working enterprise system)
./deploy_bev_real_implementations.sh

# Multi-node distributed deployment with Vault
./deploy-complete-with-vault.sh

# Deployment validation
./validate_bev_deployment.sh
./verify_multinode_deployment.sh

# Emergency rollback
./rollback_bev_deployment.sh
```

### Development Workflow
```bash
# Complete code quality pipeline
python -m black . && python -m flake8 src/ tests/ && python -m mypy src/

# Comprehensive test suite
./run_all_tests.sh

# Performance validation (1000+ concurrent, <100ms latency)
./run_all_tests.sh --parallel --performance

# Security validation
python run_security_tests.py

# System health verification
./validate_bev_deployment.sh
```

### Desktop Application Management
```bash
# Tauri desktop application deployment
./bev-complete-frontend.sh

# Frontend security validation
cd bev-frontend && ./validate-security.sh

# Desktop app development server
cd bev-frontend && npm run tauri dev
```

### Workflow Orchestration
```bash
# Airflow DAG management
airflow dags list
airflow dags trigger research_pipeline_dag
airflow dags trigger bev_health_monitoring

# N8N workflow automation
docker-compose -f docker-compose.complete.yml exec n8n n8n list
```

## 🏗️ Enterprise Architecture

### Multi-Node Distributed Infrastructure
- **THANOS Node**: Primary compute, 89 services, GPU acceleration (RTX 3080)
- **ORACLE1 Node**: ARM optimization, 62 services, edge computing
- **STARLORD Node**: Development environment, 12 services, Vault coordination
- **Global Edge Network**: 4-region deployment (US-East, US-West, EU-Central, Asia-Pacific)

### Major Platform Components

#### 1. Alternative Market Intelligence (`src/alternative_market/`)
**5,608+ lines of production code**
- **DarkNet Market Crawler** (`dm_crawler.py`): Advanced Tor-based market intelligence
- **Cryptocurrency Analyzer** (`crypto_analyzer.py`): Bitcoin/Ethereum transaction analysis
- **Reputation Systems** (`reputation_analyzer.py`): Actor reputation analysis
- **Economic Intelligence** (`economics_processor.py`): Market economics analysis

#### 2. Security Operations Center (`src/security/`)
**11,189+ lines of production code**
- **Intelligence Fusion** (`intel_fusion.py`): Multi-source threat intelligence
- **OpSec Enforcement** (`opsec_enforcer.py`): Operational security automation
- **Defense Automation** (`defense_automation.py`): Automated threat response
- **Tactical Intelligence** (`tactical_intelligence.py`): Real-time threat analysis

#### 3. Autonomous Systems (`src/autonomous/`)
**8,377+ lines of production code**
- **Enhanced Autonomous Controller** (`enhanced_autonomous_controller.py`): AI-driven operations
- **Adaptive Learning** (`adaptive_learning.py`): Machine learning adaptation
- **Knowledge Evolution** (`knowledge_evolution.py`): Continuous learning systems
- **Resource Optimization** (`resource_optimizer.py`): Dynamic resource management

#### 4. Tauri Desktop Application (`bev-frontend/`)
**112 Svelte components, complete desktop ecosystem**
- **Rust Backend** (`src-tauri/`): High-performance desktop backend
- **Svelte Frontend** (`src/`): Modern reactive UI framework
- **Knowledge Graph UI**: Interactive graph visualization
- **Security Validation**: Built-in security testing
- **SSL Integration**: Enterprise certificate management

#### 5. Airflow Orchestration (`dags/`)
**1,812 lines of production workflow code**
- **Research Pipeline** (`research_pipeline_dag.py`): OSINT investigation workflows
- **Health Monitoring** (`bev_health_monitoring.py`): System health orchestration
- **Data Lake Processing** (`data_lake_medallion_dag.py`): Data pipeline management
- **ML Training Pipeline** (`ml_training_pipeline_dag.py`): Machine learning workflows
- **Cost Optimization** (`cost_optimization_dag.py`): Resource optimization

#### 6. Edge Computing Network (`src/edge/`)
**Global 4-region infrastructure**
- **Edge Management Service**: Distributed node management
- **Geographic Routing**: Intelligent traffic routing
- **Model Synchronization**: AI model distribution
- **Regional Deployment Scripts**: Automated edge deployment

#### 7. Chaos Engineering (`chaos-engineering/`)
**Production resilience testing**
- **Chaos Engineer** (`src/testing/chaos_engineer.py`): Resilience testing automation
- **Chaos API** (`src/testing/chaos_api.py`): Programmatic chaos introduction
- **Experiment Framework**: Structured chaos experiments
- **Recovery Validation**: Automated recovery testing

#### 8. Tor Network Infrastructure (`tor/`)
**Multi-node anonymous networking**
- **Entry/Middle/Exit Nodes**: Complete Tor network deployment
- **Traffic Anonymization**: Advanced traffic routing
- **Monitoring Integration**: Network health monitoring
- **Docker Orchestration**: Containerized Tor services

### Service Architecture Layers
```
┌─────────────────────────────────────────────────────────────┐
│ Desktop Layer: Tauri Apps (112 components)                 │
├─────────────────────────────────────────────────────────────┤
│ API Layer: MCP Servers + FastAPI + REST/WebSocket          │
├─────────────────────────────────────────────────────────────┤
│ Orchestration: Airflow (5 DAGs) + N8N Workflows           │
├─────────────────────────────────────────────────────────────┤
│ Processing: Alternative Market + Security + Autonomous      │
├─────────────────────────────────────────────────────────────┤
│ Storage: PostgreSQL + Neo4j + Redis + Elasticsearch        │
├─────────────────────────────────────────────────────────────┤
│ Infrastructure: Vault + Tor + Chaos + Edge Network         │
└─────────────────────────────────────────────────────────────┘
```

### Enterprise Service Endpoints
```yaml
# Multi-Node Access Points
THANOS_SERVICES:
  - Neo4j: http://100.122.12.54:7474 (neo4j/BevGraphMaster2024)
  - PostgreSQL: 100.122.12.54:5432
  - GPU Services: 100.122.12.54:8080-8090

ORACLE1_SERVICES:
  - Grafana: http://100.96.197.84:3000 (admin/admin)
  - Prometheus: http://100.96.197.84:9090
  - Edge Gateway: 100.96.197.84:8443

STARLORD_SERVICES:
  - Vault UI: http://100.122.12.35:8200/ui
  - Development: localhost:3000-3010
  - MCP Servers: localhost:3011-3020

# Global Edge Network
EDGE_REGIONS:
  - US-East: edge-us-east.bev-network.local
  - US-West: edge-us-west.bev-network.local
  - EU-Central: edge-eu-central.bev-network.local
  - Asia-Pacific: edge-asia-pacific.bev-network.local
```

## 💾 Enterprise Data Architecture

### Multi-Database Ecosystem
- **PostgreSQL**: Primary data store (THANOS), vector search capabilities
- **Neo4j**: Graph relationships (THANOS), network analysis
- **Redis**: Distributed cache (multi-node), session management
- **Elasticsearch**: Search engine (ORACLE1), analytics indexing
- **InfluxDB**: Time-series data (ORACLE1), metrics storage
- **Qdrant**: Vector database (STARLORD), semantic search
- **Weaviate**: Knowledge graphs (distributed), AI-powered search

### Distributed Storage Patterns
```bash
# Multi-node database access
# PostgreSQL (THANOS)
PGPASSWORD=researcher_pass psql -h 100.122.12.54 -U researcher -d osint

# Neo4j (THANOS)
cypher-shell -a bolt://100.122.12.54:7687 -u neo4j -p BevGraphMaster2024

# Redis (distributed)
redis-cli -h 100.122.12.54 -p 6379
redis-cli -h 100.96.197.84 -p 6379

# Vault (STARLORD coordination)
export VAULT_ADDR="http://100.122.12.35:8200"
vault auth -method=token
```

## 🔐 Enterprise Security Architecture

### HashiCorp Vault Integration
```bash
# Vault initialization and management
./setup-vault-multinode.sh

# Generate secure credentials
./generate-secure-credentials.sh

# Vault UI access
export VAULT_ADDR="http://100.122.12.35:8200"
vault auth -method=approle
```

### Security Components
- **AppRole Authentication**: Node-based role authentication
- **Dynamic Secrets**: Vault-generated database credentials
- **Tor Network Integration**: Multi-node anonymous networking
- **Tailscale VPN**: Cross-node secure communication
- **Certificate Management**: SSL/TLS automation
- **OPSEC Enforcement**: Automated operational security

### Network Security
```yaml
SECURITY_LAYERS:
  - Network: Tailscale VPN (100.x.x.x addresses)
  - Application: Vault credential management
  - Transport: SSL/TLS encryption
  - Anonymization: Multi-node Tor integration
  - Monitoring: Real-time security analytics
```

## 🎯 Development Patterns

### Enterprise Code Organization
```
src/
├── alternative_market/     # Market intelligence (5,608 lines)
├── security/               # Security operations (11,189 lines)
├── autonomous/             # Autonomous systems (8,377 lines)
├── edge/                   # Global edge computing
├── pipeline/               # Data processing pipelines
├── infrastructure/         # Infrastructure management
├── monitoring/             # System monitoring
├── testing/                # Chaos engineering
└── mcp_server/             # MCP protocol servers

bev-frontend/               # Tauri desktop application
├── src/                    # Svelte frontend (112 components)
├── src-tauri/              # Rust backend
└── config/                 # Desktop app configuration

dags/                       # Airflow orchestration (1,812 lines)
├── research_pipeline_dag.py
├── bev_health_monitoring.py
├── data_lake_medallion_dag.py
├── ml_training_pipeline_dag.py
└── cost_optimization_dag.py

chaos-engineering/          # Resilience testing
├── experiments/
├── scenarios/
└── monitoring/

tor/                        # Anonymous networking
├── torrc_node1
├── torrc_node2
├── torrc_node3
└── monitoring/
```

### Enterprise Testing Framework
```bash
# Comprehensive testing suite
./run_all_tests.sh --enterprise

# Performance testing (enterprise targets)
pytest tests/performance/ -v --concurrent=1000 --latency=100ms

# Chaos engineering
python src/testing/chaos_engineer.py --experiment=network_partition

# Multi-node integration testing
./verify_multinode_deployment.sh

# Security validation
python run_security_tests.py --comprehensive

# Desktop application testing
cd bev-frontend && npm run test:tauri
```

### Service Development Guidelines
```python
# Enterprise service pattern
class EnterpriseOSINTService:
    """Base class for all BEV enterprise services"""

    def __init__(self):
        self.vault_client = VaultClient()
        self.metrics_collector = PrometheusMetrics()
        self.chaos_monkey = ChaosEngineer()

    async def process_intelligence(self, data):
        # Vault-secured processing
        credentials = await self.vault_client.get_dynamic_secret()

        # Metrics collection
        with self.metrics_collector.time_operation():
            result = await self._analyze_data(data)

        # Chaos testing integration
        await self.chaos_monkey.introduce_controlled_failure()

        return result
```

## 📊 Enterprise Performance Standards

### System Performance Targets
- **Concurrent Users**: 1000+ simultaneous connections
- **Response Latency**: <100ms average, <500ms P99
- **Throughput**: 10,000+ requests/second
- **Cache Hit Rate**: >80% efficiency
- **Recovery Time**: <5 minutes (RTO)
- **System Availability**: 99.9% uptime (SLA)

### Resource Requirements
```yaml
THANOS_NODE:
  CPU: 18 cores
  RAM: 50GB
  GPU: NVIDIA RTX 3080 (6.5GB VRAM)
  Storage: 1TB NVMe SSD

ORACLE1_NODE:
  CPU: 3 cores (ARM optimized)
  RAM: 15GB
  Storage: 500GB SSD

STARLORD_NODE:
  CPU: 8 cores
  RAM: 32GB
  Storage: 500GB SSD

NETWORK:
  Tailscale VPN: 100Mbps minimum
  Internet: 1Gbps for Tor network
```

### Performance Validation
```bash
# Enterprise performance suite
./run_comprehensive_tests.sh --performance

# Load testing
k6 run tests/performance/load_test.js --vus=1000 --duration=10m

# Database performance
python tests/performance/test_distributed_queries.py

# Edge network latency
python tests/performance/test_edge_latency.py --regions=all

# Desktop app performance
cd bev-frontend && npm run test:performance
```

## 🌐 Global Infrastructure Management

### Edge Computing Network
```bash
# Deploy to specific regions
./scripts/edge_deployment/deploy_edge_us_east.sh
./scripts/edge_deployment/deploy_edge_us_west.sh
./scripts/edge_deployment/deploy_edge_eu_central.sh
./scripts/edge_deployment/deploy_edge_asia_pacific.sh

# Global deployment coordination
./deploy_multinode_bev.sh --edge-regions=all

# Edge network monitoring
curl http://edge-us-east.bev-network.local/health
curl http://edge-eu-central.bev-network.local/metrics
```

### Deployment Automation
**47+ deployment scripts** for enterprise orchestration:
- **Master Deployment Controller**: `master-deployment-controller.sh`
- **Intelligent Distribution**: `deploy-intelligent-distributed.sh`
- **Phase-based Rollouts**: `deployment_phases/phase[7-9]/`
- **Edge Network Deployment**: `scripts/edge_deployment/`
- **Vault Integration**: `setup-vault-multinode.sh`
- **Security Hardening**: `fix_security_critical.sh`

## 🔄 Workflow Orchestration

### Airflow Enterprise DAGs
```python
# Example: Research Pipeline DAG (127 lines)
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

research_pipeline = DAG(
    'research_pipeline_dag',
    schedule_interval='@daily',
    default_args={
        'retries': 3,
        'retry_delay': timedelta(minutes=5)
    }
)

# Health Monitoring DAG (816 lines)
# Comprehensive system health orchestration
health_monitoring = DAG(
    'bev_health_monitoring',
    schedule_interval='*/5 * * * *',  # Every 5 minutes
    catchup=False
)

# ML Training Pipeline DAG (301 lines)
# Automated model training and deployment
ml_training = DAG(
    'ml_training_pipeline_dag',
    schedule_interval='@weekly'
)
```

### N8N Workflow Automation
```json
// Intelligence Gathering Workflow
{
  "name": "intelligence_gathering",
  "nodes": [
    {
      "name": "DarkWeb_Monitor",
      "type": "n8n-nodes-bev-darkweb"
    },
    {
      "name": "Crypto_Tracker",
      "type": "n8n-nodes-bev-crypto"
    },
    {
      "name": "Threat_Analyzer",
      "type": "n8n-nodes-bev-threat"
    }
  ]
}
```

## 🧪 Chaos Engineering

### Resilience Testing Framework
```bash
# Chaos engineering experiments
python src/testing/chaos_engineer.py --experiment=database_failure
python src/testing/chaos_engineer.py --experiment=network_partition
python src/testing/chaos_engineer.py --experiment=service_overload

# Chaos API integration
curl -X POST http://localhost:8080/chaos/experiment \
  -d '{"type": "latency", "target": "postgres", "duration": "5m"}'

# Recovery validation
./scripts/validate_recovery.sh --scenario=node_failure
```

### Chaos Scenarios
- **Database Failures**: PostgreSQL/Neo4j connection loss
- **Network Partitions**: Inter-node communication failures
- **Service Overload**: High traffic simulation
- **Memory Pressure**: Resource exhaustion testing
- **Disk Failures**: Storage subsystem failures

## 🛠️ Troubleshooting & Operations

### Enterprise Monitoring
```bash
# Comprehensive health check
./validate_bev_deployment.sh --enterprise

# Multi-node system health
./verify_multinode_deployment.sh --detailed

# Service dependency validation
python scripts/validate_service_dependencies.py

# Performance monitoring
curl http://100.96.197.84:9090/api/v1/query?query=bev_request_rate
```

### Common Enterprise Issues
```yaml
DEPLOYMENT_BLOCKERS:
  - Docker Compose validation: "docker-compose config"
  - Vault initialization: "./setup-vault-multinode.sh"
  - Environment variables: "source .env.complete"
  - Node connectivity: "tailscale status"

PERFORMANCE_ISSUES:
  - Database optimization: PostgreSQL/Neo4j tuning
  - Cache warming: Redis pre-population
  - Edge latency: Regional deployment validation
  - Resource allocation: CPU/memory optimization

SECURITY_CONCERNS:
  - Vault token rotation: AppRole policy enforcement
  - Tor network health: Multi-node connectivity
  - Certificate expiration: SSL/TLS monitoring
  - Network isolation: Firewall rule validation
```

### Emergency Procedures
```bash
# Complete system recovery
./scripts/emergency_recovery.sh --full-restore

# Multi-node rollback
./rollback_bev_deployment.sh --preserve-vault

# Security incident isolation
./scripts/emergency_isolation.sh --network-lockdown

# Database backup and restore
./scripts/backup_all_databases.sh --vault-encrypted
./scripts/restore_from_backup.sh --timestamp=latest
```

## 📋 Enterprise Compliance

### Legal and Regulatory Framework
- **Authorized Cybersecurity Research**: Enterprise deployment framework
- **Academic Institution Compliance**: University security research programs
- **Professional Threat Intelligence**: Corporate security operations
- **Regulatory Adherence**: GDPR, SOX, HIPAA compliance capabilities
- **Ethical Guidelines**: Responsible disclosure and research ethics

### Audit and Compliance
```bash
# Compliance validation
python scripts/compliance_audit.py --framework=SOX
python scripts/compliance_audit.py --framework=GDPR

# Security audit
./run_security_audit.sh --comprehensive

# Data retention validation
python scripts/data_retention_audit.py --policy=enterprise
```

## 🎓 Enterprise Training and Documentation

### Learning Paths
1. **Platform Architecture**: Multi-node distributed systems
2. **OSINT Operations**: Alternative market intelligence
3. **Security Operations**: Threat intelligence and response
4. **Chaos Engineering**: Resilience testing and validation
5. **Desktop Applications**: Tauri and Svelte development
6. **Workflow Orchestration**: Airflow and N8N automation

### Advanced Topics
- **Vault Integration**: HashiCorp Vault credential management
- **Edge Computing**: Global distributed deployments
- **Autonomous Systems**: AI-driven operations
- **Graph Analytics**: Neo4j and network analysis
- **Performance Optimization**: Enterprise-scale optimization

---

**Enterprise Deployment Ready**: This BEV OSINT Framework represents a complete, production-ready enterprise platform with over 130,534 lines of code, 151+ microservices, multi-node architecture, desktop applications, global edge computing, and advanced automation capabilities.

**Next Actions for Enterprise Deployment**:
1. Execute: `./deploy_bev_real_implementations.sh`
2. Initialize Vault: `./setup-vault-multinode.sh`
3. Validate deployment: `./verify_multinode_deployment.sh`
4. Deploy edge network: Regional deployment scripts
5. Launch desktop applications: `./bev-complete-frontend.sh`

**Prepared by**: BEV Enterprise Development Team
**Classification**: Authorized Cybersecurity Research Platform
**Version**: Enterprise Completion Branch (September 2025)