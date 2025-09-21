# BEV OSINT Framework - Project Index

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Components](#architecture-components)
3. [Source Code Structure](#source-code-structure)
4. [Configuration Systems](#configuration-systems)
5. [Deployment Infrastructure](#deployment-infrastructure)
6. [Testing Framework](#testing-framework)
7. [Documentation Map](#documentation-map)
8. [API Reference](#api-reference)
9. [Development Workflows](#development-workflows)
10. [Operational Procedures](#operational-procedures)

---

## 🎯 Project Overview

**BEV (Beyond Extreme Vision)** is a distributed OSINT intelligence platform designed for comprehensive threat intelligence, darknet monitoring, and security operations. The system operates in dual-server architecture with advanced autonomous capabilities.

### Key Characteristics
- **Deployment Mode**: Single-user, no authentication
- **Architecture**: Distributed microservices with THANOS/ORACLE1 nodes
- **Security Focus**: OPSEC-compliant with Tor integration
- **Scale**: Enterprise-grade with autonomous enhancement capabilities

---

## 🏗️ Architecture Components

### Core Systems
| Component | Location | Purpose | Dependencies |
|-----------|----------|---------|--------------|
| **Intelligence Hub** | `src/agents/` | Multi-agent coordination | Sequential reasoning, memory management |
| **Security Operations** | `src/security/` | Tactical intelligence, OPSEC | Traffic analysis, anomaly detection |
| **Infrastructure** | `src/infrastructure/` | Auto-recovery, proxy management | Circuit breakers, vector databases |
| **Pipeline Processing** | `src/pipeline/` | OCR, compression, multiplexing | Message queues, semantic deduplication |
| **Autonomous Systems** | `src/autonomous/` | Adaptive learning, evolution | Intelligence coordination, resource optimization |

### Service Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BEV OSINT FRAMEWORK                         │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   AGENTS    │  │  SECURITY   │  │ AUTONOMOUS  │             │
│  │             │  │             │  │             │             │
│  │ • Research  │  │ • Intel     │  │ • Learning  │             │
│  │ • Memory    │  │ • OPSEC     │  │ • Evolution │             │
│  │ • Swarm     │  │ • Defense   │  │ • Resource  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│           │               │               │                     │
│  ┌────────▼───────────────▼───────────────▼─────────────┐       │
│  │              INFRASTRUCTURE LAYER                    │       │
│  │                                                       │       │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │       │
│  │  │   PIPELINE  │  │   VECTORS   │  │   PROXIES   │   │       │
│  │  │             │  │             │  │             │   │       │
│  │  │ • OCR       │  │ • Embeddings│  │ • Tor       │   │       │
│  │  │ • Compress  │  │ • Database  │  │ • Circuit   │   │       │
│  │  │ • Multiplex │  │ • Cache     │  │ • Recovery  │   │       │
│  │  └─────────────┘  └─────────────┘  └─────────────┘   │       │
│  └───────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Source Code Structure

### Primary Modules

#### `/src/agents/` - Intelligence Coordination
```
agents/
├── __init__.py                    # Agent protocol definitions
├── agent_protocol.py              # Communication protocols
├── code-optimizer.py              # Code optimization agent
├── counterfactual_analyzer.py     # Alternative analysis
├── extended_reasoning.py          # Complex reasoning engine
├── extended_reasoning_service.py  # Reasoning service API
├── integration_client.py          # Service integration
├── knowledge_synthesizer.py       # Information synthesis
├── memory-manager.py              # Memory persistence
├── memory_manager.py              # Alternative memory system
├── research_coordinator.py        # Research orchestration
├── research_workflow.py           # Workflow management
├── swarm-orchestrator.py          # Multi-agent coordination
├── swarm_master.py                # Swarm control
└── tool-coordinator.py            # Tool management
```
**Purpose**: Multi-agent coordination with advanced reasoning capabilities
**Key Features**: Extended reasoning, memory persistence, swarm orchestration
**Dependencies**: `infrastructure/`, `pipeline/`, external reasoning services

#### `/src/security/` - Security Operations
```
security/
├── __pycache__/                   # Python cache
├── anomaly_detector.py            # Behavioral anomaly detection
├── defense_automation.py          # Automated defense systems
├── guardian-security-enforcer.py  # Legacy security enforcer
├── guardian_security_enforcer.py  # Security policy enforcement
├── intel_fusion.py                # Intelligence data fusion
├── intrusion_detection.py         # Network intrusion detection
├── opsec_enforcer.py              # OPSEC compliance
├── security_framework.py          # Security architecture
├── tactical_intelligence.py       # Tactical threat intelligence
└── traffic_analyzer.py            # Network traffic analysis
```
**Purpose**: Comprehensive security operations and threat intelligence
**Key Features**: OPSEC enforcement, tactical intelligence, automated defense
**Dependencies**: `infrastructure/proxy_manager.py`, `monitoring/`

#### `/src/infrastructure/` - System Infrastructure
```
infrastructure/
├── __init__.py                    # Infrastructure initialization
├── auto_recovery.py               # Automated system recovery
├── cache_optimizer.py             # Cache performance optimization
├── cache_warmer.py                # Predictive cache warming
├── circuit_breaker.py             # Circuit breaker pattern
├── database_integration.py        # Multi-database integration
├── embedding_manager.py           # Vector embedding management
├── geo_router.py                  # Geographic routing
├── logging_alerting.py            # Logging and alerting
├── ml_predictor.py                # Machine learning predictions
├── performance_benchmarks.py      # Performance testing
├── predictive_cache.py            # Predictive caching system
├── predictive_cache_service.py    # Cache service API
├── proxy_manager.py               # Proxy management
├── recovery_validator.py          # Recovery validation
├── tor_integration.py             # Tor network integration
└── vector_db_manager.py           # Vector database management
```
**Purpose**: Core infrastructure services for reliability and performance
**Key Features**: Auto-recovery, predictive caching, proxy management
**Dependencies**: External databases, Tor network, ML services

#### `/src/pipeline/` - Data Processing
```
pipeline/
├── README.md                      # Pipeline documentation
├── README_compression.md          # Compression documentation
├── airflow-research-pipeline.py   # Airflow integration
├── benchmark.py                   # Performance benchmarks
├── compression_api.py             # Compression API
├── compression_benchmarks.py      # Compression testing
├── connection_pool.py             # Connection management
├── context_compressor.py          # Context compression
├── document_analyzer.py           # Document analysis
├── enhanced_ocr_pipeline.py       # OCR processing
├── entropy_compressor.py          # Entropy-based compression
├── genetic_prompt_optimizer.py    # Prompt optimization
├── knowledge_synthesis_engine.py  # Knowledge synthesis
├── message-queue-infrastructure.py # Message queue setup
├── ocr-pipeline.py                # OCR pipeline
├── ocr_processor.py               # OCR processing core
├── quality_validator.py           # Quality validation
├── queue_manager.py               # Queue management
├── rate_limiter.py                # Rate limiting
├── request_multiplexer.py         # Request multiplexing
├── request_multiplexer_service.py # Multiplexer service
├── request_multiplexing.py        # Multiplexing logic
├── semantic_deduplicator.py       # Semantic deduplication
├── test_multiplexer.py            # Multiplexer testing
└── toolmaster_orchestrator.py     # Tool orchestration
```
**Purpose**: Data processing and transformation pipelines
**Key Features**: OCR, compression, semantic processing, multiplexing
**Dependencies**: `infrastructure/`, message queues, external ML services

#### `/src/autonomous/` - Autonomous Systems
```
autonomous/
├── README.md                      # Autonomous systems documentation
├── adaptive_learning.py           # Adaptive learning algorithms
├── autonomous_controller.py       # Main autonomous controller
├── autonomous_integration.py      # System integration
├── enhanced_autonomous_controller.py # Enhanced controller
├── intelligence_coordinator.py    # Intelligence coordination
├── knowledge_evolution.py         # Knowledge evolution
└── resource_optimizer.py          # Resource optimization
```
**Purpose**: Self-adapting and evolving system capabilities
**Key Features**: Adaptive learning, knowledge evolution, resource optimization
**Dependencies**: `agents/`, `security/`, `infrastructure/`

### Specialized Modules

#### `/src/alternative_market/` - Market Intelligence
```
alternative_market/
├── __init__.py                    # Market module initialization
├── crypto_analyzer.py             # Cryptocurrency analysis
├── dm_crawler.py                  # Darknet market crawler
├── economics_processor.py         # Economic data processing
└── reputation_analyzer.py         # Reputation analysis
```
**Purpose**: Alternative market and cryptocurrency intelligence
**Cross-Reference**: [Phase 7 Documentation](../PHASE7_ALTERNATIVE_MARKET_INTELLIGENCE.md)

#### `/src/monitoring/` - System Monitoring
```
monitoring/
├── README.md                      # Monitoring documentation
├── __init__.py                    # Monitoring initialization
├── alert_system.py                # Alert management
├── config/                        # Monitoring configurations
├── health_monitor.py              # Health monitoring
├── metrics_collector.py           # Metrics collection
└── monitoring_api.py              # Monitoring API
```
**Purpose**: Comprehensive system monitoring and alerting
**Dependencies**: Prometheus, Grafana, InfluxDB

#### `/src/edge/` - Edge Computing
```
edge/
├── edge_compute_network.py        # Edge network management
├── edge_integration.py            # Edge system integration
├── edge_management_service.py     # Edge management API
├── edge_node_manager.py           # Edge node coordination
├── geo_router.py                  # Geographic routing
└── model_synchronizer.py          # Model synchronization
```
**Purpose**: Distributed edge computing capabilities
**Dependencies**: `infrastructure/geo_router.py`

#### `/src/testing/` - Testing Framework
```
testing/
├── README.md                      # Testing documentation
├── __main__.py                    # Testing entry point
├── chaos_api.py                   # Chaos engineering API
├── chaos_engineer.py              # Chaos engineering
├── fault_injector.py              # Fault injection
├── resilience_tester.py           # Resilience testing
└── scenario_library.py            # Test scenarios
```
**Purpose**: Comprehensive testing and chaos engineering
**Cross-Reference**: [Test Framework Documentation](../TEST_FRAMEWORK_README.md)

---

## ⚙️ Configuration Systems

### Core Configuration Files

#### System Configuration
| File | Purpose | Format | Dependencies |
|------|---------|--------|--------------|
| `.env` | Environment variables | ENV | All services |
| `config/prometheus.yml` | Metrics configuration | YAML | Monitoring stack |
| `config/litellm_config.yaml` | LLM configuration | YAML | AI services |
| `config/auto_recovery.yaml` | Recovery configuration | YAML | Infrastructure |

#### Monitoring & Observability
```
config/
├── grafana-datasources.yml       # Grafana data sources
├── grafana-dashboards.json       # Dashboard definitions
├── prometheus-alerts.yml         # Alert rules
├── metrics/
│   ├── phase7-metrics.yml        # Phase 7 specific metrics
│   ├── phase8-metrics.yml        # Phase 8 security metrics
│   └── phase9-metrics.yml        # Phase 9 autonomous metrics
└── grafana/dashboards/
    ├── bev-unified-overview.json # Unified dashboard
    ├── phase7-market-intelligence.json
    ├── phase8-security-operations.json
    └── phase9-autonomous-systems.json
```

#### Security & Compliance
```
config/
├── security/
│   └── phases-auth-config.yml    # Authentication configuration
└── logging/
    └── filebeat-phases.yml       # Log aggregation
```

#### Service Discovery
```
config/
└── consul/
    └── phases-service-discovery.json # Service registry
```

---

## 🐳 Deployment Infrastructure

### Docker Architecture

#### Main Compositions
| File | Purpose | Services | Scale |
|------|---------|----------|-------|
| `docker-compose.complete.yml` | Full system deployment | All services | Production |
| `docker-compose-infrastructure.yml` | Infrastructure only | Core systems | Development |
| `docker-compose-monitoring.yml` | Monitoring stack | Observability | Operations |

#### Specialized Deployments
```
docker/
├── thanos/docker-compose.yml      # Primary intelligence hub
├── oracle/docker-compose.yml      # Distributed cloud node
├── databases/docker-compose.yml   # Database cluster
├── minio-cluster/docker-compose.yml # Object storage
└── message-queue/docker-compose-messaging.yml # Message infrastructure
```

#### Phase-Specific Deployments
```
./docker-compose-phase7.yml       # Alternative market intelligence
./docker-compose-phase8.yml       # Advanced security operations
./docker-compose-phase9.yml       # Autonomous research enhancement
```

### Service Architecture Map

```
┌─────────────────────────────────────────────────────────┐
│                    THANOS HUB                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   IntelOwl  │  │  Cytoscape  │  │    Agents   │     │
│  │             │  │             │  │             │     │
│  │ • Analysis  │  │ • Visualize │  │ • Research  │     │
│  │ • Workflow  │  │ • Graph     │  │ • Memory    │     │
│  │ • Storage   │  │ • Network   │  │ • Swarm     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   LOAD BALANCER   │
                    │   Nginx/HAProxy   │
                    └─────────┬─────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                    ORACLE1 NODE                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│  │  Databases  │  │ Message Q   │  │  Security   │       │
│  │             │  │             │  │             │       │
│  │ • Neo4j     │  │ • RabbitMQ  │  │ • Tor       │       │
│  │ • Postgres  │  │ • Kafka     │  │ • Firewall  │       │
│  │ • Redis     │  │ • Redis     │  │ • IDS       │       │
│  │ • Elastic   │  │ • MinIO     │  │ • VPN       │       │
│  └─────────────┘  └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────┘
```

---

## 🧪 Testing Framework

### Test Structure
```
tests/
├── conftest.py                    # Pytest configuration
├── integration_tests.py          # Integration test suite
├── integration_tests.sh          # Integration test runner
├── test_config.yaml              # Test configuration
├── cache/                         # Cache testing
├── chaos/                         # Chaos engineering tests
├── edge/                          # Edge computing tests
├── end_to_end/                    # E2E test scenarios
├── integration/                   # Integration tests
└── monitoring/                    # Monitoring tests
```

### Testing Categories

#### Unit Tests
- **Location**: Co-located with source files
- **Pattern**: `test_*.py` files in module directories
- **Coverage**: Individual component functionality

#### Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Service-to-service communication
- **Dependencies**: Docker infrastructure

#### End-to-End Tests
- **Location**: `tests/end_to_end/`
- **Purpose**: Complete workflow validation
- **Tools**: Selenium, API testing frameworks

#### Chaos Engineering
- **Location**: `tests/chaos/`, `src/testing/`
- **Purpose**: Resilience and fault tolerance
- **Tools**: Custom chaos engineering framework

### Test Execution Scripts
```
./run_all_tests.sh                # Complete test suite
./run_comprehensive_tests.sh      # Extended testing
./run_security_tests.py           # Security-focused tests
./setup_test_environment.sh       # Test environment setup
```

---

## 📚 Documentation Map

### User Documentation
| Document | Purpose | Audience | Status |
|----------|---------|----------|--------|
| [README.md](../README.md) | Project overview | All users | ✅ Current |
| [BEV_QUICK_START.md](../BEV_QUICK_START.md) | Quick deployment | Operators | ✅ Current |
| [BEV_USER_GUIDES.md](../BEV_USER_GUIDES.md) | User workflows | End users | ✅ Current |

### Technical Documentation
| Document | Purpose | Audience | Status |
|----------|---------|----------|--------|
| [BEV_ARCHITECTURE_OVERVIEW.md](../BEV_ARCHITECTURE_OVERVIEW.md) | System architecture | Developers | ✅ Current |
| [BEV_DEVELOPMENT_GUIDE.md](../BEV_DEVELOPMENT_GUIDE.md) | Development workflows | Developers | ✅ Current |
| [BEV_ADVANCED_API_REFERENCE.md](../BEV_ADVANCED_API_REFERENCE.md) | API documentation | Developers | ✅ Current |

### Operational Documentation
| Document | Purpose | Audience | Status |
|----------|---------|----------|--------|
| [BEV_DEPLOYMENT_GUIDE.md](../BEV_DEPLOYMENT_GUIDE.md) | Deployment procedures | DevOps | ✅ Current |
| [BEV_OPERATIONS_MAINTENANCE_MANUAL.md](../BEV_OPERATIONS_MAINTENANCE_MANUAL.md) | Operations manual | SysAdmins | ✅ Current |
| [BEV_PERFORMANCE_SCALABILITY_GUIDE.md](../BEV_PERFORMANCE_SCALABILITY_GUIDE.md) | Performance tuning | SRE | ✅ Current |

### Compliance Documentation
| Document | Purpose | Audience | Status |
|----------|---------|----------|--------|
| [BEV_COMPLIANCE_LEGAL_FRAMEWORK.md](../BEV_COMPLIANCE_LEGAL_FRAMEWORK.md) | Legal compliance | Legal/Compliance | ✅ Current |
| [SECURITY_OPERATIONS_CENTER_SUMMARY.md](../SECURITY_OPERATIONS_CENTER_SUMMARY.md) | Security procedures | Security team | ✅ Current |

### Phase-Specific Documentation
| Document | Purpose | Phase | Status |
|----------|---------|-------|--------|
| [PHASE7_ALTERNATIVE_MARKET_INTELLIGENCE.md](../PHASE7_ALTERNATIVE_MARKET_INTELLIGENCE.md) | Market intelligence | Phase 7 | ✅ Current |
| [PHASE8_ADVANCED_SECURITY_OPERATIONS.md](../PHASE8_ADVANCED_SECURITY_OPERATIONS.md) | Security operations | Phase 8 | ✅ Current |
| [PHASE9_AUTONOMOUS_RESEARCH_ENHANCEMENT.md](../PHASE9_AUTONOMOUS_RESEARCH_ENHANCEMENT.md) | Autonomous systems | Phase 9 | ✅ Current |

---

## 🔌 API Reference

### Core Service APIs

#### Agent Coordination APIs
- **Extended Reasoning Service**: `src/agents/extended_reasoning_service.py`
  - Endpoint: `/api/v1/reasoning/`
  - Methods: POST (analysis), GET (status)
  - Purpose: Complex reasoning and analysis

- **Memory Management**: `src/agents/memory_manager.py`
  - Endpoint: `/api/v1/memory/`
  - Methods: GET, POST, PUT, DELETE
  - Purpose: Session persistence and retrieval

#### Infrastructure APIs
- **Predictive Cache Service**: `src/infrastructure/predictive_cache_service.py`
  - Endpoint: `/api/v1/cache/`
  - Methods: GET, POST, DELETE
  - Purpose: Intelligent caching operations

- **Auto Recovery**: `src/infrastructure/auto_recovery.py`
  - Endpoint: `/api/v1/recovery/`
  - Methods: GET (status), POST (trigger)
  - Purpose: System recovery operations

#### Pipeline APIs
- **Compression API**: `src/pipeline/compression_api.py`
  - Endpoint: `/api/v1/compress/`
  - Methods: POST
  - Purpose: Data compression services

- **Request Multiplexer**: `src/pipeline/request_multiplexer_service.py`
  - Endpoint: `/api/v1/multiplex/`
  - Methods: POST
  - Purpose: Request optimization

#### Monitoring APIs
- **Monitoring API**: `src/monitoring/monitoring_api.py`
  - Endpoint: `/api/v1/monitoring/`
  - Methods: GET
  - Purpose: System health and metrics

- **Chaos Engineering**: `src/testing/chaos_api.py`
  - Endpoint: `/api/v1/chaos/`
  - Methods: POST (inject), GET (status)
  - Purpose: Resilience testing

### External Service Integrations

#### Database Connections
```yaml
PostgreSQL: postgresql://bev:BevOSINT2024@localhost:5432/osint
Neo4j: bolt://localhost:7687 (neo4j/BevGraphMaster2024)
Redis: redis://:BevCacheMaster@localhost:6379
Elasticsearch: http://localhost:9200
InfluxDB: http://localhost:8086
```

#### Message Queue Endpoints
```yaml
RabbitMQ: http://localhost:15672 (management)
Kafka: localhost:9092 (broker)
```

#### Monitoring Endpoints
```yaml
Prometheus: http://localhost:9090
Grafana: http://localhost:3000
```

---

## 🔧 Development Workflows

### Development Environment Setup
1. **Prerequisites**: Docker, Docker Compose, Python 3.9+
2. **Configuration**: Copy `.env.example` to `.env`
3. **Infrastructure**: `docker-compose -f docker-compose-infrastructure.yml up -d`
4. **Development**: Install requirements in virtual environment

### Code Organization Standards
- **Module Structure**: Each module contains `__init__.py`, README.md
- **Testing**: Tests co-located or in `tests/` directory
- **Documentation**: Inline docstrings + module README
- **Configuration**: YAML/JSON configs in `config/` directory

### Git Workflow (Note: Not currently a git repository)
- **Branches**: feature/, bugfix/, hotfix/ prefixes
- **Commits**: Conventional commit format
- **Testing**: All tests must pass before merge
- **Documentation**: Update docs with code changes

### Quality Standards
- **Code Quality**: PEP 8, type hints, docstrings
- **Testing**: >80% coverage, integration tests
- **Security**: SAST scanning, dependency checks
- **Performance**: Benchmarking, profiling

---

## 🚀 Operational Procedures

### Deployment Procedures
1. **Development**: `docker-compose-infrastructure.yml`
2. **Staging**: `docker-compose.complete.yml` (limited services)
3. **Production**: Full deployment with monitoring

### Monitoring & Alerting
- **Health Checks**: Service health endpoints
- **Metrics**: Prometheus metrics collection
- **Dashboards**: Grafana visualization
- **Alerts**: Automated alerting via configured channels

### Backup & Recovery
- **Database Backups**: Automated daily backups
- **Configuration**: Version-controlled configurations
- **Recovery**: Automated recovery procedures
- **Testing**: Regular recovery testing

### Security Operations
- **OPSEC**: Operational security enforcement
- **Monitoring**: Security event monitoring
- **Incident Response**: Automated incident handling
- **Compliance**: Regular compliance validation

---

## 🔍 Quick Navigation

### By Functionality
- **Intelligence Operations**: `src/agents/`, `src/autonomous/`
- **Security Operations**: `src/security/`, `config/security/`
- **Data Processing**: `src/pipeline/`, `src/infrastructure/`
- **Monitoring**: `src/monitoring/`, `config/grafana/`
- **Testing**: `src/testing/`, `tests/`

### By Deployment Phase
- **Phase 7**: Alternative market intelligence
- **Phase 8**: Advanced security operations
- **Phase 9**: Autonomous research enhancement

### By Documentation Type
- **User Docs**: README, Quick Start, User Guides
- **Technical Docs**: Architecture, Development, API Reference
- **Operational Docs**: Deployment, Operations, Performance

### By Audience
- **End Users**: User guides, quick start
- **Developers**: Development guide, API reference
- **Operators**: Deployment guide, operations manual
- **Security**: Security operations, compliance framework

---

**Last Updated**: Generated by /sc:index command
**Maintenance**: Auto-updated on significant structural changes
**Contact**: Refer to project maintainers for updates