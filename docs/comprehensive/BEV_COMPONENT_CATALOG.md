# BEV Component Catalog & Interaction Guide

## Overview

This catalog provides detailed documentation of all BEV OSINT Framework components, their purposes, dependencies, and interaction patterns. Use this guide to understand how components work together and plan integrations or customizations.

## Table of Contents

1. [Component Architecture](#component-architecture)
2. [Foundation Layer](#foundation-layer)
3. [Intelligence Processing Layer](#intelligence-processing-layer)
4. [Analysis & Visualization Layer](#analysis--visualization-layer)
5. [Security & Proxy Layer](#security--proxy-layer)
6. [Monitoring & Operations Layer](#monitoring--operations-layer)
7. [Component Interaction Flows](#component-interaction-flows)
8. [Integration Points](#integration-points)
9. [Dependency Matrix](#dependency-matrix)

---

## Component Architecture

### System Layer Model

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                    PRESENTATION LAYER                           │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │  IntelOwl   │  │  Cytoscape  │  │   Grafana   │             │
    │  │   WebUI     │  │ Visualizer  │  │ Dashboards  │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────────────┐
    │                 INTELLIGENCE PROCESSING LAYER                   │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │ Custom OSINT│  │   Airflow   │  │    n8n      │             │
    │  │  Analyzers  │  │ Workflows   │  │ Automation  │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────────────┐
    │                   ANALYSIS & STORAGE LAYER                      │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │ PostgreSQL  │  │    Neo4j    │  │ Elasticsearch│             │
    │  │  +pgvector  │  │ Graph DB    │  │ Search Engine│             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────────────┐
    │                   SECURITY & PROXY LAYER                        │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │    Tor      │  │    HAProxy  │  │   Traefik   │             │
    │  │  Anonymity  │  │Load Balancer│  │    Router   │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────┬───────────────────────────────────────┘
                              │
    ┌─────────────────────────▼───────────────────────────────────────┐
    │                    FOUNDATION LAYER                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │    Redis    │  │  RabbitMQ   │  │   Docker    │             │
    │  │   Caching   │  │ Message Bus │  │ Containers  │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Foundation Layer

### Redis Cache Cluster
**Purpose**: High-speed caching and session management
**Container**: `bev_redis`
**Network**: `172.30.0.4`

**Key Functions**:
- Session storage for web interfaces
- Analysis result caching
- Temporary data storage for processing pipelines
- Rate limiting for OSINT analyzers

**Interaction Patterns**:
```
IntelOwl WebUI → Redis → Session Data
Custom Analyzers → Redis → Cached Results
Airflow Tasks → Redis → Temporary Storage
```

**Dependencies**:
- **Upstream**: All application services
- **Downstream**: None (foundation service)

**Configuration**:
```yaml
Memory: 2GB allocated
Persistence: RDB snapshots + AOF
Clustering: Single node (can be clustered)
Key Management: TTL-based expiration
```

### RabbitMQ Message Broker
**Purpose**: Asynchronous message queuing and task distribution
**Container**: `bev_rabbitmq`
**Network**: `172.30.0.5`

**Key Functions**:
- Task queue management for OSINT analyzers
- Inter-service communication
- Event-driven architecture support
- Workflow orchestration messaging

**Interaction Patterns**:
```
Airflow → RabbitMQ → Custom Analyzers
IntelOwl → RabbitMQ → Analysis Tasks
n8n → RabbitMQ → Automation Triggers
```

**Queue Structure**:
- `osint_analysis_queue`: Primary analysis tasks
- `breach_search_queue`: Breach database queries
- `crypto_tracking_queue`: Cryptocurrency analysis
- `darknet_monitor_queue`: Darknet monitoring tasks

### PostgreSQL with pgvector
**Purpose**: Primary relational database with vector search capabilities
**Container**: `bev_postgres`
**Network**: `172.30.0.2`

**Key Functions**:
- OSINT analysis results storage
- User session management
- Vector embeddings for similarity search
- Audit logging and compliance tracking

**Database Schema**:
```
osint: Main analysis results
intelowl: IntelOwl metadata
breach_data: Breach database correlations
crypto_analysis: Cryptocurrency tracking
vector_embeddings: Semantic search vectors
```

---

## Intelligence Processing Layer

### Custom OSINT Analyzers
**Purpose**: Specialized intelligence gathering and analysis
**Containers**: Multiple specialized analyzers

#### Breach Database Analyzer
**Container**: `breach_db_analyzer`
**Purpose**: Search and correlate data breach information

**Capabilities**:
- Email/username breach correlation
- Password hash analysis
- Breach timeline construction
- Cross-breach pattern analysis

**Data Flow**:
```
Input: Email/Username/Domain
Process: Multi-source breach search
Output: Breach timeline + affected accounts
Storage: PostgreSQL breach_data table
```

#### Darknet Market Analyzer
**Container**: `darknet_analyzer`
**Purpose**: Monitor darknet markets and forums

**Capabilities**:
- Market monitoring and scraping
- Forum sentiment analysis
- Vendor reputation tracking
- Product category analysis

**Security Considerations**:
- Mandatory Tor proxy usage
- User agent rotation
- Request rate limiting
- No persistent storage of illegal content

#### Cryptocurrency Tracker
**Container**: `crypto_tracker`
**Purpose**: Blockchain analysis and transaction tracking

**Capabilities**:
- Address clustering analysis
- Transaction flow mapping
- Exchange interaction detection
- Risk score calculation

**Supported Networks**:
- Bitcoin (BTC)
- Ethereum (ETH)
- Monero (XMR) - limited analysis
- Major altcoins

#### Social Media Analyzer
**Container**: `social_analyzer`
**Purpose**: Social media intelligence gathering

**Capabilities**:
- Profile analysis and correlation
- Network mapping
- Content sentiment analysis
- Timeline reconstruction

**Platform Support**:
- Twitter/X (via API)
- LinkedIn (limited)
- Reddit (via API)
- Telegram (channel monitoring)

### Apache Airflow Orchestration
**Purpose**: Workflow orchestration and automation
**Container**: `bev_airflow`
**Network**: `172.30.0.20`

**Key DAGs**:
```
daily_breach_monitoring: Automated breach database updates
crypto_analysis_pipeline: Scheduled cryptocurrency analysis
darknet_monitoring: Continuous darknet market monitoring
data_retention_cleanup: Automated data lifecycle management
```

**Interaction Pattern**:
```
External Trigger → Airflow DAG → Task Queue → Analyzers → Results Storage
```

### n8n Automation Platform
**Purpose**: No-code workflow automation
**Container**: `bev_n8n`
**Network**: `172.30.0.22`

**Automation Workflows**:
- Alert generation for critical findings
- Report generation and distribution
- Data synchronization between systems
- External API integrations

---

## Analysis & Visualization Layer

### Neo4j Graph Database
**Purpose**: Relationship analysis and graph visualization
**Container**: `bev_neo4j`
**Network**: `172.30.0.3`

**Graph Models**:
```
(Person)-[USES]->(Email)
(Email)-[BREACHED_IN]->(Breach)
(Address)-[TRANSACTED_WITH]->(Address)
(Domain)-[HOSTS]->(Service)
(Entity)-[CONNECTED_TO]->(Entity)
```

**Key Algorithms**:
- Shortest path analysis
- Community detection
- Centrality calculations
- Graph similarity matching

### Elasticsearch
**Purpose**: Full-text search and analytics
**Container**: `bev_elasticsearch`
**Network**: `172.30.0.15`

**Index Structure**:
```
osint_results: Analyzed intelligence data
breach_records: Structured breach data
crypto_transactions: Blockchain transaction data
darknet_listings: Market monitoring results
social_profiles: Social media intelligence
```

**Search Capabilities**:
- Full-text search across all collected data
- Aggregation analytics
- Time-series analysis
- Geographic analysis

### Cytoscape.js Visualization
**Purpose**: Interactive graph visualization
**Integration**: Frontend JavaScript library

**Visualization Types**:
- Network topology graphs
- Timeline visualizations
- Hierarchical relationship trees
- Interactive force-directed layouts

**Data Sources**:
- Neo4j graph queries
- Elasticsearch aggregations
- PostgreSQL relationship queries

---

## Security & Proxy Layer

### Tor Proxy Network
**Purpose**: Anonymous internet access for OSINT collection
**Container**: `bev_tor`
**Network**: `172.30.0.50`

**Configuration**:
- Multiple Tor circuits
- Automatic circuit rotation
- SOCKS5 proxy on port 9050
- Control port on 9051

**Circuit Management**:
```python
# Example circuit rotation
tor_controller.new_circuit()
session.proxies = {'http': 'socks5://172.30.0.50:9050'}
```

### HAProxy Load Balancer
**Purpose**: Load balancing and traffic management
**Container**: `bev_haproxy`
**Network**: `172.30.0.100`

**Load Balancing Rules**:
```
Frontend:
  Web Traffic: Port 80/443 → IntelOwl instances
  API Traffic: Port 8000 → API backend pool
  
Backend Pools:
  intelowl_web: Multiple IntelOwl instances
  api_backend: FastAPI instances
  monitoring: Grafana + Prometheus
```

### Traefik Edge Router
**Purpose**: Dynamic service discovery and routing
**Container**: `bev_traefik`
**Network**: Edge network

**Routing Rules**:
- Automatic service discovery via Docker labels
- TLS termination and certificate management
- Request routing based on hostnames and paths
- Middleware integration (auth, rate limiting)

---

## Monitoring & Operations Layer

### Prometheus Metrics Collection
**Purpose**: System and application metrics collection
**Container**: `bev_prometheus`
**Network**: `172.30.0.30`

**Metric Categories**:
```
System Metrics: CPU, memory, disk, network
Application Metrics: Request rates, error rates, latency
Business Metrics: Analysis completion rates, data quality
Security Metrics: Failed login attempts, anomaly detection
```

**Scrape Targets**:
- All application containers via `/metrics` endpoints
- Node Exporter for system metrics
- Custom exporters for specialized metrics

### Grafana Dashboards
**Purpose**: Visualization and alerting
**Container**: `bev_grafana`
**Network**: `172.30.0.31`

**Dashboard Categories**:
- **System Overview**: Resource utilization and health
- **OSINT Analytics**: Analysis performance and results
- **Security Monitoring**: Security events and anomalies
- **Business Intelligence**: Usage patterns and trends

### Loki Log Aggregation
**Purpose**: Centralized log management
**Container**: `bev_loki`
**Network**: `172.30.0.32`

**Log Sources**:
- Application logs from all containers
- System logs via Promtail
- Access logs from proxies and load balancers
- Audit logs from security components

---

## Component Interaction Flows

### OSINT Analysis Workflow
```mermaid
graph TD
    A[User Request] --> B[IntelOwl WebUI]
    B --> C[Analysis Queue]
    C --> D[Custom Analyzer]
    D --> E[Tor Proxy]
    E --> F[External Source]
    F --> E
    E --> D
    D --> G[PostgreSQL]
    D --> H[Neo4j]
    D --> I[Elasticsearch]
    G --> J[Results Display]
    H --> K[Graph Visualization]
    I --> L[Search Interface]
```

### Data Pipeline Flow
```mermaid
graph TD
    A[Airflow DAG] --> B[Task Definition]
    B --> C[RabbitMQ Queue]
    C --> D[Analyzer Pool]
    D --> E[Data Processing]
    E --> F[Validation]
    F --> G[Storage Layer]
    G --> H[Index Update]
    H --> I[Cache Refresh]
    I --> J[Notification]
```

### Security Flow
```mermaid
graph TD
    A[External Request] --> B[Traefik Router]
    B --> C[Security Middleware]
    C --> D[HAProxy]
    D --> E[Application Pool]
    E --> F[Tor Proxy Check]
    F --> G[External API Call]
    G --> H[Response Processing]
    H --> I[Audit Logging]
```

---

## Integration Points

### API Integration Points
```yaml
Primary APIs:
  IntelOwl API: /api/v1/ (REST)
  Custom Analyzers: /analyzer/v1/ (REST)
  Neo4j Cypher: bolt://172.30.0.3:7687
  Elasticsearch: http://172.30.0.15:9200
  
Webhook Endpoints:
  n8n Webhooks: /webhook/
  Airflow Triggers: /api/v1/dags/{dag_id}/dagRuns
  
Message Queues:
  RabbitMQ AMQP: amqp://172.30.0.5:5672
  Redis Pub/Sub: redis://172.30.0.4:6379
```

### Database Integration
```yaml
PostgreSQL Connections:
  Primary: postgresql://bev:password@172.30.0.2:5432/osint
  IntelOwl: postgresql://bev:password@172.30.0.2:5432/intelowl
  
Neo4j Connections:
  Bolt Protocol: bolt://neo4j:password@172.30.0.3:7687
  HTTP API: http://172.30.0.3:7474
  
Redis Connections:
  Cache: redis://172.30.0.4:6379/0
  Sessions: redis://172.30.0.4:6379/1
```

### External Service Integration
```yaml
Tor Services:
  SOCKS5 Proxy: socks5://172.30.0.50:9050
  Control Port: 172.30.0.50:9051
  
Monitoring Integration:
  Prometheus: http://172.30.0.30:9090
  Grafana: http://172.30.0.31:3000
  Loki: http://172.30.0.32:3100
```

---

## Dependency Matrix

### Service Dependencies
| Service | Hard Dependencies | Soft Dependencies | Optional |
|---------|------------------|-------------------|----------|
| PostgreSQL | None | None | Backup storage |
| Redis | None | None | Persistence |
| RabbitMQ | None | None | Management UI |
| Neo4j | None | None | APOC plugins |
| Elasticsearch | None | None | Kibana |
| IntelOwl | PostgreSQL, Redis | RabbitMQ | Analyzers |
| Custom Analyzers | RabbitMQ, PostgreSQL | Tor, Redis | Elasticsearch |
| Airflow | PostgreSQL, Redis | RabbitMQ | Custom operators |
| Grafana | Prometheus | Loki | Data sources |
| Tor | None | None | Control scripts |

### Network Dependencies
```yaml
Internal Networks:
  bev_network: 172.30.0.0/24 (Primary)
  monitoring: 172.31.0.0/24 (Monitoring)
  proxy: 172.32.0.0/24 (Proxy services)

External Dependencies:
  Internet: Required for OSINT collection
  DNS: Required for domain resolution
  NTP: Required for time synchronization
```

### Storage Dependencies
```yaml
Persistent Volumes:
  postgres_data: PostgreSQL database files
  neo4j_data: Graph database files
  elasticsearch_data: Search index files
  redis_data: Cache persistence (optional)
  
Log Volumes:
  application_logs: Container application logs
  access_logs: Proxy and web server logs
  audit_logs: Security and compliance logs
```

---

## Performance Characteristics

### Resource Requirements
| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| PostgreSQL | 2 cores | 4GB | 100GB+ | Medium |
| Neo4j | 2 cores | 4GB | 50GB+ | Low |
| Elasticsearch | 2 cores | 4GB | 100GB+ | Medium |
| IntelOwl | 1 core | 2GB | 10GB | High |
| Custom Analyzers | 4 cores | 8GB | 20GB | Very High |
| Monitoring Stack | 2 cores | 4GB | 50GB | Medium |

### Scaling Patterns
- **Horizontal**: Custom analyzers, IntelOwl instances
- **Vertical**: Databases, Elasticsearch
- **Hybrid**: Monitoring services, proxy layer

### Performance Bottlenecks
1. **Network I/O**: OSINT collection limited by external sources
2. **Database Writes**: Bulk analysis results storage
3. **Graph Queries**: Complex Neo4j relationship analysis
4. **Tor Circuits**: Proxy rotation and circuit establishment

---

*Last Updated: 2025-09-19*
*Framework Version: BEV OSINT v2.0*
*Component Catalog Version: 1.0*