# BEV OSINT Database Extensions

Comprehensive database schema extensions for the 3 new advanced phases of the BEV OSINT system, designed for massive data volumes and high-performance requirements.

## 📋 Overview

This repository contains production-ready database schemas optimized for:

- **Phase 7**: Alternative Market Intelligence (Marketplace & Cryptocurrency)
- **Phase 8**: Advanced Security Operations (Threat Intelligence & Security Operations)
- **Phase 9**: Autonomous Enhancement (Machine Learning & Autonomous Operations)

## 🏗️ Architecture

### Database Infrastructure
- **PostgreSQL** with pgvector extension for embeddings
- **Neo4j** for relationship mapping
- **Redis** cluster for caching
- **TimescaleDB** patterns for time-series optimization
- **Partitioning** for massive data volumes
- **Vector search** capabilities for similarity analysis

### Key Features
- ✅ **Partitioned tables** for optimal performance at scale
- ✅ **Vector embeddings** with optimized similarity search
- ✅ **Full-text search** with PostgreSQL's native capabilities
- ✅ **Temporal tables** with audit trails
- ✅ **Automated retention** policies and archival
- ✅ **Performance monitoring** and index optimization
- ✅ **Safe migrations** with rollback capabilities

## 📁 Structure

```
docker/databases/
├── init-scripts/postgres/          # Database initialization scripts
│   ├── 01_base_extensions.sql      # Core extensions and configurations
│   ├── 02_phase7_marketplace_intelligence.sql
│   ├── 03_phase7_crypto_intelligence.sql
│   ├── 04_phase8_threat_intelligence.sql
│   ├── 05_phase8_security_operations.sql
│   └── 06_phase9_autonomous_operations.sql
├── migrations/                     # Safe deployment migrations
│   ├── 001_phase7_marketplace_migration.sql
│   ├── 002_phase7_crypto_migration.sql
│   └── 003_phase8_threat_intel_migration.sql
├── performance/                    # Performance optimization
│   ├── index_optimization_strategies.sql
│   └── data_retention_policies.sql
└── docker-compose.yml              # Complete database infrastructure
```

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have the following installed:
- Docker and Docker Compose
- At least 16GB RAM (32GB recommended)
- 100GB+ available storage

### 2. Environment Setup

Create environment file:
```bash
cp .env.example .env
# Edit .env with your passwords and configuration
```

### 3. Deploy Database Infrastructure

```bash
# Navigate to the databases directory
cd docker/databases

# Start the complete database stack
docker-compose up -d

# Verify all services are running
docker-compose ps
```

### 4. Initialize Schemas

The initialization scripts will run automatically when PostgreSQL starts for the first time. Monitor the logs:

```bash
# Monitor PostgreSQL initialization
docker-compose logs -f postgres
```

### 5. Verify Deployment

Connect to PostgreSQL and verify schemas:
```sql
-- Connect to database
psql -h localhost -p 5432 -U swarm_admin -d ai_swarm

-- List all schemas
\dn+

-- Verify Phase 7 tables
\dt marketplace_intel.*
\dt crypto_intel.*

-- Verify Phase 8 tables
\dt threat_intel.*
\dt security_ops.*

-- Verify Phase 9 tables
\dt autonomous.*
```

## 📊 Database Schemas

### Phase 7: Alternative Market Intelligence

#### Marketplace Intelligence (`marketplace_intel`)
- **vendor_profiles**: Comprehensive vendor information with risk assessment
- **product_listings**: Product catalog with classification and pricing
- **transaction_records**: Transaction history with fraud detection
- **reputation_scores**: Dynamic reputation tracking
- **price_histories**: Market price analysis and trends

#### Cryptocurrency Intelligence (`crypto_intel`)
- **wallet_transactions**: Blockchain transaction tracking with risk scoring
- **blockchain_flows**: Multi-hop transaction flow analysis
- **mixing_patterns**: Cryptocurrency mixing service detection
- **exchange_movements**: Exchange deposit/withdrawal tracking
- **wallet_clusters**: Address clustering and entity identification

### Phase 8: Advanced Security Operations

#### Threat Intelligence (`threat_intel`)
- **threat_indicators**: IOCs/IOAs with confidence scoring
- **threat_actors**: APT and threat actor profiling
- **attack_campaigns**: Campaign tracking and attribution
- **security_events**: Security event correlation and analysis
- **behavioral_profiles**: Behavioral pattern recognition

#### Security Operations (`security_ops`)
- **defense_actions**: Automated and manual defensive actions
- **threat_hunts**: Proactive threat hunting operations
- **incident_responses**: Comprehensive incident management
- **security_metrics**: KPI and performance tracking

### Phase 9: Autonomous Enhancement

#### Autonomous Operations (`autonomous`)
- **capability_registry**: AI capability catalog and versioning
- **learning_experiments**: ML experiment tracking and results
- **performance_metrics**: System performance and effectiveness
- **resource_allocations**: Dynamic resource management
- **knowledge_evolution**: Continuous learning and adaptation
- **autonomous_agents**: Agent lifecycle and coordination

## ⚡ Performance Optimizations

### Indexing Strategy

Our indexing strategy includes:

1. **Composite Indexes**: Multi-column indexes for complex queries
2. **Partial Indexes**: Filtered indexes for active/high-priority data
3. **Vector Indexes**: IVFFlat indexes for similarity search
4. **BRIN Indexes**: Block Range Indexes for time-series data
5. **GiST/GIN Indexes**: Specialized indexes for JSONB and arrays

### Query Optimization

- **Partitioning**: Monthly partitions for time-series tables
- **Materialized Views**: Pre-computed aggregations
- **Connection Pooling**: Optimized connection management
- **Query Planning**: Statistics and cost-based optimization

### Monitoring

Built-in performance monitoring includes:
- Index usage analysis
- Query performance tracking
- Bloat detection and maintenance
- Resource utilization metrics

## 🔄 Data Lifecycle Management

### Retention Policies

Automated retention policies for each data type:

| Data Type | Retention Period | Archive | Compliance |
|-----------|------------------|---------|------------|
| Marketplace Transactions | 7 years | ✅ | AML/Financial |
| Crypto Transactions | 5 years | ✅ | Blockchain Forensics |
| Threat Intelligence | 2-10 years | ✅ | Security Operations |
| Security Events | 3 years | ✅ | Incident Response |
| Learning Experiments | 2 years | ✅ | R&D |

### Archival Strategy

- **Automated archival** to external storage
- **Compression** for space efficiency
- **Compliance** with regulatory requirements
- **Legal hold** exemptions for investigations

## 🛡️ Security Features

### Access Control
- **Role-based permissions** for different user types
- **Row-level security** for sensitive data
- **Audit trails** for all data modifications
- **Encryption at rest** and in transit

### Compliance
- **GDPR compliance** with data anonymization
- **Financial regulations** (AML, KYC)
- **Law enforcement** data retention
- **Security standards** compliance

## 📈 Scalability

### Horizontal Scaling
- **Read replicas** for query distribution
- **Sharding** strategies for massive datasets
- **Connection pooling** for high concurrency
- **Load balancing** across database nodes

### Vertical Scaling
- **Memory optimization** for large working sets
- **CPU optimization** for complex analytics
- **Storage optimization** with tiered storage
- **Network optimization** for high throughput

## 🔧 Maintenance

### Automated Maintenance
```sql
-- Execute daily maintenance
SELECT performance_monitoring.auto_maintain_indexes();
SELECT data_retention.execute_all_retention_policies(false);
```

### Manual Operations
```sql
-- Analyze index effectiveness
SELECT * FROM performance_monitoring.analyze_index_effectiveness();

-- Monitor index bloat
SELECT * FROM performance_monitoring.monitor_index_bloat();

-- Generate retention report
SELECT * FROM data_retention.generate_retention_report(30);
```

## 🚨 Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Increase `shared_buffers` and `effective_cache_size`
   - Monitor query complexity and optimize

2. **Slow Queries**
   - Check index usage with performance views
   - Analyze query plans with EXPLAIN ANALYZE

3. **Disk Space**
   - Execute retention policies
   - Monitor partition growth
   - Archive old data

### Performance Monitoring

```sql
-- Check index performance
SELECT * FROM performance_monitoring.index_performance_summary;

-- Check table performance
SELECT * FROM performance_monitoring.table_performance_summary;

-- Check retention status
SELECT * FROM data_retention.retention_policy_status;
```

## 📞 Support

For technical support:
- Check the troubleshooting section
- Review PostgreSQL logs: `docker-compose logs postgres`
- Monitor system resources: `docker stats`
- Contact the data engineering team

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**⚠️ Security Notice**: This database contains sensitive intelligence data. Ensure proper access controls and compliance with all applicable regulations before deployment.