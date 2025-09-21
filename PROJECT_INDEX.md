# BEV OSINT Framework - Project Index

**Last Updated**: September 20, 2025
**Analysis**: Comprehensive service catalog with 151 distributed services
**Status**: Production-ready multinode deployment with Vault integration

## Service Distribution Architecture

### THANOS Node (Primary - x86_64)
**Role**: Heavy computation, databases, core intelligence
**Hardware**: RTX 3080, 64GB RAM
**Service Count**: 89 services

#### Core Infrastructure (19 services)
- `bev_postgres` - Primary PostgreSQL with pgvector
- `bev_neo4j` - Graph database (enterprise edition)
- `bev_redis_[1-3]` + `bev_redis_standalone` - Redis cluster
- `bev_rabbitmq_[1-3]` - Message queue cluster
- `bev_kafka_[1-3]` + `bev_zookeeper` - Event streaming
- `bev_elasticsearch` - Search and analytics
- `bev_influxdb` - Time series metrics
- `bev_tor` - Tor proxy integration

#### IntelOwl Platform (4 services)
- `bev_intelowl_postgres` - IntelOwl database
- `bev_intelowl_celery_beat` - Task scheduler
- `bev_intelowl_celery_worker` - Task processor
- `bev_intelowl_django` - Web application
- `bev_intelowl_nginx` - Web proxy

#### OSINT Intelligence Core (26 services)
- `bev_dm_crawler` - Darknet market crawler
- `bev_crypto_intel` - Cryptocurrency analysis
- `bev_reputation_analyzer` - Reputation scoring
- `bev_economics_processor` - Economic analysis
- `bev_tactical_intel` - Tactical intelligence
- `bev_defense_automation` - Automated defense
- `bev_opsec_enforcer` - Operations security
- `bev_intel_fusion` - Intelligence fusion
- `bev_autonomous_coordinator` - Autonomous coordination
- `bev_predictive_cache` - Predictive caching
- `bev_adaptive_learning` - ML adaptation
- `bev_resource_manager` - Resource optimization
- `bev_knowledge_evolution` - Knowledge management
- Plus 13 auto-recovery services

#### Vector & AI Infrastructure (8 services)
- `bev_qdrant_primary` + `bev_qdrant_replica` - Vector databases
- `bev_weaviate` + `bev_weaviate_transformers` - AI vector processing
- `bev_proxy_manager` - Request routing
- `bev_request_multiplexer` - Request optimization
- `bev_context_compressor` - Context optimization
- `bev_extended_reasoning` - Advanced AI reasoning

#### Edge Computing Network (10 services)
- `bev-edge-us-east` - US East edge node
- `bev-edge-us-west` - US West edge node
- `bev-edge-eu-central` - EU Central edge node
- `bev-edge-asia-pacific` - Asia Pacific edge node
- `bev-edge-management` - Edge coordination
- `bev-model-synchronizer` - Model distribution
- `bev-geo-router` - Geographic routing
- Plus 3 recovery services

#### Chaos Engineering & Testing (2 services)
- `bev_chaos_engineer` - System resilience testing
- `bev_cytoscape_server` - Graph visualization

#### Monitoring Stack (20 services)
- `bev_prometheus` - Metrics collection
- `bev_grafana` - Visualization dashboards
- `bev_node_exporter` - System metrics
- `bev_airflow_scheduler` - Workflow orchestration
- `bev_airflow_webserver` - Workflow UI
- `bev_airflow_worker_[1-3]` - Task execution
- `bev_ocr_service` - OCR processing
- `bev_doc_analyzer_[1-3]` - Document analysis
- `bev_swarm_master_[1-2]` - Swarm coordination
- `bev_research_coordinator` - Research orchestration
- `bev_memory_manager` - Memory optimization
- `bev_code_optimizer` - Code optimization
- `bev_tool_coordinator` - Tool coordination
- `bev_vault` - Credential management
- `bev_guardian_[1-2]` - Security monitoring

### ORACLE1 Node (Secondary - ARM64)
**Role**: Specialized processing, overflow capacity
**Hardware**: ARM64, 24GB RAM
**Service Count**: 62 services

#### Core Services (8 services)
- `bev_redis_oracle` - Redis instance
- `bev_n8n` - Automation workflows
- `bev_nginx` - Web proxy
- `bev_research_crawler` - Research automation
- `bev_intel_processor` - Intelligence processing
- `bev_proxy_manager` - Request management
- `bev_influxdb_primary` + `bev_influxdb_replica` - Time series cluster

#### Monitoring & Metrics (4 services)
- `bev_telegraf` - Metrics collection
- `bev_node_exporter` - System monitoring
- `bev_prometheus` - Metrics storage (ORACLE1 instance)
- `bev_grafana` - Visualization (ORACLE1 instance)

#### Storage Cluster (7 services)
- `bev_minio1` + `bev_minio2` + `bev_minio3` - Object storage cluster
- `bev_minio_expansion` - Storage expansion
- Plus 3 associated services

#### Celery Processing Pipeline (4 services)
- `bev_celery_edge` - Edge processing
- `bev_celery_genetic` - Genetic algorithms
- `bev_celery_knowledge` - Knowledge processing
- `bev_celery_toolmaster` - Tool orchestration

#### LiteLLM AI Gateway (3 services)
- `bev_litellm_[1-3]` - AI model proxies

#### Pipeline Services (12 services)
- `bev_genetic_optimizer` - Genetic optimization
- `bev_request_multiplexer` - Request handling
- `bev_knowledge_synthesis` - Knowledge fusion
- `bev_toolmaster_orchestrator` - Tool management
- `bev_edge_worker_[1-3]` - Edge processing
- `bev_mq_infrastructure` - Message queue management
- Plus 5 other pipeline services

#### Research Workers (8 services)
- `bev_drm_researcher_[1-2]` - DRM research
- `bev_watermark_analyzer_[1-2]` - Watermark analysis
- `bev_crypto_researcher_[1-2]` - Crypto research
- Plus 2 workflow automation services

#### Advanced Automation (6 services)
- `bev_n8n_advanced_[1-3]` - Advanced workflows
- Plus 3 specialized processors

#### Dark Market Intelligence (10 services)
- `bev_blackmarket_crawler_[1-2]` - Market crawling
- `bev_vendor_profiler_[1-2]` - Vendor analysis
- `bev_transaction_tracker` - Transaction monitoring
- `bev_multimodal_processor_[1-4]` - Multimodal analysis
- Plus 1 supporting service

### STARLORD Node (Development Only)
**Role**: Development, testing, staging
**Hardware**: RTX 4090, development workstation
**Service Count**: 0 production services (development only)

#### Development Services (12 services - staging only)
- `bev_staging_postgres` - Development database
- `bev_staging_redis` - Development cache
- `bev_staging_vault` - Development secrets
- `bev_frontend_dev` - Frontend development
- `bev_mcp_[everything|fetch|git|memory|sequential|time]` - MCP development servers
- `bev_docs_server` - Documentation server

## Deployment Configuration Files

### Primary Deployment
- `docker-compose.complete.yml` - Complete system (151 services)
- `deploy-complete-with-vault.sh` - Vault-integrated deployment
- `vault-init.json` - Vault initialization

### Node-Specific Deployments
- `docker-compose-thanos-unified.yml` - THANOS node (89 services)
- `docker-compose-oracle1-unified.yml` - ORACLE1 node (62 services)
- `docker-compose-development.yml` - STARLORD development (12 services)

### Specialized Compositions
- `docker-compose-monitoring.yml` - Monitoring stack only
- `docker-compose-infrastructure.yml` - Core infrastructure
- `docker-compose-phase[7-9].yml` - Phased deployment

### Security & Networking
- Tor integration: 4 nodes (`bev_tor`, `bev_tor_node_[1-3]`)
- Vault credential management: Complete HashiCorp Vault setup
- Security monitoring: Guardian services, IDS, traffic analysis
- Network isolation: Proper service networking and firewalls

### Deployment Scripts
- `deploy-complete-with-vault.sh` - Main deployment with Vault
- `generate-secure-credentials.sh` - Credential generation
- `setup-vault-multinode.sh` - Vault multinode setup
- `validate_bev_deployment.sh` - Post-deployment validation

## Key Service Endpoints

### Web Interfaces
- **IntelOwl Dashboard**: http://localhost (THANOS)
- **Neo4j Browser**: http://localhost:7474 (THANOS)
- **Grafana Monitoring**: http://localhost:3000 (both nodes)
- **Prometheus Metrics**: http://localhost:9090 (both nodes)
- **Cytoscape Visualization**: http://localhost/cytoscape (THANOS)

### API Endpoints
- **MCP API Server**: http://localhost:3010
- **Vault API**: http://localhost:8200
- **N8N Automation**: http://localhost:5678 (ORACLE1)

### Database Connections
- **PostgreSQL**: localhost:5432 (pgvector enabled)
- **Neo4j**: bolt://localhost:7687
- **Redis**: localhost:6379 (multiple instances)
- **Elasticsearch**: localhost:9200

## Performance Specifications

### Target Metrics
- **Concurrent Users**: 1000+ simultaneous connections
- **Response Latency**: <100ms average
- **Cache Hit Rate**: >80% efficiency
- **System Availability**: 99.9% uptime
- **Recovery Time**: <5 minutes after failures

### Resource Allocation
- **THANOS**: Heavy computation, 89 services, full databases
- **ORACLE1**: Specialized processing, 62 services, distributed storage
- **Network**: Cross-node communication optimized for 0.0.0.0 binding

## Deployment Status

### Current State
- ✅ **Vault Integration**: Complete credential management system
- ✅ **Multinode Architecture**: THANOS + ORACLE1 distribution
- ✅ **Security**: Enterprise-grade credential management
- ✅ **Monitoring**: Comprehensive observability stack
- ✅ **Testing**: Validation and health check frameworks

### Ready for Production
- All 151 services properly configured
- Vault credential management operational
- Cross-node networking validated
- Security hardening complete
- Monitoring and alerting active

**Next Action**: Execute `./deploy-complete-with-vault.sh` for production deployment