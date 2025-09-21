# BEV ENTERPRISE PLATFORM - DISTRIBUTION STRATEGY

**Date:** September 20, 2025
**Document Version:** 1.0
**Platform:** BEV OSINT Framework Enterprise Command Center
**Target Architecture:** 3-Node Distributed Deployment

---

## 🖥️ **HARDWARE ASSESSMENT REPORT**

### **NODE 1: THANOS (Primary Compute Node)**
**Specifications:**
- **CPU:** AMD Ryzen 9 5900X (12-Core, 24 threads)
- **Memory:** 64GB RAM (62GB available)
- **Storage:** 1.8TB NVMe SSD (1.7TB available)
- **GPU:** NVIDIA RTX 3080 (10GB VRAM)
- **Network:** Gigabit Ethernet + Tailscale VPN
- **Architecture:** x86_64
- **Availability:** 24/7 Always-On
- **Role:** **Primary compute and GPU-intensive workloads**

### **NODE 2: ORACLE1 (ARM Processing Node)**
**Specifications:**
- **CPU:** ARM Neoverse-N1 (4-Core, 4 threads)
- **Memory:** 24GB RAM (22GB available)
- **Storage:** 194GB SSD (185GB available)
- **GPU:** None (CPU-only processing)
- **Network:** Gigabit Ethernet + Tailscale VPN
- **Architecture:** aarch64 (ARM64)
- **Availability:** 24/7 Always-On
- **Role:** **Lightweight services and ARM-optimized workloads**

### **NODE 3: STARLORD (Development & Frontend Node)**
**Specifications:**
- **CPU:** x86_64 Architecture
- **Memory:** Available for development workloads
- **Storage:** Development and staging environment
- **GPU:** Limited/No GPU acceleration
- **Network:** Standard connectivity
- **Architecture:** x86_64
- **Availability:** On-Demand (Active use only)
- **Role:** **Development, frontend deployment, and staging**

---

## 📊 **BEV COMPONENT RESOURCE ANALYSIS**

### **HIGH-COMPUTE REQUIREMENTS (GPU + High CPU/Memory)**

#### **Tier 1: GPU-Intensive Services** → **THANOS ONLY**
```yaml
AI/ML Processing Services:
  - adaptive-learning: 8GB RAM, 2 GPU, 4 CPU cores
  - autonomous-coordinator: 6GB RAM, 1 GPU, 2.5 CPU cores
  - knowledge-evolution: 12GB RAM, 2 GPU, 4 CPU cores
  - extended-reasoning: 4GB RAM, 1 GPU, 2 CPU cores
  - t2v-transformers: 8GB RAM, 1 GPU, 3 CPU cores
  - model-synchronizer: 2GB RAM, 0.5 GPU, 1 CPU core

Total GPU Requirements: 6.5 GPU units (RTX 3080 = 10GB sufficient)
Total Memory: 40GB RAM (64GB available on Thanos)
```

#### **Tier 2: High-Memory Database Services** → **THANOS PRIMARY**
```yaml
Database Cluster (High I/O):
  - postgres: 4GB RAM, 2 CPU cores, high disk I/O
  - neo4j: 8GB RAM, 2 CPU cores, graph processing
  - elasticsearch: 8GB RAM, 3 CPU cores, indexing
  - influxdb: 4GB RAM, 1 CPU core, time-series
  - qdrant-primary: 6GB RAM, 2 CPU cores, vector search
  - weaviate: 4GB RAM, 1.5 CPU cores, ML embeddings

Total Memory: 34GB RAM
Total CPU: 11.5 cores (24 cores available on Thanos)
```

#### **Tier 3: Message Queue Infrastructure** → **THANOS PRIMARY**
```yaml
Message Queues (High Throughput):
  - kafka-1, kafka-2, kafka-3: 6GB RAM, 3 CPU cores each
  - rabbitmq-1, rabbitmq-2, rabbitmq-3: 4GB RAM, 2 CPU cores each
  - zookeeper: 2GB RAM, 1 CPU core

Total Memory: 32GB RAM
Total CPU: 12 cores
```

### **MEDIUM-COMPUTE REQUIREMENTS (CPU-Intensive)**

#### **Tier 4: Processing Services** → **THANOS + ORACLE1**
```yaml
OSINT Processing (ARM-Compatible):
  - intelowl-django: 2GB RAM, 1.5 CPU cores ✅ ARM
  - intelowl-celery-worker: 4GB RAM, 2 CPU cores ✅ ARM
  - cytoscape-server: 1GB RAM, 1 CPU core ✅ ARM
  - breach-analyzer: 1GB RAM, 0.5 CPU cores ✅ ARM
  - darknet-analyzer: 2GB RAM, 1 CPU core ✅ ARM
  - crypto-analyzer: 1.5GB RAM, 1 CPU core ✅ ARM
  - social-analyzer: 2GB RAM, 1 CPU core ✅ ARM
```

#### **Tier 5: Infrastructure Services** → **ORACLE1 OPTIMAL**
```yaml
Lightweight Infrastructure (ARM-Optimized):
  - redis: 2GB RAM, 1 CPU core ✅ ARM
  - prometheus: 3GB RAM, 1.5 CPU cores ✅ ARM
  - grafana: 1GB RAM, 0.5 CPU cores ✅ ARM
  - vault: 1GB RAM, 0.5 CPU cores ✅ ARM
  - tor: 512MB RAM, 0.5 CPU cores ✅ ARM
  - nginx: 512MB RAM, 0.5 CPU cores ✅ ARM
  - proxy-manager: 1GB RAM, 0.5 CPU cores ✅ ARM
```

### **LOW-COMPUTE REQUIREMENTS (Utility Services)**

#### **Tier 6: Monitoring & Utilities** → **ORACLE1 + STARLORD**
```yaml
Monitoring Services (ARM-Compatible):
  - health-monitor: 512MB RAM, 0.25 CPU cores ✅ ARM
  - metrics-collector: 1GB RAM, 0.5 CPU cores ✅ ARM
  - alert-system: 512MB RAM, 0.25 CPU cores ✅ ARM
  - chaos-engineer: 1GB RAM, 0.5 CPU cores ✅ ARM
  - auto-recovery: 512MB RAM, 0.25 CPU cores ✅ ARM
```

#### **Tier 7: Development Services** → **STARLORD**
```yaml
Development Environment:
  - bev-frontend: 2GB RAM, 1 CPU core (SvelteKit dev server)
  - tauri-dev: 1GB RAM, 0.5 CPU cores (Tauri development)
  - mcp-servers: 1GB RAM, 0.5 CPU cores (MCP development)
  - testing-framework: 2GB RAM, 1 CPU core (Test execution)
```

---

## 🎯 **OPTIMAL DISTRIBUTION STRATEGY**

### **THANOS NODE (Primary Compute Center)**
**Role:** High-performance computing, AI/ML processing, primary databases
**Availability:** 24/7 Always-On
**Resource Allocation:** 80% capacity utilization

**Deployed Services (32 services):**
```yaml
AI/ML Services (GPU Required):
  - autonomous-coordinator (port 8009)
  - adaptive-learning (port 8010)
  - resource-manager (port 8011)
  - knowledge-evolution (port 8012)
  - extended-reasoning (port 8081)
  - context-compressor (port 8080)
  - t2v-transformers (port 8083)

Primary Databases:
  - postgres (port 5432)
  - neo4j (port 7474, 7687)
  - elasticsearch (port 9200)
  - influxdb (port 8086)
  - qdrant-primary (port 6333)
  - weaviate (port 8080)

Message Queue Cluster:
  - kafka-1, kafka-2, kafka-3 (ports 9092-9094)
  - rabbitmq-1, rabbitmq-2, rabbitmq-3 (ports 5672-5674)
  - zookeeper (port 2181)

Core Processing:
  - intelowl-django (port 80)
  - intelowl-celery-worker
  - intelowl-celery-beat
  - request-multiplexer (port 8001)
  - predictive-cache (port 8002)
```

**Expected Resource Usage:**
- **Memory:** 50GB/64GB (78% utilization)
- **CPU:** 18/24 cores (75% utilization)
- **GPU:** 6.5/10GB VRAM (65% utilization)
- **Storage:** 400GB/1.8TB (22% utilization)

### **ORACLE1 NODE (ARM Processing Center)**
**Role:** Lightweight services, monitoring, ARM-optimized workloads
**Availability:** 24/7 Always-On
**Resource Allocation:** 70% capacity utilization

**Deployed Services (25 services):**
```yaml
Monitoring & Infrastructure:
  - redis (port 6379)
  - prometheus (port 9090)
  - grafana (port 3000)
  - vault (port 8200)
  - consul (port 8500)

Security & Network:
  - tor (ports 9050-9052)
  - proxy-manager (port 8003)
  - opsec-enforcer (port 8004)
  - defense-automation (port 8005)

OSINT Analyzers:
  - breach-analyzer (port 8006)
  - darknet-analyzer (port 8007)
  - crypto-analyzer (port 8008)
  - social-analyzer (port 8013)
  - reputation-analyzer (port 8014)
  - economics-processor (port 8015)

Support Services:
  - health-monitor (port 8016)
  - metrics-collector (port 8017)
  - alert-system (port 8018)
  - chaos-engineer (port 8019)
  - auto-recovery (port 8070)

Edge Computing:
  - edge-node-management (port 8020)
  - geo-router (port 8021)
```

**Expected Resource Usage:**
- **Memory:** 18GB/24GB (75% utilization)
- **CPU:** 3/4 cores (75% utilization)
- **Storage:** 100GB/194GB (51% utilization)

### **STARLORD NODE (Development & Frontend)**
**Role:** Development environment, frontend hosting, staging
**Availability:** On-Demand (Active use only)
**Resource Allocation:** 60% capacity utilization

**Deployed Services (10 services):**
```yaml
Development Environment:
  - bev-frontend (port 5173) - SvelteKit dev server
  - tauri-dev (port 1420) - Tauri development
  - mcp-servers (ports 3001-3006) - MCP development stack
  - testing-framework (port 8050-8053) - Test execution

Staging Services:
  - staging-postgres (port 5433)
  - staging-redis (port 6380)
  - staging-vault (port 8201)

Development Tools:
  - code-server (port 8443) - VS Code server
  - git-server (port 9418) - Local git hosting
  - documentation-server (port 8080) - Documentation hosting
```

**Expected Resource Usage:**
- **Memory:** 8GB RAM (development workload)
- **CPU:** 4-6 cores (variable based on development activity)
- **Storage:** 50GB (code, staging data, logs)

---

## 🔗 **NETWORK ARCHITECTURE & COMMUNICATION**

### **Inter-Node Communication:**
```yaml
Primary Network: Tailscale VPN Mesh
  - thanos: 100.122.12.53 (static IP)
  - oracle1: 100.96.197.84 (static IP)
  - starlord: 100.122.12.54 (dynamic IP)

Secondary Network: Direct SSH Access
  - ssh thanos (x86_64, high-performance)
  - ssh oracle1 (aarch64, ARM processing)
  - Local development on starlord

Service Discovery: Consul + Docker Swarm
  - Service mesh across all nodes
  - Automatic failover and load balancing
  - Health checks and service registration
```

### **Data Replication Strategy:**
```yaml
Database Replication:
  - PostgreSQL: Master on Thanos, Read replica on Oracle1
  - Neo4j: Single master on Thanos (graph integrity)
  - Redis: Cluster mode across Thanos + Oracle1
  - Elasticsearch: Primary on Thanos, monitoring on Oracle1

Backup Strategy:
  - Primary backups: Thanos → Oracle1 (nightly)
  - Configuration backups: Oracle1 → Thanos (hourly)
  - Development snapshots: Starlord → Thanos (on-demand)
```

---

## 🚀 **DEPLOYMENT PLAN & IMPLEMENTATION**

### **Phase 1: Thanos Node Deployment (Primary Infrastructure)**
**Priority:** Critical - Deploy first
**Timeline:** Day 1-2

```bash
# Thanos Deployment Commands
ssh thanos
cd /opt/bev-deployment
git clone https://github.com/user/bev-platform.git
cd bev-platform

# Deploy primary infrastructure
docker-compose -f docker-compose-thanos-unified.yml up -d

# Verify GPU services
docker exec bev_autonomous_coordinator nvidia-smi
docker exec bev_adaptive_learning nvidia-smi

# Initialize databases
./scripts/initialize_databases.sh
./scripts/setup_gpu_services.sh
```

**Services to Deploy on Thanos:**
- All GPU-dependent AI/ML services
- Primary database cluster
- Message queue infrastructure
- Core OSINT processing engines
- High-throughput analytics

### **Phase 2: Oracle1 Node Deployment (ARM Services)**
**Priority:** High - Deploy second
**Timeline:** Day 2-3

```bash
# Oracle1 Deployment Commands
ssh oracle1
cd /opt/bev-deployment
git clone https://github.com/user/bev-platform.git
cd bev-platform

# Deploy ARM-optimized services
docker-compose -f docker-compose-oracle1-unified.yml up -d

# Verify ARM compatibility
docker exec bev_redis arch
docker exec bev_prometheus arch

# Setup monitoring and security
./scripts/setup_monitoring.sh
./scripts/setup_security.sh
```

**Services to Deploy on Oracle1:**
- Monitoring and alerting stack
- Security and privacy services
- Lightweight OSINT analyzers
- Infrastructure utilities
- ARM-optimized support services

### **Phase 3: Starlord Node Setup (Development)**
**Priority:** Medium - Deploy last
**Timeline:** Day 3-4

```bash
# Starlord Development Setup
cd /home/starlord/Projects/Bev

# Setup development environment
./setup_development_environment.sh
npm install
cd bev-frontend && npm run dev

# Deploy staging services
docker-compose -f docker-compose-development.yml up -d

# Setup MCP servers for development
./scripts/setup_mcp_development.sh
```

**Services to Deploy on Starlord:**
- Frontend development server
- Staging database instances
- Testing framework
- MCP development stack
- Documentation server

---

## ⚖️ **LOAD BALANCING & FAILOVER STRATEGY**

### **High Availability Configuration:**
```yaml
Critical Services Redundancy:
  Primary: Thanos
  Backup: Oracle1 (where possible)

Database Replication:
  - PostgreSQL: Master-Slave (Thanos → Oracle1)
  - Redis: Cluster with failover
  - Vault: HA mode with Oracle1 standby

Load Balancing:
  - HAProxy on Oracle1 for web traffic
  - Kafka cluster load balancing
  - Database connection pooling
  - Service mesh traffic distribution
```

### **Failover Scenarios:**
```yaml
Thanos Node Failure:
  Action: Oracle1 promotes to primary for critical services
  Recovery: 5-10 minutes automated failover
  Data Loss: <1 minute (Redis cluster + DB replication)

Oracle1 Node Failure:
  Action: Thanos handles all monitoring temporarily
  Recovery: Manual restart, services auto-reconnect
  Data Loss: None (backup monitoring only)

Starlord Node Failure:
  Action: Development continues on other nodes
  Recovery: On-demand restart for development
  Data Loss: None (development environment only)
```

---

## 🔧 **DOCKER COMPOSE DISTRIBUTION**

### **Thanos Compose Configuration:**
```yaml
# docker-compose-thanos-production.yml
services:
  # GPU Services
  autonomous-coordinator:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1

  # Primary Databases
  postgres:
    environment:
      POSTGRES_MAX_CONNECTIONS: 500
      SHARED_BUFFERS: 8GB

  # Message Queues
  kafka-cluster:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 6G
```

### **Oracle1 Compose Configuration:**
```yaml
# docker-compose-oracle1-production.yml
services:
  # ARM-Optimized Services
  redis:
    image: redis:alpine
    platform: linux/arm64

  prometheus:
    image: prom/prometheus:latest
    platform: linux/arm64

  # Security Services
  vault:
    image: vault:latest
    platform: linux/arm64
```

### **Starlord Development Configuration:**
```yaml
# docker-compose-development.yml
services:
  # Development Services
  bev-frontend:
    build: ./bev-frontend
    ports:
      - "5173:5173"

  staging-postgres:
    image: postgres:16
    ports:
      - "5433:5432"
```

---

## 📋 **DEPLOYMENT AUTOMATION SCRIPTS**

### **Master Deployment Script:**
```bash
#!/bin/bash
# deploy_distributed_bev.sh

echo "🚀 BEV Distributed Deployment Starting..."

# Phase 1: Deploy Thanos (Primary)
echo "📡 Deploying Thanos Node (Primary Compute)..."
ssh thanos 'cd /opt/bev && ./deploy_thanos_primary.sh'

# Wait for Thanos readiness
echo "⏳ Waiting for Thanos services to stabilize..."
sleep 60

# Phase 2: Deploy Oracle1 (ARM Services)
echo "🔧 Deploying Oracle1 Node (ARM Services)..."
ssh oracle1 'cd /opt/bev && ./deploy_oracle1_services.sh'

# Phase 3: Setup Development (Starlord)
echo "💻 Setting up Starlord Development Environment..."
./setup_starlord_development.sh

# Verify distributed deployment
echo "✅ Verifying distributed deployment..."
./scripts/verify_distributed_deployment.sh

echo "🎯 BEV Distributed Deployment Complete!"
```

### **Node-Specific Deployment Scripts:**

#### **Thanos Deployment Script:**
```bash
#!/bin/bash
# deploy_thanos_primary.sh

# Setup GPU environment
nvidia-docker run --rm nvidia/cuda:12.0-runtime-ubuntu20.04 nvidia-smi

# Deploy AI/ML services with GPU
docker-compose -f docker-compose-thanos-unified.yml up -d

# Initialize databases
./scripts/init_primary_databases.sh

# Verify GPU services
docker exec bev_autonomous_coordinator nvidia-smi
```

#### **Oracle1 Deployment Script:**
```bash
#!/bin/bash
# deploy_oracle1_services.sh

# Verify ARM architecture
uname -m # Should output: aarch64

# Deploy ARM-compatible services
docker-compose -f docker-compose-oracle1-unified.yml up -d

# Setup monitoring
./scripts/setup_arm_monitoring.sh

# Configure security services
./scripts/setup_arm_security.sh
```

---

## 🔍 **MONITORING & HEALTH CHECKS**

### **Cross-Node Monitoring:**
```yaml
Health Check Endpoints:
  - Thanos: http://thanos:9090/health
  - Oracle1: http://oracle1:9090/health
  - Starlord: http://localhost:9090/health

Service Discovery:
  - Consul UI: http://oracle1:8500
  - Service mesh monitoring across all nodes
  - Automatic service registration

Performance Monitoring:
  - Grafana: http://oracle1:3000 (centralized dashboards)
  - Prometheus: Federated setup across all nodes
  - Resource usage alerts and notifications
```

### **Distributed Logging:**
```yaml
Log Aggregation:
  - Central logging: Oracle1 (lightweight ARM processing)
  - Log sources: All nodes stream to Oracle1
  - Elasticsearch: Primary on Thanos, monitoring on Oracle1
  - Log correlation: Cross-node event correlation
```

---

## 🛡️ **SECURITY & NETWORK CONFIGURATION**

### **Network Security:**
```yaml
Firewall Rules:
  - Only Tailscale VPN traffic allowed between nodes
  - SSH access restricted to management
  - Service ports exposed only within VPN

Access Control:
  - Vault secrets distributed to all nodes
  - Consul service discovery with ACLs
  - TLS encryption for all inter-node communication

Tor Network:
  - Tor relays distributed across all nodes
  - Entry node: Oracle1 (ARM efficiency)
  - Middle node: Thanos (high throughput)
  - Exit node: Starlord (development isolation)
```

### **Backup & Disaster Recovery:**
```yaml
Backup Strategy:
  - Primary backups: Thanos → Oracle1 (nightly)
  - Configuration sync: All nodes → centralized vault
  - Code repositories: Git sync across all nodes

Recovery Strategy:
  - Thanos failure: Oracle1 promotes critical services
  - Oracle1 failure: Thanos handles monitoring temporarily
  - Starlord failure: Development pauses, no production impact
```

---

## 📈 **PERFORMANCE OPTIMIZATION**

### **Resource Optimization:**
```yaml
CPU Optimization:
  - Thanos: High-compute AI/ML workloads
  - Oracle1: ARM-efficient utilities and monitoring
  - Starlord: Development tools and staging

Memory Optimization:
  - Thanos: 50GB production workloads
  - Oracle1: 18GB lightweight services
  - Starlord: 8GB development environment

Network Optimization:
  - Inter-node: Tailscale mesh for secure communication
  - Load balancing: HAProxy on Oracle1
  - CDN: Static assets distributed across nodes
```

### **Scaling Strategy:**
```yaml
Horizontal Scaling:
  - Add more ARM nodes (Oracle2, Oracle3) for monitoring
  - Add GPU nodes (Thanos2) for AI/ML scaling
  - Keep Starlord as dedicated development node

Vertical Scaling:
  - Thanos: GPU upgrade for more AI workloads
  - Oracle1: Memory upgrade for larger monitoring
  - Starlord: Storage upgrade for development
```

---

## 🎯 **DEPLOYMENT COMMANDS**

### **Complete Distributed Deployment:**
```bash
# Master deployment command
./deploy_distributed_bev.sh

# Individual node deployment
ssh thanos './deploy_thanos_primary.sh'
ssh oracle1 './deploy_oracle1_services.sh'
./deploy_starlord_development.sh

# Verification
./scripts/verify_distributed_health.sh
./scripts/test_cross_node_communication.sh
```

### **Management Commands:**
```bash
# Start/stop entire distributed system
./scripts/start_distributed_bev.sh
./scripts/stop_distributed_bev.sh

# Health monitoring
./scripts/monitor_distributed_health.sh

# Backup and recovery
./scripts/backup_distributed_system.sh
./scripts/restore_from_backup.sh
```

---

## ✅ **DEPLOYMENT READINESS CHECKLIST**

### **Pre-Deployment Requirements:**
- [ ] Tailscale VPN configured on all nodes
- [ ] SSH key access established (thanos, oracle1)
- [ ] Docker installed on all nodes
- [ ] Git repositories synced
- [ ] Vault unsealed and configured
- [ ] Network firewall rules configured

### **Post-Deployment Verification:**
- [ ] All 67 services running across distributed nodes
- [ ] Cross-node communication functional
- [ ] Database replication working
- [ ] Monitoring dashboards accessible
- [ ] Frontend accessible from all nodes
- [ ] Backup and recovery tested

---

## 🎖️ **EXPECTED BENEFITS**

### **Performance Benefits:**
- **3x Processing Power:** Distributed compute across specialized hardware
- **GPU Acceleration:** Dedicated AI/ML processing on RTX 3080
- **ARM Efficiency:** Power-efficient monitoring and utilities
- **Development Isolation:** Dedicated development environment

### **Reliability Benefits:**
- **High Availability:** 24/7 services on Thanos + Oracle1
- **Automatic Failover:** Critical service redundancy
- **Disaster Recovery:** Cross-node backup and restore
- **Development Safety:** Isolated staging environment

### **Cost Benefits:**
- **Optimized Resource Usage:** Right-sized services for hardware
- **Power Efficiency:** ARM node for lightweight services
- **Development Efficiency:** Dedicated development node
- **Scaling Flexibility:** Add nodes as needed

---

**🚀 READY FOR DISTRIBUTED ENTERPRISE DEPLOYMENT!**

This distribution strategy optimally utilizes the hardware capabilities of each node while ensuring high availability, performance, and cost efficiency for the complete BEV Enterprise Command Center.