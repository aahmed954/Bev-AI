# BEV Enterprise Platform - Deployment Execution Guide

**🎯 MISSION-CRITICAL DEPLOYMENT**: 168 services across 3-node distributed architecture

---

## 🚀 PRE-FLIGHT CHECKLIST

### Infrastructure Readiness
- [ ] **Tailscale VPN**: All nodes connected and accessible
- [ ] **SSH Access**: Key-based authentication configured
- [ ] **Docker**: Version 20.10+ running on all nodes
- [ ] **GPU Drivers**: NVIDIA drivers installed on Thanos
- [ ] **Network Ports**: Required ports open (7474, 9090, 3000, 5173)
- [ ] **Disk Space**: Minimum 100GB available per node
- [ ] **Repository**: Clean git status with latest code

### Validation Commands
```bash
# Node connectivity test
ping thanos.tail-scale.ts.net
ping oracle1.tail-scale.ts.net

# SSH key authentication
ssh thanos "echo 'Thanos ready'"
ssh oracle1 "echo 'Oracle1 ready'"

# Docker and GPU verification
ssh thanos "docker --version && nvidia-smi"
ssh oracle1 "docker --version && uname -m"
docker --version

# Repository status
git status
ls -la deploy_local_distributed.sh
```

---

## 🎬 DEPLOYMENT EXECUTION

### Single Command Deployment
```bash
# Execute the complete distributed deployment
./deploy_local_distributed.sh

# Expected runtime: 35-40 minutes
# Monitor progress with distributed health check
watch -n 30 './scripts/health_check_distributed.sh'
```

### Phase-by-Phase Monitoring

#### Phase 1: THANOS Deployment (T+0-15 min)
```bash
# Monitor Thanos deployment progress
ssh thanos "watch -n 10 'docker ps --filter name=bev_ | wc -l'"

# Key milestone checks
ssh thanos "docker exec bev_postgres pg_isready"  # T+5 min
ssh thanos "curl -I http://localhost:7474"        # T+8 min
ssh thanos "docker exec bev_autonomous-coordinator nvidia-smi"  # T+12 min
```

#### Phase 2: ORACLE1 Deployment (T+15-25 min)
```bash
# Monitor Oracle1 deployment progress
ssh oracle1 "watch -n 10 'docker ps --filter name=bev_ | wc -l'"

# Key milestone checks
ssh oracle1 "curl -I http://localhost:9090"       # T+18 min (Prometheus)
ssh oracle1 "curl -I http://localhost:3000"       # T+22 min (Grafana)
ssh oracle1 "curl -s http://localhost:9090/targets" # T+25 min (Federation)
```

#### Phase 3: STARLORD Deployment (T+25-35 min)
```bash
# Monitor local deployment progress
watch -n 10 'docker ps --filter name=bev_ | wc -l'

# Key milestone checks
curl -I http://localhost:3010                     # T+28 min (MCP Server)
curl -I http://localhost:5173                     # T+32 min (Frontend)
```

---

## 📊 REAL-TIME MONITORING

### Deployment Dashboard Commands
```bash
# Overall service count monitoring
echo "=== DEPLOYMENT PROGRESS ===" && \
echo "Thanos: $(ssh thanos 'docker ps --filter name=bev_ | wc -l')" && \
echo "Oracle1: $(ssh oracle1 'docker ps --filter name=bev_ | wc -l')" && \
echo "Starlord: $(docker ps --filter name=bev_ | wc -l)" && \
echo "Total: $(($(ssh thanos 'docker ps --filter name=bev_ | wc -l') + $(ssh oracle1 'docker ps --filter name=bev_ | wc -l') + $(docker ps --filter name=bev_ | wc -l)))"

# Critical service health monitoring
echo "=== CRITICAL SERVICES ===" && \
echo -n "PostgreSQL: " && ssh thanos "docker exec bev_postgres pg_isready" && \
echo -n "Neo4j: " && curl -s http://thanos:7474 > /dev/null && echo "✅ Running" || echo "❌ Down" && \
echo -n "Prometheus: " && curl -s http://oracle1:9090 > /dev/null && echo "✅ Running" || echo "❌ Down" && \
echo -n "Frontend: " && curl -s http://localhost:5173 > /dev/null && echo "✅ Running" || echo "❌ Down"
```

### Performance Monitoring
```bash
# Resource utilization across nodes
echo "=== RESOURCE UTILIZATION ===" && \
echo "Thanos CPU: $(ssh thanos "top -bn1 | grep 'Cpu(s)' | awk '{print \$2}'")" && \
echo "Oracle1 CPU: $(ssh oracle1 "top -bn1 | grep 'Cpu(s)' | awk '{print \$2}'")" && \
echo "Starlord CPU: $(top -bn1 | grep 'Cpu(s)' | awk '{print $2}')" && \
echo "GPU Usage: $(ssh thanos 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits')%"
```

---

## 🏥 HEALTH VALIDATION

### Automated Health Check
```bash
# Comprehensive distributed health check
./scripts/health_check_distributed.sh

# Expected output:
# - Platform Status: FULLY OPERATIONAL
# - Exit code: 0 (success)
```

### Manual Service Verification
```bash
# Core Infrastructure Services
curl -I http://thanos:7474                        # Neo4j Graph Database
curl -I http://oracle1:3000                       # Grafana Monitoring
curl -I http://oracle1:9090                       # Prometheus Metrics
curl -I http://localhost:5173                     # React Frontend

# Database Connectivity
ssh thanos "docker exec bev_postgres psql -U researcher -d osint -c 'SELECT 1;'"
ssh thanos "docker exec bev_neo4j cypher-shell -u neo4j -p BevGraphMaster2024 'RETURN 1'"
ssh oracle1 "docker exec bev_redis redis-cli ping"

# GPU AI Services
ssh thanos "docker exec bev_autonomous-coordinator nvidia-smi"
ssh thanos "curl -s http://localhost:8009/health"  # Autonomous Coordinator
ssh thanos "curl -s http://localhost:8010/health"  # Adaptive Learning
```

### Cross-Node Communication Test
```bash
# Verify Tailscale mesh connectivity
curl -I http://thanos:7474/browser/              # Starlord → Thanos
curl -I http://oracle1:3000/login               # Starlord → Oracle1
ssh oracle1 "curl -I http://thanos:9090"        # Oracle1 → Thanos federation
```

---

## ⚠️ TROUBLESHOOTING GUIDE

### Common Issues & Solutions

#### Issue: GPU Services Not Starting
```bash
# Diagnosis
ssh thanos "nvidia-smi"
ssh thanos "docker exec bev_autonomous-coordinator nvidia-smi"

# Solution
ssh thanos "docker-compose -f docker-compose-thanos-unified.yml restart autonomous-coordinator adaptive-learning"
```

#### Issue: Database Connection Failures
```bash
# Diagnosis
ssh thanos "docker logs bev_postgres | tail -20"
ssh thanos "docker exec bev_postgres pg_isready -U researcher"

# Solution - Database Recovery
./scripts/emergency_rollback.sh
# Select option 2: Database Recovery
```

#### Issue: Cross-Node Communication Failure
```bash
# Diagnosis
ping thanos.tail-scale.ts.net
ping oracle1.tail-scale.ts.net
tailscale status

# Solution
sudo tailscale up
# Restart Tailscale on all nodes if needed
```

#### Issue: Frontend Not Accessible
```bash
# Diagnosis
curl -I http://localhost:5173
ps aux | grep "npm run dev"
docker ps | grep bev_

# Solution
cd bev-frontend
npm run dev &
# Or restart development services
docker-compose -f docker-compose-development.yml restart
```

### Emergency Procedures

#### Partial Rollback (Selective Services)
```bash
./scripts/emergency_rollback.sh
# Select option 1: Selective Service Restart
# Specify node and services to restart
```

#### Full Node Reset
```bash
./scripts/emergency_rollback.sh
# Select option 3: Full Node Reset
# Choose problematic node for complete reset
```

#### Emergency Stop All
```bash
./scripts/emergency_rollback.sh
# Select option 4: Emergency Stop All
# Only use in critical failure scenarios
```

---

## ✅ SUCCESS VALIDATION CRITERIA

### Minimum Viable Deployment
- **Thanos**: >80% services healthy, GPU access confirmed
- **Oracle1**: >75% services healthy, monitoring active
- **Starlord**: >90% services healthy, frontend accessible
- **Cross-node**: 2/3 communication paths working
- **Response times**: <100ms average

### Optimal Deployment
- **Thanos**: >95% services healthy, GPU utilization 60-80%
- **Oracle1**: >90% services healthy, full monitoring stack
- **Starlord**: >95% services healthy, <100ms frontend response
- **Cross-node**: 3/3 communication paths, <50ms latency
- **Federation**: Oracle1 collecting metrics from Thanos

### Final Validation Commands
```bash
# Service count verification (expect ~168 total)
TOTAL=$(ssh thanos "docker ps --filter 'name=bev_' | wc -l"; ssh oracle1 "docker ps --filter 'name=bev_' | wc -l"; docker ps --filter 'name=bev_' | wc -l | paste -sd+ | bc)
echo "Total distributed services: $TOTAL"

# Performance baseline
ssh thanos "./scripts/health_check_thanos.sh"
./scripts/health_check_distributed.sh

# Access verification
curl -I http://thanos (IntelOwl Dashboard)
curl -I http://oracle1:3000 (Grafana)
curl -I http://localhost:5173 (Frontend)
```

---

## 🎯 POST-DEPLOYMENT SETUP

### Initial Configuration
```bash
# Setup Grafana dashboards
curl -X POST http://admin:admin@oracle1:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana-dashboards/bev-platform-dashboard.json

# Configure Neo4j initial constraints
ssh thanos "docker exec bev_neo4j cypher-shell -u neo4j -p BevGraphMaster2024 < scripts/neo4j_constraints.cypher"

# Initialize OSINT analyzers
ssh thanos "docker exec bev_postgres psql -U researcher -d osint < sql/init_analyzers.sql"
```

### Security Hardening
```bash
# Vault initialization (if not auto-unsealed)
ssh oracle1 "docker exec bev_vault vault operator init"
ssh oracle1 "docker exec bev_vault vault operator unseal <key1>"
ssh oracle1 "docker exec bev_vault vault operator unseal <key2>"
ssh oracle1 "docker exec bev_vault vault operator unseal <key3>"

# Tor configuration verification
ssh oracle1 "docker exec bev_tor cat /var/log/tor/tor.log | tail -10"
```

### Development Environment Setup
```bash
# MCP server registration
curl -X POST http://localhost:3010/register \
  -H 'Content-Type: application/json' \
  -d '{"name": "osint-tools", "version": "1.0.0"}'

# Frontend development hot reload verification
echo "Making test change to verify hot reload..."
echo "// Test change $(date)" >> bev-frontend/src/App.tsx
# Watch for automatic browser refresh
```

---

## 📈 MONITORING SETUP

### Grafana Dashboard Access
- **URL**: http://oracle1:3000
- **Credentials**: admin/admin (change on first login)
- **Key Dashboards**:
  - BEV Platform Overview
  - Thanos GPU Utilization
  - Oracle1 ARM Performance
  - Cross-Node Communication
  - OSINT Analysis Metrics

### Prometheus Metrics
- **URL**: http://oracle1:9090
- **Key Metrics**:
  - `bev_services_total`: Total service count
  - `bev_gpu_utilization`: GPU usage percentage
  - `bev_osint_analyses_total`: OSINT investigation count
  - `bev_cross_node_latency`: Inter-node communication latency

### Log Aggregation
```bash
# View aggregated logs across nodes
ssh thanos "docker-compose -f docker-compose-thanos-unified.yml logs -f --tail=50"
ssh oracle1 "docker-compose -f docker-compose-oracle1-unified.yml logs -f --tail=50"
docker-compose -f docker-compose-development.yml logs -f --tail=50
```

---

## 🏆 DEPLOYMENT COMPLETION

### Success Indicators
✅ **168+ distributed services operational**
✅ **Cross-node communication <50ms latency**
✅ **GPU services with CUDA access confirmed**
✅ **Monitoring stack collecting comprehensive metrics**
✅ **Frontend accessible and responsive**
✅ **Security services operational (Vault, Tor)**
✅ **Database federation working**

### Platform Access URLs
- **Primary Control**: http://thanos (IntelOwl Dashboard)
- **Monitoring Hub**: http://oracle1:3000 (Grafana)
- **Development**: http://localhost:5173 (React Frontend)
- **Graph Explorer**: http://thanos:7474 (Neo4j Browser)
- **Metrics**: http://oracle1:9090 (Prometheus)
- **Security**: http://oracle1:8200 (Vault UI)

### Next Steps
1. **Security Configuration**: Complete Vault unsealing and secret management setup
2. **Data Ingestion**: Begin OSINT data collection and analysis workflows
3. **User Training**: Onboard team members to distributed platform capabilities
4. **Performance Optimization**: Monitor and tune resource allocation
5. **Backup Strategy**: Implement automated backup procedures
6. **Incident Response**: Test emergency procedures and rollback capabilities

**🚀 BEV Enterprise Platform is now operational and ready for production OSINT workflows!**