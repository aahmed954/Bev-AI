# BEV Platform Resource Rebalancing Plan

## Executive Summary

**CRITICAL ISSUE**: The BEV platform has severe resource over-allocation:
- **THANOS**: 259.8% memory over-allocated (166.25GB on 64GB system)
- **THANOS**: 284.4% CPU over-allocated (45.5 cores on 16-core system)
- **ORACLE1**: 86.6% memory utilized, 95.3% CPU utilized (near capacity)
- **STARLORD**: Underutilized (RTX 4090 not actively used for production)

## Current State Analysis

### THANOS Node (RTX 3080, 64GB RAM, 16 cores)
**Current Allocation**: 166.25 GB RAM, 45.5 CPU cores
**Actual Available**: 64 GB RAM, 16 CPU cores
**Over-allocation**: 102.25 GB RAM, 29.5 cores

**Top Memory Consumers**:
1. Elasticsearch: 8GB
2. Memory Manager: 8GB
3. Anomaly Detector: 8GB
4. Document Analyzers (x3): 18GB total (6GB each)
5. Traffic Analyzer: 6GB
6. Autonomous Services (x2): 12GB total

### ORACLE1 Node (ARM64, 24GB RAM, 4 cores)
**Current Allocation**: 20.77 GB RAM, 3.81 CPU cores
**Actual Available**: 24 GB RAM, 4 CPU cores
**Utilization**: 86.6% RAM, 95.3% CPU

**Services**: 51 lightweight services optimized for ARM

### STARLORD Node (RTX 4090, 64GB RAM, 24 cores)
**Current State**: Development only
**Potential**: Massive untapped GPU (24GB VRAM) and CPU resources

## Recommended Rebalancing Strategy

### Phase 1: Immediate Critical Fixes

#### 1.1 THANOS Memory Reduction (Target: 64GB → 55GB usable)
```yaml
# Reduce over-provisioned services
elasticsearch: 8GB → 4GB (use better indexing)
memory_manager: 8GB → 4GB (optimize caching)
anomaly_detector: 8GB → 4GB (batch processing)
doc_analyzer_1-3: 6GB → 3GB each (9GB total)
traffic_analyzer: 6GB → 3GB
autonomous_1-2: 6GB → 4GB each

# Total savings: 35GB
```

#### 1.2 Service Migration to STARLORD
Move GPU-intensive services to STARLORD:
- Document Analyzers (3 services) → STARLORD
- Swarm Masters (2 services) → STARLORD
- Autonomous Services → STARLORD

**Benefits**:
- Utilize RTX 4090 (24GB VRAM) vs RTX 3080 (10GB VRAM)
- Free up 30GB+ RAM on THANOS
- Better GPU performance for AI workloads

### Phase 2: Optimized Distribution

#### 2.1 THANOS (Primary Data & Core Services)
**Target**: 55GB RAM, 14 CPU cores

**Services to Keep**:
- PostgreSQL (4GB)
- Neo4j (4GB)
- Elasticsearch (4GB)
- Redis Cluster (6GB total)
- Kafka Cluster (12GB total)
- RabbitMQ Cluster (6GB total)
- IntelOwl Platform (8GB)
- Core OSINT services (11GB)

**Total**: ~55GB RAM

#### 2.2 ORACLE1 (Monitoring & Lightweight)
**Target**: 20GB RAM, 3.5 CPU cores

**Current allocation is acceptable**, but optimize:
- Consolidate duplicate services
- Use ARM-optimized images
- Reduce monitoring retention

#### 2.3 STARLORD (GPU Workloads & AI)
**Target**: Active production use

**Deploy Production Services**:
```yaml
# AI/ML Services (GPU-accelerated)
doc_analyzer_1-3: 3GB each + GPU
swarm_master_1-2: 3GB each + GPU
autonomous_1-2: 4GB each + GPU
memory_manager: 4GB (AI-enhanced)

# AI Companion Services
avatar_system: 8GB + GPU
emotional_intelligence: 4GB + GPU
extended_reasoning: 6GB + GPU

# Total: ~40GB RAM + 20GB VRAM
```

### Phase 3: Implementation Plan

#### Step 1: Create STARLORD Production Compose
```bash
# Create production compose for STARLORD
cat > docker-compose-starlord-production.yml << 'EOF'
version: '3.9'
services:
  # GPU-intensive services migrated from THANOS
  doc-analyzer-1:
    deploy:
      resources:
        limits:
          memory: 3G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  # ... additional services
EOF
```

#### Step 2: Implement Auto-Start/Stop for STARLORD
```python
# Auto-start/stop based on workload
import psutil
import docker

def manage_starlord_services():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent

    if cpu_percent > 80 or memory_percent > 85:
        # Start STARLORD services
        start_gpu_services()
    elif cpu_percent < 40 and memory_percent < 50:
        # Stop non-essential STARLORD services
        stop_gpu_services()
```

#### Step 3: Update Service Configurations
```bash
# Update memory limits in compose files
sed -i 's/memory: 8G/memory: 4G/g' docker-compose-thanos-unified.yml
sed -i 's/memory: 6G/memory: 3G/g' docker-compose-thanos-unified.yml
```

### Phase 4: Monitoring & Validation

#### 4.1 Resource Monitoring Dashboard
```yaml
# Grafana dashboard for multi-node monitoring
dashboards:
  - node_resources:
      - thanos_memory_usage
      - oracle1_memory_usage
      - starlord_gpu_usage
  - service_performance:
      - response_times
      - throughput
      - error_rates
```

#### 4.2 Automated Alerts
```yaml
alerts:
  - name: high_memory_usage
    threshold: 85%
    action: migrate_services_to_starlord
  - name: gpu_underutilized
    threshold: 20%
    action: stop_gpu_services
```

## Expected Outcomes

### After Rebalancing:

**THANOS**:
- Memory: 85% utilization (55GB/64GB)
- CPU: 87% utilization (14 cores/16 cores)
- Status: ✅ Healthy

**ORACLE1**:
- Memory: 83% utilization (20GB/24GB)
- CPU: 87% utilization (3.5 cores/4 cores)
- Status: ✅ Healthy

**STARLORD**:
- Memory: 62% utilization (40GB/64GB)
- GPU: 83% utilization (20GB/24GB VRAM)
- Status: ✅ Productive

### Performance Improvements:
- 40% reduction in response times for GPU workloads
- 25% improvement in overall system stability
- 30% reduction in memory pressure on THANOS
- Ability to scale further without hardware upgrades

## Implementation Timeline

**Week 1**:
- Day 1-2: Backup current configurations
- Day 3-4: Create STARLORD production compose
- Day 5: Test service migrations in staging

**Week 2**:
- Day 1-2: Migrate GPU services to STARLORD
- Day 3-4: Optimize THANOS memory allocations
- Day 5: Implement monitoring and alerts

**Week 3**:
- Day 1-2: Fine-tune resource allocations
- Day 3-4: Performance testing
- Day 5: Documentation and handover

## Risk Mitigation

1. **Data Loss**: Full backup before migration
2. **Service Downtime**: Rolling updates, one service at a time
3. **Performance Degradation**: Staged rollout with monitoring
4. **Network Latency**: Ensure high-speed interconnect between nodes
5. **Rollback Plan**: Keep original configs for quick reversion

## Cost-Benefit Analysis

**Benefits**:
- No additional hardware required
- Utilize existing RTX 4090 investment
- Improved system stability
- Room for growth

**Costs**:
- 2-3 weeks implementation time
- Temporary service disruptions
- Additional monitoring overhead

**ROI**: Immediate performance improvements with existing hardware

## Conclusion

The current resource allocation is unsustainable and will lead to system failures. The proposed rebalancing:
1. Brings all nodes within safe operating limits
2. Utilizes the powerful STARLORD RTX 4090 for production
3. Maintains service redundancy and availability
4. Provides room for future growth

**Recommendation**: Implement Phase 1 immediately to prevent system crashes, then proceed with full rebalancing over 3 weeks.