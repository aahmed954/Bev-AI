# BEV OSINT Framework - Deployment Guide

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [Deployment Process](#deployment-process)
5. [Post-Deployment Validation](#post-deployment-validation)
6. [Monitoring Setup](#monitoring-setup)
7. [Troubleshooting](#troubleshooting)
8. [Rollback Procedures](#rollback-procedures)
9. [Maintenance](#maintenance)

## Overview

This guide provides comprehensive instructions for deploying BEV OSINT Framework Phases 7, 8, and 9. The deployment system includes:

- **Phase 7**: Alternative Market Intelligence
- **Phase 8**: Advanced Security Operations
- **Phase 9**: Autonomous Enhancement

### Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    PHASE 7      │    │    PHASE 8      │    │    PHASE 9      │
│Market Intelligence│    │Security Operations│  │ Autonomous Ops  │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│• DM Crawler     │    │• Tactical Intel │    │• Auto Coordinator│
│• Crypto Intel   │    │• Defense Auto   │    │• Adaptive Learn │
│• Reputation     │    │• OPSEC Monitor  │    │• Resource Mgr   │
│• Economics      │    │• Intel Fusion   │    │• Knowledge Evol │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **CPU**: 16 cores (8 physical)
- **Memory**: 64GB RAM
- **Storage**: 500GB SSD
- **Network**: 1Gbps connection
- **GPU**: NVIDIA RTX 4090 (for Phase 9)

**Recommended Requirements:**
- **CPU**: 32 cores (16 physical)
- **Memory**: 128GB RAM
- **Storage**: 1TB NVMe SSD
- **Network**: 10Gbps connection
- **GPU**: Multiple NVIDIA RTX 4090s

### Software Dependencies

**Required Software:**
- Docker Engine 20.10+
- Docker Compose 2.0+
- Python 3.8+
- Git 2.30+
- NVIDIA Docker (for GPU support)

**Operating System:**
- Ubuntu 20.04 LTS or newer
- RHEL 8+ (compatible)
- Debian 11+ (compatible)

### Network Requirements

**Port Allocation:**
- **Phase 7**: 8001-8004
- **Phase 8**: 8005-8008
- **Phase 9**: 8009-8012
- **Infrastructure**: 5432, 6379, 9092-9094, 9200, 7687, 8086

**External Dependencies:**
- Internet connectivity for threat feeds
- DNS resolution for external APIs
- NTP synchronization

## Pre-Deployment Checklist

### 1. Environment Preparation

```bash
# Clone repository
git clone <repository-url> /opt/bev-osint
cd /opt/bev-osint

# Set environment variables
cp .env.example .env
# Edit .env with your configuration

# Create required directories
mkdir -p logs/{deployment,monitoring,integration_tests}
mkdir -p backups
mkdir -p models
```

### 2. System Validation

Run the pre-deployment validation:

```bash
# Full system validation
python3 deployment/validation/pre_deployment_check.py --phases 7,8,9

# Quick validation
python3 deployment/validation/pre_deployment_check.py --phases 7 --verbose
```

### 3. Infrastructure Setup

Ensure core infrastructure is running:

```bash
# Start core services
docker-compose -f docker-compose.complete.yml up -d \
  postgres neo4j elasticsearch kafka-1 kafka-2 kafka-3 redis influxdb

# Verify infrastructure
docker ps | grep -E "(postgres|neo4j|elasticsearch|kafka|redis|influx)"
```

### 4. GPU Validation (Phase 9)

```bash
# Check NVIDIA driver
nvidia-smi

# Test GPU Docker access
docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi
```

## Deployment Process

### Option 1: Complete Deployment (All Phases)

```bash
# Deploy all phases with validation
./deployment/scripts/deploy_phases_7_8_9.sh

# Deploy with custom options
./deployment/scripts/deploy_phases_7_8_9.sh \
  --phases 7,8,9 \
  --parallel \
  --timeout 1800
```

### Option 2: Sequential Phase Deployment

```bash
# Deploy Phase 7
./deployment/scripts/deploy_phase_7.sh

# Validate Phase 7
python3 deployment/validation/post_deployment_validation.py --phases 7

# Deploy Phase 8
./deployment/scripts/deploy_phase_8.sh

# Validate Phase 8
python3 deployment/validation/post_deployment_validation.py --phases 8

# Deploy Phase 9
./deployment/scripts/deploy_phase_9.sh

# Validate Phase 9
python3 deployment/validation/post_deployment_validation.py --phases 9
```

### Option 3: Individual Service Deployment

```bash
# Deploy specific phase services
docker-compose -f docker-compose-phase7.yml up -d dm-crawler crypto-intel

# Check service status
docker-compose -f docker-compose-phase7.yml ps
```

### Deployment Flags and Options

**Master Deployment Script Options:**

```bash
--phases PHASES         # Comma-separated phases (default: 7,8,9)
--skip-validation      # Skip post-deployment validation
--skip-backup         # Skip pre-deployment backup
--dry-run            # Show what would be done
--force              # Continue even if phases fail
--parallel           # Deploy phases in parallel
--timeout SECONDS    # Deployment timeout per phase
```

**Example Commands:**

```bash
# Production deployment with full validation
./deployment/scripts/deploy_phases_7_8_9.sh --timeout 3600

# Development deployment (fast, minimal validation)
./deployment/scripts/deploy_phases_7_8_9.sh \
  --skip-backup \
  --parallel \
  --timeout 900

# Emergency deployment with force
./deployment/scripts/deploy_phases_7_8_9.sh \
  --force \
  --skip-validation \
  --phases 7,8
```

## Post-Deployment Validation

### Automated Validation

```bash
# Complete validation suite
python3 deployment/validation/post_deployment_validation.py --phases 7,8,9

# Integration testing
python3 deployment/tests/integration_test_framework.py --phases 7,8,9

# Monitoring validation
python3 deployment/validation/monitoring_validation.py --phases 7,8,9
```

### Manual Validation

**Service Health Checks:**

```bash
# Check all services
docker ps --filter "name=bev_" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Health endpoint checks
for port in {8001..8012}; do
  echo "Checking port $port:"
  curl -f http://localhost:$port/health || echo "FAILED"
done
```

**Phase-Specific Validation:**

```bash
# Phase 7 - Market Intelligence
curl http://localhost:8001/api/v1/crawl/status
curl http://localhost:8002/api/v1/blockchain/status
curl http://localhost:8003/api/v1/reputation/status
curl http://localhost:8004/api/v1/economics/status

# Phase 8 - Security Operations
curl http://localhost:8005/api/v1/intel/threats
curl http://localhost:8006/api/v1/defense/status
curl http://localhost:8007/api/v1/opsec/status
curl http://localhost:8008/api/v1/fusion/status

# Phase 9 - Autonomous Operations
curl http://localhost:8009/api/v1/autonomous/status
curl http://localhost:8010/api/v1/learning/status
curl http://localhost:8011/api/v1/resources/status
curl http://localhost:8012/api/v1/knowledge/status
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "bev_alerts.yml"

scrape_configs:
  - job_name: 'bev-phase7'
    static_configs:
      - targets: ['localhost:8001', 'localhost:8002', 'localhost:8003', 'localhost:8004']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'bev-phase8'
    static_configs:
      - targets: ['localhost:8005', 'localhost:8006', 'localhost:8007', 'localhost:8008']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'bev-phase9'
    static_configs:
      - targets: ['localhost:8009', 'localhost:8010', 'localhost:8011', 'localhost:8012']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboards

**Import BEV Dashboards:**

1. Access Grafana: http://localhost:3000 (admin/admin)
2. Import dashboards from `/monitoring/grafana/dashboards/`
3. Configure data sources

**Available Dashboards:**
- BEV OSINT Overview
- Phase 7 - Market Intelligence
- Phase 8 - Security Operations
- Phase 9 - Autonomous Operations
- Infrastructure Monitoring

### Alerting Rules

```yaml
# bev_alerts.yml
groups:
  - name: bev_critical
    rules:
      - alert: BEVServiceDown
        expr: up{job=~"bev-.*"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "BEV service {{ $labels.instance }} is down"

      - alert: BEVHighMemoryUsage
        expr: container_memory_usage_bytes{name=~"bev_.*"} / container_spec_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "BEV service {{ $labels.name }} high memory usage"
```

## Troubleshooting

### Common Issues

#### Issue: Service Failed to Start

**Symptoms:**
- Container exits immediately
- Health checks fail
- Port binding errors

**Diagnosis:**
```bash
# Check container logs
docker logs bev_<service_name>

# Check resource usage
docker stats

# Check port availability
netstat -tlnp | grep <port>
```

**Solutions:**
1. Check environment variables in `.env`
2. Verify port availability
3. Check resource constraints
4. Review service dependencies

#### Issue: Database Connection Failed

**Symptoms:**
- Services report database connectivity issues
- Database containers not running

**Diagnosis:**
```bash
# Check database status
docker ps | grep -E "(postgres|neo4j|redis)"

# Test connectivity
docker exec bev_dm_crawler ping postgres
```

**Solutions:**
1. Ensure database containers are running
2. Check network connectivity
3. Verify database credentials
4. Check firewall rules

#### Issue: GPU Not Available (Phase 9)

**Symptoms:**
- Phase 9 services fail to start
- CUDA errors in logs

**Diagnosis:**
```bash
# Check NVIDIA driver
nvidia-smi

# Test Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi
```

**Solutions:**
1. Install/update NVIDIA drivers
2. Install NVIDIA Docker runtime
3. Restart Docker daemon
4. Check GPU device permissions

### Log Analysis

**Centralized Logging:**
```bash
# View all BEV service logs
docker-compose -f docker-compose-phase7.yml logs -f
docker-compose -f docker-compose-phase8.yml logs -f
docker-compose -f docker-compose-phase9.yml logs -f

# Service-specific logs
docker logs bev_<service_name> --tail 100 -f

# Log aggregation query (if ELK stack is running)
curl -X GET "localhost:9200/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "bool": {
      "must": [
        {"match": {"container.name": "bev_*"}},
        {"range": {"@timestamp": {"gte": "now-1h"}}}
      ]
    }
  }
}'
```

### Performance Optimization

**Resource Monitoring:**
```bash
# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Check system resources
htop
iotop
```

**Performance Tuning:**
1. Adjust container resource limits
2. Optimize database configurations
3. Scale services horizontally
4. Implement caching strategies

## Rollback Procedures

### Automated Rollback

```bash
# Emergency rollback (all phases)
./deployment/rollback/rollback_phases.sh \
  --reason "Critical system failure" \
  --emergency

# Selective rollback
./deployment/rollback/rollback_phases.sh \
  --phases 8,9 \
  --reason "Phase 8 security issue"

# Rollback with data preservation
./deployment/rollback/rollback_phases.sh \
  --reason "Performance issues" \
  --force

# Restore from backup
./deployment/rollback/rollback_phases.sh \
  --restore /path/to/backup \
  --reason "Data recovery"
```

### Manual Rollback Steps

1. **Stop Services:**
```bash
docker-compose -f docker-compose-phase9.yml down
docker-compose -f docker-compose-phase8.yml down
docker-compose -f docker-compose-phase7.yml down
```

2. **Backup Current State:**
```bash
# Create emergency backup
mkdir -p backups/manual_rollback_$(date +%Y%m%d_%H%M%S)
# Export databases and configurations
```

3. **Restore Previous Version:**
```bash
# Restore from previous backup
# Restart core infrastructure
# Redeploy stable version
```

### Rollback Validation

```bash
# Validate rollback success
python3 deployment/validation/post_deployment_validation.py

# Check system health
./scripts/health_check.sh

# Verify data integrity
./scripts/data_integrity_check.sh
```

## Maintenance

### Regular Maintenance Tasks

#### Daily Tasks

1. **Health Monitoring:**
```bash
# Daily health check
./scripts/daily_health_check.sh

# Review alerts and metrics
# Check log volumes and rotation
```

2. **Backup Verification:**
```bash
# Verify backup integrity
./scripts/verify_backups.sh

# Test restore procedures (weekly)
```

#### Weekly Tasks

1. **Performance Review:**
```bash
# Generate performance report
./scripts/weekly_performance_report.sh

# Review resource utilization
# Optimize slow queries
```

2. **Security Updates:**
```bash
# Check for security updates
./scripts/security_update_check.sh

# Update base images
# Scan for vulnerabilities
```

#### Monthly Tasks

1. **Capacity Planning:**
```bash
# Analyze growth trends
./scripts/capacity_analysis.sh

# Plan resource scaling
# Review storage requirements
```

2. **Disaster Recovery Testing:**
```bash
# Test backup restoration
# Validate failover procedures
# Update DR documentation
```

### Update Procedures

#### Service Updates

1. **Preparation:**
```bash
# Create pre-update backup
./deployment/rollback/rollback_phases.sh --backup-only

# Test update in staging environment
```

2. **Rolling Update:**
```bash
# Update one service at a time
docker-compose -f docker-compose-phase7.yml up -d dm-crawler

# Validate service health
curl http://localhost:8001/health

# Proceed to next service
```

3. **Validation:**
```bash
# Run post-update validation
python3 deployment/validation/post_deployment_validation.py

# Monitor for issues
```

#### Infrastructure Updates

1. **Database Updates:**
```bash
# Backup databases
./scripts/backup_databases.sh

# Update database containers
# Run migrations if required
# Validate data integrity
```

2. **Monitoring Stack Updates:**
```bash
# Update Prometheus
# Update Grafana
# Import updated dashboards
# Validate metrics collection
```

### Scaling Procedures

#### Horizontal Scaling

```bash
# Scale specific service
docker-compose -f docker-compose-phase7.yml up -d --scale dm-crawler=3

# Load balancer configuration
# Health check updates
```

#### Vertical Scaling

```bash
# Update resource limits in compose files
# Restart services with new limits
# Monitor performance improvements
```

### Monitoring and Alerting Maintenance

1. **Alert Review:**
```bash
# Review alert rules
# Update thresholds based on performance data
# Test alert delivery
```

2. **Dashboard Maintenance:**
```bash
# Update dashboard queries
# Add new metrics
# Remove obsolete panels
```

3. **Log Management:**
```bash
# Configure log rotation
# Set up log archival
# Monitor log volumes
```

## Emergency Procedures

### Emergency Contacts

- **Primary On-Call**: [Contact Information]
- **Secondary On-Call**: [Contact Information]
- **Infrastructure Team**: [Contact Information]
- **Security Team**: [Contact Information]

### Emergency Response

1. **Assess Situation:**
   - Determine severity and impact
   - Identify affected services
   - Document timeline

2. **Immediate Actions:**
   - Execute emergency procedures
   - Communicate with stakeholders
   - Begin root cause analysis

3. **Recovery Actions:**
   - Implement fixes or rollback
   - Validate recovery
   - Update documentation

### Emergency Runbooks

- [Service Outage Runbook](runbooks/service_outage.md)
- [Security Incident Runbook](runbooks/security_incident.md)
- [Data Loss Recovery Runbook](runbooks/data_recovery.md)
- [Performance Degradation Runbook](runbooks/performance_issues.md)

---

## Additional Resources

- **API Documentation**: `/docs/api/`
- **Architecture Documentation**: `/docs/architecture/`
- **Security Guidelines**: `/docs/security/`
- **Performance Tuning**: `/docs/performance/`
- **Development Guide**: `/docs/development/`

For additional support or questions, please refer to the troubleshooting section or contact the development team.