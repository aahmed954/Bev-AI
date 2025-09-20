# BEV Frontend Integration - Deployment Automation Suite

## Overview

This deployment automation suite provides comprehensive infrastructure and deployment management for the BEV Frontend Integration project. The suite consists of five interconnected scripts that handle the complete deployment lifecycle with DevOps best practices.

## Automation Scripts

### 1. Pre-Deployment Validation (`01-pre-deployment-validation.sh`)
**Purpose**: Comprehensive prerequisite checking and environment validation

**Features**:
- System resource validation (memory, disk, CPU)
- Service availability checks (Docker, Docker Compose, curl, etc.)
- Port conflict detection and resolution
- Network configuration validation
- Environment variable validation and generation
- Missing API key generation (BEV_API_KEY, MCP_API_KEY)
- SSL certificate validation
- Security settings verification
- System state backup creation

**Usage**:
```bash
./01-pre-deployment-validation.sh
```

**Outputs**:
- Validation success/failure marker (`.deployment_validation`)
- System backup location
- Comprehensive validation log

### 2. Frontend Deployment (Safe) (`02-frontend-deployment-safe.sh`)
**Purpose**: Conflict-free deployment using safe ports and network isolation

**Features**:
- Uses safe ports (3010, 8443, 8081) to avoid conflicts
- Creates isolated frontend network (172.31.0.0/16)
- SSL certificate generation and management
- HAProxy load balancer configuration
- Docker Compose orchestration
- Health checks and validation
- Service dependency management

**Ports Used**:
- Frontend HTTP: 3010 (redirects to HTTPS)
- Frontend HTTPS: 8443
- MCP Server: 3011
- WebSocket: 8081
- HAProxy Stats: 8080

**Usage**:
```bash
./02-frontend-deployment-safe.sh
```

**Outputs**:
- Deployment success/failure marker (`.frontend_deployment`)
- Complete frontend infrastructure
- Service URLs and access points

### 3. Integration Test Automation (`03-integration-test-automation.sh`)
**Purpose**: Comprehensive testing suite for deployment validation

**Test Categories**:
- **Connectivity Tests**: Service availability, HTTP/HTTPS endpoints, WebSocket connections
- **API Functionality**: MCP server endpoints, BEV integration APIs, CORS headers
- **Security Tests**: SSL certificates, security headers, input validation, rate limiting
- **Performance Tests**: Response times, concurrent connections, resource usage
- **BEV Integration**: Database connectivity, Redis cache, network connectivity, data flow
- **End-to-End Scenarios**: Complete user workflows, load balancer behavior, session persistence
- **Monitoring Health**: Container health, HAProxy backends, log generation, metrics

**Usage**:
```bash
./03-integration-test-automation.sh
```

**Outputs**:
- HTML and JSON test reports
- CSV detailed test results
- Performance metrics
- Integration validation status

### 4. Rollback Automation (`04-rollback-automation.sh`)
**Purpose**: Emergency recovery and rollback procedures

**Rollback Strategies**:
- **Graceful**: Safe shutdown and configuration restore (default)
- **Immediate**: Fast forced rollback with minimal downtime
- **Nuclear**: Complete system reset (emergency use only)
- **Selective**: Component-specific rollback

**Features**:
- Backup discovery and validation
- Multiple recovery strategies
- Emergency alert system
- Post-rollback health checks
- Recovery verification
- Interactive and automated modes

**Usage**:
```bash
# Interactive mode
./04-rollback-automation.sh

# Specific strategy
./04-rollback-automation.sh graceful

# Emergency nuclear rollback
./04-rollback-automation.sh nuclear --force

# List available backups
./04-rollback-automation.sh --list

# Check system status
./04-rollback-automation.sh --status
```

### 5. Health Monitoring Setup (`05-health-monitoring-setup.sh`)
**Purpose**: Comprehensive monitoring and alerting infrastructure

**Monitoring Stack**:
- **Prometheus**: Metrics collection and alerting rules (Port: 9090)
- **Grafana**: Visualization and dashboards (Port: 3001)
- **AlertManager**: Alert routing and notifications (Port: 9093)
- **Node Exporter**: System metrics (Port: 9100)
- **Blackbox Exporter**: Endpoint monitoring (Port: 9115)
- **cAdvisor**: Container metrics (Port: 8080)
- **PostgreSQL Exporter**: Database metrics (Port: 9187)
- **Redis Exporter**: Cache metrics (Port: 9121)

**Key Features**:
- Service availability monitoring
- Performance metrics collection
- Resource utilization tracking
- Alert routing with multiple channels
- Custom dashboards for BEV services
- Integration health monitoring

**Usage**:
```bash
./05-health-monitoring-setup.sh
```

## Complete Deployment Workflow

### Step 1: Pre-Deployment Validation
```bash
cd /home/starlord/Projects/Bev/deployment/automation
./01-pre-deployment-validation.sh
```

### Step 2: Frontend Deployment
```bash
./02-frontend-deployment-safe.sh
```

### Step 3: Integration Testing
```bash
./03-integration-test-automation.sh
```

### Step 4: Monitoring Setup
```bash
./05-health-monitoring-setup.sh
```

### Step 5: Verify Complete Deployment
```bash
# Check all services
docker ps | grep bev

# Access services
curl -k https://localhost:8443/health
curl http://localhost:3011/health
curl http://localhost:9090/targets

# View monitoring
open http://localhost:3001  # Grafana (admin/bevgrafana2024)
open http://localhost:9090  # Prometheus
open http://localhost:9093  # AlertManager
```

## Network Architecture

### Network Segmentation
- **BEV OSINT Network**: 172.30.0.0/16 (existing)
- **BEV Frontend Network**: 172.31.0.0/16 (new)
- **BEV Monitoring Network**: 172.32.0.0/16 (monitoring)
- **BEV Bridge Network**: Connects frontend to existing services

### Service Discovery
All services are accessible via DNS names within their respective networks and through the bridge network for cross-network communication.

## Security Features

### SSL/TLS Configuration
- Self-signed certificates for development
- Production-ready certificate management
- HTTPS enforcement with HTTP redirects
- Security headers (HSTS, X-Frame-Options, etc.)

### Network Security
- Network isolation and segmentation
- Port conflict prevention
- Service-specific firewall rules
- Container security best practices

### Access Control
- Service authentication
- API key management
- Rate limiting
- Input validation

## Monitoring and Alerting

### Alert Channels
- Webhook notifications
- Slack integration
- Email alerts
- System logging

### Key Metrics
- Service availability (uptime/downtime)
- Response times and latency
- Error rates and status codes
- Resource utilization (CPU, memory, disk)
- Database performance
- Network traffic and connectivity

### Dashboard Categories
- **BEV Frontend Overview**: Service status and performance
- **System Monitoring**: Infrastructure metrics
- **Database Performance**: PostgreSQL metrics
- **Security Monitoring**: SSL status, failed requests

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   - Solution: Run pre-deployment validation
   - Check: `netstat -tuln | grep ":3010\|:8443\|:3011"`

2. **Network Issues**
   - Solution: Verify Docker networks
   - Check: `docker network ls | grep bev`

3. **SSL Certificate Issues**
   - Solution: Regenerate certificates
   - Check: `openssl x509 -in config/ssl/bev-frontend.crt -text -noout`

4. **Service Health Issues**
   - Solution: Check container logs
   - Check: `docker logs bev-mcp-server`

### Log Locations
- Deployment logs: `/home/starlord/Projects/Bev/logs/deployment/`
- Test reports: `/home/starlord/Projects/Bev/logs/test-results/`
- Monitoring data: `/home/starlord/Projects/Bev/monitoring/`

### Emergency Procedures

1. **Complete Rollback**
   ```bash
   ./04-rollback-automation.sh nuclear --force
   ```

2. **Service Restart**
   ```bash
   cd frontend && docker-compose -f docker-compose.frontend.yml restart
   ```

3. **Health Check**
   ```bash
   ./04-rollback-automation.sh --status
   monitoring/monitoring-status.sh
   ```

## File Structure

```
deployment/automation/
├── 01-pre-deployment-validation.sh    # Prerequisites validation
├── 02-frontend-deployment-safe.sh     # Conflict-free deployment
├── 03-integration-test-automation.sh  # Comprehensive testing
├── 04-rollback-automation.sh          # Emergency recovery
├── 05-health-monitoring-setup.sh      # Monitoring setup
└── README.md                          # This documentation

frontend/
├── docker-compose.frontend.yml        # Frontend services
├── mcp-server/                        # MCP server implementation
├── proxy/                            # HAProxy configuration
└── websocket-server/                 # WebSocket server

monitoring/
├── docker-compose.monitoring.yml     # Monitoring stack
├── prometheus/                       # Prometheus config
├── grafana/                         # Grafana dashboards
├── alertmanager/                    # Alert configuration
└── README.md                        # Monitoring documentation
```

## Environment Variables

### Required Variables
```bash
# Database
POSTGRES_USER=bev_admin
POSTGRES_PASSWORD=BevSecureDB2024!
REDIS_PASSWORD=BevCache2024!

# Authentication
BEV_API_KEY=<generated>
MCP_API_KEY=<generated>
JWT_SECRET=<generated>
FRONTEND_SESSION_SECRET=<generated>
WEBSOCKET_SECRET=<generated>

# Monitoring
GRAFANA_ADMIN_PASSWORD=bevgrafana2024
HAPROXY_STATS_PASSWORD=bevstats2024

# Alerts (optional)
ALERT_WEBHOOK_URL=<webhook_url>
SLACK_WEBHOOK_URL=<slack_webhook>
ALERT_EMAIL=<alert_email>
```

## DevOps Best Practices

### Infrastructure as Code
- All configurations version controlled
- Idempotent operations
- Infrastructure reproducibility
- Environment consistency

### CI/CD Integration
- Automated validation gates
- Rollback capabilities
- Health checks at each stage
- Comprehensive testing

### Monitoring and Observability
- Comprehensive metrics collection
- Real-time alerting
- Performance monitoring
- Security monitoring

### Security
- Least privilege access
- Network segmentation
- SSL/TLS encryption
- Input validation
- Security scanning

## Support and Maintenance

### Regular Tasks
- Monitor system health via Grafana dashboards
- Review alert notifications
- Check log files for errors
- Validate backup integrity
- Update security certificates

### Updates and Patches
- Test updates in staging environment
- Use rollback procedures for failed updates
- Monitor system after updates
- Document changes

### Performance Optimization
- Monitor resource usage trends
- Optimize based on metrics
- Scale resources as needed
- Tune alert thresholds

---

**Generated**: 2025-01-19  
**Version**: 1.0.0  
**Framework**: DevOps Automation Suite  
**Project**: BEV OSINT Frontend Integration