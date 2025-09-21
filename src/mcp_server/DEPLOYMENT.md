# BEV OSINT MCP Server - Deployment Guide

Comprehensive deployment guide for the BEV OSINT MCP Server with security hardening and production configurations.

## 📋 Prerequisites

### System Requirements

- **CPU**: Minimum 2 cores, Recommended 4+ cores
- **Memory**: Minimum 4GB RAM, Recommended 8GB+ RAM
- **Storage**: Minimum 20GB, Recommended 100GB+ SSD
- **Network**: Stable internet connection, access to BEV infrastructure

### Software Dependencies

- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Python**: 3.12+ (for development)
- **Git**: Latest version

### Access Requirements

- Access to BEV infrastructure network (172.30.0.0/16)
- Database credentials for PostgreSQL, Neo4j, Redis, Elasticsearch
- SSL certificates (for production HTTPS/WSS)

## 🏗 Deployment Architecture

### Network Topology

```
                    Internet/External Networks
                              │
                              ▼
                    ┌─────────────────┐
                    │  Load Balancer  │
                    │   (HAProxy/     │
                    │     Nginx)      │
                    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   Firewall/     │
                    │   Security      │
                    │     Layer       │
                    └─────────────────┘
                              │
                   BEV Network │ 172.30.0.0/16
                              ▼
    ┌─────────────────┬─────────────────┬─────────────────┐
    │                 │                 │                 │
    ▼                 ▼                 ▼                 ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ MCP Server  │ │ PostgreSQL  │ │    Neo4j    │ │    Redis    │
│   :3010     │ │ 172.21.0.2  │ │ 172.21.0.3  │ │ 172.21.0.4  │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────┐
                                               │Elasticsearch│
                                               │ 172.21.0.5  │
                                               └─────────────┘
```

### Service Dependencies

```mermaid
graph TD
    A[MCP Server] --> B[PostgreSQL]
    A --> C[Neo4j]
    A --> D[Redis]
    A --> E[Elasticsearch]
    
    F[Prometheus] --> A
    G[Grafana] --> F
    
    H[Load Balancer] --> A
    I[Firewall] --> H
```

## 🚀 Quick Deployment

### Option 1: Docker Compose (Recommended)

1. **Clone and Navigate:**
```bash
git clone <repository-url>
cd bev-osint-framework/src/mcp_server
```

2. **Configure Environment:**
```bash
cp .env.example .env
# Edit .env with your production values
```

3. **Deploy Services:**
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f mcp-server
```

### Option 2: Standalone Docker

1. **Build Image:**
```bash
docker build -t bev-mcp-server:latest .
```

2. **Run Container:**
```bash
docker run -d \
  --name bev-mcp-server \
  --restart unless-stopped \
  -p 3010:3010 \
  --env-file .env \
  bev-mcp-server:latest
```

## 🔧 Configuration

### Environment Variables

Create `.env` file with production values:

```bash
# ===========================
# Server Configuration
# ===========================
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=3010
ENVIRONMENT=production

# ===========================
# Security Configuration
# ===========================
JWT_SECRET=<generate_secure_32_char_key>
DATA_ENCRYPTION_KEY=<generate_secure_32_char_key>
API_RATE_LIMIT=100
API_RATE_WINDOW=60
ALLOWED_NETWORKS=172.30.0.0/16,10.0.0.0/8
PASSWORD_MIN_LENGTH=12
SESSION_TIMEOUT=3600
AUDIT_LOG_RETENTION_DAYS=90

# ===========================
# Database Configuration
# ===========================
POSTGRES_URI=postgresql://bev_admin:${POSTGRES_PASSWORD}@172.21.0.2:5432/osint
NEO4J_URI=bolt://172.21.0.3:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=${NEO4J_PASSWORD}
REDIS_HOST=172.21.0.4
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}
ELASTICSEARCH_HOST=172.21.0.5
ELASTICSEARCH_PORT=9200

# ===========================
# Connection Pool Settings
# ===========================
POSTGRES_MIN_CONNECTIONS=10
POSTGRES_MAX_CONNECTIONS=50
REDIS_MAX_CONNECTIONS=50
CONNECTION_TIMEOUT=30
QUERY_TIMEOUT=300

# ===========================
# Logging and Monitoring
# ===========================
LOG_LEVEL=INFO
PROMETHEUS_ENABLED=true

# ===========================
# TLS/SSL Configuration
# ===========================
TLS_ENABLED=true
TLS_CERT_PATH=/app/certs/server.crt
TLS_KEY_PATH=/app/certs/server.key
```

### Security Hardening

1. **Generate Secure Keys:**
```bash
# Generate JWT secret
openssl rand -base64 32

# Generate encryption key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

2. **SSL/TLS Certificates:**
```bash
# Create certificate directory
mkdir -p certs/

# Generate self-signed certificate (for testing)
openssl req -x509 -newkey rsa:4096 -keyout certs/server.key -out certs/server.crt -days 365 -nodes

# For production, use proper CA-signed certificates
```

3. **File Permissions:**
```bash
# Secure certificate files
chmod 600 certs/server.key
chmod 644 certs/server.crt
chown root:root certs/*
```

## 🔒 Production Security Setup

### Firewall Configuration

```bash
# UFW (Ubuntu)
sudo ufw allow from 172.30.0.0/16 to any port 3010
sudo ufw allow from <monitoring_network> to any port 9090
sudo ufw deny 3010
sudo ufw enable

# iptables (manual)
iptables -A INPUT -s 172.30.0.0/16 -p tcp --dport 3010 -j ACCEPT
iptables -A INPUT -p tcp --dport 3010 -j DROP
```

### Network Security

1. **Reverse Proxy (Nginx):**
```nginx
upstream mcp_backend {
    server 127.0.0.1:3010;
    # Add more servers for load balancing
}

server {
    listen 443 ssl;
    server_name mcp.bev-osint.local;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=mcp:10m rate=10r/s;
    limit_req zone=mcp burst=20 nodelay;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    location / {
        proxy_pass http://mcp_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }
}
```

2. **Load Balancing (HAProxy):**
```
global
    daemon
    maxconn 4096
    
defaults
    mode http
    timeout connect 5s
    timeout client 30s
    timeout server 30s
    
frontend mcp_frontend
    bind *:443 ssl crt /path/to/cert.pem
    redirect scheme https if !{ ssl_fc }
    default_backend mcp_servers
    
backend mcp_servers
    balance roundrobin
    option httpchk GET /health
    server mcp1 127.0.0.1:3010 check
    server mcp2 127.0.0.1:3011 check
```

### Container Security

1. **Docker Security Options:**
```yaml
# docker-compose.yml security section
security_opt:
  - no-new-privileges:true
  - seccomp:unconfined
read_only: true
tmpfs:
  - /tmp:noexec,nosuid,size=100m
  - /var/run:noexec,nosuid,size=100m
user: "1000:1000"
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE
```

2. **Resource Limits:**
```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

## 📊 Monitoring Setup

### Prometheus Configuration

1. **Prometheus Config (`prometheus.yml`):**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mcp-server'
    static_configs:
      - targets: ['mcp-server:3010']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'docker'
    static_configs:
      - targets: ['localhost:9323']
```

2. **Grafana Dashboard:**
```json
{
  "dashboard": {
    "id": null,
    "title": "BEV MCP Server",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(mcp_requests_total[5m])",
            "legendFormat": "{{method}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(mcp_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

### Log Management

1. **Centralized Logging (ELK Stack):**
```yaml
# docker-compose.yml logging section
logging:
  driver: "fluentd"
  options:
    fluentd-address: "localhost:24224"
    tag: "docker.mcp-server"
```

2. **Log Rotation:**
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "5"
```

## 🔄 High Availability Setup

### Multi-Instance Deployment

1. **Docker Swarm Mode:**
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml bev-mcp

# Scale service
docker service scale bev-mcp_mcp-server=3
```

2. **Kubernetes Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: bev-mcp-server:latest
        ports:
        - containerPort: 3010
        env:
        - name: MCP_SERVER_PORT
          value: "3010"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Database High Availability

1. **PostgreSQL Cluster:**
```bash
# Using PostgreSQL replication
# Primary-Replica setup with automatic failover
```

2. **Redis Cluster:**
```bash
# Redis Cluster or Sentinel for high availability
```

## 🔧 Maintenance

### Regular Maintenance Tasks

1. **Database Maintenance:**
```sql
-- PostgreSQL maintenance
VACUUM ANALYZE;
REINDEX DATABASE osint;

-- Clean old audit logs
SELECT cleanup_old_data();
```

2. **Log Rotation:**
```bash
# Rotate application logs
logrotate /etc/logrotate.d/mcp-server

# Clean Docker logs
docker system prune -f
```

3. **Security Updates:**
```bash
# Update base images
docker pull python:3.12-slim
docker-compose build --no-cache

# Update dependencies
pip install --upgrade -r requirements.txt
```

### Backup Strategy

1. **Database Backups:**
```bash
#!/bin/bash
# backup.sh

# PostgreSQL backup
pg_dump -h 172.21.0.2 -U bev_admin osint > /backup/osint_$(date +%Y%m%d_%H%M%S).sql

# Redis backup
redis-cli -h 172.21.0.4 --rdb /backup/redis_$(date +%Y%m%d_%H%M%S).rdb

# Neo4j backup
docker exec neo4j neo4j-admin dump --database=neo4j --to=/backup/neo4j_$(date +%Y%m%d_%H%M%S).dump
```

2. **Configuration Backups:**
```bash
# Backup configuration files
tar -czf /backup/config_$(date +%Y%m%d_%H%M%S).tar.gz /app/config/ .env
```

## 🐛 Troubleshooting

### Common Issues

1. **Connection Issues:**
```bash
# Check network connectivity
docker exec mcp-server ping 172.21.0.2

# Check database connections
docker exec mcp-server python -c "import asyncpg; print('OK')"
```

2. **Performance Issues:**
```bash
# Check resource usage
docker stats mcp-server

# Check database performance
docker exec postgres pg_stat_activity
```

3. **Authentication Issues:**
```bash
# Verify JWT configuration
docker exec mcp-server python -c "import jwt; print(jwt.__version__)"

# Check token validation
curl -H "Authorization: Bearer <token>" http://localhost:3010/health
```

### Debug Mode

Enable debug logging:

```bash
# In .env file
LOG_LEVEL=DEBUG

# Or temporary override
docker-compose restart mcp-server
docker-compose logs -f mcp-server
```

### Health Checks

```bash
# Server health
curl http://localhost:3010/health

# Database connectivity
curl http://localhost:3010/health | jq '.services'

# Metrics endpoint
curl http://localhost:3010/metrics
```

## 📊 Performance Tuning

### Database Optimization

1. **PostgreSQL Tuning:**
```sql
-- postgresql.conf optimizations
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
```

2. **Redis Optimization:**
```bash
# redis.conf optimizations
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Application Tuning

1. **Connection Pool Sizing:**
```bash
# Adjust based on load
POSTGRES_MAX_CONNECTIONS=50
REDIS_MAX_CONNECTIONS=50
```

2. **Rate Limiting:**
```bash
# Adjust based on capacity
API_RATE_LIMIT=200
API_RATE_WINDOW=60
```

## 🔄 Rollback Procedures

### Quick Rollback

1. **Docker Compose:**
```bash
# Stop current version
docker-compose down

# Use previous image
docker tag bev-mcp-server:previous bev-mcp-server:latest

# Start services
docker-compose up -d
```

2. **Database Rollback:**
```bash
# Restore from backup
pg_restore -h 172.21.0.2 -U bev_admin -d osint /backup/osint_backup.sql
```

### Zero-Downtime Deployment

1. **Blue-Green Deployment:**
```bash
# Deploy to green environment
docker-compose -f docker-compose.green.yml up -d

# Switch traffic
# Update load balancer configuration

# Stop blue environment
docker-compose -f docker-compose.blue.yml down
```

This deployment guide provides comprehensive instructions for production deployment of the BEV OSINT MCP Server with proper security, monitoring, and maintenance procedures.