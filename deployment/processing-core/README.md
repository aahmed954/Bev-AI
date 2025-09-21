# BEV OSINT Framework - Processing Core Node

This directory contains the deployment configuration for the **Processing Core** node of the BEV OSINT Framework. This node handles the core OSINT processing capabilities through the IntelOwl platform and provides advanced visualization and analysis features.

## 🏗️ Architecture Overview

The Processing Core node includes:

- **IntelOwl Platform**: Django application with Celery workers for OSINT analysis
- **Cytoscape Server**: Graph visualization and network analysis
- **MCP Server**: BEV Model Context Protocol server for advanced analysis
- **Nginx Reverse Proxy**: Load balancing and SSL termination

## 📋 Prerequisites

### Required External Nodes

This node requires connections to:

1. **Data Core Node**: PostgreSQL, Redis, Neo4j, Elasticsearch
2. **Message Infrastructure Node**: RabbitMQ cluster

### System Requirements

- **RAM**: 16+ GB (recommended 32+ GB for production)
- **CPU**: 8+ cores (moderate to high CPU usage)
- **Storage**: 50+ GB for logs and temporary files
- **Network**: High-bandwidth connection to data services

### Software Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 50GB free disk space

## 🚀 Quick Start

### 1. Environment Configuration

```bash
# Copy the environment template
cp .env.template .env

# Edit the configuration file
nano .env
```

**Critical Configuration Values:**

```bash
# Data Core Node Connection
DATA_CORE_POSTGRES_HOST=192.168.1.100
DATA_CORE_REDIS_HOST=192.168.1.100
DATA_CORE_NEO4J_URI=bolt://192.168.1.100:7687

# Message Infrastructure Node
MSG_RABBITMQ_HOST=192.168.1.101

# Security (CHANGE THESE!)
DJANGO_SECRET_KEY=your-secure-secret-key
JWT_SECRET=your-jwt-secret
DATA_ENCRYPTION_KEY=your-32-char-encryption-key
```

### 2. Directory Structure Setup

```bash
# Create required directories
mkdir -p logs/{intelowl,nginx,cytoscape,mcp}
mkdir -p ssl
mkdir -p mcp_server/config

# Create custom analyzers and connectors directories
mkdir -p intelowl/{custom_analyzers,custom_connectors}

# Copy cytoscape application if not already present
# (This should be copied from the main BEV repository)
```

### 3. Deployment

```bash
# Start the services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4. Health Checks

```bash
# Check IntelOwl API
curl -f http://localhost:8000/api/health

# Check Cytoscape server
curl -f http://localhost:3000/health

# Check MCP server
curl -f http://localhost:3010/health

# Check Nginx
curl -f http://localhost:80
```

## 🔧 Configuration Details

### Network Configuration

The Processing Core uses the `172.20.0.0/16` subnet for internal communication. Services are configured to connect to external nodes via the specified IP addresses and ports.

### Service Dependencies

```
intelowl-nginx
├── intelowl-django
│   ├── External: PostgreSQL (Data Core)
│   ├── External: Redis (Data Core)
│   └── External: RabbitMQ (Message Infrastructure)
├── cytoscape-server
│   ├── External: Neo4j (Data Core)
│   └── External: PostgreSQL (Data Core)
└── mcp-server
    ├── External: PostgreSQL (Data Core)
    ├── External: Redis (Data Core)
    ├── External: Neo4j (Data Core)
    └── External: Elasticsearch (Data Core)

intelowl-celery-worker
├── External: PostgreSQL (Data Core)
├── External: Redis (Data Core)
└── External: RabbitMQ (Message Infrastructure)

intelowl-celery-beat
├── External: PostgreSQL (Data Core)
├── External: Redis (Data Core)
└── External: RabbitMQ (Message Infrastructure)
```

### Port Mapping

| Service | Internal Port | External Port | Description |
|---------|---------------|---------------|-------------|
| Nginx | 80, 443 | 80, 443 | Web interface |
| IntelOwl Django | 8000 | 8000 | API endpoint |
| Cytoscape Server | 3000 | 3000 | Visualization |
| MCP Server | 3010 | 3010 | Analysis API |

## 🛠️ Customization

### Custom Analyzers and Connectors

Place custom IntelOwl analyzers and connectors in:
- `./intelowl/custom_analyzers/`
- `./intelowl/custom_connectors/`

These directories are mounted as read-only volumes into the IntelOwl containers.

### Nginx Configuration

Modify `./intelowl/nginx.conf` to customize:
- SSL certificates
- Load balancing rules
- Rate limiting
- Security headers

### Resource Limits

Adjust resource limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

## 📊 Monitoring

### Health Checks

All services include health checks:
- **IntelOwl Django**: `/api/health` endpoint
- **Cytoscape**: `/health` endpoint
- **MCP Server**: Custom health check script
- **Nginx**: HTTP response check

### Logs

Logs are centralized in the `./logs/` directory:
- IntelOwl: `./logs/intelowl/`
- Nginx: `./logs/nginx/`
- Cytoscape: `./logs/cytoscape/`
- MCP Server: `./logs/mcp/`

### Metrics

The MCP server exposes Prometheus metrics on port 9090 (if enabled).

## 🔍 Troubleshooting

### Common Issues

**1. Cannot connect to external databases**
```bash
# Check network connectivity
ping <DATA_CORE_HOST>
telnet <DATA_CORE_HOST> 5432

# Verify environment variables
docker-compose config
```

**2. IntelOwl migration failures**
```bash
# Check database connection
docker-compose exec intelowl-django python manage.py dbshell

# Run migrations manually
docker-compose exec intelowl-django python manage.py migrate
```

**3. Celery workers not processing tasks**
```bash
# Check RabbitMQ connection
docker-compose exec intelowl-celery-worker celery -A intel_owl inspect active

# Check worker status
docker-compose logs intelowl-celery-worker
```

**4. High memory usage**
```bash
# Monitor resource usage
docker stats

# Adjust worker configuration
# Reduce CELERY_WORKERS or WORKERS in .env
```

### Debug Mode

Enable debug logging:

```bash
# In .env file
DJANGO_DEBUG=True
LOG_LEVEL=DEBUG

# Restart services
docker-compose restart
```

## 🔒 Security Considerations

### Production Hardening

1. **Change default secrets**: Update all keys and passwords
2. **Disable debug mode**: Set `DJANGO_DEBUG=False`
3. **Enable authentication**: Set security flags to `False`
4. **Use SSL certificates**: Configure HTTPS in Nginx
5. **Network isolation**: Use firewall rules to restrict access
6. **Regular updates**: Keep Docker images updated

### API Keys

Configure OSINT service API keys in `.env`:
- Shodan API Key
- VirusTotal API Key
- DeHashed credentials
- SNUSBase API Key
- WeLeakInfo API Key

### Network Security

The Processing Core should be isolated in a private network with controlled access to:
- Data Core node (database access)
- Message Infrastructure node (RabbitMQ access)
- Internet (for OSINT data collection)

## 📈 Scaling

### Horizontal Scaling

Increase Celery workers:
```bash
# In .env file
CELERY_WORKER_REPLICAS=4
CELERY_WORKERS=16

# Apply changes
docker-compose up -d --scale intelowl-celery-worker=4
```

### Vertical Scaling

Increase resource limits in `docker-compose.yml` and adjust worker counts:
```bash
WORKERS=8
THREADS_PER_WORKER=8
```

## 🔄 Updates and Maintenance

### Updating IntelOwl

```bash
# Update docker-compose.yml with new version
# Then recreate containers
docker-compose pull
docker-compose up -d --force-recreate
```

### Backup Considerations

This node is stateless except for:
- Custom analyzers/connectors
- Configuration files
- SSL certificates

All data is stored in the Data Core node.

## 📞 Support

For issues specific to:
- **IntelOwl**: Check [IntelOwl documentation](https://intelowl.readthedocs.io/)
- **BEV Framework**: Check main repository documentation
- **Docker/Deployment**: Check Docker Compose documentation

### Logs for Support

When reporting issues, include:
```bash
# Service status
docker-compose ps

# Service logs
docker-compose logs --tail=100 <service-name>

# System resources
docker stats --no-stream
```