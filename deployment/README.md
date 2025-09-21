# BEV OSINT Framework - Distributed Deployment

## 🚀 Overview

This directory contains the complete distributed deployment system for the BEV OSINT Framework. The system is designed to deploy across multiple nodes for horizontal scaling, high availability, and geographic distribution.

## 📁 Directory Structure

```
deployment/
├── README.md                           # This file
├── bootstrap/
│   └── node-bootstrap.sh              # Universal node deployment script
├── orchestration/
│   ├── cluster-coordinator.py         # Cluster orchestration service
│   ├── deploy-cluster.sh              # Cluster deployment automation
│   └── validate-cluster.sh            # Cluster validation script
├── shared/
│   ├── network-config.yml             # Network configuration templates
│   ├── secrets-template.yml           # Secrets management template
│   └── health-checks.sh               # Shared health check utilities
└── node-configs/
    ├── data-core/                      # Core data storage services
    ├── data-analytics/                 # Analytics databases
    ├── message-infrastructure/         # Message queuing systems
    ├── processing-core/                # Core OSINT processing
    ├── specialized-processing/         # Specialized analyzers
    ├── infrastructure-monitor/         # Monitoring and infrastructure
    ├── ml-intelligence/                # Machine learning services
    ├── frontend-api/                   # User interfaces and APIs
    └── edge-computing/                 # Geographic edge nodes
```

## 🏗️ Architecture Overview

### Node Types and Requirements

| Node Type | Purpose | RAM | CPU | Storage | Services |
|-----------|---------|-----|-----|---------|----------|
| **data-core** | Core data storage | 32+ GB | 8+ cores | 500+ GB SSD | PostgreSQL, Neo4j, Redis, InfluxDB |
| **data-analytics** | Analytics databases | 16+ GB | 8+ cores | 200+ GB SSD | Elasticsearch, Qdrant, Weaviate |
| **message-infrastructure** | Message queuing | 8+ GB | 4+ cores | 100+ GB | RabbitMQ, Kafka, Zookeeper |
| **processing-core** | Core OSINT processing | 16+ GB | 8+ cores | 100+ GB | IntelOwl, MCP Server, Cytoscape |
| **specialized-processing** | Specialized analyzers | 8-16 GB | 4-8 cores | 50+ GB | Custom analyzers, Intel services |
| **infrastructure-monitor** | Monitoring services | 8+ GB | 4+ cores | 100+ GB | Prometheus, Grafana, Tor proxy |
| **ml-intelligence** | ML and AI services | 16+ GB | 8+ cores | 100+ GB | Autonomous coordinator, ML models |
| **frontend-api** | User interfaces | 4-8 GB | 4+ cores | 50+ GB | Web interfaces, API gateways |
| **edge-computing** | Geographic distribution | 4-8 GB | 4+ cores | 50+ GB | Edge nodes, caching, geo-routing |

### Deployment Dependencies

```
1. data-core              ← Foundation layer
2. data-analytics         ← Depends on data-core
3. message-infrastructure ← Independent
4. infrastructure-monitor ← Independent  
5. processing-core        ← Depends on data-core + message-infrastructure
6. specialized-processing ← Depends on processing-core
7. ml-intelligence        ← Depends on data layers
8. frontend-api           ← Depends on processing layers
9. edge-computing         ← Depends on all layers
```

## 🚀 Quick Deployment Methods

### Method 1: Single Node Bootstrap (Fastest)

Deploy any node type on a single machine:

```bash
# Download and run bootstrap script
curl -sSL https://raw.githubusercontent.com/aahmed954/Bev-AI/main/deployment/bootstrap/node-bootstrap.sh | bash

# Or with specific node type
curl -sSL https://raw.githubusercontent.com/aahmed954/Bev-AI/main/deployment/bootstrap/node-bootstrap.sh | bash -s -- --node-type data-core
```

### Method 2: Interactive Cluster Deployment

Deploy across multiple machines with interactive configuration:

```bash
# Clone repository
git clone https://github.com/aahmed954/Bev-AI.git
cd Bev-AI/deployment/orchestration

# Run interactive cluster deployment
./deploy-cluster.sh --interactive
```

### Method 3: Configuration-Based Deployment

Deploy using a configuration file:

```bash
# Create cluster configuration
cat > cluster-config.yml << EOF
cluster:
  name: "bev-production"
  coordinator:
    host: "10.0.1.100"
    port: 8080

nodes:
  - name: "data-core-01"
    type: "data-core"
    host: "10.0.1.10"
    ssh_user: "ubuntu"
    ssh_key: "~/.ssh/bev-key.pem"
    
  - name: "processing-01"
    type: "processing-core"
    host: "10.0.1.20"
    depends_on: ["data-core-01"]
EOF

# Deploy cluster
./deploy-cluster.sh --config cluster-config.yml
```

## 🔧 Manual Node Deployment

For manual control or custom configurations:

### 1. Clone Repository on Target Node

```bash
git clone https://github.com/aahmed954/Bev-AI.git
cd Bev-AI/deployment/node-configs/[NODE_TYPE]
```

### 2. Configure Environment

```bash
# Copy and edit environment template
cp .env.template .env
nano .env  # Configure with your specific settings
```

### 3. Deploy Services

```bash
# Make deploy script executable
chmod +x deploy.sh

# Run deployment
./deploy.sh

# Check status
docker compose ps
```

## 🌐 Network Configuration

### Inter-Node Communication

The distributed deployment requires network connectivity between nodes:

- **Data Core**: Must be accessible from all other nodes
- **Message Infrastructure**: Must be accessible from processing nodes
- **Coordinator**: Must be accessible from all nodes during deployment

### Firewall Rules

Configure these ports for inter-node communication:

```bash
# Data Core Node
5432/tcp    # PostgreSQL
7474/tcp    # Neo4j HTTP
7687/tcp    # Neo4j Bolt
7001-7003/tcp # Redis Cluster
8086/tcp    # InfluxDB

# Message Infrastructure Node
5672/tcp    # RabbitMQ AMQP
15672/tcp   # RabbitMQ Management
9092/tcp    # Kafka
2181/tcp    # Zookeeper

# Processing Core Node
80/tcp      # IntelOwl HTTP
443/tcp     # IntelOwl HTTPS
3010/tcp    # MCP Server
3000/tcp    # Cytoscape

# Infrastructure Monitor Node
9090/tcp    # Prometheus
3000/tcp    # Grafana
9050/tcp    # Tor SOCKS5
```

## 🔒 Security Configuration

### Inter-Node Authentication

1. **SSH Key Setup**: Configure SSH keys for deployment automation
2. **TLS Certificates**: Generate certificates for service-to-service communication
3. **API Tokens**: Configure authentication tokens for inter-service communication
4. **Network Isolation**: Use VPNs or private networks for node communication

### Environment Security

```bash
# Generate secure passwords
openssl rand -base64 32  # For database passwords
openssl rand -hex 16     # For API keys

# Create certificate authority for cluster
openssl genrsa -out ca.key 4096
openssl req -new -x509 -days 365 -key ca.key -out ca.crt

# Generate node certificates
openssl genrsa -out node.key 2048
openssl req -new -key node.key -out node.csr
openssl x509 -req -days 365 -in node.csr -CA ca.crt -CAkey ca.key -out node.crt
```

## 📊 Monitoring and Management

### Cluster Coordinator

The cluster coordinator provides centralized management:

```bash
# Start coordinator
cd deployment/orchestration
python3 cluster-coordinator.py

# API endpoints
curl http://coordinator:8080/status      # Cluster status
curl http://coordinator:8080/nodes       # List all nodes
curl http://coordinator:8080/discovery/postgres  # Service discovery
```

### Health Monitoring

Monitor cluster health across all nodes:

```bash
# Check individual node
ssh user@node "cd /opt/bev-osint/instances/[NODE_ID] && docker compose ps"

# Check all nodes via coordinator
curl http://coordinator:8080/status | jq '.nodes_by_type'

# Grafana dashboards
# Access: http://[infrastructure-monitor-node]:3000
```

## 🚨 Troubleshooting

### Common Issues

1. **Node Registration Fails**
   ```bash
   # Check coordinator connectivity
   curl http://coordinator:8080/health
   
   # Check node logs
   docker compose logs -f
   ```

2. **Database Connection Issues**
   ```bash
   # Test PostgreSQL connectivity
   docker exec -it bev_postgres pg_isready -U researcher
   
   # Test Redis cluster
   docker exec -it bev_redis_1 redis-cli -p 7001 -a [password] cluster nodes
   ```

3. **Service Discovery Problems**
   ```bash
   # Check service registration
   curl http://coordinator:8080/discovery/postgres
   
   # Verify network connectivity
   telnet data-core-node 5432
   ```

### Debugging Commands

```bash
# View all container logs
docker compose logs -f

# Check resource usage
docker stats

# Test service connectivity
docker exec -it [container] /bin/bash

# Check cluster coordinator logs
tail -f /opt/bev-osint/deployment/orchestration/coordinator.log
```

## 📈 Scaling and Performance

### Horizontal Scaling

Add more nodes of specific types:

```bash
# Add additional processing nodes
./node-bootstrap.sh --node-type specialized-processing --node-id proc-02

# Add geographic edge nodes
./node-bootstrap.sh --node-type edge-computing --node-id edge-eu-01
```

### Performance Optimization

1. **Database Tuning**: Adjust PostgreSQL and Redis configurations based on workload
2. **Load Balancing**: Deploy multiple processing nodes behind load balancers
3. **Caching**: Configure Redis cluster with appropriate memory allocation
4. **Network**: Use high-bandwidth connections between data and processing nodes

## 📚 Additional Resources

- **[API Reference](../docs/API_REFERENCE.md)**: Complete API documentation
- **[Architecture Guide](../docs/comprehensive/BEV_COMPONENT_CATALOG.md)**: Detailed system architecture
- **[Security Guide](../docs/comprehensive/BEV_OPERATIONAL_SECURITY.md)**: Security best practices
- **[Troubleshooting Guide](../docs/comprehensive/BEV_TROUBLESHOOTING_GUIDE.md)**: Comprehensive troubleshooting

## 🎯 Production Deployment Checklist

### Pre-Deployment
- [ ] Network infrastructure configured
- [ ] SSH keys distributed to all nodes
- [ ] Firewall rules configured
- [ ] TLS certificates generated
- [ ] Environment variables configured
- [ ] Backup storage configured

### Deployment
- [ ] Deploy data-core nodes first
- [ ] Validate database connectivity
- [ ] Deploy message-infrastructure
- [ ] Deploy processing-core
- [ ] Deploy remaining node types
- [ ] Validate cluster health

### Post-Deployment
- [ ] Configure monitoring alerts
- [ ] Test backup procedures
- [ ] Validate security configuration
- [ ] Document node inventory
- [ ] Train operational team

---

## 🔗 Quick Links

- **Single Node**: `curl -sSL https://raw.githubusercontent.com/aahmed954/Bev-AI/main/deployment/bootstrap/node-bootstrap.sh | bash`
- **Cluster Deployment**: `./deployment/orchestration/deploy-cluster.sh --interactive`
- **Repository**: https://github.com/aahmed954/Bev-AI
- **Documentation**: [/docs/README.md](../docs/README.md)

**⚠️ Security Notice**: This framework is designed for authorized security research and OSINT operations only. Ensure compliance with applicable laws and regulations.