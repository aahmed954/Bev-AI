# BEV Multi-Node Deployment Guide

## 🚀 Quick Start

**Deploy everything with one command:**
```bash
./deploy_multinode_bev.sh
```

## 📋 Prerequisites

### ✅ Already Configured:
- SSH access to Thanos (100.122.12.54) and Oracle1 (100.96.197.84)
- Tailscale VPN connectivity between all nodes
- Docker installed on all nodes
- Admin access on Starlord (development machine)

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   STARLORD (Control Node)                │
│                     IP: 100.122.12.35                    │
│                                                          │
│  • HashiCorp Vault (Centralized Secrets)                │
│  • Development Environment                               │
│  • Deployment Orchestration                             │
└─────────────────┬───────────────────┬───────────────────┘
                  │                   │
                  │    Tailscale      │
                  │       VPN          │
                  │                   │
    ┌─────────────▼─────────┐   ┌────▼──────────────────┐
    │   THANOS (GPU Node)   │   │  ORACLE1 (ARM Node)   │
    │   IP: 100.122.12.54   │   │   IP: 100.96.197.84   │
    │                       │   │                       │
    │  • PostgreSQL         │   │  • Redis              │
    │  • Neo4j              │   │  • Prometheus         │
    │  • Elasticsearch      │   │  • Grafana            │
    │  • Kafka Cluster      │   │  • Consul             │
    │  • RabbitMQ           │   │  • Tor Proxy          │
    │  • AI/ML Services     │   │  • ARM Analyzers      │
    │  • GPU Processing     │   │  • Monitoring Stack   │
    └───────────────────────┘   └───────────────────────┘
```

## 🔧 Deployment Steps

### 1. Initial Deployment (First Time Only)

```bash
# Navigate to BEV directory
cd /home/starlord/Projects/Bev

# Run the master deployment script
./deploy_multinode_bev.sh
```

This script will:
1. **Setup Vault** on Starlord for centralized credential management
2. **Generate secure passwords** for all services
3. **Load secrets** into Vault
4. **Create authentication tokens** for each node
5. **Deploy services to Thanos** (GPU/compute services)
6. **Deploy services to Oracle1** (ARM/monitoring services)
7. **Verify deployment** across all nodes

### 2. Verify Deployment

```bash
# Run comprehensive verification
./verify_multinode_deployment.sh
```

Expected output:
- ✅ All critical services operational
- ✅ Cross-node connectivity working
- ✅ Databases accessible
- ✅ Monitoring stack running

### 3. Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Vault UI | http://100.122.12.35:8200/ui | Token in `vault-init.json` |
| Neo4j Browser | http://100.122.12.54:7474 | neo4j / (from Vault) |
| Grafana | http://100.96.197.84:3000 | admin / (from Vault) |
| Prometheus | http://100.96.197.84:9090 | No auth |
| Consul | http://100.96.197.84:8500 | No auth |
| RabbitMQ | http://100.122.12.54:15672 | bev / (from Vault) |

## 🔐 Security

### Critical Files to Secure

**After deployment, immediately secure these files:**

```bash
# Move Vault initialization file to secure location
mv vault-init.json ~/.vault/bev-vault-init.json
chmod 600 ~/.vault/bev-vault-init.json

# Remove generated passwords file after Vault is populated
rm .env.secure
```

### Vault Management

**Unseal Vault (if sealed):**
```bash
export VAULT_ADDR="http://localhost:8200"
vault operator unseal $(jq -r '.unseal_keys_b64[0]' ~/.vault/bev-vault-init.json)
vault operator unseal $(jq -r '.unseal_keys_b64[1]' ~/.vault/bev-vault-init.json)
vault operator unseal $(jq -r '.unseal_keys_b64[2]' ~/.vault/bev-vault-init.json)
```

**Access Vault:**
```bash
export VAULT_TOKEN=$(jq -r '.root_token' ~/.vault/bev-vault-init.json)
vault status
```

## 📊 Monitoring

### Check Service Status

**On Thanos:**
```bash
ssh thanos "docker ps --format 'table {{.Names}}\t{{.Status}}'"
```

**On Oracle1:**
```bash
ssh oracle1 "docker ps --format 'table {{.Names}}\t{{.Status}}'"
```

### View Logs

**Real-time logs on Thanos:**
```bash
ssh thanos "cd /opt/bev && docker compose logs -f"
```

**Real-time logs on Oracle1:**
```bash
ssh oracle1 "cd /opt/bev && docker compose logs -f"
```

### Grafana Dashboards

1. Access Grafana: http://100.96.197.84:3000
2. Default dashboards available:
   - System Overview
   - Docker Containers
   - Network Traffic
   - Database Performance
   - AI/ML Pipeline Metrics

## 🛠️ Troubleshooting

### Common Issues

#### 1. Vault is Sealed
```bash
# Check status
vault status

# Unseal if needed (requires 3 keys)
./scripts/unseal_vault.sh
```

#### 2. Service Can't Connect to Database
```bash
# Check if Vault token is valid
vault token lookup

# Regenerate AppRole tokens if expired
vault write -f auth/approle/role/thanos/secret-id
```

#### 3. Cross-Node Connectivity Issues
```bash
# Check Tailscale status
tailscale status

# Test connectivity
ping 100.122.12.54  # Thanos
ping 100.96.197.84  # Oracle1
```

#### 4. Container Restart Loop
```bash
# Check logs for specific container
ssh thanos "docker logs bev_postgres"

# Restart with fresh config
ssh thanos "docker restart bev_postgres"
```

## 🔄 Maintenance

### Daily Tasks
- Check Grafana dashboards for anomalies
- Review Prometheus alerts
- Monitor disk usage on all nodes

### Weekly Tasks
- Rotate AppRole tokens
- Backup Vault data
- Update container images

### Monthly Tasks
- Full system backup
- Security audit
- Performance optimization

## 📝 Service Distribution

### Thanos (GPU Node) - 32 Services
- **Databases**: PostgreSQL, Neo4j, Elasticsearch, InfluxDB
- **Message Queues**: Kafka (3 brokers), RabbitMQ (3 nodes), Zookeeper
- **AI/ML**: Autonomous Coordinator, Adaptive Learning, Knowledge Evolution
- **Processing**: IntelOwl Django, Celery Workers

### Oracle1 (ARM Node) - 25 Services
- **Caching**: Redis
- **Monitoring**: Prometheus, Grafana, Telegraf
- **Service Discovery**: Consul
- **Security**: Tor Proxy, Nginx Proxy Manager
- **Analyzers**: Breach, Crypto, Social (ARM-optimized)

## 🚨 Emergency Procedures

### Complete System Restart
```bash
# Stop all services
ssh thanos "cd /opt/bev && docker compose down"
ssh oracle1 "cd /opt/bev && docker compose down"

# Start Vault first
docker start vault
./scripts/unseal_vault.sh

# Restart services
./deploy_multinode_bev.sh
```

### Disaster Recovery
```bash
# Restore from backup
./scripts/restore_from_backup.sh

# Reinitialize Vault
./scripts/reinit_vault.sh

# Redeploy all services
./deploy_multinode_bev.sh
```

## 📚 Additional Documentation

- [BEV Distribution Strategy](./BEV_DISTRIBUTION_STRATEGY.md)
- [Vault Credential Management](./docs/vault_management.md)
- [Service Architecture](./docs/architecture.md)
- [API Documentation](./docs/api.md)

## 💡 Tips

1. **Always verify Vault is unsealed** before deploying services
2. **Monitor resource usage** on Oracle1 (limited to 24GB RAM)
3. **Use Tailscale IPs** for cross-node communication
4. **Check GPU availability** on Thanos before deploying AI services
5. **Keep vault-init.json secure** - it contains root access to all secrets

## 🆘 Support

If you encounter issues:
1. Check the verification script: `./verify_multinode_deployment.sh`
2. Review service logs: `docker compose logs [service_name]`
3. Check Vault status: `vault status`
4. Verify network connectivity: `tailscale status`

---

**Remember:** This is a distributed system. Services on Oracle1 depend on databases running on Thanos. Always ensure Thanos services are healthy before troubleshooting Oracle1.
