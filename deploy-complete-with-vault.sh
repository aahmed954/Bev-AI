#!/bin/bash
# BEV COMPLETE MULTINODE DEPLOYMENT WITH VAULT
# Deploys to THANOS and ORACLE1 with centralized credential management

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🚀 BEV COMPLETE MULTINODE DEPLOYMENT WITH VAULT${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""

# Check for credentials
if [ ! -f ".env.secure" ]; then
    echo -e "${YELLOW}No credentials found, generating...${NC}"
    ./generate-secure-credentials.sh
fi

# Source credentials
source .env.secure

echo "Using Vault Token: ${VAULT_DEV_ROOT_TOKEN:0:8}..."
echo ""

# Step 1: Deploy Vault and core services to THANOS
echo -e "${CYAN}Step 1: Deploying to THANOS (Primary Node)${NC}"

ssh starlord@thanos << THANOS_SCRIPT
set -e
echo "Preparing THANOS deployment..."

# Create directories
sudo mkdir -p /opt/bev
sudo chown starlord:starlord /opt/bev
cd /opt

# Clone or update repository
if [ -d "bev" ]; then
    cd bev && git pull origin main
else
    git clone https://github.com/aahmed954/Bev-AI.git bev
    cd bev
fi

# Deploy Vault first
echo "Starting Vault server..."
docker stop vault 2>/dev/null || true
docker rm vault 2>/dev/null || true

docker run -d \
    --name vault \
    --restart unless-stopped \
    --network host \
    -e "VAULT_DEV=1" \
    -e "VAULT_DEV_ROOT_TOKEN_ID=${VAULT_DEV_ROOT_TOKEN}" \
    -e "VAULT_DEV_LISTEN_ADDRESS=0.0.0.0:8200" \
    --cap-add IPC_LOCK \
    hashicorp/vault:latest

echo "Waiting for Vault to start..."
sleep 10

# Configure Vault
export VAULT_ADDR='http://localhost:8200'
export VAULT_TOKEN="${VAULT_DEV_ROOT_TOKEN}"

# Enable secrets engine and load credentials
docker exec -e VAULT_TOKEN="${VAULT_DEV_ROOT_TOKEN}" vault \
    vault secrets enable -path=secret kv-v2 || true

# Load database credentials
docker exec -e VAULT_TOKEN="${VAULT_DEV_ROOT_TOKEN}" vault \
    vault kv put secret/database \
    postgres_password="${POSTGRES_PASSWORD}" \
    neo4j_password="${NEO4J_PASSWORD}" \
    redis_password="${REDIS_PASSWORD}"

# Load service credentials
docker exec -e VAULT_TOKEN="${VAULT_DEV_ROOT_TOKEN}" vault \
    vault kv put secret/services \
    rabbitmq_password="${RABBITMQ_PASSWORD}" \
    kafka_password="${KAFKA_PASSWORD}"

echo "Vault configured successfully!"

# Create environment file
cat > .env << EOF
NODE_NAME=thanos
VAULT_URL=http://localhost:8200
VAULT_TOKEN=${VAULT_DEV_ROOT_TOKEN}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
NEO4J_PASSWORD=${NEO4J_PASSWORD}
REDIS_PASSWORD=${REDIS_PASSWORD}
RABBITMQ_PASSWORD=${RABBITMQ_PASSWORD}
EOF

# Start core database services
echo "Starting core services..."
docker-compose -f docker-compose-thanos-unified.yml up -d \
    postgres neo4j redis_1 elasticsearch || {
        echo "Note: Some services may need configuration adjustments"
    }

echo "THANOS deployment initiated!"
THANOS_SCRIPT

# Step 2: Deploy monitoring to ORACLE1
echo -e "${CYAN}Step 2: Deploying to ORACLE1 (Monitoring Node)${NC}"

ssh starlord@oracle1 << ORACLE1_SCRIPT
set -e
echo "Preparing ORACLE1 deployment..."

# Create directories
sudo mkdir -p /opt/bev
sudo chown starlord:starlord /opt/bev
cd /opt

# Clone or update repository
if [ -d "bev" ]; then
    cd bev && git pull origin main
else
    git clone https://github.com/aahmed954/Bev-AI.git bev
    cd bev
fi

# Create environment file pointing to THANOS Vault
cat > .env << EOF
NODE_NAME=oracle1
VAULT_URL=http://thanos:8200
VAULT_TOKEN=${VAULT_DEV_ROOT_TOKEN}
REDIS_PASSWORD=${REDIS_PASSWORD}
POSTGRES_HOST=thanos
NEO4J_HOST=thanos
EOF

# Start monitoring services
echo "Starting monitoring services..."
docker-compose -f docker-compose-oracle1-unified.yml up -d \
    prometheus grafana consul || {
        echo "Note: Some services may need configuration adjustments"
    }

echo "ORACLE1 deployment initiated!"
ORACLE1_SCRIPT

# Step 3: Verify deployments
echo -e "${CYAN}Step 3: Verifying Deployments${NC}"

echo -e "${BLUE}Checking Vault on THANOS:${NC}"
ssh starlord@thanos "curl -s http://localhost:8200/v1/sys/health | jq '.'" 2>/dev/null || echo "Vault health check pending..."

echo -e "${BLUE}THANOS Services:${NC}"
ssh starlord@thanos "docker ps --format 'table {{.Names}}\t{{.Status}}' | head -10"

echo -e "${BLUE}ORACLE1 Services:${NC}"
ssh starlord@oracle1 "docker ps --format 'table {{.Names}}\t{{.Status}}' | head -10"

# Summary
echo ""
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE WITH VAULT!${NC}"
echo ""
echo -e "${YELLOW}Access Points:${NC}"
echo "  Vault UI:       http://thanos:8200"
echo "  Vault Token:    ${VAULT_DEV_ROOT_TOKEN:0:12}..."
echo "  PostgreSQL:     thanos:5432 (password in Vault)"
echo "  Neo4j:          http://thanos:7474 (password in Vault)"
echo "  Grafana:        http://oracle1:3000"
echo "  Prometheus:     http://oracle1:9090"
echo ""
echo -e "${CYAN}To retrieve credentials from Vault:${NC}"
echo "  export VAULT_ADDR='http://thanos:8200'"
echo "  export VAULT_TOKEN='${VAULT_DEV_ROOT_TOKEN}'"
echo "  vault kv get secret/database"
echo ""
echo -e "${YELLOW}Monitor logs:${NC}"
echo "  ssh starlord@thanos 'docker logs -f vault'"
echo "  ssh starlord@thanos 'docker-compose -f docker-compose-thanos-unified.yml logs -f'"
echo ""
echo -e "${RED}SECURITY NOTE:${NC}"
echo "  Keep .env.secure file safe - it contains all credentials!"
echo "  The Vault is currently in DEV mode - configure for production use!"
