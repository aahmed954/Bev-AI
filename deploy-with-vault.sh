#!/bin/bash
# Complete deployment with Vault integration

set -e

echo "🔐 BEV Deployment with Vault Integration"
echo "========================================"

# Load secure credentials
source .env.secure

# Deploy to THANOS
echo "Deploying to THANOS with Vault..."
scp .env.secure starlord@thanos:/tmp/.env.secure
scp .env.thanos starlord@thanos:/tmp/.env.thanos
scp deploy-vault-thanos.sh starlord@thanos:/tmp/

ssh starlord@thanos << 'THANOS_DEPLOY'
set -e
cd /opt

# Update repository
if [ -d "bev" ]; then
    cd bev && git pull
else
    git clone https://github.com/aahmed954/Bev-AI.git bev
    cd bev
fi

# Copy environment files
cp /tmp/.env.secure .env.secure
cp /tmp/.env.thanos .env
source .env.secure

# Deploy Vault first
echo "Starting Vault..."
docker stop vault 2>/dev/null || true
docker rm vault 2>/dev/null || true

docker run -d \
    --name vault \
    --restart always \
    -p 8200:8200 \
    -e "VAULT_DEV=1" \
    -e "VAULT_DEV_ROOT_TOKEN_ID=${VAULT_DEV_ROOT_TOKEN}" \
    -e "VAULT_DEV_LISTEN_ADDRESS=0.0.0.0:8200" \
    --cap-add IPC_LOCK \
    hashicorp/vault:latest

sleep 5

# Load secrets into Vault
export VAULT_ADDR='http://localhost:8200'
export VAULT_TOKEN="${VAULT_DEV_ROOT_TOKEN}"

docker exec vault vault secrets enable -path=secret kv-v2 || true
docker exec vault vault kv put secret/database \
    postgres_password="${POSTGRES_PASSWORD}" \
    neo4j_password="${NEO4J_PASSWORD}" \
    redis_password="${REDIS_PASSWORD}"

# Start core services with Vault integration
docker-compose -f docker-compose-thanos-unified.yml up -d \
    postgres neo4j redis_1 elasticsearch kafka_1

echo "THANOS deployment complete with Vault!"
THANOS_DEPLOY

# Deploy to ORACLE1
echo "Deploying to ORACLE1..."
scp .env.oracle1 starlord@oracle1:/tmp/.env.oracle1

ssh starlord@oracle1 << 'ORACLE1_DEPLOY'
set -e
cd /opt

# Update repository
if [ -d "bev" ]; then
    cd bev && git pull
else
    git clone https://github.com/aahmed954/Bev-AI.git bev
    cd bev
fi

# Copy environment file
cp /tmp/.env.oracle1 .env

# Start monitoring services
docker-compose -f docker-compose-oracle1-unified.yml up -d \
    redis_oracle prometheus grafana

echo "ORACLE1 deployment complete!"
ORACLE1_DEPLOY

echo "✅ Deployment complete with Vault integration!"
echo ""
echo "Vault UI: http://thanos:8200"
echo "Vault Token: Check .env.secure file"
echo ""
echo "Services:"
echo "  PostgreSQL: thanos:5432"
echo "  Neo4j: http://thanos:7474"
echo "  Grafana: http://oracle1:3000"
