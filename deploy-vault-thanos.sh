#!/bin/bash
# Deploy Vault on THANOS node

set -e
echo "Deploying Vault on THANOS..."

# Start Vault in dev mode initially (will be replaced with production config later)
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

# Export for local use
export VAULT_ADDR='http://localhost:8200'
export VAULT_TOKEN="${VAULT_DEV_ROOT_TOKEN}"

# Enable KV secrets engine
docker exec vault vault secrets enable -path=secret kv-v2 || true

# Load database passwords
docker exec vault vault kv put secret/database \
    postgres_password="${POSTGRES_PASSWORD}" \
    neo4j_password="${NEO4J_PASSWORD}" \
    redis_password="${REDIS_PASSWORD}"

# Load service passwords
docker exec vault vault kv put secret/services \
    rabbitmq_password="${RABBITMQ_PASSWORD}" \
    kafka_password="${KAFKA_PASSWORD}"

# Load encryption keys
docker exec vault vault kv put secret/encryption \
    encryption_key="${ENCRYPTION_KEY}" \
    jwt_secret="${JWT_SECRET}" \
    session_secret="${SESSION_SECRET}"

echo "Vault deployed and configured on THANOS"
