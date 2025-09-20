#!/bin/bash
# SECURE CREDENTIAL GENERATION FOR BEV

set -e

echo "Generating secure credentials..."

# Generate passwords safely
DB_PASS=$(openssl rand -hex 16)
PG_PASS=$(openssl rand -hex 16)
NEO_PASS=$(openssl rand -hex 16)
REDIS_PASS=$(openssl rand -hex 16)
RABBIT_PASS=$(openssl rand -hex 16)
KAFKA_PASS=$(openssl rand -hex 16)
VAULT_TOKEN=$(openssl rand -hex 16)

# Generate keys
ENCRYPT_KEY=$(openssl rand -base64 32 | tr -d '\n')
JWT_KEY=$(openssl rand -base64 32 | tr -d '\n')
SESSION_KEY=$(openssl rand -base64 24 | tr -d '\n')

cat > .env.secure << EOF
# Secure Environment Variables - NEVER COMMIT THIS FILE
# Generated on $(date)

# Database Passwords
export DB_PASSWORD="${DB_PASS}"
export POSTGRES_PASSWORD="${PG_PASS}"
export NEO4J_PASSWORD="${NEO_PASS}"

# Service Passwords
export REDIS_PASSWORD="${REDIS_PASS}"
export RABBITMQ_PASSWORD="${RABBIT_PASS}"
export KAFKA_PASSWORD="${KAFKA_PASS}"

# Encryption Keys
export ENCRYPTION_KEY="${ENCRYPT_KEY}"
export JWT_SECRET="${JWT_KEY}"
export SESSION_SECRET="${SESSION_KEY}"

# Vault Root Token
export VAULT_DEV_ROOT_TOKEN="${VAULT_TOKEN}"

# API Keys (replace with real ones)
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key-here"
EOF

chmod 600 .env.secure

echo "✅ Secure credentials generated in .env.secure"
echo ""
echo "Sample passwords generated:"
echo "  PostgreSQL: ${PG_PASS:0:8}..."
echo "  Neo4j: ${NEO_PASS:0:8}..."
echo "  Vault Token: ${VAULT_TOKEN:0:8}..."
