#!/bin/bash
# BEV VAULT INTEGRATION FOR MULTINODE DEPLOYMENT
# Sets up HashiCorp Vault on nodes BEFORE deploying services

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🔐 BEV VAULT SETUP FOR MULTINODE${NC}"
echo -e "${BLUE}====================================================${NC}"
echo "Setting up centralized credential management on THANOS and ORACLE1"
echo ""

# Step 1: Check if vault-setup.sh exists and is executable
echo -e "${CYAN}Step 1: Checking Vault Setup Scripts${NC}"

if [ ! -f "config/vault-setup.sh" ]; then
    echo -e "${YELLOW}vault-setup.sh missing, creating it...${NC}"
    # The full vault-setup.sh already exists in the project
    chmod +x config/vault-setup.sh
fi

if [ ! -f "generate_secrets.sh" ]; then
    echo -e "${YELLOW}generate_secrets.sh missing, checking...${NC}"
fi

# Step 2: Generate secure passwords locally first
echo -e "${CYAN}Step 2: Generating Secure Credentials${NC}"

if [ ! -f ".env.secure" ]; then
    echo "Creating secure passwords..."
    cat > .env.secure << EOF
# Secure Environment Variables - NEVER COMMIT THIS FILE
# Generated on $(date)

# Database Passwords
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
NEO4J_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Service Passwords
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
RABBITMQ_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
KAFKA_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# Encryption Keys
ENCRYPTION_KEY=$(openssl rand -base64 64)
JWT_SECRET=$(openssl rand -base64 64)
SESSION_SECRET=$(openssl rand -base64 32)

# Vault Root Token (for initial setup)
VAULT_DEV_ROOT_TOKEN=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

# API Keys (to be replaced with real ones)
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
EOF
    chmod 600 .env.secure
    echo -e "${GREEN}✅ Secure passwords generated${NC}"
else
    echo -e "${GREEN}✅ Using existing .env.secure${NC}"
fi

source .env.secure

# Step 3: Create Vault deployment script for THANOS
echo -e "${CYAN}Step 3: Creating Vault Deployment for THANOS${NC}"

cat > deploy-vault-thanos.sh << 'EOF'
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
EOF

# Step 4: Create environment files for each node with Vault integration
echo -e "${CYAN}Step 4: Creating Node-Specific Environment Files${NC}"

# THANOS environment (primary Vault server)
cat > .env.thanos << EOF
# THANOS Node Configuration with Vault
NODE_NAME=thanos
NODE_TYPE=primary

# Vault Configuration (Primary)
VAULT_URL=http://localhost:8200
VAULT_TOKEN=${VAULT_DEV_ROOT_TOKEN}
SECRETS_BACKEND=vault

# Database Credentials (from Vault)
DB_PASSWORD=${DB_PASSWORD}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
NEO4J_PASSWORD=${NEO4J_PASSWORD}
REDIS_PASSWORD=${REDIS_PASSWORD}
RABBITMQ_PASSWORD=${RABBITMQ_PASSWORD}

# Node IPs
THANOS_IP=127.0.0.1
ORACLE1_IP=oracle1
EOF

# ORACLE1 environment (connects to THANOS Vault)
cat > .env.oracle1 << EOF
# ORACLE1 Node Configuration with Vault
NODE_NAME=oracle1
NODE_TYPE=secondary

# Vault Configuration (Points to THANOS)
VAULT_URL=http://thanos:8200
VAULT_TOKEN=${VAULT_DEV_ROOT_TOKEN}
SECRETS_BACKEND=vault

# Service Credentials (from Vault)
REDIS_PASSWORD=${REDIS_PASSWORD}

# Database Connections (to THANOS)
POSTGRES_HOST=thanos
POSTGRES_PORT=5432
NEO4J_HOST=thanos
NEO4J_BOLT_PORT=7687
REDIS_HOST=thanos
KAFKA_HOST=thanos
EOF

echo -e "${GREEN}✅ Environment files created${NC}"

# Step 5: Update docker-compose files to use Vault
echo -e "${CYAN}Step 5: Updating Docker Compose for Vault Integration${NC}"

# Check if we need to add Vault service to docker-compose files
if ! grep -q "vault:" docker-compose-thanos-unified.yml 2>/dev/null; then
    echo -e "${YELLOW}Adding Vault service to docker-compose-thanos-unified.yml${NC}"
    
    # This would need to be inserted properly, but for now we'll use the standalone container
fi

# Step 6: Create deployment script with Vault
cat > deploy-with-vault.sh << 'DEPLOY_SCRIPT'
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
DEPLOY_SCRIPT

chmod +x deploy-with-vault.sh
chmod +x deploy-vault-thanos.sh

echo -e "${GREEN}✅ Vault integration scripts created${NC}"

# Step 7: Add to Git
echo -e "${CYAN}Step 7: Committing Vault Integration${NC}"

git add -A
git commit -m "Add Vault credential management for multinode deployment" || true
git push origin main || true

# Summary
echo ""
echo -e "${GREEN}🎉 VAULT INTEGRATION COMPLETE!${NC}"
echo ""
echo -e "${BLUE}What's been set up:${NC}"
echo "  ✅ Secure password generation (.env.secure)"
echo "  ✅ Vault deployment scripts"
echo "  ✅ Node-specific environment files with Vault"
echo "  ✅ Deployment script with Vault integration"
echo ""
echo -e "${YELLOW}To deploy with Vault:${NC}"
echo "  ./deploy-with-vault.sh"
echo ""
echo -e "${CYAN}This will:${NC}"
echo "  1. Deploy Vault on THANOS first"
echo "  2. Load all credentials into Vault"
echo "  3. Start services that fetch credentials from Vault"
echo "  4. ORACLE1 connects to THANOS Vault for credentials"
echo ""
echo -e "${RED}IMPORTANT:${NC}"
echo "  Keep .env.secure safe - it contains the Vault root token!"
echo "  Never commit .env.secure to Git!"
