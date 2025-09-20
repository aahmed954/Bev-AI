#!/bin/bash
# BEV Deployment Fix - Properly Integrated with Vault Credential Management
# THIS SCRIPT FIXES THE DEPLOYMENT BY PROPERLY USING THE CENTRALIZED CREDENTIAL SYSTEM

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🔐 BEV DEPLOYMENT FIX - VAULT INTEGRATION${NC}"
echo -e "${BLUE}====================================================${NC}"
echo "This script properly integrates the centralized credential management"
echo "that was completely missed in previous deployment attempts"
echo ""

# Step 1: Check if Vault is running
echo -e "${CYAN}Step 1: Checking Vault Status${NC}"
if ! docker ps | grep -q vault; then
    echo -e "${YELLOW}Starting Vault server...${NC}"
    
    # Create Vault data directory
    sudo mkdir -p /opt/vault/data
    sudo mkdir -p /opt/vault/tls
    sudo mkdir -p /opt/vault/logs
    
    # Start Vault container
    docker run -d \
        --name vault \
        --restart always \
        -p 8200:8200 \
        -v /opt/vault/data:/opt/vault/data \
        -v /opt/vault/tls:/opt/vault/tls \
        -v $(pwd)/config/vault.hcl:/opt/vault/config/vault.hcl \
        -e 'VAULT_CONFIG_DIR=/opt/vault/config' \
        -e 'VAULT_LOG_LEVEL=info' \
        --cap-add IPC_LOCK \
        hashicorp/vault:latest server
    
    echo "Waiting for Vault to start..."
    sleep 10
fi

# Step 2: Initialize Vault if needed
echo -e "${CYAN}Step 2: Initializing Vault${NC}"
export VAULT_ADDR='http://localhost:8200'

if [ ! -f "vault-init.json" ]; then
    echo "Initializing Vault..."
    cd config && ./vault-setup.sh && cd ..
    
    if [ -f "config/vault-init.json" ]; then
        mv config/vault-init.json ./vault-init.json
    fi
else
    echo "Vault already initialized"
fi

# Extract root token
if [ -f "vault-init.json" ]; then
    export VAULT_TOKEN=$(jq -r '.root_token' vault-init.json)
    echo -e "${GREEN}✅ Vault token loaded${NC}"
else
    echo -e "${RED}❌ vault-init.json not found! Cannot proceed${NC}"
    echo "Please initialize Vault manually first"
    exit 1
fi

# Step 3: Generate secure passwords
echo -e "${CYAN}Step 3: Generating Secure Passwords${NC}"
if [ ! -f ".env.secure" ]; then
    ./generate_secrets.sh
fi

# Step 4: Load secrets into Vault
echo -e "${CYAN}Step 4: Loading Secrets into Vault${NC}"
source .env.secure

# Create KV v2 secrets engine if not exists
vault secrets enable -path=bev kv-v2 2>/dev/null || echo "KV engine already enabled"

# Load database passwords into Vault
vault kv put bev/DB_PASSWORD value="${DB_PASSWORD}"
vault kv put bev/POSTGRES_PASSWORD value="${POSTGRES_PASSWORD}"
vault kv put bev/NEO4J_PASSWORD value="${NEO4J_PASSWORD}"

# Load service passwords
vault kv put bev/REDIS_PASSWORD value="${REDIS_PASSWORD}"
vault kv put bev/RABBITMQ_PASSWORD value="${RABBITMQ_PASSWORD}"
vault kv put bev/KAFKA_PASSWORD value="${KAFKA_PASSWORD}"
vault kv put bev/SWARM_PASSWORD value="${SWARM_PASSWORD}"

# Load encryption keys
vault kv put bev/ENCRYPTION_KEY value="${ENCRYPTION_KEY}"
vault kv put bev/JWT_SECRET value="${JWT_SECRET}"
vault kv put bev/SESSION_SECRET value="${SESSION_SECRET}"

# Load Tor password
vault kv put bev/TOR_CONTROL_PASSWORD value="${TOR_CONTROL_PASSWORD}"

echo -e "${GREEN}✅ Secrets loaded into Vault${NC}"

# Step 5: Create proper environment files for services
echo -e "${CYAN}Step 5: Creating Service Environment Files${NC}"

# Create base environment with Vault integration
cat > .env.vault << EOF
# Vault Configuration
VAULT_URL=http://localhost:8200
VAULT_TOKEN=${VAULT_TOKEN}
SECRETS_BACKEND=vault

# Service Discovery
CONSUL_HOST=localhost
CONSUL_PORT=8500

# Node Configuration
NODE_ENV=production
DEPLOYMENT_MODE=DISTRIBUTED
EOF

# Create Thanos-specific environment
cat > .env.thanos << EOF
# Include base Vault config
$(cat .env.vault)

# Thanos Node Configuration
NODE_NAME=thanos
NODE_ROLE=primary-compute,gpu
CUDA_VISIBLE_DEVICES=0

# Service URLs (internal network)
POSTGRES_HOST=thanos
NEO4J_HOST=thanos
REDIS_HOST=thanos
RABBITMQ_HOST=thanos
EOF

# Create Oracle1-specific environment
cat > .env.oracle1 << EOF
# Include base Vault config
$(cat .env.vault)

# Oracle1 Node Configuration
NODE_NAME=oracle1
NODE_ROLE=distributed-compute,arm
PLATFORM=linux/arm64

# Service URLs (cross-node)
POSTGRES_HOST=thanos
NEO4J_HOST=thanos
REDIS_HOST=thanos
RABBITMQ_HOST=thanos
EOF

echo -e "${GREEN}✅ Environment files created${NC}"

# Step 6: Fix Docker Compose files to use Vault
echo -e "${CYAN}Step 6: Updating Docker Compose Configurations${NC}"

# Create a fixed docker-compose with Vault integration
cat > docker-compose-vault-integrated.yml << 'EOF'
version: '3.8'

x-vault-env: &vault-env
  VAULT_URL: ${VAULT_URL:-http://vault:8200}
  VAULT_TOKEN: ${VAULT_TOKEN}
  SECRETS_BACKEND: vault

services:
  # Vault Service (if not external)
  vault:
    image: hashicorp/vault:latest
    container_name: bev-vault
    restart: unless-stopped
    ports:
      - "8200:8200"
    volumes:
      - /opt/vault/data:/opt/vault/data
      - ./config/vault.hcl:/vault/config/vault.hcl
    environment:
      VAULT_CONFIG_DIR: /vault/config
      VAULT_LOG_LEVEL: info
    cap_add:
      - IPC_LOCK
    command: server
    networks:
      - bev-network
EOF

  # PostgreSQL with Vault integration
  postgres:
    image: postgres:15-alpine
    container_name: bev-postgres
    restart: unless-stopped
    environment:
      <<: *vault-env
      POSTGRES_DB: bev_db
      POSTGRES_USER: bev_user
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./scripts/init-postgres.sh:/docker-entrypoint-initdb.d/init.sh
    ports:
      - "5432:5432"
    networks:
      - bev-network
    depends_on:
      - vault

  # Redis with Vault integration
  redis:
    image: redis:7-alpine
    container_name: bev-redis
    restart: unless-stopped
    environment:
      <<: *vault-env
    volumes:
      - redis-data:/data
      - ./scripts/init-redis.sh:/usr/local/bin/init-redis.sh
    command: sh -c '/usr/local/bin/init-redis.sh'
    ports:
      - "6379:6379"
    networks:
      - bev-network
    depends_on:
      - vault

  # Neo4j with Vault integration
  neo4j:
    image: neo4j:5-community
    container_name: bev-neo4j
    restart: unless-stopped
    environment:
      <<: *vault-env
      NEO4J_AUTH: none  # Will be set via init script
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs
      - ./scripts/init-neo4j.sh:/var/lib/neo4j/init.sh
    ports:
      - "7474:7474"
      - "7687:7687"
    networks:
      - bev-network
    depends_on:
      - vault

volumes:
  postgres-data:
  redis-data:
  neo4j-data:
  neo4j-logs:

networks:
  bev-network:
    driver: bridge
    name: bev-network
EOF

# Step 7: Create initialization scripts for services
echo -e "${CYAN}Step 7: Creating Service Init Scripts${NC}"

mkdir -p scripts

# PostgreSQL init script
cat > scripts/init-postgres.sh << 'EOF'
#!/bin/bash
# Fetch password from Vault and set it
apt-get update && apt-get install -y curl jq
PASSWORD=$(curl -s -H "X-Vault-Token: $VAULT_TOKEN" \
    $VAULT_URL/v1/bev/data/POSTGRES_PASSWORD | jq -r '.data.data.value')
export POSTGRES_PASSWORD="$PASSWORD"
echo "ALTER USER bev_user WITH PASSWORD '$PASSWORD';" | psql -U postgres
EOF
chmod +x scripts/init-postgres.sh

# Redis init script
cat > scripts/init-redis.sh << 'EOF'
#!/bin/sh
# Fetch password from Vault and configure Redis
apk add --no-cache curl jq
PASSWORD=$(curl -s -H "X-Vault-Token: $VAULT_TOKEN" \
    $VAULT_URL/v1/bev/data/REDIS_PASSWORD | jq -r '.data.data.value')
redis-server --requirepass "$PASSWORD"
EOF
chmod +x scripts/init-redis.sh

# Neo4j init script
cat > scripts/init-neo4j.sh << 'EOF'
#!/bin/bash
# Fetch password from Vault and configure Neo4j
apt-get update && apt-get install -y curl jq
PASSWORD=$(curl -s -H "X-Vault-Token: $VAULT_TOKEN" \
    $VAULT_URL/v1/bev/data/NEO4J_PASSWORD | jq -r '.data.data.value')
neo4j-admin dbms set-initial-password "$PASSWORD"
EOF
chmod +x scripts/init-neo4j.sh

echo -e "${GREEN}✅ Service init scripts created${NC}"

# Step 8: Deploy with proper credential management
echo -e "${CYAN}Step 8: Starting Deployment with Vault Integration${NC}"

# Export environment for docker-compose
export VAULT_URL="http://localhost:8200"
export VAULT_TOKEN

# Start core services
docker-compose -f docker-compose-vault-integrated.yml up -d

echo ""
echo -e "${GREEN}🎉 DEPLOYMENT FIXED WITH VAULT INTEGRATION!${NC}"
echo ""
echo "Services are now starting with centralized credential management."
echo ""
echo -e "${YELLOW}Important Files Created:${NC}"
echo "  - vault-init.json (SECURE THIS!)"
echo "  - .env.secure (contains generated passwords)"
echo "  - .env.vault (Vault integration config)"
echo "  - docker-compose-vault-integrated.yml"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "1. Secure the vault-init.json file (move to secure location)"
echo "2. Monitor services: docker-compose logs -f"
echo "3. Access Vault UI: http://localhost:8200/ui"
echo "4. Deploy to other nodes using .env.thanos and .env.oracle1"
echo ""
echo -e "${GREEN}Vault Token: ${VAULT_TOKEN:0:8}...${NC}"
echo -e "${GREEN}Vault URL: http://localhost:8200${NC}"
