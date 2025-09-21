#!/bin/bash
# BEV Multi-Node Deployment Orchestrator
# Properly deploys BEV across Thanos (GPU/x86) and Oracle1 (ARM)
# With centralized Vault credential management

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
THANOS_HOST="thanos"
ORACLE1_HOST="oracle1"
PROJECT_DIR="/home/starlord/Projects/Bev"
VAULT_ADDR="http://100.122.12.35:8200"  # Starlord's Tailscale IP
THANOS_IP="100.122.12.54"
ORACLE1_IP="100.96.197.84"
STARLORD_IP="100.122.12.35"

echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}     BEV ENTERPRISE MULTI-NODE DEPLOYMENT ORCHESTRATOR        ${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${CYAN}Deployment Architecture:${NC}"
echo "  • Control Node (Starlord): Vault, Development, Frontend"
echo "  • Compute Node (Thanos):   GPU Services, Primary DBs, AI/ML"
echo "  • Edge Node (Oracle1):     ARM Services, Monitoring, Lightweight"
echo ""
echo -e "${YELLOW}Starting deployment at $(date)${NC}"
echo ""

# Function to log with timestamp
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        INFO)
            echo -e "${BLUE}[$timestamp]${NC} ${GREEN}[INFO]${NC} $message"
            ;;
        WARN)
            echo -e "${BLUE}[$timestamp]${NC} ${YELLOW}[WARN]${NC} $message"
            ;;
        ERROR)
            echo -e "${BLUE}[$timestamp]${NC} ${RED}[ERROR]${NC} $message"
            ;;
        SUCCESS)
            echo -e "${BLUE}[$timestamp]${NC} ${GREEN}[✓]${NC} $message"
            ;;
        *)
            echo -e "${BLUE}[$timestamp]${NC} $message"
            ;;
    esac
}

# Phase 1: Setup Vault on Control Node (Starlord)
setup_vault() {
    log INFO "Phase 1: Setting up Vault on Control Node"
    
    # Check if Vault is already running
    if docker ps | grep -q vault; then
        log WARN "Vault container already exists, checking status..."
        export VAULT_TOKEN=$(cat vault-init.json 2>/dev/null | jq -r '.root_token' || echo "")
        if [ -z "$VAULT_TOKEN" ]; then
            log ERROR "Vault running but no token found. Please check vault-init.json"
            exit 1
        fi
        log SUCCESS "Vault already configured"
        return 0
    fi
    
    # Create Vault directories
    log INFO "Creating Vault directories..."
    sudo mkdir -p /opt/vault/{data,tls,logs,policies}
    sudo chown -R $(whoami):$(whoami) /opt/vault
    
    # Copy Vault configuration
    cp config/vault.hcl /opt/vault/vault.hcl
    cp -r config/vault-policies/* /opt/vault/policies/
    
    # Generate TLS certificates for Vault
    log INFO "Generating TLS certificates for Vault..."
    openssl genrsa -out /opt/vault/tls/vault.key 4096
    openssl req -new -x509 -key /opt/vault/tls/vault.key \
        -out /opt/vault/tls/vault.crt -days 365 \
        -subj "/C=US/ST=CA/L=SF/O=BEV/CN=vault.bev.local" \
        -addext "subjectAltName = IP:$STARLORD_IP,IP:127.0.0.1,DNS:vault.bev.local,DNS:localhost"
    
    # Start Vault container
    log INFO "Starting Vault container..."
    docker run -d \
        --name vault \
        --restart always \
        --network host \
        -v /opt/vault/data:/vault/data \
        -v /opt/vault/tls:/vault/tls \
        -v /opt/vault/vault.hcl:/vault/config/vault.hcl \
        -v /opt/vault/logs:/vault/logs \
        -e 'VAULT_CONFIG_DIR=/vault/config' \
        -e 'VAULT_LOG_LEVEL=info' \
        --cap-add IPC_LOCK \
        hashicorp/vault:latest server
    
    sleep 10
    
    # Initialize Vault
    log INFO "Initializing Vault..."
    export VAULT_ADDR="http://localhost:8200"
    export VAULT_SKIP_VERIFY=1
    
    vault operator init -key-shares=5 -key-threshold=3 -format=json > vault-init.json
    
    # Unseal Vault
    log INFO "Unsealing Vault..."
    UNSEAL_KEY_1=$(jq -r '.unseal_keys_b64[0]' vault-init.json)
    UNSEAL_KEY_2=$(jq -r '.unseal_keys_b64[1]' vault-init.json)
    UNSEAL_KEY_3=$(jq -r '.unseal_keys_b64[2]' vault-init.json)
    export VAULT_TOKEN=$(jq -r '.root_token' vault-init.json)
    
    vault operator unseal "$UNSEAL_KEY_1"
    vault operator unseal "$UNSEAL_KEY_2"
    vault operator unseal "$UNSEAL_KEY_3"
    
    # Enable secrets engines
    log INFO "Configuring Vault secrets engines..."
    vault secrets enable -path=bev kv-v2
    vault secrets enable database
    vault secrets enable transit
    
    # Apply security policies
    log INFO "Applying Vault security policies..."
    for policy in /opt/vault/policies/*.hcl; do
        policy_name=$(basename "$policy" .hcl | sed 's/-policy//')
        vault policy write "$policy_name" "$policy"
    done
    
    log SUCCESS "Vault setup complete"
    echo -e "${YELLOW}IMPORTANT: Secure vault-init.json file!${NC}"
}

# Phase 2: Generate and Load Secrets into Vault
load_secrets() {
    log INFO "Phase 2: Loading secrets into Vault"
    
    export VAULT_ADDR="http://localhost:8200"
    export VAULT_TOKEN=$(jq -r '.root_token' vault-init.json)
    
    # Generate secure passwords if not exists
    if [ ! -f ".env.secure" ]; then
        log INFO "Generating secure passwords..."
        ./generate_secrets.sh
    fi
    
    # Source the secure environment
    source .env.secure
    
    # Load all secrets into Vault
    log INFO "Loading database passwords..."
    vault kv put bev/database \
        postgres_password="$POSTGRES_PASSWORD" \
        neo4j_password="$NEO4J_PASSWORD" \
        db_password="$DB_PASSWORD"
    
    log INFO "Loading service passwords..."
    vault kv put bev/services \
        redis_password="$REDIS_PASSWORD" \
        rabbitmq_password="$RABBITMQ_PASSWORD" \
        kafka_password="$KAFKA_PASSWORD" \
        swarm_password="$SWARM_PASSWORD"
    
    log INFO "Loading encryption keys..."
    vault kv put bev/encryption \
        encryption_key="$ENCRYPTION_KEY" \
        jwt_secret="$JWT_SECRET" \
        session_secret="$SESSION_SECRET"
    
    log INFO "Loading API keys..."
    vault kv put bev/api_keys \
        openai="$OPENAI_API_KEY" \
        anthropic="$ANTHROPIC_API_KEY" \
        elevenlabs="$ELEVENLABS_API_KEY" \
        shodan="$SHODAN_API_KEY" \
        virustotal="$VIRUSTOTAL_API_KEY" \
        etherscan="$ETHERSCAN_API_KEY" \
        blockcypher="$BLOCKCYPHER_API_KEY" \
        alchemy="$ALCHEMY_API_KEY"
    
    log SUCCESS "Secrets loaded into Vault"
}

# Phase 3: Prepare deployment packages for nodes
prepare_deployments() {
    log INFO "Phase 3: Preparing deployment packages"
    
    # Create deployment directories
    mkdir -p deployments/{thanos,oracle1}/{config,scripts,compose}
    
    # Generate Vault tokens for each node
    log INFO "Creating AppRole authentication for nodes..."
    
    # Enable AppRole auth
    vault auth enable approle 2>/dev/null || true
    
    # Create role for Thanos
    vault write auth/approle/role/thanos \
        token_policies="application" \
        token_ttl=24h \
        token_max_ttl=72h
    
    THANOS_ROLE_ID=$(vault read -field=role_id auth/approle/role/thanos/role-id)
    THANOS_SECRET_ID=$(vault write -field=secret_id -f auth/approle/role/thanos/secret-id)
    
    # Create role for Oracle1
    vault write auth/approle/role/oracle1 \
        token_policies="application" \
        token_ttl=24h \
        token_max_ttl=72h
    
    ORACLE1_ROLE_ID=$(vault read -field=role_id auth/approle/role/oracle1/role-id)
    ORACLE1_SECRET_ID=$(vault write -field=secret_id -f auth/approle/role/oracle1/secret-id)
    
    # Create environment files for each node
    cat > deployments/thanos/.env << EOF
# Vault Configuration
VAULT_ADDR=$VAULT_ADDR
VAULT_ROLE_ID=$THANOS_ROLE_ID
VAULT_SECRET_ID=$THANOS_SECRET_ID
SECRETS_BACKEND=vault

# Node Configuration
NODE_NAME=thanos
NODE_IP=$THANOS_IP
NODE_ROLE=primary-compute,gpu

# Cross-Node Communication
STARLORD_IP=$STARLORD_IP
ORACLE1_IP=$ORACLE1_IP

# Service Discovery
POSTGRES_HOST=$THANOS_IP
NEO4J_HOST=$THANOS_IP
REDIS_HOST=$THANOS_IP
RABBITMQ_HOST=$THANOS_IP
KAFKA_HOST=$THANOS_IP

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility
EOF

    cat > deployments/oracle1/.env << EOF
# Vault Configuration
VAULT_ADDR=$VAULT_ADDR
VAULT_ROLE_ID=$ORACLE1_ROLE_ID
VAULT_SECRET_ID=$ORACLE1_SECRET_ID
SECRETS_BACKEND=vault

# Node Configuration
NODE_NAME=oracle1
NODE_IP=$ORACLE1_IP
NODE_ROLE=edge-compute,arm

# Cross-Node Communication
STARLORD_IP=$STARLORD_IP
THANOS_IP=$THANOS_IP

# Service Discovery (pointing to Thanos for primary services)
POSTGRES_HOST=$THANOS_IP
NEO4J_HOST=$THANOS_IP
REDIS_HOST=$ORACLE1_IP
RABBITMQ_HOST=$THANOS_IP
KAFKA_HOST=$THANOS_IP

# Platform
PLATFORM=linux/arm64
EOF
    
    log SUCCESS "Deployment packages prepared"
}

# Phase 4: Deploy to Thanos (GPU/Primary Compute)
deploy_thanos() {
    log INFO "Phase 4: Deploying to Thanos (GPU/Primary Compute)"
    
    # Copy deployment files to Thanos
    log INFO "Copying files to Thanos..."
    ssh $THANOS_HOST "mkdir -p /opt/bev/{config,scripts,data,logs}"
    
    # Copy the prepared deployment package
    scp deployments/thanos/.env $THANOS_HOST:/opt/bev/.env
    scp docker-compose-thanos-unified.yml $THANOS_HOST:/opt/bev/docker-compose.yml
    
    # Create Vault authentication script
    cat > deployments/thanos/auth-vault.sh << 'EOF'
#!/bin/bash
# Authenticate with Vault using AppRole
source /opt/bev/.env
export VAULT_TOKEN=$(curl -s -X POST \
    $VAULT_ADDR/v1/auth/approle/login \
    -d "{\"role_id\":\"$VAULT_ROLE_ID\",\"secret_id\":\"$VAULT_SECRET_ID\"}" \
    | jq -r '.auth.client_token')
echo "export VAULT_TOKEN=$VAULT_TOKEN" >> /opt/bev/.env
EOF
    
    scp deployments/thanos/auth-vault.sh $THANOS_HOST:/opt/bev/auth-vault.sh
    
    # Deploy services on Thanos
    log INFO "Starting services on Thanos..."
    ssh $THANOS_HOST << 'REMOTE_SCRIPT'
cd /opt/bev
chmod +x auth-vault.sh
./auth-vault.sh
source .env

# Pull required images
docker compose pull

# Start core services
docker compose up -d postgres neo4j elasticsearch influxdb

# Wait for databases
sleep 30

# Start message queues
docker compose up -d zookeeper kafka-1 kafka-2 kafka-3 rabbitmq-1 rabbitmq-2 rabbitmq-3

# Start AI/ML services
docker compose up -d autonomous-coordinator adaptive-learning knowledge-evolution

# Start processing services
docker compose up -d intelowl-django intelowl-celery-worker

echo "Thanos deployment started"
REMOTE_SCRIPT
    
    log SUCCESS "Thanos deployment complete"
}

# Phase 5: Deploy to Oracle1 (ARM/Edge)
deploy_oracle1() {
    log INFO "Phase 5: Deploying to Oracle1 (ARM/Edge)"
    
    # Copy deployment files to Oracle1
    log INFO "Copying files to Oracle1..."
    ssh $ORACLE1_HOST "mkdir -p /opt/bev/{config,scripts,data,logs}"
    
    # Copy the prepared deployment package
    scp deployments/oracle1/.env $ORACLE1_HOST:/opt/bev/.env
    scp docker-compose-oracle1-unified.yml $ORACLE1_HOST:/opt/bev/docker-compose.yml
    
    # Create Vault authentication script
    cat > deployments/oracle1/auth-vault.sh << 'EOF'
#!/bin/bash
# Authenticate with Vault using AppRole
source /opt/bev/.env
export VAULT_TOKEN=$(curl -s -X POST \
    $VAULT_ADDR/v1/auth/approle/login \
    -d "{\"role_id\":\"$VAULT_ROLE_ID\",\"secret_id\":\"$VAULT_SECRET_ID\"}" \
    | jq -r '.auth.client_token')
echo "export VAULT_TOKEN=$VAULT_TOKEN" >> /opt/bev/.env
EOF
    
    scp deployments/oracle1/auth-vault.sh $ORACLE1_HOST:/opt/bev/auth-vault.sh
    
    # Deploy services on Oracle1
    log INFO "Starting services on Oracle1..."
    ssh $ORACLE1_HOST << 'REMOTE_SCRIPT'
cd /opt/bev
chmod +x auth-vault.sh
./auth-vault.sh
source .env

# Pull required images
docker compose pull

# Start monitoring services
docker compose up -d prometheus grafana

# Start Redis
docker compose up -d redis

# Start security services
docker compose up -d tor proxy-manager vault-proxy

# Start lightweight analyzers
docker compose up -d breach-analyzer crypto-analyzer social-analyzer

echo "Oracle1 deployment started"
REMOTE_SCRIPT
    
    log SUCCESS "Oracle1 deployment complete"
}

# Phase 6: Verify Deployment
verify_deployment() {
    log INFO "Phase 6: Verifying deployment"
    
    # Check Vault
    log INFO "Checking Vault status..."
    vault status
    
    # Check Thanos services
    log INFO "Checking Thanos services..."
    ssh $THANOS_HOST "docker ps --format 'table {{.Names}}\t{{.Status}}' | head -10"
    
    # Check Oracle1 services
    log INFO "Checking Oracle1 services..."
    ssh $ORACLE1_HOST "docker ps --format 'table {{.Names}}\t{{.Status}}' | head -10"
    
    # Test cross-node connectivity
    log INFO "Testing cross-node connectivity..."
    
    # Test Thanos -> Oracle1
    ssh $THANOS_HOST "ping -c 2 $ORACLE1_IP"
    
    # Test Oracle1 -> Thanos  
    ssh $ORACLE1_HOST "ping -c 2 $THANOS_IP"
    
    log SUCCESS "Deployment verification complete"
}

# Main execution
main() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    
    # Run all phases
    setup_vault
    load_secrets
    prepare_deployments
    deploy_thanos
    deploy_oracle1
    verify_deployment
    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}         BEV MULTI-NODE DEPLOYMENT COMPLETE!                   ${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}Access Points:${NC}"
    echo "  • Vault UI:       http://$STARLORD_IP:8200/ui"
    echo "  • Neo4j Browser:  http://$THANOS_IP:7474"
    echo "  • Grafana:        http://$ORACLE1_IP:3000"
    echo "  • Prometheus:     http://$ORACLE1_IP:9090"
    echo ""
    echo -e "${YELLOW}Important:${NC}"
    echo "  1. Secure the vault-init.json file immediately"
    echo "  2. Monitor logs: docker compose logs -f (on each node)"
    echo "  3. AppRole tokens expire in 24h - renew as needed"
    echo ""
    echo -e "${GREEN}Deployment completed at $(date)${NC}"
}

# Run main function
main "$@"
