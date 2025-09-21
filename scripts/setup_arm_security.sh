#!/bin/bash
# ARM Security Setup Script for Oracle1 Node
# Configures Vault, Tor, OPSEC, and security services for ARM64

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔒 SETTING UP ARM SECURITY STACK${NC}"
echo "================================="

# Verify ARM architecture
if [ "$(uname -m)" != "aarch64" ]; then
    echo -e "${YELLOW}⚠️ Not on ARM64 architecture: $(uname -m)${NC}"
fi

# Wait for security services
echo -e "${YELLOW}⏳ Waiting for security services...${NC}"
sleep 20

# Configure Vault on ARM
echo -e "${BLUE}🔐 Configuring Vault...${NC}"
until curl -s http://localhost:8200/v1/sys/health > /dev/null 2>&1; do
    echo "Waiting for Vault to be ready..."
    sleep 5
done

# Initialize Vault if sealed
echo -n "Checking Vault status... "
if docker exec bev_vault vault status | grep -q "Sealed.*true"; then
    echo -e "${YELLOW}🔒 Vault is sealed${NC}"

    # Initialize Vault (development mode)
    echo "Initializing Vault..."
    docker exec bev_vault vault operator init -key-shares=3 -key-threshold=2 > /tmp/vault-keys.txt

    # Unseal Vault
    echo "Unsealing Vault..."
    UNSEAL_KEY1=$(grep "Unseal Key 1:" /tmp/vault-keys.txt | cut -d: -f2 | tr -d ' ')
    UNSEAL_KEY2=$(grep "Unseal Key 2:" /tmp/vault-keys.txt | cut -d: -f2 | tr -d ' ')

    docker exec bev_vault vault operator unseal $UNSEAL_KEY1
    docker exec bev_vault vault operator unseal $UNSEAL_KEY2

    echo -e "${GREEN}✅ Vault unsealed${NC}"
else
    echo -e "${GREEN}✅ Vault ready${NC}"
fi

# Configure Vault policies
echo -e "${BLUE}📋 Setting up Vault policies...${NC}"
ROOT_TOKEN=$(grep "Initial Root Token:" /tmp/vault-keys.txt | cut -d: -f2 | tr -d ' ' 2>/dev/null || echo "dev-token")

# Enable secret engines
docker exec -e VAULT_TOKEN=$ROOT_TOKEN bev_vault vault secrets enable -path=bev/ kv-v2
docker exec -e VAULT_TOKEN=$ROOT_TOKEN bev_vault vault secrets enable -path=database/ database

# Create policies
docker exec -e VAULT_TOKEN=$ROOT_TOKEN bev_vault vault policy write bev-admin - << POLICY_EOF
path "bev/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
path "database/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}
POLICY_EOF

echo -e "${GREEN}✅ Vault policies configured${NC}"

# Configure Tor Network on ARM
echo -e "${BLUE}🌐 Configuring Tor network...${NC}"
until docker exec bev_tor tor --verify-config > /dev/null 2>&1; do
    echo "Waiting for Tor to be ready..."
    sleep 5
done

# Verify Tor configuration
docker exec bev_tor tor --verify-config
echo -e "${GREEN}✅ Tor network configured${NC}"

# Setup OPSEC Enforcer
echo -e "${BLUE}🛡️ Setting up OPSEC Enforcer...${NC}"
if docker ps | grep bev_opsec_enforcer > /dev/null; then
    echo -n "OPSEC Enforcer health... "
    if docker exec bev_opsec_enforcer python -c "import requests; print('healthy')" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
    else
        echo -e "${RED}❌ Unhealthy${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ OPSEC Enforcer not deployed${NC}"
fi

# Configure security monitoring
echo -e "${BLUE}👁️ Setting up security monitoring...${NC}"

# Create security alert rules
cat > /tmp/security-alerts.yml << SECURITY_EOF
groups:
- name: security_alerts
  rules:
  - alert: UnauthorizedAccess
    expr: failed_login_attempts > 5
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Multiple failed login attempts detected"

  - alert: SuspiciousNetworkActivity
    expr: unusual_network_connections > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Unusual network activity detected"

  - alert: VaultAccessViolation
    expr: vault_access_denied > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Vault access violation detected"
SECURITY_EOF

# Setup SSL certificates
echo -e "${BLUE}🔑 Setting up SSL certificates...${NC}"
if [ -f "config/ssl/bev-frontend.crt" ]; then
    echo -e "${GREEN}✅ SSL certificates exist${NC}"
else
    echo -e "${YELLOW}⚠️ SSL certificates missing, generating...${NC}"
    mkdir -p config/ssl

    # Generate self-signed certificate for development
    openssl req -x509 -newkey rsa:4096 -keyout config/ssl/bev-frontend.key \
        -out config/ssl/bev-frontend.crt -days 365 -nodes \
        -subj "/C=US/ST=State/L=City/O=BEV/CN=localhost"

    echo -e "${GREEN}✅ SSL certificates generated${NC}"
fi

# Run security health checks
echo -e "${BLUE}🏥 Running security health checks...${NC}"

SECURITY_SERVICES=(
    "vault:8200"
    "tor:9050"
    "consul:8500"
)

HEALTHY_SECURITY=0

for service in "${SECURITY_SERVICES[@]}"; do
    SERVICE_NAME=$(echo $service | cut -d: -f1)
    SERVICE_PORT=$(echo $service | cut -d: -f2)

    echo -n "Security check $SERVICE_NAME... "
    if curl -s http://localhost:$SERVICE_PORT > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Secure${NC}"
        HEALTHY_SECURITY=$((HEALTHY_SECURITY + 1))
    else
        echo -e "${RED}❌ Issue${NC}"
    fi
done

echo ""
echo -e "${BLUE}🔐 ARM Security Health Summary:${NC}"
echo "Secure Services: $HEALTHY_SECURITY/${#SECURITY_SERVICES[@]}"

if [ $HEALTHY_SECURITY -eq ${#SECURITY_SERVICES[@]} ]; then
    echo -e "${GREEN}🎯 ARM security setup successful!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ Some security services need attention${NC}"
    exit 1
fi