#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║           ORACLE1 REMOTE DEPLOYMENT SCRIPT                       ║
# ║        DEPLOYS TO ORACLE1 VIA SSH - NOT LOCAL!                   ║
# ╔══════════════════════════════════════════════════════════════════╝

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

ORACLE1_HOST="oracle1"
ORACLE1_IP="100.96.197.84"
STARLORD_IP="100.122.12.35"
THANOS_IP="100.122.12.54"
DEPLOYMENT_DIR="/opt/bev"

echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}   ORACLE1 NODE DEPLOYMENT (ARM64, 24GB RAM)                   ${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""

# CRITICAL CHECK - Ensure we're NOT deploying locally
CURRENT_HOST=$(hostname)
if [ "$CURRENT_HOST" == "oracle1" ]; then
    echo -e "${RED}❌ ERROR: You're ON ORACLE1! This script deploys FROM STARLORD TO ORACLE1!${NC}"
    exit 1
fi

if [ "$CURRENT_HOST" != "starlord" ]; then
    echo -e "${YELLOW}⚠️  Warning: Expected to run from STARLORD, current: $CURRENT_HOST${NC}"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Test SSH connectivity
echo -e "${BLUE}→ Testing SSH connectivity to ORACLE1...${NC}"
if ! ssh -q $ORACLE1_HOST exit; then
    echo -e "${RED}❌ Cannot connect to ORACLE1 via SSH${NC}"
    echo -e "${YELLOW}   Please ensure:${NC}"
    echo -e "${YELLOW}   1. SSH key is configured${NC}"
    echo -e "${YELLOW}   2. ORACLE1 is reachable${NC}"
    echo -e "${YELLOW}   3. Tailscale is connected${NC}"
    exit 1
fi
echo -e "${GREEN}✅ SSH connection successful${NC}"

# Check Vault is running on STARLORD
echo -e "${BLUE}→ Checking Vault on STARLORD...${NC}"
if ! curl -s http://localhost:8200/v1/sys/health > /dev/null 2>&1; then
    echo -e "${RED}❌ Vault not running on STARLORD!${NC}"
    echo -e "${YELLOW}   Run ./deploy_starlord_vault_only.sh first${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Vault is running${NC}"

# Prepare deployment package
echo -e "${BLUE}→ Preparing deployment package...${NC}"
TEMP_DIR=$(mktemp -d)
cp docker-compose-oracle1-unified.yml $TEMP_DIR/
cp -r docker/oracle $TEMP_DIR/docker/ 2>/dev/null || true
cp -r config $TEMP_DIR/ 2>/dev/null || true
cp -r grafana $TEMP_DIR/ 2>/dev/null || true
cp -r prometheus $TEMP_DIR/ 2>/dev/null || true
cp .env $TEMP_DIR/ 2>/dev/null || true

# Create deployment script for ORACLE1
cat << 'REMOTE_SCRIPT' > $TEMP_DIR/deploy_on_oracle1.sh
#!/bin/bash
cd /opt/bev

# Set environment variables for cross-node communication
export VAULT_ADDR=http://100.122.12.35:8200
export THANOS_ENDPOINT=http://100.122.12.54:8000
export POSTGRES_HOST=100.122.12.54

echo "Stopping any existing services..."
docker-compose -f docker-compose-oracle1-unified.yml down 2>/dev/null || true

echo "Starting ORACLE1 services..."
docker-compose -f docker-compose-oracle1-unified.yml up -d

echo "Waiting for services to stabilize..."
sleep 10

echo "Service status:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep bev_ || echo "No services running"
REMOTE_SCRIPT

chmod +x $TEMP_DIR/deploy_on_oracle1.sh

# Copy files to ORACLE1
echo -e "${BLUE}→ Copying files to ORACLE1...${NC}"
ssh $ORACLE1_HOST "sudo mkdir -p $DEPLOYMENT_DIR && sudo chown \$USER:\$USER $DEPLOYMENT_DIR"
scp -r $TEMP_DIR/* $ORACLE1_HOST:$DEPLOYMENT_DIR/

# Execute deployment on ORACLE1
echo -e "${CYAN}→ DEPLOYING SERVICES ON ORACLE1 (REMOTE)...${NC}"
ssh -t $ORACLE1_HOST "cd $DEPLOYMENT_DIR && bash deploy_on_oracle1.sh"

# Cleanup
rm -rf $TEMP_DIR

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ ORACLE1 DEPLOYMENT COMPLETE${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Services deployed to ORACLE1 (${ORACLE1_IP}):${NC}"
echo -e "  • Prometheus & Grafana monitoring"
echo -e "  • AlertManager"
echo -e "  • Redis (ARM-optimized)"
echo -e "  • n8n workflow automation"
echo -e "  • MinIO cluster"
echo -e "  • InfluxDB time-series"
echo -e "  • LiteLLM gateways"
echo -e "  • Edge workers"
echo -e "  • Multimodal processors"
echo ""
echo -e "${BLUE}Access points:${NC}"
echo -e "  • Grafana: http://${ORACLE1_IP}:3000"
echo -e "  • Prometheus: http://${ORACLE1_IP}:9090"
echo -e "  • n8n: http://${ORACLE1_IP}:5678"
echo -e "  • MinIO Console: http://${ORACLE1_IP}:9011"
echo -e "  • AlertManager: http://${ORACLE1_IP}:9093"