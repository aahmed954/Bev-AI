#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║           THANOS REMOTE DEPLOYMENT SCRIPT                        ║
# ║        DEPLOYS TO THANOS VIA SSH - NOT LOCAL!                    ║
# ╔══════════════════════════════════════════════════════════════════╝

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

THANOS_HOST="thanos"
THANOS_IP="100.122.12.54"
STARLORD_IP="100.122.12.35"
DEPLOYMENT_DIR="/opt/bev"

echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}   THANOS NODE DEPLOYMENT (RTX 3080, 64GB RAM)                 ${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# CRITICAL CHECK - Ensure we're NOT deploying locally
CURRENT_HOST=$(hostname)
if [ "$CURRENT_HOST" == "thanos" ]; then
    echo -e "${RED}❌ ERROR: You're ON THANOS! This script deploys FROM STARLORD TO THANOS!${NC}"
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
echo -e "${BLUE}→ Testing SSH connectivity to THANOS...${NC}"
if ! ssh -q $THANOS_HOST exit; then
    echo -e "${RED}❌ Cannot connect to THANOS via SSH${NC}"
    echo -e "${YELLOW}   Please ensure:${NC}"
    echo -e "${YELLOW}   1. SSH key is configured${NC}"
    echo -e "${YELLOW}   2. THANOS is reachable${NC}"
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
cp docker-compose-thanos-unified.yml $TEMP_DIR/
cp -r cytoscape $TEMP_DIR/ 2>/dev/null || true
cp -r thanos $TEMP_DIR/ 2>/dev/null || true
cp -r init_scripts $TEMP_DIR/ 2>/dev/null || true
cp -r config $TEMP_DIR/ 2>/dev/null || true
cp -r intelowl $TEMP_DIR/ 2>/dev/null || true
cp -r airflow $TEMP_DIR/ 2>/dev/null || true
cp .env $TEMP_DIR/ 2>/dev/null || true

# Create deployment script for THANOS
cat << 'REMOTE_SCRIPT' > $TEMP_DIR/deploy_on_thanos.sh
#!/bin/bash
cd /opt/bev

# Set Vault address to STARLORD's IP
export VAULT_ADDR=http://100.122.12.35:8200

echo "Stopping any existing services..."
docker-compose -f docker-compose-thanos-unified.yml down 2>/dev/null || true

echo "Starting THANOS services..."
docker-compose -f docker-compose-thanos-unified.yml up -d

echo "Waiting for services to stabilize..."
sleep 10

echo "Service status:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep bev_ || echo "No services running"
REMOTE_SCRIPT

chmod +x $TEMP_DIR/deploy_on_thanos.sh

# Copy files to THANOS
echo -e "${BLUE}→ Copying files to THANOS...${NC}"
ssh $THANOS_HOST "sudo mkdir -p $DEPLOYMENT_DIR && sudo chown \$USER:\$USER $DEPLOYMENT_DIR"
scp -r $TEMP_DIR/* $THANOS_HOST:$DEPLOYMENT_DIR/

# Execute deployment on THANOS
echo -e "${PURPLE}→ DEPLOYING SERVICES ON THANOS (REMOTE)...${NC}"
ssh -t $THANOS_HOST "cd $DEPLOYMENT_DIR && bash deploy_on_thanos.sh"

# Cleanup
rm -rf $TEMP_DIR

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ THANOS DEPLOYMENT COMPLETE${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Services deployed to THANOS (${THANOS_IP}):${NC}"
echo -e "  • PostgreSQL, Neo4j, Elasticsearch, InfluxDB"
echo -e "  • Kafka cluster, RabbitMQ cluster, Redis cluster"
echo -e "  • IntelOwl + Celery workers"
echo -e "  • AI/ML services (GPU-accelerated)"
echo -e "  • Airflow orchestration"
echo -e "  • Document analyzers"
echo -e "  • Swarm intelligence"
echo ""
echo -e "${BLUE}Access points:${NC}"
echo -e "  • Neo4j: http://${THANOS_IP}:7474"
echo -e "  • Elasticsearch: http://${THANOS_IP}:9200"
echo -e "  • RabbitMQ: http://${THANOS_IP}:15672"
echo -e "  • Airflow: http://${THANOS_IP}:8080"
echo -e "  • IntelOwl: http://${THANOS_IP}:80"