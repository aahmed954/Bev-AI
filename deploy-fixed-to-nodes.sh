#!/bin/bash
# BEV MULTINODE DEPLOYMENT FROM GITHUB
# Deploys the FIXED project to THANOS and ORACLE1

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🚀 BEV MULTINODE DEPLOYMENT${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""
echo "This will deploy the FIXED BEV project to:"
echo "  • THANOS (Local): Heavy compute, databases, GPU services"
echo "  • ORACLE1 (Cloud): Lightweight services, monitoring"
echo ""

# Step 1: Deploy to THANOS
echo -e "${CYAN}Step 1: Deploying to THANOS${NC}"

ssh starlord@thanos << 'THANOS_SCRIPT'
set -e

# Clean and prepare
echo "Preparing THANOS for deployment..."
sudo mkdir -p /opt/bev
sudo chown starlord:starlord /opt/bev
cd /opt

# Clone or update from GitHub
if [ -d "bev" ]; then
    echo "Updating existing repository..."
    cd bev
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/aahmed954/Bev-AI.git bev
    cd bev
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --user docker-compose

# Start core services only (not all 151 services at once!)
echo "Starting core THANOS services..."
docker-compose -f docker-compose-thanos-unified.yml up -d \
    postgres neo4j redis_1 elasticsearch kafka_1 rabbitmq_1

echo "THANOS deployment complete!"
THANOS_SCRIPT

# Step 2: Deploy to ORACLE1
echo -e "${CYAN}Step 2: Deploying to ORACLE1${NC}"

ssh starlord@oracle1 << 'ORACLE1_SCRIPT'
set -e

# Clean and prepare
echo "Preparing ORACLE1 for deployment..."
sudo mkdir -p /opt/bev
sudo chown starlord:starlord /opt/bev
cd /opt

# Clone or update from GitHub
if [ -d "bev" ]; then
    echo "Updating existing repository..."
    cd bev
    git pull origin main
else
    echo "Cloning repository..."
    git clone https://github.com/aahmed954/Bev-AI.git bev
    cd bev
fi

# Install Python dependencies (ARM compatible)
echo "Installing Python dependencies..."
pip3 install --user docker-compose

# Start monitoring services (ARM optimized)
echo "Starting ORACLE1 monitoring services..."
docker-compose -f docker-compose-oracle1-unified.yml up -d \
    redis_oracle prometheus grafana consul

echo "ORACLE1 deployment complete!"
ORACLE1_SCRIPT

# Step 3: Verify deployment
echo -e "${CYAN}Step 3: Verifying Deployment${NC}"

echo -e "${BLUE}THANOS Services:${NC}"
ssh starlord@thanos "docker ps --format 'table {{.Names}}\t{{.Status}}' | grep bev || echo 'No services running yet'"

echo -e "${BLUE}ORACLE1 Services:${NC}"
ssh starlord@oracle1 "docker ps --format 'table {{.Names}}\t{{.Status}}' | grep bev || echo 'No services running yet'"

# Summary
echo ""
echo -e "${GREEN}🎉 DEPLOYMENT COMPLETE!${NC}"
echo ""
echo -e "${YELLOW}Access Points:${NC}"
echo "  • PostgreSQL: thanos:5432"
echo "  • Neo4j Browser: http://thanos:7474"
echo "  • Grafana: http://oracle1:3000"
echo "  • Prometheus: http://oracle1:9090"
echo ""
echo -e "${CYAN}To start more services gradually:${NC}"
echo "  ssh starlord@thanos 'cd /opt/bev && docker-compose -f docker-compose-thanos-unified.yml up -d'"
echo "  ssh starlord@oracle1 'cd /opt/bev && docker-compose -f docker-compose-oracle1-unified.yml up -d'"
echo ""
echo -e "${YELLOW}Monitor logs:${NC}"
echo "  ssh starlord@thanos 'cd /opt/bev && docker-compose -f docker-compose-thanos-unified.yml logs -f'"
echo "  ssh starlord@oracle1 'cd /opt/bev && docker-compose -f docker-compose-oracle1-unified.yml logs -f'"
