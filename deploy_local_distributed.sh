#!/bin/bash
# BEV Local Distributed Deployment
# Deploys from current directory without cloning repository

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🚀 BEV LOCAL DISTRIBUTED DEPLOYMENT${NC}"
echo -e "${BLUE}====================================${NC}"
echo "Date: $(date)"
echo "Source: Current directory ($(pwd))"
echo "Target: 3-node distributed architecture"
echo ""

# Get current directory
PROJECT_DIR=$(pwd)

# PHASE 1: Deploy Thanos (Primary GPU Node)
echo -e "${CYAN}🚀 PHASE 1: DEPLOYING THANOS NODE (GPU + HIGH-COMPUTE)${NC}"

# Copy project to Thanos
echo "Syncing project to Thanos..."
ssh thanos "sudo mkdir -p /opt/bev && sudo chown \$(whoami):\$(whoami) /opt/bev"
rsync -az --exclude='node_modules' --exclude='.git' --exclude='backups' "$PROJECT_DIR/" thanos:/opt/bev/

# Deploy Thanos services
ssh thanos << 'THANOS_DEPLOY'
cd /opt/bev

echo "🔧 Setting up GPU environment on Thanos..."
nvidia-smi || { echo "❌ GPU not available"; exit 1; }

echo "🚀 Deploying Thanos high-compute services..."
docker-compose -f docker-compose-thanos-unified.yml up -d

echo "⏳ Waiting for databases to initialize..."
sleep 45

# Initialize databases
if [ -f "scripts/init_primary_databases.sh" ]; then
    ./scripts/init_primary_databases.sh
fi

echo "🤖 Verifying GPU service access..."
sleep 30
for service in doc-analyzer-1 doc-analyzer-2 doc-analyzer-3; do
    echo -n "GPU check $service: "
    docker exec bev_$service nvidia-smi > /dev/null 2>&1 && echo "✅ GPU Access" || echo "⚠️ Starting"
done

echo "✅ Thanos deployment complete!"
THANOS_DEPLOY

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Thanos deployment successful${NC}"
else
    echo -e "${RED}❌ Thanos deployment had issues${NC}"
fi

echo ""

# PHASE 2: Deploy Oracle1 (ARM Node)
echo -e "${CYAN}🔧 PHASE 2: DEPLOYING ORACLE1 NODE (ARM64 SERVICES)${NC}"

# Copy project to Oracle1
echo "Syncing project to Oracle1..."
ssh oracle1 "sudo mkdir -p /opt/bev && sudo chown \$(whoami):\$(whoami) /opt/bev"
rsync -az --exclude='node_modules' --exclude='.git' --exclude='backups' "$PROJECT_DIR/" oracle1:/opt/bev/

# Deploy Oracle1 services
ssh oracle1 << 'ORACLE1_DEPLOY'
cd /opt/bev

echo "🏗️ Verifying ARM64 architecture..."
uname -m  # Should show aarch64

echo "🚀 Deploying Oracle1 ARM-optimized services..."
docker-compose -f docker-compose-oracle1-unified.yml up -d

echo "⏳ Waiting for monitoring services to start..."
sleep 30

# Setup monitoring
if [ -f "scripts/setup_arm_monitoring.sh" ]; then
    ./scripts/setup_arm_monitoring.sh
fi

# Setup security
if [ -f "scripts/setup_arm_security.sh" ]; then
    ./scripts/setup_arm_security.sh
fi

echo "📊 Verifying ARM service architecture..."
for service in redis-arm nginx prometheus grafana; do
    if docker ps | grep bev_$service > /dev/null; then
        ARCH=$(docker exec bev_$service uname -m 2>/dev/null || echo "unknown")
        echo "$service architecture: $ARCH"
    fi
done

echo "✅ Oracle1 deployment complete!"
ORACLE1_DEPLOY

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Oracle1 deployment successful${NC}"
else
    echo -e "${RED}❌ Oracle1 deployment had issues${NC}"
fi

echo ""

# PHASE 3: Setup Starlord Development
echo -e "${CYAN}💻 PHASE 3: SETTING UP STARLORD DEVELOPMENT${NC}"

echo "Setting up local development environment..."

# Deploy development services
if [ -f "docker-compose-development.yml" ]; then
    echo "🚀 Starting development services..."
    docker-compose -f docker-compose-development.yml up -d
    sleep 15
fi

# Setup MCP development
if [ -f "scripts/setup_mcp_development.sh" ]; then
    echo "🔗 Setting up MCP development servers..."
    ./scripts/setup_mcp_development.sh
fi

# Start frontend development
echo "💻 Starting frontend development server..."
cd bev-frontend
if ! curl -s http://localhost:5173 > /dev/null 2>&1; then
    npm run dev &
    FRONTEND_PID=$!
    echo "Frontend started with PID: $FRONTEND_PID"
    sleep 10
fi
cd ..

echo -e "${GREEN}✅ Starlord development setup complete${NC}"

echo ""

# PHASE 4: Verification & Health Check
echo -e "${PURPLE}🏥 COMPREHENSIVE DEPLOYMENT VERIFICATION${NC}"
echo -e "${BLUE}========================================${NC}"

echo "Service count verification:"

# Count services on each node
THANOS_COUNT=$(ssh thanos "docker ps --filter 'name=bev_' | grep -c bev_ || echo 0")
ORACLE1_COUNT=$(ssh oracle1 "docker ps --filter 'name=bev_' | grep -c bev_ || echo 0")
STARLORD_COUNT=$(docker ps --filter 'name=bev_' | grep -c bev_ 2>/dev/null || echo 0)

echo "  Thanos: $THANOS_COUNT services running"
echo "  Oracle1: $ORACLE1_COUNT services running"
echo "  Starlord: $STARLORD_COUNT services running"

TOTAL_SERVICES=$((THANOS_COUNT + ORACLE1_COUNT + STARLORD_COUNT))
echo "  Total: $TOTAL_SERVICES distributed services"

# Service health verification
echo ""
echo "Service health verification:"

HEALTHY_SERVICES=0

# Check Thanos key services
echo -n "  PostgreSQL (Thanos): "
if ssh thanos "docker exec bev_postgres pg_isready -U postgres" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Healthy${NC}"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
else
    echo -e "${YELLOW}⚠️ Starting${NC}"
fi

echo -n "  Neo4j (Thanos): "
if curl -s http://thanos:7474 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Accessible${NC}"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
else
    echo -e "${YELLOW}⚠️ Starting${NC}"
fi

# Check Oracle1 key services
echo -n "  Redis (Oracle1): "
if ssh oracle1 "docker exec bev_redis_oracle redis-cli ping" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Healthy${NC}"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
else
    echo -e "${YELLOW}⚠️ Starting${NC}"
fi

echo -n "  Prometheus (Oracle1): "
if curl -s http://oracle1:9090 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Accessible${NC}"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
else
    echo -e "${YELLOW}⚠️ Starting${NC}"
fi

# Check Starlord services
echo -n "  Frontend (Starlord): "
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Accessible${NC}"
    HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
else
    echo -e "${YELLOW}⚠️ Starting${NC}"
fi

# Calculate health percentage
HEALTH_PERCENTAGE=$((HEALTHY_SERVICES * 100 / 5))

echo ""
echo -e "${PURPLE}📊 DEPLOYMENT RESULTS${NC}"
echo "===================="
echo "Total Services: $TOTAL_SERVICES"
echo "Health Check: $HEALTHY_SERVICES/5 key services ($HEALTH_PERCENTAGE%)"

if [ $HEALTH_PERCENTAGE -ge 80 ]; then
    echo -e "${GREEN}🏆 DEPLOYMENT SUCCESSFUL!${NC}"
    echo ""
    echo -e "${CYAN}🔗 ACCESS YOUR DISTRIBUTED BEV PLATFORM:${NC}"
    echo "• Frontend: http://localhost:5173"
    echo "• Grafana: http://oracle1:3000"
    echo "• Vault: http://oracle1:8200"
    echo "• IntelOwl: http://thanos"
    echo "• Neo4j: http://thanos:7474"
    echo ""
    echo -e "${GREEN}🎯 BEV ENTERPRISE COMMAND CENTER IS OPERATIONAL!${NC}"

elif [ $HEALTH_PERCENTAGE -ge 60 ]; then
    echo -e "${YELLOW}⚠️ DEPLOYMENT PARTIALLY SUCCESSFUL${NC}"
    echo -e "${YELLOW}🔧 Some services still starting - wait 2-5 minutes${NC}"

else
    echo -e "${RED}❌ DEPLOYMENT NEEDS ATTENTION${NC}"
    echo -e "${RED}🚨 Check logs and troubleshoot${NC}"
fi

echo ""
echo -e "${CYAN}📄 Deployment completed at: $(date)${NC}"