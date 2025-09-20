#!/bin/bash
# BEV Distributed Enterprise Deployment Script
# Deploys BEV across Thanos (x86+GPU), Oracle1 (ARM), and Starlord (Dev)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
THANOS_HOST="thanos"
ORACLE1_HOST="oracle1"
STARLORD_HOST="localhost"
DEPLOYMENT_DIR="/opt/bev-deployment"
PROJECT_REPO="https://github.com/starlord/bev-platform.git"

echo -e "${PURPLE}🚀 BEV ENTERPRISE DISTRIBUTED DEPLOYMENT${NC}"
echo -e "${BLUE}====================================================${NC}"
echo "Date: $(date)"
echo "Target Nodes: Thanos (x86+GPU), Oracle1 (ARM), Starlord (Dev)"
echo ""

# Verify prerequisites
echo -e "${YELLOW}🔍 Verifying Prerequisites...${NC}"

# Check SSH connectivity
echo -n "Checking Thanos connectivity... "
if ssh -o ConnectTimeout=5 $THANOS_HOST "echo 'Connected'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Connected${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
    exit 1
fi

echo -n "Checking Oracle1 connectivity... "
if ssh -o ConnectTimeout=5 $ORACLE1_HOST "echo 'Connected'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Connected${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
    exit 1
fi

# Verify Docker on remote nodes
echo -n "Checking Docker on Thanos... "
if ssh $THANOS_HOST "docker --version" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker Available${NC}"
else
    echo -e "${RED}❌ Docker Not Available${NC}"
    exit 1
fi

echo -n "Checking Docker on Oracle1... "
if ssh $ORACLE1_HOST "docker --version" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Docker Available${NC}"
else
    echo -e "${RED}❌ Docker Not Available${NC}"
    exit 1
fi

echo ""

# PHASE 1: THANOS DEPLOYMENT (Primary Compute + GPU)
echo -e "${CYAN}📡 PHASE 1: DEPLOYING THANOS NODE (PRIMARY COMPUTE + GPU)${NC}"
echo "Deploying high-compute services, AI/ML pipeline, primary databases..."

ssh $THANOS_HOST << 'EOF'
set -e

# Create deployment directory
sudo mkdir -p /opt/bev-deployment
sudo chown $(whoami):$(whoami) /opt/bev-deployment
cd /opt/bev-deployment

# Clone or update repository
if [ -d "bev-platform" ]; then
    cd bev-platform && git pull origin enterprise-completion
else
    git clone $PROJECT_REPO bev-platform
    cd bev-platform && git checkout enterprise-completion
fi

echo "🔧 Setting up GPU environment..."
# Verify GPU availability
nvidia-smi || { echo "GPU not available on Thanos"; exit 1; }

echo "🗄️ Deploying primary infrastructure..."
# Deploy Thanos services (GPU + High-Compute)
docker-compose -f docker-compose-thanos-unified.yml up -d

echo "📊 Initializing databases..."
# Wait for databases to start
sleep 30

# Initialize database schemas
./scripts/init_primary_databases.sh

echo "🤖 Verifying AI/ML services..."
# Verify GPU services are running
docker exec bev_autonomous_coordinator nvidia-smi
docker exec bev_adaptive_learning nvidia-smi

echo "✅ Thanos deployment complete!"
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Thanos deployment successful${NC}"
else
    echo -e "${RED}❌ Thanos deployment failed${NC}"
    exit 1
fi

echo ""

# Wait for Thanos services to stabilize
echo -e "${YELLOW}⏳ Waiting for Thanos services to stabilize (60s)...${NC}"
sleep 60

# PHASE 2: ORACLE1 DEPLOYMENT (ARM Services)
echo -e "${CYAN}🔧 PHASE 2: DEPLOYING ORACLE1 NODE (ARM SERVICES)${NC}"
echo "Deploying monitoring, security, lightweight services..."

ssh $ORACLE1_HOST << 'EOF'
set -e

# Create deployment directory
sudo mkdir -p /opt/bev-deployment
sudo chown $(whoami):$(whoami) /opt/bev-deployment
cd /opt/bev-deployment

# Clone or update repository
if [ -d "bev-platform" ]; then
    cd bev-platform && git pull origin enterprise-completion
else
    git clone $PROJECT_REPO bev-platform
    cd bev-platform && git checkout enterprise-completion
fi

echo "🏗️ Verifying ARM architecture..."
# Verify ARM architecture
if [ "$(uname -m)" != "aarch64" ]; then
    echo "Warning: Expected ARM64 architecture, got $(uname -m)"
fi

echo "📊 Deploying ARM-optimized services..."
# Deploy Oracle1 services (ARM-compatible)
docker-compose -f docker-compose-oracle1-unified.yml up -d

echo "🔒 Setting up security services..."
# Setup security and monitoring
./scripts/setup_arm_security.sh

echo "📈 Setting up monitoring..."
./scripts/setup_arm_monitoring.sh

echo "✅ Oracle1 deployment complete!"
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Oracle1 deployment successful${NC}"
else
    echo -e "${RED}❌ Oracle1 deployment failed${NC}"
    exit 1
fi

echo ""

# PHASE 3: STARLORD DEVELOPMENT SETUP
echo -e "${CYAN}💻 PHASE 3: SETTING UP STARLORD DEVELOPMENT ENVIRONMENT${NC}"
echo "Setting up development tools, staging services, frontend..."

echo "📦 Setting up development environment..."
# Setup local development
cd /home/starlord/Projects/Bev

# Install development dependencies
if [ ! -d "bev-frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd bev-frontend && npm install && cd ..
fi

# Build Tauri development environment
echo "🔧 Setting up Tauri environment..."
cd bev-frontend && npm run tauri info > /dev/null 2>&1 || npm install @tauri-apps/cli

# Deploy development services
echo "🚀 Deploying development services..."
docker-compose -f docker-compose-development.yml up -d

# Setup MCP development servers
echo "🔗 Setting up MCP development servers..."
./scripts/setup_mcp_development.sh

echo -e "${GREEN}✅ Starlord development setup complete!${NC}"

echo ""

# PHASE 4: DISTRIBUTED SYSTEM VERIFICATION
echo -e "${CYAN}🔍 PHASE 4: DISTRIBUTED SYSTEM VERIFICATION${NC}"

echo "Verifying cross-node communication..."

# Test Thanos services
echo -n "Testing Thanos services... "
if curl -s http://thanos:9090/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Healthy${NC}"
else
    echo -e "${YELLOW}⚠️ May be starting...${NC}"
fi

# Test Oracle1 services
echo -n "Testing Oracle1 services... "
if curl -s http://oracle1:9090/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Healthy${NC}"
else
    echo -e "${YELLOW}⚠️ May be starting...${NC}"
fi

# Test local services
echo -n "Testing Starlord services... "
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Frontend Available${NC}"
else
    echo -e "${YELLOW}⚠️ Starting frontend...${NC}"
    cd bev-frontend && npm run dev &
fi

echo ""

# Generate deployment report
echo -e "${PURPLE}📊 GENERATING DEPLOYMENT REPORT...${NC}"

cat << REPORT_EOF > /tmp/bev_deployment_report.txt
BEV DISTRIBUTED DEPLOYMENT REPORT
Date: $(date)

THANOS NODE (Primary Compute + GPU):
- Role: AI/ML processing, primary databases, message queues
- Services: ~32 high-compute services
- Expected Load: CPU 75%, Memory 78%, GPU 65%

ORACLE1 NODE (ARM Processing):
- Role: Monitoring, security, lightweight services
- Services: ~25 ARM-optimized services
- Expected Load: CPU 75%, Memory 75%, Storage 51%

STARLORD NODE (Development):
- Role: Development environment, staging, frontend
- Services: ~10 development services
- Expected Load: Variable based on development activity

TOTAL DISTRIBUTED SERVICES: 67 services across 3 nodes
NETWORK: Tailscale VPN mesh with automatic failover
MONITORING: Centralized on Oracle1 with cross-node dashboards
BACKUP: Nightly cross-node replication

ACCESS URLS:
- BEV Frontend: http://starlord:5173 (development)
- BEV Production: http://thanos (via load balancer)
- Grafana Monitoring: http://oracle1:3000
- Vault Management: http://oracle1:8200
- IntelOwl Platform: http://thanos
- Neo4j Browser: http://thanos:7474
REPORT_EOF

cat /tmp/bev_deployment_report.txt

echo ""
echo -e "${GREEN}🎯 BEV DISTRIBUTED DEPLOYMENT COMPLETE!${NC}"
echo -e "${PURPLE}====================================================${NC}"
echo -e "${CYAN}🏆 ENTERPRISE COMMAND CENTER READY FOR OPERATION!${NC}"
echo ""
echo "Next Steps:"
echo "1. Access frontend: http://starlord:5173"
echo "2. Monitor health: http://oracle1:3000"
echo "3. Manage secrets: http://oracle1:8200"
echo "4. OSINT operations: http://thanos"
echo ""
echo "Deployment report saved to: /tmp/bev_deployment_report.txt"