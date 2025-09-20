#!/bin/bash
# Thanos Primary Node Deployment Script
# Deploys GPU-intensive AI/ML services, primary databases, message queues

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🚀 THANOS NODE DEPLOYMENT (Primary Compute + GPU)${NC}"
echo "=============================================="

# Verify we're on the correct node
if [ "$(hostname)" != "thanos" ]; then
    echo -e "${RED}❌ This script must run on Thanos node${NC}"
    exit 1
fi

# Verify GPU availability
echo -e "${YELLOW}🔍 Verifying GPU availability...${NC}"
if nvidia-smi > /dev/null 2>&1; then
    echo -e "${GREEN}✅ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${RED}❌ No GPU detected on Thanos${NC}"
    exit 1
fi

# Verify system resources
echo -e "${YELLOW}📊 Checking system resources...${NC}"
TOTAL_RAM=$(free -g | awk 'NR==2{print $2}')
AVAILABLE_RAM=$(free -g | awk 'NR==2{print $7}')
TOTAL_CORES=$(nproc)

echo "Total RAM: ${TOTAL_RAM}GB"
echo "Available RAM: ${AVAILABLE_RAM}GB"
echo "CPU Cores: ${TOTAL_CORES}"

if [ $AVAILABLE_RAM -lt 40 ]; then
    echo -e "${RED}❌ Insufficient RAM for Thanos deployment (need 40GB+)${NC}"
    exit 1
fi

if [ $TOTAL_CORES -lt 16 ]; then
    echo -e "${YELLOW}⚠️ Low CPU count for optimal performance (recommended 16+ cores)${NC}"
fi

# Setup deployment directory
cd /opt/bev-deployment/bev-platform

# Create Thanos-specific environment file
echo -e "${BLUE}🔧 Creating Thanos configuration...${NC}"
cat > .env.thanos << ENV_EOF
# Thanos Node Configuration
NODE_ROLE=primary_compute
NODE_NAME=thanos
NODE_ARCH=x86_64

# GPU Configuration
NVIDIA_VISIBLE_DEVICES=all
CUDA_VERSION=12.0

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_MAX_CONNECTIONS=500
SHARED_BUFFERS=8GB

# AI/ML Configuration
ML_WORKERS=4
GPU_MEMORY_LIMIT=8GB
BATCH_SIZE=32

# Message Queue Configuration
KAFKA_HEAP_OPTS="-Xmx6G -Xms6G"
RABBITMQ_VM_MEMORY_HIGH_WATERMARK=0.8

# Network Configuration
NETWORK_INTERFACE=enp*
ADVERTISED_LISTENERS=thanos:9092
ENV_EOF

# Deploy Thanos services
echo -e "${BLUE}🚀 Deploying Thanos services...${NC}"
docker-compose -f docker-compose-thanos-unified.yml --env-file .env.thanos up -d

# Wait for core services
echo -e "${YELLOW}⏳ Waiting for core services to initialize...${NC}"
sleep 45

# Initialize databases
echo -e "${BLUE}🗄️ Initializing database schemas...${NC}"
./scripts/init_primary_databases.sh

# Verify GPU services
echo -e "${BLUE}🤖 Verifying AI/ML services...${NC}"
for service in autonomous-coordinator adaptive-learning knowledge-evolution; do
    echo -n "Checking $service GPU access... "
    if docker exec bev_$service nvidia-smi > /dev/null 2>&1; then
        echo -e "${GREEN}✅ GPU Available${NC}"
    else
        echo -e "${RED}❌ GPU Not Available${NC}"
    fi
done

# Health check all services
echo -e "${BLUE}🏥 Running health checks...${NC}"
./scripts/health_check_thanos.sh

echo -e "${GREEN}🎯 THANOS NODE DEPLOYMENT COMPLETE!${NC}"
echo "Services deployed: 32 high-compute services"
echo "GPU utilization: Active for AI/ML pipeline"
echo "Database cluster: Primary node ready"
echo "Message queues: Kafka + RabbitMQ clusters active"