#!/bin/bash
# Thanos Node Health Check Script
# Validates GPU services, databases, and high-compute workloads

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🏥 THANOS NODE HEALTH CHECK${NC}"
echo "============================"

# GPU Health Check
echo -e "${BLUE}🎮 Checking GPU availability...${NC}"
if nvidia-smi > /dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader,nounits)
    echo -e "${GREEN}✅ GPU: $GPU_INFO${NC}"
else
    echo -e "${RED}❌ GPU not available${NC}"
    exit 1
fi

# Check GPU services
echo -e "${BLUE}🤖 Checking AI/ML GPU services...${NC}"
GPU_SERVICES=("autonomous-coordinator" "adaptive-learning" "knowledge-evolution" "extended-reasoning")
HEALTHY_GPU=0

for service in "${GPU_SERVICES[@]}"; do
    echo -n "GPU check $service... "
    if docker exec bev_$service nvidia-smi > /dev/null 2>&1; then
        echo -e "${GREEN}✅ GPU Access${NC}"
        HEALTHY_GPU=$((HEALTHY_GPU + 1))
    else
        echo -e "${RED}❌ No GPU Access${NC}"
    fi
done

# Database Health Checks
echo -e "${BLUE}🗄 Checking primary databases...${NC}"
DATABASE_SERVICES=(
    "postgres:pg_isready -U researcher"
    "neo4j:cypher-shell -u neo4j -p BevGraphMaster2024 'RETURN 1'"
    "redis:redis-cli ping"
    "elasticsearch:curl -s http://localhost:9200/_cluster/health"
    "influxdb:curl -s http://localhost:8086/health"
    "qdrant-primary:curl -s http://localhost:6333/health"
    "weaviate:curl -s http://localhost:8080/v1/meta"
)

HEALTHY_DBS=0

for check in "${DATABASE_SERVICES[@]}"; do
    DB_NAME=$(echo $check | cut -d: -f1)
    DB_CHECK=$(echo $check | cut -d: -f2-)

    echo -n "Database check $DB_NAME... "
    if docker exec bev_$DB_NAME bash -c "$DB_CHECK" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
        HEALTHY_DBS=$((HEALTHY_DBS + 1))
    else
        echo -e "${RED}❌ Unhealthy${NC}"
    fi
done

# Message Queue Health Checks
echo -e "${BLUE}📨 Checking message queues...${NC}"
MQ_SERVICES=("kafka-1:9092" "kafka-2:9092" "kafka-3:9092" "rabbitmq-1:5672" "rabbitmq-2:5672" "rabbitmq-3:5672")
HEALTHY_MQ=0

for service in "${MQ_SERVICES[@]}"; do
    SERVICE_NAME=$(echo $service | cut -d: -f1)
    SERVICE_PORT=$(echo $service | cut -d: -f2)

    echo -n "Message queue check $SERVICE_NAME... "
    if docker exec bev_$SERVICE_NAME netstat -tln | grep $SERVICE_PORT > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Listening${NC}"
        HEALTHY_MQ=$((HEALTHY_MQ + 1))
    else
        echo -e "${RED}❌ Not Listening${NC}"
    fi
done

# AI/ML Pipeline Health
echo -e "${BLUE}🧠 Checking AI/ML pipeline...${NC}"
AI_SERVICES=("autonomous-coordinator:8009" "adaptive-learning:8010" "resource-manager:8011" "knowledge-evolution:8012")
HEALTHY_AI=0

for service in "${AI_SERVICES[@]}"; do
    SERVICE_NAME=$(echo $service | cut -d: -f1)
    SERVICE_PORT=$(echo $service | cut -d: -f2)

    echo -n "AI service check $SERVICE_NAME... "
    if curl -s http://localhost:$SERVICE_PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
        HEALTHY_AI=$((HEALTHY_AI + 1))
    else
        echo -e "${YELLOW}⚠️ Starting...${NC}"
    fi
done

# Resource Usage Check
echo -e "${BLUE}📊 Checking resource utilization...${NC}"
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEM_USAGE=$(free | awk 'NR==2{printf "%.1f", $3*100/$2 }')
DISK_USAGE=$(df / | awk 'NR==2{print $5}' | cut -d'%' -f1)

echo "CPU Usage: ${CPU_USAGE}%"
echo "Memory Usage: ${MEM_USAGE}%"
echo "Disk Usage: ${DISK_USAGE}%"

# GPU Usage Check
if nvidia-smi > /dev/null 2>&1; then
    GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU_MEMORY=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits)
    echo "GPU Usage: ${GPU_USAGE}%"
    echo "GPU Memory: ${GPU_MEMORY}%"
fi

# Health Summary
echo ""
echo -e "${BLUE}📊 THANOS HEALTH SUMMARY:${NC}"
echo "GPU Services: $HEALTHY_GPU/${#GPU_SERVICES[@]}"
echo "Databases: $HEALTHY_DBS/${#DATABASE_SERVICES[@]}"
echo "Message Queues: $HEALTHY_MQ/${#MQ_SERVICES[@]}"
echo "AI Services: $HEALTHY_AI/${#AI_SERVICES[@]}"

# Calculate overall health
TOTAL_SERVICES=$((${#GPU_SERVICES[@]} + ${#DATABASE_SERVICES[@]} + ${#MQ_SERVICES[@]} + ${#AI_SERVICES[@]}))
TOTAL_HEALTHY=$((HEALTHY_GPU + HEALTHY_DBS + HEALTHY_MQ + HEALTHY_AI))
HEALTH_PERCENTAGE=$((TOTAL_HEALTHY * 100 / TOTAL_SERVICES))

echo "Overall Health: $TOTAL_HEALTHY/$TOTAL_SERVICES ($HEALTH_PERCENTAGE%)"

if [ $HEALTH_PERCENTAGE -ge 85 ]; then
    echo -e "${GREEN}🎯 Thanos node is healthy and ready!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ Thanos node needs attention${NC}"
    exit 1
fi