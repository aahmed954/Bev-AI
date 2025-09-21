#!/bin/bash
# Distributed BEV Platform Health Check
# Validates all 3 nodes and cross-node communication

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🏥 BEV DISTRIBUTED PLATFORM HEALTH CHECK${NC}"
echo -e "${BLUE}===========================================${NC}"
echo "Date: $(date)"
echo ""

# Initialize counters
TOTAL_SERVICES=0
HEALTHY_SERVICES=0
CRITICAL_SERVICES=0

# PHASE 1: THANOS NODE HEALTH CHECK
echo -e "${CYAN}🎮 THANOS NODE (GPU + HIGH-COMPUTE) HEALTH CHECK${NC}"
echo "================================================="

# Check Thanos connectivity
echo -n "Thanos SSH connectivity... "
if ssh -o ConnectTimeout=5 thanos "echo 'Connected'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Connected${NC}"
    THANOS_ACCESSIBLE=true
else
    echo -e "${RED}❌ Not accessible${NC}"
    THANOS_ACCESSIBLE=false
fi

if [ "$THANOS_ACCESSIBLE" = true ]; then
    # GPU Health Check
    echo -e "${BLUE}🤖 Checking GPU services...${NC}"
    THANOS_GPU_HEALTHY=0
    GPU_SERVICES=("autonomous-coordinator" "adaptive-learning" "knowledge-evolution" "extended-reasoning")

    for service in "${GPU_SERVICES[@]}"; do
        echo -n "GPU check $service... "
        if ssh thanos "docker exec bev_$service nvidia-smi > /dev/null 2>&1"; then
            echo -e "${GREEN}✅ GPU Access${NC}"
            THANOS_GPU_HEALTHY=$((THANOS_GPU_HEALTHY + 1))
        else
            echo -e "${RED}❌ No GPU Access${NC}"
        fi
    done

    # Database Health Checks
    echo -e "${BLUE}🗄 Checking primary databases...${NC}"
    THANOS_DB_HEALTHY=0
    DATABASE_CHECKS=(
        "postgres:pg_isready -U researcher"
        "neo4j:curl -s http://localhost:7474"
        "redis:redis-cli ping"
        "elasticsearch:curl -s http://localhost:9200/_cluster/health"
    )

    for check in "${DATABASE_CHECKS[@]}"; do
        DB_NAME=$(echo $check | cut -d: -f1)
        DB_CHECK=$(echo $check | cut -d: -f2-)

        echo -n "Database check $DB_NAME... "
        if ssh thanos "docker exec bev_$DB_NAME bash -c \"$DB_CHECK\" > /dev/null 2>&1"; then
            echo -e "${GREEN}✅ Healthy${NC}"
            THANOS_DB_HEALTHY=$((THANOS_DB_HEALTHY + 1))
            CRITICAL_SERVICES=$((CRITICAL_SERVICES + 1))
        else
            echo -e "${RED}❌ Unhealthy${NC}"
        fi
    done

    # Count total Thanos services
    THANOS_SERVICE_COUNT=$(ssh thanos "docker ps --filter 'name=bev_' | grep -c bev_ || echo 0")
    echo "Thanos services running: $THANOS_SERVICE_COUNT"
    TOTAL_SERVICES=$((TOTAL_SERVICES + THANOS_SERVICE_COUNT))

else
    echo -e "${RED}❌ Skipping Thanos checks - node not accessible${NC}"
fi

echo ""

# PHASE 2: ORACLE1 NODE HEALTH CHECK
echo -e "${CYAN}🔧 ORACLE1 NODE (ARM MONITORING) HEALTH CHECK${NC}"
echo "=============================================="

# Check Oracle1 connectivity
echo -n "Oracle1 SSH connectivity... "
if ssh -o ConnectTimeout=5 oracle1 "echo 'Connected'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Connected${NC}"
    ORACLE1_ACCESSIBLE=true
else
    echo -e "${RED}❌ Not accessible${NC}"
    ORACLE1_ACCESSIBLE=false
fi

if [ "$ORACLE1_ACCESSIBLE" = true ]; then
    # ARM Architecture Verification
    ORACLE1_ARCH=$(ssh oracle1 "uname -m")
    echo "Oracle1 architecture: $ORACLE1_ARCH"

    # Monitoring Services Health
    echo -e "${BLUE}📊 Checking monitoring services...${NC}"
    ORACLE1_MONITORING_HEALTHY=0
    MONITORING_SERVICES=("prometheus:9090" "grafana:3000" "redis:6379")

    for service in "${MONITORING_SERVICES[@]}"; do
        SERVICE_NAME=$(echo $service | cut -d: -f1)
        SERVICE_PORT=$(echo $service | cut -d: -f2)

        echo -n "Monitoring check $SERVICE_NAME... "
        if ssh oracle1 "curl -s http://localhost:$SERVICE_PORT > /dev/null 2>&1"; then
            echo -e "${GREEN}✅ Healthy${NC}"
            ORACLE1_MONITORING_HEALTHY=$((ORACLE1_MONITORING_HEALTHY + 1))
            HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
        else
            echo -e "${RED}❌ Unhealthy${NC}"
        fi
    done

    # Security Services Health
    echo -e "${BLUE}🛡 Checking security services...${NC}"
    ORACLE1_SECURITY_HEALTHY=0
    SECURITY_SERVICES=("vault:8200" "consul:8500" "tor:9050")

    for service in "${SECURITY_SERVICES[@]}"; do
        SERVICE_NAME=$(echo $service | cut -d: -f1)
        SERVICE_PORT=$(echo $service | cut -d: -f2)

        echo -n "Security check $SERVICE_NAME... "
        if ssh oracle1 "docker ps | grep bev_$SERVICE_NAME > /dev/null 2>&1"; then
            echo -e "${GREEN}✅ Running${NC}"
            ORACLE1_SECURITY_HEALTHY=$((ORACLE1_SECURITY_HEALTHY + 1))
            HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
        else
            echo -e "${RED}❌ Not Running${NC}"
        fi
    done

    # Count total Oracle1 services
    ORACLE1_SERVICE_COUNT=$(ssh oracle1 "docker ps --filter 'name=bev_' | grep -c bev_ || echo 0")
    echo "Oracle1 services running: $ORACLE1_SERVICE_COUNT"
    TOTAL_SERVICES=$((TOTAL_SERVICES + ORACLE1_SERVICE_COUNT))

else
    echo -e "${RED}❌ Skipping Oracle1 checks - node not accessible${NC}"
fi

echo ""

# PHASE 3: STARLORD NODE HEALTH CHECK
echo -e "${CYAN}💻 STARLORD NODE (DEVELOPMENT) HEALTH CHECK${NC}"
echo "==========================================="

# Development Services Health
echo -e "${BLUE}🚀 Checking development services...${NC}"
STARLORD_DEV_HEALTHY=0
DEV_SERVICES=("frontend:5173" "mcp-server:3010")

for service in "${DEV_SERVICES[@]}"; do
    SERVICE_NAME=$(echo $service | cut -d: -f1)
    SERVICE_PORT=$(echo $service | cut -d: -f2)

    echo -n "Development check $SERVICE_NAME... "
    if curl -s http://localhost:$SERVICE_PORT > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Accessible${NC}"
        STARLORD_DEV_HEALTHY=$((STARLORD_DEV_HEALTHY + 1))
        HEALTHY_SERVICES=$((HEALTHY_SERVICES + 1))
    else
        echo -e "${RED}❌ Not Accessible${NC}"
    fi
done

# Count total Starlord services
STARLORD_SERVICE_COUNT=$(docker ps --filter 'name=bev_' | grep -c bev_ 2>/dev/null || echo 0)
echo "Starlord services running: $STARLORD_SERVICE_COUNT"
TOTAL_SERVICES=$((TOTAL_SERVICES + STARLORD_SERVICE_COUNT))

echo ""

# PHASE 4: CROSS-NODE CONNECTIVITY CHECK
echo -e "${CYAN}🌐 CROSS-NODE CONNECTIVITY CHECK${NC}"
echo "================================="

# Test cross-node communication
CROSS_NODE_HEALTHY=0

echo -n "Starlord → Thanos (Neo4j)... "
if curl -s http://thanos:7474 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Connected${NC}"
    CROSS_NODE_HEALTHY=$((CROSS_NODE_HEALTHY + 1))
else
    echo -e "${RED}❌ Failed${NC}"
fi

echo -n "Starlord → Oracle1 (Grafana)... "
if curl -s http://oracle1:3000 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Connected${NC}"
    CROSS_NODE_HEALTHY=$((CROSS_NODE_HEALTHY + 1))
else
    echo -e "${RED}❌ Failed${NC}"
fi

echo -n "Oracle1 → Thanos federation... "
if [ "$ORACLE1_ACCESSIBLE" = true ]; then
    if ssh oracle1 "curl -s http://thanos:9090/metrics > /dev/null 2>&1"; then
        echo -e "${GREEN}✅ Federated${NC}"
        CROSS_NODE_HEALTHY=$((CROSS_NODE_HEALTHY + 1))
    else
        echo -e "${RED}❌ Not Federated${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ Oracle1 not accessible${NC}"
fi

echo ""

# PHASE 5: PERFORMANCE METRICS CHECK
echo -e "${CYAN}📊 PERFORMANCE METRICS CHECK${NC}"
echo "============================="

# Thanos performance
if [ "$THANOS_ACCESSIBLE" = true ]; then
    echo -e "${BLUE}Thanos Performance:${NC}"
    THANOS_CPU=$(ssh thanos "top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | cut -d'%' -f1" || echo "N/A")
    THANOS_MEM=$(ssh thanos "free | awk 'NR==2{printf \"%.1f\", \$3*100/\$2 }'" || echo "N/A")
    echo "  CPU Usage: ${THANOS_CPU}%"
    echo "  Memory Usage: ${THANOS_MEM}%"

    # GPU utilization
    if ssh thanos "nvidia-smi > /dev/null 2>&1"; then
        GPU_USAGE=$(ssh thanos "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits" || echo "N/A")
        echo "  GPU Usage: ${GPU_USAGE}%"
    fi
fi

# Oracle1 performance
if [ "$ORACLE1_ACCESSIBLE" = true ]; then
    echo -e "${BLUE}Oracle1 Performance:${NC}"
    ORACLE1_CPU=$(ssh oracle1 "top -bn1 | grep 'Cpu(s)' | awk '{print \$2}' | cut -d'%' -f1" || echo "N/A")
    ORACLE1_MEM=$(ssh oracle1 "free | awk 'NR==2{printf \"%.1f\", \$3*100/\$2 }'" || echo "N/A")
    echo "  CPU Usage: ${ORACLE1_CPU}%"
    echo "  Memory Usage: ${ORACLE1_MEM}%"
fi

# Starlord performance
echo -e "${BLUE}Starlord Performance:${NC}"
STARLORD_CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 || echo "N/A")
STARLORD_MEM=$(free | awk 'NR==2{printf "%.1f", $3*100/$2 }' || echo "N/A")
echo "  CPU Usage: ${STARLORD_CPU}%"
echo "  Memory Usage: ${STARLORD_MEM}%"

echo ""

# COMPREHENSIVE HEALTH SUMMARY
echo -e "${PURPLE}📊 COMPREHENSIVE HEALTH SUMMARY${NC}"
echo "==============================="

# Calculate health percentages
if [ "$THANOS_ACCESSIBLE" = true ]; then
    THANOS_HEALTH=$((($THANOS_GPU_HEALTHY + $THANOS_DB_HEALTHY) * 100 / (${#GPU_SERVICES[@]} + ${#DATABASE_CHECKS[@]})))
    echo "Thanos Health: $THANOS_HEALTH% (GPU: $THANOS_GPU_HEALTHY/${#GPU_SERVICES[@]}, DB: $THANOS_DB_HEALTHY/${#DATABASE_CHECKS[@]})"
else
    echo "Thanos Health: ❌ Node not accessible"
fi

if [ "$ORACLE1_ACCESSIBLE" = true ]; then
    ORACLE1_HEALTH=$(($ORACLE1_MONITORING_HEALTHY + $ORACLE1_SECURITY_HEALTHY))
    ORACLE1_TOTAL=$((${#MONITORING_SERVICES[@]} + ${#SECURITY_SERVICES[@]}))
    ORACLE1_HEALTH_PCT=$(($ORACLE1_HEALTH * 100 / $ORACLE1_TOTAL))
    echo "Oracle1 Health: $ORACLE1_HEALTH_PCT% (Monitoring: $ORACLE1_MONITORING_HEALTHY/${#MONITORING_SERVICES[@]}, Security: $ORACLE1_SECURITY_HEALTHY/${#SECURITY_SERVICES[@]})"
else
    echo "Oracle1 Health: ❌ Node not accessible"
fi

STARLORD_HEALTH=$(($STARLORD_DEV_HEALTHY * 100 / ${#DEV_SERVICES[@]}))
echo "Starlord Health: $STARLORD_HEALTH% (Dev: $STARLORD_DEV_HEALTHY/${#DEV_SERVICES[@]})"

echo "Cross-Node Connectivity: $CROSS_NODE_HEALTHY/3 connections"
echo ""

# Overall platform status
echo -e "${BLUE}📈 PLATFORM STATUS:${NC}"
echo "Total Services Running: $TOTAL_SERVICES"
echo "Critical Services Healthy: $CRITICAL_SERVICES"

# Determine overall health
if [ "$THANOS_ACCESSIBLE" = true ] && [ "$ORACLE1_ACCESSIBLE" = true ] && [ $CROSS_NODE_HEALTHY -ge 2 ] && [ $STARLORD_HEALTH -ge 50 ]; then
    if [ $THANOS_HEALTH -ge 80 ] && [ $ORACLE1_HEALTH_PCT -ge 70 ] && [ $STARLORD_HEALTH -ge 80 ]; then
        echo -e "${GREEN}🏆 PLATFORM STATUS: FULLY OPERATIONAL${NC}"
        echo -e "${GREEN}🎯 BEV Enterprise Platform is ready for production workloads!${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️ PLATFORM STATUS: PARTIALLY OPERATIONAL${NC}"
        echo -e "${YELLOW}🔧 Some services need attention but core functionality available${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ PLATFORM STATUS: DEGRADED${NC}"
    echo -e "${RED}🚨 Critical infrastructure issues detected - troubleshooting required${NC}"
    exit 2
fi