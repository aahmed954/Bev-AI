#!/bin/bash

# BEV OSINT Platform - Advanced Phases Deployment Verification Script
# This script verifies the successful deployment of Phases 7, 8, and 9

set -e

echo "=============================================="
echo "BEV OSINT Advanced Phases Deployment Verification"
echo "=============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check service health
check_service_health() {
    local service_name=$1
    local port=$2
    local ip=$3

    echo -n "Checking $service_name (${ip}:${port})... "

    # Check if container is running
    if ! docker ps | grep -q "bev_${service_name}"; then
        echo -e "${RED}FAILED${NC} - Container not running"
        return 1
    fi

    # Check health endpoint with timeout
    if timeout 10 curl -s -f "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        return 0
    else
        echo -e "${YELLOW}WARNING${NC} - Health endpoint not responding"
        return 1
    fi
}

# Function to check database connections
check_database_connections() {
    echo -e "\n${BLUE}=== Database Connectivity ===${NC}"

    # Check PostgreSQL
    echo -n "PostgreSQL connection... "
    if docker exec bev_postgres pg_isready -U bev_admin > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
    fi

    # Check Neo4j
    echo -n "Neo4j connection... "
    if docker exec bev_neo4j cypher-shell -u neo4j -p BevGraph2024! "RETURN 1" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
    fi

    # Check Redis
    echo -n "Redis connection... "
    if docker exec bev_redis_standalone redis-cli -a BevCache2024! ping > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
    fi

    # Check Elasticsearch
    echo -n "Elasticsearch connection... "
    if curl -s "http://localhost:9200/_cluster/health" | grep -q "green\|yellow"; then
        echo -e "${GREEN}OK${NC}"
    else
        echo -e "${RED}FAILED${NC}"
    fi
}

# Function to check GPU availability
check_gpu_availability() {
    echo -e "\n${BLUE}=== GPU Availability ===${NC}"

    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
        echo "Available GPUs: $gpu_count"

        if [ "$gpu_count" -ge 7 ]; then
            echo -e "${GREEN}Sufficient GPUs available${NC}"
        else
            echo -e "${YELLOW}Warning: Only $gpu_count GPUs available, 7 recommended${NC}"
        fi

        # Check GPU utilization
        echo "Current GPU utilization:"
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits
    else
        echo -e "${RED}NVIDIA drivers not found${NC}"
    fi
}

# Main verification process
echo -e "\n${BLUE}=== Phase 7 - Alternative Market Intelligence ===${NC}"
check_service_health "dm_crawler" "8001" "172.30.0.24"
check_service_health "crypto_intel" "8002" "172.30.0.25"
check_service_health "reputation_analyzer" "8003" "172.30.0.26"
check_service_health "economics_processor" "8004" "172.30.0.27"

echo -e "\n${BLUE}=== Phase 8 - Advanced Security Operations ===${NC}"
check_service_health "tactical_intel" "8005" "172.30.0.28"
check_service_health "defense_automation" "8006" "172.30.0.29"
check_service_health "opsec_enforcer" "8007" "172.30.0.30"
check_service_health "intel_fusion" "8008" "172.30.0.31"

echo -e "\n${BLUE}=== Phase 9 - Autonomous Enhancement ===${NC}"
check_service_health "autonomous_coordinator" "8009" "172.30.0.32"
check_service_health "adaptive_learning" "8010" "172.30.0.33"
check_service_health "resource_manager" "8011" "172.30.0.34"
check_service_health "knowledge_evolution" "8012" "172.30.0.35"

# Additional checks
check_database_connections
check_gpu_availability

echo -e "\n${GREEN}Verification complete!${NC}"