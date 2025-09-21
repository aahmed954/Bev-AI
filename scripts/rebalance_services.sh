#!/bin/bash
#
# BEV Platform Service Rebalancing Script
# Redistributes services across nodes for optimal resource utilization
#

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
THANOS_HOST="thanos"
ORACLE1_HOST="oracle1"
STARLORD_HOST="starlord"
BACKUP_DIR="/opt/bev-backup-$(date +%Y%m%d-%H%M%S)"

echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}       BEV PLATFORM SERVICE REBALANCING TOOL${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}"

# Function to check node connectivity
check_node_connectivity() {
    local node=$1
    echo -n "Checking connectivity to $node... "
    if ping -c 1 $node > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Connected${NC}"
        return 0
    else
        echo -e "${RED}❌ Not reachable${NC}"
        return 1
    fi
}

# Function to check node resources
check_node_resources() {
    local node=$1
    echo -e "\n${BLUE}Checking resources on $node...${NC}"

    if [ "$node" == "localhost" ] || [ "$node" == "starlord" ]; then
        # Local check
        TOTAL_MEM=$(free -g | awk 'NR==2{print $2}')
        USED_MEM=$(free -g | awk 'NR==2{print $3}')
        AVAILABLE_MEM=$(free -g | awk 'NR==2{print $7}')
        CPU_COUNT=$(nproc)
        CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

        echo "  Memory: ${USED_MEM}/${TOTAL_MEM}GB used (${AVAILABLE_MEM}GB available)"
        echo "  CPU: ${CPU_COUNT} cores, ${CPU_USAGE}% usage"

        # Check for GPU
        if command -v nvidia-smi > /dev/null 2>&1; then
            GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)
            echo "  GPU: $GPU_INFO"
        fi
    else
        # Remote check via SSH
        ssh $node "free -g; nproc; top -bn1 | grep 'Cpu(s)'" 2>/dev/null || echo "  Unable to get remote stats"
    fi
}

# Function to backup current configurations
backup_configurations() {
    echo -e "\n${BLUE}📦 Creating backup of current configurations...${NC}"

    mkdir -p $BACKUP_DIR

    # Backup docker-compose files
    cp docker-compose-thanos-unified.yml $BACKUP_DIR/ 2>/dev/null || true
    cp docker-compose-oracle1-unified.yml $BACKUP_DIR/ 2>/dev/null || true
    cp docker-compose-starlord-production.yml $BACKUP_DIR/ 2>/dev/null || true

    # Backup environment files
    cp .env* $BACKUP_DIR/ 2>/dev/null || true

    echo -e "${GREEN}✅ Backup created at: $BACKUP_DIR${NC}"
}

# Function to create optimized THANOS configuration
create_optimized_thanos_config() {
    echo -e "\n${BLUE}🔧 Creating optimized THANOS configuration...${NC}"

    cat > docker-compose-thanos-optimized.yml << 'EOF'
version: '3.9'

# Optimized for 55GB RAM, 14 CPU cores
# Focus on data storage and core services

services:
  # PostgreSQL - Primary Database
  postgres:
    image: pgvector/pgvector:pg16
    container_name: bev_postgres
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '1.0'
    environment:
      SHARED_BUFFERS: 1GB
      EFFECTIVE_CACHE_SIZE: 3GB
      MAX_CONNECTIONS: 200
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Neo4j - Graph Database
  neo4j:
    image: neo4j:5.14-enterprise
    container_name: bev_neo4j
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '1.0'
    environment:
      NEO4J_server_memory_heap_max__size: 2G
      NEO4J_server_memory_pagecache__size: 1G

  # Elasticsearch - Search and Analytics
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    container_name: bev_elasticsearch
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
    environment:
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
      - "discovery.type=single-node"

  # Redis Cluster (3 nodes, 2GB each = 6GB total)
  redis-1:
    image: redis:7-alpine
    container_name: bev_redis_1
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '0.5'

  redis-2:
    image: redis:7-alpine
    container_name: bev_redis_2
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '0.5'

  redis-3:
    image: redis:7-alpine
    container_name: bev_redis_3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '0.5'

  # Kafka Cluster (3 nodes, 3GB each = 9GB total)
  kafka-1:
    image: confluentinc/cp-kafka:7.5.0
    container_name: bev_kafka_1
    deploy:
      resources:
        limits:
          memory: 3G
          cpus: '1.0'
    environment:
      KAFKA_HEAP_OPTS: "-Xmx2G -Xms2G"

  kafka-2:
    image: confluentinc/cp-kafka:7.5.0
    container_name: bev_kafka_2
    deploy:
      resources:
        limits:
          memory: 3G
          cpus: '1.0'
    environment:
      KAFKA_HEAP_OPTS: "-Xmx2G -Xms2G"

  kafka-3:
    image: confluentinc/cp-kafka:7.5.0
    container_name: bev_kafka_3
    deploy:
      resources:
        limits:
          memory: 3G
          cpus: '1.0'
    environment:
      KAFKA_HEAP_OPTS: "-Xmx2G -Xms2G"

  # RabbitMQ Cluster (3 nodes, 2GB each = 6GB total)
  rabbitmq-1:
    image: rabbitmq:3-management
    container_name: bev_rabbitmq_1
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '0.5'

  rabbitmq-2:
    image: rabbitmq:3-management
    container_name: bev_rabbitmq_2
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '0.5'

  rabbitmq-3:
    image: rabbitmq:3-management
    container_name: bev_rabbitmq_3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '0.5'

  # IntelOwl Platform
  intelowl:
    image: intelowlproject/intelowl:latest
    container_name: bev_intelowl
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '1.0'

  # Core OSINT Services (reduced memory)
  osint-analyzer:
    build: ./src/analyzers
    container_name: bev_osint_analyzer
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  breach-monitor:
    build: ./src/breach
    container_name: bev_breach_monitor
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  threat-intel:
    build: ./src/threat_intel
    container_name: bev_threat_intel
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'

  # Zookeeper for Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:7.5.0
    container_name: bev_zookeeper
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'

# Total: ~55GB RAM, ~14 CPU cores
EOF

    echo -e "${GREEN}✅ Optimized THANOS configuration created${NC}"
}

# Function to create STARLORD production configuration
create_starlord_production_config() {
    echo -e "\n${BLUE}🚀 Creating STARLORD production configuration...${NC}"

    cat > docker-compose-starlord-production.yml << 'EOF'
version: '3.9'

# Optimized for RTX 4090 GPU workloads
# 40GB RAM, 20GB VRAM allocation

services:
  # Document Analyzers (GPU-accelerated)
  doc-analyzer-1:
    build: ./src/analyzers/document
    container_name: bev_doc_analyzer_1
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 3G
          cpus: '2.0'
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0

  doc-analyzer-2:
    build: ./src/analyzers/document
    container_name: bev_doc_analyzer_2
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 3G
          cpus: '2.0'
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0

  doc-analyzer-3:
    build: ./src/analyzers/document
    container_name: bev_doc_analyzer_3
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 3G
          cpus: '2.0'
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_VISIBLE_DEVICES=0

  # Swarm Intelligence Masters (GPU-enhanced)
  swarm-master-1:
    build: ./src/swarm
    container_name: bev_swarm_master_1
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 3G
          cpus: '2.0'
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

  swarm-master-2:
    build: ./src/swarm
    container_name: bev_swarm_master_2
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 3G
          cpus: '2.0'
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

  # Autonomous AI Services
  autonomous-controller-1:
    build: ./src/autonomous
    container_name: bev_autonomous_1
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

  autonomous-controller-2:
    build: ./src/autonomous
    container_name: bev_autonomous_2
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

  # AI Memory Manager
  memory-manager:
    build: ./src/memory
    container_name: bev_memory_manager
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

  # AI Companion Services
  avatar-system:
    build: ./src/avatar
    container_name: bev_avatar_system
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 6G
          cpus: '3.0'
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
    ports:
      - "8091:8091"

  emotional-intelligence:
    build: ./src/avatar/emotional
    container_name: bev_emotional_intelligence
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

  extended-reasoning:
    build: ./src/reasoning
    container_name: bev_extended_reasoning
    runtime: nvidia
    deploy:
      resources:
        limits:
          memory: 6G
          cpus: '3.0'
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]

# Total: ~40GB RAM, utilizing RTX 4090 GPU
EOF

    echo -e "${GREEN}✅ STARLORD production configuration created${NC}"
}

# Function to perform gradual service migration
migrate_services() {
    echo -e "\n${CYAN}🚀 Starting service migration...${NC}"

    # Phase 1: Stop non-critical services on THANOS
    echo -e "\n${YELLOW}Phase 1: Stopping GPU services on THANOS...${NC}"
    ssh $THANOS_HOST "docker stop bev_doc_analyzer_1 bev_doc_analyzer_2 bev_doc_analyzer_3" 2>/dev/null || true
    ssh $THANOS_HOST "docker stop bev_swarm_master_1 bev_swarm_master_2" 2>/dev/null || true
    ssh $THANOS_HOST "docker stop bev_autonomous_1 bev_autonomous_2" 2>/dev/null || true

    # Phase 2: Deploy services to STARLORD
    echo -e "\n${YELLOW}Phase 2: Deploying GPU services to STARLORD...${NC}"
    docker-compose -f docker-compose-starlord-production.yml up -d

    # Phase 3: Update THANOS with optimized configuration
    echo -e "\n${YELLOW}Phase 3: Applying optimized configuration to THANOS...${NC}"
    ssh $THANOS_HOST "cd /opt/bev && docker-compose -f docker-compose-thanos-optimized.yml up -d"

    # Phase 4: Verify services
    echo -e "\n${YELLOW}Phase 4: Verifying service health...${NC}"
    sleep 30  # Wait for services to stabilize
    verify_service_health
}

# Function to verify service health
verify_service_health() {
    echo -e "\n${BLUE}🏥 Checking service health...${NC}"

    # Check THANOS services
    echo -e "\n${CYAN}THANOS Services:${NC}"
    THANOS_HEALTHY=0
    THANOS_TOTAL=0
    for service in postgres neo4j elasticsearch redis_1 kafka_1 rabbitmq_1; do
        THANOS_TOTAL=$((THANOS_TOTAL + 1))
        if ssh $THANOS_HOST "docker ps | grep -q bev_$service" 2>/dev/null; then
            echo -e "  ✅ $service: Running"
            THANOS_HEALTHY=$((THANOS_HEALTHY + 1))
        else
            echo -e "  ❌ $service: Not running"
        fi
    done

    # Check STARLORD services
    echo -e "\n${CYAN}STARLORD Services:${NC}"
    STARLORD_HEALTHY=0
    STARLORD_TOTAL=0
    for service in doc_analyzer_1 swarm_master_1 autonomous_1 avatar_system; do
        STARLORD_TOTAL=$((STARLORD_TOTAL + 1))
        if docker ps | grep -q "bev_$service"; then
            echo -e "  ✅ $service: Running"
            STARLORD_HEALTHY=$((STARLORD_HEALTHY + 1))
        else
            echo -e "  ❌ $service: Not running"
        fi
    done

    # Summary
    echo -e "\n${BLUE}═══════════════════════════════════════${NC}"
    echo -e "${BLUE}Migration Summary:${NC}"
    echo -e "  THANOS: $THANOS_HEALTHY/$THANOS_TOTAL services healthy"
    echo -e "  STARLORD: $STARLORD_HEALTHY/$STARLORD_TOTAL services healthy"

    if [ $THANOS_HEALTHY -eq $THANOS_TOTAL ] && [ $STARLORD_HEALTHY -eq $STARLORD_TOTAL ]; then
        echo -e "\n${GREEN}🎉 Migration completed successfully!${NC}"
    else
        echo -e "\n${YELLOW}⚠️ Some services need attention${NC}"
    fi
}

# Function to rollback changes
rollback() {
    echo -e "\n${RED}🔄 Rolling back to previous configuration...${NC}"

    if [ -d "$BACKUP_DIR" ]; then
        cp $BACKUP_DIR/docker-compose-*.yml ./ 2>/dev/null || true
        cp $BACKUP_DIR/.env* ./ 2>/dev/null || true

        # Restart services with original configs
        docker-compose -f docker-compose-thanos-unified.yml up -d
        docker-compose -f docker-compose-oracle1-unified.yml up -d

        echo -e "${GREEN}✅ Rollback completed${NC}"
    else
        echo -e "${RED}❌ No backup found${NC}"
    fi
}

# Main menu
show_menu() {
    echo -e "\n${CYAN}Select an action:${NC}"
    echo "1) Check node connectivity and resources"
    echo "2) Create optimized configurations"
    echo "3) Perform full migration (backup + migrate)"
    echo "4) Verify service health"
    echo "5) Rollback to previous configuration"
    echo "6) Exit"
}

# Main execution
main() {
    while true; do
        show_menu
        read -p "Enter choice [1-6]: " choice

        case $choice in
            1)
                check_node_connectivity $THANOS_HOST
                check_node_connectivity $ORACLE1_HOST
                check_node_connectivity $STARLORD_HOST
                check_node_resources $THANOS_HOST
                check_node_resources $ORACLE1_HOST
                check_node_resources $STARLORD_HOST
                ;;
            2)
                create_optimized_thanos_config
                create_starlord_production_config
                ;;
            3)
                echo -e "${YELLOW}⚠️ This will migrate services across nodes.${NC}"
                read -p "Continue? (y/n): " confirm
                if [ "$confirm" == "y" ]; then
                    backup_configurations
                    create_optimized_thanos_config
                    create_starlord_production_config
                    migrate_services
                fi
                ;;
            4)
                verify_service_health
                ;;
            5)
                echo -e "${YELLOW}⚠️ This will restore previous configuration.${NC}"
                read -p "Continue? (y/n): " confirm
                if [ "$confirm" == "y" ]; then
                    rollback
                fi
                ;;
            6)
                echo -e "${GREEN}Exiting...${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option${NC}"
                ;;
        esac
    done
}

# Check if running as root/sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root or with sudo${NC}"
    exit 1
fi

# Start main program
main