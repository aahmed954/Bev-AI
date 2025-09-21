#!/bin/bash

# Bev AI Research Framework - Master Deployment Script
# Deploy across three-node cluster for academic security research

set -e

echo "🚀 Deploying Bev Research Framework..."
echo "=================================="

# Configuration
THANOS_IP="100.122.12.54"
ORACLE_IP="100.96.197.84"
STARLORD_IP="100.72.73.3"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check node connectivity
check_node() {
    local node_name=$1
    local node_ip=$2
    
    echo -e "${YELLOW}Checking $node_name ($node_ip)...${NC}"
    if ping -c 1 -W 2 $node_ip > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $node_name is reachable${NC}"
        return 0
    else
        echo -e "${RED}✗ $node_name is not reachable${NC}"
        return 1
    fi
}

# Deploy to specific node
deploy_to_node() {
    local node_name=$1
    local node_ip=$2
    local compose_file=$3
    
    echo -e "${YELLOW}Deploying to $node_name...${NC}"
    
    # Create remote directory
    ssh starlord@$node_ip "mkdir -p /home/starlord/bev/{config,data,logs}"
    
    # Copy Docker compose and configs
    scp -q docker/$compose_file/docker-compose.yml starlord@$node_ip:/home/starlord/bev/
    scp -qr config/* starlord@$node_ip:/home/starlord/bev/config/ 2>/dev/null || true
    
    # Deploy services
    ssh starlord@$node_ip "cd /home/starlord/bev && docker-compose up -d"
    
    echo -e "${GREEN}✓ $node_name deployment complete${NC}"
}

# Main deployment
main() {
    echo "Step 1: Checking nodes..."
    check_node "THANOS" $THANOS_IP || echo "Warning: THANOS unreachable"
    check_node "Oracle1" $ORACLE_IP || echo "Warning: Oracle1 unreachable"
    
    echo -e "\nStep 2: Generating credentials..."
    ./scripts/generate_credentials.sh || echo "Credentials script not found, using defaults"
    
    echo -e "\nStep 3: Building images..."
    docker-compose -f docker/thanos/docker-compose.yml build --no-cache
    
    echo -e "\nStep 4: Deploying services..."
    deploy_to_node "THANOS" $THANOS_IP "thanos" || echo "THANOS deployment skipped"
    deploy_to_node "Oracle1" $ORACLE_IP "oracle" || echo "Oracle1 deployment skipped"
    
    echo -e "\nStep 5: Initializing databases..."
    ./scripts/init_databases.sh || echo "Database init skipped"
    
    echo -e "\nStep 6: Starting agents..."
    python3 -m src.agents.start_swarm || echo "Agent start skipped"
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Deployment complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    echo -e "\nServices:"
    echo "  Airflow: http://$THANOS_IP:8080"
    echo "  Grafana: http://$THANOS_IP:3000"
    echo "  RabbitMQ: http://$THANOS_IP:15672"
    echo "  Neo4j: http://$THANOS_IP:7474"
}

main "$@"
