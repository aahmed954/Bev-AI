#!/bin/bash
# Clean up all previous deployment containers and services on all nodes
# Ensures fresh deployment environment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🧹 CLEANING UP ALL NODES FOR FRESH DEPLOYMENT${NC}"
echo -e "${BLUE}===============================================${NC}"
echo "Date: $(date)"
echo ""

# Function to clean up a node
cleanup_node() {
    local NODE_NAME="$1"
    local NODE_HOST="$2"

    echo -e "${BLUE}🧹 Cleaning up $NODE_NAME node...${NC}"

    if [ "$NODE_HOST" = "localhost" ]; then
        # Local cleanup
        cleanup_local_node "$NODE_NAME"
    else
        # Remote cleanup via SSH
        ssh "$NODE_HOST" "$(declare -f cleanup_remote_node); cleanup_remote_node '$NODE_NAME'"
    fi
}

cleanup_local_node() {
    local NODE_NAME="$1"

    echo "Stopping all BEV containers on $NODE_NAME..."

    # Stop all containers with bev prefix
    BEV_CONTAINERS=$(docker ps -a --filter "name=bev_" --format "{{.Names}}" 2>/dev/null || echo "")
    if [ -n "$BEV_CONTAINERS" ]; then
        echo "Found BEV containers: $BEV_CONTAINERS"
        docker stop $BEV_CONTAINERS 2>/dev/null || echo "Some containers already stopped"
        docker rm $BEV_CONTAINERS 2>/dev/null || echo "Some containers already removed"
        echo -e "${GREEN}✅ BEV containers cleaned${NC}"
    else
        echo "No BEV containers found"
    fi

    # Stop all docker-compose services
    for compose_file in docker-compose*.yml; do
        if [ -f "$compose_file" ]; then
            echo "Stopping services from $compose_file..."
            docker-compose -f "$compose_file" down --remove-orphans 2>/dev/null || echo "Services already down"
        fi
    done

    # Clean up volumes (optional - preserves data)
    echo "Cleaning up unused Docker resources..."
    docker system prune -f 2>/dev/null || echo "System already clean"

    # Kill any remaining development servers
    echo "Stopping development servers..."
    pkill -f "npm.*dev" 2>/dev/null || echo "No npm dev servers running"
    pkill -f "node.*server" 2>/dev/null || echo "No node servers running"

    echo -e "${GREEN}✅ $NODE_NAME cleanup complete${NC}"
}

cleanup_remote_node() {
    local NODE_NAME="$1"

    echo "Cleaning up $NODE_NAME remotely..."

    # Stop all BEV containers
    BEV_CONTAINERS=$(docker ps -a --filter "name=bev_" --format "{{.Names}}" 2>/dev/null || echo "")
    if [ -n "$BEV_CONTAINERS" ]; then
        echo "Stopping BEV containers: $BEV_CONTAINERS"
        docker stop $BEV_CONTAINERS 2>/dev/null || echo "Containers already stopped"
        docker rm $BEV_CONTAINERS 2>/dev/null || echo "Containers already removed"
    fi

    # Clean up docker-compose services in deployment directory
    if [ -d "/opt/bev-deployment/bev-platform" ]; then
        cd /opt/bev-deployment/bev-platform
        for compose_file in docker-compose*.yml; do
            if [ -f "$compose_file" ]; then
                echo "Stopping $compose_file services..."
                docker-compose -f "$compose_file" down --remove-orphans 2>/dev/null || echo "Already down"
            fi
        done
    fi

    # System cleanup
    docker system prune -f 2>/dev/null || echo "System clean"

    echo "✅ $NODE_NAME remote cleanup complete"
}

# Clean up Starlord (local)
cleanup_node "STARLORD" "localhost"

echo ""

# Clean up Thanos (remote)
echo -e "${BLUE}🧹 Cleaning up THANOS node...${NC}"
ssh thanos << 'THANOS_CLEANUP'
echo "Cleaning up Thanos node..."

# Stop all BEV containers
BEV_CONTAINERS=$(docker ps -a --filter "name=bev_" --format "{{.Names}}" 2>/dev/null || echo "")
if [ -n "$BEV_CONTAINERS" ]; then
    echo "Stopping BEV containers: $BEV_CONTAINERS"
    docker stop $BEV_CONTAINERS 2>/dev/null || echo "Already stopped"
    docker rm $BEV_CONTAINERS 2>/dev/null || echo "Already removed"
    echo "✅ BEV containers cleaned"
else
    echo "No BEV containers found"
fi

# Clean up docker-compose services
if [ -d "/opt/bev-deployment/bev-platform" ]; then
    cd /opt/bev-deployment/bev-platform
    for compose_file in docker-compose*.yml; do
        if [ -f "$compose_file" ]; then
            echo "Stopping $compose_file services..."
            docker-compose -f "$compose_file" down --remove-orphans --volumes 2>/dev/null || echo "Already down"
        fi
    done
fi

# System cleanup
echo "Cleaning Docker system..."
docker system prune -f 2>/dev/null || echo "System clean"

# Remove old deployment directory
if [ -d "/opt/bev-deployment" ]; then
    echo "Removing old deployment directory..."
    sudo rm -rf /opt/bev-deployment 2>/dev/null || echo "Directory already clean"
fi

echo "✅ Thanos cleanup complete"
THANOS_CLEANUP

echo ""

# Clean up Oracle1 (remote)
echo -e "${BLUE}🧹 Cleaning up ORACLE1 node...${NC}"
ssh oracle1 << 'ORACLE1_CLEANUP'
echo "Cleaning up Oracle1 node..."

# Stop all BEV containers
BEV_CONTAINERS=$(docker ps -a --filter "name=bev_" --format "{{.Names}}" 2>/dev/null || echo "")
if [ -n "$BEV_CONTAINERS" ]; then
    echo "Stopping BEV containers: $BEV_CONTAINERS"
    docker stop $BEV_CONTAINERS 2>/dev/null || echo "Already stopped"
    docker rm $BEV_CONTAINERS 2>/dev/null || echo "Already removed"
    echo "✅ BEV containers cleaned"
else
    echo "No BEV containers found"
fi

# Clean up docker-compose services
if [ -d "/opt/bev-deployment/bev-platform" ]; then
    cd /opt/bev-deployment/bev-platform
    for compose_file in docker-compose*.yml; do
        if [ -f "$compose_file" ]; then
            echo "Stopping $compose_file services..."
            docker-compose -f "$compose_file" down --remove-orphans --volumes 2>/dev/null || echo "Already down"
        fi
    done
fi

# System cleanup
echo "Cleaning Docker system..."
docker system prune -f 2>/dev/null || echo "System clean"

# Remove old deployment directory
if [ -d "/opt/bev-deployment" ]; then
    echo "Removing old deployment directory..."
    sudo rm -rf /opt/bev-deployment 2>/dev/null || echo "Directory already clean"
fi

# Verify architecture while we're here
echo "Verifying Oracle1 architecture:"
echo "Hostname: $(hostname)"
echo "Architecture: $(uname -m)"
echo "CPU Info: $(lscpu | grep "Architecture:" | awk '{print $2}')"

echo "✅ Oracle1 cleanup complete"
ORACLE1_CLEANUP

echo ""

# Verify cleanup
echo -e "${BLUE}🔍 Verifying cleanup on all nodes...${NC}"

echo -n "Starlord containers: "
STARLORD_CONTAINERS=$(docker ps --filter "name=bev_" --format "{{.Names}}" | wc -l)
echo "$STARLORD_CONTAINERS running"

echo -n "Thanos containers: "
THANOS_CONTAINERS=$(ssh thanos "docker ps --filter 'name=bev_' --format '{{.Names}}' | wc -l")
echo "$THANOS_CONTAINERS running"

echo -n "Oracle1 containers: "
ORACLE1_CONTAINERS=$(ssh oracle1 "docker ps --filter 'name=bev_' --format '{{.Names}}' | wc -l")
echo "$ORACLE1_CONTAINERS running"

# Check port availability
echo ""
echo -e "${BLUE}🔌 Verifying critical ports are available...${NC}"
CRITICAL_PORTS=(3000 5432 6379 7474 8200 9090)

for port in "${CRITICAL_PORTS[@]}"; do
    echo -n "Port $port: "
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}⚠️ In Use${NC}"
    else
        echo -e "${GREEN}✅ Available${NC}"
    fi
done

echo ""
echo -e "${GREEN}🎯 ALL NODES CLEANED AND READY FOR FRESH DEPLOYMENT${NC}"
echo -e "${CYAN}Next step: Run ./deploy_distributed_bev.sh${NC}"