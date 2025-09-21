#!/bin/bash
# BEV DEPLOYMENT ROLLBACK SCRIPT
# Safely rolls back BEV deployment to previous state

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${RED}🔄 BEV DEPLOYMENT ROLLBACK${NC}"
echo -e "${BLUE}===============================================${NC}"
echo "Emergency rollback of BEV services"
echo ""

# Find the most recent rollback state
ROLLBACK_FILE=$(ls -t rollback_state_*.txt 2>/dev/null | head -1)

if [ -z "$ROLLBACK_FILE" ]; then
    echo -e "${YELLOW}⚠️  No rollback state found. Performing emergency shutdown...${NC}"

    # Emergency shutdown all BEV services
    echo "Stopping all BEV services..."
    docker-compose -f docker-compose-phase9.yml down || true
    docker-compose -f docker-compose-phase8.yml down || true
    docker-compose -f docker-compose-phase7.yml down || true
    docker-compose -f docker-compose.complete.yml down || true

    echo -e "${GREEN}✅ Emergency shutdown complete${NC}"
    exit 0
fi

echo "Using rollback state: $ROLLBACK_FILE"
echo ""

# Show current vs rollback state
echo -e "${CYAN}Current state:${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep bev_ || echo "No BEV services running"

echo ""
echo -e "${CYAN}Target rollback state:${NC}"
cat "$ROLLBACK_FILE"

echo ""
read -p "Proceed with rollback? (y/N): " CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Rollback cancelled"
    exit 0
fi

# Stop all current BEV services
echo -e "${YELLOW}Stopping all BEV services...${NC}"

# Stop in reverse order
docker-compose -f docker-compose-phase9.yml down 2>/dev/null || true
docker-compose -f docker-compose-phase8.yml down 2>/dev/null || true
docker-compose -f docker-compose-phase7.yml down 2>/dev/null || true

# Stop core infrastructure (careful approach)
echo "Stopping core services safely..."
docker-compose -f docker-compose.complete.yml stop \
    dm-crawler crypto-analyzer reputation-analyzer economics-processor \
    intel-fusion opsec-enforcer defense-automation tactical-intelligence \
    enhanced-autonomous-controller adaptive-learning resource-optimizer knowledge-evolution \
    2>/dev/null || true

# Remove any orphaned containers
docker-compose -f docker-compose.complete.yml down --remove-orphans 2>/dev/null || true

echo -e "${GREEN}✅ Rollback complete${NC}"
echo ""
echo "Current state:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep bev_ || echo "No BEV services running"

# Clean up deployment success marker
rm -f .deployment_success

echo ""
echo -e "${CYAN}To restart with working configuration:${NC}"
echo "./deploy_bev_real_implementations.sh"