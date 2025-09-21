#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║           MULTI-NODE DEPLOYMENT VERIFICATION                     ║
# ║        CHECKS ALL NODES WITHOUT DEPLOYING ANYTHING               ║
# ╔══════════════════════════════════════════════════════════════════╝

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}   BEV MULTI-NODE DEPLOYMENT VERIFICATION                      ${NC}"
echo -e "${PURPLE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Function to check services on a node
check_node() {
    local NODE_NAME=$1
    local NODE_IP=$2
    local NODE_HOST=$3
    local COLOR=$4

    echo -e "${COLOR}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${COLOR} $NODE_NAME ($NODE_IP)${NC}"
    echo -e "${COLOR}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    if [ "$NODE_NAME" == "STARLORD" ]; then
        # Check local services
        echo -e "${BLUE}→ Checking local services...${NC}"
        RUNNING=$(docker ps --filter "name=bev_" --format "{{.Names}}" | wc -l)
        VAULT_STATUS=$(docker ps --filter "name=bev_vault" --format "{{.Status}}" || echo "Not running")

        echo -e "  Services running: $RUNNING"
        echo -e "  Vault status: $VAULT_STATUS"

        if [ "$RUNNING" -gt 1 ]; then
            echo -e "${YELLOW}  ⚠️  WARNING: More than just Vault running on dev machine!${NC}"
            docker ps --filter "name=bev_" --format "table {{.Names}}\t{{.Status}}"
        elif [ "$RUNNING" -eq 0 ]; then
            echo -e "${RED}  ❌ No services running (Vault should be running)${NC}"
        else
            echo -e "${GREEN}  ✅ Only Vault running (correct)${NC}"
        fi
    else
        # Check remote node
        echo -e "${BLUE}→ Testing SSH connectivity...${NC}"
        if ssh -q $NODE_HOST exit 2>/dev/null; then
            echo -e "${GREEN}  ✅ SSH connection successful${NC}"

            # Get service count
            SERVICE_COUNT=$(ssh $NODE_HOST 'docker ps --filter "name=bev_" --format "{{.Names}}" 2>/dev/null | wc -l' || echo "0")
            echo -e "  Services running: $SERVICE_COUNT"

            if [ "$SERVICE_COUNT" -gt 0 ]; then
                echo -e "${BLUE}  Running services:${NC}"
                ssh $NODE_HOST 'docker ps --filter "name=bev_" --format "table {{.Names}}\t{{.Status}}" | head -10'
            else
                echo -e "${YELLOW}  No services running${NC}"
            fi

            # Check disk space
            DISK_USAGE=$(ssh $NODE_HOST 'df -h / | tail -1 | awk "{print \$5}"')
            echo -e "  Disk usage: $DISK_USAGE"

            # Check memory
            MEM_AVAILABLE=$(ssh $NODE_HOST 'free -h | grep "^Mem" | awk "{print \$7}"')
            echo -e "  Memory available: $MEM_AVAILABLE"
        else
            echo -e "${RED}  ❌ Cannot connect via SSH${NC}"
        fi
    fi
    echo ""
}

# Check STARLORD (local)
check_node "STARLORD" "100.122.12.35" "localhost" "$YELLOW"

# Check THANOS
check_node "THANOS" "100.122.12.54" "thanos" "$PURPLE"

# Check ORACLE1
check_node "ORACLE1" "100.96.197.84" "oracle1" "$CYAN"

# Network connectivity tests
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE} NETWORK CONNECTIVITY TESTS${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Test Vault accessibility from nodes
echo -e "${BLUE}→ Testing Vault accessibility...${NC}"
if curl -s http://100.122.12.35:8200/v1/sys/health > /dev/null 2>&1; then
    echo -e "${GREEN}  ✅ Vault accessible from STARLORD${NC}"
else
    echo -e "${RED}  ❌ Vault not accessible${NC}"
fi

# Test cross-node connectivity
echo -e "${BLUE}→ Testing cross-node connectivity...${NC}"
ping -c 1 -W 2 100.122.12.54 > /dev/null 2>&1 && echo -e "${GREEN}  ✅ THANOS reachable${NC}" || echo -e "${RED}  ❌ THANOS unreachable${NC}"
ping -c 1 -W 2 100.96.197.84 > /dev/null 2>&1 && echo -e "${GREEN}  ✅ ORACLE1 reachable${NC}" || echo -e "${RED}  ❌ ORACLE1 unreachable${NC}"

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN} VERIFICATION COMPLETE${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Deployment commands:${NC}"
echo -e "  1. ./deploy_starlord_vault_only.sh  - Deploy Vault on STARLORD"
echo -e "  2. ./deploy_to_thanos.sh            - Deploy services to THANOS"
echo -e "  3. ./deploy_to_oracle1.sh           - Deploy services to ORACLE1"