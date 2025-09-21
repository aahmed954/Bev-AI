#!/bin/bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║           STARLORD VAULT DEPLOYMENT - VAULT ONLY!                ║
# ║        THIS RUNS ON YOUR PERSONAL DEV MACHINE                    ║
# ╔══════════════════════════════════════════════════════════════════╝

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}   STARLORD (DEV MACHINE) - VAULT ONLY DEPLOYMENT              ${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"
echo ""

# CRITICAL CHECK - Confirm we're on STARLORD
CURRENT_HOST=$(hostname)
if [ "$CURRENT_HOST" != "starlord" ]; then
    echo -e "${RED}❌ ERROR: This script must ONLY run on STARLORD!${NC}"
    echo -e "${RED}   Current host: $CURRENT_HOST${NC}"
    echo -e "${RED}   Expected: starlord${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Confirmed: Running on STARLORD (personal dev machine)${NC}"
echo ""

# Check if any BEV services are already running
RUNNING_SERVICES=$(docker ps --filter "name=bev_" --format "{{.Names}}" | grep -v "bev_vault" || true)
if [ ! -z "$RUNNING_SERVICES" ]; then
    echo -e "${YELLOW}⚠️  Warning: Found running BEV services:${NC}"
    echo "$RUNNING_SERVICES"
    echo ""
    read -p "Stop these services? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "$RUNNING_SERVICES" | xargs -r docker stop
        echo "$RUNNING_SERVICES" | xargs -r docker rm
    fi
fi

# Deploy ONLY Vault
echo -e "${BLUE}→ Deploying Vault (credential management only)...${NC}"
docker-compose -f docker-compose-starlord-vault.yml up -d

# Wait for Vault to be ready
echo -e "${BLUE}→ Waiting for Vault to initialize...${NC}"
sleep 5

# Check Vault status
docker ps --filter "name=bev_vault_starlord"

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ STARLORD VAULT DEPLOYMENT COMPLETE${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Vault UI: http://100.122.12.35:8200/ui${NC}"
echo -e "${YELLOW}Vault is the ONLY service running on this dev machine${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo -e "  1. Initialize Vault if first time"
echo -e "  2. Run ./deploy_to_thanos.sh to deploy services to THANOS"
echo -e "  3. Run ./deploy_to_oracle1.sh to deploy services to ORACLE1"