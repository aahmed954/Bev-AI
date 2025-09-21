#!/bin/bash
#
# BEV Advanced Avatar Systemd Setup Verification
# Quick verification that all components are properly configured
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEV_ROOT="/home/starlord/Projects/Bev"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

CHECK="✓"
CROSS="✗"
WARNING="⚠"

echo "=== BEV Advanced Avatar Systemd Setup Verification ==="
echo ""

# Check all required files exist
echo "Checking required files..."
files=(
    "$SCRIPT_DIR/bev-advanced-avatar.service"
    "$SCRIPT_DIR/install-avatar-service.sh"
    "$SCRIPT_DIR/avatar-service-manager.sh"
    "$SCRIPT_DIR/monitor-avatar-health.sh"
    "$SCRIPT_DIR/scripts/pre-start-validation.sh"
    "$SCRIPT_DIR/scripts/gpu-check.sh"
    "$SCRIPT_DIR/scripts/dependencies-check.sh"
    "$SCRIPT_DIR/scripts/start-avatar.sh"
    "$SCRIPT_DIR/scripts/stop-avatar.sh"
    "$SCRIPT_DIR/scripts/cleanup-gpu.sh"
)

all_files_ok=true
for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo -e "${GREEN}${CHECK} $(basename "$file")${NC}"
    else
        echo -e "${RED}${CROSS} $(basename "$file") - MISSING${NC}"
        all_files_ok=false
    fi
done

echo ""

# Check file permissions
echo "Checking file permissions..."
for file in "${files[@]}"; do
    if [[ -f "$file" && -x "$file" ]]; then
        echo -e "${GREEN}${CHECK} $(basename "$file") - executable${NC}"
    elif [[ -f "$file" ]]; then
        echo -e "${YELLOW}${WARNING} $(basename "$file") - not executable${NC}"
    fi
done

echo ""

# Check directory structure
echo "Checking directory structure..."
dirs=(
    "$BEV_ROOT/src/avatar"
    "$BEV_ROOT/logs"
    "$BEV_ROOT/config"
    "$SCRIPT_DIR/scripts"
)

for dir in "${dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        echo -e "${GREEN}${CHECK} $dir${NC}"
    else
        echo -e "${RED}${CROSS} $dir - MISSING${NC}"
    fi
done

echo ""

# Check if avatar controller exists
echo "Checking BEV Avatar components..."
if [[ -f "$BEV_ROOT/src/avatar/advanced_avatar_controller.py" ]]; then
    echo -e "${GREEN}${CHECK} Advanced Avatar Controller${NC}"
else
    echo -e "${RED}${CROSS} Advanced Avatar Controller - MISSING${NC}"
fi

echo ""

# Summary
if [[ "$all_files_ok" == "true" ]]; then
    echo -e "${GREEN}${CHECK} All systemd service files are present${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Run: ./install-avatar-service.sh"
    echo "2. Start service: ./avatar-service-manager.sh start"
    echo "3. Check status: ./avatar-service-manager.sh status"
    echo ""
else
    echo -e "${RED}${CROSS} Some required files are missing${NC}"
    echo "Please ensure all files are created properly."
    exit 1
fi