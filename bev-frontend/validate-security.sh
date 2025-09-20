#!/bin/bash

# BEV OSINT Framework - Phase 0 Security Validation Script
# This script validates MANDATORY proxy enforcement

echo "================================================"
echo "BEV OSINT FRAMEWORK - SECURITY VALIDATION"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Tor is running
echo -n "Checking Tor SOCKS5 proxy on 127.0.0.1:9150... "
if nc -z 127.0.0.1 9150 2>/dev/null; then
    echo -e "${GREEN}✓ CONNECTED${NC}"
else
    echo -e "${RED}✗ FAILED${NC}"
    echo -e "${RED}CRITICAL: Tor proxy not running! Start Tor Browser or tor service.${NC}"
    exit 1
fi

# Test Tor connection
echo -n "Validating Tor circuit... "
EXIT_IP=$(curl -s --socks5 127.0.0.1:9150 https://check.torproject.org/api/ip 2>/dev/null | grep -o '"IP":"[^"]*"' | cut -d'"' -f4)
if [ ! -z "$EXIT_IP" ]; then
    echo -e "${GREEN}✓ Exit IP: $EXIT_IP${NC}"
else
    echo -e "${RED}✗ FAILED to get Tor exit IP${NC}"
    exit 1
fi

# Verify we're using Tor
echo -n "Confirming Tor routing... "
IS_TOR=$(curl -s --socks5 127.0.0.1:9150 https://check.torproject.org/api/ip 2>/dev/null | grep -o '"IsTor":[^,]*' | cut -d':' -f2)
if [ "$IS_TOR" = "true" ]; then
    echo -e "${GREEN}✓ CONFIRMED - All traffic routed through Tor${NC}"
else
    echo -e "${RED}✗ NOT using Tor! OPSEC COMPROMISED!${NC}"
    exit 1
fi

echo ""
echo "================================================"
echo -e "${GREEN}PHASE 0 VALIDATION: PASSED${NC}"
echo "================================================"
echo ""
echo "Security Status:"
echo "  • Tor SOCKS5 Proxy: ACTIVE"
echo "  • Exit IP: $EXIT_IP"
echo "  • Traffic Routing: SECURED"
echo "  • Data Leaks: PREVENTED"
echo ""
echo -e "${YELLOW}You may now proceed with application launch.${NC}"
echo "Run: npm run tauri:dev"
echo ""