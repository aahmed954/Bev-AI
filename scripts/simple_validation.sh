#!/bin/bash
# Simple BEV deployment validation
set -e

echo "=== BEV DEPLOYMENT VALIDATION ==="

# Check Docker
echo -n "Docker daemon: "
if docker info &>/dev/null; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
    exit 1
fi

# Check Docker Compose files
echo -n "THANOS compose syntax: "
if docker compose -f docker-compose-thanos-unified.yml config --quiet &>/dev/null; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
    exit 1
fi

echo -n "ORACLE1 compose syntax: "
if docker compose -f docker-compose-oracle1-unified.yml config --quiet &>/dev/null; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
    exit 1
fi

# Check required files
echo -n "Required files: "
if [[ -f docker-compose-thanos-unified.yml && -f docker-compose-oracle1-unified.yml ]]; then
    echo "✓ PASS"
else
    echo "✗ FAIL"
    exit 1
fi

# Check resources
echo -n "Memory check: "
MEM=$(free -g | awk '/^Mem:/{print $2}')
if (( MEM >= 32 )); then
    echo "✓ PASS (${MEM}GB)"
else
    echo "⚠ WARN (${MEM}GB - may be tight)"
fi

# Check ports
echo -n "Port availability: "
USED_PORTS=$(netstat -tlnp 2>/dev/null | grep -E ":(5432|6379|7474|7687)" | wc -l)
if (( USED_PORTS == 0 )); then
    echo "✓ PASS"
else
    echo "⚠ WARN (${USED_PORTS} critical ports in use)"
fi

echo "=== VALIDATION COMPLETE ==="