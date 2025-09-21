#!/bin/bash

# ORACLE1 ARM64 Resource Validation Script
# Validates that all services fit within 24GB RAM / 4 CPU cores

echo "=== ORACLE1 ARM64 Resource Validation ==="
echo

# Resource definitions from docker-compose
ARM_STANDARD_MEMORY="400M"
ARM_STANDARD_CPU="0.07"
ARM_SMALL_MEMORY="200M"
ARM_SMALL_CPU="0.04"
ARM_MONITORING_MEMORY="1G"
ARM_MONITORING_CPU="0.2"

# Convert to MB for calculation
ARM_STANDARD_MEMORY_MB=400
ARM_SMALL_MEMORY_MB=200
ARM_MONITORING_MEMORY_MB=1024

# Count services by resource template
echo "Analyzing services in docker-compose-oracle1-unified.yml..."

# Count standard ARM resource services
STANDARD_SERVICES=$(grep -c "<<: \*arm-resources" docker-compose-oracle1-unified.yml)
echo "Services using standard ARM resources (400MB/0.07CPU): $STANDARD_SERVICES"

# Count small ARM resource services
SMALL_SERVICES=$(grep -c "<<: \*arm-small-resources" docker-compose-oracle1-unified.yml)
echo "Services using small ARM resources (200MB/0.04CPU): $SMALL_SERVICES"

# Count monitoring resource services
MONITORING_SERVICES=$(grep -c "<<: \*arm-monitoring-resources" docker-compose-oracle1-unified.yml)
echo "Services using monitoring ARM resources (1GB/0.2CPU): $MONITORING_SERVICES"

# Count total services
TOTAL_SERVICES=$(grep -c "container_name:" docker-compose-oracle1-unified.yml)
echo "Total services: $TOTAL_SERVICES"

echo
echo "=== Resource Calculations ==="

# Memory calculation
STANDARD_MEMORY_TOTAL=$((STANDARD_SERVICES * ARM_STANDARD_MEMORY_MB))
SMALL_MEMORY_TOTAL=$((SMALL_SERVICES * ARM_SMALL_MEMORY_MB))
MONITORING_MEMORY_TOTAL=$((MONITORING_SERVICES * ARM_MONITORING_MEMORY_MB))
TOTAL_MEMORY_MB=$((STANDARD_MEMORY_TOTAL + SMALL_MEMORY_TOTAL + MONITORING_MEMORY_TOTAL))
TOTAL_MEMORY_GB=$((TOTAL_MEMORY_MB / 1024))

echo "Standard services memory: ${STANDARD_SERVICES} × 400MB = ${STANDARD_MEMORY_TOTAL}MB"
echo "Small services memory: ${SMALL_SERVICES} × 200MB = ${SMALL_MEMORY_TOTAL}MB"
echo "Monitoring services memory: ${MONITORING_SERVICES} × 1024MB = ${MONITORING_MEMORY_TOTAL}MB"
echo "Total memory allocation: ${TOTAL_MEMORY_MB}MB (${TOTAL_MEMORY_GB}GB)"

# CPU calculation
STANDARD_CPU_TOTAL=$(echo "$STANDARD_SERVICES * 0.07" | bc -l)
SMALL_CPU_TOTAL=$(echo "$SMALL_SERVICES * 0.04" | bc -l)
MONITORING_CPU_TOTAL=$(echo "$MONITORING_SERVICES * 0.2" | bc -l)
TOTAL_CPU=$(echo "$STANDARD_CPU_TOTAL + $SMALL_CPU_TOTAL + $MONITORING_CPU_TOTAL" | bc -l)

echo "Standard services CPU: ${STANDARD_SERVICES} × 0.07 = ${STANDARD_CPU_TOTAL} cores"
echo "Small services CPU: ${SMALL_SERVICES} × 0.04 = ${SMALL_CPU_TOTAL} cores"
echo "Monitoring services CPU: ${MONITORING_SERVICES} × 0.2 = ${MONITORING_CPU_TOTAL} cores"
echo "Total CPU allocation: ${TOTAL_CPU} cores"

echo
echo "=== ARM64 Capacity Analysis ==="

# Server specs
SERVER_MEMORY_GB=24
SERVER_CPU_CORES=4

echo "ORACLE1 ARM server capacity:"
echo "- Memory: ${SERVER_MEMORY_GB}GB"
echo "- CPU: ${SERVER_CPU_CORES} cores"

echo
echo "Resource utilization:"

# Memory utilization
MEMORY_USAGE_PERCENT=$(echo "scale=1; $TOTAL_MEMORY_GB * 100 / $SERVER_MEMORY_GB" | bc -l)
echo "- Memory: ${TOTAL_MEMORY_GB}GB / ${SERVER_MEMORY_GB}GB (${MEMORY_USAGE_PERCENT}%)"

# CPU utilization
CPU_USAGE_PERCENT=$(echo "scale=1; $TOTAL_CPU * 100 / $SERVER_CPU_CORES" | bc -l)
echo "- CPU: ${TOTAL_CPU} / ${SERVER_CPU_CORES} cores (${CPU_USAGE_PERCENT}%)"

echo
echo "=== Validation Results ==="

# Validation flags
MEMORY_OK=false
CPU_OK=false

if (( $(echo "$TOTAL_MEMORY_GB <= $SERVER_MEMORY_GB" | bc -l) )); then
    echo "✅ Memory allocation: WITHIN LIMITS"
    MEMORY_OK=true
else
    echo "❌ Memory allocation: EXCEEDS LIMITS"
fi

if (( $(echo "$TOTAL_CPU <= $SERVER_CPU_CORES" | bc -l) )); then
    echo "✅ CPU allocation: WITHIN LIMITS"
    CPU_OK=true
else
    echo "❌ CPU allocation: EXCEEDS LIMITS"
fi

echo
echo "=== Service Breakdown ==="

echo "Key monitoring services added:"
echo "- prometheus (ARM64): 1GB memory, 0.2 CPU"
echo "- grafana (ARM64): 1GB memory, 0.2 CPU"
echo "- alertmanager (ARM64): 200MB memory, 0.04 CPU (using small template)"
echo "- vault (ARM64): 1GB memory, 0.2 CPU"

echo
echo "=== Network Configuration ==="

echo "Monitoring services network integration:"
echo "- bev_oracle: Internal ORACLE1 communication"
echo "- external_thanos: Cross-node coordination with THANOS (100.122.12.54)"

echo "Exposed ports:"
echo "- Prometheus: 9090 (metrics collection)"
echo "- Grafana: 3000 (dashboard interface)"
echo "- AlertManager: 9093 (alert management)"
echo "- Vault: 8200 (API), 8201 (cluster)"

echo
echo "=== Final Assessment ==="

if $MEMORY_OK && $CPU_OK; then
    echo "🎯 VALIDATION PASSED: Resource allocation within ARM64 constraints"
    echo "   Recommended buffer: Keep 10-15% resources free for system overhead"
    exit 0
else
    echo "⚠️  VALIDATION FAILED: Resource allocation exceeds ARM64 capacity"
    echo "   Consider optimizing resource templates or reducing service count"
    exit 1
fi