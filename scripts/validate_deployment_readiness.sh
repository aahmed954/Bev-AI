#!/bin/bash

# BEV OSINT Framework - Deployment Readiness Validation Script
# Generated: 2025-09-21
# Purpose: Validate Docker Compose configurations before deployment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="/home/starlord/Projects/Bev"
VALIDATION_LOG="${PROJECT_ROOT}/claudedocs/deployment_validation.log"

echo -e "${BLUE}================================"
echo "BEV Deployment Readiness Validator"
echo -e "================================${NC}"
echo "Timestamp: $(date)"
echo "Project Root: ${PROJECT_ROOT}"
echo ""

# Initialize counters
CRITICAL_ISSUES=0
WARNINGS=0
PASSED_CHECKS=0

log_result() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >> "$VALIDATION_LOG"
}

print_status() {
    local status=$1
    local message=$2
    case $status in
        "PASS")
            echo -e "${GREEN}✅ PASS${NC}: $message"
            ((PASSED_CHECKS++))
            log_result "PASS" "$message"
            ;;
        "WARN")
            echo -e "${YELLOW}⚠️  WARN${NC}: $message"
            ((WARNINGS++))
            log_result "WARN" "$message"
            ;;
        "FAIL")
            echo -e "${RED}❌ FAIL${NC}: $message"
            ((CRITICAL_ISSUES++))
            log_result "FAIL" "$message"
            ;;
        "INFO")
            echo -e "${BLUE}ℹ️  INFO${NC}: $message"
            log_result "INFO" "$message"
            ;;
    esac
}

# Clear previous log
> "$VALIDATION_LOG"

print_status "INFO" "Starting comprehensive deployment validation"

echo -e "\n${BLUE}[1/6] Docker Compose File Validation${NC}"
echo "================================================"

# Check if compose files exist
COMPOSE_FILES=(
    "docker-compose.complete.yml"
    "docker-compose-thanos-unified.yml"
    "docker-compose-oracle1-unified.yml"
    "docker-compose-development.yml"
)

for file in "${COMPOSE_FILES[@]}"; do
    if [[ -f "$PROJECT_ROOT/$file" ]]; then
        print_status "PASS" "Compose file exists: $file"

        # Validate YAML syntax
        if docker-compose -f "$PROJECT_ROOT/$file" config > /dev/null 2>&1; then
            print_status "PASS" "YAML syntax valid: $file"
        else
            print_status "FAIL" "YAML syntax invalid: $file"
        fi
    else
        print_status "FAIL" "Missing compose file: $file"
    fi
done

echo -e "\n${BLUE}[2/6] Build Context Validation${NC}"
echo "========================================="

# Check Oracle Dockerfiles
echo "Checking Oracle-specific Dockerfiles..."
ORACLE_DOCKERFILES=(
    "docker/oracle/Dockerfile.research"
    "docker/oracle/Dockerfile.intel"
    "docker/oracle/Dockerfile.celery"
    "docker/oracle/Dockerfile.genetic"
    "docker/oracle/Dockerfile.multiplexer"
    "docker/oracle/Dockerfile.knowledge"
    "docker/oracle/Dockerfile.toolmaster"
    "docker/oracle/Dockerfile.edge"
    "docker/oracle/Dockerfile.drm"
    "docker/oracle/Dockerfile.watermark"
    "docker/oracle/Dockerfile.crypto"
    "docker/oracle/Dockerfile.blackmarket"
    "docker/oracle/Dockerfile.vendor"
    "docker/oracle/Dockerfile.transaction"
    "docker/oracle/Dockerfile.multimodal"
)

missing_oracle_dockerfiles=0
for dockerfile in "${ORACLE_DOCKERFILES[@]}"; do
    if [[ -f "$PROJECT_ROOT/$dockerfile" ]]; then
        print_status "PASS" "Oracle Dockerfile exists: $dockerfile"
    else
        print_status "FAIL" "Missing Oracle Dockerfile: $dockerfile"
        ((missing_oracle_dockerfiles++))
    fi
done

if [[ $missing_oracle_dockerfiles -eq 0 ]]; then
    print_status "PASS" "All Oracle Dockerfiles present"
else
    print_status "FAIL" "$missing_oracle_dockerfiles Oracle Dockerfiles missing"
fi

# Check THANOS Dockerfiles
echo "Checking THANOS-specific Dockerfiles..."
THANOS_DOCKERFILES=(
    "thanos/phase2/ocr/Dockerfile"
    "thanos/phase2/analyzer/Dockerfile"
    "thanos/phase3/swarm/Dockerfile"
    "thanos/phase4/guardian/Dockerfile"
)

for dockerfile in "${THANOS_DOCKERFILES[@]}"; do
    if [[ -f "$PROJECT_ROOT/$dockerfile" ]]; then
        print_status "PASS" "THANOS Dockerfile exists: $dockerfile"
    else
        print_status "FAIL" "Missing THANOS Dockerfile: $dockerfile"
    fi
done

echo -e "\n${BLUE}[3/6] Port Conflict Detection${NC}"
echo "===================================="

# Extract all port mappings and check for conflicts
echo "Analyzing port mappings across compose files..."

# Common conflicting ports
CRITICAL_PORTS=(5432 7474 7687 6379 3000 8086 9090)

for port in "${CRITICAL_PORTS[@]}"; do
    files_using_port=$(grep -l "\"$port:" "$PROJECT_ROOT"/docker-compose*.yml 2>/dev/null || true)
    count=$(echo "$files_using_port" | wc -w)

    if [[ $count -gt 1 ]]; then
        print_status "FAIL" "Port $port conflict in $count files: $(echo $files_using_port | tr '\n' ' ')"
    elif [[ $count -eq 1 ]]; then
        print_status "PASS" "Port $port properly allocated"
    else
        print_status "INFO" "Port $port not used"
    fi
done

echo -e "\n${BLUE}[4/6] Resource Allocation Analysis${NC}"
echo "======================================"

# Check for GPU over-allocation
echo "Analyzing GPU resource allocation..."
gpu_services=$(grep -r "driver: nvidia" "$PROJECT_ROOT"/docker-compose*.yml 2>/dev/null | wc -l)

if [[ $gpu_services -gt 5 ]]; then
    print_status "FAIL" "Potential GPU over-allocation: $gpu_services services request GPU"
elif [[ $gpu_services -gt 2 ]]; then
    print_status "WARN" "Multiple GPU services detected: $gpu_services services"
else
    print_status "PASS" "Reasonable GPU allocation: $gpu_services services"
fi

# Check memory allocation patterns
echo "Analyzing memory allocation..."
high_memory_services=$(grep -r "memory: [0-9][0-9]G" "$PROJECT_ROOT"/docker-compose*.yml 2>/dev/null | wc -l)

if [[ $high_memory_services -gt 10 ]]; then
    print_status "WARN" "High memory allocation detected: $high_memory_services services with 10G+ memory"
else
    print_status "PASS" "Reasonable memory allocation pattern"
fi

echo -e "\n${BLUE}[5/6] Network Configuration Validation${NC}"
echo "=========================================="

# Check network isolation
NETWORKS=(
    "172.30.0.0/16"  # STARLORD
    "172.21.0.0/16"  # THANOS
    "172.31.0.0/16"  # ORACLE1
)

for network in "${NETWORKS[@]}"; do
    if grep -q "$network" "$PROJECT_ROOT"/docker-compose*.yml; then
        print_status "PASS" "Network subnet configured: $network"
    else
        print_status "WARN" "Network subnet not found: $network"
    fi
done

echo -e "\n${BLUE}[6/6] Service Dependencies Check${NC}"
echo "===================================="

# Check for critical service dependencies
CRITICAL_SERVICES=("postgres" "neo4j" "redis" "rabbitmq")

for service in "${CRITICAL_SERVICES[@]}"; do
    if grep -q "$service" "$PROJECT_ROOT"/docker-compose*.yml; then
        print_status "PASS" "Critical service defined: $service"
    else
        print_status "WARN" "Critical service not found: $service"
    fi
done

# Check for health checks
healthcheck_count=$(grep -r "healthcheck:" "$PROJECT_ROOT"/docker-compose*.yml 2>/dev/null | wc -l)
if [[ $healthcheck_count -gt 5 ]]; then
    print_status "PASS" "Health checks implemented: $healthcheck_count services"
else
    print_status "WARN" "Limited health check coverage: $healthcheck_count services"
fi

echo -e "\n${BLUE}==================================="
echo "VALIDATION SUMMARY"
echo -e "===================================${NC}"

echo -e "✅ Passed Checks: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "⚠️  Warnings: ${YELLOW}$WARNINGS${NC}"
echo -e "❌ Critical Issues: ${RED}$CRITICAL_ISSUES${NC}"

if [[ $CRITICAL_ISSUES -eq 0 ]]; then
    echo -e "\n${GREEN}🎉 DEPLOYMENT READY${NC}"
    echo "All critical checks passed. Deployment can proceed."
    exit 0
elif [[ $CRITICAL_ISSUES -lt 5 ]]; then
    echo -e "\n${YELLOW}⚠️  DEPLOYMENT NOT RECOMMENDED${NC}"
    echo "Critical issues found but may be recoverable."
    echo "Review issues and fix before deployment."
    exit 1
else
    echo -e "\n${RED}🚫 DEPLOYMENT BLOCKED${NC}"
    echo "Too many critical issues for safe deployment."
    echo "Must fix critical issues before attempting deployment."
    exit 2
fi