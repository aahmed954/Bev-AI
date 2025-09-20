#!/bin/bash
# BEV Platform Compatibility Validation Script
# Validates ARM64, GPU, and CUDA 13.0 compatibility

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}🔍 BEV PLATFORM COMPATIBILITY VALIDATION${NC}"
echo -e "${BLUE}=========================================${NC}"
echo "Date: $(date)"
echo ""

# Initialize validation counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=()

validate_check() {
    local check_name="$1"
    local check_command="$2"
    local check_description="$3"

    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    echo -n "[$TOTAL_CHECKS] $check_name... "

    if eval "$check_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAILED_CHECKS+=("$check_name: $check_description")
    fi
}

# DOCKER PLATFORM SUPPORT VALIDATION
echo -e "${CYAN}🐳 DOCKER PLATFORM SUPPORT VALIDATION${NC}"
echo "======================================"

validate_check "Docker BuildX" "docker buildx version" "Docker BuildX for multi-platform builds"
validate_check "ARM64 Platform" "docker buildx ls | grep linux/arm64" "ARM64 platform support"
validate_check "AMD64 Platform" "docker buildx ls | grep linux/amd64" "AMD64 platform support"

echo ""

# ARM64 SERVICE VALIDATION (ORACLE1)
echo -e "${CYAN}🔧 ARM64 SERVICE VALIDATION (ORACLE1)${NC}"
echo "==================================="

# Count services with ARM64 platform specifications
ARM64_SERVICES=$(grep -c "platform: linux/arm64" docker-compose-oracle1-unified.yml 2>/dev/null || echo "0")
TOTAL_ORACLE1_SERVICES=$(grep -c "^  [a-zA-Z][a-zA-Z0-9_-]*:$" docker-compose-oracle1-unified.yml 2>/dev/null || echo "0")

validate_check "Oracle1 ARM64 Specs" "test $ARM64_SERVICES -gt 10" "Sufficient ARM64 platform specifications"
validate_check "Oracle1 Compose Valid" "docker-compose -f docker-compose-oracle1-unified.yml config" "Oracle1 compose file valid"

echo "ARM64 services configured: $ARM64_SERVICES"
echo "Total Oracle1 services: $TOTAL_ORACLE1_SERVICES"

echo ""

# GPU/CUDA VALIDATION (THANOS)
echo -e "${CYAN}🚀 GPU/CUDA VALIDATION (THANOS)${NC}"
echo "==============================="

validate_check "NVIDIA Runtime" "docker info | grep 'Runtimes:.*nvidia'" "NVIDIA Docker runtime available"
validate_check "CUDA Installation" "nvidia-smi" "NVIDIA drivers and CUDA available"
validate_check "CUDA Version" "nvidia-smi | grep 'CUDA Version: 13'" "CUDA 13.0 version compatibility"

# Count services with GPU configurations
GPU_SERVICES=$(grep -c "runtime: nvidia" docker-compose-thanos-unified.yml 2>/dev/null || echo "0")
GPU_DEVICE_SPECS=$(grep -c "driver: nvidia" docker-compose-thanos-unified.yml 2>/dev/null || echo "0")
GPU_ENABLED_ENV=$(grep -c "ENABLE_GPU: true" docker-compose-thanos-unified.yml 2>/dev/null || echo "0")

validate_check "GPU Runtime Config" "test $GPU_SERVICES -gt 0" "GPU runtime configurations present"
validate_check "GPU Device Specs" "test $GPU_DEVICE_SPECS -gt 0" "GPU device specifications present"
validate_check "GPU Environment" "test $GPU_ENABLED_ENV -gt 0" "GPU enabled in environment variables"

echo "Services with GPU runtime: $GPU_SERVICES"
echo "Services with GPU device specs: $GPU_DEVICE_SPECS"
echo "Services with GPU enabled: $GPU_ENABLED_ENV"

echo ""

# PLATFORM SPECIFICATION VALIDATION
echo -e "${CYAN}📱 PLATFORM SPECIFICATION VALIDATION${NC}"
echo "===================================="

# Check Thanos services for platform specs
THANOS_PLATFORM_SPECS=$(grep -c "platform:" docker-compose-thanos-unified.yml 2>/dev/null || echo "0")
TOTAL_THANOS_SERVICES=$(grep -c "^  [a-zA-Z][a-zA-Z0-9_-]*:$" docker-compose-thanos-unified.yml 2>/dev/null || echo "0")

validate_check "Thanos Platform Specs" "test $THANOS_PLATFORM_SPECS -gt 20" "Sufficient platform specifications"
validate_check "Thanos Compose Valid" "docker-compose -f docker-compose-thanos-unified.yml config" "Thanos compose file valid"

echo "Thanos platform specs: $THANOS_PLATFORM_SPECS"
echo "Total Thanos services: $TOTAL_THANOS_SERVICES"

echo ""

# DOCKERFILE VALIDATION
echo -e "${CYAN}📦 DOCKERFILE VALIDATION${NC}"
echo "========================"

# Check for missing build contexts
MISSING_CONTEXTS=0
BUILD_CONTEXTS=$(grep -h "context:" docker-compose-thanos-unified.yml docker-compose-oracle1-unified.yml | sed 's/.*context: *//' | sort -u)

echo "Checking build contexts:"
for context in $BUILD_CONTEXTS; do
    echo -n "  $context: "
    if [ -d "$context" ]; then
        echo -e "${GREEN}✅ Exists${NC}"
    else
        echo -e "${RED}❌ Missing${NC}"
        MISSING_CONTEXTS=$((MISSING_CONTEXTS + 1))
    fi
done

validate_check "Build Contexts" "test $MISSING_CONTEXTS -eq 0" "All build contexts exist"

echo ""

# ENVIRONMENT VALIDATION
echo -e "${CYAN}🌍 ENVIRONMENT VALIDATION${NC}"
echo "========================="

validate_check "Main Environment" "test -f .env" "Main environment file exists"
validate_check "CUDA Variables" "grep -q CUDA .env" "CUDA environment variables configured"
validate_check "GPU Variables" "grep -q GPU .env" "GPU environment variables configured"

echo ""

# PYTHON DEPENDENCY VALIDATION
echo -e "${CYAN}🐍 PYTHON DEPENDENCY VALIDATION${NC}"
echo "==============================="

# Check for CUDA-compatible PyTorch
PYTORCH_FILES=$(find . -name "requirements*.txt" -exec grep -l "torch" {} \; 2>/dev/null || echo "")
CUDA_PYTORCH_COUNT=0

if [ -n "$PYTORCH_FILES" ]; then
    echo "Checking PyTorch CUDA compatibility:"
    for file in $PYTORCH_FILES; do
        echo -n "  $file: "
        if grep -q "torch.*cu" "$file" 2>/dev/null; then
            echo -e "${GREEN}✅ CUDA compatible${NC}"
            CUDA_PYTORCH_COUNT=$((CUDA_PYTORCH_COUNT + 1))
        else
            echo -e "${YELLOW}⚠️ CPU-only version${NC}"
        fi
    done
fi

validate_check "CUDA PyTorch" "test $CUDA_PYTORCH_COUNT -gt 0" "CUDA-compatible PyTorch versions found"

echo ""

# NETWORK CONFIGURATION VALIDATION
echo -e "${CYAN}🌐 NETWORK CONFIGURATION VALIDATION${NC}"
echo "==================================="

validate_check "Thanos Network" "grep -q 'bev_osint:' docker-compose-thanos-unified.yml" "Thanos network configuration"
validate_check "Oracle1 Network" "grep -q 'bev_oracle:' docker-compose-oracle1-unified.yml" "Oracle1 network configuration"
validate_check "Cross-Node Network" "grep -q 'external_thanos:' docker-compose-oracle1-unified.yml" "Cross-node network configuration"

echo ""

# SERVICE DISTRIBUTION VALIDATION
echo -e "${CYAN}⚖️ SERVICE DISTRIBUTION VALIDATION${NC}"
echo "=================================="

# Calculate service distribution
THANOS_SERVICE_COUNT=$(grep -c "^  [a-zA-Z][a-zA-Z0-9_-]*:$" docker-compose-thanos-unified.yml 2>/dev/null || echo "0")
ORACLE1_SERVICE_COUNT=$(grep -c "^  [a-zA-Z][a-zA-Z0-9_-]*:$" docker-compose-oracle1-unified.yml 2>/dev/null || echo "0")
TOTAL_SERVICES=$((THANOS_SERVICE_COUNT + ORACLE1_SERVICE_COUNT))

echo "Service distribution:"
echo "  Thanos (High-Compute): $THANOS_SERVICE_COUNT services"
echo "  Oracle1 (ARM): $ORACLE1_SERVICE_COUNT services"
echo "  Total distributed: $TOTAL_SERVICES services"

validate_check "Service Distribution" "test $TOTAL_SERVICES -gt 50" "Adequate service distribution"
validate_check "Thanos Load" "test $THANOS_SERVICE_COUNT -gt 20" "Adequate Thanos service load"
validate_check "Oracle1 Load" "test $ORACLE1_SERVICE_COUNT -gt 15" "Adequate Oracle1 service load"

echo ""

# COMPATIBILITY SUMMARY
echo -e "${PURPLE}📊 COMPATIBILITY VALIDATION SUMMARY${NC}"
echo -e "${BLUE}====================================${NC}"

PASS_PERCENTAGE=$((PASSED_CHECKS * 100 / TOTAL_CHECKS))

echo "Total Checks: $TOTAL_CHECKS"
echo "Passed Checks: $PASSED_CHECKS"
echo "Failed Checks: $((TOTAL_CHECKS - PASSED_CHECKS))"
echo "Success Rate: $PASS_PERCENTAGE%"

echo ""

if [ ${#FAILED_CHECKS[@]} -gt 0 ]; then
    echo -e "${RED}❌ FAILED CHECKS:${NC}"
    for failure in "${FAILED_CHECKS[@]}"; do
        echo -e "${RED}  • $failure${NC}"
    done
    echo ""
fi

# Compatibility assessment
echo -e "${BLUE}🎯 COMPATIBILITY ASSESSMENT:${NC}"

if [ $PASS_PERCENTAGE -ge 95 ]; then
    echo -e "${GREEN}🏆 EXCELLENT: Full compatibility achieved${NC}"
    echo -e "${GREEN}✅ Ready for distributed deployment${NC}"
    COMPATIBILITY="EXCELLENT"
elif [ $PASS_PERCENTAGE -ge 85 ]; then
    echo -e "${GREEN}✅ GOOD: High compatibility with minor issues${NC}"
    echo -e "${GREEN}🚀 Ready for deployment with monitoring${NC}"
    COMPATIBILITY="GOOD"
elif [ $PASS_PERCENTAGE -ge 70 ]; then
    echo -e "${YELLOW}⚠️ FAIR: Deployment possible with some risks${NC}"
    echo -e "${YELLOW}🔧 Address critical issues before production${NC}"
    COMPATIBILITY="FAIR"
else
    echo -e "${RED}❌ POOR: Significant compatibility issues${NC}"
    echo -e "${RED}🚨 Resolve critical issues before deployment${NC}"
    COMPATIBILITY="POOR"
fi

echo ""

# Generate detailed report
echo -e "${BLUE}📄 Generating detailed compatibility report...${NC}"
cat > /tmp/bev_compatibility_validation_$(date +%Y%m%d_%H%M%S).txt << REPORT_EOF
BEV PLATFORM COMPATIBILITY VALIDATION REPORT
Date: $(date)

VALIDATION SUMMARY:
Total Checks: $TOTAL_CHECKS
Passed: $PASSED_CHECKS
Failed: $((TOTAL_CHECKS - PASSED_CHECKS))
Success Rate: $PASS_PERCENTAGE%
Compatibility Level: $COMPATIBILITY

SERVICE DISTRIBUTION:
Thanos (High-Compute): $THANOS_SERVICE_COUNT services
Oracle1 (ARM): $ORACLE1_SERVICE_COUNT services
Total Services: $TOTAL_SERVICES

PLATFORM SPECIFICATIONS:
Thanos Platform Specs: $THANOS_PLATFORM_SPECS
Oracle1 ARM64 Specs: $ARM64_SERVICES

GPU CONFIGURATION:
Services with GPU Runtime: $GPU_SERVICES
Services with GPU Devices: $GPU_DEVICE_SPECS
Services with GPU Enabled: $GPU_ENABLED_ENV

FAILED CHECKS:
$(printf '%s\n' "${FAILED_CHECKS[@]}")

NEXT STEPS:
$(case $COMPATIBILITY in
    "EXCELLENT") echo "✅ Proceed with distributed deployment immediately";;
    "GOOD") echo "🚀 Deploy with standard monitoring";;
    "FAIR") echo "🔧 Address remaining issues, then deploy with caution";;
    "POOR") echo "🚨 Resolve all critical issues before attempting deployment";;
esac)

DEPLOYMENT COMMAND:
./deploy_distributed_bev.sh

MONITORING URLS (post-deployment):
- Grafana: http://oracle1:3000
- Prometheus: http://oracle1:9090
- Vault: http://oracle1:8200
- IntelOwl: http://thanos
REPORT_EOF

echo "Validation report saved to: /tmp/bev_compatibility_validation_*.txt"
echo ""

# Final recommendation
case $COMPATIBILITY in
    "EXCELLENT"|"GOOD")
        echo -e "${GREEN}🚀 RECOMMENDATION: Proceed with distributed deployment${NC}"
        echo -e "${GREEN}   Command: ./deploy_distributed_bev.sh${NC}"
        exit 0
        ;;
    "FAIR")
        echo -e "${YELLOW}🔧 RECOMMENDATION: Address issues, then deploy${NC}"
        echo -e "${YELLOW}   Review failed checks above${NC}"
        exit 1
        ;;
    "POOR")
        echo -e "${RED}🚨 RECOMMENDATION: Do not deploy${NC}"
        echo -e "${RED}   Critical compatibility issues must be resolved${NC}"
        exit 1
        ;;
esac