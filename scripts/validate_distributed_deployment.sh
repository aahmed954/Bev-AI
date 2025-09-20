#!/bin/bash
# BEV Distributed Deployment Validation Script
# Comprehensive validation of 3-node distributed deployment

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🔍 BEV DISTRIBUTED DEPLOYMENT VALIDATION${NC}"
echo -e "${BLUE}=========================================${NC}"
echo "Date: $(date)"
echo ""

# Initialize validation results
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

# NODE CONNECTIVITY VALIDATION
echo -e "${CYAN}🌐 NODE CONNECTIVITY VALIDATION${NC}"
echo "================================"

validate_check "Thanos SSH" "ssh -o ConnectTimeout=5 thanos 'echo success'" "SSH connectivity to Thanos node"
validate_check "Oracle1 SSH" "ssh -o ConnectTimeout=5 oracle1 'echo success'" "SSH connectivity to Oracle1 node"
validate_check "Thanos Docker" "ssh thanos 'docker --version'" "Docker availability on Thanos"
validate_check "Oracle1 Docker" "ssh oracle1 'docker --version'" "Docker availability on Oracle1"
validate_check "Starlord Docker" "docker --version" "Docker availability on Starlord"

echo ""

# HARDWARE VALIDATION
echo -e "${CYAN}🖥️ HARDWARE VALIDATION${NC}"
echo "======================"

validate_check "Thanos GPU" "ssh thanos 'nvidia-smi'" "GPU availability on Thanos"
validate_check "Oracle1 ARM" "ssh oracle1 'test $(uname -m) = aarch64'" "ARM64 architecture on Oracle1"
validate_check "Thanos x86" "ssh thanos 'test $(uname -m) = x86_64'" "x86_64 architecture on Thanos"
validate_check "Thanos Memory" "ssh thanos 'test $(free -g | awk \"NR==2{print \\$2}\") -ge 32'" "Sufficient memory on Thanos (32GB+)"
validate_check "Oracle1 Memory" "ssh oracle1 'test $(free -g | awk \"NR==2{print \\$2}\") -ge 16'" "Sufficient memory on Oracle1 (16GB+)"

echo ""

# DOCKER COMPOSE VALIDATION
echo -e "${CYAN}📋 DOCKER COMPOSE VALIDATION${NC}"
echo "============================="

validate_check "Thanos Compose" "test -f docker-compose-thanos-unified.yml" "Thanos compose file exists"
validate_check "Oracle1 Compose" "test -f docker-compose-oracle1-unified.yml" "Oracle1 compose file exists"
validate_check "Development Compose" "test -f docker-compose-development.yml" "Development compose file exists"
validate_check "Complete Compose" "test -f docker-compose.complete.yml" "Complete compose file exists"

echo ""

# SCRIPT DEPENDENCY VALIDATION
echo -e "${CYAN}📜 SCRIPT DEPENDENCY VALIDATION${NC}"
echo "==============================="

validate_check "Database Init Script" "test -x scripts/init_primary_databases.sh" "Database initialization script executable"
validate_check "ARM Monitoring Script" "test -x scripts/setup_arm_monitoring.sh" "ARM monitoring setup script executable"
validate_check "ARM Security Script" "test -x scripts/setup_arm_security.sh" "ARM security setup script executable"
validate_check "Thanos Health Script" "test -x scripts/health_check_thanos.sh" "Thanos health check script executable"
validate_check "MCP Development Script" "test -x scripts/setup_mcp_development.sh" "MCP development setup script executable"

echo ""

# ENVIRONMENT VALIDATION
echo -e "${CYAN}🌍 ENVIRONMENT VALIDATION${NC}"
echo "========================="

validate_check "Main Environment" "test -f .env" "Main environment file exists"
validate_check "Environment Example" "test -f .env.example" "Environment example file exists"
validate_check "Frontend Dependencies" "test -d bev-frontend/node_modules" "Frontend dependencies installed"
validate_check "Package JSON" "test -f bev-frontend/package.json" "Frontend package.json exists"

echo ""

# NETWORK VALIDATION
echo -e "${CYAN}🔗 NETWORK VALIDATION${NC}"
echo "===================="

validate_check "Tailscale Thanos" "ssh thanos 'ip addr show tailscale0'" "Tailscale VPN on Thanos"
validate_check "Tailscale Oracle1" "ssh oracle1 'ip addr show tailscale0'" "Tailscale VPN on Oracle1"
validate_check "Docker Network Thanos" "ssh thanos 'docker network ls | grep bridge'" "Docker networking on Thanos"
validate_check "Docker Network Oracle1" "ssh oracle1 'docker network ls | grep bridge'" "Docker networking on Oracle1"

echo ""

# PORT AVAILABILITY VALIDATION
echo -e "${CYAN}🔌 PORT AVAILABILITY VALIDATION${NC}"
echo "==============================="

# Critical ports that should be available
CRITICAL_PORTS=(5432 6379 7474 9200 8086 9090 3000 8200 9050)

for port in "${CRITICAL_PORTS[@]}"; do
    validate_check "Port $port Available" "! lsof -i:$port" "Port $port should be available for deployment"
done

echo ""

# CONFIGURATION VALIDATION
echo -e "${CYAN}⚙️ CONFIGURATION VALIDATION${NC}"
echo "==========================="

validate_check "Git Repository" "test -d .git" "Git repository initialized"
validate_check "Branch Status" "git status --porcelain" "Git working directory clean"
validate_check "Config Directory" "test -d config" "Configuration directory exists"
validate_check "Scripts Directory" "test -d scripts" "Scripts directory exists"

echo ""

# DEPLOYMENT READINESS SUMMARY
echo -e "${PURPLE}📊 DEPLOYMENT READINESS SUMMARY${NC}"
echo -e "${BLUE}=================================${NC}"

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

# Deployment readiness assessment
echo -e "${BLUE}🎯 DEPLOYMENT READINESS ASSESSMENT:${NC}"

if [ $PASS_PERCENTAGE -ge 95 ]; then
    echo -e "${GREEN}🏆 EXCELLENT: Ready for production deployment${NC}"
    echo -e "${GREEN}✅ All critical systems validated${NC}"
    READINESS="READY"
elif [ $PASS_PERCENTAGE -ge 85 ]; then
    echo -e "${YELLOW}⚠️ GOOD: Ready with minor issues${NC}"
    echo -e "${YELLOW}🔧 Some non-critical issues detected${NC}"
    READINESS="READY_WITH_WARNINGS"
elif [ $PASS_PERCENTAGE -ge 70 ]; then
    echo -e "${YELLOW}⚠️ FAIR: Deployment possible but risky${NC}"
    echo -e "${YELLOW}🛠️ Several issues need resolution${NC}"
    READINESS="RISKY"
else
    echo -e "${RED}❌ POOR: Not ready for deployment${NC}"
    echo -e "${RED}🚨 Critical issues must be resolved${NC}"
    READINESS="NOT_READY"
fi

echo ""

# Generate validation report
echo -e "${BLUE}📄 Generating validation report...${NC}"
cat > /tmp/bev_validation_report.txt << REPORT_EOF
BEV DISTRIBUTED DEPLOYMENT VALIDATION REPORT
Date: $(date)

VALIDATION SUMMARY:
Total Checks: $TOTAL_CHECKS
Passed: $PASSED_CHECKS
Failed: $((TOTAL_CHECKS - PASSED_CHECKS))
Success Rate: $PASS_PERCENTAGE%
Deployment Readiness: $READINESS

FAILED CHECKS:
$(printf '%s\n' "${FAILED_CHECKS[@]}")

RECOMMENDATIONS:
$(case $READINESS in
    "READY") echo "✅ Proceed with distributed deployment";;
    "READY_WITH_WARNINGS") echo "⚠️ Proceed with caution, monitor closely";;
    "RISKY") echo "🛠️ Fix critical issues before deployment";;
    "NOT_READY") echo "🚨 Resolve all critical issues before attempting deployment";;
esac)

NEXT STEPS:
1. Review failed checks above
2. Resolve critical infrastructure issues
3. Re-run validation script
4. Proceed with deployment when ready

ACCESS INFORMATION:
- Thanos (Primary): ssh thanos
- Oracle1 (ARM): ssh oracle1
- Starlord (Dev): localhost

REPORT_EOF

echo "Validation report saved to: /tmp/bev_validation_report.txt"
echo ""

# Final recommendation
case $READINESS in
    "READY")
        echo -e "${GREEN}🚀 RECOMMENDATION: Proceed with deployment${NC}"
        echo -e "${GREEN}   Command: ./deploy_distributed_bev.sh${NC}"
        exit 0
        ;;
    "READY_WITH_WARNINGS")
        echo -e "${YELLOW}⚠️ RECOMMENDATION: Proceed with caution${NC}"
        echo -e "${YELLOW}   Monitor deployment closely${NC}"
        exit 0
        ;;
    "RISKY")
        echo -e "${YELLOW}🛠️ RECOMMENDATION: Fix issues first${NC}"
        echo -e "${YELLOW}   Resolve critical problems before deployment${NC}"
        exit 1
        ;;
    "NOT_READY")
        echo -e "${RED}🚨 RECOMMENDATION: Do not deploy${NC}"
        echo -e "${RED}   Critical issues must be resolved${NC}"
        exit 1
        ;;
esac