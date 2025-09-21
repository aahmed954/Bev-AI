#!/bin/bash
# BEV OSINT Framework - Complete Deployment Script
# Definitive deployment procedure for THANOS + ORACLE1 multinode setup
# Generated: $(date)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_LOG="${SCRIPT_DIR}/deployment_$(date +%Y%m%d_%H%M%S).log"
DEPLOYMENT_MODE="${1:-full}"  # full, thanos-only, oracle1-only, validate

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$DEPLOYMENT_LOG"
}

log_header() {
    echo -e "${BOLD}${PURPLE}$1${NC}" | tee -a "$DEPLOYMENT_LOG"
    echo "============================================" | tee -a "$DEPLOYMENT_LOG"
}

# Pre-deployment checks
check_system_requirements() {
    log_header "🔍 SYSTEM REQUIREMENTS CHECK"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    # Check system resources
    local total_memory=$(free -g | awk '/^Mem:/{print $2}')
    local available_disk=$(df "$SCRIPT_DIR" | awk 'NR==2{print int($4/1000000)}')

    log_info "System Resources:"
    log_info "  Memory: ${total_memory}GB"
    log_info "  Available Disk: ${available_disk}GB"

    if (( total_memory < 16 )); then
        log_warning "System has ${total_memory}GB RAM. Recommended: 16GB+ for full deployment"
    fi

    if (( available_disk < 100 )); then
        log_warning "System has ${available_disk}GB free disk. Recommended: 100GB+ for full deployment"
    fi

    # Check required files
    local required_files=(
        "docker-compose-thanos-unified.yml"
        "docker-compose-oracle1-unified.yml"
        ".env.thanos.complete"
        ".env.oracle1.complete"
        "deploy_thanos_node.sh"
        "deploy_oracle1_node.sh"
    )

    for file in "${required_files[@]}"; do
        if [[ ! -f "$SCRIPT_DIR/$file" ]]; then
            log_error "Required file missing: $file"
            exit 1
        fi
    done

    log_success "System requirements check passed"
}

# Deployment orchestration
deploy_thanos_node() {
    log_header "🚀 DEPLOYING THANOS NODE (Primary Compute)"

    if [[ -x "$SCRIPT_DIR/deploy_thanos_node.sh" ]]; then
        log_info "Starting THANOS node deployment..."
        if "$SCRIPT_DIR/deploy_thanos_node.sh" 2>&1 | tee -a "$DEPLOYMENT_LOG"; then
            log_success "THANOS node deployment completed successfully"
            return 0
        else
            log_error "THANOS node deployment failed"
            return 1
        fi
    else
        log_error "THANOS deployment script not found or not executable"
        return 1
    fi
}

deploy_oracle1_node() {
    log_header "📊 DEPLOYING ORACLE1 NODE (ARM64 Monitoring)"

    if [[ -x "$SCRIPT_DIR/deploy_oracle1_node.sh" ]]; then
        log_info "Starting ORACLE1 node deployment..."
        if "$SCRIPT_DIR/deploy_oracle1_node.sh" 2>&1 | tee -a "$DEPLOYMENT_LOG"; then
            log_success "ORACLE1 node deployment completed successfully"
            return 0
        else
            log_error "ORACLE1 node deployment failed"
            return 1
        fi
    else
        log_error "ORACLE1 deployment script not found or not executable"
        return 1
    fi
}

# Cross-node integration testing
test_integration() {
    log_header "🔗 TESTING CROSS-NODE INTEGRATION"

    if [[ -x "$SCRIPT_DIR/test_cross_node_integration.sh" ]]; then
        log_info "Starting cross-node integration tests..."
        if "$SCRIPT_DIR/test_cross_node_integration.sh" 2>&1 | tee -a "$DEPLOYMENT_LOG"; then
            log_success "Cross-node integration tests passed"
            return 0
        else
            log_warning "Cross-node integration tests failed - check logs"
            return 1
        fi
    else
        log_error "Cross-node integration test script not found"
        return 1
    fi
}

# Comprehensive validation
run_validation() {
    log_header "✅ COMPREHENSIVE DEPLOYMENT VALIDATION"

    if [[ -x "$SCRIPT_DIR/validate_complete_deployment.sh" ]]; then
        log_info "Running comprehensive validation..."
        if "$SCRIPT_DIR/validate_complete_deployment.sh" 2>&1 | tee -a "$DEPLOYMENT_LOG"; then
            log_success "Comprehensive validation passed"
            return 0
        else
            log_warning "Comprehensive validation failed - check specific issues"
            return 1
        fi
    else
        log_warning "Comprehensive validation script not found - skipping"
        return 0
    fi
}

# Display deployment summary
show_deployment_summary() {
    log_header "📋 DEPLOYMENT SUMMARY"

    echo "🎉 BEV OSINT Framework Deployment Complete!" | tee -a "$DEPLOYMENT_LOG"
    echo "" | tee -a "$DEPLOYMENT_LOG"

    echo "🌐 THANOS Node (Primary Compute):" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ IntelOwl Dashboard: http://localhost" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ Neo4j Browser: http://localhost:7474" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ Grafana Monitoring: http://localhost:3000" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ Prometheus Metrics: http://localhost:9090" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ MCP API Server: http://localhost:3010" | tee -a "$DEPLOYMENT_LOG"
    echo "  └─ Vault Security: http://localhost:8200" | tee -a "$DEPLOYMENT_LOG"
    echo "" | tee -a "$DEPLOYMENT_LOG"

    echo "📊 ORACLE1 Node (ARM64 Monitoring):" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ Prometheus: http://localhost:9091" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ Grafana: http://localhost:3001" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ AlertManager: http://localhost:9093" | tee -a "$DEPLOYMENT_LOG"
    echo "  └─ Node Exporter: http://localhost:9100" | tee -a "$DEPLOYMENT_LOG"
    echo "" | tee -a "$DEPLOYMENT_LOG"

    echo "🔧 Management Commands:" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ Health Check: ./health_check_all_services.sh" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ Cross-Node Test: ./test_cross_node_integration.sh" | tee -a "$DEPLOYMENT_LOG"
    echo "  ├─ Emergency Stop: ./emergency_stop_all.sh" | tee -a "$DEPLOYMENT_LOG"
    echo "  └─ Full Validation: ./validate_complete_deployment.sh" | tee -a "$DEPLOYMENT_LOG"
    echo "" | tee -a "$DEPLOYMENT_LOG"

    echo "📁 Deployment Log: $DEPLOYMENT_LOG" | tee -a "$DEPLOYMENT_LOG"
    echo "📅 Deployment Time: $(date)" | tee -a "$DEPLOYMENT_LOG"
}

# Emergency procedures
create_emergency_procedures() {
    log_info "Creating emergency management scripts..."

    # Emergency stop script
    cat > "$SCRIPT_DIR/emergency_stop_all.sh" << 'EOF'
#!/bin/bash
# Emergency Stop - All BEV Services
echo "🚨 EMERGENCY STOP: Stopping all BEV services..."

# Stop THANOS services
echo "Stopping THANOS services..."
docker-compose -f docker-compose-thanos-unified.yml --env-file .env.thanos.complete down -t 5 || true

# Stop ORACLE1 services
echo "Stopping ORACLE1 services..."
docker-compose -f docker-compose-oracle1-unified.yml --env-file .env.oracle1.complete down -t 5 || true

# Stop any remaining BEV containers
echo "Stopping any remaining BEV containers..."
docker ps --format "{{.Names}}" | grep "bev_" | xargs -r docker stop || true

echo "✅ Emergency stop completed"
EOF

    chmod +x "$SCRIPT_DIR/emergency_stop_all.sh"

    # Health check script
    cat > "$SCRIPT_DIR/health_check_all_services.sh" << 'EOF'
#!/bin/bash
# Health Check - All BEV Services
echo "🔍 HEALTH CHECK: Checking all BEV services..."

echo "=== THANOS Services ==="
docker-compose -f docker-compose-thanos-unified.yml --env-file .env.thanos.complete ps

echo "=== ORACLE1 Services ==="
docker-compose -f docker-compose-oracle1-unified.yml --env-file .env.oracle1.complete ps

echo "=== Resource Usage ==="
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep bev || echo "No BEV containers running"

echo "✅ Health check completed"
EOF

    chmod +x "$SCRIPT_DIR/health_check_all_services.sh"

    log_success "Emergency procedures created"
}

# Usage information
show_usage() {
    echo "BEV OSINT Framework - Complete Deployment Script"
    echo ""
    echo "Usage: $0 [MODE]"
    echo ""
    echo "Deployment Modes:"
    echo "  full         - Deploy both THANOS and ORACLE1 nodes (default)"
    echo "  thanos-only  - Deploy only THANOS node"
    echo "  oracle1-only - Deploy only ORACLE1 node"
    echo "  validate     - Run validation tests only"
    echo "  test         - Run integration tests only"
    echo ""
    echo "Examples:"
    echo "  $0                # Full deployment"
    echo "  $0 thanos-only    # Deploy THANOS only"
    echo "  $0 validate       # Validate existing deployment"
    echo ""
}

# Main deployment orchestration
main() {
    # Start deployment logging
    echo "Starting BEV deployment at $(date)" > "$DEPLOYMENT_LOG"

    case "$DEPLOYMENT_MODE" in
        "full")
            log_header "🚀 BEV OSINT FRAMEWORK - FULL DEPLOYMENT"
            check_system_requirements
            create_emergency_procedures

            if deploy_thanos_node; then
                log_success "THANOS deployment phase completed"
                sleep 30  # Allow THANOS to stabilize

                if deploy_oracle1_node; then
                    log_success "ORACLE1 deployment phase completed"
                    sleep 20  # Allow ORACLE1 to stabilize

                    test_integration
                    run_validation
                    show_deployment_summary
                    log_success "🎉 FULL DEPLOYMENT COMPLETED SUCCESSFULLY!"
                else
                    log_error "ORACLE1 deployment failed"
                    exit 1
                fi
            else
                log_error "THANOS deployment failed"
                exit 1
            fi
            ;;

        "thanos-only")
            log_header "🚀 BEV OSINT FRAMEWORK - THANOS NODE ONLY"
            check_system_requirements
            create_emergency_procedures

            if deploy_thanos_node; then
                log_success "🎉 THANOS DEPLOYMENT COMPLETED SUCCESSFULLY!"
                echo "Next step: Deploy ORACLE1 with: $0 oracle1-only"
            else
                log_error "THANOS deployment failed"
                exit 1
            fi
            ;;

        "oracle1-only")
            log_header "📊 BEV OSINT FRAMEWORK - ORACLE1 NODE ONLY"
            check_system_requirements
            create_emergency_procedures

            if deploy_oracle1_node; then
                test_integration
                log_success "🎉 ORACLE1 DEPLOYMENT COMPLETED SUCCESSFULLY!"
            else
                log_error "ORACLE1 deployment failed"
                exit 1
            fi
            ;;

        "validate")
            log_header "✅ BEV OSINT FRAMEWORK - VALIDATION ONLY"
            run_validation
            ;;

        "test")
            log_header "🔗 BEV OSINT FRAMEWORK - INTEGRATION TESTING"
            test_integration
            ;;

        "help"|"-h"|"--help")
            show_usage
            exit 0
            ;;

        *)
            log_error "Unknown deployment mode: $DEPLOYMENT_MODE"
            show_usage
            exit 1
            ;;
    esac
}

# Handle script termination
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Check if help is requested
if [[ "${1:-}" =~ ^(-h|--help|help)$ ]]; then
    show_usage
    exit 0
fi

# Run main deployment
main "$@"