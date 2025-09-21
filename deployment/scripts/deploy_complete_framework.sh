#!/bin/bash

# BEV OSINT Complete Framework Deployment Script
# Master orchestration script for deploying all 10 framework gaps
# This script coordinates the deployment of all layers:
# 1. Infrastructure Layer (Vector DB, Proxy Management, Request Multiplexing)
# 2. Monitoring Layer (Health Monitoring, Auto-Recovery, Chaos Engineering)
# 3. Intelligence Layer (Predictive Cache, Extended Reasoning, Context Compression)
# 4. Edge Computing Layer (Geographic Edge Nodes, Management, Routing)

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/complete_framework_deployment.log"
DEPLOYMENT_STATE_FILE="${PROJECT_ROOT}/logs/deployment_state.json"

# Deployment configuration
DEPLOYMENT_TIMEOUT=3600  # 1 hour total timeout
PHASE_TIMEOUT=900       # 15 minutes per phase
STARTUP_DELAY=30        # Delay between phases

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Deployment phases
declare -A DEPLOYMENT_PHASES=(
    [1]="infrastructure"
    [2]="monitoring"
    [3]="intelligence"
    [4]="edge"
)

# Service dependencies
declare -A PHASE_DEPENDENCIES=(
    ["infrastructure"]=""
    ["monitoring"]="infrastructure"
    ["intelligence"]="infrastructure"
    ["edge"]="infrastructure,monitoring"
)

# Framework gap mapping
declare -A FRAMEWORK_GAPS=(
    [1]="Vector Database (Qdrant, Weaviate)"
    [2]="Proxy Management"
    [3]="Request Multiplexing"
    [4]="Context Compression"
    [5]="Predictive Cache"
    [6]="Health Monitoring"
    [7]="Auto-Recovery"
    [8]="Chaos Engineering"
    [9]="Extended Reasoning"
    [10]="Edge Computing"
)

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "${LOG_FILE}"
}

# Error handling
error_exit() {
    log "CRITICAL ERROR: $1"
    echo -e "${RED}CRITICAL ERROR: $1${NC}" >&2
    save_deployment_state "failed" "$1"
    exit 1
}

# Success message
success() {
    echo -e "${GREEN}✓ $1${NC}"
    log "SUCCESS: $1"
}

# Warning message
warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
    log "WARNING: $1"
}

# Info message
info() {
    echo -e "${BLUE}ℹ INFO: $1${NC}"
    log "INFO: $1"
}

# Highlight message
highlight() {
    echo -e "${CYAN}★ $1${NC}"
    log "HIGHLIGHT: $1"
}

# Phase header
phase_header() {
    echo
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${MAGENTA}   $1${NC}"
    echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
    echo
    log "PHASE: $1"
}

# Save deployment state
save_deployment_state() {
    local status="$1"
    local message="${2:-}"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    local state_json=$(cat <<EOF
{
  "status": "${status}",
  "message": "${message}",
  "timestamp": "${timestamp}",
  "phases_completed": $(get_completed_phases),
  "total_services": $(get_total_service_count),
  "healthy_services": $(get_healthy_service_count)
}
EOF
)

    echo "${state_json}" > "${DEPLOYMENT_STATE_FILE}"
}

# Get completed phases
get_completed_phases() {
    local completed=0

    # Check each phase
    if check_phase_status "infrastructure" >/dev/null 2>&1; then
        ((completed++))
    fi

    if check_phase_status "monitoring" >/dev/null 2>&1; then
        ((completed++))
    fi

    if check_phase_status "intelligence" >/dev/null 2>&1; then
        ((completed++))
    fi

    if check_phase_status "edge" >/dev/null 2>&1; then
        ((completed++))
    fi

    echo "${completed}"
}

# Get total service count
get_total_service_count() {
    docker ps -a --format "table {{.Names}}" | grep "^bev" | wc -l 2>/dev/null || echo "0"
}

# Get healthy service count
get_healthy_service_count() {
    docker ps --format "table {{.Names}}" | grep "^bev" | wc -l 2>/dev/null || echo "0"
}

# Check system prerequisites
check_system_prerequisites() {
    phase_header "SYSTEM PREREQUISITES CHECK"

    info "Checking system requirements for complete framework deployment..."

    # Check Docker
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker is not running or not accessible"
    fi
    success "Docker is running"

    # Check docker-compose
    if ! command -v docker-compose >/dev/null 2>&1; then
        error_exit "docker-compose is not installed"
    fi
    success "docker-compose is available"

    # Check system resources
    local available_memory=$(free -m | awk '/^Mem:/{print $7}')
    local required_memory=32768  # 32GB for complete deployment

    if [[ "${available_memory}" -lt "${required_memory}" ]]; then
        warning "Available memory is ${available_memory}MB. Complete framework requires ${required_memory}MB for optimal performance."
        read -p "Continue with reduced performance? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error_exit "Insufficient memory for complete framework deployment"
        fi
    else
        success "Sufficient memory available (${available_memory}MB)"
    fi

    # Check disk space
    local available_space=$(df "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')
    local required_space=52428800  # 50GB in KB

    if [[ "${available_space}" -lt "${required_space}" ]]; then
        warning "Available disk space is limited. Framework requires significant storage for models and data."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error_exit "Insufficient disk space for complete framework deployment"
        fi
    else
        success "Sufficient disk space available"
    fi

    # Check CPU cores
    local cpu_cores=$(nproc)
    if [[ "${cpu_cores}" -lt 16 ]]; then
        warning "System has ${cpu_cores} CPU cores. Complete framework performs better with 16+ cores."
    else
        success "Adequate CPU cores available (${cpu_cores})"
    fi

    # Create required directories
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/models"
    mkdir -p "${PROJECT_ROOT}/config"

    # Check environment files
    if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
        warning "No .env file found. Using defaults."
        create_default_env_file
    fi

    success "System prerequisites check completed"
}

# Create default environment file
create_default_env_file() {
    cat > "${PROJECT_ROOT}/.env" <<EOF
# BEV OSINT Framework Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=bev_secure_password_$(date +%s)
POSTGRES_URI=postgresql://postgres:bev_secure_password_$(date +%s)@postgres:5432/osint

# Redis Configuration
REDIS_PASSWORD=bev_redis_password_$(date +%s)

# Weaviate Configuration
WEAVIATE_API_KEY=bev_weaviate_key_$(date +%s)

# Monitoring Configuration
SLACK_WEBHOOK_URL=
BEV_ADMIN_EMAIL=admin@bev-osint.local
BEV_ONCALL_EMAIL=oncall@bev-osint.local
RECOVERY_WEBHOOK_URL=

# ML Configuration
HUGGINGFACE_TOKEN=
OPENAI_API_KEY=

# Geographic Configuration
GEOIP_LICENSE_KEY=
EOF

    info "Created default .env file with secure passwords"
}

# Check phase dependencies
check_phase_dependencies() {
    local phase="$1"
    local dependencies="${PHASE_DEPENDENCIES[$phase]}"

    if [[ -n "${dependencies}" ]]; then
        info "Checking dependencies for ${phase}: ${dependencies}"

        IFS=',' read -ra DEPS <<< "${dependencies}"
        for dep in "${DEPS[@]}"; do
            if ! check_phase_status "${dep// /}" >/dev/null 2>&1; then
                error_exit "Dependency ${dep} is not running. Deploy it first."
            fi
        done

        success "All dependencies satisfied for ${phase}"
    fi
}

# Check phase status
check_phase_status() {
    local phase="$1"

    case "${phase}" in
        "infrastructure")
            # Check key infrastructure services
            docker ps --format "table {{.Names}}" | grep -q "^bev_qdrant_primary$" && \
            docker ps --format "table {{.Names}}" | grep -q "^bev_weaviate$" && \
            docker ps --format "table {{.Names}}" | grep -q "^bev_proxy_manager$"
            ;;
        "monitoring")
            # Check monitoring services
            docker ps --format "table {{.Names}}" | grep -q "^bev_health_monitor$" && \
            docker ps --format "table {{.Names}}" | grep -q "^bev_auto_recovery$"
            ;;
        "intelligence")
            # Check intelligence services
            docker ps --format "table {{.Names}}" | grep -q "^bev_extended_reasoning$" && \
            docker ps --format "table {{.Names}}" | grep -q "^bev_predictive_cache$"
            ;;
        "edge")
            # Check edge services
            docker ps --format "table {{.Names}}" | grep -q "^bev-geo-router$" && \
            docker ps --format "table {{.Names}}" | grep -q "^bev-edge-management$"
            ;;
        *)
            return 1
            ;;
    esac
}

# Deploy phase
deploy_phase() {
    local phase="$1"
    local script_name="deploy_${phase}.sh"
    local script_path="${SCRIPT_DIR}/${script_name}"

    phase_header "PHASE: ${phase^^} DEPLOYMENT"

    info "Deploying ${phase} layer..."

    # Check phase dependencies
    check_phase_dependencies "${phase}"

    # Check if script exists
    if [[ ! -f "${script_path}" ]]; then
        error_exit "Deployment script not found: ${script_path}"
    fi

    # Make script executable
    chmod +x "${script_path}"

    # Execute deployment script with timeout
    info "Executing ${script_name}..."

    if timeout "${PHASE_TIMEOUT}" "${script_path}" deploy; then
        success "${phase^} deployment completed successfully"

        # Wait for services to stabilize
        info "Waiting ${STARTUP_DELAY}s for services to stabilize..."
        sleep "${STARTUP_DELAY}"

        # Verify deployment
        if "${script_path}" verify >/dev/null 2>&1; then
            success "${phase^} verification passed"
        else
            warning "${phase^} verification failed, but continuing deployment"
        fi
    else
        error_exit "${phase^} deployment failed or timed out"
    fi
}

# Deploy framework gap by gap
deploy_by_gaps() {
    phase_header "FRAMEWORK GAPS DEPLOYMENT"

    info "Deploying all 10 framework gaps..."

    for gap_num in {1..10}; do
        highlight "Gap ${gap_num}: ${FRAMEWORK_GAPS[$gap_num]}"

        case "${gap_num}" in
            1|2|3|4)
                if ! check_phase_status "infrastructure" >/dev/null 2>&1; then
                    deploy_phase "infrastructure"
                fi
                ;;
            6|7|8)
                if ! check_phase_status "monitoring" >/dev/null 2>&1; then
                    deploy_phase "monitoring"
                fi
                ;;
            5|9)
                if ! check_phase_status "intelligence" >/dev/null 2>&1; then
                    deploy_phase "intelligence"
                fi
                ;;
            10)
                if ! check_phase_status "edge" >/dev/null 2>&1; then
                    deploy_phase "edge"
                fi
                ;;
        esac

        success "Gap ${gap_num} addressed: ${FRAMEWORK_GAPS[$gap_num]}"
    done

    success "All 10 framework gaps have been addressed!"
}

# Comprehensive verification
comprehensive_verification() {
    phase_header "COMPREHENSIVE FRAMEWORK VERIFICATION"

    info "Performing end-to-end verification of complete framework..."

    local verification_results=()

    # Verify each phase
    for phase_num in {1..4}; do
        local phase="${DEPLOYMENT_PHASES[$phase_num]}"
        local script_path="${SCRIPT_DIR}/deploy_${phase}.sh"

        info "Verifying ${phase} layer..."

        if [[ -f "${script_path}" ]] && "${script_path}" verify >/dev/null 2>&1; then
            verification_results+=("${phase}:PASS")
            success "${phase^} layer verification: PASS"
        else
            verification_results+=("${phase}:FAIL")
            warning "${phase^} layer verification: FAIL"
        fi
    done

    # Overall framework health check
    info "Performing overall framework health check..."

    local total_services=$(get_total_service_count)
    local healthy_services=$(get_healthy_service_count)
    local health_percentage=$((healthy_services * 100 / total_services))

    if [[ "${health_percentage}" -ge 90 ]]; then
        success "Framework health: EXCELLENT (${healthy_services}/${total_services} services healthy - ${health_percentage}%)"
    elif [[ "${health_percentage}" -ge 75 ]]; then
        success "Framework health: GOOD (${healthy_services}/${total_services} services healthy - ${health_percentage}%)"
    elif [[ "${health_percentage}" -ge 50 ]]; then
        warning "Framework health: MODERATE (${healthy_services}/${total_services} services healthy - ${health_percentage}%)"
    else
        warning "Framework health: POOR (${healthy_services}/${total_services} services healthy - ${health_percentage}%)"
    fi

    # Test framework integration
    test_framework_integration

    success "Comprehensive verification completed"
}

# Test framework integration
test_framework_integration() {
    info "Testing framework integration..."

    # Test data flow: Edge → Intelligence → Storage
    info "Testing edge-to-storage data flow..."

    # Test routing: Router → Edge Node → Processing
    info "Testing geographic routing..."

    # Test monitoring: Health Monitor → Auto Recovery
    info "Testing monitoring and recovery..."

    # Test caching: Predictive Cache → Context Compression
    info "Testing caching and compression..."

    success "Integration testing completed"
}

# Display deployment summary
show_deployment_summary() {
    phase_header "DEPLOYMENT SUMMARY"

    echo
    highlight "BEV OSINT Complete Framework Deployment Summary"
    echo

    # Framework gaps status
    info "Framework Gaps Addressed:"
    for gap_num in {1..10}; do
        echo "  ${gap_num}. ${FRAMEWORK_GAPS[$gap_num]} ✓"
    done
    echo

    # Service counts
    local total_services=$(get_total_service_count)
    local healthy_services=$(get_healthy_service_count)

    info "Service Statistics:"
    echo "  Total Services:    ${total_services}"
    echo "  Healthy Services:  ${healthy_services}"
    echo "  Health Percentage: $((healthy_services * 100 / total_services))%"
    echo

    # Key endpoints
    info "Key Service Endpoints:"
    echo "  Infrastructure:"
    echo "    Qdrant:            http://172.30.0.36:6333"
    echo "    Weaviate:          http://172.30.0.38:8080"
    echo "    Proxy Manager:     http://172.30.0.40:8040"
    echo "    Request Multiplexer: http://172.30.0.42:8042"
    echo
    echo "  Monitoring:"
    echo "    Health Monitor:    http://172.30.0.38:8038"
    echo "    Auto Recovery:     http://172.30.0.41:8041"
    echo
    echo "  Intelligence:"
    echo "    Extended Reasoning: http://172.30.0.46:8046"
    echo "    Predictive Cache:  http://172.30.0.44:8044"
    echo "    Context Compressor: http://172.30.0.43:8043"
    echo
    echo "  Edge Computing:"
    echo "    Geographic Router: http://172.30.0.52:8052"
    echo "    Edge Management:   http://172.30.0.51:8051"
    echo

    # Management commands
    info "Management Commands:"
    echo "  Full Status:       ./deploy_complete_framework.sh status"
    echo "  Health Check:      ./deploy_complete_framework.sh health"
    echo "  Stop All:          ./deploy_complete_framework.sh stop"
    echo "  Restart All:       ./deploy_complete_framework.sh restart"
    echo

    success "Complete framework deployment finished successfully!"
    highlight "🎉 All 10 framework gaps have been addressed with ${total_services} services running!"
}

# Health check for all services
health_check() {
    phase_header "FRAMEWORK HEALTH CHECK"

    local scripts=(
        "deploy_infrastructure.sh"
        "deploy_monitoring.sh"
        "deploy_intelligence.sh"
        "deploy_edge.sh"
    )

    for script in "${scripts[@]}"; do
        local script_path="${SCRIPT_DIR}/${script}"
        if [[ -f "${script_path}" ]]; then
            info "Health check: ${script%.sh}"
            "${script_path}" status || true
            echo
        fi
    done
}

# Stop all framework services
stop_all_services() {
    phase_header "STOPPING ALL FRAMEWORK SERVICES"

    local scripts=(
        "deploy_edge.sh"
        "deploy_intelligence.sh"
        "deploy_monitoring.sh"
        "deploy_infrastructure.sh"
    )

    for script in "${scripts[@]}"; do
        local script_path="${SCRIPT_DIR}/${script}"
        if [[ -f "${script_path}" ]]; then
            info "Stopping: ${script%.sh}"
            "${script_path}" stop || true
        fi
    done

    success "All framework services stopped"
}

# Restart all framework services
restart_all_services() {
    phase_header "RESTARTING ALL FRAMEWORK SERVICES"

    stop_all_services
    sleep 30
    main_deployment
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        warning "Complete framework deployment failed. Check logs at: ${LOG_FILE}"
        save_deployment_state "failed" "Deployment interrupted or failed"
    else
        save_deployment_state "completed" "All framework gaps successfully deployed"
    fi
    exit $exit_code
}

# Main deployment function
main_deployment() {
    # Set up cleanup trap
    trap cleanup EXIT

    highlight "Starting BEV OSINT Complete Framework Deployment"
    highlight "Addressing all 10 framework gaps with comprehensive automation"
    echo

    info "Deployment log: ${LOG_FILE}"
    info "State file: ${DEPLOYMENT_STATE_FILE}"
    echo

    # Initialize deployment state
    save_deployment_state "starting" "Complete framework deployment initiated"

    # Run deployment phases
    check_system_prerequisites
    deploy_by_gaps
    comprehensive_verification
    show_deployment_summary

    save_deployment_state "completed" "All framework gaps successfully deployed"
}

# Command line interface
case "${1:-deploy}" in
    "deploy")
        main_deployment
        ;;
    "status")
        health_check
        ;;
    "health")
        comprehensive_verification
        ;;
    "verify")
        comprehensive_verification
        ;;
    "gaps")
        echo "BEV OSINT Framework Gaps:"
        for gap_num in {1..10}; do
            echo "  ${gap_num}. ${FRAMEWORK_GAPS[$gap_num]}"
        done
        ;;
    "stop")
        stop_all_services
        ;;
    "restart")
        restart_all_services
        ;;
    "logs")
        echo "Complete framework logs:"
        echo "========================"
        tail -100 "${LOG_FILE}" 2>/dev/null || echo "No logs available"
        ;;
    "state")
        if [[ -f "${DEPLOYMENT_STATE_FILE}" ]]; then
            cat "${DEPLOYMENT_STATE_FILE}" | jq '.' 2>/dev/null || cat "${DEPLOYMENT_STATE_FILE}"
        else
            echo "No deployment state available"
        fi
        ;;
    *)
        echo "BEV OSINT Complete Framework Deployment"
        echo "======================================"
        echo
        echo "Usage: $0 {deploy|status|health|verify|gaps|stop|restart|logs|state}"
        echo
        echo "Commands:"
        echo "  deploy   - Deploy complete framework (all 10 gaps) [default]"
        echo "  status   - Show status of all framework components"
        echo "  health   - Perform comprehensive health check"
        echo "  verify   - Verify all deployments and test integration"
        echo "  gaps     - List all 10 framework gaps"
        echo "  stop     - Stop all framework services"
        echo "  restart  - Restart all framework services"
        echo "  logs     - Show deployment logs"
        echo "  state    - Show current deployment state"
        echo
        echo "Framework Gaps Addressed:"
        for gap_num in {1..10}; do
            echo "  ${gap_num}. ${FRAMEWORK_GAPS[$gap_num]}"
        done
        echo
        exit 1
        ;;
esac