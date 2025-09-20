#!/bin/bash

# BEV OSINT Monitoring Systems Deployment Script
# This script deploys the monitoring and recovery services:
# - Health Monitoring
# - Auto-Recovery
# - Chaos Engineering

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INFRASTRUCTURE_COMPOSE="${PROJECT_ROOT}/docker-compose-infrastructure.yml"
MAIN_COMPOSE="${PROJECT_ROOT}/docker-compose.complete.yml"
LOG_FILE="${PROJECT_ROOT}/logs/monitoring_deployment.log"
TIMEOUT=600  # 10 minutes
HEALTH_CHECK_RETRIES=30
HEALTH_CHECK_INTERVAL=10

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "${LOG_FILE}"
}

# Error handling
error_exit() {
    log "ERROR: $1"
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

# Success message
success() {
    echo -e "${GREEN}$1${NC}"
    log "SUCCESS: $1"
}

# Warning message
warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
    log "WARNING: $1"
}

# Info message
info() {
    echo -e "${BLUE}INFO: $1${NC}"
    log "INFO: $1"
}

# Check prerequisites
check_prerequisites() {
    info "Checking monitoring prerequisites..."

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker is not running or not accessible"
    fi

    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        error_exit "docker-compose is not installed"
    fi

    # Check if compose files exist
    if [[ ! -f "${INFRASTRUCTURE_COMPOSE}" ]]; then
        error_exit "Infrastructure compose file not found: ${INFRASTRUCTURE_COMPOSE}"
    fi

    if [[ ! -f "${MAIN_COMPOSE}" ]]; then
        error_exit "Main compose file not found: ${MAIN_COMPOSE}"
    fi

    # Check if core infrastructure is running
    local required_services=(
        "bev_postgres"
        "bev_redis_1"
    )

    for service in "${required_services[@]}"; do
        if ! docker ps --format "table {{.Names}}" | grep -q "^${service}$"; then
            error_exit "Required service ${service} is not running. Deploy infrastructure first."
        fi
    done

    # Create logs directory
    mkdir -p "${PROJECT_ROOT}/logs"

    success "Prerequisites check completed"
}

# Check existing monitoring services
check_existing_monitoring() {
    info "Checking existing monitoring services..."

    local running_services=()
    local monitoring_services=(
        "bev_health_monitor"
        "bev_auto_recovery"
        "bev_chaos_engineer"
        "bev_extended_reasoning_recovery"
        "bev_context_compressor_recovery"
    )

    for service in "${monitoring_services[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^${service}$"; then
            running_services+=("${service}")
        fi
    done

    if [[ ${#running_services[@]} -gt 0 ]]; then
        warning "Found running monitoring services: ${running_services[*]}"
        read -p "Do you want to stop and recreate them? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info "Stopping existing monitoring services..."
            stop_monitoring_services
        else
            info "Skipping deployment - existing services will remain"
            return 1
        fi
    fi

    return 0
}

# Stop monitoring services
stop_monitoring_services() {
    info "Stopping monitoring services..."

    # Stop services from main compose
    docker-compose -f "${MAIN_COMPOSE}" stop \
        extended-reasoning-recovery \
        context-compressor-recovery \
        2>/dev/null || true

    # Stop services from infrastructure compose
    docker-compose -f "${INFRASTRUCTURE_COMPOSE}" stop \
        health-monitor \
        auto-recovery \
        chaos-engineer \
        2>/dev/null || true

    success "Monitoring services stopped"
}

# Deploy monitoring services
deploy_monitoring() {
    info "Starting monitoring systems deployment..."

    # Phase 1: Core Monitoring Infrastructure
    info "Phase 1: Deploying health monitoring..."
    docker-compose -f "${INFRASTRUCTURE_COMPOSE}" up -d health-monitor \
        || error_exit "Failed to deploy health monitor"

    wait_for_service_health "health-monitor" "http://172.30.0.38:8080/health"
    success "Phase 1 completed: Health monitoring deployed"

    # Phase 2: Auto-Recovery System
    info "Phase 2: Deploying auto-recovery system..."
    docker-compose -f "${INFRASTRUCTURE_COMPOSE}" up -d auto-recovery \
        || error_exit "Failed to deploy auto-recovery"

    wait_for_service_health "auto-recovery" "http://172.30.0.41:8080/health"
    success "Phase 2 completed: Auto-recovery system deployed"

    # Phase 3: Service-Specific Recovery Services
    info "Phase 3: Deploying service-specific recovery services..."

    # Deploy extended reasoning recovery if extended reasoning exists
    if docker ps --format "table {{.Names}}" | grep -q "^bev_extended_reasoning$"; then
        docker-compose -f "${MAIN_COMPOSE}" up -d extended-reasoning-recovery \
            || warning "Failed to deploy extended reasoning recovery"
        wait_for_service_health "extended-reasoning-recovery" "http://172.30.0.53:8080/health" || true
    fi

    # Deploy context compressor recovery if context compressor exists
    if docker ps --format "table {{.Names}}" | grep -q "^bev_context_compressor$"; then
        docker-compose -f "${MAIN_COMPOSE}" up -d context-compressor-recovery \
            || warning "Failed to deploy context compressor recovery"
        wait_for_service_health "context-compressor-recovery" "http://172.30.0.54:8080/health" || true
    fi

    success "Phase 3 completed: Service-specific recovery services deployed"

    # Phase 4: Chaos Engineering (Optional)
    info "Phase 4: Deploying chaos engineering (optional)..."
    read -p "Do you want to deploy chaos engineering? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f "${INFRASTRUCTURE_COMPOSE}" up -d chaos-engineer \
            || warning "Failed to deploy chaos engineer"
        wait_for_service_health "chaos-engineer" "http://172.30.0.45:8080/health" || true
        success "Chaos engineering deployed"
    else
        info "Skipping chaos engineering deployment"
    fi

    success "Monitoring systems deployment completed"
}

# Wait for service health
wait_for_service_health() {
    local service_name="$1"
    local health_url="$2"
    local max_attempts="${HEALTH_CHECK_RETRIES}"
    local attempt=1

    info "Waiting for ${service_name} to become healthy..."

    while [[ $attempt -le $max_attempts ]]; do
        if curl -sf "${health_url}" >/dev/null 2>&1; then
            success "${service_name} is healthy"
            return 0
        fi

        info "Attempt ${attempt}/${max_attempts}: ${service_name} not ready yet, waiting ${HEALTH_CHECK_INTERVAL}s..."
        sleep "${HEALTH_CHECK_INTERVAL}"
        ((attempt++))
    done

    warning "${service_name} failed to become healthy after ${max_attempts} attempts"
    return 1
}

# Verify monitoring deployment
verify_monitoring() {
    info "Verifying monitoring deployment..."

    local services=(
        "health-monitor:http://172.30.0.38:8080/health"
        "auto-recovery:http://172.30.0.41:8080/health"
    )

    # Check optional services
    if docker ps --format "table {{.Names}}" | grep -q "^bev_extended_reasoning_recovery$"; then
        services+=("extended-reasoning-recovery:http://172.30.0.53:8080/health")
    fi

    if docker ps --format "table {{.Names}}" | grep -q "^bev_context_compressor_recovery$"; then
        services+=("context-compressor-recovery:http://172.30.0.54:8080/health")
    fi

    if docker ps --format "table {{.Names}}" | grep -q "^bev_chaos_engineer$"; then
        services+=("chaos-engineer:http://172.30.0.45:8080/health")
    fi

    local failed_services=()

    for service_health in "${services[@]}"; do
        local service_name="${service_health%%:*}"
        local health_url="${service_health#*:}"

        if ! curl -sf "${health_url}" >/dev/null 2>&1; then
            failed_services+=("${service_name}")
        fi
    done

    if [[ ${#failed_services[@]} -gt 0 ]]; then
        warning "Some monitoring services are not healthy: ${failed_services[*]}"
        return 1
    fi

    success "All monitoring services are healthy"
    return 0
}

# Test monitoring functionality
test_monitoring() {
    info "Testing monitoring functionality..."

    local health_monitor_url="http://172.30.0.38:8080"
    local auto_recovery_url="http://172.30.0.41:8080"

    # Test health monitor API
    info "Testing health monitor API..."
    if curl -sf "${health_monitor_url}/api/services" >/dev/null 2>&1; then
        success "Health monitor API is responding"
    else
        warning "Health monitor API is not responding"
    fi

    # Test auto-recovery API
    info "Testing auto-recovery API..."
    if curl -sf "${auto_recovery_url}/api/status" >/dev/null 2>&1; then
        success "Auto-recovery API is responding"
    else
        warning "Auto-recovery API is not responding"
    fi

    # Test monitoring data collection
    info "Checking monitoring data collection..."
    sleep 30  # Wait for some data to be collected

    local health_data_check=$(curl -s "${health_monitor_url}/api/metrics" | jq -r '.services | length' 2>/dev/null || echo "0")
    if [[ "${health_data_check}" -gt 0 ]]; then
        success "Health monitoring is collecting data (${health_data_check} services)"
    else
        warning "Health monitoring may not be collecting data properly"
    fi

    success "Monitoring functionality test completed"
}

# Configure monitoring alerts
configure_alerts() {
    info "Configuring monitoring alerts..."

    local config_file="${PROJECT_ROOT}/config/monitoring/alerts.yml"

    if [[ -f "${config_file}" ]]; then
        info "Loading alert configuration from ${config_file}"

        # Apply alert configuration via API
        if curl -sf -X POST \
            -H "Content-Type: application/yaml" \
            --data-binary "@${config_file}" \
            "http://172.30.0.38:8080/api/alerts/config" >/dev/null 2>&1; then
            success "Alert configuration applied"
        else
            warning "Failed to apply alert configuration"
        fi
    else
        info "No alert configuration file found, using defaults"
    fi
}

# Display monitoring status
show_monitoring_status() {
    info "Monitoring Systems Status:"
    echo

    # Show container status
    echo "Container Status:"
    docker ps --filter "label=bev.category=monitoring" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo

    # Show service endpoints
    info "Monitoring Endpoints:"
    echo "  Health Monitor:     http://172.30.0.38:8038"
    echo "  Auto Recovery:      http://172.30.0.41:8041"

    if docker ps --format "table {{.Names}}" | grep -q "^bev_chaos_engineer$"; then
        echo "  Chaos Engineer:     http://172.30.0.45:8045"
    fi

    if docker ps --format "table {{.Names}}" | grep -q "^bev_extended_reasoning_recovery$"; then
        echo "  Extended Reasoning Recovery: http://172.30.0.53:8047"
    fi

    if docker ps --format "table {{.Names}}" | grep -q "^bev_context_compressor_recovery$"; then
        echo "  Context Compressor Recovery: http://172.30.0.54:8054"
    fi
    echo

    # Show monitoring stats
    info "Monitoring Statistics:"
    local health_stats=$(curl -s "http://172.30.0.38:8080/api/stats" 2>/dev/null | jq -r '.' 2>/dev/null || echo "{}")
    if [[ "${health_stats}" != "{}" ]]; then
        echo "  Monitored Services: $(echo "${health_stats}" | jq -r '.monitored_services // "N/A"')"
        echo "  Healthy Services:   $(echo "${health_stats}" | jq -r '.healthy_services // "N/A"')"
        echo "  Failed Services:    $(echo "${health_stats}" | jq -r '.failed_services // "N/A"')"
        echo "  Recovery Actions:   $(echo "${health_stats}" | jq -r '.recovery_actions // "N/A"')"
    else
        echo "  Statistics not available yet"
    fi
    echo
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        warning "Monitoring deployment failed. Check logs at: ${LOG_FILE}"
        info "To debug, run: docker logs <service_name>"
    fi
    exit $exit_code
}

# Main deployment function
main() {
    # Set up cleanup trap
    trap cleanup EXIT

    info "Starting BEV OSINT Monitoring Systems Deployment"
    info "Log file: ${LOG_FILE}"
    echo

    # Run deployment steps
    check_prerequisites

    if check_existing_monitoring; then
        deploy_monitoring
        configure_alerts

        if verify_monitoring; then
            test_monitoring
            show_monitoring_status
            success "Monitoring systems deployment completed successfully!"
        else
            warning "Deployment completed but some services may not be fully ready"
            warning "Run './deploy_monitoring.sh verify' to check status again"
        fi
    else
        info "Monitoring deployment skipped"
    fi
}

# Command line interface
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "verify")
        check_prerequisites
        verify_monitoring && show_monitoring_status
        ;;
    "test")
        check_prerequisites
        test_monitoring
        ;;
    "status")
        check_prerequisites
        show_monitoring_status
        ;;
    "stop")
        info "Stopping monitoring services..."
        stop_monitoring_services
        success "Monitoring services stopped"
        ;;
    "restart")
        info "Restarting monitoring services..."
        stop_monitoring_services
        sleep 5
        deploy_monitoring
        success "Monitoring services restarted"
        ;;
    "logs")
        echo "Monitoring service logs:"
        echo "========================"
        docker-compose -f "${INFRASTRUCTURE_COMPOSE}" logs --tail=50 health-monitor auto-recovery chaos-engineer 2>/dev/null || true
        docker-compose -f "${MAIN_COMPOSE}" logs --tail=50 extended-reasoning-recovery context-compressor-recovery 2>/dev/null || true
        ;;
    *)
        echo "Usage: $0 {deploy|verify|test|status|stop|restart|logs}"
        echo
        echo "Commands:"
        echo "  deploy   - Deploy monitoring services (default)"
        echo "  verify   - Verify deployment and show status"
        echo "  test     - Test monitoring functionality"
        echo "  status   - Show current status of monitoring services"
        echo "  stop     - Stop all monitoring services"
        echo "  restart  - Restart all monitoring services"
        echo "  logs     - Show logs from monitoring services"
        exit 1
        ;;
esac