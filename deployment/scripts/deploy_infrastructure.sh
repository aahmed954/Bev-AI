#!/bin/bash

# BEV OSINT Infrastructure Layer Deployment Script
# This script deploys the foundation infrastructure services:
# - Vector Databases (Qdrant, Weaviate)
# - Proxy Management
# - Request Multiplexing
# - Context Compression

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose-infrastructure.yml"
LOG_FILE="${PROJECT_ROOT}/logs/infrastructure_deployment.log"
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
    info "Checking prerequisites..."

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker is not running or not accessible"
    fi

    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        error_exit "docker-compose is not installed"
    fi

    # Check if compose file exists
    if [[ ! -f "${COMPOSE_FILE}" ]]; then
        error_exit "Infrastructure compose file not found: ${COMPOSE_FILE}"
    fi

    # Check if network exists
    if ! docker network inspect bev_osint >/dev/null 2>&1; then
        info "Creating bev_osint network..."
        docker network create --driver bridge \
            --subnet=172.30.0.0/16 \
            bev_osint || error_exit "Failed to create network"
    fi

    # Create logs directory
    mkdir -p "${PROJECT_ROOT}/logs"

    success "Prerequisites check completed"
}

# Check if infrastructure is already running
check_existing_infrastructure() {
    info "Checking existing infrastructure..."

    local running_services=()
    local services=(
        "bev_qdrant_primary"
        "bev_qdrant_replica"
        "bev_weaviate"
        "bev_weaviate_transformers"
        "bev_proxy_manager"
        "bev_request_multiplexer"
        "bev_context_compressor"
        "bev_health_monitor"
        "bev_auto_recovery"
        "bev_chaos_engineer"
    )

    for service in "${services[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^${service}$"; then
            running_services+=("${service}")
        fi
    done

    if [[ ${#running_services[@]} -gt 0 ]]; then
        warning "Found running infrastructure services: ${running_services[*]}"
        read -p "Do you want to stop and recreate them? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info "Stopping existing infrastructure services..."
            docker-compose -f "${COMPOSE_FILE}" down --remove-orphans
        else
            info "Skipping deployment - existing services will remain"
            return 1
        fi
    fi

    return 0
}

# Validate environment variables
validate_environment() {
    info "Validating environment variables..."

    local required_vars=(
        "WEAVIATE_API_KEY"
    )

    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("${var}")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error_exit "Missing required environment variables: ${missing_vars[*]}"
    fi

    success "Environment validation completed"
}

# Deploy infrastructure services in phases
deploy_infrastructure() {
    info "Starting infrastructure deployment..."

    # Phase 1: Foundation Services (Vector Databases)
    info "Phase 1: Deploying vector databases..."
    docker-compose -f "${COMPOSE_FILE}" up -d \
        qdrant-primary \
        qdrant-replica \
        weaviate \
        weaviate-transformers \
        || error_exit "Failed to deploy vector databases"

    # Wait for vector databases to be ready
    wait_for_service_health "qdrant-primary" "http://172.30.0.36:6333/health"
    wait_for_service_health "qdrant-replica" "http://172.30.0.37:6333/health"
    wait_for_service_health "weaviate" "http://172.30.0.38:8080/v1/.well-known/ready"

    success "Phase 1 completed: Vector databases deployed"

    # Phase 2: Infrastructure Services
    info "Phase 2: Deploying infrastructure services..."
    docker-compose -f "${COMPOSE_FILE}" up -d \
        proxy-manager \
        request-multiplexer \
        context-compressor \
        || error_exit "Failed to deploy infrastructure services"

    # Wait for infrastructure services to be ready
    wait_for_service_health "proxy-manager" "http://172.30.0.40:8080/health"
    wait_for_service_health "request-multiplexer" "http://172.30.0.42:8080/health"
    wait_for_service_health "context-compressor" "http://172.30.0.43:8080/health"

    success "Phase 2 completed: Infrastructure services deployed"

    # Phase 3: Monitoring and Recovery
    info "Phase 3: Deploying monitoring and recovery services..."
    docker-compose -f "${COMPOSE_FILE}" up -d \
        health-monitor \
        auto-recovery \
        chaos-engineer \
        || error_exit "Failed to deploy monitoring services"

    # Wait for monitoring services to be ready
    wait_for_service_health "health-monitor" "http://172.30.0.38:8080/health"
    wait_for_service_health "auto-recovery" "http://172.30.0.41:8080/health"
    wait_for_service_health "chaos-engineer" "http://172.30.0.45:8080/health"

    success "Phase 3 completed: Monitoring and recovery services deployed"

    success "Infrastructure deployment completed successfully"
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

    error_exit "${service_name} failed to become healthy after ${max_attempts} attempts"
}

# Verify deployment
verify_deployment() {
    info "Verifying infrastructure deployment..."

    local services=(
        "qdrant-primary:http://172.30.0.36:6333/health"
        "qdrant-replica:http://172.30.0.37:6333/health"
        "weaviate:http://172.30.0.38:8080/v1/.well-known/ready"
        "proxy-manager:http://172.30.0.40:8080/health"
        "request-multiplexer:http://172.30.0.42:8080/health"
        "context-compressor:http://172.30.0.43:8080/health"
        "health-monitor:http://172.30.0.38:8080/health"
        "auto-recovery:http://172.30.0.41:8080/health"
        "chaos-engineer:http://172.30.0.45:8080/health"
    )

    local failed_services=()

    for service_health in "${services[@]}"; do
        local service_name="${service_health%%:*}"
        local health_url="${service_health#*:}"

        if ! curl -sf "${health_url}" >/dev/null 2>&1; then
            failed_services+=("${service_name}")
        fi
    done

    if [[ ${#failed_services[@]} -gt 0 ]]; then
        warning "Some services are not healthy: ${failed_services[*]}"
        return 1
    fi

    success "All infrastructure services are healthy"
    return 0
}

# Display service status
show_service_status() {
    info "Infrastructure Service Status:"
    echo

    docker-compose -f "${COMPOSE_FILE}" ps

    echo
    info "Service Endpoints:"
    echo "  Qdrant Primary:      http://172.30.0.36:6333"
    echo "  Qdrant Replica:      http://172.30.0.37:6343"
    echo "  Weaviate:            http://172.30.0.38:8080"
    echo "  Proxy Manager:       http://172.30.0.40:8040"
    echo "  Request Multiplexer: http://172.30.0.42:8042"
    echo "  Context Compressor:  http://172.30.0.43:8043"
    echo "  Health Monitor:      http://172.30.0.38:8038"
    echo "  Auto Recovery:       http://172.30.0.41:8041"
    echo "  Chaos Engineer:      http://172.30.0.45:8045"
    echo
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        warning "Deployment failed. Check logs at: ${LOG_FILE}"
        info "To debug, run: docker-compose -f ${COMPOSE_FILE} logs --tail=100"
    fi
    exit $exit_code
}

# Main deployment function
main() {
    # Set up cleanup trap
    trap cleanup EXIT

    info "Starting BEV OSINT Infrastructure Deployment"
    info "Compose file: ${COMPOSE_FILE}"
    info "Log file: ${LOG_FILE}"
    echo

    # Run deployment steps
    check_prerequisites
    validate_environment

    if check_existing_infrastructure; then
        deploy_infrastructure

        if verify_deployment; then
            show_service_status
            success "Infrastructure deployment completed successfully!"
        else
            warning "Deployment completed but some services may not be fully ready"
            warning "Run './deploy_infrastructure.sh verify' to check status again"
        fi
    else
        info "Infrastructure deployment skipped"
    fi
}

# Command line interface
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "verify")
        check_prerequisites
        verify_deployment && show_service_status
        ;;
    "status")
        check_prerequisites
        show_service_status
        ;;
    "stop")
        info "Stopping infrastructure services..."
        docker-compose -f "${COMPOSE_FILE}" down
        success "Infrastructure services stopped"
        ;;
    "restart")
        info "Restarting infrastructure services..."
        docker-compose -f "${COMPOSE_FILE}" restart
        success "Infrastructure services restarted"
        ;;
    "logs")
        docker-compose -f "${COMPOSE_FILE}" logs --tail=100 -f
        ;;
    *)
        echo "Usage: $0 {deploy|verify|status|stop|restart|logs}"
        echo
        echo "Commands:"
        echo "  deploy   - Deploy infrastructure services (default)"
        echo "  verify   - Verify deployment and show status"
        echo "  status   - Show current status of services"
        echo "  stop     - Stop all infrastructure services"
        echo "  restart  - Restart all infrastructure services"
        echo "  logs     - Show logs from all services"
        exit 1
        ;;
esac