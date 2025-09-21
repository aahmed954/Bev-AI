#!/bin/bash

# BEV OSINT Intelligence Enhancement Deployment Script
# This script deploys the intelligence enhancement services:
# - Predictive Cache
# - Extended Reasoning
# - Context Compression

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAIN_COMPOSE="${PROJECT_ROOT}/docker-compose.complete.yml"
INFRASTRUCTURE_COMPOSE="${PROJECT_ROOT}/docker-compose-infrastructure.yml"
LOG_FILE="${PROJECT_ROOT}/logs/intelligence_deployment.log"
TIMEOUT=900  # 15 minutes for ML models
HEALTH_CHECK_RETRIES=45
HEALTH_CHECK_INTERVAL=20

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
    info "Checking intelligence enhancement prerequisites..."

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker is not running or not accessible"
    fi

    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        error_exit "docker-compose is not installed"
    fi

    # Check if compose files exist
    if [[ ! -f "${MAIN_COMPOSE}" ]]; then
        error_exit "Main compose file not found: ${MAIN_COMPOSE}"
    fi

    if [[ ! -f "${INFRASTRUCTURE_COMPOSE}" ]]; then
        error_exit "Infrastructure compose file not found: ${INFRASTRUCTURE_COMPOSE}"
    fi

    # Check if required infrastructure is running
    local required_services=(
        "bev_postgres"
        "bev_redis_1"
        "bev_qdrant_primary"
        "bev_weaviate"
    )

    local missing_services=()
    for service in "${required_services[@]}"; do
        if ! docker ps --format "table {{.Names}}" | grep -q "^${service}$"; then
            missing_services+=("${service}")
        fi
    done

    if [[ ${#missing_services[@]} -gt 0 ]]; then
        error_exit "Required services not running: ${missing_services[*]}. Deploy infrastructure first."
    fi

    # Check if context compressor is available
    if ! docker ps --format "table {{.Names}}" | grep -q "^bev_context_compressor$"; then
        warning "Context compressor not found. Some features may be limited."
    fi

    # Create logs directory
    mkdir -p "${PROJECT_ROOT}/logs"

    success "Prerequisites check completed"
}

# Check existing intelligence services
check_existing_intelligence() {
    info "Checking existing intelligence enhancement services..."

    local running_services=()
    local intelligence_services=(
        "bev_predictive_cache"
        "bev_extended_reasoning"
        "bev_context_compressor"
    )

    for service in "${intelligence_services[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^${service}$"; then
            running_services+=("${service}")
        fi
    done

    if [[ ${#running_services[@]} -gt 0 ]]; then
        warning "Found running intelligence services: ${running_services[*]}"
        read -p "Do you want to stop and recreate them? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info "Stopping existing intelligence services..."
            stop_intelligence_services
        else
            info "Skipping deployment - existing services will remain"
            return 1
        fi
    fi

    return 0
}

# Stop intelligence services
stop_intelligence_services() {
    info "Stopping intelligence enhancement services..."

    # Stop services from main compose
    docker-compose -f "${MAIN_COMPOSE}" stop \
        predictive-cache \
        extended-reasoning \
        2>/dev/null || true

    # Stop services from infrastructure compose
    docker-compose -f "${INFRASTRUCTURE_COMPOSE}" stop \
        context-compressor \
        2>/dev/null || true

    success "Intelligence services stopped"
}

# Validate environment for intelligence services
validate_intelligence_environment() {
    info "Validating intelligence environment..."

    # Check available memory
    local available_memory=$(free -m | awk '/^Mem:/{print $7}')
    if [[ "${available_memory}" -lt 8192 ]]; then
        warning "Available memory is ${available_memory}MB. Intelligence services require at least 8GB for optimal performance."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error_exit "Insufficient memory for intelligence services"
        fi
    fi

    # Check disk space for models
    local available_space=$(df "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')
    if [[ "${available_space}" -lt 10485760 ]]; then  # 10GB in KB
        warning "Available disk space is limited. ML models require significant storage."
    fi

    # Check if GPU is available
    if command -v nvidia-smi >/dev/null 2>&1; then
        info "NVIDIA GPU detected. Intelligence services will use GPU acceleration."
    else
        info "No GPU detected. Intelligence services will use CPU-only mode."
    fi

    success "Environment validation completed"
}

# Pre-download ML models
predownload_models() {
    info "Pre-downloading ML models..."

    local model_dir="${PROJECT_ROOT}/models"
    mkdir -p "${model_dir}"

    # Download sentence transformer model
    info "Downloading sentence transformer model..."
    if ! docker run --rm \
        -v "${model_dir}:/models" \
        python:3.11-slim \
        bash -c "
            pip install sentence-transformers &&
            python -c \"
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('/models/all-MiniLM-L6-v2')
            \"
        " >/dev/null 2>&1; then
        warning "Failed to pre-download models. Models will be downloaded during startup."
    else
        success "ML models pre-downloaded"
    fi
}

# Deploy intelligence services
deploy_intelligence() {
    info "Starting intelligence enhancement deployment..."

    # Phase 1: Context Compression (if not already running)
    if ! docker ps --format "table {{.Names}}" | grep -q "^bev_context_compressor$"; then
        info "Phase 1: Deploying context compression..."
        docker-compose -f "${INFRASTRUCTURE_COMPOSE}" up -d context-compressor \
            || error_exit "Failed to deploy context compressor"

        wait_for_service_health "context-compressor" "http://172.30.0.43:8080/health"
        success "Phase 1 completed: Context compression deployed"
    else
        info "Phase 1: Context compressor already running"
    fi

    # Phase 2: Extended Reasoning
    info "Phase 2: Deploying extended reasoning..."
    docker-compose -f "${MAIN_COMPOSE}" up -d extended-reasoning \
        || error_exit "Failed to deploy extended reasoning"

    wait_for_service_health "extended-reasoning" "http://172.30.0.46:8080/health"
    success "Phase 2 completed: Extended reasoning deployed"

    # Phase 3: Predictive Cache
    info "Phase 3: Deploying predictive cache..."
    docker-compose -f "${MAIN_COMPOSE}" up -d predictive-cache \
        || error_exit "Failed to deploy predictive cache"

    wait_for_service_health "predictive-cache" "http://172.30.0.44:8080/health"
    success "Phase 3 completed: Predictive cache deployed"

    success "Intelligence enhancement deployment completed"
}

# Wait for service health (extended timeout for ML services)
wait_for_service_health() {
    local service_name="$1"
    local health_url="$2"
    local max_attempts="${HEALTH_CHECK_RETRIES}"
    local attempt=1

    info "Waiting for ${service_name} to become healthy (this may take several minutes for ML models)..."

    while [[ $attempt -le $max_attempts ]]; do
        if curl -sf "${health_url}" >/dev/null 2>&1; then
            success "${service_name} is healthy"
            return 0
        fi

        # Show progress for ML services
        if [[ "${service_name}" =~ (extended-reasoning|predictive-cache) ]]; then
            info "Attempt ${attempt}/${max_attempts}: ${service_name} loading ML models, waiting ${HEALTH_CHECK_INTERVAL}s..."
        else
            info "Attempt ${attempt}/${max_attempts}: ${service_name} not ready yet, waiting ${HEALTH_CHECK_INTERVAL}s..."
        fi

        sleep "${HEALTH_CHECK_INTERVAL}"
        ((attempt++))
    done

    warning "${service_name} failed to become healthy after ${max_attempts} attempts"
    return 1
}

# Verify intelligence deployment
verify_intelligence() {
    info "Verifying intelligence enhancement deployment..."

    local services=(
        "extended-reasoning:http://172.30.0.46:8080/health"
        "predictive-cache:http://172.30.0.44:8080/health"
    )

    # Check context compressor if it exists
    if docker ps --format "table {{.Names}}" | grep -q "^bev_context_compressor$"; then
        services+=("context-compressor:http://172.30.0.43:8080/health")
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
        warning "Some intelligence services are not healthy: ${failed_services[*]}"
        return 1
    fi

    success "All intelligence services are healthy"
    return 0
}

# Test intelligence functionality
test_intelligence() {
    info "Testing intelligence enhancement functionality..."

    # Test extended reasoning
    info "Testing extended reasoning..."
    local reasoning_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"query": "test reasoning", "context": "testing"}' \
        "http://172.30.0.46:8080/api/reason" 2>/dev/null || echo "")

    if [[ -n "${reasoning_response}" ]]; then
        success "Extended reasoning is responding"
    else
        warning "Extended reasoning is not responding properly"
    fi

    # Test predictive cache
    info "Testing predictive cache..."
    local cache_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"query": "test cache"}' \
        "http://172.30.0.44:8080/api/predict" 2>/dev/null || echo "")

    if [[ -n "${cache_response}" ]]; then
        success "Predictive cache is responding"
    else
        warning "Predictive cache is not responding properly"
    fi

    # Test context compression (if available)
    if docker ps --format "table {{.Names}}" | grep -q "^bev_context_compressor$"; then
        info "Testing context compression..."
        local compression_response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d '{"text": "This is a test text for compression"}' \
            "http://172.30.0.43:8080/api/compress" 2>/dev/null || echo "")

        if [[ -n "${compression_response}" ]]; then
            success "Context compression is responding"
        else
            warning "Context compression is not responding properly"
        fi
    fi

    success "Intelligence functionality test completed"
}

# Configure intelligence services
configure_intelligence() {
    info "Configuring intelligence services..."

    # Configure predictive cache learning
    info "Initializing predictive cache learning..."
    curl -s -X POST "http://172.30.0.44:8080/api/learning/start" >/dev/null 2>&1 || true

    # Configure extended reasoning models
    info "Loading extended reasoning models..."
    curl -s -X POST "http://172.30.0.46:8080/api/models/load" >/dev/null 2>&1 || true

    success "Intelligence services configured"
}

# Display intelligence status
show_intelligence_status() {
    info "Intelligence Enhancement Status:"
    echo

    # Show container status
    echo "Container Status:"
    docker ps --filter "label=bev.category=intelligence" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo

    # Show service endpoints
    info "Intelligence Endpoints:"
    echo "  Extended Reasoning:  http://172.30.0.46:8046"
    echo "  Predictive Cache:    http://172.30.0.44:8044"

    if docker ps --format "table {{.Names}}" | grep -q "^bev_context_compressor$"; then
        echo "  Context Compressor:  http://172.30.0.43:8043"
    fi
    echo

    # Show intelligence metrics
    info "Intelligence Metrics:"

    # Extended reasoning stats
    local reasoning_stats=$(curl -s "http://172.30.0.46:8080/api/stats" 2>/dev/null | jq -r '.' 2>/dev/null || echo "{}")
    if [[ "${reasoning_stats}" != "{}" ]]; then
        echo "  Extended Reasoning:"
        echo "    Processed Queries: $(echo "${reasoning_stats}" | jq -r '.processed_queries // "N/A"')"
        echo "    Average Response:  $(echo "${reasoning_stats}" | jq -r '.avg_response_time // "N/A"')ms"
        echo "    Model Status:      $(echo "${reasoning_stats}" | jq -r '.model_status // "N/A"')"
    fi

    # Predictive cache stats
    local cache_stats=$(curl -s "http://172.30.0.44:8080/api/stats" 2>/dev/null | jq -r '.' 2>/dev/null || echo "{}")
    if [[ "${cache_stats}" != "{}" ]]; then
        echo "  Predictive Cache:"
        echo "    Cache Hit Rate:    $(echo "${cache_stats}" | jq -r '.hit_rate // "N/A"')%"
        echo "    Predictions Made:  $(echo "${cache_stats}" | jq -r '.predictions_made // "N/A"')"
        echo "    Learning Status:   $(echo "${cache_stats}" | jq -r '.learning_status // "N/A"')"
    fi

    # Context compression stats (if available)
    if docker ps --format "table {{.Names}}" | grep -q "^bev_context_compressor$"; then
        local compression_stats=$(curl -s "http://172.30.0.43:8080/api/stats" 2>/dev/null | jq -r '.' 2>/dev/null || echo "{}")
        if [[ "${compression_stats}" != "{}" ]]; then
            echo "  Context Compression:"
            echo "    Compression Ratio: $(echo "${compression_stats}" | jq -r '.avg_compression_ratio // "N/A"')"
            echo "    Processed Texts:   $(echo "${compression_stats}" | jq -r '.processed_texts // "N/A"')"
        fi
    fi
    echo
}

# Monitor intelligence performance
monitor_performance() {
    info "Monitoring intelligence performance for 60 seconds..."

    local start_time=$(date +%s)
    local end_time=$((start_time + 60))

    while [[ $(date +%s) -lt $end_time ]]; do
        # Check service health
        local healthy_services=0
        local total_services=0

        for service in extended-reasoning predictive-cache; do
            ((total_services++))
            if curl -sf "http://172.30.0.46:8080/health" >/dev/null 2>&1 || curl -sf "http://172.30.0.44:8080/health" >/dev/null 2>&1; then
                ((healthy_services++))
            fi
        done

        local remaining=$((end_time - $(date +%s)))
        echo -ne "\rMonitoring: ${healthy_services}/${total_services} services healthy, ${remaining}s remaining..."
        sleep 5
    done

    echo
    success "Performance monitoring completed"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        warning "Intelligence deployment failed. Check logs at: ${LOG_FILE}"
        info "To debug, run: docker logs <service_name>"
    fi
    exit $exit_code
}

# Main deployment function
main() {
    # Set up cleanup trap
    trap cleanup EXIT

    info "Starting BEV OSINT Intelligence Enhancement Deployment"
    info "Log file: ${LOG_FILE}"
    echo

    # Run deployment steps
    check_prerequisites
    validate_intelligence_environment

    if check_existing_intelligence; then
        predownload_models
        deploy_intelligence
        configure_intelligence

        if verify_intelligence; then
            test_intelligence
            show_intelligence_status
            success "Intelligence enhancement deployment completed successfully!"
        else
            warning "Deployment completed but some services may not be fully ready"
            warning "Run './deploy_intelligence.sh verify' to check status again"
        fi
    else
        info "Intelligence deployment skipped"
    fi
}

# Command line interface
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "verify")
        check_prerequisites
        verify_intelligence && show_intelligence_status
        ;;
    "test")
        check_prerequisites
        test_intelligence
        ;;
    "status")
        check_prerequisites
        show_intelligence_status
        ;;
    "monitor")
        check_prerequisites
        monitor_performance
        ;;
    "stop")
        info "Stopping intelligence services..."
        stop_intelligence_services
        success "Intelligence services stopped"
        ;;
    "restart")
        info "Restarting intelligence services..."
        stop_intelligence_services
        sleep 10
        deploy_intelligence
        success "Intelligence services restarted"
        ;;
    "logs")
        echo "Intelligence service logs:"
        echo "=========================="
        docker-compose -f "${MAIN_COMPOSE}" logs --tail=50 extended-reasoning predictive-cache 2>/dev/null || true
        docker-compose -f "${INFRASTRUCTURE_COMPOSE}" logs --tail=50 context-compressor 2>/dev/null || true
        ;;
    *)
        echo "Usage: $0 {deploy|verify|test|status|monitor|stop|restart|logs}"
        echo
        echo "Commands:"
        echo "  deploy   - Deploy intelligence services (default)"
        echo "  verify   - Verify deployment and show status"
        echo "  test     - Test intelligence functionality"
        echo "  status   - Show current status of intelligence services"
        echo "  monitor  - Monitor performance for 60 seconds"
        echo "  stop     - Stop all intelligence services"
        echo "  restart  - Restart all intelligence services"
        echo "  logs     - Show logs from intelligence services"
        exit 1
        ;;
esac