#!/bin/bash

# ===================================================================
# BEV OSINT Framework - Phase 7 Deployment Script
# Phase: Alternative Market Intelligence
# Services: dm-crawler, crypto-intel, reputation-analyzer, economics-processor
# ===================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHASE="7"
PHASE_NAME="Alternative Market Intelligence"

# Service definitions
SERVICES=(
    "dm-crawler:8001:172.30.0.24"
    "crypto-intel:8002:172.30.0.25"
    "reputation-analyzer:8003:172.30.0.26"
    "economics-processor:8004:172.30.0.27"
)

# Resource requirements
declare -A SERVICE_MEMORY=(
    [dm-crawler]="2G"
    [crypto-intel]="3G"
    [reputation-analyzer]="4G"
    [economics-processor]="6G"
)

declare -A SERVICE_CPU=(
    [dm-crawler]="1.0"
    [crypto-intel]="1.5"
    [reputation-analyzer]="2.0"
    [economics-processor]="2.0"
)

# GPU requirements
GPU_SERVICES=("economics-processor")

# Logging setup
LOG_DIR="${PROJECT_ROOT}/logs/deployment"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/phase_${PHASE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ===================================================================
# Utility Functions
# ===================================================================

log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        INFO)  echo -e "${timestamp} ${BLUE}[INFO]${NC} $message" ;;
        WARN)  echo -e "${timestamp} ${YELLOW}[WARN]${NC} $message" ;;
        ERROR) echo -e "${timestamp} ${RED}[ERROR]${NC} $message" ;;
        SUCCESS) echo -e "${timestamp} ${GREEN}[SUCCESS]${NC} $message" ;;
    esac
}

check_service_prerequisites() {
    local service=$1
    log INFO "Checking prerequisites for $service..."

    # Check if service directory exists
    local service_dir="${PROJECT_ROOT}/phase${PHASE}/${service}"
    if [[ ! -d "$service_dir" ]]; then
        log ERROR "Service directory not found: $service_dir"
        return 1
    fi

    # Check if Dockerfile exists
    if [[ ! -f "${service_dir}/Dockerfile" ]]; then
        log ERROR "Dockerfile not found for $service"
        return 1
    fi

    # Check if config directory exists
    if [[ ! -d "${service_dir}/config" ]]; then
        log WARN "Config directory not found for $service, creating..."
        mkdir -p "${service_dir}/config"
    fi

    # GPU-specific checks
    if [[ " ${GPU_SERVICES[*]} " =~ " ${service} " ]]; then
        if ! docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
            log ERROR "GPU support required for $service but not available"
            return 1
        fi
        log INFO "GPU support verified for $service"
    fi

    log SUCCESS "Prerequisites check passed for $service"
    return 0
}

build_service() {
    local service=$1
    log INFO "Building $service..."

    local service_dir="${PROJECT_ROOT}/phase${PHASE}/${service}"
    local build_start_time=$(date +%s)

    # Build with no cache if forced
    local build_args=""
    if [[ "${FORCE_REBUILD:-false}" == "true" ]]; then
        build_args="--no-cache"
    fi

    # Build the service
    if ! docker build $build_args -t "bev_${service}:latest" "$service_dir"; then
        log ERROR "Failed to build $service"
        return 1
    fi

    local build_end_time=$(date +%s)
    local build_duration=$((build_end_time - build_start_time))

    log SUCCESS "$service built successfully in ${build_duration}s"
    return 0
}

create_service_volumes() {
    log INFO "Creating Phase $PHASE volumes..."

    local volumes=(
        "dm_crawler_data"
        "crypto_intel_data"
        "reputation_data"
        "economics_data"
        "ml_models"
    )

    for volume in "${volumes[@]}"; do
        if ! docker volume ls | grep -q "$volume"; then
            docker volume create "$volume"
            log INFO "Created volume: $volume"
        else
            log INFO "Volume already exists: $volume"
        fi
    done
}

check_network() {
    log INFO "Checking BEV OSINT network..."

    if ! docker network ls | grep -q "bev_osint"; then
        log INFO "Creating BEV OSINT network..."
        docker network create \
            --driver bridge \
            --subnet=172.30.0.0/16 \
            --gateway=172.30.0.1 \
            bev_osint
        log SUCCESS "BEV OSINT network created"
    else
        log INFO "BEV OSINT network already exists"
    fi
}

check_dependencies() {
    log INFO "Checking Phase $PHASE dependencies..."

    local required_services=(
        "postgres"
        "neo4j"
        "elasticsearch"
        "kafka-1"
        "redis"
        "influxdb"
        "tor"
    )

    local missing_services=()

    for service in "${required_services[@]}"; do
        if ! docker ps --format '{{.Names}}' | grep -q "^${service}$"; then
            missing_services+=("$service")
        fi
    done

    if [[ ${#missing_services[@]} -gt 0 ]]; then
        log WARN "Missing dependencies: ${missing_services[*]}"
        log INFO "Attempting to start dependencies..."

        # Start core infrastructure if needed
        if [[ -f "${PROJECT_ROOT}/docker-compose.complete.yml" ]]; then
            docker-compose -f "${PROJECT_ROOT}/docker-compose.complete.yml" up -d \
                postgres neo4j elasticsearch kafka-1 kafka-2 kafka-3 redis influxdb tor
            sleep 30  # Wait for services to initialize
        else
            log ERROR "Core infrastructure compose file not found"
            return 1
        fi
    fi

    log SUCCESS "All dependencies are available"
    return 0
}

deploy_services() {
    log INFO "Deploying Phase $PHASE services..."

    # Change to project root for docker-compose
    cd "$PROJECT_ROOT"

    # Deploy using docker-compose
    if ! docker-compose -f "docker-compose-phase${PHASE}.yml" up -d; then
        log ERROR "Failed to deploy Phase $PHASE services"
        return 1
    fi

    log SUCCESS "Phase $PHASE services deployment initiated"

    # Wait for services to be healthy
    wait_for_services
}

wait_for_services() {
    log INFO "Waiting for services to become healthy..."

    local max_wait=300  # 5 minutes
    local check_interval=10
    local elapsed=0

    while [[ $elapsed -lt $max_wait ]]; do
        local all_healthy=true

        for service_info in "${SERVICES[@]}"; do
            local service=$(echo "$service_info" | cut -d':' -f1)
            local port=$(echo "$service_info" | cut -d':' -f2)

            if ! docker ps --format '{{.Names}}' | grep -q "bev_${service}"; then
                log WARN "Service not running: $service"
                all_healthy=false
                continue
            fi

            # Check health endpoint
            if ! curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
                log WARN "Service not healthy: $service (port $port)"
                all_healthy=false
            else
                log INFO "Service healthy: $service"
            fi
        done

        if [[ "$all_healthy" == "true" ]]; then
            log SUCCESS "All Phase $PHASE services are healthy"
            return 0
        fi

        log INFO "Waiting for services to become healthy... (${elapsed}s elapsed)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    log ERROR "Timeout waiting for services to become healthy"
    return 1
}

check_resource_usage() {
    log INFO "Checking resource usage for Phase $PHASE services..."

    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        local container_name="bev_${service}"

        if docker ps --format '{{.Names}}' | grep -q "$container_name"; then
            local stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}" "$container_name")
            log INFO "Resource usage for $service:"
            echo "$stats"
        fi
    done
}

verify_functionality() {
    log INFO "Verifying Phase $PHASE functionality..."

    # Test service endpoints
    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        local port=$(echo "$service_info" | cut -d':' -f2)

        log INFO "Testing $service endpoint..."

        # Test health endpoint
        if curl -sf "http://localhost:${port}/health" >/dev/null; then
            log SUCCESS "$service health endpoint responding"
        else
            log ERROR "$service health endpoint not responding"
            return 1
        fi

        # Test API endpoint if available
        if curl -sf "http://localhost:${port}/api/v1/status" >/dev/null; then
            log SUCCESS "$service API endpoint responding"
        else
            log WARN "$service API endpoint not responding (may be expected)"
        fi
    done

    # Phase-specific functionality tests
    test_dm_crawler
    test_crypto_intel
    test_reputation_analyzer
    test_economics_processor

    log SUCCESS "Phase $PHASE functionality verification completed"
}

test_dm_crawler() {
    log INFO "Testing DM Crawler functionality..."

    local response=$(curl -sf "http://localhost:8001/api/v1/crawl/status" 2>/dev/null || echo "ERROR")
    if [[ "$response" != "ERROR" ]]; then
        log SUCCESS "DM Crawler API responding"
    else
        log WARN "DM Crawler API not accessible"
    fi
}

test_crypto_intel() {
    log INFO "Testing Crypto Intel functionality..."

    local response=$(curl -sf "http://localhost:8002/api/v1/blockchain/status" 2>/dev/null || echo "ERROR")
    if [[ "$response" != "ERROR" ]]; then
        log SUCCESS "Crypto Intel API responding"
    else
        log WARN "Crypto Intel API not accessible"
    fi
}

test_reputation_analyzer() {
    log INFO "Testing Reputation Analyzer functionality..."

    local response=$(curl -sf "http://localhost:8003/api/v1/reputation/status" 2>/dev/null || echo "ERROR")
    if [[ "$response" != "ERROR" ]]; then
        log SUCCESS "Reputation Analyzer API responding"
    else
        log WARN "Reputation Analyzer API not accessible"
    fi
}

test_economics_processor() {
    log INFO "Testing Economics Processor functionality..."

    local response=$(curl -sf "http://localhost:8004/api/v1/economics/status" 2>/dev/null || echo "ERROR")
    if [[ "$response" != "ERROR" ]]; then
        log SUCCESS "Economics Processor API responding"
    else
        log WARN "Economics Processor API not accessible"
    fi
}

show_deployment_summary() {
    log INFO "Phase $PHASE Deployment Summary:"
    echo "=============================================="
    echo "Phase: $PHASE - $PHASE_NAME"
    echo "Services deployed: ${#SERVICES[@]}"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Service Endpoints:"
    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        local port=$(echo "$service_info" | cut -d':' -f2)
        echo "  $service: http://localhost:$port"
    done
    echo ""
    echo "Next steps:"
    echo "  1. Monitor service logs: docker-compose -f docker-compose-phase${PHASE}.yml logs -f"
    echo "  2. Check service status: docker ps --filter name=bev_"
    echo "  3. Run integration tests: python3 ${PROJECT_ROOT}/deployment/tests/test_phase_${PHASE}.py"
    echo "=============================================="
}

cleanup_on_failure() {
    log ERROR "Phase $PHASE deployment failed, cleaning up..."

    # Stop and remove services
    docker-compose -f "docker-compose-phase${PHASE}.yml" down 2>/dev/null || true

    # Remove built images
    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        docker rmi "bev_${service}:latest" 2>/dev/null || true
    done
}

# ===================================================================
# Main Execution
# ===================================================================

main() {
    log INFO "Starting Phase $PHASE deployment: $PHASE_NAME"

    # Set cleanup trap
    trap cleanup_on_failure ERR

    # Pre-deployment checks
    check_network
    create_service_volumes
    check_dependencies

    # Build services
    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        check_service_prerequisites "$service"
        build_service "$service"
    done

    # Deploy services
    deploy_services

    # Verify deployment
    check_resource_usage
    verify_functionality

    # Show summary
    show_deployment_summary

    log SUCCESS "Phase $PHASE deployment completed successfully!"
}

# Execute main function
main "$@"