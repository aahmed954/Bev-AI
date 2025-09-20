#!/bin/bash

# BEV OSINT Edge Computing Deployment Script
# This script deploys the edge computing services:
# - Edge Nodes (US East, US West, EU Central, Asia Pacific)
# - Edge Management
# - Model Synchronizer
# - Geographic Router

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MAIN_COMPOSE="${PROJECT_ROOT}/docker-compose.complete.yml"
LOG_FILE="${PROJECT_ROOT}/logs/edge_deployment.log"
TIMEOUT=1200  # 20 minutes for edge nodes
HEALTH_CHECK_RETRIES=60
HEALTH_CHECK_INTERVAL=20

# Edge node configuration
declare -A EDGE_NODES=(
    ["us-east"]="172.30.0.47:8047"
    ["us-west"]="172.30.0.48:8048"
    ["eu-central"]="172.30.0.49:8049"
    ["asia-pacific"]="172.30.0.50:8050"
)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

# Highlight message
highlight() {
    echo -e "${CYAN}$1${NC}"
    log "HIGHLIGHT: $1"
}

# Check prerequisites
check_prerequisites() {
    info "Checking edge computing prerequisites..."

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error_exit "Docker is not running or not accessible"
    fi

    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        error_exit "docker-compose is not installed"
    fi

    # Check if compose file exists
    if [[ ! -f "${MAIN_COMPOSE}" ]]; then
        error_exit "Main compose file not found: ${MAIN_COMPOSE}"
    fi

    # Check if required infrastructure is running
    local required_services=(
        "bev_postgres"
        "bev_redis_1"
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

    # Create logs directory
    mkdir -p "${PROJECT_ROOT}/logs"

    success "Prerequisites check completed"
}

# Check system resources for edge computing
check_system_resources() {
    info "Checking system resources for edge computing..."

    # Check available memory
    local available_memory=$(free -m | awk '/^Mem:/{print $7}')
    local required_memory=16384  # 16GB

    if [[ "${available_memory}" -lt "${required_memory}" ]]; then
        warning "Available memory is ${available_memory}MB. Edge computing requires at least ${required_memory}MB for optimal performance."
        read -p "Continue with reduced edge nodes? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error_exit "Insufficient memory for full edge deployment"
        fi
    fi

    # Check disk space for edge models
    local available_space=$(df "${PROJECT_ROOT}" | awk 'NR==2 {print $4}')
    local required_space=20971520  # 20GB in KB

    if [[ "${available_space}" -lt "${required_space}" ]]; then
        warning "Available disk space is ${available_space}KB. Edge models require at least 20GB."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error_exit "Insufficient disk space for edge deployment"
        fi
    fi

    # Check CPU cores
    local cpu_cores=$(nproc)
    if [[ "${cpu_cores}" -lt 8 ]]; then
        warning "System has ${cpu_cores} CPU cores. Edge computing performs better with 8+ cores."
    fi

    success "System resources check completed"
}

# Check existing edge services
check_existing_edge() {
    info "Checking existing edge computing services..."

    local running_services=()
    local edge_services=(
        "bev-edge-us-east"
        "bev-edge-us-west"
        "bev-edge-eu-central"
        "bev-edge-asia-pacific"
        "bev-edge-management"
        "bev-model-synchronizer"
        "bev-geo-router"
    )

    for service in "${edge_services[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^${service}$"; then
            running_services+=("${service}")
        fi
    done

    if [[ ${#running_services[@]} -gt 0 ]]; then
        warning "Found running edge services: ${running_services[*]}"
        read -p "Do you want to stop and recreate them? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            info "Stopping existing edge services..."
            stop_edge_services
        else
            info "Skipping deployment - existing services will remain"
            return 1
        fi
    fi

    return 0
}

# Stop edge services
stop_edge_services() {
    info "Stopping edge computing services..."

    docker-compose -f "${MAIN_COMPOSE}" stop \
        edge-node-us-east \
        edge-node-us-west \
        edge-node-eu-central \
        edge-node-asia-pacific \
        edge-management \
        model-synchronizer \
        geo-router \
        2>/dev/null || true

    success "Edge services stopped"
}

# Select edge nodes to deploy
select_edge_nodes() {
    info "Edge Node Selection:"
    echo
    echo "Available edge regions:"
    echo "  1) US East (Virginia)"
    echo "  2) US West (California)"
    echo "  3) EU Central (Frankfurt)"
    echo "  4) Asia Pacific (Singapore)"
    echo "  5) All regions (requires 16GB+ RAM)"
    echo

    read -p "Select regions to deploy (1-5, comma-separated, or 'all'): " -r selection

    local selected_nodes=()

    if [[ "${selection}" == "all" || "${selection}" == "5" ]]; then
        selected_nodes=("us-east" "us-west" "eu-central" "asia-pacific")
    else
        IFS=',' read -ra ADDR <<< "${selection}"
        for region in "${ADDR[@]}"; do
            case "${region// /}" in
                "1") selected_nodes+=("us-east") ;;
                "2") selected_nodes+=("us-west") ;;
                "3") selected_nodes+=("eu-central") ;;
                "4") selected_nodes+=("asia-pacific") ;;
                *) warning "Invalid selection: ${region}" ;;
            esac
        done
    fi

    if [[ ${#selected_nodes[@]} -eq 0 ]]; then
        error_exit "No valid edge nodes selected"
    fi

    info "Selected edge nodes: ${selected_nodes[*]}"
    echo "${selected_nodes[@]}"
}

# Deploy edge nodes in parallel
deploy_edge_nodes() {
    local nodes=("$@")
    info "Deploying edge nodes: ${nodes[*]}"

    # Start edge nodes in parallel
    for node in "${nodes[@]}"; do
        highlight "Starting edge node: ${node}"
        docker-compose -f "${MAIN_COMPOSE}" up -d "edge-node-${node}" &
    done

    # Wait for all background processes to complete
    wait

    # Wait for edge nodes to become healthy
    for node in "${nodes[@]}"; do
        local health_url="http://${EDGE_NODES[${node}]}/health"
        wait_for_service_health "edge-node-${node}" "${health_url}"
    done

    success "Edge nodes deployed successfully: ${nodes[*]}"
}

# Deploy edge management services
deploy_edge_management() {
    info "Deploying edge management services..."

    # Deploy edge management
    highlight "Starting edge management..."
    docker-compose -f "${MAIN_COMPOSE}" up -d edge-management \
        || error_exit "Failed to deploy edge management"

    wait_for_service_health "edge-management" "http://172.30.0.51:8080/health"

    # Deploy model synchronizer
    highlight "Starting model synchronizer..."
    docker-compose -f "${MAIN_COMPOSE}" up -d model-synchronizer \
        || error_exit "Failed to deploy model synchronizer"

    wait_for_service_health "model-synchronizer" "http://172.30.0.46:8080/health"

    success "Edge management services deployed"
}

# Deploy geographic router
deploy_geo_router() {
    info "Deploying geographic router..."

    docker-compose -f "${MAIN_COMPOSE}" up -d geo-router \
        || error_exit "Failed to deploy geographic router"

    wait_for_service_health "geo-router" "http://172.30.0.52:8080/health"

    success "Geographic router deployed"
}

# Wait for service health (extended timeout for edge services)
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

        # Show progress for edge services
        if [[ "${service_name}" =~ edge-node ]]; then
            info "Attempt ${attempt}/${max_attempts}: ${service_name} initializing models, waiting ${HEALTH_CHECK_INTERVAL}s..."
        else
            info "Attempt ${attempt}/${max_attempts}: ${service_name} not ready yet, waiting ${HEALTH_CHECK_INTERVAL}s..."
        fi

        sleep "${HEALTH_CHECK_INTERVAL}"
        ((attempt++))
    done

    warning "${service_name} failed to become healthy after ${max_attempts} attempts"
    return 1
}

# Verify edge deployment
verify_edge_deployment() {
    info "Verifying edge computing deployment..."

    local services=(
        "edge-management:http://172.30.0.51:8080/health"
        "model-synchronizer:http://172.30.0.46:8080/health"
        "geo-router:http://172.30.0.52:8080/health"
    )

    # Add running edge nodes
    for node_name in "${!EDGE_NODES[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^bev-edge-${node_name}$"; then
            services+=("edge-node-${node_name}:http://${EDGE_NODES[${node_name}]}/health")
        fi
    done

    local failed_services=()

    for service_health in "${services[@]}"; do
        local service_name="${service_health%%:*}"
        local health_url="${service_health#*:}"

        if ! curl -sf "${health_url}" >/dev/null 2>&1; then
            failed_services+=("${service_name}")
        fi
    done

    if [[ ${#failed_services[@]} -gt 0 ]]; then
        warning "Some edge services are not healthy: ${failed_services[*]}"
        return 1
    fi

    success "All edge services are healthy"
    return 0
}

# Test edge functionality
test_edge_functionality() {
    info "Testing edge computing functionality..."

    # Test geographic router
    info "Testing geographic router..."
    local router_response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"query": "test routing", "location": "us-east"}' \
        "http://172.30.0.52:8080/api/route" 2>/dev/null || echo "")

    if [[ -n "${router_response}" ]]; then
        success "Geographic router is responding"
    else
        warning "Geographic router is not responding properly"
    fi

    # Test edge nodes
    for node_name in "${!EDGE_NODES[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^bev-edge-${node_name}$"; then
            info "Testing edge node: ${node_name}..."
            local edge_response=$(curl -s -X POST \
                -H "Content-Type: application/json" \
                -d '{"query": "test processing"}' \
                "http://${EDGE_NODES[${node_name}]}/api/process" 2>/dev/null || echo "")

            if [[ -n "${edge_response}" ]]; then
                success "Edge node ${node_name} is responding"
            else
                warning "Edge node ${node_name} is not responding properly"
            fi
        fi
    done

    # Test model synchronization
    info "Testing model synchronization..."
    local sync_response=$(curl -s "http://172.30.0.46:8080/api/sync/status" 2>/dev/null || echo "")
    if [[ -n "${sync_response}" ]]; then
        success "Model synchronizer is responding"
    else
        warning "Model synchronizer is not responding properly"
    fi

    success "Edge functionality test completed"
}

# Configure edge network
configure_edge_network() {
    info "Configuring edge network..."

    # Configure geographic router with edge nodes
    local edge_config="["
    local first=true

    for node_name in "${!EDGE_NODES[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^bev-edge-${node_name}$"; then
            if [[ "${first}" == false ]]; then
                edge_config+=","
            fi
            edge_config+="{\"name\":\"${node_name}\",\"url\":\"http://${EDGE_NODES[${node_name}]}\"}"
            first=false
        fi
    done

    edge_config+="]"

    # Register edge nodes with router
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "${edge_config}" \
        "http://172.30.0.52:8080/api/nodes/register" >/dev/null 2>&1 || true

    success "Edge network configured"
}

# Display edge status
show_edge_status() {
    info "Edge Computing Status:"
    echo

    # Show container status
    echo "Container Status:"
    docker ps --filter "label=bev.component=edge-computing" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo

    # Show edge endpoints
    info "Edge Endpoints:"
    echo "  Edge Management:     http://172.30.0.51:8051"
    echo "  Model Synchronizer:  http://172.30.0.46:8046"
    echo "  Geographic Router:   http://172.30.0.52:8052"
    echo

    # Show active edge nodes
    info "Active Edge Nodes:"
    for node_name in "${!EDGE_NODES[@]}"; do
        if docker ps --format "table {{.Names}}" | grep -q "^bev-edge-${node_name}$"; then
            echo "  ${node_name}: http://${EDGE_NODES[${node_name}]}"
        fi
    done
    echo

    # Show edge metrics
    info "Edge Metrics:"

    # Router stats
    local router_stats=$(curl -s "http://172.30.0.52:8080/api/stats" 2>/dev/null | jq -r '.' 2>/dev/null || echo "{}")
    if [[ "${router_stats}" != "{}" ]]; then
        echo "  Geographic Router:"
        echo "    Routing Requests:  $(echo "${router_stats}" | jq -r '.routing_requests // "N/A"')"
        echo "    Active Nodes:      $(echo "${router_stats}" | jq -r '.active_nodes // "N/A"')"
        echo "    Average Latency:   $(echo "${router_stats}" | jq -r '.avg_latency // "N/A"')ms"
    fi

    # Model sync stats
    local sync_stats=$(curl -s "http://172.30.0.46:8080/api/stats" 2>/dev/null | jq -r '.' 2>/dev/null || echo "{}")
    if [[ "${sync_stats}" != "{}" ]]; then
        echo "  Model Synchronization:"
        echo "    Models Synced:     $(echo "${sync_stats}" | jq -r '.models_synced // "N/A"')"
        echo "    Last Sync:         $(echo "${sync_stats}" | jq -r '.last_sync // "N/A"')"
        echo "    Sync Status:       $(echo "${sync_stats}" | jq -r '.sync_status // "N/A"')"
    fi
    echo
}

# Monitor edge performance
monitor_edge_performance() {
    info "Monitoring edge performance for 120 seconds..."

    local start_time=$(date +%s)
    local end_time=$((start_time + 120))

    while [[ $(date +%s) -lt $end_time ]]; do
        # Check service health
        local healthy_services=0
        local total_services=0

        # Check management services
        for service in edge-management model-synchronizer geo-router; do
            ((total_services++))
            case "${service}" in
                "edge-management")
                    if curl -sf "http://172.30.0.51:8080/health" >/dev/null 2>&1; then
                        ((healthy_services++))
                    fi
                    ;;
                "model-synchronizer")
                    if curl -sf "http://172.30.0.46:8080/health" >/dev/null 2>&1; then
                        ((healthy_services++))
                    fi
                    ;;
                "geo-router")
                    if curl -sf "http://172.30.0.52:8080/health" >/dev/null 2>&1; then
                        ((healthy_services++))
                    fi
                    ;;
            esac
        done

        # Check edge nodes
        for node_name in "${!EDGE_NODES[@]}"; do
            if docker ps --format "table {{.Names}}" | grep -q "^bev-edge-${node_name}$"; then
                ((total_services++))
                if curl -sf "http://${EDGE_NODES[${node_name}]}/health" >/dev/null 2>&1; then
                    ((healthy_services++))
                fi
            fi
        done

        local remaining=$((end_time - $(date +%s)))
        echo -ne "\rMonitoring: ${healthy_services}/${total_services} services healthy, ${remaining}s remaining..."
        sleep 10
    done

    echo
    success "Performance monitoring completed"
}

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        warning "Edge deployment failed. Check logs at: ${LOG_FILE}"
        info "To debug, run: docker logs <service_name>"
    fi
    exit $exit_code
}

# Main deployment function
main() {
    # Set up cleanup trap
    trap cleanup EXIT

    info "Starting BEV OSINT Edge Computing Deployment"
    info "Log file: ${LOG_FILE}"
    echo

    # Run deployment steps
    check_prerequisites
    check_system_resources

    if check_existing_edge; then
        # Select edge nodes to deploy
        local selected_nodes
        IFS=' ' read -ra selected_nodes <<< "$(select_edge_nodes)"

        # Deploy edge services in phases
        deploy_edge_nodes "${selected_nodes[@]}"
        deploy_edge_management
        deploy_geo_router

        # Configure and verify
        configure_edge_network

        if verify_edge_deployment; then
            test_edge_functionality
            show_edge_status
            success "Edge computing deployment completed successfully!"
            highlight "Edge nodes are now processing requests in geographic regions: ${selected_nodes[*]}"
        else
            warning "Deployment completed but some services may not be fully ready"
            warning "Run './deploy_edge.sh verify' to check status again"
        fi
    else
        info "Edge deployment skipped"
    fi
}

# Command line interface
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "verify")
        check_prerequisites
        verify_edge_deployment && show_edge_status
        ;;
    "test")
        check_prerequisites
        test_edge_functionality
        ;;
    "status")
        check_prerequisites
        show_edge_status
        ;;
    "monitor")
        check_prerequisites
        monitor_edge_performance
        ;;
    "stop")
        info "Stopping edge services..."
        stop_edge_services
        success "Edge services stopped"
        ;;
    "restart")
        info "Restarting edge services..."
        stop_edge_services
        sleep 15
        # Restart with all available nodes
        local all_nodes=("us-east" "us-west" "eu-central" "asia-pacific")
        deploy_edge_nodes "${all_nodes[@]}"
        deploy_edge_management
        deploy_geo_router
        success "Edge services restarted"
        ;;
    "logs")
        echo "Edge service logs:"
        echo "=================="
        docker-compose -f "${MAIN_COMPOSE}" logs --tail=50 \
            edge-node-us-east edge-node-us-west edge-node-eu-central edge-node-asia-pacific \
            edge-management model-synchronizer geo-router \
            2>/dev/null || true
        ;;
    *)
        echo "Usage: $0 {deploy|verify|test|status|monitor|stop|restart|logs}"
        echo
        echo "Commands:"
        echo "  deploy   - Deploy edge computing services (default)"
        echo "  verify   - Verify deployment and show status"
        echo "  test     - Test edge functionality"
        echo "  status   - Show current status of edge services"
        echo "  monitor  - Monitor performance for 120 seconds"
        echo "  stop     - Stop all edge services"
        echo "  restart  - Restart all edge services"
        echo "  logs     - Show logs from edge services"
        exit 1
        ;;
esac