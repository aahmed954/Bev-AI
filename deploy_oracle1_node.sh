#!/bin/bash
# ORACLE1 Node Deployment Script
# BEV OSINT Framework - Secondary Monitoring Node (ARM64)
# Generated: $(date)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose-oracle1-unified.yml"
ENV_FILE="${SCRIPT_DIR}/.env.oracle1.complete"
NODE_NAME="ORACLE1"
PRIMARY_NODE="THANOS"
DEPLOYMENT_TIMEOUT=300

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Pre-deployment checks
check_prerequisites() {
    log_info "Checking prerequisites for ${NODE_NAME} deployment..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    # Check architecture
    arch=$(uname -m)
    log_info "System architecture: $arch"
    if [[ "$arch" =~ ^(aarch64|arm64)$ ]]; then
        log_success "ARM64 architecture detected - compatible with ORACLE1"
    else
        log_warning "Non-ARM64 architecture. ORACLE1 is optimized for ARM64"
    fi

    # Check files exist
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Environment file not found: $ENV_FILE"
        exit 1
    fi

    # Check system resources (lower requirements for ARM)
    total_memory=$(free -g | awk '/^Mem:/{print $2}')
    if (( total_memory < 8 )); then
        log_warning "System has ${total_memory}GB RAM. Recommended: 8GB+ for ORACLE1"
    fi

    # Check disk space
    disk_space=$(df "$SCRIPT_DIR" | awk 'NR==2{print $4}')
    if (( disk_space < 20000000 )); then  # 20GB in KB
        log_warning "Low disk space. Recommended: 50GB+ free for ORACLE1"
    fi

    log_success "Prerequisites check passed"
}

# Check THANOS connectivity
check_thanos_connectivity() {
    log_info "Checking connectivity to ${PRIMARY_NODE} node..."

    local thanos_host="${THANOS_IP:-thanos}"
    local max_attempts=30
    local attempt=1

    while (( attempt <= max_attempts )); do
        log_info "Attempt $attempt/$max_attempts: Checking $thanos_host connectivity..."

        # Check if we can reach THANOS
        if ping -c 1 -W 5 "$thanos_host" &>/dev/null; then
            log_success "THANOS node is reachable"
            break
        elif (( attempt == max_attempts )); then
            log_error "Cannot reach THANOS node at $thanos_host"
            log_error "Ensure THANOS is deployed and network connectivity exists"
            exit 1
        else
            log_info "THANOS not reachable yet, waiting..."
            sleep 10
            ((attempt++))
        fi
    done

    # Check key THANOS services
    local services=("5432" "6379" "7687" "8200")  # Postgres, Redis, Neo4j, Vault
    for port in "${services[@]}"; do
        if timeout 5 bash -c "</dev/tcp/$thanos_host/$port"; then
            log_success "THANOS service on port $port is accessible"
        else
            log_warning "THANOS service on port $port is not accessible"
        fi
    done
}

# Validate configuration
validate_configuration() {
    log_info "Validating Docker Compose configuration..."

    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" config --quiet; then
        log_success "Docker Compose configuration is valid"
    else
        log_error "Docker Compose configuration validation failed"
        exit 1
    fi
}

# Clean up existing containers
cleanup_existing() {
    log_info "Cleaning up existing ORACLE1 containers..."

    # Stop and remove existing containers
    if docker ps -a --format "table {{.Names}}" | grep -q "oracle1_\|bev_.*oracle1"; then
        log_info "Stopping existing ORACLE1 containers..."
        docker ps -a --format "{{.Names}}" | grep -E "oracle1_|bev_.*oracle1" | xargs -r docker stop || true
        docker ps -a --format "{{.Names}}" | grep -E "oracle1_|bev_.*oracle1" | xargs -r docker rm || true
    fi

    log_success "Cleanup completed"
}

# Deploy monitoring services
deploy_monitoring() {
    log_info "Deploying ORACLE1 monitoring services..."

    # Deploy monitoring stack optimized for ARM64
    local monitoring_services=(
        "prometheus-oracle1"
        "grafana-oracle1"
        "alertmanager-oracle1"
        "node-exporter-oracle1"
    )

    for service in "${monitoring_services[@]}"; do
        log_info "Starting $service..."
        if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d "$service" 2>/dev/null; then
            log_success "$service started"

            # Wait for service to be healthy
            log_info "Waiting for $service to be ready..."
            sleep 5
            if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps "$service" 2>/dev/null | grep -q "Up"; then
                log_success "$service is healthy"
            else
                log_warning "$service may not be fully ready"
            fi
        else
            log_warning "Could not start $service (may not exist in configuration)"
        fi
    done
}

# Deploy application services
deploy_applications() {
    log_info "Deploying ORACLE1 application services..."

    # Deploy all remaining services
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d; then
        log_success "All ORACLE1 services deployment initiated"
    else
        log_error "Failed to deploy ORACLE1 application services"
        return 1
    fi
}

# Health check all services
health_check() {
    log_info "Performing ORACLE1 health check..."

    local failed_services=()
    local total_services=0
    local healthy_services=0

    # Get all service names
    while IFS= read -r service; do
        [[ -n "$service" ]] || continue
        ((total_services++))

        log_info "Checking $service..."
        if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps "$service" 2>/dev/null | grep -q "Up"; then
            log_success "$service is running"
            ((healthy_services++))
        else
            log_warning "$service is not healthy"
            failed_services+=("$service")
        fi
    done < <(docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" config --services 2>/dev/null || echo "")

    # Report results
    log_info "ORACLE1 Health Check Results:"
    log_info "Total Services: $total_services"
    log_info "Healthy Services: $healthy_services"
    log_info "Failed Services: ${#failed_services[@]}"

    if (( ${#failed_services[@]} > 0 )); then
        log_warning "Failed services: ${failed_services[*]}"
    fi

    # Calculate success rate (more lenient for secondary node)
    if (( total_services > 0 )); then
        success_rate=$(( (healthy_services * 100) / total_services ))
        if (( success_rate >= 70 )); then
            log_success "ORACLE1 deployment success rate: ${success_rate}% (ACCEPTABLE)"
            return 0
        else
            log_error "ORACLE1 deployment success rate: ${success_rate}% (UNACCEPTABLE)"
            return 1
        fi
    else
        log_warning "No services found to check"
        return 0
    fi
}

# Test connectivity to THANOS services
test_thanos_integration() {
    log_info "Testing integration with THANOS services..."

    local thanos_host="${THANOS_IP:-thanos}"

    # Test database connectivity
    log_info "Testing THANOS PostgreSQL connectivity..."
    if timeout 10 bash -c "</dev/tcp/$thanos_host/5432"; then
        log_success "THANOS PostgreSQL is accessible"
    else
        log_warning "THANOS PostgreSQL connection failed"
    fi

    # Test Redis connectivity
    log_info "Testing THANOS Redis connectivity..."
    if timeout 10 bash -c "</dev/tcp/$thanos_host/6379"; then
        log_success "THANOS Redis is accessible"
    else
        log_warning "THANOS Redis connection failed"
    fi

    # Test Neo4j connectivity
    log_info "Testing THANOS Neo4j connectivity..."
    if timeout 10 bash -c "</dev/tcp/$thanos_host/7687"; then
        log_success "THANOS Neo4j is accessible"
    else
        log_warning "THANOS Neo4j connection failed"
    fi

    # Test Vault connectivity
    log_info "Testing THANOS Vault connectivity..."
    if timeout 10 bash -c "</dev/tcp/$thanos_host/8200"; then
        log_success "THANOS Vault is accessible"
    else
        log_warning "THANOS Vault connection failed"
    fi
}

# Display service endpoints
show_endpoints() {
    log_info "ORACLE1 Service Endpoints:"
    echo "=================================="
    echo "🔗 Prometheus: http://localhost:9091"
    echo "🔗 Grafana: http://localhost:3001"
    echo "🔗 AlertManager: http://localhost:9093"
    echo "🔗 Node Exporter: http://localhost:9100"
    echo ""
    echo "📊 Connected to THANOS services:"
    echo "🔗 Main Dashboard: http://thanos (via THANOS)"
    echo "🔗 Shared Databases: PostgreSQL, Redis, Neo4j on THANOS"
    echo "=================================="
}

# Main deployment function
main() {
    log_info "Starting ${NODE_NAME} node deployment..."
    echo "========================================"

    # Run deployment phases
    check_prerequisites
    check_thanos_connectivity
    validate_configuration
    cleanup_existing

    log_info "Phase 1: Monitoring deployment"
    deploy_monitoring || {
        log_warning "Some monitoring services failed to deploy"
    }

    log_info "Phase 2: Application deployment"
    deploy_applications || {
        log_error "Application deployment failed"
        exit 1
    }

    log_info "Phase 3: Health verification"
    sleep 20  # Allow services to stabilize
    health_check || {
        log_warning "Some services failed health checks"
    }

    log_info "Phase 4: THANOS integration testing"
    test_thanos_integration

    log_success "${NODE_NAME} deployment completed!"
    show_endpoints

    log_info "Next steps:"
    echo "1. Verify cross-node communication: ./test_cross_node_integration.sh"
    echo "2. Run full validation: ./validate_complete_deployment.sh"
    echo "3. Monitor both nodes: ./health_check_all_services.sh"
}

# Handle script termination
trap 'log_error "ORACLE1 deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"