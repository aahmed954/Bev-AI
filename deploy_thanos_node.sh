#!/bin/bash
# THANOS Node Deployment Script
# BEV OSINT Framework - Primary Compute Node
# Generated: $(date)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose-thanos-unified.yml"
ENV_FILE="${SCRIPT_DIR}/.env.thanos.complete"
NODE_NAME="THANOS"
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

    # Check files exist
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        log_error "Docker Compose file not found: $COMPOSE_FILE"
        exit 1
    fi

    if [[ ! -f "$ENV_FILE" ]]; then
        log_error "Environment file not found: $ENV_FILE"
        exit 1
    fi

    # Check system resources
    total_memory=$(free -g | awk '/^Mem:/{print $2}')
    if (( total_memory < 16 )); then
        log_warning "System has ${total_memory}GB RAM. Recommended: 16GB+"
    fi

    # Check disk space
    disk_space=$(df "$SCRIPT_DIR" | awk 'NR==2{print $4}')
    if (( disk_space < 50000000 )); then  # 50GB in KB
        log_warning "Low disk space. Recommended: 100GB+ free"
    fi

    log_success "Prerequisites check passed"
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
    log_info "Cleaning up existing BEV containers..."

    # Stop and remove existing BEV containers
    if docker ps -a --format "table {{.Names}}" | grep -q "bev_"; then
        log_info "Stopping existing BEV containers..."
        docker ps -a --format "{{.Names}}" | grep "bev_" | xargs -r docker stop || true
        docker ps -a --format "{{.Names}}" | grep "bev_" | xargs -r docker rm || true
    fi

    # Clean up networks
    if docker network ls | grep -q "bev_"; then
        log_info "Removing BEV networks..."
        docker network ls --format "{{.Name}}" | grep "bev_" | xargs -r docker network rm || true
    fi

    log_success "Cleanup completed"
}

# Deploy core infrastructure first
deploy_infrastructure() {
    log_info "Deploying core infrastructure services..."

    # Deploy in phases to manage dependencies
    local infrastructure_services=(
        "postgres"
        "redis"
        "neo4j"
        "elasticsearch"
        "rabbitmq"
        "influxdb"
    )

    for service in "${infrastructure_services[@]}"; do
        log_info "Starting $service..."
        if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d "$service"; then
            log_success "$service started"

            # Wait for service to be healthy
            log_info "Waiting for $service to be healthy..."
            timeout=60
            while (( timeout > 0 )); do
                if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps "$service" | grep -q "Up"; then
                    log_success "$service is healthy"
                    break
                fi
                sleep 2
                ((timeout -= 2))
            done

            if (( timeout <= 0 )); then
                log_warning "$service health check timeout"
            fi
        else
            log_error "Failed to start $service"
            return 1
        fi
    done
}

# Deploy application services
deploy_applications() {
    log_info "Deploying application services..."

    # Deploy remaining services
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d; then
        log_success "All services deployment initiated"
    else
        log_error "Failed to deploy application services"
        return 1
    fi
}

# Health check all services
health_check() {
    log_info "Performing comprehensive health check..."

    local failed_services=()
    local total_services=0
    local healthy_services=0

    # Get all service names
    while IFS= read -r service; do
        [[ -n "$service" ]] || continue
        ((total_services++))

        log_info "Checking $service..."
        if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps "$service" | grep -q "Up"; then
            log_success "$service is running"
            ((healthy_services++))
        else
            log_warning "$service is not healthy"
            failed_services+=("$service")
        fi
    done < <(docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" config --services)

    # Report results
    log_info "Health Check Results:"
    log_info "Total Services: $total_services"
    log_info "Healthy Services: $healthy_services"
    log_info "Failed Services: ${#failed_services[@]}"

    if (( ${#failed_services[@]} > 0 )); then
        log_warning "Failed services: ${failed_services[*]}"
    fi

    # Calculate success rate
    success_rate=$(( (healthy_services * 100) / total_services ))
    if (( success_rate >= 80 )); then
        log_success "Deployment success rate: ${success_rate}% (ACCEPTABLE)"
        return 0
    else
        log_error "Deployment success rate: ${success_rate}% (UNACCEPTABLE)"
        return 1
    fi
}

# Test key functionality
test_functionality() {
    log_info "Testing key functionality..."

    # Test database connections
    log_info "Testing PostgreSQL connection..."
    if docker exec -it $(docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps -q postgres 2>/dev/null) pg_isready -U bev -d osint 2>/dev/null; then
        log_success "PostgreSQL connection successful"
    else
        log_warning "PostgreSQL connection failed"
    fi

    # Test Redis
    log_info "Testing Redis connection..."
    if docker exec -it $(docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps -q redis 2>/dev/null) redis-cli ping 2>/dev/null | grep -q "PONG"; then
        log_success "Redis connection successful"
    else
        log_warning "Redis connection failed"
    fi

    # Test Neo4j
    log_info "Testing Neo4j connection..."
    if docker exec -it $(docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps -q neo4j 2>/dev/null) cypher-shell -u neo4j -p "${NEO4J_PASSWORD:-a5fc5f41681920109074ad28ba9cff3c}" "RETURN 1" 2>/dev/null; then
        log_success "Neo4j connection successful"
    else
        log_warning "Neo4j connection failed"
    fi
}

# Display service endpoints
show_endpoints() {
    log_info "Service Endpoints:"
    echo "=================================="
    echo "🔗 IntelOwl Dashboard: http://localhost"
    echo "🔗 Neo4j Browser: http://localhost:7474"
    echo "🔗 Grafana Monitoring: http://localhost:3000"
    echo "🔗 MCP API Server: http://localhost:3010"
    echo "🔗 Prometheus Metrics: http://localhost:9090"
    echo "🔗 Elasticsearch: http://localhost:9200"
    echo "🔗 InfluxDB: http://localhost:8086"
    echo "=================================="
}

# Main deployment function
main() {
    log_info "Starting ${NODE_NAME} node deployment..."
    echo "========================================"

    # Run deployment phases
    check_prerequisites
    validate_configuration
    cleanup_existing

    log_info "Phase 1: Infrastructure deployment"
    deploy_infrastructure || {
        log_error "Infrastructure deployment failed"
        exit 1
    }

    log_info "Phase 2: Application deployment"
    deploy_applications || {
        log_error "Application deployment failed"
        exit 1
    }

    log_info "Phase 3: Health verification"
    sleep 30  # Allow services to stabilize
    health_check || {
        log_warning "Some services failed health checks"
    }

    log_info "Phase 4: Functionality testing"
    test_functionality

    log_success "${NODE_NAME} deployment completed!"
    show_endpoints

    log_info "Next steps:"
    echo "1. Deploy ORACLE1 node: ./deploy_oracle1_node.sh"
    echo "2. Test cross-node communication"
    echo "3. Run validation: ./validate_complete_deployment.sh"
}

# Handle script termination
trap 'log_error "Deployment interrupted"; exit 1' INT TERM

# Run main function
main "$@"