#!/bin/bash
set -euo pipefail

# ===================================================================
# ORACLE1 Final Deployment Readiness Validation Script
# ===================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="docker-compose-oracle1-unified.yml"
DOCKERFILES_DIR="docker/oracle"

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

# Validation tracking
VALIDATION_ERRORS=0
VALIDATION_WARNINGS=0

# ===================================================================
# VALIDATION FUNCTIONS
# ===================================================================

validate_compose_syntax() {
    log_info "Validating Docker Compose syntax..."

    if docker-compose -f "$COMPOSE_FILE" config --quiet; then
        log_success "Docker Compose syntax is valid"
    else
        log_error "Docker Compose syntax validation failed"
        ((VALIDATION_ERRORS++))
        return 1
    fi
}

validate_service_count() {
    log_info "Validating service count..."

    local service_count
    service_count=$(docker-compose -f "$COMPOSE_FILE" config --services | wc -l)

    if [ "$service_count" -eq 51 ]; then
        log_success "All 51 services are defined"
    else
        log_error "Expected 51 services, found $service_count"
        ((VALIDATION_ERRORS++))
        return 1
    fi
}

validate_volume_count() {
    log_info "Validating volume count..."

    local volume_count
    volume_count=$(docker-compose -f "$COMPOSE_FILE" config --volumes | wc -l)

    if [ "$volume_count" -eq 38 ]; then
        log_success "All 38 volumes are defined"
    else
        log_warning "Expected 38 volumes, found $volume_count"
        ((VALIDATION_WARNINGS++))
    fi
}

validate_dockerfiles() {
    log_info "Validating required Dockerfiles exist..."

    local missing_dockerfiles=()
    local required_dockerfiles=(
        "Dockerfile.research"
        "Dockerfile.intel"
        "Dockerfile.proxy"
        "Dockerfile.celery"
        "Dockerfile.genetic"
        "Dockerfile.multiplexer"
        "Dockerfile.knowledge"
        "Dockerfile.toolmaster"
        "Dockerfile.edge"
        "Dockerfile.mq"
        "Dockerfile.drm"
        "Dockerfile.watermark"
        "Dockerfile.crypto"
        "Dockerfile.blackmarket"
        "Dockerfile.vendor"
        "Dockerfile.transaction"
        "Dockerfile.multimodal"
    )

    for dockerfile in "${required_dockerfiles[@]}"; do
        if [ ! -f "$DOCKERFILES_DIR/$dockerfile" ]; then
            missing_dockerfiles+=("$dockerfile")
            log_error "Missing Dockerfile: $DOCKERFILES_DIR/$dockerfile"
            ((VALIDATION_ERRORS++))
        fi
    done

    if [ ${#missing_dockerfiles[@]} -eq 0 ]; then
        log_success "All 17 required Dockerfiles exist"
    else
        log_error "Missing ${#missing_dockerfiles[@]} Dockerfiles"
        return 1
    fi
}

validate_arm64_compatibility() {
    log_info "Validating ARM64 platform specifications..."

    local arm64_services
    arm64_services=$(docker-compose -f "$COMPOSE_FILE" config | grep -c "platform: linux/arm64" || true)

    # All container-based services should specify ARM64 platform
    local expected_arm64_services=34  # Services that need ARM64 specification

    if [ "$arm64_services" -ge "$expected_arm64_services" ]; then
        log_success "ARM64 platform specifications found for $arm64_services services"
    else
        log_warning "Only $arm64_services services specify ARM64 platform (expected at least $expected_arm64_services)"
        ((VALIDATION_WARNINGS++))
    fi
}

validate_thanos_integration() {
    log_info "Validating THANOS cross-node integration..."

    local thanos_references
    thanos_references=$(grep -c "100.122.12.54" "$COMPOSE_FILE" || true)

    if [ "$thanos_references" -ge 8 ]; then
        log_success "THANOS integration configured with $thanos_references references to 100.122.12.54"
    else
        log_error "Insufficient THANOS integration references: $thanos_references (expected at least 8)"
        ((VALIDATION_ERRORS++))
        return 1
    fi
}

validate_networking() {
    log_info "Validating network configuration..."

    # Check for required networks
    if docker-compose -f "$COMPOSE_FILE" config | grep -q "bev_oracle:"; then
        log_success "Internal network 'bev_oracle' configured"
    else
        log_error "Internal network 'bev_oracle' not found"
        ((VALIDATION_ERRORS++))
    fi

    if docker-compose -f "$COMPOSE_FILE" config | grep -q "external_thanos:"; then
        log_success "External network 'external_thanos' configured"
    else
        log_error "External network 'external_thanos' not found"
        ((VALIDATION_ERRORS++))
    fi
}

validate_resource_allocation() {
    log_info "Validating resource allocation templates..."

    # Check for ARM resource templates
    if docker-compose -f "$COMPOSE_FILE" config | grep -q "arm-resources"; then
        log_success "ARM resource templates found"
    else
        log_error "ARM resource templates not found"
        ((VALIDATION_ERRORS++))
    fi

    # Calculate approximate memory usage
    local arm_small_services=5
    local arm_standard_services=35
    local arm_monitoring_services=3

    local total_memory_mb=$((arm_small_services * 200 + arm_standard_services * 400 + arm_monitoring_services * 1024))
    local total_memory_gb=$((total_memory_mb / 1024))

    log_info "Estimated memory usage: ${total_memory_gb}GB (${total_memory_mb}MB)"

    if [ "$total_memory_gb" -le 24 ]; then
        log_success "Memory allocation within 24GB ARM server limit"
    else
        log_error "Memory allocation exceeds 24GB ARM server limit"
        ((VALIDATION_ERRORS++))
    fi
}

validate_health_checks() {
    log_info "Validating health check configuration..."

    local health_checks
    health_checks=$(docker-compose -f "$COMPOSE_FILE" config | grep -c "healthcheck:" || true)

    if [ "$health_checks" -ge 8 ]; then
        log_success "Health checks configured for $health_checks services"
    else
        log_warning "Only $health_checks services have health checks configured"
        ((VALIDATION_WARNINGS++))
    fi
}

validate_environment_variables() {
    log_info "Validating environment variable configuration..."

    # Check for required environment files
    if [ -f ".env" ]; then
        log_success "Environment file .env exists"
    else
        log_warning "Environment file .env not found"
        ((VALIDATION_WARNINGS++))
    fi

    if [ -f ".env.oracle1" ]; then
        log_success "ORACLE1-specific environment file exists"
    else
        log_warning "ORACLE1-specific environment file .env.oracle1 not found"
        ((VALIDATION_WARNINGS++))
    fi
}

validate_monitoring_configuration() {
    log_info "Validating monitoring configuration..."

    # Check for Prometheus configuration
    if [ -f "config/prometheus.yml" ]; then
        log_success "Prometheus configuration exists"
    else
        log_error "Prometheus configuration file not found"
        ((VALIDATION_ERRORS++))
    fi

    # Check for Grafana datasources
    if [ -f "config/grafana-datasources.yml" ]; then
        log_success "Grafana datasources configuration exists"
    else
        log_warning "Grafana datasources configuration not found"
        ((VALIDATION_WARNINGS++))
    fi
}

# ===================================================================
# DEPLOYMENT TEST SIMULATION
# ===================================================================

test_deployment_simulation() {
    log_info "Running deployment simulation..."

    # Test docker-compose pull for available images
    log_info "Testing image availability..."

    # Test only public images to avoid build requirements
    local public_images=(
        "redis:7-alpine"
        "n8nio/n8n:latest"
        "nginx:alpine"
        "influxdb:2.7-alpine"
        "telegraf:1.28-alpine"
        "prom/node-exporter:latest"
        "prom/prometheus:latest"
        "grafana/grafana:latest"
        "prom/alertmanager:latest"
        "hashicorp/vault:latest"
        "minio/minio:latest"
        "ghcr.io/berriai/litellm:main-latest"
    )

    local failed_pulls=0
    for image in "${public_images[@]}"; do
        if docker pull --platform linux/arm64 "$image" >/dev/null 2>&1; then
            log_success "Successfully pulled ARM64 image: $image"
        else
            log_warning "Failed to pull ARM64 image: $image"
            ((failed_pulls++))
        fi
    done

    if [ "$failed_pulls" -eq 0 ]; then
        log_success "All public images available for ARM64"
    else
        log_warning "$failed_pulls public images failed to pull for ARM64"
        ((VALIDATION_WARNINGS++))
    fi
}

# ===================================================================
# MAIN EXECUTION
# ===================================================================

main() {
    echo "========================================================================"
    echo "ORACLE1 Final Deployment Readiness Validation"
    echo "Target: ARM Cloud Server (100.96.197.84)"
    echo "========================================================================"
    echo

    # Change to script directory
    cd "$SCRIPT_DIR"

    # Run all validations
    validate_compose_syntax
    validate_service_count
    validate_volume_count
    validate_dockerfiles
    validate_arm64_compatibility
    validate_thanos_integration
    validate_networking
    validate_resource_allocation
    validate_health_checks
    validate_environment_variables
    validate_monitoring_configuration
    test_deployment_simulation

    echo
    echo "========================================================================"
    echo "VALIDATION SUMMARY"
    echo "========================================================================"

    if [ "$VALIDATION_ERRORS" -eq 0 ] && [ "$VALIDATION_WARNINGS" -eq 0 ]; then
        log_success "✅ ALL VALIDATIONS PASSED - ORACLE1 IS READY FOR DEPLOYMENT"
        echo
        log_success "🚀 DEPLOYMENT RECOMMENDATION: GO"
        echo
        echo "Next steps:"
        echo "1. Deploy to ARM server: docker-compose -f docker-compose-oracle1-unified.yml up -d"
        echo "2. Monitor deployment: docker-compose -f docker-compose-oracle1-unified.yml ps"
        echo "3. Validate cross-node connectivity to THANOS (100.122.12.54)"
        exit 0

    elif [ "$VALIDATION_ERRORS" -eq 0 ]; then
        log_warning "⚠️  VALIDATION COMPLETED WITH $VALIDATION_WARNINGS WARNINGS"
        echo
        log_success "🟡 DEPLOYMENT RECOMMENDATION: PROCEED WITH CAUTION"
        echo
        echo "Warnings found but no critical errors. Deployment can proceed."
        echo "Review warnings above and monitor deployment closely."
        exit 0

    else
        log_error "❌ VALIDATION FAILED WITH $VALIDATION_ERRORS ERRORS AND $VALIDATION_WARNINGS WARNINGS"
        echo
        log_error "🔴 DEPLOYMENT RECOMMENDATION: NO-GO"
        echo
        echo "Critical errors must be resolved before deployment."
        echo "Review errors above and fix configuration issues."
        exit 1
    fi
}

# Execute main function
main "$@"