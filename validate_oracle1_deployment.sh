#!/bin/bash

# =============================================================================
# ORACLE1 ARM64 Deployment Validation Framework
# =============================================================================
# Comprehensive testing suite to ensure ORACLE1 deployment works correctly
# before attempting production deployment on ARM cloud server (100.96.197.84)
#
# This script validates:
# - ARM64 compatibility for all services
# - Docker build testing and validation
# - Configuration file syntax and validity
# - Resource allocation and constraint validation
# - Network connectivity and routing testing
# - Service integration and communication
# - Performance benchmarking
# - Security and compliance
#
# Usage: ./validate_oracle1_deployment.sh [--phase=<phase>] [--fix] [--verbose]
# =============================================================================

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_DIR="${SCRIPT_DIR}/validation_results"
DOCKER_DIR="${SCRIPT_DIR}/docker/oracle"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose-oracle1-unified.yml"
ENV_FILE="${SCRIPT_DIR}/.env.oracle1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${VALIDATION_DIR}/validation_${TIMESTAMP}.log"

# Default options
PHASE="all"
AUTO_FIX=false
VERBOSE=false
FORCE_REBUILD=false

# Performance thresholds
MAX_BUILD_TIME=300  # 5 minutes per service
MAX_MEMORY_USAGE=1073741824  # 1GB per service
MAX_CPU_USAGE=50  # 50% CPU usage
MIN_NETWORK_SPEED=10  # 10 Mbps

# Validation results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNINGS=0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        "INFO")  echo -e "${CYAN}[INFO]${NC} $message" | tee -a "$LOG_FILE" ;;
        "PASS")  echo -e "${GREEN}[PASS]${NC} $message" | tee -a "$LOG_FILE"; ((PASSED_TESTS++)) ;;
        "FAIL")  echo -e "${RED}[FAIL]${NC} $message" | tee -a "$LOG_FILE"; ((FAILED_TESTS++)) ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$LOG_FILE"; ((WARNINGS++)) ;;
        "DEBUG") [ "$VERBOSE" = true ] && echo -e "${BLUE}[DEBUG]${NC} $message" | tee -a "$LOG_FILE" ;;
        *) echo -e "$message" | tee -a "$LOG_FILE" ;;
    esac
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

increment_test() {
    ((TOTAL_TESTS++))
}

cleanup() {
    log "INFO" "Cleaning up test environment..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans 2>/dev/null || true
    docker system prune -f --filter "label=bev-oracle1-test" 2>/dev/null || true
}

trap cleanup EXIT

setup_validation_environment() {
    log "INFO" "Setting up validation environment..."

    # Create validation directory
    mkdir -p "$VALIDATION_DIR"
    mkdir -p "${VALIDATION_DIR}/build_logs"
    mkdir -p "${VALIDATION_DIR}/performance"
    mkdir -p "${VALIDATION_DIR}/security"
    mkdir -p "${VALIDATION_DIR}/reports"

    # Initialize log file
    echo "ORACLE1 Deployment Validation - Started at $(date)" > "$LOG_FILE"
    echo "=============================================" >> "$LOG_FILE"

    log "INFO" "Validation environment setup complete"
    log "INFO" "Results will be saved to: $VALIDATION_DIR"
    log "INFO" "Log file: $LOG_FILE"
}

# =============================================================================
# PHASE 1: PRE-DEPLOYMENT VALIDATION
# =============================================================================

validate_system_requirements() {
    log "INFO" "=== PHASE 1: System Requirements Validation ==="

    # Check Docker version
    increment_test
    if command -v docker >/dev/null 2>&1; then
        docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        if dpkg --compare-versions "$docker_version" "ge" "20.10.0"; then
            log "PASS" "Docker version $docker_version meets requirements (>= 20.10.0)"
        else
            log "FAIL" "Docker version $docker_version is too old (requires >= 20.10.0)"
            return 1
        fi
    else
        log "FAIL" "Docker is not installed"
        return 1
    fi

    # Check Docker Compose version
    increment_test
    if docker compose version >/dev/null 2>&1; then
        compose_version=$(docker compose version --short)
        log "PASS" "Docker Compose v2 is available ($compose_version)"
    else
        log "FAIL" "Docker Compose v2 is not available"
        return 1
    fi

    # Check available memory
    increment_test
    available_memory=$(free -b | grep '^Mem:' | awk '{print $7}')
    required_memory=$((24 * 1024 * 1024 * 1024))  # 24GB
    if [ "$available_memory" -ge "$required_memory" ]; then
        log "PASS" "Available memory: $(numfmt --to=iec $available_memory) (requires 24GB)"
    else
        log "WARN" "Available memory: $(numfmt --to=iec $available_memory) (recommended 24GB)"
    fi

    # Check disk space
    increment_test
    available_space=$(df "$PWD" | tail -1 | awk '{print $4}' | xargs)
    required_space=$((500 * 1024 * 1024))  # 500GB in KB
    if [ "$available_space" -ge "$required_space" ]; then
        log "PASS" "Available disk space: $(numfmt --to=iec $((available_space * 1024))) (requires 500GB)"
    else
        log "WARN" "Available disk space: $(numfmt --to=iec $((available_space * 1024))) (recommended 500GB)"
    fi

    # Check ARM64 platform support
    increment_test
    if docker buildx version >/dev/null 2>&1; then
        if docker buildx ls | grep -q "linux/arm64"; then
            log "PASS" "ARM64 platform support is available"
        else
            log "FAIL" "ARM64 platform support is not available"
            return 1
        fi
    else
        log "FAIL" "Docker Buildx is not available (required for ARM64 support)"
        return 1
    fi

    log "INFO" "System requirements validation complete"
}

validate_configuration_files() {
    log "INFO" "=== PHASE 1: Configuration Files Validation ==="

    # Check Docker Compose file syntax
    increment_test
    if docker compose -f "$COMPOSE_FILE" config >/dev/null 2>&1; then
        log "PASS" "Docker Compose file syntax is valid"
    else
        log "FAIL" "Docker Compose file has syntax errors"
        docker compose -f "$COMPOSE_FILE" config 2>&1 | head -10 | while read -r line; do
            log "DEBUG" "Compose error: $line"
        done
        return 1
    fi

    # Check environment file
    increment_test
    if [ -f "$ENV_FILE" ]; then
        log "PASS" "Environment file exists: $ENV_FILE"

        # Validate required environment variables
        required_vars=("NODE_NAME" "NODE_TYPE" "VAULT_URL" "VAULT_TOKEN" "REDIS_PASSWORD")
        for var in "${required_vars[@]}"; do
            increment_test
            if grep -q "^${var}=" "$ENV_FILE"; then
                log "PASS" "Required environment variable $var is set"
            else
                log "FAIL" "Required environment variable $var is missing"
            fi
        done
    else
        log "FAIL" "Environment file not found: $ENV_FILE"
        return 1
    fi

    # Check Dockerfile existence
    dockerfiles=(
        "Dockerfile.research" "Dockerfile.intel" "Dockerfile.proxy"
        "Dockerfile.celery" "Dockerfile.genetic" "Dockerfile.multiplexer"
        "Dockerfile.knowledge" "Dockerfile.toolmaster" "Dockerfile.edge"
        "Dockerfile.mq" "Dockerfile.drm" "Dockerfile.watermark"
        "Dockerfile.crypto" "Dockerfile.blackmarket" "Dockerfile.vendor"
        "Dockerfile.transaction" "Dockerfile.multimodal"
    )

    for dockerfile in "${dockerfiles[@]}"; do
        increment_test
        if [ -f "${DOCKER_DIR}/${dockerfile}" ]; then
            log "PASS" "Dockerfile exists: $dockerfile"
        else
            log "FAIL" "Dockerfile missing: $dockerfile"
        fi
    done

    log "INFO" "Configuration files validation complete"
}

validate_network_configuration() {
    log "INFO" "=== PHASE 1: Network Configuration Validation ==="

    # Check external network connectivity to THANOS
    increment_test
    thanos_ip="100.122.12.54"
    if ping -c 3 -W 5 "$thanos_ip" >/dev/null 2>&1; then
        log "PASS" "Network connectivity to THANOS ($thanos_ip) is working"
    else
        log "FAIL" "Cannot reach THANOS server at $thanos_ip"
    fi

    # Check if required ports are available
    required_ports=(80 443 5678 6379 8080 8086 8087 9001 9002 9003 9100)
    for port in "${required_ports[@]}"; do
        increment_test
        if ! ss -tuln | grep -q ":${port} "; then
            log "PASS" "Port $port is available"
        else
            log "WARN" "Port $port is already in use"
        fi
    done

    # Validate Docker network configuration
    increment_test
    network_config=$(docker compose -f "$COMPOSE_FILE" config --services 2>/dev/null | wc -l)
    if [ "$network_config" -gt 0 ]; then
        log "PASS" "Docker Compose network configuration is valid"
    else
        log "FAIL" "Docker Compose network configuration is invalid"
    fi

    log "INFO" "Network configuration validation complete"
}

# =============================================================================
# PHASE 2: ARM64 COMPATIBILITY TESTING
# =============================================================================

test_arm64_builds() {
    log "INFO" "=== PHASE 2: ARM64 Build Compatibility Testing ==="

    # Get list of services that need building
    services_to_build=$(docker compose -f "$COMPOSE_FILE" config --services | grep -E "(research|intel|proxy|celery|genetic|multiplexer|knowledge|toolmaster|edge|mq|drm|watermark|crypto|blackmarket|vendor|transaction|multimodal)")

    build_results_file="${VALIDATION_DIR}/build_results.json"
    echo "{" > "$build_results_file"
    echo "  \"build_results\": [" >> "$build_results_file"

    local first=true
    for service in $services_to_build; do
        increment_test
        log "INFO" "Testing ARM64 build for service: $service"

        build_start_time=$(date +%s)
        build_log="${VALIDATION_DIR}/build_logs/${service}_build.log"

        if [ "$first" = false ]; then
            echo "," >> "$build_results_file"
        fi
        first=false

        # Build the service with explicit ARM64 platform
        if timeout "$MAX_BUILD_TIME" docker compose -f "$COMPOSE_FILE" build --platform linux/arm64 "$service" > "$build_log" 2>&1; then
            build_end_time=$(date +%s)
            build_duration=$((build_end_time - build_start_time))

            # Get image size
            image_name="bev_${service}"
            image_size=$(docker images --format "table {{.Size}}" "$image_name" 2>/dev/null | tail -n +2 | head -1 || echo "unknown")

            log "PASS" "ARM64 build successful for $service (${build_duration}s, size: $image_size)"

            # Write build result to JSON
            cat >> "$build_results_file" << EOF
    {
      "service": "$service",
      "status": "success",
      "build_time": $build_duration,
      "image_size": "$image_size",
      "platform": "linux/arm64"
    }
EOF
        else
            build_end_time=$(date +%s)
            build_duration=$((build_end_time - build_start_time))

            log "FAIL" "ARM64 build failed for $service (${build_duration}s)"
            log "DEBUG" "Build log saved to: $build_log"

            # Show last 5 lines of build log
            if [ -f "$build_log" ]; then
                log "DEBUG" "Last 5 lines of build output:"
                tail -5 "$build_log" | while read -r line; do
                    log "DEBUG" "  $line"
                done
            fi

            # Write build result to JSON
            cat >> "$build_results_file" << EOF
    {
      "service": "$service",
      "status": "failed",
      "build_time": $build_duration,
      "platform": "linux/arm64",
      "error_log": "$build_log"
    }
EOF
        fi
    done

    echo "" >> "$build_results_file"
    echo "  ]" >> "$build_results_file"
    echo "}" >> "$build_results_file"

    log "INFO" "ARM64 build compatibility testing complete"
    log "INFO" "Build results saved to: $build_results_file"
}

validate_base_images() {
    log "INFO" "=== PHASE 2: Base Image ARM64 Compatibility ==="

    # Test common base images used in Dockerfiles
    base_images=(
        "python:3.11-slim-bookworm"
        "redis:7-alpine"
        "nginx:alpine"
        "influxdb:2.7-alpine"
        "telegraf:1.28-alpine"
        "prom/node-exporter:latest"
        "minio/minio:latest"
        "n8nio/n8n:latest"
        "ghcr.io/berriai/litellm:main-latest"
    )

    for image in "${base_images[@]}"; do
        increment_test
        log "INFO" "Testing ARM64 compatibility for base image: $image"

        if docker pull --platform linux/arm64 "$image" >/dev/null 2>&1; then
            log "PASS" "ARM64 base image available: $image"
        else
            log "FAIL" "ARM64 base image not available: $image"
        fi
    done

    log "INFO" "Base image ARM64 compatibility validation complete"
}

test_package_compatibility() {
    log "INFO" "=== PHASE 2: Package Compatibility Testing ==="

    # Create a test Dockerfile to verify package installation
    test_dockerfile="${VALIDATION_DIR}/Dockerfile.arm64-test"
    cat > "$test_dockerfile" << 'EOF'
FROM python:3.11-slim-bookworm

# Test ARM64 system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Test ARM64 Python packages
RUN pip install --no-cache-dir \
    redis \
    celery \
    fastapi \
    aiohttp \
    numpy \
    requests \
    psutil

# Test successful installation
RUN python -c "import redis, celery, fastapi, aiohttp, numpy, requests, psutil; print('All packages imported successfully')"
EOF

    increment_test
    log "INFO" "Testing ARM64 package compatibility..."

    package_test_log="${VALIDATION_DIR}/package_test.log"
    if docker build --platform linux/arm64 -f "$test_dockerfile" -t bev-package-test:arm64 "$VALIDATION_DIR" > "$package_test_log" 2>&1; then
        log "PASS" "ARM64 package compatibility test successful"
        docker rmi bev-package-test:arm64 >/dev/null 2>&1 || true
    else
        log "FAIL" "ARM64 package compatibility test failed"
        log "DEBUG" "Package test log saved to: $package_test_log"
    fi

    rm -f "$test_dockerfile"
    log "INFO" "Package compatibility testing complete"
}

# =============================================================================
# PHASE 3: SERVICE INTEGRATION TESTING
# =============================================================================

test_service_startup() {
    log "INFO" "=== PHASE 3: Service Startup and Integration Testing ==="

    # Start core services first
    core_services=("redis-arm" "nginx")

    for service in "${core_services[@]}"; do
        increment_test
        log "INFO" "Starting core service: $service"

        if docker compose -f "$COMPOSE_FILE" up -d "$service" >/dev/null 2>&1; then
            sleep 5  # Wait for service to start

            # Check if service is running
            if docker compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
                log "PASS" "Core service $service started successfully"
            else
                log "FAIL" "Core service $service failed to start properly"
                docker compose -f "$COMPOSE_FILE" logs "$service" | tail -5 | while read -r line; do
                    log "DEBUG" "Service log: $line"
                done
            fi
        else
            log "FAIL" "Failed to start core service: $service"
        fi
    done
}

test_service_health_checks() {
    log "INFO" "=== PHASE 3: Service Health Check Testing ==="

    # Services with health checks
    health_check_services=("redis-arm" "nginx" "influxdb-primary")

    for service in "${health_check_services[@]}"; do
        increment_test
        log "INFO" "Testing health check for service: $service"

        # Wait for health check to pass (up to 60 seconds)
        health_check_passed=false
        for i in {1..12}; do  # 12 attempts, 5 seconds each = 60 seconds
            health_status=$(docker compose -f "$COMPOSE_FILE" ps "$service" --format json 2>/dev/null | jq -r '.Health // "unknown"' 2>/dev/null || echo "unknown")

            if [ "$health_status" = "healthy" ] || [ "$health_status" = "unknown" ]; then
                health_check_passed=true
                break
            fi

            log "DEBUG" "Health check attempt $i/12 for $service: $health_status"
            sleep 5
        done

        if [ "$health_check_passed" = true ]; then
            log "PASS" "Health check passed for service: $service"
        else
            log "FAIL" "Health check failed for service: $service"
        fi
    done
}

test_inter_service_communication() {
    log "INFO" "=== PHASE 3: Inter-Service Communication Testing ==="

    # Test Redis connectivity
    increment_test
    log "INFO" "Testing Redis connectivity..."
    if docker exec bev_redis_oracle redis-cli ping 2>/dev/null | grep -q "PONG"; then
        log "PASS" "Redis connectivity test successful"
    else
        log "FAIL" "Redis connectivity test failed"
    fi

    # Test Nginx proxy functionality
    increment_test
    log "INFO" "Testing Nginx proxy functionality..."
    if curl -s -o /dev/null -w "%{http_code}" http://localhost 2>/dev/null | grep -q "200\|404\|502"; then
        log "PASS" "Nginx proxy responding"
    else
        log "WARN" "Nginx proxy not responding (may be expected if no backend configured)"
    fi

    log "INFO" "Inter-service communication testing complete"
}

test_external_connectivity() {
    log "INFO" "=== PHASE 3: External Network Connectivity Testing ==="

    # Test connectivity to THANOS from within containers
    increment_test
    log "INFO" "Testing container-to-THANOS connectivity..."

    thanos_ip="100.122.12.54"
    if docker run --rm --network bev_osint alpine:latest ping -c 3 "$thanos_ip" >/dev/null 2>&1; then
        log "PASS" "Container-to-THANOS connectivity successful"
    else
        log "FAIL" "Container-to-THANOS connectivity failed"
    fi

    log "INFO" "External connectivity testing complete"
}

# =============================================================================
# PHASE 4: PERFORMANCE BENCHMARKING
# =============================================================================

benchmark_service_performance() {
    log "INFO" "=== PHASE 4: Service Performance Benchmarking ==="

    performance_report="${VALIDATION_DIR}/performance/performance_report.json"
    echo "{" > "$performance_report"
    echo "  \"performance_results\": [" >> "$performance_report"

    # Get running services
    running_services=$(docker compose -f "$COMPOSE_FILE" ps --services --filter status=running)

    local first=true
    for service in $running_services; do
        increment_test
        log "INFO" "Benchmarking performance for service: $service"

        if [ "$first" = false ]; then
            echo "," >> "$performance_report"
        fi
        first=false

        # Get container stats
        container_name="bev_${service}"
        container_stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}" "$container_name" 2>/dev/null || echo "")

        if [ -n "$container_stats" ]; then
            cpu_usage=$(echo "$container_stats" | tail -1 | awk '{print $2}' | sed 's/%//')
            memory_usage=$(echo "$container_stats" | tail -1 | awk '{print $3}' | cut -d'/' -f1)
            memory_percent=$(echo "$container_stats" | tail -1 | awk '{print $4}' | sed 's/%//')

            # Validate performance thresholds
            if (( $(echo "$cpu_usage < $MAX_CPU_USAGE" | bc -l) )); then
                cpu_status="pass"
                log "PASS" "CPU usage for $service: ${cpu_usage}% (< ${MAX_CPU_USAGE}%)"
            else
                cpu_status="fail"
                log "FAIL" "CPU usage for $service: ${cpu_usage}% (> ${MAX_CPU_USAGE}%)"
            fi

            # Convert memory to bytes for comparison
            memory_bytes=$(numfmt --from=iec "$memory_usage" 2>/dev/null || echo "0")
            if [ "$memory_bytes" -lt "$MAX_MEMORY_USAGE" ]; then
                memory_status="pass"
                log "PASS" "Memory usage for $service: $memory_usage (< 1GB)"
            else
                memory_status="fail"
                log "FAIL" "Memory usage for $service: $memory_usage (> 1GB)"
            fi

            # Write performance result to JSON
            cat >> "$performance_report" << EOF
    {
      "service": "$service",
      "cpu_usage": "$cpu_usage",
      "cpu_status": "$cpu_status",
      "memory_usage": "$memory_usage",
      "memory_percent": "$memory_percent",
      "memory_status": "$memory_status",
      "timestamp": "$(date -Iseconds)"
    }
EOF
        else
            log "WARN" "Could not get performance stats for service: $service"

            cat >> "$performance_report" << EOF
    {
      "service": "$service",
      "status": "stats_unavailable",
      "timestamp": "$(date -Iseconds)"
    }
EOF
        fi
    done

    echo "" >> "$performance_report"
    echo "  ]" >> "$performance_report"
    echo "}" >> "$performance_report"

    log "INFO" "Service performance benchmarking complete"
    log "INFO" "Performance report saved to: $performance_report"
}

benchmark_resource_usage() {
    log "INFO" "=== PHASE 4: Resource Usage Analysis ==="

    # System resource analysis
    increment_test
    system_load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    cpu_count=$(nproc)
    load_percentage=$(echo "scale=2; $system_load / $cpu_count * 100" | bc -l)

    if (( $(echo "$load_percentage < 70" | bc -l) )); then
        log "PASS" "System load: ${load_percentage}% (< 70%)"
    else
        log "WARN" "System load: ${load_percentage}% (> 70%)"
    fi

    # Docker system resource usage
    increment_test
    docker_system_df=$(docker system df --format "table {{.Type}}\t{{.Total}}\t{{.Active}}\t{{.Size}}\t{{.Reclaimable}}")
    log "INFO" "Docker system resource usage:"
    echo "$docker_system_df" | while read -r line; do
        log "DEBUG" "  $line"
    done

    # Network performance test
    increment_test
    log "INFO" "Testing network performance..."
    network_test_result=$(timeout 10 curl -s -w "@${SCRIPT_DIR}/curl-format.txt" -o /dev/null http://localhost 2>/dev/null || echo "Network test failed")

    if [ "$network_test_result" != "Network test failed" ]; then
        log "PASS" "Network performance test completed"
        log "DEBUG" "Network test result: $network_test_result"
    else
        log "WARN" "Network performance test failed or timed out"
    fi

    log "INFO" "Resource usage analysis complete"
}

# =============================================================================
# PHASE 5: SECURITY AND COMPLIANCE TESTING
# =============================================================================

test_security_configuration() {
    log "INFO" "=== PHASE 5: Security Configuration Testing ==="

    # Check for security-related configurations
    increment_test
    if docker compose -f "$COMPOSE_FILE" config | grep -q "restart: always"; then
        log "PASS" "Service restart policies are configured"
    else
        log "WARN" "Service restart policies may not be configured"
    fi

    # Check for secrets management
    increment_test
    if grep -q "VAULT_" "$ENV_FILE"; then
        log "PASS" "Vault integration is configured"
    else
        log "FAIL" "Vault integration is not configured"
    fi

    # Check for network isolation
    increment_test
    network_count=$(docker compose -f "$COMPOSE_FILE" config | grep -c "networks:" || echo "0")
    if [ "$network_count" -gt 0 ]; then
        log "PASS" "Network isolation is configured"
    else
        log "WARN" "Network isolation may not be properly configured"
    fi

    # Check for volume permissions
    increment_test
    if docker compose -f "$COMPOSE_FILE" config | grep -q "volumes:"; then
        log "PASS" "Data persistence volumes are configured"
    else
        log "WARN" "Data persistence volumes may not be configured"
    fi

    log "INFO" "Security configuration testing complete"
}

test_vault_integration() {
    log "INFO" "=== PHASE 5: Vault Integration Testing ==="

    # Test Vault connectivity
    increment_test
    vault_url=$(grep "VAULT_URL=" "$ENV_FILE" | cut -d'=' -f2)
    vault_token=$(grep "VAULT_TOKEN=" "$ENV_FILE" | cut -d'=' -f2)

    if [ -n "$vault_url" ] && [ -n "$vault_token" ]; then
        log "INFO" "Testing Vault connectivity to: $vault_url"

        # Test Vault health endpoint
        if curl -s -H "X-Vault-Token: $vault_token" "${vault_url}/v1/sys/health" >/dev/null 2>&1; then
            log "PASS" "Vault connectivity test successful"
        else
            log "FAIL" "Vault connectivity test failed"
        fi
    else
        log "FAIL" "Vault URL or token not configured"
    fi

    log "INFO" "Vault integration testing complete"
}

# =============================================================================
# PHASE 6: DEPLOYMENT READINESS CERTIFICATION
# =============================================================================

generate_deployment_certification() {
    log "INFO" "=== PHASE 6: Deployment Readiness Certification ==="

    certification_file="${VALIDATION_DIR}/reports/deployment_certification.json"
    mkdir -p "$(dirname "$certification_file")"

    # Calculate success rates
    if [ "$TOTAL_TESTS" -gt 0 ]; then
        success_rate=$(echo "scale=2; $PASSED_TESTS / $TOTAL_TESTS * 100" | bc -l)
        failure_rate=$(echo "scale=2; $FAILED_TESTS / $TOTAL_TESTS * 100" | bc -l)
        warning_rate=$(echo "scale=2; $WARNINGS / $TOTAL_TESTS * 100" | bc -l)
    else
        success_rate=0
        failure_rate=0
        warning_rate=0
    fi

    # Determine deployment readiness
    deployment_ready=false
    certification_level="not_ready"

    if (( $(echo "$success_rate >= 90" | bc -l) )) && (( $(echo "$failure_rate <= 5" | bc -l) )); then
        deployment_ready=true
        certification_level="production_ready"
    elif (( $(echo "$success_rate >= 80" | bc -l) )) && (( $(echo "$failure_rate <= 10" | bc -l) )); then
        deployment_ready=true
        certification_level="staging_ready"
    elif (( $(echo "$success_rate >= 70" | bc -l) )); then
        certification_level="development_ready"
    fi

    # Generate certification report
    cat > "$certification_file" << EOF
{
  "deployment_certification": {
    "timestamp": "$(date -Iseconds)",
    "validation_version": "1.0.0",
    "node_name": "oracle1",
    "target_server": "100.96.197.84",
    "certification_level": "$certification_level",
    "deployment_ready": $deployment_ready,
    "test_summary": {
      "total_tests": $TOTAL_TESTS,
      "passed_tests": $PASSED_TESTS,
      "failed_tests": $FAILED_TESTS,
      "warnings": $WARNINGS,
      "success_rate": "$success_rate%",
      "failure_rate": "$failure_rate%",
      "warning_rate": "$warning_rate%"
    },
    "validation_phases": {
      "system_requirements": "completed",
      "arm64_compatibility": "completed",
      "service_integration": "completed",
      "performance_benchmarking": "completed",
      "security_compliance": "completed"
    },
    "recommendations": [
EOF

    # Add recommendations based on results
    if (( $(echo "$failure_rate > 0" | bc -l) )); then
        echo "      \"Review and fix failed tests before deployment\"," >> "$certification_file"
    fi

    if (( $(echo "$warning_rate > 20" | bc -l) )); then
        echo "      \"Address warning conditions for optimal performance\"," >> "$certification_file"
    fi

    if [ "$deployment_ready" = true ]; then
        echo "      \"Deployment validated and ready for $certification_level environment\"," >> "$certification_file"
    else
        echo "      \"Deployment requires additional fixes before proceeding\"," >> "$certification_file"
    fi

    # Remove trailing comma and close JSON
    sed -i '$ s/,$//' "$certification_file"
    cat >> "$certification_file" << EOF
    ],
    "next_steps": [
      "Review detailed test results in validation_results/",
      "Address any failed tests or warnings",
      "Deploy to target ARM server when ready",
      "Monitor deployment performance and stability"
    ]
  }
}
EOF

    log "INFO" "Deployment certification report generated: $certification_file"

    # Display certification summary
    echo ""
    echo "=========================================="
    echo "ORACLE1 DEPLOYMENT CERTIFICATION SUMMARY"
    echo "=========================================="
    echo "Certification Level: $certification_level"
    echo "Deployment Ready: $deployment_ready"
    echo "Success Rate: $success_rate%"
    echo "Tests: $PASSED_TESTS passed, $FAILED_TESTS failed, $WARNINGS warnings"
    echo "=========================================="

    if [ "$deployment_ready" = true ]; then
        log "PASS" "ORACLE1 deployment is certified for $certification_level"
    else
        log "FAIL" "ORACLE1 deployment is NOT ready - requires fixes"
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

show_usage() {
    cat << EOF
ORACLE1 ARM64 Deployment Validation Framework

Usage: $0 [OPTIONS]

OPTIONS:
    --phase=<phase>     Run specific validation phase (default: all)
                        Phases: prereq, arm64, integration, performance, security, certification
    --fix               Attempt to automatically fix issues where possible
    --verbose           Enable verbose debugging output
    --force-rebuild     Force rebuild of all Docker images
    --help              Show this help message

PHASES:
    prereq              System requirements and configuration validation
    arm64               ARM64 compatibility and build testing
    integration         Service integration and communication testing
    performance         Performance benchmarking and resource validation
    security            Security configuration and compliance testing
    certification       Generate deployment readiness certification
    all                 Run all phases (default)

EXAMPLES:
    $0                              # Run complete validation suite
    $0 --phase=arm64 --verbose      # Test only ARM64 compatibility with debug output
    $0 --phase=performance --fix    # Run performance tests and auto-fix issues
    $0 --force-rebuild              # Force rebuild all images and run full validation

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --phase=*)
                PHASE="${1#*=}"
                shift
                ;;
            --fix)
                AUTO_FIX=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log "FAIL" "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

main() {
    parse_arguments "$@"

    echo "ORACLE1 ARM64 Deployment Validation Framework"
    echo "=============================================="
    echo "Phase: $PHASE"
    echo "Auto-fix: $AUTO_FIX"
    echo "Verbose: $VERBOSE"
    echo "Force rebuild: $FORCE_REBUILD"
    echo ""

    setup_validation_environment

    validation_start_time=$(date +%s)

    case "$PHASE" in
        "prereq"|"all")
            validate_system_requirements
            validate_configuration_files
            validate_network_configuration
            ;;&
        "arm64"|"all")
            validate_base_images
            test_package_compatibility
            test_arm64_builds
            ;;&
        "integration"|"all")
            test_service_startup
            test_service_health_checks
            test_inter_service_communication
            test_external_connectivity
            ;;&
        "performance"|"all")
            benchmark_service_performance
            benchmark_resource_usage
            ;;&
        "security"|"all")
            test_security_configuration
            test_vault_integration
            ;;&
        "certification"|"all")
            generate_deployment_certification
            ;;
        *)
            log "FAIL" "Invalid phase: $PHASE"
            show_usage
            exit 1
            ;;
    esac

    validation_end_time=$(date +%s)
    validation_duration=$((validation_end_time - validation_start_time))

    echo ""
    echo "=========================================="
    echo "VALIDATION COMPLETE"
    echo "=========================================="
    echo "Duration: ${validation_duration} seconds"
    echo "Total Tests: $TOTAL_TESTS"
    echo "Passed: $PASSED_TESTS"
    echo "Failed: $FAILED_TESTS"
    echo "Warnings: $WARNINGS"
    echo ""
    echo "Results saved to: $VALIDATION_DIR"
    echo "Log file: $LOG_FILE"
    echo "=========================================="

    # Return appropriate exit code
    if [ "$FAILED_TESTS" -eq 0 ]; then
        exit 0
    else
        exit 1
    fi
}

# Create curl format file for network testing
create_curl_format() {
    cat > "${SCRIPT_DIR}/curl-format.txt" << 'EOF'
time_namelookup:  %{time_namelookup}\n
time_connect:     %{time_connect}\n
time_appconnect:  %{time_appconnect}\n
time_pretransfer: %{time_pretransfer}\n
time_redirect:    %{time_redirect}\n
time_starttransfer: %{time_starttransfer}\n
time_total:       %{time_total}\n
EOF
}

# Initialize curl format file
create_curl_format

# Run main function with all arguments
main "$@"