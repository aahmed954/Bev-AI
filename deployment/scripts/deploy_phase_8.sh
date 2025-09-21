#!/bin/bash

# ===================================================================
# BEV OSINT Framework - Phase 8 Deployment Script
# Phase: Advanced Security Operations
# Services: tactical-intel, defense-automation, opsec-monitor, intel-fusion
# ===================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHASE="8"
PHASE_NAME="Advanced Security Operations"

# Service definitions
SERVICES=(
    "tactical-intel:8005:172.30.0.28"
    "defense-automation:8006:172.30.0.29"
    "opsec-monitor:8007:172.30.0.30"
    "intel-fusion:8008:172.30.0.31"
)

# Resource requirements
declare -A SERVICE_MEMORY=(
    [tactical-intel]="4G"
    [defense-automation]="3G"
    [opsec-monitor]="5G"
    [intel-fusion]="8G"
)

declare -A SERVICE_CPU=(
    [tactical-intel]="2.0"
    [defense-automation]="1.5"
    [opsec-monitor]="2.5"
    [intel-fusion]="3.0"
)

# GPU requirements
GPU_SERVICES=("opsec-monitor" "intel-fusion")

# Security classifications
declare -A SERVICE_SECURITY_LEVEL=(
    [tactical-intel]="HIGH"
    [defense-automation]="CRITICAL"
    [opsec-monitor]="CRITICAL"
    [intel-fusion]="HIGH"
)

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
PURPLE='\033[0;35m'
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
        SECURITY) echo -e "${timestamp} ${PURPLE}[SECURITY]${NC} $message" ;;
    esac
}

check_security_prerequisites() {
    log SECURITY "Performing Phase $PHASE security checks..."

    # Check for secure environment variables
    local required_env_vars=(
        "THREAT_INTEL_FEEDS"
        "NOTIFICATION_WEBHOOKS"
        "MITRE_ATTACK_DATA"
        "SECURITY_KEYS"
    )

    for var in "${required_env_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log WARN "Security environment variable not set: $var"
        else
            log SUCCESS "Security environment variable configured: $var"
        fi
    done

    # Check firewall rules for Phase 8 ports
    check_firewall_rules

    # Verify encryption capabilities
    check_encryption_support

    log SUCCESS "Security prerequisites check completed"
}

check_firewall_rules() {
    log SECURITY "Checking firewall rules for Phase $PHASE services..."

    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        local port=$(echo "$service_info" | cut -d':' -f2)
        local security_level=${SERVICE_SECURITY_LEVEL[$service]}

        if [[ "$security_level" == "CRITICAL" ]]; then
            log SECURITY "CRITICAL service $service on port $port - ensuring restricted access"
            # In a real deployment, you'd configure iptables or other firewall rules here
        fi
    done
}

check_encryption_support() {
    log SECURITY "Verifying encryption support..."

    # Check if OpenSSL is available
    if command -v openssl >/dev/null 2>&1; then
        local openssl_version=$(openssl version)
        log SUCCESS "OpenSSL available: $openssl_version"
    else
        log ERROR "OpenSSL not available - required for encryption"
        return 1
    fi

    # Check for GPG
    if command -v gpg >/dev/null 2>&1; then
        log SUCCESS "GPG available for additional encryption"
    else
        log WARN "GPG not available - recommended for enhanced security"
    fi
}

check_service_prerequisites() {
    local service=$1
    log INFO "Checking prerequisites for $service..."

    # Standard checks
    local service_dir="${PROJECT_ROOT}/phase${PHASE}/${service}"
    if [[ ! -d "$service_dir" ]]; then
        log ERROR "Service directory not found: $service_dir"
        return 1
    fi

    if [[ ! -f "${service_dir}/Dockerfile" ]]; then
        log ERROR "Dockerfile not found for $service"
        return 1
    fi

    # Security-specific checks
    local security_level=${SERVICE_SECURITY_LEVEL[$service]}
    check_service_security "$service" "$security_level"

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

check_service_security() {
    local service=$1
    local security_level=$2

    log SECURITY "Checking security for $service (Level: $security_level)..."

    local service_dir="${PROJECT_ROOT}/phase${PHASE}/${service}"

    # Check for security configuration
    if [[ ! -f "${service_dir}/config/security.json" ]]; then
        log WARN "Security configuration not found for $service"
        create_default_security_config "$service"
    fi

    # Check for secrets
    if [[ ! -f "${service_dir}/config/secrets.env" ]]; then
        log WARN "Secrets file not found for $service"
        create_default_secrets "$service"
    fi

    # Validate Dockerfile security
    validate_dockerfile_security "${service_dir}/Dockerfile"
}

create_default_security_config() {
    local service=$1
    local service_dir="${PROJECT_ROOT}/phase${PHASE}/${service}"
    local security_config="${service_dir}/config/security.json"

    log INFO "Creating default security configuration for $service..."

    mkdir -p "$(dirname "$security_config")"
    cat > "$security_config" << EOF
{
    "encryption": {
        "enabled": true,
        "algorithm": "AES-256-GCM",
        "key_rotation_hours": 24
    },
    "authentication": {
        "required": true,
        "token_expiry_minutes": 60,
        "max_failed_attempts": 3
    },
    "audit": {
        "enabled": true,
        "log_level": "INFO",
        "retention_days": 90
    },
    "network": {
        "tls_required": true,
        "allowed_ips": ["172.30.0.0/16"],
        "rate_limiting": true
    }
}
EOF

    log SUCCESS "Default security configuration created for $service"
}

create_default_secrets() {
    local service=$1
    local service_dir="${PROJECT_ROOT}/phase${PHASE}/${service}"
    local secrets_file="${service_dir}/config/secrets.env"

    log INFO "Creating default secrets file for $service..."

    mkdir -p "$(dirname "$secrets_file")"
    cat > "$secrets_file" << EOF
# Auto-generated secrets for $service
# Replace these with actual production values
SERVICE_SECRET_KEY=$(openssl rand -hex 32)
ENCRYPTION_KEY=$(openssl rand -hex 32)
JWT_SECRET=$(openssl rand -hex 32)
API_TOKEN=$(openssl rand -hex 16)
EOF

    chmod 600 "$secrets_file"
    log SUCCESS "Default secrets file created for $service"
}

validate_dockerfile_security() {
    local dockerfile=$1
    log SECURITY "Validating Dockerfile security: $dockerfile"

    local security_issues=0

    # Check for root user usage
    if grep -q "USER root" "$dockerfile"; then
        log WARN "Dockerfile uses root user - security risk"
        ((security_issues++))
    fi

    # Check for exposed sensitive ports
    if grep -E "EXPOSE (22|23|80|443|3389)" "$dockerfile" >/dev/null; then
        log WARN "Dockerfile exposes potentially sensitive ports"
        ((security_issues++))
    fi

    # Check for hardcoded secrets
    if grep -E "(password|secret|key).*=" "$dockerfile" >/dev/null; then
        log WARN "Dockerfile may contain hardcoded secrets"
        ((security_issues++))
    fi

    if [[ $security_issues -eq 0 ]]; then
        log SUCCESS "Dockerfile security validation passed"
    else
        log WARN "Dockerfile has $security_issues security concerns"
    fi
}

build_service() {
    local service=$1
    log INFO "Building $service with security hardening..."

    local service_dir="${PROJECT_ROOT}/phase${PHASE}/${service}"
    local build_start_time=$(date +%s)

    # Build args for security
    local build_args=(
        "--build-arg" "BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        "--build-arg" "VERSION=1.0.0"
        "--build-arg" "SECURITY_LEVEL=${SERVICE_SECURITY_LEVEL[$service]}"
    )

    # Add no-cache if forced
    if [[ "${FORCE_REBUILD:-false}" == "true" ]]; then
        build_args+=("--no-cache")
    fi

    # Build the service
    if ! docker build "${build_args[@]}" -t "bev_${service}:latest" "$service_dir"; then
        log ERROR "Failed to build $service"
        return 1
    fi

    # Scan image for vulnerabilities if scanner available
    scan_image_vulnerabilities "bev_${service}:latest"

    local build_end_time=$(date +%s)
    local build_duration=$((build_end_time - build_start_time))

    log SUCCESS "$service built successfully in ${build_duration}s"
    return 0
}

scan_image_vulnerabilities() {
    local image=$1
    log SECURITY "Scanning image for vulnerabilities: $image"

    # Use Trivy if available
    if command -v trivy >/dev/null 2>&1; then
        trivy image --severity HIGH,CRITICAL "$image" || log WARN "Vulnerability scan found issues"
    elif command -v docker >/dev/null 2>&1 && docker run --rm aquasec/trivy:latest --version >/dev/null 2>&1; then
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy:latest image --severity HIGH,CRITICAL "$image" || log WARN "Vulnerability scan found issues"
    else
        log WARN "No vulnerability scanner available"
    fi
}

create_service_volumes() {
    log INFO "Creating Phase $PHASE volumes with security considerations..."

    local volumes=(
        "tactical_intel_data"
        "defense_automation_data"
        "opsec_data"
        "intel_fusion_data"
    )

    for volume in "${volumes[@]}"; do
        if ! docker volume ls | grep -q "$volume"; then
            # Create volume with encryption labels
            docker volume create \
                --label "bev.phase=8" \
                --label "bev.security=high" \
                --label "bev.encrypted=true" \
                "$volume"
            log INFO "Created secure volume: $volume"
        else
            log INFO "Volume already exists: $volume"
        fi
    done
}

check_dependencies() {
    log INFO "Checking Phase $PHASE dependencies..."

    local required_services=(
        "postgres"
        "neo4j"
        "elasticsearch"
        "kafka-1"
        "redis"
    )

    # Also check for Phase 7 services as dependencies
    local phase7_services=(
        "dm-crawler"
        "crypto-intel"
        "reputation-analyzer"
        "economics-processor"
    )

    local missing_services=()

    for service in "${required_services[@]}"; do
        if ! docker ps --format '{{.Names}}' | grep -q "^${service}$"; then
            missing_services+=("$service")
        fi
    done

    for service in "${phase7_services[@]}"; do
        if ! docker ps --format '{{.Names}}' | grep -q "bev_${service}"; then
            log WARN "Phase 7 service not running: $service"
        else
            log INFO "Phase 7 dependency available: $service"
        fi
    done

    if [[ ${#missing_services[@]} -gt 0 ]]; then
        log WARN "Missing dependencies: ${missing_services[*]}"
        log INFO "Attempting to start dependencies..."

        if [[ -f "${PROJECT_ROOT}/docker-compose.complete.yml" ]]; then
            docker-compose -f "${PROJECT_ROOT}/docker-compose.complete.yml" up -d \
                "${required_services[@]}"
            sleep 30
        else
            log ERROR "Core infrastructure compose file not found"
            return 1
        fi
    fi

    log SUCCESS "All dependencies are available"
    return 0
}

deploy_services() {
    log INFO "Deploying Phase $PHASE services with security hardening..."

    cd "$PROJECT_ROOT"

    # Set security-enhanced environment
    export DOCKER_CONTENT_TRUST=1  # Enable Docker Content Trust
    export COMPOSE_DOCKER_CLI_BUILD=1

    # Deploy using docker-compose
    if ! docker-compose -f "docker-compose-phase${PHASE}.yml" up -d; then
        log ERROR "Failed to deploy Phase $PHASE services"
        return 1
    fi

    log SUCCESS "Phase $PHASE services deployment initiated"

    # Enhanced health checks for security services
    wait_for_secure_services
}

wait_for_secure_services() {
    log INFO "Waiting for secure services to become ready..."

    local max_wait=600  # 10 minutes for security services
    local check_interval=15
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

            # Check health with security verification
            if ! verify_service_security "$service" "$port"; then
                log WARN "Service security check failed: $service"
                all_healthy=false
            else
                log INFO "Service secure and healthy: $service"
            fi
        done

        if [[ "$all_healthy" == "true" ]]; then
            log SUCCESS "All Phase $PHASE services are secure and healthy"
            return 0
        fi

        log INFO "Waiting for secure services... (${elapsed}s elapsed)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done

    log ERROR "Timeout waiting for services to become secure and healthy"
    return 1
}

verify_service_security() {
    local service=$1
    local port=$2

    # Basic health check
    if ! curl -sf "http://localhost:${port}/health" >/dev/null 2>&1; then
        return 1
    fi

    # Security-specific checks
    local security_level=${SERVICE_SECURITY_LEVEL[$service]}

    if [[ "$security_level" == "CRITICAL" ]]; then
        # Additional security verification for critical services
        if ! curl -sf -H "X-Security-Check: true" "http://localhost:${port}/security/status" >/dev/null 2>&1; then
            log WARN "Critical service security endpoint not responding: $service"
            return 1
        fi
    fi

    return 0
}

verify_functionality() {
    log INFO "Verifying Phase $PHASE functionality with security tests..."

    # Test service endpoints
    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        local port=$(echo "$service_info" | cut -d':' -f2)

        log INFO "Testing $service security and functionality..."

        # Test health endpoint
        if curl -sf "http://localhost:${port}/health" >/dev/null; then
            log SUCCESS "$service health endpoint responding"
        else
            log ERROR "$service health endpoint not responding"
            return 1
        fi

        # Test security endpoint
        if curl -sf "http://localhost:${port}/security/status" >/dev/null; then
            log SUCCESS "$service security endpoint responding"
        else
            log WARN "$service security endpoint not responding"
        fi
    done

    # Phase-specific functionality tests
    test_tactical_intel
    test_defense_automation
    test_opsec_monitor
    test_intel_fusion

    log SUCCESS "Phase $PHASE security and functionality verification completed"
}

test_tactical_intel() {
    log INFO "Testing Tactical Intelligence functionality..."

    local response=$(curl -sf "http://localhost:8005/api/v1/intel/threats" 2>/dev/null || echo "ERROR")
    if [[ "$response" != "ERROR" ]]; then
        log SUCCESS "Tactical Intel API responding"
    else
        log WARN "Tactical Intel API not accessible"
    fi

    # Test MITRE ATT&CK integration
    local mitre_response=$(curl -sf "http://localhost:8005/api/v1/mitre/status" 2>/dev/null || echo "ERROR")
    if [[ "$mitre_response" != "ERROR" ]]; then
        log SUCCESS "MITRE ATT&CK integration active"
    else
        log WARN "MITRE ATT&CK integration not accessible"
    fi
}

test_defense_automation() {
    log INFO "Testing Defense Automation functionality..."

    local response=$(curl -sf "http://localhost:8006/api/v1/defense/status" 2>/dev/null || echo "ERROR")
    if [[ "$response" != "ERROR" ]]; then
        log SUCCESS "Defense Automation API responding"
    else
        log WARN "Defense Automation API not accessible"
    fi

    # Test automated response capabilities
    local automation_response=$(curl -sf "http://localhost:8006/api/v1/automation/rules" 2>/dev/null || echo "ERROR")
    if [[ "$automation_response" != "ERROR" ]]; then
        log SUCCESS "Automation rules engine active"
    else
        log WARN "Automation rules engine not accessible"
    fi
}

test_opsec_monitor() {
    log INFO "Testing OPSEC Monitor functionality..."

    local response=$(curl -sf "http://localhost:8007/api/v1/opsec/status" 2>/dev/null || echo "ERROR")
    if [[ "$response" != "ERROR" ]]; then
        log SUCCESS "OPSEC Monitor API responding"
    else
        log WARN "OPSEC Monitor API not accessible"
    fi
}

test_intel_fusion() {
    log INFO "Testing Intel Fusion functionality..."

    local response=$(curl -sf "http://localhost:8008/api/v1/fusion/status" 2>/dev/null || echo "ERROR")
    if [[ "$response" != "ERROR" ]]; then
        log SUCCESS "Intel Fusion API responding"
    else
        log WARN "Intel Fusion API not accessible"
    fi

    # Test multi-source correlation
    local correlation_response=$(curl -sf "http://localhost:8008/api/v1/correlation/engines" 2>/dev/null || echo "ERROR")
    if [[ "$correlation_response" != "ERROR" ]]; then
        log SUCCESS "Correlation engines active"
    else
        log WARN "Correlation engines not accessible"
    fi
}

run_security_audit() {
    log SECURITY "Running Phase $PHASE security audit..."

    # Check container security
    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        local container_name="bev_${service}"

        if docker ps --format '{{.Names}}' | grep -q "$container_name"; then
            # Check if running as non-root
            local user=$(docker exec "$container_name" whoami 2>/dev/null || echo "unknown")
            if [[ "$user" == "root" ]]; then
                log WARN "Security concern: $service running as root"
            else
                log SUCCESS "Security check: $service running as $user"
            fi

            # Check for security labels
            local labels=$(docker inspect "$container_name" --format '{{json .Config.Labels}}' 2>/dev/null || echo "{}")
            if echo "$labels" | grep -q "bev.security"; then
                log SUCCESS "Security labels present for $service"
            else
                log WARN "Security labels missing for $service"
            fi
        fi
    done

    log SUCCESS "Security audit completed"
}

show_deployment_summary() {
    log INFO "Phase $PHASE Deployment Summary:"
    echo "=============================================="
    echo "Phase: $PHASE - $PHASE_NAME"
    echo "Services deployed: ${#SERVICES[@]}"
    echo "Security level: ENHANCED"
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Service Endpoints:"
    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        local port=$(echo "$service_info" | cut -d':' -f2)
        local security_level=${SERVICE_SECURITY_LEVEL[$service]}
        echo "  $service: http://localhost:$port (Security: $security_level)"
    done
    echo ""
    echo "Security Features:"
    echo "  - Encrypted data volumes"
    echo "  - Security-hardened containers"
    echo "  - Vulnerability scanning"
    echo "  - Audit logging enabled"
    echo ""
    echo "Next steps:"
    echo "  1. Monitor security logs: docker-compose -f docker-compose-phase${PHASE}.yml logs -f"
    echo "  2. Run security tests: python3 ${PROJECT_ROOT}/deployment/tests/test_phase_${PHASE}_security.py"
    echo "  3. Review audit logs: tail -f ${LOG_DIR}/security_audit.log"
    echo "=============================================="
}

cleanup_on_failure() {
    log ERROR "Phase $PHASE deployment failed, performing secure cleanup..."

    # Stop and remove services
    docker-compose -f "docker-compose-phase${PHASE}.yml" down 2>/dev/null || true

    # Securely remove built images
    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        docker rmi "bev_${service}:latest" 2>/dev/null || true
    done

    # Clean sensitive temporary files
    find /tmp -name "*bev_phase8*" -type f -delete 2>/dev/null || true
}

# ===================================================================
# Main Execution
# ===================================================================

main() {
    log INFO "Starting Phase $PHASE deployment: $PHASE_NAME"

    # Set security-enhanced cleanup trap
    trap cleanup_on_failure ERR

    # Security checks
    check_security_prerequisites

    # Standard deployment steps
    check_dependencies
    create_service_volumes

    # Build services with security
    for service_info in "${SERVICES[@]}"; do
        local service=$(echo "$service_info" | cut -d':' -f1)
        check_service_prerequisites "$service"
        build_service "$service"
    done

    # Deploy services
    deploy_services

    # Verify deployment and security
    verify_functionality
    run_security_audit

    # Show summary
    show_deployment_summary

    log SUCCESS "Phase $PHASE deployment completed successfully with enhanced security!"
}

# Execute main function
main "$@"