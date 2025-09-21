#!/bin/bash
# BEV Frontend Integration - Pre-Deployment Validation Script
# Comprehensive prerequisite checking and environment validation
# Author: DevOps Automation Framework
# Version: 1.0.0

set -euo pipefail

# =====================================================
# Configuration and Constants
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/deployment"
LOG_FILE="${LOG_DIR}/pre-deployment-$(date +%Y%m%d_%H%M%S).log"
BACKUP_DIR="${PROJECT_ROOT}/backups/$(date +%Y%m%d_%H%M%S)"

# Required ports for conflict-free deployment
REQUIRED_PORTS=(3010 8443 8081 5432 6379 9200 3000 9090 3001 15672)
SAFE_FRONTEND_PORTS=(3010 8443 8081)
EXISTING_NETWORK="172.30.0.0/16"
NEW_NETWORK="172.31.0.0/16"

# Required environment variables
REQUIRED_ENV_VARS=(
    "POSTGRES_USER"
    "POSTGRES_PASSWORD"
    "REDIS_PASSWORD"
    "NEO4J_PASSWORD"
    "INFLUXDB_TOKEN"
)

# Required services for validation
REQUIRED_SERVICES=(
    "docker"
    "docker-compose"
    "curl"
    "netstat"
    "jq"
)

# =====================================================
# Logging and Utility Functions
# =====================================================

setup_logging() {
    mkdir -p "${LOG_DIR}"
    exec 1> >(tee -a "${LOG_FILE}")
    exec 2> >(tee -a "${LOG_FILE}" >&2)
    echo "=== BEV Pre-Deployment Validation Started at $(date) ===" | tee -a "${LOG_FILE}"
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "${LOG_FILE}"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*" | tee -a "${LOG_FILE}"
}

# =====================================================
# Validation Functions
# =====================================================

check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check available memory (minimum 8GB)
    local available_memory=$(free -g | awk '/^Mem:/{print $7}')
    if [ "${available_memory}" -lt 8 ]; then
        log_error "Insufficient memory. Available: ${available_memory}GB, Required: 8GB"
        return 1
    fi
    
    # Check disk space (minimum 50GB)
    local available_disk=$(df -BG "${PROJECT_ROOT}" | awk 'NR==2{print $4}' | sed 's/G//')
    if [ "${available_disk}" -lt 50 ]; then
        log_error "Insufficient disk space. Available: ${available_disk}GB, Required: 50GB"
        return 1
    fi
    
    log_success "System requirements check passed"
    return 0
}

check_required_services() {
    log_info "Checking required services..."
    
    for service in "${REQUIRED_SERVICES[@]}"; do
        if ! command -v "${service}" &> /dev/null; then
            log_error "Required service not found: ${service}"
            return 1
        else
            log_info "Service available: ${service}"
        fi
    done
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        return 1
    fi
    
    # Check Docker Compose version
    local compose_version=$(docker-compose version --short)
    log_info "Docker Compose version: ${compose_version}"
    
    log_success "Required services check passed"
    return 0
}

check_port_availability() {
    log_info "Checking port availability..."
    local conflicts_found=false
    
    for port in "${REQUIRED_PORTS[@]}"; do
        if netstat -tuln | grep -q ":${port} "; then
            log_warn "Port ${port} is already in use"
            local process=$(netstat -tulnp 2>/dev/null | grep ":${port} " | awk '{print $7}' | head -1)
            log_warn "Process using port ${port}: ${process}"
            
            # Check if it's one of our safe frontend ports
            if [[ " ${SAFE_FRONTEND_PORTS[@]} " =~ " ${port} " ]]; then
                log_error "Critical frontend port ${port} is occupied - deployment will fail"
                conflicts_found=true
            fi
        else
            log_info "Port ${port} is available"
        fi
    done
    
    if [ "${conflicts_found}" = true ]; then
        log_error "Critical port conflicts detected"
        return 1
    fi
    
    log_success "Port availability check passed"
    return 0
}

check_network_configuration() {
    log_info "Checking network configuration..."
    
    # Check if our proposed network conflicts with existing routes
    if ip route | grep -q "${NEW_NETWORK}"; then
        log_error "Network conflict: ${NEW_NETWORK} already exists in routing table"
        return 1
    fi
    
    # Check Docker networks
    if docker network ls --format "{{.Name}}" | grep -q "bev_frontend"; then
        log_warn "BEV frontend network already exists - will be recreated"
    fi
    
    # Validate existing BEV network
    if docker network ls --format "{{.Name}}" | grep -q "bev_osint"; then
        log_info "Existing BEV OSINT network found"
        local existing_subnet=$(docker network inspect bev_osint --format '{{range .IPAM.Config}}{{.Subnet}}{{end}}' 2>/dev/null || echo "")
        if [ -n "${existing_subnet}" ]; then
            log_info "Existing network subnet: ${existing_subnet}"
        fi
    fi
    
    log_success "Network configuration check passed"
    return 0
}

validate_environment_variables() {
    log_info "Validating environment variables..."
    
    # Check if .env file exists
    if [ ! -f "${PROJECT_ROOT}/.env" ]; then
        log_warn ".env file not found, will generate from template"
        cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
    fi
    
    # Source environment file
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
    
    # Check required variables
    local missing_vars=()
    for var in "${REQUIRED_ENV_VARS[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("${var}")
        else
            log_info "Environment variable present: ${var}"
        fi
    done
    
    # Generate missing API keys
    generate_missing_api_keys
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi
    
    log_success "Environment variables validation passed"
    return 0
}

generate_missing_api_keys() {
    log_info "Generating missing API keys..."
    
    local env_file="${PROJECT_ROOT}/.env"
    local backup_env="${env_file}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Backup existing .env
    cp "${env_file}" "${backup_env}"
    log_info "Backed up .env to ${backup_env}"
    
    # Generate BEV_API_KEY if missing
    if ! grep -q "^BEV_API_KEY=" "${env_file}" || grep -q "^BEV_API_KEY=$" "${env_file}"; then
        local bev_api_key=$(openssl rand -hex 32)
        echo "BEV_API_KEY=${bev_api_key}" >> "${env_file}"
        log_info "Generated BEV_API_KEY"
    fi
    
    # Generate MCP_API_KEY if missing
    if ! grep -q "^MCP_API_KEY=" "${env_file}" || grep -q "^MCP_API_KEY=$" "${env_file}"; then
        local mcp_api_key=$(openssl rand -hex 32)
        echo "MCP_API_KEY=${mcp_api_key}" >> "${env_file}"
        log_info "Generated MCP_API_KEY"
    fi
    
    # Generate FRONTEND_SESSION_SECRET if missing
    if ! grep -q "^FRONTEND_SESSION_SECRET=" "${env_file}" || grep -q "^FRONTEND_SESSION_SECRET=$" "${env_file}"; then
        local session_secret=$(openssl rand -base64 64 | tr -d '\n')
        echo "FRONTEND_SESSION_SECRET=${session_secret}" >> "${env_file}"
        log_info "Generated FRONTEND_SESSION_SECRET"
    fi
    
    # Generate WEBSOCKET_SECRET if missing
    if ! grep -q "^WEBSOCKET_SECRET=" "${env_file}" || grep -q "^WEBSOCKET_SECRET=$" "${env_file}"; then
        local ws_secret=$(openssl rand -hex 24)
        echo "WEBSOCKET_SECRET=${ws_secret}" >> "${env_file}"
        log_info "Generated WEBSOCKET_SECRET"
    fi
}

validate_existing_services() {
    log_info "Validating existing services..."
    
    # Check if core services are running
    local running_containers=$(docker ps --format "{{.Names}}")
    local core_services=("bev_postgres" "bev_redis" "bev_neo4j")
    
    for service in "${core_services[@]}"; do
        if echo "${running_containers}" | grep -q "${service}"; then
            log_info "Core service running: ${service}"
            
            # Perform health check
            case "${service}" in
                "bev_postgres")
                    if docker exec "${service}" pg_isready -U "${POSTGRES_USER}" &>/dev/null; then
                        log_success "PostgreSQL health check passed"
                    else
                        log_warn "PostgreSQL health check failed"
                    fi
                    ;;
                "bev_redis")
                    if docker exec "${service}" redis-cli ping | grep -q "PONG"; then
                        log_success "Redis health check passed"
                    else
                        log_warn "Redis health check failed"
                    fi
                    ;;
                "bev_neo4j")
                    log_info "Neo4j service detected (health check skipped)"
                    ;;
            esac
        else
            log_warn "Core service not running: ${service}"
        fi
    done
    
    log_success "Existing services validation completed"
    return 0
}

create_system_backup() {
    log_info "Creating system state backup..."
    
    mkdir -p "${BACKUP_DIR}"
    
    # Backup Docker Compose configurations
    cp -r "${PROJECT_ROOT}/docker-compose"* "${BACKUP_DIR}/" 2>/dev/null || true
    
    # Backup environment configuration
    cp "${PROJECT_ROOT}/.env" "${BACKUP_DIR}/" 2>/dev/null || true
    
    # Export Docker network configuration
    docker network ls --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}" > "${BACKUP_DIR}/docker_networks.txt"
    
    # Export running containers
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" > "${BACKUP_DIR}/running_containers.txt"
    
    # Create backup manifest
    cat > "${BACKUP_DIR}/backup_manifest.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "backup_type": "pre_deployment",
    "project_root": "${PROJECT_ROOT}",
    "docker_version": "$(docker version --format '{{.Server.Version}}')",
    "compose_version": "$(docker-compose version --short)",
    "system_info": {
        "hostname": "$(hostname)",
        "kernel": "$(uname -r)",
        "os": "$(lsb_release -d 2>/dev/null | cut -f2 || echo 'Unknown')"
    }
}
EOF
    
    log_success "System backup created at ${BACKUP_DIR}"
    echo "BACKUP_LOCATION=${BACKUP_DIR}" >> "${LOG_FILE}"
    return 0
}

# =====================================================
# Security Validation Functions
# =====================================================

validate_ssl_certificates() {
    log_info "Validating SSL certificate configuration..."
    
    local ssl_dir="${PROJECT_ROOT}/config/ssl"
    if [ ! -d "${ssl_dir}" ]; then
        log_warn "SSL directory not found, will be created during deployment"
        return 0
    fi
    
    # Check for existing certificates
    if [ -f "${ssl_dir}/bev-frontend.crt" ] && [ -f "${ssl_dir}/bev-frontend.key" ]; then
        log_info "Existing SSL certificates found"
        
        # Validate certificate expiry
        local cert_expiry=$(openssl x509 -in "${ssl_dir}/bev-frontend.crt" -noout -enddate 2>/dev/null | cut -d= -f2)
        if [ -n "${cert_expiry}" ]; then
            log_info "Certificate expires: ${cert_expiry}"
        fi
    else
        log_warn "SSL certificates not found, will be generated during deployment"
    fi
    
    log_success "SSL certificate validation completed"
    return 0
}

validate_security_settings() {
    log_info "Validating security settings..."
    
    # Check file permissions on sensitive files
    local sensitive_files=(".env" "config/ssl" "scripts")
    for file in "${sensitive_files[@]}"; do
        local full_path="${PROJECT_ROOT}/${file}"
        if [ -e "${full_path}" ]; then
            local permissions=$(stat -c "%a" "${full_path}")
            log_info "Permissions for ${file}: ${permissions}"
        fi
    done
    
    # Validate environment contains required security settings
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        if grep -q "JWT_SECRET" "${PROJECT_ROOT}/.env"; then
            log_info "JWT secret configuration found"
        else
            log_warn "JWT secret not configured"
        fi
        
        if grep -q "DATA_ENCRYPTION_KEY" "${PROJECT_ROOT}/.env"; then
            log_info "Data encryption key configuration found"
        else
            log_warn "Data encryption key not configured"
        fi
    fi
    
    log_success "Security settings validation completed"
    return 0
}

# =====================================================
# Performance and Resource Validation
# =====================================================

validate_resource_limits() {
    log_info "Validating resource limits..."
    
    # Check Docker daemon resource limits
    local docker_info=$(docker system info 2>/dev/null)
    
    # Extract memory information
    local total_memory=$(echo "${docker_info}" | grep "Total Memory" | awk '{print $3 $4}' || echo "Unknown")
    log_info "Total system memory: ${total_memory}"
    
    # Check CPU information
    local cpu_count=$(nproc)
    log_info "Available CPU cores: ${cpu_count}"
    
    # Validate Docker storage driver
    local storage_driver=$(echo "${docker_info}" | grep "Storage Driver" | awk '{print $3}' || echo "Unknown")
    log_info "Docker storage driver: ${storage_driver}"
    
    # Check for adequate resources for deployment
    if [ "${cpu_count}" -lt 4 ]; then
        log_warn "Low CPU count detected. Recommended: 4+ cores"
    fi
    
    log_success "Resource limits validation completed"
    return 0
}

# =====================================================
# Main Execution Flow
# =====================================================

main() {
    setup_logging
    
    log_info "Starting BEV Frontend Integration pre-deployment validation"
    log_info "Project root: ${PROJECT_ROOT}"
    log_info "Script directory: ${SCRIPT_DIR}"
    
    local validation_steps=(
        "check_system_requirements"
        "check_required_services"
        "check_port_availability"
        "check_network_configuration"
        "validate_environment_variables"
        "validate_existing_services"
        "validate_ssl_certificates"
        "validate_security_settings"
        "validate_resource_limits"
        "create_system_backup"
    )
    
    local failed_steps=()
    
    for step in "${validation_steps[@]}"; do
        log_info "Executing validation step: ${step}"
        if ! ${step}; then
            log_error "Validation step failed: ${step}"
            failed_steps+=("${step}")
        else
            log_success "Validation step passed: ${step}"
        fi
        echo "---"
    done
    
    # Summary
    echo "=============================================="
    log_info "Pre-deployment validation summary:"
    
    if [ ${#failed_steps[@]} -eq 0 ]; then
        log_success "All validation steps passed!"
        log_info "System is ready for BEV frontend deployment"
        
        # Write validation success marker
        echo "VALIDATION_STATUS=PASSED" > "${PROJECT_ROOT}/.deployment_validation"
        echo "VALIDATION_TIMESTAMP=$(date -Iseconds)" >> "${PROJECT_ROOT}/.deployment_validation"
        echo "BACKUP_LOCATION=${BACKUP_DIR}" >> "${PROJECT_ROOT}/.deployment_validation"
        
        echo "=============================================="
        echo "✅ PRE-DEPLOYMENT VALIDATION SUCCESSFUL"
        echo "   Log file: ${LOG_FILE}"
        echo "   Backup location: ${BACKUP_DIR}"
        echo "   Next step: Run deployment script"
        echo "=============================================="
        
        exit 0
    else
        log_error "Validation failed for ${#failed_steps[@]} step(s): ${failed_steps[*]}"
        
        # Write validation failure marker
        echo "VALIDATION_STATUS=FAILED" > "${PROJECT_ROOT}/.deployment_validation"
        echo "VALIDATION_TIMESTAMP=$(date -Iseconds)" >> "${PROJECT_ROOT}/.deployment_validation"
        echo "FAILED_STEPS=${failed_steps[*]}" >> "${PROJECT_ROOT}/.deployment_validation"
        
        echo "=============================================="
        echo "❌ PRE-DEPLOYMENT VALIDATION FAILED"
        echo "   Failed steps: ${failed_steps[*]}"
        echo "   Log file: ${LOG_FILE}"
        echo "   Please resolve issues before deployment"
        echo "=============================================="
        
        exit 1
    fi
}

# Trap for cleanup
trap 'log_error "Script interrupted"; exit 130' INT TERM

# Execute main function
main "$@"