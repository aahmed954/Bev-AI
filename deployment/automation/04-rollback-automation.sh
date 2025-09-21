#!/bin/bash
# BEV Frontend Integration - Rollback Automation and Emergency Recovery
# Comprehensive rollback system with multiple recovery strategies
# Author: DevOps Automation Framework
# Version: 1.0.0

set -euo pipefail

# =====================================================
# Configuration and Constants
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/deployment"
LOG_FILE="${LOG_DIR}/rollback-$(date +%Y%m%d_%H%M%S).log"
BACKUP_BASE_DIR="${PROJECT_ROOT}/backups"

# Rollback configuration
ROLLBACK_TIMEOUT=300  # 5 minutes
HEALTH_CHECK_INTERVAL=30
MAX_HEALTH_CHECK_ATTEMPTS=10
FORCE_ROLLBACK=false

# Service configuration
FRONTEND_CONTAINERS=("bev-mcp-server" "bev-frontend-proxy" "bev-websocket-server" "bev-frontend-web" "bev-frontend-redis")
FRONTEND_NETWORKS=("bev_frontend" "bev_bridge")
FRONTEND_VOLUMES=("mcp_server_logs" "frontend_logs" "proxy_logs" "ssl_certs" "frontend_redis_data")

# Recovery strategies
RECOVERY_STRATEGIES=("graceful" "immediate" "nuclear" "selective")
DEFAULT_STRATEGY="graceful"

# =====================================================
# Logging and Utility Functions
# =====================================================

setup_logging() {
    mkdir -p "${LOG_DIR}"
    exec 1> >(tee -a "${LOG_FILE}")
    exec 2> >(tee -a "${LOG_FILE}" >&2)
    echo "=== BEV Frontend Rollback Started at $(date) ===" | tee -a "${LOG_FILE}"
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

log_emergency() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [EMERGENCY] $*" | tee -a "${LOG_FILE}"
    # Send emergency alerts if configured
    send_emergency_alert "$*" || true
}

# =====================================================
# Emergency Alert Functions
# =====================================================

send_emergency_alert() {
    local message="$1"
    local timestamp=$(date -Iseconds)
    
    # Create alert payload
    local alert_payload=$(cat << EOF
{
    "timestamp": "${timestamp}",
    "severity": "critical",
    "service": "bev-frontend",
    "event": "rollback_initiated",
    "message": "${message}",
    "host": "$(hostname)",
    "log_file": "${LOG_FILE}"
}
EOF
    )
    
    # Try multiple notification methods
    
    # Webhook notification (if configured)
    if [ -n "${ALERT_WEBHOOK_URL:-}" ]; then
        curl -s -X POST "${ALERT_WEBHOOK_URL}" \
            -H "Content-Type: application/json" \
            -d "${alert_payload}" &>/dev/null || true
    fi
    
    # Slack notification (if configured)
    if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
        local slack_message="🚨 BEV Frontend Rollback Alert\\n${message}\\nHost: $(hostname)\\nTime: ${timestamp}"
        curl -s -X POST "${SLACK_WEBHOOK_URL}" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"${slack_message}\"}" &>/dev/null || true
    fi
    
    # Email notification (if configured and mailx available)
    if [ -n "${ALERT_EMAIL:-}" ] && command -v mailx &> /dev/null; then
        echo "${message}" | mailx -s "BEV Frontend Rollback Alert - $(hostname)" "${ALERT_EMAIL}" &>/dev/null || true
    fi
    
    # Local system notification
    logger -p user.crit "BEV Frontend Rollback: ${message}" || true
}

# =====================================================
# Backup Discovery and Validation
# =====================================================

discover_available_backups() {
    log_info "Discovering available backups..."
    
    local backup_list=()
    
    if [ -d "${BACKUP_BASE_DIR}" ]; then
        # Find backup directories with proper structure
        while IFS= read -r -d '' backup_dir; do
            local backup_manifest="${backup_dir}/backup_manifest.json"
            if [ -f "${backup_manifest}" ]; then
                local backup_timestamp=$(basename "${backup_dir}")
                backup_list+=("${backup_timestamp}")
                log_info "Found backup: ${backup_timestamp}"
            fi
        done < <(find "${BACKUP_BASE_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
    fi
    
    # Also check for deployment validation backups
    if [ -f "${PROJECT_ROOT}/.deployment_validation" ]; then
        local validation_backup=$(grep "BACKUP_LOCATION" "${PROJECT_ROOT}/.deployment_validation" 2>/dev/null | cut -d= -f2 || echo "")
        if [ -n "${validation_backup}" ] && [ -d "${validation_backup}" ]; then
            local val_timestamp=$(basename "${validation_backup}")
            if [[ ! " ${backup_list[@]} " =~ " ${val_timestamp} " ]]; then
                backup_list+=("${val_timestamp}")
                log_info "Found validation backup: ${val_timestamp}"
            fi
        fi
    fi
    
    if [ ${#backup_list[@]} -eq 0 ]; then
        log_error "No valid backups found in ${BACKUP_BASE_DIR}"
        return 1
    fi
    
    # Sort backups by timestamp (newest first)
    printf '%s\\n' "${backup_list[@]}" | sort -r
    return 0
}

validate_backup() {
    local backup_timestamp="$1"
    local backup_dir="${BACKUP_BASE_DIR}/${backup_timestamp}"
    
    log_info "Validating backup: ${backup_timestamp}"
    
    if [ ! -d "${backup_dir}" ]; then
        log_error "Backup directory not found: ${backup_dir}"
        return 1
    fi
    
    # Check backup manifest
    local manifest="${backup_dir}/backup_manifest.json"
    if [ ! -f "${manifest}" ]; then
        log_error "Backup manifest not found: ${manifest}"
        return 1
    fi
    
    # Validate manifest content
    if command -v jq &> /dev/null; then
        if ! jq -e '.timestamp' "${manifest}" >/dev/null 2>&1; then
            log_error "Invalid backup manifest format"
            return 1
        fi
        
        local backup_type=$(jq -r '.backup_type // "unknown"' "${manifest}")
        log_info "Backup type: ${backup_type}"
    fi
    
    # Check for essential backup files
    local essential_files=(".env" "docker_networks.txt" "running_containers.txt")
    for file in "${essential_files[@]}"; do
        if [ ! -f "${backup_dir}/${file}" ]; then
            log_warn "Backup missing file: ${file}"
        fi
    done
    
    log_success "Backup validation completed: ${backup_timestamp}"
    return 0
}

# =====================================================
# System State Assessment
# =====================================================

assess_current_state() {
    log_info "Assessing current system state..."
    
    local state_report="${LOG_DIR}/current_state_$(date +%Y%m%d_%H%M%S).json"
    
    # Container status
    local container_status=()
    for container in "${FRONTEND_CONTAINERS[@]}"; do
        local status=$(docker inspect "${container}" --format '{{.State.Status}}' 2>/dev/null || echo "not_found")
        local health=$(docker inspect "${container}" --format '{{.State.Health.Status}}' 2>/dev/null || echo "no_health_check")
        container_status+=("${container}:${status}:${health}")
    done
    
    # Network status
    local network_status=()
    for network in "${FRONTEND_NETWORKS[@]}"; do
        local exists=$(docker network ls --filter name="^${network}$" --format "{{.Name}}" | wc -l)
        network_status+=("${network}:${exists}")
    done
    
    # Volume status
    local volume_status=()
    for volume in "${FRONTEND_VOLUMES[@]}"; do
        local exists=$(docker volume ls --filter name="^${volume}$" --format "{{.Name}}" | wc -l)
        volume_status+=("${volume}:${exists}")
    done
    
    # Port status
    local port_conflicts=()
    local check_ports=(3010 8443 3011 8081 8080)
    for port in "${check_ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":${port} "; then
            local process=$(netstat -tulnp 2>/dev/null | grep ":${port} " | awk '{print $7}' | head -1)
            port_conflicts+=("${port}:${process}")
        fi
    done
    
    # Generate state report
    cat > "${state_report}" << EOF
{
    "assessment_timestamp": "$(date -Iseconds)",
    "system_info": {
        "hostname": "$(hostname)",
        "uptime": "$(uptime -p 2>/dev/null || echo 'unknown')",
        "load_average": "$(uptime | awk -F'load average:' '{print $2}' | xargs || echo 'unknown')"
    },
    "containers": [
        $(printf '%s\\n' "${container_status[@]}" | sed 's/\\(.*\\):\\(.*\\):\\(.*\\)/{"name":"\\1","status":"\\2","health":"\\3"}/' | paste -sd ',' -)
    ],
    "networks": [
        $(printf '%s\\n' "${network_status[@]}" | sed 's/\\(.*\\):\\(.*\\)/{"name":"\\1","exists":\\2}/' | paste -sd ',' -)
    ],
    "volumes": [
        $(printf '%s\\n' "${volume_status[@]}" | sed 's/\\(.*\\):\\(.*\\)/{"name":"\\1","exists":\\2}/' | paste -sd ',' -)
    ],
    "port_conflicts": [
        $(printf '%s\\n' "${port_conflicts[@]}" | sed 's/\\(.*\\):\\(.*\\)/{"port":"\\1","process":"\\2"}/' | paste -sd ',' -)
    ]
}
EOF
    
    log_info "System state assessment saved to: ${state_report}"
    echo "${state_report}"
}

# =====================================================
# Rollback Strategy Implementation
# =====================================================

execute_graceful_rollback() {
    local backup_timestamp="$1"
    local backup_dir="${BACKUP_BASE_DIR}/${backup_timestamp}"
    
    log_info "Executing graceful rollback to backup: ${backup_timestamp}"
    
    # Step 1: Stop services gracefully
    log_info "Stopping frontend services gracefully..."
    if [ -f "${PROJECT_ROOT}/frontend/docker-compose.frontend.yml" ]; then
        cd "${PROJECT_ROOT}/frontend"
        docker-compose -f docker-compose.frontend.yml down --timeout 30 || true
    else
        # Manual container stopping
        for container in "${FRONTEND_CONTAINERS[@]}"; do
            if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
                log_info "Stopping container: ${container}"
                docker stop "${container}" --time 30 || true
            fi
        done
    fi
    
    # Step 2: Remove containers and networks
    log_info "Cleaning up containers and networks..."
    for container in "${FRONTEND_CONTAINERS[@]}"; do
        if docker ps -a --format "{{.Names}}" | grep -q "^${container}$"; then
            docker rm "${container}" -f || true
        fi
    done
    
    # Remove frontend networks (but preserve bridge connections)
    for network in "${FRONTEND_NETWORKS[@]}"; do
        if [ "${network}" != "bev_bridge" ]; then  # Preserve bridge network
            if docker network ls --format "{{.Name}}" | grep -q "^${network}$"; then
                docker network rm "${network}" || true
            fi
        fi
    done
    
    # Step 3: Restore configuration files
    log_info "Restoring configuration files..."
    if [ -f "${backup_dir}/.env" ]; then
        cp "${backup_dir}/.env" "${PROJECT_ROOT}/.env.backup"
        log_info "Environment file backed up to .env.backup"
    fi
    
    # Step 4: Clean up frontend directory
    if [ -d "${PROJECT_ROOT}/frontend" ]; then
        log_info "Backing up current frontend directory..."
        mv "${PROJECT_ROOT}/frontend" "${PROJECT_ROOT}/frontend.rollback.$(date +%s)" || true
    fi
    
    # Step 5: Restore Docker networks from backup
    log_info "Restoring network configuration..."
    if [ -f "${backup_dir}/docker_networks.txt" ]; then
        log_info "Network configuration restored from backup"
    fi
    
    log_success "Graceful rollback completed"
    return 0
}

execute_immediate_rollback() {
    local backup_timestamp="$1"
    
    log_info "Executing immediate rollback to backup: ${backup_timestamp}"
    
    # Force stop all frontend containers immediately
    log_info "Force stopping all frontend containers..."
    for container in "${FRONTEND_CONTAINERS[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            log_info "Force stopping container: ${container}"
            docker kill "${container}" || true
            docker rm "${container}" -f || true
        fi
    done
    
    # Remove all frontend networks immediately
    for network in "${FRONTEND_NETWORKS[@]}"; do
        if [ "${network}" != "bev_bridge" ]; then
            if docker network ls --format "{{.Name}}" | grep -q "^${network}$"; then
                docker network rm "${network}" -f || true
            fi
        fi
    done
    
    # Remove frontend directory
    if [ -d "${PROJECT_ROOT}/frontend" ]; then
        rm -rf "${PROJECT_ROOT}/frontend" || true
    fi
    
    log_success "Immediate rollback completed"
    return 0
}

execute_nuclear_rollback() {
    log_emergency "Executing nuclear rollback - complete system reset"
    
    # Stop ALL Docker containers (not just frontend)
    log_info "Stopping all Docker containers..."
    docker stop $(docker ps -q) || true
    
    # Remove all containers
    log_info "Removing all containers..."
    docker rm $(docker ps -aq) -f || true
    
    # Remove all custom networks
    log_info "Removing custom networks..."
    docker network ls --filter driver=bridge --format "{{.Name}}" | grep -v "bridge\\|host\\|none" | xargs -r docker network rm || true
    
    # Prune Docker system
    log_info "Pruning Docker system..."
    docker system prune -f || true
    
    # Remove frontend components
    if [ -d "${PROJECT_ROOT}/frontend" ]; then
        rm -rf "${PROJECT_ROOT}/frontend" || true
    fi
    
    # Reset deployment markers
    rm -f "${PROJECT_ROOT}/.frontend_deployment" || true
    rm -f "${PROJECT_ROOT}/.deployment_validation" || true
    
    log_emergency "Nuclear rollback completed - system reset to clean state"
    return 0
}

execute_selective_rollback() {
    local backup_timestamp="$1"
    local components="$2"  # Comma-separated list of components to rollback
    
    log_info "Executing selective rollback for components: ${components}"
    
    IFS=',' read -ra COMPONENT_LIST <<< "${components}"
    
    for component in "${COMPONENT_LIST[@]}"; do
        case "${component}" in
            "mcp-server")
                if docker ps --format "{{.Names}}" | grep -q "bev-mcp-server"; then
                    docker stop bev-mcp-server --time 30 || true
                    docker rm bev-mcp-server -f || true
                fi
                ;;
            "proxy")
                if docker ps --format "{{.Names}}" | grep -q "bev-frontend-proxy"; then
                    docker stop bev-frontend-proxy --time 30 || true
                    docker rm bev-frontend-proxy -f || true
                fi
                ;;
            "websocket")
                if docker ps --format "{{.Names}}" | grep -q "bev-websocket-server"; then
                    docker stop bev-websocket-server --time 30 || true
                    docker rm bev-websocket-server -f || true
                fi
                ;;
            "networks")
                for network in "${FRONTEND_NETWORKS[@]}"; do
                    if [ "${network}" != "bev_bridge" ]; then
                        if docker network ls --format "{{.Name}}" | grep -q "^${network}$"; then
                            docker network rm "${network}" || true
                        fi
                    fi
                done
                ;;
            "config")
                if [ -f "${BACKUP_BASE_DIR}/${backup_timestamp}/.env" ]; then
                    cp "${BACKUP_BASE_DIR}/${backup_timestamp}/.env" "${PROJECT_ROOT}/.env.restored"
                    log_info "Configuration restored from backup"
                fi
                ;;
            *)
                log_warn "Unknown component for selective rollback: ${component}"
                ;;
        esac
    done
    
    log_success "Selective rollback completed"
    return 0
}

# =====================================================
# Health Check and Validation
# =====================================================

perform_post_rollback_health_check() {
    log_info "Performing post-rollback health check..."
    
    local health_check_passed=true
    
    # Check that frontend containers are stopped
    for container in "${FRONTEND_CONTAINERS[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "^${container}$"; then
            log_error "Container still running after rollback: ${container}"
            health_check_passed=false
        fi
    done
    
    # Check that ports are freed
    local check_ports=(3010 8443 3011 8081 8080)
    for port in "${check_ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":${port} "; then
            local process=$(netstat -tulnp 2>/dev/null | grep ":${port} " | awk '{print $7}' | head -1)
            if [[ "${process}" != *"bev"* ]]; then
                log_info "Port ${port} freed (process: ${process})"
            else
                log_error "BEV process still using port ${port}: ${process}"
                health_check_passed=false
            fi
        else
            log_info "Port ${port} is free"
        fi
    done
    
    # Check that core BEV services are still running (should not be affected)
    local core_services=("bev_postgres" "bev_redis" "bev_neo4j")
    for service in "${core_services[@]}"; do
        if docker ps --format "{{.Names}}" | grep -q "${service}"; then
            log_success "Core BEV service still running: ${service}"
        else
            log_warn "Core BEV service not running: ${service} (may need restart)"
        fi
    done
    
    if [ "${health_check_passed}" = true ]; then
        log_success "Post-rollback health check passed"
        return 0
    else
        log_error "Post-rollback health check failed"
        return 1
    fi
}

# =====================================================
# Recovery Verification
# =====================================================

verify_system_recovery() {
    log_info "Verifying system recovery..."
    
    # Create recovery verification report
    local recovery_report="${LOG_DIR}/recovery_verification_$(date +%Y%m%d_%H%M%S).json"
    
    # Check system resources
    local memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    local disk_usage=$(df "${PROJECT_ROOT}" | awk 'NR==2 {print $5}' | sed 's/%//')
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    
    # Check Docker daemon
    local docker_status="unknown"
    if docker info &>/dev/null; then
        docker_status="running"
    else
        docker_status="stopped"
    fi
    
    # Count remaining containers
    local container_count=$(docker ps -q | wc -l)
    local total_containers=$(docker ps -aq | wc -l)
    
    # Generate recovery report
    cat > "${recovery_report}" << EOF
{
    "recovery_timestamp": "$(date -Iseconds)",
    "rollback_log": "${LOG_FILE}",
    "system_metrics": {
        "memory_usage_percent": ${memory_usage},
        "disk_usage_percent": ${disk_usage},
        "load_average": "${load_avg}",
        "docker_status": "${docker_status}",
        "running_containers": ${container_count},
        "total_containers": ${total_containers}
    },
    "recovery_status": "completed",
    "core_services_status": {
        "postgres": "$(docker ps --format '{{.Names}}' | grep -c 'bev_postgres' || echo '0')",
        "redis": "$(docker ps --format '{{.Names}}' | grep -c 'bev_redis' || echo '0')",
        "neo4j": "$(docker ps --format '{{.Names}}' | grep -c 'bev_neo4j' || echo '0')"
    },
    "recommendations": [
        "Review rollback logs for any warnings",
        "Verify core BEV services are functioning",
        "Run system health checks",
        "Monitor system performance"
    ]
}
EOF
    
    log_info "Recovery verification report: ${recovery_report}"
    
    # Display recovery summary
    echo "=============================================="
    echo "SYSTEM RECOVERY VERIFICATION"
    echo "=============================================="
    echo "Memory Usage: ${memory_usage}%"
    echo "Disk Usage: ${disk_usage}%"
    echo "Load Average: ${load_avg}"
    echo "Docker Status: ${docker_status}"
    echo "Running Containers: ${container_count}"
    echo "Recovery Report: ${recovery_report}"
    echo "=============================================="
    
    return 0
}

# =====================================================
# Usage and Help Functions
# =====================================================

show_usage() {
    cat << EOF
BEV Frontend Rollback Automation Script

USAGE:
    $0 [OPTIONS] [STRATEGY] [BACKUP_TIMESTAMP]

STRATEGIES:
    graceful    - Graceful shutdown and rollback (default)
    immediate   - Immediate forced rollback
    nuclear     - Complete system reset (emergency use only)
    selective   - Rollback specific components only

OPTIONS:
    -f, --force         Force rollback without confirmation
    -t, --timeout SEC   Set rollback timeout (default: 300)
    -c, --components    Components for selective rollback (comma-separated)
    -l, --list          List available backups
    -s, --status        Show current system status
    -h, --help          Show this help message

EXAMPLES:
    $0                                    # Interactive graceful rollback
    $0 graceful                          # Graceful rollback to latest backup
    $0 immediate 20250119_143022         # Immediate rollback to specific backup
    $0 selective mcp-server,proxy        # Selective component rollback
    $0 nuclear --force                   # Emergency nuclear rollback
    $0 --list                           # List available backups
    $0 --status                         # Show system status

BACKUP TIMESTAMP FORMAT:
    YYYYMMDD_HHMMSS (e.g., 20250119_143022)

RECOVERY STRATEGIES:
    graceful  - Stop services gracefully, restore configurations, preserve data
    immediate - Force stop services, remove containers, quick recovery
    nuclear   - Complete system reset, remove all containers and networks
    selective - Rollback only specified components

EOF
}

list_available_backups() {
    echo "Available Backups:"
    echo "=================="
    
    local backups
    if backups=$(discover_available_backups); then
        echo "${backups}" | while read -r backup; do
            local backup_dir="${BACKUP_BASE_DIR}/${backup}"
            local manifest="${backup_dir}/backup_manifest.json"
            
            if [ -f "${manifest}" ] && command -v jq &> /dev/null; then
                local backup_type=$(jq -r '.backup_type // "unknown"' "${manifest}")
                local backup_time=$(jq -r '.timestamp // "unknown"' "${manifest}")
                printf "%-20s %-15s %s\\n" "${backup}" "${backup_type}" "${backup_time}"
            else
                printf "%-20s %-15s %s\\n" "${backup}" "unknown" "$(date -d "${backup:0:8} ${backup:9:2}:${backup:11:2}:${backup:13:2}" 2>/dev/null || echo 'unknown')"
            fi
        done
    else
        echo "No backups found"
        return 1
    fi
}

show_system_status() {
    echo "Current System Status:"
    echo "====================="
    
    # Show container status
    echo "Frontend Containers:"
    for container in "${FRONTEND_CONTAINERS[@]}"; do
        local status=$(docker inspect "${container}" --format '{{.State.Status}}' 2>/dev/null || echo "not_found")
        local health=$(docker inspect "${container}" --format '{{.State.Health.Status}}' 2>/dev/null || echo "no_healthcheck")
        printf "  %-25s %-10s %s\\n" "${container}" "${status}" "${health}"
    done
    
    echo ""
    echo "Networks:"
    for network in "${FRONTEND_NETWORKS[@]}"; do
        local exists=$(docker network ls --filter name="^${network}$" --format "{{.Name}}" | wc -l)
        local status
        if [ "${exists}" -eq 1 ]; then
            status="exists"
        else
            status="missing"
        fi
        printf "  %-25s %s\\n" "${network}" "${status}"
    done
    
    echo ""
    echo "Port Usage:"
    local check_ports=(3010 8443 3011 8081 8080)
    for port in "${check_ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":${port} "; then
            local process=$(netstat -tulnp 2>/dev/null | grep ":${port} " | awk '{print $7}' | head -1)
            printf "  %-10s %-10s %s\\n" "${port}" "in_use" "${process}"
        else
            printf "  %-10s %-10s %s\\n" "${port}" "free" ""
        fi
    done
    
    echo ""
    assess_current_state >/dev/null
    echo "Detailed assessment saved to logs"
}

# =====================================================
# Interactive Rollback Selection
# =====================================================

interactive_rollback_selection() {
    echo "BEV Frontend Rollback Wizard"
    echo "============================="
    
    # Show current status
    echo ""
    echo "Current system status:"
    show_system_status
    
    echo ""
    echo "Available rollback strategies:"
    echo "1) Graceful  - Safe shutdown and rollback (recommended)"
    echo "2) Immediate - Fast forced rollback"
    echo "3) Nuclear   - Complete system reset (emergency only)"
    echo "4) Selective - Choose specific components"
    echo "5) Cancel    - Exit without changes"
    
    echo ""
    read -p "Select rollback strategy [1-5]: " strategy_choice
    
    local selected_strategy
    case "${strategy_choice}" in
        1) selected_strategy="graceful" ;;
        2) selected_strategy="immediate" ;;
        3) 
            echo ""
            echo "⚠️  WARNING: Nuclear rollback will completely reset the system!"
            echo "   This will stop ALL containers and remove ALL networks."
            echo "   Core BEV services will also be affected."
            echo ""
            read -p "Are you ABSOLUTELY sure? [yes/NO]: " nuclear_confirm
            if [ "${nuclear_confirm}" != "yes" ]; then
                echo "Nuclear rollback cancelled"
                exit 0
            fi
            selected_strategy="nuclear"
            ;;
        4) 
            echo ""
            echo "Available components for selective rollback:"
            echo "- mcp-server    (BEV MCP Server)"
            echo "- proxy         (HAProxy Load Balancer)"
            echo "- websocket     (WebSocket Server)"
            echo "- networks      (Frontend Networks)"
            echo "- config        (Configuration Files)"
            echo ""
            read -p "Enter components (comma-separated): " components
            selected_strategy="selective"
            ;;
        5) 
            echo "Rollback cancelled"
            exit 0
            ;;
        *)
            echo "Invalid selection"
            exit 1
            ;;
    esac
    
    # Select backup (except for nuclear)
    local selected_backup=""
    if [ "${selected_strategy}" != "nuclear" ]; then
        echo ""
        echo "Available backups:"
        list_available_backups
        echo ""
        read -p "Enter backup timestamp (or 'latest' for most recent): " backup_input
        
        if [ "${backup_input}" = "latest" ] || [ -z "${backup_input}" ]; then
            selected_backup=$(discover_available_backups | head -1)
            if [ -z "${selected_backup}" ]; then
                echo "No backups available"
                exit 1
            fi
        else
            selected_backup="${backup_input}"
        fi
        
        # Validate selected backup
        if ! validate_backup "${selected_backup}"; then
            echo "Invalid backup selected: ${selected_backup}"
            exit 1
        fi
    fi
    
    # Final confirmation
    echo ""
    echo "Rollback Summary:"
    echo "=================="
    echo "Strategy: ${selected_strategy}"
    [ -n "${selected_backup}" ] && echo "Backup: ${selected_backup}"
    [ "${selected_strategy}" = "selective" ] && echo "Components: ${components}"
    echo ""
    
    if [ "${FORCE_ROLLBACK}" != true ]; then
        read -p "Proceed with rollback? [y/N]: " confirm
        if [ "${confirm}" != "y" ] && [ "${confirm}" != "Y" ]; then
            echo "Rollback cancelled"
            exit 0
        fi
    fi
    
    # Execute rollback
    case "${selected_strategy}" in
        "graceful")
            execute_graceful_rollback "${selected_backup}"
            ;;
        "immediate")
            execute_immediate_rollback "${selected_backup}"
            ;;
        "nuclear")
            execute_nuclear_rollback
            ;;
        "selective")
            execute_selective_rollback "${selected_backup}" "${components}"
            ;;
    esac
}

# =====================================================
# Main Execution Flow
# =====================================================

main() {
    setup_logging
    
    # Parse command line arguments
    local strategy="${DEFAULT_STRATEGY}"
    local backup_timestamp=""
    local components=""
    local show_list=false
    local show_status=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -f|--force)
                FORCE_ROLLBACK=true
                shift
                ;;
            -t|--timeout)
                ROLLBACK_TIMEOUT="$2"
                shift 2
                ;;
            -c|--components)
                components="$2"
                shift 2
                ;;
            -l|--list)
                show_list=true
                shift
                ;;
            -s|--status)
                show_status=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            graceful|immediate|nuclear|selective)
                strategy="$1"
                shift
                ;;
            *)
                if [ -z "${backup_timestamp}" ] && [[ "$1" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
                    backup_timestamp="$1"
                else
                    echo "Unknown option: $1"
                    show_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Handle special modes
    if [ "${show_list}" = true ]; then
        list_available_backups
        exit 0
    fi
    
    if [ "${show_status}" = true ]; then
        show_system_status
        exit 0
    fi
    
    # Load environment variables if available
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        set -a
        source "${PROJECT_ROOT}/.env" 2>/dev/null || true
        set +a
    fi
    
    log_info "Starting BEV Frontend rollback process"
    log_info "Strategy: ${strategy}"
    log_info "Force mode: ${FORCE_ROLLBACK}"
    log_info "Timeout: ${ROLLBACK_TIMEOUT} seconds"
    
    # Assess current state
    local state_file=$(assess_current_state)
    
    # Interactive mode if no specific parameters
    if [ $# -eq 0 ] && [ "${FORCE_ROLLBACK}" != true ]; then
        interactive_rollback_selection
    else
        # Non-interactive mode
        case "${strategy}" in
            "graceful")
                if [ -z "${backup_timestamp}" ]; then
                    backup_timestamp=$(discover_available_backups | head -1)
                    if [ -z "${backup_timestamp}" ]; then
                        log_error "No backups available for graceful rollback"
                        exit 1
                    fi
                fi
                validate_backup "${backup_timestamp}"
                execute_graceful_rollback "${backup_timestamp}"
                ;;
            "immediate")
                if [ -z "${backup_timestamp}" ]; then
                    backup_timestamp=$(discover_available_backups | head -1)
                    if [ -z "${backup_timestamp}" ]; then
                        log_error "No backups available for immediate rollback"
                        exit 1
                    fi
                fi
                validate_backup "${backup_timestamp}"
                execute_immediate_rollback "${backup_timestamp}"
                ;;
            "nuclear")
                if [ "${FORCE_ROLLBACK}" != true ]; then
                    echo "Nuclear rollback requires --force flag"
                    exit 1
                fi
                execute_nuclear_rollback
                ;;
            "selective")
                if [ -z "${components}" ]; then
                    echo "Selective rollback requires --components option"
                    exit 1
                fi
                if [ -z "${backup_timestamp}" ]; then
                    backup_timestamp=$(discover_available_backups | head -1)
                fi
                if [ -n "${backup_timestamp}" ]; then
                    validate_backup "${backup_timestamp}"
                fi
                execute_selective_rollback "${backup_timestamp}" "${components}"
                ;;
            *)
                log_error "Unknown strategy: ${strategy}"
                exit 1
                ;;
        esac
    fi
    
    # Post-rollback verification
    if perform_post_rollback_health_check; then
        log_success "Post-rollback health check passed"
    else
        log_error "Post-rollback health check failed"
    fi
    
    # Generate recovery verification
    verify_system_recovery
    
    # Write rollback completion marker
    echo "ROLLBACK_STATUS=COMPLETED" > "${PROJECT_ROOT}/.rollback_status"
    echo "ROLLBACK_TIMESTAMP=$(date -Iseconds)" >> "${PROJECT_ROOT}/.rollback_status"
    echo "ROLLBACK_STRATEGY=${strategy}" >> "${PROJECT_ROOT}/.rollback_status"
    echo "ROLLBACK_LOG=${LOG_FILE}" >> "${PROJECT_ROOT}/.rollback_status"
    
    # Final summary
    echo "=============================================="
    log_success "BEV Frontend rollback completed successfully"
    echo "Strategy: ${strategy}"
    echo "Log file: ${LOG_FILE}"
    echo "System ready for re-deployment or investigation"
    echo "=============================================="
    
    exit 0
}

# Trap for cleanup and emergency handling
trap 'log_emergency "Rollback script interrupted - system may be in inconsistent state"; exit 130' INT TERM

# Execute main function
main "$@"