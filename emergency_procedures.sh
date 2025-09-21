#!/bin/bash
# BEV OSINT Framework - Emergency Procedures and Recovery
# Complete emergency management for THANOS + ORACLE1 deployment
# Generated: $(date)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EMERGENCY_LOG="${SCRIPT_DIR}/emergency_$(date +%Y%m%d_%H%M%S).log"
BACKUP_DIR="${SCRIPT_DIR}/backups/$(date +%Y%m%d_%H%M%S)"
PROCEDURE="${1:-help}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$EMERGENCY_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$EMERGENCY_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$EMERGENCY_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$EMERGENCY_LOG"
}

log_emergency() {
    echo -e "${BOLD}${RED}[EMERGENCY]${NC} $1" | tee -a "$EMERGENCY_LOG"
}

log_header() {
    echo -e "${BOLD}${PURPLE}$1${NC}" | tee -a "$EMERGENCY_LOG"
    echo "============================================" | tee -a "$EMERGENCY_LOG"
}

# Emergency stop - immediate shutdown
emergency_stop() {
    log_header "🚨 EMERGENCY STOP - IMMEDIATE SHUTDOWN"

    log_emergency "Initiating emergency shutdown of all BEV services..."

    # Stop all BEV containers immediately
    log_info "Stopping all BEV containers..."
    if docker ps --format "{{.Names}}" | grep -E "(bev_|thanos|oracle1)" | xargs -r docker stop -t 5; then
        log_success "All BEV containers stopped"
    else
        log_warning "Some containers may not have stopped cleanly"
    fi

    # Stop THANOS unified services
    log_info "Stopping THANOS unified services..."
    if docker-compose -f docker-compose-thanos-unified.yml --env-file .env.thanos.complete down -t 10 2>/dev/null; then
        log_success "THANOS services stopped"
    else
        log_warning "THANOS services may not have stopped cleanly"
    fi

    # Stop ORACLE1 unified services
    log_info "Stopping ORACLE1 unified services..."
    if docker-compose -f docker-compose-oracle1-unified.yml --env-file .env.oracle1.complete down -t 10 2>/dev/null; then
        log_success "ORACLE1 services stopped"
    else
        log_warning "ORACLE1 services may not have stopped cleanly"
    fi

    # Force kill any remaining containers
    log_info "Force stopping any remaining BEV containers..."
    if docker ps -a --format "{{.Names}}" | grep -E "(bev_|thanos|oracle1)" | xargs -r docker kill 2>/dev/null; then
        log_info "Forced stop completed"
    else
        log_info "No containers required forced stop"
    fi

    # Remove stopped containers
    log_info "Removing stopped BEV containers..."
    if docker ps -a --format "{{.Names}}" | grep -E "(bev_|thanos|oracle1)" | xargs -r docker rm -f 2>/dev/null; then
        log_success "Stopped containers removed"
    else
        log_info "No containers to remove"
    fi

    log_success "🔥 EMERGENCY STOP COMPLETED"
    echo "System Status: ALL BEV SERVICES STOPPED"
    echo "Recovery: Run './emergency_procedures.sh recover' to restart services"
}

# Backup critical data
backup_data() {
    log_header "💾 BACKING UP CRITICAL DATA"

    # Create backup directory
    mkdir -p "$BACKUP_DIR"
    log_info "Backup directory: $BACKUP_DIR"

    # Backup Docker volumes
    log_info "Backing up Docker volumes..."
    local volumes=(
        "postgres_data"
        "postgres_test_data"
        "neo4j_data"
        "redis_data"
        "redis_test_data"
        "vault_data"
        "prometheus_data"
        "grafana_data"
    )

    for volume in "${volumes[@]}"; do
        if docker volume ls | grep -q "$volume"; then
            log_info "Backing up volume: $volume"
            docker run --rm -v "${volume}:/data" -v "${BACKUP_DIR}:/backup" alpine:latest tar czf "/backup/${volume}.tar.gz" -C /data . 2>/dev/null || log_warning "Failed to backup $volume"
        fi
    done

    # Backup configuration files
    log_info "Backing up configuration files..."
    local config_files=(
        ".env"
        ".env.thanos.complete"
        ".env.oracle1.complete"
        "docker-compose-thanos-unified.yml"
        "docker-compose-oracle1-unified.yml"
        "vault-init.json"
    )

    for config_file in "${config_files[@]}"; do
        if [[ -f "$SCRIPT_DIR/$config_file" ]]; then
            cp "$SCRIPT_DIR/$config_file" "$BACKUP_DIR/" || log_warning "Failed to backup $config_file"
        fi
    done

    # Backup deployment logs
    log_info "Backing up deployment logs..."
    find "$SCRIPT_DIR" -name "deployment_*.log" -o -name "validation_*.log" -o -name "emergency_*.log" 2>/dev/null | while read -r log_file; do
        cp "$log_file" "$BACKUP_DIR/" 2>/dev/null || true
    done

    # Create backup manifest
    cat > "$BACKUP_DIR/backup_manifest.txt" << EOF
BEV OSINT Framework - Emergency Backup
Created: $(date)
Backup Directory: $BACKUP_DIR

Docker Volumes Backed Up:
$(ls -la "$BACKUP_DIR"/*.tar.gz 2>/dev/null || echo "No volume backups found")

Configuration Files:
$(ls -la "$BACKUP_DIR"/*.yml "$BACKUP_DIR"/.env* "$BACKUP_DIR"/*.json 2>/dev/null || echo "No config files found")

System Status at Backup:
$(docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(bev_|thanos|oracle1)" || echo "No BEV containers running")
EOF

    log_success "Backup completed: $BACKUP_DIR"
    echo "Backup manifest: $BACKUP_DIR/backup_manifest.txt"
}

# System recovery
recover_system() {
    log_header "🔄 SYSTEM RECOVERY"

    log_info "Starting BEV system recovery..."

    # Check if backup exists
    local latest_backup=$(find "$SCRIPT_DIR/backups" -type d -name "20*" 2>/dev/null | sort | tail -1)
    if [[ -n "$latest_backup" && -d "$latest_backup" ]]; then
        log_info "Latest backup found: $latest_backup"
    else
        log_warning "No recent backup found - proceeding with clean recovery"
    fi

    # Clean up any remaining containers/networks
    log_info "Cleaning up existing resources..."
    emergency_stop >/dev/null 2>&1 || true

    # Restart deployment
    log_info "Attempting system restart..."
    if [[ -x "$SCRIPT_DIR/deploy_bev_complete.sh" ]]; then
        log_info "Starting full deployment recovery..."
        if "$SCRIPT_DIR/deploy_bev_complete.sh" full 2>&1 | tee -a "$EMERGENCY_LOG"; then
            log_success "System recovery completed successfully"
            return 0
        else
            log_error "System recovery failed"
            return 1
        fi
    else
        log_error "Main deployment script not found"
        return 1
    fi
}

# Restore from backup
restore_from_backup() {
    local backup_path="${2:-}"
    log_header "📦 RESTORING FROM BACKUP"

    if [[ -z "$backup_path" ]]; then
        # Find latest backup
        backup_path=$(find "$SCRIPT_DIR/backups" -type d -name "20*" 2>/dev/null | sort | tail -1)
    fi

    if [[ -z "$backup_path" || ! -d "$backup_path" ]]; then
        log_error "No backup directory specified or found"
        log_info "Usage: $0 restore [backup_directory]"
        log_info "Available backups:"
        find "$SCRIPT_DIR/backups" -type d -name "20*" 2>/dev/null | sort || echo "No backups found"
        return 1
    fi

    log_info "Restoring from backup: $backup_path"

    # Stop all services first
    emergency_stop >/dev/null 2>&1 || true

    # Restore configuration files
    log_info "Restoring configuration files..."
    local config_files=(
        ".env"
        ".env.thanos.complete"
        ".env.oracle1.complete"
        "docker-compose-thanos-unified.yml"
        "docker-compose-oracle1-unified.yml"
        "vault-init.json"
    )

    for config_file in "${config_files[@]}"; do
        if [[ -f "$backup_path/$config_file" ]]; then
            cp "$backup_path/$config_file" "$SCRIPT_DIR/" && log_success "Restored $config_file"
        fi
    done

    # Restore Docker volumes
    log_info "Restoring Docker volumes..."
    for volume_backup in "$backup_path"/*.tar.gz; do
        if [[ -f "$volume_backup" ]]; then
            local volume_name=$(basename "$volume_backup" .tar.gz)
            log_info "Restoring volume: $volume_name"

            # Remove existing volume
            docker volume rm "$volume_name" 2>/dev/null || true

            # Create and restore volume
            docker volume create "$volume_name"
            docker run --rm -v "${volume_name}:/data" -v "${backup_path}:/backup" alpine:latest tar xzf "/backup/${volume_name}.tar.gz" -C /data 2>/dev/null && log_success "Restored volume: $volume_name" || log_warning "Failed to restore volume: $volume_name"
        fi
    done

    log_success "Restore completed from: $backup_path"
    log_info "Next step: Run recovery to restart services"
}

# Health check
health_check() {
    log_header "🏥 EMERGENCY HEALTH CHECK"

    # Quick system assessment
    log_info "System Status Assessment:"

    # Docker daemon
    if docker info &>/dev/null; then
        log_success "Docker daemon is responsive"
    else
        log_error "Docker daemon is not responsive"
    fi

    # Running containers
    local running_containers=$(docker ps --format "{{.Names}}" | grep -E "(bev_|thanos|oracle1)" | wc -l)
    log_info "BEV containers running: $running_containers"

    # System resources
    local memory_usage=$(free | awk '/^Mem:/{printf "%.1f", $3/$2 * 100}')
    local disk_usage=$(df "$SCRIPT_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')

    log_info "Memory usage: ${memory_usage}%"
    log_info "Disk usage: ${disk_usage}%"
    log_info "Load average: ${load_avg}"

    # Network connectivity
    if ping -c 1 localhost &>/dev/null; then
        log_success "Network connectivity is functional"
    else
        log_warning "Network connectivity issues detected"
    fi

    # Key service ports
    local critical_ports=("5432" "6379" "7474" "7687" "8200" "9090" "3000")
    local accessible_ports=0

    for port in "${critical_ports[@]}"; do
        if timeout 2 bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
            ((accessible_ports++))
        fi
    done

    log_info "Critical services accessible: $accessible_ports/${#critical_ports[@]}"

    # Overall assessment
    if (( running_containers > 10 && accessible_ports > 5 )); then
        log_success "🟢 System appears to be healthy"
    elif (( running_containers > 5 && accessible_ports > 3 )); then
        log_warning "🟡 System is partially functional"
    else
        log_error "🔴 System has significant issues"
    fi
}

# Network reset
network_reset() {
    log_header "🌐 NETWORK RESET"

    log_info "Resetting Docker networks..."

    # Remove BEV networks
    docker network ls --format "{{.Name}}" | grep -E "(bev_|thanos|oracle1)" | xargs -r docker network rm 2>/dev/null || true

    # Prune unused networks
    docker network prune -f >/dev/null 2>&1

    # Restart Docker daemon if needed
    if ! docker info &>/dev/null; then
        log_warning "Docker daemon seems unresponsive - manual restart may be required"
        log_info "Try: sudo systemctl restart docker"
    else
        log_success "Network reset completed"
    fi
}

# Show usage
show_usage() {
    echo "BEV OSINT Framework - Emergency Procedures"
    echo ""
    echo "Usage: $0 [PROCEDURE]"
    echo ""
    echo "Emergency Procedures:"
    echo "  stop        - Emergency stop all BEV services"
    echo "  backup      - Backup critical data and configurations"
    echo "  recover     - Attempt full system recovery"
    echo "  restore     - Restore from backup [backup_directory]"
    echo "  health      - Emergency health check"
    echo "  network     - Reset Docker networks"
    echo "  help        - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 stop                           # Emergency stop"
    echo "  $0 backup                         # Create backup"
    echo "  $0 restore                        # Restore from latest backup"
    echo "  $0 restore ./backups/20241221_123456  # Restore from specific backup"
    echo ""
    echo "Emergency Contact:"
    echo "  Check deployment logs for detailed error information"
    echo "  Run health check to assess system status"
    echo "  Create backup before attempting recovery"
    echo ""
}

# Main procedure handler
main() {
    echo "Emergency procedure started at $(date)" > "$EMERGENCY_LOG"

    case "$PROCEDURE" in
        "stop")
            emergency_stop
            ;;
        "backup")
            backup_data
            ;;
        "recover")
            backup_data
            recover_system
            ;;
        "restore")
            restore_from_backup "$@"
            ;;
        "health")
            health_check
            ;;
        "network")
            network_reset
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown procedure: $PROCEDURE"
            show_usage
            exit 1
            ;;
    esac
}

# Handle script termination
trap 'log_warning "Emergency procedure interrupted"; exit 1' INT TERM

# Run main procedure
main "$@"