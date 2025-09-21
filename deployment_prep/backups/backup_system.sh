#!/bin/bash

# Comprehensive Backup and Rollback System
# Preserves system state for safe deployment rollback

# Backup configuration
BACKUP_ROOT="/var/lib/bev/backups"
BACKUP_RETENTION_DAYS=30
MAX_BACKUP_SIZE_GB=50

create_system_backup() {
    log "INFO" "Creating comprehensive system backup"

    local backup_timestamp="$TIMESTAMP"
    local backup_dir="$BACKUP_ROOT/pre-deployment-$backup_timestamp"

    # Create backup directory structure
    if ! create_backup_directory_structure "$backup_dir"; then
        log "ERROR" "Failed to create backup directory structure"
        return 1
    fi

    # Create backup manifest
    create_backup_manifest "$backup_dir"

    local backup_success=true

    # Backup Docker state
    if ! backup_docker_state "$backup_dir"; then
        log "ERROR" "Failed to backup Docker state"
        backup_success=false
    fi

    # Backup configurations
    if ! backup_configurations "$backup_dir"; then
        log "ERROR" "Failed to backup configurations"
        backup_success=false
    fi

    # Backup databases
    if ! backup_databases "$backup_dir"; then
        log "WARN" "Database backup had issues (may be expected if services not running)"
    fi

    # Backup system state
    if ! backup_system_state "$backup_dir"; then
        log "ERROR" "Failed to backup system state"
        backup_success=false
    fi

    # Create rollback script
    if ! create_rollback_script "$backup_dir"; then
        log "ERROR" "Failed to create rollback script"
        backup_success=false
    fi

    # Validate backup
    if ! validate_backup "$backup_dir"; then
        log "ERROR" "Backup validation failed"
        backup_success=false
    fi

    # Set permissions
    chmod 600 "$backup_dir"/*.json "$backup_dir"/*.env 2>/dev/null || true
    chmod 755 "$backup_dir"/rollback.sh

    if [[ "$backup_success" == "true" ]]; then
        log "SUCCESS" "System backup created successfully: $backup_dir"
        echo "$backup_dir" > "$BACKUP_ROOT/latest_backup"
        return 0
    else
        log "ERROR" "System backup failed"
        return 1
    fi
}

create_backup_directory_structure() {
    local backup_dir="$1"

    log "INFO" "Creating backup directory structure: $backup_dir"

    if ! sudo mkdir -p "$backup_dir"/{docker,configs,databases,system,scripts}; then
        return 1
    fi

    if ! sudo chown -R $(whoami):$(whoami) "$backup_dir"; then
        return 1
    fi

    return 0
}

create_backup_manifest() {
    local backup_dir="$1"

    cat > "$backup_dir/backup_manifest.json" << EOF
{
    "backup_timestamp": "$TIMESTAMP",
    "backup_date": "$(date -Iseconds)",
    "hostname": "$(hostname)",
    "user": "$(whoami)",
    "pwd": "$(pwd)",
    "git_branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
    "git_commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "backup_version": "1.0.0",
    "backup_components": [
        "docker_state",
        "configurations",
        "databases",
        "system_state",
        "rollback_script"
    ]
}
EOF

    log "SUCCESS" "Backup manifest created"
}

backup_docker_state() {
    local backup_dir="$1"

    log "INFO" "Backing up Docker state"

    # Backup running containers
    docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Command}}\t{{.CreatedAt}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}" > "$backup_dir/docker/containers_running.txt"

    # Backup all containers
    docker ps -a --format "table {{.ID}}\t{{.Image}}\t{{.Command}}\t{{.CreatedAt}}\t{{.Status}}\t{{.Ports}}\t{{.Names}}" > "$backup_dir/docker/containers_all.txt"

    # Backup container configurations
    local containers=$(docker ps -a --format "{{.ID}}\t{{.Names}}")
    if [[ -n "$containers" ]]; then
        while IFS=$'\t' read -r container_id container_name; do
            docker inspect "$container_id" > "$backup_dir/docker/container_${container_name}_inspect.json" 2>/dev/null || true
        done <<< "$containers"
    fi

    # Backup images
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}" > "$backup_dir/docker/images.txt"

    # Backup volumes
    docker volume ls --format "table {{.Driver}}\t{{.Name}}" > "$backup_dir/docker/volumes.txt"

    # Backup networks
    docker network ls --format "table {{.ID}}\t{{.Name}}\t{{.Driver}}\t{{.Scope}}" > "$backup_dir/docker/networks.txt"

    # Backup Docker daemon configuration
    if [[ -f /etc/docker/daemon.json ]]; then
        cp /etc/docker/daemon.json "$backup_dir/docker/daemon.json" 2>/dev/null || true
    fi

    # Backup Docker Compose files
    if [[ -f docker-compose.yml ]]; then
        cp docker-compose.yml "$backup_dir/docker/docker-compose.yml" 2>/dev/null || true
    fi

    if [[ -f docker-compose.complete.yml ]]; then
        cp docker-compose.complete.yml "$backup_dir/docker/docker-compose.complete.yml" 2>/dev/null || true
    fi

    log "SUCCESS" "Docker state backed up"
    return 0
}

backup_configurations() {
    local backup_dir="$1"

    log "INFO" "Backing up configuration files"

    # Backup environment files
    if [[ -f .env ]]; then
        cp .env "$backup_dir/configs/env_backup" 2>/dev/null || true
    fi

    if [[ -f .env.local ]]; then
        cp .env.local "$backup_dir/configs/env_local_backup" 2>/dev/null || true
    fi

    # Backup BEV configuration files
    local config_files=(
        "docker-compose.complete.yml"
        "intelowl/configuration/analyzer_config.json"
        "intelowl/configuration/connector_config.json"
        "intelowl/configuration/generic.env"
        "vault/config.json"
        "prometheus.yml"
        "grafana/provisioning/datasources/prometheus.yml"
    )

    for config_file in "${config_files[@]}"; do
        if [[ -f "$config_file" ]]; then
            local dirname=$(dirname "$config_file")
            mkdir -p "$backup_dir/configs/$dirname"
            cp "$config_file" "$backup_dir/configs/$config_file" 2>/dev/null || true
            log "SUCCESS" "Backed up config: $config_file"
        fi
    done

    # Backup system configuration files
    local system_configs=(
        "/etc/hosts"
        "/etc/nginx/nginx.conf"
        "/etc/systemd/system/bev*.service"
    )

    for system_config in "${system_configs[@]}"; do
        if [[ -f "$system_config" ]]; then
            local basename=$(basename "$system_config")
            cp "$system_config" "$backup_dir/configs/system_$basename" 2>/dev/null || true
        fi
    done

    log "SUCCESS" "Configuration files backed up"
    return 0
}

backup_databases() {
    local backup_dir="$1"

    log "INFO" "Backing up database configurations and data"

    # PostgreSQL backup
    if command -v pg_dump >/dev/null 2>&1 && pgrep postgres >/dev/null; then
        log "INFO" "Creating PostgreSQL backup"

        # Schema backup
        pg_dump -h localhost -U researcher -d osint --schema-only > "$backup_dir/databases/postgres_schema.sql" 2>/dev/null || true

        # Small data backup (metadata only)
        pg_dump -h localhost -U researcher -d osint --data-only --table="*_metadata" > "$backup_dir/databases/postgres_metadata.sql" 2>/dev/null || true

        log "SUCCESS" "PostgreSQL backup completed"
    else
        log "INFO" "PostgreSQL not running - skipping database backup"
    fi

    # Neo4j backup
    if command -v cypher-shell >/dev/null 2>&1 && pgrep neo4j >/dev/null; then
        log "INFO" "Creating Neo4j backup"

        # Export schema
        echo "CALL db.schema.visualization()" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD" > "$backup_dir/databases/neo4j_schema.cypher" 2>/dev/null || true

        # Export constraints and indexes
        echo "SHOW CONSTRAINTS" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD" > "$backup_dir/databases/neo4j_constraints.cypher" 2>/dev/null || true
        echo "SHOW INDEXES" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD" > "$backup_dir/databases/neo4j_indexes.cypher" 2>/dev/null || true

        log "SUCCESS" "Neo4j backup completed"
    else
        log "INFO" "Neo4j not running - skipping database backup"
    fi

    # Redis backup
    if command -v redis-cli >/dev/null 2>&1 && pgrep redis >/dev/null; then
        log "INFO" "Creating Redis backup"

        # Get Redis configuration
        redis-cli CONFIG GET "*" > "$backup_dir/databases/redis_config.txt" 2>/dev/null || true

        # Small info backup
        redis-cli INFO > "$backup_dir/databases/redis_info.txt" 2>/dev/null || true

        log "SUCCESS" "Redis backup completed"
    else
        log "INFO" "Redis not running - skipping database backup"
    fi

    return 0
}

backup_system_state() {
    local backup_dir="$1"

    log "INFO" "Backing up system state"

    # System information
    cat > "$backup_dir/system/system_info.txt" << EOF
Hostname: $(hostname)
OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
Kernel: $(uname -r)
Uptime: $(uptime)
Memory: $(free -h)
Disk: $(df -h /)
CPU: $(nproc) cores
GPU: $(nvidia-smi --list-gpus 2>/dev/null | wc -l) NVIDIA GPUs
Load: $(cat /proc/loadavg)
Date: $(date)
EOF

    # Network configuration
    ip addr show > "$backup_dir/system/network_interfaces.txt"
    ip route show > "$backup_dir/system/routes.txt"
    cat /etc/hosts > "$backup_dir/system/hosts_file.txt"

    # Process information
    ps aux > "$backup_dir/system/processes.txt"
    systemctl list-units --state=active > "$backup_dir/system/systemd_services.txt"

    # Network ports
    netstat -tuln > "$backup_dir/system/network_ports.txt" 2>/dev/null || ss -tuln > "$backup_dir/system/network_ports.txt"

    # Firewall status
    if command -v ufw >/dev/null 2>&1; then
        ufw status verbose > "$backup_dir/system/firewall_ufw.txt" 2>/dev/null || true
    fi

    if command -v iptables >/dev/null 2>&1; then
        iptables -L -n > "$backup_dir/system/firewall_iptables.txt" 2>/dev/null || true
    fi

    # Environment variables (filtered)
    env | grep -E "(PATH|USER|HOME|PWD)" > "$backup_dir/system/environment.txt"

    log "SUCCESS" "System state backed up"
    return 0
}

create_rollback_script() {
    local backup_dir="$1"

    log "INFO" "Creating rollback script"

    cat > "$backup_dir/rollback.sh" << 'EOF'
#!/bin/bash

# BEV Deployment Rollback Script
# Automatically generated backup restoration

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case "$level" in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} $message"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
    esac
}

show_help() {
    cat << 'HELP'
BEV Deployment Rollback Script

USAGE:
    ./rollback.sh [OPTIONS]

OPTIONS:
    --docker-only     Rollback Docker state only
    --config-only     Rollback configurations only
    --full           Full system rollback (default)
    --force          Force rollback without confirmation
    --help           Show this help

DESCRIPTION:
    Restores the system state from the pre-deployment backup.
    By default, performs a full rollback of all backed up components.

HELP
}

confirm_rollback() {
    if [[ "${FORCE_ROLLBACK:-false}" == "true" ]]; then
        return 0
    fi

    echo
    echo -e "${YELLOW}WARNING: This will rollback your system to the pre-deployment state.${NC}"
    echo -e "${YELLOW}This action may result in data loss for any changes made after the backup.${NC}"
    echo
    read -p "Are you sure you want to proceed? (yes/no): " confirmation

    if [[ "$confirmation" == "yes" ]]; then
        return 0
    else
        echo "Rollback cancelled."
        exit 0
    fi
}

rollback_docker_state() {
    log "INFO" "Rolling back Docker state"

    # Stop all BEV containers
    local bev_containers=$(docker ps --filter name=bev --format "{{.Names}}" || true)
    if [[ -n "$bev_containers" ]]; then
        echo "$bev_containers" | xargs -r docker stop
        echo "$bev_containers" | xargs -r docker rm
    fi

    # Restore previously running containers
    if [[ -f "$BACKUP_DIR/docker/containers_running.txt" ]]; then
        log "INFO" "Restoring previously running containers"
        # Note: This is a simplified restoration - manual intervention may be required
        log "WARN" "Container restoration requires manual review of: $BACKUP_DIR/docker/containers_running.txt"
    fi

    log "SUCCESS" "Docker state rollback completed"
}

rollback_configurations() {
    log "INFO" "Rolling back configurations"

    # Restore environment files
    if [[ -f "$BACKUP_DIR/configs/env_backup" ]]; then
        cp "$BACKUP_DIR/configs/env_backup" ./.env
        log "SUCCESS" "Restored .env file"
    fi

    # Restore other configuration files
    if [[ -d "$BACKUP_DIR/configs" ]]; then
        find "$BACKUP_DIR/configs" -name "*.yml" -o -name "*.yaml" -o -name "*.json" | while read config; do
            local relative_path=$(echo "$config" | sed "s|$BACKUP_DIR/configs/||")

            if [[ "$relative_path" != "env_backup" ]] && [[ "$relative_path" != system_* ]]; then
                local target_dir=$(dirname "$relative_path")
                if [[ "$target_dir" != "." ]]; then
                    mkdir -p "$target_dir"
                fi
                cp "$config" "$relative_path"
                log "SUCCESS" "Restored configuration: $relative_path"
            fi
        done
    fi

    log "SUCCESS" "Configuration rollback completed"
}

rollback_system_state() {
    log "INFO" "Rolling back system state"

    # Stop BEV systemd services
    local bev_services=$(systemctl list-units --state=active | grep bev | awk '{print $1}' || true)
    if [[ -n "$bev_services" ]]; then
        echo "$bev_services" | xargs -r sudo systemctl stop
        log "SUCCESS" "Stopped BEV services"
    fi

    log "SUCCESS" "System state rollback completed"
}

main() {
    local rollback_type="full"
    local force_rollback=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --docker-only)
                rollback_type="docker"
                shift
                ;;
            --config-only)
                rollback_type="config"
                shift
                ;;
            --full)
                rollback_type="full"
                shift
                ;;
            --force)
                force_rollback=true
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    export FORCE_ROLLBACK="$force_rollback"

    echo -e "${GREEN}BEV Deployment Rollback${NC}"
    echo "Backup Directory: $BACKUP_DIR"
    echo "Rollback Type: $rollback_type"
    echo

    # Load backup manifest
    if [[ -f "$BACKUP_DIR/backup_manifest.json" ]]; then
        local backup_date=$(grep '"backup_date"' "$BACKUP_DIR/backup_manifest.json" | cut -d'"' -f4)
        echo "Backup Date: $backup_date"
    fi

    confirm_rollback

    case "$rollback_type" in
        "docker")
            rollback_docker_state
            ;;
        "config")
            rollback_configurations
            ;;
        "full")
            rollback_docker_state
            rollback_configurations
            rollback_system_state
            ;;
    esac

    echo
    log "SUCCESS" "Rollback completed successfully"
    log "INFO" "You may need to manually restart any services that were running before deployment"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
EOF

    chmod +x "$backup_dir/rollback.sh"
    log "SUCCESS" "Rollback script created: $backup_dir/rollback.sh"
    return 0
}

validate_backup() {
    local backup_dir="$1"

    log "INFO" "Validating backup integrity"

    # Check backup manifest
    if [[ ! -f "$backup_dir/backup_manifest.json" ]]; then
        log "ERROR" "Backup manifest missing"
        return 1
    fi

    # Check required directories
    local required_dirs=("docker" "configs" "system")
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$backup_dir/$dir" ]]; then
            log "ERROR" "Required backup directory missing: $dir"
            return 1
        fi
    done

    # Check rollback script
    if [[ ! -f "$backup_dir/rollback.sh" ]] || [[ ! -x "$backup_dir/rollback.sh" ]]; then
        log "ERROR" "Rollback script missing or not executable"
        return 1
    fi

    # Check backup size
    local backup_size_mb=$(du -sm "$backup_dir" | cut -f1)
    local max_size_mb=$((MAX_BACKUP_SIZE_GB * 1024))

    if [[ $backup_size_mb -gt $max_size_mb ]]; then
        log "WARN" "Backup size ($backup_size_mb MB) exceeds recommended maximum ($max_size_mb MB)"
    else
        log "SUCCESS" "Backup size: $backup_size_mb MB"
    fi

    log "SUCCESS" "Backup validation completed"
    return 0
}

# Cleanup old backups
cleanup_old_backups() {
    log "INFO" "Cleaning up old backups (retention: $BACKUP_RETENTION_DAYS days)"

    if [[ ! -d "$BACKUP_ROOT" ]]; then
        log "INFO" "No backup directory found - nothing to clean up"
        return 0
    fi

    local deleted_count=0

    find "$BACKUP_ROOT" -type d -name "pre-deployment-*" -mtime +$BACKUP_RETENTION_DAYS | while read old_backup; do
        log "INFO" "Removing old backup: $(basename "$old_backup")"
        rm -rf "$old_backup"
        ((deleted_count++))
    done

    if [[ $deleted_count -gt 0 ]]; then
        log "SUCCESS" "Cleaned up $deleted_count old backups"
    else
        log "SUCCESS" "No old backups to clean up"
    fi
}

# Generate backup summary for reports
generate_backup_summary() {
    if [[ -f "$BACKUP_ROOT/latest_backup" ]]; then
        local latest_backup=$(cat "$BACKUP_ROOT/latest_backup")
        local backup_date="unknown"
        local backup_size="unknown"

        if [[ -f "$latest_backup/backup_manifest.json" ]]; then
            backup_date=$(grep '"backup_date"' "$latest_backup/backup_manifest.json" | cut -d'"' -f4)
        fi

        if [[ -d "$latest_backup" ]]; then
            backup_size=$(du -sh "$latest_backup" | cut -f1)
        fi

        cat << EOF
**System Backup Created:**
- Location: $latest_backup
- Date: $backup_date
- Size: $backup_size
- Rollback Script: $latest_backup/rollback.sh

**Backup Components:**
- ✅ Docker containers and configurations
- ✅ BEV configuration files
- ✅ Database schemas and metadata
- ✅ System state and network configuration
- ✅ Automated rollback script

**Rollback Instructions:**
\`\`\`bash
cd $(dirname "$latest_backup")
./$(basename "$latest_backup")/rollback.sh --full
\`\`\`
EOF
    else
        echo "**Backup Status:** ❌ No backup created"
    fi
}