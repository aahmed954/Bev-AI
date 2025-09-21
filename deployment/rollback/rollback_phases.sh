#!/bin/bash

# ===================================================================
# BEV OSINT Framework - Comprehensive Rollback System
# Safely rollback Phase 7, 8, 9 deployments with data preservation
# ===================================================================

set -euo pipefail

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"

# Rollback configuration
DEFAULT_PHASES="7,8,9"
BACKUP_RETENTION_HOURS=72
EMERGENCY_MODE=false
PRESERVE_DATA=true
FORCE_ROLLBACK=false

# Logging setup
LOG_DIR="${PROJECT_ROOT}/logs/rollback"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rollback_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Global variables
ROLLBACK_PHASES=""
ROLLBACK_REASON=""
BACKUP_PATH=""

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
        ROLLBACK) echo -e "${timestamp} ${PURPLE}[ROLLBACK]${NC} $message" ;;
    esac
}

show_banner() {
    echo -e "${RED}"
    cat << 'EOF'
██████╗  ██████╗ ██╗     ██╗     ██████╗  █████╗  ██████╗██╗  ██╗
██╔══██╗██╔═══██╗██║     ██║     ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝
██████╔╝██║   ██║██║     ██║     ██████╔╝███████║██║     █████╔╝
██╔══██╗██║   ██║██║     ██║     ██╔══██╗██╔══██║██║     ██╔═██╗
██║  ██║╚██████╔╝███████╗███████╗██████╔╝██║  ██║╚██████╗██║  ██╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝

                    BEV OSINT EMERGENCY ROLLBACK SYSTEM
                        Safe Deployment State Recovery
EOF
    echo -e "${NC}"
}

check_rollback_prerequisites() {
    log INFO "Checking rollback prerequisites..."

    # Check if Docker is available
    if ! command -v docker >/dev/null 2>&1; then
        log ERROR "Docker not available - cannot perform rollback"
        return 1
    fi

    if ! command -v docker-compose >/dev/null 2>&1; then
        log ERROR "Docker Compose not available - cannot perform rollback"
        return 1
    fi

    # Check if running with sufficient privileges
    if ! docker ps >/dev/null 2>&1; then
        log ERROR "Cannot access Docker - check permissions"
        return 1
    fi

    # Check project structure
    if [[ ! -d "$PROJECT_ROOT" ]]; then
        log ERROR "Project root not found: $PROJECT_ROOT"
        return 1
    fi

    log SUCCESS "Rollback prerequisites check passed"
}

identify_running_services() {
    log INFO "Identifying currently running Phase services..."

    local running_services=()
    local phases_array=(${ROLLBACK_PHASES//,/ })

    # Service definitions by phase
    declare -A PHASE_SERVICES=(
        ["7"]="dm-crawler crypto-intel reputation-analyzer economics-processor"
        ["8"]="tactical-intel defense-automation opsec-monitor intel-fusion"
        ["9"]="autonomous-coordinator adaptive-learning resource-manager knowledge-evolution"
    )

    for phase in "${phases_array[@]}"; do
        if [[ -n "${PHASE_SERVICES[$phase]:-}" ]]; then
            local services=${PHASE_SERVICES[$phase]}
            for service in $services; do
                if docker ps --format '{{.Names}}' | grep -q "bev_${service}"; then
                    running_services+=("bev_${service}")
                    log INFO "Found running service: bev_${service}"
                fi
            done
        fi
    done

    if [[ ${#running_services[@]} -eq 0 ]]; then
        log WARN "No Phase services currently running"
        return 1
    fi

    log INFO "Total running services to rollback: ${#running_services[@]}"
    return 0
}

create_emergency_backup() {
    if [[ "$PRESERVE_DATA" != "true" ]]; then
        log INFO "Data preservation disabled - skipping emergency backup"
        return 0
    fi

    log ROLLBACK "Creating emergency backup before rollback..."

    local backup_dir="${PROJECT_ROOT}/backups/emergency_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    # Export running container configurations
    log INFO "Backing up container configurations..."

    local phases_array=(${ROLLBACK_PHASES//,/ })
    for phase in "${phases_array[@]}"; do
        local compose_file="${PROJECT_ROOT}/docker-compose-phase${phase}.yml"
        if [[ -f "$compose_file" ]]; then
            cp "$compose_file" "$backup_dir/"
            log INFO "Backed up compose file for Phase $phase"
        fi
    done

    # Backup database data
    backup_database_data "$backup_dir"

    # Backup volumes if they exist
    backup_volume_data "$backup_dir"

    # Save backup path for potential recovery
    BACKUP_PATH="$backup_dir"
    echo "$backup_dir" > "${PROJECT_ROOT}/.last_rollback_backup"

    log SUCCESS "Emergency backup created at: $backup_dir"
}

backup_database_data() {
    local backup_dir=$1
    log INFO "Backing up database data..."

    # PostgreSQL backup
    if docker ps --format '{{.Names}}' | grep -q "postgres"; then
        log INFO "Creating PostgreSQL backup..."
        docker exec postgres pg_dumpall -U bev > "$backup_dir/postgres_emergency_backup.sql" 2>/dev/null || {
            log WARN "PostgreSQL backup failed - continuing rollback"
        }
    fi

    # Neo4j backup
    if docker ps --format '{{.Names}}' | grep -q "neo4j"; then
        log INFO "Creating Neo4j backup..."
        docker exec neo4j cypher-shell -u neo4j -p BevGraphMaster2024 \
            "CALL apoc.export.cypher.all('emergency_backup.cypher', {})" > "$backup_dir/neo4j_backup.log" 2>&1 || {
            log WARN "Neo4j backup failed - continuing rollback"
        }
    fi

    # Redis backup
    if docker ps --format '{{.Names}}' | grep -q "redis"; then
        log INFO "Creating Redis backup..."
        docker exec redis redis-cli BGSAVE > "$backup_dir/redis_backup.log" 2>&1 || {
            log WARN "Redis backup failed - continuing rollback"
        }
    fi
}

backup_volume_data() {
    local backup_dir=$1
    log INFO "Backing up volume data..."

    # Define volumes by phase
    local phase_volumes=(
        "dm_crawler_data"
        "crypto_intel_data"
        "reputation_data"
        "economics_data"
        "tactical_intel_data"
        "defense_automation_data"
        "opsec_data"
        "intel_fusion_data"
        "autonomous_data"
        "adaptive_learning_data"
        "resource_manager_data"
        "knowledge_evolution_data"
        "ml_models"
    )

    mkdir -p "$backup_dir/volumes"

    for volume in "${phase_volumes[@]}"; do
        if docker volume ls | grep -q "$volume"; then
            log INFO "Backing up volume: $volume"

            # Create temporary container to access volume
            local temp_container="backup_${volume}_$(date +%s)"
            docker run -d --name "$temp_container" \
                -v "$volume":/data \
                alpine sleep 60 >/dev/null 2>&1

            # Create archive of volume data
            docker exec "$temp_container" tar czf "/data.tar.gz" -C /data . 2>/dev/null || {
                log WARN "Failed to archive volume: $volume"
                continue
            }

            # Copy archive to backup directory
            docker cp "$temp_container":/data.tar.gz "$backup_dir/volumes/${volume}.tar.gz" 2>/dev/null || {
                log WARN "Failed to copy volume backup: $volume"
            }

            # Cleanup
            docker rm -f "$temp_container" >/dev/null 2>&1

            log SUCCESS "Volume backup completed: $volume"
        fi
    done
}

stop_phase_services() {
    log ROLLBACK "Stopping Phase services gracefully..."

    local phases_array=(${ROLLBACK_PHASES//,/ })

    # Stop services in reverse order (9, 8, 7) to respect dependencies
    local sorted_phases=($(printf '%s\n' "${phases_array[@]}" | sort -nr))

    for phase in "${sorted_phases[@]}"; do
        log INFO "Stopping Phase $phase services..."

        local compose_file="${PROJECT_ROOT}/docker-compose-phase${phase}.yml"
        if [[ -f "$compose_file" ]]; then
            # Graceful stop with timeout
            timeout 120 docker-compose -f "$compose_file" stop || {
                log WARN "Graceful stop timed out for Phase $phase, forcing stop..."
                docker-compose -f "$compose_file" kill
            }

            # Remove containers
            docker-compose -f "$compose_file" rm -f

            log SUCCESS "Phase $phase services stopped"
        else
            log WARN "Compose file not found for Phase $phase: $compose_file"
        fi
    done
}

cleanup_phase_resources() {
    log ROLLBACK "Cleaning up Phase resources..."

    local phases_array=(${ROLLBACK_PHASES//,/ })

    # Remove built images
    for phase in "${phases_array[@]}"; do
        cleanup_phase_images "$phase"
    done

    # Remove phase-specific volumes if not preserving data
    if [[ "$PRESERVE_DATA" != "true" ]]; then
        cleanup_phase_volumes
    fi

    # Remove networks if no other services are using them
    cleanup_phase_networks
}

cleanup_phase_images() {
    local phase=$1
    log INFO "Cleaning up Phase $phase Docker images..."

    # Define services by phase
    declare -A PHASE_SERVICES=(
        ["7"]="dm-crawler crypto-intel reputation-analyzer economics-processor"
        ["8"]="tactical-intel defense-automation opsec-monitor intel-fusion"
        ["9"]="autonomous-coordinator adaptive-learning resource-manager knowledge-evolution"
    )

    if [[ -n "${PHASE_SERVICES[$phase]:-}" ]]; then
        local services=${PHASE_SERVICES[$phase]}
        for service in $services; do
            local image_name="bev_${service}:latest"
            if docker images --format '{{.Repository}}:{{.Tag}}' | grep -q "$image_name"; then
                if [[ "$FORCE_ROLLBACK" == "true" ]]; then
                    docker rmi "$image_name" 2>/dev/null || {
                        log WARN "Could not remove image: $image_name"
                    }
                    log INFO "Removed image: $image_name"
                else
                    log INFO "Preserving image: $image_name (use --force to remove)"
                fi
            fi
        done
    fi
}

cleanup_phase_volumes() {
    log WARN "Removing Phase volumes (data will be lost)..."

    local phase_volumes=(
        "dm_crawler_data"
        "crypto_intel_data"
        "reputation_data"
        "economics_data"
        "tactical_intel_data"
        "defense_automation_data"
        "opsec_data"
        "intel_fusion_data"
        "autonomous_data"
        "adaptive_learning_data"
        "resource_manager_data"
        "knowledge_evolution_data"
    )

    for volume in "${phase_volumes[@]}"; do
        if docker volume ls | grep -q "$volume"; then
            docker volume rm "$volume" 2>/dev/null || {
                log WARN "Could not remove volume: $volume (may be in use)"
            }
            log INFO "Removed volume: $volume"
        fi
    done
}

cleanup_phase_networks() {
    log INFO "Checking Phase networks for cleanup..."

    # Check if any other containers are using bev_osint network
    if docker network ls | grep -q "bev_osint"; then
        local network_containers=$(docker network inspect bev_osint --format '{{range .Containers}}{{.Name}} {{end}}' 2>/dev/null || echo "")

        if [[ -z "$network_containers" ]]; then
            if [[ "$FORCE_ROLLBACK" == "true" ]]; then
                docker network rm bev_osint 2>/dev/null || {
                    log WARN "Could not remove bev_osint network"
                }
                log INFO "Removed bev_osint network"
            else
                log INFO "bev_osint network empty but preserved (use --force to remove)"
            fi
        else
            log INFO "bev_osint network still in use by: $network_containers"
        fi
    fi
}

verify_rollback() {
    log ROLLBACK "Verifying rollback completion..."

    local phases_array=(${ROLLBACK_PHASES//,/ })
    local verification_failed=false

    # Check that phase services are no longer running
    for phase in "${phases_array[@]}"; do
        # Define services by phase
        declare -A PHASE_SERVICES=(
            ["7"]="dm-crawler crypto-intel reputation-analyzer economics-processor"
            ["8"]="tactical-intel defense-automation opsec-monitor intel-fusion"
            ["9"]="autonomous-coordinator adaptive-learning resource-manager knowledge-evolution"
        )

        if [[ -n "${PHASE_SERVICES[$phase]:-}" ]]; then
            local services=${PHASE_SERVICES[$phase]}
            for service in $services; do
                if docker ps --format '{{.Names}}' | grep -q "bev_${service}"; then
                    log ERROR "Service still running: bev_${service}"
                    verification_failed=true
                fi
            done
        fi
    done

    # Check that ports are freed
    local phase_ports=(
        "8001" "8002" "8003" "8004"  # Phase 7
        "8005" "8006" "8007" "8008"  # Phase 8
        "8009" "8010" "8011" "8012"  # Phase 9
    )

    for port in "${phase_ports[@]}"; do
        if netstat -ln 2>/dev/null | grep -q ":$port "; then
            log WARN "Port still in use: $port"
        fi
    done

    if [[ "$verification_failed" == "true" ]]; then
        log ERROR "Rollback verification failed - some services still running"
        return 1
    fi

    log SUCCESS "Rollback verification completed successfully"
    return 0
}

restore_from_backup() {
    local backup_path=$1

    if [[ -z "$backup_path" || ! -d "$backup_path" ]]; then
        log ERROR "Invalid backup path: $backup_path"
        return 1
    fi

    log ROLLBACK "Restoring from backup: $backup_path"

    # Restore database data
    restore_database_data "$backup_path"

    # Restore volume data
    restore_volume_data "$backup_path"

    log SUCCESS "Backup restoration completed"
}

restore_database_data() {
    local backup_path=$1

    # Restore PostgreSQL
    if [[ -f "$backup_path/postgres_emergency_backup.sql" ]]; then
        if docker ps --format '{{.Names}}' | grep -q "postgres"; then
            log INFO "Restoring PostgreSQL data..."
            docker exec -i postgres psql -U bev < "$backup_path/postgres_emergency_backup.sql" || {
                log WARN "PostgreSQL restore failed"
            }
        fi
    fi

    # Note: Neo4j and Redis restoration would require specific procedures
    # and are left for manual intervention if needed
}

restore_volume_data() {
    local backup_path=$1
    local volumes_backup_dir="$backup_path/volumes"

    if [[ ! -d "$volumes_backup_dir" ]]; then
        log WARN "No volume backups found"
        return 0
    fi

    log INFO "Restoring volume data..."

    for backup_file in "$volumes_backup_dir"/*.tar.gz; do
        if [[ -f "$backup_file" ]]; then
            local volume_name=$(basename "$backup_file" .tar.gz)
            log INFO "Restoring volume: $volume_name"

            # Create volume if it doesn't exist
            docker volume create "$volume_name" >/dev/null 2>&1 || true

            # Create temporary container to restore data
            local temp_container="restore_${volume_name}_$(date +%s)"
            docker run -d --name "$temp_container" \
                -v "$volume_name":/data \
                alpine sleep 60 >/dev/null 2>&1

            # Copy backup to container and extract
            docker cp "$backup_file" "$temp_container":/backup.tar.gz
            docker exec "$temp_container" sh -c "cd /data && tar xzf /backup.tar.gz" || {
                log WARN "Failed to restore volume: $volume_name"
            }

            # Cleanup
            docker rm -f "$temp_container" >/dev/null 2>&1

            log SUCCESS "Volume restored: $volume_name"
        fi
    done
}

emergency_stop_all() {
    log ROLLBACK "EMERGENCY STOP - Stopping all BEV services immediately"

    # Stop all bev containers immediately
    local bev_containers=$(docker ps --filter "name=bev_" --format "{{.Names}}")
    if [[ -n "$bev_containers" ]]; then
        echo "$bev_containers" | xargs -r docker kill
        echo "$bev_containers" | xargs -r docker rm -f
        log SUCCESS "Emergency stop completed"
    else
        log INFO "No BEV containers found running"
    fi

    # Create emergency marker
    touch /tmp/bev_emergency_stop
    log INFO "Emergency stop marker created"
}

generate_rollback_report() {
    local report_file="${LOG_DIR}/rollback_report_$(date +%Y%m%d_%H%M%S).txt"

    cat > "$report_file" << EOF
===============================================================================
BEV OSINT Framework - Rollback Report
===============================================================================
Rollback Date: $(date '+%Y-%m-%d %H:%M:%S')
Phases Rolled Back: $ROLLBACK_PHASES
Reason: $ROLLBACK_REASON
Emergency Mode: $EMERGENCY_MODE
Data Preserved: $PRESERVE_DATA
Backup Path: $BACKUP_PATH

ROLLBACK SUMMARY:
- Services stopped and removed
- Docker images $([ "$FORCE_ROLLBACK" == "true" ] && echo "removed" || echo "preserved")
- Volume data $([ "$PRESERVE_DATA" == "true" ] && echo "preserved" || echo "removed")
- Emergency backup created at: $BACKUP_PATH

RECOVERY INSTRUCTIONS:
1. To restore from backup:
   $0 --restore "$BACKUP_PATH"

2. To redeploy phases:
   ${PROJECT_ROOT}/deployment/scripts/deploy_phases_7_8_9.sh --phases="$ROLLBACK_PHASES"

3. To check system status:
   docker ps
   docker volume ls
   docker network ls

LOG FILES:
- Rollback log: $LOG_FILE
- Backup details: $BACKUP_PATH/backup.log

===============================================================================
EOF

    log INFO "Rollback report generated: $report_file"
    echo "$report_file"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

BEV OSINT Framework - Emergency Rollback System

OPTIONS:
    -p, --phases PHASES      Comma-separated list of phases to rollback (default: 7,8,9)
    -r, --reason REASON      Reason for rollback (required)
    -e, --emergency          Emergency mode - immediate stop without backups
    -f, --force              Force rollback - remove images and networks
    -n, --no-preserve        Do not preserve data (remove volumes)
    --restore PATH           Restore from specific backup path
    --emergency-stop         Emergency stop all BEV services immediately
    -h, --help               Show this help message

EXAMPLES:
    $0 --reason "Phase 8 security issue"                    # Standard rollback
    $0 --phases 8,9 --reason "Performance problems"         # Rollback phases 8,9
    $0 --emergency --reason "Critical system failure"       # Emergency rollback
    $0 --restore /path/to/backup --reason "Data recovery"   # Restore from backup
    $0 --emergency-stop                                     # Immediate stop all

NOTES:
    - Always provide a reason for rollback documentation
    - Emergency mode skips backups for immediate action
    - Data is preserved by default unless --no-preserve is used
    - Use --force to completely clean up Docker resources

For more information, see: ${DEPLOYMENT_DIR}/docs/rollback_guide.md
EOF
}

# ===================================================================
# Main Execution
# ===================================================================

main() {
    # Parse command line arguments
    local restore_path=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--phases)
                ROLLBACK_PHASES="$2"
                shift 2
                ;;
            -r|--reason)
                ROLLBACK_REASON="$2"
                shift 2
                ;;
            -e|--emergency)
                EMERGENCY_MODE=true
                shift
                ;;
            -f|--force)
                FORCE_ROLLBACK=true
                shift
                ;;
            -n|--no-preserve)
                PRESERVE_DATA=false
                shift
                ;;
            --restore)
                restore_path="$2"
                shift 2
                ;;
            --emergency-stop)
                emergency_stop_all
                exit 0
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log ERROR "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Handle restore mode
    if [[ -n "$restore_path" ]]; then
        show_banner
        log ROLLBACK "Starting restore from backup: $restore_path"

        if [[ -z "$ROLLBACK_REASON" ]]; then
            ROLLBACK_REASON="Data recovery from backup"
        fi

        check_rollback_prerequisites
        restore_from_backup "$restore_path"
        log SUCCESS "Restore completed successfully"
        exit 0
    fi

    # Set defaults
    if [[ -z "$ROLLBACK_PHASES" ]]; then
        ROLLBACK_PHASES="$DEFAULT_PHASES"
    fi

    if [[ -z "$ROLLBACK_REASON" ]]; then
        log ERROR "Rollback reason is required (use --reason)"
        show_usage
        exit 1
    fi

    # Show banner and start rollback
    show_banner
    log ROLLBACK "Starting BEV OSINT Framework rollback"
    log INFO "Phases to rollback: $ROLLBACK_PHASES"
    log INFO "Reason: $ROLLBACK_REASON"
    log INFO "Emergency mode: $EMERGENCY_MODE"
    log INFO "Preserve data: $PRESERVE_DATA"

    # Confirmation for non-emergency rollbacks
    if [[ "$EMERGENCY_MODE" != "true" ]]; then
        read -p "Are you sure you want to proceed with rollback? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log INFO "Rollback cancelled by user"
            exit 0
        fi
    fi

    # Execute rollback workflow
    local rollback_start_time=$(date +%s)

    if ! check_rollback_prerequisites; then
        log ERROR "Rollback prerequisites check failed"
        exit 1
    fi

    if ! identify_running_services; then
        log WARN "No services to rollback, but continuing cleanup"
    fi

    # Skip backup in emergency mode
    if [[ "$EMERGENCY_MODE" != "true" ]]; then
        create_emergency_backup || {
            log ERROR "Emergency backup failed"
            read -p "Continue without backup? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log INFO "Rollback cancelled due to backup failure"
                exit 1
            fi
        }
    else
        log WARN "Emergency mode - skipping backup creation"
    fi

    stop_phase_services
    cleanup_phase_resources

    if ! verify_rollback; then
        log ERROR "Rollback verification failed"
        exit 1
    fi

    local rollback_end_time=$(date +%s)
    local rollback_duration=$((rollback_end_time - rollback_start_time))

    # Generate report
    local report_file=$(generate_rollback_report)

    log SUCCESS "============================================"
    log SUCCESS "BEV OSINT Framework rollback completed!"
    log SUCCESS "Phases rolled back: $ROLLBACK_PHASES"
    log SUCCESS "Total rollback time: ${rollback_duration}s"
    log SUCCESS "Backup location: $BACKUP_PATH"
    log SUCCESS "Report file: $report_file"
    log SUCCESS "============================================"

    # Show next steps
    echo ""
    echo "NEXT STEPS:"
    echo "1. Review rollback report: cat $report_file"
    echo "2. Check system status: docker ps && docker volume ls"
    if [[ -n "$BACKUP_PATH" ]]; then
        echo "3. To restore data: $0 --restore '$BACKUP_PATH' --reason 'Data recovery'"
    fi
    echo "4. To redeploy: ${PROJECT_ROOT}/deployment/scripts/deploy_phases_7_8_9.sh"

    log SUCCESS "Rollback procedure completed successfully"
}

# Set up signal handlers
trap 'log WARN "Rollback interrupted by user"; exit 130' INT TERM

# Execute main function with all arguments
main "$@"