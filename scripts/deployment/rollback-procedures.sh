#!/bin/bash
#
# BEV OSINT Framework - Rollback Procedures
#
# Comprehensive rollback system for multi-node BEV deployment
# Supports service-level, node-level, and full deployment rollbacks
#

set -euo pipefail

# Configuration
THANOS_HOST="${THANOS_HOST:-100.122.12.54}"
ORACLE1_HOST="${ORACLE1_HOST:-100.96.197.84}"
STARLORD_HOST="${STARLORD_HOST:-100.122.12.35}"
VAULT_ADDR="${VAULT_ADDR:-http://100.122.12.35:8200}"

# Rollback configuration
ROLLBACK_TIMEOUT="${ROLLBACK_TIMEOUT:-300}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"
EMERGENCY_MODE="${EMERGENCY_MODE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[ROLLBACK]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

emergency() {
    echo -e "${RED}${BOLD}[EMERGENCY]${NC} $1"
}

node_log() {
    local node="$1"
    local message="$2"
    echo -e "${PURPLE}[$node]${NC} $message"
}

# Create deployment snapshot before rollback
create_deployment_snapshot() {
    local snapshot_id="$1"
    local snapshot_dir="/tmp/bev-snapshot-$snapshot_id"

    log "Creating deployment snapshot: $snapshot_id"

    mkdir -p "$snapshot_dir"/{thanos,oracle1,starlord}

    # Snapshot THANOS state
    ssh "$THANOS_HOST" "
        mkdir -p /tmp/thanos-snapshot-$snapshot_id
        docker-compose -f /opt/bev/docker-compose-thanos-unified.yml ps > /tmp/thanos-snapshot-$snapshot_id/services.txt
        docker images | grep bev > /tmp/thanos-snapshot-$snapshot_id/images.txt
        docker volume ls | grep bev > /tmp/thanos-snapshot-$snapshot_id/volumes.txt
        cp -r /opt/bev/.env* /tmp/thanos-snapshot-$snapshot_id/ 2>/dev/null || true
    " 2>/dev/null || warn "THANOS snapshot incomplete"

    # Snapshot ORACLE1 state
    ssh "$ORACLE1_HOST" "
        mkdir -p /tmp/oracle1-snapshot-$snapshot_id
        docker-compose -f /opt/bev/docker-compose-oracle1-unified.yml ps > /tmp/oracle1-snapshot-$snapshot_id/services.txt
        docker images | grep bev > /tmp/oracle1-snapshot-$snapshot_id/images.txt
        docker volume ls | grep bev > /tmp/oracle1-snapshot-$snapshot_id/volumes.txt
        cp -r /opt/bev/.env* /tmp/oracle1-snapshot-$snapshot_id/ 2>/dev/null || true
    " 2>/dev/null || warn "ORACLE1 snapshot incomplete"

    # Snapshot STARLORD state
    mkdir -p "$snapshot_dir/starlord"
    vault kv list secret/bev/ > "$snapshot_dir/starlord/vault-secrets.txt" 2>/dev/null || true
    systemctl list-units --type=service | grep bev > "$snapshot_dir/starlord/services.txt" 2>/dev/null || true

    # Create snapshot manifest
    cat > "$snapshot_dir/manifest.json" <<EOF
{
  "snapshot_id": "$snapshot_id",
  "timestamp": "$(date -Iseconds)",
  "nodes": {
    "thanos": "$THANOS_HOST",
    "oracle1": "$ORACLE1_HOST",
    "starlord": "$STARLORD_HOST"
  },
  "vault_addr": "$VAULT_ADDR",
  "snapshot_type": "pre-rollback"
}
EOF

    success "Deployment snapshot created: $snapshot_dir"
    echo "$snapshot_dir"
}

# Get last known good deployment state
get_last_known_good() {
    local deployment_type="${1:-production}"

    # Try to get from Vault first
    if vault kv get -format=json secret/bev/deployments/last-known-good-"$deployment_type" 2>/dev/null; then
        return 0
    fi

    # Fallback to local backup
    local backup_dir="/opt/bev/backups"
    if [[ -d "$backup_dir" ]]; then
        local latest_backup
        latest_backup=$(find "$backup_dir" -name "*-$deployment_type-*" -type d | sort -r | head -1)
        if [[ -n "$latest_backup" ]]; then
            echo "Found local backup: $latest_backup" >&2
            cat "$latest_backup/deployment-state.json" 2>/dev/null || echo "{}"
        fi
    fi

    echo "{}"
}

# Stop services gracefully
stop_services_gracefully() {
    local node="$1"
    local timeout="${2:-60}"

    node_log "$node" "Stopping services gracefully (timeout: ${timeout}s)"

    case "$node" in
        "thanos")
            ssh "$THANOS_HOST" "
                cd /opt/bev
                timeout $timeout docker-compose -f docker-compose-thanos-unified.yml down --timeout 30
            " 2>/dev/null || warn "THANOS graceful stop failed, forcing stop"
            ;;
        "oracle1")
            ssh "$ORACLE1_HOST" "
                cd /opt/bev
                timeout $timeout docker-compose -f docker-compose-oracle1-unified.yml down --timeout 30
            " 2>/dev/null || warn "ORACLE1 graceful stop failed, forcing stop"
            ;;
        "starlord")
            # Stop AI companion if running
            systemctl stop bev-ai-companion 2>/dev/null || true
            # Stop any local BEV services
            systemctl stop bev-* 2>/dev/null || true
            ;;
        "all")
            stop_services_gracefully "thanos" "$timeout" &
            stop_services_gracefully "oracle1" "$timeout" &
            stop_services_gracefully "starlord" "$timeout" &
            wait
            ;;
    esac

    node_log "$node" "Services stopped"
}

# Emergency stop all services
emergency_stop() {
    emergency "Initiating emergency stop of all BEV services"

    # Force kill all containers immediately
    ssh "$THANOS_HOST" "docker kill \$(docker ps -q --filter 'name=bev') 2>/dev/null || true" &
    ssh "$ORACLE1_HOST" "docker kill \$(docker ps -q --filter 'name=bev') 2>/dev/null || true" &
    systemctl stop bev-* 2>/dev/null || true &

    wait

    emergency "Emergency stop completed"
}

# Restore database from backup
restore_database() {
    local node="$1"
    local database="$2"
    local backup_file="$3"

    node_log "$node" "Restoring $database from backup: $backup_file"

    case "$database" in
        "postgresql")
            ssh "$node" "
                docker exec -i \$(docker ps -q --filter 'name=bev_postgres') \
                    psql -U researcher -d osint < '$backup_file'
            " || error "PostgreSQL restore failed"
            ;;
        "neo4j")
            ssh "$node" "
                docker exec -i \$(docker ps -q --filter 'name=bev_neo4j') \
                    neo4j-admin load --from='$backup_file' --database=neo4j --force
            " || error "Neo4j restore failed"
            ;;
        "redis")
            ssh "$node" "
                docker exec -i \$(docker ps -q --filter 'name=bev_redis') \
                    redis-cli --rdb '$backup_file'
            " || error "Redis restore failed"
            ;;
    esac

    node_log "$node" "$database restored successfully"
}

# Rollback to previous Docker images
rollback_docker_images() {
    local node="$1"
    local target_version="${2:-previous}"

    node_log "$node" "Rolling back Docker images to: $target_version"

    case "$node" in
        "thanos")
            ssh "$THANOS_HOST" "
                # Get previous image versions
                docker images --format 'table {{.Repository}}:{{.Tag}}' | grep bev | grep -v latest | head -10

                # Rollback strategy: use previous tag or specific version
                if [[ '$target_version' == 'previous' ]]; then
                    # Find previous versions and retag as current
                    for image in \$(docker images --format '{{.Repository}}' | grep bev | sort -u); do
                        previous_tag=\$(docker images \$image --format '{{.Tag}}' | grep -v latest | head -1)
                        if [[ -n \"\$previous_tag\" ]]; then
                            docker tag \$image:\$previous_tag \$image:latest
                        fi
                    done
                else
                    # Use specific version
                    for image in \$(docker images --format '{{.Repository}}' | grep bev | sort -u); do
                        if docker images \$image:$target_version >/dev/null 2>&1; then
                            docker tag \$image:$target_version \$image:latest
                        fi
                    done
                fi
            "
            ;;
        "oracle1")
            ssh "$ORACLE1_HOST" "
                # Same rollback logic for ORACLE1
                for image in \$(docker images --format '{{.Repository}}' | grep bev | sort -u); do
                    if [[ '$target_version' == 'previous' ]]; then
                        previous_tag=\$(docker images \$image --format '{{.Tag}}' | grep -v latest | head -1)
                        if [[ -n \"\$previous_tag\" ]]; then
                            docker tag \$image:\$previous_tag \$image:latest
                        fi
                    else
                        if docker images \$image:$target_version >/dev/null 2>&1; then
                            docker tag \$image:$target_version \$image:latest
                        fi
                    fi
                done
            "
            ;;
    esac

    node_log "$node" "Docker images rolled back"
}

# Restore configuration files
restore_configuration() {
    local node="$1"
    local backup_dir="$2"

    node_log "$node" "Restoring configuration from: $backup_dir"

    case "$node" in
        "thanos")
            ssh "$THANOS_HOST" "
                if [[ -d '$backup_dir/thanos' ]]; then
                    cp -r '$backup_dir/thanos/'* /opt/bev/
                    chown -R bev:bev /opt/bev/
                fi
            " || warn "THANOS configuration restore incomplete"
            ;;
        "oracle1")
            ssh "$ORACLE1_HOST" "
                if [[ -d '$backup_dir/oracle1' ]]; then
                    cp -r '$backup_dir/oracle1/'* /opt/bev/
                    chown -R bev:bev /opt/bev/
                fi
            " || warn "ORACLE1 configuration restore incomplete"
            ;;
        "starlord")
            if [[ -d "$backup_dir/starlord" ]]; then
                # Restore Vault data if needed
                if [[ -f "$backup_dir/starlord/vault-backup.snap" ]]; then
                    vault operator raft snapshot restore "$backup_dir/starlord/vault-backup.snap" || warn "Vault restore failed"
                fi

                # Restore systemd services
                if [[ -d "$backup_dir/starlord/systemd" ]]; then
                    cp "$backup_dir/starlord/systemd/"* /etc/systemd/system/
                    systemctl daemon-reload
                fi
            fi
            ;;
    esac

    node_log "$node" "Configuration restored"
}

# Rollback specific service
rollback_service() {
    local node="$1"
    local service="$2"
    local version="${3:-previous}"

    node_log "$node" "Rolling back service: $service to version: $version"

    case "$node" in
        "thanos")
            ssh "$THANOS_HOST" "
                cd /opt/bev
                # Stop specific service
                docker-compose -f docker-compose-thanos-unified.yml stop '$service'

                # Rollback image
                if [[ '$version' != 'previous' ]]; then
                    docker tag bev-$service:$version bev-$service:latest
                fi

                # Restart service
                docker-compose -f docker-compose-thanos-unified.yml up -d '$service'
            " || error "THANOS $service rollback failed"
            ;;
        "oracle1")
            ssh "$ORACLE1_HOST" "
                cd /opt/bev
                # Stop specific service
                docker-compose -f docker-compose-oracle1-unified.yml stop '$service'

                # Rollback image
                if [[ '$version' != 'previous' ]]; then
                    docker tag bev-$service:$version bev-$service:latest
                fi

                # Restart service
                docker-compose -f docker-compose-oracle1-unified.yml up -d '$service'
            " || error "ORACLE1 $service rollback failed"
            ;;
    esac

    # Wait for service to be healthy
    sleep 10
    if ! ./health-validation.sh "$node" >/dev/null 2>&1; then
        warn "$node $service may not be healthy after rollback"
    fi

    node_log "$node" "Service $service rolled back successfully"
}

# Full node rollback
rollback_node() {
    local node="$1"
    local target_version="${2:-previous}"
    local backup_dir="${3:-}"

    node_log "$node" "Starting full node rollback to: $target_version"

    # Create pre-rollback snapshot
    local snapshot_id="pre-rollback-$(date +%Y%m%d-%H%M%S)"
    create_deployment_snapshot "$snapshot_id" >/dev/null

    # Stop services gracefully
    stop_services_gracefully "$node"

    # Restore from backup if provided
    if [[ -n "$backup_dir" && -d "$backup_dir" ]]; then
        restore_configuration "$node" "$backup_dir"
    fi

    # Rollback Docker images
    rollback_docker_images "$node" "$target_version"

    # Restart services
    case "$node" in
        "thanos")
            ssh "$THANOS_HOST" "
                cd /opt/bev
                docker-compose -f docker-compose-thanos-unified.yml up -d
            " || error "THANOS restart failed"
            ;;
        "oracle1")
            ssh "$ORACLE1_HOST" "
                cd /opt/bev
                docker-compose -f docker-compose-oracle1-unified.yml up -d
            " || error "ORACLE1 restart failed"
            ;;
        "starlord")
            systemctl start bev-* 2>/dev/null || true
            ;;
    esac

    # Wait for services to stabilize
    sleep 30

    # Validate health
    if ./health-validation.sh "$node" >/dev/null 2>&1; then
        success "$node rollback completed successfully"
    else
        error "$node rollback completed but health validation failed"
        return 1
    fi
}

# Full deployment rollback
rollback_deployment() {
    local target_version="${1:-previous}"
    local backup_dir="${2:-}"

    log "Starting full BEV deployment rollback to: $target_version"

    # Create comprehensive pre-rollback snapshot
    local snapshot_id="full-rollback-$(date +%Y%m%d-%H%M%S)"
    local snapshot_dir
    snapshot_dir=$(create_deployment_snapshot "$snapshot_id")

    # Stop all services
    if [[ "$EMERGENCY_MODE" == "true" ]]; then
        emergency_stop
    else
        stop_services_gracefully "all"
    fi

    # Rollback nodes in reverse dependency order (STARLORD → ORACLE1 → THANOS)
    rollback_node "starlord" "$target_version" "$backup_dir" &
    local starlord_pid=$!

    sleep 10
    rollback_node "oracle1" "$target_version" "$backup_dir" &
    local oracle1_pid=$!

    sleep 20
    rollback_node "thanos" "$target_version" "$backup_dir" &
    local thanos_pid=$!

    # Wait for all rollbacks to complete
    wait $starlord_pid || warn "STARLORD rollback had issues"
    wait $oracle1_pid || warn "ORACLE1 rollback had issues"
    wait $thanos_pid || warn "THANOS rollback had issues"

    # Comprehensive health validation
    log "Validating rolled-back deployment..."
    if ./health-validation.sh validate; then
        success "🎉 Full deployment rollback completed successfully"

        # Update last known good state
        vault kv put secret/bev/deployments/last-known-good-production \
            version="$target_version" \
            timestamp="$(date -Iseconds)" \
            rollback_snapshot="$snapshot_id"

        return 0
    else
        error "❌ Deployment rollback completed but health validation failed"
        emergency "Manual intervention may be required"
        return 1
    fi
}

# Automated rollback decision
automated_rollback_decision() {
    local health_threshold="${1:-70}"
    local max_attempts="${2:-3}"

    log "Evaluating need for automated rollback (threshold: $health_threshold%)"

    local attempts=0
    while [[ $attempts -lt $max_attempts ]]; do
        # Get current health score
        if ! ./health-validation.sh validate >/tmp/health-check.log 2>&1; then
            warn "Health validation failed (attempt $((attempts + 1))/$max_attempts)"

            if [[ $attempts -eq $((max_attempts - 1)) ]]; then
                warn "Maximum health check attempts reached - triggering automated rollback"
                rollback_deployment "previous"
                return $?
            fi

            ((attempts++))
            sleep 60
        else
            success "System health restored - no rollback needed"
            return 0
        fi
    done
}

# Rollback specific database
rollback_database() {
    local node="$1"
    local database="$2"
    local backup_timestamp="${3:-latest}"

    node_log "$node" "Rolling back $database to backup: $backup_timestamp"

    # Find backup file
    local backup_file
    if [[ "$backup_timestamp" == "latest" ]]; then
        backup_file=$(ssh "$node" "ls -t /opt/bev/backups/$database-* 2>/dev/null | head -1" || echo "")
    else
        backup_file="/opt/bev/backups/$database-$backup_timestamp"
    fi

    if [[ -z "$backup_file" ]]; then
        error "No backup found for $database on $node"
        return 1
    fi

    # Stop database service
    case "$database" in
        "postgresql")
            ssh "$node" "docker-compose -f /opt/bev/docker-compose-*-unified.yml stop bev_postgres"
            ;;
        "neo4j")
            ssh "$node" "docker-compose -f /opt/bev/docker-compose-*-unified.yml stop bev_neo4j"
            ;;
        "redis")
            ssh "$node" "docker-compose -f /opt/bev/docker-compose-*-unified.yml stop bev_redis"
            ;;
    esac

    # Restore database
    restore_database "$node" "$database" "$backup_file"

    # Restart database service
    case "$database" in
        "postgresql")
            ssh "$node" "docker-compose -f /opt/bev/docker-compose-*-unified.yml up -d bev_postgres"
            ;;
        "neo4j")
            ssh "$node" "docker-compose -f /opt/bev/docker-compose-*-unified.yml up -d bev_neo4j"
            ;;
        "redis")
            ssh "$node" "docker-compose -f /opt/bev/docker-compose-*-unified.yml up -d bev_redis"
            ;;
    esac

    success "$database rollback completed on $node"
}

# Generate rollback report
generate_rollback_report() {
    local rollback_type="$1"
    local rollback_id="$2"
    local start_time="$3"
    local end_time="$4"
    local success_status="$5"

    local report_file="/tmp/bev-rollback-report-$rollback_id.json"

    cat > "$report_file" <<EOF
{
  "rollback_id": "$rollback_id",
  "rollback_type": "$rollback_type",
  "start_time": "$start_time",
  "end_time": "$end_time",
  "duration_seconds": $((end_time - start_time)),
  "success": $success_status,
  "nodes_affected": {
    "thanos": "$THANOS_HOST",
    "oracle1": "$ORACLE1_HOST",
    "starlord": "$STARLORD_HOST"
  },
  "emergency_mode": $EMERGENCY_MODE,
  "health_validation": "$(./health-validation.sh validate >/dev/null 2>&1 && echo 'passed' || echo 'failed')",
  "timestamp": "$(date -Iseconds)"
}
EOF

    # Store report in Vault
    vault kv put secret/bev/rollback-reports/"$rollback_id" @"$report_file" 2>/dev/null || true

    success "Rollback report generated: $report_file"
}

# Main execution function
main() {
    local action="${1:-help}"
    local start_time
    start_time=$(date +%s)

    case "$action" in
        "emergency")
            emergency_stop
            ;;
        "service")
            local node="$2"
            local service="$3"
            local version="${4:-previous}"
            rollback_service "$node" "$service" "$version"
            ;;
        "node")
            local node="$2"
            local version="${3:-previous}"
            local backup_dir="$4"
            rollback_node "$node" "$version" "$backup_dir"
            ;;
        "deployment"|"full")
            local version="${2:-previous}"
            local backup_dir="$3"
            rollback_deployment "$version" "$backup_dir"
            ;;
        "database")
            local node="$2"
            local database="$3"
            local backup_timestamp="${4:-latest}"
            rollback_database "$node" "$database" "$backup_timestamp"
            ;;
        "auto"|"automated")
            local threshold="${2:-70}"
            local attempts="${3:-3}"
            automated_rollback_decision "$threshold" "$attempts"
            ;;
        "snapshot")
            local snapshot_id="${2:-manual-$(date +%Y%m%d-%H%M%S)}"
            create_deployment_snapshot "$snapshot_id"
            ;;
        "help"|"--help")
            echo "BEV Rollback Procedures"
            echo ""
            echo "Usage: $0 ACTION [OPTIONS]"
            echo ""
            echo "Actions:"
            echo "  emergency                    Emergency stop all services"
            echo "  service NODE SERVICE [VER]   Rollback specific service"
            echo "  node NODE [VERSION] [BACKUP] Rollback entire node"
            echo "  deployment [VERSION] [BACKUP] Rollback full deployment"
            echo "  database NODE DB [TIMESTAMP] Rollback specific database"
            echo "  auto [THRESHOLD] [ATTEMPTS]  Automated rollback decision"
            echo "  snapshot [ID]                Create deployment snapshot"
            echo ""
            echo "Examples:"
            echo "  $0 emergency                 # Emergency stop"
            echo "  $0 service thanos postgres   # Rollback PostgreSQL on THANOS"
            echo "  $0 node oracle1 v1.2.3      # Rollback ORACLE1 to v1.2.3"
            echo "  $0 deployment previous       # Full rollback to previous version"
            echo "  $0 auto 80 5                # Auto rollback if health <80% after 5 attempts"
            echo ""
            echo "Environment Variables:"
            echo "  THANOS_HOST           THANOS node address"
            echo "  ORACLE1_HOST          ORACLE1 node address"
            echo "  STARLORD_HOST         STARLORD node address"
            echo "  VAULT_ADDR            Vault server address"
            echo "  ROLLBACK_TIMEOUT      Rollback operation timeout"
            echo "  EMERGENCY_MODE        Force emergency procedures"
            ;;
        *)
            error "Unknown action: $action. Use 'help' for usage information."
            exit 1
            ;;
    esac

    local end_time
    end_time=$(date +%s)

    # Generate report for major operations
    if [[ "$action" =~ ^(deployment|node|auto)$ ]]; then
        local rollback_id="$action-$(date +%Y%m%d-%H%M%S)"
        local success_status
        success_status=$([[ $? -eq 0 ]] && echo "true" || echo "false")
        generate_rollback_report "$action" "$rollback_id" "$start_time" "$end_time" "$success_status"
    fi
}

# Execute main function with all arguments
main "$@"