#!/bin/bash

##############################################################################
# BEV OSINT Framework - System Manager Script
# Complete system management for multi-node BEV deployment
##############################################################################

set -euo pipefail

# Color codes
declare -r RED='\033[0;31m'
declare -r GREEN='\033[0;32m'
declare -r YELLOW='\033[1;33m'
declare -r BLUE='\033[0;34m'
declare -r PURPLE='\033[0;35m'
declare -r CYAN='\033[0;36m'
declare -r WHITE='\033[1;37m'
declare -r NC='\033[0m'

# Configuration
declare -r BEV_HOME="/home/starlord/Projects/Bev"
declare -r SCRIPTS_DIR="$BEV_HOME/scripts"
declare -r THANOS_HOST="localhost"
declare -r ORACLE1_HOST="100.96.197.84"
declare -r ORACLE1_USER="starlord"
declare -r SSH_KEY="$HOME/.ssh/bev_deployment_key"

# Service groups for organized management
declare -ra FOUNDATION_SERVICES=("postgres" "neo4j" "redis" "elasticsearch" "influxdb")
declare -ra MONITORING_SERVICES=("prometheus" "grafana" "airflow-webserver" "airflow-scheduler")
declare -ra PROCESSING_SERVICES=("minio" "ocr-service" "document-analyzer" "intelowl")
declare -ra AGENT_SERVICES=("swarm-coordinator" "research-agent" "memory-agent" "optimization-agent")
declare -ra SECURITY_SERVICES=("vault" "guardian-service" "tor-relay" "ids-system")
declare -ra ADVANCED_SERVICES=("autonomous-agent" "live2d-service" "multimodal-processor")

##############################################################################
# Utility Functions
##############################################################################
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${PURPLE}║                      BEV OSINT System Manager                           ║${NC}"
    echo -e "${PURPLE}║                        $(date '+%Y-%m-%d %H:%M:%S %Z')                        ║${NC}"
    echo -e "${PURPLE}╚══════════════════════════════════════════════════════════════════════════╝${NC}\n"
}

##############################################################################
# Service Management Functions
##############################################################################
start_service_group() {
    local group_name="$1"
    local node="${2:-thanos}"

    case "$group_name" in
        "foundation")
            local services=("${FOUNDATION_SERVICES[@]}")
            ;;
        "monitoring")
            local services=("${MONITORING_SERVICES[@]}")
            ;;
        "processing")
            local services=("${PROCESSING_SERVICES[@]}")
            ;;
        "agents")
            local services=("${AGENT_SERVICES[@]}")
            ;;
        "security")
            local services=("${SECURITY_SERVICES[@]}")
            ;;
        "advanced")
            local services=("${ADVANCED_SERVICES[@]}")
            ;;
        *)
            log_error "Unknown service group: $group_name"
            return 1
            ;;
    esac

    log_step "Starting $group_name services on $node"

    if [[ "$node" == "thanos" ]]; then
        cd "$BEV_HOME"
        docker-compose -f docker-compose-thanos-unified.yml up -d "${services[@]}" || {
            log_error "Failed to start some $group_name services on THANOS"
            return 1
        }
    else
        # Oracle1 services mapping
        local oracle1_services=()
        case "$group_name" in
            "foundation")
                oracle1_services=("redis" "n8n" "nginx-proxy")
                ;;
            "processing")
                oracle1_services=("crawler" "intel-collector")
                ;;
            "agents")
                oracle1_services=("social-media-agent" "osint-crawler")
                ;;
            "advanced")
                oracle1_services=("ai-coordinator" "blockchain-monitor")
                ;;
            *)
                log_warning "No $group_name services defined for ORACLE1"
                return 0
                ;;
        esac

        if [[ ${#oracle1_services[@]} -gt 0 ]]; then
            ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
                "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml up -d ${oracle1_services[*]}" || {
                log_error "Failed to start some $group_name services on ORACLE1"
                return 1
            }
        fi
    fi

    log_success "$group_name services started on $node"
}

stop_service_group() {
    local group_name="$1"
    local node="${2:-thanos}"

    case "$group_name" in
        "foundation")
            local services=("${FOUNDATION_SERVICES[@]}")
            ;;
        "monitoring")
            local services=("${MONITORING_SERVICES[@]}")
            ;;
        "processing")
            local services=("${PROCESSING_SERVICES[@]}")
            ;;
        "agents")
            local services=("${AGENT_SERVICES[@]}")
            ;;
        "security")
            local services=("${SECURITY_SERVICES[@]}")
            ;;
        "advanced")
            local services=("${ADVANCED_SERVICES[@]}")
            ;;
        *)
            log_error "Unknown service group: $group_name"
            return 1
            ;;
    esac

    log_step "Stopping $group_name services on $node"

    if [[ "$node" == "thanos" ]]; then
        cd "$BEV_HOME"
        docker-compose -f docker-compose-thanos-unified.yml stop "${services[@]}" || {
            log_warning "Some $group_name services on THANOS may not have stopped cleanly"
        }
    else
        # Oracle1 services mapping
        local oracle1_services=()
        case "$group_name" in
            "foundation")
                oracle1_services=("redis" "n8n" "nginx-proxy")
                ;;
            "processing")
                oracle1_services=("crawler" "intel-collector")
                ;;
            "agents")
                oracle1_services=("social-media-agent" "osint-crawler")
                ;;
            "advanced")
                oracle1_services=("ai-coordinator" "blockchain-monitor")
                ;;
            *)
                log_warning "No $group_name services defined for ORACLE1"
                return 0
                ;;
        esac

        if [[ ${#oracle1_services[@]} -gt 0 ]]; then
            ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
                "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml stop ${oracle1_services[*]}" || {
                log_warning "Some $group_name services on ORACLE1 may not have stopped cleanly"
            }
        fi
    fi

    log_success "$group_name services stopped on $node"
}

restart_service_group() {
    local group_name="$1"
    local node="${2:-thanos}"

    log_step "Restarting $group_name services on $node"
    stop_service_group "$group_name" "$node"
    sleep 5
    start_service_group "$group_name" "$node"
}

##############################################################################
# Full System Operations
##############################################################################
start_full_system() {
    log_step "Starting complete BEV OSINT Framework"

    # Start services in dependency order
    log_info "Phase 1: Foundation services"
    start_service_group "foundation" "thanos"
    start_service_group "foundation" "oracle1"
    sleep 15

    log_info "Phase 2: Monitoring services"
    start_service_group "monitoring" "thanos"
    sleep 10

    log_info "Phase 3: Processing services"
    start_service_group "processing" "thanos"
    start_service_group "processing" "oracle1"
    sleep 10

    log_info "Phase 4: Agent services"
    start_service_group "agents" "thanos"
    start_service_group "agents" "oracle1"
    sleep 10

    log_info "Phase 5: Security services"
    start_service_group "security" "thanos"
    sleep 10

    log_info "Phase 6: Advanced services"
    start_service_group "advanced" "thanos"
    start_service_group "advanced" "oracle1"

    log_success "Complete BEV OSINT Framework started"

    # Run health check
    log_step "Running post-startup health check"
    sleep 30
    "$SCRIPTS_DIR/bev_status_monitor.sh" --critical
}

stop_full_system() {
    log_step "Stopping complete BEV OSINT Framework"

    # Stop services in reverse dependency order
    log_info "Stopping advanced services"
    stop_service_group "advanced" "thanos"
    stop_service_group "advanced" "oracle1"

    log_info "Stopping security services"
    stop_service_group "security" "thanos"

    log_info "Stopping agent services"
    stop_service_group "agents" "thanos"
    stop_service_group "agents" "oracle1"

    log_info "Stopping processing services"
    stop_service_group "processing" "thanos"
    stop_service_group "processing" "oracle1"

    log_info "Stopping monitoring services"
    stop_service_group "monitoring" "thanos"

    log_info "Stopping foundation services"
    stop_service_group "foundation" "oracle1"
    stop_service_group "foundation" "thanos"

    log_success "Complete BEV OSINT Framework stopped"
}

restart_full_system() {
    log_step "Restarting complete BEV OSINT Framework"
    stop_full_system
    sleep 10
    start_full_system
}

##############################################################################
# Maintenance Operations
##############################################################################
update_system() {
    log_step "Updating BEV OSINT Framework"

    # Pull latest images
    log_info "Pulling latest Docker images for THANOS"
    cd "$BEV_HOME"
    docker-compose -f docker-compose-thanos-unified.yml pull || {
        log_warning "Some images could not be updated on THANOS"
    }

    log_info "Pulling latest Docker images for ORACLE1"
    ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
        "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml pull" || {
        log_warning "Some images could not be updated on ORACLE1"
    }

    # Restart services to use new images
    log_info "Restarting services with updated images"
    restart_full_system

    log_success "System update completed"
}

cleanup_system() {
    log_step "Cleaning up BEV OSINT Framework"

    # Cleanup THANOS
    log_info "Cleaning up THANOS"
    cd "$BEV_HOME"
    docker system prune -f || true
    docker volume prune -f || true

    # Cleanup ORACLE1
    log_info "Cleaning up ORACLE1"
    ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
        "docker system prune -f && docker volume prune -f" || {
        log_warning "ORACLE1 cleanup may have failed"
    }

    log_success "System cleanup completed"
}

backup_system() {
    local backup_name="${1:-bev_backup_$(date +%Y%m%d_%H%M%S)}"
    local backup_dir="$BEV_HOME/backups/$backup_name"

    log_step "Creating system backup: $backup_name"

    mkdir -p "$backup_dir"

    # Backup configurations
    log_info "Backing up configurations"
    cp -r "$BEV_HOME/config" "$backup_dir/"
    cp "$BEV_HOME/.env" "$backup_dir/"
    cp "$BEV_HOME"/*.yml "$backup_dir/"

    # Backup database data (if volumes exist)
    log_info "Backing up database volumes"
    docker run --rm -v bev_postgres_data:/data -v "$backup_dir":/backup alpine \
        tar czf /backup/postgres_data.tar.gz -C /data . 2>/dev/null || {
        log_warning "PostgreSQL backup failed"
    }

    # Create backup manifest
    cat > "$backup_dir/manifest.txt" << EOF
BEV OSINT Framework Backup
Created: $(date)
Backup Name: $backup_name
Contents:
- Configuration files
- Environment files
- Docker Compose files
- Database volumes (if available)
EOF

    log_success "System backup created: $backup_dir"
}

##############################################################################
# Resource Management
##############################################################################
check_resources() {
    log_step "Checking system resources"

    echo -e "${WHITE}THANOS Resource Usage:${NC}"
    echo "Memory: $(free -h | awk 'NR==2{printf "Used: %s/%s (%.1f%%)", $3, $2, $3*100/$2}')"
    echo "Disk: $(df -h "$BEV_HOME" | awk 'NR==2{printf "Used: %s/%s (%s)", $3, $2, $5}')"
    echo "Docker: $(docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}")"

    echo ""

    if ssh -o ConnectTimeout=5 -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" "echo 'connected'" &>/dev/null; then
        echo -e "${WHITE}ORACLE1 Resource Usage:${NC}"
        ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" "
            echo \"Memory: \$(free -h | awk 'NR==2{printf \"Used: %s/%s (%.1f%%)\", \$3, \$2, \$3*100/\$2}')\"
            echo \"Disk: \$(df -h ~ | awk 'NR==2{printf \"Used: %s/%s (%s)\", \$3, \$2, \$5}')\"
            echo \"Docker: \$(docker system df --format 'table {{.Type}}\t{{.TotalCount}}\t{{.Size}}\t{{.Reclaimable}}')\"
        "
    else
        log_warning "Cannot connect to ORACLE1 for resource check"
    fi
}

optimize_resources() {
    log_step "Optimizing system resources"

    # Optimize THANOS
    log_info "Optimizing THANOS resources"
    cd "$BEV_HOME"
    docker system prune -af --volumes || true

    # Optimize ORACLE1
    log_info "Optimizing ORACLE1 resources"
    ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
        "docker system prune -af --volumes" || {
        log_warning "ORACLE1 optimization may have failed"
    }

    log_success "Resource optimization completed"
}

##############################################################################
# Troubleshooting
##############################################################################
diagnose_issues() {
    log_step "Running system diagnostics"

    # Check service status
    log_info "Checking service status"
    "$SCRIPTS_DIR/bev_status_monitor.sh" --critical

    # Check logs for errors
    log_info "Collecting and analyzing logs"
    "$SCRIPTS_DIR/bev_log_aggregator.sh" --collect --analyze

    # Check connectivity
    log_info "Testing network connectivity"
    if ping -c 3 "$ORACLE1_HOST" &>/dev/null; then
        log_success "Network connectivity to ORACLE1: OK"
    else
        log_error "Network connectivity to ORACLE1: FAILED"
    fi

    # Check resource usage
    check_resources

    log_success "System diagnostics completed"
}

quick_fix() {
    local issue_type="$1"

    case "$issue_type" in
        "services")
            log_step "Quick fix: Restarting failed services"
            restart_full_system
            ;;
        "network")
            log_step "Quick fix: Resetting network configuration"
            cd "$BEV_HOME"
            docker network prune -f
            ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" "docker network prune -f"
            restart_service_group "foundation"
            ;;
        "storage")
            log_step "Quick fix: Cleaning up storage"
            cleanup_system
            ;;
        "permissions")
            log_step "Quick fix: Fixing permissions"
            chmod +x "$BEV_HOME"/*.sh
            chmod +x "$SCRIPTS_DIR"/*.sh
            ;;
        *)
            log_error "Unknown issue type: $issue_type"
            log_info "Available quick fixes: services, network, storage, permissions"
            return 1
            ;;
    esac

    log_success "Quick fix for $issue_type completed"
}

##############################################################################
# Command Line Interface
##############################################################################
show_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Service Management:"
    echo "  start GROUP [NODE]      Start service group (foundation/monitoring/processing/agents/security/advanced)"
    echo "  stop GROUP [NODE]       Stop service group"
    echo "  restart GROUP [NODE]    Restart service group"
    echo "  status                  Show system status"
    echo ""
    echo "Full System Operations:"
    echo "  start-all               Start complete BEV system"
    echo "  stop-all                Stop complete BEV system"
    echo "  restart-all             Restart complete BEV system"
    echo ""
    echo "Maintenance:"
    echo "  update                  Update system (pull latest images)"
    echo "  cleanup                 Clean up unused Docker resources"
    echo "  backup [NAME]           Create system backup"
    echo "  optimize                Optimize system resources"
    echo ""
    echo "Monitoring:"
    echo "  resources               Check resource usage"
    echo "  logs                    Show recent logs"
    echo "  monitor                 Start real-time monitoring"
    echo ""
    echo "Troubleshooting:"
    echo "  diagnose                Run full system diagnostics"
    echo "  fix TYPE                Quick fix (services/network/storage/permissions)"
    echo ""
    echo "Utility:"
    echo "  --help, -h              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start foundation     # Start foundation services on THANOS"
    echo "  $0 restart agents oracle1  # Restart agent services on ORACLE1"
    echo "  $0 start-all            # Start complete system"
    echo "  $0 diagnose             # Run full diagnostics"
}

##############################################################################
# Main Function
##############################################################################
main() {
    case "${1:-}" in
        --help|-h)
            show_usage
            exit 0
            ;;
        start)
            if [[ -z "${2:-}" ]]; then
                log_error "Service group required"
                show_usage
                exit 1
            fi
            print_header
            start_service_group "$2" "${3:-thanos}"
            ;;
        stop)
            if [[ -z "${2:-}" ]]; then
                log_error "Service group required"
                show_usage
                exit 1
            fi
            print_header
            stop_service_group "$2" "${3:-thanos}"
            ;;
        restart)
            if [[ -z "${2:-}" ]]; then
                log_error "Service group required"
                show_usage
                exit 1
            fi
            print_header
            restart_service_group "$2" "${3:-thanos}"
            ;;
        status)
            "$SCRIPTS_DIR/bev_status_monitor.sh"
            ;;
        start-all)
            print_header
            start_full_system
            ;;
        stop-all)
            print_header
            stop_full_system
            ;;
        restart-all)
            print_header
            restart_full_system
            ;;
        update)
            print_header
            update_system
            ;;
        cleanup)
            print_header
            cleanup_system
            ;;
        backup)
            print_header
            backup_system "${2:-}"
            ;;
        optimize)
            print_header
            optimize_resources
            ;;
        resources)
            print_header
            check_resources
            ;;
        logs)
            "$SCRIPTS_DIR/bev_log_aggregator.sh" --health
            ;;
        monitor)
            "$SCRIPTS_DIR/bev_status_monitor.sh" --monitor
            ;;
        diagnose)
            print_header
            diagnose_issues
            ;;
        fix)
            if [[ -z "${2:-}" ]]; then
                log_error "Issue type required for fix"
                show_usage
                exit 1
            fi
            print_header
            quick_fix "$2"
            ;;
        "")
            print_header
            log_info "No command specified. Use --help for usage information."
            log_info "Quick start: $0 status"
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"