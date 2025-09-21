#!/bin/bash
# Standalone AI Companion Deployment Script
# Complete deployment and management for STARLORD companion

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/companion/deploy-$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging
log() {
    echo -e "${1}" | tee -a "$LOG_FILE"
}

error_exit() {
    log "${RED}ERROR: ${1}${NC}"
    exit 1
}

success() {
    log "${GREEN}✓ ${1}${NC}"
}

warning() {
    log "${YELLOW}⚠ ${1}${NC}"
}

info() {
    log "${BLUE}ℹ ${1}${NC}"
}

header() {
    log "${PURPLE}=== ${1} ===${NC}"
}

# Ensure log directory exists
sudo mkdir -p "$(dirname "$LOG_FILE")"
sudo touch "$LOG_FILE"
sudo chown starlord:starlord "$LOG_FILE"

# ASCII Art Header
log "${CYAN}"
cat << 'EOF'
     ╔══════════════════════════════════════════════════════════════╗
     ║                                                              ║
     ║             AI COMPANION STANDALONE DEPLOYMENT              ║
     ║                                                              ║
     ║              🤖 Autonomous • Isolated • Powerful             ║
     ║                                                              ║
     ╚══════════════════════════════════════════════════════════════╝
EOF
log "${NC}"

# Command line options
OPERATION="deploy"
SKIP_CHECKS=false
FORCE_REBUILD=false
ENABLE_INTEGRATION=false
CLEANUP_VOLUMES=false

show_help() {
    cat << EOF
AI Companion Standalone Deployment Script

Usage: $0 [OPTIONS] [OPERATION]

OPERATIONS:
    deploy      Deploy the AI companion (default)
    start       Start existing deployment
    stop        Stop running deployment
    restart     Restart all services
    status      Show deployment status
    logs        Show service logs
    health      Run health check
    cleanup     Clean up resources
    uninstall   Complete removal

OPTIONS:
    -h, --help              Show this help message
    -s, --skip-checks       Skip pre-deployment checks
    -f, --force-rebuild     Force rebuild of all images
    -i, --enable-integration Enable core platform integration detection
    -c, --cleanup-volumes   Remove all data volumes (destructive)
    -v, --verbose           Enable verbose output

Examples:
    $0 deploy                    # Standard deployment
    $0 deploy -f                 # Force rebuild deployment
    $0 deploy -i                 # Deploy with integration enabled
    $0 status                    # Check status
    $0 logs companion_core       # View core service logs
    $0 cleanup -c                # Full cleanup including data

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--skip-checks)
            SKIP_CHECKS=true
            shift
            ;;
        -f|--force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        -i|--enable-integration)
            ENABLE_INTEGRATION=true
            shift
            ;;
        -c|--cleanup-volumes)
            CLEANUP_VOLUMES=true
            shift
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        deploy|start|stop|restart|status|logs|health|cleanup|uninstall)
            OPERATION="$1"
            shift
            ;;
        *)
            error_exit "Unknown option: $1"
            ;;
    esac
done

# Main operations
case "$OPERATION" in
    "deploy")
        header "STARTING AI COMPANION DEPLOYMENT"

        # Pre-deployment checks
        if [[ "$SKIP_CHECKS" == "false" ]]; then
            info "Running pre-deployment checks..."
            bash "$SCRIPT_DIR/scripts/pre-start-checks.sh" || error_exit "Pre-deployment checks failed"
        fi

        # Prepare resources
        info "Preparing system resources..."
        bash "$SCRIPT_DIR/scripts/prepare-resources.sh" || error_exit "Resource preparation failed"

        # Set environment variables
        if [[ "$ENABLE_INTEGRATION" == "true" ]]; then
            export CORE_PLATFORM_ENABLED=true
            export COMPANION_GATEWAY_MODE=integrated
            info "Integration mode enabled"
        else
            export CORE_PLATFORM_ENABLED=false
            export COMPANION_GATEWAY_MODE=standalone
            info "Standalone mode enabled"
        fi

        # Build images if needed
        if [[ "$FORCE_REBUILD" == "true" ]]; then
            info "Force rebuilding all images..."
            docker-compose -f docker-compose.companion.yml build --no-cache
        else
            info "Building updated images..."
            docker-compose -f docker-compose.companion.yml build
        fi

        # Deploy services
        info "Deploying companion services..."
        docker-compose -f docker-compose.companion.yml up -d

        # Wait and run health check
        info "Waiting for services to start..."
        sleep 30

        info "Running health check..."
        bash "$SCRIPT_DIR/scripts/health-check.sh" || error_exit "Health check failed"

        success "AI Companion deployed successfully!"
        info "Access the companion at: http://localhost:18080"
        info "Grafana dashboard: http://localhost:19000 (admin/companion_admin_2024)"
        info "Prometheus metrics: http://localhost:19090"
        ;;

    "start")
        header "STARTING AI COMPANION SERVICES"
        docker-compose -f docker-compose.companion.yml start
        sleep 15
        bash "$SCRIPT_DIR/scripts/health-check.sh"
        success "AI Companion services started"
        ;;

    "stop")
        header "STOPPING AI COMPANION SERVICES"
        docker-compose -f docker-compose.companion.yml stop
        success "AI Companion services stopped"
        ;;

    "restart")
        header "RESTARTING AI COMPANION SERVICES"
        docker-compose -f docker-compose.companion.yml restart
        sleep 30
        bash "$SCRIPT_DIR/scripts/health-check.sh"
        success "AI Companion services restarted"
        ;;

    "status")
        header "AI COMPANION STATUS"

        info "Container Status:"
        docker-compose -f docker-compose.companion.yml ps

        echo ""
        info "Resource Usage:"
        docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

        echo ""
        info "GPU Status:"
        nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

        echo ""
        info "Service Health:"
        curl -s http://localhost:18000/health | jq . 2>/dev/null || echo "Core service unavailable"
        ;;

    "logs")
        SERVICE_NAME="${2:-}"
        if [[ -n "$SERVICE_NAME" ]]; then
            docker-compose -f docker-compose.companion.yml logs -f "$SERVICE_NAME"
        else
            info "Available services:"
            docker-compose -f docker-compose.companion.yml ps --services
            echo ""
            info "Usage: $0 logs <service_name>"
            info "Example: $0 logs companion_core"
        fi
        ;;

    "health")
        header "AI COMPANION HEALTH CHECK"
        bash "$SCRIPT_DIR/scripts/health-check.sh"
        ;;

    "cleanup")
        header "CLEANING UP AI COMPANION RESOURCES"

        if [[ "$CLEANUP_VOLUMES" == "true" ]]; then
            warning "This will remove ALL companion data including databases!"
            read -p "Are you sure? (yes/no): " -r
            if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
                error_exit "Cleanup cancelled"
            fi
            export CLEANUP_VOLUMES=true
        fi

        docker-compose -f docker-compose.companion.yml down
        bash "$SCRIPT_DIR/scripts/cleanup-resources.sh"
        success "Cleanup completed"
        ;;

    "uninstall")
        header "UNINSTALLING AI COMPANION"

        warning "This will completely remove the AI companion including all data!"
        read -p "Are you sure you want to uninstall? (yes/no): " -r
        if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
            error_exit "Uninstall cancelled"
        fi

        # Stop services
        docker-compose -f docker-compose.companion.yml down -v

        # Remove images
        docker rmi -f $(docker images -q --filter "reference=companion-*") 2>/dev/null || true

        # Cleanup resources
        export CLEANUP_VOLUMES=true
        bash "$SCRIPT_DIR/scripts/cleanup-resources.sh"

        # Remove systemd service
        if [[ -f /etc/systemd/system/companion.service ]]; then
            sudo systemctl disable companion.service
            sudo rm /etc/systemd/system/companion.service
            sudo systemctl daemon-reload
            success "Systemd service removed"
        fi

        # Remove data directories
        sudo rm -rf /opt/companion
        sudo rm -rf /var/log/companion

        success "AI Companion completely uninstalled"
        ;;

    *)
        error_exit "Unknown operation: $OPERATION"
        ;;
esac

# Footer
log ""
log "${PURPLE}╔════════════════════════════════════════════════════════════════╗${NC}"
log "${PURPLE}║                     Operation Completed                       ║${NC}"
log "${PURPLE}╚════════════════════════════════════════════════════════════════╝${NC}"
log ""

exit 0