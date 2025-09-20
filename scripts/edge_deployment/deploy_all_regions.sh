#!/bin/bash

# Master Edge Computing Deployment Script
# BEV OSINT Framework Edge Computing Network
# Deploys edge computing infrastructure to all geographic regions

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Regions configuration
declare -A REGIONS=(
    ["us-east"]="172.30.0.47"
    ["us-west"]="172.30.0.48"
    ["eu-central"]="172.30.0.49"
    ["asia-pacific"]="172.30.0.50"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

header() {
    echo -e "${CYAN}${BOLD}"
    echo "================================================================"
    echo "$1"
    echo "================================================================"
    echo -e "${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root (sudo)"
        exit 1
    fi
}

# Display usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [REGIONS]

Deploy BEV OSINT Edge Computing Network to specified regions.

OPTIONS:
    --help              Show this help message
    --all               Deploy to all regions (default)
    --parallel          Deploy regions in parallel (faster but uses more resources)
    --sequential        Deploy regions sequentially (default, safer)
    --verify-only       Only verify existing deployments
    --start-services    Start services on all regions
    --stop-services     Stop services on all regions
    --restart-services  Restart services on all regions
    --status            Show status of all regions
    --uninstall         Uninstall edge computing from all regions
    --dry-run          Show what would be deployed without executing

REGIONS:
    us-east            US East (New York) - 172.30.0.47
    us-west            US West (San Francisco) - 172.30.0.48
    eu-central         EU Central (Berlin) - 172.30.0.49
    asia-pacific       Asia Pacific (Singapore) - 172.30.0.50

EXAMPLES:
    $0 --all                    # Deploy to all regions sequentially
    $0 --parallel --all         # Deploy to all regions in parallel
    $0 us-east us-west          # Deploy only to US regions
    $0 --verify-only            # Verify all existing deployments
    $0 --status                 # Show status of all regions

EOF
}

# Pre-deployment checks
pre_deployment_checks() {
    header "PRE-DEPLOYMENT CHECKS"

    # Check if all deployment scripts exist
    for region in "${!REGIONS[@]}"; do
        script_path="$SCRIPT_DIR/deploy_edge_${region//-/_}.sh"
        if [[ ! -f "$script_path" ]]; then
            error "Deployment script not found: $script_path"
            exit 1
        fi

        if [[ ! -x "$script_path" ]]; then
            log "Making deployment script executable: $script_path"
            chmod +x "$script_path"
        fi
    done

    # Check system requirements
    log "Checking system requirements..."

    # Check available memory (require at least 32GB total for all regions)
    local mem_gb
    mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $mem_gb -lt 32 ]]; then
        warn "Total memory ${mem_gb}GB may be insufficient for all regions (32GB+ recommended)"
    else
        info "Memory check passed: ${mem_gb}GB available"
    fi

    # Check available disk space (require at least 400GB)
    local disk_gb
    disk_gb=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $disk_gb -lt 400 ]]; then
        error "Insufficient disk space: ${disk_gb}GB available, 400GB required"
        exit 1
    fi
    info "Disk space check passed: ${disk_gb}GB available"

    # Check for required tools
    local required_tools=("python3" "pip3" "curl" "wget" "git" "systemctl" "nginx")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            error "Required tool not found: $tool"
            exit 1
        fi
    done
    info "Required tools check passed"

    # Check network connectivity
    log "Checking network connectivity..."
    for region in "${!REGIONS[@]}"; do
        local ip="${REGIONS[$region]}"
        if ping -c 1 -W 3 "$ip" &> /dev/null; then
            info "✓ Network connectivity to $region ($ip)"
        else
            warn "✗ Cannot reach $region ($ip) - may need configuration"
        fi
    done

    log "Pre-deployment checks completed"
}

# Deploy to a single region
deploy_region() {
    local region="$1"
    local ip="${REGIONS[$region]}"
    local script_path="$SCRIPT_DIR/deploy_edge_${region//-/_}.sh"

    header "DEPLOYING REGION: $region ($ip)"

    if [[ ! -f "$script_path" ]]; then
        error "Deployment script not found for region: $region"
        return 1
    fi

    log "Starting deployment for $region..."

    # Execute deployment script
    if bash "$script_path"; then
        log "✓ Deployment completed successfully for $region"
        return 0
    else
        error "✗ Deployment failed for $region"
        return 1
    fi
}

# Deploy regions sequentially
deploy_sequential() {
    local regions=("$@")
    local success_count=0
    local total_count=${#regions[@]}

    header "SEQUENTIAL DEPLOYMENT TO ${total_count} REGIONS"

    for region in "${regions[@]}"; do
        if deploy_region "$region"; then
            ((success_count++))
        fi

        # Add delay between regions to avoid resource conflicts
        if [[ $success_count -lt $total_count ]]; then
            log "Waiting 30 seconds before next region deployment..."
            sleep 30
        fi
    done

    log "Sequential deployment completed: $success_count/$total_count regions successful"
    return $((total_count - success_count))
}

# Deploy regions in parallel
deploy_parallel() {
    local regions=("$@")
    local total_count=${#regions[@]}
    local pids=()
    local results=()

    header "PARALLEL DEPLOYMENT TO ${total_count} REGIONS"

    warn "Parallel deployment uses more system resources"
    warn "Ensure system has sufficient memory and CPU capacity"

    # Start all deployments in background
    for region in "${regions[@]}"; do
        log "Starting parallel deployment for $region..."
        deploy_region "$region" &
        pids+=($!)
    done

    # Wait for all deployments to complete
    local success_count=0
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local region=${regions[$i]}

        if wait "$pid"; then
            info "✓ Parallel deployment completed for $region"
            ((success_count++))
        else
            error "✗ Parallel deployment failed for $region"
        fi
    done

    log "Parallel deployment completed: $success_count/$total_count regions successful"
    return $((total_count - success_count))
}

# Verify all deployments
verify_all_deployments() {
    header "VERIFYING ALL DEPLOYMENTS"

    local success_count=0
    local total_count=${#REGIONS[@]}

    for region in "${!REGIONS[@]}"; do
        local ip="${REGIONS[$region]}"
        local script_path="$SCRIPT_DIR/deploy_edge_${region//-/_}.sh"

        log "Verifying deployment for $region..."

        if bash "$script_path" --verify-only; then
            info "✓ Verification passed for $region"
            ((success_count++))
        else
            error "✗ Verification failed for $region"
        fi
    done

    log "Verification completed: $success_count/$total_count regions verified"
    return $((total_count - success_count))
}

# Manage services across all regions
manage_services() {
    local action="$1"
    header "MANAGING SERVICES: $action"

    for region in "${!REGIONS[@]}"; do
        local script_path="$SCRIPT_DIR/deploy_edge_${region//-/_}.sh"

        log "Executing $action for $region..."

        if bash "$script_path" "--${action}"; then
            info "✓ Service $action completed for $region"
        else
            error "✗ Service $action failed for $region"
        fi
    done
}

# Show status of all regions
show_status() {
    header "EDGE COMPUTING NETWORK STATUS"

    for region in "${!REGIONS[@]}"; do
        local ip="${REGIONS[$region]}"
        local script_path="$SCRIPT_DIR/deploy_edge_${region//-/_}.sh"

        echo -e "${CYAN}Region: $region ($ip)${NC}"
        echo "----------------------------------------"

        # Check if deployment script exists
        if [[ ! -f "$script_path" ]]; then
            echo "❌ Deployment script missing"
            continue
        fi

        # Check services status
        bash "$script_path" --status 2>/dev/null || echo "❌ Services not deployed or not running"

        # Check endpoint connectivity
        local endpoints=(
            "http://$ip:8000/health"
            "http://$ip:8002/admin/dashboard"
            "http://$ip:9090/metrics"
        )

        echo "Endpoint Status:"
        for endpoint in "${endpoints[@]}"; do
            if curl -s -f --max-time 5 "$endpoint" >/dev/null 2>&1; then
                echo "  ✅ $endpoint"
            else
                echo "  ❌ $endpoint"
            fi
        done

        echo ""
    done
}

# Uninstall from all regions
uninstall_all() {
    header "UNINSTALLING EDGE COMPUTING FROM ALL REGIONS"

    warn "This will completely remove edge computing infrastructure from all regions"
    read -p "Are you sure you want to continue? (yes/no): " confirm

    if [[ "$confirm" != "yes" ]]; then
        log "Uninstallation cancelled"
        return 0
    fi

    for region in "${!REGIONS[@]}"; do
        local script_path="$SCRIPT_DIR/deploy_edge_${region//-/_}.sh"

        log "Uninstalling edge computing from $region..."

        if bash "$script_path" --uninstall; then
            info "✓ Uninstallation completed for $region"
        else
            error "✗ Uninstallation failed for $region"
        fi
    done

    log "Uninstallation completed for all regions"
}

# Dry run - show what would be deployed
dry_run() {
    header "DRY RUN - DEPLOYMENT PLAN"

    echo "The following would be deployed:"
    echo ""

    for region in "${!REGIONS[@]}"; do
        local ip="${REGIONS[$region]}"
        echo "📍 Region: $region"
        echo "   IP Address: $ip"
        echo "   Services:"
        echo "     - Edge Node Manager (port 8000)"
        echo "     - Edge Management Service (port 8002)"
        echo "     - Model Synchronizer"
        echo "     - Geographic Router"
        echo "   Endpoints:"
        echo "     - API: http://$ip:8000/api/v1/"
        echo "     - Admin: http://$ip:8002/admin/"
        echo "     - Metrics: http://$ip:9090/metrics"
        echo "     - Health: http://$ip:8000/health"
        echo ""
    done

    echo "System Requirements:"
    echo "  - Memory: 32GB+ recommended for all regions"
    echo "  - Disk: 400GB+ required"
    echo "  - Python 3.8+"
    echo "  - Docker & Docker Compose"
    echo "  - PostgreSQL & Redis"
    echo "  - Nginx"
    echo ""

    echo "Deployment would create:"
    echo "  - /opt/bev_edge/[region]/ - Deployment directories"
    echo "  - /var/log/bev_edge/[region]/ - Log directories"
    echo "  - /opt/models/[region]/ - Model directories"
    echo "  - Systemd services for each region"
    echo "  - Nginx reverse proxy configurations"
    echo "  - Monitoring and logging setup"
}

# Parse command line arguments
DEPLOY_MODE="sequential"
REGIONS_TO_DEPLOY=()
ACTION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            usage
            exit 0
            ;;
        --all)
            REGIONS_TO_DEPLOY=($(printf '%s\n' "${!REGIONS[@]}" | sort))
            shift
            ;;
        --parallel)
            DEPLOY_MODE="parallel"
            shift
            ;;
        --sequential)
            DEPLOY_MODE="sequential"
            shift
            ;;
        --verify-only)
            ACTION="verify"
            shift
            ;;
        --start-services)
            ACTION="start_services"
            shift
            ;;
        --stop-services)
            ACTION="stop_services"
            shift
            ;;
        --restart-services)
            ACTION="restart_services"
            shift
            ;;
        --status)
            ACTION="status"
            shift
            ;;
        --uninstall)
            ACTION="uninstall"
            shift
            ;;
        --dry-run)
            ACTION="dry_run"
            shift
            ;;
        us-east|us-west|eu-central|asia-pacific)
            REGIONS_TO_DEPLOY+=("$1")
            shift
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Default to all regions if none specified
if [[ ${#REGIONS_TO_DEPLOY[@]} -eq 0 && "$ACTION" != "status" && "$ACTION" != "uninstall" && "$ACTION" != "dry_run" ]]; then
    REGIONS_TO_DEPLOY=($(printf '%s\n' "${!REGIONS[@]}" | sort))
fi

# Main execution
main() {
    header "BEV OSINT EDGE COMPUTING NETWORK DEPLOYMENT"

    echo "Deployment Configuration:"
    echo "  Mode: $DEPLOY_MODE"
    echo "  Regions: ${REGIONS_TO_DEPLOY[*]:-all}"
    echo "  Action: ${ACTION:-deploy}"
    echo ""

    case "$ACTION" in
        verify)
            verify_all_deployments
            ;;
        start_services)
            manage_services "start-services"
            ;;
        stop_services)
            manage_services "stop-services"
            ;;
        restart_services)
            manage_services "restart-services"
            ;;
        status)
            show_status
            ;;
        uninstall)
            uninstall_all
            ;;
        dry_run)
            dry_run
            ;;
        *)
            # Default deployment action
            check_root
            pre_deployment_checks

            if [[ "$DEPLOY_MODE" == "parallel" ]]; then
                deploy_parallel "${REGIONS_TO_DEPLOY[@]}"
            else
                deploy_sequential "${REGIONS_TO_DEPLOY[@]}"
            fi

            # Post-deployment verification
            echo ""
            verify_all_deployments

            # Show final status
            echo ""
            show_status

            header "DEPLOYMENT COMPLETED"
            log "Edge computing network deployed successfully!"
            log "Use '$0 --status' to check regional status"
            log "Use '$0 --verify-only' to verify deployments"
            ;;
    esac
}

# Run main function
main "$@"