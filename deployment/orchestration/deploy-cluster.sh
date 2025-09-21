#!/bin/bash

#################################################################
# BEV OSINT Framework - Cluster Deployment Script
#
# This script orchestrates the deployment of a complete BEV OSINT
# cluster across multiple nodes using the GitHub repository.
#
# Usage:
#   ./deploy-cluster.sh --config cluster-config.yml
#   ./deploy-cluster.sh --interactive
#
#################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GITHUB_REPO="https://github.com/aahmed954/Bev-AI.git"
GITHUB_BRANCH="${BEV_BRANCH:-main}"
BOOTSTRAP_URL="https://raw.githubusercontent.com/aahmed954/Bev-AI/main/deployment/bootstrap/node-bootstrap.sh"

# Default settings
CLUSTER_CONFIG=""
INTERACTIVE_MODE=false
DRY_RUN=false
PARALLEL_DEPLOYMENT=true
COORDINATOR_HOST=""
COORDINATOR_PORT=8080
SSH_KEY_PATH=""
SSH_USER="ubuntu"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}${BOLD}============================================${NC}"
    echo -e "${CYAN}${BOLD} $1 ${NC}"
    echo -e "${CYAN}${BOLD}============================================${NC}\n"
}

# Show banner
show_banner() {
    cat << 'EOF'

 ██████╗ ███████╗██╗   ██╗     ██████╗ ██╗     ██╗   ██╗███████╗████████╗███████╗██████╗ 
 ██╔══██╗██╔════╝██║   ██║    ██╔════╝ ██║     ██║   ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗
 ██████╔╝█████╗  ██║   ██║    ██║  ███╗██║     ██║   ██║███████╗   ██║   █████╗  ██████╔╝
 ██╔══██╗██╔══╝  ╚██╗ ██╔╝    ██║   ██║██║     ██║   ██║╚════██║   ██║   ██╔══╝  ██╔══██╗
 ██████╔╝███████╗ ╚████╔╝     ╚██████╔╝███████╗╚██████╔╝███████║   ██║   ███████╗██║  ██║
 ╚═════╝ ╚══════╝  ╚═══╝       ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

    BEV OSINT Framework - Distributed Cluster Deployment

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CLUSTER_CONFIG="$2"
                shift 2
                ;;
            --interactive)
                INTERACTIVE_MODE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --sequential)
                PARALLEL_DEPLOYMENT=false
                shift
                ;;
            --coordinator-host)
                COORDINATOR_HOST="$2"
                shift 2
                ;;
            --coordinator-port)
                COORDINATOR_PORT="$2"
                shift 2
                ;;
            --ssh-key)
                SSH_KEY_PATH="$2"
                shift 2
                ;;
            --ssh-user)
                SSH_USER="$2"
                shift 2
                ;;
            --branch)
                GITHUB_BRANCH="$2"
                shift 2
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Show help
show_help() {
    cat << EOF
BEV OSINT Framework - Cluster Deployment Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --config FILE           Cluster configuration file (YAML)
    --interactive           Interactive cluster configuration
    --dry-run              Show what would be deployed without executing
    --sequential           Deploy nodes sequentially instead of parallel
    --coordinator-host IP  Cluster coordinator IP address
    --coordinator-port N   Cluster coordinator port (default: 8080)
    --ssh-key PATH         SSH private key for node access
    --ssh-user USER        SSH username (default: ubuntu)
    --branch BRANCH        Git branch to deploy (default: main)
    --help                 Show this help message

EXAMPLES:
    # Interactive cluster setup
    $0 --interactive

    # Deploy from configuration file
    $0 --config production-cluster.yml

    # Dry run to see deployment plan
    $0 --config test-cluster.yml --dry-run

    # Deploy with custom coordinator
    $0 --config cluster.yml --coordinator-host 10.0.1.100

CLUSTER CONFIGURATION FORMAT:
    cluster:
      name: "bev-production"
      coordinator:
        host: "10.0.1.100"
        port: 8080
      
    nodes:
      - name: "data-core-01"
        type: "data-core"
        host: "10.0.1.10"
        ssh_user: "ubuntu"
        ssh_key: "~/.ssh/bev-key.pem"
        
      - name: "processing-01"
        type: "processing-core"
        host: "10.0.1.20"
        depends_on: ["data-core-01"]

EOF
}

# Load cluster configuration
load_cluster_config() {
    if [[ -n "$CLUSTER_CONFIG" ]]; then
        if [[ ! -f "$CLUSTER_CONFIG" ]]; then
            log_error "Cluster configuration file not found: $CLUSTER_CONFIG"
            exit 1
        fi
        log_info "Loading cluster configuration from $CLUSTER_CONFIG"
        # Parse YAML configuration (simplified - would use yq in production)
    elif [[ "$INTERACTIVE_MODE" == "true" ]]; then
        interactive_cluster_config
    else
        log_error "No cluster configuration provided. Use --config or --interactive"
        exit 1
    fi
}

# Interactive cluster configuration
interactive_cluster_config() {
    log_header "Interactive Cluster Configuration"
    
    # Cluster name
    read -p "Cluster name [bev-osint-cluster]: " cluster_name
    cluster_name=${cluster_name:-bev-osint-cluster}
    
    # Coordinator setup
    if [[ -z "$COORDINATOR_HOST" ]]; then
        read -p "Coordinator host IP: " COORDINATOR_HOST
        if [[ -z "$COORDINATOR_HOST" ]]; then
            log_error "Coordinator host is required"
            exit 1
        fi
    fi
    
    # SSH configuration
    if [[ -z "$SSH_KEY_PATH" ]]; then
        read -p "SSH private key path [~/.ssh/id_rsa]: " SSH_KEY_PATH
        SSH_KEY_PATH=${SSH_KEY_PATH:-~/.ssh/id_rsa}
    fi
    
    # Expand tilde
    SSH_KEY_PATH="${SSH_KEY_PATH/#\~/$HOME}"
    
    if [[ ! -f "$SSH_KEY_PATH" ]]; then
        log_error "SSH key not found: $SSH_KEY_PATH"
        exit 1
    fi
    
    # Node configuration
    configure_cluster_nodes
}

# Configure cluster nodes interactively
configure_cluster_nodes() {
    log_info "Configuring cluster nodes..."
    
    declare -A CLUSTER_NODES
    local node_types=("data-core" "data-analytics" "message-infrastructure" "processing-core" "infrastructure-monitor")
    
    for node_type in "${node_types[@]}"; do
        echo ""
        read -p "Deploy $node_type nodes? (y/N): " deploy_type
        if [[ "$deploy_type" =~ ^[Yy]$ ]]; then
            read -p "Number of $node_type nodes [1]: " node_count
            node_count=${node_count:-1}
            
            for ((i=1; i<=node_count; i++)); do
                local node_name="${node_type}-$(printf "%02d" $i)"
                read -p "Host IP for $node_name: " node_host
                if [[ -n "$node_host" ]]; then
                    CLUSTER_NODES["$node_name"]="$node_type:$node_host"
                fi
            done
        fi
    done
    
    # Export node configuration for later use
    export CLUSTER_NODES
}

# Start cluster coordinator
start_coordinator() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would start cluster coordinator on $COORDINATOR_HOST:$COORDINATOR_PORT"
        return 0
    fi
    
    log_header "Starting Cluster Coordinator"
    
    # Check if coordinator is already running
    if curl -s "http://$COORDINATOR_HOST:$COORDINATOR_PORT/health" &>/dev/null; then
        log_info "Cluster coordinator already running"
        return 0
    fi
    
    log_info "Starting cluster coordinator on $COORDINATOR_HOST..."
    
    # Deploy coordinator to the coordinator host
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$SSH_USER@$COORDINATOR_HOST" << EOF
        set -e
        
        # Install dependencies if needed
        if ! command -v python3 &> /dev/null; then
            sudo apt update && sudo apt install -y python3 python3-pip
        fi
        
        if ! command -v docker &> /dev/null; then
            curl -fsSL https://get.docker.com | sudo sh
            sudo usermod -aG docker \$USER
        fi
        
        # Clone repository
        if [[ ! -d "/opt/bev-osint" ]]; then
            sudo mkdir -p /opt/bev-osint
            sudo chown \$USER:\$USER /opt/bev-osint
            git clone -b $GITHUB_BRANCH $GITHUB_REPO /opt/bev-osint
        fi
        
        # Start coordinator
        cd /opt/bev-osint/deployment/orchestration
        
        # Install Python dependencies
        pip3 install fastapi uvicorn aiohttp pydantic
        
        # Start coordinator in background
        nohup python3 cluster-coordinator.py > coordinator.log 2>&1 &
        
        echo "Cluster coordinator started"
EOF
    
    # Wait for coordinator to be ready
    log_info "Waiting for coordinator to start..."
    local max_wait=60
    local wait_time=0
    
    while [[ $wait_time -lt $max_wait ]]; do
        if curl -s "http://$COORDINATOR_HOST:$COORDINATOR_PORT/health" &>/dev/null; then
            log_success "Cluster coordinator is ready"
            return 0
        fi
        sleep 2
        wait_time=$((wait_time + 2))
    done
    
    log_error "Cluster coordinator failed to start within $max_wait seconds"
    return 1
}

# Deploy node to target host
deploy_node_to_host() {
    local node_name="$1"
    local node_type="$2"
    local node_host="$3"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy $node_type on $node_host as $node_name"
        return 0
    fi
    
    log_info "Deploying $node_name ($node_type) on $node_host..."
    
    # Deploy node via SSH
    ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=no "$SSH_USER@$node_host" << EOF
        set -e
        
        # Set environment variables for the bootstrap script
        export BEV_NODE_TYPE="$node_type"
        export BEV_NODE_ID="$node_name"
        export BEV_INTERACTIVE="false"
        export BEV_BRANCH="$GITHUB_BRANCH"
        export CLUSTER_COORDINATOR_URL="http://$COORDINATOR_HOST:$COORDINATOR_PORT"
        
        # Download and run bootstrap script
        curl -sSL "$BOOTSTRAP_URL" | bash
        
        echo "Node $node_name deployed successfully"
EOF
    
    if [[ $? -eq 0 ]]; then
        log_success "Successfully deployed $node_name"
    else
        log_error "Failed to deploy $node_name"
        return 1
    fi
}

# Deploy all cluster nodes
deploy_cluster_nodes() {
    log_header "Deploying Cluster Nodes"
    
    if [[ ${#CLUSTER_NODES[@]} -eq 0 ]]; then
        log_error "No nodes configured for deployment"
        exit 1
    fi
    
    # Node deployment order for dependencies
    local deployment_order=("data-core" "data-analytics" "message-infrastructure" "infrastructure-monitor" "processing-core" "specialized-processing" "ml-intelligence" "frontend-api" "edge-computing")
    
    for node_type in "${deployment_order[@]}"; do
        local type_nodes=()
        
        # Find nodes of current type
        for node_name in "${!CLUSTER_NODES[@]}"; do
            local node_info="${CLUSTER_NODES[$node_name]}"
            local current_type="${node_info%%:*}"
            if [[ "$current_type" == "$node_type" ]]; then
                type_nodes+=("$node_name")
            fi
        done
        
        if [[ ${#type_nodes[@]} -eq 0 ]]; then
            continue
        fi
        
        log_info "Deploying ${#type_nodes[@]} $node_type nodes..."
        
        if [[ "$PARALLEL_DEPLOYMENT" == "true" ]]; then
            # Deploy nodes of same type in parallel
            local pids=()
            
            for node_name in "${type_nodes[@]}"; do
                local node_info="${CLUSTER_NODES[$node_name]}"
                local node_host="${node_info##*:}"
                
                deploy_node_to_host "$node_name" "$node_type" "$node_host" &
                pids+=($!)
            done
            
            # Wait for all parallel deployments to complete
            for pid in "${pids[@]}"; do
                wait $pid
            done
        else
            # Deploy nodes sequentially
            for node_name in "${type_nodes[@]}"; do
                local node_info="${CLUSTER_NODES[$node_name]}"
                local node_host="${node_info##*:}"
                
                deploy_node_to_host "$node_name" "$node_type" "$node_host"
            done
        fi
        
        # Wait for nodes to be healthy before proceeding to next type
        if [[ "$DRY_RUN" != "true" ]]; then
            wait_for_node_type_health "$node_type"
        fi
    done
}

# Wait for node type to be healthy
wait_for_node_type_health() {
    local node_type="$1"
    local max_wait=300  # 5 minutes
    local wait_time=0
    
    log_info "Waiting for $node_type nodes to become healthy..."
    
    while [[ $wait_time -lt $max_wait ]]; do
        local cluster_status=$(curl -s "http://$COORDINATOR_HOST:$COORDINATOR_PORT/status" 2>/dev/null || echo "{}")
        
        if echo "$cluster_status" | grep -q "healthy"; then
            log_success "$node_type nodes are healthy"
            return 0
        fi
        
        sleep 10
        wait_time=$((wait_time + 10))
        log_info "Still waiting for $node_type nodes... (${wait_time}s/${max_wait}s)"
    done
    
    log_warning "Timeout waiting for $node_type nodes to become healthy"
    return 1
}

# Validate cluster deployment
validate_cluster() {
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would validate cluster deployment"
        return 0
    fi
    
    log_header "Validating Cluster Deployment"
    
    # Get cluster status
    local cluster_status=$(curl -s "http://$COORDINATOR_HOST:$COORDINATOR_PORT/status" 2>/dev/null || echo "{}")
    
    if [[ -z "$cluster_status" ]]; then
        log_error "Unable to get cluster status from coordinator"
        return 1
    fi
    
    log_info "Cluster validation completed"
    echo "$cluster_status" | python3 -m json.tool
}

# Show deployment summary
show_deployment_summary() {
    log_header "Deployment Summary"
    
    echo "Cluster Configuration:"
    echo "  Coordinator: $COORDINATOR_HOST:$COORDINATOR_PORT"
    echo "  Deployment Mode: $([ "$PARALLEL_DEPLOYMENT" == "true" ] && echo "Parallel" || echo "Sequential")"
    echo "  Branch: $GITHUB_BRANCH"
    echo ""
    
    echo "Deployed Nodes:"
    for node_name in "${!CLUSTER_NODES[@]}"; do
        local node_info="${CLUSTER_NODES[$node_name]}"
        local node_type="${node_info%%:*}"
        local node_host="${node_info##*:}"
        echo "  $node_name ($node_type) on $node_host"
    done
    
    echo ""
    echo "Management URLs:"
    echo "  Cluster Status: http://$COORDINATOR_HOST:$COORDINATOR_PORT/status"
    echo "  Node Discovery: http://$COORDINATOR_HOST:$COORDINATOR_PORT/discovery/<service>"
    echo ""
    
    if [[ "$DRY_RUN" != "true" ]]; then
        log_success "BEV OSINT cluster deployment completed!"
    else
        log_info "Dry run completed - no actual deployment performed"
    fi
}

# Main execution
main() {
    show_banner
    parse_args "$@"
    
    load_cluster_config
    start_coordinator
    deploy_cluster_nodes
    validate_cluster
    show_deployment_summary
}

# Handle interrupts gracefully
trap 'log_error "Deployment interrupted"; exit 130' INT TERM

# Run main function
main "$@"