#!/bin/bash

#################################################################
# BEV OSINT Framework - Node Bootstrap Script
#
# This script can be run on any machine to bootstrap a BEV node
# It clones the repository and deploys the specified node type
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/aahmed954/Bev-AI/main/deployment/bootstrap/node-bootstrap.sh | bash
#   or
#   curl -sSL https://raw.githubusercontent.com/aahmed954/Bev-AI/main/deployment/bootstrap/node-bootstrap.sh | bash -s -- --node-type data-core
#
#################################################################

set -euo pipefail

# Default configuration
GITHUB_REPO="https://github.com/aahmed954/Bev-AI.git"
GITHUB_BRANCH="${BEV_BRANCH:-main}"
INSTALL_DIR="${BEV_INSTALL_DIR:-/opt/bev-osint}"
NODE_TYPE="${BEV_NODE_TYPE:-}"
NODE_ID="${BEV_NODE_ID:-}"
INTERACTIVE="${BEV_INTERACTIVE:-true}"

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

# Show welcome banner
show_banner() {
    cat << 'EOF'

 ██████╗ ███████╗██╗   ██╗     ██████╗ ███████╗██╗███╗   ██╗████████╗
 ██╔══██╗██╔════╝██║   ██║    ██╔═══██╗██╔════╝██║████╗  ██║╚══██╔══╝
 ██████╔╝█████╗  ██║   ██║    ██║   ██║███████╗██║██╔██╗ ██║   ██║   
 ██╔══██╗██╔══╝  ╚██╗ ██╔╝    ██║   ██║╚════██║██║██║╚██╗██║   ██║   
 ██████╔╝███████╗ ╚████╔╝     ╚██████╔╝███████║██║██║ ╚████║   ██║   
 ╚═════╝ ╚══════╝  ╚═══╝       ╚═════╝ ╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝   

    Distributed OSINT Framework - Node Bootstrap Script

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --node-type)
                NODE_TYPE="$2"
                shift 2
                ;;
            --node-id)
                NODE_ID="$2"
                shift 2
                ;;
            --install-dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            --branch)
                GITHUB_BRANCH="$2"
                shift 2
                ;;
            --non-interactive)
                INTERACTIVE="false"
                shift
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

# Show help information
show_help() {
    cat << EOF
BEV OSINT Framework - Node Bootstrap Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --node-type TYPE         Specify node type to deploy
    --node-id ID            Specify unique node identifier
    --install-dir DIR       Installation directory (default: /opt/bev-osint)
    --branch BRANCH         Git branch to use (default: main)
    --non-interactive       Run without user prompts
    --help                  Show this help message

AVAILABLE NODE TYPES:
    data-core              Core data storage (PostgreSQL, Neo4j, Redis, InfluxDB)
    data-analytics         Analytics databases (Elasticsearch, Qdrant, Weaviate)
    message-infrastructure Message queuing (RabbitMQ, Kafka, Zookeeper)
    processing-core        Core OSINT processing (IntelOwl, MCP server)
    specialized-processing Specialized analyzers and intelligence services
    infrastructure-monitor Monitoring and infrastructure services
    ml-intelligence        Machine learning and AI services
    frontend-api           User interfaces and API gateways
    edge-computing         Geographic edge nodes

EXAMPLES:
    # Interactive deployment
    curl -sSL <bootstrap-url> | bash

    # Deploy specific node type
    curl -sSL <bootstrap-url> | bash -s -- --node-type data-core

    # Deploy with custom configuration
    export BEV_NODE_TYPE=processing-core
    export BEV_NODE_ID=proc-core-west-01
    curl -sSL <bootstrap-url> | bash

ENVIRONMENT VARIABLES:
    BEV_NODE_TYPE          Node type to deploy
    BEV_NODE_ID           Unique node identifier
    BEV_INSTALL_DIR       Installation directory
    BEV_BRANCH            Git branch to use
    BEV_INTERACTIVE       Enable/disable interactive mode (true/false)

EOF
}

# Detect operating system and architecture
detect_system() {
    log_info "Detecting system information..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        log_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64) ARCH="amd64" ;;
        aarch64|arm64) ARCH="arm64" ;;
        *) 
            log_error "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac
    
    # Get system info
    TOTAL_RAM=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "unknown")
    TOTAL_DISK=$(df -BG / 2>/dev/null | awk 'NR==2 {print $2}' | sed 's/G//' || echo "unknown")
    CPU_CORES=$(nproc 2>/dev/null || echo "unknown")
    
    log_success "System detected: $OS/$ARCH"
    log_info "System specs: ${CPU_CORES} cores, ${TOTAL_RAM}GB RAM, ${TOTAL_DISK}GB disk"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    local required_tools=("git" "docker" "curl")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null && ! docker-compose version &> /dev/null; then
        missing_tools+=("docker-compose")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and run this script again"
        
        # Provide installation hints
        case $OS in
            "linux")
                log_info "Installation hints for Ubuntu/Debian:"
                echo "  sudo apt update"
                echo "  sudo apt install -y git curl"
                echo "  # Install Docker: https://docs.docker.com/engine/install/"
                ;;
            "macos")
                log_info "Installation hints for macOS:"
                echo "  brew install git curl"
                echo "  # Install Docker Desktop: https://docs.docker.com/desktop/mac/install/"
                ;;
        esac
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    
    log_success "All prerequisites are met"
}

# Interactive node type selection
select_node_type() {
    if [[ -n "$NODE_TYPE" ]]; then
        log_info "Using specified node type: $NODE_TYPE"
        return 0
    fi
    
    if [[ "$INTERACTIVE" != "true" ]]; then
        log_error "Node type not specified and running in non-interactive mode"
        log_info "Use --node-type or set BEV_NODE_TYPE environment variable"
        exit 1
    fi
    
    log_header "Node Type Selection"
    
    echo "Available node types:"
    echo ""
    echo "  1) data-core              - Core data storage (32+ GB RAM)"
    echo "  2) data-analytics         - Analytics databases (16+ GB RAM)"
    echo "  3) message-infrastructure - Message queuing (8+ GB RAM)"
    echo "  4) processing-core        - Core OSINT processing (16+ GB RAM)"
    echo "  5) specialized-processing - Specialized analyzers (8-16 GB RAM)"
    echo "  6) infrastructure-monitor - Monitoring services (8+ GB RAM)"
    echo "  7) ml-intelligence        - ML and AI services (16+ GB RAM)"
    echo "  8) frontend-api           - User interfaces (4-8 GB RAM)"
    echo "  9) edge-computing         - Geographic edge nodes (4-8 GB RAM)"
    echo ""
    
    # Show recommendations based on system specs
    if [[ "$TOTAL_RAM" != "unknown" ]]; then
        local ram_num=$(echo "$TOTAL_RAM" | grep -o '[0-9]*')
        if [[ -n "$ram_num" ]]; then
            echo "Recommendations for your system (${TOTAL_RAM}GB RAM):"
            if [[ $ram_num -ge 32 ]]; then
                echo "  ✅ All node types supported"
            elif [[ $ram_num -ge 16 ]]; then
                echo "  ✅ data-analytics, processing-core, ml-intelligence"
                echo "  ⚠️  data-core needs more RAM (32+ GB recommended)"
            elif [[ $ram_num -ge 8 ]]; then
                echo "  ✅ message-infrastructure, specialized-processing, infrastructure-monitor"
                echo "  ⚠️  Other types need more RAM"
            else
                echo "  ✅ frontend-api, edge-computing"
                echo "  ⚠️  Most types need more RAM"
            fi
            echo ""
        fi
    fi
    
    while true; do
        read -p "Select node type (1-9): " choice
        case $choice in
            1) NODE_TYPE="data-core"; break ;;
            2) NODE_TYPE="data-analytics"; break ;;
            3) NODE_TYPE="message-infrastructure"; break ;;
            4) NODE_TYPE="processing-core"; break ;;
            5) NODE_TYPE="specialized-processing"; break ;;
            6) NODE_TYPE="infrastructure-monitor"; break ;;
            7) NODE_TYPE="ml-intelligence"; break ;;
            8) NODE_TYPE="frontend-api"; break ;;
            9) NODE_TYPE="edge-computing"; break ;;
            *) echo "Invalid choice. Please select 1-9." ;;
        esac
    done
    
    log_success "Selected node type: $NODE_TYPE"
}

# Generate node ID if not provided
generate_node_id() {
    if [[ -n "$NODE_ID" ]]; then
        log_info "Using specified node ID: $NODE_ID"
        return 0
    fi
    
    # Generate based on node type and hostname
    local hostname_short=$(hostname -s 2>/dev/null || echo "node")
    local random_suffix=$(openssl rand -hex 3 2>/dev/null || echo "$(date +%s | tail -c 4)")
    
    NODE_ID="${NODE_TYPE}-${hostname_short}-${random_suffix}"
    
    if [[ "$INTERACTIVE" == "true" ]]; then
        read -p "Node ID [$NODE_ID]: " user_node_id
        if [[ -n "$user_node_id" ]]; then
            NODE_ID="$user_node_id"
        fi
    fi
    
    log_success "Using node ID: $NODE_ID"
}

# Clone repository
clone_repository() {
    log_header "Repository Setup"
    
    # Create installation directory
    if [[ ! -d "$INSTALL_DIR" ]]; then
        log_info "Creating installation directory: $INSTALL_DIR"
        sudo mkdir -p "$INSTALL_DIR"
        sudo chown "$USER:$(id -gn)" "$INSTALL_DIR"
    fi
    
    # Clone or update repository
    if [[ -d "$INSTALL_DIR/.git" ]]; then
        log_info "Updating existing repository..."
        cd "$INSTALL_DIR"
        git fetch origin
        git checkout "$GITHUB_BRANCH"
        git pull origin "$GITHUB_BRANCH"
    else
        log_info "Cloning repository from $GITHUB_REPO (branch: $GITHUB_BRANCH)..."
        git clone -b "$GITHUB_BRANCH" "$GITHUB_REPO" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
    
    log_success "Repository ready at $INSTALL_DIR"
}

# Validate node type
validate_node_type() {
    local node_config_dir="$INSTALL_DIR/deployment/node-configs/$NODE_TYPE"
    
    if [[ ! -d "$node_config_dir" ]]; then
        log_error "Invalid node type: $NODE_TYPE"
        log_info "Available node types:"
        find "$INSTALL_DIR/deployment/node-configs" -maxdepth 1 -type d -exec basename {} \; | grep -v node-configs | sort
        exit 1
    fi
    
    if [[ ! -f "$node_config_dir/docker-compose.yml" ]]; then
        log_error "Node configuration incomplete: missing docker-compose.yml for $NODE_TYPE"
        exit 1
    fi
    
    log_success "Node type validated: $NODE_TYPE"
}

# Setup node configuration
setup_node_config() {
    log_header "Node Configuration"
    
    local node_config_dir="$INSTALL_DIR/deployment/node-configs/$NODE_TYPE"
    local node_instance_dir="$INSTALL_DIR/instances/$NODE_ID"
    
    # Create instance directory
    mkdir -p "$node_instance_dir"
    
    # Copy node configuration
    log_info "Setting up node configuration..."
    cp -r "$node_config_dir"/* "$node_instance_dir/"
    
    # Setup environment file
    if [[ -f "$node_instance_dir/.env.template" ]]; then
        if [[ ! -f "$node_instance_dir/.env" ]]; then
            log_info "Creating environment configuration from template..."
            cp "$node_instance_dir/.env.template" "$node_instance_dir/.env"
            
            # Update node-specific values
            sed -i "s/BEV_NODE_TYPE=.*/BEV_NODE_TYPE=$NODE_TYPE/" "$node_instance_dir/.env"
            sed -i "s/BEV_NODE_ID=.*/BEV_NODE_ID=$NODE_ID/" "$node_instance_dir/.env"
            
            log_warning "Please edit $node_instance_dir/.env with your specific configuration"
            log_warning "Pay special attention to passwords and connection settings"
        else
            log_info "Environment file already exists, not overwriting"
        fi
    fi
    
    log_success "Node configuration ready at $node_instance_dir"
}

# Deploy node
deploy_node() {
    log_header "Node Deployment"
    
    local node_instance_dir="$INSTALL_DIR/instances/$NODE_ID"
    cd "$node_instance_dir"
    
    # Check if deploy script exists
    if [[ -f "./deploy.sh" ]]; then
        log_info "Running node-specific deployment script..."
        chmod +x ./deploy.sh
        ./deploy.sh
    else
        log_info "Running generic deployment with docker-compose..."
        
        # Pull images
        docker compose pull
        
        # Start services
        docker compose up -d
        
        # Wait for services to be ready
        log_info "Waiting for services to start..."
        sleep 30
        
        # Check status
        if docker compose ps | grep -q "unhealthy\|exited"; then
            log_warning "Some services may not be healthy. Check with: docker compose ps"
        else
            log_success "All services appear to be running"
        fi
    fi
}

# Show deployment summary
show_summary() {
    log_header "Deployment Summary"
    
    echo "Node Type: $NODE_TYPE"
    echo "Node ID: $NODE_ID"
    echo "Installation Directory: $INSTALL_DIR"
    echo "Instance Directory: $INSTALL_DIR/instances/$NODE_ID"
    echo ""
    echo "Management Commands:"
    echo "  cd $INSTALL_DIR/instances/$NODE_ID"
    echo "  docker compose ps                 # Check service status"
    echo "  docker compose logs -f            # View logs"
    echo "  docker compose down               # Stop services"
    echo "  docker compose up -d              # Start services"
    echo ""
    echo "For detailed documentation, see:"
    echo "  $INSTALL_DIR/docs/"
    echo ""
    
    log_success "BEV OSINT node deployment completed!"
}

# Main execution
main() {
    show_banner
    parse_args "$@"
    
    detect_system
    check_prerequisites
    select_node_type
    generate_node_id
    clone_repository
    validate_node_type
    setup_node_config
    deploy_node
    show_summary
}

# Handle interrupts gracefully
trap 'log_error "Script interrupted"; exit 130' INT TERM

# Run main function with all arguments
main "$@"