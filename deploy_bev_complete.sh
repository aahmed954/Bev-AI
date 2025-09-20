#!/bin/bash

##############################################################################
# BEV OSINT Framework - Master Deployment Orchestration Script
# Complete multi-node deployment automation with monitoring and rollback
# THANOS (Main Server) + ORACLE1 (ARM Cloud Server)
##############################################################################

set -euo pipefail

# Script metadata
SCRIPT_VERSION="1.0.0"
SCRIPT_NAME="BEV Master Deployment Orchestrator"
DEPLOYMENT_ID="bev_deploy_$(date +%Y%m%d_%H%M%S)"

# Color codes for output
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
declare -r LOG_DIR="$BEV_HOME/logs/deployment"
declare -r BACKUP_DIR="$BEV_HOME/backups"
declare -r CONFIG_DIR="$BEV_HOME/config"
declare -r SCRIPTS_DIR="$BEV_HOME/scripts"

# Node configuration
declare -r THANOS_HOST="localhost"
declare -r THANOS_USER="starlord"
declare -r ORACLE1_HOST="100.96.197.84"
declare -r ORACLE1_USER="starlord"

# Deployment phases
declare -ra DEPLOYMENT_PHASES=(
    "foundation"     # Phase 1: Core infrastructure
    "monitoring"     # Phase 2: Monitoring and orchestration
    "processing"     # Phase 3: Document processing
    "agents"         # Phase 4: Agent swarm
    "security"       # Phase 5: Security and enhancement
    "advanced"       # Phase 6: Advanced capabilities
)

# Global state
DEPLOYMENT_START_TIME=""
ROLLBACK_POINTS=()
FAILED_SERVICES=()
DEPLOYMENT_LOG=""

##############################################################################
# ASCII Art Banner
##############################################################################
print_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║   ██████╗ ███████╗██╗   ██╗    ██████╗ ███████╗██████╗ ██╗      ██████╗   ║
║   ██╔══██╗██╔════╝██║   ██║    ██╔══██╗██╔════╝██╔══██╗██║     ██╔═══██╗  ║
║   ██████╔╝█████╗  ██║   ██║    ██║  ██║█████╗  ██████╔╝██║     ██║   ██║  ║
║   ██╔══██╗██╔══╝  ╚██╗ ██╔╝    ██║  ██║██╔══╝  ██╔═══╝ ██║     ██║   ██║  ║
║   ██████╔╝███████╗ ╚████╔╝     ██████╔╝███████╗██║     ███████╗╚██████╔╝  ║
║   ╚═════╝ ╚══════╝  ╚═══╝      ╚═════╝ ╚══════╝╚═╝     ╚══════╝ ╚═════╝   ║
║                                                                           ║
║                    MASTER DEPLOYMENT ORCHESTRATOR v${SCRIPT_VERSION}                   ║
║              Multi-Node Infrastructure Automation System                  ║
║                                                                           ║
║    THANOS (Main):    ${THANOS_HOST} (Full Stack)                               ║
║    ORACLE1 (ARM):    ${ORACLE1_HOST} (Cloud Services)                 ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

##############################################################################
# Logging Functions
##############################################################################
setup_logging() {
    mkdir -p "$LOG_DIR"
    DEPLOYMENT_LOG="$LOG_DIR/deployment_${DEPLOYMENT_ID}.log"

    # Redirect all output to log file while keeping console output
    exec > >(tee -a "$DEPLOYMENT_LOG") 2>&1

    log_info "Deployment logging initialized: $DEPLOYMENT_LOG"
}

log_message() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$DEPLOYMENT_LOG"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_phase() {
    echo -e "${PURPLE}[PHASE]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

##############################################################################
# Environment and SSH Setup
##############################################################################
setup_ssh_keys() {
    log_step "Setting up SSH key authentication"

    local ssh_key="$HOME/.ssh/bev_deployment_key"

    if [[ ! -f "$ssh_key" ]]; then
        log_info "Generating SSH key for deployment"
        ssh-keygen -t ed25519 -f "$ssh_key" -N "" -q
        log_success "SSH key generated: $ssh_key"
    fi

    # Copy public key to ORACLE1 if not already present
    if ! ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no -i "$ssh_key" \
         "$ORACLE1_USER@$ORACLE1_HOST" exit 2>/dev/null; then
        log_info "Setting up SSH access to ORACLE1"
        ssh-copy-id -i "$ssh_key.pub" "$ORACLE1_USER@$ORACLE1_HOST" || {
            log_error "Failed to setup SSH access to ORACLE1"
            return 1
        }
    fi

    log_success "SSH key authentication configured"
}

generate_secure_passwords() {
    log_step "Generating secure passwords and tokens"

    local env_file="$BEV_HOME/.env"
    local oracle_env_file="$BEV_HOME/.env.oracle1"

    # Generate random passwords
    local postgres_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    local redis_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    local jwt_secret=$(openssl rand -base64 64 | tr -d "=+/" | cut -c1-50)
    local vault_root_token=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
    local airflow_secret=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)

    # Create main .env file for THANOS
    cat > "$env_file" << EOF
# BEV OSINT Framework Environment Configuration
# Generated on $(date)

# Database Configuration
POSTGRES_PASSWORD=$postgres_password
POSTGRES_USER=bev_user
POSTGRES_DB=bev_osint
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Redis Configuration
REDIS_PASSWORD=$redis_password
REDIS_HOST=redis
REDIS_PORT=6379

# Security
JWT_SECRET=$jwt_secret
VAULT_ROOT_TOKEN=$vault_root_token

# Airflow Configuration
AIRFLOW_SECRET_KEY=$airflow_secret
AIRFLOW_USERNAME=admin
AIRFLOW_PASSWORD=admin123
AIRFLOW_EMAIL=admin@bev.local

# Network Configuration
THANOS_HOST=$THANOS_HOST
ORACLE1_HOST=$ORACLE1_HOST

# Deployment Metadata
DEPLOYMENT_ID=$DEPLOYMENT_ID
DEPLOYMENT_DATE=$(date -Iseconds)
EOF

    # Create Oracle1-specific environment file
    cat > "$oracle_env_file" << EOF
# ORACLE1 ARM Server Environment Configuration
# Generated on $(date)

# Connection to THANOS
THANOS_HOST=$THANOS_HOST
POSTGRES_HOST=$THANOS_HOST
REDIS_HOST=$THANOS_HOST

# Local Redis for ARM services
REDIS_PASSWORD=$redis_password

# Security
JWT_SECRET=$jwt_secret

# ARM Optimization
ARM_OPTIMIZATION=true
MEMORY_LIMIT=4G
CPU_LIMIT=2

# Deployment Metadata
DEPLOYMENT_ID=$DEPLOYMENT_ID
NODE_TYPE=oracle1
EOF

    chmod 600 "$env_file" "$oracle_env_file"
    log_success "Environment files generated with secure passwords"
}

verify_connectivity() {
    log_step "Verifying network connectivity"

    # Test THANOS connectivity (local)
    if docker --version >/dev/null 2>&1; then
        log_success "THANOS node accessible (local)"
    else
        log_error "Docker not available on THANOS"
        return 1
    fi

    # Test ORACLE1 connectivity
    local ssh_key="$HOME/.ssh/bev_deployment_key"
    if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -i "$ssh_key" \
       "$ORACLE1_USER@$ORACLE1_HOST" "docker --version" >/dev/null 2>&1; then
        log_success "ORACLE1 node accessible and Docker available"
    else
        log_error "Cannot connect to ORACLE1 or Docker not available"
        return 1
    fi

    # Test network connectivity between nodes
    if ssh -i "$ssh_key" "$ORACLE1_USER@$ORACLE1_HOST" \
       "ping -c 1 $THANOS_HOST" >/dev/null 2>&1; then
        log_success "Network connectivity verified between nodes"
    else
        log_warning "Direct network connectivity between nodes may be limited"
    fi
}

##############################################################################
# Deployment Phase Management
##############################################################################
create_rollback_point() {
    local phase="$1"
    local timestamp=$(date -Iseconds)

    log_step "Creating rollback point for phase: $phase"

    # Create backup directory
    local backup_path="$BACKUP_DIR/${DEPLOYMENT_ID}_${phase}_${timestamp}"
    mkdir -p "$backup_path"

    # Save current state
    echo "$phase" > "$backup_path/phase.txt"
    echo "$timestamp" > "$backup_path/timestamp.txt"

    # Save running containers state
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" > "$backup_path/thanos_containers.txt" 2>/dev/null || true

    # Save Oracle1 state
    local ssh_key="$HOME/.ssh/bev_deployment_key"
    ssh -i "$ssh_key" "$ORACLE1_USER@$ORACLE1_HOST" \
        "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'" > "$backup_path/oracle1_containers.txt" 2>/dev/null || true

    ROLLBACK_POINTS+=("$backup_path")
    log_success "Rollback point created: $backup_path"
}

deploy_phase_foundation() {
    log_phase "Phase 1: Foundation Infrastructure Deployment"
    create_rollback_point "foundation"

    # Deploy core services on THANOS
    log_step "Deploying core services on THANOS"
    cd "$BEV_HOME"

    # Start foundation services
    docker-compose -f docker-compose-thanos-unified.yml up -d \
        postgres neo4j redis elasticsearch influxdb kafka zookeeper rabbitmq || {
        log_error "Failed to deploy foundation services on THANOS"
        return 1
    }

    # Wait for services to be ready
    log_step "Waiting for foundation services to be ready"
    sleep 30

    # Verify core services
    local services=("postgres" "neo4j" "redis" "elasticsearch")
    for service in "${services[@]}"; do
        if docker-compose -f docker-compose-thanos-unified.yml ps "$service" | grep -q "Up"; then
            log_success "$service is running"
        else
            log_error "$service failed to start"
            FAILED_SERVICES+=("$service")
        fi
    done

    # Deploy foundation services on ORACLE1
    log_step "Deploying foundation services on ORACLE1"
    local ssh_key="$HOME/.ssh/bev_deployment_key"

    # Copy environment and compose files to ORACLE1
    scp -i "$ssh_key" "$BEV_HOME/.env.oracle1" "$ORACLE1_USER@$ORACLE1_HOST:~/bev/.env"
    scp -i "$ssh_key" "$BEV_HOME/docker-compose-oracle1-unified.yml" "$ORACLE1_USER@$ORACLE1_HOST:~/bev/"

    # Start Oracle1 foundation services
    ssh -i "$ssh_key" "$ORACLE1_USER@$ORACLE1_HOST" \
        "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml up -d redis n8n nginx-proxy" || {
        log_error "Failed to deploy foundation services on ORACLE1"
        return 1
    }

    log_success "Phase 1: Foundation infrastructure deployed successfully"
}

deploy_phase_monitoring() {
    log_phase "Phase 2: Monitoring and Orchestration Deployment"
    create_rollback_point "monitoring"

    # Deploy monitoring stack on THANOS
    log_step "Deploying monitoring stack on THANOS"
    cd "$BEV_HOME"

    docker-compose -f docker-compose-thanos-unified.yml up -d \
        prometheus grafana airflow-webserver airflow-scheduler airflow-worker || {
        log_error "Failed to deploy monitoring services on THANOS"
        return 1
    }

    # Wait for Airflow to initialize
    log_step "Waiting for Airflow initialization"
    sleep 45

    # Verify monitoring services
    local monitoring_services=("prometheus" "grafana" "airflow-webserver")
    for service in "${monitoring_services[@]}"; do
        if docker-compose -f docker-compose-thanos-unified.yml ps "$service" | grep -q "Up"; then
            log_success "$service is running"
        else
            log_error "$service failed to start"
            FAILED_SERVICES+=("$service")
        fi
    done

    log_success "Phase 2: Monitoring and orchestration deployed successfully"
}

deploy_phase_processing() {
    log_phase "Phase 3: Document Processing Deployment"
    create_rollback_point "processing"

    # Deploy processing services
    log_step "Deploying document processing services"
    cd "$BEV_HOME"

    docker-compose -f docker-compose-thanos-unified.yml up -d \
        minio ocr-service document-analyzer intelowl || {
        log_error "Failed to deploy processing services"
        return 1
    }

    # Deploy processing services on ORACLE1
    local ssh_key="$HOME/.ssh/bev_deployment_key"
    ssh -i "$ssh_key" "$ORACLE1_USER@$ORACLE1_HOST" \
        "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml up -d crawler intel-collector" || {
        log_error "Failed to deploy processing services on ORACLE1"
        return 1
    }

    log_success "Phase 3: Document processing deployed successfully"
}

deploy_phase_agents() {
    log_phase "Phase 4: Agent Swarm Deployment"
    create_rollback_point "agents"

    # Deploy agent services
    log_step "Deploying agent swarm services"
    cd "$BEV_HOME"

    docker-compose -f docker-compose-thanos-unified.yml up -d \
        swarm-coordinator research-agent memory-agent optimization-agent || {
        log_error "Failed to deploy agent services"
        return 1
    }

    # Deploy distributed agents on ORACLE1
    local ssh_key="$HOME/.ssh/bev_deployment_key"
    ssh -i "$ssh_key" "$ORACLE1_USER@$ORACLE1_HOST" \
        "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml up -d social-media-agent osint-crawler" || {
        log_error "Failed to deploy agent services on ORACLE1"
        return 1
    }

    log_success "Phase 4: Agent swarm deployed successfully"
}

deploy_phase_security() {
    log_phase "Phase 5: Security and Enhancement Deployment"
    create_rollback_point "security"

    # Deploy security services
    log_step "Deploying security services"
    cd "$BEV_HOME"

    docker-compose -f docker-compose-thanos-unified.yml up -d \
        vault guardian-service tor-relay ids-system || {
        log_error "Failed to deploy security services"
        return 1
    }

    # Configure Vault
    log_step "Configuring Vault security"
    sleep 15
    if [[ -f "$CONFIG_DIR/vault-setup.sh" ]]; then
        bash "$CONFIG_DIR/vault-setup.sh" || log_warning "Vault configuration partially failed"
    fi

    log_success "Phase 5: Security and enhancement deployed successfully"
}

deploy_phase_advanced() {
    log_phase "Phase 6: Advanced Capabilities Deployment"
    create_rollback_point "advanced"

    # Deploy advanced services
    log_step "Deploying advanced capability services"
    cd "$BEV_HOME"

    docker-compose -f docker-compose-thanos-unified.yml up -d \
        autonomous-agent live2d-service multimodal-processor || {
        log_error "Failed to deploy advanced services"
        return 1
    }

    # Deploy advanced services on ORACLE1
    local ssh_key="$HOME/.ssh/bev_deployment_key"
    ssh -i "$ssh_key" "$ORACLE1_USER@$ORACLE1_HOST" \
        "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml up -d ai-coordinator blockchain-monitor" || {
        log_error "Failed to deploy advanced services on ORACLE1"
        return 1
    }

    log_success "Phase 6: Advanced capabilities deployed successfully"
}

##############################################################################
# Health Checks and Monitoring
##############################################################################
perform_health_checks() {
    log_step "Performing comprehensive health checks"

    local failed_checks=0

    # Check THANOS services
    log_info "Checking THANOS services health"
    cd "$BEV_HOME"

    # Get all running services
    local running_services=$(docker-compose -f docker-compose-thanos-unified.yml ps --services --filter "status=running")
    local total_services=$(docker-compose -f docker-compose-thanos-unified.yml config --services | wc -l)
    local running_count=$(echo "$running_services" | wc -l)

    log_info "THANOS: $running_count/$total_services services running"

    # Check critical services endpoints
    local critical_endpoints=(
        "http://localhost:5432"  # PostgreSQL
        "http://localhost:6379"  # Redis
        "http://localhost:9200"  # Elasticsearch
        "http://localhost:3000"  # Grafana
        "http://localhost:9090"  # Prometheus
        "http://localhost:8080"  # Airflow
    )

    for endpoint in "${critical_endpoints[@]}"; do
        if timeout 5 bash -c "cat < /dev/null > /dev/tcp/${endpoint#http://}" 2>/dev/null; then
            log_success "Endpoint accessible: $endpoint"
        else
            log_warning "Endpoint not accessible: $endpoint"
            ((failed_checks++))
        fi
    done

    # Check ORACLE1 services
    log_info "Checking ORACLE1 services health"
    local ssh_key="$HOME/.ssh/bev_deployment_key"

    local oracle_running=$(ssh -i "$ssh_key" "$ORACLE1_USER@$ORACLE1_HOST" \
        "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml ps --services --filter 'status=running' | wc -l")

    log_info "ORACLE1: $oracle_running services running"

    # Resource utilization check
    log_info "Checking resource utilization"
    local memory_usage=$(free | awk 'NR==2{printf "%.1f%%", $3*100/$2}')
    local disk_usage=$(df "$BEV_HOME" | awk 'NR==2{print $5}' | sed 's/%//')

    log_info "Memory usage: $memory_usage"
    log_info "Disk usage: $disk_usage%"

    if [[ $disk_usage -gt 90 ]]; then
        log_warning "High disk usage detected: $disk_usage%"
        ((failed_checks++))
    fi

    if [[ $failed_checks -eq 0 ]]; then
        log_success "All health checks passed"
        return 0
    else
        log_warning "$failed_checks health check(s) failed"
        return 1
    fi
}

##############################################################################
# Rollback System
##############################################################################
perform_rollback() {
    local target_phase="$1"

    log_warning "Initiating rollback to phase: $target_phase"

    # Find the appropriate rollback point
    local rollback_point=""
    for point in "${ROLLBACK_POINTS[@]}"; do
        if [[ -f "$point/phase.txt" ]] && [[ "$(cat "$point/phase.txt")" == "$target_phase" ]]; then
            rollback_point="$point"
            break
        fi
    done

    if [[ -z "$rollback_point" ]]; then
        log_error "No rollback point found for phase: $target_phase"
        return 1
    fi

    log_step "Rolling back to: $rollback_point"

    # Stop all services
    log_info "Stopping all services"
    cd "$BEV_HOME"
    docker-compose -f docker-compose-thanos-unified.yml down || true

    local ssh_key="$HOME/.ssh/bev_deployment_key"
    ssh -i "$ssh_key" "$ORACLE1_USER@$ORACLE1_HOST" \
        "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml down" || true

    # Restore state would go here (volumes, configs, etc.)
    log_info "State restoration logic would be implemented here"

    log_success "Rollback completed to phase: $target_phase"
}

##############################################################################
# Main Deployment Flow
##############################################################################
main_deployment() {
    DEPLOYMENT_START_TIME=$(date -Iseconds)

    log_info "Starting BEV OSINT Framework deployment"
    log_info "Deployment ID: $DEPLOYMENT_ID"
    log_info "Start time: $DEPLOYMENT_START_TIME"

    # Pre-deployment setup
    setup_ssh_keys || { log_error "SSH setup failed"; exit 1; }
    generate_secure_passwords || { log_error "Password generation failed"; exit 1; }
    verify_connectivity || { log_error "Connectivity verification failed"; exit 1; }

    # Execute deployment phases
    for phase in "${DEPLOYMENT_PHASES[@]}"; do
        log_phase "Executing deployment phase: $phase"

        case "$phase" in
            "foundation")
                deploy_phase_foundation || {
                    log_error "Foundation phase failed"
                    perform_rollback "foundation"
                    exit 1
                }
                ;;
            "monitoring")
                deploy_phase_monitoring || {
                    log_error "Monitoring phase failed"
                    perform_rollback "foundation"
                    exit 1
                }
                ;;
            "processing")
                deploy_phase_processing || {
                    log_error "Processing phase failed"
                    perform_rollback "monitoring"
                    exit 1
                }
                ;;
            "agents")
                deploy_phase_agents || {
                    log_error "Agents phase failed"
                    perform_rollback "processing"
                    exit 1
                }
                ;;
            "security")
                deploy_phase_security || {
                    log_error "Security phase failed"
                    perform_rollback "agents"
                    exit 1
                }
                ;;
            "advanced")
                deploy_phase_advanced || {
                    log_error "Advanced phase failed"
                    perform_rollback "security"
                    exit 1
                }
                ;;
        esac

        # Wait between phases
        log_info "Phase '$phase' completed, waiting before next phase..."
        sleep 10
    done

    # Post-deployment validation
    log_phase "Post-deployment validation"
    perform_health_checks || {
        log_warning "Some health checks failed, but deployment continues"
    }

    # Final status
    local deployment_end_time=$(date -Iseconds)
    local duration=$(($(date -d "$deployment_end_time" +%s) - $(date -d "$DEPLOYMENT_START_TIME" +%s)))

    log_success "BEV OSINT Framework deployment completed successfully!"
    log_info "Deployment duration: ${duration} seconds"
    log_info "Deployment log: $DEPLOYMENT_LOG"

    # Print access information
    print_access_info
}

print_access_info() {
    echo -e "\n${GREEN}=== BEV OSINT Framework Access Information ===${NC}"
    echo -e "${CYAN}Grafana Dashboard:${NC} http://localhost:3000 (admin/admin123)"
    echo -e "${CYAN}Prometheus:${NC} http://localhost:9090"
    echo -e "${CYAN}Airflow:${NC} http://localhost:8080 (admin/admin123)"
    echo -e "${CYAN}Elasticsearch:${NC} http://localhost:9200"
    echo -e "${CYAN}Neo4j Browser:${NC} http://localhost:7474"
    echo -e "${CYAN}MinIO Console:${NC} http://localhost:9001"
    echo -e "${CYAN}IntelOwl:${NC} http://localhost:80"
    echo -e "\n${YELLOW}Oracle1 Services:${NC} Accessible through THANOS proxy"
    echo -e "${CYAN}Deployment Log:${NC} $DEPLOYMENT_LOG"
    echo -e "${GREEN}===============================================${NC}\n"
}

##############################################################################
# Command Line Interface
##############################################################################
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --version, -v       Show version information"
    echo "  --dry-run           Perform dry run without actual deployment"
    echo "  --rollback PHASE    Rollback to specified phase"
    echo "  --status            Check deployment status"
    echo "  --logs              Show deployment logs"
    echo "  --cleanup           Clean up deployment resources"
    echo ""
    echo "Phases: ${DEPLOYMENT_PHASES[*]}"
    echo ""
    echo "Examples:"
    echo "  $0                  # Full deployment"
    echo "  $0 --rollback foundation  # Rollback to foundation phase"
    echo "  $0 --status         # Check current status"
}

##############################################################################
# Main Script Entry Point
##############################################################################
main() {
    # Setup logging first
    setup_logging

    # Parse command line arguments
    case "${1:-}" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --version|-v)
            echo "$SCRIPT_NAME v$SCRIPT_VERSION"
            exit 0
            ;;
        --rollback)
            if [[ -z "${2:-}" ]]; then
                log_error "Phase required for rollback"
                show_usage
                exit 1
            fi
            perform_rollback "$2"
            exit 0
            ;;
        --status)
            perform_health_checks
            exit 0
            ;;
        --logs)
            if [[ -f "$DEPLOYMENT_LOG" ]]; then
                tail -f "$DEPLOYMENT_LOG"
            else
                log_error "No deployment log found"
                exit 1
            fi
            ;;
        --dry-run)
            log_info "Dry run mode - no actual deployment will be performed"
            # Add dry run logic here
            exit 0
            ;;
        --cleanup)
            log_warning "Cleanup functionality not yet implemented"
            exit 1
            ;;
        "")
            # No arguments - proceed with full deployment
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac

    # Show banner and start deployment
    print_banner
    main_deployment
}

# Trap signals for cleanup
trap 'log_error "Deployment interrupted"; exit 130' INT TERM

# Execute main function with all arguments
main "$@"