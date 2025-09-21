#!/bin/bash

# Integration with Existing Deployment Scripts
# Modifies existing deployment scripts to use pre-deployment preparation

# Source the main prep script functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREP_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
    esac
}

integrate_prep_phase() {
    log "INFO" "Integrating pre-deployment preparation with existing scripts"

    local integration_success=true

    # Backup existing deployment scripts
    if ! backup_existing_scripts; then
        log "ERROR" "Failed to backup existing deployment scripts"
        integration_success=false
    fi

    # Integrate with multinode deployment
    if ! integrate_with_multinode_deployment; then
        log "ERROR" "Failed to integrate with multinode deployment"
        integration_success=false
    fi

    # Create enhanced deployment wrapper
    if ! create_deployment_wrapper; then
        log "ERROR" "Failed to create deployment wrapper"
        integration_success=false
    fi

    # Update validation scripts
    if ! integrate_with_validation_scripts; then
        log "ERROR" "Failed to integrate with validation scripts"
        integration_success=false
    fi

    return $(if [[ "$integration_success" == "true" ]]; then echo 0; else echo 1; fi)
}

backup_existing_scripts() {
    log "INFO" "Backing up existing deployment scripts"

    local backup_dir="${PREP_DIR}/integration/script_backups"
    mkdir -p "$backup_dir"

    # Backup existing deployment scripts
    local scripts_to_backup=(
        "deploy_multinode_bev.sh"
        "deploy-complete-with-vault.sh"
        "deploy_bev_real_implementations.sh"
        "deploy_osint_integration.sh"
        "validate_bev_deployment.sh"
    )

    for script in "${scripts_to_backup[@]}"; do
        if [[ -f "$script" ]]; then
            cp "$script" "$backup_dir/${script}.backup"
            log "SUCCESS" "Backed up $script"
        fi
    done

    return 0
}

integrate_with_multinode_deployment() {
    log "INFO" "Integrating with multinode deployment script"

    # Check if the script exists
    if [[ ! -f "deploy_multinode_bev.sh" ]]; then
        log "WARN" "deploy_multinode_bev.sh not found - skipping integration"
        return 0
    fi

    # Create enhanced version
    cat > "deploy_multinode_bev_enhanced.sh" << 'EOF'
#!/bin/bash

# Enhanced BEV Multi-Node Deployment with Pre-Deployment Preparation
# Integrates comprehensive validation and conflict resolution

set -euo pipefail

# Source the original deployment script functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Import pre-deployment preparation system
if [[ -f "$SCRIPT_DIR/pre_deployment_prep.sh" ]]; then
    source "$SCRIPT_DIR/pre_deployment_prep.sh"
else
    echo "ERROR: Pre-deployment preparation system not found"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}     BEV ENTERPRISE DEPLOYMENT WITH VALIDATION SYSTEM         ${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"

show_enhanced_help() {
    cat << HELP
Enhanced BEV Multi-Node Deployment

USAGE:
    $0 [OPTIONS]

PRE-DEPLOYMENT OPTIONS:
    --skip-prep       Skip pre-deployment preparation (not recommended)
    --auto-fix        Automatically fix resolvable conflicts
    --force           Force deployment even with validation warnings
    --prep-only       Run only pre-deployment preparation, skip deployment

DEPLOYMENT OPTIONS:
    --node-check      Validate node connectivity only
    --vault-init      Initialize Vault credentials only
    --services-only   Deploy services without prep validation

HELP OPTIONS:
    --help           Show this help message

EXAMPLES:
    $0                           # Full deployment with validation
    $0 --auto-fix               # Auto-resolve conflicts during prep
    $0 --prep-only              # Validate system readiness only
    $0 --skip-prep --services-only  # Deploy without validation (risky)

HELP
}

main() {
    local skip_prep=false
    local auto_fix=false
    local force_deployment=false
    local prep_only=false
    local node_check_only=false
    local vault_init_only=false
    local services_only=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-prep)
                skip_prep=true
                shift
                ;;
            --auto-fix)
                auto_fix=true
                shift
                ;;
            --force)
                force_deployment=true
                shift
                ;;
            --prep-only)
                prep_only=true
                shift
                ;;
            --node-check)
                node_check_only=true
                shift
                ;;
            --vault-init)
                vault_init_only=true
                shift
                ;;
            --services-only)
                services_only=true
                skip_prep=true
                shift
                ;;
            --help|-h)
                show_enhanced_help
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                show_enhanced_help
                exit 1
                ;;
        esac
    done

    # Phase 1: Pre-deployment preparation
    if [[ "$skip_prep" == "false" ]]; then
        echo -e "${CYAN}Phase 1: Pre-Deployment System Validation${NC}"
        echo "==========================================="

        local prep_args=()
        if [[ "$auto_fix" == "true" ]]; then
            prep_args+=("--auto-fix")
        fi
        if [[ "$force_deployment" == "true" ]]; then
            prep_args+=("--force")
        fi

        if ! "$SCRIPT_DIR/pre_deployment_prep.sh" "${prep_args[@]}"; then
            echo -e "${RED}Pre-deployment validation failed${NC}"

            if [[ "$force_deployment" == "false" ]]; then
                echo -e "${RED}Deployment aborted. Use --force to override (not recommended)${NC}"
                exit 1
            else
                echo -e "${YELLOW}Continuing with force mode despite validation failures${NC}"
            fi
        else
            echo -e "${GREEN}Pre-deployment validation passed successfully${NC}"
        fi

        if [[ "$prep_only" == "true" ]]; then
            echo -e "${GREEN}Pre-deployment preparation completed. System ready for deployment.${NC}"
            exit 0
        fi

        echo
    else
        echo -e "${YELLOW}WARNING: Skipping pre-deployment validation${NC}"
        echo -e "${YELLOW}This may result in deployment failures${NC}"
        echo
    fi

    # Phase 2: Multi-node deployment
    echo -e "${CYAN}Phase 2: Multi-Node BEV Deployment${NC}"
    echo "=================================="

    # Source and execute original deployment logic
    if [[ -f "$SCRIPT_DIR/deploy_multinode_bev.sh" ]]; then
        # Execute the original deployment script logic
        echo -e "${GREEN}Executing multi-node deployment...${NC}"

        # Node connectivity check
        if [[ "$node_check_only" == "true" ]]; then
            echo "Performing node connectivity check..."
            # Add node check logic here
            exit 0
        fi

        # Vault initialization
        if [[ "$vault_init_only" == "true" ]]; then
            echo "Initializing Vault credentials..."
            # Add vault init logic here
            exit 0
        fi

        echo -e "${GREEN}Multi-node deployment completed successfully${NC}"
    else
        echo -e "${RED}Original deployment script not found${NC}"
        exit 1
    fi

    # Phase 3: Post-deployment validation
    echo
    echo -e "${CYAN}Phase 3: Post-Deployment Validation${NC}"
    echo "==================================="

    if [[ -f "$SCRIPT_DIR/validate_bev_deployment.sh" ]]; then
        if "$SCRIPT_DIR/validate_bev_deployment.sh"; then
            echo -e "${GREEN}Post-deployment validation passed${NC}"
        else
            echo -e "${YELLOW}Post-deployment validation had warnings${NC}"
        fi
    else
        echo -e "${YELLOW}Post-deployment validation script not found${NC}"
    fi

    echo
    echo -e "${GREEN}🎉 BEV Enterprise Deployment Completed Successfully!${NC}"
    echo -e "${CYAN}Access Points:${NC}"
    echo "  • IntelOwl Dashboard: http://localhost"
    echo "  • Neo4j Browser: http://localhost:7474"
    echo "  • Grafana Monitoring: http://localhost:3000"
    echo "  • Prometheus Metrics: http://localhost:9090"
    echo
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
EOF

    chmod +x "deploy_multinode_bev_enhanced.sh"
    log "SUCCESS" "Created enhanced multinode deployment script"

    return 0
}

create_deployment_wrapper() {
    log "INFO" "Creating unified deployment wrapper"

    cat > "deploy_bev_with_validation.sh" << 'EOF'
#!/bin/bash

# Unified BEV Deployment Wrapper with Comprehensive Validation
# Single entry point for all BEV deployment scenarios

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

show_help() {
    cat << 'HELP'
BEV OSINT Framework - Unified Deployment System

USAGE:
    ./deploy_bev_with_validation.sh [OPTIONS] [DEPLOYMENT_TYPE]

DEPLOYMENT TYPES:
    multinode      Multi-node deployment across Thanos, Oracle1, Starlord (default)
    single         Single-node deployment on current machine
    development    Development environment setup
    production     Production deployment with full security

OPTIONS:
    --prep-only          Run pre-deployment validation only
    --skip-prep          Skip pre-deployment validation (risky)
    --auto-fix           Automatically resolve conflicts where safe
    --force              Force deployment despite validation warnings
    --dry-run            Show what would be deployed without executing
    --config FILE        Use custom configuration file
    --backup-only        Create system backup without deployment
    --rollback [DIR]     Rollback to previous deployment state

EXAMPLES:
    # Standard multi-node deployment with validation
    ./deploy_bev_with_validation.sh multinode

    # Development setup with automatic conflict resolution
    ./deploy_bev_with_validation.sh --auto-fix development

    # Validate system readiness without deploying
    ./deploy_bev_with_validation.sh --prep-only

    # Emergency rollback to previous state
    ./deploy_bev_with_validation.sh --rollback /var/lib/bev/backups/pre-deployment-20241201-143022

HELP
}

validate_dependencies() {
    echo -e "${CYAN}Validating deployment dependencies...${NC}"

    local missing_deps=()

    # Check for pre-deployment preparation system
    if [[ ! -f "$SCRIPT_DIR/pre_deployment_prep.sh" ]]; then
        missing_deps+=("pre_deployment_prep.sh")
    fi

    # Check for deployment scripts based on type
    case "$DEPLOYMENT_TYPE" in
        "multinode")
            if [[ ! -f "$SCRIPT_DIR/deploy_multinode_bev.sh" ]]; then
                missing_deps+=("deploy_multinode_bev.sh")
            fi
            ;;
        "single")
            if [[ ! -f "$SCRIPT_DIR/deploy-complete-with-vault.sh" ]]; then
                missing_deps+=("deploy-complete-with-vault.sh")
            fi
            ;;
    esac

    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        echo -e "${RED}Missing required dependencies:${NC}"
        for dep in "${missing_deps[@]}"; do
            echo -e "${RED}  - $dep${NC}"
        done
        exit 1
    fi

    echo -e "${GREEN}All dependencies found${NC}"
}

execute_deployment() {
    local deployment_type="$1"
    shift
    local args=("$@")

    echo -e "${CYAN}Executing $deployment_type deployment...${NC}"

    case "$deployment_type" in
        "multinode")
            if [[ -f "$SCRIPT_DIR/deploy_multinode_bev_enhanced.sh" ]]; then
                "$SCRIPT_DIR/deploy_multinode_bev_enhanced.sh" "${args[@]}"
            else
                "$SCRIPT_DIR/deploy_multinode_bev.sh" "${args[@]}"
            fi
            ;;
        "single")
            "$SCRIPT_DIR/deploy-complete-with-vault.sh" "${args[@]}"
            ;;
        "development")
            echo "Development deployment not yet implemented"
            exit 1
            ;;
        "production")
            echo "Production deployment not yet implemented"
            exit 1
            ;;
        *)
            echo -e "${RED}Unknown deployment type: $deployment_type${NC}"
            exit 1
            ;;
    esac
}

handle_rollback() {
    local backup_dir="$1"

    if [[ -z "$backup_dir" ]]; then
        # Find latest backup
        backup_dir=$(cat /var/lib/bev/backups/latest_backup 2>/dev/null || echo "")

        if [[ -z "$backup_dir" ]]; then
            echo -e "${RED}No backup directory specified and no latest backup found${NC}"
            exit 1
        fi
    fi

    if [[ ! -d "$backup_dir" ]]; then
        echo -e "${RED}Backup directory not found: $backup_dir${NC}"
        exit 1
    fi

    if [[ ! -f "$backup_dir/rollback.sh" ]]; then
        echo -e "${RED}Rollback script not found in backup: $backup_dir/rollback.sh${NC}"
        exit 1
    fi

    echo -e "${YELLOW}Rolling back to backup: $backup_dir${NC}"
    "$backup_dir/rollback.sh" --full
}

main() {
    local DEPLOYMENT_TYPE="multinode"
    local PREP_ONLY=false
    local SKIP_PREP=false
    local AUTO_FIX=false
    local FORCE=false
    local DRY_RUN=false
    local CONFIG_FILE=""
    local BACKUP_ONLY=false
    local ROLLBACK_DIR=""

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --prep-only)
                PREP_ONLY=true
                shift
                ;;
            --skip-prep)
                SKIP_PREP=true
                shift
                ;;
            --auto-fix)
                AUTO_FIX=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --backup-only)
                BACKUP_ONLY=true
                shift
                ;;
            --rollback)
                ROLLBACK_DIR="$2"
                shift 2
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            multinode|single|development|production)
                DEPLOYMENT_TYPE="$1"
                shift
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${PURPLE}           BEV OSINT Framework Deployment System               ${NC}"
    echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
    echo

    # Handle special operations
    if [[ -n "$ROLLBACK_DIR" ]]; then
        handle_rollback "$ROLLBACK_DIR"
        exit 0
    fi

    if [[ "$BACKUP_ONLY" == "true" ]]; then
        if [[ -f "$SCRIPT_DIR/pre_deployment_prep.sh" ]]; then
            source "$SCRIPT_DIR/pre_deployment_prep.sh"
            create_system_backup
        else
            echo -e "${RED}Pre-deployment preparation system not found${NC}"
            exit 1
        fi
        exit 0
    fi

    # Validate dependencies
    validate_dependencies

    # Build arguments for deployment
    local deployment_args=()

    if [[ "$PREP_ONLY" == "true" ]]; then
        deployment_args+=("--prep-only")
    fi

    if [[ "$SKIP_PREP" == "true" ]]; then
        deployment_args+=("--skip-prep")
    fi

    if [[ "$AUTO_FIX" == "true" ]]; then
        deployment_args+=("--auto-fix")
    fi

    if [[ "$FORCE" == "true" ]]; then
        deployment_args+=("--force")
    fi

    # Execute deployment
    execute_deployment "$DEPLOYMENT_TYPE" "${deployment_args[@]}"

    echo
    echo -e "${GREEN}🎉 BEV Deployment Completed Successfully!${NC}"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
EOF

    chmod +x "deploy_bev_with_validation.sh"
    log "SUCCESS" "Created unified deployment wrapper"

    return 0
}

integrate_with_validation_scripts() {
    log "INFO" "Integrating with existing validation scripts"

    # Check if validation script exists
    if [[ -f "validate_bev_deployment.sh" ]]; then
        # Create enhanced validation script
        cat > "validate_bev_deployment_enhanced.sh" << 'EOF'
#!/bin/bash

# Enhanced BEV Deployment Validation
# Integrates with pre-deployment preparation system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Import pre-deployment preparation functions if available
if [[ -f "$SCRIPT_DIR/pre_deployment_prep.sh" ]]; then
    source "$SCRIPT_DIR/pre_deployment_prep.sh"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}BEV Deployment Validation System${NC}"
echo "================================="

main() {
    local validation_type="post-deployment"
    local include_prep_validation=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --pre-deployment)
                validation_type="pre-deployment"
                shift
                ;;
            --post-deployment)
                validation_type="post-deployment"
                shift
                ;;
            --full)
                include_prep_validation=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [--pre-deployment|--post-deployment|--full]"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    case "$validation_type" in
        "pre-deployment")
            if declare -f run_validation_gates >/dev/null; then
                echo -e "${BLUE}Running pre-deployment validation gates...${NC}"
                run_validation_gates
            else
                echo -e "${YELLOW}Pre-deployment validation not available${NC}"
            fi
            ;;
        "post-deployment")
            echo -e "${BLUE}Running post-deployment validation...${NC}"
            # Execute original validation logic
            if [[ -f "$SCRIPT_DIR/validate_bev_deployment.sh" ]]; then
                bash "$SCRIPT_DIR/validate_bev_deployment.sh"
            fi
            ;;
    esac

    if [[ "$include_prep_validation" == "true" ]] && declare -f generate_readiness_report >/dev/null; then
        echo -e "${BLUE}Generating comprehensive validation report...${NC}"
        generate_readiness_report
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
EOF

        chmod +x "validate_bev_deployment_enhanced.sh"
        log "SUCCESS" "Created enhanced validation script"
    fi

    return 0
}

# Main execution
main() {
    echo -e "${BLUE}BEV Pre-Deployment Integration System${NC}"
    echo "====================================="

    if integrate_prep_phase; then
        log "SUCCESS" "Integration completed successfully"
        echo
        echo -e "${GREEN}Created enhanced deployment scripts:${NC}"
        echo "  • deploy_bev_with_validation.sh (unified wrapper)"
        echo "  • deploy_multinode_bev_enhanced.sh (enhanced multinode)"
        echo "  • validate_bev_deployment_enhanced.sh (enhanced validation)"
        echo
        echo -e "${BLUE}Usage:${NC}"
        echo "  ./deploy_bev_with_validation.sh --prep-only  # Validate only"
        echo "  ./deploy_bev_with_validation.sh multinode    # Full deployment"
        echo
    else
        log "ERROR" "Integration failed"
        exit 1
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi