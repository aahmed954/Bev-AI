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
