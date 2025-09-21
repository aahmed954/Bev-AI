#!/bin/bash

# ===================================================================
# BEV OSINT Framework - Master Deployment Script for Phases 7, 8, 9
# Version: 1.0.0
# Date: 2024-09-19
# Description: Comprehensive deployment orchestrator with validation
# ===================================================================

set -euo pipefail

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"

# Logging configuration
LOG_DIR="${PROJECT_ROOT}/logs/deployment"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/deploy_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
DEFAULT_PHASES="7,8,9"
DEPLOYMENT_TIMEOUT=1800  # 30 minutes
VALIDATION_TIMEOUT=600   # 10 minutes
MAX_RETRIES=3
BACKUP_RETENTION_DAYS=30

# Global variables
DEPLOY_PHASES=""
SKIP_VALIDATION=false
SKIP_BACKUP=false
DRY_RUN=false
FORCE_DEPLOY=false
PARALLEL_DEPLOY=false

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
        DEBUG) echo -e "${timestamp} ${PURPLE}[DEBUG]${NC} $message" ;;
    esac
}

show_banner() {
    echo -e "${CYAN}"
    cat << 'EOF'
██████╗ ███████╗██╗   ██╗     ██████╗ ███████╗██╗███╗   ██╗████████╗
██╔══██╗██╔════╝██║   ██║    ██╔═══██╗██╔════╝██║████╗  ██║╚══██╔══╝
██████╔╝█████╗  ██║   ██║    ██║   ██║███████╗██║██╔██╗ ██║   ██║
██╔══██╗██╔══╝  ╚██╗ ██╔╝    ██║   ██║╚════██║██║██║╚██╗██║   ██║
██████╔╝███████╗ ╚████╔╝     ╚██████╔╝███████║██║██║ ╚████║   ██║
╚═════╝ ╚══════╝  ╚═══╝       ╚═════╝ ╚══════╝╚═╝╚═╝  ╚═══╝   ╚═╝

        MASTER DEPLOYMENT ORCHESTRATOR - PHASES 7, 8, 9
                    Production-Ready Deployment System
EOF
    echo -e "${NC}"
}

cleanup_on_exit() {
    local exit_code=$?
    log INFO "Deployment script exiting with code: $exit_code"

    if [[ $exit_code -ne 0 ]]; then
        log ERROR "Deployment failed. Check logs for details: $LOG_FILE"

        # Optional: Send notification about failure
        if command -v notify-send >/dev/null 2>&1; then
            notify-send "BEV OSINT Deployment" "Deployment failed. Check logs."
        fi
    fi

    exit $exit_code
}

check_prerequisites() {
    log INFO "Checking deployment prerequisites..."

    local failed_checks=0

    # Check Docker and Docker Compose
    if ! command -v docker >/dev/null 2>&1; then
        log ERROR "Docker is not installed or not in PATH"
        ((failed_checks++))
    else
        local docker_version=$(docker --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        log INFO "Docker version: $docker_version"
    fi

    if ! command -v docker-compose >/dev/null 2>&1; then
        log ERROR "Docker Compose is not installed or not in PATH"
        ((failed_checks++))
    else
        local compose_version=$(docker-compose --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        log INFO "Docker Compose version: $compose_version"
    fi

    # Check NVIDIA Docker for GPU support
    if ! docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
        log WARN "NVIDIA Docker runtime not available. GPU-dependent services may fail."
    else
        log INFO "NVIDIA Docker runtime available"
    fi

    # Check disk space (minimum 50GB free)
    local available_space=$(df "${PROJECT_ROOT}" | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ $available_space -lt 50 ]]; then
        log WARN "Less than 50GB disk space available: ${available_space}GB"
    else
        log INFO "Available disk space: ${available_space}GB"
    fi

    # Check memory (minimum 16GB)
    local total_memory=$(free -g | awk 'NR==2{print $2}')
    if [[ $total_memory -lt 16 ]]; then
        log WARN "Less than 16GB RAM available: ${total_memory}GB"
    else
        log INFO "Total memory: ${total_memory}GB"
    fi

    # Check if required networks exist
    if ! docker network ls | grep -q "bev_osint"; then
        log WARN "BEV OSINT network not found. Will create during deployment."
    else
        log INFO "BEV OSINT network exists"
    fi

    # Check environment files
    if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
        log ERROR "Environment file not found: ${PROJECT_ROOT}/.env"
        ((failed_checks++))
    else
        log INFO "Environment file found"
    fi

    # Run pre-deployment validation script
    if [[ -f "${DEPLOYMENT_DIR}/validation/pre_deployment_check.py" ]]; then
        log INFO "Running pre-deployment validation..."
        if python3 "${DEPLOYMENT_DIR}/validation/pre_deployment_check.py"; then
            log SUCCESS "Pre-deployment validation passed"
        else
            log ERROR "Pre-deployment validation failed"
            ((failed_checks++))
        fi
    fi

    if [[ $failed_checks -gt 0 ]]; then
        log ERROR "Failed $failed_checks prerequisite checks"
        return 1
    fi

    log SUCCESS "All prerequisite checks passed"
    return 0
}

create_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log INFO "Skipping backup creation (--skip-backup flag)"
        return 0
    fi

    log INFO "Creating system backup before deployment..."

    local backup_dir="${PROJECT_ROOT}/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"

    # Backup configurations
    log INFO "Backing up configurations..."
    cp -r "${PROJECT_ROOT}/.env" "$backup_dir/" 2>/dev/null || true
    cp -r "${PROJECT_ROOT}/docker-compose"*.yml "$backup_dir/" 2>/dev/null || true

    # Backup database data (if containers are running)
    if docker ps --format '{{.Names}}' | grep -q postgres; then
        log INFO "Backing up PostgreSQL database..."
        docker exec postgres pg_dumpall -U bev > "$backup_dir/postgres_backup.sql" 2>/dev/null || true
    fi

    if docker ps --format '{{.Names}}' | grep -q neo4j; then
        log INFO "Backing up Neo4j database..."
        docker exec neo4j cypher-shell -u neo4j -p BevGraphMaster2024 "CALL apoc.export.cypher.all('backup.cypher', {})" 2>/dev/null || true
    fi

    # Clean old backups
    find "${PROJECT_ROOT}/backups" -type d -mtime +$BACKUP_RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true

    log SUCCESS "Backup created at: $backup_dir"
    echo "$backup_dir" > "${PROJECT_ROOT}/.last_backup"
}

deploy_phase() {
    local phase=$1
    local phase_script="${DEPLOYMENT_DIR}/scripts/deploy_phase_${phase}.sh"

    log INFO "Starting deployment of Phase $phase..."

    if [[ ! -f "$phase_script" ]]; then
        log ERROR "Phase deployment script not found: $phase_script"
        return 1
    fi

    # Make script executable
    chmod +x "$phase_script"

    # Execute phase deployment
    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "DRY RUN: Would execute $phase_script"
        return 0
    fi

    local start_time=$(date +%s)
    timeout $DEPLOYMENT_TIMEOUT "$phase_script" || {
        local exit_code=$?
        log ERROR "Phase $phase deployment failed with exit code: $exit_code"
        return $exit_code
    }
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log SUCCESS "Phase $phase deployed successfully in ${duration}s"
    return 0
}

validate_deployment() {
    if [[ "$SKIP_VALIDATION" == "true" ]]; then
        log INFO "Skipping deployment validation (--skip-validation flag)"
        return 0
    fi

    log INFO "Starting post-deployment validation..."

    local validation_script="${DEPLOYMENT_DIR}/validation/post_deployment_validation.py"
    if [[ ! -f "$validation_script" ]]; then
        log WARN "Post-deployment validation script not found: $validation_script"
        return 0
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "DRY RUN: Would execute validation script"
        return 0
    fi

    local start_time=$(date +%s)
    timeout $VALIDATION_TIMEOUT python3 "$validation_script" --phases="$DEPLOY_PHASES" || {
        local exit_code=$?
        log ERROR "Post-deployment validation failed with exit code: $exit_code"
        return $exit_code
    }
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log SUCCESS "Post-deployment validation completed in ${duration}s"
    return 0
}

rollback_deployment() {
    log WARN "Initiating deployment rollback..."

    local rollback_script="${DEPLOYMENT_DIR}/rollback/rollback_phases.sh"
    if [[ ! -f "$rollback_script" ]]; then
        log ERROR "Rollback script not found: $rollback_script"
        return 1
    fi

    chmod +x "$rollback_script"
    "$rollback_script" --phases="$DEPLOY_PHASES" || {
        log ERROR "Rollback failed"
        return 1
    }

    log SUCCESS "Rollback completed"
    return 0
}

deploy_phases() {
    local phases_array=(${DEPLOY_PHASES//,/ })
    local failed_phases=()

    if [[ "$PARALLEL_DEPLOY" == "true" ]]; then
        log INFO "Starting parallel deployment of phases: ${DEPLOY_PHASES}"

        local pids=()
        for phase in "${phases_array[@]}"; do
            deploy_phase "$phase" &
            pids+=($!)
        done

        # Wait for all parallel deployments
        local failed=false
        for i in "${!pids[@]}"; do
            if ! wait "${pids[$i]}"; then
                failed_phases+=("${phases_array[$i]}")
                failed=true
            fi
        done

        if [[ "$failed" == "true" ]]; then
            log ERROR "Failed phases: ${failed_phases[*]}"
            return 1
        fi
    else
        log INFO "Starting sequential deployment of phases: ${DEPLOY_PHASES}"

        for phase in "${phases_array[@]}"; do
            if ! deploy_phase "$phase"; then
                failed_phases+=("$phase")
                log ERROR "Phase $phase deployment failed"

                if [[ "$FORCE_DEPLOY" != "true" ]]; then
                    log ERROR "Stopping deployment due to failed phase"
                    return 1
                fi
            fi
        done
    fi

    if [[ ${#failed_phases[@]} -gt 0 ]]; then
        log ERROR "Some phases failed: ${failed_phases[*]}"
        return 1
    fi

    log SUCCESS "All phases deployed successfully"
    return 0
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

BEV OSINT Framework - Master Deployment Script for Phases 7, 8, 9

OPTIONS:
    -p, --phases PHASES     Comma-separated list of phases to deploy (default: 7,8,9)
    -s, --skip-validation   Skip post-deployment validation
    -b, --skip-backup      Skip pre-deployment backup
    -d, --dry-run          Show what would be done without executing
    -f, --force            Continue deployment even if phases fail
    -P, --parallel         Deploy phases in parallel (faster but riskier)
    -t, --timeout SECONDS  Deployment timeout per phase (default: 1800)
    -v, --verbose          Enable verbose logging
    -h, --help             Show this help message

EXAMPLES:
    $0                                    # Deploy all phases (7,8,9)
    $0 --phases 7,8                      # Deploy only phases 7 and 8
    $0 --dry-run                         # Show what would be deployed
    $0 --parallel --skip-validation      # Fast deployment without validation
    $0 --force --phases 9                # Force deploy phase 9 even if it fails

ENVIRONMENT VARIABLES:
    BEV_DEPLOYMENT_MODE     Deployment mode (production, staging, development)
    BEV_LOG_LEVEL           Log level (DEBUG, INFO, WARN, ERROR)
    BEV_BACKUP_RETENTION    Backup retention in days (default: 30)

FILES:
    ${PROJECT_ROOT}/.env                     Main environment configuration
    ${LOG_FILE}                              Deployment log file
    ${PROJECT_ROOT}/.last_backup             Last backup location

For more information, see: ${DEPLOYMENT_DIR}/docs/deployment_guide.md
EOF
}

# ===================================================================
# Main Execution
# ===================================================================

main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -p|--phases)
                DEPLOY_PHASES="$2"
                shift 2
                ;;
            -s|--skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            -b|--skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -f|--force)
                FORCE_DEPLOY=true
                shift
                ;;
            -P|--parallel)
                PARALLEL_DEPLOY=true
                shift
                ;;
            -t|--timeout)
                DEPLOYMENT_TIMEOUT="$2"
                shift 2
                ;;
            -v|--verbose)
                set -x
                shift
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

    # Set default phases if not specified
    if [[ -z "$DEPLOY_PHASES" ]]; then
        DEPLOY_PHASES="$DEFAULT_PHASES"
    fi

    # Set up signal handlers
    trap cleanup_on_exit EXIT
    trap 'log WARN "Deployment interrupted by user"; exit 130' INT TERM

    # Start deployment
    show_banner
    log INFO "Starting BEV OSINT Framework deployment"
    log INFO "Phases to deploy: $DEPLOY_PHASES"
    log INFO "Log file: $LOG_FILE"

    if [[ "$DRY_RUN" == "true" ]]; then
        log INFO "DRY RUN MODE - No actual changes will be made"
    fi

    # Execute deployment workflow
    local overall_start_time=$(date +%s)

    if ! check_prerequisites; then
        log ERROR "Prerequisites check failed"
        exit 1
    fi

    create_backup || {
        log ERROR "Backup creation failed"
        exit 1
    }

    if ! deploy_phases; then
        log ERROR "Phase deployment failed"

        if [[ "$FORCE_DEPLOY" != "true" ]]; then
            read -p "Deployment failed. Attempt rollback? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rollback_deployment
            fi
        fi
        exit 1
    fi

    if ! validate_deployment; then
        log ERROR "Deployment validation failed"
        exit 1
    fi

    local overall_end_time=$(date +%s)
    local total_duration=$((overall_end_time - overall_start_time))

    log SUCCESS "===================================================="
    log SUCCESS "BEV OSINT Framework deployment completed successfully!"
    log SUCCESS "Phases deployed: $DEPLOY_PHASES"
    log SUCCESS "Total deployment time: ${total_duration}s"
    log SUCCESS "Log file: $LOG_FILE"
    log SUCCESS "===================================================="

    # Optional: Send success notification
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "BEV OSINT Deployment" "Deployment completed successfully in ${total_duration}s"
    fi
}

# Execute main function with all arguments
main "$@"