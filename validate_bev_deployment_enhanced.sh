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
