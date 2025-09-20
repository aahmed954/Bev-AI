#!/bin/bash

#################################################################
# BEV Test Environment Setup Script
#
# Ensures all necessary directories and permissions are set up
# for the BEV testing framework
#################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Create necessary directories
log_info "Creating test framework directories..."

mkdir -p "$SCRIPT_DIR/test-reports"
mkdir -p "$SCRIPT_DIR/tests"
mkdir -p "$SCRIPT_DIR/tests/test_data"

# Set proper permissions
log_info "Setting permissions for test scripts..."

chmod +x "$SCRIPT_DIR"/*.sh 2>/dev/null || true
chmod +x "$SCRIPT_DIR/tests"/*.sh 2>/dev/null || true

# Create .gitignore for test reports if it doesn't exist
if [[ ! -f "$SCRIPT_DIR/test-reports/.gitignore" ]]; then
    cat > "$SCRIPT_DIR/test-reports/.gitignore" << 'EOF'
# Ignore all test reports except README
*
!.gitignore
!README.md
EOF
    log_info "Created .gitignore for test reports"
fi

# Create README for test-reports directory
if [[ ! -f "$SCRIPT_DIR/test-reports/README.md" ]]; then
    cat > "$SCRIPT_DIR/test-reports/README.md" << 'EOF'
# Test Reports Directory

This directory contains generated test reports from the BEV testing framework.

## Report Types

- `bev_complete_test_report_*.html` - Master test report with all results
- `validation_*.log` - System validation test logs
- `integration_test_report_*.html` - Integration test results
- `performance_report_*.html` - Performance test analysis
- `security_report_*.html` - Security assessment report
- `monitoring_report_*.html` - Monitoring system validation

## Cleanup

Reports are automatically cleaned up, keeping only the 10 most recent reports.
EOF
    log_info "Created README for test reports directory"
fi

# Verify test scripts exist
log_info "Verifying test scripts..."

required_scripts=(
    "validate_bev_deployment.sh"
    "run_all_tests.sh"
    "tests/integration_tests.sh"
    "tests/performance_tests.sh"
    "tests/security_tests.sh"
    "tests/monitoring_tests.sh"
)

missing_scripts=()

for script in "${required_scripts[@]}"; do
    if [[ -f "$SCRIPT_DIR/$script" ]]; then
        log_success "✓ $script"
    else
        missing_scripts+=("$script")
    fi
done

if [[ ${#missing_scripts[@]} -gt 0 ]]; then
    echo "Missing required scripts:"
    printf ' - %s\n' "${missing_scripts[@]}"
    exit 1
fi

# Check if Docker is available
if command -v docker &> /dev/null; then
    log_success "Docker is available"
else
    log_info "Docker not found - please install Docker before running tests"
fi

# Check if basic tools are available
tools_check=(
    "curl:HTTP client"
    "jq:JSON processor"
    "bc:Calculator for performance math"
)

missing_tools=()

for tool_info in "${tools_check[@]}"; do
    IFS=':' read -ra tool_parts <<< "$tool_info"
    tool="${tool_parts[0]}"
    description="${tool_parts[1]}"

    if command -v "$tool" &> /dev/null; then
        log_success "✓ $tool ($description)"
    else
        missing_tools+=("$tool")
    fi
done

if [[ ${#missing_tools[@]} -gt 0 ]]; then
    log_info "Some testing tools are missing but will be auto-installed if needed:"
    printf ' - %s\n' "${missing_tools[@]}"
fi

log_success "Test environment setup completed!"
log_info "You can now run tests with: ./run_all_tests.sh"