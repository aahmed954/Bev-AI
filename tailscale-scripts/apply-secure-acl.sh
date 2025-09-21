#!/bin/bash

# Tailscale Secure ACL Application Script
# Safely applies new ACL policy with backup and rollback capability

set -euo pipefail

# Configuration
API_KEY="tskey-api-khe6MHjBEK11CNTRL-a1qHLAujhmMYiAvfwNqKnMPrmgiyGLH7"
BASE_URL="https://api.tailscale.com/api/v2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$(dirname "$SCRIPT_DIR")/tailscale-configs"
NEW_ACL_FILE="$CONFIG_DIR/secure-acl-policy.json"
BACKUP_DIR="$HOME/tailscale-backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# Function to validate ACL JSON
validate_acl() {
    local acl_file="$1"

    if [ ! -f "$acl_file" ]; then
        error "ACL file not found: $acl_file"
    fi

    # Validate JSON syntax
    if ! jq empty "$acl_file" 2>/dev/null; then
        error "Invalid JSON syntax in ACL file: $acl_file"
    fi

    # Check for required fields
    if ! jq -e '.acls' "$acl_file" >/dev/null 2>&1; then
        error "ACL file missing required 'acls' field"
    fi

    # Validate ACL structure
    local acl_count=$(jq '.acls | length' "$acl_file")
    if [ "$acl_count" -eq 0 ]; then
        error "ACL file contains no ACL rules"
    fi

    # Check for dangerous patterns
    local allow_all=$(jq -r '.acls[] | select(.src == ["*"] and .dst == ["*:*"]) | .action' "$acl_file" 2>/dev/null || echo "")
    if [ "$allow_all" = "accept" ]; then
        warn "ACL contains allow-all rule - this may be insecure"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "ACL application cancelled by user"
        fi
    fi

    log "ACL validation passed"
}

# Function to backup current ACL
backup_current_acl() {
    mkdir -p "$BACKUP_DIR"

    local backup_file="$BACKUP_DIR/acl-backup-$(date +%Y%m%d_%H%M%S).json"

    log "Backing up current ACL policy..."

    if curl -s -u "${API_KEY}:" \
        -H "Accept: application/json" \
        "${BASE_URL}/tailnet/-/acl" > "$backup_file"; then
        log "Current ACL backed up to: $backup_file"
        echo "$backup_file"
    else
        error "Failed to backup current ACL policy"
    fi
}

# Function to apply new ACL
apply_acl() {
    local acl_file="$1"

    log "Applying new ACL policy..."

    local response=$(curl -s -w "\n%{http_code}" -u "${API_KEY}:" \
        -X POST \
        -H "Content-Type: application/json" \
        -d @"$acl_file" \
        "${BASE_URL}/tailnet/-/acl")

    local http_code=$(echo "$response" | tail -n1)
    local body=$(echo "$response" | head -n -1)

    if [ "$http_code" -eq 200 ] || [ "$http_code" -eq 201 ]; then
        log "ACL policy applied successfully!"
        return 0
    else
        error "Failed to apply ACL policy. HTTP code: $http_code\nResponse: $body"
        return 1
    fi
}

# Function to test connectivity after ACL change
test_connectivity() {
    log "Testing connectivity after ACL change..."

    local test_passed=true

    # Test API access
    if curl -s -u "${API_KEY}:" \
        -H "Accept: application/json" \
        "${BASE_URL}/tailnet/-/devices" >/dev/null; then
        info "API access test: PASSED"
    else
        warn "API access test: FAILED"
        test_passed=false
    fi

    # Test device ping if tailscale is available
    if command -v tailscale >/dev/null 2>&1; then
        local devices=("starlord.tailcd97c5.ts.net" "oracle1-vllm.tailcd97c5.ts.net")
        for device in "${devices[@]}"; do
            if timeout 5 tailscale ping "$device" >/dev/null 2>&1; then
                info "Ping test to $device: PASSED"
            else
                warn "Ping test to $device: FAILED"
                test_passed=false
            fi
        done
    else
        info "Tailscale CLI not available, skipping ping tests"
    fi

    if [ "$test_passed" = true ]; then
        log "All connectivity tests passed"
    else
        warn "Some connectivity tests failed - review ACL policy"
    fi

    return 0
}

# Function to rollback ACL
rollback_acl() {
    local backup_file="$1"

    warn "Rolling back to previous ACL policy..."

    if [ ! -f "$backup_file" ]; then
        error "Backup file not found: $backup_file"
    fi

    if apply_acl "$backup_file"; then
        log "Successfully rolled back to previous ACL policy"
    else
        error "Failed to rollback ACL policy!"
    fi
}

# Function to show ACL diff
show_acl_diff() {
    local old_acl="$1"
    local new_acl="$2"

    info "Showing differences between current and new ACL policy..."

    if command -v diff >/dev/null 2>&1; then
        echo "=== ACL Policy Differences ==="
        diff -u "$old_acl" "$new_acl" || true
        echo "=============================="
    else
        warn "diff command not available, cannot show differences"
    fi
}

# Function to display current ACL summary
show_current_acl() {
    log "Current ACL policy summary:"

    local current_acl=$(curl -s -u "${API_KEY}:" \
        -H "Accept: application/json" \
        "${BASE_URL}/tailnet/-/acl")

    if [ -n "$current_acl" ]; then
        echo "$current_acl" | jq -r '.acls[] | "Action: \(.action), Src: \(.src | join(", ")), Dst: \(.dst | join(", "))"'
    else
        warn "Could not fetch current ACL policy"
    fi
}

# Main execution
main() {
    local dry_run=false
    local force=false
    local backup_file=""

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dry-run)
                dry_run=true
                shift
                ;;
            --force)
                force=true
                shift
                ;;
            --acl-file)
                NEW_ACL_FILE="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --dry-run       Show what would be changed without applying"
                echo "  --force         Skip confirmation prompts"
                echo "  --acl-file FILE Use specific ACL file (default: $NEW_ACL_FILE)"
                echo "  --help          Show this help message"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done

    log "Starting ACL policy application..."

    # Validate new ACL file
    validate_acl "$NEW_ACL_FILE"

    # Show current ACL
    show_current_acl

    # Backup current ACL
    backup_file=$(backup_current_acl)

    # Show differences if possible
    if [ -f "$backup_file" ]; then
        show_acl_diff "$backup_file" "$NEW_ACL_FILE"
    fi

    # Dry run mode
    if [ "$dry_run" = true ]; then
        info "DRY RUN: Would apply ACL from $NEW_ACL_FILE"
        info "Backup would be saved to: $backup_file"
        log "Dry run completed - no changes made"
        exit 0
    fi

    # Confirmation prompt
    if [ "$force" = false ]; then
        echo
        warn "This will replace your current ACL policy!"
        echo "Backup saved to: $backup_file"
        read -p "Continue with ACL application? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "ACL application cancelled by user"
            exit 0
        fi
    fi

    # Apply new ACL
    if apply_acl "$NEW_ACL_FILE"; then
        log "ACL policy applied successfully"

        # Test connectivity
        test_connectivity

        # Ask if user wants to keep changes
        if [ "$force" = false ]; then
            echo
            read -p "Keep the new ACL policy? (Y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                rollback_acl "$backup_file"
            else
                log "New ACL policy is now active"
            fi
        else
            log "New ACL policy is now active (force mode)"
        fi
    else
        error "Failed to apply ACL policy"
    fi

    log "ACL application process completed"
}

# Handle script interruption
trap 'error "Script interrupted"' INT TERM

# Run main function
main "$@"