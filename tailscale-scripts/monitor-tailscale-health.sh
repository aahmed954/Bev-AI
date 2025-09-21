#!/bin/bash

# Tailscale Network Health Monitoring Script
# Monitors device connectivity, key expiration, route health, and service availability

set -euo pipefail

# Configuration
API_KEY="tskey-api-khe6MHjBEK11CNTRL-a1qHLAujhmMYiAvfwNqKnMPrmgiyGLH7"
BASE_URL="https://api.tailscale.com/api/v2"
WARNING_DAYS=30
CRITICAL_DAYS=7
LOG_FILE="$HOME/tailscale-health.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] $1"
    echo -e "${GREEN}${message}${NC}"
    echo "$message" >> "$LOG_FILE"
}

warn() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1"
    echo -e "${YELLOW}${message}${NC}"
    echo "$message" >> "$LOG_FILE"
}

error() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1"
    echo -e "${RED}${message}${NC}"
    echo "$message" >> "$LOG_FILE"
}

info() {
    local message="[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1"
    echo -e "${BLUE}${message}${NC}"
    echo "$message" >> "$LOG_FILE"
}

# Function to make API calls
api_call() {
    local endpoint="$1"
    curl -s -u "${API_KEY}:" \
        -H "Accept: application/json" \
        "${BASE_URL}${endpoint}" 2>/dev/null
}

# Function to calculate days between dates
days_until() {
    local target_date="$1"
    local target_epoch=$(date -d "$target_date" +%s 2>/dev/null || echo "0")
    local now_epoch=$(date +%s)
    local days=$(( (target_epoch - now_epoch) / 86400 ))
    echo "$days"
}

# Function to check device connectivity
check_device_connectivity() {
    log "Checking device connectivity..."

    local devices_json=$(api_call "/tailnet/-/devices")
    if [ -z "$devices_json" ]; then
        error "Failed to fetch device information"
        return 1
    fi

    local offline_devices=0
    local total_devices=0
    local outdated_devices=0

    echo "$devices_json" | jq -r '.devices[] | @base64' | while read -r device_data; do
        local device=$(echo "$device_data" | base64 -d)
        local name=$(echo "$device" | jq -r '.name')
        local hostname=$(echo "$device" | jq -r '.hostname')
        local last_seen=$(echo "$device" | jq -r '.lastSeen')
        local update_available=$(echo "$device" | jq -r '.updateAvailable')
        local client_version=$(echo "$device" | jq -r '.clientVersion')
        local os=$(echo "$device" | jq -r '.os')

        total_devices=$((total_devices + 1))

        # Check if device is online (last seen within 10 minutes)
        local last_seen_epoch=$(date -d "$last_seen" +%s 2>/dev/null || echo "0")
        local now_epoch=$(date +%s)
        local minutes_offline=$(( (now_epoch - last_seen_epoch) / 60 ))

        if [ "$minutes_offline" -gt 10 ]; then
            warn "Device $hostname ($name) offline for $minutes_offline minutes"
            offline_devices=$((offline_devices + 1))
        else
            info "Device $hostname ($name) online - $os $client_version"
        fi

        # Check for available updates
        if [ "$update_available" = "true" ]; then
            warn "Update available for $hostname ($name) - $os $client_version"
            outdated_devices=$((outdated_devices + 1))
        fi
    done

    info "Connectivity Summary: $((total_devices - offline_devices))/$total_devices devices online"
    if [ "$outdated_devices" -gt 0 ]; then
        warn "$outdated_devices devices have updates available"
    fi
}

# Function to check key expiration
check_key_expiration() {
    log "Checking key expiration status..."

    # Check API/Auth keys
    local keys_json=$(api_call "/tailnet/-/keys")
    if [ -n "$keys_json" ]; then
        echo "$keys_json" | jq -r '.keys[] | @base64' | while read -r key_data; do
            local key=$(echo "$key_data" | base64 -d)
            local key_id=$(echo "$key" | jq -r '.id')
            local key_type=$(echo "$key" | jq -r '.keyType')
            local expires=$(echo "$key" | jq -r '.expires')
            local description=$(echo "$key" | jq -r '.description // "No description"')

            if [ "$expires" != "null" ]; then
                local days_left=$(days_until "$expires")

                if [ "$days_left" -le "$CRITICAL_DAYS" ]; then
                    error "CRITICAL: $key_type key '$description' ($key_id) expires in $days_left days"
                elif [ "$days_left" -le "$WARNING_DAYS" ]; then
                    warn "$key_type key '$description' ($key_id) expires in $days_left days"
                else
                    info "$key_type key '$description' ($key_id) expires in $days_left days"
                fi
            fi
        done
    fi

    # Check device key expiration
    local devices_json=$(api_call "/tailnet/-/devices")
    if [ -n "$devices_json" ]; then
        echo "$devices_json" | jq -r '.devices[] | @base64' | while read -r device_data; do
            local device=$(echo "$device_data" | base64 -d)
            local name=$(echo "$device" | jq -r '.name')
            local hostname=$(echo "$device" | jq -r '.hostname')
            local expires=$(echo "$device" | jq -r '.expires')

            if [ "$expires" != "null" ]; then
                local days_left=$(days_until "$expires")

                if [ "$days_left" -le "$CRITICAL_DAYS" ]; then
                    error "CRITICAL: Device $hostname ($name) key expires in $days_left days"
                elif [ "$days_left" -le "$WARNING_DAYS" ]; then
                    warn "Device $hostname ($name) key expires in $days_left days"
                else
                    info "Device $hostname ($name) key expires in $days_left days"
                fi
            fi
        done
    fi
}

# Function to check route health
check_route_health() {
    log "Checking route health..."

    local devices_json=$(api_call "/tailnet/-/devices")
    if [ -z "$devices_json" ]; then
        error "Failed to fetch device information for route check"
        return 1
    fi

    local route_conflicts=0
    local advertised_routes=()
    local enabled_routes=()

    echo "$devices_json" | jq -r '.devices[] | @base64' | while read -r device_data; do
        local device=$(echo "$device_data" | base64 -d)
        local device_id=$(echo "$device" | jq -r '.id')
        local hostname=$(echo "$device" | jq -r '.hostname')

        local routes_json=$(api_call "/device/${device_id}/routes")
        if [ -n "$routes_json" ]; then
            local advertised=$(echo "$routes_json" | jq -r '.advertisedRoutes[]?' 2>/dev/null || echo "")
            local enabled=$(echo "$routes_json" | jq -r '.enabledRoutes[]?' 2>/dev/null || echo "")

            if [ -n "$advertised" ]; then
                info "Device $hostname advertising routes: $advertised"
            fi

            if [ -n "$enabled" ]; then
                info "Device $hostname has enabled routes: $enabled"
            fi
        fi
    done
}

# Function to test network connectivity
test_network_connectivity() {
    log "Testing network connectivity..."

    local devices_json=$(api_call "/tailnet/-/devices")
    if [ -z "$devices_json" ]; then
        error "Failed to fetch device information for connectivity test"
        return 1
    fi

    # Test ping to key devices
    local key_devices=("starlord.tailcd97c5.ts.net" "oracle1-vllm.tailcd97c5.ts.net" "thanos.tailcd97c5.ts.net")

    for device in "${key_devices[@]}"; do
        if command -v tailscale >/dev/null 2>&1; then
            if timeout 5 tailscale ping "$device" >/dev/null 2>&1; then
                info "Successfully pinged $device"
            else
                warn "Failed to ping $device"
            fi
        else
            # Fallback to regular ping if tailscale command not available
            local ip=$(echo "$devices_json" | jq -r ".devices[] | select(.name == \"$device\") | .addresses[0]" 2>/dev/null)
            if [ -n "$ip" ] && [ "$ip" != "null" ]; then
                if timeout 3 ping -c 1 "$ip" >/dev/null 2>&1; then
                    info "Successfully pinged $device ($ip)"
                else
                    warn "Failed to ping $device ($ip)"
                fi
            fi
        fi
    done
}

# Function to check ACL policy health
check_acl_health() {
    log "Checking ACL policy health..."

    local acl_json=$(api_call "/tailnet/-/acl")
    if [ -z "$acl_json" ]; then
        error "Failed to fetch ACL policy"
        return 1
    fi

    # Check for overly permissive policies
    local allow_all=$(echo "$acl_json" | jq -r '.acls[] | select(.src == ["*"] and .dst == ["*:*"]) | .action' 2>/dev/null || echo "")
    if [ "$allow_all" = "accept" ]; then
        error "SECURITY: Overly permissive ACL policy detected (allow all from * to *:*)"
    else
        info "ACL policy appears to have appropriate restrictions"
    fi

    # Count ACL rules
    local rule_count=$(echo "$acl_json" | jq '.acls | length' 2>/dev/null || echo "0")
    info "ACL policy contains $rule_count rules"
}

# Function to generate health report
generate_health_report() {
    local report_file="$HOME/tailscale-health-report-$(date +%Y%m%d_%H%M%S).txt"

    {
        echo "Tailscale Network Health Report"
        echo "Generated: $(date)"
        echo "=============================="
        echo

        echo "Recent health check results:"
        tail -100 "$LOG_FILE" | grep "$(date +%Y-%m-%d)" || echo "No recent logs found"

        echo
        echo "Device Status:"
        api_call "/tailnet/-/devices" | jq -r '.devices[] | "\(.hostname) (\(.name)) - \(.os) \(.clientVersion) - Last seen: \(.lastSeen)"'

        echo
        echo "Key Status:"
        api_call "/tailnet/-/keys" | jq -r '.keys[] | "\(.keyType) key \(.id) - Expires: \(.expires) - \(.description // "No description")"'

    } > "$report_file"

    info "Health report generated: $report_file"
}

# Main execution
main() {
    log "Starting Tailscale health monitoring..."

    check_device_connectivity
    echo
    check_key_expiration
    echo
    check_route_health
    echo
    test_network_connectivity
    echo
    check_acl_health
    echo

    generate_health_report

    log "Health monitoring completed!"

    # Show critical issues summary
    local critical_count=$(grep -c "CRITICAL\|ERROR" "$LOG_FILE" | tail -1 || echo "0")
    local warning_count=$(grep -c "WARNING" "$LOG_FILE" | tail -1 || echo "0")

    echo
    echo "=== Health Summary ==="
    if [ "$critical_count" -gt 0 ]; then
        error "Found $critical_count critical issues"
    fi
    if [ "$warning_count" -gt 0 ]; then
        warn "Found $warning_count warnings"
    fi
    if [ "$critical_count" -eq 0 ] && [ "$warning_count" -eq 0 ]; then
        log "No critical issues or warnings found"
    fi
}

# Command line argument handling
case "${1:-monitor}" in
    "monitor"|"")
        main
        ;;
    "connectivity")
        check_device_connectivity
        ;;
    "keys")
        check_key_expiration
        ;;
    "routes")
        check_route_health
        ;;
    "acl")
        check_acl_health
        ;;
    "report")
        generate_health_report
        ;;
    *)
        echo "Usage: $0 [monitor|connectivity|keys|routes|acl|report]"
        echo "  monitor      - Run all health checks (default)"
        echo "  connectivity - Check device connectivity only"
        echo "  keys         - Check key expiration only"
        echo "  routes       - Check route health only"
        echo "  acl          - Check ACL policy health only"
        echo "  report       - Generate detailed health report"
        exit 1
        ;;
esac