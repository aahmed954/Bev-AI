#!/bin/bash

# Vault Token Renewal Service
# Automatically renews Vault tokens before expiration

set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"
VAULT_TOKEN_FILE="${VAULT_TOKEN_FILE:-/opt/vault/token}"
LOG_FILE="${LOG_FILE:-/var/log/vault-renewal.log}"
RENEWAL_INTERVAL="${RENEWAL_INTERVAL:-21600}" # 6 hours (for 24h tokens)

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [VAULT-RENEWAL] $*" | tee -a "$LOG_FILE"
}

check_vault_status() {
    if ! curl -s "$VAULT_ADDR/v1/sys/health" > /dev/null; then
        log "ERROR: Vault is not accessible at $VAULT_ADDR"
        return 1
    fi
    log "INFO: Vault is accessible"
    return 0
}

get_token_info() {
    local token="$1"
    vault auth -method=token "$token" > /dev/null 2>&1
    vault token lookup -format=json 2>/dev/null || echo "{}"
}

renew_token() {
    local token="$1"

    log "INFO: Attempting to renew token..."

    # Get current token info
    local token_info
    token_info=$(get_token_info "$token")

    local ttl
    ttl=$(echo "$token_info" | jq -r '.data.ttl // 0')

    if [[ "$ttl" -lt 3600 ]]; then
        log "WARNING: Token TTL is less than 1 hour ($ttl seconds)"
    fi

    # Attempt renewal
    if vault token renew "$token" > /dev/null 2>&1; then
        log "SUCCESS: Token renewed successfully"
        return 0
    else
        log "ERROR: Failed to renew token"
        return 1
    fi
}

generate_new_approle_token() {
    local role_id="$1"
    local secret_id="$2"

    log "INFO: Generating new AppRole token..."

    local response
    response=$(vault write -format=json auth/approle/login \
        role_id="$role_id" \
        secret_id="$secret_id" 2>/dev/null || echo "{}")

    local new_token
    new_token=$(echo "$response" | jq -r '.auth.client_token // empty')

    if [[ -n "$new_token" && "$new_token" != "null" ]]; then
        echo "$new_token" > "$VAULT_TOKEN_FILE"
        log "SUCCESS: New AppRole token generated and saved"
        return 0
    else
        log "ERROR: Failed to generate new AppRole token"
        return 1
    fi
}

main() {
    log "INFO: Starting Vault renewal service"

    while true; do
        if ! check_vault_status; then
            log "WARNING: Vault not accessible, waiting..."
            sleep 60
            continue
        fi

        if [[ ! -f "$VAULT_TOKEN_FILE" ]]; then
            log "ERROR: Token file not found: $VAULT_TOKEN_FILE"
            sleep 60
            continue
        fi

        local current_token
        current_token=$(cat "$VAULT_TOKEN_FILE")

        if ! renew_token "$current_token"; then
            log "WARNING: Token renewal failed, attempting to generate new token..."

            # Try to use AppRole for new token
            local role_id="${VAULT_ROLE_ID:-}"
            local secret_id="${VAULT_SECRET_ID:-}"

            if [[ -n "$role_id" && -n "$secret_id" ]]; then
                if ! generate_new_approle_token "$role_id" "$secret_id"; then
                    log "ERROR: Failed to generate new token via AppRole"
                fi
            else
                log "ERROR: No AppRole credentials available for token regeneration"
            fi
        fi

        log "INFO: Sleeping for $RENEWAL_INTERVAL seconds..."
        sleep "$RENEWAL_INTERVAL"
    done
}

# Handle signals gracefully
trap 'log "INFO: Received shutdown signal, exiting..."; exit 0' SIGTERM SIGINT

main "$@"
