#!/bin/bash

# Setup Vault AppRole for automated authentication

set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [APPROLE-SETUP] $*"
}

setup_approle() {
    log "INFO: Setting up AppRole authentication..."

    # Enable AppRole auth method
    vault auth enable approle 2>/dev/null || log "INFO: AppRole already enabled"

    # Create policy for BEV services
    vault policy write bev-services - << EOL
# BEV Services Policy
path "secret/data/bev/*" {
  capabilities = ["read", "list"]
}

path "secret/metadata/bev/*" {
  capabilities = ["read", "list"]
}

path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

path "sys/capabilities-self" {
  capabilities = ["update"]
}
EOL

    # Create AppRole
    vault write auth/approle/role/bev-auto-renewal \
        token_policies="bev-services" \
        token_ttl=24h \
        token_max_ttl=48h \
        bind_secret_id=true \
        secret_id_ttl=0

    log "SUCCESS: AppRole 'bev-auto-renewal' created"

    # Get Role ID
    local role_id
    role_id=$(vault read -field=role_id auth/approle/role/bev-auto-renewal/role-id)
    echo "$role_id" > /opt/vault/role-id
    log "INFO: Role ID saved to /opt/vault/role-id"

    # Generate Secret ID
    local secret_response
    secret_response=$(vault write -format=json auth/approle/role/bev-auto-renewal/secret-id)
    local secret_id
    secret_id=$(echo "$secret_response" | jq -r '.data.secret_id')
    echo "$secret_id" > /opt/vault/secret-id
    log "INFO: Secret ID saved to /opt/vault/secret-id"

    # Set proper permissions
    chmod 600 /opt/vault/role-id /opt/vault/secret-id

    log "SUCCESS: AppRole setup complete"
}

# Run setup
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    setup_approle
fi
