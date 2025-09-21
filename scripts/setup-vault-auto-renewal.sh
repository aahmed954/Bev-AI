#!/bin/bash

# BEV OSINT Framework - Automated Vault Credential Renewal System
# Replaces manual 24-hour token rotation with automated renewal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔐 Setting up Vault Automated Credential Renewal..."

# 1. Create Vault renewal service script
cat > scripts/vault-renewal-service.sh << 'EOF'
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
EOF

chmod +x scripts/vault-renewal-service.sh

# 2. Create systemd service for automatic startup
cat > config/vault-renewal.service << EOF
[Unit]
Description=BEV Vault Token Renewal Service
After=network.target
Requires=network.target

[Service]
Type=simple
User=root
Group=root
ExecStart=$PROJECT_ROOT/scripts/vault-renewal-service.sh
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=vault-renewal

# Environment variables
Environment=VAULT_ADDR=http://localhost:8200
Environment=VAULT_TOKEN_FILE=/opt/vault/token
Environment=LOG_FILE=/var/log/vault-renewal.log
Environment=RENEWAL_INTERVAL=21600

# Security settings
NoNewPrivileges=yes
PrivateTmp=yes
ProtectHome=yes
ProtectSystem=strict
ReadWritePaths=/var/log /opt/vault

[Install]
WantedBy=multi-user.target
EOF

# 3. Create Vault AppRole setup script
cat > scripts/setup-vault-approle.sh << 'EOF'
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
EOF

chmod +x scripts/setup-vault-approle.sh

# 4. Create credential rotation monitoring script
cat > scripts/monitor-vault-credentials.sh << 'EOF'
#!/bin/bash

# Monitor Vault credential status and rotation

set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-http://localhost:8200}"
VAULT_TOKEN_FILE="${VAULT_TOKEN_FILE:-/opt/vault/token}"

echo "🔐 Vault Credential Status Monitor"
echo "=================================="

# Check Vault status
if curl -s "$VAULT_ADDR/v1/sys/health" > /dev/null; then
    echo "✅ Vault Status: Accessible"
else
    echo "❌ Vault Status: Not accessible"
    exit 1
fi

# Check token file
if [[ -f "$VAULT_TOKEN_FILE" ]]; then
    echo "✅ Token File: Found"

    # Get token info
    local token
    token=$(cat "$VAULT_TOKEN_FILE")

    if vault auth -method=token "$token" > /dev/null 2>&1; then
        echo "✅ Token Status: Valid"

        # Get token details
        local token_info
        token_info=$(vault token lookup -format=json 2>/dev/null)

        local ttl
        ttl=$(echo "$token_info" | jq -r '.data.ttl // 0')

        local creation_time
        creation_time=$(echo "$token_info" | jq -r '.data.creation_time // 0')

        echo "📊 Token TTL: $ttl seconds ($(($ttl / 3600)) hours)"
        echo "📅 Created: $(date -d "@$creation_time" 2>/dev/null || echo "Unknown")"

        if [[ "$ttl" -lt 3600 ]]; then
            echo "⚠️  WARNING: Token expires in less than 1 hour!"
        elif [[ "$ttl" -lt 7200 ]]; then
            echo "🔶 CAUTION: Token expires in less than 2 hours"
        else
            echo "✅ Token TTL: Healthy"
        fi
    else
        echo "❌ Token Status: Invalid or expired"
    fi
else
    echo "❌ Token File: Not found"
fi

# Check renewal service status
if systemctl is-active --quiet vault-renewal; then
    echo "✅ Renewal Service: Active"
    echo "📊 Service Status:"
    systemctl status vault-renewal --no-pager -l | tail -5
else
    echo "❌ Renewal Service: Inactive"
fi

# Check AppRole credentials
echo ""
echo "🔑 AppRole Credentials:"
if [[ -f "/opt/vault/role-id" ]]; then
    echo "✅ Role ID: Available"
else
    echo "❌ Role ID: Missing"
fi

if [[ -f "/opt/vault/secret-id" ]]; then
    echo "✅ Secret ID: Available"
else
    echo "❌ Secret ID: Missing"
fi

echo ""
echo "📋 Recent renewal logs:"
if [[ -f "/var/log/vault-renewal.log" ]]; then
    tail -5 /var/log/vault-renewal.log
else
    echo "No renewal logs found"
fi
EOF

chmod +x scripts/monitor-vault-credentials.sh

# 5. Create installation script
cat > scripts/install-vault-renewal.sh << 'EOF'
#!/bin/bash

# Install Vault renewal service

set -euo pipefail

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INSTALL] $*"
}

# Create directories
sudo mkdir -p /opt/vault /var/log
sudo chmod 755 /opt/vault

# Install systemd service
sudo cp config/vault-renewal.service /etc/systemd/system/
sudo systemctl daemon-reload

log "INFO: Vault renewal service installed"
log "INFO: Use 'sudo systemctl enable vault-renewal' to enable auto-start"
log "INFO: Use 'sudo systemctl start vault-renewal' to start service"
EOF

chmod +x scripts/install-vault-renewal.sh

echo "✅ Vault automated renewal system created!"
echo ""
echo "📋 Setup Steps:"
echo "1. Initialize Vault and create initial token"
echo "2. Run: ./scripts/setup-vault-approle.sh"
echo "3. Run: ./scripts/install-vault-renewal.sh"
echo "4. Enable service: sudo systemctl enable vault-renewal"
echo "5. Start service: sudo systemctl start vault-renewal"
echo "6. Monitor: ./scripts/monitor-vault-credentials.sh"

echo ""
echo "🔍 Service will automatically:"
echo "- Renew tokens every 6 hours (for 24h TTL tokens)"
echo "- Generate new tokens via AppRole if renewal fails"
echo "- Log all activities to /var/log/vault-renewal.log"
echo "- Restart automatically if service fails"