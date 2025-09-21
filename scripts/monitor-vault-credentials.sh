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
