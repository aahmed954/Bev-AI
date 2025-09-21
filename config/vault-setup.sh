#!/bin/bash
# Vault Setup and Configuration Script
# Initializes Vault with comprehensive security policies

set -euo pipefail

# Configuration
export VAULT_ADDR="https://localhost:8200"
export VAULT_SKIP_VERIFY=1  # Only for development
VAULT_DATA_DIR="/opt/vault/data"
VAULT_TLS_DIR="/opt/vault/tls"
POLICY_DIR="./vault-policies"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Vault is initialized
check_vault_status() {
    if vault status >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Generate self-signed certificates for development
generate_tls_certs() {
    log_info "Generating TLS certificates for Vault"

    mkdir -p "$VAULT_TLS_DIR"

    # Generate private key
    openssl genrsa -out "$VAULT_TLS_DIR/vault.key" 2048

    # Generate certificate
    openssl req -new -x509 -key "$VAULT_TLS_DIR/vault.key" \
        -out "$VAULT_TLS_DIR/vault.crt" -days 365 \
        -subj "/C=US/ST=CA/L=SF/O=BEV/CN=vault.local" \
        -extensions v3_req \
        -config <(cat <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C=US
ST=CA
L=SF
O=BEV
CN=vault.local

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = vault.local
DNS.2 = localhost
IP.1 = 127.0.0.1
EOF
    )

    # Set permissions
    chmod 600 "$VAULT_TLS_DIR/vault.key"
    chmod 644 "$VAULT_TLS_DIR/vault.crt"

    log_info "TLS certificates generated"
}

# Initialize Vault
initialize_vault() {
    log_info "Initializing Vault"

    # Create data directory
    mkdir -p "$VAULT_DATA_DIR"

    # Initialize with 5 key shares, threshold of 3
    vault operator init -key-shares=5 -key-threshold=3 -format=json > vault-init.json

    # Extract keys and root token
    VAULT_UNSEAL_KEY_1=$(jq -r '.unseal_keys_b64[0]' vault-init.json)
    VAULT_UNSEAL_KEY_2=$(jq -r '.unseal_keys_b64[1]' vault-init.json)
    VAULT_UNSEAL_KEY_3=$(jq -r '.unseal_keys_b64[2]' vault-init.json)
    VAULT_ROOT_TOKEN=$(jq -r '.root_token' vault-init.json)

    # Unseal Vault
    log_info "Unsealing Vault"
    vault operator unseal "$VAULT_UNSEAL_KEY_1"
    vault operator unseal "$VAULT_UNSEAL_KEY_2"
    vault operator unseal "$VAULT_UNSEAL_KEY_3"

    # Export root token for subsequent commands
    export VAULT_TOKEN="$VAULT_ROOT_TOKEN"

    log_info "Vault initialized and unsealed"
    log_warn "Root token and unseal keys saved to vault-init.json - SECURE THIS FILE!"
}

# Enable auth methods
enable_auth_methods() {
    log_info "Enabling authentication methods"

    # Enable userpass auth
    vault auth enable userpass

    # Enable AppRole auth
    vault auth enable approle

    # Enable Kubernetes auth (if in K8s environment)
    vault auth enable kubernetes || log_warn "Kubernetes auth not enabled (not in K8s environment)"

    # Enable LDAP auth
    vault auth enable ldap

    # Enable GitHub auth
    vault auth enable github

    # Enable AWS auth
    vault auth enable aws

    log_info "Authentication methods enabled"
}

# Enable secrets engines
enable_secrets_engines() {
    log_info "Enabling secrets engines"

    # KV v2 secrets engine
    vault secrets enable -path=kv2 kv-v2

    # Database secrets engine
    vault secrets enable database

    # Transit encryption engine
    vault secrets enable transit

    # PKI secrets engine
    vault secrets enable pki
    vault secrets enable -path=pki_int pki

    # SSH secrets engine
    vault secrets enable ssh

    # Cloud secrets engines
    vault secrets enable aws
    vault secrets enable gcp

    log_info "Secrets engines enabled"
}

# Configure PKI
configure_pki() {
    log_info "Configuring PKI"

    # Configure root CA
    vault secrets tune -max-lease-ttl=87600h pki

    # Generate root certificate
    vault write pki/root/generate/internal \
        common_name="BEV Root CA" \
        ttl=87600h

    # Configure intermediate CA
    vault secrets tune -max-lease-ttl=43800h pki_int

    # Generate intermediate CSR
    vault write -format=json pki_int/intermediate/generate/internal \
        common_name="BEV Intermediate Authority" \
        | jq -r '.data.csr' > pki_intermediate.csr

    # Sign intermediate certificate
    vault write -format=json pki/root/sign-intermediate \
        csr=@pki_intermediate.csr \
        format=pem_bundle ttl="43800h" \
        | jq -r '.data.certificate' > intermediate.cert.pem

    # Set signed certificate
    vault write pki_int/intermediate/set-signed \
        certificate=@intermediate.cert.pem

    # Create role for service certificates
    vault write pki_int/roles/service-certs \
        allowed_domains="bev.local,localhost" \
        allow_subdomains=true \
        max_ttl="720h"

    log_info "PKI configured"
}

# Configure transit encryption
configure_transit() {
    log_info "Configuring transit encryption"

    # Create encryption keys
    vault write -f transit/keys/app-data
    vault write -f transit/keys/research-data
    vault write -f transit/keys/security-tools
    vault write -f transit/keys/dev-data
    vault write -f transit/keys/build-artifacts

    log_info "Transit encryption configured"
}

# Apply security policies
apply_policies() {
    log_info "Applying security policies"

    if [[ ! -d "$POLICY_DIR" ]]; then
        log_error "Policy directory $POLICY_DIR not found"
        exit 1
    fi

    for policy_file in "$POLICY_DIR"/*.hcl; do
        if [[ -f "$policy_file" ]]; then
            policy_name=$(basename "$policy_file" .hcl)
            policy_name=${policy_name%-policy}

            log_info "Applying policy: $policy_name"
            vault policy write "$policy_name" "$policy_file"
        fi
    done

    log_info "Security policies applied"
}

# Create AppRole configurations
create_approles() {
    log_info "Creating AppRole configurations"

    # CI/CD AppRole
    vault write auth/approle/role/cicd \
        token_policies="cicd" \
        token_ttl=1h \
        token_max_ttl=4h \
        bind_secret_id=true

    # Application AppRole
    vault write auth/approle/role/app-services \
        token_policies="application" \
        token_ttl=30m \
        token_max_ttl=2h \
        bind_secret_id=true

    # Development AppRole
    vault write auth/approle/role/dev-services \
        token_policies="developer" \
        token_ttl=2h \
        token_max_ttl=8h \
        bind_secret_id=true

    # ORACLE Workers AppRole
    vault write auth/approle/role/oracle-workers \
        token_policies="oracle-worker" \
        token_ttl=1h \
        token_max_ttl=6h \
        bind_secret_id=true

    log_info "AppRoles created"
}

# Create user accounts
create_users() {
    log_info "Creating user accounts"

    # Security team user
    vault write auth/userpass/users/security-admin \
        password="ChangeMe123!" \
        policies="security-team"

    # Developer user
    vault write auth/userpass/users/developer \
        password="DevPass123!" \
        policies="developer"

    log_info "User accounts created"
    log_warn "Change default passwords immediately!"
}

# Configure audit logging
configure_audit() {
    log_info "Configuring audit logging"

    # Enable file audit
    vault audit enable file file_path=/opt/vault/audit.log

    # Enable syslog audit (if available)
    vault audit enable syslog tag="vault" facility="AUTH" || log_warn "Syslog audit not available"

    log_info "Audit logging configured"
}

# Create initial secrets structure
create_initial_secrets() {
    log_info "Creating initial secrets structure"

    # Create basic secret paths
    vault kv put kv2/shared/config/database url="postgres://localhost:5432/bev"
    vault kv put kv2/shared/config/redis url="redis://localhost:6379"
    vault kv put kv2/shared/config/influxdb url="http://localhost:8086"

    # Create API keys structure
    vault kv put kv2/apis/research/placeholder key="replace-with-real-key"

    # Create monitoring secrets
    vault kv put kv2/monitoring/grafana admin_password="ChangeMe123!"

    log_info "Initial secrets created"
}

# Health check
health_check() {
    log_info "Performing health check"

    if vault status; then
        log_info "Vault is healthy and operational"
        return 0
    else
        log_error "Vault health check failed"
        return 1
    fi
}

# Main setup function
main() {
    log_info "Starting Vault setup and configuration"

    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed"
        exit 1
    fi

    # Generate TLS certificates
    generate_tls_certs

    # Wait for Vault to be ready
    log_info "Waiting for Vault to be ready..."
    sleep 5

    # Check if Vault is already initialized
    if check_vault_status; then
        log_info "Vault is already initialized"

        # Check if we have the root token
        if [[ -f "vault-init.json" ]]; then
            VAULT_ROOT_TOKEN=$(jq -r '.root_token' vault-init.json)
            export VAULT_TOKEN="$VAULT_ROOT_TOKEN"
        else
            log_error "Vault is initialized but vault-init.json not found"
            log_error "Please provide root token manually"
            exit 1
        fi
    else
        # Initialize Vault
        initialize_vault
    fi

    # Enable auth methods
    enable_auth_methods

    # Enable secrets engines
    enable_secrets_engines

    # Configure PKI
    configure_pki

    # Configure transit encryption
    configure_transit

    # Apply security policies
    apply_policies

    # Create AppRole configurations
    create_approles

    # Create user accounts
    create_users

    # Configure audit logging
    configure_audit

    # Create initial secrets
    create_initial_secrets

    # Final health check
    if health_check; then
        log_info "Vault setup completed successfully!"
        log_info "Vault UI available at: $VAULT_ADDR/ui"
        log_warn "Secure the vault-init.json file containing root token and unseal keys!"
    else
        log_error "Vault setup completed with errors"
        exit 1
    fi
}

# Run main function
main "$@"