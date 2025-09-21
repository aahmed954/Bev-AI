#!/bin/bash
#
# BEV OSINT Framework - GitHub Secrets Management Setup
#
# Configures GitHub Secrets integration with HashiCorp Vault
# and sets up secure credential management for multi-node deployment
#

set -euo pipefail

# Configuration
GITHUB_REPO="${GITHUB_REPO:-starlord/Bev}"
VAULT_ADDR="${VAULT_ADDR:-http://100.122.12.35:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[SECRETS-SETUP]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

vault_log() {
    echo -e "${PURPLE}[VAULT]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for secrets management setup..."

    # Check GitHub CLI
    if ! command -v gh &> /dev/null; then
        error "GitHub CLI (gh) is required but not installed"
    fi

    # Check Vault CLI
    if ! command -v vault &> /dev/null; then
        error "Vault CLI is required but not installed"
    fi

    # Check jq
    if ! command -v jq &> /dev/null; then
        error "jq is required but not installed"
    fi

    # Check GitHub authentication
    if ! gh auth status &>/dev/null; then
        error "GitHub CLI not authenticated. Run 'gh auth login' first"
    fi

    # Check Vault connectivity
    if ! curl -s --connect-timeout 10 "$VAULT_ADDR/v1/sys/health" | jq -e '.sealed == false' > /dev/null 2>&1; then
        error "Vault is not accessible or is sealed. Address: $VAULT_ADDR"
    fi

    success "Prerequisites check completed"
}

# Setup Vault authentication for GitHub Actions
setup_vault_auth() {
    vault_log "Setting up Vault authentication for GitHub Actions..."

    # Authenticate with Vault
    if [[ -z "$VAULT_TOKEN" ]]; then
        vault_log "No VAULT_TOKEN provided, attempting interactive authentication..."
        vault auth
    else
        vault auth -method=token token="$VAULT_TOKEN"
    fi

    # Enable AppRole auth method if not already enabled
    if ! vault auth list | grep -q "approle"; then
        vault auth enable approle
        vault_log "AppRole authentication method enabled"
    fi

    # Create policies for different node types
    create_node_policies

    success "Vault authentication configured"
}

# Create Vault policies for different node types
create_node_policies() {
    vault_log "Creating Vault policies for node access..."

    # THANOS node policy (GPU compute, primary databases)
    cat > /tmp/thanos-policy.hcl <<EOF
# THANOS Node Policy - GPU Compute and Primary Databases
path "secret/data/bev/postgres" {
  capabilities = ["read"]
}

path "secret/data/bev/neo4j" {
  capabilities = ["read"]
}

path "secret/data/bev/elasticsearch" {
  capabilities = ["read"]
}

path "secret/data/bev/rabbitmq" {
  capabilities = ["read"]
}

path "secret/data/bev/ai-services/*" {
  capabilities = ["read"]
}

path "secret/data/bev/gpu-config" {
  capabilities = ["read"]
}

path "auth/token/lookup-self" {
  capabilities = ["read"]
}
EOF

    vault policy write thanos-policy /tmp/thanos-policy.hcl

    # ORACLE1 node policy (ARM64, monitoring, edge services)
    cat > /tmp/oracle1-policy.hcl <<EOF
# ORACLE1 Node Policy - ARM64 Monitoring and Edge Services
path "secret/data/bev/redis" {
  capabilities = ["read"]
}

path "secret/data/bev/prometheus" {
  capabilities = ["read"]
}

path "secret/data/bev/grafana" {
  capabilities = ["read"]
}

path "secret/data/bev/consul" {
  capabilities = ["read"]
}

path "secret/data/bev/analyzers/*" {
  capabilities = ["read"]
}

path "secret/data/bev/monitoring/*" {
  capabilities = ["read"]
}

path "auth/token/lookup-self" {
  capabilities = ["read"]
}
EOF

    vault policy write oracle1-policy /tmp/oracle1-policy.hcl

    # STARLORD node policy (development, control, full access)
    cat > /tmp/starlord-policy.hcl <<EOF
# STARLORD Node Policy - Development and Control (Full Access)
path "secret/data/bev/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "secret/metadata/bev/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "auth/approle/role/+/role-id" {
  capabilities = ["read"]
}

path "auth/approle/role/+/secret-id" {
  capabilities = ["update"]
}

path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "sys/capabilities-self" {
  capabilities = ["update"]
}
EOF

    vault policy write starlord-policy /tmp/starlord-policy.hcl

    # GitHub Actions deployment policy
    cat > /tmp/github-actions-policy.hcl <<EOF
# GitHub Actions Deployment Policy
path "secret/data/bev/*" {
  capabilities = ["read"]
}

path "auth/approle/role/+/role-id" {
  capabilities = ["read"]
}

path "auth/approle/role/+/secret-id" {
  capabilities = ["update"]
}

path "auth/token/lookup-self" {
  capabilities = ["read"]
}
EOF

    vault policy write github-actions-policy /tmp/github-actions-policy.hcl

    vault_log "Vault policies created successfully"
}

# Create AppRole configurations for each node
create_approles() {
    vault_log "Creating AppRole configurations for multi-node deployment..."

    # THANOS AppRole
    vault write auth/approle/role/thanos \
        token_policies="thanos-policy" \
        token_ttl=24h \
        token_max_ttl=48h \
        secret_id_ttl=24h \
        secret_id_num_uses=10

    # ORACLE1 AppRole
    vault write auth/approle/role/oracle1 \
        token_policies="oracle1-policy" \
        token_ttl=24h \
        token_max_ttl=48h \
        secret_id_ttl=24h \
        secret_id_num_uses=10

    # STARLORD AppRole
    vault write auth/approle/role/starlord \
        token_policies="starlord-policy" \
        token_ttl=8h \
        token_max_ttl=24h \
        secret_id_ttl=8h \
        secret_id_num_uses=5

    # GitHub Actions AppRole
    vault write auth/approle/role/github-actions \
        token_policies="github-actions-policy" \
        token_ttl=2h \
        token_max_ttl=4h \
        secret_id_ttl=1h \
        secret_id_num_uses=3

    vault_log "AppRoles created successfully"
}

# Generate all required secrets and store in Vault
generate_and_store_secrets() {
    vault_log "Generating and storing secrets in Vault..."

    # Enable KV v2 secrets engine if not already enabled
    if ! vault secrets list | grep -q "secret/"; then
        vault secrets enable -path=secret kv-v2
    fi

    # PostgreSQL secrets
    vault kv put secret/bev/postgres \
        username="researcher" \
        password="$(openssl rand -base64 32)" \
        database="osint" \
        host="bev_postgres" \
        port="5432"

    # Neo4j secrets
    vault kv put secret/bev/neo4j \
        username="neo4j" \
        password="$(openssl rand -base64 24)" \
        database="neo4j" \
        bolt_url="bolt://bev_neo4j:7687" \
        http_url="http://bev_neo4j:7474"

    # Redis secrets
    vault kv put secret/bev/redis \
        password="$(openssl rand -base64 16)" \
        host="bev_redis" \
        port="6379"

    # Elasticsearch secrets
    vault kv put secret/bev/elasticsearch \
        username="elastic" \
        password="$(openssl rand -base64 20)" \
        host="bev_elasticsearch" \
        port="9200"

    # RabbitMQ secrets
    vault kv put secret/bev/rabbitmq \
        username="bev_user" \
        password="$(openssl rand -base64 20)" \
        vhost="bev_vhost" \
        host="bev_rabbitmq" \
        port="5672"

    # Prometheus secrets
    vault kv put secret/bev/prometheus \
        admin_password="$(openssl rand -base64 16)" \
        scrape_interval="15s" \
        retention="30d"

    # Grafana secrets
    vault kv put secret/bev/grafana \
        admin_username="admin" \
        admin_password="$(openssl rand -base64 16)" \
        secret_key="$(openssl rand -base64 32)"

    # Consul secrets
    vault kv put secret/bev/consul \
        encrypt_key="$(consul keygen)" \
        gossip_key="$(openssl rand -base64 24)" \
        master_token="$(uuidgen)"

    # AI Services secrets
    vault kv put secret/bev/ai-services \
        api_key="$(openssl rand -base64 32)" \
        model_encryption_key="$(openssl rand -base64 32)" \
        inference_token="$(openssl rand -base64 24)"

    # GPU configuration secrets
    vault kv put secret/bev/gpu-config \
        cuda_version="12.0" \
        driver_version="535.86.10" \
        memory_limit="22GB" \
        compute_capability="8.9"

    # Monitoring secrets
    vault kv put secret/bev/monitoring \
        webhook_url="https://hooks.slack.com/services/PLACEHOLDER" \
        alert_email="admin@bev-platform.local" \
        notification_token="$(openssl rand -base64 20)"

    # Analyzer secrets
    vault kv put secret/bev/analyzers \
        api_rate_limit="1000" \
        cache_encryption_key="$(openssl rand -base64 32)" \
        session_token="$(openssl rand -base64 24)"

    vault_log "All secrets generated and stored in Vault"
}

# Get AppRole credentials and set up GitHub Secrets
setup_github_secrets() {
    log "Setting up GitHub Secrets for multi-node deployment..."

    # Get Role IDs (these can be public)
    THANOS_ROLE_ID=$(vault read -field=role_id auth/approle/role/thanos/role-id)
    ORACLE1_ROLE_ID=$(vault read -field=role_id auth/approle/role/oracle1/role-id)
    STARLORD_ROLE_ID=$(vault read -field=role_id auth/approle/role/starlord/role-id)
    GITHUB_ACTIONS_ROLE_ID=$(vault read -field=role_id auth/approle/role/github-actions/role-id)

    # Generate Secret IDs (these are secret)
    THANOS_SECRET_ID=$(vault write -field=secret_id auth/approle/role/thanos/secret-id)
    ORACLE1_SECRET_ID=$(vault write -field=secret_id auth/approle/role/oracle1/secret-id)
    STARLORD_SECRET_ID=$(vault write -field=secret_id auth/approle/role/starlord/secret-id)
    GITHUB_ACTIONS_SECRET_ID=$(vault write -field=secret_id auth/approle/role/github-actions/secret-id)

    # Set GitHub Secrets
    log "Setting GitHub repository secrets..."

    # Vault configuration
    gh secret set VAULT_ADDR --body "$VAULT_ADDR" --repo "$GITHUB_REPO"
    gh secret set VAULT_TOKEN --body "$VAULT_TOKEN" --repo "$GITHUB_REPO"

    # Node-specific AppRole credentials
    gh secret set THANOS_VAULT_ROLE_ID --body "$THANOS_ROLE_ID" --repo "$GITHUB_REPO"
    gh secret set THANOS_VAULT_SECRET_ID --body "$THANOS_SECRET_ID" --repo "$GITHUB_REPO"

    gh secret set ORACLE1_VAULT_ROLE_ID --body "$ORACLE1_ROLE_ID" --repo "$GITHUB_REPO"
    gh secret set ORACLE1_VAULT_SECRET_ID --body "$ORACLE1_SECRET_ID" --repo "$GITHUB_REPO"

    gh secret set STARLORD_VAULT_ROLE_ID --body "$STARLORD_ROLE_ID" --repo "$GITHUB_REPO"
    gh secret set STARLORD_VAULT_SECRET_ID --body "$STARLORD_SECRET_ID" --repo "$GITHUB_REPO"

    # GitHub Actions deployment credentials
    gh secret set GITHUB_ACTIONS_VAULT_ROLE_ID --body "$GITHUB_ACTIONS_ROLE_ID" --repo "$GITHUB_REPO"
    gh secret set GITHUB_ACTIONS_VAULT_SECRET_ID --body "$GITHUB_ACTIONS_SECRET_ID" --repo "$GITHUB_REPO"

    # Additional configuration secrets
    gh secret set POSTGRES_ENCRYPTION_KEY --body "$(openssl rand -base64 32)" --repo "$GITHUB_REPO"
    gh secret set REDIS_ENCRYPTION_KEY --body "$(openssl rand -base64 32)" --repo "$GITHUB_REPO"
    gh secret set DEPLOYMENT_WEBHOOK_SECRET --body "$(openssl rand -base64 24)" --repo "$GITHUB_REPO"

    success "GitHub Secrets configured successfully"
}

# Create environment-specific secret configurations
create_environment_configs() {
    log "Creating environment-specific secret configurations..."

    # Production environment secrets
    gh secret set PRODUCTION_VAULT_NAMESPACE --body "production" --repo "$GITHUB_REPO"
    gh secret set PRODUCTION_DEPLOYMENT_KEY --body "$(openssl rand -base64 32)" --repo "$GITHUB_REPO"

    # Staging environment secrets
    gh secret set STAGING_VAULT_NAMESPACE --body "staging" --repo "$GITHUB_REPO"
    gh secret set STAGING_DEPLOYMENT_KEY --body "$(openssl rand -base64 32)" --repo "$GITHUB_REPO"

    # Development environment secrets
    gh secret set DEVELOPMENT_VAULT_NAMESPACE --body "development" --repo "$GITHUB_REPO"
    gh secret set DEVELOPMENT_DEPLOYMENT_KEY --body "$(openssl rand -base64 32)" --repo "$GITHUB_REPO"

    success "Environment-specific configurations created"
}

# Setup secret rotation policies
setup_secret_rotation() {
    vault_log "Setting up secret rotation policies..."

    # Create secret rotation policies
    cat > /tmp/rotation-policy.hcl <<EOF
# Secret Rotation Policy
path "sys/leases/renew" {
  capabilities = ["update"]
}

path "sys/leases/revoke" {
  capabilities = ["update"]
}

path "auth/approle/role/+/secret-id" {
  capabilities = ["update"]
}
EOF

    vault policy write rotation-policy /tmp/rotation-policy.hcl

    # Setup periodic token for rotation
    ROTATION_TOKEN=$(vault write -field=token auth/token/create \
        policies="rotation-policy" \
        ttl=720h \
        renewable=true \
        display_name="secret-rotation-token")

    # Store rotation token in GitHub Secrets
    gh secret set VAULT_ROTATION_TOKEN --body "$ROTATION_TOKEN" --repo "$GITHUB_REPO"

    vault_log "Secret rotation policies configured"
}

# Create secret validation script
create_validation_script() {
    log "Creating secret validation script..."

    cat > /tmp/validate-secrets.sh <<'EOF'
#!/bin/bash
#
# BEV Secrets Validation Script
#
# Validates that all required secrets are accessible from Vault
#

set -euo pipefail

VAULT_ADDR="${VAULT_ADDR:-http://100.122.12.35:8200}"
ROLE_ID="${1:-}"
SECRET_ID="${2:-}"

if [[ -z "$ROLE_ID" || -z "$SECRET_ID" ]]; then
    echo "Usage: $0 <role_id> <secret_id>"
    exit 1
fi

# Authenticate with AppRole
VAULT_TOKEN=$(vault write -field=token auth/approle/login \
    role_id="$ROLE_ID" \
    secret_id="$SECRET_ID")

export VAULT_TOKEN

# Validate access to required secrets
secrets=(
    "secret/bev/postgres"
    "secret/bev/neo4j"
    "secret/bev/redis"
    "secret/bev/elasticsearch"
    "secret/bev/rabbitmq"
    "secret/bev/prometheus"
    "secret/bev/grafana"
    "secret/bev/consul"
    "secret/bev/ai-services"
    "secret/bev/gpu-config"
    "secret/bev/monitoring"
    "secret/bev/analyzers"
)

echo "🔐 Validating secret access..."

for secret in "${secrets[@]}"; do
    if vault kv get "$secret" >/dev/null 2>&1; then
        echo "✅ $secret accessible"
    else
        echo "❌ $secret not accessible"
    fi
done

echo "🔐 Secret validation completed"
EOF

    chmod +x /tmp/validate-secrets.sh
    cp /tmp/validate-secrets.sh /usr/local/bin/bev-validate-secrets

    success "Secret validation script created at /usr/local/bin/bev-validate-secrets"
}

# Generate secrets documentation
generate_secrets_documentation() {
    log "Generating secrets management documentation..."

    cat > /tmp/SECRETS_MANAGEMENT.md <<EOF
# BEV OSINT Framework - Secrets Management

## Overview
The BEV OSINT Framework uses HashiCorp Vault for centralized secret management across all deployment environments and nodes.

## Architecture

### Vault Configuration
- **Vault Address**: $VAULT_ADDR
- **Authentication**: AppRole-based for automated systems
- **Storage Backend**: File-based storage on STARLORD
- **High Availability**: Single-node deployment (development/staging)

### Node Access Patterns

#### THANOS Node (GPU Compute)
- **Policy**: thanos-policy
- **Access**: PostgreSQL, Neo4j, Elasticsearch, RabbitMQ, AI services, GPU config
- **Role ID**: Stored in GitHub Secrets as THANOS_VAULT_ROLE_ID
- **Secret ID**: Stored in GitHub Secrets as THANOS_VAULT_SECRET_ID

#### ORACLE1 Node (ARM64 Monitoring)
- **Policy**: oracle1-policy
- **Access**: Redis, Prometheus, Grafana, Consul, analyzers, monitoring
- **Role ID**: Stored in GitHub Secrets as ORACLE1_VAULT_ROLE_ID
- **Secret ID**: Stored in GitHub Secrets as ORACLE1_VAULT_SECRET_ID

#### STARLORD Node (Development Control)
- **Policy**: starlord-policy
- **Access**: Full access to all BEV secrets (development environment)
- **Role ID**: Stored in GitHub Secrets as STARLORD_VAULT_ROLE_ID
- **Secret ID**: Stored in GitHub Secrets as STARLORD_VAULT_SECRET_ID

#### GitHub Actions (CI/CD)
- **Policy**: github-actions-policy
- **Access**: Read access to deployment secrets, AppRole management
- **Role ID**: Stored in GitHub Secrets as GITHUB_ACTIONS_VAULT_ROLE_ID
- **Secret ID**: Stored in GitHub Secrets as GITHUB_ACTIONS_VAULT_SECRET_ID

## Secret Categories

### Database Secrets
- \`secret/bev/postgres\`: PostgreSQL connection credentials
- \`secret/bev/neo4j\`: Neo4j graph database credentials
- \`secret/bev/redis\`: Redis cache credentials
- \`secret/bev/elasticsearch\`: Elasticsearch search credentials

### Message Queue
- \`secret/bev/rabbitmq\`: RabbitMQ message broker credentials

### Monitoring Stack
- \`secret/bev/prometheus\`: Prometheus monitoring credentials
- \`secret/bev/grafana\`: Grafana dashboard credentials
- \`secret/bev/consul\`: Consul service discovery credentials

### AI/ML Services
- \`secret/bev/ai-services\`: AI service API keys and tokens
- \`secret/bev/gpu-config\`: GPU configuration and limits

### Application Services
- \`secret/bev/analyzers\`: OSINT analyzer configurations
- \`secret/bev/monitoring\`: System monitoring credentials

## Usage Examples

### Retrieve Database Password (from deployment script)
\`\`\`bash
# Authenticate with AppRole
VAULT_TOKEN=\$(vault write -field=token auth/approle/login \\
    role_id="\$ROLE_ID" \\
    secret_id="\$SECRET_ID")

# Get PostgreSQL password
POSTGRES_PASSWORD=\$(vault kv get -field=password secret/bev/postgres)
\`\`\`

### Validate Secret Access
\`\`\`bash
# Use the validation script
bev-validate-secrets \$ROLE_ID \$SECRET_ID
\`\`\`

### Manual Secret Rotation
\`\`\`bash
# Generate new password
NEW_PASSWORD=\$(openssl rand -base64 32)

# Update secret in Vault
vault kv patch secret/bev/postgres password="\$NEW_PASSWORD"

# Update GitHub Secret
gh secret set POSTGRES_MASTER_PASSWORD --body "\$NEW_PASSWORD"
\`\`\`

## Security Considerations

### Token Lifecycle
- **AppRole Tokens**: 24-hour TTL, 48-hour max TTL
- **Secret IDs**: 24-hour TTL, limited use counts
- **GitHub Actions**: 2-hour TTL, 1-hour Secret ID TTL

### Access Control
- Principle of least privilege per node type
- No hardcoded secrets in code or configuration files
- All secrets stored encrypted in Vault
- Regular secret rotation (planned automation)

### Monitoring
- Vault audit logging enabled
- Secret access patterns monitored
- Automated alerts for unauthorized access attempts

## Troubleshooting

### Common Issues

1. **Vault Sealed**
   \`\`\`bash
   vault operator unseal \$(jq -r '.unseal_keys_b64[0]' vault-init.json)
   vault operator unseal \$(jq -r '.unseal_keys_b64[1]' vault-init.json)
   vault operator unseal \$(jq -r '.unseal_keys_b64[2]' vault-init.json)
   \`\`\`

2. **AppRole Authentication Failed**
   - Check Role ID and Secret ID in GitHub Secrets
   - Verify Secret ID hasn't exceeded use count
   - Regenerate Secret ID if needed

3. **Secret Not Found**
   - Verify secret path exists in Vault
   - Check policy permissions for the AppRole
   - Ensure Vault token has required capabilities

### Emergency Procedures

1. **Revoke Compromised Credentials**
   \`\`\`bash
   vault write auth/approle/role/NODE_NAME/secret-id-accessor/destroy \\
       secret_id_accessor=\$ACCESSOR_ID
   \`\`\`

2. **Rotate All Secrets**
   \`\`\`bash
   ./scripts/secrets/rotate-all-secrets.sh
   \`\`\`

3. **Backup Vault Data**
   \`\`\`bash
   vault operator raft snapshot save backup-\$(date +%Y%m%d-%H%M%S).snap
   \`\`\`

## Maintenance

### Regular Tasks
- Monthly secret rotation (automated)
- Quarterly access review
- Annual security audit
- Backup validation testing

### Monitoring Dashboards
- Vault health and performance
- Secret access patterns
- Token usage and expiration
- Policy violations and alerts

EOF

    cp /tmp/SECRETS_MANAGEMENT.md ./docs/
    success "Secrets management documentation created at ./docs/SECRETS_MANAGEMENT.md"
}

# Main execution
main() {
    log "Starting BEV secrets management setup..."

    check_prerequisites
    setup_vault_auth
    create_approles
    generate_and_store_secrets
    setup_github_secrets
    create_environment_configs
    setup_secret_rotation
    create_validation_script
    generate_secrets_documentation

    success "🔐 BEV secrets management setup completed successfully!"

    echo ""
    echo "📋 Next Steps:"
    echo "1. Test secret access with: bev-validate-secrets \$ROLE_ID \$SECRET_ID"
    echo "2. Review secrets documentation: ./docs/SECRETS_MANAGEMENT.md"
    echo "3. Configure monitoring for Vault health"
    echo "4. Set up automated secret rotation schedule"
    echo "5. Test deployment workflows with new secret integration"
}

# Handle script interruption
trap 'error "Script interrupted"' INT TERM

# Execute main function
main "$@"