#!/bin/bash
#
# BEV OSINT Framework - Secret Rotation Script
#
# Automates rotation of secrets across all BEV services
# and updates them in both Vault and GitHub Secrets
#

set -euo pipefail

# Configuration
VAULT_ADDR="${VAULT_ADDR:-http://100.122.12.35:8200}"
VAULT_TOKEN="${VAULT_TOKEN:-}"
GITHUB_REPO="${GITHUB_REPO:-starlord/Bev}"
ROTATION_LOG="/var/log/bev-secret-rotation.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

log() {
    local message="$1"
    echo -e "${BLUE}[ROTATION]${NC} $message"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $message" >> "$ROTATION_LOG"
}

error() {
    local message="$1"
    echo -e "${RED}[ERROR]${NC} $message" >&2
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $message" >> "$ROTATION_LOG"
    exit 1
}

success() {
    local message="$1"
    echo -e "${GREEN}[SUCCESS]${NC} $message"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $message" >> "$ROTATION_LOG"
}

warn() {
    local message="$1"
    echo -e "${YELLOW}[WARNING]${NC} $message"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $message" >> "$ROTATION_LOG"
}

vault_log() {
    local message="$1"
    echo -e "${PURPLE}[VAULT]${NC} $message"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] VAULT: $message" >> "$ROTATION_LOG"
}

# Initialize rotation log
init_log() {
    mkdir -p "$(dirname "$ROTATION_LOG")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting BEV secret rotation..." >> "$ROTATION_LOG"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for secret rotation..."

    # Check Vault CLI
    if ! command -v vault &> /dev/null; then
        error "Vault CLI is required but not installed"
    fi

    # Check GitHub CLI
    if ! command -v gh &> /dev/null; then
        error "GitHub CLI (gh) is required but not installed"
    fi

    # Check jq
    if ! command -v jq &> /dev/null; then
        error "jq is required but not installed"
    fi

    # Check openssl
    if ! command -v openssl &> /dev/null; then
        error "openssl is required but not installed"
    fi

    # Check consul (for consul keygen)
    if ! command -v consul &> /dev/null; then
        warn "consul not found - will use openssl for Consul keys"
    fi

    # Check uuidgen
    if ! command -v uuidgen &> /dev/null; then
        warn "uuidgen not found - will use openssl for UUIDs"
    fi

    # Check Vault connectivity
    if ! curl -s --connect-timeout 10 "$VAULT_ADDR/v1/sys/health" | jq -e '.sealed == false' > /dev/null 2>&1; then
        error "Vault is not accessible or is sealed. Address: $VAULT_ADDR"
    fi

    # Check GitHub authentication
    if ! gh auth status &>/dev/null; then
        error "GitHub CLI not authenticated. Run 'gh auth login' first"
    fi

    success "Prerequisites check completed"
}

# Authenticate with Vault
authenticate_vault() {
    vault_log "Authenticating with Vault..."

    if [[ -z "$VAULT_TOKEN" ]]; then
        # Try to get token from rotation token if available
        if gh secret list --repo "$GITHUB_REPO" | grep -q "VAULT_ROTATION_TOKEN"; then
            VAULT_TOKEN=$(gh secret get VAULT_ROTATION_TOKEN --repo "$GITHUB_REPO")
        else
            error "No VAULT_TOKEN provided and no rotation token found"
        fi
    fi

    vault auth -method=token token="$VAULT_TOKEN"
    vault_log "Vault authentication successful"
}

# Generate secure password
generate_password() {
    local length="${1:-32}"
    openssl rand -base64 "$length" | tr -d '=' | tr '+/' '-_'
}

# Generate secure key
generate_key() {
    local length="${1:-32}"
    openssl rand -base64 "$length"
}

# Generate UUID
generate_uuid() {
    if command -v uuidgen &> /dev/null; then
        uuidgen
    else
        openssl rand -hex 16 | sed 's/\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)\(..\)/\1\2\3\4-\5\6-\7\8-\9\10-\11\12\13\14\15\16/'
    fi
}

# Generate Consul encrypt key
generate_consul_key() {
    if command -v consul &> /dev/null; then
        consul keygen
    else
        openssl rand -base64 24
    fi
}

# Rotate database secrets
rotate_database_secrets() {
    log "Rotating database secrets..."

    # PostgreSQL
    vault_log "Rotating PostgreSQL secrets..."
    vault kv patch secret/bev/postgres \
        password="$(generate_password 32)"

    # Neo4j
    vault_log "Rotating Neo4j secrets..."
    vault kv patch secret/bev/neo4j \
        password="$(generate_password 24)"

    # Redis
    vault_log "Rotating Redis secrets..."
    vault kv patch secret/bev/redis \
        password="$(generate_password 16)"

    # Elasticsearch
    vault_log "Rotating Elasticsearch secrets..."
    vault kv patch secret/bev/elasticsearch \
        password="$(generate_password 20)"

    success "Database secrets rotated"
}

# Rotate message queue secrets
rotate_message_queue_secrets() {
    log "Rotating message queue secrets..."

    # RabbitMQ
    vault_log "Rotating RabbitMQ secrets..."
    vault kv patch secret/bev/rabbitmq \
        password="$(generate_password 20)"

    success "Message queue secrets rotated"
}

# Rotate monitoring secrets
rotate_monitoring_secrets() {
    log "Rotating monitoring secrets..."

    # Prometheus
    vault_log "Rotating Prometheus secrets..."
    vault kv patch secret/bev/prometheus \
        admin_password="$(generate_password 16)"

    # Grafana
    vault_log "Rotating Grafana secrets..."
    vault kv patch secret/bev/grafana \
        admin_password="$(generate_password 16)" \
        secret_key="$(generate_key 32)"

    # Consul
    vault_log "Rotating Consul secrets..."
    vault kv patch secret/bev/consul \
        encrypt_key="$(generate_consul_key)" \
        gossip_key="$(generate_key 24)" \
        master_token="$(generate_uuid)"

    success "Monitoring secrets rotated"
}

# Rotate AI service secrets
rotate_ai_secrets() {
    log "Rotating AI service secrets..."

    # AI Services
    vault_log "Rotating AI service secrets..."
    vault kv patch secret/bev/ai-services \
        api_key="$(generate_key 32)" \
        model_encryption_key="$(generate_key 32)" \
        inference_token="$(generate_key 24)"

    success "AI service secrets rotated"
}

# Rotate application secrets
rotate_application_secrets() {
    log "Rotating application secrets..."

    # Monitoring
    vault_log "Rotating monitoring secrets..."
    vault kv patch secret/bev/monitoring \
        notification_token="$(generate_key 20)"

    # Analyzers
    vault_log "Rotating analyzer secrets..."
    vault kv patch secret/bev/analyzers \
        cache_encryption_key="$(generate_key 32)" \
        session_token="$(generate_key 24)"

    success "Application secrets rotated"
}

# Rotate AppRole secret IDs
rotate_approle_secrets() {
    log "Rotating AppRole secret IDs..."

    local roles=("thanos" "oracle1" "starlord" "github-actions")

    for role in "${roles[@]}"; do
        vault_log "Rotating $role AppRole secret ID..."

        # Generate new secret ID
        local new_secret_id
        new_secret_id=$(vault write -field=secret_id auth/approle/role/"$role"/secret-id)

        # Update GitHub Secret
        local github_secret_name
        case "$role" in
            "github-actions")
                github_secret_name="GITHUB_ACTIONS_VAULT_SECRET_ID"
                ;;
            *)
                github_secret_name="$(echo "$role" | tr '[:lower:]' '[:upper:]')_VAULT_SECRET_ID"
                ;;
        esac

        gh secret set "$github_secret_name" --body "$new_secret_id" --repo "$GITHUB_REPO"
        vault_log "$role AppRole secret ID rotated and updated in GitHub"
    done

    success "AppRole secret IDs rotated"
}

# Rotate GitHub-specific secrets
rotate_github_secrets() {
    log "Rotating GitHub-specific secrets..."

    # Deployment keys
    gh secret set POSTGRES_ENCRYPTION_KEY --body "$(generate_key 32)" --repo "$GITHUB_REPO"
    gh secret set REDIS_ENCRYPTION_KEY --body "$(generate_key 32)" --repo "$GITHUB_REPO"
    gh secret set DEPLOYMENT_WEBHOOK_SECRET --body "$(generate_key 24)" --repo "$GITHUB_REPO"

    # Environment-specific deployment keys
    gh secret set PRODUCTION_DEPLOYMENT_KEY --body "$(generate_key 32)" --repo "$GITHUB_REPO"
    gh secret set STAGING_DEPLOYMENT_KEY --body "$(generate_key 32)" --repo "$GITHUB_REPO"
    gh secret set DEVELOPMENT_DEPLOYMENT_KEY --body "$(generate_key 32)" --repo "$GITHUB_REPO"

    success "GitHub-specific secrets rotated"
}

# Test secret access after rotation
test_secret_access() {
    log "Testing secret access after rotation..."

    local secrets=(
        "secret/bev/postgres"
        "secret/bev/neo4j"
        "secret/bev/redis"
        "secret/bev/elasticsearch"
        "secret/bev/rabbitmq"
        "secret/bev/prometheus"
        "secret/bev/grafana"
        "secret/bev/consul"
        "secret/bev/ai-services"
        "secret/bev/monitoring"
        "secret/bev/analyzers"
    )

    local failed_secrets=()

    for secret in "${secrets[@]}"; do
        if vault kv get "$secret" >/dev/null 2>&1; then
            vault_log "✅ $secret accessible"
        else
            vault_log "❌ $secret not accessible"
            failed_secrets+=("$secret")
        fi
    done

    if [[ ${#failed_secrets[@]} -gt 0 ]]; then
        error "Failed to access rotated secrets: ${failed_secrets[*]}"
    fi

    success "All rotated secrets accessible"
}

# Test AppRole authentication after rotation
test_approle_auth() {
    log "Testing AppRole authentication after rotation..."

    local roles=("thanos" "oracle1" "starlord" "github-actions")

    for role in "${roles[@]}"; do
        vault_log "Testing $role AppRole authentication..."

        # Get Role ID and Secret ID
        local role_id
        role_id=$(vault read -field=role_id auth/approle/role/"$role"/role-id)

        local secret_id_key
        case "$role" in
            "github-actions")
                secret_id_key="GITHUB_ACTIONS_VAULT_SECRET_ID"
                ;;
            *)
                secret_id_key="$(echo "$role" | tr '[:lower:]' '[:upper:]')_VAULT_SECRET_ID"
                ;;
        esac

        local secret_id
        secret_id=$(gh secret get "$secret_id_key" --repo "$GITHUB_REPO")

        # Test authentication
        if vault write auth/approle/login role_id="$role_id" secret_id="$secret_id" >/dev/null 2>&1; then
            vault_log "✅ $role AppRole authentication successful"
        else
            vault_log "❌ $role AppRole authentication failed"
            error "$role AppRole authentication test failed"
        fi
    done

    success "All AppRole authentication tests passed"
}

# Create rotation report
create_rotation_report() {
    log "Creating rotation report..."

    local report_file="/tmp/bev-rotation-report-$(date +%Y%m%d-%H%M%S).json"

    cat > "$report_file" <<EOF
{
  "rotation_timestamp": "$(date -Iseconds)",
  "vault_address": "$VAULT_ADDR",
  "github_repository": "$GITHUB_REPO",
  "rotated_secrets": {
    "database_secrets": [
      "secret/bev/postgres",
      "secret/bev/neo4j",
      "secret/bev/redis",
      "secret/bev/elasticsearch"
    ],
    "message_queue_secrets": [
      "secret/bev/rabbitmq"
    ],
    "monitoring_secrets": [
      "secret/bev/prometheus",
      "secret/bev/grafana",
      "secret/bev/consul"
    ],
    "ai_service_secrets": [
      "secret/bev/ai-services"
    ],
    "application_secrets": [
      "secret/bev/monitoring",
      "secret/bev/analyzers"
    ],
    "approle_secrets": [
      "auth/approle/role/thanos/secret-id",
      "auth/approle/role/oracle1/secret-id",
      "auth/approle/role/starlord/secret-id",
      "auth/approle/role/github-actions/secret-id"
    ],
    "github_secrets": [
      "POSTGRES_ENCRYPTION_KEY",
      "REDIS_ENCRYPTION_KEY",
      "DEPLOYMENT_WEBHOOK_SECRET",
      "PRODUCTION_DEPLOYMENT_KEY",
      "STAGING_DEPLOYMENT_KEY",
      "DEVELOPMENT_DEPLOYMENT_KEY"
    ]
  },
  "validation_status": {
    "vault_secrets_accessible": true,
    "approle_authentication": true,
    "github_secrets_updated": true
  },
  "next_rotation_due": "$(date -d '+30 days' -Iseconds)",
  "rotation_log": "$ROTATION_LOG"
}
EOF

    # Store report in Vault
    vault kv put secret/bev/rotation-reports/$(date +%Y%m%d-%H%M%S) @"$report_file"

    success "Rotation report created: $report_file"
    log "Report also stored in Vault at secret/bev/rotation-reports/$(date +%Y%m%d-%H%M%S)"
}

# Send rotation notification
send_notification() {
    log "Sending rotation notification..."

    # Create notification message
    local notification_file="/tmp/bev-rotation-notification.md"

    cat > "$notification_file" <<EOF
# 🔐 BEV Secret Rotation Completed

**Rotation Time**: $(date '+%Y-%m-%d %H:%M:%S %Z')
**Repository**: $GITHUB_REPO
**Vault Address**: $VAULT_ADDR

## Rotated Components

### 🗄️ Database Secrets
- PostgreSQL credentials
- Neo4j credentials
- Redis credentials
- Elasticsearch credentials

### 📊 Monitoring Stack
- Prometheus admin credentials
- Grafana admin credentials and secret key
- Consul encryption and gossip keys

### 🤖 AI Services
- AI service API keys
- Model encryption keys
- Inference tokens

### 🔑 Access Credentials
- All AppRole secret IDs refreshed
- GitHub repository secrets updated
- Deployment encryption keys rotated

## Validation Results
- ✅ All Vault secrets accessible
- ✅ AppRole authentication verified
- ✅ GitHub secrets synchronized

## Next Actions
- Services will automatically pick up new credentials on restart
- Monitor for any authentication issues
- Next rotation scheduled: $(date -d '+30 days' '+%Y-%m-%d')

**Rotation Log**: $ROTATION_LOG
EOF

    # In a real environment, this would send to Slack, Teams, email, etc.
    success "Rotation notification prepared: $notification_file"
    log "Manual notification recommended for production deployments"
}

# Cleanup old rotation artifacts
cleanup_old_artifacts() {
    log "Cleaning up old rotation artifacts..."

    # Clean up old rotation reports (keep last 10)
    local old_reports
    old_reports=$(vault kv list secret/bev/rotation-reports/ | tail -n +2 | head -n -10)

    if [[ -n "$old_reports" ]]; then
        while IFS= read -r report; do
            vault kv delete "secret/bev/rotation-reports/$report"
            vault_log "Deleted old rotation report: $report"
        done <<< "$old_reports"
    fi

    # Clean up old log files (keep last 30 days)
    find "$(dirname "$ROTATION_LOG")" -name "bev-secret-rotation.log.*" -mtime +30 -delete 2>/dev/null || true

    success "Old rotation artifacts cleaned up"
}

# Main execution
main() {
    init_log
    log "Starting BEV comprehensive secret rotation..."

    check_prerequisites
    authenticate_vault

    # Rotate all secret categories
    rotate_database_secrets
    rotate_message_queue_secrets
    rotate_monitoring_secrets
    rotate_ai_secrets
    rotate_application_secrets
    rotate_approle_secrets
    rotate_github_secrets

    # Validate rotation
    test_secret_access
    test_approle_auth

    # Reporting and cleanup
    create_rotation_report
    send_notification
    cleanup_old_artifacts

    success "🔐 BEV secret rotation completed successfully!"

    echo ""
    echo "📋 Rotation Summary:"
    echo "• Database credentials: Rotated"
    echo "• Monitoring stack: Rotated"
    echo "• AI services: Rotated"
    echo "• AppRole credentials: Rotated"
    echo "• GitHub secrets: Updated"
    echo ""
    echo "🔄 Next rotation due: $(date -d '+30 days' '+%Y-%m-%d')"
    echo "📊 Rotation log: $ROTATION_LOG"
}

# Usage information
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --vault-addr ADDR    Vault server address (default: $VAULT_ADDR)"
    echo "  --vault-token TOKEN  Vault authentication token"
    echo "  --github-repo REPO   GitHub repository (default: $GITHUB_REPO)"
    echo "  --log-file FILE      Custom log file path"
    echo "  --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  VAULT_ADDR           Vault server address"
    echo "  VAULT_TOKEN          Vault authentication token"
    echo "  GITHUB_REPO          GitHub repository"
    echo ""
    echo "Examples:"
    echo "  $0                                          # Use defaults"
    echo "  $0 --vault-token \$TOKEN                     # Specify token"
    echo "  $0 --github-repo user/repo                  # Specify repository"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vault-addr)
            VAULT_ADDR="$2"
            shift 2
            ;;
        --vault-token)
            VAULT_TOKEN="$2"
            shift 2
            ;;
        --github-repo)
            GITHUB_REPO="$2"
            shift 2
            ;;
        --log-file)
            ROTATION_LOG="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Handle script interruption
trap 'error "Script interrupted"' INT TERM

# Execute main function
main "$@"