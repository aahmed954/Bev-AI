# ORACLE Worker Policy
# Access for ORACLE1 enhancement research workers

# Research data access
path "kv2/data/oracle/research/*" {
  capabilities = ["read", "create", "update"]
}

path "kv2/metadata/oracle/research/*" {
  capabilities = ["read", "list"]
}

# API keys for research tools and services
path "kv2/data/apis/research/*" {
  capabilities = ["read"]
}

# Crypto research credentials
path "kv2/data/crypto/exchanges/*" {
  capabilities = ["read"]
}

path "kv2/data/crypto/wallets/*" {
  capabilities = ["read"]
}

# DRM analysis tools access
path "kv2/data/drm/tools/*" {
  capabilities = ["read"]
}

# Watermark analysis credentials
path "kv2/data/watermark/services/*" {
  capabilities = ["read"]
}

# Database access for research data storage
path "database/creds/research-writer" {
  capabilities = ["read"]
}

# Transit encryption for sensitive research data
path "transit/encrypt/research-data" {
  capabilities = ["update"]
}

path "transit/decrypt/research-data" {
  capabilities = ["update"]
}

# SSH access for remote research systems
path "ssh/creds/research-ops" {
  capabilities = ["update"]
}

# Cloud storage credentials for research artifacts
path "aws/sts/research-storage" {
  capabilities = ["update"]
}

# Results storage (write-only for workers)
path "kv2/data/oracle/results/*" {
  capabilities = ["create", "update"]
}

# Configuration access
path "kv2/data/oracle/config/*" {
  capabilities = ["read"]
}

# Token management
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Temporary secrets for short-lived research tasks
path "kv2/data/temp/research/*" {
  capabilities = ["create", "read", "update", "delete"]
}

# Approval workflow access
path "kv2/data/approvals/research/*" {
  capabilities = ["read", "create", "update"]
}