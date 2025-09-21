# Security Team Policy
# Access for security analysts and SOC team members

# Read-only access to security-related paths
path "kv2/data/security/*" {
  capabilities = ["read"]
}

path "kv2/metadata/security/*" {
  capabilities = ["read", "list"]
}

# Read access to certificates for security analysis
path "pki/cert/*" {
  capabilities = ["read"]
}

path "pki/crl" {
  capabilities = ["read"]
}

# Access to security monitoring credentials
path "kv2/data/monitoring/*" {
  capabilities = ["read"]
}

# Database access for security analytics
path "database/creds/security-analyst" {
  capabilities = ["read"]
}

# Transit encryption for security tools
path "transit/encrypt/security-tools" {
  capabilities = ["update"]
}

path "transit/decrypt/security-tools" {
  capabilities = ["update"]
}

# SSH access for security operations
path "ssh/creds/security-ops" {
  capabilities = ["update"]
}

# Limited system information for troubleshooting
path "sys/health" {
  capabilities = ["read"]
}

path "sys/seal-status" {
  capabilities = ["read"]
}

path "sys/mounts" {
  capabilities = ["read"]
}

# Auth token management (own tokens only)
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

path "auth/token/revoke-self" {
  capabilities = ["update"]
}

# Identity information (read-only)
path "identity/entity/id/*" {
  capabilities = ["read"]
}

# Audit log access (read-only)
path "sys/audit" {
  capabilities = ["read"]
}