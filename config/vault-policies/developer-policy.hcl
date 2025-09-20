# Developer Policy
# Access for development team members

# Development environment secrets
path "kv2/data/dev/*" {
  capabilities = ["read", "create", "update", "delete"]
}

path "kv2/metadata/dev/*" {
  capabilities = ["read", "list"]
}

# Personal development secrets
path "kv2/data/personal/{{identity.entity.aliases.AUTH_METHOD_ACCESSOR.name}}/*" {
  capabilities = ["read", "create", "update", "delete"]
}

# Shared development configuration
path "kv2/data/shared/dev-config/*" {
  capabilities = ["read"]
}

# Development database credentials
path "database/creds/dev-full" {
  capabilities = ["read"]
}

# Transit encryption for development
path "transit/encrypt/dev-data" {
  capabilities = ["update"]
}

path "transit/decrypt/dev-data" {
  capabilities = ["update"]
}

# SSH access for development servers
path "ssh/creds/dev-access" {
  capabilities = ["update"]
}

# Development PKI certificates
path "pki/issue/dev-certs" {
  capabilities = ["update"]
}

# Cloud development credentials
path "aws/sts/dev-role" {
  capabilities = ["update"]
}

# Token management
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Read-only access to non-sensitive system information
path "sys/health" {
  capabilities = ["read"]
}

path "sys/mounts" {
  capabilities = ["read"]
}

# AppRole for development services
path "auth/approle/role/dev-services/role-id" {
  capabilities = ["read"]
}

path "auth/approle/role/dev-services/secret-id" {
  capabilities = ["update"]
}