# Application Policy
# Access for application services and microservices

# Application secrets access
path "kv2/data/app/{{identity.entity.aliases.AUTH_METHOD_ACCESSOR.name}}/*" {
  capabilities = ["read"]
}

path "kv2/metadata/app/{{identity.entity.aliases.AUTH_METHOD_ACCESSOR.name}}/*" {
  capabilities = ["read", "list"]
}

# Shared application configuration
path "kv2/data/shared/config/*" {
  capabilities = ["read"]
}

# Database credentials for applications
path "database/creds/app-read" {
  capabilities = ["read"]
}

path "database/creds/app-write" {
  capabilities = ["read"]
}

# Transit encryption for application data
path "transit/encrypt/app-data" {
  capabilities = ["update"]
}

path "transit/decrypt/app-data" {
  capabilities = ["update"]
}

path "transit/datakey/plaintext/app-data" {
  capabilities = ["update"]
}

# PKI for service-to-service communication
path "pki/issue/service-certs" {
  capabilities = ["update"]
}

# SSH access for application deployments
path "ssh/creds/app-deploy" {
  capabilities = ["update"]
}

# Cloud provider credentials
path "aws/sts/app-role" {
  capabilities = ["update"]
}

path "gcp/roleset/app-service/token" {
  capabilities = ["read", "update"]
}

# Token management
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Health check access
path "sys/health" {
  capabilities = ["read"]
}