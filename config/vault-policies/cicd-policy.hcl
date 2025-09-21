# CI/CD Pipeline Policy
# Access for continuous integration and deployment systems

# Deployment secrets access
path "kv2/data/cicd/deployments/*" {
  capabilities = ["read"]
}

path "kv2/data/cicd/environments/*" {
  capabilities = ["read"]
}

# Container registry credentials
path "kv2/data/registry/*" {
  capabilities = ["read"]
}

# Cloud provider credentials for deployments
path "aws/sts/cicd-deploy" {
  capabilities = ["update"]
}

path "gcp/roleset/cicd-deploy/token" {
  capabilities = ["read", "update"]
}

# SSH keys for deployment access
path "ssh/creds/deployment" {
  capabilities = ["update"]
}

# PKI for service certificates during deployment
path "pki/issue/deployment-certs" {
  capabilities = ["update"]
}

# Database credentials for migrations
path "database/creds/migration" {
  capabilities = ["read"]
}

# Transit encryption for build artifacts
path "transit/encrypt/build-artifacts" {
  capabilities = ["update"]
}

path "transit/decrypt/build-artifacts" {
  capabilities = ["update"]
}

# Limited write access to deployment status
path "kv2/data/status/deployments/*" {
  capabilities = ["create", "update"]
}

# Read access to application configuration during deployment
path "kv2/data/app/*/config" {
  capabilities = ["read"]
}

# Token management for CI/CD runners
path "auth/token/lookup-self" {
  capabilities = ["read"]
}

path "auth/token/renew-self" {
  capabilities = ["update"]
}

# Kubernetes auth for container deployments
path "auth/kubernetes/login" {
  capabilities = ["update"]
}

# AppRole auth for service authentication
path "auth/approle/role/cicd/secret-id" {
  capabilities = ["update"]
}