# Vault Admin Policy
# Full administrative access to Vault cluster

# System backend operations
path "sys/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Auth method management
path "auth/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Secrets engine management
path "secret/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "kv2/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Database secrets engine
path "database/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Transit encryption
path "transit/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# PKI management
path "pki/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# SSH secrets engine
path "ssh/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Cloud secrets engines
path "aws/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

path "gcp/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Identity management
path "identity/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Audit log management
path "sys/audit" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

path "sys/audit/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}

# Policy management
path "sys/policies/acl/*" {
  capabilities = ["create", "read", "update", "delete", "list"]
}

# Leases management
path "sys/leases/*" {
  capabilities = ["create", "read", "update", "delete", "list", "sudo"]
}