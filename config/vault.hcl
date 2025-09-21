# HashiCorp Vault Configuration
# Military-grade secrets management with comprehensive security policies

# Storage backend - File storage for development, use Consul/etcd for production
storage "file" {
  path = "/opt/vault/data"
}

# Alternative storage for production (uncomment and configure)
# storage "consul" {
#   address = "127.0.0.1:8500"
#   path    = "vault/"
# }

# Listener configuration - HTTPS only with strong TLS
listener "tcp" {
  address         = "0.0.0.0:8200"
  tls_cert_file   = "/opt/vault/tls/vault.crt"
  tls_key_file    = "/opt/vault/tls/vault.key"
  tls_min_version = "tls12"
  tls_cipher_suites = [
    "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
    "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
    "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305",
    "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305"
  ]
}

# Cluster configuration for HA deployment
cluster_addr  = "https://127.0.0.1:8201"
api_addr      = "https://127.0.0.1:8200"

# Disable mlock for development (enable in production)
disable_mlock = true

# Enable UI
ui = true

# Logging configuration
log_level = "INFO"
log_format = "json"

# Telemetry for monitoring
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname = true
  dogstatsd_addr = "localhost:8125"
  dogstatsd_tags = ["vault", "security"]
}

# Seal configuration - Auto-unseal with cloud KMS in production
# seal "awskms" {
#   region     = "us-west-2"
#   kms_key_id = "your-kms-key-id"
# }

# Default lease TTL
default_lease_ttl = "768h"   # 32 days
max_lease_ttl     = "8760h"  # 1 year

# Cache configuration
cache {
  use_auto_auth_token = true
}

# Entropy augmentation (production)
entropy "seal" {
  mode = "augmentation"
}

#------------------------------------------------------------------------------
# VAULT INITIALIZATION SCRIPT EMBEDDED AS COMMENTS
#------------------------------------------------------------------------------

# After starting Vault, run these commands to initialize and configure:

# 1. Initialize Vault
# vault operator init -key-shares=5 -key-threshold=3

# 2. Unseal Vault (run 3 times with different keys)
# vault operator unseal <key1>
# vault operator unseal <key2>
# vault operator unseal <key3>

# 3. Login with root token
# vault auth <root_token>

# 4. Enable auth methods
# vault auth enable userpass
# vault auth enable approle
# vault auth enable aws
# vault auth enable kubernetes
# vault auth enable ldap
# vault auth enable github

# 5. Enable secrets engines
# vault secrets enable -path=kv2 kv-v2
# vault secrets enable -path=database database
# vault secrets enable -path=transit transit
# vault secrets enable -path=pki pki
# vault secrets enable -path=ssh ssh
# vault secrets enable -path=aws aws
# vault secrets enable -path=gcp gcp

# 6. Configure PKI
# vault secrets tune -max-lease-ttl=87600h pki
# vault write pki/root/generate/internal \
#   common_name="BEV Root CA" \
#   ttl=87600h

# 7. Apply security policies (see policies below)

#------------------------------------------------------------------------------
# SECURITY POLICIES
#------------------------------------------------------------------------------