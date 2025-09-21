# HashiCorp Vault Development Configuration
# For automated renewal system setup

# Storage backend - File storage for development
storage "file" {
  path = "/opt/vault/data"
}

# Listener configuration - HTTP for development
listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = 1
}

# Cluster configuration
cluster_addr  = "http://127.0.0.1:8201"
api_addr      = "http://127.0.0.1:8200"

# Disable mlock for development
disable_mlock = true

# Enable UI
ui = true

# Logging configuration
log_level = "INFO"