#!/bin/bash

# Install Vault renewal service

set -euo pipefail

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [INSTALL] $*"
}

# Create directories
sudo mkdir -p /opt/vault /var/log
sudo chmod 755 /opt/vault

# Install systemd service
sudo cp config/vault-renewal.service /etc/systemd/system/
sudo systemctl daemon-reload

log "INFO: Vault renewal service installed"
log "INFO: Use 'sudo systemctl enable vault-renewal' to enable auto-start"
log "INFO: Use 'sudo systemctl start vault-renewal' to start service"
