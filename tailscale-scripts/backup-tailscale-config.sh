#!/bin/bash

# Tailscale Configuration Backup Script
# Backs up ACL policies, device configurations, DNS settings, and key information

set -euo pipefail

# Configuration
API_KEY="tskey-api-khe6MHjBEK11CNTRL-a1qHLAujhmMYiAvfwNqKnMPrmgiyGLH7"
BASE_URL="https://api.tailscale.com/api/v2"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$HOME/tailscale-backups"
RETENTION_DAYS=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
    exit 1
}

# Create backup directory
mkdir -p "$BACKUP_DIR"

log "Starting Tailscale configuration backup..."

# Function to make API calls
api_call() {
    local endpoint="$1"
    local output_file="$2"

    if curl -s -u "${API_KEY}:" \
        -H "Accept: application/json" \
        "${BASE_URL}${endpoint}" > "$output_file"; then
        log "Backed up: $(basename "$output_file")"
    else
        warn "Failed to backup: $endpoint"
        return 1
    fi
}

# Backup ACL policy
log "Backing up ACL policy..."
api_call "/tailnet/-/acl" "$BACKUP_DIR/acl-policy-$DATE.json"

# Backup device list
log "Backing up device configuration..."
api_call "/tailnet/-/devices" "$BACKUP_DIR/devices-$DATE.json"

# Backup DNS nameservers
log "Backing up DNS nameservers..."
api_call "/tailnet/-/dns/nameservers" "$BACKUP_DIR/dns-nameservers-$DATE.json"

# Backup DNS preferences
log "Backing up DNS preferences..."
api_call "/tailnet/-/dns/preferences" "$BACKUP_DIR/dns-preferences-$DATE.json"

# Backup keys information
log "Backing up keys information..."
api_call "/tailnet/-/keys" "$BACKUP_DIR/keys-$DATE.json"

# Backup individual device routes
log "Backing up device routes..."
ROUTES_DIR="$BACKUP_DIR/routes-$DATE"
mkdir -p "$ROUTES_DIR"

# Extract device IDs and back up routes for each
if [ -f "$BACKUP_DIR/devices-$DATE.json" ]; then
    jq -r '.devices[].id' "$BACKUP_DIR/devices-$DATE.json" | while read -r device_id; do
        if [ -n "$device_id" ]; then
            api_call "/device/${device_id}/routes" "$ROUTES_DIR/device-${device_id}-routes.json" || true
        fi
    done
fi

# Create backup manifest
log "Creating backup manifest..."
cat > "$BACKUP_DIR/manifest-$DATE.txt" << EOF
Tailscale Configuration Backup
Date: $(date)
API Key: ${API_KEY:0:20}...
Files:
$(ls -la "$BACKUP_DIR"/*-$DATE.* "$ROUTES_DIR"/ 2>/dev/null | grep -v '^total' || echo "No files")

Device Summary:
$(jq -r '.devices[] | "\(.name) (\(.hostname)) - \(.os) \(.clientVersion)"' "$BACKUP_DIR/devices-$DATE.json" 2>/dev/null || echo "Failed to parse devices")

ACL Summary:
$(jq -r '.acls[] | "Action: \(.action), Src: \(.src | join(", ")), Dst: \(.dst | join(", "))"' "$BACKUP_DIR/acl-policy-$DATE.json" 2>/dev/null || echo "Failed to parse ACLs")
EOF

# Clean up old backups
log "Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "*-[0-9]*" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true

# Create latest symlinks for easy access
log "Creating latest backup symlinks..."
cd "$BACKUP_DIR"
for file in *-$DATE.*; do
    if [ -f "$file" ]; then
        base_name=$(echo "$file" | sed "s/-$DATE//")
        ln -sf "$file" "$base_name-latest"
    fi
done

# Backup verification
log "Verifying backup integrity..."
BACKUP_COUNT=$(ls -1 *-$DATE.* 2>/dev/null | wc -l)
if [ "$BACKUP_COUNT" -ge 5 ]; then
    log "Backup completed successfully! $BACKUP_COUNT files backed up to $BACKUP_DIR"
else
    warn "Backup may be incomplete. Only $BACKUP_COUNT files were backed up."
fi

# Display backup summary
echo
echo "=== Backup Summary ==="
echo "Location: $BACKUP_DIR"
echo "Files:"
ls -lh "$BACKUP_DIR"/*-$DATE.* 2>/dev/null || echo "No files found"
echo
echo "Latest backups available as:"
ls -lh "$BACKUP_DIR"/*-latest 2>/dev/null || echo "No latest links found"

log "Backup process completed!"