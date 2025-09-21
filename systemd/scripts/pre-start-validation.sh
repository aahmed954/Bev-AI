#!/bin/bash
#
# BEV Advanced Avatar Pre-Start Validation Script
# Validates system readiness before starting avatar service
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEV_ROOT="/home/starlord/Projects/Bev"
LOG_FILE="$BEV_ROOT/logs/avatar-prestart.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log "=== BEV Avatar Pre-Start Validation Starting ==="

# Check if running as correct user
if [[ "$USER" != "starlord" ]]; then
    log "ERROR: Service must run as user 'starlord', currently running as '$USER'"
    exit 1
fi

# Validate working directory
if [[ ! -d "$BEV_ROOT" ]]; then
    log "ERROR: BEV root directory not found: $BEV_ROOT"
    exit 1
fi

cd "$BEV_ROOT"

# Check for essential directories
REQUIRED_DIRS=(
    "src/avatar"
    "config"
    "logs"
    "data"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        log "ERROR: Required directory missing: $dir"
        exit 1
    fi
done

# Check for avatar controller file
AVATAR_CONTROLLER="$BEV_ROOT/src/avatar/advanced_avatar_controller.py"
if [[ ! -f "$AVATAR_CONTROLLER" ]]; then
    log "ERROR: Avatar controller not found: $AVATAR_CONTROLLER"
    exit 1
fi

# Validate configuration file
AVATAR_CONFIG="$BEV_ROOT/config/avatar.yaml"
if [[ ! -f "$AVATAR_CONFIG" ]]; then
    log "WARNING: Avatar config not found, will use defaults: $AVATAR_CONFIG"
    # Create default config if it doesn't exist
    mkdir -p "$(dirname "$AVATAR_CONFIG")"
    cat > "$AVATAR_CONFIG" << 'EOF'
# BEV Advanced Avatar Configuration
avatar:
  model_path: "/app/models/avatar.model"
  device: "cuda:0"
  memory_fraction: 0.6
  fps: 30
  resolution:
    width: 1024
    height: 1024

osint:
  redis_url: "redis://localhost:6379"
  mcp_server_url: "http://localhost:3010"
  
performance:
  batch_size: 1
  max_concurrent_requests: 10
  cache_size: 1000

emotion_engine:
  response_time_ms: 100
  transition_speed: 0.5
  
tts:
  enabled: true
  voice_model: "bark"
  sample_rate: 24000
EOF
    log "Created default avatar configuration"
fi

# Check Python environment
if ! command -v python3 &> /dev/null; then
    log "ERROR: Python3 not found in PATH"
    exit 1
fi

# Validate Python dependencies
PYTHON_DEPS=(
    "torch"
    "torchvision" 
    "fastapi"
    "uvicorn"
    "redis"
    "numpy"
    "asyncio"
    "websockets"
)

log "Checking Python dependencies..."
for dep in "${PYTHON_DEPS[@]}"; do
    if ! python3 -c "import $dep" 2>/dev/null; then
        log "ERROR: Python dependency missing: $dep"
        exit 1
    fi
done

# Check disk space (minimum 10GB)
AVAILABLE_SPACE=$(df "$BEV_ROOT" | awk 'NR==2 {print $4}')
REQUIRED_SPACE=$((10 * 1024 * 1024))  # 10GB in KB

if [[ "$AVAILABLE_SPACE" -lt "$REQUIRED_SPACE" ]]; then
    log "ERROR: Insufficient disk space. Available: ${AVAILABLE_SPACE}KB, Required: ${REQUIRED_SPACE}KB"
    exit 1
fi

# Check memory (minimum 8GB)
AVAILABLE_MEM=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
REQUIRED_MEM=$((8 * 1024 * 1024))  # 8GB in KB

if [[ "$AVAILABLE_MEM" -lt "$REQUIRED_MEM" ]]; then
    log "WARNING: Low available memory. Available: ${AVAILABLE_MEM}KB, Recommended: ${REQUIRED_MEM}KB"
fi

# Validate network connectivity for Redis
if ! timeout 5 redis-cli -h localhost -p 6379 ping &>/dev/null; then
    log "ERROR: Cannot connect to Redis at localhost:6379"
    exit 1
fi

# Check if avatar port is available
AVATAR_PORT=8080
if ss -tuln | grep -q ":$AVATAR_PORT "; then
    log "ERROR: Avatar port $AVATAR_PORT is already in use"
    exit 1
fi

# Validate log rotation setup
if [[ ! -f "/etc/logrotate.d/bev-avatar" ]]; then
    log "WARNING: Log rotation not configured for BEV avatar"
fi

log "Pre-start validation completed successfully"
log "=== BEV Avatar Pre-Start Validation Complete ==="

exit 0