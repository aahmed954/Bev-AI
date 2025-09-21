#!/bin/bash
#
# BEV Advanced Avatar Startup Script
# Starts the avatar system with RTX 4090 optimization
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEV_ROOT="/home/starlord/Projects/Bev"
LOG_FILE="$BEV_ROOT/logs/avatar-startup.log"
PID_FILE="$BEV_ROOT/logs/avatar.pid"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $*" | tee -a "$LOG_FILE"
}

# Cleanup function
cleanup() {
    log "Cleanup triggered by signal"
    if [[ -f "$PID_FILE" ]]; then
        local pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log "Terminating avatar process: $pid"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 5
            if kill -0 "$pid" 2>/dev/null; then
                log "Force killing avatar process: $pid"
                kill -KILL "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
    fi
    
    # Clean up GPU memory
    if command -v nvidia-smi &>/dev/null; then
        python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    fi
    
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT SIGHUP

log "=== BEV Advanced Avatar Starting ==="

# Change to BEV directory
cd "$BEV_ROOT"

# Set up environment variables for RTX 4090 optimization
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
export NVIDIA_VISIBLE_DEVICES=0
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
export GPU_MEMORY_FRACTION=0.6
export TORCH_CUDNN_V8_API_ENABLED=1

# BEV specific environment
export BEV_AVATAR_CONFIG="$BEV_ROOT/config/avatar.yaml"
export BEV_REDIS_URL="redis://localhost:6379"
export BEV_MCP_SERVER_URL="http://localhost:3010"
export BEV_AVATAR_PORT=8080
export UVICORN_HOST=0.0.0.0
export UVICORN_PORT=8080
export UVICORN_LOG_LEVEL=info

# Python path setup
export PYTHONPATH="$BEV_ROOT/src:$BEV_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

log "Environment variables configured"

# Verify GPU is ready
if ! nvidia-smi &>/dev/null; then
    log "ERROR: NVIDIA GPU not accessible"
    exit 1
fi

# Set GPU performance mode
log "Optimizing GPU performance..."
sudo nvidia-smi -pm 1 2>/dev/null || log "WARNING: Could not enable persistence mode"
sudo nvidia-smi -ac 10501,2230 2>/dev/null || log "WARNING: Could not set application clocks"

# Clear any existing GPU memory
log "Clearing GPU memory..."
python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f'GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
" 2>>"$LOG_FILE"

# Validate avatar controller exists
AVATAR_CONTROLLER="$BEV_ROOT/src/avatar/advanced_avatar_controller.py"
if [[ ! -f "$AVATAR_CONTROLLER" ]]; then
    log "ERROR: Avatar controller not found: $AVATAR_CONTROLLER"
    exit 1
fi

# Create avatar configuration if it doesn't exist
if [[ ! -f "$BEV_AVATAR_CONFIG" ]]; then
    log "Creating default avatar configuration..."
    mkdir -p "$(dirname "$BEV_AVATAR_CONFIG")"
    cat > "$BEV_AVATAR_CONFIG" << 'EOF'
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

rendering:
  gaussian_splatting: true
  real_time_optimization: true
  memory_management: "aggressive"

health_check:
  enabled: true
  interval_seconds: 30
  endpoint: "/health"
EOF
fi

# Start Redis connection test
log "Testing Redis connection..."
if ! timeout 10 redis-cli -h localhost -p 6379 ping &>/dev/null; then
    log "ERROR: Cannot connect to Redis"
    exit 1
fi

# Clear any existing avatar state in Redis
redis-cli -h localhost -p 6379 DEL "avatar:state" "avatar:health" "avatar:performance" 2>/dev/null || true

# Set up avatar service health endpoint
log "Setting up health monitoring..."
cat > "$BEV_ROOT/logs/avatar-health.py" << 'EOF'
#!/usr/bin/env python3
import asyncio
import aiohttp
import time
import json
import sys

async def health_monitor():
    """Monitor avatar service health"""
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8080/health', timeout=5) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        timestamp = int(time.time())
                        health_record = {
                            'timestamp': timestamp,
                            'status': health_data.get('status', 'unknown'),
                            'uptime': health_data.get('uptime', 0),
                            'gpu_utilization': health_data.get('gpu_utilization', 0)
                        }
                        with open('/home/starlord/Projects/Bev/logs/avatar-health.json', 'w') as f:
                            json.dump(health_record, f)
                    else:
                        print(f"Health check failed: {response.status}")
        except Exception as e:
            print(f"Health monitor error: {e}")
        
        await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(health_monitor())
EOF

chmod +x "$BEV_ROOT/logs/avatar-health.py"

# Start the avatar service
log "Starting BEV Advanced Avatar Controller..."

# Use uvicorn to start the FastAPI app with optimized settings
exec python3 -m uvicorn src.avatar.advanced_avatar_controller:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --log-level info \
    --access-log \
    --use-colors \
    --reload-delay 1 \
    --timeout-keep-alive 30 \
    --timeout-graceful-shutdown 10 \
    --max-concurrency 1000 \
    --limit-max-requests 10000 \
    --backlog 2048 \
    >> "$LOG_FILE" 2>&1 &

# Get the PID and save it
AVATAR_PID=$!
echo "$AVATAR_PID" > "$PID_FILE"

log "Avatar service started with PID: $AVATAR_PID"

# Start health monitoring in background
python3 "$BEV_ROOT/logs/avatar-health.py" >> "$BEV_ROOT/logs/avatar-health.log" 2>&1 &
HEALTH_PID=$!
echo "$HEALTH_PID" > "$BEV_ROOT/logs/health-monitor.pid"

log "Health monitor started with PID: $HEALTH_PID"

# Wait for avatar service to be ready
log "Waiting for avatar service to be ready..."
for i in {1..30}; do
    if curl -s -f http://localhost:8080/health >/dev/null 2>&1; then
        log "Avatar service is ready and responding"
        break
    fi
    if [[ $i -eq 30 ]]; then
        log "ERROR: Avatar service failed to start within 30 seconds"
        cleanup
        exit 1
    fi
    sleep 1
done

# Verify the service is actually running
if ! kill -0 "$AVATAR_PID" 2>/dev/null; then
    log "ERROR: Avatar process died immediately after startup"
    exit 1
fi

# Set up systemd watchdog notification
if [[ -n "${WATCHDOG_USEC:-}" ]]; then
    # Calculate watchdog interval (half of timeout)
    WATCHDOG_INTERVAL=$((WATCHDOG_USEC / 2000000))
    log "Systemd watchdog enabled with ${WATCHDOG_INTERVAL}s interval"
    
    # Start watchdog notification loop
    (
        while kill -0 "$AVATAR_PID" 2>/dev/null; do
            sleep "$WATCHDOG_INTERVAL"
            systemd-notify --pid="$AVATAR_PID" WATCHDOG=1 2>/dev/null || true
        done
    ) &
fi

log "BEV Advanced Avatar startup completed successfully"
log "Avatar PID: $AVATAR_PID"
log "Health monitor PID: $HEALTH_PID"
log "Service endpoint: http://localhost:8080"
log "WebSocket endpoint: ws://localhost:8080/ws"

# Keep the script running and forward signals to the avatar process
wait "$AVATAR_PID"
EXIT_CODE=$?

log "Avatar process exited with code: $EXIT_CODE"
cleanup
exit $EXIT_CODE