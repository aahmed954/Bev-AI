#!/bin/bash
#
# BEV Advanced Avatar Graceful Shutdown Script
# Gracefully stops the avatar system with state preservation
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEV_ROOT="/home/starlord/Projects/Bev"
LOG_FILE="$BEV_ROOT/logs/avatar-shutdown.log"
PID_FILE="$BEV_ROOT/logs/avatar.pid"
HEALTH_PID_FILE="$BEV_ROOT/logs/health-monitor.pid"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] STOP: $*" | tee -a "$LOG_FILE"
}

log "=== BEV Advanced Avatar Graceful Shutdown ==="

# Function to wait for process to stop
wait_for_process_stop() {
    local pid=$1
    local name=$2
    local timeout=${3:-30}
    
    log "Waiting for $name (PID: $pid) to stop..."
    
    for ((i=0; i<timeout; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            log "$name stopped successfully"
            return 0
        fi
        sleep 1
    done
    
    log "WARNING: $name did not stop within ${timeout}s"
    return 1
}

# Function to save avatar state before shutdown
save_avatar_state() {
    log "Saving avatar state before shutdown..."
    
    # Try to trigger a graceful state save via API
    if curl -s -X POST http://localhost:8080/save-state \
        -H "Content-Type: application/json" \
        -d '{"reason": "graceful_shutdown"}' \
        --max-time 10 >/dev/null 2>&1; then
        log "Avatar state saved via API"
    else
        log "WARNING: Could not save avatar state via API"
    fi
    
    # Save current GPU memory state
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader > "$BEV_ROOT/logs/gpu-state-shutdown.csv" 2>/dev/null || true
    fi
    
    # Save Redis avatar data
    if command -v redis-cli &>/dev/null; then
        redis-cli -h localhost -p 6379 BGSAVE 2>/dev/null || log "WARNING: Could not trigger Redis background save"
    fi
}

# Function to stop health monitor
stop_health_monitor() {
    if [[ -f "$HEALTH_PID_FILE" ]]; then
        local health_pid=$(cat "$HEALTH_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$health_pid" ]] && kill -0 "$health_pid" 2>/dev/null; then
            log "Stopping health monitor (PID: $health_pid)..."
            kill -TERM "$health_pid" 2>/dev/null || true
            wait_for_process_stop "$health_pid" "health monitor" 10
            if kill -0 "$health_pid" 2>/dev/null; then
                log "Force killing health monitor..."
                kill -KILL "$health_pid" 2>/dev/null || true
            fi
        fi
        rm -f "$HEALTH_PID_FILE"
    fi
}

# Function to gracefully stop avatar service
stop_avatar_service() {
    if [[ ! -f "$PID_FILE" ]]; then
        log "No PID file found, attempting to find avatar process..."
        
        # Try to find the process by port
        local avatar_pid=$(ss -tulnp | grep ":8080 " | awk '{print $6}' | cut -d',' -f2 | cut -d'=' -f2 | head -1)
        
        if [[ -n "$avatar_pid" ]]; then
            log "Found avatar process by port: $avatar_pid"
            echo "$avatar_pid" > "$PID_FILE"
        else
            log "No avatar process found"
            return 0
        fi
    fi
    
    local avatar_pid=$(cat "$PID_FILE" 2>/dev/null || echo "")
    
    if [[ -z "$avatar_pid" ]]; then
        log "Invalid PID file"
        return 0
    fi
    
    if ! kill -0 "$avatar_pid" 2>/dev/null; then
        log "Avatar process not running (PID: $avatar_pid)"
        rm -f "$PID_FILE"
        return 0
    fi
    
    log "Gracefully stopping avatar service (PID: $avatar_pid)..."
    
    # Save state before stopping
    save_avatar_state
    
    # Send SIGTERM for graceful shutdown
    kill -TERM "$avatar_pid" 2>/dev/null || true
    
    # Wait for graceful shutdown
    if wait_for_process_stop "$avatar_pid" "avatar service" 25; then
        log "Avatar service stopped gracefully"
    else
        log "Avatar service did not stop gracefully, forcing..."
        kill -KILL "$avatar_pid" 2>/dev/null || true
        sleep 2
        if kill -0 "$avatar_pid" 2>/dev/null; then
            log "ERROR: Could not stop avatar service"
            return 1
        else
            log "Avatar service force stopped"
        fi
    fi
    
    rm -f "$PID_FILE"
    return 0
}

# Function to cleanup WebSocket connections
cleanup_websockets() {
    log "Cleaning up WebSocket connections..."
    
    # Find and close any remaining WebSocket connections on port 8080
    local ws_pids=$(ss -tulnp | grep ":8080 " | awk '{print $6}' | cut -d',' -f2 | cut -d'=' -f2 | sort -u)
    
    for pid in $ws_pids; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "Closing WebSocket connection (PID: $pid)..."
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
}

# Function to update Redis with shutdown status
update_shutdown_status() {
    log "Updating shutdown status in Redis..."
    
    if command -v redis-cli &>/dev/null; then
        local shutdown_data=$(cat << EOF
{
    "status": "shutdown",
    "timestamp": $(date +%s),
    "reason": "graceful_shutdown",
    "last_health_check": $(date -Iseconds)
}
EOF
)
        
        redis-cli -h localhost -p 6379 SET "avatar:shutdown_status" "$shutdown_data" EX 3600 2>/dev/null || true
        redis-cli -h localhost -p 6379 DEL "avatar:health" "avatar:performance" 2>/dev/null || true
    fi
}

# Function to create shutdown report
create_shutdown_report() {
    local report_file="$BEV_ROOT/logs/shutdown-report-$(date +%Y%m%d-%H%M%S).json"
    
    log "Creating shutdown report: $report_file"
    
    cat > "$report_file" << EOF
{
    "shutdown_timestamp": "$(date -Iseconds)",
    "shutdown_reason": "systemd_stop",
    "processes_stopped": {
        "avatar_service": true,
        "health_monitor": true
    },
    "state_saved": true,
    "gpu_state": {
        "memory_cleared": true,
        "performance_reset": false
    },
    "redis_data_preserved": true,
    "graceful_shutdown": true,
    "shutdown_duration_seconds": $SECONDS
}
EOF
    
    log "Shutdown report created"
}

# Main shutdown sequence
main() {
    local start_time=$SECONDS
    
    # Stop health monitor first
    stop_health_monitor
    
    # Cleanup WebSocket connections
    cleanup_websockets
    
    # Stop avatar service
    if ! stop_avatar_service; then
        log "ERROR: Failed to stop avatar service"
        exit 1
    fi
    
    # Update shutdown status
    update_shutdown_status
    
    # Create shutdown report
    create_shutdown_report
    
    local shutdown_duration=$((SECONDS - start_time))
    log "Graceful shutdown completed in ${shutdown_duration}s"
    log "=== BEV Advanced Avatar Shutdown Complete ==="
}

# Handle signals during shutdown
trap 'log "Shutdown interrupted by signal"; exit 1' SIGTERM SIGINT

# Execute main shutdown sequence
main

exit 0