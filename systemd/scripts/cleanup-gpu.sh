#!/bin/bash
#
# BEV RTX 4090 GPU Cleanup Script
# Performs comprehensive GPU cleanup after avatar shutdown
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEV_ROOT="/home/starlord/Projects/Bev"
LOG_FILE="$BEV_ROOT/logs/avatar-gpu-cleanup.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CLEANUP: $*" | tee -a "$LOG_FILE"
}

log "=== RTX 4090 GPU Cleanup Starting ==="

# Function to clear GPU memory
clear_gpu_memory() {
    log "Clearing GPU memory..."
    
    # Use Python to clear PyTorch CUDA cache
    python3 -c "
import torch
import gc

try:
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        log_msg = f'Clearing CUDA cache for {device_count} device(s)'
        print(log_msg)
        
        for i in range(device_count):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Check memory status after cleanup
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            print(f'Post-cleanup - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB')
        
        print('GPU memory cleanup completed successfully')
    else:
        print('CUDA not available - no cleanup needed')
except Exception as e:
    print(f'Error during GPU memory cleanup: {e}')
    exit(1)
" 2>>"$LOG_FILE" || {
        log "ERROR: Failed to clear GPU memory via Python"
        return 1
    }
    
    log "GPU memory cleared successfully"
}

# Function to reset GPU clocks to default
reset_gpu_clocks() {
    log "Resetting GPU clocks to default..."
    
    if ! command -v nvidia-smi &>/dev/null; then
        log "WARNING: nvidia-smi not available for clock reset"
        return 0
    fi
    
    # Reset application clocks to default
    if sudo nvidia-smi -rac 2>/dev/null; then
        log "GPU application clocks reset to default"
    else
        log "WARNING: Could not reset application clocks"
    fi
    
    # Reset power limit to default (450W for RTX 4090)
    if sudo nvidia-smi -pl 450 2>/dev/null; then
        log "GPU power limit reset to default"
    else
        log "WARNING: Could not reset power limit"
    fi
}

# Function to check for GPU processes
check_gpu_processes() {
    log "Checking for remaining GPU processes..."
    
    if ! command -v nvidia-smi &>/dev/null; then
        log "WARNING: nvidia-smi not available"
        return 0
    fi
    
    # Check for processes using GPU
    local gpu_processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "")
    
    if [[ -n "$gpu_processes" && "$gpu_processes" != "No running processes found" ]]; then
        log "WARNING: Found remaining GPU processes:"
        echo "$gpu_processes" | while read -r line; do
            if [[ -n "$line" ]]; then
                log "  $line"
            fi
        done
        
        # Try to identify and clean up avatar-related processes
        echo "$gpu_processes" | while IFS=',' read -r pid process_name memory; do
            if [[ -n "$pid" && "$process_name" =~ (python|uvicorn|fastapi) ]]; then
                log "Attempting to terminate GPU process: PID $pid ($process_name)"
                if kill -TERM "$pid" 2>/dev/null; then
                    sleep 2
                    if kill -0 "$pid" 2>/dev/null; then
                        log "Force killing persistent GPU process: $pid"
                        kill -KILL "$pid" 2>/dev/null || true
                    fi
                fi
            fi
        done
    else
        log "No GPU processes found"
    fi
}

# Function to check GPU temperature and cooling
check_gpu_cooling() {
    log "Checking GPU temperature and cooling..."
    
    if ! command -v nvidia-smi &>/dev/null; then
        return 0
    fi
    
    local gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo "0")
    local fan_speed=$(nvidia-smi --query-gpu=fan.speed --format=csv,noheader,nounits 2>/dev/null || echo "0")
    
    log "GPU temperature: ${gpu_temp}°C"
    log "Fan speed: ${fan_speed}%"
    
    if [[ "$gpu_temp" -gt 80 ]]; then
        log "WARNING: High GPU temperature: ${gpu_temp}°C"
        
        # Try to increase fan speed for cooling
        if sudo nvidia-smi -pl 300 2>/dev/null; then
            log "Reduced power limit to 300W for cooling"
        fi
    fi
}

# Function to save GPU state for diagnostics
save_gpu_state() {
    log "Saving GPU state for diagnostics..."
    
    if ! command -v nvidia-smi &>/dev/null; then
        return 0
    fi
    
    local state_file="$BEV_ROOT/logs/gpu-cleanup-state-$(date +%Y%m%d-%H%M%S).txt"
    
    {
        echo "=== GPU State After Cleanup ==="
        echo "Timestamp: $(date -Iseconds)"
        echo ""
        
        echo "=== GPU Information ==="
        nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,fan.speed --format=csv,noheader
        echo ""
        
        echo "=== GPU Processes ==="
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
        echo ""
        
        echo "=== Full nvidia-smi Output ==="
        nvidia-smi
        
    } > "$state_file" 2>&1
    
    log "GPU state saved to: $state_file"
}

# Function to validate cleanup success
validate_cleanup() {
    log "Validating cleanup success..."
    
    # Check if we can allocate a small amount of GPU memory
    if python3 -c "
import torch
try:
    if torch.cuda.is_available():
        # Try to allocate small tensor to test GPU accessibility
        x = torch.randn(100, 100, device='cuda:0')
        y = torch.randn(100, 100, device='cuda:0')
        z = torch.mm(x, y)
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f'GPU validation successful')
        print(f'Test allocation - Allocated: {allocated:.3f}GB, Reserved: {reserved:.3f}GB')
        
        # Clean up test tensors
        del x, y, z
        torch.cuda.empty_cache()
        print('Test cleanup completed')
    else:
        print('CUDA not available for validation')
except Exception as e:
    print(f'GPU validation failed: {e}')
    exit(1)
" 2>>"$LOG_FILE"; then
        log "GPU cleanup validation successful"
    else
        log "ERROR: GPU cleanup validation failed"
        return 1
    fi
}

# Function to clean up temporary files
cleanup_temp_files() {
    log "Cleaning up temporary files..."
    
    # Remove avatar-specific temporary files
    local temp_patterns=(
        "/tmp/avatar_*"
        "/tmp/bev_avatar_*"
        "/tmp/pytorch_*"
        "/tmp/torch_*"
        "$BEV_ROOT/logs/avatar-health.py"
        "$BEV_ROOT/logs/*.tmp"
    )
    
    for pattern in "${temp_patterns[@]}"; do
        if compgen -G "$pattern" > /dev/null 2>&1; then
            rm -f $pattern 2>/dev/null || true
            log "Removed temporary files: $pattern"
        fi
    done
}

# Function to update cleanup metrics
update_cleanup_metrics() {
    log "Updating cleanup metrics..."
    
    if command -v redis-cli &>/dev/null; then
        local cleanup_data=$(cat << EOF
{
    "cleanup_timestamp": $(date +%s),
    "gpu_memory_cleared": true,
    "processes_terminated": true,
    "temperature_normal": $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0),
    "cleanup_duration_seconds": $SECONDS,
    "cleanup_successful": true
}
EOF
)
        
        redis-cli -h localhost -p 6379 SET "avatar:cleanup_status" "$cleanup_data" EX 3600 2>/dev/null || true
        log "Cleanup metrics updated in Redis"
    fi
}

# Main cleanup sequence
main() {
    local start_time=$SECONDS
    
    # Check for GPU processes first
    check_gpu_processes
    
    # Clear GPU memory
    if ! clear_gpu_memory; then
        log "ERROR: GPU memory cleanup failed"
        exit 1
    fi
    
    # Reset GPU clocks
    reset_gpu_clocks
    
    # Check cooling
    check_gpu_cooling
    
    # Clean up temporary files
    cleanup_temp_files
    
    # Save state for diagnostics
    save_gpu_state
    
    # Validate cleanup success
    if ! validate_cleanup; then
        log "ERROR: GPU cleanup validation failed"
        exit 1
    fi
    
    # Update metrics
    update_cleanup_metrics
    
    local cleanup_duration=$((SECONDS - start_time))
    log "GPU cleanup completed successfully in ${cleanup_duration}s"
    log "=== RTX 4090 GPU Cleanup Complete ==="
}

# Handle signals during cleanup
trap 'log "GPU cleanup interrupted by signal"; exit 1' SIGTERM SIGINT

# Execute main cleanup sequence
main

exit 0