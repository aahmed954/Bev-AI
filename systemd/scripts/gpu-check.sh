#!/bin/bash
#
# BEV RTX 4090 GPU Check and Optimization Script
# Validates and optimizes RTX 4090 for avatar rendering
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEV_ROOT="/home/starlord/Projects/Bev"
LOG_FILE="$BEV_ROOT/logs/avatar-gpu.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU: $*" | tee -a "$LOG_FILE"
}

log "=== RTX 4090 GPU Validation and Optimization ==="

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    log "ERROR: NVIDIA drivers not installed or nvidia-smi not found"
    exit 1
fi

# Get GPU information
GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,power.draw --format=csv,noheader,nounits)
log "GPU Information: $GPU_INFO"

# Parse GPU details
GPU_NAME=$(echo "$GPU_INFO" | cut -d',' -f1 | xargs)
DRIVER_VERSION=$(echo "$GPU_INFO" | cut -d',' -f2 | xargs)
TOTAL_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f3 | xargs)
FREE_MEMORY=$(echo "$GPU_INFO" | cut -d',' -f4 | xargs)
TEMPERATURE=$(echo "$GPU_INFO" | cut -d',' -f5 | xargs)
POWER_DRAW=$(echo "$GPU_INFO" | cut -d',' -f6 | xargs)

# Validate RTX 4090
if [[ "$GPU_NAME" != *"RTX 4090"* ]]; then
    log "WARNING: Expected RTX 4090, found: $GPU_NAME"
    log "Continuing with available GPU..."
fi

log "Driver Version: $DRIVER_VERSION"
log "Total Memory: ${TOTAL_MEMORY}MB"
log "Free Memory: ${FREE_MEMORY}MB"
log "Temperature: ${TEMPERATURE}°C"
log "Power Draw: ${POWER_DRAW}W"

# Check minimum driver version for RTX 4090 (525.60.11+)
MIN_DRIVER_VERSION="525.60.11"
if ! printf '%s\n%s\n' "$MIN_DRIVER_VERSION" "$DRIVER_VERSION" | sort -V -C; then
    log "WARNING: Driver version $DRIVER_VERSION may not fully support RTX 4090. Minimum recommended: $MIN_DRIVER_VERSION"
fi

# Check memory availability (minimum 16GB for RTX 4090)
if [[ "$TOTAL_MEMORY" -lt 16000 ]]; then
    log "WARNING: GPU memory less than 16GB: ${TOTAL_MEMORY}MB"
fi

# Check available memory (minimum 12GB free for avatar)
if [[ "$FREE_MEMORY" -lt 12000 ]]; then
    log "ERROR: Insufficient free GPU memory for avatar. Available: ${FREE_MEMORY}MB, Required: 12000MB"
    exit 1
fi

# Check temperature (should be under 83°C)
if [[ "$TEMPERATURE" -gt 83 ]]; then
    log "WARNING: High GPU temperature: ${TEMPERATURE}°C. Consider improving cooling."
fi

# Validate CUDA installation
if [[ ! -d "/usr/local/cuda" ]]; then
    log "ERROR: CUDA installation not found at /usr/local/cuda"
    exit 1
fi

# Check CUDA version compatibility
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d',' -f1)
if [[ -z "$CUDA_VERSION" ]]; then
    log "ERROR: CUDA compiler (nvcc) not found"
    exit 1
fi

log "CUDA Version: $CUDA_VERSION"

# Validate CUDA version (12.x+ recommended for RTX 4090)
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
if [[ "$CUDA_MAJOR" -lt 12 ]]; then
    log "WARNING: CUDA version $CUDA_VERSION may not be optimal for RTX 4090. Recommended: 12.x+"
fi

# Check cuDNN availability
if ! python3 -c "import torch; print(torch.backends.cudnn.enabled)" 2>/dev/null | grep -q "True"; then
    log "ERROR: cuDNN not available or not enabled"
    exit 1
fi

# Validate PyTorch CUDA support
if ! python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    log "ERROR: PyTorch CUDA support not available"
    exit 1
fi

# Check PyTorch can see the GPU
GPU_COUNT=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [[ "$GPU_COUNT" -eq 0 ]]; then
    log "ERROR: PyTorch cannot detect any CUDA devices"
    exit 1
fi

log "PyTorch detects $GPU_COUNT CUDA device(s)"

# Test GPU memory allocation
log "Testing GPU memory allocation..."
if ! python3 -c "
import torch
torch.cuda.set_device(0)
x = torch.randn(1000, 1000, device='cuda:0')
y = torch.randn(1000, 1000, device='cuda:0')
z = torch.mm(x, y)
print(f'GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
torch.cuda.empty_cache()
print('GPU memory test successful')
" 2>>"$LOG_FILE"; then
    log "ERROR: GPU memory allocation test failed"
    exit 1
fi

# Set GPU performance mode
if command -v nvidia-smi &> /dev/null; then
    # Set persistence mode
    sudo nvidia-smi -pm 1 2>/dev/null || log "WARNING: Could not enable persistence mode"
    
    # Set maximum performance mode
    sudo nvidia-smi -ac 10501,2230 2>/dev/null || log "WARNING: Could not set application clocks"
    
    # Set power limit to maximum (450W for RTX 4090)
    sudo nvidia-smi -pl 450 2>/dev/null || log "WARNING: Could not set power limit"
    
    log "GPU performance optimization applied"
fi

# Set optimal GPU governor
if [[ -f "/sys/class/drm/card0/device/power_dpm_force_performance_level" ]]; then
    echo "high" | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level >/dev/null 2>&1 || true
fi

# Verify nvidia-persistenced is running
if ! systemctl is-active --quiet nvidia-persistenced; then
    log "WARNING: nvidia-persistenced service is not running"
    sudo systemctl start nvidia-persistenced 2>/dev/null || log "WARNING: Could not start nvidia-persistenced"
fi

# Check GPU utilization (should be low before starting)
CURRENT_UTILIZATION=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
if [[ "$CURRENT_UTILIZATION" -gt 50 ]]; then
    log "WARNING: GPU utilization already high: ${CURRENT_UTILIZATION}%"
fi

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# Validate GPU compute capability for Gaussian Splatting
COMPUTE_CAPABILITY=$(python3 -c "import torch; print(torch.cuda.get_device_capability(0))" 2>/dev/null || echo "(0, 0)")
log "GPU Compute Capability: $COMPUTE_CAPABILITY"

# RTX 4090 should have compute capability 8.9
if [[ "$COMPUTE_CAPABILITY" != "(8, 9)" ]]; then
    log "WARNING: Unexpected compute capability for RTX 4090: $COMPUTE_CAPABILITY"
fi

# Final GPU status
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    while IFS=',' read -r name util mem_used mem_total temp; do
        log "Final Status - Name: $name, Utilization: $util, Memory: $mem_used/$mem_total, Temp: $temp"
    done

log "RTX 4090 GPU validation and optimization completed successfully"
log "=== GPU Check Complete ==="

exit 0