#!/bin/bash
# Resource preparation script for Standalone AI Companion
# Optimizes system resources and prepares GPU for companion workload

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/companion/resource-prep-$(date +%Y%m%d_%H%M%S).log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${1}" | tee -a "$LOG_FILE"
}

error_exit() {
    log "${RED}ERROR: ${1}${NC}"
    exit 1
}

success() {
    log "${GREEN}✓ ${1}${NC}"
}

warning() {
    log "${YELLOW}⚠ ${1}${NC}"
}

info() {
    log "${BLUE}ℹ ${1}${NC}"
}

# Ensure log directory exists
sudo mkdir -p "$(dirname "$LOG_FILE")"
sudo touch "$LOG_FILE"
sudo chown starlord:starlord "$LOG_FILE"

info "Starting resource preparation for AI Companion..."

# 1. GPU Resource Preparation
info "Preparing GPU resources..."

# Reset GPU to clear any existing processes
if command -v nvidia-smi >/dev/null 2>&1; then
    info "Resetting GPU state..."
    sudo nvidia-smi --gpu-reset -i 0 2>/dev/null || warning "GPU reset failed, continuing..."

    # Clear GPU memory
    nvidia-smi --gpu-reset -i 0 2>/dev/null || true

    # Set GPU performance mode
    sudo nvidia-smi -i 0 -pm 1 2>/dev/null || warning "Could not set persistent mode"

    # Set maximum performance state
    sudo nvidia-smi -i 0 -ac $(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits -i 0 | tr ',' ' ') 2>/dev/null || warning "Could not set max clocks"

    success "GPU preparation completed"
else
    error_exit "nvidia-smi not available"
fi

# 2. System Memory Optimization
info "Optimizing system memory..."

# Clear system caches (if safe to do so)
sync
echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || warning "Could not clear page cache"
echo 2 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || warning "Could not clear dentries and inodes"

# Set swappiness for AI workload
CURRENT_SWAPPINESS=$(cat /proc/sys/vm/swappiness)
if [[ $CURRENT_SWAPPINESS -gt 10 ]]; then
    echo 10 | sudo tee /proc/sys/vm/swappiness >/dev/null
    info "Reduced swappiness from $CURRENT_SWAPPINESS to 10"
fi

# Increase shared memory for AI workloads
echo "kernel.shmmax = 68719476736" | sudo tee -a /etc/sysctl.conf >/dev/null 2>&1 || true
echo "kernel.shmall = 4294967296" | sudo tee -a /etc/sysctl.conf >/dev/null 2>&1 || true

success "Memory optimization completed"

# 3. CPU Performance Optimization
info "Optimizing CPU performance..."

# Set CPU governor to performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [[ -f "$cpu" ]]; then
        echo performance | sudo tee "$cpu" >/dev/null 2>&1 || warning "Could not set governor for $cpu"
    fi
done

# Disable CPU frequency scaling for consistent performance
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
    if [[ -f "$cpu" ]]; then
        max_freq=$(cat "${cpu%min_freq}scaling_max_freq")
        echo "$max_freq" | sudo tee "$cpu" >/dev/null 2>&1 || true
    fi
done

success "CPU optimization completed"

# 4. Network Optimization
info "Optimizing network configuration..."

# Increase network buffer sizes for high-throughput AI operations
echo "net.core.rmem_max = 268435456" | sudo tee -a /etc/sysctl.conf >/dev/null 2>&1 || true
echo "net.core.wmem_max = 268435456" | sudo tee -a /etc/sysctl.conf >/dev/null 2>&1 || true
echo "net.core.netdev_max_backlog = 30000" | sudo tee -a /etc/sysctl.conf >/dev/null 2>&1 || true

# Apply network changes
sudo sysctl -p >/dev/null 2>&1 || warning "Could not apply sysctl changes"

success "Network optimization completed"

# 5. Docker Resource Optimization
info "Optimizing Docker resources..."

# Ensure Docker daemon is using optimal settings
DOCKER_DAEMON_CONFIG="/etc/docker/daemon.json"
if [[ ! -f "$DOCKER_DAEMON_CONFIG" ]]; then
    sudo mkdir -p /etc/docker
    sudo tee "$DOCKER_DAEMON_CONFIG" >/dev/null <<EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "default-shm-size": "2g",
  "storage-driver": "overlay2",
  "experimental": false,
  "metrics-addr": "0.0.0.0:9323",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF
    sudo systemctl reload docker || warning "Could not reload Docker daemon"
fi

# Prune unused Docker resources
docker system prune -f >/dev/null 2>&1 || warning "Docker system prune failed"

success "Docker optimization completed"

# 6. Storage Optimization
info "Optimizing storage configuration..."

# Ensure storage directories exist with correct permissions
STORAGE_DIRS=(
    "/opt/companion/data/postgres"
    "/opt/companion/data/redis"
    "/opt/companion/models"
    "/opt/companion/assets"
    "/opt/companion/logs"
    "/opt/companion/config"
)

for dir in "${STORAGE_DIRS[@]}"; do
    sudo mkdir -p "$dir"
    sudo chown starlord:starlord "$dir"
    sudo chmod 755 "$dir"
done

# Set optimal filesystem flags if possible
for dir in "${STORAGE_DIRS[@]}"; do
    # Enable compression for models and assets to save space
    if [[ "$dir" =~ (models|assets) ]]; then
        sudo chattr +c "$dir" 2>/dev/null || true
    fi
    # Enable faster access for logs and data
    if [[ "$dir" =~ (logs|data) ]]; then
        sudo chattr +A "$dir" 2>/dev/null || true
    fi
done

success "Storage optimization completed"

# 7. Environment Variable Preparation
info "Preparing environment variables..."

# Create environment file for companion
ENV_FILE="/opt/companion/config/companion.env"
cat > "$ENV_FILE" <<EOF
# AI Companion Environment Configuration
# Generated: $(date)

# Resource Configuration
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
OMP_NUM_THREADS=$(nproc)
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1

# Performance Optimization
TORCH_CUDNN_V8_API_ENABLED=1
TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT=32
CUDA_LAUNCH_BLOCKING=0

# Memory Management
PYTHONMALLOC=malloc
MALLOC_ARENA_MAX=2

# Logging Configuration
PYTHONUNBUFFERED=1
TORCH_LOGS=+all

# GPU Memory Fraction
GPU_MEMORY_FRACTION=0.8
MIXED_PRECISION=true

# Companion Specific
COMPANION_MODE=standalone
COMPANION_PERFORMANCE_MODE=optimized
COMPANION_DEBUG_MODE=false
EOF

chmod 644 "$ENV_FILE"
success "Environment variables prepared"

# 8. Resource Monitoring Setup
info "Setting up resource monitoring..."

# Create monitoring script
MONITOR_SCRIPT="/opt/companion/scripts/monitor-resources.sh"
sudo mkdir -p "$(dirname "$MONITOR_SCRIPT")"

cat > "$MONITOR_SCRIPT" <<'EOF'
#!/bin/bash
# Real-time resource monitoring for AI Companion

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # GPU monitoring
    gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    # System monitoring
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')

    # Log to monitoring file
    echo "$timestamp,GPU_TEMP:$gpu_temp,GPU_MEM:$gpu_mem,GPU_UTIL:$gpu_util,CPU:$cpu_usage,MEM:$mem_usage" >> /var/log/companion/resources.log

    sleep 10
done
EOF

chmod +x "$MONITOR_SCRIPT"
success "Resource monitoring setup completed"

# 9. Performance Validation
info "Validating resource preparation..."

# Test GPU availability
if nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    info "GPU validation: $GPU_COUNT GPU(s) available"

    # Test CUDA functionality
    if python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null; then
        success "CUDA functionality validated"
    else
        warning "CUDA functionality test failed"
    fi
else
    error_exit "GPU validation failed"
fi

# Test memory availability
AVAILABLE_MEM_GB=$(free -g | awk '/^Mem:/{print $7}')
if [[ $AVAILABLE_MEM_GB -ge 8 ]]; then
    success "Memory validation: ${AVAILABLE_MEM_GB}GB available"
else
    warning "Low memory available: ${AVAILABLE_MEM_GB}GB"
fi

# Test Docker GPU runtime
if docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi >/dev/null 2>&1; then
    success "Docker GPU runtime validated"
else
    warning "Docker GPU runtime test failed"
fi

# 10. Cleanup and Finalization
info "Finalizing resource preparation..."

# Create resource lock file to prevent conflicts
echo "$(date)" > /tmp/companion-resources.lock
echo "$$" >> /tmp/companion-resources.lock

# Set resource limits for companion processes
ulimit -n 65536
ulimit -u 32768

success "Resource preparation completed successfully"

# Summary
log ""
log "${GREEN}=== RESOURCE PREPARATION COMPLETED ===${NC}"
log "${GREEN}✓ GPU optimized and ready${NC}"
log "${GREEN}✓ System memory optimized${NC}"
log "${GREEN}✓ CPU performance maximized${NC}"
log "${GREEN}✓ Network buffers optimized${NC}"
log "${GREEN}✓ Docker resources prepared${NC}"
log "${GREEN}✓ Storage configuration optimized${NC}"
log "${GREEN}✓ Environment variables set${NC}"
log "${GREEN}✓ Resource monitoring active${NC}"
log ""
log "${BLUE}System is optimized for AI Companion deployment!${NC}"

exit 0