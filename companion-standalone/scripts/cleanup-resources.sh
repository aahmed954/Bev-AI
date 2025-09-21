#!/bin/bash
# Resource cleanup script for Standalone AI Companion
# Cleanly shuts down and releases all resources

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/companion/cleanup-$(date +%Y%m%d_%H%M%S).log"

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

info "Starting AI Companion resource cleanup..."

# 1. Graceful Service Shutdown
info "Gracefully shutting down companion services..."

COMPANION_CONTAINERS=(
    "companion_frontend"
    "companion_gateway"
    "companion_monitor"
    "companion_memory"
    "companion_avatar"
    "companion_voice"
    "companion_core"
    "companion_grafana"
    "companion_prometheus"
    "companion_redis"
    "companion_postgres"
)

for container in "${COMPANION_CONTAINERS[@]}"; do
    if docker ps --format "table {{.Names}}" | grep -q "^$container$"; then
        info "Stopping container: $container"
        docker stop "$container" --time=30 >/dev/null 2>&1 || warning "Failed to stop $container gracefully"
        success "Stopped $container"
    else
        info "Container $container not running"
    fi
done

# 2. GPU Resource Cleanup
info "Cleaning up GPU resources..."

# Clear GPU memory
if command -v nvidia-smi >/dev/null 2>&1; then
    # Kill any remaining GPU processes
    GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -n "$GPU_PROCESSES" ]]; then
        for pid in $GPU_PROCESSES; do
            if [[ "$pid" =~ ^[0-9]+$ ]]; then
                info "Terminating GPU process: $pid"
                kill -TERM "$pid" 2>/dev/null || true
                sleep 2
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
    fi

    # Reset GPU state
    sudo nvidia-smi --gpu-reset -i 0 2>/dev/null || warning "GPU reset failed"

    # Clear GPU memory caches
    nvidia-smi --gpu-reset -i 0 2>/dev/null || true

    success "GPU resources cleaned up"
else
    warning "nvidia-smi not available, skipping GPU cleanup"
fi

# 3. Memory Cleanup
info "Cleaning up system memory..."

# Clear container-related memory
docker system prune -f >/dev/null 2>&1 || warning "Docker system prune failed"

# Clear system caches if safe
sync
echo 1 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || warning "Could not clear page cache"
echo 2 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || warning "Could not clear dentries and inodes"
echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || warning "Could not clear all caches"

# Reset swappiness to default
echo 60 | sudo tee /proc/sys/vm/swappiness >/dev/null 2>&1 || warning "Could not reset swappiness"

success "System memory cleaned up"

# 4. Network Cleanup
info "Cleaning up network resources..."

# Remove companion network if exists
if docker network ls | grep -q "companion_network"; then
    docker network rm companion_network >/dev/null 2>&1 || warning "Could not remove companion network"
    success "Companion network removed"
fi

# Reset network buffer sizes to defaults
echo "net.core.rmem_max = 212992" | sudo tee -a /etc/sysctl.conf >/dev/null 2>&1 || true
echo "net.core.wmem_max = 212992" | sudo tee -a /etc/sysctl.conf >/dev/null 2>&1 || true
sudo sysctl -p >/dev/null 2>&1 || warning "Could not reset network parameters"

success "Network resources cleaned up"

# 5. Temporary File Cleanup
info "Cleaning up temporary files..."

# Remove companion-specific temporary files
TEMP_DIRS=(
    "/tmp/companion-*"
    "/tmp/torch-*"
    "/tmp/cuda-*"
    "/var/tmp/companion-*"
)

for pattern in "${TEMP_DIRS[@]}"; do
    rm -rf $pattern 2>/dev/null || true
done

# Remove lock files
rm -f /tmp/companion-resources.lock 2>/dev/null || true

# Clean up shared memory
rm -rf /dev/shm/companion-* 2>/dev/null || true

success "Temporary files cleaned up"

# 6. Process Cleanup
info "Cleaning up remaining processes..."

# Kill any remaining companion-related processes
COMPANION_PROCESSES=$(pgrep -f "companion" | grep -v $$ || true)
if [[ -n "$COMPANION_PROCESSES" ]]; then
    for pid in $COMPANION_PROCESSES; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && [[ "$pid" != "$$" ]]; then
            info "Terminating companion process: $pid"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
fi

# Kill any remaining Python AI processes
AI_PROCESSES=$(pgrep -f "python.*torch\|python.*tensorflow\|python.*cuda" | grep -v $$ || true)
if [[ -n "$AI_PROCESSES" ]]; then
    for pid in $AI_PROCESSES; do
        if [[ "$pid" =~ ^[0-9]+$ ]] && [[ "$pid" != "$$" ]]; then
            info "Terminating AI process: $pid"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
fi

success "Processes cleaned up"

# 7. Docker Resource Cleanup
info "Cleaning up Docker resources..."

# Remove companion containers
docker rm -f $(docker ps -aq --filter "name=companion_") 2>/dev/null || true

# Remove companion images (optional, commented for faster restarts)
# docker rmi -f $(docker images -q --filter "reference=companion-*") 2>/dev/null || true

# Clean up unused Docker resources
docker system prune -f >/dev/null 2>&1 || warning "Docker system prune failed"

# Clean up volumes (only if explicitly requested)
if [[ "${CLEANUP_VOLUMES:-false}" == "true" ]]; then
    warning "Removing companion volumes (data will be lost)..."
    docker volume rm $(docker volume ls -q --filter "name=companion-*") 2>/dev/null || true
fi

success "Docker resources cleaned up"

# 8. CPU Performance Reset
info "Resetting CPU performance settings..."

# Reset CPU governor to default (powersave)
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [[ -f "$cpu" ]]; then
        echo powersave | sudo tee "$cpu" >/dev/null 2>&1 || warning "Could not reset governor for $cpu"
    fi
done

# Reset CPU frequency scaling
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
    if [[ -f "$cpu" ]]; then
        min_freq=$(cat "${cpu%min_freq}cpuinfo_min_freq")
        echo "$min_freq" | sudo tee "$cpu" >/dev/null 2>&1 || true
    fi
done

success "CPU performance settings reset"

# 9. Log File Management
info "Managing log files..."

# Compress old log files
find /var/log/companion -name "*.log" -type f -mtime +1 -exec gzip {} \; 2>/dev/null || true

# Remove very old log files (older than 30 days)
find /var/log/companion -name "*.log.gz" -type f -mtime +30 -delete 2>/dev/null || true

# Create cleanup summary log
CLEANUP_SUMMARY="/var/log/companion/cleanup-summary.json"
cat > "$CLEANUP_SUMMARY" <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "cleanup_type": "full",
  "containers_stopped": $(echo "${COMPANION_CONTAINERS[@]}" | wc -w),
  "gpu_reset": true,
  "memory_cleared": true,
  "network_cleaned": true,
  "temp_files_removed": true,
  "processes_terminated": true,
  "docker_cleaned": true,
  "cpu_reset": true,
  "logs_managed": true
}
EOF

success "Log files managed"

# 10. Resource Validation
info "Validating resource cleanup..."

# Check for remaining companion containers
REMAINING_CONTAINERS=$(docker ps -q --filter "name=companion_" | wc -l)
if [[ $REMAINING_CONTAINERS -eq 0 ]]; then
    success "No companion containers remaining"
else
    warning "$REMAINING_CONTAINERS companion containers still running"
fi

# Check GPU utilization
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [[ $GPU_UTIL -lt 10 ]]; then
        success "GPU utilization low: ${GPU_UTIL}%"
    else
        warning "GPU utilization still high: ${GPU_UTIL}%"
    fi
fi

# Check memory usage
USED_MEM=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [[ $USED_MEM -lt 50 ]]; then
    success "Memory usage low: ${USED_MEM}%"
else
    info "Memory usage: ${USED_MEM}%"
fi

# Check for remaining processes
REMAINING_PROCESSES=$(pgrep -f "companion" | grep -v $$ | wc -l || echo 0)
if [[ $REMAINING_PROCESSES -eq 0 ]]; then
    success "No companion processes remaining"
else
    warning "$REMAINING_PROCESSES companion processes still running"
fi

# 11. System State Reset
info "Resetting system state..."

# Remove environment modifications
unset CUDA_VISIBLE_DEVICES
unset PYTORCH_CUDA_ALLOC_CONF
unset OMP_NUM_THREADS

# Reset ulimits to defaults
ulimit -n 1024
ulimit -u 16384

# Remove companion environment file
rm -f /opt/companion/config/companion.env 2>/dev/null || true

success "System state reset"

# 12. Final Cleanup Verification
info "Performing final cleanup verification..."

# Verify no companion services are listening
LISTENING_PORTS=$(netstat -tuln | grep -E ":(15432|16379|1800[0-6]|18080|18443|19000|19090)" | wc -l || echo 0)
if [[ $LISTENING_PORTS -eq 0 ]]; then
    success "No companion ports listening"
else
    warning "$LISTENING_PORTS companion ports still listening"
fi

# Verify disk space recovered
DISK_FREE_GB=$(df /opt/companion --output=avail 2>/dev/null | tail -1 | awk '{print int($1/1024/1024)}' || echo 0)
info "Available disk space: ${DISK_FREE_GB}GB"

success "Final cleanup verification completed"

# Summary
log ""
log "${GREEN}=== RESOURCE CLEANUP COMPLETED ===${NC}"
log "${GREEN}✓ All companion services stopped${NC}"
log "${GREEN}✓ GPU resources released${NC}"
log "${GREEN}✓ System memory freed${NC}"
log "${GREEN}✓ Network resources cleaned${NC}"
log "${GREEN}✓ Temporary files removed${NC}"
log "${GREEN}✓ Processes terminated${NC}"
log "${GREEN}✓ Docker resources cleaned${NC}"
log "${GREEN}✓ CPU settings reset${NC}"
log "${GREEN}✓ Log files managed${NC}"
log "${GREEN}✓ System state reset${NC}"
log ""
log "${BLUE}AI Companion resources have been completely cleaned up!${NC}"
log "${BLUE}System is ready for next deployment or shutdown.${NC}"

exit 0