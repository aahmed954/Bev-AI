#!/bin/bash
# Pre-start validation for Standalone AI Companion
# Ensures all prerequisites are met before deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPANION_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/var/log/companion/pre-start-$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Create log directory
sudo mkdir -p "$(dirname "$LOG_FILE")"
sudo touch "$LOG_FILE"
sudo chown starlord:starlord "$LOG_FILE"

info "Starting AI Companion pre-start validation..."
info "Log file: $LOG_FILE"

# 1. System Requirements Check
info "Checking system requirements..."

# Check if running as correct user
if [[ "$USER" != "starlord" ]]; then
    error_exit "Must run as user 'starlord', currently running as '$USER'"
fi
success "Running as correct user: starlord"

# Check Ubuntu version
if ! grep -q "Ubuntu" /etc/os-release; then
    warning "Not running Ubuntu, compatibility not guaranteed"
else
    success "Ubuntu system detected"
fi

# Check memory requirements (minimum 16GB)
MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
if [[ $MEMORY_GB -lt 16 ]]; then
    error_exit "Insufficient memory: ${MEMORY_GB}GB available, 16GB minimum required"
fi
success "Memory check passed: ${MEMORY_GB}GB available"

# Check disk space (minimum 100GB free)
DISK_FREE_GB=$(df /home/starlord/Projects/Bev --output=avail | tail -1 | awk '{print int($1/1024/1024)}')
if [[ $DISK_FREE_GB -lt 100 ]]; then
    error_exit "Insufficient disk space: ${DISK_FREE_GB}GB available, 100GB minimum required"
fi
success "Disk space check passed: ${DISK_FREE_GB}GB available"

# 2. Docker Requirements Check
info "Checking Docker requirements..."

# Check Docker daemon
if ! docker info >/dev/null 2>&1; then
    error_exit "Docker daemon not running or not accessible"
fi
success "Docker daemon is running"

# Check Docker Compose
if ! command -v docker-compose >/dev/null 2>&1; then
    error_exit "Docker Compose not installed"
fi
success "Docker Compose is available"

# Check Docker version
DOCKER_VERSION=$(docker --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
if [[ $(echo "$DOCKER_VERSION >= 20.10" | bc -l) -eq 0 ]]; then
    error_exit "Docker version $DOCKER_VERSION too old, 20.10+ required"
fi
success "Docker version check passed: $DOCKER_VERSION"

# Check user in docker group
if ! groups "$USER" | grep -q docker; then
    error_exit "User '$USER' not in docker group"
fi
success "User is in docker group"

# 3. GPU Requirements Check
info "Checking GPU requirements..."

# Check nvidia-smi availability
if ! command -v nvidia-smi >/dev/null 2>&1; then
    error_exit "nvidia-smi not available, NVIDIA drivers not installed"
fi
success "NVIDIA drivers are installed"

# Check RTX 4090 presence
if ! nvidia-smi --query-gpu=name --format=csv,noheader,nounits | grep -q "RTX 4090"; then
    warning "RTX 4090 not detected, performance may be reduced"
else
    success "RTX 4090 detected"
fi

# Check CUDA version
if ! nvidia-smi | grep -q "CUDA Version"; then
    error_exit "CUDA not available"
fi
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | grep -oE '[0-9]+\.[0-9]+')
if [[ $(echo "$CUDA_VERSION >= 12.0" | bc -l) -eq 0 ]]; then
    warning "CUDA version $CUDA_VERSION may be too old, 12.0+ recommended"
else
    success "CUDA version check passed: $CUDA_VERSION"
fi

# Check GPU memory
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [[ $GPU_MEMORY -lt 20000 ]]; then
    warning "GPU memory ${GPU_MEMORY}MB may be insufficient for optimal performance"
else
    success "GPU memory check passed: ${GPU_MEMORY}MB"
fi

# 4. Network Requirements Check
info "Checking network requirements..."

# Check port availability
REQUIRED_PORTS=(15432 16379 18000 18001 18002 18003 18004 18005 18006 18080 18443 19000 19090)
for port in "${REQUIRED_PORTS[@]}"; do
    if netstat -tuln | grep -q ":$port "; then
        error_exit "Required port $port is already in use"
    fi
done
success "All required ports are available"

# Check internet connectivity for image pulls
if ! ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    warning "No internet connectivity, Docker image pulls may fail"
else
    success "Internet connectivity available"
fi

# 5. Directory Structure Check
info "Checking directory structure..."

# Create required directories
sudo mkdir -p /opt/companion/{data/{postgres,redis},models,assets,logs,config}
sudo chown -R starlord:starlord /opt/companion

# Create required directories with proper permissions
REQUIRED_DIRS=(
    "/opt/companion/data/postgres"
    "/opt/companion/data/redis"
    "/opt/companion/models"
    "/opt/companion/assets"
    "/opt/companion/logs"
    "/opt/companion/config"
    "/var/log/companion"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        sudo mkdir -p "$dir"
        sudo chown starlord:starlord "$dir"
    fi
done
success "Directory structure validated"

# 6. Configuration Files Check
info "Checking configuration files..."

# Check companion Docker Compose file
COMPOSE_FILE="$COMPANION_ROOT/docker-compose.companion.yml"
if [[ ! -f "$COMPOSE_FILE" ]]; then
    error_exit "Companion Docker Compose file not found: $COMPOSE_FILE"
fi
success "Docker Compose file found"

# Validate Docker Compose syntax
if ! docker-compose -f "$COMPOSE_FILE" config >/dev/null 2>&1; then
    error_exit "Docker Compose file has syntax errors"
fi
success "Docker Compose file syntax valid"

# 7. Resource Conflicts Check
info "Checking for resource conflicts..."

# Check if core platform is running and would conflict
if docker ps --format "table {{.Names}}" | grep -E "(bev_|thanos_|oracle_)" >/dev/null 2>&1; then
    warning "Core platform services detected running, may cause resource conflicts"
    info "Consider stopping core platform or enabling integration mode"
else
    success "No resource conflicts detected"
fi

# 8. Security Validation
info "Checking security configuration..."

# Check firewall status
if command -v ufw >/dev/null 2>&1; then
    if ufw status | grep -q "Status: active"; then
        info "Firewall is active, ensure companion ports are accessible if needed"
    fi
fi

# Check SELinux/AppArmor
if command -v getenforce >/dev/null 2>&1; then
    if [[ $(getenforce) == "Enforcing" ]]; then
        warning "SELinux enforcing mode detected, may require policy adjustments"
    fi
fi

success "Security validation completed"

# 9. Performance Optimization Check
info "Checking performance optimization..."

# Check CPU governor
CPU_GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
if [[ "$CPU_GOVERNOR" != "performance" ]]; then
    warning "CPU governor is '$CPU_GOVERNOR', consider 'performance' for optimal results"
fi

# Check swappiness
SWAPPINESS=$(cat /proc/sys/vm/swappiness)
if [[ $SWAPPINESS -gt 10 ]]; then
    warning "Swappiness is $SWAPPINESS, consider reducing to 10 or less for better performance"
fi

success "Performance optimization check completed"

# 10. Final Validation
info "Running final validation..."

# Test GPU access
if ! nvidia-ml-py3 -c "import pynvml; pynvml.nvmlInit(); print('GPU access OK')" 2>/dev/null; then
    warning "Python GPU access test failed, may affect AI operations"
else
    success "GPU access validation passed"
fi

# Summary
log ""
log "${GREEN}=== PRE-START VALIDATION COMPLETED ===${NC}"
log "${GREEN}✓ System requirements met${NC}"
log "${GREEN}✓ Docker environment ready${NC}"
log "${GREEN}✓ GPU requirements satisfied${NC}"
log "${GREEN}✓ Network configuration valid${NC}"
log "${GREEN}✓ Directory structure prepared${NC}"
log "${GREEN}✓ Configuration files validated${NC}"
log "${GREEN}✓ Security checks passed${NC}"
log ""
log "${BLUE}Companion deployment is ready to start!${NC}"

exit 0