#!/bin/bash
#
# BEV Avatar Dependencies Check Script
# Validates all dependencies and service readiness
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEV_ROOT="/home/starlord/Projects/Bev"
LOG_FILE="$BEV_ROOT/logs/avatar-dependencies.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DEPS: $*" | tee -a "$LOG_FILE"
}

log "=== BEV Avatar Dependencies Check ==="

# Check Docker service
if ! systemctl is-active --quiet docker; then
    log "ERROR: Docker service is not running"
    exit 1
fi

log "Docker service: Active"

# Check Redis availability and configuration
check_redis() {
    local redis_host="localhost"
    local redis_port="6379"
    
    if ! timeout 10 redis-cli -h "$redis_host" -p "$redis_port" ping &>/dev/null; then
        log "ERROR: Redis not accessible at $redis_host:$redis_port"
        return 1
    fi
    
    # Check Redis memory configuration
    local redis_memory=$(redis-cli -h "$redis_host" -p "$redis_port" CONFIG GET maxmemory | tail -1)
    if [[ "$redis_memory" == "0" ]]; then
        log "WARNING: Redis maxmemory not configured (unlimited)"
    else
        log "Redis maxmemory: $redis_memory bytes"
    fi
    
    # Test Redis operations
    local test_key="bev:avatar:healthcheck:$(date +%s)"
    if ! redis-cli -h "$redis_host" -p "$redis_port" SET "$test_key" "test" EX 60 &>/dev/null; then
        log "ERROR: Cannot write to Redis"
        return 1
    fi
    
    if ! redis-cli -h "$redis_host" -p "$redis_port" GET "$test_key" &>/dev/null; then
        log "ERROR: Cannot read from Redis"
        return 1
    fi
    
    redis-cli -h "$redis_host" -p "$redis_port" DEL "$test_key" &>/dev/null
    
    log "Redis: Available and functional"
    return 0
}

check_redis || exit 1

# Check MCP Server availability
check_mcp_server() {
    local mcp_url="http://localhost:3010"
    
    if ! timeout 10 curl -s "$mcp_url/health" &>/dev/null; then
        log "WARNING: MCP Server not available at $mcp_url (will start with avatar)"
        return 0
    fi
    
    log "MCP Server: Available at $mcp_url"
    return 0
}

check_mcp_server

# Check network interfaces
check_network() {
    # Check if localhost is resolvable
    if ! getent hosts localhost &>/dev/null; then
        log "ERROR: localhost not resolvable"
        exit 1
    fi
    
    # Check if we can bind to avatar port
    local avatar_port="8080"
    if ss -tuln | grep -q ":$avatar_port "; then
        local pid=$(ss -tulnp | grep ":$avatar_port " | awk '{print $6}' | cut -d',' -f2 | cut -d'=' -f2)
        log "ERROR: Port $avatar_port already in use by PID: $pid"
        exit 1
    fi
    
    log "Network: Ready (port $avatar_port available)"
}

check_network

# Check Python environment and packages
check_python_environment() {
    local python_cmd="python3"
    
    if ! command -v "$python_cmd" &>/dev/null; then
        log "ERROR: Python3 not found"
        exit 1
    fi
    
    local python_version=$($python_cmd --version 2>&1 | awk '{print $2}')
    log "Python version: $python_version"
    
    # Check critical packages with version requirements
    local packages=(
        "torch>=2.0.0"
        "torchvision>=0.15.0"
        "fastapi>=0.100.0"
        "uvicorn[standard]>=0.23.0"
        "redis>=4.5.0"
        "numpy>=1.24.0"
        "scipy>=1.10.0"
        "pillow>=9.5.0"
        "transformers>=4.30.0"
        "accelerate>=0.20.0"
        "safetensors>=0.3.0"
    )
    
    log "Checking Python packages..."
    for package in "${packages[@]}"; do
        local pkg_name=$(echo "$package" | cut -d'>' -f1 | cut -d'=' -f1)
        if ! $python_cmd -c "import $pkg_name" 2>/dev/null; then
            log "ERROR: Python package missing: $pkg_name"
            exit 1
        fi
    done
    
    # Check PyTorch CUDA availability
    if ! $python_cmd -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        log "ERROR: PyTorch CUDA support not available"
        exit 1
    fi
    
    # Check specific avatar dependencies
    local avatar_deps=(
        "websockets"
        "asyncio"
        "aioredis"
        "pydantic"
        "typing_extensions"
    )
    
    for dep in "${avatar_deps[@]}"; do
        if ! $python_cmd -c "import $dep" 2>/dev/null; then
            log "ERROR: Avatar dependency missing: $dep"
            exit 1
        fi
    done
    
    log "Python environment: All packages available"
}

check_python_environment

# Check optional AI model dependencies
check_optional_dependencies() {
    local python_cmd="python3"
    
    # Check for advanced TTS (Bark)
    if $python_cmd -c "import bark" 2>/dev/null; then
        log "Advanced TTS (Bark): Available"
    else
        log "Advanced TTS (Bark): Not available (optional)"
    fi
    
    # Check for Gaussian Splatting
    if $python_cmd -c "import gsplat" 2>/dev/null; then
        log "Gaussian Splatting: Available"
    else
        log "Gaussian Splatting: Not available (optional)"
    fi
    
    # Check for speech recognition
    if $python_cmd -c "import whisper" 2>/dev/null; then
        log "Whisper STT: Available"
    else
        log "Whisper STT: Not available (optional)"
    fi
}

check_optional_dependencies

# Check filesystem permissions and space
check_filesystem() {
    local required_dirs=(
        "$BEV_ROOT/logs"
        "$BEV_ROOT/data"
        "$BEV_ROOT/config"
        "$BEV_ROOT/src/avatar"
    )
    
    for dir in "${required_dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            log "Creating directory: $dir"
            mkdir -p "$dir"
        fi
        
        if [[ ! -w "$dir" ]]; then
            log "ERROR: No write permission for: $dir"
            exit 1
        fi
    done
    
    # Check disk space (minimum 5GB available)
    local available_space=$(df "$BEV_ROOT" | awk 'NR==2 {print $4}')
    local required_space=$((5 * 1024 * 1024))  # 5GB in KB
    
    if [[ "$available_space" -lt "$required_space" ]]; then
        log "ERROR: Insufficient disk space. Available: ${available_space}KB, Required: ${required_space}KB"
        exit 1
    fi
    
    log "Filesystem: Permissions and space OK"
}

check_filesystem

# Check systemd journal configuration
check_logging() {
    # Check if systemd journal is available
    if ! command -v journalctl &>/dev/null; then
        log "WARNING: journalctl not available"
        return 0
    fi
    
    # Check journal disk usage
    local journal_size=$(journalctl --disk-usage 2>/dev/null | awk '{print $7}' | sed 's/\.$//')
    if [[ -n "$journal_size" ]]; then
        log "Journal disk usage: $journal_size"
    fi
    
    # Ensure log rotation is configured
    if [[ ! -f "/etc/logrotate.d/bev-avatar" ]]; then
        log "WARNING: Log rotation not configured for BEV avatar logs"
    fi
    
    log "Logging: Journal available"
}

check_logging

# Check system resources
check_system_resources() {
    # Check memory
    local total_mem=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
    local available_mem=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
    local required_mem=$((8 * 1024 * 1024))  # 8GB in KB
    
    log "Memory - Total: ${total_mem}KB, Available: ${available_mem}KB"
    
    if [[ "$available_mem" -lt "$required_mem" ]]; then
        log "WARNING: Low available memory for avatar operation"
    fi
    
    # Check CPU cores
    local cpu_cores=$(nproc)
    log "CPU cores: $cpu_cores"
    
    if [[ "$cpu_cores" -lt 4 ]]; then
        log "WARNING: Low CPU core count for optimal avatar performance"
    fi
    
    # Check load average
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    log "Load average: $load_avg"
}

check_system_resources

# Test critical service communications
test_service_communication() {
    log "Testing service communications..."
    
    # Test Redis pub/sub for avatar updates
    local test_channel="bev:avatar:test:$(date +%s)"
    timeout 5 bash -c "
        redis-cli -h localhost -p 6379 SUBSCRIBE '$test_channel' &
        sleep 1
        redis-cli -h localhost -p 6379 PUBLISH '$test_channel' 'test_message'
        sleep 1
        pkill -f 'redis-cli.*SUBSCRIBE'
    " &>/dev/null || log "WARNING: Redis pub/sub test failed"
    
    log "Service communication tests completed"
}

test_service_communication

# Final validation summary
log "Dependencies check completed successfully"
log "System ready for BEV Advanced Avatar service"
log "=== Dependencies Check Complete ==="

exit 0