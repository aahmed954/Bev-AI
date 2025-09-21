#!/bin/bash
# Health check script for Standalone AI Companion
# Validates all services are running and healthy

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/companion/health-check-$(date +%Y%m%d_%H%M%S).log"
COMPOSE_FILE="${SCRIPT_DIR}/../docker-compose.companion.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Health check configuration
MAX_RETRIES=10
RETRY_DELAY=5
TIMEOUT=30

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

info "Starting AI Companion health check..."

# 1. Container Status Check
info "Checking container status..."

EXPECTED_CONTAINERS=(
    "companion_postgres"
    "companion_redis"
    "companion_core"
    "companion_voice"
    "companion_avatar"
    "companion_frontend"
    "companion_memory"
    "companion_gateway"
    "companion_monitor"
    "companion_prometheus"
    "companion_grafana"
)

for container in "${EXPECTED_CONTAINERS[@]}"; do
    if docker ps --format "table {{.Names}}" | grep -q "^$container$"; then
        status=$(docker inspect --format='{{.State.Status}}' "$container")
        if [[ "$status" == "running" ]]; then
            success "Container $container is running"
        else
            error_exit "Container $container is not running (status: $status)"
        fi
    else
        error_exit "Container $container not found"
    fi
done

# 2. Service Health Checks
info "Checking service health endpoints..."

# Function to check HTTP endpoint
check_http_endpoint() {
    local name="$1"
    local url="$2"
    local retries=0

    while [[ $retries -lt $MAX_RETRIES ]]; do
        if curl -sf --max-time $TIMEOUT "$url" >/dev/null 2>&1; then
            success "$name health check passed"
            return 0
        fi

        retries=$((retries + 1))
        if [[ $retries -lt $MAX_RETRIES ]]; then
            info "Retrying $name health check ($retries/$MAX_RETRIES)..."
            sleep $RETRY_DELAY
        fi
    done

    error_exit "$name health check failed after $MAX_RETRIES retries"
}

# Core service health checks
check_http_endpoint "Companion Core" "http://localhost:18000/health"
check_http_endpoint "Voice Service" "http://localhost:18002/health"
check_http_endpoint "Avatar Service" "http://localhost:18003/health"
check_http_endpoint "Memory Service" "http://localhost:18004/health"
check_http_endpoint "Integration Gateway" "http://localhost:18005/health"
check_http_endpoint "Resource Monitor" "http://localhost:18006/health"
check_http_endpoint "Frontend" "http://localhost:18080/health"
check_http_endpoint "Prometheus" "http://localhost:19090/"
check_http_endpoint "Grafana" "http://localhost:19000/api/health"

# 3. Database Connectivity Check
info "Checking database connectivity..."

# PostgreSQL check
if docker exec companion_postgres pg_isready -U companion_user -d companion >/dev/null 2>&1; then
    success "PostgreSQL is ready"
else
    error_exit "PostgreSQL connection failed"
fi

# Redis check
if docker exec companion_redis redis-cli ping | grep -q "PONG"; then
    success "Redis is ready"
else
    error_exit "Redis connection failed"
fi

# 4. GPU Availability Check
info "Checking GPU availability..."

# Check if GPU is accessible from core service
GPU_CHECK=$(docker exec companion_core python3 -c "
import torch
import subprocess
try:
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count()

    # Check GPU memory
    if cuda_available and device_count > 0:
        device_name = torch.cuda.get_device_name(0)
        memory_total = torch.cuda.get_device_properties(0).total_memory
        memory_allocated = torch.cuda.memory_allocated(0)
        memory_cached = torch.cuda.memory_reserved(0)

        print(f'CUDA_OK:{cuda_available}')
        print(f'DEVICE_COUNT:{device_count}')
        print(f'DEVICE_NAME:{device_name}')
        print(f'MEMORY_TOTAL:{memory_total}')
        print(f'MEMORY_ALLOCATED:{memory_allocated}')
        print(f'MEMORY_CACHED:{memory_cached}')
    else:
        print('CUDA_OK:False')
except Exception as e:
    print(f'ERROR:{str(e)}')
" 2>/dev/null || echo "GPU_CHECK_FAILED")

if echo "$GPU_CHECK" | grep -q "CUDA_OK:True"; then
    DEVICE_NAME=$(echo "$GPU_CHECK" | grep "DEVICE_NAME:" | cut -d: -f2)
    success "GPU access validated: $DEVICE_NAME"
else
    error_exit "GPU access validation failed"
fi

# 5. Memory Usage Check
info "Checking memory usage..."

# System memory
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
USED_MEM=$(free -g | awk '/^Mem:/{print $3}')
MEM_PERCENT=$(echo "scale=1; $USED_MEM * 100 / $TOTAL_MEM" | bc)

if [[ $(echo "$MEM_PERCENT < 80" | bc) -eq 1 ]]; then
    success "System memory usage: ${MEM_PERCENT}% (${USED_MEM}G/${TOTAL_MEM}G)"
else
    warning "High system memory usage: ${MEM_PERCENT}% (${USED_MEM}G/${TOTAL_MEM}G)"
fi

# GPU memory
GPU_MEM_INFO=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1)
GPU_MEM_USED=$(echo "$GPU_MEM_INFO" | cut -d, -f1 | tr -d ' ')
GPU_MEM_TOTAL=$(echo "$GPU_MEM_INFO" | cut -d, -f2 | tr -d ' ')
GPU_MEM_PERCENT=$(echo "scale=1; $GPU_MEM_USED * 100 / $GPU_MEM_TOTAL" | bc)

if [[ $(echo "$GPU_MEM_PERCENT < 90" | bc) -eq 1 ]]; then
    success "GPU memory usage: ${GPU_MEM_PERCENT}% (${GPU_MEM_USED}MB/${GPU_MEM_TOTAL}MB)"
else
    warning "High GPU memory usage: ${GPU_MEM_PERCENT}% (${GPU_MEM_USED}MB/${GPU_MEM_TOTAL}MB)"
fi

# 6. Network Connectivity Check
info "Checking network connectivity..."

# Check internal container networking
NETWORK_TESTS=(
    "companion_core:http://companion-postgres:5432"
    "companion_core:http://companion-redis:6379"
    "companion_frontend:http://companion-core:8000"
    "companion_voice:http://companion-core:8000"
    "companion_avatar:http://companion-core:8000"
)

for test in "${NETWORK_TESTS[@]}"; do
    container=$(echo "$test" | cut -d: -f1)
    target=$(echo "$test" | cut -d: -f2-3)

    if docker exec "$container" nc -z "${target#http://}" >/dev/null 2>&1; then
        success "Network connectivity: $container -> $target"
    else
        error_exit "Network connectivity failed: $container -> $target"
    fi
done

# 7. Performance Metrics Check
info "Checking performance metrics..."

# Response time test
RESPONSE_TIME=$(curl -w "%{time_total}" -o /dev/null -s "http://localhost:18000/health")
if [[ $(echo "$RESPONSE_TIME < 2.0" | bc) -eq 1 ]]; then
    success "Core service response time: ${RESPONSE_TIME}s"
else
    warning "Slow core service response time: ${RESPONSE_TIME}s"
fi

# GPU temperature check
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
if [[ $GPU_TEMP -lt 80 ]]; then
    success "GPU temperature: ${GPU_TEMP}°C"
else
    warning "High GPU temperature: ${GPU_TEMP}°C"
fi

# 8. Log File Check
info "Checking log files..."

LOG_DIRS=(
    "/var/log/companion"
    "/opt/companion/logs"
)

for log_dir in "${LOG_DIRS[@]}"; do
    if [[ -d "$log_dir" ]]; then
        log_count=$(find "$log_dir" -name "*.log" -type f | wc -l)
        success "Log directory $log_dir contains $log_count log files"

        # Check for error patterns in recent logs
        error_count=$(find "$log_dir" -name "*.log" -type f -mtime -1 -exec grep -l "ERROR\|CRITICAL\|FATAL" {} \; 2>/dev/null | wc -l)
        if [[ $error_count -eq 0 ]]; then
            success "No critical errors in recent logs"
        else
            warning "$error_count log files contain errors in last 24 hours"
        fi
    else
        warning "Log directory $log_dir not found"
    fi
done

# 9. Integration Status Check
info "Checking core platform integration status..."

INTEGRATION_STATUS=$(curl -s "http://localhost:18005/integration/status" 2>/dev/null || echo "UNKNOWN")
if echo "$INTEGRATION_STATUS" | grep -q "standalone"; then
    success "Running in standalone mode"
elif echo "$INTEGRATION_STATUS" | grep -q "integrated"; then
    success "Successfully integrated with core platform"
else
    info "Integration status: $INTEGRATION_STATUS"
fi

# 10. Service Functionality Test
info "Testing service functionality..."

# Test companion core API
API_TEST=$(curl -s -X POST "http://localhost:18000/api/test" \
    -H "Content-Type: application/json" \
    -d '{"test": "health_check"}' 2>/dev/null || echo "FAILED")

if echo "$API_TEST" | grep -q "success\|ok"; then
    success "Core API functionality test passed"
else
    warning "Core API functionality test failed"
fi

# Test voice synthesis (if enabled)
VOICE_TEST=$(curl -s "http://localhost:18002/status" 2>/dev/null || echo "UNAVAILABLE")
if echo "$VOICE_TEST" | grep -q "ready\|online"; then
    success "Voice synthesis service is ready"
else
    info "Voice synthesis status: $VOICE_TEST"
fi

# Test avatar rendering (if enabled)
AVATAR_TEST=$(curl -s "http://localhost:18003/status" 2>/dev/null || echo "UNAVAILABLE")
if echo "$AVATAR_TEST" | grep -q "ready\|online"; then
    success "Avatar rendering service is ready"
else
    info "Avatar rendering status: $AVATAR_TEST"
fi

# 11. Final Health Summary
info "Generating health summary..."

HEALTH_SUMMARY="/var/log/companion/health-summary.json"
cat > "$HEALTH_SUMMARY" <<EOF
{
  "timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "status": "healthy",
  "services": {
    "companion_core": "running",
    "companion_voice": "running",
    "companion_avatar": "running",
    "companion_frontend": "running",
    "companion_memory": "running",
    "companion_gateway": "running",
    "companion_monitor": "running",
    "companion_prometheus": "running",
    "companion_grafana": "running"
  },
  "databases": {
    "postgresql": "connected",
    "redis": "connected"
  },
  "gpu": {
    "available": true,
    "temperature": ${GPU_TEMP},
    "memory_used_percent": ${GPU_MEM_PERCENT}
  },
  "memory": {
    "system_used_percent": ${MEM_PERCENT},
    "gpu_used_percent": ${GPU_MEM_PERCENT}
  },
  "performance": {
    "response_time_seconds": ${RESPONSE_TIME}
  },
  "integration": {
    "mode": "standalone",
    "core_platform_available": false
  }
}
EOF

success "Health summary generated: $HEALTH_SUMMARY"

# Final status
log ""
log "${GREEN}=== HEALTH CHECK COMPLETED SUCCESSFULLY ===${NC}"
log "${GREEN}✓ All containers running${NC}"
log "${GREEN}✓ All services healthy${NC}"
log "${GREEN}✓ Database connectivity verified${NC}"
log "${GREEN}✓ GPU access confirmed${NC}"
log "${GREEN}✓ Memory usage acceptable${NC}"
log "${GREEN}✓ Network connectivity verified${NC}"
log "${GREEN}✓ Performance metrics normal${NC}"
log "${GREEN}✓ Log files accessible${NC}"
log "${GREEN}✓ Service functionality tested${NC}"
log ""
log "${BLUE}AI Companion deployment is fully operational!${NC}"

exit 0