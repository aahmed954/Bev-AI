#!/bin/bash
#
# BEV Advanced Avatar Health Monitoring Script
# Comprehensive health monitoring with alerts and recovery
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEV_ROOT="/home/starlord/Projects/Bev"
LOG_FILE="$BEV_ROOT/logs/avatar-health-monitor.log"
HEALTH_DATA_DIR="$BEV_ROOT/logs/health"
ALERT_FILE="$HEALTH_DATA_DIR/alerts.json"
SERVICE_NAME="bev-advanced-avatar"

# Ensure directories exist
mkdir -p "$HEALTH_DATA_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] HEALTH: $*" | tee -a "$LOG_FILE"
}

# Alert levels
ALERT_CRITICAL=3
ALERT_WARNING=2
ALERT_INFO=1

# Configuration
MONITORING_INTERVAL=30
MAX_RESPONSE_TIME=5000  # 5 seconds
MIN_FREE_MEMORY_MB=2048
MAX_GPU_TEMP=80
MAX_CPU_USAGE=90
MAX_MEMORY_USAGE=85

# Function to send alert
send_alert() {
    local level=$1
    local message=$2
    local metric=${3:-"general"}
    
    local alert_data=$(cat << EOF
{
    "timestamp": $(date +%s),
    "level": $level,
    "message": "$message",
    "metric": "$metric",
    "hostname": "$(hostname)",
    "service": "$SERVICE_NAME"
}
EOF
)
    
    # Log to file
    echo "$alert_data" >> "$ALERT_FILE"
    
    # Log to console with color
    case $level in
        $ALERT_CRITICAL)
            log "${RED}🚨 CRITICAL: $message${NC}"
            ;;
        $ALERT_WARNING)
            log "${YELLOW}⚠ WARNING: $message${NC}"
            ;;
        $ALERT_INFO)
            log "${BLUE}ℹ INFO: $message${NC}"
            ;;
    esac
    
    # Store in Redis if available
    if command -v redis-cli &>/dev/null && redis-cli -h localhost -p 6379 ping &>/dev/null; then
        redis-cli -h localhost -p 6379 LPUSH "avatar:alerts" "$alert_data" >/dev/null 2>&1 || true
        redis-cli -h localhost -p 6379 LTRIM "avatar:alerts" 0 100 >/dev/null 2>&1 || true  # Keep last 100 alerts
    fi
}

# Function to check service status
check_service_status() {
    local status_file="$HEALTH_DATA_DIR/service_status.json"
    
    local is_active=$(systemctl is-active --quiet "$SERVICE_NAME" && echo "true" || echo "false")
    local is_enabled=$(systemctl is-enabled --quiet "$SERVICE_NAME" && echo "true" || echo "false")
    local uptime_seconds=0
    
    if [[ "$is_active" == "true" ]]; then
        # Get service uptime
        local start_time=$(systemctl show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value)
        if [[ -n "$start_time" ]]; then
            local start_epoch=$(date -d "$start_time" +%s 2>/dev/null || echo 0)
            local current_epoch=$(date +%s)
            uptime_seconds=$((current_epoch - start_epoch))
        fi
    else
        send_alert $ALERT_CRITICAL "Avatar service is not active" "service_status"
    fi
    
    local status_data=$(cat << EOF
{
    "timestamp": $(date +%s),
    "active": $is_active,
    "enabled": $is_enabled,
    "uptime_seconds": $uptime_seconds,
    "uptime_human": "$(printf '%02d:%02d:%02d' $((uptime_seconds/3600)) $((uptime_seconds%3600/60)) $((uptime_seconds%60)))"
}
EOF
)
    
    echo "$status_data" > "$status_file"
    
    if [[ "$is_active" == "false" ]]; then
        return 1
    fi
    
    return 0
}

# Function to check avatar endpoint health
check_avatar_endpoint() {
    local endpoint_file="$HEALTH_DATA_DIR/endpoint_health.json"
    local avatar_url="http://localhost:8080/health"
    
    local start_time=$(date +%s%3N)
    local http_status=0
    local response_body=""
    local error_message=""
    
    if response=$(curl -s -w "%{http_code}" --max-time 10 "$avatar_url" 2>/dev/null); then
        http_status="${response: -3}"
        response_body="${response%???}"
    else
        error_message="Connection failed"
        send_alert $ALERT_CRITICAL "Avatar endpoint unreachable: $avatar_url" "endpoint"
    fi
    
    local end_time=$(date +%s%3N)
    local response_time=$((end_time - start_time))
    
    local endpoint_data=$(cat << EOF
{
    "timestamp": $(date +%s),
    "url": "$avatar_url",
    "http_status": $http_status,
    "response_time_ms": $response_time,
    "accessible": $(if [[ $http_status -eq 200 ]]; then echo "true"; else echo "false"; fi),
    "error": "$error_message"
}
EOF
)
    
    echo "$endpoint_data" > "$endpoint_file"
    
    # Check response time
    if [[ $response_time -gt $MAX_RESPONSE_TIME ]]; then
        send_alert $ALERT_WARNING "High response time: ${response_time}ms (max: ${MAX_RESPONSE_TIME}ms)" "performance"
    fi
    
    # Check HTTP status
    if [[ $http_status -ne 200 && $http_status -ne 0 ]]; then
        send_alert $ALERT_WARNING "Avatar endpoint returned HTTP $http_status" "endpoint"
        return 1
    fi
    
    return 0
}

# Function to check GPU health
check_gpu_health() {
    local gpu_file="$HEALTH_DATA_DIR/gpu_health.json"
    
    if ! command -v nvidia-smi &>/dev/null; then
        send_alert $ALERT_WARNING "nvidia-smi not available for GPU monitoring" "gpu"
        return 1
    fi
    
    local gpu_data=$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null || echo ",,,,," )
    
    if [[ -z "$gpu_data" || "$gpu_data" == ",,,,," ]]; then
        send_alert $ALERT_CRITICAL "Failed to retrieve GPU data" "gpu"
        return 1
    fi
    
    IFS=',' read -r name temperature utilization memory_used memory_total power_draw <<< "$gpu_data"
    
    # Clean up values
    name=$(echo "$name" | xargs)
    temperature=$(echo "$temperature" | xargs)
    utilization=$(echo "$utilization" | xargs)
    memory_used=$(echo "$memory_used" | xargs)
    memory_total=$(echo "$memory_total" | xargs)
    power_draw=$(echo "$power_draw" | xargs)
    
    local memory_usage_percent=0
    if [[ "$memory_total" -gt 0 ]]; then
        memory_usage_percent=$(( (memory_used * 100) / memory_total ))
    fi
    
    local gpu_health_data=$(cat << EOF
{
    "timestamp": $(date +%s),
    "name": "$name",
    "temperature_celsius": $temperature,
    "utilization_percent": $utilization,
    "memory_used_mb": $memory_used,
    "memory_total_mb": $memory_total,
    "memory_usage_percent": $memory_usage_percent,
    "power_draw_watts": $power_draw
}
EOF
)
    
    echo "$gpu_health_data" > "$gpu_file"
    
    # Check temperature
    if [[ "$temperature" -gt $MAX_GPU_TEMP ]]; then
        send_alert $ALERT_WARNING "High GPU temperature: ${temperature}°C (max: ${MAX_GPU_TEMP}°C)" "gpu_temperature"
    fi
    
    # Check memory usage
    if [[ "$memory_usage_percent" -gt 90 ]]; then
        send_alert $ALERT_WARNING "High GPU memory usage: ${memory_usage_percent}%" "gpu_memory"
    fi
    
    return 0
}

# Function to check system resources
check_system_resources() {
    local system_file="$HEALTH_DATA_DIR/system_resources.json"
    
    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1 | cut -d'u' -f1)
    
    # Memory usage
    local memory_info=$(free -m | awk 'NR==2{printf "%.1f %.1f %.1f", $3*100/$2, $4, $2}')
    read -r memory_usage_percent memory_free_mb memory_total_mb <<< "$memory_info"
    
    # Disk usage for BEV directory
    local disk_usage=$(df "$BEV_ROOT" | awk 'NR==2 {print $5}' | cut -d'%' -f1)
    local disk_free=$(df -BM "$BEV_ROOT" | awk 'NR==2 {print $4}' | cut -d'M' -f1)
    
    # Load average
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | cut -d',' -f1)
    
    local system_data=$(cat << EOF
{
    "timestamp": $(date +%s),
    "cpu_usage_percent": $cpu_usage,
    "memory_usage_percent": $memory_usage_percent,
    "memory_free_mb": $memory_free_mb,
    "memory_total_mb": $memory_total_mb,
    "disk_usage_percent": $disk_usage,
    "disk_free_mb": $disk_free,
    "load_average": $load_avg
}
EOF
)
    
    echo "$system_data" > "$system_file"
    
    # Check CPU usage
    if (( $(echo "$cpu_usage > $MAX_CPU_USAGE" | bc -l) )); then
        send_alert $ALERT_WARNING "High CPU usage: ${cpu_usage}% (max: ${MAX_CPU_USAGE}%)" "cpu"
    fi
    
    # Check memory usage
    if (( $(echo "$memory_usage_percent > $MAX_MEMORY_USAGE" | bc -l) )); then
        send_alert $ALERT_WARNING "High memory usage: ${memory_usage_percent}% (max: ${MAX_MEMORY_USAGE}%)" "memory"
    fi
    
    # Check free memory
    if [[ "$memory_free_mb" -lt $MIN_FREE_MEMORY_MB ]]; then
        send_alert $ALERT_WARNING "Low free memory: ${memory_free_mb}MB (min: ${MIN_FREE_MEMORY_MB}MB)" "memory"
    fi
    
    # Check disk space
    if [[ "$disk_usage" -gt 90 ]]; then
        send_alert $ALERT_WARNING "High disk usage: ${disk_usage}%" "disk"
    fi
    
    return 0
}

# Function to check Redis connectivity
check_redis_health() {
    local redis_file="$HEALTH_DATA_DIR/redis_health.json"
    
    local redis_accessible="false"
    local redis_memory=""
    local redis_keys=0
    local error_message=""
    
    if command -v redis-cli &>/dev/null; then
        if timeout 5 redis-cli -h localhost -p 6379 ping &>/dev/null; then
            redis_accessible="true"
            redis_memory=$(redis-cli -h localhost -p 6379 INFO memory | grep "used_memory_human:" | cut -d: -f2 | tr -d '\r' 2>/dev/null || echo "unknown")
            redis_keys=$(redis-cli -h localhost -p 6379 DBSIZE 2>/dev/null || echo 0)
        else
            error_message="Redis connection failed"
            send_alert $ALERT_WARNING "Redis not accessible" "redis"
        fi
    else
        error_message="redis-cli not available"
    fi
    
    local redis_data=$(cat << EOF
{
    "timestamp": $(date +%s),
    "accessible": $redis_accessible,
    "memory_usage": "$redis_memory",
    "total_keys": $redis_keys,
    "error": "$error_message"
}
EOF
)
    
    echo "$redis_data" > "$redis_file"
    
    return 0
}

# Function to generate health summary
generate_health_summary() {
    local summary_file="$HEALTH_DATA_DIR/health_summary.json"
    
    local overall_status="healthy"
    local critical_issues=0
    local warning_issues=0
    
    # Count recent alerts
    if [[ -f "$ALERT_FILE" ]]; then
        local recent_alerts=$(tail -100 "$ALERT_FILE" | grep "\"level\": $ALERT_CRITICAL" | wc -l)
        critical_issues=$recent_alerts
        
        recent_alerts=$(tail -100 "$ALERT_FILE" | grep "\"level\": $ALERT_WARNING" | wc -l)
        warning_issues=$recent_alerts
    fi
    
    if [[ $critical_issues -gt 0 ]]; then
        overall_status="critical"
    elif [[ $warning_issues -gt 5 ]]; then
        overall_status="degraded"
    fi
    
    local summary_data=$(cat << EOF
{
    "timestamp": $(date +%s),
    "overall_status": "$overall_status",
    "critical_issues": $critical_issues,
    "warning_issues": $warning_issues,
    "monitoring_interval": $MONITORING_INTERVAL,
    "last_check": "$(date -Iseconds)"
}
EOF
)
    
    echo "$summary_data" > "$summary_file"
    
    # Update Redis if available
    if command -v redis-cli &>/dev/null && redis-cli -h localhost -p 6379 ping &>/dev/null; then
        redis-cli -h localhost -p 6379 SET "avatar:health_summary" "$summary_data" EX 300 >/dev/null 2>&1 || true
    fi
}

# Function to attempt service recovery
attempt_recovery() {
    local issue_type=$1
    
    log "Attempting recovery for issue: $issue_type"
    
    case $issue_type in
        "service_down")
            log "Attempting to restart avatar service..."
            if sudo systemctl restart "$SERVICE_NAME"; then
                send_alert $ALERT_INFO "Service restart successful" "recovery"
                sleep 30  # Wait for service to fully start
                return 0
            else
                send_alert $ALERT_CRITICAL "Service restart failed" "recovery"
                return 1
            fi
            ;;
        "gpu_memory")
            log "Attempting GPU memory cleanup..."
            if python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null; then
                send_alert $ALERT_INFO "GPU memory cleanup successful" "recovery"
                return 0
            else
                send_alert $ALERT_WARNING "GPU memory cleanup failed" "recovery"
                return 1
            fi
            ;;
        *)
            log "No recovery action defined for: $issue_type"
            return 1
            ;;
    esac
}

# Function to run health check cycle
run_health_check() {
    log "Starting health check cycle..."
    
    local checks_passed=0
    local checks_total=5
    
    # Check service status
    if check_service_status; then
        ((checks_passed++))
    else
        attempt_recovery "service_down"
    fi
    
    # Check avatar endpoint
    if check_avatar_endpoint; then
        ((checks_passed++))
    fi
    
    # Check GPU health
    if check_gpu_health; then
        ((checks_passed++))
    fi
    
    # Check system resources
    if check_system_resources; then
        ((checks_passed++))
    fi
    
    # Check Redis
    if check_redis_health; then
        ((checks_passed++))
    fi
    
    # Generate summary
    generate_health_summary
    
    log "Health check completed: $checks_passed/$checks_total checks passed"
    
    if [[ $checks_passed -lt $((checks_total / 2)) ]]; then
        send_alert $ALERT_CRITICAL "Multiple health checks failing: $checks_passed/$checks_total" "general"
    fi
}

# Function to display current health status
display_health_status() {
    echo "=== BEV Avatar Health Status ==="
    echo ""
    
    if [[ -f "$HEALTH_DATA_DIR/health_summary.json" ]]; then
        local summary=$(cat "$HEALTH_DATA_DIR/health_summary.json")
        local status=$(echo "$summary" | grep -o '"overall_status": "[^"]*"' | cut -d'"' -f4)
        local last_check=$(echo "$summary" | grep -o '"last_check": "[^"]*"' | cut -d'"' -f4)
        
        case $status in
            "healthy")
                echo -e "${GREEN}Status: ✓ Healthy${NC}"
                ;;
            "degraded")
                echo -e "${YELLOW}Status: ⚠ Degraded${NC}"
                ;;
            "critical")
                echo -e "${RED}Status: ✗ Critical${NC}"
                ;;
        esac
        
        echo "Last Check: $last_check"
    else
        echo -e "${YELLOW}Status: Unknown (no health data available)${NC}"
    fi
    
    echo ""
    echo "Health data directory: $HEALTH_DATA_DIR"
    echo "Monitor logs: $LOG_FILE"
    echo ""
}

# Main function
main() {
    case "${1:-}" in
        "status")
            display_health_status
            ;;
        "once")
            run_health_check
            ;;
        "monitor")
            log "Starting continuous health monitoring (interval: ${MONITORING_INTERVAL}s)"
            while true; do
                run_health_check
                sleep $MONITORING_INTERVAL
            done
            ;;
        *)
            echo "Usage: $0 {status|once|monitor}"
            echo ""
            echo "Commands:"
            echo "  status  - Display current health status"
            echo "  once    - Run health check once"
            echo "  monitor - Start continuous monitoring"
            echo ""
            exit 1
            ;;
    esac
}

# Handle signals for graceful shutdown
trap 'log "Health monitoring stopped"; exit 0' SIGTERM SIGINT

# Execute main function
main "$@"