#!/bin/bash
#
# BEV Advanced Avatar Service Management Script
# Comprehensive service control with diagnostics and recovery
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEV_ROOT="/home/starlord/Projects/Bev"
SERVICE_NAME="bev-advanced-avatar"
LOG_FILE="$BEV_ROOT/logs/service-manager.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Icons
CHECK="✓"
CROSS="✗"
WARNING="⚠"
INFO="ℹ"
ROCKET="🚀"
GEAR="⚙"
HEALTH="🏥"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"
}

# Function to display service status with colors
display_status() {
    echo -e "${BOLD}=== BEV Advanced Avatar Service Status ===${NC}"
    echo ""
    
    # Service status
    local is_active=$(systemctl is-active --quiet "$SERVICE_NAME" && echo "active" || echo "inactive")
    local is_enabled=$(systemctl is-enabled --quiet "$SERVICE_NAME" && echo "enabled" || echo "disabled")
    
    if [[ "$is_active" == "active" ]]; then
        echo -e "${GREEN}${CHECK} Service Status: ${BOLD}ACTIVE${NC}"
    else
        echo -e "${RED}${CROSS} Service Status: ${BOLD}INACTIVE${NC}"
    fi
    
    if [[ "$is_enabled" == "enabled" ]]; then
        echo -e "${GREEN}${CHECK} Auto-start: ${BOLD}ENABLED${NC}"
    else
        echo -e "${YELLOW}${WARNING} Auto-start: ${BOLD}DISABLED${NC}"
    fi
    
    # Uptime
    if [[ "$is_active" == "active" ]]; then
        local start_time=$(systemctl show "$SERVICE_NAME" --property=ActiveEnterTimestamp --value)
        if [[ -n "$start_time" ]]; then
            local start_epoch=$(date -d "$start_time" +%s 2>/dev/null || echo 0)
            local current_epoch=$(date +%s)
            local uptime_seconds=$((current_epoch - start_epoch))
            local uptime_human=$(printf '%02d:%02d:%02d' $((uptime_seconds/3600)) $((uptime_seconds%3600/60)) $((uptime_seconds%60)))
            echo -e "${BLUE}${INFO} Uptime: ${BOLD}$uptime_human${NC}"
        fi
    fi
    
    # Endpoint status
    if curl -s -f http://localhost:8080/health >/dev/null 2>&1; then
        echo -e "${GREEN}${CHECK} Endpoint: ${BOLD}RESPONDING${NC} (http://localhost:8080)"
    else
        echo -e "${RED}${CROSS} Endpoint: ${BOLD}NOT RESPONDING${NC}"
    fi
    
    # GPU status
    if command -v nvidia-smi &>/dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo ",,")
        IFS=',' read -r gpu_name gpu_temp gpu_util <<< "$gpu_info"
        
        if [[ -n "$gpu_name" ]]; then
            gpu_name=$(echo "$gpu_name" | xargs)
            gpu_temp=$(echo "$gpu_temp" | xargs)
            gpu_util=$(echo "$gpu_util" | xargs)
            
            echo -e "${CYAN}${GEAR} GPU: ${BOLD}$gpu_name${NC}"
            echo -e "${CYAN}${INFO} Temperature: ${BOLD}${gpu_temp}°C${NC}, Utilization: ${BOLD}${gpu_util}%${NC}"
        fi
    fi
    
    # Memory usage
    local memory_info=$(free -h | awk 'NR==2{printf "%s/%s (%.1f%%)", $3, $2, $3*100/$2}')
    echo -e "${BLUE}${INFO} Memory: ${BOLD}$memory_info${NC}"
    
    echo ""
}

# Function to start the service
start_service() {
    echo -e "${ROCKET} ${BOLD}Starting BEV Advanced Avatar Service...${NC}"
    log "Starting service via manager"
    
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}${WARNING} Service is already running${NC}"
        return 0
    fi
    
    # Pre-start checks
    echo -e "${GEAR} Running pre-start validation..."
    if ! "$SCRIPT_DIR/scripts/pre-start-validation.sh" >/dev/null 2>&1; then
        echo -e "${RED}${CROSS} Pre-start validation failed${NC}"
        echo "Check logs: $BEV_ROOT/logs/avatar-prestart.log"
        return 1
    fi
    
    echo -e "${GEAR} Checking GPU..."
    if ! "$SCRIPT_DIR/scripts/gpu-check.sh" >/dev/null 2>&1; then
        echo -e "${RED}${CROSS} GPU check failed${NC}"
        echo "Check logs: $BEV_ROOT/logs/avatar-gpu.log"
        return 1
    fi
    
    echo -e "${GEAR} Checking dependencies..."
    if ! "$SCRIPT_DIR/scripts/dependencies-check.sh" >/dev/null 2>&1; then
        echo -e "${RED}${CROSS} Dependencies check failed${NC}"
        echo "Check logs: $BEV_ROOT/logs/avatar-dependencies.log"
        return 1
    fi
    
    # Start the service
    if sudo systemctl start "$SERVICE_NAME"; then
        echo -e "${GREEN}${CHECK} Service start command sent${NC}"
        
        # Wait for service to be ready
        echo -e "${GEAR} Waiting for service to be ready..."
        local max_wait=60
        local waited=0
        
        while [[ $waited -lt $max_wait ]]; do
            if curl -s -f http://localhost:8080/health >/dev/null 2>&1; then
                echo -e "${GREEN}${CHECK} Service is ready and responding${NC}"
                log "Service started successfully"
                return 0
            fi
            sleep 2
            waited=$((waited + 2))
            echo -n "."
        done
        
        echo ""
        echo -e "${YELLOW}${WARNING} Service started but endpoint not responding after ${max_wait}s${NC}"
        return 1
    else
        echo -e "${RED}${CROSS} Failed to start service${NC}"
        log "Service start failed"
        return 1
    fi
}

# Function to stop the service
stop_service() {
    echo -e "${GEAR} ${BOLD}Stopping BEV Advanced Avatar Service...${NC}"
    log "Stopping service via manager"
    
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
        echo -e "${YELLOW}${WARNING} Service is not running${NC}"
        return 0
    fi
    
    # Graceful stop
    if sudo systemctl stop "$SERVICE_NAME"; then
        echo -e "${GREEN}${CHECK} Service stopped successfully${NC}"
        log "Service stopped successfully"
        return 0
    else
        echo -e "${RED}${CROSS} Failed to stop service${NC}"
        log "Service stop failed"
        return 1
    fi
}

# Function to restart the service
restart_service() {
    echo -e "${GEAR} ${BOLD}Restarting BEV Advanced Avatar Service...${NC}"
    log "Restarting service via manager"
    
    if sudo systemctl restart "$SERVICE_NAME"; then
        echo -e "${GREEN}${CHECK} Service restart command sent${NC}"
        
        # Wait for service to be ready
        echo -e "${GEAR} Waiting for service to be ready..."
        local max_wait=60
        local waited=0
        
        while [[ $waited -lt $max_wait ]]; do
            if curl -s -f http://localhost:8080/health >/dev/null 2>&1; then
                echo -e "${GREEN}${CHECK} Service restarted and responding${NC}"
                log "Service restarted successfully"
                return 0
            fi
            sleep 2
            waited=$((waited + 2))
            echo -n "."
        done
        
        echo ""
        echo -e "${YELLOW}${WARNING} Service restarted but endpoint not responding after ${max_wait}s${NC}"
        return 1
    else
        echo -e "${RED}${CROSS} Failed to restart service${NC}"
        log "Service restart failed"
        return 1
    fi
}

# Function to enable auto-start
enable_service() {
    echo -e "${GEAR} ${BOLD}Enabling auto-start for service...${NC}"
    log "Enabling service auto-start"
    
    if sudo systemctl enable "$SERVICE_NAME"; then
        echo -e "${GREEN}${CHECK} Auto-start enabled${NC}"
        log "Auto-start enabled successfully"
        return 0
    else
        echo -e "${RED}${CROSS} Failed to enable auto-start${NC}"
        log "Failed to enable auto-start"
        return 1
    fi
}

# Function to disable auto-start
disable_service() {
    echo -e "${GEAR} ${BOLD}Disabling auto-start for service...${NC}"
    log "Disabling service auto-start"
    
    if sudo systemctl disable "$SERVICE_NAME"; then
        echo -e "${GREEN}${CHECK} Auto-start disabled${NC}"
        log "Auto-start disabled successfully"
        return 0
    else
        echo -e "${RED}${CROSS} Failed to disable auto-start${NC}"
        log "Failed to disable auto-start"
        return 1
    fi
}

# Function to show logs
show_logs() {
    local lines=${1:-50}
    echo -e "${BOLD}=== Recent Service Logs (last $lines lines) ===${NC}"
    echo ""
    
    if [[ "$lines" == "follow" ]]; then
        echo -e "${INFO} Following logs (Ctrl+C to stop)..."
        journalctl -u "$SERVICE_NAME" -f --no-pager
    else
        journalctl -u "$SERVICE_NAME" -n "$lines" --no-pager
    fi
}

# Function to run diagnostics
run_diagnostics() {
    echo -e "${BOLD}=== BEV Avatar Service Diagnostics ===${NC}"
    echo ""
    
    # Service status
    display_status
    
    # Check configuration files
    echo -e "${BOLD}Configuration Files:${NC}"
    local config_files=(
        "/etc/systemd/system/$SERVICE_NAME.service"
        "$BEV_ROOT/config/avatar.yaml"
        "$BEV_ROOT/systemd/scripts/start-avatar.sh"
    )
    
    for file in "${config_files[@]}"; do
        if [[ -f "$file" ]]; then
            echo -e "${GREEN}${CHECK} $file${NC}"
        else
            echo -e "${RED}${CROSS} $file (missing)${NC}"
        fi
    done
    echo ""
    
    # Dependencies
    echo -e "${BOLD}Dependencies:${NC}"
    local deps=("python3" "nvidia-smi" "redis-cli" "curl" "systemctl")
    for dep in "${deps[@]}"; do
        if command -v "$dep" &>/dev/null; then
            echo -e "${GREEN}${CHECK} $dep${NC}"
        else
            echo -e "${RED}${CROSS} $dep (missing)${NC}"
        fi
    done
    echo ""
    
    # Network connectivity
    echo -e "${BOLD}Network Connectivity:${NC}"
    if timeout 5 redis-cli -h localhost -p 6379 ping &>/dev/null; then
        echo -e "${GREEN}${CHECK} Redis (localhost:6379)${NC}"
    else
        echo -e "${RED}${CROSS} Redis (localhost:6379)${NC}"
    fi
    
    if curl -s -f http://localhost:8080/health >/dev/null 2>&1; then
        echo -e "${GREEN}${CHECK} Avatar endpoint (localhost:8080)${NC}"
    else
        echo -e "${RED}${CROSS} Avatar endpoint (localhost:8080)${NC}"
    fi
    echo ""
    
    # Recent errors
    echo -e "${BOLD}Recent Errors:${NC}"
    local error_count=$(journalctl -u "$SERVICE_NAME" --since "1 hour ago" -p err --no-pager | wc -l)
    if [[ $error_count -eq 0 ]]; then
        echo -e "${GREEN}${CHECK} No errors in the last hour${NC}"
    else
        echo -e "${YELLOW}${WARNING} $error_count errors in the last hour${NC}"
        echo "Use '$0 logs' to view detailed logs"
    fi
    echo ""
}

# Function to run health check
run_health_check() {
    echo -e "${HEALTH} ${BOLD}Running health check...${NC}"
    
    if [[ -f "$SCRIPT_DIR/monitor-avatar-health.sh" ]]; then
        "$SCRIPT_DIR/monitor-avatar-health.sh" once
        echo ""
        echo "Use '$0 health status' for detailed health information"
    else
        echo -e "${RED}${CROSS} Health monitoring script not found${NC}"
        return 1
    fi
}

# Function to show health status
show_health_status() {
    if [[ -f "$SCRIPT_DIR/monitor-avatar-health.sh" ]]; then
        "$SCRIPT_DIR/monitor-avatar-health.sh" status
    else
        echo -e "${RED}${CROSS} Health monitoring script not found${NC}"
        return 1
    fi
}

# Function to start health monitoring
start_health_monitoring() {
    echo -e "${HEALTH} ${BOLD}Starting health monitoring...${NC}"
    
    if [[ -f "$SCRIPT_DIR/monitor-avatar-health.sh" ]]; then
        echo "Health monitoring will run in the background"
        echo "Use 'pkill -f monitor-avatar-health' to stop"
        echo ""
        nohup "$SCRIPT_DIR/monitor-avatar-health.sh" monitor > "$BEV_ROOT/logs/health-monitor-background.log" 2>&1 &
        echo "Health monitoring started with PID: $!"
    else
        echo -e "${RED}${CROSS} Health monitoring script not found${NC}"
        return 1
    fi
}

# Function to display help
show_help() {
    echo -e "${BOLD}BEV Advanced Avatar Service Manager${NC}"
    echo ""
    echo -e "${BOLD}Usage:${NC} $0 {command} [options]"
    echo ""
    echo -e "${BOLD}Service Control:${NC}"
    echo "  start               Start the avatar service"
    echo "  stop                Stop the avatar service"
    echo "  restart             Restart the avatar service"
    echo "  status              Show service status"
    echo "  enable              Enable auto-start on boot"
    echo "  disable             Disable auto-start on boot"
    echo ""
    echo -e "${BOLD}Monitoring:${NC}"
    echo "  logs [lines]        Show recent logs (default: 50)"
    echo "  logs follow         Follow logs in real-time"
    echo "  diagnostics         Run comprehensive diagnostics"
    echo "  health              Run health check once"
    echo "  health status       Show current health status"
    echo "  health monitor      Start continuous health monitoring"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  $0 start"
    echo "  $0 logs 100"
    echo "  $0 logs follow"
    echo "  $0 health status"
    echo ""
}

# Main function
main() {
    local command=${1:-}
    local option=${2:-}
    
    case "$command" in
        "start")
            start_service
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            restart_service
            ;;
        "status")
            display_status
            ;;
        "enable")
            enable_service
            ;;
        "disable")
            disable_service
            ;;
        "logs")
            if [[ "$option" == "follow" ]]; then
                show_logs "follow"
            else
                show_logs "${option:-50}"
            fi
            ;;
        "diagnostics")
            run_diagnostics
            ;;
        "health")
            case "$option" in
                "status")
                    show_health_status
                    ;;
                "monitor")
                    start_health_monitoring
                    ;;
                *)
                    run_health_check
                    ;;
            esac
            ;;
        *)
            show_help
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"