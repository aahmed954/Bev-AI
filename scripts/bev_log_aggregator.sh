#!/bin/bash

##############################################################################
# BEV OSINT Framework - Log Aggregator Script
# Centralized log collection and analysis for multi-node deployment
##############################################################################

set -euo pipefail

# Color codes
declare -r RED='\033[0;31m'
declare -r GREEN='\033[0;32m'
declare -r YELLOW='\033[1;33m'
declare -r BLUE='\033[0;34m'
declare -r PURPLE='\033[0;35m'
declare -r CYAN='\033[0;36m'
declare -r WHITE='\033[1;37m'
declare -r NC='\033[0m'

# Configuration
declare -r BEV_HOME="/home/starlord/Projects/Bev"
declare -r LOG_DIR="$BEV_HOME/logs"
declare -r AGGREGATED_DIR="$LOG_DIR/aggregated"
declare -r THANOS_HOST="localhost"
declare -r ORACLE1_HOST="100.96.197.84"
declare -r ORACLE1_USER="starlord"
declare -r SSH_KEY="$HOME/.ssh/bev_deployment_key"

# Log levels
declare -ra LOG_LEVELS=("ERROR" "WARN" "INFO" "DEBUG")

##############################################################################
# Utility Functions
##############################################################################
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                      BEV OSINT Log Aggregator                           ║${NC}"
    echo -e "${CYAN}║                        $(date '+%Y-%m-%d %H:%M:%S %Z')                        ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════╝${NC}\n"
}

##############################################################################
# Log Collection Functions
##############################################################################
setup_log_directories() {
    mkdir -p "$AGGREGATED_DIR"
    mkdir -p "$AGGREGATED_DIR/thanos"
    mkdir -p "$AGGREGATED_DIR/oracle1"
    mkdir -p "$AGGREGATED_DIR/analysis"
}

collect_thanos_logs() {
    local output_dir="$AGGREGATED_DIR/thanos"
    local timestamp=$(date +%Y%m%d_%H%M%S)

    log_info "Collecting THANOS logs..."

    cd "$BEV_HOME"

    # Get list of running containers
    local containers=$(docker-compose -f docker-compose-thanos-unified.yml ps --services --filter "status=running" 2>/dev/null || true)

    if [[ -z "$containers" ]]; then
        log_warning "No running containers found on THANOS"
        return 1
    fi

    # Collect logs from each container
    while IFS= read -r container; do
        if [[ -n "$container" ]]; then
            log_info "Collecting logs for: $container"

            # Get container logs
            docker-compose -f docker-compose-thanos-unified.yml logs --no-color "$container" > "$output_dir/${container}_${timestamp}.log" 2>/dev/null || {
                log_warning "Failed to collect logs for $container"
                continue
            }

            # Get container stats
            docker stats --no-stream --format "json" "$container" > "$output_dir/${container}_stats_${timestamp}.json" 2>/dev/null || true
        fi
    done <<< "$containers"

    # Collect system logs
    log_info "Collecting system logs..."
    journalctl -u docker --since "1 hour ago" --no-pager > "$output_dir/docker_system_${timestamp}.log" 2>/dev/null || true

    log_success "THANOS logs collected to: $output_dir"
}

collect_oracle1_logs() {
    local output_dir="$AGGREGATED_DIR/oracle1"
    local timestamp=$(date +%Y%m%d_%H%M%S)

    log_info "Collecting ORACLE1 logs..."

    # Test connectivity
    if ! ssh -o ConnectTimeout=5 -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" "echo 'connected'" &>/dev/null; then
        log_error "Cannot connect to ORACLE1"
        return 1
    fi

    # Create remote log collection script
    local remote_script=$(cat << 'EOF'
#!/bin/bash
cd ~/bev
timestamp=$(date +%Y%m%d_%H%M%S)
containers=$(docker-compose -f docker-compose-oracle1-unified.yml ps --services --filter "status=running" 2>/dev/null || true)

mkdir -p ~/bev_logs

if [[ -n "$containers" ]]; then
    while IFS= read -r container; do
        if [[ -n "$container" ]]; then
            echo "Collecting logs for: $container"
            docker-compose -f docker-compose-oracle1-unified.yml logs --no-color "$container" > ~/bev_logs/${container}_${timestamp}.log 2>/dev/null || true
            docker stats --no-stream --format "json" "$container" > ~/bev_logs/${container}_stats_${timestamp}.json 2>/dev/null || true
        fi
    done <<< "$containers"
fi

# System information
free -h > ~/bev_logs/memory_${timestamp}.txt
df -h > ~/bev_logs/disk_${timestamp}.txt
uptime > ~/bev_logs/uptime_${timestamp}.txt

echo "Log collection completed"
EOF
)

    # Execute log collection on ORACLE1
    ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" "$remote_script"

    # Download collected logs
    log_info "Downloading logs from ORACLE1..."
    scp -r -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST:~/bev_logs/*" "$output_dir/" 2>/dev/null || {
        log_warning "Some logs could not be downloaded from ORACLE1"
    }

    # Cleanup remote logs
    ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" "rm -rf ~/bev_logs" 2>/dev/null || true

    log_success "ORACLE1 logs collected to: $output_dir"
}

##############################################################################
# Log Analysis Functions
##############################################################################
analyze_error_patterns() {
    local analysis_dir="$AGGREGATED_DIR/analysis"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local error_report="$analysis_dir/error_analysis_${timestamp}.txt"

    log_info "Analyzing error patterns..."

    {
        echo "BEV OSINT Framework - Error Analysis Report"
        echo "Generated: $(date)"
        echo "=========================================="
        echo ""

        # Analyze THANOS errors
        echo "THANOS Error Analysis:"
        echo "----------------------"
        if find "$AGGREGATED_DIR/thanos" -name "*.log" -type f | head -1 > /dev/null 2>&1; then
            find "$AGGREGATED_DIR/thanos" -name "*.log" -type f -exec grep -l -i "error\|exception\|fail\|critical" {} \; | while read -r logfile; do
                echo "File: $(basename "$logfile")"
                grep -i "error\|exception\|fail\|critical" "$logfile" | head -10
                echo ""
            done
        else
            echo "No THANOS log files found"
        fi

        echo ""

        # Analyze ORACLE1 errors
        echo "ORACLE1 Error Analysis:"
        echo "-----------------------"
        if find "$AGGREGATED_DIR/oracle1" -name "*.log" -type f | head -1 > /dev/null 2>&1; then
            find "$AGGREGATED_DIR/oracle1" -name "*.log" -type f -exec grep -l -i "error\|exception\|fail\|critical" {} \; | while read -r logfile; do
                echo "File: $(basename "$logfile")"
                grep -i "error\|exception\|fail\|critical" "$logfile" | head -10
                echo ""
            done
        else
            echo "No ORACLE1 log files found"
        fi

    } > "$error_report"

    log_success "Error analysis saved to: $error_report"
}

analyze_performance_metrics() {
    local analysis_dir="$AGGREGATED_DIR/analysis"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local perf_report="$analysis_dir/performance_analysis_${timestamp}.txt"

    log_info "Analyzing performance metrics..."

    {
        echo "BEV OSINT Framework - Performance Analysis Report"
        echo "Generated: $(date)"
        echo "================================================="
        echo ""

        # Analyze container stats
        echo "Container Resource Usage:"
        echo "-------------------------"

        # THANOS stats
        if find "$AGGREGATED_DIR/thanos" -name "*_stats_*.json" -type f | head -1 > /dev/null 2>&1; then
            echo "THANOS Containers:"
            find "$AGGREGATED_DIR/thanos" -name "*_stats_*.json" -type f | while read -r statsfile; do
                local container_name=$(basename "$statsfile" | sed 's/_stats_.*\.json$//')
                echo "  $container_name:"
                if command -v jq >/dev/null 2>&1; then
                    jq -r '"    CPU: " + .CPUPerc + " | Memory: " + .MemUsage' "$statsfile" 2>/dev/null || echo "    Stats parsing failed"
                else
                    echo "    Raw data: $(cat "$statsfile")"
                fi
                echo ""
            done
        fi

        # ORACLE1 stats
        if find "$AGGREGATED_DIR/oracle1" -name "*_stats_*.json" -type f | head -1 > /dev/null 2>&1; then
            echo "ORACLE1 Containers:"
            find "$AGGREGATED_DIR/oracle1" -name "*_stats_*.json" -type f | while read -r statsfile; do
                local container_name=$(basename "$statsfile" | sed 's/_stats_.*\.json$//')
                echo "  $container_name:"
                if command -v jq >/dev/null 2>&1; then
                    jq -r '"    CPU: " + .CPUPerc + " | Memory: " + .MemUsage' "$statsfile" 2>/dev/null || echo "    Stats parsing failed"
                else
                    echo "    Raw data: $(cat "$statsfile")"
                fi
                echo ""
            done
        fi

        # System resource analysis
        echo "System Resources:"
        echo "----------------"
        if [[ -f "$AGGREGATED_DIR/oracle1/memory_"*".txt" ]]; then
            echo "ORACLE1 Memory:"
            cat "$AGGREGATED_DIR/oracle1/memory_"*".txt" | head -2
            echo ""
        fi

        if [[ -f "$AGGREGATED_DIR/oracle1/disk_"*".txt" ]]; then
            echo "ORACLE1 Disk:"
            cat "$AGGREGATED_DIR/oracle1/disk_"*".txt" | head -5
            echo ""
        fi

    } > "$perf_report"

    log_success "Performance analysis saved to: $perf_report"
}

generate_health_summary() {
    local analysis_dir="$AGGREGATED_DIR/analysis"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local health_report="$analysis_dir/health_summary_${timestamp}.txt"

    log_info "Generating health summary..."

    {
        echo "BEV OSINT Framework - Health Summary"
        echo "Generated: $(date)"
        echo "==================================="
        echo ""

        # Count log files and errors
        local thanos_logs=$(find "$AGGREGATED_DIR/thanos" -name "*.log" -type f | wc -l)
        local oracle1_logs=$(find "$AGGREGATED_DIR/oracle1" -name "*.log" -type f | wc -l)

        echo "Log Collection Summary:"
        echo "  THANOS log files: $thanos_logs"
        echo "  ORACLE1 log files: $oracle1_logs"
        echo ""

        # Error summary
        echo "Error Summary:"
        local total_errors=0
        if [[ $thanos_logs -gt 0 ]]; then
            local thanos_errors=$(find "$AGGREGATED_DIR/thanos" -name "*.log" -type f -exec grep -c -i "error\|exception\|fail\|critical" {} \; | awk '{sum+=$1} END {print sum+0}')
            echo "  THANOS errors: $thanos_errors"
            total_errors=$((total_errors + thanos_errors))
        fi

        if [[ $oracle1_logs -gt 0 ]]; then
            local oracle1_errors=$(find "$AGGREGATED_DIR/oracle1" -name "*.log" -type f -exec grep -c -i "error\|exception\|fail\|critical" {} \; | awk '{sum+=$1} END {print sum+0}')
            echo "  ORACLE1 errors: $oracle1_errors"
            total_errors=$((total_errors + oracle1_errors))
        fi

        echo "  Total errors: $total_errors"
        echo ""

        # Health assessment
        echo "Health Assessment:"
        if [[ $total_errors -eq 0 ]]; then
            echo "  Status: HEALTHY ✓"
        elif [[ $total_errors -lt 10 ]]; then
            echo "  Status: WARNING ⚠ (Minor issues detected)"
        else
            echo "  Status: CRITICAL ✗ (Multiple errors detected)"
        fi

        echo ""
        echo "Recommendations:"
        if [[ $total_errors -gt 0 ]]; then
            echo "  - Review error analysis report for detailed error information"
            echo "  - Check service status with bev_status_monitor.sh"
            echo "  - Consider restarting failing services"
        else
            echo "  - System appears healthy"
            echo "  - Continue monitoring for any changes"
        fi

    } > "$health_report"

    log_success "Health summary saved to: $health_report"
}

##############################################################################
# Real-time Log Monitoring
##############################################################################
tail_service_logs() {
    local service_name="$1"
    local node="${2:-thanos}"

    if [[ "$node" == "thanos" ]]; then
        log_info "Tailing logs for $service_name on THANOS..."
        cd "$BEV_HOME"
        docker-compose -f docker-compose-thanos-unified.yml logs -f "$service_name"
    else
        log_info "Tailing logs for $service_name on ORACLE1..."
        ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
            "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml logs -f $service_name"
    fi
}

follow_all_logs() {
    local node="${1:-thanos}"

    if [[ "$node" == "thanos" ]]; then
        log_info "Following all logs on THANOS..."
        cd "$BEV_HOME"
        docker-compose -f docker-compose-thanos-unified.yml logs -f
    else
        log_info "Following all logs on ORACLE1..."
        ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
            "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml logs -f"
    fi
}

##############################################################################
# Log Search and Filtering
##############################################################################
search_logs() {
    local pattern="$1"
    local timeframe="${2:-1h}"

    log_info "Searching for pattern: '$pattern' in logs from last $timeframe"

    # Search in collected logs
    if [[ -d "$AGGREGATED_DIR" ]]; then
        echo -e "\n${CYAN}Recent Log Entries:${NC}"
        find "$AGGREGATED_DIR" -name "*.log" -type f -newer "$AGGREGATED_DIR" -exec grep -l "$pattern" {} \; | while read -r logfile; do
            echo -e "\n${YELLOW}File: $logfile${NC}"
            grep --color=always -n "$pattern" "$logfile" | tail -20
        done
    fi

    # Search in live containers
    echo -e "\n${CYAN}Live Container Logs:${NC}"
    cd "$BEV_HOME"
    docker-compose -f docker-compose-thanos-unified.yml logs --since="$timeframe" 2>/dev/null | grep --color=always "$pattern" | tail -50
}

##############################################################################
# Command Line Interface
##############################################################################
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Log Collection:"
    echo "  --collect           Collect all logs from both nodes"
    echo "  --collect-thanos    Collect logs from THANOS only"
    echo "  --collect-oracle1   Collect logs from ORACLE1 only"
    echo ""
    echo "Log Analysis:"
    echo "  --analyze           Perform full log analysis"
    echo "  --errors            Analyze error patterns only"
    echo "  --performance       Analyze performance metrics only"
    echo "  --health            Generate health summary"
    echo ""
    echo "Real-time Monitoring:"
    echo "  --tail SERVICE [NODE]    Tail logs for specific service"
    echo "  --follow [NODE]          Follow all logs on node (thanos/oracle1)"
    echo ""
    echo "Search and Filter:"
    echo "  --search PATTERN [TIME]  Search for pattern in logs"
    echo "  --filter LEVEL           Filter by log level (ERROR/WARN/INFO/DEBUG)"
    echo ""
    echo "Utility:"
    echo "  --help, -h          Show this help message"
    echo "  --clean             Clean old aggregated logs"
    echo ""
    echo "Examples:"
    echo "  $0 --collect                    # Collect all logs"
    echo "  $0 --tail postgres thanos      # Tail postgres logs on THANOS"
    echo "  $0 --search 'connection error'  # Search for connection errors"
    echo "  $0 --analyze                    # Full analysis of collected logs"
}

clean_old_logs() {
    log_info "Cleaning old aggregated logs (older than 7 days)..."
    find "$AGGREGATED_DIR" -type f -mtime +7 -delete 2>/dev/null || true
    log_success "Old logs cleaned"
}

##############################################################################
# Main Function
##############################################################################
main() {
    setup_log_directories

    case "${1:-}" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --collect)
            print_header
            collect_thanos_logs
            collect_oracle1_logs
            analyze_error_patterns
            analyze_performance_metrics
            generate_health_summary
            ;;
        --collect-thanos)
            print_header
            collect_thanos_logs
            ;;
        --collect-oracle1)
            print_header
            collect_oracle1_logs
            ;;
        --analyze)
            print_header
            analyze_error_patterns
            analyze_performance_metrics
            generate_health_summary
            ;;
        --errors)
            print_header
            analyze_error_patterns
            ;;
        --performance)
            print_header
            analyze_performance_metrics
            ;;
        --health)
            print_header
            generate_health_summary
            ;;
        --tail)
            if [[ -z "${2:-}" ]]; then
                log_error "Service name required for --tail"
                show_usage
                exit 1
            fi
            tail_service_logs "$2" "${3:-thanos}"
            ;;
        --follow)
            follow_all_logs "${2:-thanos}"
            ;;
        --search)
            if [[ -z "${2:-}" ]]; then
                log_error "Search pattern required"
                show_usage
                exit 1
            fi
            search_logs "$2" "${3:-1h}"
            ;;
        --clean)
            clean_old_logs
            ;;
        "")
            print_header
            log_info "No action specified. Use --help for usage information."
            log_info "Quick start: $0 --collect && $0 --analyze"
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"