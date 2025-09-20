#!/bin/bash

##############################################################################
# BEV OSINT Framework - Status Monitor Script
# Real-time monitoring and health checking for multi-node deployment
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
declare -r THANOS_HOST="localhost"
declare -r ORACLE1_HOST="100.96.197.84"
declare -r ORACLE1_USER="starlord"
declare -r SSH_KEY="$HOME/.ssh/bev_deployment_key"

# Status symbols
declare -r STATUS_UP="🟢"
declare -r STATUS_DOWN="🔴"
declare -r STATUS_WARNING="🟡"
declare -r STATUS_UNKNOWN="⚪"

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
    echo -e "${CYAN}║                     BEV OSINT Framework Status Monitor                  ║${NC}"
    echo -e "${CYAN}║                        $(date '+%Y-%m-%d %H:%M:%S %Z')                        ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════════════╝${NC}\n"
}

##############################################################################
# Service Status Checking
##############################################################################
check_service_status() {
    local service_name="$1"
    local compose_file="$2"
    local host="$3"

    cd "$BEV_HOME"

    if [[ "$host" == "localhost" ]]; then
        # Local THANOS check
        if docker-compose -f "$compose_file" ps "$service_name" 2>/dev/null | grep -q "Up"; then
            echo "UP"
        else
            echo "DOWN"
        fi
    else
        # Remote ORACLE1 check
        local status=$(ssh -i "$SSH_KEY" "$ORACLE1_USER@$host" \
            "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml ps $service_name 2>/dev/null | grep -q 'Up' && echo 'UP' || echo 'DOWN'")
        echo "$status"
    fi
}

check_port_status() {
    local host="$1"
    local port="$2"

    if timeout 3 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null; then
        echo "OPEN"
    else
        echo "CLOSED"
    fi
}

get_container_stats() {
    local container_name="$1"
    local host="$2"

    if [[ "$host" == "localhost" ]]; then
        # Local stats
        docker stats --no-stream --format "table {{.CPUPerc}}\t{{.MemUsage}}" "$container_name" 2>/dev/null | tail -n +2
    else
        # Remote stats
        ssh -i "$SSH_KEY" "$ORACLE1_USER@$host" \
            "docker stats --no-stream --format 'table {{.CPUPerc}}\t{{.MemUsage}}' $container_name 2>/dev/null | tail -n +2"
    fi
}

##############################################################################
# Status Display Functions
##############################################################################
display_thanos_status() {
    echo -e "${PURPLE}┌─ THANOS (Main Server) ─────────────────────────────────────────────────┐${NC}"

    # Foundation services
    echo -e "${WHITE}Foundation Services:${NC}"
    local foundation_services=("postgres" "neo4j" "redis" "elasticsearch" "influxdb")

    for service in "${foundation_services[@]}"; do
        local status=$(check_service_status "$service" "docker-compose-thanos-unified.yml" "localhost")
        local icon=$([[ "$status" == "UP" ]] && echo "$STATUS_UP" || echo "$STATUS_DOWN")
        printf "  %-15s %s %s\n" "$service" "$icon" "$status"
    done

    echo ""

    # Monitoring services
    echo -e "${WHITE}Monitoring Services:${NC}"
    local monitoring_services=("prometheus" "grafana" "airflow-webserver")

    for service in "${monitoring_services[@]}"; do
        local status=$(check_service_status "$service" "docker-compose-thanos-unified.yml" "localhost")
        local icon=$([[ "$status" == "UP" ]] && echo "$STATUS_UP" || echo "$STATUS_DOWN")
        printf "  %-15s %s %s\n" "$service" "$icon" "$status"
    done

    echo ""

    # Key endpoints
    echo -e "${WHITE}Key Endpoints:${NC}"
    local endpoints=(
        "PostgreSQL:5432"
        "Redis:6379"
        "Elasticsearch:9200"
        "Neo4j:7474"
        "Grafana:3000"
        "Prometheus:9090"
        "Airflow:8080"
    )

    for endpoint in "${endpoints[@]}"; do
        local service="${endpoint%:*}"
        local port="${endpoint#*:}"
        local status=$(check_port_status "localhost" "$port")
        local icon=$([[ "$status" == "OPEN" ]] && echo "$STATUS_UP" || echo "$STATUS_DOWN")
        printf "  %-15s %s Port %s (%s)\n" "$service" "$icon" "$port" "$status"
    done

    echo -e "${PURPLE}└────────────────────────────────────────────────────────────────────────┘${NC}\n"
}

display_oracle1_status() {
    echo -e "${PURPLE}┌─ ORACLE1 (ARM Cloud Server - $ORACLE1_HOST) ────────────────────────────┐${NC}"

    # Test connectivity first
    if ! ssh -o ConnectTimeout=5 -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" "echo 'connected'" &>/dev/null; then
        echo -e "  ${STATUS_DOWN} ${RED}Unable to connect to ORACLE1${NC}"
        echo -e "${PURPLE}└────────────────────────────────────────────────────────────────────────┘${NC}\n"
        return 1
    fi

    # ARM services
    echo -e "${WHITE}ARM-Optimized Services:${NC}"
    local arm_services=("redis" "n8n" "nginx-proxy" "crawler" "social-media-agent")

    for service in "${arm_services[@]}"; do
        local status=$(check_service_status "$service" "docker-compose-oracle1-unified.yml" "$ORACLE1_HOST")
        local icon=$([[ "$status" == "UP" ]] && echo "$STATUS_UP" || echo "$STATUS_DOWN")
        printf "  %-20s %s %s\n" "$service" "$icon" "$status"
    done

    echo ""

    # Resource usage on ORACLE1
    echo -e "${WHITE}Resource Usage:${NC}"
    local memory_info=$(ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
        "free -h | awk 'NR==2{printf \"Used: %s/%s (%.1f%%)\", \$3, \$2, \$3*100/\$2}'")
    local disk_info=$(ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
        "df -h ~ | awk 'NR==2{printf \"Used: %s/%s (%s)\", \$3, \$2, \$5}'")

    echo "  Memory:      $memory_info"
    echo "  Disk:        $disk_info"

    echo -e "${PURPLE}└────────────────────────────────────────────────────────────────────────┘${NC}\n"
}

display_network_status() {
    echo -e "${PURPLE}┌─ Network Connectivity ─────────────────────────────────────────────────┐${NC}"

    # THANOS to ORACLE1 connectivity
    local thanos_to_oracle1="UNKNOWN"
    if ping -c 1 -W 3 "$ORACLE1_HOST" &>/dev/null; then
        thanos_to_oracle1="CONNECTED"
    else
        thanos_to_oracle1="DISCONNECTED"
    fi

    local icon=$([[ "$thanos_to_oracle1" == "CONNECTED" ]] && echo "$STATUS_UP" || echo "$STATUS_DOWN")
    printf "  %-25s %s %s\n" "THANOS → ORACLE1" "$icon" "$thanos_to_oracle1"

    # ORACLE1 to THANOS connectivity (if Oracle1 is reachable)
    if ssh -o ConnectTimeout=5 -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" "echo 'connected'" &>/dev/null; then
        local oracle1_to_thanos=$(ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
            "ping -c 1 -W 3 $THANOS_HOST &>/dev/null && echo 'CONNECTED' || echo 'DISCONNECTED'")
        local icon=$([[ "$oracle1_to_thanos" == "CONNECTED" ]] && echo "$STATUS_UP" || echo "$STATUS_DOWN")
        printf "  %-25s %s %s\n" "ORACLE1 → THANOS" "$icon" "$oracle1_to_thanos"
    else
        printf "  %-25s %s %s\n" "ORACLE1 → THANOS" "$STATUS_DOWN" "ORACLE1 UNREACHABLE"
    fi

    echo -e "${PURPLE}└────────────────────────────────────────────────────────────────────────┘${NC}\n"
}

display_deployment_summary() {
    echo -e "${PURPLE}┌─ Deployment Summary ───────────────────────────────────────────────────┐${NC}"

    # Count total services
    local thanos_total=$(cd "$BEV_HOME" && docker-compose -f docker-compose-thanos-unified.yml config --services | wc -l)
    local thanos_running=$(cd "$BEV_HOME" && docker-compose -f docker-compose-thanos-unified.yml ps --services --filter "status=running" 2>/dev/null | wc -l)

    local oracle1_running=0
    if ssh -o ConnectTimeout=5 -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" "echo 'connected'" &>/dev/null; then
        oracle1_running=$(ssh -i "$SSH_KEY" "$ORACLE1_USER@$ORACLE1_HOST" \
            "cd ~/bev && docker-compose -f docker-compose-oracle1-unified.yml ps --services --filter 'status=running' 2>/dev/null | wc -l")
    fi

    echo -e "${WHITE}Service Statistics:${NC}"
    printf "  %-20s %d/%d running\n" "THANOS Services:" "$thanos_running" "$thanos_total"
    printf "  %-20s %d running\n" "ORACLE1 Services:" "$oracle1_running"

    # Overall health
    local total_expected=$((thanos_total + 5)) # Estimate Oracle1 services
    local total_running=$((thanos_running + oracle1_running))
    local health_percentage=$((total_running * 100 / total_expected))

    echo ""
    echo -e "${WHITE}Overall Health:${NC}"
    if [[ $health_percentage -ge 90 ]]; then
        echo -e "  ${STATUS_UP} Excellent ($health_percentage%)"
    elif [[ $health_percentage -ge 70 ]]; then
        echo -e "  ${STATUS_WARNING} Good ($health_percentage%)"
    else
        echo -e "  ${STATUS_DOWN} Poor ($health_percentage%)"
    fi

    echo -e "${PURPLE}└────────────────────────────────────────────────────────────────────────┘${NC}\n"
}

##############################################################################
# Real-time Monitoring
##############################################################################
continuous_monitor() {
    local refresh_interval="${1:-5}"

    log_info "Starting continuous monitoring (refresh every ${refresh_interval}s, press Ctrl+C to stop)"
    echo ""

    while true; do
        clear
        print_header
        display_thanos_status
        display_oracle1_status
        display_network_status
        display_deployment_summary

        echo -e "${CYAN}Press Ctrl+C to stop monitoring${NC}"
        sleep "$refresh_interval"
    done
}

##############################################################################
# Specific Health Checks
##############################################################################
check_critical_services() {
    echo -e "${CYAN}Critical Services Health Check${NC}\n"

    local critical_failed=0

    # Database services
    echo -e "${WHITE}Database Services:${NC}"
    local db_services=("postgres" "neo4j" "redis" "elasticsearch")

    for service in "${db_services[@]}"; do
        local status=$(check_service_status "$service" "docker-compose-thanos-unified.yml" "localhost")
        if [[ "$status" == "UP" ]]; then
            log_success "$service is healthy"
        else
            log_error "$service is down"
            ((critical_failed++))
        fi
    done

    echo ""

    # Monitoring services
    echo -e "${WHITE}Monitoring Services:${NC}"
    local monitor_services=("prometheus" "grafana")

    for service in "${monitor_services[@]}"; do
        local status=$(check_service_status "$service" "docker-compose-thanos-unified.yml" "localhost")
        if [[ "$status" == "UP" ]]; then
            log_success "$service is healthy"
        else
            log_error "$service is down"
            ((critical_failed++))
        fi
    done

    echo ""

    if [[ $critical_failed -eq 0 ]]; then
        log_success "All critical services are healthy"
        return 0
    else
        log_error "$critical_failed critical service(s) failed"
        return 1
    fi
}

##############################################################################
# Command Line Interface
##############################################################################
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --status            Show current status (default)"
    echo "  --critical          Check only critical services"
    echo "  --monitor [INTERVAL] Start continuous monitoring (default interval: 5s)"
    echo "  --thanos            Show only THANOS status"
    echo "  --oracle1           Show only ORACLE1 status"
    echo "  --network           Show only network status"
    echo "  --summary           Show only deployment summary"
    echo ""
    echo "Examples:"
    echo "  $0                  # Show full status"
    echo "  $0 --monitor 10     # Monitor with 10-second refresh"
    echo "  $0 --critical       # Check critical services only"
}

##############################################################################
# Main Function
##############################################################################
main() {
    case "${1:-}" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --critical)
            check_critical_services
            exit $?
            ;;
        --monitor)
            local interval="${2:-5}"
            continuous_monitor "$interval"
            ;;
        --thanos)
            print_header
            display_thanos_status
            ;;
        --oracle1)
            print_header
            display_oracle1_status
            ;;
        --network)
            print_header
            display_network_status
            ;;
        --summary)
            print_header
            display_deployment_summary
            ;;
        --status|"")
            print_header
            display_thanos_status
            display_oracle1_status
            display_network_status
            display_deployment_summary
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Trap signals for cleanup
trap 'echo -e "\n${YELLOW}Monitoring stopped${NC}"; exit 0' INT TERM

# Execute main function
main "$@"