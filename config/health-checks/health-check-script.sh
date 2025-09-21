#!/bin/bash

# BEV OSINT Health Check Script for Phase 7-9 Services
# Comprehensive health monitoring and automated recovery

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/var/log/bev-health-checks.log"
ALERT_WEBHOOK="${ALERT_WEBHOOK:-}"
MAX_RETRIES=3
RETRY_DELAY=10

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level="$1"
    shift
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $*" | tee -a "$LOG_FILE"
}

# Send alert function
send_alert() {
    local service="$1"
    local status="$2"
    local message="$3"

    if [ -n "$ALERT_WEBHOOK" ]; then
        curl -s -X POST "$ALERT_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{
                \"service\": \"$service\",
                \"status\": \"$status\",
                \"message\": \"$message\",
                \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
            }" || log "ERROR" "Failed to send alert for $service"
    fi
}

# Health check function
check_service_health() {
    local service_name="$1"
    local health_url="$2"
    local expected_status="${3:-200}"

    log "INFO" "Checking health for $service_name"

    local response_code
    local response_time

    # Perform health check with timeout
    if response_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 "$health_url" 2>/dev/null); then
        response_time=$(curl -s -o /dev/null -w "%{time_total}" --max-time 30 "$health_url" 2>/dev/null)

        if [ "$response_code" = "$expected_status" ]; then
            log "INFO" "✅ $service_name: HEALTHY (${response_code}, ${response_time}s)"
            return 0
        else
            log "ERROR" "❌ $service_name: UNHEALTHY (HTTP $response_code)"
            send_alert "$service_name" "UNHEALTHY" "HTTP response code: $response_code"
            return 1
        fi
    else
        log "ERROR" "❌ $service_name: CONNECTION FAILED"
        send_alert "$service_name" "CONNECTION_FAILED" "Cannot connect to health endpoint"
        return 1
    fi
}

# Metrics check function
check_service_metrics() {
    local service_name="$1"
    local metrics_url="$2"

    log "INFO" "Checking metrics for $service_name"

    if curl -s --max-time 30 "$metrics_url" | grep -q "^# HELP"; then
        log "INFO" "✅ $service_name: METRICS AVAILABLE"
        return 0
    else
        log "WARNING" "⚠️ $service_name: METRICS UNAVAILABLE"
        send_alert "$service_name" "METRICS_UNAVAILABLE" "Metrics endpoint not responding properly"
        return 1
    fi
}

# Readiness check function
check_service_readiness() {
    local service_name="$1"
    local ready_url="$2"

    log "INFO" "Checking readiness for $service_name"

    if check_service_health "$service_name" "$ready_url" "200"; then
        log "INFO" "✅ $service_name: READY"
        return 0
    else
        log "WARNING" "⚠️ $service_name: NOT READY"
        return 1
    fi
}

# Docker container health check
check_container_health() {
    local container_name="$1"

    log "INFO" "Checking Docker container health for $container_name"

    if docker ps --filter "name=$container_name" --filter "status=running" --format "{{.Names}}" | grep -q "^$container_name$"; then
        # Check container health status if health check is configured
        local health_status
        health_status=$(docker inspect --format='{{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}' "$container_name" 2>/dev/null || echo "not-found")

        case "$health_status" in
            "healthy")
                log "INFO" "✅ $container_name: CONTAINER HEALTHY"
                return 0
                ;;
            "unhealthy")
                log "ERROR" "❌ $container_name: CONTAINER UNHEALTHY"
                send_alert "$container_name" "CONTAINER_UNHEALTHY" "Docker health check failed"
                return 1
                ;;
            "starting")
                log "INFO" "🔄 $container_name: CONTAINER STARTING"
                return 0
                ;;
            "no-healthcheck")
                log "INFO" "ℹ️ $container_name: CONTAINER RUNNING (no health check configured)"
                return 0
                ;;
            "not-found")
                log "ERROR" "❌ $container_name: CONTAINER NOT FOUND"
                send_alert "$container_name" "CONTAINER_NOT_FOUND" "Container does not exist"
                return 1
                ;;
        esac
    else
        log "ERROR" "❌ $container_name: CONTAINER NOT RUNNING"
        send_alert "$container_name" "CONTAINER_NOT_RUNNING" "Container is not in running state"
        return 1
    fi
}

# Comprehensive service check
check_service_comprehensive() {
    local service_name="$1"
    local container_name="$2"
    local base_url="$3"

    local health_passed=0
    local metrics_passed=0
    local readiness_passed=0
    local container_passed=0

    echo -e "${BLUE}🔍 Comprehensive check for $service_name${NC}"

    # Container health
    if check_container_health "$container_name"; then
        container_passed=1
    fi

    # Service health
    if check_service_health "$service_name" "$base_url/health"; then
        health_passed=1
    fi

    # Service metrics
    if check_service_metrics "$service_name" "$base_url/metrics"; then
        metrics_passed=1
    fi

    # Service readiness (if endpoint exists)
    if check_service_readiness "$service_name" "$base_url/ready"; then
        readiness_passed=1
    fi

    # Calculate overall health score
    local total_checks=4
    local passed_checks=$((container_passed + health_passed + metrics_passed + readiness_passed))
    local health_percentage=$(( (passed_checks * 100) / total_checks ))

    echo -e "${BLUE}📊 $service_name Health Score: $health_percentage% ($passed_checks/$total_checks)${NC}"

    if [ $health_percentage -ge 75 ]; then
        echo -e "${GREEN}✅ $service_name: OVERALL HEALTHY${NC}"
        return 0
    elif [ $health_percentage -ge 50 ]; then
        echo -e "${YELLOW}⚠️ $service_name: DEGRADED${NC}"
        send_alert "$service_name" "DEGRADED" "Health score: $health_percentage%"
        return 1
    else
        echo -e "${RED}❌ $service_name: CRITICAL${NC}"
        send_alert "$service_name" "CRITICAL" "Health score: $health_percentage%"
        return 2
    fi
}

# Restart service function
restart_service() {
    local container_name="$1"

    log "INFO" "Attempting to restart $container_name"

    if docker restart "$container_name"; then
        log "INFO" "✅ Successfully restarted $container_name"
        sleep 30  # Wait for service to start
        return 0
    else
        log "ERROR" "❌ Failed to restart $container_name"
        return 1
    fi
}

# Automated recovery function
attempt_recovery() {
    local service_name="$1"
    local container_name="$2"
    local base_url="$3"

    log "INFO" "Attempting automated recovery for $service_name"

    # Try restarting the container
    if restart_service "$container_name"; then
        # Wait and recheck
        sleep 60
        if check_service_comprehensive "$service_name" "$container_name" "$base_url"; then
            log "INFO" "✅ Automated recovery successful for $service_name"
            send_alert "$service_name" "RECOVERY_SUCCESSFUL" "Service recovered after restart"
            return 0
        else
            log "ERROR" "❌ Automated recovery failed for $service_name"
            send_alert "$service_name" "RECOVERY_FAILED" "Service still unhealthy after restart"
            return 1
        fi
    else
        log "ERROR" "❌ Could not restart $service_name"
        return 1
    fi
}

# Main health check routine
main() {
    log "INFO" "Starting BEV OSINT health check cycle"

    # Service definitions: name, container, base_url
    declare -a services=(
        "dm-crawler|bev_dm_crawler|http://172.30.0.24:8000"
        "crypto-intel|bev_crypto_intel|http://172.30.0.25:8000"
        "reputation-analyzer|bev_reputation_analyzer|http://172.30.0.26:8000"
        "economics-processor|bev_economics_processor|http://172.30.0.27:8000"
        "tactical-intel|bev_tactical_intel|http://172.30.0.28:8000"
        "defense-automation|bev_defense_automation|http://172.30.0.29:8000"
        "opsec-enforcer|bev_opsec_enforcer|http://172.30.0.30:8000"
        "intel-fusion|bev_intel_fusion|http://172.30.0.31:8000"
        "autonomous-coordinator|bev_autonomous_coordinator|http://172.30.0.32:8000"
        "adaptive-learning|bev_adaptive_learning|http://172.30.0.33:8000"
        "resource-manager|bev_resource_manager|http://172.30.0.34:8000"
        "knowledge-evolution|bev_knowledge_evolution|http://172.30.0.35:8000"
    )

    local total_services=${#services[@]}
    local healthy_services=0
    local degraded_services=0
    local critical_services=0
    local recovery_attempts=0

    # Check each service
    for service_def in "${services[@]}"; do
        IFS='|' read -r service_name container_name base_url <<< "$service_def"

        echo ""
        echo "================================================"
        echo "Checking: $service_name"
        echo "================================================"

        local result
        check_service_comprehensive "$service_name" "$container_name" "$base_url"
        result=$?

        case $result in
            0)
                healthy_services=$((healthy_services + 1))
                ;;
            1)
                degraded_services=$((degraded_services + 1))
                # Attempt recovery for degraded services
                if [ "${AUTO_RECOVERY:-true}" = "true" ]; then
                    echo -e "${YELLOW}🔧 Attempting recovery for degraded service: $service_name${NC}"
                    if attempt_recovery "$service_name" "$container_name" "$base_url"; then
                        healthy_services=$((healthy_services + 1))
                        degraded_services=$((degraded_services - 1))
                        recovery_attempts=$((recovery_attempts + 1))
                    fi
                fi
                ;;
            2)
                critical_services=$((critical_services + 1))
                # Attempt recovery for critical services
                if [ "${AUTO_RECOVERY:-true}" = "true" ]; then
                    echo -e "${RED}🚨 Attempting emergency recovery for critical service: $service_name${NC}"
                    if attempt_recovery "$service_name" "$container_name" "$base_url"; then
                        healthy_services=$((healthy_services + 1))
                        critical_services=$((critical_services - 1))
                        recovery_attempts=$((recovery_attempts + 1))
                    fi
                fi
                ;;
        esac
    done

    # Generate summary report
    echo ""
    echo "================================================"
    echo "HEALTH CHECK SUMMARY"
    echo "================================================"
    echo -e "Total Services: $total_services"
    echo -e "${GREEN}Healthy: $healthy_services${NC}"
    echo -e "${YELLOW}Degraded: $degraded_services${NC}"
    echo -e "${RED}Critical: $critical_services${NC}"
    echo -e "${BLUE}Recovery Attempts: $recovery_attempts${NC}"

    local overall_health_percentage=$(( (healthy_services * 100) / total_services ))
    echo -e "Overall System Health: $overall_health_percentage%"

    # System-wide health assessment
    if [ $overall_health_percentage -ge 90 ]; then
        echo -e "${GREEN}✅ BEV OSINT System: HEALTHY${NC}"
        exit_code=0
    elif [ $overall_health_percentage -ge 70 ]; then
        echo -e "${YELLOW}⚠️ BEV OSINT System: DEGRADED${NC}"
        send_alert "BEV-OSINT-SYSTEM" "DEGRADED" "System health: $overall_health_percentage%"
        exit_code=1
    else
        echo -e "${RED}❌ BEV OSINT System: CRITICAL${NC}"
        send_alert "BEV-OSINT-SYSTEM" "CRITICAL" "System health: $overall_health_percentage%"
        exit_code=2
    fi

    log "INFO" "Health check cycle completed. Overall health: $overall_health_percentage%"

    exit $exit_code
}

# Script entry point
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Create log directory if it doesn't exist
    mkdir -p "$(dirname "$LOG_FILE")"

    # Run main function
    main "$@"
fi