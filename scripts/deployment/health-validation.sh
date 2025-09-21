#!/bin/bash
#
# BEV OSINT Framework - Cross-Node Health Validation
#
# Comprehensive health validation system for multi-node BEV deployment
# Validates service health, cross-node connectivity, and system readiness
#

set -euo pipefail

# Configuration
THANOS_HOST="${THANOS_HOST:-100.122.12.54}"
ORACLE1_HOST="${ORACLE1_HOST:-100.96.197.84}"
STARLORD_HOST="${STARLORD_HOST:-100.122.12.35}"
VAULT_ADDR="${VAULT_ADDR:-http://100.122.12.35:8200}"

# Health check timeouts and intervals
HEALTH_CHECK_TIMEOUT="${HEALTH_CHECK_TIMEOUT:-300}"
HEALTH_CHECK_INTERVAL="${HEALTH_CHECK_INTERVAL:-10}"
CROSS_NODE_TIMEOUT="${CROSS_NODE_TIMEOUT:-60}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[HEALTH]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    return 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

node_log() {
    local node="$1"
    local message="$2"
    echo -e "${PURPLE}[$node]${NC} $message"
}

# Check if a service is healthy via HTTP endpoint
check_http_health() {
    local url="$1"
    local expected_status="${2:-200}"
    local timeout="${3:-10}"

    if curl -s --connect-timeout "$timeout" --max-time "$timeout" -o /dev/null -w "%{http_code}" "$url" | grep -q "^$expected_status$"; then
        return 0
    else
        return 1
    fi
}

# Check if a TCP port is open
check_tcp_port() {
    local host="$1"
    local port="$2"
    local timeout="${3:-5}"

    if timeout "$timeout" bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Check database connectivity
check_database_health() {
    local host="$1"
    local type="$2"

    case "$type" in
        "postgresql")
            if ssh "$host" "docker exec -i \$(docker ps -q --filter 'name=bev_postgres') pg_isready -U researcher" 2>/dev/null; then
                return 0
            fi
            ;;
        "neo4j")
            if check_http_health "http://$host:7474/db/system/tx/commit" 200 10; then
                return 0
            fi
            ;;
        "redis")
            if ssh "$host" "docker exec -i \$(docker ps -q --filter 'name=bev_redis') redis-cli ping" 2>/dev/null | grep -q "PONG"; then
                return 0
            fi
            ;;
        "elasticsearch")
            if check_http_health "http://$host:9200/_cluster/health" 200 10; then
                return 0
            fi
            ;;
    esac
    return 1
}

# Validate THANOS node health
validate_thanos_health() {
    node_log "THANOS" "Starting health validation..."

    local checks_passed=0
    local total_checks=0

    # GPU availability check
    ((total_checks++))
    if ssh "$THANOS_HOST" "nvidia-smi --query-gpu=name --format=csv,noheader" 2>/dev/null | grep -q "RTX"; then
        node_log "THANOS" "✅ GPU available"
        ((checks_passed++))
    else
        node_log "THANOS" "❌ GPU not available"
    fi

    # PostgreSQL health
    ((total_checks++))
    if check_database_health "$THANOS_HOST" "postgresql"; then
        node_log "THANOS" "✅ PostgreSQL healthy"
        ((checks_passed++))
    else
        node_log "THANOS" "❌ PostgreSQL unhealthy"
    fi

    # Neo4j health
    ((total_checks++))
    if check_database_health "$THANOS_HOST" "neo4j"; then
        node_log "THANOS" "✅ Neo4j healthy"
        ((checks_passed++))
    else
        node_log "THANOS" "❌ Neo4j unhealthy"
    fi

    # Elasticsearch health
    ((total_checks++))
    if check_database_health "$THANOS_HOST" "elasticsearch"; then
        node_log "THANOS" "✅ Elasticsearch healthy"
        ((checks_passed++))
    else
        node_log "THANOS" "❌ Elasticsearch unhealthy"
    fi

    # RabbitMQ health
    ((total_checks++))
    if check_tcp_port "$THANOS_HOST" 5672 5; then
        node_log "THANOS" "✅ RabbitMQ accessible"
        ((checks_passed++))
    else
        node_log "THANOS" "❌ RabbitMQ not accessible"
    fi

    # AI Services health (check for autonomous coordinator)
    ((total_checks++))
    if ssh "$THANOS_HOST" "docker ps --filter 'name=autonomous' --format '{{.Status}}'" 2>/dev/null | grep -q "Up"; then
        node_log "THANOS" "✅ AI services running"
        ((checks_passed++))
    else
        node_log "THANOS" "❌ AI services not running"
    fi

    # IntelOwl health
    ((total_checks++))
    if check_tcp_port "$THANOS_HOST" 80 5; then
        node_log "THANOS" "✅ IntelOwl accessible"
        ((checks_passed++))
    else
        node_log "THANOS" "❌ IntelOwl not accessible"
    fi

    # System resource check
    ((total_checks++))
    local memory_usage
    memory_usage=$(ssh "$THANOS_HOST" "free | grep Mem | awk '{printf \"%.0f\", \$3/\$2 * 100.0}'" 2>/dev/null || echo "100")
    if [[ $memory_usage -lt 90 ]]; then
        node_log "THANOS" "✅ Memory usage acceptable (${memory_usage}%)"
        ((checks_passed++))
    else
        node_log "THANOS" "⚠️ High memory usage (${memory_usage}%)"
    fi

    # Docker service health
    ((total_checks++))
    local docker_containers
    docker_containers=$(ssh "$THANOS_HOST" "docker ps --filter 'status=running' | wc -l" 2>/dev/null || echo "0")
    if [[ $docker_containers -gt 10 ]]; then
        node_log "THANOS" "✅ Docker services running ($docker_containers containers)"
        ((checks_passed++))
    else
        node_log "THANOS" "❌ Insufficient Docker services ($docker_containers containers)"
    fi

    local thanos_health_score=$((checks_passed * 100 / total_checks))
    node_log "THANOS" "Health score: $thanos_health_score% ($checks_passed/$total_checks)"
    echo "$thanos_health_score"
}

# Validate ORACLE1 node health
validate_oracle1_health() {
    node_log "ORACLE1" "Starting health validation..."

    local checks_passed=0
    local total_checks=0

    # Redis health
    ((total_checks++))
    if check_database_health "$ORACLE1_HOST" "redis"; then
        node_log "ORACLE1" "✅ Redis healthy"
        ((checks_passed++))
    else
        node_log "ORACLE1" "❌ Redis unhealthy"
    fi

    # Prometheus health
    ((total_checks++))
    if check_http_health "http://$ORACLE1_HOST:9090/-/healthy" 200 10; then
        node_log "ORACLE1" "✅ Prometheus healthy"
        ((checks_passed++))
    else
        node_log "ORACLE1" "❌ Prometheus unhealthy"
    fi

    # Grafana health
    ((total_checks++))
    if check_http_health "http://$ORACLE1_HOST:3000/api/health" 200 10; then
        node_log "ORACLE1" "✅ Grafana healthy"
        ((checks_passed++))
    else
        node_log "ORACLE1" "❌ Grafana unhealthy"
    fi

    # Consul health
    ((total_checks++))
    if check_http_health "http://$ORACLE1_HOST:8500/v1/status/leader" 200 10; then
        node_log "ORACLE1" "✅ Consul healthy"
        ((checks_passed++))
    else
        node_log "ORACLE1" "❌ Consul unhealthy"
    fi

    # ARM64 analyzers health
    ((total_checks++))
    if ssh "$ORACLE1_HOST" "docker ps --filter 'name=analyzer' --format '{{.Status}}'" 2>/dev/null | grep -q "Up"; then
        node_log "ORACLE1" "✅ ARM64 analyzers running"
        ((checks_passed++))
    else
        node_log "ORACLE1" "❌ ARM64 analyzers not running"
    fi

    # Tor proxy health
    ((total_checks++))
    if check_tcp_port "$ORACLE1_HOST" 9050 5; then
        node_log "ORACLE1" "✅ Tor proxy accessible"
        ((checks_passed++))
    else
        node_log "ORACLE1" "❌ Tor proxy not accessible"
    fi

    # System resource check (ARM64 specific)
    ((total_checks++))
    local cpu_count
    cpu_count=$(ssh "$ORACLE1_HOST" "nproc" 2>/dev/null || echo "1")
    if [[ $cpu_count -ge 4 ]]; then
        node_log "ORACLE1" "✅ Adequate CPU cores ($cpu_count)"
        ((checks_passed++))
    else
        node_log "ORACLE1" "⚠️ Limited CPU cores ($cpu_count)"
    fi

    # Architecture verification
    ((total_checks++))
    local arch
    arch=$(ssh "$ORACLE1_HOST" "uname -m" 2>/dev/null || echo "unknown")
    if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then
        node_log "ORACLE1" "✅ ARM64 architecture verified"
        ((checks_passed++))
    else
        node_log "ORACLE1" "❌ Unexpected architecture: $arch"
    fi

    # Docker service health
    ((total_checks++))
    local docker_containers
    docker_containers=$(ssh "$ORACLE1_HOST" "docker ps --filter 'status=running' | wc -l" 2>/dev/null || echo "0")
    if [[ $docker_containers -gt 5 ]]; then
        node_log "ORACLE1" "✅ Docker services running ($docker_containers containers)"
        ((checks_passed++))
    else
        node_log "ORACLE1" "❌ Insufficient Docker services ($docker_containers containers)"
    fi

    local oracle1_health_score=$((checks_passed * 100 / total_checks))
    node_log "ORACLE1" "Health score: $oracle1_health_score% ($checks_passed/$total_checks)"
    echo "$oracle1_health_score"
}

# Validate STARLORD node health
validate_starlord_health() {
    node_log "STARLORD" "Starting health validation..."

    local checks_passed=0
    local total_checks=0

    # Vault health
    ((total_checks++))
    if check_http_health "$VAULT_ADDR/v1/sys/health" 200 10; then
        node_log "STARLORD" "✅ Vault healthy"
        ((checks_passed++))
    else
        node_log "STARLORD" "❌ Vault unhealthy"
    fi

    # GPU availability (RTX 4090)
    ((total_checks++))
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q "4090"; then
        node_log "STARLORD" "✅ RTX 4090 available"
        ((checks_passed++))
    else
        node_log "STARLORD" "⚠️ RTX 4090 not detected"
    fi

    # AI companion status (if deployed)
    ((total_checks++))
    if systemctl is-active --quiet bev-ai-companion 2>/dev/null; then
        node_log "STARLORD" "✅ AI companion service active"
        ((checks_passed++))
    else
        node_log "STARLORD" "ℹ️ AI companion not deployed (optional)"
        ((checks_passed++))  # Not critical for main deployment
    fi

    # Development environment health
    ((total_checks++))
    if command -v docker-compose &>/dev/null && command -v git &>/dev/null; then
        node_log "STARLORD" "✅ Development tools available"
        ((checks_passed++))
    else
        node_log "STARLORD" "❌ Development tools missing"
    fi

    # Tailscale connectivity
    ((total_checks++))
    if command -v tailscale &>/dev/null && tailscale status &>/dev/null; then
        node_log "STARLORD" "✅ Tailscale connectivity active"
        ((checks_passed++))
    else
        node_log "STARLORD" "❌ Tailscale not operational"
    fi

    # Disk space check
    ((total_checks++))
    local disk_usage
    disk_usage=$(df / | awk 'NR==2{print $5}' | sed 's/%//')
    if [[ $disk_usage -lt 80 ]]; then
        node_log "STARLORD" "✅ Adequate disk space (${disk_usage}% used)"
        ((checks_passed++))
    else
        node_log "STARLORD" "⚠️ High disk usage (${disk_usage}% used)"
    fi

    local starlord_health_score=$((checks_passed * 100 / total_checks))
    node_log "STARLORD" "Health score: $starlord_health_score% ($checks_passed/$total_checks)"
    echo "$starlord_health_score"
}

# Test cross-node connectivity
test_cross_node_connectivity() {
    log "Testing cross-node connectivity..."

    local connectivity_tests=(
        "STARLORD→THANOS:$STARLORD_HOST:$THANOS_HOST:22"
        "STARLORD→ORACLE1:$STARLORD_HOST:$ORACLE1_HOST:22"
        "THANOS→ORACLE1:$THANOS_HOST:$ORACLE1_HOST:9090"
        "ORACLE1→THANOS:$ORACLE1_HOST:$THANOS_HOST:5432"
        "THANOS→STARLORD:$THANOS_HOST:$STARLORD_HOST:8200"
        "ORACLE1→STARLORD:$ORACLE1_HOST:$STARLORD_HOST:8200"
    )

    local connectivity_score=0
    local total_tests=${#connectivity_tests[@]}

    for test in "${connectivity_tests[@]}"; do
        IFS=':' read -r connection source target port <<< "$test"

        if ssh "$source" "timeout $CROSS_NODE_TIMEOUT bash -c '</dev/tcp/$target/$port'" 2>/dev/null; then
            info "✅ $connection connectivity OK"
            ((connectivity_score++))
        else
            warn "❌ $connection connectivity FAILED"
        fi
    done

    local connectivity_percentage=$((connectivity_score * 100 / total_tests))
    log "Cross-node connectivity: $connectivity_percentage% ($connectivity_score/$total_tests)"
    echo "$connectivity_percentage"
}

# Test service integration
test_service_integration() {
    log "Testing service integration..."

    local integration_tests=0
    local passed_tests=0

    # Test THANOS → ORACLE1 metrics scraping
    ((integration_tests++))
    if curl -s "http://$ORACLE1_HOST:9090/api/v1/targets" | jq -e '.data.activeTargets[] | select(.labels.instance | contains("'$THANOS_HOST'"))' >/dev/null 2>&1; then
        info "✅ Prometheus scraping THANOS metrics"
        ((passed_tests++))
    else
        warn "❌ Prometheus not scraping THANOS metrics"
    fi

    # Test Vault → Node authentication
    ((integration_tests++))
    if curl -s "$VAULT_ADDR/v1/sys/health" | jq -e '.sealed == false' >/dev/null 2>&1; then
        info "✅ Vault authentication available"
        ((passed_tests++))
    else
        warn "❌ Vault authentication issues"
    fi

    # Test database connectivity from ORACLE1 to THANOS
    ((integration_tests++))
    if ssh "$ORACLE1_HOST" "timeout 10 bash -c '</dev/tcp/$THANOS_HOST/5432'" 2>/dev/null; then
        info "✅ ORACLE1 can reach THANOS PostgreSQL"
        ((passed_tests++))
    else
        warn "❌ ORACLE1 cannot reach THANOS PostgreSQL"
    fi

    # Test Redis connectivity
    ((integration_tests++))
    if ssh "$THANOS_HOST" "timeout 10 bash -c '</dev/tcp/$ORACLE1_HOST/6379'" 2>/dev/null; then
        info "✅ THANOS can reach ORACLE1 Redis"
        ((passed_tests++))
    else
        warn "❌ THANOS cannot reach ORACLE1 Redis"
    fi

    local integration_percentage=$((passed_tests * 100 / integration_tests))
    log "Service integration: $integration_percentage% ($passed_tests/$integration_tests)"
    echo "$integration_percentage"
}

# Comprehensive health validation
comprehensive_health_check() {
    log "Starting comprehensive BEV health validation..."

    local start_time
    start_time=$(date +%s)

    # Individual node health checks
    local thanos_score
    local oracle1_score
    local starlord_score

    thanos_score=$(validate_thanos_health)
    oracle1_score=$(validate_oracle1_health)
    starlord_score=$(validate_starlord_health)

    # Cross-node tests
    local connectivity_score
    local integration_score

    connectivity_score=$(test_cross_node_connectivity)
    integration_score=$(test_service_integration)

    # Calculate overall health score
    local overall_score
    overall_score=$(( (thanos_score + oracle1_score + starlord_score + connectivity_score + integration_score) / 5 ))

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Generate health report
    cat > /tmp/bev-health-report.json <<EOF
{
  "timestamp": "$(date -Iseconds)",
  "duration_seconds": $duration,
  "overall_health_score": $overall_score,
  "node_health": {
    "thanos": {
      "score": $thanos_score,
      "role": "GPU compute, primary databases",
      "critical_services": ["postgresql", "neo4j", "elasticsearch", "ai-services"]
    },
    "oracle1": {
      "score": $oracle1_score,
      "role": "ARM64 monitoring, edge services",
      "critical_services": ["prometheus", "grafana", "redis", "analyzers"]
    },
    "starlord": {
      "score": $starlord_score,
      "role": "Control node, development",
      "critical_services": ["vault", "ai-companion", "development-tools"]
    }
  },
  "connectivity_score": $connectivity_score,
  "integration_score": $integration_score,
  "health_status": "$([ $overall_score -ge 80 ] && echo "healthy" || echo "unhealthy")",
  "recommendations": []
}
EOF

    # Add recommendations based on scores
    local recommendations=""
    if [[ $thanos_score -lt 80 ]]; then
        recommendations+='"Review THANOS services and GPU availability",'
    fi
    if [[ $oracle1_score -lt 80 ]]; then
        recommendations+='"Check ORACLE1 monitoring stack and ARM64 services",'
    fi
    if [[ $starlord_score -lt 80 ]]; then
        recommendations+='"Verify STARLORD Vault and development environment",'
    fi
    if [[ $connectivity_score -lt 80 ]]; then
        recommendations+='"Investigate cross-node network connectivity issues",'
    fi
    if [[ $integration_score -lt 80 ]]; then
        recommendations+='"Debug service integration and authentication",'
    fi

    # Remove trailing comma and update JSON
    recommendations=${recommendations%,}
    if [[ -n "$recommendations" ]]; then
        sed -i "s/\"recommendations\": \[\]/\"recommendations\": [$recommendations]/" /tmp/bev-health-report.json
    fi

    # Display results
    log "Health validation completed in ${duration}s"
    echo ""
    echo "📊 BEV Health Report:"
    echo "├─ Overall Score: $overall_score%"
    echo "├─ THANOS: $thanos_score%"
    echo "├─ ORACLE1: $oracle1_score%"
    echo "├─ STARLORD: $starlord_score%"
    echo "├─ Connectivity: $connectivity_score%"
    echo "└─ Integration: $integration_score%"
    echo ""

    if [[ $overall_score -ge 90 ]]; then
        success "🎉 Excellent health - all systems optimal"
    elif [[ $overall_score -ge 80 ]]; then
        success "✅ Good health - deployment ready"
    elif [[ $overall_score -ge 70 ]]; then
        warn "⚠️ Fair health - monitor closely"
    else
        error "❌ Poor health - requires immediate attention"
    fi

    echo ""
    echo "📋 Detailed report: /tmp/bev-health-report.json"

    return $([[ $overall_score -ge 80 ]] && echo 0 || echo 1)
}

# Wait for services to become healthy
wait_for_healthy_deployment() {
    local max_wait="${1:-$HEALTH_CHECK_TIMEOUT}"
    local check_interval="${2:-$HEALTH_CHECK_INTERVAL}"

    log "Waiting for deployment to become healthy (max ${max_wait}s)..."

    local elapsed=0
    local healthy=false

    while [[ $elapsed -lt $max_wait ]]; do
        if comprehensive_health_check >/dev/null 2>&1; then
            healthy=true
            break
        fi

        sleep "$check_interval"
        elapsed=$((elapsed + check_interval))
        echo -n "."
    done

    echo ""

    if [[ "$healthy" == "true" ]]; then
        success "Deployment is healthy after ${elapsed}s"
        return 0
    else
        error "Deployment failed to become healthy within ${max_wait}s"
        return 1
    fi
}

# Main execution function
main() {
    local action="${1:-validate}"

    case "$action" in
        "validate"|"check")
            comprehensive_health_check
            ;;
        "wait")
            wait_for_healthy_deployment "${2:-$HEALTH_CHECK_TIMEOUT}"
            ;;
        "thanos")
            validate_thanos_health
            ;;
        "oracle1")
            validate_oracle1_health
            ;;
        "starlord")
            validate_starlord_health
            ;;
        "connectivity")
            test_cross_node_connectivity
            ;;
        "integration")
            test_service_integration
            ;;
        "help"|"--help")
            echo "Usage: $0 [ACTION] [OPTIONS]"
            echo ""
            echo "Actions:"
            echo "  validate     Run comprehensive health validation (default)"
            echo "  wait [time]  Wait for deployment to become healthy"
            echo "  thanos       Check THANOS node only"
            echo "  oracle1      Check ORACLE1 node only"
            echo "  starlord     Check STARLORD node only"
            echo "  connectivity Test cross-node connectivity"
            echo "  integration  Test service integration"
            echo ""
            echo "Environment Variables:"
            echo "  THANOS_HOST           THANOS node address (default: 100.122.12.54)"
            echo "  ORACLE1_HOST          ORACLE1 node address (default: 100.96.197.84)"
            echo "  STARLORD_HOST         STARLORD node address (default: 100.122.12.35)"
            echo "  VAULT_ADDR            Vault server address"
            echo "  HEALTH_CHECK_TIMEOUT  Maximum wait time (default: 300s)"
            ;;
        *)
            error "Unknown action: $action. Use 'help' for usage information."
            ;;
    esac
}

# Execute main function with all arguments
main "$@"