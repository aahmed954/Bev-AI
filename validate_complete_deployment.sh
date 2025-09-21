#!/bin/bash
# BEV OSINT Framework - Comprehensive Deployment Validation
# Complete system validation and health assessment
# Generated: $(date)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATION_LOG="${SCRIPT_DIR}/validation_$(date +%Y%m%d_%H%M%S).log"
DETAILED_REPORT="${SCRIPT_DIR}/deployment_health_report_$(date +%Y%m%d_%H%M%S).html"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

# Test counters
TOTAL_VALIDATIONS=0
PASSED_VALIDATIONS=0
FAILED_VALIDATIONS=0
WARNING_VALIDATIONS=0

# Validation categories
declare -A CATEGORY_RESULTS=(
    ["Infrastructure"]=0
    ["Databases"]=0
    ["Security"]=0
    ["Monitoring"]=0
    ["Integration"]=0
    ["Performance"]=0
)

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$VALIDATION_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$VALIDATION_LOG"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$VALIDATION_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$VALIDATION_LOG"
}

log_header() {
    echo -e "${BOLD}${PURPLE}$1${NC}" | tee -a "$VALIDATION_LOG"
    echo "============================================" | tee -a "$VALIDATION_LOG"
}

# Validation tracking
start_validation() {
    ((TOTAL_VALIDATIONS++))
    echo -e "${BLUE}[VALIDATE]${NC} $1" | tee -a "$VALIDATION_LOG"
}

pass_validation() {
    ((PASSED_VALIDATIONS++))
    log_success "✓ PASS: $1"
}

warn_validation() {
    ((WARNING_VALIDATIONS++))
    log_warning "⚠ WARN: $1"
}

fail_validation() {
    ((FAILED_VALIDATIONS++))
    log_error "✗ FAIL: $1"
}

# Infrastructure validation
validate_infrastructure() {
    log_header "🏗️ INFRASTRUCTURE VALIDATION"

    # Docker daemon
    start_validation "Docker daemon status"
    if docker info &>/dev/null; then
        pass_validation "Docker daemon is running"
    else
        fail_validation "Docker daemon is not accessible"
    fi

    # Docker Compose availability
    start_validation "Docker Compose functionality"
    if docker-compose --version &>/dev/null; then
        pass_validation "Docker Compose is functional"
    else
        fail_validation "Docker Compose is not working"
    fi

    # System resources
    start_validation "System resource adequacy"
    local total_memory=$(free -g | awk '/^Mem:/{print $2}')
    local available_disk=$(df "$SCRIPT_DIR" | awk 'NR==2{print int($4/1000000)}')

    if (( total_memory >= 16 )); then
        pass_validation "Memory: ${total_memory}GB (adequate)"
    elif (( total_memory >= 8 )); then
        warn_validation "Memory: ${total_memory}GB (minimal)"
    else
        fail_validation "Memory: ${total_memory}GB (insufficient)"
    fi

    if (( available_disk >= 100 )); then
        pass_validation "Disk space: ${available_disk}GB (adequate)"
    elif (( available_disk >= 50 )); then
        warn_validation "Disk space: ${available_disk}GB (minimal)"
    else
        fail_validation "Disk space: ${available_disk}GB (insufficient)"
    fi

    # Network interfaces
    start_validation "Network interface availability"
    if ip addr show | grep -q "inet "; then
        pass_validation "Network interfaces are configured"
    else
        fail_validation "No network interfaces found"
    fi

    CATEGORY_RESULTS["Infrastructure"]=$((PASSED_VALIDATIONS))
}

# Database validation
validate_databases() {
    log_header "🗄️ DATABASE VALIDATION"

    # PostgreSQL validation
    start_validation "PostgreSQL availability"
    if timeout 10 bash -c "</dev/tcp/localhost/5432" 2>/dev/null || timeout 10 bash -c "</dev/tcp/localhost/5433" 2>/dev/null; then
        pass_validation "PostgreSQL is accessible"

        # Test connection if psql is available
        if command -v psql &>/dev/null; then
            start_validation "PostgreSQL connection test"
            if PGPASSWORD="testpass123" psql -h localhost -p 5435 -U bev -d osint -c "SELECT 1;" &>/dev/null; then
                pass_validation "PostgreSQL connection successful"
            else
                warn_validation "PostgreSQL connection failed"
            fi
        fi
    else
        fail_validation "PostgreSQL is not accessible"
    fi

    # Redis validation
    start_validation "Redis availability"
    if timeout 10 bash -c "</dev/tcp/localhost/6379" 2>/dev/null || timeout 10 bash -c "</dev/tcp/localhost/6380" 2>/dev/null; then
        pass_validation "Redis is accessible"

        # Test Redis if redis-cli is available
        if command -v redis-cli &>/dev/null; then
            start_validation "Redis connection test"
            if redis-cli -p 6380 -a testpass123 ping 2>/dev/null | grep -q "PONG"; then
                pass_validation "Redis connection successful"
            else
                warn_validation "Redis connection failed"
            fi
        fi
    else
        fail_validation "Redis is not accessible"
    fi

    # Neo4j validation
    start_validation "Neo4j availability"
    if timeout 10 bash -c "</dev/tcp/localhost/7474" 2>/dev/null || timeout 10 bash -c "</dev/tcp/localhost/7475" 2>/dev/null; then
        pass_validation "Neo4j HTTP interface is accessible"
    else
        fail_validation "Neo4j is not accessible"
    fi

    if timeout 10 bash -c "</dev/tcp/localhost/7687" 2>/dev/null || timeout 10 bash -c "</dev/tcp/localhost/7688" 2>/dev/null; then
        pass_validation "Neo4j Bolt interface is accessible"
    else
        fail_validation "Neo4j Bolt interface is not accessible"
    fi

    # Elasticsearch validation
    start_validation "Elasticsearch availability"
    if timeout 10 bash -c "</dev/tcp/localhost/9200" 2>/dev/null; then
        pass_validation "Elasticsearch is accessible"

        start_validation "Elasticsearch cluster health"
        if curl -s --max-time 10 "http://localhost:9200/_cluster/health" | grep -q "green\|yellow"; then
            pass_validation "Elasticsearch cluster is healthy"
        else
            warn_validation "Elasticsearch cluster health unknown"
        fi
    else
        warn_validation "Elasticsearch is not accessible"
    fi

    CATEGORY_RESULTS["Databases"]=$((PASSED_VALIDATIONS - CATEGORY_RESULTS["Infrastructure"]))
}

# Security validation
validate_security() {
    log_header "🔒 SECURITY VALIDATION"

    # Vault availability
    start_validation "Vault availability"
    if timeout 10 bash -c "</dev/tcp/localhost/8200" 2>/dev/null; then
        pass_validation "Vault is accessible"

        start_validation "Vault health status"
        if curl -s --max-time 10 "http://localhost:8200/v1/sys/health" | grep -q "initialized"; then
            pass_validation "Vault is initialized"
        else
            warn_validation "Vault initialization status unknown"
        fi
    else
        warn_validation "Vault is not accessible"
    fi

    # SSL/TLS certificate validation
    start_validation "SSL certificate validity"
    # Check if any services are using SSL
    local ssl_services=()
    if netstat -tlnp 2>/dev/null | grep -q ":443"; then
        ssl_services+=("443")
    fi

    if [[ ${#ssl_services[@]} -gt 0 ]]; then
        warn_validation "SSL services detected - manual certificate validation recommended"
    else
        pass_validation "No SSL services detected (development mode)"
    fi

    # Environment file security
    start_validation "Environment file security"
    local env_files=(".env" ".env.thanos.complete" ".env.oracle1.complete")
    local secure_files=0

    for env_file in "${env_files[@]}"; do
        if [[ -f "$SCRIPT_DIR/$env_file" ]]; then
            local permissions=$(stat -c %a "$SCRIPT_DIR/$env_file" 2>/dev/null)
            if [[ "$permissions" =~ ^[67][04][04]$ ]]; then
                ((secure_files++))
            fi
        fi
    done

    if (( secure_files == ${#env_files[@]} )); then
        pass_validation "Environment files have secure permissions"
    else
        warn_validation "Some environment files may have insecure permissions"
    fi

    # Port exposure validation
    start_validation "Port exposure security"
    local open_ports=$(netstat -tlnp 2>/dev/null | grep ":.*:" | wc -l)
    if (( open_ports < 20 )); then
        pass_validation "Port exposure is reasonable ($open_ports ports)"
    else
        warn_validation "Many ports are exposed ($open_ports ports) - review for security"
    fi

    CATEGORY_RESULTS["Security"]=$((PASSED_VALIDATIONS - CATEGORY_RESULTS["Infrastructure"] - CATEGORY_RESULTS["Databases"]))
}

# Monitoring validation
validate_monitoring() {
    log_header "📊 MONITORING VALIDATION"

    # Prometheus validation
    start_validation "Prometheus availability"
    if timeout 10 bash -c "</dev/tcp/localhost/9090" 2>/dev/null; then
        pass_validation "THANOS Prometheus is accessible"

        start_validation "Prometheus targets"
        if curl -s --max-time 10 "http://localhost:9090/-/healthy" | grep -q "Prometheus is Healthy"; then
            pass_validation "THANOS Prometheus is healthy"
        else
            warn_validation "THANOS Prometheus health status unknown"
        fi
    else
        warn_validation "THANOS Prometheus is not accessible"
    fi

    # ORACLE1 Prometheus
    start_validation "ORACLE1 Prometheus availability"
    if timeout 10 bash -c "</dev/tcp/localhost/9091" 2>/dev/null; then
        pass_validation "ORACLE1 Prometheus is accessible"
    else
        warn_validation "ORACLE1 Prometheus is not accessible"
    fi

    # Grafana validation
    start_validation "Grafana availability"
    if timeout 10 bash -c "</dev/tcp/localhost/3000" 2>/dev/null; then
        pass_validation "THANOS Grafana is accessible"
    else
        warn_validation "THANOS Grafana is not accessible"
    fi

    start_validation "ORACLE1 Grafana availability"
    if timeout 10 bash -c "</dev/tcp/localhost/3001" 2>/dev/null; then
        pass_validation "ORACLE1 Grafana is accessible"
    else
        warn_validation "ORACLE1 Grafana is not accessible"
    fi

    # InfluxDB validation
    start_validation "InfluxDB availability"
    if timeout 10 bash -c "</dev/tcp/localhost/8086" 2>/dev/null; then
        pass_validation "InfluxDB is accessible"
    else
        warn_validation "InfluxDB is not accessible"
    fi

    # AlertManager validation
    start_validation "AlertManager availability"
    if timeout 10 bash -c "</dev/tcp/localhost/9093" 2>/dev/null; then
        pass_validation "AlertManager is accessible"
    else
        warn_validation "AlertManager is not accessible"
    fi

    CATEGORY_RESULTS["Monitoring"]=$((PASSED_VALIDATIONS - CATEGORY_RESULTS["Infrastructure"] - CATEGORY_RESULTS["Databases"] - CATEGORY_RESULTS["Security"]))
}

# Integration validation
validate_integration() {
    log_header "🔗 INTEGRATION VALIDATION"

    # Container connectivity
    start_validation "Container network connectivity"
    local running_containers=$(docker ps --format "{{.Names}}" | grep -c "bev_" || echo "0")
    if (( running_containers > 5 )); then
        pass_validation "Multiple BEV containers are running ($running_containers)"
    elif (( running_containers > 0 )); then
        warn_validation "Some BEV containers are running ($running_containers)"
    else
        fail_validation "No BEV containers are running"
    fi

    # Service discovery
    start_validation "Service discovery functionality"
    if docker network ls | grep -q "bev"; then
        pass_validation "BEV networks exist"
    else
        warn_validation "No BEV networks found"
    fi

    # Cross-node communication (if applicable)
    start_validation "Cross-node communication readiness"
    if [[ -x "$SCRIPT_DIR/test_cross_node_integration.sh" ]]; then
        pass_validation "Cross-node integration test script is available"
    else
        warn_validation "Cross-node integration test script not found"
    fi

    # API endpoints
    start_validation "API endpoint accessibility"
    local api_endpoints=(
        "3010:MCP API Server"
        "80:IntelOwl Dashboard"
        "8080:Alternative HTTP"
    )

    local accessible_apis=0
    for endpoint_info in "${api_endpoints[@]}"; do
        local port="${endpoint_info%%:*}"
        local service="${endpoint_info##*:}"

        if timeout 5 bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
            ((accessible_apis++))
        fi
    done

    if (( accessible_apis > 0 )); then
        pass_validation "$accessible_apis API endpoints are accessible"
    else
        warn_validation "No API endpoints are accessible"
    fi

    CATEGORY_RESULTS["Integration"]=$((PASSED_VALIDATIONS - CATEGORY_RESULTS["Infrastructure"] - CATEGORY_RESULTS["Databases"] - CATEGORY_RESULTS["Security"] - CATEGORY_RESULTS["Monitoring"]))
}

# Performance validation
validate_performance() {
    log_header "⚡ PERFORMANCE VALIDATION"

    # Container resource usage
    start_validation "Container resource usage"
    if command -v docker &>/dev/null; then
        local high_cpu_containers=$(docker stats --no-stream --format "{{.Container}} {{.CPUPerc}}" | awk '{gsub(/%/, "", $2); if($2+0 > 80) print $1}' | wc -l)
        local high_mem_containers=$(docker stats --no-stream --format "{{.Container}} {{.MemPerc}}" | awk '{gsub(/%/, "", $2); if($2+0 > 80) print $1}' | wc -l)

        if (( high_cpu_containers == 0 && high_mem_containers == 0 )); then
            pass_validation "Container resource usage is healthy"
        else
            warn_validation "Some containers have high resource usage (CPU: $high_cpu_containers, Mem: $high_mem_containers)"
        fi
    else
        warn_validation "Cannot check container resource usage"
    fi

    # Disk I/O performance
    start_validation "Disk I/O performance"
    local disk_usage=$(df "$SCRIPT_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
    if (( disk_usage < 80 )); then
        pass_validation "Disk usage is healthy ($disk_usage%)"
    elif (( disk_usage < 90 )); then
        warn_validation "Disk usage is getting high ($disk_usage%)"
    else
        fail_validation "Disk usage is critical ($disk_usage%)"
    fi

    # Network latency
    start_validation "Network latency"
    local latency=$(ping -c 3 localhost 2>/dev/null | tail -1 | awk -F'/' '{print $5}' | cut -d'.' -f1)
    if [[ -n "$latency" ]] && (( latency < 10 )); then
        pass_validation "Network latency is excellent ($latency ms)"
    elif [[ -n "$latency" ]] && (( latency < 50 )); then
        pass_validation "Network latency is acceptable ($latency ms)"
    else
        warn_validation "Network latency may be high"
    fi

    # Load average
    start_validation "System load average"
    local load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cpu_cores=$(nproc)
    if [[ -n "$load_avg" ]] && (( $(echo "$load_avg < $cpu_cores" | bc -l 2>/dev/null || echo "1") )); then
        pass_validation "System load is healthy ($load_avg on $cpu_cores cores)"
    else
        warn_validation "System load may be high ($load_avg on $cpu_cores cores)"
    fi

    CATEGORY_RESULTS["Performance"]=$((PASSED_VALIDATIONS - CATEGORY_RESULTS["Infrastructure"] - CATEGORY_RESULTS["Databases"] - CATEGORY_RESULTS["Security"] - CATEGORY_RESULTS["Monitoring"] - CATEGORY_RESULTS["Integration"]))
}

# Generate HTML report
generate_html_report() {
    log_info "Generating detailed HTML report..."

    cat > "$DETAILED_REPORT" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEV OSINT Framework - Deployment Health Report</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { color: #2c3e50; margin-bottom: 10px; }
        .header .subtitle { color: #7f8c8d; font-size: 18px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .summary-card { background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }
        .summary-card h3 { margin: 0 0 10px 0; color: #2c3e50; }
        .summary-card .number { font-size: 2em; font-weight: bold; }
        .passed { color: #27ae60; }
        .failed { color: #e74c3c; }
        .warning { color: #f39c12; }
        .category-results { margin-bottom: 30px; }
        .category { background: #f8f9fa; margin-bottom: 15px; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
        .category h4 { margin: 0 0 10px 0; color: #2c3e50; }
        .status-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }
        .status-item { padding: 10px; border-radius: 4px; border-left: 4px solid #bdc3c7; }
        .status-item.pass { background: #d5f4e6; border-left-color: #27ae60; }
        .status-item.fail { background: #ffeaea; border-left-color: #e74c3c; }
        .status-item.warn { background: #fff3cd; border-left-color: #f39c12; }
        .footer { text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ecf0f1; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 BEV OSINT Framework</h1>
            <div class="subtitle">Deployment Health Report</div>
            <div class="subtitle">Generated: $(date)</div>
        </div>

        <div class="summary">
            <div class="summary-card">
                <h3>Total Validations</h3>
                <div class="number">$TOTAL_VALIDATIONS</div>
            </div>
            <div class="summary-card">
                <h3>Passed</h3>
                <div class="number passed">$PASSED_VALIDATIONS</div>
            </div>
            <div class="summary-card">
                <h3>Failed</h3>
                <div class="number failed">$FAILED_VALIDATIONS</div>
            </div>
            <div class="summary-card">
                <h3>Warnings</h3>
                <div class="number warning">$WARNING_VALIDATIONS</div>
            </div>
        </div>

        <div class="category-results">
            <h3>Results by Category</h3>
EOF

    for category in "${!CATEGORY_RESULTS[@]}"; do
        echo "            <div class=\"category\">
                <h4>$category</h4>
                <div>Validations passed: ${CATEGORY_RESULTS[$category]}</div>
            </div>" >> "$DETAILED_REPORT"
    done

    cat >> "$DETAILED_REPORT" << EOF
        </div>

        <div class="footer">
            <p>For detailed logs, see: $VALIDATION_LOG</p>
            <p>BEV OSINT Framework - Comprehensive Security Research Platform</p>
        </div>
    </div>
</body>
</html>
EOF

    log_success "HTML report generated: $DETAILED_REPORT"
}

# Generate summary report
generate_summary() {
    log_header "📋 VALIDATION SUMMARY REPORT"

    local success_rate=0
    if (( TOTAL_VALIDATIONS > 0 )); then
        success_rate=$(( (PASSED_VALIDATIONS * 100) / TOTAL_VALIDATIONS ))
    fi

    echo "🎯 Overall Results:" | tee -a "$VALIDATION_LOG"
    echo "  Total Validations: $TOTAL_VALIDATIONS" | tee -a "$VALIDATION_LOG"
    echo "  Passed: $PASSED_VALIDATIONS" | tee -a "$VALIDATION_LOG"
    echo "  Failed: $FAILED_VALIDATIONS" | tee -a "$VALIDATION_LOG"
    echo "  Warnings: $WARNING_VALIDATIONS" | tee -a "$VALIDATION_LOG"
    echo "  Success Rate: ${success_rate}%" | tee -a "$VALIDATION_LOG"
    echo "" | tee -a "$VALIDATION_LOG"

    echo "📊 Results by Category:" | tee -a "$VALIDATION_LOG"
    for category in "${!CATEGORY_RESULTS[@]}"; do
        echo "  $category: ${CATEGORY_RESULTS[$category]} validations passed" | tee -a "$VALIDATION_LOG"
    done
    echo "" | tee -a "$VALIDATION_LOG"

    # Overall assessment
    if (( success_rate >= 90 )); then
        log_success "🎉 EXCELLENT: Deployment is production-ready!"
        echo "✅ The BEV OSINT Framework deployment is highly reliable and ready for use." | tee -a "$VALIDATION_LOG"
    elif (( success_rate >= 75 )); then
        log_success "✅ GOOD: Deployment is functional with minor issues"
        echo "⚠️ The deployment is working well but may benefit from addressing warnings." | tee -a "$VALIDATION_LOG"
    elif (( success_rate >= 50 )); then
        log_warning "⚠️ PARTIAL: Deployment has significant issues"
        echo "🔧 Several components need attention before production use." | tee -a "$VALIDATION_LOG"
    else
        log_error "❌ POOR: Deployment has major problems"
        echo "🚨 Critical issues must be resolved before using the system." | tee -a "$VALIDATION_LOG"
    fi

    echo "" | tee -a "$VALIDATION_LOG"
    echo "📁 Detailed logs: $VALIDATION_LOG" | tee -a "$VALIDATION_LOG"
    echo "📊 HTML report: $DETAILED_REPORT" | tee -a "$VALIDATION_LOG"
    echo "📅 Validation time: $(date)" | tee -a "$VALIDATION_LOG"

    # Return appropriate exit code
    if (( FAILED_VALIDATIONS == 0 )); then
        return 0
    else
        return 1
    fi
}

# Main validation execution
main() {
    log_header "🔍 BEV OSINT FRAMEWORK - COMPREHENSIVE VALIDATION"
    echo "Starting comprehensive deployment validation at $(date)" > "$VALIDATION_LOG"

    # Run all validation categories
    validate_infrastructure
    validate_databases
    validate_security
    validate_monitoring
    validate_integration
    validate_performance

    # Generate reports
    generate_html_report
    generate_summary

    log_info "🏁 Validation completed!"
}

# Handle script termination
trap 'log_error "Validation interrupted"; exit 1' INT TERM

# Run main validation
main "$@"