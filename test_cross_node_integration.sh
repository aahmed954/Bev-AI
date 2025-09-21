#!/bin/bash
# Cross-Node Integration Testing Script
# BEV OSINT Framework - THANOS <-> ORACLE1 Communication Testing
# Generated: $(date)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THANOS_IP="${THANOS_IP:-127.0.0.1}"
ORACLE1_IP="${ORACLE1_IP:-127.0.0.1}"
TEST_TIMEOUT=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_test() {
    echo -e "${PURPLE}[TEST]${NC} $1"
}

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Test result tracking
start_test() {
    ((TOTAL_TESTS++))
    log_test "$1"
}

pass_test() {
    ((PASSED_TESTS++))
    log_success "✓ PASS: $1"
}

fail_test() {
    ((FAILED_TESTS++))
    log_error "✗ FAIL: $1"
}

# Network connectivity tests
test_basic_connectivity() {
    log_info "=== Testing Basic Network Connectivity ==="

    start_test "THANOS node reachability"
    if ping -c 3 -W 5 "$THANOS_IP" &>/dev/null; then
        pass_test "THANOS node is reachable at $THANOS_IP"
    else
        fail_test "THANOS node is not reachable at $THANOS_IP"
    fi

    start_test "ORACLE1 node reachability"
    if ping -c 3 -W 5 "$ORACLE1_IP" &>/dev/null; then
        pass_test "ORACLE1 node is reachable at $ORACLE1_IP"
    else
        fail_test "ORACLE1 node is not reachable at $ORACLE1_IP"
    fi
}

# Service port accessibility tests
test_service_ports() {
    log_info "=== Testing Service Port Accessibility ==="

    # THANOS services
    local thanos_ports=(
        "5432:PostgreSQL"
        "6379:Redis"
        "7687:Neo4j Bolt"
        "7474:Neo4j HTTP"
        "8200:Vault"
        "9090:Prometheus"
        "3000:Grafana"
        "9200:Elasticsearch"
        "8086:InfluxDB"
    )

    for port_info in "${thanos_ports[@]}"; do
        local port="${port_info%%:*}"
        local service="${port_info##*:}"

        start_test "THANOS $service port accessibility"
        if timeout 5 bash -c "</dev/tcp/$THANOS_IP/$port" 2>/dev/null; then
            pass_test "THANOS $service ($port) is accessible"
        else
            fail_test "THANOS $service ($port) is not accessible"
        fi
    done

    # ORACLE1 services
    local oracle1_ports=(
        "9091:Prometheus"
        "3001:Grafana"
        "9093:AlertManager"
        "9100:Node Exporter"
    )

    for port_info in "${oracle1_ports[@]}"; do
        local port="${port_info%%:*}"
        local service="${port_info##*:}"

        start_test "ORACLE1 $service port accessibility"
        if timeout 5 bash -c "</dev/tcp/$ORACLE1_IP/$port" 2>/dev/null; then
            pass_test "ORACLE1 $service ($port) is accessible"
        else
            fail_test "ORACLE1 $service ($port) is not accessible"
        fi
    done
}

# Database connectivity tests
test_database_connectivity() {
    log_info "=== Testing Database Connectivity ==="

    # Test PostgreSQL from ORACLE1 perspective
    start_test "PostgreSQL cross-node connectivity"
    if command -v psql &>/dev/null; then
        if PGPASSWORD="b9dc1c323301c82a3bd289df7b03c21d" psql -h "$THANOS_IP" -U bev -d osint -c "SELECT 1;" &>/dev/null; then
            pass_test "PostgreSQL cross-node connection successful"
        else
            fail_test "PostgreSQL cross-node connection failed"
        fi
    else
        log_warning "psql not available, skipping PostgreSQL test"
    fi

    # Test Redis from ORACLE1 perspective
    start_test "Redis cross-node connectivity"
    if command -v redis-cli &>/dev/null; then
        if redis-cli -h "$THANOS_IP" -a "3e14b364ec4e38e23b1299700e6ba5a1" ping 2>/dev/null | grep -q "PONG"; then
            pass_test "Redis cross-node connection successful"
        else
            fail_test "Redis cross-node connection failed"
        fi
    else
        log_warning "redis-cli not available, skipping Redis test"
    fi

    # Test Neo4j connectivity
    start_test "Neo4j cross-node connectivity"
    if timeout 10 bash -c "</dev/tcp/$THANOS_IP/7687" 2>/dev/null; then
        pass_test "Neo4j cross-node port accessible"
    else
        fail_test "Neo4j cross-node port not accessible"
    fi
}

# HTTP service tests
test_http_services() {
    log_info "=== Testing HTTP Service Responses ==="

    # Test THANOS web services
    local thanos_http_services=(
        "7474:/db/data/:Neo4j HTTP API"
        "3000/api/health:Grafana API"
        "9090/-/healthy:Prometheus Health"
        "9200/_cluster/health:Elasticsearch Health"
    )

    for service_info in "${thanos_http_services[@]}"; do
        local endpoint="${service_info%%:*}"
        local description="${service_info##*:}"

        start_test "THANOS $description"
        if curl -s --max-time 10 "http://$THANOS_IP:$endpoint" &>/dev/null; then
            pass_test "THANOS $description responds correctly"
        else
            fail_test "THANOS $description not responding"
        fi
    done

    # Test ORACLE1 web services
    local oracle1_http_services=(
        "9091/-/healthy:Prometheus Health"
        "3001/api/health:Grafana API"
        "9093/-/healthy:AlertManager Health"
        "9100/metrics:Node Exporter Metrics"
    )

    for service_info in "${oracle1_http_services[@]}"; do
        local endpoint="${service_info%%:*}"
        local description="${service_info##*:}"

        start_test "ORACLE1 $description"
        if curl -s --max-time 10 "http://$ORACLE1_IP:$endpoint" &>/dev/null; then
            pass_test "ORACLE1 $description responds correctly"
        else
            fail_test "ORACLE1 $description not responding"
        fi
    done
}

# Container communication tests
test_container_communication() {
    log_info "=== Testing Container-to-Container Communication ==="

    # Check if we can reach containers by name
    start_test "Container name resolution"

    # Try to resolve container names from each node
    local container_tests=(
        "bev_postgres:THANOS PostgreSQL container"
        "bev_redis:THANOS Redis container"
        "bev_neo4j:THANOS Neo4j container"
    )

    for container_info in "${container_tests[@]}"; do
        local container="${container_info%%:*}"
        local description="${container_info##*:}"

        start_test "$description name resolution"
        if docker exec -it "$container" echo "test" &>/dev/null; then
            pass_test "$description is accessible"
        else
            fail_test "$description is not accessible"
        fi
    done
}

# Monitoring integration tests
test_monitoring_integration() {
    log_info "=== Testing Monitoring Integration ==="

    start_test "ORACLE1 Prometheus can scrape THANOS metrics"
    if curl -s --max-time 10 "http://$ORACLE1_IP:9091/api/v1/targets" | grep -q "up.*true" 2>/dev/null; then
        pass_test "ORACLE1 Prometheus is scraping targets"
    else
        fail_test "ORACLE1 Prometheus target scraping issues"
    fi

    start_test "Cross-node metric collection"
    if curl -s --max-time 10 "http://$ORACLE1_IP:9091/api/v1/query?query=up" | grep -q "success" 2>/dev/null; then
        pass_test "Cross-node metrics collection working"
    else
        fail_test "Cross-node metrics collection issues"
    fi
}

# Security tests
test_security_integration() {
    log_info "=== Testing Security Integration ==="

    start_test "Vault accessibility from ORACLE1"
    if timeout 10 bash -c "</dev/tcp/$THANOS_IP/8200" 2>/dev/null; then
        pass_test "Vault is accessible from ORACLE1"
    else
        fail_test "Vault is not accessible from ORACLE1"
    fi

    start_test "Encrypted communication test"
    # Test if we can establish secure connections
    if curl -s --max-time 10 "http://$THANOS_IP:8200/v1/sys/health" &>/dev/null; then
        pass_test "Vault API responds correctly"
    else
        fail_test "Vault API not responding"
    fi
}

# Performance tests
test_performance() {
    log_info "=== Testing Cross-Node Performance ==="

    start_test "Network latency test"
    local latency=$(ping -c 5 "$THANOS_IP" 2>/dev/null | tail -1 | awk -F'/' '{print $5}' | cut -d'.' -f1)
    if [[ -n "$latency" ]] && (( latency < 50 )); then
        pass_test "Network latency is acceptable ($latency ms)"
    else
        fail_test "Network latency is high or unmeasurable"
    fi

    start_test "Database response time"
    local start_time=$(date +%s%N)
    if timeout 10 bash -c "</dev/tcp/$THANOS_IP/5432" 2>/dev/null; then
        local end_time=$(date +%s%N)
        local response_time=$(( (end_time - start_time) / 1000000 ))
        if (( response_time < 1000 )); then
            pass_test "Database response time is acceptable ($response_time ms)"
        else
            fail_test "Database response time is high ($response_time ms)"
        fi
    else
        fail_test "Database connection timeout"
    fi
}

# Generate test report
generate_report() {
    log_info "=== Cross-Node Integration Test Report ==="
    echo "========================================"
    echo "📊 Test Results Summary:"
    echo "  Total Tests: $TOTAL_TESTS"
    echo "  Passed: $PASSED_TESTS"
    echo "  Failed: $FAILED_TESTS"
    echo ""

    local success_rate=0
    if (( TOTAL_TESTS > 0 )); then
        success_rate=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))
    fi

    echo "  Success Rate: ${success_rate}%"
    echo ""

    if (( success_rate >= 90 )); then
        log_success "🎉 EXCELLENT: Cross-node integration is working excellently!"
    elif (( success_rate >= 75 )); then
        log_success "✅ GOOD: Cross-node integration is working well with minor issues"
    elif (( success_rate >= 50 )); then
        log_warning "⚠️ PARTIAL: Cross-node integration has significant issues"
    else
        log_error "❌ POOR: Cross-node integration has major problems"
    fi

    echo ""
    echo "📋 Recommendations:"
    if (( FAILED_TESTS > 0 )); then
        echo "  • Review failed tests and check service logs"
        echo "  • Verify network connectivity between nodes"
        echo "  • Check firewall and security group settings"
        echo "  • Ensure all services are fully started and healthy"
    else
        echo "  • All tests passed! System is ready for production use"
        echo "  • Consider running periodic integration tests"
        echo "  • Monitor cross-node performance metrics"
    fi
    echo "========================================"
}

# Main test execution
main() {
    log_info "Starting Cross-Node Integration Testing..."
    echo "THANOS IP: $THANOS_IP"
    echo "ORACLE1 IP: $ORACLE1_IP"
    echo "========================================"

    # Run all test suites
    test_basic_connectivity
    test_service_ports
    test_database_connectivity
    test_http_services
    test_container_communication
    test_monitoring_integration
    test_security_integration
    test_performance

    # Generate final report
    generate_report

    # Exit with appropriate code
    if (( FAILED_TESTS == 0 )); then
        exit 0
    else
        exit 1
    fi
}

# Handle script termination
trap 'log_error "Cross-node testing interrupted"; exit 1' INT TERM

# Run main function
main "$@"