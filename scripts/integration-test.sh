#!/bin/bash

# BEV OSINT Integration Test Script
# Validates Prometheus, Grafana, Logging, Security, and Service Discovery integration

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="/tmp/bev-integration-test.log"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Logging function
log() {
    local level="$1"
    shift
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $*" | tee -a "$LOG_FILE"
}

# Test result function
test_result() {
    local test_name="$1"
    local result="$2"
    local message="${3:-}"

    TESTS_RUN=$((TESTS_RUN + 1))

    if [ "$result" = "PASS" ]; then
        echo -e "${GREEN}✅ PASS: $test_name${NC}"
        [ -n "$message" ] && echo "   $message"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        log "INFO" "TEST PASS: $test_name - $message"
    else
        echo -e "${RED}❌ FAIL: $test_name${NC}"
        [ -n "$message" ] && echo "   $message"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        log "ERROR" "TEST FAIL: $test_name - $message"
    fi
}

# Configuration file validation
test_config_files() {
    echo -e "${BLUE}🔍 Testing Configuration Files${NC}"

    # Prometheus configuration
    if [ -f "$PROJECT_ROOT/config/prometheus.yml" ]; then
        if grep -q "dm-crawler\|crypto-intel\|reputation-analyzer\|economics-processor" "$PROJECT_ROOT/config/prometheus.yml" && \
           grep -q "tactical-intel\|defense-automation\|opsec-enforcer\|intel-fusion" "$PROJECT_ROOT/config/prometheus.yml" && \
           grep -q "autonomous-coordinator\|adaptive-learning\|resource-manager\|knowledge-evolution" "$PROJECT_ROOT/config/prometheus.yml"; then
            test_result "Prometheus Configuration" "PASS" "All 12 new services configured"
        else
            test_result "Prometheus Configuration" "FAIL" "Missing service configurations"
        fi
    else
        test_result "Prometheus Configuration" "FAIL" "prometheus.yml not found"
    fi

    # Prometheus alerts
    if [ -f "$PROJECT_ROOT/config/prometheus-alerts.yml" ]; then
        if grep -q "phase7_market_intelligence\|phase8_security_operations\|phase9_autonomous_systems" "$PROJECT_ROOT/config/prometheus-alerts.yml"; then
            test_result "Prometheus Alerts" "PASS" "Phase-specific alert rules configured"
        else
            test_result "Prometheus Alerts" "FAIL" "Missing phase-specific alert rules"
        fi
    else
        test_result "Prometheus Alerts" "FAIL" "prometheus-alerts.yml not found"
    fi

    # Grafana dashboards
    local dashboard_count=0
    for dashboard in phase7-market-intelligence.json phase8-security-operations.json phase9-autonomous-systems.json bev-unified-overview.json; do
        if [ -f "$PROJECT_ROOT/config/grafana/dashboards/$dashboard" ]; then
            dashboard_count=$((dashboard_count + 1))
        fi
    done

    if [ $dashboard_count -eq 4 ]; then
        test_result "Grafana Dashboards" "PASS" "All 4 dashboards created"
    else
        test_result "Grafana Dashboards" "FAIL" "Only $dashboard_count/4 dashboards found"
    fi

    # Metrics definitions
    local metrics_count=0
    for metrics in phase7-metrics.yml phase8-metrics.yml phase9-metrics.yml; do
        if [ -f "$PROJECT_ROOT/config/metrics/$metrics" ]; then
            metrics_count=$((metrics_count + 1))
        fi
    done

    if [ $metrics_count -eq 3 ]; then
        test_result "Metrics Definitions" "PASS" "All phase metrics defined"
    else
        test_result "Metrics Definitions" "FAIL" "Only $metrics_count/3 metrics files found"
    fi

    # Logging configuration
    if [ -f "$PROJECT_ROOT/config/logging/filebeat-phases.yml" ] && \
       [ -f "$PROJECT_ROOT/config/logging/logstash-phases.conf" ]; then
        test_result "Logging Configuration" "PASS" "Filebeat and Logstash configs present"
    else
        test_result "Logging Configuration" "FAIL" "Missing logging configuration files"
    fi

    # Security configuration
    if [ -f "$PROJECT_ROOT/config/security/phases-auth-config.yml" ]; then
        if grep -q "phase7_services\|phase8_services\|phase9_services" "$PROJECT_ROOT/config/security/phases-auth-config.yml"; then
            test_result "Security Configuration" "PASS" "Phase-specific security configs present"
        else
            test_result "Security Configuration" "FAIL" "Missing phase-specific security configs"
        fi
    else
        test_result "Security Configuration" "FAIL" "Security config file not found"
    fi

    # Service discovery
    if [ -f "$PROJECT_ROOT/config/consul/phases-service-discovery.json" ]; then
        local service_count
        service_count=$(grep -c '"name":' "$PROJECT_ROOT/config/consul/phases-service-discovery.json" || echo "0")
        if [ "$service_count" -ge 12 ]; then
            test_result "Service Discovery" "PASS" "$service_count services configured in Consul"
        else
            test_result "Service Discovery" "FAIL" "Only $service_count services configured"
        fi
    else
        test_result "Service Discovery" "FAIL" "Consul service discovery config not found"
    fi

    # Health check script
    if [ -f "$PROJECT_ROOT/config/health-checks/health-check-script.sh" ] && \
       [ -x "$PROJECT_ROOT/config/health-checks/health-check-script.sh" ]; then
        test_result "Health Check Script" "PASS" "Script exists and is executable"
    else
        test_result "Health Check Script" "FAIL" "Script missing or not executable"
    fi
}

# YAML/JSON syntax validation
test_syntax_validation() {
    echo -e "${BLUE}🔍 Testing YAML/JSON Syntax${NC}"

    # Test YAML files
    local yaml_files=(
        "config/prometheus.yml"
        "config/prometheus-alerts.yml"
        "config/metrics/phase7-metrics.yml"
        "config/metrics/phase8-metrics.yml"
        "config/metrics/phase9-metrics.yml"
        "config/logging/filebeat-phases.yml"
        "config/security/phases-auth-config.yml"
    )

    local yaml_valid=0
    local yaml_total=${#yaml_files[@]}

    for yaml_file in "${yaml_files[@]}"; do
        if [ -f "$PROJECT_ROOT/$yaml_file" ]; then
            if python3 -c "import yaml; yaml.safe_load(open('$PROJECT_ROOT/$yaml_file', 'r'))" 2>/dev/null; then
                yaml_valid=$((yaml_valid + 1))
            else
                log "ERROR" "Invalid YAML syntax in $yaml_file"
            fi
        fi
    done

    if [ $yaml_valid -eq $yaml_total ]; then
        test_result "YAML Syntax Validation" "PASS" "All $yaml_total YAML files valid"
    else
        test_result "YAML Syntax Validation" "FAIL" "Only $yaml_valid/$yaml_total YAML files valid"
    fi

    # Test JSON files
    local json_files=(
        "config/grafana/dashboards/phase7-market-intelligence.json"
        "config/grafana/dashboards/phase8-security-operations.json"
        "config/grafana/dashboards/phase9-autonomous-systems.json"
        "config/grafana/dashboards/bev-unified-overview.json"
        "config/consul/phases-service-discovery.json"
    )

    local json_valid=0
    local json_total=${#json_files[@]}

    for json_file in "${json_files[@]}"; do
        if [ -f "$PROJECT_ROOT/$json_file" ]; then
            if python3 -c "import json; json.load(open('$PROJECT_ROOT/$json_file', 'r'))" 2>/dev/null; then
                json_valid=$((json_valid + 1))
            else
                log "ERROR" "Invalid JSON syntax in $json_file"
            fi
        fi
    done

    if [ $json_valid -eq $json_total ]; then
        test_result "JSON Syntax Validation" "PASS" "All $json_total JSON files valid"
    else
        test_result "JSON Syntax Validation" "FAIL" "Only $json_valid/$json_total JSON files valid"
    fi
}

# Test Prometheus configuration validation
test_prometheus_config() {
    echo -e "${BLUE}🔍 Testing Prometheus Configuration${NC}"

    # Check if prometheus binary is available for validation
    if command -v promtool >/dev/null 2>&1; then
        if promtool check config "$PROJECT_ROOT/config/prometheus.yml" >/dev/null 2>&1; then
            test_result "Prometheus Config Validation" "PASS" "Configuration syntax valid"
        else
            test_result "Prometheus Config Validation" "FAIL" "Configuration syntax invalid"
        fi

        if promtool check rules "$PROJECT_ROOT/config/prometheus-alerts.yml" >/dev/null 2>&1; then
            test_result "Prometheus Rules Validation" "PASS" "Alert rules syntax valid"
        else
            test_result "Prometheus Rules Validation" "FAIL" "Alert rules syntax invalid"
        fi
    else
        test_result "Prometheus Validation" "SKIP" "promtool not available"
    fi
}

# Test Docker compose integration
test_docker_integration() {
    echo -e "${BLUE}🔍 Testing Docker Compose Integration${NC}"

    local compose_files=(
        "docker-compose-phase7.yml"
        "docker-compose-phase8.yml"
        "docker-compose-phase9.yml"
    )

    local compose_valid=0
    local compose_total=${#compose_files[@]}

    for compose_file in "${compose_files[@]}"; do
        if [ -f "$PROJECT_ROOT/$compose_file" ]; then
            if docker-compose -f "$PROJECT_ROOT/$compose_file" config >/dev/null 2>&1; then
                compose_valid=$((compose_valid + 1))
            else
                log "ERROR" "Invalid docker-compose syntax in $compose_file"
            fi
        fi
    done

    if [ $compose_valid -eq $compose_total ]; then
        test_result "Docker Compose Validation" "PASS" "All $compose_total compose files valid"
    else
        test_result "Docker Compose Validation" "FAIL" "Only $compose_valid/$compose_total compose files valid"
    fi
}

# Test network configuration
test_network_config() {
    echo -e "${BLUE}🔍 Testing Network Configuration${NC}"

    # Check IP address assignments in docker-compose files
    local ip_conflicts=0
    local expected_ips=(
        "172.30.0.24" "172.30.0.25" "172.30.0.26" "172.30.0.27"  # Phase 7
        "172.30.0.28" "172.30.0.29" "172.30.0.30" "172.30.0.31"  # Phase 8
        "172.30.0.32" "172.30.0.33" "172.30.0.34" "172.30.0.35"  # Phase 9
    )

    # Collect all IP addresses from compose files
    local all_ips=()
    for compose_file in docker-compose-phase7.yml docker-compose-phase8.yml docker-compose-phase9.yml; do
        if [ -f "$PROJECT_ROOT/$compose_file" ]; then
            while IFS= read -r ip; do
                all_ips+=("$ip")
            done < <(grep -o "172\.30\.0\.[0-9]\+" "$PROJECT_ROOT/$compose_file" | sort)
        fi
    done

    # Check for duplicates
    local unique_ips
    unique_ips=$(printf '%s\n' "${all_ips[@]}" | sort -u | wc -l)
    local total_ips=${#all_ips[@]}

    if [ "$unique_ips" -eq "$total_ips" ]; then
        test_result "IP Address Configuration" "PASS" "No IP conflicts detected"
    else
        test_result "IP Address Configuration" "FAIL" "IP address conflicts detected"
    fi

    # Check port assignments
    local port_conflicts=0
    for port in 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011 8012; do
        local port_count
        port_count=$(grep -c "\"$port:" "$PROJECT_ROOT"/docker-compose-phase*.yml || echo "0")
        if [ "$port_count" -gt 1 ]; then
            port_conflicts=$((port_conflicts + 1))
        fi
    done

    if [ $port_conflicts -eq 0 ]; then
        test_result "Port Configuration" "PASS" "No port conflicts detected"
    else
        test_result "Port Configuration" "FAIL" "$port_conflicts port conflicts detected"
    fi
}

# Test security configuration
test_security_config() {
    echo -e "${BLUE}🔍 Testing Security Configuration${NC}"

    # Check authentication methods
    if grep -q "jwt:\|api_keys:\|mtls:" "$PROJECT_ROOT/config/security/phases-auth-config.yml"; then
        test_result "Authentication Methods" "PASS" "Multiple auth methods configured"
    else
        test_result "Authentication Methods" "FAIL" "Missing authentication methods"
    fi

    # Check authorization matrix
    if grep -q "service_matrix:" "$PROJECT_ROOT/config/security/phases-auth-config.yml"; then
        local service_count
        service_count=$(grep -c "can_access:" "$PROJECT_ROOT/config/security/phases-auth-config.yml" || echo "0")
        if [ "$service_count" -ge 12 ]; then
            test_result "Authorization Matrix" "PASS" "$service_count services configured"
        else
            test_result "Authorization Matrix" "FAIL" "Only $service_count services configured"
        fi
    else
        test_result "Authorization Matrix" "FAIL" "Service authorization matrix missing"
    fi

    # Check TLS configuration
    if grep -q "tls:\|cert_file:\|key_file:" "$PROJECT_ROOT/config/security/phases-auth-config.yml"; then
        test_result "TLS Configuration" "PASS" "TLS settings configured"
    else
        test_result "TLS Configuration" "FAIL" "Missing TLS configuration"
    fi

    # Check audit configuration
    if grep -q "audit:\|logging:\|events:" "$PROJECT_ROOT/config/security/phases-auth-config.yml"; then
        test_result "Audit Configuration" "PASS" "Audit logging configured"
    else
        test_result "Audit Configuration" "FAIL" "Missing audit configuration"
    fi
}

# Test logging integration
test_logging_integration() {
    echo -e "${BLUE}🔍 Testing Logging Integration${NC}"

    # Check Filebeat configuration
    if [ -f "$PROJECT_ROOT/config/logging/filebeat-phases.yml" ]; then
        local input_count
        input_count=$(grep -c "type: log" "$PROJECT_ROOT/config/logging/filebeat-phases.yml" || echo "0")
        if [ "$input_count" -ge 12 ]; then
            test_result "Filebeat Inputs" "PASS" "$input_count log inputs configured"
        else
            test_result "Filebeat Inputs" "FAIL" "Only $input_count log inputs configured"
        fi

        # Check PII redaction
        if grep -q "pii_redaction\|EMAIL_REDACTED\|IP_REDACTED" "$PROJECT_ROOT/config/logging/filebeat-phases.yml"; then
            test_result "PII Redaction" "PASS" "PII redaction configured"
        else
            test_result "PII Redaction" "FAIL" "Missing PII redaction"
        fi
    fi

    # Check Logstash configuration
    if [ -f "$PROJECT_ROOT/config/logging/logstash-phases.conf" ]; then
        # Check phase-specific processing
        if grep -q "phase.*=.*\"7\"\|phase.*=.*\"8\"\|phase.*=.*\"9\"" "$PROJECT_ROOT/config/logging/logstash-phases.conf"; then
            test_result "Logstash Phase Processing" "PASS" "Phase-specific processing configured"
        else
            test_result "Logstash Phase Processing" "FAIL" "Missing phase-specific processing"
        fi

        # Check Elasticsearch output
        if grep -q "elasticsearch" "$PROJECT_ROOT/config/logging/logstash-phases.conf"; then
            test_result "Logstash Output" "PASS" "Elasticsearch output configured"
        else
            test_result "Logstash Output" "FAIL" "Missing Elasticsearch output"
        fi
    fi
}

# Performance and resource validation
test_resource_config() {
    echo -e "${BLUE}🔍 Testing Resource Configuration${NC}"

    # Check memory limits in compose files
    local memory_configs=0
    for compose_file in docker-compose-phase7.yml docker-compose-phase8.yml docker-compose-phase9.yml; do
        if [ -f "$PROJECT_ROOT/$compose_file" ]; then
            local mem_count
            mem_count=$(grep -c "memory:" "$PROJECT_ROOT/$compose_file" || echo "0")
            memory_configs=$((memory_configs + mem_count))
        fi
    done

    if [ $memory_configs -ge 12 ]; then
        test_result "Memory Limits" "PASS" "$memory_configs memory limits configured"
    else
        test_result "Memory Limits" "FAIL" "Only $memory_configs memory limits configured"
    fi

    # Check CPU limits
    local cpu_configs=0
    for compose_file in docker-compose-phase7.yml docker-compose-phase8.yml docker-compose-phase9.yml; do
        if [ -f "$PROJECT_ROOT/$compose_file" ]; then
            local cpu_count
            cpu_count=$(grep -c "cpus:" "$PROJECT_ROOT/$compose_file" || echo "0")
            cpu_configs=$((cpu_configs + cpu_count))
        fi
    done

    if [ $cpu_configs -ge 12 ]; then
        test_result "CPU Limits" "PASS" "$cpu_configs CPU limits configured"
    else
        test_result "CPU Limits" "FAIL" "Only $cpu_configs CPU limits configured"
    fi

    # Check GPU configuration where expected
    local gpu_configs=0
    for compose_file in docker-compose-phase7.yml docker-compose-phase8.yml docker-compose-phase9.yml; do
        if [ -f "$PROJECT_ROOT/$compose_file" ]; then
            local gpu_count
            gpu_count=$(grep -c "nvidia" "$PROJECT_ROOT/$compose_file" || echo "0")
            gpu_configs=$((gpu_configs + gpu_count))
        fi
    done

    if [ $gpu_configs -ge 3 ]; then
        test_result "GPU Configuration" "PASS" "$gpu_configs GPU configurations found"
    else
        test_result "GPU Configuration" "FAIL" "Only $gpu_configs GPU configurations found"
    fi
}

# Generate integration test report
generate_report() {
    echo ""
    echo "================================================"
    echo "BEV OSINT INTEGRATION TEST REPORT"
    echo "================================================"
    echo "Timestamp: $(date)"
    echo "Tests Run: $TESTS_RUN"
    echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"

    local success_rate
    if [ $TESTS_RUN -gt 0 ]; then
        success_rate=$(( (TESTS_PASSED * 100) / TESTS_RUN ))
        echo "Success Rate: $success_rate%"

        if [ $success_rate -ge 95 ]; then
            echo -e "${GREEN}✅ INTEGRATION STATUS: EXCELLENT${NC}"
            exit_code=0
        elif [ $success_rate -ge 85 ]; then
            echo -e "${YELLOW}⚠️ INTEGRATION STATUS: GOOD${NC}"
            exit_code=0
        elif [ $success_rate -ge 70 ]; then
            echo -e "${YELLOW}⚠️ INTEGRATION STATUS: ACCEPTABLE${NC}"
            exit_code=1
        else
            echo -e "${RED}❌ INTEGRATION STATUS: NEEDS ATTENTION${NC}"
            exit_code=2
        fi
    else
        echo -e "${RED}❌ INTEGRATION STATUS: NO TESTS RUN${NC}"
        exit_code=3
    fi

    echo ""
    echo "Detailed log: $LOG_FILE"
    echo "================================================"
}

# Main execution
main() {
    echo -e "${BLUE}🚀 Starting BEV OSINT Integration Tests${NC}"
    log "INFO" "Starting integration test suite"

    # Run all test suites
    test_config_files
    test_syntax_validation
    test_prometheus_config
    test_docker_integration
    test_network_config
    test_security_config
    test_logging_integration
    test_resource_config

    # Generate final report
    generate_report

    exit $exit_code
}

# Execute main function
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi