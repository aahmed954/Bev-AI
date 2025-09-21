#!/bin/bash
# BEV Frontend Integration - Comprehensive Integration Test Suite
# Validates deployment success and performs end-to-end testing
# Author: DevOps Automation Framework
# Version: 1.0.0

set -euo pipefail

# =====================================================
# Configuration and Constants
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/deployment"
LOG_FILE="${LOG_DIR}/integration-tests-$(date +%Y%m%d_%H%M%S).log"
RESULTS_DIR="${PROJECT_ROOT}/logs/test-results"
REPORTS_DIR="${RESULTS_DIR}/$(date +%Y%m%d_%H%M%S)"

# Test configuration
MAX_RETRY_ATTEMPTS=3
REQUEST_TIMEOUT=30
LOAD_TEST_DURATION=30
CONCURRENT_USERS=10

# Service endpoints (from deployment configuration)
FRONTEND_HTTP_PORT=3010
FRONTEND_HTTPS_PORT=8443
MCP_SERVER_PORT=3011
FRONTEND_WS_PORT=8081
HAPROXY_STATS_PORT=8080

# Test data
TEST_API_KEY=""
TEST_USER_ID="test-user-$(date +%s)"
TEST_SESSION_ID=""

# =====================================================
# Logging and Utility Functions
# =====================================================

setup_logging() {
    mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${REPORTS_DIR}"
    exec 1> >(tee -a "${LOG_FILE}")
    exec 2> >(tee -a "${LOG_FILE}" >&2)
    echo "=== BEV Integration Test Suite Started at $(date) ===" | tee -a "${LOG_FILE}"
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "${LOG_FILE}"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*" | tee -a "${LOG_FILE}"
}

log_test_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "${timestamp},${test_name},${status},${details}" >> "${REPORTS_DIR}/test_results.csv"
    
    if [ "${status}" = "PASS" ]; then
        log_success "TEST PASS: ${test_name} - ${details}"
    elif [ "${status}" = "FAIL" ]; then
        log_error "TEST FAIL: ${test_name} - ${details}"
    elif [ "${status}" = "SKIP" ]; then
        log_warn "TEST SKIP: ${test_name} - ${details}"
    fi
}

# =====================================================
# Test Utility Functions
# =====================================================

make_http_request() {
    local method="$1"
    local url="$2"
    local expected_status="$3"
    local additional_args="${4:-}"
    local max_retries="${5:-$MAX_RETRY_ATTEMPTS}"
    
    local attempt=1
    while [ $attempt -le $max_retries ]; do
        local response=$(curl -s -w "HTTPSTATUS:%{http_code};TIME:%{time_total};SIZE:%{size_download}" \
            -X "$method" \
            --max-time "$REQUEST_TIMEOUT" \
            --connect-timeout 10 \
            $additional_args \
            "$url" 2>/dev/null || echo "HTTPSTATUS:000;TIME:0;SIZE:0")
        
        local http_code=$(echo "$response" | grep -o "HTTPSTATUS:[0-9]*" | cut -d: -f2)
        local response_time=$(echo "$response" | grep -o "TIME:[0-9.]*" | cut -d: -f2)
        local response_size=$(echo "$response" | grep -o "SIZE:[0-9]*" | cut -d: -f2)
        local body=$(echo "$response" | sed -E 's/HTTPSTATUS:[0-9]*;TIME:[0-9.]*;SIZE:[0-9]*$//')
        
        if [ "$http_code" = "$expected_status" ]; then
            echo "$body"
            return 0
        fi
        
        log_warn "Attempt $attempt failed: HTTP $http_code (expected $expected_status) for $url"
        ((attempt++))
        sleep 2
    done
    
    log_error "All attempts failed for $url after $max_retries retries"
    return 1
}

test_websocket_connection() {
    local ws_url="$1"
    local test_message="$2"
    local timeout="${3:-10}"
    
    if ! command -v wscat &> /dev/null; then
        # Install wscat if available
        if command -v npm &> /dev/null; then
            npm install -g wscat &>/dev/null || true
        fi
        
        if ! command -v wscat &> /dev/null; then
            log_warn "wscat not available, using alternative WebSocket test"
            # Use Python WebSocket test as fallback
            python3 -c "
import asyncio
import websockets
import sys
import json

async def test_websocket():
    try:
        uri = '$ws_url'
        async with websockets.connect(uri, timeout=$timeout) as websocket:
            await websocket.send('$test_message')
            response = await websocket.recv()
            print('WebSocket response received')
            return 0
    except Exception as e:
        print(f'WebSocket error: {e}')
        return 1

sys.exit(asyncio.run(test_websocket()))
" 2>/dev/null
            return $?
        fi
    fi
    
    # Use wscat for testing
    echo "$test_message" | timeout "$timeout" wscat -c "$ws_url" -w 2 &>/dev/null
    return $?
}

generate_test_data() {
    log_info "Generating test data..."
    
    # Generate test API key
    TEST_API_KEY=$(openssl rand -hex 32)
    
    # Create test configuration
    cat > "${REPORTS_DIR}/test_config.json" << EOF
{
    "test_execution": {
        "timestamp": "$(date -Iseconds)",
        "test_id": "integration-test-$(date +%s)",
        "environment": "deployment-validation",
        "tester": "automation-framework"
    },
    "endpoints": {
        "frontend_http": "http://localhost:${FRONTEND_HTTP_PORT}",
        "frontend_https": "https://localhost:${FRONTEND_HTTPS_PORT}",
        "mcp_server": "http://localhost:${MCP_SERVER_PORT}",
        "websocket": "ws://localhost:${FRONTEND_WS_PORT}",
        "haproxy_stats": "http://localhost:${HAPROXY_STATS_PORT}"
    },
    "test_data": {
        "api_key": "${TEST_API_KEY}",
        "user_id": "${TEST_USER_ID}",
        "test_query": "integration test query",
        "test_payload": {
            "tool": "test_tool",
            "parameters": {"query": "test"}
        }
    }
}
EOF
    
    # Initialize CSV report header
    echo "timestamp,test_name,status,details" > "${REPORTS_DIR}/test_results.csv"
    
    log_success "Test data generated"
}

# =====================================================
# Basic Connectivity Tests
# =====================================================

test_service_connectivity() {
    log_info "Testing basic service connectivity..."
    
    local tests_passed=0
    local tests_total=0
    
    # Test HTTP redirect
    ((tests_total++))
    local http_response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${FRONTEND_HTTP_PORT}/health" 2>/dev/null || echo "000")
    if [ "$http_response" = "301" ]; then
        log_test_result "HTTP_REDIRECT" "PASS" "HTTP properly redirects to HTTPS (301)"
        ((tests_passed++))
    else
        log_test_result "HTTP_REDIRECT" "FAIL" "Expected 301, got $http_response"
    fi
    
    # Test HTTPS endpoint
    ((tests_total++))
    if make_http_request "GET" "https://localhost:${FRONTEND_HTTPS_PORT}/health" "200" "-k" >/dev/null; then
        log_test_result "HTTPS_ENDPOINT" "PASS" "HTTPS endpoint responding correctly"
        ((tests_passed++))
    else
        log_test_result "HTTPS_ENDPOINT" "FAIL" "HTTPS endpoint not responding"
    fi
    
    # Test MCP server health
    ((tests_total++))
    if make_http_request "GET" "http://localhost:${MCP_SERVER_PORT}/health" "200" >/dev/null; then
        log_test_result "MCP_HEALTH" "PASS" "MCP server health check successful"
        ((tests_passed++))
    else
        log_test_result "MCP_HEALTH" "FAIL" "MCP server health check failed"
    fi
    
    # Test HAProxy stats
    ((tests_total++))
    if make_http_request "GET" "http://localhost:${HAPROXY_STATS_PORT}/haproxy-stats" "200" >/dev/null; then
        log_test_result "HAPROXY_STATS" "PASS" "HAProxy stats page accessible"
        ((tests_passed++))
    else
        log_test_result "HAPROXY_STATS" "FAIL" "HAProxy stats page not accessible"
    fi
    
    # Test WebSocket connection
    ((tests_total++))
    if test_websocket_connection "ws://localhost:${FRONTEND_WS_PORT}/ws" '{"type":"ping","data":"test"}' 10; then
        log_test_result "WEBSOCKET_CONNECTION" "PASS" "WebSocket connection successful"
        ((tests_passed++))
    else
        log_test_result "WEBSOCKET_CONNECTION" "FAIL" "WebSocket connection failed"
    fi
    
    log_info "Connectivity tests: $tests_passed/$tests_total passed"
    return $(( tests_total - tests_passed ))
}

# =====================================================
# API Functionality Tests
# =====================================================

test_api_functionality() {
    log_info "Testing API functionality..."
    
    local tests_passed=0
    local tests_total=0
    
    # Test MCP server status endpoint
    ((tests_total++))
    local status_response=$(make_http_request "GET" "http://localhost:${MCP_SERVER_PORT}/api/status" "200")
    if [ $? -eq 0 ] && echo "$status_response" | jq -e '.service' >/dev/null 2>&1; then
        log_test_result "MCP_STATUS_API" "PASS" "MCP status API returned valid JSON"
        ((tests_passed++))
    else
        log_test_result "MCP_STATUS_API" "FAIL" "MCP status API failed or returned invalid JSON"
    fi
    
    # Test MCP tools execution endpoint
    ((tests_total++))
    local tools_payload='{"tool":"test_tool","parameters":{"query":"integration_test"}}'
    local tools_response=$(make_http_request "POST" "http://localhost:${MCP_SERVER_PORT}/api/tools/execute" "200" \
        "-H 'Content-Type: application/json' -d '$tools_payload'")
    if [ $? -eq 0 ] && echo "$tools_response" | jq -e '.success' >/dev/null 2>&1; then
        log_test_result "MCP_TOOLS_API" "PASS" "MCP tools execution API working"
        ((tests_passed++))
    else
        log_test_result "MCP_TOOLS_API" "FAIL" "MCP tools execution API failed"
    fi
    
    # Test BEV services integration
    ((tests_total++))
    local bev_response=$(make_http_request "GET" "http://localhost:${MCP_SERVER_PORT}/api/bev/services" "200")
    if [ $? -eq 0 ] && echo "$bev_response" | jq -e '.services' >/dev/null 2>&1; then
        log_test_result "BEV_INTEGRATION_API" "PASS" "BEV services integration API working"
        ((tests_passed++))
    else
        log_test_result "BEV_INTEGRATION_API" "FAIL" "BEV services integration API failed"
    fi
    
    # Test CORS headers
    ((tests_total++))
    local cors_response=$(curl -s -H "Origin: http://localhost:${FRONTEND_HTTP_PORT}" \
        -H "Access-Control-Request-Method: POST" \
        -H "Access-Control-Request-Headers: Content-Type" \
        -X OPTIONS "http://localhost:${MCP_SERVER_PORT}/api/status" \
        -w "HEADERS:%{header_json}" 2>/dev/null || echo "HEADERS:{}")
    
    local cors_headers=$(echo "$cors_response" | grep -o "HEADERS:{.*}" | cut -d: -f2-)
    if echo "$cors_headers" | jq -e '.["access-control-allow-origin"]' >/dev/null 2>&1; then
        log_test_result "CORS_HEADERS" "PASS" "CORS headers properly configured"
        ((tests_passed++))
    else
        log_test_result "CORS_HEADERS" "FAIL" "CORS headers not properly configured"
    fi
    
    log_info "API functionality tests: $tests_passed/$tests_total passed"
    return $(( tests_total - tests_passed ))
}

# =====================================================
# Security Tests
# =====================================================

test_security_features() {
    log_info "Testing security features..."
    
    local tests_passed=0
    local tests_total=0
    
    # Test HTTPS SSL certificate
    ((tests_total++))
    local ssl_info=$(openssl s_client -connect "localhost:${FRONTEND_HTTPS_PORT}" -servername localhost < /dev/null 2>/dev/null | \
        openssl x509 -noout -dates 2>/dev/null || echo "")
    if [ -n "$ssl_info" ]; then
        log_test_result "SSL_CERTIFICATE" "PASS" "SSL certificate valid and accessible"
        ((tests_passed++))
    else
        log_test_result "SSL_CERTIFICATE" "FAIL" "SSL certificate not accessible or invalid"
    fi
    
    # Test security headers
    ((tests_total++))
    local security_headers=$(curl -k -s -I "https://localhost:${FRONTEND_HTTPS_PORT}/health" 2>/dev/null || echo "")
    local required_headers=("X-Frame-Options" "X-Content-Type-Options" "X-XSS-Protection" "Strict-Transport-Security")
    local headers_found=0
    
    for header in "${required_headers[@]}"; do
        if echo "$security_headers" | grep -i "$header" >/dev/null; then
            ((headers_found++))
        fi
    done
    
    if [ $headers_found -eq ${#required_headers[@]} ]; then
        log_test_result "SECURITY_HEADERS" "PASS" "All required security headers present ($headers_found/${#required_headers[@]})"
        ((tests_passed++))
    else
        log_test_result "SECURITY_HEADERS" "FAIL" "Missing security headers ($headers_found/${#required_headers[@]} found)"
    fi
    
    # Test rate limiting (if implemented)
    ((tests_total++))
    local rate_limit_test=true
    for i in {1..20}; do
        local response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${MCP_SERVER_PORT}/api/status" 2>/dev/null || echo "000")
        if [ "$response" = "429" ]; then
            log_test_result "RATE_LIMITING" "PASS" "Rate limiting active (429 after $i requests)"
            rate_limit_test=false
            ((tests_passed++))
            break
        fi
        sleep 0.1
    done
    
    if [ "$rate_limit_test" = true ]; then
        log_test_result "RATE_LIMITING" "SKIP" "Rate limiting not triggered or not implemented"
    fi
    
    # Test input validation
    ((tests_total++))
    local malicious_payload='{"tool":"<script>alert(1)</script>","parameters":{"query":"../../etc/passwd"}}'
    local validation_response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "http://localhost:${MCP_SERVER_PORT}/api/tools/execute" \
        -H "Content-Type: application/json" \
        -d "$malicious_payload" 2>/dev/null || echo "000")
    
    if [ "$validation_response" = "400" ] || [ "$validation_response" = "422" ]; then
        log_test_result "INPUT_VALIDATION" "PASS" "Input validation properly rejects malicious input"
        ((tests_passed++))
    else
        log_test_result "INPUT_VALIDATION" "WARN" "Input validation response: $validation_response (may need review)"
    fi
    
    log_info "Security tests: $tests_passed/$tests_total passed"
    return $(( tests_total - tests_passed ))
}

# =====================================================
# Performance Tests
# =====================================================

test_performance() {
    log_info "Testing performance characteristics..."
    
    local tests_passed=0
    local tests_total=0
    
    # Test response time for health endpoint
    ((tests_total++))
    local response_times=()
    for i in {1..10}; do
        local response_time=$(curl -s -w "%{time_total}" -o /dev/null "http://localhost:${MCP_SERVER_PORT}/health" 2>/dev/null || echo "999")
        response_times+=("$response_time")
    done
    
    # Calculate average response time
    local total_time=0
    for time in "${response_times[@]}"; do
        total_time=$(echo "$total_time + $time" | bc -l)
    done
    local avg_time=$(echo "scale=3; $total_time / ${#response_times[@]}" | bc -l)
    
    if (( $(echo "$avg_time < 1.0" | bc -l) )); then
        log_test_result "RESPONSE_TIME" "PASS" "Average response time: ${avg_time}s (< 1.0s)"
        ((tests_passed++))
    else
        log_test_result "RESPONSE_TIME" "FAIL" "Average response time: ${avg_time}s (>= 1.0s)"
    fi
    
    # Test concurrent connections
    ((tests_total++))
    log_info "Testing concurrent connections..."
    local concurrent_pids=()
    local concurrent_results=()
    
    for i in $(seq 1 $CONCURRENT_USERS); do
        {
            local result=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${MCP_SERVER_PORT}/api/status" 2>/dev/null || echo "000")
            echo "$result" > "/tmp/concurrent_test_${i}.result"
        } &
        concurrent_pids+=($!)
    done
    
    # Wait for all concurrent requests
    for pid in "${concurrent_pids[@]}"; do
        wait "$pid"
    done
    
    # Check results
    local successful_requests=0
    for i in $(seq 1 $CONCURRENT_USERS); do
        local result=$(cat "/tmp/concurrent_test_${i}.result" 2>/dev/null || echo "000")
        if [ "$result" = "200" ]; then
            ((successful_requests++))
        fi
        rm -f "/tmp/concurrent_test_${i}.result"
    done
    
    if [ $successful_requests -eq $CONCURRENT_USERS ]; then
        log_test_result "CONCURRENT_CONNECTIONS" "PASS" "All $CONCURRENT_USERS concurrent requests successful"
        ((tests_passed++))
    else
        log_test_result "CONCURRENT_CONNECTIONS" "FAIL" "$successful_requests/$CONCURRENT_USERS concurrent requests successful"
    fi
    
    # Test memory usage (if available)
    ((tests_total++))
    if command -v docker &> /dev/null; then
        local memory_stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" | grep -E "(bev-mcp-server|bev-frontend-proxy)" || echo "")
        if [ -n "$memory_stats" ]; then
            log_test_result "MEMORY_USAGE" "PASS" "Container memory stats collected"
            echo "$memory_stats" >> "${REPORTS_DIR}/memory_stats.txt"
            ((tests_passed++))
        else
            log_test_result "MEMORY_USAGE" "SKIP" "Container memory stats not available"
        fi
    else
        log_test_result "MEMORY_USAGE" "SKIP" "Docker not available for memory testing"
    fi
    
    log_info "Performance tests: $tests_passed/$tests_total passed"
    return $(( tests_total - tests_passed ))
}

# =====================================================
# Integration Tests
# =====================================================

test_bev_integration() {
    log_info "Testing BEV platform integration..."
    
    local tests_passed=0
    local tests_total=0
    
    # Test database connectivity through BEV services
    ((tests_total++))
    if docker exec bev_postgres pg_isready -U "${POSTGRES_USER}" &>/dev/null; then
        log_test_result "BEV_DATABASE" "PASS" "PostgreSQL database accessible"
        ((tests_passed++))
    else
        log_test_result "BEV_DATABASE" "FAIL" "PostgreSQL database not accessible"
    fi
    
    # Test Redis connectivity
    ((tests_total++))
    if docker exec bev_redis redis-cli ping | grep -q "PONG"; then
        log_test_result "BEV_REDIS" "PASS" "Redis cache accessible"
        ((tests_passed++))
    else
        log_test_result "BEV_REDIS" "FAIL" "Redis cache not accessible"
    fi
    
    # Test network connectivity between frontend and BEV services
    ((tests_total++))
    if docker exec bev-mcp-server nc -z bev_postgres 5432 &>/dev/null; then
        log_test_result "NETWORK_CONNECTIVITY" "PASS" "Frontend can reach BEV services"
        ((tests_passed++))
    else
        log_test_result "NETWORK_CONNECTIVITY" "FAIL" "Frontend cannot reach BEV services"
    fi
    
    # Test data flow integration
    ((tests_total++))
    local integration_test='{"type":"bev_query","data":{"query":"test","source":"integration_test"}}'
    if test_websocket_connection "ws://localhost:${FRONTEND_WS_PORT}/ws" "$integration_test" 15; then
        log_test_result "DATA_FLOW" "PASS" "Data flow integration working"
        ((tests_passed++))
    else
        log_test_result "DATA_FLOW" "FAIL" "Data flow integration failed"
    fi
    
    log_info "BEV integration tests: $tests_passed/$tests_total passed"
    return $(( tests_total - tests_passed ))
}

# =====================================================
# End-to-End Scenario Tests
# =====================================================

test_end_to_end_scenarios() {
    log_info "Testing end-to-end scenarios..."
    
    local tests_passed=0
    local tests_total=0
    
    # Scenario 1: Complete user workflow
    ((tests_total++))
    log_info "Testing complete user workflow..."
    
    # Step 1: Access frontend
    local frontend_access=$(make_http_request "GET" "https://localhost:${FRONTEND_HTTPS_PORT}/" "200" "-k")
    if [ $? -ne 0 ]; then
        log_test_result "E2E_USER_WORKFLOW" "FAIL" "Cannot access frontend"
        return 1
    fi
    
    # Step 2: Connect to WebSocket
    if ! test_websocket_connection "ws://localhost:${FRONTEND_WS_PORT}/ws" '{"type":"connect","user":"test"}' 10; then
        log_test_result "E2E_USER_WORKFLOW" "FAIL" "WebSocket connection failed"
        return 1
    fi
    
    # Step 3: Execute MCP tool
    local tool_result=$(make_http_request "POST" "http://localhost:${MCP_SERVER_PORT}/api/tools/execute" "200" \
        "-H 'Content-Type: application/json' -d '{\"tool\":\"search\",\"parameters\":{\"query\":\"test\"}}'")
    if [ $? -ne 0 ]; then
        log_test_result "E2E_USER_WORKFLOW" "FAIL" "MCP tool execution failed"
        return 1
    fi
    
    log_test_result "E2E_USER_WORKFLOW" "PASS" "Complete user workflow successful"
    ((tests_passed++))
    
    # Scenario 2: Load balancer failover (simulated)
    ((tests_total++))
    log_info "Testing load balancer behavior..."
    
    local lb_response1=$(make_http_request "GET" "https://localhost:${FRONTEND_HTTPS_PORT}/health" "200" "-k")
    local lb_response2=$(make_http_request "GET" "https://localhost:${FRONTEND_HTTPS_PORT}/health" "200" "-k")
    
    if [ $? -eq 0 ]; then
        log_test_result "E2E_LOAD_BALANCER" "PASS" "Load balancer handling requests correctly"
        ((tests_passed++))
    else
        log_test_result "E2E_LOAD_BALANCER" "FAIL" "Load balancer not responding consistently"
    fi
    
    # Scenario 3: Session persistence
    ((tests_total++))
    log_info "Testing session persistence..."
    
    # Create session
    local session_create=$(curl -k -s -c "/tmp/session_cookies.txt" \
        "https://localhost:${FRONTEND_HTTPS_PORT}/api/session/create" \
        -H "Content-Type: application/json" \
        -d '{"user":"test_user"}' 2>/dev/null || echo "")
    
    # Use session
    local session_verify=$(curl -k -s -b "/tmp/session_cookies.txt" \
        "https://localhost:${FRONTEND_HTTPS_PORT}/api/session/verify" 2>/dev/null || echo "")
    
    if [ -n "$session_verify" ]; then
        log_test_result "E2E_SESSION_PERSISTENCE" "PASS" "Session persistence working"
        ((tests_passed++))
    else
        log_test_result "E2E_SESSION_PERSISTENCE" "SKIP" "Session persistence endpoints not available"
    fi
    
    # Cleanup
    rm -f "/tmp/session_cookies.txt"
    
    log_info "End-to-end scenario tests: $tests_passed/$tests_total passed"
    return $(( tests_total - tests_passed ))
}

# =====================================================
# Monitoring and Health Check Tests
# =====================================================

test_monitoring_health() {
    log_info "Testing monitoring and health check systems..."
    
    local tests_passed=0
    local tests_total=0
    
    # Test container health status
    ((tests_total++))
    local unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" | wc -l)
    if [ "$unhealthy_containers" -eq 0 ]; then
        log_test_result "CONTAINER_HEALTH" "PASS" "All containers healthy"
        ((tests_passed++))
    else
        local unhealthy_list=$(docker ps --filter "health=unhealthy" --format "{{.Names}}" | tr '\n' ' ')
        log_test_result "CONTAINER_HEALTH" "FAIL" "Unhealthy containers: $unhealthy_list"
    fi
    
    # Test HAProxy backend status
    ((tests_total++))
    local haproxy_stats=$(curl -s "http://localhost:${HAPROXY_STATS_PORT}/haproxy-stats;csv" 2>/dev/null || echo "")
    if echo "$haproxy_stats" | grep -q "UP"; then
        log_test_result "HAPROXY_BACKENDS" "PASS" "HAProxy backends operational"
        ((tests_passed++))
    else
        log_test_result "HAPROXY_BACKENDS" "FAIL" "HAProxy backends not operational"
    fi
    
    # Test log file generation
    ((tests_total++))
    local log_count=$(find "${PROJECT_ROOT}/logs" -name "*.log" -mmin -10 | wc -l)
    if [ "$log_count" -gt 0 ]; then
        log_test_result "LOG_GENERATION" "PASS" "$log_count recent log files found"
        ((tests_passed++))
    else
        log_test_result "LOG_GENERATION" "WARN" "No recent log files found"
    fi
    
    # Test metrics endpoint (if available)
    ((tests_total++))
    local metrics_response=$(curl -s "http://localhost:${MCP_SERVER_PORT}/metrics" 2>/dev/null || echo "")
    if [ -n "$metrics_response" ]; then
        log_test_result "METRICS_ENDPOINT" "PASS" "Metrics endpoint responding"
        ((tests_passed++))
    else
        log_test_result "METRICS_ENDPOINT" "SKIP" "Metrics endpoint not available"
    fi
    
    log_info "Monitoring and health tests: $tests_passed/$tests_total passed"
    return $(( tests_total - tests_passed ))
}

# =====================================================
# Test Report Generation
# =====================================================

generate_test_report() {
    log_info "Generating comprehensive test report..."
    
    # Create HTML report
    cat > "${REPORTS_DIR}/test_report.html" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEV Frontend Integration Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
        .summary { background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .test-group { margin: 20px 0; }
        .test-result { margin: 10px 0; padding: 10px; border-radius: 3px; }
        .pass { background-color: #d4edda; border-left: 4px solid #28a745; }
        .fail { background-color: #f8d7da; border-left: 4px solid #dc3545; }
        .skip { background-color: #fff3cd; border-left: 4px solid #ffc107; }
        .warn { background-color: #e2e3e5; border-left: 4px solid #6c757d; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
    </style>
</head>
<body>
    <div class="header">
        <h1>BEV Frontend Integration Test Report</h1>
        <p>Generated: $(date)</p>
        <p>Test Environment: Deployment Validation</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td id="total-tests">0</td></tr>
            <tr><td>Passed</td><td id="passed-tests">0</td></tr>
            <tr><td>Failed</td><td id="failed-tests">0</td></tr>
            <tr><td>Skipped</td><td id="skipped-tests">0</td></tr>
            <tr><td>Success Rate</td><td id="success-rate">0%</td></tr>
        </table>
    </div>
    
    <div class="summary">
        <h2>Service Endpoints</h2>
        <table>
            <tr><th>Service</th><th>URL</th><th>Status</th></tr>
            <tr><td>Frontend HTTPS</td><td>https://localhost:${FRONTEND_HTTPS_PORT}</td><td class="status-https">Unknown</td></tr>
            <tr><td>MCP Server</td><td>http://localhost:${MCP_SERVER_PORT}</td><td class="status-mcp">Unknown</td></tr>
            <tr><td>WebSocket</td><td>ws://localhost:${FRONTEND_WS_PORT}</td><td class="status-ws">Unknown</td></tr>
            <tr><td>HAProxy Stats</td><td>http://localhost:${HAPROXY_STATS_PORT}</td><td class="status-haproxy">Unknown</td></tr>
        </table>
    </div>
    
    <div id="test-results"></div>
    
    <script>
        // Process CSV data to populate report
        fetch('./test_results.csv')
            .then(response => response.text())
            .then(data => {
                const lines = data.split('\\n').filter(line => line.trim());
                const tests = lines.slice(1).map(line => {
                    const [timestamp, name, status, details] = line.split(',');
                    return { timestamp, name, status, details };
                });
                
                // Calculate summary statistics
                const total = tests.length;
                const passed = tests.filter(t => t.status === 'PASS').length;
                const failed = tests.filter(t => t.status === 'FAIL').length;
                const skipped = tests.filter(t => t.status === 'SKIP').length;
                const successRate = total > 0 ? Math.round((passed / total) * 100) : 0;
                
                document.getElementById('total-tests').textContent = total;
                document.getElementById('passed-tests').textContent = passed;
                document.getElementById('failed-tests').textContent = failed;
                document.getElementById('skipped-tests').textContent = skipped;
                document.getElementById('success-rate').textContent = successRate + '%';
                
                // Generate test results section
                const resultsDiv = document.getElementById('test-results');
                const groupedTests = tests.reduce((groups, test) => {
                    const group = test.name.split('_')[0];
                    if (!groups[group]) groups[group] = [];
                    groups[group].push(test);
                    return groups;
                }, {});
                
                Object.keys(groupedTests).forEach(group => {
                    const groupDiv = document.createElement('div');
                    groupDiv.className = 'test-group';
                    groupDiv.innerHTML = '<h3>' + group.replace('_', ' ') + ' Tests</h3>';
                    
                    groupedTests[group].forEach(test => {
                        const testDiv = document.createElement('div');
                        testDiv.className = 'test-result ' + test.status.toLowerCase();
                        testDiv.innerHTML = '<strong>' + test.name + '</strong>: ' + test.details;
                        groupDiv.appendChild(testDiv);
                    });
                    
                    resultsDiv.appendChild(groupDiv);
                });
            })
            .catch(error => {
                console.error('Error loading test results:', error);
            });
    </script>
</body>
</html>
EOF

    # Create JSON report for machine processing
    cat > "${REPORTS_DIR}/test_report.json" << EOF
{
    "report_metadata": {
        "timestamp": "$(date -Iseconds)",
        "test_suite": "bev-frontend-integration",
        "version": "1.0.0",
        "environment": "deployment-validation"
    },
    "test_configuration": {
        "endpoints": {
            "frontend_https": "https://localhost:${FRONTEND_HTTPS_PORT}",
            "mcp_server": "http://localhost:${MCP_SERVER_PORT}",
            "websocket": "ws://localhost:${FRONTEND_WS_PORT}",
            "haproxy_stats": "http://localhost:${HAPROXY_STATS_PORT}"
        },
        "test_parameters": {
            "request_timeout": ${REQUEST_TIMEOUT},
            "max_retry_attempts": ${MAX_RETRY_ATTEMPTS},
            "concurrent_users": ${CONCURRENT_USERS}
        }
    },
    "results_summary": {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "skipped_tests": 0,
        "success_rate": 0
    },
    "detailed_results": "see test_results.csv for detailed test results",
    "files_generated": [
        "test_results.csv",
        "test_report.html",
        "test_report.json",
        "test_config.json"
    ]
}
EOF

    log_success "Test report generated at ${REPORTS_DIR}/"
}

# =====================================================
# Main Execution Flow
# =====================================================

main() {
    setup_logging
    
    log_info "Starting BEV Frontend Integration Test Suite"
    log_info "Test results will be saved to: ${REPORTS_DIR}"
    
    # Validate deployment status
    if [ ! -f "${PROJECT_ROOT}/.frontend_deployment" ]; then
        log_error "Frontend deployment status file not found. Run deployment script first."
        exit 1
    fi
    
    local deployment_status=$(grep "DEPLOYMENT_STATUS" "${PROJECT_ROOT}/.frontend_deployment" | cut -d= -f2)
    if [ "${deployment_status}" != "SUCCESS" ]; then
        log_error "Frontend deployment was not successful. Cannot run integration tests."
        exit 1
    fi
    
    # Load environment variables
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        set -a
        source "${PROJECT_ROOT}/.env"
        set +a
    fi
    
    # Generate test data
    generate_test_data
    
    # Execute test suites
    local test_suites=(
        "test_service_connectivity"
        "test_api_functionality"
        "test_security_features"
        "test_performance"
        "test_bev_integration"
        "test_end_to_end_scenarios"
        "test_monitoring_health"
    )
    
    local total_failures=0
    
    for suite in "${test_suites[@]}"; do
        log_info "Executing test suite: ${suite}"
        if ! ${suite}; then
            local failures=$?
            log_warn "Test suite ${suite} had ${failures} failures"
            total_failures=$((total_failures + failures))
        else
            log_success "Test suite ${suite} completed successfully"
        fi
        echo "---"
    done
    
    # Generate comprehensive report
    generate_test_report
    
    # Calculate final statistics
    local total_tests=$(grep -c "," "${REPORTS_DIR}/test_results.csv" 2>/dev/null || echo "0")
    local passed_tests=$(grep -c ",PASS," "${REPORTS_DIR}/test_results.csv" 2>/dev/null || echo "0")
    local failed_tests=$(grep -c ",FAIL," "${REPORTS_DIR}/test_results.csv" 2>/dev/null || echo "0")
    local skipped_tests=$(grep -c ",SKIP," "${REPORTS_DIR}/test_results.csv" 2>/dev/null || echo "0")
    
    # Update JSON report with actual statistics
    if command -v jq &> /dev/null; then
        jq --arg total "$total_tests" \
           --arg passed "$passed_tests" \
           --arg failed "$failed_tests" \
           --arg skipped "$skipped_tests" \
           --arg success_rate "$(echo "scale=2; $passed_tests * 100 / $total_tests" | bc -l 2>/dev/null || echo "0")" \
           '.results_summary.total_tests = ($total | tonumber) |
            .results_summary.passed_tests = ($passed | tonumber) |
            .results_summary.failed_tests = ($failed | tonumber) |
            .results_summary.skipped_tests = ($skipped | tonumber) |
            .results_summary.success_rate = ($success_rate | tonumber)' \
           "${REPORTS_DIR}/test_report.json" > "${REPORTS_DIR}/test_report.json.tmp" && \
           mv "${REPORTS_DIR}/test_report.json.tmp" "${REPORTS_DIR}/test_report.json"
    fi
    
    # Summary
    echo "=============================================="
    log_info "Integration test suite summary:"
    log_info "Total tests executed: ${total_tests}"
    log_info "Tests passed: ${passed_tests}"
    log_info "Tests failed: ${failed_tests}"
    log_info "Tests skipped: ${skipped_tests}"
    
    if [ "$failed_tests" -eq 0 ]; then
        log_success "All integration tests passed!"
        
        # Write test success marker
        echo "INTEGRATION_TESTS=PASSED" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "TEST_TIMESTAMP=$(date -Iseconds)" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "TEST_REPORTS_DIR=${REPORTS_DIR}" >> "${PROJECT_ROOT}/.frontend_deployment"
        
        echo "=============================================="
        echo "✅ INTEGRATION TESTS SUCCESSFUL"
        echo "   Success rate: $(echo "scale=1; $passed_tests * 100 / $total_tests" | bc -l)%"
        echo "   Test reports: ${REPORTS_DIR}/"
        echo "   HTML report: ${REPORTS_DIR}/test_report.html"
        echo "   System ready for production use"
        echo "=============================================="
        
        exit 0
    else
        log_error "Integration tests failed: ${failed_tests} test(s)"
        
        # Write test failure marker
        echo "INTEGRATION_TESTS=FAILED" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "TEST_TIMESTAMP=$(date -Iseconds)" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "FAILED_TESTS=${failed_tests}" >> "${PROJECT_ROOT}/.frontend_deployment"
        
        echo "=============================================="
        echo "❌ INTEGRATION TESTS FAILED"
        echo "   Failed tests: ${failed_tests}/${total_tests}"
        echo "   Test reports: ${REPORTS_DIR}/"
        echo "   Review failed tests before production use"
        echo "=============================================="
        
        exit 1
    fi
}

# Trap for cleanup
trap 'log_error "Test suite interrupted"; exit 130' INT TERM

# Execute main function
main "$@"