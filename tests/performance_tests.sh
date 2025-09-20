#!/bin/bash

#################################################################
# BEV Performance Test Suite
#
# Load testing and scalability validation for BEV system
# Tests all major services under various load conditions
#################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_DIR/test-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$REPORTS_DIR/performance_tests_$TIMESTAMP.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test configuration
CONCURRENT_USERS=10
TEST_DURATION=60
RAMP_UP_TIME=30
REQUEST_TIMEOUT=30

# Performance thresholds
MAX_RESPONSE_TIME=5000  # milliseconds
MAX_ERROR_RATE=5        # percentage
MIN_THROUGHPUT=10       # requests per second

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Load environment
if [[ -f "$PROJECT_DIR/.env" ]]; then
    source "$PROJECT_DIR/.env"
fi

# Utility functions
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

pass_test() {
    ((PASSED_TESTS++))
    ((TOTAL_TESTS++))
    log_success "$1"
}

fail_test() {
    ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
    log_error "$1"
}

# Check required tools
check_dependencies() {
    local required_tools=("curl" "jq" "ab" "siege")
    local missing_tools=()

    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_warning "Missing tools: ${missing_tools[*]}"
        log_warning "Installing missing tools..."

        # Try to install missing tools
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq
            for tool in "${missing_tools[@]}"; do
                case "$tool" in
                    "ab") sudo apt-get install -y apache2-utils ;;
                    "siege") sudo apt-get install -y siege ;;
                    *) sudo apt-get install -y "$tool" ;;
                esac
            done
        elif command -v yum &> /dev/null; then
            for tool in "${missing_tools[@]}"; do
                case "$tool" in
                    "ab") sudo yum install -y httpd-tools ;;
                    *) sudo yum install -y "$tool" ;;
                esac
            done
        else
            log_error "Cannot install missing tools automatically. Please install: ${missing_tools[*]}"
            return 1
        fi
    fi

    log_info "All required tools are available"
}

# Generate test data
generate_test_data() {
    local test_data_dir="$SCRIPT_DIR/test_data"
    mkdir -p "$test_data_dir"

    # Generate various sized test files
    log_info "Generating test data..."

    # Small document (1KB)
    dd if=/dev/urandom of="$test_data_dir/small_doc.txt" bs=1024 count=1 2>/dev/null
    echo "Small test document for performance testing" > "$test_data_dir/small_doc.txt"

    # Medium document (100KB)
    dd if=/dev/urandom of="$test_data_dir/medium_doc.txt" bs=1024 count=100 2>/dev/null

    # Large document (1MB)
    dd if=/dev/urandom of="$test_data_dir/large_doc.txt" bs=1024 count=1024 2>/dev/null

    # Generate JSON payloads for API testing
    cat > "$test_data_dir/api_payload_small.json" << 'EOF'
{
    "target": "192.168.1.100",
    "analysis_type": "basic",
    "priority": "normal"
}
EOF

    cat > "$test_data_dir/api_payload_large.json" << 'EOF'
{
    "target": "test-domain.com",
    "analysis_type": "comprehensive",
    "priority": "high",
    "parameters": {
        "subdomain_enumeration": true,
        "port_scanning": true,
        "vulnerability_assessment": true,
        "threat_intelligence": true,
        "social_media_analysis": true,
        "financial_analysis": false,
        "deep_web_search": true,
        "correlation_analysis": true
    },
    "depth": "maximum",
    "timeout": 3600,
    "callbacks": [
        "http://localhost:8080/callback1",
        "http://localhost:8080/callback2"
    ],
    "metadata": {
        "case_id": "CASE-2024-001",
        "analyst": "test_user",
        "department": "cyber_intelligence",
        "classification": "restricted",
        "retention_period": 365
    }
}
EOF

    log_info "Test data generated successfully"
}

# Basic performance metrics collection
collect_system_metrics() {
    local metrics_file="$REPORTS_DIR/system_metrics_$TIMESTAMP.json"

    log_info "Collecting baseline system metrics..."

    # CPU usage
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')

    # Memory usage
    local memory_info=$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')

    # Disk usage
    local disk_usage=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')

    # Network connections
    local connections=$(ss -tuln | wc -l)

    # Docker stats
    local docker_stats=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}")

    cat > "$metrics_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "system": {
        "cpu_usage_percent": "$cpu_usage",
        "memory_usage_percent": "$memory_info",
        "disk_usage_percent": "$disk_usage",
        "network_connections": "$connections"
    },
    "docker": {
        "stats": "See docker_stats_$TIMESTAMP.txt"
    }
}
EOF

    echo "$docker_stats" > "$REPORTS_DIR/docker_stats_$TIMESTAMP.txt"

    log_info "System metrics collected"
}

# HTTP load testing with Apache Bench
run_ab_load_test() {
    local url="$1"
    local test_name="$2"
    local requests="${3:-1000}"
    local concurrency="${4:-10}"

    log_info "Running Apache Bench load test for $test_name..."

    local ab_output=$(ab -n "$requests" -c "$concurrency" -k -t "$TEST_DURATION" "$url" 2>&1)
    local ab_results_file="$REPORTS_DIR/ab_results_${test_name}_$TIMESTAMP.txt"
    echo "$ab_output" > "$ab_results_file"

    # Parse results
    local requests_per_sec=$(echo "$ab_output" | grep "Requests per second" | awk '{print $4}')
    local avg_response_time=$(echo "$ab_output" | grep "Time per request" | head -1 | awk '{print $4}')
    local failed_requests=$(echo "$ab_output" | grep "Failed requests" | awk '{print $3}')
    local total_requests=$(echo "$ab_output" | grep "Complete requests" | awk '{print $3}')

    # Calculate error rate
    local error_rate=0
    if [[ "$total_requests" -gt 0 ]]; then
        error_rate=$(( (failed_requests * 100) / total_requests ))
    fi

    log_info "$test_name Results:"
    log_info "  Requests per second: $requests_per_sec"
    log_info "  Average response time: ${avg_response_time}ms"
    log_info "  Failed requests: $failed_requests"
    log_info "  Error rate: ${error_rate}%"

    # Evaluate against thresholds
    if (( $(echo "$avg_response_time < $MAX_RESPONSE_TIME" | bc -l) )); then
        pass_test "$test_name: Response time within threshold (${avg_response_time}ms < ${MAX_RESPONSE_TIME}ms)"
    else
        fail_test "$test_name: Response time exceeded threshold (${avg_response_time}ms > ${MAX_RESPONSE_TIME}ms)"
    fi

    if [[ "$error_rate" -le "$MAX_ERROR_RATE" ]]; then
        pass_test "$test_name: Error rate within threshold (${error_rate}% <= ${MAX_ERROR_RATE}%)"
    else
        fail_test "$test_name: Error rate exceeded threshold (${error_rate}% > ${MAX_ERROR_RATE}%)"
    fi

    if (( $(echo "$requests_per_sec >= $MIN_THROUGHPUT" | bc -l) )); then
        pass_test "$test_name: Throughput meets minimum requirement (${requests_per_sec} >= ${MIN_THROUGHPUT} req/s)"
    else
        fail_test "$test_name: Throughput below minimum requirement (${requests_per_sec} < ${MIN_THROUGHPUT} req/s)"
    fi
}

# Siege load testing
run_siege_load_test() {
    local url="$1"
    local test_name="$2"
    local users="${3:-10}"
    local duration="${4:-60s}"

    log_info "Running Siege load test for $test_name..."

    local siege_output=$(siege -c "$users" -t "$duration" "$url" 2>&1)
    local siege_results_file="$REPORTS_DIR/siege_results_${test_name}_$TIMESTAMP.txt"
    echo "$siege_output" > "$siege_results_file"

    # Parse results
    local availability=$(echo "$siege_output" | grep "Availability" | awk '{print $2}' | sed 's/%//')
    local response_time=$(echo "$siege_output" | grep "Response time" | awk '{print $3}')
    local throughput=$(echo "$siege_output" | grep "Throughput" | awk '{print $2}')
    local successful_transactions=$(echo "$siege_output" | grep "Successful transactions" | awk '{print $3}')
    local failed_transactions=$(echo "$siege_output" | grep "Failed transactions" | awk '{print $3}')

    log_info "$test_name Siege Results:"
    log_info "  Availability: ${availability}%"
    log_info "  Response time: ${response_time}s"
    log_info "  Throughput: ${throughput} MB/sec"
    log_info "  Successful transactions: $successful_transactions"
    log_info "  Failed transactions: $failed_transactions"

    # Evaluate results
    if (( $(echo "$availability >= 95" | bc -l) )); then
        pass_test "$test_name: High availability maintained (${availability}% >= 95%)"
    else
        fail_test "$test_name: Low availability detected (${availability}% < 95%)"
    fi
}

# Database performance testing
test_database_performance() {
    log_info "Testing database performance..."

    # PostgreSQL performance test
    if command -v pgbench &> /dev/null; then
        log_info "Running PostgreSQL performance test..."

        # Initialize pgbench database
        PGPASSWORD="${POSTGRES_PASSWORD:-password}" pgbench -h localhost -p 5432 -U "${POSTGRES_USER:-postgres}" -i -s 10 osint 2>/dev/null || {
            log_warning "Could not initialize pgbench database"
            return
        }

        # Run pgbench test
        local pgbench_output=$(PGPASSWORD="${POSTGRES_PASSWORD:-password}" pgbench -h localhost -p 5432 -U "${POSTGRES_USER:-postgres}" -c 10 -j 2 -t 1000 osint 2>&1)
        local pgbench_results_file="$REPORTS_DIR/pgbench_results_$TIMESTAMP.txt"
        echo "$pgbench_output" > "$pgbench_results_file"

        local tps=$(echo "$pgbench_output" | grep "tps" | awk '{print $3}')
        log_info "PostgreSQL TPS: $tps"

        if (( $(echo "$tps > 100" | bc -l) )); then
            pass_test "PostgreSQL performance acceptable (TPS: $tps)"
        else
            fail_test "PostgreSQL performance below expectation (TPS: $tps)"
        fi
    else
        log_warning "pgbench not available for PostgreSQL testing"
    fi

    # Redis performance test
    if command -v redis-benchmark &> /dev/null; then
        log_info "Running Redis performance test..."

        local redis_output=$(redis-benchmark -h localhost -p 6379 -a "${REDIS_PASSWORD:-}" -n 10000 -c 10 -q 2>&1)
        local redis_results_file="$REPORTS_DIR/redis_benchmark_$TIMESTAMP.txt"
        echo "$redis_output" > "$redis_results_file"

        local redis_ops=$(echo "$redis_output" | grep "GET" | awk '{print $2}')
        log_info "Redis GET operations per second: $redis_ops"

        if [[ "$redis_ops" -gt 10000 ]]; then
            pass_test "Redis performance acceptable (GET OPS: $redis_ops)"
        else
            fail_test "Redis performance below expectation (GET OPS: $redis_ops)"
        fi
    else
        log_warning "redis-benchmark not available for Redis testing"
    fi
}

# Test core service performance
test_core_services_performance() {
    log_info "Testing core services performance..."

    # Elasticsearch performance
    run_ab_load_test "http://localhost:9200/" "Elasticsearch_Health" 500 5

    # Test Elasticsearch search performance
    local search_query='{"query":{"match_all":{}}}'
    local es_search_time=$(time (curl -s -X POST -H "Content-Type: application/json" -d "$search_query" "http://localhost:9200/_search" > /dev/null) 2>&1 | grep real | awk '{print $2}')
    log_info "Elasticsearch search time: $es_search_time"

    # InfluxDB performance
    run_ab_load_test "http://localhost:8086/health" "InfluxDB_Health" 300 5

    # Neo4j performance
    run_ab_load_test "http://localhost:7474/db/data/" "Neo4j_API" 200 3

    # Test message queue performance
    test_message_queue_performance
}

# Test message queue performance
test_message_queue_performance() {
    log_info "Testing message queue performance..."

    # RabbitMQ management API performance
    local rabbitmq_auth="${RABBITMQ_USER:-admin}:${RABBITMQ_PASSWORD:-password}"
    local rabbitmq_response_time=$(curl -w "%{time_total}" -s -o /dev/null -u "$rabbitmq_auth" "http://localhost:15672/api/overview")

    log_info "RabbitMQ management API response time: ${rabbitmq_response_time}s"

    if (( $(echo "$rabbitmq_response_time < 2.0" | bc -l) )); then
        pass_test "RabbitMQ management API responsive (${rabbitmq_response_time}s)"
    else
        fail_test "RabbitMQ management API slow (${rabbitmq_response_time}s)"
    fi

    # Test Kafka performance if tools available
    if command -v kafka-producer-perf-test.sh &> /dev/null; then
        log_info "Running Kafka performance test..."

        local kafka_output=$(kafka-producer-perf-test.sh --topic perf-test --num-records 10000 --record-size 1024 --throughput 1000 --producer-props bootstrap.servers=localhost:19092 2>&1)
        local kafka_results_file="$REPORTS_DIR/kafka_perf_$TIMESTAMP.txt"
        echo "$kafka_output" > "$kafka_results_file"

        local kafka_throughput=$(echo "$kafka_output" | grep "records/sec" | awk '{print $2}')
        log_info "Kafka throughput: $kafka_throughput records/sec"

        if [[ "$kafka_throughput" -gt 500 ]]; then
            pass_test "Kafka performance acceptable ($kafka_throughput records/sec)"
        else
            fail_test "Kafka performance below expectation ($kafka_throughput records/sec)"
        fi
    else
        log_warning "Kafka performance tools not available"
    fi
}

# Test IntelOwl performance
test_intelowl_performance() {
    log_info "Testing IntelOwl performance..."

    # Test IntelOwl API health endpoint
    run_ab_load_test "http://localhost:8000/api/health" "IntelOwl_Health" 200 5

    # Test IntelOwl web interface
    run_ab_load_test "http://localhost:80/" "IntelOwl_Web" 100 3

    # Test job submission performance
    local job_submission_times=()
    local test_payload="$SCRIPT_DIR/test_data/api_payload_small.json"

    for i in {1..10}; do
        local start_time=$(date +%s.%N)
        curl -s -X POST -H "Content-Type: application/json" -d @"$test_payload" "http://localhost:8000/api/jobs" > /dev/null
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        job_submission_times+=("$duration")
    done

    # Calculate average job submission time
    local total_time=0
    for time in "${job_submission_times[@]}"; do
        total_time=$(echo "$total_time + $time" | bc)
    done
    local avg_submission_time=$(echo "scale=3; $total_time / ${#job_submission_times[@]}" | bc)

    log_info "Average job submission time: ${avg_submission_time}s"

    if (( $(echo "$avg_submission_time < 2.0" | bc -l) )); then
        pass_test "IntelOwl job submission responsive (${avg_submission_time}s)"
    else
        fail_test "IntelOwl job submission slow (${avg_submission_time}s)"
    fi
}

# Test document processing performance
test_document_processing_performance() {
    log_info "Testing document processing performance..."

    # Test OCR service performance
    run_ab_load_test "http://localhost:8001/health" "OCR_Service_Health" 100 3

    # Test document upload performance
    local test_files=("$SCRIPT_DIR/test_data/small_doc.txt" "$SCRIPT_DIR/test_data/medium_doc.txt")

    for test_file in "${test_files[@]}"; do
        local file_size=$(du -h "$test_file" | cut -f1)
        log_info "Testing document upload performance for $file_size file..."

        local upload_times=()
        for i in {1..5}; do
            local start_time=$(date +%s.%N)
            curl -s -X POST -F "file=@$test_file" "http://localhost:8001/api/v1/ocr/process" > /dev/null
            local end_time=$(date +%s.%N)
            local duration=$(echo "$end_time - $start_time" | bc)
            upload_times+=("$duration")
        done

        # Calculate average upload time
        local total_time=0
        for time in "${upload_times[@]}"; do
            total_time=$(echo "$total_time + $time" | bc)
        done
        local avg_upload_time=$(echo "scale=3; $total_time / ${#upload_times[@]}" | bc)

        log_info "Average upload time for $file_size file: ${avg_upload_time}s"

        local threshold=10.0
        if [[ "$file_size" == *"K" ]]; then
            threshold=5.0
        fi

        if (( $(echo "$avg_upload_time < $threshold" | bc -l) )); then
            pass_test "Document upload performance acceptable for $file_size file (${avg_upload_time}s)"
        else
            fail_test "Document upload performance slow for $file_size file (${avg_upload_time}s)"
        fi
    done
}

# Test swarm services performance
test_swarm_services_performance() {
    log_info "Testing swarm services performance..."

    # Test swarm masters
    run_ab_load_test "http://localhost:8002/health" "Swarm_Master_1" 100 3
    run_ab_load_test "http://localhost:8003/health" "Swarm_Master_2" 100 3

    # Test research coordinator
    run_ab_load_test "http://localhost:8004/health" "Research_Coordinator" 100 3

    # Test memory manager
    run_ab_load_test "http://localhost:8005/health" "Memory_Manager" 100 3

    # Test memory search performance
    local search_payload='{"query": "test search", "limit": 10}'
    local memory_search_times=()

    for i in {1..5}; do
        local start_time=$(date +%s.%N)
        curl -s -X POST -H "Content-Type: application/json" -d "$search_payload" "http://localhost:8005/api/v1/memory/search" > /dev/null
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        memory_search_times+=("$duration")
    done

    local total_time=0
    for time in "${memory_search_times[@]}"; do
        total_time=$(echo "$total_time + $time" | bc)
    done
    local avg_search_time=$(echo "scale=3; $total_time / ${#memory_search_times[@]}" | bc)

    log_info "Average memory search time: ${avg_search_time}s"

    if (( $(echo "$avg_search_time < 3.0" | bc -l) )); then
        pass_test "Memory search performance acceptable (${avg_search_time}s)"
    else
        fail_test "Memory search performance slow (${avg_search_time}s)"
    fi
}

# Test security services performance
test_security_services_performance() {
    log_info "Testing security services performance..."

    # Test Vault performance
    run_ab_load_test "http://localhost:8200/v1/sys/health" "Vault_Health" 100 3

    # Test Guardian enforcers
    run_ab_load_test "http://localhost:8008/health" "Guardian_Enforcer_1" 50 2
    run_ab_load_test "http://localhost:8009/health" "Guardian_Enforcer_2" 50 2

    # Test IDS performance
    run_ab_load_test "http://localhost:8010/health" "IDS_Service" 50 2

    # Test traffic analyzer
    run_ab_load_test "http://localhost:8011/health" "Traffic_Analyzer" 50 2

    # Test anomaly detector
    run_ab_load_test "http://localhost:8012/health" "Anomaly_Detector" 50 2
}

# Test monitoring services performance
test_monitoring_services_performance() {
    log_info "Testing monitoring services performance..."

    # Test Prometheus performance
    run_ab_load_test "http://localhost:9090/-/healthy" "Prometheus_Health" 200 5

    # Test Prometheus query performance
    local query_times=()
    local prometheus_query="up"

    for i in {1..10}; do
        local start_time=$(date +%s.%N)
        curl -s "http://localhost:9090/api/v1/query?query=$prometheus_query" > /dev/null
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc)
        query_times+=("$duration")
    done

    local total_time=0
    for time in "${query_times[@]}"; do
        total_time=$(echo "$total_time + $time" | bc)
    done
    local avg_query_time=$(echo "scale=3; $total_time / ${#query_times[@]}" | bc)

    log_info "Average Prometheus query time: ${avg_query_time}s"

    if (( $(echo "$avg_query_time < 1.0" | bc -l) )); then
        pass_test "Prometheus query performance acceptable (${avg_query_time}s)"
    else
        fail_test "Prometheus query performance slow (${avg_query_time}s)"
    fi

    # Test Grafana performance
    run_ab_load_test "http://localhost:3001/api/health" "Grafana_Health" 100 3

    # Test Node Exporter
    run_ab_load_test "http://localhost:9100/metrics" "Node_Exporter" 100 5
}

# Test autonomous services performance
test_autonomous_services_performance() {
    log_info "Testing autonomous services performance..."

    # Test autonomous controllers
    run_ab_load_test "http://localhost:8013/health" "Autonomous_Controller_1" 50 2
    run_ab_load_test "http://localhost:8014/health" "Autonomous_Controller_2" 50 2

    # Test Live2D avatar
    run_ab_load_test "http://localhost:8015/health" "Live2D_Avatar" 50 2

    # Test frontend performance
    run_ab_load_test "http://localhost:3002/" "Live2D_Frontend" 100 3
}

# Stress testing
run_stress_tests() {
    log_info "Running stress tests..."

    # High load test on critical services
    local critical_services=(
        "http://localhost:8000/api/health:IntelOwl_Stress"
        "http://localhost:9200/:Elasticsearch_Stress"
        "http://localhost:9090/-/healthy:Prometheus_Stress"
    )

    for service_info in "${critical_services[@]}"; do
        IFS=':' read -ra service_parts <<< "$service_info"
        local url="${service_parts[0]}"
        local name="${service_parts[1]}"

        log_info "Running stress test for $name..."

        # Gradual load increase
        for concurrency in 5 10 20 50; do
            log_info "Testing $name with $concurrency concurrent users..."
            run_siege_load_test "$url" "${name}_Stress_${concurrency}" "$concurrency" "30s"
            sleep 10  # Recovery time between tests
        done
    done
}

# Resource utilization monitoring during tests
monitor_resource_utilization() {
    log_info "Monitoring resource utilization during performance tests..."

    local monitor_file="$REPORTS_DIR/resource_monitor_$TIMESTAMP.log"

    # Start background monitoring
    {
        while true; do
            echo "$(date): CPU=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'), Memory=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')%, Disk=$(df -h / | awk 'NR==2 {print $5}')"
            sleep 5
        done
    } > "$monitor_file" &

    local monitor_pid=$!

    # Run performance tests
    test_core_services_performance
    test_intelowl_performance
    test_document_processing_performance
    test_swarm_services_performance
    test_security_services_performance
    test_monitoring_services_performance
    test_autonomous_services_performance

    # Stop monitoring
    kill $monitor_pid 2>/dev/null || true

    log_info "Resource monitoring completed. See $monitor_file for details."
}

# Scalability testing
test_scalability() {
    log_info "Testing system scalability..."

    # Test with increasing load
    local base_url="http://localhost:8000/api/health"
    local concurrent_users=(1 5 10 25 50 100)

    for users in "${concurrent_users[@]}"; do
        log_info "Testing scalability with $users concurrent users..."

        local start_time=$(date +%s.%N)
        run_siege_load_test "$base_url" "Scalability_${users}_users" "$users" "60s"
        local end_time=$(date +%s.%N)

        local test_duration=$(echo "$end_time - $start_time" | bc)
        log_info "Test with $users users completed in ${test_duration}s"

        # Brief recovery period
        sleep 30
    done
}

# Network performance testing
test_network_performance() {
    log_info "Testing network performance..."

    # Test network latency between services
    local services=("localhost:5432" "localhost:7474" "localhost:9200" "localhost:8000")

    for service in "${services[@]}"; do
        local host=$(echo "$service" | cut -d':' -f1)
        local port=$(echo "$service" | cut -d':' -f2)

        local ping_time=$(ping -c 5 "$host" 2>/dev/null | tail -1 | awk -F'/' '{print $5}' || echo "N/A")
        log_info "Average ping time to $service: ${ping_time}ms"

        # TCP connection test
        local tcp_time=$(time (echo > /dev/tcp/"$host"/"$port") 2>&1 | grep real | awk '{print $2}' || echo "N/A")
        log_info "TCP connection time to $service: $tcp_time"
    done

    # Test throughput with large file transfer
    if command -v iperf3 &> /dev/null; then
        log_info "Testing network throughput with iperf3..."
        # This would require iperf3 server setup
        log_warning "iperf3 server not configured for throughput testing"
    fi
}

# Generate performance report
generate_performance_report() {
    local report_file="$REPORTS_DIR/performance_report_$TIMESTAMP.html"

    log_info "Generating performance test report..."

    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEV Performance Test Report - $TIMESTAMP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat-box { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; min-width: 120px; }
        .stat-box.passed { border-left: 5px solid #28a745; }
        .stat-box.failed { border-left: 5px solid #dc3545; }
        .section { margin: 20px 0; }
        .section h3 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }
        .metric { background: #f8f9fa; padding: 10px; margin: 5px 0; border-left: 3px solid #007bff; }
        .threshold { color: #666; font-size: 0.9em; }
        .summary { background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BEV Performance Test Report</h1>
            <div>Generated on: $(date)</div>
            <div>Test Duration: $TEST_DURATION seconds</div>
            <div>Concurrent Users: $CONCURRENT_USERS</div>
        </div>

        <div class="stats">
            <div class="stat-box passed">
                <h3>$PASSED_TESTS</h3>
                <p>Passed</p>
            </div>
            <div class="stat-box failed">
                <h3>$FAILED_TESTS</h3>
                <p>Failed</p>
            </div>
            <div class="stat-box">
                <h3>$TOTAL_TESTS</h3>
                <p>Total</p>
            </div>
            <div class="stat-box">
                <h3>$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%</h3>
                <p>Success Rate</p>
            </div>
        </div>

        <div class="section">
            <h3>Performance Thresholds</h3>
            <div class="metric">
                <strong>Maximum Response Time:</strong> ${MAX_RESPONSE_TIME}ms
                <div class="threshold">Services should respond within this time limit</div>
            </div>
            <div class="metric">
                <strong>Maximum Error Rate:</strong> ${MAX_ERROR_RATE}%
                <div class="threshold">Acceptable percentage of failed requests</div>
            </div>
            <div class="metric">
                <strong>Minimum Throughput:</strong> ${MIN_THROUGHPUT} req/s
                <div class="threshold">Minimum requests per second requirement</div>
            </div>
        </div>

        <div class="section">
            <h3>Test Categories Executed</h3>
            <ul>
                <li>Core Services Performance (Database, Message Queues, Storage)</li>
                <li>IntelOwl Intelligence Platform Performance</li>
                <li>Document Processing Pipeline Performance</li>
                <li>Swarm Intelligence Services Performance</li>
                <li>Security Services Performance</li>
                <li>Monitoring and Metrics Performance</li>
                <li>Autonomous Interface Performance</li>
                <li>Network and Connectivity Performance</li>
                <li>Scalability and Stress Testing</li>
            </ul>
        </div>

        <div class="summary">
            <h3>Performance Summary</h3>
            <p>Overall System Performance: $( [[ $FAILED_TESTS -eq 0 ]] && echo "EXCELLENT" || echo "NEEDS OPTIMIZATION" )</p>
            <p>Detailed test results and metrics available in log files.</p>
            <p>Detailed logs: <code>$LOG_FILE</code></p>
        </div>

        <div class="section">
            <h3>Recommendations</h3>
            <ul>
                <li>Monitor response times during peak usage periods</li>
                <li>Consider horizontal scaling for services showing high latency</li>
                <li>Implement caching strategies for frequently accessed data</li>
                <li>Regular performance testing should be part of the CI/CD pipeline</li>
                <li>Set up alerting for performance threshold violations</li>
            </ul>
        </div>
    </div>
</body>
</html>
EOF

    log_success "Performance report generated: $report_file"
}

# Main execution
main() {
    log_info "Starting BEV performance test suite..."

    mkdir -p "$REPORTS_DIR"

    # Check dependencies and setup
    check_dependencies
    generate_test_data
    collect_system_metrics

    # Run performance tests with resource monitoring
    monitor_resource_utilization

    # Additional specialized tests
    test_database_performance
    test_network_performance

    # Stress and scalability tests
    run_stress_tests
    test_scalability

    # Generate comprehensive report
    generate_performance_report

    # Cleanup
    rm -rf "$SCRIPT_DIR/test_data" 2>/dev/null || true

    log_info "Performance testing completed!"
    log_info "Results: $PASSED_TESTS passed, $FAILED_TESTS failed"
    log_info "Success rate: $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%"
    log_info "Detailed logs: $LOG_FILE"

    if [[ $FAILED_TESTS -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Command line interface
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--users)
                CONCURRENT_USERS="$2"
                shift 2
                ;;
            -d|--duration)
                TEST_DURATION="$2"
                shift 2
                ;;
            -t|--timeout)
                REQUEST_TIMEOUT="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  -u, --users NUMBER     Number of concurrent users (default: $CONCURRENT_USERS)"
                echo "  -d, --duration SECONDS Test duration in seconds (default: $TEST_DURATION)"
                echo "  -t, --timeout SECONDS  Request timeout in seconds (default: $REQUEST_TIMEOUT)"
                echo "  -h, --help             Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    main "$@"
fi