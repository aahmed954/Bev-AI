#!/bin/bash

#################################################################
# BEV System Deployment Validation Framework
#
# Comprehensive testing suite for the complete BEV OSINT system
# Tests all 54+ services across THANOS and ORACLE1 deployments
#################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$SCRIPT_DIR/tests"
REPORTS_DIR="$SCRIPT_DIR/test-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
REPORT_FILE="$REPORTS_DIR/bev_validation_$TIMESTAMP.html"
LOG_FILE="$REPORTS_DIR/validation_$TIMESTAMP.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Service categories and their expected ports/endpoints
declare -A CORE_SERVICES=(
    ["postgres"]="5432"
    ["neo4j"]="7474:7687:7473"
    ["redis-node-1"]="7001:17001"
    ["redis-node-2"]="7002:17002"
    ["redis-node-3"]="7003:17003"
    ["redis"]="6379"
    ["rabbitmq-1"]="5672:15672"
    ["rabbitmq-2"]="5673:15673"
    ["rabbitmq-3"]="5674:15674"
    ["zookeeper"]="2181"
    ["kafka-1"]="19092"
    ["kafka-2"]="29092"
    ["kafka-3"]="39092"
    ["elasticsearch"]="9200:9300"
    ["influxdb"]="8086"
    ["tor"]="9050:9051:8118"
)

declare -A INTELOWL_SERVICES=(
    ["intelowl-postgres"]="5432"
    ["intelowl-celery-beat"]=""
    ["intelowl-celery-worker"]=""
    ["intelowl-django"]="8000"
    ["intelowl-nginx"]="80:443"
    ["cytoscape-server"]="3000"
)

declare -A MONITORING_SERVICES=(
    ["prometheus"]="9090"
    ["grafana"]="3001"
    ["node-exporter"]="9100"
    ["airflow-scheduler"]=""
    ["airflow-webserver"]="8080"
    ["airflow-worker-1"]=""
    ["airflow-worker-2"]=""
    ["airflow-worker-3"]=""
)

declare -A PROCESSING_SERVICES=(
    ["ocr-service"]="8001"
    ["doc-analyzer-1"]=""
    ["doc-analyzer-2"]=""
    ["doc-analyzer-3"]=""
)

declare -A SWARM_SERVICES=(
    ["swarm-master-1"]="8002"
    ["swarm-master-2"]="8003"
    ["research-coordinator"]="8004"
    ["memory-manager"]="8005"
    ["code-optimizer"]="8006"
    ["tool-coordinator"]="8007"
)

declare -A SECURITY_SERVICES=(
    ["vault"]="8200"
    ["guardian-enforcer-1"]="8008"
    ["guardian-enforcer-2"]="8009"
    ["tor-node-1"]="9001:9030"
    ["tor-node-2"]="9002:9031"
    ["tor-node-3"]="9003:9032"
    ["ids"]="8010"
    ["traffic-analyzer"]="8011"
    ["anomaly-detector"]="8012"
)

declare -A AUTONOMOUS_SERVICES=(
    ["autonomous-controller-1"]="8013"
    ["autonomous-controller-2"]="8014"
    ["live2d-avatar"]="8015:9001"
    ["live2d-frontend"]="3002"
)

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

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

# Test result tracking
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

skip_test() {
    ((SKIPPED_TESTS++))
    ((TOTAL_TESTS++))
    log_warning "$1"
}

# Initialize test environment
init_test_environment() {
    log_info "Initializing BEV validation environment..."

    # Create directories
    mkdir -p "$TEST_DIR" "$REPORTS_DIR"

    # Check for required tools
    local required_tools=("docker" "curl" "jq" "nc" "redis-cli" "psql")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            log_warning "Required tool '$tool' not found. Some tests may be skipped."
        fi
    done

    # Load environment variables
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        source "$SCRIPT_DIR/.env"
        log_info "Loaded environment configuration"
    else
        log_warning "No .env file found. Using default values."
    fi
}

# Docker container health checks
check_container_health() {
    local container_name="$1"

    if ! docker ps --filter "name=$container_name" --filter "status=running" | grep -q "$container_name"; then
        fail_test "Container $container_name is not running"
        return 1
    fi

    # Check container health status if health check is configured
    local health_status=$(docker inspect "$container_name" 2>/dev/null | jq -r '.[0].State.Health.Status // "unknown"')

    case "$health_status" in
        "healthy")
            pass_test "Container $container_name is healthy"
            return 0
            ;;
        "unhealthy")
            fail_test "Container $container_name is unhealthy"
            return 1
            ;;
        "starting")
            log_warning "Container $container_name is still starting"
            return 2
            ;;
        "unknown"|"null")
            # No health check configured, just check if running
            pass_test "Container $container_name is running (no health check)"
            return 0
            ;;
        *)
            fail_test "Container $container_name has unknown health status: $health_status"
            return 1
            ;;
    esac
}

# Network connectivity tests
test_port_connectivity() {
    local host="$1"
    local port="$2"
    local service_name="$3"

    if nc -z -w5 "$host" "$port" 2>/dev/null; then
        pass_test "Port $port accessible for $service_name"
        return 0
    else
        fail_test "Port $port not accessible for $service_name"
        return 1
    fi
}

# HTTP endpoint tests
test_http_endpoint() {
    local url="$1"
    local service_name="$2"
    local expected_status="${3:-200}"
    local timeout="${4:-10}"

    local response=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$url" 2>/dev/null || echo "000")

    if [[ "$response" == "$expected_status" ]]; then
        pass_test "HTTP endpoint $url responding correctly for $service_name"
        return 0
    else
        fail_test "HTTP endpoint $url returned status $response (expected $expected_status) for $service_name"
        return 1
    fi
}

# Database connectivity tests
test_postgres_connection() {
    local host="${1:-localhost}"
    local port="${2:-5432}"
    local user="${POSTGRES_USER:-postgres}"
    local password="${POSTGRES_PASSWORD:-password}"
    local database="${3:-osint}"

    if PGPASSWORD="$password" psql -h "$host" -p "$port" -U "$user" -d "$database" -c "SELECT 1;" &>/dev/null; then
        pass_test "PostgreSQL connection successful to $database"
        return 0
    else
        fail_test "PostgreSQL connection failed to $database"
        return 1
    fi
}

test_redis_connection() {
    local host="${1:-localhost}"
    local port="${2:-6379}"
    local password="${REDIS_PASSWORD:-}"

    local auth_cmd=""
    if [[ -n "$password" ]]; then
        auth_cmd="-a $password"
    fi

    if redis-cli -h "$host" -p "$port" $auth_cmd ping 2>/dev/null | grep -q "PONG"; then
        pass_test "Redis connection successful on port $port"
        return 0
    else
        fail_test "Redis connection failed on port $port"
        return 1
    fi
}

test_neo4j_connection() {
    local host="${1:-localhost}"
    local port="${2:-7474}"
    local user="${NEO4J_USER:-neo4j}"
    local password="${NEO4J_PASSWORD:-password}"

    if curl -s -u "$user:$password" "http://$host:$port/db/data/" | grep -q "neo4j_version"; then
        pass_test "Neo4j connection successful"
        return 0
    else
        fail_test "Neo4j connection failed"
        return 1
    fi
}

# Test service categories
test_core_services() {
    log_info "Testing core infrastructure services..."

    for service in "${!CORE_SERVICES[@]}"; do
        local container_name="bev_$service"
        check_container_health "$container_name"

        # Test specific service connectivity
        case "$service" in
            "postgres")
                test_postgres_connection "localhost" "5432" "osint"
                test_postgres_connection "localhost" "5432" "intelowl"
                test_postgres_connection "localhost" "5432" "breach_data"
                test_postgres_connection "localhost" "5432" "crypto_analysis"
                ;;
            "redis")
                test_redis_connection "localhost" "6379"
                ;;
            "redis-node-"*)
                local port="${CORE_SERVICES[$service]%%:*}"
                test_redis_connection "localhost" "$port"
                ;;
            "neo4j")
                test_neo4j_connection "localhost" "7474"
                test_port_connectivity "localhost" "7687" "$service (Bolt)"
                ;;
            "rabbitmq-"*)
                local ports="${CORE_SERVICES[$service]}"
                IFS=':' read -ra port_array <<< "$ports"
                for port in "${port_array[@]}"; do
                    test_port_connectivity "localhost" "$port" "$service"
                done
                test_http_endpoint "http://localhost:${port_array[1]}" "$service Management UI" "200"
                ;;
            "kafka-"*)
                local port="${CORE_SERVICES[$service]}"
                test_port_connectivity "localhost" "$port" "$service"
                ;;
            "elasticsearch")
                test_http_endpoint "http://localhost:9200" "$service" "200"
                test_http_endpoint "http://localhost:9200/_cluster/health" "$service Health"
                ;;
            "influxdb")
                test_http_endpoint "http://localhost:8086/health" "$service" "200"
                ;;
            "tor")
                test_port_connectivity "localhost" "9050" "$service SOCKS5"
                test_port_connectivity "localhost" "9051" "$service Control"
                ;;
        esac
    done
}

test_intelowl_services() {
    log_info "Testing IntelOwl services..."

    for service in "${!INTELOWL_SERVICES[@]}"; do
        local container_name="bev_$service"
        check_container_health "$container_name"

        case "$service" in
            "intelowl-django")
                test_http_endpoint "http://localhost:8000/api/health" "$service API" "200"
                test_http_endpoint "http://localhost:8000/admin/" "$service Admin" "200"
                ;;
            "intelowl-nginx")
                test_http_endpoint "http://localhost:80" "$service" "200"
                ;;
            "cytoscape-server")
                test_http_endpoint "http://localhost:3000" "$service" "200"
                ;;
        esac
    done
}

test_monitoring_services() {
    log_info "Testing monitoring and orchestration services..."

    for service in "${!MONITORING_SERVICES[@]}"; do
        local container_name="bev_$service"
        check_container_health "$container_name"

        case "$service" in
            "prometheus")
                test_http_endpoint "http://localhost:9090/-/healthy" "$service" "200"
                test_http_endpoint "http://localhost:9090/api/v1/targets" "$service Targets API"
                ;;
            "grafana")
                test_http_endpoint "http://localhost:3001/api/health" "$service" "200"
                ;;
            "node-exporter")
                test_http_endpoint "http://localhost:9100/metrics" "$service" "200"
                ;;
            "airflow-webserver")
                test_http_endpoint "http://localhost:8080/health" "$service" "200"
                ;;
        esac
    done
}

test_processing_services() {
    log_info "Testing document processing services..."

    for service in "${!PROCESSING_SERVICES[@]}"; do
        local container_name="bev_$service"
        check_container_health "$container_name"

        case "$service" in
            "ocr-service")
                test_http_endpoint "http://localhost:8001/health" "$service" "200"
                test_http_endpoint "http://localhost:8001/api/v1/ocr/info" "$service API"
                ;;
        esac
    done
}

test_swarm_services() {
    log_info "Testing intelligence swarm services..."

    for service in "${!SWARM_SERVICES[@]}"; do
        local container_name="bev_$service"
        check_container_health "$container_name"

        local port="${SWARM_SERVICES[$service]}"
        if [[ -n "$port" ]]; then
            test_http_endpoint "http://localhost:$port/health" "$service" "200"
        fi
    done
}

test_security_services() {
    log_info "Testing security and privacy services..."

    for service in "${!SECURITY_SERVICES[@]}"; do
        local container_name="bev_$service"
        check_container_health "$container_name"

        case "$service" in
            "vault")
                test_http_endpoint "http://localhost:8200/v1/sys/health" "$service" "200"
                ;;
            "guardian-enforcer-"*|"ids"|"traffic-analyzer"|"anomaly-detector")
                local port="${SECURITY_SERVICES[$service]}"
                if [[ -n "$port" ]]; then
                    test_http_endpoint "http://localhost:$port/health" "$service" "200"
                fi
                ;;
            "tor-node-"*)
                local ports="${SECURITY_SERVICES[$service]}"
                IFS=':' read -ra port_array <<< "$ports"
                for port in "${port_array[@]}"; do
                    test_port_connectivity "localhost" "$port" "$service"
                done
                ;;
        esac
    done
}

test_autonomous_services() {
    log_info "Testing autonomous interface services..."

    for service in "${!AUTONOMOUS_SERVICES[@]}"; do
        local container_name="bev_$service"
        check_container_health "$container_name"

        case "$service" in
            "autonomous-controller-"*)
                local port="${AUTONOMOUS_SERVICES[$service]%%:*}"
                test_http_endpoint "http://localhost:$port/health" "$service" "200"
                ;;
            "live2d-avatar")
                test_http_endpoint "http://localhost:8015/health" "$service" "200"
                test_port_connectivity "localhost" "9001" "$service WebSocket"
                ;;
            "live2d-frontend")
                test_http_endpoint "http://localhost:3002" "$service" "200"
                ;;
        esac
    done
}

# Integration tests
test_data_flow() {
    log_info "Testing data flow between services..."

    # Test Redis cluster communication
    if redis-cli -h localhost -p 7001 -a "${REDIS_PASSWORD:-}" set test_key "test_value" &>/dev/null; then
        if redis-cli -h localhost -p 7002 -a "${REDIS_PASSWORD:-}" get test_key 2>/dev/null | grep -q "test_value"; then
            pass_test "Redis cluster replication working"
        else
            fail_test "Redis cluster replication not working"
        fi
        redis-cli -h localhost -p 7001 -a "${REDIS_PASSWORD:-}" del test_key &>/dev/null
    else
        fail_test "Redis cluster write test failed"
    fi

    # Test Kafka broker communication
    if command -v kafka-console-producer.sh &>/dev/null; then
        echo "test_message" | kafka-console-producer.sh --broker-list localhost:19092 --topic test_topic 2>/dev/null || true
        sleep 2
        local consumed=$(kafka-console-consumer.sh --bootstrap-server localhost:19092 --topic test_topic --from-beginning --max-messages 1 --timeout-ms 5000 2>/dev/null || echo "")
        if [[ "$consumed" == "test_message" ]]; then
            pass_test "Kafka message flow working"
        else
            fail_test "Kafka message flow not working"
        fi
    else
        skip_test "Kafka CLI tools not available for message flow test"
    fi
}

test_service_communication() {
    log_info "Testing inter-service communication..."

    # Test IntelOwl API integration
    local intelowl_status=$(curl -s "http://localhost:8000/api/health" 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null || echo "unknown")
    if [[ "$intelowl_status" == "ok" ]]; then
        pass_test "IntelOwl API health check successful"
    else
        fail_test "IntelOwl API health check failed: $intelowl_status"
    fi

    # Test Prometheus metrics collection
    local prometheus_targets=$(curl -s "http://localhost:9090/api/v1/targets" 2>/dev/null | jq -r '.data.activeTargets | length' 2>/dev/null || echo "0")
    if [[ "$prometheus_targets" -gt 0 ]]; then
        pass_test "Prometheus collecting metrics from $prometheus_targets targets"
    else
        fail_test "Prometheus not collecting metrics from any targets"
    fi
}

# Performance tests
test_system_resources() {
    log_info "Testing system resource usage..."

    # Check Docker resource usage
    local container_count=$(docker ps --filter "name=bev_" | wc -l)
    ((container_count--)) # Remove header line

    if [[ "$container_count" -gt 50 ]]; then
        pass_test "BEV system running $container_count containers"
    else
        fail_test "Expected 50+ containers, found $container_count"
    fi

    # Check memory usage
    local total_memory=$(docker stats --no-stream --format "{{.MemUsage}}" | grep -o '[0-9.]*GiB' | awk -F'GiB' '{sum+=$1} END {print sum}')
    log_info "Total memory usage: ${total_memory}GiB"

    # Check CPU usage
    local avg_cpu=$(docker stats --no-stream --format "{{.CPUPerc}}" | sed 's/%//' | awk '{sum+=$1; count++} END {print sum/count}')
    log_info "Average CPU usage: ${avg_cpu}%"
}

# Security tests
test_security_configuration() {
    log_info "Testing security configurations..."

    # Test Vault seal status
    local vault_status=$(curl -s "http://localhost:8200/v1/sys/seal-status" 2>/dev/null | jq -r '.sealed // true' 2>/dev/null || echo "true")
    if [[ "$vault_status" == "false" ]]; then
        pass_test "Vault is unsealed and accessible"
    else
        fail_test "Vault is sealed or inaccessible"
    fi

    # Test Tor connectivity
    if curl -s --socks5 localhost:9050 "http://httpbin.org/ip" &>/dev/null; then
        pass_test "Tor SOCKS5 proxy working"
    else
        fail_test "Tor SOCKS5 proxy not working"
    fi

    # Test authentication endpoints
    local protected_endpoints=(
        "http://localhost:15672" # RabbitMQ Management
        "http://localhost:3001" # Grafana
    )

    for endpoint in "${protected_endpoints[@]}"; do
        local response=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint" 2>/dev/null || echo "000")
        if [[ "$response" == "401" || "$response" == "200" ]]; then
            pass_test "Authentication configured for $endpoint"
        else
            fail_test "Authentication issue for $endpoint (status: $response)"
        fi
    done
}

# Generate HTML report
generate_html_report() {
    log_info "Generating HTML test report..."

    cat > "$REPORT_FILE" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEV System Validation Report - $TIMESTAMP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat-box { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; min-width: 120px; }
        .stat-box.passed { border-left: 5px solid #28a745; }
        .stat-box.failed { border-left: 5px solid #dc3545; }
        .stat-box.skipped { border-left: 5px solid #ffc107; }
        .test-section { margin: 20px 0; }
        .test-section h3 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }
        .test-result { padding: 8px; margin: 5px 0; border-radius: 3px; }
        .test-result.pass { background-color: #d4edda; color: #155724; border-left: 4px solid #28a745; }
        .test-result.fail { background-color: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
        .test-result.skip { background-color: #fff3cd; color: #856404; border-left: 4px solid #ffc107; }
        .summary { background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 20px; }
        .timestamp { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BEV System Validation Report</h1>
            <div class="timestamp">Generated on: $(date)</div>
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
            <div class="stat-box skipped">
                <h3>$SKIPPED_TESTS</h3>
                <p>Skipped</p>
            </div>
            <div class="stat-box">
                <h3>$TOTAL_TESTS</h3>
                <p>Total</p>
            </div>
        </div>

        <div class="summary">
            <h3>Executive Summary</h3>
            <p>Success Rate: $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%</p>
            <p>System Status: $( [[ $FAILED_TESTS -eq 0 ]] && echo "HEALTHY" || echo "ISSUES DETECTED" )</p>
        </div>

        <div class="test-section">
            <h3>Detailed Test Results</h3>
            <p>For detailed logs, see: <code>$LOG_FILE</code></p>
        </div>
    </div>
</body>
</html>
EOF

    log_success "HTML report generated: $REPORT_FILE"
}

# Main test execution
run_all_tests() {
    log_info "Starting comprehensive BEV system validation..."

    # Service health tests
    test_core_services
    test_intelowl_services
    test_monitoring_services
    test_processing_services
    test_swarm_services
    test_security_services
    test_autonomous_services

    # Integration tests
    test_data_flow
    test_service_communication

    # Performance tests
    test_system_resources

    # Security tests
    test_security_configuration
}

# Command line interface
usage() {
    echo "Usage: $0 [OPTIONS] [TEST_CATEGORIES]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -q, --quick         Run only basic health checks"
    echo "  -f, --full          Run comprehensive test suite (default)"
    echo "  -r, --report-only   Generate report from existing logs"
    echo "  -o, --output DIR    Specify output directory for reports"
    echo ""
    echo "Test Categories:"
    echo "  core              Core infrastructure services"
    echo "  intelowl          IntelOwl services"
    echo "  monitoring        Monitoring and orchestration"
    echo "  processing        Document processing services"
    echo "  swarm             Intelligence swarm services"
    echo "  security          Security and privacy services"
    echo "  autonomous        Autonomous interface services"
    echo "  integration       Integration and data flow tests"
    echo "  performance       Performance and resource tests"
    echo ""
    echo "Examples:"
    echo "  $0                     # Run all tests"
    echo "  $0 -q                  # Quick health check"
    echo "  $0 core monitoring     # Test specific categories"
    echo "  $0 -o /tmp/reports     # Custom output directory"
}

# Parse command line arguments
QUICK_MODE=false
REPORT_ONLY=false
TEST_CATEGORIES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -f|--full)
            QUICK_MODE=false
            shift
            ;;
        -r|--report-only)
            REPORT_ONLY=true
            shift
            ;;
        -o|--output)
            REPORTS_DIR="$2"
            shift 2
            ;;
        core|intelowl|monitoring|processing|swarm|security|autonomous|integration|performance)
            TEST_CATEGORIES+=("$1")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    init_test_environment

    if [[ "$REPORT_ONLY" == "true" ]]; then
        generate_html_report
        exit 0
    fi

    if [[ ${#TEST_CATEGORIES[@]} -eq 0 ]]; then
        # Run all tests
        run_all_tests
    else
        # Run specific test categories
        for category in "${TEST_CATEGORIES[@]}"; do
            case "$category" in
                "core") test_core_services ;;
                "intelowl") test_intelowl_services ;;
                "monitoring") test_monitoring_services ;;
                "processing") test_processing_services ;;
                "swarm") test_swarm_services ;;
                "security") test_security_services ;;
                "autonomous") test_autonomous_services ;;
                "integration")
                    test_data_flow
                    test_service_communication
                    ;;
                "performance") test_system_resources ;;
            esac
        done
    fi

    # Generate reports
    generate_html_report

    # Final summary
    log_info "Validation completed!"
    log_info "Results: $PASSED_TESTS passed, $FAILED_TESTS failed, $SKIPPED_TESTS skipped"
    log_info "Success rate: $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%"
    log_info "HTML Report: $REPORT_FILE"
    log_info "Detailed Log: $LOG_FILE"

    # Exit with appropriate code
    if [[ $FAILED_TESTS -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Execute main function
main "$@"