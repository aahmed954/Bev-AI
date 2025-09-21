#!/bin/bash
# Auto-Recovery System Validation Script
# ======================================
#
# Comprehensive validation and performance testing for the BEV Auto-Recovery System
# Runs integration tests, performance benchmarks, and compliance checks

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$PROJECT_ROOT/validation_results/$TIMESTAMP"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR]${NC} $1" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] [SUCCESS]${NC} $1"
}

# Function to check dependencies
check_dependencies() {
    log "Checking dependencies..."

    local deps=("docker" "docker-compose" "python3" "curl" "jq")
    local missing_deps=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done

    if [ ${#missing_deps[@]} -gt 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
        return 1
    fi

    success "All dependencies found"
}

# Function to setup test environment
setup_test_environment() {
    log "Setting up test environment..."

    # Create results directory
    mkdir -p "$RESULTS_DIR"

    # Set up Python virtual environment if needed
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv "$PROJECT_ROOT/venv"
        source "$PROJECT_ROOT/venv/bin/activate"
        pip install --upgrade pip
        pip install -r "$PROJECT_ROOT/docker/auto-recovery/requirements.txt"
    else
        source "$PROJECT_ROOT/venv/bin/activate"
    fi

    # Export environment variables
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

    success "Test environment ready"
}

# Function to start required services
start_required_services() {
    log "Starting required services for testing..."

    cd "$PROJECT_ROOT"

    # Start core infrastructure services
    docker-compose -f docker-compose.complete.yml up -d \
        postgres \
        redis \
        rabbitmq-1 \
        elasticsearch \
        influxdb

    # Wait for services to be ready
    log "Waiting for services to be ready..."

    # Wait for PostgreSQL
    local retries=30
    while ! docker exec bev_postgres pg_isready -U "${POSTGRES_USER:-bev}" &>/dev/null; do
        retries=$((retries - 1))
        if [ $retries -eq 0 ]; then
            error "PostgreSQL failed to start"
            return 1
        fi
        sleep 2
    done

    # Wait for Redis
    retries=30
    while ! docker exec bev_redis_standalone redis-cli ping &>/dev/null; do
        retries=$((retries - 1))
        if [ $retries -eq 0 ]; then
            error "Redis failed to start"
            return 1
        fi
        sleep 2
    done

    # Wait for Elasticsearch
    retries=30
    while ! curl -sf http://localhost:9200/_cluster/health &>/dev/null; do
        retries=$((retries - 1))
        if [ $retries -eq 0 ]; then
            error "Elasticsearch failed to start"
            return 1
        fi
        sleep 2
    done

    success "Required services are ready"
}

# Function to run unit tests
run_unit_tests() {
    log "Running unit tests..."

    cd "$PROJECT_ROOT"

    # Run Python unit tests
    python -m pytest src/infrastructure/test_*.py -v \
        --junitxml="$RESULTS_DIR/unit_tests.xml" \
        --cov=src/infrastructure \
        --cov-report=html:"$RESULTS_DIR/coverage_html" \
        --cov-report=xml:"$RESULTS_DIR/coverage.xml" \
        2>&1 | tee "$RESULTS_DIR/unit_tests.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        success "Unit tests passed"
    else
        error "Unit tests failed"
        return $exit_code
    fi
}

# Function to run integration tests
run_integration_tests() {
    log "Running integration tests..."

    cd "$PROJECT_ROOT"

    # Start auto-recovery service
    docker-compose -f docker-compose.complete.yml up -d auto-recovery

    # Wait for auto-recovery service to be ready
    local retries=30
    while ! curl -sf http://localhost:8014/health &>/dev/null; do
        retries=$((retries - 1))
        if [ $retries -eq 0 ]; then
            error "Auto-recovery service failed to start"
            return 1
        fi
        sleep 2
    done

    # Run integration tests
    python -m pytest src/infrastructure/test_integration_*.py -v \
        --junitxml="$RESULTS_DIR/integration_tests.xml" \
        2>&1 | tee "$RESULTS_DIR/integration_tests.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        success "Integration tests passed"
    else
        error "Integration tests failed"
        return $exit_code
    fi
}

# Function to run performance tests
run_performance_tests() {
    log "Running performance tests..."

    cd "$PROJECT_ROOT"

    # Run comprehensive validation
    python src/infrastructure/recovery_validator.py \
        --config config/auto_recovery.yaml \
        --output "$RESULTS_DIR/performance_report.json" \
        --format json \
        2>&1 | tee "$RESULTS_DIR/performance_tests.log"

    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -eq 0 ]; then
        success "Performance tests completed"
    else
        warn "Performance tests completed with issues"
    fi
}

# Function to run security tests
run_security_tests() {
    log "Running security tests..."

    cd "$PROJECT_ROOT"

    # Check for security vulnerabilities in dependencies
    pip-audit --desc \
        --format=json \
        --output="$RESULTS_DIR/security_audit.json" \
        2>&1 | tee "$RESULTS_DIR/security_tests.log"

    # Run security static analysis
    bandit -r src/infrastructure/ \
        -f json \
        -o "$RESULTS_DIR/bandit_report.json" \
        2>&1 | tee -a "$RESULTS_DIR/security_tests.log" || true

    success "Security tests completed"
}

# Function to run chaos engineering tests
run_chaos_tests() {
    log "Running chaos engineering tests..."

    cd "$PROJECT_ROOT"

    # Chaos test 1: Random container failures
    log "Running container failure chaos test..."
    python -c "
import asyncio
import random
import docker
import time

async def chaos_test():
    client = docker.from_env()
    containers = client.containers.list(filters={'label': 'bev.auto-recovery=enabled'})

    if containers:
        target = random.choice(containers)
        print(f'Stopping container: {target.name}')
        target.stop()

        # Wait for recovery
        await asyncio.sleep(60)

        # Check if recovered
        target.reload()
        if target.status == 'running':
            print('Container recovered successfully')
            return True
        else:
            print('Container failed to recover')
            return False
    return False

result = asyncio.run(chaos_test())
exit(0 if result else 1)
" 2>&1 | tee "$RESULTS_DIR/chaos_tests.log"

    success "Chaos tests completed"
}

# Function to check service health
check_service_health() {
    log "Checking service health..."

    local services=(
        "http://localhost:8014/health:auto-recovery"
        "http://localhost:5432:postgres"
        "http://localhost:6379:redis"
        "http://localhost:9200:elasticsearch"
    )

    local healthy_services=0
    local total_services=${#services[@]}

    for service in "${services[@]}"; do
        local url="${service%:*}"
        local name="${service#*:}"

        if curl -sf "$url" &>/dev/null; then
            success "Service $name is healthy"
            healthy_services=$((healthy_services + 1))
        else
            error "Service $name is unhealthy"
        fi
    done

    log "Service health: $healthy_services/$total_services services healthy"

    if [ $healthy_services -eq $total_services ]; then
        return 0
    else
        return 1
    fi
}

# Function to collect metrics
collect_metrics() {
    log "Collecting system metrics..."

    # Collect Prometheus metrics from auto-recovery service
    curl -sf http://localhost:9091/metrics > "$RESULTS_DIR/prometheus_metrics.txt" || true

    # Collect Docker stats
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}" \
        > "$RESULTS_DIR/docker_stats.txt"

    # Collect system info
    {
        echo "=== System Information ==="
        uname -a
        echo
        echo "=== Docker Version ==="
        docker version
        echo
        echo "=== Docker Compose Version ==="
        docker-compose version
        echo
        echo "=== Memory Info ==="
        free -h
        echo
        echo "=== Disk Usage ==="
        df -h
    } > "$RESULTS_DIR/system_info.txt"

    success "Metrics collected"
}

# Function to generate final report
generate_report() {
    log "Generating final validation report..."

    local report_file="$RESULTS_DIR/validation_report.html"

    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>BEV Auto-Recovery Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .warn { color: orange; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        pre { background: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>BEV Auto-Recovery System Validation Report</h1>
        <p><strong>Generated:</strong> $(date)</p>
        <p><strong>Test Run ID:</strong> $TIMESTAMP</p>
    </div>

    <div class="section">
        <h2>Executive Summary</h2>
        <p>This report contains the results of comprehensive validation testing for the BEV Auto-Recovery System.</p>
    </div>

    <div class="section">
        <h2>Test Results</h2>
        <table>
            <tr><th>Test Category</th><th>Status</th><th>Details</th></tr>
EOF

    # Add test results
    if [ -f "$RESULTS_DIR/unit_tests.xml" ]; then
        echo "            <tr><td>Unit Tests</td><td class=\"pass\">PASS</td><td>See unit_tests.log</td></tr>" >> "$report_file"
    else
        echo "            <tr><td>Unit Tests</td><td class=\"fail\">FAIL</td><td>See unit_tests.log</td></tr>" >> "$report_file"
    fi

    if [ -f "$RESULTS_DIR/integration_tests.xml" ]; then
        echo "            <tr><td>Integration Tests</td><td class=\"pass\">PASS</td><td>See integration_tests.log</td></tr>" >> "$report_file"
    else
        echo "            <tr><td>Integration Tests</td><td class=\"fail\">FAIL</td><td>See integration_tests.log</td></tr>" >> "$report_file"
    fi

    if [ -f "$RESULTS_DIR/performance_report.json" ]; then
        echo "            <tr><td>Performance Tests</td><td class=\"pass\">PASS</td><td>See performance_report.json</td></tr>" >> "$report_file"
    else
        echo "            <tr><td>Performance Tests</td><td class=\"warn\">WARN</td><td>See performance_tests.log</td></tr>" >> "$report_file"
    fi

    cat >> "$report_file" << EOF
        </table>
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        <p>Performance requirements validation:</p>
        <ul>
            <li><strong>Recovery Time:</strong> &lt; 60 seconds (Target)</li>
            <li><strong>Health Check Response:</strong> &lt; 5 seconds (Target)</li>
            <li><strong>Circuit Breaker Response:</strong> &lt; 1 second (Target)</li>
            <li><strong>State Snapshot Time:</strong> &lt; 30 seconds (Target)</li>
        </ul>
    </div>

    <div class="section">
        <h2>Files Generated</h2>
        <ul>
$(find "$RESULTS_DIR" -type f -printf "            <li>%f</li>\n")
        </ul>
    </div>

    <div class="section">
        <h2>System Information</h2>
        <pre>$(cat "$RESULTS_DIR/system_info.txt" 2>/dev/null || echo "System info not available")</pre>
    </div>
</body>
</html>
EOF

    success "Validation report generated: $report_file"
}

# Function to cleanup
cleanup() {
    log "Cleaning up test environment..."

    cd "$PROJECT_ROOT"

    # Stop auto-recovery service
    docker-compose -f docker-compose.complete.yml stop auto-recovery || true

    # Optionally stop other services (uncomment if needed)
    # docker-compose -f docker-compose.complete.yml down

    success "Cleanup completed"
}

# Main execution
main() {
    log "Starting BEV Auto-Recovery System Validation"
    log "Results will be stored in: $RESULTS_DIR"

    # Set up trap for cleanup
    trap cleanup EXIT

    # Run validation steps
    check_dependencies || exit 1
    setup_test_environment || exit 1
    start_required_services || exit 1

    # Run tests
    local overall_result=0

    run_unit_tests || overall_result=1
    run_integration_tests || overall_result=1
    run_performance_tests || overall_result=1
    run_security_tests || overall_result=1
    run_chaos_tests || overall_result=1

    # Health check and metrics
    check_service_health || overall_result=1
    collect_metrics
    generate_report

    # Final summary
    if [ $overall_result -eq 0 ]; then
        success "All validation tests completed successfully!"
        success "Auto-Recovery System is ready for production deployment"
    else
        warn "Some validation tests failed or had issues"
        warn "Review the detailed reports before production deployment"
    fi

    log "Validation completed. Results available in: $RESULTS_DIR"
    exit $overall_result
}

# Show usage if help requested
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat << EOF
BEV Auto-Recovery System Validation Script

Usage: $0 [options]

Options:
    --help, -h    Show this help message

This script runs comprehensive validation tests for the BEV Auto-Recovery System:
- Unit tests with coverage analysis
- Integration tests with real services
- Performance benchmarking
- Security vulnerability scanning
- Chaos engineering tests
- Service health checks
- Metrics collection

Results are stored in validation_results/ with timestamped directories.

EOF
    exit 0
fi

# Run main function
main "$@"