#!/bin/bash

#################################################################
# BEV Complete Test Suite Runner
#
# Master test runner for comprehensive BEV system validation
# Executes all test categories and generates unified reports
#################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TESTS_DIR="$SCRIPT_DIR/tests"
REPORTS_DIR="$SCRIPT_DIR/test-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_REPORT="$REPORTS_DIR/bev_complete_test_report_$TIMESTAMP.html"
MASTER_LOG="$REPORTS_DIR/master_test_log_$TIMESTAMP.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Test configuration
RUN_VALIDATION=true
RUN_INTEGRATION=true
RUN_PERFORMANCE=true
RUN_SECURITY=true
RUN_MONITORING=true
PARALLEL_EXECUTION=false
FAIL_FAST=false
GENERATE_REPORT=true

# Test results tracking
declare -A TEST_RESULTS
declare -A TEST_DURATIONS
declare -A TEST_LOGS

# Load environment
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    source "$SCRIPT_DIR/.env"
fi

# Utility functions
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MASTER_LOG"
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

log_banner() {
    log "${MAGENTA}[BANNER]${NC} $1"
}

# Initialize test environment
init_test_environment() {
    log_banner "Initializing BEV Test Environment"

    mkdir -p "$REPORTS_DIR"

    # Check if Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker before running tests."
        exit 1
    fi

    # Check if BEV services are running
    local running_containers=$(docker ps --filter "name=bev_" --format "{{.Names}}" | wc -l)

    if [[ "$running_containers" -lt 10 ]]; then
        log_warning "Only $running_containers BEV containers are running. Some tests may fail."
        read -p "Continue anyway? (y/N): " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_info "Found $running_containers BEV containers running"
    fi

    # Check test script availability
    check_test_scripts

    log_success "Test environment initialized"
}

# Check test script availability
check_test_scripts() {
    local test_scripts=(
        "validate_bev_deployment.sh:Validation Tests"
        "integration_tests.sh:Integration Tests"
        "performance_tests.sh:Performance Tests"
        "security_tests.sh:Security Tests"
        "monitoring_tests.sh:Monitoring Tests"
    )

    for script_info in "${test_scripts[@]}"; do
        IFS=':' read -ra script_parts <<< "$script_info"
        local script_name="${script_parts[0]}"
        local script_description="${script_parts[1]}"
        local script_path="$TESTS_DIR/$script_name"

        if [[ -x "$script_path" ]]; then
            log_info "$script_description script available: $script_path"
        else
            log_error "$script_description script not found or not executable: $script_path"
            exit 1
        fi
    done
}

# Execute test suite
execute_test_suite() {
    local test_name="$1"
    local test_script="$2"
    local test_description="$3"

    log_banner "Starting $test_description"

    local start_time=$(date +%s)
    local test_log="$REPORTS_DIR/${test_name}_$TIMESTAMP.log"

    if [[ "$PARALLEL_EXECUTION" == true ]]; then
        # Run in background for parallel execution
        {
            if "$test_script" > "$test_log" 2>&1; then
                TEST_RESULTS["$test_name"]="PASSED"
            else
                TEST_RESULTS["$test_name"]="FAILED"
            fi
        } &
    else
        # Run sequentially
        if "$test_script" 2>&1 | tee "$test_log"; then
            TEST_RESULTS["$test_name"]="PASSED"
            log_success "$test_description completed successfully"
        else
            TEST_RESULTS["$test_name"]="FAILED"
            log_error "$test_description failed"

            if [[ "$FAIL_FAST" == true ]]; then
                log_error "Fail-fast enabled. Stopping test execution."
                exit 1
            fi
        fi
    fi

    local end_time=$(date +%s)
    TEST_DURATIONS["$test_name"]=$((end_time - start_time))
    TEST_LOGS["$test_name"]="$test_log"

    log_info "$test_description duration: ${TEST_DURATIONS[$test_name]} seconds"
}

# Wait for parallel tests to complete
wait_for_parallel_tests() {
    if [[ "$PARALLEL_EXECUTION" == true ]]; then
        log_info "Waiting for parallel tests to complete..."
        wait
        log_success "All parallel tests completed"
    fi
}

# Run main validation tests
run_validation_tests() {
    if [[ "$RUN_VALIDATION" == true ]]; then
        execute_test_suite "validation" "$SCRIPT_DIR/validate_bev_deployment.sh" "System Validation Tests"
    fi
}

# Run integration tests
run_integration_tests() {
    if [[ "$RUN_INTEGRATION" == true ]]; then
        execute_test_suite "integration" "$TESTS_DIR/integration_tests.sh" "Integration Tests"
    fi
}

# Run performance tests
run_performance_tests() {
    if [[ "$RUN_PERFORMANCE" == true ]]; then
        execute_test_suite "performance" "$TESTS_DIR/performance_tests.sh" "Performance Tests"
    fi
}

# Run security tests
run_security_tests() {
    if [[ "$RUN_SECURITY" == true ]]; then
        execute_test_suite "security" "$TESTS_DIR/security_tests.sh" "Security Tests"
    fi
}

# Run monitoring tests
run_monitoring_tests() {
    if [[ "$RUN_MONITORING" == true ]]; then
        execute_test_suite "monitoring" "$TESTS_DIR/monitoring_tests.sh" "Monitoring Tests"
    fi
}

# Generate comprehensive test report
generate_comprehensive_report() {
    if [[ "$GENERATE_REPORT" != true ]]; then
        return
    fi

    log_banner "Generating Comprehensive Test Report"

    # Calculate overall statistics
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    local total_duration=0

    for test_name in "${!TEST_RESULTS[@]}"; do
        ((total_tests++))
        if [[ "${TEST_RESULTS[$test_name]}" == "PASSED" ]]; then
            ((passed_tests++))
        else
            ((failed_tests++))
        fi
        total_duration=$((total_duration + TEST_DURATIONS[$test_name]))
    done

    local success_rate=0
    if [[ $total_tests -gt 0 ]]; then
        success_rate=$(( (passed_tests * 100) / total_tests ))
    fi

    # Determine overall status
    local overall_status="EXCELLENT"
    local status_color="#28a745"

    if [[ $failed_tests -gt 0 ]]; then
        if [[ $success_rate -lt 70 ]]; then
            overall_status="CRITICAL"
            status_color="#dc3545"
        elif [[ $success_rate -lt 85 ]]; then
            overall_status="NEEDS IMPROVEMENT"
            status_color="#fd7e14"
        else
            overall_status="GOOD"
            status_color="#ffc107"
        fi
    fi

    # Generate HTML report
    cat > "$MASTER_REPORT" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEV Complete System Test Report - $TIMESTAMP</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header .subtitle {
            margin-top: 10px;
            opacity: 0.8;
            font-size: 1.1em;
        }
        .overall-status {
            background: $status_color;
            color: white;
            padding: 20px;
            text-align: center;
            font-size: 1.3em;
            font-weight: bold;
            margin: 20px;
            border-radius: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
        }
        .stat-card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            border-left: 5px solid #007bff;
            transition: transform 0.2s ease;
        }
        .stat-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .stat-card.passed { border-left-color: #28a745; }
        .stat-card.failed { border-left-color: #dc3545; }
        .stat-card.duration { border-left-color: #17a2b8; }
        .stat-card.success-rate { border-left-color: #ffc107; }
        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .test-results {
            padding: 30px;
        }
        .test-category {
            margin-bottom: 30px;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #e9ecef;
        }
        .test-category-header {
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #e9ecef;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .test-category-content {
            padding: 20px;
        }
        .test-result {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #dee2e6;
        }
        .test-result.passed {
            background: #d4edda;
            border-left-color: #28a745;
            color: #155724;
        }
        .test-result.failed {
            background: #f8d7da;
            border-left-color: #dc3545;
            color: #721c24;
        }
        .test-meta {
            display: flex;
            gap: 20px;
            font-size: 0.9em;
            opacity: 0.8;
        }
        .recommendations {
            background: #e7f3ff;
            padding: 25px;
            margin: 20px;
            border-radius: 10px;
            border-left: 5px solid #007bff;
        }
        .recommendations h3 {
            margin-top: 0;
            color: #0056b3;
        }
        .footer {
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #666;
            border-top: 1px solid #e9ecef;
        }
        .icon {
            font-size: 1.2em;
            margin-right: 8px;
        }
        .status-badge {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
            text-transform: uppercase;
        }
        .status-passed { background: #28a745; color: white; }
        .status-failed { background: #dc3545; color: white; }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            transition: width 0.3s ease;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 BEV System Test Report</h1>
            <div class="subtitle">Complete Validation & Assessment Report</div>
            <div class="subtitle">Generated: $(date)</div>
        </div>

        <div class="overall-status">
            <span class="icon">📊</span>OVERALL SYSTEM STATUS: $overall_status
        </div>

        <div class="stats-grid">
            <div class="stat-card passed">
                <div class="stat-number">$passed_tests</div>
                <div class="stat-label">Tests Passed</div>
            </div>
            <div class="stat-card failed">
                <div class="stat-number">$failed_tests</div>
                <div class="stat-label">Tests Failed</div>
            </div>
            <div class="stat-card duration">
                <div class="stat-number">$(($total_duration / 60))</div>
                <div class="stat-label">Minutes Total</div>
            </div>
            <div class="stat-card success-rate">
                <div class="stat-number">$success_rate%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>

        <div class="progress-bar">
            <div class="progress-fill" style="width: ${success_rate}%"></div>
        </div>

        <div class="test-results">
            <h2>📋 Test Results by Category</h2>
EOF

    # Add test results for each category
    for test_name in "${!TEST_RESULTS[@]}"; do
        local result="${TEST_RESULTS[$test_name]}"
        local duration="${TEST_DURATIONS[$test_name]}"
        local log_file="${TEST_LOGS[$test_name]}"

        local test_display_name
        case "$test_name" in
            "validation") test_display_name="🔍 System Validation" ;;
            "integration") test_display_name="🔗 Integration Tests" ;;
            "performance") test_display_name="⚡ Performance Tests" ;;
            "security") test_display_name="🛡️ Security Tests" ;;
            "monitoring") test_display_name="📊 Monitoring Tests" ;;
            *) test_display_name="📝 $test_name" ;;
        esac

        local status_class="passed"
        local status_badge_class="status-passed"
        if [[ "$result" == "FAILED" ]]; then
            status_class="failed"
            status_badge_class="status-failed"
        fi

        cat >> "$MASTER_REPORT" << EOF
            <div class="test-category">
                <div class="test-category-header">
                    <span>$test_display_name</span>
                    <span class="status-badge $status_badge_class">$result</span>
                </div>
                <div class="test-category-content">
                    <div class="test-result $status_class">
                        <div>
                            <strong>$test_display_name</strong>
                            <div class="test-meta">
                                <span>⏱️ Duration: ${duration}s</span>
                                <span>📁 Log: $(basename "$log_file")</span>
                            </div>
                        </div>
                        <div class="status-badge $status_badge_class">$result</div>
                    </div>
                </div>
            </div>
EOF
    done

    # Add recommendations section
    cat >> "$MASTER_REPORT" << EOF
        </div>

        <div class="recommendations">
            <h3>🎯 Recommendations</h3>
            $(if [[ $failed_tests -eq 0 ]]; then
                echo "<p><strong>Excellent!</strong> All tests passed. Your BEV system is ready for production deployment.</p>"
                echo "<ul>"
                echo "<li>✅ Continue with regular monitoring and maintenance</li>"
                echo "<li>✅ Schedule periodic security assessments</li>"
                echo "<li>✅ Implement automated testing in CI/CD pipeline</li>"
                echo "<li>✅ Set up alerting for critical system metrics</li>"
                echo "</ul>"
            elif [[ $success_rate -ge 85 ]]; then
                echo "<p><strong>Good progress!</strong> Most tests passed, but some issues need attention.</p>"
                echo "<ul>"
                echo "<li>🔧 Review and fix failed test cases</li>"
                echo "<li>🔍 Check system logs for error details</li>"
                echo "<li>⚠️ Address security and performance issues</li>"
                echo "<li>🔄 Re-run tests after fixes</li>"
                echo "</ul>"
            else
                echo "<p><strong>Critical issues detected!</strong> Significant problems need immediate attention.</p>"
                echo "<ul>"
                echo "<li>🚨 Do not deploy to production until issues are resolved</li>"
                echo "<li>🔧 Review all failed tests and their logs</li>"
                echo "<li>🛡️ Pay special attention to security failures</li>"
                echo "<li>📞 Consider getting expert assistance</li>"
                echo "</ul>"
            fi)
        </div>

        <div class="footer">
            <p>BEV System Test Suite v1.0 | Generated on $(date)</p>
            <p>For detailed logs and analysis, check individual test log files in the reports directory</p>
        </div>
    </div>
</body>
</html>
EOF

    log_success "Comprehensive test report generated: $MASTER_REPORT"
}

# Generate summary statistics
generate_summary_statistics() {
    log_banner "Test Execution Summary"

    local total_tests=${#TEST_RESULTS[@]}
    local passed_tests=0
    local failed_tests=0
    local total_duration=0

    for test_name in "${!TEST_RESULTS[@]}"; do
        if [[ "${TEST_RESULTS[$test_name]}" == "PASSED" ]]; then
            ((passed_tests++))
        else
            ((failed_tests++))
        fi
        total_duration=$((total_duration + TEST_DURATIONS[$test_name]))
    done

    log_info "═══════════════════════════════════════"
    log_info "BEV SYSTEM TEST EXECUTION SUMMARY"
    log_info "═══════════════════════════════════════"
    log_info "Total Test Suites: $total_tests"
    log_success "Passed: $passed_tests"
    if [[ $failed_tests -gt 0 ]]; then
        log_error "Failed: $failed_tests"
    else
        log_info "Failed: $failed_tests"
    fi
    log_info "Total Duration: $(($total_duration / 60)) minutes $(($total_duration % 60)) seconds"

    if [[ $total_tests -gt 0 ]]; then
        local success_rate=$(( (passed_tests * 100) / total_tests ))
        log_info "Success Rate: $success_rate%"
    fi

    log_info "═══════════════════════════════════════"

    # Detailed results
    log_info "Detailed Results:"
    for test_name in "${!TEST_RESULTS[@]}"; do
        local result="${TEST_RESULTS[$test_name]}"
        local duration="${TEST_DURATIONS[$test_name]}"

        if [[ "$result" == "PASSED" ]]; then
            log_success "$test_name: $result (${duration}s)"
        else
            log_error "$test_name: $result (${duration}s)"
        fi
    done

    log_info "═══════════════════════════════════════"
    log_info "Master Report: $MASTER_REPORT"
    log_info "Master Log: $MASTER_LOG"
    log_info "═══════════════════════════════════════"
}

# Cleanup function
cleanup() {
    log_info "Performing cleanup..."

    # Kill any background processes if parallel execution was used
    if [[ "$PARALLEL_EXECUTION" == true ]]; then
        jobs -p | xargs -r kill 2>/dev/null || true
    fi

    # Archive old reports (keep last 10)
    local old_reports=$(ls -1t "$REPORTS_DIR"/bev_complete_test_report_*.html 2>/dev/null | tail -n +11)
    if [[ -n "$old_reports" ]]; then
        echo "$old_reports" | xargs rm -f
        log_info "Cleaned up old test reports"
    fi
}

# Signal handlers
trap cleanup EXIT
trap 'log_error "Test execution interrupted"; exit 130' INT TERM

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

BEV Complete Test Suite Runner

OPTIONS:
    --skip-validation     Skip system validation tests
    --skip-integration    Skip integration tests
    --skip-performance    Skip performance tests
    --skip-security       Skip security tests
    --skip-monitoring     Skip monitoring tests
    --parallel           Run tests in parallel (faster but less detailed logs)
    --fail-fast          Stop on first test failure
    --no-report          Skip HTML report generation
    -q, --quick          Quick mode (skip time-intensive tests)
    -h, --help           Show this help message

EXAMPLES:
    $0                                    # Run all tests
    $0 --skip-performance --skip-security # Run only validation, integration, and monitoring
    $0 --parallel                         # Run all tests in parallel
    $0 --quick --fail-fast               # Quick mode with early exit on failure

ENVIRONMENT:
    Set environment variables in .env file for customization
    Check individual test scripts for specific configuration options

OUTPUT:
    - Comprehensive HTML report in test-reports/
    - Individual test logs for detailed analysis
    - Master execution log for overall tracking

EOF
}

# Main execution function
main() {
    log_banner "BEV Complete Test Suite Runner Starting"
    log_info "Timestamp: $TIMESTAMP"

    # Initialize environment
    init_test_environment

    # Execute test suites
    log_banner "Executing Test Suites"

    if [[ "$PARALLEL_EXECUTION" == true ]]; then
        log_info "Running tests in parallel mode..."
        run_validation_tests &
        run_integration_tests &
        run_performance_tests &
        run_security_tests &
        run_monitoring_tests &
        wait_for_parallel_tests
    else
        log_info "Running tests sequentially..."
        run_validation_tests
        run_integration_tests
        run_performance_tests
        run_security_tests
        run_monitoring_tests
    fi

    # Generate reports and summary
    generate_comprehensive_report
    generate_summary_statistics

    # Determine exit code
    local failed_count=0
    for test_name in "${!TEST_RESULTS[@]}"; do
        if [[ "${TEST_RESULTS[$test_name]}" == "FAILED" ]]; then
            ((failed_count++))
        fi
    done

    if [[ $failed_count -eq 0 ]]; then
        log_success "All test suites completed successfully!"
        log_success "BEV system is ready for production deployment."
        exit 0
    else
        log_error "$failed_count test suite(s) failed."
        log_error "Review the detailed reports and fix issues before deployment."
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-validation)
            RUN_VALIDATION=false
            shift
            ;;
        --skip-integration)
            RUN_INTEGRATION=false
            shift
            ;;
        --skip-performance)
            RUN_PERFORMANCE=false
            shift
            ;;
        --skip-security)
            RUN_SECURITY=false
            shift
            ;;
        --skip-monitoring)
            RUN_MONITORING=false
            shift
            ;;
        --parallel)
            PARALLEL_EXECUTION=true
            shift
            ;;
        --fail-fast)
            FAIL_FAST=true
            shift
            ;;
        --no-report)
            GENERATE_REPORT=false
            shift
            ;;
        -q|--quick)
            # Quick mode - will be passed to individual test scripts
            export QUICK_MODE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"