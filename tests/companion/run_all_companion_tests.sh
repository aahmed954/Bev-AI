#!/bin/bash

# Comprehensive AI Companion Testing Suite Execution Script
# Executes all companion tests with proper sequencing, monitoring, and reporting

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEST_REPORTS_DIR="$PROJECT_ROOT/test_reports/companion"
LOG_FILE="$TEST_REPORTS_DIR/test_execution.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test execution options
PARALLEL=${PARALLEL:-false}
QUICK=${QUICK:-false}
MONITOR_PERFORMANCE=${MONITOR_PERFORMANCE:-true}
GENERATE_REPORTS=${GENERATE_REPORTS:-true}
CLEANUP_AFTER=${CLEANUP_AFTER:-true}

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date '+%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Function to check prerequisites
check_prerequisites() {
    print_status $BLUE "🔍 Checking test prerequisites..."

    # Check Python environment
    if ! command -v python &> /dev/null; then
        print_status $RED "❌ Python not found"
        exit 1
    fi

    # Check pytest installation
    if ! python -c "import pytest" &> /dev/null; then
        print_status $RED "❌ pytest not installed"
        exit 1
    fi

    # Check GPU availability for RTX 4090 tests
    if ! python -c "import GPUtil; gpus=GPUtil.getGPUs(); assert len(gpus)>0 and 'RTX' in gpus[0].name" &> /dev/null; then
        print_status $YELLOW "⚠️ RTX 4090 not detected - GPU tests will be skipped"
        export SKIP_GPU_TESTS=true
    fi

    # Check available memory
    available_memory=$(free -g | awk '/^Mem:/{print $7}')
    if [ "$available_memory" -lt 8 ]; then
        print_status $YELLOW "⚠️ Low available memory (${available_memory}GB) - some tests may be skipped"
    fi

    # Check companion service dependencies
    if ! python -c "import asyncio, aiohttp, websockets" &> /dev/null; then
        print_status $RED "❌ Companion service dependencies not found"
        exit 1
    fi

    print_status $GREEN "✅ Prerequisites check passed"
}

# Function to setup test environment
setup_test_environment() {
    print_status $BLUE "🏗️ Setting up test environment..."

    # Create test reports directory
    mkdir -p "$TEST_REPORTS_DIR"/{core,performance,ux,security,integration,automation}

    # Initialize log file
    echo "Companion Test Suite Execution Log" > "$LOG_FILE"
    echo "Started at: $(date)" >> "$LOG_FILE"
    echo "Configuration: PARALLEL=$PARALLEL, QUICK=$QUICK, MONITOR_PERFORMANCE=$MONITOR_PERFORMANCE" >> "$LOG_FILE"

    # Copy test configuration
    if [ -f "$SCRIPT_DIR/config/companion_test_config.yaml" ]; then
        cp "$SCRIPT_DIR/config/companion_test_config.yaml" "$TEST_REPORTS_DIR/"
    fi

    # Start performance monitoring if enabled
    if [ "$MONITOR_PERFORMANCE" = true ]; then
        print_status $BLUE "📊 Starting performance monitoring..."
        python -c "
import sys
sys.path.append('$SCRIPT_DIR')
from utils.performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor()
monitor.start_monitoring()
print('Performance monitoring started')
" &
        MONITOR_PID=$!
        echo $MONITOR_PID > "$TEST_REPORTS_DIR/monitor.pid"
    fi

    print_status $GREEN "✅ Test environment setup complete"
}

# Function to execute core companion tests
run_core_tests() {
    print_status $BLUE "🧠 Running core companion tests..."

    local test_args=""
    if [ "$QUICK" = true ]; then
        test_args="$test_args -k 'not slow'"
    fi

    if [ "$PARALLEL" = true ]; then
        test_args="$test_args -n auto"
    fi

    # Personality Consistency Tests
    print_status $BLUE "  Testing personality consistency..."
    python -m pytest "$SCRIPT_DIR/core/test_personality_consistency.py" \
        -v --tb=short \
        --json-report --json-report-file="$TEST_REPORTS_DIR/core/personality_results.json" \
        --timeout=1800 \
        $test_args || {
        print_status $RED "❌ Personality consistency tests failed"
        return 1
    }

    print_status $GREEN "✅ Core companion tests completed"
}

# Function to execute performance tests
run_performance_tests() {
    print_status $BLUE "⚡ Running performance tests..."

    if [ "$SKIP_GPU_TESTS" = true ]; then
        print_status $YELLOW "⚠️ Skipping GPU performance tests - RTX 4090 not available"
        return 0
    fi

    # RTX 4090 Performance Tests
    print_status $BLUE "  Testing RTX 4090 companion workloads..."
    python -m pytest "$SCRIPT_DIR/performance/test_rtx4090_companion_workloads.py" \
        -v --tb=short \
        --json-report --json-report-file="$TEST_REPORTS_DIR/performance/rtx4090_results.json" \
        --timeout=3600 \
        -m "not slow" || {
        print_status $RED "❌ RTX 4090 performance tests failed"
        return 1
    }

    print_status $GREEN "✅ Performance tests completed"
}

# Function to execute user experience tests
run_ux_tests() {
    print_status $BLUE "👤 Running user experience tests..."

    local test_args=""
    if [ "$PARALLEL" = true ]; then
        test_args="$test_args -n auto"
    fi

    # User Experience Tests
    print_status $BLUE "  Testing companion user experience..."
    python -m pytest "$SCRIPT_DIR/ux/test_companion_user_experience.py" \
        -v --tb=short \
        --json-report --json-report-file="$TEST_REPORTS_DIR/ux/ux_results.json" \
        --timeout=2400 \
        $test_args || {
        print_status $RED "❌ User experience tests failed"
        return 1
    }

    print_status $GREEN "✅ User experience tests completed"
}

# Function to execute security and privacy tests
run_security_tests() {
    print_status $BLUE "🔒 Running security and privacy tests..."

    # Security and Privacy Tests
    print_status $BLUE "  Testing companion security and privacy..."
    python -m pytest "$SCRIPT_DIR/security/test_companion_privacy_security.py" \
        -v --tb=short \
        --json-report --json-report-file="$TEST_REPORTS_DIR/security/security_results.json" \
        --timeout=2400 || {
        print_status $RED "❌ Security and privacy tests failed"
        return 1
    }

    print_status $GREEN "✅ Security and privacy tests completed"
}

# Function to execute integration tests
run_integration_tests() {
    print_status $BLUE "🔗 Running integration tests..."

    # Companion-OSINT Integration Tests
    print_status $BLUE "  Testing companion-OSINT integration..."
    python -m pytest "$SCRIPT_DIR/integration/test_companion_osint_integration.py" \
        -v --tb=short \
        --json-report --json-report-file="$TEST_REPORTS_DIR/integration/integration_results.json" \
        --timeout=3000 || {
        print_status $RED "❌ Integration tests failed"
        return 1
    }

    print_status $GREEN "✅ Integration tests completed"
}

# Function to execute automation tests
run_automation_tests() {
    print_status $BLUE "🤖 Running automation and orchestration tests..."

    # Automation Suite Tests
    print_status $BLUE "  Testing automation framework..."
    python -m pytest "$SCRIPT_DIR/automation/test_companion_automation_suite.py" \
        -v --tb=short \
        --json-report --json-report-file="$TEST_REPORTS_DIR/automation/automation_results.json" \
        --timeout=1800 || {
        print_status $RED "❌ Automation tests failed"
        return 1
    }

    print_status $GREEN "✅ Automation tests completed"
}

# Function to stop performance monitoring
stop_performance_monitoring() {
    if [ -f "$TEST_REPORTS_DIR/monitor.pid" ]; then
        local monitor_pid=$(cat "$TEST_REPORTS_DIR/monitor.pid")
        if kill -0 $monitor_pid 2>/dev/null; then
            print_status $BLUE "📊 Stopping performance monitoring..."
            kill $monitor_pid
            rm "$TEST_REPORTS_DIR/monitor.pid"
        fi
    fi
}

# Function to generate comprehensive reports
generate_reports() {
    if [ "$GENERATE_REPORTS" = false ]; then
        return 0
    fi

    print_status $BLUE "📋 Generating comprehensive test reports..."

    # Generate HTML report
    python -c "
import sys
sys.path.append('$SCRIPT_DIR')
from utils.report_generator import ReportGenerator
generator = ReportGenerator()
generator.generate_comprehensive_html_report('$TEST_REPORTS_DIR')
print('HTML report generated')
" || print_status $YELLOW "⚠️ HTML report generation failed"

    # Generate summary JSON
    python -c "
import sys, json, glob, os
sys.path.append('$SCRIPT_DIR')

# Collect all test results
results = {}
for result_file in glob.glob('$TEST_REPORTS_DIR/*/*.json'):
    if 'results.json' in result_file:
        category = os.path.basename(os.path.dirname(result_file))
        try:
            with open(result_file, 'r') as f:
                results[category] = json.load(f)
        except:
            pass

# Generate summary
summary = {
    'timestamp': '$(date -Iseconds)',
    'total_categories': len(results),
    'category_results': {},
    'overall_stats': {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'success_rate': 0.0
    }
}

for category, result in results.items():
    if 'summary' in result:
        summary['category_results'][category] = result['summary']
        summary['overall_stats']['total_tests'] += result['summary'].get('total', 0)
        summary['overall_stats']['passed_tests'] += result['summary'].get('passed', 0)
        summary['overall_stats']['failed_tests'] += result['summary'].get('failed', 0)

if summary['overall_stats']['total_tests'] > 0:
    summary['overall_stats']['success_rate'] = summary['overall_stats']['passed_tests'] / summary['overall_stats']['total_tests']

with open('$TEST_REPORTS_DIR/test_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Summary report generated')
"

    print_status $GREEN "✅ Reports generated in $TEST_REPORTS_DIR"
}

# Function to cleanup test environment
cleanup_test_environment() {
    if [ "$CLEANUP_AFTER" = false ]; then
        return 0
    fi

    print_status $BLUE "🧹 Cleaning up test environment..."

    # Stop any remaining processes
    stop_performance_monitoring

    # Clean up temporary files
    find "$TEST_REPORTS_DIR" -name "*.tmp" -delete 2>/dev/null || true
    find "$TEST_REPORTS_DIR" -name "*.lock" -delete 2>/dev/null || true

    # Archive old logs
    if [ -f "$TEST_REPORTS_DIR/test_execution.log" ]; then
        mv "$TEST_REPORTS_DIR/test_execution.log" "$TEST_REPORTS_DIR/test_execution_$(date +%Y%m%d_%H%M%S).log"
    fi

    print_status $GREEN "✅ Cleanup completed"
}

# Function to display final summary
display_summary() {
    print_status $BLUE "📊 Test Execution Summary"

    if [ -f "$TEST_REPORTS_DIR/test_summary.json" ]; then
        python -c "
import json
with open('$TEST_REPORTS_DIR/test_summary.json', 'r') as f:
    summary = json.load(f)

print(f\"Total test categories: {summary['total_categories']}\")
print(f\"Total tests executed: {summary['overall_stats']['total_tests']}\")
print(f\"Tests passed: {summary['overall_stats']['passed_tests']}\")
print(f\"Tests failed: {summary['overall_stats']['failed_tests']}\")
print(f\"Overall success rate: {summary['overall_stats']['success_rate']:.1%}\")

print(\"\\nCategory Results:\")
for category, result in summary['category_results'].items():
    success_rate = result['passed'] / result['total'] if result['total'] > 0 else 0
    print(f\"  {category}: {result['passed']}/{result['total']} ({success_rate:.1%})\")
"
    fi

    print_status $BLUE "📁 Test reports available in: $TEST_REPORTS_DIR"
    print_status $BLUE "📝 Execution log: $LOG_FILE"
}

# Main execution function
main() {
    local start_time=$(date +%s)

    print_status $GREEN "🚀 Starting Comprehensive AI Companion Test Suite"
    print_status $BLUE "Configuration: PARALLEL=$PARALLEL, QUICK=$QUICK, MONITOR_PERFORMANCE=$MONITOR_PERFORMANCE"

    # Setup
    check_prerequisites
    setup_test_environment

    # Execute test suites
    local test_failures=0

    # Core tests (required for all other tests)
    if ! run_core_tests; then
        ((test_failures++))
    fi

    # Performance tests (can run in parallel)
    if [ "$PARALLEL" = true ]; then
        {
            if ! run_performance_tests; then
                echo "performance_failed" > "$TEST_REPORTS_DIR/performance_status"
            fi
        } &
        PERF_PID=$!

        {
            if ! run_ux_tests; then
                echo "ux_failed" > "$TEST_REPORTS_DIR/ux_status"
            fi
        } &
        UX_PID=$!

        # Wait for parallel tests
        wait $PERF_PID
        wait $UX_PID

        # Check results
        if [ -f "$TEST_REPORTS_DIR/performance_status" ]; then
            ((test_failures++))
            rm "$TEST_REPORTS_DIR/performance_status"
        fi
        if [ -f "$TEST_REPORTS_DIR/ux_status" ]; then
            ((test_failures++))
            rm "$TEST_REPORTS_DIR/ux_status"
        fi
    else
        # Sequential execution
        if ! run_performance_tests; then
            ((test_failures++))
        fi

        if ! run_ux_tests; then
            ((test_failures++))
        fi
    fi

    # Security tests (sequential - required)
    if ! run_security_tests; then
        ((test_failures++))
    fi

    # Integration tests (sequential - depends on all components)
    if ! run_integration_tests; then
        ((test_failures++))
    fi

    # Automation tests (final validation)
    if ! run_automation_tests; then
        ((test_failures++))
    fi

    # Finalization
    stop_performance_monitoring
    generate_reports
    cleanup_test_environment

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # Display results
    display_summary

    print_status $BLUE "⏱️ Total execution time: ${duration} seconds"

    if [ $test_failures -eq 0 ]; then
        print_status $GREEN "🎉 All companion tests completed successfully!"
        exit 0
    else
        print_status $RED "❌ $test_failures test suite(s) failed"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel)
            PARALLEL=true
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --no-monitor)
            MONITOR_PERFORMANCE=false
            shift
            ;;
        --no-reports)
            GENERATE_REPORTS=false
            shift
            ;;
        --no-cleanup)
            CLEANUP_AFTER=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --parallel      Run tests in parallel where possible"
            echo "  --quick         Run quick test suite (skip slow tests)"
            echo "  --no-monitor    Disable performance monitoring"
            echo "  --no-reports    Skip report generation"
            echo "  --no-cleanup    Skip cleanup after tests"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            print_status $RED "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Execute main function
main "$@"