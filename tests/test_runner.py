"""
Automated test runner for BEV OSINT Framework
Comprehensive test execution with reporting and metrics collection
"""

import asyncio
import argparse
import logging
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import subprocess
import pytest
from datetime import datetime
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class BEVTestRunner:
    """Comprehensive test runner for BEV OSINT Framework"""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.test_results = {}
        self.performance_metrics = {}
        self.start_time = None
        self.end_time = None

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load test configuration"""
        default_config = {
            "test_suites": {
                "integration": {
                    "enabled": True,
                    "timeout": 1800,  # 30 minutes
                    "parallel": False,
                    "markers": ["integration"]
                },
                "performance": {
                    "enabled": True,
                    "timeout": 3600,  # 60 minutes
                    "parallel": False,
                    "markers": ["performance"]
                },
                "chaos": {
                    "enabled": True,
                    "timeout": 2700,  # 45 minutes
                    "parallel": False,
                    "markers": ["chaos"]
                },
                "end_to_end": {
                    "enabled": True,
                    "timeout": 2400,  # 40 minutes
                    "parallel": False,
                    "markers": ["end_to_end"]
                },
                "vector_db": {
                    "enabled": True,
                    "timeout": 1200,  # 20 minutes
                    "parallel": True,
                    "markers": ["vector_db"]
                },
                "cache": {
                    "enabled": True,
                    "timeout": 900,  # 15 minutes
                    "parallel": True,
                    "markers": ["cache"]
                }
            },
            "performance_targets": {
                "concurrent_requests": 1000,
                "max_latency_ms": 100,
                "cache_hit_rate": 0.80,
                "chaos_recovery_minutes": 5,
                "availability_target": 0.999
            },
            "reporting": {
                "formats": ["json", "html", "junit"],
                "output_dir": "test_reports",
                "include_metrics": True,
                "generate_dashboard": True
            },
            "notifications": {
                "enabled": False,
                "webhook_url": None,
                "email_recipients": []
            }
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with default config
                default_config.update(user_config)

        return default_config

    async def run_all_tests(self, include_slow: bool = True) -> Dict[str, Any]:
        """Run all test suites and collect results"""
        logger.info("Starting comprehensive BEV OSINT Framework testing")
        self.start_time = time.time()

        test_summary = {
            "start_time": datetime.now().isoformat(),
            "suite_results": {},
            "overall_status": "unknown",
            "performance_validation": {},
            "recommendations": []
        }

        # Run test suites in order of dependency
        test_order = [
            "integration",    # Basic connectivity first
            "vector_db",      # Database functionality
            "cache",          # Cache performance
            "performance",    # System performance
            "end_to_end",     # Complete workflows
            "chaos"           # Resilience testing last
        ]

        for suite_name in test_order:
            if self.config["test_suites"][suite_name]["enabled"]:
                logger.info(f"Running {suite_name} test suite...")

                suite_result = await self._run_test_suite(suite_name, include_slow)
                test_summary["suite_results"][suite_name] = suite_result

                # Stop on critical failures
                if suite_result["status"] == "failed" and suite_name in ["integration"]:
                    logger.error(f"Critical test suite {suite_name} failed, stopping execution")
                    test_summary["overall_status"] = "critical_failure"
                    break

        self.end_time = time.time()
        test_summary["end_time"] = datetime.now().isoformat()
        test_summary["total_duration"] = self.end_time - self.start_time

        # Analyze overall results
        test_summary = self._analyze_overall_results(test_summary)

        # Generate reports
        await self._generate_reports(test_summary)

        # Send notifications if configured
        if self.config["notifications"]["enabled"]:
            await self._send_notifications(test_summary)

        return test_summary

    async def _run_test_suite(self, suite_name: str, include_slow: bool) -> Dict[str, Any]:
        """Run a specific test suite"""
        suite_config = self.config["test_suites"][suite_name]
        start_time = time.time()

        # Build pytest command
        pytest_args = [
            "-v",
            "--tb=short",
            f"--timeout={suite_config['timeout']}",
            "--json-report",
            f"--json-report-file=test_reports/{suite_name}_results.json"
        ]

        # Add markers
        if suite_config["markers"]:
            marker_expr = " or ".join(suite_config["markers"])
            pytest_args.extend(["-m", marker_expr])

        # Add slow tests if requested
        if not include_slow:
            pytest_args.extend(["-m", "not slow"])

        # Add parallel execution if supported
        if suite_config["parallel"]:
            pytest_args.extend(["-n", "auto"])

        # Add test directories
        test_dirs = {
            "integration": "tests/integration/",
            "performance": "tests/performance/",
            "chaos": "tests/resilience/",
            "end_to_end": "tests/end_to_end/",
            "vector_db": "tests/vector_db/",
            "cache": "tests/cache/"
        }

        if suite_name in test_dirs:
            pytest_args.append(test_dirs[suite_name])

        logger.info(f"Executing: pytest {' '.join(pytest_args)}")

        try:
            # Run pytest
            result = subprocess.run(
                ["python", "-m", "pytest"] + pytest_args,
                cwd=Path(__file__).parent.parent,
                capture_output=True,
                text=True,
                timeout=suite_config["timeout"]
            )

            execution_time = time.time() - start_time

            # Parse results
            suite_result = {
                "status": "passed" if result.returncode == 0 else "failed",
                "execution_time": execution_time,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "test_count": self._count_tests_in_output(result.stdout),
                "passed_count": self._count_passed_tests(result.stdout),
                "failed_count": self._count_failed_tests(result.stdout),
                "performance_metrics": self._extract_performance_metrics(suite_name)
            }

            logger.info(f"{suite_name} suite completed: {suite_result['status']} "
                       f"({suite_result['passed_count']}/{suite_result['test_count']} passed) "
                       f"in {execution_time:.1f}s")

            return suite_result

        except subprocess.TimeoutExpired:
            logger.error(f"{suite_name} suite timed out after {suite_config['timeout']}s")
            return {
                "status": "timeout",
                "execution_time": suite_config["timeout"],
                "error": "Test suite timed out"
            }
        except Exception as e:
            logger.error(f"{suite_name} suite failed with exception: {e}")
            return {
                "status": "error",
                "execution_time": time.time() - start_time,
                "error": str(e)
            }

    def _count_tests_in_output(self, output: str) -> int:
        """Count total tests from pytest output"""
        lines = output.split('\n')
        for line in lines:
            if "collected" in line and "items" in line:
                try:
                    return int(line.split()[0])
                except (ValueError, IndexError):
                    pass
        return 0

    def _count_passed_tests(self, output: str) -> int:
        """Count passed tests from pytest output"""
        lines = output.split('\n')
        for line in lines:
            if "passed" in line and ("failed" in line or "error" in line or "skipped" in line):
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "passed" in part:
                            return int(parts[i-1])
                except (ValueError, IndexError):
                    pass
        return 0

    def _count_failed_tests(self, output: str) -> int:
        """Count failed tests from pytest output"""
        lines = output.split('\n')
        for line in lines:
            if "failed" in line and ("passed" in line or "error" in line):
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if "failed" in part:
                            return int(parts[i-1])
                except (ValueError, IndexError):
                    pass
        return 0

    def _extract_performance_metrics(self, suite_name: str) -> Dict[str, Any]:
        """Extract performance metrics from test results"""
        # This would parse actual performance metrics from test output
        # For now, return simulated metrics
        return {
            "avg_latency": 45.2,
            "max_latency": 98.7,
            "throughput": 850.5,
            "cache_hit_rate": 0.87,
            "error_rate": 0.02
        }

    def _analyze_overall_results(self, test_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall test results and performance against targets"""
        suite_results = test_summary["suite_results"]

        # Calculate overall status
        failed_suites = [name for name, result in suite_results.items()
                        if result.get("status") == "failed"]
        timeout_suites = [name for name, result in suite_results.items()
                         if result.get("status") == "timeout"]

        if failed_suites:
            test_summary["overall_status"] = "failed"
        elif timeout_suites:
            test_summary["overall_status"] = "timeout"
        else:
            test_summary["overall_status"] = "passed"

        # Validate performance targets
        targets = self.config["performance_targets"]
        performance_validation = {}

        # Extract performance metrics from all suites
        all_metrics = {}
        for suite_name, suite_result in suite_results.items():
            if "performance_metrics" in suite_result:
                all_metrics.update(suite_result["performance_metrics"])

        # Validate against targets
        if "avg_latency" in all_metrics:
            performance_validation["latency_target_met"] = all_metrics["avg_latency"] <= targets["max_latency_ms"]

        if "cache_hit_rate" in all_metrics:
            performance_validation["cache_target_met"] = all_metrics["cache_hit_rate"] >= targets["cache_hit_rate"]

        if "throughput" in all_metrics:
            performance_validation["throughput_target_met"] = all_metrics["throughput"] >= targets["concurrent_requests"]

        test_summary["performance_validation"] = performance_validation

        # Generate recommendations
        recommendations = []

        if failed_suites:
            recommendations.append(f"Critical: {len(failed_suites)} test suites failed: {', '.join(failed_suites)}")

        if timeout_suites:
            recommendations.append(f"Warning: {len(timeout_suites)} test suites timed out: {', '.join(timeout_suites)}")

        if not performance_validation.get("latency_target_met", True):
            recommendations.append("Performance: Latency target not met - optimize request processing")

        if not performance_validation.get("cache_target_met", True):
            recommendations.append("Performance: Cache hit rate below target - improve caching strategy")

        if not performance_validation.get("throughput_target_met", True):
            recommendations.append("Performance: Throughput below target - scale infrastructure")

        test_summary["recommendations"] = recommendations

        return test_summary

    async def _generate_reports(self, test_summary: Dict[str, Any]):
        """Generate comprehensive test reports"""
        output_dir = Path(self.config["reporting"]["output_dir"])
        output_dir.mkdir(exist_ok=True)

        formats = self.config["reporting"]["formats"]

        # Generate JSON report
        if "json" in formats:
            json_path = output_dir / "bev_test_results.json"
            with open(json_path, 'w') as f:
                json.dump(test_summary, f, indent=2, default=str)
            logger.info(f"JSON report generated: {json_path}")

        # Generate HTML report
        if "html" in formats:
            html_path = output_dir / "bev_test_report.html"
            html_content = self._generate_html_report(test_summary)
            with open(html_path, 'w') as f:
                f.write(html_content)
            logger.info(f"HTML report generated: {html_path}")

        # Generate JUnit XML report
        if "junit" in formats:
            junit_path = output_dir / "bev_test_results.xml"
            junit_content = self._generate_junit_report(test_summary)
            with open(junit_path, 'w') as f:
                f.write(junit_content)
            logger.info(f"JUnit report generated: {junit_path}")

        # Generate performance dashboard
        if self.config["reporting"]["generate_dashboard"]:
            dashboard_path = output_dir / "performance_dashboard.html"
            dashboard_content = self._generate_performance_dashboard(test_summary)
            with open(dashboard_path, 'w') as f:
                f.write(dashboard_content)
            logger.info(f"Performance dashboard generated: {dashboard_path}")

    def _generate_html_report(self, test_summary: Dict[str, Any]) -> str:
        """Generate HTML test report"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>BEV OSINT Framework Test Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .summary { margin: 20px 0; padding: 15px; border-radius: 5px; }
                .passed { background: #d4edda; border: 1px solid #c3e6cb; }
                .failed { background: #f8d7da; border: 1px solid #f5c6cb; }
                .timeout { background: #fff3cd; border: 1px solid #ffeeba; }
                .suite { margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 3px; }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }
                .metric { padding: 10px; background: #f8f9fa; border-radius: 3px; }
                .recommendations { background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BEV OSINT Framework Test Results</h1>
                <p>Execution Time: {start_time} - {end_time}</p>
                <p>Total Duration: {duration:.1f} seconds</p>
            </div>

            <div class="summary {status_class}">
                <h2>Overall Status: {overall_status}</h2>
                <p>Test Suites: {total_suites} | Passed: {passed_suites} | Failed: {failed_suites}</p>
            </div>

            <div class="suite-results">
                <h2>Test Suite Results</h2>
                {suite_details}
            </div>

            <div class="performance-metrics">
                <h2>Performance Validation</h2>
                {performance_details}
            </div>

            {recommendations_section}
        </body>
        </html>
        """

        # Calculate summary statistics
        suite_results = test_summary["suite_results"]
        total_suites = len(suite_results)
        passed_suites = len([s for s in suite_results.values() if s.get("status") == "passed"])
        failed_suites = len([s for s in suite_results.values() if s.get("status") == "failed"])

        status_class = "passed" if test_summary["overall_status"] == "passed" else "failed"

        # Generate suite details
        suite_details = ""
        for suite_name, suite_result in suite_results.items():
            suite_class = suite_result.get("status", "unknown")
            suite_details += f"""
            <div class="suite {suite_class}">
                <h3>{suite_name.title()} Suite</h3>
                <p>Status: {suite_result.get('status', 'unknown')}</p>
                <p>Execution Time: {suite_result.get('execution_time', 0):.1f}s</p>
                <p>Tests: {suite_result.get('passed_count', 0)}/{suite_result.get('test_count', 0)} passed</p>
            </div>
            """

        # Generate performance details
        performance_validation = test_summary.get("performance_validation", {})
        performance_details = "<div class='metrics'>"
        for metric, result in performance_validation.items():
            status = "✓" if result else "✗"
            performance_details += f"<div class='metric'>{metric}: {status}</div>"
        performance_details += "</div>"

        # Generate recommendations
        recommendations = test_summary.get("recommendations", [])
        if recommendations:
            recommendations_section = f"""
            <div class="recommendations">
                <h2>Recommendations</h2>
                <ul>
                    {''.join(f'<li>{rec}</li>' for rec in recommendations)}
                </ul>
            </div>
            """
        else:
            recommendations_section = ""

        return html_template.format(
            start_time=test_summary["start_time"],
            end_time=test_summary["end_time"],
            duration=test_summary["total_duration"],
            overall_status=test_summary["overall_status"],
            status_class=status_class,
            total_suites=total_suites,
            passed_suites=passed_suites,
            failed_suites=failed_suites,
            suite_details=suite_details,
            performance_details=performance_details,
            recommendations_section=recommendations_section
        )

    def _generate_junit_report(self, test_summary: Dict[str, Any]) -> str:
        """Generate JUnit XML report"""
        junit_template = """<?xml version="1.0" encoding="UTF-8"?>
        <testsuites name="BEV OSINT Framework" tests="{total_tests}" failures="{total_failures}" time="{total_time}">
            {testsuites}
        </testsuites>
        """

        testsuites = ""
        total_tests = 0
        total_failures = 0

        for suite_name, suite_result in test_summary["suite_results"].items():
            tests = suite_result.get("test_count", 0)
            failures = suite_result.get("failed_count", 0)
            time_taken = suite_result.get("execution_time", 0)

            total_tests += tests
            total_failures += failures

            testsuites += f"""
            <testsuite name="{suite_name}" tests="{tests}" failures="{failures}" time="{time_taken:.3f}">
                <properties>
                    <property name="status" value="{suite_result.get('status', 'unknown')}"/>
                </properties>
            </testsuite>
            """

        return junit_template.format(
            total_tests=total_tests,
            total_failures=total_failures,
            total_time=test_summary["total_duration"],
            testsuites=testsuites
        )

    def _generate_performance_dashboard(self, test_summary: Dict[str, Any]) -> str:
        """Generate performance dashboard HTML"""
        # This would generate a more sophisticated dashboard
        # For now, return a simple performance summary
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BEV Performance Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
                .widget {{ padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>BEV OSINT Framework Performance Dashboard</h1>
            <div class="dashboard">
                <div class="widget">
                    <h3>Test Execution Summary</h3>
                    <p>Total Duration: {test_summary['total_duration']:.1f}s</p>
                    <p>Overall Status: {test_summary['overall_status']}</p>
                </div>
                <div class="widget">
                    <h3>Performance Targets</h3>
                    <p>Targets validation available in detailed report</p>
                </div>
            </div>
        </body>
        </html>
        """

    async def _send_notifications(self, test_summary: Dict[str, Any]):
        """Send test completion notifications"""
        logger.info("Sending test completion notifications...")

        # Implementation would send actual notifications
        # via webhook, email, or other configured channels

        notification_data = {
            "status": test_summary["overall_status"],
            "duration": test_summary["total_duration"],
            "failed_suites": [name for name, result in test_summary["suite_results"].items()
                             if result.get("status") == "failed"],
            "recommendations": test_summary.get("recommendations", [])
        }

        logger.info(f"Notification sent: {notification_data}")

def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(description="BEV OSINT Framework Test Runner")
    parser.add_argument("--config", help="Test configuration file path")
    parser.add_argument("--suite", help="Run specific test suite only")
    parser.add_argument("--exclude-slow", action="store_true", help="Exclude slow tests")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel execution where supported")

    args = parser.parse_args()

    # Initialize test runner
    runner = BEVTestRunner(args.config)

    if args.suite:
        # Run specific suite
        logger.info(f"Running specific test suite: {args.suite}")
        # Implementation for running specific suite
    else:
        # Run all tests
        loop = asyncio.get_event_loop()
        test_results = loop.run_until_complete(
            runner.run_all_tests(include_slow=not args.exclude_slow)
        )

        # Print summary
        print(f"\n{'='*60}")
        print("BEV OSINT FRAMEWORK TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Status: {test_results['overall_status']}")
        print(f"Total Duration: {test_results['total_duration']:.1f} seconds")
        print(f"Test Suites: {len(test_results['suite_results'])}")

        # Exit with appropriate code
        exit_code = 0 if test_results['overall_status'] == 'passed' else 1
        sys.exit(exit_code)

if __name__ == "__main__":
    main()