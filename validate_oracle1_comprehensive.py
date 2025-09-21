#!/usr/bin/env python3
"""
ORACLE1 Comprehensive Deployment Validation Master Script

This is the main validation orchestrator that coordinates all testing components:
- ARM64 compatibility testing
- Performance benchmarking
- Service integration testing
- Security and compliance validation
- Deployment readiness certification

Usage:
    python validate_oracle1_comprehensive.py [options]

Options:
    --phase <phase>         Run specific phase only
    --parallel              Run tests in parallel where possible
    --output-dir <dir>      Custom output directory for results
    --config <file>         Custom configuration file
    --fix                   Attempt automatic fixes
    --verbose               Enable detailed logging
    --no-docker             Skip Docker-dependent tests
    --quick                 Run quick validation (reduced test set)
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ValidationOrchestrator:
    """Master orchestrator for ORACLE1 deployment validation."""

    def __init__(self, project_root: str = "/home/starlord/Projects/Bev",
                 output_dir: Optional[str] = None, config_file: Optional[str] = None):
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "validation_results"
        self.config_file = Path(config_file) if config_file else None

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"oracle1_validation_{self.timestamp}"

        # Master results
        self.results = {
            "session_id": self.session_id,
            "timestamp": time.time(),
            "project_root": str(self.project_root),
            "validation_phases": {},
            "summary": {
                "total_phases": 0,
                "completed_phases": 0,
                "failed_phases": 0,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "warnings": 0,
                "critical_issues": 0,
                "overall_score": 0.0,
                "deployment_ready": False,
                "certification_level": "not_ready"
            },
            "recommendations": [],
            "next_steps": []
        }

        # Configuration
        self.config = self._load_configuration()

        # Test phases configuration
        self.phases = {
            "prerequisites": {
                "name": "System Prerequisites",
                "script": str(self.project_root / "validate_oracle1_deployment.sh"),
                "args": ["--phase=prereq"],
                "weight": 0.15,
                "required": True,
                "timeout": 300
            },
            "arm64_compatibility": {
                "name": "ARM64 Compatibility",
                "script": str(self.project_root / "tests" / "oracle1" / "test_arm64_compatibility.py"),
                "args": [str(self.project_root)],
                "weight": 0.25,
                "required": True,
                "timeout": 1800
            },
            "service_integration": {
                "name": "Service Integration",
                "script": str(self.project_root / "tests" / "oracle1" / "test_service_integration.py"),
                "args": [str(self.project_root)],
                "weight": 0.25,
                "required": True,
                "timeout": 600
            },
            "performance_benchmarks": {
                "name": "Performance Benchmarks",
                "script": str(self.project_root / "tests" / "oracle1" / "test_performance_benchmarks.py"),
                "args": [str(self.project_root)],
                "weight": 0.15,
                "required": False,
                "timeout": 900
            },
            "security_compliance": {
                "name": "Security & Compliance",
                "script": str(self.project_root / "tests" / "oracle1" / "test_security_compliance.py"),
                "args": [str(self.project_root)],
                "weight": 0.20,
                "required": True,
                "timeout": 300
            }
        }

    def _load_configuration(self) -> Dict:
        """Load validation configuration."""
        default_config = {
            "thresholds": {
                "min_success_rate": 80.0,
                "max_critical_issues": 0,
                "max_failed_tests": 5,
                "min_security_score": 70.0
            },
            "deployment_levels": {
                "production_ready": {"min_score": 90, "max_critical": 0, "max_failed": 2},
                "staging_ready": {"min_score": 80, "max_critical": 0, "max_failed": 5},
                "development_ready": {"min_score": 70, "max_critical": 1, "max_failed": 10}
            },
            "quick_mode": {
                "enabled": False,
                "skip_phases": ["performance_benchmarks"],
                "reduced_timeouts": True
            },
            "parallel_execution": {
                "enabled": True,
                "max_workers": 3
            },
            "auto_fix": {
                "enabled": False,
                "safe_fixes_only": True
            }
        }

        if self.config_file and self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")

        return default_config

    async def run_phase(self, phase_name: str, phase_config: Dict,
                       options: Dict) -> Tuple[bool, Dict]:
        """Run a single validation phase."""
        logger.info(f"Starting phase: {phase_config['name']}")

        phase_start_time = time.time()
        phase_result = {
            "name": phase_config["name"],
            "phase_id": phase_name,
            "start_time": phase_start_time,
            "end_time": None,
            "duration": 0.0,
            "status": "running",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "results_file": None,
            "parsed_results": {},
            "score": 0.0,
            "issues": []
        }

        try:
            # Prepare command
            script_path = Path(phase_config["script"])
            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")

            cmd = []
            if script_path.suffix == '.py':
                cmd = [sys.executable, str(script_path)]
            elif script_path.suffix == '.sh':
                cmd = ['bash', str(script_path)]
            else:
                raise ValueError(f"Unsupported script type: {script_path.suffix}")

            cmd.extend(phase_config.get("args", []))

            # Add options to command
            if options.get("verbose"):
                cmd.append("--verbose")
            if options.get("fix"):
                cmd.append("--fix")

            # Execute phase
            timeout = phase_config.get("timeout", 600)
            if self.config["quick_mode"]["enabled"] and self.config["quick_mode"]["reduced_timeouts"]:
                timeout = min(timeout, 300)

            logger.info(f"Executing: {' '.join(cmd)}")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.project_root)
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )

                phase_result["exit_code"] = process.returncode
                phase_result["stdout"] = stdout.decode('utf-8', errors='ignore')
                phase_result["stderr"] = stderr.decode('utf-8', errors='ignore')

            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                phase_result["exit_code"] = -1
                phase_result["stderr"] = f"Phase timeout after {timeout} seconds"
                logger.error(f"Phase {phase_name} timed out after {timeout} seconds")

            # Parse results if available
            if phase_result["exit_code"] == 0:
                phase_result["status"] = "completed"
                await self._parse_phase_results(phase_name, phase_result)
            else:
                phase_result["status"] = "failed"
                logger.error(f"Phase {phase_name} failed with exit code {phase_result['exit_code']}")

        except Exception as e:
            phase_result["status"] = "error"
            phase_result["stderr"] = str(e)
            logger.error(f"Phase {phase_name} error: {e}")

        finally:
            phase_result["end_time"] = time.time()
            phase_result["duration"] = phase_result["end_time"] - phase_start_time

        # Log phase completion
        logger.info(f"Phase {phase_config['name']} completed in {phase_result['duration']:.2f}s: {phase_result['status']}")

        return phase_result["status"] == "completed", phase_result

    async def _parse_phase_results(self, phase_name: str, phase_result: Dict):
        """Parse results from a completed phase."""
        try:
            # Look for JSON result files in output directory
            result_patterns = [
                f"arm64_compatibility_*.json",
                f"service_integration_*.json",
                f"performance_benchmarks_*.json",
                f"security_compliance_*.json"
            ]

            latest_result_file = None
            latest_time = 0

            for pattern in result_patterns:
                for file_path in self.output_dir.glob(pattern):
                    if file_path.stat().st_mtime > latest_time:
                        latest_time = file_path.stat().st_mtime
                        latest_result_file = file_path

            if latest_result_file:
                with open(latest_result_file, 'r') as f:
                    results_data = json.load(f)

                phase_result["results_file"] = str(latest_result_file)
                phase_result["parsed_results"] = results_data

                # Extract key metrics based on phase type
                if phase_name == "arm64_compatibility":
                    summary = results_data.get("summary", {})
                    phase_result["score"] = summary.get("success_rate", 0.0)

                elif phase_name == "service_integration":
                    summary = results_data.get("summary", {})
                    phase_result["score"] = summary.get("success_rate", 0.0)

                elif phase_name == "performance_benchmarks":
                    summary = results_data.get("summary", {})
                    phase_result["score"] = summary.get("success_rate", 0.0)

                elif phase_name == "security_compliance":
                    phase_result["score"] = results_data.get("security_score", 0.0)
                    summary = results_data.get("summary", {})
                    phase_result["issues"] = [
                        f"Critical: {summary.get('critical_issues', 0)}",
                        f"High: {summary.get('high_issues', 0)}",
                        f"Medium: {summary.get('medium_issues', 0)}"
                    ]

                logger.info(f"Parsed results for {phase_name}: score={phase_result['score']:.1f}")

        except Exception as e:
            logger.warning(f"Failed to parse results for {phase_name}: {e}")

    async def run_validation(self, options: Dict) -> Dict:
        """Run the complete validation process."""
        logger.info(f"Starting ORACLE1 comprehensive validation - Session: {self.session_id}")

        validation_start_time = time.time()

        # Filter phases based on options
        phases_to_run = {}
        if options.get("phase"):
            if options["phase"] in self.phases:
                phases_to_run = {options["phase"]: self.phases[options["phase"]]}
            else:
                logger.error(f"Invalid phase: {options['phase']}")
                return self.results
        else:
            phases_to_run = self.phases.copy()

            # Skip phases in quick mode
            if self.config["quick_mode"]["enabled"]:
                for skip_phase in self.config["quick_mode"]["skip_phases"]:
                    phases_to_run.pop(skip_phase, None)

        self.results["summary"]["total_phases"] = len(phases_to_run)

        # Run phases
        if options.get("parallel") and self.config["parallel_execution"]["enabled"]:
            # Run compatible phases in parallel
            await self._run_phases_parallel(phases_to_run, options)
        else:
            # Run phases sequentially
            await self._run_phases_sequential(phases_to_run, options)

        # Calculate overall results
        self._calculate_overall_results()

        # Generate recommendations
        self._generate_recommendations()

        # Determine deployment readiness
        self._determine_deployment_readiness()

        # Save master results
        self._save_master_results()

        validation_end_time = time.time()
        self.results["total_duration"] = validation_end_time - validation_start_time

        logger.info(f"Validation completed in {self.results['total_duration']:.2f}s")

        return self.results

    async def _run_phases_sequential(self, phases: Dict, options: Dict):
        """Run validation phases sequentially."""
        for phase_name, phase_config in phases.items():
            success, phase_result = await self.run_phase(phase_name, phase_config, options)

            self.results["validation_phases"][phase_name] = phase_result

            if success:
                self.results["summary"]["completed_phases"] += 1
            else:
                self.results["summary"]["failed_phases"] += 1

                # Stop on critical phase failure if required
                if phase_config.get("required", False) and not options.get("continue_on_failure", False):
                    logger.error(f"Critical phase {phase_name} failed, stopping validation")
                    break

    async def _run_phases_parallel(self, phases: Dict, options: Dict):
        """Run compatible validation phases in parallel."""
        # Group phases by dependencies
        independent_phases = ["arm64_compatibility", "security_compliance"]
        dependent_phases = ["service_integration", "performance_benchmarks"]

        # Run independent phases first
        independent_tasks = []
        for phase_name in independent_phases:
            if phase_name in phases:
                task = asyncio.create_task(
                    self.run_phase(phase_name, phases[phase_name], options)
                )
                independent_tasks.append((phase_name, task))

        # Wait for independent phases
        for phase_name, task in independent_tasks:
            try:
                success, phase_result = await task
                self.results["validation_phases"][phase_name] = phase_result

                if success:
                    self.results["summary"]["completed_phases"] += 1
                else:
                    self.results["summary"]["failed_phases"] += 1

            except Exception as e:
                logger.error(f"Phase {phase_name} failed with exception: {e}")
                self.results["summary"]["failed_phases"] += 1

        # Run dependent phases sequentially
        for phase_name in dependent_phases:
            if phase_name in phases:
                success, phase_result = await self.run_phase(phase_name, phases[phase_name], options)
                self.results["validation_phases"][phase_name] = phase_result

                if success:
                    self.results["summary"]["completed_phases"] += 1
                else:
                    self.results["summary"]["failed_phases"] += 1

    def _calculate_overall_results(self):
        """Calculate overall validation results."""
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warnings = 0
        critical_issues = 0

        weighted_score = 0.0
        total_weight = 0.0

        for phase_name, phase_result in self.results["validation_phases"].items():
            phase_config = self.phases.get(phase_name, {})
            weight = phase_config.get("weight", 0.0)

            # Extract metrics from parsed results
            parsed_results = phase_result.get("parsed_results", {})

            if "summary" in parsed_results:
                summary = parsed_results["summary"]
                total_tests += summary.get("total_tests", 0)
                passed_tests += summary.get("passed_tests", 0)
                failed_tests += summary.get("failed_tests", 0)
                warnings += summary.get("warnings", 0)
                critical_issues += summary.get("critical_issues", 0)

            # Add weighted score
            if weight > 0:
                phase_score = phase_result.get("score", 0.0)
                weighted_score += phase_score * weight
                total_weight += weight

        # Update summary
        self.results["summary"]["total_tests"] = total_tests
        self.results["summary"]["passed_tests"] = passed_tests
        self.results["summary"]["failed_tests"] = failed_tests
        self.results["summary"]["warnings"] = warnings
        self.results["summary"]["critical_issues"] = critical_issues

        if total_weight > 0:
            self.results["summary"]["overall_score"] = weighted_score / total_weight
        else:
            self.results["summary"]["overall_score"] = 0.0

    def _generate_recommendations(self):
        """Generate recommendations based on validation results."""
        recommendations = []

        # Check critical issues
        if self.results["summary"]["critical_issues"] > 0:
            recommendations.append(
                f"🚨 Address {self.results['summary']['critical_issues']} critical security issues before deployment"
            )

        # Check failed tests
        failed_tests = self.results["summary"]["failed_tests"]
        if failed_tests > self.config["thresholds"]["max_failed_tests"]:
            recommendations.append(
                f"🔧 Fix {failed_tests} failed tests (threshold: {self.config['thresholds']['max_failed_tests']})"
            )

        # Check overall score
        score = self.results["summary"]["overall_score"]
        min_score = self.config["thresholds"]["min_success_rate"]
        if score < min_score:
            recommendations.append(
                f"📊 Improve overall validation score: {score:.1f}% (target: {min_score}%)"
            )

        # Phase-specific recommendations
        for phase_name, phase_result in self.results["validation_phases"].items():
            if phase_result["status"] == "failed":
                recommendations.append(
                    f"❌ Fix {phase_result['name']} phase failures"
                )
            elif phase_result.get("score", 100) < 80:
                recommendations.append(
                    f"⚠️ Improve {phase_result['name']} score: {phase_result.get('score', 0):.1f}%"
                )

        # Add general recommendations
        if not recommendations:
            recommendations.append("✅ All validation checks passed - deployment ready")
        else:
            recommendations.append("📋 Review detailed test results in validation_results/ directory")
            recommendations.append("🔄 Re-run validation after addressing issues")

        self.results["recommendations"] = recommendations

    def _determine_deployment_readiness(self):
        """Determine deployment readiness and certification level."""
        score = self.results["summary"]["overall_score"]
        critical_issues = self.results["summary"]["critical_issues"]
        failed_tests = self.results["summary"]["failed_tests"]

        deployment_levels = self.config["deployment_levels"]

        # Check against deployment levels
        if (score >= deployment_levels["production_ready"]["min_score"] and
            critical_issues <= deployment_levels["production_ready"]["max_critical"] and
            failed_tests <= deployment_levels["production_ready"]["max_failed"]):
            self.results["summary"]["deployment_ready"] = True
            self.results["summary"]["certification_level"] = "production_ready"

        elif (score >= deployment_levels["staging_ready"]["min_score"] and
              critical_issues <= deployment_levels["staging_ready"]["max_critical"] and
              failed_tests <= deployment_levels["staging_ready"]["max_failed"]):
            self.results["summary"]["deployment_ready"] = True
            self.results["summary"]["certification_level"] = "staging_ready"

        elif (score >= deployment_levels["development_ready"]["min_score"] and
              critical_issues <= deployment_levels["development_ready"]["max_critical"] and
              failed_tests <= deployment_levels["development_ready"]["max_failed"]):
            self.results["summary"]["deployment_ready"] = True
            self.results["summary"]["certification_level"] = "development_ready"

        else:
            self.results["summary"]["deployment_ready"] = False
            self.results["summary"]["certification_level"] = "not_ready"

        # Add next steps
        if self.results["summary"]["deployment_ready"]:
            level = self.results["summary"]["certification_level"].replace("_", " ").title()
            self.results["next_steps"] = [
                f"✅ ORACLE1 deployment certified for {level} environment",
                "🚀 Proceed with deployment to target ARM server (100.96.197.84)",
                "📊 Monitor deployment performance and stability",
                "🔄 Run post-deployment validation to confirm operation"
            ]
        else:
            self.results["next_steps"] = [
                "❌ ORACLE1 deployment NOT ready - requires fixes",
                "🔧 Address recommendations listed above",
                "🧪 Re-run validation after implementing fixes",
                "📋 Review detailed test results for specific issues"
            ]

    def _save_master_results(self):
        """Save master validation results."""
        results_file = self.output_dir / f"oracle1_validation_master_{self.timestamp}.json"

        try:
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)

            logger.info(f"Master validation results saved to: {results_file}")

            # Also save a latest results symlink
            latest_link = self.output_dir / "oracle1_validation_latest.json"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(results_file.name)

        except Exception as e:
            logger.error(f"Failed to save master results: {e}")

    def print_summary(self):
        """Print validation summary to console."""
        print("\n" + "="*80)
        print(f"ORACLE1 COMPREHENSIVE DEPLOYMENT VALIDATION SUMMARY")
        print("="*80)
        print(f"Session ID: {self.session_id}")
        print(f"Duration: {self.results.get('total_duration', 0):.2f} seconds")
        print("")

        # Overall metrics
        summary = self.results["summary"]
        print("📊 OVERALL METRICS")
        print(f"  Overall Score: {summary['overall_score']:.1f}/100")
        print(f"  Phases: {summary['completed_phases']}/{summary['total_phases']} completed")
        print(f"  Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
        print(f"  Warnings: {summary['warnings']}")
        print(f"  Critical Issues: {summary['critical_issues']}")
        print("")

        # Deployment status
        print("🎯 DEPLOYMENT STATUS")
        if summary["deployment_ready"]:
            level = summary["certification_level"].replace("_", " ").title()
            print(f"  ✅ READY for {level}")
        else:
            print(f"  ❌ NOT READY for deployment")
        print("")

        # Phase results
        print("📋 PHASE RESULTS")
        for phase_name, phase_result in self.results["validation_phases"].items():
            status_icon = "✅" if phase_result["status"] == "completed" else "❌"
            score = phase_result.get("score", 0)
            duration = phase_result.get("duration", 0)
            print(f"  {status_icon} {phase_result['name']}: {score:.1f}% ({duration:.1f}s)")
        print("")

        # Recommendations
        if self.results["recommendations"]:
            print("💡 RECOMMENDATIONS")
            for rec in self.results["recommendations"]:
                print(f"  {rec}")
            print("")

        # Next steps
        if self.results["next_steps"]:
            print("🚀 NEXT STEPS")
            for step in self.results["next_steps"]:
                print(f"  {step}")

        print("="*80)
        print(f"Results saved to: {self.output_dir}")
        print("="*80)


async def main():
    """Main entry point for comprehensive validation."""
    parser = argparse.ArgumentParser(
        description="ORACLE1 Comprehensive Deployment Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--phase", type=str,
                       choices=["prerequisites", "arm64_compatibility", "service_integration",
                               "performance_benchmarks", "security_compliance"],
                       help="Run specific validation phase only")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tests in parallel where possible")
    parser.add_argument("--output-dir", type=str,
                       help="Custom output directory for results")
    parser.add_argument("--config", type=str,
                       help="Custom configuration file")
    parser.add_argument("--fix", action="store_true",
                       help="Attempt automatic fixes")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable detailed logging")
    parser.add_argument("--no-docker", action="store_true",
                       help="Skip Docker-dependent tests")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick validation (reduced test set)")
    parser.add_argument("--continue-on-failure", action="store_true",
                       help="Continue validation even if critical phases fail")
    parser.add_argument("--project-root", type=str, default="/home/starlord/Projects/Bev",
                       help="Project root directory")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create orchestrator
    orchestrator = ValidationOrchestrator(
        project_root=args.project_root,
        output_dir=args.output_dir,
        config_file=args.config
    )

    # Apply quick mode if requested
    if args.quick:
        orchestrator.config["quick_mode"]["enabled"] = True

    # Prepare options
    options = {
        "phase": args.phase,
        "parallel": args.parallel,
        "fix": args.fix,
        "verbose": args.verbose,
        "no_docker": args.no_docker,
        "quick": args.quick,
        "continue_on_failure": args.continue_on_failure
    }

    try:
        # Run validation
        results = await orchestrator.run_validation(options)

        # Print summary
        orchestrator.print_summary()

        # Exit with appropriate code
        if results["summary"]["deployment_ready"]:
            print(f"\n🎉 ORACLE1 deployment validation successful!")
            print(f"   Certified for: {results['summary']['certification_level'].replace('_', ' ').title()}")
            sys.exit(0)
        else:
            print(f"\n💥 ORACLE1 deployment validation failed!")
            print(f"   Critical issues: {results['summary']['critical_issues']}")
            print(f"   Failed tests: {results['summary']['failed_tests']}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())