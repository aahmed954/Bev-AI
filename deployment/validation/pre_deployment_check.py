#!/usr/bin/env python3

"""
BEV OSINT Framework - Pre-Deployment Validation System
Comprehensive system checks before Phase 7, 8, 9 deployment
"""

import sys
import os
import json
import subprocess
import socket
import platform
import psutil
import docker
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

@dataclass
class ValidationResult:
    """Validation result data structure"""
    check_name: str
    status: str  # PASS, WARN, FAIL
    message: str
    details: Optional[Dict] = None
    severity: str = "INFO"  # INFO, WARN, ERROR, CRITICAL

class PreDeploymentValidator:
    """Comprehensive pre-deployment validation system"""

    def __init__(self, phases: List[str] = None):
        self.phases = phases or ["7", "8", "9"]
        self.project_root = project_root
        self.results: List[ValidationResult] = []
        self.docker_client = None
        self.critical_failures = 0
        self.warnings = 0

        # Setup logging
        self.setup_logging()

        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")
            self.add_result("docker_client", "FAIL", f"Docker client initialization failed: {e}", severity="CRITICAL")

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.project_root / "logs" / "deployment"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"pre_deployment_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def add_result(self, check_name: str, status: str, message: str, details: Dict = None, severity: str = "INFO"):
        """Add a validation result"""
        result = ValidationResult(check_name, status, message, details, severity)
        self.results.append(result)

        if status == "FAIL" and severity in ["ERROR", "CRITICAL"]:
            self.critical_failures += 1
        elif status == "WARN":
            self.warnings += 1

        # Log the result
        log_level = {
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }.get(severity, logging.INFO)

        self.logger.log(log_level, f"{check_name}: {status} - {message}")

    def check_system_requirements(self) -> None:
        """Check basic system requirements"""
        self.logger.info("Checking system requirements...")

        # Check OS
        os_info = platform.system()
        if os_info == "Linux":
            self.add_result("os_check", "PASS", f"Operating system: {os_info}")
        else:
            self.add_result("os_check", "WARN", f"Untested OS: {os_info}", severity="WARN")

        # Check Python version
        python_version = platform.python_version()
        if python_version >= "3.8":
            self.add_result("python_version", "PASS", f"Python version: {python_version}")
        else:
            self.add_result("python_version", "FAIL", f"Python version {python_version} < 3.8", severity="ERROR")

        # Check disk space
        total, used, free = self._get_disk_usage()
        free_gb = free / (1024**3)
        if free_gb >= 100:
            self.add_result("disk_space", "PASS", f"Free disk space: {free_gb:.1f}GB")
        elif free_gb >= 50:
            self.add_result("disk_space", "WARN", f"Limited disk space: {free_gb:.1f}GB", severity="WARN")
        else:
            self.add_result("disk_space", "FAIL", f"Insufficient disk space: {free_gb:.1f}GB", severity="ERROR")

        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        if memory_gb >= 32:
            self.add_result("memory", "PASS", f"Total memory: {memory_gb:.1f}GB")
        elif memory_gb >= 16:
            self.add_result("memory", "WARN", f"Limited memory: {memory_gb:.1f}GB", severity="WARN")
        else:
            self.add_result("memory", "FAIL", f"Insufficient memory: {memory_gb:.1f}GB", severity="ERROR")

        # Check CPU cores
        cpu_count = psutil.cpu_count(logical=False)
        if cpu_count >= 8:
            self.add_result("cpu_cores", "PASS", f"CPU cores: {cpu_count}")
        elif cpu_count >= 4:
            self.add_result("cpu_cores", "WARN", f"Limited CPU cores: {cpu_count}", severity="WARN")
        else:
            self.add_result("cpu_cores", "FAIL", f"Insufficient CPU cores: {cpu_count}", severity="ERROR")

    def _get_disk_usage(self) -> Tuple[int, int, int]:
        """Get disk usage for project root"""
        statvfs = os.statvfs(self.project_root)
        total = statvfs.f_frsize * statvfs.f_blocks
        free = statvfs.f_frsize * statvfs.f_bavail
        used = total - free
        return total, used, free

    def check_docker_environment(self) -> None:
        """Check Docker environment"""
        self.logger.info("Checking Docker environment...")

        if not self.docker_client:
            return

        try:
            # Check Docker version
            docker_version = self.docker_client.version()
            self.add_result("docker_version", "PASS", f"Docker version: {docker_version['Version']}")

            # Check Docker Compose
            try:
                result = subprocess.run(['docker-compose', '--version'],
                                     capture_output=True, text=True, check=True)
                self.add_result("docker_compose", "PASS", f"Docker Compose: {result.stdout.strip()}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.add_result("docker_compose", "FAIL", "Docker Compose not available", severity="CRITICAL")

            # Check Docker daemon status
            try:
                self.docker_client.ping()
                self.add_result("docker_daemon", "PASS", "Docker daemon is running")
            except Exception as e:
                self.add_result("docker_daemon", "FAIL", f"Docker daemon not responding: {e}", severity="CRITICAL")

            # Check disk space for Docker
            try:
                df_info = self.docker_client.df()
                volumes_size = sum(vol.get('Size', 0) for vol in df_info.get('Volumes', []))
                images_size = sum(img.get('Size', 0) for img in df_info.get('Images', []))
                total_docker_size = (volumes_size + images_size) / (1024**3)

                self.add_result("docker_disk_usage", "PASS", f"Docker disk usage: {total_docker_size:.1f}GB")
            except Exception as e:
                self.add_result("docker_disk_usage", "WARN", f"Could not check Docker disk usage: {e}", severity="WARN")

        except Exception as e:
            self.add_result("docker_environment", "FAIL", f"Docker environment check failed: {e}", severity="CRITICAL")

    def check_gpu_support(self) -> None:
        """Check GPU and NVIDIA Docker support"""
        self.logger.info("Checking GPU support...")

        # Check NVIDIA driver
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
            # Parse GPU info
            lines = result.stdout.split('\n')
            gpu_info = []
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Tesla' in line or 'Quadro' in line:
                    gpu_info.append(line.strip())

            if gpu_info:
                self.add_result("nvidia_driver", "PASS", f"NVIDIA driver available. GPUs: {len(gpu_info)}")
            else:
                self.add_result("nvidia_driver", "WARN", "NVIDIA driver available but no GPUs detected", severity="WARN")

        except (subprocess.CalledProcessError, FileNotFoundError):
            self.add_result("nvidia_driver", "FAIL", "NVIDIA driver not available", severity="ERROR")
            return

        # Check NVIDIA Docker runtime
        if self.docker_client:
            try:
                # Try to run a CUDA container
                result = self.docker_client.containers.run(
                    "nvidia/cuda:11.2.2-base-ubuntu20.04",
                    "nvidia-smi",
                    remove=True,
                    device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
                )
                self.add_result("nvidia_docker", "PASS", "NVIDIA Docker runtime available")
            except Exception as e:
                self.add_result("nvidia_docker", "FAIL", f"NVIDIA Docker runtime not available: {e}", severity="ERROR")

    def check_network_configuration(self) -> None:
        """Check network configuration and port availability"""
        self.logger.info("Checking network configuration...")

        # Define required ports for each phase
        phase_ports = {
            "7": [8001, 8002, 8003, 8004],  # Phase 7 services
            "8": [8005, 8006, 8007, 8008],  # Phase 8 services
            "9": [8009, 8010, 8011, 8012],  # Phase 9 services
        }

        # Check port availability
        unavailable_ports = []
        for phase in self.phases:
            if phase in phase_ports:
                for port in phase_ports[phase]:
                    if self._is_port_in_use(port):
                        unavailable_ports.append(port)

        if unavailable_ports:
            self.add_result("port_availability", "FAIL",
                          f"Ports in use: {unavailable_ports}",
                          {"unavailable_ports": unavailable_ports},
                          severity="ERROR")
        else:
            all_ports = []
            for phase in self.phases:
                if phase in phase_ports:
                    all_ports.extend(phase_ports[phase])
            self.add_result("port_availability", "PASS", f"All required ports available: {all_ports}")

        # Check if BEV network exists
        if self.docker_client:
            try:
                networks = self.docker_client.networks.list(names=["bev_osint"])
                if networks:
                    network = networks[0]
                    self.add_result("bev_network", "PASS", f"BEV OSINT network exists: {network.id[:12]}")
                else:
                    self.add_result("bev_network", "WARN", "BEV OSINT network does not exist (will be created)", severity="WARN")
            except Exception as e:
                self.add_result("bev_network", "WARN", f"Could not check BEV network: {e}", severity="WARN")

    def _is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return False
            except OSError:
                return True

    def check_dependencies(self) -> None:
        """Check for required dependencies and services"""
        self.logger.info("Checking dependencies...")

        # Check core infrastructure services
        required_services = [
            "postgres", "neo4j", "elasticsearch", "kafka-1", "redis", "influxdb"
        ]

        if self.docker_client:
            running_containers = {c.name: c for c in self.docker_client.containers.list()}

            missing_services = []
            for service in required_services:
                if service not in running_containers:
                    missing_services.append(service)
                else:
                    container = running_containers[service]
                    if container.status != "running":
                        missing_services.append(f"{service} (not running)")

            if missing_services:
                self.add_result("core_services", "WARN",
                              f"Missing core services: {missing_services}",
                              {"missing_services": missing_services},
                              severity="WARN")
            else:
                self.add_result("core_services", "PASS", "All core services are running")

        # Check for required Python packages
        required_packages = [
            "docker", "psutil", "requests", "pyyaml", "pytest"
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            self.add_result("python_packages", "FAIL",
                          f"Missing Python packages: {missing_packages}",
                          {"missing_packages": missing_packages},
                          severity="ERROR")
        else:
            self.add_result("python_packages", "PASS", "All required Python packages are available")

    def check_configuration_files(self) -> None:
        """Check for required configuration files"""
        self.logger.info("Checking configuration files...")

        # Check main environment file
        env_file = self.project_root / ".env"
        if env_file.exists():
            self.add_result("env_file", "PASS", "Environment file exists")

            # Check for required environment variables
            self._validate_env_file(env_file)
        else:
            self.add_result("env_file", "FAIL", "Environment file missing", severity="CRITICAL")

        # Check phase-specific docker-compose files
        for phase in self.phases:
            compose_file = self.project_root / f"docker-compose-phase{phase}.yml"
            if compose_file.exists():
                self.add_result(f"compose_phase_{phase}", "PASS", f"Phase {phase} compose file exists")

                # Validate compose file
                self._validate_compose_file(compose_file, phase)
            else:
                self.add_result(f"compose_phase_{phase}", "FAIL",
                              f"Phase {phase} compose file missing", severity="CRITICAL")

    def _validate_env_file(self, env_file: Path) -> None:
        """Validate environment file contents"""
        try:
            with open(env_file) as f:
                content = f.read()

            # Check for critical environment variables
            critical_vars = [
                "POSTGRES_URI", "NEO4J_URI", "REDIS_URI", "KAFKA_BROKERS"
            ]

            missing_vars = []
            for var in critical_vars:
                if var not in content:
                    missing_vars.append(var)

            if missing_vars:
                self.add_result("env_variables", "WARN",
                              f"Missing environment variables: {missing_vars}",
                              {"missing_vars": missing_vars},
                              severity="WARN")
            else:
                self.add_result("env_variables", "PASS", "Critical environment variables present")

        except Exception as e:
            self.add_result("env_file_validation", "FAIL",
                          f"Could not validate environment file: {e}", severity="ERROR")

    def _validate_compose_file(self, compose_file: Path, phase: str) -> None:
        """Validate docker-compose file"""
        try:
            # Check if file is valid YAML and can be parsed by docker-compose
            result = subprocess.run(
                ['docker-compose', '-f', str(compose_file), 'config'],
                capture_output=True, text=True, check=True
            )
            self.add_result(f"compose_validation_phase_{phase}", "PASS",
                          f"Phase {phase} compose file is valid")

        except subprocess.CalledProcessError as e:
            self.add_result(f"compose_validation_phase_{phase}", "FAIL",
                          f"Phase {phase} compose file validation failed: {e.stderr}",
                          severity="ERROR")
        except Exception as e:
            self.add_result(f"compose_validation_phase_{phase}", "WARN",
                          f"Could not validate Phase {phase} compose file: {e}",
                          severity="WARN")

    def check_security_requirements(self) -> None:
        """Check security-related requirements"""
        self.logger.info("Checking security requirements...")

        # Check if running as root (security concern)
        if os.geteuid() == 0:
            self.add_result("root_user", "WARN",
                          "Running as root - consider using non-root user",
                          severity="WARN")
        else:
            self.add_result("root_user", "PASS", "Not running as root")

        # Check Docker socket permissions
        docker_socket = Path("/var/run/docker.sock")
        if docker_socket.exists():
            stat = docker_socket.stat()
            # Check if current user can access Docker socket
            if os.access(docker_socket, os.R_OK | os.W_OK):
                self.add_result("docker_permissions", "PASS", "Docker socket accessible")
            else:
                self.add_result("docker_permissions", "FAIL",
                              "Cannot access Docker socket", severity="CRITICAL")

        # Check for firewall status
        try:
            result = subprocess.run(['ufw', 'status'], capture_output=True, text=True)
            if "Status: active" in result.stdout:
                self.add_result("firewall", "WARN",
                              "UFW firewall is active - may block services", severity="WARN")
            else:
                self.add_result("firewall", "PASS", "UFW firewall not blocking")
        except FileNotFoundError:
            self.add_result("firewall", "PASS", "UFW not installed")

    def check_resource_availability(self) -> None:
        """Check current resource availability"""
        self.logger.info("Checking resource availability...")

        # Calculate resource requirements for all phases
        memory_requirements = {
            "7": 15,  # 2+3+4+6 GB for Phase 7 services
            "8": 20,  # 4+3+5+8 GB for Phase 8 services
            "9": 28,  # 6+8+4+10 GB for Phase 9 services
        }

        cpu_requirements = {
            "7": 6.5,  # 1+1.5+2+2 cores for Phase 7 services
            "8": 8.5,  # 2+1.5+2.5+3 cores for Phase 8 services
            "9": 11.5, # 2.5+3+2+4 cores for Phase 9 services
        }

        total_memory_needed = sum(memory_requirements[phase] for phase in self.phases)
        total_cpu_needed = sum(cpu_requirements[phase] for phase in self.phases)

        # Check available memory
        memory = psutil.virtual_memory()
        available_memory = memory.available / (1024**3)

        if available_memory >= total_memory_needed:
            self.add_result("memory_availability", "PASS",
                          f"Sufficient memory: {available_memory:.1f}GB available, {total_memory_needed}GB needed")
        else:
            self.add_result("memory_availability", "FAIL",
                          f"Insufficient memory: {available_memory:.1f}GB available, {total_memory_needed}GB needed",
                          severity="ERROR")

        # Check CPU load
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent < 70:
            self.add_result("cpu_availability", "PASS", f"CPU load acceptable: {cpu_percent:.1f}%")
        else:
            self.add_result("cpu_availability", "WARN",
                          f"High CPU load: {cpu_percent:.1f}%", severity="WARN")

    def run_all_checks(self) -> bool:
        """Run all validation checks"""
        self.logger.info(f"Starting pre-deployment validation for phases: {', '.join(self.phases)}")

        checks = [
            self.check_system_requirements,
            self.check_docker_environment,
            self.check_gpu_support,
            self.check_network_configuration,
            self.check_dependencies,
            self.check_configuration_files,
            self.check_security_requirements,
            self.check_resource_availability,
        ]

        for check in checks:
            try:
                check()
            except Exception as e:
                self.logger.error(f"Check {check.__name__} failed with exception: {e}")
                self.add_result(check.__name__, "FAIL", f"Check failed with exception: {e}", severity="ERROR")

        return self.critical_failures == 0

    def generate_report(self) -> str:
        """Generate a comprehensive validation report"""
        report = []
        report.append("="*80)
        report.append("BEV OSINT Framework - Pre-Deployment Validation Report")
        report.append("="*80)
        report.append(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Phases to Deploy: {', '.join(self.phases)}")
        report.append(f"Total Checks: {len(self.results)}")
        report.append(f"Critical Failures: {self.critical_failures}")
        report.append(f"Warnings: {self.warnings}")
        report.append("")

        # Summary
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warned = sum(1 for r in self.results if r.status == "WARN")

        report.append("SUMMARY:")
        report.append(f"  ✅ Passed: {passed}")
        report.append(f"  ❌ Failed: {failed}")
        report.append(f"  ⚠️  Warnings: {warned}")
        report.append("")

        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 50)

        for result in self.results:
            status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️"}.get(result.status, "❓")
            report.append(f"{status_icon} {result.check_name}: {result.status}")
            report.append(f"    {result.message}")
            if result.details:
                for key, value in result.details.items():
                    report.append(f"    {key}: {value}")
            report.append("")

        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 50)

        if self.critical_failures > 0:
            report.append("🚨 CRITICAL ISSUES FOUND - DEPLOYMENT NOT RECOMMENDED")
            report.append("   Please resolve all critical failures before proceeding.")
        elif self.warnings > 0:
            report.append("⚠️  WARNINGS DETECTED - PROCEED WITH CAUTION")
            report.append("   Review warnings and consider resolving them.")
        else:
            report.append("✅ ALL CHECKS PASSED - DEPLOYMENT READY")
            report.append("   System is ready for deployment.")

        report.append("")
        report.append("="*80)

        return "\n".join(report)

    def save_report(self, filename: str = None) -> Path:
        """Save validation report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"pre_deployment_validation_{timestamp}.txt"

        report_path = self.project_root / "logs" / "deployment" / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write(self.generate_report())

        return report_path

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="BEV OSINT Pre-Deployment Validation")
    parser.add_argument("--phases", default="7,8,9",
                       help="Comma-separated list of phases to validate (default: 7,8,9)")
    parser.add_argument("--output", help="Output report file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Parse phases
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    # Create validator
    validator = PreDeploymentValidator(phases)

    # Run validation
    success = validator.run_all_checks()

    # Generate and display report
    report = validator.generate_report()
    print(report)

    # Save report
    report_path = validator.save_report(args.output)
    print(f"\nValidation report saved to: {report_path}")

    # Exit with appropriate code
    if success:
        print("\n✅ Validation successful - system ready for deployment")
        sys.exit(0)
    else:
        print(f"\n❌ Validation failed - {validator.critical_failures} critical issues found")
        sys.exit(1)

if __name__ == "__main__":
    main()