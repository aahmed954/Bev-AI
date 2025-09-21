#!/usr/bin/env python3
"""
ARM64 Compatibility Testing Suite for ORACLE1 Deployment

This module provides comprehensive testing for ARM64 architecture compatibility
including Docker builds, package installations, and runtime validation.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import docker
import psutil
import requests
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ARM64CompatibilityTester:
    """Comprehensive ARM64 compatibility testing for ORACLE1 services."""

    def __init__(self, project_root: str = "/home/starlord/Projects/Bev"):
        self.project_root = Path(project_root)
        self.docker_client = docker.from_env()
        self.results = {
            "timestamp": time.time(),
            "tests": [],
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0
            }
        }

        # ARM64 optimized base images to test
        self.arm64_base_images = [
            "python:3.11-slim-bookworm",
            "redis:7-alpine",
            "nginx:alpine",
            "influxdb:2.7-alpine",
            "telegraf:1.28-alpine",
            "prom/node-exporter:latest",
            "prom/prometheus:latest",
            "grafana/grafana:latest",
            "prom/alertmanager:latest",
            "hashicorp/vault:latest",
            "minio/minio:latest",
            "n8nio/n8n:latest",
            "ghcr.io/berriai/litellm:main-latest"
        ]

        # Critical ARM64 packages to test
        self.critical_packages = [
            "build-essential",
            "curl",
            "git",
            "libopenblas-dev",
            "liblapack-dev",
            "pkg-config",
            "python3-dev",
            "libffi-dev",
            "libssl-dev",
            "tor",
            "imagemagick",
            "ffmpeg"
        ]

        # Python packages for ARM64 testing
        self.python_packages = [
            "redis",
            "celery",
            "fastapi",
            "aiohttp",
            "numpy",
            "pandas",
            "requests",
            "psutil",
            "cryptography",
            "pillow",
            "opencv-python-headless",
            "torch",
            "transformers"
        ]

    def log_test_result(self, test_name: str, status: str, details: str = "",
                       execution_time: float = 0.0, metadata: Dict = None):
        """Log a test result."""
        result = {
            "test_name": test_name,
            "status": status,  # "PASS", "FAIL", "WARN"
            "details": details,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        self.results["tests"].append(result)
        self.results["summary"]["total"] += 1

        if status == "PASS":
            self.results["summary"]["passed"] += 1
            logger.info(f"[PASS] {test_name}: {details}")
        elif status == "FAIL":
            self.results["summary"]["failed"] += 1
            logger.error(f"[FAIL] {test_name}: {details}")
        elif status == "WARN":
            self.results["summary"]["warnings"] += 1
            logger.warning(f"[WARN] {test_name}: {details}")

    async def test_base_image_availability(self) -> bool:
        """Test ARM64 availability of all base images."""
        logger.info("Testing ARM64 base image availability...")

        all_passed = True
        for image in self.arm64_base_images:
            start_time = time.time()

            try:
                # Try to pull ARM64 version
                result = subprocess.run([
                    "docker", "pull", "--platform", "linux/arm64", image
                ], capture_output=True, text=True, timeout=120)

                execution_time = time.time() - start_time

                if result.returncode == 0:
                    # Get image details
                    inspect_result = subprocess.run([
                        "docker", "inspect", image
                    ], capture_output=True, text=True)

                    if inspect_result.returncode == 0:
                        image_data = json.loads(inspect_result.stdout)[0]
                        architecture = image_data.get("Architecture", "unknown")
                        size = image_data.get("Size", 0)

                        self.log_test_result(
                            f"base_image_{image.replace('/', '_').replace(':', '_')}",
                            "PASS",
                            f"ARM64 image available, size: {size / 1024 / 1024:.1f} MB",
                            execution_time,
                            {"image": image, "architecture": architecture, "size": size}
                        )
                    else:
                        self.log_test_result(
                            f"base_image_{image.replace('/', '_').replace(':', '_')}",
                            "WARN",
                            "Image pulled but inspection failed",
                            execution_time
                        )
                else:
                    self.log_test_result(
                        f"base_image_{image.replace('/', '_').replace(':', '_')}",
                        "FAIL",
                        f"ARM64 image not available: {result.stderr.strip()}",
                        execution_time
                    )
                    all_passed = False

            except subprocess.TimeoutExpired:
                self.log_test_result(
                    f"base_image_{image.replace('/', '_').replace(':', '_')}",
                    "FAIL",
                    "Image pull timeout (>120s)",
                    120.0
                )
                all_passed = False
            except Exception as e:
                self.log_test_result(
                    f"base_image_{image.replace('/', '_').replace(':', '_')}",
                    "FAIL",
                    f"Unexpected error: {str(e)}",
                    time.time() - start_time
                )
                all_passed = False

        return all_passed

    async def test_package_installation(self) -> bool:
        """Test ARM64 system package installation."""
        logger.info("Testing ARM64 system package installation...")

        # Create test Dockerfile
        test_dockerfile = self.project_root / "tests" / "oracle1" / "Dockerfile.package-test"
        test_dockerfile.parent.mkdir(parents=True, exist_ok=True)

        dockerfile_content = f"""
FROM python:3.11-slim-bookworm

# Test ARM64 system packages
RUN apt-get update && apt-get install -y \\
    {' '.join(self.critical_packages)} \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Test package availability
RUN which curl && curl --version
RUN which git && git --version
RUN python3 -c "import sys; print(f'Python {{sys.version}} on {{sys.platform}}')"

# Test build tools
RUN gcc --version
RUN pkg-config --version

# Create test file to verify success
RUN echo "ARM64 package installation successful" > /tmp/package_test_success
"""

        with open(test_dockerfile, 'w') as f:
            f.write(dockerfile_content)

        start_time = time.time()

        try:
            # Build test image
            result = subprocess.run([
                "docker", "build",
                "--platform", "linux/arm64",
                "-f", str(test_dockerfile),
                "-t", "bev-arm64-package-test",
                str(test_dockerfile.parent)
            ], capture_output=True, text=True, timeout=300)

            execution_time = time.time() - start_time

            if result.returncode == 0:
                # Test package functionality
                test_result = subprocess.run([
                    "docker", "run", "--rm", "--platform", "linux/arm64",
                    "bev-arm64-package-test",
                    "cat", "/tmp/package_test_success"
                ], capture_output=True, text=True)

                if test_result.returncode == 0:
                    self.log_test_result(
                        "arm64_package_installation",
                        "PASS",
                        "All critical packages installed successfully",
                        execution_time,
                        {"packages": self.critical_packages}
                    )

                    # Clean up test image
                    subprocess.run(["docker", "rmi", "bev-arm64-package-test"],
                                 capture_output=True)
                    return True
                else:
                    self.log_test_result(
                        "arm64_package_installation",
                        "FAIL",
                        f"Package verification failed: {test_result.stderr.strip()}",
                        execution_time
                    )
            else:
                self.log_test_result(
                    "arm64_package_installation",
                    "FAIL",
                    f"Package installation build failed: {result.stderr.strip()}",
                    execution_time
                )

        except subprocess.TimeoutExpired:
            self.log_test_result(
                "arm64_package_installation",
                "FAIL",
                "Package installation timeout (>300s)",
                300.0
            )
        except Exception as e:
            self.log_test_result(
                "arm64_package_installation",
                "FAIL",
                f"Unexpected error: {str(e)}",
                time.time() - start_time
            )
        finally:
            # Clean up test files
            if test_dockerfile.exists():
                test_dockerfile.unlink()

        return False

    async def test_python_packages(self) -> bool:
        """Test ARM64 Python package installation."""
        logger.info("Testing ARM64 Python package installation...")

        # Create test Dockerfile
        test_dockerfile = self.project_root / "tests" / "oracle1" / "Dockerfile.python-test"

        dockerfile_content = f"""
FROM python:3.11-slim-bookworm

# Install system dependencies for Python packages
RUN apt-get update && apt-get install -y \\
    build-essential \\
    libopenblas-dev \\
    liblapack-dev \\
    pkg-config \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Test Python packages installation
RUN pip install --no-cache-dir \\
    {' '.join(self.python_packages[:5])}  # Test first 5 critical packages

# Test package imports
RUN python -c "import redis, celery, fastapi, aiohttp, numpy; print('Critical packages imported successfully')"

# Create success marker
RUN echo "ARM64 Python packages installation successful" > /tmp/python_test_success
"""

        with open(test_dockerfile, 'w') as f:
            f.write(dockerfile_content)

        start_time = time.time()

        try:
            # Build test image
            result = subprocess.run([
                "docker", "build",
                "--platform", "linux/arm64",
                "-f", str(test_dockerfile),
                "-t", "bev-arm64-python-test",
                str(test_dockerfile.parent)
            ], capture_output=True, text=True, timeout=600)

            execution_time = time.time() - start_time

            if result.returncode == 0:
                # Test package functionality
                test_result = subprocess.run([
                    "docker", "run", "--rm", "--platform", "linux/arm64",
                    "bev-arm64-python-test",
                    "python", "-c", "import redis, celery, fastapi, aiohttp, numpy; print('All packages working')"
                ], capture_output=True, text=True)

                if test_result.returncode == 0:
                    self.log_test_result(
                        "arm64_python_packages",
                        "PASS",
                        "Critical Python packages installed and working",
                        execution_time,
                        {"packages": self.python_packages[:5]}
                    )

                    # Clean up test image
                    subprocess.run(["docker", "rmi", "bev-arm64-python-test"],
                                 capture_output=True)
                    return True
                else:
                    self.log_test_result(
                        "arm64_python_packages",
                        "FAIL",
                        f"Python package import failed: {test_result.stderr.strip()}",
                        execution_time
                    )
            else:
                self.log_test_result(
                    "arm64_python_packages",
                    "FAIL",
                    f"Python package build failed: {result.stderr.strip()}",
                    execution_time
                )

        except subprocess.TimeoutExpired:
            self.log_test_result(
                "arm64_python_packages",
                "FAIL",
                "Python package installation timeout (>600s)",
                600.0
            )
        except Exception as e:
            self.log_test_result(
                "arm64_python_packages",
                "FAIL",
                f"Unexpected error: {str(e)}",
                time.time() - start_time
            )
        finally:
            # Clean up test files
            if test_dockerfile.exists():
                test_dockerfile.unlink()

        return False

    async def test_service_builds(self) -> bool:
        """Test individual service Docker builds for ARM64."""
        logger.info("Testing individual service ARM64 builds...")

        docker_dir = self.project_root / "docker" / "oracle"
        compose_file = self.project_root / "docker-compose-oracle1-unified.yml"

        if not compose_file.exists():
            self.log_test_result(
                "service_builds_prerequisite",
                "FAIL",
                f"Compose file not found: {compose_file}"
            )
            return False

        # Get services that need building from compose file
        try:
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)

            build_services = []
            for service_name, service_config in compose_data.get('services', {}).items():
                if 'build' in service_config:
                    build_services.append(service_name)

            if not build_services:
                self.log_test_result(
                    "service_builds_detection",
                    "WARN",
                    "No services with build configuration found"
                )
                return True

            self.log_test_result(
                "service_builds_detection",
                "PASS",
                f"Found {len(build_services)} services to build: {', '.join(build_services)}"
            )

        except Exception as e:
            self.log_test_result(
                "service_builds_prerequisite",
                "FAIL",
                f"Failed to parse compose file: {str(e)}"
            )
            return False

        all_builds_passed = True

        for service in build_services:
            start_time = time.time()

            try:
                # Build individual service
                result = subprocess.run([
                    "docker", "compose", "-f", str(compose_file),
                    "build", "--platform", "linux/arm64", service
                ], capture_output=True, text=True, timeout=300, cwd=str(self.project_root))

                execution_time = time.time() - start_time

                if result.returncode == 0:
                    # Get built image info
                    image_name = f"bev_{service}"
                    inspect_result = subprocess.run([
                        "docker", "inspect", image_name
                    ], capture_output=True, text=True)

                    if inspect_result.returncode == 0:
                        image_data = json.loads(inspect_result.stdout)[0]
                        size = image_data.get("Size", 0)
                        architecture = image_data.get("Architecture", "unknown")

                        self.log_test_result(
                            f"service_build_{service}",
                            "PASS",
                            f"ARM64 build successful, size: {size / 1024 / 1024:.1f} MB, arch: {architecture}",
                            execution_time,
                            {"service": service, "image_size": size, "architecture": architecture}
                        )
                    else:
                        self.log_test_result(
                            f"service_build_{service}",
                            "PASS",
                            f"ARM64 build successful, inspection failed",
                            execution_time
                        )
                else:
                    self.log_test_result(
                        f"service_build_{service}",
                        "FAIL",
                        f"ARM64 build failed: {result.stderr.strip()[:200]}...",
                        execution_time
                    )
                    all_builds_passed = False

            except subprocess.TimeoutExpired:
                self.log_test_result(
                    f"service_build_{service}",
                    "FAIL",
                    "Build timeout (>300s)",
                    300.0
                )
                all_builds_passed = False
            except Exception as e:
                self.log_test_result(
                    f"service_build_{service}",
                    "FAIL",
                    f"Unexpected error: {str(e)}",
                    time.time() - start_time
                )
                all_builds_passed = False

        return all_builds_passed

    async def test_runtime_performance(self) -> bool:
        """Test ARM64 runtime performance characteristics."""
        logger.info("Testing ARM64 runtime performance...")

        # Test basic Python performance
        performance_test_script = """
import time
import sys
import platform

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def test_cpu_performance():
    start_time = time.time()
    result = fibonacci(35)
    end_time = time.time()
    return end_time - start_time, result

def test_memory_allocation():
    start_time = time.time()
    data = [[i * j for j in range(1000)] for i in range(1000)]
    end_time = time.time()
    return end_time - start_time, len(data)

# Run performance tests
print(f"Platform: {platform.machine()}")
print(f"Python: {sys.version}")

cpu_time, fib_result = test_cpu_performance()
print(f"CPU Test: {cpu_time:.3f}s (fibonacci(35) = {fib_result})")

mem_time, mem_result = test_memory_allocation()
print(f"Memory Test: {mem_time:.3f}s (allocated {mem_result} arrays)")

# Overall performance score (lower is better)
performance_score = cpu_time + mem_time
print(f"Performance Score: {performance_score:.3f}s")
"""

        test_dockerfile = self.project_root / "tests" / "oracle1" / "Dockerfile.performance-test"

        dockerfile_content = f"""
FROM python:3.11-slim-bookworm

# Copy performance test
COPY performance_test.py /app/performance_test.py
WORKDIR /app

# Run performance test
CMD ["python", "performance_test.py"]
"""

        with open(test_dockerfile.parent / "performance_test.py", 'w') as f:
            f.write(performance_test_script)

        with open(test_dockerfile, 'w') as f:
            f.write(dockerfile_content)

        start_time = time.time()

        try:
            # Build performance test image
            build_result = subprocess.run([
                "docker", "build",
                "--platform", "linux/arm64",
                "-f", str(test_dockerfile),
                "-t", "bev-arm64-performance-test",
                str(test_dockerfile.parent)
            ], capture_output=True, text=True, timeout=180)

            if build_result.returncode != 0:
                self.log_test_result(
                    "arm64_runtime_performance",
                    "FAIL",
                    f"Performance test build failed: {build_result.stderr.strip()}"
                )
                return False

            # Run performance test
            run_result = subprocess.run([
                "docker", "run", "--rm", "--platform", "linux/arm64",
                "bev-arm64-performance-test"
            ], capture_output=True, text=True, timeout=60)

            execution_time = time.time() - start_time

            if run_result.returncode == 0:
                output = run_result.stdout.strip()

                # Parse performance results
                performance_score = None
                for line in output.split('\n'):
                    if line.startswith('Performance Score:'):
                        try:
                            performance_score = float(line.split(':')[1].strip().replace('s', ''))
                        except:
                            pass

                if performance_score and performance_score < 10.0:  # 10 seconds threshold
                    self.log_test_result(
                        "arm64_runtime_performance",
                        "PASS",
                        f"ARM64 performance acceptable: {performance_score:.3f}s score",
                        execution_time,
                        {"performance_score": performance_score, "output": output}
                    )
                    result = True
                else:
                    self.log_test_result(
                        "arm64_runtime_performance",
                        "WARN",
                        f"ARM64 performance slow: {performance_score or 'unknown'}s score",
                        execution_time,
                        {"performance_score": performance_score, "output": output}
                    )
                    result = True  # Warn but don't fail
            else:
                self.log_test_result(
                    "arm64_runtime_performance",
                    "FAIL",
                    f"Performance test execution failed: {run_result.stderr.strip()}",
                    execution_time
                )
                result = False

            # Clean up
            subprocess.run(["docker", "rmi", "bev-arm64-performance-test"],
                         capture_output=True)
            return result

        except subprocess.TimeoutExpired:
            self.log_test_result(
                "arm64_runtime_performance",
                "FAIL",
                "Performance test timeout",
                time.time() - start_time
            )
            return False
        except Exception as e:
            self.log_test_result(
                "arm64_runtime_performance",
                "FAIL",
                f"Unexpected error: {str(e)}",
                time.time() - start_time
            )
            return False
        finally:
            # Clean up test files
            for file in [test_dockerfile, test_dockerfile.parent / "performance_test.py"]:
                if file.exists():
                    file.unlink()

    async def run_all_tests(self) -> Dict:
        """Run all ARM64 compatibility tests."""
        logger.info("Starting comprehensive ARM64 compatibility testing...")

        start_time = time.time()

        # Run all test phases
        test_phases = [
            ("Base Image Availability", self.test_base_image_availability),
            ("Package Installation", self.test_package_installation),
            ("Python Packages", self.test_python_packages),
            ("Service Builds", self.test_service_builds),
            ("Runtime Performance", self.test_runtime_performance)
        ]

        for phase_name, test_func in test_phases:
            logger.info(f"Running test phase: {phase_name}")
            try:
                await test_func()
            except Exception as e:
                logger.error(f"Test phase {phase_name} failed with exception: {str(e)}")
                self.log_test_result(
                    f"phase_{phase_name.lower().replace(' ', '_')}",
                    "FAIL",
                    f"Phase failed with exception: {str(e)}"
                )

        # Calculate final results
        total_time = time.time() - start_time
        self.results["total_execution_time"] = total_time

        success_rate = (self.results["summary"]["passed"] /
                       self.results["summary"]["total"] * 100) if self.results["summary"]["total"] > 0 else 0

        self.results["summary"]["success_rate"] = success_rate

        logger.info(f"ARM64 compatibility testing completed in {total_time:.2f}s")
        logger.info(f"Results: {self.results['summary']['passed']}/{self.results['summary']['total']} passed "
                   f"({success_rate:.1f}% success rate)")

        return self.results

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save test results to JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"/home/starlord/Projects/Bev/validation_results/arm64_compatibility_{timestamp}.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"ARM64 compatibility test results saved to: {output_path}")
        return str(output_path)


async def main():
    """Main entry point for ARM64 compatibility testing."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "/home/starlord/Projects/Bev"

    tester = ARM64CompatibilityTester(project_root)

    try:
        results = await tester.run_all_tests()
        output_file = tester.save_results()

        # Print summary
        print("\n" + "="*60)
        print("ARM64 COMPATIBILITY TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {results['summary']['total']}")
        print(f"Passed: {results['summary']['passed']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Warnings: {results['summary']['warnings']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Execution Time: {results['total_execution_time']:.2f}s")
        print(f"Results saved to: {output_file}")
        print("="*60)

        # Exit with appropriate code
        if results['summary']['failed'] == 0:
            print("✅ All ARM64 compatibility tests passed!")
            sys.exit(0)
        else:
            print("❌ Some ARM64 compatibility tests failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("ARM64 compatibility testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"ARM64 compatibility testing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())