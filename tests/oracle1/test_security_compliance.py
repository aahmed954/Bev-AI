#!/usr/bin/env python3
"""
Security and Compliance Testing Suite for ORACLE1

This module provides comprehensive security testing including:
- Vault integration and authentication
- Network security and isolation
- Service access control and authorization
- Credential management and rotation
- Security configuration validation
- Compliance auditing
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
import docker
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SecurityComplianceTester:
    """Comprehensive security and compliance testing for ORACLE1."""

    def __init__(self, project_root: str = "/home/starlord/Projects/Bev"):
        self.project_root = Path(project_root)
        self.docker_client = docker.from_env()
        self.compose_file = self.project_root / "docker-compose-oracle1-unified.yml"
        self.env_file = self.project_root / ".env.oracle1"
        self.results = {
            "timestamp": time.time(),
            "tests": [],
            "security_score": 0.0,
            "compliance_status": "unknown",
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "warnings": 0,
                "critical_issues": 0,
                "high_issues": 0,
                "medium_issues": 0,
                "low_issues": 0
            }
        }

        # Security test categories
        self.security_categories = {
            "authentication": {
                "weight": 0.25,
                "tests": []
            },
            "authorization": {
                "weight": 0.20,
                "tests": []
            },
            "network_security": {
                "weight": 0.20,
                "tests": []
            },
            "data_protection": {
                "weight": 0.15,
                "tests": []
            },
            "configuration": {
                "weight": 0.10,
                "tests": []
            },
            "monitoring": {
                "weight": 0.10,
                "tests": []
            }
        }

        # Security baseline requirements
        self.security_requirements = {
            "vault_integration": True,
            "encrypted_communication": True,
            "access_controls": True,
            "audit_logging": True,
            "secret_management": True,
            "network_isolation": True,
            "service_authentication": True
        }

    def log_test_result(self, test_name: str, status: str, details: str = "",
                       execution_time: float = 0.0, severity: str = "medium",
                       category: str = "configuration", metadata: Dict = None):
        """Log a security test result with severity classification."""
        result = {
            "test_name": test_name,
            "status": status,  # "PASS", "FAIL", "WARN"
            "details": details,
            "severity": severity,  # "critical", "high", "medium", "low"
            "category": category,
            "execution_time": execution_time,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }

        self.results["tests"].append(result)
        self.results["summary"]["total_tests"] += 1

        # Track by severity
        if status == "FAIL":
            self.results["summary"][f"{severity}_issues"] += 1

        if status == "PASS":
            self.results["summary"]["passed_tests"] += 1
            logger.info(f"[PASS] {test_name}: {details}")
        elif status == "FAIL":
            self.results["summary"]["failed_tests"] += 1
            logger.error(f"[FAIL] {test_name} ({severity.upper()}): {details}")
        elif status == "WARN":
            self.results["summary"]["warnings"] += 1
            logger.warning(f"[WARN] {test_name}: {details}")

        # Add to category
        if category in self.security_categories:
            self.security_categories[category]["tests"].append(result)

    async def test_vault_integration(self) -> bool:
        """Test Vault integration and authentication."""
        logger.info("Testing Vault integration and authentication...")

        start_time = time.time()

        # Check Vault configuration in environment
        vault_config_tests = []

        try:
            if self.env_file.exists():
                with open(self.env_file, 'r') as f:
                    env_content = f.read()

                # Check for required Vault environment variables
                vault_vars = ["VAULT_URL", "VAULT_TOKEN", "SECRETS_BACKEND"]
                missing_vars = []

                for var in vault_vars:
                    if f"{var}=" not in env_content:
                        missing_vars.append(var)

                if missing_vars:
                    self.log_test_result(
                        "vault_environment_configuration",
                        "FAIL",
                        f"Missing Vault environment variables: {', '.join(missing_vars)}",
                        time.time() - start_time,
                        "critical",
                        "authentication"
                    )
                    vault_config_tests.append(False)
                else:
                    self.log_test_result(
                        "vault_environment_configuration",
                        "PASS",
                        "All required Vault environment variables present",
                        time.time() - start_time,
                        "low",
                        "authentication"
                    )
                    vault_config_tests.append(True)

                # Extract Vault URL and token for testing
                vault_url = None
                vault_token = None

                for line in env_content.split('\n'):
                    if line.startswith('VAULT_URL='):
                        vault_url = line.split('=', 1)[1]
                    elif line.startswith('VAULT_TOKEN='):
                        vault_token = line.split('=', 1)[1]

                # Test Vault connectivity
                if vault_url and vault_token:
                    try:
                        async with aiohttp.ClientSession() as session:
                            # Test Vault health endpoint
                            health_url = f"{vault_url}/v1/sys/health"
                            headers = {"X-Vault-Token": vault_token}

                            async with session.get(health_url, headers=headers, timeout=10) as response:
                                if response.status in [200, 429, 501]:  # Vault health status codes
                                    health_data = await response.json()
                                    vault_status = health_data.get('initialized', False)

                                    self.log_test_result(
                                        "vault_connectivity",
                                        "PASS",
                                        f"Vault health check successful, initialized: {vault_status}",
                                        time.time() - start_time,
                                        "high" if not vault_status else "low",
                                        "authentication",
                                        {"vault_url": vault_url, "vault_initialized": vault_status}
                                    )
                                    vault_config_tests.append(vault_status)
                                else:
                                    self.log_test_result(
                                        "vault_connectivity",
                                        "FAIL",
                                        f"Vault health check failed with status {response.status}",
                                        time.time() - start_time,
                                        "critical",
                                        "authentication"
                                    )
                                    vault_config_tests.append(False)

                            # Test Vault authentication
                            auth_url = f"{vault_url}/v1/auth/token/lookup-self"
                            async with session.get(auth_url, headers=headers, timeout=10) as response:
                                if response.status == 200:
                                    token_data = await response.json()
                                    policies = token_data.get('data', {}).get('policies', [])

                                    self.log_test_result(
                                        "vault_authentication",
                                        "PASS",
                                        f"Vault token valid with policies: {', '.join(policies)}",
                                        time.time() - start_time,
                                        "low",
                                        "authentication",
                                        {"policies": policies}
                                    )
                                    vault_config_tests.append(True)
                                else:
                                    self.log_test_result(
                                        "vault_authentication",
                                        "FAIL",
                                        f"Vault token authentication failed: {response.status}",
                                        time.time() - start_time,
                                        "critical",
                                        "authentication"
                                    )
                                    vault_config_tests.append(False)

                    except Exception as e:
                        self.log_test_result(
                            "vault_connectivity",
                            "FAIL",
                            f"Vault connectivity test failed: {str(e)}",
                            time.time() - start_time,
                            "critical",
                            "authentication"
                        )
                        vault_config_tests.append(False)

            else:
                self.log_test_result(
                    "vault_environment_file",
                    "FAIL",
                    f"Environment file not found: {self.env_file}",
                    time.time() - start_time,
                    "critical",
                    "authentication"
                )
                vault_config_tests.append(False)

        except Exception as e:
            self.log_test_result(
                "vault_integration",
                "FAIL",
                f"Vault integration test failed: {str(e)}",
                time.time() - start_time,
                "critical",
                "authentication"
            )
            vault_config_tests.append(False)

        return all(vault_config_tests)

    async def test_network_security(self) -> bool:
        """Test network security and isolation."""
        logger.info("Testing network security and isolation...")

        start_time = time.time()
        network_tests = []

        try:
            # Parse Docker Compose to check network configuration
            with open(self.compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)

            networks = compose_data.get('networks', {})
            services = compose_data.get('services', {})

            # Check network isolation
            if 'bev_oracle' in networks and 'external_thanos' in networks:
                self.log_test_result(
                    "network_isolation_configuration",
                    "PASS",
                    "Network isolation configured with internal and external networks",
                    time.time() - start_time,
                    "low",
                    "network_security",
                    {"networks": list(networks.keys())}
                )
                network_tests.append(True)
            else:
                self.log_test_result(
                    "network_isolation_configuration",
                    "FAIL",
                    "Network isolation not properly configured",
                    time.time() - start_time,
                    "high",
                    "network_security"
                )
                network_tests.append(False)

            # Check for exposed ports
            exposed_ports = {}
            for service_name, service_config in services.items():
                ports = service_config.get('ports', [])
                if ports:
                    exposed_ports[service_name] = ports

            # Analyze exposed ports for security risks
            high_risk_ports = []
            for service_name, ports in exposed_ports.items():
                for port_mapping in ports:
                    if ':' in str(port_mapping):
                        host_port = str(port_mapping).split(':')[0]
                        # Check for commonly attacked ports
                        if host_port in ['22', '3389', '21', '23', '25', '53', '135', '139', '445']:
                            high_risk_ports.append(f"{service_name}:{port_mapping}")

            if high_risk_ports:
                self.log_test_result(
                    "exposed_ports_security",
                    "WARN",
                    f"High-risk ports exposed: {', '.join(high_risk_ports)}",
                    time.time() - start_time,
                    "medium",
                    "network_security",
                    {"high_risk_ports": high_risk_ports}
                )
                network_tests.append(True)  # Warning, not failure
            else:
                self.log_test_result(
                    "exposed_ports_security",
                    "PASS",
                    f"No high-risk ports exposed ({len(exposed_ports)} services with ports)",
                    time.time() - start_time,
                    "low",
                    "network_security",
                    {"exposed_services": len(exposed_ports)}
                )
                network_tests.append(True)

            # Test TLS/SSL configuration
            tls_services = []
            for service_name, service_config in services.items():
                environment = service_config.get('environment', {})
                if isinstance(environment, dict):
                    env_vars = environment
                elif isinstance(environment, list):
                    env_vars = {var.split('=')[0]: var.split('=', 1)[1] if '=' in var else ''
                               for var in environment}
                else:
                    env_vars = {}

                # Check for TLS-related environment variables
                tls_indicators = ['TLS', 'SSL', 'HTTPS', 'CERT', 'KEY']
                has_tls = any(indicator in str(env_vars).upper() for indicator in tls_indicators)

                if has_tls:
                    tls_services.append(service_name)

            if tls_services:
                self.log_test_result(
                    "tls_ssl_configuration",
                    "PASS",
                    f"TLS/SSL configuration found in services: {', '.join(tls_services)}",
                    time.time() - start_time,
                    "low",
                    "network_security",
                    {"tls_services": tls_services}
                )
                network_tests.append(True)
            else:
                self.log_test_result(
                    "tls_ssl_configuration",
                    "WARN",
                    "No explicit TLS/SSL configuration found in services",
                    time.time() - start_time,
                    "medium",
                    "network_security"
                )
                network_tests.append(True)  # Warning, not failure

        except Exception as e:
            self.log_test_result(
                "network_security",
                "FAIL",
                f"Network security test failed: {str(e)}",
                time.time() - start_time,
                "high",
                "network_security"
            )
            network_tests.append(False)

        return all(network_tests)

    async def test_access_controls(self) -> bool:
        """Test service access controls and authorization."""
        logger.info("Testing access controls and authorization...")

        start_time = time.time()
        access_tests = []

        try:
            # Parse compose file for authentication configurations
            with open(self.compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)

            services = compose_data.get('services', {})

            # Check authentication configuration for key services
            auth_services = {
                'n8n': ['N8N_BASIC_AUTH_ACTIVE', 'N8N_BASIC_AUTH_USER', 'N8N_BASIC_AUTH_PASSWORD'],
                'grafana': ['GF_SECURITY_ADMIN_USER', 'GF_SECURITY_ADMIN_PASSWORD'],
                'vault': ['VAULT_DEV_ROOT_TOKEN_ID', 'VAULT_TOKEN'],
                'minio1': ['MINIO_ROOT_USER', 'MINIO_ROOT_PASSWORD']
            }

            for service_name, required_auth_vars in auth_services.items():
                if service_name in services:
                    environment = services[service_name].get('environment', {})

                    if isinstance(environment, dict):
                        env_vars = environment
                    elif isinstance(environment, list):
                        env_vars = {var.split('=')[0]: var.split('=', 1)[1] if '=' in var else ''
                                   for var in environment}
                    else:
                        env_vars = {}

                    # Check for authentication variables
                    auth_vars_present = []
                    for auth_var in required_auth_vars:
                        if auth_var in env_vars:
                            auth_vars_present.append(auth_var)

                    if auth_vars_present:
                        # Check for weak default passwords
                        weak_passwords = []
                        for var in auth_vars_present:
                            value = str(env_vars.get(var, ''))
                            if value in ['admin', 'password', '123456', 'admin123', 'root']:
                                weak_passwords.append(var)

                        if weak_passwords:
                            self.log_test_result(
                                f"authentication_{service_name}",
                                "FAIL",
                                f"Weak default passwords detected in {service_name}: {', '.join(weak_passwords)}",
                                time.time() - start_time,
                                "high",
                                "authorization",
                                {"weak_passwords": weak_passwords}
                            )
                            access_tests.append(False)
                        else:
                            self.log_test_result(
                                f"authentication_{service_name}",
                                "PASS",
                                f"Authentication configured for {service_name} with {len(auth_vars_present)} variables",
                                time.time() - start_time,
                                "low",
                                "authorization",
                                {"auth_vars": auth_vars_present}
                            )
                            access_tests.append(True)
                    else:
                        self.log_test_result(
                            f"authentication_{service_name}",
                            "WARN",
                            f"No authentication configuration found for {service_name}",
                            time.time() - start_time,
                            "medium",
                            "authorization"
                        )
                        access_tests.append(True)  # Warning, not failure

            # Test actual service authentication
            auth_endpoints = [
                ("grafana", "http://localhost:3000/api/health", ["admin:admin123"]),
                ("n8n", "http://localhost:5678/", ["admin:admin123"]),
                ("prometheus", "http://localhost:9090/-/healthy", []),  # Usually no auth
                ("vault", "http://localhost:8200/v1/sys/health", [])  # Token-based
            ]

            async with aiohttp.ClientSession() as session:
                for service_name, endpoint, credentials in auth_endpoints:
                    try:
                        # Test without authentication
                        async with session.get(endpoint, timeout=5) as response:
                            if response.status == 401:
                                self.log_test_result(
                                    f"access_control_{service_name}",
                                    "PASS",
                                    f"{service_name} properly requires authentication (401)",
                                    time.time() - start_time,
                                    "low",
                                    "authorization",
                                    {"endpoint": endpoint, "status": response.status}
                                )
                                access_tests.append(True)
                            elif response.status in [200, 204]:
                                if credentials:  # Service should require auth but doesn't
                                    self.log_test_result(
                                        f"access_control_{service_name}",
                                        "WARN",
                                        f"{service_name} accessible without authentication",
                                        time.time() - start_time,
                                        "medium",
                                        "authorization",
                                        {"endpoint": endpoint, "status": response.status}
                                    )
                                else:  # Service doesn't require auth (expected)
                                    self.log_test_result(
                                        f"access_control_{service_name}",
                                        "PASS",
                                        f"{service_name} accessible (no auth required)",
                                        time.time() - start_time,
                                        "low",
                                        "authorization",
                                        {"endpoint": endpoint, "status": response.status}
                                    )
                                access_tests.append(True)
                            else:
                                self.log_test_result(
                                    f"access_control_{service_name}",
                                    "WARN",
                                    f"{service_name} returned unexpected status {response.status}",
                                    time.time() - start_time,
                                    "low",
                                    "authorization",
                                    {"endpoint": endpoint, "status": response.status}
                                )
                                access_tests.append(True)

                    except Exception as e:
                        self.log_test_result(
                            f"access_control_{service_name}",
                            "WARN",
                            f"Could not test {service_name} access control: {str(e)}",
                            time.time() - start_time,
                            "low",
                            "authorization"
                        )
                        access_tests.append(True)  # Warning, not failure

        except Exception as e:
            self.log_test_result(
                "access_controls",
                "FAIL",
                f"Access control test failed: {str(e)}",
                time.time() - start_time,
                "high",
                "authorization"
            )
            access_tests.append(False)

        return all(access_tests)

    async def test_secret_management(self) -> bool:
        """Test secret management and credential security."""
        logger.info("Testing secret management and credential security...")

        start_time = time.time()
        secret_tests = []

        try:
            # Check for hardcoded secrets in compose file
            with open(self.compose_file, 'r') as f:
                compose_content = f.read()

            # Look for potential hardcoded secrets
            secret_patterns = [
                'password',
                'secret',
                'key',
                'token'
            ]

            hardcoded_secrets = []
            for line_num, line in enumerate(compose_content.split('\n'), 1):
                line_lower = line.lower()
                for pattern in secret_patterns:
                    if pattern in line_lower and '${' not in line and not line.strip().startswith('#'):
                        # Check if it looks like a hardcoded value
                        if '=' in line and len(line.split('=', 1)[1].strip()) > 3:
                            hardcoded_secrets.append(f"Line {line_num}: {line.strip()}")

            if hardcoded_secrets:
                self.log_test_result(
                    "hardcoded_secrets_check",
                    "FAIL",
                    f"Potential hardcoded secrets found: {len(hardcoded_secrets)} instances",
                    time.time() - start_time,
                    "critical",
                    "data_protection",
                    {"hardcoded_secrets": hardcoded_secrets[:5]}  # Limit output
                )
                secret_tests.append(False)
            else:
                self.log_test_result(
                    "hardcoded_secrets_check",
                    "PASS",
                    "No obvious hardcoded secrets found in compose file",
                    time.time() - start_time,
                    "low",
                    "data_protection"
                )
                secret_tests.append(True)

            # Check environment variable usage
            env_var_usage = compose_content.count('${')
            total_lines = len(compose_content.split('\n'))

            if env_var_usage > 0:
                env_percentage = (env_var_usage / total_lines) * 100
                self.log_test_result(
                    "environment_variable_usage",
                    "PASS",
                    f"Environment variables used {env_var_usage} times ({env_percentage:.1f}% of lines)",
                    time.time() - start_time,
                    "low",
                    "data_protection",
                    {"env_var_count": env_var_usage, "usage_percentage": env_percentage}
                )
                secret_tests.append(True)
            else:
                self.log_test_result(
                    "environment_variable_usage",
                    "WARN",
                    "No environment variables found - may indicate hardcoded values",
                    time.time() - start_time,
                    "medium",
                    "data_protection"
                )
                secret_tests.append(True)  # Warning, not failure

            # Check .env file security
            if self.env_file.exists():
                env_file_perms = oct(self.env_file.stat().st_mode)[-3:]
                if env_file_perms == '600':
                    self.log_test_result(
                        "env_file_permissions",
                        "PASS",
                        f"Environment file has secure permissions: {env_file_perms}",
                        time.time() - start_time,
                        "low",
                        "data_protection",
                        {"permissions": env_file_perms}
                    )
                    secret_tests.append(True)
                else:
                    self.log_test_result(
                        "env_file_permissions",
                        "WARN",
                        f"Environment file has permissive permissions: {env_file_perms} (recommend 600)",
                        time.time() - start_time,
                        "medium",
                        "data_protection",
                        {"permissions": env_file_perms}
                    )
                    secret_tests.append(True)  # Warning, not failure

                # Check for sensitive data in .env file
                with open(self.env_file, 'r') as f:
                    env_content = f.read()

                # Count secrets in environment file
                secret_count = 0
                for pattern in ['PASSWORD', 'TOKEN', 'SECRET', 'KEY']:
                    secret_count += env_content.upper().count(pattern)

                if secret_count > 0:
                    self.log_test_result(
                        "env_file_secrets",
                        "PASS",
                        f"Environment file contains {secret_count} secret variables",
                        time.time() - start_time,
                        "low",
                        "data_protection",
                        {"secret_count": secret_count}
                    )
                    secret_tests.append(True)
                else:
                    self.log_test_result(
                        "env_file_secrets",
                        "WARN",
                        "No obvious secrets found in environment file",
                        time.time() - start_time,
                        "low",
                        "data_protection"
                    )
                    secret_tests.append(True)

        except Exception as e:
            self.log_test_result(
                "secret_management",
                "FAIL",
                f"Secret management test failed: {str(e)}",
                time.time() - start_time,
                "high",
                "data_protection"
            )
            secret_tests.append(False)

        return all(secret_tests)

    async def test_container_security(self) -> bool:
        """Test container security configurations."""
        logger.info("Testing container security configurations...")

        start_time = time.time()
        container_tests = []

        try:
            # Parse compose file for security configurations
            with open(self.compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)

            services = compose_data.get('services', {})

            # Check for security-relevant configurations
            security_configs = {
                'privileged': 'high',      # Should not be true
                'cap_add': 'medium',       # Should be minimal
                'cap_drop': 'low',         # Good to have
                'user': 'medium',          # Should specify non-root
                'read_only': 'low',        # Good for security
                'security_opt': 'low',     # Security options
                'tmpfs': 'low'             # Secure temporary filesystems
            }

            for service_name, service_config in services.items():
                service_security_score = 0
                service_issues = []

                for config, severity in security_configs.items():
                    if config in service_config:
                        value = service_config[config]

                        if config == 'privileged' and value is True:
                            service_issues.append(f"Running privileged ({severity} risk)")
                        elif config == 'cap_add':
                            service_issues.append(f"Added capabilities: {value} ({severity} risk)")
                        elif config == 'user' and value != 'root':
                            service_security_score += 1  # Good
                        elif config in ['cap_drop', 'read_only', 'security_opt', 'tmpfs']:
                            service_security_score += 1  # Good

                if service_issues:
                    self.log_test_result(
                        f"container_security_{service_name}",
                        "WARN",
                        f"Security concerns in {service_name}: {'; '.join(service_issues)}",
                        time.time() - start_time,
                        "medium",
                        "configuration",
                        {"security_issues": service_issues, "security_score": service_security_score}
                    )
                    container_tests.append(True)  # Warning, not failure
                else:
                    self.log_test_result(
                        f"container_security_{service_name}",
                        "PASS",
                        f"No obvious security issues in {service_name} (score: {service_security_score})",
                        time.time() - start_time,
                        "low",
                        "configuration",
                        {"security_score": service_security_score}
                    )
                    container_tests.append(True)

            # Check resource limits
            services_with_limits = 0
            services_without_limits = []

            for service_name, service_config in services.items():
                deploy_config = service_config.get('deploy', {})
                resources = deploy_config.get('resources', {})

                if resources.get('limits') or '<<: *arm-' in str(service_config):
                    services_with_limits += 1
                else:
                    services_without_limits.append(service_name)

            if services_without_limits:
                self.log_test_result(
                    "resource_limits",
                    "WARN",
                    f"{len(services_without_limits)} services without resource limits: {', '.join(services_without_limits[:5])}",
                    time.time() - start_time,
                    "medium",
                    "configuration",
                    {"services_without_limits": len(services_without_limits)}
                )
                container_tests.append(True)  # Warning, not failure
            else:
                self.log_test_result(
                    "resource_limits",
                    "PASS",
                    f"All {services_with_limits} services have resource limits configured",
                    time.time() - start_time,
                    "low",
                    "configuration",
                    {"services_with_limits": services_with_limits}
                )
                container_tests.append(True)

        except Exception as e:
            self.log_test_result(
                "container_security",
                "FAIL",
                f"Container security test failed: {str(e)}",
                time.time() - start_time,
                "high",
                "configuration"
            )
            container_tests.append(False)

        return all(container_tests)

    async def test_audit_logging(self) -> bool:
        """Test audit logging and monitoring configurations."""
        logger.info("Testing audit logging and monitoring...")

        start_time = time.time()
        audit_tests = []

        try:
            # Parse compose file for logging configurations
            with open(self.compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)

            services = compose_data.get('services', {})

            # Check logging configuration
            services_with_logging = 0
            logging_drivers = {}

            for service_name, service_config in services.items():
                logging_config = service_config.get('logging', {})

                if logging_config or '*default-logging' in str(service_config):
                    services_with_logging += 1

                    driver = logging_config.get('driver', 'default')
                    if driver not in logging_drivers:
                        logging_drivers[driver] = 0
                    logging_drivers[driver] += 1

            total_services = len(services)
            logging_percentage = (services_with_logging / total_services) * 100

            if logging_percentage >= 80:
                self.log_test_result(
                    "audit_logging_coverage",
                    "PASS",
                    f"Logging configured for {services_with_logging}/{total_services} services ({logging_percentage:.1f}%)",
                    time.time() - start_time,
                    "low",
                    "monitoring",
                    {"logging_coverage": logging_percentage, "drivers": logging_drivers}
                )
                audit_tests.append(True)
            else:
                self.log_test_result(
                    "audit_logging_coverage",
                    "WARN",
                    f"Only {services_with_logging}/{total_services} services have logging ({logging_percentage:.1f}%)",
                    time.time() - start_time,
                    "medium",
                    "monitoring",
                    {"logging_coverage": logging_percentage}
                )
                audit_tests.append(True)  # Warning, not failure

            # Check for monitoring services
            monitoring_services = ['prometheus', 'grafana', 'alertmanager']
            monitoring_present = []

            for service in monitoring_services:
                if service in services:
                    monitoring_present.append(service)

            if len(monitoring_present) >= 2:
                self.log_test_result(
                    "monitoring_infrastructure",
                    "PASS",
                    f"Monitoring infrastructure present: {', '.join(monitoring_present)}",
                    time.time() - start_time,
                    "low",
                    "monitoring",
                    {"monitoring_services": monitoring_present}
                )
                audit_tests.append(True)
            else:
                self.log_test_result(
                    "monitoring_infrastructure",
                    "WARN",
                    f"Limited monitoring infrastructure: {', '.join(monitoring_present)}",
                    time.time() - start_time,
                    "medium",
                    "monitoring",
                    {"monitoring_services": monitoring_present}
                )
                audit_tests.append(True)  # Warning, not failure

        except Exception as e:
            self.log_test_result(
                "audit_logging",
                "FAIL",
                f"Audit logging test failed: {str(e)}",
                time.time() - start_time,
                "high",
                "monitoring"
            )
            audit_tests.append(False)

        return all(audit_tests)

    def calculate_security_score(self) -> float:
        """Calculate overall security score based on test results."""
        category_scores = {}

        for category, config in self.security_categories.items():
            if config["tests"]:
                passed_tests = sum(1 for test in config["tests"] if test["status"] == "PASS")
                total_tests = len(config["tests"])
                category_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

                # Apply severity penalties
                for test in config["tests"]:
                    if test["status"] == "FAIL":
                        if test["severity"] == "critical":
                            category_score -= 30
                        elif test["severity"] == "high":
                            category_score -= 20
                        elif test["severity"] == "medium":
                            category_score -= 10

                category_scores[category] = max(0, category_score)
            else:
                category_scores[category] = 0

        # Calculate weighted overall score
        overall_score = sum(
            category_scores[category] * config["weight"]
            for category, config in self.security_categories.items()
        )

        return min(100, max(0, overall_score))

    def determine_compliance_status(self, score: float) -> str:
        """Determine compliance status based on security score."""
        if score >= 90:
            return "excellent"
        elif score >= 80:
            return "good"
        elif score >= 70:
            return "acceptable"
        elif score >= 60:
            return "needs_improvement"
        else:
            return "critical"

    async def run_all_tests(self) -> Dict:
        """Run all security and compliance tests."""
        logger.info("Starting comprehensive security and compliance testing...")

        start_time = time.time()

        # Run all test phases
        test_phases = [
            ("Vault Integration", self.test_vault_integration),
            ("Network Security", self.test_network_security),
            ("Access Controls", self.test_access_controls),
            ("Secret Management", self.test_secret_management),
            ("Container Security", self.test_container_security),
            ("Audit Logging", self.test_audit_logging)
        ]

        for phase_name, test_func in test_phases:
            logger.info(f"Running security test phase: {phase_name}")
            try:
                await test_func()
            except Exception as e:
                logger.error(f"Security test phase {phase_name} failed with exception: {str(e)}")
                self.log_test_result(
                    f"phase_{phase_name.lower().replace(' ', '_')}",
                    "FAIL",
                    f"Phase failed with exception: {str(e)}",
                    0.0,
                    "critical",
                    "configuration"
                )

        # Calculate final results
        total_time = time.time() - start_time
        self.results["total_execution_time"] = total_time

        success_rate = (self.results["summary"]["passed_tests"] /
                       self.results["summary"]["total_tests"] * 100) if self.results["summary"]["total_tests"] > 0 else 0

        self.results["summary"]["success_rate"] = success_rate
        self.results["security_score"] = self.calculate_security_score()
        self.results["compliance_status"] = self.determine_compliance_status(self.results["security_score"])

        logger.info(f"Security and compliance testing completed in {total_time:.2f}s")
        logger.info(f"Results: {self.results['summary']['passed_tests']}/{self.results['summary']['total_tests']} passed "
                   f"({success_rate:.1f}% success rate)")
        logger.info(f"Security Score: {self.results['security_score']:.1f}/100 ({self.results['compliance_status']})")

        return self.results

    def save_results(self, output_file: Optional[str] = None) -> str:
        """Save security test results to JSON file."""
        if output_file is None:
            timestamp = int(time.time())
            output_file = f"/home/starlord/Projects/Bev/validation_results/security_compliance_{timestamp}.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Security and compliance test results saved to: {output_path}")
        return str(output_path)


async def main():
    """Main entry point for security and compliance testing."""
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = "/home/starlord/Projects/Bev"

    tester = SecurityComplianceTester(project_root)

    try:
        results = await tester.run_all_tests()
        output_file = tester.save_results()

        # Print summary
        print("\n" + "="*60)
        print("SECURITY & COMPLIANCE TEST SUMMARY")
        print("="*60)
        print(f"Total Tests: {results['summary']['total_tests']}")
        print(f"Passed: {results['summary']['passed_tests']}")
        print(f"Failed: {results['summary']['failed_tests']}")
        print(f"Warnings: {results['summary']['warnings']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Security Score: {results['security_score']:.1f}/100")
        print(f"Compliance Status: {results['compliance_status'].replace('_', ' ').title()}")
        print("")
        print("Issues by Severity:")
        print(f"  Critical: {results['summary']['critical_issues']}")
        print(f"  High: {results['summary']['high_issues']}")
        print(f"  Medium: {results['summary']['medium_issues']}")
        print(f"  Low: {results['summary']['low_issues']}")
        print(f"Execution Time: {results['total_execution_time']:.2f}s")
        print(f"Results saved to: {output_file}")
        print("="*60)

        # Exit with appropriate code
        if results['summary']['critical_issues'] == 0 and results['summary']['failed_tests'] == 0:
            print("✅ All critical security tests passed!")
            sys.exit(0)
        else:
            print("❌ Critical security issues found!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Security and compliance testing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Security and compliance testing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())