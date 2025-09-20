#!/usr/bin/env python3

"""
BEV OSINT Framework - Monitoring and Alerting Validation System
Comprehensive validation of monitoring, metrics, logs, and alerting
"""

import sys
import os
import json
import time
import asyncio
import aiohttp
import requests
import docker
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

@dataclass
class MonitoringEndpoint:
    """Monitoring endpoint definition"""
    name: str
    url: str
    service_type: str  # prometheus, grafana, elasticsearch, jaeger
    health_endpoint: str = "/health"
    timeout: int = 30
    auth_required: bool = False
    auth_user: Optional[str] = None
    auth_pass: Optional[str] = None

@dataclass
class ValidationResult:
    """Validation result data structure"""
    component: str
    test_name: str
    status: str  # PASS, FAIL, WARN, SKIP
    message: str
    duration: float = 0.0
    details: Optional[Dict] = None
    severity: str = "INFO"

class MonitoringValidator:
    """Comprehensive monitoring and alerting validation system"""

    def __init__(self, phases: List[str] = None):
        self.phases = phases or ["7", "8", "9"]
        self.project_root = project_root
        self.results: List[ValidationResult] = []
        self.docker_client = None
        self.failed_tests = 0
        self.warnings = 0
        self.session = None

        # Monitoring endpoints
        self.monitoring_endpoints = self._define_monitoring_endpoints()

        # Setup logging
        self.setup_logging()

        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            self.logger.info("Docker client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Docker client: {e}")

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.project_root / "logs" / "monitoring"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"monitoring_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _define_monitoring_endpoints(self) -> List[MonitoringEndpoint]:
        """Define monitoring service endpoints"""
        return [
            MonitoringEndpoint(
                name="prometheus",
                url="http://localhost:9090",
                service_type="prometheus",
                health_endpoint="/-/healthy"
            ),
            MonitoringEndpoint(
                name="grafana",
                url="http://localhost:3000",
                service_type="grafana",
                health_endpoint="/api/health",
                auth_required=True,
                auth_user="admin",
                auth_pass="admin"
            ),
            MonitoringEndpoint(
                name="elasticsearch",
                url="http://localhost:9200",
                service_type="elasticsearch",
                health_endpoint="/_cluster/health"
            ),
            MonitoringEndpoint(
                name="kibana",
                url="http://localhost:5601",
                service_type="kibana",
                health_endpoint="/api/status"
            ),
            MonitoringEndpoint(
                name="jaeger",
                url="http://localhost:16686",
                service_type="jaeger",
                health_endpoint="/api/health"
            ),
            MonitoringEndpoint(
                name="alertmanager",
                url="http://localhost:9093",
                service_type="alertmanager",
                health_endpoint="/-/healthy"
            )
        ]

    def add_result(self, component: str, test_name: str, status: str, message: str,
                   duration: float = 0.0, details: Dict = None, severity: str = "INFO"):
        """Add a validation result"""
        result = ValidationResult(component, test_name, status, message, duration, details, severity)
        self.results.append(result)

        if status == "FAIL":
            self.failed_tests += 1
        elif status == "WARN":
            self.warnings += 1

        # Log the result
        log_level = {
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }.get(severity, logging.INFO)

        self.logger.log(log_level, f"{component}/{test_name}: {status} - {message} ({duration:.2f}s)")

    async def validate_monitoring_service(self, endpoint: MonitoringEndpoint) -> bool:
        """Validate a monitoring service"""
        component = endpoint.name
        start_time = time.time()

        try:
            # Check if container is running
            container_name = endpoint.name
            try:
                container = self.docker_client.containers.get(container_name)
                if container.status == "running":
                    self.add_result(component, "container_status", "PASS",
                                  f"Container running (ID: {container.id[:12]})")
                else:
                    self.add_result(component, "container_status", "FAIL",
                                  f"Container not running (status: {container.status})", severity="ERROR")
                    return False
            except docker.errors.NotFound:
                self.add_result(component, "container_status", "SKIP",
                              f"Container not found - may be external service")

            # Check service health
            health_url = f"{endpoint.url}{endpoint.health_endpoint}"
            auth = None
            if endpoint.auth_required:
                auth = aiohttp.BasicAuth(endpoint.auth_user, endpoint.auth_pass)

            async with self.session.get(health_url, auth=auth,
                                      timeout=aiohttp.ClientTimeout(total=endpoint.timeout)) as response:
                duration = time.time() - start_time

                if response.status in [200, 201]:
                    self.add_result(component, "health_check", "PASS",
                                  f"Health endpoint responding (HTTP {response.status})", duration)

                    # Service-specific validation
                    await self._validate_service_specific(endpoint, response)
                    return True
                else:
                    self.add_result(component, "health_check", "FAIL",
                                  f"Health endpoint returned HTTP {response.status}", duration, severity="ERROR")
                    return False

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.add_result(component, "health_check", "FAIL",
                          f"Health check timeout after {endpoint.timeout}s", duration, severity="ERROR")
            return False
        except Exception as e:
            duration = time.time() - start_time
            self.add_result(component, "health_check", "FAIL",
                          f"Health check failed: {str(e)}", duration, severity="ERROR")
            return False

    async def _validate_service_specific(self, endpoint: MonitoringEndpoint, response):
        """Perform service-specific validation"""
        if endpoint.service_type == "prometheus":
            await self._validate_prometheus(endpoint)
        elif endpoint.service_type == "grafana":
            await self._validate_grafana(endpoint)
        elif endpoint.service_type == "elasticsearch":
            await self._validate_elasticsearch(endpoint)
        elif endpoint.service_type == "kibana":
            await self._validate_kibana(endpoint)
        elif endpoint.service_type == "jaeger":
            await self._validate_jaeger(endpoint)
        elif endpoint.service_type == "alertmanager":
            await self._validate_alertmanager(endpoint)

    async def _validate_prometheus(self, endpoint: MonitoringEndpoint):
        """Validate Prometheus specific functionality"""
        component = endpoint.name

        try:
            # Check targets
            targets_url = f"{endpoint.url}/api/v1/targets"
            async with self.session.get(targets_url) as response:
                if response.status == 200:
                    data = await response.json()
                    active_targets = data.get("data", {}).get("activeTargets", [])
                    up_targets = [t for t in active_targets if t.get("health") == "up"]

                    self.add_result(component, "targets_check", "PASS",
                                  f"Prometheus targets: {len(up_targets)}/{len(active_targets)} up",
                                  details={"up_targets": len(up_targets), "total_targets": len(active_targets)})

                    # Check for BEV service targets
                    bev_targets = [t for t in active_targets if "bev_" in t.get("labels", {}).get("job", "")]
                    if bev_targets:
                        self.add_result(component, "bev_targets", "PASS",
                                      f"BEV service targets found: {len(bev_targets)}")
                    else:
                        self.add_result(component, "bev_targets", "WARN",
                                      "No BEV service targets found", severity="WARN")

            # Check metrics availability
            metrics_url = f"{endpoint.url}/api/v1/query"
            test_queries = [
                "up",
                "container_memory_usage_bytes",
                "http_requests_total",
                "process_cpu_seconds_total"
            ]

            for query in test_queries:
                params = {"query": query}
                async with self.session.get(metrics_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        result_count = len(data.get("data", {}).get("result", []))
                        if result_count > 0:
                            self.add_result(component, f"metric_{query}", "PASS",
                                          f"Metric available: {query} ({result_count} series)")
                        else:
                            self.add_result(component, f"metric_{query}", "WARN",
                                          f"No data for metric: {query}", severity="WARN")

            # Check rules
            rules_url = f"{endpoint.url}/api/v1/rules"
            async with self.session.get(rules_url) as response:
                if response.status == 200:
                    data = await response.json()
                    groups = data.get("data", {}).get("groups", [])
                    total_rules = sum(len(group.get("rules", [])) for group in groups)

                    self.add_result(component, "rules_check", "PASS",
                                  f"Prometheus rules loaded: {total_rules} rules in {len(groups)} groups",
                                  details={"rule_groups": len(groups), "total_rules": total_rules})

        except Exception as e:
            self.add_result(component, "prometheus_validation", "FAIL",
                          f"Prometheus validation failed: {str(e)}", severity="ERROR")

    async def _validate_grafana(self, endpoint: MonitoringEndpoint):
        """Validate Grafana specific functionality"""
        component = endpoint.name
        auth = aiohttp.BasicAuth(endpoint.auth_user, endpoint.auth_pass)

        try:
            # Check datasources
            datasources_url = f"{endpoint.url}/api/datasources"
            async with self.session.get(datasources_url, auth=auth) as response:
                if response.status == 200:
                    datasources = await response.json()
                    prometheus_sources = [ds for ds in datasources if ds.get("type") == "prometheus"]

                    if prometheus_sources:
                        self.add_result(component, "datasources", "PASS",
                                      f"Grafana datasources: {len(prometheus_sources)} Prometheus, {len(datasources)} total")
                    else:
                        self.add_result(component, "datasources", "WARN",
                                      "No Prometheus datasources found", severity="WARN")

            # Check dashboards
            dashboards_url = f"{endpoint.url}/api/search?type=dash-db"
            async with self.session.get(dashboards_url, auth=auth) as response:
                if response.status == 200:
                    dashboards = await response.json()
                    bev_dashboards = [d for d in dashboards if "bev" in d.get("title", "").lower()]

                    self.add_result(component, "dashboards", "PASS",
                                  f"Grafana dashboards: {len(bev_dashboards)} BEV, {len(dashboards)} total",
                                  details={"bev_dashboards": len(bev_dashboards), "total_dashboards": len(dashboards)})

            # Check alerts
            alerts_url = f"{endpoint.url}/api/alerts"
            async with self.session.get(alerts_url, auth=auth) as response:
                if response.status == 200:
                    alerts = await response.json()
                    self.add_result(component, "alerts", "PASS",
                                  f"Grafana alerts configured: {len(alerts)}",
                                  details={"alert_count": len(alerts)})

        except Exception as e:
            self.add_result(component, "grafana_validation", "FAIL",
                          f"Grafana validation failed: {str(e)}", severity="ERROR")

    async def _validate_elasticsearch(self, endpoint: MonitoringEndpoint):
        """Validate Elasticsearch specific functionality"""
        component = endpoint.name

        try:
            # Check cluster health
            health_url = f"{endpoint.url}/_cluster/health"
            async with self.session.get(health_url) as response:
                if response.status == 200:
                    health = await response.json()
                    status = health.get("status")
                    nodes = health.get("number_of_nodes")
                    indices = health.get("active_primary_shards")

                    if status == "green":
                        self.add_result(component, "cluster_health", "PASS",
                                      f"Elasticsearch cluster healthy: {nodes} nodes, {indices} shards")
                    elif status == "yellow":
                        self.add_result(component, "cluster_health", "WARN",
                                      f"Elasticsearch cluster yellow: {nodes} nodes, {indices} shards", severity="WARN")
                    else:
                        self.add_result(component, "cluster_health", "FAIL",
                                      f"Elasticsearch cluster red: {nodes} nodes, {indices} shards", severity="ERROR")

            # Check indices
            indices_url = f"{endpoint.url}/_cat/indices?format=json"
            async with self.session.get(indices_url) as response:
                if response.status == 200:
                    indices = await response.json()
                    bev_indices = [idx for idx in indices if "bev" in idx.get("index", "")]
                    log_indices = [idx for idx in indices if "log" in idx.get("index", "")]

                    self.add_result(component, "indices_check", "PASS",
                                  f"Elasticsearch indices: {len(bev_indices)} BEV, {len(log_indices)} logs, {len(indices)} total",
                                  details={"bev_indices": len(bev_indices), "log_indices": len(log_indices)})

            # Check if logs are being indexed
            search_url = f"{endpoint.url}/_search"
            query = {
                "query": {"match_all": {}},
                "size": 1,
                "sort": [{"@timestamp": {"order": "desc"}}]
            }
            async with self.session.post(search_url, json=query) as response:
                if response.status == 200:
                    result = await response.json()
                    total_docs = result.get("hits", {}).get("total", {}).get("value", 0)
                    if total_docs > 0:
                        self.add_result(component, "data_ingestion", "PASS",
                                      f"Elasticsearch ingesting data: {total_docs} documents")
                    else:
                        self.add_result(component, "data_ingestion", "WARN",
                                      "No documents found in Elasticsearch", severity="WARN")

        except Exception as e:
            self.add_result(component, "elasticsearch_validation", "FAIL",
                          f"Elasticsearch validation failed: {str(e)}", severity="ERROR")

    async def _validate_kibana(self, endpoint: MonitoringEndpoint):
        """Validate Kibana specific functionality"""
        component = endpoint.name

        try:
            # Check status
            status_url = f"{endpoint.url}/api/status"
            async with self.session.get(status_url) as response:
                if response.status == 200:
                    status = await response.json()
                    overall_status = status.get("status", {}).get("overall", {}).get("state")

                    if overall_status == "green":
                        self.add_result(component, "kibana_status", "PASS", "Kibana status green")
                    else:
                        self.add_result(component, "kibana_status", "WARN",
                                      f"Kibana status: {overall_status}", severity="WARN")

            # Check saved objects (dashboards, visualizations)
            saved_objects_url = f"{endpoint.url}/api/saved_objects/_find"
            params = {"type": "dashboard", "per_page": 100}
            async with self.session.get(saved_objects_url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    dashboards = result.get("saved_objects", [])
                    bev_dashboards = [d for d in dashboards if "bev" in d.get("attributes", {}).get("title", "").lower()]

                    self.add_result(component, "dashboards", "PASS",
                                  f"Kibana dashboards: {len(bev_dashboards)} BEV, {len(dashboards)} total")

        except Exception as e:
            self.add_result(component, "kibana_validation", "FAIL",
                          f"Kibana validation failed: {str(e)}", severity="ERROR")

    async def _validate_jaeger(self, endpoint: MonitoringEndpoint):
        """Validate Jaeger specific functionality"""
        component = endpoint.name

        try:
            # Check services
            services_url = f"{endpoint.url}/api/services"
            async with self.session.get(services_url) as response:
                if response.status == 200:
                    result = await response.json()
                    services = result.get("data", [])
                    bev_services = [s for s in services if "bev" in s.lower()]

                    if bev_services:
                        self.add_result(component, "traced_services", "PASS",
                                      f"Jaeger tracing services: {len(bev_services)} BEV services")
                    else:
                        self.add_result(component, "traced_services", "WARN",
                                      "No BEV services found in Jaeger", severity="WARN")

            # Check recent traces
            traces_url = f"{endpoint.url}/api/traces"
            params = {"limit": 20, "lookback": "1h"}
            async with self.session.get(traces_url, params=params) as response:
                if response.status == 200:
                    result = await response.json()
                    traces = result.get("data", [])

                    if traces:
                        self.add_result(component, "recent_traces", "PASS",
                                      f"Jaeger recent traces: {len(traces)} traces in last hour")
                    else:
                        self.add_result(component, "recent_traces", "WARN",
                                      "No recent traces found", severity="WARN")

        except Exception as e:
            self.add_result(component, "jaeger_validation", "FAIL",
                          f"Jaeger validation failed: {str(e)}", severity="ERROR")

    async def _validate_alertmanager(self, endpoint: MonitoringEndpoint):
        """Validate Alertmanager specific functionality"""
        component = endpoint.name

        try:
            # Check configuration
            config_url = f"{endpoint.url}/api/v1/status"
            async with self.session.get(config_url) as response:
                if response.status == 200:
                    status = await response.json()
                    config_hash = status.get("data", {}).get("configHash")

                    if config_hash:
                        self.add_result(component, "configuration", "PASS",
                                      f"Alertmanager configuration loaded (hash: {config_hash[:8]})")
                    else:
                        self.add_result(component, "configuration", "WARN",
                                      "Alertmanager configuration hash not found", severity="WARN")

            # Check alerts
            alerts_url = f"{endpoint.url}/api/v1/alerts"
            async with self.session.get(alerts_url) as response:
                if response.status == 200:
                    alerts = await response.json()
                    active_alerts = alerts.get("data", [])
                    firing_alerts = [a for a in active_alerts if a.get("status", {}).get("state") == "active"]

                    self.add_result(component, "alerts_status", "PASS",
                                  f"Alertmanager alerts: {len(firing_alerts)} firing, {len(active_alerts)} total",
                                  details={"firing_alerts": len(firing_alerts), "total_alerts": len(active_alerts)})

            # Check receivers
            receivers_url = f"{endpoint.url}/api/v1/receivers"
            async with self.session.get(receivers_url) as response:
                if response.status == 200:
                    receivers = await response.json()
                    receiver_count = len(receivers.get("data", []))

                    if receiver_count > 0:
                        self.add_result(component, "receivers", "PASS",
                                      f"Alertmanager receivers configured: {receiver_count}")
                    else:
                        self.add_result(component, "receivers", "WARN",
                                      "No alert receivers configured", severity="WARN")

        except Exception as e:
            self.add_result(component, "alertmanager_validation", "FAIL",
                          f"Alertmanager validation failed: {str(e)}", severity="ERROR")

    def validate_log_aggregation(self) -> bool:
        """Validate log aggregation and centralized logging"""
        component = "log_aggregation"
        self.logger.info("Validating log aggregation...")

        try:
            # Check if log volumes exist
            log_volume_found = False
            try:
                volume = self.docker_client.volumes.get("logs")
                self.add_result(component, "log_volume", "PASS", "Centralized log volume exists")
                log_volume_found = True
            except docker.errors.NotFound:
                self.add_result(component, "log_volume", "WARN",
                              "Centralized log volume not found", severity="WARN")

            # Check log files for each phase service
            phase_services = {
                "7": ["dm-crawler", "crypto-intel", "reputation-analyzer", "economics-processor"],
                "8": ["tactical-intel", "defense-automation", "opsec-monitor", "intel-fusion"],
                "9": ["autonomous-coordinator", "adaptive-learning", "resource-manager", "knowledge-evolution"]
            }

            total_services_logging = 0
            for phase in self.phases:
                if phase in phase_services:
                    for service in phase_services[phase]:
                        container_name = f"bev_{service}"
                        try:
                            container = self.docker_client.containers.get(container_name)
                            # Check if container is producing logs
                            logs = container.logs(tail=10, timestamps=True)
                            if logs:
                                total_services_logging += 1
                                self.add_result(component, f"service_logs_{service}", "PASS",
                                              f"Service {service} producing logs")
                            else:
                                self.add_result(component, f"service_logs_{service}", "WARN",
                                              f"Service {service} not producing logs", severity="WARN")
                        except docker.errors.NotFound:
                            self.add_result(component, f"service_logs_{service}", "SKIP",
                                          f"Service {service} container not found")

            self.add_result(component, "overall_logging", "PASS",
                          f"Log aggregation: {total_services_logging} services logging",
                          details={"logging_services": total_services_logging})

            return log_volume_found and total_services_logging > 0

        except Exception as e:
            self.add_result(component, "log_aggregation_check", "FAIL",
                          f"Log aggregation validation failed: {str(e)}", severity="ERROR")
            return False

    def validate_metrics_collection(self) -> bool:
        """Validate metrics collection and exposition"""
        component = "metrics_collection"
        self.logger.info("Validating metrics collection...")

        try:
            # Check if services expose metrics endpoints
            phase_services = {
                "7": [("dm-crawler", 8001), ("crypto-intel", 8002), ("reputation-analyzer", 8003), ("economics-processor", 8004)],
                "8": [("tactical-intel", 8005), ("defense-automation", 8006), ("opsec-monitor", 8007), ("intel-fusion", 8008)],
                "9": [("autonomous-coordinator", 8009), ("adaptive-learning", 8010), ("resource-manager", 8011), ("knowledge-evolution", 8012)]
            }

            services_with_metrics = 0
            for phase in self.phases:
                if phase in phase_services:
                    for service, port in phase_services[phase]:
                        try:
                            # Check for metrics endpoint
                            metrics_url = f"http://localhost:{port}/metrics"
                            response = requests.get(metrics_url, timeout=10)

                            if response.status_code == 200 and "# HELP" in response.text:
                                services_with_metrics += 1
                                self.add_result(component, f"metrics_{service}", "PASS",
                                              f"Service {service} exposing Prometheus metrics")
                            else:
                                self.add_result(component, f"metrics_{service}", "WARN",
                                              f"Service {service} metrics endpoint not found", severity="WARN")

                        except Exception as e:
                            self.add_result(component, f"metrics_{service}", "WARN",
                                          f"Service {service} metrics check failed: {str(e)}", severity="WARN")

            self.add_result(component, "overall_metrics", "PASS",
                          f"Metrics collection: {services_with_metrics} services exposing metrics",
                          details={"metrics_services": services_with_metrics})

            return services_with_metrics > 0

        except Exception as e:
            self.add_result(component, "metrics_collection_check", "FAIL",
                          f"Metrics collection validation failed: {str(e)}", severity="ERROR")
            return False

    def validate_alerting_rules(self) -> bool:
        """Validate alerting rules configuration"""
        component = "alerting_rules"
        self.logger.info("Validating alerting rules...")

        try:
            # Check if Prometheus has alerting rules
            prometheus_url = "http://localhost:9090"
            rules_url = f"{prometheus_url}/api/v1/rules"

            response = requests.get(rules_url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                groups = data.get("data", {}).get("groups", [])

                alert_rules = []
                for group in groups:
                    for rule in group.get("rules", []):
                        if rule.get("type") == "alerting":
                            alert_rules.append(rule)

                if alert_rules:
                    self.add_result(component, "prometheus_rules", "PASS",
                                  f"Prometheus alerting rules: {len(alert_rules)} rules configured")

                    # Check for BEV-specific rules
                    bev_rules = [r for r in alert_rules if "bev" in r.get("name", "").lower()]
                    if bev_rules:
                        self.add_result(component, "bev_specific_rules", "PASS",
                                      f"BEV-specific alerting rules: {len(bev_rules)} rules")
                    else:
                        self.add_result(component, "bev_specific_rules", "WARN",
                                      "No BEV-specific alerting rules found", severity="WARN")

                else:
                    self.add_result(component, "prometheus_rules", "WARN",
                                  "No alerting rules configured in Prometheus", severity="WARN")

                return len(alert_rules) > 0

            else:
                self.add_result(component, "prometheus_rules", "FAIL",
                              f"Failed to check Prometheus rules: HTTP {response.status_code}", severity="ERROR")
                return False

        except Exception as e:
            self.add_result(component, "alerting_rules_check", "FAIL",
                          f"Alerting rules validation failed: {str(e)}", severity="ERROR")
            return False

    async def validate_dashboard_availability(self) -> bool:
        """Validate dashboard availability and functionality"""
        component = "dashboards"
        self.logger.info("Validating dashboard availability...")

        try:
            # Check Grafana dashboards
            grafana_url = "http://localhost:3000"
            auth = aiohttp.BasicAuth("admin", "admin")

            # Get list of dashboards
            dashboards_url = f"{grafana_url}/api/search?type=dash-db"
            async with self.session.get(dashboards_url, auth=auth) as response:
                if response.status == 200:
                    dashboards = await response.json()

                    if dashboards:
                        self.add_result(component, "grafana_dashboards", "PASS",
                                      f"Grafana dashboards available: {len(dashboards)} dashboards")

                        # Check specific BEV dashboards
                        phase_dashboards = {}
                        for dashboard in dashboards:
                            title = dashboard.get("title", "").lower()
                            for phase in self.phases:
                                if f"phase {phase}" in title or f"phase{phase}" in title:
                                    if phase not in phase_dashboards:
                                        phase_dashboards[phase] = []
                                    phase_dashboards[phase].append(dashboard)

                        for phase in self.phases:
                            if phase in phase_dashboards:
                                self.add_result(component, f"phase_{phase}_dashboards", "PASS",
                                              f"Phase {phase} dashboards: {len(phase_dashboards[phase])}")
                            else:
                                self.add_result(component, f"phase_{phase}_dashboards", "WARN",
                                              f"No Phase {phase} dashboards found", severity="WARN")

                        return True

                    else:
                        self.add_result(component, "grafana_dashboards", "WARN",
                                      "No Grafana dashboards found", severity="WARN")
                        return False

                else:
                    self.add_result(component, "grafana_dashboards", "FAIL",
                                  f"Failed to access Grafana dashboards: HTTP {response.status}", severity="ERROR")
                    return False

        except Exception as e:
            self.add_result(component, "dashboard_validation", "FAIL",
                          f"Dashboard validation failed: {str(e)}", severity="ERROR")
            return False

    async def run_all_monitoring_validations(self) -> bool:
        """Run all monitoring and alerting validations"""
        self.logger.info("Starting comprehensive monitoring validation...")

        # Initialize HTTP session
        self.session = aiohttp.ClientSession()

        try:
            overall_success = True

            # Validate monitoring services
            for endpoint in self.monitoring_endpoints:
                self.logger.info(f"Validating monitoring service: {endpoint.name}")
                service_success = await self.validate_monitoring_service(endpoint)
                if not service_success:
                    # Don't fail overall validation for individual service failures
                    pass

            # Validate log aggregation
            log_success = self.validate_log_aggregation()
            if not log_success:
                self.logger.warning("Log aggregation validation issues detected")

            # Validate metrics collection
            metrics_success = self.validate_metrics_collection()
            if not metrics_success:
                self.logger.warning("Metrics collection validation issues detected")

            # Validate alerting rules
            alerting_success = self.validate_alerting_rules()
            if not alerting_success:
                self.logger.warning("Alerting rules validation issues detected")

            # Validate dashboards
            dashboard_success = await self.validate_dashboard_availability()
            if not dashboard_success:
                self.logger.warning("Dashboard validation issues detected")

            # Overall success is based on critical failures
            critical_failures = sum(1 for r in self.results if r.status == "FAIL" and r.severity in ["ERROR", "CRITICAL"])
            overall_success = critical_failures == 0

            return overall_success

        finally:
            await self.session.close()

    def generate_report(self) -> str:
        """Generate comprehensive monitoring validation report"""
        report = []
        report.append("="*80)
        report.append("BEV OSINT Framework - Monitoring & Alerting Validation Report")
        report.append("="*80)
        report.append(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Phases Monitored: {', '.join(self.phases)}")
        report.append(f"Total Checks: {len(self.results)}")
        report.append(f"Failed Checks: {self.failed_tests}")
        report.append(f"Warnings: {self.warnings}")
        report.append("")

        # Summary by status
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warned = sum(1 for r in self.results if r.status == "WARN")
        skipped = sum(1 for r in self.results if r.status == "SKIP")

        report.append("SUMMARY:")
        report.append(f"  ✅ Passed: {passed}")
        report.append(f"  ❌ Failed: {failed}")
        report.append(f"  ⚠️  Warnings: {warned}")
        report.append(f"  ⏭️  Skipped: {skipped}")
        report.append("")

        # Summary by component
        components = {}
        for result in self.results:
            if result.component not in components:
                components[result.component] = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0}
            components[result.component][result.status] += 1

        report.append("COMPONENT STATUS:")
        for component, counts in components.items():
            total = sum(counts.values())
            status_icon = "✅" if counts["FAIL"] == 0 else ("⚠️" if counts["FAIL"] == 0 and counts["WARN"] > 0 else "❌")
            report.append(f"  {status_icon} {component}: {total} checks ({counts['PASS']} passed, {counts['FAIL']} failed, {counts['WARN']} warnings)")

        report.append("")

        # Monitoring stack health
        monitoring_services = [r for r in self.results if r.component in ["prometheus", "grafana", "elasticsearch", "kibana"]]
        healthy_services = [r for r in monitoring_services if r.test_name == "health_check" and r.status == "PASS"]

        report.append("MONITORING STACK HEALTH:")
        report.append(f"  Healthy Services: {len(healthy_services)}/{len([e for e in self.monitoring_endpoints if e.name in ['prometheus', 'grafana', 'elasticsearch', 'kibana']])}")
        for service in healthy_services:
            report.append(f"    ✅ {service.component}")

        failed_services = [r for r in monitoring_services if r.test_name == "health_check" and r.status == "FAIL"]
        for service in failed_services:
            report.append(f"    ❌ {service.component}: {service.message}")

        report.append("")

        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 50)

        for component in components.keys():
            component_results = [r for r in self.results if r.component == component]
            component_failed = any(r.status == "FAIL" for r in component_results)

            status_icon = "❌" if component_failed else "✅"
            report.append(f"\n{status_icon} {component.upper()}:")

            for result in component_results:
                status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}.get(result.status, "❓")
                report.append(f"    {status_icon} {result.test_name}: {result.message}")
                if result.duration > 0:
                    report.append(f"      Duration: {result.duration:.3f}s")
                if result.details:
                    for key, value in result.details.items():
                        report.append(f"      {key}: {value}")

        # Recommendations
        report.append("\n" + "="*50)
        report.append("RECOMMENDATIONS:")
        report.append("-" * 50)

        if self.failed_tests > 0:
            report.append("🚨 MONITORING ISSUES FOUND")
            report.append("   Some monitoring components have failed validation.")
            report.append("   Investigate and resolve issues for complete observability.")
        elif self.warnings > 0:
            report.append("⚠️  MONITORING WARNINGS")
            report.append("   Some monitoring features may not be fully configured.")
            report.append("   Review warnings and optimize monitoring setup.")
        else:
            report.append("✅ MONITORING FULLY OPERATIONAL")
            report.append("   All monitoring and alerting components are functioning correctly.")

        # Critical issues
        critical_issues = [r for r in self.results if r.status == "FAIL" and r.severity in ["ERROR", "CRITICAL"]]
        if critical_issues:
            report.append("\nCRITICAL ISSUES TO RESOLVE:")
            for issue in critical_issues:
                report.append(f"  • {issue.component}/{issue.test_name}: {issue.message}")

        report.append("\n" + "="*80)
        return "\n".join(report)

    def save_report(self, filename: str = None) -> Path:
        """Save monitoring validation report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"monitoring_validation_{timestamp}.txt"

        report_path = self.project_root / "logs" / "monitoring" / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write(self.generate_report())

        return report_path

async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="BEV OSINT Monitoring & Alerting Validation")
    parser.add_argument("--phases", default="7,8,9",
                       help="Comma-separated list of phases to validate (default: 7,8,9)")
    parser.add_argument("--output", help="Output report file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")

    args = parser.parse_args()

    # Parse phases
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]

    # Create validator
    validator = MonitoringValidator(phases)

    # Run validation
    success = await validator.run_all_monitoring_validations()

    # Generate and display report
    report = validator.generate_report()
    print(report)

    # Save report
    report_path = validator.save_report(args.output)
    print(f"\nMonitoring validation report saved to: {report_path}")

    # Exit with appropriate code
    if success:
        print("\n✅ Monitoring validation successful - all systems operational")
        sys.exit(0)
    else:
        print(f"\n❌ Monitoring validation issues found - {validator.failed_tests} critical failures")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())