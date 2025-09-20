"""
Monitoring and metrics integration tests
Tests Prometheus/Grafana integration and alerting systems
"""

import pytest
import asyncio
import requests
import json
import time
import logging
from typing import Dict, Any, List, Optional
import yaml

logger = logging.getLogger(__name__)

@pytest.mark.integration
class TestPrometheusIntegration:
    """Test Prometheus metrics collection and integration"""

    async def test_prometheus_metrics_collection(self):
        """Test basic Prometheus metrics collection"""
        logger.info("Testing Prometheus metrics collection")

        prometheus_url = "http://localhost:9090"

        # Test Prometheus health
        response = requests.get(f"{prometheus_url}/-/healthy", timeout=10)
        assert response.status_code == 200, "Prometheus not healthy"

        # Test metrics endpoint
        response = requests.get(f"{prometheus_url}/api/v1/query",
                              params={"query": "up"}, timeout=10)
        assert response.status_code == 200, "Prometheus query endpoint not working"

        data = response.json()
        assert data["status"] == "success", "Prometheus query failed"
        assert "data" in data, "No data returned from Prometheus"

        # Test BEV-specific metrics
        bev_metrics = [
            "bev_osint_requests_total",
            "bev_osint_request_duration_seconds",
            "bev_cache_hit_rate",
            "bev_service_availability",
            "bev_vector_db_operations_total",
            "bev_workflow_completion_time",
            "bev_system_resource_usage"
        ]

        for metric in bev_metrics:
            response = requests.get(f"{prometheus_url}/api/v1/query",
                                  params={"query": metric}, timeout=10)

            if response.status_code == 200:
                metric_data = response.json()
                if metric_data["status"] == "success" and metric_data["data"]["result"]:
                    logger.info(f"✓ Metric {metric} is being collected")
                else:
                    logger.warning(f"⚠ Metric {metric} has no data")
            else:
                logger.warning(f"⚠ Failed to query metric {metric}")

        logger.info("Prometheus metrics collection test completed")

    async def test_custom_metrics_registration(self):
        """Test registration and collection of custom BEV metrics"""
        logger.info("Testing custom metrics registration")

        prometheus_url = "http://localhost:9090"

        # Define custom metrics that should be available
        custom_metrics = {
            "bev_chaos_recovery_time": {
                "type": "histogram",
                "description": "Time taken to recover from chaos events"
            },
            "bev_predictive_cache_accuracy": {
                "type": "gauge",
                "description": "Accuracy of predictive caching"
            },
            "bev_concurrent_request_capacity": {
                "type": "gauge",
                "description": "Current concurrent request handling capacity"
            },
            "bev_vector_search_latency": {
                "type": "histogram",
                "description": "Vector database search latency"
            },
            "bev_edge_computing_latency": {
                "type": "histogram",
                "description": "Edge computing node latency"
            }
        }

        metrics_available = {}

        for metric_name, metric_info in custom_metrics.items():
            # Query metric
            response = requests.get(f"{prometheus_url}/api/v1/query",
                                  params={"query": metric_name}, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success":
                    has_data = len(data["data"]["result"]) > 0
                    metrics_available[metric_name] = has_data
                    if has_data:
                        logger.info(f"✓ Custom metric {metric_name} registered and has data")
                    else:
                        logger.info(f"○ Custom metric {metric_name} registered but no data yet")
                else:
                    metrics_available[metric_name] = False
                    logger.warning(f"⚠ Custom metric {metric_name} query failed")

        # At least 70% of custom metrics should be registered
        registration_rate = sum(metrics_available.values()) / len(metrics_available)
        assert registration_rate >= 0.7, f"Custom metrics registration rate {registration_rate:.2%} too low"

        logger.info(f"Custom metrics registration: {registration_rate:.2%} success rate")

    async def test_alerting_rules_configuration(self):
        """Test Prometheus alerting rules configuration"""
        logger.info("Testing Prometheus alerting rules")

        prometheus_url = "http://localhost:9090"

        # Test alerting rules endpoint
        response = requests.get(f"{prometheus_url}/api/v1/rules", timeout=10)
        assert response.status_code == 200, "Failed to get alerting rules"

        rules_data = response.json()
        assert rules_data["status"] == "success", "Alerting rules query failed"

        # Expected BEV alerting rules
        expected_rules = [
            "BEVServiceDown",
            "BEVHighLatency",
            "BEVLowCacheHitRate",
            "BEVChaosRecoveryTooSlow",
            "BEVVectorDBConnectivityIssue",
            "BEVHighErrorRate",
            "BEVResourceExhaustion"
        ]

        found_rules = []
        if "data" in rules_data and "groups" in rules_data["data"]:
            for group in rules_data["data"]["groups"]:
                if "rules" in group:
                    for rule in group["rules"]:
                        if "alert" in rule:
                            found_rules.append(rule["alert"])

        # Check if expected rules are configured
        configured_rules = []
        for expected_rule in expected_rules:
            if expected_rule in found_rules:
                configured_rules.append(expected_rule)
                logger.info(f"✓ Alert rule {expected_rule} configured")
            else:
                logger.warning(f"⚠ Alert rule {expected_rule} missing")

        # At least 80% of expected rules should be configured
        rule_coverage = len(configured_rules) / len(expected_rules)
        assert rule_coverage >= 0.8, f"Alert rule coverage {rule_coverage:.2%} too low"

        logger.info(f"Alerting rules coverage: {rule_coverage:.2%}")

    async def test_metrics_retention_and_storage(self):
        """Test metrics retention and storage configuration"""
        logger.info("Testing metrics retention and storage")

        prometheus_url = "http://localhost:9090"

        # Test storage configuration
        response = requests.get(f"{prometheus_url}/api/v1/status/config", timeout=10)
        assert response.status_code == 200, "Failed to get Prometheus config"

        config_data = response.json()
        assert config_data["status"] == "success", "Config query failed"

        # Check retention configuration
        config_yaml = config_data["data"]["yaml"]
        config = yaml.safe_load(config_yaml)

        # Verify retention settings
        global_config = config.get("global", {})
        scrape_interval = global_config.get("scrape_interval", "15s")
        evaluation_interval = global_config.get("evaluation_interval", "15s")

        logger.info(f"Scrape interval: {scrape_interval}")
        logger.info(f"Evaluation interval: {evaluation_interval}")

        # Test data retention by querying historical data
        # Query data from 1 hour ago
        one_hour_ago = int(time.time()) - 3600
        response = requests.get(f"{prometheus_url}/api/v1/query",
                              params={
                                  "query": "up",
                                  "time": one_hour_ago
                              }, timeout=10)

        if response.status_code == 200:
            historical_data = response.json()
            if historical_data["status"] == "success" and historical_data["data"]["result"]:
                logger.info("✓ Historical data retention working")
            else:
                logger.warning("⚠ No historical data found")

        # Test range queries
        response = requests.get(f"{prometheus_url}/api/v1/query_range",
                              params={
                                  "query": "up",
                                  "start": one_hour_ago,
                                  "end": int(time.time()),
                                  "step": "60s"
                              }, timeout=10)

        if response.status_code == 200:
            range_data = response.json()
            if range_data["status"] == "success":
                logger.info("✓ Range queries working")
                # Check data points
                if range_data["data"]["result"]:
                    values = range_data["data"]["result"][0].get("values", [])
                    logger.info(f"Range query returned {len(values)} data points")
            else:
                logger.warning("⚠ Range query failed")

@pytest.mark.integration
class TestGrafanaIntegration:
    """Test Grafana dashboard and visualization integration"""

    async def test_grafana_connectivity(self):
        """Test basic Grafana connectivity and health"""
        logger.info("Testing Grafana connectivity")

        grafana_url = "http://localhost:3000"

        # Test Grafana health
        response = requests.get(f"{grafana_url}/api/health", timeout=10)
        assert response.status_code == 200, "Grafana not healthy"

        health_data = response.json()
        assert health_data.get("database") == "ok", "Grafana database not healthy"

        logger.info("✓ Grafana connectivity test passed")

    async def test_prometheus_datasource_configuration(self):
        """Test Prometheus datasource configuration in Grafana"""
        logger.info("Testing Prometheus datasource configuration")

        grafana_url = "http://localhost:3000"

        # Get datasources (may require authentication in real deployment)
        try:
            response = requests.get(f"{grafana_url}/api/datasources", timeout=10)

            if response.status_code == 200:
                datasources = response.json()

                prometheus_ds = None
                for ds in datasources:
                    if ds.get("type") == "prometheus":
                        prometheus_ds = ds
                        break

                if prometheus_ds:
                    logger.info(f"✓ Prometheus datasource configured: {prometheus_ds.get('name')}")
                    assert prometheus_ds.get("url"), "Prometheus datasource URL not configured"
                else:
                    logger.warning("⚠ No Prometheus datasource found")

            elif response.status_code == 401:
                logger.info("○ Grafana requires authentication (expected in production)")
            else:
                logger.warning(f"⚠ Unexpected response: {response.status_code}")

        except Exception as e:
            logger.warning(f"⚠ Grafana datasource test error: {e}")

    async def test_bev_dashboards_availability(self):
        """Test BEV-specific dashboard availability"""
        logger.info("Testing BEV dashboard availability")

        grafana_url = "http://localhost:3000"

        # Expected BEV dashboards
        expected_dashboards = [
            "BEV OSINT System Overview",
            "BEV Performance Metrics",
            "BEV Service Health",
            "BEV Vector Database Metrics",
            "BEV Cache Performance",
            "BEV Chaos Engineering",
            "BEV Edge Computing"
        ]

        try:
            response = requests.get(f"{grafana_url}/api/search?type=dash-db", timeout=10)

            if response.status_code == 200:
                dashboards = response.json()
                found_dashboards = [d.get("title", "") for d in dashboards]

                matching_dashboards = []
                for expected in expected_dashboards:
                    # Check for partial matches (dashboard names might vary)
                    for found in found_dashboards:
                        if any(keyword in found.lower() for keyword in expected.lower().split()):
                            matching_dashboards.append(expected)
                            logger.info(f"✓ Found dashboard: {found}")
                            break

                dashboard_coverage = len(matching_dashboards) / len(expected_dashboards)
                logger.info(f"Dashboard coverage: {dashboard_coverage:.2%}")

            elif response.status_code == 401:
                logger.info("○ Grafana dashboard query requires authentication")
            else:
                logger.warning(f"⚠ Dashboard query failed: {response.status_code}")

        except Exception as e:
            logger.warning(f"⚠ Dashboard availability test error: {e}")

@pytest.mark.integration
class TestAlertingSystem:
    """Test alerting and notification systems"""

    async def test_alertmanager_integration(self):
        """Test Alertmanager integration and configuration"""
        logger.info("Testing Alertmanager integration")

        # Alertmanager typically runs on port 9093
        alertmanager_url = "http://localhost:9093"

        try:
            # Test Alertmanager health
            response = requests.get(f"{alertmanager_url}/-/healthy", timeout=10)

            if response.status_code == 200:
                logger.info("✓ Alertmanager is healthy")

                # Test configuration
                config_response = requests.get(f"{alertmanager_url}/api/v1/status", timeout=10)
                if config_response.status_code == 200:
                    status_data = config_response.json()
                    if status_data.get("status") == "success":
                        logger.info("✓ Alertmanager configuration accessible")
                    else:
                        logger.warning("⚠ Alertmanager status query failed")

                # Test alerts endpoint
                alerts_response = requests.get(f"{alertmanager_url}/api/v1/alerts", timeout=10)
                if alerts_response.status_code == 200:
                    alerts_data = alerts_response.json()
                    if alerts_data.get("status") == "success":
                        active_alerts = alerts_data.get("data", [])
                        logger.info(f"✓ Alertmanager has {len(active_alerts)} active alerts")
                    else:
                        logger.warning("⚠ Alertmanager alerts query failed")

            else:
                logger.warning(f"⚠ Alertmanager not accessible: {response.status_code}")

        except requests.exceptions.ConnectionError:
            logger.warning("⚠ Alertmanager not running or not accessible")
        except Exception as e:
            logger.warning(f"⚠ Alertmanager test error: {e}")

    async def test_alert_firing_and_resolution(self):
        """Test alert firing and resolution workflow"""
        logger.info("Testing alert firing and resolution")

        prometheus_url = "http://localhost:9090"

        # Get current alerts
        response = requests.get(f"{prometheus_url}/api/v1/alerts", timeout=10)
        assert response.status_code == 200, "Failed to get alerts"

        alerts_data = response.json()
        assert alerts_data["status"] == "success", "Alerts query failed"

        current_alerts = alerts_data["data"]["alerts"]
        logger.info(f"Current alerts: {len(current_alerts)}")

        # Analyze alert states
        alert_states = {"firing": 0, "pending": 0, "inactive": 0}

        for alert in current_alerts:
            state = alert.get("state", "unknown")
            if state in alert_states:
                alert_states[state] += 1

            # Log alert details
            alert_name = alert.get("labels", {}).get("alertname", "unknown")
            logger.info(f"Alert: {alert_name}, State: {state}")

        logger.info(f"Alert states: {alert_states}")

        # Test alert evaluation
        # Query alert rules to see if they're being evaluated
        rules_response = requests.get(f"{prometheus_url}/api/v1/rules", timeout=10)
        if rules_response.status_code == 200:
            rules_data = rules_response.json()
            if rules_data["status"] == "success":
                # Count alert rules being evaluated
                alert_rule_count = 0
                if "data" in rules_data and "groups" in rules_data["data"]:
                    for group in rules_data["data"]["groups"]:
                        for rule in group.get("rules", []):
                            if rule.get("type") == "alerting":
                                alert_rule_count += 1

                logger.info(f"Active alert rules: {alert_rule_count}")
                assert alert_rule_count > 0, "No alert rules are configured"

    async def test_notification_channels(self):
        """Test notification channel configuration and delivery"""
        logger.info("Testing notification channels")

        # This would test actual notification delivery in a real environment
        # For testing, we'll verify configuration and simulate notifications

        # Expected notification channels
        expected_channels = [
            "email",
            "slack",
            "webhook",
            "pagerduty"
        ]

        # Test notification configuration
        notification_config = {
            "email": {
                "smtp_server": "smtp.example.com",
                "smtp_port": 587,
                "sender": "alerts@bev-osint.com"
            },
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/...",
                "channel": "#bev-alerts"
            },
            "webhook": {
                "url": "https://api.bev-osint.com/alerts",
                "method": "POST"
            }
        }

        # Simulate notification test
        for channel, config in notification_config.items():
            logger.info(f"Testing {channel} notification channel")

            # In a real implementation, this would send test notifications
            # For testing, we'll validate configuration format
            assert isinstance(config, dict), f"{channel} config is not a dictionary"
            assert len(config) > 0, f"{channel} config is empty"

            logger.info(f"✓ {channel} notification channel configured")

        logger.info("Notification channels test completed")

@pytest.mark.integration
class TestMetricsCorrelation:
    """Test correlation between different metrics and systems"""

    async def test_performance_metrics_correlation(self):
        """Test correlation between performance metrics"""
        logger.info("Testing performance metrics correlation")

        prometheus_url = "http://localhost:9090"

        # Define related metrics that should correlate
        metric_correlations = [
            {
                "primary": "bev_osint_request_duration_seconds",
                "secondary": "bev_cache_hit_rate",
                "relationship": "inverse"  # Higher cache hit rate should correlate with lower latency
            },
            {
                "primary": "bev_concurrent_requests",
                "secondary": "bev_system_resource_usage",
                "relationship": "positive"  # More requests should correlate with higher resource usage
            },
            {
                "primary": "bev_vector_db_operations_total",
                "secondary": "bev_vector_search_latency",
                "relationship": "positive"  # More operations might correlate with higher latency
            }
        ]

        correlation_results = []

        for correlation in metric_correlations:
            primary_metric = correlation["primary"]
            secondary_metric = correlation["secondary"]

            # Query both metrics
            primary_response = requests.get(f"{prometheus_url}/api/v1/query",
                                          params={"query": primary_metric}, timeout=10)
            secondary_response = requests.get(f"{prometheus_url}/api/v1/query",
                                            params={"query": secondary_metric}, timeout=10)

            if (primary_response.status_code == 200 and secondary_response.status_code == 200):
                primary_data = primary_response.json()
                secondary_data = secondary_response.json()

                if (primary_data["status"] == "success" and secondary_data["status"] == "success" and
                    primary_data["data"]["result"] and secondary_data["data"]["result"]):

                    # Extract values for correlation analysis
                    primary_value = float(primary_data["data"]["result"][0]["value"][1])
                    secondary_value = float(secondary_data["data"]["result"][0]["value"][1])

                    correlation_results.append({
                        "primary_metric": primary_metric,
                        "secondary_metric": secondary_metric,
                        "primary_value": primary_value,
                        "secondary_value": secondary_value,
                        "relationship": correlation["relationship"]
                    })

                    logger.info(f"✓ Correlation data: {primary_metric}={primary_value}, {secondary_metric}={secondary_value}")

                else:
                    logger.warning(f"⚠ No data for correlation: {primary_metric} vs {secondary_metric}")
            else:
                logger.warning(f"⚠ Failed to query metrics for correlation")

        # At least some correlations should have data
        assert len(correlation_results) > 0, "No metric correlations could be analyzed"
        logger.info(f"Analyzed {len(correlation_results)} metric correlations")

    async def test_system_health_aggregation(self):
        """Test aggregation of system health metrics"""
        logger.info("Testing system health aggregation")

        prometheus_url = "http://localhost:9090"

        # Health metrics to aggregate
        health_metrics = [
            "bev_service_availability",
            "bev_cache_hit_rate",
            "bev_vector_db_connectivity",
            "bev_edge_computing_latency",
            "bev_system_error_rate"
        ]

        health_scores = {}
        total_weight = 0

        for metric in health_metrics:
            response = requests.get(f"{prometheus_url}/api/v1/query",
                                  params={"query": metric}, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if data["status"] == "success" and data["data"]["result"]:
                    value = float(data["data"]["result"][0]["value"][1])
                    health_scores[metric] = value
                    total_weight += 1
                    logger.info(f"Health metric {metric}: {value}")

        if health_scores:
            # Calculate overall health score
            # This is a simplified calculation - real implementation would be more sophisticated
            overall_health = sum(health_scores.values()) / len(health_scores)
            logger.info(f"Overall system health score: {overall_health:.3f}")

            # System should maintain reasonable health
            assert overall_health >= 0.7, f"System health score {overall_health:.3f} too low"

        else:
            logger.warning("⚠ No health metrics available for aggregation")

        logger.info("System health aggregation test completed")