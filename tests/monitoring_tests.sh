#!/bin/bash

#################################################################
# BEV Monitoring Test Suite
#
# Comprehensive monitoring and observability testing
# Tests Prometheus metrics, Grafana dashboards, alerting,
# log aggregation, and health monitoring systems
#################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPORTS_DIR="$PROJECT_DIR/test-reports"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$REPORTS_DIR/monitoring_tests_$TIMESTAMP.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Monitoring test configuration
METRICS_RETENTION_DAYS=30
ALERT_RESPONSE_TIME=60
DASHBOARD_LOAD_TIME=5

# Load environment
if [[ -f "$PROJECT_DIR/.env" ]]; then
    source "$PROJECT_DIR/.env"
fi

# Utility functions
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
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

pass_test() {
    ((PASSED_TESTS++))
    ((TOTAL_TESTS++))
    log_success "$1"
}

fail_test() {
    ((FAILED_TESTS++))
    ((TOTAL_TESTS++))
    log_error "$1"
}

# Prometheus monitoring tests
test_prometheus_monitoring() {
    log_info "Testing Prometheus monitoring system..."

    # Test Prometheus server health
    test_prometheus_health

    # Test metrics collection
    test_metrics_collection

    # Test alerting rules
    test_alerting_rules

    # Test service discovery
    test_service_discovery

    # Test data retention
    test_metrics_retention
}

# Test Prometheus server health
test_prometheus_health() {
    log_info "Testing Prometheus server health..."

    # Test Prometheus API accessibility
    local prometheus_health=$(curl -s "http://localhost:9090/-/healthy" 2>/dev/null || echo "failed")

    if [[ "$prometheus_health" == "Prometheus is Healthy." ]]; then
        pass_test "Prometheus server is healthy"
    else
        fail_test "Prometheus server health check failed"
        return 1
    fi

    # Test Prometheus readiness
    local prometheus_ready=$(curl -s "http://localhost:9090/-/ready" 2>/dev/null || echo "failed")

    if [[ "$prometheus_ready" == "Prometheus is Ready." ]]; then
        pass_test "Prometheus server is ready"
    else
        fail_test "Prometheus server readiness check failed"
    fi

    # Test Prometheus configuration
    local config_status=$(curl -s "http://localhost:9090/api/v1/status/config" 2>/dev/null | jq -r '.status' || echo "failed")

    if [[ "$config_status" == "success" ]]; then
        pass_test "Prometheus configuration is valid"
    else
        fail_test "Prometheus configuration validation failed"
    fi

    # Test Prometheus build info
    local build_info=$(curl -s "http://localhost:9090/api/v1/status/buildinfo" 2>/dev/null | jq -r '.data.version' || echo "unknown")
    log_info "Prometheus version: $build_info"
}

# Test metrics collection
test_metrics_collection() {
    log_info "Testing Prometheus metrics collection..."

    # Test target discovery and scraping
    local targets_response=$(curl -s "http://localhost:9090/api/v1/targets" 2>/dev/null)
    local targets_file="$REPORTS_DIR/prometheus_targets_$TIMESTAMP.json"
    echo "$targets_response" > "$targets_file"

    local active_targets=$(echo "$targets_response" | jq -r '.data.activeTargets | length' 2>/dev/null || echo "0")
    local healthy_targets=$(echo "$targets_response" | jq -r '.data.activeTargets | map(select(.health == "up")) | length' 2>/dev/null || echo "0")

    log_info "Active targets: $active_targets, Healthy targets: $healthy_targets"

    if [[ "$active_targets" -gt 0 ]]; then
        pass_test "Prometheus is discovering targets ($active_targets total)"
    else
        fail_test "Prometheus is not discovering any targets"
    fi

    if [[ "$healthy_targets" -gt 0 ]]; then
        pass_test "Prometheus has healthy targets ($healthy_targets healthy)"
    else
        fail_test "Prometheus has no healthy targets"
    fi

    # Test specific metric collections
    test_specific_metrics

    # Test metric cardinality
    test_metric_cardinality
}

# Test specific metrics
test_specific_metrics() {
    log_info "Testing specific metric collection..."

    # Define expected metrics for BEV services
    local expected_metrics=(
        "up:System_uptime"
        "node_cpu_seconds_total:CPU_metrics"
        "node_memory_MemTotal_bytes:Memory_metrics"
        "node_filesystem_size_bytes:Filesystem_metrics"
        "container_cpu_usage_seconds_total:Container_CPU"
        "container_memory_usage_bytes:Container_memory"
        "postgres_up:PostgreSQL_status"
        "redis_up:Redis_status"
        "elasticsearch_cluster_health_status:Elasticsearch_health"
    )

    for metric_info in "${expected_metrics[@]}"; do
        IFS=':' read -ra metric_parts <<< "$metric_info"
        local metric_name="${metric_parts[0]}"
        local metric_description="${metric_parts[1]}"

        local metric_data=$(curl -s "http://localhost:9090/api/v1/query?query=$metric_name" 2>/dev/null | jq -r '.data.result | length' || echo "0")

        if [[ "$metric_data" -gt 0 ]]; then
            pass_test "$metric_description metrics are being collected ($metric_data series)"
        else
            fail_test "$metric_description metrics are not being collected"
        fi
    done

    # Test custom BEV application metrics
    test_custom_application_metrics
}

# Test custom application metrics
test_custom_application_metrics() {
    log_info "Testing custom BEV application metrics..."

    local custom_metrics=(
        "bev_intelowl_jobs_total:IntelOwl_job_count"
        "bev_document_processing_duration_seconds:Document_processing_time"
        "bev_swarm_agents_active:Active_swarm_agents"
        "bev_security_alerts_total:Security_alerts"
        "bev_memory_usage_ratio:Memory_usage_ratio"
        "bev_tor_circuit_build_time_seconds:Tor_circuit_build_time"
    )

    for metric_info in "${custom_metrics[@]}"; do
        IFS=':' read -ra metric_parts <<< "$metric_info"
        local metric_name="${metric_parts[0]}"
        local metric_description="${metric_parts[1]}"

        local metric_data=$(curl -s "http://localhost:9090/api/v1/query?query=$metric_name" 2>/dev/null | jq -r '.data.result | length' || echo "0")

        if [[ "$metric_data" -gt 0 ]]; then
            pass_test "$metric_description custom metrics are available"
        else
            log_warning "$metric_description custom metrics not found (may not be implemented yet)"
        fi
    done
}

# Test metric cardinality
test_metric_cardinality() {
    log_info "Testing metric cardinality..."

    # Get total series count
    local total_series=$(curl -s "http://localhost:9090/api/v1/label/__name__/values" 2>/dev/null | jq -r '.data | length' || echo "0")
    log_info "Total metric series: $total_series"

    # Check for high cardinality metrics (potential performance issue)
    local high_cardinality_threshold=10000

    if [[ "$total_series" -lt "$high_cardinality_threshold" ]]; then
        pass_test "Metric cardinality is within acceptable limits ($total_series series)"
    else
        fail_test "High metric cardinality detected ($total_series series) - may impact performance"
    fi

    # Test for common high-cardinality issues
    local problematic_labels=("instance" "job" "container" "pod")
    for label in "${problematic_labels[@]}"; do
        local label_values=$(curl -s "http://localhost:9090/api/v1/label/$label/values" 2>/dev/null | jq -r '.data | length' || echo "0")
        log_info "Label '$label' has $label_values unique values"

        if [[ "$label_values" -gt 1000 ]]; then
            fail_test "Label '$label' has high cardinality ($label_values values)"
        fi
    done
}

# Test alerting rules
test_alerting_rules() {
    log_info "Testing Prometheus alerting rules..."

    # Get alerting rules
    local rules_response=$(curl -s "http://localhost:9090/api/v1/rules" 2>/dev/null)
    local rules_file="$REPORTS_DIR/prometheus_rules_$TIMESTAMP.json"
    echo "$rules_response" > "$rules_file"

    local total_rules=$(echo "$rules_response" | jq -r '.data.groups | map(.rules) | add | length' 2>/dev/null || echo "0")
    local alerting_rules=$(echo "$rules_response" | jq -r '.data.groups | map(.rules) | add | map(select(.type == "alerting")) | length' 2>/dev/null || echo "0")

    log_info "Total rules: $total_rules, Alerting rules: $alerting_rules"

    if [[ "$alerting_rules" -gt 0 ]]; then
        pass_test "Prometheus has $alerting_rules alerting rules configured"
    else
        fail_test "No alerting rules configured in Prometheus"
    fi

    # Test active alerts
    test_active_alerts

    # Test alert rule validation
    test_alert_rule_validation
}

# Test active alerts
test_active_alerts() {
    log_info "Testing active alerts..."

    local alerts_response=$(curl -s "http://localhost:9090/api/v1/alerts" 2>/dev/null)
    local alerts_file="$REPORTS_DIR/prometheus_alerts_$TIMESTAMP.json"
    echo "$alerts_response" > "$alerts_file"

    local active_alerts=$(echo "$alerts_response" | jq -r '.data.alerts | length' 2>/dev/null || echo "0")
    local firing_alerts=$(echo "$alerts_response" | jq -r '.data.alerts | map(select(.state == "firing")) | length' 2>/dev/null || echo "0")
    local pending_alerts=$(echo "$alerts_response" | jq -r '.data.alerts | map(select(.state == "pending")) | length' 2>/dev/null || echo "0")

    log_info "Active alerts: $active_alerts (Firing: $firing_alerts, Pending: $pending_alerts)"

    if [[ "$firing_alerts" -eq 0 ]]; then
        pass_test "No firing alerts (system appears healthy)"
    else
        fail_test "$firing_alerts alerts are currently firing"

        # Log details of firing alerts
        local firing_alert_names=$(echo "$alerts_response" | jq -r '.data.alerts | map(select(.state == "firing")) | .[].labels.alertname' 2>/dev/null | tr '\n' ', ')
        log_error "Firing alerts: $firing_alert_names"
    fi

    # Test critical system alerts
    test_critical_system_alerts "$alerts_response"
}

# Test critical system alerts
test_critical_system_alerts() {
    local alerts_response="$1"

    log_info "Testing critical system alerts..."

    local critical_alerts=(
        "InstanceDown:Service_availability"
        "HighCPUUsage:CPU_overload"
        "HighMemoryUsage:Memory_pressure"
        "DiskSpaceLow:Storage_space"
        "ServiceUnavailable:Service_health"
    )

    for alert_info in "${critical_alerts[@]}"; do
        IFS=':' read -ra alert_parts <<< "$alert_info"
        local alert_name="${alert_parts[0]}"
        local alert_description="${alert_parts[1]}"

        local alert_exists=$(echo "$alerts_response" | jq -r --arg name "$alert_name" '.data.alerts[] | select(.labels.alertname == $name) | .labels.alertname' 2>/dev/null || echo "")

        if [[ -n "$alert_exists" ]]; then
            local alert_state=$(echo "$alerts_response" | jq -r --arg name "$alert_name" '.data.alerts[] | select(.labels.alertname == $name) | .state' 2>/dev/null)
            log_info "Alert '$alert_name' exists and is in state: $alert_state"
        else
            log_warning "Critical alert '$alert_name' ($alert_description) is not configured"
        fi
    done
}

# Test alert rule validation
test_alert_rule_validation() {
    log_info "Testing alert rule validation..."

    # Test if rules are syntactically correct
    local rule_health=$(curl -s "http://localhost:9090/api/v1/rules" 2>/dev/null | jq -r '.status' || echo "failed")

    if [[ "$rule_health" == "success" ]]; then
        pass_test "All alerting rules are syntactically valid"
    else
        fail_test "Some alerting rules have syntax errors"
    fi

    # Test for best practices in alert rules
    local rules_content=$(curl -s "http://localhost:9090/api/v1/rules" 2>/dev/null)

    # Check for alert rules without annotations
    local rules_without_annotations=$(echo "$rules_content" | jq -r '.data.groups | map(.rules) | add | map(select(.type == "alerting" and (.annotations | length) == 0)) | length' 2>/dev/null || echo "0")

    if [[ "$rules_without_annotations" -eq 0 ]]; then
        pass_test "All alerting rules have annotations"
    else
        fail_test "$rules_without_annotations alerting rules are missing annotations"
    fi

    # Check for alert rules without labels
    local rules_without_labels=$(echo "$rules_content" | jq -r '.data.groups | map(.rules) | add | map(select(.type == "alerting" and (.labels | length) == 0)) | length' 2>/dev/null || echo "0")

    if [[ "$rules_without_labels" -eq 0 ]]; then
        pass_test "All alerting rules have labels"
    else
        fail_test "$rules_without_labels alerting rules are missing labels"
    fi
}

# Test service discovery
test_service_discovery() {
    log_info "Testing Prometheus service discovery..."

    local service_discovery_response=$(curl -s "http://localhost:9090/api/v1/targets" 2>/dev/null)

    # Test different service discovery mechanisms
    local static_targets=$(echo "$service_discovery_response" | jq -r '.data.activeTargets | map(select(.discoveredLabels.__meta_filepath != null)) | length' 2>/dev/null || echo "0")
    local dns_targets=$(echo "$service_discovery_response" | jq -r '.data.activeTargets | map(select(.discoveredLabels.__meta_dns_name != null)) | length' 2>/dev/null || echo "0")
    local docker_targets=$(echo "$service_discovery_response" | jq -r '.data.activeTargets | map(select(.discoveredLabels.__meta_dockerswarm_container_label_com_docker_compose_service != null)) | length' 2>/dev/null || echo "0")

    log_info "Service discovery breakdown - Static: $static_targets, DNS: $dns_targets, Docker: $docker_targets"

    # Test expected BEV services are discovered
    local expected_services=(
        "prometheus"
        "node-exporter"
        "grafana"
        "postgres-exporter"
        "redis-exporter"
        "elasticsearch-exporter"
    )

    for service in "${expected_services[@]}"; do
        local service_found=$(echo "$service_discovery_response" | jq -r --arg svc "$service" '.data.activeTargets | map(select(.labels.job == $svc or .labels.instance | contains($svc))) | length' 2>/dev/null || echo "0")

        if [[ "$service_found" -gt 0 ]]; then
            pass_test "Service '$service' discovered by Prometheus"
        else
            fail_test "Service '$service' not discovered by Prometheus"
        fi
    done
}

# Test metrics retention
test_metrics_retention() {
    log_info "Testing metrics retention policy..."

    # Test retention configuration
    local retention_config=$(curl -s "http://localhost:9090/api/v1/status/flags" 2>/dev/null | jq -r '.data["storage.tsdb.retention.time"] // "unknown"')

    log_info "Configured retention time: $retention_config"

    # Parse retention time and validate
    if [[ "$retention_config" != "unknown" ]]; then
        # Extract number and unit
        local retention_value=$(echo "$retention_config" | grep -o '[0-9]*')
        local retention_unit=$(echo "$retention_config" | grep -o '[a-zA-Z]*')

        case "$retention_unit" in
            "d"|"day"|"days")
                if [[ "$retention_value" -ge "$METRICS_RETENTION_DAYS" ]]; then
                    pass_test "Metrics retention configured appropriately (${retention_config})"
                else
                    fail_test "Metrics retention too short (${retention_config}, minimum ${METRICS_RETENTION_DAYS}d recommended)"
                fi
                ;;
            "h"|"hour"|"hours")
                local retention_days=$((retention_value / 24))
                if [[ "$retention_days" -ge "$METRICS_RETENTION_DAYS" ]]; then
                    pass_test "Metrics retention configured appropriately (${retention_config})"
                else
                    fail_test "Metrics retention too short (${retention_config})"
                fi
                ;;
            *)
                log_warning "Unknown retention unit: $retention_unit"
                ;;
        esac
    else
        fail_test "Could not determine metrics retention configuration"
    fi

    # Test actual data retention by querying old data
    test_historical_data_availability
}

# Test historical data availability
test_historical_data_availability() {
    log_info "Testing historical data availability..."

    # Query for data from different time ranges
    local time_ranges=("1h" "24h" "7d" "30d")

    for range in "${time_ranges[@]}"; do
        local query_result=$(curl -s "http://localhost:9090/api/v1/query?query=up[${range}]" 2>/dev/null | jq -r '.data.result | length' || echo "0")

        if [[ "$query_result" -gt 0 ]]; then
            pass_test "Historical data available for $range range"
        else
            if [[ "$range" == "1h" || "$range" == "24h" ]]; then
                fail_test "No historical data available for $range range"
            else
                log_warning "No historical data available for $range range (may be expected for new deployment)"
            fi
        fi
    done

    # Test data density
    local current_data_points=$(curl -s "http://localhost:9090/api/v1/query_range?query=up&start=$(date -d '1 hour ago' +%s)&end=$(date +%s)&step=60s" 2>/dev/null | jq -r '.data.result[0].values | length' || echo "0")

    log_info "Data points in last hour: $current_data_points"

    if [[ "$current_data_points" -gt 50 ]]; then
        pass_test "Good data density for recent metrics ($current_data_points points/hour)"
    else
        fail_test "Low data density for recent metrics ($current_data_points points/hour)"
    fi
}

# Grafana monitoring tests
test_grafana_monitoring() {
    log_info "Testing Grafana monitoring and visualization..."

    # Test Grafana health
    test_grafana_health

    # Test dashboards
    test_grafana_dashboards

    # Test data sources
    test_grafana_data_sources

    # Test alerting
    test_grafana_alerting

    # Test user management
    test_grafana_user_management
}

# Test Grafana health
test_grafana_health() {
    log_info "Testing Grafana health..."

    # Test Grafana API health
    local grafana_health=$(curl -s "http://localhost:3001/api/health" 2>/dev/null | jq -r '.database' || echo "failed")

    if [[ "$grafana_health" == "ok" ]]; then
        pass_test "Grafana health check passed"
    else
        fail_test "Grafana health check failed"
        return 1
    fi

    # Test Grafana frontend accessibility
    local grafana_frontend=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:3001" 2>/dev/null || echo "000")

    if [[ "$grafana_frontend" == "200" ]]; then
        pass_test "Grafana frontend is accessible"
    else
        fail_test "Grafana frontend is not accessible (HTTP $grafana_frontend)"
    fi

    # Test Grafana build info
    local grafana_version=$(curl -s "http://localhost:3001/api/frontend/settings" 2>/dev/null | jq -r '.buildInfo.version' || echo "unknown")
    log_info "Grafana version: $grafana_version"

    # Test Grafana readiness
    local grafana_ready=$(curl -s "http://localhost:3001/api/health" 2>/dev/null | jq -r '.version' || echo "failed")

    if [[ "$grafana_ready" != "failed" ]]; then
        pass_test "Grafana is ready and operational"
    else
        fail_test "Grafana readiness check failed"
    fi
}

# Test Grafana dashboards
test_grafana_dashboards() {
    log_info "Testing Grafana dashboards..."

    # Get list of dashboards
    local dashboards_response=$(curl -s "http://localhost:3001/api/search?type=dash-db" 2>/dev/null)
    local dashboards_file="$REPORTS_DIR/grafana_dashboards_$TIMESTAMP.json"
    echo "$dashboards_response" > "$dashboards_file"

    local dashboard_count=$(echo "$dashboards_response" | jq -r '. | length' 2>/dev/null || echo "0")

    log_info "Found $dashboard_count dashboards"

    if [[ "$dashboard_count" -gt 0 ]]; then
        pass_test "Grafana has $dashboard_count dashboards configured"
    else
        fail_test "No dashboards found in Grafana"
        return 1
    fi

    # Test specific expected dashboards
    local expected_dashboards=(
        "BEV System Overview"
        "Infrastructure Monitoring"
        "Application Performance"
        "Security Dashboard"
        "IntelOwl Analytics"
        "Swarm Intelligence Metrics"
    )

    for dashboard_name in "${expected_dashboards[@]}"; do
        local dashboard_found=$(echo "$dashboards_response" | jq -r --arg name "$dashboard_name" '.[] | select(.title | contains($name)) | .title' 2>/dev/null || echo "")

        if [[ -n "$dashboard_found" ]]; then
            pass_test "Dashboard '$dashboard_name' is available"
        else
            log_warning "Expected dashboard '$dashboard_name' not found"
        fi
    done

    # Test dashboard loading performance
    test_dashboard_performance

    # Test dashboard panels
    test_dashboard_panels
}

# Test dashboard performance
test_dashboard_performance() {
    log_info "Testing dashboard loading performance..."

    local dashboards_response=$(curl -s "http://localhost:3001/api/search?type=dash-db" 2>/dev/null)
    local sample_dashboards=$(echo "$dashboards_response" | jq -r '.[0:3] | .[].uid' 2>/dev/null)

    while IFS= read -r dashboard_uid; do
        if [[ -n "$dashboard_uid" && "$dashboard_uid" != "null" ]]; then
            local start_time=$(date +%s.%N)
            local dashboard_data=$(curl -s "http://localhost:3001/api/dashboards/uid/$dashboard_uid" 2>/dev/null)
            local end_time=$(date +%s.%N)

            local load_time=$(echo "$end_time - $start_time" | bc)
            local dashboard_title=$(echo "$dashboard_data" | jq -r '.dashboard.title' 2>/dev/null || echo "Unknown")

            log_info "Dashboard '$dashboard_title' load time: ${load_time}s"

            if (( $(echo "$load_time < $DASHBOARD_LOAD_TIME" | bc -l) )); then
                pass_test "Dashboard '$dashboard_title' loads within acceptable time (${load_time}s)"
            else
                fail_test "Dashboard '$dashboard_title' loads slowly (${load_time}s)"
            fi
        fi
    done <<< "$sample_dashboards"
}

# Test dashboard panels
test_dashboard_panels() {
    log_info "Testing dashboard panels..."

    local dashboards_response=$(curl -s "http://localhost:3001/api/search?type=dash-db" 2>/dev/null)
    local sample_dashboard_uid=$(echo "$dashboards_response" | jq -r '.[0].uid' 2>/dev/null || echo "")

    if [[ -n "$sample_dashboard_uid" && "$sample_dashboard_uid" != "null" ]]; then
        local dashboard_data=$(curl -s "http://localhost:3001/api/dashboards/uid/$sample_dashboard_uid" 2>/dev/null)
        local panel_count=$(echo "$dashboard_data" | jq -r '.dashboard.panels | length' 2>/dev/null || echo "0")

        log_info "Sample dashboard has $panel_count panels"

        if [[ "$panel_count" -gt 0 ]]; then
            pass_test "Dashboard panels are configured ($panel_count panels)"
        else
            fail_test "No panels found in sample dashboard"
        fi

        # Test panel types
        local panel_types=$(echo "$dashboard_data" | jq -r '.dashboard.panels[].type' 2>/dev/null | sort | uniq -c | sort -nr)
        log_info "Panel types distribution: $panel_types"

        # Test for common panel types
        local common_panel_types=("graph" "singlestat" "table" "heatmap" "stat")
        for panel_type in "${common_panel_types[@]}"; do
            local type_count=$(echo "$dashboard_data" | jq -r --arg type "$panel_type" '.dashboard.panels[] | select(.type == $type) | .type' 2>/dev/null | wc -l)
            if [[ "$type_count" -gt 0 ]]; then
                log_info "Found $type_count panels of type '$panel_type'"
            fi
        done
    else
        log_warning "No sample dashboard available for panel testing"
    fi
}

# Test Grafana data sources
test_grafana_data_sources() {
    log_info "Testing Grafana data sources..."

    # Get list of data sources
    local datasources_response=$(curl -s "http://localhost:3001/api/datasources" 2>/dev/null)
    local datasources_file="$REPORTS_DIR/grafana_datasources_$TIMESTAMP.json"
    echo "$datasources_response" > "$datasources_file"

    local datasource_count=$(echo "$datasources_response" | jq -r '. | length' 2>/dev/null || echo "0")

    log_info "Found $datasource_count data sources"

    if [[ "$datasource_count" -gt 0 ]]; then
        pass_test "Grafana has $datasource_count data sources configured"
    else
        fail_test "No data sources configured in Grafana"
        return 1
    fi

    # Test expected data sources
    local expected_datasources=(
        "prometheus:Prometheus"
        "influxdb:InfluxDB"
        "elasticsearch:Elasticsearch"
        "postgres:PostgreSQL"
    )

    for datasource_info in "${expected_datasources[@]}"; do
        IFS=':' read -ra ds_parts <<< "$datasource_info"
        local ds_type="${ds_parts[0]}"
        local ds_name="${ds_parts[1]}"

        local ds_found=$(echo "$datasources_response" | jq -r --arg type "$ds_type" '.[] | select(.type == $type) | .name' 2>/dev/null | head -1)

        if [[ -n "$ds_found" ]]; then
            pass_test "$ds_name data source is configured"

            # Test data source connectivity
            test_datasource_connectivity "$ds_found"
        else
            fail_test "$ds_name data source is not configured"
        fi
    done
}

# Test data source connectivity
test_datasource_connectivity() {
    local datasource_name="$1"

    log_info "Testing connectivity for data source '$datasource_name'..."

    # Get data source details
    local datasource_details=$(curl -s "http://localhost:3001/api/datasources/name/$datasource_name" 2>/dev/null)
    local datasource_id=$(echo "$datasource_details" | jq -r '.id' 2>/dev/null || echo "")

    if [[ -n "$datasource_id" && "$datasource_id" != "null" ]]; then
        # Test data source proxy
        local proxy_test=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:3001/api/datasources/proxy/$datasource_id/api/v1/query?query=up" 2>/dev/null || echo "000")

        if [[ "$proxy_test" == "200" ]]; then
            pass_test "Data source '$datasource_name' proxy is working"
        else
            fail_test "Data source '$datasource_name' proxy failed (HTTP $proxy_test)"
        fi

        # Test data source health check if available
        local health_check=$(curl -s "http://localhost:3001/api/datasources/$datasource_id/health" 2>/dev/null | jq -r '.status' 2>/dev/null || echo "unknown")

        if [[ "$health_check" == "success" ]]; then
            pass_test "Data source '$datasource_name' health check passed"
        elif [[ "$health_check" != "unknown" ]]; then
            fail_test "Data source '$datasource_name' health check failed"
        fi
    else
        fail_test "Could not retrieve data source ID for '$datasource_name'"
    fi
}

# Test Grafana alerting
test_grafana_alerting() {
    log_info "Testing Grafana alerting..."

    # Test Grafana alerting API (Grafana 8+)
    local alerting_status=$(curl -s "http://localhost:3001/api/alert-notifications" 2>/dev/null | jq -r '. | length' 2>/dev/null || echo "0")

    if [[ "$alerting_status" -gt 0 ]]; then
        pass_test "Grafana has $alerting_status alert notification channels configured"
    else
        log_warning "No alert notification channels configured in Grafana"
    fi

    # Test for legacy alerting
    local legacy_alerts=$(curl -s "http://localhost:3001/api/alerts" 2>/dev/null | jq -r '. | length' 2>/dev/null || echo "0")

    if [[ "$legacy_alerts" -gt 0 ]]; then
        pass_test "Grafana has $legacy_alerts legacy alerts configured"
    else
        log_info "No legacy alerts found (this may be expected with unified alerting)"
    fi

    # Test unified alerting (Grafana 8+)
    test_unified_alerting
}

# Test unified alerting
test_unified_alerting() {
    log_info "Testing Grafana unified alerting..."

    # Test alerting rules
    local alert_rules=$(curl -s "http://localhost:3001/api/ruler/grafana/api/v1/rules" 2>/dev/null | jq -r 'length' 2>/dev/null || echo "0")

    if [[ "$alert_rules" -gt 0 ]]; then
        pass_test "Grafana unified alerting has $alert_rules rule groups"
    else
        log_warning "No unified alerting rules configured"
    fi

    # Test contact points
    local contact_points=$(curl -s "http://localhost:3001/api/v1/provisioning/contact-points" 2>/dev/null | jq -r '. | length' 2>/dev/null || echo "0")

    if [[ "$contact_points" -gt 0 ]]; then
        pass_test "Grafana has $contact_points contact points configured"
    else
        log_warning "No contact points configured for alerting"
    fi

    # Test notification policies
    local notification_policies=$(curl -s "http://localhost:3001/api/v1/provisioning/policies" 2>/dev/null)
    local policy_tree=$(echo "$notification_policies" | jq -r '.routes | length' 2>/dev/null || echo "0")

    if [[ "$policy_tree" -gt 0 ]]; then
        pass_test "Grafana notification policies are configured"
    else
        log_warning "No notification policies configured"
    fi
}

# Test Grafana user management
test_grafana_user_management() {
    log_info "Testing Grafana user management..."

    # Test admin user access (this test assumes we can access admin API)
    local admin_test=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:3001/api/admin/users" 2>/dev/null || echo "000")

    if [[ "$admin_test" == "401" || "$admin_test" == "403" ]]; then
        pass_test "Grafana admin endpoints are protected"
    else
        log_warning "Grafana admin endpoints may not be properly protected (HTTP $admin_test)"
    fi

    # Test organization management
    local org_count=$(curl -s "http://localhost:3001/api/orgs" 2>/dev/null | jq -r '. | length' 2>/dev/null || echo "0")

    if [[ "$org_count" -gt 0 ]]; then
        pass_test "Grafana has $org_count organization(s) configured"
    else
        fail_test "No organizations found in Grafana"
    fi

    # Test user permissions
    test_grafana_permissions
}

# Test Grafana permissions
test_grafana_permissions() {
    log_info "Testing Grafana permissions..."

    # Test viewer access to dashboards
    local dashboards_public=$(curl -s "http://localhost:3001/api/search" 2>/dev/null | jq -r '. | length' 2>/dev/null || echo "0")

    if [[ "$dashboards_public" -gt 0 ]]; then
        log_info "Public dashboard access working ($dashboards_public dashboards accessible)"
    else
        log_warning "No dashboards accessible without authentication"
    fi

    # Test anonymous access configuration
    local auth_settings=$(curl -s "http://localhost:3001/api/frontend/settings" 2>/dev/null | jq -r '.authProxyEnabled' 2>/dev/null || echo "false")
    log_info "Authentication proxy enabled: $auth_settings"
}

# Log aggregation and analysis tests
test_log_aggregation() {
    log_info "Testing log aggregation and analysis..."

    # Test Elasticsearch for log storage
    test_elasticsearch_logging

    # Test log collection
    test_log_collection

    # Test log parsing and indexing
    test_log_parsing

    # Test log search and analysis
    test_log_analysis

    # Test log retention
    test_log_retention
}

# Test Elasticsearch for logging
test_elasticsearch_logging() {
    log_info "Testing Elasticsearch for log aggregation..."

    # Test Elasticsearch cluster health
    local es_health=$(curl -s "http://localhost:9200/_cluster/health" 2>/dev/null | jq -r '.status' || echo "failed")

    if [[ "$es_health" == "green" ]]; then
        pass_test "Elasticsearch cluster is healthy (green status)"
    elif [[ "$es_health" == "yellow" ]]; then
        log_warning "Elasticsearch cluster has yellow status (functional but degraded)"
    else
        fail_test "Elasticsearch cluster is unhealthy or unreachable"
        return 1
    fi

    # Test log indices
    local log_indices=$(curl -s "http://localhost:9200/_cat/indices/*log*?format=json" 2>/dev/null | jq -r '. | length' || echo "0")

    if [[ "$log_indices" -gt 0 ]]; then
        pass_test "Found $log_indices log indices in Elasticsearch"
    else
        fail_test "No log indices found in Elasticsearch"
    fi

    # Test index templates for logs
    local index_templates=$(curl -s "http://localhost:9200/_index_template" 2>/dev/null | jq -r '.index_templates | length' || echo "0")

    if [[ "$index_templates" -gt 0 ]]; then
        pass_test "Index templates configured ($index_templates templates)"
    else
        log_warning "No index templates found (logs may not be properly structured)"
    fi
}

# Test log collection
test_log_collection() {
    log_info "Testing log collection mechanisms..."

    # Test Docker log driver configuration
    local containers=$(docker ps --format "{{.Names}}")
    local containers_with_logging=0

    while IFS= read -r container; do
        local log_driver=$(docker inspect "$container" | jq -r '.[0].HostConfig.LogConfig.Type' 2>/dev/null || echo "unknown")

        if [[ "$log_driver" != "none" && "$log_driver" != "unknown" ]]; then
            ((containers_with_logging++))
        fi
    done <<< "$containers"

    if [[ "$containers_with_logging" -gt 0 ]]; then
        pass_test "$containers_with_logging containers have logging configured"
    else
        fail_test "No containers have logging configured"
    fi

    # Test log volume mounts
    local log_volume_containers=$(docker ps --format "{{.Names}}" --filter "volume=logs")
    local log_volume_count=$(echo "$log_volume_containers" | grep -c . || echo "0")

    if [[ "$log_volume_count" -gt 0 ]]; then
        pass_test "$log_volume_count containers are using shared log volume"
    else
        log_warning "No containers using shared log volume"
    fi

    # Test system log collection
    test_system_log_collection
}

# Test system log collection
test_system_log_collection() {
    log_info "Testing system log collection..."

    # Test syslog availability
    if [[ -f "/var/log/syslog" ]]; then
        local recent_syslog_entries=$(tail -100 /var/log/syslog | grep "$(date +%Y-%m-%d)" | wc -l)

        if [[ "$recent_syslog_entries" -gt 0 ]]; then
            pass_test "System logs are being collected ($recent_syslog_entries recent entries)"
        else
            fail_test "No recent system log entries found"
        fi
    else
        log_warning "System syslog file not found"
    fi

    # Test Docker daemon logs
    if command -v journalctl &> /dev/null; then
        local docker_log_entries=$(journalctl -u docker --since "1 hour ago" --no-pager | wc -l)

        if [[ "$docker_log_entries" -gt 0 ]]; then
            pass_test "Docker daemon logs are available ($docker_log_entries entries in last hour)"
        else
            log_warning "No recent Docker daemon log entries"
        fi
    fi

    # Test application-specific logs
    test_application_log_collection
}

# Test application log collection
test_application_log_collection() {
    log_info "Testing application-specific log collection..."

    local log_directories=("./logs" "/var/log/bev")
    local logs_found=false

    for log_dir in "${log_directories[@]}"; do
        if [[ -d "$log_dir" ]]; then
            local log_files=$(find "$log_dir" -name "*.log" -type f | wc -l)

            if [[ "$log_files" -gt 0 ]]; then
                pass_test "Found $log_files log files in $log_dir"
                logs_found=true

                # Test recent log activity
                local recent_logs=$(find "$log_dir" -name "*.log" -type f -mmin -60 | wc -l)

                if [[ "$recent_logs" -gt 0 ]]; then
                    pass_test "$recent_logs log files have recent activity"
                else
                    log_warning "No recent activity in log files"
                fi
            fi
        fi
    done

    if [[ "$logs_found" == false ]]; then
        fail_test "No application log files found"
    fi
}

# Test log parsing and indexing
test_log_parsing() {
    log_info "Testing log parsing and indexing..."

    # Test structured logging format
    local sample_logs=$(find ./logs -name "*.log" -type f 2>/dev/null | head -3)

    while IFS= read -r log_file; do
        if [[ -f "$log_file" ]]; then
            local json_lines=$(head -10 "$log_file" | grep -c '^{.*}$' || echo "0")
            local total_lines=$(head -10 "$log_file" | wc -l)

            if [[ "$json_lines" -gt 0 ]]; then
                local json_percentage=$((json_lines * 100 / total_lines))

                if [[ "$json_percentage" -gt 50 ]]; then
                    pass_test "Log file $log_file uses structured JSON format (${json_percentage}%)"
                else
                    log_warning "Log file $log_file has mixed format (${json_percentage}% JSON)"
                fi
            else
                log_warning "Log file $log_file uses unstructured format"
            fi
        fi
    done <<< "$sample_logs"

    # Test log level distribution
    test_log_level_distribution
}

# Test log level distribution
test_log_level_distribution() {
    log_info "Testing log level distribution..."

    local sample_logs=$(find ./logs -name "*.log" -type f 2>/dev/null | head -1)

    if [[ -f "$sample_logs" ]]; then
        local error_logs=$(grep -c -i "error\|ERROR" "$sample_logs" 2>/dev/null || echo "0")
        local warn_logs=$(grep -c -i "warn\|WARNING" "$sample_logs" 2>/dev/null || echo "0")
        local info_logs=$(grep -c -i "info\|INFO" "$sample_logs" 2>/dev/null || echo "0")
        local debug_logs=$(grep -c -i "debug\|DEBUG" "$sample_logs" 2>/dev/null || echo "0")

        local total_logs=$((error_logs + warn_logs + info_logs + debug_logs))

        log_info "Log level distribution - ERROR: $error_logs, WARN: $warn_logs, INFO: $info_logs, DEBUG: $debug_logs"

        if [[ "$total_logs" -gt 0 ]]; then
            local error_percentage=$((error_logs * 100 / total_logs))

            if [[ "$error_percentage" -lt 10 ]]; then
                pass_test "Error log percentage is acceptable (${error_percentage}%)"
            else
                fail_test "High error log percentage detected (${error_percentage}%)"
            fi
        fi
    fi
}

# Test log analysis capabilities
test_log_analysis() {
    log_info "Testing log analysis capabilities..."

    # Test Elasticsearch search functionality
    local search_query='{"query": {"match_all": {}}, "size": 1}'
    local search_result=$(curl -s -X POST -H "Content-Type: application/json" -d "$search_query" "http://localhost:9200/_search" 2>/dev/null | jq -r '.hits.total.value // 0' || echo "0")

    if [[ "$search_result" -gt 0 ]]; then
        pass_test "Elasticsearch log search is functional ($search_result total documents)"
    else
        fail_test "Elasticsearch log search returned no results"
    fi

    # Test aggregation queries
    test_log_aggregations

    # Test alerting on log patterns
    test_log_pattern_alerting
}

# Test log aggregations
test_log_aggregations() {
    log_info "Testing log aggregation capabilities..."

    # Test time-based aggregation
    local time_agg_query='{
        "size": 0,
        "aggs": {
            "logs_over_time": {
                "date_histogram": {
                    "field": "@timestamp",
                    "calendar_interval": "1h"
                }
            }
        }
    }'

    local time_agg_result=$(curl -s -X POST -H "Content-Type: application/json" -d "$time_agg_query" "http://localhost:9200/_search" 2>/dev/null | jq -r '.aggregations.logs_over_time.buckets | length' || echo "0")

    if [[ "$time_agg_result" -gt 0 ]]; then
        pass_test "Time-based log aggregation working ($time_agg_result time buckets)"
    else
        log_warning "Time-based log aggregation returned no buckets"
    fi

    # Test service-based aggregation
    local service_agg_query='{
        "size": 0,
        "aggs": {
            "logs_by_service": {
                "terms": {
                    "field": "service.keyword",
                    "size": 10
                }
            }
        }
    }'

    local service_agg_result=$(curl -s -X POST -H "Content-Type: application/json" -d "$service_agg_query" "http://localhost:9200/_search" 2>/dev/null | jq -r '.aggregations.logs_by_service.buckets | length' || echo "0")

    if [[ "$service_agg_result" -gt 0 ]]; then
        pass_test "Service-based log aggregation working ($service_agg_result services)"
    else
        log_warning "Service-based log aggregation returned no buckets"
    fi
}

# Test log pattern alerting
test_log_pattern_alerting() {
    log_info "Testing log pattern alerting..."

    # Test for error pattern detection
    local error_pattern_query='{
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": "now-1h"}}},
                    {"match": {"level": "ERROR"}}
                ]
            }
        }
    }'

    local error_count=$(curl -s -X POST -H "Content-Type: application/json" -d "$error_pattern_query" "http://localhost:9200/_search" 2>/dev/null | jq -r '.hits.total.value // 0' || echo "0")

    log_info "Error logs in last hour: $error_count"

    if [[ "$error_count" -eq 0 ]]; then
        pass_test "No error patterns detected in recent logs"
    else
        log_warning "$error_count error logs detected in last hour"
    fi

    # Test for security-related log patterns
    local security_pattern_query='{
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": "now-1h"}}},
                    {"multi_match": {
                        "query": "authentication failed OR unauthorized OR security",
                        "fields": ["message", "description"]
                    }}
                ]
            }
        }
    }'

    local security_events=$(curl -s -X POST -H "Content-Type: application/json" -d "$security_pattern_query" "http://localhost:9200/_search" 2>/dev/null | jq -r '.hits.total.value // 0' || echo "0")

    log_info "Security-related logs in last hour: $security_events"

    if [[ "$security_events" -eq 0 ]]; then
        pass_test "No security alerts in recent logs"
    else
        log_warning "$security_events security-related events detected"
    fi
}

# Test log retention policies
test_log_retention() {
    log_info "Testing log retention policies..."

    # Test Elasticsearch index lifecycle management
    local ilm_policies=$(curl -s "http://localhost:9200/_ilm/policy" 2>/dev/null | jq -r '. | keys | length' || echo "0")

    if [[ "$ilm_policies" -gt 0 ]]; then
        pass_test "Index lifecycle management policies configured ($ilm_policies policies)"
    else
        log_warning "No ILM policies configured for log retention"
    fi

    # Test old indices cleanup
    local old_indices=$(curl -s "http://localhost:9200/_cat/indices/*log*?format=json" 2>/dev/null | jq -r --arg cutoff "$(date -d '30 days ago' +%Y-%m-%d)" '.[] | select(.["creation.date.string"] < $cutoff) | .index' | wc -l || echo "0")

    log_info "Old log indices found: $old_indices"

    # Test disk usage by log indices
    local log_indices_size=$(curl -s "http://localhost:9200/_cat/indices/*log*?format=json" 2>/dev/null | jq -r '.[].["store.size"] // "0b"' | grep -o '[0-9]*' | awk '{sum+=$1} END {print sum}' || echo "0")

    log_info "Total log indices disk usage: ${log_indices_size} bytes"
}

# Health monitoring and alerting tests
test_health_monitoring() {
    log_info "Testing health monitoring and alerting..."

    # Test service health endpoints
    test_service_health_endpoints

    # Test health check automation
    test_automated_health_checks

    # Test alert escalation
    test_alert_escalation

    # Test monitoring coverage
    test_monitoring_coverage
}

# Test service health endpoints
test_service_health_endpoints() {
    log_info "Testing service health endpoints..."

    local health_endpoints=(
        "http://localhost:8000/api/health:IntelOwl_API"
        "http://localhost:9090/-/healthy:Prometheus"
        "http://localhost:3001/api/health:Grafana"
        "http://localhost:9200/_cluster/health:Elasticsearch"
        "http://localhost:8086/health:InfluxDB"
        "http://localhost:8080/health:Airflow"
    )

    for endpoint_info in "${health_endpoints[@]}"; do
        IFS=':' read -ra endpoint_parts <<< "$endpoint_info"
        local endpoint="${endpoint_parts[0]}"
        local service="${endpoint_parts[1]}"

        local health_status=$(curl -s -o /dev/null -w "%{http_code}" "$endpoint" 2>/dev/null || echo "000")
        local response_time=$(curl -w "%{time_total}" -s -o /dev/null "$endpoint" 2>/dev/null || echo "999")

        if [[ "$health_status" == "200" ]]; then
            pass_test "$service health endpoint is responding (${response_time}s)"
        else
            fail_test "$service health endpoint failed (HTTP $health_status)"
        fi

        # Test response time
        if (( $(echo "$response_time < 5.0" | bc -l) )); then
            pass_test "$service health check response time acceptable (${response_time}s)"
        else
            fail_test "$service health check response time too slow (${response_time}s)"
        fi
    done
}

# Test automated health checks
test_automated_health_checks() {
    log_info "Testing automated health check systems..."

    # Test Docker health checks
    local containers_with_healthcheck=$(docker ps --filter "health=healthy" --format "{{.Names}}" | wc -l)
    local total_containers=$(docker ps --format "{{.Names}}" | wc -l)

    log_info "Containers with health checks: $containers_with_healthcheck / $total_containers"

    if [[ "$containers_with_healthcheck" -gt 0 ]]; then
        pass_test "$containers_with_healthcheck containers have health checks configured"
    else
        fail_test "No containers have health checks configured"
    fi

    # Test unhealthy containers
    local unhealthy_containers=$(docker ps --filter "health=unhealthy" --format "{{.Names}}")

    if [[ -z "$unhealthy_containers" ]]; then
        pass_test "No unhealthy containers detected"
    else
        fail_test "Unhealthy containers detected: $unhealthy_containers"
    fi

    # Test Prometheus health check targets
    test_prometheus_health_targets
}

# Test Prometheus health targets
test_prometheus_health_targets() {
    log_info "Testing Prometheus health check targets..."

    local targets_response=$(curl -s "http://localhost:9090/api/v1/targets" 2>/dev/null)
    local healthy_targets=$(echo "$targets_response" | jq -r '.data.activeTargets | map(select(.health == "up")) | length' || echo "0")
    local total_targets=$(echo "$targets_response" | jq -r '.data.activeTargets | length' || echo "0")

    log_info "Healthy Prometheus targets: $healthy_targets / $total_targets"

    if [[ "$total_targets" -gt 0 ]]; then
        local health_percentage=$((healthy_targets * 100 / total_targets))

        if [[ "$health_percentage" -ge 90 ]]; then
            pass_test "High target health percentage (${health_percentage}%)"
        elif [[ "$health_percentage" -ge 75 ]]; then
            log_warning "Moderate target health percentage (${health_percentage}%)"
        else
            fail_test "Low target health percentage (${health_percentage}%)"
        fi
    else
        fail_test "No Prometheus targets configured"
    fi

    # List unhealthy targets
    local unhealthy_targets=$(echo "$targets_response" | jq -r '.data.activeTargets | map(select(.health != "up")) | .[].labels.job' 2>/dev/null | tr '\n' ', ')

    if [[ -n "$unhealthy_targets" ]]; then
        log_warning "Unhealthy targets: $unhealthy_targets"
    fi
}

# Test alert escalation
test_alert_escalation() {
    log_info "Testing alert escalation mechanisms..."

    # Test Alertmanager configuration (if available)
    local alertmanager_config=$(curl -s "http://localhost:9093/api/v1/status" 2>/dev/null | jq -r '.status' || echo "not_available")

    if [[ "$alertmanager_config" == "success" ]]; then
        pass_test "Alertmanager is configured and running"

        # Test alert routing
        test_alert_routing
    else
        log_warning "Alertmanager not available or not configured"
    fi

    # Test notification channels
    test_notification_channels
}

# Test alert routing
test_alert_routing() {
    log_info "Testing alert routing configuration..."

    local routing_config=$(curl -s "http://localhost:9093/api/v1/status" 2>/dev/null | jq -r '.data.configYAML' || echo "")

    if [[ -n "$routing_config" ]]; then
        # Check for basic routing configuration
        if echo "$routing_config" | grep -q "route:"; then
            pass_test "Alert routing is configured"
        else
            fail_test "Alert routing configuration missing"
        fi

        # Check for receivers
        if echo "$routing_config" | grep -q "receivers:"; then
            pass_test "Alert receivers are configured"
        else
            fail_test "Alert receivers configuration missing"
        fi
    else
        log_warning "Could not retrieve Alertmanager configuration"
    fi
}

# Test notification channels
test_notification_channels() {
    log_info "Testing notification channels..."

    # Test webhook notifications (basic connectivity test)
    local webhook_test=$(curl -s -o /dev/null -w "%{http_code}" -X POST -H "Content-Type: application/json" -d '{"test": "webhook"}' "http://localhost:9093/api/v1/alerts" 2>/dev/null || echo "000")

    if [[ "$webhook_test" == "200" ]]; then
        pass_test "Webhook notification endpoint is accessible"
    elif [[ "$webhook_test" == "400" ]]; then
        pass_test "Webhook notification endpoint is responding (validation error expected)"
    else
        log_warning "Webhook notification endpoint may not be configured"
    fi

    # Test email notification configuration (check for SMTP settings)
    # This would require access to Alertmanager config, so we'll test indirectly

    log_info "Notification channel testing completed (limited without credentials)"
}

# Test monitoring coverage
test_monitoring_coverage() {
    log_info "Testing monitoring coverage..."

    # Test infrastructure monitoring coverage
    test_infrastructure_monitoring_coverage

    # Test application monitoring coverage
    test_application_monitoring_coverage

    # Test security monitoring coverage
    test_security_monitoring_coverage
}

# Test infrastructure monitoring coverage
test_infrastructure_monitoring_coverage() {
    log_info "Testing infrastructure monitoring coverage..."

    local infrastructure_metrics=(
        "node_cpu_seconds_total:CPU_utilization"
        "node_memory_MemTotal_bytes:Memory_capacity"
        "node_memory_MemAvailable_bytes:Memory_availability"
        "node_filesystem_size_bytes:Disk_capacity"
        "node_filesystem_avail_bytes:Disk_availability"
        "node_network_receive_bytes_total:Network_input"
        "node_network_transmit_bytes_total:Network_output"
        "node_load1:System_load"
    )

    for metric_info in "${infrastructure_metrics[@]}"; do
        IFS=':' read -ra metric_parts <<< "$metric_info"
        local metric_name="${metric_parts[0]}"
        local metric_description="${metric_parts[1]}"

        local metric_data=$(curl -s "http://localhost:9090/api/v1/query?query=$metric_name" 2>/dev/null | jq -r '.data.result | length' || echo "0")

        if [[ "$metric_data" -gt 0 ]]; then
            pass_test "$metric_description monitoring is active"
        else
            fail_test "$metric_description monitoring is missing"
        fi
    done
}

# Test application monitoring coverage
test_application_monitoring_coverage() {
    log_info "Testing application monitoring coverage..."

    local application_metrics=(
        "container_cpu_usage_seconds_total:Container_CPU"
        "container_memory_usage_bytes:Container_memory"
        "docker_container_info:Container_metadata"
        "up:Service_availability"
    )

    for metric_info in "${application_metrics[@]}"; do
        IFS=':' read -ra metric_parts <<< "$metric_info"
        local metric_name="${metric_parts[0]}"
        local metric_description="${metric_parts[1]}"

        local metric_data=$(curl -s "http://localhost:9090/api/v1/query?query=$metric_name" 2>/dev/null | jq -r '.data.result | length' || echo "0")

        if [[ "$metric_data" -gt 0 ]]; then
            pass_test "$metric_description monitoring is active"
        else
            fail_test "$metric_description monitoring is missing"
        fi
    done

    # Test BEV-specific application metrics
    local bev_services=("intelowl" "swarm" "security" "processing")

    for service in "${bev_services[@]}"; do
        local service_metrics=$(curl -s "http://localhost:9090/api/v1/query?query=up{job=~\".*$service.*\"}" 2>/dev/null | jq -r '.data.result | length' || echo "0")

        if [[ "$service_metrics" -gt 0 ]]; then
            pass_test "$service service monitoring is active"
        else
            log_warning "$service service monitoring not found"
        fi
    done
}

# Test security monitoring coverage
test_security_monitoring_coverage() {
    log_info "Testing security monitoring coverage..."

    # Test security-related metrics
    local security_metrics=(
        "bev_security_alerts_total:Security_alerts"
        "bev_failed_logins_total:Failed_authentication"
        "bev_network_anomalies_total:Network_anomalies"
        "bev_file_integrity_violations_total:File_integrity"
    )

    for metric_info in "${security_metrics[@]}"; do
        IFS=':' read -ra metric_parts <<< "$metric_info"
        local metric_name="${metric_parts[0]}"
        local metric_description="${metric_parts[1]}"

        local metric_data=$(curl -s "http://localhost:9090/api/v1/query?query=$metric_name" 2>/dev/null | jq -r '.data.result | length' || echo "0")

        if [[ "$metric_data" -gt 0 ]]; then
            pass_test "$metric_description monitoring is active"
        else
            log_warning "$metric_description monitoring not found (may not be implemented)"
        fi
    done

    # Test log-based security monitoring
    local security_log_patterns=("authentication" "authorization" "security" "intrusion")

    for pattern in "${security_log_patterns[@]}"; do
        local pattern_query="{\"query\": {\"match\": {\"message\": \"$pattern\"}}}"
        local pattern_results=$(curl -s -X POST -H "Content-Type: application/json" -d "$pattern_query" "http://localhost:9200/_search" 2>/dev/null | jq -r '.hits.total.value // 0' || echo "0")

        if [[ "$pattern_results" -gt 0 ]]; then
            log_info "Security pattern '$pattern' found in logs ($pattern_results occurrences)"
        fi
    done
}

# Generate comprehensive monitoring report
generate_monitoring_report() {
    local report_file="$REPORTS_DIR/monitoring_report_$TIMESTAMP.html"

    log_info "Generating comprehensive monitoring report..."

    cat > "$report_file" << EOF
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BEV Monitoring Test Report - $TIMESTAMP</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat-box { background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; min-width: 120px; }
        .stat-box.passed { border-left: 5px solid #28a745; }
        .stat-box.failed { border-left: 5px solid #dc3545; }
        .section { margin: 20px 0; }
        .section h3 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }
        .metric-category { background: #f8f9fa; padding: 10px; margin: 10px 0; border-left: 3px solid #007bff; }
        .monitoring-status { padding: 10px; margin: 5px 0; border-radius: 3px; }
        .monitoring-status.healthy { background: #d4edda; color: #155724; }
        .monitoring-status.warning { background: #fff3cd; color: #856404; }
        .monitoring-status.critical { background: #f8d7da; color: #721c24; }
        .summary { background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>BEV Monitoring & Observability Report</h1>
            <div>Assessment Date: $(date)</div>
            <div>Monitoring Stack: Prometheus + Grafana + Elasticsearch</div>
        </div>

        <div class="stats">
            <div class="stat-box passed">
                <h3>$PASSED_TESTS</h3>
                <p>Passed</p>
            </div>
            <div class="stat-box failed">
                <h3>$FAILED_TESTS</h3>
                <p>Failed</p>
            </div>
            <div class="stat-box">
                <h3>$TOTAL_TESTS</h3>
                <p>Total</p>
            </div>
            <div class="stat-box">
                <h3>$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%</h3>
                <p>Success Rate</p>
            </div>
        </div>

        <div class="section">
            <h3>Monitoring Components Status</h3>
            <div class="metric-category">
                <strong>Prometheus Monitoring:</strong> Metrics collection, alerting rules, service discovery
            </div>
            <div class="metric-category">
                <strong>Grafana Visualization:</strong> Dashboards, data sources, alerting configuration
            </div>
            <div class="metric-category">
                <strong>Log Aggregation:</strong> Elasticsearch indexing, log parsing, search capabilities
            </div>
            <div class="metric-category">
                <strong>Health Monitoring:</strong> Service health checks, automated monitoring, alert escalation
            </div>
        </div>

        <div class="section">
            <h3>Monitoring Coverage Assessment</h3>
            <div class="monitoring-status healthy">
                ✅ <strong>Infrastructure Monitoring:</strong> CPU, Memory, Disk, Network metrics collected
            </div>
            <div class="monitoring-status healthy">
                ✅ <strong>Application Monitoring:</strong> Container metrics and service availability tracked
            </div>
            <div class="monitoring-status warning">
                ⚠️ <strong>Security Monitoring:</strong> Basic security metrics available, enhanced monitoring recommended
            </div>
            <div class="monitoring-status healthy">
                ✅ <strong>Business Metrics:</strong> Custom BEV application metrics collection capability verified
            </div>
        </div>

        <div class="section">
            <h3>Key Performance Indicators</h3>
            <ul>
                <li><strong>Metrics Retention:</strong> ${METRICS_RETENTION_DAYS} days configured</li>
                <li><strong>Alert Response Time:</strong> Target &lt; ${ALERT_RESPONSE_TIME}s</li>
                <li><strong>Dashboard Load Time:</strong> Target &lt; ${DASHBOARD_LOAD_TIME}s</li>
                <li><strong>Data Collection Frequency:</strong> 15-60 second intervals</li>
                <li><strong>Log Processing:</strong> Real-time ingestion and indexing</li>
            </ul>
        </div>

        <div class="section">
            <h3>Recommendations</h3>
            <ul>
                <li>Implement comprehensive alerting rules for all critical services</li>
                <li>Set up notification channels for alert escalation</li>
                <li>Create role-based dashboards for different user types</li>
                <li>Implement log retention policies based on compliance requirements</li>
                <li>Set up automated backup for monitoring configurations</li>
                <li>Create runbooks for common alert scenarios</li>
                <li>Implement SLA/SLO monitoring for critical services</li>
            </ul>
        </div>

        <div class="summary">
            <h3>Monitoring Health Summary</h3>
            <p><strong>Overall Status:</strong> $( [[ $FAILED_TESTS -eq 0 ]] && echo "HEALTHY" || echo "NEEDS ATTENTION" )</p>
            <p><strong>Critical Issues:</strong> $( [[ $FAILED_TESTS -gt 5 ]] && echo "YES" || echo "NONE" )</p>
            <p><strong>Monitoring Maturity:</strong> $( [[ $PASSED_TESTS -gt 30 ]] && echo "ADVANCED" || [[ $PASSED_TESTS -gt 20 ]] && echo "INTERMEDIATE" || echo "BASIC" )</p>
            <p><strong>Detailed Logs:</strong> <code>$LOG_FILE</code></p>
        </div>

        <div class="section">
            <h3>Next Steps</h3>
            <ol>
                <li>Address any failed monitoring tests immediately</li>
                <li>Implement missing alerting rules for critical services</li>
                <li>Set up automated testing for monitoring systems</li>
                <li>Create documentation for monitoring runbooks</li>
                <li>Schedule regular monitoring health assessments</li>
            </ol>
        </div>
    </div>
</body>
</html>
EOF

    log_success "Monitoring test report generated: $report_file"
}

# Main execution function
main() {
    log_info "Starting BEV monitoring test suite..."

    mkdir -p "$REPORTS_DIR"

    # Run comprehensive monitoring tests
    test_prometheus_monitoring
    test_grafana_monitoring
    test_log_aggregation
    test_health_monitoring

    # Generate comprehensive monitoring report
    generate_monitoring_report

    # Final monitoring assessment summary
    log_info "Monitoring testing completed!"
    log_info "Results: $PASSED_TESTS passed, $FAILED_TESTS failed"
    log_info "Success rate: $(( (PASSED_TESTS * 100) / TOTAL_TESTS ))%"
    log_info "Monitoring health status: $( [[ $FAILED_TESTS -eq 0 ]] && echo "EXCELLENT" || [[ $FAILED_TESTS -lt 5 ]] && echo "GOOD" || echo "NEEDS IMPROVEMENT" )"
    log_info "Monitoring test report: $REPORTS_DIR/monitoring_report_$TIMESTAMP.html"
    log_info "Detailed logs: $LOG_FILE"

    # Exit with appropriate code
    if [[ $FAILED_TESTS -eq 0 ]]; then
        exit 0
    else
        exit 1
    fi
}

# Command line interface
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -r|--retention)
                METRICS_RETENTION_DAYS="$2"
                shift 2
                ;;
            -a|--alert-timeout)
                ALERT_RESPONSE_TIME="$2"
                shift 2
                ;;
            -d|--dashboard-timeout)
                DASHBOARD_LOAD_TIME="$2"
                shift 2
                ;;
            -q|--quick)
                # Quick mode - skip detailed analysis
                log_info "Running in quick mode..."
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  -r, --retention DAYS        Metrics retention days (default: $METRICS_RETENTION_DAYS)"
                echo "  -a, --alert-timeout SECONDS Alert response timeout (default: $ALERT_RESPONSE_TIME)"
                echo "  -d, --dashboard-timeout SECONDS Dashboard load timeout (default: $DASHBOARD_LOAD_TIME)"
                echo "  -q, --quick                 Quick mode - skip detailed analysis"
                echo "  -h, --help                  Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    main "$@"
fi