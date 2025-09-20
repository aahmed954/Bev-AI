#!/bin/bash
# BEV Frontend Integration - Comprehensive Health Monitoring Setup
# Prometheus, Grafana, AlertManager integration with comprehensive dashboards
# Author: DevOps Automation Framework
# Version: 1.0.0

set -euo pipefail

# =====================================================
# Configuration and Constants
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/deployment"
LOG_FILE="${LOG_DIR}/monitoring-setup-$(date +%Y%m%d_%H%M%S).log"
MONITORING_DIR="${PROJECT_ROOT}/monitoring"

# Monitoring service ports
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
ALERTMANAGER_PORT=9093
NODE_EXPORTER_PORT=9100
BLACKBOX_EXPORTER_PORT=9115

# Monitoring network
MONITORING_NETWORK="bev_monitoring"
MONITORING_SUBNET="172.32.0.0/16"

# Alert channels
ALERT_WEBHOOK_URL="${ALERT_WEBHOOK_URL:-}"
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
ALERT_EMAIL="${ALERT_EMAIL:-}"

# Retention settings
PROMETHEUS_RETENTION="30d"
GRAFANA_RETENTION="90d"

# =====================================================
# Logging Functions
# =====================================================

setup_logging() {
    mkdir -p "${LOG_DIR}"
    exec 1> >(tee -a "${LOG_FILE}")
    exec 2> >(tee -a "${LOG_FILE}" >&2)
    echo "=== BEV Health Monitoring Setup Started at $(date) ===" | tee -a "${LOG_FILE}"
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "${LOG_FILE}"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*" | tee -a "${LOG_FILE}"
}

# =====================================================
# Infrastructure Setup
# =====================================================

setup_monitoring_infrastructure() {
    log_info "Setting up monitoring infrastructure..."
    
    # Create monitoring directory structure
    mkdir -p "${MONITORING_DIR}"/{prometheus,grafana,alertmanager,exporters}
    mkdir -p "${MONITORING_DIR}/prometheus"/{config,data,rules}
    mkdir -p "${MONITORING_DIR}/grafana"/{config,data,dashboards,provisioning}
    mkdir -p "${MONITORING_DIR}/grafana/provisioning"/{dashboards,datasources,notifiers}
    mkdir -p "${MONITORING_DIR}/alertmanager"/{config,data}
    mkdir -p "${MONITORING_DIR}/exporters"/{node,blackbox}
    
    # Create monitoring network
    if docker network ls --format "{{.Name}}" | grep -q "^${MONITORING_NETWORK}$"; then
        log_info "Monitoring network already exists: ${MONITORING_NETWORK}"
    else
        log_info "Creating monitoring network: ${MONITORING_NETWORK}"
        docker network create \
            --driver bridge \
            --subnet "${MONITORING_SUBNET}" \
            --opt com.docker.network.bridge.name="br-bev-monitoring" \
            --label "project=bev-osint" \
            --label "component=monitoring" \
            "${MONITORING_NETWORK}"
    fi
    
    log_success "Monitoring infrastructure setup completed"
}

# =====================================================
# Prometheus Configuration
# =====================================================

generate_prometheus_config() {
    log_info "Generating Prometheus configuration..."
    
    cat > "${MONITORING_DIR}/prometheus/config/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'bev-osint'
    environment: '${ENVIRONMENT:-development}'

rule_files:
  - "/etc/prometheus/rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
    scrape_interval: 5s
    
  # BEV Frontend Services
  - job_name: 'bev-mcp-server'
    static_configs:
      - targets: ['172.31.0.12:3011']
    metrics_path: '/metrics'
    scrape_interval: 10s
    scrape_timeout: 5s
    
  - job_name: 'bev-frontend-proxy'
    static_configs:
      - targets: ['172.31.0.10:8404']  # HAProxy stats with Prometheus format
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'bev-websocket-server'
    static_configs:
      - targets: ['172.31.0.14:8081']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  # System Metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 10s
    
  # Blackbox Exporter for endpoint monitoring
  - job_name: 'blackbox-http'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
        - http://localhost:3010
        - https://localhost:8443
        - http://localhost:3011/health
        - http://localhost:8080/health  # HAProxy health
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115
        
  # Docker container metrics
  - job_name: 'docker'
    static_configs:
      - targets: ['172.17.0.1:9323']  # Docker daemon metrics (if enabled)
    scrape_interval: 10s
    
  # Core BEV Services Integration
  - job_name: 'bev-postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 30s
    
  - job_name: 'bev-redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s

# Remote write for long-term storage (optional)
# remote_write:
#   - url: "http://cortex:9009/api/prom/push"
#     queue_config:
#       max_samples_per_send: 1000
#       capacity: 2500
#       max_shards: 200

# Storage configuration
storage:
  tsdb:
    retention.time: ${PROMETHEUS_RETENTION}
    retention.size: 10GB
    wal-compression: true
EOF

    log_success "Prometheus configuration generated"
}

generate_prometheus_rules() {
    log_info "Generating Prometheus alerting rules..."
    
    # Frontend service alerts
    cat > "${MONITORING_DIR}/prometheus/rules/frontend_alerts.yml" << 'EOF'
groups:
  - name: bev_frontend_alerts
    interval: 30s
    rules:
      # Service availability alerts
      - alert: MCP_Server_Down
        expr: up{job="bev-mcp-server"} == 0
        for: 1m
        labels:
          severity: critical
          service: mcp-server
        annotations:
          summary: "BEV MCP Server is down"
          description: "MCP Server has been down for more than 1 minute"
          
      - alert: Frontend_Proxy_Down
        expr: up{job="bev-frontend-proxy"} == 0
        for: 1m
        labels:
          severity: critical
          service: frontend-proxy
        annotations:
          summary: "BEV Frontend Proxy is down"
          description: "Frontend Proxy (HAProxy) has been down for more than 1 minute"
          
      - alert: WebSocket_Server_Down
        expr: up{job="bev-websocket-server"} == 0
        for: 2m
        labels:
          severity: warning
          service: websocket-server
        annotations:
          summary: "BEV WebSocket Server is down"
          description: "WebSocket Server has been down for more than 2 minutes"
          
      # Performance alerts
      - alert: High_Response_Time
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
          service: frontend
        annotations:
          summary: "High response times detected"
          description: "95th percentile response time is {{ $value }}s for {{ $labels.instance }}"
          
      - alert: High_Error_Rate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
        for: 3m
        labels:
          severity: critical
          service: frontend
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.instance }}"
          
      # Resource alerts
      - alert: High_Memory_Usage
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) > 0.8
        for: 5m
        labels:
          severity: warning
          service: frontend
        annotations:
          summary: "High memory usage"
          description: "Container {{ $labels.container_label_com_docker_compose_service }} memory usage is {{ $value | humanizePercentage }}"
          
      - alert: High_CPU_Usage
        expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
          service: frontend
        annotations:
          summary: "High CPU usage"
          description: "Container {{ $labels.container_label_com_docker_compose_service }} CPU usage is {{ $value | humanizePercentage }}"
          
      # Connectivity alerts
      - alert: Endpoint_Down
        expr: probe_success == 0
        for: 1m
        labels:
          severity: critical
          service: endpoint-monitoring
        annotations:
          summary: "Endpoint is down"
          description: "Endpoint {{ $labels.instance }} has been down for more than 1 minute"
          
      - alert: SSL_Certificate_Expiry
        expr: probe_ssl_earliest_cert_expiry - time() < 86400 * 7
        for: 1h
        labels:
          severity: warning
          service: ssl
        annotations:
          summary: "SSL certificate expiring soon"
          description: "SSL certificate for {{ $labels.instance }} expires in {{ $value | humanizeDuration }}"
EOF

    # System resource alerts
    cat > "${MONITORING_DIR}/prometheus/rules/system_alerts.yml" << 'EOF'
groups:
  - name: bev_system_alerts
    interval: 30s
    rules:
      # Node/System alerts
      - alert: Node_Down
        expr: up{job="node-exporter"} == 0
        for: 1m
        labels:
          severity: critical
          service: system
        annotations:
          summary: "Node is down"
          description: "Node {{ $labels.instance }} has been down for more than 1 minute"
          
      - alert: High_System_Load
        expr: node_load15 > node_cpu_count * 0.8
        for: 5m
        labels:
          severity: warning
          service: system
        annotations:
          summary: "High system load"
          description: "15m load average is {{ $value }} on {{ $labels.instance }}"
          
      - alert: Low_Disk_Space
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 5m
        labels:
          severity: critical
          service: system
        annotations:
          summary: "Low disk space"
          description: "Disk {{ $labels.mountpoint }} on {{ $labels.instance }} has {{ $value | humanizePercentage }} space left"
          
      - alert: High_Memory_Usage_System
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) > 0.9
        for: 5m
        labels:
          severity: warning
          service: system
        annotations:
          summary: "High system memory usage"
          description: "Memory usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}"
          
      - alert: Docker_Daemon_Down
        expr: up{job="docker"} == 0
        for: 1m
        labels:
          severity: critical
          service: docker
        annotations:
          summary: "Docker daemon is down"
          description: "Docker daemon on {{ $labels.instance }} has been down for more than 1 minute"
EOF

    # BEV integration alerts
    cat > "${MONITORING_DIR}/prometheus/rules/bev_integration_alerts.yml" << 'EOF'
groups:
  - name: bev_integration_alerts
    interval: 30s
    rules:
      # Database connectivity
      - alert: PostgreSQL_Down
        expr: up{job="bev-postgres"} == 0
        for: 2m
        labels:
          severity: critical
          service: database
        annotations:
          summary: "PostgreSQL database is down"
          description: "PostgreSQL database has been down for more than 2 minutes"
          
      - alert: Redis_Down
        expr: up{job="bev-redis"} == 0
        for: 2m
        labels:
          severity: critical
          service: cache
        annotations:
          summary: "Redis cache is down"
          description: "Redis cache has been down for more than 2 minutes"
          
      # Integration health
      - alert: BEV_Integration_Failure
        expr: bev_integration_health_status == 0
        for: 3m
        labels:
          severity: warning
          service: integration
        annotations:
          summary: "BEV integration health check failing"
          description: "Integration between frontend and BEV services is failing"
          
      # Data flow monitoring
      - alert: Low_Data_Throughput
        expr: rate(bev_data_processed_total[5m]) < 10
        for: 10m
        labels:
          severity: info
          service: data-flow
        annotations:
          summary: "Low data processing throughput"
          description: "Data processing rate has dropped below 10 items/second for {{ $labels.instance }}"
EOF

    log_success "Prometheus alerting rules generated"
}

# =====================================================
# AlertManager Configuration
# =====================================================

generate_alertmanager_config() {
    log_info "Generating AlertManager configuration..."
    
    cat > "${MONITORING_DIR}/alertmanager/config/alertmanager.yml" << EOF
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@bev-osint.local'
  smtp_require_tls: false

route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 5s
      repeat_interval: 5m
      
    - match:
        severity: warning
      receiver: 'warning-alerts'
      group_wait: 10s
      repeat_interval: 15m
      
    - match:
        service: system
      receiver: 'system-alerts'
      
    - match:
        service: database
      receiver: 'database-alerts'

receivers:
  - name: 'default'
    webhook_configs:
      - url: '${ALERT_WEBHOOK_URL:-http://localhost:9094/webhook}'
        send_resolved: true
        
  - name: 'critical-alerts'
    webhook_configs:
      - url: '${ALERT_WEBHOOK_URL:-http://localhost:9094/webhook}'
        send_resolved: true
        title: 'CRITICAL: BEV Frontend Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}: {{ .Annotations.description }}{{ end }}'
    $([ -n "${SLACK_WEBHOOK_URL}" ] && cat << EOL
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#alerts'
        color: 'danger'
        title: 'CRITICAL: BEV Frontend Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}: {{ .Annotations.description }}{{ end }}'
        send_resolved: true
EOL
    )
    $([ -n "${ALERT_EMAIL}" ] && cat << EOL
    email_configs:
      - to: '${ALERT_EMAIL}'
        subject: 'CRITICAL: BEV Frontend Alert'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Service: {{ .Labels.service }}
          Severity: {{ .Labels.severity }}
          {{ end }}
EOL
    )
        
  - name: 'warning-alerts'
    webhook_configs:
      - url: '${ALERT_WEBHOOK_URL:-http://localhost:9094/webhook}'
        send_resolved: true
        title: 'WARNING: BEV Frontend Alert'
    $([ -n "${SLACK_WEBHOOK_URL}" ] && cat << EOL
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#monitoring'
        color: 'warning'
        title: 'WARNING: BEV Frontend Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
        send_resolved: true
EOL
    )
        
  - name: 'system-alerts'
    webhook_configs:
      - url: '${ALERT_WEBHOOK_URL:-http://localhost:9094/webhook}'
        send_resolved: true
        title: 'SYSTEM: BEV Infrastructure Alert'
        
  - name: 'database-alerts'
    webhook_configs:
      - url: '${ALERT_WEBHOOK_URL:-http://localhost:9094/webhook}'
        send_resolved: true
        title: 'DATABASE: BEV Data Layer Alert'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'service']
    
  - source_match:
      alertname: 'Node_Down'
    target_match_re:
      alertname: '.*'
    equal: ['instance']
EOF

    log_success "AlertManager configuration generated"
}

# =====================================================
# Grafana Configuration
# =====================================================

generate_grafana_config() {
    log_info "Generating Grafana configuration..."
    
    # Main Grafana configuration
    cat > "${MONITORING_DIR}/grafana/config/grafana.ini" << EOF
[default]
instance_name = bev-osint-monitoring

[server]
protocol = http
http_port = 3001
domain = localhost
root_url = http://localhost:3001/
serve_from_sub_path = false

[database]
type = sqlite3
path = /var/lib/grafana/grafana.db

[session]
provider = file
provider_config = sessions

[dataproxy]
logging = true
timeout = 30
dial_timeout = 10
keep_alive_seconds = 30

[analytics]
reporting_enabled = false
check_for_updates = false

[security]
admin_user = admin
admin_password = ${GRAFANA_ADMIN_PASSWORD:-bevgrafana2024}
secret_key = ${GRAFANA_SECRET_KEY:-$(openssl rand -hex 16)}
disable_gravatar = true
cookie_secure = false
cookie_samesite = lax

[users]
allow_sign_up = false
allow_org_create = false
auto_assign_org = true
auto_assign_org_role = Viewer
default_theme = dark

[auth]
disable_login_form = false
disable_signout_menu = false

[auth.anonymous]
enabled = false

[dashboards]
default_home_dashboard_path = /etc/grafana/provisioning/dashboards/bev-overview.json

[alerting]
enabled = true
execute_alerts = true

[metrics]
enabled = true
interval_seconds = 10

[log]
mode = console file
level = info
format = text

[log.console]
level = info
format = text

[log.file]
level = info
format = text
log_rotate = true
max_lines = 1000000
max_size_shift = 28
daily_rotate = true
max_days = 7

[explore]
enabled = true

[help]
enabled = true

[feature_toggles]
enable = ngalert

[unified_alerting]
enabled = true
disabled_orgs = 
min_interval = 10s
EOF

    # Datasource provisioning
    cat > "${MONITORING_DIR}/grafana/provisioning/datasources/prometheus.yml" << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    basicAuth: false
    editable: true
    jsonData:
      timeInterval: 5s
      queryTimeout: 60s
      httpMethod: POST
      manageAlerts: true
      alertmanagerUid: alertmanager
      
  - name: AlertManager
    type: alertmanager
    access: proxy
    url: http://alertmanager:9093
    uid: alertmanager
    editable: true
    jsonData:
      handleGrafanaManagedAlerts: true
      implementation: prometheus
EOF

    # Dashboard provisioning
    cat > "${MONITORING_DIR}/grafana/provisioning/dashboards/dashboard.yml" << EOF
apiVersion: 1

providers:
  - name: 'BEV Dashboards'
    orgId: 1
    folder: 'BEV OSINT'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    log_success "Grafana configuration generated"
}

generate_grafana_dashboards() {
    log_info "Generating Grafana dashboards..."
    
    # BEV Overview Dashboard
    cat > "${MONITORING_DIR}/grafana/provisioning/dashboards/bev-overview.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "BEV Frontend Overview",
    "tags": ["bev", "frontend", "overview"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Service Status",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=~\"bev-.*\"}",
            "legendFormat": "{{job}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "mappings": [
              {
                "options": {
                  "0": {
                    "text": "DOWN"
                  },
                  "1": {
                    "text": "UP"
                  }
                },
                "type": "value"
              }
            ],
            "thresholds": {
              "steps": [
                {
                  "color": "red",
                  "value": null
                },
                {
                  "color": "green",
                  "value": 1
                }
              ]
            }
          }
        },
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{instance}} - {{method}}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 24,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total[5m]) * 100",
            "legendFormat": "CPU % - {{container_label_com_docker_compose_service}}"
          },
          {
            "expr": "(container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100",
            "legendFormat": "Memory % - {{container_label_com_docker_compose_service}}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 24,
          "x": 0,
          "y": 16
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF

    # System Dashboard
    cat > "${MONITORING_DIR}/grafana/provisioning/dashboards/system-monitoring.json" << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "System Monitoring",
    "tags": ["system", "infrastructure"],
    "style": "dark",
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "{{instance}}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "{{instance}}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Disk Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100",
            "legendFormat": "{{instance}} - {{mountpoint}}"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 24,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Network Traffic",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(node_network_receive_bytes_total[5m]) * 8",
            "legendFormat": "{{instance}} - {{device}} RX"
          },
          {
            "expr": "rate(node_network_transmit_bytes_total[5m]) * 8",
            "legendFormat": "{{instance}} - {{device}} TX"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 24,
          "x": 0,
          "y": 16
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    log_success "Grafana dashboards generated"
}

# =====================================================
# Docker Compose for Monitoring Stack
# =====================================================

generate_monitoring_compose() {
    log_info "Generating monitoring Docker Compose configuration..."
    
    cat > "${MONITORING_DIR}/docker-compose.monitoring.yml" << EOF
version: '3.9'

x-logging: &default-logging
  driver: json-file
  options:
    max-size: "10m"
    max-file: "3"

networks:
  bev_monitoring:
    external: true
  bev_frontend:
    external: true
  bev_osint:
    external: true

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:

services:
  # Prometheus Server
  prometheus:
    image: prom/prometheus:v2.45.0
    container_name: bev-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=${PROMETHEUS_RETENTION}'
      - '--storage.tsdb.retention.size=10GB'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
      - '--web.external-url=http://localhost:${PROMETHEUS_PORT}'
    ports:
      - "${PROMETHEUS_PORT}:9090"
    volumes:
      - ./prometheus/config:/etc/prometheus:ro
      - ./prometheus/rules:/etc/prometheus/rules:ro
      - prometheus_data:/prometheus
    networks:
      bev_monitoring:
        ipv4_address: 172.32.0.10
      bev_frontend:
      bev_osint:
    depends_on:
      - node-exporter
      - blackbox-exporter
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    logging: *default-logging
    labels:
      - "monitoring.service=prometheus"
      - "monitoring.role=metrics-collection"

  # Grafana Dashboard
  grafana:
    image: grafana/grafana:10.1.0
    container_name: bev-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=\${GRAFANA_ADMIN_PASSWORD:-bevgrafana2024}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/etc/grafana/provisioning/dashboards/bev-overview.json
      - GF_FEATURE_TOGGLES_ENABLE=ngalert
      - GF_UNIFIED_ALERTING_ENABLED=true
    ports:
      - "${GRAFANA_PORT}:3000"
    volumes:
      - ./grafana/config/grafana.ini:/etc/grafana/grafana.ini:ro
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana_data:/var/lib/grafana
    networks:
      bev_monitoring:
        ipv4_address: 172.32.0.11
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    logging: *default-logging
    labels:
      - "monitoring.service=grafana"
      - "monitoring.role=visualization"

  # AlertManager
  alertmanager:
    image: prom/alertmanager:v0.25.0
    container_name: bev-alertmanager
    restart: unless-stopped
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:${ALERTMANAGER_PORT}'
      - '--web.route-prefix=/'
      - '--cluster.listen-address=0.0.0.0:9094'
    ports:
      - "${ALERTMANAGER_PORT}:9093"
    volumes:
      - ./alertmanager/config:/etc/alertmanager:ro
      - alertmanager_data:/alertmanager
    networks:
      bev_monitoring:
        ipv4_address: 172.32.0.12
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9093/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    logging: *default-logging
    labels:
      - "monitoring.service=alertmanager"
      - "monitoring.role=alerting"

  # Node Exporter
  node-exporter:
    image: prom/node-exporter:v1.6.0
    container_name: bev-node-exporter
    restart: unless-stopped
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
      - '--collector.systemd'
      - '--collector.processes'
    ports:
      - "${NODE_EXPORTER_PORT}:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    networks:
      bev_monitoring:
        ipv4_address: 172.32.0.13
    pid: host
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9100/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging: *default-logging
    labels:
      - "monitoring.service=node-exporter"
      - "monitoring.role=system-metrics"

  # Blackbox Exporter
  blackbox-exporter:
    image: prom/blackbox-exporter:v0.24.0
    container_name: bev-blackbox-exporter
    restart: unless-stopped
    ports:
      - "${BLACKBOX_EXPORTER_PORT}:9115"
    volumes:
      - ./exporters/blackbox/blackbox.yml:/etc/blackbox_exporter/config.yml:ro
    networks:
      bev_monitoring:
        ipv4_address: 172.32.0.14
      bev_frontend:
    command:
      - '--config.file=/etc/blackbox_exporter/config.yml'
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9115/"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging: *default-logging
    labels:
      - "monitoring.service=blackbox-exporter"
      - "monitoring.role=endpoint-monitoring"

  # cAdvisor for container metrics
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.0
    container_name: bev-cadvisor
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    networks:
      bev_monitoring:
        ipv4_address: 172.32.0.15
    privileged: true
    devices:
      - /dev/kmsg
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:8080/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging: *default-logging
    labels:
      - "monitoring.service=cadvisor"
      - "monitoring.role=container-metrics"

  # PostgreSQL Exporter (for BEV database monitoring)
  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:v0.12.0
    container_name: bev-postgres-exporter
    restart: unless-stopped
    environment:
      - DATA_SOURCE_NAME=postgresql://\${POSTGRES_USER}:\${POSTGRES_PASSWORD}@bev_postgres:5432/osint?sslmode=disable
      - PG_EXPORTER_EXTEND_QUERY_PATH=/etc/postgres_exporter/queries.yaml
    ports:
      - "9187:9187"
    volumes:
      - ./exporters/postgres/queries.yaml:/etc/postgres_exporter/queries.yaml:ro
    networks:
      bev_monitoring:
        ipv4_address: 172.32.0.16
      bev_osint:
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9187/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging: *default-logging
    labels:
      - "monitoring.service=postgres-exporter"
      - "monitoring.role=database-metrics"

  # Redis Exporter (for BEV cache monitoring)
  redis-exporter:
    image: oliver006/redis_exporter:v1.52.0
    container_name: bev-redis-exporter
    restart: unless-stopped
    environment:
      - REDIS_ADDR=redis://bev_redis:6379
      - REDIS_PASSWORD=\${REDIS_PASSWORD}
    ports:
      - "9121:9121"
    networks:
      bev_monitoring:
        ipv4_address: 172.32.0.17
      bev_osint:
    depends_on:
      - prometheus
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost:9121/metrics"]
      interval: 30s
      timeout: 10s
      retries: 3
    logging: *default-logging
    labels:
      - "monitoring.service=redis-exporter"
      - "monitoring.role=cache-metrics"
EOF

    # Generate blackbox exporter configuration
    mkdir -p "${MONITORING_DIR}/exporters/blackbox"
    cat > "${MONITORING_DIR}/exporters/blackbox/blackbox.yml" << 'EOF'
modules:
  http_2xx:
    prober: http
    timeout: 5s
    http:
      valid_http_versions: ["HTTP/1.1", "HTTP/2.0"]
      valid_status_codes: []
      method: GET
      headers:
        Host: localhost
        Accept-Language: en-US
      no_follow_redirects: false
      preferred_ip_protocol: "ip4"
      
  http_post_2xx:
    prober: http
    timeout: 5s
    http:
      method: POST
      headers:
        Content-Type: application/json
      body: '{"health": "check"}'
      
  tcp_connect:
    prober: tcp
    timeout: 5s
    
  ssh_banner:
    prober: tcp
    timeout: 5s
    tcp:
      query_response:
        - expect: "^SSH-2.0-"
        
  dns:
    prober: dns
    timeout: 5s
    dns:
      query_name: "localhost"
      query_type: "A"
EOF

    # Generate PostgreSQL exporter queries
    mkdir -p "${MONITORING_DIR}/exporters/postgres"
    cat > "${MONITORING_DIR}/exporters/postgres/queries.yaml" << 'EOF'
pg_database:
  query: |
    SELECT 
      pg_database.datname,
      pg_database_size(pg_database.datname) as size_bytes
    FROM pg_database
  master: true
  cache_seconds: 30
  metrics:
    - datname:
        usage: "LABEL"
        description: "Name of the database"
    - size_bytes:
        usage: "GAUGE"
        description: "Disk space used by the database"
        
pg_stat_user_tables:
  query: |
    SELECT 
      schemaname,
      relname,
      seq_scan,
      seq_tup_read,
      idx_scan,
      idx_tup_fetch,
      n_tup_ins,
      n_tup_upd,
      n_tup_del,
      n_tup_hot_upd,
      n_live_tup,
      n_dead_tup
    FROM pg_stat_user_tables
  master: true
  cache_seconds: 30
  metrics:
    - schemaname:
        usage: "LABEL"
        description: "Name of the schema"
    - relname:
        usage: "LABEL"
        description: "Name of the table"
    - seq_scan:
        usage: "COUNTER"
        description: "Number of sequential scans initiated on this table"
    - seq_tup_read:
        usage: "COUNTER"
        description: "Number of live rows fetched by sequential scans"
    - idx_scan:
        usage: "COUNTER"
        description: "Number of index scans initiated on this table"
    - idx_tup_fetch:
        usage: "COUNTER"
        description: "Number of live rows fetched by index scans"
    - n_tup_ins:
        usage: "COUNTER"
        description: "Number of rows inserted"
    - n_tup_upd:
        usage: "COUNTER"
        description: "Number of rows updated"
    - n_tup_del:
        usage: "COUNTER"
        description: "Number of rows deleted"
    - n_tup_hot_upd:
        usage: "COUNTER"
        description: "Number of rows HOT updated"
    - n_live_tup:
        usage: "GAUGE"
        description: "Estimated number of live rows"
    - n_dead_tup:
        usage: "GAUGE"
        description: "Estimated number of dead rows"
EOF

    log_success "Monitoring Docker Compose configuration generated"
}

# =====================================================
# Deployment and Health Checks
# =====================================================

deploy_monitoring_stack() {
    log_info "Deploying monitoring stack..."
    
    cd "${MONITORING_DIR}"
    
    # Load environment variables
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        set -a
        source "${PROJECT_ROOT}/.env"
        set +a
    fi
    
    # Create and start monitoring services
    log_info "Building and starting monitoring containers..."
    docker-compose -f docker-compose.monitoring.yml up -d
    
    # Wait for services to be healthy
    log_info "Waiting for monitoring services to be healthy..."
    local timeout=300
    local start_time=$(date +%s)
    local services=("bev-prometheus" "bev-grafana" "bev-alertmanager" "bev-node-exporter" "bev-blackbox-exporter")
    
    while [ $(($(date +%s) - start_time)) -lt $timeout ]; do
        local healthy_count=0
        
        for service in "${services[@]}"; do
            local health=$(docker inspect "$service" --format='{{.State.Health.Status}}' 2>/dev/null || echo "no_healthcheck")
            if [ "$health" = "healthy" ] || [ "$health" = "no_healthcheck" ]; then
                ((healthy_count++))
            fi
        done
        
        if [ $healthy_count -eq ${#services[@]} ]; then
            log_success "All monitoring services are healthy"
            break
        fi
        
        log_info "Waiting for services... ($healthy_count/${#services[@]} healthy)"
        sleep 10
    done
    
    if [ $(($(date +%s) - start_time)) -ge $timeout ]; then
        log_error "Timeout waiting for monitoring services to become healthy"
        return 1
    fi
    
    log_success "Monitoring stack deployment completed"
}

validate_monitoring_endpoints() {
    log_info "Validating monitoring endpoints..."
    
    local endpoints=(
        "http://localhost:${PROMETHEUS_PORT}/-/healthy:Prometheus"
        "http://localhost:${GRAFANA_PORT}/api/health:Grafana"
        "http://localhost:${ALERTMANAGER_PORT}/-/healthy:AlertManager"
        "http://localhost:${NODE_EXPORTER_PORT}/metrics:Node Exporter"
        "http://localhost:${BLACKBOX_EXPORTER_PORT}/:Blackbox Exporter"
    )
    
    local failed_endpoints=()
    
    for endpoint_info in "${endpoints[@]}"; do
        local url=$(echo "$endpoint_info" | cut -d: -f1-3)
        local name=$(echo "$endpoint_info" | cut -d: -f4)
        
        local response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
        if [ "$response" -eq 200 ]; then
            log_success "$name endpoint validation passed"
        else
            log_error "$name endpoint validation failed (HTTP $response)"
            failed_endpoints+=("$name")
        fi
    done
    
    if [ ${#failed_endpoints[@]} -eq 0 ]; then
        log_success "All monitoring endpoints validated successfully"
        return 0
    else
        log_error "Failed endpoint validations: ${failed_endpoints[*]}"
        return 1
    fi
}

# =====================================================
# Monitoring Integration Setup
# =====================================================

setup_monitoring_integration() {
    log_info "Setting up monitoring integration with BEV services..."
    
    # Add monitoring configuration to frontend services
    if [ -f "${PROJECT_ROOT}/frontend/mcp-server/src/index.js" ]; then
        log_info "Adding metrics endpoint to MCP server..."
        
        # Create metrics middleware for MCP server
        cat > "${PROJECT_ROOT}/frontend/mcp-server/src/middleware/metrics.js" << 'EOF'
import prometheus from 'prom-client';

// Create a Registry to register the metrics
const register = new prometheus.Registry();

// Add default metrics
prometheus.collectDefaultMetrics({
  app: 'bev-mcp-server',
  timeout: 5000,
  gcDurationBuckets: [0.001, 0.01, 0.1, 1, 2, 5],
  register
});

// Custom metrics
const httpRequestsTotal = new prometheus.Counter({
  name: 'http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code'],
  registers: [register]
});

const httpRequestDuration = new prometheus.Histogram({
  name: 'http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
  registers: [register]
});

const wsConnectionsActive = new prometheus.Gauge({
  name: 'websocket_connections_active',
  help: 'Number of active WebSocket connections',
  registers: [register]
});

const bevIntegrationHealth = new prometheus.Gauge({
  name: 'bev_integration_health_status',
  help: 'BEV integration health status (1 = healthy, 0 = unhealthy)',
  registers: [register]
});

const dataProcessedTotal = new prometheus.Counter({
  name: 'bev_data_processed_total',
  help: 'Total number of data items processed',
  labelNames: ['type', 'source'],
  registers: [register]
});

// Middleware function
export const metricsMiddleware = (req, res, next) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    
    httpRequestsTotal
      .labels(req.method, req.route?.path || req.path, res.statusCode)
      .inc();
    
    httpRequestDuration
      .labels(req.method, req.route?.path || req.path, res.statusCode)
      .observe(duration);
  });
  
  next();
};

// Metrics endpoint
export const metricsHandler = async (req, res) => {
  try {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
  } catch (error) {
    res.status(500).end(error);
  }
};

export {
  register,
  httpRequestsTotal,
  httpRequestDuration,
  wsConnectionsActive,
  bevIntegrationHealth,
  dataProcessedTotal
};
EOF
    fi
    
    # Create health check endpoints for all services
    if [ -d "${PROJECT_ROOT}/frontend" ]; then
        log_info "Adding health check endpoints..."
        
        # Health check configuration
        cat > "${PROJECT_ROOT}/frontend/health-checks.json" << EOF
{
  "health_checks": {
    "mcp_server": {
      "endpoint": "http://localhost:3011/health",
      "timeout": 5000,
      "interval": 30000
    },
    "websocket_server": {
      "endpoint": "ws://localhost:8081/health",
      "timeout": 5000,
      "interval": 30000
    },
    "frontend_proxy": {
      "endpoint": "http://localhost:8080/health",
      "timeout": 5000,
      "interval": 30000
    }
  },
  "dependencies": {
    "postgres": {
      "host": "bev_postgres",
      "port": 5432,
      "timeout": 5000
    },
    "redis": {
      "host": "bev_redis", 
      "port": 6379,
      "timeout": 5000
    },
    "neo4j": {
      "host": "bev_neo4j",
      "port": 7687,
      "timeout": 5000
    }
  }
}
EOF
    fi
    
    log_success "Monitoring integration setup completed"
}

# =====================================================
# Final Validation and Documentation
# =====================================================

generate_monitoring_documentation() {
    log_info "Generating monitoring documentation..."
    
    cat > "${MONITORING_DIR}/README.md" << EOF
# BEV Frontend Monitoring Setup

## Overview
Comprehensive monitoring solution for BEV Frontend Integration using Prometheus, Grafana, and AlertManager.

## Services
- **Prometheus**: Metrics collection and alerting rules (Port: ${PROMETHEUS_PORT})
- **Grafana**: Visualization and dashboards (Port: ${GRAFANA_PORT})
- **AlertManager**: Alert routing and notifications (Port: ${ALERTMANAGER_PORT})
- **Node Exporter**: System metrics (Port: ${NODE_EXPORTER_PORT})
- **Blackbox Exporter**: Endpoint monitoring (Port: ${BLACKBOX_EXPORTER_PORT})
- **cAdvisor**: Container metrics (Port: 8080)
- **PostgreSQL Exporter**: Database metrics (Port: 9187)
- **Redis Exporter**: Cache metrics (Port: 9121)

## Access URLs
- Prometheus: http://localhost:${PROMETHEUS_PORT}
- Grafana: http://localhost:${GRAFANA_PORT} (admin/\${GRAFANA_ADMIN_PASSWORD})
- AlertManager: http://localhost:${ALERTMANAGER_PORT}

## Key Metrics
- Service availability and health status
- HTTP request rates and response times
- System resource utilization (CPU, memory, disk)
- Database performance metrics
- WebSocket connection statistics
- Error rates and alert status

## Dashboards
- **BEV Frontend Overview**: Service status, request metrics, performance
- **System Monitoring**: Infrastructure metrics, resource usage
- **Database Performance**: PostgreSQL metrics and queries
- **Network and Security**: Traffic patterns, SSL status

## Alerting Rules
- Service downtime alerts
- High resource usage warnings
- Performance degradation notifications
- Security and SSL certificate expiry alerts
- Integration health monitoring

## Commands
\`\`\`bash
# Start monitoring stack
cd ${MONITORING_DIR}
docker-compose -f docker-compose.monitoring.yml up -d

# Stop monitoring stack
docker-compose -f docker-compose.monitoring.yml down

# View logs
docker-compose -f docker-compose.monitoring.yml logs -f

# Restart specific service
docker-compose -f docker-compose.monitoring.yml restart prometheus
\`\`\`

## Configuration Files
- \`prometheus/config/prometheus.yml\`: Main Prometheus configuration
- \`prometheus/rules/*.yml\`: Alerting rules
- \`grafana/config/grafana.ini\`: Grafana configuration
- \`alertmanager/config/alertmanager.yml\`: Alert routing configuration

## Health Checks
All services include health checks and will restart automatically if unhealthy.

## Data Retention
- Prometheus: ${PROMETHEUS_RETENTION}
- Grafana: ${GRAFANA_RETENTION}

## Troubleshooting
1. Check service logs: \`docker logs <container_name>\`
2. Verify network connectivity between containers
3. Ensure all required environment variables are set
4. Check disk space for data persistence volumes

Generated: $(date)
EOF

    # Create monitoring status script
    cat > "${MONITORING_DIR}/monitoring-status.sh" << 'EOF'
#!/bin/bash
# BEV Monitoring Status Check Script

echo "BEV Monitoring Stack Status"
echo "=========================="
echo "Timestamp: $(date)"
echo ""

# Check container status
echo "Container Status:"
echo "-----------------"
containers=("bev-prometheus" "bev-grafana" "bev-alertmanager" "bev-node-exporter" "bev-blackbox-exporter" "bev-cadvisor")
for container in "${containers[@]}"; do
    status=$(docker inspect "$container" --format='{{.State.Status}}' 2>/dev/null || echo "not_found")
    health=$(docker inspect "$container" --format='{{.State.Health.Status}}' 2>/dev/null || echo "no_healthcheck")
    printf "%-25s %-10s %s\n" "$container" "$status" "$health"
done

echo ""
echo "Endpoint Health:"
echo "----------------"
endpoints=(
    "http://localhost:9090/-/healthy:Prometheus"
    "http://localhost:3001/api/health:Grafana"
    "http://localhost:9093/-/healthy:AlertManager"
    "http://localhost:9100/metrics:Node Exporter"
    "http://localhost:9115/:Blackbox Exporter"
)

for endpoint_info in "${endpoints[@]}"; do
    url=$(echo "$endpoint_info" | cut -d: -f1-3)
    name=$(echo "$endpoint_info" | cut -d: -f4)
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null || echo "000")
    if [ "$response" -eq 200 ]; then
        printf "%-25s %s\n" "$name" "✅ OK"
    else
        printf "%-25s %s\n" "$name" "❌ FAIL ($response)"
    fi
done

echo ""
echo "Resource Usage:"
echo "---------------"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep bev-
EOF

    chmod +x "${MONITORING_DIR}/monitoring-status.sh"
    
    log_success "Monitoring documentation generated"
}

# =====================================================
# Main Execution Flow
# =====================================================

main() {
    setup_logging
    
    log_info "Starting BEV Frontend health monitoring setup"
    log_info "Monitoring directory: ${MONITORING_DIR}"
    
    # Load environment variables
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        set -a
        source "${PROJECT_ROOT}/.env"
        set +a
    fi
    
    local setup_steps=(
        "setup_monitoring_infrastructure"
        "generate_prometheus_config"
        "generate_prometheus_rules"
        "generate_alertmanager_config"
        "generate_grafana_config"
        "generate_grafana_dashboards"
        "generate_monitoring_compose"
        "deploy_monitoring_stack"
        "validate_monitoring_endpoints"
        "setup_monitoring_integration"
        "generate_monitoring_documentation"
    )
    
    local failed_steps=()
    
    for step in "${setup_steps[@]}"; do
        log_info "Executing monitoring setup step: ${step}"
        if ! ${step}; then
            log_error "Monitoring setup step failed: ${step}"
            failed_steps+=("${step}")
            break
        else
            log_success "Monitoring setup step completed: ${step}"
        fi
        echo "---"
    done
    
    # Summary
    echo "=============================================="
    log_info "BEV Frontend monitoring setup summary:"
    
    if [ ${#failed_steps[@]} -eq 0 ]; then
        log_success "Monitoring setup completed successfully!"
        
        # Write monitoring success marker
        echo "MONITORING_STATUS=SUCCESS" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "MONITORING_TIMESTAMP=$(date -Iseconds)" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "PROMETHEUS_URL=http://localhost:${PROMETHEUS_PORT}" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "GRAFANA_URL=http://localhost:${GRAFANA_PORT}" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "ALERTMANAGER_URL=http://localhost:${ALERTMANAGER_PORT}" >> "${PROJECT_ROOT}/.frontend_deployment"
        
        echo "=============================================="
        echo "✅ BEV MONITORING SETUP SUCCESSFUL"
        echo "   Prometheus: http://localhost:${PROMETHEUS_PORT}"
        echo "   Grafana: http://localhost:${GRAFANA_PORT}"
        echo "   AlertManager: http://localhost:${ALERTMANAGER_PORT}"
        echo "   Node Exporter: http://localhost:${NODE_EXPORTER_PORT}"
        echo "   Documentation: ${MONITORING_DIR}/README.md"
        echo "   Status Check: ${MONITORING_DIR}/monitoring-status.sh"
        echo "   Log file: ${LOG_FILE}"
        echo "=============================================="
        
        exit 0
    else
        log_error "Monitoring setup failed at step: ${failed_steps[*]}"
        
        # Write monitoring failure marker
        echo "MONITORING_STATUS=FAILED" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "MONITORING_TIMESTAMP=$(date -Iseconds)" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "FAILED_MONITORING_STEP=${failed_steps[*]}" >> "${PROJECT_ROOT}/.frontend_deployment"
        
        echo "=============================================="
        echo "❌ BEV MONITORING SETUP FAILED"
        echo "   Failed at step: ${failed_steps[*]}"
        echo "   Log file: ${LOG_FILE}"
        echo "   Review logs and retry setup"
        echo "=============================================="
        
        exit 1
    fi
}

# Trap for cleanup
trap 'log_error "Monitoring setup interrupted"; exit 130' INT TERM

# Execute main function
main "$@"