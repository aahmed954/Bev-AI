#!/bin/bash
# ARM Monitoring Setup Script for Oracle1 Node
# Configures Prometheus, Grafana, and monitoring stack for ARM64

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}📈 SETTING UP ARM MONITORING STACK${NC}"
echo "======================================"

# Verify ARM architecture
if [ "$(uname -m)" != "aarch64" ]; then
    echo -e "${YELLOW}⚠️ Not on ARM64 architecture: $(uname -m)${NC}"
fi

# Wait for monitoring services
echo -e "${YELLOW}⏳ Waiting for monitoring services...${NC}"
sleep 20

# Configure Prometheus for ARM
echo -e "${BLUE}🎯 Configuring Prometheus...${NC}"
until curl -s http://localhost:9090/-/ready > /dev/null 2>&1; do
    echo "Waiting for Prometheus..."
    sleep 5
done

# Verify Prometheus targets
echo -n "Checking Prometheus targets... "
TARGET_COUNT=$(curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets | length' 2>/dev/null || echo "0")
echo -e "${GREEN}✅ $TARGET_COUNT targets${NC}"

# Configure Grafana
echo -e "${BLUE}📊 Configuring Grafana...${NC}"
until curl -s http://localhost:3000/api/health > /dev/null 2>&1; do
    echo "Waiting for Grafana..."
    sleep 5
done

# Setup Grafana data sources
curl -X POST http://admin:admin@localhost:3000/api/datasources \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy",
    "isDefault": true
  }' > /dev/null 2>&1 || echo "Data source may already exist"

curl -X POST http://admin:admin@localhost:3000/api/datasources \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "InfluxDB",
    "type": "influxdb",
    "url": "http://thanos:8086",
    "access": "proxy",
    "database": "bev-metrics"
  }' > /dev/null 2>&1 || echo "Data source may already exist"

echo -e "${GREEN}✅ Grafana configured${NC}"

# Setup alert manager
echo -e "${BLUE}🚨 Setting up Alert Manager...${NC}"
if docker ps | grep bev_alertmanager > /dev/null; then
    echo -e "${GREEN}✅ Alert Manager running${NC}"
else
    echo -e "${YELLOW}⚠️ Alert Manager not deployed${NC}"
fi

# Create ARM-specific monitoring configuration
echo -e "${BLUE}🔧 Creating ARM monitoring configuration...${NC}"
mkdir -p /tmp/arm-monitoring-config

cat > /tmp/arm-monitoring-config/prometheus-arm.yml << PROM_EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "arm_alert_rules.yml"

scrape_configs:
  - job_name: 'oracle1-node'
    static_configs:
      - targets: ['localhost:9100']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'arm-services'
    static_configs:
      - targets: [
          'redis:6379',
          'vault:8200',
          'consul:8500',
          'tor:9050'
        ]

  - job_name: 'thanos-federation'
    honor_labels: true
    static_configs:
      - targets: ['thanos:9090']
PROM_EOF

# Create ARM alert rules
cat > /tmp/arm-monitoring-config/arm_alert_rules.yml << ALERT_EOF
groups:
- name: arm_node_alerts
  rules:
  - alert: HighCPUUsage
    expr: cpu_usage_percent > 90
    for: 5m
    labels:
      severity: warning
      node: oracle1
    annotations:
      summary: "High CPU usage on ARM node"

  - alert: HighMemoryUsage
    expr: memory_usage_percent > 85
    for: 5m
    labels:
      severity: warning
      node: oracle1
    annotations:
      summary: "High memory usage on ARM node"

  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service {{ $labels.instance }} is down"
ALERT_EOF

echo -e "${GREEN}✅ ARM monitoring configuration created${NC}"

# Verify monitoring health
echo -e "${BLUE}🏥 Running monitoring health checks...${NC}"

MONITORING_SERVICES=("prometheus:9090" "grafana:3000")
HEALTHY_MONITORING=0

for service in "${MONITORING_SERVICES[@]}"; do
    SERVICE_NAME=$(echo $service | cut -d: -f1)
    SERVICE_PORT=$(echo $service | cut -d: -f2)
    
    echo -n "Health check $SERVICE_NAME... "
    if curl -s http://localhost:$SERVICE_PORT > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
        HEALTHY_MONITORING=$((HEALTHY_MONITORING + 1))
    else
        echo -e "${RED}❌ Unhealthy${NC}"
    fi
done

echo ""
echo -e "${BLUE}📊 ARM Monitoring Health Summary:${NC}"
echo "Healthy Services: $HEALTHY_MONITORING/${#MONITORING_SERVICES[@]}"

if [ $HEALTHY_MONITORING -eq ${#MONITORING_SERVICES[@]} ]; then
    echo -e "${GREEN}🎯 ARM monitoring setup successful!${NC}"
    exit 0
else
    echo -e "${YELLOW}⚠️ Some monitoring services need attention${NC}"
    exit 1
fi