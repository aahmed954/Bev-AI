#!/bin/bash

# BEV OSINT Framework - THANOS Node Memory Optimization
# Target: Reduce memory utilization from 85% to 75%

set -euo pipefail

echo "🔧 Starting THANOS Node Memory Optimization..."

# 1. System-level memory optimization
echo "⚙️ Applying system memory optimizations..."

# Configure kernel memory parameters
sudo tee -a /etc/sysctl.conf << EOF

# BEV Memory Optimization Settings
vm.swappiness=10
vm.vfs_cache_pressure=50
vm.dirty_ratio=5
vm.dirty_background_ratio=2
vm.overcommit_memory=1
vm.min_free_kbytes=65536
EOF

# Apply settings immediately
sudo sysctl -p

# 2. Docker memory optimization
echo "🐳 Optimizing Docker memory settings..."

# Create Docker daemon configuration for memory optimization
sudo tee /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-ulimits": {
    "memlock": {
      "Hard": -1,
      "Name": "memlock",
      "Soft": -1
    }
  }
}
EOF

# 3. Service-specific memory limits
echo "📊 Updating service memory limits..."

# Create memory-optimized Docker Compose override
cat > docker-compose.memory-optimized.yml << EOF
version: '3.8'

services:
  # High-memory services optimization
  intelowl_postgres:
    mem_limit: 8g
    mem_reservation: 4g

  intelowl_elasticsearch:
    mem_limit: 6g
    mem_reservation: 3g
    environment:
      - "ES_JAVA_OPTS=-Xms2g -Xmx4g"

  bev_neo4j:
    mem_limit: 4g
    mem_reservation: 2g
    environment:
      - NEO4J_dbms_memory_heap_initial__size=1g
      - NEO4J_dbms_memory_heap_max__size=2g

  bev_redis:
    mem_limit: 2g
    mem_reservation: 1g

  intelowl_nginx:
    mem_limit: 512m
    mem_reservation: 256m

  # Vector databases optimization
  bev_qdrant:
    mem_limit: 3g
    mem_reservation: 1.5g

  bev_weaviate:
    mem_limit: 3g
    mem_reservation: 1.5g

  # Monitoring services optimization
  bev_prometheus:
    mem_limit: 2g
    mem_reservation: 1g
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=10GB'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
EOF

# 4. Create memory monitoring script
cat > scripts/monitor-memory.sh << 'EOF'
#!/bin/bash

# Memory monitoring for THANOS node
echo "🔍 THANOS Node Memory Status:"
echo "================================"

# System memory
free -h

echo ""
echo "📊 Memory Utilization:"
TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $2}')
USED_MEM=$(free -m | awk 'NR==2{printf "%.0f", $3}')
UTILIZATION=$(awk "BEGIN {printf \"%.1f\", $USED_MEM/$TOTAL_MEM*100}")

echo "Total: ${TOTAL_MEM}MB"
echo "Used: ${USED_MEM}MB"
echo "Utilization: ${UTILIZATION}%"

if (( $(echo "$UTILIZATION > 80.0" | bc -l) )); then
    echo "⚠️  WARNING: Memory utilization above 80%"
elif (( $(echo "$UTILIZATION > 75.0" | bc -l) )); then
    echo "🔶 CAUTION: Memory utilization above 75%"
else
    echo "✅ GOOD: Memory utilization within target"
fi

echo ""
echo "🐳 Docker Container Memory Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}" | head -15

echo ""
echo "📈 Top Memory Consumers:"
ps aux --sort=-%mem | head -10
EOF

chmod +x scripts/monitor-memory.sh

# 5. Create memory alert configuration
cat > config/memory-alerts.yml << EOF
groups:
  - name: memory_alerts
    rules:
      - alert: HighMemoryUsage
        expr: (1 - node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes) > 0.80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected on THANOS node"
          description: "Memory usage is above 80%: {{ \$value }}%"

      - alert: CriticalMemoryUsage
        expr: (1 - node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes) > 0.90
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "CRITICAL: Memory usage on THANOS node"
          description: "Memory usage is critically high: {{ \$value }}%"
EOF

# 6. Clean up unnecessary files and caches
echo "🧹 Cleaning up system caches..."

# Clear package caches
sudo apt-get clean
sudo apt-get autoclean

# Clear Docker system
docker system prune -f

# Clear logs older than 7 days
sudo find /var/log -name "*.log" -type f -mtime +7 -delete

echo "✅ Memory optimization complete!"
echo ""
echo "🔍 Current memory status:"
./scripts/monitor-memory.sh

echo ""
echo "📋 Next steps:"
echo "1. Restart Docker daemon: sudo systemctl restart docker"
echo "2. Deploy with memory limits: docker-compose -f docker-compose.complete.yml -f docker-compose.memory-optimized.yml up -d"
echo "3. Monitor memory usage: ./scripts/monitor-memory.sh"