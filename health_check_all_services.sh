#!/bin/bash
# Health Check - All BEV Services
echo "🔍 HEALTH CHECK: Checking all BEV services..."

echo "=== THANOS Services ==="
docker-compose -f docker-compose-thanos-unified.yml --env-file .env.thanos.complete ps

echo "=== ORACLE1 Services ==="
docker-compose -f docker-compose-oracle1-unified.yml --env-file .env.oracle1.complete ps

echo "=== Resource Usage ==="
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | grep bev || echo "No BEV containers running"

echo "✅ Health check completed"
