#!/bin/bash
# Emergency Stop - All BEV Services
echo "🚨 EMERGENCY STOP: Stopping all BEV services..."

# Stop THANOS services
echo "Stopping THANOS services..."
docker-compose -f docker-compose-thanos-unified.yml --env-file .env.thanos.complete down -t 5 || true

# Stop ORACLE1 services
echo "Stopping ORACLE1 services..."
docker-compose -f docker-compose-oracle1-unified.yml --env-file .env.oracle1.complete down -t 5 || true

# Stop any remaining BEV containers
echo "Stopping any remaining BEV containers..."
docker ps --format "{{.Names}}" | grep "bev_" | xargs -r docker stop || true

echo "✅ Emergency stop completed"
