#!/bin/bash
# Complete BEV Deployment Script
# Deploy all components in correct order

set -e

echo "🚀 BEV COMPLETE SYSTEM DEPLOYMENT"
echo "=================================="

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Environment Setup
echo -e "\n${YELLOW}Step 1: Environment Setup${NC}"
cp .env.example .env
source .env

# Step 2: Database Infrastructure
echo -e "\n${YELLOW}Step 2: Database Infrastructure${NC}"
cd docker/databases
docker-compose up -d
sleep 30

# Initialize databases
for script in init-scripts/postgres/*.sql; do
    docker exec -i bev_postgres psql -U swarm_admin -d ai_swarm < "$script"
    echo -e "${GREEN}✓ Applied: $(basename $script)${NC}"
done
cd ../..

# Step 3: Message Queue Infrastructure
echo -e "\n${YELLOW}Step 3: Message Queue Infrastructure${NC}"
cd docker/message-queue
docker-compose -f docker-compose-messaging.yml up -d
sleep 20
cd ../..

# Step 4: Core Services
echo -e "\n${YELLOW}Step 4: Core Services${NC}"
docker-compose -f docker-compose.complete.yml up -d

# Step 5: Initialize Agents
echo -e "\n${YELLOW}Step 5: Initialize Agents${NC}"
python3 src/agents/agent_protocol.py &
AGENT_PID=$!
sleep 10

# Step 6: Deploy Airflow
echo -e "\n${YELLOW}Step 6: Deploy Airflow${NC}"
docker exec bev_airflow airflow db init
docker exec bev_airflow airflow users create \
    --username admin \
    --password BevAdmin2024! \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@bev.ai

# Copy DAGs
cp dags/*.py /opt/airflow/dags/

# Step 7: Deploy N8N Workflows
echo -e "\n${YELLOW}Step 7: Deploy N8N Workflows${NC}"
for workflow in n8n-workflows/*.json; do
    curl -X POST http://localhost:5678/rest/workflows \
        -H "Content-Type: application/json" \
        -d "@$workflow"
    echo -e "${GREEN}✓ Deployed: $(basename $workflow)${NC}"
done

# Step 8: Initialize OCR Pipeline
echo -e "\n${YELLOW}Step 8: Initialize OCR Pipeline${NC}"
python3 -c "from src.pipeline.enhanced_ocr_pipeline import EnhancedOCRPipeline; pipeline = EnhancedOCRPipeline()"

# Step 9: Multi-Node Setup (if applicable)
echo -e "\n${YELLOW}Step 9: Multi-Node Setup${NC}"
if [ "$DEPLOY_MULTI_NODE" == "true" ]; then
    python3 deploy/multi_node_orchestrator.py
else
    echo "Skipping multi-node deployment (single node mode)"
fi

# Step 10: Run Integration Tests
echo -e "\n${YELLOW}Step 10: Running Integration Tests${NC}"
python3 scripts/complete_integration_test.py

# Step 11: Start Monitoring
echo -e "\n${YELLOW}Step 11: Start Monitoring${NC}"
docker-compose -f docker-compose-monitoring.yml up -d

echo -e "\n${GREEN}=================================="
echo -e "🎉 BEV DEPLOYMENT COMPLETE!"
echo -e "==================================${NC}"
echo
echo "Access Points:"
echo "  • Web UI: http://localhost:8000"
echo "  • Airflow: http://localhost:8080 (admin/BevAdmin2024!)"
echo "  • RabbitMQ: http://localhost:15672 (admin/BevSwarm2024!)"
echo "  • Kafka UI: http://localhost:8090"
echo "  • Grafana: http://localhost:3000 (admin/admin)"
echo "  • N8N: http://localhost:5678"
echo
echo "Next Steps:"
echo "  1. Review integration test report: cat /tmp/bev_integration_report.json"
echo "  2. Monitor system health: http://localhost:3000/d/bev-overview"
echo "  3. Check agent status: curl http://localhost:8000/agents/status"
echo
echo -e "${GREEN}✨ BEV is now 100% operational! ✨${NC}"
