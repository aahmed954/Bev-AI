#!/bin/bash
# Deploy BEV to THANOS (Production Server)
# IP: 100.122.12.54

set -e

echo "🚀 Deploying BEV to THANOS Production Server"

THANOS_HOST="100.122.12.54"
THANOS_USER="bev"
DEPLOY_DIR="/opt/bev"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check connectivity
echo -e "${YELLOW}Checking connection to THANOS...${NC}"
if ! ssh -q $THANOS_USER@$THANOS_HOST exit; then
    echo -e "${RED}Cannot connect to THANOS. Check SSH configuration.${NC}"
    exit 1
fi

# Prepare deployment package
echo -e "${YELLOW}Preparing deployment package...${NC}"
tar -czf /tmp/bev-deploy.tar.gz \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='logs' \
    --exclude='data' \
    -C /home/starlord/Projects/Bev .

# Copy to THANOS
echo -e "${YELLOW}Copying deployment package to THANOS...${NC}"
scp /tmp/bev-deploy.tar.gz $THANOS_USER@$THANOS_HOST:/tmp/

# Deploy on THANOS
echo -e "${YELLOW}Deploying on THANOS...${NC}"
ssh $THANOS_USER@$THANOS_HOST << 'EOF'
set -e

# Create deployment directory
sudo mkdir -p /opt/bev
sudo chown bev:bev /opt/bev

# Extract deployment
cd /opt/bev
tar -xzf /tmp/bev-deploy.tar.gz

# Install dependencies
echo "Installing Python dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment
cp .env.example .env

# Initialize databases
echo "Initializing databases..."
cd docker/databases
docker-compose up -d

# Wait for databases to be ready
sleep 30

# Run initialization scripts
for script in init-scripts/postgres/*.sql; do
    docker exec -i bev_postgres psql -U swarm_admin -d ai_swarm < $script
done

# Deploy message queues
echo "Deploying message queues..."
cd ../message-queue
docker-compose -f docker-compose-messaging.yml up -d

# Deploy core services
echo "Deploying core services..."
cd /opt/bev
docker-compose -f docker-compose-thanos-unified.yml up -d

# Initialize Airflow
echo "Initializing Airflow..."
docker exec bev_airflow airflow db init
docker exec bev_airflow airflow users create \
    --username admin \
    --password BevAdmin2024! \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@bev.ai

# Deploy DAGs
cp dags/*.py /opt/airflow/dags/

# Start services
echo "Starting BEV services..."
systemctl start bev-swarm
systemctl start bev-agent-coordinator
systemctl start bev-message-queue
systemctl start bev-monitoring

echo "✅ Deployment to THANOS complete!"
EOF

echo -e "${GREEN}✅ BEV successfully deployed to THANOS!${NC}"
echo -e "${GREEN}Access points:${NC}"
echo "  - Web UI: http://$THANOS_HOST:8000"
echo "  - Airflow: http://$THANOS_HOST:8080"
echo "  - RabbitMQ: http://$THANOS_HOST:15672"
echo "  - Kafka UI: http://$THANOS_HOST:8090"
echo "  - Grafana: http://$THANOS_HOST:3000"
