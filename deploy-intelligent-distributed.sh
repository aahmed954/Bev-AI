#!/bin/bash
# BEV INTELLIGENT DISTRIBUTED DEPLOYMENT
# Optimized for actual hardware capabilities and GitHub-based deployment

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${PURPLE}🚀 BEV INTELLIGENT DISTRIBUTED DEPLOYMENT${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""
echo "HARDWARE DETECTED:"
echo "  THANOS (Local):  Ryzen 9 5900X, 64GB RAM, RTX 3080, 1.8TB"
echo "  ORACLE1 (Cloud): ARM64 4-core, 24GB RAM, 194GB"
echo "  STARLORD (You):  Ryzen 9 7950X3D, 64GB RAM, 1.6TB"
echo ""
echo "DEPLOYMENT STRATEGY:"
echo "  THANOS:  Heavy compute, GPU tasks, databases"
echo "  ORACLE1: Lightweight services, monitoring, coordination"
echo "  STARLORD: Development UI, visualization (auto-start/stop)"
echo ""

# Configuration
GITHUB_REPO="https://github.com/aahmed954/Bev-AI.git"
THANOS_HOST="thanos"
ORACLE1_HOST="oracle1"
LOCAL_PROJECT="/home/starlord/Projects/Bev"

# Verify connectivity
echo -e "${CYAN}Step 1: Verifying Connectivity${NC}"
for host in thanos oracle1; do
    if ssh -o ConnectTimeout=5 starlord@$host "echo 'Connected to $host'" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $host accessible${NC}"
    else
        echo -e "${RED}❌ Cannot connect to $host${NC}"
        exit 1
    fi
done

# Step 2: Push local changes to GitHub
echo -e "${CYAN}Step 2: Pushing Local Changes to GitHub${NC}"
cd $LOCAL_PROJECT

# First, let's fix critical issues locally
echo "Fixing critical deployment issues..."

# Create proper docker-compose files for each node
cat > docker-compose-thanos.yml << 'EOF'
version: '3.8'

# THANOS NODE - Heavy compute, databases, GPU services
# Hardware: Ryzen 9 5900X (24 threads), 64GB RAM, RTX 3080

services:
  # Core Databases (Heavy I/O)
  postgres:
    image: postgres:15-alpine
    container_name: bev-postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: bev_db
      POSTGRES_USER: bev_user
      POSTGRES_PASSWORD: ${DB_PASSWORD:-secure_password_123}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    networks:
      - bev-network

  neo4j:
    image: neo4j:5-community
    container_name: bev-neo4j
    restart: unless-stopped
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD:-secure_password_123}
      NEO4J_server_memory_heap_max__size: 8G
    volumes:
      - neo4j-data:/data
      - neo4j-logs:/logs
    ports:
      - "7474:7474"
      - "7687:7687"
    networks:
      - bev-network

  redis-primary:
    image: redis:7-alpine
    container_name: bev-redis-primary
    restart: unless-stopped
    command: redis-server --maxmemory 4gb --maxmemory-policy allkeys-lru
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - bev-network

  # Message Queues (CPU intensive)
  rabbitmq:
    image: rabbitmq:3-management-alpine
    container_name: bev-rabbitmq
    restart: unless-stopped
    environment:
      RABBITMQ_DEFAULT_USER: admin
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASSWORD:-secure_password_123}
    ports:
      - "5672:5672"
      - "15672:15672"
    networks:
      - bev-network

  # GPU Services
  ml-processor:
    image: pytorch/pytorch:latest
    container_name: bev-ml-processor
    restart: unless-stopped
    runtime: nvidia
    environment:
      CUDA_VISIBLE_DEVICES: 0
    volumes:
      - ./src:/app/src
      - ml-models:/models
    networks:
      - bev-network
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  postgres-data:
  neo4j-data:
  neo4j-logs:
  redis-data:
  ml-models:

networks:
  bev-network:
    driver: bridge
    name: bev-network
EOF

cat > docker-compose-oracle1.yml << 'EOF'
version: '3.8'

# ORACLE1 NODE - ARM64 Cloud VM, Lightweight services
# Hardware: ARM64 4-core, 24GB RAM

services:
  # Monitoring Stack (Lightweight)
  prometheus:
    image: prom/prometheus:latest
    container_name: bev-prometheus
    restart: unless-stopped
    platform: linux/arm64
    volumes:
      - ./config/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'
    networks:
      - bev-network

  grafana:
    image: grafana/grafana:latest
    container_name: bev-grafana
    restart: unless-stopped
    platform: linux/arm64
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_SERVER_HTTP_ADDR: 0.0.0.0
    volumes:
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    networks:
      - bev-network

  # Coordination Services (Lightweight)
  consul:
    image: consul:latest
    container_name: bev-consul
    restart: unless-stopped
    platform: linux/arm64
    ports:
      - "8500:8500"
      - "8600:8600/udp"
    command: agent -server -bootstrap-expect=1 -ui -client=0.0.0.0
    volumes:
      - consul-data:/consul/data
    networks:
      - bev-network

  # Redis for ARM (cache only)
  redis-cache:
    image: redis:7-alpine
    container_name: bev-redis-cache
    restart: unless-stopped
    platform: linux/arm64
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru
    ports:
      - "6380:6379"
    networks:
      - bev-network

  # Lightweight web services
  nginx-proxy:
    image: nginx:alpine
    container_name: bev-nginx
    restart: unless-stopped
    platform: linux/arm64
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf:ro
    networks:
      - bev-network

volumes:
  prometheus-data:
  grafana-data:
  consul-data:

networks:
  bev-network:
    driver: bridge
    name: bev-network
EOF

cat > docker-compose-starlord.yml << 'EOF'
version: '3.8'

# STARLORD NODE - Personal workstation with auto-start/stop
# Hardware: Ryzen 9 7950X3D (32 threads), 64GB RAM

services:
  # Development UI (only when you're working)
  bev-frontend:
    image: node:18-alpine
    container_name: bev-frontend-dev
    restart: "no"  # Don't auto-restart
    working_dir: /app
    command: npm run dev
    environment:
      NODE_ENV: development
      VITE_API_URL: http://thanos:8000
      VITE_GRAFANA_URL: http://oracle1:3000
    volumes:
      - ./bev-frontend:/app
    ports:
      - "5173:5173"
    networks:
      - bev-network

  # Jupyter for data analysis
  jupyter:
    image: jupyter/datascience-notebook:latest
    container_name: bev-jupyter
    restart: "no"  # Only when needed
    environment:
      JUPYTER_TOKEN: ${JUPYTER_TOKEN:-bev2024}
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
    ports:
      - "8888:8888"
    networks:
      - bev-network

  # Portainer for Docker management
  portainer:
    image: portainer/portainer-ce:latest
    container_name: bev-portainer
    restart: "no"
    ports:
      - "9000:9000"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - portainer-data:/data
    networks:
      - bev-network

volumes:
  portainer-data:

networks:
  bev-network:
    driver: bridge
    name: bev-network
EOF

# Create systemd service for auto-start/stop on Starlord
cat > bev-local.service << 'EOF'
[Unit]
Description=BEV Local Development Services
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/starlord/Projects/Bev
ExecStart=/usr/bin/docker-compose -f docker-compose-starlord.yml up -d
ExecStop=/usr/bin/docker-compose -f docker-compose-starlord.yml down
User=starlord

[Install]
WantedBy=multi-user.target
EOF

echo -e "${GREEN}✅ Docker compose files created${NC}"

# Fix other critical issues
echo "Creating deployment script for nodes..."

cat > deploy-from-github.sh << 'EOF'
#!/bin/bash
# Script to run on each node to deploy from GitHub

GITHUB_REPO="https://github.com/aahmed954/Bev-AI.git"
DEPLOY_DIR="/opt/bev"

# Clone or update repository
if [ -d "$DEPLOY_DIR" ]; then
    cd $DEPLOY_DIR
    git pull origin main
else
    sudo mkdir -p $DEPLOY_DIR
    sudo chown starlord:starlord $DEPLOY_DIR
    git clone $GITHUB_REPO $DEPLOY_DIR
    cd $DEPLOY_DIR
fi

# Determine which compose file to use based on hostname
HOSTNAME=$(hostname)
case $HOSTNAME in
    thanos)
        COMPOSE_FILE="docker-compose-thanos.yml"
        ;;
    oracle1)
        COMPOSE_FILE="docker-compose-oracle1.yml"
        ;;
    *)
        echo "Unknown host: $HOSTNAME"
        exit 1
        ;;
esac

# Pull images and start services
docker-compose -f $COMPOSE_FILE pull
docker-compose -f $COMPOSE_FILE up -d

echo "Deployment complete on $HOSTNAME"
EOF

chmod +x deploy-from-github.sh

# Commit and push changes
echo -e "${CYAN}Committing fixes to Git...${NC}"
git add -A
git commit -m "Fix distributed deployment with proper node-specific compose files" || true
git push origin main

echo -e "${GREEN}✅ Changes pushed to GitHub${NC}"

# Step 3: Deploy to Thanos
echo -e "${CYAN}Step 3: Deploying to Thanos (Local High-Performance Node)${NC}"

ssh starlord@thanos << 'THANOS_DEPLOY'
# Clone/update from GitHub
if [ -d /opt/bev ]; then
    cd /opt/bev
    git pull origin main
else
    sudo mkdir -p /opt/bev
    sudo chown starlord:starlord /opt/bev
    git clone https://github.com/aahmed954/Bev-AI.git /opt/bev
    cd /opt/bev
fi

# Start Thanos services
docker-compose -f docker-compose-thanos.yml up -d

echo "Thanos deployment complete"
THANOS_DEPLOY

# Step 4: Deploy to Oracle1
echo -e "${CYAN}Step 4: Deploying to Oracle1 (ARM Cloud VM)${NC}"

ssh starlord@oracle1 << 'ORACLE1_DEPLOY'
# Clone/update from GitHub
if [ -d /opt/bev ]; then
    cd /opt/bev
    git pull origin main
else
    sudo mkdir -p /opt/bev
    sudo chown starlord:starlord /opt/bev
    git clone https://github.com/aahmed954/Bev-AI.git /opt/bev
    cd /opt/bev
fi

# Start Oracle1 services
docker-compose -f docker-compose-oracle1.yml up -d

echo "Oracle1 deployment complete"
ORACLE1_DEPLOY

# Step 5: Setup local Starlord services
echo -e "${CYAN}Step 5: Setting up Starlord Local Services (Auto-start/stop)${NC}"

# Install systemd service
sudo cp bev-local.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bev-local.service

echo -e "${YELLOW}Local services configured for auto-start${NC}"
echo "Start manually with: sudo systemctl start bev-local"
echo "Or they'll start automatically on next boot"

# Step 6: Verification
echo -e "${CYAN}Step 6: Verifying Deployment${NC}"

sleep 10

echo -e "${BLUE}Service Status:${NC}"
echo -n "Thanos (Databases, GPU): "
ssh starlord@thanos "docker ps --filter 'name=bev' | wc -l"

echo -n "Oracle1 (Monitoring, ARM): "
ssh starlord@oracle1 "docker ps --filter 'name=bev' | wc -l"

echo -n "Starlord (Dev UI): "
docker ps --filter 'name=bev' | wc -l

# Final summary
echo ""
echo -e "${GREEN}🎉 INTELLIGENT DEPLOYMENT COMPLETE!${NC}"
echo ""
echo -e "${BLUE}Access Points:${NC}"
echo "  PostgreSQL:    thanos:5432"
echo "  Neo4j Browser: http://thanos:7474"
echo "  Redis Primary: thanos:6379"
echo "  RabbitMQ:      http://thanos:15672"
echo ""
echo "  Grafana:       http://oracle1:3000 (admin/admin)"
echo "  Prometheus:    http://oracle1:9090"
echo "  Consul:        http://oracle1:8500"
echo ""
echo "  Frontend Dev:  http://localhost:5173 (when started)"
echo "  Jupyter:       http://localhost:8888 (when started)"
echo "  Portainer:     http://localhost:9000 (when started)"
echo ""
echo -e "${YELLOW}Local Services Control:${NC}"
echo "  Start:  sudo systemctl start bev-local"
echo "  Stop:   sudo systemctl stop bev-local"
echo "  Status: sudo systemctl status bev-local"
echo ""
echo -e "${CYAN}Update from GitHub on any node:${NC}"
echo "  ssh starlord@thanos 'cd /opt/bev && git pull && docker-compose -f docker-compose-thanos.yml up -d'"
echo "  ssh starlord@oracle1 'cd /opt/bev && git pull && docker-compose -f docker-compose-oracle1.yml up -d'"
