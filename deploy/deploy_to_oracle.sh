#!/bin/bash
# Deploy BEV OSINT Services to Oracle1-Cloud
# IP: 100.96.197.84 (ARM Architecture)

set -e

echo "🔮 Deploying BEV OSINT to Oracle1-Cloud"

ORACLE_HOST="100.96.197.84"
ORACLE_USER="bev"
DEPLOY_DIR="/opt/bev-osint"

# ARM-specific build
echo "Building ARM-compatible containers..."

# Create ARM-specific docker-compose
cat > docker-compose-oracle-arm.yml << 'YAML'
version: '3.8'

services:
  # Research Coordinator (ARM-optimized)
  research-coordinator:
    build:
      context: .
      dockerfile: docker/agents/Dockerfile.research.arm
    image: bev/research-coordinator:arm64
    container_name: bev_research_oracle
    environment:
      - AGENT_ID=research_oracle_1
      - REDIS_URL=redis://redis:6379
      - POSTGRES_URL=postgresql://swarm_admin:swarm_password@postgres:5432/ai_swarm
    volumes:
      - ./src/agents:/app/agents
      - research_data:/data
    networks:
      - bev_network
    restart: unless-stopped

  # OSINT Tools
  intelowl:
    image: intelowlproject/intelowl:arm64
    container_name: bev_intelowl
    environment:
      - DJANGO_SECRET_KEY=${DJANGO_SECRET_KEY}
      - DB_HOST=postgres
      - BROKER_URL=redis://redis:6379
    ports:
      - "8001:80"
    volumes:
      - intelowl_data:/opt/deploy/intel_owl/files
    networks:
      - bev_network

  # Tor Proxy for Dark Web
  tor-proxy:
    image: dperson/torproxy:arm64
    container_name: bev_tor_proxy
    ports:
      - "9050:9050"
      - "9051:9051"
    networks:
      - bev_network
    restart: unless-stopped

  # Spider for web crawling
  spider:
    build:
      context: .
      dockerfile: docker/crawler/Dockerfile.arm
    container_name: bev_spider
    environment:
      - TOR_PROXY=socks5://tor-proxy:9050
      - REDIS_URL=redis://redis:6379
    volumes:
      - crawl_data:/data/crawls
    networks:
      - bev_network

  # Elasticsearch for OSINT data
  elasticsearch:
    image: elasticsearch:8.11.0
    container_name: bev_elasticsearch_oracle
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms2g -Xmx2g"
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data
    networks:
      - bev_network

  # Kibana for visualization
  kibana:
    image: kibana:8.11.0
    container_name: bev_kibana_oracle
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    networks:
      - bev_network

volumes:
  research_data:
  intelowl_data:
  crawl_data:
  es_data:

networks:
  bev_network:
    driver: bridge
YAML

# Build ARM images
docker buildx build --platform linux/arm64 \
    -f docker/agents/Dockerfile.research.arm \
    -t bev/research-coordinator:arm64 .

# Deploy to Oracle
scp docker-compose-oracle-arm.yml $ORACLE_USER@$ORACLE_HOST:/tmp/
scp -r src/agents $ORACLE_USER@$ORACLE_HOST:/tmp/
scp -r src/alternative_market $ORACLE_USER@$ORACLE_HOST:/tmp/

ssh $ORACLE_USER@$ORACLE_HOST << 'EOF'
set -e

# Setup deployment directory
sudo mkdir -p /opt/bev-osint
sudo chown oracle:oracle /opt/bev-osint
cd /opt/bev-osint

# Copy files
mv /tmp/docker-compose-oracle-arm.yml docker-compose.yml
mv /tmp/agents ./src/
mv /tmp/alternative_market ./src/

# Start services
docker-compose up -d

# Setup cron jobs for OSINT collection
(crontab -l 2>/dev/null; echo "0 */6 * * * /opt/bev-osint/scripts/collect_osint.sh") | crontab -

echo "✅ Oracle1 OSINT deployment complete!"
EOF

echo "✅ BEV OSINT successfully deployed to Oracle1-Cloud!"
echo "Access points:"
echo "  - IntelOwl: http://$ORACLE_HOST:8001"
echo "  - Elasticsearch: http://$ORACLE_HOST:9200"
echo "  - Kibana: http://$ORACLE_HOST:5601"
