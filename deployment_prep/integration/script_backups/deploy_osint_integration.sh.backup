#!/bin/bash

# Deploy OSINT Integration Layer for BEV Framework
# This script sets up the integration between OSINT analyzers and the Avatar system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==============================================="
echo "BEV OSINT Integration Layer Deployment"
echo "==============================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi

# Check GPU support for avatar system
echo -e "${YELLOW}Checking GPU support...${NC}"
if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    GPU_AVAILABLE=true
else
    echo -e "${YELLOW}⚠ No NVIDIA GPU detected - Avatar rendering will be limited${NC}"
    GPU_AVAILABLE=false
fi

# Create necessary directories
echo -e "${YELLOW}Creating required directories...${NC}"
mkdir -p logs/osint-integration logs/avatar models/metahuman data/investigations

# Check if core services are running
echo -e "${YELLOW}Checking core services...${NC}"

check_service() {
    local service=$1
    local container=$2

    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo -e "${GREEN}✓ ${service} is running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠ ${service} is not running${NC}"
        return 1
    fi
}

# Check required services
SERVICES_OK=true

check_service "PostgreSQL" "bev_postgres" || SERVICES_OK=false
check_service "Neo4j" "bev_neo4j" || SERVICES_OK=false
check_service "Redis" "bev_redis" || SERVICES_OK=false
check_service "Qdrant" "bev_qdrant" || SERVICES_OK=false

if [ "$SERVICES_OK" = false ]; then
    echo -e "${YELLOW}Starting required services...${NC}"
    docker-compose -f docker-compose.osint-integration.yml up -d postgres neo4j redis qdrant

    # Wait for services to be ready
    echo -e "${YELLOW}Waiting for services to be ready...${NC}"
    sleep 10
fi

# Start message queue services
echo -e "${YELLOW}Starting message queue services...${NC}"
docker-compose -f docker-compose.osint-integration.yml up -d zookeeper kafka rabbitmq nats

# Wait for message queues
echo -e "${YELLOW}Waiting for message queues to initialize...${NC}"
sleep 15

# Build and start the integration layer
echo -e "${YELLOW}Building OSINT Integration Layer...${NC}"
docker-compose -f docker-compose.osint-integration.yml build osint-integration

echo -e "${YELLOW}Starting OSINT Integration Layer...${NC}"
docker-compose -f docker-compose.osint-integration.yml up -d osint-integration

# Build and start avatar system if GPU available
if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${YELLOW}Building Advanced Avatar System...${NC}"
    docker-compose -f docker-compose.osint-integration.yml build avatar-system

    echo -e "${YELLOW}Starting Advanced Avatar System...${NC}"
    docker-compose -f docker-compose.osint-integration.yml up -d avatar-system
else
    echo -e "${YELLOW}Skipping Avatar System (no GPU)${NC}"
fi

# Wait for services to be healthy
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"
sleep 10

# Health check function
health_check() {
    local url=$1
    local service=$2

    if curl -f -s "${url}" > /dev/null; then
        echo -e "${GREEN}✓ ${service} is healthy${NC}"
        return 0
    else
        echo -e "${RED}✗ ${service} health check failed${NC}"
        return 1
    fi
}

# Check service health
echo -e "${YELLOW}Performing health checks...${NC}"

HEALTH_OK=true
health_check "http://localhost:8092/health" "OSINT Integration Layer" || HEALTH_OK=false

if [ "$GPU_AVAILABLE" = true ]; then
    health_check "http://localhost:8091/health" "Avatar System" || HEALTH_OK=false
fi

# Create Kafka topics
echo -e "${YELLOW}Creating Kafka topics...${NC}"
docker exec bev_kafka kafka-topics --create --topic osint-events \
    --bootstrap-server localhost:9092 \
    --partitions 3 \
    --replication-factor 1 \
    --if-not-exists 2>/dev/null || true

# Create RabbitMQ exchanges
echo -e "${YELLOW}Configuring RabbitMQ...${NC}"
docker exec bev_rabbitmq rabbitmqctl add_vhost osint 2>/dev/null || true

# Initialize database schemas
echo -e "${YELLOW}Initializing database schemas...${NC}"

# PostgreSQL schema
cat << 'EOF' | docker exec -i bev_postgres psql -U researcher -d osint
CREATE TABLE IF NOT EXISTS osint_investigations (
    investigation_id VARCHAR(100) PRIMARY KEY,
    investigation_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    findings JSONB,
    threat_indicators JSONB,
    progress FLOAT DEFAULT 0.0,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS osint_events (
    event_id VARCHAR(100) PRIMARY KEY,
    investigation_id VARCHAR(100) REFERENCES osint_investigations(investigation_id),
    event_type VARCHAR(50) NOT NULL,
    threat_level INTEGER DEFAULT 0,
    confidence FLOAT DEFAULT 0.5,
    data JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_investigation_status ON osint_investigations(status);
CREATE INDEX IF NOT EXISTS idx_event_investigation ON osint_events(investigation_id);
CREATE INDEX IF NOT EXISTS idx_event_type ON osint_events(event_type);
CREATE INDEX IF NOT EXISTS idx_event_timestamp ON osint_events(timestamp);
EOF

# Neo4j constraints
echo -e "${YELLOW}Setting up Neo4j constraints...${NC}"
docker exec bev_neo4j cypher-shell -u neo4j -p BevGraphMaster2024 \
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;" 2>/dev/null || true

# Print service URLs
echo ""
echo "==============================================="
echo -e "${GREEN}OSINT Integration Layer Deployed Successfully!${NC}"
echo "==============================================="
echo ""
echo "Service URLs:"
echo "  OSINT Integration API: http://localhost:8092"
echo "  Integration WebSocket: ws://localhost:8092/ws"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "  Avatar System: http://localhost:8091"
    echo "  Avatar WebSocket: ws://localhost:8091/ws"
fi
echo ""
echo "Message Queue UIs:"
echo "  RabbitMQ Management: http://localhost:15672 (guest/guest)"
echo "  Kafka (via docker logs): docker logs bev_kafka"
echo ""
echo "Database Access:"
echo "  PostgreSQL: localhost:5432 (researcher/osint_research_2024)"
echo "  Neo4j Browser: http://localhost:7474 (neo4j/BevGraphMaster2024)"
echo "  Redis: localhost:6379"
echo "  Qdrant: http://localhost:6333"
echo ""

# Test the integration
echo -e "${YELLOW}Testing OSINT Integration...${NC}"

# Test investigation start
RESPONSE=$(curl -s -X POST http://localhost:8092/investigation/start \
    -H "Content-Type: application/json" \
    -d '{
        "investigation_type": "breach_database",
        "target": "test@example.com"
    }' 2>/dev/null || echo "{}")

if echo "$RESPONSE" | grep -q "investigation_id"; then
    echo -e "${GREEN}✓ Integration test successful${NC}"
    INVESTIGATION_ID=$(echo "$RESPONSE" | grep -o '"investigation_id":"[^"]*' | cut -d'"' -f4)
    echo "  Test investigation ID: $INVESTIGATION_ID"
else
    echo -e "${YELLOW}⚠ Integration test could not be verified${NC}"
fi

# View logs
echo ""
echo "To view logs:"
echo "  docker-compose -f docker-compose.osint-integration.yml logs -f osint-integration"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "  docker-compose -f docker-compose.osint-integration.yml logs -f avatar-system"
fi

echo ""
echo -e "${GREEN}Deployment complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Start an OSINT investigation via the API"
echo "2. Monitor avatar responses in real-time via WebSocket"
echo "3. View investigation progress and correlations"
echo "4. Generate threat reports from findings"