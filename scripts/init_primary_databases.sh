#!/bin/bash
# Primary Database Initialization Script for Thanos Node
# Initializes PostgreSQL, Neo4j, Elasticsearch, InfluxDB, and Vector DBs

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🗄️ INITIALIZING PRIMARY DATABASES ON THANOS${NC}"
echo "============================================="

# Wait for databases to be ready
echo -e "${YELLOW}⏳ Waiting for database services to be ready...${NC}"
sleep 30

# Initialize PostgreSQL
echo -e "${BLUE}📊 Initializing PostgreSQL...${NC}"
until docker exec bev_postgres pg_isready -U ${POSTGRES_USER:-researcher} > /dev/null 2>&1; do
    echo "Waiting for PostgreSQL to be ready..."
    sleep 5
done

# Create database schemas
docker exec bev_postgres psql -U ${POSTGRES_USER:-researcher} -d osint << 'PSQL_EOF'
-- Create BEV OSINT database schema
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- OSINT Tables
CREATE TABLE IF NOT EXISTS osint_analyses (
    id SERIAL PRIMARY KEY,
    target_type VARCHAR(50) NOT NULL,
    target_value TEXT NOT NULL,
    analyzer_name VARCHAR(100) NOT NULL,
    analysis_result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Vector embeddings table
CREATE TABLE IF NOT EXISTS document_embeddings (
    id SERIAL PRIMARY KEY,
    document_id TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Threat intelligence table
CREATE TABLE IF NOT EXISTS threat_intel (
    id SERIAL PRIMARY KEY,
    ioc_type VARCHAR(50) NOT NULL,
    ioc_value TEXT NOT NULL,
    threat_score INTEGER,
    source VARCHAR(100),
    tags TEXT[],
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indices for performance
CREATE INDEX IF NOT EXISTS idx_osint_analyses_target ON osint_analyses(target_type, target_value);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_hash ON document_embeddings(content_hash);
CREATE INDEX IF NOT EXISTS idx_threat_intel_ioc ON threat_intel(ioc_type, ioc_value);

-- Create vector similarity index
CREATE INDEX IF NOT EXISTS idx_document_embeddings_vector ON document_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

PSQL_EOF

echo -e "${GREEN}✅ PostgreSQL initialized${NC}"

# Initialize Neo4j
echo -e "${BLUE}🕸️ Initializing Neo4j...${NC}"
until docker exec bev_neo4j cypher-shell -u neo4j -p ${NEO4J_PASSWORD:-BevGraphMaster2024} "RETURN 1" > /dev/null 2>&1; do
    echo "Waiting for Neo4j to be ready..."
    sleep 5
done

# Create Neo4j constraints and indices
docker exec bev_neo4j cypher-shell -u neo4j -p ${NEO4J_PASSWORD:-BevGraphMaster2024} << 'CYPHER_EOF'
// Create constraints
CREATE CONSTRAINT unique_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT unique_analysis_id IF NOT EXISTS FOR (a:Analysis) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT unique_threat_id IF NOT EXISTS FOR (t:Threat) REQUIRE t.id IS UNIQUE;

// Create indices
CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX analysis_timestamp_index IF NOT EXISTS FOR (a:Analysis) ON (a.timestamp);
CREATE INDEX threat_score_index IF NOT EXISTS FOR (t:Threat) ON (t.score);

// Create sample nodes
MERGE (sys:System {name: "BEV OSINT Framework", version: "2.0", type: "Intelligence Platform"});
MERGE (admin:User {username: "admin", role: "system_administrator", created: datetime()});
MERGE (sys)-[:ADMINISTERED_BY]->(admin);
CYPHER_EOF

echo -e "${GREEN}✅ Neo4j initialized${NC}"

# Initialize Elasticsearch
echo -e "${BLUE}🔍 Initializing Elasticsearch...${NC}"
until curl -s http://localhost:9200/_cluster/health > /dev/null 2>&1; do
    echo "Waiting for Elasticsearch to be ready..."
    sleep 5
done

# Create Elasticsearch indices
curl -X PUT "localhost:9200/bev-osint-logs" -H 'Content-Type: application/json' -d'
{
  "mappings": {
    "properties": {
      "timestamp": {"type": "date"},
      "service": {"type": "keyword"},
      "level": {"type": "keyword"},
      "message": {"type": "text"},
      "metadata": {"type": "object"}
    }
  },
  "settings": {
    "number_of_shards": 2,
    "number_of_replicas": 1
  }
}'

curl -X PUT "localhost:9200/bev-threat-intel" -H 'Content-Type: application/json' -d'
{
  "mappings": {
    "properties": {
      "ioc_type": {"type": "keyword"},
      "ioc_value": {"type": "keyword"},
      "threat_score": {"type": "integer"},
      "source": {"type": "keyword"},
      "tags": {"type": "keyword"},
      "timestamp": {"type": "date"},
      "metadata": {"type": "object"}
    }
  }
}'

echo -e "${GREEN}✅ Elasticsearch initialized${NC}"

# Initialize InfluxDB
echo -e "${BLUE}⏱️ Initializing InfluxDB...${NC}"
until curl -s http://localhost:8086/health > /dev/null 2>&1; do
    echo "Waiting for InfluxDB to be ready..."
    sleep 5
done

# Create InfluxDB buckets
docker exec bev_influxdb influx bucket create \
  --name bev-metrics \
  --org bev-osint \
  --retention 30d \
  --token ${INFLUXDB_TOKEN:-dev-token} > /dev/null 2>&1 || echo "Bucket may already exist"

docker exec bev_influxdb influx bucket create \
  --name bev-performance \
  --org bev-osint \
  --retention 7d \
  --token ${INFLUXDB_TOKEN:-dev-token} > /dev/null 2>&1 || echo "Bucket may already exist"

echo -e "${GREEN}✅ InfluxDB initialized${NC}"

# Initialize Qdrant Vector Database
echo -e "${BLUE}🧠 Initializing Qdrant Vector Database...${NC}"
until curl -s http://localhost:6333/health > /dev/null 2>&1; do
    echo "Waiting for Qdrant to be ready..."
    sleep 5
done

# Create Qdrant collections
curl -X PUT "localhost:6333/collections/bev-documents" \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 1536,
      "distance": "Cosine"
    },
    "optimizers_config": {
      "default_segment_number": 2
    }
  }'

curl -X PUT "localhost:6333/collections/bev-embeddings" \
  -H 'Content-Type: application/json' \
  -d '{
    "vectors": {
      "size": 768,
      "distance": "Cosine"
    }
  }'

echo -e "${GREEN}✅ Qdrant initialized${NC}"

# Initialize Weaviate
echo -e "${BLUE}🕷️ Initializing Weaviate...${NC}"
until curl -s http://localhost:8080/v1/meta > /dev/null 2>&1; do
    echo "Waiting for Weaviate to be ready..."
    sleep 5
done

# Create Weaviate schema
curl -X POST "localhost:8080/v1/schema" \
  -H 'Content-Type: application/json' \
  -d '{
    "class": "Document",
    "vectorizer": "text2vec-transformers",
    "properties": [
      {"name": "title", "dataType": ["string"]},
      {"name": "content", "dataType": ["text"]},
      {"name": "source", "dataType": ["string"]},
      {"name": "timestamp", "dataType": ["date"]}
    ]
  }' > /dev/null 2>&1 || echo "Schema may already exist"

echo -e "${GREEN}✅ Weaviate initialized${NC}"

# Verify all database connections
echo -e "${BLUE}🏥 Running database health checks...${NC}"

HEALTH_CHECKS=(
    "PostgreSQL:docker exec bev_postgres pg_isready"
    "Neo4j:docker exec bev_neo4j cypher-shell -u neo4j -p ${NEO4J_PASSWORD:-BevGraphMaster2024} 'RETURN 1'"
    "Redis:docker exec bev_redis redis-cli ping"
    "Elasticsearch:curl -s http://localhost:9200/_cluster/health"
    "InfluxDB:curl -s http://localhost:8086/health"
    "Qdrant:curl -s http://localhost:6333/health"
    "Weaviate:curl -s http://localhost:8080/v1/meta"
)

HEALTHY_DBS=0
TOTAL_DBS=${#HEALTH_CHECKS[@]}

for check in "${HEALTH_CHECKS[@]}"; do
    DB_NAME=$(echo $check | cut -d: -f1)
    DB_CHECK=$(echo $check | cut -d: -f2-)

    echo -n "Health check $DB_NAME... "
    if eval $DB_CHECK > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Healthy${NC}"
        HEALTHY_DBS=$((HEALTHY_DBS + 1))
    else
        echo -e "${RED}❌ Unhealthy${NC}"
    fi
done

# Calculate health percentage
HEALTH_PERCENTAGE=$((HEALTHY_DBS * 100 / TOTAL_DBS))
echo ""
echo -e "${BLUE}📊 Database Health Summary:${NC}"
echo "Healthy Databases: $HEALTHY_DBS/$TOTAL_DBS ($HEALTH_PERCENTAGE%)"

if [ $HEALTH_PERCENTAGE -ge 85 ]; then
    echo -e "${GREEN}🎯 Database initialization successful!${NC}"
    exit 0
else
    echo -e "${RED}❌ Database initialization issues detected${NC}"
    exit 1
fi