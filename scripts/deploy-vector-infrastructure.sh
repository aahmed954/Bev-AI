#!/bin/bash
"""
Vector Database Infrastructure Deployment Script
Deploy comprehensive vector database infrastructure for BEV OSINT
Author: BEV OSINT Team
"""

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.complete.yml"
ENV_FILE="$PROJECT_ROOT/.env"

# Logging
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $*" >&2
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "🔍 Checking prerequisites..."

    # Check Docker and Docker Compose
    if ! command -v docker &> /dev/null; then
        error "Docker not found. Please install Docker."
    fi

    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose not found. Please install Docker Compose."
    fi

    # Check available resources
    AVAILABLE_MEMORY=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $AVAILABLE_MEMORY -lt 16 ]]; then
        log "⚠️  Warning: Only ${AVAILABLE_MEMORY}GB RAM available. Vector databases require at least 16GB for optimal performance."
    fi

    # Check disk space
    AVAILABLE_DISK=$(df -BG "$PROJECT_ROOT" | awk 'NR==2{gsub(/G/, "", $4); print $4}')
    if [[ $AVAILABLE_DISK -lt 50 ]]; then
        error "Insufficient disk space. Need at least 50GB, only ${AVAILABLE_DISK}GB available."
    fi

    log "✅ Prerequisites check passed"
}

# Setup environment variables
setup_environment() {
    log "🔧 Setting up environment variables..."

    # Create .env file if it doesn't exist
    if [[ ! -f "$ENV_FILE" ]]; then
        log "Creating .env file..."
        cat > "$ENV_FILE" << EOF
# BEV OSINT Environment Configuration

# Database Configuration
POSTGRES_USER=bev_user
POSTGRES_PASSWORD=$(openssl rand -hex 16)
POSTGRES_DB=osint
NEO4J_USER=neo4j
NEO4J_PASSWORD=$(openssl rand -hex 16)
REDIS_PASSWORD=$(openssl rand -hex 16)

# Vector Database Configuration
WEAVIATE_API_KEY=$(openssl rand -hex 32)
QDRANT_API_KEY=$(openssl rand -hex 32)

# Security Configuration
ENCRYPTION_KEY=$(openssl rand -base64 32)
JWT_SECRET=$(openssl rand -hex 32)

# Performance Configuration
BATCH_SIZE=32
MAX_WORKERS=16
CACHE_TTL=3600

# Monitoring Configuration
PROMETHEUS_RETENTION=15d
GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 16)

# Vector Database Hosts
QDRANT_PRIMARY_HOST=172.30.0.36
QDRANT_REPLICA_HOST=172.30.0.37
WEAVIATE_HOST=172.30.0.38

# API Configuration
VECTOR_DB_MANAGER_PORT=8080
EMBEDDING_PIPELINE_PORT=8081
EOF
        log "✅ Environment file created"
    else
        log "✅ Environment file exists"
    fi

    # Source environment variables
    set -a
    source "$ENV_FILE"
    set +a
}

# Create required directories
create_directories() {
    log "📁 Creating required directories..."

    local dirs=(
        "$PROJECT_ROOT/data/qdrant"
        "$PROJECT_ROOT/data/weaviate"
        "$PROJECT_ROOT/data/vector_backups"
        "$PROJECT_ROOT/logs/vector_db"
        "$PROJECT_ROOT/config/vector_db"
        "$PROJECT_ROOT/benchmark_results"
    )

    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        log "Created directory: $dir"
    done

    # Set appropriate permissions
    chmod 755 "${dirs[@]}"
    log "✅ Directories created and permissions set"
}

# Deploy vector database services
deploy_vector_services() {
    log "🚀 Deploying vector database services..."

    # Pull required images
    log "📥 Pulling Docker images..."
    docker-compose -f "$COMPOSE_FILE" pull qdrant-primary qdrant-replica weaviate t2v-transformers

    # Start vector database services
    log "🔄 Starting vector database services..."
    docker-compose -f "$COMPOSE_FILE" up -d qdrant-primary qdrant-replica weaviate t2v-transformers

    # Wait for services to be ready
    log "⏳ Waiting for services to be ready..."
    wait_for_service "http://${QDRANT_PRIMARY_HOST}:6333/health" "Qdrant Primary"
    wait_for_service "http://${QDRANT_REPLICA_HOST}:6336/health" "Qdrant Replica"
    wait_for_service "http://${WEAVIATE_HOST}:8080/v1/.well-known/ready" "Weaviate"

    log "✅ Vector database services deployed successfully"
}

# Wait for service to be ready
wait_for_service() {
    local url="$1"
    local service_name="$2"
    local max_attempts=30
    local attempt=1

    log "Waiting for $service_name to be ready..."

    while [[ $attempt -le $max_attempts ]]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            log "✅ $service_name is ready"
            return 0
        fi

        log "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 10
        ((attempt++))
    done

    error "$service_name failed to start within $(($max_attempts * 10)) seconds"
}

# Initialize vector collections
initialize_collections() {
    log "🗄️  Initializing vector collections..."

    # Create Qdrant collections
    create_qdrant_collection "osint_intel" 768
    create_qdrant_collection "threat_indicators" 384
    create_qdrant_collection "social_media" 768
    create_qdrant_collection "dark_web" 768

    # Create Weaviate schemas
    python3 << 'EOF'
import requests
import json

weaviate_url = "http://172.30.0.38:8080"

# OSINT Intelligence schema
osint_schema = {
    "class": "OSINTIntel",
    "description": "OSINT intelligence documents",
    "vectorizer": "text2vec-transformers",
    "properties": [
        {"name": "content", "dataType": ["text"]},
        {"name": "source", "dataType": ["string"]},
        {"name": "timestamp", "dataType": ["date"]},
        {"name": "classification", "dataType": ["string"]},
        {"name": "confidence", "dataType": ["number"]},
        {"name": "tags", "dataType": ["string[]"]},
    ]
}

# Threat Indicator schema
threat_schema = {
    "class": "ThreatIndicator",
    "description": "Threat indicators and IOCs",
    "vectorizer": "text2vec-transformers",
    "properties": [
        {"name": "indicator", "dataType": ["text"]},
        {"name": "type", "dataType": ["string"]},
        {"name": "malware_family", "dataType": ["string"]},
        {"name": "severity", "dataType": ["string"]},
        {"name": "first_seen", "dataType": ["date"]},
        {"name": "last_seen", "dataType": ["date"]},
        {"name": "sources", "dataType": ["string[]"]},
    ]
}

# Create schemas
for schema in [osint_schema, threat_schema]:
    try:
        response = requests.post(f"{weaviate_url}/v1/schema", json=schema)
        if response.status_code in [200, 422]:  # 422 = already exists
            print(f"✅ Created schema: {schema['class']}")
        else:
            print(f"❌ Failed to create schema {schema['class']}: {response.text}")
    except Exception as e:
        print(f"❌ Error creating schema {schema['class']}: {e}")
EOF

    log "✅ Vector collections initialized"
}

# Create Qdrant collection
create_qdrant_collection() {
    local collection_name="$1"
    local vector_size="$2"

    log "Creating Qdrant collection: $collection_name (size: $vector_size)"

    curl -X PUT "http://${QDRANT_PRIMARY_HOST}:6333/collections/$collection_name" \
        -H "Content-Type: application/json" \
        -d "{
            \"vectors\": {
                \"size\": $vector_size,
                \"distance\": \"Cosine\"
            },
            \"optimizers_config\": {
                \"default_segment_number\": 2
            },
            \"replication_factor\": 2
        }" || log "⚠️  Collection $collection_name may already exist"
}

# Setup monitoring
setup_monitoring() {
    log "📊 Setting up monitoring..."

    # Update Prometheus configuration
    log "Updating Prometheus configuration..."

    # Start monitoring services if not already running
    docker-compose -f "$COMPOSE_FILE" up -d prometheus grafana

    wait_for_service "http://localhost:9090/-/ready" "Prometheus"
    wait_for_service "http://localhost:3000/api/health" "Grafana"

    log "✅ Monitoring setup complete"
}

# Validate deployment
validate_deployment() {
    log "🔍 Validating deployment..."

    local errors=0

    # Check Qdrant cluster
    if ! curl -s -f "http://${QDRANT_PRIMARY_HOST}:6333/health" > /dev/null; then
        log "❌ Qdrant primary health check failed"
        ((errors++))
    fi

    if ! curl -s -f "http://${QDRANT_REPLICA_HOST}:6336/health" > /dev/null; then
        log "❌ Qdrant replica health check failed"
        ((errors++))
    fi

    # Check Weaviate
    if ! curl -s -f "http://${WEAVIATE_HOST}:8080/v1/.well-known/ready" > /dev/null; then
        log "❌ Weaviate health check failed"
        ((errors++))
    fi

    # Check collections
    local qdrant_collections=$(curl -s "http://${QDRANT_PRIMARY_HOST}:6333/collections" | jq -r '.result.collections[].name' 2>/dev/null | wc -l)
    if [[ $qdrant_collections -lt 4 ]]; then
        log "❌ Expected 4 Qdrant collections, found $qdrant_collections"
        ((errors++))
    fi

    # Test vector insertion
    log "Testing vector insertion..."
    local test_result=$(curl -s -X PUT "http://${QDRANT_PRIMARY_HOST}:6333/collections/osint_intel/points" \
        -H "Content-Type: application/json" \
        -d '{
            "points": [{
                "id": "test-001",
                "vector": [0.1, 0.2, 0.3, 0.4],
                "payload": {"test": true, "timestamp": "'$(date -Iseconds)'"}
            }]
        }' | jq -r '.status' 2>/dev/null)

    if [[ "$test_result" != "ok" ]]; then
        log "❌ Vector insertion test failed"
        ((errors++))
    else
        log "✅ Vector insertion test passed"
    fi

    # Test vector search
    log "Testing vector search..."
    local search_result=$(curl -s -X POST "http://${QDRANT_PRIMARY_HOST}:6333/collections/osint_intel/points/search" \
        -H "Content-Type: application/json" \
        -d '{
            "vector": [0.1, 0.2, 0.3, 0.4],
            "limit": 1
        }' | jq -r '.status' 2>/dev/null)

    if [[ "$search_result" != "ok" ]]; then
        log "❌ Vector search test failed"
        ((errors++))
    else
        log "✅ Vector search test passed"
    fi

    if [[ $errors -eq 0 ]]; then
        log "✅ Deployment validation passed"
        return 0
    else
        log "❌ Deployment validation failed with $errors errors"
        return 1
    fi
}

# Run performance benchmark
run_benchmark() {
    log "🏁 Running performance benchmark..."

    if [[ -f "$PROJECT_ROOT/src/infrastructure/performance_benchmarks.py" ]]; then
        cd "$PROJECT_ROOT"
        python3 -c "
import asyncio
import sys
sys.path.append('src')
from infrastructure.performance_benchmarks import main
asyncio.run(main())
        " || log "⚠️  Benchmark failed, but deployment continues"
    else
        log "⚠️  Benchmark script not found, skipping"
    fi
}

# Cleanup function
cleanup() {
    log "🧹 Cleaning up temporary files..."
    # Add any cleanup logic here
}

# Main deployment function
main() {
    log "🚀 Starting BEV OSINT Vector Database Infrastructure Deployment"

    # Set up trap for cleanup
    trap cleanup EXIT

    # Execute deployment steps
    check_prerequisites
    setup_environment
    create_directories
    deploy_vector_services
    initialize_collections
    setup_monitoring

    # Validate deployment
    if validate_deployment; then
        log "✅ Vector database infrastructure deployed successfully!"

        # Optional benchmark
        if [[ "${RUN_BENCHMARK:-false}" == "true" ]]; then
            run_benchmark
        fi

        # Display connection information
        log "📋 Connection Information:"
        log "  Qdrant Primary: http://${QDRANT_PRIMARY_HOST}:6333"
        log "  Qdrant Replica: http://${QDRANT_REPLICA_HOST}:6336"
        log "  Weaviate: http://${WEAVIATE_HOST}:8080"
        log "  Prometheus: http://localhost:9090"
        log "  Grafana: http://localhost:3000"

        log "🎯 Performance Target: 10K+ embeddings per minute"
        log "📊 Monitor performance via Grafana dashboards"
        log "🔧 Configuration files in: $PROJECT_ROOT/config/"

    else
        error "Deployment validation failed. Please check logs and services."
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi