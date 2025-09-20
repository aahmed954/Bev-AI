#!/bin/bash

#################################################################
# BEV OSINT Framework - Data Core Node Deployment Script
#
# This script deploys the foundational data storage services
# for the BEV OSINT distributed framework.
#
# Requirements: 32+ GB RAM, Fast SSD, High I/O
#################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_TYPE="data-core"
NODE_NAME="${BEV_NODE_ID:-data-core-01}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_warning() {
    log "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites for Data Core node..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available"
        exit 1
    fi
    
    # Check available memory (require at least 24GB)
    local available_memory=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $available_memory -lt 24 ]]; then
        log_error "Insufficient memory. Data Core requires at least 24GB RAM, found ${available_memory}GB"
        exit 1
    fi
    
    # Check available disk space (require at least 100GB)
    local available_disk=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $available_disk -lt 100 ]]; then
        log_error "Insufficient disk space. Data Core requires at least 100GB, found ${available_disk}GB"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Setup environment
setup_environment() {
    log_info "Setting up environment for Data Core node..."
    
    # Create .env file if it doesn't exist
    if [[ ! -f "$SCRIPT_DIR/.env" ]]; then
        if [[ -f "$SCRIPT_DIR/.env.template" ]]; then
            log_info "Creating .env file from template..."
            cp "$SCRIPT_DIR/.env.template" "$SCRIPT_DIR/.env"
            log_warning "Please edit .env file with your specific configuration"
            log_warning "Pay special attention to passwords and cluster coordination settings"
        else
            log_error ".env.template not found. Cannot create environment configuration."
            exit 1
        fi
    fi
    
    # Source environment variables
    source "$SCRIPT_DIR/.env"
    
    # Create required directories
    local dirs=(
        "./init_scripts"
        "./redis/node1"
        "./redis/node2" 
        "./redis/node3"
        "./backups"
        "./logs"
        "./certs"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$SCRIPT_DIR/$dir"
    done
    
    log_success "Environment setup completed"
}

# Initialize database schemas
setup_database_schemas() {
    log_info "Setting up database initialization scripts..."
    
    # PostgreSQL initialization script
    cat > "$SCRIPT_DIR/init_scripts/postgres_init.sql" << 'EOF'
-- BEV OSINT Framework - PostgreSQL Initialization
-- This script sets up the required databases and extensions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create main OSINT database structure
\c osint;

-- OSINT Results table
CREATE TABLE IF NOT EXISTS osint_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    target VARCHAR(255) NOT NULL,
    analyzer VARCHAR(100) NOT NULL,
    result_type VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    embedding vector(1536),
    confidence DECIMAL(3,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Threat Indicators table
CREATE TABLE IF NOT EXISTS threat_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    indicator_type VARCHAR(50) NOT NULL,
    indicator_value VARCHAR(500) NOT NULL,
    threat_type VARCHAR(100),
    confidence DECIMAL(3,2),
    source VARCHAR(100),
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags TEXT[],
    metadata JSONB
);

-- Analysis Sessions table
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_name VARCHAR(200),
    analyst_id VARCHAR(100),
    status VARCHAR(50) DEFAULT 'active',
    targets JSONB,
    results_summary JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_osint_results_target ON osint_results(target);
CREATE INDEX IF NOT EXISTS idx_osint_results_analyzer ON osint_results(analyzer);
CREATE INDEX IF NOT EXISTS idx_osint_results_created_at ON osint_results(created_at);
CREATE INDEX IF NOT EXISTS idx_threat_indicators_type ON threat_indicators(indicator_type);
CREATE INDEX IF NOT EXISTS idx_threat_indicators_value ON threat_indicators(indicator_value);

-- IntelOwl database setup
\c intelowl;
-- IntelOwl will handle its own schema initialization

-- Breach data database
\c breach_data;
CREATE TABLE IF NOT EXISTS breach_records (
    id BIGSERIAL PRIMARY KEY,
    email VARCHAR(255),
    username VARCHAR(255),
    password_hash VARCHAR(255),
    source VARCHAR(100),
    breach_date DATE,
    discovered_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_breach_email ON breach_records(email);
CREATE INDEX IF NOT EXISTS idx_breach_username ON breach_records(username);

-- Crypto analysis database
\c crypto_analysis;
CREATE TABLE IF NOT EXISTS crypto_transactions (
    id BIGSERIAL PRIMARY KEY,
    transaction_hash VARCHAR(100) UNIQUE,
    blockchain VARCHAR(20),
    from_address VARCHAR(100),
    to_address VARCHAR(100),
    amount DECIMAL(20,8),
    transaction_date TIMESTAMP WITH TIME ZONE,
    risk_score DECIMAL(3,2),
    analyzed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_crypto_hash ON crypto_transactions(transaction_hash);
CREATE INDEX IF NOT EXISTS idx_crypto_addresses ON crypto_transactions(from_address, to_address);
EOF

    # Neo4j initialization script
    cat > "$SCRIPT_DIR/init_scripts/neo4j_init.cypher" << 'EOF'
// BEV OSINT Framework - Neo4j Initialization
// This script sets up the required node types and relationships

// Create constraints for unique identifiers
CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT analysis_id IF NOT EXISTS FOR (a:Analysis) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT threat_id IF NOT EXISTS FOR (t:Threat) REQUIRE t.id IS UNIQUE;
CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE;

// Create indexes for performance
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_value IF NOT EXISTS FOR (e:Entity) ON (e.value);
CREATE INDEX analysis_timestamp IF NOT EXISTS FOR (a:Analysis) ON (a.timestamp);
CREATE INDEX threat_type IF NOT EXISTS FOR (t:Threat) ON (t.type);

// Sample data structure
MERGE (source:Source {id: 'bev-osint', name: 'BEV OSINT Framework', type: 'analysis_platform'});
EOF
    
    log_success "Database initialization scripts created"
}

# Start services
start_services() {
    log_info "Starting Data Core services..."
    
    # Pull latest images
    log_info "Pulling latest Docker images..."
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" pull
    
    # Start services
    log_info "Starting services with docker-compose..."
    docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to become healthy..."
    local max_wait=300  # 5 minutes
    local wait_time=0
    
    while [[ $wait_time -lt $max_wait ]]; do
        if docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps | grep -q "unhealthy"; then
            log_info "Services still starting... (${wait_time}s/${max_wait}s)"
            sleep 10
            wait_time=$((wait_time + 10))
        else
            break
        fi
    done
    
    if [[ $wait_time -ge $max_wait ]]; then
        log_error "Services failed to start within ${max_wait} seconds"
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps
        exit 1
    fi
    
    log_success "Data Core services started successfully"
}

# Setup Redis cluster
setup_redis_cluster() {
    log_info "Setting up Redis cluster..."
    
    # Wait for Redis nodes to be ready
    sleep 15
    
    # Check if cluster is already initialized
    if docker exec bev_redis_1 redis-cli -p 7001 -a "${REDIS_PASSWORD}" cluster nodes 2>/dev/null | grep -q "master"; then
        log_info "Redis cluster already initialized"
        return 0
    fi
    
    # Initialize cluster
    log_info "Initializing Redis cluster..."
    docker exec bev_redis_1 redis-cli --cluster create \
        172.30.0.4:7001 172.30.0.5:7002 172.30.0.6:7003 \
        --cluster-replicas 0 --cluster-yes -a "${REDIS_PASSWORD}" || {
        log_warning "Redis cluster initialization failed, but continuing..."
    }
    
    log_success "Redis cluster setup completed"
}

# Validate deployment
validate_deployment() {
    log_info "Validating Data Core deployment..."
    
    local services=(
        "bev_postgres:PostgreSQL"
        "bev_neo4j:Neo4j"
        "bev_redis_1:Redis Node 1"
        "bev_redis_2:Redis Node 2"
        "bev_redis_3:Redis Node 3"
        "bev_influxdb:InfluxDB"
    )
    
    local failed_services=()
    
    for service in "${services[@]}"; do
        local container_name="${service%%:*}"
        local service_name="${service##*:}"
        
        if docker ps --filter "name=$container_name" --filter "status=running" | grep -q "$container_name"; then
            log_success "$service_name is running"
        else
            log_error "$service_name is not running"
            failed_services+=("$service_name")
        fi
    done
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log_error "The following services failed to start: ${failed_services[*]}"
        return 1
    fi
    
    # Test database connections
    log_info "Testing database connections..."
    
    # Test PostgreSQL
    if docker exec bev_postgres pg_isready -U "${POSTGRES_USER}" > /dev/null 2>&1; then
        log_success "PostgreSQL connection test passed"
    else
        log_error "PostgreSQL connection test failed"
        return 1
    fi
    
    # Test Redis
    if docker exec bev_redis_1 redis-cli -p 7001 -a "${REDIS_PASSWORD}" ping > /dev/null 2>&1; then
        log_success "Redis connection test passed"
    else
        log_error "Redis connection test failed"
        return 1
    fi
    
    # Test Neo4j (may take longer to start)
    local neo4j_ready=false
    for i in {1..12}; do  # Wait up to 2 minutes
        if curl -s http://localhost:7474/db/data/ > /dev/null 2>&1; then
            neo4j_ready=true
            break
        fi
        sleep 10
    done
    
    if [[ "$neo4j_ready" == "true" ]]; then
        log_success "Neo4j connection test passed"
    else
        log_error "Neo4j connection test failed"
        return 1
    fi
    
    # Test InfluxDB
    if curl -s http://localhost:8086/health > /dev/null 2>&1; then
        log_success "InfluxDB connection test passed"
    else
        log_error "InfluxDB connection test failed"
        return 1
    fi
    
    log_success "All Data Core services are healthy and accessible"
    return 0
}

# Show deployment information
show_deployment_info() {
    log_info "Data Core Node Deployment Information:"
    echo ""
    echo "Node Type: $NODE_TYPE"
    echo "Node Name: $NODE_NAME"
    echo ""
    echo "Service Endpoints:"
    echo "  PostgreSQL: localhost:5432"
    echo "  Neo4j HTTP: localhost:7474"
    echo "  Neo4j Bolt: localhost:7687"
    echo "  Redis Cluster: localhost:7001,7002,7003"
    echo "  InfluxDB: localhost:8086"
    echo ""
    echo "Management Commands:"
    echo "  View logs: docker compose -f $SCRIPT_DIR/docker-compose.yml logs -f"
    echo "  Stop services: docker compose -f $SCRIPT_DIR/docker-compose.yml down"
    echo "  Restart services: docker compose -f $SCRIPT_DIR/docker-compose.yml restart"
    echo ""
}

# Main deployment function
main() {
    log_info "Starting BEV OSINT Data Core node deployment..."
    
    check_prerequisites
    setup_environment
    setup_database_schemas
    start_services
    setup_redis_cluster
    
    if validate_deployment; then
        show_deployment_info
        log_success "Data Core node deployment completed successfully!"
        
        # Register with cluster coordinator if configured
        if [[ -n "${CLUSTER_COORDINATOR_URL:-}" ]]; then
            log_info "Registering with cluster coordinator..."
            curl -X POST "${CLUSTER_COORDINATOR_URL}/register" \
                -H "Content-Type: application/json" \
                -d "{
                    \"node_type\": \"$NODE_TYPE\",
                    \"node_id\": \"$NODE_NAME\",
                    \"endpoints\": {
                        \"postgres\": \"${HOSTNAME}:5432\",
                        \"neo4j_bolt\": \"${HOSTNAME}:7687\",
                        \"neo4j_http\": \"${HOSTNAME}:7474\",
                        \"redis_cluster\": \"${HOSTNAME}:7001,${HOSTNAME}:7002,${HOSTNAME}:7003\",
                        \"influxdb\": \"${HOSTNAME}:8086\"
                    }
                }" || log_warning "Failed to register with cluster coordinator"
        fi
        
        exit 0
    else
        log_error "Data Core node deployment validation failed!"
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" logs --tail=50
        exit 1
    fi
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "validate")
        validate_deployment
        ;;
    "stop")
        log_info "Stopping Data Core services..."
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" down
        log_success "Data Core services stopped"
        ;;
    "restart")
        log_info "Restarting Data Core services..."
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" restart
        log_success "Data Core services restarted"
        ;;
    "logs")
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" logs -f
        ;;
    "status")
        docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps
        ;;
    *)
        echo "Usage: $0 {deploy|validate|stop|restart|logs|status}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the Data Core node (default)"
        echo "  validate - Validate the current deployment"
        echo "  stop     - Stop all Data Core services"
        echo "  restart  - Restart all Data Core services"
        echo "  logs     - Show logs from all services"
        echo "  status   - Show status of all services"
        exit 1
        ;;
esac