#!/bin/bash

# Phase 2 Startup Script for ORACLE1 OCR and Document Processing Pipeline
# Automated deployment and health verification

set -e

echo "🚀 Starting ORACLE1 Phase 2 OCR and Document Processing Pipeline"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_prerequisites() {
    print_status "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi

    print_success "Prerequisites checked"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories..."

    sudo mkdir -p /tmp/ocr-uploads /tmp/ocr-processed
    sudo mkdir -p /tmp/minio-cache1 /tmp/minio-cache2 /tmp/minio-cache3
    sudo chmod 755 /tmp/ocr-uploads /tmp/ocr-processed
    sudo chmod 755 /tmp/minio-cache*

    print_success "Directories created"
}

# Pull latest images
pull_images() {
    print_status "Pulling latest base images..."

    docker pull python:3.12-slim
    docker pull redis:7-alpine
    docker pull rabbitmq:3-management-alpine
    docker pull neo4j:5-community
    docker pull minio/minio:latest
    docker pull nginx:alpine

    print_success "Base images pulled"
}

# Build custom images
build_images() {
    print_status "Building custom images..."

    cd "$(dirname "$0")"

    # Build OCR Service
    print_status "Building OCR Service..."
    docker build -t oracle1/ocr-service:latest ./ocr-service/

    # Build Document Analyzer
    print_status "Building Document Analyzer..."
    docker build -t oracle1/document-analyzer:latest ./document-analyzer/

    # Build Celery Pipeline
    print_status "Building Celery Pipeline..."
    docker build -t oracle1/celery-pipeline:latest ./celery-pipeline/

    print_success "Custom images built"
}

# Start infrastructure services first
start_infrastructure() {
    print_status "Starting infrastructure services..."

    docker-compose -f phase2-docker-compose.yml up -d redis rabbitmq neo4j

    # Wait for services to be healthy
    print_status "Waiting for infrastructure services to be healthy..."

    # Wait for Redis
    until docker exec oracle1_redis redis-cli ping | grep -q PONG; do
        print_status "Waiting for Redis..."
        sleep 5
    done
    print_success "Redis is healthy"

    # Wait for RabbitMQ
    until docker exec oracle1_rabbitmq rabbitmq-diagnostics check_port_connectivity | grep -q "Port connectivity check passed"; do
        print_status "Waiting for RabbitMQ..."
        sleep 5
    done
    print_success "RabbitMQ is healthy"

    # Wait for Neo4j
    until docker exec oracle1_neo4j cypher-shell -u neo4j -p admin123 "RETURN 1" &>/dev/null; do
        print_status "Waiting for Neo4j..."
        sleep 5
    done
    print_success "Neo4j is healthy"
}

# Start MinIO cluster
start_minio() {
    print_status "Starting MinIO cluster..."

    docker-compose -f phase2-docker-compose.yml up -d minio1 minio2 minio3 minio-lb

    # Wait for MinIO nodes
    for i in {1..3}; do
        until curl -f http://localhost:900$i/minio/health/live &>/dev/null; do
            print_status "Waiting for MinIO node $i..."
            sleep 5
        done
        print_success "MinIO node $i is healthy"
    done

    # Wait for load balancer
    until curl -f http://localhost:9000/health &>/dev/null; do
        print_status "Waiting for MinIO load balancer..."
        sleep 5
    done
    print_success "MinIO load balancer is healthy"
}

# Start application services
start_applications() {
    print_status "Starting application services..."

    docker-compose -f phase2-docker-compose.yml up -d ocr-service document-analyzer

    # Wait for services
    until curl -f http://localhost:8080/health &>/dev/null; do
        print_status "Waiting for OCR Service..."
        sleep 5
    done
    print_success "OCR Service is healthy"

    until curl -f http://localhost:8081/health &>/dev/null; do
        print_status "Waiting for Document Analyzer..."
        sleep 5
    done
    print_success "Document Analyzer is healthy"
}

# Start Celery workers
start_workers() {
    print_status "Starting Celery workers..."

    docker-compose -f phase2-docker-compose.yml up -d \
        celery-edge-worker \
        celery-genetic-worker \
        celery-knowledge-worker \
        celery-toolmaster-worker \
        celery-flower

    # Wait for Flower
    until curl -f http://localhost:5555 &>/dev/null; do
        print_status "Waiting for Celery Flower..."
        sleep 5
    done
    print_success "Celery workers and Flower are running"
}

# Initialize MinIO buckets
initialize_minio() {
    print_status "Initializing MinIO buckets..."

    # Wait a bit more for MinIO to be fully ready
    sleep 10

    # Use MinIO client to create buckets
    docker run --rm --network docker_oracle1-net \
        minio/mc:latest sh -c "
        mc alias set oracle1 http://minio-lb admin admin123456;
        mc mb oracle1/oracle1-documents --ignore-existing;
        mc mb oracle1/oracle1-models --ignore-existing;
        mc mb oracle1/oracle1-cache --ignore-existing;
        mc mb oracle1/oracle1-backups --ignore-existing;
        mc mb oracle1/oracle1-logs --ignore-existing;
        mc policy set public oracle1/oracle1-documents;
        mc policy set private oracle1/oracle1-models;
        mc policy set private oracle1/oracle1-backups;
        echo 'MinIO buckets initialized successfully';
    "

    print_success "MinIO buckets initialized"
}

# Health check all services
health_check() {
    print_status "Performing comprehensive health check..."

    # Service endpoints to check
    declare -A services=(
        ["Redis"]="redis://localhost:6379"
        ["RabbitMQ Management"]="http://localhost:15672"
        ["Neo4j Browser"]="http://localhost:7474"
        ["OCR Service"]="http://localhost:8080/health"
        ["Document Analyzer"]="http://localhost:8081/health"
        ["MinIO API"]="http://localhost:9000/minio/health/live"
        ["MinIO Console"]="http://localhost:9011"
        ["Celery Flower"]="http://localhost:5555"
    )

    echo ""
    echo "🏥 Health Check Results:"
    echo "========================"

    for service in "${!services[@]}"; do
        url="${services[$service]}"
        if [[ $url == redis://* ]]; then
            if docker exec oracle1_redis redis-cli ping | grep -q PONG; then
                print_success "$service: ✅ Healthy"
            else
                print_error "$service: ❌ Unhealthy"
            fi
        else
            if curl -f "$url" &>/dev/null; then
                print_success "$service: ✅ Healthy"
            else
                print_error "$service: ❌ Unhealthy"
            fi
        fi
    done
}

# Display service information
show_services() {
    echo ""
    echo "🔗 Service Access Information:"
    echo "=============================="
    echo ""
    echo "📄 OCR Service:"
    echo "   API: http://localhost:8080"
    echo "   Health: http://localhost:8080/health"
    echo "   Docs: http://localhost:8080/docs"
    echo ""
    echo "🧠 Document Analyzer:"
    echo "   API: http://localhost:8081"
    echo "   Health: http://localhost:8081/health"
    echo "   Docs: http://localhost:8081/docs"
    echo ""
    echo "🗃️ Storage (MinIO):"
    echo "   API: http://localhost:9000"
    echo "   Console: http://localhost:9011"
    echo "   Credentials: admin / admin123456"
    echo ""
    echo "📊 Monitoring:"
    echo "   Celery Flower: http://localhost:5555"
    echo "   RabbitMQ: http://localhost:15672 (admin/admin123)"
    echo "   Neo4j: http://localhost:7474 (neo4j/admin123)"
    echo ""
    echo "🔧 Management:"
    echo "   Redis: localhost:6379"
    echo "   RabbitMQ AMQP: localhost:5672"
    echo "   Neo4j Bolt: localhost:7687"
    echo ""
}

# Display test commands
show_test_commands() {
    echo "🧪 Test Commands:"
    echo "=================="
    echo ""
    echo "# Test OCR Service"
    echo "curl -X POST \"http://localhost:8080/ocr/process\" \\"
    echo "  -F \"file=@test-document.pdf\" \\"
    echo "  -F \"language=auto\""
    echo ""
    echo "# Test Document Analyzer"
    echo "curl -X POST \"http://localhost:8081/analyze\" \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"document_id\": \"test-001\", \"text\": \"Sample text\", \"analysis_types\": [\"entities\"]}'"
    echo ""
    echo "# Test MinIO"
    echo "docker run --rm -v \$(pwd):/data --network docker_oracle1-net \\"
    echo "  minio/mc:latest sh -c \\"
    echo "  \"mc alias set oracle1 http://minio-lb admin admin123456 && \\"
    echo "   echo 'Hello ORACLE1' > /data/test.txt && \\"
    echo "   mc cp /data/test.txt oracle1/oracle1-documents/\""
    echo ""
}

# Main execution
main() {
    check_prerequisites
    create_directories
    pull_images
    build_images

    print_status "Starting services in order..."
    start_infrastructure
    start_minio
    initialize_minio
    start_applications
    start_workers

    health_check
    show_services
    show_test_commands

    echo ""
    print_success "🎉 ORACLE1 Phase 2 Pipeline is now running!"
    print_status "Monitor logs with: docker-compose -f phase2-docker-compose.yml logs -f"
    print_status "Stop services with: docker-compose -f phase2-docker-compose.yml down"
    echo ""
}

# Handle script interruption
cleanup() {
    print_warning "Interrupted! Cleaning up..."
    docker-compose -f phase2-docker-compose.yml down
    exit 1
}

trap cleanup INT

# Run main function
main "$@"