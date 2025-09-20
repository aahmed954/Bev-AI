#!/bin/bash
# BEV Platform Compatibility Fix Script
# Addresses ARM64, GPU, and platform specification issues
# Author: BEV OSINT Team

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${PROJECT_ROOT}/backups/compatibility_$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Create backup
create_backup() {
    log "Creating backup of original files..."
    mkdir -p "$BACKUP_DIR"

    # Backup Docker Compose files
    cp "$PROJECT_ROOT/docker-compose-thanos-unified.yml" "$BACKUP_DIR/"
    cp "$PROJECT_ROOT/docker-compose-oracle1-unified.yml" "$BACKUP_DIR/"

    # Backup requirements files that need modification
    find "$PROJECT_ROOT" -name "requirements*.txt" -exec cp --parents {} "$BACKUP_DIR" \;

    success "Backup created at: $BACKUP_DIR"
}

# Fix 1: Add ARM64 platform specifications to Oracle1 compose file
fix_arm64_platforms() {
    log "Adding ARM64 platform specifications to Oracle1 compose file..."

    local oracle_file="$PROJECT_ROOT/docker-compose-oracle1-unified.yml"
    local temp_file=$(mktemp)

    # Add platform specifications for all services in Oracle1
    sed -E '
        /^  [a-z-]+:$/,/^  [a-z-]+:$|^$/ {
            /^    image:/ a\
    platform: linux/arm64
        }
    ' "$oracle_file" > "$temp_file"

    mv "$temp_file" "$oracle_file"
    success "ARM64 platform specifications added to Oracle1"
}

# Fix 2: Add GPU access and runtime configurations to Thanos services
fix_gpu_configurations() {
    log "Fixing GPU configurations in Thanos compose file..."

    local thanos_file="$PROJECT_ROOT/docker-compose-thanos-unified.yml"
    local temp_file=$(mktemp)

    # Replace ENABLE_GPU: false with ENABLE_GPU: true for document analyzers
    sed 's/ENABLE_GPU: false/ENABLE_GPU: true/g' "$thanos_file" > "$temp_file"

    # Add runtime: nvidia and device requests for GPU services
    awk '
    /^  doc-analyzer-[1-3]:$/ { gpu_service = 1 }
    /^  [a-z-]+:$/ && !/^  doc-analyzer-[1-3]:$/ { gpu_service = 0 }

    # Add runtime and device configs after container_name for GPU services
    gpu_service && /^    container_name:/ {
        print $0
        print "    runtime: nvidia"
        print "    deploy:"
        print "      resources:"
        print "        reservations:"
        print "          devices:"
        print "            - driver: nvidia"
        print "              count: 1"
        print "              capabilities: [gpu]"
        next
    }

    # Skip existing deploy sections for GPU services to avoid duplication
    gpu_service && /^    deploy:$/,/^    [a-z_]+:$|^  [a-z-]+:$/ {
        if (/^  [a-z-]+:$/) {
            gpu_service = 0
            print $0
        }
        next
    }

    { print $0 }
    ' "$temp_file" > "$thanos_file"

    success "GPU configurations updated in Thanos"
}

# Fix 3: Update PyTorch installations for CUDA compatibility
fix_pytorch_cuda() {
    log "Updating PyTorch installations for CUDA 13.0 compatibility..."

    # Update main requirements.txt
    sed -i 's/torch==2\.1\.0/torch==2.1.0+cu121/' "$PROJECT_ROOT/requirements.txt"

    # Update document analyzer requirements
    if [ -f "$PROJECT_ROOT/docker/document-analyzer/requirements.txt" ]; then
        sed -i 's/torch==2\.1\.1/torch==2.1.1+cu121/' "$PROJECT_ROOT/docker/document-analyzer/requirements.txt"
        echo "torchvision==0.16.1+cu121" >> "$PROJECT_ROOT/docker/document-analyzer/requirements.txt"
        echo "torchaudio==2.1.1+cu121" >> "$PROJECT_ROOT/docker/document-analyzer/requirements.txt"
    fi

    # Update celery pipeline requirements
    if [ -f "$PROJECT_ROOT/docker/celery-pipeline/requirements.txt" ]; then
        sed -i 's/torch==2\.1\.1/torch==2.1.1+cu121/' "$PROJECT_ROOT/docker/celery-pipeline/requirements.txt"
    fi

    # Update autonomous requirements
    if [ -f "$PROJECT_ROOT/src/autonomous/requirements.txt" ]; then
        sed -i 's/torch>=1\.10\.0/torch==2.1.0+cu121/' "$PROJECT_ROOT/src/autonomous/requirements.txt"
        sed -i 's/torch-geometric>=2\.0\.0/torch-geometric==2.4.0/' "$PROJECT_ROOT/src/autonomous/requirements.txt"
    fi

    success "PyTorch CUDA compatibility updated"
}

# Fix 4: Add missing build contexts
fix_missing_build_contexts() {
    log "Creating missing build contexts..."

    # Create missing thanos directories structure
    mkdir -p "$PROJECT_ROOT/thanos/phase2/"{ocr,analyzer}
    mkdir -p "$PROJECT_ROOT/thanos/phase3/"{swarm,coordinator,memory,optimizer,tools}
    mkdir -p "$PROJECT_ROOT/thanos/phase4/"{guardian,ids,traffic,anomaly}
    mkdir -p "$PROJECT_ROOT/thanos/phase5/"{controller,live2d/backend,live2d/frontend}

    # Create minimal Dockerfiles for missing contexts
    create_minimal_dockerfile() {
        local dir="$1"
        local base_image="$2"
        local description="$3"

        mkdir -p "$dir"
        cat > "$dir/Dockerfile" << EOF
# $description
FROM $base_image

LABEL maintainer="BEV OSINT Team"
LABEL description="$description"

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install dependencies if requirements exist
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "main.py"]
EOF
    }

    # OCR Service
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase2/ocr" "python:3.11-slim" "OCR Processing Service"

    # Document Analyzer
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase2/analyzer" "python:3.11-slim" "Document Analysis Service"

    # Swarm Intelligence
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase3/swarm" "python:3.11-slim" "Swarm Intelligence Service"

    # Research Coordinator
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase3/coordinator" "python:3.11-slim" "Research Coordination Service"

    # Memory Manager
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase3/memory" "python:3.11-slim" "Memory Management Service"

    # Code Optimizer
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase3/optimizer" "python:3.11-slim" "Code Optimization Service"

    # Tool Coordinator
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase3/tools" "python:3.11-slim" "Tool Coordination Service"

    # Guardian Services
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase4/guardian" "python:3.11-slim" "Security Guardian Service"

    # IDS
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase4/ids" "python:3.11-slim" "Intrusion Detection Service"

    # Traffic Analyzer
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase4/traffic" "python:3.11-slim" "Traffic Analysis Service"

    # Anomaly Detector
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase4/anomaly" "python:3.11-slim" "Anomaly Detection Service"

    # Autonomous Controllers
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase5/controller" "python:3.11-slim" "Autonomous Controller Service"

    # Live2D Backend
    create_minimal_dockerfile "$PROJECT_ROOT/thanos/phase5/live2d/backend" "python:3.11-slim" "Live2D Avatar Backend"

    # Live2D Frontend
    cat > "$PROJECT_ROOT/thanos/phase5/live2d/frontend/Dockerfile" << 'EOF'
# Live2D Avatar Frontend
FROM node:18-alpine

LABEL maintainer="BEV OSINT Team"
LABEL description="Live2D Avatar Frontend Service"

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Expose port
EXPOSE 3000

# Start application
CMD ["npm", "start"]
EOF

    success "Missing build contexts created"
}

# Fix 5: Add platform specifications to Thanos services that need ARM fallback
fix_thanos_platform_specs() {
    log "Adding platform specifications to Thanos services..."

    local thanos_file="$PROJECT_ROOT/docker-compose-thanos-unified.yml"
    local temp_file=$(mktemp)

    # Add platform: linux/amd64 for x86-only images
    awk '
    /^    image: (neo4j:|confluentinc\/|docker\.elastic\.co\/|dperson\/|intelowlproject\/|prom\/|grafana\/|apache\/|vault:|torproject\/)/ {
        print $0
        print "    platform: linux/amd64"
        next
    }

    /^    image: (pgvector\/|redis:|rabbitmq:|influxdb:|nginx:)/ {
        print $0
        print "    platform: linux/amd64,linux/arm64"
        next
    }

    { print $0 }
    ' "$thanos_file" > "$temp_file"

    mv "$temp_file" "$thanos_file"
    success "Platform specifications added to Thanos services"
}

# Fix 6: Update .env file with CUDA configuration
fix_env_cuda_config() {
    log "Adding CUDA configuration to .env file..."

    if ! grep -q "CUDA_VERSION" "$PROJECT_ROOT/.env"; then
        cat >> "$PROJECT_ROOT/.env" << 'EOF'

# CUDA Configuration
CUDA_VERSION=13.0
CUDA_HOME=/usr/local/cuda
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# GPU Memory Configuration
GPU_MEMORY_FRACTION=0.8
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# PyTorch Configuration
TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6"
EOF
    fi

    success "CUDA configuration added to .env"
}

# Fix 7: Create Oracle1-specific Dockerfiles for ARM compatibility
create_oracle_arm_dockerfiles() {
    log "Creating ARM-optimized Dockerfiles for Oracle1..."

    mkdir -p "$PROJECT_ROOT/docker/oracle/arm"

    # ARM-optimized Python base
    cat > "$PROJECT_ROOT/docker/oracle/arm/Dockerfile.base" << 'EOF'
# ARM64-optimized Python base image
FROM python:3.11-slim-bookworm

# Set platform-specific labels
LABEL platform="linux/arm64"
LABEL maintainer="BEV OSINT Team"

# ARM64 optimizations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install ARM64-optimized packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Default health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]
EOF

    success "ARM-optimized Dockerfiles created"
}

# Fix 8: Create validation script
create_validation_script() {
    log "Creating platform validation script..."

    cat > "$PROJECT_ROOT/scripts/validate_platform_compatibility.sh" << 'EOF'
#!/bin/bash
# Platform Compatibility Validation Script

set -euo pipefail

validate_docker_platforms() {
    echo "Validating Docker platform support..."

    # Check if Docker supports multi-platform builds
    if ! docker buildx version >/dev/null 2>&1; then
        echo "ERROR: Docker buildx not available. Multi-platform builds not supported."
        return 1
    fi

    # Check available platforms
    docker buildx ls

    echo "✓ Docker multi-platform support validated"
}

validate_gpu_access() {
    echo "Validating GPU access..."

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
        echo "✓ NVIDIA GPU detected"
    else
        echo "⚠ No NVIDIA GPU detected (running on CPU-only node)"
    fi

    # Check Docker GPU runtime
    if docker info 2>/dev/null | grep -q "nvidia"; then
        echo "✓ Docker NVIDIA runtime available"
    else
        echo "⚠ Docker NVIDIA runtime not configured"
    fi
}

validate_cuda_version() {
    echo "Validating CUDA version..."

    if command -v nvcc >/dev/null 2>&1; then
        nvcc --version
        echo "✓ CUDA toolkit detected"
    else
        echo "⚠ CUDA toolkit not found"
    fi
}

validate_arm_compatibility() {
    echo "Validating ARM64 compatibility..."

    arch=$(uname -m)
    case $arch in
        x86_64|amd64)
            echo "ℹ Running on x86_64 architecture"
            ;;
        aarch64|arm64)
            echo "✓ Running on ARM64 architecture"
            ;;
        *)
            echo "⚠ Unknown architecture: $arch"
            ;;
    esac
}

main() {
    echo "=== BEV Platform Compatibility Validation ==="
    echo

    validate_docker_platforms
    echo
    validate_gpu_access
    echo
    validate_cuda_version
    echo
    validate_arm_compatibility

    echo
    echo "=== Validation Complete ==="
}

main "$@"
EOF

    chmod +x "$PROJECT_ROOT/scripts/validate_platform_compatibility.sh"
    success "Platform validation script created"
}

# Main execution
main() {
    echo "======================================================================"
    echo "BEV Platform Compatibility Fix Script"
    echo "======================================================================"
    echo

    log "Starting compatibility fixes..."

    # Create backup
    create_backup

    # Apply fixes
    fix_arm64_platforms
    fix_gpu_configurations
    fix_pytorch_cuda
    fix_missing_build_contexts
    fix_thanos_platform_specs
    fix_env_cuda_config
    create_oracle_arm_dockerfiles
    create_validation_script

    echo
    success "All compatibility fixes applied successfully!"
    echo
    echo "Next steps:"
    echo "1. Review the changes in your Docker Compose files"
    echo "2. Run: ./scripts/validate_platform_compatibility.sh"
    echo "3. Test deployment on both ARM64 and x86_64 platforms"
    echo "4. Verify GPU access on Thanos node"
    echo
    warn "Backup of original files saved to: $BACKUP_DIR"
}

# Execute main function
main "$@"