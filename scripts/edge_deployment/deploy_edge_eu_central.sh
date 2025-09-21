#!/bin/bash

# Edge Computing Deployment Script - EU Central Region
# BEV OSINT Framework Edge Computing Network

set -euo pipefail

# Configuration
REGION="eu-central"
NODE_ID="edge-eu-central-001"
IP_ADDRESS="172.30.0.49"
SERVICE_PORT="8000"
MODEL_PORT="8001"
ADMIN_PORT="8002"
METRICS_PORT="9090"

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EDGE_SRC_DIR="$PROJECT_ROOT/src/edge"
DEPLOY_DIR="/opt/bev_edge/$REGION"
LOG_DIR="/var/log/bev_edge/$REGION"
MODEL_DIR="/opt/models/$REGION"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi
}

# Check system requirements
check_requirements() {
    log "Checking system requirements for $REGION deployment..."

    # Check available memory (require at least 12GB for EU region)
    local mem_gb
    mem_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $mem_gb -lt 12 ]]; then
        error "Insufficient memory: ${mem_gb}GB available, 12GB required"
    fi
    info "Memory check passed: ${mem_gb}GB available"

    # Check available disk space (require at least 80GB)
    local disk_gb
    disk_gb=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [[ $disk_gb -lt 80 ]]; then
        error "Insufficient disk space: ${disk_gb}GB available, 80GB required"
    fi
    info "Disk space check passed: ${disk_gb}GB available"

    # Check for Python 3.8+
    if ! command -v python3 &> /dev/null; then
        error "Python3 not found"
    fi

    local python_version
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        error "Python 3.8+ required, found $python_version"
    fi
    info "Python version check passed: $python_version"

    # Check for Docker
    if ! command -v docker &> /dev/null; then
        warn "Docker not found, will install"
    fi

    # Check for NVIDIA drivers (optional)
    if command -v nvidia-smi &> /dev/null; then
        info "NVIDIA drivers detected"
        nvidia-smi
    else
        warn "NVIDIA drivers not found, GPU acceleration will be disabled"
    fi
}

# Install system dependencies (EU specific packages)
install_dependencies() {
    log "Installing system dependencies for EU region..."

    # Update package list
    apt-get update

    # Install required packages
    apt-get install -y \
        curl \
        wget \
        git \
        python3-pip \
        python3-venv \
        build-essential \
        nginx \
        redis-server \
        postgresql-client \
        htop \
        iotop \
        net-tools \
        jq \
        bc \
        unzip \
        locales \
        tzdata

    # Set timezone for EU Central
    timedatectl set-timezone Europe/Berlin

    # Generate locales
    locale-gen en_US.UTF-8 de_DE.UTF-8

    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        log "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        usermod -aG docker $SUDO_USER || true
        systemctl enable docker
        systemctl start docker
    fi

    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log "Installing Docker Compose..."
        curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    fi

    # Install Node Exporter for monitoring
    if ! command -v node_exporter &> /dev/null; then
        log "Installing Node Exporter..."
        wget https://github.com/prometheus/node_exporter/releases/latest/download/node_exporter-1.6.1.linux-amd64.tar.gz
        tar xvfz node_exporter-*.tar.gz
        cp node_exporter-*/node_exporter /usr/local/bin/
        rm -rf node_exporter-*

        # Create systemd service
        cat > /etc/systemd/system/node_exporter.service << EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
Type=simple
ExecStart=/usr/local/bin/node_exporter --web.listen-address=:9100
Restart=always

[Install]
WantedBy=multi-user.target
EOF

        systemctl daemon-reload
        systemctl enable node_exporter
        systemctl start node_exporter
    fi
}

# Setup directories and permissions
setup_directories() {
    log "Setting up directories for $REGION..."

    # Create directories
    mkdir -p "$DEPLOY_DIR"
    mkdir -p "$LOG_DIR"
    mkdir -p "$MODEL_DIR"
    mkdir -p "/etc/bev_edge/$REGION"

    # Set permissions
    chown -R $SUDO_USER:$SUDO_USER "$DEPLOY_DIR"
    chown -R $SUDO_USER:$SUDO_USER "$LOG_DIR"
    chown -R $SUDO_USER:$SUDO_USER "$MODEL_DIR"

    # Create logs directory structure
    mkdir -p "$LOG_DIR/edge_network"
    mkdir -p "$LOG_DIR/node_manager"
    mkdir -p "$LOG_DIR/model_sync"
    mkdir -p "$LOG_DIR/geo_router"
}

# Setup Python environment
setup_python_environment() {
    log "Setting up Python environment for $REGION..."

    # Create virtual environment
    python3 -m venv "$DEPLOY_DIR/venv"
    source "$DEPLOY_DIR/venv/bin/activate"

    # Upgrade pip
    pip install --upgrade pip

    # Install Python dependencies
    cat > "$DEPLOY_DIR/requirements.txt" << EOF
# Core dependencies
aiohttp==3.8.5
asyncpg==0.28.0
asyncio==3.4.3
aiofiles==23.2.1

# ML and AI (lighter models for EU region)
torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.3
huggingface-hub>=0.15.1
accelerate>=0.20.3

# Monitoring and metrics
prometheus-client==0.17.1
psutil==5.9.5
GPUtil==1.4.0
nvidia-ml-py3==7.352.0

# Geographic and networking
geoip2==4.7.0
geopy==2.3.0
redis==4.6.0

# Utilities
python-dotenv==1.0.0
pydantic==2.0.3
click==8.1.6
uvloop==0.17.0
boto3==1.28.25
requests==2.31.0
EOF

    pip install -r "$DEPLOY_DIR/requirements.txt"

    # Install PyTorch with CUDA support if available
    if command -v nvidia-smi &> /dev/null; then
        log "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi

    deactivate
}

# Copy edge computing source code
deploy_source_code() {
    log "Deploying edge computing source code for $REGION..."

    # Copy source files
    cp -r "$EDGE_SRC_DIR"/* "$DEPLOY_DIR/"

    # Create configuration file with EU-specific settings
    cat > "$DEPLOY_DIR/config.json" << EOF
{
    "node_id": "$NODE_ID",
    "region": "$REGION",
    "ip_address": "$IP_ADDRESS",
    "timezone": "Europe/Berlin",
    "locale": "de_DE.UTF-8",
    "ports": {
        "service": $SERVICE_PORT,
        "model": $MODEL_PORT,
        "admin": $ADMIN_PORT,
        "metrics": $METRICS_PORT
    },
    "directories": {
        "models": "$MODEL_DIR",
        "logs": "$LOG_DIR",
        "temp": "/tmp/bev_edge_$REGION"
    },
    "database": {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "postgres",
        "database": "bev_osint"
    },
    "redis": {
        "host": "localhost",
        "port": 6379
    },
    "limits": {
        "max_concurrent_requests": 80,
        "memory_limit_gb": 10,
        "gpu_memory_limit_gb": 6,
        "model_cache_size": 2
    },
    "monitoring": {
        "health_check_interval": 30,
        "model_sync_interval": 300,
        "metrics_collection_interval": 60
    },
    "compliance": {
        "gdpr_enabled": true,
        "data_retention_days": 30,
        "logging_level": "INFO"
    }
}
EOF

    # Set permissions
    chown -R $SUDO_USER:$SUDO_USER "$DEPLOY_DIR"
    chmod +x "$DEPLOY_DIR"/*.py
}

# Setup systemd services
setup_systemd_services() {
    log "Setting up systemd services for $REGION..."

    # Edge Node Manager service
    cat > "/etc/systemd/system/bev-edge-node-$REGION.service" << EOF
[Unit]
Description=BEV Edge Node Manager - $REGION
After=network.target postgresql.service redis.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=$SUDO_USER
Group=$SUDO_USER
WorkingDirectory=$DEPLOY_DIR
Environment=PATH=$DEPLOY_DIR/venv/bin
Environment=TZ=Europe/Berlin
Environment=LANG=en_US.UTF-8
ExecStart=$DEPLOY_DIR/venv/bin/python edge_node_manager.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bev-edge-node-$REGION

# Resource limits (conservative for EU)
LimitNOFILE=32768
LimitNPROC=16384

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DEPLOY_DIR $LOG_DIR $MODEL_DIR /tmp

[Install]
WantedBy=multi-user.target
EOF

    # Edge Management Service
    cat > "/etc/systemd/system/bev-edge-management-$REGION.service" << EOF
[Unit]
Description=BEV Edge Management Service - $REGION
After=network.target postgresql.service redis.service bev-edge-node-$REGION.service
Requires=postgresql.service redis.service

[Service]
Type=simple
User=$SUDO_USER
Group=$SUDO_USER
WorkingDirectory=$DEPLOY_DIR
Environment=PATH=$DEPLOY_DIR/venv/bin
Environment=TZ=Europe/Berlin
Environment=LANG=en_US.UTF-8
ExecStart=$DEPLOY_DIR/venv/bin/python edge_management_service.py
ExecReload=/bin/kill -HUP \$MAINPID
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bev-edge-management-$REGION

# Resource limits
LimitNOFILE=32768
LimitNPROC=16384

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DEPLOY_DIR $LOG_DIR $MODEL_DIR /tmp

[Install]
WantedBy=multi-user.target
EOF

    # Model Synchronizer service
    cat > "/etc/systemd/system/bev-model-sync-$REGION.service" << EOF
[Unit]
Description=BEV Model Synchronizer - $REGION
After=network.target postgresql.service
Requires=postgresql.service

[Service]
Type=simple
User=$SUDO_USER
Group=$SUDO_USER
WorkingDirectory=$DEPLOY_DIR
Environment=PATH=$DEPLOY_DIR/venv/bin
Environment=TZ=Europe/Berlin
ExecStart=$DEPLOY_DIR/venv/bin/python model_synchronizer.py
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal
SyslogIdentifier=bev-model-sync-$REGION

# Resource limits
LimitNOFILE=32768

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$DEPLOY_DIR $LOG_DIR $MODEL_DIR /tmp

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd
    systemctl daemon-reload
}

# Setup nginx reverse proxy
setup_nginx() {
    log "Setting up nginx reverse proxy for $REGION..."

    cat > "/etc/nginx/sites-available/bev-edge-$REGION" << EOF
upstream bev_edge_$REGION {
    server $IP_ADDRESS:$SERVICE_PORT;
    keepalive 16;
}

upstream bev_edge_admin_$REGION {
    server $IP_ADDRESS:$ADMIN_PORT;
    keepalive 8;
}

server {
    listen 80;
    server_name edge-$REGION.bev.local;

    # Security headers (GDPR compliant)
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Rate limiting (conservative for EU)
    limit_req_zone \$binary_remote_addr zone=edge_api_eu:10m rate=60r/m;

    # API endpoints
    location /api/ {
        limit_req zone=edge_api_eu burst=10 nodelay;

        proxy_pass http://bev_edge_$REGION;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Health check
    location /health {
        proxy_pass http://bev_edge_$REGION;
        access_log off;
    }

    # Admin interface (restricted)
    location /admin/ {
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;

        proxy_pass http://bev_edge_admin_$REGION;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
    }

    # Metrics (Prometheus) - restricted
    location /metrics {
        allow 127.0.0.1;
        allow 10.0.0.0/8;
        allow 172.16.0.0/12;
        allow 192.168.0.0/16;
        deny all;

        proxy_pass http://$IP_ADDRESS:$METRICS_PORT;
        access_log off;
    }
}
EOF

    # Enable site
    ln -sf "/etc/nginx/sites-available/bev-edge-$REGION" "/etc/nginx/sites-enabled/"

    # Test nginx configuration
    nginx -t || error "Nginx configuration test failed"

    # Reload nginx
    systemctl reload nginx
}

# Setup monitoring and logging
setup_monitoring() {
    log "Setting up monitoring and logging for $REGION..."

    # Create log rotation configuration (GDPR compliant)
    cat > "/etc/logrotate.d/bev-edge-$REGION" << EOF
$LOG_DIR/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $SUDO_USER $SUDO_USER
    postrotate
        systemctl reload bev-edge-node-$REGION bev-edge-management-$REGION bev-model-sync-$REGION || true
    endscript
}
EOF

    # Create monitoring script
    cat > "$DEPLOY_DIR/monitor.sh" << EOF
#!/bin/bash

# Edge Node Monitoring Script - EU Central

REGION="$REGION"
LOG_FILE="$LOG_DIR/monitor.log"

log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] \$1" >> "\$LOG_FILE"
}

# Check service health
check_services() {
    for service in "bev-edge-node-\$REGION" "bev-edge-management-\$REGION" "bev-model-sync-\$REGION"; do
        if ! systemctl is-active --quiet "\$service"; then
            log_message "WARNING: Service \$service is not running"
            systemctl start "\$service"
        fi
    done
}

# Check resource usage
check_resources() {
    # Memory usage (lower thresholds for EU)
    local mem_usage
    mem_usage=\$(free | awk '/^Mem:/ {printf "%.1f", \$3/\$2 * 100.0}')
    if (( \$(echo "\$mem_usage > 85" | bc -l) )); then
        log_message "WARNING: High memory usage: \${mem_usage}%"
    fi

    # Disk usage
    local disk_usage
    disk_usage=\$(df / | awk 'NR==2 {print \$5}' | sed 's/%//')
    if [[ \$disk_usage -gt 80 ]]; then
        log_message "WARNING: High disk usage: \${disk_usage}%"
    fi

    # GPU usage (if available)
    if command -v nvidia-smi &> /dev/null; then
        local gpu_usage
        gpu_usage=\$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        if [[ \$gpu_usage -gt 90 ]]; then
            log_message "WARNING: High GPU usage: \${gpu_usage}%"
        fi
    fi
}

# Check network connectivity (including US regions)
check_network() {
    # Check connectivity to other regions
    local regions=("172.30.0.47" "172.30.0.48" "172.30.0.50")

    for ip in "\${regions[@]}"; do
        if ! ping -c 1 -W 5 "\$ip" &> /dev/null; then
            log_message "WARNING: Cannot reach edge node at \$ip"
        fi
    done

    # Check database connectivity
    if ! pg_isready -h localhost -p 5432 &> /dev/null; then
        log_message "WARNING: PostgreSQL is not responding"
    fi

    # Check Redis connectivity
    if ! redis-cli ping &> /dev/null; then
        log_message "WARNING: Redis is not responding"
    fi
}

# Main monitoring loop
main() {
    log_message "Starting monitoring check"
    check_services
    check_resources
    check_network
    log_message "Monitoring check completed"
}

main
EOF

    chmod +x "$DEPLOY_DIR/monitor.sh"

    # Setup cron job for monitoring
    echo "*/5 * * * * $DEPLOY_DIR/monitor.sh" | crontab -u $SUDO_USER -
}

# Download and setup models (EU optimized)
setup_models() {
    log "Setting up default models for $REGION..."

    # Create model download script with EU-specific models
    cat > "$DEPLOY_DIR/download_models.py" << 'EOF'
#!/usr/bin/env python3

import asyncio
import logging
from model_synchronizer import ModelSynchronizer, ModelVersion, SyncPriority
from datetime import datetime

async def download_default_models():
    """Download default models for EU Central edge node"""
    synchronizer = ModelSynchronizer()

    try:
        await synchronizer.initialize()

        # Default models for EU Central region (smaller models for GDPR compliance)
        models = [
            {
                "name": "microsoft/Phi-3-mini-4k-instruct",
                "version": "latest",
                "priority": SyncPriority.HIGH
            }
        ]

        for model_config in models:
            model_version = ModelVersion(
                model_name=model_config["name"],
                version=model_config["version"],
                model_path=model_config["name"],
                model_size_bytes=1500000000,  # Smaller estimated size
                checksum="",
                release_date=datetime.utcnow(),
                compatibility_version="1.0",
                deployment_regions=["eu-central"],
                priority=model_config["priority"],
                metadata={"auto_download": True, "gdpr_compliant": True}
            )

            await synchronizer.add_model_version(model_version)

            task_id = await synchronizer.sync_model_to_regions(
                model_config["name"],
                model_config["version"],
                ["eu-central"],
                model_config["priority"]
            )

            print(f"Started download for {model_config['name']}: {task_id}")

    finally:
        await synchronizer.cleanup()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(download_default_models())
EOF

    chmod +x "$DEPLOY_DIR/download_models.py"
}

# Start services
start_services() {
    log "Starting edge computing services for $REGION..."

    # Enable and start services
    systemctl enable "bev-edge-node-$REGION"
    systemctl enable "bev-edge-management-$REGION"
    systemctl enable "bev-model-sync-$REGION"

    # Start in order
    systemctl start "bev-edge-node-$REGION"
    sleep 15  # Longer wait for EU region
    systemctl start "bev-edge-management-$REGION"
    sleep 10
    systemctl start "bev-model-sync-$REGION"

    # Check service status
    for service in "bev-edge-node-$REGION" "bev-edge-management-$REGION" "bev-model-sync-$REGION"; do
        if systemctl is-active --quiet "$service"; then
            info "✓ $service is running"
        else
            error "✗ $service failed to start"
        fi
    done
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment for $REGION..."

    # Check service endpoints
    local endpoints=(
        "http://$IP_ADDRESS:$SERVICE_PORT/health"
        "http://$IP_ADDRESS:$ADMIN_PORT/admin/dashboard"
        "http://$IP_ADDRESS:$METRICS_PORT/metrics"
    )

    for endpoint in "${endpoints[@]}"; do
        if curl -s -f "$endpoint" > /dev/null; then
            info "✓ $endpoint is responding"
        else
            warn "✗ $endpoint is not responding"
        fi
    done

    # Test basic functionality
    info "Testing basic edge computing functionality..."

    local test_payload='{"request_type": "inference", "payload": {"prompt": "Hello world"}, "priority": 1, "max_latency_ms": 5000}'

    if curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        "http://$IP_ADDRESS:$SERVICE_PORT/api/v1/process" > /dev/null; then
        info "✓ Basic inference test passed"
    else
        warn "✗ Basic inference test failed"
    fi
}

# Print deployment summary
print_summary() {
    log "Deployment Summary for $REGION"
    echo "=================================="
    echo "Region: $REGION (Europe/Berlin)"
    echo "Node ID: $NODE_ID"
    echo "IP Address: $IP_ADDRESS"
    echo "Service Port: $SERVICE_PORT"
    echo "Admin Port: $ADMIN_PORT"
    echo "Metrics Port: $METRICS_PORT"
    echo "GDPR Compliance: Enabled"
    echo ""
    echo "Directories:"
    echo "  Deploy: $DEPLOY_DIR"
    echo "  Logs: $LOG_DIR"
    echo "  Models: $MODEL_DIR"
    echo ""
    echo "Services:"
    echo "  Edge Node: bev-edge-node-$REGION"
    echo "  Management: bev-edge-management-$REGION"
    echo "  Model Sync: bev-model-sync-$REGION"
    echo ""
    echo "Endpoints:"
    echo "  API: http://$IP_ADDRESS:$SERVICE_PORT/api/v1/"
    echo "  Admin: http://$IP_ADDRESS:$ADMIN_PORT/admin/"
    echo "  Metrics: http://$IP_ADDRESS:$METRICS_PORT/metrics"
    echo "  Health: http://$IP_ADDRESS:$SERVICE_PORT/health"
    echo ""
    echo "Next steps:"
    echo "1. Monitor services: systemctl status bev-edge-*-$REGION"
    echo "2. View logs: journalctl -u bev-edge-node-$REGION -f"
    echo "3. Download models: $DEPLOY_DIR/download_models.py"
    echo "4. Test deployment: curl http://$IP_ADDRESS:$SERVICE_PORT/health"
}

# Main deployment function
main() {
    log "Starting edge computing deployment for $REGION"

    check_root
    check_requirements
    install_dependencies
    setup_directories
    setup_python_environment
    deploy_source_code
    setup_systemd_services
    setup_nginx
    setup_monitoring
    setup_models
    start_services
    sleep 45  # Allow more time for EU region startup
    verify_deployment
    print_summary

    log "Edge computing deployment for $REGION completed successfully!"
}

# Handle script arguments
case "${1:-}" in
    --verify-only)
        verify_deployment
        ;;
    --start-services)
        start_services
        ;;
    --stop-services)
        systemctl stop "bev-edge-node-$REGION" "bev-edge-management-$REGION" "bev-model-sync-$REGION"
        ;;
    --restart-services)
        systemctl restart "bev-edge-node-$REGION" "bev-edge-management-$REGION" "bev-model-sync-$REGION"
        ;;
    --status)
        systemctl status "bev-edge-node-$REGION" "bev-edge-management-$REGION" "bev-model-sync-$REGION"
        ;;
    --uninstall)
        warn "Uninstalling edge computing services for $REGION..."
        systemctl stop "bev-edge-node-$REGION" "bev-edge-management-$REGION" "bev-model-sync-$REGION" || true
        systemctl disable "bev-edge-node-$REGION" "bev-edge-management-$REGION" "bev-model-sync-$REGION" || true
        rm -f "/etc/systemd/system/bev-edge-*-$REGION.service"
        rm -f "/etc/nginx/sites-enabled/bev-edge-$REGION"
        rm -rf "$DEPLOY_DIR" "$LOG_DIR"
        systemctl daemon-reload
        systemctl reload nginx
        log "Uninstallation completed"
        ;;
    *)
        main
        ;;
esac