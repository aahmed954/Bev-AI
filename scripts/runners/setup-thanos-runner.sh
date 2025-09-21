#!/bin/bash
#
# BEV OSINT Framework - THANOS Node Self-Hosted Runner Setup
#
# Configures GitHub Actions self-hosted runner for THANOS node
# Architecture: x86_64, GPU: RTX 3080 (10GB VRAM), Primary compute node
#

set -euo pipefail

# Configuration
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GITHUB_REPO="${GITHUB_REPO:-starlord/Bev}"
RUNNER_NAME="thanos-runner"
RUNNER_WORK_DIR="/opt/actions-runner"
RUNNER_USER="actions-runner"
LABELS="thanos,gpu,x86_64,rtx3080,primary-compute"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[THANOS-RUNNER]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for THANOS runner setup..."

    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi

    # Check GitHub token
    if [[ -z "$GITHUB_TOKEN" ]]; then
        error "GITHUB_TOKEN environment variable must be set"
    fi

    # Check GPU availability
    if ! command -v nvidia-smi &> /dev/null; then
        warn "nvidia-smi not found - GPU support may not be available"
    else
        log "GPU Information:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is required but not installed"
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is required but not installed"
    fi

    success "Prerequisites check completed"
}

# Install runner dependencies
install_dependencies() {
    log "Installing runner dependencies..."

    # Update package list
    apt-get update

    # Install required packages
    apt-get install -y \
        curl \
        wget \
        tar \
        git \
        sudo \
        systemd \
        jq \
        unzip \
        build-essential \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release

    # Install NVIDIA Container Toolkit if not present
    if ! command -v nvidia-container-runtime &> /dev/null; then
        log "Installing NVIDIA Container Toolkit..."

        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
        curl -s -L "https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list" | tee /etc/apt/sources.list.d/nvidia-docker.list

        apt-get update
        apt-get install -y nvidia-container-toolkit nvidia-container-runtime

        # Configure Docker to use NVIDIA runtime
        cat > /etc/docker/daemon.json <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

        systemctl restart docker
        success "NVIDIA Container Toolkit installed and configured"
    fi

    success "Dependencies installation completed"
}

# Create runner user
create_runner_user() {
    log "Creating runner user..."

    if id "$RUNNER_USER" &>/dev/null; then
        warn "User $RUNNER_USER already exists"
    else
        useradd -m -s /bin/bash "$RUNNER_USER"
        usermod -aG docker "$RUNNER_USER"
        usermod -aG sudo "$RUNNER_USER"

        # Allow passwordless sudo for runner user
        echo "$RUNNER_USER ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$RUNNER_USER"

        success "Runner user $RUNNER_USER created"
    fi
}

# Download and install GitHub Actions runner
install_runner() {
    log "Installing GitHub Actions runner..."

    # Create runner directory
    mkdir -p "$RUNNER_WORK_DIR"
    cd "$RUNNER_WORK_DIR"

    # Get latest runner version
    RUNNER_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | jq -r '.tag_name' | sed 's/^v//')

    # Download runner
    log "Downloading runner version $RUNNER_VERSION..."
    wget -O actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"

    # Extract runner
    tar xzf "./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"

    # Set ownership
    chown -R "$RUNNER_USER:$RUNNER_USER" "$RUNNER_WORK_DIR"

    # Clean up
    rm "./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"

    success "GitHub Actions runner installed"
}

# Get registration token
get_registration_token() {
    log "Getting registration token from GitHub..."

    local token_response
    token_response=$(curl -s -X POST \
        -H "Authorization: token $GITHUB_TOKEN" \
        -H "Accept: application/vnd.github.v3+json" \
        "https://api.github.com/repos/$GITHUB_REPO/actions/runners/registration-token")

    local registration_token
    registration_token=$(echo "$token_response" | jq -r '.token')

    if [[ "$registration_token" == "null" ]]; then
        error "Failed to get registration token. Check GitHub token permissions."
    fi

    echo "$registration_token"
}

# Configure runner
configure_runner() {
    log "Configuring GitHub Actions runner..."

    cd "$RUNNER_WORK_DIR"

    # Get registration token
    local registration_token
    registration_token=$(get_registration_token)

    # Configure runner as the runner user
    sudo -u "$RUNNER_USER" ./config.sh \
        --url "https://github.com/$GITHUB_REPO" \
        --token "$registration_token" \
        --name "$RUNNER_NAME" \
        --labels "$LABELS" \
        --work "_work" \
        --unattended \
        --replace

    success "Runner configured"
}

# Install runner as systemd service
install_service() {
    log "Installing runner as systemd service..."

    cd "$RUNNER_WORK_DIR"

    # Install service
    ./svc.sh install "$RUNNER_USER"

    # Enable and start service
    systemctl enable actions.runner.*
    systemctl start actions.runner.*

    # Wait a moment and check status
    sleep 5
    if systemctl is-active --quiet actions.runner.*; then
        success "Runner service is running"
    else
        error "Failed to start runner service"
    fi
}

# Setup GPU monitoring for runner
setup_gpu_monitoring() {
    log "Setting up GPU monitoring for runner..."

    # Create GPU monitoring script
    cat > /usr/local/bin/gpu-monitor.sh <<'EOF'
#!/bin/bash
# GPU monitoring script for THANOS runner

LOGFILE="/var/log/gpu-monitor.log"

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits)
    echo "[$timestamp] GPU: $gpu_info" >> "$LOGFILE"
    sleep 60
done
EOF

    chmod +x /usr/local/bin/gpu-monitor.sh

    # Create systemd service for GPU monitoring
    cat > /etc/systemd/system/gpu-monitor.service <<EOF
[Unit]
Description=GPU Monitoring Service for THANOS Runner
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/gpu-monitor.sh
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable gpu-monitor.service
    systemctl start gpu-monitor.service

    success "GPU monitoring service installed and started"
}

# Create runner configuration file
create_runner_config() {
    log "Creating runner configuration file..."

    cat > /etc/bev-thanos-runner.conf <<EOF
# BEV THANOS Runner Configuration
RUNNER_NAME="$RUNNER_NAME"
RUNNER_WORK_DIR="$RUNNER_WORK_DIR"
RUNNER_USER="$RUNNER_USER"
GITHUB_REPO="$GITHUB_REPO"
LABELS="$LABELS"

# Hardware specifications
GPU_MODEL="RTX3080"
GPU_MEMORY="10GB"
ARCHITECTURE="x86_64"

# Service directories
BEV_PROJECT_DIR="/home/actions-runner/bev"
DOCKER_WORK_DIR="/var/lib/docker"

# Resource limits
MAX_CONCURRENT_JOBS=4
GPU_MEMORY_LIMIT="8GB"
CPU_LIMIT="14"
MEMORY_LIMIT="48GB"
EOF

    success "Runner configuration file created at /etc/bev-thanos-runner.conf"
}

# Verify installation
verify_installation() {
    log "Verifying runner installation..."

    # Check service status
    if systemctl is-active --quiet actions.runner.*; then
        success "✓ Runner service is active"
    else
        error "✗ Runner service is not active"
    fi

    # Check GPU access
    if sudo -u "$RUNNER_USER" nvidia-smi &>/dev/null; then
        success "✓ GPU access verified for runner user"
    else
        warn "✗ GPU access not available for runner user"
    fi

    # Check Docker access
    if sudo -u "$RUNNER_USER" docker ps &>/dev/null; then
        success "✓ Docker access verified for runner user"
    else
        error "✗ Docker access not available for runner user"
    fi

    # Check disk space
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    if [[ $disk_usage -lt 80 ]]; then
        success "✓ Disk usage: ${disk_usage}% (adequate)"
    else
        warn "✗ Disk usage: ${disk_usage}% (may need attention)"
    fi

    log "Runner verification completed"
}

# Print runner information
print_runner_info() {
    log "THANOS Runner Installation Summary:"
    echo ""
    echo "  Runner Name: $RUNNER_NAME"
    echo "  Work Directory: $RUNNER_WORK_DIR"
    echo "  User: $RUNNER_USER"
    echo "  Labels: $LABELS"
    echo "  Repository: $GITHUB_REPO"
    echo ""
    echo "Hardware Specifications:"
    if command -v nvidia-smi &> /dev/null; then
        echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
        echo "  GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
    fi
    echo "  Architecture: $(uname -m)"
    echo "  CPU Cores: $(nproc)"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    echo ""
    echo "Service Management:"
    echo "  Start:   sudo systemctl start actions.runner.*"
    echo "  Stop:    sudo systemctl stop actions.runner.*"
    echo "  Status:  sudo systemctl status actions.runner.*"
    echo "  Logs:    sudo journalctl -f -u actions.runner.*"
    echo ""
    echo "GPU Monitoring:"
    echo "  Log:     tail -f /var/log/gpu-monitor.log"
    echo "  Status:  sudo systemctl status gpu-monitor"
    echo ""
    success "THANOS runner setup completed successfully!"
}

# Main execution
main() {
    log "Starting THANOS runner setup..."

    check_prerequisites
    install_dependencies
    create_runner_user
    install_runner
    configure_runner
    install_service
    setup_gpu_monitoring
    create_runner_config
    verify_installation
    print_runner_info
}

# Handle script interruption
trap 'error "Script interrupted"' INT TERM

# Execute main function
main "$@"