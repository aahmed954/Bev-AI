#!/bin/bash
#
# BEV OSINT Framework - ORACLE1 Node Self-Hosted Runner Setup
#
# Configures GitHub Actions self-hosted runner for ORACLE1 node
# Architecture: ARM64 (aarch64), Cloud-hosted, Monitoring and coordination
#

set -euo pipefail

# Configuration
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GITHUB_REPO="${GITHUB_REPO:-starlord/Bev}"
RUNNER_NAME="oracle1-runner"
RUNNER_WORK_DIR="/opt/actions-runner"
RUNNER_USER="actions-runner"
LABELS="oracle1,arm64,aarch64,monitoring,cloud-hosted"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[ORACLE1-RUNNER]${NC} $1"
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
    log "Checking prerequisites for ORACLE1 runner setup..."

    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi

    # Check GitHub token
    if [[ -z "$GITHUB_TOKEN" ]]; then
        error "GITHUB_TOKEN environment variable must be set"
    fi

    # Verify ARM64 architecture
    local arch
    arch=$(uname -m)
    if [[ "$arch" != "aarch64" && "$arch" != "arm64" ]]; then
        error "This script is designed for ARM64 architecture, detected: $arch"
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is required but not installed"
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is required but not installed"
    fi

    # Check available memory (ARM systems often have limited RAM)
    local mem_gb
    mem_gb=$(free -g | awk '/^Mem:/ {print $2}')
    if [[ $mem_gb -lt 4 ]]; then
        warn "Available memory is ${mem_gb}GB - may be insufficient for BEV operations"
    fi

    success "Prerequisites check completed for ARM64 architecture"
}

# Install runner dependencies for ARM64
install_dependencies() {
    log "Installing runner dependencies for ARM64..."

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
        lsb-release \
        htop \
        iotop \
        net-tools

    # Install ARM64-specific monitoring tools
    apt-get install -y \
        sysstat \
        iostat \
        nmon \
        stress-ng

    success "Dependencies installation completed for ARM64"
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

# Download and install GitHub Actions runner for ARM64
install_runner() {
    log "Installing GitHub Actions runner for ARM64..."

    # Create runner directory
    mkdir -p "$RUNNER_WORK_DIR"
    cd "$RUNNER_WORK_DIR"

    # Get latest runner version
    RUNNER_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | jq -r '.tag_name' | sed 's/^v//')

    # Download ARM64 runner
    log "Downloading ARM64 runner version $RUNNER_VERSION..."
    wget -O actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz \
        "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"

    # Extract runner
    tar xzf "./actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"

    # Set ownership
    chown -R "$RUNNER_USER:$RUNNER_USER" "$RUNNER_WORK_DIR"

    # Clean up
    rm "./actions-runner-linux-arm64-${RUNNER_VERSION}.tar.gz"

    success "GitHub Actions ARM64 runner installed"
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

# Setup ARM64 system monitoring
setup_arm64_monitoring() {
    log "Setting up ARM64 system monitoring..."

    # Create ARM64 monitoring script
    cat > /usr/local/bin/arm64-monitor.sh <<'EOF'
#!/bin/bash
# ARM64 system monitoring script for ORACLE1 runner

LOGFILE="/var/log/arm64-monitor.log"

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # CPU usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)

    # Memory usage
    mem_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')

    # Load average
    load_avg=$(uptime | awk -F'load average:' '{print $2}')

    # Temperature (if available)
    temp="N/A"
    if [[ -f "/sys/class/thermal/thermal_zone0/temp" ]]; then
        temp_raw=$(cat /sys/class/thermal/thermal_zone0/temp)
        temp=$((temp_raw / 1000))
    fi

    # Disk usage
    disk_usage=$(df -h / | awk 'NR==2{print $5}')

    echo "[$timestamp] CPU: ${cpu_usage}%, Memory: ${mem_usage}%, Load:${load_avg}, Temp: ${temp}°C, Disk: ${disk_usage}" >> "$LOGFILE"
    sleep 60
done
EOF

    chmod +x /usr/local/bin/arm64-monitor.sh

    # Create systemd service for ARM64 monitoring
    cat > /etc/systemd/system/arm64-monitor.service <<EOF
[Unit]
Description=ARM64 System Monitoring Service for ORACLE1 Runner
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/arm64-monitor.sh
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable arm64-monitor.service
    systemctl start arm64-monitor.service

    success "ARM64 monitoring service installed and started"
}

# Setup memory optimization for ARM64
setup_memory_optimization() {
    log "Setting up memory optimization for ARM64..."

    # Create swap if not exists and memory is limited
    local mem_gb
    mem_gb=$(free -g | awk '/^Mem:/ {print $2}')

    if [[ $mem_gb -le 8 ]] && [[ ! -f /swapfile ]]; then
        log "Creating swap file for memory-constrained ARM64 system..."

        # Create 4GB swap file
        fallocate -l 4G /swapfile
        chmod 600 /swapfile
        mkswap /swapfile
        swapon /swapfile

        # Add to fstab
        echo '/swapfile none swap sw 0 0' >> /etc/fstab

        # Optimize swappiness for ARM64
        echo 'vm.swappiness=10' >> /etc/sysctl.conf
        sysctl vm.swappiness=10

        success "Swap file created and optimized for ARM64"
    fi

    # Setup memory monitoring for ARM64
    cat > /usr/local/bin/memory-pressure-monitor.sh <<'EOF'
#!/bin/bash
# Memory pressure monitoring for ARM64

THRESHOLD=85
LOGFILE="/var/log/memory-pressure.log"

while true; do
    mem_usage=$(free | grep Mem | awk '{printf "%d", $3/$2 * 100.0}')

    if [[ $mem_usage -gt $THRESHOLD ]]; then
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        echo "[$timestamp] High memory usage: ${mem_usage}%" >> "$LOGFILE"

        # Log top memory consumers
        echo "Top memory consumers:" >> "$LOGFILE"
        ps aux --sort=-%mem | head -10 >> "$LOGFILE"
        echo "---" >> "$LOGFILE"
    fi

    sleep 30
done
EOF

    chmod +x /usr/local/bin/memory-pressure-monitor.sh

    # Create systemd service for memory monitoring
    cat > /etc/systemd/system/memory-pressure-monitor.service <<EOF
[Unit]
Description=Memory Pressure Monitoring for ARM64
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/memory-pressure-monitor.sh
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable memory-pressure-monitor.service
    systemctl start memory-pressure-monitor.service

    success "Memory optimization and monitoring configured"
}

# Create runner configuration file
create_runner_config() {
    log "Creating runner configuration file..."

    cat > /etc/bev-oracle1-runner.conf <<EOF
# BEV ORACLE1 Runner Configuration
RUNNER_NAME="$RUNNER_NAME"
RUNNER_WORK_DIR="$RUNNER_WORK_DIR"
RUNNER_USER="$RUNNER_USER"
GITHUB_REPO="$GITHUB_REPO"
LABELS="$LABELS"

# Hardware specifications
ARCHITECTURE="ARM64"
CPU_CORES="$(nproc)"
MEMORY_GB="$(free -g | awk '/^Mem:/ {print $2}')"

# Service directories
BEV_PROJECT_DIR="/home/actions-runner/bev"
DOCKER_WORK_DIR="/var/lib/docker"

# Resource limits for ARM64
MAX_CONCURRENT_JOBS=2
CPU_LIMIT="$(nproc)"
MEMORY_LIMIT="$(free -g | awk '/^Mem:/ {print $2}')GB"

# Monitoring endpoints
PROMETHEUS_ENDPOINT="http://localhost:9090"
GRAFANA_ENDPOINT="http://localhost:3000"
CONSUL_ENDPOINT="http://localhost:8500"
EOF

    success "Runner configuration file created at /etc/bev-oracle1-runner.conf"
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

    # Check memory availability
    local mem_usage
    mem_usage=$(free | grep Mem | awk '{printf "%d", $3/$2 * 100.0}')
    if [[ $mem_usage -lt 70 ]]; then
        success "✓ Memory usage: ${mem_usage}% (good)"
    else
        warn "✗ Memory usage: ${mem_usage}% (monitor closely)"
    fi

    # Check ARM64 specific features
    local cpu_features
    cpu_features=$(lscpu | grep "Flags" || echo "ARM64 CPU features available")
    success "✓ ARM64 architecture verified"

    log "Runner verification completed"
}

# Print runner information
print_runner_info() {
    log "ORACLE1 Runner Installation Summary:"
    echo ""
    echo "  Runner Name: $RUNNER_NAME"
    echo "  Work Directory: $RUNNER_WORK_DIR"
    echo "  User: $RUNNER_USER"
    echo "  Labels: $LABELS"
    echo "  Repository: $GITHUB_REPO"
    echo ""
    echo "Hardware Specifications:"
    echo "  Architecture: $(uname -m)"
    echo "  CPU Cores: $(nproc)"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    if [[ -f "/sys/class/thermal/thermal_zone0/temp" ]]; then
        local temp_raw temp
        temp_raw=$(cat /sys/class/thermal/thermal_zone0/temp)
        temp=$((temp_raw / 1000))
        echo "  Temperature: ${temp}°C"
    fi
    echo ""
    echo "Service Management:"
    echo "  Start:   sudo systemctl start actions.runner.*"
    echo "  Stop:    sudo systemctl stop actions.runner.*"
    echo "  Status:  sudo systemctl status actions.runner.*"
    echo "  Logs:    sudo journalctl -f -u actions.runner.*"
    echo ""
    echo "ARM64 Monitoring:"
    echo "  System:  tail -f /var/log/arm64-monitor.log"
    echo "  Memory:  tail -f /var/log/memory-pressure.log"
    echo "  Status:  sudo systemctl status arm64-monitor"
    echo ""
    echo "BEV Services (when deployed):"
    echo "  Prometheus: http://localhost:9090"
    echo "  Grafana:    http://localhost:3000"
    echo "  Consul:     http://localhost:8500"
    echo ""
    success "ORACLE1 runner setup completed successfully!"
}

# Main execution
main() {
    log "Starting ORACLE1 ARM64 runner setup..."

    check_prerequisites
    install_dependencies
    create_runner_user
    install_runner
    configure_runner
    install_service
    setup_arm64_monitoring
    setup_memory_optimization
    create_runner_config
    verify_installation
    print_runner_info
}

# Handle script interruption
trap 'error "Script interrupted"' INT TERM

# Execute main function
main "$@"