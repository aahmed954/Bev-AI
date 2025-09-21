#!/bin/bash
#
# BEV OSINT Framework - STARLORD Node Self-Hosted Runner Setup
#
# Configures GitHub Actions self-hosted runner for STARLORD node
# Architecture: x86_64, GPU: RTX 4090 (24GB VRAM), AI companion development
#

set -euo pipefail

# Configuration
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
GITHUB_REPO="${GITHUB_REPO:-starlord/Bev}"
RUNNER_NAME="starlord-runner"
RUNNER_WORK_DIR="/opt/actions-runner"
RUNNER_USER="actions-runner"
LABELS="starlord,gpu,x86_64,rtx4090,ai-companion,development"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[STARLORD-RUNNER]${NC} $1"
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

ai_log() {
    echo -e "${PURPLE}[AI-COMPANION]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites for STARLORD runner setup..."

    # Check if running as root
    if [[ $EUID -ne 0 ]]; then
        error "This script must be run as root"
    fi

    # Check GitHub token
    if [[ -z "$GITHUB_TOKEN" ]]; then
        error "GITHUB_TOKEN environment variable must be set"
    fi

    # Check GPU availability (RTX 4090 specific)
    if ! command -v nvidia-smi &> /dev/null; then
        warn "nvidia-smi not found - GPU support may not be available"
    else
        log "GPU Information:"
        local gpu_info
        gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
        echo "  $gpu_info"

        # Check for RTX 4090 specifically
        if echo "$gpu_info" | grep -qi "4090"; then
            success "RTX 4090 detected - optimal for AI companion development"
        else
            warn "RTX 4090 not detected - AI companion performance may be limited"
        fi
    fi

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is required but not installed"
    fi

    # Check Python for AI companion
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required for AI companion development"
    fi

    # Check CUDA for AI workloads
    if ! command -v nvcc &> /dev/null; then
        warn "CUDA toolkit not found - AI development may be limited"
    fi

    success "Prerequisites check completed"
}

# Install runner dependencies
install_dependencies() {
    log "Installing runner dependencies with AI development tools..."

    # Update package list
    apt-get update

    # Install base packages
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

    # Install AI/ML development tools
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-setuptools \
        ffmpeg \
        libsm6 \
        libxext6 \
        libfontconfig1 \
        libxrender1 \
        libgl1-mesa-glx \
        libasound2-dev \
        portaudio19-dev

    # Install NVIDIA Container Toolkit if not present
    if ! command -v nvidia-container-runtime &> /dev/null; then
        log "Installing NVIDIA Container Toolkit for RTX 4090..."

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
        success "NVIDIA Container Toolkit installed and configured for RTX 4090"
    fi

    # Install Node.js for frontend development
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs

    success "Dependencies installation completed with AI development tools"
}

# Create runner user with AI development setup
create_runner_user() {
    log "Creating runner user with AI development environment..."

    if id "$RUNNER_USER" &>/dev/null; then
        warn "User $RUNNER_USER already exists"
    else
        useradd -m -s /bin/bash "$RUNNER_USER"
        usermod -aG docker "$RUNNER_USER"
        usermod -aG sudo "$RUNNER_USER"
        usermod -aG audio "$RUNNER_USER"
        usermod -aG video "$RUNNER_USER"

        # Allow passwordless sudo for runner user
        echo "$RUNNER_USER ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/$RUNNER_USER"

        success "Runner user $RUNNER_USER created with multimedia access"
    fi

    # Setup AI development environment for runner user
    sudo -u "$RUNNER_USER" bash <<'EOF'
# Create virtual environment for AI development
python3 -m venv /home/actions-runner/ai-env
source /home/actions-runner/ai-env/bin/activate

# Install AI/ML packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes
pip install opencv-python Pillow numpy scipy
pip install fastapi uvicorn websockets
pip install streamlit gradio

# Create AI workspace
mkdir -p /home/actions-runner/ai-workspace
mkdir -p /home/actions-runner/ai-models
mkdir -p /home/actions-runner/ai-datasets

echo "source /home/actions-runner/ai-env/bin/activate" >> /home/actions-runner/.bashrc
EOF

    ai_log "AI development environment configured for $RUNNER_USER"
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

# Setup RTX 4090 monitoring and optimization
setup_gpu_monitoring() {
    log "Setting up RTX 4090 monitoring and optimization..."

    # Create GPU monitoring script optimized for RTX 4090
    cat > /usr/local/bin/rtx4090-monitor.sh <<'EOF'
#!/bin/bash
# RTX 4090 monitoring and optimization script for STARLORD runner

LOGFILE="/var/log/rtx4090-monitor.log"
GPU_MEMORY_LIMIT=22000  # 22GB out of 24GB to prevent OOM

while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Comprehensive GPU monitoring
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    gpu_mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    gpu_mem_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    gpu_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits)
    gpu_clock=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits)

    # Memory percentage
    gpu_mem_percent=$(echo "scale=1; $gpu_mem_used * 100 / $gpu_mem_total" | bc)

    # Log comprehensive GPU status
    echo "[$timestamp] RTX4090 - Util: ${gpu_util}%, Mem: ${gpu_mem_used}MB/${gpu_mem_total}MB (${gpu_mem_percent}%), Temp: ${gpu_temp}°C, Power: ${gpu_power}W, Clock: ${gpu_clock}MHz" >> "$LOGFILE"

    # Check for memory pressure and take action
    if [[ $gpu_mem_used -gt $GPU_MEMORY_LIMIT ]]; then
        echo "[$timestamp] WARNING: GPU memory usage high (${gpu_mem_used}MB > ${GPU_MEMORY_LIMIT}MB)" >> "$LOGFILE"
        # Could implement automatic cleanup here if needed
    fi

    # Check for thermal throttling
    if [[ $gpu_temp -gt 80 ]]; then
        echo "[$timestamp] WARNING: GPU temperature high (${gpu_temp}°C)" >> "$LOGFILE"
    fi

    sleep 30
done
EOF

    chmod +x /usr/local/bin/rtx4090-monitor.sh

    # Install bc for calculations
    apt-get install -y bc

    # Create systemd service for RTX 4090 monitoring
    cat > /etc/systemd/system/rtx4090-monitor.service <<EOF
[Unit]
Description=RTX 4090 Monitoring Service for STARLORD Runner
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/rtx4090-monitor.sh
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable rtx4090-monitor.service
    systemctl start rtx4090-monitor.service

    success "RTX 4090 monitoring service installed and started"
}

# Setup AI companion monitoring
setup_ai_companion_monitoring() {
    ai_log "Setting up AI companion monitoring..."

    # Create AI companion monitoring script
    cat > /usr/local/bin/ai-companion-monitor.sh <<'EOF'
#!/bin/bash
# AI companion monitoring script for STARLORD

LOGFILE="/var/log/ai-companion.log"
AI_ENV="/home/actions-runner/ai-env"

monitor_ai_processes() {
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Check for AI-related processes
    python_procs=$(ps aux | grep -E "(python|torch|transformers)" | grep -v grep | wc -l)
    gpu_procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits | wc -l)

    # Check AI environment
    ai_env_status="inactive"
    if [[ -d "$AI_ENV" ]]; then
        ai_env_status="available"
    fi

    # Log AI companion status
    echo "[$timestamp] AI Companion - Python processes: $python_procs, GPU processes: $gpu_procs, Environment: $ai_env_status" >> "$LOGFILE"

    # Check for AI model downloads
    if [[ -d "/home/actions-runner/ai-models" ]]; then
        model_count=$(find /home/actions-runner/ai-models -name "*.bin" -o -name "*.safetensors" | wc -l)
        if [[ $model_count -gt 0 ]]; then
            echo "[$timestamp] AI Models: $model_count models available" >> "$LOGFILE"
        fi
    fi
}

# Monitor AI companion every 2 minutes
while true; do
    monitor_ai_processes
    sleep 120
done
EOF

    chmod +x /usr/local/bin/ai-companion-monitor.sh

    # Create systemd service for AI companion monitoring
    cat > /etc/systemd/system/ai-companion-monitor.service <<EOF
[Unit]
Description=AI Companion Monitoring Service
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/ai-companion-monitor.sh
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable ai-companion-monitor.service
    systemctl start ai-companion-monitor.service

    ai_log "AI companion monitoring service installed and started"
}

# Create runner configuration file
create_runner_config() {
    log "Creating runner configuration file..."

    cat > /etc/bev-starlord-runner.conf <<EOF
# BEV STARLORD Runner Configuration
RUNNER_NAME="$RUNNER_NAME"
RUNNER_WORK_DIR="$RUNNER_WORK_DIR"
RUNNER_USER="$RUNNER_USER"
GITHUB_REPO="$GITHUB_REPO"
LABELS="$LABELS"

# Hardware specifications
GPU_MODEL="RTX4090"
GPU_MEMORY="24GB"
ARCHITECTURE="x86_64"

# AI Development Environment
AI_ENV_PATH="/home/actions-runner/ai-env"
AI_WORKSPACE="/home/actions-runner/ai-workspace"
AI_MODELS_DIR="/home/actions-runner/ai-models"

# Service directories
BEV_PROJECT_DIR="/home/actions-runner/bev"
DOCKER_WORK_DIR="/var/lib/docker"

# Resource limits optimized for RTX 4090
MAX_CONCURRENT_JOBS=6
GPU_MEMORY_LIMIT="22GB"
CPU_LIMIT="16"
MEMORY_LIMIT="64GB"

# AI Companion specific
AI_MODEL_MAX_SIZE="20GB"
INFERENCE_BATCH_SIZE="8"
MAX_CONTEXT_LENGTH="32768"
EOF

    success "Runner configuration file created at /etc/bev-starlord-runner.conf"
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

        # Verify RTX 4090 specifically
        local gpu_name
        gpu_name=$(sudo -u "$RUNNER_USER" nvidia-smi --query-gpu=name --format=csv,noheader)
        if echo "$gpu_name" | grep -qi "4090"; then
            success "✓ RTX 4090 access confirmed"
        else
            warn "✗ RTX 4090 not detected: $gpu_name"
        fi
    else
        warn "✗ GPU access not available for runner user"
    fi

    # Check Docker access
    if sudo -u "$RUNNER_USER" docker ps &>/dev/null; then
        success "✓ Docker access verified for runner user"
    else
        error "✗ Docker access not available for runner user"
    fi

    # Check AI environment
    if [[ -d "/home/actions-runner/ai-env" ]]; then
        success "✓ AI development environment available"

        # Test PyTorch GPU access
        if sudo -u "$RUNNER_USER" bash -c "source /home/actions-runner/ai-env/bin/activate && python -c 'import torch; print(f\"PyTorch GPU available: {torch.cuda.is_available()}\")'" 2>/dev/null | grep -q "True"; then
            success "✓ PyTorch GPU acceleration verified"
        else
            warn "✗ PyTorch GPU acceleration not working"
        fi
    else
        warn "✗ AI development environment not found"
    fi

    # Check disk space for AI models
    local disk_usage
    disk_usage=$(df -h / | awk 'NR==2{print $5}' | sed 's/%//')
    if [[ $disk_usage -lt 70 ]]; then
        success "✓ Disk usage: ${disk_usage}% (adequate for AI models)"
    else
        warn "✗ Disk usage: ${disk_usage}% (may limit AI model storage)"
    fi

    log "Runner verification completed"
}

# Print runner information
print_runner_info() {
    log "STARLORD Runner Installation Summary:"
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
        echo "  GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
        echo "  CUDA Version: $(nvidia-smi | grep "CUDA Version" | awk '{print $9}')"
    fi
    echo "  Architecture: $(uname -m)"
    echo "  CPU Cores: $(nproc)"
    echo "  Memory: $(free -h | awk '/^Mem:/ {print $2}')"
    echo ""
    echo "AI Development Environment:"
    echo "  Python Environment: /home/actions-runner/ai-env"
    echo "  AI Workspace: /home/actions-runner/ai-workspace"
    echo "  Models Directory: /home/actions-runner/ai-models"
    echo "  Activate: source /home/actions-runner/ai-env/bin/activate"
    echo ""
    echo "Service Management:"
    echo "  Start:   sudo systemctl start actions.runner.*"
    echo "  Stop:    sudo systemctl stop actions.runner.*"
    echo "  Status:  sudo systemctl status actions.runner.*"
    echo "  Logs:    sudo journalctl -f -u actions.runner.*"
    echo ""
    echo "Monitoring:"
    echo "  RTX 4090:     tail -f /var/log/rtx4090-monitor.log"
    echo "  AI Companion: tail -f /var/log/ai-companion.log"
    echo "  GPU Status:   nvidia-smi"
    echo ""
    echo "AI Companion Quick Start:"
    echo "  1. sudo -u actions-runner bash"
    echo "  2. source /home/actions-runner/ai-env/bin/activate"
    echo "  3. cd /home/actions-runner/ai-workspace"
    echo "  4. python -c \"import torch; print(f'GPU: {torch.cuda.get_device_name()}')\""
    echo ""
    success "STARLORD runner setup completed successfully!"
    ai_log "Ready for AI companion development with RTX 4090 acceleration!"
}

# Main execution
main() {
    log "Starting STARLORD AI development runner setup..."

    check_prerequisites
    install_dependencies
    create_runner_user
    install_runner
    configure_runner
    install_service
    setup_gpu_monitoring
    setup_ai_companion_monitoring
    create_runner_config
    verify_installation
    print_runner_info
}

# Handle script interruption
trap 'error "Script interrupted"' INT TERM

# Execute main function
main "$@"