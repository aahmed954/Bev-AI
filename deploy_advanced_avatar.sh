#!/bin/bash

# BEV Advanced Avatar System - Complete Deployment Script
# Deploys state-of-the-art 3D avatar system with RTX 4090 optimization

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

banner() {
    echo -e "${PURPLE}"
    cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██████╗ ███████╗██╗   ██╗     █████╗ ██╗   ██╗ █████╗      ║
║   ██╔══██╗██╔════╝██║   ██║    ██╔══██╗██║   ██║██╔══██╗     ║
║   ██████╔╝█████╗  ██║   ██║    ███████║██║   ██║███████║     ║
║   ██╔══██╗██╔══╝  ╚██╗ ██╔╝    ██╔══██║╚██╗ ██╔╝██╔══██║     ║
║   ██████╔╝███████╗ ╚████╔╝     ██║  ██║ ╚████╔╝ ██║  ██║     ║
║   ╚═════╝ ╚══════╝  ╚═══╝      ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝      ║
║                                                               ║
║            ADVANCED AVATAR SYSTEM DEPLOYMENT                 ║
║        Gaussian Splatting + MetaHuman + RTX 4090            ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_prerequisites() {
    log "Checking deployment prerequisites..."

    local failed_checks=0

    # Check if running on STARLORD
    if [[ "$(hostname)" != "starlord"* ]]; then
        error "This script must be run on STARLORD workstation"
        ((failed_checks++))
    fi

    # Check RTX 4090 availability
    if nvidia-smi | grep -q "RTX 4090"; then
        success "RTX 4090 detected"
    else
        error "RTX 4090 not found - this system requires RTX 4090"
        ((failed_checks++))
    fi

    # Check Docker
    if command -v docker >/dev/null 2>&1; then
        success "Docker available"
    else
        error "Docker not installed"
        ((failed_checks++))
    fi

    # Check CUDA
    if command -v nvcc >/dev/null 2>&1; then
        local cuda_version=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
        success "CUDA $cuda_version available"
    else
        error "CUDA not installed"
        ((failed_checks++))
    fi

    # Check Python 3.11+
    if python3 --version | grep -E "3\.1[1-9]" >/dev/null; then
        success "Python 3.11+ available"
    else
        error "Python 3.11+ required"
        ((failed_checks++))
    fi

    # Check available GPU memory
    local gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    if [[ $gpu_memory -ge 20000 ]]; then
        success "Sufficient GPU memory: ${gpu_memory}MB"
    else
        error "Insufficient GPU memory: ${gpu_memory}MB (need 20GB+)"
        ((failed_checks++))
    fi

    # Check available system memory
    local sys_memory=$(free -g | awk 'NR==2{print $2}')
    if [[ $sys_memory -ge 16 ]]; then
        success "Sufficient system memory: ${sys_memory}GB"
    else
        warning "Limited system memory: ${sys_memory}GB (recommended 32GB+)"
    fi

    if [[ $failed_checks -gt 0 ]]; then
        error "Failed $failed_checks prerequisite checks"
        return 1
    fi

    success "All prerequisites met"
    return 0
}

install_dependencies() {
    log "Installing avatar system dependencies..."

    # Create Python virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip wheel setuptools

    # Install PyTorch with CUDA support
    log "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    # Install gsplat for Gaussian Splatting
    log "Installing gsplat for 3D rendering..."
    pip install git+https://github.com/nerfstudio-project/gsplat.git

    # Install Bark AI for advanced TTS
    log "Installing Bark AI for voice synthesis..."
    pip install git+https://github.com/suno-ai/bark.git

    # Install avatar system requirements
    log "Installing avatar system requirements..."
    pip install -r src/avatar/requirements-avatar.txt

    # Install GPU monitoring
    pip install pynvml gpustat

    success "Dependencies installed successfully"
}

build_avatar_system() {
    log "Building advanced avatar system..."

    # Build Docker image for avatar system
    log "Building avatar Docker image..."
    docker build -t bev-advanced-avatar:latest src/avatar/

    # Create necessary directories
    mkdir -p {models,cache,logs}/avatar
    mkdir -p models/metahuman
    mkdir -p cache/{audio,renders}

    # Set permissions
    chmod -R 755 models cache logs
    chown -R starlord:starlord models cache logs

    success "Avatar system built successfully"
}

configure_gpu_optimization() {
    log "Configuring RTX 4090 optimization..."

    # Enable persistence mode
    sudo nvidia-smi -pm 1

    # Set optimal power limit (450W for RTX 4090)
    sudo nvidia-smi -pl 450

    # Configure GPU for maximum performance
    sudo nvidia-smi -lgc 2230,2230  # Lock GPU clock to max
    sudo nvidia-smi -lmc 10501,10501  # Lock memory clock to max

    # Set aggressive cooling
    sudo nvidia-smi -fcs 80  # 80% fan speed

    success "RTX 4090 optimization configured"
}

deploy_avatar_service() {
    log "Deploying avatar service..."

    # Copy systemd service files
    sudo cp systemd/bev-advanced-avatar.service /etc/systemd/system/
    sudo systemctl daemon-reload

    # Install and configure service
    cd systemd
    sudo ./install-avatar-service.sh

    # Enable auto-start
    sudo systemctl enable bev-advanced-avatar

    success "Avatar service deployed and configured"
}

start_avatar_system() {
    log "Starting advanced avatar system..."

    # Start avatar service
    sudo systemctl start bev-advanced-avatar

    # Wait for service to be ready
    local max_wait=60
    local wait_time=0

    while [[ $wait_time -lt $max_wait ]]; do
        if systemctl is-active --quiet bev-advanced-avatar; then
            success "Avatar service started successfully"
            break
        fi

        log "Waiting for avatar service to start..."
        sleep 2
        ((wait_time += 2))
    done

    if [[ $wait_time -ge $max_wait ]]; then
        error "Avatar service failed to start within $max_wait seconds"
        return 1
    fi

    # Check service health
    if curl -f http://localhost:8091/health >/dev/null 2>&1; then
        success "Avatar service health check passed"
    else
        warning "Avatar service health check failed"
    fi

    return 0
}

run_system_tests() {
    log "Running avatar system tests..."

    # Run quick validation test
    cd src/avatar
    if python3 test_avatar_system.py quick; then
        success "Quick validation tests passed"
    else
        warning "Some validation tests failed"
    fi

    # Test GPU performance
    log "Testing RTX 4090 performance..."
    if python3 rtx4090_optimizer.py; then
        success "GPU performance tests passed"
    else
        warning "GPU performance tests failed"
    fi

    cd "$PROJECT_ROOT"
}

print_access_information() {
    echo ""
    echo -e "${GREEN}=== BEV Advanced Avatar System - Access Information ===${NC}"
    echo -e "${CYAN}Avatar WebSocket:${NC} ws://localhost:8091/ws"
    echo -e "${CYAN}Avatar Health:${NC} http://localhost:8091/health"
    echo -e "${CYAN}Avatar API:${NC} http://localhost:8091"
    echo -e "${CYAN}System Status:${NC} systemctl status bev-advanced-avatar"
    echo -e "${CYAN}Service Logs:${NC} journalctl -u bev-advanced-avatar -f"
    echo -e "${CYAN}GPU Monitoring:${NC} nvidia-smi"
    echo ""
    echo -e "${YELLOW}Service Management Commands:${NC}"
    echo -e "${CYAN}  Start:${NC} sudo systemctl start bev-advanced-avatar"
    echo -e "${CYAN}  Stop:${NC} sudo systemctl stop bev-advanced-avatar"
    echo -e "${CYAN}  Restart:${NC} sudo systemctl restart bev-advanced-avatar"
    echo -e "${CYAN}  Status:${NC} systemctl status bev-advanced-avatar"
    echo -e "${CYAN}  Logs:${NC} journalctl -u bev-advanced-avatar -f"
    echo ""
    echo -e "${GREEN}Avatar system ready for OSINT research operations!${NC}"
    echo -e "${GREEN}===============================================${NC}"
}

main() {
    banner

    log "Starting BEV Advanced Avatar System deployment..."

    # Run deployment steps
    if ! check_prerequisites; then
        error "Prerequisites check failed"
        exit 1
    fi

    if ! install_dependencies; then
        error "Dependency installation failed"
        exit 1
    fi

    if ! build_avatar_system; then
        error "Avatar system build failed"
        exit 1
    fi

    configure_gpu_optimization || warning "GPU optimization partially failed"

    if ! deploy_avatar_service; then
        error "Service deployment failed"
        exit 1
    fi

    if ! start_avatar_system; then
        error "Avatar system startup failed"
        exit 1
    fi

    run_system_tests || warning "Some system tests failed"

    success "🎉 BEV Advanced Avatar System deployment complete!"
    print_access_information
}

# Handle script interruption
trap 'error "Deployment interrupted"; exit 130' INT TERM

# Execute main deployment
main "$@"