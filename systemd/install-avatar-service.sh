#!/bin/bash
#
# BEV Advanced Avatar Service Installation Script
# Installs and configures the systemd service with RTX 4090 optimization
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEV_ROOT="/home/starlord/Projects/Bev"
SERVICE_NAME="bev-advanced-avatar"
SERVICE_FILE="$SCRIPT_DIR/$SERVICE_NAME.service"
SYSTEMD_DIR="/etc/systemd/system"
LOG_FILE="$BEV_ROOT/logs/service-installation.log"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

# Success message
success() {
    log "${GREEN}✓ $*${NC}"
}

# Warning message
warn() {
    log "${YELLOW}⚠ $*${NC}"
}

# Error message
error() {
    log "${RED}✗ $*${NC}"
}

# Info message
info() {
    log "${BLUE}ℹ $*${NC}"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root. Run as user 'starlord'."
        exit 1
    fi
}

# Function to validate system requirements
validate_requirements() {
    info "Validating system requirements..."
    
    # Check if user is starlord
    if [[ "$USER" != "starlord" ]]; then
        error "Script must be run as user 'starlord', currently running as '$USER'"
        exit 1
    fi
    
    # Check if systemd is available
    if ! command -v systemctl &>/dev/null; then
        error "systemd not available"
        exit 1
    fi
    
    # Check if sudo is available
    if ! command -v sudo &>/dev/null; then
        error "sudo not available"
        exit 1
    fi
    
    # Check sudo access
    if ! sudo -n true 2>/dev/null; then
        error "User does not have sudo access or sudo requires password"
        error "Please ensure 'starlord' user has passwordless sudo for systemd operations"
        exit 1
    fi
    
    # Check if BEV directory exists
    if [[ ! -d "$BEV_ROOT" ]]; then
        error "BEV root directory not found: $BEV_ROOT"
        exit 1
    fi
    
    # Check if service file exists
    if [[ ! -f "$SERVICE_FILE" ]]; then
        error "Service file not found: $SERVICE_FILE"
        exit 1
    fi
    
    # Check if scripts exist and are executable
    local scripts=(
        "$SCRIPT_DIR/scripts/pre-start-validation.sh"
        "$SCRIPT_DIR/scripts/gpu-check.sh"
        "$SCRIPT_DIR/scripts/dependencies-check.sh"
        "$SCRIPT_DIR/scripts/start-avatar.sh"
        "$SCRIPT_DIR/scripts/stop-avatar.sh"
        "$SCRIPT_DIR/scripts/cleanup-gpu.sh"
    )
    
    for script in "${scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            error "Required script not found: $script"
            exit 1
        fi
    done
    
    success "System requirements validated"
}

# Function to check NVIDIA driver and GPU
validate_gpu() {
    info "Validating RTX 4090 and NVIDIA drivers..."
    
    if ! command -v nvidia-smi &>/dev/null; then
        error "NVIDIA drivers not installed or nvidia-smi not available"
        exit 1
    fi
    
    # Check GPU
    local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "Unknown")
    info "Detected GPU: $gpu_name"
    
    if [[ "$gpu_name" != *"RTX 4090"* ]]; then
        warn "Expected RTX 4090, found: $gpu_name"
        warn "Service will continue but may not be optimally configured"
    fi
    
    # Check CUDA
    if [[ ! -d "/usr/local/cuda" ]]; then
        error "CUDA installation not found at /usr/local/cuda"
        exit 1
    fi
    
    local cuda_version=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | cut -d',' -f1 || echo "Unknown")
    info "CUDA version: $cuda_version"
    
    success "GPU and CUDA validation completed"
}

# Function to make scripts executable
setup_script_permissions() {
    info "Setting up script permissions..."
    
    local scripts=(
        "$SCRIPT_DIR/scripts/pre-start-validation.sh"
        "$SCRIPT_DIR/scripts/gpu-check.sh"
        "$SCRIPT_DIR/scripts/dependencies-check.sh"
        "$SCRIPT_DIR/scripts/start-avatar.sh"
        "$SCRIPT_DIR/scripts/stop-avatar.sh"
        "$SCRIPT_DIR/scripts/cleanup-gpu.sh"
    )
    
    for script in "${scripts[@]}"; do
        chmod +x "$script"
        success "Made executable: $(basename "$script")"
    done
}

# Function to create necessary directories
create_directories() {
    info "Creating necessary directories..."
    
    local dirs=(
        "$BEV_ROOT/logs"
        "$BEV_ROOT/data"
        "$BEV_ROOT/config"
        "/var/log/bev-avatar"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            if [[ "$dir" == "/var/log/bev-avatar" ]]; then
                sudo mkdir -p "$dir"
                sudo chown starlord:starlord "$dir"
            else
                mkdir -p "$dir"
            fi
            success "Created directory: $dir"
        fi
    done
}

# Function to install systemd service
install_service() {
    info "Installing systemd service..."
    
    # Copy service file to systemd directory
    sudo cp "$SERVICE_FILE" "$SYSTEMD_DIR/"
    success "Service file copied to $SYSTEMD_DIR"
    
    # Set correct permissions
    sudo chmod 644 "$SYSTEMD_DIR/$SERVICE_NAME.service"
    
    # Reload systemd
    sudo systemctl daemon-reload
    success "Systemd daemon reloaded"
    
    # Enable service
    sudo systemctl enable "$SERVICE_NAME.service"
    success "Service enabled for auto-start"
}

# Function to configure log rotation
setup_log_rotation() {
    info "Setting up log rotation..."
    
    local logrotate_config="/etc/logrotate.d/bev-avatar"
    
    sudo tee "$logrotate_config" > /dev/null << 'EOF'
/home/starlord/Projects/Bev/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0644 starlord starlord
    postrotate
        systemctl reload bev-advanced-avatar.service || true
    endscript
}

/var/log/bev-avatar/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 0644 starlord starlord
}
EOF
    
    success "Log rotation configured"
}

# Function to configure nvidia-persistenced
setup_nvidia_persistence() {
    info "Setting up NVIDIA persistence daemon..."
    
    # Enable nvidia-persistenced service
    if sudo systemctl is-enabled nvidia-persistenced &>/dev/null; then
        success "nvidia-persistenced already enabled"
    else
        sudo systemctl enable nvidia-persistenced
        success "nvidia-persistenced enabled"
    fi
    
    # Start nvidia-persistenced if not running
    if ! sudo systemctl is-active --quiet nvidia-persistenced; then
        sudo systemctl start nvidia-persistenced
        success "nvidia-persistenced started"
    fi
}

# Function to configure udev rules for GPU
setup_gpu_udev_rules() {
    info "Setting up GPU udev rules..."
    
    local udev_rule="/etc/udev/rules.d/99-bev-avatar-gpu.rules"
    
    sudo tee "$udev_rule" > /dev/null << 'EOF'
# BEV Avatar GPU Rules
# Set GPU performance mode and permissions

# NVIDIA GPU device permissions
SUBSYSTEM=="nvidia", OWNER="starlord", GROUP="starlord", MODE="0664"

# Set GPU to performance mode on boot
ACTION=="add", SUBSYSTEM=="pci", ATTRS{vendor}=="0x10de", ATTRS{class}=="0x030000", \
    RUN+="/usr/bin/nvidia-smi -pm 1", \
    RUN+="/usr/bin/nvidia-smi -ac 10501,2230"
EOF
    
    sudo udevadm control --reload-rules
    success "GPU udev rules configured"
}

# Function to create service management aliases
create_aliases() {
    info "Creating service management aliases..."
    
    local alias_file="$HOME/.bash_aliases"
    
    # Create or update aliases
    cat >> "$alias_file" << 'EOF'

# BEV Avatar Service Management Aliases
alias avatar-start='sudo systemctl start bev-advanced-avatar'
alias avatar-stop='sudo systemctl stop bev-advanced-avatar'
alias avatar-restart='sudo systemctl restart bev-advanced-avatar'
alias avatar-status='systemctl status bev-advanced-avatar'
alias avatar-logs='journalctl -u bev-advanced-avatar -f'
alias avatar-health='curl -s http://localhost:8080/health | jq .'
alias avatar-gpu='nvidia-smi'

# BEV Avatar Log Management
alias avatar-log-errors='grep -i error /home/starlord/Projects/Bev/logs/avatar-*.log'
alias avatar-log-cleanup='sudo logrotate -f /etc/logrotate.d/bev-avatar'

# BEV Avatar Service Control
alias avatar-enable='sudo systemctl enable bev-advanced-avatar'
alias avatar-disable='sudo systemctl disable bev-advanced-avatar'

EOF
    
    success "Service management aliases created in $alias_file"
    info "Source the file or restart your shell to use aliases: source $alias_file"
}

# Function to validate installation
validate_installation() {
    info "Validating installation..."
    
    # Check if service is installed
    if ! systemctl list-unit-files | grep -q "$SERVICE_NAME.service"; then
        error "Service not found in systemd"
        exit 1
    fi
    
    # Check if service is enabled
    if ! systemctl is-enabled --quiet "$SERVICE_NAME.service"; then
        error "Service is not enabled"
        exit 1
    fi
    
    # Run pre-start validation
    if "$SCRIPT_DIR/scripts/pre-start-validation.sh"; then
        success "Pre-start validation passed"
    else
        error "Pre-start validation failed"
        exit 1
    fi
    
    # Run GPU check
    if "$SCRIPT_DIR/scripts/gpu-check.sh"; then
        success "GPU check passed"
    else
        error "GPU check failed"
        exit 1
    fi
    
    # Run dependencies check
    if "$SCRIPT_DIR/scripts/dependencies-check.sh"; then
        success "Dependencies check passed"
    else
        error "Dependencies check failed"
        exit 1
    fi
    
    success "Installation validation completed"
}

# Function to create uninstall script
create_uninstall_script() {
    info "Creating uninstall script..."
    
    local uninstall_script="$SCRIPT_DIR/uninstall-avatar-service.sh"
    
    cat > "$uninstall_script" << 'EOF'
#!/bin/bash
#
# BEV Advanced Avatar Service Uninstall Script
#

set -euo pipefail

SERVICE_NAME="bev-advanced-avatar"
SYSTEMD_DIR="/etc/systemd/system"

echo "Uninstalling BEV Advanced Avatar Service..."

# Stop and disable service
sudo systemctl stop "$SERVICE_NAME.service" 2>/dev/null || true
sudo systemctl disable "$SERVICE_NAME.service" 2>/dev/null || true

# Remove service file
sudo rm -f "$SYSTEMD_DIR/$SERVICE_NAME.service"

# Reload systemd
sudo systemctl daemon-reload

# Remove log rotation
sudo rm -f "/etc/logrotate.d/bev-avatar"

# Remove udev rules
sudo rm -f "/etc/udev/rules.d/99-bev-avatar-gpu.rules"
sudo udevadm control --reload-rules

# Remove log directory
sudo rm -rf "/var/log/bev-avatar"

echo "BEV Advanced Avatar Service uninstalled successfully"
echo "Note: BEV project files and logs in /home/starlord/Projects/Bev remain intact"
EOF
    
    chmod +x "$uninstall_script"
    success "Uninstall script created: $uninstall_script"
}

# Function to display installation summary
display_summary() {
    info "=== BEV Advanced Avatar Service Installation Summary ==="
    echo ""
    success "Service Name: $SERVICE_NAME"
    success "Service File: $SYSTEMD_DIR/$SERVICE_NAME.service"
    success "Service Status: $(systemctl is-enabled $SERVICE_NAME.service) / $(systemctl is-active $SERVICE_NAME.service 2>/dev/null || echo 'inactive')"
    echo ""
    info "Service Management Commands:"
    echo "  Start:    sudo systemctl start $SERVICE_NAME"
    echo "  Stop:     sudo systemctl stop $SERVICE_NAME"
    echo "  Restart:  sudo systemctl restart $SERVICE_NAME"
    echo "  Status:   systemctl status $SERVICE_NAME"
    echo "  Logs:     journalctl -u $SERVICE_NAME -f"
    echo ""
    info "Avatar Endpoints (when running):"
    echo "  Health:   http://localhost:8080/health"
    echo "  WebSocket: ws://localhost:8080/ws"
    echo ""
    info "Aliases available (source ~/.bash_aliases):"
    echo "  avatar-start, avatar-stop, avatar-restart, avatar-status"
    echo "  avatar-logs, avatar-health, avatar-gpu"
    echo ""
    info "Log Files:"
    echo "  Service logs: /home/starlord/Projects/Bev/logs/"
    echo "  System logs: journalctl -u $SERVICE_NAME"
    echo ""
    info "To start the service now:"
    echo "  sudo systemctl start $SERVICE_NAME"
    echo ""
}

# Main installation function
main() {
    local start_time=$SECONDS
    
    echo "================================================"
    echo "BEV Advanced Avatar Service Installation"
    echo "================================================"
    echo ""
    
    # Validation steps
    check_root
    validate_requirements
    validate_gpu
    
    # Installation steps
    setup_script_permissions
    create_directories
    install_service
    setup_log_rotation
    setup_nvidia_persistence
    setup_gpu_udev_rules
    create_aliases
    create_uninstall_script
    
    # Validation
    validate_installation
    
    local install_duration=$((SECONDS - start_time))
    
    echo ""
    success "Installation completed successfully in ${install_duration}s"
    echo ""
    
    display_summary
}

# Handle signals during installation
trap 'error "Installation interrupted by signal"; exit 1' SIGTERM SIGINT

# Execute main installation
main "$@"

exit 0