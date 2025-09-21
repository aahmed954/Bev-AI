#!/bin/bash
# Install AI Companion as a systemd service
# Enables auto-start/stop capability for STARLORD

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="companion"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

log() {
    echo -e "${1}"
}

error_exit() {
    log "${RED}ERROR: ${1}${NC}"
    exit 1
}

success() {
    log "${GREEN}✓ ${1}${NC}"
}

warning() {
    log "${YELLOW}⚠ ${1}${NC}"
}

info() {
    log "${BLUE}ℹ ${1}${NC}"
}

header() {
    log "${PURPLE}=== ${1} ===${NC}"
}

# Check if running as root or with sudo
if [[ $EUID -ne 0 ]]; then
    error_exit "This script must be run as root or with sudo"
fi

header "AI COMPANION SERVICE INSTALLATION"

# Check prerequisites
info "Checking prerequisites..."

# Check if Docker is installed
if ! command -v docker >/dev/null 2>&1; then
    error_exit "Docker is not installed"
fi

# Check if Docker Compose is installed
if ! command -v docker-compose >/dev/null 2>&1; then
    error_exit "Docker Compose is not installed"
fi

# Check if user starlord exists
if ! id "starlord" >/dev/null 2>&1; then
    error_exit "User 'starlord' does not exist"
fi

# Check if starlord is in docker group
if ! groups starlord | grep -q docker; then
    error_exit "User 'starlord' is not in the docker group"
fi

success "Prerequisites check passed"

# Create required directories
info "Creating system directories..."

mkdir -p /var/log/companion
mkdir -p /opt/companion/{data/{postgres,redis},models,assets,logs,config,scripts}

# Set ownership
chown -R starlord:starlord /var/log/companion
chown -R starlord:starlord /opt/companion

# Copy scripts to system location
info "Installing companion scripts..."

cp "$SCRIPT_DIR/scripts/"*.sh /opt/companion/scripts/
chmod +x /opt/companion/scripts/*.sh
chown starlord:starlord /opt/companion/scripts/*.sh

success "Scripts installed"

# Install systemd service
info "Installing systemd service..."

cp "$SCRIPT_DIR/companion.service" "$SERVICE_FILE"

# Update paths in service file
sed -i "s|/home/starlord/Projects/Bev/companion-standalone|$SCRIPT_DIR|g" "$SERVICE_FILE"

# Reload systemd
systemctl daemon-reload

success "Systemd service installed"

# Create environment file
info "Creating environment configuration..."

cat > /opt/companion/config/service.env << 'EOF'
# AI Companion Service Environment
# Auto-generated during installation

# Service Configuration
COMPANION_MODE=standalone
COMPANION_AUTO_START=false
COMPANION_AUTO_RESTART=true

# Resource Limits
COMPANION_MEMORY_LIMIT=8G
COMPANION_GPU_MEMORY_FRACTION=0.8
COMPANION_MAX_SESSIONS=10

# Integration Settings
CORE_PLATFORM_ENABLED=false
CORE_PLATFORM_AUTO_DETECT=true
INTEGRATION_TIMEOUT=5

# Security Settings
COMPANION_SECRET_KEY=companion_ultra_secure_key_2024
COMPANION_ENCRYPTION_KEY=companion_encryption_key_2024

# Logging
COMPANION_LOG_LEVEL=INFO
COMPANION_LOG_RETENTION_DAYS=30

# Performance
COMPANION_PERFORMANCE_MODE=optimized
COMPANION_GPU_OPTIMIZATION=true
EOF

chown starlord:starlord /opt/companion/config/service.env

success "Environment configuration created"

# Create management commands
info "Creating management commands..."

# Companion control command
cat > /usr/local/bin/companion << 'EOF'
#!/bin/bash
# AI Companion Service Management Command

set -euo pipefail

COMMAND="${1:-status}"
SERVICE_NAME="companion"

case "$COMMAND" in
    start)
        echo "Starting AI Companion service..."
        systemctl start $SERVICE_NAME
        echo "AI Companion started"
        ;;
    stop)
        echo "Stopping AI Companion service..."
        systemctl stop $SERVICE_NAME
        echo "AI Companion stopped"
        ;;
    restart)
        echo "Restarting AI Companion service..."
        systemctl restart $SERVICE_NAME
        echo "AI Companion restarted"
        ;;
    status)
        systemctl status $SERVICE_NAME --no-pager
        ;;
    enable)
        echo "Enabling AI Companion auto-start..."
        systemctl enable $SERVICE_NAME
        echo "AI Companion will start automatically on boot"
        ;;
    disable)
        echo "Disabling AI Companion auto-start..."
        systemctl disable $SERVICE_NAME
        echo "AI Companion auto-start disabled"
        ;;
    logs)
        journalctl -u $SERVICE_NAME -f
        ;;
    health)
        if systemctl is-active --quiet $SERVICE_NAME; then
            echo "Service is running, checking health..."
            curl -sf http://localhost:18000/health >/dev/null 2>&1 && echo "✓ Healthy" || echo "✗ Unhealthy"
        else
            echo "✗ Service is not running"
        fi
        ;;
    install)
        echo "Installing AI Companion..."
        /opt/companion/scripts/deploy-companion.sh deploy
        ;;
    uninstall)
        echo "Uninstalling AI Companion..."
        /opt/companion/scripts/deploy-companion.sh uninstall
        ;;
    *)
        echo "Usage: companion {start|stop|restart|status|enable|disable|logs|health|install|uninstall}"
        echo ""
        echo "Commands:"
        echo "  start      Start the AI companion service"
        echo "  stop       Stop the AI companion service"
        echo "  restart    Restart the AI companion service"
        echo "  status     Show service status"
        echo "  enable     Enable auto-start on boot"
        echo "  disable    Disable auto-start on boot"
        echo "  logs       Show service logs"
        echo "  health     Check service health"
        echo "  install    Install/deploy companion"
        echo "  uninstall  Completely remove companion"
        exit 1
        ;;
esac
EOF

chmod +x /usr/local/bin/companion

success "Management command installed"

# Create monitoring scripts
info "Creating monitoring scripts..."

cat > /opt/companion/scripts/monitor-companion.sh << 'EOF'
#!/bin/bash
# Companion monitoring script for Prometheus/Grafana

METRICS_FILE="/tmp/companion-metrics.prom"

# Generate Prometheus metrics
cat > "$METRICS_FILE" << METRICS
# HELP companion_service_status Service status (1=running, 0=stopped)
# TYPE companion_service_status gauge
companion_service_status $(systemctl is-active --quiet companion && echo 1 || echo 0)

# HELP companion_containers_running Number of running containers
# TYPE companion_containers_running gauge
companion_containers_running $(docker ps -q --filter "name=companion_" | wc -l)

# HELP companion_gpu_temperature_celsius GPU temperature
# TYPE companion_gpu_temperature_celsius gauge
companion_gpu_temperature_celsius $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)

# HELP companion_gpu_utilization_percent GPU utilization percentage
# TYPE companion_gpu_utilization_percent gauge
companion_gpu_utilization_percent $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo 0)

# HELP companion_memory_usage_bytes Memory usage in bytes
# TYPE companion_memory_usage_bytes gauge
companion_memory_usage_bytes $(free -b | awk '/^Mem:/{print $3}')
METRICS

echo "Metrics written to $METRICS_FILE"
EOF

chmod +x /opt/companion/scripts/monitor-companion.sh

success "Monitoring scripts created"

# Set up log rotation
info "Setting up log rotation..."

cat > /etc/logrotate.d/companion << 'EOF'
/var/log/companion/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 starlord starlord
    postrotate
        systemctl reload companion >/dev/null 2>&1 || true
    endscript
}

/opt/companion/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 starlord starlord
}
EOF

success "Log rotation configured"

# Create desktop shortcut (optional)
info "Creating desktop integration..."

if [[ -d /home/starlord/Desktop ]]; then
    cat > /home/starlord/Desktop/AI-Companion.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=AI Companion
Comment=Standalone AI Companion Service
Exec=xdg-open http://localhost:18080
Icon=applications-science
Terminal=false
Categories=Development;Science;
EOF

    chmod +x /home/starlord/Desktop/AI-Companion.desktop
    chown starlord:starlord /home/starlord/Desktop/AI-Companion.desktop

    success "Desktop shortcut created"
fi

# Final setup
info "Completing installation..."

# Set proper SELinux contexts if needed
if command -v semanage >/dev/null 2>&1; then
    setsebool -P container_manage_cgroup on 2>/dev/null || true
fi

# Create completion script for bash
cat > /etc/bash_completion.d/companion << 'EOF'
_companion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="start stop restart status enable disable logs health install uninstall"

    COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
    return 0
}
complete -F _companion companion
EOF

success "Installation completed successfully!"

# Summary
header "INSTALLATION SUMMARY"

log "${GREEN}✓ Systemd service installed: ${SERVICE_FILE}${NC}"
log "${GREEN}✓ Management command: /usr/local/bin/companion${NC}"
log "${GREEN}✓ Scripts location: /opt/companion/scripts/${NC}"
log "${GREEN}✓ Configuration: /opt/companion/config/${NC}"
log "${GREEN}✓ Logs directory: /var/log/companion${NC}"
log "${GREEN}✓ Log rotation configured${NC}"

if [[ -f /home/starlord/Desktop/AI-Companion.desktop ]]; then
    log "${GREEN}✓ Desktop shortcut created${NC}"
fi

log ""
log "${BLUE}Next steps:${NC}"
log "${BLUE}1. Run 'companion install' to deploy the AI companion${NC}"
log "${BLUE}2. Run 'companion enable' to enable auto-start on boot${NC}"
log "${BLUE}3. Run 'companion status' to check service status${NC}"
log "${BLUE}4. Access the companion at http://localhost:18080${NC}"
log ""
log "${BLUE}Management commands:${NC}"
log "${BLUE}  companion start/stop/restart   - Control service${NC}"
log "${BLUE}  companion status/logs/health   - Monitor service${NC}"
log "${BLUE}  companion enable/disable       - Auto-start control${NC}"
log "${BLUE}  companion install/uninstall    - Deployment control${NC}"

exit 0