# BEV Advanced Avatar Systemd Service

Complete systemd service configuration for the BEV Advanced Avatar system with RTX 4090 optimization and comprehensive monitoring.

## 🚀 Quick Start

### Installation
```bash
cd /home/starlord/Projects/Bev/systemd
./install-avatar-service.sh
```

### Basic Usage
```bash
# Start the service
sudo systemctl start bev-advanced-avatar

# Check status
./avatar-service-manager.sh status

# View logs
./avatar-service-manager.sh logs follow
```

## 📁 Components

### Core Service Files
- **`bev-advanced-avatar.service`** - Main systemd service definition
- **`install-avatar-service.sh`** - Installation and setup script
- **`avatar-service-manager.sh`** - Service management and diagnostics

### Validation Scripts (`scripts/`)
- **`pre-start-validation.sh`** - System readiness validation
- **`gpu-check.sh`** - RTX 4090 optimization and validation
- **`dependencies-check.sh`** - Dependencies and connectivity checks

### Lifecycle Scripts (`scripts/`)
- **`start-avatar.sh`** - Service startup with GPU optimization
- **`stop-avatar.sh`** - Graceful shutdown with state preservation
- **`cleanup-gpu.sh`** - Post-shutdown GPU cleanup

### Monitoring
- **`monitor-avatar-health.sh`** - Comprehensive health monitoring

## ⚙️ Service Configuration

### Environment Variables
```bash
# GPU Optimization
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
GPU_MEMORY_FRACTION=0.6

# BEV Configuration
BEV_AVATAR_CONFIG=/home/starlord/Projects/Bev/config/avatar.yaml
BEV_REDIS_URL=redis://localhost:6379
BEV_MCP_SERVER_URL=http://localhost:3010
BEV_AVATAR_PORT=8080
```

### Resource Limits
- **Memory**: 16GB limit, 24GB max
- **CPU**: 800% quota (8 cores)
- **GPU**: Exclusive RTX 4090 access
- **Files**: 65536 file descriptor limit

### Dependencies
- **After**: docker.service, nvidia-persistenced.service, redis.service
- **Requires**: docker.service
- **Wants**: network-online.target

## 🎛️ Service Management

### Using Service Manager
```bash
# Service control
./avatar-service-manager.sh start
./avatar-service-manager.sh stop
./avatar-service-manager.sh restart
./avatar-service-manager.sh status

# Auto-start configuration
./avatar-service-manager.sh enable
./avatar-service-manager.sh disable

# Monitoring
./avatar-service-manager.sh logs
./avatar-service-manager.sh logs follow
./avatar-service-manager.sh diagnostics
./avatar-service-manager.sh health
./avatar-service-manager.sh health status
```

### Direct systemctl Commands
```bash
sudo systemctl start bev-advanced-avatar
sudo systemctl stop bev-advanced-avatar
sudo systemctl restart bev-advanced-avatar
systemctl status bev-advanced-avatar
journalctl -u bev-advanced-avatar -f
```

### Bash Aliases (after installation)
```bash
avatar-start      # Start service
avatar-stop       # Stop service
avatar-restart    # Restart service
avatar-status     # Show status
avatar-logs       # Follow logs
avatar-health     # Check health endpoint
avatar-gpu        # Show GPU status
```

## 🏥 Health Monitoring

### Health Check Commands
```bash
# Single health check
./monitor-avatar-health.sh once

# Current health status
./monitor-avatar-health.sh status

# Continuous monitoring
./monitor-avatar-health.sh monitor
```

### Health Metrics
- **Service Status**: Active/inactive state and uptime
- **Endpoint Health**: HTTP response time and accessibility
- **GPU Health**: Temperature, utilization, memory usage
- **System Resources**: CPU, memory, disk usage
- **Redis Connectivity**: Connection and data validation

### Alert Levels
- **Critical (🚨)**: Service down, GPU overheating, memory exhausted
- **Warning (⚠️)**: High resource usage, slow response times
- **Info (ℹ️)**: Service events, recovery actions

### Health Data Storage
- **Local Files**: `/home/starlord/Projects/Bev/logs/health/`
- **Redis**: Real-time health data and alerts
- **Logs**: `/home/starlord/Projects/Bev/logs/avatar-health-monitor.log`

## 🔧 GPU Optimization

### RTX 4090 Specific Optimizations
- **Persistence Mode**: Enabled for consistent performance
- **Application Clocks**: Set to maximum (10501 MHz memory, 2230 MHz graphics)
- **Power Limit**: 450W maximum
- **Memory Management**: 60% allocation fraction with 512MB split size
- **CUDA Configuration**: Optimized for real-time rendering

### GPU Health Monitoring
- **Temperature**: Alert if >80°C
- **Memory Usage**: Alert if >90%
- **Utilization**: Tracked for performance analysis
- **Process Monitoring**: Automatic cleanup of stuck processes

## 📋 Pre-Start Validation

### System Checks
- **User Validation**: Must run as 'starlord'
- **Directory Structure**: Required BEV directories exist
- **Configuration Files**: Avatar config validation
- **Disk Space**: Minimum 10GB available
- **Memory**: Minimum 8GB available

### GPU Validation
- **NVIDIA Drivers**: Version compatibility check
- **CUDA Installation**: CUDA 12.x+ validation
- **GPU Memory**: Minimum 12GB free for avatar
- **PyTorch CUDA**: CUDA support verification
- **Performance Mode**: GPU optimization applied

### Dependencies Check
- **Docker**: Service active and accessible
- **Redis**: Connection and operation validation
- **MCP Server**: Optional connectivity check
- **Python Environment**: Required packages verification
- **Network**: Port availability and connectivity

## 📊 Service Integration

### BEV Platform Integration
- **Redis**: Avatar state and OSINT updates
- **MCP Server**: Claude Code proxy integration
- **Docker Network**: Container communication
- **OSINT Pipeline**: Real-time intelligence updates

### Monitoring Integration
- **Prometheus Metrics**: Performance and health metrics
- **Log Aggregation**: Structured logging with rotation
- **Alert System**: Redis-based alert distribution
- **Recovery Actions**: Automated issue resolution

## 🛠️ Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check validation logs
./avatar-service-manager.sh diagnostics

# View detailed startup logs
journalctl -u bev-advanced-avatar -n 100

# Validate dependencies
./systemd/scripts/dependencies-check.sh
```

#### GPU Issues
```bash
# Check GPU status
nvidia-smi

# Validate GPU configuration
./systemd/scripts/gpu-check.sh

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"
```

#### Performance Issues
```bash
# Monitor resource usage
./avatar-service-manager.sh health monitor

# Check system resources
htop
nvidia-smi -l 1

# Review performance logs
grep -i performance /home/starlord/Projects/Bev/logs/avatar-*.log
```

### Recovery Procedures
1. **Service Recovery**: Automatic restart on failure with exponential backoff
2. **GPU Recovery**: Memory cleanup and performance reset
3. **Health Recovery**: Automated issue detection and remediation
4. **State Recovery**: Redis state preservation and restoration

## 📝 Log Management

### Log Locations
- **Service Logs**: `/home/starlord/Projects/Bev/logs/`
- **System Logs**: `journalctl -u bev-advanced-avatar`
- **Health Logs**: `/home/starlord/Projects/Bev/logs/health/`
- **GPU Logs**: `/home/starlord/Projects/Bev/logs/avatar-gpu*.log`

### Log Rotation
- **Daily Rotation**: Automatic log rotation configured
- **Retention**: 30 days for service logs, 7 days for system logs
- **Compression**: Automatic compression of rotated logs
- **Manual Rotation**: `sudo logrotate -f /etc/logrotate.d/bev-avatar`

## 🔒 Security Configuration

### Service Security
- **User Isolation**: Runs as dedicated 'starlord' user
- **Filesystem Protection**: Read-only system, restricted paths
- **Resource Limits**: CPU, memory, and process limits
- **No New Privileges**: Prevents privilege escalation

### Network Security
- **Local Only**: Binds to localhost interfaces only
- **Firewall**: Should be configured to block external access
- **No Authentication**: Single-user deployment (private network only)

### GPU Security
- **Exclusive Access**: Single service GPU allocation
- **Performance Isolation**: Dedicated GPU resource management
- **Memory Protection**: Isolated CUDA memory spaces

## 📦 Installation Details

### System Requirements
- **OS**: Ubuntu 20.04+ or compatible Linux
- **GPU**: NVIDIA RTX 4090 with 525.60.11+ drivers
- **CUDA**: Version 12.x or higher
- **Memory**: 16GB RAM minimum (32GB recommended)
- **Storage**: 500GB SSD with 10GB free minimum
- **User**: 'starlord' with sudo access

### Installation Steps
1. **Validation**: System requirements and dependencies
2. **Script Setup**: Executable permissions and directory structure
3. **Service Installation**: systemd service registration
4. **Configuration**: Log rotation, GPU rules, aliases
5. **Validation**: End-to-end functionality verification

### Post-Installation
- **Auto-start**: Service enabled for boot startup
- **Health Monitoring**: Background health checks
- **Log Rotation**: Automated log management
- **GPU Optimization**: Performance mode enabled
- **Aliases**: Service management shortcuts

## 🔄 Maintenance

### Regular Maintenance
```bash
# Health check
./avatar-service-manager.sh health

# Log cleanup
sudo logrotate -f /etc/logrotate.d/bev-avatar

# GPU maintenance
nvidia-smi --reset-applications-clocks

# Service restart (weekly)
./avatar-service-manager.sh restart
```

### Updates and Upgrades
```bash
# Before system updates
./avatar-service-manager.sh stop

# After driver updates
sudo systemctl restart nvidia-persistenced
./avatar-service-manager.sh restart

# Service reconfiguration
./install-avatar-service.sh  # Safe to re-run
```

## 📞 Support

### Getting Help
- **Diagnostics**: `./avatar-service-manager.sh diagnostics`
- **Health Status**: `./avatar-service-manager.sh health status`
- **Logs**: `./avatar-service-manager.sh logs follow`
- **Documentation**: This README and `/home/starlord/Projects/Bev/CLAUDE.md`

### Emergency Recovery
```bash
# Force stop all avatar processes
sudo pkill -f "avatar"

# Reset GPU state
sudo nvidia-smi --reset-gpu

# Clean restart
./avatar-service-manager.sh stop
./systemd/scripts/cleanup-gpu.sh
./avatar-service-manager.sh start
```

---

**Note**: This service is optimized for the STARLORD development environment with RTX 4090. Ensure all prerequisites are met before installation and operation.