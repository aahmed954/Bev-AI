# AI Companion Standalone Deployment

Complete isolation strategy for AI companion deployment on STARLORD with auto-start/stop capability and optional core platform integration.

## 🎯 Architecture Overview

The standalone AI companion runs completely independently from the core OSINT platform with these key features:

- **Complete Isolation**: Self-contained ecosystem with dedicated databases and services
- **Auto-Start/Stop**: Systemd service management with boot integration
- **Optional Integration**: Detects and connects to core platform when available
- **GPU Optimization**: Exclusive RTX 4090 access with performance tuning
- **Resource Management**: Intelligent cleanup and resource allocation

## 🚀 Quick Start

### Installation

```bash
# Install as systemd service (requires sudo)
sudo ./install-companion-service.sh

# Deploy the companion
companion install

# Enable auto-start on boot
companion enable

# Check status
companion status
```

### Manual Deployment

```bash
# Basic deployment
./deploy-companion.sh deploy

# Force rebuild with integration
./deploy-companion.sh deploy -f -i

# Check health
./deploy-companion.sh health
```

## 📋 Service Management

### Systemd Commands

```bash
# Service control
companion start    # Start companion
companion stop     # Stop companion
companion restart  # Restart companion
companion status   # Show status

# Auto-start control
companion enable   # Enable auto-start
companion disable  # Disable auto-start

# Monitoring
companion logs     # View logs
companion health   # Health check
```

### Manual Control

```bash
# Docker Compose control
docker-compose -f docker-compose.companion.yml up -d    # Start
docker-compose -f docker-compose.companion.yml down     # Stop
docker-compose -f docker-compose.companion.yml restart  # Restart

# Individual service control
docker-compose -f docker-compose.companion.yml restart companion_core
```

## 🌐 Service Endpoints

| Service | Port | Description |
|---------|------|-------------|
| **Companion Web UI** | 18080 | Main companion interface |
| **Core API** | 18000 | Companion API endpoints |
| **WebSocket** | 18001 | Real-time communication |
| **Voice Service** | 18002 | Voice synthesis API |
| **Avatar Service** | 18003 | Live2D avatar rendering |
| **Memory Service** | 18004 | Context and memory API |
| **Integration Gateway** | 18005 | Core platform bridge |
| **Resource Monitor** | 18006 | System monitoring |
| **Grafana Dashboard** | 19000 | Monitoring dashboard |
| **Prometheus Metrics** | 19090 | Metrics collection |

### Database Ports

| Database | Port | Credentials |
|----------|------|-------------|
| **PostgreSQL** | 15432 | `companion_user` / `companion_secure_pass_2024` |
| **Redis** | 16379 | No authentication (isolated) |

## 🏗️ Architecture Components

### Core Services

- **companion-core**: Main AI engine with personality and memory
- **companion-voice**: TorToise-TTS voice synthesis (GPU accelerated)
- **companion-avatar**: Live2D avatar rendering (GPU accelerated)
- **companion-frontend**: React-based web interface
- **companion-memory**: Context and long-term memory management

### Infrastructure Services

- **companion-postgres**: Isolated PostgreSQL database
- **companion-redis**: Isolated Redis cache and sessions
- **companion-gateway**: Optional core platform integration
- **companion-monitor**: Resource monitoring and auto-scaling
- **companion-prometheus**: Metrics collection
- **companion-grafana**: Monitoring dashboards

## 💾 Data Storage

### Persistent Volumes

```
/opt/companion/
├── data/
│   ├── postgres/     # Database storage
│   └── redis/        # Redis persistence
├── models/           # AI model storage
├── assets/           # Avatar and voice assets
├── logs/             # Service logs
└── config/           # Configuration files
```

### Backup and Recovery

```bash
# Backup companion data
./scripts/backup-companion.sh

# Restore from backup
./scripts/restore-companion.sh backup-20241121.tar.gz

# Export configuration
companion export-config > companion-config.yml
```

## 🔧 Configuration

### Environment Variables

```bash
# Core Configuration
COMPANION_MODE=standalone                    # standalone|integrated
COMPANION_NAME="AI Companion"               # Display name
GPU_MEMORY_FRACTION=0.8                     # GPU memory allocation

# Integration Settings
CORE_PLATFORM_ENABLED=false                 # Enable integration detection
THANOS_HOST=172.21.0.10                     # THANOS service host
ORACLE1_HOST=172.21.0.20                    # ORACLE1 service host

# Performance Tuning
MAX_CONCURRENT_SESSIONS=10                   # Concurrent user limit
MEMORY_LIMIT_GB=8                           # System memory limit
VOICE_SYNTHESIS_ENABLED=true                # Enable voice
AVATAR_RENDERING_ENABLED=true               # Enable avatar
```

### Feature Flags

```bash
# Enable specific features
EMOTIONAL_INTELLIGENCE_ENABLED=true
AUTONOMOUS_RESEARCH_ENABLED=true
REAL_TIME_RESPONSE=true
PHYSICS_SIMULATION=true
FACIAL_TRACKING=true
```

## 🔄 Integration Modes

### Standalone Mode (Default)

- Complete independence from core platform
- Self-contained functionality
- Basic OSINT capabilities
- No external dependencies

### Integrated Mode

```bash
# Enable integration detection
./deploy-companion.sh deploy -i

# Or set environment variables
export CORE_PLATFORM_ENABLED=true
export COMPANION_GATEWAY_MODE=integrated
```

**Integration Features:**
- Auto-detection of THANOS/ORACLE1 services
- OSINT query routing to core platform
- Graceful fallback to standalone mode
- Health monitoring of core services

## 📊 Monitoring and Metrics

### Grafana Dashboard

Access at `http://localhost:19000` (admin/companion_admin_2024)

**Key Metrics:**
- GPU utilization and temperature
- Memory usage (system and GPU)
- Service response times
- Active companion sessions
- Integration status

### Prometheus Metrics

Access at `http://localhost:19090`

**Custom Metrics:**
```
companion_gpu_temperature_celsius
companion_memory_usage_bytes
companion_response_time_seconds
companion_active_sessions
companion_core_platform_available
```

### Log Management

```bash
# View all logs
companion logs

# View specific service logs
docker-compose -f docker-compose.companion.yml logs companion_core

# Log locations
/var/log/companion/           # System logs
/opt/companion/logs/          # Service logs
```

## 🔒 Security Features

### Isolation Security

- Dedicated network namespace (172.30.0.0/16)
- No external authentication (single-user design)
- Encrypted inter-service communication
- Resource access controls

### Data Protection

- Personal data encryption at rest
- Secure memory management
- Privacy-focused logging
- Data retention policies

## ⚡ Performance Optimization

### GPU Optimization

- Exclusive RTX 4090 access
- CUDA memory optimization
- Dynamic GPU scaling
- Temperature monitoring

### System Tuning

```bash
# Automatic optimizations applied:
# - CPU governor set to performance
# - Swappiness reduced to 10
# - GPU clocks maximized
# - Memory buffers optimized
```

### Resource Limits

| Resource | Limit | Purpose |
|----------|-------|---------|
| **System Memory** | 8GB | Prevent system impact |
| **GPU Memory** | 80% | Reserve for critical ops |
| **Concurrent Sessions** | 10 | Performance balance |
| **Log Size** | 100MB/file | Disk space management |

## 🧪 Testing and Validation

### Health Checks

```bash
# Comprehensive health check
./scripts/health-check.sh

# Quick status check
companion health

# Performance validation
./scripts/performance-test.sh
```

### Service Testing

```bash
# Test API endpoints
curl http://localhost:18000/health
curl http://localhost:18002/status  # Voice service
curl http://localhost:18003/status  # Avatar service

# Test WebSocket connection
wscat -c ws://localhost:18001/ws
```

## 🚨 Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check Docker daemon
sudo systemctl status docker

# Check GPU access
nvidia-smi

# Verify prerequisites
./scripts/pre-start-checks.sh
```

**Poor performance:**
```bash
# Check GPU temperature
nvidia-smi

# Monitor resource usage
docker stats

# Check system optimization
./scripts/check-optimization.sh
```

**Integration failures:**
```bash
# Check core platform status
companion status

# Test integration manually
curl http://localhost:18005/integration/status

# Force detection
curl -X POST http://localhost:18005/integration/detect
```

### Recovery Procedures

```bash
# Full reset
companion stop
./deploy-companion.sh cleanup -c
companion start

# GPU recovery
sudo nvidia-smi --gpu-reset -i 0
companion restart

# Database recovery
docker-compose exec companion_postgres pg_dump companion > backup.sql
```

## 📚 Development

### Adding New Services

1. Create service directory in `companion-standalone/`
2. Add Dockerfile and requirements
3. Update `docker-compose.companion.yml`
4. Add health check endpoint
5. Update monitoring configuration

### Custom Configuration

```bash
# Override defaults
cp docker-compose.companion.yml docker-compose.override.yml
# Edit docker-compose.override.yml

# Custom environment
cp .env.example .env
# Edit .env with your settings
```

## 🔄 Backup and Recovery

### Automated Backups

```bash
# Enable daily backups
sudo cp scripts/backup-companion.sh /etc/cron.daily/

# Manual backup
./scripts/backup-companion.sh
```

### Disaster Recovery

```bash
# Complete reinstallation
companion uninstall
sudo ./install-companion-service.sh
companion install
```

## 📋 Maintenance

### Regular Tasks

```bash
# Weekly maintenance
./scripts/maintenance-weekly.sh

# Monthly cleanup
./scripts/cleanup-old-logs.sh

# Update companion
git pull origin main
./deploy-companion.sh deploy -f
```

### Resource Monitoring

```bash
# Check disk usage
df -h /opt/companion

# Monitor GPU health
watch nvidia-smi

# Service health
watch "companion status"
```

## 🎯 Future Enhancements

- [ ] Multi-GPU support
- [ ] Distributed deployment
- [ ] Advanced analytics dashboard
- [ ] Plugin architecture
- [ ] Mobile companion app
- [ ] Cloud integration options

---

**Contact:** STARLORD DevOps Team
**Documentation Version:** 1.0.0
**Last Updated:** November 2024