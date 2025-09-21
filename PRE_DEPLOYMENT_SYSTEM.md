# BEV OSINT Framework - Pre-Deployment Preparation System

## 🛡️ Overview

The BEV Pre-Deployment Preparation System is a comprehensive validation and conflict resolution framework designed to ensure successful deployment of the BEV OSINT Framework. It provides thorough system validation, automated conflict resolution, and rollback capabilities to prevent deployment failures.

## ✨ Key Features

### 🔍 **5-Gate Validation System**
1. **Infrastructure Readiness** - Hardware, software, and network requirements
2. **Service Conflict Detection** - Port conflicts, container conflicts, resource conflicts
3. **Configuration Validation** - Environment variables, credentials, file integrity
4. **Dependency Chain Validation** - Service dependencies and startup order
5. **Resource Allocation** - Memory, CPU, storage, GPU capacity validation

### 🔧 **Automated Conflict Resolution**
- **Safe Process Termination** - Automatically stops non-critical conflicting processes
- **Container Cleanup** - Removes conflicting Docker containers and resources
- **Port Resolution** - Resolves port conflicts where safe to do so
- **Volume & Network Cleanup** - Cleans unused Docker volumes and networks

### 💾 **Comprehensive Backup & Rollback**
- **System State Backup** - Complete system state preservation before deployment
- **Configuration Backup** - All BEV configuration files and environment variables
- **Docker State Backup** - Container configurations and images
- **Automated Rollback Script** - One-command system restoration

### 🚀 **Enhanced Deployment Integration**
- **Unified Deployment Wrapper** - Single entry point for all deployment scenarios
- **Multi-Node Support** - Enhanced multinode deployment with validation
- **Pre/Post Validation** - Comprehensive validation before and after deployment

## 📁 System Architecture

```
pre_deployment_prep.sh              # Main orchestrator script
├── deployment_prep/
│   ├── validation_modules/          # 5 validation gates
│   │   ├── 01_infrastructure_readiness.sh
│   │   ├── 02_conflict_detection.sh
│   │   ├── 03_configuration_validation.sh
│   │   ├── 04_dependency_validation.sh
│   │   └── 05_resource_allocation.sh
│   ├── conflict_resolution/         # Automated conflict fixes
│   │   ├── auto_port_resolution.sh
│   │   └── auto_container_resolution.sh
│   ├── backups/                     # Backup and rollback system
│   │   └── backup_system.sh
│   ├── integration/                 # Enhanced deployment scripts
│   │   └── integrate_with_existing.sh
│   └── validation_config.yml        # Configuration parameters
├── deploy_bev_with_validation.sh    # Unified deployment wrapper
├── deploy_multinode_bev_enhanced.sh # Enhanced multinode deployment
└── validate_bev_deployment_enhanced.sh # Enhanced validation
```

## 🚀 Quick Start

### 1. **Validate System Only (Recommended First Step)**
```bash
./deploy_bev_with_validation.sh --prep-only
```

### 2. **Full Deployment with Validation**
```bash
./deploy_bev_with_validation.sh multinode
```

### 3. **Auto-Fix Conflicts During Validation**
```bash
./deploy_bev_with_validation.sh --auto-fix multinode
```

### 4. **Emergency Rollback**
```bash
./deploy_bev_with_validation.sh --rollback
```

## 📋 Detailed Usage

### Pre-Deployment Preparation Script

```bash
./pre_deployment_prep.sh [OPTIONS]
```

**Options:**
- `--auto-fix` - Automatically resolve safe conflicts
- `--force` - Continue despite validation failures (not recommended)
- `--quiet` - Minimize output (errors only)
- `--help` - Show help message

**Examples:**
```bash
# Standard validation with manual conflict resolution
./pre_deployment_prep.sh

# Automatic conflict resolution where safe
./pre_deployment_prep.sh --auto-fix

# Force deployment with minimal output (emergency use)
./pre_deployment_prep.sh --force --quiet
```

### Unified Deployment Wrapper

```bash
./deploy_bev_with_validation.sh [OPTIONS] [DEPLOYMENT_TYPE]
```

**Deployment Types:**
- `multinode` - Multi-node deployment (default)
- `single` - Single-node deployment
- `development` - Development environment
- `production` - Production deployment

**Options:**
- `--prep-only` - Run validation only, skip deployment
- `--skip-prep` - Skip validation (risky)
- `--auto-fix` - Auto-resolve conflicts
- `--force` - Force deployment despite warnings
- `--backup-only` - Create backup without deployment
- `--rollback [DIR]` - Rollback to previous state

**Examples:**
```bash
# Validate system readiness without deploying
./deploy_bev_with_validation.sh --prep-only

# Standard multi-node deployment with validation
./deploy_bev_with_validation.sh multinode

# Development setup with automatic conflict resolution
./deploy_bev_with_validation.sh --auto-fix development

# Emergency rollback to previous state
./deploy_bev_with_validation.sh --rollback /var/lib/bev/backups/pre-deployment-20241201-143022
```

## 🔍 Validation Gates Details

### Gate 1: Infrastructure Readiness
**Validates:**
- Node connectivity (THANOS, ORACLE1, STARLORD)
- Hardware requirements (16GB+ RAM, 500GB+ storage)
- Software requirements (Docker 20.10+, Docker Compose 2.0+)
- Network connectivity (internet, DNS, Docker Hub)
- GPU requirements (NVIDIA drivers, CUDA support)

**Auto-Fix Capabilities:** None (requires manual intervention)

### Gate 2: Service Conflict Detection
**Validates:**
- Port availability (80, 443, 5432, 6379, 7474, 7687, 9090, 9200, etc.)
- Container name conflicts
- Docker volume conflicts
- Docker network conflicts
- System process conflicts

**Auto-Fix Capabilities:**
- ✅ Stop non-critical containers using conflicting ports
- ✅ Remove unused Docker volumes and networks
- ✅ Terminate safe processes (nginx, apache, http-server)
- ⚠️ Alert on critical processes requiring manual intervention

### Gate 3: Configuration Validation
**Validates:**
- Environment variable completeness (.env file)
- Configuration file syntax (YAML, JSON)
- API key formats and accessibility
- Password strength requirements
- File permissions and security

**Auto-Fix Capabilities:**
- ✅ Generate missing secure passwords
- ✅ Fix file permissions
- ✅ Create missing configuration templates
- ⚠️ Alert on missing API keys requiring manual setup

### Gate 4: Dependency Chain Validation
**Validates:**
- Service startup order and dependencies
- Health check endpoint availability
- Network routing between services
- Security policies and access controls
- Service initialization requirements

**Auto-Fix Capabilities:**
- ✅ Validate dependency chains
- ✅ Detect circular dependencies
- ⚠️ Alert on missing initialization scripts

### Gate 5: Resource Allocation
**Validates:**
- Memory allocation (16GB+ system, per-service requirements)
- CPU allocation (4+ cores, per-service CPU needs)
- Storage allocation (500GB+ available, per-service storage)
- GPU allocation (NVIDIA GPU memory, CUDA compatibility)
- Network bandwidth requirements

**Auto-Fix Capabilities:** None (requires hardware upgrades or config changes)

## 🛡️ Automated Conflict Resolution

### Safe Operations (Auto-Fix Enabled)
- Stop non-critical web servers (nginx, apache, development servers)
- Remove unused Docker containers, volumes, and networks
- Fix file permissions on configuration files
- Generate secure passwords for missing environment variables
- Clean up temporary files and directories

### Manual Intervention Required
- Critical system services (postgres, mysql, redis, elasticsearch)
- System-level network configuration
- Insufficient hardware resources (RAM, disk space, CPU)
- Missing API keys and external service credentials
- Firewall and security policy modifications

## 💾 Backup & Rollback System

### Automatic Backup Creation
Before any deployment, the system automatically creates:

```
/var/lib/bev/backups/pre-deployment-YYYYMMDD-HHMMSS/
├── backup_manifest.json           # Backup metadata
├── docker/                        # Docker state
│   ├── containers_running.txt
│   ├── containers_all.txt
│   ├── images.txt
│   ├── volumes.txt
│   └── networks.txt
├── configs/                       # Configuration files
│   ├── env_backup
│   ├── docker-compose.complete.yml
│   └── intelowl/
├── databases/                     # Database schemas
│   ├── postgres_schema.sql
│   ├── neo4j_constraints.cypher
│   └── redis_config.txt
├── system/                        # System state
│   ├── system_info.txt
│   ├── network_interfaces.txt
│   └── processes.txt
└── rollback.sh                    # Automated rollback script
```

### Rollback Capabilities
```bash
# Automatic rollback to latest backup
./deploy_bev_with_validation.sh --rollback

# Rollback to specific backup
./deploy_bev_with_validation.sh --rollback /var/lib/bev/backups/pre-deployment-20241201-143022

# Manual rollback using backup script
cd /var/lib/bev/backups/pre-deployment-20241201-143022
./rollback.sh --full
```

## 📊 Validation Reports

The system generates comprehensive reports in Markdown format:

```markdown
# BEV OSINT Framework - Deployment Readiness Report

**Generated:** 2024-12-01 14:30:22
**Validation Status:** ✅ READY
**Auto-Fix Mode:** Enabled
**Force Mode:** Disabled

## Validation Gates Summary
| Gate | Status | Details |
|------|--------|---------|
| Infrastructure Readiness | ✅ PASS | All requirements met |
| Service Conflict Detection | ⚠️ WARN | 2 conflicts resolved |
| Configuration Validation | ✅ PASS | All configs valid |
| Dependency Chain Validation | ✅ PASS | Dependencies validated |
| Resource Allocation | ✅ PASS | Sufficient resources |

## System Information
**System Resources:**
- Memory: 32GB total, 28GB available
- Storage: 1TB total, 800GB available
- CPU Cores: 16
- GPU: 1 NVIDIA RTX 4090

## Recommendations
✅ System is ready for deployment
**Next Steps:**
1. Run: `./deploy_bev_with_validation.sh multinode`
2. Monitor deployment progress
3. Validate post-deployment
```

## ⚙️ Configuration

### System Requirements

**Minimum Requirements:**
- **Memory:** 16GB RAM
- **Storage:** 500GB available disk space
- **CPU:** 4 cores
- **Network:** 10 Mbps internet connection
- **OS:** Ubuntu 20.04+ or compatible Linux

**Recommended Requirements:**
- **Memory:** 32GB RAM
- **Storage:** 1TB available disk space
- **CPU:** 8+ cores
- **GPU:** NVIDIA GPU with 8GB+ VRAM
- **Network:** 100+ Mbps internet connection

### Multi-Node Configuration

The system supports multi-node deployment across:

- **THANOS** (192.168.1.100) - GPU services, primary databases, AI/ML
- **ORACLE1** (192.168.1.101) - ARM services, monitoring, lightweight services
- **STARLORD** (192.168.1.102) - Control node, Vault, development, frontend

### Environment Variables

**Required Variables (Auto-generated if missing):**
```bash
POSTGRES_PASSWORD=<auto-generated-secure-password>
NEO4J_PASSWORD=<auto-generated-secure-password>
REDIS_PASSWORD=<auto-generated-secure-password>
ELASTICSEARCH_PASSWORD=<auto-generated-secure-password>
GRAFANA_ADMIN_PASSWORD=<auto-generated-secure-password>
INTELOWL_ADMIN_PASSWORD=<auto-generated-secure-password>
SECRET_KEY=<auto-generated-hex-key>
```

**Optional Variables (OSINT Capabilities):**
```bash
DEHASHED_API_KEY=<your-dehashed-api-key>
SNUSBASE_API_KEY=<your-snusbase-api-key>
ALPHAHQ_API_KEY=<your-alphahq-api-key>
SHODAN_API_KEY=<your-shodan-api-key>
VIRUSTOTAL_API_KEY=<your-virustotal-api-key>
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### "Infrastructure validation failed"
**Symptoms:** Gate 1 fails with hardware/software requirements
**Solutions:**
1. Check system requirements (16GB RAM, 500GB storage, 4 CPU cores)
2. Update Docker to version 20.10+ and Docker Compose to 2.0+
3. Install NVIDIA drivers if GPU acceleration is needed
4. Verify network connectivity and DNS resolution

#### "Port conflicts detected"
**Symptoms:** Gate 2 fails with port conflicts
**Solutions:**
1. Run with `--auto-fix` to automatically resolve safe conflicts
2. Manually stop conflicting services: `sudo systemctl stop nginx`
3. Check for running Docker containers: `docker ps`
4. Use different ports in configuration if conflicts persist

#### "Configuration validation failed"
**Symptoms:** Gate 3 fails with missing environment variables or configs
**Solutions:**
1. Run with `--auto-fix` to generate missing passwords
2. Create `.env` file with required variables
3. Verify configuration file syntax (YAML/JSON)
4. Check file permissions on sensitive files

#### "Resource allocation insufficient"
**Symptoms:** Gate 5 fails with insufficient resources
**Solutions:**
1. Free up disk space: `docker system prune -a`
2. Stop unnecessary services: `systemctl list-units --state=active`
3. Add more RAM or upgrade system hardware
4. Adjust resource allocation in configuration

#### "Backup creation failed"
**Symptoms:** Cannot create system backup
**Solutions:**
1. Check disk space: `df -h`
2. Verify permissions: `sudo chown -R $(whoami) /var/lib/bev`
3. Create backup directory: `sudo mkdir -p /var/lib/bev/backups`

### Debug Mode

For detailed debugging information:

```bash
# Enable debug logging
export DEBUG=1
./pre_deployment_prep.sh --auto-fix

# Check validation module logs
tail -f deployment_prep/logs/prep_*.log

# Verbose validation output
./pre_deployment_prep.sh --auto-fix 2>&1 | tee deployment_debug.log
```

### Manual Validation Steps

If automated validation fails, you can manually verify each component:

```bash
# Check Docker setup
docker --version
docker-compose --version
docker info

# Check system resources
free -h
df -h
nproc

# Check network connectivity
ping -c 3 8.8.8.8
nslookup google.com

# Check GPU (if applicable)
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.0-base-ubuntu20.04 nvidia-smi

# Check port availability
netstat -tuln | grep -E ':(80|443|5432|6379|7474|7687|9090|9200)'
```

## 🔗 Integration with Existing Scripts

The preparation system integrates seamlessly with existing BEV deployment scripts:

### Enhanced Scripts Created
- **deploy_bev_with_validation.sh** - Unified deployment wrapper
- **deploy_multinode_bev_enhanced.sh** - Enhanced multinode deployment
- **validate_bev_deployment_enhanced.sh** - Enhanced validation

### Original Scripts Preserved
All original deployment scripts are backed up and preserved:
- `deployment_prep/integration/script_backups/`

### Migration Path
1. **Start with validation-only:** `./deploy_bev_with_validation.sh --prep-only`
2. **Use enhanced scripts:** `./deploy_bev_with_validation.sh multinode`
3. **Fall back to originals:** Use original scripts if needed (not recommended)

## 📈 Performance and Monitoring

### Validation Performance
- **Standard validation:** ~2-5 minutes
- **With auto-fix:** ~3-8 minutes
- **Full backup creation:** ~5-15 minutes (depending on data size)

### System Monitoring During Validation
The system monitors:
- CPU usage during validation
- Memory usage trends
- Disk I/O performance
- Network connectivity stability
- Docker daemon performance

### Validation Metrics
- **Infrastructure readiness:** Pass/fail for each component
- **Conflict resolution:** Count of conflicts found and resolved
- **Resource utilization:** Current vs. required resource usage
- **Configuration completeness:** Percentage of required configs present

## 🔐 Security Considerations

### Credential Management
- **Auto-generated passwords:** 25+ character secure passwords
- **File permissions:** Restrictive permissions on sensitive files (600/700)
- **Environment isolation:** Separate environment files for different deployments
- **Vault integration:** Supports HashiCorp Vault for credential management

### Network Security
- **Firewall validation:** Checks UFW/iptables configuration
- **Port security:** Validates only required ports are open
- **Private networks:** Designed for private network deployment only
- **No external exposure:** System creates no external network exposure

### Data Protection
- **Backup encryption:** Sensitive data in backups is access-controlled
- **Temporary file cleanup:** All temporary files cleaned after use
- **Log security:** Sensitive information filtered from logs
- **Rollback safety:** Rollback scripts preserve security configurations

## 📚 API Reference

### Main Functions

#### `pre_deployment_prep.sh`
Main orchestrator script with comprehensive validation pipeline.

**Parameters:**
- `--auto-fix`: Enable automatic conflict resolution
- `--force`: Continue despite validation failures
- `--quiet`: Minimize output verbosity

**Return Codes:**
- `0`: All validations passed
- `1`: Validation failures detected

#### Validation Gates
Each validation gate returns detailed status information:

```bash
# Infrastructure readiness
infrastructure_readiness() -> 0|1

# Conflict detection
conflict_detection() -> 0|1

# Configuration validation
configuration_validation() -> 0|1

# Dependency validation
dependency_validation() -> 0|1

# Resource allocation
resource_allocation() -> 0|1
```

### Backup System API

```bash
# Create system backup
create_system_backup() -> backup_directory_path

# Restore from backup
restore_system_backup(backup_directory) -> 0|1

# Validate backup integrity
validate_backup(backup_directory) -> 0|1
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/enhancement`
3. Follow existing code patterns and documentation standards
4. Test thoroughly with different system configurations
5. Submit pull request with detailed description

### Testing Guidelines
- Test on multiple Linux distributions (Ubuntu, CentOS, Debian)
- Test with different hardware configurations
- Test conflict resolution scenarios
- Test backup and rollback functionality
- Validate multi-node deployment scenarios

### Code Style
- Follow existing bash scripting patterns
- Use comprehensive error handling
- Include detailed logging and status reporting
- Maintain backward compatibility where possible
- Document all new functions and parameters

## 📄 License

This pre-deployment preparation system is part of the BEV OSINT Framework and follows the same licensing terms as the main project.

## 🆘 Support

For issues with the pre-deployment preparation system:

1. **Check the troubleshooting section** in this documentation
2. **Review validation logs** in `deployment_prep/logs/`
3. **Run with debug mode** for detailed information
4. **Create an issue** with system information and error logs
5. **Include validation report** generated by the system

### System Information for Support

When reporting issues, include:

```bash
# Generate system information
./pre_deployment_prep.sh --prep-only > system_validation.log 2>&1

# Include system details
uname -a > system_info.txt
docker --version >> system_info.txt
docker-compose --version >> system_info.txt
free -h >> system_info.txt
df -h >> system_info.txt
```

---

**🎯 The BEV Pre-Deployment Preparation System ensures reliable, validated, and safe deployment of the BEV OSINT Framework across all supported configurations and environments.**