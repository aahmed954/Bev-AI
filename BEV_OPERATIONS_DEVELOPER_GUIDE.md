# BEV Operations & Developer Guide

**Version**: 2.0 Enterprise
**Target Audience**: Developers, Operators, and System Administrators
**Platform**: BEV AI Assistant Platform for Cybersecurity Research

---

## 🎯 Quick Reference Guide

### Essential Commands for Daily Operations

**System Status (Run First Every Day)**:
```bash
# Multi-node health check
./scripts/health_check_all_nodes.sh

# Avatar system status (STARLORD)
systemctl status bev-advanced-avatar && curl http://localhost:8091/health

# AI services status (THANOS)
curl http://localhost:8081/health && curl http://localhost:3010/health

# Monitoring status (ORACLE1)
curl http://oracle1:9090/-/healthy && curl http://oracle1:3000/api/health
```

**Emergency Commands**:
```bash
# Emergency shutdown
./scripts/emergency_shutdown_all_nodes.sh

# Emergency backup
./scripts/emergency_backup_all_data.sh

# Security isolation
./scripts/emergency_isolation.sh
```

**Development Workflow**:
```bash
# Start development environment
docker-compose -f docker-compose-development.yml up -d

# Desktop application development
cd bev-frontend && npm run tauri dev

# Avatar system development
cd src/avatar && python3 test_avatar_system.py

# Code quality checks
python -m black . && python -m flake8 src/ tests/ && python -m mypy src/
```

---

## 🛠️ Development Environment Setup

### Prerequisites and Dependencies

**System Requirements**:
```yaml
Operating_System: Ubuntu 20.04+ or compatible Linux
Docker: 20.10+ with Docker Compose v2
Node.js: 18+ with npm 8+
Python: 3.11+ with pip and virtual environments
Rust: 1.70+ with Cargo for Tauri development
CUDA: 12.0+ for GPU features (NVIDIA drivers 525+)

Hardware_Minimum:
  RAM: 16GB (32GB+ recommended)
  Storage: 500GB SSD (1TB+ recommended)
  GPU: NVIDIA RTX 3080+ for AI features
  Network: Gigabit internet for model downloads
```

**Environment Setup Script**:
```bash
#!/bin/bash
# setup_development_environment.sh

echo "🔧 Setting up BEV Development Environment"

# Install system dependencies
sudo apt update && sudo apt install -y \
  docker.io docker-compose \
  nodejs npm \
  python3 python3-pip python3-venv \
  curl wget git \
  nvidia-docker2

# Install Rust for Tauri development
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Setup Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Tauri CLI
cargo install tauri-cli

# Setup frontend dependencies
cd bev-frontend
npm install
cd ..

# Generate development configuration
cp .env.example .env
./generate-secure-credentials.sh

echo "✅ Development environment setup complete"
echo "📋 Next steps:"
echo "   1. Update .env with your API keys"
echo "   2. Run './validate_development_setup.sh'"
echo "   3. Start with 'docker-compose -f docker-compose-development.yml up -d'"
```

### Development Workflow

**Daily Development Process**:
```bash
# 1. Start development services
docker-compose -f docker-compose-development.yml up -d

# 2. Verify services are running
./scripts/validate_development_setup.sh

# 3. Start desktop application development
cd bev-frontend
npm run tauri dev

# 4. Start avatar system (if STARLORD)
cd src/avatar
python3 test_avatar_system.py

# 5. Run tests before commits
./run_all_tests.sh --quick
```

**Code Quality Pipeline**:
```bash
# Pre-commit quality checks (required)
python -m black .                    # Code formatting
python -m flake8 src/ tests/         # Linting
python -m mypy src/                  # Type checking
pytest tests/ --cov=src             # Unit testing

# Security validation
cd bev-frontend && ./validate-security.sh

# Integration testing
pytest tests/integration/ -v

# Performance testing
pytest tests/performance/ --quick
```

---

## 🤖 AI Assistant Development

### Avatar System Development

**Avatar Development Workflow** (STARLORD only):
```bash
# Setup avatar development environment
cd src/avatar
python3 -m pip install -r requirements-avatar.txt

# Run avatar system tests
python3 test_avatar_system.py        # Full test suite
python3 test_avatar_system.py quick  # Quick validation

# RTX 4090 optimization
python3 rtx4090_optimizer.py

# Start avatar service for development
sudo systemctl start bev-advanced-avatar

# Monitor avatar performance
curl http://localhost:8091/metrics
```

**Avatar Service Development Pattern**:
```python
# Template for avatar-integrated services
from src.avatar.advanced_avatar_controller import AdvancedAvatarController

class MyOSINTService:
    def __init__(self):
        self.avatar_controller = AdvancedAvatarController()

    async def process_investigation(self, target: str):
        # Notify avatar of investigation start
        await self.avatar_controller.process_osint_update({
            'type': 'investigation_started',
            'target': target,
            'emotion_context': 'focused_professional'
        })

        # Perform investigation logic
        results = await self._investigate(target)

        # Update avatar with results
        await self.avatar_controller.process_osint_update({
            'type': 'investigation_completed',
            'findings': results,
            'emotion_context': 'satisfied_professional'
        })

        return results
```

### Extended Reasoning Development

**Extended Reasoning Service Development**:
```python
# Template for extended reasoning integration
from src.agents.extended_reasoning_service import ExtendedReasoningService

class ComplexAnalysisService:
    def __init__(self):
        self.reasoning_service = ExtendedReasoningService()

    async def analyze_complex_threat(self, threat_data: Dict) -> Dict[str, Any]:
        # Use extended reasoning for complex analysis
        analysis_request = {
            'context': threat_data,
            'analysis_type': 'threat_attribution',
            'max_tokens': 50000,
            'depth': 'comprehensive'
        }

        reasoning_result = await self.reasoning_service.process_extended_analysis(
            analysis_request
        )

        return reasoning_result
```

### MCP Server Development

**OSINT Tool Development Pattern**:
```python
# Template for new OSINT tools
from src.mcp_server.tools import OSINTToolBase
from typing import Dict, Any

class NewOSINTTool(OSINTToolBase):
    def __init__(self):
        super().__init__("new_osint_tool")
        self.description = "Description of what this tool does"
        self.parameters = {
            "target": {"type": "string", "description": "Investigation target"},
            "depth": {"type": "string", "description": "Analysis depth", "default": "standard"}
        }

    async def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the OSINT tool with security validation
        """
        # Security validation
        if not await self.validate_request(parameters):
            raise SecurityError("Request validation failed")

        # Rate limiting
        await self.rate_limiter.acquire()

        try:
            # Tool implementation
            target = parameters.get("target")
            depth = parameters.get("depth", "standard")

            # Perform OSINT analysis
            results = await self._perform_analysis(target, depth)

            # Audit logging
            await self.audit_logger.log_operation(
                tool_name=self.tool_name,
                parameters=parameters,
                results_summary=self._summarize_results(results)
            )

            return results

        finally:
            await self.rate_limiter.release()

    async def _perform_analysis(self, target: str, depth: str) -> Dict[str, Any]:
        """
        Implement the actual OSINT analysis logic here
        """
        # Implementation specific to this tool
        pass
```

---

## 🔧 Operations Management

### Service Management

**Multi-Node Service Control**:
```bash
# Start all services
./scripts/start_all_services.sh

# Stop all services
./scripts/stop_all_services.sh

# Restart services by node
docker-compose -f docker-compose-starlord-dev.yml restart
docker-compose -f docker-compose-thanos-unified.yml restart
docker-compose -f docker-compose-oracle1-unified.yml restart

# Service status by node
./scripts/status_starlord.sh
./scripts/status_thanos.sh
./scripts/status_oracle1.sh
```

**Avatar System Management** (STARLORD):
```bash
# Avatar service management
sudo systemctl start bev-advanced-avatar
sudo systemctl stop bev-advanced-avatar
sudo systemctl restart bev-advanced-avatar
sudo systemctl status bev-advanced-avatar

# Avatar configuration reload
sudo systemctl reload bev-advanced-avatar

# Avatar logs
journalctl -u bev-advanced-avatar -f
journalctl -u bev-advanced-avatar --since "1 hour ago"
```

**Database Management** (THANOS):
```bash
# PostgreSQL management
docker exec -it bev_postgres psql -U researcher -d osint

# Neo4j management
docker exec -it bev_neo4j cypher-shell -u neo4j -p BevGraphMaster2024

# Redis management
docker exec -it bev_redis redis-cli

# Elasticsearch management
curl http://localhost:9200/_cluster/health
```

### Monitoring and Alerting

**Health Monitoring Commands**:
```bash
# System health dashboard
curl http://oracle1:3000/d/bev-overview

# Prometheus metrics
curl http://oracle1:9090/api/v1/query?query=up

# Avatar system metrics
curl http://localhost:8091/metrics

# AI services metrics
curl http://localhost:8081/metrics
curl http://localhost:3010/metrics

# OSINT service metrics
curl http://localhost:8082/metrics
```

**Alert Management**:
```bash
# View active alerts
curl http://oracle1:9090/api/v1/alerts

# Silence alerts
curl -X POST http://oracle1:9090/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '{"matchers": [{"name": "alertname", "value": "AvatarSystemDown"}]}'

# Test alert rules
./scripts/test_alert_rules.sh
```

### Backup and Recovery

**Automated Backup Procedures**:
```bash
# Daily backup routine
./scripts/daily_backup.sh

# Component-specific backups
./scripts/backup_avatar_state.sh      # Avatar system state
./scripts/backup_databases.sh         # All databases
./scripts/backup_configurations.sh   # System configurations
./scripts/backup_vault_data.sh       # Security credentials

# Backup validation
./scripts/validate_backups.sh
```

**Recovery Procedures**:
```bash
# Complete system recovery
./scripts/disaster_recovery.sh

# Component recovery
./scripts/recover_avatar_system.sh
./scripts/recover_databases.sh
./scripts/recover_configurations.sh

# Partial recovery
./scripts/recover_single_service.sh bev-avatar-controller
```

---

## 🔍 Debugging and Troubleshooting

### Common Issues and Solutions

**Avatar System Issues (STARLORD)**:

*Issue: Avatar not responding*
```bash
# Check service status
systemctl status bev-advanced-avatar

# Check GPU availability
nvidia-smi

# Check logs
journalctl -u bev-advanced-avatar -f

# Restart service
sudo systemctl restart bev-advanced-avatar

# Test avatar API
curl http://localhost:8091/health
```

*Issue: Poor avatar performance*
```bash
# GPU optimization
cd src/avatar && python3 rtx4090_optimizer.py

# Check GPU utilization
nvidia-smi dmon -s pucvmet -d 1

# Monitor avatar metrics
curl http://localhost:8091/metrics | grep fps
```

*Issue: Voice synthesis delays*
```bash
# Check Bark AI installation
pip list | grep bark

# Reinstall voice synthesis
pip install bark-tts --upgrade

# Test voice synthesis
cd src/avatar && python3 test_voice_synthesis.py
```

**AI Services Issues (THANOS)**:

*Issue: Extended reasoning timeouts*
```bash
# Check MCP server status
curl http://localhost:3010/health

# Check Docker logs
docker logs bev_mcp_server

# Check Claude Code proxy configuration
cat config/mcp-proxy.json

# Restart MCP services
docker-compose -f docker-compose-thanos-unified.yml restart bev_mcp_server
```

*Issue: Memory exhaustion during analysis*
```bash
# Check memory usage
docker stats bev_extended_reasoning

# Implement context compression
python3 -c "from src.agents.extended_reasoning import compress_context"

# Restart with increased memory limits
docker-compose -f docker-compose-thanos-unified.yml up -d --scale bev_extended_reasoning=2
```

**Cross-Node Communication Issues**:

*Issue: Vault connectivity problems*
```bash
# Check Vault status
vault status

# Test authentication
vault auth -method=approle

# Check network connectivity
ping oracle1
telnet oracle1 8200

# Restart Vault services
docker-compose -f docker-compose-oracle1-unified.yml restart bev_vault
```

*Issue: Database connectivity failures*
```bash
# Test database connections
docker exec -it bev_postgres pg_isready

# Check network between nodes
ping thanos
ping oracle1

# Restart database services
docker-compose -f docker-compose-thanos-unified.yml restart bev_postgres
```

### Debugging Tools and Techniques

**Log Analysis**:
```bash
# Centralized log viewing
./scripts/view_all_logs.sh

# Service-specific logs
docker logs bev_avatar_controller -f
docker logs bev_extended_reasoning -f
docker logs bev_mcp_server -f

# System logs
journalctl -f
journalctl -u bev-advanced-avatar -f
```

**Performance Debugging**:
```bash
# System performance monitoring
htop
iotop
nvidia-smi

# Service performance
docker stats
./scripts/performance_report.sh

# Network performance
iftop
./scripts/network_performance.sh
```

**Memory Debugging**:
```bash
# Memory usage analysis
free -h
docker system df

# Memory optimization
docker system prune -f
./scripts/optimize_memory.sh

# GPU memory debugging
nvidia-smi --query-gpu=memory.used,memory.free --format=csv
```

---

## 🧪 Testing and Quality Assurance

### Testing Framework

**Complete Test Suite**:
```bash
# Run all tests
./run_all_tests.sh

# Quick validation tests
./run_all_tests.sh --quick

# Performance tests
./run_all_tests.sh --performance

# Security tests
python run_security_tests.py

# Avatar system tests
cd src/avatar && python3 test_avatar_system.py

# Integration tests
pytest tests/integration/ -v

# OSINT capability tests
pytest tests/osint/ -v
```

**Test Categories**:
```yaml
Unit_Tests:
  location: tests/unit/
  command: pytest tests/unit/ -v
  coverage: "> 80%"

Integration_Tests:
  location: tests/integration/
  command: pytest tests/integration/ -v
  focus: service connectivity, AI workflows

Performance_Tests:
  location: tests/performance/
  command: pytest tests/performance/ --concurrent=100
  targets: 1000+ concurrent ops, <100ms latency

Security_Tests:
  location: tests/security/
  command: python run_security_tests.py
  focus: AI security, data protection

Avatar_Tests:
  location: src/avatar/test_avatar_system.py
  command: python3 test_avatar_system.py
  focus: 3D rendering, emotion processing

OSINT_Tests:
  location: tests/osint/
  command: pytest tests/osint/ -v
  focus: investigation workflows, tool integration
```

### Quality Gates

**Pre-Commit Requirements**:
```bash
# Code formatting (required)
python -m black .

# Linting (required)
python -m flake8 src/ tests/

# Type checking (required)
python -m mypy src/

# Unit tests (required)
pytest tests/unit/ --cov=src --cov-min-percentage=80

# Security validation (required)
python run_security_tests.py --quick
```

**Pre-Deployment Requirements**:
```bash
# Complete test suite
./run_all_tests.sh

# System validation
./validate_bev_deployment.sh

# Performance benchmarking
pytest tests/performance/ --benchmark

# Security assessment
python run_security_tests.py --comprehensive

# Integration validation
pytest tests/integration/ -v
```

---

## 📊 Performance Optimization

### System Performance Tuning

**GPU Optimization**:
```bash
# RTX 4090 optimization (STARLORD)
cd src/avatar && python3 rtx4090_optimizer.py

# RTX 3080 optimization (THANOS)
cd src/infrastructure && python3 gpu_optimizer.py --gpu rtx3080

# Monitor GPU performance
nvidia-smi dmon -s pucvmet -d 1

# GPU memory optimization
python3 -c "import torch; torch.cuda.empty_cache()"
```

**Memory Optimization**:
```bash
# System memory optimization
echo 3 > /proc/sys/vm/drop_caches
./scripts/optimize_system_memory.sh

# Docker memory management
docker system prune -f
docker volume prune -f

# Service memory optimization
./scripts/optimize_service_memory.sh
```

**Database Performance Tuning**:
```bash
# PostgreSQL optimization
./scripts/optimize_postgresql.sh

# Neo4j optimization
./scripts/optimize_neo4j.sh

# Redis optimization
./scripts/optimize_redis.sh

# Elasticsearch optimization
./scripts/optimize_elasticsearch.sh
```

### Performance Monitoring

**Real-Time Performance Dashboard**:
```bash
# Access Grafana dashboard
open http://oracle1:3000/d/bev-performance

# Command-line monitoring
./scripts/performance_monitor.sh

# Resource utilization
htop
iotop -o
nvidia-smi -l 1
```

**Performance Benchmarking**:
```bash
# Avatar system benchmarks
cd src/avatar && python3 benchmark_avatar.py

# AI services benchmarks
cd src/agents && python3 benchmark_reasoning.py

# OSINT performance benchmarks
cd tests/performance && python3 benchmark_osint.py

# System benchmarks
./scripts/system_benchmark.sh
```

---

## 🔐 Security Operations

### Security Management

**Daily Security Checks**:
```bash
# Vault health check
vault status

# Security service validation
./scripts/validate_security_services.sh

# Access audit review
./scripts/audit_access_logs.sh

# Vulnerability scanning
./scripts/security_scan.sh
```

**Incident Response**:
```bash
# Security incident isolation
./scripts/security_incident_isolation.sh

# Evidence preservation
./scripts/preserve_evidence.sh

# System hardening
./scripts/emergency_hardening.sh

# Forensic data collection
./scripts/collect_forensic_data.sh
```

**Access Management**:
```bash
# User access management
vault write auth/userpass/users/newuser password=securepassword policies=developer-policy

# Service access management
vault write auth/approle/role/newservice policies=application-policy

# Access review
vault list auth/userpass/users
vault list auth/approle/role
```

### Security Validation

**Security Testing**:
```bash
# Comprehensive security tests
python run_security_tests.py

# Penetration testing
./scripts/penetration_test.sh

# Vulnerability assessment
./scripts/vulnerability_assessment.sh

# Compliance validation
./scripts/compliance_check.sh
```

---

## 📋 Operational Runbooks

### Emergency Procedures

**System Emergency Response**:
```bash
# Emergency shutdown sequence
./scripts/emergency_shutdown_all_nodes.sh

# Emergency backup
./scripts/emergency_backup_all_data.sh

# Security isolation
./scripts/emergency_isolation.sh

# Incident reporting
./scripts/generate_incident_report.sh
```

**Recovery Procedures**:
```bash
# Complete disaster recovery
./scripts/disaster_recovery_full.sh

# Partial service recovery
./scripts/recover_critical_services.sh

# Data recovery
./scripts/recover_critical_data.sh

# System validation post-recovery
./scripts/validate_recovery.sh
```

### Maintenance Procedures

**Weekly Maintenance**:
```bash
# System updates
./scripts/weekly_system_updates.sh

# Performance optimization
./scripts/weekly_performance_optimization.sh

# Security updates
./scripts/weekly_security_updates.sh

# Backup validation
./scripts/weekly_backup_validation.sh
```

**Monthly Maintenance**:
```bash
# Comprehensive system audit
./scripts/monthly_system_audit.sh

# Performance review
./scripts/monthly_performance_review.sh

# Security assessment
./scripts/monthly_security_assessment.sh

# Capacity planning
./scripts/monthly_capacity_planning.sh
```

---

## 📈 Development Best Practices

### Code Standards

**Python Development Standards**:
```python
# Type hints required
def process_osint_data(target: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process OSINT data with comprehensive type hints

    Args:
        target: Investigation target (domain, IP, email, etc.)
        options: Processing options and configuration

    Returns:
        Processed OSINT intelligence data

    Raises:
        SecurityError: If target validation fails
        RateLimitError: If rate limits are exceeded
    """

# Error handling pattern
try:
    result = await osint_service.investigate(target)
except SecurityError as e:
    logger.error(f"Security validation failed: {e}")
    await avatar_controller.display_error("Security validation failed")
    raise
except RateLimitError as e:
    logger.warning(f"Rate limit exceeded: {e}")
    await avatar_controller.display_warning("Rate limit exceeded, retrying...")
    await asyncio.sleep(30)
    return await process_osint_data(target, options)
```

**JavaScript/TypeScript Standards** (Frontend):
```typescript
// Svelte component development pattern
<script lang="ts">
  import type { OSINTInvestigation, InvestigationResult } from '$lib/types';
  import { avatarStore } from '$lib/stores/avatar';
  import { osintService } from '$lib/services/osint';

  export let investigation: OSINTInvestigation;

  let results: InvestigationResult[] = [];
  let loading = false;

  async function startInvestigation(): Promise<void> {
    loading = true;

    try {
      // Notify avatar of investigation start
      avatarStore.update('investigation_started', {
        target: investigation.target,
        type: investigation.type
      });

      // Execute investigation
      results = await osintService.investigate(investigation);

      // Notify avatar of completion
      avatarStore.update('investigation_completed', {
        results: results.length,
        duration: investigation.duration
      });
    } catch (error) {
      console.error('Investigation failed:', error);
      avatarStore.update('investigation_failed', { error: error.message });
    } finally {
      loading = false;
    }
  }
</script>
```

### Documentation Standards

**Code Documentation Requirements**:
```python
class NewOSINTAnalyzer:
    """
    Advanced OSINT analyzer for [specific purpose]

    This analyzer implements [description of what it does] using [methodology].
    It integrates with the BEV avatar system to provide real-time feedback
    during investigation processes.

    Attributes:
        api_client: External API client for data collection
        avatar_controller: Avatar system integration
        cache_manager: Result caching system

    Example:
        >>> analyzer = NewOSINTAnalyzer()
        >>> await analyzer.initialize()
        >>> results = await analyzer.analyze("example.com")
        >>> print(results['threat_score'])
    """

    def __init__(self, api_key: str, avatar_enabled: bool = True):
        """
        Initialize the OSINT analyzer

        Args:
            api_key: API key for external service authentication
            avatar_enabled: Whether to enable avatar integration

        Raises:
            ConfigurationError: If API key is invalid
        """
```

### Git Workflow

**Branch Management**:
```bash
# Feature development workflow
git checkout -b feature/new-osint-analyzer
git add .
git commit -m "feat: add new OSINT analyzer for threat attribution"
git push origin feature/new-osint-analyzer

# Create pull request via GitHub CLI
gh pr create --title "Add new OSINT analyzer" --body "Implements threat attribution analyzer with avatar integration"

# Code review and merge
gh pr review --approve
gh pr merge --squash
```

**Commit Message Standards**:
```bash
# Format: type(scope): description
feat(avatar): add emotion recognition for threat responses
fix(osint): resolve rate limiting issue in darknet crawler
docs(api): update MCP server documentation
test(integration): add avatar system integration tests
perf(reasoning): optimize token processing for large contexts
security(auth): implement additional Vault authentication methods
```

---

## 🎓 Training and Knowledge Transfer

### Onboarding Checklist

**New Developer Onboarding**:
```yaml
Week_1_Foundation:
  - [ ] Complete development environment setup
  - [ ] Review platform architecture documentation
  - [ ] Complete security training and access setup
  - [ ] Set up IDE with recommended extensions
  - [ ] Complete first tutorial project

Week_2_Platform_Familiarization:
  - [ ] Deploy development environment
  - [ ] Complete avatar system tutorial
  - [ ] Build first OSINT tool integration
  - [ ] Review code standards and contribute first PR
  - [ ] Complete testing framework training

Week_3_Advanced_Integration:
  - [ ] Implement extended reasoning integration
  - [ ] Complete MCP server development tutorial
  - [ ] Contribute to existing OSINT analyzer
  - [ ] Complete security validation training
  - [ ] Participate in code review process

Week_4_Production_Readiness:
  - [ ] Complete operations training
  - [ ] Deploy feature to staging environment
  - [ ] Complete incident response training
  - [ ] Mentor another new developer
  - [ ] Contribute to documentation
```

### Knowledge Resources

**Essential Reading**:
```yaml
Platform_Architecture:
  - BEV_AI_ASSISTANT_PLATFORM_DOCUMENTATION.md
  - BEV_OSINT_FRAMEWORK_TECHNICAL_SPECIFICATION.md
  - BEV_ENTERPRISE_INFRASTRUCTURE_DEPLOYMENT_GUIDE.md

Development_Guides:
  - Avatar system development tutorial
  - Extended reasoning integration guide
  - MCP server development documentation
  - OSINT tool development patterns

Operations_Documentation:
  - Daily operations procedures
  - Incident response playbooks
  - Performance optimization guides
  - Security procedures and protocols
```

---

**Document Version**: 2.0
**Last Updated**: September 21, 2025
**Maintainer**: BEV Operations & Development Team
**Target Audience**: Developers, Operators, System Administrators

---

*This operations and developer guide provides comprehensive procedures for developing, deploying, and operating the BEV AI Assistant Platform, the world's most advanced AI-powered cybersecurity intelligence platform.*