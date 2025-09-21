# BEV OSINT Framework - GitHub Actions CI/CD Pipeline

This directory contains comprehensive GitHub Actions workflows for the BEV OSINT Framework, implementing enterprise-grade CI/CD practices for a complex multi-service OSINT platform.

## 🚀 Pipeline Overview

The BEV CI/CD pipeline implements Phase 1 of enterprise DevOps automation with 5 core workflows designed to handle the complexity of 50+ Dockerfiles, 25,174+ lines of Python code, and multi-architecture deployment requirements.

### Workflow Architecture

```
Pull Request → CI → Build Validation → Security Scan → Code Quality
                                                            ↓
                                                      Merge to Main
                                                            ↓
                                              Deploy Pipeline → Production
```

## 📋 Workflows

### 1. CI - Pull Request Validation (`ci.yml`)

**Purpose**: Comprehensive validation of all pull requests with multi-architecture testing matrix

**Triggers**:
- Pull requests to `main`, `develop`, `enterprise-completion`
- Pushes to `main`, `develop`

**Key Features**:
- **Multi-Architecture Testing**: AMD64 (THANOS services) and ARM64 (ORACLE1 services)
- **Service Group Matrix**: Tests 6 service categories independently
- **Python Quality**: Black, Flake8, MyPy validation for 25,174+ lines
- **Docker Validation**: All 50+ Dockerfiles and Docker Compose files
- **Security Integration**: Bandit, Safety, dependency scanning
- **Performance Testing**: Conditional performance validation

**Service Groups Tested**:
- `thanos` - THANOS services (AMD64 optimized)
- `oracle1` - ORACLE1 services (ARM64 optimized) 
- `alternative-market` - Market Intelligence (5,608+ lines)
- `security` - Security Operations (11,189+ lines)
- `autonomous` - Autonomous Systems (8,377+ lines)
- `infrastructure` - Infrastructure Services

**Success Criteria**:
- All service group tests pass
- Docker Compose files validate
- Python code quality >95% pass rate
- Security scans complete without critical issues

### 2. Build Validation (`build-validation.yml`)

**Purpose**: Matrix build testing for all Docker services with intelligent caching

**Triggers**:
- Changes to Dockerfiles or Docker Compose files
- Nightly builds (2 AM UTC)
- Manual workflow dispatch

**Key Features**:
- **Dynamic Matrix Generation**: Discovers and categorizes all Dockerfiles
- **Intelligent Caching**: Aggressive/moderate/minimal strategies by service type
- **Parallel Building**: Up to 4 concurrent builds per service group
- **Multi-Architecture Support**: Cross-platform compatibility testing
- **Service Integration Testing**: Validates service coordination

**Build Matrix**:
- **THANOS Services**: 14 Dockerfiles (AMD64, aggressive caching)
- **ORACLE1 Services**: 18 Dockerfiles (ARM64, moderate caching)
- **Alternative Market**: 4 specialized analyzers
- **Security Operations**: 8 security services
- **Autonomous Systems**: 5 AI/ML services
- **Infrastructure**: 6 core platform services
- **IntelOwl Analyzers**: 4 custom OSINT analyzers
- **Agent Swarm**: 6 AI agent services
- **MCP Servers**: 7 MCP protocol servers
- **Tor Infrastructure**: 3 anonymization services

**Performance Targets**:
- Build time <30 minutes for full validation
- Cache hit rate >70% for faster iterations
- All Dockerfiles build successfully
- Service integration tests pass

### 3. Security Scanning (`security-scan.yml`)

**Purpose**: Comprehensive security analysis with SAST/DAST/Container security

**Triggers**:
- All pull requests and pushes
- Daily security scans (3 AM UTC)
- Manual dispatch with scan type selection

**Security Components**:

#### SAST (Static Application Security Testing)
- **Bandit**: Python security vulnerability detection
- **Semgrep**: Multi-language security patterns
- **Custom OSINT Rules**: OSINT-specific security patterns
- **Dependency Scanning**: Safety + pip-audit for vulnerabilities

#### Container Security
- **Trivy**: Container vulnerability scanning
- **Docker Bench Security**: Container runtime security
- **Dockerfile Analysis**: Security configuration validation
- **Runtime Security**: Live container security assessment

#### DAST (Dynamic Application Security Testing)
- **OWASP ZAP**: Web application security testing
- **Custom OSINT Tests**: OSINT-specific security validation
- **API Security**: FastAPI endpoint security testing
- **Rate Limiting**: DDoS protection validation

#### Infrastructure Security
- **Kubernetes Security**: K8s manifest security analysis (if applicable)
- **Docker Compose Security**: Service configuration security
- **Network Security**: Container networking validation

**Security Standards**:
- Zero critical vulnerabilities in production
- All security findings documented in GitHub Security tab
- SARIF format for standardized reporting
- Security metrics tracked and reported

### 4. Code Quality (`code-quality.yml`)

**Purpose**: Comprehensive code quality validation and documentation standards

**Triggers**:
- All pull requests and pushes
- Manual dispatch with quality check type selection

**Quality Dimensions**:

#### Python Code Quality
- **Black**: Consistent code formatting (88 character line length)
- **isort**: Import sorting and organization
- **Flake8**: PEP 8 compliance and error detection
- **MyPy**: Type checking and annotation validation
- **Pylint**: Advanced linting and code analysis
- **pydocstyle**: Documentation style validation

#### JavaScript/TypeScript Quality (Conditional)
- **Prettier**: Code formatting consistency
- **ESLint**: Linting with TypeScript support
- **Vue.js**: Component-specific linting

#### Docker and Configuration Quality
- **Hadolint**: Dockerfile best practices
- **yamllint**: YAML file validation
- **Docker Compose**: Service configuration validation

#### Documentation Quality
- **markdownlint**: Markdown formatting and style
- **Link Validation**: Broken link detection
- **Documentation Coverage**: Missing documentation detection
- **Structure Analysis**: Documentation organization validation

#### Dependency Quality
- **Security Audit**: Python dependency vulnerabilities
- **License Analysis**: License compliance checking
- **Dependency Tree**: Dependency relationship analysis

**Quality Targets**:
- 100% code formatting compliance
- >90% type annotation coverage
- Zero critical linting issues
- Complete documentation coverage
- License compliance validation

### 5. Deploy Pipeline (`deploy-pipeline.yml`)

**Purpose**: Production-ready deployment orchestration with environment management

**Triggers**:
- Pushes to `main` branch (staging)
- Git tags `v*` (production)
- Manual deployment with environment selection

**Deployment Environments**:
- **Staging**: Development/testing environment
- **Production**: Live production deployment
- **Multinode THANOS**: AMD64 optimized cluster
- **Multinode ORACLE1**: ARM64 optimized cluster

**Deployment Flow**:

1. **Pre-deployment Validation**
   - Deployment file verification
   - Environment configuration validation
   - Force deployment option for emergency deployments

2. **Pipeline Orchestration**
   - Calls CI workflow for validation
   - Executes build validation
   - Runs security scanning
   - Performs code quality checks

3. **Image Building and Registry**
   - Multi-platform image building (AMD64/ARM64)
   - Container registry push (GHCR)
   - Image tagging with semantic versioning
   - Build caching optimization

4. **Environment Deployment**
   - **Staging**: Development compose with health checks
   - **Production**: Complete production stack deployment
   - **Multinode**: Distributed deployment configuration

5. **Post-deployment Validation**
   - Comprehensive system testing
   - Health monitoring validation
   - Performance verification
   - Deployment reporting

## 🔧 Configuration

### Required GitHub Secrets

```yaml
GITHUB_TOKEN: # Automatically provided by GitHub
# Additional secrets for production deployments:
PRODUCTION_DEPLOY_KEY: # Production environment access
STAGING_DEPLOY_KEY: # Staging environment access
DOCKER_HUB_TOKEN: # Optional: Docker Hub registry access
```

### Environment Variables

```yaml
REGISTRY: ghcr.io
IMAGE_NAME: ${{ github.repository }}
PYTHON_VERSION: '3.11'
NODE_VERSION: '18'
```

### Repository Settings

#### Branch Protection Rules
- Require pull request reviews
- Require status checks to pass
- Require branches to be up to date
- Include administrators

#### Required Status Checks
- `ci / ci-status`
- `build-validation / build-status`  
- `security-scan / security-summary`
- `code-quality / quality-summary`

## 📊 Success Metrics

### Build Performance
- **Full Pipeline Time**: <45 minutes
- **CI Validation**: <20 minutes
- **Build Validation**: <30 minutes
- **Security Scanning**: <15 minutes
- **Code Quality**: <10 minutes

### Quality Targets
- **Test Coverage**: >80%
- **Code Quality**: >95% pass rate
- **Security**: Zero critical vulnerabilities
- **Build Success**: >99% for main branch
- **Deployment Success**: >99% for production

### Resource Efficiency
- **Cache Hit Rate**: >70%
- **Parallel Execution**: 60% time savings
- **Resource Usage**: <4 concurrent runners
- **Storage**: <10GB artifact storage

## 🚨 Troubleshooting

### Common Issues

#### Build Failures
```bash
# Check specific service group logs
# Review Docker build context
# Verify dependency versions
# Check resource availability
```

#### Test Failures
```bash
# Review test logs in artifacts
# Check service dependencies
# Verify environment configuration
# Run tests locally for debugging
```

#### Security Scan Issues
```bash
# Review SARIF reports in GitHub Security tab
# Check dependency vulnerabilities
# Verify container configurations
# Update security baselines
```

#### Deployment Issues
```bash
# Verify environment configuration
# Check deployment file syntax
# Review infrastructure availability
# Validate credentials and access
```

### Debug Commands

```bash
# Local CI simulation
act pull_request

# Local build validation  
docker-compose -f docker-compose.complete.yml config -q

# Local security testing
python run_security_tests.py

# Local code quality
python -m black --check src/
python -m flake8 src/
python -m mypy src/
```

## 🔄 Workflow Maintenance

### Regular Updates
- **Monthly**: Update action versions
- **Quarterly**: Review security baselines
- **Bi-annually**: Optimize build performance
- **Annually**: Architecture review

### Monitoring
- Pipeline success rates
- Build duration trends
- Security vulnerability trends
- Code quality metrics

### Optimization Opportunities
- Build cache optimization
- Test parallelization
- Resource allocation tuning
- Workflow orchestration improvements

## 📖 Documentation

### Related Documentation
- **CLAUDE.md**: Project-specific development guidelines
- **README.md**: Project overview and setup
- **DEPLOYMENT_GUIDE.md**: Deployment procedures
- **SECURITY.md**: Security policies and procedures

### External Resources
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [OWASP Security Guidelines](https://owasp.org/)
- [Python Quality Tools](https://github.com/psf/black)

---

**Note**: This CI/CD implementation represents Phase 1 of enterprise DevOps automation for the BEV OSINT Framework. Future phases will include advanced deployment strategies, comprehensive monitoring, and automated rollback procedures.