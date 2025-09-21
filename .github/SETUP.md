# GitHub Actions CI/CD Setup Guide

This guide provides step-by-step instructions for setting up and configuring the BEV OSINT Framework CI/CD pipeline.

## 🚀 Quick Setup

### 1. Repository Configuration

#### Enable GitHub Actions
1. Go to your repository **Settings** → **Actions** → **General**
2. Select **"Allow all actions and reusable workflows"**
3. Enable **"Allow GitHub Actions to create and approve pull requests"**

#### Configure Branch Protection
1. Go to **Settings** → **Branches**
2. Add rule for `main` branch:
   ```yaml
   Require pull request reviews: ✅
   Require status checks: ✅ 
   Required status checks:
     - ci / ci-status
     - build-validation / build-status
     - security-scan / security-summary
     - code-quality / quality-summary
   Require branches to be up to date: ✅
   Include administrators: ✅
   ```

#### Enable Security Features
1. Go to **Settings** → **Security & analysis**
2. Enable:
   - **Dependency graph**: ✅
   - **Dependabot alerts**: ✅
   - **Dependabot security updates**: ✅
   - **Code scanning alerts**: ✅
   - **Secret scanning alerts**: ✅

### 2. Environment Setup

#### Create Environments
1. Go to **Settings** → **Environments**
2. Create environments:
   - `staging` - For development deployments
   - `production` - For production deployments  
   - `multinode-thanos` - For THANOS cluster deployments
   - `multinode-oracle1` - For ORACLE1 cluster deployments

#### Configure Environment Protection
For `production` environment:
```yaml
Required reviewers: [admin-users]
Wait timer: 0 minutes
Deployment branches: Selected branches
  - main
  - refs/tags/v*
```

### 3. Secrets Configuration

#### Repository Secrets
Go to **Settings** → **Secrets and variables** → **Actions**:

```yaml
# GitHub Token (automatically provided)
GITHUB_TOKEN: # Auto-generated

# Optional: Additional registry access
DOCKER_HUB_USERNAME: your-docker-username
DOCKER_HUB_TOKEN: your-docker-token

# Production deployment (if using external infrastructure)
PRODUCTION_DEPLOY_KEY: your-production-ssh-key
STAGING_DEPLOY_KEY: your-staging-ssh-key

# Database credentials for testing
TEST_POSTGRES_PASSWORD: secure-test-password
TEST_REDIS_PASSWORD: secure-test-redis-password
TEST_NEO4J_PASSWORD: secure-test-neo4j-password
```

#### Environment-Specific Secrets
For each environment, add specific configuration:

**Staging Environment:**
```yaml
STAGING_DATABASE_URL: postgresql://user:pass@staging-db:5432/bev_staging
STAGING_REDIS_URL: redis://:pass@staging-redis:6379
STAGING_SECRET_KEY: staging-secret-key-change-me
```

**Production Environment:**
```yaml
PRODUCTION_DATABASE_URL: postgresql://user:pass@prod-db:5432/bev_production
PRODUCTION_REDIS_URL: redis://:pass@prod-redis:6379
PRODUCTION_SECRET_KEY: production-secret-key-secure
```

## 🔧 Advanced Configuration

### Custom Workflow Triggers

#### Manual Deployment
Add workflow dispatch to any workflow:
```yaml
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options: [staging, production]
      force_deploy:
        description: 'Force deployment'
        type: boolean
        default: false
```

#### Conditional Workflows
Configure workflows to run only when relevant:
```yaml
on:
  pull_request:
    paths:
      - 'src/**'
      - 'docker/**'
      - 'requirements*.txt'
      - '.github/workflows/**'
```

### Performance Optimization

#### Build Caching
Configure aggressive caching for faster builds:
```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3
  with:
    driver-opts: |
      image=moby/buildkit:buildx-stable-1
      network=host

- name: Build with cache
  uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

#### Parallel Execution
Optimize job dependencies for maximum parallelism:
```yaml
jobs:
  test-group-1:
    runs-on: ubuntu-latest
    # Independent execution
    
  test-group-2:
    runs-on: ubuntu-latest  
    # Independent execution
    
  integration-tests:
    needs: [test-group-1, test-group-2]
    # Runs after both groups complete
```

### Security Configuration

#### SARIF Upload
Ensure security findings appear in GitHub Security tab:
```yaml
- name: Upload Security Results
  uses: github/codeql-action/upload-sarif@v2
  if: always()
  with:
    sarif_file: security-results.sarif
    category: security-scan
```

#### Dependency Management
Configure Dependabot for automated dependency updates:

Create `.github/dependabot.yml`:
```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 5
    
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    open-pull-requests-limit: 5
```

## 🎯 Workflow Customization

### Service-Specific Testing

#### Alternative Market Services
```yaml
- name: Test Alternative Market Intelligence
  run: |
    pytest tests/integration/test_alternative_market.py -v \
      --cov=src/alternative_market \
      --cov-report=xml \
      --timeout=300
```

#### Security Operations
```yaml
- name: Test Security Operations
  run: |
    pytest tests/integration/test_security.py -v \
      --cov=src/security \
      --security-tests \
      --timeout=600
```

#### Autonomous Systems
```yaml
- name: Test Autonomous Systems
  run: |
    pytest tests/integration/test_autonomous.py -v \
      --cov=src/autonomous \
      --gpu-tests \
      --timeout=900
```

### Multi-Architecture Builds

#### THANOS Services (AMD64)
```yaml
- name: Build THANOS Services
  run: |
    docker buildx build \
      --platform linux/amd64 \
      --file thanos/phase5/controller/Dockerfile \
      --tag bev-thanos-controller:latest \
      --load .
```

#### ORACLE1 Services (ARM64)
```yaml  
- name: Build ORACLE1 Services
  run: |
    docker buildx build \
      --platform linux/arm64 \
      --file docker/oracle/Dockerfile.blackmarket \
      --tag bev-oracle-blackmarket:latest \
      --load .
```

## 📊 Monitoring and Observability

### Workflow Metrics

#### Success Rate Monitoring
Track workflow success rates:
```bash
# Get workflow success rate for last 30 days
gh api repos/owner/repo/actions/workflows/ci.yml/runs \
  --jq '.workflow_runs | map(select(.created_at > (now - 30*24*60*60 | todate))) | group_by(.conclusion) | map({conclusion: .[0].conclusion, count: length})'
```

#### Performance Tracking
Monitor build duration trends:
```bash
# Get average build duration
gh api repos/owner/repo/actions/workflows/build-validation.yml/runs \
  --jq '.workflow_runs | map(.run_started_at as $start | .updated_at as $end | ($end | fromdateiso8601) - ($start | fromdateiso8601)) | add / length'
```

### Alert Configuration

#### Workflow Failure Notifications
Set up Slack/email notifications for critical failures:
```yaml
- name: Notify on Failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

#### Security Alert Integration
Configure security finding notifications:
```yaml
- name: Security Alert
  if: contains(steps.security-scan.outputs.findings, 'CRITICAL')
  run: |
    echo "Critical security findings detected"
    # Send alert to security team
```

## 🔍 Troubleshooting

### Common Issues

#### Workflow Permission Errors
```yaml
permissions:
  contents: read
  security-events: write
  actions: read
  pull-requests: write
```

#### Docker Build Context Issues
```bash
# Ensure .dockerignore is properly configured
echo "node_modules/" >> .dockerignore
echo ".git/" >> .dockerignore
echo "*.log" >> .dockerignore
```

#### Large Artifact Storage
```yaml
- name: Upload Artifacts
  uses: actions/upload-artifact@v3
  with:
    retention-days: 5  # Reduce retention
    if-no-files-found: ignore
```

### Debug Commands

#### Local Workflow Testing
```bash
# Install act for local testing
curl https://raw.githubusercontent.com/nektos/act/master/install.sh | bash

# Run specific workflow locally
act pull_request -W .github/workflows/ci.yml

# Run with custom environment
act -s GITHUB_TOKEN=$GITHUB_TOKEN pull_request
```

#### Workflow Debugging
```yaml
- name: Debug Information
  run: |
    echo "Runner OS: $RUNNER_OS"
    echo "Runner Arch: $RUNNER_ARCH"
    echo "GitHub Event: $GITHUB_EVENT_NAME"
    echo "GitHub Ref: $GITHUB_REF"
    echo "GitHub SHA: $GITHUB_SHA"
    df -h
    free -h
    docker version
```

## 📚 Best Practices

### Security Best Practices
1. **Never commit secrets** - Use GitHub Secrets
2. **Minimal permissions** - Grant only required permissions
3. **Pin action versions** - Use specific version tags
4. **Regular updates** - Keep actions and dependencies current
5. **Secure defaults** - Fail secure, validate inputs

### Performance Best Practices
1. **Cache dependencies** - Use caching for faster builds
2. **Parallel execution** - Maximize job parallelism
3. **Conditional execution** - Skip unnecessary steps
4. **Resource optimization** - Match resources to workload
5. **Artifact management** - Clean up old artifacts

### Maintenance Best Practices
1. **Regular reviews** - Monthly workflow review
2. **Metric tracking** - Monitor success rates and duration
3. **Documentation updates** - Keep docs current
4. **Version updates** - Regular action and tool updates
5. **Security audits** - Quarterly security reviews

## 🎓 Learning Resources

### GitHub Actions
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Action Marketplace](https://github.com/marketplace?type=actions)
- [Workflow Syntax](https://docs.github.com/en/actions/learn-github-actions/workflow-syntax-for-github-actions)

### DevOps Practices
- [DevOps Best Practices](https://docs.microsoft.com/en-us/devops/)
- [CI/CD Pipeline Best Practices](https://about.gitlab.com/topics/ci-cd/)
- [Infrastructure as Code](https://www.terraform.io/intro/index.html)

### Security
- [OWASP DevSecOps](https://owasp.org/www-project-devsecops-guideline/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [Container Security](https://kubernetes.io/docs/concepts/security/)

---

**Next Steps**: After completing this setup, proceed to test the pipeline with a small pull request to verify all workflows execute correctly.