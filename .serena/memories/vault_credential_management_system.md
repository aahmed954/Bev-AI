# BEV Centralized Credential Management System - CRITICAL ARCHITECTURE

## DISCOVERY: HashiCorp Vault System (Completely Missed Previously)

The BEV project has a comprehensive, military-grade centralized credential management system using HashiCorp Vault that was completely overlooked in previous deployment attempts.

## Core Components

### 1. HashiCorp Vault Configuration
- **Location**: `config/vault.hcl`
- **Setup Script**: `config/vault-setup.sh`
- **Policies Directory**: `config/vault-policies/`
- **Default Address**: https://localhost:8200

### 2. SecretsManager Python Library
- **Location**: `src/infrastructure/secrets_manager.py`
- **Backends Supported**:
  - Environment variables (default/dev)
  - Encrypted file storage
  - HashiCorp Vault (production)
  - AWS Secrets Manager

### 3. Vault Security Policies
Multiple role-based access control policies:
- `admin-policy.hcl` - Full administrative access
- `application-policy.hcl` - Application service access
- `developer-policy.hcl` - Development access
- `cicd-policy.hcl` - CI/CD pipeline access
- `oracle-worker-policy.hcl` - ORACLE worker node access
- `security-team-policy.hcl` - Security team access

### 4. Secrets Engines Configured
- KV v2 secrets engine at `kv2/`
- Database secrets engine
- Transit encryption engine
- PKI (Public Key Infrastructure)
- SSH secrets engine
- AWS/GCP cloud secrets engines

## Critical Integration Points

### Python Service Integration
All Python services use the `SecretsManager` class:
```python
from src.infrastructure.secrets_manager import get_secrets_manager
sm = get_secrets_manager()
password = sm.get_secret('DB_PASSWORD')
```

### Convenience Functions
Pre-built functions for common credentials:
- `get_database_password()`
- `get_postgres_password()`
- `get_neo4j_password()`
- `get_redis_password()`
- `get_rabbitmq_password()`
- `get_api_key(service)`
- `get_encryption_key()`
- `get_jwt_secret()`

## Deployment Configuration

### Environment Variables Required
```
VAULT_URL=https://localhost:8200
VAULT_TOKEN=<root_or_role_token>
SECRETS_BACKEND=vault  # or 'env' for development
```

### AppRole Authentication
Services authenticate via AppRole:
- `cicd` - CI/CD pipelines
- `app-services` - Application services
- `dev-services` - Development services
- `oracle-workers` - ORACLE distributed workers

## Critical Deployment Steps

1. **Initialize Vault First**
   ```bash
   cd config
   ./vault-setup.sh
   ```

2. **Secure vault-init.json**
   Contains root token and unseal keys - MUST BE SECURED

3. **Set Environment Variables**
   Export VAULT_TOKEN and VAULT_URL before deployments

4. **Generate Secure Passwords**
   ```bash
   ./generate_secrets.sh
   ```

5. **Load Secrets into Vault**
   Use vault CLI or SecretsManager to populate

## Service Dependencies

Services expecting centralized credentials:
- All database connections (PostgreSQL, Neo4j, Redis)
- Message queues (RabbitMQ, Kafka)
- API integrations (OpenAI, Anthropic, etc.)
- Inter-service authentication
- Encryption keys

## Failure Points

Previous deployments failed because:
1. No Vault server was running
2. VAULT_TOKEN was not provided
3. Services defaulted to environment variables
4. Hardcoded dev passwords were used
5. Cross-node authentication was broken

## Recovery Actions

1. Deploy Vault server first
2. Initialize and unseal Vault
3. Populate all required secrets
4. Update all .env files with VAULT_TOKEN
5. Set SECRETS_BACKEND=vault for production
6. Restart all services with proper credentials
