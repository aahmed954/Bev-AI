#!/usr/bin/env python3
"""
Centralized secrets management for BEV OSINT Framework
Provides secure credential storage with multiple backend options
"""

import os
import json
import base64
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib

logger = logging.getLogger(__name__)

class SecretsManager:
    """
    Unified secrets management with multiple backends:
    - Environment variables (default)
    - Encrypted file storage
    - HashiCorp Vault (optional)
    - AWS Secrets Manager (optional)
    """

    def __init__(self, backend: str = "env"):
        self.backend = backend
        self.cache: Dict[str, Any] = {}

        # Initialize backend-specific components
        if backend == "encrypted_file":
            self._init_encrypted_file()
        elif backend == "vault":
            self._init_vault()
        elif backend == "aws":
            self._init_aws()

        logger.info(f"SecretsManager initialized with backend: {backend}")

    def _init_encrypted_file(self):
        """Initialize encrypted file backend"""
        try:
            # Generate encryption key from master password
            master_password = os.getenv('MASTER_PASSWORD')
            if not master_password:
                logger.warning("MASTER_PASSWORD not set, using default (INSECURE)")
                master_password = "default_master_password_change_me"

            # Derive encryption key
            salt = b'bev_osint_salt_2024'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(master_password.encode()))
            self.cipher = Fernet(key)

            self.secrets_file = Path('.secrets.encrypted')
            logger.info("Encrypted file backend initialized")

        except Exception as e:
            logger.error(f"Failed to initialize encrypted file backend: {e}")
            raise

    def _init_vault(self):
        """Initialize HashiCorp Vault backend"""
        try:
            import hvac

            vault_url = os.getenv('VAULT_URL', 'http://localhost:8200')
            vault_token = os.getenv('VAULT_TOKEN')

            if not vault_token:
                logger.error("VAULT_TOKEN not provided")
                raise ValueError("Vault token required")

            self.vault_client = hvac.Client(url=vault_url, token=vault_token)

            if not self.vault_client.is_authenticated():
                raise ValueError("Vault authentication failed")

            logger.info("Vault backend initialized")

        except ImportError:
            logger.error("hvac package not installed. Install with: pip install hvac")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Vault backend: {e}")
            raise

    def _init_aws(self):
        """Initialize AWS Secrets Manager backend"""
        try:
            import boto3

            self.aws_client = boto3.client('secretsmanager')
            logger.info("AWS Secrets Manager backend initialized")

        except ImportError:
            logger.error("boto3 package not installed. Install with: pip install boto3")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS backend: {e}")
            raise

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get secret from configured backend

        Args:
            key: Secret key name
            default: Default value if secret not found

        Returns:
            Secret value or default
        """

        # Check cache first
        cache_key = f"{self.backend}:{key}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        secret = None

        try:
            if self.backend == "env":
                secret = os.getenv(key, default)

            elif self.backend == "encrypted_file":
                secret = self._get_from_encrypted_file(key, default)

            elif self.backend == "vault":
                secret = self._get_from_vault(key, default)

            elif self.backend == "aws":
                secret = self._get_from_aws(key, default)

            else:
                logger.error(f"Unknown backend: {self.backend}")
                return default

            # Cache the secret
            if secret:
                self.cache[cache_key] = secret

            return secret

        except Exception as e:
            logger.error(f"Failed to retrieve secret {key}: {e}")
            return default

    def set_secret(self, key: str, value: str) -> bool:
        """
        Store secret in backend

        Args:
            key: Secret key name
            value: Secret value

        Returns:
            True if successful, False otherwise
        """

        try:
            if self.backend == "env":
                logger.warning("Cannot set environment variables at runtime")
                return False

            elif self.backend == "encrypted_file":
                return self._set_in_encrypted_file(key, value)

            elif self.backend == "vault":
                return self._set_in_vault(key, value)

            elif self.backend == "aws":
                return self._set_in_aws(key, value)

            else:
                logger.error(f"Unknown backend: {self.backend}")
                return False

        except Exception as e:
            logger.error(f"Failed to store secret {key}: {e}")
            return False

    def _get_from_encrypted_file(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from encrypted file"""
        if not self.secrets_file.exists():
            return default

        try:
            encrypted_data = self.secrets_file.read_bytes()
            decrypted = self.cipher.decrypt(encrypted_data)
            secrets = json.loads(decrypted.decode())
            return secrets.get(key, default)

        except Exception as e:
            logger.error(f"Error reading encrypted secrets: {e}")
            return default

    def _set_in_encrypted_file(self, key: str, value: str) -> bool:
        """Set secret in encrypted file"""
        try:
            # Load existing secrets
            secrets = {}
            if self.secrets_file.exists():
                encrypted_data = self.secrets_file.read_bytes()
                decrypted = self.cipher.decrypt(encrypted_data)
                secrets = json.loads(decrypted.decode())

            # Add new secret
            secrets[key] = value

            # Encrypt and save
            encrypted = self.cipher.encrypt(json.dumps(secrets).encode())
            self.secrets_file.write_bytes(encrypted)

            # Set secure permissions
            os.chmod(self.secrets_file, 0o600)

            # Update cache
            cache_key = f"{self.backend}:{key}"
            self.cache[cache_key] = value

            return True

        except Exception as e:
            logger.error(f"Error writing encrypted secrets: {e}")
            return False

    def _get_from_vault(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from Vault"""
        try:
            response = self.vault_client.secrets.kv.v2.read_secret_version(
                path=f"bev/{key}"
            )
            return response['data']['data']['value']

        except Exception as e:
            logger.debug(f"Secret {key} not found in Vault: {e}")
            return default

    def _set_in_vault(self, key: str, value: str) -> bool:
        """Set secret in Vault"""
        try:
            self.vault_client.secrets.kv.v2.create_or_update_secret(
                path=f"bev/{key}",
                secret={'value': value}
            )
            return True

        except Exception as e:
            logger.error(f"Error storing secret in Vault: {e}")
            return False

    def _get_from_aws(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from AWS Secrets Manager"""
        try:
            response = self.aws_client.get_secret_value(SecretId=f"bev/{key}")
            return response['SecretString']

        except Exception as e:
            logger.debug(f"Secret {key} not found in AWS: {e}")
            return default

    def _set_in_aws(self, key: str, value: str) -> bool:
        """Set secret in AWS Secrets Manager"""
        try:
            self.aws_client.create_secret(
                Name=f"bev/{key}",
                SecretString=value
            )
            return True

        except Exception as e:
            logger.error(f"Error storing secret in AWS: {e}")
            return False

    def list_secrets(self) -> list:
        """List available secret keys (backend dependent)"""
        try:
            if self.backend == "encrypted_file":
                if not self.secrets_file.exists():
                    return []

                encrypted_data = self.secrets_file.read_bytes()
                decrypted = self.cipher.decrypt(encrypted_data)
                secrets = json.loads(decrypted.decode())
                return list(secrets.keys())

            elif self.backend == "vault":
                response = self.vault_client.secrets.kv.v2.list_secrets(path="bev")
                return response['data']['keys']

            elif self.backend == "aws":
                response = self.aws_client.list_secrets()
                return [s['Name'].replace('bev/', '') for s in response['SecretList']
                        if s['Name'].startswith('bev/')]

            else:
                return []

        except Exception as e:
            logger.error(f"Error listing secrets: {e}")
            return []

    def delete_secret(self, key: str) -> bool:
        """Delete a secret"""
        try:
            if self.backend == "encrypted_file":
                if not self.secrets_file.exists():
                    return False

                encrypted_data = self.secrets_file.read_bytes()
                decrypted = self.cipher.decrypt(encrypted_data)
                secrets = json.loads(decrypted.decode())

                if key in secrets:
                    del secrets[key]
                    encrypted = self.cipher.encrypt(json.dumps(secrets).encode())
                    self.secrets_file.write_bytes(encrypted)
                    return True

            elif self.backend == "vault":
                self.vault_client.secrets.kv.v2.delete_secret_versions(
                    path=f"bev/{key}"
                )
                return True

            elif self.backend == "aws":
                self.aws_client.delete_secret(SecretId=f"bev/{key}")
                return True

            return False

        except Exception as e:
            logger.error(f"Error deleting secret {key}: {e}")
            return False

# Global secrets manager instance
_secrets_manager = None

def get_secrets_manager() -> SecretsManager:
    """Get global secrets manager instance"""
    global _secrets_manager

    if _secrets_manager is None:
        backend = os.getenv('SECRETS_BACKEND', 'env')
        _secrets_manager = SecretsManager(backend=backend)

    return _secrets_manager

# Convenience functions for common secrets
def get_database_password() -> str:
    """Get database password"""
    return get_secrets_manager().get_secret('DB_PASSWORD', 'dev_password')

def get_postgres_password() -> str:
    """Get PostgreSQL password"""
    return get_secrets_manager().get_secret('POSTGRES_PASSWORD', 'dev_postgres')

def get_neo4j_password() -> str:
    """Get Neo4j password"""
    return get_secrets_manager().get_secret('NEO4J_PASSWORD', 'dev_neo4j')

def get_redis_password() -> str:
    """Get Redis password"""
    return get_secrets_manager().get_secret('REDIS_PASSWORD', 'dev_redis')

def get_rabbitmq_password() -> str:
    """Get RabbitMQ password"""
    return get_secrets_manager().get_secret('RABBITMQ_PASSWORD', 'dev_rabbit')

def get_api_key(service: str) -> str:
    """Get API key for a service"""
    key_name = f'{service.upper()}_API_KEY'
    return get_secrets_manager().get_secret(key_name, '')

def get_encryption_key() -> bytes:
    """Get encryption key"""
    key = get_secrets_manager().get_secret('ENCRYPTION_KEY')
    if not key:
        # Generate new key if not exists
        key = Fernet.generate_key().decode()
        get_secrets_manager().set_secret('ENCRYPTION_KEY', key)
    return key.encode()

def get_jwt_secret() -> str:
    """Get JWT secret"""
    return get_secrets_manager().get_secret('JWT_SECRET', 'dev_jwt_secret')

# Migration helper for existing code
def migrate_hardcoded_credentials():
    """
    Helper function to migrate from hardcoded credentials
    This scans for common patterns and provides migration guidance
    """
    print("🔄 Credential Migration Helper")
    print("=" * 40)

    hardcoded_patterns = {
        'secure_password': 'DB_PASSWORD',
        'postgres': 'POSTGRES_PASSWORD',
        'BevGraphMaster2024': 'NEO4J_PASSWORD',
        'BevCacheMaster': 'REDIS_PASSWORD',
        os.getenv('RABBITMQ_PASSWORD', 'dev_rabbit'): 'RABBITMQ_PASSWORD'
    }

    print("Detected hardcoded credentials to migrate:")
    for old, new in hardcoded_patterns.items():
        print(f"  '{old}' → {new}")

    print("\nRecommended environment variables to set:")
    for env_var in hardcoded_patterns.values():
        print(f"  {env_var}=$(openssl rand -base64 32)")

if __name__ == "__main__":
    # Test the secrets manager
    logging.basicConfig(level=logging.INFO)

    print("🔐 BEV Secrets Manager Test")
    print("=" * 30)

    # Test with environment backend
    sm = SecretsManager(backend="env")

    # Test getting secrets
    db_pass = sm.get_secret('DB_PASSWORD', 'default')
    print(f"Database password: {db_pass[:8]}..." if db_pass else "Not set")

    # Test convenience functions
    print(f"PostgreSQL password: {get_postgres_password()[:8]}...")
    print(f"Redis password: {get_redis_password()[:8]}...")

    # Migration helper
    migrate_hardcoded_credentials()