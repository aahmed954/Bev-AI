"""
Pytest configuration and shared fixtures
========================================

Common test fixtures and configuration for all test modules.
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, AsyncMock
from pathlib import Path

# Set up asyncio event loop policy for tests
@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()

@pytest.fixture(scope="session")
def event_loop(event_loop_policy):
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()

# Test configuration constants
TEST_CONFIG = {
    "jwt_secret": "test_jwt_secret_key_for_testing_purposes_32_chars",
    "encryption_key": "test_encryption_key_32_characters",
    "postgres_uri": "postgresql://test_user:test_pass@localhost:5432/test_db",
    "neo4j_uri": "bolt://localhost:7687",
    "redis_host": "localhost",
    "elasticsearch_host": "localhost"
}

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    env_vars = {
        "JWT_SECRET": TEST_CONFIG["jwt_secret"],
        "DATA_ENCRYPTION_KEY": TEST_CONFIG["encryption_key"],
        "POSTGRES_URI": TEST_CONFIG["postgres_uri"],
        "NEO4J_URI": TEST_CONFIG["neo4j_uri"],
        "NEO4J_USER": "test_user",
        "NEO4J_PASSWORD": "test_pass",
        "REDIS_HOST": TEST_CONFIG["redis_host"],
        "REDIS_PASSWORD": "test_redis_pass",
        "ELASTICSEARCH_HOST": TEST_CONFIG["elasticsearch_host"],
        "LOG_LEVEL": "DEBUG",
        "API_RATE_LIMIT": "100",
        "ALLOWED_NETWORKS": "127.0.0.1/32,192.168.1.0/24"
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars

@pytest.fixture
def sample_test_data():
    """Sample test data for various test scenarios"""
    return {
        "valid_email": "test@example.com",
        "invalid_email": "invalid.email",
        "valid_domain": "example.com",
        "invalid_domain": "invalid..domain",
        "valid_ip": "192.168.1.1",
        "invalid_ip": "256.1.1.1",
        "valid_hash_md5": "d41d8cd98f00b204e9800998ecf8427e",
        "valid_hash_sha1": "da39a3ee5e6b4b0d3255bfef95601890afd80709",
        "invalid_hash": "not_hex_characters",
        "sql_injection": "'; DROP TABLE users; --",
        "command_injection": "test; rm -rf /",
        "xss_payload": "<script>alert('xss')</script>",
        "clean_input": "normal search query"
    }

@pytest.fixture
def mock_redis_responses():
    """Common Redis response mocking patterns"""
    return {
        "empty": None,
        "rate_limit_ok": "50",  # Current request count
        "rate_limit_exceeded": "100",  # At limit
        "session_valid": "valid_session_data",
        "cached_result": '{"cached": true, "data": "test"}'
    }

# Marks for test categorization
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.security = pytest.mark.security
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow