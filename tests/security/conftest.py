#!/usr/bin/env python3
"""
Pytest configuration for security module tests
"""

import pytest
import asyncio
import os
import sys
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock external dependencies that aren't available in test environment"""
    with patch('docker.from_env', return_value=MagicMock()):
        with patch('asyncpg.create_pool', return_value=AsyncMock()):
            with patch('redis.asyncio.from_url', return_value=AsyncMock()):
                with patch('aiohttp.ClientSession', return_value=AsyncMock()):
                    with patch('subprocess.run', return_value=MagicMock(returncode=0)):
                        with patch('geoip2.database.Reader', return_value=MagicMock()):
                            with patch('scapy.all.sniff', return_value=None):
                                yield

@pytest.fixture
def mock_ml_models():
    """Mock machine learning models to avoid loading large model files"""
    with patch('torch.load', return_value={}):
        with patch('transformers.AutoTokenizer.from_pretrained', return_value=MagicMock()):
            with patch('transformers.AutoModel.from_pretrained', return_value=MagicMock()):
                yield

@pytest.fixture
def temp_directory():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_prometheus_metrics():
    """Mock Prometheus metrics to avoid actual metric collection"""
    with patch('prometheus_client.Counter'):
        with patch('prometheus_client.Histogram'):
            with patch('prometheus_client.Gauge'):
                yield

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)