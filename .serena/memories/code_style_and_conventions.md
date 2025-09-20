# BEV OSINT Framework - Code Style & Conventions

## Python Style Guidelines

### Code Formatting
- **Black** formatter with default settings (88 character line length)
- **Automatic formatting**: `python -m black .`
- **Check without changes**: `python -m black --check .`

### Naming Conventions
```python
# Variables and functions: snake_case
user_data = get_user_information()
breach_results = search_breach_database()

# Classes: PascalCase
class BreachDatabaseAnalyzer:
class CryptoTrackerService:

# Constants: UPPER_SNAKE_CASE
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 30

# Private attributes: leading underscore
class Service:
    def __init__(self):
        self._connection = None
        self.__private_key = None
```

### Type Hints (Required)
```python
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Function signatures
def process_osint_data(
    data: List[Dict[str, Any]], 
    filters: Optional[Dict[str, str]] = None
) -> Dict[str, Union[str, int]]:
    """Process OSINT data with optional filtering."""
    pass

# Class attributes
class Analyzer:
    results: List[Dict[str, Any]]
    timeout: int = 30
    
# Async functions
async def fetch_breach_data(email: str) -> Optional[Dict[str, Any]]:
    """Fetch breach data asynchronously."""
    pass
```

### Docstrings (Required for Public APIs)
```python
def analyze_cryptocurrency_address(address: str, network: str = "bitcoin") -> Dict[str, Any]:
    """
    Analyze cryptocurrency address for transaction patterns.
    
    Args:
        address: The cryptocurrency address to analyze
        network: Blockchain network (bitcoin, ethereum, etc.)
        
    Returns:
        Dictionary containing analysis results with keys:
        - 'transactions': List of transaction data
        - 'risk_score': Risk assessment (0-100)
        - 'exchange_deposits': Known exchange interactions
        
    Raises:
        ValueError: If address format is invalid
        ConnectionError: If blockchain API is unavailable
        
    Example:
        >>> result = analyze_cryptocurrency_address("1A2B3C...")
        >>> print(result['risk_score'])
        85
    """
    pass
```

### Error Handling
```python
# Specific exceptions
class OSINTAnalysisError(Exception):
    """Base exception for OSINT analysis errors."""
    pass

class BreachDatabaseConnectionError(OSINTAnalysisError):
    """Raised when breach database connection fails."""
    pass

# Proper exception handling
try:
    results = search_breach_database(email)
except BreachDatabaseConnectionError as e:
    logger.error(f"Database connection failed: {e}")
    raise
except ValueError as e:
    logger.warning(f"Invalid email format: {e}")
    return {"error": "invalid_email", "message": str(e)}
```

### Async/Await Patterns
```python
import asyncio
import aiohttp
from typing import List

async def fetch_multiple_sources(targets: List[str]) -> List[Dict[str, Any]]:
    """Fetch data from multiple OSINT sources concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_single_source(session, target) for target in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]

async def fetch_single_source(session: aiohttp.ClientSession, target: str) -> Dict[str, Any]:
    """Fetch data from a single OSINT source."""
    async with session.get(f"https://api.osint-source.com/query/{target}") as response:
        return await response.json()
```

## Project Structure Conventions

### File Organization
```python
# Module imports order (isort standard)
# 1. Standard library
import os
import json
from pathlib import Path
from typing import Dict, List

# 2. Third-party packages
import aiohttp
import pandas as pd
from pydantic import BaseModel

# 3. Local application imports
from src.analyzers.base import BaseAnalyzer
from src.utils.logging import get_logger
from .exceptions import BreachDatabaseError
```

### Class Structure
```python
class BreachDatabaseAnalyzer(BaseAnalyzer):
    """Analyzer for breach database searches."""
    
    # Class constants
    DEFAULT_TIMEOUT = 30
    MAX_RESULTS = 1000
    
    def __init__(self, api_key: str, timeout: int = DEFAULT_TIMEOUT):
        """Initialize the breach database analyzer."""
        super().__init__()
        self._api_key = api_key
        self._timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def analyze(self, target: str) -> Dict[str, Any]:
        """Main analysis method."""
        pass
        
    async def _fetch_data(self, endpoint: str) -> Dict[str, Any]:
        """Private method for data fetching."""
        pass
        
    def __repr__(self) -> str:
        return f"BreachDatabaseAnalyzer(timeout={self._timeout})"
```

### Configuration Management
```python
from pydantic import BaseSettings, Field
from typing import Optional

class OSINTConfig(BaseSettings):
    """Configuration for OSINT framework."""
    
    # Database configuration
    postgres_host: str = Field(default="localhost", env="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="POSTGRES_PORT")
    postgres_user: str = Field(env="POSTGRES_USER")
    postgres_password: str = Field(env="POSTGRES_PASSWORD")
    
    # API keys
    dehashed_api_key: Optional[str] = Field(env="DEHASHED_API_KEY")
    snusbase_api_key: Optional[str] = Field(env="SNUSBASE_API_KEY")
    
    # Tor configuration
    tor_proxy: str = Field(default="socks5://localhost:9050", env="TOR_PROXY")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

## Testing Conventions

### Test Structure
```python
import pytest
from unittest.mock import AsyncMock, patch
from src.analyzers.breach_database import BreachDatabaseAnalyzer

class TestBreachDatabaseAnalyzer:
    """Test suite for BreachDatabaseAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Analyzer fixture for testing."""
        return BreachDatabaseAnalyzer(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_analyze_email_success(self, analyzer):
        """Test successful email analysis."""
        # Given
        email = "test@example.com"
        expected_result = {"breaches": [], "risk_score": 0}
        
        # When
        with patch.object(analyzer, '_fetch_data', return_value=expected_result):
            result = await analyzer.analyze(email)
        
        # Then
        assert result == expected_result
        assert result["risk_score"] >= 0
    
    @pytest.mark.asyncio
    async def test_analyze_invalid_email(self, analyzer):
        """Test analysis with invalid email."""
        with pytest.raises(ValueError, match="Invalid email format"):
            await analyzer.analyze("invalid-email")
```

### Test Markers
```python
# Performance tests
@pytest.mark.performance
def test_high_load_processing():
    pass

# Integration tests
@pytest.mark.integration
def test_database_connectivity():
    pass

# Slow tests
@pytest.mark.slow
def test_comprehensive_analysis():
    pass

# Security tests
@pytest.mark.security
def test_input_sanitization():
    pass
```

## Logging Conventions

### Logging Setup
```python
import logging
from typing import Dict, Any

# Configure logger
logger = logging.getLogger(__name__)

class OSINTAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def analyze(self, target: str) -> Dict[str, Any]:
        """Analyze target with comprehensive logging."""
        self.logger.info(f"Starting analysis for target: {target}")
        
        try:
            results = await self._perform_analysis(target)
            self.logger.info(f"Analysis completed successfully. Found {len(results)} items")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {target}: {e}", exc_info=True)
            raise
```

## Security Conventions

### Input Validation
```python
import re
from typing import Union

def validate_email(email: str) -> bool:
    """Validate email format for OSINT analysis."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def sanitize_input(user_input: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`']
    sanitized = user_input
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    return sanitized.strip()
```

### API Key Management
```python
import os
from typing import Optional

def get_api_key(service_name: str) -> Optional[str]:
    """Safely retrieve API key from environment."""
    key = os.getenv(f"{service_name.upper()}_API_KEY")
    if not key:
        logger.warning(f"API key not found for {service_name}")
    return key

# Never log API keys
def log_request_info(url: str, headers: Dict[str, str]):
    """Log request information without sensitive data."""
    safe_headers = {k: v for k, v in headers.items() if 'key' not in k.lower()}
    logger.info(f"Making request to {url} with headers: {safe_headers}")
```

## Performance Conventions

### Async Best Practices
```python
import asyncio
from contextlib import asynccontextmanager

class AnalyzerPool:
    """Pool of analyzers for concurrent processing."""
    
    def __init__(self, max_concurrent: int = 10):
        self._semaphore = asyncio.Semaphore(max_concurrent)
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire analyzer with rate limiting."""
        async with self._semaphore:
            yield
    
    async def analyze_batch(self, targets: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple targets concurrently."""
        tasks = []
        for target in targets:
            task = self._analyze_single(target)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
```

## Documentation Conventions

### README Structure
- Clear project description and purpose
- Quick start guide with essential commands
- Architecture overview
- Security considerations
- Troubleshooting section

### Code Comments
```python
# Good: Explain why, not what
# Use exponential backoff to handle rate limiting from API
retry_delay = min(2 ** attempt, 60)

# Bad: Explain obvious operations
# Set retry_delay to 2 to the power of attempt
retry_delay = 2 ** attempt
```