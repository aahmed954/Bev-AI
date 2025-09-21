"""
BEV OSINT Framework - Comprehensive Testing Suite

This testing framework validates all components of the BEV OSINT framework including:
- Integration tests for all 13+ services
- Performance tests for 1000+ concurrent requests
- Resilience tests with chaos engineering
- Vector database connectivity validation
- Cache performance and predictive caching
- Edge computing latency tests
- End-to-end workflow validation
- Health monitoring integration

Performance Targets:
- 1000+ concurrent requests
- <100ms latency
- >80% cache hit rates
- <5 minute chaos recovery
- 99.9% availability
"""

__version__ = "1.0.0"
__author__ = "BEV OSINT Testing Framework"

# Test Configuration
TEST_CONFIG = {
    "performance_targets": {
        "concurrent_requests": 1000,
        "max_latency_ms": 100,
        "cache_hit_rate": 0.80,
        "chaos_recovery_minutes": 5,
        "availability_target": 0.999
    },
    "test_environments": {
        "integration": "http://localhost",
        "performance": "http://localhost",
        "chaos": "http://localhost"
    },
    "timeouts": {
        "integration_test": 300,  # 5 minutes
        "performance_test": 1800,  # 30 minutes
        "chaos_test": 900,  # 15 minutes
        "end_to_end_test": 600  # 10 minutes
    }
}