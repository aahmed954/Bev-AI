# BEV OSINT Framework - Project Structure

## Root Directory Structure
```
Bev/
├── src/                          # Source code modules
├── intelowl/                     # IntelOwl integration
├── tests/                        # Comprehensive testing suite
├── dags/                         # Airflow DAG definitions
├── docker/                       # Docker configurations
├── scripts/                      # Utility and deployment scripts
├── config/                       # Configuration files
├── docs/                         # Documentation
├── frontend/                     # Frontend components
├── deployment/                   # Deployment configurations
├── requirements/                 # Python dependencies
├── data/                         # Data storage
├── logs/                         # Log files
├── backups/                      # Backup storage
└── etc/                          # Additional configurations
```

## Source Code Organization (`src/`)
```
src/
├── pipeline/                     # Data processing pipelines
├── security/                     # Security modules
├── agents/                       # AI agents and automation
├── enhancement/                  # System enhancements
├── autonomous/                   # Autonomous research components
├── advanced/                     # Advanced analytics
├── oracle/                       # Oracle database integration
├── alternative_market/           # Market intelligence
├── infrastructure/               # Infrastructure management
├── monitoring/                   # Monitoring and metrics
├── edge/                         # Edge computing
├── testing/                      # Testing utilities
└── mcp_server/                   # MCP server implementation
```

## IntelOwl Integration (`intelowl/`)
```
intelowl/
├── custom_analyzers/             # Custom OSINT analyzers
│   ├── BreachDatabaseAnalyzer/   # Breach database searches
│   ├── DarknetMarketAnalyzer/    # Darknet market scraping
│   ├── CryptoTrackerAnalyzer/    # Cryptocurrency analysis
│   └── SocialMediaAnalyzer/      # Social media OSINT
└── custom_connectors/            # Data connectors
```

## Testing Structure (`tests/`)
```
tests/
├── integration/                  # Service connectivity tests
├── performance/                  # Performance and load tests
├── resilience/                   # Chaos engineering tests
├── end_to_end/                   # Complete workflow tests
├── security/                     # Security validation tests
├── vector_db/                    # Vector database tests
├── cache/                        # Cache performance tests
├── edge/                         # Edge computing tests
├── monitoring/                   # Monitoring integration tests
├── chaos/                        # Chaos engineering
├── conftest.py                   # Pytest configuration
├── test_runner.py                # Automated test execution
├── validate_system.py            # System validation
└── requirements.txt              # Testing dependencies
```

## Key Configuration Files
- **docker-compose.complete.yml** - Complete system orchestration
- **requirements.txt** - Python dependencies
- **.env** - Environment configuration
- **README.md** - Project documentation
- **run_all_tests.sh** - Master test runner

## Deployment Files
- **deploy_everything.sh** - Complete deployment script
- **validate_bev_deployment.sh** - Deployment validation
- **generate_secrets.sh** - Security setup
- **verify_deployment.sh** - Deployment verification

## Documentation Structure (`docs/`)
- Architecture overviews and technical documentation
- User guides and operational manuals
- API reference and integration guides
- Deployment and configuration documentation

## Data Organization (`data/`)
- OSINT analysis results
- Graph database exports
- Cache and temporary files
- Backup and archival data

## Scripts Organization (`scripts/`)
- Health check scripts
- Maintenance utilities
- Deployment automation
- System administration tools

## Frontend Structure (`frontend/`)
- Web interface components
- Visualization dashboards
- User experience enhancements
- Static assets and themes

## File Naming Conventions
- **Configuration files**: lowercase with underscores (docker-compose.yml)
- **Python modules**: lowercase with underscores (custom_analyzers)
- **Shell scripts**: descriptive names with .sh extension
- **Documentation**: uppercase for main files (README.md)
- **Test files**: test_ prefix for Python tests

## Important Files to Know
- **docker-compose.complete.yml** - Full system deployment
- **run_all_tests.sh** - Complete test execution
- **validate_bev_deployment.sh** - System validation
- **requirements.txt** - Python dependencies
- **.env** - Environment configuration
- **tests/conftest.py** - Testing configuration