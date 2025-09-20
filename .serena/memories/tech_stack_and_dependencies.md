# BEV OSINT Framework - Technology Stack

## Core Technologies
- **Python 3.13** - Primary programming language
- **Docker & Docker Compose** - Containerization and orchestration
- **FastAPI** - Modern async web framework
- **PostgreSQL with pgvector** - Primary database with vector search
- **Neo4j** - Graph database for relationship analysis
- **Redis** - Caching and session storage
- **Elasticsearch** - Search and analytics engine

## Key Python Dependencies
### Web Framework & API
- FastAPI 0.105.0 - High-performance async API framework
- Uvicorn 0.24.0 - ASGI server
- Pydantic 2.5.0 - Data validation and settings

### Async & Networking
- aiohttp 3.9.1 - Async HTTP client/server
- asyncpg 0.29.0 - Async PostgreSQL driver
- httpx 0.25.2 - Modern HTTP client

### Data Processing & ML
- pandas 2.1.3 - Data manipulation
- numpy 1.24.3 - Numerical computing
- scikit-learn 1.3.2 - Machine learning
- transformers 4.36.0 - NLP and language models
- sentence-transformers 2.2.2 - Text embeddings

### Web Scraping & Research
- Beautiful Soup 4.12.2 - HTML parsing
- Selenium 4.16.0 - Browser automation
- Playwright 1.40.0 - Modern browser automation
- Scrapy 2.11.0 - Web crawling framework

### Security & Privacy
- PySocks 1.7.1 - SOCKS proxy support
- stem 1.8.2 - Tor control library
- cryptography 41.0.7 - Cryptographic primitives
- fake-useragent 1.4.0 - User agent rotation

### Databases & Vector Storage
- neo4j 5.15.0 - Graph database driver
- qdrant-client 1.7.0 - Vector database client
- weaviate-client 4.4.0 - Vector search engine
- influxdb-client 1.38.0 - Time-series database

### Message Queues & Orchestration
- celery 5.3.4 - Distributed task queue
- pika 1.3.2 - RabbitMQ client
- apache-airflow 2.8.0 - Workflow orchestration

### Testing & Quality
- pytest 7.4.3 - Testing framework
- pytest-asyncio 0.21.1 - Async testing
- pytest-cov 4.1.0 - Coverage reporting
- black 23.12.0 - Code formatting
- mypy 1.7.1 - Static type checking
- flake8 6.1.0 - Linting

## Infrastructure Components
- **IntelOwl** - OSINT analysis platform
- **Cytoscape.js** - Graph visualization
- **Tor** - Anonymization proxy
- **Prometheus** - Metrics collection
- **Grafana** - Monitoring dashboards
- **RabbitMQ** - Message broker
- **Apache Kafka** - Stream processing

## Development Environment
- **Ubuntu/Linux** - Primary OS
- **Docker** - Containerization
- **Git** - Version control
- **VS Code/IDE** - Development environment