# BEV OSINT Framework - MCP Server

A production-ready Model Context Protocol (MCP) server implementation for the BEV OSINT Framework with comprehensive security features and OSINT tool integration.

## 🔒 Security Features

- **JWT-based Authentication** with secure token validation and blacklisting
- **Input Validation & Sanitization** with injection attack prevention
- **Rate Limiting** (100 requests/minute per client by default)
- **SQL Injection Prevention** with parameterized queries
- **Command Injection Protection** with command whitelisting
- **Network Access Control** with IP allowlisting
- **Comprehensive Audit Logging** with 90-day retention
- **Encrypted Credential Management** using Fernet encryption

## 🚀 Performance Features

- **Async/Await Operations** for non-blocking I/O
- **Database Connection Pooling** for PostgreSQL, Neo4j, Redis, Elasticsearch
- **Redis Caching** for frequently accessed data
- **Background Task Processing** for heavy operations
- **WebSocket Support** for real-time communication
- **Load Balancing Support** (primary/replica configuration)

## 🛠 OSINT Tools

The server implements 8 specialized OSINT tools:

1. **collect_osint** - Multi-source data gathering with Tor support
2. **analyze_threat** - IOC analysis with ML-based classification
3. **graph_analysis** - Neo4j-based relationship mapping
4. **coordinate_agents** - Multi-agent task orchestration
5. **monitor_targets** - Real-time surveillance with alerts
6. **crawl_darkweb** - Tor-based market intelligence
7. **analyze_crypto** - Blockchain transaction analysis
8. **security_scan** - Vulnerability assessment tools

## 🏗 Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Protocol  │    │  Security Mgr   │    │  Tool Registry  │
│     Handler     │◄──►│                 │◄──►│                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WebSocket     │    │   Auth & Rate   │    │   OSINT Tools   │
│   Connection    │    │    Limiting     │    │   (8 tools)     │
│    Manager      │    └─────────────────┘    └─────────────────┘
└─────────────────┘             │                       │
         │                      ▼                       ▼
         ▼              ┌─────────────────┐    ┌─────────────────┐
┌─────────────────┐    │  Audit Logger   │    │   Performance   │
│   Background    │    │                 │    │   Monitoring    │
│     Tasks       │    └─────────────────┘    └─────────────────┘
└─────────────────┘             │                       │
         │                      ▼                       ▼
         └──────────────┐ ┌─────────────────┐ ┌─────────────────┐
                        ▼ │                 │ │                 │
                 ┌─────────────────────────────────────────────┐
                 │            Database Manager                 │
                 │  PostgreSQL │ Neo4j │ Redis │ Elasticsearch │
                 └─────────────────────────────────────────────┘
```

### Database Integration

- **PostgreSQL**: Structured data storage (OSINT results, threat intelligence, audit logs)
- **Neo4j**: Graph-based relationship mapping and analysis
- **Redis**: Caching, session management, and rate limiting
- **Elasticsearch**: Search and analytics for large datasets

## 🚢 Quick Start

### Prerequisites

- Python 3.12+
- Docker and Docker Compose
- Access to BEV infrastructure databases

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd bev-osint-framework/src/mcp_server
```

2. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run with Docker Compose:**
```bash
docker-compose up -d
```

### Configuration

Key environment variables:

```bash
# Server Configuration
MCP_SERVER_HOST=0.0.0.0
MCP_SERVER_PORT=3010  # Changed from 3000 to avoid Grafana conflict

# Security
JWT_SECRET=your_jwt_secret_key
DATA_ENCRYPTION_KEY=your_encryption_key
API_RATE_LIMIT=100
ALLOWED_NETWORKS=172.30.0.0/16,127.0.0.1/32

# Databases (BEV Infrastructure IPs)
POSTGRES_URI=postgresql://bev_admin:password@172.21.0.2:5432/osint
NEO4J_URI=bolt://172.21.0.3:7687
REDIS_HOST=172.21.0.4
ELASTICSEARCH_HOST=172.21.0.5
```

## 📡 API Usage

### WebSocket Connection

Connect to the MCP server via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:3010/mcp');

// Send authentication
ws.send(JSON.stringify({
    token: "your_jwt_token"
}));

// Send MCP messages
ws.send(JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "tools/list"
}));
```

### HTTP Endpoints

- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /info` - Server information
- `GET /docs` - API documentation

### Tool Execution

Example tool call via MCP protocol:

```json
{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "collect_osint",
        "arguments": {
            "target_type": "domain",
            "target_value": "example.com",
            "sources": "whois,virustotal,shodan",
            "use_tor": false,
            "max_results": 100
        }
    }
}
```

## 🐳 Deployment

### Docker Deployment

The server includes a multi-stage Dockerfile for production deployment:

```bash
# Build the image
docker build -t bev-mcp-server .

# Run with docker-compose
docker-compose up -d
```

### Integration with BEV Infrastructure

The server is designed to integrate with the existing BEV OSINT infrastructure:

1. **Network Configuration**: Uses BEV network (172.30.0.0/16)
2. **Database Connections**: Connects to existing BEV databases
3. **Service Discovery**: Integrates with BEV service mesh
4. **Monitoring**: Prometheus metrics for Grafana dashboards

### Health Checks

The server includes comprehensive health checks:

- Database connectivity (PostgreSQL, Neo4j, Redis, Elasticsearch)
- Memory and CPU usage monitoring
- Active connection tracking
- Background task status

## 🔧 Development

### Running Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=mcp_server --cov-report=html

# Run specific test categories
pytest -m security  # Security tests only
pytest -m integration  # Integration tests only
```

### Code Quality

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Adding New Tools

1. Create a new tool class inheriting from `OSINTToolBase`
2. Implement required methods: `execute()` and `get_tool_definition()`
3. Add to `OSINTToolRegistry._initialize_tools()`
4. Write comprehensive tests

Example:

```python
class NewOSINTTool(OSINTToolBase):
    def __init__(self, db_manager, security_manager):
        super().__init__(
            name="new_tool",
            description="Description of new tool",
            category=ToolCategory.ANALYSIS,
            security_level=SecurityLevel.MEDIUM,
            db_manager=db_manager,
            security_manager=security_manager
        )
    
    async def execute(self, target: OSINTTarget, params: Dict[str, Any]) -> ToolResult:
        # Implementation here
        pass
    
    def get_tool_definition(self) -> ToolDefinition:
        # Tool definition here
        pass
```

## 📊 Monitoring

### Prometheus Metrics

The server exposes various metrics:

- `mcp_requests_total` - Total MCP requests by method and status
- `mcp_request_duration_seconds` - Request duration histogram
- `mcp_active_connections` - Active WebSocket connections
- `mcp_tool_executions_total` - Tool executions by tool and status
- `mcp_errors_total` - Error counts by type

### Grafana Dashboard

Use the included Grafana configuration for monitoring:

- Server performance metrics
- Database connection health
- Tool execution statistics
- Security events and rate limiting
- Error rates and response times

## 🔐 Security Considerations

### Production Deployment

1. **Change Default Credentials**: Update all default passwords and secrets
2. **Network Isolation**: Deploy in isolated network with proper firewall rules
3. **TLS/SSL**: Enable HTTPS/WSS for all communications
4. **Regular Updates**: Keep dependencies and base images updated
5. **Log Monitoring**: Monitor audit logs for suspicious activity
6. **Backup Strategy**: Regular backups of database and configuration

### Security Hardening

- Run containers as non-root user
- Use read-only file systems where possible
- Implement proper secret management
- Regular security scanning of images
- Network segmentation and access controls

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**: Check if databases are running and accessible
2. **Authentication Failed**: Verify JWT secret and token format
3. **Rate Limit Exceeded**: Check client request patterns
4. **Tool Execution Failed**: Review tool parameters and logs

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m mcp_server.main
```

### Health Check

Check server health:

```bash
curl http://localhost:3010/health
```

## 📝 License

This project is part of the BEV OSINT Framework and follows the same licensing terms.

## 🤝 Contributing

1. Follow existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Follow security best practices
5. Submit pull requests with detailed descriptions

## 📞 Support

For issues and questions:

1. Check existing documentation and troubleshooting
2. Review logs for error details
3. Submit issues with detailed reproduction steps
4. Include relevant configuration and environment details