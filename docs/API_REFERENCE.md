# BEV OSINT Framework - API Reference

## 🚀 Overview

The BEV OSINT Framework provides a comprehensive REST API and WebSocket interface for intelligence gathering, analysis, and visualization. The API is built on FastAPI with async support and includes integrated security, authentication, and monitoring.

## 📡 Server Endpoints

### Base URL
```
http://localhost:3010
```

### Authentication
All API endpoints require bearer token authentication:
```http
Authorization: Bearer <your_token_here>
```

## 🔗 Core API Endpoints

### Health & Monitoring

#### GET /health
**Description**: System health check endpoint
**Authentication**: Required
**Response**:
```json
{
  "status": "healthy|unhealthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "services": {
    "postgres": true,
    "redis": true,
    "neo4j": true,
    "elasticsearch": true
  }
}
```

#### GET /metrics
**Description**: Prometheus metrics endpoint
**Authentication**: Required
**Response**: Prometheus format metrics

#### GET /info
**Description**: Server information and capabilities
**Authentication**: Required
**Response**:
```json
{
  "name": "BEV OSINT MCP Server",
  "version": "1.0.0",
  "protocol_version": "2024-11-05",
  "capabilities": {
    "tools": true,
    "resources": true,
    "prompts": true,
    "logging": true
  }
}
```

## 🧰 OSINT Tools API

### Tool Registry

The framework provides specialized OSINT tools accessible through the MCP protocol:

#### 1. OSINT Collector Tool
**Purpose**: General-purpose OSINT data collection
**Parameters**:
- `target` (string): Target identifier (email, IP, domain, etc.)
- `sources` (array): Data sources to query
- `depth` (integer): Analysis depth level (1-5)

**Example Usage**:
```json
{
  "method": "tools/call",
  "params": {
    "name": "osint_collector",
    "arguments": {
      "target": "example@domain.com",
      "sources": ["breaches", "social", "whois"],
      "depth": 3
    }
  }
}
```

#### 2. Threat Analyzer Tool
**Purpose**: Advanced threat analysis and risk assessment
**Parameters**:
- `indicators` (array): IOCs to analyze
- `threat_types` (array): Threat categories to focus on
- `confidence_threshold` (number): Minimum confidence level

**Example Usage**:
```json
{
  "method": "tools/call",
  "params": {
    "name": "threat_analyzer",
    "arguments": {
      "indicators": ["192.168.1.1", "suspicious.domain.com"],
      "threat_types": ["malware", "phishing", "botnet"],
      "confidence_threshold": 0.7
    }
  }
}
```

#### 3. Graph Analyzer Tool
**Purpose**: Relationship analysis and network mapping
**Parameters**:
- `entities` (array): Entities to analyze
- `relationship_types` (array): Types of relationships to explore
- `max_depth` (integer): Maximum graph traversal depth

**Example Usage**:
```json
{
  "method": "tools/call",
  "params": {
    "name": "graph_analyzer",
    "arguments": {
      "entities": ["entity1", "entity2"],
      "relationship_types": ["associates", "communicates", "owns"],
      "max_depth": 3
    }
  }
}
```

## 🌐 WebSocket API

### Connection Endpoint
```
ws://localhost:3010/mcp
```

### Authentication Flow
1. **Connect**: Establish WebSocket connection
2. **Authenticate**: Send authentication message
3. **Operate**: Use MCP protocol for tool operations

#### Authentication Message
```json
{
  "token": "your_bearer_token_here"
}
```

#### Success Response
```json
{
  "id": "auth_success",
  "result": {
    "authenticated": true,
    "client_id": "unique_client_identifier"
  }
}
```

### MCP Protocol Messages

#### Tool List Request
```json
{
  "id": "req_001",
  "method": "tools/list",
  "params": {}
}
```

#### Tool List Response
```json
{
  "id": "req_001",
  "result": {
    "tools": [
      {
        "name": "osint_collector",
        "description": "General-purpose OSINT data collection",
        "inputSchema": {
          "type": "object",
          "properties": {
            "target": {"type": "string"},
            "sources": {"type": "array"},
            "depth": {"type": "integer"}
          },
          "required": ["target"]
        }
      }
    ]
  }
}
```

#### Tool Execution Request
```json
{
  "id": "req_002",
  "method": "tools/call",
  "params": {
    "name": "osint_collector",
    "arguments": {
      "target": "example@domain.com",
      "sources": ["breaches", "social"],
      "depth": 2
    }
  }
}
```

#### Tool Execution Response
```json
{
  "id": "req_002",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "OSINT analysis completed successfully"
      },
      {
        "type": "resource",
        "resource": {
          "uri": "bev://analysis/12345",
          "name": "Analysis Results",
          "description": "Complete OSINT analysis for example@domain.com"
        }
      }
    ]
  }
}
```

## 🔧 IntelOwl Integration API

### Custom Analyzers

#### Breach Database Analyzer
**Endpoint**: Via IntelOwl job creation
**Purpose**: Search multiple breach databases
**Supported Sources**:
- Dehashed
- Snusbase
- WeLeakInfo
- HaveIBeenPwned

**Configuration**:
```json
{
  "analyzer": "BreachDatabaseAnalyzer",
  "configuration": {
    "sources": ["dehashed", "snusbase"],
    "include_passwords": false,
    "risk_scoring": true
  }
}
```

#### Darknet Market Analyzer
**Endpoint**: Via IntelOwl job creation
**Purpose**: Scrape darknet marketplaces through Tor
**Supported Markets**:
- AlphaBay
- White House Market
- Torrez Market

**Configuration**:
```json
{
  "analyzer": "DarknetMarketAnalyzer",
  "configuration": {
    "markets": ["alphabay", "whitehouse"],
    "vendor_profiling": true,
    "product_tracking": true
  }
}
```

#### Cryptocurrency Tracker
**Endpoint**: Via IntelOwl job creation
**Purpose**: Blockchain analysis and transaction tracking
**Supported Networks**:
- Bitcoin
- Ethereum
- Monero (limited)

**Configuration**:
```json
{
  "analyzer": "CryptoTrackerAnalyzer",
  "configuration": {
    "networks": ["bitcoin", "ethereum"],
    "transaction_depth": 5,
    "mixer_detection": true
  }
}
```

#### Social Media Analyzer
**Endpoint**: Via IntelOwl job creation
**Purpose**: Social media profile analysis
**Supported Platforms**:
- Instagram
- Twitter/X
- LinkedIn
- Facebook (limited)

**Configuration**:
```json
{
  "analyzer": "SocialMediaAnalyzer",
  "configuration": {
    "platforms": ["instagram", "twitter"],
    "network_analysis": true,
    "sentiment_analysis": false
  }
}
```

## 📊 Database APIs

### PostgreSQL Database
**Connection**: `postgresql://bev:BevOSINT2024@localhost:5432/osint`
**Purpose**: Primary data storage with vector search capabilities

#### Key Tables:
- `osint_results` - Analysis results
- `threat_indicators` - IOC storage
- `user_sessions` - Session management
- `api_keys` - Authentication tokens

### Neo4j Graph Database
**Connection**: `bolt://localhost:7687`
**Credentials**: `neo4j/BevGraphMaster2024`
**Purpose**: Relationship mapping and graph analysis

#### Key Node Types:
- `Entity` - OSINT entities
- `Threat` - Threat indicators
- `Analysis` - Analysis sessions
- `Source` - Data sources

#### Key Relationships:
- `RELATES_TO` - General relationships
- `COMMUNICATES_WITH` - Communication patterns
- `ASSOCIATES_WITH` - Association patterns
- `THREATENS` - Threat relationships

### Redis Cache
**Connection**: `redis://:BevCacheMaster@localhost:6379`
**Purpose**: Session storage, caching, and rate limiting

## 📈 Monitoring & Metrics

### Prometheus Metrics

#### Server Metrics
- `bev_request_count` - Total API requests
- `bev_request_duration` - Request duration histogram
- `bev_active_connections` - Active WebSocket connections
- `bev_tool_executions` - Tool execution counter
- `bev_error_count` - Error counter by type

#### Custom OSINT Metrics
- `bev_osint_analyses_total` - Total OSINT analyses
- `bev_threat_detections` - Threat detection counter
- `bev_data_sources_queried` - Data source query counter
- `bev_cache_hit_rate` - Cache efficiency metrics

### Grafana Dashboards
Access: `http://localhost:3000`
Default credentials: `admin/admin`

Available dashboards:
- **System Overview** - Overall system health
- **OSINT Operations** - Analysis performance metrics
- **Security Dashboard** - Threat detection and security events
- **Performance Monitor** - System performance metrics

## 🔒 Security Considerations

### Rate Limiting
- **Global Rate Limit**: 1000 requests/hour per IP
- **Tool Execution Limit**: 100 executions/hour per client
- **WebSocket Connections**: 10 concurrent per client

### Input Validation
- All inputs sanitized and validated
- SQL injection protection via parameterized queries
- XSS prevention through output encoding
- File upload restrictions and scanning

### Data Protection
- Sensitive data encrypted at rest
- API tokens hashed and salted
- PII data retention policies enforced
- GDPR compliance measures implemented

## 📚 SDK & Client Libraries

### Python Client Example
```python
import asyncio
import websockets
import json

async def connect_to_bev():
    uri = "ws://localhost:3010/mcp"

    async with websockets.connect(uri) as websocket:
        # Authenticate
        auth_msg = {"token": "your_token_here"}
        await websocket.send(json.dumps(auth_msg))

        # Receive auth response
        response = await websocket.recv()
        print(f"Auth response: {response}")

        # List available tools
        list_msg = {
            "id": "req_001",
            "method": "tools/list",
            "params": {}
        }
        await websocket.send(json.dumps(list_msg))

        # Receive tools list
        tools_response = await websocket.recv()
        print(f"Available tools: {tools_response}")

# Run the client
asyncio.run(connect_to_bev())
```

### JavaScript Client Example
```javascript
const ws = new WebSocket('ws://localhost:3010/mcp');

ws.onopen = function() {
    // Authenticate
    ws.send(JSON.stringify({
        token: 'your_token_here'
    }));
};

ws.onmessage = function(event) {
    const response = JSON.parse(event.data);
    console.log('Received:', response);

    if (response.authenticated) {
        // Execute OSINT tool
        ws.send(JSON.stringify({
            id: 'req_001',
            method: 'tools/call',
            params: {
                name: 'osint_collector',
                arguments: {
                    target: 'example@domain.com',
                    sources: ['breaches', 'social'],
                    depth: 2
                }
            }
        }));
    }
};
```

## 🚨 Error Handling

### Standard Error Codes
- `4001` - Authentication required
- `4003` - Authentication failed
- `4004` - Authorization failed
- `4005` - Rate limit exceeded
- `5000` - Internal server error
- `5001` - Database connection error
- `5002` - External service unavailable

### Error Response Format
```json
{
  "error": {
    "code": -32603,
    "message": "Internal error",
    "data": {
      "type": "DatabaseConnectionError",
      "details": "PostgreSQL connection failed"
    }
  }
}
```

## 📞 Support & Resources

### Documentation
- **Architecture Overview**: `/docs/ARCHITECTURE.md`
- **Security Guide**: `/docs/SECURITY.md`
- **Deployment Guide**: `/docs/DEPLOYMENT.md`

### Community
- **Issues**: Report bugs and feature requests
- **Discussions**: Community support and questions
- **Contributing**: Contribution guidelines and development setup

### Contact
- **Security Issues**: Report security vulnerabilities responsibly
- **Technical Support**: Enterprise support and consulting available

---

**⚠️ Important**: This API is designed for authorized security research and OSINT operations only. Use responsibly and in compliance with applicable laws and regulations.