# BEV Proxy Management Infrastructure

Comprehensive proxy pool management system with geographic distribution, health checking, and automatic failover for the BEV OSINT framework.

## Features

### Core Capabilities
- **10K+ Proxy Pool Support**: Manages residential, datacenter, rotating, and Tor proxy pools
- **Geographic Distribution**: US-East, US-West, EU-Central, Asia-Pacific regions with intelligent routing
- **Health Checking**: 30-second health check intervals with automatic failover
- **Load Balancing**: Multiple strategies including round-robin, least-connections, weighted, geographic
- **Tor Integration**: Full integration with existing Tor infrastructure including circuit management
- **Performance Monitoring**: Real-time metrics, Prometheus integration, comprehensive statistics

### Advanced Features
- **Stream Isolation**: Tor stream isolation for enhanced anonymity
- **Circuit Rotation**: Automatic Tor circuit rotation and management
- **Compliance Support**: GDPR, CCPA, PDPA compliance with regional data sovereignty
- **Provider Integration**: Support for major proxy providers (Bright Data, SmartProxy, Oxylabs)
- **Geographic Optimization**: Latency-based routing and target-specific region selection
- **Security Controls**: Request sanitization, SSL verification, privacy protection

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BEV Proxy Manager                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │ Proxy Pool      │ │ Geographic      │ │ Health Checker  │ │
│  │ Manager         │ │ Router          │ │                 │ │
│  │                 │ │                 │ │                 │ │
│  │ • 10K+ proxies  │ │ • GeoIP lookup  │ │ • 30s intervals │ │
│  │ • Multiple types│ │ • Region optim  │ │ • Auto failover │ │
│  │ • Load balancing│ │ • Compliance    │ │ • Health metrics│ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ │
│  │ Tor Integration │ │ Provider APIs   │ │ Monitoring      │ │
│  │                 │ │                 │ │                 │ │
│  │ • Circuit mgmt  │ │ • BrightData    │ │ • Prometheus    │ │
│  │ • Stream isolat │ │ • SmartProxy    │ │ • Real-time     │ │
│  │ • Control port  │ │ • Oxylabs       │ │ • Alerting      │ │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup

Set required environment variables:

```bash
# Database connections
export POSTGRES_URI="postgresql://user:pass@postgres:5432/osint"
export REDIS_URL="redis://:password@redis:6379/11"

# Tor configuration
export TOR_PROXY="socks5://tor:9050"
export TOR_CONTROL_PASSWORD="BevTorControl2024"

# Proxy providers (optional)
export PROXY_PROVIDERS="brightdata,smartproxy,oxylabs"
export PROVIDER_API_KEYS="brightdata:key1,smartproxy:key2"

# Geographic configuration
export DEFAULT_REGIONS="us-east,us-west,eu-central"
export GEOIP_DB_PATH="/app/data/GeoLite2-City.mmdb"
```

### 2. Start the Service

Using Docker Compose:
```bash
# Start proxy manager service
docker-compose up proxy-manager

# Check health
curl http://localhost:8013/health
```

### 3. Basic Usage

#### Get a Proxy
```bash
# Get optimal proxy for a target
curl -X POST http://localhost:8013/proxy/get \
  -H "Content-Type: application/json" \
  -d '{
    "target": "facebook.com",
    "operation_type": "osint",
    "region_preference": "us-east"
  }'
```

#### Release a Proxy
```bash
# Release proxy after use
curl -X POST http://localhost:8013/proxy/release \
  -H "Content-Type: application/json" \
  -d '{
    "proxy_id": "proxy1.example.com:8080",
    "success": true,
    "response_time": 0.5
  }'
```

## API Reference

### Endpoints

#### Health Check
```http
GET /health
```
Returns service health status and proxy pool statistics.

#### Get Proxy
```http
POST /proxy/get
```

Request body:
```json
{
  "target": "example.com",           // Optional: target for geo optimization
  "region_preference": "us-east",    // Optional: preferred region
  "proxy_type_preference": "residential", // Optional: proxy type
  "operation_type": "osint",         // Operation type for optimization
  "load_balance_strategy": "least_connections" // Optional: LB strategy
}
```

Response:
```json
{
  "proxy_id": "proxy1.example.com:8080",
  "host": "proxy1.example.com",
  "port": 8080,
  "proxy_url": "socks5://user:pass@proxy1.example.com:8080",
  "http_proxy_url": "http://user:pass@proxy1.example.com:8080",
  "region": "us-east",
  "proxy_type": "residential",
  "weight": 1.0,
  "response_time": 0.25,
  "utilization": 45.2
}
```

#### Release Proxy
```http
POST /proxy/release
```

Request body:
```json
{
  "proxy_id": "proxy1.example.com:8080",
  "success": true,                   // Was the request successful?
  "response_time": 0.5              // Optional: response time in seconds
}
```

#### Add Proxy
```http
POST /proxy/add
```

Request body:
```json
{
  "host": "proxy1.example.com",
  "port": 8080,
  "username": "user",               // Optional
  "password": "pass",               // Optional
  "proxy_type": "residential",      // residential|datacenter|rotating|tor
  "region": "us-east",              // Region identifier
  "weight": 1.0,                    // Load balancing weight
  "max_connections": 100,           // Connection limit
  "rotation_interval": 1800,        // Optional: rotation interval in seconds
  "provider": "brightdata",         // Optional: provider name
  "cost_per_gb": 0.015             // Optional: cost information
}
```

#### Remove Proxy
```http
DELETE /proxy/{host}/{port}
```

#### Get Statistics
```http
GET /proxy/statistics
```

Returns comprehensive proxy pool and geographic statistics.

#### Get Optimal Regions
```http
GET /proxy/regions?target=example.com&operation_type=osint
```

Returns optimal proxy regions for a target based on geographic analysis.

#### Prometheus Metrics
```http
GET /metrics
```

Returns Prometheus-formatted metrics for monitoring.

## Configuration

### Main Configuration Files

- `config/default_proxies.json`: Default proxy configurations and templates
- `config/provider_config.json`: Proxy provider settings and API configurations
- `config/integration_config.json`: Service integration and runtime settings

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_URI` | PostgreSQL connection string | `postgresql://localhost:5432/osint` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/11` |
| `TOR_PROXY` | Tor SOCKS proxy URL | `socks5://localhost:9050` |
| `MAX_POOL_SIZE` | Maximum proxy pool size | `10000` |
| `HEALTH_CHECK_INTERVAL` | Health check interval (seconds) | `30` |
| `DEFAULT_STRATEGY` | Default load balancing strategy | `least_connections` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Proxy Types

- **residential**: High-anonymity residential IP addresses
- **datacenter**: Fast datacenter proxy servers
- **rotating**: Automatically rotating proxy endpoints
- **tor**: Tor network integration
- **mobile**: Mobile carrier IP addresses

### Regions

- **us-east**: US East Coast (Virginia, New York)
- **us-west**: US West Coast (California, Oregon)
- **eu-central**: Central Europe (Germany, Netherlands)
- **eu-west**: Western Europe (UK, France)
- **asia-pacific**: Asia Pacific (Singapore, Japan)
- **global**: Global/mixed regions

### Load Balancing Strategies

- **round_robin**: Simple round-robin distribution
- **least_connections**: Select proxy with fewest active connections
- **weighted_round_robin**: Weighted selection based on proxy performance
- **least_response_time**: Select proxy with best response time
- **geographic**: Geographic proximity-based selection
- **random**: Random proxy selection

## Integration Examples

### Python Integration

```python
import aiohttp
import asyncio

async def use_proxy_manager():
    # Get a proxy
    async with aiohttp.ClientSession() as session:
        async with session.post(
            'http://localhost:8013/proxy/get',
            json={
                'target': 'example.com',
                'operation_type': 'osint',
                'region_preference': 'us-east'
            }
        ) as response:
            proxy_info = await response.json()

    if not proxy_info:
        print("No proxy available")
        return

    # Use the proxy
    proxy_url = proxy_info['proxy_url']
    connector = aiohttp.ProxyConnector.from_url(proxy_url)

    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get('http://example.com') as response:
                content = await response.text()
                print(f"Response: {response.status}")

    except Exception as e:
        print(f"Request failed: {e}")
        success = False
    else:
        success = True

    # Release the proxy
    async with aiohttp.ClientSession() as session:
        await session.post(
            'http://localhost:8013/proxy/release',
            json={
                'proxy_id': proxy_info['proxy_id'],
                'success': success,
                'response_time': 1.5
            }
        )

# Run example
asyncio.run(use_proxy_manager())
```

### cURL Examples

```bash
# Get proxy for social media operation
curl -X POST http://localhost:8013/proxy/get \
  -H "Content-Type: application/json" \
  -d '{
    "target": "twitter.com",
    "operation_type": "social_media",
    "proxy_type_preference": "residential"
  }'

# Add a new proxy to the pool
curl -X POST http://localhost:8013/proxy/add \
  -H "Content-Type: application/json" \
  -d '{
    "host": "proxy.example.com",
    "port": 8080,
    "username": "user123",
    "password": "pass123",
    "proxy_type": "residential",
    "region": "us-west",
    "provider": "brightdata"
  }'

# Get comprehensive statistics
curl http://localhost:8013/proxy/statistics | jq '.'

# Get optimal regions for a target
curl "http://localhost:8013/proxy/regions?target=baidu.com&operation_type=osint"
```

## Monitoring

### Prometheus Metrics

The service exposes metrics on port 9090:

- `proxy_requests_total`: Total proxy requests by region, type, and status
- `proxy_response_time_seconds`: Proxy response time histogram
- `active_proxies_total`: Number of active proxies by region, type, and status
- `proxy_pool_size_total`: Total proxy pool size

### Health Monitoring

Health checks run every 30 seconds and test:
- Proxy connectivity and response time
- Geographic routing functionality
- Tor integration status
- Database and cache connectivity

### Alerting

Configure alerts for:
- Proxy failure rate > 10%
- Average response time > 5 seconds
- Pool utilization > 90%
- Tor circuit failures
- Geographic routing errors

## Security Considerations

### Privacy Protection
- Request sanitization and header removal
- User agent randomization
- DNS leak protection
- Traffic analysis resistance

### Compliance
- GDPR compliance for EU regions
- CCPA compliance for California
- Data localization enforcement
- Privacy-first design principles

### Access Control
- API key authentication (optional)
- Rate limiting and abuse prevention
- IP whitelisting support
- Audit logging

## Troubleshooting

### Common Issues

#### No Proxies Available
```bash
# Check proxy pool status
curl http://localhost:8013/proxy/statistics

# Check health status
curl http://localhost:8013/health
```

#### High Response Times
```bash
# Check regional latency
curl "http://localhost:8013/proxy/regions?target=example.com"

# Monitor metrics
curl http://localhost:8013/metrics | grep response_time
```

#### Tor Integration Issues
```bash
# Check Tor status in container
docker exec bev_proxy_manager curl -s http://localhost:8013/proxy/statistics | jq '.proxy_statistics.by_type.tor'

# Check Tor container logs
docker logs bev_tor
```

### Debug Mode

Enable debug logging:
```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Restart service
docker-compose restart proxy-manager
```

### Performance Tuning

Optimize for high throughput:
```json
{
  "MAX_CONCURRENT_CONNECTIONS": 2000,
  "HEALTH_CHECK_INTERVAL": 60,
  "CONNECTION_TIMEOUT": 10,
  "POOL_SIZE_LIMIT": 15000
}
```

## Contributing

### Development Setup

1. Clone repository and install dependencies
2. Set up local PostgreSQL and Redis instances
3. Configure environment variables
4. Run tests: `pytest tests/`
5. Start development server: `python app.py`

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write comprehensive docstrings
- Include unit tests for new features

## Support

For issues and questions:
- Check the troubleshooting section
- Review service logs: `docker logs bev_proxy_manager`
- Monitor metrics at `http://localhost:9090/metrics`

## License

Part of the BEV OSINT Framework - Internal Use Only