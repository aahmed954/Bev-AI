# Bev OSINT Framework - Complete Deployment

## 🔥 Single-User OSINT Intelligence Platform

A fully integrated OSINT framework with IntelOwl + Cytoscape.js for darknet intelligence, breach monitoring, crypto tracking, and social media analysis.

### ⚡ Features

- **No Authentication** - Single user deployment with maximum performance
- **Dark Theme** - Hacker aesthetic throughout
- **Tor Integration** - Built-in SOCKS5 proxy with automatic circuit rotation
- **Custom Analyzers**:
  - Breach Database Search (Dehashed, Snusbase, WeLeakInfo)
  - Darknet Market Scraping via Tor
  - Cryptocurrency Transaction Tracking
  - Social Media Profile Analysis
  - Metadata Extraction
  - Digital Watermark Detection

### 🚀 Quick Deploy

```bash
# Make script executable
chmod +x deploy_everything.sh

# Configure API keys in .env file
nano .env

# Deploy everything
./deploy_everything.sh
```

### 📡 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| IntelOwl Dashboard | http://localhost | Main OSINT interface |
| Cytoscape Graph | http://localhost/cytoscape | Network visualization |
| Neo4j Browser | http://localhost:7474 | Graph database |
| RabbitMQ | http://localhost:15672 | Message queue management |
| Elasticsearch | http://localhost:9200 | Search and analytics |
| InfluxDB | http://localhost:8086 | Time-series metrics |

### 🌐 Tor Configuration

- SOCKS5 Proxy: `socks5://localhost:9050`
- HTTP Proxy: `http://localhost:8118`
- Control Port: `localhost:9051`
- Automatic circuit rotation every 10 minutes

### 💾 Database Connections

```yaml
PostgreSQL: postgresql://bev:BevOSINT2024@localhost:5432/osint
Neo4j: bolt://localhost:7687 (neo4j/BevGraphMaster2024)
Redis: redis://:BevCacheMaster@localhost:6379
```

### 🔧 Architecture

```
┌─────────────────────────────────────────────────┐
│               IntelOwl Web Interface            │
│                  (Dark Theme)                   │
└────────────┬────────────────────────────────────┘
             │
    ┌────────▼────────┬─────────────┬────────────┐
    │   Custom        │  Cytoscape  │    Tor     │
    │   Analyzers     │    Graph    │   Proxy    │
    └────────┬────────┴──────┬──────┴────────────┘
             │                │
    ┌────────▼────────────────▼──────────────────┐
    │         Message Queue & Cache              │
    │    RabbitMQ (3 nodes) + Redis Cluster     │
    └────────────────┬───────────────────────────┘
                     │
    ┌────────────────▼───────────────────────────┐
    │           Data Storage Layer               │
    │  PostgreSQL + Neo4j + Elasticsearch        │
    └────────────────────────────────────────────┘
```

### 📊 Custom Analyzers

#### BreachDatabaseAnalyzer
- Searches Dehashed, Snusbase, WeLeakInfo APIs
- Correlates accounts across breaches
- Risk scoring and timeline analysis

#### DarknetMarketAnalyzer  
- Scrapes Alphabay, White House, Torrez markets
- Vendor profiling and product tracking
- Cryptocurrency address extraction

#### CryptoTrackerAnalyzer
- Bitcoin/Ethereum transaction analysis
- Wallet clustering
- Mixer detection
- Exchange identification

#### SocialMediaAnalyzer
- Instagram, Twitter, LinkedIn scraping
- Network graph building
- Pattern analysis
- Risk assessment

### 🛠 Management Commands

```bash
# View all logs
docker-compose -f docker-compose.complete.yml logs -f

# Stop all services
docker-compose -f docker-compose.complete.yml down

# Restart specific service
docker-compose -f docker-compose.complete.yml restart intelowl-django

# Access container shell
docker exec -it bev_intelowl_django bash

# Backup Neo4j database
docker exec bev_neo4j neo4j-admin dump --to=/data/backup.dump

# Clear Redis cache
docker exec bev_redis redis-cli FLUSHALL
```

### ⚠️ Security Notice

**This deployment has NO AUTHENTICATION enabled!**

- Only run on private networks
- Use firewall rules to block external access
- Never expose to public internet
- All data is stored unencrypted

### 🔥 Performance Tuning

The system is configured for maximum performance:
- 16 workers per service
- 4 threads per worker  
- 100 connection pool size
- No rate limiting
- No authentication overhead
- Aggressive caching

### 📝 Environment Variables

Key variables to configure in `.env`:

```bash
# API Keys (Required)
DEHASHED_API_KEY=xxx
SNUSBASE_API_KEY=xxx
SHODAN_API_KEY=xxx
VIRUSTOTAL_API_KEY=xxx

# Social Media (Optional)
INSTAGRAM_USERNAME=xxx
TWITTER_API_KEY=xxx
LINKEDIN_USERNAME=xxx

# Crypto APIs (Optional)
ETHERSCAN_API_KEY=xxx
BLOCKCHAIN_INFO_API_KEY=xxx
```

### 🐛 Troubleshooting

#### IntelOwl not starting
```bash
docker logs bev_intelowl_django
docker exec bev_intelowl_django python manage.py migrate
```

#### Neo4j connection issues
```bash
docker logs bev_neo4j
# Default password: BevGraphMaster2024
```

#### Tor not connecting
```bash
docker logs bev_tor
docker restart bev_tor
```

### 📚 Custom Analyzer Usage

1. Navigate to IntelOwl dashboard
2. Create new analysis job
3. Select observable type (email, IP, username, etc.)
4. Choose custom analyzers:
   - BreachDatabaseAnalyzer
   - DarknetMarketAnalyzer
   - CryptoTrackerAnalyzer
   - SocialMediaAnalyzer
5. View results in dashboard
6. Export to Neo4j for graph visualization

### 🎯 Example Workflows

#### Investigate Email Address
1. Input email in IntelOwl
2. Run BreachDatabaseAnalyzer
3. View breach timeline
4. Check associated accounts
5. Export to Neo4j
6. Visualize in Cytoscape

#### Track Cryptocurrency
1. Input Bitcoin address
2. Run CryptoTrackerAnalyzer
3. View transaction flow
4. Detect mixer usage
5. Identify exchanges
6. Generate risk report

#### Darknet Vendor Research
1. Input vendor username
2. Run DarknetMarketAnalyzer
3. View profile across markets
4. Extract crypto addresses
5. Track product listings
6. Build vendor network

### 🔄 Updates

```bash
# Pull latest images
docker-compose -f docker-compose.complete.yml pull

# Restart services
docker-compose -f docker-compose.complete.yml up -d
```

### 💀 Uninstall

```bash
# Stop and remove all containers
docker-compose -f docker-compose.complete.yml down -v

# Remove all data
rm -rf /home/starlord/Bev/logs
rm -rf /home/starlord/Bev/redis
rm -rf /home/starlord/Bev/rabbitmq
rm -rf /home/starlord/Bev/kafka
```

---

**Built for single-user OSINT operations on private networks**

🔒 Use responsibly - This tool is for authorized security research only
