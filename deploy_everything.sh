#!/bin/bash

##############################################################################
# Bev OSINT Framework - Complete Deployment Script
# Single-user deployment with IntelOwl + Cytoscape.js
# No authentication overhead - Maximum performance
##############################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Banner
echo -e "${PURPLE}"
cat << "EOF"
╔══════════════════════════════════════════════════════════════════╗
║   ____                  ___  ____  ___ _   _ _____              ║
║  |  _ \                / _ \/ ___||_ _| \ | |_   _|             ║
║  | |_) | _____   __   | | | \___ \ | ||  \| | | |               ║
║  |  _ < / _ \ \ / /   | |_| |___) || || |\  | | |               ║
║  |_| \_\  __/\ V /     \___/|____/|___|_| \_| |_|               ║
║          \___|\_/                                                ║
║                                                                  ║
║        DARKNET INTELLIGENCE FRAMEWORK v3.0                      ║
║        Single User Mode - No Auth - Maximum Power               ║
╚══════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# Configuration
BEV_HOME="/home/starlord/Bev"
DOCKER_COMPOSE_FILE="$BEV_HOME/docker-compose.complete.yml"
ENV_FILE="$BEV_HOME/.env"
LOG_DIR="$BEV_HOME/logs"

# Function to print colored messages
print_status() {
    echo -e "${BLUE}[*]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    print_success "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    print_success "Docker Compose found: $(docker-compose --version)"
    
    # Check if running as root
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root - this is fine for single-user deployment"
    fi
    
    # Check disk space
    available_space=$(df -BG "$BEV_HOME" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 50 ]; then
        print_warning "Low disk space: ${available_space}GB available (50GB recommended)"
    else
        print_success "Disk space: ${available_space}GB available"
    fi
    
    # Check memory
    total_mem=$(free -g | awk 'NR==2 {print $2}')
    if [ "$total_mem" -lt 16 ]; then
        print_warning "Low memory: ${total_mem}GB (16GB recommended)"
    else
        print_success "Memory: ${total_mem}GB available"
    fi
}

# Create required directories
setup_directories() {
    print_status "Setting up directory structure..."
    
    directories=(
        "$LOG_DIR"
        "$BEV_HOME/redis/node1"
        "$BEV_HOME/redis/node2"
        "$BEV_HOME/redis/node3"
        "$BEV_HOME/rabbitmq/node1"
        "$BEV_HOME/rabbitmq/node2"
        "$BEV_HOME/rabbitmq/node3"
        "$BEV_HOME/kafka/broker1"
        "$BEV_HOME/kafka/broker2"
        "$BEV_HOME/kafka/broker3"
        "$BEV_HOME/zookeeper/log"
        "$BEV_HOME/influxdb/config"
        "$BEV_HOME/tor"
        "$BEV_HOME/init_scripts"
        "$BEV_HOME/intelowl/sql_init"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_success "Created: $dir"
    done
}

# Create initialization scripts
create_init_scripts() {
    print_status "Creating database initialization scripts..."
    
    # PostgreSQL init script
    cat > "$BEV_HOME/init_scripts/postgres_init.sql" << 'EOF'
-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create databases
CREATE DATABASE IF NOT EXISTS osint;
CREATE DATABASE IF NOT EXISTS breach_data;
CREATE DATABASE IF NOT EXISTS crypto_analysis;

-- Create breach cache table
\c breach_data;
CREATE TABLE IF NOT EXISTS breach_cache (
    query VARCHAR(255) PRIMARY KEY,
    results JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_breach_cache_created ON breach_cache(created_at);
CREATE INDEX idx_breach_results ON breach_cache USING gin(results);

-- Create crypto tracking tables
\c crypto_analysis;
CREATE TABLE IF NOT EXISTS addresses (
    address VARCHAR(255) PRIMARY KEY,
    currency VARCHAR(10),
    balance DECIMAL(30,10),
    risk_score INTEGER,
    last_analyzed TIMESTAMP,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS transactions (
    hash VARCHAR(255) PRIMARY KEY,
    from_address VARCHAR(255),
    to_address VARCHAR(255),
    amount DECIMAL(30,10),
    timestamp TIMESTAMP,
    block_height INTEGER,
    metadata JSONB
);

CREATE INDEX idx_tx_from ON transactions(from_address);
CREATE INDEX idx_tx_to ON transactions(to_address);
CREATE INDEX idx_tx_time ON transactions(timestamp);

-- Create OSINT metadata tables
\c osint;
CREATE TABLE IF NOT EXISTS search_history (
    id SERIAL PRIMARY KEY,
    query TEXT,
    search_type VARCHAR(50),
    results JSONB,
    timestamp TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS entity_profiles (
    entity_id VARCHAR(255) PRIMARY KEY,
    entity_type VARCHAR(50),
    profile_data JSONB,
    risk_assessment JSONB,
    last_updated TIMESTAMP DEFAULT NOW(),
    embedding vector(384)
);

CREATE INDEX idx_entity_type ON entity_profiles(entity_type);
CREATE INDEX idx_entity_embedding ON entity_profiles USING ivfflat (embedding vector_cosine_ops);
EOF
    print_success "PostgreSQL init script created"
    
    # Neo4j init script
    cat > "$BEV_HOME/init_scripts/neo4j_init.cypher" << 'EOF'
// Create indexes for performance
CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);
CREATE INDEX identity_value IF NOT EXISTS FOR (i:Identity) ON (i.value);
CREATE INDEX breach_database IF NOT EXISTS FOR (b:Breach) ON (b.database);
CREATE INDEX crypto_address IF NOT EXISTS FOR (c:CryptoAddress) ON (c.address);
CREATE INDEX darknet_vendor IF NOT EXISTS FOR (v:DarknetVendor) ON (v.name);
CREATE INDEX social_profile IF NOT EXISTS FOR (s:SocialProfile) ON (s.username, s.platform);

// Create constraints
CREATE CONSTRAINT unique_crypto_address IF NOT EXISTS FOR (c:CryptoAddress) REQUIRE c.address IS UNIQUE;
CREATE CONSTRAINT unique_identity IF NOT EXISTS FOR (i:Identity) REQUIRE i.value IS UNIQUE;
CREATE CONSTRAINT unique_vendor IF NOT EXISTS FOR (v:DarknetVendor) REQUIRE v.name IS UNIQUE;
EOF
    print_success "Neo4j init script created"
    
    # Tor configuration
    cat > "$BEV_HOME/tor/torrc" << 'EOF'
# Bev OSINT Tor Configuration
SocksPort 0.0.0.0:9050
ControlPort 0.0.0.0:9051
HashedControlPassword 16:872860B76453A77D60CA2BB8C1A7042072093276A3D701AD684053EC4C

# Circuit settings
NewCircuitPeriod 600
MaxCircuitDirtiness 600
NumEntryGuards 6

# Performance
CircuitBuildTimeout 30
LearnCircuitBuildTimeout 0
CircuitStreamTimeout 30

# Security
SafeLogging 1
EOF
    print_success "Tor configuration created"
}

# Create IntelOwl configuration
configure_intelowl() {
    print_status "Configuring IntelOwl..."
    
    # Create nginx config for IntelOwl
    cat > "$BEV_HOME/intelowl/nginx.conf" << 'EOF'
worker_processes auto;

events {
    worker_connections 2048;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 10G;
    
    # Disable auth for single user
    auth_basic off;
    
    upstream django {
        server intelowl-django:8000;
    }
    
    upstream cytoscape {
        server cytoscape-server:3000;
    }
    
    server {
        listen 80 default_server;
        listen [::]:80 default_server;
        server_name _;
        
        # Static files
        location /static/ {
            alias /usr/share/nginx/html/static/;
            expires 30d;
            add_header Cache-Control "public";
        }
        
        # Cytoscape.js interface
        location /cytoscape/ {
            proxy_pass http://cytoscape/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support for real-time updates
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
        
        # IntelOwl API
        location / {
            proxy_pass http://django;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_read_timeout 600s;
            proxy_connect_timeout 600s;
            proxy_send_timeout 600s;
        }
    }
}
EOF
    print_success "Nginx configuration created"
    
    # Create dark theme CSS
    cat > "$BEV_HOME/intelowl/dark_theme.css" << 'EOF'
/* Bev OSINT Dark Theme */
:root {
    --bg-primary: #0a0a0a;
    --bg-secondary: #1a1a1a;
    --bg-tertiary: #252525;
    --text-primary: #00ff00;
    --text-secondary: #00cc00;
    --text-muted: #008800;
    --border-color: #00ff00;
    --accent-color: #ff0000;
    --warning-color: #ffff00;
}

body {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Courier New', monospace !important;
}

.navbar {
    background-color: var(--bg-secondary) !important;
    border-bottom: 2px solid var(--border-color) !important;
}

.card {
    background-color: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    box-shadow: 0 0 10px rgba(0, 255, 0, 0.3) !important;
}

.btn-primary {
    background-color: var(--text-muted) !important;
    border-color: var(--text-primary) !important;
    color: var(--bg-primary) !important;
}

.btn-primary:hover {
    background-color: var(--text-primary) !important;
    box-shadow: 0 0 15px var(--text-primary) !important;
}

.table {
    color: var(--text-primary) !important;
    border-color: var(--border-color) !important;
}

.modal-content {
    background-color: var(--bg-secondary) !important;
    border: 2px solid var(--border-color) !important;
}

/* Matrix rain effect for background */
body::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 255, 0, 0.03) 2px,
        rgba(0, 255, 0, 0.03) 4px
    );
    pointer-events: none;
    z-index: -1;
}

/* Glowing text effect */
h1, h2, h3 {
    text-shadow: 0 0 10px var(--text-primary);
}

/* Terminal-style inputs */
input, textarea, select {
    background-color: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    font-family: 'Courier New', monospace !important;
}

input:focus, textarea:focus, select:focus {
    box-shadow: 0 0 5px var(--text-primary) !important;
    outline: none !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
    background-color: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background-color: var(--text-muted);
    border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
    background-color: var(--text-primary);
}
EOF
    print_success "Dark theme created"
}

# Create Cytoscape configuration
configure_cytoscape() {
    print_status "Configuring Cytoscape.js..."
    
    # Create Cytoscape config
    cat > "$BEV_HOME/cytoscape/config.js" << 'EOF'
// Bev OSINT Cytoscape Configuration
const cytoscapeConfig = {
    container: document.getElementById('cy'),
    
    style: [
        {
            selector: 'node',
            style: {
                'background-color': '#00ff00',
                'label': 'data(label)',
                'color': '#00ff00',
                'text-outline-color': '#000',
                'text-outline-width': 2,
                'font-family': 'Courier New',
                'font-size': '12px',
                'border-width': 2,
                'border-color': '#00ff00',
                'width': 'mapData(importance, 0, 10, 30, 80)',
                'height': 'mapData(importance, 0, 10, 30, 80)'
            }
        },
        {
            selector: 'node[type="threat"]',
            style: {
                'background-color': '#ff0000',
                'border-color': '#ff0000',
                'shape': 'triangle'
            }
        },
        {
            selector: 'node[type="vendor"]',
            style: {
                'background-color': '#ffff00',
                'border-color': '#ffff00',
                'shape': 'diamond'
            }
        },
        {
            selector: 'node[type="crypto"]',
            style: {
                'background-color': '#00ffff',
                'border-color': '#00ffff',
                'shape': 'hexagon'
            }
        },
        {
            selector: 'edge',
            style: {
                'width': 'mapData(weight, 0, 10, 1, 8)',
                'line-color': '#00ff00',
                'target-arrow-color': '#00ff00',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'opacity': 0.7
            }
        },
        {
            selector: 'edge[type="transaction"]',
            style: {
                'line-color': '#00ffff',
                'line-style': 'dashed'
            }
        },
        {
            selector: 'edge[type="threat"]',
            style: {
                'line-color': '#ff0000',
                'width': 3
            }
        }
    ],
    
    layout: {
        name: 'cose-bilkent',
        animate: true,
        animationDuration: 1000,
        quality: 'proof',
        nodeDimensionsIncludeLabels: true,
        nodeRepulsion: 50000,
        idealEdgeLength: 100,
        edgeElasticity: 0.1,
        nestingFactor: 0.1,
        gravity: 0.25,
        numIter: 2500,
        tile: true,
        tilingPaddingVertical: 10,
        tilingPaddingHorizontal: 10
    },
    
    minZoom: 0.1,
    maxZoom: 10,
    wheelSensitivity: 0.2
};

// Dark theme for Cytoscape container
document.body.style.backgroundColor = '#0a0a0a';
document.body.style.color = '#00ff00';
document.body.style.fontFamily = 'Courier New, monospace';

// Export configuration
module.exports = cytoscapeConfig;
EOF
    print_success "Cytoscape configuration created"
    
    # Create Cytoscape Dockerfile
    cat > "$BEV_HOME/cytoscape/Dockerfile" << 'EOF'
FROM node:18-alpine

WORKDIR /app

# Install dependencies
RUN npm init -y && \
    npm install express \
                cytoscape \
                cytoscape-cose-bilkent \
                neo4j-driver \
                pg \
                ws \
                cors

# Copy application files
COPY . .

# Create server
RUN cat > server.js << 'EOSERVER'
const express = require('express');
const neo4j = require('neo4j-driver');
const { Client } = require('pg');
const WebSocket = require('ws');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('.'));

// Neo4j connection
const neo4jDriver = neo4j.driver(
    process.env.NEO4J_URI,
    neo4j.auth.basic(process.env.NEO4J_USER, process.env.NEO4J_PASSWORD)
);

// PostgreSQL connection
const pgClient = new Client({
    connectionString: process.env.POSTGRES_URI
});
pgClient.connect();

// WebSocket server for real-time updates
const wss = new WebSocket.Server({ port: 3001 });

// API endpoints
app.get('/api/graph/:query', async (req, res) => {
    const session = neo4jDriver.session();
    try {
        const result = await session.run(
            'MATCH (n)-[r]-(m) WHERE n.value = $query RETURN n, r, m LIMIT 500',
            { query: req.params.query }
        );
        
        const nodes = [];
        const edges = [];
        const nodeIds = new Set();
        
        result.records.forEach(record => {
            const n = record.get('n');
            const r = record.get('r');
            const m = record.get('m');
            
            if (!nodeIds.has(n.identity.toString())) {
                nodes.push({
                    data: {
                        id: n.identity.toString(),
                        label: n.properties.value || n.properties.name,
                        ...n.properties
                    }
                });
                nodeIds.add(n.identity.toString());
            }
            
            if (!nodeIds.has(m.identity.toString())) {
                nodes.push({
                    data: {
                        id: m.identity.toString(),
                        label: m.properties.value || m.properties.name,
                        ...m.properties
                    }
                });
                nodeIds.add(m.identity.toString());
            }
            
            edges.push({
                data: {
                    id: r.identity.toString(),
                    source: r.start.toString(),
                    target: r.end.toString(),
                    type: r.type,
                    ...r.properties
                }
            });
        });
        
        res.json({ nodes, edges });
    } finally {
        await session.close();
    }
});

// WebSocket broadcasting for real-time updates
wss.on('connection', (ws) => {
    console.log('New WebSocket client connected');
    
    ws.on('message', (message) => {
        // Broadcast updates to all clients
        wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(message);
            }
        });
    });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(\`Cytoscape server running on port \${PORT}\`);
});
EOSERVER

EXPOSE 3000 3001
CMD ["node", "server.js"]
EOF
    print_success "Cytoscape Docker image configured"
}

# Pull Docker images
pull_images() {
    print_status "Pulling Docker images..."
    
    images=(
        "pgvector/pgvector:pg16"
        "neo4j:5.14-enterprise"
        "redis:7-alpine"
        "rabbitmq:3-management"
        "confluentinc/cp-zookeeper:7.5.0"
        "confluentinc/cp-kafka:7.5.0"
        "docker.elastic.co/elasticsearch/elasticsearch:8.11.0"
        "influxdb:2.7"
        "dperson/torproxy:latest"
        "intelowlproject/intelowl:v5.2.0"
        "nginx:alpine"
    )
    
    for image in "${images[@]}"; do
        print_status "Pulling $image..."
        docker pull "$image" || print_warning "Failed to pull $image"
    done
    
    print_success "All images pulled"
}

# Start services
start_services() {
    print_status "Starting all services..."
    
    cd "$BEV_HOME"
    
    # Start infrastructure first
    print_status "Starting infrastructure services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d \
        postgres neo4j redis redis-node-1 redis-node-2 redis-node-3 \
        rabbitmq-1 zookeeper tor
    
    sleep 10
    
    # Initialize Redis cluster
    print_status "Initializing Redis cluster..."
    docker exec bev_redis_1 redis-cli --cluster create \
        172.30.0.4:7001 172.30.0.5:7002 172.30.0.6:7003 \
        --cluster-replicas 0 --cluster-yes -a "${REDIS_PASSWORD}" || true
    
    # Start Kafka cluster
    print_status "Starting Kafka cluster..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d \
        kafka-1 kafka-2 kafka-3
    
    sleep 10
    
    # Initialize RabbitMQ cluster
    print_status "Initializing RabbitMQ cluster..."
    docker exec bev_rabbitmq_2 rabbitmqctl stop_app || true
    docker exec bev_rabbitmq_2 rabbitmqctl reset || true
    docker exec bev_rabbitmq_2 rabbitmqctl join_cluster rabbit@rabbitmq-1 || true
    docker exec bev_rabbitmq_2 rabbitmqctl start_app || true
    
    docker exec bev_rabbitmq_3 rabbitmqctl stop_app || true
    docker exec bev_rabbitmq_3 rabbitmqctl reset || true
    docker exec bev_rabbitmq_3 rabbitmqctl join_cluster rabbit@rabbitmq-1 || true
    docker exec bev_rabbitmq_3 rabbitmqctl start_app || true
    
    # Start remaining services
    print_status "Starting application services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    print_success "All services started"
}

# Health checks
health_check() {
    print_status "Running health checks..."
    
    services=(
        "postgres:5432"
        "neo4j:7474"
        "redis:6379"
        "rabbitmq-1:15672"
        "elasticsearch:9200"
        "influxdb:8086"
        "tor:9050"
        "intelowl-django:8000"
    )
    
    for service in "${services[@]}"; do
        IFS=':' read -ra ADDR <<< "$service"
        container="${ADDR[0]}"
        port="${ADDR[1]}"
        
        if nc -zv localhost "$port" 2>/dev/null; then
            print_success "$container is responding on port $port"
        else
            print_warning "$container is not responding on port $port"
        fi
    done
}

# Display access information
display_info() {
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║              BEV OSINT FRAMEWORK DEPLOYED SUCCESSFULLY          ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
    
    echo -e "${CYAN}Access Points:${NC}"
    echo -e "${GREEN}►${NC} IntelOwl Dashboard:    ${YELLOW}http://localhost${NC}"
    echo -e "${GREEN}►${NC} Cytoscape Graph:       ${YELLOW}http://localhost/cytoscape${NC}"
    echo -e "${GREEN}►${NC} Neo4j Browser:         ${YELLOW}http://localhost:7474${NC}"
    echo -e "${GREEN}►${NC} RabbitMQ Management:   ${YELLOW}http://localhost:15672${NC}"
    echo -e "${GREEN}►${NC} Elasticsearch:         ${YELLOW}http://localhost:9200${NC}"
    echo -e "${GREEN}►${NC} InfluxDB:              ${YELLOW}http://localhost:8086${NC}"
    
    echo -e "\n${CYAN}Tor SOCKS5 Proxy:${NC}"
    echo -e "${GREEN}►${NC} SOCKS5: ${YELLOW}socks5://localhost:9050${NC}"
    echo -e "${GREEN}►${NC} HTTP:   ${YELLOW}http://localhost:8118${NC}"
    
    echo -e "\n${CYAN}Database Connections:${NC}"
    echo -e "${GREEN}►${NC} PostgreSQL: ${YELLOW}postgresql://bev:BevOSINT2024@localhost:5432/osint${NC}"
    echo -e "${GREEN}►${NC} Neo4j:      ${YELLOW}bolt://localhost:7687${NC} (user: neo4j, pass: BevGraphMaster2024)"
    echo -e "${GREEN}►${NC} Redis:      ${YELLOW}redis://:BevCacheMaster@localhost:6379${NC}"
    
    echo -e "\n${PURPLE}Custom Analyzers Available:${NC}"
    echo -e "${GREEN}►${NC} BreachDatabaseAnalyzer - Search Dehashed, Snusbase, WeLeakInfo"
    echo -e "${GREEN}►${NC} DarknetMarketAnalyzer - Scrape darknet markets via Tor"
    echo -e "${GREEN}►${NC} CryptoTrackerAnalyzer - Track Bitcoin/Ethereum transactions"
    echo -e "${GREEN}►${NC} SocialMediaAnalyzer - Analyze Instagram, Twitter, LinkedIn"
    echo -e "${GREEN}►${NC} MetadataAnalyzer - Extract file metadata"
    echo -e "${GREEN}►${NC} WatermarkAnalyzer - Detect digital watermarks"
    
    echo -e "\n${RED}⚠ SECURITY NOTICE:${NC}"
    echo -e "This deployment has ${RED}NO AUTHENTICATION${NC} enabled."
    echo -e "Only run on ${YELLOW}PRIVATE NETWORKS${NC} with no external access."
    
    echo -e "\n${CYAN}Logs Directory:${NC} ${YELLOW}$LOG_DIR${NC}"
    echo -e "${CYAN}Stop All Services:${NC} ${YELLOW}docker-compose -f $DOCKER_COMPOSE_FILE down${NC}"
    echo -e "${CYAN}View Logs:${NC} ${YELLOW}docker-compose -f $DOCKER_COMPOSE_FILE logs -f [service_name]${NC}"
}

# Main execution
main() {
    echo -e "${PURPLE}Starting Bev OSINT Framework deployment...${NC}\n"
    
    check_prerequisites
    setup_directories
    create_init_scripts
    configure_intelowl
    configure_cytoscape
    
    # Source environment variables
    if [ -f "$ENV_FILE" ]; then
        export $(cat "$ENV_FILE" | grep -v '^#' | xargs)
        print_success "Environment variables loaded"
    else
        print_error "Environment file not found: $ENV_FILE"
        exit 1
    fi
    
    pull_images
    start_services
    
    # Wait for services to be ready
    print_status "Waiting for services to initialize..."
    sleep 30
    
    health_check
    display_info
    
    # Open browser
    print_status "Opening IntelOwl dashboard in browser..."
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost" &
    elif command -v open &> /dev/null; then
        open "http://localhost" &
    else
        print_warning "Please open http://localhost in your browser"
    fi
    
    echo -e "\n${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    DEPLOYMENT COMPLETE!                         ║${NC}"
    echo -e "${GREEN}║           Your OSINT framework is ready for action              ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
}

# Run main function
main "$@"
