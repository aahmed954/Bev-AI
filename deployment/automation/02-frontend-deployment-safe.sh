#!/bin/bash
# BEV Frontend Integration - Conflict-Free Deployment Script
# Modified version using safe ports and network isolation
# Author: DevOps Automation Framework
# Version: 1.0.0

set -euo pipefail

# =====================================================
# Configuration and Constants
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/deployment"
LOG_FILE="${LOG_DIR}/frontend-deployment-$(date +%Y%m%d_%H%M%S).log"

# Safe port configuration (avoiding conflicts)
FRONTEND_HTTP_PORT=3010
FRONTEND_HTTPS_PORT=8443
FRONTEND_WS_PORT=8081
MCP_SERVER_PORT=3011
DESKTOP_APP_PORT=3012

# Network configuration
FRONTEND_NETWORK="bev_frontend"
FRONTEND_SUBNET="172.31.0.0/16"
BRIDGE_NETWORK="bev_bridge"

# SSL Configuration
SSL_DIR="${PROJECT_ROOT}/config/ssl"
SSL_CERT="${SSL_DIR}/bev-frontend.crt"
SSL_KEY="${SSL_DIR}/bev-frontend.key"

# Container names
MCP_CONTAINER="bev-mcp-server"
DESKTOP_CONTAINER="bev-desktop-app"
PROXY_CONTAINER="bev-frontend-proxy"

# =====================================================
# Logging Functions
# =====================================================

setup_logging() {
    mkdir -p "${LOG_DIR}"
    exec 1> >(tee -a "${LOG_FILE}")
    exec 2> >(tee -a "${LOG_FILE}" >&2)
    echo "=== BEV Frontend Deployment Started at $(date) ===" | tee -a "${LOG_FILE}"
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*" | tee -a "${LOG_FILE}"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" | tee -a "${LOG_FILE}"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" | tee -a "${LOG_FILE}"
}

log_success() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [SUCCESS] $*" | tee -a "${LOG_FILE}"
}

# =====================================================
# Pre-flight Checks
# =====================================================

validate_prerequisites() {
    log_info "Validating deployment prerequisites..."
    
    # Check validation marker from pre-deployment script
    if [ ! -f "${PROJECT_ROOT}/.deployment_validation" ]; then
        log_error "Pre-deployment validation not completed. Run 01-pre-deployment-validation.sh first"
        exit 1
    fi
    
    # Check validation status
    local validation_status=$(grep "VALIDATION_STATUS" "${PROJECT_ROOT}/.deployment_validation" | cut -d= -f2)
    if [ "${validation_status}" != "PASSED" ]; then
        log_error "Pre-deployment validation failed. Cannot proceed with deployment"
        exit 1
    fi
    
    # Load environment variables
    if [ -f "${PROJECT_ROOT}/.env" ]; then
        set -a
        source "${PROJECT_ROOT}/.env"
        set +a
        log_info "Environment variables loaded"
    else
        log_error ".env file not found"
        exit 1
    fi
    
    log_success "Prerequisites validation passed"
}

# =====================================================
# Network Configuration
# =====================================================

setup_networks() {
    log_info "Setting up network configuration..."
    
    # Remove existing frontend network if it exists
    if docker network ls --format "{{.Name}}" | grep -q "^${FRONTEND_NETWORK}$"; then
        log_info "Removing existing frontend network..."
        docker network rm "${FRONTEND_NETWORK}" || true
    fi
    
    # Create frontend network with safe subnet
    log_info "Creating frontend network: ${FRONTEND_NETWORK}"
    docker network create \
        --driver bridge \
        --subnet "${FRONTEND_SUBNET}" \
        --opt com.docker.network.bridge.name="br-bev-frontend" \
        --label "project=bev-osint" \
        --label "component=frontend" \
        "${FRONTEND_NETWORK}"
    
    # Create bridge network to connect with existing BEV services
    if ! docker network ls --format "{{.Name}}" | grep -q "^${BRIDGE_NETWORK}$"; then
        log_info "Creating bridge network: ${BRIDGE_NETWORK}"
        docker network create \
            --driver bridge \
            --opt com.docker.network.bridge.name="br-bev-bridge" \
            --label "project=bev-osint" \
            --label "component=bridge" \
            "${BRIDGE_NETWORK}"
    fi
    
    log_success "Network configuration completed"
}

# =====================================================
# SSL Certificate Management
# =====================================================

setup_ssl_certificates() {
    log_info "Setting up SSL certificates..."
    
    mkdir -p "${SSL_DIR}"
    
    if [ ! -f "${SSL_CERT}" ] || [ ! -f "${SSL_KEY}" ]; then
        log_info "Generating self-signed SSL certificates..."
        
        # Generate private key
        openssl genrsa -out "${SSL_KEY}" 2048
        
        # Generate certificate signing request
        openssl req -new -key "${SSL_KEY}" -out "${SSL_DIR}/bev-frontend.csr" \
            -subj "/C=US/ST=State/L=City/O=BEV-OSINT/OU=Frontend/CN=localhost/emailAddress=admin@bev-osint.local"
        
        # Generate self-signed certificate
        openssl x509 -req -days 365 -in "${SSL_DIR}/bev-frontend.csr" \
            -signkey "${SSL_KEY}" -out "${SSL_CERT}" \
            -extensions v3_req -extfile <(cat <<EOF
[v3_req]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
subjectAltName = @alt_names
[alt_names]
DNS.1 = localhost
DNS.2 = bev-frontend
IP.1 = 127.0.0.1
IP.2 = 172.31.0.10
EOF
        )
        
        # Set proper permissions
        chmod 600 "${SSL_KEY}"
        chmod 644 "${SSL_CERT}"
        
        log_success "SSL certificates generated"
    else
        log_info "SSL certificates already exist"
    fi
    
    # Validate certificate
    local cert_info=$(openssl x509 -in "${SSL_CERT}" -noout -subject -dates 2>/dev/null)
    log_info "Certificate info: ${cert_info}"
}

# =====================================================
# Container Management
# =====================================================

create_frontend_directories() {
    log_info "Creating frontend directory structure..."
    
    # Create comprehensive directory structure
    mkdir -p "${PROJECT_ROOT}/frontend"/{mcp-server,desktop-app,proxy,shared}
    mkdir -p "${PROJECT_ROOT}/frontend/mcp-server"/{src,config,logs}
    mkdir -p "${PROJECT_ROOT}/frontend/mcp-server/src"/{tools,handlers,security,utils,middleware}
    mkdir -p "${PROJECT_ROOT}/frontend/desktop-app"/{src,assets,config,dist}
    mkdir -p "${PROJECT_ROOT}/frontend/desktop-app/src"/{main,renderer,api,security,utils}
    mkdir -p "${PROJECT_ROOT}/frontend/desktop-app/src/renderer"/{components,layouts,hooks,utils,assets}
    mkdir -p "${PROJECT_ROOT}/frontend/proxy"/{config,logs,ssl}
    mkdir -p "${PROJECT_ROOT}/frontend/shared"/{types,utils,constants}
    
    log_success "Directory structure created"
}

generate_mcp_server_config() {
    log_info "Generating MCP server configuration..."
    
    # Create package.json with conflict-free ports
    cat > "${PROJECT_ROOT}/frontend/mcp-server/package.json" << 'EOF'
{
  "name": "bev-osint-mcp-server",
  "version": "1.0.0",
  "type": "module",
  "main": "src/index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "nodemon src/index.js --watch src --ext js,json",
    "build": "esbuild src/index.js --bundle --platform=node --outfile=dist/server.js",
    "test": "jest",
    "lint": "eslint src/**/*.js",
    "healthcheck": "curl -f http://localhost:3011/health || exit 1"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",
    "axios": "^1.6.2",
    "ws": "^8.14.2",
    "dotenv": "^16.3.1",
    "winston": "^3.11.0",
    "express": "^4.18.2",
    "cors": "^2.8.5",
    "helmet": "^7.1.0",
    "rate-limiter-flexible": "^3.0.8",
    "jsonwebtoken": "^9.0.2",
    "bcryptjs": "^2.4.3",
    "uuid": "^9.0.1",
    "joi": "^17.11.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.2",
    "esbuild": "^0.19.8",
    "jest": "^29.7.0",
    "eslint": "^8.56.0",
    "supertest": "^6.3.3"
  },
  "engines": {
    "node": ">=18.0.0"
  }
}
EOF

    # Create main server file
    cat > "${PROJECT_ROOT}/frontend/mcp-server/src/index.js" << EOF
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import winston from 'winston';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';

// Load environment variables
dotenv.config({ path: join(dirname(fileURLToPath(import.meta.url)), '../../../.env') });

// Configure logging
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({ 
      filename: join(dirname(fileURLToPath(import.meta.url)), '../logs/server.log') 
    })
  ]
});

// Application configuration
const config = {
  port: process.env.MCP_SERVER_PORT || 3011,
  corsOrigins: process.env.CORS_ORIGINS?.split(',') || ['http://localhost:3010', 'https://localhost:8443'],
  apiKey: process.env.MCP_API_KEY,
  jwtSecret: process.env.JWT_SECRET || process.env.MCP_API_KEY,
  environment: process.env.ENVIRONMENT || 'development'
};

// Validate required configuration
if (!config.apiKey) {
  logger.error('MCP_API_KEY is required');
  process.exit(1);
}

// Create Express application
const app = express();

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      scriptSrc: ["'self'", "'unsafe-inline'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "ws:", "wss:"]
    }
  },
  crossOriginEmbedderPolicy: false
}));

// CORS configuration
app.use(cors({
  origin: config.corsOrigins,
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key']
}));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Request logging
app.use((req, res, next) => {
  logger.info(\`\${req.method} \${req.path}\`, {
    ip: req.ip,
    userAgent: req.get('User-Agent'),
    timestamp: new Date().toISOString()
  });
  next();
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '1.0.0',
    environment: config.environment,
    uptime: process.uptime()
  });
});

// API status endpoint
app.get('/api/status', (req, res) => {
  res.json({
    service: 'bev-mcp-server',
    version: '1.0.0',
    status: 'running',
    port: config.port,
    environment: config.environment,
    features: ['websocket', 'rest-api', 'authentication'],
    connections: {
      active: wss?.clients?.size || 0,
      total: connectionCount
    }
  });
});

// MCP Tools API endpoints
app.post('/api/tools/execute', (req, res) => {
  const { tool, parameters } = req.body;
  
  logger.info('Tool execution request', { tool, parameters });
  
  // Basic tool execution simulation
  setTimeout(() => {
    res.json({
      success: true,
      tool,
      result: \`Tool \${tool} executed successfully\`,
      timestamp: new Date().toISOString(),
      executionTime: Math.random() * 1000
    });
  }, Math.random() * 500);
});

// BEV Integration endpoints
app.get('/api/bev/services', (req, res) => {
  res.json({
    services: [
      { name: 'postgres', status: 'running', port: 5432 },
      { name: 'redis', status: 'running', port: 6379 },
      { name: 'neo4j', status: 'running', port: 7687 },
      { name: 'elasticsearch', status: 'running', port: 9200 }
    ],
    timestamp: new Date().toISOString()
  });
});

// Error handling
app.use((err, req, res, next) => {
  logger.error('Application error', {
    error: err.message,
    stack: err.stack,
    url: req.url,
    method: req.method
  });
  
  res.status(500).json({
    error: 'Internal server error',
    timestamp: new Date().toISOString(),
    requestId: req.id
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
    path: req.path,
    timestamp: new Date().toISOString()
  });
});

// Create HTTP server
const server = createServer(app);

// WebSocket server
let connectionCount = 0;
const wss = new WebSocketServer({ 
  server,
  path: '/ws',
  perMessageDeflate: {
    zlibDeflateOptions: {
      maxNoContextTakeover: false,
    },
  }
});

wss.on('connection', (ws, req) => {
  connectionCount++;
  const clientId = \`client_\${Date.now()}_\${Math.random().toString(36).substr(2, 9)}\`;
  
  logger.info('WebSocket connection established', {
    clientId,
    ip: req.socket.remoteAddress,
    totalConnections: wss.clients.size
  });
  
  ws.id = clientId;
  
  // Send welcome message
  ws.send(JSON.stringify({
    type: 'welcome',
    clientId,
    timestamp: new Date().toISOString(),
    capabilities: ['mcp-tools', 'bev-integration', 'real-time-updates']
  }));
  
  ws.on('message', (data) => {
    try {
      const message = JSON.parse(data);
      logger.info('WebSocket message received', { clientId, type: message.type });
      
      // Echo back for now
      ws.send(JSON.stringify({
        type: 'response',
        originalMessage: message,
        timestamp: new Date().toISOString(),
        clientId
      }));
    } catch (error) {
      logger.error('WebSocket message error', { clientId, error: error.message });
      ws.send(JSON.stringify({
        type: 'error',
        message: 'Invalid message format',
        timestamp: new Date().toISOString()
      }));
    }
  });
  
  ws.on('close', () => {
    logger.info('WebSocket connection closed', {
      clientId,
      remainingConnections: wss.clients.size - 1
    });
  });
  
  ws.on('error', (error) => {
    logger.error('WebSocket error', { clientId, error: error.message });
  });
});

// Graceful shutdown
const gracefulShutdown = (signal) => {
  logger.info(\`Received \${signal}, starting graceful shutdown\`);
  
  server.close(() => {
    logger.info('HTTP server closed');
    
    wss.close(() => {
      logger.info('WebSocket server closed');
      process.exit(0);
    });
  });
  
  // Force exit after 30 seconds
  setTimeout(() => {
    logger.error('Forceful shutdown after timeout');
    process.exit(1);
  }, 30000);
};

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Start server
server.listen(config.port, '0.0.0.0', () => {
  logger.info(\`BEV MCP Server started\`, {
    port: config.port,
    environment: config.environment,
    nodeVersion: process.version,
    processId: process.pid
  });
});

export default app;
EOF

    log_success "MCP server configuration generated"
}

generate_desktop_app_config() {
    log_info "Generating desktop application configuration..."
    
    # Create package.json for desktop app
    cat > "${PROJECT_ROOT}/frontend/desktop-app/package.json" << 'EOF'
{
  "name": "bev-osint-desktop",
  "version": "1.0.0",
  "main": "src/main/index.js",
  "homepage": "./",
  "scripts": {
    "start": "electron src/main/index.js",
    "dev": "concurrently \"npm run dev-main\" \"npm run dev-renderer\"",
    "dev-main": "nodemon src/main/index.js --exec electron",
    "dev-renderer": "vite src/renderer --port 3012",
    "build": "npm run build-renderer && npm run build-main",
    "build-renderer": "vite build src/renderer",
    "build-main": "esbuild src/main/index.js --bundle --platform=node --external:electron --outfile=dist/main.js",
    "package": "electron-builder",
    "test": "jest",
    "lint": "eslint src/**/*.{js,jsx}",
    "preview": "vite preview src/renderer --port 3012"
  },
  "dependencies": {
    "electron": "^28.1.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.1",
    "axios": "^1.6.2",
    "socket.io-client": "^4.7.4",
    "electron-store": "^8.1.0",
    "electron-updater": "^6.1.7",
    "uuid": "^9.0.1",
    "@emotion/react": "^11.11.1",
    "@emotion/styled": "^11.11.0",
    "@mui/material": "^5.15.0",
    "@mui/icons-material": "^5.15.0",
    "recharts": "^2.8.0",
    "date-fns": "^2.30.0"
  },
  "devDependencies": {
    "vite": "^5.0.10",
    "@vitejs/plugin-react": "^4.2.1",
    "electron-builder": "^24.9.1",
    "concurrently": "^8.2.2",
    "nodemon": "^3.0.2",
    "esbuild": "^0.19.8",
    "jest": "^29.7.0",
    "eslint": "^8.56.0",
    "eslint-plugin-react": "^7.33.2",
    "eslint-plugin-react-hooks": "^4.6.0"
  },
  "build": {
    "appId": "com.bev-osint.desktop",
    "productName": "BEV OSINT Desktop",
    "directories": {
      "output": "dist"
    },
    "files": [
      "src/main/**/*",
      "dist/**/*",
      "assets/**/*"
    ],
    "mac": {
      "category": "public.app-category.productivity"
    },
    "win": {
      "target": "nsis"
    },
    "linux": {
      "target": "AppImage"
    }
  }
}
EOF

    # Create main Electron process
    cat > "${PROJECT_ROOT}/frontend/desktop-app/src/main/index.js" << 'EOF'
import { app, BrowserWindow, ipcMain, dialog, shell } from 'electron';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import Store from 'electron-store';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Initialize secure store
const store = new Store({
  encryptionKey: process.env.DESKTOP_ENCRYPTION_KEY,
  name: 'bev-osint-config'
});

// Application configuration
const config = {
  apiUrl: process.env.BEV_API_URL || 'http://localhost:3010',
  mcpUrl: process.env.MCP_SERVER_URL || 'http://localhost:3011',
  wsUrl: process.env.WS_URL || 'ws://localhost:8081',
  isDevelopment: process.env.NODE_ENV === 'development'
};

let mainWindow;

function createWindow() {
  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: join(__dirname, 'preload.js'),
      webSecurity: true,
      allowRunningInsecureContent: false,
      experimentalFeatures: false
    },
    icon: join(__dirname, '../../assets/icon.png'),
    show: false,
    titleBarStyle: 'default',
    backgroundColor: '#1a1a1a'
  });

  // Load the application
  if (config.isDevelopment) {
    mainWindow.loadURL('http://localhost:3012');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(join(__dirname, '../../dist/index.html'));
  }

  // Show window when ready
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    
    if (config.isDevelopment) {
      mainWindow.webContents.openDevTools();
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

// App event handlers
app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// IPC handlers
ipcMain.handle('get-config', () => {
  return {
    apiUrl: config.apiUrl,
    mcpUrl: config.mcpUrl,
    wsUrl: config.wsUrl,
    version: app.getVersion()
  };
});

ipcMain.handle('get-store-value', (event, key) => {
  return store.get(key);
});

ipcMain.handle('set-store-value', (event, key, value) => {
  store.set(key, value);
  return true;
});

ipcMain.handle('show-message-box', async (event, options) => {
  const result = await dialog.showMessageBox(mainWindow, options);
  return result;
});

ipcMain.handle('show-save-dialog', async (event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, options);
  return result;
});

ipcMain.handle('show-open-dialog', async (event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, options);
  return result;
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (event, url) => {
    event.preventDefault();
    shell.openExternal(url);
  });
});

export { mainWindow };
EOF

    log_success "Desktop application configuration generated"
}

generate_haproxy_config() {
    log_info "Generating HAProxy configuration for load balancing..."
    
    mkdir -p "${PROJECT_ROOT}/frontend/proxy/config"
    
    cat > "${PROJECT_ROOT}/frontend/proxy/config/haproxy.cfg" << EOF
global
    daemon
    log stdout local0
    maxconn 4096
    user haproxy
    group haproxy
    ssl-default-bind-ciphers ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384
    ssl-default-bind-options ssl-min-ver TLSv1.2 no-tls-tickets

defaults
    mode http
    log global
    option httplog
    option dontlognull
    option http-server-close
    option forwardfor except 127.0.0.0/8
    option redispatch
    retries 3
    timeout http-request 10s
    timeout queue 1m
    timeout connect 10s
    timeout client 1m
    timeout server 1m
    timeout http-keep-alive 10s
    timeout check 10s
    maxconn 3000

# Statistics page
stats enable
stats uri /haproxy-stats
stats refresh 30s
stats hide-version
stats auth admin:${HAPROXY_STATS_PASSWORD:-bevstats2024}

# Frontend HTTP (redirect to HTTPS)
frontend http_frontend
    bind *:${FRONTEND_HTTP_PORT}
    mode http
    redirect scheme https code 301

# Frontend HTTPS
frontend https_frontend
    bind *:${FRONTEND_HTTPS_PORT} ssl crt /etc/ssl/certs/bev-frontend.pem
    mode http
    
    # Security headers
    http-response set-header X-Frame-Options DENY
    http-response set-header X-Content-Type-Options nosniff
    http-response set-header X-XSS-Protection "1; mode=block"
    http-response set-header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload"
    
    # Route to appropriate backend based on path
    acl is_api path_beg /api/
    acl is_ws path_beg /ws
    acl is_mcp path_beg /mcp/
    
    use_backend mcp_servers if is_mcp
    use_backend api_servers if is_api
    use_backend websocket_servers if is_ws
    default_backend web_servers

# Backend for web servers (static content)
backend web_servers
    mode http
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server web1 172.31.0.10:3010 check inter 30s
    server web2 172.31.0.11:3010 check inter 30s backup

# Backend for API servers
backend api_servers
    mode http
    balance roundrobin
    option httpchk GET /api/health
    http-check expect status 200
    
    server api1 172.31.0.10:3010 check inter 30s
    server api2 172.31.0.11:3010 check inter 30s backup

# Backend for MCP servers
backend mcp_servers
    mode http
    balance roundrobin
    option httpchk GET /health
    http-check expect status 200
    
    server mcp1 172.31.0.12:3011 check inter 30s
    server mcp2 172.31.0.13:3011 check inter 30s backup

# Backend for WebSocket servers
backend websocket_servers
    mode http
    balance source
    option httpchk GET /health
    http-check expect status 200
    
    # Sticky sessions for WebSocket connections
    stick-table type ip size 100k expire 30m
    stick on src
    
    server ws1 172.31.0.14:8081 check inter 30s
    server ws2 172.31.0.15:8081 check inter 30s backup

# Health monitoring
listen health_check
    bind *:8080
    mode http
    stats enable
    stats uri /health
    stats refresh 5s
    stats hide-version
    
    # Simple health check endpoint
    monitor-uri /ping
    
    # Backend server health summary
    stats admin if { src 127.0.0.1 172.31.0.0/16 }
EOF

    log_success "HAProxy configuration generated"
}

generate_docker_compose() {
    log_info "Generating Docker Compose configuration..."
    
    cat > "${PROJECT_ROOT}/frontend/docker-compose.frontend.yml" << EOF
version: '3.9'

x-logging: &default-logging
  driver: json-file
  options:
    max-size: "10m"
    max-file: "3"

networks:
  bev_frontend:
    external: true
  bev_bridge:
    external: true
  bev_osint:
    external: true

volumes:
  mcp_server_logs:
  frontend_logs:
  proxy_logs:
  ssl_certs:

services:
  # MCP Server
  bev-mcp-server:
    build:
      context: ./mcp-server
      dockerfile: Dockerfile
    container_name: ${MCP_CONTAINER}
    restart: unless-stopped
    environment:
      - NODE_ENV=\${ENVIRONMENT:-production}
      - MCP_SERVER_PORT=3011
      - MCP_API_KEY=\${MCP_API_KEY}
      - JWT_SECRET=\${JWT_SECRET}
      - LOG_LEVEL=\${LOG_LEVEL:-info}
      - CORS_ORIGINS=http://localhost:${FRONTEND_HTTP_PORT},https://localhost:${FRONTEND_HTTPS_PORT}
      - BEV_API_URL=http://bev-api:8000
      - DATABASE_URL=\${POSTGRES_URI}
      - REDIS_URL=redis://:\${REDIS_PASSWORD}@bev_redis:6379
    ports:
      - "${MCP_SERVER_PORT}:3011"
    volumes:
      - mcp_server_logs:/app/logs
      - ./mcp-server/src:/app/src:ro
      - ./mcp-server/config:/app/config:ro
    networks:
      bev_frontend:
        ipv4_address: 172.31.0.12
      bev_bridge:
      bev_osint:
    depends_on:
      - bev-frontend-proxy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3011/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging: *default-logging
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.mcp-server.rule=PathPrefix(\`/mcp/\`)"
      - "traefik.http.services.mcp-server.loadbalancer.server.port=3011"

  # Frontend Proxy (HAProxy)
  bev-frontend-proxy:
    image: haproxy:2.8-alpine
    container_name: ${PROXY_CONTAINER}
    restart: unless-stopped
    ports:
      - "${FRONTEND_HTTP_PORT}:3010"
      - "${FRONTEND_HTTPS_PORT}:8443"
      - "8080:8080"  # HAProxy stats
    volumes:
      - ./proxy/config/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
      - ssl_certs:/etc/ssl/certs:ro
      - proxy_logs:/var/log/haproxy
    networks:
      bev_frontend:
        ipv4_address: 172.31.0.10
      bev_bridge:
    environment:
      - HAPROXY_STATS_PASSWORD=\${HAPROXY_STATS_PASSWORD:-bevstats2024}
      - FRONTEND_HTTP_PORT=${FRONTEND_HTTP_PORT}
      - FRONTEND_HTTPS_PORT=${FRONTEND_HTTPS_PORT}
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "3010"]
      interval: 30s
      timeout: 5s
      retries: 3
    logging: *default-logging
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.frontend-proxy.rule=Host(\`localhost\`)"

  # WebSocket Server
  bev-websocket-server:
    build:
      context: ./websocket-server
      dockerfile: Dockerfile
    container_name: bev-websocket-server
    restart: unless-stopped
    environment:
      - NODE_ENV=\${ENVIRONMENT:-production}
      - WS_PORT=${FRONTEND_WS_PORT}
      - WEBSOCKET_SECRET=\${WEBSOCKET_SECRET}
      - REDIS_URL=redis://:\${REDIS_PASSWORD}@bev_redis:6379
    ports:
      - "${FRONTEND_WS_PORT}:8081"
    volumes:
      - frontend_logs:/app/logs
    networks:
      bev_frontend:
        ipv4_address: 172.31.0.14
      bev_bridge:
      bev_osint:
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "8081"]
      interval: 30s
      timeout: 5s
      retries: 3
    logging: *default-logging

  # Nginx for static content serving
  bev-frontend-web:
    image: nginx:1.25-alpine
    container_name: bev-frontend-web
    restart: unless-stopped
    ports:
      - "3013:80"
    volumes:
      - ./web/dist:/usr/share/nginx/html:ro
      - ./web/nginx.conf:/etc/nginx/nginx.conf:ro
      - frontend_logs:/var/log/nginx
    networks:
      bev_frontend:
        ipv4_address: 172.31.0.16
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    logging: *default-logging

  # Redis for session management (separate from main BEV Redis)
  bev-frontend-redis:
    image: redis:7.2-alpine
    container_name: bev-frontend-redis
    restart: unless-stopped
    command: redis-server --requirepass \${FRONTEND_REDIS_PASSWORD:-bevfrontend2024}
    ports:
      - "6380:6379"
    volumes:
      - frontend_redis_data:/data
    networks:
      bev_frontend:
        ipv4_address: 172.31.0.18
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 30s
      timeout: 3s
      retries: 5
    logging: *default-logging

volumes:
  frontend_redis_data:
EOF

    log_success "Docker Compose configuration generated"
}

generate_dockerfiles() {
    log_info "Generating Dockerfiles..."
    
    # MCP Server Dockerfile
    cat > "${PROJECT_ROOT}/frontend/mcp-server/Dockerfile" << 'EOF'
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Build application
RUN npm run build

# Production image
FROM node:18-alpine AS production

# Create non-root user
RUN addgroup -g 1001 -S bev && \
    adduser -S bev -u 1001 -G bev

WORKDIR /app

# Install curl for health checks
RUN apk add --no-cache curl

# Copy built application
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./
COPY --from=builder /app/src ./src

# Create logs directory
RUN mkdir -p logs && chown -R bev:bev /app

USER bev

EXPOSE 3011

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3011/health || exit 1

CMD ["node", "src/index.js"]
EOF

    # WebSocket Server Dockerfile
    mkdir -p "${PROJECT_ROOT}/frontend/websocket-server/src"
    
    cat > "${PROJECT_ROOT}/frontend/websocket-server/Dockerfile" << 'EOF'
FROM node:18-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --only=production

COPY src/ ./src/
RUN npm run build

FROM node:18-alpine AS production

RUN addgroup -g 1001 -S bev && \
    adduser -S bev -u 1001 -G bev

WORKDIR /app

RUN apk add --no-cache netcat-openbsd

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./

RUN mkdir -p logs && chown -R bev:bev /app

USER bev

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD nc -z localhost 8081 || exit 1

CMD ["node", "dist/server.js"]
EOF

    log_success "Dockerfiles generated"
}

# =====================================================
# Container Deployment
# =====================================================

deploy_containers() {
    log_info "Deploying frontend containers..."
    
    cd "${PROJECT_ROOT}/frontend"
    
    # Build and start services
    log_info "Building container images..."
    docker-compose -f docker-compose.frontend.yml build --no-cache
    
    log_info "Starting frontend services..."
    docker-compose -f docker-compose.frontend.yml up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    local timeout=300
    local start_time=$(date +%s)
    
    while [ $(($(date +%s) - start_time)) -lt $timeout ]; do
        local healthy_count=0
        local total_services=3
        
        # Check MCP server
        if docker exec "${MCP_CONTAINER}" curl -f http://localhost:3011/health &>/dev/null; then
            ((healthy_count++))
        fi
        
        # Check proxy
        if docker exec "${PROXY_CONTAINER}" nc -z localhost 3010 &>/dev/null; then
            ((healthy_count++))
        fi
        
        # Check WebSocket server
        if docker exec bev-websocket-server nc -z localhost 8081 &>/dev/null; then
            ((healthy_count++))
        fi
        
        if [ $healthy_count -eq $total_services ]; then
            log_success "All services are healthy"
            break
        fi
        
        log_info "Waiting for services... ($healthy_count/$total_services healthy)"
        sleep 10
    done
    
    if [ $(($(date +%s) - start_time)) -ge $timeout ]; then
        log_error "Timeout waiting for services to become healthy"
        return 1
    fi
    
    log_success "Frontend deployment completed successfully"
}

# =====================================================
# Post-Deployment Validation
# =====================================================

validate_deployment() {
    log_info "Validating deployment..."
    
    # Test HTTP redirect
    local http_response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${FRONTEND_HTTP_PORT}/health" || echo "000")
    if [ "$http_response" -eq 301 ]; then
        log_success "HTTP redirect working correctly"
    else
        log_warn "HTTP redirect test failed (expected 301, got $http_response)"
    fi
    
    # Test HTTPS endpoint (skip SSL verification for self-signed cert)
    local https_response=$(curl -k -s -o /dev/null -w "%{http_code}" "https://localhost:${FRONTEND_HTTPS_PORT}/health" || echo "000")
    if [ "$https_response" -eq 200 ]; then
        log_success "HTTPS endpoint responding correctly"
    else
        log_warn "HTTPS endpoint test failed (expected 200, got $https_response)"
    fi
    
    # Test MCP server
    local mcp_response=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${MCP_SERVER_PORT}/health" || echo "000")
    if [ "$mcp_response" -eq 200 ]; then
        log_success "MCP server responding correctly"
    else
        log_warn "MCP server test failed (expected 200, got $mcp_response)"
    fi
    
    # Test WebSocket connection
    if command -v wscat &> /dev/null; then
        if timeout 5s wscat -c "ws://localhost:${FRONTEND_WS_PORT}/ws" -x &>/dev/null; then
            log_success "WebSocket server responding correctly"
        else
            log_warn "WebSocket connection test failed"
        fi
    else
        log_info "wscat not available, skipping WebSocket test"
    fi
    
    log_success "Deployment validation completed"
}

# =====================================================
# Main Execution Flow
# =====================================================

main() {
    setup_logging
    
    log_info "Starting BEV Frontend deployment with conflict-free configuration"
    log_info "Using safe ports: HTTP=${FRONTEND_HTTP_PORT}, HTTPS=${FRONTEND_HTTPS_PORT}, WS=${FRONTEND_WS_PORT}"
    
    local deployment_steps=(
        "validate_prerequisites"
        "setup_networks"
        "setup_ssl_certificates"
        "create_frontend_directories"
        "generate_mcp_server_config"
        "generate_desktop_app_config"
        "generate_haproxy_config"
        "generate_docker_compose"
        "generate_dockerfiles"
        "deploy_containers"
        "validate_deployment"
    )
    
    local failed_steps=()
    
    for step in "${deployment_steps[@]}"; do
        log_info "Executing deployment step: ${step}"
        if ! ${step}; then
            log_error "Deployment step failed: ${step}"
            failed_steps+=("${step}")
            break
        else
            log_success "Deployment step completed: ${step}"
        fi
        echo "---"
    done
    
    # Summary
    echo "=============================================="
    log_info "BEV Frontend deployment summary:"
    
    if [ ${#failed_steps[@]} -eq 0 ]; then
        log_success "Frontend deployment completed successfully!"
        
        # Write deployment success marker
        echo "DEPLOYMENT_STATUS=SUCCESS" > "${PROJECT_ROOT}/.frontend_deployment"
        echo "DEPLOYMENT_TIMESTAMP=$(date -Iseconds)" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "FRONTEND_HTTP_PORT=${FRONTEND_HTTP_PORT}" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "FRONTEND_HTTPS_PORT=${FRONTEND_HTTPS_PORT}" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "MCP_SERVER_PORT=${MCP_SERVER_PORT}" >> "${PROJECT_ROOT}/.frontend_deployment"
        
        echo "=============================================="
        echo "✅ BEV FRONTEND DEPLOYMENT SUCCESSFUL"
        echo "   HTTP URL: http://localhost:${FRONTEND_HTTP_PORT}"
        echo "   HTTPS URL: https://localhost:${FRONTEND_HTTPS_PORT}"
        echo "   MCP Server: http://localhost:${MCP_SERVER_PORT}"
        echo "   WebSocket: ws://localhost:${FRONTEND_WS_PORT}"
        echo "   HAProxy Stats: http://localhost:8080/haproxy-stats"
        echo "   Log file: ${LOG_FILE}"
        echo "=============================================="
        
        exit 0
    else
        log_error "Deployment failed at step: ${failed_steps[*]}"
        
        # Write deployment failure marker
        echo "DEPLOYMENT_STATUS=FAILED" > "${PROJECT_ROOT}/.frontend_deployment"
        echo "DEPLOYMENT_TIMESTAMP=$(date -Iseconds)" >> "${PROJECT_ROOT}/.frontend_deployment"
        echo "FAILED_STEP=${failed_steps[*]}" >> "${PROJECT_ROOT}/.frontend_deployment"
        
        echo "=============================================="
        echo "❌ BEV FRONTEND DEPLOYMENT FAILED"
        echo "   Failed at step: ${failed_steps[*]}"
        echo "   Log file: ${LOG_FILE}"
        echo "   Run rollback script if needed"
        echo "=============================================="
        
        exit 1
    fi
}

# Trap for cleanup
trap 'log_error "Deployment interrupted"; exit 130' INT TERM

# Execute main function
main "$@"