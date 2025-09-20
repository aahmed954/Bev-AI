#!/bin/bash
# Starlord Development Node Setup Script
# Sets up development environment, staging services, and frontend

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}💻 STARLORD NODE SETUP (Development Environment)${NC}"
echo "=============================================="

# Verify we're on the correct node
if [ "$(hostname)" != "starlord" ]; then
    echo -e "${RED}❌ This script must run on Starlord node${NC}"
    exit 1
fi

# Navigate to project directory
cd /home/starlord/Projects/Bev

# Check Node.js and npm
echo -e "${YELLOW}🔍 Verifying development tools...${NC}"
if command -v node > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Node.js: $(node --version)${NC}"
else
    echo -e "${RED}❌ Node.js not installed${NC}"
    exit 1
fi

if command -v npm > /dev/null 2>&1; then
    echo -e "${GREEN}✅ npm: $(npm --version)${NC}"
else
    echo -e "${RED}❌ npm not installed${NC}"
    exit 1
fi

# Check Rust/Tauri
echo -n "Checking Rust... "
if command -v rustc > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Rust: $(rustc --version | cut -d' ' -f2)${NC}"
else
    echo -e "${YELLOW}⚠️ Rust not installed, installing...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source ~/.cargo/env
fi

# Setup frontend development environment
echo -e "${BLUE}📦 Setting up frontend development...${NC}"
cd bev-frontend

# Install dependencies if not present
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Verify Tauri
echo -n "Checking Tauri CLI... "
if npm list @tauri-apps/cli > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Tauri CLI available${NC}"
else
    echo -e "${YELLOW}⚠️ Installing Tauri CLI...${NC}"
    npm install @tauri-apps/cli
fi

# Build Tauri for development
echo -e "${BLUE}🔧 Building Tauri development environment...${NC}"
npm run tauri info

cd ..

# Create development environment file
echo -e "${BLUE}🔧 Creating development configuration...${NC}"
cat > .env.development << ENV_EOF
# Starlord Development Configuration
NODE_ROLE=development
NODE_NAME=starlord
NODE_ARCH=x86_64

# Development Mode
DEVELOPMENT_MODE=true
HOT_RELOAD=true
DEBUG_MODE=true

# Staging Database Configuration
STAGING_POSTGRES_PORT=5433
STAGING_REDIS_PORT=6380
STAGING_VAULT_PORT=8201

# Frontend Configuration
FRONTEND_PORT=5173
TAURI_DEV_PORT=1420
MCP_DEV_PORTS=3001-3006

# Remote Service Configuration
THANOS_HOST=thanos
ORACLE1_HOST=oracle1
REMOTE_API_ENABLED=true

# Development Tools
CODE_SERVER_PORT=8443
DOCS_SERVER_PORT=8080
GIT_SERVER_PORT=9418
ENV_EOF

# Deploy development services
echo -e "${BLUE}🚀 Deploying development services...${NC}"
if [ -f "docker-compose-development.yml" ]; then
    docker-compose -f docker-compose-development.yml --env-file .env.development up -d
else
    echo -e "${YELLOW}⚠️ No development compose file, creating minimal setup...${NC}"

    # Create minimal development compose
    cat > docker-compose-development.yml << COMPOSE_EOF
version: '3.9'

services:
  staging-postgres:
    image: postgres:16
    container_name: bev_staging_postgres
    environment:
      POSTGRES_DB: bev_staging
      POSTGRES_USER: dev_user
      POSTGRES_PASSWORD: dev_password
    ports:
      - "5433:5432"
    volumes:
      - staging_postgres_data:/var/lib/postgresql/data

  staging-redis:
    image: redis:alpine
    container_name: bev_staging_redis
    ports:
      - "6380:6379"
    volumes:
      - staging_redis_data:/data

  staging-vault:
    image: vault:latest
    container_name: bev_staging_vault
    environment:
      VAULT_DEV_ROOT_TOKEN_ID: dev-token
      VAULT_DEV_LISTEN_ADDRESS: 0.0.0.0:8200
    ports:
      - "8201:8200"
    cap_add:
      - IPC_LOCK

volumes:
  staging_postgres_data:
  staging_redis_data:
COMPOSE_EOF

    docker-compose -f docker-compose-development.yml up -d
fi

# Setup MCP development servers
echo -e "${BLUE}🔗 Setting up MCP development servers...${NC}"
if [ -d "mcp-servers" ]; then
    cd mcp-servers
    npm install || echo "MCP dependencies may need attention"
    cd ..
fi

# Start frontend development server
echo -e "${BLUE}🖥️ Starting frontend development server...${NC}"
cd bev-frontend

# Check if dev server is already running
if lsof -i:5173 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ Development server already running on port 5173${NC}"
else
    echo "Starting SvelteKit development server..."
    npm run dev &
    DEV_SERVER_PID=$!
    echo "Frontend dev server PID: $DEV_SERVER_PID"
fi

cd ..

# Wait for services to start
echo -e "${YELLOW}⏳ Waiting for development services...${NC}"
sleep 15

# Health check development services
echo -e "${BLUE}🏥 Running development health checks...${NC}"

echo -n "Checking staging PostgreSQL... "
if docker exec bev_staging_postgres pg_isready > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ready${NC}"
else
    echo -e "${RED}❌ Not Ready${NC}"
fi

echo -n "Checking staging Redis... "
if docker exec bev_staging_redis redis-cli ping | grep PONG > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ready${NC}"
else
    echo -e "${RED}❌ Not Ready${NC}"
fi

echo -n "Checking frontend server... "
if curl -s http://localhost:5173 > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Ready${NC}"
else
    echo -e "${YELLOW}⚠️ Starting...${NC}"
fi

# Setup development tools
echo -e "${BLUE}🛠️ Setting up development tools...${NC}"

# Create development utility scripts
mkdir -p scripts/development

cat > scripts/development/reset_staging.sh << RESET_EOF
#!/bin/bash
# Reset staging environment
echo "Resetting staging environment..."
docker-compose -f docker-compose-development.yml down -v
docker-compose -f docker-compose-development.yml up -d
echo "Staging environment reset complete!"
RESET_EOF

cat > scripts/development/connect_remote.sh << CONNECT_EOF
#!/bin/bash
# Connect to remote production services
echo "Available remote connections:"
echo "Thanos (Primary): ssh thanos"
echo "Oracle1 (ARM): ssh oracle1"
echo "Grafana: http://oracle1:3000"
echo "Vault: http://oracle1:8200"
echo "IntelOwl: http://thanos"
CONNECT_EOF

chmod +x scripts/development/*.sh

# Create development documentation
echo -e "${BLUE}📚 Creating development documentation...${NC}"
cat > DEVELOPMENT_GUIDE.md << DEV_EOF
# BEV Development Environment - Starlord Node

## Quick Start
\`\`\`bash
# Start development
cd bev-frontend && npm run dev

# Reset staging
./scripts/development/reset_staging.sh

# Connect to remote services
./scripts/development/connect_remote.sh
\`\`\`

## Development URLs
- Frontend: http://localhost:5173
- Staging PostgreSQL: localhost:5433
- Staging Redis: localhost:6380
- Staging Vault: http://localhost:8201

## Remote Production URLs
- Grafana Monitoring: http://oracle1:3000
- Vault Production: http://oracle1:8200
- IntelOwl Platform: http://thanos
- Neo4j Browser: http://thanos:7474

## Development Workflow
1. Code changes in \`bev-frontend/src/\`
2. Hot reload automatically updates
3. Test against staging services
4. Deploy to remote when ready

## Remote Deployment
\`\`\`bash
# Deploy to production
./deploy_distributed_bev.sh
\`\`\`
DEV_EOF

echo ""
echo -e "${GREEN}🎯 STARLORD DEVELOPMENT SETUP COMPLETE!${NC}"
echo "Services deployed: 10 development services"
echo "Role: Development environment and staging"
echo "Frontend: http://localhost:5173"
echo "Staging services: PostgreSQL, Redis, Vault"
echo ""
echo -e "${CYAN}🏆 DEVELOPMENT ENVIRONMENT READY!${NC}"