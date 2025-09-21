#!/bin/bash
# MCP Development Setup Script for Starlord Node
# Sets up MCP servers for development and testing

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔗 SETTING UP MCP DEVELOPMENT ENVIRONMENT${NC}"
echo "========================================"

# Navigate to MCP servers directory
cd mcp-servers

# Check Node.js availability
echo -e "${BLUE}📦 Checking Node.js environment...${NC}"
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

# Install MCP server dependencies
echo -e "${BLUE}📚 Installing MCP dependencies...${NC}"
if [ -f "package.json" ]; then
    npm install
    echo -e "${GREEN}✅ MCP dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️ No package.json found, creating basic setup...${NC}"

    # Create basic package.json
    cat > package.json << MCP_EOF
{
  "name": "bev-mcp-servers",
  "version": "1.0.0",
  "description": "BEV MCP Development Servers",
  "scripts": {
    "start": "node index.js",
    "dev": "nodemon index.js",
    "test": "jest"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.0.0",
    "express": "^4.18.0",
    "ws": "^8.0.0"
  },
  "devDependencies": {
    "nodemon": "^3.0.0",
    "jest": "^29.0.0"
  }
}
MCP_EOF

    npm install
fi

# Setup individual MCP servers
echo -e "${BLUE}🛠️ Setting up individual MCP servers...${NC}"

MCP_SERVERS=("everything" "fetch" "git" "memory" "sequentialthinking" "time")
MCP_PORTS=(3001 3002 3003 3004 3005 3006)

for i in "${!MCP_SERVERS[@]}"; do
    SERVER=${MCP_SERVERS[$i]}
    PORT=${MCP_PORTS[$i]}

    echo -n "Setting up MCP $SERVER... "

    if [ -d "src/$SERVER" ]; then
        cd "src/$SERVER"

        # Install server-specific dependencies
        if [ -f "package.json" ]; then
            npm install > /dev/null 2>&1
            echo -e "${GREEN}✅ Ready (port $PORT)${NC}"
        else
            echo -e "${YELLOW}⚠️ No package.json${NC}"
        fi

        cd ../..
    else
        echo -e "${RED}❌ Directory missing${NC}"
    fi
done

# Create MCP development configuration
echo -e "${BLUE}⚙️ Creating MCP development configuration...${NC}"
cat > mcp-dev-config.json << MCP_CONFIG_EOF
{
  "servers": {
    "everything": {
      "port": 3001,
      "description": "File operations and system utilities",
      "capabilities": ["read_file", "write_file", "list_directory", "execute_command"]
    },
    "fetch": {
      "port": 3002,
      "description": "Web fetching and API integration",
      "capabilities": ["http_request", "web_scraping", "api_integration"]
    },
    "git": {
      "port": 3003,
      "description": "Git repository management",
      "capabilities": ["git_operations", "version_control", "repository_management"]
    },
    "memory": {
      "port": 3004,
      "description": "Memory management and caching",
      "capabilities": ["cache_management", "memory_operations", "data_persistence"]
    },
    "sequentialthinking": {
      "port": 3005,
      "description": "Sequential reasoning and analysis",
      "capabilities": ["logical_reasoning", "step_analysis", "problem_solving"]
    },
    "time": {
      "port": 3006,
      "description": "Time operations and scheduling",
      "capabilities": ["datetime_operations", "scheduling", "timezone_conversion"]
    }
  },
  "development": {
    "auto_restart": true,
    "debug_mode": true,
    "log_level": "debug",
    "cors_enabled": true
  }
}\nMCP_CONFIG_EOF\n\n# Create MCP development launcher\necho -e \"${BLUE}\ud83d\ude80 Creating MCP launcher script...${NC}\"\ncat > start_mcp_development.sh << LAUNCHER_EOF\n#!/bin/bash\n# Start all MCP servers for development\n\necho \"Starting MCP development servers...\"\n\n# Start each MCP server in background\ncd src/everything && npm start &\ncd ../fetch && python -m mcp_server_fetch &\ncd ../git && python -m mcp_server_git &\ncd ../memory && npm start &\ncd ../sequentialthinking && npm start &\ncd ../time && python -m mcp_server_time &\n\necho \"All MCP servers started!\"\necho \"Ports: 3001-3006\"\necho \"Use 'pkill -f mcp' to stop all servers\"\nLAUNCHER_EOF\n\nchmod +x start_mcp_development.sh\n\n# Create MCP health check script\ncat > check_mcp_health.sh << HEALTH_EOF\n#!/bin/bash\n# Check health of all MCP servers\n\necho \"MCP Server Health Check\"\necho \"======================\"\n\nfor port in {3001..3006}; do\n    echo -n \"Port $port: \"\n    if curl -s http://localhost:$port/health > /dev/null 2>&1; then\n        echo \"✅ Healthy\"\n    else\n        echo \"❌ Unhealthy\"\n    fi\ndone\nHEALTH_EOF\n\nchmod +x check_mcp_health.sh\n\n# Test MCP server availability (if running)\necho -e \"${BLUE}\ud83d\udd0d Testing MCP server availability...${NC}\"\nMCP_AVAILABLE=0\n\nfor port in {3001..3006}; do\n    echo -n \"Testing port $port... \"\n    if lsof -i:$port > /dev/null 2>&1; then\n        echo -e \"${GREEN}\u2705 In Use${NC}\"\n        MCP_AVAILABLE=$((MCP_AVAILABLE + 1))\n    else\n        echo -e \"${YELLOW}\u26a0\ufe0f Available${NC}\"\n    fi\ndone\n\necho \"\"\necho -e \"${BLUE}\ud83d\udd17 MCP Development Summary:${NC}\"\necho \"Servers configured: ${#MCP_SERVERS[@]}\"\necho \"Ports allocated: 3001-3006\"\necho \"Currently running: $MCP_AVAILABLE servers\"\n\nif [ $MCP_AVAILABLE -gt 0 ]; then\n    echo -e \"${YELLOW}\u26a0\ufe0f Some MCP servers already running${NC}\"\nelse\n    echo -e \"${GREEN}\ud83c\udfaf MCP development environment ready!${NC}\"\nfi\n\n# Create development guide\ncat > MCP_DEVELOPMENT_GUIDE.md << GUIDE_EOF\n# MCP Development Guide\n\n## Quick Start\n\\`\\`\\`bash\n# Start all MCP servers\n./start_mcp_development.sh\n\n# Check server health\n./check_mcp_health.sh\n\n# Stop all servers\npkill -f mcp\n\\`\\`\\`\n\n## Individual Server Management\n\\`\\`\\`bash\n# Start specific server\ncd src/everything && npm start\n\n# Test specific server\ncurl http://localhost:3001/health\n\\`\\`\\`\n\n## Server Ports\n- everything: 3001\n- fetch: 3002\n- git: 3003\n- memory: 3004\n- sequentialthinking: 3005\n- time: 3006\n\n## Development URLs\n- MCP Admin: http://localhost:8120\n- Frontend: http://localhost:5173\n- Documentation: http://localhost:8080\nGUIDE_EOF\n\necho -e \"${GREEN}\ud83c\udfaf MCP development setup complete!${NC}"}