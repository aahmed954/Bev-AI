#!/bin/bash
# Update Frontend Components for Distributed Deployment
# Replaces hardcoded localhost references with distributed endpoint configuration

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🔗 UPDATING FRONTEND FOR DISTRIBUTED DEPLOYMENT${NC}"
echo -e "${BLUE}===============================================${NC}"

# Navigate to frontend directory
cd bev-frontend/src

# Create backup
echo -e "${YELLOW}📦 Creating backup of current frontend...${NC}"
cp -r routes routes.backup.$(date +%Y%m%d_%H%M%S)

# Update all route files to use distributed endpoints
echo -e "${BLUE}🔧 Updating route files for distributed endpoints...${NC}"

# Find all files with localhost references
LOCALHOST_FILES=$(grep -r "localhost\|127.0.0.1" routes/ --include="*.svelte" | cut -d: -f1 | sort -u)

echo "Files with localhost references:"
echo "$LOCALHOST_FILES"
echo ""

# Update each file
UPDATED_FILES=0
for file in $LOCALHOST_FILES; do
    echo -n "Updating $file... "

    # Create temporary updated file
    sed \
        -e "s|ws://localhost:|ws://\${getWebSocketHost()}:|g" \
        -e "s|http://localhost:|http://\${getServiceHost()}:|g" \
        -e "s|'ws://localhost|'ws://' + getWebSocketHost() + '|g" \
        -e "s|'http://localhost|'http://' + getServiceHost() + '|g" \
        "$file" > "$file.tmp"

    # Add import for endpoint configuration at the top of the script section
    if grep -q "<script" "$file"; then
        # Add import after script tag
        sed -i '/<script/a\  import { endpoints, websockets, getEndpoint, getWebSocket } from "$lib/config/endpoints";' "$file.tmp"

        # Add helper functions
        sed -i '/import.*endpoints/a\\n  // Distributed endpoint helpers\n  const getServiceHost = () => {\n    const service = typeof window !== "undefined" && window.location.hostname;\n    return service === "localhost" ? "localhost" : service;\n  };\n\n  const getWebSocketHost = () => {\n    const service = typeof window !== "undefined" && window.location.hostname;\n    return service === "localhost" ? "localhost" : service;\n  };' "$file.tmp"
    fi

    # Replace original file
    mv "$file.tmp" "$file"
    echo -e "${GREEN}✅ Updated${NC}"
    UPDATED_FILES=$((UPDATED_FILES + 1))
done

echo ""
echo -e "${GREEN}✅ Updated $UPDATED_FILES files for distributed deployment${NC}"

# Create distributed environment configuration
echo -e "${BLUE}⚙️ Creating distributed environment configuration...${NC}"

cat > lib/config/distributed.ts << DIST_EOF
// Distributed Environment Configuration
// Handles service discovery across Thanos, Oracle1, and Starlord nodes

export interface NodeConfig {
  name: string;
  host: string;
  services: string[];
  primary: boolean;
}

export const nodeConfiguration: Record<string, NodeConfig> = {
  thanos: {
    name: 'Thanos',
    host: 'thanos',
    services: [
      'postgres', 'neo4j', 'elasticsearch', 'influxdb',
      'kafka', 'rabbitmq', 'autonomous-coordinator',
      'adaptive-learning', 'knowledge-evolution', 'intelowl'
    ],
    primary: true
  },
  oracle1: {
    name: 'Oracle1',
    host: 'oracle1',
    services: [
      'prometheus', 'grafana', 'vault', 'consul', 'redis',
      'tor', 'security-ops', 'monitoring', 'alerts'
    ],
    primary: true
  },
  starlord: {
    name: 'Starlord',
    host: 'localhost',
    services: [
      'frontend', 'staging-postgres', 'staging-redis',
      'mcp-servers', 'development'
    ],
    primary: false
  }
};

// Dynamic service discovery based on current environment
export const getNodeForService = (service: string): string => {
  for (const [nodeName, nodeConfig] of Object.entries(nodeConfiguration)) {
    if (nodeConfig.services.includes(service)) {
      return nodeConfig.host;
    }
  }
  return 'localhost'; // fallback
};

// Environment detection
export const getCurrentEnvironment = (): 'development' | 'staging' | 'production' => {
  if (typeof window !== 'undefined') {
    const hostname = window.location.hostname;
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      return 'development';
    }
    if (hostname.includes('staging')) {
      return 'staging';
    }
    return 'production';
  }
  return 'development';
};

// Service URL builder
export const buildServiceUrl = (service: string, port: number, path: string = ''): string => {
  const host = getNodeForService(service);
  const protocol = port === 443 ? 'https' : 'http';
  return \`\${protocol}://\${host}:\${port}\${path}\`;
};

// WebSocket URL builder
export const buildWebSocketUrl = (service: string, port: number, path: string = '/ws'): string => {
  const host = getNodeForService(service);
  const protocol = port === 443 ? 'wss' : 'ws';
  return \`\${protocol}://\${host}:\${port}\${path}\`;
};
DIST_EOF

echo -e "${GREEN}✅ Distributed configuration created${NC}"

# Update main layout to inject configuration
echo -e "${BLUE}🌐 Updating main layout for distributed configuration...${NC}"

if [ -f "routes/+layout.svelte" ]; then
    # Add distributed configuration injection
    cat >> routes/+layout.svelte << LAYOUT_EOF

<script lang="ts">
  import { nodeConfiguration, getCurrentEnvironment } from '\$lib/config/distributed';
  import { onMount } from 'svelte';

  onMount(() => {
    // Inject distributed configuration into global scope
    if (typeof window !== 'undefined') {
      (window as any).__BEV_CONFIG__ = {
        environment: getCurrentEnvironment(),
        nodeConfiguration,
        currentNode: window.location.hostname,
        serviceHosts: Object.fromEntries(
          Object.entries(nodeConfiguration).flatMap(([nodeName, config]) =>
            config.services.map(service => [service, config.host])
          )
        )
      };
    }
  });
</script>
LAYOUT_EOF

    echo -e "${GREEN}✅ Layout updated with distributed configuration${NC}"
fi

# Create network verification script
echo -e "${BLUE}🔍 Creating network connectivity verification...${NC}"

cat > ../scripts/verify_cross_node_connectivity.sh << VERIFY_EOF
#!/bin/bash
# Verify cross-node connectivity for distributed deployment

echo "🔗 Verifying Cross-Node Connectivity"
echo "=================================="

# Test Thanos connectivity
echo -n "Thanos connectivity: "
if ping -c 1 thanos > /dev/null 2>&1; then
    echo "✅ Reachable"
else
    echo "❌ Unreachable"
fi

# Test Oracle1 connectivity
echo -n "Oracle1 connectivity: "
if ping -c 1 oracle1 > /dev/null 2>&1; then
    echo "✅ Reachable"
else
    echo "❌ Unreachable"
fi

# Test service ports
echo ""
echo "Testing service ports:"

# Thanos services
THANOS_PORTS=(5432 7474 9200 8086 9092 8009 8010)
for port in "\${THANOS_PORTS[@]}"; do
    echo -n "Thanos:\$port: "
    if timeout 3 bash -c "echo >/dev/tcp/thanos/\$port" 2>/dev/null; then
        echo "✅ Open"
    else
        echo "❌ Closed"
    fi
done

# Oracle1 services
ORACLE1_PORTS=(9090 3000 8200 6379 9050)
for port in "\${ORACLE1_PORTS[@]}"; do
    echo -n "Oracle1:\$port: "
    if timeout 3 bash -c "echo >/dev/tcp/oracle1/\$port" 2>/dev/null; then
        echo "✅ Open"
    else
        echo "❌ Closed"
    fi
done

echo ""
echo "Cross-node connectivity verification complete!"
VERIFY_EOF

chmod +x ../scripts/verify_cross_node_connectivity.sh

echo ""
echo -e "${GREEN}🎯 FRONTEND DISTRIBUTED DEPLOYMENT UPDATE COMPLETE!${NC}"
echo ""
echo "Changes made:"
echo "• Updated $UPDATED_FILES route files for distributed endpoints"
echo "• Created distributed endpoint configuration system"
echo "• Added environment-based service discovery"
echo "• Created cross-node connectivity verification"
echo ""
echo "Next steps:"
echo "1. Test frontend with: npm run dev"
echo "2. Verify connectivity: ../scripts/verify_cross_node_connectivity.sh"
echo "3. Deploy distributed: ../deploy_distributed_bev.sh"