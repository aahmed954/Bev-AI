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
  return `${protocol}://${host}:${port}${path}`;
};

// WebSocket URL builder
export const buildWebSocketUrl = (service: string, port: number, path: string = '/ws'): string => {
  const host = getNodeForService(service);
  const protocol = port === 443 ? 'wss' : 'ws';
  return `${protocol}://${host}:${port}${path}`;
};
