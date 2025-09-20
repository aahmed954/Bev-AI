// BEV Service Endpoint Configuration
// Supports distributed deployment across Thanos, Oracle1, and Starlord

// Environment-based service discovery
const getServiceHost = (service: string): string => {
  // Check if we have environment-specific overrides
  if (typeof window !== 'undefined') {
    // Browser environment - check for injected config
    const config = (window as any).__BEV_CONFIG__;
    if (config && config.serviceHosts && config.serviceHosts[service]) {
      return config.serviceHosts[service];
    }
  }

  // Default to distributed node configuration based on service type
  switch (service) {
    // High-compute services → Thanos
    case 'autonomous':
    case 'adaptive-learning':
    case 'knowledge-evolution':
    case 'extended-reasoning':
    case 'context-compression':
    case 't2v-transformers':
    case 'postgres':
    case 'neo4j':
    case 'elasticsearch':
    case 'influxdb':
    case 'kafka':
    case 'rabbitmq':
    case 'intelowl':
      return 'thanos';

    // Monitoring and security services → Oracle1
    case 'prometheus':
    case 'grafana':
    case 'vault':
    case 'consul':
    case 'tor':
    case 'redis':
    case 'alert-manager':
    case 'monitoring':
    case 'security':
      return 'oracle1';

    // Development services → localhost (Starlord)
    case 'frontend':
    case 'development':
    case 'staging':
    case 'mcp':
      return 'localhost';

    // Default to localhost for unknown services
    default:
      return 'localhost';
  }
};

// Service endpoint configuration
export const endpoints = {
  // Core OSINT Services (Thanos)
  mcp_server: `http://${getServiceHost('mcp')}:3010`,
  intelowl: `http://${getServiceHost('intelowl')}`,

  // Databases (Thanos)
  postgres: `http://${getServiceHost('postgres')}:5432`,
  neo4j: `http://${getServiceHost('neo4j')}:7474`,
  elasticsearch: `http://${getServiceHost('elasticsearch')}:9200`,
  influxdb: `http://${getServiceHost('influxdb')}:8086`,

  // AI/ML Services (Thanos)
  autonomous_coordinator: `http://${getServiceHost('autonomous')}:8009`,
  adaptive_learning: `http://${getServiceHost('adaptive-learning')}:8010`,
  resource_manager: `http://${getServiceHost('autonomous')}:8011`,
  knowledge_evolution: `http://${getServiceHost('knowledge-evolution')}:8012`,
  extended_reasoning: `http://${getServiceHost('extended-reasoning')}:8081`,
  context_compression: `http://${getServiceHost('context-compression')}:8080`,

  // Monitoring Services (Oracle1)
  prometheus: `http://${getServiceHost('prometheus')}:9090`,
  grafana: `http://${getServiceHost('grafana')}:3000`,
  vault: `http://${getServiceHost('vault')}:8200`,

  // Security Services (Oracle1)
  tor: `http://${getServiceHost('tor')}:9050`,
  security_ops: `http://${getServiceHost('security')}:8004`,

  // Development Services (Starlord)
  staging_postgres: `http://${getServiceHost('staging')}:5433`,
  staging_redis: `http://${getServiceHost('staging')}:6380`,
  staging_vault: `http://${getServiceHost('staging')}:8201`,
};

// WebSocket endpoint configuration
export const websockets = {
  // Core Services (Thanos)
  mcp_stream: `ws://${getServiceHost('mcp')}:3010/ws`,

  // AI/ML Services (Thanos)
  autonomous: `ws://${getServiceHost('autonomous')}:8009/ws`,
  adaptive_learning: `ws://${getServiceHost('adaptive-learning')}:8010/ws`,
  resource_manager: `ws://${getServiceHost('autonomous')}:8011/ws`,
  knowledge_evolution: `ws://${getServiceHost('knowledge-evolution')}:8012/ws`,

  // Monitoring Services (Oracle1)
  prometheus_stream: `ws://${getServiceHost('prometheus')}:9090/metrics-stream`,
  grafana_stream: `ws://${getServiceHost('grafana')}:3000/api/live/ws`,

  // Logging Services (Oracle1)
  log_stream: `ws://${getServiceHost('monitoring')}:8110/logs/stream`,
  log_search: `ws://${getServiceHost('monitoring')}:8111/logs/search`,
  log_correlation: `ws://${getServiceHost('monitoring')}:8112/logs/correlation`,
  log_alerts: `ws://${getServiceHost('monitoring')}:8113/logs/alerts`,

  // Development Services (Starlord)
  development: `ws://${getServiceHost('development')}:5173/ws`,
};

// Service health check endpoints
export const healthChecks = {
  thanos: `http://thanos:9090/-/healthy`,
  oracle1: `http://oracle1:9090/-/healthy`,
  starlord: `http://localhost:5173/health`,
};

// Cross-node service discovery
export const serviceDiscovery = {
  thanos_services: [
    'autonomous-coordinator', 'adaptive-learning', 'knowledge-evolution',
    'postgres', 'neo4j', 'elasticsearch', 'influxdb', 'kafka', 'rabbitmq'
  ],
  oracle1_services: [
    'prometheus', 'grafana', 'vault', 'consul', 'redis', 'tor',
    'security-ops', 'monitoring', 'alerts'
  ],
  starlord_services: [
    'frontend', 'staging-postgres', 'staging-redis', 'mcp-servers'
  ]
};

// Dynamic endpoint resolution for environment
export const getEndpoint = (service: string, fallback: string = 'localhost'): string => {
  const host = getServiceHost(service);
  return endpoints[service] || `http://${host}:8000`;
};

export const getWebSocket = (service: string, fallback: string = 'localhost'): string => {
  const host = getServiceHost(service);
  return websockets[service] || `ws://${host}:8000/ws`;
};

// Service port mappings for distributed deployment
export const servicePorts = {
  // Thanos (High-compute services)
  thanos: {
    postgres: 5432,
    neo4j_http: 7474,
    neo4j_bolt: 7687,
    elasticsearch: 9200,
    influxdb: 8086,
    kafka1: 9092,
    kafka2: 9093,
    kafka3: 9094,
    rabbitmq1: 5672,
    rabbitmq2: 5673,
    rabbitmq3: 5674,
    autonomous_coordinator: 8009,
    adaptive_learning: 8010,
    resource_manager: 8011,
    knowledge_evolution: 8012,
    extended_reasoning: 8081,
    context_compression: 8080,
    intelowl: 80,
  },

  // Oracle1 (ARM services)
  oracle1: {
    prometheus: 9090,
    grafana: 3000,
    vault: 8200,
    consul: 8500,
    redis: 6379,
    tor: 9050,
    security_ops: 8004,
    monitoring: 8017,
    alerts: 8018,
    log_aggregation: 8110,
    log_search: 8111,
    log_correlation: 8112,
    log_alerts: 8113,
  },

  // Starlord (Development services)
  starlord: {
    frontend: 5173,
    tauri: 1420,
    staging_postgres: 5433,
    staging_redis: 6380,
    staging_vault: 8201,
    mcp_everything: 3001,
    mcp_fetch: 3002,
    mcp_git: 3003,
    mcp_memory: 3004,
    mcp_sequential: 3005,
    mcp_time: 3006,
    docs_server: 8080,
  }
};

export default endpoints;