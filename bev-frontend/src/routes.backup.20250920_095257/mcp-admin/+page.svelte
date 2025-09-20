<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // MCP Server Administration state
  const mcpState = writable({
    server_status: {
      total_servers: 6,
      active_servers: 6,
      healthy_servers: 6,
      failed_servers: 0,
      avg_response_time: '23ms',
      total_requests_24h: 15847,
      success_rate: 99.4
    },
    server_performance: {
      memory_usage: 234.7, // MB
      cpu_usage: 12.4,
      network_io: '147KB/s',
      request_throughput: 234.7,
      error_rate: 0.6,
      connection_pool_size: 156
    },
    security: {
      auth_enabled: true,
      tls_enabled: true,
      rate_limiting: true,
      access_violations: 0,
      security_score: 98.7,
      vulnerability_scans: 12
    },
    configuration: {
      config_files: 18,
      last_update: '4h ago',
      pending_changes: 2,
      config_validation: 'passed',
      deployment_ready: true,
      rollback_available: true
    }
  });

  // MCP Server details
  const mcpServers = [
    {
      name: 'everything',
      port: 3001,
      status: 'healthy',
      description: 'File operations and system utilities',
      requests_24h: 2847,
      avg_response: '15ms'
    },
    {
      name: 'fetch',
      port: 3002,
      status: 'healthy',
      description: 'Web fetching and API integration',
      requests_24h: 1567,
      avg_response: '78ms'
    },
    {
      name: 'git',
      port: 3003,
      status: 'healthy',
      description: 'Git repository management',
      requests_24h: 567,
      avg_response: '45ms'
    },
    {
      name: 'memory',
      port: 3004,
      status: 'healthy',
      description: 'Memory management and caching',
      requests_24h: 3456,
      avg_response: '8ms'
    },
    {
      name: 'sequentialthinking',
      port: 3005,
      status: 'healthy',
      description: 'Sequential reasoning and analysis',
      requests_24h: 1234,
      avg_response: '156ms'
    },
    {
      name: 'time',
      port: 3006,
      status: 'healthy',
      description: 'Time operations and scheduling',
      requests_24h: 234,
      avg_response: '12ms'
    }
  ];

  // MCP management controls
  let selectedServer = '';
  let operationMode = 'status';
  let configEdit = '';
  let serverAction = '';

  // Server configuration
  let serverConfig = {
    auto_restart: true,
    health_check_interval: 30,
    max_connections: 100,
    timeout_seconds: 300,
    log_level: 'INFO',
    security_enabled: true
  };

  // Live data stores
  const serverMetrics = writable({});
  const serverLogs = writable([]);
  const configHistory = writable([]);

  // WebSocket connections
  let mcpAdminWs: WebSocket | null = null;
  let serverMetricsWs: WebSocket | null = null;
  let configWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadMCPData();
    startMetricsCollection();
  });

  onDestroy(() => {
    if (mcpAdminWs) mcpAdminWs.close();
    if (serverMetricsWs) serverMetricsWs.close();
    if (configWs) configWs.close();
  });

  function initializeWebSockets() {
    // MCP admin WebSocket
    mcpAdminWs = new WebSocket('ws://localhost:8120/mcp-admin');
    mcpAdminWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      mcpState.update(state => ({
        ...state,
        ...data
      }));
    };

    // Server metrics WebSocket
    serverMetricsWs = new WebSocket('ws://localhost:8121/mcp-metrics');
    serverMetricsWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      serverMetrics.set(data);
      mcpState.update(state => ({
        ...state,
        server_performance: { ...state.server_performance, ...data.performance }
      }));
    };

    // Configuration WebSocket
    configWs = new WebSocket('ws://localhost:8122/mcp-config');
    configWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      mcpState.update(state => ({
        ...state,
        configuration: { ...state.configuration, ...data }
      }));
    };
  }

  async function loadMCPData() {
    try {
      const [metricsRes, logsRes, configRes] = await Promise.all([
        fetch('http://localhost:8120/api/metrics'),
        fetch('http://localhost:8120/api/logs'),
        fetch('http://localhost:8122/api/config')
      ]);

      const metrics = await metricsRes.json();
      const logs = await logsRes.json();
      const config = await configRes.json();

      serverMetrics.set(metrics);
      serverLogs.set(logs);
      configHistory.set(config);
    } catch (error) {
      console.error('Failed to load MCP data:', error);
    }
  }

  async function startMetricsCollection() {
    setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8121/api/realtime');
        const metrics = await response.json();
        mcpState.update(state => ({
          ...state,
          server_performance: { ...state.server_performance, ...metrics }
        }));
      } catch (error) {
        console.error('MCP metrics collection error:', error);
      }
    }, 5000);
  }

  async function executeServerAction() {
    if (!selectedServer || !serverAction) return;

    try {
      const response = await fetch(`http://localhost:8120/api/server/${selectedServer}/${serverAction}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(serverConfig)
      });

      if (response.ok) {
        console.log(`${serverAction} executed for ${selectedServer}`);
      }
    } catch (error) {
      console.error('Server action failed:', error);
    }
  }

  async function updateServerConfig() {
    if (!selectedServer) return;

    try {
      const response = await fetch(`http://localhost:8122/api/server/${selectedServer}/config`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(serverConfig)
      });

      if (response.ok) {
        console.log('Server configuration updated');
      }
    } catch (error) {
      console.error('Configuration update failed:', error);
    }
  }

  async function performSecurityScan() {
    try {
      const response = await fetch('http://localhost:8120/api/security-scan', {
        method: 'POST'
      });

      if (response.ok) {
        console.log('Security scan initiated');
      }
    } catch (error) {
      console.error('Security scan failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': case 'active': case 'passed': return 'text-green-400';
      case 'warning': case 'degraded': return 'text-yellow-400';
      case 'failed': case 'error': case 'critical': return 'text-red-400';
      case 'pending': case 'updating': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  }

  function getHealthColor(value: number, threshold: number = 95): string {
    if (value >= threshold) return 'text-green-400';
    if (value >= threshold - 10) return 'text-yellow-400';
    return 'text-red-400';
  }
</script>

<svelte:head>
  <title>MCP Server Administration | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-lime-400 to-green-500 bg-clip-text text-transparent">
        MCP Server Administration
      </h1>
      <p class="text-gray-300">Model Context Protocol server management and configuration</p>
    </div>

    {#if $mcpState}
      <!-- MCP Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
        <!-- Server Status -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Server Status
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Total:</span>
              <span class="text-green-400">{$mcpState.server_status.total_servers}</span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-blue-400">{$mcpState.server_status.active_servers}</span>
            </div>
            <div class="flex justify-between">
              <span>Healthy:</span>
              <span class="text-green-400">{$mcpState.server_status.healthy_servers}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class={getHealthColor($mcpState.server_status.success_rate)}>{$mcpState.server_status.success_rate}%</span>
            </div>
          </div>
        </div>

        <!-- Performance -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Performance
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Memory:</span>
              <span class="text-blue-400">{$mcpState.server_performance.memory_usage}MB</span>
            </div>
            <div class="flex justify-between">
              <span>CPU:</span>
              <span class="text-yellow-400">{$mcpState.server_performance.cpu_usage}%</span>
            </div>
            <div class="flex justify-between">
              <span>Throughput:</span>
              <span class="text-green-400">{$mcpState.server_performance.request_throughput}/s</span>
            </div>
            <div class="flex justify-between">
              <span>Error Rate:</span>
              <span class="text-red-400">{$mcpState.server_performance.error_rate}%</span>
            </div>
          </div>
        </div>

        <!-- Security -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Security
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Auth:</span>
              <span class={$mcpState.security.auth_enabled ? 'text-green-400' : 'text-red-400'}>
                {$mcpState.security.auth_enabled ? 'ENABLED' : 'DISABLED'}
              </span>
            </div>
            <div class="flex justify-between">
              <span>TLS:</span>
              <span class={$mcpState.security.tls_enabled ? 'text-green-400' : 'text-red-400'}>
                {$mcpState.security.tls_enabled ? 'ENABLED' : 'DISABLED'}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Security Score:</span>
              <span class={getHealthColor($mcpState.security.security_score)}>{$mcpState.security.security_score}%</span>
            </div>
            <div class="flex justify-between">
              <span>Violations:</span>
              <span class="text-green-400">{$mcpState.security.access_violations}</span>
            </div>
          </div>
        </div>

        <!-- Configuration -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-orange-400 mr-2"></span>
            Configuration
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Config Files:</span>
              <span class="text-orange-400">{$mcpState.configuration.config_files}</span>
            </div>
            <div class="flex justify-between">
              <span>Pending:</span>
              <span class="text-yellow-400">{$mcpState.configuration.pending_changes}</span>
            </div>
            <div class="flex justify-between">
              <span>Validation:</span>
              <span class={getStatusColor($mcpState.configuration.config_validation)}>
                {$mcpState.configuration.config_validation.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Last Update:</span>
              <span class="text-gray-400">{$mcpState.configuration.last_update}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- MCP Server Management -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
        <h3 class="text-lg font-semibold mb-4">MCP Server Management</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {#each mcpServers as server}
            <div class="bg-gray-700 rounded p-4">
              <div class="flex justify-between items-center mb-3">
                <span class="font-medium">{server.name}</span>
                <span class={getStatusColor(server.status)} class="text-xs px-2 py-1 rounded bg-opacity-20">
                  {server.status.toUpperCase()}
                </span>
              </div>

              <div class="text-sm text-gray-300 space-y-1 mb-3">
                <div>Port: {server.port}</div>
                <div>Requests (24h): {server.requests_24h.toLocaleString()}</div>
                <div>Avg Response: {server.avg_response}</div>
              </div>

              <div class="text-xs text-gray-400 mb-3">
                {server.description}
              </div>

              <div class="flex space-x-2">
                <button class="flex-1 bg-blue-600 hover:bg-blue-700 px-2 py-1 rounded text-xs">
                  Restart
                </button>
                <button class="flex-1 bg-green-600 hover:bg-green-700 px-2 py-1 rounded text-xs">
                  Config
                </button>
                <button class="flex-1 bg-yellow-600 hover:bg-yellow-700 px-2 py-1 rounded text-xs">
                  Logs
                </button>
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Server Control Panel -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <!-- Server Operations -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Server Operations</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Target Server</label>
              <select bind:value={selectedServer} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="">Select Server</option>
                {#each mcpServers as server}
                  <option value={server.name}>{server.name}</option>
                {/each}
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Action</label>
              <select bind:value={serverAction} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="">Select Action</option>
                <option value="restart">Restart Server</option>
                <option value="stop">Stop Server</option>
                <option value="start">Start Server</option>
                <option value="reload">Reload Configuration</option>
                <option value="health_check">Health Check</option>
              </select>
            </div>
            <button
              on:click={executeServerAction}
              disabled={!selectedServer || !serverAction}
              class="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded"
            >
              Execute Action
            </button>

            <div class="pt-4 border-t border-gray-600">
              <h4 class="font-medium mb-2">Bulk Operations</h4>
              <div class="space-y-2">
                <button class="w-full bg-blue-600 hover:bg-blue-700 px-3 py-2 rounded text-sm">
                  Restart All Servers
                </button>
                <button class="w-full bg-yellow-600 hover:bg-yellow-700 px-3 py-2 rounded text-sm">
                  Health Check All
                </button>
                <button
                  on:click={performSecurityScan}
                  class="w-full bg-purple-600 hover:bg-purple-700 px-3 py-2 rounded text-sm"
                >
                  Security Scan All
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Configuration Editor -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Configuration Management</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Health Check Interval (s): {serverConfig.health_check_interval}</label>
              <input
                type="range"
                bind:value={serverConfig.health_check_interval}
                min="10"
                max="300"
                step="10"
                class="w-full"
              >
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Max Connections: {serverConfig.max_connections}</label>
              <input
                type="range"
                bind:value={serverConfig.max_connections}
                min="10"
                max="1000"
                step="10"
                class="w-full"
              >
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Timeout (s): {serverConfig.timeout_seconds}</label>
              <input
                type="range"
                bind:value={serverConfig.timeout_seconds}
                min="30"
                max="600"
                step="30"
                class="w-full"
              >
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Log Level</label>
              <select bind:value={serverConfig.log_level} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="DEBUG">Debug</option>
                <option value="INFO">Info</option>
                <option value="WARN">Warning</option>
                <option value="ERROR">Error</option>
              </select>
            </div>
            <div class="space-y-2">
              <label class="flex items-center">
                <input type="checkbox" bind:checked={serverConfig.auto_restart} class="mr-2">
                <span class="text-sm">Auto-Restart on Failure</span>
              </label>
              <label class="flex items-center">
                <input type="checkbox" bind:checked={serverConfig.security_enabled} class="mr-2">
                <span class="text-sm">Enhanced Security</span>
              </label>
            </div>
            <button
              on:click={updateServerConfig}
              class="w-full bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded"
            >
              Update Configuration
            </button>
          </div>
        </div>

        <!-- Performance Metrics -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Real-time Performance</h3>
          <div class="space-y-4">
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>Memory Usage</span>
                <span>{$mcpState.server_performance.memory_usage}MB</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-blue-400 h-2 rounded-full" style="width: {Math.min(($mcpState.server_performance.memory_usage / 1000) * 100, 100)}%"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>CPU Usage</span>
                <span>{$mcpState.server_performance.cpu_usage}%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-yellow-400 h-2 rounded-full" style="width: {$mcpState.server_performance.cpu_usage}%"></div>
              </div>
            </div>
            <div class="text-sm space-y-1">
              <div class="flex justify-between">
                <span>Network I/O:</span>
                <span class="text-cyan-400">{$mcpState.server_performance.network_io}</span>
              </div>
              <div class="flex justify-between">
                <span>Connections:</span>
                <span class="text-green-400">{$mcpState.server_performance.connection_pool_size}</span>
              </div>
              <div class="flex justify-between">
                <span>Requests (24h):</span>
                <span class="text-purple-400">{$mcpState.server_status.total_requests_24h.toLocaleString()}</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Server Request Analytics -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
        <h3 class="text-lg font-semibold mb-4">Server Request Analytics</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div class="bg-gray-700 rounded p-4">
            <h4 class="font-medium mb-2">Request Distribution</h4>
            <div class="h-32 bg-gray-900 rounded flex items-center justify-center">
              <span class="text-gray-400">Request distribution chart</span>
            </div>
          </div>

          <div class="bg-gray-700 rounded p-4">
            <h4 class="font-medium mb-2">Response Times</h4>
            <div class="h-32 bg-gray-900 rounded flex items-center justify-center">
              <span class="text-gray-400">Response time trends</span>
            </div>
          </div>

          <div class="bg-gray-700 rounded p-4">
            <h4 class="font-medium mb-2">Error Patterns</h4>
            <div class="h-32 bg-gray-900 rounded flex items-center justify-center">
              <span class="text-gray-400">Error pattern analysis</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Server Logs -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-lg font-semibold mb-4">MCP Server Logs</h3>
        <div class="bg-gray-900 rounded p-4 h-64 overflow-y-auto font-mono text-sm">
          {#if $serverLogs && $serverLogs.length > 0}
            {#each $serverLogs.slice(0, 50) as log}
              <div class="mb-1">
                <span class="text-gray-500">[{log.timestamp}]</span>
                <span class="text-green-400">{log.server}</span>
                <span class={getStatusColor(log.level)} class="px-1 rounded text-xs">
                  {log.level}
                </span>
                <span class="text-gray-300">{log.message}</span>
              </div>
            {/each}
          {:else}
            <div class="text-gray-400 text-center py-8">
              MCP server logs will appear here
            </div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>