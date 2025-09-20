<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // API Management state
  const apiState = writable({
    gateway: {
      status: 'active',
      total_apis: 147,
      active_endpoints: 1284,
      requests_per_second: 2847,
      success_rate: 99.2,
      avg_response_time: 34,
      bandwidth_usage: 2.8,
      cached_responses: 78.4
    },
    authentication: {
      total_tokens: 892,
      active_sessions: 156,
      auth_methods: ['JWT', 'OAuth2', 'API_KEY', 'SAML'],
      token_expiry_rate: 2.3,
      failed_auth_rate: 0.7,
      multi_factor_enabled: 89.2
    },
    rate_limiting: {
      policies_active: 34,
      requests_throttled: 127,
      burst_protection: true,
      adaptive_limits: true,
      ddos_protection: 'active',
      compliance_score: 96.8
    },
    monitoring: {
      health_checks: 'passing',
      alerts_active: 3,
      uptime: 99.97,
      sla_compliance: 99.1,
      error_tracking: 'enabled',
      performance_baseline: 97.3
    },
    documentation: {
      apis_documented: 142,
      coverage_score: 96.6,
      auto_generation: true,
      interactive_docs: true,
      sdk_generation: 'enabled',
      version_tracking: true
    }
  });

  // API Management workspace
  let activeSection = 'gateway';
  let selectedAPI = null;
  let selectedEndpoint = null;

  // API Gateway Configuration
  let gatewayConfig = {
    load_balancing: 'round_robin',
    circuit_breaker: true,
    timeout: 30000,
    retry_policy: 'exponential_backoff',
    compression: true,
    cors_enabled: true
  };

  // Rate Limiting Configuration
  let rateLimitConfig = {
    policy_name: '',
    requests_per_minute: 1000,
    burst_limit: 100,
    time_window: 60,
    throttle_action: 'block',
    whitelist_ips: [],
    blacklist_ips: []
  };

  // Authentication Configuration
  let authConfig = {
    method: 'JWT',
    token_expiry: 3600,
    refresh_enabled: true,
    multi_factor: false,
    session_timeout: 1800,
    password_policy: 'strong'
  };

  // API Lists
  const registeredAPIs = writable([]);
  const apiEndpoints = writable([]);
  const rateLimitPolicies = writable([]);
  const authTokens = writable([]);

  // WebSocket connections
  let gatewayWs: WebSocket | null = null;
  let monitoringWs: WebSocket | null = null;
  let alertsWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadAPIs();
    loadEndpoints();
    loadRateLimitPolicies();
    startMetricsCollection();
  });

  onDestroy(() => {
    if (gatewayWs) gatewayWs.close();
    if (monitoringWs) monitoringWs.close();
    if (alertsWs) alertsWs.close();
  });

  function initializeWebSockets() {
    // API Gateway WebSocket
    gatewayWs = new WebSocket('ws://localhost:8020/gateway');
    gatewayWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      apiState.update(state => ({
        ...state,
        gateway: { ...state.gateway, ...data.gateway }
      }));
    };

    // Monitoring WebSocket
    monitoringWs = new WebSocket('ws://localhost:8021/monitoring');
    monitoringWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      apiState.update(state => ({
        ...state,
        monitoring: { ...state.monitoring, ...data }
      }));
    };

    // Alerts WebSocket
    alertsWs = new WebSocket('ws://localhost:8022/alerts');
    alertsWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleAlert(data);
    };
  }

  async function loadAPIs() {
    try {
      const response = await fetch('http://localhost:8020/api/registry');
      const apis = await response.json();
      registeredAPIs.set(apis);
    } catch (error) {
      console.error('Failed to load APIs:', error);
    }
  }

  async function loadEndpoints() {
    try {
      const response = await fetch('http://localhost:8020/api/endpoints');
      const endpoints = await response.json();
      apiEndpoints.set(endpoints);
    } catch (error) {
      console.error('Failed to load endpoints:', error);
    }
  }

  async function loadRateLimitPolicies() {
    try {
      const response = await fetch('http://localhost:8021/api/rate-limits');
      const policies = await response.json();
      rateLimitPolicies.set(policies);
    } catch (error) {
      console.error('Failed to load rate limit policies:', error);
    }
  }

  async function startMetricsCollection() {
    setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8020/api/metrics');
        const metrics = await response.json();
        apiState.update(state => ({
          ...state,
          ...metrics
        }));
      } catch (error) {
        console.error('Metrics collection error:', error);
      }
    }, 5000);
  }

  async function updateGatewayConfig() {
    try {
      const response = await fetch('http://localhost:8020/api/gateway/configure', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gatewayConfig)
      });

      if (response.ok) {
        console.log('Gateway configuration updated');
      }
    } catch (error) {
      console.error('Gateway configuration failed:', error);
    }
  }

  async function createRateLimitPolicy() {
    if (!rateLimitConfig.policy_name) return;

    try {
      const response = await fetch('http://localhost:8021/api/rate-limits', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rateLimitConfig)
      });

      if (response.ok) {
        const policy = await response.json();
        rateLimitPolicies.update(policies => [...policies, policy]);

        // Reset form
        rateLimitConfig = {
          policy_name: '',
          requests_per_minute: 1000,
          burst_limit: 100,
          time_window: 60,
          throttle_action: 'block',
          whitelist_ips: [],
          blacklist_ips: []
        };
      }
    } catch (error) {
      console.error('Rate limit policy creation failed:', error);
    }
  }

  async function updateAuthConfig() {
    try {
      const response = await fetch('http://localhost:8022/api/auth/configure', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(authConfig)
      });

      if (response.ok) {
        console.log('Authentication configuration updated');
      }
    } catch (error) {
      console.error('Authentication configuration failed:', error);
    }
  }

  async function generateAPIDocumentation() {
    try {
      const response = await fetch('http://localhost:8023/api/docs/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ auto_generate: true })
      });

      if (response.ok) {
        console.log('API documentation generation started');
      }
    } catch (error) {
      console.error('Documentation generation failed:', error);
    }
  }

  async function generateSDK(language: string) {
    try {
      const response = await fetch('http://localhost:8023/api/sdk/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ language, include_examples: true })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('SDK generation started:', result);
      }
    } catch (error) {
      console.error('SDK generation failed:', error);
    }
  }

  function handleAlert(alert: any) {
    // Handle real-time alerts
    console.log('API Alert:', alert);
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      case 'passing': return 'text-green-400';
      case 'enabled': return 'text-blue-400';
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
  <title>Phase 9 - API Management Platform | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-green-400 to-blue-500 bg-clip-text text-transparent">
        Phase 9 - API Management Platform
      </h1>
      <p class="text-gray-300">Enterprise API gateway, security, and documentation platform</p>
    </div>

    {#if $apiState}
      <!-- API Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-5 gap-6 mb-8">
        <!-- Gateway Status -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            API Gateway
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($apiState.gateway.status)}>
                {$apiState.gateway.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>APIs:</span>
              <span class="text-blue-400">{$apiState.gateway.total_apis}</span>
            </div>
            <div class="flex justify-between">
              <span>Endpoints:</span>
              <span class="text-green-400">{$apiState.gateway.active_endpoints.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>RPS:</span>
              <span class="text-yellow-400">{$apiState.gateway.requests_per_second.toLocaleString()}</span>
            </div>
          </div>
        </div>

        <!-- Authentication -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Authentication
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Tokens:</span>
              <span class="text-blue-400">{$apiState.authentication.total_tokens}</span>
            </div>
            <div class="flex justify-between">
              <span>Sessions:</span>
              <span class="text-green-400">{$apiState.authentication.active_sessions}</span>
            </div>
            <div class="flex justify-between">
              <span>MFA Rate:</span>
              <span class="text-purple-400">{$apiState.authentication.multi_factor_enabled}%</span>
            </div>
            <div class="flex justify-between">
              <span>Fail Rate:</span>
              <span class="text-yellow-400">{$apiState.authentication.failed_auth_rate}%</span>
            </div>
          </div>
        </div>

        <!-- Rate Limiting -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-yellow-400 mr-2"></span>
            Rate Limiting
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Policies:</span>
              <span class="text-yellow-400">{$apiState.rate_limiting.policies_active}</span>
            </div>
            <div class="flex justify-between">
              <span>Throttled:</span>
              <span class="text-red-400">{$apiState.rate_limiting.requests_throttled}</span>
            </div>
            <div class="flex justify-between">
              <span>DDoS Protection:</span>
              <span class={getStatusColor($apiState.rate_limiting.ddos_protection)}>
                {$apiState.rate_limiting.ddos_protection.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Compliance:</span>
              <span class="text-green-400">{$apiState.rate_limiting.compliance_score}%</span>
            </div>
          </div>
        </div>

        <!-- Monitoring -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Monitoring
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Health:</span>
              <span class={getStatusColor($apiState.monitoring.health_checks)}>
                {$apiState.monitoring.health_checks.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Alerts:</span>
              <span class="text-red-400">{$apiState.monitoring.alerts_active}</span>
            </div>
            <div class="flex justify-between">
              <span>Uptime:</span>
              <span class={getHealthColor($apiState.monitoring.uptime)}>{$apiState.monitoring.uptime}%</span>
            </div>
            <div class="flex justify-between">
              <span>SLA:</span>
              <span class={getHealthColor($apiState.monitoring.sla_compliance)}>{$apiState.monitoring.sla_compliance}%</span>
            </div>
          </div>
        </div>

        <!-- Documentation -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-cyan-400 mr-2"></span>
            Documentation
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Documented:</span>
              <span class="text-cyan-400">{$apiState.documentation.apis_documented}</span>
            </div>
            <div class="flex justify-between">
              <span>Coverage:</span>
              <span class={getHealthColor($apiState.documentation.coverage_score)}>{$apiState.documentation.coverage_score}%</span>
            </div>
            <div class="flex justify-between">
              <span>Auto-Gen:</span>
              <span class={getStatusColor($apiState.documentation.auto_generation ? 'enabled' : 'disabled')}>
                {$apiState.documentation.auto_generation ? 'ENABLED' : 'DISABLED'}
              </span>
            </div>
            <div class="flex justify-between">
              <span>SDK Gen:</span>
              <span class={getStatusColor($apiState.documentation.sdk_generation)}>
                {$apiState.documentation.sdk_generation.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Management Workspace Tabs -->
      <div class="mb-6">
        <div class="flex space-x-1 bg-gray-800 p-1 rounded-lg">
          {#each ['gateway', 'authentication', 'rate_limiting', 'monitoring', 'documentation'] as section}
            <button
              on:click={() => activeSection = section}
              class="flex-1 py-2 px-4 rounded text-center transition-colors text-sm
                {activeSection === section
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'}"
            >
              {section.charAt(0).toUpperCase() + section.slice(1).replace('_', ' ')}
            </button>
          {/each}
        </div>
      </div>

      <!-- Workspace Content -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {#if activeSection === 'gateway'}
          <!-- Gateway Configuration -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">API Gateway Configuration</h3>
            <div class="space-y-4">
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Load Balancing</label>
                  <select bind:value={gatewayConfig.load_balancing} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="round_robin">Round Robin</option>
                    <option value="least_connections">Least Connections</option>
                    <option value="weighted_round_robin">Weighted Round Robin</option>
                    <option value="ip_hash">IP Hash</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Retry Policy</label>
                  <select bind:value={gatewayConfig.retry_policy} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="immediate">Immediate</option>
                    <option value="linear_backoff">Linear Backoff</option>
                    <option value="exponential_backoff">Exponential Backoff</option>
                    <option value="no_retry">No Retry</option>
                  </select>
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Timeout (ms): {gatewayConfig.timeout}</label>
                <input
                  type="range"
                  bind:value={gatewayConfig.timeout}
                  min="1000"
                  max="60000"
                  step="1000"
                  class="w-full"
                >
              </div>
              <div class="grid grid-cols-3 gap-4">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={gatewayConfig.circuit_breaker} class="mr-2">
                  <span class="text-sm">Circuit Breaker</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={gatewayConfig.compression} class="mr-2">
                  <span class="text-sm">Compression</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={gatewayConfig.cors_enabled} class="mr-2">
                  <span class="text-sm">CORS Enabled</span>
                </label>
              </div>
              <button
                on:click={updateGatewayConfig}
                class="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
              >
                Update Gateway Configuration
              </button>
            </div>
          </div>

          <!-- Gateway Metrics -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Real-time Metrics</h3>
            <div class="space-y-4">
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span>Success Rate</span>
                  <span>{$apiState.gateway.success_rate}%</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2">
                  <div class="bg-green-400 h-2 rounded-full" style="width: {$apiState.gateway.success_rate}%"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span>Cache Hit Rate</span>
                  <span>{$apiState.gateway.cached_responses}%</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2">
                  <div class="bg-blue-400 h-2 rounded-full" style="width: {$apiState.gateway.cached_responses}%"></div>
                </div>
              </div>
              <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                  <span>Avg Response Time:</span>
                  <span class="text-cyan-400">{$apiState.gateway.avg_response_time}ms</span>
                </div>
                <div class="flex justify-between">
                  <span>Bandwidth Usage:</span>
                  <span class="text-yellow-400">{$apiState.gateway.bandwidth_usage}GB/h</span>
                </div>
                <div class="flex justify-between">
                  <span>Requests/sec:</span>
                  <span class="text-green-400">{$apiState.gateway.requests_per_second.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>

        {:else if activeSection === 'authentication'}
          <!-- Authentication Configuration -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Authentication Configuration</h3>
            <div class="space-y-4">
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Authentication Method</label>
                  <select bind:value={authConfig.method} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="JWT">JWT (JSON Web Token)</option>
                    <option value="OAuth2">OAuth 2.0</option>
                    <option value="API_KEY">API Key</option>
                    <option value="SAML">SAML</option>
                    <option value="BASIC">Basic Auth</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Password Policy</label>
                  <select bind:value={authConfig.password_policy} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="weak">Weak</option>
                    <option value="medium">Medium</option>
                    <option value="strong">Strong</option>
                    <option value="enterprise">Enterprise</option>
                  </select>
                </div>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Token Expiry (seconds)</label>
                  <input
                    type="number"
                    bind:value={authConfig.token_expiry}
                    min="300"
                    max="86400"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Session Timeout (seconds)</label>
                  <input
                    type="number"
                    bind:value={authConfig.session_timeout}
                    min="600"
                    max="7200"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={authConfig.refresh_enabled} class="mr-2">
                  <span class="text-sm">Token Refresh Enabled</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={authConfig.multi_factor} class="mr-2">
                  <span class="text-sm">Multi-Factor Authentication</span>
                </label>
              </div>
              <button
                on:click={updateAuthConfig}
                class="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
              >
                Update Authentication
              </button>
            </div>
          </div>

          <!-- Active Sessions -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Active Sessions</h3>
            <div class="space-y-3">
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Admin Session</span>
                  <span class="text-green-400">Active</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Method: JWT</div>
                  <div>Expires: 14:32:45</div>
                </div>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">API Client</span>
                  <span class="text-blue-400">API Key</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Rate Limit: 1000/min</div>
                  <div>Usage: 247 requests</div>
                </div>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">OAuth App</span>
                  <span class="text-purple-400">OAuth2</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Scope: read, write</div>
                  <div>Last Access: 2m ago</div>
                </div>
              </div>
            </div>
          </div>

        {:else if activeSection === 'rate_limiting'}
          <!-- Rate Limiting Configuration -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Rate Limiting Policies</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Policy Name</label>
                <input
                  bind:value={rateLimitConfig.policy_name}
                  placeholder="Enter policy name..."
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
              <div class="grid grid-cols-3 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Requests/Min</label>
                  <input
                    type="number"
                    bind:value={rateLimitConfig.requests_per_minute}
                    min="1"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Burst Limit</label>
                  <input
                    type="number"
                    bind:value={rateLimitConfig.burst_limit}
                    min="1"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Time Window (s)</label>
                  <input
                    type="number"
                    bind:value={rateLimitConfig.time_window}
                    min="1"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Throttle Action</label>
                <select bind:value={rateLimitConfig.throttle_action} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="block">Block Request</option>
                  <option value="delay">Delay Response</option>
                  <option value="queue">Queue Request</option>
                  <option value="redirect">Redirect</option>
                </select>
              </div>
              <button
                on:click={createRateLimitPolicy}
                disabled={!rateLimitConfig.policy_name}
                class="w-full bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                Create Rate Limit Policy
              </button>
            </div>
          </div>

          <!-- Active Policies -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Active Policies</h3>
            <div class="space-y-3">
              {#if $rateLimitPolicies && $rateLimitPolicies.length > 0}
                {#each $rateLimitPolicies as policy}
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">{policy.name}</span>
                      <span class="text-green-400">{policy.status}</span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Limit: {policy.requests_per_minute}/min</div>
                      <div>Burst: {policy.burst_limit}</div>
                      <div>Action: {policy.throttle_action}</div>
                    </div>
                  </div>
                {/each}
              {:else}
                <div class="text-center text-gray-400 py-4">
                  No rate limit policies configured
                </div>
              {/if}
            </div>
          </div>

        {:else if activeSection === 'monitoring'}
          <!-- Monitoring Dashboard -->
          <div class="lg:col-span-3 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">API Monitoring Dashboard</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <!-- Response Time Chart -->
              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Response Time Trend</h4>
                <div class="h-32 bg-gray-900 rounded flex items-center justify-center">
                  <span class="text-gray-400">Chart visualization</span>
                </div>
                <div class="text-sm text-gray-300 mt-2">
                  Avg: {$apiState.gateway.avg_response_time}ms
                </div>
              </div>

              <!-- Request Volume -->
              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Request Volume</h4>
                <div class="h-32 bg-gray-900 rounded flex items-center justify-center">
                  <span class="text-gray-400">Chart visualization</span>
                </div>
                <div class="text-sm text-gray-300 mt-2">
                  Current: {$apiState.gateway.requests_per_second}/s
                </div>
              </div>

              <!-- Error Rate -->
              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Error Rate</h4>
                <div class="h-32 bg-gray-900 rounded flex items-center justify-center">
                  <span class="text-gray-400">Chart visualization</span>
                </div>
                <div class="text-sm text-gray-300 mt-2">
                  Rate: {(100 - $apiState.gateway.success_rate).toFixed(1)}%
                </div>
              </div>

              <!-- Performance Score -->
              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Performance Score</h4>
                <div class="h-32 bg-gray-900 rounded flex items-center justify-center">
                  <span class="text-4xl font-bold text-green-400">
                    {$apiState.monitoring.performance_baseline}
                  </span>
                </div>
                <div class="text-sm text-gray-300 mt-2 text-center">
                  Out of 100
                </div>
              </div>
            </div>

            <!-- Alerts Section -->
            <div class="mt-6">
              <h4 class="font-medium mb-4">Active Alerts</h4>
              <div class="space-y-2">
                <div class="bg-red-900 border border-red-600 rounded p-3">
                  <div class="flex justify-between items-center">
                    <span class="font-medium">High Error Rate on /api/analysis</span>
                    <span class="text-red-400">CRITICAL</span>
                  </div>
                  <p class="text-sm text-gray-300 mt-1">Error rate exceeded 5% threshold (current: 7.2%)</p>
                </div>
                <div class="bg-yellow-900 border border-yellow-600 rounded p-3">
                  <div class="flex justify-between items-center">
                    <span class="font-medium">Rate Limit Threshold Reached</span>
                    <span class="text-yellow-400">WARNING</span>
                  </div>
                  <p class="text-sm text-gray-300 mt-1">Client API key approaching rate limit (89% of quota used)</p>
                </div>
                <div class="bg-blue-900 border border-blue-600 rounded p-3">
                  <div class="flex justify-between items-center">
                    <span class="font-medium">New API Version Available</span>
                    <span class="text-blue-400">INFO</span>
                  </div>
                  <p class="text-sm text-gray-300 mt-1">API v2.1.3 with performance improvements is ready for deployment</p>
                </div>
              </div>
            </div>
          </div>

        {:else if activeSection === 'documentation'}
          <!-- Documentation Management -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">API Documentation Generator</h3>
            <div class="space-y-4">
              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Auto-Generation Settings</h4>
                <div class="space-y-2">
                  <label class="flex items-center">
                    <input type="checkbox" checked class="mr-2">
                    <span class="text-sm">Auto-generate on API changes</span>
                  </label>
                  <label class="flex items-center">
                    <input type="checkbox" checked class="mr-2">
                    <span class="text-sm">Include code examples</span>
                  </label>
                  <label class="flex items-center">
                    <input type="checkbox" checked class="mr-2">
                    <span class="text-sm">Interactive documentation</span>
                  </label>
                  <label class="flex items-center">
                    <input type="checkbox" class="mr-2">
                    <span class="text-sm">Version comparison</span>
                  </label>
                </div>
              </div>

              <div>
                <h4 class="font-medium mb-2">SDK Generation</h4>
                <div class="grid grid-cols-2 gap-2">
                  {#each ['Python', 'JavaScript', 'Java', 'Go', 'Ruby', 'PHP'] as language}
                    <button
                      on:click={() => generateSDK(language.toLowerCase())}
                      class="bg-gray-700 hover:bg-gray-600 px-3 py-2 rounded text-sm"
                    >
                      Generate {language} SDK
                    </button>
                  {/each}
                </div>
              </div>

              <button
                on:click={generateAPIDocumentation}
                class="w-full bg-cyan-600 hover:bg-cyan-700 px-4 py-2 rounded"
              >
                Regenerate All Documentation
              </button>
            </div>
          </div>

          <!-- Documentation Status -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Documentation Status</h3>
            <div class="space-y-3">
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">OpenAPI Spec</span>
                  <span class="text-green-400">Current</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Version: 3.1.0</div>
                  <div>Last Updated: 2h ago</div>
                </div>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Interactive Docs</span>
                  <span class="text-blue-400">Live</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Swagger UI: Available</div>
                  <div>Redoc: Available</div>
                </div>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">SDK Status</span>
                  <span class="text-yellow-400">Partial</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Python: v2.1.2</div>
                  <div>JavaScript: v2.1.1</div>
                  <div>Java: v2.0.8</div>
                </div>
              </div>
            </div>
          </div>
        {/if}
      </div>

      <!-- API Registry -->
      {#if $registeredAPIs && $registeredAPIs.length > 0}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Registered APIs</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {#each $registeredAPIs as api}
              <div class="bg-gray-700 rounded p-4 cursor-pointer hover:bg-gray-600" on:click={() => selectedAPI = api}>
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">{api.name}</span>
                  <span class="text-xs px-2 py-1 rounded bg-green-600">{api.status}</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Version: {api.version}</div>
                  <div>Endpoints: {api.endpoint_count}</div>
                  <div>RPS: {api.requests_per_second}</div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>