<script lang="ts">
  import { endpoints, websockets, getEndpoint, getWebSocket } from "$lib/config/endpoints";

  // Distributed endpoint helpers
  const getServiceHost = () => {
    const service = typeof window !== "undefined" && window.location.hostname;
    return service === "localhost" ? "localhost" : service;
  };

  const getWebSocketHost = () => {
    const service = typeof window !== "undefined" && window.location.hostname;
    return service === "localhost" ? "localhost" : service;
  };
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // Advanced Log Management state
  const logState = writable({
    aggregation: {
      total_services: 67,
      active_loggers: 63,
      log_volume_24h: '2.3TB',
      avg_ingestion_rate: '47MB/s',
      correlation_accuracy: 94.7,
      pattern_detection: 89.3
    },
    search: {
      elasticsearch_health: 98.4,
      index_count: 156,
      search_performance: '12ms avg',
      query_success_rate: 99.2,
      full_text_enabled: true,
      real_time_indexing: true
    },
    correlation: {
      cross_service_events: 2847,
      correlation_patterns: 134,
      anomaly_detection: 'active',
      alert_triggers: 23,
      false_positive_rate: 2.1,
      machine_learning_enabled: true
    },
    alerts: {
      active_alerts: 8,
      alert_rules: 47,
      notification_channels: 6,
      escalation_policies: 12,
      alert_accuracy: 96.8,
      response_time_avg: '23s'
    },
    visualization: {
      dashboard_count: 12,
      real_time_charts: 34,
      custom_visualizations: 8,
      export_formats: 5,
      interactive_graphs: true,
      d3js_enabled: true
    }
  });

  // Log management controls
  let activeLogWorkspace = 'aggregation';
  let searchQuery = '';
  let selectedTimeRange = '1h';
  let selectedServices = [];
  let logLevel = 'INFO';

  // Search and filter configuration
  let searchConfig = {
    case_sensitive: false,
    regex_enabled: false,
    fuzzy_search: true,
    context_lines: 3,
    highlight_matches: true,
    max_results: 1000
  };

  // Alert configuration
  let alertConfig = {
    pattern: '',
    threshold: 10,
    time_window: 300,
    notification_channel: 'slack',
    severity: 'warning',
    auto_escalate: true
  };

  // Correlation configuration
  let correlationConfig = {
    correlation_window: 600,
    min_confidence: 0.8,
    pattern_learning: true,
    anomaly_detection: true,
    cross_service: true
  };

  // Live data stores
  const logResults = writable([]);
  const correlationResults = writable([]);
  const alertHistory = writable([]);
  const logMetrics = writable({});

  // WebSocket connections
  let logStreamWs: WebSocket | null = null;
  let searchWs: WebSocket | null = null;
  let correlationWs: WebSocket | null = null;
  let alertWs: WebSocket | null = null;

  // Available services for filtering
  const availableServices = [
    'postgres', 'neo4j', 'redis', 'elasticsearch', 'kafka', 'rabbitmq',
    'intelowl-django', 'intelowl-celery-worker', 'autonomous-coordinator',
    'adaptive-learning', 'resource-manager', 'chaos-engineer', 'tor',
    'vault', 'prometheus', 'grafana', 'jaeger'
  ];

  onMount(() => {
    initializeWebSockets();
    loadLogData();
    startMetricsCollection();
  });

  onDestroy(() => {
    if (logStreamWs) logStreamWs.close();
    if (searchWs) searchWs.close();
    if (correlationWs) correlationWs.close();
    if (alertWs) alertWs.close();
  });

  function initializeWebSockets() {
    // Real-time log stream WebSocket
    logStreamWs = new WebSocket('ws://${getWebSocketHost()}:8110/logs/stream');
    logStreamWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      logResults.update(logs => [data, ...logs.slice(0, 999)]);
    };

    // Log search WebSocket
    searchWs = new WebSocket('ws://${getWebSocketHost()}:8111/logs/search');
    searchWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.search_results) {
        logResults.set(data.search_results);
      }
    };

    // Log correlation WebSocket
    correlationWs = new WebSocket('ws://${getWebSocketHost()}:8112/logs/correlation');
    correlationWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      logState.update(state => ({
        ...state,
        correlation: { ...state.correlation, ...data.correlation }
      }));
      if (data.correlations) {
        correlationResults.set(data.correlations);
      }
    };

    // Alert management WebSocket
    alertWs = new WebSocket('ws://${getWebSocketHost()}:8113/logs/alerts');
    alertWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      logState.update(state => ({
        ...state,
        alerts: { ...state.alerts, ...data }
      }));
      if (data.alert_history) {
        alertHistory.set(data.alert_history);
      }
    };
  }

  async function loadLogData() {
    try {
      const [resultsRes, correlationsRes, alertsRes] = await Promise.all([
        fetch('http://${getServiceHost()}:8110/api/recent'),
        fetch('http://${getServiceHost()}:8112/api/correlations'),
        fetch('http://${getServiceHost()}:8113/api/alerts')
      ]);

      const results = await resultsRes.json();
      const correlations = await correlationsRes.json();
      const alerts = await alertsRes.json();

      logResults.set(results);
      correlationResults.set(correlations);
      alertHistory.set(alerts);
    } catch (error) {
      console.error('Failed to load log data:', error);
    }
  }

  async function startMetricsCollection() {
    setInterval(async () => {
      try {
        const response = await fetch('http://${getServiceHost()}:8110/api/metrics');
        const metrics = await response.json();
        logState.update(state => ({
          ...state,
          ...metrics
        }));
        logMetrics.set(metrics);
      } catch (error) {
        console.error('Log metrics collection error:', error);
      }
    }, 5000);
  }

  async function performLogSearch() {
    if (!searchQuery) return;

    try {
      const response = await fetch('http://${getServiceHost()}:8111/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          time_range: selectedTimeRange,
          services: selectedServices,
          log_level: logLevel,
          config: searchConfig
        })
      });

      if (response.ok) {
        const results = await response.json();
        logResults.set(results);
      }
    } catch (error) {
      console.error('Log search failed:', error);
    }
  }

  async function createLogAlert() {
    if (!alertConfig.pattern) return;

    try {
      const response = await fetch('http://${getServiceHost()}:8113/api/create-alert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(alertConfig)
      });

      if (response.ok) {
        console.log('Log alert created');
        // Reset form
        alertConfig.pattern = '';
      }
    } catch (error) {
      console.error('Alert creation failed:', error);
    }
  }

  async function startCorrelationAnalysis() {
    try {
      const response = await fetch('http://${getServiceHost()}:8112/api/correlate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(correlationConfig)
      });

      if (response.ok) {
        console.log('Correlation analysis started');
      }
    } catch (error) {
      console.error('Correlation analysis failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': case 'active': case 'enabled': return 'text-green-400';
      case 'warning': case 'degraded': return 'text-yellow-400';
      case 'error': case 'failed': case 'critical': return 'text-red-400';
      case 'info': case 'debug': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  }

  function getLogLevelColor(level: string): string {
    switch (level.toUpperCase()) {
      case 'ERROR': return 'text-red-400';
      case 'WARN': case 'WARNING': return 'text-yellow-400';
      case 'INFO': return 'text-blue-400';
      case 'DEBUG': return 'text-gray-400';
      default: return 'text-gray-300';
    }
  }

  function getHealthColor(value: number, threshold: number = 95): string {
    if (value >= threshold) return 'text-green-400';
    if (value >= threshold - 10) return 'text-yellow-400';
    return 'text-red-400';
  }
</script>

<svelte:head>
  <title>Advanced Log Management | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-cyan-400 to-teal-500 bg-clip-text text-transparent">
        Advanced Log Management System
      </h1>
      <p class="text-gray-300">Centralized log aggregation, correlation, and intelligent analysis</p>
    </div>

    {#if $logState}
      <!-- Log Management Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-5 gap-6 mb-8">
        <!-- Log Aggregation -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-cyan-400 mr-2"></span>
            Log Aggregation
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Services:</span>
              <span class="text-cyan-400">{$logState.aggregation.total_services}</span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-green-400">{$logState.aggregation.active_loggers}</span>
            </div>
            <div class="flex justify-between">
              <span>Volume (24h):</span>
              <span class="text-blue-400">{$logState.aggregation.log_volume_24h}</span>
            </div>
            <div class="flex justify-between">
              <span>Rate:</span>
              <span class="text-yellow-400">{$logState.aggregation.avg_ingestion_rate}</span>
            </div>
          </div>
        </div>

        <!-- Search Engine -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Search Engine
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>ES Health:</span>
              <span class={getHealthColor($logState.search.elasticsearch_health)}>{$logState.search.elasticsearch_health}%</span>
            </div>
            <div class="flex justify-between">
              <span>Indices:</span>
              <span class="text-green-400">{$logState.search.index_count}</span>
            </div>
            <div class="flex justify-between">
              <span>Performance:</span>
              <span class="text-cyan-400">{$logState.search.search_performance}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class={getHealthColor($logState.search.query_success_rate)}>{$logState.search.query_success_rate}%</span>
            </div>
          </div>
        </div>

        <!-- Correlation -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Log Correlation
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Events:</span>
              <span class="text-purple-400">{$logState.correlation.cross_service_events.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Patterns:</span>
              <span class="text-blue-400">{$logState.correlation.correlation_patterns}</span>
            </div>
            <div class="flex justify-between">
              <span>Anomalies:</span>
              <span class={getStatusColor($logState.correlation.anomaly_detection)}>
                {$logState.correlation.anomaly_detection.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>False Positive:</span>
              <span class="text-yellow-400">{$logState.correlation.false_positive_rate}%</span>
            </div>
          </div>
        </div>

        <!-- Alerts -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-red-400 mr-2"></span>
            Log Alerts
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-red-400">{$logState.alerts.active_alerts}</span>
            </div>
            <div class="flex justify-between">
              <span>Rules:</span>
              <span class="text-blue-400">{$logState.alerts.alert_rules}</span>
            </div>
            <div class="flex justify-between">
              <span>Accuracy:</span>
              <span class={getHealthColor($logState.alerts.alert_accuracy)}>{$logState.alerts.alert_accuracy}%</span>
            </div>
            <div class="flex justify-between">
              <span>Response:</span>
              <span class="text-cyan-400">{$logState.alerts.response_time_avg}</span>
            </div>
          </div>
        </div>

        <!-- Visualization -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-orange-400 mr-2"></span>
            Visualization
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Dashboards:</span>
              <span class="text-orange-400">{$logState.visualization.dashboard_count}</span>
            </div>
            <div class="flex justify-between">
              <span>Real-time:</span>
              <span class="text-green-400">{$logState.visualization.real_time_charts}</span>
            </div>
            <div class="flex justify-between">
              <span>Custom:</span>
              <span class="text-purple-400">{$logState.visualization.custom_visualizations}</span>
            </div>
            <div class="flex justify-between">
              <span>D3.js:</span>
              <span class={$logState.visualization.d3js_enabled ? 'text-green-400' : 'text-gray-400'}>
                {$logState.visualization.d3js_enabled ? 'ENABLED' : 'DISABLED'}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Log Management Workspace Tabs -->
      <div class="mb-6">
        <div class="flex space-x-1 bg-gray-800 p-1 rounded-lg">
          {#each ['aggregation', 'search', 'correlation', 'alerts', 'visualization'] as workspace}
            <button
              on:click={() => activeLogWorkspace = workspace}
              class="flex-1 py-2 px-4 rounded text-center transition-colors text-sm
                {activeLogWorkspace === workspace
                  ? 'bg-cyan-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'}"
            >
              {workspace.charAt(0).toUpperCase() + workspace.slice(1)}
            </button>
          {/each}
        </div>
      </div>

      <!-- Workspace Content -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {#if activeLogWorkspace === 'search'}
          <!-- Log Search Interface -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Advanced Log Search</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Search Query</label>
                <input
                  bind:value={searchQuery}
                  placeholder="Enter search query... (supports regex, wildcards, and boolean operators)"
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
              <div class="grid grid-cols-3 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Time Range</label>
                  <select bind:value={selectedTimeRange} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="15m">Last 15 minutes</option>
                    <option value="1h">Last hour</option>
                    <option value="6h">Last 6 hours</option>
                    <option value="24h">Last 24 hours</option>
                    <option value="7d">Last 7 days</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Log Level</label>
                  <select bind:value={logLevel} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="DEBUG">Debug</option>
                    <option value="INFO">Info</option>
                    <option value="WARN">Warning</option>
                    <option value="ERROR">Error</option>
                    <option value="CRITICAL">Critical</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Context Lines</label>
                  <input
                    type="number"
                    bind:value={searchConfig.context_lines}
                    min="0"
                    max="10"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Filter Services</label>
                <div class="grid grid-cols-3 gap-2 max-h-24 overflow-y-auto">
                  {#each availableServices as service}
                    <label class="flex items-center">
                      <input type="checkbox" bind:group={selectedServices} value={service} class="mr-2">
                      <span class="text-sm">{service}</span>
                    </label>
                  {/each}
                </div>
              </div>
              <div class="grid grid-cols-3 gap-4">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={searchConfig.regex_enabled} class="mr-2">
                  <span class="text-sm">Regex</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={searchConfig.case_sensitive} class="mr-2">
                  <span class="text-sm">Case Sensitive</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={searchConfig.fuzzy_search} class="mr-2">
                  <span class="text-sm">Fuzzy Search</span>
                </label>
              </div>
              <button
                on:click={performLogSearch}
                disabled={!searchQuery}
                class="w-full bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                Search Logs
              </button>
            </div>
          </div>

          <!-- Search Results -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Search Results</h3>
            <div class="bg-gray-900 rounded p-4 h-96 overflow-y-auto font-mono text-sm">
              {#if $logResults && $logResults.length > 0}
                {#each $logResults.slice(0, 50) as log}
                  <div class="mb-2">
                    <div class="flex items-center space-x-2">
                      <span class="text-gray-500">[{log.timestamp}]</span>
                      <span class="text-blue-400">{log.service}</span>
                      <span class={getLogLevelColor(log.level)} class="px-1 rounded text-xs">
                        {log.level}
                      </span>
                    </div>
                    <div class="text-gray-300 ml-4">
                      {log.message}
                    </div>
                  </div>
                {/each}
              {:else}
                <div class="text-gray-400 text-center py-8">
                  Enter search query to view results
                </div>
              {/if}
            </div>
          </div>

        {:else if activeLogWorkspace === 'correlation'}
          <!-- Log Correlation -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Cross-Service Log Correlation</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Correlation Window (seconds): {correlationConfig.correlation_window}</label>
                <input
                  type="range"
                  bind:value={correlationConfig.correlation_window}
                  min="60"
                  max="3600"
                  step="60"
                  class="w-full"
                >
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Minimum Confidence: {correlationConfig.min_confidence}</label>
                <input
                  type="range"
                  bind:value={correlationConfig.min_confidence}
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  class="w-full"
                >
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={correlationConfig.pattern_learning} class="mr-2">
                  <span class="text-sm">Pattern Learning</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={correlationConfig.anomaly_detection} class="mr-2">
                  <span class="text-sm">Anomaly Detection</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={correlationConfig.cross_service} class="mr-2">
                  <span class="text-sm">Cross-Service Analysis</span>
                </label>
              </div>
              <button
                on:click={startCorrelationAnalysis}
                class="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded"
              >
                Start Correlation Analysis
              </button>
            </div>
          </div>

          <!-- Correlation Results -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Correlation Results</h3>
            <div class="space-y-3">
              {#if $correlationResults && $correlationResults.length > 0}
                {#each $correlationResults.slice(0, 5) as correlation}
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">Pattern {correlation.id}</span>
                      <span class="text-purple-400 text-xs">{correlation.confidence}%</span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Services: {correlation.services.join(', ')}</div>
                      <div>Events: {correlation.event_count}</div>
                      <div>Time Window: {correlation.time_window}s</div>
                    </div>
                  </div>
                {/each}
              {:else}
                <div class="text-center text-gray-400 py-4">
                  No correlation patterns detected
                </div>
              {/if}
            </div>
          </div>

        {:else if activeLogWorkspace === 'alerts'}
          <!-- Alert Management -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Log Alert Configuration</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Alert Pattern</label>
                <input
                  bind:value={alertConfig.pattern}
                  placeholder="Enter pattern to alert on (regex supported)..."
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
              <div class="grid grid-cols-3 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Threshold</label>
                  <input
                    type="number"
                    bind:value={alertConfig.threshold}
                    min="1"
                    max="1000"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Time Window (s)</label>
                  <input
                    type="number"
                    bind:value={alertConfig.time_window}
                    min="60"
                    max="3600"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Severity</label>
                  <select bind:value={alertConfig.severity} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="info">Info</option>
                    <option value="warning">Warning</option>
                    <option value="critical">Critical</option>
                  </select>
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Notification Channel</label>
                <select bind:value={alertConfig.notification_channel} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="slack">Slack</option>
                  <option value="email">Email</option>
                  <option value="webhook">Webhook</option>
                  <option value="pagerduty">PagerDuty</option>
                  <option value="discord">Discord</option>
                </select>
              </div>
              <div class="flex items-center">
                <input type="checkbox" bind:checked={alertConfig.auto_escalate} id="auto-escalate" class="mr-2">
                <label for="auto-escalate" class="text-sm">Auto-Escalate Unacknowledged Alerts</label>
              </div>
              <button
                on:click={createLogAlert}
                disabled={!alertConfig.pattern}
                class="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                Create Log Alert
              </button>
            </div>
          </div>

          <!-- Active Alerts -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Active Log Alerts</h3>
            <div class="space-y-3">
              <div class="bg-red-900 border border-red-600 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">High Error Rate</span>
                  <span class="text-red-400 text-xs">CRITICAL</span>
                </div>
                <div class="text-sm text-red-300">
                  ERROR patterns exceeded threshold (23/min)
                </div>
              </div>
              <div class="bg-yellow-900 border border-yellow-600 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Slow Query Alert</span>
                  <span class="text-yellow-400 text-xs">WARNING</span>
                </div>
                <div class="text-sm text-yellow-300">
                  Database queries >1s detected (PostgreSQL)
                </div>
              </div>
              <div class="bg-blue-900 border border-blue-600 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">API Rate Limit</span>
                  <span class="text-blue-400 text-xs">INFO</span>
                </div>
                <div class="text-sm text-blue-300">
                  Rate limiting activated for external APIs
                </div>
              </div>
            </div>
          </div>

        {:else if activeLogWorkspace === 'visualization'}
          <!-- Log Visualization -->
          <div class="lg:col-span-3 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Log Flow Visualization</h3>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <!-- Service Log Flow -->
              <div class="bg-gray-900 rounded p-4 h-64">
                <h4 class="font-medium mb-2">Service Log Flow (Real-time)</h4>
                <div class="h-48 flex items-center justify-center">
                  <span class="text-gray-400">D3.js log flow visualization</span>
                </div>
              </div>

              <!-- Error Pattern Analysis -->
              <div class="bg-gray-900 rounded p-4 h-64">
                <h4 class="font-medium mb-2">Error Pattern Analysis</h4>
                <div class="h-48 flex items-center justify-center">
                  <span class="text-gray-400">Error correlation heatmap</span>
                </div>
              </div>

              <!-- Log Volume Trends -->
              <div class="bg-gray-900 rounded p-4 h-64">
                <h4 class="font-medium mb-2">Log Volume Trends</h4>
                <div class="h-48 flex items-center justify-center">
                  <span class="text-gray-400">Volume trend chart</span>
                </div>
              </div>

              <!-- Service Health Matrix -->
              <div class="bg-gray-900 rounded p-4 h-64">
                <h4 class="font-medium mb-2">Service Health Matrix</h4>
                <div class="h-48 flex items-center justify-center">
                  <span class="text-gray-400">Health correlation matrix</span>
                </div>
              </div>
            </div>
          </div>

        {:else}
          <!-- Default: Aggregation View -->
          <div class="lg:col-span-3 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Real-time Log Aggregation</h3>
            <div class="bg-gray-900 rounded p-4 h-96 overflow-y-auto font-mono text-sm">
              {#if $logResults && $logResults.length > 0}
                {#each $logResults.slice(0, 100) as log}
                  <div class="mb-1">
                    <span class="text-gray-500">[{log.timestamp}]</span>
                    <span class="text-blue-400">{log.service}</span>
                    <span class={getLogLevelColor(log.level)} class="px-1 rounded text-xs">
                      {log.level}
                    </span>
                    <span class="text-gray-300">{log.message}</span>
                  </div>
                {/each}
              {:else}
                <div class="text-gray-400 text-center py-8">
                  Real-time logs from 67 services will appear here
                </div>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>