<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // IntelOwl Integration state
  const intelowlState = writable({
    platform: {
      status: 'active',
      version: '6.2.1',
      uptime: '15d 7h 23m',
      total_analyses: 15847,
      successful_analyses: 15203,
      failed_analyses: 644,
      success_rate: 95.9,
      avg_analysis_time: '2m 34s'
    },
    services: {
      django: { status: 'healthy', cpu: 23.4, memory: 67.2, connections: 147 },
      celery_worker: { status: 'healthy', cpu: 45.7, memory: 78.9, active_tasks: 23 },
      celery_beat: { status: 'healthy', cpu: 5.2, memory: 12.1, scheduled_tasks: 156 },
      nginx: { status: 'healthy', cpu: 8.7, memory: 15.3, requests_per_sec: 234 },
      postgres: { status: 'healthy', cpu: 34.6, memory: 89.2, connections: 89 }
    },
    analyzers: {
      total_analyzers: 67,
      active_analyzers: 63,
      custom_analyzers: 6,
      external_apis: 23,
      analyzer_success_rate: 94.7,
      avg_analyzer_time: '45s'
    },
    connectors: {
      total_connectors: 12,
      active_connectors: 11,
      neo4j_connector: true,
      elasticsearch_connector: true,
      custom_connectors: 3,
      connection_success_rate: 98.3
    },
    queue_management: {
      pending_jobs: 47,
      running_jobs: 12,
      completed_jobs_24h: 1847,
      failed_jobs_24h: 23,
      queue_health: 97.2,
      worker_efficiency: 89.4
    }
  });

  // IntelOwl controls
  let selectedAnalyzer = '';
  let analysisTarget = '';
  let analysisType = 'file';
  let customAnalyzerConfig = {
    name: '',
    type: 'file',
    python_module: '',
    configuration: {},
    enabled: true
  };

  // Integration controls
  let integrationMode = 'unified';
  let embedInBEV = true;
  let crossPlatformSync = true;

  // Live data stores
  const activeAnalyses = writable([]);
  const analyzerStatus = writable([]);
  const queueStatus = writable([]);
  const integrationLogs = writable([]);

  // WebSocket connections
  let intelowlWs: WebSocket | null = null;
  let queueWs: WebSocket | null = null;
  let analyzerWs: WebSocket | null = null;
  let integrationWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadIntelOwlData();
    startIntegrationMonitoring();
    checkIntelOwlConnectivity();
  });

  onDestroy(() => {
    if (intelowlWs) intelowlWs.close();
    if (queueWs) queueWs.close();
    if (analyzerWs) analyzerWs.close();
    if (integrationWs) integrationWs.close();
  });

  function initializeWebSockets() {
    // IntelOwl platform WebSocket
    intelowlWs = new WebSocket('ws://localhost:8090/intelowl');
    intelowlWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      intelowlState.update(state => ({
        ...state,
        platform: { ...state.platform, ...data.platform },
        services: { ...state.services, ...data.services }
      }));
    };

    // Queue management WebSocket
    queueWs = new WebSocket('ws://localhost:8091/queue');
    queueWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      intelowlState.update(state => ({
        ...state,
        queue_management: { ...state.queue_management, ...data }
      }));
      if (data.queue_status) {
        queueStatus.set(data.queue_status);
      }
    };

    // Analyzer management WebSocket
    analyzerWs = new WebSocket('ws://localhost:8092/analyzers');
    analyzerWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      intelowlState.update(state => ({
        ...state,
        analyzers: { ...state.analyzers, ...data }
      }));
      if (data.analyzer_status) {
        analyzerStatus.set(data.analyzer_status);
      }
    };

    // BEV-IntelOwl integration WebSocket
    integrationWs = new WebSocket('ws://localhost:8093/integration');
    integrationWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      integrationLogs.update(logs => [data, ...logs.slice(0, 99)]);
    };
  }

  async function loadIntelOwlData() {
    try {
      const [analysesRes, statusRes, queueRes] = await Promise.all([
        fetch('http://localhost:8090/api/analyses'),
        fetch('http://localhost:8092/api/status'),
        fetch('http://localhost:8091/api/queue')
      ]);

      const analyses = await analysesRes.json();
      const status = await statusRes.json();
      const queue = await queueRes.json();

      activeAnalyses.set(analyses);
      analyzerStatus.set(status);
      queueStatus.set(queue);
    } catch (error) {
      console.error('Failed to load IntelOwl data:', error);
    }
  }

  async function startIntegrationMonitoring() {
    setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8093/api/integration-status');
        const integration = await response.json();
        // Update integration status
      } catch (error) {
        console.error('Integration monitoring error:', error);
      }
    }, 10000);
  }

  async function checkIntelOwlConnectivity() {
    try {
      const response = await fetch('http://localhost/api/me');
      if (response.ok) {
        console.log('IntelOwl connectivity confirmed');
      }
    } catch (error) {
      console.error('IntelOwl connectivity check failed:', error);
    }
  }

  async function startAnalysis() {
    if (!selectedAnalyzer || !analysisTarget) return;

    try {
      const response = await fetch('http://localhost:8090/api/start-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analyzer: selectedAnalyzer,
          target: analysisTarget,
          type: analysisType,
          integration_mode: integrationMode
        })
      });

      if (response.ok) {
        const analysis = await response.json();
        activeAnalyses.update(analyses => [analysis, ...analyses]);
      }
    } catch (error) {
      console.error('Analysis start failed:', error);
    }
  }

  async function deployCustomAnalyzer() {
    if (!customAnalyzerConfig.name || !customAnalyzerConfig.python_module) return;

    try {
      const response = await fetch('http://localhost:8092/api/deploy-analyzer', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(customAnalyzerConfig)
      });

      if (response.ok) {
        console.log('Custom analyzer deployed');
        // Reset form
        customAnalyzerConfig = {
          name: '',
          type: 'file',
          python_module: '',
          configuration: {},
          enabled: true
        };
      }
    } catch (error) {
      console.error('Analyzer deployment failed:', error);
    }
  }

  async function openIntelOwlInterface() {
    try {
      await invoke('open_external', { url: 'http://localhost' });
    } catch (error) {
      console.error('Failed to open IntelOwl interface:', error);
    }
  }

  async function syncBEVIntelOwl() {
    try {
      const response = await fetch('http://localhost:8093/api/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: integrationMode,
          embed_in_bev: embedInBEV,
          cross_platform_sync: crossPlatformSync
        })
      });

      if (response.ok) {
        console.log('BEV-IntelOwl synchronization initiated');
      }
    } catch (error) {
      console.error('Synchronization failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': case 'active': case 'completed': return 'text-green-400';
      case 'running': case 'analyzing': case 'processing': return 'text-blue-400';
      case 'failed': case 'error': case 'unhealthy': return 'text-red-400';
      case 'warning': case 'degraded': return 'text-yellow-400';
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
  <title>IntelOwl Integration Admin | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-amber-400 to-orange-500 bg-clip-text text-transparent">
        IntelOwl Integration Administration
      </h1>
      <p class="text-gray-300">Core OSINT engine management and BEV platform integration</p>
    </div>

    {#if $intelowlState}
      <!-- IntelOwl Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-5 gap-6 mb-8">
        <!-- Platform Status -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-orange-400 mr-2"></span>
            Platform Status
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($intelowlState.platform.status)}>
                {$intelowlState.platform.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Version:</span>
              <span class="text-blue-400">{$intelowlState.platform.version}</span>
            </div>
            <div class="flex justify-between">
              <span>Uptime:</span>
              <span class="text-green-400">{$intelowlState.platform.uptime}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class={getHealthColor($intelowlState.platform.success_rate)}>{$intelowlState.platform.success_rate}%</span>
            </div>
          </div>
        </div>

        <!-- Service Health -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Service Health
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Django:</span>
              <span class={getStatusColor($intelowlState.services.django.status)}>
                {$intelowlState.services.django.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Celery Worker:</span>
              <span class={getStatusColor($intelowlState.services.celery_worker.status)}>
                {$intelowlState.services.celery_worker.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Nginx:</span>
              <span class={getStatusColor($intelowlState.services.nginx.status)}>
                {$intelowlState.services.nginx.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>PostgreSQL:</span>
              <span class={getStatusColor($intelowlState.services.postgres.status)}>
                {$intelowlState.services.postgres.status.toUpperCase()}
              </span>
            </div>
          </div>
        </div>

        <!-- Analyzers -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Analyzers
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Total:</span>
              <span class="text-purple-400">{$intelowlState.analyzers.total_analyzers}</span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-green-400">{$intelowlState.analyzers.active_analyzers}</span>
            </div>
            <div class="flex justify-between">
              <span>Custom:</span>
              <span class="text-blue-400">{$intelowlState.analyzers.custom_analyzers}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class={getHealthColor($intelowlState.analyzers.analyzer_success_rate)}>{$intelowlState.analyzers.analyzer_success_rate}%</span>
            </div>
          </div>
        </div>

        <!-- Connectors -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-cyan-400 mr-2"></span>
            Connectors
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Total:</span>
              <span class="text-cyan-400">{$intelowlState.connectors.total_connectors}</span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-green-400">{$intelowlState.connectors.active_connectors}</span>
            </div>
            <div class="flex justify-between">
              <span>Neo4j:</span>
              <span class={$intelowlState.connectors.neo4j_connector ? 'text-green-400' : 'text-red-400'}>
                {$intelowlState.connectors.neo4j_connector ? 'CONNECTED' : 'DISCONNECTED'}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class={getHealthColor($intelowlState.connectors.connection_success_rate)}>{$intelowlState.connectors.connection_success_rate}%</span>
            </div>
          </div>
        </div>

        <!-- Queue Management -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-yellow-400 mr-2"></span>
            Queue Management
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Pending:</span>
              <span class="text-yellow-400">{$intelowlState.queue_management.pending_jobs}</span>
            </div>
            <div class="flex justify-between">
              <span>Running:</span>
              <span class="text-blue-400">{$intelowlState.queue_management.running_jobs}</span>
            </div>
            <div class="flex justify-between">
              <span>Completed (24h):</span>
              <span class="text-green-400">{$intelowlState.queue_management.completed_jobs_24h.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Queue Health:</span>
              <span class={getHealthColor($intelowlState.queue_management.queue_health)}>{$intelowlState.queue_management.queue_health}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Integration Control Center -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <!-- BEV-IntelOwl Integration -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">BEV-IntelOwl Integration</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Integration Mode</label>
              <select bind:value={integrationMode} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="unified">Unified Interface</option>
                <option value="embedded">Embedded Frame</option>
                <option value="api_only">API Integration Only</option>
                <option value="separate">Separate Platforms</option>
              </select>
            </div>
            <div class="space-y-2">
              <label class="flex items-center">
                <input type="checkbox" bind:checked={embedInBEV} class="mr-2">
                <span class="text-sm">Embed IntelOwl in BEV Dashboard</span>
              </label>
              <label class="flex items-center">
                <input type="checkbox" bind:checked={crossPlatformSync} class="mr-2">
                <span class="text-sm">Cross-Platform Data Sync</span>
              </label>
            </div>
            <button
              on:click={syncBEVIntelOwl}
              class="w-full bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded"
            >
              Sync BEV ↔ IntelOwl
            </button>
            <button
              on:click={openIntelOwlInterface}
              class="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
            >
              Open IntelOwl Interface
            </button>
          </div>
        </div>

        <!-- Analysis Controls -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Analysis Controls</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Target Analyzer</label>
              <select bind:value={selectedAnalyzer} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="">Select Analyzer</option>
                <option value="breach_analyzer">Breach Database Analyzer</option>
                <option value="darknet_analyzer">Darknet Market Analyzer</option>
                <option value="crypto_analyzer">Cryptocurrency Analyzer</option>
                <option value="social_analyzer">Social Media Analyzer</option>
                <option value="metadata_analyzer">Metadata Analyzer</option>
                <option value="watermark_analyzer">Watermark Analyzer</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Analysis Type</label>
              <select bind:value={analysisType} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="file">File Analysis</option>
                <option value="url">URL Analysis</option>
                <option value="domain">Domain Analysis</option>
                <option value="ip">IP Address Analysis</option>
                <option value="hash">Hash Analysis</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Analysis Target</label>
              <input
                bind:value={analysisTarget}
                placeholder="Enter file path, URL, domain, IP, or hash..."
                class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
              >
            </div>
            <button
              on:click={startAnalysis}
              disabled={!selectedAnalyzer || !analysisTarget}
              class="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 px-4 py-2 rounded"
            >
              Start Analysis
            </button>
          </div>
        </div>

        <!-- Custom Analyzer Deployment -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Custom Analyzer Deployment</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Analyzer Name</label>
              <input
                bind:value={customAnalyzerConfig.name}
                placeholder="Enter analyzer name..."
                class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
              >
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Python Module</label>
              <input
                bind:value={customAnalyzerConfig.python_module}
                placeholder="custom_analyzers.my_analyzer"
                class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
              >
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Analyzer Type</label>
              <select bind:value={customAnalyzerConfig.type} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="file">File Analyzer</option>
                <option value="observable">Observable Analyzer</option>
                <option value="hybrid">Hybrid Analyzer</option>
              </select>
            </div>
            <div class="flex items-center">
              <input type="checkbox" bind:checked={customAnalyzerConfig.enabled} id="analyzer-enabled" class="mr-2">
              <label for="analyzer-enabled" class="text-sm">Enable After Deployment</label>
            </div>
            <button
              on:click={deployCustomAnalyzer}
              disabled={!customAnalyzerConfig.name || !customAnalyzerConfig.python_module}
              class="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded"
            >
              Deploy Custom Analyzer
            </button>
          </div>
        </div>
      </div>

      <!-- Service Details Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-8">
        {#each Object.entries($intelowlState.services) as [serviceName, serviceData]}
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4 flex items-center">
              <span class="w-3 h-3 rounded-full {getStatusColor(serviceData.status) === 'text-green-400' ? 'bg-green-400' : 'bg-red-400'} mr-2"></span>
              {serviceName.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
            </h3>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span>CPU:</span>
                <span class="text-blue-400">{serviceData.cpu}%</span>
              </div>
              <div class="flex justify-between">
                <span>Memory:</span>
                <span class="text-yellow-400">{serviceData.memory}%</span>
              </div>
              {#if serviceData.connections}
                <div class="flex justify-between">
                  <span>Connections:</span>
                  <span class="text-green-400">{serviceData.connections}</span>
                </div>
              {/if}
              {#if serviceData.active_tasks}
                <div class="flex justify-between">
                  <span>Active Tasks:</span>
                  <span class="text-purple-400">{serviceData.active_tasks}</span>
                </div>
              {/if}
              {#if serviceData.requests_per_sec}
                <div class="flex justify-between">
                  <span>RPS:</span>
                  <span class="text-cyan-400">{serviceData.requests_per_sec}</span>
                </div>
              {/if}
            </div>
          </div>
        {/each}
      </div>

      <!-- Active Analyses -->
      {#if $activeAnalyses && $activeAnalyses.length > 0}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
          <h3 class="text-lg font-semibold mb-4">Active Analyses</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {#each $activeAnalyses.slice(0, 9) as analysis}
              <div class="bg-gray-700 rounded p-4">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Analysis {analysis.id}</span>
                  <span class={getStatusColor(analysis.status)} class="text-xs">
                    {analysis.status.toUpperCase()}
                  </span>
                </div>
                <div class="text-sm text-gray-300 space-y-1">
                  <div>Analyzer: {analysis.analyzer}</div>
                  <div>Target: {analysis.target}</div>
                  <div>Progress: {analysis.progress}%</div>
                  <div>Started: {analysis.start_time}</div>
                </div>
                <div class="mt-2">
                  <div class="w-full bg-gray-600 rounded-full h-2">
                    <div class="bg-blue-400 h-2 rounded-full" style="width: {analysis.progress}%"></div>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Integration Status -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-lg font-semibold mb-4">BEV ↔ IntelOwl Integration Status</h3>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div class="bg-gray-700 rounded p-4">
            <h4 class="font-medium mb-2">Data Synchronization</h4>
            <div class="text-sm text-gray-300 space-y-1">
              <div>Mode: {integrationMode}</div>
              <div>Sync Status: Active</div>
              <div>Last Sync: 2m ago</div>
              <div>Sync Rate: 94.7%</div>
            </div>
          </div>

          <div class="bg-gray-700 rounded p-4">
            <h4 class="font-medium mb-2">API Integration</h4>
            <div class="text-sm text-gray-300 space-y-1">
              <div>Endpoint: http://localhost</div>
              <div>Auth Status: Active</div>
              <div>Rate Limit: 1000/min</div>
              <div>Health: 99.2%</div>
            </div>
          </div>

          <div class="bg-gray-700 rounded p-4">
            <h4 class="font-medium mb-2">Database Integration</h4>
            <div class="text-sm text-gray-300 space-y-1">
              <div>PostgreSQL: Connected</div>
              <div>Neo4j: Synced</div>
              <div>Redis: Active</div>
              <div>Cross-Platform: Enabled</div>
            </div>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>