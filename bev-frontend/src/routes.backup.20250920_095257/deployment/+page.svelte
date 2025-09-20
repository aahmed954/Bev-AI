<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // Deployment Automation state
  const deploymentState = writable({
    automation: {
      total_scripts: 6,
      active_deployments: 2,
      pending_approvals: 1,
      deployment_queue: 3,
      success_rate: 97.8,
      avg_deployment_time: '8m 47s',
      last_deployment: '2h ago'
    },
    node_configuration: {
      total_nodes: 3,
      data_core_nodes: 1,
      processing_core_nodes: 1,
      edge_nodes: 1,
      node_health: 98.7,
      sync_status: 'synchronized',
      config_drift: 0
    },
    cluster_orchestration: {
      cluster_health: 96.4,
      coordination_efficiency: 89.7,
      load_balancing: 94.2,
      failover_readiness: 100,
      resource_distribution: 87.3,
      auto_scaling_events: 23
    },
    pipeline_status: {
      ci_pipelines: 8,
      cd_pipelines: 5,
      build_success_rate: 96.8,
      test_success_rate: 98.2,
      deployment_success_rate: 94.7,
      artifact_management: 'active'
    }
  });

  // Deployment controls
  let selectedScript = '';
  let selectedEnvironment = 'staging';
  let selectedNodeConfig = 'data-core';
  let automationEnabled = true;
  let approvalRequired = true;

  // Deployment automation scripts
  const automationScripts = [
    { name: '01-pre-deployment-validation.sh', description: 'Pre-deployment validation and checks' },
    { name: '02-frontend-deployment-safe.sh', description: 'Safe frontend deployment with rollback' },
    { name: '03-integration-test-automation.sh', description: 'Automated integration testing' },
    { name: '04-rollback-automation.sh', description: 'Automated rollback procedures' },
    { name: '05-health-monitoring-setup.sh', description: 'Health monitoring configuration' },
    { name: 'deploy_phases_7_8_9.sh', description: 'Complete phase deployment' }
  ];

  // Node configurations
  const nodeConfigurations = [
    { name: 'data-core', description: 'Data storage and processing core', status: 'healthy' },
    { name: 'processing-core', description: 'IntelOwl and analysis processing', status: 'healthy' },
    { name: 'edge-nodes', description: 'Edge computing nodes', status: 'syncing' }
  ];

  // Live data stores
  const activeDeployments = writable([]);
  const deploymentHistory = writable([]);
  const nodeStatus = writable([]);
  const automationLogs = writable([]);

  // WebSocket connections
  let deploymentWs: WebSocket | null = null;
  let nodeWs: WebSocket | null = null;
  let orchestrationWs: WebSocket | null = null;
  let automationWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadDeploymentData();
    loadNodeStatus();
    startAutomationMonitoring();
  });

  onDestroy(() => {
    if (deploymentWs) deploymentWs.close();
    if (nodeWs) nodeWs.close();
    if (orchestrationWs) orchestrationWs.close();
    if (automationWs) automationWs.close();
  });

  function initializeWebSockets() {
    // Deployment automation WebSocket
    deploymentWs = new WebSocket('ws://localhost:8100/deployment');
    deploymentWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      deploymentState.update(state => ({
        ...state,
        automation: { ...state.automation, ...data.automation }
      }));
      if (data.active_deployments) {
        activeDeployments.set(data.active_deployments);
      }
    };

    // Node configuration WebSocket
    nodeWs = new WebSocket('ws://localhost:8101/nodes');
    nodeWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      deploymentState.update(state => ({
        ...state,
        node_configuration: { ...state.node_configuration, ...data }
      }));
      if (data.node_status) {
        nodeStatus.set(data.node_status);
      }
    };

    // Cluster orchestration WebSocket
    orchestrationWs = new WebSocket('ws://localhost:8102/orchestration');
    orchestrationWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      deploymentState.update(state => ({
        ...state,
        cluster_orchestration: { ...state.cluster_orchestration, ...data }
      }));
    };

    // Automation logs WebSocket
    automationWs = new WebSocket('ws://localhost:8103/automation-logs');
    automationWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      automationLogs.update(logs => [data, ...logs.slice(0, 99)]);
    };
  }

  async function loadDeploymentData() {
    try {
      const [deploymentsRes, historyRes] = await Promise.all([
        fetch('http://localhost:8100/api/deployments'),
        fetch('http://localhost:8100/api/history')
      ]);

      const deployments = await deploymentsRes.json();
      const history = await historyRes.json();

      activeDeployments.set(deployments);
      deploymentHistory.set(history);
    } catch (error) {
      console.error('Failed to load deployment data:', error);
    }
  }

  async function loadNodeStatus() {
    try {
      const response = await fetch('http://localhost:8101/api/nodes');
      const nodes = await response.json();
      nodeStatus.set(nodes);
    } catch (error) {
      console.error('Failed to load node status:', error);
    }
  }

  async function startAutomationMonitoring() {
    setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8100/api/metrics');
        const metrics = await response.json();
        deploymentState.update(state => ({
          ...state,
          ...metrics
        }));
      } catch (error) {
        console.error('Automation monitoring error:', error);
      }
    }, 10000);
  }

  async function executeDeploymentScript() {
    if (!selectedScript) return;

    try {
      const response = await fetch('http://localhost:8100/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          script: selectedScript,
          environment: selectedEnvironment,
          automation_enabled: automationEnabled,
          approval_required: approvalRequired
        })
      });

      if (response.ok) {
        const deployment = await response.json();
        activeDeployments.update(deployments => [deployment, ...deployments]);
      }
    } catch (error) {
      console.error('Script execution failed:', error);
    }
  }

  async function configureNode() {
    try {
      const response = await fetch('http://localhost:8101/api/configure', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          node: selectedNodeConfig,
          environment: selectedEnvironment,
          sync_enabled: true
        })
      });

      if (response.ok) {
        console.log('Node configuration updated');
      }
    } catch (error) {
      console.error('Node configuration failed:', error);
    }
  }

  async function triggerClusterRebalance() {
    try {
      const response = await fetch('http://localhost:8102/api/rebalance', {
        method: 'POST'
      });

      if (response.ok) {
        console.log('Cluster rebalance initiated');
      }
    } catch (error) {
      console.error('Cluster rebalance failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': case 'active': case 'synchronized': return 'text-green-400';
      case 'syncing': case 'deploying': case 'running': return 'text-blue-400';
      case 'pending': case 'queued': return 'text-yellow-400';
      case 'failed': case 'error': case 'degraded': return 'text-red-400';
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
  <title>Deployment Automation Console | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-red-400 to-pink-500 bg-clip-text text-transparent">
        Deployment Automation Console
      </h1>
      <p class="text-gray-300">Unified deployment orchestration, node configuration, and cluster management</p>
    </div>

    {#if $deploymentState}
      <!-- Deployment Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
        <!-- Automation Status -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-red-400 mr-2"></span>
            Automation Status
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Scripts:</span>
              <span class="text-red-400">{$deploymentState.automation.total_scripts}</span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-blue-400">{$deploymentState.automation.active_deployments}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class={getHealthColor($deploymentState.automation.success_rate)}>{$deploymentState.automation.success_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Avg Time:</span>
              <span class="text-cyan-400">{$deploymentState.automation.avg_deployment_time}</span>
            </div>
          </div>
        </div>

        <!-- Node Configuration -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Node Configuration
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Total Nodes:</span>
              <span class="text-green-400">{$deploymentState.node_configuration.total_nodes}</span>
            </div>
            <div class="flex justify-between">
              <span>Health:</span>
              <span class={getHealthColor($deploymentState.node_configuration.node_health)}>{$deploymentState.node_configuration.node_health}%</span>
            </div>
            <div class="flex justify-between">
              <span>Sync Status:</span>
              <span class={getStatusColor($deploymentState.node_configuration.sync_status)}>
                {$deploymentState.node_configuration.sync_status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Config Drift:</span>
              <span class="text-green-400">{$deploymentState.node_configuration.config_drift}</span>
            </div>
          </div>
        </div>

        <!-- Cluster Orchestration -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Cluster Orchestration
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Health:</span>
              <span class={getHealthColor($deploymentState.cluster_orchestration.cluster_health)}>{$deploymentState.cluster_orchestration.cluster_health}%</span>
            </div>
            <div class="flex justify-between">
              <span>Efficiency:</span>
              <span class={getHealthColor($deploymentState.cluster_orchestration.coordination_efficiency, 85)}>{$deploymentState.cluster_orchestration.coordination_efficiency}%</span>
            </div>
            <div class="flex justify-between">
              <span>Load Balance:</span>
              <span class={getHealthColor($deploymentState.cluster_orchestration.load_balancing)}>{$deploymentState.cluster_orchestration.load_balancing}%</span>
            </div>
            <div class="flex justify-between">
              <span>Failover Ready:</span>
              <span class="text-green-400">{$deploymentState.cluster_orchestration.failover_readiness}%</span>
            </div>
          </div>
        </div>

        <!-- Pipeline Status -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Pipeline Status
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>CI Pipelines:</span>
              <span class="text-purple-400">{$deploymentState.pipeline_status.ci_pipelines}</span>
            </div>
            <div class="flex justify-between">
              <span>CD Pipelines:</span>
              <span class="text-blue-400">{$deploymentState.pipeline_status.cd_pipelines}</span>
            </div>
            <div class="flex justify-between">
              <span>Build Success:</span>
              <span class={getHealthColor($deploymentState.pipeline_status.build_success_rate)}>{$deploymentState.pipeline_status.build_success_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Deploy Success:</span>
              <span class={getHealthColor($deploymentState.pipeline_status.deployment_success_rate)}>{$deploymentState.pipeline_status.deployment_success_rate}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Deployment Control Panel -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <!-- Automation Scripts -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Deployment Automation Scripts</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Automation Script</label>
              <select bind:value={selectedScript} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="">Select Script</option>
                {#each automationScripts as script}
                  <option value={script.name}>{script.name}</option>
                {/each}
              </select>
            </div>
            {#if selectedScript}
              <div class="bg-gray-700 rounded p-3">
                <h4 class="font-medium mb-2">Script Details</h4>
                <p class="text-sm text-gray-300">
                  {automationScripts.find(s => s.name === selectedScript)?.description}
                </p>
              </div>
            {/if}
            <div>
              <label class="block text-sm font-medium mb-2">Target Environment</label>
              <select bind:value={selectedEnvironment} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="development">Development</option>
                <option value="staging">Staging</option>
                <option value="production">Production</option>
              </select>
            </div>
            <div class="space-y-2">
              <label class="flex items-center">
                <input type="checkbox" bind:checked={automationEnabled} class="mr-2">
                <span class="text-sm">Full Automation Enabled</span>
              </label>
              <label class="flex items-center">
                <input type="checkbox" bind:checked={approvalRequired} class="mr-2">
                <span class="text-sm">Manual Approval Required</span>
              </label>
            </div>
            <button
              on:click={executeDeploymentScript}
              disabled={!selectedScript}
              class="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-600 px-4 py-2 rounded font-semibold"
            >
              Execute Deployment
            </button>
          </div>
        </div>

        <!-- Node Configuration -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Node Configuration Management</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Node Configuration</label>
              <select bind:value={selectedNodeConfig} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                {#each nodeConfigurations as config}
                  <option value={config.name}>{config.name}</option>
                {/each}
              </select>
            </div>
            {#if selectedNodeConfig}
              <div class="bg-gray-700 rounded p-3">
                <h4 class="font-medium mb-2">Node Details</h4>
                <div class="text-sm text-gray-300">
                  {nodeConfigurations.find(n => n.name === selectedNodeConfig)?.description}
                </div>
              </div>
            {/if}
            <div class="bg-gray-700 rounded p-3">
              <h4 class="font-medium mb-2">Configuration Status</h4>
              <div class="grid grid-cols-3 gap-2 text-sm">
                {#each nodeConfigurations as config}
                  <div class="text-center">
                    <div class={getStatusColor(config.status)}>{config.name}</div>
                    <div class="text-gray-400">{config.status}</div>
                  </div>
                {/each}
              </div>
            </div>
            <button
              on:click={configureNode}
              class="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
            >
              Configure Node
            </button>
          </div>
        </div>

        <!-- Cluster Management -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Cluster Orchestration</h3>
          <div class="space-y-4">
            <div class="space-y-3">
              <button
                on:click={triggerClusterRebalance}
                class="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-left"
              >
                🔄 Rebalance Cluster
              </button>
              <button class="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded text-left">
                📊 Resource Analysis
              </button>
              <button class="w-full bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded text-left">
                🔧 Maintenance Mode
              </button>
              <button class="w-full bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded text-left">
                📈 Scale Cluster
              </button>
            </div>

            <div class="bg-gray-700 rounded p-3">
              <h4 class="font-medium mb-2">Cluster Metrics</h4>
              <div class="text-sm text-gray-300 space-y-1">
                <div>Health: {$deploymentState.cluster_orchestration.cluster_health}%</div>
                <div>Load Balance: {$deploymentState.cluster_orchestration.load_balancing}%</div>
                <div>Auto-Scale Events: {$deploymentState.cluster_orchestration.auto_scaling_events}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Active Deployments -->
      {#if $activeDeployments && $activeDeployments.length > 0}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
          <h3 class="text-lg font-semibold mb-4">Active Deployment Operations</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {#each $activeDeployments as deployment}
              <div class="bg-gray-700 rounded p-4">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">{deployment.script}</span>
                  <span class={getStatusColor(deployment.status)} class="text-xs">
                    {deployment.status.toUpperCase()}
                  </span>
                </div>
                <div class="text-sm text-gray-300 space-y-1">
                  <div>Environment: {deployment.environment}</div>
                  <div>Progress: {deployment.progress}%</div>
                  <div>Started: {deployment.start_time}</div>
                  <div>ETA: {deployment.eta}</div>
                </div>
                <div class="mt-2">
                  <div class="w-full bg-gray-600 rounded-full h-2">
                    <div class="bg-blue-400 h-2 rounded-full" style="width: {deployment.progress}%"></div>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Automation Logs -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
        <h3 class="text-lg font-semibold mb-4">Real-time Automation Logs</h3>
        <div class="bg-gray-900 rounded p-4 h-64 overflow-y-auto font-mono text-sm">
          {#if $automationLogs && $automationLogs.length > 0}
            {#each $automationLogs as log}
              <div class="mb-1 {log.level === 'ERROR' ? 'text-red-400' : log.level === 'WARN' ? 'text-yellow-400' : 'text-gray-300'}">
                <span class="text-gray-500">[{log.timestamp}]</span>
                <span class="text-blue-400">{log.script}</span>
                <span>{log.message}</span>
              </div>
            {/each}
          {:else}
            <div class="text-gray-400 text-center py-8">
              Automation logs will appear here in real-time
            </div>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>