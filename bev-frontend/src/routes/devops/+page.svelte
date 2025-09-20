<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // DevOps Platform state
  const devopsState = writable({
    deployment: {
      active_deployments: 3,
      pending_deployments: 2,
      successful_deployments: 147,
      failed_deployments: 8,
      rollback_count: 12,
      avg_deployment_time: '4m 32s',
      success_rate: 94.6,
      current_phase: 'Phase 9'
    },
    cicd: {
      active_pipelines: 8,
      build_queue: 4,
      test_success_rate: 97.8,
      artifact_count: 2847,
      pipeline_efficiency: 89.3,
      build_time_avg: '2m 18s',
      test_time_avg: '1m 47s'
    },
    infrastructure: {
      cluster_health: 96.7,
      node_count: 23,
      service_mesh_health: 94.2,
      load_balancer_efficiency: 91.8,
      auto_scaling_events: 47,
      resource_optimization: 87.4
    },
    rollback: {
      rollback_points: 23,
      last_rollback: '2d ago',
      rollback_success_rate: 100,
      avg_rollback_time: '47s',
      automated_rollbacks: 156,
      manual_rollbacks: 12
    }
  });

  // Active deployments and operations
  const activeDeployments = writable([]);
  const deploymentHistory = writable([]);
  const cicdPipelines = writable([]);

  // DevOps controls
  let selectedPhase = 'phase9';
  let deploymentStrategy = 'blue_green';
  let approvalRequired = true;
  let autoRollback = true;
  let deploymentConfig = {
    environment: 'production',
    replica_count: 3,
    health_check_timeout: 300,
    rollback_threshold: 5
  };

  // CI/CD Configuration
  let pipelineConfig = {
    trigger: 'git_push',
    build_parallel: true,
    test_parallel: true,
    security_scan: true,
    deploy_approval: true
  };

  // Cluster management
  let clusterOperations = {
    scaling_enabled: true,
    auto_healing: true,
    resource_limits: true,
    network_policies: true
  };

  // WebSocket connections
  let deploymentWs: WebSocket | null = null;
  let cicdWs: WebSocket | null = null;
  let clusterWs: WebSocket | null = null;
  let metricsWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadDeploymentData();
    loadPipelineData();
    startMetricsCollection();
  });

  onDestroy(() => {
    if (deploymentWs) deploymentWs.close();
    if (cicdWs) cicdWs.close();
    if (clusterWs) clusterWs.close();
    if (metricsWs) metricsWs.close();
  });

  function initializeWebSockets() {
    // Deployment orchestration WebSocket
    deploymentWs = new WebSocket('ws://localhost:8030/deployment');
    deploymentWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      devopsState.update(state => ({
        ...state,
        deployment: { ...state.deployment, ...data.deployment }
      }));
      if (data.active_deployments) {
        activeDeployments.set(data.active_deployments);
      }
    };

    // CI/CD pipeline WebSocket
    cicdWs = new WebSocket('ws://localhost:8031/cicd');
    cicdWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      devopsState.update(state => ({
        ...state,
        cicd: { ...state.cicd, ...data }
      }));
      if (data.pipelines) {
        cicdPipelines.set(data.pipelines);
      }
    };

    // Cluster management WebSocket
    clusterWs = new WebSocket('ws://localhost:8032/cluster');
    clusterWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      devopsState.update(state => ({
        ...state,
        infrastructure: { ...state.infrastructure, ...data }
      }));
    };

    // DevOps metrics WebSocket
    metricsWs = new WebSocket('ws://localhost:8033/metrics');
    metricsWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      devopsState.update(state => ({
        ...state,
        ...data
      }));
    };
  }

  async function loadDeploymentData() {
    try {
      const [deploymentsRes, historyRes] = await Promise.all([
        fetch('http://localhost:8030/api/deployments/active'),
        fetch('http://localhost:8030/api/deployments/history')
      ]);

      const deployments = await deploymentsRes.json();
      const history = await historyRes.json();

      activeDeployments.set(deployments);
      deploymentHistory.set(history);
    } catch (error) {
      console.error('Failed to load deployment data:', error);
    }
  }

  async function loadPipelineData() {
    try {
      const response = await fetch('http://localhost:8031/api/pipelines');
      const pipelines = await response.json();
      cicdPipelines.set(pipelines);
    } catch (error) {
      console.error('Failed to load pipeline data:', error);
    }
  }

  async function startMetricsCollection() {
    setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8033/api/metrics');
        const metrics = await response.json();
        devopsState.update(state => ({
          ...state,
          ...metrics
        }));
      } catch (error) {
        console.error('Metrics collection error:', error);
      }
    }, 10000);
  }

  async function startDeployment() {
    try {
      const response = await fetch('http://localhost:8030/api/deploy', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          phase: selectedPhase,
          strategy: deploymentStrategy,
          config: deploymentConfig,
          approval_required: approvalRequired,
          auto_rollback: autoRollback
        })
      });

      if (response.ok) {
        const deployment = await response.json();
        console.log('Deployment started:', deployment);
      }
    } catch (error) {
      console.error('Deployment failed:', error);
    }
  }

  async function triggerRollback(deploymentId: string) {
    try {
      const response = await fetch(`http://localhost:8030/api/rollback/${deploymentId}`, {
        method: 'POST'
      });

      if (response.ok) {
        console.log('Rollback initiated');
      }
    } catch (error) {
      console.error('Rollback failed:', error);
    }
  }

  async function approveDeployment(deploymentId: string) {
    try {
      const response = await fetch(`http://localhost:8030/api/approve/${deploymentId}`, {
        method: 'POST'
      });

      if (response.ok) {
        console.log('Deployment approved');
      }
    } catch (error) {
      console.error('Approval failed:', error);
    }
  }

  async function pauseDeployment(deploymentId: string) {
    try {
      const response = await fetch(`http://localhost:8030/api/pause/${deploymentId}`, {
        method: 'POST'
      });

      if (response.ok) {
        console.log('Deployment paused');
      }
    } catch (error) {
      console.error('Pause failed:', error);
    }
  }

  async function triggerCICDPipeline(pipelineId: string) {
    try {
      const response = await fetch(`http://localhost:8031/api/trigger/${pipelineId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pipelineConfig)
      });

      if (response.ok) {
        console.log('Pipeline triggered');
      }
    } catch (error) {
      console.error('Pipeline trigger failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'success': case 'completed': case 'healthy': return 'text-green-400';
      case 'running': case 'deploying': case 'building': return 'text-blue-400';
      case 'pending': case 'queued': case 'waiting': return 'text-yellow-400';
      case 'failed': case 'error': case 'unhealthy': return 'text-red-400';
      case 'paused': case 'stopped': return 'text-gray-400';
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
  <title>DevOps Automation Platform | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent">
        DevOps Automation Platform
      </h1>
      <p class="text-gray-300">Enterprise deployment orchestration and CI/CD management</p>
    </div>

    {#if $devopsState}
      <!-- DevOps Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
        <!-- Deployment Status -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-orange-400 mr-2"></span>
            Deployment Status
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-blue-400">{$devopsState.deployment.active_deployments}</span>
            </div>
            <div class="flex justify-between">
              <span>Pending:</span>
              <span class="text-yellow-400">{$devopsState.deployment.pending_deployments}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class={getHealthColor($devopsState.deployment.success_rate)}>{$devopsState.deployment.success_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Avg Time:</span>
              <span class="text-cyan-400">{$devopsState.deployment.avg_deployment_time}</span>
            </div>
          </div>
        </div>

        <!-- CI/CD Pipelines -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            CI/CD Pipelines
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-green-400">{$devopsState.cicd.active_pipelines}</span>
            </div>
            <div class="flex justify-between">
              <span>Queue:</span>
              <span class="text-yellow-400">{$devopsState.cicd.build_queue}</span>
            </div>
            <div class="flex justify-between">
              <span>Test Success:</span>
              <span class={getHealthColor($devopsState.cicd.test_success_rate)}>{$devopsState.cicd.test_success_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Efficiency:</span>
              <span class="text-purple-400">{$devopsState.cicd.pipeline_efficiency}%</span>
            </div>
          </div>
        </div>

        <!-- Infrastructure Health -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Infrastructure Health
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Cluster Health:</span>
              <span class={getHealthColor($devopsState.infrastructure.cluster_health)}>{$devopsState.infrastructure.cluster_health}%</span>
            </div>
            <div class="flex justify-between">
              <span>Nodes:</span>
              <span class="text-blue-400">{$devopsState.infrastructure.node_count}</span>
            </div>
            <div class="flex justify-between">
              <span>Service Mesh:</span>
              <span class={getHealthColor($devopsState.infrastructure.service_mesh_health)}>{$devopsState.infrastructure.service_mesh_health}%</span>
            </div>
            <div class="flex justify-between">
              <span>Auto-Scale Events:</span>
              <span class="text-green-400">{$devopsState.infrastructure.auto_scaling_events}</span>
            </div>
          </div>
        </div>

        <!-- Rollback Management -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-red-400 mr-2"></span>
            Rollback Management
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Rollback Points:</span>
              <span class="text-red-400">{$devopsState.rollback.rollback_points}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class="text-green-400">{$devopsState.rollback.rollback_success_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Avg Time:</span>
              <span class="text-cyan-400">{$devopsState.rollback.avg_rollback_time}</span>
            </div>
            <div class="flex justify-between">
              <span>Last Rollback:</span>
              <span class="text-gray-400">{$devopsState.rollback.last_rollback}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Deployment Control Center -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <!-- Deployment Configuration -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Deployment Configuration</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Deployment Phase</label>
              <select bind:value={selectedPhase} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="phase7">Phase 7 - Market Intelligence</option>
                <option value="phase8">Phase 8 - Security Operations</option>
                <option value="phase9">Phase 9 - Autonomous Systems</option>
                <option value="all_phases">All Phases</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Deployment Strategy</label>
              <select bind:value={deploymentStrategy} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="blue_green">Blue-Green Deployment</option>
                <option value="canary">Canary Release</option>
                <option value="rolling">Rolling Update</option>
                <option value="recreate">Recreate</option>
              </select>
            </div>
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="block text-sm font-medium mb-2">Environment</label>
                <select bind:value={deploymentConfig.environment} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="development">Development</option>
                  <option value="staging">Staging</option>
                  <option value="production">Production</option>
                </select>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Replicas</label>
                <input
                  type="number"
                  bind:value={deploymentConfig.replica_count}
                  min="1"
                  max="10"
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
            </div>
            <div class="space-y-2">
              <label class="flex items-center">
                <input type="checkbox" bind:checked={approvalRequired} class="mr-2">
                <span class="text-sm">Require Manual Approval</span>
              </label>
              <label class="flex items-center">
                <input type="checkbox" bind:checked={autoRollback} class="mr-2">
                <span class="text-sm">Auto-Rollback on Failure</span>
              </label>
            </div>
            <button
              on:click={startDeployment}
              class="w-full bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded font-semibold"
            >
              START DEPLOYMENT
            </button>
          </div>
        </div>

        <!-- CI/CD Pipeline Control -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">CI/CD Pipeline Control</h3>
          <div class="space-y-4">
            <div class="bg-gray-700 rounded p-3">
              <h4 class="font-medium mb-2">Pipeline Configuration</h4>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={pipelineConfig.build_parallel} class="mr-2">
                  <span class="text-sm">Parallel Building</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={pipelineConfig.test_parallel} class="mr-2">
                  <span class="text-sm">Parallel Testing</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={pipelineConfig.security_scan} class="mr-2">
                  <span class="text-sm">Security Scanning</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={pipelineConfig.deploy_approval} class="mr-2">
                  <span class="text-sm">Deployment Approval</span>
                </label>
              </div>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Trigger Condition</label>
              <select bind:value={pipelineConfig.trigger} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="git_push">Git Push</option>
                <option value="pull_request">Pull Request</option>
                <option value="manual">Manual Trigger</option>
                <option value="scheduled">Scheduled</option>
              </select>
            </div>
            <button
              on:click={() => triggerCICDPipeline('main')}
              class="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
            >
              Trigger Pipeline
            </button>
          </div>
        </div>

        <!-- Quick Actions -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Quick Actions</h3>
          <div class="space-y-3">
            <button class="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-left">
              🔄 Health Check All Services
            </button>
            <button class="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded text-left">
              📊 Generate Deployment Report
            </button>
            <button class="w-full bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded text-left">
              🔧 System Maintenance Mode
            </button>
            <button class="w-full bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-left">
              🚨 Emergency Rollback All
            </button>
            <button class="w-full bg-cyan-600 hover:bg-cyan-700 px-4 py-2 rounded text-left">
              📋 Export Configuration
            </button>
          </div>
        </div>
      </div>

      <!-- Active Deployments -->
      {#if $activeDeployments && $activeDeployments.length > 0}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
          <h3 class="text-lg font-semibold mb-4">Active Deployments</h3>
          <div class="space-y-4">
            {#each $activeDeployments as deployment}
              <div class="bg-gray-700 rounded p-4">
                <div class="flex justify-between items-center mb-3">
                  <div>
                    <span class="font-medium">{deployment.name || 'Deployment ' + deployment.id}</span>
                    <span class="text-sm text-gray-400 ml-2">({deployment.phase})</span>
                  </div>
                  <span class="text-xs px-2 py-1 rounded {getStatusColor(deployment.status)} bg-opacity-20">
                    {deployment.status.toUpperCase()}
                  </span>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-3">
                  <div class="text-sm">
                    <span class="text-gray-400">Strategy:</span>
                    <span class="text-blue-400 ml-1">{deployment.strategy}</span>
                  </div>
                  <div class="text-sm">
                    <span class="text-gray-400">Progress:</span>
                    <span class="text-green-400 ml-1">{deployment.progress}%</span>
                  </div>
                  <div class="text-sm">
                    <span class="text-gray-400">Started:</span>
                    <span class="text-gray-300 ml-1">{deployment.start_time}</span>
                  </div>
                  <div class="text-sm">
                    <span class="text-gray-400">ETA:</span>
                    <span class="text-cyan-400 ml-1">{deployment.eta}</span>
                  </div>
                </div>

                <div class="w-full bg-gray-600 rounded-full h-2 mb-3">
                  <div class="bg-blue-400 h-2 rounded-full" style="width: {deployment.progress}%"></div>
                </div>

                <div class="flex space-x-2">
                  {#if deployment.status === 'pending'}
                    <button
                      on:click={() => approveDeployment(deployment.id)}
                      class="bg-green-600 hover:bg-green-700 px-3 py-1 rounded text-sm"
                    >
                      Approve
                    </button>
                  {/if}
                  {#if deployment.status === 'running'}
                    <button
                      on:click={() => pauseDeployment(deployment.id)}
                      class="bg-yellow-600 hover:bg-yellow-700 px-3 py-1 rounded text-sm"
                    >
                      Pause
                    </button>
                  {/if}
                  <button
                    on:click={() => triggerRollback(deployment.id)}
                    class="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm"
                  >
                    Rollback
                  </button>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- CI/CD Pipeline Status -->
      {#if $cicdPipelines && $cicdPipelines.length > 0}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
          <h3 class="text-lg font-semibold mb-4">CI/CD Pipeline Status</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {#each $cicdPipelines as pipeline}
              <div class="bg-gray-700 rounded p-4">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">{pipeline.name}</span>
                  <span class="text-xs px-2 py-1 rounded {getStatusColor(pipeline.status)} bg-opacity-20">
                    {pipeline.status.toUpperCase()}
                  </span>
                </div>
                <div class="text-sm text-gray-300 space-y-1">
                  <div>Branch: {pipeline.branch}</div>
                  <div>Build: #{pipeline.build_number}</div>
                  <div>Duration: {pipeline.duration}</div>
                </div>
                <div class="mt-3">
                  <div class="w-full bg-gray-600 rounded-full h-2">
                    <div class="bg-green-400 h-2 rounded-full" style="width: {pipeline.progress}%"></div>
                  </div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Performance Metrics -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Deployment Metrics</h3>
          <div class="h-32 bg-gray-900 rounded flex items-center justify-center mb-4">
            <span class="text-gray-400">Deployment frequency chart</span>
          </div>
          <div class="text-sm text-gray-300">
            <div>Daily Deployments: 12</div>
            <div>Weekly Success Rate: 96.8%</div>
            <div>MTTR: 8m 23s</div>
          </div>
        </div>

        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Build Performance</h3>
          <div class="h-32 bg-gray-900 rounded flex items-center justify-center mb-4">
            <span class="text-gray-400">Build time trend chart</span>
          </div>
          <div class="text-sm text-gray-300">
            <div>Avg Build Time: {$devopsState.cicd.build_time_avg}</div>
            <div>Avg Test Time: {$devopsState.cicd.test_time_avg}</div>
            <div>Artifacts: {$devopsState.cicd.artifact_count.toLocaleString()}</div>
          </div>
        </div>

        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Infrastructure Efficiency</h3>
          <div class="h-32 bg-gray-900 rounded flex items-center justify-center mb-4">
            <span class="text-gray-400">Resource utilization chart</span>
          </div>
          <div class="text-sm text-gray-300">
            <div>Load Balancer: {$devopsState.infrastructure.load_balancer_efficiency}%</div>
            <div>Resource Optimization: {$devopsState.infrastructure.resource_optimization}%</div>
            <div>Auto-Scaling Events: {$devopsState.infrastructure.auto_scaling_events}</div>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>