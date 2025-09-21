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

  // Docker Infrastructure state
  const dockerState = writable({
    cluster: {
      total_services: 76,
      running_services: 73,
      healthy_services: 68,
      restarting_services: 3,
      failed_services: 2,
      cpu_usage: 67.3,
      memory_usage: 72.8,
      disk_usage: 45.2,
      network_io: '2.3GB/s'
    },
    databases: {
      postgresql: { status: 'healthy', replicas: 3, connections: 147 },
      neo4j: { status: 'healthy', replicas: 2, connections: 89 },
      redis: { status: 'healthy', replicas: 5, connections: 234 },
      elasticsearch: { status: 'healthy', replicas: 4, connections: 156 },
      mongodb: { status: 'healthy', replicas: 2, connections: 78 },
      influxdb: { status: 'healthy', replicas: 3, connections: 93 }
    },
    message_queues: {
      kafka: {
        status: 'healthy',
        brokers: 3,
        topics: 47,
        consumers: 23,
        throughput: '125MB/s'
      },
      rabbitmq: {
        status: 'healthy',
        nodes: 3,
        queues: 34,
        consumers: 18,
        throughput: '89MB/s'
      }
    },
    microservices: {
      total: 58,
      healthy: 54,
      scaling: 2,
      failed: 2,
      auto_scaling_enabled: true,
      load_balancing_active: true
    },
    monitoring: {
      prometheus: { status: 'active', targets: 156, alerts: 3 },
      grafana: { status: 'active', dashboards: 23, users: 7 },
      jaeger: { status: 'active', traces: '2.3M', spans: '47M' }
    }
  });

  // Infrastructure controls
  let selectedAction = '';
  let selectedServices = [];
  let scalingConfig = {
    service_name: '',
    target_replicas: 3,
    cpu_threshold: 70,
    memory_threshold: 80,
    auto_scaling: true
  };

  // Active operations
  const activeOperations = writable([]);
  const serviceCategories = [
    'Core Services', 'Databases', 'Message Queues', 'AI/ML Services',
    'Security Services', 'Monitoring', 'Networking', 'Storage'
  ];

  // WebSocket connections
  let dockerWs: WebSocket | null = null;
  let metricsWs: WebSocket | null = null;
  let logsWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadServiceStatus();
    startResourceMonitoring();
  });

  onDestroy(() => {
    if (dockerWs) dockerWs.close();
    if (metricsWs) metricsWs.close();
    if (logsWs) logsWs.close();
  });

  function initializeWebSockets() {
    // Docker daemon WebSocket
    dockerWs = new WebSocket('ws://${getWebSocketHost()}:8024/docker');
    dockerWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      dockerState.update(state => ({
        ...state,
        ...data
      }));
    };

    // Metrics WebSocket
    metricsWs = new WebSocket('ws://${getWebSocketHost()}:8025/metrics');
    metricsWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      dockerState.update(state => ({
        ...state,
        cluster: { ...state.cluster, ...data }
      }));
    };

    // Logs WebSocket
    logsWs = new WebSocket('ws://${getWebSocketHost()}:8026/logs');
    logsWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleLogEvent(data);
    };
  }

  async function loadServiceStatus() {
    try {
      const response = await fetch('http://${getServiceHost()}:8024/api/services');
      const services = await response.json();
      // Process service data
    } catch (error) {
      console.error('Failed to load service status:', error);
    }
  }

  async function startResourceMonitoring() {
    setInterval(async () => {
      try {
        const response = await fetch('http://${getServiceHost()}:8025/api/cluster-metrics');
        const metrics = await response.json();
        dockerState.update(state => ({
          ...state,
          cluster: { ...state.cluster, ...metrics }
        }));
      } catch (error) {
        console.error('Resource monitoring error:', error);
      }
    }, 5000);
  }

  async function executeClusterAction() {
    if (!selectedAction || selectedServices.length === 0) return;

    try {
      const response = await fetch('http://${getServiceHost()}:8024/api/cluster/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action: selectedAction,
          services: selectedServices,
          options: {}
        })
      });

      if (response.ok) {
        const result = await response.json();
        activeOperations.update(ops => [...ops, {
          id: result.operation_id,
          action: selectedAction,
          services: selectedServices,
          status: 'running',
          timestamp: Date.now()
        }]);
      }
    } catch (error) {
      console.error('Cluster action failed:', error);
    }
  }

  async function scaleService() {
    if (!scalingConfig.service_name) return;

    try {
      const response = await fetch('http://${getServiceHost()}:8024/api/scale', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scalingConfig)
      });

      if (response.ok) {
        console.log('Service scaling initiated');
      }
    } catch (error) {
      console.error('Service scaling failed:', error);
    }
  }

  async function performHealthCheck() {
    try {
      const response = await fetch('http://${getServiceHost()}:8024/api/health-check', {
        method: 'POST'
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Health check completed:', result);
      }
    } catch (error) {
      console.error('Health check failed:', error);
    }
  }

  async function triggerRollingUpdate() {
    try {
      const response = await fetch('http://${getServiceHost()}:8024/api/rolling-update', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          strategy: 'rolling',
          max_surge: 1,
          max_unavailable: 0
        })
      });

      if (response.ok) {
        console.log('Rolling update initiated');
      }
    } catch (error) {
      console.error('Rolling update failed:', error);
    }
  }

  function handleLogEvent(data: any) {
    // Process log events for alerts and notifications
    console.log('Log event:', data);
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'active': return 'text-green-400';
      case 'running': return 'text-blue-400';
      case 'restarting': return 'text-yellow-400';
      case 'failed': return 'text-red-400';
      case 'scaling': return 'text-purple-400';
      default: return 'text-gray-400';
    }
  }

  function getHealthColor(value: number, threshold: number = 90): string {
    if (value >= threshold) return 'text-green-400';
    if (value >= threshold - 15) return 'text-yellow-400';
    return 'text-red-400';
  }
</script>

<svelte:head>
  <title>Docker Infrastructure Command Center | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-cyan-500 bg-clip-text text-transparent">
        Docker Infrastructure Command Center
      </h1>
      <p class="text-gray-300">Enterprise container orchestration and microservice management</p>
    </div>

    {#if $dockerState}
      <!-- Cluster Overview -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
        <!-- Service Status -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Service Status
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Total Services:</span>
              <span class="text-blue-400">{$dockerState.cluster.total_services}</span>
            </div>
            <div class="flex justify-between">
              <span>Running:</span>
              <span class="text-green-400">{$dockerState.cluster.running_services}</span>
            </div>
            <div class="flex justify-between">
              <span>Healthy:</span>
              <span class={getHealthColor(($dockerState.cluster.healthy_services / $dockerState.cluster.total_services) * 100)}>
                {$dockerState.cluster.healthy_services}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Failed:</span>
              <span class="text-red-400">{$dockerState.cluster.failed_services}</span>
            </div>
          </div>
        </div>

        <!-- Resource Usage -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-yellow-400 mr-2"></span>
            Resource Usage
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>CPU:</span>
              <span class={getHealthColor(100 - $dockerState.cluster.cpu_usage, 30)}>{$dockerState.cluster.cpu_usage}%</span>
            </div>
            <div class="flex justify-between">
              <span>Memory:</span>
              <span class={getHealthColor(100 - $dockerState.cluster.memory_usage, 30)}>{$dockerState.cluster.memory_usage}%</span>
            </div>
            <div class="flex justify-between">
              <span>Disk:</span>
              <span class={getHealthColor(100 - $dockerState.cluster.disk_usage, 30)}>{$dockerState.cluster.disk_usage}%</span>
            </div>
            <div class="flex justify-between">
              <span>Network I/O:</span>
              <span class="text-cyan-400">{$dockerState.cluster.network_io}</span>
            </div>
          </div>
        </div>

        <!-- Microservices -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Microservices
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Total:</span>
              <span class="text-purple-400">{$dockerState.microservices.total}</span>
            </div>
            <div class="flex justify-between">
              <span>Healthy:</span>
              <span class="text-green-400">{$dockerState.microservices.healthy}</span>
            </div>
            <div class="flex justify-between">
              <span>Scaling:</span>
              <span class="text-blue-400">{$dockerState.microservices.scaling}</span>
            </div>
            <div class="flex justify-between">
              <span>Auto-Scale:</span>
              <span class={$dockerState.microservices.auto_scaling_enabled ? 'text-green-400' : 'text-gray-400'}>
                {$dockerState.microservices.auto_scaling_enabled ? 'ENABLED' : 'DISABLED'}
              </span>
            </div>
          </div>
        </div>

        <!-- Monitoring -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Monitoring Stack
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Prometheus:</span>
              <span class={getStatusColor($dockerState.monitoring.prometheus.status)}>
                {$dockerState.monitoring.prometheus.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Grafana:</span>
              <span class={getStatusColor($dockerState.monitoring.grafana.status)}>
                {$dockerState.monitoring.grafana.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Targets:</span>
              <span class="text-blue-400">{$dockerState.monitoring.prometheus.targets}</span>
            </div>
            <div class="flex justify-between">
              <span>Alerts:</span>
              <span class="text-red-400">{$dockerState.monitoring.prometheus.alerts}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Database Cluster Status -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
        <h3 class="text-lg font-semibold mb-4">Database Cluster Management</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          {#each Object.entries($dockerState.databases) as [dbName, dbData]}
            <div class="bg-gray-700 rounded p-4">
              <div class="flex justify-between items-center mb-2">
                <span class="font-medium capitalize">{dbName}</span>
                <span class={getStatusColor(dbData.status)} class="text-xs">
                  {dbData.status.toUpperCase()}
                </span>
              </div>
              <div class="text-sm text-gray-300 space-y-1">
                <div>Replicas: {dbData.replicas}</div>
                <div>Connections: {dbData.connections}</div>
              </div>
              <div class="mt-2">
                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-blue-400 h-2 rounded-full" style="width: {Math.min((dbData.connections / 200) * 100, 100)}%"></div>
                </div>
              </div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Message Queue Status -->
      <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
        <h3 class="text-lg font-semibold mb-4">Message Queue Infrastructure</h3>
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <!-- Kafka Cluster -->
          <div class="bg-gray-700 rounded p-4">
            <h4 class="font-medium mb-3 flex items-center">
              <span class="w-2 h-2 rounded-full bg-orange-400 mr-2"></span>
              Apache Kafka Cluster
            </h4>
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div class="text-gray-400">Status</div>
                <div class={getStatusColor($dockerState.message_queues.kafka.status)}>
                  {$dockerState.message_queues.kafka.status.toUpperCase()}
                </div>
              </div>
              <div>
                <div class="text-gray-400">Brokers</div>
                <div class="text-orange-400">{$dockerState.message_queues.kafka.brokers}</div>
              </div>
              <div>
                <div class="text-gray-400">Topics</div>
                <div class="text-blue-400">{$dockerState.message_queues.kafka.topics}</div>
              </div>
              <div>
                <div class="text-gray-400">Consumers</div>
                <div class="text-green-400">{$dockerState.message_queues.kafka.consumers}</div>
              </div>
            </div>
            <div class="mt-3">
              <div class="text-gray-400 text-sm">Throughput</div>
              <div class="text-lg font-semibold text-orange-400">{$dockerState.message_queues.kafka.throughput}</div>
            </div>
          </div>

          <!-- RabbitMQ Cluster -->
          <div class="bg-gray-700 rounded p-4">
            <h4 class="font-medium mb-3 flex items-center">
              <span class="w-2 h-2 rounded-full bg-green-400 mr-2"></span>
              RabbitMQ Cluster
            </h4>
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <div class="text-gray-400">Status</div>
                <div class={getStatusColor($dockerState.message_queues.rabbitmq.status)}>
                  {$dockerState.message_queues.rabbitmq.status.toUpperCase()}
                </div>
              </div>
              <div>
                <div class="text-gray-400">Nodes</div>
                <div class="text-green-400">{$dockerState.message_queues.rabbitmq.nodes}</div>
              </div>
              <div>
                <div class="text-gray-400">Queues</div>
                <div class="text-blue-400">{$dockerState.message_queues.rabbitmq.queues}</div>
              </div>
              <div>
                <div class="text-gray-400">Consumers</div>
                <div class="text-purple-400">{$dockerState.message_queues.rabbitmq.consumers}</div>
              </div>
            </div>
            <div class="mt-3">
              <div class="text-gray-400 text-sm">Throughput</div>
              <div class="text-lg font-semibold text-green-400">{$dockerState.message_queues.rabbitmq.throughput}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Control Panel -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <!-- Cluster Actions -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Cluster Operations</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Action Type</label>
              <select bind:value={selectedAction} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="">Select Action</option>
                <option value="restart">Restart Services</option>
                <option value="update">Update Services</option>
                <option value="scale">Scale Services</option>
                <option value="health_check">Health Check</option>
                <option value="backup">Backup Data</option>
                <option value="restore">Restore Data</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Service Category</label>
              <select multiple class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 h-24">
                {#each serviceCategories as category}
                  <option value={category}>{category}</option>
                {/each}
              </select>
            </div>
            <button
              on:click={executeClusterAction}
              disabled={!selectedAction}
              class="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-2 rounded"
            >
              Execute Cluster Action
            </button>
          </div>
        </div>

        <!-- Service Scaling -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Auto-Scaling Configuration</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Service Name</label>
              <input
                bind:value={scalingConfig.service_name}
                placeholder="Enter service name..."
                class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
              >
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Target Replicas: {scalingConfig.target_replicas}</label>
              <input
                type="range"
                bind:value={scalingConfig.target_replicas}
                min="1"
                max="20"
                class="w-full"
              >
            </div>
            <div class="grid grid-cols-2 gap-2">
              <div>
                <label class="block text-sm font-medium mb-2">CPU Threshold</label>
                <input
                  type="number"
                  bind:value={scalingConfig.cpu_threshold}
                  min="1"
                  max="100"
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Memory Threshold</label>
                <input
                  type="number"
                  bind:value={scalingConfig.memory_threshold}
                  min="1"
                  max="100"
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
            </div>
            <div class="flex items-center">
              <input type="checkbox" bind:checked={scalingConfig.auto_scaling} id="auto-scaling" class="mr-2">
              <label for="auto-scaling" class="text-sm">Enable Auto-Scaling</label>
            </div>
            <button
              on:click={scaleService}
              disabled={!scalingConfig.service_name}
              class="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded"
            >
              Configure Scaling
            </button>
          </div>
        </div>

        <!-- Health & Updates -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Health & Updates</h3>
          <div class="space-y-4">
            <button
              on:click={performHealthCheck}
              class="w-full bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded"
            >
              Run Health Check
            </button>
            <button
              on:click={triggerRollingUpdate}
              class="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded"
            >
              Rolling Update
            </button>
            <div class="bg-gray-700 rounded p-3">
              <h4 class="font-medium mb-2">Update Status</h4>
              <div class="text-sm text-gray-300">
                <div>Last Update: 2h ago</div>
                <div>Success Rate: 98.7%</div>
                <div>Rollback Count: 2</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Resource Utilization Charts -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">CPU Usage Trend</h3>
          <div class="h-32 bg-gray-900 rounded flex items-center justify-center">
            <span class="text-gray-400">CPU utilization chart</span>
          </div>
          <div class="mt-2 text-sm text-gray-300">
            Current: {$dockerState.cluster.cpu_usage}% | Avg: 64.2%
          </div>
        </div>

        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Memory Usage Trend</h3>
          <div class="h-32 bg-gray-900 rounded flex items-center justify-center">
            <span class="text-gray-400">Memory utilization chart</span>
          </div>
          <div class="mt-2 text-sm text-gray-300">
            Current: {$dockerState.cluster.memory_usage}% | Avg: 69.8%
          </div>
        </div>

        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Network I/O</h3>
          <div class="h-32 bg-gray-900 rounded flex items-center justify-center">
            <span class="text-gray-400">Network I/O chart</span>
          </div>
          <div class="mt-2 text-sm text-gray-300">
            Current: {$dockerState.cluster.network_io} | Peak: 3.1GB/s
          </div>
        </div>
      </div>

      <!-- Active Operations -->
      {#if $activeOperations && $activeOperations.length > 0}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Active Operations</h3>
          <div class="space-y-3">
            {#each $activeOperations as operation}
              <div class="bg-gray-700 rounded p-4">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">{operation.action}</span>
                  <span class="text-xs px-2 py-1 rounded bg-blue-600">{operation.status}</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Services: {operation.services.join(', ')}</div>
                  <div>Started: {new Date(operation.timestamp).toLocaleTimeString()}</div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>