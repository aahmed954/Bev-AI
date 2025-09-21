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

  // Auto-Recovery System state
  const recoveryState = writable({
    system_health: {
      overall_health: 96.8,
      critical_services: 67,
      healthy_services: 64,
      degraded_services: 2,
      failed_services: 1,
      recovery_events_24h: 12,
      success_rate: 97.3
    },
    auto_recovery: {
      status: 'active',
      recovery_strategies: 15,
      active_recoveries: 3,
      completed_recoveries: 147,
      failed_recoveries: 8,
      avg_recovery_time: '2m 34s',
      escalation_rate: 5.2
    },
    failure_detection: {
      monitoring_agents: 67,
      detection_accuracy: 98.7,
      false_positive_rate: 1.3,
      mean_time_to_detection: '12s',
      alert_channels: 8,
      auto_diagnosis: true
    },
    disaster_recovery: {
      backup_points: 234,
      last_full_backup: '6h ago',
      rto_compliance: 94.7,
      rpo_compliance: 98.1,
      failover_ready: true,
      geo_redundancy: true
    },
    resource_management: {
      cpu_optimization: 87.4,
      memory_optimization: 91.2,
      disk_optimization: 85.7,
      network_optimization: 94.3,
      cost_optimization: 78.9,
      predictive_scaling: true
    }
  });

  // Recovery controls
  let selectedService = '';
  let recoveryAction = '';
  let manualRecoveryMode = false;
  let escalationLevel = 1;

  // Recovery configuration
  let recoveryConfig = {
    auto_recovery_enabled: true,
    max_retry_attempts: 3,
    escalation_timeout: 300,
    notification_channels: ['slack', 'email', 'webhook'],
    backup_before_recovery: true,
    rollback_on_failure: true
  };

  // Disaster recovery configuration
  let drConfig = {
    backup_frequency: 'hourly',
    retention_period: 30,
    geo_replication: true,
    encryption_enabled: true,
    compression_enabled: true,
    incremental_backups: true
  };

  // Live data stores
  const activeRecoveries = writable([]);
  const failureAlerts = writable([]);
  const recoveryHistory = writable([]);
  const systemMetrics = writable({});

  // WebSocket connections
  let recoveryWs: WebSocket | null = null;
  let healthWs: WebSocket | null = null;
  let alertsWs: WebSocket | null = null;
  let metricsWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadRecoveryData();
    startHealthMonitoring();
  });

  onDestroy(() => {
    if (recoveryWs) recoveryWs.close();
    if (healthWs) healthWs.close();
    if (alertsWs) alertsWs.close();
    if (metricsWs) metricsWs.close();
  });

  function initializeWebSockets() {
    // Auto-recovery WebSocket
    recoveryWs = new WebSocket('ws://${getWebSocketHost()}:8070/recovery');
    recoveryWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      recoveryState.update(state => ({
        ...state,
        auto_recovery: { ...state.auto_recovery, ...data.recovery }
      }));
      if (data.active_recoveries) {
        activeRecoveries.set(data.active_recoveries);
      }
    };

    // System health WebSocket
    healthWs = new WebSocket('ws://${getWebSocketHost()}:8071/health');
    healthWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      recoveryState.update(state => ({
        ...state,
        system_health: { ...state.system_health, ...data }
      }));
    };

    // Failure alerts WebSocket
    alertsWs = new WebSocket('ws://${getWebSocketHost()}:8072/alerts');
    alertsWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      failureAlerts.update(alerts => [data, ...alerts.slice(0, 49)]);
    };

    // Resource metrics WebSocket
    metricsWs = new WebSocket('ws://${getWebSocketHost()}:8073/metrics');
    metricsWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      systemMetrics.set(data);
      recoveryState.update(state => ({
        ...state,
        resource_management: { ...state.resource_management, ...data.resource_management }
      }));
    };
  }

  async function loadRecoveryData() {
    try {
      const [recoveriesRes, historyRes, alertsRes] = await Promise.all([
        fetch('http://${getServiceHost()}:8070/api/active'),
        fetch('http://${getServiceHost()}:8070/api/history'),
        fetch('http://${getServiceHost()}:8072/api/recent')
      ]);

      const recoveries = await recoveriesRes.json();
      const history = await historyRes.json();
      const alerts = await alertsRes.json();

      activeRecoveries.set(recoveries);
      recoveryHistory.set(history);
      failureAlerts.set(alerts);
    } catch (error) {
      console.error('Failed to load recovery data:', error);
    }
  }

  async function startHealthMonitoring() {
    setInterval(async () => {
      try {
        const response = await fetch('http://${getServiceHost()}:8071/api/health');
        const health = await response.json();
        recoveryState.update(state => ({
          ...state,
          system_health: { ...state.system_health, ...health }
        }));
      } catch (error) {
        console.error('Health monitoring error:', error);
      }
    }, 5000);
  }

  async function triggerManualRecovery() {
    if (!selectedService || !recoveryAction) return;

    try {
      const response = await fetch('http://${getServiceHost()}:8070/api/recover', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          service: selectedService,
          action: recoveryAction,
          escalation_level: escalationLevel,
          manual_mode: manualRecoveryMode
        })
      });

      if (response.ok) {
        const recovery = await response.json();
        console.log('Manual recovery initiated:', recovery);
      }
    } catch (error) {
      console.error('Manual recovery failed:', error);
    }
  }

  async function updateRecoveryConfig() {
    try {
      const response = await fetch('http://${getServiceHost()}:8070/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(recoveryConfig)
      });

      if (response.ok) {
        console.log('Recovery configuration updated');
      }
    } catch (error) {
      console.error('Configuration update failed:', error);
    }
  }

  async function triggerDisasterRecovery() {
    try {
      const response = await fetch('http://${getServiceHost()}:8074/api/disaster-recovery', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(drConfig)
      });

      if (response.ok) {
        console.log('Disaster recovery initiated');
      }
    } catch (error) {
      console.error('Disaster recovery failed:', error);
    }
  }

  async function createSystemBackup() {
    try {
      const response = await fetch('http://${getServiceHost()}:8074/api/backup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          type: 'full',
          encryption: true,
          compression: true
        })
      });

      if (response.ok) {
        console.log('System backup initiated');
      }
    } catch (error) {
      console.error('Backup failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'healthy': case 'active': case 'completed': return 'text-green-400';
      case 'recovering': case 'running': case 'processing': return 'text-blue-400';
      case 'degraded': case 'warning': return 'text-yellow-400';
      case 'failed': case 'critical': case 'error': return 'text-red-400';
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
  <title>Auto-Recovery Management | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-emerald-400 to-teal-500 bg-clip-text text-transparent">
        Auto-Recovery Management System
      </h1>
      <p class="text-gray-300">Intelligent system recovery, disaster recovery, and resource optimization</p>
    </div>

    {#if $recoveryState}
      <!-- Recovery Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-5 gap-6 mb-8">
        <!-- System Health -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            System Health
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Overall Health:</span>
              <span class={getHealthColor($recoveryState.system_health.overall_health)}>{$recoveryState.system_health.overall_health}%</span>
            </div>
            <div class="flex justify-between">
              <span>Healthy:</span>
              <span class="text-green-400">{$recoveryState.system_health.healthy_services}</span>
            </div>
            <div class="flex justify-between">
              <span>Degraded:</span>
              <span class="text-yellow-400">{$recoveryState.system_health.degraded_services}</span>
            </div>
            <div class="flex justify-between">
              <span>Failed:</span>
              <span class="text-red-400">{$recoveryState.system_health.failed_services}</span>
            </div>
          </div>
        </div>

        <!-- Auto-Recovery -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Auto-Recovery
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($recoveryState.auto_recovery.status)}>
                {$recoveryState.auto_recovery.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-blue-400">{$recoveryState.auto_recovery.active_recoveries}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class={getHealthColor($recoveryState.auto_recovery.success_rate)}>{$recoveryState.auto_recovery.success_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Avg Time:</span>
              <span class="text-cyan-400">{$recoveryState.auto_recovery.avg_recovery_time}</span>
            </div>
          </div>
        </div>

        <!-- Failure Detection -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-yellow-400 mr-2"></span>
            Failure Detection
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Accuracy:</span>
              <span class={getHealthColor($recoveryState.failure_detection.detection_accuracy)}>{$recoveryState.failure_detection.detection_accuracy}%</span>
            </div>
            <div class="flex justify-between">
              <span>False Positive:</span>
              <span class="text-yellow-400">{$recoveryState.failure_detection.false_positive_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>MTTD:</span>
              <span class="text-green-400">{$recoveryState.failure_detection.mean_time_to_detection}</span>
            </div>
            <div class="flex justify-between">
              <span>Agents:</span>
              <span class="text-blue-400">{$recoveryState.failure_detection.monitoring_agents}</span>
            </div>
          </div>
        </div>

        <!-- Disaster Recovery -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-red-400 mr-2"></span>
            Disaster Recovery
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Backup Points:</span>
              <span class="text-red-400">{$recoveryState.disaster_recovery.backup_points}</span>
            </div>
            <div class="flex justify-between">
              <span>RTO Compliance:</span>
              <span class={getHealthColor($recoveryState.disaster_recovery.rto_compliance)}>{$recoveryState.disaster_recovery.rto_compliance}%</span>
            </div>
            <div class="flex justify-between">
              <span>RPO Compliance:</span>
              <span class={getHealthColor($recoveryState.disaster_recovery.rpo_compliance)}>{$recoveryState.disaster_recovery.rpo_compliance}%</span>
            </div>
            <div class="flex justify-between">
              <span>Last Backup:</span>
              <span class="text-gray-400">{$recoveryState.disaster_recovery.last_full_backup}</span>
            </div>
          </div>
        </div>

        <!-- Resource Optimization -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Resource Optimization
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>CPU:</span>
              <span class={getHealthColor($recoveryState.resource_management.cpu_optimization, 85)}>{$recoveryState.resource_management.cpu_optimization}%</span>
            </div>
            <div class="flex justify-between">
              <span>Memory:</span>
              <span class={getHealthColor($recoveryState.resource_management.memory_optimization, 85)}>{$recoveryState.resource_management.memory_optimization}%</span>
            </div>
            <div class="flex justify-between">
              <span>Network:</span>
              <span class={getHealthColor($recoveryState.resource_management.network_optimization, 85)}>{$recoveryState.resource_management.network_optimization}%</span>
            </div>
            <div class="flex justify-between">
              <span>Cost Savings:</span>
              <span class="text-green-400">{$recoveryState.resource_management.cost_optimization}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Recovery Control Center -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <!-- Manual Recovery Controls -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Manual Recovery Controls</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Target Service</label>
              <select bind:value={selectedService} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="">Select Service</option>
                <option value="postgres">PostgreSQL Cluster</option>
                <option value="neo4j">Neo4j Database</option>
                <option value="redis">Redis Cluster</option>
                <option value="elasticsearch">Elasticsearch</option>
                <option value="kafka">Kafka Cluster</option>
                <option value="rabbitmq">RabbitMQ Cluster</option>
                <option value="intelowl">IntelOwl Platform</option>
                <option value="autonomous-coordinator">Autonomous Systems</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Recovery Action</label>
              <select bind:value={recoveryAction} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="">Select Action</option>
                <option value="restart">Restart Service</option>
                <option value="scale_up">Scale Up</option>
                <option value="rollback">Rollback to Previous Version</option>
                <option value="failover">Failover to Backup</option>
                <option value="restore">Restore from Backup</option>
                <option value="reset">Reset to Clean State</option>
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Escalation Level: {escalationLevel}</label>
              <input
                type="range"
                bind:value={escalationLevel}
                min="1"
                max="5"
                class="w-full"
              >
              <div class="flex justify-between text-xs text-gray-400 mt-1">
                <span>Auto</span>
                <span>Manual</span>
                <span>Escalated</span>
                <span>Critical</span>
                <span>Emergency</span>
              </div>
            </div>
            <div class="flex items-center">
              <input type="checkbox" bind:checked={manualRecoveryMode} id="manual-mode" class="mr-2">
              <label for="manual-mode" class="text-sm">Manual Recovery Mode</label>
            </div>
            <button
              on:click={triggerManualRecovery}
              disabled={!selectedService || !recoveryAction}
              class="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-2 rounded"
            >
              Trigger Recovery
            </button>
          </div>
        </div>

        <!-- Recovery Configuration -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Recovery Configuration</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Max Retry Attempts: {recoveryConfig.max_retry_attempts}</label>
              <input
                type="range"
                bind:value={recoveryConfig.max_retry_attempts}
                min="1"
                max="10"
                class="w-full"
              >
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Escalation Timeout (s): {recoveryConfig.escalation_timeout}</label>
              <input
                type="range"
                bind:value={recoveryConfig.escalation_timeout}
                min="60"
                max="1800"
                step="60"
                class="w-full"
              >
            </div>
            <div class="space-y-2">
              <label class="flex items-center">
                <input type="checkbox" bind:checked={recoveryConfig.auto_recovery_enabled} class="mr-2">
                <span class="text-sm">Auto-Recovery Enabled</span>
              </label>
              <label class="flex items-center">
                <input type="checkbox" bind:checked={recoveryConfig.backup_before_recovery} class="mr-2">
                <span class="text-sm">Backup Before Recovery</span>
              </label>
              <label class="flex items-center">
                <input type="checkbox" bind:checked={recoveryConfig.rollback_on_failure} class="mr-2">
                <span class="text-sm">Rollback on Failure</span>
              </label>
            </div>
            <button
              on:click={updateRecoveryConfig}
              class="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
            >
              Update Configuration
            </button>
          </div>
        </div>

        <!-- Emergency Actions -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Emergency Actions</h3>
          <div class="space-y-3">
            <button
              on:click={createSystemBackup}
              class="w-full bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded text-left"
            >
              🔄 Create System Backup
            </button>
            <button
              on:click={triggerDisasterRecovery}
              class="w-full bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-left"
            >
              🚨 Disaster Recovery
            </button>
            <button class="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded text-left">
              🔧 System Maintenance Mode
            </button>
            <button class="w-full bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded text-left">
              📊 Generate Recovery Report
            </button>
          </div>

          <div class="mt-4 p-3 bg-gray-700 rounded">
            <h4 class="font-medium mb-2">Recovery Statistics</h4>
            <div class="text-sm text-gray-300 space-y-1">
              <div>Events (24h): {$recoveryState.system_health.recovery_events_24h}</div>
              <div>Success Rate: {$recoveryState.auto_recovery.success_rate}%</div>
              <div>Escalation Rate: {$recoveryState.auto_recovery.escalation_rate}%</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Active Recovery Operations -->
      {#if $activeRecoveries && $activeRecoveries.length > 0}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
          <h3 class="text-lg font-semibold mb-4">Active Recovery Operations</h3>
          <div class="space-y-4">
            {#each $activeRecoveries as recovery}
              <div class="bg-gray-700 rounded p-4">
                <div class="flex justify-between items-center mb-3">
                  <div>
                    <span class="font-medium">{recovery.service}</span>
                    <span class="text-sm text-gray-400 ml-2">({recovery.action})</span>
                  </div>
                  <span class="text-xs px-2 py-1 rounded {getStatusColor(recovery.status)} bg-opacity-20">
                    {recovery.status.toUpperCase()}
                  </span>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-3">
                  <div class="text-sm">
                    <span class="text-gray-400">Started:</span>
                    <span class="text-gray-300 ml-1">{recovery.start_time}</span>
                  </div>
                  <div class="text-sm">
                    <span class="text-gray-400">Progress:</span>
                    <span class="text-blue-400 ml-1">{recovery.progress}%</span>
                  </div>
                  <div class="text-sm">
                    <span class="text-gray-400">Attempt:</span>
                    <span class="text-yellow-400 ml-1">{recovery.attempt}/{recovery.max_attempts}</span>
                  </div>
                  <div class="text-sm">
                    <span class="text-gray-400">ETA:</span>
                    <span class="text-cyan-400 ml-1">{recovery.eta}</span>
                  </div>
                </div>

                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-blue-400 h-2 rounded-full" style="width: {recovery.progress}%"></div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Failure Alerts -->
      {#if $failureAlerts && $failureAlerts.length > 0}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Recent Failure Alerts</h3>
          <div class="space-y-3">
            {#each $failureAlerts.slice(0, 10) as alert}
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">{alert.service}</span>
                  <span class={getStatusColor(alert.severity)} class="text-xs">
                    {alert.severity.toUpperCase()}
                  </span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Issue: {alert.description}</div>
                  <div>Time: {alert.timestamp}</div>
                  <div>Recovery: {alert.recovery_status}</div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>