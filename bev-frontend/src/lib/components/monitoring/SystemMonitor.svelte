<!-- BEV System Monitor - Real-time Performance & Security Monitoring -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { mcpStore } from '$lib/stores/mcpStore';
  import Chart from 'chart.js/auto';
  import type { SystemMetrics } from '$lib/mcp/types';
  
  let cpuChart: Chart | null = null;
  let networkChart: Chart | null = null;
  let cpuCanvas: HTMLCanvasElement;
  let networkCanvas: HTMLCanvasElement;
  
  let cpuHistory: number[] = Array(20).fill(0);
  let networkInHistory: number[] = Array(20).fill(0);
  let networkOutHistory: number[] = Array(20).fill(0);
  let timeLabels: string[] = Array(20).fill('');
  
  // Update charts when metrics change
  $: if ($mcpStore.systemMetrics) {
    updateCharts($mcpStore.systemMetrics);
  }
  
  function updateCharts(metrics: SystemMetrics) {
    // Update data arrays
    cpuHistory = [...cpuHistory.slice(1), metrics.resources.cpu];
    networkInHistory = [...networkInHistory.slice(1), metrics.resources.network.in];
    networkOutHistory = [...networkOutHistory.slice(1), metrics.resources.network.out];
    timeLabels = [...timeLabels.slice(1), new Date().toLocaleTimeString()];
    
    // Update CPU chart
    if (cpuChart) {      cpuChart.data.labels = timeLabels;
      cpuChart.data.datasets[0].data = cpuHistory;
      cpuChart.update('none');
    }
    
    // Update network chart
    if (networkChart) {
      networkChart.data.labels = timeLabels;
      networkChart.data.datasets[0].data = networkInHistory;
      networkChart.data.datasets[1].data = networkOutHistory;
      networkChart.update('none');
    }
  }
  
  function initCharts() {
    // CPU Usage Chart
    if (cpuCanvas) {
      cpuChart = new Chart(cpuCanvas, {
        type: 'line',
        data: {
          labels: timeLabels,
          datasets: [{
            label: 'CPU %',
            data: cpuHistory,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            tension: 0.4,
            fill: true
          }]
        },        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              grid: { color: '#1f2937' },
              ticks: { color: '#6b7280' }
            },
            x: {
              display: false
            }
          }
        }
      });
    }
    
    // Network Chart
    if (networkCanvas) {
      networkChart = new Chart(networkCanvas, {
        type: 'line',
        data: {
          labels: timeLabels,
          datasets: [{
            label: 'IN',
            data: networkInHistory,            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.1)',
            tension: 0.4,
            fill: true
          }, {
            label: 'OUT',
            data: networkOutHistory,
            borderColor: '#f59e0b',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            tension: 0.4,
            fill: true
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: true,
              labels: { color: '#6b7280' }
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              grid: { color: '#1f2937' },
              ticks: { color: '#6b7280' }
            },
            x: {
              display: false            }
          }
        }
      });
    }
  }
  
  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B/s`;
    if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB/s`;
    return `${(bytes / 1048576).toFixed(1)} MB/s`;
  }
  
  onMount(() => {
    initCharts();
  });
  
  onDestroy(() => {
    if (cpuChart) cpuChart.destroy();
    if (networkChart) networkChart.destroy();
  });
</script>

<div class="system-monitor bg-black text-green-400 font-mono p-4 space-y-4">
  <!-- Header -->
  <div class="header mb-4">
    <h2 class="text-xl font-bold text-green-400">SYSTEM MONITOR</h2>
    <p class="text-xs text-gray-500">Real-time performance metrics</p>  </div>
  
  <!-- Quick Stats Grid -->
  <div class="stats-grid grid grid-cols-2 md:grid-cols-4 gap-4">
    <div class="stat-card bg-gray-900 border border-gray-700 rounded p-3">
      <div class="text-xs text-gray-500 mb-1">CPU USAGE</div>
      <div class="text-2xl font-bold text-cyan-400">
        {$mcpStore.systemMetrics?.resources.cpu.toFixed(1) || '0'}%
      </div>
    </div>
    
    <div class="stat-card bg-gray-900 border border-gray-700 rounded p-3">
      <div class="text-xs text-gray-500 mb-1">MEMORY</div>
      <div class="text-2xl font-bold text-yellow-400">
        {$mcpStore.systemMetrics?.resources.memory.toFixed(1) || '0'}%
      </div>
    </div>
    
    <div class="stat-card bg-gray-900 border border-gray-700 rounded p-3">
      <div class="text-xs text-gray-500 mb-1">DISK</div>
      <div class="text-2xl font-bold text-orange-400">
        {$mcpStore.systemMetrics?.resources.disk.toFixed(1) || '0'}%
      </div>
    </div>
    
    <div class="stat-card bg-gray-900 border border-gray-700 rounded p-3">
      <div class="text-xs text-gray-500 mb-1">AGENTS</div>
      <div class="text-2xl font-bold text-green-400">
        {$mcpStore.systemMetrics?.mcp.activeAgents || 0}      </div>
    </div>
  </div>
  
  <!-- Charts Row -->
  <div class="charts-row grid grid-cols-1 md:grid-cols-2 gap-4">
    <!-- CPU Chart -->
    <div class="chart-container bg-gray-900 border border-gray-700 rounded p-4">
      <h3 class="text-sm font-bold text-green-400 mb-2">CPU USAGE</h3>
      <div class="h-32">
        <canvas bind:this={cpuCanvas}></canvas>
      </div>
    </div>
    
    <!-- Network Chart -->
    <div class="chart-container bg-gray-900 border border-gray-700 rounded p-4">
      <h3 class="text-sm font-bold text-green-400 mb-2">NETWORK I/O</h3>
      <div class="h-32">
        <canvas bind:this={networkCanvas}></canvas>
      </div>
      <div class="flex justify-between text-xs text-gray-500 mt-2">
        <span>IN: {formatBytes($mcpStore.systemMetrics?.resources.network.in || 0)}</span>
        <span>OUT: {formatBytes($mcpStore.systemMetrics?.resources.network.out || 0)}</span>
      </div>
    </div>
  </div>
  
  <!-- MCP Status -->
  <div class="mcp-status bg-gray-900 border border-gray-700 rounded p-4">    <h3 class="text-sm font-bold text-green-400 mb-3">MCP STATUS</h3>
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
      <div>
        <span class="text-gray-500">Connection:</span>
        <span class="ml-2 {$mcpStore.systemMetrics?.mcp.connected ? 'text-green-400' : 'text-red-400'}">
          {$mcpStore.systemMetrics?.mcp.connected ? 'CONNECTED' : 'DISCONNECTED'}
        </span>
      </div>
      <div>
        <span class="text-gray-500">Active:</span>
        <span class="text-cyan-400 ml-2">{$mcpStore.systemMetrics?.mcp.activeAgents || 0} agents</span>
      </div>
      <div>
        <span class="text-gray-500">Queued:</span>
        <span class="text-yellow-400 ml-2">{$mcpStore.systemMetrics?.mcp.queuedTasks || 0} tasks</span>
      </div>
      <div>
        <span class="text-gray-500">Completed:</span>
        <span class="text-green-400 ml-2">{$mcpStore.systemMetrics?.mcp.completedTasks || 0} tasks</span>
      </div>
    </div>
  </div>
  
  <!-- Security Alerts -->
  {#if $mcpStore.systemMetrics?.security.activeAlerts.length > 0}
    <div class="security-alerts bg-gray-900 border border-red-500 rounded p-4">
      <h3 class="text-sm font-bold text-red-400 mb-3">SECURITY ALERTS</h3>
      <div class="space-y-2">
        {#each $mcpStore.systemMetrics.security.activeAlerts as alert}          <div class="alert flex items-center justify-between bg-black p-2 rounded">
            <div class="flex items-center gap-2">
              <span class="alert-icon">
                {#if alert.severity === 'critical'}
                  🚨
                {:else if alert.severity === 'warning'}
                  ⚠️
                {:else}
                  ℹ️
                {/if}
              </span>
              <span class="text-sm {
                alert.severity === 'critical' ? 'text-red-400' :
                alert.severity === 'warning' ? 'text-yellow-400' :
                'text-blue-400'
              }">
                {alert.message}
              </span>
            </div>
            <button
              class="text-xs text-gray-500 hover:text-gray-300"
              on:click={() => mcpStore.acknowledgeAlert(alert.id)}
            >
              DISMISS
            </button>
          </div>
        {/each}
      </div>
      <div class="mt-2 text-xs text-gray-500">
        Threats: {$mcpStore.systemMetrics.security.threatsDetected} | 
        Blocked: {$mcpStore.systemMetrics.security.blockedRequests}
      </div>
    </div>
  {/if}
</div>