<!-- System Performance & Monitoring Platform -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import * as echarts from 'echarts';
  import { invoke } from '@tauri-apps/api/core';
  
  let systemMetrics = writable({
    services: [],
    overallHealth: 'healthy',
    totalServices: 70,
    onlineServices: 68,
    cpuUsage: 34.7,
    memoryUsage: 42.1,
    diskUsage: 23.8,
    networkThroughput: 128.4,
    activeConnections: 1247,
    responseTime: 45.2
  });

  let performanceChart: HTMLElement;
  let servicesChart: HTMLElement;
  let chartInstances = { performance: null, services: null };
  let websocket: WebSocket | null = null;
  let connectionStatus = 'disconnected';
  let selectedTimeRange = '1h';
  let autoRefresh = true;

  onMount(() => {
    loadSystemMetrics();
    initializeCharts();
    connectWebSocket();
  });

  onDestroy(() => {
    if (websocket) websocket.close();
    Object.values(chartInstances).forEach(chart => chart?.dispose());
  });

  async function loadSystemMetrics() {
    try {
      const metrics = await invoke('get_system_performance');
      systemMetrics.set(metrics);
      updateCharts(metrics);
    } catch (error) {
      console.error('Failed to load metrics:', error);
    }
  }

  function connectWebSocket() {
    websocket = new WebSocket('ws://localhost:9090/metrics-stream');
    websocket.onopen = () => { connectionStatus = 'connected'; };
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      systemMetrics.set(data);
      updateCharts(data);
    };
    websocket.onclose = () => { 
      connectionStatus = 'disconnected';
      setTimeout(connectWebSocket, 3000);
    };
  }

  function initializeCharts() {
    if (performanceChart) {
      chartInstances.performance = echarts.init(performanceChart, 'dark');
      chartInstances.performance.setOption({
        title: { text: 'System Performance', textStyle: { color: '#00ff41' }},
        tooltip: { trigger: 'axis' },
        legend: { data: ['CPU', 'Memory', 'Disk'], textStyle: { color: '#00ff4199' }},
        xAxis: { type: 'time' },
        yAxis: { type: 'value', max: 100 },
        series: [
          { name: 'CPU', type: 'line', data: [], lineStyle: { color: '#00ff41' }},
          { name: 'Memory', type: 'line', data: [], lineStyle: { color: '#00ccff' }},
          { name: 'Disk', type: 'line', data: [], lineStyle: { color: '#ff9500' }}
        ]
      });
    }
  }

  function updateCharts(metrics) {
    // Update charts with real-time data
    if (chartInstances.performance) {
      const timestamp = Date.now();
      chartInstances.performance.setOption({
        series: [
          { data: [[timestamp, metrics.cpuUsage]] },
          { data: [[timestamp, metrics.memoryUsage]] },
          { data: [[timestamp, metrics.diskUsage]] }
        ]
      });
    }
  }
</script>

<div class="performance-platform min-h-screen bg-dark-bg-primary">
  <div class="border-b border-dark-border bg-dark-bg-secondary">
    <div class="container mx-auto px-6 py-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <h1 class="text-2xl font-bold text-green-400">PERFORMANCE MONITORING</h1>
          <Badge variant={$systemMetrics.overallHealth === 'healthy' ? 'success' : 'danger'}>
            {$systemMetrics.overallHealth.toUpperCase()}
          </Badge>
        </div>
        <div class="flex items-center gap-3">
          <Badge variant={connectionStatus === 'connected' ? 'success' : 'danger'}>
            {connectionStatus.toUpperCase()}
          </Badge>
        </div>
      </div>
    </div>
  </div>

  <div class="bg-dark-bg-secondary border-b border-dark-border">
    <div class="container mx-auto px-6 py-3">
      <div class="grid grid-cols-3 md:grid-cols-6 gap-4 text-center">
        <div>
          <div class="text-xs text-dark-text-tertiary">SERVICES</div>
          <div class="text-lg font-bold text-green-400">{$systemMetrics.onlineServices}/{$systemMetrics.totalServices}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">CPU</div>
          <div class="text-lg font-bold text-cyan-400">{$systemMetrics.cpuUsage.toFixed(1)}%</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">MEMORY</div>
          <div class="text-lg font-bold text-purple-400">{$systemMetrics.memoryUsage.toFixed(1)}%</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">DISK</div>
          <div class="text-lg font-bold text-yellow-400">{$systemMetrics.diskUsage.toFixed(1)}%</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">NETWORK</div>
          <div class="text-lg font-bold text-blue-400">{$systemMetrics.networkThroughput.toFixed(1)} MB/s</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">RESPONSE</div>
          <div class="text-lg font-bold text-red-400">{$systemMetrics.responseTime.toFixed(1)}ms</div>
        </div>
      </div>
    </div>
  </div>

  <div class="container mx-auto px-6 py-6">
    <Card variant="bordered">
      <div class="p-6">
        <h2 class="text-lg font-semibold text-dark-text-primary mb-4">Real-Time Performance</h2>
        <div bind:this={performanceChart} class="w-full h-80"></div>
      </div>
    </Card>
  </div>
</div>