<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { io, Socket } from 'socket.io-client';
  import * as echarts from 'echarts';
  import { invoke } from '@tauri-apps/api/tauri';
  
  let socket: Socket;
  let bandwidthChart: echarts.ECharts;
  let latencyChart: echarts.ECharts;
  let circuitMap: echarts.ECharts;
  
  // Tor Circuit Management
  let torStatus = 'disconnected';
  let currentCircuit = null;
  let exitNodes = [];
  let selectedExitCountry = 'auto';
  let circuits = [];
  let activeCircuitId = null;
  
  // System Metrics
  let systemMetrics = {
    cpu: 0,
    memory: 0,
    diskIO: 0,
    networkIO: 0,
    activeConnections: 0,
    openPorts: []
  };
  
  // OPSEC Compliance
  let opsecStatus = {
    torEnabled: false,
    vpnActive: false,
    dnsLeakProtection: false,
    webRtcBlocked: false,
    fingerprintResistance: false,
    cookiesCleared: true,
    jsDisabled: false,
    score: 0
  };  
  // Bandwidth & Latency Tracking
  let bandwidthData = {
    download: [],
    upload: [],
    timestamps: []
  };
  
  let latencyData = {
    tor: [],
    direct: [],
    timestamps: []
  };
  
  // Security Audit Trail
  let auditLog = [];
  let securityAlerts = [];
  
  // Exit Node Countries
  const exitCountries = [
    { code: 'auto', name: 'Automatic' },
    { code: 'us', name: 'United States' },
    { code: 'gb', name: 'United Kingdom' },
    { code: 'de', name: 'Germany' },
    { code: 'nl', name: 'Netherlands' },
    { code: 'ch', name: 'Switzerland' },
    { code: 'se', name: 'Sweden' },
    { code: 'ro', name: 'Romania' },
    { code: 'jp', name: 'Japan' },
    { code: 'sg', name: 'Singapore' }
  ];
  
  let updateInterval;
  
  onMount(() => {
    initializeWebSocket();
    initializeCharts();
    startSystemMonitoring();
    checkOpsecCompliance();
    
    return () => {
      if (socket) socket.disconnect();
      if (updateInterval) clearInterval(updateInterval);
      if (bandwidthChart) bandwidthChart.dispose();
      if (latencyChart) latencyChart.dispose();
      if (circuitMap) circuitMap.dispose();
    };
  });  
  // Bandwidth & Latency Tracking
  let bandwidthData = {
    download: [],
    upload: [],
    timestamps: []
  };
  
  let latencyData = {
    tor: [],
    direct: [],
    timestamps: []
  };
  
  // Security Audit Trail
  let auditLog = [];
  let securityAlerts = [];
  
  // Exit Node Countries
  const exitCountries = [
    { code: 'auto', name: 'Automatic' },
    { code: 'us', name: 'United States' },
    { code: 'gb', name: 'United Kingdom' },
    { code: 'de', name: 'Germany' },
    { code: 'nl', name: 'Netherlands' },
    { code: 'ch', name: 'Switzerland' },
    { code: 'se', name: 'Sweden' },
    { code: 'ro', name: 'Romania' },
    { code: 'jp', name: 'Japan' },
    { code: 'sg', name: 'Singapore' }
  ];
  
  let updateInterval;
  
  onMount(() => {
    initializeWebSocket();
    initializeCharts();
    startSystemMonitoring();
    checkOpsecCompliance();
    
    return () => {
      if (socket) socket.disconnect();
      if (updateInterval) clearInterval(updateInterval);
      if (bandwidthChart) bandwidthChart.dispose();
      if (latencyChart) latencyChart.dispose();
      if (circuitMap) circuitMap.dispose();
    };
  });