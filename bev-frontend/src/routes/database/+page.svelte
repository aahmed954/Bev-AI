<!-- Multi-Database Administration Platform -->
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
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import PostgreSQLAdmin from '$lib/components/database/PostgreSQLAdmin.svelte';
  import Neo4jAdmin from '$lib/components/database/Neo4jAdmin.svelte';
  import RedisAdmin from '$lib/components/database/RedisAdmin.svelte';
  import ElasticsearchAdmin from '$lib/components/database/ElasticsearchAdmin.svelte';
  import MongoDBAdmin from '$lib/components/database/MongoDBAdmin.svelte';
  import InfluxDBAdmin from '$lib/components/database/InfluxDBAdmin.svelte';
  import DatabaseSync from '$lib/components/database/DatabaseSync.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  interface DatabaseStatus {
    id: string;
    name: string;
    type: string;
    host: string;
    port: number;
    status: 'connected' | 'disconnected' | 'error' | 'unknown';
    version: string;
    size: number;
    connections: number;
    maxConnections: number;
    uptime: number;
    lastCheck: string;
    metrics: Record<string, any>;
  }

  interface QueryResult {
    id: string;
    database: string;
    query: string;
    results: any[];
    rowCount: number;
    executionTime: number;
    timestamp: string;
    error?: string;
  }

  let databases = writable<DatabaseStatus[]>([]);
  let selectedDatabase = writable<string>('postgresql');
  let currentView: 'overview' | 'query' | 'monitor' | 'sync' = 'overview';
  let queryHistory = writable<QueryResult[]>([]);
  let connectionStatus = 'disconnected';
  let websocket: WebSocket | null = null;
  let autoRefresh = true;
  let refreshInterval = 5000; // 5 seconds

  // Database configurations
  const databaseConfigs = {
    postgresql: {
      name: 'PostgreSQL',
      icon: '🐘',
      color: '#336791',
      defaultPort: 5432,
      queryLanguage: 'SQL'
    },
    neo4j: {
      name: 'Neo4j',
      icon: '🕸️',
      color: '#00ff41',
      defaultPort: 7687,
      queryLanguage: 'Cypher'
    },
    redis: {
      name: 'Redis',
      icon: '⚡',
      color: '#dc382d',
      defaultPort: 6379,
      queryLanguage: 'Redis CLI'
    },
    elasticsearch: {
      name: 'Elasticsearch',
      icon: '🔍',
      color: '#f4d03f',
      defaultPort: 9200,
      queryLanguage: 'Query DSL'
    },
    mongodb: {
      name: 'MongoDB',
      icon: '🍃',
      color: '#47a248',
      defaultPort: 27017,
      queryLanguage: 'MongoDB Query'
    },
    influxdb: {
      name: 'InfluxDB',
      icon: '📈',
      color: '#22adf6',
      defaultPort: 8086,
      queryLanguage: 'InfluxQL/Flux'
    }
  };

  onMount(() => {
    loadDatabaseStatuses();
    connectWebSocket();
    loadQueryHistory();
    
    if (autoRefresh) {
      const interval = setInterval(loadDatabaseStatuses, refreshInterval);
      return () => clearInterval(interval);
    }
  });

  onDestroy(() => {
    if (websocket) {
      websocket.close();
    }
  });

  async function loadDatabaseStatuses() {
    try {
      const statuses = await invoke('get_database_statuses');
      databases.set(statuses);
    } catch (error) {
      console.error('Failed to load database statuses:', error);
      // Generate mock data for development
      generateMockDatabaseStatuses();
    }
  }

  function generateMockDatabaseStatuses() {
    const mockStatuses = Object.keys(databaseConfigs).map((dbType, index) => ({
      id: dbType,
      name: databaseConfigs[dbType].name,
      type: dbType,
      host: '172.21.0.' + (index + 2),
      port: databaseConfigs[dbType].defaultPort,
      status: ['connected', 'connected', 'connected', 'connected', 'connected', 'disconnected'][index],
      version: ['14.9', '5.12.0', '7.0.5', '8.5.0', '6.0.2', '2.4.1'][index],
      size: [8.2, 2.1, 0.5, 12.7, 3.4, 1.8][index], // GB
      connections: [15, 8, 12, 6, 10, 3],
      maxConnections: [100, 50, 100, 20, 100, 50][index],
      uptime: [72, 68, 71, 69, 70, 12][index], // hours
      lastCheck: new Date().toISOString(),
      metrics: {
        cpu: Math.random() * 100,
        memory: Math.random() * 100,
        disk: Math.random() * 100,
        queries_per_second: Math.random() * 1000
      }
    }));

    databases.set(mockStatuses);
  }

  function connectWebSocket() {
    try {
      connectionStatus = 'connecting';
      websocket = new WebSocket('ws://${getWebSocketHost()}:3010/database-stream');

      websocket.onopen = () => {
        connectionStatus = 'connected';
        console.log('Connected to database monitoring stream');
      };

      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleDatabaseUpdate(data);
      };

      websocket.onerror = (error) => {
        console.error('Database WebSocket error:', error);
        connectionStatus = 'disconnected';
      };

      websocket.onclose = () => {
        connectionStatus = 'disconnected';
        setTimeout(connectWebSocket, 3000);
      };
    } catch (error) {
      console.error('Failed to connect database WebSocket:', error);
      connectionStatus = 'disconnected';
    }
  }

  function handleDatabaseUpdate(data: any) {
    switch (data.type) {
      case 'status_update':
        updateDatabaseStatus(data.databaseId, data.status);
        break;
      case 'query_result':
        addQueryResult(data.result);
        break;
      case 'metrics_update':
        updateDatabaseMetrics(data.databaseId, data.metrics);
        break;
      case 'connection_change':
        updateConnectionCount(data.databaseId, data.connections);
        break;
    }
  }

  function updateDatabaseStatus(databaseId: string, status: any) {
    databases.update(dbs => 
      dbs.map(db => 
        db.id === databaseId 
          ? { ...db, ...status, lastCheck: new Date().toISOString() }
          : db
      )
    );
  }

  function updateDatabaseMetrics(databaseId: string, metrics: any) {
    databases.update(dbs => 
      dbs.map(db => 
        db.id === databaseId 
          ? { ...db, metrics: { ...db.metrics, ...metrics } }
          : db
      )
    );
  }

  function updateConnectionCount(databaseId: string, connections: number) {
    databases.update(dbs => 
      dbs.map(db => 
        db.id === databaseId 
          ? { ...db, connections }
          : db
      )
    );
  }

  function addQueryResult(result: QueryResult) {
    queryHistory.update(history => [result, ...history.slice(0, 99)]);
    saveQueryHistory();
  }

  function loadQueryHistory() {
    const saved = localStorage.getItem('bev-db-query-history');
    if (saved) {
      try {
        queryHistory.set(JSON.parse(saved));
      } catch (e) {
        console.warn('Failed to load query history:', e);
      }
    }
  }

  function saveQueryHistory() {
    queryHistory.subscribe(history => {
      localStorage.setItem('bev-db-query-history', JSON.stringify(history.slice(0, 100)));
    })();
  }

  function selectDatabase(databaseId: string) {
    selectedDatabase.set(databaseId);
    currentView = 'query';
  }

  async function testConnection(databaseId: string) {
    try {
      const result = await invoke('test_database_connection', { databaseId });
      if (result.success) {
        updateDatabaseStatus(databaseId, { status: 'connected' });
      } else {
        updateDatabaseStatus(databaseId, { status: 'error' });
      }
    } catch (error) {
      console.error('Connection test failed:', error);
      updateDatabaseStatus(databaseId, { status: 'error' });
    }
  }

  async function restartDatabase(databaseId: string) {
    if (!confirm(`Restart ${databaseConfigs[databaseId]?.name}? This will briefly interrupt service.`)) {
      return;
    }

    try {
      await invoke('restart_database', { databaseId });
      updateDatabaseStatus(databaseId, { status: 'disconnected' });
      
      // Check status after restart
      setTimeout(() => testConnection(databaseId), 5000);
    } catch (error) {
      console.error('Database restart failed:', error);
    }
  }

  function getStatusBadgeVariant(status: string) {
    switch (status) {
      case 'connected': return 'success';
      case 'disconnected': return 'warning';
      case 'error': return 'danger';
      default: return 'info';
    }
  }

  function formatUptime(hours: number) {
    if (hours < 24) return `${hours.toFixed(1)}h`;
    const days = Math.floor(hours / 24);
    const remainingHours = hours % 24;
    return `${days}d ${remainingHours.toFixed(0)}h`;
  }

  function formatSize(gb: number) {
    if (gb < 1) return `${(gb * 1024).toFixed(0)} MB`;
    if (gb < 1024) return `${gb.toFixed(1)} GB`;
    return `${(gb / 1024).toFixed(1)} TB`;
  }

  function exportDatabaseStatus() {
    const exportData = {
      timestamp: new Date().toISOString(),
      databases: $databases,
      queryHistory: $queryHistory.slice(0, 50),
      connectionStatus,
      autoRefresh,
      refreshInterval
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `database-status-export-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="database-admin min-h-screen bg-dark-bg-primary text-dark-text-primary">
  <!-- Header -->
  <div class="border-b border-dark-border bg-dark-bg-secondary">
    <div class="container mx-auto px-6 py-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <h1 class="text-2xl font-bold text-green-400">DATABASE ADMINISTRATION</h1>
          <Badge variant={connectionStatus === 'connected' ? 'success' : 'danger'}>
            MONITORING {connectionStatus.toUpperCase()}
          </Badge>
        </div>
        
        <div class="flex items-center gap-3">
          <!-- View Toggle -->
          <div class="flex bg-dark-bg-tertiary rounded-lg p-1">
            {#each ['overview', 'query', 'monitor', 'sync'] as view}
              <button
                class="px-3 py-1 text-xs font-medium rounded transition-colors {
                  currentView === view 
                    ? 'bg-green-600 text-black' 
                    : 'text-dark-text-secondary hover:text-dark-text-primary'
                }"
                on:click={() => currentView = view}
              >
                {view.toUpperCase()}
              </button>
            {/each}
          </div>
          
          <!-- Auto Refresh Toggle -->
          <label class="flex items-center gap-2 text-sm">
            <input type="checkbox" bind:checked={autoRefresh} class="checkbox" />
            <span class="text-dark-text-secondary">Auto Refresh</span>
          </label>
          
          <Button variant="outline" size="sm" on:click={exportDatabaseStatus}>
            Export Status
          </Button>
        </div>
      </div>
    </div>
  </div>

  <!-- Database Overview Grid -->
  {#if currentView === 'overview'}
    <div class="container mx-auto px-6 py-6">
      <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {#each $databases as db}
          <Card variant="bordered" class="database-card cursor-pointer hover:border-green-500 transition-all">
            <div class="p-6" on:click={() => selectDatabase(db.id)}>
              <!-- Database Header -->
              <div class="flex items-center justify-between mb-4">
                <div class="flex items-center gap-3">
                  <span class="text-2xl">{databaseConfigs[db.type]?.icon || '💾'}</span>
                  <div>
                    <h3 class="text-lg font-semibold text-dark-text-primary">{db.name}</h3>
                    <div class="text-sm text-dark-text-secondary">{db.host}:{db.port}</div>
                  </div>
                </div>
                
                <Badge variant={getStatusBadgeVariant(db.status)}>
                  {db.status.toUpperCase()}
                </Badge>
              </div>

              <!-- Database Metrics -->
              <div class="grid grid-cols-2 gap-4 mb-4">
                <div class="metric">
                  <div class="text-xs text-dark-text-tertiary">Version</div>
                  <div class="text-sm font-medium text-dark-text-primary">{db.version}</div>
                </div>
                <div class="metric">
                  <div class="text-xs text-dark-text-tertiary">Size</div>
                  <div class="text-sm font-medium text-cyan-400">{formatSize(db.size)}</div>
                </div>
                <div class="metric">
                  <div class="text-xs text-dark-text-tertiary">Connections</div>
                  <div class="text-sm font-medium text-purple-400">
                    {db.connections}/{db.maxConnections}
                  </div>
                </div>
                <div class="metric">
                  <div class="text-xs text-dark-text-tertiary">Uptime</div>
                  <div class="text-sm font-medium text-yellow-400">{formatUptime(db.uptime)}</div>
                </div>
              </div>

              <!-- Performance Metrics -->
              {#if db.metrics}
                <div class="performance-metrics mb-4">
                  <div class="text-xs text-dark-text-tertiary mb-2">Performance</div>
                  <div class="grid grid-cols-2 gap-2">
                    <div class="metric-bar">
                      <div class="flex justify-between text-xs mb-1">
                        <span class="text-dark-text-tertiary">CPU</span>
                        <span class="text-dark-text-secondary">{db.metrics.cpu?.toFixed(1) || 0}%</span>
                      </div>
                      <div class="w-full bg-dark-bg-primary rounded-full h-1.5">
                        <div 
                          class="h-1.5 rounded-full transition-all {
                            (db.metrics.cpu || 0) > 80 ? 'bg-red-500' : 
                            (db.metrics.cpu || 0) > 60 ? 'bg-yellow-500' : 'bg-green-500'
                          }"
                          style="width: {Math.min(100, db.metrics.cpu || 0)}%"
                        ></div>
                      </div>
                    </div>
                    
                    <div class="metric-bar">
                      <div class="flex justify-between text-xs mb-1">
                        <span class="text-dark-text-tertiary">Memory</span>
                        <span class="text-dark-text-secondary">{db.metrics.memory?.toFixed(1) || 0}%</span>
                      </div>
                      <div class="w-full bg-dark-bg-primary rounded-full h-1.5">
                        <div 
                          class="h-1.5 rounded-full transition-all {
                            (db.metrics.memory || 0) > 80 ? 'bg-red-500' : 
                            (db.metrics.memory || 0) > 60 ? 'bg-yellow-500' : 'bg-green-500'
                          }"
                          style="width: {Math.min(100, db.metrics.memory || 0)}%"
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              {/if}

              <!-- Quick Actions -->
              <div class="actions flex gap-2">
                <Button 
                  variant="outline" 
                  size="sm" 
                  on:click|stopPropagation={() => testConnection(db.id)}
                  disabled={db.status === 'connected'}
                >
                  <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                  </svg>
                  Test
                </Button>
                
                <Button 
                  variant="outline" 
                  size="sm"
                  on:click|stopPropagation={() => {
                    selectedDatabase.set(db.id);
                    currentView = 'monitor';
                  }}
                >
                  <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Monitor
                </Button>
                
                {#if db.status === 'error' || db.status === 'disconnected'}
                  <Button 
                    variant="outline" 
                    size="sm"
                    on:click|stopPropagation={() => restartDatabase(db.id)}
                  >
                    <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Restart
                  </Button>
                {/if}
              </div>
            </div>
          </Card>
        {/each}
      </div>

      <!-- System Overview -->
      <div class="grid grid-cols-1 lg:grid-cols-4 gap-6 mt-6">
        <Card variant="bordered">
          <div class="p-4 text-center">
            <div class="text-xs text-dark-text-tertiary mb-1">Connected Databases</div>
            <div class="text-2xl font-bold text-green-400">
              {$databases.filter(db => db.status === 'connected').length}/{$databases.length}
            </div>
          </div>
        </Card>
        
        <Card variant="bordered">
          <div class="p-4 text-center">
            <div class="text-xs text-dark-text-tertiary mb-1">Total Connections</div>
            <div class="text-2xl font-bold text-cyan-400">
              {$databases.reduce((sum, db) => sum + db.connections, 0)}
            </div>
          </div>
        </Card>
        
        <Card variant="bordered">
          <div class="p-4 text-center">
            <div class="text-xs text-dark-text-tertiary mb-1">Total Size</div>
            <div class="text-2xl font-bold text-purple-400">
              {formatSize($databases.reduce((sum, db) => sum + db.size, 0))}
            </div>
          </div>
        </Card>
        
        <Card variant="bordered">
          <div class="p-4 text-center">
            <div class="text-xs text-dark-text-tertiary mb-1">Avg Query Rate</div>
            <div class="text-2xl font-bold text-yellow-400">
              {($databases.reduce((sum, db) => sum + (db.metrics?.queries_per_second || 0), 0) / $databases.length).toFixed(0)}/s
            </div>
          </div>
        </Card>
      </div>
    </div>
  {/if}

  <!-- Database-Specific Query Interfaces -->
  {#if currentView === 'query'}
    <div class="container mx-auto px-6 py-6">
      {#if $selectedDatabase === 'postgresql'}
        <PostgreSQLAdmin 
          database={$databases.find(db => db.id === 'postgresql')}
          on:queryExecuted={(e) => addQueryResult(e.detail)}
          on:backToOverview={() => currentView = 'overview'}
        />
      {:else if $selectedDatabase === 'neo4j'}
        <Neo4jAdmin 
          database={$databases.find(db => db.id === 'neo4j')}
          on:queryExecuted={(e) => addQueryResult(e.detail)}
          on:backToOverview={() => currentView = 'overview'}
        />
      {:else if $selectedDatabase === 'redis'}
        <RedisAdmin 
          database={$databases.find(db => db.id === 'redis')}
          on:queryExecuted={(e) => addQueryResult(e.detail)}
          on:backToOverview={() => currentView = 'overview'}
        />
      {:else if $selectedDatabase === 'elasticsearch'}
        <ElasticsearchAdmin 
          database={$databases.find(db => db.id === 'elasticsearch')}
          on:queryExecuted={(e) => addQueryResult(e.detail)}
          on:backToOverview={() => currentView = 'overview'}
        />
      {:else if $selectedDatabase === 'mongodb'}
        <MongoDBAdmin 
          database={$databases.find(db => db.id === 'mongodb')}
          on:queryExecuted={(e) => addQueryResult(e.detail)}
          on:backToOverview={() => currentView = 'overview'}
        />
      {:else if $selectedDatabase === 'influxdb'}
        <InfluxDBAdmin 
          database={$databases.find(db => db.id === 'influxdb')}
          on:queryExecuted={(e) => addQueryResult(e.detail)}
          on:backToOverview={() => currentView = 'overview'}
        />
      {/if}
    </div>
  {/if}

  <!-- Database Synchronization -->
  {#if currentView === 'sync'}
    <div class="container mx-auto px-6 py-6">
      <DatabaseSync 
        databases={$databases}
        on:syncStatusChanged={(e) => handleDatabaseUpdate(e.detail)}
        on:backToOverview={() => currentView = 'overview'}
      />
    </div>
  {/if}

  <!-- Performance Monitoring -->
  {#if currentView === 'monitor'}
    <div class="container mx-auto px-6 py-6">
      {@const selectedDB = $databases.find(db => db.id === $selectedDatabase)}
      
      {#if selectedDB}
        <Card variant="bordered">
          <div class="p-6">
            <div class="flex items-center justify-between mb-6">
              <div class="flex items-center gap-3">
                <Button variant="outline" size="sm" on:click={() => currentView = 'overview'}>
                  ← Back
                </Button>
                <h2 class="text-lg font-semibold text-dark-text-primary">
                  {selectedDB.name} Performance Monitor
                </h2>
                <Badge variant={getStatusBadgeVariant(selectedDB.status)}>
                  {selectedDB.status.toUpperCase()}
                </Badge>
              </div>
              
              <div class="flex items-center gap-2">
                <select 
                  bind:value={refreshInterval}
                  class="px-2 py-1 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-xs"
                >
                  <option value={1000}>1s</option>
                  <option value={5000}>5s</option>
                  <option value={10000}>10s</option>
                  <option value={30000}>30s</option>
                </select>
                <Button variant="outline" size="sm" on:click={() => loadDatabaseStatuses()}>
                  Refresh
                </Button>
              </div>
            </div>

            <!-- Real-time Metrics -->
            <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
              <!-- CPU Usage -->
              <div class="metric-card p-4 bg-dark-bg-tertiary rounded border border-dark-border">
                <div class="flex items-center justify-between mb-2">
                  <span class="text-sm text-dark-text-tertiary">CPU Usage</span>
                  <span class="text-sm font-medium text-dark-text-primary">
                    {selectedDB.metrics?.cpu?.toFixed(1) || 0}%
                  </span>
                </div>
                <div class="w-full bg-dark-bg-primary rounded-full h-2">
                  <div 
                    class="h-2 rounded-full transition-all {
                      (selectedDB.metrics?.cpu || 0) > 80 ? 'bg-red-500' : 
                      (selectedDB.metrics?.cpu || 0) > 60 ? 'bg-yellow-500' : 'bg-green-500'
                    }"
                    style="width: {Math.min(100, selectedDB.metrics?.cpu || 0)}%"
                  ></div>
                </div>
              </div>

              <!-- Memory Usage -->
              <div class="metric-card p-4 bg-dark-bg-tertiary rounded border border-dark-border">
                <div class="flex items-center justify-between mb-2">
                  <span class="text-sm text-dark-text-tertiary">Memory Usage</span>
                  <span class="text-sm font-medium text-dark-text-primary">
                    {selectedDB.metrics?.memory?.toFixed(1) || 0}%
                  </span>
                </div>
                <div class="w-full bg-dark-bg-primary rounded-full h-2">
                  <div 
                    class="h-2 rounded-full transition-all {
                      (selectedDB.metrics?.memory || 0) > 80 ? 'bg-red-500' : 
                      (selectedDB.metrics?.memory || 0) > 60 ? 'bg-yellow-500' : 'bg-green-500'
                    }"
                    style="width: {Math.min(100, selectedDB.metrics?.memory || 0)}%"
                  ></div>
                </div>
              </div>

              <!-- Disk Usage -->
              <div class="metric-card p-4 bg-dark-bg-tertiary rounded border border-dark-border">
                <div class="flex items-center justify-between mb-2">
                  <span class="text-sm text-dark-text-tertiary">Disk Usage</span>
                  <span class="text-sm font-medium text-dark-text-primary">
                    {selectedDB.metrics?.disk?.toFixed(1) || 0}%
                  </span>
                </div>
                <div class="w-full bg-dark-bg-primary rounded-full h-2">
                  <div 
                    class="h-2 rounded-full transition-all {
                      (selectedDB.metrics?.disk || 0) > 80 ? 'bg-red-500' : 
                      (selectedDB.metrics?.disk || 0) > 60 ? 'bg-yellow-500' : 'bg-green-500'
                    }"
                    style="width: {Math.min(100, selectedDB.metrics?.disk || 0)}%"
                  ></div>
                </div>
              </div>

              <!-- Query Rate -->
              <div class="metric-card p-4 bg-dark-bg-tertiary rounded border border-dark-border">
                <div class="text-sm text-dark-text-tertiary mb-1">Query Rate</div>
                <div class="text-lg font-bold text-cyan-400">
                  {selectedDB.metrics?.queries_per_second?.toFixed(0) || 0}
                </div>
                <div class="text-xs text-dark-text-secondary">queries/sec</div>
              </div>
            </div>
          </div>
        </Card>
      {/if}
    </div>
  {/if}
</div>

<style>
  .database-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 255, 65, 0.15);
  }

  .metric {
    @apply text-center;
  }

  .metric-card {
    @apply transition-all hover:bg-dark-bg-secondary;
  }

  .checkbox {
    @apply w-4 h-4 rounded border-2 border-dark-border;
    background: var(--dark-bg-tertiary, #0f0f0f);
    accent-color: #00ff41;
  }

  /* Ensure proper dark theme */
  :global(.dark-bg-primary) { background-color: #0a0a0a; }
  :global(.dark-bg-secondary) { background-color: #1a1a1a; }
  :global(.dark-bg-tertiary) { background-color: #0f0f0f; }
  :global(.dark-text-primary) { color: #00ff41; }
  :global(.dark-text-secondary) { color: #00ff4199; }
  :global(.dark-text-tertiary) { color: #00ff4166; }
  :global(.dark-border) { border-color: #00ff4133; }
</style>