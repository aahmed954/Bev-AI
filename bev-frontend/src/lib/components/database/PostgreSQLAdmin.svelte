<!-- PostgreSQL Administration Interface -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  const dispatch = createEventDispatcher();
  
  export let database;
  
  let sqlQuery = '';
  let queryResults = [];
  let queryHistory = [];
  let isExecuting = false;
  let selectedTable = '';
  let tables = [];
  let schemas = [];
  let currentSchema = 'public';
  
  const sampleQueries = [
    'SELECT * FROM investigations LIMIT 10;',
    'SELECT COUNT(*) FROM breach_data;',
    'SHOW TABLES;',
    'SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\';',
    'SELECT * FROM pg_stat_activity;',
    'VACUUM ANALYZE;'
  ];

  onMount(() => {
    loadTables();
    loadSchemas();
    loadQueryHistory();
  });

  async function loadTables() {
    try {
      const result = await invoke('execute_database_query', {
        databaseId: 'postgresql',
        query: `SELECT table_name FROM information_schema.tables WHERE table_schema = '${currentSchema}' ORDER BY table_name;`
      });
      tables = result.results?.map(row => row.table_name) || [];
    } catch (error) {
      console.error('Failed to load tables:', error);
    }
  }

  async function loadSchemas() {
    try {
      const result = await invoke('execute_database_query', {
        databaseId: 'postgresql',
        query: 'SELECT schema_name FROM information_schema.schemata ORDER BY schema_name;'
      });
      schemas = result.results?.map(row => row.schema_name) || ['public'];
    } catch (error) {
      console.error('Failed to load schemas:', error);
      schemas = ['public'];
    }
  }

  function loadQueryHistory() {
    const saved = localStorage.getItem('bev-postgresql-history');
    if (saved) {
      try {
        queryHistory = JSON.parse(saved);
      } catch (e) {
        queryHistory = [];
      }
    }
  }

  function saveQueryHistory() {
    localStorage.setItem('bev-postgresql-history', JSON.stringify(queryHistory.slice(0, 50)));
  }

  async function executeQuery() {
    if (!sqlQuery.trim() || isExecuting) return;
    
    isExecuting = true;
    
    const historyItem = {
      query: sqlQuery,
      timestamp: new Date().toISOString(),
      database: 'postgresql'
    };

    try {
      const result = await invoke('execute_database_query', {
        databaseId: 'postgresql',
        query: sqlQuery
      });

      const queryResult = {
        id: `query_${Date.now()}`,
        database: 'postgresql',
        query: sqlQuery,
        results: result.results || [],
        rowCount: result.rowCount || 0,
        executionTime: result.executionTime || 0,
        timestamp: new Date().toISOString()
      };

      queryResults = [queryResult, ...queryResults.slice(0, 9)];
      queryHistory = [historyItem, ...queryHistory.filter(h => h.query !== sqlQuery)].slice(0, 50);
      
      dispatch('queryExecuted', queryResult);
      saveQueryHistory();

    } catch (error) {
      console.error('Query execution failed:', error);
      const errorResult = {
        id: `query_${Date.now()}`,
        database: 'postgresql',
        query: sqlQuery,
        results: [],
        rowCount: 0,
        executionTime: 0,
        timestamp: new Date().toISOString(),
        error: error.message || 'Query execution failed'
      };
      
      queryResults = [errorResult, ...queryResults.slice(0, 9)];
    } finally {
      isExecuting = false;
    }
  }

  function loadSampleQuery(query) {
    sqlQuery = query;
  }

  function loadHistoryQuery(query) {
    sqlQuery = query.query;
  }

  function exploreTable(tableName) {
    sqlQuery = `SELECT * FROM ${tableName} LIMIT 100;`;
    selectedTable = tableName;
  }

  function analyzeTable(tableName) {
    sqlQuery = `
SELECT 
  column_name,
  data_type,
  is_nullable,
  column_default
FROM information_schema.columns 
WHERE table_name = '${tableName}' 
  AND table_schema = '${currentSchema}'
ORDER BY ordinal_position;`;
  }

  async function exportResults(result) {
    const exportData = {
      query: result.query,
      database: result.database,
      timestamp: result.timestamp,
      rowCount: result.rowCount,
      executionTime: result.executionTime,
      results: result.results
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `postgresql-query-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="postgresql-admin space-y-6">
  <!-- Header -->
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>
            ← Back
          </Button>
          <span class="text-2xl">🐘</span>
          <div>
            <h2 class="text-lg font-semibold text-dark-text-primary">PostgreSQL Admin</h2>
            <div class="text-sm text-dark-text-secondary">
              {database?.host}:{database?.port} • {database?.version}
            </div>
          </div>
          <Badge variant={getStatusBadgeVariant(database?.status)}>
            {database?.status?.toUpperCase()}
          </Badge>
        </div>
        
        <div class="flex items-center gap-2">
          <select 
            bind:value={currentSchema}
            on:change={loadTables}
            class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-sm focus:border-green-500 focus:outline-none"
          >
            {#each schemas as schema}
              <option value={schema}>{schema}</option>
            {/each}
          </select>
        </div>
      </div>
    </div>
  </Card>

  <div class="grid grid-cols-1 xl:grid-cols-4 gap-6">
    <!-- Query Interface -->
    <div class="xl:col-span-3 space-y-4">
      <!-- SQL Editor -->
      <Card variant="bordered">
        <div class="p-4">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-md font-medium text-dark-text-primary">SQL Query Editor</h3>
            <div class="flex gap-2">
              <Button 
                variant="primary" 
                on:click={executeQuery}
                disabled={isExecuting || !sqlQuery.trim()}
              >
                {#if isExecuting}
                  <div class="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin mr-2"></div>
                  Executing...
                {:else}
                  <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Execute
                {/if}
              </Button>
              <Button variant="outline" on:click={() => sqlQuery = ''}>
                Clear
              </Button>
            </div>
          </div>

          <textarea
            bind:value={sqlQuery}
            placeholder="Enter your SQL query here..."
            class="w-full h-32 px-3 py-2 bg-dark-bg-primary border border-dark-border rounded text-dark-text-primary placeholder-dark-text-tertiary font-mono text-sm focus:border-green-500 focus:outline-none resize-none"
            disabled={isExecuting}
          ></textarea>
        </div>
      </Card>

      <!-- Query Results -->
      {#if queryResults.length > 0}
        <Card variant="bordered">
          <div class="p-4">
            <h3 class="text-md font-medium text-dark-text-primary mb-3">Query Results</h3>
            
            {#each queryResults as result}
              <div class="result-section mb-6 p-4 bg-dark-bg-tertiary rounded border border-dark-border">
                <div class="flex items-center justify-between mb-3">
                  <div class="flex items-center gap-2">
                    <Badge variant={result.error ? 'danger' : 'success'} size="sm">
                      {result.error ? 'ERROR' : 'SUCCESS'}
                    </Badge>
                    <span class="text-xs text-dark-text-tertiary">
                      {result.rowCount} rows • {result.executionTime}ms • {new Date(result.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <Button variant="outline" size="xs" on:click={() => exportResults(result)}>
                    Export
                  </Button>
                </div>

                <div class="query-text mb-3 p-2 bg-dark-bg-primary rounded border border-dark-border">
                  <pre class="text-xs text-dark-text-secondary font-mono whitespace-pre-wrap">{result.query}</pre>
                </div>

                {#if result.error}
                  <div class="error-message p-3 bg-red-600/20 border border-red-500/30 rounded">
                    <div class="text-sm text-red-400">{result.error}</div>
                  </div>
                {:else if result.results.length > 0}
                  <div class="results-table overflow-x-auto">
                    <table class="w-full text-xs">
                      <thead>
                        <tr class="border-b border-dark-border">
                          {#each Object.keys(result.results[0]) as column}
                            <th class="text-left p-2 text-dark-text-tertiary font-medium">
                              {column}
                            </th>
                          {/each}
                        </tr>
                      </thead>
                      <tbody>
                        {#each result.results.slice(0, 20) as row}
                          <tr class="border-b border-dark-border/50 hover:bg-dark-bg-primary">
                            {#each Object.values(row) as value}
                              <td class="p-2 text-dark-text-secondary">
                                {value !== null ? String(value).slice(0, 50) : 'NULL'}
                              </td>
                            {/each}
                          </tr>
                        {/each}
                      </tbody>
                    </table>
                    {#if result.results.length > 20}
                      <div class="text-xs text-dark-text-tertiary p-2 bg-dark-bg-primary">
                        Showing 20 of {result.results.length} rows
                      </div>
                    {/if}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </Card>
      {/if}
    </div>

    <!-- Sidebar -->
    <div class="space-y-4">
      <!-- Sample Queries -->
      <Card variant="bordered">
        <div class="p-4">
          <h4 class="text-sm font-medium text-dark-text-primary mb-3">Sample Queries</h4>
          <div class="space-y-1">
            {#each sampleQueries as sample}
              <button 
                class="w-full text-left p-2 text-xs bg-dark-bg-tertiary rounded border border-dark-border hover:border-green-500 text-dark-text-secondary hover:text-dark-text-primary transition-colors"
                on:click={() => loadSampleQuery(sample)}
              >
                {sample.slice(0, 40)}...
              </button>
            {/each}
          </div>
        </div>
      </Card>

      <!-- Tables -->
      <Card variant="bordered">
        <div class="p-4">
          <h4 class="text-sm font-medium text-dark-text-primary mb-3">Tables ({tables.length})</h4>
          <div class="space-y-1 max-h-64 overflow-y-auto">
            {#each tables as table}
              <div class="table-item p-2 bg-dark-bg-tertiary rounded border border-dark-border">
                <div class="flex items-center justify-between">
                  <button 
                    class="text-xs text-dark-text-primary hover:text-green-400 transition-colors"
                    on:click={() => exploreTable(table)}
                  >
                    {table}
                  </button>
                  <button 
                    class="text-xs text-dark-text-tertiary hover:text-cyan-400"
                    on:click={() => analyzeTable(table)}
                    title="Analyze table structure"
                  >
                    📊
                  </button>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </Card>

      <!-- Query History -->
      {#if queryHistory.length > 0}
        <Card variant="bordered">
          <div class="p-4">
            <h4 class="text-sm font-medium text-dark-text-primary mb-3">Query History</h4>
            <div class="space-y-1 max-h-48 overflow-y-auto">
              {#each queryHistory.slice(0, 10) as query}
                <button 
                  class="w-full text-left p-2 text-xs bg-dark-bg-tertiary rounded border border-dark-border hover:border-green-500 text-dark-text-secondary hover:text-dark-text-primary transition-colors"
                  on:click={() => loadHistoryQuery(query)}
                >
                  <div class="truncate">{query.query}</div>
                  <div class="text-dark-text-tertiary">{new Date(query.timestamp).toLocaleTimeString()}</div>
                </button>
              {/each}
            </div>
          </div>
        </Card>
      {/if}
    </div>
  </div>
</div>

<style>
  .results-table {
    background: var(--dark-bg-primary, #0a0a0a);
    border-radius: 0.375rem;
    border: 1px solid var(--dark-border, #00ff4133);
  }

  .results-table table {
    border-collapse: collapse;
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar {
    width: 6px;
    height: 6px;
  }
  
  ::-webkit-scrollbar-track {
    background: var(--dark-bg-tertiary, #0f0f0f);
  }
  
  ::-webkit-scrollbar-thumb {
    background: var(--dark-border, #00ff4133);
    border-radius: 3px;
  }
</style>