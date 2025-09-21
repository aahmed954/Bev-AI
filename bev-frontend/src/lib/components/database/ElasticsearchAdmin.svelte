<!-- Elasticsearch Administration Interface -->  
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  const dispatch = createEventDispatcher();
  export let database;
  
  let esQuery = '';
  let queryResults = [];
  let isExecuting = false;
  let indices = [];
  let selectedIndex = '';
  
  const sampleQueries = [
    '{"query": {"match_all": {}}}',
    '{"query": {"match": {"title": "cryptocurrency"}}}',
    '{"query": {"range": {"timestamp": {"gte": "now-1d"}}}}',
    '{"aggs": {"by_type": {"terms": {"field": "type.keyword"}}}}'
  ];

  async function executeQuery() {
    if (!esQuery.trim() || isExecuting) return;
    
    isExecuting = true;
    
    try {
      const result = await invoke('execute_database_query', {
        databaseId: 'elasticsearch',
        query: esQuery,
        index: selectedIndex
      });

      const queryResult = {
        id: `es_${Date.now()}`,
        database: 'elasticsearch',
        query: esQuery,
        index: selectedIndex,
        results: result.results || [],
        hits: result.hits || 0,
        executionTime: result.executionTime || 0,
        timestamp: new Date().toISOString()
      };

      queryResults = [queryResult, ...queryResults.slice(0, 9)];
      dispatch('queryExecuted', queryResult);

    } catch (error) {
      console.error('Elasticsearch query failed:', error);
    } finally {
      isExecuting = false;
    }
  }
</script>

<div class="elasticsearch-admin space-y-6">
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center gap-3 mb-4">
        <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>← Back</Button>
        <span class="text-2xl">🔍</span>
        <h2 class="text-lg font-semibold text-dark-text-primary">Elasticsearch Admin</h2>
      </div>

      <div class="space-y-4">
        <div class="flex gap-2">
          <select bind:value={selectedIndex} class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary">
            <option value="">All Indices</option>
            <option value="osint-data">OSINT Data</option>
            <option value="investigations">Investigations</option>
          </select>
        </div>

        <textarea
          bind:value={esQuery}
          placeholder='{"query": {"match_all": {}}}'
          class="w-full h-32 px-3 py-2 bg-dark-bg-primary border border-dark-border rounded text-dark-text-primary font-mono text-sm"
        ></textarea>

        <Button variant="primary" on:click={executeQuery} disabled={isExecuting}>
          Execute Query
        </Button>
      </div>
    </div>
  </Card>
</div>