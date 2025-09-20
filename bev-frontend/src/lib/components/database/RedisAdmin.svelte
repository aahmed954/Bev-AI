<!-- Redis Cache Administration Interface -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  const dispatch = createEventDispatcher();
  export let database;
  
  let redisCommand = '';
  let commandResults = [];
  let isExecuting = false;
  let keys = [];
  let keyPattern = '*';
  let selectedKey = '';
  let keyValue = '';
  let keyTTL = -1;
  
  const commonCommands = [
    'INFO',
    'PING',
    'KEYS *',
    'FLUSHALL',
    'MEMORY USAGE',
    'CONFIG GET *',
    'CLIENT LIST',
    'MONITOR'
  ];

  async function executeCommand() {
    if (!redisCommand.trim() || isExecuting) return;
    
    isExecuting = true;
    
    try {
      const result = await invoke('execute_database_query', {
        databaseId: 'redis',
        query: redisCommand
      });

      const commandResult = {
        id: `redis_${Date.now()}`,
        command: redisCommand,
        result: result.results,
        timestamp: new Date().toISOString(),
        executionTime: result.executionTime || 0
      };

      commandResults = [commandResult, ...commandResults.slice(0, 9)];
      dispatch('queryExecuted', commandResult);

    } catch (error) {
      console.error('Redis command failed:', error);
      const errorResult = {
        id: `redis_${Date.now()}`,
        command: redisCommand,
        result: null,
        error: error.message,
        timestamp: new Date().toISOString(),
        executionTime: 0
      };
      commandResults = [errorResult, ...commandResults.slice(0, 9)];
    } finally {
      isExecuting = false;
    }
  }

  async function searchKeys() {
    try {
      const result = await invoke('execute_database_query', {
        databaseId: 'redis',
        query: `KEYS ${keyPattern}`
      });
      keys = result.results || [];
    } catch (error) {
      console.error('Failed to search keys:', error);
    }
  }

  async function getKeyValue(key) {
    try {
      selectedKey = key;
      const valueResult = await invoke('execute_database_query', {
        databaseId: 'redis',
        query: `GET ${key}`
      });
      keyValue = valueResult.results?.[0] || '';

      const ttlResult = await invoke('execute_database_query', {
        databaseId: 'redis',
        query: `TTL ${key}`
      });
      keyTTL = ttlResult.results?.[0] || -1;
    } catch (error) {
      console.error('Failed to get key value:', error);
    }
  }

  async function deleteKey(key) {
    if (!confirm(`Delete key "${key}"?`)) return;
    
    try {
      await invoke('execute_database_query', {
        databaseId: 'redis',
        query: `DEL ${key}`
      });
      keys = keys.filter(k => k !== key);
      if (selectedKey === key) {
        selectedKey = '';
        keyValue = '';
      }
    } catch (error) {
      console.error('Failed to delete key:', error);
    }
  }
</script>

<div class="redis-admin space-y-6">
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center gap-3 mb-4">
        <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>← Back</Button>
        <span class="text-2xl">⚡</span>
        <div>
          <h2 class="text-lg font-semibold text-dark-text-primary">Redis Admin</h2>
          <div class="text-sm text-dark-text-secondary">{database?.host}:{database?.port}</div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Command Interface -->
        <div>
          <div class="flex gap-2 mb-3">
            <input
              bind:value={redisCommand}
              placeholder="Redis command (e.g., GET mykey)"
              class="flex-1 px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary placeholder-dark-text-tertiary font-mono text-sm focus:border-green-500 focus:outline-none"
              disabled={isExecuting}
              on:keypress={(e) => e.key === 'Enter' && executeCommand()}
            />
            <Button variant="primary" on:click={executeCommand} disabled={isExecuting}>
              Execute
            </Button>
          </div>

          <div class="common-commands mb-4">
            <div class="text-xs text-dark-text-tertiary mb-2">Common Commands:</div>
            <div class="flex flex-wrap gap-1">
              {#each commonCommands as cmd}
                <button 
                  class="px-2 py-1 text-xs bg-dark-bg-primary border border-dark-border rounded hover:border-green-500 text-dark-text-secondary hover:text-dark-text-primary transition-colors"
                  on:click={() => redisCommand = cmd}
                >
                  {cmd}
                </button>
              {/each}
            </div>
          </div>
        </div>

        <!-- Key Browser -->
        <div>
          <div class="flex gap-2 mb-3">
            <input
              bind:value={keyPattern}
              placeholder="Key pattern (e.g., user:*)"
              class="flex-1 px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary placeholder-dark-text-tertiary font-mono text-sm focus:border-green-500 focus:outline-none"
            />
            <Button variant="outline" on:click={searchKeys}>Search</Button>
          </div>

          <div class="keys-list space-y-1 max-h-64 overflow-y-auto">
            {#each keys.slice(0, 100) as key}
              <div class="key-item p-2 bg-dark-bg-primary rounded border border-dark-border flex items-center justify-between">
                <button 
                  class="flex-1 text-left text-xs text-dark-text-primary hover:text-green-400 transition-colors truncate"
                  on:click={() => getKeyValue(key)}
                >
                  {key}
                </button>
                <button 
                  class="text-xs text-red-400 hover:text-red-300 ml-2"
                  on:click={() => deleteKey(key)}
                >
                  DEL
                </button>
              </div>
            {/each}
          </div>
        </div>
      </div>
    </div>
  </Card>

  <!-- Command Results -->
  {#if commandResults.length > 0}
    <Card variant="bordered">
      <div class="p-4">
        <h3 class="text-md font-medium text-dark-text-primary mb-3">Command Results</h3>
        {#each commandResults as result}
          <div class="result p-3 bg-dark-bg-tertiary rounded border border-dark-border mb-3">
            <div class="flex justify-between text-xs text-dark-text-tertiary mb-2">
              <span>{result.command}</span>
              <span>{new Date(result.timestamp).toLocaleTimeString()}</span>
            </div>
            <pre class="text-xs text-dark-text-primary font-mono overflow-x-auto">{JSON.stringify(result.result, null, 2)}</pre>
          </div>
        {/each}
      </div>
    </Card>
  {/if}
</div>