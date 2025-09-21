<!-- Cross-Database Synchronization Manager -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  const dispatch = createEventDispatcher();
  
  export let databases = [];
  
  let syncJobs = writable([]);
  let syncHistory = writable([]);
  let activeSyncs = 0;
  let lastSyncCheck = new Date();

  interface SyncJob {
    id: string;
    name: string;
    sourceDB: string;
    targetDB: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    recordsProcessed: number;
    totalRecords: number;
    startTime: string;
    endTime?: string;
    error?: string;
  }

  const syncTemplates = [
    { name: 'OSINT Data → Neo4j', source: 'postgresql', target: 'neo4j', description: 'Sync investigation data to graph' },
    { name: 'Cache Warm → Redis', source: 'postgresql', target: 'redis', description: 'Pre-warm cache with frequent queries' },
    { name: 'Search Index → Elasticsearch', source: 'postgresql', target: 'elasticsearch', description: 'Update search indexes' },
    { name: 'Analytics → InfluxDB', source: 'postgresql', target: 'influxdb', description: 'Push metrics for time-series analysis' },
    { name: 'Documents → MongoDB', source: 'postgresql', target: 'mongodb', description: 'Sync document metadata' }
  ];

  onMount(() => {
    loadSyncStatus();
    loadSyncHistory();
  });

  async function loadSyncStatus() {
    try {
      const status = await invoke('get_sync_status');
      syncJobs.set(status.jobs || []);
      activeSyncs = status.active || 0;
      lastSyncCheck = new Date();
    } catch (error) {
      console.error('Failed to load sync status:', error);
    }
  }

  function loadSyncHistory() {
    const saved = localStorage.getItem('bev-sync-history');
    if (saved) {
      try {
        syncHistory.set(JSON.parse(saved));
      } catch (e) {
        syncHistory.set([]);
      }
    }
  }

  async function startSyncJob(template) {
    try {
      const jobId = await invoke('start_database_sync', {
        sourceDB: template.source,
        targetDB: template.target,
        syncType: template.name,
        options: {
          batchSize: 1000,
          validateData: true,
          createBackup: true
        }
      });

      const newJob: SyncJob = {
        id: jobId,
        name: template.name,
        sourceDB: template.source,
        targetDB: template.target,
        status: 'pending',
        progress: 0,
        recordsProcessed: 0,
        totalRecords: 0,
        startTime: new Date().toISOString()
      };

      syncJobs.update(jobs => [newJob, ...jobs]);
      activeSyncs++;

    } catch (error) {
      console.error('Failed to start sync job:', error);
    }
  }

  async function cancelSyncJob(jobId: string) {
    try {
      await invoke('cancel_database_sync', { jobId });
      
      syncJobs.update(jobs => 
        jobs.map(job => 
          job.id === jobId 
            ? { ...job, status: 'failed', error: 'Cancelled by user' }
            : job
        )
      );
      
      activeSyncs--;
    } catch (error) {
      console.error('Failed to cancel sync job:', error);
    }
  }

  function getConnectedDatabases() {
    return databases.filter(db => db.status === 'connected');
  }

  function formatDuration(start: string, end?: string) {
    const startTime = new Date(start);
    const endTime = end ? new Date(end) : new Date();
    const duration = endTime.getTime() - startTime.getTime();
    
    if (duration < 1000) return `${duration}ms`;
    if (duration < 60000) return `${(duration / 1000).toFixed(1)}s`;
    return `${(duration / 60000).toFixed(1)}m`;
  }
</script>

<div class="database-sync space-y-6">
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>
            ← Back
          </Button>
          <h2 class="text-lg font-semibold text-dark-text-primary">Database Synchronization</h2>
          <Badge variant={activeSyncs > 0 ? 'warning' : 'success'}>
            {activeSyncs} ACTIVE
          </Badge>
        </div>
        
        <div class="flex items-center gap-2">
          <span class="text-sm text-dark-text-tertiary">
            Last check: {lastSyncCheck.toLocaleTimeString()}
          </span>
          <Button variant="outline" size="sm" on:click={loadSyncStatus}>
            Refresh
          </Button>
        </div>
      </div>
    </div>
  </Card>

  <!-- Sync Templates -->
  <Card variant="bordered">
    <div class="p-6">
      <h3 class="text-md font-medium text-dark-text-primary mb-4">Available Sync Operations</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        {#each syncTemplates as template}
          {@const sourceConnected = databases.find(db => db.id === template.source)?.status === 'connected'}
          {@const targetConnected = databases.find(db => db.id === template.target)?.status === 'connected'}
          {@const canSync = sourceConnected && targetConnected}
          
          <div class="sync-template p-4 bg-dark-bg-tertiary rounded border border-dark-border {
            canSync ? 'hover:border-green-500 cursor-pointer' : 'opacity-50'
          }">
            <div class="flex items-start justify-between mb-3">
              <div class="flex-1">
                <h4 class="text-sm font-medium text-dark-text-primary mb-1">{template.name}</h4>
                <p class="text-xs text-dark-text-secondary mb-2">{template.description}</p>
                <div class="flex items-center gap-2 text-xs">
                  <span class="text-dark-text-tertiary">{template.source}</span>
                  <svg class="w-3 h-3 text-dark-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                  </svg>
                  <span class="text-dark-text-tertiary">{template.target}</span>
                </div>
              </div>
              
              <div class="flex flex-col gap-1">
                <Badge variant={sourceConnected ? 'success' : 'danger'} size="xs">
                  SRC: {sourceConnected ? 'UP' : 'DOWN'}
                </Badge>
                <Badge variant={targetConnected ? 'success' : 'danger'} size="xs">
                  TGT: {targetConnected ? 'UP' : 'DOWN'}
                </Badge>
              </div>
            </div>
            
            <Button 
              variant={canSync ? 'primary' : 'outline'} 
              size="sm" 
              fullWidth
              disabled={!canSync || activeSyncs >= 3}
              on:click={() => startSyncJob(template)}
            >
              {canSync ? 'Start Sync' : 'Databases Offline'}
            </Button>
          </div>
        {/each}
      </div>
    </div>
  </Card>

  <!-- Active Sync Jobs -->
  {#if $syncJobs.length > 0}
    <Card variant="bordered">
      <div class="p-6">
        <h3 class="text-md font-medium text-dark-text-primary mb-4">Sync Jobs Status</h3>
        <div class="space-y-3">
          {#each $syncJobs as job}
            <div class="sync-job p-4 bg-dark-bg-tertiary rounded border border-dark-border">
              <div class="flex items-center justify-between mb-3">
                <div class="flex items-center gap-3">
                  <div>
                    <div class="text-sm font-medium text-dark-text-primary">{job.name}</div>
                    <div class="text-xs text-dark-text-secondary">
                      {job.sourceDB} → {job.targetDB}
                    </div>
                  </div>
                </div>
                
                <div class="flex items-center gap-2">
                  <Badge variant={
                    job.status === 'running' ? 'warning' :
                    job.status === 'completed' ? 'success' :
                    job.status === 'failed' ? 'danger' : 'info'
                  } size="sm">
                    {job.status.toUpperCase()}
                  </Badge>
                  
                  {#if job.status === 'running'}
                    <Button variant="outline" size="xs" on:click={() => cancelSyncJob(job.id)}>
                      Cancel
                    </Button>
                  {/if}
                </div>
              </div>

              {#if job.status === 'running'}
                <div class="progress mb-2">
                  <div class="flex justify-between text-xs mb-1">
                    <span class="text-dark-text-tertiary">
                      Progress: {job.recordsProcessed}/{job.totalRecords}
                    </span>
                    <span class="text-dark-text-secondary">{job.progress}%</span>
                  </div>
                  <div class="w-full bg-dark-bg-primary rounded-full h-2">
                    <div 
                      class="bg-green-600 h-2 rounded-full transition-all duration-300"
                      style="width: {job.progress}%"
                    ></div>
                  </div>
                </div>
              {/if}

              <div class="job-details flex items-center justify-between text-xs text-dark-text-tertiary">
                <span>Started: {new Date(job.startTime).toLocaleTimeString()}</span>
                {#if job.endTime}
                  <span>Duration: {formatDuration(job.startTime, job.endTime)}</span>
                {:else if job.status === 'running'}
                  <span>Running: {formatDuration(job.startTime)}</span>
                {/if}
              </div>

              {#if job.error}
                <div class="error-message mt-2 p-2 bg-red-600/20 border border-red-500/30 rounded">
                  <div class="text-xs text-red-400">{job.error}</div>
                </div>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    </Card>
  {/if}
</div>