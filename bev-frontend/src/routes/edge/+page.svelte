<!-- Edge Computing Management Platform -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  let edgeNodes = writable([
    { id: 'us-east', name: 'US East', region: 'Virginia', status: 'online', load: 23.4, latency: 12.3 },
    { id: 'us-west', name: 'US West', region: 'California', status: 'online', load: 45.7, latency: 18.9 },
    { id: 'eu-central', name: 'EU Central', region: 'Frankfurt', status: 'online', load: 67.2, latency: 34.1 },
    { id: 'asia-pacific', name: 'Asia Pacific', region: 'Singapore', status: 'offline', load: 0, latency: 0 }
  ]);
</script>

<div class="edge-platform min-h-screen bg-dark-bg-primary">
  <div class="border-b border-dark-border bg-dark-bg-secondary">
    <div class="container mx-auto px-6 py-4">
      <h1 class="text-2xl font-bold text-green-400">EDGE COMPUTING NETWORK</h1>
    </div>
  </div>

  <div class="container mx-auto px-6 py-6">
    <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
      {#each $edgeNodes as node}
        <Card variant="bordered">
          <div class="p-6">
            <div class="flex items-center justify-between mb-4">
              <div>
                <h3 class="text-lg font-semibold text-dark-text-primary">{node.name}</h3>
                <div class="text-sm text-dark-text-secondary">{node.region}</div>
              </div>
              <Badge variant={node.status === 'online' ? 'success' : 'danger'}>
                {node.status.toUpperCase()}
              </Badge>
            </div>
            
            <div class="grid grid-cols-2 gap-3">
              <div class="metric text-center">
                <div class="text-xs text-dark-text-tertiary">Load</div>
                <div class="text-lg font-bold text-cyan-400">{node.load.toFixed(1)}%</div>
              </div>
              <div class="metric text-center">
                <div class="text-xs text-dark-text-tertiary">Latency</div>
                <div class="text-lg font-bold text-purple-400">{node.latency.toFixed(1)}ms</div>
              </div>
            </div>
          </div>
        </Card>
      {/each}
    </div>
  </div>
</div>