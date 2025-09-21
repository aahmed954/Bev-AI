<!-- ML Model Management Interface -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  const dispatch = createEventDispatcher();
  export let models = [];
  
  let searchQuery = '';
  let filterStatus = 'all';
  let sortBy = 'accuracy';
  
  function getStatusBadge(status) {
    const variants = {
      deployed: 'success',
      training: 'warning',
      testing: 'info',
      archived: 'secondary'
    };
    return variants[status] || 'info';
  }

  function deployModel(model) {
    dispatch('modelAction', { action: 'deploy', model });
  }

  function archiveModel(model) {
    if (confirm(`Archive model ${model.name}?`)) {
      dispatch('modelAction', { action: 'archive', model });
    }
  }

  function retrainModel(model) {
    dispatch('modelAction', { action: 'retrain', model });
  }

  $: filteredModels = models
    .filter(model => {
      const matchesSearch = model.name.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesStatus = filterStatus === 'all' || model.status === filterStatus;
      return matchesSearch && matchesStatus;
    })
    .sort((a, b) => {
      if (sortBy === 'accuracy') return b.accuracy - a.accuracy;
      if (sortBy === 'created') return new Date(b.created).getTime() - new Date(a.created).getTime();
      if (sortBy === 'size') return b.size - a.size;
      return a.name.localeCompare(b.name);
    });
</script>

<div class="model-manager space-y-6">
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>← Back</Button>
          <h2 class="text-lg font-semibold text-dark-text-primary">Model Management</h2>
        </div>
        
        <div class="flex items-center gap-3">
          <input
            bind:value={searchQuery}
            placeholder="Search models..."
            class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-sm"
          />
          <select bind:value={filterStatus} class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-sm">
            <option value="all">All Status</option>
            <option value="deployed">Deployed</option>
            <option value="training">Training</option>
            <option value="testing">Testing</option>
            <option value="archived">Archived</option>
          </select>
        </div>
      </div>
    </div>
  </Card>

  <div class="models-grid grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
    {#each filteredModels as model}
      <Card variant="bordered" class="model-card">
        <div class="p-6">
          <div class="flex items-start justify-between mb-4">
            <div>
              <h3 class="text-md font-semibold text-dark-text-primary mb-1">{model.name}</h3>
              <div class="text-sm text-dark-text-secondary">{model.version} • {model.framework}</div>
            </div>
            <Badge variant={getStatusBadge(model.status)}>{model.status.toUpperCase()}</Badge>
          </div>

          <div class="model-metrics grid grid-cols-2 gap-3 mb-4">
            <div class="metric text-center">
              <div class="text-xs text-dark-text-tertiary">Accuracy</div>
              <div class="text-lg font-bold text-green-400">{model.accuracy.toFixed(1)}%</div>
            </div>
            <div class="metric text-center">
              <div class="text-xs text-dark-text-tertiary">Size</div>
              <div class="text-lg font-bold text-cyan-400">{(model.size / 1024).toFixed(1)}GB</div>
            </div>
            <div class="metric text-center">
              <div class="text-xs text-dark-text-tertiary">Deployments</div>
              <div class="text-sm font-medium text-purple-400">{model.deployments}</div>
            </div>
            <div class="metric text-center">
              <div class="text-xs text-dark-text-tertiary">Last Trained</div>
              <div class="text-xs text-dark-text-secondary">{new Date(model.lastTrained).toLocaleDateString()}</div>
            </div>
          </div>

          <div class="model-actions flex gap-2">
            {#if model.status === 'testing'}
              <Button variant="primary" size="sm" fullWidth on:click={() => deployModel(model)}>Deploy</Button>
            {:else if model.status === 'deployed'}
              <Button variant="outline" size="sm" fullWidth on:click={() => retrainModel(model)}>Retrain</Button>
            {:else if model.status === 'archived'}
              <Button variant="outline" size="sm" fullWidth on:click={() => deployModel(model)}>Restore</Button>
            {/if}
            
            {#if model.status !== 'archived'}
              <Button variant="outline" size="sm" on:click={() => archiveModel(model)}>Archive</Button>
            {/if}
          </div>
        </div>
      </Card>
    {/each}
  </div>
</div>