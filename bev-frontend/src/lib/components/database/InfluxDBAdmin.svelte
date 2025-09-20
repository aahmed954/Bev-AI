<!-- InfluxDB Time Series Administration -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  
  const dispatch = createEventDispatcher();
  export let database;
  
  let fluxQuery = '';
  let isExecuting = false;
</script>

<div class="influxdb-admin">
  <Card variant="bordered">
    <div class="p-4">
      <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>← Back</Button>
      <span class="text-2xl">📈</span>
      <h2 class="text-lg font-semibold text-dark-text-primary">InfluxDB Admin</h2>
      <textarea bind:value={fluxQuery} class="w-full h-32 bg-dark-bg-primary border border-dark-border rounded text-dark-text-primary font-mono"></textarea>
      <Button variant="primary" on:click={() => {}} disabled={isExecuting}>Execute</Button>
    </div>
  </Card>
</div>