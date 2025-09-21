<!-- Genetic Prompt Optimizer Interface -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  
  const dispatch = createEventDispatcher();
</script>

<div class="genetic-optimizer">
  <Card variant="bordered">
    <div class="p-4">
      <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>← Back</Button>
      <h2 class="text-lg font-semibold text-dark-text-primary">Genetic Optimizer</h2>
    </div>
  </Card>
</div>