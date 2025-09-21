<!-- ML Training Job Monitor -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  const dispatch = createEventDispatcher();
  export let trainingJobs = [];
  export let models = [];
</script>

<div class="training-monitor">
  <Card variant="bordered">
    <div class="p-4">
      <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>← Back</Button>
      <h2 class="text-lg font-semibold text-dark-text-primary">Training Monitor</h2>
    </div>
  </Card>
</div>