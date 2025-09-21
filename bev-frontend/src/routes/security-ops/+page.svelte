<!-- Security Operations Center Interface -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  let securityAlerts = writable([]);
  let intrusionEvents = writable([]);
  let chaosTests = writable([]);
  let securityScore = 94.7;
</script>

<div class="security-ops min-h-screen bg-dark-bg-primary">
  <div class="border-b border-dark-border bg-dark-bg-secondary">
    <div class="container mx-auto px-6 py-4">
      <div class="flex items-center gap-4">
        <h1 class="text-2xl font-bold text-green-400">SECURITY OPERATIONS CENTER</h1>
        <Badge variant="success">SECURE</Badge>
        <div class="text-sm text-dark-text-secondary">Security Score: {securityScore}%</div>
      </div>
    </div>
  </div>

  <div class="container mx-auto px-6 py-6">
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <Card variant="bordered">
        <div class="p-6">
          <h3 class="text-lg font-semibold text-dark-text-primary mb-4">Intrusion Detection</h3>
          <div class="text-center">
            <div class="text-3xl font-bold text-green-400 mb-2">0</div>
            <div class="text-sm text-dark-text-secondary">Active Threats</div>
          </div>
        </div>
      </Card>

      <Card variant="bordered">
        <div class="p-6">
          <h3 class="text-lg font-semibold text-dark-text-primary mb-4">Chaos Engineering</h3>
          <div class="space-y-2">
            <Button variant="outline" fullWidth size="sm">Run Chaos Test</Button>
            <Button variant="outline" fullWidth size="sm">Network Partition</Button>
            <Button variant="outline" fullWidth size="sm">Service Failure</Button>
          </div>
        </div>
      </Card>

      <Card variant="bordered">
        <div class="p-6">
          <h3 class="text-lg font-semibold text-dark-text-primary mb-4">Tactical Intel</h3>
          <div class="text-center">
            <div class="text-2xl font-bold text-cyan-400 mb-2">12</div>
            <div class="text-sm text-dark-text-secondary">Intel Sources</div>
          </div>
        </div>
      </Card>
    </div>
  </div>
</div>