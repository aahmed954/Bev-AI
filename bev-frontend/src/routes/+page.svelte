<!-- BEV OSINT Main Dashboard -->
<script lang="ts">
  import { onMount } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import Panel from '$lib/components/ui/Panel.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import { dashboardMetrics, opsecState, systemHealth } from '$lib/stores/app';
  
  // Mock data for demonstration
  onMount(() => {
    // Update metrics with mock data
    dashboardMetrics.set({
      darknetMarkets: {
        online: 12,
        offline: 3,
        suspicious: 2,
      },
      cryptoTransactions: {
        tracked: 1847,
        flagged: 23,
        total: 2103,
      },
      threats: {
        critical: 2,
        high: 8,
        medium: 27,
        low: 45,
      },
      agents: {
        active: 7,
        idle: 2,
        error: 0,
      },
    });
  });
</script>

<div class="p-6 space-y-6">
  <!-- System Status Bar -->
  <div class="flex items-center justify-between">
    <h1 class="text-2xl font-bold text-dark-text-primary">Intelligence Overview</h1>
    <div class="flex items-center gap-4">
      <Badge variant={$systemHealth === 'healthy' ? 'success' : $systemHealth === 'warning' ? 'warning' : 'danger'}>
        System: {$systemHealth.toUpperCase()}
      </Badge>
      <Button variant="primary" size="sm">
        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
        Refresh All
      </Button>
    </div>
  </div>
  
  <!-- Metrics Grid -->
  <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
    <!-- Darknet Markets -->
    <Card variant="bordered">
      <div class="flex items-center justify-between">
        <div>
          <p class="text-sm text-dark-text-tertiary">Darknet Markets</p>
          <p class="text-2xl font-bold text-dark-text-primary mt-1">
            {$dashboardMetrics.darknetMarkets.online}
          </p>
          <p class="text-xs text-dark-text-secondary mt-2">
            {$dashboardMetrics.darknetMarkets.suspicious} suspicious
          </p>
        </div>
        <div class="p-3 bg-purple-600/10 rounded-lg">
          <svg class="w-8 h-8 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M3 3h2l.4 2M7 13h10l4-8H5.4M7 13L5.4 5M7 13l-2.293 2.293c-.63.63-.184 1.707.707 1.707H17m0 0a2 2 0 100 4 2 2 0 000-4zm-8 2a2 2 0 11-4 0 2 2 0 014 0z" />
          </svg>
        </div>
      </div>
    </Card>
    
    <!-- Crypto Transactions -->
    <Card variant="bordered">
      <div class="flex items-center justify-between">
        <div>
          <p class="text-sm text-dark-text-tertiary">Crypto Tracked</p>
          <p class="text-2xl font-bold text-dark-text-primary mt-1">
            {$dashboardMetrics.cryptoTransactions.tracked}
          </p>
          <p class="text-xs text-dark-text-secondary mt-2">
            {$dashboardMetrics.cryptoTransactions.flagged} flagged
          </p>
        </div>
        <div class="p-3 bg-blue-600/10 rounded-lg">
          <svg class="w-8 h-8 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
      </div>
    </Card>
    
    <!-- Active Threats -->
    <Card variant="bordered">
      <div class="flex items-center justify-between">
        <div>
          <p class="text-sm text-dark-text-tertiary">Active Threats</p>
          <p class="text-2xl font-bold text-dark-text-primary mt-1">
            {$dashboardMetrics.threats.critical + $dashboardMetrics.threats.high}
          </p>
          <p class="text-xs text-dark-text-secondary mt-2">
            {$dashboardMetrics.threats.critical} critical
          </p>
        </div>
        <div class="p-3 bg-red-600/10 rounded-lg">
          <svg class="w-8 h-8 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
      </div>
    </Card>
    
    <!-- Active Agents -->
    <Card variant="bordered">
      <div class="flex items-center justify-between">
        <div>
          <p class="text-sm text-dark-text-tertiary">Active Agents</p>
          <p class="text-2xl font-bold text-dark-text-primary mt-1">
            {$dashboardMetrics.agents.active}
          </p>
          <p class="text-xs text-dark-text-secondary mt-2">
            {$dashboardMetrics.agents.idle} idle
          </p>
        </div>
        <div class="p-3 bg-green-600/10 rounded-lg">
          <svg class="w-8 h-8 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </div>
      </div>
    </Card>
  </div>
  
  <!-- Main Content Area -->
  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    <!-- Recent Activity Panel -->
    <div class="xl:col-span-2">
      <Panel title="Recent Intelligence Activity" subtitle="Real-time feed of OSINT operations">
        <div class="space-y-3">
          {#each Array(5) as _, i}
            <div class="flex items-start gap-3 p-3 bg-dark-bg-tertiary rounded-lg">
              <div class="w-2 h-2 bg-green-400 rounded-full mt-2 animate-pulse"></div>
              <div class="flex-1">
                <p class="text-sm font-medium text-dark-text-primary">
                  New vendor detected on AlphaBay
                </p>
                <p class="text-xs text-dark-text-tertiary mt-1">
                  Vendor "CryptoKing" listed 47 new products
                </p>
                <p class="text-xs text-dark-text-tertiary mt-1">
                  2 minutes ago
                </p>
              </div>
              <Badge variant="warning" size="xs">NEW</Badge>
            </div>
          {/each}
        </div>
      </Panel>
    </div>
    
    <!-- Quick Actions Panel -->
    <div>
      <Panel title="Quick Actions" subtitle="Common operations">
        <div class="space-y-3">
          <Button variant="outline" fullWidth>
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            New Investigation
          </Button>
          <Button variant="outline" fullWidth>
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Generate Report
          </Button>
          <Button variant="outline" fullWidth>
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Export Data
          </Button>
          <Button variant="outline" fullWidth>
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            </svg>
            Configure Alerts
          </Button>
        </div>
      </Panel>
    </div>
  </div>
</div>
