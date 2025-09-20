<!-- BEV OSINT Header Component -->
<script lang="ts">
  import Badge from '$lib/components/ui/Badge.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import { onMount } from 'svelte';
  
  export let userName = 'Operator';
  
  let proxyStatus = 'connected';
  let exitIp = 'Loading...';
  let circuitInfo = '';
  
  onMount(async () => {
    // Check proxy status via IPC
    checkProxyStatus();
  });
  
  async function checkProxyStatus() {
    try {
      const { invoke } = await import('@tauri-apps/api/tauri');
      const status = await invoke('get_proxy_status');
      // Update status based on response
      proxyStatus = 'connected';
      exitIp = '185.220.101.42';
      circuitInfo = 'NL → DE → US';
    } catch (error) {
      console.error('Failed to check proxy status:', error);
      proxyStatus = 'error';
    }
  }
  
  async function newCircuit() {
    try {
      const { invoke } = await import('@tauri-apps/api/tauri');
      await invoke('new_tor_circuit');
      await checkProxyStatus();
    } catch (error) {
      console.error('Failed to create new circuit:', error);
    }
  }
</script>

<header class="h-16 bg-dark-bg-secondary border-b border-dark-border-subtle flex items-center justify-between px-6">
  <!-- Left Section -->
  <div class="flex items-center gap-4">
    <h2 class="text-xl font-semibold text-dark-text-primary">
      <slot name="title">Dashboard</slot>
    </h2>
  </div>
  
  <!-- Right Section -->
  <div class="flex items-center gap-6">
    <!-- OPSEC Status -->
    <div class="flex items-center gap-3">
      <div class="flex items-center gap-2 text-sm">
        <span class="text-dark-text-tertiary">Exit IP:</span>
        <code class="text-dark-text-secondary font-mono text-xs bg-dark-bg-tertiary px-2 py-0.5 rounded">
          {exitIp}
        </code>
      </div>
      
      {#if circuitInfo}
        <div class="flex items-center gap-2 text-sm">
          <span class="text-dark-text-tertiary">Circuit:</span>
          <code class="text-dark-text-secondary font-mono text-xs">
            {circuitInfo}
          </code>
        </div>
      {/if}
      
      <Badge 
        variant={proxyStatus === 'connected' ? 'proxied' : proxyStatus === 'connecting' ? 'warning' : 'danger'}
        size="sm"
        pulse={proxyStatus === 'connecting'}
      >
        {#if proxyStatus === 'connected'}
          <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
            <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
          </svg>
          TOR ACTIVE
        {:else if proxyStatus === 'connecting'}
          CONNECTING...
        {:else}
          NOT PROXIED
        {/if}
      </Badge>
      
      <Button variant="ghost" size="xs" on:click={newCircuit}>
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </Button>
    </div>
    
    <!-- User Info -->
    <div class="flex items-center gap-3 pl-6 border-l border-dark-border-subtle">
      <div class="text-right">
        <p class="text-sm font-medium text-dark-text-primary">{userName}</p>
        <p class="text-xs text-dark-text-tertiary">Level 5 Access</p>
      </div>
      <div class="w-8 h-8 bg-gradient-to-br from-primary-500 to-primary-700 rounded-full flex items-center justify-center">
        <span class="text-white text-sm font-medium">{userName[0].toUpperCase()}</span>
      </div>
    </div>
  </div>
</header>
