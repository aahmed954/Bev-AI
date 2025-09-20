<!-- Infrastructure Management Platform -->
<script lang="ts">
  import { onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  let torStatus = writable({
    connected: true,
    exitIP: '185.220.101.182',
    circuitId: 'circuit_12345',
    relayCount: 3,
    bandwidth: '125 MB/s'
  });
  
  let proxyHealth = writable({
    haproxy: 'online',
    nginx: 'online', 
    tor: 'online',
    activeConnections: 247
  });

  let serviceMesh = writable([
    { name: 'PostgreSQL', status: 'healthy', connections: 15, latency: 2.3 },
    { name: 'Neo4j', status: 'healthy', connections: 8, latency: 12.1 },
    { name: 'Redis', status: 'healthy', connections: 42, latency: 0.8 },
    { name: 'IntelOwl', status: 'healthy', connections: 23, latency: 45.2 }
  ]);

  async function rotateCircuit() {
    try {
      const result = await invoke('rotate_circuit');
      torStatus.update(status => ({
        ...status,
        circuitId: result.circuitId,
        exitIP: result.exitIP
      }));
    } catch (error) {
      console.error('Failed to rotate circuit:', error);
    }
  }

  async function restartProxy(proxyName) {
    try {
      await invoke('restart_proxy', { proxyName });
      proxyHealth.update(health => ({
        ...health,
        [proxyName]: 'restarting'
      }));
      
      setTimeout(() => {
        proxyHealth.update(health => ({
          ...health,
          [proxyName]: 'online'
        }));
      }, 5000);
    } catch (error) {
      console.error('Failed to restart proxy:', error);
    }
  }
</script>

<div class="infrastructure-platform min-h-screen bg-dark-bg-primary">
  <div class="border-b border-dark-border bg-dark-bg-secondary">
    <div class="container mx-auto px-6 py-4">
      <div class="flex items-center gap-4">
        <h1 class="text-2xl font-bold text-green-400">INFRASTRUCTURE MANAGEMENT</h1>
        <Badge variant="success">OPERATIONAL</Badge>
      </div>
    </div>
  </div>

  <div class="container mx-auto px-6 py-6">
    <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
      <!-- Tor Network Status -->
      <Card variant="bordered">
        <div class="p-6">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-lg font-semibold text-dark-text-primary">Tor Network</h3>
            <Badge variant={$torStatus.connected ? 'success' : 'danger'}>
              {$torStatus.connected ? 'CONNECTED' : 'DISCONNECTED'}
            </Badge>
          </div>
          
          <div class="space-y-3">
            <div class="flex justify-between text-sm">
              <span class="text-dark-text-tertiary">Exit IP:</span>
              <span class="text-green-400 font-mono">{$torStatus.exitIP}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-dark-text-tertiary">Circuit:</span>
              <span class="text-dark-text-secondary font-mono">{$torStatus.circuitId}</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-dark-text-tertiary">Relays:</span>
              <span class="text-dark-text-secondary">{$torStatus.relayCount} hops</span>
            </div>
            <div class="flex justify-between text-sm">
              <span class="text-dark-text-tertiary">Bandwidth:</span>
              <span class="text-dark-text-secondary">{$torStatus.bandwidth}</span>
            </div>
          </div>
          
          <div class="mt-4 space-y-2">
            <Button variant="primary" fullWidth on:click={rotateCircuit}>
              🔄 New Circuit
            </Button>
            <Button variant="outline" fullWidth on:click={() => invoke('check_tor_status')}>
              🔍 Check Status
            </Button>
          </div>
        </div>
      </Card>

      <!-- Proxy Infrastructure -->
      <Card variant="bordered">
        <div class="p-6">
          <h3 class="text-lg font-semibold text-dark-text-primary mb-4">Proxy Infrastructure</h3>
          
          <div class="space-y-3">
            {#each Object.entries($proxyHealth) as [proxy, status]}
              {#if proxy !== 'activeConnections'}
                <div class="proxy-item flex items-center justify-between p-3 bg-dark-bg-tertiary rounded">
                  <div class="flex items-center gap-2">
                    <Badge variant={status === 'online' ? 'success' : 'danger'} size="sm">
                      {status.toUpperCase()}
                    </Badge>
                    <span class="text-sm text-dark-text-primary">{proxy.toUpperCase()}</span>
                  </div>
                  <Button variant="outline" size="xs" on:click={() => restartProxy(proxy)}>
                    Restart
                  </Button>
                </div>
              {/if}
            {/each}
          </div>
          
          <div class="mt-4 pt-4 border-t border-dark-border">
            <div class="text-center">
              <div class="text-2xl font-bold text-cyan-400">{$proxyHealth.activeConnections}</div>
              <div class="text-sm text-dark-text-secondary">Active Connections</div>
            </div>
          </div>
        </div>
      </Card>

      <!-- Service Mesh -->
      <Card variant="bordered">
        <div class="p-6">
          <h3 class="text-lg font-semibold text-dark-text-primary mb-4">Service Mesh</h3>
          
          <div class="space-y-2">
            {#each $serviceMesh as service}
              <div class="service-item p-2 bg-dark-bg-tertiary rounded border border-dark-border">
                <div class="flex items-center justify-between">
                  <div>
                    <div class="text-sm text-dark-text-primary">{service.name}</div>
                    <div class="text-xs text-dark-text-secondary">
                      {service.connections} connections • {service.latency.toFixed(1)}ms
                    </div>
                  </div>
                  <Badge variant={service.status === 'healthy' ? 'success' : 'danger'} size="xs">
                    {service.status.toUpperCase()}
                  </Badge>
                </div>
              </div>
            {/each}
          </div>
        </div>
      </Card>
    </div>
  </div>
</div>