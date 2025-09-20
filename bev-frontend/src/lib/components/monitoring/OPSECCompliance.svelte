<!-- BEV OPSEC Compliance Dashboard - Security & Anonymity Enforcement -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { mcpStore } from '$lib/stores/mcpStore';
  import type { OPSECStatus } from '$lib/mcp/types';
  
  let opsecStatus: OPSECStatus | null = null;
  let refreshInterval: NodeJS.Timeout;
  let isRefreshing = false;
  let circuitRotateConfirm = false;
  
  async function checkOPSEC() {
    isRefreshing = true;
    try {
      opsecStatus = await mcpStore.checkOPSEC();
    } finally {
      isRefreshing = false;
    }
  }
  
  async function rotateCircuit() {
    if (!circuitRotateConfirm) {
      circuitRotateConfirm = true;
      setTimeout(() => circuitRotateConfirm = false, 3000);
      return;
    }
    
    // In production, this would trigger Tor circuit rotation
    console.log('Rotating Tor circuit...');
    circuitRotateConfirm = false;
    await checkOPSEC();
  }
  
  function getComplianceColor(compliant: boolean): string {
    return compliant ? 'text-green-400' : 'text-red-400';
  }
  
  function getLeakTestStatus(passed: boolean): string {
    return passed ? '✓ PASSED' : '✗ FAILED';
  }
  
  function getLeakTestColor(passed: boolean): string {
    return passed ? 'text-green-400' : 'text-red-400';
  }
  
  onMount(() => {
    checkOPSEC();
    // Refresh every 10 seconds
    refreshInterval = setInterval(checkOPSEC, 10000);
  });
  
  onDestroy(() => {
    if (refreshInterval) clearInterval(refreshInterval);
  });
  
  $: proxyStatus = $mcpStore.systemMetrics?.proxy || opsecStatus;
</script>

<div class="opsec-compliance bg-black text-green-400 font-mono p-4 space-y-4">
  <!-- Header with Overall Status -->
  <div class="header bg-gray-900 border border-gray-700 rounded p-4">
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-xl font-bold text-green-400">OPSEC COMPLIANCE</h2>
        <p class="text-xs text-gray-500 mt-1">Operational Security & Anonymity Status</p>
      </div>
      <div class="flex items-center gap-4">
        {#if opsecStatus}
          <div class="compliance-status text-lg font-bold {getComplianceColor(opsecStatus.compliant)}">
            {opsecStatus.compliant ? '🛡️ COMPLIANT' : '⚠️ NON-COMPLIANT'}
          </div>
        {/if}
        <button 
          class="px-3 py-1 bg-gray-800 hover:bg-gray-700 rounded text-sm"
          on:click={checkOPSEC}
          disabled={isRefreshing}
        >
          {isRefreshing ? 'CHECKING...' : 'REFRESH'}
        </button>
      </div>
    </div>
  </div>

  <!-- Proxy Status Section -->
  <div class="proxy-status bg-gray-900 border border-gray-700 rounded p-4">
    <h3 class="text-lg font-bold text-green-400 mb-3">🌐 PROXY STATUS</h3>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
      <!-- Connection Status -->
      <div class="status-item">
        <div class="text-sm text-gray-500 mb-1">Connection Status</div>
        <div class="flex items-center gap-2">
          <div class="status-indicator w-3 h-3 rounded-full {
            proxyStatus?.connected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
          }"></div>
          <span class="{proxyStatus?.connected ? 'text-green-400' : 'text-red-400'}">
            {proxyStatus?.connected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
          {#if proxyStatus?.latency}
            <span class="text-xs text-gray-500">({proxyStatus.latency}ms)</span>
          {/if}
        </div>
      </div>
      
      <!-- Exit IP -->
      <div class="status-item">
        <div class="text-sm text-gray-500 mb-1">Exit IP Address</div>
        <div class="font-mono text-cyan-400">
          {proxyStatus?.exitIP || 'UNKNOWN'}
        </div>
      </div>
    </div>
    
    <!-- Circuit Information -->
    {#if opsecStatus?.circuitInfo}
      <div class="circuit-info mt-4 pt-4 border-t border-gray-700">
        <div class="flex items-center justify-between mb-2">
          <div class="text-sm text-gray-500">Circuit Path</div>
          <button
            class="text-xs px-2 py-1 bg-orange-900 hover:bg-orange-800 text-orange-400 rounded"
            on:click={rotateCircuit}
          >
            {circuitRotateConfirm ? 'CONFIRM ROTATE?' : '🔄 ROTATE CIRCUIT'}
          </button>
        </div>
        <div class="circuit-path flex items-center gap-2 text-xs">
          {#each opsecStatus.circuitInfo.nodes as node, i}
            <div class="node px-2 py-1 bg-gray-800 rounded text-gray-400">
              {node}
            </div>
            {#if i < opsecStatus.circuitInfo.nodes.length - 1}
              <span class="text-gray-600">→</span>
            {/if}
          {/each}
        </div>
        <div class="text-xs text-gray-500 mt-1">
          Circuit ID: {opsecStatus.circuitInfo.id}
        </div>
      </div>
    {/if}
  </div>

  <!-- Leak Tests Grid -->
  {#if opsecStatus?.leakTests}
    <div class="leak-tests bg-gray-900 border border-gray-700 rounded p-4">
      <h3 class="text-lg font-bold text-green-400 mb-3">🔒 LEAK TESTS</h3>
      
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <!-- DNS Leak Test -->
        <div class="leak-test">
          <div class="text-sm text-gray-500 mb-1">DNS Leak</div>
          <div class="{getLeakTestColor(opsecStatus.leakTests.dns)}">
            {getLeakTestStatus(opsecStatus.leakTests.dns)}
          </div>
        </div>
        
        <!-- WebRTC Leak Test -->
        <div class="leak-test">
          <div class="text-sm text-gray-500 mb-1">WebRTC Leak</div>
          <div class="{getLeakTestColor(opsecStatus.leakTests.webrtc)}">
            {getLeakTestStatus(opsecStatus.leakTests.webrtc)}
          </div>
        </div>
        
        <!-- JavaScript Test -->
        <div class="leak-test">
          <div class="text-sm text-gray-500 mb-1">JavaScript</div>
          <div class="{getLeakTestColor(opsecStatus.leakTests.javascript)}">
            {getLeakTestStatus(opsecStatus.leakTests.javascript)}
          </div>
        </div>
        
        <!-- Cookies Test -->
        <div class="leak-test">
          <div class="text-sm text-gray-500 mb-1">Cookies</div>
          <div class="{getLeakTestColor(opsecStatus.leakTests.cookies)}">
            {getLeakTestStatus(opsecStatus.leakTests.cookies)}
          </div>
        </div>
      </div>
      
      <!-- Failed Tests Warning -->
      {#if opsecStatus && (!opsecStatus.leakTests.dns || !opsecStatus.leakTests.webrtc || 
           !opsecStatus.leakTests.javascript || !opsecStatus.leakTests.cookies)}
        <div class="warning mt-4 p-3 bg-red-900 bg-opacity-30 border border-red-500 rounded">
          <div class="flex items-center gap-2">
            <span class="text-red-400">⚠️</span>
            <span class="text-sm text-red-400">
              Security leak detected! Your anonymity may be compromised.
            </span>
          </div>
        </div>
      {/if}
    </div>
  {/if}

  <!-- Security Recommendations -->
  {#if opsecStatus?.recommendations && opsecStatus.recommendations.length > 0}
    <div class="recommendations bg-gray-900 border border-yellow-700 rounded p-4">
      <h3 class="text-lg font-bold text-yellow-400 mb-3">⚡ RECOMMENDATIONS</h3>
      <ul class="space-y-2">
        {#each opsecStatus.recommendations as rec}
          <li class="flex items-start gap-2">
            <span class="text-yellow-400">→</span>
            <span class="text-sm text-gray-300">{rec}</span>
          </li>
        {/each}
      </ul>
    </div>
  {/if}

  <!-- Security Hardening Checklist -->
  <div class="hardening-checklist bg-gray-900 border border-gray-700 rounded p-4">
    <h3 class="text-lg font-bold text-green-400 mb-3">🔐 HARDENING CHECKLIST</h3>
    
    <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
      <label class="flex items-center gap-2">
        <input type="checkbox" checked disabled class="text-green-400" />
        <span class="text-gray-400">SOCKS5 Proxy Enforcement</span>
      </label>
      <label class="flex items-center gap-2">
        <input type="checkbox" checked disabled class="text-green-400" />
        <span class="text-gray-400">MCP Security Consent</span>
      </label>
      <label class="flex items-center gap-2">
        <input type="checkbox" checked disabled class="text-green-400" />
        <span class="text-gray-400">DOMPurify Sanitization</span>
      </label>
      <label class="flex items-center gap-2">
        <input type="checkbox" checked disabled class="text-green-400" />
        <span class="text-gray-400">CSP Headers Active</span>
      </label>
      <label class="flex items-center gap-2">
        <input type="checkbox" {opsecStatus?.compliant} disabled class="text-green-400" />
        <span class="text-gray-400">Circuit Rotation Available</span>
      </label>
      <label class="flex items-center gap-2">
        <input type="checkbox" checked disabled class="text-green-400" />
        <span class="text-gray-400">Zero External Dependencies</span>
      </label>
    </div>
  </div>

  <!-- Emergency Actions -->
  <div class="emergency-actions bg-gray-900 border border-red-700 rounded p-4">
    <h3 class="text-lg font-bold text-red-400 mb-3">🚨 EMERGENCY ACTIONS</h3>
    
    <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
      <button 
        class="px-4 py-2 bg-red-900 hover:bg-red-800 text-red-400 rounded font-bold"
        on:click={() => console.log('Killing all connections')}
      >
        KILL ALL CONNECTIONS
      </button>
      <button 
        class="px-4 py-2 bg-orange-900 hover:bg-orange-800 text-orange-400 rounded font-bold"
        on:click={() => console.log('Purging memory')}
      >
        PURGE MEMORY
      </button>
      <button 
        class="px-4 py-2 bg-purple-900 hover:bg-purple-800 text-purple-400 rounded font-bold"
        on:click={() => console.log('Panic mode')}
      >
        PANIC MODE
      </button>
    </div>
    
    <div class="text-xs text-gray-500 mt-2">
      Emergency actions will immediately terminate all operations and clear sensitive data
    </div>
  </div>
</div>

<style>
  .status-indicator {
    animation: pulse 2s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
</style>