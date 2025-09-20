<script lang="ts">
    import { onMount, onDestroy } from 'svelte';
    import { invoke } from '@tauri-apps/api/core';
    
    export let proxyConnected = false;
    export let exitIp = '';
    
    let circuitId = '';
    let torVersion = '';
    let checkInterval: NodeJS.Timeout;
    
    async function rotateCircuit() {
        try {
            await invoke('rotate_circuit');
            await checkStatus();
        } catch (error) {
            console.error('Circuit rotation failed:', error);
        }
    }
    
    async function checkStatus() {
        try {
            const status = await invoke('verify_proxy_status');
            proxyConnected = status.connected;
            exitIp = status.exit_ip || 'Unknown';
            circuitId = status.circuit_id || 'N/A';
            torVersion = status.tor_version || 'Unknown';
        } catch (error) {
            proxyConnected = false;
        }
    }
    
    onMount(() => {
        checkStatus();
        checkInterval = setInterval(checkStatus, 30000); // Check every 30s
    });
    
    onDestroy(() => {
        if (checkInterval) clearInterval(checkInterval);
    });
</script>

<div class="proxy-indicator">
    <div class="status-dot" class:connected={proxyConnected} class:disconnected={!proxyConnected}></div>
    <span class="status-text">
        {#if proxyConnected}
            TOR: ACTIVE | Exit: {exitIp} | Circuit: {circuitId.slice(0, 8)}...
        {:else}
            TOR: DISCONNECTED - OPSEC COMPROMISED
        {/if}
    </span>
    {#if proxyConnected}
        <button on:click={rotateCircuit} class="rotate-btn">
            ⟲ Rotate
        </button>
    {/if}
</div>

<style>
    .proxy-indicator {
        display: flex;
        align-items: center;
        gap: 12px;
        font-family: monospace;
    }
    
    .status-text {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .rotate-btn {
        background: transparent;
        border: 1px solid var(--text-primary);
        color: var(--text-primary);
        padding: 2px 8px;
        font-size: 10px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .rotate-btn:hover {
        background: var(--text-primary);
        color: #000;
    }
</style>