<script lang="ts">
    import { onMount } from 'svelte';
    import { Client } from '@modelcontextprotocol/sdk';
    
    interface Agent {
        id: string;
        name: string;
        status: 'idle' | 'running' | 'error' | 'complete';
        task: string;
        progress: number;
        output?: string;
    }
    
    interface MCPTool {
        name: string;
        description: string;
        requiresConsent: boolean;
    }
    
    let agents: Agent[] = [
        { id: 'osint-01', name: 'DarknetCrawler', status: 'running', task: 'Scanning AlphaBay3 listings', progress: 67 },
        { id: 'osint-02', name: 'BlockchainAnalyzer', status: 'idle', task: 'Awaiting wallet addresses', progress: 0 },
        { id: 'osint-03', name: 'ThreatHunter', status: 'complete', task: 'IOC extraction complete', progress: 100, output: '342 IOCs extracted' }
    ];
    
    let mcpConnected = false;
    let mcpEndpoint = 'http://localhost:3010';
    let pendingConsent: MCPTool | null = null;
    
    async function connectMCP() {
        try {
            // Initialize MCP client
            mcpConnected = true;
        } catch (error) {
            console.error('MCP connection failed:', error);
        }
    }
    
    function handleConsent(approved: boolean) {
        if (pendingConsent && approved) {
            console.log(`Tool ${pendingConsent.name} approved`);
        }
        pendingConsent = null;
    }
    
    function deployAgent(agentId: string) {
        const agent = agents.find(a => a.id === agentId);
        if (agent) {
            agent.status = 'running';
            agents = agents;
        }
    }
</script>

<div class="agent-coordinator">
    <h2>Multi-Agent Coordination</h2>
    
    <div class="mcp-status">
        <h3>MCP Server Status</h3>
        <div class="status-panel">
            <span class="endpoint">Endpoint: {mcpEndpoint}</span>
            <span class="status" class:connected={mcpConnected}>
                {mcpConnected ? '● CONNECTED' : '○ DISCONNECTED'}
            </span>
            {#if !mcpConnected}
                <button on:click={connectMCP} class="connect-btn">Connect</button>
            {/if}
        </div>
    </div>
    
    <div class="agents-grid">
        {#each agents as agent}
            <div class="agent-card status-{agent.status}">
                <div class="agent-header">
                    <span class="agent-name">{agent.name}</span>
                    <span class="agent-id">{agent.id}</span>
                </div>
                <div class="agent-task">{agent.task}</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {agent.progress}%"></div>
                </div>
                <div class="agent-status">
                    Status: <span class="status-badge">{agent.status.toUpperCase()}</span>
                </div>
                {#if agent.output}
                    <div class="agent-output">{agent.output}</div>
                {/if}
                {#if agent.status === 'idle'}
                    <button on:click={() => deployAgent(agent.id)} class="deploy-btn">
                        Deploy Agent
                    </button>
                {/if}
            </div>
        {/each}
    </div>
    
    {#if pendingConsent}
        <div class="consent-modal">
            <div class="consent-content">
                <h3>Security Consent Required</h3>
                <p>Tool: {pendingConsent.name}</p>
                <p>{pendingConsent.description}</p>
                <div class="consent-actions">
                    <button on:click={() => handleConsent(true)} class="approve-btn">
                        APPROVE
                    </button>
                    <button on:click={() => handleConsent(false)} class="deny-btn">
                        DENY
                    </button>
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .agent-coordinator {
        padding: 1rem;
    }
    
    h2, h3 {
        color: var(--text-primary);
        margin-bottom: 1rem;
        text-transform: uppercase;
    }
    
    .mcp-status {
        margin-bottom: 2rem;
    }
    
    .status-panel {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        padding: 1rem;
        display: flex;
        align-items: center;
        gap: 2rem;
    }
    
    .endpoint {
        font-family: monospace;
        color: var(--text-muted);
    }
    
    .status.connected {
        color: var(--text-primary);
    }
    
    .connect-btn, .deploy-btn {
        background: transparent;
        border: 1px solid var(--text-primary);
        color: var(--text-primary);
        padding: 4px 12px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .connect-btn:hover, .deploy-btn:hover {
        background: var(--text-primary);
        color: #000;
    }
    
    .agents-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1rem;
    }
    
    .agent-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        padding: 1rem;
    }
    
    .agent-card.status-running {
        border-color: var(--text-primary);
    }
    
    .agent-card.status-error {
        border-color: var(--accent);
    }
    
    .agent-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    
    .agent-name {
        font-weight: bold;
        color: var(--text-primary);
    }
    
    .agent-id {
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    
    .agent-task {
        font-size: 0.875rem;
        margin-bottom: 0.5rem;
    }
    
    .progress-bar {
        height: 4px;
        background: var(--bg-tertiary);
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: var(--text-primary);
        transition: width 0.3s ease;
    }
    
    .status-badge {
        font-weight: bold;
    }
    
    .consent-modal {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    }
    
    .consent-content {
        background: var(--bg-secondary);
        border: 2px solid var(--text-primary);
        padding: 2rem;
        max-width: 500px;
    }
    
    .consent-actions {
        display: flex;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .approve-btn, .deny-btn {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .approve-btn {
        border-color: var(--text-primary);
        color: var(--text-primary);
    }
    
    .deny-btn {
        border-color: var(--accent);
        color: var(--accent);
    }
</style>