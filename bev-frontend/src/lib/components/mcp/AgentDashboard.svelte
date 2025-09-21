<!-- BEV Agent Dashboard - Multi-Agent Coordination & Monitoring -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { mcpStore, connectedAgents, workingAgents } from '$lib/stores/mcpStore';
  import type { Agent, AgentTask, Workflow } from '$lib/mcp/types';
  
  let selectedAgent: Agent | null = null;
  let showWorkflowModal = false;
  let workflowName = '';
  let workflowAgents: string[] = [];

  // Agent status colors
  function getStatusColor(status: string): string {
    switch(status) {
      case 'working': return 'text-green-400 bg-green-900';
      case 'idle': return 'text-cyan-400 bg-cyan-900';
      case 'error': return 'text-red-400 bg-red-900';
      case 'offline': return 'text-gray-400 bg-gray-800';
      default: return 'text-gray-400 bg-gray-800';
    }
  }

  // Task priority colors
  function getPriorityColor(priority: string): string {
    switch(priority) {
      case 'critical': return 'text-red-400';
      case 'high': return 'text-orange-400';
      case 'medium': return 'text-yellow-400';
      case 'low': return 'text-green-400';
      default: return 'text-gray-400';
    }
  }

  // Format resource percentage
  function formatPercent(value: number): string {
    return `${value.toFixed(1)}%`;
  }

  // Format uptime
  function formatUptime(seconds: number): string {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  }

  // Create workflow
  function createWorkflow() {
    if (!workflowName || workflowAgents.length === 0) return;
    
    // In production, this would send to backend
    console.log('Creating workflow:', {
      name: workflowName,
      agents: workflowAgents
    });
    
    showWorkflowModal = false;
    workflowName = '';
    workflowAgents = [];
  }

  // Toggle agent selection for workflow
  function toggleAgentForWorkflow(agentId: string) {
    if (workflowAgents.includes(agentId)) {
      workflowAgents = workflowAgents.filter(id => id !== agentId);
    } else {
      workflowAgents = [...workflowAgents, agentId];
    }
  }

  onMount(() => {
    mcpStore.refreshAgentStatus();
  });
</script>

<div class="agent-dashboard h-full bg-black text-green-400 font-mono p-4 overflow-y-auto">
  <!-- Header -->
  <div class="dashboard-header mb-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-2xl font-bold text-green-400">AGENT COORDINATION CENTER</h2>
      <div class="flex items-center gap-4">
        <div class="status-summary flex items-center gap-4 text-sm">
          <div>Active: <span class="text-cyan-400">{$connectedAgents.length}</span></div>
          <div>Working: <span class="text-yellow-400">{$workingAgents.length}</span></div>
          <div>Tasks: <span class="text-orange-400">{$mcpStore.systemMetrics?.mcp.queuedTasks || 0}</span></div>
        </div>
        <button
          class="px-4 py-2 bg-green-600 hover:bg-green-500 text-black font-bold rounded"
          on:click={() => showWorkflowModal = true}
        >
          CREATE WORKFLOW
        </button>
      </div>
    </div>
  </div>

  <!-- Agent Grid -->
  <div class="agent-grid grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4 mb-6">
    {#each $mcpStore.agents as agent}
      <div 
        class="agent-card bg-gray-900 border border-gray-700 rounded-lg p-4 cursor-pointer hover:border-green-500 transition-colors"
        class:border-green-500={selectedAgent?.id === agent.id}
        on:click={() => selectedAgent = agent}
      >
        <!-- Agent Header -->
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-2">
            <div class="agent-icon text-2xl">
              {#if agent.type === 'osint'}
                🔍
              {:else if agent.type === 'crypto'}
                ₿
              {:else if agent.type === 'darknet'}
                🌑
              {:else if agent.type === 'monitor'}
                📡
              {:else}
                🤖
              {/if}
            </div>
            <div>
              <h3 class="font-bold text-cyan-400">{agent.name}</h3>
              <span class="text-xs text-gray-500">{agent.type.toUpperCase()}</span>
            </div>
          </div>
          <span class="status-badge px-2 py-1 text-xs rounded {getStatusColor(agent.status)}">
            {agent.status.toUpperCase()}
          </span>
        </div>

        <!-- Metrics -->
        <div class="agent-metrics space-y-2 text-xs">
          <div class="flex justify-between">
            <span class="text-gray-500">Tasks:</span>
            <span>{agent.metrics.tasksCompleted} completed / {agent.metrics.tasksFailed} failed</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-500">Avg Response:</span>
            <span>{agent.metrics.averageResponseTime}ms</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-500">Uptime:</span>
            <span>{formatUptime(agent.metrics.uptime)}</span>
          </div>
        </div>

        <!-- Resource Usage -->
        <div class="resource-usage mt-3 pt-3 border-t border-gray-700">
          <div class="text-xs text-gray-500 mb-1">Resources:</div>
          <div class="grid grid-cols-3 gap-2 text-xs">
            <div>
              <div class="text-gray-500">CPU</div>
              <div class="text-cyan-400">{formatPercent(agent.metrics.resourceUsage.cpu)}</div>
            </div>
            <div>
              <div class="text-gray-500">MEM</div>
              <div class="text-yellow-400">{formatPercent(agent.metrics.resourceUsage.memory)}</div>
            </div>
            <div>
              <div class="text-gray-500">NET</div>
              <div class="text-green-400">{formatPercent(agent.metrics.resourceUsage.network)}</div>
            </div>
          </div>
        </div>

        <!-- Current Task -->
        {#if agent.currentTask}
          <div class="current-task mt-3 pt-3 border-t border-gray-700">
            <div class="text-xs text-gray-500 mb-1">Current Task:</div>
            <div class="bg-black p-2 rounded">
              <div class="flex items-center justify-between mb-1">
                <span class="text-xs {getPriorityColor(agent.currentTask.priority)}">
                  [{agent.currentTask.priority.toUpperCase()}]
                </span>
                <span class="text-xs text-gray-400">{agent.currentTask.progress}%</span>
              </div>
              <div class="text-xs text-gray-300">{agent.currentTask.description}</div>
              <div class="progress-bar mt-2 h-1 bg-gray-800 rounded overflow-hidden">
                <div 
                  class="progress-fill h-full bg-green-500 transition-all"
                  style="width: {agent.currentTask.progress}%"
                ></div>
              </div>
            </div>
          </div>
        {/if}

        <!-- Capabilities -->
        <div class="capabilities mt-3 pt-3 border-t border-gray-700">
          <div class="text-xs text-gray-500 mb-1">Capabilities:</div>
          <div class="flex flex-wrap gap-1">
            {#each agent.capabilities.slice(0, 3) as capability}
              <span class="px-2 py-0.5 text-xs bg-gray-800 text-gray-400 rounded">
                {capability}
              </span>
            {/each}
            {#if agent.capabilities.length > 3}
              <span class="px-2 py-0.5 text-xs bg-gray-800 text-gray-500 rounded">
                +{agent.capabilities.length - 3}
              </span>
            {/if}
          </div>
        </div>
      </div>
    {/each}
  </div>

  <!-- Active Workflows -->
  {#if $mcpStore.activeWorkflows.length > 0}
    <div class="workflows-section mb-6">
      <h3 class="text-lg font-bold text-green-400 mb-4">ACTIVE WORKFLOWS</h3>
      <div class="space-y-3">
        {#each $mcpStore.activeWorkflows as workflow}
          <div class="workflow-card bg-gray-900 border border-gray-700 rounded-lg p-4">
            <div class="flex items-center justify-between mb-3">
              <div>
                <h4 class="font-bold text-cyan-400">{workflow.name}</h4>
                <p class="text-xs text-gray-500">{workflow.description}</p>
              </div>
              <span class="status-badge px-2 py-1 text-xs rounded {
                workflow.status === 'running' ? 'bg-green-900 text-green-400' :
                workflow.status === 'completed' ? 'bg-blue-900 text-blue-400' :
                workflow.status === 'failed' ? 'bg-red-900 text-red-400' :
                'bg-gray-800 text-gray-400'
              }">
                {workflow.status.toUpperCase()}
              </span>
            </div>
            
            <!-- Workflow Steps -->
            <div class="workflow-steps space-y-2">
              {#each workflow.steps as step}
                <div class="step flex items-center gap-2 text-xs">
                  <div class="step-status w-2 h-2 rounded-full {
                    step.status === 'completed' ? 'bg-green-500' :
                    step.status === 'running' ? 'bg-yellow-500 animate-pulse' :
                    step.status === 'failed' ? 'bg-red-500' :
                    'bg-gray-600'
                  }"></div>
                  <span class="text-gray-400">{step.action}</span>
                  <span class="text-gray-600">→</span>
                  <span class="text-gray-500">{step.agentId}</span>
                </div>
              {/each}
            </div>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- Selected Agent Details -->
  {#if selectedAgent}
    <div class="agent-details bg-gray-900 border border-green-500 rounded-lg p-4">
      <h3 class="text-lg font-bold text-green-400 mb-4">
        AGENT DETAILS: {selectedAgent.name}
      </h3>
      <div class="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span class="text-gray-500">ID:</span>
          <span class="text-cyan-400 ml-2 font-mono">{selectedAgent.id}</span>
        </div>
        <div>
          <span class="text-gray-500">Last Active:</span>
          <span class="text-yellow-400 ml-2">
            {new Date(selectedAgent.metrics.lastActive).toLocaleTimeString()}
          </span>
        </div>
        <div class="col-span-2">
          <span class="text-gray-500">Full Capabilities:</span>
          <div class="flex flex-wrap gap-2 mt-2">
            {#each selectedAgent.capabilities as capability}
              <span class="px-3 py-1 text-xs bg-gray-800 text-green-400 rounded">
                {capability}
              </span>
            {/each}
          </div>
        </div>
      </div>
    </div>
  {/if}

  <!-- Workflow Creation Modal -->
  {#if showWorkflowModal}
    <div class="modal fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div class="modal-content bg-gray-900 border-2 border-green-500 rounded-lg p-6 max-w-lg w-full">
        <h3 class="text-xl font-bold text-green-400 mb-4">CREATE WORKFLOW</h3>
        
        <input
          bind:value={workflowName}
          class="w-full px-4 py-2 mb-4 bg-black text-green-400 border border-green-500 rounded"
          placeholder="Workflow name..."
        />
        
        <div class="agents-selection mb-4">
          <label class="text-sm text-gray-400 mb-2 block">Select Agents:</label>
          <div class="space-y-2 max-h-48 overflow-y-auto">
            {#each $connectedAgents as agent}
              <label class="flex items-center gap-2 cursor-pointer hover:bg-gray-800 p-2 rounded">
                <input
                  type="checkbox"
                  checked={workflowAgents.includes(agent.id)}
                  on:change={() => toggleAgentForWorkflow(agent.id)}
                  class="text-green-400"
                />
                <span class="text-sm">{agent.name} ({agent.type})</span>
              </label>
            {/each}
          </div>
        </div>
        
        <div class="flex gap-3">
          <button
            class="flex-1 px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded"
            on:click={() => showWorkflowModal = false}
          >
            CANCEL
          </button>
          <button
            class="flex-1 px-4 py-2 bg-green-600 hover:bg-green-500 text-black font-bold rounded"
            on:click={createWorkflow}
            disabled={!workflowName || workflowAgents.length === 0}
          >
            CREATE
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .progress-fill {
    transition: width 0.5s ease-in-out;
  }
</style>
