<script lang="ts">
  import { endpoints, websockets, getEndpoint, getWebSocket } from "$lib/config/endpoints";

  // Distributed endpoint helpers
  const getServiceHost = () => {
    const service = typeof window !== "undefined" && window.location.hostname;
    return service === "localhost" ? "localhost" : service;
  };

  const getWebSocketHost = () => {
    const service = typeof window !== "undefined" && window.location.hostname;
    return service === "localhost" ? "localhost" : service;
  };
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // Real-time autonomous systems state
  const autonomousState = writable({
    coordinator: {
      status: 'active',
      operations_count: 0,
      decision_model_version: '3.2.1',
      autonomous_level: 3,
      max_operations: 50,
      safety_constraints_active: true,
      operation_timeout: 3600
    },
    adaptive_learning: {
      status: 'learning',
      model_version: '2.1.5',
      learning_rate: 0.001,
      performance_score: 8.7,
      improvement_rate: 12.3,
      training_iterations: 15847,
      hyperparameter_tuning: true
    },
    resource_manager: {
      status: 'optimizing',
      cpu_usage: 72.4,
      memory_usage: 68.1,
      disk_usage: 45.2,
      efficiency_gain: 18.3,
      cost_optimization: 14.7,
      scaling_events: 23
    },
    knowledge_evolution: {
      status: 'evolving',
      ontology_version: '4.1.2',
      concept_count: 15673,
      semantic_relations: 47291,
      pruning_actions: 127,
      evolution_cycles: 8934
    },
    active_agents: [],
    decision_tree: null,
    communication_matrix: []
  });

  // WebSocket connections for real-time data
  let autonomousWs: WebSocket | null = null;
  let adaptiveLearningWs: WebSocket | null = null;
  let resourceManagerWs: WebSocket | null = null;
  let knowledgeEvolutionWs: WebSocket | null = null;

  // Agent orchestration controls
  let selectedOperation = '';
  let operationPriority = 'normal';
  let autonomyLevel = 3;
  let safetyOverride = false;

  // Task delegation matrix
  const taskCategories = [
    'data_collection', 'analysis', 'threat_detection', 'vulnerability_scan',
    'network_monitoring', 'log_analysis', 'pattern_recognition', 'prediction',
    'resource_optimization', 'knowledge_synthesis', 'decision_making', 'learning'
  ];

  onMount(() => {
    initializeWebSockets();
    loadDecisionTree();
    startPerformanceMonitoring();
  });

  onDestroy(() => {
    if (autonomousWs) autonomousWs.close();
    if (adaptiveLearningWs) adaptiveLearningWs.close();
    if (resourceManagerWs) resourceManagerWs.close();
    if (knowledgeEvolutionWs) knowledgeEvolutionWs.close();
  });

  function initializeWebSockets() {
    // Autonomous coordinator WebSocket
    autonomousWs = new WebSocket('ws://${getWebSocketHost()}:8009/ws');
    autonomousWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      autonomousState.update(state => ({
        ...state,
        coordinator: { ...state.coordinator, ...data.coordinator },
        active_agents: data.active_agents || state.active_agents
      }));
    };

    // Adaptive learning WebSocket
    adaptiveLearningWs = new WebSocket('ws://${getWebSocketHost()}:8010/ws');
    adaptiveLearningWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      autonomousState.update(state => ({
        ...state,
        adaptive_learning: { ...state.adaptive_learning, ...data }
      }));
    };

    // Resource manager WebSocket
    resourceManagerWs = new WebSocket('ws://${getWebSocketHost()}:8011/ws');
    resourceManagerWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      autonomousState.update(state => ({
        ...state,
        resource_manager: { ...state.resource_manager, ...data }
      }));
    };

    // Knowledge evolution WebSocket
    knowledgeEvolutionWs = new WebSocket('ws://${getWebSocketHost()}:8012/ws');
    knowledgeEvolutionWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      autonomousState.update(state => ({
        ...state,
        knowledge_evolution: { ...state.knowledge_evolution, ...data }
      }));
    };
  }

  async function loadDecisionTree() {
    try {
      const response = await fetch('http://${getServiceHost()}:8009/api/decision-tree');
      const decisionTree = await response.json();
      autonomousState.update(state => ({ ...state, decision_tree: decisionTree }));
    } catch (error) {
      console.error('Failed to load decision tree:', error);
    }
  }

  async function startPerformanceMonitoring() {
    setInterval(async () => {
      try {
        const response = await fetch('http://${getServiceHost()}:8009/api/performance');
        const performance = await response.json();
        autonomousState.update(state => ({
          ...state,
          coordinator: { ...state.coordinator, ...performance }
        }));
      } catch (error) {
        console.error('Performance monitoring error:', error);
      }
    }, 5000);
  }

  async function delegateTask() {
    if (!selectedOperation) return;

    try {
      const response = await fetch('http://${getServiceHost()}:8009/api/delegate-task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          operation: selectedOperation,
          priority: operationPriority,
          autonomy_level: autonomyLevel,
          safety_override: safetyOverride
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Task delegated:', result);
      }
    } catch (error) {
      console.error('Task delegation failed:', error);
    }
  }

  async function adjustAutonomyLevel(level: number) {
    try {
      await fetch('http://${getServiceHost()}:8009/api/autonomy-level', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level })
      });
      autonomyLevel = level;
    } catch (error) {
      console.error('Failed to adjust autonomy level:', error);
    }
  }

  async function emergencyStop() {
    try {
      await fetch('http://${getServiceHost()}:8009/api/emergency-stop', {
        method: 'POST'
      });
    } catch (error) {
      console.error('Emergency stop failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'learning': return 'text-blue-400';
      case 'optimizing': return 'text-yellow-400';
      case 'evolving': return 'text-purple-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  }
</script>

<svelte:head>
  <title>Phase 9 - Autonomous Systems | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
        Phase 9 - Autonomous Systems
      </h1>
      <p class="text-gray-300">Advanced autonomous operations with intelligent coordination</p>
    </div>

    {#if $autonomousState}
      <!-- System Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
        <!-- Autonomous Coordinator -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Autonomous Coordinator
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($autonomousState.coordinator.status)}>
                {$autonomousState.coordinator.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Operations:</span>
              <span class="text-yellow-400">{$autonomousState.coordinator.operations_count}/50</span>
            </div>
            <div class="flex justify-between">
              <span>Autonomy Level:</span>
              <span class="text-blue-400">Level {$autonomousState.coordinator.autonomous_level}</span>
            </div>
            <div class="flex justify-between">
              <span>Model Version:</span>
              <span class="text-green-400">{$autonomousState.coordinator.decision_model_version}</span>
            </div>
          </div>
        </div>

        <!-- Adaptive Learning -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Adaptive Learning
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($autonomousState.adaptive_learning.status)}>
                {$autonomousState.adaptive_learning.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Performance:</span>
              <span class="text-green-400">{$autonomousState.adaptive_learning.performance_score}/10</span>
            </div>
            <div class="flex justify-between">
              <span>Improvement:</span>
              <span class="text-blue-400">+{$autonomousState.adaptive_learning.improvement_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Iterations:</span>
              <span class="text-yellow-400">{$autonomousState.adaptive_learning.training_iterations.toLocaleString()}</span>
            </div>
          </div>
        </div>

        <!-- Resource Manager -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-yellow-400 mr-2"></span>
            Resource Manager
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>CPU Usage:</span>
              <span class="text-yellow-400">{$autonomousState.resource_manager.cpu_usage}%</span>
            </div>
            <div class="flex justify-between">
              <span>Memory:</span>
              <span class="text-blue-400">{$autonomousState.resource_manager.memory_usage}%</span>
            </div>
            <div class="flex justify-between">
              <span>Efficiency Gain:</span>
              <span class="text-green-400">+{$autonomousState.resource_manager.efficiency_gain}%</span>
            </div>
            <div class="flex justify-between">
              <span>Cost Savings:</span>
              <span class="text-green-400">{$autonomousState.resource_manager.cost_optimization}%</span>
            </div>
          </div>
        </div>

        <!-- Knowledge Evolution -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Knowledge Evolution
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($autonomousState.knowledge_evolution.status)}>
                {$autonomousState.knowledge_evolution.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Concepts:</span>
              <span class="text-blue-400">{$autonomousState.knowledge_evolution.concept_count.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Relations:</span>
              <span class="text-purple-400">{$autonomousState.knowledge_evolution.semantic_relations.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Evolution Cycles:</span>
              <span class="text-green-400">{$autonomousState.knowledge_evolution.evolution_cycles.toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Control Panel -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <!-- Task Delegation -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Task Delegation</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Operation Type</label>
              <select bind:value={selectedOperation} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="">Select Operation</option>
                {#each taskCategories as category}
                  <option value={category}>{category.replace('_', ' ').toUpperCase()}</option>
                {/each}
              </select>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Priority Level</label>
              <select bind:value={operationPriority} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                <option value="low">Low</option>
                <option value="normal">Normal</option>
                <option value="high">High</option>
                <option value="critical">Critical</option>
              </select>
            </div>
            <div class="flex items-center">
              <input type="checkbox" bind:checked={safetyOverride} id="safety-override" class="mr-2">
              <label for="safety-override" class="text-sm">Safety Override</label>
            </div>
            <button
              on:click={delegateTask}
              disabled={!selectedOperation}
              class="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-2 rounded"
            >
              Delegate Task
            </button>
          </div>
        </div>

        <!-- Autonomy Controls -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Autonomy Controls</h3>
          <div class="space-y-4">
            <div>
              <label class="block text-sm font-medium mb-2">Autonomy Level: {autonomyLevel}</label>
              <input
                type="range"
                min="1"
                max="5"
                bind:value={autonomyLevel}
                on:change={() => adjustAutonomyLevel(autonomyLevel)}
                class="w-full"
              >
              <div class="flex justify-between text-xs text-gray-400 mt-1">
                <span>Manual</span>
                <span>Supervised</span>
                <span>Semi-Auto</span>
                <span>Auto</span>
                <span>Full Auto</span>
              </div>
            </div>
            <div class="bg-gray-700 rounded p-3">
              <h4 class="font-medium mb-2">Current Mode</h4>
              <p class="text-sm text-gray-300">
                {autonomyLevel === 1 ? 'Manual operation with human approval' :
                 autonomyLevel === 2 ? 'Supervised autonomous with oversight' :
                 autonomyLevel === 3 ? 'Semi-autonomous with safety constraints' :
                 autonomyLevel === 4 ? 'Autonomous with minimal oversight' :
                 'Fully autonomous operation'}
              </p>
            </div>
            <button
              on:click={emergencyStop}
              class="w-full bg-red-600 hover:bg-red-700 px-4 py-2 rounded font-semibold"
            >
              EMERGENCY STOP
            </button>
          </div>
        </div>

        <!-- Performance Metrics -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Performance Metrics</h3>
          <div class="space-y-4">
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>Decision Accuracy</span>
                <span>94.7%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-green-400 h-2 rounded-full" style="width: 94.7%"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>Task Completion Rate</span>
                <span>98.2%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-blue-400 h-2 rounded-full" style="width: 98.2%"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>Learning Efficiency</span>
                <span>87.3%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-purple-400 h-2 rounded-full" style="width: 87.3%"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>Resource Optimization</span>
                <span>91.8%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-yellow-400 h-2 rounded-full" style="width: 91.8%"></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Active Agents Matrix -->
      {#if $autonomousState.active_agents && $autonomousState.active_agents.length > 0}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
          <h3 class="text-lg font-semibold mb-4">Active Agents Matrix</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {#each $autonomousState.active_agents as agent}
              <div class="bg-gray-700 rounded p-4">
                <div class="flex items-center justify-between mb-2">
                  <span class="font-medium">{agent.name || 'Agent ' + agent.id}</span>
                  <span class="text-xs px-2 py-1 rounded bg-green-600">{agent.status}</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Task: {agent.current_task || 'Idle'}</div>
                  <div>Progress: {agent.progress || 0}%</div>
                  <div>Uptime: {agent.uptime || '0h 0m'}</div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Decision Tree Visualization -->
      {#if $autonomousState.decision_tree}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Decision Tree Visualization</h3>
          <div class="bg-gray-900 rounded p-4 min-h-96 flex items-center justify-center">
            <p class="text-gray-400">Decision tree visualization component would be rendered here</p>
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>