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

  // Advanced AI Pipeline state
  const aiPipelineState = writable({
    context_compression: {
      status: 'active',
      compression_ratio: 87.4,
      token_reduction: 156234,
      processing_speed: '2.3M tokens/sec',
      model_efficiency: 94.7,
      recovery_status: 'healthy',
      cache_hit_rate: 78.9
    },
    extended_reasoning: {
      status: 'reasoning',
      active_sessions: 23,
      token_capacity: '100K+',
      reasoning_depth: 8,
      hypothesis_generation: 156,
      conclusion_accuracy: 92.8,
      recovery_status: 'healthy'
    },
    model_synchronization: {
      status: 'syncing',
      models_synchronized: 47,
      sync_frequency: '15min',
      version_conflicts: 0,
      distribution_efficiency: 89.3,
      edge_sync_status: 'active'
    },
    t2v_transformers: {
      status: 'transforming',
      active_transformations: 12,
      queue_length: 34,
      success_rate: 96.2,
      avg_processing_time: '3m 47s',
      gpu_utilization: 84.7
    },
    adaptive_learning: {
      status: 'learning',
      learning_rate: 0.001,
      model_updates: 2847,
      performance_improvement: 18.3,
      adaptation_cycles: 156,
      knowledge_retention: 94.7
    }
  });

  // AI Pipeline controls
  let activeAIWorkspace = 'compression';
  let selectedModel = '';
  let processingMode = 'batch';

  // Context compression configuration
  let compressionConfig = {
    algorithm: 'semantic_preserving',
    target_ratio: 0.75,
    quality_threshold: 0.9,
    batch_size: 100,
    enable_caching: true,
    recovery_enabled: true
  };

  // Extended reasoning configuration
  let reasoningConfig = {
    max_depth: 10,
    token_limit: 100000,
    hypothesis_count: 5,
    confidence_threshold: 0.8,
    parallel_reasoning: true,
    result_validation: true
  };

  // Model synchronization configuration
  let syncConfig = {
    sync_frequency: 15,
    automatic_updates: true,
    version_control: true,
    rollback_enabled: true,
    edge_distribution: true,
    conflict_resolution: 'latest_wins'
  };

  // T2V transformation configuration
  let t2vConfig = {
    model_version: 'stable-diffusion-xl',
    output_resolution: '1024x1024',
    batch_processing: true,
    quality_setting: 'high',
    gpu_optimization: true,
    concurrent_jobs: 4
  };

  // Live data stores
  const activeCompressions = writable([]);
  const reasoningSessions = writable([]);
  const modelStatus = writable([]);
  const transformationQueue = writable([]);

  // WebSocket connections
  let compressionWs: WebSocket | null = null;
  let reasoningWs: WebSocket | null = null;
  let syncWs: WebSocket | null = null;
  let t2vWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadPipelineData();
    startMetricsCollection();
  });

  onDestroy(() => {
    if (compressionWs) compressionWs.close();
    if (reasoningWs) reasoningWs.close();
    if (syncWs) syncWs.close();
    if (t2vWs) t2vWs.close();
  });

  function initializeWebSockets() {
    // Context compression WebSocket
    compressionWs = new WebSocket('ws://${getWebSocketHost()}:8080/compression');
    compressionWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      aiPipelineState.update(state => ({
        ...state,
        context_compression: { ...state.context_compression, ...data }
      }));
      if (data.active_compressions) {
        activeCompressions.set(data.active_compressions);
      }
    };

    // Extended reasoning WebSocket
    reasoningWs = new WebSocket('ws://${getWebSocketHost()}:8081/reasoning');
    reasoningWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      aiPipelineState.update(state => ({
        ...state,
        extended_reasoning: { ...state.extended_reasoning, ...data }
      }));
      if (data.sessions) {
        reasoningSessions.set(data.sessions);
      }
    };

    // Model synchronization WebSocket
    syncWs = new WebSocket('ws://${getWebSocketHost()}:8082/sync');
    syncWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      aiPipelineState.update(state => ({
        ...state,
        model_synchronization: { ...state.model_synchronization, ...data }
      }));
      if (data.models) {
        modelStatus.set(data.models);
      }
    };

    // T2V transformers WebSocket
    t2vWs = new WebSocket('ws://${getWebSocketHost()}:8083/t2v');
    t2vWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      aiPipelineState.update(state => ({
        ...state,
        t2v_transformers: { ...state.t2v_transformers, ...data }
      }));
      if (data.queue) {
        transformationQueue.set(data.queue);
      }
    };
  }

  async function loadPipelineData() {
    try {
      const [compressionRes, reasoningRes, modelsRes, queueRes] = await Promise.all([
        fetch('http://${getServiceHost()}:8080/api/compressions'),
        fetch('http://${getServiceHost()}:8081/api/sessions'),
        fetch('http://${getServiceHost()}:8082/api/models'),
        fetch('http://${getServiceHost()}:8083/api/queue')
      ]);

      const compressions = await compressionRes.json();
      const sessions = await reasoningRes.json();
      const models = await modelsRes.json();
      const queue = await queueRes.json();

      activeCompressions.set(compressions);
      reasoningSessions.set(sessions);
      modelStatus.set(models);
      transformationQueue.set(queue);
    } catch (error) {
      console.error('Failed to load pipeline data:', error);
    }
  }

  async function startMetricsCollection() {
    setInterval(async () => {
      try {
        const response = await fetch('http://${getServiceHost()}:8080/api/metrics');
        const metrics = await response.json();
        aiPipelineState.update(state => ({
          ...state,
          ...metrics
        }));
      } catch (error) {
        console.error('AI Pipeline metrics collection error:', error);
      }
    }, 5000);
  }

  async function startContextCompression() {
    try {
      const response = await fetch('http://${getServiceHost()}:8080/api/compress', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(compressionConfig)
      });

      if (response.ok) {
        const compression = await response.json();
        console.log('Context compression started:', compression);
      }
    } catch (error) {
      console.error('Context compression failed:', error);
    }
  }

  async function startExtendedReasoning() {
    try {
      const response = await fetch('http://${getServiceHost()}:8081/api/reason', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(reasoningConfig)
      });

      if (response.ok) {
        const session = await response.json();
        console.log('Extended reasoning started:', session);
      }
    } catch (error) {
      console.error('Extended reasoning failed:', error);
    }
  }

  async function synchronizeModels() {
    try {
      const response = await fetch('http://${getServiceHost()}:8082/api/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(syncConfig)
      });

      if (response.ok) {
        console.log('Model synchronization initiated');
      }
    } catch (error) {
      console.error('Model sync failed:', error);
    }
  }

  async function submitT2VTransformation() {
    try {
      const response = await fetch('http://${getServiceHost()}:8083/api/transform', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(t2vConfig)
      });

      if (response.ok) {
        const transformation = await response.json();
        console.log('T2V transformation queued:', transformation);
      }
    } catch (error) {
      console.error('T2V transformation failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'active': case 'healthy': case 'completed': return 'text-green-400';
      case 'reasoning': case 'syncing': case 'transforming': case 'learning': return 'text-blue-400';
      case 'failed': case 'error': case 'critical': return 'text-red-400';
      case 'warning': case 'degraded': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  }

  function getHealthColor(value: number, threshold: number = 90): string {
    if (value >= threshold) return 'text-green-400';
    if (value >= threshold - 15) return 'text-yellow-400';
    return 'text-red-400';
  }
</script>

<svelte:head>
  <title>Advanced AI Pipeline Controls | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-indigo-400 to-purple-500 bg-clip-text text-transparent">
        Advanced AI Pipeline Controls
      </h1>
      <p class="text-gray-300">Context compression, extended reasoning, and model coordination</p>
    </div>

    {#if $aiPipelineState}
      <!-- AI Pipeline Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-5 gap-6 mb-8">
        <!-- Context Compression -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Context Compression
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($aiPipelineState.context_compression.status)}>
                {$aiPipelineState.context_compression.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Compression:</span>
              <span class={getHealthColor($aiPipelineState.context_compression.compression_ratio, 80)}>{$aiPipelineState.context_compression.compression_ratio}%</span>
            </div>
            <div class="flex justify-between">
              <span>Tokens Saved:</span>
              <span class="text-green-400">{$aiPipelineState.context_compression.token_reduction.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Speed:</span>
              <span class="text-cyan-400">{$aiPipelineState.context_compression.processing_speed}</span>
            </div>
          </div>
        </div>

        <!-- Extended Reasoning -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Extended Reasoning
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($aiPipelineState.extended_reasoning.status)}>
                {$aiPipelineState.extended_reasoning.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Sessions:</span>
              <span class="text-purple-400">{$aiPipelineState.extended_reasoning.active_sessions}</span>
            </div>
            <div class="flex justify-between">
              <span>Capacity:</span>
              <span class="text-blue-400">{$aiPipelineState.extended_reasoning.token_capacity}</span>
            </div>
            <div class="flex justify-between">
              <span>Accuracy:</span>
              <span class={getHealthColor($aiPipelineState.extended_reasoning.conclusion_accuracy)}>{$aiPipelineState.extended_reasoning.conclusion_accuracy}%</span>
            </div>
          </div>
        </div>

        <!-- Model Synchronization -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Model Sync
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($aiPipelineState.model_synchronization.status)}>
                {$aiPipelineState.model_synchronization.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Models:</span>
              <span class="text-green-400">{$aiPipelineState.model_synchronization.models_synchronized}</span>
            </div>
            <div class="flex justify-between">
              <span>Frequency:</span>
              <span class="text-blue-400">{$aiPipelineState.model_synchronization.sync_frequency}</span>
            </div>
            <div class="flex justify-between">
              <span>Conflicts:</span>
              <span class="text-green-400">{$aiPipelineState.model_synchronization.version_conflicts}</span>
            </div>
          </div>
        </div>

        <!-- T2V Transformers -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-yellow-400 mr-2"></span>
            T2V Transformers
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($aiPipelineState.t2v_transformers.status)}>
                {$aiPipelineState.t2v_transformers.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-yellow-400">{$aiPipelineState.t2v_transformers.active_transformations}</span>
            </div>
            <div class="flex justify-between">
              <span>Queue:</span>
              <span class="text-blue-400">{$aiPipelineState.t2v_transformers.queue_length}</span>
            </div>
            <div class="flex justify-between">
              <span>GPU Usage:</span>
              <span class="text-green-400">{$aiPipelineState.t2v_transformers.gpu_utilization}%</span>
            </div>
          </div>
        </div>

        <!-- Adaptive Learning -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-orange-400 mr-2"></span>
            Adaptive Learning
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($aiPipelineState.adaptive_learning.status)}>
                {$aiPipelineState.adaptive_learning.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Updates:</span>
              <span class="text-orange-400">{$aiPipelineState.adaptive_learning.model_updates.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Improvement:</span>
              <span class="text-green-400">+{$aiPipelineState.adaptive_learning.performance_improvement}%</span>
            </div>
            <div class="flex justify-between">
              <span>Retention:</span>
              <span class={getHealthColor($aiPipelineState.adaptive_learning.knowledge_retention)}>{$aiPipelineState.adaptive_learning.knowledge_retention}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- AI Pipeline Workspace Tabs -->
      <div class="mb-6">
        <div class="flex space-x-1 bg-gray-800 p-1 rounded-lg">
          {#each ['compression', 'reasoning', 'synchronization', 't2v', 'learning'] as workspace}
            <button
              on:click={() => activeAIWorkspace = workspace}
              class="flex-1 py-2 px-4 rounded text-center transition-colors text-sm
                {activeAIWorkspace === workspace
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'}"
            >
              {workspace.charAt(0).toUpperCase() + workspace.slice(1)}
            </button>
          {/each}
        </div>
      </div>

      <!-- Workspace Content -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {#if activeAIWorkspace === 'compression'}
          <!-- Context Compression Controls -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Context Compression Pipeline</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Compression Algorithm</label>
                <select bind:value={compressionConfig.algorithm} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="semantic_preserving">Semantic Preserving</option>
                  <option value="lossy_optimized">Lossy Optimized</option>
                  <option value="neural_compression">Neural Compression</option>
                  <option value="hybrid_approach">Hybrid Approach</option>
                </select>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Target Compression Ratio: {compressionConfig.target_ratio}</label>
                <input
                  type="range"
                  bind:value={compressionConfig.target_ratio}
                  min="0.1"
                  max="0.95"
                  step="0.05"
                  class="w-full"
                >
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Quality Threshold: {compressionConfig.quality_threshold}</label>
                <input
                  type="range"
                  bind:value={compressionConfig.quality_threshold}
                  min="0.5"
                  max="1.0"
                  step="0.05"
                  class="w-full"
                >
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Batch Size</label>
                  <input
                    type="number"
                    bind:value={compressionConfig.batch_size}
                    min="1"
                    max="1000"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div class="space-y-2 pt-6">
                  <label class="flex items-center">
                    <input type="checkbox" bind:checked={compressionConfig.enable_caching} class="mr-2">
                    <span class="text-sm">Enable Caching</span>
                  </label>
                  <label class="flex items-center">
                    <input type="checkbox" bind:checked={compressionConfig.recovery_enabled} class="mr-2">
                    <span class="text-sm">Recovery Enabled</span>
                  </label>
                </div>
              </div>
              <button
                on:click={startContextCompression}
                class="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
              >
                Start Context Compression
              </button>
            </div>
          </div>

          <!-- Compression Status -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Compression Status</h3>
            <div class="space-y-4">
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span>Model Efficiency</span>
                  <span>{$aiPipelineState.context_compression.model_efficiency}%</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2">
                  <div class="bg-blue-400 h-2 rounded-full" style="width: {$aiPipelineState.context_compression.model_efficiency}%"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span>Cache Hit Rate</span>
                  <span>{$aiPipelineState.context_compression.cache_hit_rate}%</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2">
                  <div class="bg-green-400 h-2 rounded-full" style="width: {$aiPipelineState.context_compression.cache_hit_rate}%"></div>
                </div>
              </div>
              <div class="text-sm space-y-1">
                <div class="flex justify-between">
                  <span>Tokens Reduced:</span>
                  <span class="text-green-400">{$aiPipelineState.context_compression.token_reduction.toLocaleString()}</span>
                </div>
                <div class="flex justify-between">
                  <span>Recovery Status:</span>
                  <span class={getStatusColor($aiPipelineState.context_compression.recovery_status)}>
                    {$aiPipelineState.context_compression.recovery_status.toUpperCase()}
                  </span>
                </div>
              </div>
            </div>
          </div>

        {:else if activeAIWorkspace === 'reasoning'}
          <!-- Extended Reasoning Controls -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Extended Reasoning Configuration</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Maximum Reasoning Depth: {reasoningConfig.max_depth}</label>
                <input
                  type="range"
                  bind:value={reasoningConfig.max_depth}
                  min="1"
                  max="20"
                  class="w-full"
                >
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Token Limit: {reasoningConfig.token_limit.toLocaleString()}</label>
                <input
                  type="range"
                  bind:value={reasoningConfig.token_limit}
                  min="10000"
                  max="200000"
                  step="10000"
                  class="w-full"
                >
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Hypothesis Count</label>
                  <input
                    type="number"
                    bind:value={reasoningConfig.hypothesis_count}
                    min="1"
                    max="10"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Confidence Threshold</label>
                  <input
                    type="number"
                    bind:value={reasoningConfig.confidence_threshold}
                    min="0.1"
                    max="1.0"
                    step="0.1"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={reasoningConfig.parallel_reasoning} class="mr-2">
                  <span class="text-sm">Parallel Reasoning</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={reasoningConfig.result_validation} class="mr-2">
                  <span class="text-sm">Result Validation</span>
                </label>
              </div>
              <button
                on:click={startExtendedReasoning}
                class="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded"
              >
                Start Extended Reasoning
              </button>
            </div>
          </div>

          <!-- Reasoning Sessions -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Active Sessions</h3>
            <div class="space-y-3">
              {#if $reasoningSessions && $reasoningSessions.length > 0}
                {#each $reasoningSessions.slice(0, 5) as session}
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">Session {session.id}</span>
                      <span class={getStatusColor(session.status)} class="text-xs">
                        {session.status.toUpperCase()}
                      </span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Depth: {session.current_depth}/{session.max_depth}</div>
                      <div>Tokens: {session.tokens_used.toLocaleString()}</div>
                      <div>Hypotheses: {session.hypotheses_generated}</div>
                    </div>
                  </div>
                {/each}
              {:else}
                <div class="text-center text-gray-400 py-4">
                  No active reasoning sessions
                </div>
              {/if}
            </div>
          </div>

        {:else if activeAIWorkspace === 'synchronization'}
          <!-- Model Synchronization -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Model Synchronization Control</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Sync Frequency (minutes): {syncConfig.sync_frequency}</label>
                <input
                  type="range"
                  bind:value={syncConfig.sync_frequency}
                  min="5"
                  max="120"
                  step="5"
                  class="w-full"
                >
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Conflict Resolution</label>
                <select bind:value={syncConfig.conflict_resolution} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="latest_wins">Latest Version Wins</option>
                  <option value="manual_review">Manual Review Required</option>
                  <option value="performance_based">Performance-Based Selection</option>
                  <option value="voting_consensus">Voting Consensus</option>
                </select>
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={syncConfig.automatic_updates} class="mr-2">
                  <span class="text-sm">Automatic Updates</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={syncConfig.version_control} class="mr-2">
                  <span class="text-sm">Version Control</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={syncConfig.edge_distribution} class="mr-2">
                  <span class="text-sm">Edge Distribution</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={syncConfig.rollback_enabled} class="mr-2">
                  <span class="text-sm">Rollback Enabled</span>
                </label>
              </div>
              <button
                on:click={synchronizeModels}
                class="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded"
              >
                Synchronize All Models
              </button>
            </div>
          </div>

          <!-- Model Status -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Model Status</h3>
            <div class="space-y-3">
              {#if $modelStatus && $modelStatus.length > 0}
                {#each $modelStatus.slice(0, 5) as model}
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">{model.name}</span>
                      <span class={getStatusColor(model.sync_status)} class="text-xs">
                        {model.sync_status.toUpperCase()}
                      </span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Version: {model.version}</div>
                      <div>Size: {model.size}</div>
                      <div>Last Sync: {model.last_sync}</div>
                    </div>
                  </div>
                {/each}
              {:else}
                <div class="text-center text-gray-400 py-4">
                  No model synchronization data
                </div>
              {/if}
            </div>
          </div>

        {:else if activeAIWorkspace === 't2v'}
          <!-- T2V Transformation Controls -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Text-to-Video Transformation</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Model Version</label>
                <select bind:value={t2vConfig.model_version} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="stable-diffusion-xl">Stable Diffusion XL</option>
                  <option value="runway-gen2">RunwayML Gen-2</option>
                  <option value="pika-labs">Pika Labs</option>
                  <option value="animatediff">AnimateDiff</option>
                </select>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Output Resolution</label>
                  <select bind:value={t2vConfig.output_resolution} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="512x512">512x512</option>
                    <option value="1024x1024">1024x1024</option>
                    <option value="1920x1080">1920x1080</option>
                    <option value="2048x2048">2048x2048</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Quality Setting</label>
                  <select bind:value={t2vConfig.quality_setting} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="draft">Draft</option>
                    <option value="standard">Standard</option>
                    <option value="high">High</option>
                    <option value="ultra">Ultra</option>
                  </select>
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Concurrent Jobs: {t2vConfig.concurrent_jobs}</label>
                <input
                  type="range"
                  bind:value={t2vConfig.concurrent_jobs}
                  min="1"
                  max="8"
                  class="w-full"
                >
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={t2vConfig.batch_processing} class="mr-2">
                  <span class="text-sm">Batch Processing</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={t2vConfig.gpu_optimization} class="mr-2">
                  <span class="text-sm">GPU Optimization</span>
                </label>
              </div>
              <button
                on:click={submitT2VTransformation}
                class="w-full bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded"
              >
                Queue T2V Transformation
              </button>
            </div>
          </div>

          <!-- Transformation Queue -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Transformation Queue</h3>
            <div class="space-y-3">
              {#if $transformationQueue && $transformationQueue.length > 0}
                {#each $transformationQueue.slice(0, 5) as transformation}
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">Job {transformation.id}</span>
                      <span class={getStatusColor(transformation.status)} class="text-xs">
                        {transformation.status.toUpperCase()}
                      </span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Model: {transformation.model}</div>
                      <div>Resolution: {transformation.resolution}</div>
                      <div>Progress: {transformation.progress}%</div>
                    </div>
                  </div>
                {/each}
              {:else}
                <div class="text-center text-gray-400 py-4">
                  No transformations in queue
                </div>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>