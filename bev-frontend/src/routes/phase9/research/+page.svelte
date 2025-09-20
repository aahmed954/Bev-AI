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

  // Advanced research systems state
  const researchState = writable({
    correlation_engine: {
      status: 'active',
      active_correlations: 47,
      data_sources: 23,
      confidence_threshold: 0.85,
      processing_speed: '2.3M events/sec',
      pattern_detection_rate: 94.7
    },
    pipeline_builder: {
      status: 'ready',
      active_pipelines: 12,
      total_stages: 156,
      success_rate: 98.2,
      avg_execution_time: '47s',
      optimization_level: 87.3
    },
    hypothesis_testing: {
      status: 'testing',
      active_hypotheses: 8,
      proven_hypotheses: 23,
      disproven_hypotheses: 7,
      statistical_confidence: 95.2,
      test_completion_rate: 89.1
    },
    synthesis_workbench: {
      status: 'synthesizing',
      data_points_processed: 1547892,
      synthesis_algorithms: 15,
      knowledge_graphs: 34,
      semantic_relations: 98456,
      synthesis_accuracy: 92.8
    },
    citation_graph: {
      status: 'building',
      total_citations: 456789,
      research_papers: 23456,
      cross_references: 87654,
      citation_networks: 234,
      relevance_score: 91.4
    }
  });

  // Research workspace
  let activeWorkspace = 'correlation';
  let selectedDataSources = [];
  let correlationQuery = '';
  let hypothesisFormulation = '';
  let synthesisParameters = {
    algorithm: 'neural_semantic',
    confidence_threshold: 0.8,
    max_depth: 5,
    include_external: true
  };

  // Pipeline configuration
  let pipelineConfig = {
    name: '',
    stages: [],
    triggers: [],
    data_sources: [],
    outputs: []
  };

  // Research projects
  const activeProjects = writable([]);
  const researchTemplates = [
    'Threat Intelligence Research',
    'Market Analysis Study',
    'Vulnerability Assessment',
    'Behavioral Pattern Analysis',
    'Network Topology Research',
    'Social Network Analysis',
    'Economic Impact Study',
    'Technology Trend Analysis'
  ];

  // WebSocket connections
  let correlationWs: WebSocket | null = null;
  let pipelineWs: WebSocket | null = null;
  let hypothesisWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadActiveProjects();
    startMetricsCollection();
  });

  onDestroy(() => {
    if (correlationWs) correlationWs.close();
    if (pipelineWs) pipelineWs.close();
    if (hypothesisWs) hypothesisWs.close();
  });

  function initializeWebSockets() {
    // Multi-source correlation engine
    correlationWs = new WebSocket('ws://${getWebSocketHost()}:8013/correlation');
    correlationWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      researchState.update(state => ({
        ...state,
        correlation_engine: { ...state.correlation_engine, ...data }
      }));
    };

    // Research pipeline builder
    pipelineWs = new WebSocket('ws://${getWebSocketHost()}:8014/pipeline');
    pipelineWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      researchState.update(state => ({
        ...state,
        pipeline_builder: { ...state.pipeline_builder, ...data }
      }));
    };

    // Hypothesis testing framework
    hypothesisWs = new WebSocket('ws://${getWebSocketHost()}:8015/hypothesis');
    hypothesisWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      researchState.update(state => ({
        ...state,
        hypothesis_testing: { ...state.hypothesis_testing, ...data }
      }));
    };
  }

  async function loadActiveProjects() {
    try {
      const response = await fetch('http://${getServiceHost()}:8013/api/projects');
      const projects = await response.json();
      activeProjects.set(projects);
    } catch (error) {
      console.error('Failed to load projects:', error);
    }
  }

  async function startMetricsCollection() {
    setInterval(async () => {
      try {
        const response = await fetch('http://${getServiceHost()}:8013/api/metrics');
        const metrics = await response.json();
        researchState.update(state => ({
          ...state,
          ...metrics
        }));
      } catch (error) {
        console.error('Metrics collection error:', error);
      }
    }, 10000);
  }

  async function startCorrelationAnalysis() {
    if (!correlationQuery) return;

    try {
      const response = await fetch('http://${getServiceHost()}:8013/api/correlate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: correlationQuery,
          data_sources: selectedDataSources,
          threshold: synthesisParameters.confidence_threshold
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Correlation analysis started:', result);
      }
    } catch (error) {
      console.error('Correlation analysis failed:', error);
    }
  }

  async function createResearchPipeline() {
    try {
      const response = await fetch('http://${getServiceHost()}:8014/api/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(pipelineConfig)
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Pipeline created:', result);
      }
    } catch (error) {
      console.error('Pipeline creation failed:', error);
    }
  }

  async function submitHypothesis() {
    if (!hypothesisFormulation) return;

    try {
      const response = await fetch('http://${getServiceHost()}:8015/api/hypothesis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hypothesis: hypothesisFormulation,
          test_parameters: synthesisParameters
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Hypothesis submitted:', result);
      }
    } catch (error) {
      console.error('Hypothesis submission failed:', error);
    }
  }

  async function synthesizeData() {
    try {
      const response = await fetch('http://${getServiceHost()}:8016/api/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(synthesisParameters)
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Data synthesis started:', result);
      }
    } catch (error) {
      console.error('Data synthesis failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'ready': return 'text-blue-400';
      case 'testing': return 'text-yellow-400';
      case 'synthesizing': return 'text-purple-400';
      case 'building': return 'text-cyan-400';
      default: return 'text-gray-400';
    }
  }
</script>

<svelte:head>
  <title>Phase 9 - Advanced Research Systems | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
        Phase 9 - Advanced Research Systems
      </h1>
      <p class="text-gray-300">Multi-source correlation, hypothesis testing, and intelligent synthesis</p>
    </div>

    {#if $researchState}
      <!-- System Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-5 gap-6 mb-8">
        <!-- Correlation Engine -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Correlation Engine
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($researchState.correlation_engine.status)}>
                {$researchState.correlation_engine.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-yellow-400">{$researchState.correlation_engine.active_correlations}</span>
            </div>
            <div class="flex justify-between">
              <span>Sources:</span>
              <span class="text-blue-400">{$researchState.correlation_engine.data_sources}</span>
            </div>
            <div class="flex justify-between">
              <span>Speed:</span>
              <span class="text-green-400">{$researchState.correlation_engine.processing_speed}</span>
            </div>
          </div>
        </div>

        <!-- Pipeline Builder -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Pipeline Builder
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($researchState.pipeline_builder.status)}>
                {$researchState.pipeline_builder.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Pipelines:</span>
              <span class="text-yellow-400">{$researchState.pipeline_builder.active_pipelines}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class="text-green-400">{$researchState.pipeline_builder.success_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Avg Time:</span>
              <span class="text-blue-400">{$researchState.pipeline_builder.avg_execution_time}</span>
            </div>
          </div>
        </div>

        <!-- Hypothesis Testing -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-yellow-400 mr-2"></span>
            Hypothesis Testing
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($researchState.hypothesis_testing.status)}>
                {$researchState.hypothesis_testing.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-yellow-400">{$researchState.hypothesis_testing.active_hypotheses}</span>
            </div>
            <div class="flex justify-between">
              <span>Proven:</span>
              <span class="text-green-400">{$researchState.hypothesis_testing.proven_hypotheses}</span>
            </div>
            <div class="flex justify-between">
              <span>Confidence:</span>
              <span class="text-blue-400">{$researchState.hypothesis_testing.statistical_confidence}%</span>
            </div>
          </div>
        </div>

        <!-- Synthesis Workbench -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Synthesis Workbench
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($researchState.synthesis_workbench.status)}>
                {$researchState.synthesis_workbench.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Data Points:</span>
              <span class="text-yellow-400">{$researchState.synthesis_workbench.data_points_processed.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Algorithms:</span>
              <span class="text-purple-400">{$researchState.synthesis_workbench.synthesis_algorithms}</span>
            </div>
            <div class="flex justify-between">
              <span>Accuracy:</span>
              <span class="text-green-400">{$researchState.synthesis_workbench.synthesis_accuracy}%</span>
            </div>
          </div>
        </div>

        <!-- Citation Graph -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-cyan-400 mr-2"></span>
            Citation Graph
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($researchState.citation_graph.status)}>
                {$researchState.citation_graph.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Citations:</span>
              <span class="text-yellow-400">{$researchState.citation_graph.total_citations.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Papers:</span>
              <span class="text-cyan-400">{$researchState.citation_graph.research_papers.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Relevance:</span>
              <span class="text-green-400">{$researchState.citation_graph.relevance_score}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Research Workspace Tabs -->
      <div class="mb-6">
        <div class="flex space-x-1 bg-gray-800 p-1 rounded-lg">
          {#each ['correlation', 'pipeline', 'hypothesis', 'synthesis'] as workspace}
            <button
              on:click={() => activeWorkspace = workspace}
              class="flex-1 py-2 px-4 rounded text-center transition-colors
                {activeWorkspace === workspace
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'}"
            >
              {workspace.charAt(0).toUpperCase() + workspace.slice(1)}
            </button>
          {/each}
        </div>
      </div>

      <!-- Workspace Content -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {#if activeWorkspace === 'correlation'}
          <!-- Correlation Analysis -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Multi-Source Correlation Analysis</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Correlation Query</label>
                <textarea
                  bind:value={correlationQuery}
                  placeholder="Enter correlation query or pattern to analyze..."
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 h-24"
                ></textarea>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Data Sources</label>
                <div class="grid grid-cols-2 gap-2">
                  {#each ['threat_intel', 'darknet_data', 'social_media', 'financial_data', 'technical_reports', 'news_feeds'] as source}
                    <label class="flex items-center">
                      <input type="checkbox" bind:group={selectedDataSources} value={source} class="mr-2">
                      <span class="text-sm">{source.replace('_', ' ').toUpperCase()}</span>
                    </label>
                  {/each}
                </div>
              </div>
              <button
                on:click={startCorrelationAnalysis}
                disabled={!correlationQuery || selectedDataSources.length === 0}
                class="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                Start Correlation Analysis
              </button>
            </div>
          </div>

          <!-- Correlation Results -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Live Correlations</h3>
            <div class="space-y-3">
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Network Pattern</span>
                  <span class="text-green-400">94.7%</span>
                </div>
                <p class="text-sm text-gray-300">Cross-domain attack pattern detected</p>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Financial Flow</span>
                  <span class="text-yellow-400">87.3%</span>
                </div>
                <p class="text-sm text-gray-300">Suspicious transaction correlation</p>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Social Cluster</span>
                  <span class="text-blue-400">91.2%</span>
                </div>
                <p class="text-sm text-gray-300">Communication pattern identified</p>
              </div>
            </div>
          </div>

        {:else if activeWorkspace === 'pipeline'}
          <!-- Pipeline Builder -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Research Pipeline Builder</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Pipeline Name</label>
                <input
                  bind:value={pipelineConfig.name}
                  placeholder="Enter pipeline name..."
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Research Template</label>
                <select class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="">Select template</option>
                  {#each researchTemplates as template}
                    <option value={template}>{template}</option>
                  {/each}
                </select>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Triggers</label>
                  <select multiple class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 h-24">
                    <option value="schedule">Scheduled</option>
                    <option value="event">Event-driven</option>
                    <option value="threshold">Threshold-based</option>
                    <option value="manual">Manual</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Output Formats</label>
                  <select multiple class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 h-24">
                    <option value="json">JSON</option>
                    <option value="csv">CSV</option>
                    <option value="pdf">PDF Report</option>
                    <option value="graph">Graph Data</option>
                  </select>
                </div>
              </div>
              <button
                on:click={createResearchPipeline}
                disabled={!pipelineConfig.name}
                class="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                Create Pipeline
              </button>
            </div>
          </div>

          <!-- Active Pipelines -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Active Pipelines</h3>
            <div class="space-y-3">
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Threat Intel</span>
                  <span class="text-green-400">Running</span>
                </div>
                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-green-400 h-2 rounded-full" style="width: 78%"></div>
                </div>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Market Analysis</span>
                  <span class="text-blue-400">Processing</span>
                </div>
                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-blue-400 h-2 rounded-full" style="width: 45%"></div>
                </div>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Social Network</span>
                  <span class="text-yellow-400">Queued</span>
                </div>
                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-yellow-400 h-2 rounded-full" style="width: 12%"></div>
                </div>
              </div>
            </div>
          </div>

        {:else if activeWorkspace === 'hypothesis'}
          <!-- Hypothesis Testing -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Hypothesis Testing Framework</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Hypothesis Formulation</label>
                <textarea
                  bind:value={hypothesisFormulation}
                  placeholder="Enter your research hypothesis..."
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 h-32"
                ></textarea>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Statistical Test</label>
                  <select class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="chi_square">Chi-Square Test</option>
                    <option value="t_test">T-Test</option>
                    <option value="anova">ANOVA</option>
                    <option value="correlation">Correlation Analysis</option>
                    <option value="regression">Regression Analysis</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Significance Level</label>
                  <select class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="0.05">α = 0.05 (95% confidence)</option>
                    <option value="0.01">α = 0.01 (99% confidence)</option>
                    <option value="0.001">α = 0.001 (99.9% confidence)</option>
                  </select>
                </div>
              </div>
              <button
                on:click={submitHypothesis}
                disabled={!hypothesisFormulation}
                class="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                Submit for Testing
              </button>
            </div>
          </div>

          <!-- Testing Results -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Testing Results</h3>
            <div class="space-y-3">
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Network Hypothesis</span>
                  <span class="text-green-400">Proven</span>
                </div>
                <p class="text-sm text-gray-300">p-value: 0.003 (α = 0.05)</p>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Market Correlation</span>
                  <span class="text-red-400">Disproven</span>
                </div>
                <p class="text-sm text-gray-300">p-value: 0.087 (α = 0.05)</p>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Behavioral Pattern</span>
                  <span class="text-yellow-400">Testing</span>
                </div>
                <p class="text-sm text-gray-300">Progress: 67%</p>
              </div>
            </div>
          </div>

        {:else if activeWorkspace === 'synthesis'}
          <!-- Data Synthesis -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Data Synthesis Workbench</h3>
            <div class="space-y-4">
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Synthesis Algorithm</label>
                  <select bind:value={synthesisParameters.algorithm} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="neural_semantic">Neural Semantic</option>
                    <option value="graph_neural">Graph Neural Network</option>
                    <option value="transformer">Transformer-based</option>
                    <option value="ensemble">Ensemble Methods</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Confidence Threshold</label>
                  <input
                    type="range"
                    bind:value={synthesisParameters.confidence_threshold}
                    min="0.1"
                    max="1.0"
                    step="0.1"
                    class="w-full"
                  >
                  <span class="text-sm text-gray-400">{synthesisParameters.confidence_threshold}</span>
                </div>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Max Depth</label>
                  <input
                    type="number"
                    bind:value={synthesisParameters.max_depth}
                    min="1"
                    max="10"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div class="flex items-center">
                  <input type="checkbox" bind:checked={synthesisParameters.include_external} id="include-external" class="mr-2">
                  <label for="include-external" class="text-sm">Include External Sources</label>
                </div>
              </div>
              <button
                on:click={synthesizeData}
                class="w-full bg-orange-600 hover:bg-orange-700 px-4 py-2 rounded"
              >
                Start Synthesis
              </button>
            </div>
          </div>

          <!-- Synthesis Output -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Synthesis Output</h3>
            <div class="space-y-3">
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Knowledge Graph</span>
                  <span class="text-green-400">Complete</span>
                </div>
                <p class="text-sm text-gray-300">1,247 nodes, 3,891 edges</p>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Semantic Relations</span>
                  <span class="text-blue-400">Processing</span>
                </div>
                <p class="text-sm text-gray-300">78% complete</p>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Report Generation</span>
                  <span class="text-yellow-400">Queued</span>
                </div>
                <p class="text-sm text-gray-300">Waiting for synthesis</p>
              </div>
            </div>
          </div>
        {/if}
      </div>

      <!-- Active Research Projects -->
      {#if $activeProjects && $activeProjects.length > 0}
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Active Research Projects</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {#each $activeProjects as project}
              <div class="bg-gray-700 rounded p-4">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">{project.name}</span>
                  <span class="text-xs px-2 py-1 rounded bg-blue-600">{project.status}</span>
                </div>
                <div class="text-sm text-gray-300 mb-2">
                  <div>Type: {project.type}</div>
                  <div>Progress: {project.progress}%</div>
                </div>
                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-blue-400 h-2 rounded-full" style="width: {project.progress}%"></div>
                </div>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    {/if}
  </div>
</div>