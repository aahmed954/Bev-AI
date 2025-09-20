<!-- Interactive Knowledge Graph Visualization -->
<script lang="ts">
  import { createEventDispatcher, onMount, onDestroy } from 'svelte';
  import cytoscape from 'cytoscape';
  import fcose from 'cytoscape-fcose';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  const dispatch = createEventDispatcher();
  
  export let nodes = [];
  export let selectedResults = [];
  export let selectedVectorDB = 'qdrant';
  
  cytoscape.use(fcose);
  
  let graphContainer: HTMLElement;
  let cy: any;
  let selectedNode: any = null;
  let layoutType = 'fcose';
  let showLabels = true;
  let showEdgeLabels = false;
  let nodeSize = 'centrality'; // 'uniform', 'degree', 'centrality', 'similarity'
  let colorScheme = 'type'; // 'type', 'cluster', 'similarity'
  let filterByType = 'all';
  let searchGraph = '';
  let graphStats = {
    totalNodes: 0,
    totalEdges: 0,
    clusters: 0,
    density: 0
  };

  const nodeTypes = {
    document: { color: '#00ff41', icon: '📄', size: 40 },
    entity: { color: '#00ccff', icon: '🏷️', size: 30 },
    concept: { color: '#ff9500', icon: '💡', size: 35 },
    relationship: { color: '#ff00ff', icon: '🔗', size: 25 }
  };

  const layouts = [
    { value: 'fcose', name: 'Force-Directed' },
    { value: 'cose', name: 'Compound' },
    { value: 'circle', name: 'Circular' },
    { value: 'grid', name: 'Grid' },
    { value: 'breadthfirst', name: 'Hierarchical' },
    { value: 'concentric', name: 'Concentric' }
  ];

  onMount(() => {
    initializeGraph();
    loadKnowledgeGraph();
  });

  onDestroy(() => {
    if (cy) {
      cy.destroy();
    }
  });

  function initializeGraph() {
    cy = cytoscape({
      container: graphContainer,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'label': showLabels ? 'data(label)' : '',
            'color': '#ffffff',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 'data(size)',
            'height': 'data(size)',
            'font-size': '10px',
            'text-outline-width': 2,
            'text-outline-color': '#000000',
            'border-width': 2,
            'border-color': '#333333',
            'text-wrap': 'wrap',
            'text-max-width': '80px'
          }
        },
        {
          selector: 'node[type="document"]',
          style: {
            'shape': 'round-rectangle',
            'background-color': '#00ff41'
          }
        },
        {
          selector: 'node[type="entity"]',
          style: {
            'shape': 'ellipse',
            'background-color': '#00ccff'
          }
        },
        {
          selector: 'node[type="concept"]',
          style: {
            'shape': 'diamond',
            'background-color': '#ff9500'
          }
        },
        {
          selector: 'node[type="relationship"]',
          style: {
            'shape': 'triangle',
            'background-color': '#ff00ff'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 'data(weight)',
            'line-color': '#666666',
            'target-arrow-color': '#666666',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': showEdgeLabels ? 'data(relationship)' : '',
            'font-size': '8px',
            'color': '#cccccc',
            'text-background-color': '#000000',
            'text-background-opacity': 0.8,
            'text-background-padding': '2px'
          }
        },
        {
          selector: ':selected',
          style: {
            'border-width': 4,
            'border-color': '#ffff00',
            'box-shadow': '0 0 20px #ffff00'
          }
        },
        {
          selector: '.highlighted',
          style: {
            'background-color': '#ffff00',
            'line-color': '#ffff00',
            'target-arrow-color': '#ffff00',
            'border-color': '#ffff00',
            'opacity': 1
          }
        },
        {
          selector: '.faded',
          style: {
            'opacity': 0.3
          }
        }
      ],
      layout: {
        name: layoutType,
        animate: true,
        animationDuration: 1000,
        fit: true,
        padding: 50
      },
      minZoom: 0.1,
      maxZoom: 3,
      wheelSensitivity: 0.2
    });

    // Event handlers
    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      selectNode(node);
      highlightConnections(node);
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        clearSelection();
      }
    });

    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target;
      showNodeTooltip(node);
    });
  }

  async function loadKnowledgeGraph() {
    try {
      const graphData = await invoke('get_knowledge_graph', {
        vectorDB: selectedVectorDB,
        includeRelationships: true,
        maxNodes: 500,
        similarityThreshold: 0.6
      });

      updateGraph(graphData.nodes, graphData.edges);
      updateGraphStats(graphData);
    } catch (error) {
      console.error('Failed to load knowledge graph:', error);
      // Generate mock data for development
      generateMockGraphData();
    }
  }

  function generateMockGraphData() {
    const mockNodes = [
      { id: 'doc1', label: 'Crypto Analysis Report', type: 'document', properties: { size: 45 } },
      { id: 'entity1', label: 'Bitcoin', type: 'entity', properties: { size: 35 } },
      { id: 'concept1', label: 'Money Laundering', type: 'concept', properties: { size: 40 } },
      { id: 'rel1', label: 'Transaction', type: 'relationship', properties: { size: 25 } }
    ];

    const mockEdges = [
      { source: 'doc1', target: 'entity1', relationship: 'mentions', weight: 3 },
      { source: 'entity1', target: 'concept1', relationship: 'associated_with', weight: 2 },
      { source: 'concept1', target: 'rel1', relationship: 'involves', weight: 1 }
    ];

    updateGraph(mockNodes, mockEdges);
    graphStats = { totalNodes: 4, totalEdges: 3, clusters: 1, density: 0.75 };
  }

  function updateGraph(nodeData, edgeData) {
    if (!cy) return;

    const elements = [];

    // Add nodes
    nodeData.forEach(node => {
      const nodeType = nodeTypes[node.type] || nodeTypes.concept;
      elements.push({
        data: {
          id: node.id,
          label: node.label,
          type: node.type,
          color: nodeType.color,
          size: calculateNodeSize(node),
          properties: node.properties || {}
        }
      });
    });

    // Add edges
    edgeData.forEach(edge => {
      elements.push({
        data: {
          id: `${edge.source}-${edge.target}`,
          source: edge.source,
          target: edge.target,
          relationship: edge.relationship,
          weight: edge.weight || 1
        }
      });
    });

    cy.elements().remove();
    cy.add(elements);
    applyLayout();
  }

  function calculateNodeSize(node) {
    switch (nodeSize) {
      case 'uniform':
        return 30;
      case 'degree':
        return Math.max(20, Math.min(60, (node.connections?.length || 1) * 10));
      case 'centrality':
        return Math.max(25, Math.min(70, (node.properties?.centrality || 0.5) * 100));
      case 'similarity':
        return Math.max(20, Math.min(50, (node.properties?.similarity || 0.5) * 80));
      default:
        return nodeTypes[node.type]?.size || 30;
    }
  }

  function applyLayout() {
    if (!cy) return;

    const layoutOptions = {
      name: layoutType,
      animate: true,
      animationDuration: 1000,
      fit: true,
      padding: 50
    };

    // Layout-specific options
    if (layoutType === 'fcose') {
      Object.assign(layoutOptions, {
        quality: 'proof',
        randomize: true,
        nodeDimensionsIncludeLabels: true,
        uniformNodeDimensions: false,
        packComponents: true
      });
    }

    cy.layout(layoutOptions).run();
  }

  function selectNode(node) {
    selectedNode = node;
    
    const nodeData = {
      id: node.id(),
      label: node.data('label'),
      type: node.data('type'),
      properties: node.data('properties'),
      degree: node.degree(),
      connections: node.connectedEdges().length
    };

    dispatch('nodeSelected', nodeData);
  }

  function highlightConnections(node) {
    cy.elements().removeClass('highlighted faded');
    
    // Highlight selected node and its connections
    node.addClass('highlighted');
    node.connectedEdges().addClass('highlighted');
    node.connectedEdges().connectedNodes().addClass('highlighted');
    
    // Fade other elements
    cy.elements().difference(node.closedNeighborhood()).addClass('faded');
  }

  function clearSelection() {
    selectedNode = null;
    cy.elements().removeClass('highlighted faded');
  }

  function filterGraph() {
    if (!cy) return;

    cy.nodes().forEach(node => {
      const nodeType = node.data('type');
      const nodeLabel = node.data('label').toLowerCase();
      
      const typeMatch = filterByType === 'all' || nodeType === filterByType;
      const searchMatch = !searchGraph || nodeLabel.includes(searchGraph.toLowerCase());
      
      if (typeMatch && searchMatch) {
        node.show();
      } else {
        node.hide();
      }
    });

    // Hide edges connected to hidden nodes
    cy.edges().forEach(edge => {
      const source = edge.source();
      const target = edge.target();
      
      if (source.visible() && target.visible()) {
        edge.show();
      } else {
        edge.hide();
      }
    });
  }

  function centerGraph() {
    if (cy) {
      cy.fit();
      cy.center();
    }
  }

  function exportGraph() {
    if (!cy) return;

    const graphData = {
      nodes: cy.nodes().map(node => ({
        id: node.id(),
        label: node.data('label'),
        type: node.data('type'),
        position: node.position(),
        properties: node.data('properties')
      })),
      edges: cy.edges().map(edge => ({
        source: edge.source().id(),
        target: edge.target().id(),
        relationship: edge.data('relationship'),
        weight: edge.data('weight')
      })),
      metadata: {
        layout: layoutType,
        nodeSize,
        colorScheme,
        exportTimestamp: new Date().toISOString(),
        stats: graphStats
      }
    };

    const blob = new Blob([JSON.stringify(graphData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `knowledge-graph-export-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function showNodeTooltip(node) {
    // Could implement tooltip functionality here
    console.log('Node hover:', node.data());
  }

  function updateGraphStats(graphData) {
    graphStats = {
      totalNodes: graphData.nodes?.length || 0,
      totalEdges: graphData.edges?.length || 0,
      clusters: graphData.clusters?.length || 0,
      density: graphData.density || 0
    };
  }

  // Reactive updates
  $: if (cy && (layoutType || showLabels || showEdgeLabels || nodeSize || colorScheme)) {
    updateGraphAppearance();
  }

  $: if (filterByType !== 'all' || searchGraph) {
    filterGraph();
  }

  function updateGraphAppearance() {
    if (!cy) return;

    // Update node labels
    cy.style()
      .selector('node')
      .style('label', showLabels ? 'data(label)' : '')
      .update();

    // Update edge labels  
    cy.style()
      .selector('edge')
      .style('label', showEdgeLabels ? 'data(relationship)' : '')
      .update();

    // Update node sizes
    cy.nodes().forEach(node => {
      const newSize = calculateNodeSize({
        type: node.data('type'),
        connections: node.connectedEdges(),
        properties: node.data('properties')
      });
      node.data('size', newSize);
    });

    // Apply new layout if changed
    if (layoutType) {
      applyLayout();
    }
  }
</script>

<div class="knowledge-graph h-full flex flex-col">
  <!-- Graph Controls -->
  <Card variant="bordered" class="flex-shrink-0">
    <div class="p-4">
      <div class="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
        <div class="flex items-center gap-4">
          <Button variant="outline" size="sm" on:click={() => dispatch('backToSearch')}>
            ← Back to Search
          </Button>
          <h2 class="text-lg font-semibold text-dark-text-primary">Knowledge Graph</h2>
          <div class="flex items-center gap-2 text-sm">
            <Badge variant="info" size="sm">{graphStats.totalNodes} nodes</Badge>
            <Badge variant="info" size="sm">{graphStats.totalEdges} edges</Badge>
            <Badge variant="info" size="sm">{graphStats.clusters} clusters</Badge>
          </div>
        </div>
        
        <div class="flex items-center gap-3">
          <!-- Graph Search -->
          <div class="relative">
            <input
              type="text"
              bind:value={searchGraph}
              placeholder="Filter nodes..."
              class="px-3 py-2 pr-8 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary placeholder-dark-text-tertiary text-sm focus:border-green-500 focus:outline-none"
            />
            {#if searchGraph}
              <button 
                class="absolute right-2 top-1/2 transform -translate-y-1/2 text-dark-text-tertiary hover:text-dark-text-primary"
                on:click={() => searchGraph = ''}
              >
                ×
              </button>
            {/if}
          </div>
          
          <!-- Type Filter -->
          <select 
            bind:value={filterByType}
            class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-sm focus:border-green-500 focus:outline-none"
          >
            <option value="all">All Types</option>
            <option value="document">Documents</option>
            <option value="entity">Entities</option>
            <option value="concept">Concepts</option>
            <option value="relationship">Relationships</option>
          </select>
          
          <Button variant="outline" size="sm" on:click={exportGraph}>
            Export
          </Button>
        </div>
      </div>
    </div>
  </Card>

  <!-- Graph Visualization -->
  <div class="flex-1 flex">
    <!-- Main Graph -->
    <div class="flex-1 relative">
      <div 
        bind:this={graphContainer}
        class="w-full h-full bg-dark-bg-primary border border-dark-border"
      ></div>
      
      <!-- Graph Controls Overlay -->
      <div class="absolute top-4 right-4 space-y-2">
        <Card variant="bordered" class="p-3 bg-dark-bg-secondary/90 backdrop-blur">
          <div class="space-y-3">
            <!-- Layout Selection -->
            <div>
              <label class="block text-xs text-dark-text-tertiary mb-1">Layout</label>
              <select 
                bind:value={layoutType}
                class="w-full px-2 py-1 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-xs focus:border-green-500 focus:outline-none"
              >
                {#each layouts as layout}
                  <option value={layout.value}>{layout.name}</option>
                {/each}
              </select>
            </div>

            <!-- Node Size -->
            <div>
              <label class="block text-xs text-dark-text-tertiary mb-1">Node Size</label>
              <select 
                bind:value={nodeSize}
                class="w-full px-2 py-1 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-xs focus:border-green-500 focus:outline-none"
              >
                <option value="uniform">Uniform</option>
                <option value="degree">By Degree</option>
                <option value="centrality">By Centrality</option>
                <option value="similarity">By Similarity</option>
              </select>
            </div>

            <!-- Display Options -->
            <div class="space-y-1">
              <label class="flex items-center gap-2 text-xs">
                <input type="checkbox" bind:checked={showLabels} class="checkbox" />
                <span class="text-dark-text-secondary">Node Labels</span>
              </label>
              <label class="flex items-center gap-2 text-xs">
                <input type="checkbox" bind:checked={showEdgeLabels} class="checkbox" />
                <span class="text-dark-text-secondary">Edge Labels</span>
              </label>
            </div>

            <!-- Graph Actions -->
            <div class="pt-2 border-t border-dark-border space-y-1">
              <Button variant="outline" size="xs" fullWidth on:click={centerGraph}>
                <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5v-4m0 4h-4m4-4l-5-5" />
                </svg>
                Center
              </Button>
              <Button variant="outline" size="xs" fullWidth on:click={() => {
                if (cy) {
                  cy.fit();
                }
              }}>
                <svg class="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Fit All
              </Button>
              <Button variant="outline" size="xs" fullWidth on:click={clearSelection}>
                Clear Selection
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>

    <!-- Node Details Sidebar -->
    {#if selectedNode}
      <div class="w-80 border-l border-dark-border bg-dark-bg-secondary flex-shrink-0">
        <div class="p-4">
          <div class="flex items-center justify-between mb-4">
            <h3 class="text-md font-medium text-dark-text-primary">Node Details</h3>
            <button 
              class="text-dark-text-tertiary hover:text-dark-text-primary"
              on:click={clearSelection}
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div class="space-y-4">
            <!-- Node Info -->
            <div class="node-info p-3 bg-dark-bg-tertiary rounded border border-dark-border">
              <div class="flex items-center gap-2 mb-2">
                <span class="text-lg">{nodeTypes[selectedNode.data('type')]?.icon || '🔍'}</span>
                <span class="text-sm font-medium text-dark-text-primary">
                  {selectedNode.data('label')}
                </span>
              </div>
              
              <div class="space-y-1 text-xs">
                <div class="flex justify-between">
                  <span class="text-dark-text-tertiary">Type:</span>
                  <span class="text-dark-text-secondary">{selectedNode.data('type')}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-dark-text-tertiary">Connections:</span>
                  <span class="text-dark-text-secondary">{selectedNode.degree()}</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-dark-text-tertiary">ID:</span>
                  <span class="text-dark-text-secondary font-mono">{selectedNode.id()}</span>
                </div>
              </div>
            </div>

            <!-- Node Properties -->
            {#if selectedNode.data('properties') && Object.keys(selectedNode.data('properties')).length > 0}
              <div class="node-properties">
                <h5 class="text-sm font-medium text-dark-text-primary mb-2">Properties</h5>
                <div class="space-y-1">
                  {#each Object.entries(selectedNode.data('properties')) as [key, value]}
                    <div class="flex justify-between text-xs">
                      <span class="text-dark-text-tertiary">{key}:</span>
                      <span class="text-dark-text-secondary">{value}</span>
                    </div>
                  {/each}
                </div>
              </div>
            {/if}

            <!-- Connected Nodes -->
            <div class="connected-nodes">
              <h5 class="text-sm font-medium text-dark-text-primary mb-2">Connected Nodes</h5>
              <div class="space-y-1 max-h-48 overflow-y-auto">
                {#each selectedNode.connectedEdges().connectedNodes() as connectedNode}
                  <button 
                    class="w-full text-left p-2 text-xs bg-dark-bg-primary rounded border border-dark-border hover:border-green-500 transition-colors"
                    on:click={() => {
                      selectNode(connectedNode);
                      highlightConnections(connectedNode);
                    }}
                  >
                    <div class="flex items-center gap-2">
                      <span>{nodeTypes[connectedNode.data('type')]?.icon || '🔍'}</span>
                      <span class="text-dark-text-primary truncate">{connectedNode.data('label')}</span>
                    </div>
                  </button>
                {/each}
              </div>
            </div>

            <!-- Actions -->
            <div class="node-actions space-y-2">
              <Button variant="outline" size="sm" fullWidth on:click={() => {
                dispatch('nodeSelected', {
                  id: selectedNode.id(),
                  label: selectedNode.data('label'),
                  type: selectedNode.data('type')
                });
                dispatch('backToSearch');
              }}>
                Search Similar
              </Button>
              
              <Button variant="outline" size="sm" fullWidth on:click={() => {
                const nodeData = {
                  id: selectedNode.id(),
                  label: selectedNode.data('label'),
                  type: selectedNode.data('type'),
                  properties: selectedNode.data('properties')
                };
                navigator.clipboard.writeText(JSON.stringify(nodeData, null, 2));
              }}>
                Copy Node Data
              </Button>
            </div>
          </div>
        </div>
      </div>
    {/if}
  </div>

  <!-- Graph Stats Footer -->
  <div class="flex-shrink-0 bg-dark-bg-secondary border-t border-dark-border p-3">
    <div class="flex items-center justify-between text-sm">
      <div class="flex items-center gap-4">
        <span class="text-dark-text-tertiary">
          Graph Density: <span class="text-dark-text-secondary">{(graphStats.density * 100).toFixed(1)}%</span>
        </span>
        <span class="text-dark-text-tertiary">
          Vector DB: <span class="text-green-400">{selectedVectorDB}</span>
        </span>
      </div>
      
      <div class="flex items-center gap-2">
        <div class="flex items-center gap-1">
          {#each Object.entries(nodeTypes) as [type, info]}
            <div class="flex items-center gap-1">
              <div 
                class="w-3 h-3 rounded-full"
                style="background-color: {info.color}"
              ></div>
              <span class="text-xs text-dark-text-tertiary">{type}</span>
            </div>
          {/each}
        </div>
      </div>
    </div>
  </div>
</div>

<style>
  .checkbox {
    @apply w-3 h-3 rounded border border-dark-border;
    background: var(--dark-bg-tertiary, #0f0f0f);
    accent-color: #00ff41;
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar {
    width: 6px;
  }
  
  ::-webkit-scrollbar-track {
    background: var(--dark-bg-tertiary, #0f0f0f);
  }
  
  ::-webkit-scrollbar-thumb {
    background: var(--dark-border, #00ff4133);
    border-radius: 3px;
  }
</style>