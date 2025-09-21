<script lang="ts">
  import { onMount } from 'svelte';
  import cytoscape from 'cytoscape';
  import coseBilkent from 'cytoscape-cose-bilkent';
  import edgehandles from 'cytoscape-edgehandles';
  import { writable } from 'svelte/store';
  
  cytoscape.use(coseBilkent);
  cytoscape.use(edgehandles);
  
  let cy;
  let eh; // edge handles
  let selectedNode = null;
  let workflowName = '';
  let workflowStatus = 'idle';
  let executionLog = [];
  let agentTemplates = [
    { id: 'osint', name: 'OSINT Collector', icon: '🔍', color: '#00ff00' },
    { id: 'analyzer', name: 'Data Analyzer', icon: '📊', color: '#ffff00' },
    { id: 'scraper', name: 'Web Scraper', icon: '🕷️', color: '#ff00ff' },
    { id: 'monitor', name: 'Monitor Agent', icon: '👁️', color: '#00ffff' },
    { id: 'alert', name: 'Alert System', icon: '🚨', color: '#ff0000' },
    { id: 'transform', name: 'Data Transform', icon: '⚙️', color: '#ffffff' },
    { id: 'storage', name: 'Storage Node', icon: '💾', color: '#808080' },
    { id: 'api', name: 'API Connector', icon: '🔌', color: '#ffa500' }
  ];
  
  let nodeIdCounter = 1;
  let draggedTemplate = null;
  let executionTimeline = [];
  let performanceMetrics = {
    totalNodes: 0,
    activeNodes: 0,
    completedTasks: 0,
    avgExecutionTime: 0,
    bottlenecks: []
  };
  
  onMount(() => {
    initializeCytoscape();
    setupDragAndDrop();
    initializeEdgeHandles();
    
    return () => {
      if (cy) cy.destroy();
    };
  });
  
  function initializeCytoscape() {
    cy = cytoscape({
      container: document.getElementById('cy'),
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'label': 'data(label)',
            'color': '#fff',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 80,
            'height': 80,
            'font-size': '12px',
            'border-width': 2,
            'border-color': '#333',
            'text-wrap': 'wrap',
            'text-max-width': '70px'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 3,
            'line-color': '#666',
            'target-arrow-color': '#666',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(label)',
            'font-size': '10px',
            'color': '#888',
            'text-background-color': '#1a1a1a',
            'text-background-opacity': 0.8,
            'text-background-padding': '3px'
          }
        },        {
          selector: '.running',
          style: {
            'background-color': '#00ff00',
            'border-color': '#00ff00',
            'border-width': 4,
            'box-shadow': '0 0 20px #00ff00'
          }
        },
        {
          selector: '.completed',
          style: {
            'background-color': '#4a4a4a',
            'border-color': '#00ff00',
            'opacity': 0.7
          }
        },
        {
          selector: '.error',
          style: {
            'background-color': '#ff0000',
            'border-color': '#ff0000',
            'box-shadow': '0 0 20px #ff0000'
          }
        },
        {
          selector: ':selected',
          style: {
            'border-width': 4,
            'border-color': '#00ffff'
          }
        }
      ],
      layout: {
        name: 'cose-bilkent',
        animate: true,
        animationDuration: 500,
        fit: true,
        padding: 50,
        nodeRepulsion: 8000,
        idealEdgeLength: 100
      },
      minZoom: 0.5,
      maxZoom: 3,
      wheelSensitivity: 0.2
    });
    
    // Event handlers
    cy.on('tap', 'node', function(evt) {
      selectedNode = evt.target;
      updateNodeDetails(selectedNode.data());
    });
    
    cy.on('tap', function(evt) {
      if (evt.target === cy) {
        selectedNode = null;
      }
    });
  }
  
  function initializeEdgeHandles() {
    eh = cy.edgehandles({
      canConnect: function(sourceNode, targetNode) {
        // Prevent self-loops and duplicate edges
        return sourceNode.id() !== targetNode.id() && 
               !sourceNode.edgesWith(targetNode).length;
      },
      edgeParams: function(sourceNode, targetNode) {
        return {
          data: {
            source: sourceNode.id(),
            target: targetNode.id(),
            label: 'data flow'
          }
        };
      },
      hoverDelay: 150,
      snap: true,
      snapThreshold: 50,
      snapFrequency: 15,
      noEdgeEventsInDraw: true,
      disableBrowserGestures: true
    });
  }  
  function setupDragAndDrop() {
    // Make template cards draggable
    const templates = document.querySelectorAll('.agent-template');
    const canvas = document.getElementById('cy');
    
    templates.forEach(template => {
      template.addEventListener('dragstart', (e) => {
        draggedTemplate = agentTemplates.find(t => t.id === template.dataset.agentId);
        e.dataTransfer.effectAllowed = 'copy';
      });
    });
    
    canvas.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
    });
    
    canvas.addEventListener('drop', (e) => {
      e.preventDefault();
      if (!draggedTemplate) return;
      
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const node = {
        group: 'nodes',
        data: {
          id: `node_${nodeIdCounter++}`,
          label: draggedTemplate.name,
          type: draggedTemplate.id,
          color: draggedTemplate.color,
          icon: draggedTemplate.icon,
          status: 'idle',
          config: {}
        },
        position: cy.viewport({ x, y })
      };
      
      cy.add(node);
      updatePerformanceMetrics();
      draggedTemplate = null;
    });
  }  
  function executeWorkflow() {
    if (!cy.nodes().length) {
      alert('Add nodes to the workflow first!');
      return;
    }
    
    workflowStatus = 'running';
    executionLog = [`[${new Date().toISOString()}] Starting workflow execution...`];
    executionTimeline = [];
    
    // Find start nodes (nodes with no incoming edges)
    const startNodes = cy.nodes().filter(node => !node.incomers('edge').length);
    
    if (!startNodes.length) {
      executionLog.push('[ERROR] No start nodes found in workflow');
      workflowStatus = 'error';
      return;
    }
    
    // Begin execution simulation
    simulateExecution(startNodes);
  }
  
  async function simulateExecution(nodes) {
    for (let node of nodes) {
      node.addClass('running');
      const startTime = Date.now();
      
      executionLog.push(`[${new Date().toISOString()}] Executing ${node.data('label')}...`);
      executionTimeline.push({
        nodeId: node.id(),
        status: 'running',
        timestamp: startTime
      });
      
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, Math.random() * 2000 + 1000));
      
      node.removeClass('running').addClass('completed');
      const endTime = Date.now();      
      executionLog.push(`[${new Date().toISOString()}] ✓ ${node.data('label')} completed (${endTime - startTime}ms)`);
      executionTimeline.push({
        nodeId: node.id(),
        status: 'completed',
        timestamp: endTime,
        duration: endTime - startTime
      });
      
      // Execute downstream nodes
      const downstreamNodes = node.outgoers('node');
      if (downstreamNodes.length) {
        await simulateExecution(downstreamNodes);
      }
    }
    
    if (cy.nodes('.completed').length === cy.nodes().length) {
      workflowStatus = 'completed';
      executionLog.push(`[${new Date().toISOString()}] Workflow completed successfully!`);
      updatePerformanceMetrics();
    }
  }
  
  function updatePerformanceMetrics() {
    performanceMetrics = {
      totalNodes: cy.nodes().length,
      activeNodes: cy.nodes('.running').length,
      completedTasks: cy.nodes('.completed').length,
      avgExecutionTime: calculateAvgExecutionTime(),
      bottlenecks: identifyBottlenecks()
    };
  }
  
  function calculateAvgExecutionTime() {
    if (!executionTimeline.length) return 0;
    const completedTasks = executionTimeline.filter(t => t.status === 'completed');
    if (!completedTasks.length) return 0;
    
    const totalTime = completedTasks.reduce((sum, task) => sum + (task.duration || 0), 0);
    return Math.round(totalTime / completedTasks.length);
  }  
  function identifyBottlenecks() {
    const bottlenecks = [];
    const taskDurations = executionTimeline
      .filter(t => t.status === 'completed' && t.duration)
      .sort((a, b) => b.duration - a.duration)
      .slice(0, 3);
    
    taskDurations.forEach(task => {
      const node = cy.getElementById(task.nodeId);
      if (node) {
        bottlenecks.push({
          nodeId: task.nodeId,
          label: node.data('label'),
          duration: task.duration
        });
      }
    });
    
    return bottlenecks;
  }
  
  function updateNodeDetails(nodeData) {
    // Update the selected node's configuration panel
    console.log('Selected node:', nodeData);
  }
  
  function saveWorkflow() {
    const workflowData = {
      name: workflowName,
      nodes: cy.nodes().map(n => n.data()),
      edges: cy.edges().map(e => e.data()),
      timestamp: Date.now()
    };
    
    localStorage.setItem(`workflow_${Date.now()}`, JSON.stringify(workflowData));
    alert('Workflow saved successfully!');
  }
  
  function clearWorkflow() {
    if (confirm('Clear all nodes and edges?')) {
      cy.elements().remove();
      executionLog = [];
      executionTimeline = [];
      updatePerformanceMetrics();
    }
  }
</script>
<style>
  .workflow-container {
    @apply h-screen flex flex-col bg-gray-900 text-gray-100;
  }
  
  .toolbar {
    @apply flex items-center justify-between p-4 bg-gray-800 border-b border-gray-700;
  }
  
  .main-layout {
    @apply flex flex-1 overflow-hidden;
  }
  
  .sidebar {
    @apply w-64 bg-gray-800 p-4 overflow-y-auto border-r border-gray-700;
  }
  
  .canvas-container {
    @apply flex-1 relative bg-gray-950;
  }
  
  .agent-template {
    @apply p-3 mb-3 bg-gray-700 rounded-lg cursor-move hover:bg-gray-600 transition-colors;
  }
  
  .execution-panel {
    @apply w-80 bg-gray-800 p-4 overflow-y-auto border-l border-gray-700;
  }
  
  .metric-card {
    @apply p-3 mb-3 bg-gray-700 rounded-lg;
  }
</style>

<div class="workflow-container">
  <!-- Toolbar -->
  <div class="toolbar">
    <input 
      type="text" 
      bind:value={workflowName}
      placeholder="Workflow Name"
      class="px-4 py-2 bg-gray-700 rounded-lg border border-gray-600"
    />
    
    <div class="flex gap-2">
      <button 
        on:click={executeWorkflow}
        class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
        disabled={workflowStatus === 'running'}>
        {workflowStatus === 'running' ? 'Executing...' : 'Execute'}
      </button>
      <button 
        on:click={saveWorkflow}
        class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
        Save
      </button>
      <button 
        on:click={clearWorkflow}
        class="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700">
        Clear
      </button>
    </div>
  </div>  
  <!-- Main Layout -->
  <div class="main-layout">
    <!-- Agent Templates Sidebar -->
    <div class="sidebar">
      <h3 class="text-lg font-semibold mb-4 text-cyan-400">Agent Templates</h3>
      {#each agentTemplates as template}
        <div 
          class="agent-template"
          draggable="true"
          data-agent-id={template.id}>
          <div class="flex items-center gap-2">
            <span class="text-2xl">{template.icon}</span>
            <span class="text-sm font-medium">{template.name}</span>
          </div>
        </div>
      {/each}
    </div>
    
    <!-- Canvas -->
    <div class="canvas-container">
      <div id="cy" class="w-full h-full"></div>
    </div>
    
    <!-- Execution Panel -->
    <div class="execution-panel">
      <h3 class="text-lg font-semibold mb-4 text-cyan-400">Execution Details</h3>
      
      <!-- Performance Metrics -->
      <div class="metric-card">
        <div class="text-sm text-gray-400 mb-1">Total Nodes</div>
        <div class="text-xl font-bold text-white">{performanceMetrics.totalNodes}</div>
      </div>
      
      <div class="metric-card">
        <div class="text-sm text-gray-400 mb-1">Active Nodes</div>
        <div class="text-xl font-bold text-green-400">{performanceMetrics.activeNodes}</div>
      </div>
      
      <div class="metric-card">
        <div class="text-sm text-gray-400 mb-1">Avg Execution Time</div>
        <div class="text-xl font-bold text-yellow-400">{performanceMetrics.avgExecutionTime}ms</div>
      </div>      
      <!-- Bottlenecks -->
      {#if performanceMetrics.bottlenecks.length > 0}
        <div class="metric-card">
          <div class="text-sm text-gray-400 mb-2">Bottlenecks</div>
          {#each performanceMetrics.bottlenecks as bottleneck}
            <div class="text-xs text-red-400 mb-1">
              {bottleneck.label}: {bottleneck.duration}ms
            </div>
          {/each}
        </div>
      {/if}
      
      <!-- Execution Log -->
      <h4 class="text-md font-semibold mt-6 mb-2 text-gray-300">Execution Log</h4>
      <div class="bg-gray-900 rounded p-3 h-64 overflow-y-auto font-mono text-xs">
        {#each executionLog as log}
          <div class="mb-1 text-gray-400">{log}</div>
        {/each}
      </div>
    </div>
  </div>
</div>