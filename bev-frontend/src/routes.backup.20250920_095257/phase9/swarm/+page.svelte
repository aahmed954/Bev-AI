<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // Swarm coordination state
  const swarmState = writable({
    topology: {
      total_nodes: 156,
      active_nodes: 134,
      consensus_nodes: 89,
      leader_nodes: 12,
      follower_nodes: 122,
      network_health: 94.7,
      latency_avg: 23.4,
      throughput: '2.8GB/s'
    },
    consensus: {
      algorithm: 'PBFT',
      current_round: 45678,
      participation_rate: 91.3,
      finality_time: 2.7,
      byzantine_tolerance: 33,
      consensus_efficiency: 89.2
    },
    task_allocation: {
      total_tasks: 1847,
      completed_tasks: 1623,
      active_tasks: 187,
      queued_tasks: 37,
      success_rate: 97.8,
      load_balancing_score: 92.1
    },
    communication: {
      message_rate: '1.2M msg/sec',
      bandwidth_usage: 68.4,
      protocol_efficiency: 94.2,
      error_rate: 0.003,
      retransmission_rate: 0.012
    },
    performance: {
      computational_efficiency: 87.6,
      resource_utilization: 73.2,
      energy_efficiency: 82.9,
      scalability_index: 91.4,
      fault_tolerance: 95.1
    }
  });

  // 3D Visualization data
  let nodePositions = [];
  let connections = [];
  let selectedNode = null;
  let viewMode = '3d';
  let animationEnabled = true;

  // Swarm controls
  let swarmConfiguration = {
    consensus_algorithm: 'PBFT',
    max_nodes: 200,
    fault_tolerance: 33,
    task_scheduling: 'round_robin',
    load_balancing: true,
    auto_scaling: true
  };

  // Task management
  let newTask = {
    type: 'computation',
    priority: 'normal',
    requirements: [],
    deadline: '',
    data_size: 0
  };

  // WebSocket connections
  let swarmWs: WebSocket | null = null;
  let topologyWs: WebSocket | null = null;
  let performanceWs: WebSocket | null = null;

  // 3D Scene variables
  let canvas: HTMLCanvasElement;
  let gl: WebGLRenderingContext;
  let scene = {
    rotation: { x: 0, y: 0 },
    zoom: 1.0,
    nodeScale: 1.0
  };

  onMount(() => {
    initializeWebSockets();
    initializeVisualization();
    generateSwarmTopology();
    startPerformanceMonitoring();
  });

  onDestroy(() => {
    if (swarmWs) swarmWs.close();
    if (topologyWs) topologyWs.close();
    if (performanceWs) performanceWs.close();
  });

  function initializeWebSockets() {
    // Main swarm coordinator WebSocket
    swarmWs = new WebSocket('ws://localhost:8017/swarm');
    swarmWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      swarmState.update(state => ({
        ...state,
        ...data
      }));
      updateVisualization(data);
    };

    // Topology monitoring WebSocket
    topologyWs = new WebSocket('ws://localhost:8018/topology');
    topologyWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      updateTopology(data);
    };

    // Performance metrics WebSocket
    performanceWs = new WebSocket('ws://localhost:8019/performance');
    performanceWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      swarmState.update(state => ({
        ...state,
        performance: { ...state.performance, ...data }
      }));
    };
  }

  function initializeVisualization() {
    if (!canvas) return;

    gl = canvas.getContext('webgl');
    if (!gl) {
      console.error('WebGL not supported');
      return;
    }

    // Initialize WebGL context and shaders
    gl.clearColor(0.1, 0.1, 0.15, 1.0);
    gl.enable(gl.DEPTH_TEST);

    // Start animation loop
    animate();
  }

  function generateSwarmTopology() {
    // Generate random 3D positions for nodes
    nodePositions = [];
    connections = [];

    const nodeCount = $swarmState.topology.total_nodes;

    for (let i = 0; i < nodeCount; i++) {
      nodePositions.push({
        id: i,
        x: (Math.random() - 0.5) * 10,
        y: (Math.random() - 0.5) * 10,
        z: (Math.random() - 0.5) * 10,
        type: Math.random() > 0.8 ? 'leader' : 'follower',
        status: Math.random() > 0.1 ? 'active' : 'inactive',
        load: Math.random(),
        connections: []
      });
    }

    // Generate connections between nearby nodes
    for (let i = 0; i < nodeCount; i++) {
      const node = nodePositions[i];
      const connectionCount = Math.floor(Math.random() * 5) + 2;

      for (let j = 0; j < connectionCount; j++) {
        const targetIndex = Math.floor(Math.random() * nodeCount);
        if (targetIndex !== i && !node.connections.includes(targetIndex)) {
          node.connections.push(targetIndex);
          connections.push({
            source: i,
            target: targetIndex,
            strength: Math.random(),
            type: 'data'
          });
        }
      }
    }
  }

  function updateVisualization(data: any) {
    if (data.topology) {
      // Update node statuses based on real-time data
      nodePositions.forEach(node => {
        if (data.topology.node_updates && data.topology.node_updates[node.id]) {
          const update = data.topology.node_updates[node.id];
          node.status = update.status;
          node.load = update.load;
          node.type = update.type;
        }
      });
    }
  }

  function updateTopology(data: any) {
    if (data.new_nodes) {
      // Add new nodes to the visualization
      data.new_nodes.forEach(nodeData => {
        nodePositions.push({
          id: nodeData.id,
          x: nodeData.position.x,
          y: nodeData.position.y,
          z: nodeData.position.z,
          type: nodeData.type,
          status: nodeData.status,
          load: nodeData.load,
          connections: nodeData.connections || []
        });
      });
    }

    if (data.removed_nodes) {
      // Remove nodes from visualization
      nodePositions = nodePositions.filter(node =>
        !data.removed_nodes.includes(node.id)
      );
    }
  }

  function animate() {
    if (!gl || !canvas) return;

    // Clear canvas
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    if (animationEnabled) {
      scene.rotation.y += 0.005;

      // Animate node positions slightly for organic feel
      nodePositions.forEach(node => {
        node.x += (Math.random() - 0.5) * 0.01;
        node.y += (Math.random() - 0.5) * 0.01;
        node.z += (Math.random() - 0.5) * 0.01;
      });
    }

    // Render nodes and connections
    renderSwarmVisualization();

    requestAnimationFrame(animate);
  }

  function renderSwarmVisualization() {
    // This would implement actual WebGL rendering
    // For now, we'll update a 2D canvas representation
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw connections
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    connections.forEach(conn => {
      const source = nodePositions[conn.source];
      const target = nodePositions[conn.target];
      if (source && target) {
        ctx.beginPath();
        ctx.moveTo(
          (source.x + 5) * 30 + canvas.width / 2,
          (source.y + 5) * 30 + canvas.height / 2
        );
        ctx.lineTo(
          (target.x + 5) * 30 + canvas.width / 2,
          (target.y + 5) * 30 + canvas.height / 2
        );
        ctx.stroke();
      }
    });

    // Draw nodes
    nodePositions.forEach(node => {
      const x = (node.x + 5) * 30 + canvas.width / 2;
      const y = (node.y + 5) * 30 + canvas.height / 2;

      ctx.beginPath();
      ctx.arc(x, y, node.type === 'leader' ? 8 : 5, 0, 2 * Math.PI);

      if (node.status === 'active') {
        ctx.fillStyle = node.type === 'leader' ? '#10B981' : '#3B82F6';
      } else {
        ctx.fillStyle = '#6B7280';
      }

      ctx.fill();

      // Draw load indicator
      if (node.status === 'active') {
        ctx.beginPath();
        ctx.arc(x, y, (node.type === 'leader' ? 8 : 5) * node.load, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.fill();
      }
    });
  }

  async function startPerformanceMonitoring() {
    setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8017/api/metrics');
        const metrics = await response.json();
        swarmState.update(state => ({
          ...state,
          ...metrics
        }));
      } catch (error) {
        console.error('Performance monitoring error:', error);
      }
    }, 5000);
  }

  async function updateSwarmConfiguration() {
    try {
      const response = await fetch('http://localhost:8017/api/configure', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(swarmConfiguration)
      });

      if (response.ok) {
        console.log('Swarm configuration updated');
      }
    } catch (error) {
      console.error('Configuration update failed:', error);
    }
  }

  async function allocateTask() {
    try {
      const response = await fetch('http://localhost:8017/api/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTask)
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Task allocated:', result);

        // Reset form
        newTask = {
          type: 'computation',
          priority: 'normal',
          requirements: [],
          deadline: '',
          data_size: 0
        };
      }
    } catch (error) {
      console.error('Task allocation failed:', error);
    }
  }

  function handleCanvasClick(event: MouseEvent) {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Find clicked node
    const clickedNode = nodePositions.find(node => {
      const nodeX = (node.x + 5) * 30 + canvas.width / 2;
      const nodeY = (node.y + 5) * 30 + canvas.height / 2;
      const distance = Math.sqrt((x - nodeX) ** 2 + (y - nodeY) ** 2);
      return distance < (node.type === 'leader' ? 8 : 5);
    });

    selectedNode = clickedNode || null;
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'syncing': return 'text-blue-400';
      case 'consensus': return 'text-yellow-400';
      case 'leader': return 'text-purple-400';
      default: return 'text-gray-400';
    }
  }
</script>

<svelte:head>
  <title>Phase 9 - Swarm Coordination Center | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
        Phase 9 - Swarm Coordination Center
      </h1>
      <p class="text-gray-300">Distributed swarm intelligence with consensus mechanisms</p>
    </div>

    {#if $swarmState}
      <!-- Swarm Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-5 gap-6 mb-8">
        <!-- Topology -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Swarm Topology
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Total Nodes:</span>
              <span class="text-blue-400">{$swarmState.topology.total_nodes}</span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-green-400">{$swarmState.topology.active_nodes}</span>
            </div>
            <div class="flex justify-between">
              <span>Leaders:</span>
              <span class="text-purple-400">{$swarmState.topology.leader_nodes}</span>
            </div>
            <div class="flex justify-between">
              <span>Health:</span>
              <span class="text-green-400">{$swarmState.topology.network_health}%</span>
            </div>
          </div>
        </div>

        <!-- Consensus -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-yellow-400 mr-2"></span>
            Consensus Engine
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Algorithm:</span>
              <span class="text-yellow-400">{$swarmState.consensus.algorithm}</span>
            </div>
            <div class="flex justify-between">
              <span>Round:</span>
              <span class="text-blue-400">{$swarmState.consensus.current_round.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Participation:</span>
              <span class="text-green-400">{$swarmState.consensus.participation_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Finality:</span>
              <span class="text-cyan-400">{$swarmState.consensus.finality_time}s</span>
            </div>
          </div>
        </div>

        <!-- Task Allocation -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Task Allocation
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Total Tasks:</span>
              <span class="text-blue-400">{$swarmState.task_allocation.total_tasks.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-yellow-400">{$swarmState.task_allocation.active_tasks}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class="text-green-400">{$swarmState.task_allocation.success_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Load Balance:</span>
              <span class="text-purple-400">{$swarmState.task_allocation.load_balancing_score}%</span>
            </div>
          </div>
        </div>

        <!-- Communication -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-cyan-400 mr-2"></span>
            Communication
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Message Rate:</span>
              <span class="text-cyan-400">{$swarmState.communication.message_rate}</span>
            </div>
            <div class="flex justify-between">
              <span>Bandwidth:</span>
              <span class="text-blue-400">{$swarmState.communication.bandwidth_usage}%</span>
            </div>
            <div class="flex justify-between">
              <span>Efficiency:</span>
              <span class="text-green-400">{$swarmState.communication.protocol_efficiency}%</span>
            </div>
            <div class="flex justify-between">
              <span>Error Rate:</span>
              <span class="text-yellow-400">{$swarmState.communication.error_rate}%</span>
            </div>
          </div>
        </div>

        <!-- Performance -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Performance
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Efficiency:</span>
              <span class="text-purple-400">{$swarmState.performance.computational_efficiency}%</span>
            </div>
            <div class="flex justify-between">
              <span>Utilization:</span>
              <span class="text-blue-400">{$swarmState.performance.resource_utilization}%</span>
            </div>
            <div class="flex justify-between">
              <span>Scalability:</span>
              <span class="text-green-400">{$swarmState.performance.scalability_index}%</span>
            </div>
            <div class="flex justify-between">
              <span>Fault Tolerance:</span>
              <span class="text-yellow-400">{$swarmState.performance.fault_tolerance}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Main Content Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <!-- 3D Swarm Visualization -->
        <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
          <div class="flex justify-between items-center mb-4">
            <h3 class="text-lg font-semibold">Swarm Topology Visualization</h3>
            <div class="flex space-x-2">
              <button
                on:click={() => viewMode = '3d'}
                class="px-3 py-1 rounded text-sm {viewMode === '3d' ? 'bg-blue-600' : 'bg-gray-600'}"
              >
                3D View
              </button>
              <button
                on:click={() => viewMode = '2d'}
                class="px-3 py-1 rounded text-sm {viewMode === '2d' ? 'bg-blue-600' : 'bg-gray-600'}"
              >
                2D View
              </button>
              <button
                on:click={() => animationEnabled = !animationEnabled}
                class="px-3 py-1 rounded text-sm {animationEnabled ? 'bg-green-600' : 'bg-gray-600'}"
              >
                {animationEnabled ? 'Pause' : 'Play'}
              </button>
            </div>
          </div>

          <canvas
            bind:this={canvas}
            on:click={handleCanvasClick}
            width="800"
            height="500"
            class="w-full h-96 bg-gray-900 rounded border border-gray-600 cursor-pointer"
          ></canvas>

          <div class="flex justify-between items-center mt-4">
            <div class="flex space-x-4 text-sm">
              <div class="flex items-center">
                <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
                <span>Leader Nodes</span>
              </div>
              <div class="flex items-center">
                <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
                <span>Follower Nodes</span>
              </div>
              <div class="flex items-center">
                <span class="w-3 h-3 rounded-full bg-gray-400 mr-2"></span>
                <span>Inactive</span>
              </div>
            </div>
            <div class="text-sm text-gray-400">
              Click nodes for details | Zoom: {scene.zoom.toFixed(1)}x
            </div>
          </div>
        </div>

        <!-- Node Details & Controls -->
        <div class="space-y-6">
          <!-- Selected Node Details -->
          {#if selectedNode}
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <h3 class="text-lg font-semibold mb-4">Node Details</h3>
              <div class="space-y-2 text-sm">
                <div class="flex justify-between">
                  <span>Node ID:</span>
                  <span class="text-blue-400">{selectedNode.id}</span>
                </div>
                <div class="flex justify-between">
                  <span>Type:</span>
                  <span class="text-purple-400">{selectedNode.type.toUpperCase()}</span>
                </div>
                <div class="flex justify-between">
                  <span>Status:</span>
                  <span class={getStatusColor(selectedNode.status)}>{selectedNode.status.toUpperCase()}</span>
                </div>
                <div class="flex justify-between">
                  <span>Load:</span>
                  <span class="text-yellow-400">{(selectedNode.load * 100).toFixed(1)}%</span>
                </div>
                <div class="flex justify-between">
                  <span>Connections:</span>
                  <span class="text-green-400">{selectedNode.connections.length}</span>
                </div>
                <div class="flex justify-between">
                  <span>Position:</span>
                  <span class="text-gray-400">
                    ({selectedNode.x.toFixed(1)}, {selectedNode.y.toFixed(1)}, {selectedNode.z.toFixed(1)})
                  </span>
                </div>
              </div>
            </div>
          {/if}

          <!-- Swarm Configuration -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Swarm Configuration</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Consensus Algorithm</label>
                <select bind:value={swarmConfiguration.consensus_algorithm} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="PBFT">PBFT</option>
                  <option value="RAFT">RAFT</option>
                  <option value="PAXOS">PAXOS</option>
                  <option value="TENDERMINT">Tendermint</option>
                </select>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Max Nodes: {swarmConfiguration.max_nodes}</label>
                <input
                  type="range"
                  bind:value={swarmConfiguration.max_nodes}
                  min="50"
                  max="500"
                  step="10"
                  class="w-full"
                >
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Fault Tolerance: {swarmConfiguration.fault_tolerance}%</label>
                <input
                  type="range"
                  bind:value={swarmConfiguration.fault_tolerance}
                  min="10"
                  max="50"
                  step="1"
                  class="w-full"
                >
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={swarmConfiguration.load_balancing} class="mr-2">
                  <span class="text-sm">Load Balancing</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={swarmConfiguration.auto_scaling} class="mr-2">
                  <span class="text-sm">Auto Scaling</span>
                </label>
              </div>
              <button
                on:click={updateSwarmConfiguration}
                class="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded"
              >
                Update Configuration
              </button>
            </div>
          </div>
        </div>
      </div>

      <!-- Task Management -->
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <!-- Task Allocation -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Distributed Task Allocation</h3>
          <div class="space-y-4">
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="block text-sm font-medium mb-2">Task Type</label>
                <select bind:value={newTask.type} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="computation">Computation</option>
                  <option value="data_processing">Data Processing</option>
                  <option value="analysis">Analysis</option>
                  <option value="monitoring">Monitoring</option>
                  <option value="coordination">Coordination</option>
                </select>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Priority</label>
                <select bind:value={newTask.priority} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="low">Low</option>
                  <option value="normal">Normal</option>
                  <option value="high">High</option>
                  <option value="critical">Critical</option>
                </select>
              </div>
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Data Size (MB)</label>
              <input
                type="number"
                bind:value={newTask.data_size}
                min="0"
                class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
              >
            </div>
            <div>
              <label class="block text-sm font-medium mb-2">Deadline</label>
              <input
                type="datetime-local"
                bind:value={newTask.deadline}
                class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
              >
            </div>
            <button
              on:click={allocateTask}
              class="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
            >
              Allocate Task to Swarm
            </button>
          </div>
        </div>

        <!-- Performance Metrics -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4">Real-time Performance</h3>
          <div class="space-y-4">
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>Computational Efficiency</span>
                <span>{$swarmState.performance.computational_efficiency}%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-purple-400 h-2 rounded-full" style="width: {$swarmState.performance.computational_efficiency}%"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>Resource Utilization</span>
                <span>{$swarmState.performance.resource_utilization}%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-blue-400 h-2 rounded-full" style="width: {$swarmState.performance.resource_utilization}%"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>Energy Efficiency</span>
                <span>{$swarmState.performance.energy_efficiency}%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-green-400 h-2 rounded-full" style="width: {$swarmState.performance.energy_efficiency}%"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>Scalability Index</span>
                <span>{$swarmState.performance.scalability_index}%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-yellow-400 h-2 rounded-full" style="width: {$swarmState.performance.scalability_index}%"></div>
              </div>
            </div>
            <div>
              <div class="flex justify-between text-sm mb-1">
                <span>Fault Tolerance</span>
                <span>{$swarmState.performance.fault_tolerance}%</span>
              </div>
              <div class="w-full bg-gray-700 rounded-full h-2">
                <div class="bg-red-400 h-2 rounded-full" style="width: {$swarmState.performance.fault_tolerance}%"></div>
              </div>
            </div>
          </div>

          <div class="mt-6 pt-4 border-t border-gray-700">
            <h4 class="font-medium mb-2">Current Metrics</h4>
            <div class="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span class="text-gray-400">Avg Latency:</span>
                <span class="text-cyan-400 ml-2">{$swarmState.topology.latency_avg}ms</span>
              </div>
              <div>
                <span class="text-gray-400">Throughput:</span>
                <span class="text-green-400 ml-2">{$swarmState.topology.throughput}</span>
              </div>
              <div>
                <span class="text-gray-400">Message Rate:</span>
                <span class="text-blue-400 ml-2">{$swarmState.communication.message_rate}</span>
              </div>
              <div>
                <span class="text-gray-400">Consensus Round:</span>
                <span class="text-yellow-400 ml-2">{$swarmState.consensus.current_round.toLocaleString()}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    {/if}
  </div>
</div>