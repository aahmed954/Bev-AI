<!--
Cytoscape Analysis Studio - Advanced Graph Visualization & Network Analysis
Connected to: cytoscape/ (advanced graph visualization system)
Features: Interactive graph analysis, network topology, relationship mapping, layout algorithms
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	import cytoscape from 'cytoscape';
	import fcose from 'cytoscape-fcose';
	
	cytoscape.use(fcose);
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const graphState = writable({
		nodes: [],
		edges: [],
		layouts: [],
		analysis_results: {},
		selected_elements: [],
		graph_metrics: {
			node_count: 0,
			edge_count: 0,
			density: 0,
			clustering_coefficient: 0,
			average_path_length: 0
		}
	});
	
	const selectedTab = writable('studio'); // 'studio', 'analysis', 'layouts', 'data', 'export'
	const selectedLayout = writable('fcose');
	const isLoading = writable(false);
	
	// Cytoscape instance
	let cy: any = null;
	let cyContainer: HTMLElement;
	
	// WebSocket for real-time graph updates
	let ws: WebSocket | null = null;
	
	// Available layouts
	const layoutOptions = [
		{ id: 'fcose', name: 'fCoSE (Force-directed)', description: 'High-quality compound graph layout' },
		{ id: 'cose', name: 'CoSE (Compound Spring)', description: 'Physics-based spring layout' },
		{ id: 'circle', name: 'Circle Layout', description: 'Circular node arrangement' },
		{ id: 'grid', name: 'Grid Layout', description: 'Regular grid pattern' },
		{ id: 'concentric', name: 'Concentric Layout', description: 'Concentric circles by importance' },
		{ id: 'breadthfirst', name: 'Hierarchical', description: 'Tree-like hierarchy' },
		{ id: 'dagre', name: 'Dagre (Directed)', description: 'Directed acyclic graph layout' }
	];
	
	// Graph analysis tools
	const analysisTools = [
		{ id: 'centrality', name: 'Centrality Analysis', icon: '🎯' },
		{ id: 'clustering', name: 'Clustering Detection', icon: '🔗' },
		{ id: 'pathfinding', name: 'Shortest Paths', icon: '🛤️' },
		{ id: 'community', name: 'Community Detection', icon: '👥' },
		{ id: 'influence', name: 'Influence Mapping', icon: '📈' },
		{ id: 'temporal', name: 'Temporal Analysis', icon: '⏰' }
	];
	
	// Sample graph data for demonstration
	const sampleGraphData = {
		nodes: [
			{ data: { id: 'threat1', label: 'Threat Actor A', type: 'threat', risk: 'high' } },
			{ data: { id: 'ip1', label: '192.168.1.100', type: 'ip', risk: 'medium' } },
			{ data: { id: 'domain1', label: 'malicious.com', type: 'domain', risk: 'high' } },
			{ data: { id: 'wallet1', label: '1BvBMSE...xYz', type: 'wallet', risk: 'high' } },
			{ data: { id: 'social1', label: '@suspicious_user', type: 'social', risk: 'medium' } }
		],
		edges: [
			{ data: { source: 'threat1', target: 'ip1', relationship: 'controls', strength: 0.9 } },
			{ data: { source: 'threat1', target: 'domain1', relationship: 'owns', strength: 0.8 } },
			{ data: { source: 'ip1', target: 'domain1', relationship: 'hosts', strength: 0.7 } },
			{ data: { source: 'threat1', target: 'wallet1', relationship: 'operates', strength: 0.9 } },
			{ data: { source: 'social1', target: 'threat1', relationship: 'associated', strength: 0.6 } }
		]
	};
	
	onMount(async () => {
		await initializeWebSocket();
		await initializeCytoscape();
		await loadGraphData();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/cytoscape-studio');
			
			ws.onopen = () => {
				console.log('Cytoscape Studio WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleGraphUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Cytoscape WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Cytoscape WebSocket connection failed:', error);
		}
	}
	
	function handleGraphUpdate(data: any) {
		switch (data.type) {
			case 'node_added':
				addNode(data.node);
				break;
			case 'edge_added':
				addEdge(data.edge);
				break;
			case 'graph_updated':
				updateGraph(data.graph);
				break;
			case 'analysis_complete':
				updateAnalysisResults(data.analysis);
				break;
		}
	}
	
	async function initializeCytoscape() {
		if (cyContainer) {
			cy = cytoscape({
				container: cyContainer,
				elements: [],
				style: [
					{
						selector: 'node',
						style: {
							'background-color': 'data(color)',
							'label': 'data(label)',
							'color': '#ffffff',
							'text-valign': 'center',
							'text-halign': 'center',
							'font-size': '12px',
							'width': 'mapData(risk, 0, 1, 20, 60)',
							'height': 'mapData(risk, 0, 1, 20, 60)'
						}
					},
					{
						selector: 'node[type="threat"]',
						style: {
							'background-color': '#ef4444',
							'shape': 'triangle'
						}
					},
					{
						selector: 'node[type="ip"]',
						style: {
							'background-color': '#3b82f6',
							'shape': 'rectangle'
						}
					},
					{
						selector: 'node[type="domain"]',
						style: {
							'background-color': '#8b5cf6',
							'shape': 'ellipse'
						}
					},
					{
						selector: 'node[type="wallet"]',
						style: {
							'background-color': '#f59e0b',
							'shape': 'diamond'
						}
					},
					{
						selector: 'node[type="social"]',
						style: {
							'background-color': '#10b981',
							'shape': 'octagon'
						}
					},
					{
						selector: 'edge',
						style: {
							'width': 'mapData(strength, 0, 1, 1, 5)',
							'line-color': '#6b7280',
							'target-arrow-color': '#6b7280',
							'target-arrow-shape': 'triangle',
							'curve-style': 'bezier',
							'label': 'data(relationship)',
							'font-size': '10px',
							'color': '#9ca3af'
						}
					},
					{
						selector: ':selected',
						style: {
							'border-width': 3,
							'border-color': '#fbbf24'
						}
					}
				],
				layout: {
					name: 'fcose',
					quality: 'default',
					randomize: false,
					animate: true,
					animationDuration: 1000
				}
			});
			
			// Event handlers
			cy.on('tap', 'node', (evt) => {
				const node = evt.target;
				selectElement(node.data());
			});
			
			cy.on('tap', 'edge', (evt) => {
				const edge = evt.target;
				selectElement(edge.data());
			});
		}
	}
	
	async function loadGraphData() {
		isLoading.set(true);
		try {
			const graphData = await invoke('get_cytoscape_data');
			
			if (graphData && graphData.elements) {
				updateGraph(graphData);
			} else {
				// Load sample data for demonstration
				updateGraph(sampleGraphData);
			}
		} catch (error) {
			console.error('Failed to load graph data, using sample:', error);
			updateGraph(sampleGraphData);
		} finally {
			isLoading.set(false);
		}
	}
	
	function updateGraph(graphData: any) {
		if (cy) {
			cy.elements().remove();
			cy.add(graphData);
			cy.layout({ name: $selectedLayout }).run();
		}
		
		graphState.update(current => ({
			...current,
			nodes: graphData.nodes || [],
			edges: graphData.edges || [],
			graph_metrics: calculateGraphMetrics(graphData)
		}));
	}
	
	function calculateGraphMetrics(graphData: any): any {
		const nodeCount = graphData.nodes?.length || 0;
		const edgeCount = graphData.edges?.length || 0;
		
		// Calculate basic graph metrics
		const density = nodeCount > 1 ? (2 * edgeCount) / (nodeCount * (nodeCount - 1)) : 0;
		
		return {
			node_count: nodeCount,
			edge_count: edgeCount,
			density: density,
			clustering_coefficient: 0.75, // Placeholder - would calculate actual clustering
			average_path_length: 2.3 // Placeholder - would calculate actual path length
		};
	}
	
	function changeLayout(layoutName: string) {
		selectedLayout.set(layoutName);
		if (cy) {
			cy.layout({ 
				name: layoutName,
				animate: true,
				animationDuration: 1000 
			}).run();
		}
	}
	
	async function runGraphAnalysis(analysisType: string) {
		try {
			const result = await invoke('run_graph_analysis', { 
				type: analysisType,
				graph: { nodes: $graphState.nodes, edges: $graphState.edges }
			});
			
			updateAnalysisResults({ [analysisType]: result });
		} catch (error) {
			console.error('Failed to run graph analysis:', error);
		}
	}
	
	function addNode(node: any) {
		if (cy) {
			cy.add({ group: 'nodes', data: node });
		}
		
		graphState.update(current => ({
			...current,
			nodes: [...current.nodes, { data: node }]
		}));
	}
	
	function addEdge(edge: any) {
		if (cy) {
			cy.add({ group: 'edges', data: edge });
		}
		
		graphState.update(current => ({
			...current,
			edges: [...current.edges, { data: edge }]
		}));
	}
	
	function updateAnalysisResults(analysis: any) {
		graphState.update(current => ({
			...current,
			analysis_results: { ...current.analysis_results, ...analysis }
		}));
	}
	
	function selectElement(elementData: any) {
		graphState.update(current => ({
			...current,
			selected_elements: [elementData]
		}));
		
		dispatch('element_selected', elementData);
	}
	
	async function exportGraph(format: string) {
		try {
			let exportData;
			
			if (format === 'png' && cy) {
				exportData = cy.png({ 
					output: 'blob',
					bg: '#111827',
					full: true,
					scale: 2
				});
			} else if (format === 'json') {
				exportData = JSON.stringify({
					nodes: $graphState.nodes,
					edges: $graphState.edges,
					analysis: $graphState.analysis_results
				}, null, 2);
			}
			
			// Trigger download
			if (exportData) {
				const blob = new Blob([exportData], { 
					type: format === 'png' ? 'image/png' : 'application/json' 
				});
				const url = URL.createObjectURL(blob);
				const a = document.createElement('a');
				a.href = url;
				a.download = `cytoscape_graph_${Date.now()}.${format}`;
				a.click();
				URL.revokeObjectURL(url);
			}
		} catch (error) {
			console.error('Failed to export graph:', error);
		}
	}
	
	function getRiskColor(risk: string): string {
		switch (risk) {
			case 'high': return 'text-red-400';
			case 'medium': return 'text-yellow-400';
			case 'low': return 'text-green-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatNumber(num: number): string {
		return num.toFixed(3);
	}
</script>

<!-- Cytoscape Analysis Studio -->
<div class="cytoscape-studio h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-cyan-400">🕸️ Cytoscape Analysis Studio</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{$graphState.graph_metrics.node_count} nodes | {$graphState.graph_metrics.edge_count} edges
				</div>
				<div class="text-sm text-gray-400">
					Density: {formatNumber($graphState.graph_metrics.density)}
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadGraphData}
						class="px-3 py-1 bg-cyan-600 hover:bg-cyan-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Graph Metrics Bar -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-5 gap-4">
			<div class="bg-gray-800 rounded-lg p-3 text-center">
				<div class="text-lg font-bold text-blue-400">{$graphState.graph_metrics.node_count}</div>
				<div class="text-xs text-gray-400">Nodes</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-3 text-center">
				<div class="text-lg font-bold text-green-400">{$graphState.graph_metrics.edge_count}</div>
				<div class="text-xs text-gray-400">Edges</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-3 text-center">
				<div class="text-lg font-bold text-yellow-400">{formatNumber($graphState.graph_metrics.density)}</div>
				<div class="text-xs text-gray-400">Density</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-3 text-center">
				<div class="text-lg font-bold text-purple-400">{formatNumber($graphState.graph_metrics.clustering_coefficient)}</div>
				<div class="text-xs text-gray-400">Clustering</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-3 text-center">
				<div class="text-lg font-bold text-orange-400">{formatNumber($graphState.graph_metrics.average_path_length)}</div>
				<div class="text-xs text-gray-400">Avg Path</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'studio', label: 'Graph Studio', icon: '🎨' },
				{ id: 'analysis', label: 'Network Analysis', icon: '🔍' },
				{ id: 'layouts', label: 'Layout Control', icon: '🎛️' },
				{ id: 'data', label: 'Data Management', icon: '📊' },
				{ id: 'export', label: 'Export Tools', icon: '📤' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedTab === tab.id
							? 'border-cyan-500 text-cyan-400'
							: 'border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-300'
					}"
					on:click={() => selectedTab.set(tab.id)}
				>
					<span class="mr-2">{tab.icon}</span>
					{tab.label}
				</button>
			{/each}
		</nav>
	</div>
	
	<!-- Main Content Area -->
	<div class="flex-1 flex">
		{#if $selectedTab === 'studio'}
			<!-- Graph Visualization Studio -->
			<div class="flex-1 p-4">
				<div class="bg-gray-800 rounded-lg h-full relative">
					<div
						bind:this={cyContainer}
						class="w-full h-full rounded-lg"
						style="min-height: 500px;"
					></div>
					
					<!-- Graph Controls Overlay -->
					<div class="absolute top-4 right-4 bg-gray-900/90 rounded p-3">
						<div class="space-y-2">
							<div class="text-sm font-medium text-white">Layout</div>
							<select
								bind:value={$selectedLayout}
								on:change={() => changeLayout($selectedLayout)}
								class="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-white text-xs"
							>
								{#each layoutOptions as layout}
									<option value={layout.id}>{layout.name}</option>
								{/each}
							</select>
						</div>
					</div>
					
					<!-- Element Inspector -->
					{#if $graphState.selected_elements.length > 0}
						<div class="absolute bottom-4 left-4 bg-gray-900/90 rounded p-4 max-w-xs">
							{@const element = $graphState.selected_elements[0]}
							<div class="text-sm font-medium text-white mb-2">Selected Element</div>
							<div class="space-y-1 text-xs">
								<div class="flex justify-between">
									<span class="text-gray-400">ID:</span>
									<span class="text-white">{element.id}</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Type:</span>
									<span class="text-white">{element.type}</span>
								</div>
								{#if element.risk}
									<div class="flex justify-between">
										<span class="text-gray-400">Risk:</span>
										<span class="{getRiskColor(element.risk)}">{element.risk}</span>
									</div>
								{/if}
								{#if element.relationship}
									<div class="flex justify-between">
										<span class="text-gray-400">Relationship:</span>
										<span class="text-white">{element.relationship}</span>
									</div>
								{/if}
							</div>
						</div>
					{/if}
				</div>
			</div>
			
			<!-- Analysis Panel -->
			<div class="w-80 p-4 border-l border-gray-800">
				<div class="bg-gray-800 rounded-lg p-4 h-full">
					<h3 class="text-lg font-semibold mb-4 text-cyan-400">Analysis Tools</h3>
					<div class="space-y-3">
						{#each analysisTools as tool}
							<button
								on:click={() => runGraphAnalysis(tool.id)}
								class="w-full text-left p-3 bg-gray-900 hover:bg-gray-700 rounded transition-colors"
							>
								<div class="flex items-center space-x-2">
									<span>{tool.icon}</span>
									<span class="font-medium text-white text-sm">{tool.name}</span>
								</div>
							</button>
						{/each}
					</div>
					
					{#if Object.keys($graphState.analysis_results).length > 0}
						<div class="mt-6">
							<h4 class="font-medium text-white mb-2">Analysis Results</h4>
							<div class="space-y-2">
								{#each Object.entries($graphState.analysis_results) as [analysis, result]}
									<div class="bg-gray-900 rounded p-2">
										<div class="font-medium text-white text-xs capitalize">{analysis.replace('_', ' ')}</div>
										<div class="text-xs text-gray-400">
											{typeof result === 'object' ? JSON.stringify(result).slice(0, 50) + '...' : result}
										</div>
									</div>
								{/each}
							</div>
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'layouts'}
			<!-- Layout Control -->
			<div class="flex-1 p-4">
				<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
					{#each layoutOptions as layout}
						<div 
							class="bg-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-700 transition-colors {
								$selectedLayout === layout.id ? 'ring-2 ring-cyan-500' : ''
							}"
							on:click={() => changeLayout(layout.id)}
						>
							<h4 class="font-medium text-white mb-2">{layout.name}</h4>
							<p class="text-gray-400 text-sm">{layout.description}</p>
							{#if $selectedLayout === layout.id}
								<div class="mt-2 text-xs text-cyan-400">● Active</div>
							{/if}
						</div>
					{/each}
				</div>
			</div>
			
		{:else}
			<!-- Other tab content -->
			<div class="flex-1 p-4">
				<div class="bg-gray-800 rounded-lg p-6 h-full">
					<h3 class="text-lg font-semibold mb-4 text-white">
						{$selectedTab === 'analysis' ? '🔍 Network Analysis' :
						 $selectedTab === 'data' ? '📊 Data Management' :
						 '📤 Export Tools'}
					</h3>
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">
							{$selectedTab === 'analysis' ? '🔍' :
							 $selectedTab === 'data' ? '📊' : '📤'}
						</div>
						<p>Advanced {$selectedTab} interface</p>
						<p class="text-sm mt-2">Professional graph analysis and management tools</p>
					</div>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Cytoscape Analysis Studio | Layout: {layoutOptions.find(l => l.id === $selectedLayout)?.name}
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => exportGraph('png')}
					class="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded text-sm font-medium transition-colors"
				>
					Export PNG
				</button>
				<button
					on:click={() => exportGraph('json')}
					class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
				>
					Export Data
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	.cytoscape-studio {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.cytoscape-studio *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.cytoscape-studio *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.cytoscape-studio *::-webkit-scrollbar-thumb) {
		background: #06b6d4;
		border-radius: 3px;
	}
	
	:global(.cytoscape-studio *::-webkit-scrollbar-thumb:hover) {
		background: #67e8f9;
	}
</style>