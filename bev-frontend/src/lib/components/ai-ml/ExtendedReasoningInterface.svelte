<!--
Extended Reasoning Pipeline Interface - 100K+ Token Context Processing
Connected to: src/agents/extended_reasoning_service.py
Features: Large document processing, reasoning chain visualization, token management
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const reasoningState = writable({
		processes: [],
		active_processes: 0,
		total_context_tokens: 0,
		available_tokens: 100000,
		reasoning_chains: [],
		performance: {
			avg_processing_time: 0,
			tokens_per_second: 0,
			memory_usage: 0,
			cache_hit_rate: 0
		}
	});
	
	const selectedProcess = writable(null);
	const viewMode = writable('overview'); // 'overview', 'submit', 'processes', 'chains', 'optimization'
	const isLoading = writable(false);
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	let metricsTimer: any = null;
	
	// Analysis submission form
	let newAnalysis = {
		type: 'document_analysis',
		content: '',
		context_window: 50000,
		reasoning_depth: 'deep',
		output_format: 'structured',
		enable_chains: true,
		priority: 'normal'
	};
	
	const analysisTypes = [
		'document_analysis', 'research_synthesis', 'causal_reasoning', 
		'comparative_analysis', 'strategic_planning', 'decision_analysis',
		'pattern_recognition', 'knowledge_extraction', 'logical_inference'
	];
	
	const reasoningDepths = [
		{ id: 'surface', label: 'Surface Analysis', tokens: '5K-15K' },
		{ id: 'moderate', label: 'Moderate Analysis', tokens: '15K-40K' },
		{ id: 'deep', label: 'Deep Analysis', tokens: '40K-80K' },
		{ id: 'comprehensive', label: 'Comprehensive Analysis', tokens: '80K-100K+' }
	];
	
	const outputFormats = [
		'structured', 'narrative', 'bullet_points', 'json', 'markdown', 'executive_summary'
	];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadReasoningState();
		startMetricsCollection();
	});
	
	async function initializeWebSocket() {
		try {
			// Connect to Extended Reasoning Service WebSocket
			ws = new WebSocket('ws://localhost:3010/extended-reasoning');
			
			ws.onopen = () => {
				console.log('Extended Reasoning WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleReasoningUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Extended Reasoning WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Extended Reasoning WebSocket connection failed:', error);
		}
	}
	
	function handleReasoningUpdate(data: any) {
		switch (data.type) {
			case 'process_started':
				addProcess(data.process);
				break;
			case 'process_update':
				updateProcess(data.process_id, data.update);
				break;
			case 'process_completed':
				completeProcess(data.process_id, data.result);
				break;
			case 'reasoning_chain_update':
				updateReasoningChain(data.chain);
				break;
			case 'performance_metrics':
				updatePerformanceMetrics(data.metrics);
				break;
		}
	}
	
	async function loadReasoningState() {
		isLoading.set(true);
		try {
			// Call Extended Reasoning Service API endpoints
			const [processesResponse, metricsResponse, performanceResponse] = await Promise.all([
				fetch('http://localhost:3010/extended-reasoning/processes'),
				fetch('http://localhost:3010/extended-reasoning/metrics'),
				fetch('http://localhost:3010/extended-reasoning/performance')
			]);
			
			const [processes, metrics, performance] = await Promise.all([
				processesResponse.json(),
				metricsResponse.json(),
				performanceResponse.json()
			]);
			
			reasoningState.update(current => ({
				...current,
				processes: processes || [],
				active_processes: processes?.filter(p => p.status === 'running').length || 0,
				total_context_tokens: metrics?.total_context_tokens || 0,
				available_tokens: metrics?.available_tokens || 100000,
				performance: { ...current.performance, ...performance }
			}));
			
		} catch (error) {
			console.error('Failed to load reasoning state:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function startMetricsCollection() {
		metricsTimer = setInterval(async () => {
			try {
				const response = await fetch('http://localhost:3010/extended-reasoning/metrics');
				const metrics = await response.json();
				
				reasoningState.update(current => ({
					...current,
					total_context_tokens: metrics.total_context_tokens || current.total_context_tokens,
					available_tokens: metrics.available_tokens || current.available_tokens,
					performance: { ...current.performance, ...metrics.performance }
				}));
			} catch (error) {
				console.error('Failed to collect reasoning metrics:', error);
			}
		}, 5000);
	}
	
	async function submitAnalysis() {
		try {
			const response = await fetch('http://localhost:3010/extended-reasoning/analyze', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(newAnalysis)
			});
			
			const result = await response.json();
			
			// Add process to local state
			const process = {
				id: result.process_id,
				type: newAnalysis.type,
				status: 'starting',
				progress: 0,
				created_at: new Date().toISOString(),
				context_tokens: newAnalysis.context_window,
				reasoning_depth: newAnalysis.reasoning_depth
			};
			
			reasoningState.update(current => ({
				...current,
				processes: [...current.processes, process],
				active_processes: current.active_processes + 1
			}));
			
			// Reset form
			newAnalysis = {
				type: 'document_analysis',
				content: '',
				context_window: 50000,
				reasoning_depth: 'deep',
				output_format: 'structured',
				enable_chains: true,
				priority: 'normal'
			};
			
			dispatch('analysis_submitted', result);
		} catch (error) {
			console.error('Failed to submit analysis:', error);
		}
	}
	
	function addProcess(process: any) {
		reasoningState.update(current => ({
			...current,
			processes: [...current.processes, process],
			active_processes: current.active_processes + 1
		}));
	}
	
	function updateProcess(processId: string, update: any) {
		reasoningState.update(current => ({
			...current,
			processes: current.processes.map(p =>
				p.id === processId ? { ...p, ...update } : p
			)
		}));
	}
	
	function completeProcess(processId: string, result: any) {
		reasoningState.update(current => ({
			...current,
			processes: current.processes.map(p =>
				p.id === processId ? { ...p, status: 'completed', result, progress: 100 } : p
			),
			active_processes: Math.max(0, current.active_processes - 1)
		}));
	}
	
	function updateReasoningChain(chain: any) {
		reasoningState.update(current => ({
			...current,
			reasoning_chains: [...current.reasoning_chains.filter(c => c.id !== chain.id), chain]
		}));
	}
	
	function updatePerformanceMetrics(metrics: any) {
		reasoningState.update(current => ({
			...current,
			performance: { ...current.performance, ...metrics }
		}));
	}
	
	async function cancelProcess(processId: string) {
		try {
			await fetch(`http://localhost:3010/extended-reasoning/process/${processId}`, {
				method: 'DELETE'
			});
			
			reasoningState.update(current => ({
				...current,
				processes: current.processes.filter(p => p.id !== processId),
				active_processes: Math.max(0, current.active_processes - 1)
			}));
		} catch (error) {
			console.error('Failed to cancel process:', error);
		}
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'completed': return 'text-green-400';
			case 'running':
			case 'processing': return 'text-yellow-400';
			case 'failed':
			case 'error': return 'text-red-400';
			case 'starting': return 'text-blue-400';
			default: return 'text-gray-400';
		}
	}
	
	function getPriorityColor(priority: string): string {
		switch (priority) {
			case 'high':
			case 'urgent': return 'text-red-400';
			case 'normal': return 'text-yellow-400';
			case 'low': return 'text-green-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatTokenCount(tokens: number): string {
		if (tokens >= 1000) return (tokens / 1000).toFixed(1) + 'K';
		return tokens.toString();
	}
	
	function formatDuration(ms: number): string {
		if (ms < 1000) return `${ms}ms`;
		const seconds = Math.floor(ms / 1000);
		const minutes = Math.floor(seconds / 60);
		if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
		return `${seconds}s`;
	}
	
	function openProcessModal(process: any) {
		selectedProcess.set(process);
	}
	
	function calculateTokenUtilization(): number {
		const used = $reasoningState.total_context_tokens;
		const available = $reasoningState.available_tokens;
		return available > 0 ? (used / available) * 100 : 0;
	}
</script>

<!-- Extended Reasoning Interface -->
<div class="reasoning-interface h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-indigo-400">Extended Reasoning Pipeline</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					Context: {formatTokenCount($reasoningState.total_context_tokens)}/{formatTokenCount($reasoningState.available_tokens)} tokens
				</div>
				<div class="text-sm text-gray-400">
					{$reasoningState.active_processes} active processes
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadReasoningState}
						class="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Token Utilization Bar -->
	<div class="p-4 border-b border-gray-800">
		<div class="mb-2 flex justify-between text-sm">
			<span class="text-gray-400">Token Utilization</span>
			<span class="text-white">{calculateTokenUtilization().toFixed(1)}%</span>
		</div>
		<div class="w-full bg-gray-700 rounded-full h-2">
			<div
				class="bg-indigo-600 h-2 rounded-full transition-all duration-300"
				style="width: {calculateTokenUtilization()}%"
			></div>
		</div>
	</div>
	
	<!-- Performance Metrics -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">{($reasoningState.performance.avg_processing_time || 0).toFixed(1)}s</div>
				<div class="text-sm text-gray-400">Avg Processing</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-blue-400">{formatTokenCount($reasoningState.performance.tokens_per_second || 0)}/s</div>
				<div class="text-sm text-gray-400">Token Rate</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-yellow-400">{($reasoningState.performance.memory_usage || 0).toFixed(1)}%</div>
				<div class="text-sm text-gray-400">Memory Usage</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-purple-400">{($reasoningState.performance.cache_hit_rate || 0).toFixed(1)}%</div>
				<div class="text-sm text-gray-400">Cache Hit Rate</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Process Overview', icon: '📊' },
				{ id: 'submit', label: 'Submit Analysis', icon: '📝' },
				{ id: 'processes', label: 'Active Processes', icon: '⚙️' },
				{ id: 'chains', label: 'Reasoning Chains', icon: '🔗' },
				{ id: 'optimization', label: 'Token Optimization', icon: '🎯' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$viewMode === tab.id
							? 'border-indigo-500 text-indigo-400'
							: 'border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-300'
					}"
					on:click={() => viewMode.set(tab.id)}
				>
					<span class="mr-2">{tab.icon}</span>
					{tab.label}
				</button>
			{/each}
		</nav>
	</div>
	
	<!-- Main Content Area -->
	<div class="flex-1 overflow-y-auto p-4">
		{#if $viewMode === 'overview'}
			<!-- Process Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Recent Processes -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-indigo-400">Recent Processes</h3>
					{#if $reasoningState.processes.length === 0}
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">⚙️</div>
							<p>No reasoning processes</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $reasoningState.processes.slice(0, 8) as process}
								<div
									class="bg-gray-900 rounded p-3 cursor-pointer hover:bg-gray-700 transition-colors"
									on:click={() => openProcessModal(process)}
								>
									<div class="flex items-center justify-between mb-2">
										<span class="font-medium text-white text-sm">{process.type.replace('_', ' ')}</span>
										<div class="flex items-center space-x-2">
											<span class="text-xs {getStatusColor(process.status)}">{process.status}</span>
											{#if process.progress}
												<span class="text-xs text-gray-400">{process.progress}%</span>
											{/if}
										</div>
									</div>
									<div class="flex items-center justify-between text-xs text-gray-400">
										<span>Tokens: {formatTokenCount(process.context_tokens || 0)}</span>
										<span>Depth: {process.reasoning_depth}</span>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
				
				<!-- System Status -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">System Status</h3>
					<div class="space-y-4">
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Active Processes</span>
							<span class="text-white">{$reasoningState.active_processes}</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Total Processes</span>
							<span class="text-white">{$reasoningState.processes.length}</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Context Usage</span>
							<span class="text-white">{calculateTokenUtilization().toFixed(1)}%</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Available Tokens</span>
							<span class="text-white">{formatTokenCount($reasoningState.available_tokens)}</span>
						</div>
					</div>
					
					<div class="mt-6">
						<h4 class="font-medium text-gray-300 mb-3">Processing Queue</h4>
						<div class="space-y-2">
							{#each $reasoningState.processes.filter(p => p.status === 'running' || p.status === 'starting') as process}
								<div class="bg-gray-900 rounded p-2">
									<div class="flex justify-between items-center text-sm">
										<span class="text-white">{process.type.replace('_', ' ')}</span>
										<span class="{getStatusColor(process.status)}">{process.status}</span>
									</div>
									{#if process.progress}
										<div class="w-full bg-gray-700 rounded-full h-1 mt-2">
											<div
												class="bg-indigo-600 h-1 rounded-full transition-all"
												style="width: {process.progress}%"
											></div>
										</div>
									{/if}
								</div>
							{/each}
						</div>
					</div>
				</div>
			</div>
			
		{:else if $viewMode === 'submit'}
			<!-- Submit Analysis -->
			<div class="max-w-4xl mx-auto">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-6 text-indigo-400">Submit Extended Reasoning Analysis</h3>
					
					<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
						<!-- Analysis Configuration -->
						<div class="space-y-4">
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Analysis Type</label>
								<select
									bind:value={newAnalysis.type}
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
								>
									{#each analysisTypes as type}
										<option value={type}>{type.replace('_', ' ')}</option>
									{/each}
								</select>
							</div>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Reasoning Depth</label>
								<select
									bind:value={newAnalysis.reasoning_depth}
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
								>
									{#each reasoningDepths as depth}
										<option value={depth.id}>{depth.label} ({depth.tokens})</option>
									{/each}
								</select>
							</div>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Context Window</label>
								<input
									type="range"
									bind:value={newAnalysis.context_window}
									min="5000"
									max="100000"
									step="5000"
									class="w-full"
								/>
								<div class="flex justify-between text-sm text-gray-400 mt-1">
									<span>5K</span>
									<span class="text-white">{formatTokenCount(newAnalysis.context_window)}</span>
									<span>100K</span>
								</div>
							</div>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Output Format</label>
								<select
									bind:value={newAnalysis.output_format}
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
								>
									{#each outputFormats as format}
										<option value={format}>{format.replace('_', ' ')}</option>
									{/each}
								</select>
							</div>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Priority</label>
								<select
									bind:value={newAnalysis.priority}
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500"
								>
									<option value="low">Low</option>
									<option value="normal">Normal</option>
									<option value="high">High</option>
									<option value="urgent">Urgent</option>
								</select>
							</div>
							
							<div class="flex items-center">
								<input
									type="checkbox"
									bind:checked={newAnalysis.enable_chains}
									id="enable_chains"
									class="mr-2"
								/>
								<label for="enable_chains" class="text-sm text-gray-300">Enable reasoning chains visualization</label>
							</div>
						</div>
						
						<!-- Content Input -->
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Analysis Content</label>
							<textarea
								bind:value={newAnalysis.content}
								placeholder="Enter the content for extended reasoning analysis (documents, research papers, complex problems, etc.)"
								rows="20"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono text-sm"
							></textarea>
							<div class="mt-2 text-sm text-gray-400">
								Content length: {newAnalysis.content.length} characters (~{Math.ceil(newAnalysis.content.length / 4)} tokens)
							</div>
						</div>
					</div>
					
					<div class="mt-6 flex justify-end">
						<button
							on:click={submitAnalysis}
							class="px-6 py-2 bg-indigo-600 hover:bg-indigo-700 rounded font-medium transition-colors"
							disabled={!newAnalysis.content.trim() || calculateTokenUtilization() > 90}
						>
							Submit Analysis
						</button>
					</div>
				</div>
			</div>
			
		{:else if $viewMode === 'processes'}
			<!-- Active Processes -->
			<div class="space-y-4">
				{#if $reasoningState.processes.length === 0}
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">⚙️</div>
						<p>No reasoning processes</p>
					</div>
				{:else}
					{#each $reasoningState.processes as process}
						<div class="bg-gray-800 rounded-lg p-6">
							<div class="flex items-center justify-between mb-4">
								<div class="flex items-center space-x-4">
									<h3 class="text-lg font-medium text-white">{process.type.replace('_', ' ')}</h3>
									<span class="px-2 py-1 rounded text-xs {getStatusColor(process.status)} bg-gray-700">
										{process.status}
									</span>
									<span class="text-sm text-gray-400">
										{formatTokenCount(process.context_tokens || 0)} tokens
									</span>
								</div>
								<div class="flex items-center space-x-2">
									{#if process.status === 'running'}
										<button
											on:click={() => cancelProcess(process.id)}
											class="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors"
										>
											Cancel
										</button>
									{/if}
									<button
										on:click={() => openProcessModal(process)}
										class="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
									>
										Details
									</button>
								</div>
							</div>
							
							{#if process.progress !== undefined}
								<div class="mb-4">
									<div class="flex justify-between text-sm mb-1">
										<span class="text-gray-400">Progress</span>
										<span class="text-white">{process.progress}%</span>
									</div>
									<div class="w-full bg-gray-700 rounded-full h-2">
										<div
											class="bg-indigo-600 h-2 rounded-full transition-all duration-300"
											style="width: {process.progress}%"
										></div>
									</div>
								</div>
							{/if}
							
							<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
								<div>
									<span class="text-gray-400">Created:</span>
									<span class="text-white ml-2">{new Date(process.created_at).toLocaleString()}</span>
								</div>
								<div>
									<span class="text-gray-400">Depth:</span>
									<span class="text-white ml-2">{process.reasoning_depth}</span>
								</div>
								{#if process.processing_time}
									<div>
										<span class="text-gray-400">Duration:</span>
										<span class="text-white ml-2">{formatDuration(process.processing_time)}</span>
									</div>
								{/if}
								{#if process.result}
									<div>
										<span class="text-gray-400">Result Size:</span>
										<span class="text-white ml-2">{formatTokenCount(process.result.length || 0)}</span>
									</div>
								{/if}
							</div>
						</div>
					{/each}
				{/if}
			</div>
			
		{:else if $viewMode === 'chains'}
			<!-- Reasoning Chains -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-indigo-400">Reasoning Chain Visualization</h3>
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">🔗</div>
					<p>Reasoning chain visualization will be displayed here</p>
					<p class="text-sm mt-2">Interactive flow charts showing logical reasoning steps</p>
				</div>
			</div>
			
		{:else if $viewMode === 'optimization'}
			<!-- Token Optimization -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Token Usage Optimization</h3>
					<div class="space-y-4">
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Current Usage:</span>
							<span class="text-white">{formatTokenCount($reasoningState.total_context_tokens)}</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Available:</span>
							<span class="text-white">{formatTokenCount($reasoningState.available_tokens)}</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Utilization:</span>
							<span class="text-white">{calculateTokenUtilization().toFixed(1)}%</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Optimization Recommendations</h3>
					<div class="space-y-3">
						<div class="bg-gray-900 rounded p-3">
							<div class="font-medium text-green-400 text-sm">Context Window Management</div>
							<div class="text-gray-300 text-sm mt-1">Optimize context windows based on analysis complexity</div>
						</div>
						<div class="bg-gray-900 rounded p-3">
							<div class="font-medium text-blue-400 text-sm">Caching Strategy</div>
							<div class="text-gray-300 text-sm mt-1">Cache frequently used reasoning patterns</div>
						</div>
						<div class="bg-gray-900 rounded p-3">
							<div class="font-medium text-purple-400 text-sm">Batch Processing</div>
							<div class="text-gray-300 text-sm mt-1">Group similar analyses for efficiency</div>
						</div>
					</div>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Extended Reasoning Service | Context: {formatTokenCount($reasoningState.total_context_tokens)}/{formatTokenCount($reasoningState.available_tokens)}
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('optimize_context')}
					class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded text-sm font-medium transition-colors"
				>
					Optimize Context
				</button>
				<button
					on:click={() => dispatch('export_results')}
					class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded text-sm font-medium transition-colors"
				>
					Export Results
				</button>
			</div>
		</div>
	</div>
</div>

<!-- Process Detail Modal -->
{#if $selectedProcess}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedProcess.set(null)}>
		<div class="max-w-4xl w-full mx-4 bg-gray-800 rounded-lg p-6 max-h-[90vh] overflow-y-auto" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-indigo-400">{$selectedProcess.type.replace('_', ' ')}</h3>
				<button
					on:click={() => selectedProcess.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<div class="space-y-4">
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-2">Process Information</h4>
						<div class="space-y-2 text-sm">
							<div class="flex justify-between">
								<span class="text-gray-400">Status:</span>
								<span class="{getStatusColor($selectedProcess.status)}">{$selectedProcess.status}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Context Tokens:</span>
								<span class="text-white">{formatTokenCount($selectedProcess.context_tokens || 0)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Reasoning Depth:</span>
								<span class="text-white">{$selectedProcess.reasoning_depth}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Created:</span>
								<span class="text-white">{new Date($selectedProcess.created_at).toLocaleString()}</span>
							</div>
						</div>
					</div>
					
					{#if $selectedProcess.progress !== undefined}
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">Progress</h4>
							<div class="mb-2 flex justify-between text-sm">
								<span class="text-gray-400">Completion</span>
								<span class="text-white">{$selectedProcess.progress}%</span>
							</div>
							<div class="w-full bg-gray-700 rounded-full h-2">
								<div
									class="bg-indigo-600 h-2 rounded-full transition-all"
									style="width: {$selectedProcess.progress}%"
								></div>
							</div>
						</div>
					{/if}
				</div>
				
				{#if $selectedProcess.result}
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-2">Analysis Result</h4>
						<div class="bg-gray-800 rounded p-3 max-h-96 overflow-y-auto">
							<pre class="text-gray-300 text-sm whitespace-pre-wrap">{$selectedProcess.result}</pre>
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.reasoning-interface {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.reasoning-interface *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.reasoning-interface *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.reasoning-interface *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.reasoning-interface *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>