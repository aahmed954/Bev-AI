<!--
Airflow DAG Management Controller - ML Pipeline Orchestration
Connected to: dags/ (5 DAGs: research_pipeline, health_monitoring, ml_training, etc.)
Features: DAG visualization, execution control, log streaming, performance analytics
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const dags = writable([]);
	const dagRuns = writable([]);
	const selectedDAG = writable(null);
	const selectedTab = writable('overview'); // 'overview', 'dags', 'runs', 'logs', 'metrics'
	const isLoading = writable(false);
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	
	// DAG execution data
	let executionMetrics = {
		total_runs_today: 0,
		successful_runs: 0,
		failed_runs: 0,
		avg_duration: 0,
		active_runs: 0
	};
	
	// Available DAGs from backend
	const availableDAGs = [
		{
			dag_id: 'research_pipeline_dag',
			description: 'Automated research data collection and processing pipeline',
			schedule_interval: '@daily',
			category: 'research',
			tasks: 8,
			is_active: true
		},
		{
			dag_id: 'bev_health_monitoring',
			description: 'System health monitoring and alerting pipeline',
			schedule_interval: '*/5 * * * *',
			category: 'monitoring',
			tasks: 6,
			is_active: true
		},
		{
			dag_id: 'data_lake_medallion_dag',
			description: 'Data lake medallion architecture ETL pipeline',
			schedule_interval: '@hourly',
			category: 'data',
			tasks: 12,
			is_active: true
		},
		{
			dag_id: 'ml_training_pipeline_dag',
			description: 'Machine learning model training and deployment pipeline',
			schedule_interval: '@weekly',
			category: 'ml',
			tasks: 15,
			is_active: true
		},
		{
			dag_id: 'cost_optimization_dag',
			description: 'Infrastructure cost optimization and resource management',
			schedule_interval: '@daily',
			category: 'optimization',
			tasks: 7,
			is_active: true
		}
	];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadDAGs();
		await loadDAGRuns();
		calculateMetrics();
	});
	
	async function initializeWebSocket() {
		try {
			// Connect to Airflow WebSocket endpoint
			ws = new WebSocket('ws://localhost:8080/ws/airflow');
			
			ws.onopen = () => {
				console.log('Airflow WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleAirflowUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Airflow WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Airflow WebSocket connection failed:', error);
		}
	}
	
	function handleAirflowUpdate(data: any) {
		switch (data.type) {
			case 'dag_run_started':
				addDAGRun(data.dag_run);
				break;
			case 'dag_run_completed':
				updateDAGRun(data.dag_run_id, data.result);
				break;
			case 'task_state_change':
				updateTaskState(data.dag_run_id, data.task_id, data.state);
				break;
			case 'dag_state_change':
				updateDAGState(data.dag_id, data.state);
				break;
		}
	}
	
	async function loadDAGs() {
		isLoading.set(true);
		try {
			// Try to load from Airflow API
			const result = await invoke('get_airflow_dags');
			dags.set(result);
		} catch (error) {
			console.error('Failed to load DAGs from API, using local definitions:', error);
			// Use local DAG definitions
			dags.set(availableDAGs);
		} finally {
			isLoading.set(false);
		}
	}
	
	async function loadDAGRuns() {
		try {
			const result = await invoke('get_airflow_dag_runs');
			dagRuns.set(result || []);
		} catch (error) {
			console.error('Failed to load DAG runs:', error);
		}
	}
	
	function calculateMetrics() {
		const runs = $dagRuns;
		const today = new Date().toDateString();
		
		const todayRuns = runs.filter(run => 
			new Date(run.execution_date).toDateString() === today
		);
		
		executionMetrics = {
			total_runs_today: todayRuns.length,
			successful_runs: todayRuns.filter(run => run.state === 'success').length,
			failed_runs: todayRuns.filter(run => run.state === 'failed').length,
			avg_duration: todayRuns.length > 0 
				? todayRuns.reduce((acc, run) => acc + (run.duration || 0), 0) / todayRuns.length 
				: 0,
			active_runs: runs.filter(run => run.state === 'running').length
		};
	}
	
	async function triggerDAG(dagId: string) {
		try {
			const result = await invoke('trigger_airflow_dag', { dagId });
			addDAGRun({
				dag_run_id: result.dag_run_id,
				dag_id: dagId,
				state: 'running',
				execution_date: new Date().toISOString(),
				start_date: new Date().toISOString()
			});
			dispatch('dag_triggered', { dagId, runId: result.dag_run_id });
		} catch (error) {
			console.error('Failed to trigger DAG:', error);
		}
	}
	
	async function pauseDAG(dagId: string) {
		try {
			await invoke('pause_airflow_dag', { dagId });
			updateDAGState(dagId, 'paused');
		} catch (error) {
			console.error('Failed to pause DAG:', error);
		}
	}
	
	async function unpauseDAG(dagId: string) {
		try {
			await invoke('unpause_airflow_dag', { dagId });
			updateDAGState(dagId, 'active');
		} catch (error) {
			console.error('Failed to unpause DAG:', error);
		}
	}
	
	function addDAGRun(dagRun: any) {
		dagRuns.update(current => [dagRun, ...current.slice(0, 99)]);
		calculateMetrics();
	}
	
	function updateDAGRun(dagRunId: string, update: any) {
		dagRuns.update(current =>
			current.map(run =>
				run.dag_run_id === dagRunId ? { ...run, ...update } : run
			)
		);
		calculateMetrics();
	}
	
	function updateDAGState(dagId: string, state: string) {
		dags.update(current =>
			current.map(dag =>
				dag.dag_id === dagId ? { ...dag, state, is_active: state === 'active' } : dag
			)
		);
	}
	
	function updateTaskState(dagRunId: string, taskId: string, state: string) {
		dagRuns.update(current =>
			current.map(run =>
				run.dag_run_id === dagRunId
					? {
						...run,
						tasks: {
							...run.tasks,
							[taskId]: { ...run.tasks?.[taskId], state }
						}
					}
					: run
			)
		);
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'success':
			case 'active': return 'text-green-400';
			case 'running': return 'text-yellow-400';
			case 'failed':
			case 'error': return 'text-red-400';
			case 'paused':
			case 'stopped': return 'text-gray-400';
			case 'queued': return 'text-blue-400';
			default: return 'text-gray-400';
		}
	}
	
	function getCategoryColor(category: string): string {
		switch (category) {
			case 'research': return 'text-blue-400';
			case 'monitoring': return 'text-green-400';
			case 'data': return 'text-purple-400';
			case 'ml': return 'text-orange-400';
			case 'optimization': return 'text-yellow-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatDuration(seconds: number): string {
		if (!seconds) return 'N/A';
		const minutes = Math.floor(seconds / 60);
		const hours = Math.floor(minutes / 60);
		if (hours > 0) return `${hours}h ${minutes % 60}m`;
		return `${minutes}m ${seconds % 60}s`;
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleString();
	}
	
	function openDAGDetails(dag: any) {
		selectedDAG.set(dag);
	}
</script>

<!-- Airflow DAG Controller -->
<div class="airflow-controller h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-cyan-400">Airflow DAG Orchestration</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{$dags.length} DAGs | {executionMetrics.active_runs} running
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadDAGs}
						class="px-3 py-1 bg-cyan-600 hover:bg-cyan-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Execution Metrics -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-5 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-cyan-400">{$dags.length}</div>
				<div class="text-sm text-gray-400">Total DAGs</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">{executionMetrics.successful_runs}</div>
				<div class="text-sm text-gray-400">Success Today</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-red-400">{executionMetrics.failed_runs}</div>
				<div class="text-sm text-gray-400">Failed Today</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-yellow-400">{executionMetrics.active_runs}</div>
				<div class="text-sm text-gray-400">Active Runs</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-blue-400">{formatDuration(executionMetrics.avg_duration)}</div>
				<div class="text-sm text-gray-400">Avg Duration</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'DAG Overview', icon: '📊' },
				{ id: 'dags', label: 'DAG Management', icon: '🔧' },
				{ id: 'runs', label: 'Execution History', icon: '⚙️' },
				{ id: 'logs', label: 'Live Logs', icon: '📝' },
				{ id: 'metrics', label: 'Performance', icon: '📈' }
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
	<div class="flex-1 overflow-y-auto p-4">
		{#if $selectedTab === 'overview'}
			<!-- DAG Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- DAG Status Grid -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-cyan-400">DAG Status Matrix</h3>
					{#if $dags.length === 0}
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">🔧</div>
							<p>No DAGs available</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $dags as dag}
								<div
									class="bg-gray-900 rounded p-4 cursor-pointer hover:bg-gray-700 transition-colors"
									on:click={() => openDAGDetails(dag)}
								>
									<div class="flex items-center justify-between mb-2">
										<div class="flex items-center space-x-3">
											<h4 class="font-medium text-white">{dag.dag_id.replace('_', ' ')}</h4>
											<span class="px-2 py-1 rounded text-xs {getStatusColor(dag.is_active ? 'active' : 'paused')} bg-gray-800">
												{dag.is_active ? 'Active' : 'Paused'}
											</span>
											<span class="px-2 py-1 rounded text-xs {getCategoryColor(dag.category)} bg-gray-800">
												{dag.category}
											</span>
										</div>
										<div class="text-sm text-gray-400">
											{dag.tasks} tasks
										</div>
									</div>
									
									<div class="text-sm text-gray-400 mb-2">{dag.description}</div>
									
									<div class="flex items-center justify-between text-xs">
										<span class="text-gray-500">Schedule: {dag.schedule_interval}</span>
										<span class="text-gray-500">Last run: {dag.last_run || 'Never'}</span>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
				
				<!-- Recent Executions -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Recent Executions</h3>
					{#if $dagRuns.length === 0}
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">⚙️</div>
							<p>No recent executions</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $dagRuns.slice(0, 8) as run}
								<div class="bg-gray-900 rounded p-3">
									<div class="flex items-center justify-between mb-2">
										<span class="font-medium text-white text-sm">{run.dag_id.replace('_', ' ')}</span>
										<div class="flex items-center space-x-2">
											<span class="text-xs {getStatusColor(run.state)}">{run.state}</span>
											<span class="text-xs text-gray-400">{formatDate(run.execution_date)}</span>
										</div>
									</div>
									
									{#if run.duration}
										<div class="text-xs text-gray-400">
											Duration: {formatDuration(run.duration)}
										</div>
									{/if}
									
									{#if run.state === 'running' && run.progress}
										<div class="mt-2">
											<div class="w-full bg-gray-700 rounded-full h-1">
												<div
													class="bg-cyan-600 h-1 rounded-full transition-all"
													style="width: {run.progress}%"
												></div>
											</div>
										</div>
									{/if}
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'dags'}
			<!-- DAG Management -->
			<div class="space-y-4">
				{#each $dags as dag}
					<div class="bg-gray-800 rounded-lg p-6">
						<div class="flex items-center justify-between mb-4">
							<div class="flex items-center space-x-4">
								<h3 class="text-lg font-semibold text-white">{dag.dag_id.replace('_', ' ')}</h3>
								<span class="px-2 py-1 rounded text-xs {getCategoryColor(dag.category)} bg-gray-700">
									{dag.category}
								</span>
								<span class="px-2 py-1 rounded text-xs {getStatusColor(dag.is_active ? 'active' : 'paused')} bg-gray-700">
									{dag.is_active ? 'Active' : 'Paused'}
								</span>
							</div>
							<div class="flex items-center space-x-2">
								<button
									on:click={() => triggerDAG(dag.dag_id)}
									class="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm transition-colors"
								>
									Trigger
								</button>
								{#if dag.is_active}
									<button
										on:click={() => pauseDAG(dag.dag_id)}
										class="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-sm transition-colors"
									>
										Pause
									</button>
								{:else}
									<button
										on:click={() => unpauseDAG(dag.dag_id)}
										class="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
									>
										Unpause
									</button>
								{/if}
								<button
									on:click={() => openDAGDetails(dag)}
									class="px-3 py-1 bg-gray-600 hover:bg-gray-700 rounded text-sm transition-colors"
								>
									Details
								</button>
							</div>
						</div>
						
						<p class="text-gray-400 text-sm mb-4">{dag.description}</p>
						
						<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
							<div>
								<span class="text-gray-400">Schedule:</span>
								<span class="text-white ml-2">{dag.schedule_interval}</span>
							</div>
							<div>
								<span class="text-gray-400">Tasks:</span>
								<span class="text-white ml-2">{dag.tasks}</span>
							</div>
							<div>
								<span class="text-gray-400">Last Run:</span>
								<span class="text-white ml-2">{dag.last_run || 'Never'}</span>
							</div>
							<div>
								<span class="text-gray-400">Success Rate:</span>
								<span class="text-white ml-2">{(dag.success_rate || 0).toFixed(1)}%</span>
							</div>
						</div>
					</div>
				{/each}
			</div>
			
		{:else if $selectedTab === 'runs'}
			<!-- Execution History -->
			<div class="space-y-4">
				{#if $dagRuns.length === 0}
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">⚙️</div>
						<p>No execution history</p>
					</div>
				{:else}
					{#each $dagRuns as run}
						<div class="bg-gray-800 rounded-lg p-6">
							<div class="flex items-center justify-between mb-4">
								<div class="flex items-center space-x-4">
									<h3 class="text-lg font-medium text-white">{run.dag_id.replace('_', ' ')}</h3>
									<span class="px-2 py-1 rounded text-xs {getStatusColor(run.state)} bg-gray-700">
										{run.state}
									</span>
									<span class="text-sm text-gray-400">Run #{run.dag_run_id.slice(-8)}</span>
								</div>
								<div class="text-sm text-gray-400">
									{formatDate(run.execution_date)}
								</div>
							</div>
							
							<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
								<div>
									<span class="text-gray-400">Started:</span>
									<span class="text-white ml-2">{run.start_date ? formatDate(run.start_date) : 'N/A'}</span>
								</div>
								<div>
									<span class="text-gray-400">Duration:</span>
									<span class="text-white ml-2">{run.duration ? formatDuration(run.duration) : 'N/A'}</span>
								</div>
								<div>
									<span class="text-gray-400">Tasks:</span>
									<span class="text-white ml-2">{Object.keys(run.tasks || {}).length}</span>
								</div>
								{#if run.state === 'running'}
									<div>
										<span class="text-gray-400">Progress:</span>
										<span class="text-white ml-2">{run.progress || 0}%</span>
									</div>
								{/if}
							</div>
							
							{#if run.state === 'running' && run.progress}
								<div class="mt-4">
									<div class="w-full bg-gray-700 rounded-full h-2">
										<div
											class="bg-cyan-600 h-2 rounded-full transition-all"
											style="width: {run.progress}%"
										></div>
									</div>
								</div>
							{/if}
						</div>
					{/each}
				{/if}
			</div>
			
		{:else if $selectedTab === 'logs'}
			<!-- Live Logs -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-cyan-400">Live DAG Logs</h3>
				<div class="bg-gray-900 rounded p-4 h-96 overflow-y-auto font-mono text-sm">
					<div class="text-gray-400">
						[{new Date().toLocaleTimeString()}] Connected to Airflow log stream...<br/>
						[{new Date().toLocaleTimeString()}] Monitoring DAG executions...<br/>
						[{new Date().toLocaleTimeString()}] Ready for real-time log streaming...<br/>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'metrics'}
			<!-- Performance Metrics -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Execution Metrics</h3>
					<div class="space-y-4">
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Total Runs Today:</span>
							<span class="text-white">{executionMetrics.total_runs_today}</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Success Rate:</span>
							<span class="text-white">
								{executionMetrics.total_runs_today > 0 
									? ((executionMetrics.successful_runs / executionMetrics.total_runs_today) * 100).toFixed(1) 
									: 0}%
							</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Average Duration:</span>
							<span class="text-white">{formatDuration(executionMetrics.avg_duration)}</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Active Executions:</span>
							<span class="text-white">{executionMetrics.active_runs}</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Performance Charts</h3>
					<div class="text-center py-8 text-gray-400">
						<div class="text-3xl mb-2">📈</div>
						<p>Performance visualization</p>
						<p class="text-sm mt-2">DAG execution trends and resource utilization</p>
					</div>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Airflow Orchestration | {$dags.filter(d => d.is_active).length} active DAGs
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('export_dag_metrics')}
					class="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded text-sm font-medium transition-colors"
				>
					Export Metrics
				</button>
				<button
					on:click={() => dispatch('emergency_stop_all')}
					class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
				>
					Emergency Stop All
				</button>
			</div>
		</div>
	</div>
</div>

<!-- DAG Detail Modal -->
{#if $selectedDAG}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedDAG.set(null)}>
		<div class="max-w-4xl w-full mx-4 bg-gray-800 rounded-lg p-6 max-h-[90vh] overflow-y-auto" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-cyan-400">{$selectedDAG.dag_id.replace('_', ' ')}</h3>
				<button
					on:click={() => selectedDAG.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">DAG Information</h4>
					<div class="grid grid-cols-2 gap-4 text-sm">
						<div>
							<span class="text-gray-400">Status:</span>
							<span class="text-white ml-2 {getStatusColor($selectedDAG.is_active ? 'active' : 'paused')}">
								{$selectedDAG.is_active ? 'Active' : 'Paused'}
							</span>
						</div>
						<div>
							<span class="text-gray-400">Category:</span>
							<span class="text-white ml-2">{$selectedDAG.category}</span>
						</div>
						<div>
							<span class="text-gray-400">Schedule:</span>
							<span class="text-white ml-2">{$selectedDAG.schedule_interval}</span>
						</div>
						<div>
							<span class="text-gray-400">Tasks:</span>
							<span class="text-white ml-2">{$selectedDAG.tasks}</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Description</h4>
					<p class="text-gray-300">{$selectedDAG.description}</p>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.airflow-controller {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.airflow-controller *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.airflow-controller *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.airflow-controller *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.airflow-controller *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>