<!--
Oracle Worker System - Specialized Worker Coordination & Management
Connected to: src/oracle/workers/ (specialized worker coordination system)
Features: Worker management, task distribution, performance monitoring, resource allocation
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const oracleState = writable({
		workers: [],
		active_workers: 0,
		total_tasks: 0,
		completed_tasks: 0,
		failed_tasks: 0,
		worker_pool_utilization: 0,
		performance_metrics: {
			avg_task_completion: 0,
			worker_efficiency: 0,
			resource_utilization: 0,
			error_rate: 0
		}
	});
	
	const selectedTab = writable('overview'); // 'overview', 'workers', 'tasks', 'performance', 'config'
	const selectedWorker = writable(null);
	const isLoading = writable(false);
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	
	// Worker types based on oracle/workers/ directory
	const workerTypes = [
		{
			id: 'drm_researcher',
			name: 'DRM Researcher',
			icon: '🔐',
			description: 'Digital Rights Management research and analysis',
			specialization: 'DRM analysis, protection research, bypass detection'
		},
		{
			id: 'watermark_analyzer',
			name: 'Watermark Analyzer',
			icon: '🏷️',
			description: 'Digital watermark detection and analysis',
			specialization: 'Image/video watermarks, steganography, forensic analysis'
		},
		{
			id: 'crypto_researcher',
			name: 'Crypto Researcher',
			icon: '₿',
			description: 'Cryptocurrency research and blockchain analysis',
			specialization: 'Blockchain forensics, wallet clustering, transaction flow'
		}
	];
	
	// Task queue data
	let taskQueue: any[] = [];
	let workerMetrics = {
		total_workers: 0,
		active_workers: 0,
		idle_workers: 0,
		overloaded_workers: 0,
		avg_cpu_usage: 0,
		avg_memory_usage: 0
	};
	
	// New task form
	let newTask = {
		type: 'drm_analysis',
		priority: 'normal',
		payload: '',
		worker_preference: 'auto',
		timeout: 300,
		retry_count: 3
	};
	
	const taskTypes = [
		'drm_analysis', 'watermark_detection', 'crypto_research',
		'blockchain_analysis', 'steganography_detection', 'forensic_analysis'
	];
	
	const priorities = ['low', 'normal', 'high', 'urgent'];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadOracleSystem();
		startWorkerMonitoring();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/oracle-workers');
			
			ws.onopen = () => {
				console.log('Oracle Worker System WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleOracleUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Oracle WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Oracle WebSocket connection failed:', error);
		}
	}
	
	function handleOracleUpdate(data: any) {
		switch (data.type) {
			case 'worker_status_update':
				updateWorkerStatus(data.worker_id, data.status);
				break;
			case 'task_assigned':
				updateTaskStatus(data.task_id, 'assigned', data.worker_id);
				break;
			case 'task_completed':
				completeTask(data.task_id, data.result);
				break;
			case 'task_failed':
				failTask(data.task_id, data.error);
				break;
			case 'worker_metrics':
				updateWorkerMetrics(data.metrics);
				break;
		}
	}
	
	async function loadOracleSystem() {
		isLoading.set(true);
		try {
			const [workers, tasks, metrics] = await Promise.all([
				invoke('get_oracle_workers'),
				invoke('get_oracle_tasks'),
				invoke('get_oracle_metrics')
			]);
			
			oracleState.update(current => ({
				...current,
				workers: workers || [],
				active_workers: (workers || []).filter(w => w.status === 'active').length,
				performance_metrics: { ...current.performance_metrics, ...metrics }
			}));
			
			taskQueue = tasks || [];
			workerMetrics = { ...workerMetrics, ...metrics };
		} catch (error) {
			console.error('Failed to load Oracle system:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function startWorkerMonitoring() {
		setInterval(async () => {
			try {
				const metrics = await invoke('get_worker_metrics');
				updateWorkerMetrics(metrics);
			} catch (error) {
				console.error('Worker monitoring failed:', error);
			}
		}, 5000);
	}
	
	async function submitTask() {
		try {
			const result = await invoke('submit_oracle_task', { task: newTask });
			
			taskQueue = [{
				id: result.task_id,
				...newTask,
				status: 'queued',
				submitted_at: new Date().toISOString()
			}, ...taskQueue];
			
			// Reset form
			newTask = {
				type: 'drm_analysis',
				priority: 'normal',
				payload: '',
				worker_preference: 'auto',
				timeout: 300,
				retry_count: 3
			};
			
			dispatch('task_submitted', result);
		} catch (error) {
			console.error('Failed to submit task:', error);
		}
	}
	
	async function scaleWorkerPool(workerType: string, count: number) {
		try {
			await invoke('scale_oracle_workers', { workerType, count });
			await loadOracleSystem(); // Refresh worker list
		} catch (error) {
			console.error('Failed to scale worker pool:', error);
		}
	}
	
	function updateWorkerStatus(workerId: string, status: any) {
		oracleState.update(current => ({
			...current,
			workers: current.workers.map(worker =>
				worker.id === workerId ? { ...worker, ...status } : worker
			)
		}));
	}
	
	function updateTaskStatus(taskId: string, status: string, workerId?: string) {
		taskQueue = taskQueue.map(task =>
			task.id === taskId 
				? { ...task, status, assigned_worker: workerId, updated_at: new Date().toISOString() }
				: task
		);
	}
	
	function completeTask(taskId: string, result: any) {
		taskQueue = taskQueue.map(task =>
			task.id === taskId 
				? { ...task, status: 'completed', result, completed_at: new Date().toISOString() }
				: task
		);
		
		oracleState.update(current => ({
			...current,
			completed_tasks: current.completed_tasks + 1
		}));
	}
	
	function failTask(taskId: string, error: any) {
		taskQueue = taskQueue.map(task =>
			task.id === taskId 
				? { ...task, status: 'failed', error, failed_at: new Date().toISOString() }
				: task
		);
		
		oracleState.update(current => ({
			...current,
			failed_tasks: current.failed_tasks + 1
		}));
	}
	
	function updateWorkerMetrics(metrics: any) {
		workerMetrics = { ...workerMetrics, ...metrics };
		
		oracleState.update(current => ({
			...current,
			worker_pool_utilization: metrics.worker_pool_utilization || current.worker_pool_utilization,
			performance_metrics: { ...current.performance_metrics, ...metrics.performance }
		}));
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'active':
			case 'working':
			case 'completed': return 'text-green-400';
			case 'queued':
			case 'assigned': return 'text-yellow-400';
			case 'failed':
			case 'error': return 'text-red-400';
			case 'idle':
			case 'paused': return 'text-gray-400';
			default: return 'text-gray-400';
		}
	}
	
	function getPriorityColor(priority: string): string {
		switch (priority) {
			case 'urgent': return 'text-red-400';
			case 'high': return 'text-orange-400';
			case 'normal': return 'text-yellow-400';
			case 'low': return 'text-green-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleString();
	}
	
	function openWorkerModal(worker: any) {
		selectedWorker.set(worker);
	}
</script>

<!-- Oracle Worker System -->
<div class="oracle-system h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-indigo-400">🔮 Oracle Worker System</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{$oracleState.active_workers}/{$oracleState.workers.length} workers active
				</div>
				<div class="text-sm text-gray-400">
					{taskQueue.length} tasks queued
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-indigo-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadOracleSystem}
						class="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Worker Pool Status -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-indigo-400">{workerMetrics.total_workers}</div>
				<div class="text-sm text-gray-400">Total Workers</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">{workerMetrics.active_workers}</div>
				<div class="text-sm text-gray-400">Active Workers</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-blue-400">{workerMetrics.idle_workers}</div>
				<div class="text-sm text-gray-400">Idle Workers</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-red-400">{workerMetrics.overloaded_workers}</div>
				<div class="text-sm text-gray-400">Overloaded</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Worker Overview', icon: '📊' },
				{ id: 'workers', label: 'Worker Management', icon: '👥' },
				{ id: 'tasks', label: 'Task Queue', icon: '📋' },
				{ id: 'performance', label: 'Performance', icon: '📈' },
				{ id: 'config', label: 'Configuration', icon: '⚙️' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedTab === tab.id
							? 'border-indigo-500 text-indigo-400'
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
			<!-- Oracle Worker Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Worker Pool Status -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-indigo-400">Worker Pool Status</h3>
					<div class="space-y-4">
						{#each workerTypes as workerType}
							{@const typeWorkers = $oracleState.workers.filter(w => w.type === workerType.id)}
							<div class="bg-gray-900 rounded p-4">
								<div class="flex items-center justify-between mb-2">
									<div class="flex items-center space-x-2">
										<span class="text-xl">{workerType.icon}</span>
										<span class="font-medium text-white">{workerType.name}</span>
									</div>
									<div class="text-sm text-gray-400">
										{typeWorkers.filter(w => w.status === 'active').length}/{typeWorkers.length} active
									</div>
								</div>
								<p class="text-gray-400 text-xs">{workerType.description}</p>
								
								<div class="mt-2 flex justify-between items-center">
									<button
										on:click={() => scaleWorkerPool(workerType.id, typeWorkers.length + 1)}
										class="px-2 py-1 bg-green-600 hover:bg-green-700 rounded text-xs transition-colors"
									>
										+ Scale Up
									</button>
									{#if typeWorkers.length > 1}
										<button
											on:click={() => scaleWorkerPool(workerType.id, typeWorkers.length - 1)}
											class="px-2 py-1 bg-red-600 hover:bg-red-700 rounded text-xs transition-colors"
										>
											- Scale Down
										</button>
									{/if}
								</div>
							</div>
						{/each}
					</div>
				</div>
				
				<!-- Task Queue Summary -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Task Queue Summary</h3>
					<div class="space-y-4">
						<div class="grid grid-cols-2 gap-4 text-center">
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-yellow-400">{taskQueue.filter(t => t.status === 'queued').length}</div>
								<div class="text-xs text-gray-400">Queued</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-blue-400">{taskQueue.filter(t => t.status === 'assigned' || t.status === 'working').length}</div>
								<div class="text-xs text-gray-400">Processing</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-green-400">{$oracleState.completed_tasks}</div>
								<div class="text-xs text-gray-400">Completed</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-red-400">{$oracleState.failed_tasks}</div>
								<div class="text-xs text-gray-400">Failed</div>
							</div>
						</div>
						
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">Performance Metrics</h4>
							<div class="space-y-2 text-sm">
								<div class="flex justify-between">
									<span class="text-gray-400">Avg Completion:</span>
									<span class="text-white">{($oracleState.performance_metrics.avg_task_completion || 0).toFixed(1)}s</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Worker Efficiency:</span>
									<span class="text-white">{($oracleState.performance_metrics.worker_efficiency || 0).toFixed(1)}%</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Error Rate:</span>
									<span class="text-white">{($oracleState.performance_metrics.error_rate || 0).toFixed(2)}%</span>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'workers'}
			<!-- Worker Management -->
			<div class="space-y-6">
				<!-- Worker Grid -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-indigo-400">Worker Status Matrix</h3>
					{#if $oracleState.workers.length === 0}
						<div class="text-center py-12 text-gray-400">
							<div class="text-4xl mb-4">👥</div>
							<p>No workers available</p>
						</div>
					{:else}
						<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
							{#each $oracleState.workers as worker}
								<div 
									class="bg-gray-900 rounded p-4 cursor-pointer hover:bg-gray-700 transition-colors"
									on:click={() => openWorkerModal(worker)}
								>
									<div class="flex items-center justify-between mb-3">
										<div class="flex items-center space-x-2">
											<span class="text-xl">{workerTypes.find(t => t.id === worker.type)?.icon || '⚙️'}</span>
											<span class="font-medium text-white text-sm">{worker.name || worker.id}</span>
										</div>
										<div class="flex items-center space-x-1">
											<div class="w-2 h-2 rounded-full {getStatusColor(worker.status)}"></div>
											<span class="text-xs {getStatusColor(worker.status)}">{worker.status}</span>
										</div>
									</div>
									
									<div class="space-y-1 text-xs">
										<div class="flex justify-between">
											<span class="text-gray-400">Type:</span>
											<span class="text-white">{worker.type}</span>
										</div>
										<div class="flex justify-between">
											<span class="text-gray-400">Tasks:</span>
											<span class="text-white">{worker.current_tasks || 0}/{worker.max_tasks || 5}</span>
										</div>
										<div class="flex justify-between">
											<span class="text-gray-400">CPU:</span>
											<span class="text-white">{(worker.cpu_usage || 0).toFixed(1)}%</span>
										</div>
										<div class="flex justify-between">
											<span class="text-gray-400">Memory:</span>
											<span class="text-white">{(worker.memory_usage || 0).toFixed(1)}%</span>
										</div>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'tasks'}
			<!-- Task Management -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Submit New Task -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Submit New Task</h3>
					<div class="space-y-4">
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Task Type</label>
							<select
								bind:value={newTask.type}
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							>
								{#each taskTypes as type}
									<option value={type}>{type.replace('_', ' ')}</option>
								{/each}
							</select>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Priority</label>
							<select
								bind:value={newTask.priority}
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							>
								{#each priorities as priority}
									<option value={priority}>{priority}</option>
								{/each}
							</select>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Task Payload</label>
							<textarea
								bind:value={newTask.payload}
								placeholder="Enter task data (URL, file path, etc.)"
								rows="4"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							></textarea>
						</div>
						
						<div class="grid grid-cols-2 gap-2">
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Timeout (s)</label>
								<input
									type="number"
									bind:value={newTask.timeout}
									min="10"
									max="3600"
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
								/>
							</div>
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Retries</label>
								<input
									type="number"
									bind:value={newTask.retry_count}
									min="0"
									max="10"
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
								/>
							</div>
						</div>
						
						<button
							on:click={submitTask}
							class="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded font-medium transition-colors"
							disabled={!newTask.payload.trim()}
						>
							Submit Task
						</button>
					</div>
				</div>
				
				<!-- Task Queue -->
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Task Queue</h3>
					{#if taskQueue.length === 0}
						<div class="text-center py-12 text-gray-400">
							<div class="text-4xl mb-4">📋</div>
							<p>No tasks in queue</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each taskQueue.slice(0, 10) as task}
								<div class="bg-gray-900 rounded p-4">
									<div class="flex items-center justify-between mb-3">
										<div class="flex items-center space-x-3">
											<h4 class="font-medium text-white text-sm">Task #{task.id}</h4>
											<span class="px-2 py-1 rounded text-xs {getStatusColor(task.status)} bg-gray-800">
												{task.status}
											</span>
											<span class="px-2 py-1 rounded text-xs {getPriorityColor(task.priority)} bg-gray-800">
												{task.priority}
											</span>
										</div>
										<div class="text-sm text-gray-400">
											{task.type.replace('_', ' ')}
										</div>
									</div>
									
									<div class="text-sm text-gray-300 mb-2">
										{task.payload.length > 100 ? task.payload.slice(0, 100) + '...' : task.payload}
									</div>
									
									<div class="grid grid-cols-3 gap-4 text-xs">
										<div>
											<span class="text-gray-400">Submitted:</span>
											<span class="text-white ml-1">{formatDate(task.submitted_at)}</span>
										</div>
										{#if task.assigned_worker}
											<div>
												<span class="text-gray-400">Worker:</span>
												<span class="text-white ml-1">{task.assigned_worker}</span>
											</div>
										{/if}
										{#if task.status === 'completed' && task.completed_at}
											<div>
												<span class="text-gray-400">Completed:</span>
												<span class="text-white ml-1">{formatDate(task.completed_at)}</span>
											</div>
										{/if}
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'performance'}
			<!-- Performance Monitoring -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">System Performance</h3>
					<div class="space-y-4">
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Worker Pool Utilization:</span>
							<span class="text-white">{($oracleState.worker_pool_utilization || 0).toFixed(1)}%</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Average CPU Usage:</span>
							<span class="text-white">{(workerMetrics.avg_cpu_usage || 0).toFixed(1)}%</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Average Memory Usage:</span>
							<span class="text-white">{(workerMetrics.avg_memory_usage || 0).toFixed(1)}%</span>
						</div>
						<div class="flex justify-between items-center">
							<span class="text-gray-400">Task Success Rate:</span>
							<span class="text-green-400">
								{$oracleState.completed_tasks > 0 ? (($oracleState.completed_tasks / ($oracleState.completed_tasks + $oracleState.failed_tasks)) * 100).toFixed(1) : 0}%
							</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Performance Charts</h3>
					<div class="text-center py-8 text-gray-400">
						<div class="text-3xl mb-2">📈</div>
						<p>Performance visualization</p>
						<p class="text-sm mt-2">Worker utilization and task completion trends</p>
					</div>
				</div>
			</div>
			
		{:else}
			<!-- Other tab interfaces -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-white">
					{$selectedTab === 'config' ? '⚙️ Oracle Configuration' : '🔮 Oracle System'}
				</h3>
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">🔮</div>
					<p>Oracle Worker System interface</p>
					<p class="text-sm mt-2">Specialized worker coordination and management</p>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Oracle Worker System | {$oracleState.active_workers} workers active
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('export_worker_metrics')}
					class="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded text-sm font-medium transition-colors"
				>
					Export Metrics
				</button>
				<button
					on:click={() => dispatch('optimize_worker_pool')}
					class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium transition-colors"
				>
					Optimize Pool
				</button>
			</div>
		</div>
	</div>
</div>

<!-- Worker Detail Modal -->
{#if $selectedWorker}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedWorker.set(null)}>
		<div class="max-w-3xl w-full mx-4 bg-gray-800 rounded-lg p-6" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-indigo-400">
					{workerTypes.find(t => t.id === $selectedWorker.type)?.icon || '⚙️'} {$selectedWorker.name || $selectedWorker.id}
				</h3>
				<button
					on:click={() => selectedWorker.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Worker Information</h4>
					<div class="grid grid-cols-2 gap-4 text-sm">
						<div>
							<span class="text-gray-400">Type:</span>
							<span class="text-white ml-2">{$selectedWorker.type}</span>
						</div>
						<div>
							<span class="text-gray-400">Status:</span>
							<span class="text-white ml-2 {getStatusColor($selectedWorker.status)}">{$selectedWorker.status}</span>
						</div>
						<div>
							<span class="text-gray-400">Current Tasks:</span>
							<span class="text-white ml-2">{$selectedWorker.current_tasks || 0}</span>
						</div>
						<div>
							<span class="text-gray-400">Max Tasks:</span>
							<span class="text-white ml-2">{$selectedWorker.max_tasks || 5}</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Resource Usage</h4>
					<div class="space-y-2 text-sm">
						<div class="flex justify-between">
							<span class="text-gray-400">CPU Usage:</span>
							<span class="text-white">{($selectedWorker.cpu_usage || 0).toFixed(1)}%</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Memory Usage:</span>
							<span class="text-white">{($selectedWorker.memory_usage || 0).toFixed(1)}%</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Tasks Completed:</span>
							<span class="text-white">{$selectedWorker.tasks_completed || 0}</span>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.oracle-system {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Oracle-themed styling */
	:global(.oracle-system .oracle-active) {
		animation: pulse-indigo 2s infinite;
	}
	
	@keyframes pulse-indigo {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.8; }
	}
	
	/* Custom scrollbar */
	:global(.oracle-system *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.oracle-system *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.oracle-system *::-webkit-scrollbar-thumb) {
		background: #6366f1;
		border-radius: 3px;
	}
	
	:global(.oracle-system *::-webkit-scrollbar-thumb:hover) {
		background: #818cf8;
	}
</style>