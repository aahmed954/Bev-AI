<!--
Swarm Master Coordination Dashboard - Multi-Agent Orchestration
Connected to: src/agents/swarm_master.py (port 8100)
Features: Agent registration, task distribution, swarm monitoring, coordination modes
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const swarmState = writable({
		mode: 'hierarchical', // 'autonomous', 'hierarchical', 'collaborative', 'hybrid'
		agents: [],
		tasks: [],
		health: {
			overall: 'unknown',
			active_agents: 0,
			completed_tasks: 0,
			failed_tasks: 0,
			avg_response_time: 0
		},
		metrics: {
			task_throughput: 0,
			agent_utilization: 0,
			coordination_efficiency: 0,
			error_rate: 0
		}
	});
	
	const selectedAgent = writable(null);
	const selectedTask = writable(null);
	const viewMode = writable('overview'); // 'overview', 'agents', 'tasks', 'coordination', 'metrics'
	const isLoading = writable(false);
	
	// WebSocket for real-time swarm updates
	let ws: WebSocket | null = null;
	let metricsTimer: any = null;
	
	// Task submission form
	let newTask = {
		type: 'analysis',
		priority: 'medium',
		description: '',
		data: '',
		deadline: '',
		dependencies: []
	};
	
	// Agent registration form
	let newAgent = {
		name: '',
		type: 'analyzer',
		capabilities: [],
		max_concurrent_tasks: 5,
		resource_requirements: {
			cpu: 1,
			memory: 512,
			gpu: false
		}
	};
	
	const coordinationModes = [
		{ id: 'autonomous', label: 'Autonomous', description: 'Agents self-organize and coordinate' },
		{ id: 'hierarchical', label: 'Hierarchical', description: 'Tree-based command structure' },
		{ id: 'collaborative', label: 'Collaborative', description: 'Peer-to-peer coordination' },
		{ id: 'hybrid', label: 'Hybrid', description: 'Dynamic mode switching' }
	];
	
	const taskTypes = [
		'analysis', 'processing', 'monitoring', 'coordination', 'optimization',
		'research', 'validation', 'generation', 'classification', 'prediction'
	];
	
	const agentTypes = [
		'analyzer', 'processor', 'monitor', 'coordinator', 'optimizer',
		'researcher', 'validator', 'generator', 'classifier', 'predictor'
	];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadSwarmState();
		startMetricsCollection();
	});
	
	async function initializeWebSocket() {
		try {
			// Connect to the verified Swarm Master endpoint on port 8100
			ws = new WebSocket('ws://localhost:8100/ws/swarm');
			
			ws.onopen = () => {
				console.log('Swarm Master WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleSwarmUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Swarm WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Swarm WebSocket connection failed:', error);
		}
	}
	
	function handleSwarmUpdate(data: any) {
		swarmState.update(current => ({
			...current,
			...data
		}));
	}
	
	async function loadSwarmState() {
		isLoading.set(true);
		try {
			// Call the verified Swarm Master API endpoints
			const [statusResponse, agentsResponse, tasksResponse, healthResponse] = await Promise.all([
				fetch('http://localhost:8100/swarm/status'),
				fetch('http://localhost:8100/agents/list'),
				fetch('http://localhost:8100/tasks/list'),
				fetch('http://localhost:8100/health')
			]);
			
			const [status, agents, tasks, health] = await Promise.all([
				statusResponse.json(),
				agentsResponse.json(),
				tasksResponse.json(),
				healthResponse.json()
			]);
			
			swarmState.update(current => ({
				...current,
				mode: status.mode || current.mode,
				agents: agents || [],
				tasks: tasks || [],
				health: { ...current.health, ...health }
			}));
			
		} catch (error) {
			console.error('Failed to load swarm state:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function startMetricsCollection() {
		metricsTimer = setInterval(async () => {
			try {
				const response = await fetch('http://localhost:8100/metrics');
				const metrics = await response.json();
				
				swarmState.update(current => ({
					...current,
					metrics: { ...current.metrics, ...metrics }
				}));
			} catch (error) {
				console.error('Failed to collect metrics:', error);
			}
		}, 5000);
	}
	
	async function submitTask() {
		try {
			const response = await fetch('http://localhost:8100/tasks/submit', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(newTask)
			});
			
			const result = await response.json();
			
			// Add task to local state
			swarmState.update(current => ({
				...current,
				tasks: [...current.tasks, { ...newTask, id: result.task_id, status: 'pending' }]
			}));
			
			// Reset form
			newTask = {
				type: 'analysis',
				priority: 'medium',
				description: '',
				data: '',
				deadline: '',
				dependencies: []
			};
			
			dispatch('task_submitted', result);
		} catch (error) {
			console.error('Failed to submit task:', error);
		}
	}
	
	async function registerAgent() {
		try {
			const response = await fetch('http://localhost:8100/agents/register', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(newAgent)
			});
			
			const result = await response.json();
			
			// Add agent to local state
			swarmState.update(current => ({
				...current,
				agents: [...current.agents, { ...newAgent, id: result.agent_id, status: 'registered' }]
			}));
			
			// Reset form
			newAgent = {
				name: '',
				type: 'analyzer',
				capabilities: [],
				max_concurrent_tasks: 5,
				resource_requirements: {
					cpu: 1,
					memory: 512,
					gpu: false
				}
			};
			
			dispatch('agent_registered', result);
		} catch (error) {
			console.error('Failed to register agent:', error);
		}
	}
	
	async function changeCoordinationMode(mode: string) {
		try {
			const response = await fetch('http://localhost:8100/swarm/mode', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ mode })
			});
			
			const result = await response.json();
			
			swarmState.update(current => ({
				...current,
				mode: result.mode
			}));
			
			dispatch('mode_changed', result);
		} catch (error) {
			console.error('Failed to change coordination mode:', error);
		}
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'active':
			case 'healthy':
			case 'completed': return 'text-green-400';
			case 'pending':
			case 'running': return 'text-yellow-400';
			case 'failed':
			case 'error':
			case 'unhealthy': return 'text-red-400';
			case 'idle': return 'text-blue-400';
			default: return 'text-gray-400';
		}
	}
	
	function getPriorityColor(priority: string): string {
		switch (priority) {
			case 'high':
			case 'critical': return 'text-red-400';
			case 'medium': return 'text-yellow-400';
			case 'low': return 'text-green-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatUptime(uptime: number): string {
		if (!uptime) return 'N/A';
		const hours = Math.floor(uptime / 3600);
		const minutes = Math.floor((uptime % 3600) / 60);
		if (hours > 0) return `${hours}h ${minutes}m`;
		return `${minutes}m`;
	}
	
	function openAgentModal(agent: any) {
		selectedAgent.set(agent);
	}
	
	function openTaskModal(task: any) {
		selectedTask.set(task);
	}
</script>

<!-- Swarm Master Dashboard -->
<div class="swarm-dashboard h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-purple-400">Swarm Master Coordination</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					Mode: <span class="text-purple-400 capitalize">{$swarmState.mode}</span>
				</div>
				<div class="text-sm text-gray-400">
					{$swarmState.agents.length} agents | {$swarmState.tasks.length} tasks
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadSwarmState}
						class="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Swarm Health Overview -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-5 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold {getStatusColor($swarmState.health.overall)}">{$swarmState.health.overall}</div>
				<div class="text-sm text-gray-400">Health Status</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">{$swarmState.health.active_agents}</div>
				<div class="text-sm text-gray-400">Active Agents</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-blue-400">{$swarmState.health.completed_tasks}</div>
				<div class="text-sm text-gray-400">Completed Tasks</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-red-400">{$swarmState.health.failed_tasks}</div>
				<div class="text-sm text-gray-400">Failed Tasks</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-yellow-400">{$swarmState.health.avg_response_time}ms</div>
				<div class="text-sm text-gray-400">Avg Response</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Swarm Overview', icon: '🕸️' },
				{ id: 'agents', label: 'Agent Management', icon: '🤖' },
				{ id: 'tasks', label: 'Task Distribution', icon: '📋' },
				{ id: 'coordination', label: 'Coordination Modes', icon: '🎛️' },
				{ id: 'metrics', label: 'Performance Metrics', icon: '📊' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$viewMode === tab.id
							? 'border-purple-500 text-purple-400'
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
			<!-- Swarm Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Agent Status Grid -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-purple-400">Agent Status Matrix</h3>
					{#if $swarmState.agents.length === 0}
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">🤖</div>
							<p>No agents registered</p>
						</div>
					{:else}
						<div class="grid grid-cols-2 md:grid-cols-3 gap-3">
							{#each $swarmState.agents as agent}
								<div
									class="bg-gray-900 rounded p-3 cursor-pointer hover:bg-gray-700 transition-colors"
									on:click={() => openAgentModal(agent)}
								>
									<div class="flex items-center justify-between mb-2">
										<span class="font-medium text-white text-sm">{agent.name}</span>
										<span class="w-2 h-2 rounded-full {getStatusColor(agent.status)}"></span>
									</div>
									<div class="text-xs text-gray-400">{agent.type}</div>
									<div class="text-xs text-gray-400">{agent.current_tasks || 0}/{agent.max_concurrent_tasks} tasks</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
				
				<!-- Task Queue -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Task Queue</h3>
					{#if $swarmState.tasks.length === 0}
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">📋</div>
							<p>No active tasks</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $swarmState.tasks.slice(0, 8) as task}
								<div
									class="bg-gray-900 rounded p-3 cursor-pointer hover:bg-gray-700 transition-colors"
									on:click={() => openTaskModal(task)}
								>
									<div class="flex items-center justify-between mb-2">
										<span class="font-medium text-white text-sm">{task.description || task.type}</span>
										<div class="flex items-center space-x-2">
											<span class="text-xs {getPriorityColor(task.priority)}">{task.priority}</span>
											<span class="text-xs {getStatusColor(task.status)}">{task.status}</span>
										</div>
									</div>
									<div class="flex items-center justify-between text-xs text-gray-400">
										<span>Type: {task.type}</span>
										{#if task.assigned_agent}
											<span>Agent: {task.assigned_agent}</span>
										{/if}
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
				
				<!-- Coordination Status -->
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Coordination Status</h3>
					<div class="grid grid-cols-1 md:grid-cols-4 gap-4">
						<div class="bg-gray-900 rounded p-4 text-center">
							<div class="text-xl font-bold text-white">{($swarmState.metrics.task_throughput || 0).toFixed(1)}</div>
							<div class="text-xs text-gray-400">Tasks/Min</div>
						</div>
						<div class="bg-gray-900 rounded p-4 text-center">
							<div class="text-xl font-bold text-white">{($swarmState.metrics.agent_utilization || 0).toFixed(1)}%</div>
							<div class="text-xs text-gray-400">Agent Utilization</div>
						</div>
						<div class="bg-gray-900 rounded p-4 text-center">
							<div class="text-xl font-bold text-white">{($swarmState.metrics.coordination_efficiency || 0).toFixed(1)}%</div>
							<div class="text-xs text-gray-400">Coordination Efficiency</div>
						</div>
						<div class="bg-gray-900 rounded p-4 text-center">
							<div class="text-xl font-bold text-white">{($swarmState.metrics.error_rate || 0).toFixed(2)}%</div>
							<div class="text-xs text-gray-400">Error Rate</div>
						</div>
					</div>
				</div>
			</div>
			
		{:else if $viewMode === 'agents'}
			<!-- Agent Management -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Agent Registration -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-purple-400">Register New Agent</h3>
					<div class="space-y-4">
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Agent Name</label>
							<input
								type="text"
								bind:value={newAgent.name}
								placeholder="Enter agent name"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
							/>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Agent Type</label>
							<select
								bind:value={newAgent.type}
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
							>
								{#each agentTypes as type}
									<option value={type}>{type}</option>
								{/each}
							</select>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Max Concurrent Tasks</label>
							<input
								type="number"
								bind:value={newAgent.max_concurrent_tasks}
								min="1"
								max="20"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
							/>
						</div>
						
						<button
							on:click={registerAgent}
							class="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded font-medium transition-colors"
							disabled={!newAgent.name}
						>
							Register Agent
						</button>
					</div>
				</div>
				
				<!-- Agent List -->
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Registered Agents</h3>
					{#if $swarmState.agents.length === 0}
						<div class="text-center py-12 text-gray-400">
							<div class="text-4xl mb-4">🤖</div>
							<p>No agents registered</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $swarmState.agents as agent}
								<div class="bg-gray-900 rounded p-4">
									<div class="flex items-center justify-between mb-3">
										<div class="flex items-center space-x-3">
											<h4 class="font-medium text-white">{agent.name}</h4>
											<span class="px-2 py-1 rounded text-xs {getStatusColor(agent.status)} bg-gray-800">
												{agent.status}
											</span>
											<span class="text-sm text-gray-400 capitalize">{agent.type}</span>
										</div>
										<div class="text-sm text-gray-400">
											{agent.current_tasks || 0}/{agent.max_concurrent_tasks} tasks
										</div>
									</div>
									
									{#if agent.capabilities && agent.capabilities.length > 0}
										<div class="flex flex-wrap gap-2 mb-2">
											{#each agent.capabilities as capability}
												<span class="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded">
													{capability}
												</span>
											{/each}
										</div>
									{/if}
									
									<div class="grid grid-cols-3 gap-4 text-sm">
										<div>
											<span class="text-gray-400">Uptime:</span>
											<span class="text-white ml-2">{formatUptime(agent.uptime)}</span>
										</div>
										<div>
											<span class="text-gray-400">Completed:</span>
											<span class="text-white ml-2">{agent.completed_tasks || 0}</span>
										</div>
										<div>
											<span class="text-gray-400">Success Rate:</span>
											<span class="text-white ml-2">{(agent.success_rate || 0).toFixed(1)}%</span>
										</div>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $viewMode === 'tasks'}
			<!-- Task Distribution -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Task Submission -->
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
									<option value={type}>{type}</option>
								{/each}
							</select>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Priority</label>
							<select
								bind:value={newTask.priority}
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							>
								<option value="low">Low</option>
								<option value="medium">Medium</option>
								<option value="high">High</option>
								<option value="critical">Critical</option>
							</select>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Description</label>
							<textarea
								bind:value={newTask.description}
								placeholder="Enter task description"
								rows="3"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							></textarea>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Task Data</label>
							<textarea
								bind:value={newTask.data}
								placeholder="Enter task data (JSON, text, etc.)"
								rows="2"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							></textarea>
						</div>
						
						<button
							on:click={submitTask}
							class="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded font-medium transition-colors"
							disabled={!newTask.description}
						>
							Submit Task
						</button>
					</div>
				</div>
				
				<!-- Task List -->
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Task Queue</h3>
					{#if $swarmState.tasks.length === 0}
						<div class="text-center py-12 text-gray-400">
							<div class="text-4xl mb-4">📋</div>
							<p>No tasks in queue</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $swarmState.tasks as task}
								<div class="bg-gray-900 rounded p-4">
									<div class="flex items-center justify-between mb-3">
										<div class="flex items-center space-x-3">
											<h4 class="font-medium text-white">{task.description || task.type}</h4>
											<span class="px-2 py-1 rounded text-xs {getPriorityColor(task.priority)} bg-gray-800">
												{task.priority}
											</span>
											<span class="px-2 py-1 rounded text-xs {getStatusColor(task.status)} bg-gray-800">
												{task.status}
											</span>
										</div>
										<div class="text-sm text-gray-400">
											{task.type}
										</div>
									</div>
									
									<div class="grid grid-cols-2 gap-4 text-sm">
										{#if task.assigned_agent}
											<div>
												<span class="text-gray-400">Assigned:</span>
												<span class="text-white ml-2">{task.assigned_agent}</span>
											</div>
										{/if}
										{#if task.progress}
											<div>
												<span class="text-gray-400">Progress:</span>
												<span class="text-white ml-2">{task.progress}%</span>
											</div>
										{/if}
										{#if task.created_at}
											<div>
												<span class="text-gray-400">Created:</span>
												<span class="text-white ml-2">{new Date(task.created_at).toLocaleString()}</span>
											</div>
										{/if}
										{#if task.deadline}
											<div>
												<span class="text-gray-400">Deadline:</span>
												<span class="text-white ml-2">{new Date(task.deadline).toLocaleString()}</span>
											</div>
										{/if}
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $viewMode === 'coordination'}
			<!-- Coordination Modes -->
			<div class="space-y-6">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-purple-400">Coordination Mode Selection</h3>
					<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
						{#each coordinationModes as mode}
							<div
								class="bg-gray-900 rounded p-4 cursor-pointer transition-colors {
									$swarmState.mode === mode.id
										? 'border-2 border-purple-500'
										: 'border-2 border-transparent hover:border-gray-600'
								}"
								on:click={() => changeCoordinationMode(mode.id)}
							>
								<h4 class="font-medium text-white mb-2">{mode.label}</h4>
								<p class="text-sm text-gray-400">{mode.description}</p>
								{#if $swarmState.mode === mode.id}
									<div class="mt-2 text-xs text-purple-400">● Active</div>
								{/if}
							</div>
						{/each}
					</div>
				</div>
				
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Current Mode: {$swarmState.mode}</h3>
					<div class="bg-gray-900 rounded p-4">
						<p class="text-gray-300">
							{coordinationModes.find(m => m.id === $swarmState.mode)?.description || 'Unknown mode'}
						</p>
					</div>
				</div>
			</div>
			
		{:else if $viewMode === 'metrics'}
			<!-- Performance Metrics -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Performance Metrics</h3>
					<div class="space-y-4">
						<div class="flex items-center justify-between">
							<span class="text-gray-400">Task Throughput</span>
							<span class="text-white">{($swarmState.metrics.task_throughput || 0).toFixed(1)} tasks/min</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-gray-400">Agent Utilization</span>
							<span class="text-white">{($swarmState.metrics.agent_utilization || 0).toFixed(1)}%</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-gray-400">Coordination Efficiency</span>
							<span class="text-white">{($swarmState.metrics.coordination_efficiency || 0).toFixed(1)}%</span>
						</div>
						<div class="flex items-center justify-between">
							<span class="text-gray-400">Error Rate</span>
							<span class="text-white">{($swarmState.metrics.error_rate || 0).toFixed(2)}%</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">System Health</h3>
					<div class="text-center py-8 text-gray-400">
						<div class="text-3xl mb-2">📊</div>
						<p>Real-time metrics visualization</p>
						<p class="text-sm mt-2">Performance charts will be displayed here</p>
					</div>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Connected to Swarm Master on port 8100
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('export_swarm_config')}
					class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm font-medium transition-colors"
				>
					Export Config
				</button>
				<button
					on:click={() => dispatch('emergency_stop_swarm')}
					class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
				>
					Emergency Stop
				</button>
			</div>
		</div>
	</div>
</div>

<!-- Agent Detail Modal -->
{#if $selectedAgent}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedAgent.set(null)}>
		<div class="max-w-2xl w-full mx-4 bg-gray-800 rounded-lg p-6" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-purple-400">{$selectedAgent.name}</h3>
				<button
					on:click={() => selectedAgent.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Agent Details</h4>
					<div class="grid grid-cols-2 gap-4 text-sm">
						<div>
							<span class="text-gray-400">Type:</span>
							<span class="text-white ml-2">{$selectedAgent.type}</span>
						</div>
						<div>
							<span class="text-gray-400">Status:</span>
							<span class="text-white ml-2 {getStatusColor($selectedAgent.status)}">{$selectedAgent.status}</span>
						</div>
						<div>
							<span class="text-gray-400">Max Tasks:</span>
							<span class="text-white ml-2">{$selectedAgent.max_concurrent_tasks}</span>
						</div>
						<div>
							<span class="text-gray-400">Current Tasks:</span>
							<span class="text-white ml-2">{$selectedAgent.current_tasks || 0}</span>
						</div>
					</div>
				</div>
				
				{#if $selectedAgent.capabilities && $selectedAgent.capabilities.length > 0}
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-2">Capabilities</h4>
						<div class="flex flex-wrap gap-2">
							{#each $selectedAgent.capabilities as capability}
								<span class="px-2 py-1 bg-purple-600 text-white text-xs rounded">
									{capability}
								</span>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.swarm-dashboard {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.swarm-dashboard *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.swarm-dashboard *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.swarm-dashboard *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.swarm-dashboard *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>