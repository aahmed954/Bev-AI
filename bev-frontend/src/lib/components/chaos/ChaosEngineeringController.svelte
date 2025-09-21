<!--
Comprehensive Chaos Engineering Controller - Fault Injection & Resilience Testing
Connected to: src/testing/chaos_api.py
Features: Controlled fault injection, service targeting, experiment scheduling, impact analysis
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const chaosState = writable({
		experiments: [],
		active_experiments: 0,
		available_services: [],
		fault_types: [],
		system_health: {
			overall: 'healthy',
			affected_services: 0,
			recovery_time: 0,
			resilience_score: 95
		},
		experiment_history: []
	});
	
	const selectedTab = writable('overview'); // 'overview', 'experiments', 'builder', 'monitoring', 'reports'
	const selectedExperiment = writable(null);
	const isLoading = writable(false);
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	
	// Experiment builder form
	let newExperiment = {
		name: '',
		description: '',
		target_services: [],
		fault_type: 'latency',
		severity: 'low',
		duration: '5m',
		schedule: 'manual',
		rollback_conditions: []
	};
	
	const faultTypes = [
		{
			id: 'latency',
			name: 'Network Latency',
			description: 'Inject network delays and timeouts',
			icon: '🐌',
			parameters: ['delay_ms', 'jitter_ms', 'timeout_ms']
		},
		{
			id: 'cpu_stress',
			name: 'CPU Stress',
			description: 'High CPU load simulation',
			icon: '🔥',
			parameters: ['cpu_percent', 'duration', 'cores']
		},
		{
			id: 'memory_stress',
			name: 'Memory Exhaustion',
			description: 'Memory pressure and OOM simulation',
			icon: '💾',
			parameters: ['memory_mb', 'duration', 'oom_kill']
		},
		{
			id: 'disk_fill',
			name: 'Disk Space Exhaustion',
			description: 'Fill disk space to test handling',
			icon: '💽',
			parameters: ['fill_mb', 'path', 'duration']
		},
		{
			id: 'service_kill',
			name: 'Service Termination',
			description: 'Sudden service process termination',
			icon: '💀',
			parameters: ['signal', 'restart_delay']
		},
		{
			id: 'network_partition',
			name: 'Network Partition',
			description: 'Simulate network splits and isolations',
			icon: '🔌',
			parameters: ['partition_type', 'affected_nodes']
		},
		{
			id: 'dns_chaos',
			name: 'DNS Failures',
			description: 'DNS resolution failures and delays',
			icon: '🌐',
			parameters: ['failure_rate', 'delay_ms', 'domains']
		}
	];
	
	const severityLevels = [
		{ id: 'low', label: 'Low Impact', color: 'text-green-400' },
		{ id: 'medium', label: 'Medium Impact', color: 'text-yellow-400' },
		{ id: 'high', label: 'High Impact', color: 'text-red-400' },
		{ id: 'critical', label: 'Critical Impact', color: 'text-red-600' }
	];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadChaosSystem();
		await loadAvailableServices();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/chaos-engineering');
			
			ws.onopen = () => {
				console.log('Chaos Engineering WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleChaosUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Chaos WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Chaos WebSocket connection failed:', error);
		}
	}
	
	function handleChaosUpdate(data: any) {
		switch (data.type) {
			case 'experiment_started':
				addExperiment(data.experiment);
				break;
			case 'experiment_completed':
				updateExperiment(data.experiment_id, data.result);
				break;
			case 'system_health_update':
				updateSystemHealth(data.health);
				break;
			case 'service_impact':
				recordServiceImpact(data.service, data.impact);
				break;
		}
	}
	
	async function loadChaosSystem() {
		isLoading.set(true);
		try {
			const [experiments, history, health] = await Promise.all([
				invoke('get_chaos_experiments'),
				invoke('get_chaos_experiment_history'),
				invoke('get_system_health')
			]);
			
			chaosState.update(current => ({
				...current,
				experiments: experiments || [],
				experiment_history: history || [],
				system_health: { ...current.system_health, ...health },
				active_experiments: (experiments || []).filter(e => e.status === 'running').length
			}));
		} catch (error) {
			console.error('Failed to load chaos system:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	async function loadAvailableServices() {
		try {
			const services = await invoke('get_available_services_for_chaos');
			chaosState.update(current => ({
				...current,
				available_services: services || [
					{ name: 'bev-postgres', type: 'database' },
					{ name: 'bev-neo4j', type: 'database' },
					{ name: 'kafka-1', type: 'messaging' },
					{ name: 'rabbitmq-1', type: 'messaging' },
					{ name: 'nginx-proxy', type: 'infrastructure' },
					{ name: 'prometheus', type: 'monitoring' },
					{ name: 'swarm-master', type: 'ai' },
					{ name: 'social-analyzer', type: 'osint' }
				],
				fault_types: faultTypes
			}));
		} catch (error) {
			console.error('Failed to load available services:', error);
		}
	}
	
	async function startChaosExperiment() {
		try {
			const result = await invoke('start_chaos_experiment', { experiment: newExperiment });
			
			addExperiment({
				id: result.experiment_id,
				...newExperiment,
				status: 'running',
				started_at: new Date().toISOString()
			});
			
			// Reset form
			newExperiment = {
				name: '',
				description: '',
				target_services: [],
				fault_type: 'latency',
				severity: 'low',
				duration: '5m',
				schedule: 'manual',
				rollback_conditions: []
			};
			
			dispatch('experiment_started', result);
		} catch (error) {
			console.error('Failed to start chaos experiment:', error);
		}
	}
	
	async function stopChaosExperiment(experimentId: string) {
		try {
			await invoke('stop_chaos_experiment', { experimentId });
			updateExperiment(experimentId, { status: 'stopped', stopped_at: new Date().toISOString() });
		} catch (error) {
			console.error('Failed to stop chaos experiment:', error);
		}
	}
	
	function addExperiment(experiment: any) {
		chaosState.update(current => ({
			...current,
			experiments: [experiment, ...current.experiments],
			active_experiments: current.active_experiments + 1
		}));
	}
	
	function updateExperiment(experimentId: string, update: any) {
		chaosState.update(current => ({
			...current,
			experiments: current.experiments.map(exp =>
				exp.id === experimentId ? { ...exp, ...update } : exp
			),
			active_experiments: update.status === 'completed' || update.status === 'stopped'
				? Math.max(0, current.active_experiments - 1)
				: current.active_experiments
		}));
	}
	
	function updateSystemHealth(health: any) {
		chaosState.update(current => ({
			...current,
			system_health: { ...current.system_health, ...health }
		}));
	}
	
	function recordServiceImpact(service: string, impact: any) {
		// Record service impact for analysis
		dispatch('service_impact', { service, impact });
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'running': return 'text-yellow-400';
			case 'completed': return 'text-green-400';
			case 'failed':
			case 'error': return 'text-red-400';
			case 'stopped': return 'text-gray-400';
			default: return 'text-gray-400';
		}
	}
	
	function getSeverityColor(severity: string): string {
		const level = severityLevels.find(s => s.id === severity);
		return level ? level.color : 'text-gray-400';
	}
	
	function getHealthColor(health: string): string {
		switch (health) {
			case 'healthy': return 'text-green-400';
			case 'degraded': return 'text-yellow-400';
			case 'critical': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleString();
	}
	
	function formatDuration(ms: number): string {
		const minutes = Math.floor(ms / 60000);
		const hours = Math.floor(minutes / 60);
		if (hours > 0) return `${hours}h ${minutes % 60}m`;
		return `${minutes}m`;
	}
	
	function openExperimentModal(experiment: any) {
		selectedExperiment.set(experiment);
	}
</script>

<!-- Chaos Engineering Controller -->
<div class="chaos-controller h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-red-400">Chaos Engineering Control</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{$chaosState.active_experiments} active | {$chaosState.experiments.length} total
				</div>
				<div class="flex items-center space-x-2">
					<div class="w-2 h-2 rounded-full {getHealthColor($chaosState.system_health.overall)}"></div>
					<span class="text-sm text-gray-400">System: {$chaosState.system_health.overall}</span>
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-red-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadChaosSystem}
						class="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- System Health Overview -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold {getHealthColor($chaosState.system_health.overall)}">
					{$chaosState.system_health.overall.toUpperCase()}
				</div>
				<div class="text-sm text-gray-400">System Health</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-orange-400">{$chaosState.system_health.affected_services}</div>
				<div class="text-sm text-gray-400">Affected Services</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-blue-400">{formatDuration($chaosState.system_health.recovery_time)}</div>
				<div class="text-sm text-gray-400">Recovery Time</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">{$chaosState.system_health.resilience_score}%</div>
				<div class="text-sm text-gray-400">Resilience Score</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Experiment Overview', icon: '📊' },
				{ id: 'builder', label: 'Experiment Builder', icon: '🔧' },
				{ id: 'experiments', label: 'Active Experiments', icon: '⚡' },
				{ id: 'monitoring', label: 'Impact Monitoring', icon: '📈' },
				{ id: 'reports', label: 'Resilience Reports', icon: '📄' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedTab === tab.id
							? 'border-red-500 text-red-400'
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
			<!-- Experiment Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Active Experiments -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-red-400">Active Chaos Experiments</h3>
					{#if $chaosState.experiments.filter(e => e.status === 'running').length === 0}
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">⚡</div>
							<p>No active experiments</p>
							<p class="text-sm mt-2">System is stable</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $chaosState.experiments.filter(e => e.status === 'running') as experiment}
								<div class="bg-red-900/20 border border-red-800 rounded p-3">
									<div class="flex items-center justify-between mb-2">
										<span class="font-medium text-white text-sm">{experiment.name}</span>
										<div class="flex items-center space-x-2">
											<span class="text-xs {getStatusColor(experiment.status)}">{experiment.status}</span>
											<span class="text-xs {getSeverityColor(experiment.severity)}">{experiment.severity}</span>
										</div>
									</div>
									<div class="text-sm text-gray-300 mb-1">{experiment.description}</div>
									<div class="flex items-center justify-between text-xs">
										<span class="text-gray-400">
											{faultTypes.find(f => f.id === experiment.fault_type)?.icon} {experiment.fault_type}
										</span>
										<span class="text-gray-400">
											{experiment.target_services.length} services
										</span>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
				
				<!-- Fault Type Library -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-orange-400">Available Fault Types</h3>
					<div class="space-y-3">
						{#each faultTypes as faultType}
							<div class="bg-gray-900 rounded p-3">
								<div class="flex items-center space-x-3 mb-2">
									<span class="text-xl">{faultType.icon}</span>
									<div class="flex-1">
										<h4 class="font-medium text-white text-sm">{faultType.name}</h4>
										<p class="text-xs text-gray-400">{faultType.description}</p>
									</div>
								</div>
								<div class="flex flex-wrap gap-1">
									{#each faultType.parameters as param}
										<span class="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded">
											{param}
										</span>
									{/each}
								</div>
							</div>
						{/each}
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'builder'}
			<!-- Experiment Builder -->
			<div class="max-w-4xl mx-auto">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-6 text-red-400">Chaos Experiment Builder</h3>
					
					<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
						<!-- Experiment Configuration -->
						<div class="space-y-4">
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Experiment Name</label>
								<input
									type="text"
									bind:value={newExperiment.name}
									placeholder="Enter experiment name"
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
								/>
							</div>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Description</label>
								<textarea
									bind:value={newExperiment.description}
									placeholder="Describe the chaos experiment"
									rows="3"
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
								></textarea>
							</div>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Fault Type</label>
								<select
									bind:value={newExperiment.fault_type}
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
								>
									{#each faultTypes as faultType}
										<option value={faultType.id}>{faultType.icon} {faultType.name}</option>
									{/each}
								</select>
							</div>
							
							<div class="grid grid-cols-2 gap-4">
								<div>
									<label class="block text-sm font-medium text-gray-300 mb-2">Severity</label>
									<select
										bind:value={newExperiment.severity}
										class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
									>
										{#each severityLevels as level}
											<option value={level.id}>{level.label}</option>
										{/each}
									</select>
								</div>
								<div>
									<label class="block text-sm font-medium text-gray-300 mb-2">Duration</label>
									<input
										type="text"
										bind:value={newExperiment.duration}
										placeholder="5m"
										class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-red-500"
									/>
								</div>
							</div>
						</div>
						
						<!-- Target Services -->
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Target Services</label>
							<div class="bg-gray-900 rounded p-4 max-h-80 overflow-y-auto">
								{#each $chaosState.available_services as service}
									<label class="flex items-center mb-2">
										<input
											type="checkbox"
											bind:group={newExperiment.target_services}
											value={service.name}
											class="mr-2"
										/>
										<span class="text-white text-sm">{service.name}</span>
										<span class="text-gray-400 text-xs ml-2">({service.type})</span>
									</label>
								{/each}
							</div>
							
							<div class="mt-4 text-sm text-gray-400">
								Selected: {newExperiment.target_services.length} services
							</div>
						</div>
					</div>
					
					<div class="mt-6 flex justify-end space-x-4">
						<button
							class="px-6 py-2 bg-gray-600 hover:bg-gray-700 rounded transition-colors"
						>
							Save as Template
						</button>
						<button
							on:click={startChaosExperiment}
							class="px-6 py-2 bg-red-600 hover:bg-red-700 rounded font-medium transition-colors"
							disabled={!newExperiment.name || newExperiment.target_services.length === 0}
						>
							⚡ Start Chaos Experiment
						</button>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'experiments'}
			<!-- Active Experiments -->
			<div class="space-y-4">
				{#if $chaosState.experiments.length === 0}
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">⚡</div>
						<p>No chaos experiments</p>
					</div>
				{:else}
					{#each $chaosState.experiments as experiment}
						<div class="bg-gray-800 rounded-lg p-6">
							<div class="flex items-center justify-between mb-4">
								<div class="flex items-center space-x-4">
									<h3 class="text-lg font-medium text-white">{experiment.name}</h3>
									<span class="px-2 py-1 rounded text-xs {getStatusColor(experiment.status)} bg-gray-700">
										{experiment.status}
									</span>
									<span class="px-2 py-1 rounded text-xs {getSeverityColor(experiment.severity)} bg-gray-700">
										{experiment.severity}
									</span>
									<span class="text-sm text-gray-400">
										{faultTypes.find(f => f.id === experiment.fault_type)?.icon} {experiment.fault_type}
									</span>
								</div>
								<div class="flex items-center space-x-2">
									{#if experiment.status === 'running'}
										<button
											on:click={() => stopChaosExperiment(experiment.id)}
											class="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors"
										>
											Stop
										</button>
									{/if}
									<button
										on:click={() => openExperimentModal(experiment)}
										class="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
									>
										Details
									</button>
								</div>
							</div>
							
							<div class="mb-4">
								<p class="text-gray-300">{experiment.description}</p>
							</div>
							
							<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
								<div>
									<span class="text-gray-400">Target Services:</span>
									<span class="text-white ml-2">{experiment.target_services?.length || 0}</span>
								</div>
								<div>
									<span class="text-gray-400">Duration:</span>
									<span class="text-white ml-2">{experiment.duration}</span>
								</div>
								<div>
									<span class="text-gray-400">Started:</span>
									<span class="text-white ml-2">{experiment.started_at ? formatDate(experiment.started_at) : 'N/A'}</span>
								</div>
								{#if experiment.impact_score}
									<div>
										<span class="text-gray-400">Impact Score:</span>
										<span class="text-white ml-2">{experiment.impact_score}/10</span>
									</div>
								{/if}
							</div>
						</div>
					{/each}
				{/if}
			</div>
			
		{:else if $selectedTab === 'monitoring'}
			<!-- Impact Monitoring -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-yellow-400">Real-time Impact</h3>
					<div class="text-center py-8 text-gray-400">
						<div class="text-3xl mb-2">📈</div>
						<p>Service health monitoring</p>
						<p class="text-sm mt-2">Real-time metrics during chaos experiments</p>
					</div>
				</div>
				
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Recovery Analysis</h3>
					<div class="text-center py-8 text-gray-400">
						<div class="text-3xl mb-2">🔄</div>
						<p>Recovery time analysis</p>
						<p class="text-sm mt-2">System resilience and recovery patterns</p>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'reports'}
			<!-- Resilience Reports -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-blue-400">Resilience Reports</h3>
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">📄</div>
					<p>Chaos engineering reports</p>
					<p class="text-sm mt-2">System resilience analysis and recommendations</p>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Chaos Engineering | System Health: {$chaosState.system_health.overall}
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('emergency_stop_all')}
					class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
				>
					🚨 Emergency Stop All
				</button>
				<button
					on:click={() => dispatch('export_chaos_report')}
					class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
				>
					Export Report
				</button>
			</div>
		</div>
	</div>
</div>

<!-- Experiment Detail Modal -->
{#if $selectedExperiment}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedExperiment.set(null)}>
		<div class="max-w-3xl w-full mx-4 bg-gray-800 rounded-lg p-6" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-red-400">{$selectedExperiment.name}</h3>
				<button
					on:click={() => selectedExperiment.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Experiment Details</h4>
					<div class="grid grid-cols-2 gap-4 text-sm">
						<div>
							<span class="text-gray-400">Status:</span>
							<span class="text-white ml-2 {getStatusColor($selectedExperiment.status)}">{$selectedExperiment.status}</span>
						</div>
						<div>
							<span class="text-gray-400">Fault Type:</span>
							<span class="text-white ml-2">{$selectedExperiment.fault_type}</span>
						</div>
						<div>
							<span class="text-gray-400">Severity:</span>
							<span class="text-white ml-2">{$selectedExperiment.severity}</span>
						</div>
						<div>
							<span class="text-gray-400">Duration:</span>
							<span class="text-white ml-2">{$selectedExperiment.duration}</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Target Services</h4>
					<div class="flex flex-wrap gap-2">
						{#each $selectedExperiment.target_services || [] as service}
							<span class="px-2 py-1 bg-red-600 text-white text-xs rounded">
								{service}
							</span>
						{/each}
					</div>
				</div>
				
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Description</h4>
					<p class="text-gray-300">{$selectedExperiment.description}</p>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.chaos-controller {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.chaos-controller *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.chaos-controller *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.chaos-controller *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.chaos-controller *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>