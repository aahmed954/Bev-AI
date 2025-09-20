<!--
Container Orchestration Dashboard - Docker Compose Service Management
Connected to: docker-compose.complete.yml (70+ services)
Features: Service health matrix, container controls, resource monitoring, log streaming
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	// Sub-components
	import DockerManager from './DockerManager.svelte';
	import ResourceAllocation from './ResourceAllocation.svelte';
	import LogStreamer from './LogStreamer.svelte';
	import ServiceDependencyGraph from './ServiceDependencyGraph.svelte';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const services = writable([]);
	const selectedService = writable(null);
	const viewMode = writable('overview'); // 'overview', 'services', 'resources', 'logs', 'dependencies'
	const isLoading = writable(false);
	const refreshInterval = writable(5000); // 5 seconds
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	let refreshTimer: any = null;
	
	// Service categories based on docker-compose.complete.yml
	const serviceCategories = {
		'databases': ['postgres', 'neo4j', 'redis-node-1', 'redis-node-2', 'redis-node-3', 'elasticsearch', 'influxdb', 'mongodb'],
		'messaging': ['kafka-1', 'kafka-2', 'kafka-3', 'zookeeper-1', 'zookeeper-2', 'zookeeper-3', 'rabbitmq-1', 'rabbitmq-2'],
		'analytics': ['qdrant', 'weaviate', 'prometheus', 'grafana', 'kibana'],
		'security': ['tor-relay', 'tor-bridge', 'guardian-security', 'intrusion-detection'],
		'ai-ml': ['swarm-master', 'extended-reasoning', 'ml-pipeline', 'genetic-optimizer'],
		'infrastructure': ['nginx-proxy', 'haproxy', 'minio-1', 'minio-2', 'minio-3', 'consul'],
		'monitoring': ['health-monitor', 'metrics-collector', 'alert-system', 'log-aggregator'],
		'osint': ['breach-analyzer', 'darknet-monitor', 'crypto-tracker', 'social-analyzer'],
		'chaos': ['chaos-engineer', 'fault-injector', 'resilience-tester'],
		'edge': ['edge-us-east', 'edge-us-west', 'edge-eu-central', 'edge-asia-pacific']
	};
	
	let serviceStats = {
		total: 0,
		running: 0,
		stopped: 0,
		unhealthy: 0,
		restarting: 0
	};
	
	onMount(async () => {
		await initializeWebSocket();
		await loadServices();
		startPeriodicRefresh();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/container-management');
			
			ws.onopen = () => {
				console.log('Container Management WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleWebSocketMessage(data);
			};
			
			ws.onclose = () => {
				console.log('WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('WebSocket connection failed:', error);
		}
	}
	
	function handleWebSocketMessage(data: any) {
		switch (data.type) {
			case 'service_status_update':
				updateServiceStatus(data.service, data.status);
				break;
			case 'resource_update':
				updateResourceMetrics(data.metrics);
				break;
			case 'log_update':
				appendServiceLogs(data.service, data.logs);
				break;
			case 'bulk_status_update':
				services.set(data.services);
				calculateStats(data.services);
				break;
		}
	}
	
	async function loadServices() {
		isLoading.set(true);
		try {
			const result = await invoke('get_docker_services');
			services.set(result);
			calculateStats(result);
		} catch (error) {
			console.error('Failed to load services:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function calculateStats(serviceList: any[]) {
		const stats = {
			total: serviceList.length,
			running: serviceList.filter(s => s.status === 'running').length,
			stopped: serviceList.filter(s => s.status === 'exited' || s.status === 'stopped').length,
			unhealthy: serviceList.filter(s => s.health === 'unhealthy').length,
			restarting: serviceList.filter(s => s.status === 'restarting').length
		};
		serviceStats = stats;
	}
	
	function updateServiceStatus(serviceName: string, status: any) {
		services.update(current => {
			const index = current.findIndex(s => s.name === serviceName);
			if (index !== -1) {
				current[index] = { ...current[index], ...status };
			}
			return current;
		});
	}
	
	function updateResourceMetrics(metrics: any) {
		services.update(current => {
			const index = current.findIndex(s => s.name === metrics.service);
			if (index !== -1) {
				current[index].resources = metrics;
			}
			return current;
		});
	}
	
	function appendServiceLogs(serviceName: string, logs: string[]) {
		// Handle log updates for log streaming component
		dispatch('logs_updated', { service: serviceName, logs });
	}
	
	function startPeriodicRefresh() {
		refreshTimer = setInterval(async () => {
			if (ws && ws.readyState === WebSocket.OPEN) {
				ws.send(JSON.stringify({ type: 'refresh_status' }));
			} else {
				await loadServices();
			}
		}, $refreshInterval);
	}
	
	async function restartService(serviceName: string) {
		try {
			await invoke('restart_docker_service', { serviceName });
			dispatch('service_action', { action: 'restart', service: serviceName });
		} catch (error) {
			console.error('Failed to restart service:', error);
		}
	}
	
	async function stopService(serviceName: string) {
		try {
			await invoke('stop_docker_service', { serviceName });
			dispatch('service_action', { action: 'stop', service: serviceName });
		} catch (error) {
			console.error('Failed to stop service:', error);
		}
	}
	
	async function startService(serviceName: string) {
		try {
			await invoke('start_docker_service', { serviceName });
			dispatch('service_action', { action: 'start', service: serviceName });
		} catch (error) {
			console.error('Failed to start service:', error);
		}
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'running': return 'text-green-400';
			case 'exited':
			case 'stopped': return 'text-red-400';
			case 'restarting': return 'text-yellow-400';
			case 'paused': return 'text-blue-400';
			default: return 'text-gray-400';
		}
	}
	
	function getHealthColor(health: string): string {
		switch (health) {
			case 'healthy': return 'text-green-400';
			case 'unhealthy': return 'text-red-400';
			case 'starting': return 'text-yellow-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatUptime(uptime: number): string {
		if (!uptime) return 'N/A';
		const days = Math.floor(uptime / 86400);
		const hours = Math.floor((uptime % 86400) / 3600);
		const minutes = Math.floor((uptime % 3600) / 60);
		
		if (days > 0) return `${days}d ${hours}h`;
		if (hours > 0) return `${hours}h ${minutes}m`;
		return `${minutes}m`;
	}
	
	function formatBytes(bytes: number): string {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
	}
	
	function filterServicesByCategory(category: string): any[] {
		const categoryServices = serviceCategories[category] || [];
		return $services.filter(service => 
			categoryServices.some(catService => 
				service.name.includes(catService) || service.service.includes(catService)
			)
		);
	}
</script>

<!-- Container Dashboard -->
<div class="container-dashboard h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-blue-400">Container Orchestration</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{serviceStats.total} services | {serviceStats.running} running
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadServices}
						class="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Service Statistics -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-5 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-white">{serviceStats.total}</div>
				<div class="text-sm text-gray-400">Total Services</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">{serviceStats.running}</div>
				<div class="text-sm text-gray-400">Running</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-red-400">{serviceStats.stopped}</div>
				<div class="text-sm text-gray-400">Stopped</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-yellow-400">{serviceStats.restarting}</div>
				<div class="text-sm text-gray-400">Restarting</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-orange-400">{serviceStats.unhealthy}</div>
				<div class="text-sm text-gray-400">Unhealthy</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Service Overview', icon: '📊' },
				{ id: 'services', label: 'Service Management', icon: '🔧' },
				{ id: 'resources', label: 'Resource Allocation', icon: '💾' },
				{ id: 'logs', label: 'Log Streaming', icon: '📝' },
				{ id: 'dependencies', label: 'Dependencies', icon: '🕸️' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$viewMode === tab.id
							? 'border-blue-500 text-blue-400'
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
			<!-- Service Overview by Category -->
			<div class="space-y-6">
				{#each Object.entries(serviceCategories) as [category, categoryServices]}
					{@const categoryServiceList = filterServicesByCategory(category)}
					{#if categoryServiceList.length > 0}
						<div class="bg-gray-800 rounded-lg p-6">
							<h3 class="text-lg font-semibold mb-4 capitalize text-blue-400">
								{category.replace('-', ' ')} Services
							</h3>
							<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
								{#each categoryServiceList as service}
									<div class="bg-gray-900 rounded p-4 cursor-pointer hover:bg-gray-700 transition-colors"
										 on:click={() => selectedService.set(service)}>
										<div class="flex items-center justify-between mb-2">
											<h4 class="font-medium text-white truncate">{service.name}</h4>
											<span class="text-xs px-2 py-1 rounded {getStatusColor(service.status)} bg-gray-800">
												{service.status}
											</span>
										</div>
										
										<div class="space-y-1 text-sm">
											{#if service.health}
												<div class="flex justify-between">
													<span class="text-gray-400">Health:</span>
													<span class="{getHealthColor(service.health)}">{service.health}</span>
												</div>
											{/if}
											<div class="flex justify-between">
												<span class="text-gray-400">Uptime:</span>
												<span class="text-white">{formatUptime(service.uptime)}</span>
											</div>
											{#if service.resources}
												<div class="flex justify-between">
													<span class="text-gray-400">CPU:</span>
													<span class="text-white">{(service.resources.cpu_percent || 0).toFixed(1)}%</span>
												</div>
												<div class="flex justify-between">
													<span class="text-gray-400">Memory:</span>
													<span class="text-white">{formatBytes(service.resources.memory_usage || 0)}</span>
												</div>
											{/if}
										</div>
									</div>
								{/each}
							</div>
						</div>
					{/if}
				{/each}
			</div>
			
		{:else if $viewMode === 'services'}
			<DockerManager
				services={$services}
				on:restart={(e) => restartService(e.detail)}
				on:stop={(e) => stopService(e.detail)}
				on:start={(e) => startService(e.detail)}
			/>
			
		{:else if $viewMode === 'resources'}
			<ResourceAllocation services={$services} />
			
		{:else if $viewMode === 'logs'}
			<LogStreamer 
				services={$services}
				selectedService={$selectedService}
			/>
			
		{:else if $viewMode === 'dependencies'}
			<ServiceDependencyGraph services={$services} />
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Last updated: {new Date().toLocaleTimeString()}
			</div>
			<div class="flex space-x-2">
				<select
					bind:value={$refreshInterval}
					class="bg-gray-800 border border-gray-700 rounded px-3 py-1 text-white text-sm"
				>
					<option value={1000}>1s refresh</option>
					<option value={5000}>5s refresh</option>
					<option value={10000}>10s refresh</option>
					<option value={30000}>30s refresh</option>
				</select>
				<button
					on:click={() => dispatch('export_metrics')}
					class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium transition-colors"
				>
					Export Metrics
				</button>
				<button
					on:click={() => dispatch('emergency_stop')}
					class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
				>
					Emergency Stop
				</button>
			</div>
		</div>
	</div>
</div>

<!-- Service Detail Modal -->
{#if $selectedService}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedService.set(null)}>
		<div class="max-w-4xl w-full mx-4 bg-gray-800 rounded-lg p-6 max-h-[90vh] overflow-y-auto" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-blue-400">{$selectedService.name}</h3>
				<button
					on:click={() => selectedService.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Service Info -->
				<div class="space-y-4">
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-3">Service Information</h4>
						<div class="space-y-2 text-sm">
							<div class="flex justify-between">
								<span class="text-gray-400">Status:</span>
								<span class="{getStatusColor($selectedService.status)}">{$selectedService.status}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Health:</span>
								<span class="{getHealthColor($selectedService.health || 'unknown')}">{$selectedService.health || 'Unknown'}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Uptime:</span>
								<span class="text-white">{formatUptime($selectedService.uptime)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Image:</span>
								<span class="text-white text-xs">{$selectedService.image || 'N/A'}</span>
							</div>
						</div>
					</div>
					
					<!-- Resource Usage -->
					{#if $selectedService.resources}
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-3">Resource Usage</h4>
							<div class="space-y-3">
								<div>
									<div class="flex justify-between text-sm mb-1">
										<span class="text-gray-400">CPU Usage</span>
										<span class="text-white">{($selectedService.resources.cpu_percent || 0).toFixed(1)}%</span>
									</div>
									<div class="w-full bg-gray-700 rounded-full h-2">
										<div
											class="bg-blue-600 h-2 rounded-full transition-all"
											style="width: {$selectedService.resources.cpu_percent || 0}%"
										></div>
									</div>
								</div>
								
								<div>
									<div class="flex justify-between text-sm mb-1">
										<span class="text-gray-400">Memory Usage</span>
										<span class="text-white">{formatBytes($selectedService.resources.memory_usage || 0)}</span>
									</div>
									<div class="w-full bg-gray-700 rounded-full h-2">
										<div
											class="bg-green-600 h-2 rounded-full transition-all"
											style="width: {($selectedService.resources.memory_percent || 0)}%"
										></div>
									</div>
								</div>
							</div>
						</div>
					{/if}
				</div>
				
				<!-- Service Actions -->
				<div class="space-y-4">
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-3">Service Actions</h4>
						<div class="grid grid-cols-2 gap-2">
							<button
								on:click={() => startService($selectedService.name)}
								class="px-3 py-2 bg-green-600 hover:bg-green-700 rounded text-sm transition-colors"
								disabled={$selectedService.status === 'running'}
							>
								Start
							</button>
							<button
								on:click={() => stopService($selectedService.name)}
								class="px-3 py-2 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors"
								disabled={$selectedService.status === 'stopped'}
							>
								Stop
							</button>
							<button
								on:click={() => restartService($selectedService.name)}
								class="px-3 py-2 bg-yellow-600 hover:bg-yellow-700 rounded text-sm transition-colors"
							>
								Restart
							</button>
							<button
								class="px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
							>
								Logs
							</button>
						</div>
					</div>
					
					<!-- Environment Variables -->
					{#if $selectedService.env}
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-3">Environment Variables</h4>
							<div class="space-y-1 text-xs max-h-32 overflow-y-auto">
								{#each Object.entries($selectedService.env) as [key, value]}
									<div class="flex justify-between">
										<span class="text-gray-400">{key}:</span>
										<span class="text-white truncate ml-2">{value}</span>
									</div>
								{/each}
							</div>
						</div>
					{/if}
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.container-dashboard {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.container-dashboard *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.container-dashboard *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.container-dashboard *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.container-dashboard *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>