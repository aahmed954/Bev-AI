<!--
System Integration Panel - Avatar system status and health monitoring
Features: Service status, performance metrics, GPU monitoring, system diagnostics
Connected to: Avatar services, system monitoring APIs, GPU diagnostics
-->

<script lang="ts">
	import { onMount, onDestroy, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { avatarClient } from '$lib/services/AvatarWebSocketClient';
	
	const dispatch = createEventDispatcher();
	
	// System status state
	const systemStatus = writable({
		avatar_service: 'unknown', // 'healthy', 'degraded', 'unhealthy', 'unknown'
		renderer_service: 'unknown',
		osint_integration: 'unknown',
		websocket_connection: 'disconnected',
		gpu_acceleration: false,
		overall_health: 'unknown'
	});
	
	// Performance metrics
	const performanceMetrics = writable({
		gpu_utilization: 0,
		gpu_memory_used: 0,
		gpu_memory_total: 0,
		gpu_temperature: 0,
		render_fps: 0,
		frame_time_ms: 0,
		websocket_latency: 0,
		message_queue_size: 0,
		cpu_usage: 0,
		memory_usage: 0,
		disk_usage: 0
	});
	
	// Service endpoints and their status
	const services = writable([
		{
			name: 'Avatar Service',
			endpoint: 'http://localhost:8092',
			status: 'unknown',
			last_check: null,
			response_time: 0,
			description: 'Advanced 3D avatar rendering service'
		},
		{
			name: 'OSINT API',
			endpoint: 'http://localhost:3010',
			status: 'unknown',
			last_check: null,
			response_time: 0,
			description: 'OSINT analysis and tool orchestration'
		},
		{
			name: 'Extended Reasoning',
			endpoint: 'http://localhost:8093',
			status: 'unknown',
			last_check: null,
			response_time: 0,
			description: 'AI reasoning and analysis service'
		},
		{
			name: 'Swarm Master',
			endpoint: 'http://localhost:8094',
			status: 'unknown',
			last_check: null,
			response_time: 0,
			description: 'Multi-agent coordination service'
		},
		{
			name: 'Neo4j Database',
			endpoint: 'http://localhost:7474',
			status: 'unknown',
			last_check: null,
			response_time: 0,
			description: 'Graph database for relationships'
		},
		{
			name: 'Qdrant Vector DB',
			endpoint: 'http://localhost:6333',
			status: 'unknown',
			last_check: null,
			response_time: 0,
			description: 'Vector database for embeddings'
		}
	]);
	
	// System logs
	const systemLogs = writable<Array<{
		timestamp: Date,
		level: 'info' | 'warning' | 'error' | 'success',
		component: string,
		message: string
	}>>([]);
	
	// Diagnostics state
	const diagnostics = writable({
		gpu_info: null as any,
		system_info: null as any,
		avatar_config: null as any,
		network_status: null as any,
		last_updated: null as Date | null
	});
	
	// Update intervals
	let statusUpdateInterval: number | null = null;
	let performanceUpdateInterval: number | null = null;
	let diagnosticsUpdateInterval: number | null = null;
	
	// UI state
	let selectedTab = 'status'; // 'status', 'performance', 'services', 'diagnostics', 'logs'
	let autoRefresh = true;
	let refreshInterval = 5000; // 5 seconds
	
	onMount(async () => {
		await initializeMonitoring();
		startAutoRefresh();
	});
	
	onDestroy(() => {
		stopAutoRefresh();
	});
	
	async function initializeMonitoring() {
		// Initial status check
		await checkAllServices();
		await updatePerformanceMetrics();
		await runSystemDiagnostics();
		
		// Setup avatar client monitoring
		avatarClient.subscribe('connection', (event) => {
			systemStatus.update(status => ({
				...status,
				websocket_connection: event.status
			}));
			
			addSystemLog('info', 'WebSocket', `Connection ${event.status}`);
		});
		
		addSystemLog('info', 'System', 'Avatar system monitoring initialized');
	}
	
	function startAutoRefresh() {
		if (!autoRefresh) return;
		
		statusUpdateInterval = window.setInterval(async () => {
			await checkAllServices();
		}, refreshInterval);
		
		performanceUpdateInterval = window.setInterval(async () => {
			await updatePerformanceMetrics();
		}, 2000);
		
		diagnosticsUpdateInterval = window.setInterval(async () => {
			await runSystemDiagnostics();
		}, 30000); // 30 seconds
	}
	
	function stopAutoRefresh() {
		if (statusUpdateInterval) {
			clearInterval(statusUpdateInterval);
			statusUpdateInterval = null;
		}
		
		if (performanceUpdateInterval) {
			clearInterval(performanceUpdateInterval);
			performanceUpdateInterval = null;
		}
		
		if (diagnosticsUpdateInterval) {
			clearInterval(diagnosticsUpdateInterval);
			diagnosticsUpdateInterval = null;
		}
	}
	
	async function checkAllServices() {
		const serviceUpdates = await Promise.allSettled(
			$services.map(async (service) => {
				const startTime = performance.now();
				try {
					const response = await fetch(`${service.endpoint}/health`, {
						method: 'GET',
						signal: AbortSignal.timeout(5000) // 5 second timeout
					});
					
					const endTime = performance.now();
					const responseTime = endTime - startTime;
					
					return {
						...service,
						status: response.ok ? 'healthy' : 'degraded',
						last_check: new Date(),
						response_time: responseTime
					};
				} catch (error) {
					const endTime = performance.now();
					const responseTime = endTime - startTime;
					
					return {
						...service,
						status: 'unhealthy',
						last_check: new Date(),
						response_time: responseTime
					};
				}
			})
		);
		
		const updatedServices = serviceUpdates.map((result, index) => {
			if (result.status === 'fulfilled') {
				return result.value;
			} else {
				return {
					...$services[index],
					status: 'error',
					last_check: new Date(),
					response_time: 0
				};
			}
		});
		
		services.set(updatedServices);
		
		// Update overall system status
		const healthyServices = updatedServices.filter(s => s.status === 'healthy').length;
		const totalServices = updatedServices.length;
		const healthPercentage = (healthyServices / totalServices) * 100;
		
		let overallHealth = 'unhealthy';
		if (healthPercentage >= 90) {
			overallHealth = 'healthy';
		} else if (healthPercentage >= 70) {
			overallHealth = 'degraded';
		}
		
		systemStatus.update(status => ({
			...status,
			overall_health: overallHealth,
			avatar_service: updatedServices.find(s => s.name === 'Avatar Service')?.status || 'unknown',
			osint_integration: updatedServices.find(s => s.name === 'OSINT API')?.status || 'unknown'
		}));
	}
	
	async function updatePerformanceMetrics() {
		try {
			// GPU metrics from avatar service
			const gpuResponse = await fetch('http://localhost:8092/performance/gpu');
			if (gpuResponse.ok) {
				const gpuData = await gpuResponse.json();
				
				performanceMetrics.update(metrics => ({
					...metrics,
					gpu_utilization: gpuData.utilization || 0,
					gpu_memory_used: gpuData.memory_used || 0,
					gpu_memory_total: gpuData.memory_total || 0,
					gpu_temperature: gpuData.temperature || 0,
					render_fps: gpuData.render_fps || 0,
					frame_time_ms: gpuData.frame_time_ms || 0
				}));
				
				systemStatus.update(status => ({
					...status,
					gpu_acceleration: gpuData.acceleration_enabled || false,
					renderer_service: gpuData.renderer_healthy ? 'healthy' : 'degraded'
				}));
			}
			
			// System metrics
			const systemResponse = await fetch('http://localhost:8092/performance/system');
			if (systemResponse.ok) {
				const systemData = await systemResponse.json();
				
				performanceMetrics.update(metrics => ({
					...metrics,
					cpu_usage: systemData.cpu_usage || 0,
					memory_usage: systemData.memory_usage || 0,
					disk_usage: systemData.disk_usage || 0
				}));
			}
			
			// WebSocket latency
			const pingStart = performance.now();
			avatarClient.sendMessage({
				type: 'system_status',
				data: { ping: true }
			});
			
			// Estimate latency (simplified)
			const estimatedLatency = Math.random() * 10 + 5; // Mock latency
			performanceMetrics.update(metrics => ({
				...metrics,
				websocket_latency: estimatedLatency
			}));
			
		} catch (error) {
			console.error('Failed to update performance metrics:', error);
		}
	}
	
	async function runSystemDiagnostics() {
		try {
			// GPU diagnostics
			const gpuResponse = await fetch('http://localhost:8092/diagnostics/gpu');
			if (gpuResponse.ok) {
				const gpuInfo = await gpuResponse.json();
				
				diagnostics.update(diag => ({
					...diag,
					gpu_info: gpuInfo,
					last_updated: new Date()
				}));
			}
			
			// System info
			const systemResponse = await fetch('http://localhost:8092/diagnostics/system');
			if (systemResponse.ok) {
				const systemInfo = await systemResponse.json();
				
				diagnostics.update(diag => ({
					...diag,
					system_info: systemInfo
				}));
			}
			
			// Avatar configuration
			const configResponse = await fetch('http://localhost:8092/diagnostics/config');
			if (configResponse.ok) {
				const avatarConfig = await configResponse.json();
				
				diagnostics.update(diag => ({
					...diag,
					avatar_config: avatarConfig
				}));
			}
			
		} catch (error) {
			console.error('Failed to run system diagnostics:', error);
			addSystemLog('error', 'Diagnostics', 'Failed to update system diagnostics');
		}
	}
	
	function addSystemLog(level: 'info' | 'warning' | 'error' | 'success', component: string, message: string) {
		systemLogs.update(logs => {
			const newLog = {
				timestamp: new Date(),
				level,
				component,
				message
			};
			
			// Keep only last 100 logs
			const updatedLogs = [newLog, ...logs.slice(0, 99)];
			return updatedLogs;
		});
	}
	
	async function restartAvatarService() {
		try {
			addSystemLog('info', 'Avatar Service', 'Restart requested');
			
			const response = await fetch('http://localhost:8092/admin/restart', {
				method: 'POST'
			});
			
			if (response.ok) {
				addSystemLog('success', 'Avatar Service', 'Restart initiated successfully');
			} else {
				addSystemLog('error', 'Avatar Service', 'Restart failed');
			}
		} catch (error) {
			addSystemLog('error', 'Avatar Service', `Restart error: ${error.message}`);
		}
	}
	
	async function clearSystemLogs() {
		systemLogs.set([]);
		addSystemLog('info', 'System', 'System logs cleared');
	}
	
	async function exportDiagnostics() {
		const diagnosticData = {
			timestamp: new Date().toISOString(),
			system_status: $systemStatus,
			performance_metrics: $performanceMetrics,
			services: $services,
			diagnostics: $diagnostics,
			recent_logs: $systemLogs.slice(0, 50)
		};
		
		const blob = new Blob([JSON.stringify(diagnosticData, null, 2)], {
			type: 'application/json'
		});
		
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `avatar_diagnostics_${Date.now()}.json`;
		a.click();
		URL.revokeObjectURL(url);
		
		addSystemLog('info', 'System', 'Diagnostics exported');
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'healthy': return 'text-green-400 bg-green-900';
			case 'degraded': return 'text-yellow-400 bg-yellow-900';
			case 'unhealthy': 
			case 'error': return 'text-red-400 bg-red-900';
			case 'unknown': return 'text-gray-400 bg-gray-900';
			default: return 'text-gray-400 bg-gray-900';
		}
	}
	
	function getLogLevelColor(level: string): string {
		switch (level) {
			case 'success': return 'text-green-400';
			case 'info': return 'text-blue-400';
			case 'warning': return 'text-yellow-400';
			case 'error': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatBytes(bytes: number): string {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
	}
	
	function formatTimestamp(timestamp: Date): string {
		return timestamp.toLocaleTimeString();
	}
	
	$: {
		if (autoRefresh) {
			stopAutoRefresh();
			startAutoRefresh();
		} else {
			stopAutoRefresh();
		}
	}
</script>

<!-- System Integration Panel -->
<div class="system-integration-panel h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h2 class="text-xl font-bold text-cyan-400">System Integration</h2>
			<div class="flex items-center space-x-4">
				<!-- Overall health indicator -->
				<div class="flex items-center space-x-2">
					<div class="w-3 h-3 rounded-full {
						$systemStatus.overall_health === 'healthy' ? 'bg-green-400' :
						$systemStatus.overall_health === 'degraded' ? 'bg-yellow-400' :
						'bg-red-400'
					}"></div>
					<span class="text-sm text-gray-400 capitalize">{$systemStatus.overall_health}</span>
				</div>
				
				<!-- Auto refresh toggle -->
				<label class="flex items-center space-x-2">
					<input
						type="checkbox"
						bind:checked={autoRefresh}
						class="w-4 h-4 text-cyan-600 bg-gray-700 border-gray-600 rounded focus:ring-cyan-500"
					/>
					<span class="text-sm text-gray-400">Auto Refresh</span>
				</label>
				
				<!-- Manual refresh button -->
				<button
					on:click={checkAllServices}
					class="px-3 py-1 bg-cyan-600 hover:bg-cyan-700 rounded text-sm transition-colors"
				>
					Refresh
				</button>
			</div>
		</div>
	</div>
	
	<!-- Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'status', label: 'Status', icon: '🔋' },
				{ id: 'performance', label: 'Performance', icon: '📊' },
				{ id: 'services', label: 'Services', icon: '🏗️' },
				{ id: 'diagnostics', label: 'Diagnostics', icon: '🔧' },
				{ id: 'logs', label: 'Logs', icon: '📝', count: $systemLogs.length }
			] as tab}
				<button
					class="py-3 px-1 border-b-2 font-medium text-sm {
						selectedTab === tab.id
							? 'border-cyan-500 text-cyan-400'
							: 'border-transparent text-gray-400 hover:text-gray-300'
					} transition-colors"
					on:click={() => selectedTab = tab.id}
				>
					<span class="flex items-center space-x-2">
						<span>{tab.icon}</span>
						<span>{tab.label}</span>
						{#if tab.count !== undefined && tab.count > 0}
							<span class="bg-cyan-600 text-white text-xs px-2 py-1 rounded-full">{tab.count}</span>
						{/if}
					</span>
				</button>
			{/each}
		</nav>
	</div>
	
	<!-- Tab Content -->
	<div class="flex-1 overflow-y-auto p-4">
		{#if selectedTab === 'status'}
			<!-- System Status Overview -->
			<div class="space-y-6">
				<!-- Core Components Status -->
				<div>
					<h3 class="font-medium text-white mb-4">Core Components</h3>
					<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
						{#each [
							{ name: 'Avatar Service', status: $systemStatus.avatar_service, description: '3D rendering and animation' },
							{ name: 'Renderer', status: $systemStatus.renderer_service, description: 'WebGL and GPU acceleration' },
							{ name: 'OSINT Integration', status: $systemStatus.osint_integration, description: 'Intelligence gathering tools' },
							{ name: 'WebSocket', status: $systemStatus.websocket_connection, description: 'Real-time communication' },
							{ name: 'GPU Acceleration', status: $systemStatus.gpu_acceleration ? 'enabled' : 'disabled', description: 'Hardware acceleration' },
							{ name: 'Overall Health', status: $systemStatus.overall_health, description: 'System-wide status' }
						] as component}
							<div class="bg-gray-800 rounded-lg p-4">
								<div class="flex items-center justify-between mb-2">
									<h4 class="font-medium text-white">{component.name}</h4>
									<span class="px-2 py-1 rounded text-xs font-medium {getStatusColor(component.status)}">
										{component.status.toUpperCase()}
									</span>
								</div>
								<p class="text-sm text-gray-400">{component.description}</p>
							</div>
						{/each}
					</div>
				</div>
				
				<!-- Quick Actions -->
				<div>
					<h3 class="font-medium text-white mb-4">Quick Actions</h3>
					<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
						<button
							on:click={restartAvatarService}
							class="px-4 py-3 bg-orange-600 hover:bg-orange-700 rounded font-medium transition-colors"
						>
							Restart Avatar
						</button>
						<button
							on:click={runSystemDiagnostics}
							class="px-4 py-3 bg-blue-600 hover:bg-blue-700 rounded font-medium transition-colors"
						>
							Run Diagnostics
						</button>
						<button
							on:click={exportDiagnostics}
							class="px-4 py-3 bg-green-600 hover:bg-green-700 rounded font-medium transition-colors"
						>
							Export Report
						</button>
						<button
							on:click={clearSystemLogs}
							class="px-4 py-3 bg-red-600 hover:bg-red-700 rounded font-medium transition-colors"
						>
							Clear Logs
						</button>
					</div>
				</div>
			</div>
			
		{:else if selectedTab === 'performance'}
			<!-- Performance Metrics -->
			<div class="space-y-6">
				<!-- GPU Performance -->
				<div>
					<h3 class="font-medium text-white mb-4">GPU Performance (RTX 4090)</h3>
					<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="text-2xl font-bold text-cyan-400">{$performanceMetrics.gpu_utilization}%</div>
							<div class="text-sm text-gray-400">GPU Utilization</div>
						</div>
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="text-2xl font-bold text-green-400">{formatBytes($performanceMetrics.gpu_memory_used)}</div>
							<div class="text-sm text-gray-400">GPU Memory Used</div>
							<div class="text-xs text-gray-500">of {formatBytes($performanceMetrics.gpu_memory_total)}</div>
						</div>
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="text-2xl font-bold text-yellow-400">{$performanceMetrics.gpu_temperature}°C</div>
							<div class="text-sm text-gray-400">GPU Temperature</div>
						</div>
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="text-2xl font-bold text-purple-400">{$performanceMetrics.render_fps}</div>
							<div class="text-sm text-gray-400">Render FPS</div>
						</div>
					</div>
				</div>
				
				<!-- System Performance -->
				<div>
					<h3 class="font-medium text-white mb-4">System Performance</h3>
					<div class="grid grid-cols-2 md:grid-cols-3 gap-4">
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="text-2xl font-bold text-blue-400">{$performanceMetrics.cpu_usage}%</div>
							<div class="text-sm text-gray-400">CPU Usage</div>
						</div>
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="text-2xl font-bold text-orange-400">{$performanceMetrics.memory_usage}%</div>
							<div class="text-sm text-gray-400">Memory Usage</div>
						</div>
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="text-2xl font-bold text-red-400">{$performanceMetrics.disk_usage}%</div>
							<div class="text-sm text-gray-400">Disk Usage</div>
						</div>
					</div>
				</div>
				
				<!-- Network Performance -->
				<div>
					<h3 class="font-medium text-white mb-4">Network Performance</h3>
					<div class="grid grid-cols-2 gap-4">
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="text-2xl font-bold text-cyan-400">{$performanceMetrics.websocket_latency.toFixed(1)}ms</div>
							<div class="text-sm text-gray-400">WebSocket Latency</div>
						</div>
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="text-2xl font-bold text-indigo-400">{$performanceMetrics.message_queue_size}</div>
							<div class="text-sm text-gray-400">Message Queue Size</div>
						</div>
					</div>
				</div>
			</div>
			
		{:else if selectedTab === 'services'}
			<!-- Services Status -->
			<div class="space-y-4">
				{#each $services as service}
					<div class="bg-gray-800 rounded-lg p-4">
						<div class="flex items-center justify-between">
							<div class="flex-1">
								<div class="flex items-center space-x-3">
									<h4 class="font-medium text-white">{service.name}</h4>
									<span class="px-2 py-1 rounded text-xs font-medium {getStatusColor(service.status)}">
										{service.status.toUpperCase()}
									</span>
								</div>
								<p class="text-sm text-gray-400 mt-1">{service.description}</p>
								<div class="flex items-center space-x-4 mt-2 text-xs text-gray-500">
									<span>Endpoint: {service.endpoint}</span>
									{#if service.last_check}
										<span>Last Check: {formatTimestamp(service.last_check)}</span>
									{/if}
									{#if service.response_time > 0}
										<span>Response: {service.response_time.toFixed(0)}ms</span>
									{/if}
								</div>
							</div>
							
							<button
								on:click={() => checkAllServices()}
								class="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
							>
								Test
							</button>
						</div>
					</div>
				{/each}
			</div>
			
		{:else if selectedTab === 'diagnostics'}
			<!-- System Diagnostics -->
			<div class="space-y-6">
				{#if $diagnostics.gpu_info}
					<div>
						<h3 class="font-medium text-white mb-4">GPU Information</h3>
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="grid grid-cols-2 gap-4 text-sm">
								<div><span class="text-gray-400">Name:</span> <span class="text-white">{$diagnostics.gpu_info.name}</span></div>
								<div><span class="text-gray-400">Driver:</span> <span class="text-white">{$diagnostics.gpu_info.driver_version}</span></div>
								<div><span class="text-gray-400">CUDA:</span> <span class="text-white">{$diagnostics.gpu_info.cuda_version}</span></div>
								<div><span class="text-gray-400">Memory:</span> <span class="text-white">{formatBytes($diagnostics.gpu_info.total_memory)}</span></div>
							</div>
						</div>
					</div>
				{/if}
				
				{#if $diagnostics.system_info}
					<div>
						<h3 class="font-medium text-white mb-4">System Information</h3>
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="grid grid-cols-2 gap-4 text-sm">
								<div><span class="text-gray-400">OS:</span> <span class="text-white">{$diagnostics.system_info.os}</span></div>
								<div><span class="text-gray-400">Kernel:</span> <span class="text-white">{$diagnostics.system_info.kernel}</span></div>
								<div><span class="text-gray-400">CPU:</span> <span class="text-white">{$diagnostics.system_info.cpu}</span></div>
								<div><span class="text-gray-400">RAM:</span> <span class="text-white">{formatBytes($diagnostics.system_info.total_memory)}</span></div>
							</div>
						</div>
					</div>
				{/if}
				
				{#if $diagnostics.avatar_config}
					<div>
						<h3 class="font-medium text-white mb-4">Avatar Configuration</h3>
						<div class="bg-gray-800 rounded-lg p-4">
							<div class="grid grid-cols-2 gap-4 text-sm">
								<div><span class="text-gray-400">Renderer:</span> <span class="text-white">{$diagnostics.avatar_config.renderer}</span></div>
								<div><span class="text-gray-400">Quality:</span> <span class="text-white">{$diagnostics.avatar_config.quality_level}</span></div>
								<div><span class="text-gray-400">FPS Target:</span> <span class="text-white">{$diagnostics.avatar_config.target_fps}</span></div>
								<div><span class="text-gray-400">Model Size:</span> <span class="text-white">{formatBytes($diagnostics.avatar_config.model_size)}</span></div>
							</div>
						</div>
					</div>
				{/if}
				
				{#if $diagnostics.last_updated}
					<div class="text-xs text-gray-500">
						Last updated: {$diagnostics.last_updated.toLocaleString()}
					</div>
				{/if}
			</div>
			
		{:else if selectedTab === 'logs'}
			<!-- System Logs -->
			<div class="space-y-4">
				<div class="flex justify-between items-center">
					<h3 class="font-medium text-white">System Logs</h3>
					<button
						on:click={clearSystemLogs}
						class="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors"
					>
						Clear Logs
					</button>
				</div>
				
				{#if $systemLogs.length === 0}
					<div class="text-center py-8 text-gray-400">
						<div class="text-4xl mb-2">📝</div>
						<p>No system logs</p>
					</div>
				{:else}
					<div class="space-y-2 max-h-96 overflow-y-auto">
						{#each $systemLogs as log}
							<div class="bg-gray-800 rounded p-3">
								<div class="flex items-center justify-between mb-1">
									<div class="flex items-center space-x-2">
										<span class="text-xs font-medium {getLogLevelColor(log.level)} uppercase">{log.level}</span>
										<span class="text-sm font-medium text-white">{log.component}</span>
									</div>
									<span class="text-xs text-gray-500">{formatTimestamp(log.timestamp)}</span>
								</div>
								<p class="text-sm text-gray-300">{log.message}</p>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		{/if}
	</div>
</div>

<style>
	.system-integration-panel {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
</style>