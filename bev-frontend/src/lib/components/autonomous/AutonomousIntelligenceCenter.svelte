<!--
Autonomous Intelligence Center - Self-Managing AI Systems Command Interface
Connected to: src/autonomous/ (9 autonomous systems)
Features: System orchestration, adaptive learning, resource optimization, knowledge evolution
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management for Autonomous Systems
	const autonomousState = writable({
		operation_mode: 'AUTONOMOUS', // MANUAL, SUPERVISED, AUTONOMOUS, ENHANCED
		intelligence_coordinator: {
			status: 'active',
			coordinating: 0,
			total_agents: 0,
			coordination_efficiency: 95.2
		},
		adaptive_learning: {
			status: 'learning',
			learning_rate: 0.001,
			knowledge_updated: 0,
			improvement_score: 8.7
		},
		resource_optimizer: {
			status: 'optimizing',
			cpu_optimization: 23.5,
			memory_freed: '2.1GB',
			efficiency_gain: 18.3
		},
		knowledge_evolution: {
			status: 'evolving',
			knowledge_nodes: 15420,
			new_connections: 847,
			evolution_rate: 12.3
		},
		autonomous_controller: {
			status: 'managing',
			decisions_made: 1247,
			success_rate: 98.7,
			intervention_needed: false
		},
		enhanced_controller: {
			status: 'enhanced',
			advanced_decisions: 523,
			predictive_accuracy: 94.1,
			proactive_actions: 89
		}
	});
	
	const selectedTab = writable('overview'); // 'overview', 'controller', 'enhanced', 'coordinator', 'learning', 'optimizer', 'evolution'
	const selectedSystem = writable(null);
	const isLoading = writable(false);
	
	// WebSocket for real-time autonomous system updates
	let ws: WebSocket | null = null;
	
	// Autonomous operation metrics
	let autonomousMetrics = {
		total_decisions_24h: 0,
		successful_optimizations: 0,
		knowledge_growth_rate: 0,
		system_efficiency: 0,
		intervention_rate: 0,
		autonomous_uptime: 0
	};
	
	// System operation modes
	const operationModes = [
		{ id: 'MANUAL', label: 'Manual Control', description: 'Human-controlled operations', color: 'text-blue-400' },
		{ id: 'SUPERVISED', label: 'Supervised AI', description: 'AI with human oversight', color: 'text-yellow-400' },
		{ id: 'AUTONOMOUS', label: 'Fully Autonomous', description: 'Complete AI autonomy', color: 'text-green-400' },
		{ id: 'ENHANCED', label: 'Enhanced Autonomous', description: 'Advanced AI with predictive capabilities', color: 'text-purple-400' }
	];
	
	// Autonomous subsystems
	const autonomousSystems = [
		{
			id: 'autonomous_controller',
			name: 'Autonomous Controller',
			icon: '🤖',
			description: 'Core autonomous decision-making system',
			port: 8300
		},
		{
			id: 'enhanced_controller',
			name: 'Enhanced Controller',
			icon: '🧠',
			description: 'Advanced autonomous operations with prediction',
			port: 8301
		},
		{
			id: 'intelligence_coordinator',
			name: 'Intelligence Coordinator',
			icon: '🎯',
			description: 'Cross-system intelligence coordination',
			port: 8302
		},
		{
			id: 'adaptive_learning',
			name: 'Adaptive Learning',
			icon: '📚',
			description: 'System self-optimization and learning',
			port: 8303
		},
		{
			id: 'resource_optimizer',
			name: 'Resource Optimizer',
			icon: '⚡',
			description: 'Dynamic resource allocation optimization',
			port: 8304
		},
		{
			id: 'knowledge_evolution',
			name: 'Knowledge Evolution',
			icon: '🌱',
			description: 'Knowledge graph evolution and expansion',
			port: 8305
		}
	];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadAutonomousMetrics();
		startSystemMonitoring();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/autonomous-intelligence');
			
			ws.onopen = () => {
				console.log('Autonomous Intelligence WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleAutonomousUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Autonomous WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Autonomous WebSocket connection failed:', error);
		}
	}
	
	function handleAutonomousUpdate(data: any) {
		switch (data.type) {
			case 'decision_made':
				recordDecision(data.system, data.decision);
				break;
			case 'optimization_complete':
				recordOptimization(data.system, data.optimization);
				break;
			case 'learning_update':
				updateLearningProgress(data.progress);
				break;
			case 'knowledge_evolved':
				updateKnowledgeEvolution(data.evolution);
				break;
			case 'system_status_change':
				updateSystemStatus(data.system, data.status);
				break;
		}
	}
	
	async function loadAutonomousMetrics() {
		isLoading.set(true);
		try {
			const metrics = await invoke('get_autonomous_metrics');
			autonomousMetrics = { ...autonomousMetrics, ...metrics };
		} catch (error) {
			console.error('Failed to load autonomous metrics:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function startSystemMonitoring() {
		// Monitor all autonomous systems
		setInterval(async () => {
			try {
				for (const system of autonomousSystems) {
					const status = await fetch(`http://localhost:${system.port}/status`);
					const statusData = await status.json();
					updateSystemData(system.id, statusData);
				}
			} catch (error) {
				console.error('System monitoring failed:', error);
			}
		}, 5000);
	}
	
	async function setOperationMode(mode: string) {
		try {
			await invoke('set_autonomous_mode', { mode });
			autonomousState.update(current => ({
				...current,
				operation_mode: mode
			}));
			dispatch('mode_changed', { mode });
		} catch (error) {
			console.error('Failed to set operation mode:', error);
		}
	}
	
	async function triggerSystemOptimization() {
		try {
			await invoke('trigger_system_optimization');
			dispatch('optimization_triggered');
		} catch (error) {
			console.error('Failed to trigger optimization:', error);
		}
	}
	
	async function pauseAutonomousOperations() {
		try {
			await invoke('pause_autonomous_operations');
			autonomousState.update(current => ({
				...current,
				operation_mode: 'MANUAL'
			}));
		} catch (error) {
			console.error('Failed to pause autonomous operations:', error);
		}
	}
	
	function recordDecision(system: string, decision: any) {
		autonomousMetrics.total_decisions_24h += 1;
		dispatch('decision_recorded', { system, decision });
	}
	
	function recordOptimization(system: string, optimization: any) {
		autonomousMetrics.successful_optimizations += 1;
		dispatch('optimization_recorded', { system, optimization });
	}
	
	function updateLearningProgress(progress: any) {
		autonomousState.update(current => ({
			...current,
			adaptive_learning: { ...current.adaptive_learning, ...progress }
		}));
	}
	
	function updateKnowledgeEvolution(evolution: any) {
		autonomousState.update(current => ({
			...current,
			knowledge_evolution: { ...current.knowledge_evolution, ...evolution }
		}));
	}
	
	function updateSystemStatus(systemId: string, status: any) {
		autonomousState.update(current => ({
			...current,
			[systemId]: { ...current[systemId], ...status }
		}));
	}
	
	function updateSystemData(systemId: string, data: any) {
		autonomousState.update(current => ({
			...current,
			[systemId]: { ...current[systemId], ...data }
		}));
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'active':
			case 'managing':
			case 'optimizing':
			case 'learning':
			case 'evolving':
			case 'enhanced': return 'text-green-400';
			case 'warning':
			case 'degraded': return 'text-yellow-400';
			case 'error':
			case 'failed': return 'text-red-400';
			case 'paused':
			case 'idle': return 'text-gray-400';
			default: return 'text-gray-400';
		}
	}
	
	function getModeColor(mode: string): string {
		const modeConfig = operationModes.find(m => m.id === mode);
		return modeConfig ? modeConfig.color : 'text-gray-400';
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function openSystemModal(system: any) {
		selectedSystem.set(system);
	}
</script>

<!-- Autonomous Intelligence Center -->
<div class="autonomous-center h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-purple-400">🧠 Autonomous Intelligence Center</h1>
			<div class="flex items-center space-x-4">
				<div class="px-4 py-2 rounded-lg bg-purple-900/20 {getModeColor($autonomousState.operation_mode)} font-bold">
					MODE: {$autonomousState.operation_mode}
				</div>
				<div class="text-sm text-gray-400">
					{autonomousSystems.filter(s => $autonomousState[s.id]?.status === 'active').length}/6 systems active
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadAutonomousMetrics}
						class="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Operation Mode Controls -->
	<div class="p-4 border-b border-gray-800">
		<div class="flex items-center justify-between">
			<div class="flex space-x-2">
				{#each operationModes as mode}
					<button
						on:click={() => setOperationMode(mode.id)}
						class="px-4 py-2 rounded font-medium text-sm transition-colors {
							$autonomousState.operation_mode === mode.id
								? 'bg-purple-600 text-white'
								: 'bg-gray-700 hover:bg-gray-600 text-gray-300'
						}"
						title={mode.description}
					>
						{mode.label}
					</button>
				{/each}
			</div>
			
			<div class="flex space-x-2">
				<button
					on:click={triggerSystemOptimization}
					class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium transition-colors"
				>
					🚀 Trigger Optimization
				</button>
				<button
					on:click={pauseAutonomousOperations}
					class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
				>
					⏸️ Emergency Pause
				</button>
			</div>
		</div>
	</div>
	
	<!-- Autonomous Systems Status Grid -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
			{#each autonomousSystems as system}
				{@const systemData = $autonomousState[system.id] || {}}
				<div 
					class="bg-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-700 transition-colors"
					on:click={() => openSystemModal(system)}
				>
					<div class="flex items-center justify-between mb-3">
						<div class="flex items-center space-x-2">
							<span class="text-xl">{system.icon}</span>
							<span class="font-medium text-white text-sm">{system.name}</span>
						</div>
						<div class="flex items-center space-x-1">
							<div class="w-2 h-2 rounded-full {getStatusColor(systemData.status)}"></div>
							<span class="text-xs {getStatusColor(systemData.status)}">{systemData.status || 'unknown'}</span>
						</div>
					</div>
					
					<p class="text-gray-400 text-xs mb-3">{system.description}</p>
					
					{#if system.id === 'intelligence_coordinator'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Coordinating:</span>
								<span class="text-white">{systemData.coordinating || 0}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Efficiency:</span>
								<span class="text-green-400">{(systemData.coordination_efficiency || 0).toFixed(1)}%</span>
							</div>
						</div>
					{:else if system.id === 'adaptive_learning'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Learning Rate:</span>
								<span class="text-white">{systemData.learning_rate || 0}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Score:</span>
								<span class="text-blue-400">{(systemData.improvement_score || 0).toFixed(1)}/10</span>
							</div>
						</div>
					{:else if system.id === 'resource_optimizer'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">CPU Saved:</span>
								<span class="text-white">{(systemData.cpu_optimization || 0).toFixed(1)}%</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Memory Freed:</span>
								<span class="text-green-400">{systemData.memory_freed || '0MB'}</span>
							</div>
						</div>
					{:else if system.id === 'knowledge_evolution'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Nodes:</span>
								<span class="text-white">{formatNumber(systemData.knowledge_nodes || 0)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">New Links:</span>
								<span class="text-purple-400">{systemData.new_connections || 0}</span>
							</div>
						</div>
					{:else if system.id === 'autonomous_controller'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Decisions:</span>
								<span class="text-white">{formatNumber(systemData.decisions_made || 0)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Success Rate:</span>
								<span class="text-green-400">{(systemData.success_rate || 0).toFixed(1)}%</span>
							</div>
						</div>
					{:else if system.id === 'enhanced_controller'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Predictions:</span>
								<span class="text-white">{systemData.advanced_decisions || 0}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Accuracy:</span>
								<span class="text-purple-400">{(systemData.predictive_accuracy || 0).toFixed(1)}%</span>
							</div>
						</div>
					{/if}
				</div>
			{/each}
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Intelligence Overview', icon: '📊' },
				{ id: 'controller', label: 'Autonomous Controller', icon: '🤖' },
				{ id: 'enhanced', label: 'Enhanced Controller', icon: '🧠' },
				{ id: 'coordinator', label: 'Intelligence Coordinator', icon: '🎯' },
				{ id: 'learning', label: 'Adaptive Learning', icon: '📚' },
				{ id: 'optimizer', label: 'Resource Optimizer', icon: '⚡' },
				{ id: 'evolution', label: 'Knowledge Evolution', icon: '🌱' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedTab === tab.id
							? 'border-purple-500 text-purple-400'
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
			<!-- Autonomous Intelligence Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- System Performance Matrix -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-purple-400">System Performance Matrix</h3>
					<div class="space-y-4">
						<div class="grid grid-cols-2 gap-4 text-center">
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-green-400">{autonomousMetrics.total_decisions_24h}</div>
								<div class="text-xs text-gray-400">Decisions (24h)</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-blue-400">{autonomousMetrics.successful_optimizations}</div>
								<div class="text-xs text-gray-400">Optimizations</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-yellow-400">{(autonomousMetrics.knowledge_growth_rate || 0).toFixed(1)}%</div>
								<div class="text-xs text-gray-400">Knowledge Growth</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-purple-400">{(autonomousMetrics.system_efficiency || 0).toFixed(1)}%</div>
								<div class="text-xs text-gray-400">System Efficiency</div>
							</div>
						</div>
						
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">Autonomous Performance</h4>
							<div class="space-y-2 text-sm">
								<div class="flex justify-between">
									<span class="text-gray-400">Intervention Rate:</span>
									<span class="text-white">{(autonomousMetrics.intervention_rate || 0).toFixed(2)}%</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Autonomous Uptime:</span>
									<span class="text-white">{(autonomousMetrics.autonomous_uptime || 0).toFixed(1)} hours</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Decision Accuracy:</span>
									<span class="text-green-400">{($autonomousState.autonomous_controller.success_rate || 0).toFixed(1)}%</span>
								</div>
							</div>
						</div>
					</div>
				</div>
				
				<!-- Real-time Autonomous Operations -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Real-time Operations</h3>
					<div class="space-y-4">
						<!-- Intelligence Coordination -->
						<div class="bg-gray-900 rounded p-4">
							<div class="flex items-center justify-between mb-2">
								<span class="font-medium text-white">🎯 Intelligence Coordination</span>
								<span class="text-xs {getStatusColor($autonomousState.intelligence_coordinator.status)}">
									{$autonomousState.intelligence_coordinator.status}
								</span>
							</div>
							<div class="text-sm text-gray-400">
								Coordinating {$autonomousState.intelligence_coordinator.coordinating} operations
							</div>
							<div class="w-full bg-gray-700 rounded-full h-1 mt-2">
								<div
									class="bg-purple-600 h-1 rounded-full"
									style="width: {$autonomousState.intelligence_coordinator.coordination_efficiency}%"
								></div>
							</div>
						</div>
						
						<!-- Adaptive Learning -->
						<div class="bg-gray-900 rounded p-4">
							<div class="flex items-center justify-between mb-2">
								<span class="font-medium text-white">📚 Adaptive Learning</span>
								<span class="text-xs {getStatusColor($autonomousState.adaptive_learning.status)}">
									{$autonomousState.adaptive_learning.status}
								</span>
							</div>
							<div class="text-sm text-gray-400">
								Learning score: {$autonomousState.adaptive_learning.improvement_score}/10
							</div>
							<div class="w-full bg-gray-700 rounded-full h-1 mt-2">
								<div
									class="bg-blue-600 h-1 rounded-full"
									style="width: {($autonomousState.adaptive_learning.improvement_score / 10) * 100}%"
								></div>
							</div>
						</div>
						
						<!-- Resource Optimization -->
						<div class="bg-gray-900 rounded p-4">
							<div class="flex items-center justify-between mb-2">
								<span class="font-medium text-white">⚡ Resource Optimization</span>
								<span class="text-xs {getStatusColor($autonomousState.resource_optimizer.status)}">
									{$autonomousState.resource_optimizer.status}
								</span>
							</div>
							<div class="text-sm text-gray-400">
								Efficiency gain: +{$autonomousState.resource_optimizer.efficiency_gain}%
							</div>
							<div class="text-sm text-green-400">
								Memory freed: {$autonomousState.resource_optimizer.memory_freed}
							</div>
						</div>
						
						<!-- Knowledge Evolution -->
						<div class="bg-gray-900 rounded p-4">
							<div class="flex items-center justify-between mb-2">
								<span class="font-medium text-white">🌱 Knowledge Evolution</span>
								<span class="text-xs {getStatusColor($autonomousState.knowledge_evolution.status)}">
									{$autonomousState.knowledge_evolution.status}
								</span>
							</div>
							<div class="text-sm text-gray-400">
								{formatNumber($autonomousState.knowledge_evolution.knowledge_nodes)} nodes
							</div>
							<div class="text-sm text-purple-400">
								+{$autonomousState.knowledge_evolution.new_connections} new connections
							</div>
						</div>
					</div>
				</div>
			</div>
			
		{:else}
			<!-- Detailed subsystem interfaces -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-white">{autonomousSystems.find(s => s.id === $selectedTab)?.name} Interface</h3>
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">{autonomousSystems.find(s => s.id === $selectedTab)?.icon}</div>
					<p>Detailed {autonomousSystems.find(s => s.id === $selectedTab)?.name} interface</p>
					<p class="text-sm mt-2">Real-time monitoring and control panel</p>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Autonomous Intelligence | Mode: {$autonomousState.operation_mode} | 6 AI systems active
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('export_autonomous_report')}
					class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm font-medium transition-colors"
				>
					Export AI Report
				</button>
				<button
					on:click={() => dispatch('optimize_all_systems')}
					class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium transition-colors"
				>
					Optimize All Systems
				</button>
			</div>
		</div>
	</div>
</div>

<!-- System Detail Modal -->
{#if $selectedSystem}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedSystem.set(null)}>
		<div class="max-w-3xl w-full mx-4 bg-gray-800 rounded-lg p-6" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-purple-400">
					{$selectedSystem.icon} {$selectedSystem.name}
				</h3>
				<button
					on:click={() => selectedSystem.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">System Information</h4>
					<p class="text-gray-300 text-sm">{$selectedSystem.description}</p>
					<div class="mt-3 text-sm">
						<span class="text-gray-400">Port:</span>
						<span class="text-white ml-2">{$selectedSystem.port}</span>
					</div>
				</div>
				
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Real-time Status</h4>
					<div class="space-y-2 text-sm">
						{@const systemData = $autonomousState[$selectedSystem.id] || {}}
						{#each Object.entries(systemData) as [key, value]}
							<div class="flex justify-between">
								<span class="text-gray-400 capitalize">{key.replace('_', ' ')}:</span>
								<span class="text-white">{value}</span>
							</div>
						{/each}
					</div>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.autonomous-center {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Autonomous-themed styling */
	:global(.autonomous-center .ai-active) {
		animation: pulse-purple 3s infinite;
	}
	
	@keyframes pulse-purple {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.7; }
	}
	
	/* Custom scrollbar */
	:global(.autonomous-center *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.autonomous-center *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.autonomous-center *::-webkit-scrollbar-thumb) {
		background: #8b5cf6;
		border-radius: 3px;
	}
	
	:global(.autonomous-center *::-webkit-scrollbar-thumb:hover) {
		background: #a78bfa;
	}
</style>