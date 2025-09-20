<!--
Tor Network Management - 3-Node Relay System Administration
Connected to: tor/ (3-node Tor relay system with monitoring)
Features: Relay management, circuit visualization, bandwidth monitoring, security controls
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const torState = writable({
		relay_network: {
			entry_node: { status: 'running', bandwidth: 0, connections: 0, uptime: 0 },
			middle_node: { status: 'running', bandwidth: 0, connections: 0, uptime: 0 },
			exit_node: { status: 'running', bandwidth: 0, connections: 0, uptime: 0 }
		},
		circuits: [],
		active_connections: 0,
		total_bandwidth: 0,
		exit_countries: [],
		onion_services: [],
		network_health: 'excellent',
		consensus_status: 'synchronized',
		directory_authorities: 9
	});
	
	const selectedTab = writable('overview'); // 'overview', 'relays', 'circuits', 'connections', 'onion', 'security'
	const selectedRelay = writable(null);
	const isLoading = writable(false);
	
	// WebSocket for real-time Tor updates
	let ws: WebSocket | null = null;
	
	// Tor configuration
	let torConfig = {
		exit_policy: 'accept *:80,accept *:443,reject *:*',
		bandwidth_rate: '1MB',
		bandwidth_burst: '2MB',
		contact_info: 'BEV Research <research@bev.local>',
		nickname: 'BEVRelay',
		or_port: 9001,
		dir_port: 9030
	};
	
	// Circuit data
	let circuitData = [];
	let connectionMetrics = {
		total_connections_24h: 0,
		successful_circuits: 0,
		failed_circuits: 0,
		avg_circuit_build_time: 0,
		bandwidth_utilization: 0
	};
	
	const torNodes = [
		{
			id: 'entry',
			name: 'Entry Node',
			type: 'guard',
			icon: '🚪',
			port: 9001,
			description: 'First hop in Tor circuit, receives client connections'
		},
		{
			id: 'middle',
			name: 'Middle Relay',
			type: 'relay',
			icon: '🔄',
			port: 9002,
			description: 'Middle hop that forwards encrypted traffic'
		},
		{
			id: 'exit',
			name: 'Exit Node',
			type: 'exit',
			icon: '🚪',
			port: 9003,
			description: 'Final hop that connects to destination'
		}
	];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadTorState();
		startTorMonitoring();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/tor-management');
			
			ws.onopen = () => {
				console.log('Tor Management WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleTorUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Tor WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Tor WebSocket connection failed:', error);
		}
	}
	
	function handleTorUpdate(data: any) {
		switch (data.type) {
			case 'relay_status_update':
				updateRelayStatus(data.relay, data.status);
				break;
			case 'circuit_built':
				addCircuit(data.circuit);
				break;
			case 'circuit_failed':
				updateCircuitStatus(data.circuit_id, 'failed');
				break;
			case 'bandwidth_update':
				updateBandwidth(data.relay, data.bandwidth);
				break;
			case 'connection_update':
				updateConnections(data.connections);
				break;
		}
	}
	
	async function loadTorState() {
		isLoading.set(true);
		try {
			const [relayStatus, circuits, connections, metrics] = await Promise.all([
				invoke('get_tor_relay_status'),
				invoke('get_tor_circuits'),
				invoke('get_tor_connections'),
				invoke('get_tor_metrics')
			]);
			
			torState.update(current => ({
				...current,
				relay_network: { ...current.relay_network, ...relayStatus },
				circuits: circuits || [],
				active_connections: connections?.length || 0,
				total_bandwidth: metrics?.total_bandwidth || 0
			}));
			
			connectionMetrics = { ...connectionMetrics, ...metrics };
		} catch (error) {
			console.error('Failed to load Tor state:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function startTorMonitoring() {
		setInterval(async () => {
			try {
				const status = await invoke('get_tor_status');
				torState.update(current => ({
					...current,
					...status
				}));
			} catch (error) {
				console.error('Tor monitoring failed:', error);
			}
		}, 10000);
	}
	
	async function newTorCircuit() {
		try {
			const result = await invoke('create_new_tor_circuit');
			addCircuit(result);
		} catch (error) {
			console.error('Failed to create new circuit:', error);
		}
	}
	
	async function setExitCountry(country: string) {
		try {
			await invoke('set_tor_exit_country', { country });
			dispatch('exit_country_changed', { country });
		} catch (error) {
			console.error('Failed to set exit country:', error);
		}
	}
	
	async function restartRelay(relayId: string) {
		try {
			await invoke('restart_tor_relay', { relayId });
			updateRelayStatus(relayId, 'restarting');
		} catch (error) {
			console.error('Failed to restart relay:', error);
		}
	}
	
	function updateRelayStatus(relayId: string, status: any) {
		torState.update(current => ({
			...current,
			relay_network: {
				...current.relay_network,
				[relayId + '_node']: { ...current.relay_network[relayId + '_node'], ...status }
			}
		}));
	}
	
	function addCircuit(circuit: any) {
		torState.update(current => ({
			...current,
			circuits: [circuit, ...current.circuits.slice(0, 49)]
		}));
	}
	
	function updateCircuitStatus(circuitId: string, status: string) {
		torState.update(current => ({
			...current,
			circuits: current.circuits.map(circuit =>
				circuit.id === circuitId ? { ...circuit, status } : circuit
			)
		}));
	}
	
	function updateBandwidth(relayId: string, bandwidth: number) {
		torState.update(current => ({
			...current,
			relay_network: {
				...current.relay_network,
				[relayId + '_node']: {
					...current.relay_network[relayId + '_node'],
					bandwidth
				}
			}
		}));
	}
	
	function updateConnections(connections: any) {
		torState.update(current => ({
			...current,
			active_connections: connections.length
		}));
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'running':
			case 'active': return 'text-green-400';
			case 'starting':
			case 'restarting': return 'text-yellow-400';
			case 'stopped':
			case 'failed': return 'text-red-400';
			case 'dormant': return 'text-gray-400';
			default: return 'text-gray-400';
		}
	}
	
	function getHealthColor(health: string): string {
		switch (health) {
			case 'excellent': return 'text-green-400';
			case 'good': return 'text-blue-400';
			case 'fair': return 'text-yellow-400';
			case 'poor': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatBytes(bytes: number): string {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
	}
	
	function formatUptime(seconds: number): string {
		const hours = Math.floor(seconds / 3600);
		const minutes = Math.floor((seconds % 3600) / 60);
		if (hours > 24) {
			const days = Math.floor(hours / 24);
			return `${days}d ${hours % 24}h`;
		}
		return `${hours}h ${minutes}m`;
	}
	
	function openRelayModal(relay: any) {
		selectedRelay.set(relay);
	}
</script>

<!-- Tor Network Manager -->
<div class="tor-manager h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-purple-400">🧅 Tor Network Management</h1>
			<div class="flex items-center space-x-4">
				<div class="flex items-center space-x-2">
					<div class="w-2 h-2 rounded-full {getHealthColor($torState.network_health)}"></div>
					<span class="text-sm text-gray-400">Network: {$torState.network_health}</span>
				</div>
				<div class="text-sm text-gray-400">
					{$torState.active_connections} connections | {$torState.circuits.length} circuits
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadTorState}
						class="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Relay Network Status -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-1 md:grid-cols-3 gap-4">
			{#each torNodes as node}
				{@const nodeData = $torState.relay_network[node.id + '_node'] || {}}
				<div 
					class="bg-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-700 transition-colors"
					on:click={() => openRelayModal({...node, ...nodeData})}
				>
					<div class="flex items-center justify-between mb-3">
						<div class="flex items-center space-x-2">
							<span class="text-2xl">{node.icon}</span>
							<span class="font-medium text-white">{node.name}</span>
						</div>
						<div class="flex items-center space-x-1">
							<div class="w-2 h-2 rounded-full {getStatusColor(nodeData.status)}"></div>
							<span class="text-xs {getStatusColor(nodeData.status)}">{nodeData.status || 'unknown'}</span>
						</div>
					</div>
					
					<p class="text-gray-400 text-xs mb-3">{node.description}</p>
					
					<div class="space-y-1 text-xs">
						<div class="flex justify-between">
							<span class="text-gray-400">Bandwidth:</span>
							<span class="text-white">{formatBytes(nodeData.bandwidth || 0)}/s</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Connections:</span>
							<span class="text-white">{nodeData.connections || 0}</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Uptime:</span>
							<span class="text-white">{formatUptime(nodeData.uptime || 0)}</span>
						</div>
					</div>
				</div>
			{/each}
		</div>
	</div>
	
	<!-- Network Overview -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-5 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-purple-400">{$torState.circuits.length}</div>
				<div class="text-sm text-gray-400">Active Circuits</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">{$torState.active_connections}</div>
				<div class="text-sm text-gray-400">Connections</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-blue-400">{formatBytes($torState.total_bandwidth)}/s</div>
				<div class="text-sm text-gray-400">Total Bandwidth</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-yellow-400">{$torState.exit_countries.length}</div>
				<div class="text-sm text-gray-400">Exit Countries</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-orange-400">{$torState.onion_services.length}</div>
				<div class="text-sm text-gray-400">Onion Services</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Network Overview', icon: '📊' },
				{ id: 'relays', label: 'Relay Management', icon: '🔄' },
				{ id: 'circuits', label: 'Circuit Visualization', icon: '🕸️' },
				{ id: 'connections', label: 'Connections', icon: '🔗' },
				{ id: 'onion', label: 'Onion Services', icon: '🧅' },
				{ id: 'security', label: 'Security Controls', icon: '🔒' }
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
			<!-- Network Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Circuit Visualization -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-purple-400">Circuit Path Visualization</h3>
					<div class="relative">
						<!-- Visual representation of 3-hop circuit -->
						<div class="flex items-center justify-between">
							<div class="text-center">
								<div class="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center mb-2">
									<span class="text-2xl">🖥️</span>
								</div>
								<div class="text-sm text-gray-300">Client</div>
							</div>
							
							<div class="flex-1 px-4">
								<svg class="w-full h-2" viewBox="0 0 100 2">
									<line x1="0" y1="1" x2="100" y2="1" stroke="#8b5cf6" stroke-width="2" stroke-dasharray="5,5">
										<animate attributeName="stroke-dashoffset" values="0;10" dur="1s" repeatCount="indefinite"/>
									</line>
								</svg>
							</div>
							
							<div class="text-center">
								<div class="w-16 h-16 bg-green-600 rounded-full flex items-center justify-center mb-2">
									<span class="text-2xl">🚪</span>
								</div>
								<div class="text-sm text-gray-300">Entry</div>
								<div class="text-xs text-gray-400">{$torState.relay_network.entry_node.connections || 0} conn</div>
							</div>
							
							<div class="flex-1 px-4">
								<svg class="w-full h-2" viewBox="0 0 100 2">
									<line x1="0" y1="1" x2="100" y2="1" stroke="#8b5cf6" stroke-width="2" stroke-dasharray="5,5">
										<animate attributeName="stroke-dashoffset" values="0;10" dur="1s" repeatCount="indefinite"/>
									</line>
								</svg>
							</div>
							
							<div class="text-center">
								<div class="w-16 h-16 bg-yellow-600 rounded-full flex items-center justify-center mb-2">
									<span class="text-2xl">🔄</span>
								</div>
								<div class="text-sm text-gray-300">Middle</div>
								<div class="text-xs text-gray-400">{$torState.relay_network.middle_node.connections || 0} conn</div>
							</div>
							
							<div class="flex-1 px-4">
								<svg class="w-full h-2" viewBox="0 0 100 2">
									<line x1="0" y1="1" x2="100" y2="1" stroke="#8b5cf6" stroke-width="2" stroke-dasharray="5,5">
										<animate attributeName="stroke-dashoffset" values="0;10" dur="1s" repeatCount="indefinite"/>
									</line>
								</svg>
							</div>
							
							<div class="text-center">
								<div class="w-16 h-16 bg-red-600 rounded-full flex items-center justify-center mb-2">
									<span class="text-2xl">🌐</span>
								</div>
								<div class="text-sm text-gray-300">Exit</div>
								<div class="text-xs text-gray-400">{$torState.relay_network.exit_node.connections || 0} conn</div>
							</div>
							
							<div class="flex-1 px-4">
								<svg class="w-full h-2" viewBox="0 0 100 2">
									<line x1="0" y1="1" x2="100" y2="1" stroke="#8b5cf6" stroke-width="2" stroke-dasharray="5,5">
										<animate attributeName="stroke-dashoffset" values="0;10" dur="1s" repeatCount="indefinite"/>
									</line>
								</svg>
							</div>
							
							<div class="text-center">
								<div class="w-16 h-16 bg-gray-600 rounded-full flex items-center justify-center mb-2">
									<span class="text-2xl">🎯</span>
								</div>
								<div class="text-sm text-gray-300">Target</div>
							</div>
						</div>
					</div>
				</div>
				
				<!-- Network Metrics -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Network Performance</h3>
					<div class="space-y-4">
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">Circuit Metrics</h4>
							<div class="space-y-2 text-sm">
								<div class="flex justify-between">
									<span class="text-gray-400">Successful Circuits:</span>
									<span class="text-white">{connectionMetrics.successful_circuits}</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Failed Circuits:</span>
									<span class="text-red-400">{connectionMetrics.failed_circuits}</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Avg Build Time:</span>
									<span class="text-white">{(connectionMetrics.avg_circuit_build_time || 0).toFixed(1)}s</span>
								</div>
							</div>
						</div>
						
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">Bandwidth Utilization</h4>
							<div class="space-y-2">
								<div class="flex justify-between text-sm">
									<span class="text-gray-400">Utilization:</span>
									<span class="text-white">{(connectionMetrics.bandwidth_utilization || 0).toFixed(1)}%</span>
								</div>
								<div class="w-full bg-gray-700 rounded-full h-2">
									<div
										class="bg-purple-600 h-2 rounded-full"
										style="width: {connectionMetrics.bandwidth_utilization || 0}%"
									></div>
								</div>
							</div>
						</div>
						
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">Consensus Status</h4>
							<div class="space-y-2 text-sm">
								<div class="flex justify-between">
									<span class="text-gray-400">Status:</span>
									<span class="text-green-400">{$torState.consensus_status}</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Directory Authorities:</span>
									<span class="text-white">{$torState.directory_authorities}/9</span>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'circuits'}
			<!-- Circuit Management -->
			<div class="space-y-4">
				<div class="flex items-center justify-between">
					<h3 class="text-lg font-semibold text-purple-400">Active Circuits</h3>
					<button
						on:click={newTorCircuit}
						class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded transition-colors"
					>
						New Circuit
					</button>
				</div>
				
				{#if $torState.circuits.length === 0}
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">🕸️</div>
						<p>No active circuits</p>
					</div>
				{:else}
					{#each $torState.circuits as circuit}
						<div class="bg-gray-800 rounded-lg p-6">
							<div class="flex items-center justify-between mb-4">
								<div class="flex items-center space-x-4">
									<h4 class="font-medium text-white">Circuit #{circuit.id}</h4>
									<span class="px-2 py-1 rounded text-xs {getStatusColor(circuit.status)} bg-gray-700">
										{circuit.status}
									</span>
									<span class="text-sm text-gray-400">
										{circuit.path?.length || 3}-hop circuit
									</span>
								</div>
								<div class="text-sm text-gray-400">
									Built: {circuit.built_at ? new Date(circuit.built_at).toLocaleTimeString() : 'Unknown'}
								</div>
							</div>
							
							{#if circuit.path}
								<div class="bg-gray-900 rounded p-4">
									<h5 class="font-medium text-gray-300 mb-2">Circuit Path</h5>
									<div class="flex items-center space-x-2">
										{#each circuit.path as node, index}
											<div class="bg-gray-700 rounded px-2 py-1 text-xs">
												{node.nickname || 'Unknown'}
												<div class="text-gray-400">{node.country || 'XX'}</div>
											</div>
											{#if index < circuit.path.length - 1}
												<span class="text-purple-400">→</span>
											{/if}
										{/each}
									</div>
								</div>
							{/if}
							
							<div class="grid grid-cols-3 gap-4 text-sm mt-4">
								<div>
									<span class="text-gray-400">Build Time:</span>
									<span class="text-white ml-2">{(circuit.build_time || 0).toFixed(2)}s</span>
								</div>
								<div>
									<span class="text-gray-400">Bandwidth:</span>
									<span class="text-white ml-2">{formatBytes(circuit.bandwidth || 0)}/s</span>
								</div>
								<div>
									<span class="text-gray-400">Age:</span>
									<span class="text-white ml-2">{circuit.age || 'Unknown'}</span>
								</div>
							</div>
						</div>
					{/each}
				{/if}
			</div>
			
		{:else}
			<!-- Other tab interfaces -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-white">
					{$selectedTab === 'relays' ? '🔄 Relay Management' :
					 $selectedTab === 'connections' ? '🔗 Connection Monitor' :
					 $selectedTab === 'onion' ? '🧅 Onion Services' :
					 '🔒 Security Controls'}
				</h3>
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">
						{$selectedTab === 'relays' ? '🔄' :
						 $selectedTab === 'connections' ? '🔗' :
						 $selectedTab === 'onion' ? '🧅' : '🔒'}
					</div>
					<p>Detailed {$selectedTab} interface</p>
					<p class="text-sm mt-2">Advanced Tor network management and monitoring</p>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Tor Network | 3-node relay system | Health: {$torState.network_health}
			</div>
			<div class="flex space-x-2">
				<button
					on:click={newTorCircuit}
					class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm font-medium transition-colors"
				>
					New Circuit
				</button>
				<button
					on:click={() => dispatch('export_tor_config')}
					class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
				>
					Export Config
				</button>
			</div>
		</div>
	</div>
</div>

<!-- Relay Detail Modal -->
{#if $selectedRelay}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedRelay.set(null)}>
		<div class="max-w-3xl w-full mx-4 bg-gray-800 rounded-lg p-6" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-purple-400">
					{$selectedRelay.icon} {$selectedRelay.name}
				</h3>
				<button
					on:click={() => selectedRelay.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Relay Information</h4>
					<div class="grid grid-cols-2 gap-4 text-sm">
						<div>
							<span class="text-gray-400">Type:</span>
							<span class="text-white ml-2">{$selectedRelay.type}</span>
						</div>
						<div>
							<span class="text-gray-400">Status:</span>
							<span class="text-white ml-2 {getStatusColor($selectedRelay.status)}">{$selectedRelay.status}</span>
						</div>
						<div>
							<span class="text-gray-400">Port:</span>
							<span class="text-white ml-2">{$selectedRelay.port}</span>
						</div>
						<div>
							<span class="text-gray-400">Uptime:</span>
							<span class="text-white ml-2">{formatUptime($selectedRelay.uptime || 0)}</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Performance Metrics</h4>
					<div class="space-y-2 text-sm">
						<div class="flex justify-between">
							<span class="text-gray-400">Bandwidth:</span>
							<span class="text-white">{formatBytes($selectedRelay.bandwidth || 0)}/s</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Active Connections:</span>
							<span class="text-white">{$selectedRelay.connections || 0}</span>
						</div>
					</div>
				</div>
				
				<div class="flex space-x-2">
					<button
						on:click={() => restartRelay($selectedRelay.id)}
						class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded transition-colors"
					>
						Restart Relay
					</button>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.tor-manager {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Tor-themed styling */
	:global(.tor-manager .tor-active) {
		animation: pulse-purple 2s infinite;
	}
	
	@keyframes pulse-purple {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.8; }
	}
	
	/* Custom scrollbar */
	:global(.tor-manager *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.tor-manager *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.tor-manager *::-webkit-scrollbar-thumb) {
		background: #8b5cf6;
		border-radius: 3px;
	}
	
	:global(.tor-manager *::-webkit-scrollbar-thumb:hover) {
		background: #a78bfa;
	}
</style>