<!--
Security Operations Center - Military-Grade Security Command Dashboard
Connected to: src/security/ (8 security systems)
Features: Real-time threat monitoring, automated response, multi-system correlation
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	// Sub-components for each security system
	import GuardianSecurityDashboard from './GuardianSecurityDashboard.svelte';
	import IntrusionDetectionPanel from './IntrusionDetectionPanel.svelte';
	import TrafficAnalyzer from './TrafficAnalyzer.svelte';
	import AnomalyDetectionMatrix from './AnomalyDetectionMatrix.svelte';
	import TacticalIntelligenceWorkbench from './TacticalIntelligenceWorkbench.svelte';
	import DefenseAutomationControls from './DefenseAutomationControls.svelte';
	import OPSECEnforcerDashboard from './OPSECEnforcerDashboard.svelte';
	import IntelFusionCenter from './IntelFusionCenter.svelte';
	
	const dispatch = createEventDispatcher();
	
	// Central SOC State Management
	const socState = writable({
		threat_level: 'GREEN', // GREEN, YELLOW, ORANGE, RED, CRITICAL
		active_threats: [],
		security_systems: {
			guardian: { status: 'active', threats_blocked: 0, last_action: null },
			intrusion_detection: { status: 'active', detections: 0, false_positives: 0 },
			traffic_analyzer: { status: 'active', suspicious_flows: 0, bandwidth_anomalies: 0 },
			anomaly_detector: { status: 'active', anomalies_detected: 0, confidence: 95 },
			tactical_intelligence: { status: 'active', intel_items: 0, correlations: 0 },
			defense_automation: { status: 'active', automated_responses: 0, blocked_ips: 0 },
			opsec_enforcer: { status: 'active', violations: 0, enforcements: 0 },
			intel_fusion: { status: 'active', sources: 8, fused_intel: 0 }
		},
		alerts: [],
		defensive_posture: 'NORMAL', // NORMAL, ELEVATED, HIGH, MAXIMUM
		auto_response: true,
		correlation_engine: true
	});
	
	const selectedSystem = writable('overview');
	const alertLevel = writable('all'); // 'all', 'critical', 'high', 'medium', 'low'
	const isLoading = writable(false);
	
	// WebSocket connections for all security systems
	let securityWebSockets: Map<string, WebSocket> = new Map();
	
	// Threat correlation data
	let threatCorrelations: any[] = [];
	let securityMetrics = {
		threats_blocked_24h: 0,
		incidents_resolved: 0,
		mean_time_to_detection: 0,
		mean_time_to_response: 0,
		false_positive_rate: 0.05,
		system_availability: 99.9
	};
	
	const threatLevels = {
		'GREEN': { color: 'text-green-400', bg: 'bg-green-900/20', description: 'Normal Operations' },
		'YELLOW': { color: 'text-yellow-400', bg: 'bg-yellow-900/20', description: 'Elevated Alert' },
		'ORANGE': { color: 'text-orange-400', bg: 'bg-orange-900/20', description: 'High Alert' },
		'RED': { color: 'text-red-400', bg: 'bg-red-900/20', description: 'Critical Threat' },
		'CRITICAL': { color: 'text-red-600', bg: 'bg-red-900/40', description: 'Maximum Alert' }
	};
	
	const securitySystems = [
		{ id: 'guardian', name: 'Guardian Enforcer', icon: '🛡️', port: 8200 },
		{ id: 'intrusion_detection', name: 'Intrusion Detection', icon: '🔍', port: 8201 },
		{ id: 'traffic_analyzer', name: 'Traffic Analyzer', icon: '📊', port: 8202 },
		{ id: 'anomaly_detector', name: 'Anomaly Detector', icon: '⚠️', port: 8203 },
		{ id: 'tactical_intelligence', name: 'Tactical Intel', icon: '🎯', port: 8204 },
		{ id: 'defense_automation', name: 'Defense Automation', icon: '🤖', port: 8205 },
		{ id: 'opsec_enforcer', name: 'OPSEC Enforcer', icon: '🔒', port: 8206 },
		{ id: 'intel_fusion', name: 'Intel Fusion', icon: '🧠', port: 8207 }
	];
	
	onMount(async () => {
		await initializeAllSecuritySystems();
		await loadSecurityMetrics();
		startThreatCorrelation();
	});
	
	async function initializeAllSecuritySystems() {
		// Initialize WebSocket connections for all 8 security systems
		for (const system of securitySystems) {
			try {
				const ws = new WebSocket(`ws://localhost:${system.port}/ws`);
				
				ws.onopen = () => {
					console.log(`${system.name} WebSocket connected`);
					updateSystemStatus(system.id, 'connected');
				};
				
				ws.onmessage = (event) => {
					const data = JSON.parse(event.data);
					handleSecurityUpdate(system.id, data);
				};
				
				ws.onclose = () => {
					console.log(`${system.name} WebSocket disconnected`);
					updateSystemStatus(system.id, 'disconnected');
					setTimeout(() => reconnectSystem(system), 5000);
				};
				
				securityWebSockets.set(system.id, ws);
			} catch (error) {
				console.error(`Failed to connect to ${system.name}:`, error);
			}
		}
	}
	
	function handleSecurityUpdate(systemId: string, data: any) {
		switch (data.type) {
			case 'threat_detected':
				addThreat(systemId, data.threat);
				break;
			case 'threat_blocked':
				recordThreatBlocked(systemId, data.threat);
				break;
			case 'system_metrics':
				updateSystemMetrics(systemId, data.metrics);
				break;
			case 'alert':
				addSecurityAlert(systemId, data.alert);
				break;
			case 'correlation':
				addThreatCorrelation(data.correlation);
				break;
		}
	}
	
	async function loadSecurityMetrics() {
		isLoading.set(true);
		try {
			const metrics = await invoke('get_security_metrics');
			securityMetrics = { ...securityMetrics, ...metrics };
		} catch (error) {
			console.error('Failed to load security metrics:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function startThreatCorrelation() {
		// Start threat correlation engine
		setInterval(async () => {
			try {
				const correlations = await invoke('correlate_threats');
				threatCorrelations = correlations;
				updateThreatLevel();
			} catch (error) {
				console.error('Threat correlation failed:', error);
			}
		}, 5000);
	}
	
	function addThreat(systemId: string, threat: any) {
		socState.update(current => ({
			...current,
			active_threats: [{
				id: `${systemId}_${Date.now()}`,
				system: systemId,
				...threat,
				detected_at: new Date().toISOString()
			}, ...current.active_threats.slice(0, 99)]
		}));
		
		updateThreatLevel();
	}
	
	function recordThreatBlocked(systemId: string, threat: any) {
		socState.update(current => ({
			...current,
			security_systems: {
				...current.security_systems,
				[systemId]: {
					...current.security_systems[systemId],
					threats_blocked: current.security_systems[systemId].threats_blocked + 1,
					last_action: new Date().toISOString()
				}
			}
		}));
	}
	
	function updateSystemStatus(systemId: string, status: string) {
		socState.update(current => ({
			...current,
			security_systems: {
				...current.security_systems,
				[systemId]: {
					...current.security_systems[systemId],
					status
				}
			}
		}));
	}
	
	function updateSystemMetrics(systemId: string, metrics: any) {
		socState.update(current => ({
			...current,
			security_systems: {
				...current.security_systems,
				[systemId]: {
					...current.security_systems[systemId],
					...metrics
				}
			}
		}));
	}
	
	function addSecurityAlert(systemId: string, alert: any) {
		socState.update(current => ({
			...current,
			alerts: [{
				id: `${systemId}_alert_${Date.now()}`,
				system: systemId,
				...alert,
				timestamp: new Date().toISOString()
			}, ...current.alerts.slice(0, 49)]
		}));
	}
	
	function addThreatCorrelation(correlation: any) {
		threatCorrelations = [correlation, ...threatCorrelations.slice(0, 19)];
	}
	
	function updateThreatLevel() {
		const threats = $socState.active_threats;
		const criticalThreats = threats.filter(t => t.severity === 'critical').length;
		const highThreats = threats.filter(t => t.severity === 'high').length;
		
		let newThreatLevel = 'GREEN';
		
		if (criticalThreats > 0) {
			newThreatLevel = 'CRITICAL';
		} else if (highThreats > 5) {
			newThreatLevel = 'RED';
		} else if (highThreats > 2) {
			newThreatLevel = 'ORANGE';
		} else if (threats.length > 10) {
			newThreatLevel = 'YELLOW';
		}
		
		socState.update(current => ({
			...current,
			threat_level: newThreatLevel
		}));
	}
	
	async function setDefensivePosture(posture: string) {
		try {
			await invoke('set_defensive_posture', { posture });
			socState.update(current => ({
				...current,
				defensive_posture: posture
			}));
		} catch (error) {
			console.error('Failed to set defensive posture:', error);
		}
	}
	
	async function emergencyLockdown() {
		try {
			await invoke('emergency_security_lockdown');
			socState.update(current => ({
				...current,
				threat_level: 'CRITICAL',
				defensive_posture: 'MAXIMUM'
			}));
		} catch (error) {
			console.error('Failed to execute emergency lockdown:', error);
		}
	}
	
	function reconnectSystem(system: any) {
		// Attempt to reconnect failed system
		const ws = new WebSocket(`ws://localhost:${system.port}/ws`);
		securityWebSockets.set(system.id, ws);
	}
	
	function getSystemStatusColor(status: string): string {
		switch (status) {
			case 'active':
			case 'connected': return 'text-green-400';
			case 'warning': return 'text-yellow-400';
			case 'error':
			case 'disconnected': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleString();
	}
</script>

<!-- Security Operations Center -->
<div class="soc-dashboard h-full bg-gray-900 text-white">
	<!-- SOC Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<div class="flex items-center space-x-4">
				<h1 class="text-2xl font-bold text-red-400">🛡️ SECURITY OPERATIONS CENTER</h1>
				<div class="px-4 py-2 rounded-lg {threatLevels[$socState.threat_level].bg} {threatLevels[$socState.threat_level].color} font-bold">
					THREAT LEVEL: {$socState.threat_level}
				</div>
			</div>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{$socState.active_threats.length} active threats | {Object.values($socState.security_systems).filter(s => s.status === 'active').length}/8 systems online
				</div>
				<button
					on:click={emergencyLockdown}
					class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded font-bold transition-colors"
				>
					🚨 EMERGENCY LOCKDOWN
				</button>
			</div>
		</div>
	</div>
	
	<!-- Threat Level Status Bar -->
	<div class="p-4 border-b border-gray-800">
		<div class="flex items-center justify-between">
			<div class="flex items-center space-x-6">
				<div class="{threatLevels[$socState.threat_level].bg} {threatLevels[$socState.threat_level].color} px-4 py-2 rounded-lg">
					<div class="font-bold">THREAT CONDITION: {$socState.threat_level}</div>
					<div class="text-sm">{threatLevels[$socState.threat_level].description}</div>
				</div>
				
				<div class="bg-gray-800 px-4 py-2 rounded-lg">
					<div class="font-bold text-blue-400">DEFENSIVE POSTURE: {$socState.defensive_posture}</div>
					<div class="text-sm text-gray-400">Auto-response: {$socState.auto_response ? 'ENABLED' : 'DISABLED'}</div>
				</div>
			</div>
			
			<!-- Defensive Posture Controls -->
			<div class="flex space-x-2">
				{#each ['NORMAL', 'ELEVATED', 'HIGH', 'MAXIMUM'] as posture}
					<button
						on:click={() => setDefensivePosture(posture)}
						class="px-3 py-1 rounded text-sm font-medium transition-colors {
							$socState.defensive_posture === posture
								? 'bg-blue-600 text-white'
								: 'bg-gray-700 hover:bg-gray-600 text-gray-300'
						}"
					>
						{posture}
					</button>
				{/each}
			</div>
		</div>
	</div>
	
	<!-- Security Systems Status Grid -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
			{#each securitySystems as system}
				{@const systemData = $socState.security_systems[system.id]}
				<div 
					class="bg-gray-800 rounded-lg p-3 cursor-pointer hover:bg-gray-700 transition-colors"
					on:click={() => selectedSystem.set(system.id)}
				>
					<div class="text-center">
						<div class="text-2xl mb-1">{system.icon}</div>
						<div class="font-medium text-white text-sm">{system.name}</div>
						<div class="flex items-center justify-center mt-2">
							<div class="w-2 h-2 rounded-full {getSystemStatusColor(systemData.status)} mr-1"></div>
							<span class="text-xs {getSystemStatusColor(systemData.status)}">{systemData.status}</span>
						</div>
						{#if systemData.threats_blocked || systemData.detections || systemData.anomalies_detected}
							<div class="text-xs text-gray-400 mt-1">
								{systemData.threats_blocked || systemData.detections || systemData.anomalies_detected || 0} events
							</div>
						{/if}
					</div>
				</div>
			{/each}
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'SOC Overview', icon: '📊' },
				{ id: 'guardian', label: 'Guardian', icon: '🛡️' },
				{ id: 'intrusion_detection', label: 'IDS', icon: '🔍' },
				{ id: 'traffic_analyzer', label: 'Traffic', icon: '📊' },
				{ id: 'anomaly_detector', label: 'Anomaly', icon: '⚠️' },
				{ id: 'tactical_intelligence', label: 'Tactical', icon: '🎯' },
				{ id: 'defense_automation', label: 'Defense', icon: '🤖' },
				{ id: 'opsec_enforcer', label: 'OPSEC', icon: '🔒' },
				{ id: 'intel_fusion', label: 'Fusion', icon: '🧠' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedSystem === tab.id
							? 'border-red-500 text-red-400'
							: 'border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-300'
					}"
					on:click={() => selectedSystem.set(tab.id)}
				>
					<span class="mr-2">{tab.icon}</span>
					{tab.label}
				</button>
			{/each}
		</nav>
	</div>
	
	<!-- Main Content Area -->
	<div class="flex-1 overflow-y-auto p-4">
		{#if $selectedSystem === 'overview'}
			<!-- SOC Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Active Threats -->
				<div class="lg:col-span-2">
					<div class="bg-gray-800 rounded-lg p-6">
						<div class="flex items-center justify-between mb-4">
							<h3 class="text-lg font-semibold text-red-400">Active Threats</h3>
							<select
								bind:value={$alertLevel}
								class="bg-gray-700 border border-gray-600 rounded px-3 py-1 text-white text-sm"
							>
								<option value="all">All Threats</option>
								<option value="critical">Critical Only</option>
								<option value="high">High & Critical</option>
								<option value="medium">Medium+</option>
							</select>
						</div>
						
						{#if $socState.active_threats.length === 0}
							<div class="text-center py-8 text-gray-400">
								<div class="text-3xl mb-2">✅</div>
								<p>No active threats detected</p>
								<p class="text-sm mt-1">All systems operational</p>
							</div>
						{:else}
							<div class="space-y-3 max-h-96 overflow-y-auto">
								{#each $socState.active_threats as threat}
									<div class="bg-gray-900 border-l-4 border-red-500 rounded p-4">
										<div class="flex items-center justify-between mb-2">
											<div class="flex items-center space-x-3">
												<span class="font-medium text-white">{threat.type}</span>
												<span class="px-2 py-1 bg-red-600 text-white text-xs rounded">{threat.severity}</span>
												<span class="text-sm text-gray-400">{threat.system}</span>
											</div>
											<div class="text-sm text-gray-400">{formatDate(threat.detected_at)}</div>
										</div>
										<p class="text-gray-300 text-sm">{threat.description}</p>
										{#if threat.source_ip}
											<div class="mt-2 text-xs text-red-400">Source: {threat.source_ip}</div>
										{/if}
									</div>
								{/each}
							</div>
						{/if}
					</div>
				</div>
				
				<!-- Security Metrics -->
				<div class="space-y-6">
					<div class="bg-gray-800 rounded-lg p-6">
						<h3 class="text-lg font-semibold mb-4 text-green-400">Security Metrics</h3>
						<div class="space-y-4">
							<div class="flex justify-between">
								<span class="text-gray-400">Threats Blocked (24h):</span>
								<span class="text-white font-bold">{securityMetrics.threats_blocked_24h}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Incidents Resolved:</span>
								<span class="text-white font-bold">{securityMetrics.incidents_resolved}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">MTTD:</span>
								<span class="text-white font-bold">{securityMetrics.mean_time_to_detection}s</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">MTTR:</span>
								<span class="text-white font-bold">{securityMetrics.mean_time_to_response}s</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">False Positive Rate:</span>
								<span class="text-white font-bold">{(securityMetrics.false_positive_rate * 100).toFixed(2)}%</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">System Availability:</span>
								<span class="text-green-400 font-bold">{securityMetrics.system_availability}%</span>
							</div>
						</div>
					</div>
					
					<!-- Threat Correlations -->
					<div class="bg-gray-800 rounded-lg p-6">
						<h3 class="text-lg font-semibold mb-4 text-purple-400">Threat Correlations</h3>
						{#if threatCorrelations.length === 0}
							<div class="text-center py-4 text-gray-400">
								<p>No correlations detected</p>
							</div>
						{:else}
							<div class="space-y-2">
								{#each threatCorrelations.slice(0, 5) as correlation}
									<div class="bg-gray-900 rounded p-3">
										<div class="font-medium text-white text-sm">{correlation.pattern}</div>
										<div class="text-xs text-gray-400">
											Confidence: {(correlation.confidence * 100).toFixed(1)}%
										</div>
									</div>
								{/each}
							</div>
						{/if}
					</div>
				</div>
			</div>
			
		{:else if $selectedSystem === 'guardian'}
			<GuardianSecurityDashboard />
		{:else if $selectedSystem === 'intrusion_detection'}
			<IntrusionDetectionPanel />
		{:else if $selectedSystem === 'traffic_analyzer'}
			<TrafficAnalyzer />
		{:else if $selectedSystem === 'anomaly_detector'}
			<AnomalyDetectionMatrix />
		{:else if $selectedSystem === 'tactical_intelligence'}
			<TacticalIntelligenceWorkbench />
		{:else if $selectedSystem === 'defense_automation'}
			<DefenseAutomationControls />
		{:else if $selectedSystem === 'opsec_enforcer'}
			<OPSECEnforcerDashboard />
		{:else if $selectedSystem === 'intel_fusion'}
			<IntelFusionCenter />
		{/if}
	</div>
	
	<!-- SOC Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Security Operations Center | All 8 systems integrated
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('export_security_report')}
					class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
				>
					Export Security Report
				</button>
				<button
					on:click={() => dispatch('run_security_audit')}
					class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded text-sm font-medium transition-colors"
				>
					Run Security Audit
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	.soc-dashboard {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Security-themed styling */
	:global(.soc-dashboard .threat-critical) {
		animation: pulse-red 2s infinite;
	}
	
	@keyframes pulse-red {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.5; }
	}
	
	/* Custom scrollbar */
	:global(.soc-dashboard *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.soc-dashboard *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.soc-dashboard *::-webkit-scrollbar-thumb) {
		background: #dc2626;
		border-radius: 3px;
	}
	
	:global(.soc-dashboard *::-webkit-scrollbar-thumb:hover) {
		background: #ef4444;
	}
</style>