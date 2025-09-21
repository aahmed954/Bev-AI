<!--
OSINT Investigation Panel with Avatar Integration
Features: Real-time investigation progress, avatar feedback, result visualization
Connected to: Avatar WebSocket Client and OSINT analyzers
-->

<script lang="ts">
	import { onMount, onDestroy, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { avatarClient, OSINTAvatarIntegration } from '$lib/services/AvatarWebSocketClient';
	
	const dispatch = createEventDispatcher();
	
	// Investigation state
	const investigationState = writable({
		status: 'idle', // 'idle', 'running', 'paused', 'completed', 'error'
		current_investigation: null as any,
		progress: 0,
		active_analyzers: [] as string[],
		findings: [] as any[],
		threats: [] as any[],
		start_time: null as Date | null,
		estimated_completion: null as Date | null,
		total_tasks: 0,
		completed_tasks: 0
	});
	
	// Avatar feedback state
	const avatarFeedback = writable({
		current_emotion: 'neutral',
		last_comment: '',
		interaction_count: 0,
		suggestions: [] as string[],
		engagement_level: 'normal' // 'low', 'normal', 'high', 'excited'
	});
	
	// Investigation configuration
	let target = '';
	let selectedAnalyzers = new Set(['breach-lookup', 'social-media', 'crypto-tracker']);
	let investigationDepth = 'standard'; // 'quick', 'standard', 'deep', 'comprehensive'
	let priority = 'normal'; // 'low', 'normal', 'high', 'urgent'
	
	// Available analyzers
	const availableAnalyzers = [
		{ id: 'breach-lookup', name: 'Breach Database', icon: '🔍', description: 'Search data breach databases' },
		{ id: 'social-media', name: 'Social Media', icon: '📱', description: 'Analyze social media presence' },
		{ id: 'crypto-tracker', name: 'Crypto Tracker', icon: '₿', description: 'Track cryptocurrency transactions' },
		{ id: 'darknet-markets', name: 'Darknet Markets', icon: '🕸️', description: 'Monitor darknet marketplace activity' },
		{ id: 'reputation-scoring', name: 'Reputation Scoring', icon: '⭐', description: 'Calculate reputation scores' },
		{ id: 'metadata-extractor', name: 'Metadata Extractor', icon: '🔬', description: 'Extract file metadata' },
		{ id: 'watermark-analyzer', name: 'Watermark Analyzer', icon: '🖼️', description: 'Analyze image watermarks' },
		{ id: 'threat-intel', name: 'Threat Intelligence', icon: '⚠️', description: 'Cross-reference threat databases' }
	];
	
	// Real-time updates
	let unsubscribeAvatar: (() => void) | null = null;
	let progressUpdateInterval: number | null = null;
	
	// UI state
	let showAdvancedOptions = false;
	let selectedTab = 'progress'; // 'progress', 'findings', 'avatar', 'settings'
	
	onMount(async () => {
		await initializeAvatarIntegration();
		setupEventListeners();
	});
	
	onDestroy(() => {
		cleanup();
	});
	
	async function initializeAvatarIntegration() {
		try {
			await OSINTAvatarIntegration.initialize();
			
			// Subscribe to avatar events
			unsubscribeAvatar = avatarClient.subscribe('osint_event', handleAvatarEvent);
			
			// Initial avatar greeting
			await avatarClient.setEmotion('friendly');
			await avatarClient.speak('Ready to begin OSINT investigation. Select your target and analyzers.', 'friendly');
			
		} catch (error) {
			console.error('Failed to initialize avatar integration:', error);
		}
	}
	
	function setupEventListeners() {
		// Listen for analyzer events from other components
		window.addEventListener('analyzer_progress', handleAnalyzerProgress);
		window.addEventListener('analyzer_finding', handleAnalyzerFinding);
		window.addEventListener('analyzer_complete', handleAnalyzerComplete);
	}
	
	function handleAvatarEvent(event: any) {
		// Update UI based on avatar feedback
		avatarFeedback.update(feedback => ({
			...feedback,
			current_emotion: event.data.emotion || feedback.current_emotion,
			last_comment: event.data.comment || feedback.last_comment,
			interaction_count: feedback.interaction_count + 1
		}));
	}
	
	async function startInvestigation() {
		if (!target.trim()) {
			await avatarClient.expressConcern('Please specify a target for investigation');
			return;
		}
		
		if (selectedAnalyzers.size === 0) {
			await avatarClient.expressConcern('Please select at least one analyzer');
			return;
		}
		
		const investigationId = generateInvestigationId();
		const analyzersArray = Array.from(selectedAnalyzers);
		
		// Update state
		investigationState.update(state => ({
			...state,
			status: 'running',
			current_investigation: {
				id: investigationId,
				target,
				analyzers: analyzersArray,
				depth: investigationDepth,
				priority
			},
			progress: 0,
			active_analyzers: analyzersArray,
			findings: [],
			threats: [],
			start_time: new Date(),
			total_tasks: analyzersArray.length,
			completed_tasks: 0
		}));
		
		// Avatar announcement
		await OSINTAvatarIntegration.announceAnalyzerStart(
			`${analyzersArray.length} analyzers`,
			target
		);
		
		// Start investigation
		try {
			for (const analyzerId of analyzersArray) {
				const analyzer = availableAnalyzers.find(a => a.id === analyzerId);
				if (analyzer) {
					await startAnalyzer(analyzerId, analyzer.name);
				}
			}
			
			// Start progress monitoring
			startProgressMonitoring();
			
		} catch (error) {
			console.error('Failed to start investigation:', error);
			await avatarClient.expressConcern('Investigation failed to start. Please check system status.');
			
			investigationState.update(state => ({
				...state,
				status: 'error'
			}));
		}
	}
	
	async function startAnalyzer(analyzerId: string, analyzerName: string) {
		try {
			const response = await fetch(`http://localhost:3010/analyze/${analyzerId}`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					target,
					depth: investigationDepth,
					priority,
					investigation_id: $investigationState.current_investigation?.id
				})
			});
			
			if (!response.ok) {
				throw new Error(`Analyzer ${analyzerName} failed to start`);
			}
			
			await OSINTAvatarIntegration.announceAnalyzerStart(analyzerName, target);
			
		} catch (error) {
			console.error(`Failed to start ${analyzerName}:`, error);
			await avatarClient.expressConcern(`Failed to start ${analyzerName} analyzer`);
		}
	}
	
	function startProgressMonitoring() {
		progressUpdateInterval = window.setInterval(async () => {
			await updateInvestigationProgress();
		}, 2000);
	}
	
	async function updateInvestigationProgress() {
		try {
			const investigationId = $investigationState.current_investigation?.id;
			if (!investigationId) return;
			
			const response = await fetch(`http://localhost:3010/investigation/${investigationId}/status`);
			const statusData = await response.json();
			
			investigationState.update(state => ({
				...state,
				progress: statusData.progress,
				completed_tasks: statusData.completed_tasks,
				findings: statusData.findings || state.findings,
				threats: statusData.threats || state.threats
			}));
			
			// Avatar progress feedback
			if (statusData.progress > 75 && $avatarFeedback.engagement_level !== 'excited') {
				avatarFeedback.update(f => ({ ...f, engagement_level: 'excited' }));
				await avatarClient.setEmotion('excited');
			} else if (statusData.progress > 50 && $avatarFeedback.engagement_level === 'normal') {
				avatarFeedback.update(f => ({ ...f, engagement_level: 'high' }));
				await avatarClient.setEmotion('determined');
			}
			
			// Check for completion
			if (statusData.progress >= 100) {
				await completeInvestigation();
			}
			
		} catch (error) {
			console.error('Failed to update investigation progress:', error);
		}
	}
	
	async function completeInvestigation() {
		const state = $investigationState;
		if (!state.current_investigation) return;
		
		// Stop progress monitoring
		if (progressUpdateInterval) {
			clearInterval(progressUpdateInterval);
			progressUpdateInterval = null;
		}
		
		// Calculate completion time
		const completionTime = state.start_time ? 
			Date.now() - state.start_time.getTime() : 0;
		
		// Update state
		investigationState.update(s => ({
			...s,
			status: 'completed',
			estimated_completion: new Date()
		}));
		
		// Avatar completion announcement
		await avatarClient.completeInvestigation(
			state.current_investigation.id,
			{
				findings_count: state.findings.length,
				threats_detected: state.threats.length,
				completion_time: completionTime,
				success: true
			}
		);
		
		dispatch('investigation_complete', {
			id: state.current_investigation.id,
			findings: state.findings,
			threats: state.threats,
			duration: completionTime
		});
	}
	
	async function pauseInvestigation() {
		investigationState.update(state => ({
			...state,
			status: 'paused'
		}));
		
		if (progressUpdateInterval) {
			clearInterval(progressUpdateInterval);
			progressUpdateInterval = null;
		}
		
		await avatarClient.setEmotion('waiting');
		await avatarClient.speak('Investigation paused', 'neutral');
	}
	
	async function resumeInvestigation() {
		investigationState.update(state => ({
			...state,
			status: 'running'
		}));
		
		startProgressMonitoring();
		
		await avatarClient.setEmotion('determined');
		await avatarClient.speak('Resuming investigation', 'determined');
	}
	
	async function stopInvestigation() {
		investigationState.update(state => ({
			...state,
			status: 'idle',
			current_investigation: null,
			progress: 0,
			active_analyzers: [],
			findings: [],
			threats: []
		}));
		
		if (progressUpdateInterval) {
			clearInterval(progressUpdateInterval);
			progressUpdateInterval = null;
		}
		
		await avatarClient.setEmotion('neutral');
		await avatarClient.speak('Investigation stopped', 'neutral');
	}
	
	function handleAnalyzerProgress(event: CustomEvent) {
		const { analyzer, progress, current_task } = event.detail;
		
		// Update progress for specific analyzer
		// Implementation depends on how analyzer progress is tracked
	}
	
	function handleAnalyzerFinding(event: CustomEvent) {
		const { analyzer, finding } = event.detail;
		
		investigationState.update(state => ({
			...state,
			findings: [...state.findings, { ...finding, source: analyzer, timestamp: new Date() }]
		}));
		
		// Avatar response to findings
		if (finding.severity === 'critical') {
			avatarClient.announceThreat(finding.severity, finding.description);
		}
	}
	
	function handleAnalyzerComplete(event: CustomEvent) {
		const { analyzer, findings } = event.detail;
		
		investigationState.update(state => ({
			...state,
			completed_tasks: state.completed_tasks + 1,
			active_analyzers: state.active_analyzers.filter(a => a !== analyzer)
		}));
		
		OSINTAvatarIntegration.announceFindings(analyzer, findings);
	}
	
	function generateInvestigationId(): string {
		return `inv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
	}
	
	function formatDuration(ms: number): string {
		const seconds = Math.floor(ms / 1000);
		const minutes = Math.floor(seconds / 60);
		const hours = Math.floor(minutes / 60);
		
		if (hours > 0) {
			return `${hours}h ${minutes % 60}m`;
		} else if (minutes > 0) {
			return `${minutes}m ${seconds % 60}s`;
		} else {
			return `${seconds}s`;
		}
	}
	
	function getSeverityColor(severity: string): string {
		switch (severity) {
			case 'critical': return 'text-red-400 bg-red-900';
			case 'high': return 'text-orange-400 bg-orange-900';
			case 'medium': return 'text-yellow-400 bg-yellow-900';
			case 'low': return 'text-blue-400 bg-blue-900';
			default: return 'text-gray-400 bg-gray-900';
		}
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'running': return 'text-green-400';
			case 'paused': return 'text-yellow-400';
			case 'completed': return 'text-blue-400';
			case 'error': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}
	
	function cleanup() {
		if (unsubscribeAvatar) {
			unsubscribeAvatar();
		}
		
		if (progressUpdateInterval) {
			clearInterval(progressUpdateInterval);
		}
		
		window.removeEventListener('analyzer_progress', handleAnalyzerProgress);
		window.removeEventListener('analyzer_finding', handleAnalyzerFinding);
		window.removeEventListener('analyzer_complete', handleAnalyzerComplete);
	}
</script>

<!-- OSINT Investigation Panel -->
<div class="osint-investigation-panel h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h2 class="text-xl font-bold text-cyan-400">OSINT Investigation Center</h2>
			<div class="flex items-center space-x-4">
				<!-- Status indicator -->
				<div class="flex items-center space-x-2">
					<div class="w-2 h-2 rounded-full {getStatusColor($investigationState.status)}"></div>
					<span class="text-sm text-gray-400 capitalize">{$investigationState.status}</span>
				</div>
				
				<!-- Avatar emotion indicator -->
				<div class="flex items-center space-x-2">
					<span class="text-sm text-gray-400">Avatar:</span>
					<span class="text-sm text-purple-400 capitalize">{$avatarFeedback.current_emotion}</span>
				</div>
			</div>
		</div>
	</div>
	
	<!-- Investigation Configuration -->
	{#if $investigationState.status === 'idle'}
		<div class="p-6">
			<div class="space-y-6">
				<!-- Target input -->
				<div>
					<label class="block text-sm font-medium text-gray-300 mb-2">Investigation Target</label>
					<input
						type="text"
						bind:value={target}
						placeholder="Enter username, email, domain, IP address, or cryptocurrency address"
						class="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500 text-white"
					/>
				</div>
				
				<!-- Analyzer selection -->
				<div>
					<label class="block text-sm font-medium text-gray-300 mb-3">Select Analyzers</label>
					<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
						{#each availableAnalyzers as analyzer}
							<label class="flex items-center p-3 bg-gray-800 rounded-lg border border-gray-700 hover:border-gray-600 cursor-pointer transition-colors">
								<input
									type="checkbox"
									bind:group={selectedAnalyzers}
									value={analyzer.id}
									class="mr-3 w-4 h-4 text-cyan-600 bg-gray-700 border-gray-600 rounded focus:ring-cyan-500"
								/>
								<div class="flex-1">
									<div class="flex items-center space-x-2">
										<span class="text-lg">{analyzer.icon}</span>
										<span class="font-medium text-white">{analyzer.name}</span>
									</div>
									<p class="text-xs text-gray-400 mt-1">{analyzer.description}</p>
								</div>
							</label>
						{/each}
					</div>
				</div>
				
				<!-- Advanced options -->
				<div>
					<button
						on:click={() => showAdvancedOptions = !showAdvancedOptions}
						class="flex items-center space-x-2 text-cyan-400 hover:text-cyan-300 transition-colors"
					>
						<span>Advanced Options</span>
						<span class="transform transition-transform {showAdvancedOptions ? 'rotate-180' : ''}">(▼)</span>
					</button>
					
					{#if showAdvancedOptions}
						<div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Investigation Depth</label>
								<select
									bind:value={investigationDepth}
									class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500 text-white"
								>
									<option value="quick">Quick Scan (5-10 min)</option>
									<option value="standard">Standard Analysis (15-30 min)</option>
									<option value="deep">Deep Investigation (45-60 min)</option>
									<option value="comprehensive">Comprehensive (1-2 hours)</option>
								</select>
							</div>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Priority Level</label>
								<select
									bind:value={priority}
									class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500 text-white"
								>
									<option value="low">Low Priority</option>
									<option value="normal">Normal Priority</option>
									<option value="high">High Priority</option>
									<option value="urgent">Urgent</option>
								</select>
							</div>
						</div>
					{/if}
				</div>
				
				<!-- Start button -->
				<button
					on:click={startInvestigation}
					disabled={!target.trim() || selectedAnalyzers.size === 0}
					class="w-full px-6 py-3 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
				>
					Start Investigation
				</button>
			</div>
		</div>
	{/if}
	
	<!-- Active Investigation -->
	{#if $investigationState.status !== 'idle'}
		<div class="flex-1 flex flex-col">
			<!-- Investigation header -->
			<div class="p-4 border-b border-gray-800">
				<div class="flex items-center justify-between">
					<div>
						<h3 class="font-semibold text-white">Target: {$investigationState.current_investigation?.target}</h3>
						<p class="text-sm text-gray-400">
							{$investigationState.active_analyzers.length} analyzers active • 
							{$investigationState.findings.length} findings • 
							{$investigationState.threats.length} threats
						</p>
					</div>
					
					<div class="flex space-x-2">
						{#if $investigationState.status === 'running'}
							<button
								on:click={pauseInvestigation}
								class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded text-sm transition-colors"
							>
								Pause
							</button>
						{:else if $investigationState.status === 'paused'}
							<button
								on:click={resumeInvestigation}
								class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm transition-colors"
							>
								Resume
							</button>
						{/if}
						
						<button
							on:click={stopInvestigation}
							class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors"
						>
							Stop
						</button>
					</div>
				</div>
				
				<!-- Progress bar -->
				<div class="mt-4">
					<div class="flex justify-between text-sm text-gray-400 mb-1">
						<span>Progress</span>
						<span>{Math.round($investigationState.progress)}%</span>
					</div>
					<div class="w-full bg-gray-700 rounded-full h-2">
						<div
							class="bg-cyan-500 h-2 rounded-full transition-all duration-500"
							style="width: {$investigationState.progress}%"
						></div>
					</div>
				</div>
			</div>
			
			<!-- Tabs -->
			<div class="border-b border-gray-800">
				<nav class="flex space-x-8 px-4">
					{#each [
						{ id: 'progress', label: 'Progress', icon: '📊' },
						{ id: 'findings', label: 'Findings', icon: '🔍', count: $investigationState.findings.length },
						{ id: 'threats', label: 'Threats', icon: '⚠️', count: $investigationState.threats.length },
						{ id: 'avatar', label: 'Avatar', icon: '🤖' }
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
			
			<!-- Tab content -->
			<div class="flex-1 overflow-y-auto p-4">
				{#if selectedTab === 'progress'}
					<!-- Progress tab -->
					<div class="space-y-4">
						<!-- Active analyzers -->
						<div>
							<h4 class="font-medium text-white mb-3">Active Analyzers</h4>
							<div class="grid grid-cols-1 md:grid-cols-2 gap-3">
								{#each $investigationState.active_analyzers as analyzerId}
									{@const analyzer = availableAnalyzers.find(a => a.id === analyzerId)}
									{#if analyzer}
										<div class="bg-gray-800 rounded-lg p-4">
											<div class="flex items-center space-x-3">
												<span class="text-2xl">{analyzer.icon}</span>
												<div class="flex-1">
													<h5 class="font-medium text-white">{analyzer.name}</h5>
													<p class="text-sm text-gray-400">Running analysis...</p>
												</div>
												<div class="w-6 h-6 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
											</div>
										</div>
									{/if}
								{/each}
							</div>
						</div>
						
						<!-- Timeline -->
						<div>
							<h4 class="font-medium text-white mb-3">Investigation Timeline</h4>
							<div class="space-y-3">
								{#if $investigationState.start_time}
									<div class="flex items-center space-x-3">
										<div class="w-2 h-2 bg-green-400 rounded-full"></div>
										<span class="text-sm text-gray-300">
											Started at {$investigationState.start_time.toLocaleTimeString()}
										</span>
									</div>
								{/if}
								
								{#if $investigationState.estimated_completion}
									<div class="flex items-center space-x-3">
										<div class="w-2 h-2 bg-blue-400 rounded-full"></div>
										<span class="text-sm text-gray-300">
											Completed at {$investigationState.estimated_completion.toLocaleTimeString()}
										</span>
									</div>
								{/if}
							</div>
						</div>
					</div>
					
				{:else if selectedTab === 'findings'}
					<!-- Findings tab -->
					<div class="space-y-4">
						{#if $investigationState.findings.length === 0}
							<div class="text-center py-8 text-gray-400">
								<div class="text-4xl mb-2">🔍</div>
								<p>No findings yet</p>
							</div>
						{:else}
							{#each $investigationState.findings as finding}
								<div class="bg-gray-800 rounded-lg p-4">
									<div class="flex items-start justify-between">
										<div class="flex-1">
											<div class="flex items-center space-x-2 mb-2">
												<span class="px-2 py-1 rounded text-xs font-medium {getSeverityColor(finding.severity)}">
													{finding.severity?.toUpperCase()}
												</span>
												<span class="text-sm text-gray-400">{finding.source}</span>
											</div>
											<h5 class="font-medium text-white mb-1">{finding.title}</h5>
											<p class="text-sm text-gray-300">{finding.description}</p>
										</div>
										<span class="text-xs text-gray-500">
											{finding.timestamp?.toLocaleTimeString()}
										</span>
									</div>
								</div>
							{/each}
						{/if}
					</div>
					
				{:else if selectedTab === 'threats'}
					<!-- Threats tab -->
					<div class="space-y-4">
						{#if $investigationState.threats.length === 0}
							<div class="text-center py-8 text-gray-400">
								<div class="text-4xl mb-2">🛡️</div>
								<p>No threats detected</p>
							</div>
						{:else}
							{#each $investigationState.threats as threat}
								<div class="bg-red-900 border border-red-700 rounded-lg p-4">
									<div class="flex items-start justify-between">
										<div class="flex-1">
											<div class="flex items-center space-x-2 mb-2">
												<span class="px-2 py-1 rounded text-xs font-medium bg-red-600 text-white">
													THREAT
												</span>
												<span class="text-sm text-red-300">{threat.source}</span>
											</div>
											<h5 class="font-medium text-white mb-1">{threat.title}</h5>
											<p class="text-sm text-red-200">{threat.description}</p>
										</div>
										<span class="text-xs text-red-400">
											{threat.timestamp?.toLocaleTimeString()}
										</span>
									</div>
								</div>
							{/each}
						{/if}
					</div>
					
				{:else if selectedTab === 'avatar'}
					<!-- Avatar tab -->
					<div class="space-y-4">
						<div class="bg-gray-800 rounded-lg p-4">
							<h4 class="font-medium text-white mb-3">Avatar Status</h4>
							<div class="grid grid-cols-2 gap-4">
								<div>
									<span class="text-sm text-gray-400">Current Emotion:</span>
									<span class="ml-2 text-purple-400 capitalize">{$avatarFeedback.current_emotion}</span>
								</div>
								<div>
									<span class="text-sm text-gray-400">Engagement:</span>
									<span class="ml-2 text-cyan-400 capitalize">{$avatarFeedback.engagement_level}</span>
								</div>
								<div>
									<span class="text-sm text-gray-400">Interactions:</span>
									<span class="ml-2 text-white">{$avatarFeedback.interaction_count}</span>
								</div>
							</div>
						</div>
						
						{#if $avatarFeedback.last_comment}
							<div class="bg-purple-900 border border-purple-700 rounded-lg p-4">
								<h5 class="font-medium text-white mb-2">Latest Avatar Comment</h5>
								<p class="text-purple-200">"{$avatarFeedback.last_comment}"</p>
							</div>
						{/if}
						
						{#if $avatarFeedback.suggestions.length > 0}
							<div class="bg-cyan-900 border border-cyan-700 rounded-lg p-4">
								<h5 class="font-medium text-white mb-2">Avatar Suggestions</h5>
								<ul class="space-y-1">
									{#each $avatarFeedback.suggestions as suggestion}
										<li class="text-cyan-200 text-sm">• {suggestion}</li>
									{/each}
								</ul>
							</div>
						{/if}
					</div>
				{/if}
			</div>
		</div>
	{/if}
</div>

<style>
	.osint-investigation-panel {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
</style>