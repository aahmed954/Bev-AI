<!--
Social Media Analyzer Dashboard - Multi-Platform OSINT Intelligence
Connected to: intelowl/custom_analyzers/social_analyzer.py
Platforms: Instagram, Twitter, LinkedIn, Facebook
Features: Profile analysis, post tracking, network mapping, sentiment analysis
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	// Platform Analysis Components
	import InstagramAnalyzer from './InstagramAnalyzer.svelte';
	import TwitterAnalyzer from './TwitterAnalyzer.svelte';
	import LinkedInAnalyzer from './LinkedInAnalyzer.svelte';
	import FacebookAnalyzer from './FacebookAnalyzer.svelte';
	import CrossPlatformCorrelation from './CrossPlatformCorrelation.svelte';
	import SocialNetworkGraph from './SocialNetworkGraph.svelte';
	import SentimentAnalysis from './SentimentAnalysis.svelte';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const analysisResults = writable({
		profiles: {},
		posts: [],
		network: { nodes: [], edges: [] },
		images: [],
		patterns: {},
		risk_indicators: [],
		timeline: [],
		correlation_data: {}
	});
	
	const activeTab = writable('dashboard');
	const isAnalyzing = writable(false);
	const analysisProgress = writable(0);
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	
	// Configuration
	let analysisConfig = {
		target: '',
		platforms: ['instagram', 'twitter', 'linkedin'],
		depth: 2,
		max_posts: 100,
		extract_images: true,
		sentiment_analysis: true,
		build_network: true,
		correlation_analysis: true
	};
	
	let recentAnalyses: any[] = [];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadRecentAnalyses();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/social-analyzer');
			
			ws.onopen = () => {
				console.log('Social Media Analyzer WebSocket connected');
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
			case 'analysis_progress':
				analysisProgress.set(data.progress);
				break;
			case 'platform_complete':
				updatePlatformResults(data.platform, data.results);
				break;
			case 'analysis_complete':
				analysisResults.set(data.results);
				isAnalyzing.set(false);
				analysisProgress.set(100);
				break;
			case 'correlation_update':
				updateCorrelationData(data.correlation);
				break;
		}
	}
	
	async function startAnalysis() {
		if (!analysisConfig.target.trim()) {
			alert('Please enter a target (username, email, or profile URL)');
			return;
		}
		
		isAnalyzing.set(true);
		analysisProgress.set(0);
		
		try {
			const result = await invoke('start_social_analysis', {
				target: analysisConfig.target,
				config: analysisConfig
			});
			
			// Send WebSocket message to start analysis
			if (ws && ws.readyState === WebSocket.OPEN) {
				ws.send(JSON.stringify({
					type: 'start_analysis',
					target: analysisConfig.target,
					config: analysisConfig
				}));
			}
		} catch (error) {
			console.error('Failed to start analysis:', error);
			isAnalyzing.set(false);
		}
	}
	
	async function stopAnalysis() {
		isAnalyzing.set(false);
		analysisProgress.set(0);
		
		try {
			await invoke('stop_social_analysis');
			if (ws && ws.readyState === WebSocket.OPEN) {
				ws.send(JSON.stringify({ type: 'stop_analysis' }));
			}
		} catch (error) {
			console.error('Failed to stop analysis:', error);
		}
	}
	
	async function loadRecentAnalyses() {
		try {
			recentAnalyses = await invoke('get_recent_social_analyses');
		} catch (error) {
			console.error('Failed to load recent analyses:', error);
		}
	}
	
	function updatePlatformResults(platform: string, results: any) {
		analysisResults.update(current => ({
			...current,
			profiles: {
				...current.profiles,
				[platform]: results.profile
			},
			posts: [...current.posts, ...results.posts],
			images: [...current.images, ...(results.images || [])]
		}));
	}
	
	function updateCorrelationData(correlation: any) {
		analysisResults.update(current => ({
			...current,
			correlation_data: correlation
		}));
	}
	
	function exportResults() {
		const results = $analysisResults;
		const blob = new Blob([JSON.stringify(results, null, 2)], {
			type: 'application/json'
		});
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `social_analysis_${analysisConfig.target}_${Date.now()}.json`;
		a.click();
		URL.revokeObjectURL(url);
	}
</script>

<!-- Social Media Analyzer Dashboard -->
<div class="social-analyzer-dashboard h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-blue-400">Social Media Intelligence</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					Platforms: {analysisConfig.platforms.length}
				</div>
				{#if $isAnalyzing}
					<div class="flex items-center space-x-2">
						<div class="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
						<span class="text-blue-400">Analyzing... {$analysisProgress}%</span>
					</div>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Analysis Configuration -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-1 lg:grid-cols-4 gap-4">
			<!-- Target Input -->
			<div class="lg:col-span-2">
				<label class="block text-sm font-medium text-gray-300 mb-2">
					Target (Username/Email/Profile URL)
				</label>
				<input
					type="text"
					bind:value={analysisConfig.target}
					placeholder="@username, email@domain.com, or profile URL"
					class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
				/>
			</div>
			
			<!-- Platform Selection -->
			<div>
				<label class="block text-sm font-medium text-gray-300 mb-2">
					Platforms
				</label>
				<div class="space-y-2">
					{#each ['instagram', 'twitter', 'linkedin', 'facebook'] as platform}
						<label class="flex items-center">
							<input
								type="checkbox"
								bind:group={analysisConfig.platforms}
								value={platform}
								class="mr-2 text-blue-500"
							/>
							<span class="capitalize text-sm">{platform}</span>
						</label>
					{/each}
				</div>
			</div>
			
			<!-- Analysis Controls -->
			<div class="flex flex-col justify-end">
				{#if !$isAnalyzing}
					<button
						on:click={startAnalysis}
						class="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-md font-medium transition-colors"
					>
						Start Analysis
					</button>
				{:else}
					<button
						on:click={stopAnalysis}
						class="w-full px-4 py-2 bg-red-600 hover:bg-red-700 rounded-md font-medium transition-colors"
					>
						Stop Analysis
					</button>
				{/if}
			</div>
		</div>
		
		<!-- Progress Bar -->
		{#if $isAnalyzing}
			<div class="mt-4">
				<div class="flex justify-between text-sm text-gray-400 mb-1">
					<span>Analysis Progress</span>
					<span>{$analysisProgress}%</span>
				</div>
				<div class="w-full bg-gray-700 rounded-full h-2">
					<div
						class="bg-blue-600 h-2 rounded-full transition-all duration-300"
						style="width: {$analysisProgress}%"
					></div>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'dashboard', label: 'Dashboard', icon: '📊' },
				{ id: 'instagram', label: 'Instagram', icon: '📷' },
				{ id: 'twitter', label: 'Twitter', icon: '🐦' },
				{ id: 'linkedin', label: 'LinkedIn', icon: '💼' },
				{ id: 'facebook', label: 'Facebook', icon: '👥' },
				{ id: 'correlation', label: 'Cross-Platform', icon: '🔗' },
				{ id: 'network', label: 'Network', icon: '🕸️' },
				{ id: 'sentiment', label: 'Sentiment', icon: '🎭' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$activeTab === tab.id
							? 'border-blue-500 text-blue-400'
							: 'border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-300'
					}"
					on:click={() => activeTab.set(tab.id)}
				>
					<span class="mr-2">{tab.icon}</span>
					{tab.label}
				</button>
			{/each}
		</nav>
	</div>
	
	<!-- Main Content Area -->
	<div class="flex-1 overflow-y-auto p-4">
		{#if $activeTab === 'dashboard'}
			<!-- Dashboard Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Analysis Summary -->
				<div class="lg:col-span-2">
					<div class="bg-gray-800 rounded-lg p-6">
						<h3 class="text-lg font-semibold mb-4">Analysis Summary</h3>
						<div class="grid grid-cols-2 gap-4">
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-blue-400">
									{Object.keys($analysisResults.profiles).length}
								</div>
								<div class="text-sm text-gray-400">Profiles Found</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-green-400">
									{$analysisResults.posts.length}
								</div>
								<div class="text-sm text-gray-400">Posts Analyzed</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-yellow-400">
									{$analysisResults.network.nodes.length}
								</div>
								<div class="text-sm text-gray-400">Network Nodes</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-red-400">
									{$analysisResults.risk_indicators.length}
								</div>
								<div class="text-sm text-gray-400">Risk Indicators</div>
							</div>
						</div>
					</div>
				</div>
				
				<!-- Recent Analyses -->
				<div>
					<div class="bg-gray-800 rounded-lg p-6">
						<h3 class="text-lg font-semibold mb-4">Recent Analyses</h3>
						<div class="space-y-3">
							{#each recentAnalyses.slice(0, 5) as analysis}
								<div class="bg-gray-900 rounded p-3">
									<div class="font-medium text-sm">{analysis.target}</div>
									<div class="text-xs text-gray-400 mt-1">
										{new Date(analysis.timestamp).toLocaleDateString()}
									</div>
									<div class="flex justify-between items-center mt-2">
										<div class="text-xs text-blue-400">
											{analysis.platforms.length} platforms
										</div>
										<button class="text-xs text-green-400 hover:underline">
											View
										</button>
									</div>
								</div>
							{/each}
						</div>
					</div>
				</div>
			</div>
			
			<!-- Risk Indicators -->
			{#if $analysisResults.risk_indicators.length > 0}
				<div class="mt-6">
					<div class="bg-gray-800 rounded-lg p-6">
						<h3 class="text-lg font-semibold mb-4 text-red-400">🚨 Risk Indicators</h3>
						<div class="space-y-3">
							{#each $analysisResults.risk_indicators as indicator}
								<div class="bg-red-900/30 border border-red-800 rounded p-4">
									<div class="flex justify-between items-start">
										<div>
											<div class="font-medium text-red-300">{indicator.type}</div>
											<div class="text-sm text-gray-300 mt-1">{indicator.description}</div>
										</div>
										<div class="text-sm font-medium text-red-400">
											Risk: {indicator.severity}
										</div>
									</div>
								</div>
							{/each}
						</div>
					</div>
				</div>
			{/if}
			
		{:else if $activeTab === 'instagram'}
			<InstagramAnalyzer
				profileData={$analysisResults.profiles.instagram}
				posts={$analysisResults.posts.filter(p => p.platform === 'instagram')}
				images={$analysisResults.images.filter(i => i.platform === 'instagram')}
			/>
			
		{:else if $activeTab === 'twitter'}
			<TwitterAnalyzer
				profileData={$analysisResults.profiles.twitter}
				tweets={$analysisResults.posts.filter(p => p.platform === 'twitter')}
			/>
			
		{:else if $activeTab === 'linkedin'}
			<LinkedInAnalyzer
				profileData={$analysisResults.profiles.linkedin}
				connections={$analysisResults.network.nodes.filter(n => n.platform === 'linkedin')}
			/>
			
		{:else if $activeTab === 'facebook'}
			<FacebookAnalyzer
				profileData={$analysisResults.profiles.facebook}
				posts={$analysisResults.posts.filter(p => p.platform === 'facebook')}
			/>
			
		{:else if $activeTab === 'correlation'}
			<CrossPlatformCorrelation
				profiles={$analysisResults.profiles}
				correlationData={$analysisResults.correlation_data}
			/>
			
		{:else if $activeTab === 'network'}
			<SocialNetworkGraph
				network={$analysisResults.network}
				target={analysisConfig.target}
			/>
			
		{:else if $activeTab === 'sentiment'}
			<SentimentAnalysis
				posts={$analysisResults.posts}
				timeline={$analysisResults.timeline}
			/>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Last updated: {new Date().toLocaleTimeString()}
			</div>
			<div class="flex space-x-2">
				<button
					on:click={exportResults}
					class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-md text-sm font-medium transition-colors"
				>
					Export Results
				</button>
				<button
					on:click={loadRecentAnalyses}
					class="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-md text-sm font-medium transition-colors"
				>
					Refresh
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	.social-analyzer-dashboard {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.social-analyzer-dashboard *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.social-analyzer-dashboard *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.social-analyzer-dashboard *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.social-analyzer-dashboard *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>