<!--
Alternative Market Analysis - Advanced Market Intelligence & Economics
Connected to: src/alternative_market/ (5 market analysis systems)
Features: Darknet market crawling, crypto analysis, reputation scoring, economic processing
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const marketState = writable({
		dm_crawler: {
			status: 'crawling',
			markets_monitored: 0,
			vendors_tracked: 0,
			products_indexed: 0,
			last_crawl: null
		},
		crypto_analyzer: {
			status: 'analyzing',
			addresses_tracked: 0,
			transactions_analyzed: 0,
			suspicious_activity: 0,
			correlation_score: 0
		},
		reputation_analyzer: {
			status: 'scoring',
			entities_scored: 0,
			reputation_updates: 0,
			risk_profiles: 0,
			confidence_score: 0
		},
		economics_processor: {
			status: 'processing',
			market_analyses: 0,
			price_correlations: 0,
			economic_indicators: 0,
			trend_accuracy: 0
		}
	});
	
	const selectedTab = writable('overview'); // 'overview', 'darknet', 'crypto', 'reputation', 'economics'
	const selectedAnalysis = writable(null);
	const isLoading = writable(false);
	
	// WebSocket for real-time market updates
	let ws: WebSocket | null = null;
	
	// Market analysis tools
	const marketTools = [
		{
			id: 'dm_crawler',
			name: 'Darknet Market Crawler',
			icon: '🕷️',
			description: 'Advanced darknet marketplace intelligence gathering',
			port: 8500,
			features: ['Multi-market crawling', 'Vendor tracking', 'Product cataloging', 'Price monitoring']
		},
		{
			id: 'crypto_analyzer',
			name: 'Advanced Crypto Analyzer',
			icon: '🔗',
			description: 'Comprehensive cryptocurrency transaction analysis',
			port: 8501,
			features: ['Address clustering', 'Transaction flow', 'Mixing detection', 'Risk scoring']
		},
		{
			id: 'reputation_analyzer',
			name: 'Reputation Analyzer',
			icon: '⭐',
			description: 'Entity reputation scoring and risk assessment',
			port: 8502,
			features: ['Reputation scoring', 'Risk profiling', 'Historical analysis', 'Trust metrics']
		},
		{
			id: 'economics_processor',
			name: 'Economics Processor',
			icon: '📈',
			description: 'Market economic analysis and trend prediction',
			port: 8503,
			features: ['Market analysis', 'Price prediction', 'Economic indicators', 'Trend analysis']
		}
	];
	
	// Market data
	let marketData = {
		active_markets: [],
		crypto_transactions: [],
		reputation_scores: [],
		economic_trends: []
	};
	
	// Analysis forms
	let newAnalysis = {
		type: 'darknet_crawl',
		target: '',
		depth: 2,
		duration: '1h',
		privacy_mode: true,
		correlation: true
	};
	
	const analysisTypes = [
		'darknet_crawl', 'crypto_tracking', 'reputation_analysis', 
		'economic_analysis', 'market_correlation', 'trend_prediction'
	];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadMarketData();
		startMarketMonitoring();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/market-analysis');
			
			ws.onopen = () => {
				console.log('Market Analysis WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleMarketUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Market WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Market WebSocket connection failed:', error);
		}
	}
	
	function handleMarketUpdate(data: any) {
		switch (data.type) {
			case 'market_crawled':
				updateMarketData(data.market_data);
				break;
			case 'crypto_analyzed':
				updateCryptoData(data.crypto_data);
				break;
			case 'reputation_updated':
				updateReputationData(data.reputation_data);
				break;
			case 'economic_analysis':
				updateEconomicData(data.economic_data);
				break;
			case 'correlation_found':
				addCorrelation(data.correlation);
				break;
		}
	}
	
	async function loadMarketData() {
		isLoading.set(true);
		try {
			const [markets, crypto, reputation, economics, metrics] = await Promise.all([
				invoke('get_active_markets'),
				invoke('get_crypto_transactions'),
				invoke('get_reputation_scores'),
				invoke('get_economic_trends'),
				invoke('get_market_metrics')
			]);
			
			marketData = {
				active_markets: markets || [],
				crypto_transactions: crypto || [],
				reputation_scores: reputation || [],
				economic_trends: economics || []
			};
			
			// Update market state with metrics
			if (metrics) {
				marketState.update(current => ({
					...current,
					dm_crawler: { ...current.dm_crawler, ...metrics.dm_crawler },
					crypto_analyzer: { ...current.crypto_analyzer, ...metrics.crypto_analyzer },
					reputation_analyzer: { ...current.reputation_analyzer, ...metrics.reputation_analyzer },
					economics_processor: { ...current.economics_processor, ...metrics.economics_processor }
				}));
			}
		} catch (error) {
			console.error('Failed to load market data:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function startMarketMonitoring() {
		setInterval(async () => {
			try {
				for (const tool of marketTools) {
					const response = await fetch(`http://localhost:${tool.port}/status`);
					const status = await response.json();
					updateToolStatus(tool.id, status);
				}
			} catch (error) {
				console.error('Market monitoring failed:', error);
			}
		}, 15000);
	}
	
	async function startMarketAnalysis() {
		try {
			const result = await invoke('start_market_analysis', { analysis: newAnalysis });
			
			dispatch('analysis_started', result);
			
			// Reset form
			newAnalysis = {
				type: 'darknet_crawl',
				target: '',
				depth: 2,
				duration: '1h',
				privacy_mode: true,
				correlation: true
			};
		} catch (error) {
			console.error('Failed to start market analysis:', error);
		}
	}
	
	async function runDarknetCrawl(target: string) {
		try {
			const result = await invoke('crawl_darknet_market', { target });
			updateMarketData(result);
		} catch (error) {
			console.error('Failed to run darknet crawl:', error);
		}
	}
	
	async function analyzeCryptoFlow(addresses: string[]) {
		try {
			const result = await invoke('analyze_crypto_flow', { addresses });
			updateCryptoData(result);
		} catch (error) {
			console.error('Failed to analyze crypto flow:', error);
		}
	}
	
	async function scoreReputation(entity: string) {
		try {
			const result = await invoke('score_reputation', { entity });
			updateReputationData(result);
		} catch (error) {
			console.error('Failed to score reputation:', error);
		}
	}
	
	function updateMarketData(data: any) {
		marketData.active_markets = [...marketData.active_markets, data];
		marketState.update(current => ({
			...current,
			dm_crawler: {
				...current.dm_crawler,
				markets_monitored: current.dm_crawler.markets_monitored + 1,
				vendors_tracked: current.dm_crawler.vendors_tracked + (data.vendors?.length || 0),
				products_indexed: current.dm_crawler.products_indexed + (data.products?.length || 0)
			}
		}));
	}
	
	function updateCryptoData(data: any) {
		marketData.crypto_transactions = [...marketData.crypto_transactions, data];
		marketState.update(current => ({
			...current,
			crypto_analyzer: {
				...current.crypto_analyzer,
				addresses_tracked: current.crypto_analyzer.addresses_tracked + 1,
				transactions_analyzed: current.crypto_analyzer.transactions_analyzed + (data.transactions?.length || 0),
				suspicious_activity: current.crypto_analyzer.suspicious_activity + (data.suspicious_count || 0)
			}
		}));
	}
	
	function updateReputationData(data: any) {
		marketData.reputation_scores = [...marketData.reputation_scores, data];
		marketState.update(current => ({
			...current,
			reputation_analyzer: {
				...current.reputation_analyzer,
				entities_scored: current.reputation_analyzer.entities_scored + 1,
				reputation_updates: current.reputation_analyzer.reputation_updates + 1
			}
		}));
	}
	
	function updateEconomicData(data: any) {
		marketData.economic_trends = [...marketData.economic_trends, data];
		marketState.update(current => ({
			...current,
			economics_processor: {
				...current.economics_processor,
				market_analyses: current.economics_processor.market_analyses + 1,
				economic_indicators: current.economics_processor.economic_indicators + (data.indicators?.length || 0)
			}
		}));
	}
	
	function addCorrelation(correlation: any) {
		dispatch('correlation_found', correlation);
	}
	
	function updateToolStatus(toolId: string, status: any) {
		marketState.update(current => ({
			...current,
			[toolId]: { ...current[toolId], ...status }
		}));
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'crawling':
			case 'analyzing':
			case 'scoring':
			case 'processing': return 'text-green-400';
			case 'warning': return 'text-yellow-400';
			case 'error':
			case 'failed': return 'text-red-400';
			case 'idle':
			case 'paused': return 'text-gray-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleString();
	}
	
	function openAnalysisModal(analysis: any) {
		selectedAnalysis.set(analysis);
	}
</script>

<!-- Alternative Market Analysis -->
<div class="market-analysis h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-orange-400">📊 Alternative Market Analysis</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{marketData.active_markets.length} markets | {marketData.crypto_transactions.length} crypto flows
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-orange-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadMarketData}
						class="px-3 py-1 bg-orange-600 hover:bg-orange-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Market Tools Status -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
			{#each marketTools as tool}
				{@const toolData = $marketState[tool.id] || {}}
				<div 
					class="bg-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-700 transition-colors"
					on:click={() => selectedTab.set(tool.id)}
				>
					<div class="flex items-center justify-between mb-3">
						<div class="flex items-center space-x-2">
							<span class="text-xl">{tool.icon}</span>
							<span class="font-medium text-white text-sm">{tool.name}</span>
						</div>
						<div class="flex items-center space-x-1">
							<div class="w-2 h-2 rounded-full {getStatusColor(toolData.status)}"></div>
							<span class="text-xs {getStatusColor(toolData.status)}">{toolData.status || 'unknown'}</span>
						</div>
					</div>
					
					<p class="text-gray-400 text-xs mb-3">{tool.description}</p>
					
					{#if tool.id === 'dm_crawler'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Markets:</span>
								<span class="text-white">{toolData.markets_monitored || 0}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Vendors:</span>
								<span class="text-blue-400">{formatNumber(toolData.vendors_tracked || 0)}</span>
							</div>
						</div>
					{:else if tool.id === 'crypto_analyzer'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Addresses:</span>
								<span class="text-white">{formatNumber(toolData.addresses_tracked || 0)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Suspicious:</span>
								<span class="text-red-400">{toolData.suspicious_activity || 0}</span>
							</div>
						</div>
					{:else if tool.id === 'reputation_analyzer'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Entities:</span>
								<span class="text-white">{formatNumber(toolData.entities_scored || 0)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Updates:</span>
								<span class="text-green-400">{toolData.reputation_updates || 0}</span>
							</div>
						</div>
					{:else if tool.id === 'economics_processor'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Analyses:</span>
								<span class="text-white">{toolData.market_analyses || 0}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Accuracy:</span>
								<span class="text-purple-400">{(toolData.trend_accuracy || 0).toFixed(1)}%</span>
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
				{ id: 'overview', label: 'Market Overview', icon: '📊' },
				{ id: 'dm_crawler', label: 'Darknet Crawler', icon: '🕷️' },
				{ id: 'crypto_analyzer', label: 'Crypto Analyzer', icon: '🔗' },
				{ id: 'reputation_analyzer', label: 'Reputation Scoring', icon: '⭐' },
				{ id: 'economics_processor', label: 'Economics', icon: '📈' },
				{ id: 'correlations', label: 'Correlations', icon: '🔗' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedTab === tab.id
							? 'border-orange-500 text-orange-400'
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
			<!-- Market Analysis Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Market Intelligence Summary -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-orange-400">Market Intelligence Summary</h3>
					<div class="space-y-4">
						<div class="grid grid-cols-2 gap-4 text-center">
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-blue-400">{marketData.active_markets.length}</div>
								<div class="text-xs text-gray-400">Active Markets</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-green-400">{formatNumber($marketState.dm_crawler.vendors_tracked)}</div>
								<div class="text-xs text-gray-400">Vendors Tracked</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-yellow-400">{formatNumber($marketState.crypto_analyzer.addresses_tracked)}</div>
								<div class="text-xs text-gray-400">Crypto Addresses</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-purple-400">{formatNumber($marketState.reputation_analyzer.entities_scored)}</div>
								<div class="text-xs text-gray-400">Reputation Scores</div>
							</div>
						</div>
						
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">Analysis Performance</h4>
							<div class="space-y-2 text-sm">
								<div class="flex justify-between">
									<span class="text-gray-400">Products Indexed:</span>
									<span class="text-white">{formatNumber($marketState.dm_crawler.products_indexed)}</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Suspicious Activity:</span>
									<span class="text-red-400">{$marketState.crypto_analyzer.suspicious_activity}</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Economic Indicators:</span>
									<span class="text-white">{$marketState.economics_processor.economic_indicators}</span>
								</div>
							</div>
						</div>
					</div>
				</div>
				
				<!-- Quick Analysis Tools -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Quick Analysis Tools</h3>
					<div class="space-y-4">
						<!-- Darknet Crawl -->
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">🕷️ Darknet Market Crawl</h4>
							<div class="flex space-x-2">
								<input
									type="text"
									placeholder="Market URL or onion address"
									class="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
								/>
								<button
									on:click={() => runDarknetCrawl('')}
									class="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded transition-colors"
								>
									Crawl
								</button>
							</div>
						</div>
						
						<!-- Crypto Analysis -->
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">🔗 Crypto Flow Analysis</h4>
							<div class="flex space-x-2">
								<input
									type="text"
									placeholder="Bitcoin/Ethereum address"
									class="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
								/>
								<button
									on:click={() => analyzeCryptoFlow([''])}
									class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
								>
									Analyze
								</button>
							</div>
						</div>
						
						<!-- Reputation Scoring -->
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">⭐ Reputation Scoring</h4>
							<div class="flex space-x-2">
								<input
									type="text"
									placeholder="Entity name or identifier"
									class="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-orange-500"
								/>
								<button
									on:click={() => scoreReputation('')}
									class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded transition-colors"
								>
									Score
								</button>
							</div>
						</div>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'dm_crawler'}
			<!-- Darknet Market Crawler -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-orange-400">🕷️ Darknet Market Crawler</h3>
				<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-3">Crawler Status</h4>
						<div class="space-y-2 text-sm">
							<div class="flex justify-between">
								<span class="text-gray-400">Status:</span>
								<span class="{getStatusColor($marketState.dm_crawler.status)}">{$marketState.dm_crawler.status}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Markets Monitored:</span>
								<span class="text-white">{$marketState.dm_crawler.markets_monitored}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Vendors Tracked:</span>
								<span class="text-white">{formatNumber($marketState.dm_crawler.vendors_tracked)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Products Indexed:</span>
								<span class="text-white">{formatNumber($marketState.dm_crawler.products_indexed)}</span>
							</div>
						</div>
					</div>
					
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-3">Recent Markets</h4>
						<div class="space-y-2">
							{#each marketData.active_markets.slice(0, 5) as market}
								<div class="bg-gray-800 rounded p-2">
									<div class="font-medium text-white text-sm">{market.name}</div>
									<div class="text-xs text-gray-400">{market.vendors || 0} vendors</div>
								</div>
							{/each}
						</div>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'crypto_analyzer'}
			<!-- Advanced Crypto Analyzer -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-blue-400">🔗 Advanced Crypto Analyzer</h3>
				<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-3">Analysis Status</h4>
						<div class="space-y-2 text-sm">
							<div class="flex justify-between">
								<span class="text-gray-400">Status:</span>
								<span class="{getStatusColor($marketState.crypto_analyzer.status)}">{$marketState.crypto_analyzer.status}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Addresses Tracked:</span>
								<span class="text-white">{formatNumber($marketState.crypto_analyzer.addresses_tracked)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Transactions:</span>
								<span class="text-white">{formatNumber($marketState.crypto_analyzer.transactions_analyzed)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Suspicious Activity:</span>
								<span class="text-red-400">{$marketState.crypto_analyzer.suspicious_activity}</span>
							</div>
						</div>
					</div>
					
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-3">Flow Visualization</h4>
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">🔗</div>
							<p>Transaction flow graph</p>
							<p class="text-sm mt-1">Real-time crypto transaction visualization</p>
						</div>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'reputation_analyzer'}
			<!-- Reputation Analyzer -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-purple-400">⭐ Reputation Analysis System</h3>
				<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-3">Scoring Status</h4>
						<div class="space-y-2 text-sm">
							<div class="flex justify-between">
								<span class="text-gray-400">Status:</span>
								<span class="{getStatusColor($marketState.reputation_analyzer.status)}">{$marketState.reputation_analyzer.status}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Entities Scored:</span>
								<span class="text-white">{formatNumber($marketState.reputation_analyzer.entities_scored)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Risk Profiles:</span>
								<span class="text-white">{$marketState.reputation_analyzer.risk_profiles}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Confidence:</span>
								<span class="text-green-400">{($marketState.reputation_analyzer.confidence_score || 0).toFixed(1)}%</span>
							</div>
						</div>
					</div>
					
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-3">Recent Scores</h4>
						<div class="space-y-2">
							{#each marketData.reputation_scores.slice(0, 5) as score}
								<div class="bg-gray-800 rounded p-2">
									<div class="flex items-center justify-between">
										<span class="font-medium text-white text-sm">{score.entity}</span>
										<span class="text-xs {score.risk_level === 'high' ? 'text-red-400' : score.risk_level === 'medium' ? 'text-yellow-400' : 'text-green-400'}">
											{score.reputation_score}/100
										</span>
									</div>
								</div>
							{/each}
						</div>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'economics_processor'}
			<!-- Economics Processor -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-green-400">📈 Economic Analysis Processor</h3>
				<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-3">Processing Status</h4>
						<div class="space-y-2 text-sm">
							<div class="flex justify-between">
								<span class="text-gray-400">Status:</span>
								<span class="{getStatusColor($marketState.economics_processor.status)}">{$marketState.economics_processor.status}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Market Analyses:</span>
								<span class="text-white">{$marketState.economics_processor.market_analyses}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Price Correlations:</span>
								<span class="text-white">{$marketState.economics_processor.price_correlations}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Trend Accuracy:</span>
								<span class="text-purple-400">{($marketState.economics_processor.trend_accuracy || 0).toFixed(1)}%</span>
							</div>
						</div>
					</div>
					
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-3">Economic Trends</h4>
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">📈</div>
							<p>Economic trend visualization</p>
							<p class="text-sm mt-1">Market trend analysis and predictions</p>
						</div>
					</div>
				</div>
			</div>
			
		{:else}
			<!-- Default tool interface -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-white">{marketTools.find(t => t.id === $selectedTab)?.name} Interface</h3>
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">{marketTools.find(t => t.id === $selectedTab)?.icon}</div>
					<p>Detailed {marketTools.find(t => t.id === $selectedTab)?.name} interface</p>
					<p class="text-sm mt-2">Advanced market analysis and intelligence gathering</p>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Alternative Market Analysis | 4 analysis tools integrated
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('export_market_intelligence')}
					class="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded text-sm font-medium transition-colors"
				>
					Export Intelligence
				</button>
				<button
					on:click={() => dispatch('start_comprehensive_crawl')}
					class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
				>
					Start Comprehensive Crawl
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	.market-analysis {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Market-themed styling */
	:global(.market-analysis .market-active) {
		animation: pulse-orange 2s infinite;
	}
	
	@keyframes pulse-orange {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.8; }
	}
	
	/* Custom scrollbar */
	:global(.market-analysis *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.market-analysis *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.market-analysis *::-webkit-scrollbar-thumb) {
		background: #f97316;
		border-radius: 3px;
	}
	
	:global(.market-analysis *::-webkit-scrollbar-thumb:hover) {
		background: #fb923c;
	}
</style>