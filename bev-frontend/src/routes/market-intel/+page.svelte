<!--
Alternative Market Intelligence Route - Advanced Market Analysis
Integrates: AlternativeMarketAnalysis.svelte
Backend: src/alternative_market/ (5 market analysis systems)
Features: Darknet crawling, crypto analysis, reputation scoring, economic processing
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import AlternativeMarketAnalysis from '$lib/components/market/AlternativeMarketAnalysis.svelte';
	
	let pageTitle = 'Alternative Market Intelligence';
	let pageDescription = 'Advanced market analysis with darknet crawling, crypto tracking, and economic processing';
	
	onMount(() => {
		document.title = `${pageTitle} | BEV OSINT Framework`;
	});
</script>

<svelte:head>
	<title>{pageTitle} | BEV OSINT Framework</title>
	<meta name="description" content={pageDescription} />
</svelte:head>

<!-- Page Header -->
<div class="bg-gray-900 border-b border-gray-800">
	<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
		<div class="py-6">
			<div class="flex items-center justify-between">
				<div>
					<h1 class="text-3xl font-bold text-white">📊 {pageTitle}</h1>
					<p class="mt-2 text-gray-400">{pageDescription}</p>
				</div>
				<div class="flex items-center space-x-4">
					<div class="bg-orange-500/20 text-orange-400 px-3 py-1 rounded-full text-sm font-medium">
						🕷️ MARKET CRAWLER
					</div>
					<div class="bg-red-500/20 text-red-400 px-3 py-1 rounded-full text-sm font-medium">
						5 ANALYZERS ACTIVE
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<AlternativeMarketAnalysis />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>