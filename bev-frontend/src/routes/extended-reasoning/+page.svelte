<!--
Extended Reasoning Pipeline Route - 100K+ Token Context Processing
Integrates: ExtendedReasoningInterface.svelte
Backend: src/agents/extended_reasoning_service.py
Features: Large document analysis, reasoning chains, token optimization
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import ExtendedReasoningInterface from '$lib/components/ai-ml/ExtendedReasoningInterface.svelte';
	
	let pageTitle = 'Extended Reasoning Pipeline';
	let pageDescription = '100K+ token context processing with reasoning chain visualization and optimization';
	
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
					<h1 class="text-3xl font-bold text-white">{pageTitle}</h1>
					<p class="mt-2 text-gray-400">{pageDescription}</p>
				</div>
				<div class="flex items-center space-x-4">
					<div class="bg-indigo-500/20 text-indigo-400 px-3 py-1 rounded-full text-sm font-medium">
						AI REASONING
					</div>
					<div class="bg-yellow-500/20 text-yellow-400 px-3 py-1 rounded-full text-sm font-medium">
						100K+ TOKENS
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<ExtendedReasoningInterface />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>