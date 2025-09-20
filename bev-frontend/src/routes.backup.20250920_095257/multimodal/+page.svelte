<!--
Advanced Multimodal Processing Route - Multi-Format Content Analysis
Integrates: MultimodalProcessingCenter.svelte
Backend: src/advanced/multimodal_processor.py
Features: Image/video/audio analysis, content understanding, cross-modal correlation
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import MultimodalProcessingCenter from '$lib/components/advanced/MultimodalProcessingCenter.svelte';
	
	let pageTitle = 'Multimodal Processing Center';
	let pageDescription = 'Advanced multi-format content analysis with cross-modal correlation and AI understanding';
	
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
					<h1 class="text-3xl font-bold text-white">🎭 {pageTitle}</h1>
					<p class="mt-2 text-gray-400">{pageDescription}</p>
				</div>
				<div class="flex items-center space-x-4">
					<div class="bg-pink-500/20 text-pink-400 px-3 py-1 rounded-full text-sm font-medium">
						🎭 MULTIMODAL AI
					</div>
					<div class="bg-purple-500/20 text-purple-400 px-3 py-1 rounded-full text-sm font-medium">
						4 MODALITIES
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<MultimodalProcessingCenter />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>