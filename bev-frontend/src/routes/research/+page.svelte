<!--
Research Enhancement Framework Route - Automated Research Tools
Integrates: ResearchEnhancementFramework.svelte
Backend: src/enhancement/ (4 research systems)
Features: Research automation, watermark analysis, DRM research, metadata scrubbing
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import ResearchEnhancementFramework from '$lib/components/research/ResearchEnhancementFramework.svelte';
	
	let pageTitle = 'Research Enhancement Framework';
	let pageDescription = 'Automated research tools with watermark analysis, DRM research, and privacy protection';
	
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
					<h1 class="text-3xl font-bold text-white">🔬 {pageTitle}</h1>
					<p class="mt-2 text-gray-400">{pageDescription}</p>
				</div>
				<div class="flex items-center space-x-4">
					<div class="bg-teal-500/20 text-teal-400 px-3 py-1 rounded-full text-sm font-medium">
						🔬 RESEARCH ACTIVE
					</div>
					<div class="bg-blue-500/20 text-blue-400 px-3 py-1 rounded-full text-sm font-medium">
						4 TOOLS ONLINE
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<ResearchEnhancementFramework />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>