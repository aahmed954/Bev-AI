<!--
Social Media OSINT Route - Multi-Platform Intelligence Gathering
Integrates: SocialMediaDashboard.svelte
Backend: intelowl/custom_analyzers/social_analyzer.py
Features: Instagram, Twitter, LinkedIn, Facebook analysis with correlation
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import SocialMediaDashboard from '$lib/components/social-analyzer/SocialMediaDashboard.svelte';
	
	let pageTitle = 'Social Media Intelligence';
	let pageDescription = 'Multi-platform OSINT analysis with identity correlation and behavioral pattern detection';
	
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
					<div class="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-sm font-medium">
						OSINT ACTIVE
					</div>
					<div class="bg-blue-500/20 text-blue-400 px-3 py-1 rounded-full text-sm font-medium">
						4 PLATFORMS
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<SocialMediaDashboard />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>