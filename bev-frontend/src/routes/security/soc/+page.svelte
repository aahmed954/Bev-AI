<!--
Security Operations Center Route - Military-Grade Security Command Center
Integrates: SecurityOperationsCenter.svelte
Backend: src/security/ (8 security systems)
Features: Real-time threat monitoring, automated response, multi-system correlation
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import SecurityOperationsCenter from '$lib/components/security/SecurityOperationsCenter.svelte';
	
	let pageTitle = 'Security Operations Center';
	let pageDescription = 'Military-grade security command center with 8 integrated security systems';
	
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
					<h1 class="text-3xl font-bold text-white">🛡️ {pageTitle}</h1>
					<p class="mt-2 text-gray-400">{pageDescription}</p>
				</div>
				<div class="flex items-center space-x-4">
					<div class="bg-red-500/20 text-red-400 px-3 py-1 rounded-full text-sm font-medium">
						🚨 SOC ACTIVE
					</div>
					<div class="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-sm font-medium">
						8 SYSTEMS ONLINE
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<SecurityOperationsCenter />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>