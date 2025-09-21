<!--
Chaos Engineering Control Panel Route - Comprehensive Fault Injection
Integrates: ChaosEngineeringController.svelte
Backend: src/testing/chaos_api.py
Features: Controlled fault injection, impact monitoring, resilience testing
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import ChaosEngineeringController from '$lib/components/chaos/ChaosEngineeringController.svelte';
	
	let pageTitle = 'Chaos Engineering Control';
	let pageDescription = 'Controlled fault injection and resilience testing for system hardening';
	
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
					<div class="bg-red-500/20 text-red-400 px-3 py-1 rounded-full text-sm font-medium">
						CHAOS ACTIVE
					</div>
					<div class="bg-yellow-500/20 text-yellow-400 px-3 py-1 rounded-full text-sm font-medium">
						FAULT INJECTION
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<ChaosEngineeringController />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>