<!--
Swarm Master Coordination Route - Multi-Agent Orchestration
Integrates: SwarmMasterDashboard.svelte
Backend: src/agents/swarm_master.py (port 8100)
Features: Agent registration, task distribution, coordination modes
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import SwarmMasterDashboard from '$lib/components/ai-ml/SwarmMasterDashboard.svelte';
	
	let pageTitle = 'Swarm Master Coordination';
	let pageDescription = 'Multi-agent AI orchestration with task distribution and coordination management';
	
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
					<div class="bg-purple-500/20 text-purple-400 px-3 py-1 rounded-full text-sm font-medium">
						SWARM ACTIVE
					</div>
					<div class="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-sm font-medium">
						PORT 8100
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<SwarmMasterDashboard />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>