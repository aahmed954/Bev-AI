<!--
Oracle Worker System Route - Specialized Worker Coordination
Integrates: OracleWorkerSystem.svelte
Backend: src/oracle/workers/ (specialized worker coordination system)
Features: Worker management, task distribution, performance monitoring
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import OracleWorkerSystem from '$lib/components/oracle/OracleWorkerSystem.svelte';
	
	let pageTitle = 'Oracle Worker System';
	let pageDescription = 'Specialized worker coordination with task distribution and performance monitoring';
	
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
					<h1 class="text-3xl font-bold text-white">🔮 {pageTitle}</h1>
					<p class="mt-2 text-gray-400">{pageDescription}</p>
				</div>
				<div class="flex items-center space-x-4">
					<div class="bg-indigo-500/20 text-indigo-400 px-3 py-1 rounded-full text-sm font-medium">
						🔮 ORACLE ACTIVE
					</div>
					<div class="bg-purple-500/20 text-purple-400 px-3 py-1 rounded-full text-sm font-medium">
						WORKER SYSTEM
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<OracleWorkerSystem />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>