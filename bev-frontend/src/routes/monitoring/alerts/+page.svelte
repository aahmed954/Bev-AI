<!--
Alert Management Center Route - Prometheus Alerting System
Integrates: AlertCenter.svelte
Backend: src/monitoring/alert_system.py
Features: Alert rule configuration, notification management, correlation analysis
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import AlertCenter from '$lib/components/monitoring/AlertCenter.svelte';
	
	let pageTitle = 'Alert Management Center';
	let pageDescription = 'Prometheus alerting with rule configuration and notification management';
	
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
					<div class="bg-amber-500/20 text-amber-400 px-3 py-1 rounded-full text-sm font-medium">
						PROMETHEUS
					</div>
					<div class="bg-red-500/20 text-red-400 px-3 py-1 rounded-full text-sm font-medium">
						ALERTMANAGER
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<AlertCenter />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>