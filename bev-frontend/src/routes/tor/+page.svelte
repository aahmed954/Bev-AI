<!--
Tor Network Management Route - 3-Node Relay System Administration
Integrates: TorNetworkManager.svelte
Backend: tor/ (3-node Tor relay system with monitoring)
Features: Relay management, circuit visualization, bandwidth monitoring
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import TorNetworkManager from '$lib/components/tor/TorNetworkManager.svelte';
	
	let pageTitle = 'Tor Network Management';
	let pageDescription = '3-node Tor relay system with circuit visualization and bandwidth monitoring';
	
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
					<h1 class="text-3xl font-bold text-white">🧅 {pageTitle}</h1>
					<p class="mt-2 text-gray-400">{pageDescription}</p>
				</div>
				<div class="flex items-center space-x-4">
					<div class="bg-purple-500/20 text-purple-400 px-3 py-1 rounded-full text-sm font-medium">
						🧅 TOR NETWORK
					</div>
					<div class="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-sm font-medium">
						3 RELAYS ACTIVE
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<TorNetworkManager />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>