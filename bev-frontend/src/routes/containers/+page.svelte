<!--
Container Operations Route - Docker Orchestration Management
Integrates: ContainerDashboard.svelte
Backend: docker-compose.complete.yml (70+ services)
Features: Service management, resource monitoring, log streaming
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import ContainerDashboard from '$lib/components/infrastructure/ContainerDashboard.svelte';
	
	let pageTitle = 'Container Orchestration';
	let pageDescription = 'Manage and monitor 70+ containerized services with real-time health monitoring';
	
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
					<div class="bg-blue-500/20 text-blue-400 px-3 py-1 rounded-full text-sm font-medium">
						DOCKER ACTIVE
					</div>
					<div class="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-sm font-medium">
						70+ SERVICES
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<ContainerDashboard />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>