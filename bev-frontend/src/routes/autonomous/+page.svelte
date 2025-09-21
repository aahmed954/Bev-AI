<!--
Autonomous Intelligence Systems Route - Self-Managing AI Command Center
Integrates: AutonomousIntelligenceCenter.svelte
Backend: src/autonomous/ (9 autonomous systems)
Features: System orchestration, adaptive learning, resource optimization
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import AutonomousIntelligenceCenter from '$lib/components/autonomous/AutonomousIntelligenceCenter.svelte';
	
	let pageTitle = 'Autonomous Intelligence Center';
	let pageDescription = 'Self-managing AI systems with adaptive learning and resource optimization';
	
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
					<h1 class="text-3xl font-bold text-white">🧠 {pageTitle}</h1>
					<p class="mt-2 text-gray-400">{pageDescription}</p>
				</div>
				<div class="flex items-center space-x-4">
					<div class="bg-purple-500/20 text-purple-400 px-3 py-1 rounded-full text-sm font-medium">
						🤖 AI AUTONOMOUS
					</div>
					<div class="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-sm font-medium">
						9 SYSTEMS ACTIVE
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<AutonomousIntelligenceCenter />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>