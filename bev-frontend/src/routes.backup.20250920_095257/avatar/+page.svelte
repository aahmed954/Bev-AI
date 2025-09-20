<!--
Live2D Avatar System Route - Interactive Avatar Controller
Integrates: Live2DAvatarInterface.svelte
Backend: src/live2d/avatar_controller.py (port 8091)
Features: WebGL rendering, emotion control, voice synthesis, gestures
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import Live2DAvatarInterface from '$lib/components/avatar/Live2DAvatarInterface.svelte';
	
	let pageTitle = 'Live2D Avatar System';
	let pageDescription = 'Interactive avatar with emotion control, voice synthesis, and gesture recognition';
	
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
					<div class="bg-cyan-500/20 text-cyan-400 px-3 py-1 rounded-full text-sm font-medium">
						LIVE2D ACTIVE
					</div>
					<div class="bg-pink-500/20 text-pink-400 px-3 py-1 rounded-full text-sm font-medium">
						PORT 8091
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<Live2DAvatarInterface />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>