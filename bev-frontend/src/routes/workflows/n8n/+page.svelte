<!--
N8N Workflow Automation Route - Visual Workflow Builder & Automation
Integrates: N8NWorkflowManager.svelte
Backend: n8n-workflows/ (intelligence_gathering.json, security_monitoring.json)
Features: Drag-drop workflow designer, execution monitoring, webhook management
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import N8NWorkflowManager from '$lib/components/workflow/N8NWorkflowManager.svelte';
	
	let pageTitle = 'N8N Workflow Automation';
	let pageDescription = 'Visual workflow builder for OSINT automation and intelligence gathering pipelines';
	
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
					<div class="bg-orange-500/20 text-orange-400 px-3 py-1 rounded-full text-sm font-medium">
						N8N AUTOMATION
					</div>
					<div class="bg-green-500/20 text-green-400 px-3 py-1 rounded-full text-sm font-medium">
						WORKFLOW ENGINE
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<N8NWorkflowManager />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>