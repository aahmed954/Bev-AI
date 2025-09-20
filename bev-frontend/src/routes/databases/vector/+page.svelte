<!--
Vector Database Administration Route - Qdrant & Weaviate Management
Integrates: VectorDatabaseAdmin.svelte
Backend: src/infrastructure/vector_db_manager.py
Features: Index management, embedding visualization, similarity search
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import VectorDatabaseAdmin from '$lib/components/databases/VectorDatabaseAdmin.svelte';
	
	let pageTitle = 'Vector Database Administration';
	let pageDescription = 'Qdrant & Weaviate management with similarity search and embedding visualization';
	
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
						QDRANT ACTIVE
					</div>
					<div class="bg-purple-500/20 text-purple-400 px-3 py-1 rounded-full text-sm font-medium">
						WEAVIATE ACTIVE
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<VectorDatabaseAdmin />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>