<!--
Airflow DAG Management Route - ML Pipeline Orchestration
Integrates: AirflowDAGController.svelte
Backend: dags/ (5 DAGs: research_pipeline, health_monitoring, ml_training, etc.)
Features: DAG visualization, execution control, performance analytics
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import AirflowDAGController from '$lib/components/pipeline/AirflowDAGController.svelte';
	
	let pageTitle = 'Airflow DAG Orchestration';
	let pageDescription = 'ML pipeline management with DAG visualization and execution control';
	
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
						AIRFLOW ACTIVE
					</div>
					<div class="bg-blue-500/20 text-blue-400 px-3 py-1 rounded-full text-sm font-medium">
						5 DAGS
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<AirflowDAGController />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>