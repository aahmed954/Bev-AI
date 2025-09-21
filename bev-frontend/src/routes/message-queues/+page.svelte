<!--
Message Queue Administration Route - Kafka & RabbitMQ Management
Integrates: MessageQueueDashboard.svelte
Backend: Kafka cluster (3 brokers) + RabbitMQ cluster
Features: Topic management, consumer groups, message flow visualization
-->

<script lang="ts">
	import { onMount } from 'svelte';
	import MessageQueueDashboard from '$lib/components/infrastructure/MessageQueueDashboard.svelte';
	
	let pageTitle = 'Message Queue Administration';
	let pageDescription = 'Kafka & RabbitMQ cluster management with real-time message flow monitoring';
	
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
						KAFKA CLUSTER
					</div>
					<div class="bg-purple-500/20 text-purple-400 px-3 py-1 rounded-full text-sm font-medium">
						RABBITMQ ACTIVE
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<!-- Main Content -->
<div class="min-h-screen bg-gray-900">
	<MessageQueueDashboard />
</div>

<style>
	:global(body) {
		background-color: #111827;
		color: white;
	}
</style>