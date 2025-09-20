<!--
Message Queue Administration Dashboard - Kafka & RabbitMQ Management
Connected to: Kafka cluster (3 brokers) + RabbitMQ cluster management
Features: Topic management, consumer groups, message routing, dead letter queues
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	// Sub-components
	import KafkaAdmin from './KafkaAdmin.svelte';
	import RabbitMQManager from './RabbitMQManager.svelte';
	import MessageFlowVisualizer from './MessageFlowVisualizer.svelte';
	import DeadLetterQueue from './DeadLetterQueue.svelte';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const queueSystems = writable({
		kafka: {
			brokers: [],
			topics: [],
			consumers: [],
			producers: [],
			status: 'unknown'
		},
		rabbitmq: {
			nodes: [],
			queues: [],
			exchanges: [],
			connections: [],
			status: 'unknown'
		}
	});
	
	const activeTab = writable('overview');
	const selectedTopic = writable(null);
	const selectedQueue = writable(null);
	const isLoading = writable(false);
	const refreshInterval = writable(5000);
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	let refreshTimer: any = null;
	
	// Message flow data
	let messageFlowData = {
		kafka: { throughput: 0, messages_per_sec: 0, lag: 0 },
		rabbitmq: { throughput: 0, messages_per_sec: 0, queue_depth: 0 }
	};
	
	let systemMetrics = {
		total_topics: 0,
		total_queues: 0,
		total_consumers: 0,
		total_messages_today: 0,
		dead_letter_count: 0
	};
	
	onMount(async () => {
		await initializeWebSocket();
		await loadQueueSystems();
		startPeriodicRefresh();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/message-queue-admin');
			
			ws.onopen = () => {
				console.log('Message Queue Admin WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleWebSocketMessage(data);
			};
			
			ws.onclose = () => {
				console.log('WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('WebSocket connection failed:', error);
		}
	}
	
	function handleWebSocketMessage(data: any) {
		switch (data.type) {
			case 'kafka_metrics_update':
				updateKafkaMetrics(data.metrics);
				break;
			case 'rabbitmq_metrics_update':
				updateRabbitMQMetrics(data.metrics);
				break;
			case 'message_flow_update':
				updateMessageFlow(data.flow);
				break;
			case 'topic_created':
				addKafkaTopic(data.topic);
				break;
			case 'queue_created':
				addRabbitMQQueue(data.queue);
				break;
			case 'consumer_group_update':
				updateConsumerGroup(data.group);
				break;
			case 'dead_letter_alert':
				handleDeadLetterAlert(data.alert);
				break;
		}
	}
	
	async function loadQueueSystems() {
		isLoading.set(true);
		try {
			const [kafkaData, rabbitMQData] = await Promise.all([
				invoke('get_kafka_cluster_info'),
				invoke('get_rabbitmq_cluster_info')
			]);
			
			queueSystems.update(current => ({
				kafka: { ...current.kafka, ...kafkaData },
				rabbitmq: { ...current.rabbitmq, ...rabbitMQData }
			}));
			
			calculateSystemMetrics();
		} catch (error) {
			console.error('Failed to load queue systems:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function calculateSystemMetrics() {
		const systems = $queueSystems;
		systemMetrics = {
			total_topics: systems.kafka.topics.length,
			total_queues: systems.rabbitmq.queues.length,
			total_consumers: systems.kafka.consumers.length + systems.rabbitmq.connections.length,
			total_messages_today: systems.kafka.topics.reduce((acc, topic) => acc + (topic.messages_today || 0), 0) +
								   systems.rabbitmq.queues.reduce((acc, queue) => acc + (queue.messages_today || 0), 0),
			dead_letter_count: systems.rabbitmq.queues.filter(q => q.name.includes('dlq') || q.name.includes('dead')).length
		};
	}
	
	function updateKafkaMetrics(metrics: any) {
		queueSystems.update(current => ({
			...current,
			kafka: {
				...current.kafka,
				brokers: metrics.brokers || current.kafka.brokers,
				topics: metrics.topics || current.kafka.topics,
				consumers: metrics.consumers || current.kafka.consumers,
				status: metrics.status || current.kafka.status
			}
		}));
		calculateSystemMetrics();
	}
	
	function updateRabbitMQMetrics(metrics: any) {
		queueSystems.update(current => ({
			...current,
			rabbitmq: {
				...current.rabbitmq,
				nodes: metrics.nodes || current.rabbitmq.nodes,
				queues: metrics.queues || current.rabbitmq.queues,
				exchanges: metrics.exchanges || current.rabbitmq.exchanges,
				connections: metrics.connections || current.rabbitmq.connections,
				status: metrics.status || current.rabbitmq.status
			}
		}));
		calculateSystemMetrics();
	}
	
	function updateMessageFlow(flow: any) {
		messageFlowData = {
			kafka: flow.kafka || messageFlowData.kafka,
			rabbitmq: flow.rabbitmq || messageFlowData.rabbitmq
		};
	}
	
	function addKafkaTopic(topic: any) {
		queueSystems.update(current => ({
			...current,
			kafka: {
				...current.kafka,
				topics: [...current.kafka.topics, topic]
			}
		}));
		calculateSystemMetrics();
	}
	
	function addRabbitMQQueue(queue: any) {
		queueSystems.update(current => ({
			...current,
			rabbitmq: {
				...current.rabbitmq,
				queues: [...current.rabbitmq.queues, queue]
			}
		}));
		calculateSystemMetrics();
	}
	
	function updateConsumerGroup(group: any) {
		queueSystems.update(current => ({
			...current,
			kafka: {
				...current.kafka,
				consumers: current.kafka.consumers.map(c =>
					c.group === group.group ? { ...c, ...group } : c
				)
			}
		}));
	}
	
	function handleDeadLetterAlert(alert: any) {
		dispatch('dead_letter_alert', alert);
	}
	
	function startPeriodicRefresh() {
		refreshTimer = setInterval(async () => {
			if (ws && ws.readyState === WebSocket.OPEN) {
				ws.send(JSON.stringify({ type: 'refresh_metrics' }));
			} else {
				await loadQueueSystems();
			}
		}, $refreshInterval);
	}
	
	async function createKafkaTopic(topicConfig: any) {
		try {
			await invoke('create_kafka_topic', { config: topicConfig });
			dispatch('topic_created', topicConfig);
		} catch (error) {
			console.error('Failed to create Kafka topic:', error);
		}
	}
	
	async function createRabbitMQQueue(queueConfig: any) {
		try {
			await invoke('create_rabbitmq_queue', { config: queueConfig });
			dispatch('queue_created', queueConfig);
		} catch (error) {
			console.error('Failed to create RabbitMQ queue:', error);
		}
	}
	
	async function deleteKafkaTopic(topicName: string) {
		try {
			await invoke('delete_kafka_topic', { topicName });
			queueSystems.update(current => ({
				...current,
				kafka: {
					...current.kafka,
					topics: current.kafka.topics.filter(t => t.name !== topicName)
				}
			}));
		} catch (error) {
			console.error('Failed to delete Kafka topic:', error);
		}
	}
	
	async function purgeRabbitMQQueue(queueName: string) {
		try {
			await invoke('purge_rabbitmq_queue', { queueName });
			dispatch('queue_purged', queueName);
		} catch (error) {
			console.error('Failed to purge RabbitMQ queue:', error);
		}
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'healthy':
			case 'running': return 'text-green-400';
			case 'warning': return 'text-yellow-400';
			case 'error':
			case 'down': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function formatBytes(bytes: number): string {
		if (bytes === 0) return '0 B';
		const k = 1024;
		const sizes = ['B', 'KB', 'MB', 'GB'];
		const i = Math.floor(Math.log(bytes) / Math.log(k));
		return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
	}
</script>

<!-- Message Queue Dashboard -->
<div class="message-queue-dashboard h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-green-400">Message Queue Administration</h1>
			<div class="flex items-center space-x-4">
				<div class="flex items-center space-x-2">
					<div class="w-2 h-2 rounded-full {getStatusColor($queueSystems.kafka.status)}"></div>
					<span class="text-sm text-gray-400">Kafka</span>
				</div>
				<div class="flex items-center space-x-2">
					<div class="w-2 h-2 rounded-full {getStatusColor($queueSystems.rabbitmq.status)}"></div>
					<span class="text-sm text-gray-400">RabbitMQ</span>
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-green-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadQueueSystems}
						class="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- System Metrics -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-5 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-blue-400">{systemMetrics.total_topics}</div>
				<div class="text-sm text-gray-400">Kafka Topics</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-purple-400">{systemMetrics.total_queues}</div>
				<div class="text-sm text-gray-400">RabbitMQ Queues</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">{systemMetrics.total_consumers}</div>
				<div class="text-sm text-gray-400">Active Consumers</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-yellow-400">{formatNumber(systemMetrics.total_messages_today)}</div>
				<div class="text-sm text-gray-400">Messages Today</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-red-400">{systemMetrics.dead_letter_count}</div>
				<div class="text-sm text-gray-400">Dead Letter Queues</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'System Overview', icon: '📊' },
				{ id: 'kafka', label: 'Kafka Admin', icon: '🔗' },
				{ id: 'rabbitmq', label: 'RabbitMQ Manager', icon: '🐰' },
				{ id: 'flow', label: 'Message Flow', icon: '🌊' },
				{ id: 'deadletter', label: 'Dead Letter Queues', icon: '💀' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$activeTab === tab.id
							? 'border-green-500 text-green-400'
							: 'border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-300'
					}"
					on:click={() => activeTab.set(tab.id)}
				>
					<span class="mr-2">{tab.icon}</span>
					{tab.label}
				</button>
			{/each}
		</nav>
	</div>
	
	<!-- Main Content Area -->
	<div class="flex-1 overflow-y-auto p-4">
		{#if $activeTab === 'overview'}
			<!-- System Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Kafka Cluster Status -->
				<div class="bg-gray-800 rounded-lg p-6">
					<div class="flex items-center justify-between mb-4">
						<h3 class="text-lg font-semibold text-blue-400">Kafka Cluster</h3>
						<span class="px-2 py-1 rounded text-xs {getStatusColor($queueSystems.kafka.status)} bg-gray-700">
							{$queueSystems.kafka.status}
						</span>
					</div>
					
					<div class="space-y-4">
						<div class="grid grid-cols-3 gap-4 text-center">
							<div>
								<div class="text-xl font-bold text-white">{$queueSystems.kafka.brokers.length}</div>
								<div class="text-xs text-gray-400">Brokers</div>
							</div>
							<div>
								<div class="text-xl font-bold text-white">{$queueSystems.kafka.topics.length}</div>
								<div class="text-xs text-gray-400">Topics</div>
							</div>
							<div>
								<div class="text-xl font-bold text-white">{$queueSystems.kafka.consumers.length}</div>
								<div class="text-xs text-gray-400">Consumer Groups</div>
							</div>
						</div>
						
						<div class="space-y-2">
							<h4 class="font-medium text-gray-300">Recent Topics</h4>
							{#each $queueSystems.kafka.topics.slice(0, 5) as topic}
								<div class="flex justify-between items-center bg-gray-900 rounded p-2">
									<span class="text-white text-sm">{topic.name}</span>
									<div class="flex items-center space-x-2 text-xs">
										<span class="text-gray-400">{topic.partitions || 1} partitions</span>
										<span class="text-blue-400">{formatNumber(topic.messages || 0)} msgs</span>
									</div>
								</div>
							{/each}
						</div>
					</div>
				</div>
				
				<!-- RabbitMQ Cluster Status -->
				<div class="bg-gray-800 rounded-lg p-6">
					<div class="flex items-center justify-between mb-4">
						<h3 class="text-lg font-semibold text-purple-400">RabbitMQ Cluster</h3>
						<span class="px-2 py-1 rounded text-xs {getStatusColor($queueSystems.rabbitmq.status)} bg-gray-700">
							{$queueSystems.rabbitmq.status}
						</span>
					</div>
					
					<div class="space-y-4">
						<div class="grid grid-cols-3 gap-4 text-center">
							<div>
								<div class="text-xl font-bold text-white">{$queueSystems.rabbitmq.nodes.length}</div>
								<div class="text-xs text-gray-400">Nodes</div>
							</div>
							<div>
								<div class="text-xl font-bold text-white">{$queueSystems.rabbitmq.queues.length}</div>
								<div class="text-xs text-gray-400">Queues</div>
							</div>
							<div>
								<div class="text-xl font-bold text-white">{$queueSystems.rabbitmq.exchanges.length}</div>
								<div class="text-xs text-gray-400">Exchanges</div>
							</div>
						</div>
						
						<div class="space-y-2">
							<h4 class="font-medium text-gray-300">Active Queues</h4>
							{#each $queueSystems.rabbitmq.queues.slice(0, 5) as queue}
								<div class="flex justify-between items-center bg-gray-900 rounded p-2">
									<span class="text-white text-sm">{queue.name}</span>
									<div class="flex items-center space-x-2 text-xs">
										<span class="text-gray-400">{queue.consumers || 0} consumers</span>
										<span class="text-purple-400">{formatNumber(queue.messages || 0)} msgs</span>
									</div>
								</div>
							{/each}
						</div>
					</div>
				</div>
				
				<!-- Throughput Metrics -->
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Real-time Throughput</h3>
					<div class="grid grid-cols-2 gap-6">
						<div>
							<h4 class="font-medium text-blue-400 mb-3">Kafka Metrics</h4>
							<div class="space-y-2">
								<div class="flex justify-between">
									<span class="text-gray-400">Messages/sec:</span>
									<span class="text-white">{formatNumber(messageFlowData.kafka.messages_per_sec)}</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Throughput:</span>
									<span class="text-white">{formatBytes(messageFlowData.kafka.throughput)}/s</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Consumer Lag:</span>
									<span class="text-white">{formatNumber(messageFlowData.kafka.lag)}</span>
								</div>
							</div>
						</div>
						
						<div>
							<h4 class="font-medium text-purple-400 mb-3">RabbitMQ Metrics</h4>
							<div class="space-y-2">
								<div class="flex justify-between">
									<span class="text-gray-400">Messages/sec:</span>
									<span class="text-white">{formatNumber(messageFlowData.rabbitmq.messages_per_sec)}</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Throughput:</span>
									<span class="text-white">{formatBytes(messageFlowData.rabbitmq.throughput)}/s</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Queue Depth:</span>
									<span class="text-white">{formatNumber(messageFlowData.rabbitmq.queue_depth)}</span>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
			
		{:else if $activeTab === 'kafka'}
			<KafkaAdmin
				brokers={$queueSystems.kafka.brokers}
				topics={$queueSystems.kafka.topics}
				consumers={$queueSystems.kafka.consumers}
				on:create_topic={(e) => createKafkaTopic(e.detail)}
				on:delete_topic={(e) => deleteKafkaTopic(e.detail)}
			/>
			
		{:else if $activeTab === 'rabbitmq'}
			<RabbitMQManager
				nodes={$queueSystems.rabbitmq.nodes}
				queues={$queueSystems.rabbitmq.queues}
				exchanges={$queueSystems.rabbitmq.exchanges}
				connections={$queueSystems.rabbitmq.connections}
				on:create_queue={(e) => createRabbitMQQueue(e.detail)}
				on:purge_queue={(e) => purgeRabbitMQQueue(e.detail)}
			/>
			
		{:else if $activeTab === 'flow'}
			<MessageFlowVisualizer
				kafkaFlow={messageFlowData.kafka}
				rabbitmqFlow={messageFlowData.rabbitmq}
			/>
			
		{:else if $activeTab === 'deadletter'}
			<DeadLetterQueue
				deadLetterQueues={$queueSystems.rabbitmq.queues.filter(q => q.name.includes('dlq') || q.name.includes('dead'))}
			/>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Last updated: {new Date().toLocaleTimeString()}
			</div>
			<div class="flex space-x-2">
				<select
					bind:value={$refreshInterval}
					class="bg-gray-800 border border-gray-700 rounded px-3 py-1 text-white text-sm"
				>
					<option value={1000}>1s refresh</option>
					<option value={5000}>5s refresh</option>
					<option value={10000}>10s refresh</option>
					<option value={30000}>30s refresh</option>
				</select>
				<button
					on:click={() => dispatch('export_metrics')}
					class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium transition-colors"
				>
					Export Metrics
				</button>
				<button
					on:click={() => dispatch('emergency_shutdown')}
					class="px-4 py-2 bg-red-600 hover:bg-red-700 rounded text-sm font-medium transition-colors"
				>
					Emergency Shutdown
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	.message-queue-dashboard {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.message-queue-dashboard *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.message-queue-dashboard *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.message-queue-dashboard *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.message-queue-dashboard *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>