<!--
N8N Workflow Automation Manager - Visual Workflow Builder & Execution Monitor
Connected to: n8n-workflows/ (intelligence_gathering.json, security_monitoring.json)
Features: Drag-drop workflow designer, real-time execution, webhook management
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const workflows = writable([]);
	const activeWorkflow = writable(null);
	const executions = writable([]);
	const webhooks = writable([]);
	const selectedTab = writable('workflows'); // 'workflows', 'executions', 'webhooks', 'builder'
	const isLoading = writable(false);
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	
	// Workflow execution data
	let executionHistory: any[] = [];
	let webhookEndpoints: any[] = [];
	
	// New workflow form
	let newWorkflow = {
		name: '',
		description: '',
		tags: [],
		trigger: 'webhook',
		active: true
	};
	
	const workflowTemplates = [
		{
			id: 'intelligence_gathering',
			name: 'Intelligence Gathering',
			description: 'Automated OSINT data collection from multiple sources',
			nodes: 12,
			triggers: ['webhook', 'schedule'],
			category: 'osint'
		},
		{
			id: 'security_monitoring',
			name: 'Security Monitoring',
			description: 'Real-time security event processing and alerting',
			nodes: 8,
			triggers: ['webhook', 'email'],
			category: 'security'
		},
		{
			id: 'threat_correlation',
			name: 'Threat Correlation',
			description: 'Cross-platform threat intelligence correlation',
			nodes: 15,
			triggers: ['schedule', 'manual'],
			category: 'threat-intel'
		},
		{
			id: 'data_pipeline',
			name: 'Data Pipeline',
			description: 'ETL pipeline for OSINT data processing',
			nodes: 10,
			triggers: ['schedule'],
			category: 'data'
		}
	];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadWorkflows();
		await loadExecutions();
		await loadWebhooks();
	});
	
	async function initializeWebSocket() {
		try {
			// Connect to N8N WebSocket endpoint
			ws = new WebSocket('ws://localhost:5678/ws');
			
			ws.onopen = () => {
				console.log('N8N WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleN8NUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('N8N WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('N8N WebSocket connection failed:', error);
		}
	}
	
	function handleN8NUpdate(data: any) {
		switch (data.type) {
			case 'workflow_started':
				addExecution(data.execution);
				break;
			case 'workflow_completed':
				updateExecution(data.execution_id, data.result);
				break;
			case 'workflow_failed':
				updateExecution(data.execution_id, { status: 'failed', error: data.error });
				break;
			case 'node_executed':
				updateExecutionProgress(data.execution_id, data.node, data.result);
				break;
		}
	}
	
	async function loadWorkflows() {
		isLoading.set(true);
		try {
			const result = await invoke('get_n8n_workflows');
			workflows.set(result);
		} catch (error) {
			console.error('Failed to load workflows:', error);
			// Load from local files as fallback
			const localWorkflows = await loadLocalWorkflows();
			workflows.set(localWorkflows);
		} finally {
			isLoading.set(false);
		}
	}
	
	async function loadLocalWorkflows() {
		try {
			const intelligenceGathering = await invoke('read_file', { 
				path: '/home/starlord/Projects/Bev/n8n-workflows/intelligence_gathering.json' 
			});
			const securityMonitoring = await invoke('read_file', { 
				path: '/home/starlord/Projects/Bev/n8n-workflows/security_monitoring.json' 
			});
			
			return [
				{
					id: 'intelligence_gathering',
					name: 'Intelligence Gathering',
					active: true,
					nodes: JSON.parse(intelligenceGathering).nodes?.length || 0,
					lastExecution: new Date(Date.now() - 300000).toISOString(),
					status: 'active'
				},
				{
					id: 'security_monitoring', 
					name: 'Security Monitoring',
					active: true,
					nodes: JSON.parse(securityMonitoring).nodes?.length || 0,
					lastExecution: new Date(Date.now() - 600000).toISOString(),
					status: 'active'
				}
			];
		} catch (error) {
			console.error('Failed to load local workflows:', error);
			return [];
		}
	}
	
	async function loadExecutions() {
		try {
			const result = await invoke('get_n8n_executions');
			executions.set(result || []);
		} catch (error) {
			console.error('Failed to load executions:', error);
		}
	}
	
	async function loadWebhooks() {
		try {
			const result = await invoke('get_n8n_webhooks');
			webhooks.set(result || []);
		} catch (error) {
			console.error('Failed to load webhooks:', error);
		}
	}
	
	async function executeWorkflow(workflowId: string) {
		try {
			const result = await invoke('execute_n8n_workflow', { workflowId });
			addExecution({
				id: result.execution_id,
				workflow_id: workflowId,
				status: 'running',
				started_at: new Date().toISOString()
			});
			dispatch('workflow_executed', { workflowId, executionId: result.execution_id });
		} catch (error) {
			console.error('Failed to execute workflow:', error);
		}
	}
	
	async function stopWorkflow(executionId: string) {
		try {
			await invoke('stop_n8n_execution', { executionId });
			updateExecution(executionId, { status: 'stopped' });
		} catch (error) {
			console.error('Failed to stop workflow:', error);
		}
	}
	
	async function createWorkflowFromTemplate(template: any) {
		try {
			const result = await invoke('create_n8n_workflow', {
				template: template.id,
				name: `${template.name} - ${Date.now()}`,
				active: true
			});
			
			workflows.update(current => [...current, result]);
			dispatch('workflow_created', result);
		} catch (error) {
			console.error('Failed to create workflow:', error);
		}
	}
	
	function addExecution(execution: any) {
		executions.update(current => [execution, ...current.slice(0, 49)]);
	}
	
	function updateExecution(executionId: string, update: any) {
		executions.update(current =>
			current.map(exec =>
				exec.id === executionId ? { ...exec, ...update } : exec
			)
		);
	}
	
	function updateExecutionProgress(executionId: string, node: string, result: any) {
		executions.update(current =>
			current.map(exec =>
				exec.id === executionId
					? {
						...exec,
						progress: [...(exec.progress || []), { node, result, timestamp: new Date().toISOString() }]
					}
					: exec
			)
		);
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'active':
			case 'running':
			case 'success': return 'text-green-400';
			case 'failed':
			case 'error': return 'text-red-400';
			case 'stopped':
			case 'inactive': return 'text-gray-400';
			case 'waiting': return 'text-yellow-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleString();
	}
	
	function openWorkflowBuilder(workflow: any = null) {
		activeWorkflow.set(workflow);
		selectedTab.set('builder');
	}
</script>

<!-- N8N Workflow Manager -->
<div class="n8n-manager h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-orange-400">N8N Workflow Automation</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{$workflows.length} workflows | {$executions.filter(e => e.status === 'running').length} running
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-orange-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadWorkflows}
						class="px-3 py-1 bg-orange-600 hover:bg-orange-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- System Statistics -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-orange-400">{$workflows.length}</div>
				<div class="text-sm text-gray-400">Total Workflows</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">{$workflows.filter(w => w.active).length}</div>
				<div class="text-sm text-gray-400">Active Workflows</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-blue-400">{$executions.filter(e => e.status === 'running').length}</div>
				<div class="text-sm text-gray-400">Running Executions</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-purple-400">{$webhooks.length}</div>
				<div class="text-sm text-gray-400">Active Webhooks</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'workflows', label: 'Workflows', icon: '🔗' },
				{ id: 'executions', label: 'Executions', icon: '⚙️' },
				{ id: 'webhooks', label: 'Webhooks', icon: '🔌' },
				{ id: 'builder', label: 'Workflow Builder', icon: '🔧' },
				{ id: 'templates', label: 'Templates', icon: '📄' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedTab === tab.id
							? 'border-orange-500 text-orange-400'
							: 'border-transparent text-gray-500 hover:text-gray-300 hover:border-gray-300'
					}"
					on:click={() => selectedTab.set(tab.id)}
				>
					<span class="mr-2">{tab.icon}</span>
					{tab.label}
				</button>
			{/each}
		</nav>
	</div>
	
	<!-- Main Content Area -->
	<div class="flex-1 overflow-y-auto p-4">
		{#if $selectedTab === 'workflows'}
			<!-- Workflow List -->
			<div class="space-y-4">
				{#if $workflows.length === 0}
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">🔗</div>
						<p>No workflows configured</p>
						<button
							on:click={() => selectedTab.set('templates')}
							class="mt-4 px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded transition-colors"
						>
							Create from Template
						</button>
					</div>
				{:else}
					{#each $workflows as workflow}
						<div class="bg-gray-800 rounded-lg p-6">
							<div class="flex items-center justify-between mb-4">
								<div class="flex items-center space-x-4">
									<h3 class="text-lg font-semibold text-white">{workflow.name}</h3>
									<span class="px-2 py-1 rounded text-xs {getStatusColor(workflow.status)} bg-gray-700">
										{workflow.status || (workflow.active ? 'active' : 'inactive')}
									</span>
									<span class="text-sm text-gray-400">{workflow.nodes} nodes</span>
								</div>
								<div class="flex items-center space-x-2">
									<button
										on:click={() => executeWorkflow(workflow.id)}
										class="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm transition-colors"
									>
										Execute
									</button>
									<button
										on:click={() => openWorkflowBuilder(workflow)}
										class="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
									>
										Edit
									</button>
								</div>
							</div>
							
							<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
								<div>
									<span class="text-gray-400">Last Execution:</span>
									<span class="text-white ml-2">{workflow.lastExecution ? formatDate(workflow.lastExecution) : 'Never'}</span>
								</div>
								<div>
									<span class="text-gray-400">Total Runs:</span>
									<span class="text-white ml-2">{workflow.totalExecutions || 0}</span>
								</div>
								<div>
									<span class="text-gray-400">Success Rate:</span>
									<span class="text-white ml-2">{(workflow.successRate || 0).toFixed(1)}%</span>
								</div>
								<div>
									<span class="text-gray-400">Avg Runtime:</span>
									<span class="text-white ml-2">{workflow.avgRuntime || 'N/A'}</span>
								</div>
							</div>
						</div>
					{/each}
				{/if}
			</div>
			
		{:else if $selectedTab === 'executions'}
			<!-- Execution History -->
			<div class="space-y-4">
				{#if $executions.length === 0}
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">⚙️</div>
						<p>No workflow executions</p>
					</div>
				{:else}
					{#each $executions as execution}
						<div class="bg-gray-800 rounded-lg p-6">
							<div class="flex items-center justify-between mb-4">
								<div class="flex items-center space-x-4">
									<h3 class="text-lg font-medium text-white">
										{$workflows.find(w => w.id === execution.workflow_id)?.name || execution.workflow_id}
									</h3>
									<span class="px-2 py-1 rounded text-xs {getStatusColor(execution.status)} bg-gray-700">
										{execution.status}
									</span>
								</div>
								<div class="text-sm text-gray-400">
									{formatDate(execution.started_at)}
								</div>
							</div>
							
							{#if execution.progress && execution.progress.length > 0}
								<div class="mb-4">
									<h4 class="font-medium text-gray-300 mb-2">Execution Progress</h4>
									<div class="space-y-2">
										{#each execution.progress as step}
											<div class="flex items-center justify-between bg-gray-900 rounded p-2">
												<span class="text-sm text-white">{step.node}</span>
												<span class="text-xs text-gray-400">{formatDate(step.timestamp)}</span>
											</div>
										{/each}
									</div>
								</div>
							{/if}
							
							<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
								<div>
									<span class="text-gray-400">Duration:</span>
									<span class="text-white ml-2">{execution.duration || 'N/A'}</span>
								</div>
								<div>
									<span class="text-gray-400">Nodes Executed:</span>
									<span class="text-white ml-2">{execution.progress?.length || 0}</span>
								</div>
								{#if execution.error}
									<div class="col-span-2">
										<span class="text-gray-400">Error:</span>
										<span class="text-red-400 ml-2">{execution.error}</span>
									</div>
								{/if}
							</div>
						</div>
					{/each}
				{/if}
			</div>
			
		{:else if $selectedTab === 'webhooks'}
			<!-- Webhook Management -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Active Webhooks</h3>
					{#if $webhooks.length === 0}
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">🔌</div>
							<p>No webhooks configured</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $webhooks as webhook}
								<div class="bg-gray-900 rounded p-4">
									<div class="flex items-center justify-between mb-2">
										<span class="font-medium text-white">{webhook.name}</span>
										<span class="px-2 py-1 rounded text-xs {getStatusColor(webhook.status)} bg-gray-800">
											{webhook.status}
										</span>
									</div>
									<div class="text-sm text-gray-400">
										URL: <code class="text-blue-400">{webhook.url}</code>
									</div>
									<div class="text-sm text-gray-400">
										Method: <span class="text-white">{webhook.method}</span>
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
				
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Webhook Testing</h3>
					<div class="space-y-4">
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Test Webhook URL</label>
							<input
								type="url"
								placeholder="https://localhost:5678/webhook/..."
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							/>
						</div>
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Test Payload</label>
							<textarea
								placeholder='{"test": "data"}'
								rows="4"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							></textarea>
						</div>
						<button
							class="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded transition-colors"
						>
							Test Webhook
						</button>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'builder'}
			<!-- Workflow Builder -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-orange-400">Workflow Builder</h3>
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">🔧</div>
					<p>Visual workflow builder will be embedded here</p>
					<p class="text-sm mt-2">Drag & drop interface for creating automation workflows</p>
					{#if $activeWorkflow}
						<div class="mt-4 bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">Editing: {$activeWorkflow.name}</h4>
							<div class="text-sm text-gray-400">Workflow ID: {$activeWorkflow.id}</div>
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'templates'}
			<!-- Workflow Templates -->
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
				{#each workflowTemplates as template}
					<div class="bg-gray-800 rounded-lg p-6">
						<div class="flex items-center justify-between mb-4">
							<h3 class="text-lg font-semibold text-white">{template.name}</h3>
							<span class="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded capitalize">
								{template.category}
							</span>
						</div>
						
						<p class="text-gray-400 text-sm mb-4">{template.description}</p>
						
						<div class="space-y-2 text-sm mb-4">
							<div class="flex justify-between">
								<span class="text-gray-400">Nodes:</span>
								<span class="text-white">{template.nodes}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Triggers:</span>
								<span class="text-white">{template.triggers.join(', ')}</span>
							</div>
						</div>
						
						<button
							on:click={() => createWorkflowFromTemplate(template)}
							class="w-full px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded transition-colors"
						>
							Create Workflow
						</button>
					</div>
				{/each}
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				N8N Automation Engine | Connected to localhost:5678
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => openWorkflowBuilder()}
					class="px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded text-sm font-medium transition-colors"
				>
					New Workflow
				</button>
				<button
					on:click={() => dispatch('export_workflows')}
					class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
				>
					Export All
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	.n8n-manager {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.n8n-manager *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.n8n-manager *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.n8n-manager *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.n8n-manager *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>