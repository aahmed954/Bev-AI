<!--
Advanced Alert Management Center - Prometheus Alerting System
Connected to: src/monitoring/alert_system.py
Features: Alert rule configuration, notification channels, correlation analysis
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const alertSystem = writable({
		active_alerts: [],
		alert_rules: [],
		notification_channels: [],
		alert_history: [],
		silenced_alerts: [],
		metrics: {
			total_alerts_today: 0,
			critical_alerts: 0,
			warning_alerts: 0,
			resolved_alerts: 0,
			avg_resolution_time: 0
		}
	});
	
	const selectedTab = writable('overview'); // 'overview', 'active', 'rules', 'channels', 'history'
	const selectedAlert = writable(null);
	const isLoading = writable(false);
	
	// WebSocket for real-time alerts
	let ws: WebSocket | null = null;
	
	// Alert rule creation form
	let newRule = {
		name: '',
		query: '',
		severity: 'warning',
		duration: '5m',
		labels: {},
		annotations: {},
		channels: []
	};
	
	// Notification channel form
	let newChannel = {
		name: '',
		type: 'slack',
		config: {},
		enabled: true
	};
	
	const severityLevels = [
		{ id: 'info', label: 'Info', color: 'text-blue-400' },
		{ id: 'warning', label: 'Warning', color: 'text-yellow-400' },
		{ id: 'critical', label: 'Critical', color: 'text-red-400' },
		{ id: 'emergency', label: 'Emergency', color: 'text-red-600' }
	];
	
	const channelTypes = [
		{ id: 'slack', label: 'Slack', icon: '💬' },
		{ id: 'discord', label: 'Discord', icon: '🎮' },
		{ id: 'email', label: 'Email', icon: '📧' },
		{ id: 'webhook', label: 'Webhook', icon: '🔗' },
		{ id: 'pagerduty', label: 'PagerDuty', icon: '📟' }
	];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadAlertSystem();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/alert-center');
			
			ws.onopen = () => {
				console.log('Alert Center WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleAlertUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Alert WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Alert WebSocket connection failed:', error);
		}
	}
	
	function handleAlertUpdate(data: any) {
		switch (data.type) {
			case 'new_alert':
				addAlert(data.alert);
				break;
			case 'alert_resolved':
				resolveAlert(data.alert_id);
				break;
			case 'alert_silenced':
				silenceAlert(data.alert_id, data.duration);
				break;
			case 'rule_triggered':
				updateRuleMetrics(data.rule_id, data.metrics);
				break;
		}
	}
	
	async function loadAlertSystem() {
		isLoading.set(true);
		try {
			const [alerts, rules, channels, history, metrics] = await Promise.all([
				invoke('get_active_alerts'),
				invoke('get_alert_rules'),
				invoke('get_notification_channels'),
				invoke('get_alert_history'),
				invoke('get_alert_metrics')
			]);
			
			alertSystem.update(current => ({
				...current,
				active_alerts: alerts || [],
				alert_rules: rules || [],
				notification_channels: channels || [],
				alert_history: history || [],
				metrics: { ...current.metrics, ...metrics }
			}));
		} catch (error) {
			console.error('Failed to load alert system:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	async function createAlertRule() {
		try {
			const result = await invoke('create_alert_rule', { rule: newRule });
			
			alertSystem.update(current => ({
				...current,
				alert_rules: [...current.alert_rules, result]
			}));
			
			// Reset form
			newRule = {
				name: '',
				query: '',
				severity: 'warning',
				duration: '5m',
				labels: {},
				annotations: {},
				channels: []
			};
			
			dispatch('rule_created', result);
		} catch (error) {
			console.error('Failed to create alert rule:', error);
		}
	}
	
	async function createNotificationChannel() {
		try {
			const result = await invoke('create_notification_channel', { channel: newChannel });
			
			alertSystem.update(current => ({
				...current,
				notification_channels: [...current.notification_channels, result]
			}));
			
			// Reset form
			newChannel = {
				name: '',
				type: 'slack',
				config: {},
				enabled: true
			};
			
			dispatch('channel_created', result);
		} catch (error) {
			console.error('Failed to create notification channel:', error);
		}
	}
	
	async function silenceAlert(alertId: string, duration: string = '1h') {
		try {
			await invoke('silence_alert', { alertId, duration });
			
			// Move alert to silenced list
			const alert = $alertSystem.active_alerts.find(a => a.id === alertId);
			if (alert) {
				alertSystem.update(current => ({
					...current,
					active_alerts: current.active_alerts.filter(a => a.id !== alertId),
					silenced_alerts: [...current.silenced_alerts, { ...alert, silenced_until: new Date(Date.now() + parseDuration(duration)).toISOString() }]
				}));
			}
		} catch (error) {
			console.error('Failed to silence alert:', error);
		}
	}
	
	async function resolveAlert(alertId: string) {
		try {
			await invoke('resolve_alert', { alertId });
			
			alertSystem.update(current => ({
				...current,
				active_alerts: current.active_alerts.filter(a => a.id !== alertId),
				metrics: {
					...current.metrics,
					resolved_alerts: current.metrics.resolved_alerts + 1
				}
			}));
		} catch (error) {
			console.error('Failed to resolve alert:', error);
		}
	}
	
	function addAlert(alert: any) {
		alertSystem.update(current => ({
			...current,
			active_alerts: [alert, ...current.active_alerts],
			metrics: {
				...current.metrics,
				total_alerts_today: current.metrics.total_alerts_today + 1,
				[alert.severity + '_alerts']: (current.metrics[alert.severity + '_alerts'] || 0) + 1
			}
		}));
	}
	
	function updateRuleMetrics(ruleId: string, metrics: any) {
		alertSystem.update(current => ({
			...current,
			alert_rules: current.alert_rules.map(rule =>
				rule.id === ruleId ? { ...rule, ...metrics } : rule
			)
		}));
	}
	
	function parseDuration(duration: string): number {
		const match = duration.match(/(\d+)([smhd])/);
		if (!match) return 3600000; // Default 1 hour
		
		const [, amount, unit] = match;
		const multipliers = { s: 1000, m: 60000, h: 3600000, d: 86400000 };
		return parseInt(amount) * (multipliers[unit] || 3600000);
	}
	
	function getSeverityColor(severity: string): string {
		const level = severityLevels.find(s => s.id === severity);
		return level ? level.color : 'text-gray-400';
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleString();
	}
	
	function formatDuration(ms: number): string {
		const minutes = Math.floor(ms / 60000);
		const hours = Math.floor(minutes / 60);
		if (hours > 0) return `${hours}h ${minutes % 60}m`;
		return `${minutes}m`;
	}
	
	function openAlertModal(alert: any) {
		selectedAlert.set(alert);
	}
</script>

<!-- Alert Management Center -->
<div class="alert-center h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-amber-400">Alert Management Center</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{$alertSystem.active_alerts.length} active | {$alertSystem.alert_rules.length} rules
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-amber-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadAlertSystem}
						class="px-3 py-1 bg-amber-600 hover:bg-amber-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Alert Metrics -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-5 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-amber-400">{$alertSystem.metrics.total_alerts_today}</div>
				<div class="text-sm text-gray-400">Alerts Today</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-red-400">{$alertSystem.metrics.critical_alerts}</div>
				<div class="text-sm text-gray-400">Critical</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-yellow-400">{$alertSystem.metrics.warning_alerts}</div>
				<div class="text-sm text-gray-400">Warnings</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-green-400">{$alertSystem.metrics.resolved_alerts}</div>
				<div class="text-sm text-gray-400">Resolved</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-2xl font-bold text-blue-400">{formatDuration($alertSystem.metrics.avg_resolution_time)}</div>
				<div class="text-sm text-gray-400">Avg Resolution</div>
			</div>
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Alert Overview', icon: '📊' },
				{ id: 'active', label: 'Active Alerts', icon: '🚨' },
				{ id: 'rules', label: 'Alert Rules', icon: '📋' },
				{ id: 'channels', label: 'Notification Channels', icon: '📢' },
				{ id: 'history', label: 'Alert History', icon: '📚' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedTab === tab.id
							? 'border-amber-500 text-amber-400'
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
		{#if $selectedTab === 'overview'}
			<!-- Alert Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Active Alerts Summary -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-red-400">Critical Alerts</h3>
					{#if $alertSystem.active_alerts.filter(a => a.severity === 'critical' || a.severity === 'emergency').length === 0}
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">✅</div>
							<p>No critical alerts</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $alertSystem.active_alerts.filter(a => a.severity === 'critical' || a.severity === 'emergency').slice(0, 5) as alert}
								<div class="bg-red-900/30 border border-red-800 rounded p-3">
									<div class="flex items-center justify-between mb-2">
										<span class="font-medium text-white text-sm">{alert.rule_name}</span>
										<span class="text-xs {getSeverityColor(alert.severity)}">{alert.severity}</span>
									</div>
									<div class="text-sm text-gray-300">{alert.description}</div>
									<div class="text-xs text-gray-400 mt-1">{formatDate(alert.fired_at)}</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
				
				<!-- Alert Rules Status -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Alert Rules Status</h3>
					{#if $alertSystem.alert_rules.length === 0}
						<div class="text-center py-8 text-gray-400">
							<div class="text-3xl mb-2">📋</div>
							<p>No alert rules configured</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $alertSystem.alert_rules.slice(0, 8) as rule}
								<div class="bg-gray-900 rounded p-3">
									<div class="flex items-center justify-between mb-2">
										<span class="font-medium text-white text-sm">{rule.name}</span>
										<div class="flex items-center space-x-2">
											<span class="text-xs {getSeverityColor(rule.severity)}">{rule.severity}</span>
											<span class="text-xs text-gray-400">{rule.duration}</span>
										</div>
									</div>
									<div class="text-xs text-gray-400">
										Triggered: {rule.trigger_count || 0} times
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'active'}
			<!-- Active Alerts -->
			<div class="space-y-4">
				{#if $alertSystem.active_alerts.length === 0}
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">✅</div>
						<p>No active alerts</p>
					</div>
				{:else}
					{#each $alertSystem.active_alerts as alert}
						<div class="bg-gray-800 rounded-lg p-6">
							<div class="flex items-center justify-between mb-4">
								<div class="flex items-center space-x-4">
									<h3 class="text-lg font-medium text-white">{alert.rule_name}</h3>
									<span class="px-2 py-1 rounded text-xs {getSeverityColor(alert.severity)} bg-gray-700">
										{alert.severity}
									</span>
									<span class="text-sm text-gray-400">{alert.instance || alert.job}</span>
								</div>
								<div class="flex items-center space-x-2">
									<button
										on:click={() => silenceAlert(alert.id, '1h')}
										class="px-3 py-1 bg-yellow-600 hover:bg-yellow-700 rounded text-sm transition-colors"
									>
										Silence 1h
									</button>
									<button
										on:click={() => resolveAlert(alert.id)}
										class="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-sm transition-colors"
									>
										Resolve
									</button>
									<button
										on:click={() => openAlertModal(alert)}
										class="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
									>
										Details
									</button>
								</div>
							</div>
							
							<div class="mb-4">
								<p class="text-gray-300">{alert.description || alert.summary}</p>
							</div>
							
							<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
								<div>
									<span class="text-gray-400">Fired At:</span>
									<span class="text-white ml-2">{formatDate(alert.fired_at)}</span>
								</div>
								<div>
									<span class="text-gray-400">Duration:</span>
									<span class="text-white ml-2">{formatDuration(Date.now() - new Date(alert.fired_at).getTime())}</span>
								</div>
								{#if alert.value}
									<div>
										<span class="text-gray-400">Value:</span>
										<span class="text-white ml-2">{alert.value}</span>
									</div>
								{/if}
								{#if alert.runbook_url}
									<div>
										<a href={alert.runbook_url} target="_blank" class="text-blue-400 hover:underline">
											📖 Runbook
										</a>
									</div>
								{/if}
							</div>
						</div>
					{/each}
				{/if}
			</div>
			
		{:else if $selectedTab === 'rules'}
			<!-- Alert Rules -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Create Rule -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Create Alert Rule</h3>
					<div class="space-y-4">
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Rule Name</label>
							<input
								type="text"
								bind:value={newRule.name}
								placeholder="Enter rule name"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							/>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">PromQL Query</label>
							<textarea
								bind:value={newRule.query}
								placeholder="Enter Prometheus query"
								rows="3"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							></textarea>
						</div>
						
						<div class="grid grid-cols-2 gap-2">
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Severity</label>
								<select
									bind:value={newRule.severity}
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
								>
									{#each severityLevels as level}
										<option value={level.id}>{level.label}</option>
									{/each}
								</select>
							</div>
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Duration</label>
								<input
									type="text"
									bind:value={newRule.duration}
									placeholder="5m"
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
								/>
							</div>
						</div>
						
						<button
							on:click={createAlertRule}
							class="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded font-medium transition-colors"
							disabled={!newRule.name || !newRule.query}
						>
							Create Rule
						</button>
					</div>
				</div>
				
				<!-- Rules List -->
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Alert Rules</h3>
					{#if $alertSystem.alert_rules.length === 0}
						<div class="text-center py-12 text-gray-400">
							<div class="text-4xl mb-4">📋</div>
							<p>No alert rules configured</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $alertSystem.alert_rules as rule}
								<div class="bg-gray-900 rounded p-4">
									<div class="flex items-center justify-between mb-3">
										<div class="flex items-center space-x-3">
											<h4 class="font-medium text-white">{rule.name}</h4>
											<span class="px-2 py-1 rounded text-xs {getSeverityColor(rule.severity)} bg-gray-800">
												{rule.severity}
											</span>
											<span class="text-sm text-gray-400">for {rule.duration}</span>
										</div>
										<div class="text-sm text-gray-400">
											Triggered: {rule.trigger_count || 0}
										</div>
									</div>
									
									<div class="bg-gray-800 rounded p-2 mb-2">
										<code class="text-green-400 text-xs">{rule.query}</code>
									</div>
									
									<div class="text-sm text-gray-400">
										Last triggered: {rule.last_triggered ? formatDate(rule.last_triggered) : 'Never'}
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'channels'}
			<!-- Notification Channels -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- Create Channel -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-purple-400">Create Notification Channel</h3>
					<div class="space-y-4">
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Channel Name</label>
							<input
								type="text"
								bind:value={newChannel.name}
								placeholder="Enter channel name"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
							/>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Channel Type</label>
							<select
								bind:value={newChannel.type}
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
							>
								{#each channelTypes as type}
									<option value={type.id}>{type.icon} {type.label}</option>
								{/each}
							</select>
						</div>
						
						<div class="flex items-center">
							<input
								type="checkbox"
								bind:checked={newChannel.enabled}
								id="channel_enabled"
								class="mr-2"
							/>
							<label for="channel_enabled" class="text-sm text-gray-300">Enable notifications</label>
						</div>
						
						<button
							on:click={createNotificationChannel}
							class="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded font-medium transition-colors"
							disabled={!newChannel.name}
						>
							Create Channel
						</button>
					</div>
				</div>
				
				<!-- Channels List -->
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-purple-400">Notification Channels</h3>
					{#if $alertSystem.notification_channels.length === 0}
						<div class="text-center py-12 text-gray-400">
							<div class="text-4xl mb-4">📢</div>
							<p>No notification channels configured</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each $alertSystem.notification_channels as channel}
								<div class="bg-gray-900 rounded p-4">
									<div class="flex items-center justify-between mb-3">
										<div class="flex items-center space-x-3">
											<h4 class="font-medium text-white">{channel.name}</h4>
											<span class="text-sm text-gray-400">
												{channelTypes.find(t => t.id === channel.type)?.icon} {channel.type}
											</span>
											<span class="px-2 py-1 rounded text-xs {channel.enabled ? 'text-green-400' : 'text-gray-400'} bg-gray-800">
												{channel.enabled ? 'Enabled' : 'Disabled'}
											</span>
										</div>
										<div class="text-sm text-gray-400">
											Sent: {channel.messages_sent || 0}
										</div>
									</div>
									
									<div class="text-sm text-gray-400">
										Last used: {channel.last_used ? formatDate(channel.last_used) : 'Never'}
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else if $selectedTab === 'history'}
			<!-- Alert History -->
			<div class="space-y-4">
				{#if $alertSystem.alert_history.length === 0}
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">📚</div>
						<p>No alert history</p>
					</div>
				{:else}
					{#each $alertSystem.alert_history as alert}
						<div class="bg-gray-800 rounded-lg p-6">
							<div class="flex items-center justify-between mb-4">
								<div class="flex items-center space-x-4">
									<h3 class="text-lg font-medium text-white">{alert.rule_name}</h3>
									<span class="px-2 py-1 rounded text-xs {getSeverityColor(alert.severity)} bg-gray-700">
										{alert.severity}
									</span>
									<span class="px-2 py-1 rounded text-xs text-green-400 bg-gray-700">
										Resolved
									</span>
								</div>
								<div class="text-sm text-gray-400">
									{formatDate(alert.fired_at)} - {formatDate(alert.resolved_at)}
								</div>
							</div>
							
							<div class="mb-4">
								<p class="text-gray-300">{alert.description}</p>
							</div>
							
							<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
								<div>
									<span class="text-gray-400">Duration:</span>
									<span class="text-white ml-2">
										{formatDuration(new Date(alert.resolved_at).getTime() - new Date(alert.fired_at).getTime())}
									</span>
								</div>
								<div>
									<span class="text-gray-400">Resolved By:</span>
									<span class="text-white ml-2">{alert.resolved_by || 'Auto'}</span>
								</div>
								{#if alert.value}
									<div>
										<span class="text-gray-400">Peak Value:</span>
										<span class="text-white ml-2">{alert.value}</span>
									</div>
								{/if}
								{#if alert.notifications_sent}
									<div>
										<span class="text-gray-400">Notifications:</span>
										<span class="text-white ml-2">{alert.notifications_sent}</span>
									</div>
								{/if}
							</div>
						</div>
					{/each}
				{/if}
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Alert Management | Prometheus + AlertManager
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('test_notifications')}
					class="px-4 py-2 bg-amber-600 hover:bg-amber-700 rounded text-sm font-medium transition-colors"
				>
					Test Notifications
				</button>
				<button
					on:click={() => dispatch('export_alert_config')}
					class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
				>
					Export Config
				</button>
			</div>
		</div>
	</div>
</div>

<!-- Alert Detail Modal -->
{#if $selectedAlert}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedAlert.set(null)}>
		<div class="max-w-3xl w-full mx-4 bg-gray-800 rounded-lg p-6" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-amber-400">{$selectedAlert.rule_name}</h3>
				<button
					on:click={() => selectedAlert.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Alert Details</h4>
					<div class="space-y-2 text-sm">
						<div class="flex justify-between">
							<span class="text-gray-400">Severity:</span>
							<span class="{getSeverityColor($selectedAlert.severity)}">{$selectedAlert.severity}</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Instance:</span>
							<span class="text-white">{$selectedAlert.instance || 'N/A'}</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Value:</span>
							<span class="text-white">{$selectedAlert.value || 'N/A'}</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-400">Fired At:</span>
							<span class="text-white">{formatDate($selectedAlert.fired_at)}</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Description</h4>
					<p class="text-gray-300">{$selectedAlert.description || $selectedAlert.summary}</p>
				</div>
				
				{#if $selectedAlert.labels}
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-2">Labels</h4>
						<div class="flex flex-wrap gap-2">
							{#each Object.entries($selectedAlert.labels) as [key, value]}
								<span class="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded">
									{key}: {value}
								</span>
							{/each}
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.alert-center {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.alert-center *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.alert-center *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.alert-center *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.alert-center *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>