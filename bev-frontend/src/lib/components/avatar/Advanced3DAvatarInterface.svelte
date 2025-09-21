<!--
Advanced 3D Avatar Interface - Complete integration of all avatar components
Features: 3D rendering, OSINT integration, control panels, system monitoring
Replaces: Live2DAvatarInterface.svelte with enhanced 3D capabilities
-->

<script lang="ts">
	import { onMount, onDestroy, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	
	// Import avatar components
	import Avatar3DRenderer from './Avatar3DRenderer.svelte';
	import AvatarControlPanel from './AvatarControlPanel.svelte';
	import OSINTInvestigationPanel from './OSINTInvestigationPanel.svelte';
	import SystemIntegrationPanel from './SystemIntegrationPanel.svelte';
	
	// Import avatar services
	import { avatarClient, OSINTAvatarIntegration } from '$lib/services/AvatarWebSocketClient';
	
	const dispatch = createEventDispatcher();
	
	// Interface state
	const interfaceState = writable({
		active_panel: 'main', // 'main', 'osint', 'control', 'system'
		layout_mode: 'desktop', // 'desktop', 'tablet', 'mobile'
		avatar_visible: true,
		panels_visible: true,
		fullscreen_avatar: false,
		connection_status: 'disconnected',
		current_mode: 'interactive' // 'interactive', 'osint_analysis', 'system_monitoring'
	});
	
	// Layout configurations
	const layoutConfigs = {
		desktop: {
			avatar_width: '60%',
			panel_width: '40%',
			min_avatar_width: '400px',
			enable_split_view: true
		},
		tablet: {
			avatar_width: '100%',
			panel_width: '100%',
			min_avatar_width: '300px',
			enable_split_view: false
		},
		mobile: {
			avatar_width: '100%',
			panel_width: '100%',
			min_avatar_width: '250px',
			enable_split_view: false
		}
	};
	
	// Avatar component references
	let avatar3DRenderer: Avatar3DRenderer;
	let avatarControlPanel: AvatarControlPanel;
	let osintPanel: OSINTInvestigationPanel;
	let systemPanel: SystemIntegrationPanel;
	
	// Panel management
	let panelStack: string[] = ['main'];
	let panelHistory: string[] = [];
	
	// Performance monitoring
	const performanceMetrics = writable({
		frame_rate: 0,
		render_time: 0,
		memory_usage: 0,
		gpu_utilization: 0,
		interaction_latency: 0
	});
	
	// Navigation state
	let selectedMainTab = 'avatar'; // 'avatar', 'osint', 'system'
	let showPanelSelector = false;
	let isResizing = false;
	let splitPosition = 60; // Percentage for avatar panel
	
	onMount(async () => {
		await initializeInterface();
		setupEventListeners();
		detectLayoutMode();
	});
	
	onDestroy(() => {
		cleanup();
	});
	
	async function initializeInterface() {
		try {
			// Initialize avatar integration
			await OSINTAvatarIntegration.initialize();
			
			// Setup interface state listeners
			avatarClient.subscribe('connection', (event) => {
				interfaceState.update(state => ({
					...state,
					connection_status: event.status
				}));
			});
			
			// Initial avatar greeting for desktop interface
			setTimeout(async () => {
				if ($interfaceState.layout_mode === 'desktop') {
					await avatarClient.setEmotion('friendly');
					await avatarClient.speak('Welcome to the BEV OSINT Framework. I\'m your AI research companion ready to assist with cybersecurity investigations.', 'friendly');
				}
			}, 2000);
			
			dispatch('interface_initialized');
			
		} catch (error) {
			console.error('Failed to initialize avatar interface:', error);
		}
	}
	
	function setupEventListeners() {
		// Window resize detection
		window.addEventListener('resize', detectLayoutMode);
		
		// Keyboard shortcuts
		window.addEventListener('keydown', handleKeyboardShortcuts);
		
		// Avatar events
		window.addEventListener('avatar_mode_change', handleAvatarModeChange);
		window.addEventListener('osint_investigation_start', handleOSINTStart);
		window.addEventListener('osint_investigation_complete', handleOSINTComplete);
		
		// Performance monitoring
		setInterval(updatePerformanceMetrics, 1000);
	}
	
	function detectLayoutMode() {
		const width = window.innerWidth;
		let newMode = 'desktop';
		
		if (width < 768) {
			newMode = 'mobile';
		} else if (width < 1024) {
			newMode = 'tablet';
		}
		
		interfaceState.update(state => ({
			...state,
			layout_mode: newMode
		}));
	}
	
	function handleKeyboardShortcuts(event: KeyboardEvent) {
		// Ctrl/Cmd + shortcuts
		if (event.ctrlKey || event.metaKey) {
			switch (event.key) {
				case '1':
					event.preventDefault();
					switchToPanel('main');
					break;
				case '2':
					event.preventDefault();
					switchToPanel('osint');
					break;
				case '3':
					event.preventDefault();
					switchToPanel('control');
					break;
				case '4':
					event.preventDefault();
					switchToPanel('system');
					break;
				case 'f':
					event.preventDefault();
					toggleFullscreenAvatar();
					break;
				case 'h':
					event.preventDefault();
					togglePanelsVisibility();
					break;
			}
		}
		
		// Escape key
		if (event.key === 'Escape') {
			if ($interfaceState.fullscreen_avatar) {
				toggleFullscreenAvatar();
			} else if (panelStack.length > 1) {
				goBackPanel();
			}
		}
	}
	
	function handleAvatarModeChange(event: CustomEvent) {
		const { mode } = event.detail;
		
		interfaceState.update(state => ({
			...state,
			current_mode: mode
		}));
		
		// Auto-switch panels based on mode
		switch (mode) {
			case 'osint_analysis':
				switchToPanel('osint');
				break;
			case 'system_monitoring':
				switchToPanel('system');
				break;
			case 'interactive':
				switchToPanel('main');
				break;
		}
	}
	
	function handleOSINTStart(event: CustomEvent) {
		// Switch to OSINT panel when investigation starts
		switchToPanel('osint');
		
		interfaceState.update(state => ({
			...state,
			current_mode: 'osint_analysis'
		}));
	}
	
	function handleOSINTComplete(event: CustomEvent) {
		// Return to main panel when investigation completes
		setTimeout(() => {
			switchToPanel('main');
			interfaceState.update(state => ({
				...state,
				current_mode: 'interactive'
			}));
		}, 3000);
	}
	
	function switchToPanel(panelName: string) {
		if (panelStack[panelStack.length - 1] !== panelName) {
			panelHistory = [...panelStack];
			panelStack = [...panelStack, panelName];
		}
		
		interfaceState.update(state => ({
			...state,
			active_panel: panelName
		}));
		
		dispatch('panel_changed', { panel: panelName });
	}
	
	function goBackPanel() {
		if (panelStack.length > 1) {
			panelStack = panelStack.slice(0, -1);
			
			interfaceState.update(state => ({
				...state,
				active_panel: panelStack[panelStack.length - 1]
			}));
		}
	}
	
	function toggleFullscreenAvatar() {
		interfaceState.update(state => ({
			...state,
			fullscreen_avatar: !state.fullscreen_avatar,
			panels_visible: state.fullscreen_avatar // Hide panels when going fullscreen
		}));
	}
	
	function togglePanelsVisibility() {
		interfaceState.update(state => ({
			...state,
			panels_visible: !state.panels_visible
		}));
	}
	
	function updatePerformanceMetrics() {
		// Get performance data from avatar renderer
		if (avatar3DRenderer) {
			// Performance metrics would be collected from the 3D renderer
			performanceMetrics.update(metrics => ({
				...metrics,
				frame_rate: Math.random() * 60 + 30, // Mock data
				render_time: Math.random() * 16 + 8,
				memory_usage: Math.random() * 100 + 50,
				gpu_utilization: Math.random() * 80 + 20,
				interaction_latency: Math.random() * 10 + 5
			}));
		}
	}
	
	function handleSplitResize(event: MouseEvent) {
		if (!isResizing) return;
		
		const containerWidth = (event.currentTarget as HTMLElement).offsetWidth;
		const newPosition = (event.clientX / containerWidth) * 100;
		
		splitPosition = Math.max(30, Math.min(80, newPosition));
	}
	
	function startResize() {
		isResizing = true;
		document.addEventListener('mousemove', handleSplitResize);
		document.addEventListener('mouseup', stopResize);
	}
	
	function stopResize() {
		isResizing = false;
		document.removeEventListener('mousemove', handleSplitResize);
		document.removeEventListener('mouseup', stopResize);
	}
	
	function cleanup() {
		window.removeEventListener('resize', detectLayoutMode);
		window.removeEventListener('keydown', handleKeyboardShortcuts);
		window.removeEventListener('avatar_mode_change', handleAvatarModeChange);
		window.removeEventListener('osint_investigation_start', handleOSINTStart);
		window.removeEventListener('osint_investigation_complete', handleOSINTComplete);
	}
	
	// Reactive layout configuration
	$: layoutConfig = layoutConfigs[$interfaceState.layout_mode];
	$: showSplitView = layoutConfig.enable_split_view && $interfaceState.panels_visible && !$interfaceState.fullscreen_avatar;
</script>

<!-- Advanced 3D Avatar Interface -->
<div class="advanced-avatar-interface h-full bg-gray-900 text-white">
	<!-- Mobile/Tablet Header -->
	{#if $interfaceState.layout_mode !== 'desktop'}
		<div class="border-b border-gray-800 p-4">
			<div class="flex items-center justify-between">
				<h1 class="text-xl font-bold text-cyan-400">BEV Avatar</h1>
				<div class="flex items-center space-x-2">
					<!-- Connection status -->
					<div class="flex items-center space-x-1">
						<div class="w-2 h-2 rounded-full {
							$interfaceState.connection_status === 'connected' ? 'bg-green-400' :
							$interfaceState.connection_status === 'connecting' ? 'bg-yellow-400' :
							'bg-red-400'
						}"></div>
						<span class="text-xs text-gray-400">{$interfaceState.connection_status}</span>
					</div>
					
					<!-- Panel selector button -->
					<button
						on:click={() => showPanelSelector = !showPanelSelector}
						class="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
					>
						☰
					</button>
				</div>
			</div>
			
			<!-- Panel selector dropdown -->
			{#if showPanelSelector}
				<div class="mt-3 bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
					{#each [
						{ id: 'main', label: 'Avatar Control', icon: '🤖' },
						{ id: 'osint', label: 'OSINT Investigation', icon: '🔍' },
						{ id: 'system', label: 'System Status', icon: '⚙️' }
					] as panel}
						<button
							class="w-full p-3 text-left hover:bg-gray-700 transition-colors {
								$interfaceState.active_panel === panel.id ? 'bg-gray-700' : ''
							}"
							on:click={() => {
								switchToPanel(panel.id);
								showPanelSelector = false;
							}}
						>
							<span class="text-lg mr-2">{panel.icon}</span>
							{panel.label}
						</button>
					{/each}
				</div>
			{/if}
		</div>
	{/if}
	
	<!-- Main Content -->
	<div class="flex-1 flex {showSplitView ? 'flex-row' : 'flex-col'} overflow-hidden">
		<!-- Avatar Renderer Section -->
		<div 
			class="avatar-section {$interfaceState.fullscreen_avatar ? 'w-full' : ''} {
				showSplitView ? '' : ($interfaceState.layout_mode === 'desktop' ? 'h-2/3' : 'h-1/2')
			} flex flex-col bg-gray-900"
			style="{showSplitView ? `width: ${splitPosition}%` : ''}"
		>
			<!-- Desktop Avatar Header -->
			{#if $interfaceState.layout_mode === 'desktop' && !$interfaceState.fullscreen_avatar}
				<div class="border-b border-gray-800 p-3">
					<div class="flex items-center justify-between">
						<h2 class="font-semibold text-cyan-400">3D Avatar</h2>
						<div class="flex items-center space-x-2">
							<!-- Performance metrics -->
							<div class="text-xs text-gray-400">
								{$performanceMetrics.frame_rate.toFixed(0)} FPS
							</div>
							
							<!-- Avatar controls -->
							<button
								on:click={toggleFullscreenAvatar}
								class="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
								title="Toggle Fullscreen (Ctrl+F)"
							>
								⛶
							</button>
						</div>
					</div>
				</div>
			{/if}
			
			<!-- 3D Avatar Renderer -->
			<div class="flex-1">
				<Avatar3DRenderer
					bind:this={avatar3DRenderer}
					avatarServiceUrl="http://localhost:8092"
					enableGPUAcceleration={true}
					enablePerformanceMonitoring={true}
					maxFPS={60}
					qualityLevel="high"
					on:webgl_initialized={(e) => dispatch('avatar_initialized', e.detail)}
					on:frame_received={(e) => dispatch('avatar_frame', e.detail)}
					on:performance_update={(e) => dispatch('performance_update', e.detail)}
				/>
			</div>
		</div>
		
		<!-- Split Resize Handle -->
		{#if showSplitView}
			<div
				class="w-1 bg-gray-700 hover:bg-gray-600 cursor-col-resize transition-colors"
				on:mousedown={startResize}
			></div>
		{/if}
		
		<!-- Control Panels Section -->
		{#if $interfaceState.panels_visible && !$interfaceState.fullscreen_avatar}
			<div 
				class="panels-section {
					showSplitView ? '' : ($interfaceState.layout_mode === 'desktop' ? 'h-1/3' : 'h-1/2')
				} flex flex-col border-l border-gray-800 bg-gray-850"
				style="{showSplitView ? `width: ${100 - splitPosition}%` : ''}"
			>
				<!-- Desktop Panel Tabs -->
				{#if $interfaceState.layout_mode === 'desktop'}
					<div class="border-b border-gray-800">
						<nav class="flex">
							{#each [
								{ id: 'main', label: 'Control', icon: '🎛️' },
								{ id: 'osint', label: 'OSINT', icon: '🔍' },
								{ id: 'system', label: 'System', icon: '⚙️' }
							] as tab}
								<button
									class="px-4 py-3 border-b-2 font-medium text-sm {
										$interfaceState.active_panel === tab.id
											? 'border-cyan-500 text-cyan-400 bg-gray-800'
											: 'border-transparent text-gray-400 hover:text-gray-300 hover:bg-gray-800'
									} transition-colors"
									on:click={() => switchToPanel(tab.id)}
								>
									<span class="flex items-center space-x-2">
										<span>{tab.icon}</span>
										<span>{tab.label}</span>
									</span>
								</button>
							{/each}
							
							<!-- Panel controls -->
							<div class="ml-auto flex items-center px-3">
								<button
									on:click={togglePanelsVisibility}
									class="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
									title="Hide Panels (Ctrl+H)"
								>
									✕
								</button>
							</div>
						</nav>
					</div>
				{/if}
				
				<!-- Panel Content -->
				<div class="flex-1 overflow-hidden">
					{#if $interfaceState.active_panel === 'main'}
						<AvatarControlPanel
							bind:this={avatarControlPanel}
							on:emotion_changed={(e) => dispatch('emotion_changed', e.detail)}
							on:gesture_performed={(e) => dispatch('gesture_performed', e.detail)}
							on:speech_triggered={(e) => dispatch('speech_triggered', e.detail)}
							on:interaction_mode_changed={(e) => dispatch('mode_changed', e.detail)}
						/>
						
					{:else if $interfaceState.active_panel === 'osint'}
						<OSINTInvestigationPanel
							bind:this={osintPanel}
							on:investigation_complete={(e) => dispatch('investigation_complete', e.detail)}
						/>
						
					{:else if $interfaceState.active_panel === 'system'}
						<SystemIntegrationPanel
							bind:this={systemPanel}
						/>
					{/if}
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Mobile/Tablet Bottom Panel (when panels hidden) -->
	{#if !$interfaceState.panels_visible && $interfaceState.layout_mode !== 'desktop'}
		<div class="border-t border-gray-800 p-2">
			<button
				on:click={togglePanelsVisibility}
				class="w-full px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded font-medium transition-colors"
			>
				Show Controls
			</button>
		</div>
	{/if}
	
	<!-- Fullscreen Avatar Overlay Controls -->
	{#if $interfaceState.fullscreen_avatar}
		<div class="absolute top-4 right-4 z-10">
			<div class="bg-black bg-opacity-75 rounded-lg p-2 flex space-x-2">
				<button
					on:click={togglePanelsVisibility}
					class="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
					title="Show Panels"
				>
					☰
				</button>
				<button
					on:click={toggleFullscreenAvatar}
					class="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
					title="Exit Fullscreen (Esc)"
				>
					✕
				</button>
			</div>
		</div>
	{/if}
	
	<!-- Debug Performance Overlay (development only) -->
	{#if false}
		<div class="absolute bottom-4 left-4 bg-black bg-opacity-75 rounded p-2 text-xs font-mono">
			<div>FPS: {$performanceMetrics.frame_rate.toFixed(0)}</div>
			<div>Render: {$performanceMetrics.render_time.toFixed(1)}ms</div>
			<div>GPU: {$performanceMetrics.gpu_utilization.toFixed(0)}%</div>
			<div>Latency: {$performanceMetrics.interaction_latency.toFixed(1)}ms</div>
		</div>
	{/if}
</div>

<style>
	.advanced-avatar-interface {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
		background: #111827;
	}
	
	.avatar-section {
		position: relative;
		min-height: 300px;
	}
	
	.panels-section {
		position: relative;
		min-width: 300px;
		background: #1f2937;
	}
	
	/* Custom scrollbar for panels */
	:global(.panels-section *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.panels-section *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.panels-section *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	/* Resize handle styling */
	.cursor-col-resize {
		cursor: col-resize;
	}
	
	.cursor-col-resize:hover {
		background: #4b5563;
	}
	
	/* Responsive adjustments */
	@media (max-width: 768px) {
		.avatar-section {
			min-height: 250px;
		}
		
		.panels-section {
			min-width: 100%;
		}
	}
	
	/* Animation for panel transitions */
	.panels-section {
		transition: all 0.3s ease-in-out;
	}
	
	/* Performance optimizations */
	.advanced-avatar-interface {
		will-change: auto;
		contain: layout style paint;
	}
</style>