<!--
Advanced 3D Avatar System Route - Next-generation avatar integration
Integrates: Advanced3DAvatarInterface.svelte with full desktop integration
Backend: Advanced Avatar Service (port 8092) with Gaussian Splatting
Features: 3D rendering, OSINT integration, GPU acceleration, real-time communication
-->

<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { browser } from '$app/environment';
	import Advanced3DAvatarInterface from '$lib/components/avatar/Advanced3DAvatarInterface.svelte';
	
	let pageTitle = 'Advanced 3D Avatar System';
	let pageDescription = 'Interactive 3D avatar with Gaussian Splatting, OSINT integration, and real-time emotional intelligence';
	
	// System requirements check
	let systemCompatible = false;
	let webglSupported = false;
	let gpuAcceleration = false;
	let systemChecks = {
		webgl2: false,
		gpu_memory: false,
		performance_api: false,
		websocket: false,
		required_extensions: false
	};
	
	// Performance monitoring
	let performanceStats = {
		initialization_time: 0,
		average_fps: 0,
		gpu_utilization: 0,
		memory_usage: 0
	};
	
	// Error handling
	let initializationError = null;
	let showCompatibilityWarning = false;
	
	onMount(async () => {
		document.title = `${pageTitle} | BEV OSINT Framework`;
		
		if (browser) {
			await checkSystemCompatibility();
			await monitorPerformance();
		}
	});
	
	onDestroy(() => {
		// Cleanup any ongoing monitoring
	});
	
	async function checkSystemCompatibility() {
		try {
			const startTime = performance.now();
			
			// Check WebGL2 support
			const canvas = document.createElement('canvas');
			const gl = canvas.getContext('webgl2');
			systemChecks.webgl2 = !!gl;
			webglSupported = systemChecks.webgl2;
			
			if (gl) {
				// Check required WebGL extensions
				const requiredExtensions = [
					'EXT_color_buffer_float',
					'OES_texture_float',
					'WEBGL_color_buffer_float'
				];
				
				const supportedExtensions = requiredExtensions.filter(ext => gl.getExtension(ext));
				systemChecks.required_extensions = supportedExtensions.length >= 2;
				
				// Check GPU memory (approximation)
				const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
				if (debugInfo) {
					const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
					gpuAcceleration = renderer.includes('RTX') || renderer.includes('GeForce') || renderer.includes('Radeon');
					systemChecks.gpu_memory = gpuAcceleration;
				}
			}
			
			// Check Performance API
			systemChecks.performance_api = 'performance' in window && 'mark' in performance;
			
			// Check WebSocket support
			systemChecks.websocket = 'WebSocket' in window;
			
			// Overall compatibility check
			systemCompatible = Object.values(systemChecks).every(check => check);
			
			const endTime = performance.now();
			performanceStats.initialization_time = endTime - startTime;
			
			if (!systemCompatible) {
				showCompatibilityWarning = true;
			}
			
		} catch (error) {
			console.error('System compatibility check failed:', error);
			initializationError = error;
			showCompatibilityWarning = true;
		}
	}
	
	async function monitorPerformance() {
		if (!systemCompatible) return;
		
		// Monitor FPS and performance metrics
		let frameCount = 0;
		let lastTime = performance.now();
		
		function updatePerformance() {
			frameCount++;
			const currentTime = performance.now();
			
			if (currentTime - lastTime >= 1000) {
				performanceStats.average_fps = frameCount;
				frameCount = 0;
				lastTime = currentTime;
				
				// Update GPU utilization (mock data for display)
				performanceStats.gpu_utilization = Math.min(100, Math.random() * 80 + 20);
				performanceStats.memory_usage = Math.min(100, Math.random() * 60 + 30);
			}
			
			requestAnimationFrame(updatePerformance);
		}
		
		updatePerformance();
	}
	
	function handleAvatarInitialized(event) {
		console.log('Avatar system initialized:', event.detail);
		performanceStats.initialization_time = performance.now();
	}
	
	function handleAvatarFrame(event) {
		// Handle frame updates for performance monitoring
		const { splatCount, emotion } = event.detail;
		// Update performance stats based on frame data
	}
	
	function handlePerformanceUpdate(event) {
		const perfData = event.detail;
		performanceStats = { ...performanceStats, ...perfData };
	}
	
	function handleInvestigationComplete(event) {
		console.log('OSINT investigation completed:', event.detail);
		// Handle investigation completion
	}
	
	function dismissCompatibilityWarning() {
		showCompatibilityWarning = false;
	}
	
	function retryInitialization() {
		initializationError = null;
		showCompatibilityWarning = false;
		checkSystemCompatibility();
	}
</script>

<svelte:head>
	<title>{pageTitle} | BEV OSINT Framework</title>
	<meta name="description" content={pageDescription} />
	<meta name="viewport" content="width=device-width, initial-scale=1.0" />
	
	<!-- Preload critical resources -->
	<link rel="preconnect" href="http://localhost:8092" />
	<link rel="preconnect" href="http://localhost:3010" />
	
	<!-- GPU optimization hints -->
	<meta name="renderer" content="gpu" />
	<meta name="force-device-pixel-ratio" content="1" />
</svelte:head>

<!-- Page Container -->
<div class="avatar-page-container bg-gray-900 min-h-screen">
	<!-- Compatibility Warning Modal -->
	{#if showCompatibilityWarning}
		<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
			<div class="bg-gray-800 rounded-lg p-8 max-w-md mx-4 border border-gray-700">
				<div class="text-center">
					<div class="text-4xl mb-4">⚠️</div>
					<h3 class="text-xl font-bold text-white mb-4">System Compatibility Check</h3>
					
					{#if initializationError}
						<div class="text-red-400 mb-4">
							<p class="font-medium">Initialization Error:</p>
							<p class="text-sm">{initializationError.message}</p>
						</div>
					{:else}
						<div class="text-left space-y-2 mb-6">
							<h4 class="font-medium text-white mb-2">System Requirements:</h4>
							{#each Object.entries(systemChecks) as [requirement, supported]}
								<div class="flex items-center justify-between text-sm">
									<span class="text-gray-300 capitalize">{requirement.replace('_', ' ')}:</span>
									<span class="{supported ? 'text-green-400' : 'text-red-400'}">
										{supported ? '✅' : '❌'}
									</span>
								</div>
							{/each}
						</div>
					{/if}
					
					<div class="text-sm text-gray-400 mb-6">
						<p>For optimal performance, ensure you have:</p>
						<ul class="list-disc list-inside mt-2 text-left">
							<li>Modern GPU with WebGL2 support</li>
							<li>Hardware acceleration enabled</li>
							<li>Updated graphics drivers</li>
							<li>Chrome/Firefox/Edge browser</li>
						</ul>
					</div>
					
					<div class="flex space-x-3">
						<button
							on:click={retryInitialization}
							class="flex-1 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded font-medium transition-colors"
						>
							Retry
						</button>
						<button
							on:click={dismissCompatibilityWarning}
							class="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded font-medium transition-colors"
						>
							Continue Anyway
						</button>
					</div>
				</div>
			</div>
		</div>
	{/if}
	
	<!-- Page Header -->
	<div class="bg-gray-900 border-b border-gray-800">
		<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
			<div class="py-6">
				<div class="flex items-center justify-between">
					<div>
						<h1 class="text-3xl font-bold text-white">{pageTitle}</h1>
						<p class="mt-2 text-gray-400">{pageDescription}</p>
					</div>
					
					<!-- Status Indicators -->
					<div class="flex items-center space-x-4">
						<!-- System Status -->
						<div class="bg-gray-800 rounded-lg px-3 py-2">
							<div class="flex items-center space-x-2">
								<div class="w-2 h-2 rounded-full {systemCompatible ? 'bg-green-400' : 'bg-red-400'}"></div>
								<span class="text-sm text-gray-400">
									{systemCompatible ? 'Compatible' : 'Issues Detected'}
								</span>
							</div>
						</div>
						
						<!-- WebGL Status -->
						<div class="bg-gray-800 rounded-lg px-3 py-2">
							<div class="flex items-center space-x-2">
								<span class="text-sm text-gray-400">WebGL2:</span>
								<span class="text-sm {webglSupported ? 'text-green-400' : 'text-red-400'}">
									{webglSupported ? 'Supported' : 'Not Supported'}
								</span>
							</div>
						</div>
						
						<!-- GPU Acceleration -->
						<div class="bg-gray-800 rounded-lg px-3 py-2">
							<div class="flex items-center space-x-2">
								<span class="text-sm text-gray-400">GPU:</span>
								<span class="text-sm {gpuAcceleration ? 'text-green-400' : 'text-yellow-400'}">
									{gpuAcceleration ? 'Accelerated' : 'Software'}
								</span>
							</div>
						</div>
						
						<!-- Performance Stats -->
						{#if systemCompatible}
							<div class="bg-gray-800 rounded-lg px-3 py-2">
								<div class="flex items-center space-x-4 text-sm">
									<div class="text-gray-400">
										FPS: <span class="text-cyan-400">{performanceStats.average_fps}</span>
									</div>
									<div class="text-gray-400">
										GPU: <span class="text-green-400">{performanceStats.gpu_utilization.toFixed(0)}%</span>
									</div>
								</div>
							</div>
						{/if}
					</div>
				</div>
			</div>
		</div>
	</div>
	
	<!-- Main Avatar Interface -->
	{#if systemCompatible || !showCompatibilityWarning}
		<div class="h-screen">
			<Advanced3DAvatarInterface
				on:interface_initialized={handleAvatarInitialized}
				on:avatar_frame={handleAvatarFrame}
				on:performance_update={handlePerformanceUpdate}
				on:investigation_complete={handleInvestigationComplete}
				on:emotion_changed={(e) => console.log('Emotion changed:', e.detail)}
				on:gesture_performed={(e) => console.log('Gesture performed:', e.detail)}
				on:speech_triggered={(e) => console.log('Speech triggered:', e.detail)}
				on:mode_changed={(e) => console.log('Mode changed:', e.detail)}
				on:panel_changed={(e) => console.log('Panel changed:', e.detail)}
			/>
		</div>
	{:else}
		<!-- Fallback Content -->
		<div class="flex items-center justify-center h-96">
			<div class="text-center">
				<div class="text-6xl mb-4">🤖</div>
				<h2 class="text-2xl font-bold text-white mb-2">Avatar System Loading</h2>
				<p class="text-gray-400 mb-4">Initializing 3D avatar and checking system compatibility...</p>
				<div class="w-8 h-8 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto"></div>
			</div>
		</div>
	{/if}
	
	<!-- Performance Debug Panel (development only) -->
	{#if browser && window.location.hostname === 'localhost' && false}
		<div class="fixed bottom-4 right-4 bg-black bg-opacity-90 rounded-lg p-4 text-xs font-mono text-white">
			<h4 class="font-bold mb-2">Performance Debug</h4>
			<div class="space-y-1">
				<div>Init Time: {performanceStats.initialization_time.toFixed(2)}ms</div>
				<div>Avg FPS: {performanceStats.average_fps}</div>
				<div>GPU Util: {performanceStats.gpu_utilization.toFixed(1)}%</div>
				<div>Memory: {performanceStats.memory_usage.toFixed(1)}%</div>
			</div>
		</div>
	{/if}
</div>

<style>
	.avatar-page-container {
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
		background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
		min-height: 100vh;
	}
	
	/* Ensure proper GPU layer creation */
	:global(.avatar-page-container *) {
		transform: translateZ(0);
		backface-visibility: hidden;
		perspective: 1000px;
	}
	
	/* Optimize for high DPI displays */
	@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
		.avatar-page-container {
			image-rendering: -webkit-optimize-contrast;
			image-rendering: crisp-edges;
		}
	}
	
	/* Dark theme optimizations */
	:global(body) {
		background-color: #111827;
		color: white;
		overscroll-behavior: none;
	}
	
	/* Performance optimizations */
	.avatar-page-container {
		will-change: auto;
		contain: layout style paint;
	}
</style>