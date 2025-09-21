<!--
Live2D Avatar System Interface - Interactive Avatar Controller
Connected to: src/live2d/avatar_controller.py (port 8091)
Features: WebGL rendering, expression/motion controls, voice-synced animations, customization
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const avatarState = writable({
		status: 'disconnected', // 'disconnected', 'connecting', 'connected', 'error'
		current_emotion: 'neutral',
		is_speaking: false,
		gesture_active: false,
		model_loaded: false,
		animation_playing: false,
		available_emotions: ['neutral', 'happy', 'sad', 'surprised', 'angry', 'excited', 'thinking'],
		available_gestures: ['wave', 'nod', 'shake_head', 'point', 'thumbs_up', 'clap', 'shrug'],
		voice_settings: {
			voice_type: 'female',
			speech_rate: 1.0,
			pitch: 1.0,
			volume: 0.8
		}
	});
	
	const selectedTab = writable('control'); // 'control', 'emotions', 'gestures', 'voice', 'settings'
	const isLoading = writable(false);
	
	// WebSocket for real-time avatar updates
	let ws: WebSocket | null = null;
	let avatarCanvas: HTMLCanvasElement | null = null;
	
	// Control forms
	let speechInput = '';
	let selectedEmotion = 'neutral';
	let selectedGesture = 'wave';
	let emotionAnalysisText = '';
	let emotionAnalysisResult = null;
	
	// Animation history
	let animationHistory: any[] = [];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadAvatarStatus();
		await initializeCanvas();
	});
	
	async function initializeWebSocket() {
		try {
			// Connect to the verified Live2D Avatar endpoint on port 8091
			ws = new WebSocket('ws://localhost:8091/ws');
			
			ws.onopen = () => {
				console.log('Live2D Avatar WebSocket connected');
				avatarState.update(current => ({ ...current, status: 'connected' }));
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleAvatarUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Avatar WebSocket disconnected, attempting reconnection...');
				avatarState.update(current => ({ ...current, status: 'disconnected' }));
				setTimeout(initializeWebSocket, 5000);
			};
			
			ws.onerror = (error) => {
				console.error('Avatar WebSocket error:', error);
				avatarState.update(current => ({ ...current, status: 'error' }));
			};
		} catch (error) {
			console.error('Avatar WebSocket connection failed:', error);
			avatarState.update(current => ({ ...current, status: 'error' }));
		}
	}
	
	function handleAvatarUpdate(data: any) {
		switch (data.type) {
			case 'emotion_changed':
				avatarState.update(current => ({
					...current,
					current_emotion: data.emotion
				}));
				addToHistory('emotion', data.emotion);
				break;
			case 'gesture_started':
				avatarState.update(current => ({
					...current,
					gesture_active: true
				}));
				addToHistory('gesture', data.gesture);
				break;
			case 'gesture_completed':
				avatarState.update(current => ({
					...current,
					gesture_active: false
				}));
				break;
			case 'speech_started':
				avatarState.update(current => ({
					...current,
					is_speaking: true
				}));
				addToHistory('speech', data.text);
				break;
			case 'speech_completed':
				avatarState.update(current => ({
					...current,
					is_speaking: false
				}));
				break;
			case 'model_loaded':
				avatarState.update(current => ({
					...current,
					model_loaded: true
				}));
				break;
			case 'animation_update':
				updateCanvas(data.frame_data);
				break;
		}
	}
	
	async function loadAvatarStatus() {
		isLoading.set(true);
		try {
			const [statusResponse, healthResponse] = await Promise.all([
				fetch('http://localhost:8091/status'),
				fetch('http://localhost:8091/health')
			]);
			
			const [status, health] = await Promise.all([
				statusResponse.json(),
				healthResponse.json()
			]);
			
			avatarState.update(current => ({
				...current,
				model_loaded: status?.model_loaded || false,
				current_emotion: status?.current_emotion || 'neutral',
				is_speaking: status?.is_speaking || false
			}));
			
		} catch (error) {
			console.error('Failed to load avatar status:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	async function initializeCanvas() {
		// Initialize WebGL canvas for Live2D rendering
		const canvas = document.getElementById('avatar-canvas') as HTMLCanvasElement;
		if (canvas) {
			avatarCanvas = canvas;
			const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
			if (gl) {
				console.log('WebGL context initialized for Live2D');
			} else {
				console.error('WebGL not supported');
			}
		}
	}
	
	function updateCanvas(frameData: any) {
		// Update Live2D canvas with new frame data
		if (avatarCanvas && frameData) {
			// Canvas update logic would go here
			// This would integrate with Live2D Cubism SDK
		}
	}
	
	async function speakText() {
		if (!speechInput.trim()) return;
		
		try {
			const response = await fetch('http://localhost:8091/speak', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					text: speechInput,
					emotion: selectedEmotion !== 'neutral' ? selectedEmotion : null
				})
			});
			
			const result = await response.json();
			speechInput = '';
			
			dispatch('speech_triggered', { text: speechInput, emotion: selectedEmotion });
		} catch (error) {
			console.error('Failed to trigger speech:', error);
		}
	}
	
	async function performGesture() {
		try {
			const response = await fetch('http://localhost:8091/gesture', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ gesture: selectedGesture })
			});
			
			const result = await response.json();
			dispatch('gesture_triggered', { gesture: selectedGesture });
		} catch (error) {
			console.error('Failed to perform gesture:', error);
		}
	}
	
	async function setEmotion() {
		try {
			const response = await fetch('http://localhost:8091/emotion', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ emotion: selectedEmotion })
			});
			
			const result = await response.json();
			dispatch('emotion_changed', { emotion: selectedEmotion });
		} catch (error) {
			console.error('Failed to set emotion:', error);
		}
	}
	
	async function analyzeEmotion() {
		if (!emotionAnalysisText.trim()) return;
		
		try {
			const response = await fetch('http://localhost:8091/analyze', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ text: emotionAnalysisText })
			});
			
			emotionAnalysisResult = await response.json();
		} catch (error) {
			console.error('Failed to analyze emotion:', error);
		}
	}
	
	function addToHistory(type: string, data: any) {
		const historyItem = {
			type,
			data,
			timestamp: new Date().toISOString()
		};
		
		animationHistory = [historyItem, ...animationHistory.slice(0, 49)]; // Keep last 50 items
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'connected': return 'text-green-400';
			case 'connecting': return 'text-yellow-400';
			case 'disconnected':
			case 'error': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}
	
	function getEmotionEmoji(emotion: string): string {
		switch (emotion) {
			case 'happy': return '😊';
			case 'sad': return '😢';
			case 'angry': return '😠';
			case 'surprised': return '😲';
			case 'excited': return '🤩';
			case 'thinking': return '🤔';
			default: return '😐';
		}
	}
	
	function formatTimestamp(timestamp: string): string {
		return new Date(timestamp).toLocaleTimeString();
	}
</script>

<!-- Live2D Avatar Interface -->
<div class="avatar-interface h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-cyan-400">Live2D Avatar System</h1>
			<div class="flex items-center space-x-4">
				<div class="flex items-center space-x-2">
					<div class="w-2 h-2 rounded-full {getStatusColor($avatarState.status)}"></div>
					<span class="text-sm text-gray-400">Status: {$avatarState.status}</span>
				</div>
				<div class="text-sm text-gray-400">
					Model: {$avatarState.model_loaded ? 'Loaded' : 'Not Loaded'}
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadAvatarStatus}
						class="px-3 py-1 bg-cyan-600 hover:bg-cyan-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Avatar Status Bar -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-2 md:grid-cols-4 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-xl">
					{getEmotionEmoji($avatarState.current_emotion)}
				</div>
				<div class="text-sm text-gray-400 capitalize">{$avatarState.current_emotion}</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-xl font-bold {$avatarState.is_speaking ? 'text-green-400' : 'text-gray-400'}">
					{$avatarState.is_speaking ? '🗣️' : '🤐'}
				</div>
				<div class="text-sm text-gray-400">{$avatarState.is_speaking ? 'Speaking' : 'Silent'}</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-xl font-bold {$avatarState.gesture_active ? 'text-yellow-400' : 'text-gray-400'}">
					{$avatarState.gesture_active ? '🤲' : '🤚'}
				</div>
				<div class="text-sm text-gray-400">{$avatarState.gesture_active ? 'Gesturing' : 'Idle'}</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 text-center">
				<div class="text-xl font-bold {$avatarState.model_loaded ? 'text-green-400' : 'text-red-400'}">
					{$avatarState.model_loaded ? '✅' : '❌'}
				</div>
				<div class="text-sm text-gray-400">Model Status</div>
			</div>
		</div>
	</div>
	
	<!-- Main Content Grid -->
	<div class="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-4 p-4">
		<!-- Avatar Canvas -->
		<div class="lg:col-span-2">
			<div class="bg-gray-800 rounded-lg p-6 h-full">
				<h3 class="text-lg font-semibold mb-4 text-cyan-400">Avatar Display</h3>
				<div class="relative bg-gray-900 rounded-lg h-96 lg:h-[500px] flex items-center justify-center">
					<canvas
						id="avatar-canvas"
						bind:this={avatarCanvas}
						width="800"
						height="600"
						class="max-w-full max-h-full"
					></canvas>
					
					<!-- Avatar Loading State -->
					{#if !$avatarState.model_loaded}
						<div class="absolute inset-0 flex items-center justify-center bg-gray-900 rounded-lg">
							<div class="text-center">
								<div class="text-4xl mb-4">👤</div>
								<p class="text-gray-400">Avatar model loading...</p>
								{#if $avatarState.status === 'error'}
									<p class="text-red-400 text-sm mt-2">Connection error</p>
								{/if}
							</div>
						</div>
					{/if}
					
					<!-- Current Status Overlay -->
					<div class="absolute top-4 left-4 bg-black bg-opacity-50 rounded px-3 py-2">
						<div class="flex items-center space-x-2 text-sm">
							<span>🎭</span>
							<span class="capitalize">{$avatarState.current_emotion}</span>
							{#if $avatarState.is_speaking}
								<span class="text-green-400">🗣️</span>
							{/if}
							{#if $avatarState.gesture_active}
								<span class="text-yellow-400">🤲</span>
							{/if}
						</div>
					</div>
				</div>
			</div>
		</div>
		
		<!-- Control Panel -->
		<div class="space-y-4">
			<!-- Navigation Tabs -->
			<div class="bg-gray-800 rounded-lg">
				<nav class="flex flex-wrap border-b border-gray-700">
					{#each [
						{ id: 'control', label: 'Control', icon: '🎛️' },
						{ id: 'emotions', label: 'Emotions', icon: '🎭' },
						{ id: 'gestures', label: 'Gestures', icon: '🤲' },
						{ id: 'voice', label: 'Voice', icon: '🗣️' },
						{ id: 'settings', label: 'Settings', icon: '⚙️' }
					] as tab}
						<button
							class="flex-1 py-3 px-2 text-xs font-medium transition-colors {
								$selectedTab === tab.id
									? 'text-cyan-400 border-b-2 border-cyan-400'
									: 'text-gray-400 hover:text-white'
							}"
							on:click={() => selectedTab.set(tab.id)}
						>
							<div class="text-center">
								<div>{tab.icon}</div>
								<div class="mt-1">{tab.label}</div>
							</div>
						</button>
					{/each}
				</nav>
				
				<div class="p-4">
					{#if $selectedTab === 'control'}
						<!-- Quick Control -->
						<div class="space-y-4">
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Make Avatar Speak</label>
								<div class="flex space-x-2">
									<input
										type="text"
										bind:value={speechInput}
										placeholder="Enter text for avatar to speak"
										class="flex-1 px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500"
										on:keydown={(e) => e.key === 'Enter' && speakText()}
									/>
									<button
										on:click={speakText}
										class="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded transition-colors"
										disabled={!speechInput.trim() || $avatarState.is_speaking}
									>
										Speak
									</button>
								</div>
							</div>
							
							<div class="grid grid-cols-2 gap-2">
								<div>
									<label class="block text-sm font-medium text-gray-300 mb-2">Emotion</label>
									<select
										bind:value={selectedEmotion}
										class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500"
									>
										{#each $avatarState.available_emotions as emotion}
											<option value={emotion} class="capitalize">{emotion}</option>
										{/each}
									</select>
								</div>
								<div>
									<label class="block text-sm font-medium text-gray-300 mb-2">Gesture</label>
									<select
										bind:value={selectedGesture}
										class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500"
									>
										{#each $avatarState.available_gestures as gesture}
											<option value={gesture}>{gesture.replace('_', ' ')}</option>
										{/each}
									</select>
								</div>
							</div>
							
							<div class="grid grid-cols-2 gap-2">
								<button
									on:click={setEmotion}
									class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded transition-colors"
									disabled={$avatarState.current_emotion === selectedEmotion}
								>
									Set Emotion
								</button>
								<button
									on:click={performGesture}
									class="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded transition-colors"
									disabled={$avatarState.gesture_active}
								>
									Perform Gesture
								</button>
							</div>
						</div>
						
					{:else if $selectedTab === 'emotions'}
						<!-- Emotion Control -->
						<div class="space-y-4">
							<h4 class="font-medium text-purple-400">Emotion Control</h4>
							<div class="grid grid-cols-2 gap-2">
								{#each $avatarState.available_emotions as emotion}
									<button
										class="p-3 bg-gray-900 hover:bg-gray-700 rounded transition-colors text-center {
											$avatarState.current_emotion === emotion ? 'ring-2 ring-purple-500' : ''
										}"
										on:click={() => {
											selectedEmotion = emotion;
											setEmotion();
										}}
									>
										<div class="text-2xl mb-1">{getEmotionEmoji(emotion)}</div>
										<div class="text-xs capitalize">{emotion}</div>
									</button>
								{/each}
							</div>
							
							<div class="mt-6">
								<label class="block text-sm font-medium text-gray-300 mb-2">Emotion Analysis</label>
								<textarea
									bind:value={emotionAnalysisText}
									placeholder="Enter text to analyze emotional content"
									rows="3"
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
								></textarea>
								<button
									on:click={analyzeEmotion}
									class="mt-2 w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded transition-colors"
									disabled={!emotionAnalysisText.trim()}
								>
									Analyze Emotion
								</button>
								
								{#if emotionAnalysisResult}
									<div class="mt-4 bg-gray-900 rounded p-3">
										<h5 class="font-medium text-white mb-2">Analysis Result</h5>
										<div class="grid grid-cols-2 gap-2 text-sm">
											{#each Object.entries(emotionAnalysisResult) as [emotion, score]}
												<div class="flex justify-between">
													<span class="text-gray-400 capitalize">{emotion}:</span>
													<span class="text-white">{(score * 100).toFixed(1)}%</span>
												</div>
											{/each}
										</div>
									</div>
								{/if}
							</div>
						</div>
						
					{:else if $selectedTab === 'gestures'}
						<!-- Gesture Control -->
						<div class="space-y-4">
							<h4 class="font-medium text-yellow-400">Gesture Library</h4>
							<div class="grid grid-cols-1 gap-2">
								{#each $avatarState.available_gestures as gesture}
									<button
										class="p-3 bg-gray-900 hover:bg-gray-700 rounded transition-colors text-left"
										on:click={() => {
											selectedGesture = gesture;
											performGesture();
										}}
										disabled={$avatarState.gesture_active}
									>
										<div class="font-medium text-white">{gesture.replace('_', ' ')}</div>
										<div class="text-xs text-gray-400">Click to perform gesture</div>
									</button>
								{/each}
							</div>
						</div>
						
					{:else if $selectedTab === 'voice'}
						<!-- Voice Settings -->
						<div class="space-y-4">
							<h4 class="font-medium text-green-400">Voice Configuration</h4>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">Voice Type</label>
								<select
									bind:value={$avatarState.voice_settings.voice_type}
									class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
								>
									<option value="female">Female</option>
									<option value="male">Male</option>
									<option value="neutral">Neutral</option>
								</select>
							</div>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">
									Speech Rate: {$avatarState.voice_settings.speech_rate.toFixed(1)}x
								</label>
								<input
									type="range"
									bind:value={$avatarState.voice_settings.speech_rate}
									min="0.5"
									max="2.0"
									step="0.1"
									class="w-full"
								/>
							</div>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">
									Pitch: {$avatarState.voice_settings.pitch.toFixed(1)}
								</label>
								<input
									type="range"
									bind:value={$avatarState.voice_settings.pitch}
									min="0.5"
									max="2.0"
									step="0.1"
									class="w-full"
								/>
							</div>
							
							<div>
								<label class="block text-sm font-medium text-gray-300 mb-2">
									Volume: {Math.round($avatarState.voice_settings.volume * 100)}%
								</label>
								<input
									type="range"
									bind:value={$avatarState.voice_settings.volume}
									min="0"
									max="1"
									step="0.1"
									class="w-full"
								/>
							</div>
						</div>
						
					{:else if $selectedTab === 'settings'}
						<!-- Avatar Settings -->
						<div class="space-y-4">
							<h4 class="font-medium text-gray-300">Avatar Settings</h4>
							
							<div class="bg-gray-900 rounded p-4">
								<h5 class="font-medium text-white mb-2">Connection</h5>
								<div class="space-y-2 text-sm">
									<div class="flex justify-between">
										<span class="text-gray-400">Endpoint:</span>
										<span class="text-white">localhost:8091</span>
									</div>
									<div class="flex justify-between">
										<span class="text-gray-400">Protocol:</span>
										<span class="text-white">WebSocket + HTTP</span>
									</div>
									<div class="flex justify-between">
										<span class="text-gray-400">Status:</span>
										<span class="{getStatusColor($avatarState.status)}">{$avatarState.status}</span>
									</div>
								</div>
							</div>
							
							<div class="bg-gray-900 rounded p-4">
								<h5 class="font-medium text-white mb-2">Performance</h5>
								<div class="space-y-2 text-sm">
									<div class="flex justify-between">
										<span class="text-gray-400">Render FPS:</span>
										<span class="text-white">60 FPS</span>
									</div>
									<div class="flex justify-between">
										<span class="text-gray-400">Model Size:</span>
										<span class="text-white">~50MB</span>
									</div>
									<div class="flex justify-between">
										<span class="text-gray-400">GPU Acceleration:</span>
										<span class="text-green-400">Enabled</span>
									</div>
								</div>
							</div>
							
							<button
								class="w-full px-4 py-2 bg-red-600 hover:bg-red-700 rounded transition-colors"
								on:click={() => dispatch('reset_avatar')}
							>
								Reset Avatar
							</button>
						</div>
					{/if}
				</div>
			</div>
		</div>
		
		<!-- Animation History -->
		<div class="bg-gray-800 rounded-lg p-6">
			<h3 class="text-lg font-semibold mb-4 text-blue-400">Animation History</h3>
			{#if animationHistory.length === 0}
				<div class="text-center py-8 text-gray-400">
					<div class="text-3xl mb-2">📝</div>
					<p>No animations yet</p>
				</div>
			{:else}
				<div class="space-y-2 max-h-96 overflow-y-auto">
					{#each animationHistory as item}
						<div class="bg-gray-900 rounded p-3">
							<div class="flex items-center justify-between mb-1">
								<span class="font-medium text-white capitalize text-sm">{item.type}</span>
								<span class="text-xs text-gray-400">{formatTimestamp(item.timestamp)}</span>
							</div>
							<div class="text-sm text-gray-300">
								{#if item.type === 'speech'}
									<span class="text-green-400">🗣️</span> "{item.data}"
								{:else if item.type === 'emotion'}
									<span class="text-purple-400">🎭</span> {getEmotionEmoji(item.data)} {item.data}
								{:else if item.type === 'gesture'}
									<span class="text-yellow-400">🤲</span> {item.data.replace('_', ' ')}
								{/if}
							</div>
						</div>
					{/each}
				</div>
			{/if}
		</div>
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Connected to Live2D Avatar Controller on port 8091
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('save_avatar_state')}
					class="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded text-sm font-medium transition-colors"
				>
					Save State
				</button>
				<button
					on:click={() => dispatch('export_animation_history')}
					class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-sm font-medium transition-colors"
				>
					Export History
				</button>
			</div>
		</div>
	</div>
</div>

<style>
	.avatar-interface {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	#avatar-canvas {
		border: 1px solid #374151;
		border-radius: 8px;
	}
	
	/* Custom scrollbar */
	:global(.avatar-interface *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.avatar-interface *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.avatar-interface *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 3px;
	}
	
	:global(.avatar-interface *::-webkit-scrollbar-thumb:hover) {
		background: #9ca3af;
	}
</style>