<!--
Avatar Control Panel - Direct avatar interaction and control interface
Features: Emotion control, gesture commands, voice settings, interaction modes
Connected to: Avatar WebSocket Client and 3D renderer
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { avatarClient } from '$lib/services/AvatarWebSocketClient';
	
	const dispatch = createEventDispatcher();
	
	// Control state
	const controlState = writable({
		connection_status: 'disconnected',
		current_emotion: 'neutral',
		is_speaking: false,
		gesture_active: false,
		interaction_mode: 'manual', // 'manual', 'autonomous', 'osint_guided'
		voice_enabled: true,
		gesture_enabled: true,
		auto_responses: true
	});
	
	// Available emotions with enhanced set for OSINT context
	const emotions = [
		{ id: 'neutral', name: 'Neutral', icon: '😐', color: 'gray' },
		{ id: 'happy', name: 'Happy', icon: '😊', color: 'green' },
		{ id: 'excited', name: 'Excited', icon: '🤩', color: 'yellow' },
		{ id: 'focused', name: 'Focused', icon: '🧐', color: 'blue' },
		{ id: 'determined', name: 'Determined', icon: '😤', color: 'purple' },
		{ id: 'concerned', name: 'Concerned', icon: '😟', color: 'orange' },
		{ id: 'worried', name: 'Worried', icon: '😰', color: 'red' },
		{ id: 'alert', name: 'Alert', icon: '🚨', color: 'red' },
		{ id: 'satisfied', name: 'Satisfied', icon: '😌', color: 'green' },
		{ id: 'curious', name: 'Curious', icon: '🤔', color: 'cyan' },
		{ id: 'surprised', name: 'Surprised', icon: '😲', color: 'yellow' },
		{ id: 'thinking', name: 'Thinking', icon: '💭', color: 'indigo' }
	];
	
	// Available gestures
	const gestures = [
		{ id: 'wave', name: 'Wave', icon: '👋', description: 'Friendly greeting' },
		{ id: 'thumbs_up', name: 'Thumbs Up', icon: '👍', description: 'Approval or success' },
		{ id: 'thumbs_down', name: 'Thumbs Down', icon: '👎', description: 'Disapproval' },
		{ id: 'nod', name: 'Nod', icon: '✅', description: 'Agreement or acknowledgment' },
		{ id: 'shake_head', name: 'Shake Head', icon: '❌', description: 'Disagreement or negative' },
		{ id: 'point', name: 'Point', icon: '👉', description: 'Direct attention' },
		{ id: 'shrug', name: 'Shrug', icon: '🤷', description: 'Uncertainty or indifference' },
		{ id: 'clap', name: 'Clap', icon: '👏', description: 'Celebration or appreciation' },
		{ id: 'think', name: 'Think', icon: '🤔', description: 'Deep thought or analysis' },
		{ id: 'alert_pose', name: 'Alert Pose', icon: '⚠️', description: 'Warning or attention' }
	];
	
	// Interaction modes
	const interactionModes = [
		{ id: 'manual', name: 'Manual Control', description: 'Direct user control of avatar' },
		{ id: 'autonomous', name: 'Autonomous', description: 'Avatar responds automatically' },
		{ id: 'osint_guided', name: 'OSINT Guided', description: 'Avatar responds to investigation events' }
	];
	
	// Voice presets for different contexts
	const voicePresets = [
		{ id: 'professional', name: 'Professional', voice_type: 'female', pitch: 1.0, rate: 1.0 },
		{ id: 'friendly', name: 'Friendly', voice_type: 'female', pitch: 1.1, rate: 0.9 },
		{ id: 'serious', name: 'Serious', voice_type: 'male', pitch: 0.9, rate: 0.8 },
		{ id: 'energetic', name: 'Energetic', voice_type: 'female', pitch: 1.2, rate: 1.1 },
		{ id: 'calm', name: 'Calm', voice_type: 'neutral', pitch: 0.95, rate: 0.85 }
	];
	
	// UI state
	let selectedEmotion = 'neutral';
	let selectedGesture = 'wave';
	let speechText = '';
	let selectedVoicePreset = 'professional';
	let customVoiceSettings = {
		voice_type: 'female',
		pitch: 1.0,
		rate: 1.0,
		volume: 0.8
	};
	
	// Quick action buttons
	let quickActions = [
		{ id: 'greet', label: 'Greet User', action: () => greetUser() },
		{ id: 'status', label: 'Report Status', action: () => reportStatus() },
		{ id: 'celebrate', label: 'Celebrate', action: () => celebrate() },
		{ id: 'express_concern', label: 'Express Concern', action: () => expressConcern() },
		{ id: 'focus_mode', label: 'Focus Mode', action: () => setFocusMode() },
		{ id: 'reset', label: 'Reset to Neutral', action: () => resetToNeutral() }
	];
	
	// Performance metrics
	let performanceMetrics = writable({
		response_time_ms: 0,
		gesture_completion_time: 0,
		speech_duration: 0,
		emotion_transition_time: 0
	});
	
	onMount(async () => {
		await initializeAvatarConnection();
		setupEventListeners();
	});
	
	async function initializeAvatarConnection() {
		try {
			await avatarClient.connect();
			
			// Subscribe to avatar state changes
			avatarClient.subscribe('connection', (event) => {
				controlState.update(state => ({
					...state,
					connection_status: event.status
				}));
			});
			
			avatarClient.subscribe('emotion_change', (event) => {
				controlState.update(state => ({
					...state,
					current_emotion: event.data.emotion
				}));
				selectedEmotion = event.data.emotion;
			});
			
			avatarClient.subscribe('speech_request', (event) => {
				if (event.data.status === 'started') {
					controlState.update(state => ({ ...state, is_speaking: true }));
				} else if (event.data.status === 'completed') {
					controlState.update(state => ({ ...state, is_speaking: false }));
				}
			});
			
			avatarClient.subscribe('gesture_request', (event) => {
				if (event.data.status === 'started') {
					controlState.update(state => ({ ...state, gesture_active: true }));
				} else if (event.data.status === 'completed') {
					controlState.update(state => ({ ...state, gesture_active: false }));
				}
			});
			
		} catch (error) {
			console.error('Failed to initialize avatar connection:', error);
		}
	}
	
	function setupEventListeners() {
		// Listen for external avatar commands
		window.addEventListener('avatar_command', handleAvatarCommand);
		
		// Performance tracking
		const originalSetEmotion = avatarClient.setEmotion.bind(avatarClient);
		avatarClient.setEmotion = async (...args) => {
			const startTime = performance.now();
			await originalSetEmotion(...args);
			const endTime = performance.now();
			
			performanceMetrics.update(metrics => ({
				...metrics,
				emotion_transition_time: endTime - startTime
			}));
		};
	}
	
	function handleAvatarCommand(event: CustomEvent) {
		const { command, data } = event.detail;
		
		switch (command) {
			case 'set_emotion':
				setEmotion(data.emotion);
				break;
			case 'speak':
				speak(data.text, data.emotion);
				break;
			case 'gesture':
				performGesture(data.gesture);
				break;
			case 'set_mode':
				setInteractionMode(data.mode);
				break;
		}
	}
	
	async function setEmotion(emotion: string, context?: any) {
		try {
			await avatarClient.setEmotion(emotion, context);
			selectedEmotion = emotion;
			
			dispatch('emotion_changed', { emotion, context });
		} catch (error) {
			console.error('Failed to set emotion:', error);
		}
	}
	
	async function performGesture(gesture: string, intensity: number = 1.0) {
		try {
			const startTime = performance.now();
			await avatarClient.performGesture(gesture, intensity);
			
			// Track gesture completion time
			setTimeout(() => {
				const endTime = performance.now();
				performanceMetrics.update(metrics => ({
					...metrics,
					gesture_completion_time: endTime - startTime
				}));
			}, 2000); // Assume 2 second gesture duration
			
			dispatch('gesture_performed', { gesture, intensity });
		} catch (error) {
			console.error('Failed to perform gesture:', error);
		}
	}
	
	async function speak(text: string, emotion?: string, priority: 'low' | 'normal' | 'high' = 'normal') {
		if (!text.trim()) return;
		
		try {
			const startTime = performance.now();
			await avatarClient.speak(text, emotion || selectedEmotion, priority);
			
			// Estimate speech duration
			const estimatedDuration = text.length * 50; // ~50ms per character
			performanceMetrics.update(metrics => ({
				...metrics,
				speech_duration: estimatedDuration
			}));
			
			dispatch('speech_triggered', { text, emotion, priority });
		} catch (error) {
			console.error('Failed to trigger speech:', error);
		}
	}
	
	function setInteractionMode(mode: string) {
		controlState.update(state => ({
			...state,
			interaction_mode: mode
		}));
		
		dispatch('interaction_mode_changed', { mode });
	}
	
	function applyVoicePreset(presetId: string) {
		const preset = voicePresets.find(p => p.id === presetId);
		if (preset) {
			customVoiceSettings = {
				voice_type: preset.voice_type,
				pitch: preset.pitch,
				rate: preset.rate,
				volume: customVoiceSettings.volume
			};
			selectedVoicePreset = presetId;
		}
	}
	
	// Quick action functions
	async function greetUser() {
		await setEmotion('happy');
		await performGesture('wave');
		await speak('Hello! I\'m ready to assist with your OSINT investigation.', 'happy');
	}
	
	async function reportStatus() {
		const state = avatarClient.getState();
		await setEmotion('focused');
		await speak(`System status: Connected and operational. Current mode: ${state.interaction_mode}`, 'focused');
	}
	
	async function celebrate() {
		await setEmotion('excited');
		await performGesture('clap');
		await speak('Excellent work! Investigation completed successfully!', 'excited');
	}
	
	async function expressConcern() {
		await setEmotion('concerned');
		await performGesture('alert_pose');
		await speak('I\'ve detected something that requires your attention.', 'concerned');
	}
	
	async function setFocusMode() {
		await setEmotion('focused');
		await speak('Entering focus mode for deep analysis.', 'focused');
		setInteractionMode('osint_guided');
	}
	
	async function resetToNeutral() {
		await setEmotion('neutral');
		selectedEmotion = 'neutral';
		speechText = '';
		setInteractionMode('manual');
	}
	
	function getEmotionColor(emotion: string): string {
		const emotionObj = emotions.find(e => e.id === emotion);
		return emotionObj?.color || 'gray';
	}
	
	function getConnectionStatusColor(status: string): string {
		switch (status) {
			case 'connected': return 'text-green-400';
			case 'connecting': return 'text-yellow-400';
			case 'disconnected': 
			case 'error': return 'text-red-400';
			default: return 'text-gray-400';
		}
	}
</script>

<!-- Avatar Control Panel -->
<div class="avatar-control-panel h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h2 class="text-lg font-bold text-cyan-400">Avatar Control</h2>
			<div class="flex items-center space-x-4">
				<div class="flex items-center space-x-2">
					<div class="w-2 h-2 rounded-full {getConnectionStatusColor($controlState.connection_status)}"></div>
					<span class="text-xs text-gray-400 capitalize">{$controlState.connection_status}</span>
				</div>
				<div class="text-xs text-gray-400">
					Mode: <span class="text-cyan-400 capitalize">{$controlState.interaction_mode}</span>
				</div>
			</div>
		</div>
	</div>
	
	<!-- Control Content -->
	<div class="flex-1 overflow-y-auto p-4 space-y-6">
		<!-- Quick Actions -->
		<div>
			<h3 class="font-medium text-white mb-3">Quick Actions</h3>
			<div class="grid grid-cols-2 gap-2">
				{#each quickActions as action}
					<button
						on:click={action.action}
						class="px-3 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded text-sm transition-colors"
						disabled={$controlState.connection_status !== 'connected'}
					>
						{action.label}
					</button>
				{/each}
			</div>
		</div>
		
		<!-- Emotion Control -->
		<div>
			<h3 class="font-medium text-white mb-3">Emotion Control</h3>
			<div class="grid grid-cols-3 gap-2">
				{#each emotions as emotion}
					<button
						class="p-3 bg-gray-800 hover:bg-gray-700 border rounded transition-colors text-center {
							selectedEmotion === emotion.id ? `border-${emotion.color}-500` : 'border-gray-700'
						}"
						on:click={() => setEmotion(emotion.id)}
						disabled={$controlState.connection_status !== 'connected'}
					>
						<div class="text-xl mb-1">{emotion.icon}</div>
						<div class="text-xs text-gray-300">{emotion.name}</div>
					</button>
				{/each}
			</div>
		</div>
		
		<!-- Speech Control -->
		<div>
			<h3 class="font-medium text-white mb-3">Speech Control</h3>
			<div class="space-y-3">
				<textarea
					bind:value={speechText}
					placeholder="Enter text for avatar to speak..."
					rows="3"
					class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500 resize-none"
					disabled={$controlState.connection_status !== 'connected'}
				></textarea>
				
				<div class="flex space-x-2">
					<button
						on:click={() => speak(speechText)}
						disabled={!speechText.trim() || $controlState.is_speaking || $controlState.connection_status !== 'connected'}
						class="flex-1 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 disabled:bg-gray-700 disabled:cursor-not-allowed rounded font-medium transition-colors"
					>
						{$controlState.is_speaking ? 'Speaking...' : 'Speak'}
					</button>
					
					<button
						on:click={() => speechText = ''}
						class="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
					>
						Clear
					</button>
				</div>
			</div>
		</div>
		
		<!-- Gesture Control -->
		<div>
			<h3 class="font-medium text-white mb-3">Gesture Control</h3>
			<div class="space-y-2">
				{#each gestures as gesture}
					<button
						class="w-full p-3 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded transition-colors text-left"
						on:click={() => performGesture(gesture.id)}
						disabled={$controlState.gesture_active || $controlState.connection_status !== 'connected'}
					>
						<div class="flex items-center space-x-3">
							<span class="text-xl">{gesture.icon}</span>
							<div class="flex-1">
								<div class="font-medium text-white">{gesture.name}</div>
								<div class="text-xs text-gray-400">{gesture.description}</div>
							</div>
						</div>
					</button>
				{/each}
			</div>
		</div>
		
		<!-- Voice Settings -->
		<div>
			<h3 class="font-medium text-white mb-3">Voice Settings</h3>
			<div class="space-y-3">
				<!-- Voice presets -->
				<div>
					<label class="block text-sm font-medium text-gray-300 mb-2">Voice Preset</label>
					<select
						bind:value={selectedVoicePreset}
						on:change={() => applyVoicePreset(selectedVoicePreset)}
						class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500"
					>
						{#each voicePresets as preset}
							<option value={preset.id}>{preset.name}</option>
						{/each}
					</select>
				</div>
				
				<!-- Custom voice settings -->
				<div class="grid grid-cols-1 gap-3">
					<div>
						<label class="block text-sm font-medium text-gray-300 mb-1">
							Voice Type: {customVoiceSettings.voice_type}
						</label>
						<select
							bind:value={customVoiceSettings.voice_type}
							class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-cyan-500"
						>
							<option value="female">Female</option>
							<option value="male">Male</option>
							<option value="neutral">Neutral</option>
						</select>
					</div>
					
					<div>
						<label class="block text-sm font-medium text-gray-300 mb-1">
							Pitch: {customVoiceSettings.pitch.toFixed(1)}
						</label>
						<input
							type="range"
							bind:value={customVoiceSettings.pitch}
							min="0.5"
							max="2.0"
							step="0.1"
							class="w-full"
						/>
					</div>
					
					<div>
						<label class="block text-sm font-medium text-gray-300 mb-1">
							Speed: {customVoiceSettings.rate.toFixed(1)}x
						</label>
						<input
							type="range"
							bind:value={customVoiceSettings.rate}
							min="0.5"
							max="2.0"
							step="0.1"
							class="w-full"
						/>
					</div>
					
					<div>
						<label class="block text-sm font-medium text-gray-300 mb-1">
							Volume: {Math.round(customVoiceSettings.volume * 100)}%
						</label>
						<input
							type="range"
							bind:value={customVoiceSettings.volume}
							min="0"
							max="1"
							step="0.1"
							class="w-full"
						/>
					</div>
				</div>
			</div>
		</div>
		
		<!-- Interaction Mode -->
		<div>
			<h3 class="font-medium text-white mb-3">Interaction Mode</h3>
			<div class="space-y-2">
				{#each interactionModes as mode}
					<label class="flex items-center p-3 bg-gray-800 border border-gray-700 rounded cursor-pointer hover:bg-gray-700 transition-colors">
						<input
							type="radio"
							bind:group={$controlState.interaction_mode}
							value={mode.id}
							on:change={() => setInteractionMode(mode.id)}
							class="mr-3 w-4 h-4 text-cyan-600 bg-gray-700 border-gray-600 focus:ring-cyan-500"
						/>
						<div class="flex-1">
							<div class="font-medium text-white">{mode.name}</div>
							<div class="text-xs text-gray-400">{mode.description}</div>
						</div>
					</label>
				{/each}
			</div>
		</div>
		
		<!-- Settings Toggles -->
		<div>
			<h3 class="font-medium text-white mb-3">Settings</h3>
			<div class="space-y-3">
				<label class="flex items-center justify-between p-3 bg-gray-800 rounded">
					<span class="text-white">Voice Enabled</span>
					<input
						type="checkbox"
						bind:checked={$controlState.voice_enabled}
						class="w-4 h-4 text-cyan-600 bg-gray-700 border-gray-600 rounded focus:ring-cyan-500"
					/>
				</label>
				
				<label class="flex items-center justify-between p-3 bg-gray-800 rounded">
					<span class="text-white">Gestures Enabled</span>
					<input
						type="checkbox"
						bind:checked={$controlState.gesture_enabled}
						class="w-4 h-4 text-cyan-600 bg-gray-700 border-gray-600 rounded focus:ring-cyan-500"
					/>
				</label>
				
				<label class="flex items-center justify-between p-3 bg-gray-800 rounded">
					<span class="text-white">Auto Responses</span>
					<input
						type="checkbox"
						bind:checked={$controlState.auto_responses}
						class="w-4 h-4 text-cyan-600 bg-gray-700 border-gray-600 rounded focus:ring-cyan-500"
					/>
				</label>
			</div>
		</div>
	</div>
	
	<!-- Performance Footer -->
	<div class="border-t border-gray-800 p-3">
		<div class="text-xs text-gray-400">
			<div class="flex justify-between">
				<span>Response Time:</span>
				<span>{$performanceMetrics.response_time_ms.toFixed(0)}ms</span>
			</div>
			<div class="flex justify-between">
				<span>Emotion Transition:</span>
				<span>{$performanceMetrics.emotion_transition_time.toFixed(0)}ms</span>
			</div>
		</div>
	</div>
</div>

<style>
	.avatar-control-panel {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Custom scrollbar */
	:global(.avatar-control-panel *::-webkit-scrollbar) {
		width: 4px;
	}
	
	:global(.avatar-control-panel *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.avatar-control-panel *::-webkit-scrollbar-thumb) {
		background: #6b7280;
		border-radius: 2px;
	}
</style>