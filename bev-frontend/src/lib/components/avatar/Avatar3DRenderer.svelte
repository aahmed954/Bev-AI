<!--
3D Avatar Renderer Component - Gaussian Splatting Integration
Features: WebGL rendering, real-time frame streaming, GPU acceleration for RTX 4090
Connects to: Advanced Avatar Service with Gaussian Splatting backend
-->

<script lang="ts">
	import { onMount, onDestroy, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	
	const dispatch = createEventDispatcher();
	
	// Avatar renderer state
	const rendererState = writable({
		status: 'initializing', // 'initializing', 'ready', 'rendering', 'error'
		fps: 0,
		frame_count: 0,
		gpu_memory_usage: 0,
		render_time_ms: 0,
		gaussian_splat_count: 0,
		current_emotion: 'neutral',
		animation_playing: false,
		webgl_context: null as WebGLRenderingContext | null,
		canvas_size: { width: 800, height: 600 }
	});
	
	// WebGL rendering components
	let canvas: HTMLCanvasElement;
	let gl: WebGLRenderingContext | null = null;
	let animationFrame: number | null = null;
	let shaderProgram: WebGLProgram | null = null;
	
	// Gaussian Splatting data
	let gaussianData: Float32Array | null = null;
	let positionBuffer: WebGLBuffer | null = null;
	let colorBuffer: WebGLBuffer | null = null;
	let rotationBuffer: WebGLBuffer | null = null;
	let scaleBuffer: WebGLBuffer | null = null;
	
	// Avatar streaming
	let frameStream: ReadableStream<Uint8Array> | null = null;
	let streamReader: ReadableStreamDefaultReader<Uint8Array> | null = null;
	
	// Performance monitoring
	let lastFrameTime = 0;
	let frameCount = 0;
	let fpsUpdateTime = 0;
	
	// Camera controls
	let cameraPosition = { x: 0, y: 0, z: 5 };
	let cameraRotation = { x: 0, y: 0, z: 0 };
	let zoom = 1.0;
	
	// Interaction state
	let isDragging = false;
	let lastMousePos = { x: 0, y: 0 };
	let isInteracting = false;
	
	export let avatarServiceUrl = 'http://localhost:8092'; // Advanced avatar service
	export let enableGPUAcceleration = true;
	export let enablePerformanceMonitoring = true;
	export let maxFPS = 60;
	export let qualityLevel = 'high'; // 'low', 'medium', 'high', 'ultra'
	
	onMount(async () => {
		await initializeWebGL();
		await loadShaders();
		await connectToAvatarService();
		startRenderLoop();
		
		if (enablePerformanceMonitoring) {
			startPerformanceMonitoring();
		}
	});
	
	onDestroy(() => {
		cleanup();
	});
	
	async function initializeWebGL() {
		try {
			gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
			
			if (!gl) {
				throw new Error('WebGL not supported');
			}
			
			// Enable required extensions for Gaussian Splatting
			const extensions = [
				'EXT_color_buffer_float',
				'OES_texture_float',
				'WEBGL_color_buffer_float',
				'EXT_float_blend'
			];
			
			for (const ext of extensions) {
				const extension = gl.getExtension(ext);
				if (!extension) {
					console.warn(`WebGL extension ${ext} not supported`);
				}
			}
			
			// Configure WebGL for high performance
			gl.enable(gl.DEPTH_TEST);
			gl.enable(gl.BLEND);
			gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
			gl.clearColor(0.0, 0.0, 0.0, 1.0);
			
			rendererState.update(state => ({
				...state,
				webgl_context: gl,
				status: 'ready'
			}));
			
			dispatch('webgl_initialized', { gl, canvas });
			
		} catch (error) {
			console.error('Failed to initialize WebGL:', error);
			rendererState.update(state => ({
				...state,
				status: 'error'
			}));
		}
	}
	
	async function loadShaders() {
		if (!gl) return;
		
		try {
			const vertexShaderSource = `
				attribute vec3 a_position;
				attribute vec4 a_color;
				attribute vec4 a_rotation; // quaternion
				attribute vec3 a_scale;
				
				uniform mat4 u_viewMatrix;
				uniform mat4 u_projectionMatrix;
				uniform vec3 u_cameraPosition;
				
				varying vec4 v_color;
				varying vec2 v_uv;
				
				// Quaternion rotation function
				vec3 rotateByQuaternion(vec3 v, vec4 q) {
					return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
				}
				
				void main() {
					// Apply scaling and rotation to create splat quad
					vec3 scaledPos = a_position * a_scale;
					vec3 rotatedPos = rotateByQuaternion(scaledPos, a_rotation);
					
					// Transform to world space
					vec4 worldPos = vec4(rotatedPos, 1.0);
					
					// Apply view and projection matrices
					gl_Position = u_projectionMatrix * u_viewMatrix * worldPos;
					
					v_color = a_color;
					v_uv = a_position.xy; // Use position as UV coordinates
				}
			`;
			
			const fragmentShaderSource = `
				precision highp float;
				
				varying vec4 v_color;
				varying vec2 v_uv;
				
				uniform float u_time;
				uniform vec3 u_lightDirection;
				
				void main() {
					// Gaussian splat rendering
					float dist = length(v_uv);
					float alpha = exp(-0.5 * dist * dist);
					
					// Apply lighting
					vec3 normal = normalize(vec3(v_uv, sqrt(1.0 - min(1.0, dist * dist))));
					float lighting = max(0.3, dot(normal, normalize(u_lightDirection)));
					
					vec3 finalColor = v_color.rgb * lighting;
					gl_FragColor = vec4(finalColor, v_color.a * alpha);
				}
			`;
			
			shaderProgram = createShaderProgram(vertexShaderSource, fragmentShaderSource);
			
			if (shaderProgram) {
				gl.useProgram(shaderProgram);
				dispatch('shaders_loaded');
			}
			
		} catch (error) {
			console.error('Failed to load shaders:', error);
		}
	}
	
	function createShaderProgram(vertexSource: string, fragmentSource: string): WebGLProgram | null {
		if (!gl) return null;
		
		const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
		const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
		
		if (!vertexShader || !fragmentShader) return null;
		
		const program = gl.createProgram();
		if (!program) return null;
		
		gl.attachShader(program, vertexShader);
		gl.attachShader(program, fragmentShader);
		gl.linkProgram(program);
		
		if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
			console.error('Shader program linking failed:', gl.getProgramInfoLog(program));
			return null;
		}
		
		return program;
	}
	
	function compileShader(type: number, source: string): WebGLShader | null {
		if (!gl) return null;
		
		const shader = gl.createShader(type);
		if (!shader) return null;
		
		gl.shaderSource(shader, source);
		gl.compileShader(shader);
		
		if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
			console.error('Shader compilation failed:', gl.getShaderInfoLog(shader));
			gl.deleteShader(shader);
			return null;
		}
		
		return shader;
	}
	
	async function connectToAvatarService() {
		try {
			// Check avatar service health
			const healthResponse = await fetch(`${avatarServiceUrl}/health`);
			if (!healthResponse.ok) {
				throw new Error('Avatar service not available');
			}
			
			// Initialize frame streaming
			const streamResponse = await fetch(`${avatarServiceUrl}/stream/frames`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({
					format: 'gaussian_splat',
					quality: qualityLevel,
					fps: maxFPS,
					gpu_acceleration: enableGPUAcceleration
				})
			});
			
			if (!streamResponse.ok) {
				throw new Error('Failed to initialize frame stream');
			}
			
			frameStream = streamResponse.body;
			if (frameStream) {
				streamReader = frameStream.getReader();
				startFrameProcessing();
			}
			
			dispatch('avatar_service_connected');
			
		} catch (error) {
			console.error('Failed to connect to avatar service:', error);
			rendererState.update(state => ({
				...state,
				status: 'error'
			}));
		}
	}
	
	async function startFrameProcessing() {
		if (!streamReader) return;
		
		try {
			while (true) {
				const { done, value } = await streamReader.read();
				
				if (done) {
					console.log('Frame stream ended');
					break;
				}
				
				if (value) {
					await processFrameData(value);
				}
			}
		} catch (error) {
			console.error('Frame processing error:', error);
		}
	}
	
	async function processFrameData(frameData: Uint8Array) {
		try {
			// Parse Gaussian splat data from frame
			const dataView = new DataView(frameData.buffer);
			let offset = 0;
			
			// Read header
			const splatCount = dataView.getUint32(offset, true);
			offset += 4;
			
			const emotion = new TextDecoder().decode(frameData.slice(offset, offset + 16)).replace(/\0/g, '');
			offset += 16;
			
			// Read Gaussian splat data
			const splatDataSize = splatCount * 32; // 32 bytes per splat (position, color, rotation, scale)
			const splatData = new Float32Array(frameData.buffer, offset, splatDataSize / 4);
			
			// Update GPU buffers
			await updateGaussianBuffers(splatData, splatCount);
			
			// Update state
			rendererState.update(state => ({
				...state,
				gaussian_splat_count: splatCount,
				current_emotion: emotion.trim(),
				frame_count: state.frame_count + 1
			}));
			
			dispatch('frame_received', { splatCount, emotion });
			
		} catch (error) {
			console.error('Failed to process frame data:', error);
		}
	}
	
	async function updateGaussianBuffers(data: Float32Array, count: number) {
		if (!gl) return;
		
		// Update position buffer
		if (!positionBuffer) {
			positionBuffer = gl.createBuffer();
		}
		gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, data.slice(0, count * 3), gl.DYNAMIC_DRAW);
		
		// Update color buffer
		if (!colorBuffer) {
			colorBuffer = gl.createBuffer();
		}
		gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, data.slice(count * 3, count * 7), gl.DYNAMIC_DRAW);
		
		// Update rotation buffer (quaternions)
		if (!rotationBuffer) {
			rotationBuffer = gl.createBuffer();
		}
		gl.bindBuffer(gl.ARRAY_BUFFER, rotationBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, data.slice(count * 7, count * 11), gl.DYNAMIC_DRAW);
		
		// Update scale buffer
		if (!scaleBuffer) {
			scaleBuffer = gl.createBuffer();
		}
		gl.bindBuffer(gl.ARRAY_BUFFER, scaleBuffer);
		gl.bufferData(gl.ARRAY_BUFFER, data.slice(count * 11, count * 14), gl.DYNAMIC_DRAW);
	}
	
	function startRenderLoop() {
		function render(currentTime: number) {
			if (!gl || !shaderProgram) {
				animationFrame = requestAnimationFrame(render);
				return;
			}
			
			const deltaTime = currentTime - lastFrameTime;
			lastFrameTime = currentTime;
			
			// Update FPS
			frameCount++;
			if (currentTime - fpsUpdateTime >= 1000) {
				const fps = Math.round((frameCount * 1000) / (currentTime - fpsUpdateTime));
				rendererState.update(state => ({
					...state,
					fps,
					render_time_ms: deltaTime
				}));
				frameCount = 0;
				fpsUpdateTime = currentTime;
			}
			
			// Clear canvas
			gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
			
			// Set up matrices
			const viewMatrix = createViewMatrix();
			const projectionMatrix = createProjectionMatrix();
			
			// Set uniforms
			const viewMatrixLocation = gl.getUniformLocation(shaderProgram, 'u_viewMatrix');
			const projectionMatrixLocation = gl.getUniformLocation(shaderProgram, 'u_projectionMatrix');
			const timeLocation = gl.getUniformLocation(shaderProgram, 'u_time');
			const lightDirectionLocation = gl.getUniformLocation(shaderProgram, 'u_lightDirection');
			
			gl.uniformMatrix4fv(viewMatrixLocation, false, viewMatrix);
			gl.uniformMatrix4fv(projectionMatrixLocation, false, projectionMatrix);
			gl.uniform1f(timeLocation, currentTime / 1000);
			gl.uniform3f(lightDirectionLocation, 0.5, 0.7, 1.0);
			
			// Render Gaussian splats
			renderGaussianSplats();
			
			animationFrame = requestAnimationFrame(render);
		}
		
		animationFrame = requestAnimationFrame(render);
	}
	
	function renderGaussianSplats() {
		if (!gl || !shaderProgram || !positionBuffer) return;
		
		const positionLocation = gl.getAttribLocation(shaderProgram, 'a_position');
		const colorLocation = gl.getAttribLocation(shaderProgram, 'a_color');
		const rotationLocation = gl.getAttribLocation(shaderProgram, 'a_rotation');
		const scaleLocation = gl.getAttribLocation(shaderProgram, 'a_scale');
		
		// Bind position buffer
		gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
		gl.enableVertexAttribArray(positionLocation);
		gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
		
		// Bind color buffer
		if (colorBuffer) {
			gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
			gl.enableVertexAttribArray(colorLocation);
			gl.vertexAttribPointer(colorLocation, 4, gl.FLOAT, false, 0, 0);
		}
		
		// Bind rotation buffer
		if (rotationBuffer) {
			gl.bindBuffer(gl.ARRAY_BUFFER, rotationBuffer);
			gl.enableVertexAttribArray(rotationLocation);
			gl.vertexAttribPointer(rotationLocation, 4, gl.FLOAT, false, 0, 0);
		}
		
		// Bind scale buffer
		if (scaleBuffer) {
			gl.bindBuffer(gl.ARRAY_BUFFER, scaleBuffer);
			gl.enableVertexAttribArray(scaleLocation);
			gl.vertexAttribPointer(scaleLocation, 3, gl.FLOAT, false, 0, 0);
		}
		
		// Draw splats as points
		const splatCount = $rendererState.gaussian_splat_count;
		if (splatCount > 0) {
			gl.drawArrays(gl.POINTS, 0, splatCount);
		}
	}
	
	function createViewMatrix(): Float32Array {
		// Create view matrix from camera position and rotation
		const matrix = new Float32Array(16);
		
		// Simple view matrix implementation
		matrix[0] = 1; matrix[1] = 0; matrix[2] = 0; matrix[3] = 0;
		matrix[4] = 0; matrix[5] = 1; matrix[6] = 0; matrix[7] = 0;
		matrix[8] = 0; matrix[9] = 0; matrix[10] = 1; matrix[11] = 0;
		matrix[12] = -cameraPosition.x;
		matrix[13] = -cameraPosition.y;
		matrix[14] = -cameraPosition.z;
		matrix[15] = 1;
		
		return matrix;
	}
	
	function createProjectionMatrix(): Float32Array {
		const matrix = new Float32Array(16);
		const aspect = canvas.width / canvas.height;
		const fov = Math.PI / 4; // 45 degrees
		const near = 0.1;
		const far = 100.0;
		
		const f = Math.tan(Math.PI * 0.5 - 0.5 * fov);
		const rangeInv = 1.0 / (near - far);
		
		matrix[0] = f / aspect;
		matrix[1] = 0;
		matrix[2] = 0;
		matrix[3] = 0;
		matrix[4] = 0;
		matrix[5] = f;
		matrix[6] = 0;
		matrix[7] = 0;
		matrix[8] = 0;
		matrix[9] = 0;
		matrix[10] = (near + far) * rangeInv;
		matrix[11] = -1;
		matrix[12] = 0;
		matrix[13] = 0;
		matrix[14] = near * far * rangeInv * 2;
		matrix[15] = 0;
		
		return matrix;
	}
	
	function startPerformanceMonitoring() {
		setInterval(async () => {
			try {
				const response = await fetch(`${avatarServiceUrl}/performance`);
				const perfData = await response.json();
				
				rendererState.update(state => ({
					...state,
					gpu_memory_usage: perfData.gpu_memory_mb || 0
				}));
				
				dispatch('performance_update', perfData);
			} catch (error) {
				console.warn('Performance monitoring failed:', error);
			}
		}, 1000);
	}
	
	// Mouse interaction handlers
	function handleMouseDown(event: MouseEvent) {
		isDragging = true;
		lastMousePos = { x: event.clientX, y: event.clientY };
		isInteracting = true;
		
		dispatch('interaction_start', { type: 'mouse_down', position: lastMousePos });
	}
	
	function handleMouseMove(event: MouseEvent) {
		if (!isDragging) return;
		
		const deltaX = event.clientX - lastMousePos.x;
		const deltaY = event.clientY - lastMousePos.y;
		
		// Update camera rotation
		cameraRotation.y += deltaX * 0.01;
		cameraRotation.x += deltaY * 0.01;
		
		lastMousePos = { x: event.clientX, y: event.clientY };
		
		dispatch('camera_move', { rotation: cameraRotation });
	}
	
	function handleMouseUp(event: MouseEvent) {
		isDragging = false;
		isInteracting = false;
		
		dispatch('interaction_end', { type: 'mouse_up' });
	}
	
	function handleWheel(event: WheelEvent) {
		event.preventDefault();
		
		zoom += event.deltaY * -0.001;
		zoom = Math.max(0.1, Math.min(5.0, zoom));
		
		cameraPosition.z = 5 / zoom;
		
		dispatch('zoom_change', { zoom, cameraPosition });
	}
	
	function cleanup() {
		if (animationFrame) {
			cancelAnimationFrame(animationFrame);
		}
		
		if (streamReader) {
			streamReader.cancel();
		}
		
		if (gl) {
			// Clean up WebGL resources
			if (positionBuffer) gl.deleteBuffer(positionBuffer);
			if (colorBuffer) gl.deleteBuffer(colorBuffer);
			if (rotationBuffer) gl.deleteBuffer(rotationBuffer);
			if (scaleBuffer) gl.deleteBuffer(scaleBuffer);
			if (shaderProgram) gl.deleteProgram(shaderProgram);
		}
	}
	
	// Public methods
	export async function setEmotion(emotion: string) {
		try {
			await fetch(`${avatarServiceUrl}/emotion`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ emotion })
			});
		} catch (error) {
			console.error('Failed to set emotion:', error);
		}
	}
	
	export async function speak(text: string, emotion?: string) {
		try {
			await fetch(`${avatarServiceUrl}/speak`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ text, emotion })
			});
		} catch (error) {
			console.error('Failed to trigger speech:', error);
		}
	}
	
	export function resetCamera() {
		cameraPosition = { x: 0, y: 0, z: 5 };
		cameraRotation = { x: 0, y: 0, z: 0 };
		zoom = 1.0;
	}
</script>

<!-- 3D Avatar Renderer -->
<div class="avatar-3d-renderer w-full h-full relative">
	<canvas
		bind:this={canvas}
		width={$rendererState.canvas_size.width}
		height={$rendererState.canvas_size.height}
		class="w-full h-full bg-gray-900 rounded-lg"
		on:mousedown={handleMouseDown}
		on:mousemove={handleMouseMove}
		on:mouseup={handleMouseUp}
		on:wheel={handleWheel}
		style="cursor: {isDragging ? 'grabbing' : (isInteracting ? 'grab' : 'default')}"
	/>
	
	<!-- Status Overlay -->
	<div class="absolute top-4 left-4 bg-black bg-opacity-75 rounded-lg p-3 text-white text-sm">
		<div class="flex items-center space-x-2 mb-2">
			<div class="w-2 h-2 rounded-full {
				$rendererState.status === 'ready' ? 'bg-green-400' :
				$rendererState.status === 'rendering' ? 'bg-blue-400' :
				$rendererState.status === 'error' ? 'bg-red-400' : 'bg-yellow-400'
			}"></div>
			<span class="font-medium capitalize">{$rendererState.status}</span>
		</div>
		
		<div class="space-y-1">
			<div class="flex justify-between">
				<span class="text-gray-300">FPS:</span>
				<span class="font-mono">{$rendererState.fps}</span>
			</div>
			<div class="flex justify-between">
				<span class="text-gray-300">Splats:</span>
				<span class="font-mono">{$rendererState.gaussian_splat_count.toLocaleString()}</span>
			</div>
			<div class="flex justify-between">
				<span class="text-gray-300">GPU:</span>
				<span class="font-mono">{$rendererState.gpu_memory_usage}MB</span>
			</div>
			<div class="flex justify-between">
				<span class="text-gray-300">Emotion:</span>
				<span class="capitalize">{$rendererState.current_emotion}</span>
			</div>
		</div>
	</div>
	
	<!-- Camera Controls -->
	<div class="absolute top-4 right-4 bg-black bg-opacity-75 rounded-lg p-3 text-white text-sm">
		<div class="text-center mb-2">
			<span class="font-medium">Camera</span>
		</div>
		
		<div class="space-y-2">
			<button
				on:click={resetCamera}
				class="w-full px-3 py-1 bg-cyan-600 hover:bg-cyan-700 rounded text-xs transition-colors"
			>
				Reset View
			</button>
			
			<div class="text-xs text-gray-300 text-center">
				Drag to rotate<br/>
				Scroll to zoom
			</div>
		</div>
	</div>
	
	<!-- Performance Monitor -->
	{#if enablePerformanceMonitoring}
		<div class="absolute bottom-4 left-4 bg-black bg-opacity-75 rounded-lg p-3 text-white text-xs">
			<div class="font-medium mb-2">Performance</div>
			<div class="space-y-1">
				<div class="flex justify-between">
					<span class="text-gray-300">Render Time:</span>
					<span class="font-mono">{$rendererState.render_time_ms.toFixed(1)}ms</span>
				</div>
				<div class="flex justify-between">
					<span class="text-gray-300">Frames:</span>
					<span class="font-mono">{$rendererState.frame_count.toLocaleString()}</span>
				</div>
				<div class="flex justify-between">
					<span class="text-gray-300">Zoom:</span>
					<span class="font-mono">{zoom.toFixed(2)}x</span>
				</div>
			</div>
		</div>
	{/if}
	
	<!-- Loading State -->
	{#if $rendererState.status === 'initializing'}
		<div class="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-90 rounded-lg">
			<div class="text-center text-white">
				<div class="w-12 h-12 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
				<p class="text-lg font-medium">Initializing 3D Avatar</p>
				<p class="text-sm text-gray-400">Setting up Gaussian Splatting renderer...</p>
			</div>
		</div>
	{/if}
	
	<!-- Error State -->
	{#if $rendererState.status === 'error'}
		<div class="absolute inset-0 flex items-center justify-center bg-gray-900 bg-opacity-90 rounded-lg">
			<div class="text-center text-white">
				<div class="text-4xl mb-4 text-red-400">⚠️</div>
				<p class="text-lg font-medium text-red-400">Avatar Rendering Error</p>
				<p class="text-sm text-gray-400">Check avatar service connection</p>
				<button
					on:click={() => window.location.reload()}
					class="mt-4 px-4 py-2 bg-red-600 hover:bg-red-700 rounded transition-colors"
				>
					Retry
				</button>
			</div>
		</div>
	{/if}
</div>

<style>
	.avatar-3d-renderer {
		position: relative;
		overflow: hidden;
	}
	
	canvas {
		display: block;
		image-rendering: pixelated;
		image-rendering: -moz-crisp-edges;
		image-rendering: crisp-edges;
	}
	
	/* Disable text selection during dragging */
	.avatar-3d-renderer.dragging {
		user-select: none;
		-webkit-user-select: none;
		-moz-user-select: none;
		-ms-user-select: none;
	}
</style>