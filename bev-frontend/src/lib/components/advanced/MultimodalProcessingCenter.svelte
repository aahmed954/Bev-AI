<!--
Advanced Multimodal Processing Center - Multi-Format Content Analysis
Connected to: src/advanced/multimodal_processor.py
Features: Image/video/audio analysis, content understanding, cross-modal correlation
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const multimodalState = writable({
		processors: {
			image: { status: 'active', processed: 0, models_loaded: 3 },
			video: { status: 'active', processed: 0, models_loaded: 2 },
			audio: { status: 'active', processed: 0, models_loaded: 2 },
			text: { status: 'active', processed: 0, models_loaded: 4 }
		},
		active_jobs: [],
		completed_analyses: [],
		model_performance: {
			image_accuracy: 94.2,
			video_accuracy: 91.7,
			audio_accuracy: 88.9,
			text_accuracy: 96.1
		},
		cross_modal_correlations: []
	});
	
	const selectedTab = writable('overview'); // 'overview', 'upload', 'analysis', 'correlations', 'models'
	const selectedJob = writable(null);
	const isLoading = writable(false);
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	
	// Upload and analysis data
	let uploadQueue: any[] = [];
	let analysisResults: any[] = [];
	
	// New analysis form
	let newAnalysis = {
		files: [],
		analysis_types: ['object_detection', 'text_extraction', 'content_understanding'],
		enable_correlations: true,
		output_format: 'comprehensive',
		privacy_mode: true
	};
	
	const analysisTypes = [
		{ id: 'object_detection', name: 'Object Detection', modality: 'image', icon: '👁️' },
		{ id: 'face_recognition', name: 'Face Recognition', modality: 'image', icon: '😊' },
		{ id: 'text_extraction', name: 'Text Extraction (OCR)', modality: 'image', icon: '📝' },
		{ id: 'scene_understanding', name: 'Scene Understanding', modality: 'image', icon: '🏞️' },
		{ id: 'video_analysis', name: 'Video Content Analysis', modality: 'video', icon: '🎬' },
		{ id: 'action_recognition', name: 'Action Recognition', modality: 'video', icon: '🏃' },
		{ id: 'speech_recognition', name: 'Speech to Text', modality: 'audio', icon: '🎤' },
		{ id: 'speaker_identification', name: 'Speaker ID', modality: 'audio', icon: '👤' },
		{ id: 'sentiment_analysis', name: 'Sentiment Analysis', modality: 'text', icon: '😊' },
		{ id: 'entity_extraction', name: 'Entity Extraction', modality: 'text', icon: '🏷️' },
		{ id: 'content_understanding', name: 'Content Understanding', modality: 'multimodal', icon: '🧠' },
		{ id: 'cross_modal_search', name: 'Cross-Modal Search', modality: 'multimodal', icon: '🔍' }
	];
	
	const processingModels = {
		image: ['YOLO v8', 'ResNet-50', 'CLIP'],
		video: ['SlowFast', 'I3D'],
		audio: ['Wav2Vec2', 'Whisper'],
		text: ['BERT', 'GPT-4', 'RoBERTa', 'T5']
	};
	
	onMount(async () => {
		await initializeWebSocket();
		await loadMultimodalSystem();
		startProcessingMonitoring();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/multimodal-processing');
			
			ws.onopen = () => {
				console.log('Multimodal Processing WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleMultimodalUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Multimodal WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Multimodal WebSocket connection failed:', error);
		}
	}
	
	function handleMultimodalUpdate(data: any) {
		switch (data.type) {
			case 'job_started':
				addProcessingJob(data.job);
				break;
			case 'job_progress':
				updateJobProgress(data.job_id, data.progress);
				break;
			case 'job_completed':
				completeJob(data.job_id, data.results);
				break;
			case 'correlation_found':
				addCorrelation(data.correlation);
				break;
			case 'model_metrics':
				updateModelMetrics(data.modality, data.metrics);
				break;
		}
	}
	
	async function loadMultimodalSystem() {
		isLoading.set(true);
		try {
			const [jobs, results, metrics, correlations] = await Promise.all([
				invoke('get_active_multimodal_jobs'),
				invoke('get_multimodal_results'),
				invoke('get_multimodal_metrics'),
				invoke('get_cross_modal_correlations')
			]);
			
			multimodalState.update(current => ({
				...current,
				active_jobs: jobs || [],
				completed_analyses: results || [],
				cross_modal_correlations: correlations || []
			}));
			
			if (metrics) {
				multimodalState.update(current => ({
					...current,
					processors: { ...current.processors, ...metrics.processors },
					model_performance: { ...current.model_performance, ...metrics.performance }
				}));
			}
		} catch (error) {
			console.error('Failed to load multimodal system:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function startProcessingMonitoring() {
		setInterval(async () => {
			try {
				const metrics = await invoke('get_processing_metrics');
				updateProcessingMetrics(metrics);
			} catch (error) {
				console.error('Processing monitoring failed:', error);
			}
		}, 5000);
	}
	
	async function startMultimodalAnalysis() {
		try {
			const result = await invoke('start_multimodal_analysis', { analysis: newAnalysis });
			
			addProcessingJob({
				id: result.job_id,
				...newAnalysis,
				status: 'processing',
				started_at: new Date().toISOString(),
				progress: 0
			});
			
			// Reset form
			newAnalysis = {
				files: [],
				analysis_types: ['object_detection', 'text_extraction', 'content_understanding'],
				enable_correlations: true,
				output_format: 'comprehensive',
				privacy_mode: true
			};
			
			dispatch('analysis_started', result);
		} catch (error) {
			console.error('Failed to start multimodal analysis:', error);
		}
	}
	
	function addProcessingJob(job: any) {
		multimodalState.update(current => ({
			...current,
			active_jobs: [job, ...current.active_jobs]
		}));
	}
	
	function updateJobProgress(jobId: string, progress: number) {
		multimodalState.update(current => ({
			...current,
			active_jobs: current.active_jobs.map(job =>
				job.id === jobId ? { ...job, progress } : job
			)
		}));
	}
	
	function completeJob(jobId: string, results: any) {
		const job = $multimodalState.active_jobs.find(j => j.id === jobId);
		if (job) {
			multimodalState.update(current => ({
				...current,
				active_jobs: current.active_jobs.filter(j => j.id !== jobId),
				completed_analyses: [{ ...job, results, completed_at: new Date().toISOString() }, ...current.completed_analyses.slice(0, 49)]
			}));
		}
	}
	
	function addCorrelation(correlation: any) {
		multimodalState.update(current => ({
			...current,
			cross_modal_correlations: [correlation, ...current.cross_modal_correlations.slice(0, 19)]
		}));
	}
	
	function updateModelMetrics(modality: string, metrics: any) {
		multimodalState.update(current => ({
			...current,
			processors: {
				...current.processors,
				[modality]: { ...current.processors[modality], ...metrics }
			}
		}));
	}
	
	function updateProcessingMetrics(metrics: any) {
		multimodalState.update(current => ({
			...current,
			model_performance: { ...current.model_performance, ...metrics.performance },
			processors: { ...current.processors, ...metrics.processors }
		}));
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'active':
			case 'processing':
			case 'completed': return 'text-green-400';
			case 'queued':
			case 'loading': return 'text-yellow-400';
			case 'failed':
			case 'error': return 'text-red-400';
			case 'idle':
			case 'paused': return 'text-gray-400';
			default: return 'text-gray-400';
		}
	}
	
	function getModalityColor(modality: string): string {
		switch (modality) {
			case 'image': return 'text-blue-400';
			case 'video': return 'text-purple-400';
			case 'audio': return 'text-green-400';
			case 'text': return 'text-yellow-400';
			case 'multimodal': return 'text-pink-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleString();
	}
	
	function openJobModal(job: any) {
		selectedJob.set(job);
	}
</script>

<!-- Multimodal Processing Center -->
<div class="multimodal-center h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-pink-400">🎭 Multimodal Processing Center</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{$multimodalState.active_jobs.length} active jobs
				</div>
				<div class="text-sm text-gray-400">
					{Object.values($multimodalState.processors).filter(p => p.status === 'active').length}/4 processors online
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-pink-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadMultimodalSystem}
						class="px-3 py-1 bg-pink-600 hover:bg-pink-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Processor Status -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-1 md:grid-cols-4 gap-4">
			{#each Object.entries($multimodalState.processors) as [modality, processor]}
				<div class="bg-gray-800 rounded-lg p-4 text-center">
					<div class="text-xl mb-2">
						{modality === 'image' ? '🖼️' : 
						 modality === 'video' ? '🎬' : 
						 modality === 'audio' ? '🎵' : '📝'}
					</div>
					<div class="font-medium text-white capitalize">{modality}</div>
					<div class="flex items-center justify-center mt-2">
						<div class="w-2 h-2 rounded-full {getStatusColor(processor.status)} mr-1"></div>
						<span class="text-xs {getStatusColor(processor.status)}">{processor.status}</span>
					</div>
					<div class="text-xs text-gray-400 mt-1">
						{processor.processed} processed | {processor.models_loaded} models
					</div>
				</div>
			{/each}
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Processing Overview', icon: '📊' },
				{ id: 'upload', label: 'Content Upload', icon: '📤' },
				{ id: 'analysis', label: 'Active Analysis', icon: '🔍' },
				{ id: 'correlations', label: 'Cross-Modal', icon: '🔗' },
				{ id: 'models', label: 'Model Management', icon: '🧠' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedTab === tab.id
							? 'border-pink-500 text-pink-400'
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
			<!-- Multimodal Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Processing Statistics -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-pink-400">Processing Statistics</h3>
					<div class="space-y-4">
						<div class="grid grid-cols-2 gap-4 text-center">
							{#each Object.entries($multimodalState.processors) as [modality, processor]}
								<div class="bg-gray-900 rounded p-4">
									<div class="text-xl font-bold {getModalityColor(modality)}">{formatNumber(processor.processed)}</div>
									<div class="text-xs text-gray-400 capitalize">{modality} Files</div>
								</div>
							{/each}
						</div>
						
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">Model Accuracy</h4>
							<div class="space-y-2 text-sm">
								{#each Object.entries($multimodalState.model_performance) as [model, accuracy]}
									<div class="flex justify-between">
										<span class="text-gray-400 capitalize">{model.replace('_accuracy', '')}:</span>
										<span class="text-white">{accuracy.toFixed(1)}%</span>
									</div>
								{/each}
							</div>
						</div>
					</div>
				</div>
				
				<!-- Quick Upload Interface -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Quick Analysis</h3>
					<div class="space-y-4">
						<!-- File Upload -->
						<div class="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center">
							<input
								type="file"
								multiple
								accept="image/*,video/*,audio/*,.txt,.pdf,.doc,.docx"
								class="hidden"
								id="file-upload"
							/>
							<label for="file-upload" class="cursor-pointer">
								<div class="text-4xl mb-2">📎</div>
								<p class="text-gray-300">Drop files here or click to upload</p>
								<p class="text-sm text-gray-400 mt-1">Images, videos, audio, documents</p>
							</label>
						</div>
						
						<!-- Analysis Type Selection -->
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Analysis Types</label>
							<div class="grid grid-cols-2 gap-2 max-h-32 overflow-y-auto">
								{#each analysisTypes as type}
									<label class="flex items-center text-sm">
										<input
											type="checkbox"
											bind:group={newAnalysis.analysis_types}
											value={type.id}
											class="mr-2"
										/>
										<span class="{getModalityColor(type.modality)}">{type.icon}</span>
										<span class="text-white ml-1">{type.name}</span>
									</label>
								{/each}
							</div>
						</div>
						
						<div class="flex items-center space-x-4">
							<label class="flex items-center text-sm">
								<input
									type="checkbox"
									bind:checked={newAnalysis.enable_correlations}
									class="mr-2"
								/>
								<span class="text-gray-300">Enable cross-modal correlations</span>
							</label>
							<label class="flex items-center text-sm">
								<input
									type="checkbox"
									bind:checked={newAnalysis.privacy_mode}
									class="mr-2"
								/>
								<span class="text-gray-300">Privacy mode</span>
							</label>
						</div>
						
						<button
							on:click={startMultimodalAnalysis}
							class="w-full px-4 py-2 bg-pink-600 hover:bg-pink-700 rounded font-medium transition-colors"
							disabled={newAnalysis.analysis_types.length === 0}
						>
							Start Analysis
						</button>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'analysis'}
			<!-- Active Analysis Jobs -->
			<div class="space-y-4">
				{#if $multimodalState.active_jobs.length === 0}
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">🔍</div>
						<p>No active analysis jobs</p>
					</div>
				{:else}
					{#each $multimodalState.active_jobs as job}
						<div class="bg-gray-800 rounded-lg p-6">
							<div class="flex items-center justify-between mb-4">
								<div class="flex items-center space-x-4">
									<h3 class="text-lg font-medium text-white">Analysis Job #{job.id}</h3>
									<span class="px-2 py-1 rounded text-xs {getStatusColor(job.status)} bg-gray-700">
										{job.status}
									</span>
									<span class="text-sm text-gray-400">
										{job.analysis_types.length} analysis types
									</span>
								</div>
								<div class="text-sm text-gray-400">
									{formatDate(job.started_at)}
								</div>
							</div>
							
							{#if job.progress !== undefined}
								<div class="mb-4">
									<div class="flex justify-between text-sm mb-1">
										<span class="text-gray-400">Progress</span>
										<span class="text-white">{job.progress}%</span>
									</div>
									<div class="w-full bg-gray-700 rounded-full h-2">
										<div
											class="bg-pink-600 h-2 rounded-full transition-all duration-300"
											style="width: {job.progress}%"
										></div>
									</div>
								</div>
							{/if}
							
							<div class="flex flex-wrap gap-2 mb-4">
								{#each job.analysis_types as analysisType}
									{@const typeInfo = analysisTypes.find(t => t.id === analysisType)}
									<span class="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded">
										{typeInfo?.icon} {typeInfo?.name || analysisType}
									</span>
								{/each}
							</div>
							
							<div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
								<div>
									<span class="text-gray-400">Files:</span>
									<span class="text-white ml-2">{job.files?.length || 0}</span>
								</div>
								<div>
									<span class="text-gray-400">Format:</span>
									<span class="text-white ml-2">{job.output_format}</span>
								</div>
								{#if job.enable_correlations}
									<div>
										<span class="text-gray-400">Correlations:</span>
										<span class="text-pink-400 ml-2">Enabled</span>
									</div>
								{/if}
								{#if job.privacy_mode}
									<div>
										<span class="text-gray-400">Privacy:</span>
										<span class="text-green-400 ml-2">Protected</span>
									</div>
								{/if}
							</div>
						</div>
					{/each}
				{/if}
			</div>
			
		{:else if $selectedTab === 'correlations'}
			<!-- Cross-Modal Correlations -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-pink-400">Cross-Modal Correlations</h3>
				{#if $multimodalState.cross_modal_correlations.length === 0}
					<div class="text-center py-12 text-gray-400">
						<div class="text-4xl mb-4">🔗</div>
						<p>No cross-modal correlations found</p>
					</div>
				{:else}
					<div class="space-y-3">
						{#each $multimodalState.cross_modal_correlations as correlation}
							<div class="bg-gray-900 rounded p-4">
								<div class="flex items-center justify-between mb-2">
									<span class="font-medium text-white">{correlation.type}</span>
									<span class="text-sm text-pink-400">
										{(correlation.confidence * 100).toFixed(1)}% confidence
									</span>
								</div>
								<p class="text-gray-300 text-sm">{correlation.description}</p>
								<div class="mt-2 flex flex-wrap gap-2">
									{#each correlation.modalities as modality}
										<span class="px-2 py-1 bg-gray-700 {getModalityColor(modality)} text-xs rounded">
											{modality}
										</span>
									{/each}
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>
			
		{:else if $selectedTab === 'models'}
			<!-- Model Management -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				{#each Object.entries(processingModels) as [modality, models]}
					<div class="bg-gray-800 rounded-lg p-6">
						<h3 class="text-lg font-semibold mb-4 {getModalityColor(modality)} capitalize">
							{modality === 'image' ? '🖼️' : 
							 modality === 'video' ? '🎬' : 
							 modality === 'audio' ? '🎵' : '📝'} {modality} Models
						</h3>
						<div class="space-y-3">
							{#each models as model}
								<div class="bg-gray-900 rounded p-3">
									<div class="flex items-center justify-between mb-1">
										<span class="font-medium text-white text-sm">{model}</span>
										<span class="text-xs text-green-400">Loaded</span>
									</div>
									<div class="text-xs text-gray-400">
										Accuracy: {($multimodalState.model_performance[modality + '_accuracy'] || 0).toFixed(1)}%
									</div>
								</div>
							{/each}
						</div>
					</div>
				{/each}
			</div>
			
		{:else}
			<!-- Default tab interface -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-white">
					{$selectedTab === 'upload' ? '📤 Content Upload' : '🎭 Multimodal Processing'}
				</h3>
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">🎭</div>
					<p>Advanced multimodal processing interface</p>
					<p class="text-sm mt-2">Multi-format content analysis and understanding</p>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Multimodal Processing | 4 modalities | {analysisTypes.length} analysis types
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('export_analysis_results')}
					class="px-4 py-2 bg-pink-600 hover:bg-pink-700 rounded text-sm font-medium transition-colors"
				>
					Export Results
				</button>
				<button
					on:click={() => dispatch('optimize_models')}
					class="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm font-medium transition-colors"
				>
					Optimize Models
				</button>
			</div>
		</div>
	</div>
</div>

<!-- Job Detail Modal -->
{#if $selectedJob}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedJob.set(null)}>
		<div class="max-w-4xl w-full mx-4 bg-gray-800 rounded-lg p-6 max-h-[90vh] overflow-y-auto" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-pink-400">Analysis Job #{$selectedJob.id}</h3>
				<button
					on:click={() => selectedJob.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Job Information</h4>
					<div class="grid grid-cols-2 gap-4 text-sm">
						<div>
							<span class="text-gray-400">Status:</span>
							<span class="text-white ml-2 {getStatusColor($selectedJob.status)}">{$selectedJob.status}</span>
						</div>
						<div>
							<span class="text-gray-400">Progress:</span>
							<span class="text-white ml-2">{$selectedJob.progress || 0}%</span>
						</div>
						<div>
							<span class="text-gray-400">Files:</span>
							<span class="text-white ml-2">{$selectedJob.files?.length || 0}</span>
						</div>
						<div>
							<span class="text-gray-400">Started:</span>
							<span class="text-white ml-2">{formatDate($selectedJob.started_at)}</span>
						</div>
					</div>
				</div>
				
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Analysis Types</h4>
					<div class="flex flex-wrap gap-2">
						{#each $selectedJob.analysis_types as analysisType}
							{@const typeInfo = analysisTypes.find(t => t.id === analysisType)}
							<span class="px-2 py-1 bg-gray-700 text-gray-300 text-xs rounded">
								{typeInfo?.icon} {typeInfo?.name || analysisType}
							</span>
						{/each}
					</div>
				</div>
				
				{#if $selectedJob.results}
					<div class="bg-gray-900 rounded p-4">
						<h4 class="font-medium text-white mb-2">Analysis Results</h4>
						<div class="bg-gray-800 rounded p-3 max-h-64 overflow-y-auto">
							<pre class="text-gray-300 text-xs whitespace-pre-wrap">{JSON.stringify($selectedJob.results, null, 2)}</pre>
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>
{/if}

<style>
	.multimodal-center {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Multimodal-themed styling */
	:global(.multimodal-center .multimodal-active) {
		animation: pulse-pink 2s infinite;
	}
	
	@keyframes pulse-pink {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.8; }
	}
	
	/* Custom scrollbar */
	:global(.multimodal-center *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.multimodal-center *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.multimodal-center *::-webkit-scrollbar-thumb) {
		background: #ec4899;
		border-radius: 3px;
	}
	
	:global(.multimodal-center *::-webkit-scrollbar-thumb:hover) {
		background: #f472b6;
	}
</style>