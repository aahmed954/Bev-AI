<!--
Research Enhancement Framework - Automated Research Tools & Analysis
Connected to: src/enhancement/ (4 research systems)
Features: Research automation, watermark analysis, DRM research, metadata scrubbing
-->

<script lang="ts">
	import { onMount, createEventDispatcher } from 'svelte';
	import { writable } from 'svelte/store';
	import { invoke } from '@tauri-apps/api/core';
	
	const dispatch = createEventDispatcher();
	
	// State Management
	const researchState = writable({
		research_framework: {
			status: 'active',
			active_research: 0,
			completed_research: 0,
			automation_level: 85
		},
		watermark_research: {
			status: 'analyzing',
			watermarks_detected: 0,
			images_processed: 0,
			detection_accuracy: 94.2
		},
		drm_research: {
			status: 'scanning',
			drm_systems_analyzed: 0,
			vulnerabilities_found: 0,
			bypass_methods: 0
		},
		metadata_scrubber: {
			status: 'scrubbing',
			files_processed: 0,
			metadata_removed: 0,
			privacy_score: 98.5
		}
	});
	
	const selectedTab = writable('overview'); // 'overview', 'framework', 'watermark', 'drm', 'metadata'
	const selectedTool = writable(null);
	const isLoading = writable(false);
	
	// WebSocket for real-time updates
	let ws: WebSocket | null = null;
	
	// Research tools configuration
	const researchTools = [
		{
			id: 'research_framework',
			name: 'Research Framework',
			icon: '🔬',
			description: 'Automated research data collection and analysis',
			port: 8400,
			features: ['Auto-sourcing', 'Data correlation', 'Report generation', 'Citation tracking']
		},
		{
			id: 'watermark_research',
			name: 'Watermark Research',
			icon: '🏷️',
			description: 'Digital watermark detection and analysis',
			port: 8401,
			features: ['Image analysis', 'Audio watermarks', 'Video fingerprinting', 'Steganography detection']
		},
		{
			id: 'drm_research',
			name: 'DRM Research',
			icon: '🔐',
			description: 'Digital rights management analysis and research',
			port: 8402,
			features: ['DRM detection', 'Protection analysis', 'Bypass research', 'Vulnerability assessment']
		},
		{
			id: 'metadata_scrubber',
			name: 'Metadata Scrubber',
			icon: '🧹',
			description: 'Privacy protection through metadata removal',
			port: 8403,
			features: ['EXIF removal', 'Document cleaning', 'Privacy analysis', 'Batch processing']
		}
	];
	
	// Research project data
	let activeProjects: any[] = [];
	let researchResults: any[] = [];
	let privacyReports: any[] = [];
	
	// New research project form
	let newProject = {
		name: '',
		type: 'watermark_analysis',
		target: '',
		description: '',
		automation_level: 'high',
		privacy_mode: true
	};
	
	const projectTypes = [
		'watermark_analysis', 'drm_research', 'metadata_analysis', 
		'steganography_detection', 'privacy_audit', 'digital_forensics'
	];
	
	onMount(async () => {
		await initializeWebSocket();
		await loadResearchData();
		startResearchMonitoring();
	});
	
	async function initializeWebSocket() {
		try {
			ws = new WebSocket('ws://localhost:3010/research-enhancement');
			
			ws.onopen = () => {
				console.log('Research Enhancement WebSocket connected');
			};
			
			ws.onmessage = (event) => {
				const data = JSON.parse(event.data);
				handleResearchUpdate(data);
			};
			
			ws.onclose = () => {
				console.log('Research WebSocket disconnected, attempting reconnection...');
				setTimeout(initializeWebSocket, 5000);
			};
		} catch (error) {
			console.error('Research WebSocket connection failed:', error);
		}
	}
	
	function handleResearchUpdate(data: any) {
		switch (data.type) {
			case 'research_started':
				addActiveProject(data.project);
				break;
			case 'research_completed':
				completeProject(data.project_id, data.results);
				break;
			case 'watermark_detected':
				updateWatermarkResults(data.detection);
				break;
			case 'drm_analyzed':
				updateDRMResults(data.analysis);
				break;
			case 'metadata_scrubbed':
				updateMetadataResults(data.scrubbing);
				break;
			case 'privacy_report':
				addPrivacyReport(data.report);
				break;
		}
	}
	
	async function loadResearchData() {
		isLoading.set(true);
		try {
			const [projects, results, reports, metrics] = await Promise.all([
				invoke('get_active_research_projects'),
				invoke('get_research_results'),
				invoke('get_privacy_reports'),
				invoke('get_research_metrics')
			]);
			
			activeProjects = projects || [];
			researchResults = results || [];
			privacyReports = reports || [];
			
			// Update research state with metrics
			if (metrics) {
				researchState.update(current => ({
					...current,
					research_framework: { ...current.research_framework, ...metrics.framework },
					watermark_research: { ...current.watermark_research, ...metrics.watermark },
					drm_research: { ...current.drm_research, ...metrics.drm },
					metadata_scrubber: { ...current.metadata_scrubber, ...metrics.metadata }
				}));
			}
		} catch (error) {
			console.error('Failed to load research data:', error);
		} finally {
			isLoading.set(false);
		}
	}
	
	function startResearchMonitoring() {
		setInterval(async () => {
			try {
				for (const tool of researchTools) {
					const response = await fetch(`http://localhost:${tool.port}/status`);
					const status = await response.json();
					updateToolStatus(tool.id, status);
				}
			} catch (error) {
				console.error('Research monitoring failed:', error);
			}
		}, 10000);
	}
	
	async function startResearchProject() {
		try {
			const result = await invoke('start_research_project', { project: newProject });
			
			addActiveProject({
				id: result.project_id,
				...newProject,
				started_at: new Date().toISOString(),
				status: 'running'
			});
			
			// Reset form
			newProject = {
				name: '',
				type: 'watermark_analysis',
				target: '',
				description: '',
				automation_level: 'high',
				privacy_mode: true
			};
			
			dispatch('project_started', result);
		} catch (error) {
			console.error('Failed to start research project:', error);
		}
	}
	
	async function runWatermarkAnalysis(target: string) {
		try {
			const result = await invoke('analyze_watermarks', { target });
			updateWatermarkResults(result);
		} catch (error) {
			console.error('Failed to run watermark analysis:', error);
		}
	}
	
	async function runDRMAnalysis(target: string) {
		try {
			const result = await invoke('analyze_drm', { target });
			updateDRMResults(result);
		} catch (error) {
			console.error('Failed to run DRM analysis:', error);
		}
	}
	
	async function scrubMetadata(files: string[]) {
		try {
			const result = await invoke('scrub_metadata', { files });
			updateMetadataResults(result);
		} catch (error) {
			console.error('Failed to scrub metadata:', error);
		}
	}
	
	function addActiveProject(project: any) {
		activeProjects = [project, ...activeProjects];
	}
	
	function completeProject(projectId: string, results: any) {
		activeProjects = activeProjects.filter(p => p.id !== projectId);
		researchResults = [{ project_id: projectId, ...results }, ...researchResults.slice(0, 49)];
	}
	
	function updateWatermarkResults(detection: any) {
		researchState.update(current => ({
			...current,
			watermark_research: {
				...current.watermark_research,
				watermarks_detected: current.watermark_research.watermarks_detected + 1,
				images_processed: current.watermark_research.images_processed + 1
			}
		}));
	}
	
	function updateDRMResults(analysis: any) {
		researchState.update(current => ({
			...current,
			drm_research: {
				...current.drm_research,
				drm_systems_analyzed: current.drm_research.drm_systems_analyzed + 1,
				vulnerabilities_found: current.drm_research.vulnerabilities_found + (analysis.vulnerabilities?.length || 0)
			}
		}));
	}
	
	function updateMetadataResults(scrubbing: any) {
		researchState.update(current => ({
			...current,
			metadata_scrubber: {
				...current.metadata_scrubber,
				files_processed: current.metadata_scrubber.files_processed + (scrubbing.files_processed || 0),
				metadata_removed: current.metadata_scrubber.metadata_removed + (scrubbing.metadata_removed || 0)
			}
		}));
	}
	
	function addPrivacyReport(report: any) {
		privacyReports = [report, ...privacyReports.slice(0, 19)];
	}
	
	function updateToolStatus(toolId: string, status: any) {
		researchState.update(current => ({
			...current,
			[toolId]: { ...current[toolId], ...status }
		}));
	}
	
	function getStatusColor(status: string): string {
		switch (status) {
			case 'active':
			case 'analyzing':
			case 'scanning':
			case 'scrubbing': return 'text-green-400';
			case 'warning': return 'text-yellow-400';
			case 'error':
			case 'failed': return 'text-red-400';
			case 'idle':
			case 'paused': return 'text-gray-400';
			default: return 'text-gray-400';
		}
	}
	
	function formatNumber(num: number): string {
		if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
		if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
		return num.toString();
	}
	
	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleString();
	}
	
	function openToolModal(tool: any) {
		selectedTool.set(tool);
	}
</script>

<!-- Research Enhancement Framework -->
<div class="research-framework h-full bg-gray-900 text-white">
	<!-- Header -->
	<div class="border-b border-gray-800 p-4">
		<div class="flex items-center justify-between">
			<h1 class="text-2xl font-bold text-teal-400">🔬 Research Enhancement Framework</h1>
			<div class="flex items-center space-x-4">
				<div class="text-sm text-gray-400">
					{activeProjects.length} active projects | {researchTools.filter(t => $researchState[t.id]?.status === 'active').length}/4 tools online
				</div>
				{#if $isLoading}
					<div class="w-4 h-4 border-2 border-teal-400 border-t-transparent rounded-full animate-spin"></div>
				{:else}
					<button
						on:click={loadResearchData}
						class="px-3 py-1 bg-teal-600 hover:bg-teal-700 rounded text-sm transition-colors"
					>
						Refresh
					</button>
				{/if}
			</div>
		</div>
	</div>
	
	<!-- Research Tools Status -->
	<div class="p-4 border-b border-gray-800">
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
			{#each researchTools as tool}
				{@const toolData = $researchState[tool.id] || {}}
				<div 
					class="bg-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-700 transition-colors"
					on:click={() => openToolModal(tool)}
				>
					<div class="flex items-center justify-between mb-3">
						<div class="flex items-center space-x-2">
							<span class="text-xl">{tool.icon}</span>
							<span class="font-medium text-white text-sm">{tool.name}</span>
						</div>
						<div class="flex items-center space-x-1">
							<div class="w-2 h-2 rounded-full {getStatusColor(toolData.status)}"></div>
							<span class="text-xs {getStatusColor(toolData.status)}">{toolData.status || 'unknown'}</span>
						</div>
					</div>
					
					<p class="text-gray-400 text-xs mb-3">{tool.description}</p>
					
					{#if tool.id === 'research_framework'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Active:</span>
								<span class="text-white">{toolData.active_research || 0}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Completed:</span>
								<span class="text-green-400">{toolData.completed_research || 0}</span>
							</div>
						</div>
					{:else if tool.id === 'watermark_research'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Detected:</span>
								<span class="text-white">{toolData.watermarks_detected || 0}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Accuracy:</span>
								<span class="text-blue-400">{(toolData.detection_accuracy || 0).toFixed(1)}%</span>
							</div>
						</div>
					{:else if tool.id === 'drm_research'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Systems:</span>
								<span class="text-white">{toolData.drm_systems_analyzed || 0}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Vulns:</span>
								<span class="text-red-400">{toolData.vulnerabilities_found || 0}</span>
							</div>
						</div>
					{:else if tool.id === 'metadata_scrubber'}
						<div class="space-y-1 text-xs">
							<div class="flex justify-between">
								<span class="text-gray-400">Processed:</span>
								<span class="text-white">{toolData.files_processed || 0}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-400">Privacy:</span>
								<span class="text-green-400">{(toolData.privacy_score || 0).toFixed(1)}%</span>
							</div>
						</div>
					{/if}
				</div>
			{/each}
		</div>
	</div>
	
	<!-- Navigation Tabs -->
	<div class="border-b border-gray-800">
		<nav class="flex space-x-8 px-4">
			{#each [
				{ id: 'overview', label: 'Research Overview', icon: '📊' },
				{ id: 'framework', label: 'Research Framework', icon: '🔬' },
				{ id: 'watermark', label: 'Watermark Analysis', icon: '🏷️' },
				{ id: 'drm', label: 'DRM Research', icon: '🔐' },
				{ id: 'metadata', label: 'Metadata Scrubber', icon: '🧹' },
				{ id: 'projects', label: 'Active Projects', icon: '📋' }
			] as tab}
				<button
					class="py-4 px-1 border-b-2 font-medium text-sm transition-colors {
						$selectedTab === tab.id
							? 'border-teal-500 text-teal-400'
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
			<!-- Research Overview -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
				<!-- Research Statistics -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-teal-400">Research Statistics</h3>
					<div class="space-y-4">
						<div class="grid grid-cols-2 gap-4 text-center">
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-blue-400">{activeProjects.length}</div>
								<div class="text-xs text-gray-400">Active Projects</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-green-400">{researchResults.length}</div>
								<div class="text-xs text-gray-400">Completed Research</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-yellow-400">{$researchState.watermark_research.watermarks_detected}</div>
								<div class="text-xs text-gray-400">Watermarks Found</div>
							</div>
							<div class="bg-gray-900 rounded p-4">
								<div class="text-2xl font-bold text-purple-400">{$researchState.metadata_scrubber.files_processed}</div>
								<div class="text-xs text-gray-400">Files Processed</div>
							</div>
						</div>
						
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">Research Performance</h4>
							<div class="space-y-2 text-sm">
								<div class="flex justify-between">
									<span class="text-gray-400">Automation Level:</span>
									<span class="text-white">{($researchState.research_framework.automation_level || 0)}%</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Detection Accuracy:</span>
									<span class="text-white">{($researchState.watermark_research.detection_accuracy || 0).toFixed(1)}%</span>
								</div>
								<div class="flex justify-between">
									<span class="text-gray-400">Privacy Score:</span>
									<span class="text-green-400">{($researchState.metadata_scrubber.privacy_score || 0).toFixed(1)}%</span>
								</div>
							</div>
						</div>
					</div>
				</div>
				
				<!-- Quick Research Tools -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Quick Research Tools</h3>
					<div class="space-y-4">
						<!-- Watermark Analysis -->
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">🏷️ Watermark Analysis</h4>
							<div class="flex space-x-2">
								<input
									type="text"
									placeholder="Image/Video URL or file path"
									class="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-teal-500"
								/>
								<button
									on:click={() => runWatermarkAnalysis('')}
									class="px-4 py-2 bg-teal-600 hover:bg-teal-700 rounded transition-colors"
								>
									Analyze
								</button>
							</div>
						</div>
						
						<!-- DRM Analysis -->
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">🔐 DRM Analysis</h4>
							<div class="flex space-x-2">
								<input
									type="text"
									placeholder="Software/Media URL"
									class="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-teal-500"
								/>
								<button
									on:click={() => runDRMAnalysis('')}
									class="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded transition-colors"
								>
									Research
								</button>
							</div>
						</div>
						
						<!-- Metadata Scrubbing -->
						<div class="bg-gray-900 rounded p-4">
							<h4 class="font-medium text-white mb-2">🧹 Metadata Scrubber</h4>
							<div class="space-y-2">
								<input
									type="file"
									multiple
									class="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-teal-500"
								/>
								<button
									on:click={() => scrubMetadata([])}
									class="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded transition-colors"
								>
									Scrub Metadata
								</button>
							</div>
						</div>
					</div>
				</div>
			</div>
			
		{:else if $selectedTab === 'projects'}
			<!-- Active Projects -->
			<div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
				<!-- New Project Form -->
				<div class="bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-green-400">Start New Research</h3>
					<div class="space-y-4">
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Project Name</label>
							<input
								type="text"
								bind:value={newProject.name}
								placeholder="Enter project name"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							/>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Research Type</label>
							<select
								bind:value={newProject.type}
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							>
								{#each projectTypes as type}
									<option value={type}>{type.replace('_', ' ')}</option>
								{/each}
							</select>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Target</label>
							<input
								type="text"
								bind:value={newProject.target}
								placeholder="URL, file path, or identifier"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							/>
						</div>
						
						<div>
							<label class="block text-sm font-medium text-gray-300 mb-2">Description</label>
							<textarea
								bind:value={newProject.description}
								placeholder="Research project description"
								rows="3"
								class="w-full px-3 py-2 bg-gray-900 border border-gray-700 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
							></textarea>
						</div>
						
						<div class="flex items-center">
							<input
								type="checkbox"
								bind:checked={newProject.privacy_mode}
								id="privacy_mode"
								class="mr-2"
							/>
							<label for="privacy_mode" class="text-sm text-gray-300">Enable privacy mode</label>
						</div>
						
						<button
							on:click={startResearchProject}
							class="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded font-medium transition-colors"
							disabled={!newProject.name || !newProject.target}
						>
							Start Research
						</button>
					</div>
				</div>
				
				<!-- Active Projects List -->
				<div class="lg:col-span-2 bg-gray-800 rounded-lg p-6">
					<h3 class="text-lg font-semibold mb-4 text-blue-400">Active Research Projects</h3>
					{#if activeProjects.length === 0}
						<div class="text-center py-12 text-gray-400">
							<div class="text-4xl mb-4">📋</div>
							<p>No active research projects</p>
						</div>
					{:else}
						<div class="space-y-3">
							{#each activeProjects as project}
								<div class="bg-gray-900 rounded p-4">
									<div class="flex items-center justify-between mb-3">
										<div class="flex items-center space-x-3">
											<h4 class="font-medium text-white">{project.name}</h4>
											<span class="px-2 py-1 rounded text-xs {getStatusColor(project.status)} bg-gray-800">
												{project.status}
											</span>
											<span class="text-sm text-gray-400 capitalize">{project.type.replace('_', ' ')}</span>
										</div>
										<div class="text-sm text-gray-400">
											{formatDate(project.started_at)}
										</div>
									</div>
									
									<p class="text-gray-300 text-sm mb-2">{project.description}</p>
									
									<div class="flex items-center justify-between text-sm">
										<span class="text-gray-400">Target: {project.target}</span>
										{#if project.privacy_mode}
											<span class="text-green-400">🔒 Privacy Mode</span>
										{/if}
									</div>
								</div>
							{/each}
						</div>
					{/if}
				</div>
			</div>
			
		{:else}
			<!-- Detailed tool interfaces -->
			<div class="bg-gray-800 rounded-lg p-6">
				<h3 class="text-lg font-semibold mb-4 text-white">{researchTools.find(t => t.id === $selectedTab)?.name} Interface</h3>
				<div class="text-center py-12 text-gray-400">
					<div class="text-4xl mb-4">{researchTools.find(t => t.id === $selectedTab)?.icon}</div>
					<p>Detailed {researchTools.find(t => t.id === $selectedTab)?.name} interface</p>
					<p class="text-sm mt-2">Advanced research tools and analysis capabilities</p>
				</div>
			</div>
		{/if}
	</div>
	
	<!-- Action Bar -->
	<div class="border-t border-gray-800 p-4">
		<div class="flex justify-between items-center">
			<div class="text-sm text-gray-400">
				Research Enhancement Framework | 4 tools integrated
			</div>
			<div class="flex space-x-2">
				<button
					on:click={() => dispatch('export_research_report')}
					class="px-4 py-2 bg-teal-600 hover:bg-teal-700 rounded text-sm font-medium transition-colors"
				>
					Export Research Report
				</button>
				<button
					on:click={() => dispatch('start_automated_research')}
					class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded text-sm font-medium transition-colors"
				>
					Start Automated Research
				</button>
			</div>
		</div>
	</div>
</div>

<!-- Tool Detail Modal -->
{#if $selectedTool}
	<div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50" on:click={() => selectedTool.set(null)}>
		<div class="max-w-3xl w-full mx-4 bg-gray-800 rounded-lg p-6" on:click|stopPropagation>
			<div class="flex items-center justify-between mb-4">
				<h3 class="text-xl font-semibold text-teal-400">
					{$selectedTool.icon} {$selectedTool.name}
				</h3>
				<button
					on:click={() => selectedTool.set(null)}
					class="text-gray-400 hover:text-white"
				>
					✕
				</button>
			</div>
			
			<div class="space-y-4">
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Tool Information</h4>
					<p class="text-gray-300 text-sm">{$selectedTool.description}</p>
				</div>
				
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Features</h4>
					<div class="flex flex-wrap gap-2">
						{#each $selectedTool.features as feature}
							<span class="px-2 py-1 bg-teal-600 text-white text-xs rounded">
								{feature}
							</span>
						{/each}
					</div>
				</div>
				
				<div class="bg-gray-900 rounded p-4">
					<h4 class="font-medium text-white mb-2">Current Status</h4>
					<div class="space-y-2 text-sm">
						{@const toolData = $researchState[$selectedTool.id] || {}}
						{#each Object.entries(toolData) as [key, value]}
							<div class="flex justify-between">
								<span class="text-gray-400 capitalize">{key.replace('_', ' ')}:</span>
								<span class="text-white">{value}</span>
							</div>
						{/each}
					</div>
				</div>
			</div>
		</div>
	</div>
{/if}

<style>
	.research-framework {
		display: flex;
		flex-direction: column;
		height: 100vh;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}
	
	/* Research-themed styling */
	:global(.research-framework .research-active) {
		animation: pulse-teal 2s infinite;
	}
	
	@keyframes pulse-teal {
		0%, 100% { opacity: 1; }
		50% { opacity: 0.8; }
	}
	
	/* Custom scrollbar */
	:global(.research-framework *::-webkit-scrollbar) {
		width: 6px;
	}
	
	:global(.research-framework *::-webkit-scrollbar-track) {
		background: #374151;
	}
	
	:global(.research-framework *::-webkit-scrollbar-thumb) {
		background: #14b8a6;
		border-radius: 3px;
	}
	
	:global(.research-framework *::-webkit-scrollbar-thumb:hover) {
		background: #5eead4;
	}
</style>