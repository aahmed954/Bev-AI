<script lang="ts">
  import { endpoints, websockets, getEndpoint, getWebSocket } from "$lib/config/endpoints";

  // Distributed endpoint helpers
  const getServiceHost = () => {
    const service = typeof window !== "undefined" && window.location.hostname;
    return service === "localhost" ? "localhost" : service;
  };

  const getWebSocketHost = () => {
    const service = typeof window !== "undefined" && window.location.hostname;
    return service === "localhost" ? "localhost" : service;
  };
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // Enhancement Research state
  const enhancementState = writable({
    watermark_research: {
      status: 'active',
      detection_accuracy: 94.7,
      extraction_success_rate: 87.3,
      algorithms_available: 12,
      processed_images: 2847,
      watermarks_detected: 1567,
      steganography_found: 234
    },
    drm_research: {
      status: 'analyzing',
      schemes_analyzed: 47,
      protection_bypassed: 12,
      format_support: 23,
      extraction_tools: 8,
      compliance_checks: 156,
      legal_validations: 89
    },
    metadata_scrubbing: {
      status: 'processing',
      files_processed: 15847,
      metadata_removed: 234567,
      privacy_score: 96.8,
      batch_operations: 23,
      sanitization_rules: 47,
      verification_passed: 98.2
    },
    research_framework: {
      status: 'coordinating',
      active_research_threads: 8,
      data_correlation_score: 91.4,
      hypothesis_validation: 87.6,
      cross_reference_count: 4567,
      research_quality_score: 93.2,
      publication_readiness: 78.9
    }
  });

  // Research controls
  let activeEnhancementWorkspace = 'watermark';
  let selectedFile = null;
  let uploadedFiles = [];
  let batchProcessing = false;

  // Watermark analysis configuration
  let watermarkConfig = {
    detection_algorithms: ['dct', 'dwt', 'spatial', 'frequency'],
    sensitivity_level: 0.8,
    extract_payload: true,
    analyze_robustness: true,
    generate_report: true
  };

  // DRM research configuration
  let drmConfig = {
    target_format: 'pdf',
    analysis_depth: 'comprehensive',
    bypass_attempts: false,
    compliance_check: true,
    legal_validation: true,
    research_only: true
  };

  // Metadata scrubbing configuration
  let scrubConfig = {
    file_types: ['image', 'document', 'video', 'audio'],
    preserve_essential: true,
    deep_scrub: true,
    verify_removal: true,
    backup_original: false
  };

  // Research results
  const watermarkResults = writable([]);
  const drmAnalysisResults = writable([]);
  const metadataReports = writable([]);
  const researchProjects = writable([]);

  // WebSocket connections
  let watermarkWs: WebSocket | null = null;
  let drmWs: WebSocket | null = null;
  let metadataWs: WebSocket | null = null;
  let researchWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadResearchData();
    startProgressMonitoring();
  });

  onDestroy(() => {
    if (watermarkWs) watermarkWs.close();
    if (drmWs) drmWs.close();
    if (metadataWs) metadataWs.close();
    if (researchWs) researchWs.close();
  });

  function initializeWebSockets() {
    // Watermark research WebSocket
    watermarkWs = new WebSocket('ws://${getWebSocketHost()}:8060/watermark');
    watermarkWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      enhancementState.update(state => ({
        ...state,
        watermark_research: { ...state.watermark_research, ...data }
      }));
      if (data.results) {
        watermarkResults.set(data.results);
      }
    };

    // DRM research WebSocket
    drmWs = new WebSocket('ws://${getWebSocketHost()}:8061/drm');
    drmWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      enhancementState.update(state => ({
        ...state,
        drm_research: { ...state.drm_research, ...data }
      }));
      if (data.analysis_results) {
        drmAnalysisResults.set(data.analysis_results);
      }
    };

    // Metadata scrubbing WebSocket
    metadataWs = new WebSocket('ws://${getWebSocketHost()}:8062/metadata');
    metadataWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      enhancementState.update(state => ({
        ...state,
        metadata_scrubbing: { ...state.metadata_scrubbing, ...data }
      }));
      if (data.reports) {
        metadataReports.set(data.reports);
      }
    };

    // Research framework WebSocket
    researchWs = new WebSocket('ws://${getWebSocketHost()}:8063/research');
    researchWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      enhancementState.update(state => ({
        ...state,
        research_framework: { ...state.research_framework, ...data }
      }));
    };
  }

  async function loadResearchData() {
    try {
      const [watermarkRes, drmRes, metadataRes, projectsRes] = await Promise.all([
        fetch('http://${getServiceHost()}:8060/api/results'),
        fetch('http://${getServiceHost()}:8061/api/analysis'),
        fetch('http://${getServiceHost()}:8062/api/reports'),
        fetch('http://${getServiceHost()}:8063/api/projects')
      ]);

      const watermarkData = await watermarkRes.json();
      const drmData = await drmRes.json();
      const metadataData = await metadataRes.json();
      const projectsData = await projectsRes.json();

      watermarkResults.set(watermarkData);
      drmAnalysisResults.set(drmData);
      metadataReports.set(metadataData);
      researchProjects.set(projectsData);
    } catch (error) {
      console.error('Failed to load research data:', error);
    }
  }

  async function startProgressMonitoring() {
    setInterval(async () => {
      try {
        const response = await fetch('http://${getServiceHost()}:8060/api/metrics');
        const metrics = await response.json();
        enhancementState.update(state => ({
          ...state,
          ...metrics
        }));
      } catch (error) {
        console.error('Progress monitoring error:', error);
      }
    }, 5000);
  }

  async function analyzeWatermarks() {
    if (!selectedFile) return;

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('config', JSON.stringify(watermarkConfig));

      const response = await fetch('http://${getServiceHost()}:8060/api/analyze', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        watermarkResults.update(results => [result, ...results]);
      }
    } catch (error) {
      console.error('Watermark analysis failed:', error);
    }
  }

  async function analyzeDRM() {
    if (!selectedFile) return;

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('config', JSON.stringify(drmConfig));

      const response = await fetch('http://${getServiceHost()}:8061/api/analyze', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        drmAnalysisResults.update(results => [result, ...results]);
      }
    } catch (error) {
      console.error('DRM analysis failed:', error);
    }
  }

  async function scrubMetadata() {
    if (!selectedFile && uploadedFiles.length === 0) return;

    try {
      const formData = new FormData();

      if (batchProcessing) {
        uploadedFiles.forEach((file, index) => {
          formData.append(`files[${index}]`, file);
        });
      } else {
        formData.append('file', selectedFile);
      }

      formData.append('config', JSON.stringify(scrubConfig));

      const response = await fetch('http://${getServiceHost()}:8062/api/scrub', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        metadataReports.update(reports => [result, ...reports]);
      }
    } catch (error) {
      console.error('Metadata scrubbing failed:', error);
    }
  }

  function handleFileUpload(event: Event) {
    const target = event.target as HTMLInputElement;
    const files = target.files;

    if (files && files.length > 0) {
      if (batchProcessing) {
        uploadedFiles = Array.from(files);
      } else {
        selectedFile = files[0];
      }
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'active': case 'completed': case 'passed': return 'text-green-400';
      case 'analyzing': case 'processing': case 'coordinating': return 'text-blue-400';
      case 'failed': case 'error': return 'text-red-400';
      case 'warning': case 'pending': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  }

  function getHealthColor(value: number, threshold: number = 90): string {
    if (value >= threshold) return 'text-green-400';
    if (value >= threshold - 15) return 'text-yellow-400';
    return 'text-red-400';
  }
</script>

<svelte:head>
  <title>Enhancement Research Tools | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-pink-400 to-rose-500 bg-clip-text text-transparent">
        Enhancement Research Tools
      </h1>
      <p class="text-gray-300">Advanced digital forensics and research automation</p>
    </div>

    {#if $enhancementState}
      <!-- Research Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6 mb-8">
        <!-- Watermark Research -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Watermark Research
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($enhancementState.watermark_research.status)}>
                {$enhancementState.watermark_research.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Detection:</span>
              <span class={getHealthColor($enhancementState.watermark_research.detection_accuracy)}>{$enhancementState.watermark_research.detection_accuracy}%</span>
            </div>
            <div class="flex justify-between">
              <span>Processed:</span>
              <span class="text-blue-400">{$enhancementState.watermark_research.processed_images.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Detected:</span>
              <span class="text-green-400">{$enhancementState.watermark_research.watermarks_detected.toLocaleString()}</span>
            </div>
          </div>
        </div>

        <!-- DRM Research -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            DRM Research
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($enhancementState.drm_research.status)}>
                {$enhancementState.drm_research.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Schemes:</span>
              <span class="text-purple-400">{$enhancementState.drm_research.schemes_analyzed}</span>
            </div>
            <div class="flex justify-between">
              <span>Formats:</span>
              <span class="text-blue-400">{$enhancementState.drm_research.format_support}</span>
            </div>
            <div class="flex justify-between">
              <span>Tools:</span>
              <span class="text-green-400">{$enhancementState.drm_research.extraction_tools}</span>
            </div>
          </div>
        </div>

        <!-- Metadata Scrubbing -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Metadata Scrubbing
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($enhancementState.metadata_scrubbing.status)}>
                {$enhancementState.metadata_scrubbing.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Processed:</span>
              <span class="text-green-400">{$enhancementState.metadata_scrubbing.files_processed.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Privacy Score:</span>
              <span class={getHealthColor($enhancementState.metadata_scrubbing.privacy_score)}>{$enhancementState.metadata_scrubbing.privacy_score}%</span>
            </div>
            <div class="flex justify-between">
              <span>Verified:</span>
              <span class="text-cyan-400">{$enhancementState.metadata_scrubbing.verification_passed}%</span>
            </div>
          </div>
        </div>

        <!-- Research Framework -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-orange-400 mr-2"></span>
            Research Framework
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($enhancementState.research_framework.status)}>
                {$enhancementState.research_framework.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Threads:</span>
              <span class="text-orange-400">{$enhancementState.research_framework.active_research_threads}</span>
            </div>
            <div class="flex justify-between">
              <span>Quality:</span>
              <span class={getHealthColor($enhancementState.research_framework.research_quality_score)}>{$enhancementState.research_framework.research_quality_score}%</span>
            </div>
            <div class="flex justify-between">
              <span>Publication:</span>
              <span class="text-blue-400">{$enhancementState.research_framework.publication_readiness}%</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Research Workspace Tabs -->
      <div class="mb-6">
        <div class="flex space-x-1 bg-gray-800 p-1 rounded-lg">
          {#each ['watermark', 'drm', 'metadata', 'research'] as workspace}
            <button
              on:click={() => activeEnhancementWorkspace = workspace}
              class="flex-1 py-2 px-4 rounded text-center transition-colors text-sm
                {activeEnhancementWorkspace === workspace
                  ? 'bg-pink-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'}"
            >
              {workspace.charAt(0).toUpperCase() + workspace.slice(1)}
            </button>
          {/each}
        </div>
      </div>

      <!-- Workspace Content -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {#if activeEnhancementWorkspace === 'watermark'}
          <!-- Watermark Analysis -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Watermark & Steganography Analysis</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Upload Image File</label>
                <input
                  type="file"
                  accept="image/*"
                  on:change={handleFileUpload}
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Detection Algorithms</label>
                <div class="grid grid-cols-2 gap-2">
                  {#each watermarkConfig.detection_algorithms as algorithm}
                    <label class="flex items-center">
                      <input type="checkbox" bind:group={watermarkConfig.detection_algorithms} value={algorithm} class="mr-2">
                      <span class="text-sm">{algorithm.toUpperCase()}</span>
                    </label>
                  {/each}
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Sensitivity Level: {watermarkConfig.sensitivity_level}</label>
                <input
                  type="range"
                  bind:value={watermarkConfig.sensitivity_level}
                  min="0.1"
                  max="1.0"
                  step="0.1"
                  class="w-full"
                >
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={watermarkConfig.extract_payload} class="mr-2">
                  <span class="text-sm">Extract Hidden Payload</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={watermarkConfig.analyze_robustness} class="mr-2">
                  <span class="text-sm">Analyze Robustness</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={watermarkConfig.generate_report} class="mr-2">
                  <span class="text-sm">Generate Research Report</span>
                </label>
              </div>
              <button
                on:click={analyzeWatermarks}
                disabled={!selectedFile}
                class="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                Analyze Watermarks
              </button>
            </div>
          </div>

          <!-- Watermark Results -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Analysis Results</h3>
            <div class="space-y-3">
              {#if $watermarkResults && $watermarkResults.length > 0}
                {#each $watermarkResults.slice(0, 5) as result}
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">{result.filename}</span>
                      <span class={result.watermark_detected ? 'text-green-400' : 'text-gray-400'} class="text-xs">
                        {result.watermark_detected ? 'DETECTED' : 'NONE'}
                      </span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Algorithm: {result.algorithm}</div>
                      <div>Confidence: {result.confidence}%</div>
                      {#if result.payload}
                        <div>Payload: {result.payload.substring(0, 20)}...</div>
                      {/if}
                    </div>
                  </div>
                {/each}
              {:else}
                <div class="text-center text-gray-400 py-4">
                  No watermark analysis results yet
                </div>
              {/if}
            </div>
          </div>

        {:else if activeEnhancementWorkspace === 'drm'}
          <!-- DRM Analysis -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">DRM Research & Analysis</h3>
            <div class="space-y-4">
              <div class="bg-red-900 border border-red-600 rounded p-4 mb-4">
                <div class="flex items-center">
                  <svg class="w-5 h-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                  </svg>
                  <span class="font-medium">Research Use Only</span>
                </div>
                <p class="text-sm text-red-300 mt-1">This tool is for authorized security research and academic purposes only</p>
              </div>

              <div>
                <label class="block text-sm font-medium mb-2">Upload Protected File</label>
                <input
                  type="file"
                  on:change={handleFileUpload}
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Target Format</label>
                <select bind:value={drmConfig.target_format} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="pdf">PDF Documents</option>
                  <option value="ebook">E-books (EPUB, MOBI)</option>
                  <option value="video">Video Files</option>
                  <option value="audio">Audio Files</option>
                  <option value="software">Software Binaries</option>
                </select>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Analysis Depth</label>
                <select bind:value={drmConfig.analysis_depth} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="surface">Surface Analysis</option>
                  <option value="detailed">Detailed Analysis</option>
                  <option value="comprehensive">Comprehensive Research</option>
                </select>
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={drmConfig.compliance_check} class="mr-2">
                  <span class="text-sm">Legal Compliance Check</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={drmConfig.legal_validation} class="mr-2">
                  <span class="text-sm">Legal Framework Validation</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={drmConfig.research_only} class="mr-2">
                  <span class="text-sm">Research-Only Mode</span>
                </label>
              </div>
              <button
                on:click={analyzeDRM}
                disabled={!selectedFile || !drmConfig.research_only}
                class="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                Analyze DRM Scheme
              </button>
            </div>
          </div>

          <!-- DRM Analysis Results -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">DRM Analysis Results</h3>
            <div class="space-y-3">
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Adobe PDF DRM</span>
                  <span class="text-green-400 text-xs">ANALYZED</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Protection: Adobe ADEPT</div>
                  <div>Complexity: High</div>
                  <div>Research Notes: Available</div>
                </div>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Apple FairPlay</span>
                  <span class="text-yellow-400 text-xs">IN PROGRESS</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Protection: FairPlay DRM</div>
                  <div>Progress: 67%</div>
                </div>
              </div>
            </div>
          </div>

        {:else if activeEnhancementWorkspace === 'metadata'}
          <!-- Metadata Scrubbing -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Metadata Scrubbing & Privacy</h3>
            <div class="space-y-4">
              <div class="flex items-center mb-4">
                <input type="checkbox" bind:checked={batchProcessing} id="batch-mode" class="mr-2">
                <label for="batch-mode" class="text-sm font-medium">Batch Processing Mode</label>
              </div>

              <div>
                <label class="block text-sm font-medium mb-2">
                  {batchProcessing ? 'Upload Multiple Files' : 'Upload Single File'}
                </label>
                <input
                  type="file"
                  {multiple: batchProcessing}
                  on:change={handleFileUpload}
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>

              <div>
                <label class="block text-sm font-medium mb-2">File Types to Process</label>
                <div class="grid grid-cols-2 gap-2">
                  {#each scrubConfig.file_types as fileType}
                    <label class="flex items-center">
                      <input type="checkbox" bind:group={scrubConfig.file_types} value={fileType} class="mr-2">
                      <span class="text-sm">{fileType.toUpperCase()}</span>
                    </label>
                  {/each}
                </div>
              </div>

              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={scrubConfig.preserve_essential} class="mr-2">
                  <span class="text-sm">Preserve Essential Metadata</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={scrubConfig.deep_scrub} class="mr-2">
                  <span class="text-sm">Deep Scrubbing (slower)</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={scrubConfig.verify_removal} class="mr-2">
                  <span class="text-sm">Verify Complete Removal</span>
                </label>
              </div>

              <button
                on:click={scrubMetadata}
                disabled={!selectedFile && uploadedFiles.length === 0}
                class="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                {batchProcessing ? 'Scrub All Files' : 'Scrub Metadata'}
              </button>
            </div>
          </div>

          <!-- Scrubbing Reports -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Scrubbing Reports</h3>
            <div class="space-y-3">
              {#if $metadataReports && $metadataReports.length > 0}
                {#each $metadataReports.slice(0, 5) as report}
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">{report.filename}</span>
                      <span class="text-green-400 text-xs">SCRUBBED</span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Removed: {report.metadata_removed} fields</div>
                      <div>Privacy Score: {report.privacy_score}%</div>
                      <div>Size Reduction: {report.size_reduction}%</div>
                    </div>
                  </div>
                {/each}
              {:else}
                <div class="text-center text-gray-400 py-4">
                  No scrubbing reports available
                </div>
              {/if}
            </div>
          </div>

        {:else if activeEnhancementWorkspace === 'research'}
          <!-- Research Framework -->
          <div class="lg:col-span-3 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Coordinated Research Framework</h3>
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <!-- Research Threads -->
              <div class="space-y-4">
                <h4 class="font-medium">Active Research Threads</h4>
                <div class="space-y-3">
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">Digital Forensics Study</span>
                      <span class="text-blue-400 text-xs">ACTIVE</span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Focus: Steganography detection methods</div>
                      <div>Progress: 73%</div>
                      <div>Team: 4 researchers</div>
                    </div>
                  </div>
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">DRM Evolution Analysis</span>
                      <span class="text-green-400 text-xs">COORDINATING</span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Focus: Modern DRM scheme analysis</div>
                      <div>Progress: 45%</div>
                      <div>Cross-refs: 1,247</div>
                    </div>
                  </div>
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">Privacy Framework</span>
                      <span class="text-purple-400 text-xs">REVIEWING</span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Focus: Metadata privacy analysis</div>
                      <div>Progress: 89%</div>
                      <div>Publications: 3 ready</div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- Research Metrics -->
              <div class="space-y-4">
                <h4 class="font-medium">Research Quality Metrics</h4>
                <div class="space-y-4">
                  <div>
                    <div class="flex justify-between text-sm mb-1">
                      <span>Data Correlation Score</span>
                      <span>{$enhancementState.research_framework.data_correlation_score}%</span>
                    </div>
                    <div class="w-full bg-gray-700 rounded-full h-2">
                      <div class="bg-blue-400 h-2 rounded-full" style="width: {$enhancementState.research_framework.data_correlation_score}%"></div>
                    </div>
                  </div>
                  <div>
                    <div class="flex justify-between text-sm mb-1">
                      <span>Hypothesis Validation</span>
                      <span>{$enhancementState.research_framework.hypothesis_validation}%</span>
                    </div>
                    <div class="w-full bg-gray-700 rounded-full h-2">
                      <div class="bg-green-400 h-2 rounded-full" style="width: {$enhancementState.research_framework.hypothesis_validation}%"></div>
                    </div>
                  </div>
                  <div>
                    <div class="flex justify-between text-sm mb-1">
                      <span>Publication Readiness</span>
                      <span>{$enhancementState.research_framework.publication_readiness}%</span>
                    </div>
                    <div class="w-full bg-gray-700 rounded-full h-2">
                      <div class="bg-purple-400 h-2 rounded-full" style="width: {$enhancementState.research_framework.publication_readiness}%"></div>
                    </div>
                  </div>
                </div>

                <div class="bg-gray-700 rounded p-3 mt-4">
                  <h4 class="font-medium mb-2">Cross-References</h4>
                  <div class="text-lg font-bold text-orange-400">
                    {$enhancementState.research_framework.cross_reference_count.toLocaleString()}
                  </div>
                  <div class="text-sm text-gray-300">Academic citations and references</div>
                </div>
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>