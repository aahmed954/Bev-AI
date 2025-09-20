<!-- BEV OCR Intelligence Platform -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import OCRUploader from '$lib/components/ocr/OCRUploader.svelte';
  import OCRDashboard from '$lib/components/ocr/OCRDashboard.svelte';
  import OCRComparison from '$lib/components/ocr/OCRComparison.svelte';
  import DocumentAnalysis from '$lib/components/ocr/DocumentAnalysis.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  interface OCRJob {
    id: string;
    filename: string;
    status: 'pending' | 'processing' | 'completed' | 'failed';
    engines: string[];
    results?: OCRResult[];
    progress: number;
    timestamp: Date;
    fileSize: number;
    fileType: string;
  }

  interface OCRResult {
    engine: string;
    text: string;
    confidence: number;
    processingTime: number;
    layout?: any[];
    metadata?: Record<string, any>;
  }

  interface OCRStats {
    totalJobs: number;
    successfulJobs: number;
    failedJobs: number;
    avgProcessingTime: number;
    avgConfidence: number;
    totalDocuments: number;
  }

  let activeJobs = writable<OCRJob[]>([]);
  let completedJobs = writable<OCRJob[]>([]);
  let selectedJob = writable<OCRJob | null>(null);
  let ocrStats = writable<OCRStats>({
    totalJobs: 0,
    successfulJobs: 0,
    failedJobs: 0,
    avgProcessingTime: 0,
    avgConfidence: 0,
    totalDocuments: 0
  });

  let websocket: WebSocket | null = null;
  let connectionStatus: 'connected' | 'connecting' | 'disconnected' = 'disconnected';
  let currentView: 'upload' | 'dashboard' | 'comparison' | 'analysis' = 'upload';
  let searchQuery = '';
  let filterEngine = 'all';
  let filterStatus = 'all';

  // Available OCR engines from backend
  const availableEngines = [
    { id: 'tesseract', name: 'Tesseract OCR', description: 'Traditional OCR with layout detection' },
    { id: 'easyocr', name: 'EasyOCR', description: 'Neural network-based with GPU support' },
    { id: 'trocr', name: 'TrOCR', description: 'Transformer-based OCR for complex documents' },
    { id: 'hybrid', name: 'Hybrid Mode', description: 'Combines all engines for best accuracy' }
  ];

  onMount(() => {
    connectWebSocket();
    loadInitialData();
    
    // Load saved jobs from localStorage
    const savedJobs = localStorage.getItem('bev-ocr-jobs');
    if (savedJobs) {
      try {
        const jobs = JSON.parse(savedJobs);
        completedJobs.set(jobs);
        updateStats();
      } catch (e) {
        console.warn('Failed to load saved OCR jobs:', e);
      }
    }
  });

  onDestroy(() => {
    if (websocket) {
      websocket.close();
    }
  });

  async function loadInitialData() {
    try {
      const stats = await invoke('get_ocr_stats');
      ocrStats.set(stats);
    } catch (error) {
      console.error('Failed to load OCR stats:', error);
    }
  }

  function connectWebSocket() {
    try {
      connectionStatus = 'connecting';
      websocket = new WebSocket('ws://localhost:3020/ocr-stream');

      websocket.onopen = () => {
        connectionStatus = 'connected';
        console.log('Connected to OCR processing stream');
      };

      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleOCRUpdate(data);
      };

      websocket.onerror = (error) => {
        console.error('OCR WebSocket error:', error);
        connectionStatus = 'disconnected';
      };

      websocket.onclose = () => {
        connectionStatus = 'disconnected';
        // Attempt reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };
    } catch (error) {
      console.error('Failed to connect OCR WebSocket:', error);
      connectionStatus = 'disconnected';
    }
  }

  function handleOCRUpdate(data: any) {
    switch (data.type) {
      case 'job_started':
        updateJobStatus(data.jobId, 'processing', data.progress || 0);
        break;
      case 'progress_update':
        updateJobStatus(data.jobId, 'processing', data.progress);
        break;
      case 'job_completed':
        completeJob(data.jobId, data.results);
        break;
      case 'job_failed':
        updateJobStatus(data.jobId, 'failed', 100);
        break;
    }
  }

  function updateJobStatus(jobId: string, status: string, progress: number) {
    activeJobs.update(jobs => 
      jobs.map(job => 
        job.id === jobId 
          ? { ...job, status: status as any, progress }
          : job
      )
    );
  }

  function completeJob(jobId: string, results: OCRResult[]) {
    let completedJob: OCRJob | null = null;
    
    activeJobs.update(jobs => {
      const updatedJobs = jobs.filter(job => {
        if (job.id === jobId) {
          completedJob = { ...job, status: 'completed', results, progress: 100 };
          return false;
        }
        return true;
      });
      return updatedJobs;
    });

    if (completedJob) {
      completedJobs.update(jobs => [completedJob!, ...jobs]);
      saveJobsToStorage();
      updateStats();
    }
  }

  function saveJobsToStorage() {
    completedJobs.subscribe(jobs => {
      localStorage.setItem('bev-ocr-jobs', JSON.stringify(jobs.slice(0, 100)));
    })();
  }

  function updateStats() {
    completedJobs.subscribe(jobs => {
      const successful = jobs.filter(j => j.status === 'completed');
      const failed = jobs.filter(j => j.status === 'failed');
      
      const avgTime = successful.length > 0 
        ? successful.reduce((sum, job) => {
            const result = job.results?.[0];
            return sum + (result?.processingTime || 0);
          }, 0) / successful.length
        : 0;

      const avgConfidence = successful.length > 0
        ? successful.reduce((sum, job) => {
            const result = job.results?.[0];
            return sum + (result?.confidence || 0);
          }, 0) / successful.length
        : 0;

      ocrStats.set({
        totalJobs: jobs.length,
        successfulJobs: successful.length,
        failedJobs: failed.length,
        avgProcessingTime: avgTime,
        avgConfidence: avgConfidence,
        totalDocuments: jobs.reduce((sum, job) => sum + (job.results?.length || 0), 0)
      });
    })();
  }

  async function handleFileUpload(files: File[], engines: string[]) {
    for (const file of files) {
      const jobId = `ocr_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      const job: OCRJob = {
        id: jobId,
        filename: file.name,
        status: 'pending',
        engines,
        progress: 0,
        timestamp: new Date(),
        fileSize: file.size,
        fileType: file.type
      };

      activeJobs.update(jobs => [...jobs, job]);

      try {
        // Convert file to base64 for Tauri IPC
        const reader = new FileReader();
        reader.onload = async () => {
          try {
            await invoke('process_ocr_file', {
              jobId,
              filename: file.name,
              fileData: reader.result,
              engines,
              options: {
                preprocess: true,
                extractLayout: true,
                extractMetadata: true
              }
            });
          } catch (error) {
            console.error('OCR processing failed:', error);
            updateJobStatus(jobId, 'failed', 100);
          }
        };
        reader.readAsDataURL(file);
      } catch (error) {
        console.error('File processing failed:', error);
        updateJobStatus(jobId, 'failed', 100);
      }
    }
  }

  function selectJob(job: OCRJob) {
    selectedJob.set(job);
    if (job.status === 'completed') {
      currentView = 'comparison';
    }
  }

  function clearCompletedJobs() {
    completedJobs.set([]);
    localStorage.removeItem('bev-ocr-jobs');
    updateStats();
  }

  async function exportResults() {
    const jobs = $completedJobs.filter(job => 
      job.status === 'completed' && job.results
    );

    const exportData = {
      exportDate: new Date().toISOString(),
      totalJobs: jobs.length,
      jobs: jobs.map(job => ({
        id: job.id,
        filename: job.filename,
        timestamp: job.timestamp,
        results: job.results
      }))
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bev-ocr-export-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  $: filteredJobs = $completedJobs.filter(job => {
    const matchesSearch = job.filename.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesEngine = filterEngine === 'all' || job.engines.includes(filterEngine);
    const matchesStatus = filterStatus === 'all' || job.status === filterStatus;
    return matchesSearch && matchesEngine && matchesStatus;
  });
</script>

<div class="ocr-platform min-h-screen bg-dark-bg-primary text-dark-text-primary">
  <!-- Header -->
  <div class="border-b border-dark-border bg-dark-bg-secondary">
    <div class="container mx-auto px-6 py-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <h1 class="text-2xl font-bold text-green-400">OCR INTELLIGENCE PLATFORM</h1>
          <Badge variant={connectionStatus === 'connected' ? 'success' : 'danger'}>
            {connectionStatus.toUpperCase()}
          </Badge>
        </div>
        
        <div class="flex items-center gap-3">
          <!-- View Toggle -->
          <div class="flex bg-dark-bg-tertiary rounded-lg p-1">
            {#each ['upload', 'dashboard', 'comparison', 'analysis'] as view}
              <button
                class="px-3 py-1 text-xs font-medium rounded transition-colors {
                  currentView === view 
                    ? 'bg-green-600 text-black' 
                    : 'text-dark-text-secondary hover:text-dark-text-primary'
                }"
                on:click={() => currentView = view}
              >
                {view.toUpperCase()}
              </button>
            {/each}
          </div>
          
          <Button variant="outline" size="sm" on:click={exportResults}>
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            EXPORT
          </Button>
        </div>
      </div>
    </div>
  </div>

  <!-- Stats Bar -->
  <div class="bg-dark-bg-secondary border-b border-dark-border">
    <div class="container mx-auto px-6 py-3">
      <div class="grid grid-cols-2 md:grid-cols-6 gap-4 text-center">
        <div>
          <div class="text-xs text-dark-text-tertiary">TOTAL JOBS</div>
          <div class="text-lg font-bold text-green-400">{$ocrStats.totalJobs}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">SUCCESS RATE</div>
          <div class="text-lg font-bold text-green-400">
            {$ocrStats.totalJobs > 0 ? (($ocrStats.successfulJobs / $ocrStats.totalJobs) * 100).toFixed(1) : 0}%
          </div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">AVG CONFIDENCE</div>
          <div class="text-lg font-bold text-cyan-400">
            {($ocrStats.avgConfidence * 100).toFixed(1)}%
          </div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">AVG TIME</div>
          <div class="text-lg font-bold text-purple-400">
            {$ocrStats.avgProcessingTime.toFixed(0)}ms
          </div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">ACTIVE</div>
          <div class="text-lg font-bold text-yellow-400">{$activeJobs.length}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">DOCUMENTS</div>
          <div class="text-lg font-bold text-blue-400">{$ocrStats.totalDocuments}</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Content -->
  <div class="container mx-auto px-6 py-6">
    {#if currentView === 'upload'}
      <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <!-- File Upload Section -->
        <div class="xl:col-span-2">
          <Card variant="bordered">
            <div class="p-6">
              <h2 class="text-lg font-semibold mb-4 text-dark-text-primary">Document Upload & Processing</h2>
              <OCRUploader 
                {availableEngines}
                on:filesSelected={(e) => handleFileUpload(e.detail.files, e.detail.engines)}
              />
            </div>
          </Card>
        </div>

        <!-- Active Jobs Panel -->
        <div>
          <Card variant="bordered">
            <div class="p-6">
              <h3 class="text-md font-semibold mb-4 text-dark-text-primary">Active Processing</h3>
              <div class="space-y-3 max-h-96 overflow-y-auto">
                {#each $activeJobs as job}
                  <div class="p-3 bg-dark-bg-tertiary rounded border border-dark-border">
                    <div class="flex items-center justify-between mb-2">
                      <span class="text-sm font-medium text-dark-text-primary truncate">
                        {job.filename}
                      </span>
                      <Badge variant={
                        job.status === 'processing' ? 'warning' : 
                        job.status === 'completed' ? 'success' : 
                        job.status === 'failed' ? 'danger' : 'info'
                      } size="xs">
                        {job.status.toUpperCase()}
                      </Badge>
                    </div>
                    
                    <div class="w-full bg-dark-bg-primary rounded-full h-2 mb-2">
                      <div 
                        class="bg-green-600 h-2 rounded-full transition-all duration-300"
                        style="width: {job.progress}%"
                      ></div>
                    </div>
                    
                    <div class="flex items-center justify-between text-xs text-dark-text-tertiary">
                      <span>{job.engines.join(', ')}</span>
                      <span>{job.progress}%</span>
                    </div>
                  </div>
                {/each}
                
                {#if $activeJobs.length === 0}
                  <div class="text-center py-8 text-dark-text-tertiary">
                    <svg class="w-12 h-12 mx-auto mb-3 text-dark-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <p>No active processing jobs</p>
                  </div>
                {/if}
              </div>
            </div>
          </Card>
        </div>
      </div>
    {/if}

    {#if currentView === 'dashboard'}
      <OCRDashboard 
        jobs={$completedJobs}
        activeJobs={$activeJobs}
        stats={$ocrStats}
        {searchQuery}
        {filterEngine}
        {filterStatus}
        on:jobSelected={(e) => selectJob(e.detail)}
        on:clearJobs={clearCompletedJobs}
        bind:searchQuery
        bind:filterEngine
        bind:filterStatus
      />
    {/if}

    {#if currentView === 'comparison' && $selectedJob}
      <OCRComparison 
        job={$selectedJob}
        on:backToDashboard={() => currentView = 'dashboard'}
      />
    {/if}

    {#if currentView === 'analysis'}
      <DocumentAnalysis 
        jobs={filteredJobs}
        on:jobSelected={(e) => selectJob(e.detail)}
        on:viewComparison={(e) => {
          selectJob(e.detail);
          currentView = 'comparison';
        }}
      />
    {/if}
  </div>
</div>

<style>
  .ocr-platform {
    background: var(--dark-bg-primary, #0a0a0a);
    color: var(--dark-text-primary, #00ff41);
    min-height: 100vh;
  }

  /* Ensure proper dark theme variables */
  :global(.dark-bg-primary) { background-color: #0a0a0a; }
  :global(.dark-bg-secondary) { background-color: #1a1a1a; }
  :global(.dark-bg-tertiary) { background-color: #0f0f0f; }
  :global(.dark-text-primary) { color: #00ff41; }
  :global(.dark-text-secondary) { color: #00ff4199; }
  :global(.dark-text-tertiary) { color: #00ff4166; }
  :global(.dark-border) { border-color: #00ff4133; }
</style>