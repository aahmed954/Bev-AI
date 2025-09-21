<!-- Custom OSINT Analyzer Platform -->
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
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import BreachLookup from '$lib/components/analyzers/BreachLookup.svelte';
  import ReputationScoring from '$lib/components/analyzers/ReputationScoring.svelte';
  import MetadataExtractor from '$lib/components/analyzers/MetadataExtractor.svelte';
  import WatermarkAnalyzer from '$lib/components/analyzers/WatermarkAnalyzer.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  interface AnalyzerStatus {
    id: string;
    name: string;
    status: 'online' | 'offline' | 'error';
    activeJobs: number;
    totalJobs: number;
    successRate: number;
    avgResponseTime: number;
    lastUsed: string;
    capabilities: string[];
  }

  interface AnalysisJob {
    id: string;
    analyzer: string;
    target: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    results?: any;
    startTime: string;
    endTime?: string;
    error?: string;
  }

  let currentAnalyzer: 'breach' | 'reputation' | 'metadata' | 'watermark' | 'overview' = 'overview';
  let analyzers = writable<AnalyzerStatus[]>([]);
  let activeJobs = writable<AnalysisJob[]>([]);
  let completedJobs = writable<AnalysisJob[]>([]);
  let websocket: WebSocket | null = null;
  let connectionStatus: 'connected' | 'connecting' | 'disconnected' = 'disconnected';

  // Available analyzers with their capabilities
  const analyzerConfigs = {
    breach: {
      name: 'Breach Database Analyzer',
      icon: '🔓',
      color: '#ff0066',
      description: 'Search breach databases for compromised credentials',
      capabilities: ['dehashed', 'snusbase', 'weleakinfo', 'hibp'],
      targetTypes: ['email', 'username', 'domain']
    },
    reputation: {
      name: 'Reputation Scoring Engine',
      icon: '🎯',
      color: '#ff9500',
      description: 'Analyze threat reputation and risk scoring',
      capabilities: ['ip_reputation', 'domain_reputation', 'file_reputation', 'url_reputation'],
      targetTypes: ['ip', 'domain', 'url', 'hash']
    },
    metadata: {
      name: 'Metadata Extraction Tool',
      icon: '📋',
      color: '#00ccff',
      description: 'Extract metadata from files and documents',
      capabilities: ['exif', 'document_properties', 'file_analysis', 'steganography'],
      targetTypes: ['file', 'image', 'document']
    },
    watermark: {
      name: 'Watermark Detection System',
      icon: '🔍',
      color: '#ffff00',
      description: 'Detect and analyze digital watermarks',
      capabilities: ['visible_watermarks', 'invisible_watermarks', 'digital_signatures'],
      targetTypes: ['image', 'video', 'audio', 'document']
    }
  };

  onMount(() => {
    loadAnalyzerStatuses();
    connectWebSocket();
    loadJobHistory();
  });

  onDestroy(() => {
    if (websocket) {
      websocket.close();
    }
  });

  async function loadAnalyzerStatuses() {
    try {
      const statuses = await invoke('get_analyzer_statuses');
      analyzers.set(statuses);
    } catch (error) {
      console.error('Failed to load analyzer statuses:', error);
      // Generate mock data
      generateMockAnalyzerData();
    }
  }

  function generateMockAnalyzerData() {
    const mockAnalyzers = Object.keys(analyzerConfigs).map(id => ({
      id,
      name: analyzerConfigs[id].name,
      status: Math.random() > 0.2 ? 'online' : 'offline',
      activeJobs: Math.floor(Math.random() * 5),
      totalJobs: Math.floor(Math.random() * 1000) + 100,
      successRate: 85 + Math.random() * 10,
      avgResponseTime: 500 + Math.random() * 2000,
      lastUsed: new Date(Date.now() - Math.random() * 86400000).toISOString(),
      capabilities: analyzerConfigs[id].capabilities
    }));

    analyzers.set(mockAnalyzers);
  }

  function connectWebSocket() {
    try {
      connectionStatus = 'connecting';
      websocket = new WebSocket('ws://${getWebSocketHost()}:3010/analyzer-stream');

      websocket.onopen = () => {
        connectionStatus = 'connected';
        console.log('Connected to analyzer monitoring stream');
      };

      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleAnalyzerUpdate(data);
      };

      websocket.onerror = (error) => {
        console.error('Analyzer WebSocket error:', error);
        connectionStatus = 'disconnected';
      };

      websocket.onclose = () => {
        connectionStatus = 'disconnected';
        setTimeout(connectWebSocket, 3000);
      };
    } catch (error) {
      console.error('Failed to connect analyzer WebSocket:', error);
      connectionStatus = 'disconnected';
    }
  }

  function handleAnalyzerUpdate(data: any) {
    switch (data.type) {
      case 'job_started':
        updateJobStatus(data.jobId, 'running', data.progress || 0);
        break;
      case 'job_progress':
        updateJobStatus(data.jobId, 'running', data.progress);
        break;
      case 'job_completed':
        completeJob(data.jobId, data.results);
        break;
      case 'job_failed':
        updateJobStatus(data.jobId, 'failed', 100, data.error);
        break;
      case 'analyzer_status':
        updateAnalyzerStatus(data.analyzerId, data.status);
        break;
    }
  }

  function updateJobStatus(jobId: string, status: string, progress: number, error?: string) {
    activeJobs.update(jobs => 
      jobs.map(job => 
        job.id === jobId 
          ? { ...job, status: status as any, progress, error }
          : job
      )
    );
  }

  function completeJob(jobId: string, results: any) {
    let completedJob: AnalysisJob | null = null;
    
    activeJobs.update(jobs => {
      const updatedJobs = jobs.filter(job => {
        if (job.id === jobId) {
          completedJob = { 
            ...job, 
            status: 'completed', 
            results, 
            progress: 100,
            endTime: new Date().toISOString()
          };
          return false;
        }
        return true;
      });
      return updatedJobs;
    });

    if (completedJob) {
      completedJobs.update(jobs => [completedJob!, ...jobs.slice(0, 99)]);
      saveJobHistory();
    }
  }

  function updateAnalyzerStatus(analyzerId: string, status: any) {
    analyzers.update(analyzers => 
      analyzers.map(analyzer => 
        analyzer.id === analyzerId 
          ? { ...analyzer, ...status }
          : analyzer
      )
    );
  }

  function loadJobHistory() {
    const saved = localStorage.getItem('bev-analyzer-jobs');
    if (saved) {
      try {
        completedJobs.set(JSON.parse(saved));
      } catch (e) {
        console.warn('Failed to load analyzer job history:', e);
      }
    }
  }

  function saveJobHistory() {
    completedJobs.subscribe(jobs => {
      localStorage.setItem('bev-analyzer-jobs', JSON.stringify(jobs.slice(0, 100)));
    })();
  }

  async function startAnalysis(analyzerType: string, target: string, options: any = {}) {
    const jobId = `${analyzerType}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const job: AnalysisJob = {
      id: jobId,
      analyzer: analyzerType,
      target,
      status: 'pending',
      progress: 0,
      startTime: new Date().toISOString()
    };

    activeJobs.update(jobs => [...jobs, job]);

    try {
      await invoke('start_analyzer_job', {
        jobId,
        analyzer: analyzerType,
        target,
        options
      });
    } catch (error) {
      console.error('Failed to start analysis:', error);
      updateJobStatus(jobId, 'failed', 100, error.message);
    }
  }

  function getAnalyzerStats() {
    const total = $analyzers.length;
    const online = $analyzers.filter(a => a.status === 'online').length;
    const totalJobs = $activeJobs.length;
    const avgSuccessRate = $analyzers.reduce((sum, a) => sum + a.successRate, 0) / total;
    
    return { total, online, totalJobs, avgSuccessRate };
  }

  function exportAnalyzerReport() {
    const stats = getAnalyzerStats();
    const reportData = {
      timestamp: new Date().toISOString(),
      summary: stats,
      analyzers: $analyzers,
      activeJobs: $activeJobs,
      recentJobs: $completedJobs.slice(0, 50),
      connectionStatus
    };

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analyzer-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  $: stats = getAnalyzerStats();
</script>

<div class="analyzer-platform min-h-screen bg-dark-bg-primary text-dark-text-primary">
  <!-- Header -->
  <div class="border-b border-dark-border bg-dark-bg-secondary">
    <div class="container mx-auto px-6 py-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <h1 class="text-2xl font-bold text-green-400">OSINT ANALYZER PLATFORM</h1>
          <Badge variant={connectionStatus === 'connected' ? 'success' : 'danger'}>
            {connectionStatus.toUpperCase()}
          </Badge>
        </div>
        
        <div class="flex items-center gap-3">
          <!-- Analyzer Toggle -->
          <div class="flex bg-dark-bg-tertiary rounded-lg p-1">
            {#each ['overview', 'breach', 'reputation', 'metadata', 'watermark'] as analyzer}
              <button
                class="px-3 py-1 text-xs font-medium rounded transition-colors {
                  currentAnalyzer === analyzer 
                    ? 'bg-green-600 text-black' 
                    : 'text-dark-text-secondary hover:text-dark-text-primary'
                }"
                on:click={() => currentAnalyzer = analyzer}
              >
                {analyzer.toUpperCase()}
              </button>
            {/each}
          </div>
          
          <Button variant="outline" size="sm" on:click={exportAnalyzerReport}>
            Export Report
          </Button>
        </div>
      </div>
    </div>
  </div>

  <!-- Stats Bar -->
  <div class="bg-dark-bg-secondary border-b border-dark-border">
    <div class="container mx-auto px-6 py-3">
      <div class="grid grid-cols-2 md:grid-cols-5 gap-4 text-center">
        <div>
          <div class="text-xs text-dark-text-tertiary">ANALYZERS ONLINE</div>
          <div class="text-lg font-bold text-green-400">{stats.online}/{stats.total}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">ACTIVE JOBS</div>
          <div class="text-lg font-bold text-yellow-400">{stats.totalJobs}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">SUCCESS RATE</div>
          <div class="text-lg font-bold text-cyan-400">{stats.avgSuccessRate.toFixed(1)}%</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">COMPLETED</div>
          <div class="text-lg font-bold text-purple-400">{$completedJobs.length}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">AVG RESPONSE</div>
          <div class="text-lg font-bold text-blue-400">
            {($analyzers.reduce((sum, a) => sum + a.avgResponseTime, 0) / $analyzers.length).toFixed(0)}ms
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Content -->
  <div class="container mx-auto px-6 py-6">
    {#if currentAnalyzer === 'overview'}
      <!-- Analyzer Overview Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6 mb-6">
        {#each Object.entries(analyzerConfigs) as [id, config]}
          {@const analyzer = $analyzers.find(a => a.id === id)}
          
          <Card variant="bordered" class="analyzer-card cursor-pointer hover:border-green-500 transition-all">
            <div class="p-6" on:click={() => currentAnalyzer = id}>
              <!-- Analyzer Header -->
              <div class="flex items-center justify-between mb-4">
                <div class="flex items-center gap-3">
                  <span class="text-3xl">{config.icon}</span>
                  <div>
                    <h3 class="text-md font-semibold text-dark-text-primary">{config.name}</h3>
                    <div class="text-xs text-dark-text-secondary">
                      {config.capabilities.length} capabilities
                    </div>
                  </div>
                </div>
                
                <Badge variant={
                  analyzer?.status === 'online' ? 'success' : 
                  analyzer?.status === 'offline' ? 'warning' : 'danger'
                }>
                  {analyzer?.status?.toUpperCase() || 'UNKNOWN'}
                </Badge>
              </div>

              <!-- Description -->
              <p class="text-sm text-dark-text-secondary mb-4">{config.description}</p>

              <!-- Metrics -->
              {#if analyzer}
                <div class="grid grid-cols-2 gap-3 mb-4">
                  <div class="metric text-center">
                    <div class="text-xs text-dark-text-tertiary">Active Jobs</div>
                    <div class="text-lg font-bold text-yellow-400">{analyzer.activeJobs}</div>
                  </div>
                  <div class="metric text-center">
                    <div class="text-xs text-dark-text-tertiary">Success Rate</div>
                    <div class="text-lg font-bold text-green-400">{analyzer.successRate.toFixed(1)}%</div>
                  </div>
                  <div class="metric text-center">
                    <div class="text-xs text-dark-text-tertiary">Total Jobs</div>
                    <div class="text-sm font-medium text-cyan-400">{analyzer.totalJobs}</div>
                  </div>
                  <div class="metric text-center">
                    <div class="text-xs text-dark-text-tertiary">Avg Time</div>
                    <div class="text-sm font-medium text-purple-400">{analyzer.avgResponseTime.toFixed(0)}ms</div>
                  </div>
                </div>
              {/if}

              <!-- Capabilities -->
              <div class="capabilities">
                <div class="text-xs text-dark-text-tertiary mb-2">Capabilities:</div>
                <div class="flex flex-wrap gap-1">
                  {#each config.capabilities as capability}
                    <span class="px-2 py-1 text-xs bg-dark-bg-primary border border-dark-border rounded">
                      {capability.replace('_', ' ')}
                    </span>
                  {/each}
                </div>
              </div>

              <!-- Target Types -->
              <div class="target-types mt-3">
                <div class="text-xs text-dark-text-tertiary mb-1">Supports:</div>
                <div class="text-xs text-dark-text-secondary">
                  {config.targetTypes.join(', ')}
                </div>
              </div>
            </div>
          </Card>
        {/each}
      </div>

      <!-- Active Jobs Monitor -->
      {#if $activeJobs.length > 0}
        <Card variant="bordered">
          <div class="p-6">
            <h3 class="text-md font-medium text-dark-text-primary mb-4">Active Analysis Jobs</h3>
            <div class="space-y-3">
              {#each $activeJobs as job}
                <div class="job-item p-4 bg-dark-bg-tertiary rounded border border-dark-border">
                  <div class="flex items-center justify-between mb-3">
                    <div class="flex items-center gap-3">
                      <span class="text-xl">{analyzerConfigs[job.analyzer]?.icon || '🔍'}</span>
                      <div>
                        <div class="text-sm font-medium text-dark-text-primary">
                          {analyzerConfigs[job.analyzer]?.name || job.analyzer}
                        </div>
                        <div class="text-xs text-dark-text-secondary">
                          Target: {job.target}
                        </div>
                      </div>
                    </div>
                    
                    <div class="flex items-center gap-2">
                      <Badge variant={
                        job.status === 'running' ? 'warning' :
                        job.status === 'completed' ? 'success' :
                        job.status === 'failed' ? 'danger' : 'info'
                      } size="sm">
                        {job.status.toUpperCase()}
                      </Badge>
                      
                      {#if job.status === 'running'}
                        <Button variant="outline" size="xs" on:click={() => {
                          invoke('cancel_analyzer_job', { jobId: job.id })
                            .catch(error => console.error('Cancel failed:', error));
                        }}>
                          Cancel
                        </Button>
                      {/if}
                    </div>
                  </div>

                  <!-- Progress Bar -->
                  {#if job.status === 'running'}
                    <div class="progress mb-2">
                      <div class="flex justify-between text-xs mb-1">
                        <span class="text-dark-text-tertiary">Progress</span>
                        <span class="text-dark-text-secondary">{job.progress}%</span>
                      </div>
                      <div class="w-full bg-dark-bg-primary rounded-full h-2">
                        <div 
                          class="bg-green-600 h-2 rounded-full transition-all duration-300"
                          style="width: {job.progress}%"
                        ></div>
                      </div>
                    </div>
                  {/if}

                  <!-- Job Details -->
                  <div class="flex items-center justify-between text-xs text-dark-text-tertiary">
                    <span>Started: {new Date(job.startTime).toLocaleTimeString()}</span>
                    {#if job.endTime}
                      <span>Duration: {((new Date(job.endTime).getTime() - new Date(job.startTime).getTime()) / 1000).toFixed(1)}s</span>
                    {/if}
                  </div>

                  <!-- Error Display -->
                  {#if job.error}
                    <div class="error-message mt-2 p-2 bg-red-600/20 border border-red-500/30 rounded">
                      <div class="text-xs text-red-400">{job.error}</div>
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          </div>
        </Card>
      {/if}

      <!-- Quick Actions -->
      <Card variant="bordered">
        <div class="p-6">
          <h3 class="text-md font-medium text-dark-text-primary mb-4">Quick Analysis</h3>
          <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Button variant="outline" fullWidth on:click={() => currentAnalyzer = 'breach'}>
              <span class="text-lg mr-2">🔓</span>
              Breach Lookup
            </Button>
            <Button variant="outline" fullWidth on:click={() => currentAnalyzer = 'reputation'}>
              <span class="text-lg mr-2">🎯</span>
              Reputation Check
            </Button>
            <Button variant="outline" fullWidth on:click={() => currentAnalyzer = 'metadata'}>
              <span class="text-lg mr-2">📋</span>
              Metadata Extract
            </Button>
            <Button variant="outline" fullWidth on:click={() => currentAnalyzer = 'watermark'}>
              <span class="text-lg mr-2">🔍</span>
              Watermark Detect
            </Button>
          </div>
        </div>
      </Card>
    {/if}

    <!-- Specific Analyzer Interfaces -->
    {#if currentAnalyzer === 'breach'}
      <BreachLookup 
        analyzer={$analyzers.find(a => a.id === 'breach')}
        activeJobs={$activeJobs.filter(j => j.analyzer === 'breach')}
        completedJobs={$completedJobs.filter(j => j.analyzer === 'breach')}
        on:startAnalysis={(e) => startAnalysis('breach', e.detail.target, e.detail.options)}
        on:backToOverview={() => currentAnalyzer = 'overview'}
      />
    {/if}

    {#if currentAnalyzer === 'reputation'}
      <ReputationScoring 
        analyzer={$analyzers.find(a => a.id === 'reputation')}
        activeJobs={$activeJobs.filter(j => j.analyzer === 'reputation')}
        completedJobs={$completedJobs.filter(j => j.analyzer === 'reputation')}
        on:startAnalysis={(e) => startAnalysis('reputation', e.detail.target, e.detail.options)}
        on:backToOverview={() => currentAnalyzer = 'overview'}
      />
    {/if}

    {#if currentAnalyzer === 'metadata'}
      <MetadataExtractor 
        analyzer={$analyzers.find(a => a.id === 'metadata')}
        activeJobs={$activeJobs.filter(j => j.analyzer === 'metadata')}
        completedJobs={$completedJobs.filter(j => j.analyzer === 'metadata')}
        on:startAnalysis={(e) => startAnalysis('metadata', e.detail.target, e.detail.options)}
        on:backToOverview={() => currentAnalyzer = 'overview'}
      />
    {/if}

    {#if currentAnalyzer === 'watermark'}
      <WatermarkAnalyzer 
        analyzer={$analyzers.find(a => a.id === 'watermark')}
        activeJobs={$activeJobs.filter(j => j.analyzer === 'watermark')}
        completedJobs={$completedJobs.filter(j => j.analyzer === 'watermark')}
        on:startAnalysis={(e) => startAnalysis('watermark', e.detail.target, e.detail.options)}
        on:backToOverview={() => currentAnalyzer = 'overview'}
      />
    {/if}
  </div>
</div>

<style>
  .analyzer-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 255, 65, 0.15);
  }

  .job-item {
    @apply transition-colors hover:bg-dark-bg-secondary;
  }

  /* Ensure dark theme consistency */
  :global(.dark-bg-primary) { background-color: #0a0a0a; }
  :global(.dark-bg-secondary) { background-color: #1a1a1a; }
  :global(.dark-bg-tertiary) { background-color: #0f0f0f; }
  :global(.dark-text-primary) { color: #00ff41; }
  :global(.dark-text-secondary) { color: #00ff4199; }
  :global(.dark-text-tertiary) { color: #00ff4166; }
  :global(.dark-border) { border-color: #00ff4133; }
</style>