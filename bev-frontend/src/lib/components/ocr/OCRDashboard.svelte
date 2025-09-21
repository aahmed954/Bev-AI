<!-- OCR Processing Dashboard with Real-time Status -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  const dispatch = createEventDispatcher();
  
  export let jobs = [];
  export let activeJobs = [];
  export let stats = {};
  export let searchQuery = '';
  export let filterEngine = 'all';
  export let filterStatus = 'all';

  const statusColors = {
    pending: 'info',
    processing: 'warning', 
    completed: 'success',
    failed: 'danger'
  };

  const engineColors = {
    tesseract: '#00ff41',
    easyocr: '#00ccff', 
    trocr: '#ff9500',
    hybrid: '#ff00ff'
  };

  function selectJob(job) {
    dispatch('jobSelected', job);
  }

  function clearJobs() {
    dispatch('clearJobs');
  }

  function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
  }

  function formatDuration(ms) {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  }

  function getEngineIcon(engine) {
    const icons = {
      tesseract: '📝',
      easyocr: '🧠', 
      trocr: '🤖',
      hybrid: '⚡'
    };
    return icons[engine] || '🔍';
  }

  function downloadResult(job) {
    const exportData = {
      filename: job.filename,
      timestamp: job.timestamp,
      engines: job.engines,
      results: job.results,
      metadata: {
        fileSize: job.fileSize,
        fileType: job.fileType,
        processingTime: job.results?.reduce((sum, r) => sum + r.processingTime, 0) || 0
      }
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ocr-result-${job.filename}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function copyText(text) {
    navigator.clipboard.writeText(text).then(() => {
      // Could add a toast notification here
      console.log('Text copied to clipboard');
    });
  }

  $: filteredJobs = jobs.filter(job => {
    const matchesSearch = job.filename.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesEngine = filterEngine === 'all' || job.engines.includes(filterEngine);
    const matchesStatus = filterStatus === 'all' || job.status === filterStatus;
    return matchesSearch && matchesEngine && matchesStatus;
  });
</script>

<div class="ocr-dashboard space-y-6">
  <!-- Filters and Search -->
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
        <div class="flex-1 max-w-md">
          <input
            type="text"
            placeholder="Search by filename..."
            bind:value={searchQuery}
            class="w-full px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary placeholder-dark-text-tertiary focus:border-green-500 focus:outline-none"
          />
        </div>
        
        <div class="flex gap-3">
          <select 
            bind:value={filterEngine}
            class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary focus:border-green-500 focus:outline-none"
          >
            <option value="all">All Engines</option>
            <option value="tesseract">Tesseract</option>
            <option value="easyocr">EasyOCR</option>
            <option value="trocr">TrOCR</option>
            <option value="hybrid">Hybrid</option>
          </select>
          
          <select 
            bind:value={filterStatus}
            class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary focus:border-green-500 focus:outline-none"
          >
            <option value="all">All Status</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="processing">Processing</option>
          </select>
          
          <Button variant="outline" size="sm" on:click={clearJobs}>
            Clear History
          </Button>
        </div>
      </div>
    </div>
  </Card>

  <!-- Jobs Grid -->
  <div class="jobs-grid grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
    {#each filteredJobs as job}
      <Card variant="bordered" class="job-card cursor-pointer hover:border-green-500 transition-colors">
        <div class="p-4" on:click={() => selectJob(job)}>
          <!-- Job Header -->
          <div class="flex items-start justify-between mb-3">
            <div class="flex-1 min-w-0">
              <h3 class="text-sm font-medium text-dark-text-primary truncate mb-1">
                {job.filename}
              </h3>
              <div class="text-xs text-dark-text-tertiary">
                {formatTimestamp(job.timestamp)}
              </div>
            </div>
            
            <Badge variant={statusColors[job.status]} size="xs">
              {job.status.toUpperCase()}
            </Badge>
          </div>

          <!-- Engines Used -->
          <div class="engines mb-3">
            <div class="text-xs text-dark-text-tertiary mb-1">Engines:</div>
            <div class="flex gap-1 flex-wrap">
              {#each job.engines as engine}
                <span 
                  class="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium"
                  style="background-color: {engineColors[engine]}20; color: {engineColors[engine]}"
                >
                  {getEngineIcon(engine)} {engine.toUpperCase()}
                </span>
              {/each}
            </div>
          </div>

          <!-- Results Summary -->
          {#if job.results && job.results.length > 0}
            <div class="results-summary mb-3">
              <div class="text-xs text-dark-text-tertiary mb-2">Results:</div>
              <div class="space-y-1">
                {#each job.results as result}
                  <div class="flex items-center justify-between text-xs">
                    <span class="text-dark-text-secondary">{result.engine}</span>
                    <div class="flex items-center gap-2">
                      <span class="confidence" style="color: {result.confidence > 0.8 ? '#00ff41' : result.confidence > 0.6 ? '#ffff00' : '#ff9500'}">
                        {(result.confidence * 100).toFixed(1)}%
                      </span>
                      <span class="text-dark-text-tertiary">
                        {formatDuration(result.processingTime)}
                      </span>
                    </div>
                  </div>
                {/each}
              </div>
            </div>

            <!-- Quick Actions -->
            <div class="actions flex gap-2">
              <button 
                class="flex-1 px-2 py-1 text-xs bg-dark-bg-primary hover:bg-green-600/20 border border-dark-border hover:border-green-500 rounded transition-colors"
                on:click|stopPropagation={() => copyText(job.results[0]?.text || '')}
              >
                📋 Copy Text
              </button>
              <button 
                class="flex-1 px-2 py-1 text-xs bg-dark-bg-primary hover:bg-blue-600/20 border border-dark-border hover:border-blue-500 rounded transition-colors"
                on:click|stopPropagation={() => downloadResult(job)}
              >
                💾 Download
              </button>
            </div>
          {:else if job.status === 'processing'}
            <!-- Progress Bar -->
            <div class="progress mb-3">
              <div class="text-xs text-dark-text-tertiary mb-1">
                Processing... {job.progress}%
              </div>
              <div class="w-full bg-dark-bg-primary rounded-full h-2">
                <div 
                  class="bg-green-600 h-2 rounded-full transition-all duration-300"
                  style="width: {job.progress}%"
                ></div>
              </div>
            </div>
          {:else if job.status === 'failed'}
            <!-- Error State -->
            <div class="error-state text-center py-4">
              <svg class="w-8 h-8 text-red-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div class="text-xs text-red-400">Processing failed</div>
            </div>
          {/if}

          <!-- File Metadata -->
          <div class="metadata mt-3 pt-3 border-t border-dark-border">
            <div class="grid grid-cols-2 gap-2 text-xs">
              <div>
                <span class="text-dark-text-tertiary">Size:</span>
                <span class="text-dark-text-secondary ml-1">
                  {(job.fileSize / 1024).toFixed(1)} KB
                </span>
              </div>
              <div>
                <span class="text-dark-text-tertiary">Type:</span>
                <span class="text-dark-text-secondary ml-1">
                  {job.fileType?.split('/')[1]?.toUpperCase() || 'Unknown'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </Card>
    {/each}
  </div>

  <!-- Empty State -->
  {#if filteredJobs.length === 0}
    <Card variant="bordered">
      <div class="p-12 text-center">
        <svg class="w-16 h-16 text-dark-text-tertiary mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <h3 class="text-lg font-medium text-dark-text-primary mb-2">No OCR jobs found</h3>
        <p class="text-sm text-dark-text-secondary mb-4">
          {searchQuery || filterEngine !== 'all' || filterStatus !== 'all' 
            ? 'No jobs match your current filters' 
            : 'Upload documents to start OCR processing'}
        </p>
        <Button variant="primary" on:click={() => dispatch('switchToUpload')}>
          Upload Documents
        </Button>
      </div>
    </Card>
  {/if}

  <!-- Bulk Actions (when jobs selected) -->
  {#if filteredJobs.length > 0}
    <Card variant="bordered">
      <div class="p-4">
        <div class="flex items-center justify-between">
          <div class="text-sm text-dark-text-secondary">
            Showing {filteredJobs.length} of {jobs.length} jobs
          </div>
          
          <div class="flex gap-2">
            <Button variant="outline" size="sm" on:click={() => {
              const allText = filteredJobs
                .filter(job => job.results)
                .map(job => `// ${job.filename}\n${job.results[0]?.text || ''}`)
                .join('\n\n');
              copyText(allText);
            }}>
              Copy All Text
            </Button>
            
            <Button variant="outline" size="sm" on:click={() => {
              const exportData = {
                exportDate: new Date().toISOString(),
                totalJobs: filteredJobs.length,
                filters: { searchQuery, filterEngine, filterStatus },
                jobs: filteredJobs
              };
              
              const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
                type: 'application/json' 
              });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `bev-ocr-bulk-export-${Date.now()}.json`;
              a.click();
              URL.revokeObjectURL(url);
            }}>
              Export Filtered
            </Button>
          </div>
        </div>
      </div>
    </Card>
  {/if}
</div>

<style>
  .job-card:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 255, 65, 0.1);
  }

  .confidence {
    font-weight: 600;
  }

  .jobs-grid {
    min-height: 300px;
  }
</style>