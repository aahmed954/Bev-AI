<!-- File Metadata Extraction Interface -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  const dispatch = createEventDispatcher();
  
  export let analyzer;
  export let activeJobs = [];
  export let completedJobs = [];
  
  let dragActive = false;
  let selectedFiles = [];
  let extractionOptions = {
    extractEXIF: true,
    extractXMP: true,
    extractIPTC: true,
    detectSteganography: false,
    analyzeStructure: true,
    extractText: false,
    checkSignatures: true
  };

  const supportedTypes = [
    'image/jpeg', 'image/png', 'image/tiff', 'image/bmp',
    'application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'video/mp4', 'video/avi', 'audio/mp3', 'audio/wav'
  ];

  function handleDragOver(e) {
    e.preventDefault();
    dragActive = true;
  }

  function handleDragLeave(e) {
    e.preventDefault();
    dragActive = false;
  }

  function handleDrop(e) {
    e.preventDefault();
    dragActive = false;
    const files = Array.from(e.dataTransfer.files);
    addFiles(files);
  }

  function addFiles(newFiles) {
    const validFiles = newFiles.filter(file => 
      supportedTypes.includes(file.type) && file.size <= 100 * 1024 * 1024
    );
    selectedFiles = [...selectedFiles, ...validFiles].slice(0, 10);
  }

  function removeFile(index) {
    selectedFiles = selectedFiles.filter((_, i) => i !== index);
  }

  function startExtraction() {
    if (selectedFiles.length === 0) return;
    
    selectedFiles.forEach(file => {
      const reader = new FileReader();
      reader.onload = () => {
        dispatch('startAnalysis', {
          target: file.name,
          options: {
            type: 'file',
            fileData: reader.result,
            fileType: file.type,
            fileSize: file.size,
            ...extractionOptions
          }
        });
      };
      reader.readAsDataURL(file);
    });
    
    selectedFiles = [];
  }

  function getFileIcon(type) {
    if (type.startsWith('image/')) return '🖼️';
    if (type.startsWith('video/')) return '🎥';
    if (type.startsWith('audio/')) return '🎵';
    if (type.includes('pdf')) return '📄';
    if (type.includes('word') || type.includes('document')) return '📝';
    return '📎';
  }

  function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function renderMetadataValue(value) {
    if (typeof value === 'object') {
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  }
</script>

<div class="metadata-extractor space-y-6">
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center gap-3">
        <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>← Back</Button>
        <span class="text-2xl">📋</span>
        <h2 class="text-lg font-semibold text-dark-text-primary">Metadata Extraction Tool</h2>
      </div>
    </div>
  </Card>

  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    <div class="xl:col-span-2">
      <!-- File Upload -->
      <Card variant="bordered">
        <div class="p-6">
          <div 
            class="upload-zone border-2 border-dashed rounded-lg p-8 text-center transition-all {
              dragActive ? 'border-green-500 bg-green-500/10' : 'border-dark-border'
            }"
            on:dragover={handleDragOver}
            on:dragleave={handleDragLeave}
            on:drop={handleDrop}
          >
            <svg class="w-16 h-16 text-dark-text-tertiary mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width={1} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
            </svg>
            <h3 class="text-lg font-medium text-dark-text-primary mb-2">Drop files for metadata extraction</h3>
            <p class="text-sm text-dark-text-secondary">Images, videos, documents, audio files</p>
          </div>

          {#if selectedFiles.length > 0}
            <div class="selected-files mt-6">
              <h4 class="text-sm font-medium text-dark-text-primary mb-3">Selected Files ({selectedFiles.length})</h4>
              <div class="space-y-2">
                {#each selectedFiles as file, index}
                  <div class="file-item flex items-center gap-3 p-3 bg-dark-bg-tertiary rounded border border-dark-border">
                    <span class="text-lg">{getFileIcon(file.type)}</span>
                    <div class="flex-1 min-w-0">
                      <div class="text-sm font-medium text-dark-text-primary truncate">{file.name}</div>
                      <div class="text-xs text-dark-text-tertiary">{formatFileSize(file.size)} • {file.type}</div>
                    </div>
                    <button 
                      class="text-red-400 hover:text-red-300"
                      on:click={() => removeFile(index)}
                    >
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                {/each}
              </div>
              
              <div class="mt-4 flex justify-between">
                <Button variant="outline" on:click={() => selectedFiles = []}>Clear All</Button>
                <Button variant="primary" on:click={startExtraction}>
                  Extract Metadata ({selectedFiles.length})
                </Button>
              </div>
            </div>
          {/if}

          <!-- Extraction Options -->
          <div class="extraction-options mt-6">
            <h4 class="text-sm font-medium text-dark-text-primary mb-3">Extraction Options</h4>
            <div class="grid grid-cols-2 gap-3">
              <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={extractionOptions.extractEXIF} class="checkbox" />
                <span class="text-sm text-dark-text-secondary">EXIF Data</span>
              </label>
              <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={extractionOptions.extractXMP} class="checkbox" />
                <span class="text-sm text-dark-text-secondary">XMP Metadata</span>
              </label>
              <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={extractionOptions.detectSteganography} class="checkbox" />
                <span class="text-sm text-dark-text-secondary">Steganography</span>
              </label>
              <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={extractionOptions.checkSignatures} class="checkbox" />
                <span class="text-sm text-dark-text-secondary">Digital Signatures</span>
              </label>
            </div>
          </div>
        </div>
      </Card>

      <!-- Results -->
      {#if completedJobs.length > 0}
        <Card variant="bordered">
          <div class="p-6">
            <h3 class="text-md font-medium text-dark-text-primary mb-4">Metadata Results</h3>
            {#each completedJobs.filter(job => job.results).slice(0, 3) as job}
              <div class="metadata-result mb-4 p-4 bg-dark-bg-tertiary rounded border border-dark-border">
                <div class="flex items-center justify-between mb-3">
                  <h4 class="text-sm font-semibold text-dark-text-primary">{job.target}</h4>
                  <div class="text-xs text-dark-text-tertiary">
                    {new Date(job.startTime).toLocaleDateString()}
                  </div>
                </div>

                {#if job.results.metadata}
                  <div class="metadata-grid grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                    {#each Object.entries(job.results.metadata).slice(0, 8) as [key, value]}
                      <div class="metadata-item">
                        <div class="text-dark-text-tertiary">{key}:</div>
                        <div class="text-dark-text-secondary truncate">{renderMetadataValue(value)}</div>
                      </div>
                    {/each}
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </Card>
      {/if}
    </div>

    <!-- Quick Stats -->
    <div>
      <Card variant="bordered">
        <div class="p-4">
          <h4 class="text-sm font-medium text-dark-text-primary mb-3">Processing Stats</h4>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span class="text-dark-text-tertiary">Files Processed:</span>
              <span class="text-dark-text-primary">{completedJobs.length}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-dark-text-tertiary">Active Jobs:</span>
              <span class="text-yellow-400">{activeJobs.length}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-dark-text-tertiary">Success Rate:</span>
              <span class="text-green-400">
                {completedJobs.length > 0 ? 
                  ((completedJobs.filter(j => !j.error).length / completedJobs.length) * 100).toFixed(1) 
                  : 0}%
              </span>
            </div>
          </div>
        </div>
      </Card>
    </div>
  </div>
</div>

<style>
  .checkbox {
    @apply w-4 h-4 rounded border-2 border-dark-border;
    background: var(--dark-bg-tertiary, #0f0f0f);
    accent-color: #00ff41;
  }
</style>