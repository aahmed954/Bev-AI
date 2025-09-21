<!-- OCR File Upload Component with Drag & Drop -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  const dispatch = createEventDispatcher();
  
  export let availableEngines: Array<{id: string, name: string, description: string}> = [];
  
  let files = writable<File[]>([]);
  let selectedEngines = writable<string[]>(['hybrid']);
  let dragActive = false;
  let fileInput: HTMLInputElement;
  let processingOptions = {
    preprocess: true,
    extractLayout: true,
    extractMetadata: true,
    enhanceQuality: false,
    detectLanguage: true
  };

  // Supported file types for OCR
  const supportedTypes = [
    'image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/bmp',
    'application/pdf', 'image/tiff', 'image/webp'
  ];

  const maxFileSize = 50 * 1024 * 1024; // 50MB
  const maxFiles = 10;

  onMount(() => {
    // Default to hybrid engine
    selectedEngines.set(['hybrid']);
  });

  function handleDragOver(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragActive = true;
  }

  function handleDragLeave(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragActive = false;
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    dragActive = false;

    const droppedFiles = Array.from(e.dataTransfer?.files || []);
    addFiles(droppedFiles);
  }

  function handleFileSelect(e: Event) {
    const target = e.target as HTMLInputElement;
    const selectedFiles = Array.from(target.files || []);
    addFiles(selectedFiles);
  }

  function addFiles(newFiles: File[]) {
    const validFiles = newFiles.filter(file => {
      // Check file type
      if (!supportedTypes.includes(file.type)) {
        console.warn(`Unsupported file type: ${file.type}`);
        return false;
      }

      // Check file size
      if (file.size > maxFileSize) {
        console.warn(`File too large: ${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`);
        return false;
      }

      return true;
    });

    files.update(currentFiles => {
      const combined = [...currentFiles, ...validFiles];
      return combined.slice(0, maxFiles); // Limit total files
    });
  }

  function removeFile(index: number) {
    files.update(currentFiles => 
      currentFiles.filter((_, i) => i !== index)
    );
  }

  function toggleEngine(engineId: string) {
    selectedEngines.update(engines => {
      if (engines.includes(engineId)) {
        // Don't allow removing all engines
        if (engines.length > 1) {
          return engines.filter(id => id !== engineId);
        }
        return engines;
      } else {
        return [...engines, engineId];
      }
    });
  }

  function processFiles() {
    if ($files.length === 0) {
      alert('Please select files to process');
      return;
    }

    if ($selectedEngines.length === 0) {
      alert('Please select at least one OCR engine');
      return;
    }

    dispatch('filesSelected', {
      files: $files,
      engines: $selectedEngines,
      options: processingOptions
    });

    // Clear files after processing
    files.set([]);
    if (fileInput) {
      fileInput.value = '';
    }
  }

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  function getFileTypeIcon(type: string): string {
    if (type.startsWith('image/')) return '🖼️';
    if (type === 'application/pdf') return '📄';
    return '📎';
  }
</script>

<div class="ocr-uploader">
  <!-- Drag & Drop Zone -->
  <div 
    class="upload-zone {dragActive ? 'drag-active' : ''}"
    on:dragover={handleDragOver}
    on:dragleave={handleDragLeave}
    on:drop={handleDrop}
    role="button"
    tabindex="0"
    on:click={() => fileInput?.click()}
    on:keydown={(e) => e.key === 'Enter' && fileInput?.click()}
  >
    <div class="upload-content">
      <div class="upload-icon">
        <svg class="w-16 h-16 text-dark-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width={1} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
      </div>
      
      <div class="upload-text">
        <h3 class="text-lg font-semibold text-dark-text-primary mb-2">
          Drop documents here or click to browse
        </h3>
        <p class="text-sm text-dark-text-secondary mb-1">
          Supports: PDF, PNG, JPG, GIF, BMP, TIFF, WebP
        </p>
        <p class="text-xs text-dark-text-tertiary">
          Max {maxFiles} files, up to {formatFileSize(maxFileSize)} each
        </p>
      </div>
    </div>
    
    <input
      bind:this={fileInput}
      type="file"
      multiple
      accept=".pdf,.png,.jpg,.jpeg,.gif,.bmp,.tiff,.webp"
      on:change={handleFileSelect}
      class="hidden"
    />
  </div>

  <!-- Selected Files List -->
  {#if $files.length > 0}
    <div class="files-list mt-6">
      <h4 class="text-md font-medium mb-3 text-dark-text-primary">Selected Files ({$files.length})</h4>
      <div class="space-y-2 max-h-48 overflow-y-auto">
        {#each $files as file, index}
          <div class="file-item flex items-center justify-between p-3 bg-dark-bg-tertiary rounded border border-dark-border">
            <div class="flex items-center gap-3 flex-1 min-w-0">
              <span class="text-lg">{getFileTypeIcon(file.type)}</span>
              <div class="flex-1 min-w-0">
                <div class="text-sm font-medium text-dark-text-primary truncate">
                  {file.name}
                </div>
                <div class="text-xs text-dark-text-tertiary">
                  {formatFileSize(file.size)} • {file.type}
                </div>
              </div>
            </div>
            
            <button
              class="text-red-400 hover:text-red-300 p-1"
              on:click={() => removeFile(index)}
              title="Remove file"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        {/each}
      </div>
    </div>
  {/if}

  <!-- OCR Engine Selection -->
  <div class="engine-selection mt-6">
    <h4 class="text-md font-medium mb-3 text-dark-text-primary">OCR Engines</h4>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
      {#each availableEngines as engine}
        <div 
          class="engine-option p-3 rounded border cursor-pointer transition-all {
            $selectedEngines.includes(engine.id) 
              ? 'border-green-500 bg-green-500/10' 
              : 'border-dark-border bg-dark-bg-tertiary hover:border-dark-text-tertiary'
          }"
          on:click={() => toggleEngine(engine.id)}
          role="checkbox"
          tabindex="0"
          aria-checked={$selectedEngines.includes(engine.id)}
          on:keydown={(e) => e.key === 'Enter' && toggleEngine(engine.id)}
        >
          <div class="flex items-start justify-between">
            <div class="flex-1">
              <div class="flex items-center gap-2 mb-1">
                <span class="text-sm font-medium text-dark-text-primary">{engine.name}</span>
                {#if $selectedEngines.includes(engine.id)}
                  <Badge variant="success" size="xs">SELECTED</Badge>
                {/if}
              </div>
              <p class="text-xs text-dark-text-secondary">{engine.description}</p>
            </div>
            
            <div class="ml-3">
              <div class="w-5 h-5 rounded border-2 flex items-center justify-center {
                $selectedEngines.includes(engine.id) 
                  ? 'border-green-500 bg-green-500' 
                  : 'border-dark-border'
              }">
                {#if $selectedEngines.includes(engine.id)}
                  <svg class="w-3 h-3 text-black" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                  </svg>
                {/if}
              </div>
            </div>
          </div>
        </div>
      {/each}
    </div>
  </div>

  <!-- Processing Options -->
  <div class="processing-options mt-6">
    <h4 class="text-md font-medium mb-3 text-dark-text-primary">Processing Options</h4>
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
      <label class="option-checkbox flex items-center gap-2 cursor-pointer">
        <input 
          type="checkbox" 
          bind:checked={processingOptions.preprocess}
          class="checkbox"
        />
        <span class="text-sm text-dark-text-secondary">Image Preprocessing</span>
      </label>
      
      <label class="option-checkbox flex items-center gap-2 cursor-pointer">
        <input 
          type="checkbox" 
          bind:checked={processingOptions.extractLayout}
          class="checkbox"
        />
        <span class="text-sm text-dark-text-secondary">Extract Layout</span>
      </label>
      
      <label class="option-checkbox flex items-center gap-2 cursor-pointer">
        <input 
          type="checkbox" 
          bind:checked={processingOptions.extractMetadata}
          class="checkbox"
        />
        <span class="text-sm text-dark-text-secondary">Extract Metadata</span>
      </label>
      
      <label class="option-checkbox flex items-center gap-2 cursor-pointer">
        <input 
          type="checkbox" 
          bind:checked={processingOptions.enhanceQuality}
          class="checkbox"
        />
        <span class="text-sm text-dark-text-secondary">Enhance Quality</span>
      </label>
      
      <label class="option-checkbox flex items-center gap-2 cursor-pointer">
        <input 
          type="checkbox" 
          bind:checked={processingOptions.detectLanguage}
          class="checkbox"
        />
        <span class="text-sm text-dark-text-secondary">Language Detection</span>
      </label>
    </div>
  </div>

  <!-- Action Buttons -->
  <div class="actions mt-6 flex items-center justify-between">
    <div class="text-sm text-dark-text-tertiary">
      {$files.length} files selected • {$selectedEngines.length} engines
    </div>
    
    <div class="flex gap-3">
      <Button 
        variant="outline" 
        on:click={() => files.set([])}
        disabled={$files.length === 0}
      >
        Clear All
      </Button>
      
      <Button 
        variant="primary" 
        on:click={processFiles}
        disabled={$files.length === 0 || $selectedEngines.length === 0}
      >
        <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
        Process Documents ({$files.length})
      </Button>
    </div>
  </div>
</div>

<style>
  .upload-zone {
    @apply border-2 border-dashed border-dark-border rounded-lg p-8 text-center transition-all cursor-pointer;
    background: var(--dark-bg-tertiary, #0f0f0f);
  }

  .upload-zone:hover,
  .upload-zone.drag-active {
    @apply border-green-500;
    background: rgba(0, 255, 65, 0.05);
  }

  .upload-content {
    @apply flex flex-col items-center gap-4;
  }

  .upload-icon {
    @apply text-dark-text-tertiary;
  }

  .upload-zone.drag-active .upload-icon {
    @apply text-green-400;
  }

  .files-list {
    @apply border border-dark-border rounded-lg p-4;
    background: var(--dark-bg-secondary, #1a1a1a);
  }

  .file-item {
    @apply transition-colors hover:bg-dark-bg-primary;
  }

  .engine-selection {
    @apply border border-dark-border rounded-lg p-4;
    background: var(--dark-bg-secondary, #1a1a1a);
  }

  .engine-option {
    @apply transition-all duration-200;
  }

  .processing-options {
    @apply border border-dark-border rounded-lg p-4;
    background: var(--dark-bg-secondary, #1a1a1a);
  }

  .checkbox {
    @apply w-4 h-4 rounded border-2 border-dark-border transition-colors;
    background: var(--dark-bg-tertiary, #0f0f0f);
    accent-color: #00ff41;
  }

  .checkbox:checked {
    @apply border-green-500 bg-green-500;
  }

  .actions {
    @apply border-t border-dark-border pt-4;
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar {
    width: 6px;
  }
  
  ::-webkit-scrollbar-track {
    background: var(--dark-bg-tertiary, #0f0f0f);
  }
  
  ::-webkit-scrollbar-thumb {
    background: var(--dark-border, #00ff4133);
    border-radius: 3px;
  }
</style>