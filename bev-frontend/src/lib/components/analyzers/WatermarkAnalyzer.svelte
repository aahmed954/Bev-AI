<!-- Digital Watermark Detection System -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  const dispatch = createEventDispatcher();
  
  export let analyzer;
  export let activeJobs = [];
  export let completedJobs = [];
  
  let selectedFiles = [];
  let analysisMode = 'comprehensive'; // 'quick', 'comprehensive', 'deep'
  let watermarkTypes = {
    visible: true,
    invisible: true,
    digital: true,
    steganographic: false
  };

  function handleFileUpload(e) {
    const files = Array.from(e.target.files || []);
    selectedFiles = [...selectedFiles, ...files].slice(0, 20);
  }

  function startWatermarkAnalysis() {
    if (selectedFiles.length === 0) return;
    
    selectedFiles.forEach(file => {
      const reader = new FileReader();
      reader.onload = () => {
        dispatch('startAnalysis', {
          target: file.name,
          options: {
            type: 'watermark',
            fileData: reader.result,
            mode: analysisMode,
            detectTypes: watermarkTypes
          }
        });
      };
      reader.readAsDataURL(file);
    });
    
    selectedFiles = [];
  }
</script>

<div class="watermark-analyzer space-y-6">
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center gap-3">
        <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>← Back</Button>
        <span class="text-2xl">🔍</span>
        <h2 class="text-lg font-semibold text-dark-text-primary">Watermark Detection</h2>
      </div>
    </div>
  </Card>

  <Card variant="bordered">
    <div class="p-6">
      <input type="file" multiple accept="image/*,video/*,audio/*" on:change={handleFileUpload} />
      <Button variant="primary" on:click={startWatermarkAnalysis} disabled={selectedFiles.length === 0}>
        Analyze Watermarks ({selectedFiles.length})
      </Button>
    </div>
  </Card>
</div>