<!-- OCR Multi-Engine Result Comparison Interface -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { marked } from 'marked';
  import DOMPurify from 'isomorphic-dompurify';
  
  const dispatch = createEventDispatcher();
  
  export let job;
  
  let selectedEngine = job.results?.[0]?.engine || 'tesseract';
  let showDiff = false;
  let showMetadata = false;
  let comparisonMode = 'side-by-side'; // 'side-by-side', 'overlay', 'diff'

  const engineInfo = {
    tesseract: { name: 'Tesseract OCR', color: '#00ff41', icon: '📝' },
    easyocr: { name: 'EasyOCR', color: '#00ccff', icon: '🧠' },
    trocr: { name: 'TrOCR', color: '#ff9500', icon: '🤖' },
    hybrid: { name: 'Hybrid Mode', color: '#ff00ff', icon: '⚡' }
  };

  function getResultByEngine(engine) {
    return job.results?.find(r => r.engine === engine);
  }

  function getBestResult() {
    if (!job.results) return null;
    return job.results.reduce((best, current) => 
      current.confidence > (best?.confidence || 0) ? current : best
    );
  }

  function calculateDifferences() {
    if (!job.results || job.results.length < 2) return [];
    
    const results = job.results;
    const diffs = [];
    
    for (let i = 0; i < results.length; i++) {
      for (let j = i + 1; j < results.length; j++) {
        const similarity = calculateSimilarity(results[i].text, results[j].text);
        diffs.push({
          engine1: results[i].engine,
          engine2: results[j].engine,
          similarity: similarity,
          lengthDiff: Math.abs(results[i].text.length - results[j].text.length),
          confidenceDiff: Math.abs(results[i].confidence - results[j].confidence)
        });
      }
    }
    
    return diffs.sort((a, b) => b.similarity - a.similarity);
  }

  function calculateSimilarity(text1, text2) {
    // Simple similarity calculation using Levenshtein distance
    const longer = text1.length > text2.length ? text1 : text2;
    const shorter = text1.length > text2.length ? text2 : text1;
    
    if (longer.length === 0) return 1.0;
    
    const distance = levenshteinDistance(longer, shorter);
    return (longer.length - distance) / longer.length;
  }

  function levenshteinDistance(str1, str2) {
    const matrix = [];
    
    for (let i = 0; i <= str2.length; i++) {
      matrix[i] = [i];
    }
    
    for (let j = 0; j <= str1.length; j++) {
      matrix[0][j] = j;
    }
    
    for (let i = 1; i <= str2.length; i++) {
      for (let j = 1; j <= str1.length; j++) {
        if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
          matrix[i][j] = matrix[i - 1][j - 1];
        } else {
          matrix[i][j] = Math.min(
            matrix[i - 1][j - 1] + 1,
            matrix[i][j - 1] + 1,
            matrix[i - 1][j] + 1
          );
        }
      }
    }
    
    return matrix[str2.length][str1.length];
  }

  function highlightDifferences(text1, text2) {
    // Basic diff highlighting - could be enhanced with a proper diff library
    const words1 = text1.split(/\s+/);
    const words2 = text2.split(/\s+/);
    
    // Simple word-level comparison
    return {
      text1: words1.map(word => {
        const inText2 = words2.includes(word);
        return inText2 ? word : `<mark class="bg-red-500/30">${word}</mark>`;
      }).join(' '),
      text2: words2.map(word => {
        const inText1 = words1.includes(word);
        return inText1 ? word : `<mark class="bg-green-500/30">${word}</mark>`;
      }).join(' ')
    };
  }

  function copyToClipboard(text) {
    navigator.clipboard.writeText(text);
  }

  function exportComparison() {
    const comparisonData = {
      job: {
        id: job.id,
        filename: job.filename,
        timestamp: job.timestamp
      },
      results: job.results,
      analysis: {
        bestResult: getBestResult(),
        differences: calculateDifferences(),
        recommendations: generateRecommendations()
      },
      exportTimestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(comparisonData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ocr-comparison-${job.filename}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function generateRecommendations() {
    if (!job.results) return [];
    
    const recommendations = [];
    const bestResult = getBestResult();
    
    if (bestResult) {
      recommendations.push(`Best engine: ${bestResult.engine} (${(bestResult.confidence * 100).toFixed(1)}% confidence)`);
    }
    
    const fastestResult = job.results.reduce((fastest, current) => 
      current.processingTime < (fastest?.processingTime || Infinity) ? current : fastest
    );
    
    if (fastestResult) {
      recommendations.push(`Fastest engine: ${fastestResult.engine} (${fastestResult.processingTime}ms)`);
    }
    
    return recommendations;
  }

  $: differences = calculateDifferences();
  $: bestResult = getBestResult();
</script>

<div class="ocr-comparison space-y-6">
  <!-- Header -->
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-lg font-semibold text-dark-text-primary mb-1">OCR Results Comparison</h2>
          <div class="text-sm text-dark-text-secondary">
            {job.filename} • {job.engines.length} engines • {formatTimestamp(job.timestamp)}
          </div>
        </div>
        
        <div class="flex items-center gap-3">
          <div class="flex bg-dark-bg-tertiary rounded-lg p-1">
            {#each ['side-by-side', 'overlay', 'diff'] as mode}
              <button
                class="px-3 py-1 text-xs font-medium rounded transition-colors {
                  comparisonMode === mode 
                    ? 'bg-green-600 text-black' 
                    : 'text-dark-text-secondary hover:text-dark-text-primary'
                }"
                on:click={() => comparisonMode = mode}
              >
                {mode.replace('-', ' ').toUpperCase()}
              </button>
            {/each}
          </div>
          
          <Button variant="outline" size="sm" on:click={exportComparison}>
            Export Analysis
          </Button>
          
          <Button variant="outline" size="sm" on:click={() => dispatch('backToDashboard')}>
            ← Back
          </Button>
        </div>
      </div>
    </div>
  </Card>

  <!-- Engine Results Overview -->
  <Card variant="bordered">
    <div class="p-4">
      <h3 class="text-md font-medium mb-4 text-dark-text-primary">Engine Performance Summary</h3>
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {#each job.results || [] as result}
          <div 
            class="engine-summary p-3 rounded border cursor-pointer transition-all {
              selectedEngine === result.engine 
                ? 'border-green-500 bg-green-500/10' 
                : 'border-dark-border bg-dark-bg-tertiary hover:border-dark-text-tertiary'
            }"
            on:click={() => selectedEngine = result.engine}
          >
            <div class="flex items-center gap-2 mb-2">
              <span class="text-lg">{engineInfo[result.engine]?.icon || '🔍'}</span>
              <span class="text-sm font-medium text-dark-text-primary">
                {engineInfo[result.engine]?.name || result.engine}
              </span>
              {#if bestResult?.engine === result.engine}
                <Badge variant="success" size="xs">BEST</Badge>
              {/if}
            </div>
            
            <div class="space-y-1 text-xs">
              <div class="flex justify-between">
                <span class="text-dark-text-tertiary">Confidence:</span>
                <span 
                  class="font-medium"
                  style="color: {result.confidence > 0.8 ? '#00ff41' : result.confidence > 0.6 ? '#ffff00' : '#ff9500'}"
                >
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div class="flex justify-between">
                <span class="text-dark-text-tertiary">Time:</span>
                <span class="text-dark-text-secondary">{formatDuration(result.processingTime)}</span>
              </div>
              <div class="flex justify-between">
                <span class="text-dark-text-tertiary">Characters:</span>
                <span class="text-dark-text-secondary">{result.text.length}</span>
              </div>
            </div>
          </div>
        {/each}
      </div>
    </div>
  </Card>

  <!-- Result Comparison -->
  {#if job.results && job.results.length > 0}
    <Card variant="bordered">
      <div class="p-4">
        <div class="flex items-center justify-between mb-4">
          <h3 class="text-md font-medium text-dark-text-primary">Extracted Text Comparison</h3>
          <div class="flex items-center gap-2">
            <label class="flex items-center gap-2 text-sm">
              <input type="checkbox" bind:checked={showDiff} class="checkbox" />
              <span class="text-dark-text-secondary">Highlight Differences</span>
            </label>
            <label class="flex items-center gap-2 text-sm">
              <input type="checkbox" bind:checked={showMetadata} class="checkbox" />
              <span class="text-dark-text-secondary">Show Metadata</span>
            </label>
          </div>
        </div>

        {#if comparisonMode === 'side-by-side'}
          <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {#each job.results as result}
              <div class="result-panel">
                <div class="result-header flex items-center justify-between mb-3 p-3 bg-dark-bg-tertiary rounded-t border-b border-dark-border">
                  <div class="flex items-center gap-2">
                    <span class="text-lg">{engineInfo[result.engine]?.icon || '🔍'}</span>
                    <span class="font-medium text-dark-text-primary">{engineInfo[result.engine]?.name}</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="text-xs text-dark-text-tertiary">
                      {(result.confidence * 100).toFixed(1)}% • {formatDuration(result.processingTime)}
                    </span>
                    <button 
                      class="text-dark-text-tertiary hover:text-dark-text-primary"
                      on:click={() => copyToClipboard(result.text)}
                      title="Copy text"
                    >
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                    </button>
                  </div>
                </div>
                
                <div class="result-content p-3 bg-dark-bg-primary border border-dark-border rounded-b max-h-96 overflow-y-auto">
                  <pre class="text-sm text-dark-text-primary whitespace-pre-wrap font-mono leading-relaxed">
{result.text || 'No text extracted'}
                  </pre>
                </div>

                {#if showMetadata && result.metadata}
                  <div class="metadata mt-3 p-3 bg-dark-bg-tertiary rounded border border-dark-border">
                    <h5 class="text-xs font-medium text-dark-text-primary mb-2">Metadata</h5>
                    <pre class="text-xs text-dark-text-secondary overflow-x-auto">
{JSON.stringify(result.metadata, null, 2)}
                    </pre>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}

        {#if comparisonMode === 'diff' && job.results.length >= 2}
          <div class="diff-view">
            {#each differences as diff}
              <div class="diff-comparison mb-4 p-4 bg-dark-bg-tertiary rounded border border-dark-border">
                <div class="diff-header flex items-center justify-between mb-3">
                  <h5 class="text-sm font-medium text-dark-text-primary">
                    {engineInfo[diff.engine1]?.name} vs {engineInfo[diff.engine2]?.name}
                  </h5>
                  <div class="flex items-center gap-3 text-xs text-dark-text-secondary">
                    <span>Similarity: {(diff.similarity * 100).toFixed(1)}%</span>
                    <span>Length Diff: {diff.lengthDiff} chars</span>
                    <span>Confidence Diff: {(diff.confidenceDiff * 100).toFixed(1)}%</span>
                  </div>
                </div>
                
                {#if showDiff}
                  {@const result1 = getResultByEngine(diff.engine1)}
                  {@const result2 = getResultByEngine(diff.engine2)}
                  {@const highlighted = highlightDifferences(result1?.text || '', result2?.text || '')}
                  
                  <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div>
                      <h6 class="text-xs text-dark-text-tertiary mb-2">{engineInfo[diff.engine1]?.name}</h6>
                      <div class="diff-text p-3 bg-dark-bg-primary rounded text-sm">
                        {@html DOMPurify.sanitize(highlighted.text1)}
                      </div>
                    </div>
                    <div>
                      <h6 class="text-xs text-dark-text-tertiary mb-2">{engineInfo[diff.engine2]?.name}</h6>
                      <div class="diff-text p-3 bg-dark-bg-primary rounded text-sm">
                        {@html DOMPurify.sanitize(highlighted.text2)}
                      </div>
                    </div>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        {/if}
      </div>
    </Card>
  {/if}

  <!-- Analysis Summary -->
  <Card variant="bordered">
    <div class="p-4">
      <h3 class="text-md font-medium mb-4 text-dark-text-primary">Analysis Summary</h3>
      
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <!-- Best Result -->
        <div class="summary-stat">
          <div class="text-xs text-dark-text-tertiary mb-1">Recommended Engine</div>
          <div class="flex items-center gap-2">
            <span class="text-lg">{engineInfo[bestResult?.engine]?.icon || '🔍'}</span>
            <span class="text-sm font-medium text-green-400">
              {engineInfo[bestResult?.engine]?.name || 'Unknown'}
            </span>
          </div>
          <div class="text-xs text-dark-text-secondary mt-1">
            {(bestResult?.confidence * 100).toFixed(1)}% confidence
          </div>
        </div>

        <!-- Processing Stats -->
        <div class="summary-stat">
          <div class="text-xs text-dark-text-tertiary mb-1">Processing Time</div>
          <div class="text-lg font-medium text-cyan-400">
            {job.results?.reduce((sum, r) => sum + r.processingTime, 0) || 0}ms
          </div>
          <div class="text-xs text-dark-text-secondary mt-1">
            Total across all engines
          </div>
        </div>

        <!-- Text Quality -->
        <div class="summary-stat">
          <div class="text-xs text-dark-text-tertiary mb-1">Text Quality</div>
          <div class="text-lg font-medium text-purple-400">
            {differences.length > 0 ? (differences[0].similarity * 100).toFixed(1) : 'N/A'}%
          </div>
          <div class="text-xs text-dark-text-secondary mt-1">
            Cross-engine similarity
          </div>
        </div>
      </div>

      <!-- Recommendations -->
      <div class="recommendations">
        <h5 class="text-sm font-medium text-dark-text-primary mb-2">Recommendations</h5>
        <ul class="space-y-1 text-xs text-dark-text-secondary">
          {#each generateRecommendations() as recommendation}
            <li class="flex items-start gap-2">
              <span class="text-green-400 mt-0.5">•</span>
              <span>{recommendation}</span>
            </li>
          {/each}
        </ul>
      </div>
    </div>
  </Card>
</div>

<style>
  .result-panel {
    min-height: 300px;
  }

  .result-content {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
  }

  .diff-text :global(mark) {
    padding: 1px 2px;
    border-radius: 2px;
  }

  .summary-stat {
    @apply p-3 bg-dark-bg-tertiary rounded border border-dark-border;
  }

  .checkbox {
    @apply w-4 h-4 rounded border-2 border-dark-border transition-colors;
    background: var(--dark-bg-tertiary, #0f0f0f);
    accent-color: #00ff41;
  }

  .recommendations ul li {
    @apply leading-relaxed;
  }

  /* Scrollbar styling */
  .result-content::-webkit-scrollbar {
    width: 6px;
  }
  
  .result-content::-webkit-scrollbar-track {
    background: var(--dark-bg-tertiary, #0f0f0f);
  }
  
  .result-content::-webkit-scrollbar-thumb {
    background: var(--dark-border, #00ff4133);
    border-radius: 3px;
  }
</style>