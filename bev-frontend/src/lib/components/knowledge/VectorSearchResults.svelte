<!-- Vector Search Results Display with Actions -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { marked } from 'marked';
  import DOMPurify from 'isomorphic-dompurify';
  
  const dispatch = createEventDispatcher();
  
  export let results = [];
  export let selectedResults = [];
  
  let sortBy = 'similarity'; // 'similarity', 'timestamp', 'source', 'type'
  let sortOrder = 'desc'; // 'asc', 'desc'
  let groupBy = 'none'; // 'none', 'source', 'type', 'similarity'
  let viewMode = 'list'; // 'list', 'grid', 'compact'
  let expandedResults = new Set();

  const resultTypeIcons = {
    document: '📄',
    fragment: '📝',
    entity: '🏷️',
    concept: '💡',
    default: '🔍'
  };

  const similarityColors = {
    high: '#00ff41',    // > 0.8
    medium: '#ffff00',  // 0.6 - 0.8  
    low: '#ff9500'      // < 0.6
  };

  function getSimilarityColor(similarity) {
    if (similarity > 0.8) return similarityColors.high;
    if (similarity > 0.6) return similarityColors.medium;
    return similarityColors.low;
  }

  function getSimilarityLabel(similarity) {
    if (similarity > 0.9) return 'Excellent';
    if (similarity > 0.8) return 'Very Good';
    if (similarity > 0.7) return 'Good';
    if (similarity > 0.6) return 'Fair';
    return 'Poor';
  }

  function isResultSelected(result) {
    return selectedResults.some(r => r.id === result.id);
  }

  function toggleResult(result) {
    const selected = isResultSelected(result);
    dispatch('resultToggled', { result, selected: !selected });
  }

  function selectAllVisible() {
    sortedResults.forEach(result => {
      if (!isResultSelected(result)) {
        dispatch('resultToggled', { result, selected: true });
      }
    });
  }

  function clearAllSelected() {
    selectedResults.forEach(result => {
      dispatch('resultToggled', { result, selected: false });
    });
  }

  function expandResult(resultId) {
    if (expandedResults.has(resultId)) {
      expandedResults.delete(resultId);
    } else {
      expandedResults.add(resultId);
    }
    expandedResults = expandedResults;
  }

  function copyResultText(result) {
    navigator.clipboard.writeText(result.content);
  }

  function formatContent(content, maxLength = 200) {
    if (content.length <= maxLength) return content;
    return content.slice(0, maxLength) + '...';
  }

  function formatTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString();
  }

  function downloadResult(result) {
    const exportData = {
      id: result.id,
      content: result.content,
      metadata: result.metadata,
      similarity: result.similarity,
      source: result.source,
      timestamp: result.timestamp,
      type: result.type,
      exportTimestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `search-result-${result.id}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  $: sortedResults = results.slice().sort((a, b) => {
    let aVal, bVal;
    
    switch (sortBy) {
      case 'similarity':
        aVal = a.similarity;
        bVal = b.similarity;
        break;
      case 'timestamp':
        aVal = new Date(a.timestamp).getTime();
        bVal = new Date(b.timestamp).getTime();
        break;
      case 'source':
        aVal = a.source;
        bVal = b.source;
        break;
      case 'type':
        aVal = a.type;
        bVal = b.type;
        break;
      default:
        return 0;
    }
    
    const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
    return sortOrder === 'asc' ? comparison : -comparison;
  });

  $: groupedResults = groupBy === 'none' ? { 'All Results': sortedResults } : 
    sortedResults.reduce((groups, result) => {
      const key = groupBy === 'source' ? result.source :
                  groupBy === 'type' ? result.type :
                  groupBy === 'similarity' ? getSimilarityLabel(result.similarity) :
                  'All Results';
      
      if (!groups[key]) groups[key] = [];
      groups[key].push(result);
      return groups;
    }, {});
</script>

<Card variant="bordered">
  <div class="p-4">
    <!-- Results Header -->
    <div class="flex items-center justify-between mb-4">
      <div class="flex items-center gap-3">
        <h3 class="text-md font-medium text-dark-text-primary">
          Search Results ({results.length})
        </h3>
        {#if selectedResults.length > 0}
          <Badge variant="success" size="sm">
            {selectedResults.length} selected
          </Badge>
        {/if}
      </div>
      
      <div class="flex items-center gap-2">
        <!-- View Mode Toggle -->
        <div class="flex bg-dark-bg-tertiary rounded p-1">
          {#each ['list', 'grid', 'compact'] as mode}
            <button
              class="px-2 py-1 text-xs font-medium rounded transition-colors {
                viewMode === mode 
                  ? 'bg-green-600 text-black' 
                  : 'text-dark-text-secondary hover:text-dark-text-primary'
              }"
              on:click={() => viewMode = mode}
            >
              {mode.toUpperCase()}
            </button>
          {/each}
        </div>

        <!-- Bulk Actions -->
        {#if selectedResults.length > 0}
          <Button variant="outline" size="sm" on:click={() => {
            dispatch('createChat', { 
              documentId: selectedResults[0].id, 
              documentName: `${selectedResults.length} selected results` 
            });
          }}>
            Chat with Selected
          </Button>
          <Button variant="outline" size="sm" on:click={clearAllSelected}>
            Clear Selected
          </Button>
        {:else}
          <Button variant="outline" size="sm" on:click={selectAllVisible}>
            Select All
          </Button>
        {/if}
      </div>
    </div>

    <!-- Sort and Filter Controls -->
    <div class="flex items-center gap-3 mb-4 p-3 bg-dark-bg-tertiary rounded border border-dark-border">
      <div class="flex items-center gap-2">
        <span class="text-xs text-dark-text-tertiary">Sort by:</span>
        <select 
          bind:value={sortBy}
          class="px-2 py-1 bg-dark-bg-primary border border-dark-border rounded text-dark-text-primary text-xs focus:border-green-500 focus:outline-none"
        >
          <option value="similarity">Similarity</option>
          <option value="timestamp">Time</option>
          <option value="source">Source</option>
          <option value="type">Type</option>
        </select>
        
        <button 
          class="text-xs text-dark-text-tertiary hover:text-dark-text-primary"
          on:click={() => sortOrder = sortOrder === 'asc' ? 'desc' : 'asc'}
        >
          {sortOrder === 'asc' ? '↑' : '↓'}
        </button>
      </div>
      
      <div class="flex items-center gap-2">
        <span class="text-xs text-dark-text-tertiary">Group by:</span>
        <select 
          bind:value={groupBy}
          class="px-2 py-1 bg-dark-bg-primary border border-dark-border rounded text-dark-text-primary text-xs focus:border-green-500 focus:outline-none"
        >
          <option value="none">None</option>
          <option value="source">Source</option>
          <option value="type">Type</option>
          <option value="similarity">Similarity</option>
        </select>
      </div>
    </div>

    <!-- Results Display -->
    <div class="results-container max-h-96 overflow-y-auto">
      {#each Object.entries(groupedResults) as [groupName, groupResults]}
        {#if groupBy !== 'none'}
          <div class="group-header sticky top-0 bg-dark-bg-secondary border border-dark-border rounded p-2 mb-3 z-10">
            <h4 class="text-sm font-medium text-dark-text-primary">
              {groupName} ({groupResults.length})
            </h4>
          </div>
        {/if}

        <div class="results-group space-y-3 mb-6">
          {#each groupResults as result}
            <div class="result-item p-4 bg-dark-bg-tertiary rounded border transition-all {
              isResultSelected(result) ? 'border-green-500 bg-green-500/10' : 'border-dark-border hover:border-dark-text-tertiary'
            }">
              <!-- Result Header -->
              <div class="flex items-start justify-between mb-3">
                <div class="flex items-start gap-3 flex-1 min-w-0">
                  <!-- Selection Checkbox -->
                  <button 
                    class="mt-1 w-4 h-4 rounded border-2 flex items-center justify-center {
                      isResultSelected(result) ? 'border-green-500 bg-green-500' : 'border-dark-border'
                    }"
                    on:click={() => toggleResult(result)}
                  >
                    {#if isResultSelected(result)}
                      <svg class="w-3 h-3 text-black" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                      </svg>
                    {/if}
                  </button>

                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 mb-1">
                      <span class="text-lg">{resultTypeIcons[result.type] || resultTypeIcons.default}</span>
                      <span class="text-sm font-medium text-dark-text-primary truncate">
                        {result.source}
                      </span>
                      <Badge variant="info" size="xs">{result.type}</Badge>
                    </div>
                    
                    <div class="text-xs text-dark-text-tertiary mb-2">
                      {formatTimestamp(result.timestamp)} • ID: {result.id}
                    </div>
                  </div>
                </div>

                <!-- Similarity Score -->
                <div class="flex items-center gap-2 flex-shrink-0">
                  <div class="text-right">
                    <div 
                      class="text-sm font-bold"
                      style="color: {getSimilarityColor(result.similarity)}"
                    >
                      {(result.similarity * 100).toFixed(1)}%
                    </div>
                    <div class="text-xs text-dark-text-tertiary">
                      {getSimilarityLabel(result.similarity)}
                    </div>
                  </div>
                  
                  <!-- Actions Dropdown -->
                  <div class="relative">
                    <button class="text-dark-text-tertiary hover:text-dark-text-primary">
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M12 5v.01M12 12v.01M12 19v.01" />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>

              <!-- Result Content -->
              <div class="result-content">
                <div class="text-sm text-dark-text-primary leading-relaxed {
                  expandedResults.has(result.id) ? '' : 'line-clamp-3'
                }">
                  {expandedResults.has(result.id) ? result.content : formatContent(result.content)}
                </div>
                
                {#if result.content.length > 200}
                  <button 
                    class="text-xs text-cyan-400 hover:text-cyan-300 mt-2"
                    on:click={() => expandResult(result.id)}
                  >
                    {expandedResults.has(result.id) ? 'Show Less' : 'Show More'}
                  </button>
                {/if}
              </div>

              <!-- Metadata -->
              {#if result.metadata && Object.keys(result.metadata).length > 0}
                <div class="metadata mt-3 pt-3 border-t border-dark-border">
                  <div class="text-xs text-dark-text-tertiary mb-1">Metadata:</div>
                  <div class="grid grid-cols-2 gap-2 text-xs">
                    {#each Object.entries(result.metadata).slice(0, 4) as [key, value]}
                      <div class="flex justify-between">
                        <span class="text-dark-text-tertiary">{key}:</span>
                        <span class="text-dark-text-secondary truncate ml-2">{value}</span>
                      </div>
                    {/each}
                  </div>
                </div>
              {/if}

              <!-- Quick Actions -->
              <div class="actions mt-3 pt-3 border-t border-dark-border flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <Button variant="outline" size="xs" on:click={() => copyResultText(result)}>
                    📋 Copy
                  </Button>
                  <Button variant="outline" size="xs" on:click={() => downloadResult(result)}>
                    💾 Download
                  </Button>
                  <Button variant="outline" size="xs" on:click={() => {
                    dispatch('createChat', { 
                      documentId: result.id, 
                      documentName: result.source 
                    });
                  }}>
                    💬 Chat
                  </Button>
                </div>
                
                <div class="flex items-center gap-1">
                  <button 
                    class="text-xs text-dark-text-tertiary hover:text-green-400"
                    on:click={() => dispatch('addToGraph', result)}
                    title="Add to knowledge graph"
                  >
                    🕸️ Graph
                  </button>
                  <button 
                    class="text-xs text-dark-text-tertiary hover:text-blue-400"
                    on:click={() => {
                      // Find similar results
                      dispatch('search', { 
                        query: result.content.slice(0, 100), 
                        options: { similarityThreshold: 0.8 } 
                      });
                    }}
                    title="Find similar content"
                  >
                    🔍 Similar
                  </button>
                </div>
              </div>
            </div>
          {/each}
        </div>
      {/each}
    </div>

    <!-- Results Footer -->
    {#if results.length > 0}
      <div class="results-footer mt-4 pt-4 border-t border-dark-border">
        <div class="flex items-center justify-between text-sm">
          <div class="flex items-center gap-4">
            <span class="text-dark-text-tertiary">
              Showing {results.length} results
            </span>
            {#if selectedResults.length > 0}
              <span class="text-green-400">
                {selectedResults.length} selected
              </span>
            {/if}
          </div>
          
          <div class="flex items-center gap-2">
            <Button variant="outline" size="sm" on:click={() => {
              const allText = results.map(r => `// ${r.source}\n${r.content}`).join('\n\n');
              navigator.clipboard.writeText(allText);
            }}>
              Copy All
            </Button>
            
            <Button variant="outline" size="sm" on:click={() => {
              const exportData = {
                query: 'Search results export',
                totalResults: results.length,
                selectedCount: selectedResults.length,
                exportTimestamp: new Date().toISOString(),
                results: results.map(r => ({
                  id: r.id,
                  content: r.content,
                  similarity: r.similarity,
                  source: r.source,
                  type: r.type,
                  timestamp: r.timestamp
                }))
              };
              
              const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
                type: 'application/json' 
              });
              const url = URL.createObjectURL(blob);
              const a = document.createElement('a');
              a.href = url;
              a.download = `search-results-export-${Date.now()}.json`;
              a.click();
              URL.revokeObjectURL(url);
            }}>
              Export All
            </Button>
          </div>
        </div>
      </div>
    {/if}
  </div>
</Card>

<style>
  .line-clamp-3 {
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .result-item:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 255, 65, 0.1);
  }

  .group-header {
    backdrop-filter: blur(4px);
  }

  /* Scrollbar styling */
  .results-container::-webkit-scrollbar {
    width: 8px;
  }
  
  .results-container::-webkit-scrollbar-track {
    background: var(--dark-bg-primary, #0a0a0a);
    border-radius: 4px;
  }
  
  .results-container::-webkit-scrollbar-thumb {
    background: var(--dark-border, #00ff4133);
    border-radius: 4px;
  }

  .results-container::-webkit-scrollbar-thumb:hover {
    background: var(--dark-text-tertiary, #00ff4166);
  }
</style>