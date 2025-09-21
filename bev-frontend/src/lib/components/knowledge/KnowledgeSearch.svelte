<!-- Advanced Vector Knowledge Search Interface -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  const dispatch = createEventDispatcher();
  
  export let isProcessing = false;
  export let lastQuery = '';
  export let selectedVectorDB = 'qdrant';
  
  let searchQuery = '';
  let searchOptions = {
    similarityThreshold: 0.7,
    maxResults: 50,
    searchType: 'semantic', // 'semantic', 'keyword', 'hybrid'
    includeMetadata: true,
    timeRange: 'all', // 'hour', 'day', 'week', 'month', 'all'
    documentTypes: ['all'], // ['pdf', 'text', 'markdown', 'all']
    entityTypes: ['all'] // ['person', 'organization', 'location', 'concept', 'all']
  };
  
  let advancedMode = false;
  let searchHistory = [];
  let savedQueries = [];

  // Load search history from localStorage
  function loadSearchHistory() {
    const saved = localStorage.getItem('bev-knowledge-history');
    if (saved) {
      try {
        searchHistory = JSON.parse(saved).slice(0, 20);
      } catch (e) {
        searchHistory = [];
      }
    }
  }

  function saveSearchHistory() {
    const historyItem = {
      query: searchQuery,
      timestamp: new Date().toISOString(),
      options: { ...searchOptions }
    };
    
    searchHistory = [historyItem, ...searchHistory.filter(h => h.query !== searchQuery)].slice(0, 20);
    localStorage.setItem('bev-knowledge-history', JSON.stringify(searchHistory));
  }

  function performSearch() {
    if (!searchQuery.trim() || isProcessing) return;
    
    saveSearchHistory();
    
    dispatch('search', {
      query: searchQuery,
      options: searchOptions
    });
  }

  function loadSavedQuery(query) {
    searchQuery = query.query;
    searchOptions = { ...query.options };
    performSearch();
  }

  function saveCurrentQuery() {
    const queryName = prompt('Save this query as:') || `Query ${Date.now()}`;
    const savedQuery = {
      name: queryName,
      query: searchQuery,
      options: { ...searchOptions },
      timestamp: new Date().toISOString()
    };
    
    savedQueries = [savedQuery, ...savedQueries].slice(0, 10);
    localStorage.setItem('bev-knowledge-saved', JSON.stringify(savedQueries));
  }

  function loadSavedQueries() {
    const saved = localStorage.getItem('bev-knowledge-saved');
    if (saved) {
      try {
        savedQueries = JSON.parse(saved);
      } catch (e) {
        savedQueries = [];
      }
    }
  }

  function generateExampleQueries() {
    return [
      'cryptocurrency mixing techniques and privacy coins',
      'darknet market vendor reputation systems',
      'OSINT tools for social media investigation',
      'blockchain analysis and transaction clustering',
      'threat intelligence IOC correlation methods',
      'metadata extraction from digital forensics',
      'machine learning in cybersecurity applications',
      'Tor network traffic analysis patterns'
    ];
  }

  // Initialize component
  loadSearchHistory();
  loadSavedQueries();
</script>

<Card variant="bordered">
  <div class="p-6">
    <div class="flex items-center justify-between mb-4">
      <h2 class="text-lg font-semibold text-dark-text-primary">Knowledge Vector Search</h2>
      <div class="flex items-center gap-2">
        <Badge variant="info" size="sm">{selectedVectorDB.toUpperCase()}</Badge>
        <button 
          class="text-xs text-dark-text-tertiary hover:text-dark-text-primary transition-colors"
          on:click={() => advancedMode = !advancedMode}
        >
          {advancedMode ? 'Simple' : 'Advanced'} Mode
        </button>
      </div>
    </div>

    <!-- Main Search Interface -->
    <form on:submit|preventDefault={performSearch} class="mb-6">
      <div class="flex gap-3">
        <div class="flex-1 relative">
          <input
            type="text"
            bind:value={searchQuery}
            placeholder="Search knowledge base using semantic similarity..."
            class="w-full px-4 py-3 pr-12 bg-dark-bg-tertiary border border-dark-border rounded-lg text-dark-text-primary placeholder-dark-text-tertiary focus:border-green-500 focus:outline-none"
            disabled={isProcessing}
          />
          
          {#if isProcessing}
            <div class="absolute right-3 top-1/2 transform -translate-y-1/2">
              <div class="w-5 h-5 border-2 border-green-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
          {/if}
        </div>
        
        <Button 
          type="submit" 
          variant="primary"
          disabled={isProcessing || !searchQuery.trim()}
        >
          <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          Search
        </Button>
        
        {#if searchQuery.trim()}
          <Button variant="outline" on:click={saveCurrentQuery}>
            Save Query
          </Button>
        {/if}
      </div>
    </form>

    <!-- Advanced Options -->
    {#if advancedMode}
      <div class="advanced-options grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6 p-4 bg-dark-bg-tertiary rounded-lg border border-dark-border">
        <!-- Similarity Threshold -->
        <div>
          <label class="block text-xs text-dark-text-tertiary mb-2">
            Similarity Threshold: {searchOptions.similarityThreshold.toFixed(2)}
          </label>
          <input 
            type="range" 
            min="0.1" 
            max="1.0" 
            step="0.05"
            bind:value={searchOptions.similarityThreshold}
            class="w-full h-2 bg-dark-bg-primary rounded-lg appearance-none cursor-pointer slider"
          />
        </div>

        <!-- Max Results -->
        <div>
          <label class="block text-xs text-dark-text-tertiary mb-2">Max Results</label>
          <select 
            bind:value={searchOptions.maxResults}
            class="w-full px-2 py-1 bg-dark-bg-primary border border-dark-border rounded text-dark-text-primary text-sm focus:border-green-500 focus:outline-none"
          >
            <option value={10}>10 results</option>
            <option value={25}>25 results</option>
            <option value={50}>50 results</option>
            <option value={100}>100 results</option>
            <option value={250}>250 results</option>
          </select>
        </div>

        <!-- Search Type -->
        <div>
          <label class="block text-xs text-dark-text-tertiary mb-2">Search Type</label>
          <select 
            bind:value={searchOptions.searchType}
            class="w-full px-2 py-1 bg-dark-bg-primary border border-dark-border rounded text-dark-text-primary text-sm focus:border-green-500 focus:outline-none"
          >
            <option value="semantic">Semantic (Vector)</option>
            <option value="keyword">Keyword (BM25)</option>
            <option value="hybrid">Hybrid (Both)</option>
          </select>
        </div>

        <!-- Time Range -->
        <div>
          <label class="block text-xs text-dark-text-tertiary mb-2">Time Range</label>
          <select 
            bind:value={searchOptions.timeRange}
            class="w-full px-2 py-1 bg-dark-bg-primary border border-dark-border rounded text-dark-text-primary text-sm focus:border-green-500 focus:outline-none"
          >
            <option value="hour">Last Hour</option>
            <option value="day">Last 24 Hours</option>
            <option value="week">Last Week</option>
            <option value="month">Last Month</option>
            <option value="all">All Time</option>
          </select>
        </div>
      </div>
    {/if}

    <!-- Search Suggestions -->
    {#if !lastQuery && searchHistory.length === 0}
      <div class="search-suggestions mb-6">
        <h4 class="text-sm font-medium text-dark-text-primary mb-3">Example Queries</h4>
        <div class="flex flex-wrap gap-2">
          {#each generateExampleQueries().slice(0, 6) as example}
            <button 
              class="px-3 py-1 text-xs bg-dark-bg-tertiary border border-dark-border rounded hover:border-green-500 text-dark-text-secondary hover:text-dark-text-primary transition-colors"
              on:click={() => {
                searchQuery = example;
                performSearch();
              }}
            >
              {example}
            </button>
          {/each}
        </div>
      </div>
    {/if}

    <!-- Search History -->
    {#if searchHistory.length > 0}
      <div class="search-history mb-6">
        <div class="flex items-center justify-between mb-3">
          <h4 class="text-sm font-medium text-dark-text-primary">Recent Searches</h4>
          <button 
            class="text-xs text-dark-text-tertiary hover:text-dark-text-primary"
            on:click={() => {
              searchHistory = [];
              localStorage.removeItem('bev-knowledge-history');
            }}
          >
            Clear History
          </button>
        </div>
        <div class="space-y-1 max-h-32 overflow-y-auto">
          {#each searchHistory.slice(0, 8) as query}
            <button 
              class="w-full text-left p-2 text-xs bg-dark-bg-tertiary rounded border border-dark-border hover:border-green-500 text-dark-text-secondary hover:text-dark-text-primary transition-colors"
              on:click={() => loadSavedQuery(query)}
            >
              <div class="truncate">{query.query}</div>
              <div class="text-dark-text-tertiary">{new Date(query.timestamp).toLocaleString()}</div>
            </button>
          {/each}
        </div>
      </div>
    {/if}

    <!-- Saved Queries -->
    {#if savedQueries.length > 0}
      <div class="saved-queries">
        <h4 class="text-sm font-medium text-dark-text-primary mb-3">Saved Queries</h4>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
          {#each savedQueries as query}
            <div class="saved-query p-2 bg-dark-bg-tertiary rounded border border-dark-border">
              <div class="flex items-center justify-between">
                <button 
                  class="flex-1 text-left text-xs text-dark-text-primary hover:text-green-400 transition-colors"
                  on:click={() => loadSavedQuery(query)}
                >
                  {query.name}
                </button>
                <button 
                  class="text-xs text-red-400 hover:text-red-300 ml-2"
                  on:click={() => {
                    savedQueries = savedQueries.filter(q => q.name !== query.name);
                    localStorage.setItem('bev-knowledge-saved', JSON.stringify(savedQueries));
                  }}
                >
                  ×
                </button>
              </div>
              <div class="text-xs text-dark-text-tertiary truncate mt-1">
                {query.query}
              </div>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
</Card>

<style>
  .slider {
    background: var(--dark-bg-primary);
  }

  .slider::-webkit-slider-thumb {
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #00ff41;
    cursor: pointer;
    border: 2px solid #0a0a0a;
  }

  .slider::-webkit-slider-track {
    width: 100%;
    height: 4px;
    cursor: pointer;
    background: var(--dark-bg-primary);
    border-radius: 2px;
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