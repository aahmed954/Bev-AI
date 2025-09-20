<!-- Breach Database Lookup Interface -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  
  const dispatch = createEventDispatcher();
  
  export let analyzer;
  export let activeJobs = [];
  export let completedJobs = [];
  
  let searchTarget = '';
  let searchType = 'email'; // 'email', 'username', 'domain'
  let selectedSources = ['dehashed', 'snusbase', 'hibp'];
  let searchOptions = {
    includePasswords: false,
    includeDates: true,
    includeNames: true,
    verifyEmail: true
  };

  const breachSources = [
    { id: 'dehashed', name: 'DeHashed', description: 'Comprehensive breach database' },
    { id: 'snusbase', name: 'Snusbase', description: 'Real-time breach monitoring' },
    { id: 'weleakinfo', name: 'WeLeakInfo', description: 'Historical breach data' },
    { id: 'hibp', name: 'HaveIBeenPwned', description: 'Troy Hunt\'s breach database' }
  ];

  const riskLevels = {
    critical: { color: '#ff0000', label: 'CRITICAL' },
    high: { color: '#ff9500', label: 'HIGH' },
    medium: { color: '#ffff00', label: 'MEDIUM' },
    low: { color: '#00ff41', label: 'LOW' }
  };

  function startBreachLookup() {
    if (!searchTarget.trim()) return;
    
    dispatch('startAnalysis', {
      target: searchTarget,
      options: {
        type: searchType,
        sources: selectedSources,
        ...searchOptions
      }
    });
    
    searchTarget = '';
  }

  function toggleSource(sourceId) {
    if (selectedSources.includes(sourceId)) {
      selectedSources = selectedSources.filter(s => s !== sourceId);
    } else {
      selectedSources = [...selectedSources, sourceId];
    }
  }

  function calculateRiskScore(breachData) {
    if (!breachData) return 0;
    
    let score = 0;
    if (breachData.passwords) score += 40;
    if (breachData.emails) score += 20;
    if (breachData.names) score += 15;
    if (breachData.addresses) score += 15;
    if (breachData.phones) score += 10;
    
    return Math.min(100, score);
  }

  function getRiskLevel(score) {
    if (score >= 80) return 'critical';
    if (score >= 60) return 'high';
    if (score >= 30) return 'medium';
    return 'low';
  }

  function exportBreachResults(job) {
    const exportData = {
      target: job.target,
      searchType: searchType,
      sources: selectedSources,
      timestamp: job.startTime,
      results: job.results,
      riskScore: calculateRiskScore(job.results),
      exportTimestamp: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `breach-lookup-${job.target.replace(/[^a-zA-Z0-9]/g, '_')}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="breach-lookup space-y-6">
  <!-- Header -->
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center gap-3">
        <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>
          ← Back
        </Button>
        <span class="text-2xl">🔓</span>
        <div>
          <h2 class="text-lg font-semibold text-dark-text-primary">Breach Database Lookup</h2>
          <div class="text-sm text-dark-text-secondary">
            Search compromised credentials across multiple breach databases
          </div>
        </div>
        {#if analyzer}
          <Badge variant={analyzer.status === 'online' ? 'success' : 'danger'}>
            {analyzer.status.toUpperCase()}
          </Badge>
        {/if}
      </div>
    </div>
  </Card>

  <!-- Search Interface -->
  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    <div class="xl:col-span-2">
      <Card variant="bordered">
        <div class="p-6">
          <h3 class="text-md font-medium text-dark-text-primary mb-4">Search Parameters</h3>
          
          <!-- Search Input -->
          <div class="mb-4">
            <div class="flex gap-3">
              <select 
                bind:value={searchType}
                class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary focus:border-green-500 focus:outline-none"
              >
                <option value="email">Email Address</option>
                <option value="username">Username</option>
                <option value="domain">Domain</option>
              </select>
              
              <input
                bind:value={searchTarget}
                placeholder={searchType === 'email' ? 'user@domain.com' : searchType === 'username' ? 'username' : 'domain.com'}
                class="flex-1 px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary placeholder-dark-text-tertiary focus:border-green-500 focus:outline-none"
                on:keypress={(e) => e.key === 'Enter' && startBreachLookup()}
              />
              
              <Button 
                variant="primary" 
                on:click={startBreachLookup}
                disabled={!searchTarget.trim() || selectedSources.length === 0}
              >
                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                Search Breaches
              </Button>
            </div>
          </div>

          <!-- Breach Sources -->
          <div class="mb-4">
            <h4 class="text-sm font-medium text-dark-text-primary mb-3">Breach Sources</h4>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
              {#each breachSources as source}
                <div 
                  class="source-option p-3 rounded border cursor-pointer transition-all {
                    selectedSources.includes(source.id) 
                      ? 'border-green-500 bg-green-500/10' 
                      : 'border-dark-border bg-dark-bg-tertiary hover:border-dark-text-tertiary'
                  }"
                  on:click={() => toggleSource(source.id)}
                >
                  <div class="flex items-center justify-between">
                    <div>
                      <div class="text-sm font-medium text-dark-text-primary">{source.name}</div>
                      <div class="text-xs text-dark-text-secondary">{source.description}</div>
                    </div>
                    <div class="w-5 h-5 rounded border-2 flex items-center justify-center {
                      selectedSources.includes(source.id) 
                        ? 'border-green-500 bg-green-500' 
                        : 'border-dark-border'
                    }">
                      {#if selectedSources.includes(source.id)}
                        <svg class="w-3 h-3 text-black" fill="currentColor" viewBox="0 0 20 20">
                          <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                        </svg>
                      {/if}
                    </div>
                  </div>
                </div>
              {/each}
            </div>
          </div>

          <!-- Search Options -->
          <div class="search-options">
            <h4 class="text-sm font-medium text-dark-text-primary mb-3">Search Options</h4>
            <div class="grid grid-cols-2 gap-3">
              <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={searchOptions.includePasswords} class="checkbox" />
                <span class="text-sm text-dark-text-secondary">Include Passwords</span>
              </label>
              <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={searchOptions.includeDates} class="checkbox" />
                <span class="text-sm text-dark-text-secondary">Include Dates</span>
              </label>
              <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={searchOptions.includeNames} class="checkbox" />
                <span class="text-sm text-dark-text-secondary">Include Names</span>
              </label>
              <label class="flex items-center gap-2">
                <input type="checkbox" bind:checked={searchOptions.verifyEmail} class="checkbox" />
                <span class="text-sm text-dark-text-secondary">Verify Email</span>
              </label>
            </div>
          </div>
        </div>
      </Card>
    </div>

    <!-- Results Sidebar -->
    <div class="space-y-4">
      <!-- Active Jobs -->
      {#if activeJobs.length > 0}
        <Card variant="bordered">
          <div class="p-4">
            <h4 class="text-sm font-medium text-dark-text-primary mb-3">Active Lookups</h4>
            <div class="space-y-2">
              {#each activeJobs as job}
                <div class="job p-3 bg-dark-bg-tertiary rounded border border-dark-border">
                  <div class="flex items-center justify-between mb-2">
                    <span class="text-sm text-dark-text-primary truncate">{job.target}</span>
                    <Badge variant="warning" size="xs">{job.status.toUpperCase()}</Badge>
                  </div>
                  <div class="w-full bg-dark-bg-primary rounded-full h-1.5">
                    <div 
                      class="bg-green-600 h-1.5 rounded-full transition-all"
                      style="width: {job.progress}%"
                    ></div>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        </Card>
      {/if}

      <!-- Recent Results -->
      {#if completedJobs.length > 0}
        <Card variant="bordered">
          <div class="p-4">
            <h4 class="text-sm font-medium text-dark-text-primary mb-3">Recent Results</h4>
            <div class="space-y-2 max-h-64 overflow-y-auto">
              {#each completedJobs.slice(0, 10) as job}
                {@const riskScore = calculateRiskScore(job.results)}
                {@const riskLevel = getRiskLevel(riskScore)}
                
                <div class="result p-3 bg-dark-bg-tertiary rounded border border-dark-border">
                  <div class="flex items-center justify-between mb-2">
                    <span class="text-sm text-dark-text-primary truncate">{job.target}</span>
                    <div class="flex items-center gap-2">
                      <Badge 
                        variant={job.status === 'completed' ? 'success' : 'danger'} 
                        size="xs"
                      >
                        {job.status.toUpperCase()}
                      </Badge>
                      <button 
                        class="text-xs text-dark-text-tertiary hover:text-cyan-400"
                        on:click={() => exportBreachResults(job)}
                      >
                        💾
                      </button>
                    </div>
                  </div>
                  
                  {#if job.results && !job.error}
                    <div class="risk-score mb-2">
                      <div class="flex items-center justify-between text-xs">
                        <span class="text-dark-text-tertiary">Risk Score:</span>
                        <span 
                          class="font-bold"
                          style="color: {riskLevels[riskLevel]?.color}"
                        >
                          {riskScore}/100 ({riskLevels[riskLevel]?.label})
                        </span>
                      </div>
                    </div>
                    
                    <div class="breach-summary text-xs text-dark-text-secondary">
                      {job.results.breaches?.length || 0} breaches found
                      {#if job.results.passwords}
                        • Passwords exposed
                      {/if}
                    </div>
                  {:else if job.error}
                    <div class="text-xs text-red-400">{job.error}</div>
                  {/if}
                  
                  <div class="text-xs text-dark-text-tertiary mt-2">
                    {new Date(job.startTime).toLocaleDateString()}
                  </div>
                </div>
              {/each}
            </div>
          </div>
        </Card>
      {/if}
    </div>
  </div>

  <!-- Detailed Results -->
  {#if completedJobs.length > 0}
    <Card variant="bordered">
      <div class="p-6">
        <h3 class="text-md font-medium text-dark-text-primary mb-4">Detailed Breach Analysis</h3>
        
        {#each completedJobs.filter(job => job.results && !job.error).slice(0, 5) as job}
          {@const riskScore = calculateRiskScore(job.results)}
          {@const riskLevel = getRiskLevel(riskScore)}
          
          <div class="breach-result mb-6 p-4 bg-dark-bg-tertiary rounded border border-dark-border">
            <div class="flex items-center justify-between mb-4">
              <div>
                <h4 class="text-md font-semibold text-dark-text-primary">{job.target}</h4>
                <div class="text-sm text-dark-text-secondary">
                  Searched {job.results.sources?.length || 0} sources • 
                  Found {job.results.breaches?.length || 0} breaches
                </div>
              </div>
              
              <div class="flex items-center gap-3">
                <div class="risk-indicator text-center">
                  <div 
                    class="text-lg font-bold"
                    style="color: {riskLevels[riskLevel]?.color}"
                  >
                    {riskScore}/100
                  </div>
                  <div class="text-xs text-dark-text-tertiary">Risk Score</div>
                </div>
                
                <Button variant="outline" size="sm" on:click={() => exportBreachResults(job)}>
                  Export
                </Button>
              </div>
            </div>

            <!-- Breach Details -->
            {#if job.results.breaches && job.results.breaches.length > 0}
              <div class="breaches-list">
                <h5 class="text-sm font-medium text-dark-text-primary mb-3">Found in Breaches:</h5>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {#each job.results.breaches.slice(0, 6) as breach}
                    <div class="breach-item p-3 bg-dark-bg-primary rounded border border-dark-border">
                      <div class="flex items-center justify-between mb-2">
                        <span class="text-sm font-medium text-dark-text-primary">
                          {breach.source || 'Unknown Source'}
                        </span>
                        <Badge variant="danger" size="xs">BREACH</Badge>
                      </div>
                      
                      <div class="breach-details space-y-1 text-xs">
                        {#if breach.date}
                          <div class="flex justify-between">
                            <span class="text-dark-text-tertiary">Date:</span>
                            <span class="text-dark-text-secondary">{breach.date}</span>
                          </div>
                        {/if}
                        {#if breach.records}
                          <div class="flex justify-between">
                            <span class="text-dark-text-tertiary">Records:</span>
                            <span class="text-dark-text-secondary">{breach.records.toLocaleString()}</span>
                          </div>
                        {/if}
                        {#if breach.dataClasses}
                          <div class="data-classes mt-2">
                            <div class="text-dark-text-tertiary mb-1">Exposed Data:</div>
                            <div class="flex flex-wrap gap-1">
                              {#each breach.dataClasses.slice(0, 4) as dataClass}
                                <span class="px-1 py-0.5 text-xs bg-red-600/20 text-red-400 rounded">
                                  {dataClass}
                                </span>
                              {/each}
                              {#if breach.dataClasses.length > 4}
                                <span class="text-xs text-dark-text-tertiary">
                                  +{breach.dataClasses.length - 4} more
                                </span>
                              {/if}
                            </div>
                          </div>
                        {/if}
                      </div>
                    </div>
                  {/each}
                  
                  {#if job.results.breaches.length > 6}
                    <div class="more-breaches p-3 bg-dark-bg-primary rounded border border-dark-border text-center">
                      <div class="text-sm text-dark-text-secondary">
                        +{job.results.breaches.length - 6} more breaches
                      </div>
                      <Button variant="outline" size="xs" on:click={() => exportBreachResults(job)}>
                        View All
                      </Button>
                    </div>
                  {/if}
                </div>
              </div>
            {:else}
              <div class="no-breaches text-center py-6">
                <svg class="w-12 h-12 text-green-400 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div class="text-md font-medium text-green-400 mb-1">No Breaches Found</div>
                <div class="text-sm text-dark-text-secondary">
                  {job.target} was not found in any breach databases
                </div>
              </div>
            {/if}
          </div>
        {/each}
      </div>
    </Card>
  {/if}
</div>

<style>
  .checkbox {
    @apply w-4 h-4 rounded border-2 border-dark-border;
    background: var(--dark-bg-tertiary, #0f0f0f);
    accent-color: #00ff41;
  }

  .breach-result {
    @apply transition-all hover:bg-dark-bg-secondary;
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