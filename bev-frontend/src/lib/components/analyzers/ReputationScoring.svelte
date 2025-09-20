<!-- Threat Reputation Scoring Engine Interface -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import * as echarts from 'echarts';
  import { onMount } from 'svelte';
  
  const dispatch = createEventDispatcher();
  
  export let analyzer;
  export let activeJobs = [];
  export let completedJobs = [];
  
  let searchTarget = '';
  let targetType = 'ip'; // 'ip', 'domain', 'url', 'hash'
  let reputationSources = ['virustotal', 'abuseipdb', 'urlvoid', 'hybrid-analysis'];
  let selectedSources = ['virustotal', 'abuseipdb'];
  let riskThreshold = 50;
  let chartContainer: HTMLElement;
  let chartInstance: any;

  const targetTypes = [
    { value: 'ip', label: 'IP Address', placeholder: '192.168.1.1', pattern: /^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/ },
    { value: 'domain', label: 'Domain', placeholder: 'example.com', pattern: /^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/ },
    { value: 'url', label: 'URL', placeholder: 'https://example.com/path', pattern: /^https?:\/\/.+/ },
    { value: 'hash', label: 'File Hash', placeholder: 'md5/sha1/sha256', pattern: /^[a-fA-F0-9]{32,64}$/ }
  ];

  const reputationProviders = {
    virustotal: { name: 'VirusTotal', icon: '🦠', description: 'Multi-engine malware scanner' },
    abuseipdb: { name: 'AbuseIPDB', icon: '🚫', description: 'IP abuse database' },
    urlvoid: { name: 'URLVoid', icon: '🌐', description: 'URL reputation checker' },
    'hybrid-analysis': { name: 'Hybrid Analysis', icon: '🔬', description: 'Malware sandbox' },
    otx: { name: 'AlienVault OTX', icon: '👽', description: 'Open threat exchange' },
    threatminer: { name: 'ThreatMiner', icon: '⛏️', description: 'Threat intelligence mining' }
  };

  onMount(() => {
    initializeChart();
  });

  function initializeChart() {
    if (!chartContainer) return;
    
    chartInstance = echarts.init(chartContainer, 'dark');
    chartInstance.setOption({
      title: {
        text: 'Reputation Score Distribution',
        textStyle: { color: '#00ff41', fontSize: 16 }
      },
      tooltip: {
        trigger: 'item',
        backgroundColor: '#1a1a1a',
        borderColor: '#00ff41'
      },
      series: [{
        type: 'gauge',
        startAngle: 180,
        endAngle: 0,
        center: ['50%', '75%'],
        radius: '90%',
        min: 0,
        max: 100,
        splitNumber: 5,
        axisLine: {
          lineStyle: {
            width: 6,
            color: [
              [0.3, '#00ff41'],
              [0.7, '#ffff00'],
              [1, '#ff0000']
            ]
          }
        },
        pointer: {
          icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
          length: '12%',
          width: 20,
          offsetCenter: [0, '-60%'],
          itemStyle: { color: 'white' }
        },
        axisTick: { length: 12, lineStyle: { color: 'white', width: 2 } },
        splitLine: { length: 20, lineStyle: { color: 'white', width: 5 } },
        axisLabel: { color: 'white', fontSize: 12, distance: -60 },
        title: { offsetCenter: [0, '-10%'], fontSize: 20, color: 'white' },
        detail: {
          fontSize: 30,
          offsetCenter: [0, '-35%'],
          valueAnimation: true,
          formatter: function (value) {
            return Math.round(value) + '';
          },
          color: 'white'
        },
        data: [{ value: 0, name: 'RISK SCORE' }]
      }]
    });
  }

  function updateRiskGauge(score) {
    if (!chartInstance) return;
    
    chartInstance.setOption({
      series: [{
        data: [{ value: score, name: 'RISK SCORE' }]
      }]
    });
  }

  function validateTarget(target, type) {
    const config = targetTypes.find(t => t.value === type);
    if (!config) return false;
    return config.pattern.test(target);
  }

  function startReputationCheck() {
    if (!searchTarget.trim()) return;
    
    if (!validateTarget(searchTarget, targetType)) {
      alert(`Invalid ${targetType} format`);
      return;
    }
    
    dispatch('startAnalysis', {
      target: searchTarget,
      options: {
        type: targetType,
        sources: selectedSources,
        threshold: riskThreshold,
        includeDetails: true,
        includeMalwareFamily: true
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

  function getRiskColor(score) {
    if (score >= 80) return '#ff0000';
    if (score >= 50) return '#ff9500';
    if (score >= 20) return '#ffff00';
    return '#00ff41';
  }

  function getRiskLabel(score) {
    if (score >= 80) return 'CRITICAL';
    if (score >= 50) return 'HIGH';
    if (score >= 20) return 'MEDIUM';
    return 'LOW';
  }

  function calculateOverallScore(results) {
    if (!results || !results.scores) return 0;
    
    const scores = Object.values(results.scores);
    if (scores.length === 0) return 0;
    
    return scores.reduce((sum, score) => sum + score, 0) / scores.length;
  }

  $: if (completedJobs.length > 0) {
    const latestJob = completedJobs[0];
    if (latestJob.results) {
      const overallScore = calculateOverallScore(latestJob.results);
      updateRiskGauge(overallScore);
    }
  }
</script>

<div class="reputation-scoring space-y-6">
  <!-- Header -->
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center gap-3">
        <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>
          ← Back
        </Button>
        <span class="text-2xl">🎯</span>
        <div>
          <h2 class="text-lg font-semibold text-dark-text-primary">Threat Reputation Engine</h2>
          <div class="text-sm text-dark-text-secondary">
            Analyze threat reputation across multiple intelligence sources
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

  <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
    <!-- Analysis Interface -->
    <div class="xl:col-span-2 space-y-4">
      <!-- Target Input -->
      <Card variant="bordered">
        <div class="p-6">
          <h3 class="text-md font-medium text-dark-text-primary mb-4">Reputation Analysis</h3>
          
          <div class="target-input mb-4">
            <div class="flex gap-3 mb-3">
              <select 
                bind:value={targetType}
                class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary focus:border-green-500 focus:outline-none"
              >
                {#each targetTypes as type}
                  <option value={type.value}>{type.label}</option>
                {/each}
              </select>
              
              <input
                bind:value={searchTarget}
                placeholder={targetTypes.find(t => t.value === targetType)?.placeholder || 'Enter target'}
                class="flex-1 px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary placeholder-dark-text-tertiary focus:border-green-500 focus:outline-none"
                on:keypress={(e) => e.key === 'Enter' && startReputationCheck()}
              />
              
              <Button 
                variant="primary" 
                on:click={startReputationCheck}
                disabled={!searchTarget.trim() || selectedSources.length === 0}
              >
                Check Reputation
              </Button>
            </div>
          </div>

          <!-- Risk Threshold -->
          <div class="risk-threshold mb-4">
            <label class="block text-sm text-dark-text-primary mb-2">
              Risk Alert Threshold: {riskThreshold}
            </label>
            <input 
              type="range" 
              min="0" 
              max="100" 
              step="5"
              bind:value={riskThreshold}
              class="w-full h-2 bg-dark-bg-primary rounded-lg appearance-none cursor-pointer slider"
            />
            <div class="flex justify-between text-xs text-dark-text-tertiary mt-1">
              <span>Low Risk</span>
              <span>Medium Risk</span>
              <span>High Risk</span>
            </div>
          </div>

          <!-- Intelligence Sources -->
          <div class="sources-selection">
            <h4 class="text-sm font-medium text-dark-text-primary mb-3">Intelligence Sources</h4>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
              {#each Object.entries(reputationProviders) as [id, provider]}
                <div 
                  class="source-option p-3 rounded border cursor-pointer transition-all {
                    selectedSources.includes(id) 
                      ? 'border-green-500 bg-green-500/10' 
                      : 'border-dark-border bg-dark-bg-tertiary hover:border-dark-text-tertiary'
                  }"
                  on:click={() => toggleSource(id)}
                >
                  <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2">
                      <span class="text-lg">{provider.icon}</span>
                      <div>
                        <div class="text-sm font-medium text-dark-text-primary">{provider.name}</div>
                        <div class="text-xs text-dark-text-secondary">{provider.description}</div>
                      </div>
                    </div>
                    <div class="w-5 h-5 rounded border-2 flex items-center justify-center {
                      selectedSources.includes(id) 
                        ? 'border-green-500 bg-green-500' 
                        : 'border-dark-border'
                    }">
                      {#if selectedSources.includes(id)}
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
        </div>
      </Card>

      <!-- Results Display -->
      {#if completedJobs.length > 0}
        <Card variant="bordered">
          <div class="p-6">
            <h3 class="text-md font-medium text-dark-text-primary mb-4">Reputation Analysis Results</h3>
            
            {#each completedJobs.filter(job => job.results).slice(0, 3) as job}
              {@const overallScore = calculateOverallScore(job.results)}
              {@const riskLevel = getRiskLabel(overallScore)}
              
              <div class="reputation-result mb-6 p-4 bg-dark-bg-tertiary rounded border border-dark-border">
                <div class="flex items-center justify-between mb-4">
                  <div>
                    <h4 class="text-md font-semibold text-dark-text-primary">{job.target}</h4>
                    <div class="text-sm text-dark-text-secondary">
                      {targetTypes.find(t => t.value === targetType)?.label} • 
                      {Object.keys(job.results.scores || {}).length} sources
                    </div>
                  </div>
                  
                  <div class="risk-score text-center">
                    <div 
                      class="text-2xl font-bold"
                      style="color: {getRiskColor(overallScore)}"
                    >
                      {overallScore.toFixed(0)}
                    </div>
                    <div class="text-xs" style="color: {getRiskColor(overallScore)}">
                      {riskLevel}
                    </div>
                  </div>
                </div>

                <!-- Source Scores -->
                {#if job.results.scores}
                  <div class="source-scores mb-4">
                    <h5 class="text-sm font-medium text-dark-text-primary mb-3">Source Breakdown</h5>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                      {#each Object.entries(job.results.scores) as [source, score]}
                        <div class="source-score p-3 bg-dark-bg-primary rounded border border-dark-border">
                          <div class="flex items-center justify-between mb-2">
                            <div class="flex items-center gap-2">
                              <span class="text-lg">{reputationProviders[source]?.icon || '🔍'}</span>
                              <span class="text-sm font-medium text-dark-text-primary">
                                {reputationProviders[source]?.name || source}
                              </span>
                            </div>
                            <span 
                              class="text-sm font-bold"
                              style="color: {getRiskColor(score)}"
                            >
                              {score}/100
                            </span>
                          </div>
                          
                          <div class="w-full bg-dark-bg-tertiary rounded-full h-2">
                            <div 
                              class="h-2 rounded-full transition-all"
                              style="width: {score}%; background-color: {getRiskColor(score)}"
                            ></div>
                          </div>
                        </div>
                      {/each}
                    </div>
                  </div>
                {/if}

                <!-- Threat Details -->
                {#if job.results.details}
                  <div class="threat-details">
                    <h5 class="text-sm font-medium text-dark-text-primary mb-3">Threat Intelligence</h5>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                      {#if job.results.details.malwareFamily}
                        <div class="detail-item p-2 bg-dark-bg-primary rounded">
                          <div class="text-xs text-dark-text-tertiary">Malware Family</div>
                          <div class="text-sm text-red-400">{job.results.details.malwareFamily}</div>
                        </div>
                      {/if}
                      
                      {#if job.results.details.country}
                        <div class="detail-item p-2 bg-dark-bg-primary rounded">
                          <div class="text-xs text-dark-text-tertiary">Country</div>
                          <div class="text-sm text-dark-text-secondary">{job.results.details.country}</div>
                        </div>
                      {/if}
                      
                      {#if job.results.details.lastSeen}
                        <div class="detail-item p-2 bg-dark-bg-primary rounded">
                          <div class="text-xs text-dark-text-tertiary">Last Seen</div>
                          <div class="text-sm text-dark-text-secondary">
                            {new Date(job.results.details.lastSeen).toLocaleDateString()}
                          </div>
                        </div>
                      {/if}
                    </div>
                  </div>
                {/if}
              </div>
            {/each}
          </div>
        </Card>
      {/if}
    </div>

    <!-- Sidebar -->
    <div class="space-y-4">
      <!-- Risk Gauge -->
      <Card variant="bordered">
        <div class="p-4">
          <h4 class="text-sm font-medium text-dark-text-primary mb-3">Current Risk Assessment</h4>
          <div bind:this={chartContainer} class="w-full h-48"></div>
        </div>
      </Card>

      <!-- Active Analysis -->
      {#if activeJobs.length > 0}
        <Card variant="bordered">
          <div class="p-4">
            <h4 class="text-sm font-medium text-dark-text-primary mb-3">Active Analysis</h4>
            <div class="space-y-2">
              {#each activeJobs as job}
                <div class="active-job p-3 bg-dark-bg-tertiary rounded border border-dark-border">
                  <div class="flex items-center justify-between mb-2">
                    <span class="text-sm text-dark-text-primary truncate">{job.target}</span>
                    <Badge variant="warning" size="xs">ANALYZING</Badge>
                  </div>
                  <div class="w-full bg-dark-bg-primary rounded-full h-1.5">
                    <div 
                      class="bg-yellow-600 h-1.5 rounded-full transition-all"
                      style="width: {job.progress}%"
                    ></div>
                  </div>
                </div>
              {/each}
            </div>
          </div>
        </Card>
      {/if}

      <!-- Quick Actions -->
      <Card variant="bordered">
        <div class="p-4">
          <h4 class="text-sm font-medium text-dark-text-primary mb-3">Quick Checks</h4>
          <div class="space-y-2">
            <Button variant="outline" size="sm" fullWidth on:click={() => {
              searchTarget = '8.8.8.8';
              targetType = 'ip';
            }}>
              🔍 Check Sample IP
            </Button>
            <Button variant="outline" size="sm" fullWidth on:click={() => {
              searchTarget = 'malware-domain.com';
              targetType = 'domain';
            }}>
              🌐 Check Sample Domain
            </Button>
            <Button variant="outline" size="sm" fullWidth on:click={() => {
              searchTarget = 'https://suspicious-url.com';
              targetType = 'url';
            }}>
              🔗 Check Sample URL
            </Button>
          </div>
        </div>
      </Card>
    </div>
  </div>
</div>

<style>
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
</style>