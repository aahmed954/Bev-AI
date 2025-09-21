<!-- Document Analysis Workflow Component -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import * as echarts from 'echarts';
  
  const dispatch = createEventDispatcher();
  
  export let jobs = [];
  
  let analyticsChart: HTMLElement;
  let confidenceChart: HTMLElement;
  let performanceChart: HTMLElement;
  let selectedTimeRange = '7d';
  let selectedMetric = 'confidence';
  let analysisMode = 'overview'; // 'overview', 'trends', 'quality'
  
  let chartInstances = {
    analytics: null,
    confidence: null,
    performance: null
  };

  const timeRanges = [
    { value: '1h', label: 'Last Hour' },
    { value: '24h', label: 'Last 24 Hours' },
    { value: '7d', label: 'Last 7 Days' },
    { value: '30d', label: 'Last 30 Days' },
    { value: 'all', label: 'All Time' }
  ];

  const metrics = [
    { value: 'confidence', label: 'Confidence Scores' },
    { value: 'performance', label: 'Processing Time' },
    { value: 'accuracy', label: 'Cross-Engine Accuracy' },
    { value: 'volume', label: 'Processing Volume' }
  ];

  onMount(() => {
    initializeCharts();
    updateAnalytics();
    
    return () => {
      Object.values(chartInstances).forEach(chart => {
        if (chart) chart.dispose();
      });
    };
  });

  function initializeCharts() {
    // Analytics Overview Chart
    if (analyticsChart) {
      chartInstances.analytics = echarts.init(analyticsChart, 'dark');
      chartInstances.analytics.setOption({
        title: {
          text: 'OCR Processing Analytics',
          textStyle: { color: '#00ff41', fontSize: 16 }
        },
        tooltip: { 
          trigger: 'axis',
          backgroundColor: '#1a1a1a',
          borderColor: '#00ff41'
        },
        legend: {
          data: ['Success Rate', 'Avg Confidence', 'Processing Speed'],
          textStyle: { color: '#00ff4199' }
        },
        xAxis: {
          type: 'time',
          axisLine: { lineStyle: { color: '#00ff4133' } },
          axisLabel: { color: '#00ff4166' }
        },
        yAxis: {
          type: 'value',
          axisLine: { lineStyle: { color: '#00ff4133' } },
          axisLabel: { color: '#00ff4166' },
          splitLine: { lineStyle: { color: '#00ff4111' } }
        },
        series: [
          {
            name: 'Success Rate',
            type: 'line',
            smooth: true,
            data: [],
            lineStyle: { color: '#00ff41' },
            itemStyle: { color: '#00ff41' }
          },
          {
            name: 'Avg Confidence',
            type: 'line',
            smooth: true,
            data: [],
            lineStyle: { color: '#00ccff' },
            itemStyle: { color: '#00ccff' }
          },
          {
            name: 'Processing Speed',
            type: 'line',
            smooth: true,
            data: [],
            lineStyle: { color: '#ff9500' },
            itemStyle: { color: '#ff9500' }
          }
        ],
        grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true }
      });
    }

    // Confidence Distribution Chart
    if (confidenceChart) {
      chartInstances.confidence = echarts.init(confidenceChart, 'dark');
      chartInstances.confidence.setOption({
        title: {
          text: 'Confidence Score Distribution',
          textStyle: { color: '#00ff41', fontSize: 16 }
        },
        tooltip: {
          trigger: 'item',
          backgroundColor: '#1a1a1a',
          borderColor: '#00ff41'
        },
        series: [{
          type: 'pie',
          radius: ['40%', '70%'],
          avoidLabelOverlap: false,
          itemStyle: { borderRadius: 10, borderColor: '#0a0a0a', borderWidth: 2 },
          label: { show: false },
          emphasis: { label: { show: true, fontSize: 14, fontWeight: 'bold' } },
          labelLine: { show: false },
          data: []
        }]
      });
    }

    // Performance Trends Chart
    if (performanceChart) {
      chartInstances.performance = echarts.init(performanceChart, 'dark');
      chartInstances.performance.setOption({
        title: {
          text: 'Engine Performance Comparison',
          textStyle: { color: '#00ff41', fontSize: 16 }
        },
        tooltip: {
          trigger: 'axis',
          backgroundColor: '#1a1a1a',
          borderColor: '#00ff41'
        },
        legend: {
          data: ['Tesseract', 'EasyOCR', 'TrOCR', 'Hybrid'],
          textStyle: { color: '#00ff4199' }
        },
        radar: {
          indicator: [
            { name: 'Speed', max: 100 },
            { name: 'Accuracy', max: 100 },
            { name: 'Confidence', max: 100 },
            { name: 'Layout', max: 100 },
            { name: 'Languages', max: 100 }
          ],
          axisLine: { lineStyle: { color: '#00ff4133' } },
          splitLine: { lineStyle: { color: '#00ff4122' } },
          splitArea: { show: false },
          axisLabel: { color: '#00ff4166' }
        },
        series: [{
          type: 'radar',
          data: []
        }]
      });
    }
  }

  function updateAnalytics() {
    if (!jobs.length) return;

    // Filter jobs by time range
    const filteredJobs = filterJobsByTimeRange(jobs, selectedTimeRange);
    
    // Update analytics chart
    if (chartInstances.analytics) {
      const timeSeriesData = generateTimeSeriesData(filteredJobs);
      chartInstances.analytics.setOption({
        series: [
          { data: timeSeriesData.successRate },
          { data: timeSeriesData.avgConfidence },
          { data: timeSeriesData.processingSpeed }
        ]
      });
    }

    // Update confidence distribution
    if (chartInstances.confidence) {
      const confidenceData = generateConfidenceDistribution(filteredJobs);
      chartInstances.confidence.setOption({
        series: [{ data: confidenceData }]
      });
    }

    // Update performance radar
    if (chartInstances.performance) {
      const performanceData = generatePerformanceComparison(filteredJobs);
      chartInstances.performance.setOption({
        series: [{ data: performanceData }]
      });
    }
  }

  function filterJobsByTimeRange(jobsList, range) {
    if (range === 'all') return jobsList;
    
    const now = new Date();
    const cutoff = new Date();
    
    switch (range) {
      case '1h':
        cutoff.setHours(now.getHours() - 1);
        break;
      case '24h':
        cutoff.setDate(now.getDate() - 1);
        break;
      case '7d':
        cutoff.setDate(now.getDate() - 7);
        break;
      case '30d':
        cutoff.setDate(now.getDate() - 30);
        break;
    }
    
    return jobsList.filter(job => new Date(job.timestamp) >= cutoff);
  }

  function generateTimeSeriesData(jobsList) {
    // Group jobs by hour/day based on time range
    const groupedData = {};
    
    jobsList.forEach(job => {
      const date = new Date(job.timestamp);
      const key = selectedTimeRange === '1h' || selectedTimeRange === '24h' 
        ? date.toISOString().slice(0, 13) // Group by hour
        : date.toISOString().slice(0, 10); // Group by day
      
      if (!groupedData[key]) {
        groupedData[key] = { total: 0, successful: 0, totalConfidence: 0, totalTime: 0 };
      }
      
      groupedData[key].total++;
      if (job.status === 'completed') {
        groupedData[key].successful++;
        if (job.results) {
          const avgConfidence = job.results.reduce((sum, r) => sum + r.confidence, 0) / job.results.length;
          const totalTime = job.results.reduce((sum, r) => sum + r.processingTime, 0);
          groupedData[key].totalConfidence += avgConfidence;
          groupedData[key].totalTime += totalTime;
        }
      }
    });

    const sortedKeys = Object.keys(groupedData).sort();
    
    return {
      successRate: sortedKeys.map(key => [
        new Date(key).getTime(),
        (groupedData[key].successful / groupedData[key].total * 100).toFixed(1)
      ]),
      avgConfidence: sortedKeys.map(key => [
        new Date(key).getTime(),
        groupedData[key].successful > 0 
          ? (groupedData[key].totalConfidence / groupedData[key].successful * 100).toFixed(1)
          : 0
      ]),
      processingSpeed: sortedKeys.map(key => [
        new Date(key).getTime(),
        groupedData[key].successful > 0 
          ? (1000 / (groupedData[key].totalTime / groupedData[key].successful)).toFixed(1)
          : 0
      ])
    };
  }

  function generateConfidenceDistribution(jobsList) {
    const ranges = [
      { name: 'Excellent (90-100%)', min: 0.9, max: 1.0, color: '#00ff41' },
      { name: 'Good (80-90%)', min: 0.8, max: 0.9, color: '#00ccff' },
      { name: 'Fair (60-80%)', min: 0.6, max: 0.8, color: '#ffff00' },
      { name: 'Poor (40-60%)', min: 0.4, max: 0.6, color: '#ff9500' },
      { name: 'Very Poor (<40%)', min: 0.0, max: 0.4, color: '#ff0000' }
    ];

    const distribution = ranges.map(range => ({
      name: range.name,
      value: 0,
      itemStyle: { color: range.color }
    }));

    jobsList.forEach(job => {
      if (job.results) {
        job.results.forEach(result => {
          const range = ranges.find(r => result.confidence >= r.min && result.confidence < r.max);
          if (range) {
            const index = ranges.indexOf(range);
            distribution[index].value++;
          }
        });
      }
    });

    return distribution;
  }

  function generatePerformanceComparison(jobsList) {
    const engines = ['tesseract', 'easyocr', 'trocr', 'hybrid'];
    const engineData = {};

    engines.forEach(engine => {
      engineData[engine] = {
        speed: 0,
        accuracy: 0,
        confidence: 0,
        layout: 0,
        languages: 0,
        count: 0
      };
    });

    jobsList.forEach(job => {
      if (job.results) {
        job.results.forEach(result => {
          const engine = result.engine;
          if (engineData[engine]) {
            engineData[engine].speed += result.processingTime > 0 ? (1000 / result.processingTime) * 10 : 0;
            engineData[engine].confidence += result.confidence * 100;
            engineData[engine].accuracy += result.confidence * 100; // Simplified
            engineData[engine].layout += result.layout ? 80 : 40;
            engineData[engine].languages += 70; // Simplified
            engineData[engine].count++;
          }
        });
      }
    });

    return engines.map(engine => {
      const data = engineData[engine];
      const count = data.count || 1;
      
      return {
        name: engine.charAt(0).toUpperCase() + engine.slice(1),
        value: [
          (data.speed / count).toFixed(1),
          (data.accuracy / count).toFixed(1),
          (data.confidence / count).toFixed(1),
          (data.layout / count).toFixed(1),
          (data.languages / count).toFixed(1)
        ],
        itemStyle: {
          color: engine === 'tesseract' ? '#00ff41' :
                engine === 'easyocr' ? '#00ccff' :
                engine === 'trocr' ? '#ff9500' : '#ff00ff'
        }
      };
    });
  }

  function calculateQualityMetrics(jobsList) {
    const completedJobs = jobsList.filter(job => job.status === 'completed' && job.results);
    
    if (completedJobs.length === 0) {
      return {
        avgConfidence: 0,
        avgProcessingTime: 0,
        successRate: 0,
        totalCharacters: 0,
        mostReliableEngine: 'N/A',
        fastestEngine: 'N/A'
      };
    }

    const allResults = completedJobs.flatMap(job => job.results || []);
    const avgConfidence = allResults.reduce((sum, r) => sum + r.confidence, 0) / allResults.length;
    const avgProcessingTime = allResults.reduce((sum, r) => sum + r.processingTime, 0) / allResults.length;
    const successRate = completedJobs.length / jobsList.length;
    const totalCharacters = allResults.reduce((sum, r) => sum + r.text.length, 0);

    // Find most reliable engine
    const engineStats = {};
    allResults.forEach(result => {
      if (!engineStats[result.engine]) {
        engineStats[result.engine] = { confidence: 0, count: 0, time: 0 };
      }
      engineStats[result.engine].confidence += result.confidence;
      engineStats[result.engine].time += result.processingTime;
      engineStats[result.engine].count++;
    });

    const mostReliableEngine = Object.entries(engineStats)
      .map(([engine, stats]) => ({
        engine,
        avgConfidence: stats.confidence / stats.count,
        avgTime: stats.time / stats.count
      }))
      .sort((a, b) => b.avgConfidence - a.avgConfidence)[0];

    const fastestEngine = Object.entries(engineStats)
      .map(([engine, stats]) => ({
        engine,
        avgTime: stats.time / stats.count
      }))
      .sort((a, b) => a.avgTime - b.avgTime)[0];

    return {
      avgConfidence: avgConfidence * 100,
      avgProcessingTime,
      successRate: successRate * 100,
      totalCharacters,
      mostReliableEngine: mostReliableEngine?.engine || 'N/A',
      fastestEngine: fastestEngine?.engine || 'N/A'
    };
  }

  function exportAnalysisReport() {
    const metrics = calculateQualityMetrics(jobs);
    const timeSeriesData = generateTimeSeriesData(filterJobsByTimeRange(jobs, selectedTimeRange));
    
    const report = {
      reportDate: new Date().toISOString(),
      timeRange: selectedTimeRange,
      summary: metrics,
      trends: timeSeriesData,
      jobDetails: jobs.map(job => ({
        id: job.id,
        filename: job.filename,
        status: job.status,
        engines: job.engines,
        avgConfidence: job.results?.reduce((sum, r) => sum + r.confidence, 0) / (job.results?.length || 1),
        totalTime: job.results?.reduce((sum, r) => sum + r.processingTime, 0) || 0,
        textLength: job.results?.[0]?.text?.length || 0
      }))
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bev-ocr-analysis-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  $: qualityMetrics = calculateQualityMetrics(filterJobsByTimeRange(jobs, selectedTimeRange));
  
  $: {
    if (selectedTimeRange || selectedMetric) {
      updateAnalytics();
    }
  }
</script>

<div class="document-analysis space-y-6">
  <!-- Analysis Controls -->
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex flex-col md:flex-row gap-4 items-start md:items-center justify-between">
        <div class="flex items-center gap-4">
          <h2 class="text-lg font-semibold text-dark-text-primary">Document Analysis</h2>
          <div class="flex bg-dark-bg-tertiary rounded-lg p-1">
            {#each ['overview', 'trends', 'quality'] as mode}
              <button
                class="px-3 py-1 text-xs font-medium rounded transition-colors {
                  analysisMode === mode 
                    ? 'bg-green-600 text-black' 
                    : 'text-dark-text-secondary hover:text-dark-text-primary'
                }"
                on:click={() => analysisMode = mode}
              >
                {mode.toUpperCase()}
              </button>
            {/each}
          </div>
        </div>
        
        <div class="flex items-center gap-3">
          <select 
            bind:value={selectedTimeRange}
            class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-sm focus:border-green-500 focus:outline-none"
          >
            {#each timeRanges as range}
              <option value={range.value}>{range.label}</option>
            {/each}
          </select>
          
          <select 
            bind:value={selectedMetric}
            class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-sm focus:border-green-500 focus:outline-none"
          >
            {#each metrics as metric}
              <option value={metric.value}>{metric.label}</option>
            {/each}
          </select>
          
          <Button variant="outline" size="sm" on:click={exportAnalysisReport}>
            Export Report
          </Button>
        </div>
      </div>
    </div>
  </Card>

  {#if analysisMode === 'overview'}
    <!-- Quality Metrics Overview -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <Card variant="bordered">
        <div class="p-4 text-center">
          <div class="text-xs text-dark-text-tertiary mb-1">Success Rate</div>
          <div class="text-2xl font-bold text-green-400 mb-1">
            {qualityMetrics.successRate.toFixed(1)}%
          </div>
          <div class="text-xs text-dark-text-secondary">
            of {filterJobsByTimeRange(jobs, selectedTimeRange).length} jobs
          </div>
        </div>
      </Card>

      <Card variant="bordered">
        <div class="p-4 text-center">
          <div class="text-xs text-dark-text-tertiary mb-1">Avg Confidence</div>
          <div class="text-2xl font-bold text-cyan-400 mb-1">
            {qualityMetrics.avgConfidence.toFixed(1)}%
          </div>
          <div class="text-xs text-dark-text-secondary">
            across all engines
          </div>
        </div>
      </Card>

      <Card variant="bordered">
        <div class="p-4 text-center">
          <div class="text-xs text-dark-text-tertiary mb-1">Avg Processing</div>
          <div class="text-2xl font-bold text-purple-400 mb-1">
            {qualityMetrics.avgProcessingTime.toFixed(0)}ms
          </div>
          <div class="text-xs text-dark-text-secondary">
            per document
          </div>
        </div>
      </Card>

      <Card variant="bordered">
        <div class="p-4 text-center">
          <div class="text-xs text-dark-text-tertiary mb-1">Characters</div>
          <div class="text-2xl font-bold text-yellow-400 mb-1">
            {qualityMetrics.totalCharacters.toLocaleString()}
          </div>
          <div class="text-xs text-dark-text-secondary">
            total extracted
          </div>
        </div>
      </Card>
    </div>

    <!-- Engine Recommendations -->
    <Card variant="bordered">
      <div class="p-4">
        <h3 class="text-md font-medium text-dark-text-primary mb-4">Engine Recommendations</h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="recommendation">
            <div class="flex items-center gap-2 mb-2">
              <span class="text-lg">🏆</span>
              <span class="text-sm font-medium text-green-400">Most Reliable</span>
            </div>
            <div class="text-lg font-semibold text-dark-text-primary">
              {qualityMetrics.mostReliableEngine}
            </div>
            <div class="text-xs text-dark-text-secondary">
              Best average confidence scores
            </div>
          </div>
          
          <div class="recommendation">
            <div class="flex items-center gap-2 mb-2">
              <span class="text-lg">⚡</span>
              <span class="text-sm font-medium text-cyan-400">Fastest</span>
            </div>
            <div class="text-lg font-semibold text-dark-text-primary">
              {qualityMetrics.fastestEngine}
            </div>
            <div class="text-xs text-dark-text-secondary">
              Shortest processing times
            </div>
          </div>
        </div>
      </div>
    </Card>
  {/if}

  {#if analysisMode === 'trends'}
    <!-- Trends Analysis -->
    <Card variant="bordered">
      <div class="p-4">
        <div bind:this={analyticsChart} class="w-full h-80"></div>
      </div>
    </Card>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <Card variant="bordered">
        <div class="p-4">
          <div bind:this={confidenceChart} class="w-full h-64"></div>
        </div>
      </Card>
      
      <Card variant="bordered">
        <div class="p-4">
          <div bind:this={performanceChart} class="w-full h-64"></div>
        </div>
      </Card>
    </div>
  {/if}

  {#if analysisMode === 'quality'}
    <!-- Quality Analysis -->
    <Card variant="bordered">
      <div class="p-4">
        <h3 class="text-md font-medium text-dark-text-primary mb-4">Quality Analysis</h3>
        
        <!-- Recent Jobs List -->
        <div class="space-y-3 max-h-96 overflow-y-auto">
          {#each filterJobsByTimeRange(jobs, selectedTimeRange).slice(0, 20) as job}
            <div 
              class="job-quality-item p-3 bg-dark-bg-tertiary rounded border border-dark-border cursor-pointer hover:border-green-500 transition-colors"
              on:click={() => dispatch('jobSelected', job)}
            >
              <div class="flex items-center justify-between mb-2">
                <span class="text-sm font-medium text-dark-text-primary truncate">
                  {job.filename}
                </span>
                <div class="flex items-center gap-2">
                  {#if job.results}
                    <Badge 
                      variant={job.results[0]?.confidence > 0.8 ? 'success' : job.results[0]?.confidence > 0.6 ? 'warning' : 'danger'} 
                      size="xs"
                    >
                      {(job.results[0]?.confidence * 100).toFixed(0)}%
                    </Badge>
                  {/if}
                  <button 
                    class="text-dark-text-tertiary hover:text-green-400"
                    on:click|stopPropagation={() => dispatch('viewComparison', job)}
                    title="View detailed comparison"
                  >
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                    </svg>
                  </button>
                </div>
              </div>
              
              {#if job.results}
                <div class="text-xs text-dark-text-secondary">
                  {job.engines.length} engines • {job.results.reduce((sum, r) => sum + r.text.length, 0)} chars • 
                  {formatDuration(job.results.reduce((sum, r) => sum + r.processingTime, 0))}
                </div>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    </Card>
  {/if}
</div>

<style>
  .job-quality-item:hover {
    transform: translateY(-1px);
  }

  .recommendation {
    @apply p-3 bg-dark-bg-tertiary rounded border border-dark-border;
  }

  /* Chart containers */
  .chart-container {
    @apply w-full h-64 bg-dark-bg-primary rounded border border-dark-border;
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