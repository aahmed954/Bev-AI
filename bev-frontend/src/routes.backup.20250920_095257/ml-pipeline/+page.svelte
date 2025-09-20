<!-- ML Pipeline Management Platform -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import ModelManager from '$lib/components/ml/ModelManager.svelte';
  import TrainingMonitor from '$lib/components/ml/TrainingMonitor.svelte';
  import PipelineDAGs from '$lib/components/ml/PipelineDAGs.svelte';
  import GeneticOptimizer from '$lib/components/ml/GeneticOptimizer.svelte';
  import { invoke } from '@tauri-apps/api/core';
  import * as echarts from 'echarts';
  
  interface MLModel {
    id: string;
    name: string;
    version: string;
    status: 'training' | 'deployed' | 'testing' | 'archived';
    accuracy: number;
    size: number;
    framework: string;
    created: string;
    lastTrained: string;
    deployments: number;
  }

  interface TrainingJob {
    id: string;
    modelName: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    epoch: number;
    maxEpochs: number;
    loss: number;
    accuracy: number;
    startTime: string;
    estimatedCompletion?: string;
  }

  let currentView: 'overview' | 'models' | 'training' | 'dags' | 'optimizer' = 'overview';
  let models = writable<MLModel[]>([]);
  let trainingJobs = writable<TrainingJob[]>([]);
  let mlStats = writable({
    totalModels: 0,
    activeTraining: 0,
    deployedModels: 0,
    avgAccuracy: 0,
    totalInferences: 0,
    systemLoad: 0
  });

  let websocket: WebSocket | null = null;
  let connectionStatus: 'connected' | 'connecting' | 'disconnected' = 'disconnected';
  let performanceChart: HTMLElement;
  let chartInstance: any;

  onMount(() => {
    loadMLOverview();
    connectWebSocket();
    initializePerformanceChart();
  });

  onDestroy(() => {
    if (websocket) websocket.close();
    if (chartInstance) chartInstance.dispose();
  });

  async function loadMLOverview() {
    try {
      const [modelsData, trainingData, statsData] = await Promise.all([
        invoke('get_ml_models'),
        invoke('get_training_jobs'),
        invoke('get_ml_stats')
      ]);

      models.set(modelsData);
      trainingJobs.set(trainingData);
      mlStats.set(statsData);
    } catch (error) {
      console.error('Failed to load ML overview:', error);
      generateMockMLData();
    }
  }

  function generateMockMLData() {
    const mockModels = [
      {
        id: 'threat-classifier-v2',
        name: 'Threat Classifier',
        version: 'v2.1.0',
        status: 'deployed',
        accuracy: 94.2,
        size: 256.7,
        framework: 'PyTorch',
        created: '2024-01-15T10:30:00Z',
        lastTrained: '2024-02-01T14:20:00Z',
        deployments: 3
      },
      {
        id: 'anomaly-detector-v1',
        name: 'Anomaly Detector',
        version: 'v1.3.2',
        status: 'training',
        accuracy: 89.7,
        size: 128.4,
        framework: 'TensorFlow',
        created: '2024-01-20T09:15:00Z',
        lastTrained: '2024-02-10T11:45:00Z',
        deployments: 2
      }
    ];

    const mockTraining = [
      {
        id: 'training_001',
        modelName: 'Enhanced Threat Classifier',
        status: 'running',
        progress: 67,
        epoch: 23,
        maxEpochs: 50,
        loss: 0.234,
        accuracy: 92.1,
        startTime: new Date(Date.now() - 3600000).toISOString(),
        estimatedCompletion: new Date(Date.now() + 1800000).toISOString()
      }
    ];

    models.set(mockModels);
    trainingJobs.set(mockTraining);
    mlStats.set({
      totalModels: 12,
      activeTraining: 1,
      deployedModels: 8,
      avgAccuracy: 91.3,
      totalInferences: 1247893,
      systemLoad: 34.7
    });
  }

  function connectWebSocket() {
    try {
      connectionStatus = 'connecting';
      websocket = new WebSocket('ws://localhost:3022/ml-stream');

      websocket.onopen = () => {
        connectionStatus = 'connected';
        console.log('Connected to ML pipeline stream');
      };

      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMLUpdate(data);
      };

      websocket.onerror = (error) => {
        console.error('ML WebSocket error:', error);
        connectionStatus = 'disconnected';
      };

      websocket.onclose = () => {
        connectionStatus = 'disconnected';
        setTimeout(connectWebSocket, 3000);
      };
    } catch (error) {
      console.error('Failed to connect ML WebSocket:', error);
      connectionStatus = 'disconnected';
    }
  }

  function handleMLUpdate(data: any) {
    switch (data.type) {
      case 'training_progress':
        updateTrainingProgress(data.jobId, data.progress);
        break;
      case 'model_deployed':
        updateModelStatus(data.modelId, 'deployed');
        break;
      case 'training_completed':
        completeTraining(data.jobId, data.results);
        break;
      case 'system_metrics':
        updateSystemMetrics(data.metrics);
        break;
    }
  }

  function updateTrainingProgress(jobId: string, progress: any) {
    trainingJobs.update(jobs => 
      jobs.map(job => 
        job.id === jobId 
          ? { ...job, ...progress }
          : job
      )
    );
  }

  function updateModelStatus(modelId: string, status: string) {
    models.update(models => 
      models.map(model => 
        model.id === modelId 
          ? { ...model, status: status as any }
          : model
      )
    );
  }

  function completeTraining(jobId: string, results: any) {
    trainingJobs.update(jobs => 
      jobs.map(job => 
        job.id === jobId 
          ? { ...job, status: 'completed', progress: 100, ...results }
          : job
      )
    );

    // Add new model if training successful
    if (results.success) {
      const newModel: MLModel = {
        id: results.modelId,
        name: results.modelName,
        version: results.version,
        status: 'testing',
        accuracy: results.accuracy,
        size: results.size,
        framework: results.framework,
        created: new Date().toISOString(),
        lastTrained: new Date().toISOString(),
        deployments: 0
      };

      models.update(models => [newModel, ...models]);
    }
  }

  function updateSystemMetrics(metrics: any) {
    mlStats.update(stats => ({ ...stats, ...metrics }));
  }

  function initializePerformanceChart() {
    if (!performanceChart) return;
    
    chartInstance = echarts.init(performanceChart, 'dark');
    chartInstance.setOption({
      title: {
        text: 'ML System Performance',
        textStyle: { color: '#00ff41', fontSize: 16 }
      },
      tooltip: { trigger: 'axis' },
      legend: {
        data: ['GPU Utilization', 'Memory Usage', 'Model Accuracy'],
        textStyle: { color: '#00ff4199' }
      },
      xAxis: {
        type: 'time',
        axisLine: { lineStyle: { color: '#00ff4133' } },
        axisLabel: { color: '#00ff4166' }
      },
      yAxis: {
        type: 'value',
        max: 100,
        axisLine: { lineStyle: { color: '#00ff4133' } },
        axisLabel: { color: '#00ff4166' },
        splitLine: { lineStyle: { color: '#00ff4111' } }
      },
      series: [
        {
          name: 'GPU Utilization',
          type: 'line',
          smooth: true,
          data: [],
          lineStyle: { color: '#00ff41' },
          itemStyle: { color: '#00ff41' }
        },
        {
          name: 'Memory Usage',
          type: 'line',
          smooth: true,
          data: [],
          lineStyle: { color: '#00ccff' },
          itemStyle: { color: '#00ccff' }
        },
        {
          name: 'Model Accuracy',
          type: 'line',
          smooth: true,
          data: [],
          lineStyle: { color: '#ff9500' },
          itemStyle: { color: '#ff9500' }
        }
      ]
    });
  }
</script>

<div class="ml-pipeline min-h-screen bg-dark-bg-primary text-dark-text-primary">
  <!-- Header -->
  <div class="border-b border-dark-border bg-dark-bg-secondary">
    <div class="container mx-auto px-6 py-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <h1 class="text-2xl font-bold text-green-400">ML PIPELINE MANAGEMENT</h1>
          <Badge variant={connectionStatus === 'connected' ? 'success' : 'danger'}>
            {connectionStatus.toUpperCase()}
          </Badge>
        </div>
        
        <div class="flex items-center gap-3">
          <div class="flex bg-dark-bg-tertiary rounded-lg p-1">
            {#each ['overview', 'models', 'training', 'dags', 'optimizer'] as view}
              <button
                class="px-3 py-1 text-xs font-medium rounded transition-colors {
                  currentView === view 
                    ? 'bg-green-600 text-black' 
                    : 'text-dark-text-secondary hover:text-dark-text-primary'
                }"
                on:click={() => currentView = view}
              >
                {view.toUpperCase()}
              </button>
            {/each}
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Stats Bar -->
  <div class="bg-dark-bg-secondary border-b border-dark-border">
    <div class="container mx-auto px-6 py-3">
      <div class="grid grid-cols-2 md:grid-cols-6 gap-4 text-center">
        <div>
          <div class="text-xs text-dark-text-tertiary">TOTAL MODELS</div>
          <div class="text-lg font-bold text-green-400">{$mlStats.totalModels}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">DEPLOYED</div>
          <div class="text-lg font-bold text-cyan-400">{$mlStats.deployedModels}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">TRAINING</div>
          <div class="text-lg font-bold text-yellow-400">{$mlStats.activeTraining}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">AVG ACCURACY</div>
          <div class="text-lg font-bold text-purple-400">{$mlStats.avgAccuracy.toFixed(1)}%</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">INFERENCES</div>
          <div class="text-lg font-bold text-blue-400">{$mlStats.totalInferences.toLocaleString()}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">SYSTEM LOAD</div>
          <div class="text-lg font-bold text-red-400">{$mlStats.systemLoad.toFixed(1)}%</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Content -->
  <div class="container mx-auto px-6 py-6">
    {#if currentView === 'overview'}
      <!-- Overview Dashboard -->
      <div class="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <!-- Performance Chart -->
        <div class="xl:col-span-2">
          <Card variant="bordered">
            <div class="p-6">
              <h3 class="text-md font-medium text-dark-text-primary mb-4">System Performance</h3>
              <div bind:this={performanceChart} class="w-full h-80"></div>
            </div>
          </Card>
        </div>

        <!-- Quick Stats -->
        <div class="space-y-4">
          <Card variant="bordered">
            <div class="p-4">
              <h4 class="text-sm font-medium text-dark-text-primary mb-3">Active Training</h4>
              <div class="space-y-2">
                {#each $trainingJobs.filter(job => job.status === 'running').slice(0, 3) as job}
                  <div class="training-item p-3 bg-dark-bg-tertiary rounded border border-dark-border">
                    <div class="flex items-center justify-between mb-2">
                      <span class="text-sm text-dark-text-primary truncate">{job.modelName}</span>
                      <span class="text-xs text-dark-text-secondary">{job.progress}%</span>
                    </div>
                    <div class="w-full bg-dark-bg-primary rounded-full h-1.5">
                      <div 
                        class="bg-green-600 h-1.5 rounded-full transition-all"
                        style="width: {job.progress}%"
                      ></div>
                    </div>
                    <div class="text-xs text-dark-text-tertiary mt-1">
                      Epoch {job.epoch}/{job.maxEpochs} • Loss: {job.loss.toFixed(4)}
                    </div>
                  </div>
                {/each}
                
                {#if $trainingJobs.filter(job => job.status === 'running').length === 0}
                  <div class="text-center py-4 text-dark-text-tertiary">
                    <p class="text-xs">No active training jobs</p>
                  </div>
                {/if}
              </div>
            </div>
          </Card>

          <Card variant="bordered">
            <div class="p-4">
              <h4 class="text-sm font-medium text-dark-text-primary mb-3">Top Models</h4>
              <div class="space-y-2">
                {#each $models.filter(m => m.status === 'deployed').sort((a, b) => b.accuracy - a.accuracy).slice(0, 4) as model}
                  <div class="model-item p-2 bg-dark-bg-tertiary rounded border border-dark-border">
                    <div class="flex items-center justify-between">
                      <div>
                        <div class="text-sm text-dark-text-primary">{model.name}</div>
                        <div class="text-xs text-dark-text-secondary">{model.version} • {model.framework}</div>
                      </div>
                      <div class="text-right">
                        <div class="text-sm font-bold text-green-400">{model.accuracy.toFixed(1)}%</div>
                        <div class="text-xs text-dark-text-tertiary">{(model.size / 1024).toFixed(1)}GB</div>
                      </div>
                    </div>
                  </div>
                {/each}
              </div>
            </div>
          </Card>
        </div>
      </div>

      <!-- Quick Actions -->
      <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
        <Button variant="outline" fullWidth on:click={() => currentView = 'models'}>
          <span class="text-lg mr-2">🤖</span>
          Manage Models
        </Button>
        <Button variant="outline" fullWidth on:click={() => currentView = 'training'}>
          <span class="text-lg mr-2">🎯</span>
          Monitor Training
        </Button>
        <Button variant="outline" fullWidth on:click={() => currentView = 'dags'}>
          <span class="text-lg mr-2">🔄</span>
          Pipeline DAGs
        </Button>
        <Button variant="outline" fullWidth on:click={() => currentView = 'optimizer'}>
          <span class="text-lg mr-2">🧬</span>
          Genetic Optimizer
        </Button>
      </div>
    {/if}

    {#if currentView === 'models'}
      <ModelManager 
        models={$models}
        on:modelAction={(e) => {
          // Handle model actions (deploy, archive, retrain)
          console.log('Model action:', e.detail);
        }}
        on:backToOverview={() => currentView = 'overview'}
      />
    {/if}

    {#if currentView === 'training'}
      <TrainingMonitor 
        trainingJobs={$trainingJobs}
        models={$models}
        on:startTraining={(e) => {
          // Handle training start
          console.log('Start training:', e.detail);
        }}
        on:backToOverview={() => currentView = 'overview'}
      />
    {/if}

    {#if currentView === 'dags'}
      <PipelineDAGs 
        on:backToOverview={() => currentView = 'overview'}
      />
    {/if}

    {#if currentView === 'optimizer'}
      <GeneticOptimizer 
        on:backToOverview={() => currentView = 'overview'}
      />
    {/if}
  </div>
</div>