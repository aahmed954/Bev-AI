<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // Testing Infrastructure state
  const testingState = writable({
    test_orchestration: {
      active_suites: 12,
      total_tests: 3847,
      passed_tests: 3789,
      failed_tests: 47,
      skipped_tests: 11,
      success_rate: 98.5,
      avg_execution_time: '12m 34s',
      parallel_workers: 8
    },
    chaos_engineering: {
      active_experiments: 5,
      success_rate: 94.7,
      failure_injection_rate: 12.3,
      recovery_time_avg: '2m 18s',
      blast_radius_control: 89.4,
      steady_state_hypothesis: 91.2
    },
    performance_testing: {
      load_tests_running: 3,
      concurrent_users: 2500,
      requests_per_second: 8947,
      response_time_p95: 127,
      error_rate: 0.3,
      throughput_mbps: 234.7
    },
    security_validation: {
      vulnerability_scans: 23,
      critical_vulnerabilities: 0,
      high_vulnerabilities: 2,
      medium_vulnerabilities: 8,
      compliance_score: 97.3,
      penetration_tests: 5
    },
    quality_assurance: {
      code_coverage: 87.4,
      technical_debt_ratio: 12.7,
      maintainability_index: 78.9,
      cyclomatic_complexity: 2.8,
      duplication_percentage: 3.2,
      quality_gate_status: 'passed'
    }
  });

  // Testing controls and configuration
  let activeTestingWorkspace = 'orchestration';
  let selectedTestSuite = '';
  let selectedEnvironment = 'staging';

  // Test orchestration configuration
  let orchestrationConfig = {
    parallel_execution: true,
    worker_count: 8,
    timeout_minutes: 30,
    retry_failed: true,
    generate_reports: true,
    slack_notifications: true
  };

  // Chaos engineering configuration
  let chaosConfig = {
    experiment_type: 'network_partition',
    target_service: '',
    duration_minutes: 5,
    blast_radius_percentage: 10,
    steady_state_tolerance: 5,
    rollback_condition: 'auto'
  };

  // Performance testing configuration
  let performanceConfig = {
    test_type: 'load',
    concurrent_users: 1000,
    ramp_up_time: 300,
    duration_minutes: 30,
    target_rps: 5000,
    success_criteria: '< 200ms p95'
  };

  // Security testing configuration
  let securityConfig = {
    scan_type: 'full',
    include_authentication: true,
    test_authorization: true,
    check_input_validation: true,
    test_session_management: true,
    scan_for_owasp_top10: true
  };

  // Test data and results
  const testSuites = writable([]);
  const chaosExperiments = writable([]);
  const performanceResults = writable([]);
  const securityScans = writable([]);

  // WebSocket connections
  let testingWs: WebSocket | null = null;
  let chaosWs: WebSocket | null = null;
  let performanceWs: WebSocket | null = null;
  let securityWs: WebSocket | null = null;

  onMount(() => {
    initializeWebSockets();
    loadTestData();
    startMetricsCollection();
  });

  onDestroy(() => {
    if (testingWs) testingWs.close();
    if (chaosWs) chaosWs.close();
    if (performanceWs) performanceWs.close();
    if (securityWs) securityWs.close();
  });

  function initializeWebSockets() {
    // Test orchestration WebSocket
    testingWs = new WebSocket('ws://localhost:8050/testing');
    testingWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      testingState.update(state => ({
        ...state,
        test_orchestration: { ...state.test_orchestration, ...data }
      }));
    };

    // Chaos engineering WebSocket
    chaosWs = new WebSocket('ws://localhost:8051/chaos');
    chaosWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      testingState.update(state => ({
        ...state,
        chaos_engineering: { ...state.chaos_engineering, ...data }
      }));
      if (data.experiments) {
        chaosExperiments.set(data.experiments);
      }
    };

    // Performance testing WebSocket
    performanceWs = new WebSocket('ws://localhost:8052/performance');
    performanceWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      testingState.update(state => ({
        ...state,
        performance_testing: { ...state.performance_testing, ...data }
      }));
    };

    // Security validation WebSocket
    securityWs = new WebSocket('ws://localhost:8053/security');
    securityWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      testingState.update(state => ({
        ...state,
        security_validation: { ...state.security_validation, ...data }
      }));
    };
  }

  async function loadTestData() {
    try {
      const [suitesRes, experimentsRes, resultsRes, scansRes] = await Promise.all([
        fetch('http://localhost:8050/api/suites'),
        fetch('http://localhost:8051/api/experiments'),
        fetch('http://localhost:8052/api/results'),
        fetch('http://localhost:8053/api/scans')
      ]);

      const suites = await suitesRes.json();
      const experiments = await experimentsRes.json();
      const results = await resultsRes.json();
      const scans = await scansRes.json();

      testSuites.set(suites);
      chaosExperiments.set(experiments);
      performanceResults.set(results);
      securityScans.set(scans);
    } catch (error) {
      console.error('Failed to load test data:', error);
    }
  }

  async function startMetricsCollection() {
    setInterval(async () => {
      try {
        const response = await fetch('http://localhost:8050/api/metrics');
        const metrics = await response.json();
        testingState.update(state => ({
          ...state,
          ...metrics
        }));
      } catch (error) {
        console.error('Testing metrics collection error:', error);
      }
    }, 5000);
  }

  async function executeTestSuite() {
    if (!selectedTestSuite) return;

    try {
      const response = await fetch('http://localhost:8050/api/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          suite: selectedTestSuite,
          environment: selectedEnvironment,
          config: orchestrationConfig
        })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('Test suite execution started:', result);
      }
    } catch (error) {
      console.error('Test execution failed:', error);
    }
  }

  async function startChaosExperiment() {
    try {
      const response = await fetch('http://localhost:8051/api/experiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(chaosConfig)
      });

      if (response.ok) {
        const experiment = await response.json();
        console.log('Chaos experiment started:', experiment);
      }
    } catch (error) {
      console.error('Chaos experiment failed:', error);
    }
  }

  async function runPerformanceTest() {
    try {
      const response = await fetch('http://localhost:8052/api/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(performanceConfig)
      });

      if (response.ok) {
        const test = await response.json();
        console.log('Performance test started:', test);
      }
    } catch (error) {
      console.error('Performance test failed:', error);
    }
  }

  async function startSecurityScan() {
    try {
      const response = await fetch('http://localhost:8053/api/scan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(securityConfig)
      });

      if (response.ok) {
        const scan = await response.json();
        console.log('Security scan started:', scan);
      }
    } catch (error) {
      console.error('Security scan failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'passed': case 'success': case 'healthy': return 'text-green-400';
      case 'running': case 'executing': case 'testing': return 'text-blue-400';
      case 'failed': case 'error': case 'critical': return 'text-red-400';
      case 'warning': case 'medium': return 'text-yellow-400';
      case 'skipped': case 'pending': return 'text-gray-400';
      default: return 'text-gray-400';
    }
  }

  function getHealthColor(value: number, threshold: number = 95): string {
    if (value >= threshold) return 'text-green-400';
    if (value >= threshold - 10) return 'text-yellow-400';
    return 'text-red-400';
  }
</script>

<svelte:head>
  <title>Testing Infrastructure Platform | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-teal-400 to-blue-500 bg-clip-text text-transparent">
        Testing Infrastructure Platform
      </h1>
      <p class="text-gray-300">Comprehensive quality assurance and validation automation</p>
    </div>

    {#if $testingState}
      <!-- Testing Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-5 gap-6 mb-8">
        <!-- Test Orchestration -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-teal-400 mr-2"></span>
            Test Orchestration
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Active Suites:</span>
              <span class="text-teal-400">{$testingState.test_orchestration.active_suites}</span>
            </div>
            <div class="flex justify-between">
              <span>Total Tests:</span>
              <span class="text-blue-400">{$testingState.test_orchestration.total_tests.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class={getHealthColor($testingState.test_orchestration.success_rate)}>{$testingState.test_orchestration.success_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Avg Time:</span>
              <span class="text-cyan-400">{$testingState.test_orchestration.avg_execution_time}</span>
            </div>
          </div>
        </div>

        <!-- Chaos Engineering -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-red-400 mr-2"></span>
            Chaos Engineering
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Experiments:</span>
              <span class="text-red-400">{$testingState.chaos_engineering.active_experiments}</span>
            </div>
            <div class="flex justify-between">
              <span>Success Rate:</span>
              <span class={getHealthColor($testingState.chaos_engineering.success_rate)}>{$testingState.chaos_engineering.success_rate}%</span>
            </div>
            <div class="flex justify-between">
              <span>Recovery Time:</span>
              <span class="text-yellow-400">{$testingState.chaos_engineering.recovery_time_avg}</span>
            </div>
            <div class="flex justify-between">
              <span>Blast Radius:</span>
              <span class="text-purple-400">{$testingState.chaos_engineering.blast_radius_control}%</span>
            </div>
          </div>
        </div>

        <!-- Performance Testing -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-blue-400 mr-2"></span>
            Performance Testing
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Load Tests:</span>
              <span class="text-blue-400">{$testingState.performance_testing.load_tests_running}</span>
            </div>
            <div class="flex justify-between">
              <span>Users:</span>
              <span class="text-green-400">{$testingState.performance_testing.concurrent_users.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>RPS:</span>
              <span class="text-yellow-400">{$testingState.performance_testing.requests_per_second.toLocaleString()}</span>
            </div>
            <div class="flex justify-between">
              <span>P95 Response:</span>
              <span class="text-cyan-400">{$testingState.performance_testing.response_time_p95}ms</span>
            </div>
          </div>
        </div>

        <!-- Security Validation -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            Security Validation
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Vuln Scans:</span>
              <span class="text-purple-400">{$testingState.security_validation.vulnerability_scans}</span>
            </div>
            <div class="flex justify-between">
              <span>Critical:</span>
              <span class="text-red-400">{$testingState.security_validation.critical_vulnerabilities}</span>
            </div>
            <div class="flex justify-between">
              <span>High:</span>
              <span class="text-red-400">{$testingState.security_validation.high_vulnerabilities}</span>
            </div>
            <div class="flex justify-between">
              <span>Compliance:</span>
              <span class={getHealthColor($testingState.security_validation.compliance_score)}>{$testingState.security_validation.compliance_score}%</span>
            </div>
          </div>
        </div>

        <!-- Quality Assurance -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            Quality Assurance
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Coverage:</span>
              <span class={getHealthColor($testingState.quality_assurance.code_coverage, 80)}>{$testingState.quality_assurance.code_coverage}%</span>
            </div>
            <div class="flex justify-between">
              <span>Tech Debt:</span>
              <span class="text-yellow-400">{$testingState.quality_assurance.technical_debt_ratio}%</span>
            </div>
            <div class="flex justify-between">
              <span>Maintainability:</span>
              <span class={getHealthColor($testingState.quality_assurance.maintainability_index, 70)}>{$testingState.quality_assurance.maintainability_index}</span>
            </div>
            <div class="flex justify-between">
              <span>Quality Gate:</span>
              <span class={getStatusColor($testingState.quality_assurance.quality_gate_status)}>
                {$testingState.quality_assurance.quality_gate_status.toUpperCase()}
              </span>
            </div>
          </div>
        </div>
      </div>

      <!-- Testing Workspace Tabs -->
      <div class="mb-6">
        <div class="flex space-x-1 bg-gray-800 p-1 rounded-lg">
          {#each ['orchestration', 'chaos', 'performance', 'security', 'quality'] as workspace}
            <button
              on:click={() => activeTestingWorkspace = workspace}
              class="flex-1 py-2 px-4 rounded text-center transition-colors text-sm
                {activeTestingWorkspace === workspace
                  ? 'bg-teal-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'}"
            >
              {workspace.charAt(0).toUpperCase() + workspace.slice(1)}
            </button>
          {/each}
        </div>
      </div>

      <!-- Workspace Content -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {#if activeTestingWorkspace === 'orchestration'}
          <!-- Test Orchestration -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Test Suite Orchestration</h3>
            <div class="space-y-4">
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Test Suite</label>
                  <select bind:value={selectedTestSuite} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="">Select Test Suite</option>
                    <option value="integration">Integration Tests</option>
                    <option value="security">Security Tests</option>
                    <option value="performance">Performance Tests</option>
                    <option value="end_to_end">End-to-End Tests</option>
                    <option value="regression">Regression Tests</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Environment</label>
                  <select bind:value={selectedEnvironment} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="development">Development</option>
                    <option value="staging">Staging</option>
                    <option value="production">Production</option>
                  </select>
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Parallel Workers: {orchestrationConfig.worker_count}</label>
                <input
                  type="range"
                  bind:value={orchestrationConfig.worker_count}
                  min="1"
                  max="16"
                  class="w-full"
                >
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Timeout (minutes): {orchestrationConfig.timeout_minutes}</label>
                <input
                  type="range"
                  bind:value={orchestrationConfig.timeout_minutes}
                  min="5"
                  max="120"
                  class="w-full"
                >
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={orchestrationConfig.parallel_execution} class="mr-2">
                  <span class="text-sm">Parallel Execution</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={orchestrationConfig.retry_failed} class="mr-2">
                  <span class="text-sm">Retry Failed Tests</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={orchestrationConfig.generate_reports} class="mr-2">
                  <span class="text-sm">Generate Reports</span>
                </label>
              </div>
              <button
                on:click={executeTestSuite}
                disabled={!selectedTestSuite}
                class="w-full bg-teal-600 hover:bg-teal-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                Execute Test Suite
              </button>
            </div>
          </div>

          <!-- Test Results -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Test Results</h3>
            <div class="space-y-4">
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span>Passed Tests</span>
                  <span>{$testingState.test_orchestration.passed_tests}</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2">
                  <div class="bg-green-400 h-2 rounded-full" style="width: {($testingState.test_orchestration.passed_tests / $testingState.test_orchestration.total_tests) * 100}%"></div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span>Failed Tests</span>
                  <span>{$testingState.test_orchestration.failed_tests}</span>
                </div>
                <div class="w-full bg-gray-700 rounded-full h-2">
                  <div class="bg-red-400 h-2 rounded-full" style="width: {($testingState.test_orchestration.failed_tests / $testingState.test_orchestration.total_tests) * 100}%"></div>
                </div>
              </div>
              <div class="text-sm space-y-1">
                <div class="flex justify-between">
                  <span>Skipped:</span>
                  <span class="text-gray-400">{$testingState.test_orchestration.skipped_tests}</span>
                </div>
                <div class="flex justify-between">
                  <span>Workers:</span>
                  <span class="text-blue-400">{$testingState.test_orchestration.parallel_workers}</span>
                </div>
              </div>
            </div>
          </div>

        {:else if activeTestingWorkspace === 'chaos'}
          <!-- Chaos Engineering -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Chaos Engineering Console</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Experiment Type</label>
                <select bind:value={chaosConfig.experiment_type} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="network_partition">Network Partition</option>
                  <option value="cpu_stress">CPU Stress</option>
                  <option value="memory_exhaustion">Memory Exhaustion</option>
                  <option value="disk_pressure">Disk Pressure</option>
                  <option value="latency_injection">Latency Injection</option>
                  <option value="service_failure">Service Failure</option>
                </select>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Target Service</label>
                <input
                  bind:value={chaosConfig.target_service}
                  placeholder="Enter service name..."
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Duration (min)</label>
                  <input
                    type="number"
                    bind:value={chaosConfig.duration_minutes}
                    min="1"
                    max="60"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Blast Radius (%)</label>
                  <input
                    type="number"
                    bind:value={chaosConfig.blast_radius_percentage}
                    min="1"
                    max="50"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Rollback Condition</label>
                <select bind:value={chaosConfig.rollback_condition} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="auto">Automatic</option>
                  <option value="manual">Manual</option>
                  <option value="threshold">Threshold-based</option>
                </select>
              </div>
              <button
                on:click={startChaosExperiment}
                disabled={!chaosConfig.target_service}
                class="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-600 px-4 py-2 rounded font-semibold"
              >
                START CHAOS EXPERIMENT
              </button>
            </div>
          </div>

          <!-- Active Experiments -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Active Experiments</h3>
            <div class="space-y-3">
              {#if $chaosExperiments && $chaosExperiments.length > 0}
                {#each $chaosExperiments as experiment}
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">{experiment.name}</span>
                      <span class={getStatusColor(experiment.status)} class="text-xs">
                        {experiment.status.toUpperCase()}
                      </span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Target: {experiment.target}</div>
                      <div>Duration: {experiment.duration}</div>
                      <div>Progress: {experiment.progress}%</div>
                    </div>
                  </div>
                {/each}
              {:else}
                <div class="text-center text-gray-400 py-4">
                  No active chaos experiments
                </div>
              {/if}
            </div>
          </div>

        {:else if activeTestingWorkspace === 'performance'}
          <!-- Performance Testing -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Performance Testing Configuration</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Test Type</label>
                <select bind:value={performanceConfig.test_type} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="load">Load Testing</option>
                  <option value="stress">Stress Testing</option>
                  <option value="spike">Spike Testing</option>
                  <option value="volume">Volume Testing</option>
                  <option value="endurance">Endurance Testing</option>
                </select>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Concurrent Users</label>
                  <input
                    type="number"
                    bind:value={performanceConfig.concurrent_users}
                    min="1"
                    max="10000"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Target RPS</label>
                  <input
                    type="number"
                    bind:value={performanceConfig.target_rps}
                    min="100"
                    max="50000"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Ramp-up Time (s)</label>
                  <input
                    type="number"
                    bind:value={performanceConfig.ramp_up_time}
                    min="10"
                    max="3600"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Duration (min)</label>
                  <input
                    type="number"
                    bind:value={performanceConfig.duration_minutes}
                    min="1"
                    max="180"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Success Criteria</label>
                <input
                  bind:value={performanceConfig.success_criteria}
                  placeholder="< 200ms p95, error rate < 1%"
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
              <button
                on:click={runPerformanceTest}
                class="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
              >
                Run Performance Test
              </button>
            </div>
          </div>

          <!-- Performance Metrics -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Real-time Metrics</h3>
            <div class="space-y-4">
              <div class="bg-gray-700 rounded p-3">
                <h4 class="font-medium mb-2">Current Load Test</h4>
                <div class="text-sm text-gray-300 space-y-1">
                  <div>Users: {$testingState.performance_testing.concurrent_users.toLocaleString()}</div>
                  <div>RPS: {$testingState.performance_testing.requests_per_second.toLocaleString()}</div>
                  <div>Error Rate: {$testingState.performance_testing.error_rate}%</div>
                </div>
              </div>
              <div>
                <div class="flex justify-between text-sm mb-1">
                  <span>Throughput</span>
                  <span>{$testingState.performance_testing.throughput_mbps} Mbps</span>
                </div>
                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-blue-400 h-2 rounded-full" style="width: {Math.min(($testingState.performance_testing.throughput_mbps / 500) * 100, 100)}%"></div>
                </div>
              </div>
            </div>
          </div>

        {:else if activeTestingWorkspace === 'security'}
          <!-- Security Validation -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Security Validation & Scanning</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Scan Type</label>
                <select bind:value={securityConfig.scan_type} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="quick">Quick Scan</option>
                  <option value="full">Full Security Audit</option>
                  <option value="compliance">Compliance Check</option>
                  <option value="penetration">Penetration Test</option>
                </select>
              </div>
              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Security Test Categories</h4>
                <div class="space-y-2">
                  <label class="flex items-center">
                    <input type="checkbox" bind:checked={securityConfig.include_authentication} class="mr-2">
                    <span class="text-sm">Authentication Testing</span>
                  </label>
                  <label class="flex items-center">
                    <input type="checkbox" bind:checked={securityConfig.test_authorization} class="mr-2">
                    <span class="text-sm">Authorization Testing</span>
                  </label>
                  <label class="flex items-center">
                    <input type="checkbox" bind:checked={securityConfig.check_input_validation} class="mr-2">
                    <span class="text-sm">Input Validation</span>
                  </label>
                  <label class="flex items-center">
                    <input type="checkbox" bind:checked={securityConfig.test_session_management} class="mr-2">
                    <span class="text-sm">Session Management</span>
                  </label>
                  <label class="flex items-center">
                    <input type="checkbox" bind:checked={securityConfig.scan_for_owasp_top10} class="mr-2">
                    <span class="text-sm">OWASP Top 10</span>
                  </label>
                </div>
              </div>
              <button
                on:click={startSecurityScan}
                class="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded"
              >
                Start Security Scan
              </button>
            </div>
          </div>

          <!-- Vulnerability Dashboard -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Vulnerability Status</h3>
            <div class="space-y-3">
              <div class="bg-red-900 border border-red-600 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Critical</span>
                  <span class="text-red-400">{$testingState.security_validation.critical_vulnerabilities}</span>
                </div>
                <p class="text-sm text-red-300">Immediate attention required</p>
              </div>
              <div class="bg-orange-900 border border-orange-600 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">High</span>
                  <span class="text-orange-400">{$testingState.security_validation.high_vulnerabilities}</span>
                </div>
                <p class="text-sm text-orange-300">Should be addressed soon</p>
              </div>
              <div class="bg-yellow-900 border border-yellow-600 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">Medium</span>
                  <span class="text-yellow-400">{$testingState.security_validation.medium_vulnerabilities}</span>
                </div>
                <p class="text-sm text-yellow-300">Monitor and plan fixes</p>
              </div>
            </div>
          </div>

        {:else if activeTestingWorkspace === 'quality'}
          <!-- Quality Assurance -->
          <div class="lg:col-span-3 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Quality Assurance Dashboard</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <!-- Code Coverage -->
              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Code Coverage</h4>
                <div class="text-3xl font-bold text-green-400 mb-2">
                  {$testingState.quality_assurance.code_coverage}%
                </div>
                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-green-400 h-2 rounded-full" style="width: {$testingState.quality_assurance.code_coverage}%"></div>
                </div>
              </div>

              <!-- Technical Debt -->
              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Technical Debt</h4>
                <div class="text-3xl font-bold text-yellow-400 mb-2">
                  {$testingState.quality_assurance.technical_debt_ratio}%
                </div>
                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-yellow-400 h-2 rounded-full" style="width: {$testingState.quality_assurance.technical_debt_ratio}%"></div>
                </div>
              </div>

              <!-- Maintainability -->
              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Maintainability</h4>
                <div class="text-3xl font-bold text-blue-400 mb-2">
                  {$testingState.quality_assurance.maintainability_index}
                </div>
                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-blue-400 h-2 rounded-full" style="width: {$testingState.quality_assurance.maintainability_index}%"></div>
                </div>
              </div>

              <!-- Duplication -->
              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Code Duplication</h4>
                <div class="text-3xl font-bold text-purple-400 mb-2">
                  {$testingState.quality_assurance.duplication_percentage}%
                </div>
                <div class="w-full bg-gray-600 rounded-full h-2">
                  <div class="bg-purple-400 h-2 rounded-full" style="width: {$testingState.quality_assurance.duplication_percentage}%"></div>
                </div>
              </div>
            </div>

            <!-- Quality Gate Status -->
            <div class="mt-6 p-4 rounded {$testingState.quality_assurance.quality_gate_status === 'passed' ? 'bg-green-900 border border-green-600' : 'bg-red-900 border border-red-600'}">
              <h4 class="font-medium mb-2">Quality Gate Status</h4>
              <div class="flex items-center">
                <span class="text-2xl mr-2">
                  {$testingState.quality_assurance.quality_gate_status === 'passed' ? '✅' : '❌'}
                </span>
                <span class="font-semibold {$testingState.quality_assurance.quality_gate_status === 'passed' ? 'text-green-400' : 'text-red-400'}">
                  {$testingState.quality_assurance.quality_gate_status.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>