<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { invoke } from '@tauri-apps/api/core';

  // Configuration Management state
  const configState = writable({
    vault: {
      status: 'sealed',
      policies: 6,
      active_tokens: 23,
      secret_engines: 8,
      audit_enabled: true,
      auto_unseal: false,
      seal_status: 'sealed',
      version: '1.15.2'
    },
    ssl: {
      total_certificates: 47,
      active_certificates: 43,
      expiring_soon: 2,
      auto_renewal_enabled: true,
      ca_trust_score: 98.7,
      wildcard_certificates: 12,
      lets_encrypt_enabled: true
    },
    secrets: {
      total_secrets: 234,
      rotated_last_30d: 67,
      rotation_compliance: 94.3,
      expiring_secrets: 8,
      automatic_rotation: true,
      encryption_strength: 'AES-256'
    },
    configuration: {
      environments: ['development', 'staging', 'production'],
      config_drift_detected: 2,
      last_sync: '15m ago',
      compliance_score: 96.8,
      gitops_enabled: true,
      approval_required: true
    },
    monitoring: {
      config_changes_24h: 12,
      failed_deployments: 1,
      rollback_events: 0,
      security_violations: 0,
      audit_log_entries: 1247
    }
  });

  // Management workspaces
  let activeWorkspace = 'vault';
  let selectedEnvironment = 'production';
  let selectedPolicy = '';

  // Vault configuration
  let vaultConfig = {
    auto_unseal: false,
    seal_threshold: 3,
    seal_shares: 5,
    audit_enabled: true,
    token_ttl: 3600,
    renewal_enabled: true
  };

  // SSL certificate configuration
  let sslConfig = {
    domain: '',
    san_domains: [],
    key_size: 2048,
    auto_renewal: true,
    notification_days: 30,
    ca_provider: 'lets_encrypt'
  };

  // Secret rotation configuration
  let rotationConfig = {
    secret_type: 'database',
    rotation_interval: 30,
    auto_rotation: true,
    notification_enabled: true,
    backup_previous: true
  };

  // Available vault policies
  const vaultPolicies = [
    'admin-policy',
    'security-team-policy',
    'application-policy',
    'cicd-policy',
    'oracle-worker-policy',
    'developer-policy'
  ];

  // WebSocket connections
  let vaultWs: WebSocket | null = null;
  let sslWs: WebSocket | null = null;
  let configWs: WebSocket | null = null;
  let auditWs: WebSocket | null = null;

  // Live data stores
  const certificates = writable([]);
  const secrets = writable([]);
  const auditLogs = writable([]);

  onMount(() => {
    initializeWebSockets();
    loadVaultData();
    loadCertificates();
    loadSecrets();
    startAuditMonitoring();
  });

  onDestroy(() => {
    if (vaultWs) vaultWs.close();
    if (sslWs) sslWs.close();
    if (configWs) configWs.close();
    if (auditWs) auditWs.close();
  });

  function initializeWebSockets() {
    // Vault management WebSocket
    vaultWs = new WebSocket('ws://localhost:8040/vault');
    vaultWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      configState.update(state => ({
        ...state,
        vault: { ...state.vault, ...data.vault }
      }));
    };

    // SSL certificate WebSocket
    sslWs = new WebSocket('ws://localhost:8041/ssl');
    sslWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      configState.update(state => ({
        ...state,
        ssl: { ...state.ssl, ...data }
      }));
      if (data.certificates) {
        certificates.set(data.certificates);
      }
    };

    // Configuration sync WebSocket
    configWs = new WebSocket('ws://localhost:8042/config');
    configWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      configState.update(state => ({
        ...state,
        configuration: { ...state.configuration, ...data }
      }));
    };

    // Audit logging WebSocket
    auditWs = new WebSocket('ws://localhost:8043/audit');
    auditWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      auditLogs.update(logs => [data, ...logs.slice(0, 99)]);
    };
  }

  async function loadVaultData() {
    try {
      const response = await fetch('http://localhost:8040/api/status');
      const vaultData = await response.json();
      configState.update(state => ({
        ...state,
        vault: { ...state.vault, ...vaultData }
      }));
    } catch (error) {
      console.error('Failed to load Vault data:', error);
    }
  }

  async function loadCertificates() {
    try {
      const response = await fetch('http://localhost:8041/api/certificates');
      const certs = await response.json();
      certificates.set(certs);
    } catch (error) {
      console.error('Failed to load certificates:', error);
    }
  }

  async function loadSecrets() {
    try {
      const response = await fetch('http://localhost:8040/api/secrets');
      const secretsList = await response.json();
      secrets.set(secretsList);
    } catch (error) {
      console.error('Failed to load secrets:', error);
    }
  }

  async function startAuditMonitoring() {
    try {
      const response = await fetch('http://localhost:8043/api/audit/recent');
      const logs = await response.json();
      auditLogs.set(logs);
    } catch (error) {
      console.error('Failed to load audit logs:', error);
    }
  }

  async function unsealVault() {
    try {
      const response = await fetch('http://localhost:8040/api/unseal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(vaultConfig)
      });

      if (response.ok) {
        console.log('Vault unsealing initiated');
      }
    } catch (error) {
      console.error('Vault unseal failed:', error);
    }
  }

  async function generateCertificate() {
    if (!sslConfig.domain) return;

    try {
      const response = await fetch('http://localhost:8041/api/certificates', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(sslConfig)
      });

      if (response.ok) {
        const certificate = await response.json();
        certificates.update(certs => [certificate, ...certs]);

        // Reset form
        sslConfig = {
          domain: '',
          san_domains: [],
          key_size: 2048,
          auto_renewal: true,
          notification_days: 30,
          ca_provider: 'lets_encrypt'
        };
      }
    } catch (error) {
      console.error('Certificate generation failed:', error);
    }
  }

  async function rotateSecret(secretPath: string) {
    try {
      const response = await fetch(`http://localhost:8040/api/secrets/${secretPath}/rotate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(rotationConfig)
      });

      if (response.ok) {
        console.log('Secret rotation initiated');
      }
    } catch (error) {
      console.error('Secret rotation failed:', error);
    }
  }

  async function syncConfiguration() {
    try {
      const response = await fetch('http://localhost:8042/api/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          environment: selectedEnvironment,
          force_sync: false,
          validate_before: true
        })
      });

      if (response.ok) {
        console.log('Configuration sync initiated');
      }
    } catch (error) {
      console.error('Configuration sync failed:', error);
    }
  }

  function getStatusColor(status: string): string {
    switch (status) {
      case 'sealed': return 'text-red-400';
      case 'unsealed': case 'active': case 'valid': return 'text-green-400';
      case 'expiring': case 'warning': return 'text-yellow-400';
      case 'expired': case 'invalid': case 'failed': return 'text-red-400';
      case 'pending': case 'rotating': return 'text-blue-400';
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
  <title>Configuration Management Center | BEV OSINT</title>
</svelte:head>

<div class="min-h-screen bg-gray-900 text-white p-6">
  <div class="max-w-7xl mx-auto">
    <!-- Header -->
    <div class="mb-8">
      <h1 class="text-4xl font-bold mb-2 bg-gradient-to-r from-violet-400 to-purple-500 bg-clip-text text-transparent">
        Configuration Management Center
      </h1>
      <p class="text-gray-300">Enterprise configuration, secrets, and certificate management</p>
    </div>

    {#if $configState}
      <!-- Configuration Status Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-5 gap-6 mb-8">
        <!-- Vault Status -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-purple-400 mr-2"></span>
            HashiCorp Vault
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Status:</span>
              <span class={getStatusColor($configState.vault.status)}>
                {$configState.vault.status.toUpperCase()}
              </span>
            </div>
            <div class="flex justify-between">
              <span>Policies:</span>
              <span class="text-purple-400">{$configState.vault.policies}</span>
            </div>
            <div class="flex justify-between">
              <span>Tokens:</span>
              <span class="text-blue-400">{$configState.vault.active_tokens}</span>
            </div>
            <div class="flex justify-between">
              <span>Engines:</span>
              <span class="text-green-400">{$configState.vault.secret_engines}</span>
            </div>
          </div>
        </div>

        <!-- SSL Certificates -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-green-400 mr-2"></span>
            SSL Certificates
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Total:</span>
              <span class="text-green-400">{$configState.ssl.total_certificates}</span>
            </div>
            <div class="flex justify-between">
              <span>Active:</span>
              <span class="text-blue-400">{$configState.ssl.active_certificates}</span>
            </div>
            <div class="flex justify-between">
              <span>Expiring:</span>
              <span class="text-yellow-400">{$configState.ssl.expiring_soon}</span>
            </div>
            <div class="flex justify-between">
              <span>Trust Score:</span>
              <span class={getHealthColor($configState.ssl.ca_trust_score)}>{$configState.ssl.ca_trust_score}%</span>
            </div>
          </div>
        </div>

        <!-- Secrets Management -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-yellow-400 mr-2"></span>
            Secrets Management
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Total Secrets:</span>
              <span class="text-yellow-400">{$configState.secrets.total_secrets}</span>
            </div>
            <div class="flex justify-between">
              <span>Rotated (30d):</span>
              <span class="text-green-400">{$configState.secrets.rotated_last_30d}</span>
            </div>
            <div class="flex justify-between">
              <span>Compliance:</span>
              <span class={getHealthColor($configState.secrets.rotation_compliance)}>{$configState.secrets.rotation_compliance}%</span>
            </div>
            <div class="flex justify-between">
              <span>Expiring:</span>
              <span class="text-red-400">{$configState.secrets.expiring_secrets}</span>
            </div>
          </div>
        </div>

        <!-- Configuration Sync -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-cyan-400 mr-2"></span>
            Configuration Sync
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Drift Detected:</span>
              <span class="text-red-400">{$configState.configuration.config_drift_detected}</span>
            </div>
            <div class="flex justify-between">
              <span>Last Sync:</span>
              <span class="text-gray-400">{$configState.configuration.last_sync}</span>
            </div>
            <div class="flex justify-between">
              <span>Compliance:</span>
              <span class={getHealthColor($configState.configuration.compliance_score)}>{$configState.configuration.compliance_score}%</span>
            </div>
            <div class="flex justify-between">
              <span>GitOps:</span>
              <span class={$configState.configuration.gitops_enabled ? 'text-green-400' : 'text-gray-400'}>
                {$configState.configuration.gitops_enabled ? 'ENABLED' : 'DISABLED'}
              </span>
            </div>
          </div>
        </div>

        <!-- Monitoring & Audit -->
        <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
          <h3 class="text-lg font-semibold mb-4 flex items-center">
            <span class="w-3 h-3 rounded-full bg-red-400 mr-2"></span>
            Monitoring & Audit
          </h3>
          <div class="space-y-2 text-sm">
            <div class="flex justify-between">
              <span>Changes (24h):</span>
              <span class="text-blue-400">{$configState.monitoring.config_changes_24h}</span>
            </div>
            <div class="flex justify-between">
              <span>Failed Deploys:</span>
              <span class="text-red-400">{$configState.monitoring.failed_deployments}</span>
            </div>
            <div class="flex justify-between">
              <span>Security Violations:</span>
              <span class="text-red-400">{$configState.monitoring.security_violations}</span>
            </div>
            <div class="flex justify-between">
              <span>Audit Entries:</span>
              <span class="text-cyan-400">{$configState.monitoring.audit_log_entries.toLocaleString()}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Management Workspace Tabs -->
      <div class="mb-6">
        <div class="flex space-x-1 bg-gray-800 p-1 rounded-lg">
          {#each ['vault', 'ssl', 'secrets', 'config', 'audit'] as workspace}
            <button
              on:click={() => activeWorkspace = workspace}
              class="flex-1 py-2 px-4 rounded text-center transition-colors text-sm
                {activeWorkspace === workspace
                  ? 'bg-purple-600 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'}"
            >
              {workspace.charAt(0).toUpperCase() + workspace.slice(1)}
            </button>
          {/each}
        </div>
      </div>

      <!-- Workspace Content -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {#if activeWorkspace === 'vault'}
          <!-- Vault Management -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">HashiCorp Vault Management</h3>
            <div class="space-y-4">
              <div class="bg-red-900 border border-red-600 rounded p-4 mb-4">
                <div class="flex items-center">
                  <svg class="w-5 h-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd" />
                  </svg>
                  <span class="font-medium">Vault is currently SEALED</span>
                </div>
                <p class="text-sm text-red-300 mt-1">Unseal vault to access secrets and policies</p>
              </div>

              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Seal Threshold</label>
                  <input
                    type="number"
                    bind:value={vaultConfig.seal_threshold}
                    min="1"
                    max="10"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">Total Shares</label>
                  <input
                    type="number"
                    bind:value={vaultConfig.seal_shares}
                    min="1"
                    max="20"
                    class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                  >
                </div>
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={vaultConfig.auto_unseal} class="mr-2">
                  <span class="text-sm">Enable Auto-Unseal</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={vaultConfig.audit_enabled} class="mr-2">
                  <span class="text-sm">Enable Audit Logging</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={vaultConfig.renewal_enabled} class="mr-2">
                  <span class="text-sm">Token Auto-Renewal</span>
                </label>
              </div>
              <button
                on:click={unsealVault}
                class="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded font-semibold"
              >
                UNSEAL VAULT
              </button>
            </div>
          </div>

          <!-- Vault Policies -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Vault Policies</h3>
            <div class="space-y-3">
              {#each vaultPolicies as policy}
                <div class="bg-gray-700 rounded p-3">
                  <div class="flex justify-between items-center mb-2">
                    <span class="font-medium">{policy}</span>
                    <span class="text-green-400 text-xs">ACTIVE</span>
                  </div>
                  <div class="text-sm text-gray-300">
                    <div>Permissions: {policy.includes('admin') ? 'Full Access' : 'Limited'}</div>
                    <div>Token TTL: {vaultConfig.token_ttl}s</div>
                  </div>
                </div>
              {/each}
            </div>
          </div>

        {:else if activeWorkspace === 'ssl'}
          <!-- SSL Certificate Management -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">SSL Certificate Management</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Domain Name</label>
                <input
                  bind:value={sslConfig.domain}
                  placeholder="example.com"
                  class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                >
              </div>
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="block text-sm font-medium mb-2">Key Size</label>
                  <select bind:value={sslConfig.key_size} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="2048">2048 bits</option>
                    <option value="3072">3072 bits</option>
                    <option value="4096">4096 bits</option>
                  </select>
                </div>
                <div>
                  <label class="block text-sm font-medium mb-2">CA Provider</label>
                  <select bind:value={sslConfig.ca_provider} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                    <option value="lets_encrypt">Let's Encrypt</option>
                    <option value="digicert">DigiCert</option>
                    <option value="sectigo">Sectigo</option>
                    <option value="internal_ca">Internal CA</option>
                  </select>
                </div>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Notification Days: {sslConfig.notification_days}</label>
                <input
                  type="range"
                  bind:value={sslConfig.notification_days}
                  min="7"
                  max="90"
                  class="w-full"
                >
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={sslConfig.auto_renewal} class="mr-2">
                  <span class="text-sm">Auto-Renewal Enabled</span>
                </label>
              </div>
              <button
                on:click={generateCertificate}
                disabled={!sslConfig.domain}
                class="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-4 py-2 rounded"
              >
                Generate Certificate
              </button>
            </div>
          </div>

          <!-- Active Certificates -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Active Certificates</h3>
            <div class="space-y-3">
              {#if $certificates && $certificates.length > 0}
                {#each $certificates.slice(0, 5) as cert}
                  <div class="bg-gray-700 rounded p-3">
                    <div class="flex justify-between items-center mb-2">
                      <span class="font-medium">{cert.domain}</span>
                      <span class={getStatusColor(cert.status)} class="text-xs">
                        {cert.status.toUpperCase()}
                      </span>
                    </div>
                    <div class="text-sm text-gray-300">
                      <div>Expires: {cert.expires_at}</div>
                      <div>Days Left: {cert.days_remaining}</div>
                    </div>
                  </div>
                {/each}
              {:else}
                <div class="text-center text-gray-400 py-4">
                  No certificates available
                </div>
              {/if}
            </div>
          </div>

        {:else if activeWorkspace === 'secrets'}
          <!-- Secret Rotation Management -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Secret Rotation Management</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Secret Type</label>
                <select bind:value={rotationConfig.secret_type} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  <option value="database">Database Credentials</option>
                  <option value="api_key">API Keys</option>
                  <option value="oauth_token">OAuth Tokens</option>
                  <option value="encryption_key">Encryption Keys</option>
                  <option value="ssh_key">SSH Keys</option>
                </select>
              </div>
              <div>
                <label class="block text-sm font-medium mb-2">Rotation Interval (days): {rotationConfig.rotation_interval}</label>
                <input
                  type="range"
                  bind:value={rotationConfig.rotation_interval}
                  min="1"
                  max="365"
                  class="w-full"
                >
              </div>
              <div class="space-y-2">
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={rotationConfig.auto_rotation} class="mr-2">
                  <span class="text-sm">Automatic Rotation</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={rotationConfig.notification_enabled} class="mr-2">
                  <span class="text-sm">Rotation Notifications</span>
                </label>
                <label class="flex items-center">
                  <input type="checkbox" bind:checked={rotationConfig.backup_previous} class="mr-2">
                  <span class="text-sm">Backup Previous Version</span>
                </label>
              </div>
              <button
                on:click={() => rotateSecret('database/primary')}
                class="w-full bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded"
              >
                Rotate Selected Secrets
              </button>
            </div>
          </div>

          <!-- Secret Status -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Secret Status</h3>
            <div class="space-y-3">
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">DB Primary</span>
                  <span class="text-green-400 text-xs">HEALTHY</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Last Rotated: 15d ago</div>
                  <div>Next Rotation: 15d</div>
                </div>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">API Keys</span>
                  <span class="text-yellow-400 text-xs">EXPIRING</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Last Rotated: 27d ago</div>
                  <div>Next Rotation: 3d</div>
                </div>
              </div>
              <div class="bg-gray-700 rounded p-3">
                <div class="flex justify-between items-center mb-2">
                  <span class="font-medium">OAuth Tokens</span>
                  <span class="text-green-400 text-xs">HEALTHY</span>
                </div>
                <div class="text-sm text-gray-300">
                  <div>Last Rotated: 8d ago</div>
                  <div>Next Rotation: 22d</div>
                </div>
              </div>
            </div>
          </div>

        {:else if activeWorkspace === 'config'}
          <!-- Configuration Synchronization -->
          <div class="lg:col-span-2 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Configuration Synchronization</h3>
            <div class="space-y-4">
              <div>
                <label class="block text-sm font-medium mb-2">Target Environment</label>
                <select bind:value={selectedEnvironment} class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                  {#each $configState.configuration.environments as env}
                    <option value={env}>{env.charAt(0).toUpperCase() + env.slice(1)}</option>
                  {/each}
                </select>
              </div>

              <div class="bg-yellow-900 border border-yellow-600 rounded p-4">
                <h4 class="font-medium mb-2 text-yellow-400">Configuration Drift Detected</h4>
                <div class="text-sm text-gray-300">
                  <div>• Database connection strings differ between staging and production</div>
                  <div>• Feature flags out of sync in development environment</div>
                </div>
              </div>

              <div class="bg-gray-700 rounded p-4">
                <h4 class="font-medium mb-2">Sync Options</h4>
                <div class="space-y-2">
                  <label class="flex items-center">
                    <input type="checkbox" checked class="mr-2">
                    <span class="text-sm">Validate Before Sync</span>
                  </label>
                  <label class="flex items-center">
                    <input type="checkbox" class="mr-2">
                    <span class="text-sm">Force Overwrite</span>
                  </label>
                  <label class="flex items-center">
                    <input type="checkbox" checked class="mr-2">
                    <span class="text-sm">Create Backup</span>
                  </label>
                </div>
              </div>

              <button
                on:click={syncConfiguration}
                class="w-full bg-cyan-600 hover:bg-cyan-700 px-4 py-2 rounded"
              >
                Synchronize Configuration
              </button>
            </div>
          </div>

          <!-- Environment Status -->
          <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Environment Status</h3>
            <div class="space-y-3">
              {#each $configState.configuration.environments as env}
                <div class="bg-gray-700 rounded p-3">
                  <div class="flex justify-between items-center mb-2">
                    <span class="font-medium capitalize">{env}</span>
                    <span class="text-green-400 text-xs">SYNCED</span>
                  </div>
                  <div class="text-sm text-gray-300">
                    <div>Last Update: 2h ago</div>
                    <div>Config Version: v2.1.{env === 'production' ? '0' : '1'}</div>
                  </div>
                </div>
              {/each}
            </div>
          </div>

        {:else if activeWorkspace === 'audit'}
          <!-- Audit Log Viewer -->
          <div class="lg:col-span-3 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h3 class="text-lg font-semibold mb-4">Real-time Audit Logs</h3>
            <div class="bg-gray-900 rounded p-4 h-96 overflow-y-auto font-mono text-sm">
              {#if $auditLogs && $auditLogs.length > 0}
                {#each $auditLogs as log}
                  <div class="mb-2 {log.level === 'ERROR' ? 'text-red-400' : log.level === 'WARN' ? 'text-yellow-400' : 'text-gray-300'}">
                    <span class="text-gray-500">[{log.timestamp}]</span>
                    <span class="text-blue-400">{log.component}</span>
                    <span>{log.message}</span>
                    {#if log.user}
                      <span class="text-purple-400">by {log.user}</span>
                    {/if}
                  </div>
                {/each}
              {:else}
                <div class="text-gray-400 text-center py-8">
                  Audit logs will appear here in real-time
                </div>
              {/if}
            </div>
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>