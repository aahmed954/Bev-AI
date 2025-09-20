/**
 * BEV OSINT Application State Store
 * Global state management using Svelte stores
 */

import { writable, derived, readable } from 'svelte/store';
import type { ProxyStatus, SecurityValidation } from '$lib/ipc/bridge';
import { ipc } from '$lib/ipc/bridge';

// Application state
export const appState = writable({
  initialized: false,
  loading: false,
  error: null as string | null,
});

// User preferences
export const userPreferences = writable({
  theme: 'dark' as 'dark' | 'light',
  sidebarCollapsed: false,
  autoRefresh: true,
  refreshInterval: 30000, // 30 seconds
  notificationsEnabled: true,
});

// OPSEC & Security State
export const opsecState = writable<{
  proxyStatus: ProxyStatus | null;
  securityValidation: SecurityValidation | null;
  lastChecked: Date | null;
}>({
  proxyStatus: null,
  securityValidation: null,
  lastChecked: null,
});

// Dashboard metrics
export const dashboardMetrics = writable({
  darknetMarkets: {
    online: 0,
    offline: 0,
    suspicious: 0,
  },
  cryptoTransactions: {
    tracked: 0,
    flagged: 0,
    total: 0,
  },
  threats: {
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
  },
  agents: {
    active: 0,
    idle: 0,
    error: 0,
  },
});

// Active operations
export const activeOperations = writable<Array<{
  id: string;
  type: string;
  status: 'running' | 'completed' | 'error';
  progress: number;
  startTime: Date;
  endTime?: Date;
  result?: unknown;
  error?: string;
}>>([]);

// Derived stores
export const isProxied = derived(
  opsecState,
  ($opsec) => $opsec.proxyStatus?.connected ?? false
);

export const totalThreats = derived(
  dashboardMetrics,
  ($metrics) => 
    $metrics.threats.critical + 
    $metrics.threats.high + 
    $metrics.threats.medium + 
    $metrics.threats.low
);

export const systemHealth = derived(
  [opsecState, dashboardMetrics],
  ([$opsec, $metrics]) => {
    if (!$opsec.proxyStatus?.connected) return 'critical';
    if ($metrics.threats.critical > 0) return 'warning';
    if ($metrics.agents.error > 0) return 'degraded';
    return 'healthy';
  }
);

// Store actions
export const opsecActions = {
  async checkProxyStatus() {
    try {
      appState.update(s => ({ ...s, loading: true, error: null }));
      
      const [proxyStatus, securityValidation] = await Promise.all([
        ipc.getProxyStatus(),
        ipc.verifyProxyEnforcement(),
      ]);
      
      opsecState.update(s => ({
        ...s,
        proxyStatus,
        securityValidation,
        lastChecked: new Date(),
      }));
    } catch (error) {
      appState.update(s => ({ 
        ...s, 
        error: error instanceof Error ? error.message : 'Failed to check proxy status'
      }));
    } finally {
      appState.update(s => ({ ...s, loading: false }));
    }
  },
  
  async newCircuit() {
    try {
      appState.update(s => ({ ...s, loading: true }));
      await ipc.newTorCircuit();
      await opsecActions.checkProxyStatus();
    } catch (error) {
      appState.update(s => ({ 
        ...s, 
        error: error instanceof Error ? error.message : 'Failed to create new circuit'
      }));
    } finally {
      appState.update(s => ({ ...s, loading: false }));
    }
  },
};

// Auto-refresh functionality
let refreshInterval: ReturnType<typeof setInterval> | null = null;

export function startAutoRefresh() {
  const { autoRefresh, refreshInterval: interval } = userPreferences;
  
  userPreferences.subscribe(prefs => {
    if (refreshInterval) {
      clearInterval(refreshInterval);
      refreshInterval = null;
    }
    
    if (prefs.autoRefresh) {
      refreshInterval = setInterval(() => {
        opsecActions.checkProxyStatus();
      }, prefs.refreshInterval);
    }
  });
}

export function stopAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval);
    refreshInterval = null;
  }
}

// Initialize on app start
export async function initializeApp() {
  appState.update(s => ({ ...s, initialized: false }));
  
  try {
    // Check initial security status
    await opsecActions.checkProxyStatus();
    
    // Start auto-refresh
    startAutoRefresh();
    
    appState.update(s => ({ ...s, initialized: true }));
  } catch (error) {
    appState.update(s => ({ 
      ...s, 
      initialized: false,
      error: error instanceof Error ? error.message : 'Failed to initialize application'
    }));
  }
}
