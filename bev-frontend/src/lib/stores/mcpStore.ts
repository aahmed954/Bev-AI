/**
 * BEV MCP Store - Reactive state management for AI integration
 */

import { writable, derived, get } from 'svelte/store';
import { MCPClient } from '$lib/mcp/MCPClient';
import type { 
  MCPState, 
  Agent, 
  ChatMessage, 
  Alert,
  Workflow,
  MCPTool,
  SecurityConsent,
  SystemMetrics,
  OPSECStatus
} from '$lib/mcp/types';

// Initial state
const initialState: MCPState = {
  connected: false,
  agents: [],
  activeWorkflows: [],
  chatHistory: [],
  systemMetrics: null,
  opsecStatus: null,
  alerts: [],
  pendingConsents: []
};

// Create main store
function createMCPStore() {
  const { subscribe, set, update } = writable<MCPState>(initialState);
  
  let client: MCPClient | null = null;

  return {
    subscribe,
    
    // Initialize MCP client
    async init() {
      client = new MCPClient({
        url: 'ws://localhost:3010',
        securityLevel: 'paranoid',
        requireConsent: true
      });

      // Set up event handlers
      client.on('connected', () => {
        update(state => ({ ...state, connected: true }));
        this.refreshAgentStatus();
        this.startMetricsPolling();
      });

      client.on('disconnected', () => {
        update(state => ({ ...state, connected: false }));
      });

      client.on('agent:update', (agent: Agent) => {
        update(state => ({
          ...state,
          agents: state.agents.map(a => a.id === agent.id ? agent : a)
        }));
      });

      client.on('chat:response', (message: ChatMessage) => {
        update(state => ({
          ...state,
          chatHistory: [...state.chatHistory, message]
        }));
      });

      client.on('security:alert', (alert: Alert) => {
        update(state => ({
          ...state,
          alerts: [...state.alerts, alert]
        }));
      });

      client.on('consent:required', ({ tool, callback }: any) => {
        update(state => ({
          ...state,
          pendingConsents: [...state.pendingConsents, { tool, callback }]
        }));
      });

      client.on('metrics:update', (metrics: SystemMetrics) => {
        update(state => ({
          ...state,
          systemMetrics: metrics
        }));
      });

      // Connect to server
      await client.connect();
    },

    // Send chat message
    async sendMessage(message: string, context?: any) {
      if (!client) throw new Error('MCP client not initialized');
      
      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: message,
        timestamp: new Date(),
        metadata: { context }
      };

      update(state => ({
        ...state,
        chatHistory: [...state.chatHistory, userMessage]
      }));

      const response = await client.sendChatMessage(message, context);
      return response;
    },

    // Invoke tool
    async invokeTool(toolName: string, parameters: any) {
      if (!client) throw new Error('MCP client not initialized');
      return client.invokeTool(toolName, parameters);
    },

    // Handle consent
    handleConsent(index: number, approved: boolean, rememberChoice: boolean = false) {
      update(state => {
        const consent = state.pendingConsents[index];
        if (consent) {
          consent.callback({
            approved,
            rememberChoice,
            reason: approved ? 'User approved' : 'User denied'
          });
        }
        
        return {
          ...state,
          pendingConsents: state.pendingConsents.filter((_, i) => i !== index)
        };
      });
    },

    // Refresh agent status
    async refreshAgentStatus() {
      if (!client) return;
      
      const status = await client.getAgentStatus();
      update(state => ({
        ...state,
        agents: status.payload.agents
      }));
    },

    // Start metrics polling
    startMetricsPolling() {
      setInterval(() => {
        // In production, this would fetch from the backend
        const mockMetrics: SystemMetrics = {
          timestamp: new Date(),
          proxy: {
            connected: true,
            exitIP: '185.220.101.45',
            circuit: 'guard-middle-exit',
            latency: 145
          },
          mcp: {
            connected: get(mcpStore).connected,
            activeAgents: get(mcpStore).agents.filter(a => a.status === 'working').length,
            queuedTasks: Math.floor(Math.random() * 10),
            completedTasks: Math.floor(Math.random() * 100)
          },
          resources: {
            cpu: Math.random() * 100,
            memory: Math.random() * 100,
            disk: Math.random() * 100,
            network: {
              in: Math.random() * 1000,
              out: Math.random() * 1000
            }
          },
          security: {
            threatsDetected: 0,
            blockedRequests: Math.floor(Math.random() * 5),
            activeAlerts: get(mcpStore).alerts.filter(a => !a.acknowledged)
          }
        };

        update(state => ({
          ...state,
          systemMetrics: mockMetrics
        }));
      }, 5000);
    },

    // Check OPSEC compliance
    async checkOPSEC() {
      // In production, this would query Tauri backend
      const opsecStatus: OPSECStatus = {
        compliant: true,
        proxyStatus: 'connected',
        exitIP: '185.220.101.45',
        circuitInfo: {
          id: crypto.randomUUID(),
          nodes: ['guard-node', 'middle-relay', 'exit-node'],
          latency: 145
        },
        leakTests: {
          dns: true,
          webrtc: true,
          javascript: true,
          cookies: true
        },
        recommendations: []
      };

      update(state => ({
        ...state,
        opsecStatus
      }));

      return opsecStatus;
    },

    // Acknowledge alert
    acknowledgeAlert(alertId: string) {
      update(state => ({
        ...state,
        alerts: state.alerts.map(a => 
          a.id === alertId ? { ...a, acknowledged: true } : a
        )
      }));
    },

    // Clear chat history
    clearChat() {
      update(state => ({
        ...state,
        chatHistory: []
      }));
    },

    // Disconnect
    disconnect() {
      if (client) {
        client.disconnect();
        client = null;
      }
      set(initialState);
    }
  };
}

// Create and export store instance
export const mcpStore = createMCPStore();

// Derived stores for specific views
export const connectedAgents = derived(
  mcpStore,
  $mcp => $mcp.agents.filter(a => a.status !== 'offline')
);

export const activeAlerts = derived(
  mcpStore,
  $mcp => $mcp.alerts.filter(a => !a.acknowledged)
);

export const workingAgents = derived(
  mcpStore,
  $mcp => $mcp.agents.filter(a => a.status === 'working')
);
