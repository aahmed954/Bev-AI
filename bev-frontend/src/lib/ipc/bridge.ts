/**
 * BEV OSINT Secure IPC Bridge
 * Handles all communication between Rust backend and frontend
 * Security-first approach with command validation
 */

import { invoke as tauriInvoke } from '@tauri-apps/api/core';
import type { InvokeArgs } from '@tauri-apps/api/core';

// Command types for type-safe IPC
export enum IPCCommand {
  // Proxy & OPSEC Commands
  GET_PROXY_STATUS = 'get_proxy_status',
  NEW_TOR_CIRCUIT = 'new_tor_circuit',
  VERIFY_PROXY_ENFORCEMENT = 'verify_proxy_enforcement',
  GET_EXIT_NODES = 'get_exit_nodes',
  SET_EXIT_COUNTRY = 'set_exit_country',
  
  // OSINT Analysis Commands
  ANALYZE_DARKNET = 'analyze_darknet',
  TRACK_CRYPTOCURRENCY = 'track_cryptocurrency',
  GET_THREAT_INTEL = 'get_threat_intel',
  SEARCH_SOCIAL_INTEL = 'search_social_intel',
  
  // MCP Integration Commands
  MCP_INVOKE_TOOL = 'mcp_invoke_tool',
  MCP_GET_CONSENT = 'mcp_get_consent',
  MCP_LIST_TOOLS = 'mcp_list_tools',
  MCP_EXECUTE_WORKFLOW = 'mcp_execute_workflow',
  
  // System Commands
  GET_SYSTEM_STATUS = 'get_system_status',
  GET_AGENT_STATUS = 'get_agent_status',
  EXPORT_REPORT = 'export_report',
  SAVE_CONFIGURATION = 'save_configuration',
  LOAD_CONFIGURATION = 'load_configuration',
}

// Response types
export interface ProxyStatus {
  connected: boolean;
  exitIp: string | null;
  exitCountry: string | null;
  circuitPath: string[];
  bandwidth: {
    download: number;
    upload: number;
  };
  uptime: number;
}

export interface SecurityValidation {
  proxyEnforced: boolean;
  dnsLeakProtection: boolean;
  webrtcBlocked: boolean;
  fingerprintResistance: boolean;
  issues: string[];
}

export interface MCPToolInvocation {
  toolId: string;
  parameters: Record<string, unknown>;
  requiresConsent: boolean;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface DarknetAnalysis {
  markets: Array<{
    name: string;
    status: 'online' | 'offline' | 'suspicious';
    lastSeen: string;
    vendors: number;
    products: number;
  }>;
  threats: Array<{
    type: string;
    severity: string;
    description: string;
    timestamp: string;
  }>;
}

// Error handling
export class IPCError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'IPCError';
  }
}

// Security validation wrapper
async function secureInvoke<T>(
  command: IPCCommand,
  args?: InvokeArgs
): Promise<T> {
  try {
    // Validate proxy enforcement before sensitive operations
    const sensitiveCommands = [
      IPCCommand.ANALYZE_DARKNET,
      IPCCommand.TRACK_CRYPTOCURRENCY,
      IPCCommand.SEARCH_SOCIAL_INTEL,
      IPCCommand.MCP_INVOKE_TOOL,
    ];
    
    if (sensitiveCommands.includes(command)) {
      const validation = await tauriInvoke<SecurityValidation>(
        IPCCommand.VERIFY_PROXY_ENFORCEMENT
      );
      
      if (!validation.proxyEnforced) {
        throw new IPCError(
          'Security validation failed: Proxy not enforced',
          'SECURITY_VIOLATION',
          validation
        );
      }
    }
    
    // Execute the command
    const result = await tauriInvoke<T>(command, args);
    return result;
  } catch (error) {
    if (error instanceof IPCError) {
      throw error;
    }
    
    throw new IPCError(
      `IPC command failed: ${command}`,
      'IPC_FAILURE',
      error
    );
  }
}

// Exported API functions
export const ipc = {
  // Proxy & OPSEC
  async getProxyStatus(): Promise<ProxyStatus> {
    return secureInvoke<ProxyStatus>(IPCCommand.GET_PROXY_STATUS);
  },
  
  async newTorCircuit(): Promise<void> {
    return secureInvoke<void>(IPCCommand.NEW_TOR_CIRCUIT);
  },
  
  async verifyProxyEnforcement(): Promise<SecurityValidation> {
    return secureInvoke<SecurityValidation>(IPCCommand.VERIFY_PROXY_ENFORCEMENT);
  },
  
  async setExitCountry(countryCode: string): Promise<void> {
    return secureInvoke<void>(IPCCommand.SET_EXIT_COUNTRY, { countryCode });
  },
  
  // OSINT Analysis
  async analyzeDarknet(params: {
    markets?: string[];
    searchTerms?: string[];
    timeRange?: { start: string; end: string };
  }): Promise<DarknetAnalysis> {
    return secureInvoke<DarknetAnalysis>(IPCCommand.ANALYZE_DARKNET, params);
  },
  
  async trackCryptocurrency(params: {
    addresses: string[];
    chains: string[];
    depth?: number;
  }): Promise<unknown> {
    return secureInvoke<unknown>(IPCCommand.TRACK_CRYPTOCURRENCY, params);
  },
  
  // MCP Integration
  async invokeMCPTool(invocation: MCPToolInvocation): Promise<unknown> {
    // Check if consent is required
    if (invocation.requiresConsent) {
      const consent = await secureInvoke<boolean>(
        IPCCommand.MCP_GET_CONSENT,
        { toolId: invocation.toolId, riskLevel: invocation.riskLevel }
      );
      
      if (!consent) {
        throw new IPCError(
          'User consent denied for MCP tool invocation',
          'CONSENT_DENIED',
          invocation
        );
      }
    }
    
    return secureInvoke<unknown>(IPCCommand.MCP_INVOKE_TOOL, invocation);
  },
  
  async listMCPTools(): Promise<Array<{
    id: string;
    name: string;
    description: string;
    requiresConsent: boolean;
    riskLevel: string;
  }>> {
    return secureInvoke(IPCCommand.MCP_LIST_TOOLS);
  },
  
  // System
  async getSystemStatus(): Promise<{
    cpu: number;
    memory: number;
    disk: number;
    network: { in: number; out: number };
  }> {
    return secureInvoke(IPCCommand.GET_SYSTEM_STATUS);
  },
  
  async exportReport(params: {
    format: 'pdf' | 'json' | 'csv';
    data: unknown;
    filename: string;
  }): Promise<string> {
    return secureInvoke<string>(IPCCommand.EXPORT_REPORT, params);
  },
};

// WebSocket handler for real-time updates
export function createRealtimeConnection() {
  // This will be implemented with Tauri's event system
  return {
    onProxyStatusChange: (callback: (status: ProxyStatus) => void) => {
      // Subscribe to proxy status events
    },
    
    onThreatDetected: (callback: (threat: unknown) => void) => {
      // Subscribe to threat detection events  
    },
    
    onAgentUpdate: (callback: (update: unknown) => void) => {
      // Subscribe to agent status updates
    },
  };
}

export default ipc;
