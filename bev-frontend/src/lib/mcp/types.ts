/**
 * BEV MCP Type Definitions
 * Comprehensive types for Model Context Protocol integration
 */

// Core MCP types
export interface MCPMessage {
  type: string;
  requestId?: string;
  payload: any;
  timestamp?: string;
}

export interface MCPResponse {
  success: boolean;
  payload: any;
  error?: string;
}

export interface MCPTool {
  id: string;
  name: string;
  description: string;
  parameters: any;
  category: 'osint' | 'crypto' | 'darknet' | 'social' | 'analysis' | 'system';
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface SecurityConsent {
  approved: boolean;
  reason?: string;
  rememberChoice?: boolean;
  expiresAt?: Date;
}

export interface MCPClientConfig {
  url: string;
  autoReconnect?: boolean;
  securityLevel?: 'standard' | 'enhanced' | 'paranoid';
  requireConsent?: boolean;
  proxyConfig?: {
    host: string;
    port: number;
    type: 'socks5';
  };
}

// Chat and conversation types
export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
  timestamp: Date;
  metadata?: {
    toolInvocations?: ToolInvocation[];
    securityFlags?: string[];
    context?: any;
  };
}

export interface ToolInvocation {
  toolId: string;
  toolName: string;
  parameters: any;
  result?: any;
  status: 'pending' | 'approved' | 'rejected' | 'completed' | 'failed';
  timestamp: Date;
}

// Agent coordination types
export interface Agent {
  id: string;
  name: string;
  type: 'osint' | 'crypto' | 'darknet' | 'monitor' | 'analyzer';
  status: 'idle' | 'working' | 'error' | 'offline';
  capabilities: string[];
  currentTask?: AgentTask;
  metrics: AgentMetrics;
}

export interface AgentTask {
  id: string;
  type: string;
  description: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'queued' | 'running' | 'completed' | 'failed';
  progress: number;
  startTime: Date;
  endTime?: Date;
  result?: any;
}

export interface AgentMetrics {
  tasksCompleted: number;
  tasksFailed: number;
  averageResponseTime: number;
  uptime: number;
  lastActive: Date;
  resourceUsage: {
    cpu: number;
    memory: number;
    network: number;
  };
}

// Workflow types
export interface Workflow {
  id: string;
  name: string;
  description: string;
  agents: string[];
  steps: WorkflowStep[];
  status: 'draft' | 'running' | 'completed' | 'failed' | 'paused';
  createdAt: Date;
  startedAt?: Date;
  completedAt?: Date;
}

export interface WorkflowStep {
  id: string;
  agentId: string;
  action: string;
  parameters: any;
  dependencies: string[];
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  result?: any;
  error?: string;
}

// Monitoring types
export interface SystemMetrics {
  timestamp: Date;
  proxy: {
    connected: boolean;
    exitIP: string;
    circuit: string;
    latency: number;
  };
  mcp: {
    connected: boolean;
    activeAgents: number;
    queuedTasks: number;
    completedTasks: number;
  };
  resources: {
    cpu: number;
    memory: number;
    disk: number;
    network: {
      in: number;
      out: number;
    };
  };
  security: {
    threatsDetected: number;
    blockedRequests: number;
    activeAlerts: Alert[];
  };
}

export interface Alert {
  id: string;
  severity: 'info' | 'warning' | 'critical';
  type: string;
  message: string;
  timestamp: Date;
  acknowledged: boolean;
  metadata?: any;
}

// OPSEC compliance types
export interface OPSECStatus {
  compliant: boolean;
  proxyStatus: 'connected' | 'disconnected' | 'error';
  exitIP: string;
  circuitInfo: {
    id: string;
    nodes: string[];
    latency: number;
  };
  leakTests: {
    dns: boolean;
    webrtc: boolean;
    javascript: boolean;
    cookies: boolean;
  };
  recommendations: string[];
}

// Real-time event types
export type MCPEvent = 
  | { type: 'agent:statusChange'; payload: Agent }
  | { type: 'task:progress'; payload: { taskId: string; progress: number } }
  | { type: 'workflow:update'; payload: Workflow }
  | { type: 'security:alert'; payload: Alert }
  | { type: 'metrics:update'; payload: SystemMetrics }
  | { type: 'chat:message'; payload: ChatMessage }
  | { type: 'tool:invocation'; payload: ToolInvocation };

// Store types for Svelte
export interface MCPState {
  connected: boolean;
  agents: Agent[];
  activeWorkflows: Workflow[];
  chatHistory: ChatMessage[];
  systemMetrics: SystemMetrics | null;
  opsecStatus: OPSECStatus | null;
  alerts: Alert[];
  pendingConsents: Array<{
    tool: MCPTool;
    callback: (consent: SecurityConsent) => void;
  }>;
}
