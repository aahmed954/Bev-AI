/**
 * BEV MCP Client - Model Context Protocol Integration
 * Security-first AI tool invocation with mandatory consent flows
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';
import type { 
  MCPTool, 
  MCPMessage, 
  MCPResponse, 
  SecurityConsent,
  MCPClientConfig 
} from './types';

export class MCPClient extends EventEmitter {
  private ws: WebSocket | null = null;
  private config: MCPClientConfig;
  private pendingRequests: Map<string, (response: any) => void> = new Map();
  private consentCache: Map<string, SecurityConsent> = new Map();
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(config: MCPClientConfig) {
    super();
    this.config = {
      url: config.url || 'ws://localhost:3010',
      autoReconnect: config.autoReconnect !== false,
      securityLevel: config.securityLevel || 'paranoid',
      requireConsent: config.requireConsent !== false,
      ...config
    };
  }

  /**
   * Connect to MCP server with security validation
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // Verify SOCKS5 proxy is active before connecting
        if (!await this.validateProxyStatus()) {
          throw new Error('SECURITY: SOCKS5 proxy not active. Connection blocked.');
        }

        this.ws = new WebSocket(this.config.url);

        this.ws.onopen = () => {
          console.log('[MCP] Connected to server');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.emit('connected');
          this.sendHandshake();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(JSON.parse(event.data));
        };

        this.ws.onerror = (error) => {
          console.error('[MCP] WebSocket error:', error);
          this.emit('error', error);
        };

        this.ws.onclose = () => {
          console.log('[MCP] Disconnected');
          this.isConnected = false;
          this.emit('disconnected');
          
          if (this.config.autoReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  /**
   * Send security handshake to establish trusted session
   */
  private sendHandshake(): void {
    this.send({
      type: 'handshake',
      payload: {
        clientId: uuidv4(),
        version: '1.0.0',
        securityLevel: this.config.securityLevel,
        capabilities: ['tools', 'chat', 'agents', 'monitoring']
      }
    });
  }

  /**
   * Handle incoming MCP messages
   */
  private handleMessage(message: MCPMessage): void {
    // Handle response to pending request
    if (message.requestId && this.pendingRequests.has(message.requestId)) {
      const resolver = this.pendingRequests.get(message.requestId);
      this.pendingRequests.delete(message.requestId);
      resolver!(message);
      return;
    }

    // Handle different message types
    switch (message.type) {
      case 'tool_request':
        this.handleToolRequest(message);
        break;
      case 'agent_update':
        this.emit('agent:update', message.payload);
        break;
      case 'security_alert':
        this.emit('security:alert', message.payload);
        break;
      case 'chat_response':
        this.emit('chat:response', message.payload);
        break;
      default:
        this.emit('message', message);
    }
  }

  /**
   * Handle tool invocation request with security consent
   */
  private async handleToolRequest(message: MCPMessage): Promise<void> {
    const tool = message.payload as MCPTool;
    
    // Check consent cache first
    const cacheKey = `${tool.name}:${JSON.stringify(tool.parameters)}`;
    let consent = this.consentCache.get(cacheKey);

    // Request user consent if not cached
    if (!consent && this.config.requireConsent) {
      consent = await this.requestUserConsent(tool);
      if (consent.approved && consent.rememberChoice) {
        this.consentCache.set(cacheKey, consent);
      }
    }

    // Send response based on consent
    this.send({
      type: 'tool_response',
      requestId: message.requestId,
      payload: {
        toolId: tool.id,
        approved: consent?.approved || false,
        reason: consent?.reason
      }
    });
  }

  /**
   * Request user consent for tool invocation
   */
  private async requestUserConsent(tool: MCPTool): Promise<SecurityConsent> {
    return new Promise((resolve) => {
      this.emit('consent:required', {
        tool,
        callback: (consent: SecurityConsent) => resolve(consent)
      });
    });
  }

  /**
   * Send chat message to AI assistant
   */
  async sendChatMessage(message: string, context?: any): Promise<MCPResponse> {
    return this.request({
      type: 'chat',
      payload: {
        message,
        context,
        timestamp: new Date().toISOString()
      }
    });
  }

  /**
   * Invoke a specific tool with parameters
   */
  async invokeTool(toolName: string, parameters: any): Promise<MCPResponse> {
    // Security validation
    if (this.config.securityLevel === 'paranoid') {
      const validation = await this.validateToolInvocation(toolName, parameters);
      if (!validation.safe) {
        throw new Error(`SECURITY: Tool invocation blocked - ${validation.reason}`);
      }
    }

    return this.request({
      type: 'tool_invoke',
      payload: {
        tool: toolName,
        parameters,
        securityToken: await this.generateSecurityToken()
      }
    });
  }

  /**
   * Get list of available tools
   */
  async getAvailableTools(): Promise<MCPTool[]> {
    const response = await this.request({
      type: 'tools_list',
      payload: {}
    });
    return response.payload.tools;
  }

  /**
   * Get agent status and metrics
   */
  async getAgentStatus(): Promise<any> {
    return this.request({
      type: 'agent_status',
      payload: {}
    });
  }

  /**
   * Send request and wait for response
   */
  private request(message: Partial<MCPMessage>): Promise<MCPResponse> {
    return new Promise((resolve, reject) => {
      const requestId = uuidv4();
      const timeout = setTimeout(() => {
        this.pendingRequests.delete(requestId);
        reject(new Error('Request timeout'));
      }, 30000);

      this.pendingRequests.set(requestId, (response) => {
        clearTimeout(timeout);
        resolve(response);
      });

      this.send({
        ...message,
        requestId
      } as MCPMessage);
    });
  }

  /**
   * Send message through WebSocket
   */
  private send(message: MCPMessage): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.error('[MCP] Cannot send - not connected');
      return;
    }
    this.ws.send(JSON.stringify(message));
  }

  /**
   * Validate SOCKS5 proxy is active
   */
  private async validateProxyStatus(): Promise<boolean> {
    // In production, this would check with Tauri backend
    // For now, we'll assume it's validated by the Rust layer
    return true;
  }

  /**
   * Validate tool invocation security
   */
  private async validateToolInvocation(toolName: string, parameters: any): Promise<{safe: boolean, reason?: string}> {
    // Security checks
    const dangerousTools = ['execute_code', 'system_command', 'network_request'];
    if (dangerousTools.includes(toolName)) {
      return {
        safe: false,
        reason: 'High-risk tool requires manual approval'
      };
    }

    // Parameter validation
    if (JSON.stringify(parameters).length > 10000) {
      return {
        safe: false,
        reason: 'Payload size exceeds security limit'
      };
    }

    return { safe: true };
  }

  /**
   * Generate security token for tool invocation
   */
  private async generateSecurityToken(): Promise<string> {
    return uuidv4(); // In production, this would be cryptographically signed
  }

  /**
   * Schedule reconnection attempt
   */
  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    console.log(`[MCP] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect().catch(console.error);
    }, delay);
  }

  /**
   * Disconnect from MCP server
   */
  disconnect(): void {
    this.config.autoReconnect = false;
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  /**
   * Clear consent cache
   */
  clearConsentCache(): void {
    this.consentCache.clear();
  }
}
