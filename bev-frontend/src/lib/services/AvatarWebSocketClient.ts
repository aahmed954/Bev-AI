/**
 * Avatar WebSocket Client - Real-time communication with advanced avatar service
 * Features: Auto-reconnection, message queuing, performance monitoring
 * Connects to: Advanced Avatar Service on port 8092
 */

export interface AvatarMessage {
  type: 'emotion_change' | 'speech_request' | 'gesture_request' | 'interaction' | 'osint_event' | 'system_status';
  data: any;
  timestamp?: number;
  id?: string;
}

export interface AvatarState {
  current_emotion: string;
  is_speaking: boolean;
  is_listening: boolean;
  gesture_active: boolean;
  animation_playing: boolean;
  interaction_mode: 'idle' | 'osint_analysis' | 'user_interaction' | 'system_monitoring';
  osint_context?: {
    investigation_id: string;
    current_task: string;
    progress: number;
    findings_count: number;
  };
}

export interface ConnectionConfig {
  url: string;
  reconnectInterval: number;
  maxReconnectAttempts: number;
  messageQueueSize: number;
  heartbeatInterval: number;
}

export class AvatarWebSocketClient {
  private ws: WebSocket | null = null;
  private config: ConnectionConfig;
  private messageQueue: AvatarMessage[] = [];
  private reconnectAttempts = 0;
  private heartbeatTimer: number | null = null;
  private isConnected = false;
  private subscribers: Map<string, ((message: AvatarMessage) => void)[]> = new Map();
  private avatarState: AvatarState = {
    current_emotion: 'neutral',
    is_speaking: false,
    is_listening: false,
    gesture_active: false,
    animation_playing: false,
    interaction_mode: 'idle'
  };

  constructor(config: Partial<ConnectionConfig> = {}) {
    this.config = {
      url: 'ws://localhost:8092/ws',
      reconnectInterval: 5000,
      maxReconnectAttempts: 10,
      messageQueueSize: 100,
      heartbeatInterval: 30000,
      ...config
    };
  }

  /**
   * Connect to the avatar service
   */
  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.config.url);

        this.ws.onopen = () => {
          console.log('Avatar WebSocket connected');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.startHeartbeat();
          this.processMessageQueue();
          this.emit('connection', { status: 'connected' });
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event);
        };

        this.ws.onclose = (event) => {
          console.log('Avatar WebSocket disconnected:', event.code, event.reason);
          this.isConnected = false;
          this.stopHeartbeat();
          this.emit('connection', { status: 'disconnected', code: event.code });
          
          if (this.reconnectAttempts < this.config.maxReconnectAttempts) {
            this.scheduleReconnect();
          } else {
            this.emit('connection', { status: 'failed', reason: 'Max reconnect attempts reached' });
          }
        };

        this.ws.onerror = (error) => {
          console.error('Avatar WebSocket error:', error);
          this.emit('connection', { status: 'error', error });
          reject(error);
        };

      } catch (error) {
        console.error('Failed to create WebSocket connection:', error);
        reject(error);
      }
    });
  }

  /**
   * Disconnect from the avatar service
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.isConnected = false;
    this.stopHeartbeat();
  }

  /**
   * Send message to avatar service
   */
  async sendMessage(message: AvatarMessage): Promise<void> {
    const messageWithTimestamp: AvatarMessage = {
      ...message,
      timestamp: Date.now(),
      id: this.generateMessageId()
    };

    if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(messageWithTimestamp));
      } catch (error) {
        console.error('Failed to send message:', error);
        this.queueMessage(messageWithTimestamp);
      }
    } else {
      this.queueMessage(messageWithTimestamp);
    }
  }

  /**
   * Avatar control methods
   */
  async setEmotion(emotion: string, context?: any): Promise<void> {
    await this.sendMessage({
      type: 'emotion_change',
      data: { emotion, context }
    });
    this.avatarState.current_emotion = emotion;
  }

  async speak(text: string, emotion?: string, priority: 'low' | 'normal' | 'high' = 'normal'): Promise<void> {
    await this.sendMessage({
      type: 'speech_request',
      data: { text, emotion, priority }
    });
    this.avatarState.is_speaking = true;
  }

  async performGesture(gesture: string, intensity: number = 1.0): Promise<void> {
    await this.sendMessage({
      type: 'gesture_request',
      data: { gesture, intensity }
    });
    this.avatarState.gesture_active = true;
  }

  async reportOSINTEvent(event: {
    type: 'investigation_start' | 'investigation_progress' | 'investigation_complete' | 'finding_discovered' | 'threat_detected';
    investigation_id: string;
    data: any;
  }): Promise<void> {
    await this.sendMessage({
      type: 'osint_event',
      data: event
    });

    // Update OSINT context
    this.avatarState.osint_context = {
      investigation_id: event.investigation_id,
      current_task: event.data.task || 'Unknown',
      progress: event.data.progress || 0,
      findings_count: event.data.findings_count || 0
    };

    this.avatarState.interaction_mode = 'osint_analysis';
  }

  async reportUserInteraction(interaction: {
    type: 'click' | 'hover' | 'scroll' | 'input' | 'selection';
    element: string;
    data?: any;
  }): Promise<void> {
    await this.sendMessage({
      type: 'interaction',
      data: interaction
    });
  }

  /**
   * OSINT Integration methods
   */
  async startInvestigation(investigationId: string, target: string, analysisType: string): Promise<void> {
    await this.reportOSINTEvent({
      type: 'investigation_start',
      investigation_id: investigationId,
      data: {
        target,
        analysis_type: analysisType,
        start_time: new Date().toISOString()
      }
    });

    // Set appropriate emotion for investigation start
    await this.setEmotion('focused', { context: 'investigation_start', target });
    await this.speak(`Starting ${analysisType} investigation on ${target}`, 'focused', 'high');
  }

  async reportInvestigationProgress(investigationId: string, progress: number, currentTask: string, findings?: any[]): Promise<void> {
    await this.reportOSINTEvent({
      type: 'investigation_progress',
      investigation_id: investigationId,
      data: {
        progress,
        task: currentTask,
        findings_count: findings?.length || 0,
        timestamp: new Date().toISOString()
      }
    });

    // Emotional responses based on progress
    if (progress > 75) {
      await this.setEmotion('excited', { context: 'investigation_progress', progress });
    } else if (progress > 50) {
      await this.setEmotion('determined', { context: 'investigation_progress', progress });
    }
  }

  async reportFinding(investigationId: string, finding: {
    type: string;
    severity: 'low' | 'medium' | 'high' | 'critical';
    description: string;
    data: any;
  }): Promise<void> {
    await this.reportOSINTEvent({
      type: 'finding_discovered',
      investigation_id: investigationId,
      data: finding
    });

    // Emotional response based on finding severity
    const emotionMap = {
      low: 'satisfied',
      medium: 'interested',
      high: 'excited',
      critical: 'alert'
    };

    const emotion = emotionMap[finding.severity];
    await this.setEmotion(emotion, { context: 'finding_discovered', severity: finding.severity });
    
    // Announce critical findings
    if (finding.severity === 'critical') {
      await this.speak(`Critical finding discovered: ${finding.description}`, 'alert', 'high');
    }
  }

  async completeInvestigation(investigationId: string, summary: {
    findings_count: number;
    threats_detected: number;
    completion_time: number;
    success: boolean;
  }): Promise<void> {
    await this.reportOSINTEvent({
      type: 'investigation_complete',
      investigation_id: investigationId,
      data: {
        ...summary,
        end_time: new Date().toISOString()
      }
    });

    // Celebration or concern based on results
    if (summary.threats_detected > 0) {
      await this.setEmotion('concerned', { context: 'investigation_complete', threats: summary.threats_detected });
      await this.speak(`Investigation complete. Found ${summary.threats_detected} potential threats requiring attention.`, 'concerned', 'high');
    } else if (summary.findings_count > 0) {
      await this.setEmotion('satisfied', { context: 'investigation_complete', findings: summary.findings_count });
      await this.speak(`Investigation completed successfully with ${summary.findings_count} findings.`, 'satisfied');
    } else {
      await this.setEmotion('neutral', { context: 'investigation_complete' });
      await this.speak('Investigation complete. No significant findings detected.', 'neutral');
    }

    this.avatarState.interaction_mode = 'idle';
  }

  /**
   * Event subscription
   */
  subscribe(eventType: string, callback: (message: AvatarMessage) => void): () => void {
    if (!this.subscribers.has(eventType)) {
      this.subscribers.set(eventType, []);
    }
    
    const callbacks = this.subscribers.get(eventType)!;
    callbacks.push(callback);

    // Return unsubscribe function
    return () => {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    };
  }

  /**
   * Get current avatar state
   */
  getState(): AvatarState {
    return { ...this.avatarState };
  }

  /**
   * Private methods
   */
  private handleMessage(event: MessageEvent): void {
    try {
      const message: AvatarMessage = JSON.parse(event.data);
      
      // Update local state based on message
      this.updateState(message);
      
      // Emit to subscribers
      this.emit(message.type, message);
      this.emit('message', message);
      
    } catch (error) {
      console.error('Failed to parse avatar message:', error);
    }
  }

  private updateState(message: AvatarMessage): void {
    switch (message.type) {
      case 'emotion_change':
        this.avatarState.current_emotion = message.data.emotion;
        break;
      case 'speech_request':
        if (message.data.status === 'started') {
          this.avatarState.is_speaking = true;
        } else if (message.data.status === 'completed') {
          this.avatarState.is_speaking = false;
        }
        break;
      case 'gesture_request':
        if (message.data.status === 'started') {
          this.avatarState.gesture_active = true;
        } else if (message.data.status === 'completed') {
          this.avatarState.gesture_active = false;
        }
        break;
      case 'system_status':
        if (message.data.listening !== undefined) {
          this.avatarState.is_listening = message.data.listening;
        }
        if (message.data.animation_playing !== undefined) {
          this.avatarState.animation_playing = message.data.animation_playing;
        }
        break;
    }
  }

  private emit(eventType: string, data: any): void {
    const callbacks = this.subscribers.get(eventType);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('Error in avatar event callback:', error);
        }
      });
    }
  }

  private queueMessage(message: AvatarMessage): void {
    if (this.messageQueue.length >= this.config.messageQueueSize) {
      this.messageQueue.shift(); // Remove oldest message
    }
    this.messageQueue.push(message);
  }

  private async processMessageQueue(): Promise<void> {
    while (this.messageQueue.length > 0 && this.isConnected) {
      const message = this.messageQueue.shift();
      if (message) {
        await this.sendMessage(message);
        // Small delay to avoid overwhelming the service
        await new Promise(resolve => setTimeout(resolve, 10));
      }
    }
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = Math.min(this.config.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1), 30000);
    
    console.log(`Scheduling reconnection attempt ${this.reconnectAttempts}/${this.config.maxReconnectAttempts} in ${delay}ms`);
    
    setTimeout(() => {
      if (!this.isConnected) {
        this.connect().catch(error => {
          console.error('Reconnection failed:', error);
        });
      }
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = window.setInterval(() => {
      if (this.isConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          type: 'heartbeat',
          timestamp: Date.now()
        }));
      }
    }, this.config.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private generateMessageId(): string {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

/**
 * Singleton instance for global access
 */
export const avatarClient = new AvatarWebSocketClient();

/**
 * Utility functions for OSINT integration
 */
export class OSINTAvatarIntegration {
  private static client = avatarClient;

  static async initialize(): Promise<void> {
    try {
      await this.client.connect();
      console.log('Avatar OSINT integration initialized');
    } catch (error) {
      console.error('Failed to initialize avatar OSINT integration:', error);
    }
  }

  static async announceAnalyzerStart(analyzer: string, target: string): Promise<void> {
    await this.client.setEmotion('focused');
    await this.client.speak(`Starting ${analyzer} analysis on ${target}`, 'focused');
  }

  static async announceAnalyzerProgress(analyzer: string, progress: number): Promise<void> {
    if (progress === 50) {
      await this.client.setEmotion('determined');
    } else if (progress === 100) {
      await this.client.setEmotion('satisfied');
      await this.client.speak(`${analyzer} analysis complete`, 'satisfied');
    }
  }

  static async announceFindings(analyzer: string, findings: any[]): Promise<void> {
    const count = findings.length;
    
    if (count === 0) {
      await this.client.setEmotion('neutral');
      await this.client.speak(`${analyzer} analysis found no significant results`, 'neutral');
    } else if (count < 5) {
      await this.client.setEmotion('interested');
      await this.client.speak(`${analyzer} analysis found ${count} findings`, 'interested');
    } else {
      await this.client.setEmotion('excited');
      await this.client.speak(`${analyzer} analysis discovered ${count} significant findings`, 'excited');
    }
  }

  static async announceThreat(severity: 'low' | 'medium' | 'high' | 'critical', description: string): Promise<void> {
    const emotionMap = {
      low: 'concerned',
      medium: 'alert',
      high: 'worried',
      critical: 'alarmed'
    };

    await this.client.setEmotion(emotionMap[severity]);
    await this.client.speak(`${severity.toUpperCase()} threat detected: ${description}`, emotionMap[severity], 'high');
  }

  static async celebrateSuccess(message: string): Promise<void> {
    await this.client.setEmotion('happy');
    await this.client.performGesture('thumbs_up');
    await this.client.speak(message, 'happy');
  }

  static async expressConcern(message: string): Promise<void> {
    await this.client.setEmotion('worried');
    await this.client.speak(message, 'worried');
  }

  static getClient(): AvatarWebSocketClient {
    return this.client;
  }
}