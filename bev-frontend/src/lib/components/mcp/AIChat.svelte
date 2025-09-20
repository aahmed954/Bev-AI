<!-- BEV AI Chat Interface - Security-First Assistant Integration -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { mcpStore } from '$lib/stores/mcpStore';
  import DOMPurify from 'isomorphic-dompurify';
  import { marked } from 'marked';
  import type { ChatMessage, MCPTool } from '$lib/mcp/types';

  let messageInput = '';
  let chatContainer: HTMLElement;
  let isLoading = false;
  let showSecurityInfo = false;

  $: pendingConsent = $mcpStore.pendingConsents[0] || null;

  // Auto-scroll to bottom on new messages
  $: if ($mcpStore.chatHistory.length && chatContainer) {
    setTimeout(() => {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 100);
  }

  async function sendMessage() {
    if (!messageInput.trim() || isLoading) return;

    const message = messageInput;
    messageInput = '';
    isLoading = true;

    try {
      await mcpStore.sendMessage(message, {
        source: 'chat',
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to send message:', error);
    } finally {
      isLoading = false;
    }
  }

  function handleConsent(approved: boolean, remember: boolean = false) {
    if (pendingConsent) {
      mcpStore.handleConsent(0, approved, remember);
    }
  }

  function formatMessage(content: string): string {
    // Sanitize and parse markdown
    const html = marked(content);
    return DOMPurify.sanitize(html);
  }

  function getMessageClass(role: string): string {
    switch(role) {
      case 'user': return 'bg-gray-800 text-cyan-400';
      case 'assistant': return 'bg-gray-900 text-green-400';
      case 'system': return 'bg-purple-900 text-purple-300';
      case 'tool': return 'bg-orange-900 text-orange-300';
      default: return 'bg-gray-800 text-gray-300';
    }
  }

  onMount(() => {
    // Focus input on mount
    const input = document.querySelector('.chat-input') as HTMLInputElement;
    if (input) input.focus();
  });
</script>

<div class="ai-chat h-full flex flex-col bg-black text-green-400 font-mono">
  <!-- Header -->
  <div class="chat-header p-4 bg-gray-900 border-b border-green-500 flex items-center justify-between">
    <div class="flex items-center gap-3">
      <div class="status-indicator">
        {#if $mcpStore.connected}
          <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
        {:else}
          <div class="w-3 h-3 bg-red-500 rounded-full"></div>
        {/if}
      </div>
      <h2 class="text-xl font-bold text-green-400">AI ASSISTANT</h2>
      <span class="text-xs text-gray-500">MCP v1.0</span>
    </div>
    
    <div class="flex items-center gap-2">
      <button 
        class="px-3 py-1 text-xs bg-gray-800 hover:bg-gray-700 rounded"
        on:click={() => showSecurityInfo = !showSecurityInfo}
      >
        🔒 SECURITY
      </button>
      <button 
        class="px-3 py-1 text-xs bg-red-900 hover:bg-red-800 rounded"
        on:click={() => mcpStore.clearChat()}
      >
        CLEAR
      </button>
    </div>
  </div>

  <!-- Security Info Panel -->
  {#if showSecurityInfo}
    <div class="security-info p-4 bg-gray-900 border-b border-gray-700">
      <div class="text-xs space-y-1">
        <div>🔒 End-to-end encryption: <span class="text-green-400">ACTIVE</span></div>
        <div>🌐 SOCKS5 Proxy: <span class="text-green-400">{$mcpStore.systemMetrics?.proxy.exitIP || 'CHECKING...'}</span></div>
        <div>🛡️ Security Level: <span class="text-yellow-400">PARANOID</span></div>
        <div>✓ Tool Consent: <span class="text-cyan-400">REQUIRED</span></div>
      </div>
    </div>
  {/if}

  <!-- Messages Container -->
  <div 
    bind:this={chatContainer}
    class="chat-messages flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-gray-700"
  >
    {#each $mcpStore.chatHistory as message}
      <div class="message {message.role === 'user' ? 'ml-auto' : ''} max-w-3xl">
        <div class="message-header text-xs text-gray-500 mb-1">
          {message.role.toUpperCase()} • {new Date(message.timestamp).toLocaleTimeString()}
        </div>
        <div class="message-content {getMessageClass(message.role)} p-3 rounded">
          {#if message.role === 'tool'}
            <div class="tool-invocation">
              <span class="text-yellow-400">⚡ Tool:</span> {message.content}
            </div>
          {:else}
            {@html formatMessage(message.content)}
          {/if}
          
          {#if message.metadata?.toolInvocations}
            <div class="tool-invocations mt-2 pt-2 border-t border-gray-700">
              {#each message.metadata.toolInvocations as invocation}
                <div class="text-xs">
                  <span class="text-cyan-400">→</span> 
                  {invocation.toolName}
                  <span class="status-badge ml-2 px-1 rounded {
                    invocation.status === 'completed' ? 'bg-green-900 text-green-300' :
                    invocation.status === 'failed' ? 'bg-red-900 text-red-300' :
                    invocation.status === 'rejected' ? 'bg-orange-900 text-orange-300' :
                    'bg-gray-700 text-gray-300'
                  }">
                    {invocation.status}
                  </span>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      </div>
    {/each}

    {#if isLoading}
      <div class="loading-indicator flex items-center gap-2 text-gray-500">
        <div class="spinner w-4 h-4 border-2 border-green-500 border-t-transparent rounded-full animate-spin"></div>
        <span>AI is thinking...</span>
      </div>
    {/if}
  </div>

  <!-- Input Area -->
  <div class="chat-input-area p-4 bg-gray-900 border-t border-green-500">
    <form on:submit|preventDefault={sendMessage} class="flex gap-2">
      <input
        bind:value={messageInput}
        class="chat-input flex-1 px-4 py-2 bg-black text-green-400 border border-green-500 rounded focus:outline-none focus:border-cyan-400"
        placeholder="Ask the AI assistant..."
        disabled={!$mcpStore.connected || isLoading}
      />
      <button
        type="submit"
        class="px-6 py-2 bg-green-600 hover:bg-green-500 text-black font-bold rounded disabled:opacity-50"
        disabled={!$mcpStore.connected || isLoading || !messageInput.trim()}
      >
        SEND
      </button>
    </form>
    
    {#if !$mcpStore.connected}
      <div class="mt-2 text-xs text-red-400">
        ⚠️ Disconnected from MCP server. Attempting to reconnect...
      </div>
    {/if}
  </div>

  <!-- Security Consent Modal -->
  {#if pendingConsent}
    <div class="consent-modal fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div class="consent-dialog bg-gray-900 border-2 border-red-500 rounded-lg p-6 max-w-lg w-full">
        <div class="header mb-4">
          <h3 class="text-xl font-bold text-red-400">⚠️ TOOL INVOCATION REQUEST</h3>
          <p class="text-xs text-gray-400 mt-1">Security consent required</p>
        </div>

        <div class="tool-details bg-black p-4 rounded mb-4 text-sm">
          <div class="mb-2">
            <span class="text-gray-500">Tool:</span>
            <span class="text-cyan-400 ml-2">{pendingConsent.tool.name}</span>
          </div>
          <div class="mb-2">
            <span class="text-gray-500">Category:</span>
            <span class="text-yellow-400 ml-2">{pendingConsent.tool.category}</span>
          </div>
          <div class="mb-2">
            <span class="text-gray-500">Risk Level:</span>
            <span class="ml-2 {
              pendingConsent.tool.riskLevel === 'critical' ? 'text-red-400' :
              pendingConsent.tool.riskLevel === 'high' ? 'text-orange-400' :
              pendingConsent.tool.riskLevel === 'medium' ? 'text-yellow-400' :
              'text-green-400'
            }">
              {pendingConsent.tool.riskLevel.toUpperCase()}
            </span>
          </div>
          <div>
            <span class="text-gray-500">Description:</span>
            <div class="text-gray-300 mt-1">{pendingConsent.tool.description}</div>
          </div>
          {#if pendingConsent.tool.parameters && Object.keys(pendingConsent.tool.parameters).length > 0}
            <div class="mt-3 pt-3 border-t border-gray-700">
              <span class="text-gray-500">Parameters:</span>
              <pre class="text-xs text-gray-400 mt-1 overflow-x-auto">{JSON.stringify(pendingConsent.tool.parameters, null, 2)}</pre>
            </div>
          {/if}
        </div>

        <div class="actions flex gap-3">
          <button
            class="flex-1 px-4 py-2 bg-red-600 hover:bg-red-500 text-white font-bold rounded"
            on:click={() => handleConsent(false)}
          >
            DENY
          </button>
          <button
            class="flex-1 px-4 py-2 bg-green-600 hover:bg-green-500 text-black font-bold rounded"
            on:click={() => handleConsent(true, false)}
          >
            APPROVE ONCE
          </button>
          <button
            class="flex-1 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-black font-bold rounded"
            on:click={() => handleConsent(true, true)}
          >
            ALWAYS APPROVE
          </button>
        </div>

        <div class="security-note mt-4 text-xs text-gray-500">
          🔒 This action will be logged for security audit
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .scrollbar-thin::-webkit-scrollbar {
    width: 6px;
  }
  
  .scrollbar-thin::-webkit-scrollbar-track {
    background: #1a1a1a;
  }
  
  .scrollbar-thumb-gray-700::-webkit-scrollbar-thumb {
    background: #374151;
    border-radius: 3px;
  }
  
  .message-content :global(p) {
    margin-bottom: 0.5rem;
  }
  
  .message-content :global(pre) {
    background: #000;
    padding: 0.5rem;
    border-radius: 0.25rem;
    overflow-x: auto;
  }
  
  .message-content :global(code) {
    background: #1a1a1a;
    padding: 0.125rem 0.25rem;
    border-radius: 0.125rem;
  }
</style>
