<!-- RAG-Powered Document Q&A Chat Interface -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { invoke } from '@tauri-apps/api/core';
  import { marked } from 'marked';
  import DOMPurify from 'isomorphic-dompurify';
  
  const dispatch = createEventDispatcher();
  
  export let session;
  export let knowledgeStats;
  export let selectedVectorDB = 'qdrant';
  
  let messageInput = '';
  let isLoading = false;
  let chatContainer: HTMLElement;
  let contextSources = writable([]);
  let searchThreshold = 0.75;
  let maxContextChunks = 5;
  let streamingEnabled = true;
  let currentStreamedMessage = '';
  
  // Auto-scroll to bottom on new messages
  $: if (session?.messages?.length && chatContainer) {
    setTimeout(() => {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 100);
  }

  async function sendMessage() {
    if (!messageInput.trim() || isLoading || !session) return;

    const userMessage = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      role: 'user',
      content: messageInput.trim(),
      timestamp: new Date().toISOString()
    };

    // Add user message to session
    session.messages = [...session.messages, userMessage];
    dispatch('sessionUpdated', { sessionId: session.id, message: userMessage });

    const query = messageInput;
    messageInput = '';
    isLoading = true;

    try {
      // Perform RAG query
      const response = await invoke('ask_document', {
        sessionId: session.id,
        question: query,
        documentId: session.documentId,
        options: {
          vectorDB: selectedVectorDB,
          similarityThreshold: searchThreshold,
          maxContextChunks,
          includeMetadata: true,
          streaming: streamingEnabled
        }
      });

      const assistantMessage = {
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        role: 'assistant',
        content: response.answer || 'No response generated.',
        timestamp: new Date().toISOString(),
        sources: response.sources || [],
        confidence: response.confidence || 0
      };

      // Update context sources
      if (response.sources) {
        contextSources.set(response.sources);
      }

      // Add assistant message to session
      session.messages = [...session.messages, assistantMessage];
      dispatch('sessionUpdated', { sessionId: session.id, message: assistantMessage });

    } catch (error) {
      console.error('Chat query failed:', error);
      
      const errorMessage = {
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        role: 'system',
        content: `Error: ${error.message || 'Failed to process query'}`,
        timestamp: new Date().toISOString()
      };

      session.messages = [...session.messages, errorMessage];
      dispatch('sessionUpdated', { sessionId: session.id, message: errorMessage });
    } finally {
      isLoading = false;
    }
  }

  function formatMessage(content: string): string {
    const html = marked(content);
    return DOMPurify.sanitize(html);
  }

  function getMessageClass(role: string): string {
    switch(role) {
      case 'user': return 'bg-blue-600/20 border-blue-500/30 ml-12';
      case 'assistant': return 'bg-green-600/20 border-green-500/30 mr-12';
      case 'system': return 'bg-red-600/20 border-red-500/30 mx-12';
      default: return 'bg-gray-600/20 border-gray-500/30';
    }
  }

  function clearChat() {
    if (session) {
      session.messages = [];
      contextSources.set([]);
      dispatch('sessionUpdated', { sessionId: session.id, message: null });
    }
  }

  function exportChat() {
    if (!session) return;

    const exportData = {
      session: {
        name: session.name,
        documentName: session.documentName,
        created: session.created,
        messageCount: session.messages.length
      },
      messages: session.messages,
      context: $contextSources,
      exportTimestamp: new Date().toISOString(),
      settings: {
        vectorDB: selectedVectorDB,
        threshold: searchThreshold,
        maxContext: maxContextChunks
      }
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
      type: 'application/json' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${session.name.replace(/\s+/g, '-')}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function regenerateResponse() {
    if (!session?.messages?.length) return;
    
    // Find the last user message and resend it
    for (let i = session.messages.length - 1; i >= 0; i--) {
      if (session.messages[i].role === 'user') {
        messageInput = session.messages[i].content;
        // Remove the last assistant response if it exists
        if (i + 1 < session.messages.length && session.messages[i + 1].role === 'assistant') {
          session.messages = session.messages.slice(0, i + 1);
        }
        sendMessage();
        break;
      }
    }
  }

  async function improveContext() {
    if (!session?.messages?.length) return;
    
    try {
      const lastUserMessage = session.messages.filter(m => m.role === 'user').pop();
      if (!lastUserMessage) return;

      const improvedSources = await invoke('enhance_context', {
        query: lastUserMessage.content,
        currentSources: $contextSources,
        vectorDB: selectedVectorDB,
        expansionFactor: 1.5
      });

      contextSources.set(improvedSources);
    } catch (error) {
      console.error('Failed to improve context:', error);
    }
  }
</script>

<div class="document-chat h-full flex flex-col">
  <!-- Chat Header -->
  <Card variant="bordered" class="flex-shrink-0">
    <div class="p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <Button variant="outline" size="sm" on:click={() => dispatch('backToSearch')}>
            ← Back
          </Button>
          <div>
            <h2 class="text-lg font-semibold text-dark-text-primary">
              {session?.name || 'Knowledge Chat'}
            </h2>
            {#if session?.documentName}
              <div class="text-sm text-dark-text-secondary">
                Document: {session.documentName}
              </div>
            {/if}
          </div>
        </div>
        
        <div class="flex items-center gap-2">
          <Badge variant="info" size="sm">
            {session?.messages?.length || 0} messages
          </Badge>
          <Badge variant={$contextSources.length > 0 ? 'success' : 'warning'} size="sm">
            {$contextSources.length} sources
          </Badge>
          
          <div class="flex gap-1">
            <Button variant="outline" size="sm" on:click={clearChat}>
              Clear
            </Button>
            <Button variant="outline" size="sm" on:click={exportChat}>
              Export
            </Button>
            <Button variant="outline" size="sm" on:click={() => dispatch('newSession')}>
              New Chat
            </Button>
          </div>
        </div>
      </div>
    </div>
  </Card>

  <!-- Chat Settings Bar -->
  <div class="bg-dark-bg-secondary border-b border-dark-border p-3 flex items-center gap-4 flex-shrink-0">
    <div class="flex items-center gap-2 text-sm">
      <span class="text-dark-text-tertiary">Threshold:</span>
      <input 
        type="range" 
        min="0.3" 
        max="0.95" 
        step="0.05"
        bind:value={searchThreshold}
        class="w-20 h-2 bg-dark-bg-primary rounded-lg appearance-none cursor-pointer slider"
      />
      <span class="text-dark-text-secondary min-w-[3rem]">{searchThreshold.toFixed(2)}</span>
    </div>
    
    <div class="flex items-center gap-2 text-sm">
      <span class="text-dark-text-tertiary">Context:</span>
      <select 
        bind:value={maxContextChunks}
        class="px-2 py-1 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-xs focus:border-green-500 focus:outline-none"
      >
        <option value={3}>3 chunks</option>
        <option value={5}>5 chunks</option>
        <option value={10}>10 chunks</option>
        <option value={15}>15 chunks</option>
      </select>
    </div>
    
    <label class="flex items-center gap-2 text-sm">
      <input type="checkbox" bind:checked={streamingEnabled} class="checkbox" />
      <span class="text-dark-text-secondary">Streaming</span>
    </label>
    
    <div class="ml-auto">
      <Button variant="outline" size="xs" on:click={improveContext}>
        Enhance Context
      </Button>
    </div>
  </div>

  <!-- Messages Container -->
  <div 
    bind:this={chatContainer}
    class="flex-1 overflow-y-auto p-4 space-y-4 bg-dark-bg-primary"
  >
    {#if !session}
      <div class="text-center py-12">
        <svg class="w-16 h-16 text-dark-text-tertiary mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width={1} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
        </svg>
        <h3 class="text-lg font-medium text-dark-text-primary mb-2">No Chat Session Selected</h3>
        <p class="text-sm text-dark-text-secondary mb-4">Create a new session to start chatting with your knowledge base</p>
        <Button variant="primary" on:click={() => dispatch('newSession')}>
          Start New Chat
        </Button>
      </div>
    {:else}
      <!-- Welcome Message -->
      {#if session.messages.length === 0}
        <div class="welcome-message p-4 bg-dark-bg-secondary rounded-lg border border-dark-border">
          <div class="flex items-start gap-3">
            <div class="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center flex-shrink-0">
              <svg class="w-4 h-4 text-black" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
              </svg>
            </div>
            <div class="flex-1">
              <div class="text-sm font-medium text-green-400 mb-1">Knowledge Assistant Ready</div>
              <div class="text-sm text-dark-text-secondary mb-3">
                Ask questions about {session.documentName || 'your knowledge base'}. I'll search through {knowledgeStats.totalDocuments.toLocaleString()} documents and {knowledgeStats.totalVectors.toLocaleString()} knowledge vectors to provide accurate answers with source citations.
              </div>
              <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                {#each ['What are the main topics covered?', 'Summarize key findings', 'Find related documents', 'Extract important entities'] as suggestion}
                  <button 
                    class="text-left p-2 text-xs bg-dark-bg-tertiary rounded border border-dark-border hover:border-green-500 text-dark-text-secondary hover:text-dark-text-primary transition-colors"
                    on:click={() => {
                      messageInput = suggestion;
                      sendMessage();
                    }}
                  >
                    {suggestion}
                  </button>
                {/each}
              </div>
            </div>
          </div>
        </div>
      {/if}

      <!-- Chat Messages -->
      {#each session.messages as message}
        <div class="message flex items-start gap-3 {message.role === 'user' ? 'flex-row-reverse' : ''}">
          <!-- Avatar -->
          <div class="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 {
            message.role === 'user' ? 'bg-blue-600' : 
            message.role === 'assistant' ? 'bg-green-600' : 'bg-red-600'
          }">
            {#if message.role === 'user'}
              <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd" />
              </svg>
            {:else if message.role === 'assistant'}
              <svg class="w-4 h-4 text-black" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M3 3a1 1 0 000 2v8a2 2 0 002 2h2.586l-1.293 1.293a1 1 0 101.414 1.414L10 15.414l2.293 2.293a1 1 0 001.414-1.414L12.414 15H15a2 2 0 002-2V5a1 1 0 100-2H3zm11.707 4.707a1 1 0 00-1.414-1.414L10 9.586 8.707 8.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
              </svg>
            {:else}
              <svg class="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd" />
              </svg>
            {/if}
          </div>

          <!-- Message Content -->
          <div class="flex-1 max-w-3xl">
            <div class="message-bubble p-4 rounded-lg border {getMessageClass(message.role)}">
              <div class="message-header flex items-center justify-between mb-2">
                <div class="flex items-center gap-2">
                  <span class="text-xs font-medium text-dark-text-primary">
                    {message.role.toUpperCase()}
                  </span>
                  <span class="text-xs text-dark-text-tertiary">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </span>
                  {#if message.confidence}
                    <Badge variant="info" size="xs">
                      {(message.confidence * 100).toFixed(0)}% confidence
                    </Badge>
                  {/if}
                </div>
                
                <button 
                  class="text-xs text-dark-text-tertiary hover:text-dark-text-primary"
                  on:click={() => navigator.clipboard.writeText(message.content)}
                  title="Copy message"
                >
                  📋
                </button>
              </div>
              
              <div class="message-content text-sm text-dark-text-primary leading-relaxed">
                {@html formatMessage(message.content)}
              </div>

              <!-- Sources -->
              {#if message.sources && message.sources.length > 0}
                <div class="sources mt-3 pt-3 border-t border-dark-border">
                  <div class="text-xs font-medium text-dark-text-tertiary mb-2">
                    Sources ({message.sources.length}):
                  </div>
                  <div class="space-y-1">
                    {#each message.sources.slice(0, 3) as source}
                      <div class="source-item p-2 bg-dark-bg-tertiary rounded text-xs">
                        <div class="flex items-center justify-between mb-1">
                          <span class="font-medium text-dark-text-primary truncate">
                            {source.source}
                          </span>
                          <span class="text-green-400">
                            {(source.similarity * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div class="text-dark-text-secondary line-clamp-2">
                          {source.content.slice(0, 100)}...
                        </div>
                      </div>
                    {/each}
                    
                    {#if message.sources.length > 3}
                      <button class="text-xs text-cyan-400 hover:text-cyan-300">
                        +{message.sources.length - 3} more sources
                      </button>
                    {/if}
                  </div>
                </div>
              {/if}
            </div>
          </div>
        </div>
      {/each}

      <!-- Streaming Message -->
      {#if isLoading}
        <div class="message flex items-start gap-3">
          <div class="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center flex-shrink-0">
            <div class="w-3 h-3 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
          </div>
          <div class="flex-1 max-w-3xl">
            <div class="message-bubble p-4 rounded-lg border bg-green-600/20 border-green-500/30 mr-12">
              <div class="text-xs text-green-400 mb-2">Assistant is thinking...</div>
              <div class="text-sm text-dark-text-primary">
                {currentStreamedMessage || 'Searching knowledge base and generating response...'}
              </div>
            </div>
          </div>
        </div>
      {/if}
    {/if}
  </div>

  <!-- Chat Input -->
  <div class="flex-shrink-0 p-4 bg-dark-bg-secondary border-t border-dark-border">
    <form on:submit|preventDefault={sendMessage} class="flex gap-3">
      <div class="flex-1 relative">
        <input
          bind:value={messageInput}
          placeholder="Ask a question about your knowledge base..."
          class="w-full px-4 py-3 pr-12 bg-dark-bg-tertiary border border-dark-border rounded-lg text-dark-text-primary placeholder-dark-text-tertiary focus:border-green-500 focus:outline-none"
          disabled={isLoading}
        />
        
        {#if messageInput.trim()}
          <button
            type="button"
            class="absolute right-2 top-1/2 transform -translate-y-1/2 text-dark-text-tertiary hover:text-dark-text-primary"
            on:click={() => messageInput = ''}
          >
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        {/if}
      </div>
      
      <Button 
        type="submit" 
        variant="primary"
        disabled={isLoading || !messageInput.trim()}
        class="px-6"
      >
        {#if isLoading}
          <div class="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
        {:else}
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        {/if}
      </Button>
      
      {#if session?.messages?.length > 0}
        <Button variant="outline" on:click={regenerateResponse} disabled={isLoading}>
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
        </Button>
      {/if}
    </form>
    
    <!-- Context Sources Panel -->
    {#if $contextSources.length > 0}
      <div class="mt-3 pt-3 border-t border-dark-border">
        <div class="text-xs text-dark-text-tertiary mb-2">Active Context Sources:</div>
        <div class="flex flex-wrap gap-1">
          {#each $contextSources.slice(0, 6) as source}
            <button 
              class="px-2 py-1 text-xs bg-dark-bg-tertiary border border-dark-border rounded hover:border-green-500 text-dark-text-secondary hover:text-dark-text-primary transition-colors"
              title={source.content.slice(0, 200)}
            >
              {source.source.slice(0, 20)}... ({(source.similarity * 100).toFixed(0)}%)
            </button>
          {/each}
          {#if $contextSources.length > 6}
            <span class="text-xs text-dark-text-tertiary">+{$contextSources.length - 6} more</span>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .message-content :global(p) {
    margin-bottom: 0.5rem;
  }

  .message-content :global(pre) {
    background: #000;
    padding: 0.5rem;
    border-radius: 0.25rem;
    overflow-x: auto;
    margin: 0.5rem 0;
  }

  .message-content :global(code) {
    background: #1a1a1a;
    padding: 0.125rem 0.25rem;
    border-radius: 0.125rem;
  }

  .message-content :global(ul) {
    margin: 0.5rem 0;
    padding-left: 1rem;
  }

  .line-clamp-2 {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .slider::-webkit-slider-thumb {
    appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #00ff41;
    cursor: pointer;
    border: 1px solid #0a0a0a;
  }

  .checkbox {
    @apply w-3 h-3 rounded border border-dark-border;
    background: var(--dark-bg-tertiary, #0f0f0f);
    accent-color: #00ff41;
  }

  /* Scrollbar styling */
  ::-webkit-scrollbar {
    width: 6px;
  }
  
  ::-webkit-scrollbar-track {
    background: var(--dark-bg-tertiary, #0f0f0f);
  }
  
  ::-webkit-scrollbar-thumb {
    background: var(--dark-border, #00ff4133);
    border-radius: 3px;
  }
</style>