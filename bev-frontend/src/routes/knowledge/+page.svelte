<!-- BEV Knowledge & RAG Intelligence Platform -->
<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import Panel from '$lib/components/ui/Panel.svelte';
  import KnowledgeSearch from '$lib/components/knowledge/KnowledgeSearch.svelte';
  import DocumentChat from '$lib/components/knowledge/DocumentChat.svelte';
  import KnowledgeGraph from '$lib/components/knowledge/KnowledgeGraph.svelte';
  import VectorSearchResults from '$lib/components/knowledge/VectorSearchResults.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  interface KnowledgeStats {
    totalDocuments: number;
    totalVectors: number;
    totalQueries: number;
    avgSimilarity: number;
    knowledgeSize: number;
    lastUpdate: string;
  }

  interface SearchResult {
    id: string;
    content: string;
    metadata: Record<string, any>;
    similarity: number;
    source: string;
    timestamp: string;
    type: 'document' | 'fragment' | 'entity';
  }

  interface KnowledgeNode {
    id: string;
    label: string;
    type: 'document' | 'entity' | 'concept' | 'relationship';
    properties: Record<string, any>;
    connections: string[];
    embedding?: number[];
  }

  interface ChatSession {
    id: string;
    name: string;
    documentId?: string;
    documentName?: string;
    messages: ChatMessage[];
    context: SearchResult[];
    created: string;
    lastActive: string;
  }

  interface ChatMessage {
    id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    timestamp: string;
    sources?: SearchResult[];
    confidence?: number;
  }

  let currentView: 'search' | 'chat' | 'graph' | 'upload' = 'search';
  let knowledgeStats = writable<KnowledgeStats>({
    totalDocuments: 0,
    totalVectors: 0,
    totalQueries: 0,
    avgSimilarity: 0,
    knowledgeSize: 0,
    lastUpdate: new Date().toISOString()
  });

  let searchResults = writable<SearchResult[]>([]);
  let selectedResults = writable<SearchResult[]>([]);
  let knowledgeNodes = writable<KnowledgeNode[]>([]);
  let chatSessions = writable<ChatSession[]>([]);
  let currentChatSession = writable<ChatSession | null>(null);

  let websocket: WebSocket | null = null;
  let connectionStatus: 'connected' | 'connecting' | 'disconnected' = 'disconnected';
  let lastQuery = '';
  let isProcessing = false;

  // Knowledge base management
  let uploadProgress = 0;
  let indexingStatus = 'idle'; // 'idle', 'indexing', 'completed', 'error'
  let vectorDatabases = ['qdrant', 'weaviate'];
  let selectedVectorDB = 'qdrant';

  onMount(() => {
    loadKnowledgeStats();
    connectWebSocket();
    loadChatSessions();
  });

  onDestroy(() => {
    if (websocket) {
      websocket.close();
    }
  });

  async function loadKnowledgeStats() {
    try {
      const stats = await invoke('get_knowledge_stats');
      knowledgeStats.set(stats);
    } catch (error) {
      console.error('Failed to load knowledge stats:', error);
      // Use mock data for development
      knowledgeStats.set({
        totalDocuments: 1247,
        totalVectors: 156823,
        totalQueries: 5891,
        avgSimilarity: 0.742,
        knowledgeSize: 2.3, // GB
        lastUpdate: new Date().toISOString()
      });
    }
  }

  function connectWebSocket() {
    try {
      connectionStatus = 'connecting';
      websocket = new WebSocket('ws://localhost:3021/knowledge-stream');

      websocket.onopen = () => {
        connectionStatus = 'connected';
        console.log('Connected to knowledge processing stream');
      };

      websocket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleKnowledgeUpdate(data);
      };

      websocket.onerror = (error) => {
        console.error('Knowledge WebSocket error:', error);
        connectionStatus = 'disconnected';
      };

      websocket.onclose = () => {
        connectionStatus = 'disconnected';
        setTimeout(connectWebSocket, 3000);
      };
    } catch (error) {
      console.error('Failed to connect knowledge WebSocket:', error);
      connectionStatus = 'disconnected';
    }
  }

  function handleKnowledgeUpdate(data: any) {
    switch (data.type) {
      case 'search_results':
        searchResults.set(data.results);
        break;
      case 'knowledge_updated':
        loadKnowledgeStats();
        break;
      case 'chat_response':
        updateChatSession(data.sessionId, data.message);
        break;
      case 'graph_update':
        knowledgeNodes.set(data.nodes);
        break;
      case 'indexing_progress':
        uploadProgress = data.progress;
        indexingStatus = data.status;
        break;
    }
  }

  function loadChatSessions() {
    const saved = localStorage.getItem('bev-knowledge-sessions');
    if (saved) {
      try {
        chatSessions.set(JSON.parse(saved));
      } catch (e) {
        console.warn('Failed to load chat sessions:', e);
      }
    }
  }

  function saveChatSessions() {
    chatSessions.subscribe(sessions => {
      localStorage.setItem('bev-knowledge-sessions', JSON.stringify(sessions.slice(0, 50)));
    })();
  }

  function createNewChatSession(documentId?: string, documentName?: string) {
    const session: ChatSession = {
      id: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: documentName ? `Chat about ${documentName}` : `Knowledge Session ${new Date().toLocaleTimeString()}`,
      documentId,
      documentName,
      messages: [],
      context: [],
      created: new Date().toISOString(),
      lastActive: new Date().toISOString()
    };

    chatSessions.update(sessions => [session, ...sessions]);
    currentChatSession.set(session);
    currentView = 'chat';
    saveChatSessions();
  }

  function updateChatSession(sessionId: string, message: ChatMessage) {
    chatSessions.update(sessions => 
      sessions.map(session => {
        if (session.id === sessionId) {
          return {
            ...session,
            messages: [...session.messages, message],
            lastActive: new Date().toISOString()
          };
        }
        return session;
      })
    );
    saveChatSessions();
  }

  async function performKnowledgeSearch(query: string, options: any = {}) {
    if (!query.trim()) return;
    
    isProcessing = true;
    lastQuery = query;

    try {
      const results = await invoke('search_knowledge', {
        query,
        similarityThreshold: options.threshold || 0.7,
        maxResults: options.maxResults || 50,
        vectorDB: selectedVectorDB,
        includeMetadata: true
      });

      searchResults.set(results);
      currentView = 'search';
    } catch (error) {
      console.error('Knowledge search failed:', error);
    } finally {
      isProcessing = false;
    }
  }

  async function uploadToKnowledgeBase(files: File[]) {
    uploadProgress = 0;
    indexingStatus = 'indexing';

    for (const file of files) {
      try {
        const reader = new FileReader();
        reader.onload = async () => {
          try {
            await invoke('upload_to_knowledge_base', {
              filename: file.name,
              fileData: reader.result,
              vectorDB: selectedVectorDB,
              extractEntities: true,
              buildGraph: true
            });
          } catch (error) {
            console.error('Failed to upload to knowledge base:', error);
            indexingStatus = 'error';
          }
        };
        reader.readAsDataURL(file);
      } catch (error) {
        console.error('File processing failed:', error);
        indexingStatus = 'error';
      }
    }
  }

  function clearKnowledgeBase() {
    if (confirm('Are you sure? This will clear the entire knowledge base.')) {
      invoke('clear_knowledge_base', { vectorDB: selectedVectorDB })
        .then(() => {
          loadKnowledgeStats();
          searchResults.set([]);
          knowledgeNodes.set([]);
        })
        .catch(error => console.error('Failed to clear knowledge base:', error));
    }
  }

  async function exportKnowledgeBase() {
    try {
      const exportData = await invoke('export_knowledge_base', { 
        vectorDB: selectedVectorDB,
        includeVectors: false // Exclude vectors for smaller export
      });

      const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
        type: 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `bev-knowledge-export-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Failed to export knowledge base:', error);
    }
  }
</script>

<div class="knowledge-platform min-h-screen bg-dark-bg-primary text-dark-text-primary">
  <!-- Header -->
  <div class="border-b border-dark-border bg-dark-bg-secondary">
    <div class="container mx-auto px-6 py-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-4">
          <h1 class="text-2xl font-bold text-green-400">KNOWLEDGE & RAG PLATFORM</h1>
          <Badge variant={connectionStatus === 'connected' ? 'success' : 'danger'}>
            {connectionStatus.toUpperCase()}
          </Badge>
          <Badge variant="info" size="sm">
            {selectedVectorDB.toUpperCase()}
          </Badge>
        </div>
        
        <div class="flex items-center gap-3">
          <!-- View Toggle -->
          <div class="flex bg-dark-bg-tertiary rounded-lg p-1">
            {#each ['search', 'chat', 'graph', 'upload'] as view}
              <button
                class="px-3 py-1 text-xs font-medium rounded transition-colors {
                  currentView === view 
                    ? 'bg-green-600 text-black' 
                    : 'text-dark-text-secondary hover:text-dark-text-primary'
                }"
                on:click={() => currentView = view}
              >
                {view.toUpperCase()}
              </button>
            {/each}
          </div>
          
          <!-- Vector DB Selector -->
          <select 
            bind:value={selectedVectorDB}
            class="px-3 py-2 bg-dark-bg-tertiary border border-dark-border rounded text-dark-text-primary text-sm focus:border-green-500 focus:outline-none"
          >
            <option value="qdrant">Qdrant (Primary)</option>
            <option value="weaviate">Weaviate (Secondary)</option>
          </select>
          
          <Button variant="outline" size="sm" on:click={exportKnowledgeBase}>
            Export KB
          </Button>
        </div>
      </div>
    </div>
  </div>

  <!-- Stats Bar -->
  <div class="bg-dark-bg-secondary border-b border-dark-border">
    <div class="container mx-auto px-6 py-3">
      <div class="grid grid-cols-2 md:grid-cols-6 gap-4 text-center">
        <div>
          <div class="text-xs text-dark-text-tertiary">DOCUMENTS</div>
          <div class="text-lg font-bold text-green-400">{$knowledgeStats.totalDocuments.toLocaleString()}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">VECTORS</div>
          <div class="text-lg font-bold text-cyan-400">{$knowledgeStats.totalVectors.toLocaleString()}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">QUERIES</div>
          <div class="text-lg font-bold text-purple-400">{$knowledgeStats.totalQueries.toLocaleString()}</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">AVG SIMILARITY</div>
          <div class="text-lg font-bold text-yellow-400">{($knowledgeStats.avgSimilarity * 100).toFixed(1)}%</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">KB SIZE</div>
          <div class="text-lg font-bold text-blue-400">{$knowledgeStats.knowledgeSize.toFixed(1)} GB</div>
        </div>
        <div>
          <div class="text-xs text-dark-text-tertiary">STATUS</div>
          <div class="text-lg font-bold {indexingStatus === 'indexing' ? 'text-yellow-400' : indexingStatus === 'error' ? 'text-red-400' : 'text-green-400'}">
            {indexingStatus.toUpperCase()}
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Main Content -->
  <div class="container mx-auto px-6 py-6">
    {#if currentView === 'search'}
      <div class="grid grid-cols-1 xl:grid-cols-4 gap-6">
        <!-- Search Interface -->
        <div class="xl:col-span-3">
          <KnowledgeSearch 
            {isProcessing}
            {lastQuery}
            {selectedVectorDB}
            on:search={(e) => performKnowledgeSearch(e.detail.query, e.detail.options)}
            on:resultSelected={(e) => selectedResults.update(results => [...results, e.detail])}
          />
          
          <!-- Search Results -->
          {#if $searchResults.length > 0}
            <div class="mt-6">
              <VectorSearchResults 
                results={$searchResults}
                selectedResults={$selectedResults}
                on:createChat={(e) => createNewChatSession(e.detail.documentId, e.detail.documentName)}
                on:addToGraph={(e) => dispatch('addToGraph', e.detail)}
                on:resultToggled={(e) => {
                  if (e.detail.selected) {
                    selectedResults.update(results => [...results, e.detail.result]);
                  } else {
                    selectedResults.update(results => results.filter(r => r.id !== e.detail.result.id));
                  }
                }}
              />
            </div>
          {/if}
        </div>

        <!-- Sidebar -->
        <div class="space-y-4">
          <!-- Quick Actions -->
          <Card variant="bordered">
            <div class="p-4">
              <h3 class="text-md font-medium mb-3 text-dark-text-primary">Quick Actions</h3>
              <div class="space-y-2">
                <Button variant="outline" fullWidth on:click={() => createNewChatSession()}>
                  <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                  New Chat Session
                </Button>
                
                <Button variant="outline" fullWidth on:click={() => currentView = 'graph'}>
                  <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                  </svg>
                  Knowledge Graph
                </Button>
                
                <Button variant="outline" fullWidth on:click={() => currentView = 'upload'}>
                  <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  Upload Documents
                </Button>
                
                <Button variant="outline" fullWidth on:click={clearKnowledgeBase}>
                  <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  Clear KB
                </Button>
              </div>
            </div>
          </Card>

          <!-- Recent Chat Sessions -->
          <Card variant="bordered">
            <div class="p-4">
              <h3 class="text-md font-medium mb-3 text-dark-text-primary">Recent Sessions</h3>
              <div class="space-y-2 max-h-64 overflow-y-auto">
                {#each $chatSessions.slice(0, 8) as session}
                  <div 
                    class="p-2 bg-dark-bg-tertiary rounded border border-dark-border cursor-pointer hover:border-green-500 transition-colors"
                    on:click={() => {
                      currentChatSession.set(session);
                      currentView = 'chat';
                    }}
                  >
                    <div class="text-sm font-medium text-dark-text-primary truncate">
                      {session.name}
                    </div>
                    <div class="text-xs text-dark-text-tertiary">
                      {session.messages.length} messages • {new Date(session.lastActive).toLocaleDateString()}
                    </div>
                  </div>
                {/each}
                
                {#if $chatSessions.length === 0}
                  <div class="text-center py-4 text-dark-text-tertiary">
                    <p class="text-xs">No chat sessions yet</p>
                  </div>
                {/if}
              </div>
            </div>
          </Card>

          <!-- Selected Results -->
          {#if $selectedResults.length > 0}
            <Card variant="bordered">
              <div class="p-4">
                <div class="flex items-center justify-between mb-3">
                  <h3 class="text-md font-medium text-dark-text-primary">Selected ({$selectedResults.length})</h3>
                  <Button variant="outline" size="xs" on:click={() => selectedResults.set([])}>
                    Clear
                  </Button>
                </div>
                <div class="space-y-2 max-h-48 overflow-y-auto">
                  {#each $selectedResults as result}
                    <div class="p-2 bg-dark-bg-tertiary rounded border border-dark-border">
                      <div class="text-xs font-medium text-dark-text-primary truncate">
                        {result.source}
                      </div>
                      <div class="text-xs text-dark-text-secondary">
                        {(result.similarity * 100).toFixed(1)}% match
                      </div>
                    </div>
                  {/each}
                </div>
                
                <div class="mt-3 pt-3 border-t border-dark-border">
                  <Button variant="primary" size="sm" fullWidth on:click={() => {
                    const documentIds = $selectedResults.map(r => r.id);
                    createNewChatSession(documentIds[0], `${documentIds.length} documents`);
                  }}>
                    Chat with Selected
                  </Button>
                </div>
              </div>
            </Card>
          {/if}
        </div>
      </div>
    {/if}

    {#if currentView === 'chat'}
      <DocumentChat 
        session={$currentChatSession}
        knowledgeStats={$knowledgeStats}
        {selectedVectorDB}
        on:sessionUpdated={(e) => updateChatSession(e.detail.sessionId, e.detail.message)}
        on:backToSearch={() => currentView = 'search'}
        on:newSession={() => createNewChatSession()}
      />
    {/if}

    {#if currentView === 'graph'}
      <KnowledgeGraph 
        nodes={$knowledgeNodes}
        selectedResults={$selectedResults}
        {selectedVectorDB}
        on:nodeSelected={(e) => performKnowledgeSearch(e.detail.label, { threshold: 0.8 })}
        on:backToSearch={() => currentView = 'search'}
      />
    {/if}

    {#if currentView === 'upload'}
      <Card variant="bordered">
        <div class="p-6">
          <h2 class="text-lg font-semibold mb-4 text-dark-text-primary">Upload to Knowledge Base</h2>
          
          <!-- Upload Interface -->
          <div class="upload-zone border-2 border-dashed border-dark-border rounded-lg p-8 text-center mb-6"
               on:dragover|preventDefault={() => {}}
               on:drop|preventDefault={(e) => {
                 const files = Array.from(e.dataTransfer.files);
                 uploadToKnowledgeBase(files);
               }}>
            <svg class="w-16 h-16 text-dark-text-tertiary mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width={1} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
            </svg>
            <h3 class="text-lg font-medium text-dark-text-primary mb-2">
              Drop documents to add to knowledge base
            </h3>
            <p class="text-sm text-dark-text-secondary mb-4">
              Supports PDF, TXT, MD, DOCX files up to 100MB each
            </p>
            
            <input 
              type="file" 
              multiple 
              accept=".pdf,.txt,.md,.docx,.json"
              on:change={(e) => {
                const files = Array.from(e.target.files || []);
                uploadToKnowledgeBase(files);
              }}
              class="hidden" 
              id="knowledge-upload"
            />
            
            <Button variant="primary" on:click={() => document.getElementById('knowledge-upload')?.click()}>
              Select Files
            </Button>
          </div>

          <!-- Upload Progress -->
          {#if indexingStatus === 'indexing'}
            <div class="indexing-progress">
              <div class="flex items-center justify-between mb-2">
                <span class="text-sm text-dark-text-primary">Indexing documents...</span>
                <span class="text-sm text-dark-text-secondary">{uploadProgress}%</span>
              </div>
              <div class="w-full bg-dark-bg-primary rounded-full h-2">
                <div 
                  class="bg-green-600 h-2 rounded-full transition-all duration-300"
                  style="width: {uploadProgress}%"
                ></div>
              </div>
            </div>
          {/if}

          <!-- Knowledge Base Management -->
          <div class="kb-management mt-6 pt-6 border-t border-dark-border">
            <h3 class="text-md font-medium mb-3 text-dark-text-primary">Knowledge Base Management</h3>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
              <Button variant="outline" fullWidth>
                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Rebuild Index
              </Button>
              
              <Button variant="outline" fullWidth on:click={() => {
                invoke('optimize_knowledge_base', { vectorDB: selectedVectorDB })
                  .then(() => loadKnowledgeStats())
                  .catch(error => console.error('Optimization failed:', error));
              }}>
                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Optimize
              </Button>
              
              <Button variant="outline" fullWidth on:click={() => {
                invoke('backup_knowledge_base', { vectorDB: selectedVectorDB })
                  .then(() => alert('Backup created successfully'))
                  .catch(error => console.error('Backup failed:', error));
              }}>
                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width={2} d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                Backup
              </Button>
            </div>
          </div>
        </div>
      </div>
    {/if}

    {#if currentView === 'chat'}
      <DocumentChat 
        session={$currentChatSession}
        knowledgeStats={$knowledgeStats}
        {selectedVectorDB}
        on:sessionUpdated={(e) => updateChatSession(e.detail.sessionId, e.detail.message)}
        on:backToSearch={() => currentView = 'search'}
        on:newSession={() => createNewChatSession()}
      />
    {/if}

    {#if currentView === 'graph'}
      <KnowledgeGraph 
        nodes={$knowledgeNodes}
        selectedResults={$selectedResults}
        {selectedVectorDB}
        on:nodeSelected={(e) => performKnowledgeSearch(e.detail.label, { threshold: 0.8 })}
        on:backToSearch={() => currentView = 'search'}
      />
    {/if}
  </div>
</div>

<style>
  .upload-zone:hover {
    @apply border-green-500;
    background: rgba(0, 255, 65, 0.05);
  }

  /* Ensure proper dark theme */
  :global(.dark-bg-primary) { background-color: #0a0a0a; }
  :global(.dark-bg-secondary) { background-color: #1a1a1a; }
  :global(.dark-bg-tertiary) { background-color: #0f0f0f; }
  :global(.dark-text-primary) { color: #00ff41; }
  :global(.dark-text-secondary) { color: #00ff4199; }
  :global(.dark-text-tertiary) { color: #00ff4166; }
  :global(.dark-border) { border-color: #00ff4133; }
</style>