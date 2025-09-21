<!-- Neo4j Cypher Query Builder & Graph Admin -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import cytoscape from 'cytoscape';
  import Card from '$lib/components/ui/Card.svelte';
  import Button from '$lib/components/ui/Button.svelte';
  import Badge from '$lib/components/ui/Badge.svelte';
  import { invoke } from '@tauri-apps/api/core';
  
  const dispatch = createEventDispatcher();
  
  export let database;
  
  let cypherQuery = '';
  let queryResults = [];
  let graphContainer: HTMLElement;
  let cy: any;
  let isExecuting = false;
  let nodeLabels = [];
  let relationshipTypes = [];
  let graphStats = { nodes: 0, relationships: 0, labels: 0 };
  
  const sampleQueries = [
    'MATCH (n) RETURN n LIMIT 25',
    'MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 10',
    'CALL db.labels()',
    'CALL db.relationshipTypes()',
    'MATCH (n) RETURN DISTINCT labels(n), count(n)',
    'CALL apoc.meta.graph()',
    'MATCH (n) WITH n, rand() AS r ORDER BY r RETURN n LIMIT 10'
  ];

  onMount(() => {
    initializeGraph();
    loadMetadata();
  });

  function initializeGraph() {
    if (!graphContainer) return;
    
    cy = cytoscape({
      container: graphContainer,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'label': 'data(label)',
            'color': '#ffffff',
            'text-valign': 'center',
            'text-halign': 'center',
            'width': 'data(size)',
            'height': 'data(size)',
            'font-size': '10px',
            'text-outline-width': 2,
            'text-outline-color': '#000000'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 2,
            'line-color': '#666666',
            'target-arrow-color': '#666666',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'label': 'data(type)',
            'font-size': '8px',
            'color': '#cccccc',
            'text-background-color': '#000000',
            'text-background-opacity': 0.7
          }
        }
      ],
      layout: { name: 'cose', animate: true },
      wheelSensitivity: 0.2
    });
  }

  async function loadMetadata() {
    try {
      // Load node labels
      const labelsResult = await invoke('execute_database_query', {
        databaseId: 'neo4j',
        query: 'CALL db.labels()'
      });
      nodeLabels = labelsResult.results?.map(r => r.label) || [];

      // Load relationship types
      const relsResult = await invoke('execute_database_query', {
        databaseId: 'neo4j',
        query: 'CALL db.relationshipTypes()'
      });
      relationshipTypes = relsResult.results?.map(r => r.relationshipType) || [];

      // Load graph stats
      const statsResult = await invoke('execute_database_query', {
        databaseId: 'neo4j',
        query: 'MATCH (n) OPTIONAL MATCH (n)-[r]->() RETURN count(DISTINCT n) as nodes, count(r) as relationships, count(DISTINCT labels(n)) as labels'
      });
      
      if (statsResult.results?.length > 0) {
        graphStats = statsResult.results[0];
      }
    } catch (error) {
      console.error('Failed to load Neo4j metadata:', error);
    }
  }

  async function executeCypher() {
    if (!cypherQuery.trim() || isExecuting) return;
    
    isExecuting = true;

    try {
      const result = await invoke('execute_database_query', {
        databaseId: 'neo4j',
        query: cypherQuery
      });

      const queryResult = {
        id: `cypher_${Date.now()}`,
        database: 'neo4j',
        query: cypherQuery,
        results: result.results || [],
        rowCount: result.rowCount || 0,
        executionTime: result.executionTime || 0,
        timestamp: new Date().toISOString()
      };

      queryResults = [queryResult, ...queryResults.slice(0, 9)];
      dispatch('queryExecuted', queryResult);

      // Visualize graph results if applicable
      if (result.graph) {
        visualizeGraph(result.graph);
      }

    } catch (error) {
      console.error('Cypher execution failed:', error);
      const errorResult = {
        id: `cypher_${Date.now()}`,
        database: 'neo4j',
        query: cypherQuery,
        results: [],
        rowCount: 0,
        executionTime: 0,
        timestamp: new Date().toISOString(),
        error: error.message || 'Cypher execution failed'
      };
      
      queryResults = [errorResult, ...queryResults.slice(0, 9)];
    } finally {
      isExecuting = false;
    }
  }

  function visualizeGraph(graphData) {
    if (!cy || !graphData) return;

    const elements = [];
    
    // Add nodes
    graphData.nodes?.forEach(node => {
      elements.push({
        data: {
          id: node.id,
          label: node.labels?.[0] || node.id,
          color: getNodeColor(node.labels?.[0]),
          size: 30,
          properties: node.properties
        }
      });
    });

    // Add relationships
    graphData.relationships?.forEach(rel => {
      elements.push({
        data: {
          id: `rel_${rel.id}`,
          source: rel.startNode,
          target: rel.endNode,
          type: rel.type,
          properties: rel.properties
        }
      });
    });

    cy.elements().remove();
    cy.add(elements);
    cy.layout({ name: 'cose', animate: true }).run();
  }

  function getNodeColor(label) {
    const colors = ['#00ff41', '#00ccff', '#ff9500', '#ff00ff', '#ffff00', '#ff0066'];
    const hash = (label || '').split('').reduce((a, b) => {
      a = ((a << 5) - a) + b.charCodeAt(0);
      return a & a;
    }, 0);
    return colors[Math.abs(hash) % colors.length];
  }

  function buildCypherQuery(type) {
    switch (type) {
      case 'all_nodes':
        cypherQuery = 'MATCH (n) RETURN n LIMIT 25';
        break;
      case 'all_relationships':
        cypherQuery = 'MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 10';
        break;
      case 'node_counts':
        cypherQuery = 'MATCH (n) RETURN DISTINCT labels(n) as label, count(n) as count ORDER BY count DESC';
        break;
      case 'relationship_counts':
        cypherQuery = 'MATCH ()-[r]->() RETURN type(r) as relationship, count(r) as count ORDER BY count DESC';
        break;
    }
  }
</script>

<div class="neo4j-admin space-y-6">
  <!-- Header -->
  <Card variant="bordered">
    <div class="p-4">
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-3">
          <Button variant="outline" size="sm" on:click={() => dispatch('backToOverview')}>
            ← Back
          </Button>
          <span class="text-2xl">🕸️</span>
          <div>
            <h2 class="text-lg font-semibold text-dark-text-primary">Neo4j Graph Admin</h2>
            <div class="text-sm text-dark-text-secondary">
              {database?.host}:{database?.port} • {database?.version}
            </div>
          </div>
          <Badge variant={getStatusBadgeVariant(database?.status)}>
            {database?.status?.toUpperCase()}
          </Badge>
        </div>
        
        <div class="flex items-center gap-2 text-sm">
          <span class="text-dark-text-tertiary">
            {graphStats.nodes} nodes • {graphStats.relationships} edges • {graphStats.labels} labels
          </span>
        </div>
      </div>
    </div>
  </Card>

  <div class="grid grid-cols-1 xl:grid-cols-4 gap-6">
    <!-- Graph Visualization -->
    <div class="xl:col-span-3">
      <Card variant="bordered">
        <div class="p-4">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-md font-medium text-dark-text-primary">Graph Visualization</h3>
            <div class="flex gap-2">
              <Button variant="outline" size="sm" on:click={() => {
                if (cy) {
                  cy.fit();
                  cy.center();
                }
              }}>
                Fit Graph
              </Button>
              <Button variant="outline" size="sm" on:click={() => {
                if (cy) {
                  cy.elements().remove();
                }
              }}>
                Clear
              </Button>
            </div>
          </div>
          
          <div 
            bind:this={graphContainer}
            class="w-full h-80 bg-dark-bg-primary border border-dark-border rounded"
          ></div>
        </div>
      </Card>

      <!-- Cypher Query Editor -->
      <Card variant="bordered">
        <div class="p-4">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-md font-medium text-dark-text-primary">Cypher Query Editor</h3>
            <Button 
              variant="primary" 
              on:click={executeCypher}
              disabled={isExecuting || !cypherQuery.trim()}
            >
              {#if isExecuting}
                <div class="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin mr-2"></div>
                Executing...
              {:else}
                ▶ Execute
              {/if}
            </Button>
          </div>

          <textarea
            bind:value={cypherQuery}
            placeholder="MATCH (n) RETURN n LIMIT 25"
            class="w-full h-24 px-3 py-2 bg-dark-bg-primary border border-dark-border rounded text-dark-text-primary placeholder-dark-text-tertiary font-mono text-sm focus:border-green-500 focus:outline-none resize-none"
            disabled={isExecuting}
          ></textarea>
        </div>
      </Card>
    </div>

    <!-- Sidebar -->
    <div class="space-y-4">
      <!-- Quick Queries -->
      <Card variant="bordered">
        <div class="p-4">
          <h4 class="text-sm font-medium text-dark-text-primary mb-3">Quick Queries</h4>
          <div class="space-y-2">
            <Button variant="outline" size="sm" fullWidth on:click={() => buildCypherQuery('all_nodes')}>
              All Nodes
            </Button>
            <Button variant="outline" size="sm" fullWidth on:click={() => buildCypherQuery('all_relationships')}>
              All Relationships
            </Button>
            <Button variant="outline" size="sm" fullWidth on:click={() => buildCypherQuery('node_counts')}>
              Node Counts
            </Button>
            <Button variant="outline" size="sm" fullWidth on:click={() => buildCypherQuery('relationship_counts')}>
              Relationship Counts
            </Button>
          </div>
        </div>
      </Card>

      <!-- Node Labels -->
      <Card variant="bordered">
        <div class="p-4">
          <h4 class="text-sm font-medium text-dark-text-primary mb-3">Node Labels</h4>
          <div class="space-y-1 max-h-32 overflow-y-auto">
            {#each nodeLabels as label}
              <button 
                class="w-full text-left p-1 text-xs bg-dark-bg-tertiary rounded border border-dark-border hover:border-green-500 text-dark-text-secondary hover:text-dark-text-primary transition-colors"
                on:click={() => cypherQuery = `MATCH (n:${label}) RETURN n LIMIT 10`}
              >
                {label}
              </button>
            {/each}
          </div>
        </div>
      </Card>

      <!-- Relationship Types -->
      <Card variant="bordered">
        <div class="p-4">
          <h4 class="text-sm font-medium text-dark-text-primary mb-3">Relationships</h4>
          <div class="space-y-1 max-h-32 overflow-y-auto">
            {#each relationshipTypes as relType}
              <button 
                class="w-full text-left p-1 text-xs bg-dark-bg-tertiary rounded border border-dark-border hover:border-green-500 text-dark-text-secondary hover:text-dark-text-primary transition-colors"
                on:click={() => cypherQuery = `MATCH ()-[r:${relType}]->() RETURN r LIMIT 10`}
              >
                {relType}
              </button>
            {/each}
          </div>
        </div>
      </Card>
    </div>
  </div>
</div>