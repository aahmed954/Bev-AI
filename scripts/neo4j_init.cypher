// Neo4j Initialization Script for Bev Knowledge Graph

// Create indexes for better performance
CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX entity_timestamp IF NOT EXISTS FOR (e:Entity) ON (e.timestamp);
CREATE INDEX research_id IF NOT EXISTS FOR (r:Research) ON (r.task_id);
CREATE INDEX research_status IF NOT EXISTS FOR (r:Research) ON (r.status);
CREATE INDEX document_hash IF NOT EXISTS FOR (d:Document) ON (d.hash);
CREATE INDEX vulnerability_cve IF NOT EXISTS FOR (v:Vulnerability) ON (v.cve_id);
CREATE INDEX tool_name IF NOT EXISTS FOR (t:Tool) ON (t.name);
CREATE INDEX agent_name IF NOT EXISTS FOR (a:Agent) ON (a.name);

// Create constraints to ensure uniqueness
CREATE CONSTRAINT unique_entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT unique_research_task IF NOT EXISTS FOR (r:Research) REQUIRE r.task_id IS UNIQUE;
CREATE CONSTRAINT unique_document_hash IF NOT EXISTS FOR (d:Document) REQUIRE d.hash IS UNIQUE;
CREATE CONSTRAINT unique_agent_name IF NOT EXISTS FOR (a:Agent) REQUIRE a.name IS UNIQUE;

// Create initial agent nodes
MERGE (a1:Agent {name: 'ResearchCoordinator'})
SET a1.type = 'coordinator',
    a1.status = 'initialized',
    a1.created_at = datetime(),
    a1.capabilities = ['research', 'coordination', 'task_management'];

MERGE (a2:Agent {name: 'CodeOptimizer'})
SET a2.type = 'optimizer',
    a2.status = 'initialized',
    a2.created_at = datetime(),
    a2.capabilities = ['code_analysis', 'optimization', 'performance'];

MERGE (a3:Agent {name: 'MemoryManager'})
SET a3.type = 'memory',
    a3.status = 'initialized',
    a3.created_at = datetime(),
    a3.capabilities = ['storage', 'retrieval', 'caching'];

MERGE (a4:Agent {name: 'ToolCoordinator'})
SET a4.type = 'tools',
    a4.status = 'initialized',
    a4.created_at = datetime(),
    a4.capabilities = ['tool_selection', 'execution', 'orchestration'];

// Create agent relationships
MATCH (a1:Agent {name: 'ResearchCoordinator'}),
      (a2:Agent {name: 'CodeOptimizer'}),
      (a3:Agent {name: 'MemoryManager'}),
      (a4:Agent {name: 'ToolCoordinator'})
MERGE (a1)-[:COORDINATES]->(a2)
MERGE (a1)-[:COORDINATES]->(a3)
MERGE (a1)-[:COORDINATES]->(a4)
MERGE (a2)-[:STORES_IN]->(a3)
MERGE (a4)-[:STORES_IN]->(a3)
MERGE (a2)-[:USES]->(a4);

// Create initial tool nodes
MERGE (t1:Tool {name: 'market_research'})
SET t1.category = 'osint',
    t1.status = 'available',
    t1.created_at = datetime();

MERGE (t2:Tool {name: 'breach_analyzer'})
SET t2.category = 'osint',
    t2.status = 'available',
    t2.created_at = datetime();

MERGE (t3:Tool {name: 'watermark_remover'})
SET t3.category = 'enhancement',
    t3.status = 'available',
    t3.created_at = datetime();

MERGE (t4:Tool {name: 'metadata_scrubber'})
SET t4.category = 'enhancement',
    t4.status = 'available',
    t4.created_at = datetime();

MERGE (t5:Tool {name: 'document_analyzer'})
SET t5.category = 'analysis',
    t5.status = 'available',
    t5.created_at = datetime();

// Create research type nodes
MERGE (rt1:ResearchType {name: 'market_research'})
SET rt1.description = 'Market and competitive intelligence gathering';

MERGE (rt2:ResearchType {name: 'security_research'})
SET rt2.description = 'Security vulnerability and threat analysis';

MERGE (rt3:ResearchType {name: 'code_analysis'})
SET rt3.description = 'Code optimization and performance analysis';

// Create initial knowledge categories
MERGE (k1:KnowledgeCategory {name: 'technical'})
SET k1.description = 'Technical documentation and research';

MERGE (k2:KnowledgeCategory {name: 'intelligence'})
SET k2.description = 'OSINT and threat intelligence';

MERGE (k3:KnowledgeCategory {name: 'optimization'})
SET k3.description = 'Performance and optimization results';

// Create data source nodes
MERGE (ds1:DataSource {name: 'dark_web'})
SET ds1.type = 'external',
    ds1.risk_level = 'high',
    ds1.requires_tor = true;

MERGE (ds2:DataSource {name: 'public_apis'})
SET ds2.type = 'external',
    ds2.risk_level = 'low',
    ds2.requires_tor = false;

MERGE (ds3:DataSource {name: 'internal_db'})
SET ds3.type = 'internal',
    ds3.risk_level = 'none',
    ds3.requires_tor = false;

// Create initial metrics node
MERGE (m:Metrics {id: 'system_metrics'})
SET m.total_tasks = 0,
    m.completed_tasks = 0,
    m.failed_tasks = 0,
    m.total_documents = 0,
    m.total_entities = 0,
    m.last_updated = datetime();

// Create functions for common operations
// Note: These are Cypher query templates, not actual functions
// They should be used as templates in the application code

// Template: Create new research task
// MERGE (r:Research {task_id: $task_id})
// SET r.query = $query,
//     r.type = $type,
//     r.status = 'pending',
//     r.created_at = datetime(),
//     r.priority = $priority
// WITH r
// MATCH (rt:ResearchType {name: $type}),
//       (a:Agent {name: 'ResearchCoordinator'})
// MERGE (r)-[:TYPE_OF]->(rt)
// MERGE (a)-[:MANAGES]->(r)
// RETURN r;

// Template: Store document with relationships
// MERGE (d:Document {hash: $hash})
// SET d.title = $title,
//     d.content = $content,
//     d.source = $source,
//     d.created_at = datetime(),
//     d.size = $size
// WITH d
// MATCH (r:Research {task_id: $task_id}),
//       (ds:DataSource {name: $source_name})
// MERGE (r)-[:PRODUCED]->(d)
// MERGE (d)-[:SOURCED_FROM]->(ds)
// RETURN d;

// Template: Create entity with relationships
// MERGE (e:Entity {id: $entity_id})
// SET e.name = $name,
//     e.type = $type,
//     e.attributes = $attributes,
//     e.created_at = datetime()
// WITH e
// MATCH (d:Document {hash: $doc_hash})
// MERGE (d)-[:CONTAINS]->(e)
// RETURN e;

// Template: Link entities
// MATCH (e1:Entity {id: $entity1_id}),
//       (e2:Entity {id: $entity2_id})
// MERGE (e1)-[r:RELATED_TO {type: $relationship_type}]->(e2)
// SET r.confidence = $confidence,
//     r.discovered_at = datetime(),
//     r.source = $source
// RETURN r;

// Template: Update metrics
// MATCH (m:Metrics {id: 'system_metrics'})
// SET m.total_tasks = m.total_tasks + $task_delta,
//     m.completed_tasks = m.completed_tasks + $completed_delta,
//     m.failed_tasks = m.failed_tasks + $failed_delta,
//     m.total_documents = m.total_documents + $doc_delta,
//     m.total_entities = m.total_entities + $entity_delta,
//     m.last_updated = datetime()
// RETURN m;

// Template: Get research lineage
// MATCH path = (r:Research {task_id: $task_id})-[*]-(connected)
// RETURN path;

// Template: Find similar research
// MATCH (r1:Research {task_id: $task_id})-[:PRODUCED]->(d1:Document)-[:CONTAINS]->(e:Entity)
// WITH collect(distinct e.id) as entities1
// MATCH (r2:Research)-[:PRODUCED]->(d2:Document)-[:CONTAINS]->(e2:Entity)
// WHERE r2.task_id <> $task_id AND e2.id IN entities1
// WITH r2, count(distinct e2) as common_entities
// ORDER BY common_entities DESC
// LIMIT 10
// RETURN r2, common_entities;

// Success message
RETURN "Neo4j Knowledge Graph initialized successfully for Bev!" as message;
