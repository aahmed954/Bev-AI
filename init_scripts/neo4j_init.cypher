// Neo4j Initialization Script for BEV OSINT Framework
// Creates indexes and constraints for optimal performance

// Entity constraints
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE;
CREATE CONSTRAINT person_email_unique IF NOT EXISTS FOR (p:Person) REQUIRE p.email IS UNIQUE;
CREATE CONSTRAINT domain_name_unique IF NOT EXISTS FOR (d:Domain) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT ip_address_unique IF NOT EXISTS FOR (i:IPAddress) REQUIRE i.address IS UNIQUE;

// Performance indexes
CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type);
CREATE INDEX person_name_idx IF NOT EXISTS FOR (p:Person) ON (p.name);
CREATE INDEX domain_tld_idx IF NOT EXISTS FOR (d:Domain) ON (d.tld);
CREATE INDEX ip_country_idx IF NOT EXISTS FOR (i:IPAddress) ON (i.country);

// Temporal indexes for investigations
CREATE INDEX investigation_created_idx IF NOT EXISTS FOR (i:Investigation) ON (i.created_at);
CREATE INDEX evidence_timestamp_idx IF NOT EXISTS FOR (e:Evidence) ON (e.timestamp);

// Relationship indexes
CREATE INDEX RELATIONSHIP_TYPE IF NOT EXISTS FOR ()-[r:CONNECTED_TO]-() ON (r.type);
CREATE INDEX RELATIONSHIP_CONFIDENCE IF NOT EXISTS FOR ()-[r]-() ON (r.confidence);