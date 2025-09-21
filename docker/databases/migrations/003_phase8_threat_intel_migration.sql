-- Migration 003: Phase 8 Threat Intelligence Schema
-- Safe migration script for threat intelligence tables
-- Version: 1.0.0
-- Date: 2024-09-19

-- Check if migration has already been applied
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM migration_log.migration_history WHERE migration_name = 'phase8_threat_intelligence') THEN
        RAISE NOTICE 'Migration phase8_threat_intelligence has already been applied. Skipping.';
        RETURN;
    END IF;
END $$;

-- Begin transaction for atomic migration
BEGIN;

-- Create threat intelligence schema if not exists
CREATE SCHEMA IF NOT EXISTS threat_intel;

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA threat_intel TO swarm_admin;
GRANT CREATE ON SCHEMA threat_intel TO swarm_admin;

-- Create threat indicators table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'threat_intel'
                   AND table_name = 'threat_indicators') THEN

        CREATE TABLE threat_intel.threat_indicators (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            indicator_value TEXT NOT NULL,
            indicator_type TEXT NOT NULL CHECK (indicator_type IN (
                'ip_address', 'domain', 'url', 'file_hash', 'email', 'phone', 'bitcoin_address',
                'tor_address', 'telegram_id', 'user_agent', 'certificate_hash', 'registry_key',
                'mutex', 'pdb_path', 'campaign_id', 'malware_family', 'attack_pattern'
            )),
            threat_type TEXT[] NOT NULL,
            malware_families TEXT[],
            threat_actors TEXT[],
            campaigns TEXT[],
            confidence confidence_level NOT NULL DEFAULT 'medium',
            severity severity_level NOT NULL DEFAULT 'medium',
            priority INTEGER CHECK (priority BETWEEN 1 AND 10) DEFAULT 5,
            first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expiry_date TIMESTAMPTZ,
            is_active BOOLEAN DEFAULT TRUE,
            attributed_actor TEXT,
            actor_confidence DECIMAL(3,2),
            attribution_methods TEXT[],
            attribution_evidence JSONB,
            indicator_context JSONB,
            technical_details JSONB,
            kill_chain_phases TEXT[],
            mitre_techniques TEXT[],
            related_indicators UUID[],
            parent_campaign UUID,
            related_samples TEXT[],
            detection_rules JSONB,
            false_positive_rate DECIMAL(5,4),
            detection_confidence DECIMAL(3,2),
            source_reliability TEXT CHECK (source_reliability IN ('A', 'B', 'C', 'D', 'E', 'F')),
            information_credibility TEXT CHECK (information_credibility IN ('1', '2', '3', '4', '5', '6')),
            data_sources TEXT[] NOT NULL,
            collection_methods TEXT[],
            tlp_marking TEXT CHECK (tlp_marking IN ('WHITE', 'GREEN', 'AMBER', 'RED')) DEFAULT 'AMBER',
            sharing_restrictions JSONB,
            distribution_list TEXT[],
            indicator_embedding vector(768),
            analyst_id TEXT,
            analysis_notes TEXT,
            tags TEXT[],
            raw_intelligence JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(indicator_value, indicator_type)
        ) PARTITION BY RANGE (created_at);

        RAISE NOTICE 'Created table threat_intel.threat_indicators';
    END IF;
END $$;

-- Create partitions for threat indicators
DO $$
DECLARE
    current_month DATE := DATE_TRUNC('month', NOW());
    next_month DATE := DATE_TRUNC('month', NOW() + INTERVAL '1 month');
BEGIN
    PERFORM create_monthly_partition('threat_intel.threat_indicators', current_month);
    PERFORM create_monthly_partition('threat_intel.threat_indicators', next_month);
    RAISE NOTICE 'Created partitions for threat_intel.threat_indicators';
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Error creating partitions for threat_indicators: %', SQLERRM;
END $$;

-- Create threat actors table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'threat_intel'
                   AND table_name = 'threat_actors') THEN

        CREATE TABLE threat_intel.threat_actors (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            actor_name TEXT NOT NULL,
            actor_aliases TEXT[],
            actor_type TEXT CHECK (actor_type IN ('apt', 'cybercriminal', 'hacktivist', 'nation_state', 'insider', 'script_kiddie', 'unknown')),
            suspected_nationality TEXT,
            suspected_location TEXT,
            attribution_confidence DECIMAL(3,2),
            attribution_evidence JSONB,
            primary_motivation TEXT CHECK (primary_motivation IN ('financial', 'espionage', 'ideology', 'revenge', 'fame', 'unknown')),
            secondary_motivations TEXT[],
            target_sectors TEXT[],
            target_geographies TEXT[],
            sophistication_level TEXT CHECK (sophistication_level IN ('low', 'medium', 'high', 'expert')),
            resources_level TEXT CHECK (resources_level IN ('individual', 'small_group', 'organized_group', 'state_sponsored')),
            technical_capabilities JSONB,
            operational_capabilities JSONB,
            first_observed TIMESTAMPTZ,
            last_observed TIMESTAMPTZ,
            activity_level TEXT CHECK (activity_level IN ('dormant', 'low', 'medium', 'high', 'very_high')),
            activity_patterns JSONB,
            preferred_malware_families TEXT[],
            common_attack_vectors TEXT[],
            mitre_techniques TEXT[],
            signature_behaviors JSONB,
            known_infrastructure JSONB,
            infrastructure_patterns JSONB,
            communication_methods TEXT[],
            known_campaigns TEXT[],
            operation_names TEXT[],
            campaign_patterns JSONB,
            threat_severity severity_level DEFAULT 'medium',
            current_threat_level INTEGER CHECK (current_threat_level BETWEEN 1 AND 5),
            trend_analysis JSONB,
            intelligence_requirements TEXT[],
            knowledge_gaps TEXT[],
            assessment_confidence DECIMAL(3,2),
            associated_actors UUID[],
            allied_groups TEXT[],
            competitor_groups TEXT[],
            actor_profile_embedding vector(768),
            analyst_id TEXT,
            last_analysis_update TIMESTAMPTZ,
            intelligence_sources TEXT[],
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(actor_name)
        );

        RAISE NOTICE 'Created table threat_intel.threat_actors';
    END IF;
END $$;

-- Create security events table
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'threat_intel'
                   AND table_name = 'security_events') THEN

        CREATE TABLE threat_intel.security_events (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            event_id TEXT NOT NULL,
            event_type TEXT CHECK (event_type IN ('detection', 'incident', 'alert', 'observation', 'report')),
            event_source TEXT NOT NULL,
            event_category TEXT CHECK (event_category IN ('malware', 'network_intrusion', 'data_exfiltration', 'credential_theft', 'reconnaissance', 'lateral_movement', 'persistence', 'command_control')),
            severity severity_level NOT NULL DEFAULT 'medium',
            confidence confidence_level NOT NULL DEFAULT 'medium',
            event_timestamp TIMESTAMPTZ NOT NULL,
            detection_timestamp TIMESTAMPTZ DEFAULT NOW(),
            first_seen TIMESTAMPTZ,
            last_seen TIMESTAMPTZ,
            event_description TEXT NOT NULL,
            technical_details JSONB,
            affected_systems TEXT[],
            affected_networks TEXT[],
            related_indicators UUID[],
            related_actors UUID[],
            related_campaigns UUID[],
            threat_context JSONB,
            source_ip INET,
            destination_ip INET,
            source_port INTEGER,
            destination_port INTEGER,
            protocol TEXT,
            network_artifacts JSONB,
            file_hashes TEXT[],
            process_names TEXT[],
            registry_keys TEXT[],
            command_lines TEXT[],
            detection_rule_id TEXT,
            detection_method TEXT,
            signature_id TEXT,
            false_positive_likelihood DECIMAL(3,2),
            containment_actions TEXT[],
            mitigation_steps TEXT[],
            response_status TEXT CHECK (response_status IN ('new', 'investigating', 'contained', 'mitigated', 'resolved', 'false_positive')),
            analyst_notes TEXT,
            investigation_findings JSONB,
            recommended_actions TEXT[],
            external_ticket_id TEXT,
            siem_event_id TEXT,
            correlation_id TEXT,
            event_embedding vector(768),
            data_source data_source_type NOT NULL,
            collection_method TEXT,
            analyst_id TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(event_id, event_source)
        ) PARTITION BY RANGE (created_at);

        RAISE NOTICE 'Created table threat_intel.security_events';
    END IF;
END $$;

-- Create partitions for security events
DO $$
DECLARE
    current_month DATE := DATE_TRUNC('month', NOW());
    next_month DATE := DATE_TRUNC('month', NOW() + INTERVAL '1 month');
BEGIN
    PERFORM create_monthly_partition('threat_intel.security_events', current_month);
    PERFORM create_monthly_partition('threat_intel.security_events', next_month);
    RAISE NOTICE 'Created partitions for threat_intel.security_events';
EXCEPTION
    WHEN OTHERS THEN
        RAISE WARNING 'Error creating partitions for security_events: %', SQLERRM;
END $$;

-- Create indexes for threat intelligence tables
DO $$
BEGIN
    -- Threat indicators indexes
    CREATE INDEX IF NOT EXISTS threat_indicators_value_idx ON threat_intel.threat_indicators (indicator_value, indicator_type);
    CREATE INDEX IF NOT EXISTS threat_indicators_type_idx ON threat_intel.threat_indicators (indicator_type, threat_type);
    CREATE INDEX IF NOT EXISTS threat_indicators_severity_idx ON threat_intel.threat_indicators (severity, confidence, priority DESC);
    CREATE INDEX IF NOT EXISTS threat_indicators_active_idx ON threat_intel.threat_indicators (is_active, last_seen DESC);
    CREATE INDEX IF NOT EXISTS threat_indicators_actor_idx ON threat_intel.threat_indicators (attributed_actor, threat_actors);
    CREATE INDEX IF NOT EXISTS threat_indicators_campaign_idx ON threat_intel.threat_indicators (campaigns, parent_campaign);
    CREATE INDEX IF NOT EXISTS threat_indicators_mitre_idx ON threat_intel.threat_indicators USING gin(mitre_techniques);
    CREATE INDEX IF NOT EXISTS threat_indicators_embedding_idx ON threat_intel.threat_indicators USING ivfflat (indicator_embedding vector_cosine_ops) WITH (lists = 100);
    CREATE INDEX IF NOT EXISTS threat_indicators_source_idx ON threat_intel.threat_indicators (source_reliability, information_credibility);
    CREATE INDEX IF NOT EXISTS threat_indicators_search_idx ON threat_intel.threat_indicators USING gin(tags, malware_families);

    -- Threat actors indexes
    CREATE INDEX IF NOT EXISTS threat_actors_name_idx ON threat_intel.threat_actors (actor_name, actor_aliases);
    CREATE INDEX IF NOT EXISTS threat_actors_type_idx ON threat_intel.threat_actors (actor_type, sophistication_level);
    CREATE INDEX IF NOT EXISTS threat_actors_motivation_idx ON threat_intel.threat_actors (primary_motivation, target_sectors);
    CREATE INDEX IF NOT EXISTS threat_actors_nationality_idx ON threat_intel.threat_actors (suspected_nationality, suspected_location);
    CREATE INDEX IF NOT EXISTS threat_actors_activity_idx ON threat_intel.threat_actors (activity_level, last_observed DESC);
    CREATE INDEX IF NOT EXISTS threat_actors_capabilities_idx ON threat_intel.threat_actors (sophistication_level, resources_level);
    CREATE INDEX IF NOT EXISTS threat_actors_mitre_idx ON threat_intel.threat_actors USING gin(mitre_techniques);
    CREATE INDEX IF NOT EXISTS threat_actors_embedding_idx ON threat_intel.threat_actors USING ivfflat (actor_profile_embedding vector_cosine_ops) WITH (lists = 50);
    CREATE INDEX IF NOT EXISTS threat_actors_campaigns_idx ON threat_intel.threat_actors USING gin(known_campaigns);

    -- Security events indexes
    CREATE INDEX IF NOT EXISTS security_events_id_idx ON threat_intel.security_events (event_id, event_source);
    CREATE INDEX IF NOT EXISTS security_events_type_idx ON threat_intel.security_events (event_type, event_category);
    CREATE INDEX IF NOT EXISTS security_events_severity_idx ON threat_intel.security_events (severity, confidence, event_timestamp DESC);
    CREATE INDEX IF NOT EXISTS security_events_network_idx ON threat_intel.security_events (source_ip, destination_ip, protocol);
    CREATE INDEX IF NOT EXISTS security_events_indicators_idx ON threat_intel.security_events USING gin(related_indicators);
    CREATE INDEX IF NOT EXISTS security_events_response_idx ON threat_intel.security_events (response_status, detection_timestamp DESC);
    CREATE INDEX IF NOT EXISTS security_events_hashes_idx ON threat_intel.security_events USING gin(file_hashes);
    CREATE INDEX IF NOT EXISTS security_events_embedding_idx ON threat_intel.security_events USING ivfflat (event_embedding vector_cosine_ops) WITH (lists = 100);

    RAISE NOTICE 'Created indexes for threat intelligence tables';
END $$;

-- Add audit triggers
DO $$
BEGIN
    -- Threat indicators audit trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'threat_indicators_audit_trigger') THEN
        CREATE TRIGGER threat_indicators_audit_trigger
            BEFORE INSERT OR UPDATE ON threat_intel.threat_indicators
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
    END IF;

    -- Threat actors audit trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'threat_actors_audit_trigger') THEN
        CREATE TRIGGER threat_actors_audit_trigger
            BEFORE INSERT OR UPDATE ON threat_intel.threat_actors
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
    END IF;

    -- Security events audit trigger
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'security_events_audit_trigger') THEN
        CREATE TRIGGER security_events_audit_trigger
            BEFORE INSERT OR UPDATE ON threat_intel.security_events
            FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
    END IF;

    RAISE NOTICE 'Created audit triggers for threat intelligence tables';
END $$;

-- Validate migration
DO $$
DECLARE
    table_count INTEGER;
    index_count INTEGER;
BEGIN
    -- Count created tables
    SELECT COUNT(*) INTO table_count
    FROM information_schema.tables
    WHERE table_schema = 'threat_intel';

    -- Count created indexes
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE schemaname = 'threat_intel';

    RAISE NOTICE 'Migration validation: % tables created, % indexes created', table_count, index_count;

    IF table_count < 3 THEN
        RAISE EXCEPTION 'Migration validation failed: insufficient tables created';
    END IF;
END $$;

-- Log successful migration
INSERT INTO migration_log.migration_history (
    migration_name,
    migration_version,
    applied_by,
    rollback_script,
    checksum
) VALUES (
    'phase8_threat_intelligence',
    '1.0.0',
    current_user,
    'DROP SCHEMA threat_intel CASCADE;',
    md5('phase8_threat_intelligence_1.0.0')
);

-- Commit transaction
COMMIT;

RAISE NOTICE 'Migration phase8_threat_intelligence completed successfully';