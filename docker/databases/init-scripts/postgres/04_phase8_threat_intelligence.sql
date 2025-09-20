-- Phase 8: Advanced Security Operations - Threat Intelligence Schema
-- Comprehensive threat intelligence and indicators management

-- Create threat intelligence schema
CREATE SCHEMA IF NOT EXISTS threat_intel;

-- ========================================
-- THREAT INDICATORS (IOCs/IOAs)
-- ========================================
CREATE TABLE threat_intel.threat_indicators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Indicator identification
    indicator_value TEXT NOT NULL,
    indicator_type TEXT NOT NULL CHECK (indicator_type IN (
        'ip_address', 'domain', 'url', 'file_hash', 'email', 'phone', 'bitcoin_address',
        'tor_address', 'telegram_id', 'user_agent', 'certificate_hash', 'registry_key',
        'mutex', 'pdb_path', 'campaign_id', 'malware_family', 'attack_pattern'
    )),

    -- Threat classification
    threat_type TEXT[] NOT NULL,
    malware_families TEXT[],
    threat_actors TEXT[],
    campaigns TEXT[],

    -- Confidence and severity
    confidence confidence_level NOT NULL DEFAULT 'medium',
    severity severity_level NOT NULL DEFAULT 'medium',
    priority INTEGER CHECK (priority BETWEEN 1 AND 10) DEFAULT 5,

    -- Temporal relevance
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expiry_date TIMESTAMPTZ,
    is_active BOOLEAN DEFAULT TRUE,

    -- Attribution
    attributed_actor TEXT,
    actor_confidence DECIMAL(3,2),
    attribution_methods TEXT[],
    attribution_evidence JSONB,

    -- Technical details
    indicator_context JSONB,
    technical_details JSONB,
    kill_chain_phases TEXT[],
    mitre_techniques TEXT[],

    -- Relationships
    related_indicators UUID[],
    parent_campaign UUID,
    related_samples TEXT[],

    -- Detection capabilities
    detection_rules JSONB,
    false_positive_rate DECIMAL(5,4),
    detection_confidence DECIMAL(3,2),

    -- Intelligence sources
    source_reliability TEXT CHECK (source_reliability IN ('A', 'B', 'C', 'D', 'E', 'F')),
    information_credibility TEXT CHECK (information_credibility IN ('1', '2', '3', '4', '5', '6')),
    data_sources TEXT[] NOT NULL,
    collection_methods TEXT[],

    -- Sharing and distribution
    tlp_marking TEXT CHECK (tlp_marking IN ('WHITE', 'GREEN', 'AMBER', 'RED')) DEFAULT 'AMBER',
    sharing_restrictions JSONB,
    distribution_list TEXT[],

    -- Embedding for similarity analysis
    indicator_embedding vector(768),

    -- Metadata
    analyst_id TEXT,
    analysis_notes TEXT,
    tags TEXT[],
    raw_intelligence JSONB,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(indicator_value, indicator_type)
) PARTITION BY RANGE (created_at);

-- Create partitions for threat indicators
SELECT create_monthly_partition('threat_intel.threat_indicators', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('threat_intel.threat_indicators', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for threat indicators
CREATE INDEX threat_indicators_value_idx ON threat_intel.threat_indicators (indicator_value, indicator_type);
CREATE INDEX threat_indicators_type_idx ON threat_intel.threat_indicators (indicator_type, threat_type);
CREATE INDEX threat_indicators_severity_idx ON threat_intel.threat_indicators (severity, confidence, priority DESC);
CREATE INDEX threat_indicators_active_idx ON threat_intel.threat_indicators (is_active, last_seen DESC);
CREATE INDEX threat_indicators_actor_idx ON threat_intel.threat_indicators (attributed_actor, threat_actors);
CREATE INDEX threat_indicators_campaign_idx ON threat_intel.threat_indicators (campaigns, parent_campaign);
CREATE INDEX threat_indicators_mitre_idx ON threat_intel.threat_indicators USING gin(mitre_techniques);
CREATE INDEX threat_indicators_embedding_idx ON threat_intel.threat_indicators USING ivfflat (indicator_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX threat_indicators_source_idx ON threat_intel.threat_indicators (source_reliability, information_credibility);
CREATE INDEX threat_indicators_search_idx ON threat_intel.threat_indicators USING gin(tags, malware_families);

-- ========================================
-- THREAT ACTORS
-- ========================================
CREATE TABLE threat_intel.threat_actors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Actor identification
    actor_name TEXT NOT NULL,
    actor_aliases TEXT[],
    actor_type TEXT CHECK (actor_type IN ('apt', 'cybercriminal', 'hacktivist', 'nation_state', 'insider', 'script_kiddie', 'unknown')),

    -- Attribution and geolocation
    suspected_nationality TEXT,
    suspected_location TEXT,
    attribution_confidence DECIMAL(3,2),
    attribution_evidence JSONB,

    -- Motivations and objectives
    primary_motivation TEXT CHECK (primary_motivation IN ('financial', 'espionage', 'ideology', 'revenge', 'fame', 'unknown')),
    secondary_motivations TEXT[],
    target_sectors TEXT[],
    target_geographies TEXT[],

    -- Capabilities assessment
    sophistication_level TEXT CHECK (sophistication_level IN ('low', 'medium', 'high', 'expert')),
    resources_level TEXT CHECK (resources_level IN ('individual', 'small_group', 'organized_group', 'state_sponsored')),
    technical_capabilities JSONB,
    operational_capabilities JSONB,

    -- Activity patterns
    first_observed TIMESTAMPTZ,
    last_observed TIMESTAMPTZ,
    activity_level TEXT CHECK (activity_level IN ('dormant', 'low', 'medium', 'high', 'very_high')),
    activity_patterns JSONB,

    -- TTPs (Tactics, Techniques, Procedures)
    preferred_malware_families TEXT[],
    common_attack_vectors TEXT[],
    mitre_techniques TEXT[],
    signature_behaviors JSONB,

    -- Infrastructure
    known_infrastructure JSONB,
    infrastructure_patterns JSONB,
    communication_methods TEXT[],

    -- Campaigns and operations
    known_campaigns TEXT[],
    operation_names TEXT[],
    campaign_patterns JSONB,

    -- Threat landscape
    threat_severity severity_level DEFAULT 'medium',
    current_threat_level INTEGER CHECK (current_threat_level BETWEEN 1 AND 5),
    trend_analysis JSONB,

    -- Intelligence assessment
    intelligence_requirements TEXT[],
    knowledge_gaps TEXT[],
    assessment_confidence DECIMAL(3,2),

    -- Relationships
    associated_actors UUID[],
    allied_groups TEXT[],
    competitor_groups TEXT[],

    -- Profile embedding for similarity analysis
    actor_profile_embedding vector(768),

    -- Metadata
    analyst_id TEXT,
    last_analysis_update TIMESTAMPTZ,
    intelligence_sources TEXT[],

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(actor_name)
);

-- Indexes for threat actors
CREATE INDEX threat_actors_name_idx ON threat_intel.threat_actors (actor_name, actor_aliases);
CREATE INDEX threat_actors_type_idx ON threat_intel.threat_actors (actor_type, sophistication_level);
CREATE INDEX threat_actors_motivation_idx ON threat_intel.threat_actors (primary_motivation, target_sectors);
CREATE INDEX threat_actors_nationality_idx ON threat_intel.threat_actors (suspected_nationality, suspected_location);
CREATE INDEX threat_actors_activity_idx ON threat_intel.threat_actors (activity_level, last_observed DESC);
CREATE INDEX threat_actors_capabilities_idx ON threat_intel.threat_actors (sophistication_level, resources_level);
CREATE INDEX threat_actors_mitre_idx ON threat_intel.threat_actors USING gin(mitre_techniques);
CREATE INDEX threat_actors_embedding_idx ON threat_intel.threat_actors USING ivfflat (actor_profile_embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX threat_actors_campaigns_idx ON threat_intel.threat_actors USING gin(known_campaigns);

-- ========================================
-- ATTACK CAMPAIGNS
-- ========================================
CREATE TABLE threat_intel.attack_campaigns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Campaign identification
    campaign_name TEXT NOT NULL,
    campaign_aliases TEXT[],
    campaign_type TEXT CHECK (campaign_type IN ('targeted', 'opportunistic', 'mass', 'supply_chain', 'watering_hole', 'spear_phishing')),

    -- Attribution
    attributed_actors UUID[] REFERENCES threat_intel.threat_actors(id),
    attribution_confidence DECIMAL(3,2),
    attribution_methods TEXT[],

    -- Campaign scope and targeting
    target_industries TEXT[],
    target_countries TEXT[],
    target_organizations TEXT[],
    victim_count INTEGER DEFAULT 0,
    estimated_scope TEXT CHECK (estimated_scope IN ('limited', 'regional', 'global', 'unknown')),

    -- Temporal analysis
    campaign_start TIMESTAMPTZ,
    campaign_end TIMESTAMPTZ,
    last_activity TIMESTAMPTZ,
    campaign_status TEXT CHECK (campaign_status IN ('active', 'dormant', 'concluded', 'unknown')),

    -- Campaign objectives
    primary_objective TEXT,
    secondary_objectives TEXT[],
    success_indicators JSONB,
    achieved_objectives TEXT[],

    -- Technical analysis
    attack_vectors TEXT[],
    malware_families TEXT[],
    tools_used TEXT[],
    infrastructure_used JSONB,

    -- TTPs analysis
    mitre_techniques TEXT[],
    kill_chain_mapping JSONB,
    behavioral_patterns JSONB,

    -- Impact assessment
    impact_severity severity_level DEFAULT 'medium',
    financial_impact DECIMAL(15,2),
    data_compromised BOOLEAN DEFAULT FALSE,
    systems_compromised INTEGER DEFAULT 0,
    business_disruption JSONB,

    -- Defense and mitigation
    detection_rules JSONB,
    mitigation_strategies TEXT[],
    defensive_measures JSONB,
    lessons_learned TEXT,

    -- Intelligence collection
    information_sources TEXT[],
    evidence_artifacts JSONB,
    ioc_count INTEGER DEFAULT 0,
    confidence_assessment JSONB,

    -- Relationships
    related_campaigns UUID[],
    predecessor_campaigns UUID[],
    successor_campaigns UUID[],

    -- Campaign embedding for similarity analysis
    campaign_embedding vector(768),

    -- Metadata
    discovery_date TIMESTAMPTZ,
    analyst_team TEXT[],
    intelligence_requirements TEXT[],

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(campaign_name)
) PARTITION BY RANGE (created_at);

-- Create partitions for attack campaigns
SELECT create_monthly_partition('threat_intel.attack_campaigns', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('threat_intel.attack_campaigns', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for attack campaigns
CREATE INDEX attack_campaigns_name_idx ON threat_intel.attack_campaigns (campaign_name, campaign_aliases);
CREATE INDEX attack_campaigns_actors_idx ON threat_intel.attack_campaigns USING gin(attributed_actors);
CREATE INDEX attack_campaigns_targets_idx ON threat_intel.attack_campaigns USING gin(target_industries, target_countries);
CREATE INDEX attack_campaigns_status_idx ON threat_intel.attack_campaigns (campaign_status, last_activity DESC);
CREATE INDEX attack_campaigns_impact_idx ON threat_intel.attack_campaigns (impact_severity, financial_impact DESC);
CREATE INDEX attack_campaigns_techniques_idx ON threat_intel.attack_campaigns USING gin(mitre_techniques);
CREATE INDEX attack_campaigns_embedding_idx ON threat_intel.attack_campaigns USING ivfflat (campaign_embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX attack_campaigns_timeline_idx ON threat_intel.attack_campaigns (campaign_start, campaign_end);

-- ========================================
-- SECURITY EVENTS
-- ========================================
CREATE TABLE threat_intel.security_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Event identification
    event_id TEXT NOT NULL,
    event_type TEXT CHECK (event_type IN ('detection', 'incident', 'alert', 'observation', 'report')),
    event_source TEXT NOT NULL,

    -- Event classification
    event_category TEXT CHECK (event_category IN ('malware', 'network_intrusion', 'data_exfiltration', 'credential_theft', 'reconnaissance', 'lateral_movement', 'persistence', 'command_control')),
    severity severity_level NOT NULL DEFAULT 'medium',
    confidence confidence_level NOT NULL DEFAULT 'medium',

    -- Temporal data
    event_timestamp TIMESTAMPTZ NOT NULL,
    detection_timestamp TIMESTAMPTZ DEFAULT NOW(),
    first_seen TIMESTAMPTZ,
    last_seen TIMESTAMPTZ,

    -- Event details
    event_description TEXT NOT NULL,
    technical_details JSONB,
    affected_systems TEXT[],
    affected_networks TEXT[],

    -- Threat context
    related_indicators UUID[],
    related_actors UUID[],
    related_campaigns UUID[],
    threat_context JSONB,

    -- Network and system context
    source_ip INET,
    destination_ip INET,
    source_port INTEGER,
    destination_port INTEGER,
    protocol TEXT,
    network_artifacts JSONB,

    -- File and process context
    file_hashes TEXT[],
    process_names TEXT[],
    registry_keys TEXT[],
    command_lines TEXT[],

    -- Detection information
    detection_rule_id TEXT,
    detection_method TEXT,
    signature_id TEXT,
    false_positive_likelihood DECIMAL(3,2),

    -- Response actions
    containment_actions TEXT[],
    mitigation_steps TEXT[],
    response_status TEXT CHECK (response_status IN ('new', 'investigating', 'contained', 'mitigated', 'resolved', 'false_positive')),

    -- Analyst assessment
    analyst_notes TEXT,
    investigation_findings JSONB,
    recommended_actions TEXT[],

    -- Integration with external systems
    external_ticket_id TEXT,
    siem_event_id TEXT,
    correlation_id TEXT,

    -- Event embedding for similarity analysis
    event_embedding vector(768),

    -- Metadata
    data_source data_source_type NOT NULL,
    collection_method TEXT,
    analyst_id TEXT,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(event_id, event_source)
) PARTITION BY RANGE (created_at);

-- Create partitions for security events
SELECT create_monthly_partition('threat_intel.security_events', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('threat_intel.security_events', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for security events
CREATE INDEX security_events_id_idx ON threat_intel.security_events (event_id, event_source);
CREATE INDEX security_events_type_idx ON threat_intel.security_events (event_type, event_category);
CREATE INDEX security_events_severity_idx ON threat_intel.security_events (severity, confidence, event_timestamp DESC);
CREATE INDEX security_events_network_idx ON threat_intel.security_events (source_ip, destination_ip, protocol);
CREATE INDEX security_events_indicators_idx ON threat_intel.security_events USING gin(related_indicators);
CREATE INDEX security_events_response_idx ON threat_intel.security_events (response_status, detection_timestamp DESC);
CREATE INDEX security_events_hashes_idx ON threat_intel.security_events USING gin(file_hashes);
CREATE INDEX security_events_embedding_idx ON threat_intel.security_events USING ivfflat (event_embedding vector_cosine_ops) WITH (lists = 100);

-- ========================================
-- BEHAVIORAL PROFILES
-- ========================================
CREATE TABLE threat_intel.behavioral_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Profile identification
    profile_name TEXT NOT NULL,
    profile_type TEXT CHECK (profile_type IN ('actor', 'campaign', 'malware_family', 'attack_pattern', 'infrastructure')),
    entity_id UUID, -- References actor, campaign, etc.

    -- Behavioral characteristics
    behavior_patterns JSONB NOT NULL,
    signature_behaviors JSONB,
    anomaly_indicators JSONB,

    -- Temporal behavior
    time_patterns JSONB,
    activity_cycles JSONB,
    operational_windows JSONB,

    -- Technical behavior
    network_behavior JSONB,
    system_behavior JSONB,
    communication_patterns JSONB,

    -- Target behavior
    targeting_patterns JSONB,
    victim_selection_criteria JSONB,
    reconnaissance_behavior JSONB,

    -- Operational behavior
    persistence_methods JSONB,
    evasion_techniques JSONB,
    command_control_behavior JSONB,

    -- Statistical analysis
    behavior_frequency JSONB,
    pattern_reliability DECIMAL(3,2),
    deviation_thresholds JSONB,

    -- Machine learning features
    feature_vector JSONB,
    clustering_assignment TEXT,
    anomaly_score DECIMAL(5,4),

    -- Validation and testing
    false_positive_rate DECIMAL(5,4),
    detection_accuracy DECIMAL(3,2),
    profile_effectiveness JSONB,

    -- Profile embedding for similarity analysis
    behavior_embedding vector(768),

    -- Metadata
    model_version TEXT,
    training_data_period JSONB,
    last_updated TIMESTAMPTZ,
    analyst_id TEXT,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(profile_name, profile_type)
);

-- Indexes for behavioral profiles
CREATE INDEX behavioral_profiles_name_idx ON threat_intel.behavioral_profiles (profile_name, profile_type);
CREATE INDEX behavioral_profiles_entity_idx ON threat_intel.behavioral_profiles (entity_id, profile_type);
CREATE INDEX behavioral_profiles_reliability_idx ON threat_intel.behavioral_profiles (pattern_reliability DESC, detection_accuracy DESC);
CREATE INDEX behavioral_profiles_clustering_idx ON threat_intel.behavioral_profiles (clustering_assignment, anomaly_score DESC);
CREATE INDEX behavioral_profiles_embedding_idx ON threat_intel.behavioral_profiles USING ivfflat (behavior_embedding vector_cosine_ops) WITH (lists = 50);

-- Add audit triggers for threat intelligence tables
CREATE TRIGGER threat_indicators_audit_trigger
    BEFORE INSERT OR UPDATE ON threat_intel.threat_indicators
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER threat_actors_audit_trigger
    BEFORE INSERT OR UPDATE ON threat_intel.threat_actors
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER attack_campaigns_audit_trigger
    BEFORE INSERT OR UPDATE ON threat_intel.attack_campaigns
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER security_events_audit_trigger
    BEFORE INSERT OR UPDATE ON threat_intel.security_events
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER behavioral_profiles_audit_trigger
    BEFORE INSERT OR UPDATE ON threat_intel.behavioral_profiles
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();