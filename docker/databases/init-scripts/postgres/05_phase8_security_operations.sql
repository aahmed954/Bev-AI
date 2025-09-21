-- Phase 8: Advanced Security Operations - Security Operations Schema
-- Comprehensive security operations and incident response management

-- Create security operations schema
CREATE SCHEMA IF NOT EXISTS security_ops;

-- ========================================
-- DEFENSE ACTIONS
-- ========================================
CREATE TABLE security_ops.defense_actions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Action identification
    action_id TEXT NOT NULL,
    action_type TEXT CHECK (action_type IN (
        'block_ip', 'block_domain', 'quarantine_file', 'isolate_system',
        'patch_system', 'update_signatures', 'reset_credentials',
        'revoke_access', 'monitor_activity', 'backup_data'
    )),
    action_category TEXT CHECK (action_category IN ('preventive', 'detective', 'corrective', 'recovery')),

    -- Triggering context
    trigger_event_id UUID REFERENCES threat_intel.security_events(id),
    trigger_indicators UUID[],
    trigger_rules TEXT[],
    automated_trigger BOOLEAN DEFAULT FALSE,

    -- Action details
    action_description TEXT NOT NULL,
    target_systems TEXT[],
    target_networks TEXT[],
    affected_users TEXT[],
    action_parameters JSONB,

    -- Execution information
    execution_status TEXT CHECK (execution_status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled', 'rolled_back')),
    execution_method TEXT CHECK (execution_method IN ('manual', 'automated', 'semi_automated')),
    executor_id TEXT,
    execution_team TEXT[],

    -- Temporal tracking
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    estimated_duration INTERVAL,
    actual_duration INTERVAL,

    -- Impact assessment
    business_impact severity_level DEFAULT 'low',
    operational_impact JSONB,
    user_impact_count INTEGER DEFAULT 0,
    system_downtime INTERVAL,

    -- Effectiveness metrics
    success_rate DECIMAL(3,2),
    false_positive_count INTEGER DEFAULT 0,
    collateral_damage JSONB,
    effectiveness_score INTEGER CHECK (effectiveness_score BETWEEN 1 AND 10),

    -- Approval and authorization
    approval_required BOOLEAN DEFAULT FALSE,
    approved_by TEXT,
    approval_timestamp TIMESTAMPTZ,
    authorization_level TEXT,

    -- Rollback capability
    rollback_available BOOLEAN DEFAULT FALSE,
    rollback_procedure JSONB,
    rollback_executed BOOLEAN DEFAULT FALSE,
    rollback_timestamp TIMESTAMPTZ,

    -- Dependencies and relationships
    prerequisite_actions UUID[],
    dependent_actions UUID[],
    related_incidents UUID[],

    -- Compliance and audit
    compliance_requirements TEXT[],
    audit_trail JSONB,
    documentation_link TEXT,

    -- Learning and improvement
    lessons_learned TEXT,
    improvement_suggestions TEXT[],
    playbook_updates JSONB,

    -- Metadata
    created_by TEXT NOT NULL,
    priority INTEGER CHECK (priority BETWEEN 1 AND 5) DEFAULT 3,
    tags TEXT[],

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(action_id)
) PARTITION BY RANGE (created_at);

-- Create partitions for defense actions
SELECT create_monthly_partition('security_ops.defense_actions', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('security_ops.defense_actions', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for defense actions
CREATE INDEX defense_actions_id_idx ON security_ops.defense_actions (action_id, action_type);
CREATE INDEX defense_actions_status_idx ON security_ops.defense_actions (execution_status, requested_at DESC);
CREATE INDEX defense_actions_trigger_idx ON security_ops.defense_actions (trigger_event_id, automated_trigger);
CREATE INDEX defense_actions_priority_idx ON security_ops.defense_actions (priority DESC, business_impact);
CREATE INDEX defense_actions_executor_idx ON security_ops.defense_actions (executor_id, execution_team);
CREATE INDEX defense_actions_systems_idx ON security_ops.defense_actions USING gin(target_systems);
CREATE INDEX defense_actions_effectiveness_idx ON security_ops.defense_actions (effectiveness_score DESC, success_rate DESC);

-- ========================================
-- THREAT HUNTS
-- ========================================
CREATE TABLE security_ops.threat_hunts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Hunt identification
    hunt_id TEXT NOT NULL,
    hunt_name TEXT NOT NULL,
    hunt_type TEXT CHECK (hunt_type IN ('hypothesis_driven', 'intel_driven', 'signature_based', 'behavioral_anomaly', 'threat_landscape')),
    hunt_methodology TEXT,

    -- Hunt scope and targeting
    hunt_scope JSONB NOT NULL,
    target_environments TEXT[],
    data_sources TEXT[],
    time_range JSONB,

    -- Hunt hypothesis and objectives
    hypothesis TEXT,
    hunt_objectives TEXT[],
    success_criteria JSONB,
    intelligence_requirements TEXT[],

    -- Hunt execution
    hunt_status TEXT CHECK (hunt_status IN ('planned', 'active', 'paused', 'completed', 'cancelled')),
    execution_phase TEXT CHECK (execution_phase IN ('preparation', 'data_collection', 'analysis', 'validation', 'reporting')),
    hunt_team TEXT[],
    lead_hunter TEXT,

    -- Temporal tracking
    planned_start TIMESTAMPTZ,
    planned_end TIMESTAMPTZ,
    actual_start TIMESTAMPTZ,
    actual_end TIMESTAMPTZ,
    estimated_effort INTERVAL,
    actual_effort INTERVAL,

    -- Hunt techniques and tools
    hunt_techniques TEXT[],
    analysis_tools TEXT[],
    query_techniques JSONB,
    automation_level TEXT CHECK (automation_level IN ('manual', 'semi_automated', 'automated')),

    -- Data analysis
    data_volume_processed BIGINT,
    queries_executed INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,

    -- Findings and results
    findings_summary TEXT,
    iocs_discovered TEXT[],
    new_threats_identified UUID[],
    attack_techniques_found TEXT[],
    compromised_systems TEXT[],

    -- Impact and value
    threat_severity_found severity_level,
    business_risk_identified TEXT,
    potential_damage_prevented DECIMAL(15,2),
    hunt_value_score INTEGER CHECK (hunt_value_score BETWEEN 1 AND 10),

    -- Knowledge development
    new_detection_rules JSONB,
    updated_playbooks TEXT[],
    threat_intelligence_produced JSONB,
    capability_improvements TEXT[],

    -- Quality metrics
    hunt_effectiveness DECIMAL(3,2),
    false_positive_rate DECIMAL(5,4),
    coverage_assessment JSONB,
    gaps_identified TEXT[],

    -- Follow-up actions
    recommended_actions TEXT[],
    created_alerts UUID[],
    follow_up_hunts UUID[],
    incident_escalations UUID[],

    -- Documentation and reporting
    hunt_documentation JSONB,
    final_report_url TEXT,
    stakeholder_briefings JSONB,

    -- Metadata
    hunt_priority INTEGER CHECK (hunt_priority BETWEEN 1 AND 5) DEFAULT 3,
    resource_allocation JSONB,
    budget_consumed DECIMAL(12,2),

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(hunt_id)
) PARTITION BY RANGE (created_at);

-- Create partitions for threat hunts
SELECT create_monthly_partition('security_ops.threat_hunts', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('security_ops.threat_hunts', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for threat hunts
CREATE INDEX threat_hunts_id_idx ON security_ops.threat_hunts (hunt_id, hunt_name);
CREATE INDEX threat_hunts_status_idx ON security_ops.threat_hunts (hunt_status, execution_phase);
CREATE INDEX threat_hunts_team_idx ON security_ops.threat_hunts (lead_hunter, hunt_team);
CREATE INDEX threat_hunts_effectiveness_idx ON security_ops.threat_hunts (hunt_effectiveness DESC, hunt_value_score DESC);
CREATE INDEX threat_hunts_findings_idx ON security_ops.threat_hunts (threat_severity_found, true_positives DESC);
CREATE INDEX threat_hunts_timeline_idx ON security_ops.threat_hunts (planned_start, actual_start);
CREATE INDEX threat_hunts_techniques_idx ON security_ops.threat_hunts USING gin(hunt_techniques);

-- ========================================
-- INCIDENT RESPONSES
-- ========================================
CREATE TABLE security_ops.incident_responses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Incident identification
    incident_id TEXT NOT NULL,
    incident_title TEXT NOT NULL,
    incident_type TEXT CHECK (incident_type IN (
        'data_breach', 'malware_infection', 'network_intrusion', 'ddos_attack',
        'insider_threat', 'phishing_campaign', 'ransomware', 'data_exfiltration',
        'system_compromise', 'credential_theft', 'supply_chain_attack'
    )),

    -- Incident classification
    severity severity_level NOT NULL,
    priority INTEGER CHECK (priority BETWEEN 1 AND 5) NOT NULL,
    impact_level TEXT CHECK (impact_level IN ('low', 'medium', 'high', 'critical')),
    urgency_level TEXT CHECK (urgency_level IN ('low', 'medium', 'high', 'critical')),

    -- Incident status and workflow
    status TEXT CHECK (status IN ('new', 'assigned', 'investigating', 'containing', 'eradicating', 'recovering', 'resolved', 'closed')),
    escalation_level INTEGER DEFAULT 1,
    escalation_reasons TEXT[],

    -- Discovery and reporting
    discovered_at TIMESTAMPTZ NOT NULL,
    reported_at TIMESTAMPTZ,
    discovery_method TEXT,
    initial_detector TEXT,
    reporting_source TEXT,

    -- Incident details
    incident_description TEXT NOT NULL,
    affected_systems TEXT[],
    affected_networks TEXT[],
    affected_applications TEXT[],
    affected_data_types TEXT[],

    -- Timeline and chronology
    incident_timeline JSONB,
    first_occurrence TIMESTAMPTZ,
    last_occurrence TIMESTAMPTZ,
    attack_progression JSONB,

    -- Attribution and threat context
    suspected_actors UUID[],
    related_campaigns UUID[],
    attack_vectors TEXT[],
    mitre_techniques TEXT[],
    iocs_involved UUID[],

    -- Response team and assignment
    incident_commander TEXT,
    response_team TEXT[],
    assigned_analysts TEXT[],
    external_consultants TEXT[],

    -- Response actions and containment
    containment_actions UUID[],
    eradication_actions UUID[],
    recovery_actions UUID[],
    response_timeline JSONB,

    -- Impact assessment
    business_impact JSONB,
    financial_impact DECIMAL(15,2),
    data_compromised JSONB,
    systems_affected_count INTEGER DEFAULT 0,
    users_affected_count INTEGER DEFAULT 0,

    -- Evidence and artifacts
    evidence_collected JSONB,
    forensic_images TEXT[],
    log_files TEXT[],
    memory_dumps TEXT[],
    network_captures TEXT[],

    -- Communication and notifications
    stakeholders_notified TEXT[],
    customer_notifications JSONB,
    regulatory_notifications JSONB,
    media_response JSONB,

    -- Resolution and closure
    root_cause_analysis JSONB,
    resolution_summary TEXT,
    lessons_learned TEXT,
    preventive_measures TEXT[],

    -- Compliance and legal
    regulatory_requirements TEXT[],
    legal_holds BOOLEAN DEFAULT FALSE,
    compliance_notifications JSONB,
    breach_notification_required BOOLEAN DEFAULT FALSE,

    -- Metrics and KPIs
    detection_time INTERVAL,
    containment_time INTERVAL,
    resolution_time INTERVAL,
    response_effectiveness INTEGER CHECK (response_effectiveness BETWEEN 1 AND 10),

    -- Post-incident activities
    post_incident_review_date TIMESTAMPTZ,
    improvement_actions TEXT[],
    policy_updates TEXT[],
    training_requirements TEXT[],

    -- Related incidents and patterns
    related_incidents UUID[],
    similar_incidents UUID[],
    incident_pattern TEXT,

    -- Documentation and reporting
    incident_report_url TEXT,
    executive_summary TEXT,
    technical_details JSONB,

    -- Metadata
    created_by TEXT NOT NULL,
    last_updated_by TEXT,
    tags TEXT[],

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(incident_id)
) PARTITION BY RANGE (created_at);

-- Create partitions for incident responses
SELECT create_monthly_partition('security_ops.incident_responses', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('security_ops.incident_responses', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for incident responses
CREATE INDEX incident_responses_id_idx ON security_ops.incident_responses (incident_id, incident_type);
CREATE INDEX incident_responses_status_idx ON security_ops.incident_responses (status, priority DESC, severity);
CREATE INDEX incident_responses_team_idx ON security_ops.incident_responses (incident_commander, response_team);
CREATE INDEX incident_responses_timeline_idx ON security_ops.incident_responses (discovered_at DESC, first_occurrence);
CREATE INDEX incident_responses_attribution_idx ON security_ops.incident_responses USING gin(suspected_actors);
CREATE INDEX incident_responses_impact_idx ON security_ops.incident_responses (impact_level, financial_impact DESC);
CREATE INDEX incident_responses_techniques_idx ON security_ops.incident_responses USING gin(mitre_techniques);
CREATE INDEX incident_responses_systems_idx ON security_ops.incident_responses USING gin(affected_systems);

-- ========================================
-- SECURITY METRICS
-- ========================================
CREATE TABLE security_ops.security_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Metric identification
    metric_name TEXT NOT NULL,
    metric_type TEXT CHECK (metric_type IN ('kpi', 'sla', 'effectiveness', 'coverage', 'risk', 'operational')),
    metric_category TEXT CHECK (metric_category IN ('detection', 'response', 'prevention', 'recovery', 'compliance')),

    -- Metric definition
    metric_description TEXT,
    calculation_method TEXT,
    data_sources TEXT[],
    measurement_unit TEXT,

    -- Metric values
    metric_value DECIMAL(15,4),
    target_value DECIMAL(15,4),
    threshold_warning DECIMAL(15,4),
    threshold_critical DECIMAL(15,4),

    -- Temporal context
    measurement_period JSONB,
    measurement_timestamp TIMESTAMPTZ NOT NULL,
    reporting_frequency TEXT CHECK (reporting_frequency IN ('real_time', 'hourly', 'daily', 'weekly', 'monthly', 'quarterly')),

    -- Trend analysis
    previous_value DECIMAL(15,4),
    trend_direction TEXT CHECK (trend_direction IN ('improving', 'degrading', 'stable', 'volatile')),
    percentage_change DECIMAL(5,2),

    -- Context and metadata
    business_context JSONB,
    technical_context JSONB,
    environmental_factors TEXT[],

    -- Quality indicators
    data_quality_score DECIMAL(3,2),
    confidence_level confidence_level DEFAULT 'medium',
    measurement_accuracy DECIMAL(3,2),

    -- Stakeholder information
    metric_owner TEXT,
    stakeholders TEXT[],
    reporting_audience TEXT[],

    -- Automation and tooling
    automated_collection BOOLEAN DEFAULT FALSE,
    collection_tools TEXT[],
    validation_rules JSONB,

    -- Alerting and notifications
    alert_enabled BOOLEAN DEFAULT FALSE,
    alert_thresholds JSONB,
    notification_recipients TEXT[],

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(metric_name, measurement_timestamp)
) PARTITION BY RANGE (created_at);

-- Create partitions for security metrics
SELECT create_monthly_partition('security_ops.security_metrics', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('security_ops.security_metrics', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for security metrics
CREATE INDEX security_metrics_name_idx ON security_ops.security_metrics (metric_name, metric_type);
CREATE INDEX security_metrics_timestamp_idx ON security_ops.security_metrics (measurement_timestamp DESC, metric_category);
CREATE INDEX security_metrics_values_idx ON security_ops.security_metrics (metric_value, target_value);
CREATE INDEX security_metrics_trend_idx ON security_ops.security_metrics (trend_direction, percentage_change);
CREATE INDEX security_metrics_quality_idx ON security_ops.security_metrics (data_quality_score DESC, confidence_level);
CREATE INDEX security_metrics_owner_idx ON security_ops.security_metrics (metric_owner, stakeholders);

-- Add audit triggers for security operations tables
CREATE TRIGGER defense_actions_audit_trigger
    BEFORE INSERT OR UPDATE ON security_ops.defense_actions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER threat_hunts_audit_trigger
    BEFORE INSERT OR UPDATE ON security_ops.threat_hunts
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER incident_responses_audit_trigger
    BEFORE INSERT OR UPDATE ON security_ops.incident_responses
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER security_metrics_audit_trigger
    BEFORE INSERT OR UPDATE ON security_ops.security_metrics
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();