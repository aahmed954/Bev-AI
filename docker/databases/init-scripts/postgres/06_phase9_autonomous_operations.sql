-- Phase 9: Autonomous Enhancement Database Schema
-- Comprehensive autonomous operations and machine learning capabilities

-- Create autonomous operations schema
CREATE SCHEMA IF NOT EXISTS autonomous;

-- ========================================
-- CAPABILITY REGISTRY
-- ========================================
CREATE TABLE autonomous.capability_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Capability identification
    capability_id TEXT NOT NULL,
    capability_name TEXT NOT NULL,
    capability_type TEXT CHECK (capability_type IN (
        'data_collection', 'analysis', 'threat_detection', 'response_automation',
        'intelligence_fusion', 'pattern_recognition', 'predictive_modeling',
        'natural_language_processing', 'computer_vision', 'optimization'
    )),
    capability_category TEXT CHECK (capability_category IN ('core', 'specialized', 'experimental', 'deprecated')),

    -- Capability description and metadata
    description TEXT NOT NULL,
    technical_specifications JSONB,
    input_requirements JSONB,
    output_specifications JSONB,
    resource_requirements JSONB,

    -- Implementation details
    implementation_type TEXT CHECK (implementation_type IN ('ml_model', 'rule_based', 'hybrid', 'api_integration', 'custom_algorithm')),
    model_architecture TEXT,
    training_framework TEXT,
    deployment_platform TEXT,
    code_repository TEXT,

    -- Performance characteristics
    accuracy_metrics JSONB,
    performance_benchmarks JSONB,
    latency_characteristics JSONB,
    throughput_capacity JSONB,
    resource_utilization JSONB,

    -- Version and lifecycle
    version TEXT NOT NULL,
    release_date TIMESTAMPTZ,
    deprecation_date TIMESTAMPTZ,
    lifecycle_stage TEXT CHECK (lifecycle_stage IN ('development', 'testing', 'staging', 'production', 'deprecated')),

    -- Dependencies and integrations
    dependencies TEXT[],
    required_capabilities UUID[],
    integrated_systems TEXT[],
    api_endpoints JSONB,

    -- Quality and reliability
    reliability_score DECIMAL(3,2),
    error_rate DECIMAL(5,4),
    uptime_sla DECIMAL(3,2),
    last_validation TIMESTAMPTZ,

    -- Security and access control
    security_classification TEXT CHECK (security_classification IN ('public', 'internal', 'confidential', 'restricted')),
    access_controls JSONB,
    audit_requirements TEXT[],

    -- Learning and adaptation
    learning_enabled BOOLEAN DEFAULT FALSE,
    adaptation_frequency TEXT,
    continuous_learning BOOLEAN DEFAULT FALSE,
    feedback_mechanisms JSONB,

    -- Documentation and support
    documentation_url TEXT,
    support_contacts TEXT[],
    troubleshooting_guides JSONB,
    known_limitations TEXT[],

    -- Usage and monitoring
    usage_statistics JSONB,
    monitoring_endpoints JSONB,
    alerting_rules JSONB,

    -- Metadata
    created_by TEXT NOT NULL,
    maintainer_team TEXT[],
    tags TEXT[],

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(capability_id, version)
);

-- Indexes for capability registry
CREATE INDEX capability_registry_id_idx ON autonomous.capability_registry (capability_id, capability_type);
CREATE INDEX capability_registry_category_idx ON autonomous.capability_registry (capability_category, lifecycle_stage);
CREATE INDEX capability_registry_performance_idx ON autonomous.capability_registry (reliability_score DESC, error_rate);
CREATE INDEX capability_registry_dependencies_idx ON autonomous.capability_registry USING gin(dependencies);
CREATE INDEX capability_registry_tags_idx ON autonomous.capability_registry USING gin(tags);
CREATE INDEX capability_registry_version_idx ON autonomous.capability_registry (capability_id, version);

-- ========================================
-- LEARNING EXPERIMENTS
-- ========================================
CREATE TABLE autonomous.learning_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Experiment identification
    experiment_id TEXT NOT NULL,
    experiment_name TEXT NOT NULL,
    experiment_type TEXT CHECK (experiment_type IN (
        'model_training', 'hyperparameter_tuning', 'feature_engineering',
        'architecture_search', 'transfer_learning', 'reinforcement_learning',
        'active_learning', 'few_shot_learning', 'meta_learning'
    )),

    -- Experiment objectives
    hypothesis TEXT,
    objectives TEXT[],
    success_criteria JSONB,
    expected_outcomes TEXT[],

    -- Experiment design
    methodology TEXT,
    experimental_design JSONB,
    control_variables JSONB,
    independent_variables JSONB,
    dependent_variables JSONB,

    -- Data and datasets
    training_datasets TEXT[],
    validation_datasets TEXT[],
    test_datasets TEXT[],
    data_preprocessing JSONB,
    feature_engineering JSONB,

    -- Model configuration
    model_architecture JSONB,
    hyperparameters JSONB,
    training_configuration JSONB,
    optimization_algorithm TEXT,

    -- Execution details
    experiment_status TEXT CHECK (experiment_status IN ('planned', 'running', 'completed', 'failed', 'cancelled', 'paused')),
    execution_environment TEXT,
    compute_resources JSONB,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,

    -- Results and metrics
    performance_metrics JSONB,
    evaluation_results JSONB,
    model_artifacts JSONB,
    trained_models TEXT[],

    -- Analysis and insights
    results_analysis TEXT,
    statistical_significance JSONB,
    insights_discovered TEXT[],
    unexpected_findings TEXT[],

    -- Comparison and benchmarks
    baseline_comparison JSONB,
    competitive_analysis JSONB,
    performance_improvements JSONB,

    -- Resource utilization
    compute_time INTERVAL,
    memory_usage JSONB,
    storage_usage BIGINT,
    cost_analysis JSONB,

    -- Quality and validation
    reproducibility_score DECIMAL(3,2),
    validation_method TEXT,
    cross_validation_results JSONB,
    robustness_testing JSONB,

    -- Collaboration and team
    principal_investigator TEXT,
    research_team TEXT[],
    collaborators TEXT[],
    external_partners TEXT[],

    -- Documentation and reporting
    experiment_documentation JSONB,
    research_paper_url TEXT,
    presentation_materials TEXT[],
    code_repository TEXT,

    -- Follow-up and deployment
    deployment_readiness BOOLEAN DEFAULT FALSE,
    production_candidate BOOLEAN DEFAULT FALSE,
    follow_up_experiments UUID[],
    recommended_actions TEXT[],

    -- Metadata
    priority INTEGER CHECK (priority BETWEEN 1 AND 5) DEFAULT 3,
    funding_source TEXT,
    ethical_approval TEXT,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(experiment_id)
) PARTITION BY RANGE (created_at);

-- Create partitions for learning experiments
SELECT create_monthly_partition('autonomous.learning_experiments', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('autonomous.learning_experiments', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for learning experiments
CREATE INDEX learning_experiments_id_idx ON autonomous.learning_experiments (experiment_id, experiment_type);
CREATE INDEX learning_experiments_status_idx ON autonomous.learning_experiments (experiment_status, priority DESC);
CREATE INDEX learning_experiments_team_idx ON autonomous.learning_experiments (principal_investigator, research_team);
CREATE INDEX learning_experiments_timeline_idx ON autonomous.learning_experiments (start_time DESC, end_time);
CREATE INDEX learning_experiments_performance_idx ON autonomous.learning_experiments (deployment_readiness, production_candidate);

-- ========================================
-- PERFORMANCE METRICS
-- ========================================
CREATE TABLE autonomous.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Metric identification
    metric_id TEXT NOT NULL,
    capability_id UUID REFERENCES autonomous.capability_registry(id),
    experiment_id UUID REFERENCES autonomous.learning_experiments(id),

    -- Metric definition
    metric_name TEXT NOT NULL,
    metric_type TEXT CHECK (metric_type IN (
        'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'latency',
        'throughput', 'resource_utilization', 'cost_efficiency', 'user_satisfaction',
        'business_impact', 'security_effectiveness', 'reliability'
    )),
    metric_category TEXT CHECK (metric_category IN ('model_performance', 'system_performance', 'business_metrics', 'operational_metrics')),

    -- Metric values and context
    metric_value DECIMAL(15,6),
    metric_unit TEXT,
    measurement_context JSONB,
    data_points JSONB,

    -- Temporal information
    measurement_timestamp TIMESTAMPTZ NOT NULL,
    measurement_period JSONB,
    aggregation_method TEXT,

    -- Benchmarking and comparison
    baseline_value DECIMAL(15,6),
    target_value DECIMAL(15,6),
    industry_benchmark DECIMAL(15,6),
    previous_value DECIMAL(15,6),

    -- Quality and confidence
    confidence_interval JSONB,
    statistical_significance DECIMAL(5,4),
    measurement_accuracy DECIMAL(3,2),
    data_quality_score DECIMAL(3,2),

    -- Trend analysis
    trend_direction TEXT CHECK (trend_direction IN ('improving', 'degrading', 'stable', 'volatile')),
    rate_of_change DECIMAL(8,4),
    seasonality_patterns JSONB,

    -- Environmental factors
    environmental_conditions JSONB,
    external_factors TEXT[],
    data_source_quality DECIMAL(3,2),

    -- Alert and notification
    threshold_warning DECIMAL(15,6),
    threshold_critical DECIMAL(15,6),
    alert_triggered BOOLEAN DEFAULT FALSE,
    notification_sent BOOLEAN DEFAULT FALSE,

    -- Analysis and insights
    anomaly_detected BOOLEAN DEFAULT FALSE,
    anomaly_score DECIMAL(5,4),
    insights JSONB,
    recommended_actions TEXT[],

    -- Metadata
    measurement_method TEXT,
    collection_tool TEXT,
    analyst_notes TEXT,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(metric_id, measurement_timestamp)
) PARTITION BY RANGE (created_at);

-- Create partitions for performance metrics
SELECT create_monthly_partition('autonomous.performance_metrics', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('autonomous.performance_metrics', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for performance metrics
CREATE INDEX performance_metrics_capability_idx ON autonomous.performance_metrics (capability_id, metric_type);
CREATE INDEX performance_metrics_experiment_idx ON autonomous.performance_metrics (experiment_id, metric_name);
CREATE INDEX performance_metrics_timestamp_idx ON autonomous.performance_metrics (measurement_timestamp DESC, metric_category);
CREATE INDEX performance_metrics_values_idx ON autonomous.performance_metrics (metric_value, target_value);
CREATE INDEX performance_metrics_trends_idx ON autonomous.performance_metrics (trend_direction, rate_of_change);
CREATE INDEX performance_metrics_anomaly_idx ON autonomous.performance_metrics (anomaly_detected, anomaly_score DESC);

-- ========================================
-- RESOURCE ALLOCATIONS
-- ========================================
CREATE TABLE autonomous.resource_allocations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Allocation identification
    allocation_id TEXT NOT NULL,
    allocation_name TEXT NOT NULL,
    allocation_type TEXT CHECK (allocation_type IN ('compute', 'storage', 'network', 'memory', 'gpu', 'specialized_hardware')),

    -- Resource details
    resource_specification JSONB NOT NULL,
    resource_capacity JSONB,
    resource_location TEXT,
    availability_zone TEXT,

    -- Allocation context
    purpose TEXT NOT NULL,
    requesting_capability UUID REFERENCES autonomous.capability_registry(id),
    requesting_experiment UUID REFERENCES autonomous.learning_experiments(id),
    priority INTEGER CHECK (priority BETWEEN 1 AND 5) DEFAULT 3,

    -- Allocation timeline
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    allocated_at TIMESTAMPTZ,
    planned_start TIMESTAMPTZ,
    planned_end TIMESTAMPTZ,
    actual_start TIMESTAMPTZ,
    actual_end TIMESTAMPTZ,

    -- Status and lifecycle
    allocation_status TEXT CHECK (allocation_status IN ('requested', 'approved', 'allocated', 'active', 'completed', 'cancelled', 'expired')),
    approval_required BOOLEAN DEFAULT TRUE,
    approved_by TEXT,
    approval_timestamp TIMESTAMPTZ,

    -- Usage tracking
    utilization_metrics JSONB,
    peak_usage JSONB,
    average_usage JSONB,
    efficiency_score DECIMAL(3,2),

    -- Cost management
    cost_estimate DECIMAL(12,2),
    actual_cost DECIMAL(12,2),
    cost_model TEXT,
    billing_method TEXT,

    -- Performance monitoring
    performance_metrics JSONB,
    sla_compliance DECIMAL(3,2),
    availability_metrics JSONB,
    reliability_metrics JSONB,

    -- Optimization opportunities
    optimization_suggestions TEXT[],
    rightsizing_recommendations JSONB,
    cost_optimization_potential DECIMAL(12,2),

    -- Dependencies and constraints
    resource_dependencies TEXT[],
    scheduling_constraints JSONB,
    conflict_resolution JSONB,

    -- Security and compliance
    security_requirements TEXT[],
    compliance_standards TEXT[],
    access_controls JSONB,
    audit_trail JSONB,

    -- Scalability and elasticity
    auto_scaling_enabled BOOLEAN DEFAULT FALSE,
    scaling_policies JSONB,
    scaling_events JSONB,

    -- Metadata
    allocated_by TEXT,
    resource_owner TEXT,
    business_justification TEXT,
    tags TEXT[],

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(allocation_id)
) PARTITION BY RANGE (created_at);

-- Create partitions for resource allocations
SELECT create_monthly_partition('autonomous.resource_allocations', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('autonomous.resource_allocations', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for resource allocations
CREATE INDEX resource_allocations_id_idx ON autonomous.resource_allocations (allocation_id, allocation_type);
CREATE INDEX resource_allocations_status_idx ON autonomous.resource_allocations (allocation_status, priority DESC);
CREATE INDEX resource_allocations_capability_idx ON autonomous.resource_allocations (requesting_capability, purpose);
CREATE INDEX resource_allocations_timeline_idx ON autonomous.resource_allocations (planned_start, planned_end);
CREATE INDEX resource_allocations_utilization_idx ON autonomous.resource_allocations (efficiency_score DESC, sla_compliance DESC);
CREATE INDEX resource_allocations_cost_idx ON autonomous.resource_allocations (actual_cost DESC, cost_estimate);

-- ========================================
-- KNOWLEDGE EVOLUTION
-- ========================================
CREATE TABLE autonomous.knowledge_evolution (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Knowledge identification
    knowledge_id TEXT NOT NULL,
    knowledge_type TEXT CHECK (knowledge_type IN (
        'learned_pattern', 'behavioral_model', 'threat_signature', 'optimization_rule',
        'prediction_model', 'classification_rule', 'anomaly_detector', 'correlation_rule'
    )),
    knowledge_domain TEXT CHECK (knowledge_domain IN ('threat_intelligence', 'behavioral_analysis', 'network_security', 'data_analysis', 'optimization')),

    -- Knowledge content
    knowledge_representation JSONB NOT NULL,
    knowledge_confidence DECIMAL(3,2),
    knowledge_complexity INTEGER CHECK (knowledge_complexity BETWEEN 1 AND 10),
    knowledge_generalizability DECIMAL(3,2),

    -- Learning context
    learning_source TEXT CHECK (learning_source IN ('supervised_learning', 'unsupervised_learning', 'reinforcement_learning', 'transfer_learning', 'human_feedback')),
    training_data_sources TEXT[],
    learning_algorithm TEXT,
    learning_parameters JSONB,

    -- Validation and verification
    validation_method TEXT,
    validation_results JSONB,
    cross_validation_score DECIMAL(3,2),
    independent_verification BOOLEAN DEFAULT FALSE,

    -- Evolution tracking
    parent_knowledge UUID REFERENCES autonomous.knowledge_evolution(id),
    evolution_type TEXT CHECK (evolution_type IN ('refinement', 'expansion', 'specialization', 'generalization', 'correction')),
    evolution_trigger TEXT,
    evolution_evidence JSONB,

    -- Performance metrics
    accuracy_score DECIMAL(3,2),
    precision_score DECIMAL(3,2),
    recall_score DECIMAL(3,2),
    f1_score DECIMAL(3,2),
    performance_trends JSONB,

    -- Application and usage
    deployment_environments TEXT[],
    usage_frequency INTEGER DEFAULT 0,
    success_rate DECIMAL(3,2),
    failure_modes JSONB,

    -- Feedback and improvement
    feedback_received JSONB,
    improvement_suggestions TEXT[],
    adaptation_history JSONB,
    continuous_learning_enabled BOOLEAN DEFAULT FALSE,

    -- Knowledge relationships
    related_knowledge UUID[],
    dependent_knowledge UUID[],
    conflicting_knowledge UUID[],
    knowledge_hierarchy LTREE,

    -- Temporal dynamics
    discovery_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_validation TIMESTAMPTZ,
    expiry_conditions JSONB,
    knowledge_lifecycle TEXT CHECK (knowledge_lifecycle IN ('emerging', 'validated', 'mature', 'deprecated', 'obsolete')),

    -- Quality assurance
    peer_review_status TEXT CHECK (peer_review_status IN ('pending', 'reviewed', 'approved', 'rejected')),
    quality_score DECIMAL(3,2),
    reliability_indicators JSONB,

    -- Documentation and provenance
    discovery_method TEXT,
    evidence_sources TEXT[],
    documentation JSONB,
    research_references TEXT[],

    -- Security and access
    knowledge_classification TEXT CHECK (knowledge_classification IN ('public', 'internal', 'confidential', 'top_secret')),
    access_restrictions JSONB,

    -- Metadata
    discovered_by TEXT,
    research_team TEXT[],
    tags TEXT[],

    -- Knowledge embedding for similarity analysis
    knowledge_embedding vector(768),

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(knowledge_id)
) PARTITION BY RANGE (created_at);

-- Create partitions for knowledge evolution
SELECT create_monthly_partition('autonomous.knowledge_evolution', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('autonomous.knowledge_evolution', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for knowledge evolution
CREATE INDEX knowledge_evolution_id_idx ON autonomous.knowledge_evolution (knowledge_id, knowledge_type);
CREATE INDEX knowledge_evolution_domain_idx ON autonomous.knowledge_evolution (knowledge_domain, knowledge_lifecycle);
CREATE INDEX knowledge_evolution_confidence_idx ON autonomous.knowledge_evolution (knowledge_confidence DESC, quality_score DESC);
CREATE INDEX knowledge_evolution_parent_idx ON autonomous.knowledge_evolution (parent_knowledge, evolution_type);
CREATE INDEX knowledge_evolution_performance_idx ON autonomous.knowledge_evolution (accuracy_score DESC, f1_score DESC);
CREATE INDEX knowledge_evolution_hierarchy_idx ON autonomous.knowledge_evolution USING gist(knowledge_hierarchy);
CREATE INDEX knowledge_evolution_embedding_idx ON autonomous.knowledge_evolution USING ivfflat (knowledge_embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX knowledge_evolution_relationships_idx ON autonomous.knowledge_evolution USING gin(related_knowledge);

-- ========================================
-- AUTONOMOUS AGENTS
-- ========================================
CREATE TABLE autonomous.autonomous_agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Agent identification
    agent_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    agent_type TEXT CHECK (agent_type IN (
        'data_collector', 'threat_hunter', 'analyst', 'responder',
        'optimizer', 'learner', 'coordinator', 'specialist'
    )),
    agent_role TEXT NOT NULL,

    -- Agent capabilities
    capabilities UUID[] REFERENCES autonomous.capability_registry(id),
    primary_functions TEXT[],
    specialized_skills JSONB,
    learning_abilities JSONB,

    -- Agent configuration
    configuration JSONB NOT NULL,
    behavioral_parameters JSONB,
    decision_thresholds JSONB,
    communication_protocols JSONB,

    -- Agent status and lifecycle
    agent_status TEXT CHECK (agent_status IN ('initializing', 'active', 'idle', 'learning', 'maintenance', 'error', 'shutdown')),
    lifecycle_stage TEXT CHECK (lifecycle_stage IN ('development', 'testing', 'deployment', 'production', 'retirement')),
    last_activity TIMESTAMPTZ,

    -- Performance and metrics
    performance_metrics JSONB,
    efficiency_scores JSONB,
    success_rate DECIMAL(3,2),
    error_rate DECIMAL(5,4),

    -- Learning and adaptation
    learning_enabled BOOLEAN DEFAULT TRUE,
    adaptation_rate DECIMAL(3,2),
    knowledge_base UUID[],
    experience_log JSONB,

    -- Collaboration and communication
    peer_agents UUID[],
    supervisor_agents UUID[],
    subordinate_agents UUID[],
    communication_patterns JSONB,

    -- Resource utilization
    allocated_resources UUID[],
    resource_efficiency DECIMAL(3,2),
    cost_effectiveness DECIMAL(8,2),

    -- Security and trust
    trust_level DECIMAL(3,2),
    security_clearance TEXT,
    access_permissions JSONB,
    audit_requirements TEXT[],

    -- Metadata
    created_by TEXT NOT NULL,
    owner_team TEXT[],
    deployment_environment TEXT,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(agent_id)
);

-- Indexes for autonomous agents
CREATE INDEX autonomous_agents_id_idx ON autonomous.autonomous_agents (agent_id, agent_type);
CREATE INDEX autonomous_agents_status_idx ON autonomous.autonomous_agents (agent_status, lifecycle_stage);
CREATE INDEX autonomous_agents_capabilities_idx ON autonomous.autonomous_agents USING gin(capabilities);
CREATE INDEX autonomous_agents_performance_idx ON autonomous.autonomous_agents (success_rate DESC, efficiency_scores);
CREATE INDEX autonomous_agents_relationships_idx ON autonomous.autonomous_agents USING gin(peer_agents);

-- Add audit triggers for autonomous operations tables
CREATE TRIGGER capability_registry_audit_trigger
    BEFORE INSERT OR UPDATE ON autonomous.capability_registry
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER learning_experiments_audit_trigger
    BEFORE INSERT OR UPDATE ON autonomous.learning_experiments
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER performance_metrics_audit_trigger
    BEFORE INSERT OR UPDATE ON autonomous.performance_metrics
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER resource_allocations_audit_trigger
    BEFORE INSERT OR UPDATE ON autonomous.resource_allocations
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER knowledge_evolution_audit_trigger
    BEFORE INSERT OR UPDATE ON autonomous.knowledge_evolution
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER autonomous_agents_audit_trigger
    BEFORE INSERT OR UPDATE ON autonomous.autonomous_agents
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();