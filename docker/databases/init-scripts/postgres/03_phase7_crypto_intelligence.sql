-- Phase 7: Cryptocurrency Intelligence Database Schema
-- Comprehensive cryptocurrency transaction and blockchain analysis

-- Create crypto intelligence schema
CREATE SCHEMA IF NOT EXISTS crypto_intel;

-- ========================================
-- WALLET TRANSACTIONS
-- ========================================
CREATE TABLE crypto_intel.wallet_transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Transaction identification
    transaction_hash TEXT NOT NULL,
    blockchain_network TEXT NOT NULL,
    block_number BIGINT,
    transaction_index INTEGER,

    -- Wallet information
    from_address TEXT NOT NULL,
    to_address TEXT NOT NULL,
    from_wallet_id UUID,
    to_wallet_id UUID,

    -- Transaction details
    amount DECIMAL(30,18) NOT NULL,
    currency_symbol TEXT NOT NULL,
    currency_contract_address TEXT,
    gas_used BIGINT,
    gas_price DECIMAL(30,18),
    transaction_fee DECIMAL(30,18),

    -- Transaction metadata
    transaction_type TEXT CHECK (transaction_type IN ('transfer', 'exchange', 'mixing', 'smart_contract', 'mining', 'staking', 'defi')),
    transaction_status TEXT CHECK (transaction_status IN ('pending', 'confirmed', 'failed', 'dropped')),
    confirmations INTEGER DEFAULT 0,

    -- Temporal data
    timestamp TIMESTAMPTZ NOT NULL,
    block_timestamp TIMESTAMPTZ,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Analysis fields
    is_suspicious BOOLEAN DEFAULT FALSE,
    risk_score INTEGER CHECK (risk_score BETWEEN 0 AND 100),
    risk_factors JSONB,
    mixing_score DECIMAL(3,2),
    chain_analysis_tags TEXT[],

    -- Related entities
    related_marketplace_vendor UUID,
    related_exchange TEXT,
    related_service_provider TEXT,

    -- Pattern analysis
    transaction_pattern TEXT,
    time_pattern_score DECIMAL(3,2),
    amount_pattern_score DECIMAL(3,2),
    frequency_pattern_score DECIMAL(3,2),

    -- Compliance flags
    aml_flags TEXT[],
    sanctions_list_match BOOLEAN DEFAULT FALSE,
    high_risk_jurisdiction BOOLEAN DEFAULT FALSE,
    regulatory_alerts JSONB,

    -- Metadata
    data_source data_source_type NOT NULL,
    chain_analysis_provider TEXT,
    collection_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    raw_transaction_data JSONB,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(transaction_hash, blockchain_network)
) PARTITION BY RANGE (created_at);

-- Create partitions for wallet transactions
SELECT create_monthly_partition('crypto_intel.wallet_transactions', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('crypto_intel.wallet_transactions', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for wallet transactions
CREATE INDEX wallet_transactions_hash_idx ON crypto_intel.wallet_transactions (transaction_hash, blockchain_network);
CREATE INDEX wallet_transactions_addresses_idx ON crypto_intel.wallet_transactions (from_address, to_address);
CREATE INDEX wallet_transactions_amount_idx ON crypto_intel.wallet_transactions (currency_symbol, amount DESC);
CREATE INDEX wallet_transactions_timestamp_idx ON crypto_intel.wallet_transactions (timestamp DESC, blockchain_network);
CREATE INDEX wallet_transactions_risk_idx ON crypto_intel.wallet_transactions (risk_score DESC, is_suspicious);
CREATE INDEX wallet_transactions_mixing_idx ON crypto_intel.wallet_transactions (mixing_score DESC, chain_analysis_tags);
CREATE INDEX wallet_transactions_compliance_idx ON crypto_intel.wallet_transactions (sanctions_list_match, aml_flags);
CREATE INDEX wallet_transactions_patterns_idx ON crypto_intel.wallet_transactions (transaction_pattern, transaction_type);

-- ========================================
-- BLOCKCHAIN FLOWS
-- ========================================
CREATE TABLE crypto_intel.blockchain_flows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Flow identification
    flow_id TEXT NOT NULL,
    blockchain_network TEXT NOT NULL,
    flow_type TEXT CHECK (flow_type IN ('direct', 'multi_hop', 'mixing', 'exchange', 'peeling_chain', 'consolidation')),

    -- Flow endpoints
    source_address TEXT NOT NULL,
    destination_address TEXT NOT NULL,
    intermediate_addresses TEXT[],

    -- Flow metrics
    total_amount DECIMAL(30,18) NOT NULL,
    currency_symbol TEXT NOT NULL,
    hop_count INTEGER DEFAULT 1,
    flow_duration INTERVAL,

    -- Flow analysis
    flow_confidence DECIMAL(3,2),
    obfuscation_score DECIMAL(3,2),
    complexity_score INTEGER,
    suspicious_indicators TEXT[],

    -- Temporal tracking
    flow_start_time TIMESTAMPTZ NOT NULL,
    flow_end_time TIMESTAMPTZ,
    last_activity TIMESTAMPTZ,

    -- Related transactions
    transaction_hashes TEXT[],
    related_flows UUID[],

    -- Risk assessment
    risk_level severity_level DEFAULT 'low',
    risk_explanation TEXT,
    compliance_concerns JSONB,

    -- Pattern classification
    known_pattern_match TEXT,
    pattern_confidence DECIMAL(3,2),
    behavioral_classification TEXT[],

    -- Intelligence integration
    related_marketplace_activity BOOLEAN DEFAULT FALSE,
    related_investigations UUID[],
    law_enforcement_interest BOOLEAN DEFAULT FALSE,

    -- Metadata
    detection_algorithm TEXT,
    analysis_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    analyst_notes TEXT,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(flow_id, blockchain_network)
) PARTITION BY RANGE (created_at);

-- Create partitions for blockchain flows
SELECT create_monthly_partition('crypto_intel.blockchain_flows', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('crypto_intel.blockchain_flows', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for blockchain flows
CREATE INDEX blockchain_flows_addresses_idx ON crypto_intel.blockchain_flows (source_address, destination_address);
CREATE INDEX blockchain_flows_amount_idx ON crypto_intel.blockchain_flows (currency_symbol, total_amount DESC);
CREATE INDEX blockchain_flows_risk_idx ON crypto_intel.blockchain_flows (risk_level, obfuscation_score DESC);
CREATE INDEX blockchain_flows_pattern_idx ON crypto_intel.blockchain_flows (known_pattern_match, pattern_confidence DESC);
CREATE INDEX blockchain_flows_temporal_idx ON crypto_intel.blockchain_flows (flow_start_time DESC, flow_duration);
CREATE INDEX blockchain_flows_investigation_idx ON crypto_intel.blockchain_flows (law_enforcement_interest, related_investigations);

-- ========================================
-- MIXING PATTERNS
-- ========================================
CREATE TABLE crypto_intel.mixing_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Pattern identification
    pattern_id TEXT NOT NULL,
    mixing_service TEXT,
    blockchain_network TEXT NOT NULL,

    -- Pattern characteristics
    pattern_type TEXT CHECK (pattern_type IN ('coin_join', 'tumbler', 'chain_hopping', 'peel_chain', 'consolidation', 'custom')),
    input_addresses TEXT[] NOT NULL,
    output_addresses TEXT[] NOT NULL,
    mixing_rounds INTEGER DEFAULT 1,

    -- Financial metrics
    total_input_amount DECIMAL(30,18) NOT NULL,
    total_output_amount DECIMAL(30,18) NOT NULL,
    mixing_fee DECIMAL(30,18),
    fee_percentage DECIMAL(5,2),

    -- Mixing analysis
    anonymity_score DECIMAL(3,2),
    traceability_score DECIMAL(3,2),
    effectiveness_rating INTEGER CHECK (effectiveness_rating BETWEEN 1 AND 5),
    breaking_points JSONB,

    -- Temporal patterns
    mixing_start_time TIMESTAMPTZ NOT NULL,
    mixing_end_time TIMESTAMPTZ,
    mixing_duration INTERVAL,
    delay_patterns JSONB,

    -- Service analysis
    service_reputation DECIMAL(3,2),
    service_reliability DECIMAL(3,2),
    known_vulnerabilities TEXT[],
    law_enforcement_compromised BOOLEAN DEFAULT FALSE,

    -- Transaction tracking
    related_transactions UUID[],
    input_transaction_hashes TEXT[],
    output_transaction_hashes TEXT[],

    -- Risk factors
    high_risk_sources BOOLEAN DEFAULT FALSE,
    sanctions_involvement BOOLEAN DEFAULT FALSE,
    criminal_proceeds_suspected BOOLEAN DEFAULT FALSE,
    investigation_alerts JSONB,

    -- Detection metadata
    detection_method TEXT,
    detection_confidence DECIMAL(3,2),
    false_positive_probability DECIMAL(3,2),

    -- Metadata
    analyst_id TEXT,
    analysis_tools_used TEXT[],
    verification_status TEXT CHECK (verification_status IN ('pending', 'verified', 'disputed', 'false_positive')),

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(pattern_id, blockchain_network)
) PARTITION BY RANGE (created_at);

-- Create partitions for mixing patterns
SELECT create_monthly_partition('crypto_intel.mixing_patterns', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('crypto_intel.mixing_patterns', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for mixing patterns
CREATE INDEX mixing_patterns_service_idx ON crypto_intel.mixing_patterns (mixing_service, blockchain_network);
CREATE INDEX mixing_patterns_type_idx ON crypto_intel.mixing_patterns (pattern_type, effectiveness_rating DESC);
CREATE INDEX mixing_patterns_amount_idx ON crypto_intel.mixing_patterns (total_input_amount DESC, mixing_start_time);
CREATE INDEX mixing_patterns_anonymity_idx ON crypto_intel.mixing_patterns (anonymity_score DESC, traceability_score);
CREATE INDEX mixing_patterns_risk_idx ON crypto_intel.mixing_patterns (high_risk_sources, sanctions_involvement);
CREATE INDEX mixing_patterns_detection_idx ON crypto_intel.mixing_patterns (detection_confidence DESC, verification_status);

-- ========================================
-- EXCHANGE MOVEMENTS
-- ========================================
CREATE TABLE crypto_intel.exchange_movements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Movement identification
    movement_id TEXT NOT NULL,
    exchange_name TEXT NOT NULL,
    movement_type TEXT CHECK (movement_type IN ('deposit', 'withdrawal', 'internal_transfer', 'trade_settlement')),

    -- Address information
    wallet_address TEXT NOT NULL,
    exchange_hot_wallet TEXT,
    exchange_cold_wallet TEXT,
    user_account_id TEXT,

    -- Transaction details
    amount DECIMAL(30,18) NOT NULL,
    currency_symbol TEXT NOT NULL,
    transaction_hash TEXT,
    blockchain_network TEXT,

    -- Exchange context
    exchange_tier TEXT CHECK (exchange_tier IN ('tier1', 'tier2', 'tier3', 'unregulated', 'p2p')),
    kyc_level TEXT CHECK (kyc_level IN ('none', 'basic', 'enhanced', 'institutional')),
    jurisdiction TEXT,
    regulatory_status TEXT,

    -- Temporal data
    movement_timestamp TIMESTAMPTZ NOT NULL,
    confirmation_timestamp TIMESTAMPTZ,
    settlement_timestamp TIMESTAMPTZ,

    -- Risk assessment
    risk_score INTEGER CHECK (risk_score BETWEEN 0 AND 100),
    aml_alerts TEXT[],
    suspicious_activity_flags TEXT[],
    velocity_check_results JSONB,

    -- Pattern analysis
    user_behavior_score DECIMAL(3,2),
    typical_amount_deviation DECIMAL(5,2),
    time_pattern_anomaly BOOLEAN DEFAULT FALSE,
    geographic_anomaly BOOLEAN DEFAULT FALSE,

    -- Compliance tracking
    ctr_reportable BOOLEAN DEFAULT FALSE,
    sar_filed BOOLEAN DEFAULT FALSE,
    regulatory_reports JSONB,
    investigation_flags TEXT[],

    -- Related intelligence
    related_marketplace_vendor UUID,
    related_investigations UUID[],
    cross_exchange_patterns JSONB,

    -- Metadata
    data_source data_source_type NOT NULL,
    collection_method TEXT,
    data_confidence confidence_level DEFAULT 'medium',

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(movement_id, exchange_name)
) PARTITION BY RANGE (created_at);

-- Create partitions for exchange movements
SELECT create_monthly_partition('crypto_intel.exchange_movements', DATE_TRUNC('month', NOW()));
SELECT create_monthly_partition('crypto_intel.exchange_movements', DATE_TRUNC('month', NOW() + INTERVAL '1 month'));

-- Indexes for exchange movements
CREATE INDEX exchange_movements_exchange_idx ON crypto_intel.exchange_movements (exchange_name, movement_type);
CREATE INDEX exchange_movements_address_idx ON crypto_intel.exchange_movements (wallet_address, currency_symbol);
CREATE INDEX exchange_movements_amount_idx ON crypto_intel.exchange_movements (currency_symbol, amount DESC);
CREATE INDEX exchange_movements_risk_idx ON crypto_intel.exchange_movements (risk_score DESC, aml_alerts);
CREATE INDEX exchange_movements_compliance_idx ON crypto_intel.exchange_movements (ctr_reportable, sar_filed);
CREATE INDEX exchange_movements_pattern_idx ON crypto_intel.exchange_movements (user_behavior_score, time_pattern_anomaly);
CREATE INDEX exchange_movements_investigation_idx ON crypto_intel.exchange_movements (related_investigations, investigation_flags);

-- ========================================
-- WALLET CLUSTERS
-- ========================================
CREATE TABLE crypto_intel.wallet_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    -- Cluster identification
    cluster_id TEXT NOT NULL,
    blockchain_network TEXT NOT NULL,
    cluster_name TEXT,

    -- Cluster composition
    wallet_addresses TEXT[] NOT NULL,
    primary_address TEXT,
    cluster_size INTEGER NOT NULL,

    -- Cluster characteristics
    cluster_type TEXT CHECK (cluster_type IN ('exchange', 'mixer', 'darkmarket', 'mining_pool', 'institutional', 'personal', 'unknown')),
    entity_identification TEXT,
    confidence_score DECIMAL(3,2),

    -- Financial metrics
    total_balance DECIMAL(30,18),
    total_received DECIMAL(30,18),
    total_sent DECIMAL(30,18),
    transaction_count BIGINT DEFAULT 0,

    -- Activity patterns
    first_activity TIMESTAMPTZ,
    last_activity TIMESTAMPTZ,
    activity_frequency JSONB,
    peak_activity_periods JSONB,

    -- Risk assessment
    risk_level severity_level DEFAULT 'low',
    threat_indicators TEXT[],
    sanctions_exposure BOOLEAN DEFAULT FALSE,
    criminal_association_score DECIMAL(3,2),

    -- Clustering algorithm metadata
    clustering_algorithm TEXT,
    clustering_confidence DECIMAL(3,2),
    manual_verification BOOLEAN DEFAULT FALSE,
    verification_notes TEXT,

    -- Related intelligence
    related_marketplace_vendors UUID[],
    related_investigations UUID[],
    cross_chain_clusters JSONB,

    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(cluster_id, blockchain_network)
);

-- Indexes for wallet clusters
CREATE INDEX wallet_clusters_network_idx ON crypto_intel.wallet_clusters (blockchain_network, cluster_type);
CREATE INDEX wallet_clusters_entity_idx ON crypto_intel.wallet_clusters (entity_identification, confidence_score DESC);
CREATE INDEX wallet_clusters_risk_idx ON crypto_intel.wallet_clusters (risk_level, criminal_association_score DESC);
CREATE INDEX wallet_clusters_activity_idx ON crypto_intel.wallet_clusters (last_activity DESC, transaction_count DESC);
CREATE INDEX wallet_clusters_addresses_idx ON crypto_intel.wallet_clusters USING gin(wallet_addresses);

-- Add audit triggers for crypto intelligence tables
CREATE TRIGGER wallet_transactions_audit_trigger
    BEFORE INSERT OR UPDATE ON crypto_intel.wallet_transactions
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER blockchain_flows_audit_trigger
    BEFORE INSERT OR UPDATE ON crypto_intel.blockchain_flows
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER mixing_patterns_audit_trigger
    BEFORE INSERT OR UPDATE ON crypto_intel.mixing_patterns
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER exchange_movements_audit_trigger
    BEFORE INSERT OR UPDATE ON crypto_intel.exchange_movements
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

CREATE TRIGGER wallet_clusters_audit_trigger
    BEFORE INSERT OR UPDATE ON crypto_intel.wallet_clusters
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();